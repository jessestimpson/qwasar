/* Gated DeltaNet: causal conv, gates, and the delta-rule recurrence.
 *
 * These are the kernels most likely to be subtly wrong, because unlike a matvec
 * they carry state across calls and across timesteps.  Two properties matter
 * beyond matching the CPU twin:
 *
 *   - the state left behind must be right, not just the output;
 *   - running N timesteps in one call must equal running them one at a time,
 *     since prefill and decode take exactly those two paths through the same
 *     weights and must agree.
 *
 * Real A_log, dt_bias and conv taps are used throughout: the decay gate
 * exp(-exp(A_log)*softplus(...)) is extremely sensitive to A_log's scale, and
 * synthetic values would not reproduce the regime the model actually runs in. */

#include "qwasar.h"
#include "qwasar_gpu.h"
#include "qwasar_model.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;

#define CHECK(cond, ...) do { \
    if (!(cond)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
                   fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); fails++; } \
} while (0)

static void fill_random(float *v, size_t n, uint32_t seed) {
    uint32_t s = seed | 1u;
    for (size_t i = 0; i < n; i++) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        v[i] = (float)((int32_t)(s >> 8) - 8388608) / 8388608.0f;
    }
}

/* Compared against the magnitude of the reference vector rather than
 * elementwise, so near-zero entries do not dominate the statistic. */
static double rel_l2(const float *a, const float *b, size_t n) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = (double)a[i] - (double)b[i];
        num += d * d;
        den += (double)b[i] * (double)b[i];
    }
    return den > 0.0 ? sqrt(num / den) : sqrt(num);
}

static void test_conv(const qw_layer *L, const qw_shape *sh, int32_t ksize, int32_t rows) {
    const int32_t ch = sh->conv_dim;
    size_t xn = (size_t)rows * ch, sn = (size_t)(ksize - 1) * ch;

    qw_buf xb = qw_buf_alloc(xn * sizeof(float));
    qw_buf yb = qw_buf_alloc(xn * sizeof(float));
    qw_buf sb = qw_buf_alloc(sn * sizeof(float));
    float *x = qw_buf_contents(xb), *y = qw_buf_contents(yb), *st = qw_buf_contents(sb);
    fill_random(x, xn, 0x2468ACEu);
    fill_random(st, sn, 0x13579BDu);

    float *st_ref = malloc(sn * sizeof(float));
    float *y_ref  = malloc(xn * sizeof(float));
    memcpy(st_ref, st, sn * sizeof(float));

    qw_cmd c = qw_cmd_begin();
    qw_op_conv1d_causal_silu(c, qw_ref_at(yb, 0), qw_ref_at(xb, 0), qw_ref_at(sb, 0),
                             qw_tensor_ref(L->conv1d), ch, rows, ksize);
    qw_cmd_wait(c);
    CHECK(qw_cmd_error(c) == NULL, "conv: %s", qw_cmd_error(c));
    qw_cmd_free(c);

    qw_cpu_conv1d_causal_silu(y_ref, x, st_ref, qw_tensor_data(L->conv1d), ch, rows, ksize);

    double ey = rel_l2(y, y_ref, xn);
    double es = rel_l2(st, st_ref, sn);
    CHECK(ey < 1e-6, "conv rows=%d: output rel l2 %.3g", rows, ey);
    CHECK(es < 1e-6, "conv rows=%d: state rel l2 %.3g", rows, es);
    printf("  conv1d causal+silu   ch=%d rows=%-3d  out %.2e  state %.2e\n",
           ch, rows, ey, es);

    free(st_ref); free(y_ref);
    qw_buf_free(xb); qw_buf_free(yb); qw_buf_free(sb);
}

static void test_gates(const qw_layer *L, const qw_config *cfg, int32_t rows) {
    const int32_t hv = cfg->linear_num_value_heads;
    size_t n = (size_t)rows * hv;

    qw_buf ab = qw_buf_alloc(n * sizeof(float)), bb = qw_buf_alloc(n * sizeof(float));
    qw_buf gb = qw_buf_alloc(n * sizeof(float)), tb = qw_buf_alloc(n * sizeof(float));
    float *a = qw_buf_contents(ab), *b = qw_buf_contents(bb);
    float *g = qw_buf_contents(gb), *beta = qw_buf_contents(tb);
    /* Scale up so softplus sees both its linear and saturating regimes. */
    fill_random(a, n, 0xC0FFEEu);
    fill_random(b, n, 0xDECAFu);
    for (size_t i = 0; i < n; i++) a[i] *= 8.0f;

    qw_cmd c = qw_cmd_begin();
    qw_op_gdn_gates(c, qw_ref_at(gb, 0), qw_ref_at(tb, 0), qw_ref_at(ab, 0),
                    qw_ref_at(bb, 0), qw_tensor_ref(L->A_log), qw_tensor_ref(L->dt_bias),
                    hv, rows);
    qw_cmd_wait(c);
    CHECK(qw_cmd_error(c) == NULL, "gates: %s", qw_cmd_error(c));
    qw_cmd_free(c);

    float *g_ref = malloc(n * sizeof(float)), *beta_ref = malloc(n * sizeof(float));
    qw_cpu_gdn_gates(g_ref, beta_ref, a, b, qw_tensor_data(L->A_log),
                     qw_tensor_data(L->dt_bias), hv, rows);

    double eg = rel_l2(g, g_ref, n), eb = rel_l2(beta, beta_ref, n);
    CHECK(eg < 1e-6, "gates rows=%d: g rel l2 %.3g", rows, eg);
    CHECK(eb < 1e-6, "gates rows=%d: beta rel l2 %.3g", rows, eb);

    /* A decay outside (0,1] would make the recurrence blow up or freeze. */
    for (size_t i = 0; i < n; i++)
        CHECK(g_ref[i] > 0.0f && g_ref[i] <= 1.0f, "g[%zu] = %g out of (0,1]", i, (double)g_ref[i]);

    printf("  gdn gates            hv=%d rows=%-3d  g %.2e  beta %.2e\n", hv, rows, eg, eb);
    free(g_ref); free(beta_ref);
    qw_buf_free(ab); qw_buf_free(bb); qw_buf_free(gb); qw_buf_free(tb);
}

/* Runs the recurrence on the GPU and returns the output and final state. */
static void run_gdn(const qw_config *cfg, int32_t rows, const float *q, const float *k,
                    const float *v, const float *g, const float *beta,
                    const float *state_in, float *y_out, float *state_out) {
    const int32_t hk = cfg->linear_num_key_heads, hv = cfg->linear_num_value_heads;
    const int32_t dk = cfg->linear_key_head_dim, dv = cfg->linear_value_head_dim;
    size_t qn = (size_t)rows * hk * dk, vn = (size_t)rows * hv * dv;
    size_t gn = (size_t)rows * hv, sn = (size_t)hv * dv * dk;

    qw_buf qb = qw_buf_alloc(qn * 4), kb = qw_buf_alloc(qn * 4), vb = qw_buf_alloc(vn * 4);
    qw_buf gb = qw_buf_alloc(gn * 4), bb = qw_buf_alloc(gn * 4);
    qw_buf sb = qw_buf_alloc(sn * 4), yb = qw_buf_alloc(vn * 4);

    memcpy(qw_buf_contents(qb), q, qn * 4);
    memcpy(qw_buf_contents(kb), k, qn * 4);
    memcpy(qw_buf_contents(vb), v, vn * 4);
    memcpy(qw_buf_contents(gb), g, gn * 4);
    memcpy(qw_buf_contents(bb), beta, gn * 4);
    memcpy(qw_buf_contents(sb), state_in, sn * 4);

    qw_cmd c = qw_cmd_begin();
    qw_op_gated_delta(c, qw_ref_at(yb, 0), qw_ref_at(qb, 0), qw_ref_at(kb, 0),
                      qw_ref_at(vb, 0), qw_ref_at(gb, 0), qw_ref_at(bb, 0),
                      qw_ref_at(sb, 0), hk, hv, dk, dv, rows);
    qw_cmd_wait(c);
    CHECK(qw_cmd_error(c) == NULL, "gdn: %s", qw_cmd_error(c));
    qw_cmd_free(c);

    memcpy(y_out, qw_buf_contents(yb), vn * 4);
    memcpy(state_out, qw_buf_contents(sb), sn * 4);

    qw_buf_free(qb); qw_buf_free(kb); qw_buf_free(vb);
    qw_buf_free(gb); qw_buf_free(bb); qw_buf_free(sb); qw_buf_free(yb);
}

static void test_recurrence(const qw_config *cfg, int32_t rows) {
    const int32_t hk = cfg->linear_num_key_heads, hv = cfg->linear_num_value_heads;
    const int32_t dk = cfg->linear_key_head_dim, dv = cfg->linear_value_head_dim;
    size_t qn = (size_t)rows * hk * dk, vn = (size_t)rows * hv * dv;
    size_t gn = (size_t)rows * hv, sn = (size_t)hv * dv * dk;

    float *q = malloc(qn * 4), *k = malloc(qn * 4), *v = malloc(vn * 4);
    float *g = malloc(gn * 4), *beta = malloc(gn * 4), *s0 = malloc(sn * 4);
    fill_random(q, qn, 0x11111u); fill_random(k, qn, 0x22222u);
    fill_random(v, vn, 0x33333u); fill_random(s0, sn, 0x44444u);

    /* q and k arrive l2-normalised in the real graph; normalise here so the
     * magnitudes the kernel sees match production. */
    for (int32_t t = 0; t < rows; t++)
        for (int32_t h = 0; h < hk; h++) {
            float *qh = q + ((size_t)t * hk + h) * dk, *kh = k + ((size_t)t * hk + h) * dk;
            double nq = 0, nk = 0;
            for (int32_t i = 0; i < dk; i++) { nq += qh[i]*qh[i]; nk += kh[i]*kh[i]; }
            nq = 1.0 / (sqrt(nq) * sqrt((double)dk)); nk = 1.0 / sqrt(nk);
            for (int32_t i = 0; i < dk; i++) { qh[i] *= (float)nq; kh[i] *= (float)nk; }
        }
    /* Plausible gate values: decays near 1, write strengths near 0.5. */
    for (size_t i = 0; i < gn; i++) { g[i] = 0.90f + 0.09f * fabsf(g[i]); beta[i] = 0.5f; }
    for (size_t i = 0; i < gn; i++) beta[i] = 0.3f + 0.4f * (float)(i % 7) / 7.0f;
    for (size_t i = 0; i < sn; i++) s0[i] *= 0.05f;

    float *y_gpu = malloc(vn * 4), *s_gpu = malloc(sn * 4);
    run_gdn(cfg, rows, q, k, v, g, beta, s0, y_gpu, s_gpu);

    float *y_ref = malloc(vn * 4), *s_ref = malloc(sn * 4);
    memcpy(s_ref, s0, sn * 4);
    qw_cpu_gated_delta(y_ref, q, k, v, g, beta, s_ref, hk, hv, dk, dv, rows);

    double ey = rel_l2(y_gpu, y_ref, vn), es = rel_l2(s_gpu, s_ref, sn);
    CHECK(ey < 5e-6, "gdn rows=%d: output rel l2 %.3g", rows, ey);
    CHECK(es < 5e-6, "gdn rows=%d: state rel l2 %.3g", rows, es);
    printf("  gated delta          hv=%d rows=%-3d  out %.2e  state %.2e\n",
           hv, rows, ey, es);

    /* Streaming equivalence: feeding the same sequence one timestep at a time
     * must land on the same state and outputs as one batched call.  This is
     * exactly the prefill-versus-decode boundary, so a mismatch here would
     * show up as a model that answers differently depending on how the prompt
     * was chunked. */
    if (rows > 1) {
        float *s_step = malloc(sn * 4), *y_step = malloc(vn * 4);
        float *y_one = malloc((size_t)hv * dv * 4), *s_tmp = malloc(sn * 4);
        memcpy(s_step, s0, sn * 4);
        for (int32_t t = 0; t < rows; t++) {
            run_gdn(cfg, 1, q + (size_t)t * hk * dk, k + (size_t)t * hk * dk,
                    v + (size_t)t * hv * dv, g + (size_t)t * hv, beta + (size_t)t * hv,
                    s_step, y_one, s_tmp);
            memcpy(s_step, s_tmp, sn * 4);
            memcpy(y_step + (size_t)t * hv * dv, y_one, (size_t)hv * dv * 4);
        }
        double e1 = rel_l2(y_step, y_gpu, vn), e2 = rel_l2(s_step, s_gpu, sn);
        CHECK(e1 < 5e-6, "streaming vs batched output rel l2 %.3g", e1);
        CHECK(e2 < 5e-6, "streaming vs batched state rel l2 %.3g", e2);
        printf("  %-20s rows=%-3d %*s out %.2e  state %.2e\n",
               "streaming == batched", rows, 6, "", e1, e2);
        free(s_step); free(y_step); free(y_one); free(s_tmp);
    }

    free(q); free(k); free(v); free(g); free(beta); free(s0);
    free(y_gpu); free(s_gpu); free(y_ref); free(s_ref);
}

int main(int argc, char **argv) {
    const char *model = getenv("QWASAR_TEST_MODEL");
    if (argc > 1) model = argv[1];
    if (!model) { fprintf(stderr, "skip: set QWASAR_TEST_MODEL\n"); return 0; }

    char err[512];
    qwasar_options opts = { .model_path = model };
    qwasar_engine *e = qwasar_engine_load(&opts, err, sizeof err);
    if (!e) { fprintf(stderr, "load failed: %s\n", err); return 1; }

    const qw_config *cfg = qwasar_engine_config(e);
    const qw_shape  *sh  = qwasar_engine_shape(e);
    const qw_layer  *l0  = qwasar_engine_layer(e, 0);

    test_conv(l0, sh, cfg->linear_conv_kernel_dim, 1);
    test_conv(l0, sh, cfg->linear_conv_kernel_dim, 5);
    test_gates(l0, cfg, 1);
    test_gates(l0, cfg, 9);
    test_recurrence(cfg, 1);
    test_recurrence(cfg, 6);

    qwasar_engine_free(e);
    qw_gpu_shutdown();
    if (fails) { fprintf(stderr, "%d check(s) failed\n", fails); return 1; }
    printf("ok: gdn\n");
    return 0;
}
