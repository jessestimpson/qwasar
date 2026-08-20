/* Elementwise and reduction kernels against their CPU twins.
 *
 * These use real norm weights from the model where one exists, because the
 * weight convention (this checkpoint already carries the +1.0 offset) is
 * exactly the kind of thing a synthetic test would not notice. */

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

/* Relative L2 over the whole array.  RoPE mixes x0*c - x1*s, which cancels to
 * near zero for some elements; an elementwise relative error there measures the
 * cancellation, not the kernel.  Norm-relative is the honest metric. */
static double rel_l2(const float *a, const float *b, size_t n) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = (double)a[i] - (double)b[i];
        num += d * d;
        den += (double)b[i] * (double)b[i];
    }
    return den > 0.0 ? sqrt(num / den) : sqrt(num);
}

static double max_abs(const float *a, const float *b, size_t n, size_t *where) {
    double worst = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = fabs((double)a[i] - (double)b[i]);
        if (d > worst) { worst = d; if (where) *where = i; }
    }
    return worst;
}

static double max_rel(const float *a, const float *b, size_t n, size_t *where) {
    double worst = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = fabs((double)a[i] - (double)b[i]);
        double s = fabs((double)b[i]);
        double e = s > 1e-6 ? d / s : d;
        if (e > worst) { worst = e; if (where) *where = i; }
    }
    return worst;
}

static void test_rms_norm(const char *label, const qw_tensor *wt,
                          int32_t dim, int32_t rows, float eps, float out_scale) {
    size_t n = (size_t)rows * dim;
    qw_buf xb = qw_buf_alloc(n * sizeof(float));
    qw_buf yb = qw_buf_alloc(n * sizeof(float));
    float *x = qw_buf_contents(xb), *y = qw_buf_contents(yb);
    fill_random(x, n, 0x1234567u);
    memset(y, 0xCD, n * sizeof(float));

    qw_ref wref = wt ? qw_tensor_ref(wt) : qw_ref_at(NULL, 0);
    qw_cmd c = qw_cmd_begin();
    qw_op_rms_norm(c, qw_ref_at(yb, 0), qw_ref_at(xb, 0), wref, dim, rows, eps, out_scale);
    qw_cmd_wait(c);
    CHECK(qw_cmd_error(c) == NULL, "%s: %s", label, qw_cmd_error(c));
    qw_cmd_free(c);

    float *ref = malloc(n * sizeof(float));
    qw_cpu_rms_norm(ref, x, wt ? qw_tensor_data(wt) : NULL, dim, rows, eps, out_scale);

    size_t at = 0;
    double worst = max_rel(y, ref, n, &at);
    CHECK(worst < 1e-5, "%s: worst rel err %.3g at %zu (%g vs %g)",
          label, worst, at, (double)y[at], (double)ref[at]);
    printf("  %-30s dim=%-5d rows=%-3d  worst rel err %.2e\n", label, dim, rows, worst);

    free(ref);
    qw_buf_free(xb);
    qw_buf_free(yb);
}

static void test_rms_norm_gated(const char *label, const qw_tensor *wt,
                                int32_t dim, int32_t rows, float eps) {
    size_t n = (size_t)rows * dim;
    qw_buf xb = qw_buf_alloc(n * sizeof(float));
    qw_buf gb = qw_buf_alloc(n * sizeof(float));
    qw_buf yb = qw_buf_alloc(n * sizeof(float));
    float *x = qw_buf_contents(xb), *g = qw_buf_contents(gb), *y = qw_buf_contents(yb);
    fill_random(x, n, 0xABCDEFu);
    fill_random(g, n, 0x55AA55u);
    memset(y, 0xCD, n * sizeof(float));

    qw_cmd c = qw_cmd_begin();
    qw_op_rms_norm_gated(c, qw_ref_at(yb, 0), qw_ref_at(xb, 0), qw_tensor_ref(wt),
                         qw_ref_at(gb, 0), dim, rows, eps, 1.0f);
    qw_cmd_wait(c);
    CHECK(qw_cmd_error(c) == NULL, "%s: %s", label, qw_cmd_error(c));
    qw_cmd_free(c);

    float *ref = malloc(n * sizeof(float));
    qw_cpu_rms_norm_gated(ref, x, qw_tensor_data(wt), g, dim, rows, eps, 1.0f);

    size_t at = 0;
    double worst = max_rel(y, ref, n, &at);
    CHECK(worst < 1e-5, "%s: worst rel err %.3g at %zu", label, worst, at);
    printf("  %-30s dim=%-5d rows=%-3d  worst rel err %.2e\n", label, dim, rows, worst);

    free(ref);
    qw_buf_free(xb); qw_buf_free(gb); qw_buf_free(yb);
}

static void test_swiglu(int32_t n) {
    qw_buf ab = qw_buf_alloc((size_t)n * sizeof(float));
    qw_buf bb = qw_buf_alloc((size_t)n * sizeof(float));
    qw_buf yb = qw_buf_alloc((size_t)n * sizeof(float));
    float *a = qw_buf_contents(ab), *b = qw_buf_contents(bb), *y = qw_buf_contents(yb);
    /* Spread over several orders of magnitude so silu's saturating tails are
     * exercised, not just the near-linear region around zero. */
    fill_random(a, (size_t)n, 0xF00Du);
    fill_random(b, (size_t)n, 0xBEEFu);
    for (int32_t i = 0; i < n; i++) a[i] *= (float)(1 << (i % 5));

    qw_cmd c = qw_cmd_begin();
    qw_op_swiglu(c, qw_ref_at(yb, 0), qw_ref_at(ab, 0), qw_ref_at(bb, 0), n);
    qw_cmd_wait(c);
    CHECK(qw_cmd_error(c) == NULL, "swiglu: %s", qw_cmd_error(c));
    qw_cmd_free(c);

    float *ref = malloc((size_t)n * sizeof(float));
    qw_cpu_swiglu(ref, a, b, n);
    size_t at = 0;
    double worst = max_rel(y, ref, (size_t)n, &at);
    CHECK(worst < 1e-5, "swiglu: worst rel err %.3g at %zu", worst, at);
    printf("  %-30s n=%-7d          worst rel err %.2e\n", "swiglu", n, worst);
    free(ref);
    qw_buf_free(ab); qw_buf_free(bb); qw_buf_free(yb);
}

/* The interleaved-MRoPE axis assignment, pinned against the reference
 * implementation's own selector for mrope_section [11,11,10] at rotary_dim 64.
 * Getting this wrong is invisible for text (all three axes agree) and only
 * corrupts image positions, so it is checked directly rather than end to end. */
static void test_rope_tables(const qw_config *cfg) {
    const int32_t half = cfg->rotary_dim / 2;
    uint8_t axis[64];
    float inv[64];
    CHECK(half <= 64, "rotary_dim %d larger than this test's scratch", cfg->rotary_dim);

    qw_rope_tables(axis, inv, cfg->rotary_dim, cfg->rope_theta, cfg->mrope_section);

    static const uint8_t expect[32] = {
        0,1,2, 0,1,2, 0,1,2, 0,1,2, 0,1,2, 0,1,2,
        0,1,2, 0,1,2, 0,1,2, 0,1,2, 0,1
    };
    CHECK(half == 32, "expected 32 frequencies, got %d", half);
    for (int32_t j = 0; j < half && j < 32; j++)
        CHECK(axis[j] == expect[j], "axis[%d] = %u, expected %u", j, axis[j], expect[j]);

    int counts[3] = { 0, 0, 0 };
    for (int32_t j = 0; j < half; j++) counts[axis[j]]++;
    for (int i = 0; i < 3; i++)
        CHECK(counts[i] == cfg->mrope_section[i],
              "axis %d used %d times, mrope_section says %d",
              i, counts[i], cfg->mrope_section[i]);

    /* Reference values from mlx_vlm.models.rope_utils.compute_inv_freq(64, 1e7). */
    CHECK(fabsf(inv[0] - 1.0f) < 1e-9f, "inv_freq[0] = %g", (double)inv[0]);
    CHECK(fabsf(inv[1] - 0.6042963862f) < 1e-7f, "inv_freq[1] = %.10g", (double)inv[1]);
    CHECK(fabsf(inv[2] - 0.3651741147f) < 1e-7f, "inv_freq[2] = %.10g", (double)inv[2]);
    CHECK(fabsf(inv[31] - 1.654816941254e-07f) < 1e-13f, "inv_freq[31] = %.12g", (double)inv[31]);

    printf("  %-30s rotary=%d of %d, sections [%d,%d,%d]\n", "rope tables",
           cfg->rotary_dim, cfg->head_dim,
           counts[0], counts[1], counts[2]);
}

static void test_rope(const qw_config *cfg, int32_t rows, int32_t heads, bool multimodal,
                      int32_t base_pos) {
    const int32_t hd = cfg->head_dim, rd = cfg->rotary_dim, half = rd / 2;
    size_t n = (size_t)rows * heads * hd;

    uint8_t axis[64];
    float inv[64];
    qw_rope_tables(axis, inv, rd, cfg->rope_theta, cfg->mrope_section);

    qw_buf xb = qw_buf_alloc(n * sizeof(float));
    qw_buf pb = qw_buf_alloc((size_t)3 * rows * sizeof(int32_t));
    qw_buf ab = qw_buf_alloc((size_t)half);
    qw_buf fb = qw_buf_alloc((size_t)half * sizeof(float));
    float *x = qw_buf_contents(xb);
    int32_t *pos = qw_buf_contents(pb);
    memcpy(qw_buf_contents(ab), axis, (size_t)half);
    memcpy(qw_buf_contents(fb), inv, (size_t)half * sizeof(float));
    fill_random(x, n, 0x777u);

    for (int32_t r = 0; r < rows; r++) {
        /* Text-only positions agree on every axis; the multimodal case forces
         * them apart so a selector mistake cannot hide. */
        pos[0 * rows + r] = base_pos + r;
        pos[1 * rows + r] = multimodal ? 7 + r : base_pos + r;
        pos[2 * rows + r] = multimodal ? 40 + r : base_pos + r;
    }

    float *ref = malloc(n * sizeof(float));
    memcpy(ref, x, n * sizeof(float));

    qw_cmd c = qw_cmd_begin();
    qw_op_rope_partial(c, qw_ref_at(xb, 0), qw_ref_at(pb, 0), qw_ref_at(ab, 0),
                       qw_ref_at(fb, 0), rows, heads, hd, rd);
    qw_cmd_wait(c);
    CHECK(qw_cmd_error(c) == NULL, "rope: %s", qw_cmd_error(c));
    qw_cmd_free(c);

    qw_cpu_rope_partial(ref, pos, axis, inv, rows, heads, hd, rd);

    size_t at = 0;
    double l2   = rel_l2(x, ref, n);
    double mabs = max_abs(x, ref, n, &at);
    /* Both sides evaluate cos/sin in fp32; Metal's and libm's agree to about
     * 1e-8, so a rotated value built from two of them lands within a few
     * multiples of that. */
    CHECK(l2 < 1e-6, "rope: rel l2 %.3g", l2);
    CHECK(mabs < 1e-5, "rope: max abs err %.3g at %zu", mabs, at);

    /* Dims past rotary_dim must be untouched -- this is the "partial" in
     * partial rotary, and 192 of 256 dims depend on it. */
    float *fresh = malloc(n * sizeof(float));
    fill_random(fresh, n, 0x777u);
    bool tail_intact = true;
    for (int32_t rh = 0; rh < rows * heads; rh++)
        for (int32_t i = rd; i < hd; i++)
            if (x[(size_t)rh * hd + i] != fresh[(size_t)rh * hd + i]) tail_intact = false;
    CHECK(tail_intact, "rope modified dims past rotary_dim");

    /* And the rotation must preserve the norm of the pair it acts on. */
    double n_before = 0, n_after = 0;
    for (int32_t i = 0; i < rd; i++) {
        n_before += (double)fresh[i] * fresh[i];
        n_after  += (double)x[i] * x[i];
    }
    CHECK(fabs(n_before - n_after) / n_before < 1e-5,
          "rope changed the rotated-block norm: %g -> %g", n_before, n_after);

    printf("  %-30s rows=%-3d heads=%-3d %s  rel l2 %.2e  max abs %.2e\n",
           "rope partial", rows, heads, multimodal ? "mrope" : "text ", l2, mabs);
    free(ref); free(fresh);
    qw_buf_free(xb); qw_buf_free(pb); qw_buf_free(ab); qw_buf_free(fb);
}

static void test_embed(qwasar_engine *e, const qw_config *cfg) {
    const int32_t n_tok = 6;
    /* Real ids from the chat template plus the vocabulary edges. */
    const int32_t ids[6] = { 248045, 872, 198, 0, 248046, 248319 };

    qw_buf tb = qw_buf_alloc((size_t)n_tok * sizeof(int32_t));
    qw_buf yb = qw_buf_alloc((size_t)n_tok * cfg->hidden_size * sizeof(float));
    memcpy(qw_buf_contents(tb), ids, sizeof ids);
    float *y = qw_buf_contents(yb);
    memset(y, 0xCD, (size_t)n_tok * cfg->hidden_size * sizeof(float));

    const qw_tensor *w  = qwasar_engine_tensor(e, "language_model.model.embed_tokens.weight");
    const qw_tensor *sc = qwasar_engine_tensor(e, "language_model.model.embed_tokens.scales");
    const qw_tensor *bi = qwasar_engine_tensor(e, "language_model.model.embed_tokens.biases");
    CHECK(w && sc && bi, "embedding tensors missing");
    if (!w) return;

    qw_cmd c = qw_cmd_begin();
    qw_op_embed_q4(c, qw_ref_at(yb, 0), qw_ref_at(tb, 0), qw_tensor_ref(w),
                   qw_tensor_ref(sc), qw_tensor_ref(bi), cfg->hidden_size, n_tok);
    qw_cmd_wait(c);
    CHECK(qw_cmd_error(c) == NULL, "embed: %s", qw_cmd_error(c));
    qw_cmd_free(c);

    size_t n = (size_t)n_tok * cfg->hidden_size;
    float *ref = malloc(n * sizeof(float));
    qw_cpu_embed_q4(ref, ids, qw_tensor_data(w), qw_tensor_data(sc), qw_tensor_data(bi),
                    cfg->hidden_size, n_tok);

    size_t at = 0;
    double worst = max_rel(y, ref, n, &at);
    CHECK(worst < 1e-6, "embed: worst rel err %.3g at %zu", worst, at);

    /* Distinct tokens must give distinct embeddings; an indexing bug that
     * returned row 0 every time would otherwise pass the comparison above. */
    bool distinct = false;
    for (int32_t i = 0; i < cfg->hidden_size; i++)
        if (y[i] != y[(size_t)cfg->hidden_size + i]) { distinct = true; break; }
    CHECK(distinct, "embed returned identical rows for different tokens");

    printf("  %-30s tokens=%-3d hidden=%-5d  worst rel err %.2e\n",
           "embed lookup", n_tok, cfg->hidden_size, worst);
    free(ref);
    qw_buf_free(tb); qw_buf_free(yb);
}

static void test_split_and_add(const qw_config *cfg) {
    const int32_t rows = 3, heads = cfg->num_attention_heads, dim = cfg->head_dim;
    size_t half = (size_t)rows * heads * dim, full = half * 2;

    qw_buf sb = qw_buf_alloc(full * sizeof(float));
    qw_buf ab = qw_buf_alloc(half * sizeof(float));
    qw_buf bb = qw_buf_alloc(half * sizeof(float));
    float *src = qw_buf_contents(sb);
    fill_random(src, full, 0x9090u);

    qw_cmd c = qw_cmd_begin();
    qw_op_split_heads2(c, qw_ref_at(ab, 0), qw_ref_at(bb, 0), qw_ref_at(sb, 0),
                       rows, heads, dim);
    qw_cmd_wait(c);
    CHECK(qw_cmd_error(c) == NULL, "split: %s", qw_cmd_error(c));
    qw_cmd_free(c);

    const float *a = qw_buf_contents(ab), *b = qw_buf_contents(bb);
    bool ok = true;
    for (int32_t rh = 0; rh < rows * heads && ok; rh++)
        for (int32_t i = 0; i < dim; i++) {
            if (a[(size_t)rh * dim + i] != src[(size_t)rh * 2 * dim + i]) { ok = false; break; }
            if (b[(size_t)rh * dim + i] != src[(size_t)rh * 2 * dim + dim + i]) { ok = false; break; }
        }
    CHECK(ok, "split_heads2 produced wrong interleaving");
    printf("  %-30s rows=%-3d heads=%-3d dim=%d\n", "split heads (q | gate)", rows, heads, dim);

    /* residual add */
    const int32_t n = cfg->hidden_size;
    qw_buf yb = qw_buf_alloc((size_t)n * sizeof(float));
    qw_buf xb = qw_buf_alloc((size_t)n * sizeof(float));
    float *y = qw_buf_contents(yb), *x = qw_buf_contents(xb);
    fill_random(y, (size_t)n, 0xAAu);
    fill_random(x, (size_t)n, 0xBBu);
    float *expect = malloc((size_t)n * sizeof(float));
    for (int32_t i = 0; i < n; i++) expect[i] = y[i] + x[i];

    c = qw_cmd_begin();
    qw_op_add_inplace(c, qw_ref_at(yb, 0), qw_ref_at(xb, 0), n);
    qw_cmd_wait(c);
    qw_cmd_free(c);

    size_t at = 0;
    double worst = max_rel(y, expect, (size_t)n, &at);
    CHECK(worst == 0.0, "add_inplace: rel err %.3g at %zu", worst, at);
    printf("  %-30s n=%-7d          exact\n", "residual add", n);
    free(expect);
    qw_buf_free(sb); qw_buf_free(ab); qw_buf_free(bb);
    qw_buf_free(yb); qw_buf_free(xb);
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
    const qw_layer *l0 = qwasar_engine_layer(e, 0);
    const qw_layer *l3 = qwasar_engine_layer(e, 3);
    const float eps = cfg->rms_norm_eps;

    /* Each of the three shapes the model actually normalises at. */
    test_rms_norm("hidden (input_layernorm)", l0->input_layernorm,
                  cfg->hidden_size, 1, eps, 1.0f);
    test_rms_norm("hidden, prefill rows", l0->input_layernorm,
                  cfg->hidden_size, 7, eps, 1.0f);
    test_rms_norm("per-head q_norm", l3->q_norm,
                  cfg->head_dim, cfg->num_attention_heads, eps, 1.0f);
    test_rms_norm("per-head k_norm", l3->k_norm,
                  cfg->head_dim, cfg->num_key_value_heads, eps, 1.0f);

    /* The gated-delta L2 normalisations: no weight, and an out_scale that
     * converts rms_norm into l2norm (and, for q, folds in the 1/sqrt(Dk)
     * attention scale on top). */
    const int32_t dk = cfg->linear_key_head_dim;
    test_rms_norm("gdn k = l2norm(k)", NULL,
                  dk, cfg->linear_num_key_heads, 1e-6f, 1.0f / sqrtf((float)dk));
    test_rms_norm("gdn q = l2norm(q)/sqrt(Dk)", NULL,
                  dk, cfg->linear_num_key_heads, 1e-6f, 1.0f / (float)dk);

    test_rms_norm_gated("gdn output norm x silu(z)", l0->gdn_norm,
                        cfg->linear_value_head_dim, cfg->linear_num_value_heads, eps);

    test_rope_tables(cfg);
    test_rope(cfg, 1, cfg->num_attention_heads, false, 100);
    test_rope(cfg, 5, cfg->num_key_value_heads, false, 100);
    test_rope(cfg, 5, cfg->num_attention_heads, true, 100);
    /* Near max_position_embeddings: cos/sin must still reduce the argument
     * correctly at angles of ~2.6e5 radians. */
    test_rope(cfg, 2, cfg->num_attention_heads, false, 262000);
    test_embed(e, cfg);
    test_split_and_add(cfg);

    test_swiglu(cfg->intermediate_size);
    test_swiglu(1);          /* single element: tail handling */
    test_swiglu(257);        /* not a multiple of the threadgroup width */

    qwasar_engine_free(e);
    qw_gpu_shutdown();
    if (fails) { fprintf(stderr, "%d check(s) failed\n", fails); return 1; }
    printf("ok: ops\n");
    return 0;
}
