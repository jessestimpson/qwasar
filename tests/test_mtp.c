/* Validates the MTP draft head's load and its fusion step against the scalar
 * CPU reference, using the real head weights.
 *
 * The head is a separate download, so this test skips rather than fails when
 * QWASAR_TEST_MTP is unset -- the same shape as the model-directory tests.
 *
 * What is worth checking here is not the arithmetic, which is two ordinary
 * kernels, but the WIRING: that fifteen bare tensor names bind to the right
 * fifteen shapes, and that the concatenation lands embedding-half first.  A
 * head fused the wrong way round loads cleanly, runs at full speed, and drafts
 * nonsense -- there is no error for it to raise, so a test has to be the thing
 * that notices. */

#include "qwasar.h"
#include "qwasar_gpu.h"
#include "qwasar_model.h"

#include <math.h>
float qw_bf16_to_f32_c(uint16_t v);
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

/* Norm-relative L2 plus worst absolute, the metric PLAN.md section 4 settled
 * on: elementwise relative error on values that cancel to near zero reports
 * ordinary rounding as catastrophe. */
static void compare(const char *what, const float *got, const float *want,
                    size_t n, double tol) {
    double num = 0.0, den = 0.0, worst = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = (double)got[i] - (double)want[i];
        num += d * d;
        den += (double)want[i] * (double)want[i];
        if (fabs(d) > worst) worst = fabs(d);
        CHECK(isfinite(got[i]), "%s: element %zu is not finite", what, i);
    }
    double rel = den > 0.0 ? sqrt(num / den) : sqrt(num);
    CHECK(rel < tol, "%s: rel l2 %.3g exceeds %.3g", what, rel, tol);
    printf("  %-28s rel l2 %.2e  max abs %.2e\n", what, rel, worst);
}

int main(int argc, char **argv) {
    const char *model = getenv("QWASAR_TEST_MODEL");
    const char *head  = getenv("QWASAR_TEST_MTP");
    if (argc > 1) model = argv[1];
    if (argc > 2) head  = argv[2];
    /* make passes these through unconditionally, so an unset one arrives as an
     * empty string rather than as NULL. */
    if (!model || !*model || !head || !*head) {
        fprintf(stderr, "skip: set QWASAR_TEST_MODEL and QWASAR_TEST_MTP\n");
        return 0;
    }
    {   /* A head directory that is not there is a skip, not a failure: it is an
         * 849 MB download that most checkouts will not have. */
        char probe[1200];
        snprintf(probe, sizeof probe, "%s/model.safetensors", head);
        FILE *f = fopen(probe, "rb");
        if (!f) { fprintf(stderr, "skip: no MTP head at %s\n", head); return 0; }
        fclose(f);
    }

    char err[512];
    qwasar_options opts = { .model_path = model, .mtp_path = head };
    qwasar_engine *e = qwasar_engine_load(&opts, err, sizeof err);
    if (!e) { fprintf(stderr, "load failed: %s\n", err); return 1; }

    const qw_config *c = qwasar_engine_config(e);
    const qw_mtp *m = qwasar_engine_mtp(e);
    CHECK(m->present, "head did not bind");
    if (!m->present) return 1;

    /* The head is one full-attention layer of the base model's shape.  If that
     * ever stops being true the draft path has to be rewritten, so say it here
     * rather than discover it in a kernel. */
    const qw_layer *l3 = qwasar_engine_layer(e, 3);
    CHECK(l3 && !l3->is_linear_attn, "layer 3 should be full attention");
    CHECK(m->q_proj.out_features == l3->q_proj.out_features,
          "head q_proj is %d wide, base is %d",
          m->q_proj.out_features, l3->q_proj.out_features);
    CHECK(m->q_proj.out_features == 2 * c->num_attention_heads * c->head_dim,
          "head q_proj should carry query and gate together");
    CHECK(m->fc.in_features == 2 * c->hidden_size, "fc should take two hidden states");
    CHECK(m->block_size >= 1, "block_size should be positive");
    printf("  head bound: block_size %d, fc [%d -> %d]\n",
           m->block_size, m->fc.in_features, m->fc.out_features);

    const int32_t h = c->hidden_size, rows = 3;

    qw_buf eb = qw_buf_alloc((size_t)rows * h * sizeof(float));
    qw_buf hb = qw_buf_alloc((size_t)rows * h * sizeof(float));
    qw_buf fb = qw_buf_alloc((size_t)rows * 2 * h * sizeof(float));
    qw_buf yb = qw_buf_alloc((size_t)rows * h * sizeof(float));
    CHECK(eb && hb && fb && yb, "buffer allocation failed");
    if (!eb || !hb || !fb || !yb) return 1;

    float *ev = qw_buf_contents(eb), *hv = qw_buf_contents(hb);
    fill_random(ev, (size_t)rows * h, 0x1234567u);
    fill_random(hv, (size_t)rows * h, 0x89ABCDEu);
    memset(qw_buf_contents(fb), 0xCD, (size_t)rows * 2 * h * sizeof(float));
    memset(qw_buf_contents(yb), 0xCD, (size_t)rows * h * sizeof(float));

    qw_cmd cmd = qw_cmd_begin();
    qw_op_rms_norm_concat(cmd, qw_ref_at(fb, 0), qw_ref_at(eb, 0),
                          qw_tensor_ref(m->pre_fc_norm_embedding),
                          qw_ref_at(hb, 0), qw_tensor_ref(m->pre_fc_norm_hidden),
                          h, rows, c->rms_norm_eps);
    qw_op_dmat_bf16(cmd, qw_ref_at(yb, 0), qw_ref_at(fb, 0),
                    qw_tensor_ref(m->fc.weight), m->fc.in_features,
                    m->fc.out_features, rows);
    qw_cmd_wait(cmd);
    CHECK(qw_cmd_error(cmd) == NULL, "GPU error: %s", qw_cmd_error(cmd));
    qw_cmd_free(cmd);

    float *want_f = malloc((size_t)rows * 2 * h * sizeof(float));
    float *want_y = malloc((size_t)rows * h * sizeof(float));
    qw_cpu_rms_norm_concat(want_f, ev, qw_tensor_data(m->pre_fc_norm_embedding),
                           hv, qw_tensor_data(m->pre_fc_norm_hidden),
                           h, rows, c->rms_norm_eps);
    qw_cpu_dmv_bf16(want_y, want_f, qw_tensor_data(m->fc.weight),
                    m->fc.in_features, m->fc.out_features, rows);

    compare("concat", qw_buf_contents(fb), want_f, (size_t)rows * 2 * h, 1e-6);
    compare("fc bf16", qw_buf_contents(yb), want_y, (size_t)rows * h, 1e-5);

    /* The head is quantised at load and the bf16 weights are only the
     * quantiser's input, so what actually runs is the 4-bit path.  Checking it
     * against the bf16 one is the only test the quantiser has: nothing else can
     * catch a mis-packed nibble, because the head only proposes and a badly
     * quantised head simply drafts worse. */
    CHECK(m->quantised, "head was not quantised");
    CHECK(m->q_fc.in_features == m->fc.in_features &&
          m->q_fc.out_features == m->fc.out_features,
          "quantised fc is [%d -> %d], bf16 is [%d -> %d]",
          m->q_fc.in_features, m->q_fc.out_features,
          m->fc.in_features, m->fc.out_features);

    qw_buf yq = qw_buf_alloc((size_t)rows * h * sizeof(float));
    CHECK(yq != NULL, "buffer allocation failed");
    if (yq) {
        memset(qw_buf_contents(yq), 0xCD, (size_t)rows * h * sizeof(float));
        qw_cmd c2 = qw_cmd_begin();
        qw_op_qmat_q4(c2, qw_ref_at(yq, 0), qw_ref_at(fb, 0),
                      qw_tensor_ref(m->q_fc.weight), qw_tensor_ref(m->q_fc.scales),
                      qw_tensor_ref(m->q_fc.biases),
                      m->q_fc.in_features, m->q_fc.out_features, rows);
        qw_cmd_wait(c2);
        CHECK(qw_cmd_error(c2) == NULL, "GPU error: %s", qw_cmd_error(c2));
        qw_cmd_free(c2);

        /* Four bits over a group of 64 costs on the order of ten percent of
         * relative error in a matvec -- the step is about 0.3 standard
         * deviations and the error is uniform across it, so this is what the
         * format costs rather than a defect.  A loose bound here is therefore
         * a weak test, and it is kept only to prove the quantised path is
         * wired to the right tensors at all; the tight check is below. */
        compare("fc 4-bit vs bf16", qw_buf_contents(yq), qw_buf_contents(yb),
                (size_t)rows * h, 0.25);
        qw_buf_free(yq);
    }

    /* The real check on the quantiser: every weight must come back within half
     * a quantisation step of where it started.  That bound is structural -- it
     * holds for any correct affine quantisation and fails immediately for a
     * nibble packed at the wrong offset, a group stride off by one, or a scale
     * paired with the wrong row -- whereas an output-error threshold would
     * absorb all three. */
    {
        const uint32_t *qw = qw_tensor_data(m->q_fc.weight);
        const uint16_t *qs = qw_tensor_data(m->q_fc.scales);
        const uint16_t *qb = qw_tensor_data(m->q_fc.biases);
        const uint16_t *orig = qw_tensor_data(m->fc.weight);
        const int32_t in = m->q_fc.in_features, groups = in / 64;
        float *deq = malloc((size_t)in * sizeof(float));

        double worst_ratio = 0.0;
        int32_t worst_row = -1;
        /* A spread of rows, including the first and last, because an indexing
         * mistake usually shows at an edge. */
        for (int step = 0; step < 12; step++) {
            const int32_t row = step == 0 ? 0
                              : step == 1 ? m->q_fc.out_features - 1
                              : (int32_t)((int64_t)m->q_fc.out_features * step / 12);
            qw_cpu_dequant_row(deq, qw, qs, qb, in, row);
            for (int32_t i = 0; i < in; i++) {
                const float want = qw_bf16_to_f32_c(orig[(size_t)row * in + i]);
                const float step_sz = qw_bf16_to_f32_c(qs[(size_t)row * groups + i / 64]);
                const double tol = 0.5 * step_sz + 1e-6;
                const double ratio = tol > 0.0 ? fabs(want - deq[i]) / tol
                                               : fabs(want - deq[i]);
                if (ratio > worst_ratio) { worst_ratio = ratio; worst_row = row; }
            }
        }
        /* Slightly over one because the scale and bias are themselves rounded
         * to bf16, which widens the step a little either side. */
        CHECK(worst_ratio < 1.05,
              "quantised weight is %.3f half-steps from the original (row %d)",
              worst_ratio, worst_row);
        printf("  %-28s worst %.3f of a half-step\n", "quantiser round-trip",
               worst_ratio);
        free(deq);
    }

    /* The halves must not be interchangeable, or the test above would pass on a
     * head fused backwards.  Swapping the two norm weights has to move the
     * result: if it does not, the check is vacuous. */
    float *swapped = malloc((size_t)rows * 2 * h * sizeof(float));
    qw_cpu_rms_norm_concat(swapped, ev, qw_tensor_data(m->pre_fc_norm_hidden),
                           hv, qw_tensor_data(m->pre_fc_norm_embedding),
                           h, rows, c->rms_norm_eps);
    double diff = 0.0;
    for (size_t i = 0; i < (size_t)rows * 2 * h; i++)
        diff += fabs((double)swapped[i] - (double)want_f[i]);
    CHECK(diff > 1.0, "the two pre-fc norms are indistinguishable (sum |diff| %.3g); "
                      "this test cannot detect a reversed concatenation", diff);
    printf("  %-28s sum |diff| %.3g\n", "orientation is observable", diff);

    free(swapped); free(want_f); free(want_y);
    qw_buf_free(eb); qw_buf_free(hb); qw_buf_free(fb); qw_buf_free(yb);
    qwasar_engine_free(e);
    qw_gpu_shutdown();

    if (fails) { fprintf(stderr, "%d check(s) failed\n", fails); return 1; }
    printf("ok: mtp\n");
    return 0;
}
