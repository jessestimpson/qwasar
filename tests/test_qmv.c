/* Validates qw_qmv_q4_g64 against the scalar CPU reference using real weights
 * from the model.  Synthetic weights would not catch a misread of the MLX
 * affine layout -- packing order, group stride, signed scales -- which is the
 * failure mode this test exists for. */

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

/* Deterministic pseudo-random activations in [-1, 1]. */
static void fill_random(float *v, size_t n, uint32_t seed) {
    uint32_t s = seed | 1u;
    for (size_t i = 0; i < n; i++) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        v[i] = (float)((int32_t)(s >> 8) - 8388608) / 8388608.0f;
    }
}

/* Compares one output element against a CPU dot product over the dequantised
 * weight row, and reports relative error scaled by the row's magnitude. */
static double row_rel_err(const qw_qlinear *ql, const float *x, float got, int32_t row,
                          float *scratch) {
    const uint32_t *w  = qw_tensor_data(ql->weight);
    const uint16_t *sc = qw_tensor_data(ql->scales);
    const uint16_t *bi = qw_tensor_data(ql->biases);

    qw_cpu_dequant_row(scratch, w, sc, bi, ql->in_features, row);

    double acc = 0.0, mag = 0.0;
    for (int32_t i = 0; i < ql->in_features; i++) {
        acc += (double)scratch[i] * (double)x[i];
        mag += fabs((double)scratch[i] * (double)x[i]);
    }
    /* Scale by the sum of absolute terms, not by the result: a dot product of
     * random signs cancels to near zero, and dividing by that manufactures huge
     * "errors" out of ordinary rounding. */
    return mag > 0.0 ? fabs(acc - (double)got) / mag : fabs(acc - (double)got);
}

typedef enum { QW_USE_QMV, QW_USE_QMM } qw_impl;

static void test_linear_impl(const qw_qlinear *ql, const char *label, int32_t rows,
                             qw_impl impl) {
    const int32_t k = ql->in_features, n = ql->out_features;

    qw_buf xb = qw_buf_alloc((size_t)rows * k * sizeof(float));
    qw_buf yb = qw_buf_alloc((size_t)rows * n * sizeof(float));
    CHECK(xb && yb, "%s: buffer allocation failed", label);
    if (!xb || !yb) return;

    float *x = qw_buf_contents(xb);
    float *y = qw_buf_contents(yb);
    fill_random(x, (size_t)rows * k, 0x9E3779B9u);
    memset(y, 0xCD, (size_t)rows * n * sizeof(float));   /* poison, to catch no-ops */

    qw_cmd c = qw_cmd_begin();
    if (impl == QW_USE_QMM)
        qw_op_qmm_q4(c, qw_ref_at(yb, 0), qw_ref_at(xb, 0),
                     qw_tensor_ref(ql->weight), qw_tensor_ref(ql->scales),
                     qw_tensor_ref(ql->biases), k, n, rows);
    else
        qw_op_qmv_q4(c, qw_ref_at(yb, 0), qw_ref_at(xb, 0),
                     qw_tensor_ref(ql->weight), qw_tensor_ref(ql->scales),
                     qw_tensor_ref(ql->biases), k, n, rows);
    qw_cmd_wait(c);
    CHECK(qw_cmd_error(c) == NULL, "%s: GPU error: %s", label, qw_cmd_error(c));
    qw_cmd_free(c);

    float *scratch = malloc((size_t)k * sizeof(float));

    /* Sample across the whole output range: the first and last rows catch
     * boundary handling when n is not a multiple of the rows-per-simdgroup
     * blocking, and a spread of interior rows catches stride mistakes. */
    int32_t probes[16];
    int np = 0;
    probes[np++] = 0;
    probes[np++] = 1;
    probes[np++] = n - 1;
    if (n > 2) probes[np++] = n - 2;
    for (int i = 1; i <= 8 && np < 16; i++) {
        int32_t r = (int32_t)((int64_t)n * i / 9);
        if (r >= 0 && r < n) probes[np++] = r;
    }

    double worst = 0.0;
    int32_t worst_row = -1;
    for (int32_t b = 0; b < rows; b++) {
        for (int i = 0; i < np; i++) {
            int32_t row = probes[i];
            float got = y[(size_t)b * n + row];
            CHECK(isfinite(got), "%s: y[%d][%d] is not finite", label, b, row);
            double e = row_rel_err(ql, x + (size_t)b * k, got, row, scratch);
            if (e > worst) { worst = e; worst_row = row; }
        }
    }

    /* Both sides accumulate in fp32 but in different orders (a simdgroup tree
     * versus a sequential loop), so agreement is bounded by fp32 rounding over
     * k terms, not by the quantisation. */
    CHECK(worst < 2e-6, "%s: worst relative error %.3g at row %d (rows=%d)",
          label, worst, worst_row, rows);
    printf("  %-3s %-24s k=%-6d n=%-7d rows=%-3d  worst rel err %.2e\n",
           impl == QW_USE_QMM ? "mm" : "mv", label, k, n, rows, worst);

    free(scratch);
    qw_buf_free(xb);
    qw_buf_free(yb);
}

static void test_linear(const qw_qlinear *ql, const char *label, int32_t rows) {
    test_linear_impl(ql, label, rows, QW_USE_QMV);
}

/* The two implementations must agree; prefill and decode traverse the same
 * weights through different kernels, and generation quality would depend on
 * which path a prompt happened to take. */
static void test_both(const qw_qlinear *ql, const char *label, int32_t rows) {
    test_linear_impl(ql, label, rows, QW_USE_QMV);
    test_linear_impl(ql, label, rows, QW_USE_QMM);
}

int main(int argc, char **argv) {
    const char *model = getenv("QWASAR_TEST_MODEL");
    if (argc > 1) model = argv[1];
    if (!model) {
        fprintf(stderr, "skip: set QWASAR_TEST_MODEL to a model directory\n");
        return 0;
    }

    char err[512];
    qwasar_options opts = { .model_path = model };
    qwasar_engine *e = qwasar_engine_load(&opts, err, sizeof err);
    if (!e) { fprintf(stderr, "load failed: %s\n", err); return 1; }

    const qw_config *cfg = qwasar_engine_config(e);

    /* One of each shape the graph actually dispatches: a narrow projection, a
     * square-ish one, the widest MLP row, and a contracting one. */
    const qw_layer *l0 = qwasar_engine_layer(e, 0);                      /* gated delta */
    const qw_layer *l3 = qwasar_engine_layer(e, 3);                      /* full attn   */
    CHECK(l0 && l0->is_linear_attn, "layer 0 should be gated-delta");
    CHECK(l3 && !l3->is_linear_attn, "layer 3 should be full attention");

    if (l0 && l3) {
        test_linear(&l0->in_proj_b,   "l0.linear_attn.in_proj_b", 1);
        test_linear(&l0->in_proj_qkv, "l0.linear_attn.in_proj_qkv", 1);
        test_linear(&l3->q_proj,      "l3.self_attn.q_proj", 1);
        test_linear(&l3->k_proj,      "l3.self_attn.k_proj", 1);
        test_linear(&l0->gate_proj,   "l0.mlp.gate_proj", 1);
        test_linear(&l0->down_proj,   "l0.mlp.down_proj", 1);
        /* Multi-row is the prefill shape; it must give the same answers. */
        test_linear(&l3->k_proj,      "l3.self_attn.k_proj", 3);

        /* The tiled matmul, at sizes that exercise its blocking: exactly one
         * tile, a ragged token tail, and several full tiles. */
        test_both(&l3->k_proj,        "l3.self_attn.k_proj", 64);
        test_both(&l3->k_proj,        "l3.self_attn.k_proj", 7);
        test_both(&l0->gate_proj,     "l0.mlp.gate_proj", 65);
        test_both(&l0->down_proj,     "l0.mlp.down_proj", 128);
        test_both(&l0->in_proj_b,     "l0.linear_attn.in_proj_b", 33);
    }

    /* n = 48 is not a multiple of the kernel's 4-rows-per-simdgroup blocking
     * times 8 simdgroups, so in_proj_b also proves the tail path. */
    CHECK(cfg->linear_num_value_heads % 32 != 0,
          "in_proj_b no longer exercises the ragged-tail path");

    qwasar_engine_free(e);
    qw_gpu_shutdown();

    if (fails) { fprintf(stderr, "%d check(s) failed\n", fails); return 1; }
    printf("ok: qmv\n");
    return 0;
}
