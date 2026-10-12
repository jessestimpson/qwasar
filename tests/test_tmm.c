/* Validates the tensor-op matmuls (metal4/qmm_tensor.metal) against the
 * scalar CPU reference and against qw_op_qmm_q4, on both group sizes the
 * models use and on shapes chosen for their edges.
 *
 * Synthetic weights, unlike tests/test_qmv: that test exists to catch a
 * misread of the MLX layout, and qw_cpu_dequant_row -- which it validates on
 * real weights -- is the layout authority here.  What this one has to catch is
 * the tensor-op side of the same question: uint4b_format's nibble order, the
 * order of a cooperative tensor's indices, the extents of a slice.  Every word
 * is random, so a swapped nibble pair or a transposed tile cannot pass.
 *
 * Runs anywhere Metal 4 tensor ops do; skips where they do not.  On M1-M4
 * the ops lower onto the ordinary simdgroup path, so the answers are the
 * same and the speed is not the M5's.
 *
 *   tests/test_tmm           correctness
 *   tests/test_tmm --bench   time all three on model shapes */

#include "qwasar_gpu.h"
#include "qwasar_model.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int fails;

#define CHECK(cond, ...) do { \
    if (!(cond)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
                   fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); fails++; } \
} while (0)

static uint32_t rng_next(uint32_t *s) {
    *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5;
    return *s;
}

static float rng_unit(uint32_t *s) {   /* [-1, 1] */
    return (float)((int32_t)(rng_next(s) >> 8) - 8388608) / 8388608.0f;
}

static uint16_t f32_to_bf16(float v) {
    uint32_t u;
    memcpy(&u, &v, sizeof u);
    return (uint16_t)((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
}

/* A quantised weight laid out as in a safetensors shard: weight, scales and
 * biases at interior offsets of one buffer, not each at the start of its own.
 * Scales take both signs, as MLX's do. */
typedef struct {
    qw_buf buf;
    size_t w_off, s_off, b_off;
    int32_t k, n, group;
} qweight;

static qweight make_weight(int32_t k, int32_t n, int32_t group, uint32_t seed) {
    qweight q = { .k = k, .n = n, .group = group };
    const size_t wbytes = (size_t)n * k / 2, gbytes = (size_t)n * (k / group) * 2;
    q.w_off = 256;   /* not zero, so an op that ignores offsets fails */
    q.s_off = q.w_off + wbytes + 64;
    q.b_off = q.s_off + gbytes + 64;
    q.buf = qw_buf_alloc(q.b_off + gbytes);
    if (!q.buf) return q;
    uint8_t *base = qw_buf_contents(q.buf);
    uint32_t *w = (uint32_t *)(base + q.w_off);
    uint16_t *s = (uint16_t *)(base + q.s_off), *b = (uint16_t *)(base + q.b_off);
    uint32_t r = seed | 1u;
    for (size_t i = 0; i < wbytes / 4; i++) w[i] = rng_next(&r);
    for (size_t i = 0; i < gbytes / 2; i++) {
        s[i] = f32_to_bf16(rng_unit(&r) * 0.02f);
        b[i] = f32_to_bf16(rng_unit(&r) * 0.1f);
    }
    return q;
}

static const uint32_t *qw_w(const qweight *q) {
    return (const uint32_t *)((uint8_t *)qw_buf_contents(q->buf) + q->w_off);
}
static const uint16_t *qw_s(const qweight *q) {
    return (const uint16_t *)((uint8_t *)qw_buf_contents(q->buf) + q->s_off);
}
static const uint16_t *qw_b(const qweight *q) {
    return (const uint16_t *)((uint8_t *)qw_buf_contents(q->buf) + q->b_off);
}

typedef enum { USE_QMM, USE_DEQUANT, USE_INT4 } impl;
static const char *impl_name(impl i) {
    return i == USE_DEQUANT ? "dequant" : i == USE_INT4 ? "int4" : "qmm";
}

/* Encodes one product; returns false if the op declined it. */
static bool encode(qw_cmd c, impl which, const qweight *q, qw_ref y, qw_ref x, int32_t rows) {
    qw_ref w = qw_ref_at(q->buf, q->w_off), s = qw_ref_at(q->buf, q->s_off),
           b = qw_ref_at(q->buf, q->b_off);
    if (which == USE_QMM) {
        qw_op_qmm_q4(c, y, x, w, s, b, q->k, q->n, rows, q->group);
        return true;
    }
    return qw_op_tmm_q4(c, which == USE_INT4 ? QW_TMM_INT4 : QW_TMM_DEQUANT,
                        y, x, w, s, b, q->k, q->n, rows, q->group);
}

/* Runs `which` into a fresh, poisoned y; NULL if the op declined or failed. */
static float *run(impl which, const qweight *q, qw_buf xb, size_t x_off, int32_t rows, qw_buf *ybuf) {
    *ybuf = qw_buf_alloc((size_t)rows * q->n * sizeof(float) + 128);
    if (!*ybuf) return NULL;
    float *y = (float *)((uint8_t *)qw_buf_contents(*ybuf) + 128);
    memset(y, 0xCD, (size_t)rows * q->n * sizeof(float));   /* poison, to catch no-ops */
    qw_cmd c = qw_cmd_begin();
    bool ok = encode(c, which, q, qw_ref_at(*ybuf, 128), qw_ref_at(xb, x_off), rows);
    qw_cmd_wait(c);
    const char *err = qw_cmd_error(c);
    CHECK(err == NULL, "%s: GPU error: %s", impl_name(which), err);
    qw_cmd_free(c);
    return ok && !err ? y : NULL;
}

/* Error relative to the sum of absolute terms, as in tests/test_qmv: random
 * signs cancel, and dividing by the result would call rounding a failure. */
static void test_case(int32_t k, int32_t n, int32_t rows, int32_t group, uint32_t seed) {
    qweight q = make_weight(k, n, group, seed);
    const size_t x_off = 64;
    qw_buf xb = qw_buf_alloc(x_off + (size_t)rows * k * sizeof(float));
    CHECK(q.buf && xb, "allocation failed");
    if (!q.buf || !xb) return;
    float *x = (float *)((uint8_t *)qw_buf_contents(xb) + x_off);
    uint32_t r = seed * 2654435761u | 1u;
    for (size_t i = 0; i < (size_t)rows * k; i++) x[i] = rng_unit(&r);

    qw_buf qb = NULL;
    float *ref = run(USE_QMM, &q, xb, x_off, rows, &qb);
    CHECK(ref != NULL, "qmm reference failed");

    /* Every row of the first and last token tiles, and a stride through the
     * rest: the tile seams in n are at multiples of QW_TMM_BN. */
    float *wrow = malloc((size_t)k * sizeof(float));
    for (impl which = USE_DEQUANT; which <= USE_INT4; which++) {
        qw_buf yb = NULL;
        float *y = run(which, &q, xb, x_off, rows, &yb);
        CHECK(y != NULL, "%s declined k=%d n=%d g=%d", impl_name(which), k, n, group);
        if (!y) { qw_buf_free(yb); continue; }

        double worst = 0.0, worst_qmm = 0.0;
        int32_t wr = -1, wn = -1, checked = 0;
        for (int32_t o = 0; o < n; o++) {
            if (o % 7 && o % QW_TMM_BN && (o + 1) % QW_TMM_BN && o != n - 1) continue;
            qw_cpu_dequant_row(wrow, qw_w(&q), qw_s(&q), qw_b(&q), k, o, group);
            for (int32_t t = 0; t < rows; t++) {
                if (t > 2 && t < rows - 2 && t % 5) continue;
                const float *xt = x + (size_t)t * k;
                double acc = 0.0, mag = 0.0;
                for (int32_t i = 0; i < k; i++) {
                    acc += (double)wrow[i] * xt[i];
                    mag += fabs((double)wrow[i] * xt[i]);
                }
                const float got = y[(size_t)t * n + o];
                double e = isfinite(got) ? fabs(acc - got) / mag : 1e300;
                if (!(e <= worst)) { worst = e; wr = t; wn = o; }
                if (ref) {
                    double eq = fabs((double)ref[(size_t)t * n + o] - got) / mag;
                    if (eq > worst_qmm) worst_qmm = eq;
                }
                checked++;
            }
        }
        /* Each form rounds one operand to half: dequant the weights (x is
         * read as float), int4 the activations (the nibbles are exact).  qmm
         * rounds both, so each form's distance from qmm is the other's from
         * the CPU.  On a long contraction the errors average out and
         * qw_qmm_q4_g64's tolerance applies (tests/test_qmv explains the
         * budget); on a short one they need not, and the bound is the
         * rigorous one -- two half roundings, 2 * 2^-11 of the sum of
         * absolute terms. */
        const double tol = k >= 1024 ? 5e-5 : 1e-3;
        CHECK(worst < tol, "%s k=%d n=%d rows=%d g=%d: worst rel err %.3g at [%d][%d]",
              impl_name(which), k, n, rows, group, worst, wr, wn);
        CHECK(worst_qmm < tol, "%s k=%d n=%d rows=%d g=%d: disagrees with qmm by %.3g",
              impl_name(which), k, n, rows, group, worst_qmm);
        printf("  %-7s k=%-5d n=%-5d rows=%-4d g=%-2d  vs cpu %.2e  vs qmm %.2e  (%d checked)\n",
               impl_name(which), k, n, rows, group, worst, worst_qmm, checked);
        qw_buf_free(yb);
    }
    free(wrow);
    qw_buf_free(qb);
    qw_buf_free(xb);
    qw_buf_free(q.buf);
}

/* ---- bench ----------------------------------------------------------------
 *
 * Best of several, each its own command buffer, timed by wall clock around
 * the wait: these kernels run tens of milliseconds, so submission is noise. */

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void bench(const char *label, int32_t k, int32_t n, int32_t rows, int32_t group) {
    qweight q = make_weight(k, n, group, 7);
    qw_buf xb = qw_buf_alloc((size_t)rows * k * sizeof(float));
    qw_buf yb = qw_buf_alloc((size_t)rows * n * sizeof(float));
    if (!q.buf || !xb || !yb) { fprintf(stderr, "allocation failed\n"); return; }
    uint32_t r = 99;
    float *x = qw_buf_contents(xb);
    for (size_t i = 0; i < (size_t)rows * k; i++) x[i] = rng_unit(&r);

    printf("  %-26s k=%-5d n=%-5d rows=%-4d g=%-2d", label, k, n, rows, group);
    for (impl which = USE_QMM; which <= USE_INT4; which++) {
        double best = 1e300;
        bool declined = false;
        for (int it = 0; it < 6 && !declined; it++) {
            qw_cmd c = qw_cmd_begin();
            double t0 = now();
            bool ok = encode(c, which, &q, qw_ref_at(yb, 0), qw_ref_at(xb, 0), rows);
            qw_cmd_wait(c);
            double t = now() - t0;
            qw_cmd_free(c);
            declined = !ok;
            if (it > 0 && t < best) best = t;   /* the first pays pipeline builds */
        }
        if (declined) printf("  %s declined", impl_name(which));
        else printf("  %s %6.2f ms %5.2f TF", impl_name(which), best * 1e3,
                    2.0 * k * n * rows / best / 1e12);
    }
    printf("\n");
    qw_buf_free(q.buf);
    qw_buf_free(xb);
    qw_buf_free(yb);
}

int main(int argc, char **argv) {
    /* qw_op_qmm_q4 runs the dequant form by default; the comparison has to be
     * against the simdgroup kernel itself, whatever the caller set. */
    setenv("QWASAR_TMM", "qmm", 1);

    char err[512];
    if (!qw_gpu_init(err, sizeof err)) {
        fprintf(stderr, "GPU unavailable: %s\n", err);
        return 77;
    }
    printf("device: %s\n", qw_gpu_name());
    if (!qw_tmm_available()) {
        fprintf(stderr, "skip: Metal 4 tensor ops unavailable on this system\n");
        return 0;
    }

    if (argc > 1 && !strcmp(argv[1], "--bench")) {
        bench("27B mlp.gate_proj", 5120, 17408, 256, 64);
        bench("27B mlp.gate_proj", 5120, 17408, 1024, 64);
        bench("27B mlp.down_proj", 17408, 5120, 1024, 64);
        bench("Flash-Next in_proj_qkv", 2560, 10240, 1024, 32);
        bench("Flash-Next expert gate", 2560, 640, 64, 32);
        return 0;
    }

    /* Both group sizes; n ragged against QW_TMM_BN and smaller than it (the
     * delta layer's 48-row gates); rows of one, ragged, exactly a tile, and
     * several tiles; k from one BK step to a long contraction. */
    test_case(2560, 512, 64, 64, 1);
    test_case(2560, 48, 37, 64, 2);
    test_case(2560, 200, 1, 64, 3);
    test_case(64, 96, 70, 64, 4);
    test_case(5120, 1000, 200, 64, 5);
    test_case(2560, 640, 64, 32, 6);
    test_case(2560, 48, 130, 32, 7);
    test_case(10240, 320, 33, 32, 8);
    test_case(17408, 130, 65, 64, 9);

    /* Declines: an unsupported group and a k that is not whole steps. */
    qw_cmd c = qw_cmd_begin();
    qw_ref z = qw_ref_at(NULL, 0);
    CHECK(!qw_op_tmm_q4(c, QW_TMM_INT4, z, z, z, z, z, 2560, 64, 8, 128),
          "group 128 should be declined");
    CHECK(!qw_op_tmm_q4(c, QW_TMM_DEQUANT, z, z, z, z, z, 2592, 64, 8, 32),
          "k not a multiple of QW_TMM_BK should be declined");
    qw_cmd_wait(c);
    qw_cmd_free(c);

    if (fails) { fprintf(stderr, "%d failure(s)\n", fails); return 1; }
    printf("ok\n");
    return 0;
}
