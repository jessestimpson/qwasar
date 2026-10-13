/* Full attention and the KV cache.
 *
 * Beyond matching the CPU twin, three properties are checked directly because
 * a plausible-looking wrong answer is the failure mode here:
 *
 *   - causality: a token must not see keys past its own position;
 *   - GQA mapping: query head h must read kv head h/6, not h%4 or h/4;
 *   - softmax normalisation: the output must be a convex combination of values.
 */

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

static double rel_l2(const float *a, const float *b, size_t n) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = (double)a[i] - (double)b[i];
        num += d * d;
        den += (double)b[i] * (double)b[i];
    }
    return den > 0.0 ? sqrt(num / den) : sqrt(num);
}

/* max_ctx is kept small so the reference stays fast; the kernel's behaviour
 * does not depend on it beyond the cache stride. */
#define MAX_CTX 2048

/* Which kernel: the per-query one, the tiled prefill one (metal4/
 * attn_tensor.metal), or the split one for decode at depth -- the same answer
 * from each, to float rounding in another order. */
typedef enum { PER_QUERY, TILED, TILED_HALF, SPLIT, SPLIT_TENSOR } attn_kernel;
static const char *kernel_name[] = { "      ", " tiled", " tiled, half", " split", " split, tensor" };

static void test_attention(const qw_config *cfg, int32_t base_pos, int32_t rows, attn_kernel kern) {
    const int32_t qh = cfg->num_attention_heads, kvh = cfg->num_key_value_heads;
    const int32_t hd = cfg->head_dim;
    const float scale = 1.0f / sqrtf((float)hd);

    size_t qn = (size_t)rows * qh * hd;
    size_t kn = (size_t)rows * kvh * hd;
    size_t cn = (size_t)kvh * MAX_CTX * hd;

    qw_buf qb  = qw_buf_alloc(qn * sizeof(float));
    qw_buf ob  = qw_buf_alloc(qn * sizeof(float));
    qw_buf kb  = qw_buf_alloc(kn * sizeof(float));
    qw_buf vb  = qw_buf_alloc(kn * sizeof(float));
    qw_buf kcb = qw_buf_alloc(cn * sizeof(uint16_t));
    qw_buf vcb = qw_buf_alloc(cn * sizeof(uint16_t));

    float *q = qw_buf_contents(qb), *o = qw_buf_contents(ob);
    float *k = qw_buf_contents(kb), *v = qw_buf_contents(vb);
    uint16_t *kc = qw_buf_contents(kcb), *vc = qw_buf_contents(vcb);

    /* Prime the cache with base_pos tokens of history, exactly as a session
     * would have accumulated them. */
    uint16_t *kc_ref = calloc(cn, sizeof(uint16_t));
    uint16_t *vc_ref = calloc(cn, sizeof(uint16_t));
    memset(kc, 0, cn * sizeof(uint16_t));
    memset(vc, 0, cn * sizeof(uint16_t));
    if (base_pos > 0) {
        size_t hn = (size_t)base_pos * kvh * hd;
        float *hk = malloc(hn * sizeof(float)), *hv = malloc(hn * sizeof(float));
        fill_random(hk, hn, 0x5150u);
        fill_random(hv, hn, 0x6161u);
        qw_cpu_kv_write(kc, vc, hk, hv, base_pos, kvh, hd, MAX_CTX, 0);
        free(hk); free(hv);
    }
    /* Stale cache past what this step writes: huge keys, absurd values.  A
     * kernel that reads one position too far is no longer close. */
    for (int32_t h = 0; h < kvh; h++)
        for (int32_t t = base_pos + rows; t < MAX_CTX; t++)
            for (int32_t i = 0; i < hd; i++) {
                size_t idx = ((size_t)h * MAX_CTX + t) * hd + i;
                kc[idx] = qw_f32_to_f16_c(30.0f);
                vc[idx] = qw_f32_to_f16_c(-999.0f);
            }
    memcpy(kc_ref, kc, cn * sizeof(uint16_t));
    memcpy(vc_ref, vc, cn * sizeof(uint16_t));

    fill_random(q, qn, 0x1717u);
    fill_random(k, kn, 0x2727u);
    fill_random(v, kn, 0x3737u);
    memset(o, 0xCD, qn * sizeof(float));

    qw_cmd c = qw_cmd_begin();
    qw_op_kv_write(c, qw_ref_at(kcb, 0), qw_ref_at(vcb, 0), qw_ref_at(kb, 0),
                   qw_ref_at(vb, 0), rows, kvh, hd, MAX_CTX, base_pos);
    if (kern == PER_QUERY)
        qw_op_attn_decode(c, qw_ref_at(ob, 0), qw_ref_at(qb, 0), qw_ref_at(kcb, 0),
                          qw_ref_at(vcb, 0), rows, qh, kvh, hd, MAX_CTX, base_pos, scale);
    else if (kern == TILED || kern == TILED_HALF)
        CHECK(qw_op_attn_prefill_as(c, kern == TILED_HALF, qw_ref_at(ob, 0), qw_ref_at(qb, 0), qw_ref_at(kcb, 0),
                                    qw_ref_at(vcb, 0), rows, qh, kvh, hd, MAX_CTX, base_pos, scale),
              "the tiled kernel declined");
    else if (kern == SPLIT_TENSOR)
        CHECK(qw_op_attn_decode_tensor(c, qw_ref_at(ob, 0), qw_ref_at(qb, 0), qw_ref_at(kcb, 0),
                                       qw_ref_at(vcb, 0), rows, qh, kvh, hd, MAX_CTX, base_pos, scale),
              "the tensor split kernel declined");
    else
        CHECK(qw_op_attn_decode_split(c, qw_ref_at(ob, 0), qw_ref_at(qb, 0), qw_ref_at(kcb, 0),
                                      qw_ref_at(vcb, 0), rows, qh, kvh, hd, MAX_CTX, base_pos, scale),
              "the split kernel declined");
    qw_cmd_wait(c);
    CHECK(qw_cmd_error(c) == NULL, "attn: %s", qw_cmd_error(c));
    qw_cmd_free(c);

    qw_cpu_kv_write(kc_ref, vc_ref, k, v, rows, kvh, hd, MAX_CTX, base_pos);
    CHECK(memcmp(kc, kc_ref, cn * sizeof(uint16_t)) == 0, "kv_write: K cache mismatch");
    CHECK(memcmp(vc, vc_ref, cn * sizeof(uint16_t)) == 0, "kv_write: V cache mismatch");

    float *ref = malloc(qn * sizeof(float));
    qw_cpu_attn_decode(ref, q, kc_ref, vc_ref, rows, qh, kvh, hd, MAX_CTX, base_pos, scale);

    double e = rel_l2(o, ref, qn);
    /* The half form rounds Q and the weights to half: ~2e-4, and the 27B's
     * per-layer drift against mlx-vlm (tests/test_forward) is unchanged by it. */
    CHECK(e < (kern == TILED_HALF ? 5e-4 : 2e-5), "attn%s base_pos=%d rows=%d: rel l2 %.3g", kernel_name[kern], base_pos, rows, e);
    printf("  attention%s keys=%-4d rows=%-3d  rel l2 %.2e\n", kernel_name[kern], base_pos + rows, rows, e);

    free(ref); free(kc_ref); free(vc_ref);
    qw_buf_free(qb); qw_buf_free(ob); qw_buf_free(kb);
    qw_buf_free(vb); qw_buf_free(kcb); qw_buf_free(vcb);
}

/* Poisons the cache past the query's position: if the kernel reads it, the
 * output changes.  A causality bug is otherwise invisible in aggregate error. */
static void test_causality(const qw_config *cfg) {
    const int32_t qh = cfg->num_attention_heads, kvh = cfg->num_key_value_heads;
    const int32_t hd = cfg->head_dim, base_pos = 40;
    const float scale = 1.0f / sqrtf((float)hd);

    size_t qn = (size_t)qh * hd, cn = (size_t)kvh * MAX_CTX * hd;
    qw_buf qb = qw_buf_alloc(qn * 4), o1 = qw_buf_alloc(qn * 4), o2 = qw_buf_alloc(qn * 4);
    qw_buf kcb = qw_buf_alloc(cn * 2), vcb = qw_buf_alloc(cn * 2);
    float *q = qw_buf_contents(qb);
    uint16_t *kc = qw_buf_contents(kcb), *vc = qw_buf_contents(vcb);
    fill_random(q, qn, 0x8888u);

    size_t hn = (size_t)(base_pos + 1) * kvh * hd;
    float *hk = malloc(hn * 4), *hv = malloc(hn * 4);
    fill_random(hk, hn, 0x9999u);
    fill_random(hv, hn, 0xAAAAu);
    memset(kc, 0, cn * 2); memset(vc, 0, cn * 2);
    qw_cpu_kv_write(kc, vc, hk, hv, base_pos + 1, kvh, hd, MAX_CTX, 0);

    qw_cmd c = qw_cmd_begin();
    qw_op_attn_decode(c, qw_ref_at(o1, 0), qw_ref_at(qb, 0), qw_ref_at(kcb, 0),
                      qw_ref_at(vcb, 0), 1, qh, kvh, hd, MAX_CTX, base_pos, scale);
    qw_cmd_wait(c); qw_cmd_free(c);

    /* Everything strictly after base_pos becomes huge; a causal kernel is
     * unaffected because it never looks there. */
    for (int32_t h = 0; h < kvh; h++)
        for (int32_t t = base_pos + 1; t < MAX_CTX; t++)
            for (int32_t i = 0; i < hd; i++) {
                size_t idx = ((size_t)h * MAX_CTX + t) * hd + i;
                kc[idx] = qw_f32_to_f16_c(50.0f);
                vc[idx] = qw_f32_to_f16_c(-999.0f);
            }

    c = qw_cmd_begin();
    qw_op_attn_decode(c, qw_ref_at(o2, 0), qw_ref_at(qb, 0), qw_ref_at(kcb, 0),
                      qw_ref_at(vcb, 0), 1, qh, kvh, hd, MAX_CTX, base_pos, scale);
    qw_cmd_wait(c); qw_cmd_free(c);

    const float *a = qw_buf_contents(o1), *b = qw_buf_contents(o2);
    CHECK(memcmp(a, b, qn * 4) == 0, "attention read past its own position");
    printf("  causality  position %d ignores %d poisoned keys\n", base_pos, MAX_CTX - base_pos - 1);

    free(hk); free(hv);
    qw_buf_free(qb); qw_buf_free(o1); qw_buf_free(o2);
    qw_buf_free(kcb); qw_buf_free(vcb);
}

/* Gives each kv head a constant, distinguishable value vector.  A softmax over
 * anything still returns that constant, so the output identifies which kv head
 * each query head actually read -- and proves the weights sum to one. */
static void test_gqa_and_normalisation(const qw_config *cfg) {
    const int32_t qh = cfg->num_attention_heads, kvh = cfg->num_key_value_heads;
    const int32_t hd = cfg->head_dim, n = 33;
    const int32_t gqa = qh / kvh;
    const float scale = 1.0f / sqrtf((float)hd);

    size_t qn = (size_t)qh * hd, cn = (size_t)kvh * MAX_CTX * hd;
    qw_buf qb = qw_buf_alloc(qn * 4), ob = qw_buf_alloc(qn * 4);
    qw_buf kcb = qw_buf_alloc(cn * 2), vcb = qw_buf_alloc(cn * 2);
    float *q = qw_buf_contents(qb);
    uint16_t *kc = qw_buf_contents(kcb), *vc = qw_buf_contents(vcb);
    fill_random(q, qn, 0xBEADu);
    memset(kc, 0, cn * 2);

    for (int32_t h = 0; h < kvh; h++)
        for (int32_t t = 0; t < n; t++)
            for (int32_t i = 0; i < hd; i++)
                vc[((size_t)h * MAX_CTX + t) * hd + i] = qw_f32_to_f16_c((float)(h + 1));

    qw_cmd c = qw_cmd_begin();
    qw_op_attn_decode(c, qw_ref_at(ob, 0), qw_ref_at(qb, 0), qw_ref_at(kcb, 0),
                      qw_ref_at(vcb, 0), 1, qh, kvh, hd, MAX_CTX, n - 1, scale);
    qw_cmd_wait(c); qw_cmd_free(c);

    const float *o = qw_buf_contents(ob);
    double worst = 0.0;
    int wrong = -1;
    for (int32_t h = 0; h < qh; h++) {
        float want = (float)(h / gqa + 1);
        for (int32_t i = 0; i < hd; i++) {
            double d = fabs((double)o[(size_t)h * hd + i] - want);
            if (d > worst) { worst = d; wrong = h; }
        }
    }
    CHECK(worst < 1e-4, "gqa/normalisation: head %d off by %.3g", wrong, worst);
    printf("  gqa+softmax  %d q heads -> %d kv heads (factor %d), max dev %.2e\n",
           qh, kvh, gqa, worst);

    qw_buf_free(qb); qw_buf_free(ob); qw_buf_free(kcb); qw_buf_free(vcb);
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

    test_attention(cfg, 0, 1, PER_QUERY);      /* first token, empty history */
    test_attention(cfg, 1, 1, PER_QUERY);
    test_attention(cfg, 31, 1, PER_QUERY);     /* exactly one simdgroup's stride of keys */
    test_attention(cfg, 32, 1, PER_QUERY);     /* one past it */
    test_attention(cfg, 200, 1, PER_QUERY);
    test_attention(cfg, 0, 8, PER_QUERY);      /* prefill shape: causal over 8 queries */
    test_attention(cfg, 50, 4, PER_QUERY);     /* prefill continuing an existing cache */

    /* The tiled kernel: one tile and a ragged one, rows not a multiple of
     * QW_FA_BQ, keys not a multiple of QW_FA_BK, a deep cache, and a whole
     * prefill chunk. */
    test_attention(cfg, 0, 16, TILED);
    test_attention(cfg, 0, 32, TILED);
    test_attention(cfg, 0, 37, TILED);
    test_attention(cfg, 5, 64, TILED);
    test_attention(cfg, 300, 100, TILED);
    test_attention(cfg, 1500, 256, TILED);
    /* The half form, an M5's, on the same shapes: its tile is QW_FAH_BQ rows. */
    test_attention(cfg, 0, 16, TILED_HALF);
    test_attention(cfg, 0, 37, TILED_HALF);
    test_attention(cfg, 5, 64, TILED_HALF);
    test_attention(cfg, 300, 100, TILED_HALF);
    test_attention(cfg, 1500, 256, TILED_HALF);

    /* The split kernel: one token and a speculative verify's few, a cache
     * shallower than one slice, slice and simdgroup boundaries, and deep. */
    test_attention(cfg, 0, 1, SPLIT);
    test_attention(cfg, 40, 1, SPLIT);
    test_attention(cfg, 255, 1, SPLIT);
    test_attention(cfg, 256, 1, SPLIT);
    test_attention(cfg, 2000, 1, SPLIT);
    test_attention(cfg, 1990, 5, SPLIT);
    test_attention(cfg, 1000, 8, SPLIT);

    /* The same on tensor ops: slices of whole 64-key steps, the last one
     * ragged, rows past the head's own reading as zeros. */
    if (qw_tmm_available()) {
        test_attention(cfg, 0, 1, SPLIT_TENSOR);
        test_attention(cfg, 40, 1, SPLIT_TENSOR);
        test_attention(cfg, 255, 1, SPLIT_TENSOR);
        test_attention(cfg, 256, 1, SPLIT_TENSOR);
        test_attention(cfg, 2000, 1, SPLIT_TENSOR);
        test_attention(cfg, 1990, 5, SPLIT_TENSOR);
        test_attention(cfg, 1000, 8, SPLIT_TENSOR);
    }

    test_causality(cfg);
    test_gqa_and_normalisation(cfg);

    qwasar_engine_free(e);
    qw_gpu_shutdown();
    if (fails) { fprintf(stderr, "%d check(s) failed\n", fails); return 1; }
    printf("ok: attn\n");
    return 0;
}
