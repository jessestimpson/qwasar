/* Flash-Next on its real weights, against mlx-vlm on the same weights.
 *
 * tools/flashnext_real_oracle.py ran mlx-vlm on mlx-community's 4-bit build
 * and recorded, per prompt, its top logits at every position and a greedy
 * continuation (tests/fixtures/flashnext-real-oracle.json).  This runs the
 * engine on the same build and the same token ids.
 *
 * Two things legitimately separate them.  mlx-vlm computes in BF16 and the
 * engine in fp32, and BF16's rounding compounds over 48 layers.  And every
 * layer picks 10 of 512 experts: when the 10th and 11th are within ~1e-7 of
 * each other, which of them makes the cut is decided by summation order, and
 * a different pick in one layer shifts the logits by a few percent from then
 * on.  Measured on the prompts here: logits within ~4% of scale where no such
 * tie occurs, ~10% after one.  So a top-1 disagreement counts against the
 * engine only where mlx-vlm was decisive, a greedy continuation may only
 * split where the engine itself was nearly tied, and the logit bound is loose
 * -- the mistakes it exists for (a gain without its +1, a wrong group, a
 * wrong bank) produce noise, not a few percent.
 *
 * And with QWASAR_FLASHNEXT_CPU=<n>, the Metal path is held to the engine's
 * own fp32 CPU reference (qwasar_flash_cpu.c, itself held to transformers and
 * mlx-vlm on the toys) for each prompt's first n tokens -- the real shapes at
 * full precision, at no cost in memory (it reads the same bound weights),
 * ~20 s a token.
 *
 * Needs the model: QWASAR_TEST_FLASHNEXT=<dir>; skips without it. */

#include "qwasar.h"
#include "qwasar_json.h"
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

/* mlx-vlm's top-2 gap below which a flip is BF16's to decide, in logits. */
#define NEAR_TIE 0.5f
/* A router cut (k-th minus (k+1)-th probability) this close is a coin toss. */
#define ROUTE_TIE 1e-7f

static int32_t argmax(const float *v, int32_t n) {
    int32_t b = 0;
    for (int32_t i = 1; i < n; i++) if (v[i] > v[b]) b = i;
    return b;
}

static int32_t read_ints(const qj_doc *d, const qj_node *arr, int32_t *out, int32_t cap) {
    int32_t n = 0;
    for (const qj_node *v = qj_first(d, arr); v && n < cap; v = qj_next(d, v)) out[n++] = (int32_t)v->u.num;
    return n;
}

int main(void) {
    const char *dir = getenv("QWASAR_TEST_FLASHNEXT");
    if (!dir || !*dir) { printf("skip: set QWASAR_TEST_FLASHNEXT to the mlx-community build\n"); return 0; }

    const char *oracle = getenv("QWASAR_FLASHNEXT_ORACLE");
    if (!oracle || !*oracle) oracle = "tests/fixtures/flashnext-real-oracle.json";
    qj_doc d;
    if (!qj_parse_file(&d, oracle)) {
        fprintf(stderr, "cannot read the oracle: %s\n", d.err);
        return 1;
    }

    char err[512] = "";
    qwasar_options o = { .model_path = dir, .context_size = 4096 };
    qwasar_engine *e = qwasar_engine_load(&o, err, sizeof err);
    if (!e) { fprintf(stderr, "load %s: %s\n", dir, err); qj_free(&d); return 1; }
    const int32_t vocab = qwasar_vocab_size(e);

    int32_t prompt_i = 0;
    for (const qj_node *p = qj_first(&d, qj_get(&d, qj_root(&d), "prompts")); p;
         p = qj_next(&d, p), prompt_i++) {
        int32_t tokens[512], greedy[128];
        const int32_t n = read_ints(&d, qj_get(&d, p, "tokens"), tokens, 512);
        const int32_t ng = read_ints(&d, qj_get(&d, p, "greedy"), greedy, 128);
        const qj_node *tids = qj_get(&d, p, "top_ids"), *tvals = qj_get(&d, p, "top_logits");

        /* Every position, through the decode path. */
        qwasar_session *s = qwasar_session_new(e, err, sizeof err);
        CHECK(s != NULL, "session: %s", err);
        if (!s) break;
        int32_t agree = 0, flips_near = 0, flips_decisive = 0;
        float worst_rel = 0.0f;
        const qj_node *ti = qj_first(&d, tids), *tv = qj_first(&d, tvals);
        for (int32_t t = 0; t < n; t++, ti = qj_next(&d, ti), tv = qj_next(&d, tv)) {
            const float *lg = qwasar_session_eval(s, tokens + t, 1, err, sizeof err);
            CHECK(lg != NULL, "eval: %s", err);
            if (!lg) break;
            int32_t ids[16];
            float vals[16];
            const int32_t k = read_ints(&d, ti, ids, 16);
            int32_t kv = 0;
            for (const qj_node *v = qj_first(&d, tv); v && kv < 16; v = qj_next(&d, v)) vals[kv++] = (float)v->u.num;
            float scale = 1.0f;
            for (int32_t j = 0; j < kv; j++) if (fabsf(vals[j]) > scale) scale = fabsf(vals[j]);
            float pos_rel = 0.0f;
            for (int32_t j = 0; j < k && j < kv; j++) {
                const float rel = fabsf(lg[ids[j]] - vals[j]) / scale;
                if (rel > pos_rel) pos_rel = rel;
            }
            if (pos_rel > worst_rel) worst_rel = pos_rel;
            if (getenv("QWASAR_FLASHNEXT_DETAIL"))
                printf("    p%d t%2d: top %.2f (mlx) vs %.2f (engine), max |diff| over top-10 %.3f (%.3f of scale)\n",
                       prompt_i, t, vals[0], lg[ids[0]], pos_rel * scale, pos_rel);
            const int32_t mine = argmax(lg, vocab);
            if (mine == ids[0]) agree++;
            else if (vals[0] - vals[1] < NEAR_TIE) flips_near++;
            else {
                flips_decisive++;
                printf("  prompt %d position %d: engine %d, mlx-vlm %d by %.2f\n",
                       prompt_i, t, mine, ids[0], vals[0] - vals[1]);
            }
        }
        qwasar_session_free(s);

        const char *cpu_n = getenv("QWASAR_FLASHNEXT_CPU");
        const int32_t nc = cpu_n ? (atoi(cpu_n) < n ? atoi(cpu_n) : n) : 0;
        if (nc > 0) {
            qw_flash_ref *r = qw_flash_ref_new(e, nc + 8);
            if (r) qw_flash_ref_debug(r, true);
            qwasar_session *ms = qwasar_session_new(e, err, sizeof err);
            float *ref = malloc((size_t)vocab * sizeof(float));
            float worst = 0.0f;
            int32_t amis = 0, tie_at = -1;
            for (int32_t t = 0; r && ms && ref && t < nc; t++) {
                if (!qw_flash_ref_forward(r, tokens + t, 1, ref, NULL, err, sizeof err)) {
                    CHECK(false, "cpu reference: %s", err);
                    break;
                }
                const float *lg = qwasar_session_eval(ms, tokens + t, 1, err, sizeof err);
                if (!lg) { CHECK(false, "eval: %s", err); break; }
                float md = 0.0f, scale = 1.0f;
                for (int32_t v = 0; v < vocab; v++) {
                    if (fabsf(lg[v] - ref[v]) > md) md = fabsf(lg[v] - ref[v]);
                    if (fabsf(ref[v]) > scale) scale = fabsf(ref[v]);
                }
                const bool mis = argmax(lg, vocab) != argmax(ref, vocab);
                /* The least decisive expert cut at this position, over every
                 * layer: a cut this close is one fp summation order can move. */
                float rg = 1.0f;
                int32_t rl = -1;
                for (int32_t L = 0; L < qwasar_n_layers(e); L++)
                    if (qw_flash_ref_route_gap(r, L, t) < rg) { rg = qw_flash_ref_route_gap(r, L, t); rl = L; }
                /* From an expert cut this close on, the two may route
                 * differently and both be right; compare up to it. */
                if (rg < ROUTE_TIE && tie_at < 0) tie_at = t;
                if (tie_at >= 0 && t >= tie_at) {
                    printf("    p%d t%2d: metal vs cpu reference %.4f of scale -- after a routing tie "
                           "at %d, not compared\n", prompt_i, t, md / scale, tie_at);
                    continue;
                }
                if (mis) amis++;
                if (md / scale > worst) worst = md / scale;
                printf("    p%d t%2d: metal vs cpu reference, max |diff| over the vocabulary %.4f "
                       "(%.4f of scale); closest expert cut %.2e (layer %d)\n",
                       prompt_i, t, md, md / scale, rg, rl);
            }
            printf("  prompt %d: metal vs fp32 cpu reference over %d tokens%s: worst %.4f of scale, "
                   "%d argmax mismatches\n", prompt_i, tie_at >= 0 ? tie_at : nc,
                   tie_at >= 0 ? " (up to a routing tie)" : "", worst, amis);
            CHECK(worst < 5e-3f && amis == 0, "prompt %d: metal diverges from the cpu reference", prompt_i);
            free(ref);
            if (ms) qwasar_session_free(ms);
            qw_flash_ref_free(r);
        }

        /* The continuation: prefill the prompt in one call, then greedy. */
        s = qwasar_session_new(e, err, sizeof err);
        const float *lg = s ? qwasar_session_eval(s, tokens, n, err, sizeof err) : NULL;
        CHECK(lg != NULL, "prefill: %s", err);
        int32_t same = 0;
        float gap_at_split = -1.0f;
        for (int32_t i = 0; lg && i < ng; i++) {
            const int32_t next = argmax(lg, vocab);
            if (next != greedy[i]) {
                /* How close the engine itself was: a split at a near-tie is
                 * two correct decoders choosing differently. */
                gap_at_split = lg[next] - lg[greedy[i]];
                break;
            }
            same++;
            lg = qwasar_session_eval(s, &next, 1, err, sizeof err);
        }
        if (s) qwasar_session_free(s);

        printf("  prompt %d (%d tokens): top-1 agrees at %d/%d, %d flips at mlx-vlm near-ties, "
               "%d decisive; top-10 logits within %.3f of scale; greedy matches %d/%d%s\n",
               prompt_i, n, agree, n, flips_near, flips_decisive, worst_rel, same, ng,
               same < ng ? "" : " (all)");
        if (same < ng)
            printf("    continuation splits at %d: the engine preferred its token by %.3f\n",
                   same, gap_at_split);
        CHECK(flips_decisive == 0, "prompt %d: %d decisive top-1 disagreements", prompt_i, flips_decisive);
        CHECK(worst_rel < 0.15f, "prompt %d: top-10 logits differ by %.3f of scale", prompt_i, worst_rel);
        CHECK(same == ng || gap_at_split < NEAR_TIE,
              "prompt %d: greedy diverges at %d where the engine was decisive (%.3f)",
              prompt_i, same, gap_at_split);
    }

    qwasar_engine_free(e);
    qj_free(&d);
    if (fails) { fprintf(stderr, "%d failure(s)\n", fails); return 1; }
    printf("flashnext real: all checks pass\n");
    return 0;
}
