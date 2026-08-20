/* The whole model, against the reference implementation.
 *
 * Per-op tests prove each kernel matches its CPU twin, but they cannot catch a
 * misreading of the architecture: a swapped gate, a norm applied to the wrong
 * operand, layers wired in the wrong order.  This replays golden activations
 * captured from mlx-vlm (tools/dump_golden.py) through the C engine.
 *
 * The comparison walks forward through the network so the first divergence is
 * reported at the layer where it happens.  In a hybrid model that matters: the
 * gated-delta and full-attention layers are entirely different code, and
 * knowing which one drifted is most of the debugging. */

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

typedef struct {
    int32_t  n_tokens, hidden, vocab, n_capture;
    int32_t *tokens;
    int32_t *capture;
    float  **hidden_states;   /* [n_capture][hidden] */
    float   *final_norm;      /* [hidden] */
    float   *logits;          /* [vocab] */
} golden;

static bool golden_load(golden *g, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    char magic[8];
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "QWGOLD01", 8)) { fclose(f); return false; }
    if (fread(&g->n_tokens, 4, 4, f) != 4) { fclose(f); return false; }

    g->tokens  = malloc((size_t)g->n_tokens * 4);
    g->capture = malloc((size_t)g->n_capture * 4);
    fread(g->tokens, 4, (size_t)g->n_tokens, f);
    fread(g->capture, 4, (size_t)g->n_capture, f);

    g->hidden_states = malloc((size_t)g->n_capture * sizeof(float *));
    for (int32_t i = 0; i < g->n_capture; i++) {
        g->hidden_states[i] = malloc((size_t)g->hidden * 4);
        fread(g->hidden_states[i], 4, (size_t)g->hidden, f);
    }
    g->final_norm = malloc((size_t)g->hidden * 4);
    fread(g->final_norm, 4, (size_t)g->hidden, f);
    g->logits = malloc((size_t)g->vocab * 4);
    size_t got = fread(g->logits, 4, (size_t)g->vocab, f);
    fclose(f);
    return got == (size_t)g->vocab;
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

static void top_k(const float *v, int32_t n, int k, int32_t *idx) {
    for (int i = 0; i < k; i++) {
        int32_t best = -1;
        for (int32_t j = 0; j < n; j++) {
            bool taken = false;
            for (int t = 0; t < i; t++) if (idx[t] == j) { taken = true; break; }
            if (taken) continue;
            if (best < 0 || v[j] > v[best]) best = j;
        }
        idx[i] = best;
    }
}

int main(int argc, char **argv) {
    const char *model  = getenv("QWASAR_TEST_MODEL");
    const char *gpath  = getenv("QWASAR_GOLDEN");
    if (argc > 1) model = argv[1];
    if (argc > 2) gpath = argv[2];
    if (!gpath) gpath = "tests/golden.bin";
    if (!model) { fprintf(stderr, "skip: set QWASAR_TEST_MODEL\n"); return 0; }

    golden g;
    memset(&g, 0, sizeof g);
    if (!golden_load(&g, gpath)) {
        fprintf(stderr, "skip: no golden vectors at %s\n"
                        "      generate with: reference/mlx-vlm/.venv/bin/python \\\n"
                        "        tools/dump_golden.py <model-dir> %s\n", gpath, gpath);
        return 0;
    }

    char err[512];
    qwasar_options opts = { .model_path = model, .context_size = 4096 };
    qwasar_engine *e = qwasar_engine_load(&opts, err, sizeof err);
    if (!e) { fprintf(stderr, "load failed: %s\n", err); return 1; }

    CHECK(qwasar_vocab_size(e) == g.vocab, "vocab %d vs golden %d",
          qwasar_vocab_size(e), g.vocab);

    qwasar_session *s = qwasar_session_new(e, err, sizeof err);
    if (!s) { fprintf(stderr, "session failed: %s\n", err); return 1; }

    CHECK(qwasar_session_set_capture(s, g.capture, g.n_capture), "cannot set capture");

    printf("  prompt: %d tokens\n", g.n_tokens);
    const float *logits = qwasar_session_eval(s, g.tokens, g.n_tokens, err, sizeof err);
    if (!logits) { fprintf(stderr, "eval failed: %s\n", err); return 1; }

    /* Walk the drift forward.  Smooth growth is bf16-versus-fp32 accumulation;
     * a jump at one layer is a bug in that layer's kind. */
    printf("  %-8s %-14s %s\n", "layer", "kind", "rel l2 vs reference");
    for (int32_t i = 0; i < g.n_capture; i++) {
        const float *got = qwasar_session_captured(s, i);
        CHECK(got != NULL, "no capture for layer %d", g.capture[i]);
        if (!got) continue;
        double d = rel_l2(got, g.hidden_states[i], (size_t)g.hidden);
        const char *kind = ((g.capture[i] + 1) % 4 == 0) ? "attention" : "gated-delta";
        printf("  %-8d %-14s %.3e\n", g.capture[i], kind, d);
        /* Layer 0 has had one layer's worth of rounding; anything large there
         * is structural, not accumulated. */
        if (i == 0) CHECK(d < 5e-3, "layer %d already diverged: %.3g", g.capture[i], d);
    }

    double e_logits = rel_l2(logits, g.logits, (size_t)g.vocab);
    /* Threshold set from the reference's own noise floor, not from hope: run
     * the same prompt through mlx-vlm batched versus one token at a time and
     * its logits differ by 7.4e-2, because its activations are bf16 and the
     * accumulation order changes.  Anything at or below that is indistinguishable
     * from a rounding-order difference, so logit L2 is a weak signal here and
     * the argmax and top-5 order below are the checks that carry weight. */
    CHECK(e_logits < 8e-2, "logits rel l2 %.4g", e_logits);
    printf("  logits rel l2: %.3e\n", e_logits);

    int32_t got[5], want[5];
    top_k(logits, g.vocab, 5, got);
    top_k(g.logits, g.vocab, 5, want);
    printf("  top5 qwasar   :");
    for (int i = 0; i < 5; i++) printf(" %d(%.2f)", got[i], (double)logits[got[i]]);
    printf("\n  top5 reference:");
    for (int i = 0; i < 5; i++) printf(" %d(%.2f)", want[i], (double)g.logits[want[i]]);
    printf("\n");

    /* The argmax is what generation actually consumes; it must agree exactly. */
    CHECK(got[0] == want[0], "argmax %d, reference says %d", got[0], want[0]);
    for (int i = 0; i < 5; i++)
        CHECK(got[i] == want[i], "top5 rank %d: %d, reference says %d", i, got[i], want[i]);

    qwasar_session_free(s);
    qwasar_engine_free(e);
    qw_gpu_shutdown();

    if (fails) { fprintf(stderr, "%d check(s) failed\n", fails); return 1; }
    printf("ok: forward\n");
    return 0;
}
