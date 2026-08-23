/* Disk checkpoints.
 *
 * The property that matters is not that a restore is fast but that it is
 * indistinguishable: a session continued from disk must produce exactly what
 * the original would have produced next.  For this model that means the
 * recurrent conv and delta-rule state have to survive the round trip along with
 * the KV cache, and a mistake there would show up as a model that answers
 * differently after a restart -- which no amount of eyeballing would catch.
 *
 * Runs against a private HOME so it cannot disturb the real cache. */

#include "qwasar.h"
#include "qwasar_model.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int fails;

#define CHECK(cond, ...) do { \
    if (!(cond)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
                   fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); fails++; } \
} while (0)

static int32_t argmax(const float *v, int32_t n) {
    int32_t best = 0;
    for (int32_t i = 1; i < n; i++) if (v[i] > v[best]) best = i;
    return best;
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

static void rm_cache(const char *home) {
    char cmd[1024];
    snprintf(cmd, sizeof cmd, "rm -rf '%s/.cache/qwasar'", home);
    if (system(cmd) != 0) { /* best effort */ }
}

int main(int argc, char **argv) {
    const char *model = getenv("QWASAR_TEST_MODEL");
    if (argc > 1) model = argv[1];
    if (!model) { fprintf(stderr, "skip: set QWASAR_TEST_MODEL\n"); return 0; }

    char home[] = "/tmp/qwasar_kvtest_XXXXXX";
    if (!mkdtemp(home)) { fprintf(stderr, "cannot make a temp HOME\n"); return 1; }
    setenv("HOME", home, 1);

    char err[512];
    qwasar_options opts = { .model_path = model, .context_size = 2048 };
    qwasar_engine *e = qwasar_engine_load(&opts, err, sizeof err);
    if (!e) { fprintf(stderr, "load failed: %s\n", err); return 1; }

    const int32_t vocab = qwasar_vocab_size(e);

    /* Long enough to clear the store's minimum, and deliberately not a natural
     * sentence: the state must round-trip regardless of content. */
    const int32_t n = 300;
    int32_t *prompt = malloc((size_t)n * sizeof *prompt);
    for (int32_t i = 0; i < n; i++) prompt[i] = 1000 + (i * 7919) % 40000;
    const int32_t probe = 12345;      /* the token evaluated after the restore */

    /* Reference: one session, straight through. */
    qwasar_session *a = qwasar_session_new(e, err, sizeof err);
    if (!a) { fprintf(stderr, "session: %s\n", err); return 1; }
    if (!qwasar_session_eval(a, prompt, n, err, sizeof err)) {
        fprintf(stderr, "eval: %s\n", err);
        return 1;
    }

    /* Checkpoint at exactly the prompt, before the probe: the restored session
     * must be able to take the same next step, not merely reach the same place. */
    CHECK(qwasar_session_n_past(a) == n, "n_past %d, expected %d",
          qwasar_session_n_past(a), n);
    CHECK(qwasar_session_save(a, e, err, sizeof err), "save failed: %s", err);

    const float *la = qwasar_session_eval(a, &probe, 1, err, sizeof err);
    if (!la) { fprintf(stderr, "eval: %s\n", err); return 1; }
    float *ref = malloc((size_t)vocab * sizeof *ref);
    memcpy(ref, la, (size_t)vocab * sizeof *ref);
    const int32_t ref_argmax = argmax(ref, vocab);

    uint64_t bytes = 0;
    int entries = 0;
    qwasar_kv_cache_stats(&bytes, &entries);
    CHECK(entries == 1, "expected 1 cache entry, got %d", entries);
    printf("  checkpoint: %d tokens, %.0f MB on disk\n", n, (double)bytes / 1e6);

    /* Restored: a fresh session continues from disk. */
    qwasar_session *b = qwasar_session_new(e, err, sizeof err);
    int32_t covered = qwasar_session_restore(b, e, prompt, n);
    CHECK(covered == n, "restored %d tokens, expected %d", covered, n);

    /* The probe must agree with the restore it predicts: same scan, same
     * validation, no load.  And it must say 0 for tokens nothing covers --
     * a UI trusts this to claim "resumes from checkpoint" (spec 4.4). */
    CHECK(qwasar_kv_probe(e, prompt, n) == n,
          "probe disagrees with the restore it predicts");
    int32_t bogus[4] = { 9, 9, 9, 9 };
    CHECK(qwasar_kv_probe(e, bogus, 4) == 0, "probe matched tokens it should not");
    CHECK(qwasar_session_n_past(b) == n, "restored n_past %d", qwasar_session_n_past(b));

    const float *lb = qwasar_session_eval(b, &probe, 1, err, sizeof err);
    CHECK(lb != NULL, "eval after restore: %s", err);
    if (lb) {
        double d = rel_l2(lb, ref, (size_t)vocab);
        /* The restored buffers are byte copies and the graph is deterministic,
         * so this is not a tolerance question -- any drift means some part of
         * the state did not travel. */
        CHECK(d == 0.0, "logits differ after restore: rel l2 %.3g", d);
        CHECK(argmax(lb, vocab) == ref_argmax, "argmax differs after restore");
        printf("  continuation after restore: rel l2 %.1e (argmax %d)\n",
               d, argmax(lb, vocab));
    }
    qwasar_session_free(b);

    /* A longer prompt that begins with the checkpoint must reuse it. */
    int32_t *longer = malloc((size_t)(n + 50) * sizeof *longer);
    memcpy(longer, prompt, (size_t)n * sizeof *longer);
    for (int32_t i = 0; i < 50; i++) longer[n + i] = 2000 + i;
    qwasar_session *c = qwasar_session_new(e, err, sizeof err);
    CHECK(qwasar_session_restore(c, e, longer, n + 50) == n,
          "a checkpoint should match a prompt that extends it");
    qwasar_session_free(c);

    /* One differing token anywhere in the prefix must miss: the recurrent state
     * depends on the whole history, so a near-match is not a match. */
    int32_t *altered = malloc((size_t)n * sizeof *altered);
    memcpy(altered, prompt, (size_t)n * sizeof *altered);
    altered[n / 2] ^= 1;
    qwasar_session *d2 = qwasar_session_new(e, err, sizeof err);
    CHECK(qwasar_session_restore(d2, e, altered, n) == 0,
          "a prompt differing mid-prefix must not restore");
    qwasar_session_free(d2);

    /* A shorter prompt cannot use a longer checkpoint: there is nothing to
     * truncate a recurrent state down to. */
    qwasar_session *f = qwasar_session_new(e, err, sizeof err);
    CHECK(qwasar_session_restore(f, e, prompt, n - 10) == 0,
          "a checkpoint longer than the prompt must not restore");
    qwasar_session_free(f);

    /* A truncated file must be rejected rather than restored as garbage. */
    {
        char path[1024];
        snprintf(path, sizeof path, "%s/.cache/qwasar/kv", home);
        char cmd[1200];
        snprintf(cmd, sizeof cmd, "for f in '%s'/*.qwkv; do "
                                  "  dd if=\"$f\" of=\"$f.cut\" bs=1m count=2 2>/dev/null; "
                                  "  mv \"$f.cut\" \"$f\"; done", path);
        if (system(cmd) == 0) {
            qwasar_session *g = qwasar_session_new(e, err, sizeof err);
            CHECK(qwasar_session_restore(g, e, prompt, n) == 0,
                  "a truncated checkpoint must not restore");
            qwasar_session_free(g);
            printf("  truncated checkpoint rejected\n");
        }
    }

    free(prompt); free(longer); free(altered); free(ref);
    qwasar_session_free(a);
    qwasar_engine_free(e);
    rm_cache(home);
    rmdir(home);

    if (fails) { fprintf(stderr, "%d check(s) failed\n", fails); return 1; }
    printf("ok: kvstore\n");
    return 0;
}
