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
#include <sys/time.h>
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

/* The in-memory rewind point (qwasar_session_mark / _rewind), as the server
 * uses it: a prompt, a rewind point one token short of its end, a reply;
 * then a next prompt that keeps the first but not the reply.  Rewinding and
 * evaluating the rest must be exactly a fresh session evaluating the same
 * tokens in the same chunks -- and so must a checkpoint written at the
 * rewind point on the way out (qwasar_session_rewind_to_mark), restored. */
static void check_rewind(qwasar_engine *e, int32_t vocab) {
    char err[512];
    const int32_t np = 300, ng = 20, nx = 30, nq = np + nx;
    int32_t *q = malloc((size_t)nq * sizeof *q), *gen = malloc((size_t)ng * sizeof *gen);
    for (int32_t i = 0; i < nq; i++) q[i] = (int32_t)((3000 + (i * 7907) % 50000) % vocab);
    for (int32_t i = 0; i < ng; i++) gen[i] = (int32_t)((500 + 37 * i) % vocab);
    float *ref = malloc((size_t)vocab * sizeof *ref);

    /* The reference: the next prompt from scratch, chunked as after a rewind. */
    qwasar_session *b = qwasar_session_new(e, err, sizeof err);
    const float *lb = b ? qwasar_session_eval(b, q, np - 1, err, sizeof err) : NULL;
    if (lb) lb = qwasar_session_eval(b, q + np - 1, nq - (np - 1), err, sizeof err);
    CHECK(lb != NULL, "reference eval: %s", err);
    if (lb) memcpy(ref, lb, (size_t)vocab * sizeof *ref);

    qwasar_session *a = qwasar_session_new(e, err, sizeof err);
    bool ok = a && qwasar_session_eval(a, q, np - 1, err, sizeof err);
    CHECK(ok && qwasar_session_mark(a), "mark: %s", err);
    ok = ok && qwasar_session_eval(a, q + np - 1, 1, err, sizeof err);
    for (int32_t i = 0; ok && i < ng; i++) ok = qwasar_session_eval(a, gen + i, 1, err, sizeof err);
    CHECK(ok, "prompt and reply: %s", err);

    /* Refused: a prompt differing inside the marked part, or not past it. */
    int32_t *bad = malloc((size_t)nq * sizeof *bad);
    memcpy(bad, q, (size_t)nq * sizeof *bad);
    bad[np / 2] = (bad[np / 2] + 1) % vocab;
    CHECK(qwasar_session_rewind(a, bad, nq) == 0, "rewound for a prompt that differs before the mark");
    CHECK(qwasar_session_rewind(a, q, np - 1) == 0, "rewound for a prompt no longer than the mark");
    CHECK(qwasar_session_n_past(a) == np + ng, "a refused rewind changed the session");

    CHECK(qwasar_session_rewind(a, q, nq) == np - 1, "rewind did not return to the mark");
    const float *la = qwasar_session_eval(a, q + np - 1, nq - (np - 1), err, sizeof err);
    CHECK(la != NULL, "eval after rewind: %s", err);
    if (la && lb) {
        const double d = rel_l2(la, ref, (size_t)vocab);
        CHECK(d == 0.0, "after a rewind the logits differ: rel l2 %.3g", d);
        printf("  after a rewind past a 20-token reply: rel l2 %.1e\n", d);
    }

    /* On the way out: back to the mark, saved, restored by a new session. */
    for (int32_t i = 0; ok && i < ng; i++) ok = qwasar_session_eval(a, gen + i, 1, err, sizeof err);
    CHECK(qwasar_session_rewind_to_mark(a) == np - 1, "rewind_to_mark");
    CHECK(qwasar_session_save(a, e, err, sizeof err), "save at the mark: %s", err);
    qwasar_session *c = qwasar_session_new(e, err, sizeof err);
    CHECK(c && qwasar_session_restore(c, e, q, nq) == np - 1, "the saved rewind point did not restore");
    const float *lc = c ? qwasar_session_eval(c, q + np - 1, nq - (np - 1), err, sizeof err) : NULL;
    if (lc && lb) {
        const double d = rel_l2(lc, ref, (size_t)vocab);
        CHECK(d == 0.0, "restored from the rewind point, the logits differ: rel l2 %.3g", d);
        printf("  saved at the rewind point and restored: rel l2 %.1e\n", d);
    }

    if (a) qwasar_session_free(a);
    if (b) qwasar_session_free(b);
    if (c) qwasar_session_free(c);
    free(q); free(gen); free(bad); free(ref);
}

/* Explicit-path checkpoints (qwasar_session_save_file / _restore_file /
 * qwasar_kv_probe_file), as the server's store parks a session: a file the
 * caller names, in no cache, with no floor and no eviction.  Held to the same
 * bar as the cache: a restored session continues bit-identically, and
 * anything that is not exactly a prefix of this session is refused. */
static void check_files(qwasar_engine *e, int32_t vocab, const char *home) {
#define TOK(x) ((int32_t)((x) % vocab))
    char err[512], path[1024];
    snprintf(path, sizeof path, "%s/parked.qwkv", home);
    uint64_t cache_before = 0;
    int entries_before = 0;
    qwasar_kv_cache_stats(&cache_before, &entries_before);

    const int32_t n = 300, probe = TOK(777);
    int32_t *tok = malloc((size_t)(n + 1) * sizeof *tok);
    for (int32_t i = 0; i < n; i++) tok[i] = TOK(3000 + (i * 104729) % 50000);
    tok[n] = probe;

    qwasar_session *a = qwasar_session_new(e, err, sizeof err);
    CHECK(a && qwasar_session_eval(a, tok, n, err, sizeof err), "eval: %s", err);
    CHECK(qwasar_session_save_file(a, e, path, err, sizeof err), "save_file: %s", err);
    const float *la = qwasar_session_eval(a, &probe, 1, err, sizeof err);
    float *ref = malloc((size_t)vocab * sizeof *ref);
    if (la) memcpy(ref, la, (size_t)vocab * sizeof *ref);

    uint64_t cache_after = 0;
    int entries_after = 0;
    qwasar_kv_cache_stats(&cache_after, &entries_after);
    CHECK(entries_after == entries_before && cache_after == cache_before,
          "save_file touched the cache (%d -> %d entries)", entries_before, entries_after);
    struct stat st;
    CHECK(stat(path, &st) == 0, "no file at %s", path);

    /* The probe predicts the restore; a restored session continues exactly. */
    CHECK(qwasar_kv_probe_file(e, path, tok, n + 1) == n, "probe_file did not cover the prefix");
    qwasar_session *b = qwasar_session_new(e, err, sizeof err);
    CHECK(qwasar_session_restore_file(b, e, path, tok, n + 1) == n, "restore_file did not restore %d", n);
    CHECK(qwasar_session_n_past(b) == n, "restored n_past %d", qwasar_session_n_past(b));
    const float *lb = qwasar_session_eval(b, &probe, 1, err, sizeof err);
    CHECK(lb != NULL, "eval after restore_file: %s", err);
    if (lb && la) {
        const double d = rel_l2(lb, ref, (size_t)vocab);
        CHECK(d == 0.0, "logits differ after restore_file: rel l2 %.3g", d);
        printf("  explicit-path checkpoint: %.0f MB, continuation rel l2 %.1e\n",
               (double)st.st_size / 1e6, d);
    }

    /* Refusals: a different prefix, a shorter history, a session that is not
     * fresh, a missing file, a truncated file. */
    int32_t *alt = malloc((size_t)n * sizeof *alt);
    memcpy(alt, tok, (size_t)n * sizeof *alt);
    alt[n - 1] = TOK(alt[n - 1] + 1);
    qwasar_session *c = qwasar_session_new(e, err, sizeof err);
    CHECK(qwasar_session_restore_file(c, e, path, alt, n) == 0, "restored a different prefix");
    CHECK(qwasar_kv_probe_file(e, path, alt, n) == 0, "probed a different prefix");
    CHECK(qwasar_session_restore_file(c, e, path, tok, n - 1) == 0, "restored into a shorter history");
    CHECK(qwasar_session_n_past(c) == 0, "a refused restore touched the session");
    CHECK(qwasar_session_restore_file(b, e, path, tok, n + 1) == 0, "restored into a session that is not fresh");

    /* A reset session is a fresh one that kept its memory: it takes the
     * restore and continues exactly -- and, evaluating from nothing, it
     * matches a new session too, so no state survived the reset. */
    CHECK(qwasar_session_reset(b, err, sizeof err), "reset: %s", err);
    CHECK(qwasar_session_n_past(b) == 0, "reset left n_past %d", qwasar_session_n_past(b));
    CHECK(qwasar_session_restore_file(b, e, path, tok, n + 1) == n, "restore_file into a reset session");
    lb = qwasar_session_eval(b, &probe, 1, err, sizeof err);
    CHECK(lb && la && rel_l2(lb, ref, (size_t)vocab) == 0.0, "logits differ after a reset and restore");
    CHECK(qwasar_session_reset(a, err, sizeof err), "reset: %s", err);
    CHECK(qwasar_session_eval(a, tok, n, err, sizeof err) != NULL, "eval after reset: %s", err);
    la = qwasar_session_eval(a, &probe, 1, err, sizeof err);
    CHECK(la && rel_l2(la, ref, (size_t)vocab) == 0.0, "logits differ after a reset and prefill");
    if (la) printf("  reset session: restore and prefill both match\n");

    char missing[1100];
    snprintf(missing, sizeof missing, "%s/nothing-here.qwkv", home);
    CHECK(qwasar_session_restore_file(c, e, missing, tok, n) == 0, "restored a missing file");
    if (truncate(path, (off_t)(st.st_size / 2)) == 0)
        CHECK(qwasar_session_restore_file(c, e, path, tok, n) == 0, "restored a truncated file");

    /* No floor: a session far under the cache's 256-token minimum parks. */
    const int32_t m = 40;
    qwasar_session *s = qwasar_session_new(e, err, sizeof err);
    CHECK(s && qwasar_session_eval(s, tok, m, err, sizeof err), "eval short: %s", err);
    CHECK(qwasar_session_save_file(s, e, path, err, sizeof err), "save_file short: %s", err);
    qwasar_session *r = qwasar_session_new(e, err, sizeof err);
    CHECK(qwasar_session_restore_file(r, e, path, tok, n) == m, "short restore did not cover %d", m);

    unlink(path);
    qwasar_session_free(a); qwasar_session_free(b); qwasar_session_free(c);
    qwasar_session_free(s); qwasar_session_free(r);
    free(tok); free(alt); free(ref);
#undef TOK
}

/* The whole round trip against one model, in a private HOME. */
static int check_model(const char *model) {
    printf("== %s\n", model);
    char home[] = "/tmp/qwasar_kvtest_XXXXXX";
    if (!mkdtemp(home)) { fprintf(stderr, "cannot make a temp HOME\n"); return 1; }
    setenv("HOME", home, 1);

    char err[512];
    qwasar_options opts = { .model_path = model, .context_size = 2048 };
    qwasar_engine *e = qwasar_engine_load(&opts, err, sizeof err);
    if (!e) { fprintf(stderr, "load failed: %s\n", err); return 1; }

    const int32_t vocab = qwasar_vocab_size(e);
    /* Token ids wrap to the vocabulary: the Flash-Next toys have 256. */
#define TOK(x) ((int32_t)((x) % vocab))

    /* Long enough to clear the store's minimum, and deliberately not a natural
     * sentence: the state must round-trip regardless of content. */
    const int32_t n = 300;
    int32_t *prompt = malloc((size_t)n * sizeof *prompt);
    for (int32_t i = 0; i < n; i++) prompt[i] = TOK(1000 + (i * 7919) % 40000);
    const int32_t probe = TOK(12345); /* the token evaluated after the restore */

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
    for (int32_t i = 0; i < 50; i++) longer[n + i] = TOK(2000 + i);
    qwasar_session *c = qwasar_session_new(e, err, sizeof err);
    CHECK(qwasar_session_restore(c, e, longer, n + 50) == n,
          "a checkpoint should match a prompt that extends it");
    qwasar_session_free(c);

    /* One differing token anywhere in the prefix must miss: the recurrent state
     * depends on the whole history, so a near-match is not a match. */
    int32_t *altered = malloc((size_t)n * sizeof *altered);
    memcpy(altered, prompt, (size_t)n * sizeof *altered);
    altered[n / 2] = TOK(altered[n / 2] + 1);
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

    check_rewind(e, vocab);
    check_files(e, vocab, home);

    free(prompt); free(longer); free(altered); free(ref);
    qwasar_session_free(a);
    qwasar_engine_free(e);
    rm_cache(home);
    rmdir(home);

#undef TOK
    return 0;
}

/* ---- timing a step after a restore (opt-in) ---------------------------------
 *
 * QWASAR_TEST_RESTORE_TIMING=<tokens> with QWASAR_TEST_MODEL: times a short
 * step on a fresh session, on that session once it holds <tokens>, on a
 * new session restored from its checkpoint, and on a reset one (the parked
 * session's memory kept) restored from it -- the first step and the one
 * after it each time.  Prints the numbers; asserts only that the restore
 * covered the tokens.  The context is the server's, 262144, since what a
 * session allocates scales with it. */

static double secs(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + tv.tv_usec * 1e-6;
}

static double timed_step(qwasar_session *s, int32_t base, int32_t n, int32_t vocab) {
    int32_t tok[64];
    for (int32_t i = 0; i < n; i++) tok[i] = (base + i * 7919) % vocab;
    char err[256];
    const double t0 = secs();
    const float *l = qwasar_session_eval(s, tok, n, err, sizeof err);
    const double dt = secs() - t0;
    CHECK(l != NULL, "step: %s", err);
    return dt;
}

static void time_restore(const char *model, int32_t n) {
    printf("== restore timing: %s, %d tokens\n", model, n);
    char home[] = "/tmp/qwasar_kvtime_XXXXXX";
    if (!mkdtemp(home)) return;
    setenv("HOME", home, 1);
    char err[512], path[1024];
    snprintf(path, sizeof path, "%s/timed.qwkv", home);

    qwasar_options opts = { .model_path = model, .context_size = 262144 };
    qwasar_engine *e = qwasar_engine_load(&opts, err, sizeof err);
    if (!e) { fprintf(stderr, "load failed: %s\n", err); fails++; return; }
    const int32_t vocab = qwasar_vocab_size(e);
    const int32_t step = 19;

    double t0 = secs();
    qwasar_session *a = qwasar_session_new(e, err, sizeof err);
    printf("  new session: %.2fs\n", secs() - t0);
    printf("  fresh session, first step:  %.3fs\n", timed_step(a, 1000, step, vocab));
    printf("  fresh session, second step: %.3fs\n", timed_step(a, 2000, step, vocab));

    int32_t *tok = malloc((size_t)n * sizeof *tok);
    for (int32_t i = 0; i < n; i++) tok[i] = (3000 + (i * 104729)) % vocab;
    t0 = secs();
    CHECK(qwasar_session_eval(a, tok, n, err, sizeof err) != NULL, "prefill: %s", err);
    printf("  prefill %d tokens: %.1fs\n", n, secs() - t0);
    printf("  at %d tokens, a step:       %.3fs\n", qwasar_session_n_past(a), timed_step(a, 4000, step, vocab));
    printf("  at %d tokens, another step: %.3fs\n", qwasar_session_n_past(a), timed_step(a, 5000, step, vocab));

    int32_t hn = 0;
    const int32_t *hist = qw_session_history(a, &hn);
    int32_t *keep = malloc((size_t)hn * sizeof *keep);
    memcpy(keep, hist, (size_t)hn * sizeof *keep);
    t0 = secs();
    CHECK(qwasar_session_save_file(a, e, path, err, sizeof err), "save_file: %s", err);
    struct stat st;
    stat(path, &st);
    printf("  save_file: %.2fs, %.2f GB\n", secs() - t0, st.st_size / 1e9);
    qwasar_session_free(a);                      /* as a park does */

    t0 = secs();
    qwasar_session *b = qwasar_session_new(e, err, sizeof err);
    printf("  new session: %.2fs\n", secs() - t0);
    t0 = secs();
    const int32_t got = qwasar_session_restore_file(b, e, path, keep, hn);
    printf("  restore_file: %.2fs (%d tokens)\n", secs() - t0, got);
    CHECK(got == hn, "restored %d of %d", got, hn);
    printf("  restored, first step:       %.3fs\n", timed_step(b, 6000, step, vocab));
    printf("  restored, second step:      %.3fs\n", timed_step(b, 7000, step, vocab));

    t0 = secs();
    CHECK(qwasar_session_reset(b, err, sizeof err), "reset: %s", err);
    printf("  reset session: %.2fs\n", secs() - t0);
    t0 = secs();
    const int32_t again = qwasar_session_restore_file(b, e, path, keep, hn);
    printf("  restore_file into it: %.2fs (%d tokens)\n", secs() - t0, again);
    CHECK(again == hn, "restored %d of %d into a reset session", again, hn);
    printf("  reset+restored, first step:  %.3fs\n", timed_step(b, 6000, step, vocab));
    printf("  reset+restored, second step: %.3fs\n", timed_step(b, 7000, step, vocab));

    qwasar_session_free(b);
    unlink(path);
    rmdir(home);
    free(tok); free(keep);
    qwasar_engine_free(e);
}

/* The dense model when one is given; Flash-Next's toys always, in both
 * formats -- its checkpoint carries the indexer keys and the engram's state
 * too, and the toys' 8-token QSA budget puts a 300-token prompt well past
 * the point where the indexer's keys decide what attention sees. */
int main(int argc, char **argv) {
    const char *model = getenv("QWASAR_TEST_MODEL");
    if (argc > 1) model = argv[1];
    const char *timing = getenv("QWASAR_TEST_RESTORE_TIMING");
    if (timing && *timing && model && *model) {
        time_restore(model, atoi(timing));
        if (fails) { fprintf(stderr, "%d check(s) failed\n", fails); return 1; }
        return 0;
    }
    if (model && *model) { if (check_model(model)) return 1; }
    else printf("skip: dense model (set QWASAR_TEST_MODEL)\n");
    const char *toys[] = { "tests/fixtures/flashnext-tiny-q4", "tests/fixtures/flashnext-tiny-mlx" };
    for (size_t i = 0; i < sizeof toys / sizeof *toys; i++)
        if (check_model(toys[i])) return 1;

    if (fails) { fprintf(stderr, "%d check(s) failed\n", fails); return 1; }
    printf("ok: kvstore\n");
    return 0;
}
