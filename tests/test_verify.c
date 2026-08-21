/* The gate this milestone turns on: speculative decoding must not change a
 * single emitted token.
 *
 * The check is exact rather than statistical, and that is the point.  A draft
 * head only proposes, so no bug in it can be caught by looking at the output --
 * a head that drafts badly produces the same tokens more slowly.  What CAN
 * change the output is the machinery around it: a recurrent state rewound to
 * the wrong boundary, a KV cursor left too far forward, a head history that
 * kept a rejected row.  All of those are silent, and all of them break this. */

#include "qwasar.h"
#include "qwasar_gpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;

#define CHECK(cond, ...) do { \
    if (!(cond)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
                   fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); fails++; } \
} while (0)

#define MAX_GEN 64

static int32_t argmax(const float *v, int32_t n) {
    int32_t best = 0;
    for (int32_t i = 1; i < n; i++) if (v[i] > v[best]) best = i;
    return best;
}

/* Plain greedy decoding, the trajectory everything else must reproduce. */
static int serial(qwasar_engine *e, const int32_t *prompt, int32_t n_prompt,
                  int32_t *out, int max_out) {
    char err[256];
    qwasar_session *s = qwasar_session_new(e, err, sizeof err);
    if (!s) { fprintf(stderr, "session: %s\n", err); return -1; }

    const float *logits = qwasar_session_eval(s, prompt, n_prompt, err, sizeof err);
    if (!logits) { fprintf(stderr, "eval: %s\n", err); qwasar_session_free(s); return -1; }

    const int32_t vocab = qwasar_vocab_size(e);
    int n = 0;
    while (n < max_out) {
        int32_t next = argmax(logits, vocab);
        if (qwasar_is_eos(e, next)) break;
        out[n++] = next;
        logits = qwasar_session_eval(s, &next, 1, err, sizeof err);
        if (!logits) { fprintf(stderr, "eval: %s\n", err); qwasar_session_free(s); return -1; }
    }
    qwasar_session_free(s);
    return n;
}

/* The same thing through the draft head: propose a block, verify it in one
 * pass, keep the longest correct prefix. */
static int speculative(qwasar_engine *e, const int32_t *prompt, int32_t n_prompt,
                       int32_t depth, int32_t *out, int max_out,
                       int64_t *rounds, int64_t *committed, int64_t *rejected) {
    char err[256];
    qwasar_session *s = qwasar_session_new(e, err, sizeof err);
    if (!s) { fprintf(stderr, "session: %s\n", err); return -1; }
    if (!qwasar_session_has_mtp(s)) { qwasar_session_free(s); return -2; }

    const float *logits = qwasar_session_eval(s, prompt, n_prompt, err, sizeof err);
    if (!logits) { fprintf(stderr, "eval: %s\n", err); qwasar_session_free(s); return -1; }

    const int32_t vocab = qwasar_vocab_size(e);
    int32_t next = argmax(logits, vocab);
    int n = 0;

    while (n < max_out) {
        if (qwasar_is_eos(e, next)) break;
        out[n++] = next;

        int32_t blk[1 + QWASAR_MAX_DRAFT], got[1 + QWASAR_MAX_DRAFT];
        blk[0] = next;
        int32_t nd = qwasar_session_draft(s, next, blk + 1, depth, err, sizeof err);
        if (nd < 0) { fprintf(stderr, "draft: %s\n", err); qwasar_session_free(s); return -1; }

        int32_t nc = qwasar_session_verify(s, blk, nd + 1, got, err, sizeof err);
        if (nc < 0) { fprintf(stderr, "verify: %s\n", err); qwasar_session_free(s); return -1; }
        (*rounds)++;
        *committed += nc;
        if (nc < nd + 1) (*rejected)++;   /* a boundary was rewound to */

        /* All but the last committed token are settled; the last becomes the
         * next round's starting point, exactly as a decode step's would. */
        for (int32_t i = 0; i + 1 < nc && n < max_out; i++) {
            if (qwasar_is_eos(e, got[i])) { qwasar_session_free(s); return n; }
            out[n++] = got[i];
        }
        next = got[nc - 1];
    }
    qwasar_session_free(s);
    return n;
}

int main(int argc, char **argv) {
    const char *model = getenv("QWASAR_TEST_MODEL");
    const char *head  = getenv("QWASAR_TEST_MTP");
    if (argc > 1) model = argv[1];
    if (argc > 2) head  = argv[2];
    if (!model || !*model || !head || !*head) {
        fprintf(stderr, "skip: set QWASAR_TEST_MODEL and QWASAR_TEST_MTP\n");
        return 0;
    }
    { char probe[1200]; snprintf(probe, sizeof probe, "%s/model.safetensors", head);
      FILE *f = fopen(probe, "rb");
      if (!f) { fprintf(stderr, "skip: no MTP head at %s\n", head); return 0; }
      fclose(f); }

    char err[512];
    qwasar_options opts = { .model_path = model, .mtp_path = head };
    qwasar_engine *e = qwasar_engine_load(&opts, err, sizeof err);
    if (!e) { fprintf(stderr, "load failed: %s\n", err); return 1; }

    qwasar_tokenizer *tok = qwasar_tokenizer_load(model, err, sizeof err);
    if (!tok) { fprintf(stderr, "tokenizer: %s\n", err); return 1; }

    /* Two prompts on purpose.  A list of primes is nearly all predictable and
     * exercises the accept path; open prose is where the head is wrong often
     * enough to exercise the rewind, which is the half that can corrupt a
     * session. */
    static const char *const prompts[] = {
        "List the first eight prime numbers.",
        "Describe, in ordinary prose, what makes a good error message.",
    };
    qwasar_chat_options chat = { .enable_thinking = false, .reasoning_effort = "low",
                                 .add_generation_prompt = true };
    int64_t total_rejected = 0;

    for (size_t pi = 0; pi < sizeof prompts / sizeof *prompts; pi++) {
        qwasar_message msg = { "user", prompts[pi], NULL, NULL };
        int32_t n_prompt = 0;
        int32_t *prompt = qwasar_apply_chat_template(tok, &msg, 1, &chat, &n_prompt,
                                                     err, sizeof err);
        if (!prompt) { fprintf(stderr, "template: %s\n", err); return 1; }

        int32_t want[MAX_GEN], got[MAX_GEN];
        int n_want = serial(e, prompt, n_prompt, want, MAX_GEN);
        CHECK(n_want > 8, "serial decoding produced only %d tokens", n_want);
        if (n_want < 0) return 1;

        /* Every depth, because each exercises a different rewind: depth 1 can
         * only accept or reject outright, deeper ones have interior boundaries
         * to land on. */
        for (int32_t depth = 1; depth <= 4; depth++) {
            int64_t rounds = 0, committed = 0, rejected = 0;
            int n_got = speculative(e, prompt, n_prompt, depth, got, MAX_GEN,
                                    &rounds, &committed, &rejected);
            if (n_got == -2) { fprintf(stderr, "skip: no MTP head bound\n"); return 0; }
            if (n_got < 0) return 1;
            total_rejected += rejected;

            CHECK(n_got == n_want, "prompt %zu depth %d produced %d tokens, "
                  "serial produced %d", pi, depth, n_got, n_want);
            int first_bad = -1;
            for (int i = 0; i < (n_got < n_want ? n_got : n_want); i++)
                if (got[i] != want[i]) { first_bad = i; break; }
            CHECK(first_bad < 0, "prompt %zu depth %d diverges at token %d: %d, "
                  "serial says %d", pi, depth, first_bad,
                  first_bad >= 0 ? got[first_bad] : 0,
                  first_bad >= 0 ? want[first_bad] : 0);

            printf("  prompt %zu depth %d: %d tokens identical, %lld rounds, "
                   "%.2f per round, %lld rewound\n", pi, depth, n_got,
                   (long long)rounds, rounds ? (double)committed / (double)rounds : 0.0,
                   (long long)rejected);
        }
        free(prompt);
    }

    /* Without a rejection somewhere, every round accepted everything and the
     * rewind never ran -- which would make the identity above prove nothing
     * about the half of this that can corrupt a session. */
    CHECK(total_rejected > 0,
          "no round was ever rewound, so this test did not exercise the rewind");
    qwasar_tokenizer_free(tok);
    qwasar_engine_free(e);
    qw_gpu_shutdown();

    if (fails) { fprintf(stderr, "%d check(s) failed\n", fails); return 1; }
    printf("ok: verify\n");
    return 0;
}
