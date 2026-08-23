/* Validates qw_op_argmax_top2 against the serial host scan it replaced.
 *
 * The two functions below are that scan, verbatim from qwasar_graph.c before
 * the selection moved to the GPU.  Keeping the original here is the point of
 * the test: the kernel is only a speedup if it returns what this returned, for
 * every input, and the claim is exactness rather than closeness.  Both outputs
 * are selections of existing floats -- no arithmetic -- so the comparison is
 * bit-for-bit and any tolerance would be hiding something.
 *
 * The case that decides it is a tie.  A parallel reduction is free to combine
 * equal maxima in any order, and the serial scan's strict `>` kept the lowest
 * index, so a merge that is careless about ties would pick a different token
 * without ever being wrong about the value.  Those cases are constructed
 * deliberately below, and `check_tie_rule_is_load_bearing` confirms they would
 * actually catch it. */

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

/* ---- the host scan this kernel replaced ---------------------------------- */

static int32_t ref_argmax(const float *v, int32_t n) {
    int32_t best = 0;
    for (int32_t i = 1; i < n; i++) if (v[i] > v[best]) best = i;
    return best;
}

static int32_t ref_top2(const float *v, int32_t n, double *margin) {
    int32_t best = 0;
    float b = v[0], second = -3.0e38f;
    for (int32_t i = 1; i < n; i++) {
        if (v[i] > b) { second = b; b = v[i]; best = i; }
        else if (v[i] > second) second = v[i];
    }
    *margin = (double)b - (double)second;
    return best;
}

/* ---- harness ------------------------------------------------------------- */

static void fill_random(float *v, size_t n, uint32_t seed) {
    uint32_t s = seed | 1u;
    for (size_t i = 0; i < n; i++) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        v[i] = (float)((int32_t)(s >> 8) - 8388608) / 8388608.0f;
    }
}

/* Runs the kernel over `rows` x `n` and compares every row with the scan.
 *
 * `publish` selects whether the winning ids are also written out as int32
 * tokens, which is what lets a draft chain stay on the device.  Both settings
 * are exercised: with it on the tokens must equal the indices, with it off the
 * destination must be left exactly as it was. */
static void run_case_ex(const char *what, const float *logits, int32_t n, int32_t rows,
                        int publish) {
    qw_buf lg  = qw_buf_alloc((size_t)rows * n * 4);
    qw_buf out = qw_buf_alloc((size_t)rows * sizeof(qw_cand));
    qw_buf scr = qw_buf_alloc((size_t)rows * QW_SEL_TILES * sizeof(qw_cand));
    qw_buf tok = qw_buf_alloc((size_t)rows * sizeof(int32_t));
    if (!lg || !out || !scr || !tok) { CHECK(0, "%s: out of memory", what); return; }
    memcpy(qw_buf_contents(lg), logits, (size_t)rows * n * 4);

    const int32_t sentinel = -12345;
    int32_t *tv = qw_buf_contents(tok);
    for (int32_t r = 0; r < rows; r++) tv[r] = sentinel;

    qw_cmd c = qw_cmd_begin();
    if (!c) { CHECK(0, "%s: no command buffer", what); return; }
    qw_op_argmax_top2(c, qw_ref_at(out, 0), qw_ref_at(scr, 0), qw_ref_at(lg, 0), n, rows,
                      publish ? qw_ref_at(tok, 0) : qw_ref_at(NULL, 0), 0, 0);
    qw_cmd_wait(c);
    const char *cerr = qw_cmd_error(c);
    CHECK(!cerr, "%s: GPU error: %s", what, cerr ? cerr : "");
    qw_cmd_free(c);

    const qw_cand *got = qw_buf_contents(out);
    for (int32_t r = 0; r < rows; r++) {
        const float *v = logits + (size_t)r * n;
        double want_margin;
        const int32_t want_i = ref_argmax(v, n);
        const int32_t want_t = ref_top2(v, n, &want_margin);
        const double  got_margin = (double)got[r].best - (double)got[r].second;

        CHECK(want_i == want_t, "%s row %d: the two references disagree (%d vs %d)",
              what, r, want_i, want_t);
        CHECK((int32_t)got[r].index == want_i,
              "%s row %d: index %u, scan says %d", what, r, got[r].index, want_i);
        CHECK(got[r].best == v[want_i],
              "%s row %d: best %.9g, row holds %.9g at the argmax",
              what, r, (double)got[r].best, (double)v[want_i]);
        CHECK(got_margin == want_margin,
              "%s row %d: margin %.9g, scan says %.9g", what, r, got_margin, want_margin);
        if (publish)
            CHECK(tv[r] == want_i, "%s row %d: published token %d, scan says %d",
                  what, r, tv[r], want_i);
        else
            CHECK(tv[r] == sentinel,
                  "%s row %d: wrote token %d with publishing off", what, r, tv[r]);
    }

    qw_buf_free(lg); qw_buf_free(out); qw_buf_free(scr); qw_buf_free(tok);
}

static void run_case(const char *what, const float *logits, int32_t n, int32_t rows) {
    run_case_ex(what, logits, n, rows, 1);
}

/* Scores `n` compacted rows and checks the winner comes back as a TOKEN id
 * under the prefix/tail mapping, not as the row it was found at. */
static void run_mapped(const char *what, const float *logits, int32_t n,
                       int32_t prefix, int32_t tail_base, int32_t want_token) {
    qw_buf lg  = qw_buf_alloc((size_t)n * 4);
    qw_buf out = qw_buf_alloc(sizeof(qw_cand));
    qw_buf scr = qw_buf_alloc((size_t)QW_SEL_TILES * sizeof(qw_cand));
    qw_buf tok = qw_buf_alloc(sizeof(int32_t));
    if (!lg || !out || !scr || !tok) { CHECK(0, "%s: out of memory", what); return; }
    memcpy(qw_buf_contents(lg), logits, (size_t)n * 4);

    qw_cmd c = qw_cmd_begin();
    if (!c) { CHECK(0, "%s: no command buffer", what); return; }
    qw_op_argmax_top2(c, qw_ref_at(out, 0), qw_ref_at(scr, 0), qw_ref_at(lg, 0), n, 1,
                      qw_ref_at(tok, 0), prefix, tail_base);
    qw_cmd_wait(c);
    qw_cmd_free(c);

    const qw_cand *got = qw_buf_contents(out);
    const int32_t  pub = *(const int32_t *)qw_buf_contents(tok);
    CHECK((int32_t)got->index == want_token,
          "%s: mapped index %u, want token %d", what, got->index, want_token);
    CHECK(pub == want_token, "%s: published %d, want token %d", what, pub, want_token);

    qw_buf_free(lg); qw_buf_free(out); qw_buf_free(scr); qw_buf_free(tok);
}

/* The tie cases only prove anything if the lowest-index rule can actually be
 * observed, i.e. if the duplicate maximum sits somewhere other than index 0.
 * Asserted rather than assumed, so a later edit to the fixtures cannot quietly
 * turn the tie tests into a check that ordinary argmax works. */
static void check_tie_rule_is_load_bearing(const float *v, int32_t n, const char *what) {
    const int32_t first = ref_argmax(v, n);
    int32_t dups = 0;
    for (int32_t i = 0; i < n; i++) if (v[i] == v[first]) dups++;
    CHECK(dups >= 2, "%s: fixture has no duplicate maximum, tie rule untested", what);
    CHECK(first != n - 1, "%s: duplicate maximum is not split across the row", what);
}

int main(void) {
    char err[512];
    if (!qw_gpu_init(err, sizeof err)) {
        fprintf(stderr, "GPU unavailable: %s\n", err);
        return 77;
    }

    /* The real shape: a full vocabulary, one row for a draft and a whole block
     * for a verify. */
    const int32_t vocab = 248320;
    float *big = malloc((size_t)(QWASAR_MAX_DRAFT + 1) * vocab * sizeof *big);
    if (!big) { fprintf(stderr, "out of memory\n"); return 1; }

    fill_random(big, (size_t)(QWASAR_MAX_DRAFT + 1) * vocab, 12345);
    run_case("vocab x1", big, vocab, 1);
    run_case("vocab x9", big, vocab, QWASAR_MAX_DRAFT + 1);

    /* Each row must get its own answer: plant a distinct, unmissable maximum
     * per row so a row-indexing slip cannot pass. */
    for (int32_t r = 0; r <= QWASAR_MAX_DRAFT; r++)
        big[(size_t)r * vocab + (size_t)(r * 9001 + 17)] = 100.0f + (float)r;
    run_case("per-row maxima", big, vocab, QWASAR_MAX_DRAFT + 1);

    /* Boundaries of the tile split: first element, last element, and the seam
     * between two tiles. */
    const int32_t per = (vocab + QW_SEL_TILES - 1) / QW_SEL_TILES;
    const int32_t spots[] = { 0, 1, per - 1, per, per + 1, vocab - 1 };
    for (size_t i = 0; i < sizeof spots / sizeof *spots; i++) {
        fill_random(big, (size_t)vocab, 999);
        big[spots[i]] = 50.0f;
        char what[64];
        snprintf(what, sizeof what, "max at %d", spots[i]);
        run_case(what, big, vocab, 1);
    }

    /* Ties.  Two equal maxima in different tiles, and two adjacent: the scan
     * takes the lower index in both, and the margin is exactly zero. */
    fill_random(big, (size_t)vocab, 4242);
    big[1000] = 50.0f;
    big[200000] = 50.0f;
    check_tie_rule_is_load_bearing(big, vocab, "tie across tiles");
    run_case("tie across tiles", big, vocab, 1);

    fill_random(big, (size_t)vocab, 4243);
    big[77777] = 50.0f;
    big[77778] = 50.0f;
    check_tie_rule_is_load_bearing(big, vocab, "tie adjacent");
    run_case("tie adjacent", big, vocab, 1);

    /* Three equal maxima, one of them in the last tile. */
    fill_random(big, (size_t)vocab, 4244);
    big[5] = 50.0f;
    big[per * 15 + 3] = 50.0f;
    big[vocab - 2] = 50.0f;
    check_tie_rule_is_load_bearing(big, vocab, "tie triple");
    run_case("tie triple", big, vocab, 1);

    /* Lengths that do not divide evenly by tiles or by the threadgroup, plus
     * the degenerate ones, so the empty-tile path is exercised. */
    const int32_t lens[] = { 1, 2, 3, 31, 32, 33, 255, 257, 1023, 4097, 100003 };
    for (size_t i = 0; i < sizeof lens / sizeof *lens; i++) {
        fill_random(big, (size_t)lens[i] * 3, (uint32_t)(700 + i));
        char what[64];
        snprintf(what, sizeof what, "n=%d", lens[i]);
        run_case(what, big, lens[i], lens[i] >= 3 ? 3 : 1);
    }

    /* Publishing off must leave the destination alone: the verify path passes
     * no token buffer and would otherwise scribble on whatever was bound. */
    fill_random(big, (size_t)vocab * 3, 31337);
    run_case_ex("no publish", big, vocab, 3, 0);

    /* A row that is entirely one value: every index ties, so the answer is 0
     * and the margin is zero. */
    for (int32_t i = 0; i < vocab; i++) big[i] = -1.5f;
    run_case("all equal", big, vocab, 1);

    /* The compact draft head scores [0, prefix) then the tail run starting at
     * QW_DRAFT_TAIL_LO, so a winning row has to come back as a token id.  Both
     * sides of the seam, and the seam itself. */
    const int32_t tail_n = vocab - QW_DRAFT_TAIL_LO;
    const int32_t compact = QW_DRAFT_PREFIX + tail_n;
    struct { int32_t row; int32_t token; const char *what; } maps[] = {
        { 0,                    0,                     "map first" },
        { 777,                  777,                   "map in prefix" },
        { QW_DRAFT_PREFIX - 1,  QW_DRAFT_PREFIX - 1,   "map last of prefix" },
        { QW_DRAFT_PREFIX,      QW_DRAFT_TAIL_LO,      "map first of tail" },
        { compact - 1,          vocab - 1,             "map last of tail" },
    };
    for (size_t i = 0; i < sizeof maps / sizeof *maps; i++) {
        fill_random(big, (size_t)compact, (uint32_t)(5000 + i));
        big[maps[i].row] = 60.0f;
        run_mapped(maps[i].what, big, compact, QW_DRAFT_PREFIX, QW_DRAFT_TAIL_LO,
                   maps[i].token);
    }

    /* The mapping must be the identity when the head scored everything, or the
     * verify path -- which passes zeroes -- would rewrite its own answers. */
    fill_random(big, (size_t)vocab, 6001);
    big[123456] = 60.0f;
    run_mapped("map identity", big, vocab, 0, 0, 123456);

    /* QW_DRAFT_TAIL_LO is a constant because the graph cannot see the
     * tokenizer.  Check it against the real one: every added special token must
     * fall inside the kept tail, and the tail must not overlap the prefix.
     * Without this the draft could silently lose the ability to propose an
     * end-of-turn marker, which no output comparison would catch -- it would
     * only show up as a worse acceptance rate. */
    const char *model = getenv("QWASAR_TEST_MODEL");
    if (model) {
        char terr[512];
        qwasar_tokenizer *tk = qwasar_tokenizer_load(model, terr, sizeof terr);
        if (!tk) {
            fprintf(stderr, "tokenizer: %s\n", terr);
            fails++;
        } else {
            CHECK(QW_DRAFT_TAIL_LO > QW_DRAFT_PREFIX,
                  "tail overlaps the prefix: %d <= %d", QW_DRAFT_TAIL_LO, QW_DRAFT_PREFIX);
            /* The head is padded past the tokenizer -- 248,320 rows against
             * 248,077 tokens -- so the kept tail carries some rows that decode
             * to nothing.  That is not a new hazard: the target argmaxes over
             * those same rows today, and a draft proposal is verified anyway.
             * The direction is what matters, and it is asserted. */
            CHECK(vocab >= qwasar_tokenizer_size(tk),
                  "head has %d rows but the tokenizer has %d tokens",
                  vocab, qwasar_tokenizer_size(tk));
            int32_t lowest = qwasar_tokenizer_size(tk), n_special = 0;
            for (int32_t id = 0; id < qwasar_tokenizer_size(tk); id++) {
                size_t len; bool sp = false;
                if (!qwasar_token_bytes(tk, id, &len, &sp)) continue;
                if (!sp) continue;
                n_special++;
                if (id < lowest) lowest = id;
            }
            CHECK(n_special > 0, "no special tokens found; the check is vacuous");
            CHECK(lowest >= QW_DRAFT_TAIL_LO,
                  "special token %d is below QW_DRAFT_TAIL_LO %d, so the draft "
                  "cannot propose it", lowest, QW_DRAFT_TAIL_LO);
            printf("  %d special tokens, lowest id %d, tail keeps from %d\n",
                   n_special, lowest, QW_DRAFT_TAIL_LO);
            qwasar_tokenizer_free(tk);
        }
    } else {
        printf("  (set QWASAR_TEST_MODEL to check the vocabulary split)\n");
    }

    free(big);
    qw_gpu_shutdown();

    if (fails) { fprintf(stderr, "%d failure(s)\n", fails); return 1; }
    printf("ok: select\n");
    return 0;
}
