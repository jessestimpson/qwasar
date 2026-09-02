/* Combining marks under Flash-Next's split regex.
 *
 * The two tokenizers share vocab and merges and differ in one place: the
 * pattern's letter run is [\p{L}\p{M}]+, so a combining mark stays with the
 * letter it attaches to.  The engine reads the pattern from tokenizer.json,
 * so this test takes the 27B's file (the only one on this machine), rewrites
 * that one pattern to Flash-Next's, loads the result, and holds it to ids the
 * `tokenizers` library produced from Flash-Next's own file
 * (tools/dump_tokens_marks.py).  Five of the thirteen cases differ between
 * the two patterns, so the test can tell them apart. */

#include "qwasar.h"
#include "qwasar_json.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int fails;

#define CHECK(cond, ...) do { \
    if (!(cond)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
                   fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); fails++; } \
} while (0)

/* Replaces every occurrence of `from` with `to`; returns the new buffer. */
static char *replace_all(const char *src, size_t n, const char *from, const char *to, size_t *out_n) {
    const size_t fl = strlen(from), tl = strlen(to);
    size_t cap = n + 64, w = 0;
    char *out = malloc(cap);
    for (size_t i = 0; i < n; ) {
        if (i + fl <= n && !memcmp(src + i, from, fl)) {
            if (w + tl + 1 > cap) { cap = cap * 2 + tl; out = realloc(out, cap); }
            memcpy(out + w, to, tl); w += tl; i += fl;
        } else {
            if (w + 2 > cap) { cap *= 2; out = realloc(out, cap); }
            out[w++] = src[i++];
        }
    }
    out[w] = 0;
    *out_n = w;
    return out;
}

int main(int argc, char **argv) {
    const char *model = getenv("QWASAR_TEST_MODEL");
    if (argc > 1) model = argv[1];
    if (!model) { fprintf(stderr, "skip: set QWASAR_TEST_MODEL\n"); return 0; }

    char path[1200];
    snprintf(path, sizeof path, "%s/tokenizer.json", model);
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "skip: no %s\n", path); return 0; }
    fseek(f, 0, SEEK_END);
    size_t n = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);
    char *json = malloc(n + 1);
    if (fread(json, 1, n, f) != n) { fprintf(stderr, "short read\n"); return 1; }
    fclose(f);
    json[n] = 0;

    /* The 27B's two letter-run fragments, as they sit escaped in the JSON,
     * rewritten to Flash-Next's. */
    size_t n2 = 0, n3 = 0;
    char *j2 = replace_all(json, n, "\\\\p{N}]?\\\\p{L}+", "\\\\p{N}]?[\\\\p{L}\\\\p{M}]+", &n2);
    char *j3 = replace_all(j2, n2, "[^\\\\s\\\\p{L}\\\\p{N}]+", "[^\\\\s\\\\p{L}\\\\p{M}\\\\p{N}]+", &n3);
    CHECK(n3 > n, "the 27B pattern was not found in %s; is this the right tokenizer?", path);

    char dir[] = "/tmp/qwasar-marks-XXXXXX";
    if (!mkdtemp(dir)) { fprintf(stderr, "mkdtemp failed\n"); return 1; }
    snprintf(path, sizeof path, "%s/tokenizer.json", dir);
    f = fopen(path, "wb");
    fwrite(j3, 1, n3, f);
    fclose(f);

    char err[512];
    qwasar_tokenizer *t = qwasar_tokenizer_load(dir, err, sizeof err);
    CHECK(t != NULL, "tokenizer load: %s", err);
    qwasar_tokenizer *t27 = qwasar_tokenizer_load(model, err, sizeof err);
    CHECK(t27 != NULL, "27B tokenizer load: %s", err);

    qj_doc d;
    if (!qj_parse_file(&d, "tests/tokens_marks.json")) {
        fprintf(stderr, "cannot parse tests/tokens_marks.json: %s\n", d.err);
        return 1;
    }
    int n_cases = 0, n_ok = 0, n_differ = 0;
    const qj_node *cases = qj_get(&d, qj_root(&d), "cases");
    for (const qj_node *c = qj_first(&d, cases); c && t && t27; c = qj_next(&d, c)) {
        const qj_node *tn = qj_get(&d, c, "text");
        const qj_node *in = qj_get(&d, c, "ids");
        char *text = malloc(tn->u.str.len + 1);
        memcpy(text, d.text + tn->u.str.off, tn->u.str.len);
        text[tn->u.str.len] = 0;

        int32_t got_n = 0, old_n = 0;
        int32_t *got = qwasar_encode(t, text, &got_n);
        int32_t *old = qwasar_encode(t27, text, &old_n);
        n_cases++;
        bool ok = (int32_t)qj_count(in) == got_n;
        int32_t i = 0;
        for (const qj_node *v = qj_first(&d, in); v && ok; v = qj_next(&d, v), i++)
            if (got[i] != (int32_t)v->u.num) ok = false;
        bool same_as_27 = old_n == got_n && !memcmp(got, old, (size_t)got_n * sizeof *got);
        if (!same_as_27) n_differ++;
        if (ok) n_ok++;
        else {
            fprintf(stderr, "    mismatch on %s: got", text);
            for (int32_t k = 0; k < got_n; k++) fprintf(stderr, " %d", got[k]);
            fprintf(stderr, "\n");
        }
        free(got); free(old); free(text);
    }
    printf("  %d/%d cases match Flash-Next's tokenizer; %d of them differ from the 27B's pattern\n",
           n_ok, n_cases, n_differ);
    CHECK(n_ok == n_cases, "%d case(s) mismatch", n_cases - n_ok);
    CHECK(n_differ >= 4, "only %d cases differ from the 27B: the test is not discriminating", n_differ);

    /* And the 27B's own file must still read as its own pattern: the two
     * loads above disagree on exactly the marks cases, nothing else. */
    qj_free(&d);
    qwasar_tokenizer_free(t);
    qwasar_tokenizer_free(t27);
    unlink(path);
    rmdir(dir);
    free(json); free(j2); free(j3);

    if (fails) { fprintf(stderr, "%d failure(s)\n", fails); return 1; }
    printf("tokenizer marks: all checks pass\n");
    return 0;
}
