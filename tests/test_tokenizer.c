/* BPE encoding, decoding, and the chat template, against reference fixtures
 * captured from the model's own tokenizer (tools/dump_tokens.py).
 *
 * Tokenization is all-or-nothing: one wrong piece boundary shifts every
 * subsequent id, so these compare exact id sequences rather than any tolerance. */

#include "qwasar.h"
#include "qwasar_json.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;

#define CHECK(cond, ...) do { \
    if (!(cond)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
                   fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); fails++; } \
} while (0)

/* Renders a string with escapes so a failing case is readable in the log. */
static void show(const char *s, size_t n, char *out, size_t cap) {
    size_t w = 0;
    for (size_t i = 0; i < n && w + 5 < cap; i++) {
        unsigned char c = (unsigned char)s[i];
        if      (c == '\n') { out[w++] = '\\'; out[w++] = 'n'; }
        else if (c == '\t') { out[w++] = '\\'; out[w++] = 't'; }
        else if (c == '\r') { out[w++] = '\\'; out[w++] = 'r'; }
        else if (c < 32)    { w += (size_t)snprintf(out + w, cap - w, "\\x%02x", c); }
        else                { out[w++] = (char)c; }
    }
    out[w] = 0;
}

static void report_mismatch(const char *label, const int32_t *got, int32_t n_got,
                            const qj_doc *d, const qj_node *want, int32_t n_want) {
    fprintf(stderr, "    %s: got %d ids, reference has %d\n", label, n_got, n_want);
    fprintf(stderr, "      got :");
    for (int32_t i = 0; i < n_got && i < 24; i++) fprintf(stderr, " %d", got[i]);
    fprintf(stderr, "\n      want:");
    int32_t i = 0;
    for (const qj_node *v = qj_first(d, want); v && i < 24; v = qj_next(d, v), i++)
        fprintf(stderr, " %d", (int32_t)v->u.num);
    fprintf(stderr, "\n");
}

static bool ids_match(const int32_t *got, int32_t n_got,
                      const qj_doc *d, const qj_node *want) {
    if ((int32_t)qj_count(want) != n_got) return false;
    int32_t i = 0;
    for (const qj_node *v = qj_first(d, want); v; v = qj_next(d, v), i++)
        if (got[i] != (int32_t)v->u.num) return false;
    return true;
}

int main(int argc, char **argv) {
    const char *model = getenv("QWASAR_TEST_MODEL");
    const char *fix   = getenv("QWASAR_TOKENS");
    if (argc > 1) model = argv[1];
    if (argc > 2) fix = argv[2];
    if (!fix) fix = "tests/tokens.json";
    if (!model) { fprintf(stderr, "skip: set QWASAR_TEST_MODEL\n"); return 0; }

    qj_doc d;
    if (!qj_parse_file(&d, fix)) {
        fprintf(stderr, "skip: no token fixtures at %s (%s)\n"
                        "      generate with: reference/mlx-vlm/.venv/bin/python \\\n"
                        "        tools/dump_tokens.py <model-dir> %s\n", fix, d.err, fix);
        return 0;
    }

    char err[512];
    qwasar_tokenizer *t = qwasar_tokenizer_load(model, err, sizeof err);
    if (!t) { fprintf(stderr, "tokenizer load failed: %s\n", err); return 1; }
    printf("  vocabulary: %d tokens\n", qwasar_tokenizer_size(t));

    /* Control tokens must resolve by spelling; the chat template depends on it. */
    CHECK(qwasar_token_id(t, "<|im_start|>") == 248045, "<|im_start|> id");
    CHECK(qwasar_token_id(t, "<|im_end|>")   == 248046, "<|im_end|> id");
    CHECK(qwasar_token_id(t, "<think>")      == 248068, "<think> id");
    CHECK(qwasar_token_id(t, "</think>")     == 248069, "</think> id");

    int n_cases = 0, n_ok = 0;
    const qj_node *cases = qj_get(&d, qj_root(&d), "cases");
    for (const qj_node *c = qj_first(&d, cases); c; c = qj_next(&d, c)) {
        const qj_node *tn = qj_get(&d, c, "text");
        const qj_node *in = qj_get(&d, c, "ids");
        if (!tn || !in) continue;

        char *text = malloc(tn->u.str.len + 1);
        memcpy(text, d.text + tn->u.str.off, tn->u.str.len);
        text[tn->u.str.len] = 0;

        int32_t n = 0;
        int32_t *ids = qwasar_encode(t, text, &n);
        n_cases++;

        char pretty[96];
        show(text, tn->u.str.len, pretty, sizeof pretty);

        /* Deliberate divergence: the reference splits input on added tokens, so
         * text containing "<|im_start|>" becomes the control token.  qwasar
         * never does, because a chat prompt built that way lets message content
         * forge a role boundary.  Assert the property we want instead of the
         * reference's ids. */
        if (strstr(text, "<|im_start|>") || strstr(text, "<|im_end|>")) {
            bool leaked = false;
            for (int32_t i = 0; i < n; i++) {
                bool sp = false;
                qwasar_token_bytes(t, ids[i], NULL, &sp);
                if (sp) leaked = true;
            }
            CHECK(!leaked, "literal \"%s\" produced a control token", pretty);
            n_ok++;
        } else if (ids_match(ids, n, &d, in)) {
            n_ok++;
        } else {
            fails++;
            fprintf(stderr, "FAIL encode \"%s\"\n", pretty);
            report_mismatch("encode", ids, n, &d, in, (int32_t)qj_count(in));
        }

        /* Decoding the ids back must reproduce the input byte for byte. */
        char round[4096];
        size_t w = 0;
        for (int32_t i = 0; i < n; i++) {
            size_t len = 0;
            const char *b = qwasar_token_bytes(t, ids[i], &len, NULL);
            if (b && w + len < sizeof round) { memcpy(round + w, b, len); w += len; }
        }
        round[w] = 0;
        CHECK(w == tn->u.str.len && !memcmp(round, text, w),
              "round-trip differs for \"%s\"", pretty);

        free(ids);
        free(text);
    }
    printf("  encode: %d/%d cases (control-token literals held back by design)\n",
           n_ok, n_cases);

    /* Chat template.  Each fixture pins one behaviour of the Qwen3.8 template:
     * the synthesised reasoning system message, the open <think>, the closed
     * one when thinking is disabled, and multi-turn assistant reasoning. */
    /* Byte-identical to the JSON the fixture generator hands the reference
     * template; the tools block is embedded in the prompt verbatim, so any
     * difference in spacing or key order would change the token ids. */
    static const char *const TOOLS[] = {
        "{\"type\": \"function\", \"function\": {\"name\": \"read\", "
        "\"description\": \"Read a file from disk.\", \"parameters\": "
        "{\"type\": \"object\", \"properties\": {\"path\": {\"type\": "
        "\"string\", \"description\": \"File path.\"}}, \"required\": "
        "[\"path\"]}}}",
        "{\"type\": \"function\", \"function\": {\"name\": \"bash\", "
        "\"description\": \"Run a shell command.\", \"parameters\": "
        "{\"type\": \"object\", \"properties\": {\"command\": {\"type\": "
        "\"string\", \"description\": \"Command.\"}}, \"required\": "
        "[\"command\"]}}}",
    };

    struct { const char *name; qwasar_chat_options o; int n; qwasar_message m[4]; } chats[] = {
        { "simple", { true, "xhigh", true, NULL, 0 }, 1,
          { { "user", "What is 2+2?", NULL } } },
        { "with_system", { true, "xhigh", true, NULL, 0 }, 2,
          { { "system", "You are terse.", NULL }, { "user", "Hi", NULL } } },
        { "no_thinking", { false, "xhigh", true, NULL, 0 }, 1,
          { { "user", "Hi", NULL } } },
        { "low_effort", { true, "low", true, NULL, 0 }, 1,
          { { "user", "Hi", NULL } } },
        { "medium_effort", { true, "medium", true, NULL, 0 }, 1,
          { { "user", "Hi", NULL } } },
        { "with_tools", { true, "xhigh", true, TOOLS, 2 }, 1,
          { { "user", "Read /tmp/a.txt", NULL } } },
        { "tools_and_system", { true, "xhigh", true, TOOLS, 2 }, 2,
          { { "system", "Be careful.", NULL }, { "user", "Read /tmp/a.txt", NULL } } },
        { "multi_turn", { true, "xhigh", true, NULL, 0 }, 3,
          { { "user", "First question", NULL },
            { "assistant", "First answer", "thinking here" },
            { "user", "Second question", NULL } } },
    };

    const qj_node *jchats = qj_get(&d, qj_root(&d), "chats");
    for (size_t i = 0; i < sizeof chats / sizeof *chats; i++) {
        const qj_node *want = NULL;
        for (const qj_node *j = qj_first(&d, jchats); j; j = qj_next(&d, j))
            if (qj_str_eq(&d, qj_get(&d, j, "name"), chats[i].name))
                { want = qj_get(&d, j, "ids"); break; }
        if (!want) { fprintf(stderr, "FAIL no fixture for chat '%s'\n", chats[i].name); fails++; continue; }

        int32_t n = 0;
        int32_t *ids = qwasar_apply_chat_template(t, chats[i].m, chats[i].n,
                                                  &chats[i].o, &n, err, sizeof err);
        if (!ids) { fprintf(stderr, "FAIL chat '%s': %s\n", chats[i].name, err); fails++; continue; }

        if (ids_match(ids, n, &d, want)) {
            printf("  chat %-16s %3d tokens  exact\n", chats[i].name, n);
        } else {
            fails++;
            fprintf(stderr, "FAIL chat template '%s'\n", chats[i].name);
            report_mismatch(chats[i].name, ids, n, &d, want, (int32_t)qj_count(want));
        }
        free(ids);
    }

    qwasar_tokenizer_free(t);
    qj_free(&d);
    if (fails) { fprintf(stderr, "%d check(s) failed\n", fails); return 1; }
    printf("ok: tokenizer\n");
    return 0;
}
