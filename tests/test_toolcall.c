/* Tool-call parsing and the file-edit matcher.
 *
 * These run without a model, so they are fast and cover the cases that matter:
 * values that span lines or contain markup, malformed calls the model will
 * eventually emit, and every way an edit can fail to identify a unique target. */

#include "qwasar_toolcall.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;

#define CHECK(cond, ...) do { \
    if (!(cond)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
                   fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); fails++; } \
} while (0)

#define CHECK_STR(got, want, ...) do { \
    const char *g_ = (got), *w_ = (want); \
    if (!g_ || strcmp(g_, w_)) { \
        fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
        fprintf(stderr, __VA_ARGS__); \
        fprintf(stderr, "\n      got  <<%s>>\n      want <<%s>>\n", g_ ? g_ : "(null)", w_); \
        fails++; } \
} while (0)

static void test_simple_call(void) {
    const char *text =
        "<tool_call>\n"
        "<function=read>\n"
        "<parameter=path>\n"
        "/tmp/example.c\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>";

    qw_tool_calls c;
    char err[256] = "";
    int n = qw_tool_parse(text, &c, err, sizeof err);
    CHECK(n == 1, "expected 1 call, got %d (%s)", n, err);
    if (n == 1) {
        CHECK_STR(c.calls[0].name, "read", "function name");
        CHECK(c.calls[0].n_params == 1, "param count %d", c.calls[0].n_params);
        CHECK_STR(qw_tool_arg(&c.calls[0], "path"), "/tmp/example.c", "path");
        CHECK(qw_tool_arg(&c.calls[0], "missing") == NULL, "absent arg");
    }
    qw_tool_calls_free(&c);
}

/* A file body is the whole point of the write tool, so a value has to survive
 * newlines, blank lines, braces, and XML-looking text unchanged. */
static void test_multiline_value(void) {
    const char *body =
        "#include <stdio.h>\n"
        "\n"
        "int main(void) {\n"
        "    printf(\"a < b && c > d\\n\");\n"
        "    return 0;\n"
        "}";
    char text[2048];
    snprintf(text, sizeof text,
             "I'll write the file now.\n\n"
             "<tool_call>\n<function=write>\n"
             "<parameter=path>\n/tmp/a.c\n</parameter>\n"
             "<parameter=content>\n%s\n</parameter>\n"
             "</function>\n</tool_call>", body);

    qw_tool_calls c;
    char err[256] = "";
    int n = qw_tool_parse(text, &c, err, sizeof err);
    CHECK(n == 1, "expected 1 call, got %d (%s)", n, err);
    if (n == 1) {
        CHECK_STR(qw_tool_arg(&c.calls[0], "content"), body, "multiline body");
        CHECK_STR(c.preamble, "I'll write the file now.", "preamble");
    }
    qw_tool_calls_free(&c);
}

/* Leading and trailing blank lines inside a value are content, not padding. */
static void test_value_keeps_blank_lines(void) {
    const char *text =
        "<tool_call>\n<function=write>\n"
        "<parameter=content>\n\nmiddle\n\n</parameter>\n"
        "</function>\n</tool_call>";
    qw_tool_calls c;
    char err[256] = "";
    CHECK(qw_tool_parse(text, &c, err, sizeof err) == 1, "parse: %s", err);
    CHECK_STR(qw_tool_arg(&c.calls[0], "content"), "\nmiddle\n", "blank lines kept");
    qw_tool_calls_free(&c);
}

static void test_multiple_calls_and_params(void) {
    const char *text =
        "<tool_call>\n<function=edit>\n"
        "<parameter=path>\na.c\n</parameter>\n"
        "<parameter=old>\nint x;\n</parameter>\n"
        "<parameter=new>\nlong x;\n</parameter>\n"
        "</function>\n</tool_call>\n"
        "<tool_call>\n<function=bash>\n"
        "<parameter=command>\nmake\n</parameter>\n"
        "</function>\n</tool_call>";
    qw_tool_calls c;
    char err[256] = "";
    int n = qw_tool_parse(text, &c, err, sizeof err);
    CHECK(n == 2, "expected 2 calls, got %d (%s)", n, err);
    if (n == 2) {
        CHECK_STR(c.calls[0].name, "edit", "first name");
        CHECK(c.calls[0].n_params == 3, "first param count %d", c.calls[0].n_params);
        CHECK_STR(qw_tool_arg(&c.calls[0], "new"), "long x;", "new arg");
        CHECK_STR(c.calls[1].name, "bash", "second name");
        CHECK_STR(qw_tool_arg(&c.calls[1], "command"), "make", "command arg");
    }
    qw_tool_calls_free(&c);
}

static void test_malformed(void) {
    struct { const char *text; const char *what; int expect; } cases[] = {
        { "no tool call here at all", "plain prose", 0 },
        { "<tool_call>\n</tool_call>", "no function block", -1 },
        { "<tool_call>\n<function=read\n</tool_call>", "unterminated function tag", -1 },
        { "<tool_call>\n<function=read>\n<parameter=path>\nx\n</function>\n</tool_call>",
          "unclosed parameter", -1 },
    };
    for (size_t i = 0; i < sizeof cases / sizeof *cases; i++) {
        qw_tool_calls c;
        char err[256] = "";
        int n = qw_tool_parse(cases[i].text, &c, err, sizeof err);
        CHECK(n == cases[i].expect, "%s: got %d, expected %d",
              cases[i].what, n, cases[i].expect);
        if (cases[i].expect < 0)
            CHECK(err[0] != 0, "%s: should explain the failure", cases[i].what);
        qw_tool_calls_free(&c);
    }
}

static void test_completion_detection(void) {
    const char *partial = "<tool_call>\n<function=read>\n<parameter=path>\n/tmp/a";
    const char *whole   = "<tool_call>\n<function=read>\n</function>\n</tool_call>";
    CHECK(!qw_tool_call_complete(partial, strlen(partial)), "partial is not complete");
    CHECK(qw_tool_call_complete(whole, strlen(whole)), "closed call is complete");
}

/* ---- edit ------------------------------------------------------------------ */

static const char *FILE_A =
    "#include <stdio.h>\n"
    "\n"
    "int add(int a, int b) {\n"
    "    return a + b;\n"
    "}\n"
    "\n"
    "int sub(int a, int b) {\n"
    "    return a - b;\n"
    "}\n";

static void check_edit(const char *label, const char *content, const char *old,
                       const char *new, qw_edit_status want_status,
                       const char *want_result, int want_matches) {
    char *out = NULL;
    size_t out_len = 0;
    int matches = -1;
    qw_edit_status s = qw_edit_apply(content, strlen(content), old, new,
                                     &out, &out_len, &matches);
    CHECK(s == want_status, "%s: status %s, expected %s",
          label, qw_edit_status_text(s), qw_edit_status_text(want_status));
    if (want_matches >= 0)
        CHECK(matches == want_matches, "%s: %d matches, expected %d",
              label, matches, want_matches);
    if (s == QW_EDIT_OK && want_result) {
        CHECK(out_len == strlen(out), "%s: length disagrees with contents", label);
        CHECK_STR(out, want_result, "%s: result", label);
    }
    free(out);
}

static void test_edit_unique(void) {
    check_edit("replace a body", FILE_A,
               "    return a + b;",
               "    return b + a;",
               QW_EDIT_OK,
               "#include <stdio.h>\n\nint add(int a, int b) {\n    return b + a;\n}\n"
               "\nint sub(int a, int b) {\n    return a - b;\n}\n", 1);

    check_edit("replace a multi-line run", FILE_A,
               "int add(int a, int b) {\n    return a + b;\n}",
               "int add(int a, int b) { return a + b; }",
               QW_EDIT_OK,
               "#include <stdio.h>\n\nint add(int a, int b) { return a + b; }\n"
               "\nint sub(int a, int b) {\n    return a - b;\n}\n", 1);

    /* A trailing newline on `old` is presentation; quoting a block with or
     * without one must behave the same. */
    check_edit("old with trailing newline", FILE_A,
               "    return a + b;\n",
               "    return 0;",
               QW_EDIT_OK, NULL, 1);
}

static void test_edit_rejects_ambiguity(void) {
    const char *dup = "x = 1;\ny = 2;\nx = 1;\n";
    check_edit("two identical lines", dup, "x = 1;", "x = 3;",
               QW_EDIT_AMBIGUOUS, NULL, 2);

    /* A bare closing brace is exactly the anchor ds4's [upto] format warns
     * about; here it is rejected outright instead of guessing. */
    check_edit("bare closing brace", FILE_A, "}", "} /* end */",
               QW_EDIT_AMBIGUOUS, NULL, 2);
}

static void test_edit_not_found(void) {
    check_edit("absent text", FILE_A, "int mul(int a, int b) {", "x",
               QW_EDIT_NOT_FOUND, NULL, 0);

    /* Whole-line anchoring: a fragment of a line must not match. */
    check_edit("partial line", FILE_A, "return a + b", "return b + a",
               QW_EDIT_NOT_FOUND, NULL, 0);

    /* Indentation is content. */
    check_edit("wrong indentation", FILE_A, "return a + b;", "return b + a;",
               QW_EDIT_NOT_FOUND, NULL, 0);
}

static void test_edit_delete_and_edges(void) {
    check_edit("delete a line", "a\nb\nc\n", "b", "",
               QW_EDIT_OK, "a\nc\n", 1);

    check_edit("first line", "a\nb\nc\n", "a", "A", QW_EDIT_OK, "A\nb\nc\n", 1);
    check_edit("last line",  "a\nb\nc\n", "c", "C", QW_EDIT_OK, "a\nb\nC\n", 1);

    /* A file with no trailing newline must not gain or lose one. */
    check_edit("no trailing newline", "a\nb", "b", "B", QW_EDIT_OK, "a\nB", 1);

    check_edit("whole file", "a\nb\n", "a\nb", "x", QW_EDIT_OK, "x\n", 1);
    check_edit("empty old", FILE_A, "", "x", QW_EDIT_EMPTY_OLD, NULL, -1);
    check_edit("insert via anchor", "a\nc\n", "a", "a\nb", QW_EDIT_OK, "a\nb\nc\n", 1);
}

int main(void) {
    test_simple_call();
    test_multiline_value();
    test_value_keeps_blank_lines();
    test_multiple_calls_and_params();
    test_malformed();
    test_completion_detection();

    test_edit_unique();
    test_edit_rejects_ambiguity();
    test_edit_not_found();
    test_edit_delete_and_edges();

    if (fails) { fprintf(stderr, "%d check(s) failed\n", fails); return 1; }
    printf("ok: toolcall\n");
    return 0;
}
