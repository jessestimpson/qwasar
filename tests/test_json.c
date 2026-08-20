#include "qwasar_json.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;

#define CHECK(cond, ...) do { \
    if (!(cond)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
                   fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); fails++; } \
} while (0)

static void test_scalars(void) {
    qj_doc d;
    const char *src =
        "{\"a\": 1, \"b\": -2.5e3, \"c\": true, \"d\": false, \"e\": null,"
        " \"s\": \"hi\\nthere\", \"nested\": {\"deep\": {\"x\": 42}}}";
    CHECK(qj_parse(&d, src, strlen(src)), "parse failed: %s", d.err);

    const qj_node *r = qj_root(&d);
    CHECK(r->type == QJ_OBJECT, "root should be an object");
    CHECK(qj_int_or(&d, r, "a", 0) == 1, "a");
    CHECK(qj_num_or(&d, r, "b", 0) == -2500.0, "b");
    CHECK(qj_bool_or(&d, r, "c", false) == true, "c");
    CHECK(qj_bool_or(&d, r, "d", true) == false, "d");
    CHECK(qj_int_or(&d, r, "missing", 7) == 7, "default");
    CHECK(qj_int_or(&d, r, "nested.deep.x", 0) == 42, "dotted path");

    char buf[32];
    CHECK(qj_str_copy(&d, r, "s", buf, sizeof buf), "string present");
    CHECK(!strcmp(buf, "hi\nthere"), "escape decoding, got '%s'", buf);
    qj_free(&d);
}

static void test_arrays(void) {
    qj_doc d;
    const char *src = "{\"v\": [10, 20, 30], \"empty\": [], \"deep\": [[1,2],[3]]}";
    CHECK(qj_parse(&d, src, strlen(src)), "parse failed: %s", d.err);
    const qj_node *r = qj_root(&d);

    const qj_node *v = qj_get(&d, r, "v");
    CHECK(qj_count(v) == 3, "array length");
    CHECK((int)qj_idx(&d, v, 1)->u.num == 20, "element 1");
    CHECK(qj_idx(&d, v, 3) == NULL, "out of range");
    CHECK(qj_count(qj_get(&d, r, "empty")) == 0, "empty array");

    int sum = 0;
    for (const qj_node *c = qj_first(&d, v); c; c = qj_next(&d, c)) sum += (int)c->u.num;
    CHECK(sum == 60, "sibling walk, got %d", sum);

    const qj_node *deep = qj_get(&d, r, "deep");
    CHECK(qj_count(qj_idx(&d, deep, 0)) == 2, "nested array");
    qj_free(&d);
}

static void test_unicode(void) {
    qj_doc d;
    /* BMP escape, a surrogate pair, and a lone high surrogate. */
    const char *src = "{\"a\":\"\\u00e9\",\"b\":\"\\ud83d\\ude00\",\"c\":\"\\ud800x\"}";
    CHECK(qj_parse(&d, src, strlen(src)), "parse failed: %s", d.err);
    const qj_node *r = qj_root(&d);

    char buf[32];
    qj_str_copy(&d, r, "a", buf, sizeof buf);
    CHECK(!strcmp(buf, "\xc3\xa9"), "U+00E9");
    qj_str_copy(&d, r, "b", buf, sizeof buf);
    CHECK(!strcmp(buf, "\xf0\x9f\x98\x80"), "surrogate pair -> U+1F600");
    qj_str_copy(&d, r, "c", buf, sizeof buf);
    CHECK(!strcmp(buf, "\xef\xbf\xbdx"), "lone surrogate -> U+FFFD");
    qj_free(&d);
}

static void test_errors(void) {
    qj_doc d;
    const char *bad[] = { "{", "{\"a\"}", "{\"a\":}", "[1,", "tru", "{\"a\":\"x}" };
    for (size_t i = 0; i < sizeof bad / sizeof *bad; i++) {
        bool ok = qj_parse(&d, bad[i], strlen(bad[i]));
        CHECK(!ok, "should reject '%s'", bad[i]);
        CHECK(d.err[0] != 0, "should set an error for '%s'", bad[i]);
        qj_free(&d);
    }
}

/* The parser's real workload is a 20 MB tokenizer.json with ~250k object
 * members, so exercise a wide object rather than only toy input. */
static void test_wide_object(void) {
    const int n = 20000;
    size_t cap = (size_t)n * 24 + 8;
    char *src = malloc(cap);
    size_t len = 0;
    len += (size_t)snprintf(src + len, cap - len, "{");
    for (int i = 0; i < n; i++)
        len += (size_t)snprintf(src + len, cap - len, "%s\"k%d\":%d", i ? "," : "", i, i);
    len += (size_t)snprintf(src + len, cap - len, "}");

    qj_doc d;
    CHECK(qj_parse(&d, src, len), "wide parse failed: %s", d.err);
    const qj_node *r = qj_root(&d);
    CHECK(qj_count(r) == (uint32_t)n, "member count %u", qj_count(r));

    long sum = 0;
    for (const qj_node *c = qj_first(&d, r); c; c = qj_next(&d, c)) sum += (long)c->u.num;
    CHECK(sum == (long)n * (n - 1) / 2, "sum over members");
    qj_free(&d);
    free(src);
}

int main(void) {
    test_scalars();
    test_arrays();
    test_unicode();
    test_errors();
    test_wide_object();
    if (fails) { fprintf(stderr, "%d check(s) failed\n", fails); return 1; }
    printf("ok: json\n");
    return 0;
}
