#include "qwasar_json.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- node arena ---------------------------------------------------------- */

/* Index 0 is the reserved "absent" sentinel and is a perfectly valid result of
 * the first allocation, so failure has to be signalled out of band. */
#define QJ_NOMEM UINT32_MAX

static uint32_t qj_alloc(qj_doc *d) {
    if (d->count == d->cap) {
        uint32_t cap = d->cap ? d->cap * 2 : 64;
        qj_node *n = realloc(d->nodes, (size_t)cap * sizeof *n);
        if (!n) return QJ_NOMEM;
        d->nodes = n;
        d->cap = cap;
    }
    uint32_t idx = d->count++;
    memset(&d->nodes[idx], 0, sizeof d->nodes[idx]);
    return idx;
}

/* ---- scanner -------------------------------------------------------------
 *
 * `p` walks the mutable copy.  String bodies are unescaped in place: `w`
 * trails `p` inside a literal, so the unescaped bytes overwrite the raw ones
 * and we hand back an (offset, length) into the same buffer. */

typedef struct {
    qj_doc *d;
    char   *base;
    char   *p, *end;
} qj_scan;

static void qj_fail(qj_scan *s, const char *what) {
    if (!s->d->err[0])
        snprintf(s->d->err, sizeof s->d->err, "%s at byte %ld",
                 what, (long)(s->p - s->base));
}

static void qj_ws(qj_scan *s) {
    while (s->p < s->end) {
        char c = *s->p;
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') s->p++;
        else break;
    }
}

static int qj_hex4(const char *p) {
    int v = 0;
    for (int i = 0; i < 4; i++) {
        char c = p[i];
        int d;
        if (c >= '0' && c <= '9') d = c - '0';
        else if (c >= 'a' && c <= 'f') d = c - 'a' + 10;
        else if (c >= 'A' && c <= 'F') d = c - 'A' + 10;
        else return -1;
        v = v * 16 + d;
    }
    return v;
}

static char *qj_utf8_put(char *w, uint32_t cp) {
    if (cp < 0x80) { *w++ = (char)cp; }
    else if (cp < 0x800) {
        *w++ = (char)(0xC0 | (cp >> 6));
        *w++ = (char)(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        *w++ = (char)(0xE0 | (cp >> 12));
        *w++ = (char)(0x80 | ((cp >> 6) & 0x3F));
        *w++ = (char)(0x80 | (cp & 0x3F));
    } else {
        *w++ = (char)(0xF0 | (cp >> 18));
        *w++ = (char)(0x80 | ((cp >> 12) & 0x3F));
        *w++ = (char)(0x80 | ((cp >> 6) & 0x3F));
        *w++ = (char)(0x80 | (cp & 0x3F));
    }
    return w;
}

/* Consumes a quoted string starting at s->p (which must point at '"').
 * Returns false on error; on success writes the in-place span. */
static bool qj_string(qj_scan *s, uint32_t *off, uint32_t *len) {
    if (s->p >= s->end || *s->p != '"') { qj_fail(s, "expected string"); return false; }
    s->p++;
    char *w = s->p;
    char *start = w;
    while (s->p < s->end) {
        unsigned char c = (unsigned char)*s->p;
        if (c == '"') {
            s->p++;
            *off = (uint32_t)(start - s->base);
            *len = (uint32_t)(w - start);
            return true;
        }
        if (c != '\\') { *w++ = (char)c; s->p++; continue; }

        /* escape */
        if (s->p + 1 >= s->end) { qj_fail(s, "truncated escape"); return false; }
        char e = s->p[1];
        s->p += 2;
        switch (e) {
        case '"':  *w++ = '"';  break;
        case '\\': *w++ = '\\'; break;
        case '/':  *w++ = '/';  break;
        case 'b':  *w++ = '\b'; break;
        case 'f':  *w++ = '\f'; break;
        case 'n':  *w++ = '\n'; break;
        case 'r':  *w++ = '\r'; break;
        case 't':  *w++ = '\t'; break;
        case 'u': {
            if (s->p + 4 > s->end) { qj_fail(s, "truncated \\u"); return false; }
            int hi = qj_hex4(s->p);
            if (hi < 0) { qj_fail(s, "bad \\u escape"); return false; }
            s->p += 4;
            uint32_t cp = (uint32_t)hi;
            /* Surrogate pair: \uD800-\uDBFF must be followed by \uDC00-\uDFFF.
             * Lone surrogates are passed through as U+FFFD rather than
             * rejected -- some vocabularies contain them. */
            if (cp >= 0xD800 && cp <= 0xDBFF) {
                if (s->p + 6 <= s->end && s->p[0] == '\\' && s->p[1] == 'u') {
                    int lo = qj_hex4(s->p + 2);
                    if (lo >= 0xDC00 && lo <= 0xDFFF) {
                        cp = 0x10000 + ((cp - 0xD800) << 10) + ((uint32_t)lo - 0xDC00);
                        s->p += 6;
                    } else cp = 0xFFFD;
                } else cp = 0xFFFD;
            } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                cp = 0xFFFD;
            }
            w = qj_utf8_put(w, cp);
            break;
        }
        default: qj_fail(s, "unknown escape"); return false;
        }
    }
    qj_fail(s, "unterminated string");
    return false;
}

static uint32_t qj_value(qj_scan *s);

/* Parses the body of an array or object into a child list. */
static bool qj_container(qj_scan *s, uint32_t self, char close) {
    uint32_t first = 0, prev = 0, n = 0;
    bool is_obj = (close == '}');
    for (;;) {
        qj_ws(s);
        if (s->p >= s->end) { qj_fail(s, "unterminated container"); return false; }
        if (*s->p == close) { s->p++; break; }
        if (n > 0) {
            if (*s->p != ',') { qj_fail(s, "expected ','"); return false; }
            s->p++;
            qj_ws(s);
            if (s->p < s->end && *s->p == close) { s->p++; break; } /* trailing comma */
        }

        uint32_t key_off = 0, key_len = 0;
        if (is_obj) {
            qj_ws(s);
            if (!qj_string(s, &key_off, &key_len)) return false;
            qj_ws(s);
            if (s->p >= s->end || *s->p != ':') { qj_fail(s, "expected ':'"); return false; }
            s->p++;
        }

        uint32_t child = qj_value(s);
        if (child == QJ_NOMEM) return false;
        /* qj_value may have grown the arena, so index rather than cache pointers. */
        s->d->nodes[child].key_off = key_off;
        s->d->nodes[child].key_len = key_len;
        if (prev) s->d->nodes[prev].next = child; else first = child;
        prev = child;
        n++;
    }
    s->d->nodes[self].u.list.first = first;
    s->d->nodes[self].u.list.count = n;
    return true;
}

static uint32_t qj_value(qj_scan *s) {
    qj_ws(s);
    if (s->p >= s->end) { qj_fail(s, "unexpected end of input"); return QJ_NOMEM; }

    uint32_t self = qj_alloc(s->d);
    if (self == QJ_NOMEM) { qj_fail(s, "out of memory"); return QJ_NOMEM; }
    char c = *s->p;

    switch (c) {
    case '{':
    case '[':
        s->p++;
        s->d->nodes[self].type = (c == '{') ? QJ_OBJECT : QJ_ARRAY;
        if (!qj_container(s, self, c == '{' ? '}' : ']')) return QJ_NOMEM;
        return self;

    case '"': {
        uint32_t off, len;
        if (!qj_string(s, &off, &len)) return QJ_NOMEM;
        s->d->nodes[self].type = QJ_STRING;
        s->d->nodes[self].u.str.off = off;
        s->d->nodes[self].u.str.len = len;
        return self;
    }

    case 't':
        if (s->end - s->p < 4 || memcmp(s->p, "true", 4)) { qj_fail(s, "bad literal"); return QJ_NOMEM; }
        s->p += 4; s->d->nodes[self].type = QJ_TRUE; return self;
    case 'f':
        if (s->end - s->p < 5 || memcmp(s->p, "false", 5)) { qj_fail(s, "bad literal"); return QJ_NOMEM; }
        s->p += 5; s->d->nodes[self].type = QJ_FALSE; return self;
    case 'n':
        if (s->end - s->p < 4 || memcmp(s->p, "null", 4)) { qj_fail(s, "bad literal"); return QJ_NOMEM; }
        s->p += 4; s->d->nodes[self].type = QJ_NULL; return self;

    default: {
        char *endp = NULL;
        double v = strtod(s->p, &endp);
        if (endp == s->p) { qj_fail(s, "bad value"); return QJ_NOMEM; }
        s->p = endp;
        s->d->nodes[self].type = QJ_NUMBER;
        s->d->nodes[self].u.num = v;
        return self;
    }
    }
}

bool qj_parse(qj_doc *d, const char *text, size_t len) {
    memset(d, 0, sizeof *d);
    d->text = malloc(len + 1);
    if (!d->text) { snprintf(d->err, sizeof d->err, "out of memory"); return false; }
    memcpy(d->text, text, len);
    d->text[len] = 0;
    d->len = len;

    if (qj_alloc(d) != 0) { snprintf(d->err, sizeof d->err, "out of memory"); return false; }
    d->nodes[0].type = QJ_NULL;   /* sentinel: index 0 means "absent" */

    qj_scan s = { d, d->text, d->text, d->text + len };
    /* Skip a UTF-8 BOM if present. */
    if (len >= 3 && (unsigned char)s.p[0] == 0xEF && (unsigned char)s.p[1] == 0xBB
                 && (unsigned char)s.p[2] == 0xBF) s.p += 3;
    uint32_t root = qj_value(&s);
    if (root == QJ_NOMEM) return false;
    if (root != 1) { snprintf(d->err, sizeof d->err, "internal: root is not node 1"); return false; }
    qj_ws(&s);
    return true;
}

bool qj_parse_file(qj_doc *d, const char *path) {
    memset(d, 0, sizeof *d);
    FILE *f = fopen(path, "rb");
    if (!f) { snprintf(d->err, sizeof d->err, "cannot open %s", path); return false; }
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); snprintf(d->err, sizeof d->err, "seek failed"); return false; }
    long sz = ftell(f);
    if (sz < 0) { fclose(f); snprintf(d->err, sizeof d->err, "tell failed"); return false; }
    rewind(f);

    char *buf = malloc((size_t)sz + 1);
    if (!buf) { fclose(f); snprintf(d->err, sizeof d->err, "out of memory"); return false; }
    size_t got = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    if (got != (size_t)sz) { free(buf); snprintf(d->err, sizeof d->err, "short read on %s", path); return false; }

    bool ok = qj_parse(d, buf, got);
    free(buf);
    return ok;
}

void qj_free(qj_doc *d) {
    free(d->text);
    free(d->nodes);
    d->text = NULL;
    d->nodes = NULL;
    d->count = d->cap = 0;
}

/* ---- accessors ----------------------------------------------------------- */

const qj_node *qj_get(const qj_doc *d, const qj_node *obj, const char *key) {
    if (!obj || obj->type != QJ_OBJECT) return NULL;
    size_t klen = strlen(key);
    for (const qj_node *c = qj_first(d, obj); c; c = qj_next(d, c))
        if (c->key_len == klen && !memcmp(d->text + c->key_off, key, klen)) return c;
    return NULL;
}

const qj_node *qj_idx(const qj_doc *d, const qj_node *arr, uint32_t i) {
    if (!arr || arr->type != QJ_ARRAY) return NULL;
    for (const qj_node *c = qj_first(d, arr); c; c = qj_next(d, c))
        if (i-- == 0) return c;
    return NULL;
}

const qj_node *qj_path(const qj_doc *d, const qj_node *obj, const char *path) {
    const qj_node *cur = obj ? obj : qj_root(d);
    const char *p = path;
    char seg[128];
    while (*p) {
        const char *dot = strchr(p, '.');
        size_t n = dot ? (size_t)(dot - p) : strlen(p);
        if (n >= sizeof seg) return NULL;
        memcpy(seg, p, n);
        seg[n] = 0;
        cur = qj_get(d, cur, seg);
        if (!cur) return NULL;
        p = dot ? dot + 1 : p + n;
    }
    return cur;
}

double qj_num_or(const qj_doc *d, const qj_node *obj, const char *path, double dflt) {
    const qj_node *n = qj_path(d, obj, path);
    return (n && n->type == QJ_NUMBER) ? n->u.num : dflt;
}

int64_t qj_int_or(const qj_doc *d, const qj_node *obj, const char *path, int64_t dflt) {
    const qj_node *n = qj_path(d, obj, path);
    return (n && n->type == QJ_NUMBER) ? (int64_t)n->u.num : dflt;
}

bool qj_bool_or(const qj_doc *d, const qj_node *obj, const char *path, bool dflt) {
    const qj_node *n = qj_path(d, obj, path);
    if (!n) return dflt;
    if (n->type == QJ_TRUE) return true;
    if (n->type == QJ_FALSE) return false;
    return dflt;
}

bool qj_str_copy(const qj_doc *d, const qj_node *obj, const char *path,
                 char *out, size_t cap) {
    const qj_node *n = qj_path(d, obj, path);
    if (!n || n->type != QJ_STRING || cap == 0) return false;
    size_t len = n->u.str.len;
    if (len > cap - 1) len = cap - 1;
    memcpy(out, d->text + n->u.str.off, len);
    out[len] = 0;
    return true;
}

bool qj_str_eq(const qj_doc *d, const qj_node *n, const char *s) {
    if (!n || n->type != QJ_STRING) return false;
    size_t len = strlen(s);
    return n->u.str.len == len && !memcmp(d->text + n->u.str.off, s, len);
}
