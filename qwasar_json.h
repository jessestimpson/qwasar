#ifndef QWASAR_JSON_H
#define QWASAR_JSON_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

/* Minimal index-based JSON parser.
 *
 * Nodes live in one flat array and reference their text by offset into a
 * mutable copy of the document, which the parser unescapes in place (escaping
 * only ever shrinks a string, so this is safe and allocation-free).  Children
 * form a singly linked list so an object with 250k members costs one node each
 * and no per-member allocation -- tokenizer.json is 20 MB and ~1.1M nodes, and
 * this keeps parsing it a sub-second, single-arena affair.
 *
 * Lookups are linear walks.  That is correct for config-shaped access; callers
 * that need random access into a huge object (the BPE vocab) iterate members
 * once and build their own index. */

typedef enum {
    QJ_NULL = 0,
    QJ_FALSE,
    QJ_TRUE,
    QJ_NUMBER,
    QJ_STRING,
    QJ_ARRAY,
    QJ_OBJECT,
} qj_type;

typedef struct {
    uint8_t  type;
    uint32_t next;              /* sibling index, 0 = end of list */
    uint32_t key_off, key_len;  /* member name, for object children */
    union {
        double   num;
        struct { uint32_t off, len; } str;
        struct { uint32_t first, count; } list;
    } u;
} qj_node;

typedef struct {
    char    *text;   /* owned, mutable, unescaped in place */
    size_t   len;
    qj_node *nodes;  /* nodes[0] is a reserved NULL sentinel; root is nodes[1] */
    uint32_t count, cap;
    char     err[128];
} qj_doc;

/* Parses `len` bytes, taking a private copy.  Returns false and fills doc->err
 * on failure.  Always call qj_free() on the doc afterwards. */
bool qj_parse(qj_doc *doc, const char *text, size_t len);
bool qj_parse_file(qj_doc *doc, const char *path);
void qj_free(qj_doc *doc);

static inline const qj_node *qj_root(const qj_doc *d) {
    return d->count > 1 ? &d->nodes[1] : &d->nodes[0];
}
static inline const qj_node *qj_at(const qj_doc *d, uint32_t idx) {
    return &d->nodes[idx < d->count ? idx : 0];
}
static inline const char *qj_str(const qj_doc *d, const qj_node *n) {
    return (n && n->type == QJ_STRING) ? d->text + n->u.str.off : NULL;
}
static inline uint32_t qj_strlen(const qj_node *n) {
    return (n && n->type == QJ_STRING) ? n->u.str.len : 0;
}
/* The accessors below all tolerate a NULL node, because the idiomatic use is to
 * chain them straight off qj_get(), which returns NULL for an absent key:
 *
 *     for (c = qj_first(d, qj_get(d, root, "tools")); c; c = qj_next(d, c))
 *
 * An absent optional field is ordinary, not an error, so it must not fault. */
static inline uint32_t qj_count(const qj_node *n) {
    if (!n) return 0;
    return (n->type == QJ_ARRAY || n->type == QJ_OBJECT) ? n->u.list.count : 0;
}
/* First child of an array/object, or NULL.  Walk with qj_next(). */
static inline const qj_node *qj_first(const qj_doc *d, const qj_node *n) {
    if (!n || (n->type != QJ_ARRAY && n->type != QJ_OBJECT)) return NULL;
    return n->u.list.first ? &d->nodes[n->u.list.first] : NULL;
}
static inline const qj_node *qj_next(const qj_doc *d, const qj_node *n) {
    return (n && n->next) ? &d->nodes[n->next] : NULL;
}

/* Object member by name, or NULL. */
const qj_node *qj_get(const qj_doc *d, const qj_node *obj, const char *key);
/* Array element by position, or NULL. */
const qj_node *qj_idx(const qj_doc *d, const qj_node *arr, uint32_t i);

/* Typed accessors with defaults; `path` is a dotted key path ("a.b.c"). */
const qj_node *qj_path(const qj_doc *d, const qj_node *obj, const char *path);
double  qj_num_or (const qj_doc *d, const qj_node *obj, const char *path, double dflt);
int64_t qj_int_or (const qj_doc *d, const qj_node *obj, const char *path, int64_t dflt);
bool    qj_bool_or(const qj_doc *d, const qj_node *obj, const char *path, bool dflt);
/* Copies at most cap-1 bytes plus NUL into out; returns false if absent. */
bool    qj_str_copy(const qj_doc *d, const qj_node *obj, const char *path,
                    char *out, size_t cap);

/* True if the node's string equals a NUL-terminated C string. */
bool qj_str_eq(const qj_doc *d, const qj_node *n, const char *s);

#endif /* QWASAR_JSON_H */
