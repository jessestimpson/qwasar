/* Byte-level BPE tokenizer.
 *
 * Qwen3.8 uses the GPT-2 byte-level scheme: every byte is first mapped to a
 * printable codepoint so the vocabulary contains no control characters, then
 * BPE runs over those codepoints.  Decoding therefore has to undo that mapping,
 * which is why token strings are converted back to raw bytes once at load
 * rather than on every emitted token.
 *
 * The 33 added tokens (<|im_start|>, <think>, ...) bypass the byte mapping
 * entirely -- their `content` is literal text -- and are flagged so callers can
 * suppress them from display without string-matching. */

#include "qwasar.h"
#include "qwasar_json.h"

#include <stdlib.h>
#include <string.h>

struct qwasar_tokenizer {
    char     *arena;      /* raw bytes of every token, back to back */
    size_t    arena_len;
    uint32_t *off;        /* id -> offset into arena */
    uint32_t *len;        /* id -> byte length */
    uint8_t  *special;    /* id -> is a control token */
    int32_t   n;
};

/* GPT-2's byte-to-codepoint map keeps the printable ASCII and Latin-1 ranges
 * as themselves and pushes the remaining 68 bytes up into U+0100.. .  We only
 * ever need the inverse. */
static void qw_build_byte_decoder(int16_t *cp_to_byte /* [512] */) {
    for (int i = 0; i < 512; i++) cp_to_byte[i] = -1;

    bool direct[256] = { false };
    for (int b = '!'; b <= '~'; b++) direct[b] = true;
    for (int b = 0xA1; b <= 0xAC; b++) direct[b] = true;
    for (int b = 0xAE; b <= 0xFF; b++) direct[b] = true;

    for (int b = 0; b < 256; b++) if (direct[b]) cp_to_byte[b] = (int16_t)b;
    int n = 0;
    for (int b = 0; b < 256; b++)
        if (!direct[b]) cp_to_byte[256 + n++] = (int16_t)b;
}

/* Decodes one UTF-8 codepoint; returns bytes consumed, 0 on malformed input. */
static int qw_utf8_next(const char *s, size_t len, uint32_t *cp) {
    unsigned char c = (unsigned char)s[0];
    if (c < 0x80)                  { *cp = c; return 1; }
    if ((c & 0xE0) == 0xC0 && len >= 2) {
        *cp = ((uint32_t)(c & 0x1F) << 6) | ((unsigned char)s[1] & 0x3F);
        return 2;
    }
    if ((c & 0xF0) == 0xE0 && len >= 3) {
        *cp = ((uint32_t)(c & 0x0F) << 12) | (((unsigned char)s[1] & 0x3F) << 6)
            | ((unsigned char)s[2] & 0x3F);
        return 3;
    }
    if ((c & 0xF8) == 0xF0 && len >= 4) {
        *cp = ((uint32_t)(c & 0x07) << 18) | (((unsigned char)s[1] & 0x3F) << 12)
            | (((unsigned char)s[2] & 0x3F) << 6) | ((unsigned char)s[3] & 0x3F);
        return 4;
    }
    return 0;
}

typedef struct { char *p; size_t used, cap; } qw_bytes;

static bool qw_bytes_reserve(qw_bytes *b, size_t extra) {
    if (b->used + extra <= b->cap) return true;
    size_t cap = b->cap ? b->cap * 2 : (1u << 20);
    while (cap < b->used + extra) cap *= 2;
    char *p = realloc(b->p, cap);
    if (!p) return false;
    b->p = p;
    b->cap = cap;
    return true;
}

qwasar_tokenizer *qwasar_tokenizer_load(const char *model_path, char *err, size_t errcap) {
    char path[1200];
    snprintf(path, sizeof path, "%s/tokenizer.json", model_path);

    qj_doc d;
    if (!qj_parse_file(&d, path)) {
        snprintf(err, errcap, "cannot read tokenizer.json: %s", d.err);
        qj_free(&d);
        return NULL;
    }

    const qj_node *root  = qj_root(&d);
    const qj_node *vocab = qj_path(&d, root, "model.vocab");
    const qj_node *added = qj_get(&d, root, "added_tokens");
    if (!vocab) {
        snprintf(err, errcap, "tokenizer.json has no model.vocab");
        qj_free(&d);
        return NULL;
    }

    /* One pass to find the highest id, so the tables can be sized exactly. */
    int32_t max_id = -1;
    for (const qj_node *m = qj_first(&d, vocab); m; m = qj_next(&d, m))
        if (m->type == QJ_NUMBER && (int32_t)m->u.num > max_id) max_id = (int32_t)m->u.num;
    for (const qj_node *m = qj_first(&d, added); m; m = qj_next(&d, m)) {
        const qj_node *id = qj_get(&d, m, "id");
        if (id && (int32_t)id->u.num > max_id) max_id = (int32_t)id->u.num;
    }
    if (max_id < 0) {
        snprintf(err, errcap, "tokenizer.json vocabulary is empty");
        qj_free(&d);
        return NULL;
    }

    qwasar_tokenizer *t = calloc(1, sizeof *t);
    if (!t) { qj_free(&d); snprintf(err, errcap, "out of memory"); return NULL; }
    t->n = max_id + 1;
    t->off     = calloc((size_t)t->n, sizeof *t->off);
    t->len     = calloc((size_t)t->n, sizeof *t->len);
    t->special = calloc((size_t)t->n, sizeof *t->special);
    if (!t->off || !t->len || !t->special) {
        qj_free(&d); qwasar_tokenizer_free(t);
        snprintf(err, errcap, "out of memory");
        return NULL;
    }

    int16_t cp_to_byte[512];
    qw_build_byte_decoder(cp_to_byte);

    qw_bytes arena = { NULL, 0, 0 };

    for (const qj_node *m = qj_first(&d, vocab); m; m = qj_next(&d, m)) {
        if (m->type != QJ_NUMBER) continue;
        int32_t id = (int32_t)m->u.num;
        if (id < 0 || id >= t->n) continue;

        const char *key = d.text + m->key_off;
        size_t klen = m->key_len;
        if (!qw_bytes_reserve(&arena, klen + 1)) goto oom;

        size_t start = arena.used;
        for (size_t i = 0; i < klen; ) {
            uint32_t cp;
            int adv = qw_utf8_next(key + i, klen - i, &cp);
            if (adv == 0) { i++; continue; }
            i += (size_t)adv;
            /* Anything outside the byte-level range cannot appear in a
             * well-formed byte-level vocabulary; drop it rather than emitting
             * a byte we cannot justify. */
            if (cp < 512 && cp_to_byte[cp] >= 0)
                arena.p[arena.used++] = (char)(unsigned char)cp_to_byte[cp];
        }
        t->off[id] = (uint32_t)start;
        t->len[id] = (uint32_t)(arena.used - start);
    }

    /* Added tokens override: their content is literal, not byte-level encoded. */
    for (const qj_node *m = qj_first(&d, added); m; m = qj_next(&d, m)) {
        const qj_node *id = qj_get(&d, m, "id");
        const qj_node *content = qj_get(&d, m, "content");
        if (!id || !content || content->type != QJ_STRING) continue;
        int32_t i = (int32_t)id->u.num;
        if (i < 0 || i >= t->n) continue;

        size_t clen = content->u.str.len;
        if (!qw_bytes_reserve(&arena, clen)) goto oom;
        memcpy(arena.p + arena.used, d.text + content->u.str.off, clen);
        t->off[i] = (uint32_t)arena.used;
        t->len[i] = (uint32_t)clen;
        t->special[i] = 1;
        arena.used += clen;
    }

    t->arena = arena.p;
    t->arena_len = arena.used;
    qj_free(&d);
    return t;

oom:
    free(arena.p);
    qj_free(&d);
    qwasar_tokenizer_free(t);
    snprintf(err, errcap, "out of memory building the vocabulary");
    return NULL;
}

void qwasar_tokenizer_free(qwasar_tokenizer *t) {
    if (!t) return;
    free(t->arena);
    free(t->off);
    free(t->len);
    free(t->special);
    free(t);
}

int32_t qwasar_tokenizer_size(const qwasar_tokenizer *t) { return t ? t->n : 0; }

const char *qwasar_token_bytes(const qwasar_tokenizer *t, int32_t id,
                               size_t *len, bool *special) {
    if (!t || id < 0 || id >= t->n) { if (len) *len = 0; return NULL; }
    if (len) *len = t->len[id];
    if (special) *special = t->special[id] != 0;
    return t->arena + t->off[id];
}
