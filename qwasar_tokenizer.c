/* Byte-level BPE tokenizer.
 *
 * Qwen3.8 uses the GPT-2 byte-level scheme: every byte is first mapped to a
 * printable codepoint so the vocabulary contains no control characters, then
 * BPE merges run over those codepoints.  qwasar undoes that mapping once at
 * load and works in raw bytes throughout, which makes decoding a memcpy and
 * lets encoding hash on the bytes directly.
 *
 * Encoding is three stages:
 *
 *   1. pre-tokenize    split text on the GPT-4 pattern (qw_pretokenize)
 *   2. seed            one symbol per byte, each already a vocabulary entry
 *   3. merge           repeatedly apply the lowest-ranked adjacent merge
 *
 * Stage 3 works in token-id space rather than on strings.  Every BPE merge
 * result is itself a vocabulary entry, so a merge is a lookup on a pair of ids
 * instead of a string concatenation and hash.
 *
 * Known gap: tokenizer.json specifies an NFC normalizer, which is not applied.
 * For ASCII and already-normalised text -- which is nearly everything -- it is
 * a no-op; decomposed input would tokenize differently from the reference. */

#include "qwasar.h"
#include "qwasar_json.h"

#include <stdlib.h>
#include <string.h>

#include "qwasar_unicode.inc"   /* qw_letter_ranges / qw_number_ranges / qw_space_ranges */

/* ---- codepoint classes ----------------------------------------------------- */

static bool qw_in_ranges(const qw_cp_range *r, size_t n, uint32_t cp) {
    size_t lo = 0, hi = n;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (cp < r[mid].lo)      hi = mid;
        else if (cp > r[mid].hi) lo = mid + 1;
        else return true;
    }
    return false;
}

static bool qw_is_letter(uint32_t cp) {
    return qw_in_ranges(qw_letter_ranges, QW_LETTER_COUNT, cp);
}
static bool qw_is_number(uint32_t cp) {
    return qw_in_ranges(qw_number_ranges, QW_NUMBER_COUNT, cp);
}
static bool qw_is_space(uint32_t cp) {
    return qw_in_ranges(qw_space_ranges, QW_SPACE_COUNT, cp);
}

/* Decodes one UTF-8 codepoint; returns bytes consumed, 0 on malformed input. */
static int qw_utf8_next(const char *s, size_t len, uint32_t *cp) {
    unsigned char c = (unsigned char)s[0];
    if (c < 0x80)                       { *cp = c; return 1; }
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

/* ---- tables ---------------------------------------------------------------- */

typedef struct { uint32_t hash_next; } qw_unused;

typedef struct {
    uint64_t key;      /* (uint64)left_id << 32 | right_id */
    uint32_t rank;
    int32_t  merged;
    bool     used;
} qw_merge_slot;

struct qwasar_tokenizer {
    char     *arena;      /* raw bytes of every token, back to back */
    uint32_t *off;
    uint32_t *len;
    uint8_t  *special;
    int32_t   n;

    /* raw bytes -> id, for non-special tokens only */
    int32_t  *vhash;      /* slot -> id+1, 0 = empty */
    uint32_t  vhash_cap;

    qw_merge_slot *merges;
    uint32_t       merge_cap;

    int32_t byte_token[256];   /* single-byte token per byte value */
};

static uint64_t qw_fnv(const char *s, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; i++) { h ^= (unsigned char)s[i]; h *= 1099511628211ull; }
    return h;
}

static void qw_vhash_put(qwasar_tokenizer *t, int32_t id) {
    uint32_t m = (uint32_t)(qw_fnv(t->arena + t->off[id], t->len[id]) & (t->vhash_cap - 1));
    while (t->vhash[m]) m = (m + 1) & (t->vhash_cap - 1);
    t->vhash[m] = id + 1;
}

static int32_t qw_vhash_get(const qwasar_tokenizer *t, const char *s, size_t n) {
    if (!t->vhash_cap) return -1;
    uint32_t m = (uint32_t)(qw_fnv(s, n) & (t->vhash_cap - 1));
    while (t->vhash[m]) {
        int32_t id = t->vhash[m] - 1;
        if (t->len[id] == n && !memcmp(t->arena + t->off[id], s, n)) return id;
        m = (m + 1) & (t->vhash_cap - 1);
    }
    return -1;
}

static void qw_merge_put(qwasar_tokenizer *t, int32_t a, int32_t b,
                         uint32_t rank, int32_t merged) {
    uint64_t key = ((uint64_t)(uint32_t)a << 32) | (uint32_t)b;
    uint64_t h = key * 1099511628211ull;
    uint32_t m = (uint32_t)((h ^ (h >> 29)) & (t->merge_cap - 1));
    while (t->merges[m].used) {
        if (t->merges[m].key == key) return;      /* first rank wins */
        m = (m + 1) & (t->merge_cap - 1);
    }
    t->merges[m].key = key;
    t->merges[m].rank = rank;
    t->merges[m].merged = merged;
    t->merges[m].used = true;
}

static const qw_merge_slot *qw_merge_get(const qwasar_tokenizer *t, int32_t a, int32_t b) {
    if (!t->merge_cap || a < 0 || b < 0) return NULL;
    uint64_t key = ((uint64_t)(uint32_t)a << 32) | (uint32_t)b;
    uint64_t h = key * 1099511628211ull;
    uint32_t m = (uint32_t)((h ^ (h >> 29)) & (t->merge_cap - 1));
    while (t->merges[m].used) {
        if (t->merges[m].key == key) return &t->merges[m];
        m = (m + 1) & (t->merge_cap - 1);
    }
    return NULL;
}

/* ---- load ------------------------------------------------------------------ */

/* GPT-2's byte-to-codepoint map keeps printable ASCII and Latin-1 as themselves
 * and pushes the remaining 68 bytes up into U+0100..  We need the inverse. */
static void qw_build_byte_decoder(int16_t *cp_to_byte /* [512] */) {
    for (int i = 0; i < 512; i++) cp_to_byte[i] = -1;
    bool direct[256] = { false };
    for (int b = '!';  b <= '~';  b++) direct[b] = true;
    for (int b = 0xA1; b <= 0xAC; b++) direct[b] = true;
    for (int b = 0xAE; b <= 0xFF; b++) direct[b] = true;
    for (int b = 0; b < 256; b++) if (direct[b]) cp_to_byte[b] = (int16_t)b;
    int n = 0;
    for (int b = 0; b < 256; b++) if (!direct[b]) cp_to_byte[256 + n++] = (int16_t)b;
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

/* Byte-level decode of a vocabulary key into raw bytes appended to `out`. */
static size_t qw_bytelevel_decode(const char *s, size_t n, const int16_t *cp_to_byte,
                                  char *out) {
    size_t w = 0;
    for (size_t i = 0; i < n; ) {
        uint32_t cp;
        int adv = qw_utf8_next(s + i, n - i, &cp);
        if (adv == 0) { i++; continue; }
        i += (size_t)adv;
        if (cp < 512 && cp_to_byte[cp] >= 0) out[w++] = (char)(unsigned char)cp_to_byte[cp];
    }
    return w;
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

    const qj_node *root   = qj_root(&d);
    const qj_node *vocab  = qj_path(&d, root, "model.vocab");
    const qj_node *merges = qj_path(&d, root, "model.merges");
    const qj_node *added  = qj_get(&d, root, "added_tokens");
    if (!vocab) {
        snprintf(err, errcap, "tokenizer.json has no model.vocab");
        qj_free(&d);
        return NULL;
    }

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
    for (int i = 0; i < 256; i++) t->byte_token[i] = -1;

    t->vhash_cap = 1;
    while (t->vhash_cap < (uint32_t)t->n * 2) t->vhash_cap <<= 1;
    t->vhash = calloc(t->vhash_cap, sizeof *t->vhash);

    if (!t->off || !t->len || !t->special || !t->vhash) {
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

        size_t klen = m->key_len;
        if (!qw_bytes_reserve(&arena, klen + 1)) goto oom;
        size_t start = arena.used;
        arena.used += qw_bytelevel_decode(d.text + m->key_off, klen, cp_to_byte,
                                          arena.p + start);
        t->off[id] = (uint32_t)start;
        t->len[id] = (uint32_t)(arena.used - start);
    }
    t->arena = arena.p;

    /* Index the byte-level vocabulary for encoding.  Added tokens are excluded:
     * their content is literal text, so hashing it would make a control marker
     * reachable from ordinary input. */
    for (int32_t id = 0; id < t->n; id++) {
        if (t->len[id] == 0) continue;
        qw_vhash_put(t, id);
        if (t->len[id] == 1) t->byte_token[(unsigned char)t->arena[t->off[id]]] = id;
    }

    /* Merges, keyed by the pair of ids they join. */
    uint32_t n_merges = merges ? qj_count(merges) : 0;
    if (n_merges) {
        t->merge_cap = 1;
        while (t->merge_cap < n_merges * 2) t->merge_cap <<= 1;
        t->merges = calloc(t->merge_cap, sizeof *t->merges);
        if (!t->merges) goto oom;

        char lbuf[512], rbuf[512], both[1024];
        uint32_t rank = 0;
        for (const qj_node *m = qj_first(&d, merges); m; m = qj_next(&d, m), rank++) {
            const qj_node *l, *r;
            if (m->type == QJ_ARRAY) {
                l = qj_idx(&d, m, 0);
                r = qj_idx(&d, m, 1);
            } else continue;
            if (!l || !r || l->type != QJ_STRING || r->type != QJ_STRING) continue;
            if (l->u.str.len >= sizeof lbuf || r->u.str.len >= sizeof rbuf) continue;

            size_t ln = qw_bytelevel_decode(d.text + l->u.str.off, l->u.str.len,
                                            cp_to_byte, lbuf);
            size_t rn = qw_bytelevel_decode(d.text + r->u.str.off, r->u.str.len,
                                            cp_to_byte, rbuf);
            if (ln + rn >= sizeof both) continue;
            memcpy(both, lbuf, ln);
            memcpy(both + ln, rbuf, rn);

            int32_t a = qw_vhash_get(t, lbuf, ln);
            int32_t b = qw_vhash_get(t, rbuf, rn);
            int32_t c = qw_vhash_get(t, both, ln + rn);
            if (a >= 0 && b >= 0 && c >= 0) qw_merge_put(t, a, b, rank, c);
        }
    }

    /* Added tokens last: they replace the decoded text for their ids without
     * entering the encode index. */
    for (const qj_node *m = qj_first(&d, added); m; m = qj_next(&d, m)) {
        const qj_node *id = qj_get(&d, m, "id");
        const qj_node *content = qj_get(&d, m, "content");
        if (!id || !content || content->type != QJ_STRING) continue;
        int32_t i = (int32_t)id->u.num;
        if (i < 0 || i >= t->n) continue;

        size_t clen = content->u.str.len;
        if (!qw_bytes_reserve(&arena, clen)) goto oom;
        t->arena = arena.p;
        memcpy(arena.p + arena.used, d.text + content->u.str.off, clen);
        t->off[i] = (uint32_t)arena.used;
        t->len[i] = (uint32_t)clen;
        t->special[i] = 1;
        arena.used += clen;
    }
    t->arena = arena.p;

    qj_free(&d);
    return t;

oom:
    free(arena.p);
    t->arena = NULL;
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
    free(t->vhash);
    free(t->merges);
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

int32_t qwasar_token_id(const qwasar_tokenizer *t, const char *literal) {
    if (!t || !literal) return -1;
    size_t n = strlen(literal);
    for (int32_t id = 0; id < t->n; id++)
        if (t->special[id] && t->len[id] == n && !memcmp(t->arena + t->off[id], literal, n))
            return id;
    return -1;
}

/* ---- pre-tokenizer ---------------------------------------------------------
 *
 * The split pattern from tokenizer.json, as a state machine:
 *
 *   (?i:'s|'t|'re|'ve|'m|'ll|'d)
 *   | [^\r\n\p{L}\p{N}]?\p{L}+
 *   | \p{N}
 *   |  ?[^\s\p{L}\p{N}]+[\r\n]*
 *   | \s*[\r\n]+
 *   | \s+(?!\S)
 *   | \s+
 *
 * Alternation is first-match, so the order below is the order above.  Note that
 * \p{N} has no quantifier: digits are emitted one at a time, which is what
 * makes this Qwen's variant rather than the more common \p{N}{1,3}. */

typedef struct { const char *s; size_t len, pos; } qw_scan;

static uint32_t qw_peek(const qw_scan *sc, size_t at, int *adv) {
    uint32_t cp = 0;
    if (at >= sc->len) { *adv = 0; return 0; }
    *adv = qw_utf8_next(sc->s + at, sc->len - at, &cp);
    if (*adv == 0) { *adv = 1; return 0xFFFD; }
    return cp;
}

static bool qw_ci_match(const char *s, size_t len, size_t pos, const char *lit) {
    size_t n = strlen(lit);
    if (pos + n > len) return false;
    for (size_t i = 0; i < n; i++) {
        char a = s[pos + i], b = lit[i];
        if (a >= 'A' && a <= 'Z') a = (char)(a - 'A' + 'a');
        if (a != b) return false;
    }
    return true;
}

/* Returns the end offset of the piece starting at sc->pos. */
static size_t qw_next_piece(qw_scan *sc) {
    const char *s = sc->s;
    const size_t len = sc->len, p = sc->pos;
    int adv;

    /* 1. contractions */
    if (s[p] == '\'') {
        static const char *const forms[] = { "'re", "'ve", "'ll", "'s", "'t", "'m", "'d" };
        for (size_t i = 0; i < sizeof forms / sizeof *forms; i++)
            if (qw_ci_match(s, len, p, forms[i])) return p + strlen(forms[i]);
    }

    /* 2. [^\r\n\p{L}\p{N}]? \p{L}+ */
    {
        size_t q = p;
        uint32_t cp = qw_peek(sc, q, &adv);
        if (adv && cp != '\r' && cp != '\n' && !qw_is_letter(cp) && !qw_is_number(cp))
            q += (size_t)adv;   /* optional single leading character */
        size_t letters = q;
        for (;;) {
            uint32_t c2 = qw_peek(sc, letters, &adv);
            if (!adv || !qw_is_letter(c2)) break;
            letters += (size_t)adv;
        }
        if (letters > q) return letters;   /* needs at least one letter */
    }

    /* 3. \p{N} -- exactly one */
    {
        uint32_t cp = qw_peek(sc, p, &adv);
        if (adv && qw_is_number(cp)) return p + (size_t)adv;
    }

    /* 4. " ?" [^\s\p{L}\p{N}]+ [\r\n]* */
    {
        size_t q = p;
        if (s[q] == ' ') q++;
        size_t body = q;
        for (;;) {
            uint32_t cp = qw_peek(sc, body, &adv);
            if (!adv || qw_is_space(cp) || qw_is_letter(cp) || qw_is_number(cp)) break;
            body += (size_t)adv;
        }
        if (body > q) {
            while (body < len && (s[body] == '\r' || s[body] == '\n')) body++;
            return body;
        }
    }

    /* Everything below consumes whitespace; find the run once. */
    size_t ws = p;
    for (;;) {
        uint32_t cp = qw_peek(sc, ws, &adv);
        if (!adv || !qw_is_space(cp)) break;
        ws += (size_t)adv;
    }

    /* 5. \s* [\r\n]+  -- greedy \s* backs off to the last \r\n run in the span,
     *    so a run of blank lines stays with its indentation. */
    if (ws > p) {
        size_t last_nl = (size_t)-1;
        for (size_t i = p; i < ws; i++)
            if (s[i] == '\r' || s[i] == '\n') last_nl = i;
        if (last_nl != (size_t)-1) return last_nl + 1;
    }

    /* 6. \s+(?!\S) -- a whitespace run keeps its final character for the next
     *    piece, unless the run reaches the end of the text. */
    if (ws > p) {
        if (ws == len) return ws;
        size_t back = ws;
        while (back > p) {                       /* step back one codepoint */
            back--;
            if (((unsigned char)s[back] & 0xC0) != 0x80) break;
        }
        if (back > p) return back;
    }

    /* 7. \s+ */
    if (ws > p) return ws;

    /* Nothing matched (malformed byte): consume one so scanning terminates. */
    return p + 1;
}

/* ---- encode ---------------------------------------------------------------- */

typedef struct { int32_t *v; int32_t n, cap; } qw_tokvec;

static bool qw_tokvec_push(qw_tokvec *t, int32_t id) {
    if (t->n == t->cap) {
        int32_t cap = t->cap ? t->cap * 2 : 64;
        int32_t *v = realloc(t->v, (size_t)cap * sizeof *v);
        if (!v) return false;
        t->v = v;
        t->cap = cap;
    }
    t->v[t->n++] = id;
    return true;
}

/* Applies BPE to one pre-tokenized piece and appends the resulting ids. */
static bool qw_bpe_piece(const qwasar_tokenizer *t, const char *s, size_t n,
                         qw_tokvec *out) {
    if (n == 0) return true;

    /* A whole piece is very often a single vocabulary entry; skip the merge
     * loop when it is. */
    int32_t whole = qw_vhash_get(t, s, n);
    if (whole >= 0) return qw_tokvec_push(out, whole);

    int32_t *sym = malloc(n * sizeof *sym);
    if (!sym) return false;
    int32_t m = 0;
    for (size_t i = 0; i < n; i++) {
        int32_t id = t->byte_token[(unsigned char)s[i]];
        /* Byte-level vocabularies contain every single byte, so this cannot
         * normally fail; skipping is better than emitting a wrong id. */
        if (id >= 0) sym[m++] = id;
    }

    for (;;) {
        uint32_t best_rank = 0xFFFFFFFFu;
        int32_t best_at = -1, best_id = -1;
        for (int32_t i = 0; i + 1 < m; i++) {
            const qw_merge_slot *slot = qw_merge_get(t, sym[i], sym[i + 1]);
            if (slot && slot->rank < best_rank) {
                best_rank = slot->rank;
                best_at = i;
                best_id = slot->merged;
            }
        }
        if (best_at < 0) break;
        sym[best_at] = best_id;
        memmove(sym + best_at + 1, sym + best_at + 2,
                (size_t)(m - best_at - 2) * sizeof *sym);
        m--;
    }

    bool ok = true;
    for (int32_t i = 0; i < m && ok; i++) ok = qw_tokvec_push(out, sym[i]);
    free(sym);
    return ok;
}

static bool qw_encode_into(const qwasar_tokenizer *t, const char *text, size_t len,
                           qw_tokvec *out) {
    qw_scan sc = { text, len, 0 };
    while (sc.pos < len) {
        size_t end = qw_next_piece(&sc);
        if (end <= sc.pos) end = sc.pos + 1;
        if (!qw_bpe_piece(t, text + sc.pos, end - sc.pos, out)) return false;
        sc.pos = end;
    }
    return true;
}

int32_t *qwasar_encode(const qwasar_tokenizer *t, const char *text, int32_t *out_n) {
    qw_tokvec v = { NULL, 0, 0 };
    if (!t || !text) { *out_n = 0; return NULL; }
    if (!qw_encode_into(t, text, strlen(text), &v)) { free(v.v); *out_n = 0; return NULL; }
    *out_n = v.n;
    return v.v;
}

/* ---- chat template ---------------------------------------------------------
 *
 * A C rendering of the model's ChatML template.  Two behaviours are specific to
 * Qwen3.8 and easy to miss:
 *
 *   - Reasoning is on by default, and the generation prompt therefore ends with
 *     an open <think>.  Disabling it does not remove the block; it emits an
 *     empty one, because the model expects the marker either way.
 *   - A system message carrying the reasoning-effort instruction is synthesised
 *     even when the caller supplies none.
 *
 * Control tokens are emitted by id and never round-tripped through text, so
 * message content cannot introduce a role marker however it is written. */

static const char *QW_REASONING_XHIGH =
    "Reasoning effort is set to xhigh. Please think carefully through the task, "
    "validate key assumptions, consider plausible alternatives, and prioritize "
    "correctness, consistency, and clarity in the final answer.";
static const char *QW_REASONING_LOW =
    "Reasoning effort is set to low. Keep your thinking brief and focused, "
    "moving directly to the conclusion without unnecessary elaboration.";

static const char *qw_reasoning_text(const qwasar_chat_options *o) {
    if (!o->enable_thinking) return NULL;
    if (!o->reasoning_effort || !strcmp(o->reasoning_effort, "xhigh")) return QW_REASONING_XHIGH;
    if (!strcmp(o->reasoning_effort, "low")) return QW_REASONING_LOW;
    return NULL;    /* "medium" carries no instruction */
}

/* Trims ASCII whitespace, matching the template's |trim. */
static void qw_trim(const char *s, size_t *off, size_t *len) {
    size_t a = 0, b = s ? strlen(s) : 0;
    while (a < b && (s[a] == ' ' || s[a] == '\t' || s[a] == '\n' || s[a] == '\r')) a++;
    while (b > a && (s[b-1] == ' ' || s[b-1] == '\t' || s[b-1] == '\n' || s[b-1] == '\r')) b--;
    *off = a;
    *len = b - a;
}

typedef struct {
    const qwasar_tokenizer *t;
    qw_tokvec v;
    bool ok;
    int32_t im_start, im_end, think_open, think_close, tr_open, tr_close;
} qw_chat;

static void qw_put_id(qw_chat *c, int32_t id) {
    if (!c->ok) return;
    if (id < 0) { c->ok = false; return; }
    c->ok = qw_tokvec_push(&c->v, id);
}

static void qw_put_text(qw_chat *c, const char *s, size_t len) {
    if (!c->ok || !s || !len) return;
    c->ok = qw_encode_into(c->t, s, len, &c->v);
}

static void qw_put_str(qw_chat *c, const char *s) { qw_put_text(c, s, s ? strlen(s) : 0); }

int32_t *qwasar_apply_chat_template(const qwasar_tokenizer *t,
                                    const qwasar_message *msgs, int32_t n_msgs,
                                    const qwasar_chat_options *opts,
                                    int32_t *out_n, char *err, size_t errcap) {
    qwasar_chat_options defaults = { .enable_thinking = true,
                                     .reasoning_effort = "xhigh",
                                     .add_generation_prompt = true };
    if (!opts) opts = &defaults;
    *out_n = 0;

    if (!t || !msgs || n_msgs <= 0) {
        snprintf(err, errcap, "no messages to render");
        return NULL;
    }

    qw_chat c = { .t = t, .ok = true };
    c.im_start    = qwasar_token_id(t, "<|im_start|>");
    c.im_end      = qwasar_token_id(t, "<|im_end|>");
    c.think_open  = qwasar_token_id(t, "<think>");
    c.think_close = qwasar_token_id(t, "</think>");
    c.tr_open     = qwasar_token_id(t, "<tool_response>");
    c.tr_close    = qwasar_token_id(t, "</tool_response>");
    if (c.im_start < 0 || c.im_end < 0 || c.think_open < 0 || c.think_close < 0) {
        snprintf(err, errcap, "tokenizer is missing ChatML control tokens");
        return NULL;
    }

    const char *reasoning = qw_reasoning_text(opts);
    const bool has_system = !strcmp(msgs[0].role, "system");

    size_t soff = 0, slen = 0;
    if (has_system) qw_trim(msgs[0].content, &soff, &slen);

    if (slen || reasoning) {
        qw_put_id(&c, c.im_start);
        qw_put_str(&c, "system\n");
        if (reasoning) {
            qw_put_str(&c, reasoning);
            if (slen) qw_put_str(&c, "\n\n");
        }
        if (slen) qw_put_text(&c, msgs[0].content + soff, slen);
        qw_put_id(&c, c.im_end);
        qw_put_str(&c, "\n");
    }

    for (int32_t i = 0; i < n_msgs; i++) {
        const qwasar_message *m = &msgs[i];
        size_t off, len;
        qw_trim(m->content, &off, &len);

        if (!strcmp(m->role, "system")) {
            if (i != 0) {
                snprintf(err, errcap, "system message must come first");
                free(c.v.v);
                return NULL;
            }
            continue;   /* already emitted above */
        }

        if (!strcmp(m->role, "user")) {
            qw_put_id(&c, c.im_start);
            qw_put_str(&c, "user\n");
            qw_put_text(&c, m->content + off, len);
            qw_put_id(&c, c.im_end);
            qw_put_str(&c, "\n");
        } else if (!strcmp(m->role, "assistant")) {
            size_t roff, rlen;
            qw_trim(m->reasoning, &roff, &rlen);
            qw_put_id(&c, c.im_start);
            qw_put_str(&c, "assistant\n");
            qw_put_id(&c, c.think_open);
            qw_put_str(&c, "\n");
            if (rlen) qw_put_text(&c, m->reasoning + roff, rlen);
            qw_put_str(&c, "\n");
            qw_put_id(&c, c.think_close);
            qw_put_str(&c, "\n\n");
            qw_put_text(&c, m->content + off, len);
            qw_put_id(&c, c.im_end);
            qw_put_str(&c, "\n");
        } else if (!strcmp(m->role, "tool")) {
            /* Consecutive tool results share one user turn. */
            const bool first = (i == 0) || strcmp(msgs[i-1].role, "tool");
            const bool last  = (i + 1 == n_msgs) || strcmp(msgs[i+1].role, "tool");
            if (first) { qw_put_id(&c, c.im_start); qw_put_str(&c, "user"); }
            qw_put_str(&c, "\n");
            qw_put_id(&c, c.tr_open);
            qw_put_str(&c, "\n");
            qw_put_text(&c, m->content + off, len);
            qw_put_str(&c, "\n");
            qw_put_id(&c, c.tr_close);
            if (last) { qw_put_id(&c, c.im_end); qw_put_str(&c, "\n"); }
        } else {
            snprintf(err, errcap, "unknown message role '%s'", m->role);
            free(c.v.v);
            return NULL;
        }
    }

    if (opts->add_generation_prompt) {
        qw_put_id(&c, c.im_start);
        qw_put_str(&c, "assistant\n");
        qw_put_id(&c, c.think_open);
        if (opts->enable_thinking) {
            qw_put_str(&c, "\n");
        } else {
            /* The marker still has to be there; it is just closed immediately. */
            qw_put_str(&c, "\n\n");
            qw_put_id(&c, c.think_close);
            qw_put_str(&c, "\n\n");
        }
    }

    if (!c.ok) {
        snprintf(err, errcap, "out of memory encoding the prompt");
        free(c.v.v);
        return NULL;
    }
    *out_n = c.v.n;
    return c.v.v;
}
