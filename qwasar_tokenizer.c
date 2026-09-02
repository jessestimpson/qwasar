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
 * NFC is applied when tokenizer.json declares it (both families do); see qw_nfc.
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
static bool qw_is_mark(uint32_t cp) {
    return qw_in_ranges(qw_mark_ranges, QW_MARK_COUNT, cp);
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

    /* The 33 added tokens, listed so lookups by literal do not scan a 248k
     * vocabulary. */
    int32_t  *special_ids;
    int32_t   n_special;

    int32_t byte_token[256];   /* single-byte token per byte value */

    /* Flash-Next's split regex: [\p{L}\p{M}]+ where the 27B's has \p{L}+,
     * so combining marks ride with the letters they attach to instead of
     * splitting off.  Read from tokenizer.json's own pattern, not from the
     * model family, because it is the tokenizer's fact. */
    bool marks_join;
    /* tokenizer.json declares an NFC normalizer (both families do); decomposed
     * input composes before it is split, or the ids differ. */
    bool nfc;
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
    bool marks_join = false;
    {
        const qj_node *pre = qj_path(&d, root, "pre_tokenizer.pretokenizers");
        const qj_node *first = pre ? qj_idx(&d, pre, 0) : NULL;
        const qj_node *rx = first ? qj_path(&d, first, "pattern.Regex") : NULL;
        if (rx && rx->type == QJ_STRING) {
            const char *p = d.text + rx->u.str.off;
            for (uint32_t i = 0; i + 4 <= rx->u.str.len; i++)
                if (!memcmp(p + i, "p{M}", 4)) { marks_join = true; break; }
        }
    }
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
    t->marks_join = marks_join;
    {
        char ntype[16] = "";
        const qj_node *norm = qj_get(&d, root, "normalizer");
        if (norm) qj_str_copy(&d, norm, "type", ntype, sizeof ntype);
        t->nfc = !strcmp(ntype, "NFC");
    }
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

        int32_t *ids = realloc(t->special_ids, (size_t)(t->n_special + 1) * sizeof *ids);
        if (!ids) goto oom;
        t->special_ids = ids;
        t->special_ids[t->n_special++] = i;
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
    free(t->special_ids);
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
    for (int32_t i = 0; i < t->n_special; i++) {
        int32_t id = t->special_ids[i];
        if (t->len[id] == n && !memcmp(t->arena + t->off[id], literal, n)) return id;
    }
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

typedef struct { const char *s; size_t len, pos; bool marks; } qw_scan;

/* A letter for the purposes of a run: \p{L}, or \p{M} too under
 * Flash-Next's pattern. */
static bool qw_run_char(const qw_scan *sc, uint32_t cp) {
    return qw_is_letter(cp) || (sc->marks && qw_is_mark(cp));
}

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

    /* 2. [^\r\n\p{L}\p{N}]? \p{L}+   -- or [\p{L}\p{M}]+ with marks joining.
     * A leading mark is then part of the run, never the optional character:
     * the regex would backtrack to the same answer. */
    {
        size_t q = p;
        uint32_t cp = qw_peek(sc, q, &adv);
        if (adv && cp != '\r' && cp != '\n' && !qw_run_char(sc, cp) && !qw_is_number(cp))
            q += (size_t)adv;   /* optional single leading character */
        size_t letters = q;
        for (;;) {
            uint32_t c2 = qw_peek(sc, letters, &adv);
            if (!adv || !qw_run_char(sc, c2)) break;
            letters += (size_t)adv;
        }
        if (letters > q) return letters;   /* needs at least one letter */
    }

    /* 3. \p{N} -- exactly one */
    {
        uint32_t cp = qw_peek(sc, p, &adv);
        if (adv && qw_is_number(cp)) return p + (size_t)adv;
    }

    /* 4. " ?" [^\s\p{L}\p{N}]+ [\r\n]*   -- marks excluded too when they join */
    {
        size_t q = p;
        if (s[q] == ' ') q++;
        size_t body = q;
        for (;;) {
            uint32_t cp = qw_peek(sc, body, &adv);
            if (!adv || qw_is_space(cp) || qw_run_char(sc, cp) || qw_is_number(cp)) break;
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


/* ---- NFC ---------------------------------------------------------------------
 *
 * Canonical decomposition, canonical ordering, canonical composition -- UAX
 * #15's three steps, over tables tools/gen_unicode.py derived from Python's
 * unicodedata (NFD strings for the decompositions, so no recursion here;
 * primary composites only, so the exclusions are already applied).  Hangul
 * syllables are arithmetic, as the standard specifies.  Compatibility forms
 * are not NFC's business and pass through. */

static const qw_nfd_entry *qw_nfd_find(uint32_t cp) {
    size_t lo = 0, hi = QW_NFD_COUNT;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (qw_nfd_index[mid].cp < cp) lo = mid + 1;
        else if (qw_nfd_index[mid].cp > cp) hi = mid;
        else return &qw_nfd_index[mid];
    }
    return NULL;
}

static uint8_t qw_ccc(uint32_t cp) {
    size_t lo = 0, hi = QW_CCC_COUNT;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (qw_ccc_table[mid].cp < cp) lo = mid + 1;
        else if (qw_ccc_table[mid].cp > cp) hi = mid;
        else return qw_ccc_table[mid].ccc;
    }
    return 0;
}

#define QW_HANGUL_S     0xAC00
#define QW_HANGUL_L     0x1100
#define QW_HANGUL_V     0x1161
#define QW_HANGUL_T     0x11A7
#define QW_HANGUL_LCNT  19
#define QW_HANGUL_VCNT  21
#define QW_HANGUL_TCNT  28
#define QW_HANGUL_NCNT  (QW_HANGUL_VCNT * QW_HANGUL_TCNT)
#define QW_HANGUL_SCNT  (QW_HANGUL_LCNT * QW_HANGUL_NCNT)

/* A starter that can be the second of a composing pair (jamo, and a few
 * others); the reason already-composed text cannot be recognised by the
 * absence of combining marks alone. */
static bool qw_composes_as_second(uint32_t cp) {
    if (cp >= QW_HANGUL_V && cp < QW_HANGUL_T + QW_HANGUL_TCNT) return true;
    size_t lo = 0, hi = QW_COMPOSE_SECOND_COUNT;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (qw_compose_second[mid] < cp) lo = mid + 1;
        else if (qw_compose_second[mid] > cp) hi = mid;
        else return true;
    }
    return false;
}

/* The primary composite of a and b, or 0. */
static uint32_t qw_compose_pair(uint32_t a, uint32_t b) {
    /* Hangul: L + V, then LV + T. */
    if (a >= QW_HANGUL_L && a < QW_HANGUL_L + QW_HANGUL_LCNT
        && b >= QW_HANGUL_V && b < QW_HANGUL_V + QW_HANGUL_VCNT)
        return QW_HANGUL_S + ((a - QW_HANGUL_L) * QW_HANGUL_VCNT + (b - QW_HANGUL_V)) * QW_HANGUL_TCNT;
    if (a >= QW_HANGUL_S && a < QW_HANGUL_S + QW_HANGUL_SCNT && (a - QW_HANGUL_S) % QW_HANGUL_TCNT == 0
        && b > QW_HANGUL_T && b < QW_HANGUL_T + QW_HANGUL_TCNT)
        return a + (b - QW_HANGUL_T);
    size_t lo = 0, hi = QW_COMPOSE_COUNT;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        const qw_compose_entry *e = &qw_compose_table[mid];
        if (e->a < a || (e->a == a && e->b < b)) lo = mid + 1;
        else if (e->a > a || e->b > b) hi = mid;
        else return e->c;
    }
    return 0;
}

static size_t qw_utf8_put(char *out, uint32_t cp) {
    if (cp < 0x80)    { out[0] = (char)cp; return 1; }
    if (cp < 0x800)   { out[0] = (char)(0xC0 | (cp >> 6)); out[1] = (char)(0x80 | (cp & 0x3F)); return 2; }
    if (cp < 0x10000) { out[0] = (char)(0xE0 | (cp >> 12)); out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
                        out[2] = (char)(0x80 | (cp & 0x3F)); return 3; }
    out[0] = (char)(0xF0 | (cp >> 18)); out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    out[2] = (char)(0x80 | ((cp >> 6) & 0x3F)); out[3] = (char)(0x80 | (cp & 0x3F)); return 4;
}

/* NFC of `s`, malloc'd; NULL if the input is already composed (the common
 * case, detected cheaply: nothing decomposable and no combining marks). */
static char *qw_nfc(const char *s, size_t len, size_t *out_len) {
    /* Decode, decomposing as we go.  Worst-case growth is a Hangul syllable
     * into three jamo or a decomposition of four codepoints. */
    size_t cap = len * 4 + 8, n = 0;
    uint32_t *d = malloc(cap * sizeof *d);
    bool touched = false;
    for (size_t i = 0; i < len; ) {
        uint32_t cp;
        int adv = qw_utf8_next(s + i, len - i, &cp);
        if (adv <= 0) { cp = 0xFFFD; adv = 1; }
        i += (size_t)adv;
        if (cp >= QW_HANGUL_S && cp < QW_HANGUL_S + QW_HANGUL_SCNT) {
            const uint32_t si = cp - QW_HANGUL_S;
            d[n++] = QW_HANGUL_L + si / QW_HANGUL_NCNT;
            d[n++] = QW_HANGUL_V + (si % QW_HANGUL_NCNT) / QW_HANGUL_TCNT;
            if (si % QW_HANGUL_TCNT) d[n++] = QW_HANGUL_T + si % QW_HANGUL_TCNT;
            touched = true;
            continue;
        }
        const qw_nfd_entry *e = qw_nfd_find(cp);
        if (e) {
            for (uint32_t k = 0; k < e->len; k++) d[n++] = qw_nfd_data[e->off + k];
            touched = true;
        } else {
            d[n++] = cp;
            if (qw_ccc(cp) || qw_composes_as_second(cp)) touched = true;
        }
    }
    if (!touched) { free(d); return NULL; }

    /* Canonical ordering: stable sort of every run of non-starters by class. */
    for (size_t i = 0; i < n; ) {
        if (!qw_ccc(d[i])) { i++; continue; }
        size_t j = i;
        while (j < n && qw_ccc(d[j])) j++;
        for (size_t a = i + 1; a < j; a++)
            for (size_t b = a; b > i && qw_ccc(d[b - 1]) > qw_ccc(d[b]); b--) {
                uint32_t t = d[b]; d[b] = d[b - 1]; d[b - 1] = t;
            }
        i = j;
    }

    /* Canonical composition, the sample algorithm from UAX #15. */
    if (n > 0) {
        size_t starter = 0, w = 1;
        uint32_t starter_cp = d[0];
        int last_class = qw_ccc(starter_cp) ? 256 : 0;
        for (size_t i = 1; i < n; i++) {
            const uint32_t ch = d[i];
            const int cls = qw_ccc(ch);
            const uint32_t comp = qw_compose_pair(starter_cp, ch);
            if (comp && (last_class < cls || last_class == 0)) {
                d[starter] = comp;
                starter_cp = comp;
            } else {
                if (cls == 0) { starter = w; starter_cp = ch; }
                last_class = cls;
                d[w++] = ch;
            }
        }
        n = w;
    }

    char *out = malloc(n * 4 + 1);
    size_t w = 0;
    for (size_t i = 0; i < n; i++) w += qw_utf8_put(out + w, d[i]);
    out[w] = 0;
    free(d);
    *out_len = w;
    return out;
}

static bool qw_encode_pieces(const qwasar_tokenizer *t, const char *text, size_t len, qw_tokvec *out);

static bool qw_encode_into(const qwasar_tokenizer *t, const char *text, size_t len,
                           qw_tokvec *out) {
    char *normed = NULL;
    if (t->nfc) {
        size_t nlen = 0;
        normed = qw_nfc(text, len, &nlen);
        if (normed) { text = normed; len = nlen; }
    }
    bool ok = qw_encode_pieces(t, text, len, out);
    free(normed);
    return ok;
}

static bool qw_encode_pieces(const qwasar_tokenizer *t, const char *text, size_t len,
                             qw_tokvec *out) {
    qw_scan sc = { text, len, 0, t->marks_join };
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
    int32_t vision_start, vision_end, image_pad, video_pad;
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

/* Longest control-token literal starting at `s`, or -1.  Every added token in
 * this vocabulary begins with '<', so the scan is cheap. */
static int32_t qw_special_at(const qwasar_tokenizer *t, const char *s, size_t *len) {
    if (*s != '<') return -1;
    int32_t best = -1;
    size_t best_len = 0;
    for (int32_t i = 0; i < t->n_special; i++) {
        const int32_t id = t->special_ids[i];
        size_t n = t->len[id];
        if (n == 0) continue;
        if (n > best_len && !strncmp(s, t->arena + t->off[id], n)) {
            best = id;
            best_len = n;
        }
    }
    if (best >= 0) *len = best_len;
    return best;
}

/* Emits one of qwasar's own template strings, mapping control-token literals to
 * their ids.
 *
 * The model's tool-format description contains "<tool_call>" and friends as
 * literal text, and in training those tokenized as control tokens -- encoding
 * them as plain text would hand the model a prompt it has never seen.  This is
 * the one place that mapping is applied: message content still goes through
 * qw_put_text, which never emits a control token, so user text cannot forge a
 * role boundary while the trained prompt still tokenizes as it did. */
static void qw_put_template(qw_chat *c, const char *s) {
    if (!c->ok || !s) return;
    const char *run = s;
    for (const char *p = s; *p; ) {
        size_t n = 0;
        int32_t id = qw_special_at(c->t, p, &n);
        if (id < 0) { p++; continue; }
        qw_put_text(c, run, (size_t)(p - run));
        qw_put_id(c, id);
        p += n;
        run = p;
    }
    qw_put_text(c, run, strlen(run));
}

/* Verbatim from the model's chat template.  The call-format description is not
 * documentation for us -- it is the text the model was trained to condition on,
 * so it is reproduced exactly rather than paraphrased. */
static const char *QW_TOOLS_HEAD =
    "# Tools\n\nYou have access to the following functions:\n\n<tools>";
static const char *QW_TOOLS_TAIL =
    "\n</tools>\n\nIf you choose to call a function ONLY reply in the following "
    "format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n"
    "<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
    "<parameter=example_parameter_2>\nThis is the value for the second parameter\n"
    "that can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n"
    "<IMPORTANT>\nReminder:\n"
    "- Function calls MUST follow the specified format: an inner "
    "<function=...></function> block must be nested within <tool_call></tool_call> "
    "XML tags\n"
    "- Required parameters MUST be specified\n"
    "- You may provide optional reasoning for your function call in natural "
    "language BEFORE the function call, but NOT after\n"
    "- If there is no function call available, answer the question like normal "
    "with your current knowledge and do not tell the user about function calls\n"
    "</IMPORTANT>";

/* Opens an assistant turn and its reasoning block. */
static void qw_put_generation_prompt(qw_chat *c, const qwasar_chat_options *opts) {
    qw_put_id(c, c->im_start);
    qw_put_str(c, "assistant\n");
    qw_put_id(c, c->think_open);
    if (opts->enable_thinking) {
        qw_put_str(c, "\n");
    } else {
        qw_put_str(c, "\n\n");
        qw_put_id(c, c->think_close);
        qw_put_str(c, "\n\n");
    }
}

static bool qw_chat_init(qw_chat *c, const qwasar_tokenizer *t, char *err, size_t errcap) {
    memset(c, 0, sizeof *c);
    c->t = t;
    c->ok = true;
    c->im_start    = qwasar_token_id(t, "<|im_start|>");
    c->im_end      = qwasar_token_id(t, "<|im_end|>");
    c->think_open  = qwasar_token_id(t, "<think>");
    c->think_close = qwasar_token_id(t, "</think>");
    c->tr_open     = qwasar_token_id(t, "<tool_response>");
    c->tr_close    = qwasar_token_id(t, "</tool_response>");
    /* Absent on a text-only checkpoint, which is fine as long as nothing asks
     * for image tokens; qw_put_id refuses a negative id. */
    c->vision_start = qwasar_token_id(t, "<|vision_start|>");
    c->vision_end   = qwasar_token_id(t, "<|vision_end|>");
    c->image_pad    = qwasar_token_id(t, "<|image_pad|>");
    c->video_pad    = qwasar_token_id(t, "<|video_pad|>");
    if (c->im_start < 0 || c->im_end < 0 || c->think_open < 0 || c->think_close < 0) {
        snprintf(err, errcap, "tokenizer is missing ChatML control tokens");
        return false;
    }
    return true;
}

int32_t *qwasar_render_tool_result(const qwasar_tokenizer *t, const char *result,
                                   const qwasar_chat_options *opts, int32_t *out_n) {
    char err[128];
    qw_chat c;
    *out_n = 0;
    if (!qw_chat_init(&c, t, err, sizeof err)) return NULL;
    if (c.tr_open < 0 || c.tr_close < 0) return NULL;

    /* Close the assistant turn the model left open when it stopped at
     * </tool_call>, then deliver the result as a user turn. */
    qw_put_id(&c, c.im_end);
    qw_put_str(&c, "\n");
    qw_put_id(&c, c.im_start);
    qw_put_str(&c, "user\n");
    qw_put_id(&c, c.tr_open);
    qw_put_str(&c, "\n");
    qw_put_str(&c, result);
    qw_put_str(&c, "\n");
    qw_put_id(&c, c.tr_close);
    qw_put_id(&c, c.im_end);
    qw_put_str(&c, "\n");
    qw_put_generation_prompt(&c, opts);

    if (!c.ok) { free(c.v.v); return NULL; }
    *out_n = c.v.n;
    return c.v.v;
}

int32_t *qwasar_render_user_turn(const qwasar_tokenizer *t, const char *text,
                                 int32_t n_image_tokens, bool is_video,
                                 const qwasar_chat_options *opts, int32_t *out_n) {
    char err[128];
    qw_chat c;
    *out_n = 0;
    if (!qw_chat_init(&c, t, err, sizeof err)) return NULL;

    qw_put_id(&c, c.im_end);
    qw_put_str(&c, "\n");
    qw_put_id(&c, c.im_start);
    qw_put_str(&c, "user\n");
    const int32_t pad = is_video ? c.video_pad : c.image_pad;
    for (int32_t k = 0; k < n_image_tokens; k++) {
        if (k == 0) qw_put_id(&c, c.vision_start);
        qw_put_id(&c, pad);
        if (k + 1 == n_image_tokens) qw_put_id(&c, c.vision_end);
    }
    qw_put_str(&c, text);
    qw_put_id(&c, c.im_end);
    qw_put_str(&c, "\n");
    qw_put_generation_prompt(&c, opts);

    if (!c.ok) { free(c.v.v); return NULL; }
    *out_n = c.v.n;
    return c.v.v;
}

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

    qw_chat c;
    if (!qw_chat_init(&c, t, err, errcap)) return NULL;

    const char *reasoning = qw_reasoning_text(opts);
    const bool has_system = !strcmp(msgs[0].role, "system");

    size_t soff = 0, slen = 0;
    if (has_system) qw_trim(msgs[0].content, &soff, &slen);

    if (opts->n_tools > 0 && opts->tools) {
        /* With tools the system turn is rebuilt around them, and it is always
         * emitted -- the format description has to reach the model even when
         * the caller supplied no system message of its own. */
        qw_put_id(&c, c.im_start);
        qw_put_str(&c, "system\n");
        if (reasoning) { qw_put_str(&c, reasoning); qw_put_str(&c, "\n\n"); }
        qw_put_template(&c, QW_TOOLS_HEAD);
        for (int32_t i = 0; i < opts->n_tools; i++) {
            qw_put_str(&c, "\n");
            qw_put_str(&c, opts->tools[i]);
        }
        qw_put_template(&c, QW_TOOLS_TAIL);
        if (slen) {
            qw_put_str(&c, "\n\n");
            qw_put_text(&c, msgs[0].content + soff, slen);
        }
        qw_put_id(&c, c.im_end);
        qw_put_str(&c, "\n");
    } else if (slen || reasoning) {
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
            /* Images are declared by the caller rather than written into the
             * content, because the tokenizer deliberately does not honour
             * control tokens appearing in user text.  The template emits one
             * <|image_pad|> and the processor expands it; the count is known
             * here, so it is expanded here. */
            const int32_t pad = m->vision_is_video ? c.video_pad : c.image_pad;
            for (int32_t k = 0; k < m->n_image_tokens; k++) {
                if (k == 0) qw_put_id(&c, c.vision_start);
                qw_put_id(&c, pad);
                if (k + 1 == m->n_image_tokens) qw_put_id(&c, c.vision_end);
            }
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
            if (m->tool_calls && *m->tool_calls) {
                if (len) qw_put_str(&c, "\n\n");
                qw_put_template(&c, m->tool_calls);
            }
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

    if (opts->add_generation_prompt) qw_put_generation_prompt(&c, opts);

    if (!c.ok) {
        snprintf(err, errcap, "out of memory encoding the prompt");
        free(c.v.v);
        return NULL;
    }
    *out_n = c.v.n;
    return c.v.v;
}
