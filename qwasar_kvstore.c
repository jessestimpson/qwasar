/* Disk checkpoints of session state.
 *
 * Prefilling a prompt is the slowest visible part of a turn, and an agent
 * reopens the same ~900-token system prefix on every run.  A checkpoint lets a
 * later session skip straight past whatever prefix it already covers.
 *
 * Two properties of this model shape the design, and both differ from a
 * pure-attention engine:
 *
 *   - A checkpoint is not just the KV cache.  Forty-eight of sixty-four layers
 *     are recurrent, so the conv and delta-rule state travel with it.  Those are
 *     the same size for a ten-token prefix as for a ten-thousand-token one --
 *     about 149 MB -- so every checkpoint carries a large fixed floor and a
 *     shallow slope.  That argues for few large checkpoints, not many small
 *     ones, and the minimum-token threshold below exists because of it.
 *
 *   - Reuse is prefix-only, and that is a hard constraint rather than a
 *     simplification.  A KV cache can be truncated to any length; a recurrent
 *     state keeps no per-position history and cannot be rewound.  So a
 *     checkpoint is usable exactly when its tokens are a prefix of the incoming
 *     prompt.
 *
 * Files are keyed by a hash of their token sequence, but the stored tokens are
 * always compared before use, so a hash collision costs a miss rather than
 * corruption. */

#include "qwasar.h"
#include "qwasar_model.h"

#include <dirent.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define QW_KV_MAGIC   "QWKV0001"
#define QW_KV_MIN_TOKENS 256          /* below this the fixed 149 MB is not worth it */
#define QW_KV_DEFAULT_BUDGET (6ull * 1024ull * 1024ull * 1024ull)

/* Fixed-size header, little-endian by construction on every target we build
 * for.  `dims` pins the shapes the payload was written with so a checkpoint
 * from a different model or quantisation is rejected rather than misread. */
typedef struct {
    char     magic[8];
    uint64_t model_id;
    uint64_t token_hash;
    uint32_t n_tokens;
    uint32_t dims[8];      /* head_dim, kv_heads, n_full, n_linear, hv, dv, dk, conv_dim */
    uint64_t payload_bytes;
    uint64_t created_at;
    uint64_t last_used;
    uint32_t hits;
    uint32_t reserved;
} qw_kv_header;

static uint64_t qw_fnv64(const void *data, size_t n, uint64_t h) {
    const unsigned char *p = data;
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

/* Identifies the model closely enough that a checkpoint cannot be read back
 * against different weights or a different quantisation. */
static uint64_t qw_model_id(const qw_config *c) {
    uint32_t f[12] = {
        (uint32_t)c->hidden_size, (uint32_t)c->num_hidden_layers,
        (uint32_t)c->vocab_size, (uint32_t)c->head_dim,
        (uint32_t)c->num_attention_heads, (uint32_t)c->num_key_value_heads,
        (uint32_t)c->intermediate_size, (uint32_t)c->full_attention_interval,
        (uint32_t)c->linear_num_value_heads, (uint32_t)c->linear_key_head_dim,
        (uint32_t)c->quant_bits, (uint32_t)c->quant_group_size,
    };
    return qw_fnv64(f, sizeof f, 1469598103934665603ull);
}

static void qw_fill_dims(const qw_config *c, const qw_shape *sh, uint32_t d[8]) {
    d[0] = (uint32_t)c->head_dim;
    d[1] = (uint32_t)c->num_key_value_heads;
    d[2] = (uint32_t)sh->n_full_attn_layers;
    d[3] = (uint32_t)sh->n_linear_attn_layers;
    d[4] = (uint32_t)c->linear_num_value_heads;
    d[5] = (uint32_t)c->linear_value_head_dim;
    d[6] = (uint32_t)c->linear_key_head_dim;
    d[7] = (uint32_t)sh->conv_dim;
}

/* Creates every missing level, not just the last: a fresh account may not have
 * ~/.cache at all, and mkdir does not create intermediates. */
static bool qw_mkdir_p(const char *path) {
    char tmp[1024];
    snprintf(tmp, sizeof tmp, "%s", path);
    for (char *p = tmp + 1; *p; p++) {
        if (*p != '/') continue;
        *p = 0;
        if (mkdir(tmp, 0700) != 0 && errno != EEXIST) return false;
        *p = '/';
    }
    return mkdir(tmp, 0700) == 0 || errno == EEXIST;
}

static bool qw_kv_dir(char *out, size_t cap) {
    const char *home = getenv("HOME");
    if (!home || !*home) return false;
    snprintf(out, cap, "%s/.cache/qwasar/kv", home);
    return qw_mkdir_p(out);
}

static uint64_t qw_now(void) { return (uint64_t)time(NULL); }

/* ---- eviction ---------------------------------------------------------------
 *
 * Least-recently-used against a byte budget.  ds4 weights this by hit count
 * with a six-hour half-life; plain recency is easier to predict and the
 * checkpoints here are large enough that only a handful ever coexist. */

typedef struct { char path[1200]; uint64_t last_used, size; } qw_kv_file;

static int qw_by_age(const void *a, const void *b) {
    uint64_t x = ((const qw_kv_file *)a)->last_used;
    uint64_t y = ((const qw_kv_file *)b)->last_used;
    return x < y ? -1 : x > y ? 1 : 0;
}

static void qw_kv_evict(const char *dir, uint64_t budget) {
    DIR *d = opendir(dir);
    if (!d) return;

    qw_kv_file *files = NULL;
    size_t n = 0, cap = 0, total = 0;
    struct dirent *e;
    while ((e = readdir(d))) {
        if (e->d_name[0] == '.') continue;
        char path[1200];
        snprintf(path, sizeof path, "%s/%s", dir, e->d_name);

        FILE *f = fopen(path, "rb");
        if (!f) continue;
        qw_kv_header h;
        bool ok = fread(&h, sizeof h, 1, f) == 1 && !memcmp(h.magic, QW_KV_MAGIC, 8);
        fclose(f);
        if (!ok) continue;

        struct stat st;
        if (stat(path, &st) != 0) continue;

        if (n == cap) {
            cap = cap ? cap * 2 : 16;
            qw_kv_file *nf = realloc(files, cap * sizeof *nf);
            if (!nf) break;
            files = nf;
        }
        snprintf(files[n].path, sizeof files[n].path, "%s", path);
        files[n].last_used = h.last_used;
        files[n].size = (uint64_t)st.st_size;
        total += files[n].size;
        n++;
    }
    closedir(d);

    if (total > budget && n > 0) {
        qsort(files, n, sizeof *files, qw_by_age);
        for (size_t i = 0; i < n && total > budget; i++) {
            if (unlink(files[i].path) == 0) total -= files[i].size;
        }
    }
    free(files);
}

/* ---- save ------------------------------------------------------------------- */

bool qwasar_session_save(qwasar_session *s, const qwasar_engine *e,
                         char *err, size_t errcap) {
    int32_t n = 0;
    const int32_t *tokens = qw_session_history(s, &n);
    if (!tokens || n < QW_KV_MIN_TOKENS) {
        snprintf(err, errcap, "nothing worth saving (%d tokens, minimum %d)",
                 n, QW_KV_MIN_TOKENS);
        return false;
    }
    if (n != qwasar_session_n_past(s)) {
        snprintf(err, errcap, "session history and cache position disagree");
        return false;
    }

    char dir[1024];
    if (!qw_kv_dir(dir, sizeof dir)) {
        snprintf(err, errcap, "cannot create the cache directory");
        return false;
    }

    const qw_config *c = qwasar_engine_config(e);
    const qw_shape  *sh = qwasar_engine_shape(e);

    qw_kv_header h;
    memset(&h, 0, sizeof h);
    memcpy(h.magic, QW_KV_MAGIC, 8);
    h.model_id   = qw_model_id(c);
    h.token_hash = qw_fnv64(tokens, (size_t)n * sizeof *tokens, h.model_id);
    h.n_tokens   = (uint32_t)n;
    qw_fill_dims(c, sh, h.dims);
    h.payload_bytes = qw_session_state_bytes(s, n);
    h.created_at = h.last_used = qw_now();
    h.hits = 0;

    void *payload = malloc(h.payload_bytes);
    if (!payload) { snprintf(err, errcap, "out of memory"); return false; }
    if (!qw_session_pack(s, payload, h.payload_bytes)) {
        free(payload);
        snprintf(err, errcap, "cannot pack the session state");
        return false;
    }

    char path[1200], tmp[1264];
    snprintf(path, sizeof path, "%s/%016llx.qwkv", dir, (unsigned long long)h.token_hash);
    snprintf(tmp, sizeof tmp, "%s.tmp%d", path, (int)getpid());

    /* Written to a temporary and renamed, so a checkpoint is either complete or
     * absent -- a half-written one would restore silently corrupt state. */
    FILE *f = fopen(tmp, "wb");
    if (!f) {
        free(payload);
        snprintf(err, errcap, "cannot create %s: %s", tmp, strerror(errno));
        return false;
    }
    bool ok = fwrite(&h, sizeof h, 1, f) == 1
           && fwrite(tokens, sizeof *tokens, (size_t)n, f) == (size_t)n
           && fwrite(payload, 1, h.payload_bytes, f) == h.payload_bytes;
    ok = (fclose(f) == 0) && ok;
    free(payload);

    if (!ok || rename(tmp, path) != 0) {
        unlink(tmp);
        snprintf(err, errcap, "cannot write %s: %s", path, strerror(errno));
        return false;
    }

    qw_kv_evict(dir, QW_KV_DEFAULT_BUDGET);
    return true;
}

/* ---- restore ---------------------------------------------------------------- */

/* True if the file's stored tokens are a prefix of `tokens`, and long enough to
 * be worth loading.  The header is read first so a candidate is rejected
 * without touching its hundreds of megabytes of payload. */
static int32_t qw_kv_try(const char *path, const qw_kv_header *want,
                         const int32_t *tokens, int32_t n,
                         qw_kv_header *out, FILE **out_f) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;

    qw_kv_header h;
    if (fread(&h, sizeof h, 1, f) != 1 || memcmp(h.magic, QW_KV_MAGIC, 8)
        || h.model_id != want->model_id
        || memcmp(h.dims, want->dims, sizeof h.dims)
        || h.n_tokens == 0 || (int32_t)h.n_tokens > n) {
        fclose(f);
        return 0;
    }

    struct stat st;
    if (stat(path, &st) != 0
        || (uint64_t)st.st_size != sizeof h + (uint64_t)h.n_tokens * sizeof(int32_t)
                                 + h.payload_bytes) {
        fclose(f);   /* truncated or from a different build */
        return 0;
    }

    int32_t *stored = malloc((size_t)h.n_tokens * sizeof *stored);
    if (!stored) { fclose(f); return 0; }
    if (fread(stored, sizeof *stored, h.n_tokens, f) != h.n_tokens
        || memcmp(stored, tokens, (size_t)h.n_tokens * sizeof *stored) != 0) {
        free(stored);
        fclose(f);
        return 0;
    }
    free(stored);

    *out = h;
    *out_f = f;
    return (int32_t)h.n_tokens;
}

int32_t qwasar_session_restore(qwasar_session *s, const qwasar_engine *e,
                               const int32_t *tokens, int32_t n) {
    if (!tokens || n <= 0 || qwasar_session_n_past(s) != 0) return 0;

    char dir[1024];
    if (!qw_kv_dir(dir, sizeof dir)) return 0;

    const qw_config *c = qwasar_engine_config(e);
    const qw_shape  *sh = qwasar_engine_shape(e);
    qw_kv_header want;
    memset(&want, 0, sizeof want);
    want.model_id = qw_model_id(c);
    qw_fill_dims(c, sh, want.dims);

    DIR *d = opendir(dir);
    if (!d) return 0;

    char best_path[1200] = "";
    int32_t best_n = 0;
    struct dirent *ent;
    while ((ent = readdir(d))) {
        if (ent->d_name[0] == '.') continue;
        char path[1200];
        snprintf(path, sizeof path, "%s/%s", dir, ent->d_name);

        qw_kv_header h;
        FILE *f = NULL;
        int32_t got = qw_kv_try(path, &want, tokens, n, &h, &f);
        if (f) fclose(f);
        /* Longest prefix wins; a shorter one would leave more to prefill. */
        if (got > best_n) { best_n = got; snprintf(best_path, sizeof best_path, "%s", path); }
    }
    closedir(d);
    if (best_n == 0) return 0;

    qw_kv_header h;
    FILE *f = NULL;
    if (qw_kv_try(best_path, &want, tokens, n, &h, &f) != best_n || !f) {
        if (f) fclose(f);
        return 0;
    }

    void *payload = malloc(h.payload_bytes);
    bool ok = payload && fread(payload, 1, h.payload_bytes, f) == h.payload_bytes;
    fclose(f);
    if (ok) ok = qw_session_unpack(s, payload, h.payload_bytes, tokens, best_n);
    free(payload);
    if (!ok) return 0;

    /* Record the hit in place; the header is fixed-size and at offset zero. */
    FILE *up = fopen(best_path, "r+b");
    if (up) {
        h.last_used = qw_now();
        h.hits++;
        fwrite(&h, sizeof h, 1, up);
        fclose(up);
    }
    return best_n;
}

void qwasar_kv_cache_stats(uint64_t *bytes, int *entries) {
    if (bytes) *bytes = 0;
    if (entries) *entries = 0;
    char dir[1024];
    if (!qw_kv_dir(dir, sizeof dir)) return;
    DIR *d = opendir(dir);
    if (!d) return;
    struct dirent *e;
    while ((e = readdir(d))) {
        if (e->d_name[0] == '.') continue;
        char path[1200];
        snprintf(path, sizeof path, "%s/%s", dir, e->d_name);
        struct stat st;
        if (stat(path, &st) != 0) continue;
        if (bytes) *bytes += (uint64_t)st.st_size;
        if (entries) (*entries)++;
    }
    closedir(d);
}
