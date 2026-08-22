/* qwasar -- Qwen3.8 27B inference on macOS Metal.
 *
 * This file owns model loading and the engine/session lifetime.  Weights are
 * mmap-backed and never copied: each safetensors shard is mapped once and
 * wrapped in a single device buffer, and every tensor is an (buffer, offset)
 * pair into it.  A 16 GB model therefore "loads" in milliseconds and becomes
 * resident lazily, as the graph first touches each layer. */

#include "qwasar.h"
#include "qwasar_gpu.h"
#include "qwasar_json.h"
#include "qwasar_model.h"

#include <fcntl.h>
#include <mach-o/dyld.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

/* ---- small utilities ----------------------------------------------------- */

static void qw_errf(char *err, size_t cap, const char *fmt, ...) {
    if (!err || !cap) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, cap, fmt, ap);
    va_end(ap);
}

const char *qw_dtype_name(qw_dtype d) {
    switch (d) {
    case QW_DT_F32:  return "F32";
    case QW_DT_F16:  return "F16";
    case QW_DT_BF16: return "BF16";
    case QW_DT_U32:  return "U32";
    case QW_DT_I32:  return "I32";
    case QW_DT_U8:   return "U8";
    default:         return "?";
    }
}

size_t qw_dtype_size(qw_dtype d) {
    switch (d) {
    case QW_DT_F32: case QW_DT_U32: case QW_DT_I32: return 4;
    case QW_DT_F16: case QW_DT_BF16:                return 2;
    case QW_DT_U8:                                  return 1;
    default:                                        return 0;
    }
}

static qw_dtype qw_dtype_parse(const char *s, uint32_t len) {
    struct { const char *n; qw_dtype d; } map[] = {
        { "F32", QW_DT_F32 }, { "F16", QW_DT_F16 }, { "BF16", QW_DT_BF16 },
        { "U32", QW_DT_U32 }, { "I32", QW_DT_I32 }, { "U8", QW_DT_U8 },
    };
    for (size_t i = 0; i < sizeof map / sizeof *map; i++)
        if (strlen(map[i].n) == len && !memcmp(map[i].n, s, len)) return map[i].d;
    return QW_DT_UNKNOWN;
}

void qw_config_derive(const qw_config *c, qw_shape *s) {
    memset(s, 0, sizeof *s);
    for (int i = 0; i < c->num_hidden_layers; i++)
        if (qw_layer_is_linear(c, i)) s->n_linear_attn_layers++;
        else                          s->n_full_attn_layers++;

    s->q_dim      = c->num_attention_heads * c->head_dim;
    s->q_proj_out = s->q_dim * (c->attn_output_gate ? 2 : 1);
    s->kv_dim     = c->num_key_value_heads * c->head_dim;
    s->gqa_factor = c->num_key_value_heads ? c->num_attention_heads / c->num_key_value_heads : 0;

    s->key_dim   = c->linear_num_key_heads   * c->linear_key_head_dim;
    s->value_dim = c->linear_num_value_heads * c->linear_value_head_dim;
    s->conv_dim  = 2 * s->key_dim + s->value_dim;
    s->gdn_gqa_factor = c->linear_num_key_heads
                      ? c->linear_num_value_heads / c->linear_num_key_heads : 0;
}

/* ---- name arena ---------------------------------------------------------- */

typedef struct qw_arena_block {
    struct qw_arena_block *next;
    size_t used, cap;
    char   data[];
} qw_arena_block;

typedef struct { qw_arena_block *head; } qw_arena;

static char *qw_arena_str(qw_arena *a, const char *s, size_t len) {
    size_t need = len + 1;
    if (!a->head || a->head->cap - a->head->used < need) {
        size_t cap = need > (1u << 20) ? need : (1u << 20);
        qw_arena_block *b = malloc(sizeof *b + cap);
        if (!b) return NULL;
        b->next = a->head;
        b->used = 0;
        b->cap  = cap;
        a->head = b;
    }
    char *out = a->head->data + a->head->used;
    memcpy(out, s, len);
    out[len] = 0;
    a->head->used += need;
    return out;
}

static void qw_arena_free(qw_arena *a) {
    for (qw_arena_block *b = a->head; b; ) {
        qw_arena_block *n = b->next;
        free(b);
        b = n;
    }
    a->head = NULL;
}

/* ---- tensor table --------------------------------------------------------
 *
 * Open-addressed hash of name -> tensor index.  Names are resolved to direct
 * pointers once at load, so the hot path never hashes a string. */

typedef struct {
    qw_tensor *v;
    uint32_t   count, cap;
    uint32_t  *hash;      /* slot -> index+1, 0 = empty */
    uint32_t   hash_cap;  /* power of two */
} qw_table;

static uint64_t qw_hash_name(const char *s) {
    uint64_t h = 1469598103934665603ull;
    for (; *s; s++) { h ^= (unsigned char)*s; h *= 1099511628211ull; }
    return h;
}

static bool qw_table_rehash(qw_table *t, uint32_t cap) {
    uint32_t *h = calloc(cap, sizeof *h);
    if (!h) return false;
    free(t->hash);
    t->hash = h;
    t->hash_cap = cap;
    for (uint32_t i = 0; i < t->count; i++) {
        uint32_t m = (uint32_t)(qw_hash_name(t->v[i].name) & (cap - 1));
        while (h[m]) m = (m + 1) & (cap - 1);
        h[m] = i + 1;
    }
    return true;
}

static qw_tensor *qw_table_add(qw_table *t) {
    if (t->count == t->cap) {
        uint32_t cap = t->cap ? t->cap * 2 : 256;
        qw_tensor *v = realloc(t->v, (size_t)cap * sizeof *v);
        if (!v) return NULL;
        t->v = v;
        t->cap = cap;
    }
    qw_tensor *e = &t->v[t->count++];
    memset(e, 0, sizeof *e);
    return e;
}

static bool qw_table_index(qw_table *t) {
    uint32_t cap = 64;
    while (cap < t->count * 2) cap <<= 1;
    return qw_table_rehash(t, cap);
}

static const qw_tensor *qw_table_find(const qw_table *t, const char *name) {
    if (!t->hash_cap) return NULL;
    uint32_t m = (uint32_t)(qw_hash_name(name) & (t->hash_cap - 1));
    while (t->hash[m]) {
        const qw_tensor *e = &t->v[t->hash[m] - 1];
        if (!strcmp(e->name, name)) return e;
        m = (m + 1) & (t->hash_cap - 1);
    }
    return NULL;
}

static const qw_tensor *qw_table_findf(const qw_table *t, const char *fmt, ...) {
    char name[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(name, sizeof name, fmt, ap);
    va_end(ap);
    return qw_table_find(t, name);
}

static void qw_table_free(qw_table *t) {
    free(t->v);
    free(t->hash);
    memset(t, 0, sizeof *t);
}

/* ---- engine -------------------------------------------------------------- */

/* Tensors are read as uint (packed 4-bit words) and ushort (bf16 scales), so
 * their addresses must be 4-byte aligned.  Every safetensors tensor offset is
 * relative to the shard's data section and is itself 32-byte aligned, so
 * alignment is decided once per shard by where that data section starts --
 * which the writer is under no obligation to align at all.
 *
 * When a shard's data section is misaligned we copy it into an aligned device
 * buffer; when it is aligned we wrap the mmap and never touch the bytes.
 * (Phase 7's vectorised uint4 loads will want 16 here, at which point the right
 * answer is a disk-cached aligned repack rather than more resident copies.) */
#define QW_TENSOR_ALIGN 4

typedef struct {
    void  *addr;         /* mmap base, or NULL once materialized */
    size_t len;          /* mapped length (page-rounded) */
    size_t file_len;
    size_t data_base;    /* byte offset of the data section within buf */
    qw_buf buf;
    bool   materialized; /* data section was copied to satisfy alignment */
    char   path[1024];
} qw_shard;

struct qwasar_engine {
    qwasar_options opts;
    char           model_path[1024];

    qw_config config;
    qw_shape  shape;

    qw_shard  shards[QW_MAX_SHARDS];
    int       n_shards;

    qw_arena  names;
    qw_table  tensors;

    qw_layer  layers[QW_MAX_LAYERS];
    qw_qlinear embed_tokens, lm_head;
    const qw_tensor *final_norm;
    qw_mtp    mtp;

    size_t weight_bytes;       /* total bytes of bound tensors */
    size_t weight_bytes_text;
    size_t weight_bytes_mtp;
    size_t weight_bytes_mtp_q4;
    size_t weight_bytes_vision;
    size_t bytes_mapped;       /* zero-copy, file-backed */
    size_t bytes_copied;       /* materialized for alignment */

    int32_t context_size;      /* resolved from options and the model's limit */
    int32_t prefill_chunk;
};

/* ---- safetensors ---------------------------------------------------------
 *
 * Layout: [8-byte LE header length][JSON header][tensor data].  Every tensor's
 * data_offsets are relative to the start of the data section, so the absolute
 * offset into our whole-file mapping is 8 + header_len + start.  Mapping the
 * entire file (rather than each tensor) keeps us to one device buffer per
 * shard, well under the 20 GB per-buffer cap. */

static bool qw_load_shard(qwasar_engine *e, const char *path, char *err, size_t errcap) {
    if (e->n_shards >= QW_MAX_SHARDS) {
        qw_errf(err, errcap, "too many safetensors shards (max %d)", QW_MAX_SHARDS);
        return false;
    }
    qw_shard *sh = &e->shards[e->n_shards];
    memset(sh, 0, sizeof *sh);
    snprintf(sh->path, sizeof sh->path, "%s", path);

    int fd = open(path, O_RDONLY);
    if (fd < 0) { qw_errf(err, errcap, "cannot open %s", path); return false; }

    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < 8) {
        close(fd);
        qw_errf(err, errcap, "cannot stat %s", path);
        return false;
    }
    sh->file_len = (size_t)st.st_size;

    uint64_t header_len = 0;
    if (pread(fd, &header_len, 8, 0) != 8) {
        close(fd);
        qw_errf(err, errcap, "short read on %s", path);
        return false;
    }
    if (header_len == 0 || header_len > sh->file_len - 8) {
        close(fd);
        qw_errf(err, errcap, "bogus safetensors header length in %s", path);
        return false;
    }

    char *header = malloc(header_len + 1);
    if (!header) { close(fd); qw_errf(err, errcap, "out of memory"); return false; }
    if ((uint64_t)pread(fd, header, header_len, 8) != header_len) {
        free(header); close(fd);
        qw_errf(err, errcap, "short header read on %s", path);
        return false;
    }

    /* Map the whole file.  MAP_PRIVATE|PROT_READ keeps it file-backed and
     * evictable: resident set tracks what the graph actually touches. */
    size_t page = (size_t)getpagesize();
    size_t map_len = (sh->file_len + page - 1) & ~(page - 1);
    void  *map = mmap(NULL, map_len, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (map == MAP_FAILED) {
        free(header);
        qw_errf(err, errcap, "mmap failed for %s", path);
        return false;
    }

    const size_t data_base = 8 + header_len;
    const size_t data_len  = sh->file_len - data_base;

    if (data_base % QW_TENSOR_ALIGN == 0) {
        sh->addr      = map;
        sh->len       = map_len;
        sh->data_base = data_base;
        sh->buf       = qw_buf_wrap(map, map_len);
        if (sh->buf) e->bytes_mapped += data_len;
    } else {
        /* The writer left the data section at an odd byte.  Copy it once into
         * an aligned buffer; tensor offsets then become relative to the buffer
         * start, which is 16-byte aligned by allocation. */
        sh->materialized = true;
        sh->data_base    = 0;
        sh->buf          = qw_buf_alloc(data_len);
        if (sh->buf) {
            memcpy(qw_buf_contents(sh->buf), (const char *)map + data_base, data_len);
            e->bytes_copied += data_len;
            if (e->opts.verbose)
                fprintf(stderr, "qwasar: %s data section is %zu-byte misaligned, "
                                "copied %.2f GB into aligned memory\n",
                        path, data_base % QW_TENSOR_ALIGN,
                        (double)data_len / (1024.0 * 1024.0 * 1024.0));
        }
        munmap(map, map_len);
    }

    if (!sh->buf) {
        free(header);
        if (sh->addr) { munmap(sh->addr, sh->len); sh->addr = NULL; }
        qw_errf(err, errcap, "cannot bind %s as a device buffer", path);
        return false;
    }
    e->n_shards++;

    qj_doc doc;
    if (!qj_parse(&doc, header, header_len)) {
        qw_errf(err, errcap, "bad safetensors header in %s: %s", path, doc.err);
        qj_free(&doc);
        free(header);
        return false;
    }
    free(header);

    const qj_node *root = qj_root(&doc);
    for (const qj_node *m = qj_first(&doc, root); m; m = qj_next(&doc, m)) {
        if (m->key_len == 12 && !memcmp(doc.text + m->key_off, "__metadata__", 12)) continue;
        if (m->type != QJ_OBJECT) continue;

        const qj_node *dt   = qj_get(&doc, m, "dtype");
        const qj_node *shp  = qj_get(&doc, m, "shape");
        const qj_node *offs = qj_get(&doc, m, "data_offsets");
        if (!dt || !shp || !offs) continue;

        const qj_node *o0 = qj_idx(&doc, offs, 0), *o1 = qj_idx(&doc, offs, 1);
        if (!o0 || !o1) continue;

        qw_tensor *t = qw_table_add(&e->tensors);
        if (!t) { qj_free(&doc); qw_errf(err, errcap, "out of memory"); return false; }

        t->name   = qw_arena_str(&e->names, doc.text + m->key_off, m->key_len);
        t->buf    = sh->buf;
        t->offset = sh->data_base + (size_t)o0->u.num;
        t->nbytes = (size_t)o1->u.num - (size_t)o0->u.num;
        t->dtype  = qw_dtype_parse(doc.text + dt->u.str.off, dt->u.str.len);
        t->ndim   = 0;
        for (const qj_node *d = qj_first(&doc, shp); d && t->ndim < QW_MAX_DIMS;
             d = qj_next(&doc, d))
            t->shape[t->ndim++] = (int64_t)d->u.num;

        if (!t->name) { qj_free(&doc); qw_errf(err, errcap, "out of memory"); return false; }
        if (t->offset + t->nbytes > qw_buf_length(sh->buf)) {
            qw_errf(err, errcap, "tensor %s runs past end of %s", t->name, path);
            qj_free(&doc);
            return false;
        }
    }
    qj_free(&doc);
    return true;
}

static bool qw_load_shards(qwasar_engine *e, char *err, size_t errcap) {
    /* Prefer the index, which names every shard; fall back to the single-file
     * layout that small conversions use. */
    char path[1200];
    snprintf(path, sizeof path, "%s/model.safetensors.index.json", e->model_path);

    qj_doc idx;
    if (qj_parse_file(&idx, path)) {
        const qj_node *map = qj_get(&idx, qj_root(&idx), "weight_map");
        if (!map) {
            qw_errf(err, errcap, "index has no weight_map");
            qj_free(&idx);
            return false;
        }
        /* Collect the distinct shard filenames in first-seen order. */
        char seen[QW_MAX_SHARDS][256];
        int n_seen = 0;
        for (const qj_node *m = qj_first(&idx, map); m; m = qj_next(&idx, m)) {
            if (m->type != QJ_STRING) continue;
            const char *f = idx.text + m->u.str.off;
            uint32_t fl = m->u.str.len;
            bool dup = false;
            for (int i = 0; i < n_seen; i++)
                if (strlen(seen[i]) == fl && !memcmp(seen[i], f, fl)) { dup = true; break; }
            if (dup) continue;
            if (n_seen >= QW_MAX_SHARDS || fl >= sizeof seen[0]) {
                qw_errf(err, errcap, "too many shards in index");
                qj_free(&idx);
                return false;
            }
            memcpy(seen[n_seen], f, fl);
            seen[n_seen][fl] = 0;
            n_seen++;
        }
        qj_free(&idx);

        for (int i = 0; i < n_seen; i++) {
            snprintf(path, sizeof path, "%s/%s", e->model_path, seen[i]);
            if (!qw_load_shard(e, path, err, errcap)) return false;
        }
        return n_seen > 0;
    }
    qj_free(&idx);

    snprintf(path, sizeof path, "%s/model.safetensors", e->model_path);
    return qw_load_shard(e, path, err, errcap);
}

/* ---- config -------------------------------------------------------------- */

static bool qw_load_config(qwasar_engine *e, char *err, size_t errcap) {
    char path[1200];
    snprintf(path, sizeof path, "%s/config.json", e->model_path);

    qj_doc d;
    if (!qj_parse_file(&d, path)) {
        qw_errf(err, errcap, "cannot read config.json: %s", d.err);
        qj_free(&d);
        return false;
    }
    const qj_node *root = qj_root(&d);

    char model_type[64] = "";
    qj_str_copy(&d, root, "model_type", model_type, sizeof model_type);
    const qj_node *tc = qj_get(&d, root, "text_config");
    if (!tc) {
        qw_errf(err, errcap, "config.json has no text_config -- is this a Qwen3.8 VLM checkpoint?");
        qj_free(&d);
        return false;
    }
    if (strcmp(model_type, "qwen3_5") != 0) {
        qw_errf(err, errcap,
                "unsupported model_type \"%s\" (qwasar implements qwen3_5 / Qwen3.8 only)",
                model_type);
        qj_free(&d);
        return false;
    }

    qw_config *c = &e->config;
    memset(c, 0, sizeof *c);

    c->hidden_size             = (int32_t)qj_int_or(&d, tc, "hidden_size", 0);
    c->intermediate_size       = (int32_t)qj_int_or(&d, tc, "intermediate_size", 0);
    c->num_hidden_layers       = (int32_t)qj_int_or(&d, tc, "num_hidden_layers", 0);
    c->num_attention_heads     = (int32_t)qj_int_or(&d, tc, "num_attention_heads", 0);
    c->num_key_value_heads     = (int32_t)qj_int_or(&d, tc, "num_key_value_heads", 0);
    c->head_dim                = (int32_t)qj_int_or(&d, tc, "head_dim", 0);
    c->vocab_size              = (int32_t)qj_int_or(&d, tc, "vocab_size", 0);
    c->full_attention_interval = (int32_t)qj_int_or(&d, tc, "full_attention_interval", 4);
    c->rms_norm_eps            = (float)qj_num_or(&d, tc, "rms_norm_eps", 1e-6);
    c->tie_word_embeddings     = qj_bool_or(&d, tc, "tie_word_embeddings", false);
    c->attn_output_gate        = qj_bool_or(&d, tc, "attn_output_gate", false);
    c->max_position_embeddings = (int32_t)qj_int_or(&d, tc, "max_position_embeddings", 32768);

    c->linear_num_value_heads  = (int32_t)qj_int_or(&d, tc, "linear_num_value_heads", 0);
    c->linear_num_key_heads    = (int32_t)qj_int_or(&d, tc, "linear_num_key_heads", 0);
    c->linear_key_head_dim     = (int32_t)qj_int_or(&d, tc, "linear_key_head_dim", 0);
    c->linear_value_head_dim   = (int32_t)qj_int_or(&d, tc, "linear_value_head_dim", 0);
    c->linear_conv_kernel_dim  = (int32_t)qj_int_or(&d, tc, "linear_conv_kernel_dim", 4);

    const qj_node *rp = qj_get(&d, tc, "rope_parameters");
    c->rope_theta            = (float)qj_num_or(&d, rp, "rope_theta", 10000.0);
    c->partial_rotary_factor = (float)qj_num_or(&d, rp, "partial_rotary_factor",
                                   qj_num_or(&d, tc, "partial_rotary_factor", 1.0));
    c->mrope_interleaved     = qj_bool_or(&d, rp, "mrope_interleaved", false);
    c->rotary_dim            = (int32_t)(c->head_dim * c->partial_rotary_factor);

    const qj_node *ms = qj_get(&d, rp, "mrope_section");
    for (int i = 0; i < 3; i++) {
        const qj_node *v = qj_idx(&d, ms, (uint32_t)i);
        c->mrope_section[i] = v ? (int32_t)v->u.num : 0;
    }

    /* Quantisation may appear as "quantization" or "quantization_config". */
    const qj_node *q = qj_get(&d, root, "quantization");
    if (!q) q = qj_get(&d, root, "quantization_config");
    c->quant_bits       = (int32_t)qj_int_or(&d, q, "bits", 0);
    c->quant_group_size = (int32_t)qj_int_or(&d, q, "group_size", 0);

    c->bos_token_id           = (int32_t)qj_int_or(&d, tc, "bos_token_id", -1);
    c->image_token_id         = (int32_t)qj_int_or(&d, root, "image_token_id", -1);
    c->video_token_id         = (int32_t)qj_int_or(&d, root, "video_token_id", -1);
    c->vision_start_token_id  = (int32_t)qj_int_or(&d, root, "vision_start_token_id", -1);
    c->vision_end_token_id    = (int32_t)qj_int_or(&d, root, "vision_end_token_id", -1);

    const qj_node *eos = qj_get(&d, root, "eos_token_id");
    if (eos && eos->type == QJ_ARRAY) {
        for (const qj_node *v = qj_first(&d, eos); v && c->n_eos < 8; v = qj_next(&d, v))
            c->eos_token_ids[c->n_eos++] = (int32_t)v->u.num;
    } else if (eos && eos->type == QJ_NUMBER) {
        c->eos_token_ids[c->n_eos++] = (int32_t)eos->u.num;
    }

    const qj_node *vc = qj_get(&d, root, "vision_config");
    if (vc) {
        c->has_vision                = true;
        c->vis_depth                 = (int32_t)qj_int_or(&d, vc, "depth", 0);
        c->vis_hidden_size           = (int32_t)qj_int_or(&d, vc, "hidden_size", 0);
        c->vis_intermediate_size     = (int32_t)qj_int_or(&d, vc, "intermediate_size", 0);
        c->vis_num_heads             = (int32_t)qj_int_or(&d, vc, "num_heads", 0);
        c->vis_patch_size            = (int32_t)qj_int_or(&d, vc, "patch_size", 0);
        c->vis_temporal_patch_size   = (int32_t)qj_int_or(&d, vc, "temporal_patch_size", 0);
        c->vis_spatial_merge_size    = (int32_t)qj_int_or(&d, vc, "spatial_merge_size", 0);
        c->vis_in_channels           = (int32_t)qj_int_or(&d, vc, "in_channels", 3);
        c->vis_out_hidden_size       = (int32_t)qj_int_or(&d, vc, "out_hidden_size", 0);
        c->vis_num_position_embeddings = (int32_t)qj_int_or(&d, vc, "num_position_embeddings", 0);
    }

    qj_free(&d);

    if (c->num_hidden_layers <= 0 || c->num_hidden_layers > QW_MAX_LAYERS) {
        qw_errf(err, errcap, "unsupported num_hidden_layers %d", c->num_hidden_layers);
        return false;
    }
    if (c->quant_bits != 4 || c->quant_group_size != 64) {
        qw_errf(err, errcap,
                "unsupported quantisation (bits=%d group_size=%d); qwasar implements "
                "MLX affine 4-bit with group_size 64",
                c->quant_bits, c->quant_group_size);
        return false;
    }
    qw_config_derive(c, &e->shape);
    return true;
}

/* ---- weight resolution ---------------------------------------------------- */

/* Binds one quantised linear and cross-checks the shapes implied by the MLX
 * affine layout, which is where a mis-parsed config shows up first. */
static bool qw_bind_qlinear(qwasar_engine *e, qw_qlinear *ql, const char *prefix,
                            int32_t in_features, int32_t out_features,
                            char *err, size_t errcap) {
    ql->weight = qw_table_findf(&e->tensors, "%s.weight", prefix);
    ql->scales = qw_table_findf(&e->tensors, "%s.scales", prefix);
    ql->biases = qw_table_findf(&e->tensors, "%s.biases", prefix);
    if (!ql->weight || !ql->scales || !ql->biases) {
        qw_errf(err, errcap, "missing quantised tensors for %s", prefix);
        return false;
    }
    ql->in_features  = in_features;
    ql->out_features = out_features;

    const int32_t words   = in_features / 8;                    /* 8 nibbles per u32 */
    const int32_t groups  = in_features / e->config.quant_group_size;
    if (ql->weight->ndim != 2 || ql->weight->shape[0] != out_features
        || ql->weight->shape[1] != words) {
        qw_errf(err, errcap, "%s.weight is [%lld,%lld], expected [%d,%d]", prefix,
                (long long)ql->weight->shape[0], (long long)ql->weight->shape[1],
                out_features, words);
        return false;
    }
    if (ql->scales->ndim != 2 || ql->scales->shape[0] != out_features
        || ql->scales->shape[1] != groups) {
        qw_errf(err, errcap, "%s.scales is [%lld,%lld], expected [%d,%d]", prefix,
                (long long)ql->scales->shape[0], (long long)ql->scales->shape[1],
                out_features, groups);
        return false;
    }
    e->weight_bytes_text += ql->weight->nbytes + ql->scales->nbytes + ql->biases->nbytes;
    return true;
}

static const qw_tensor *qw_bind_norm(qwasar_engine *e, const char *fmt, int layer,
                                     int32_t dim, char *err, size_t errcap) {
    const qw_tensor *t = qw_table_findf(&e->tensors, fmt, layer);
    if (!t) {
        char name[256];
        snprintf(name, sizeof name, fmt, layer);
        qw_errf(err, errcap, "missing tensor %s", name);
        return NULL;
    }
    if (t->ndim != 1 || t->shape[0] != dim) {
        qw_errf(err, errcap, "%s has shape [%lld], expected [%d]", t->name,
                (long long)t->shape[0], dim);
        return NULL;
    }
    e->weight_bytes_text += t->nbytes;
    return t;
}

/* ---- MTP draft head -------------------------------------------------------
 *
 * The head is a separate repository, not part of the checkpoint, and it has to
 * be: mlx-lm's qwen3_5 sanitize() reacts to the presence of any `mtp.` key by
 * shifting every trunk norm weight by one, so a merged directory is a trap for
 * the Python stack even though it would be nothing to us.  Its tensor names are
 * bare -- `fc.weight`, `layers.0.self_attn.q_proj.weight` -- and the base
 * model's are all under `language_model.`, so the two name sets share one table
 * without prefixing or collision. */

static bool qw_bind_dense(qwasar_engine *e, qw_dense *d, const char *name,
                          int32_t in_features, int32_t out_features,
                          char *err, size_t errcap) {
    d->weight = qw_table_find(&e->tensors, name);
    if (!d->weight) {
        qw_errf(err, errcap, "MTP head is missing %s", name);
        return false;
    }
    if (d->weight->dtype != QW_DT_BF16) {
        qw_errf(err, errcap, "%s is %s, expected bf16", name,
                qw_dtype_name(d->weight->dtype));
        return false;
    }
    if (d->weight->ndim != 2 || d->weight->shape[0] != out_features
        || d->weight->shape[1] != in_features) {
        qw_errf(err, errcap, "%s is [%lld,%lld], expected [%d,%d]", name,
                (long long)d->weight->shape[0], (long long)d->weight->shape[1],
                out_features, in_features);
        return false;
    }
    d->in_features  = in_features;
    d->out_features = out_features;
    e->weight_bytes_mtp += d->weight->nbytes;
    return true;
}

static const qw_tensor *qw_bind_mtp_norm(qwasar_engine *e, const char *name,
                                         int32_t dim, char *err, size_t errcap) {
    const qw_tensor *t = qw_table_find(&e->tensors, name);
    if (!t) {
        qw_errf(err, errcap, "MTP head is missing %s", name);
        return NULL;
    }
    if (t->ndim != 1 || t->shape[0] != dim) {
        qw_errf(err, errcap, "%s has shape [%lld], expected [%d]", name,
                (long long)t->shape[0], dim);
        return NULL;
    }
    e->weight_bytes_mtp += t->nbytes;
    return t;
}

/* ---- quantising the head --------------------------------------------------
 *
 * The head ships in bf16 and every drafted token reads all 849 MB of it, next
 * to the 715 MB of the base model's output head.  Converting it to the same
 * MLX affine 4-bit format the rest of the model uses cuts that to 239 MB, and
 * costs nothing that matters: the head only proposes, so quantisation error can
 * change which token it suggests but never which one is emitted.  It is the one
 * place in this engine where precision is purely an efficiency question.
 *
 * The format is described at the top of qwasar_gpu.h.  Groups are 64 wide along
 * the input dimension, the scale and bias are stored bf16, and the nibble is
 * unsigned -- so the bias carries the group's minimum rather than a zero point.
 *
 * Scale and bias are rounded to bf16 BEFORE the nibbles are chosen.  Choosing
 * them against the full-precision scale and then storing a rounded one would
 * bias every weight in the group by whatever the rounding moved. */

static uint16_t qw_f32_to_bf16_c(float v) {
    uint32_t u;
    memcpy(&u, &v, sizeof u);
    return (uint16_t)((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
}

static void qw_quant_row(const uint16_t *src, int32_t in_features,
                         uint32_t *w_out, uint16_t *sc_out, uint16_t *bi_out) {
    const int32_t groups = in_features / 64;
    for (int32_t g = 0; g < groups; g++) {
        const uint16_t *v = src + (size_t)g * 64;
        float lo = qw_bf16_to_f32_c(v[0]), hi = lo;
        for (int j = 1; j < 64; j++) {
            const float x = qw_bf16_to_f32_c(v[j]);
            if (x < lo) lo = x;
            if (x > hi) hi = x;
        }
        const uint16_t bi_b = qw_f32_to_bf16_c(lo);
        const float    bi   = qw_bf16_to_f32_c(bi_b);
        const uint16_t sc_b = qw_f32_to_bf16_c((hi - lo) / 15.0f);
        const float    sc   = qw_bf16_to_f32_c(sc_b);
        sc_out[g] = sc_b;
        bi_out[g] = bi_b;

        const float inv = sc > 0.0f ? 1.0f / sc : 0.0f;
        for (int j = 0; j < 64; j += 8) {
            uint32_t word = 0;
            for (int t = 0; t < 8; t++) {
                float q = (qw_bf16_to_f32_c(v[j + t]) - bi) * inv + 0.5f;
                int32_t n = (int32_t)q;
                if (n < 0) n = 0;
                if (n > 15) n = 15;
                word |= (uint32_t)n << (4 * t);
            }
            w_out[(g * 64 + j) / 8] = word;
        }
    }
}

/* Fills `ql` from `d`, writing into the arena at `*off` and advancing it. */
static void qw_quant_dense(qwasar_engine *e, qw_qlinear *ql, const qw_dense *d,
                           qw_tensor *slots, qw_buf buf, size_t *off) {
    const int32_t in = d->in_features, out = d->out_features;
    const int32_t words = in / 8, groups = in / 64;
    const size_t w_bytes  = (size_t)out * words  * 4;
    const size_t sb_bytes = (size_t)out * groups * 2;

    char *base = qw_buf_contents(buf);
    qw_tensor *tw = &slots[0], *ts = &slots[1], *tb = &slots[2];
    *tw = (qw_tensor){ "mtp.weight", buf, *off,            w_bytes,  QW_DT_U32,  2, { out, words } };
    *ts = (qw_tensor){ "mtp.scales", buf, *off + w_bytes,  sb_bytes, QW_DT_BF16, 2, { out, groups } };
    *tb = (qw_tensor){ "mtp.biases", buf, *off + w_bytes + sb_bytes, sb_bytes, QW_DT_BF16, 2, { out, groups } };

    const uint16_t *src = qw_tensor_data(d->weight);
    uint32_t *w  = (uint32_t *)(base + tw->offset);
    uint16_t *sc = (uint16_t *)(base + ts->offset);
    uint16_t *bi = (uint16_t *)(base + tb->offset);
    for (int32_t r = 0; r < out; r++)
        qw_quant_row(src + (size_t)r * in, in,
                     w + (size_t)r * words, sc + (size_t)r * groups,
                     bi + (size_t)r * groups);

    ql->weight = tw; ql->scales = ts; ql->biases = tb;
    ql->in_features = in; ql->out_features = out;
    *off += w_bytes + 2 * sb_bytes;
    e->weight_bytes_mtp_q4 += w_bytes + 2 * sb_bytes;
}

static bool qw_quantise_mtp(qwasar_engine *e, char *err, size_t errcap) {
    qw_mtp *m = &e->mtp;
    qw_dense *dense[8] = { &m->fc, &m->q_proj, &m->k_proj, &m->v_proj,
                           &m->o_proj, &m->gate_proj, &m->up_proj, &m->down_proj };
    qw_qlinear *out[8] = { &m->q_fc, &m->q_q, &m->q_k, &m->q_v,
                           &m->q_o, &m->q_gate, &m->q_up, &m->q_down };

    size_t total = 0;
    for (int i = 0; i < 8; i++) {
        const int32_t in = dense[i]->in_features, o = dense[i]->out_features;
        total += (size_t)o * (in / 8) * 4 + 2 * (size_t)o * (in / 64) * 2;
    }
    m->q4 = qw_buf_alloc(total);
    if (!m->q4) {
        qw_errf(err, errcap, "cannot allocate %.0f MB for the quantised MTP head",
                (double)total / 1e6);
        return false;
    }

    size_t off = 0;
    for (int i = 0; i < 8; i++)
        qw_quant_dense(e, out[i], dense[i], &m->q4t[i * 3], m->q4, &off);
    m->quantised = true;
    return true;
}

static bool qw_load_mtp(qwasar_engine *e, const char *dir, char *err, size_t errcap) {
    const qw_config *c = &e->config;
    qw_mtp *m = &e->mtp;
    char path[1200];

    /* The head's own config is not a load input -- the head is built from the
     * base model's dimensions -- but block_size lives only there, and reading it
     * is also the cheapest check that this directory is a head and not, say,
     * another copy of the model. */
    snprintf(path, sizeof path, "%s/config.json", dir);
    qj_doc cfg;
    if (!qj_parse_file(&cfg, path)) {
        qw_errf(err, errcap, "cannot read %s", path);
        return false;
    }
    char mt[64] = "";
    qj_str_copy(&cfg, qj_root(&cfg), "model_type", mt, sizeof mt);
    if (strcmp(mt, "qwen3_5_mtp")) {
        qw_errf(err, errcap, "%s declares model_type \"%s\", expected qwen3_5_mtp",
                path, mt);
        qj_free(&cfg);
        return false;
    }
    m->block_size = (int32_t)qj_int_or(&cfg, qj_root(&cfg), "block_size", 0);
    qj_free(&cfg);
    if (m->block_size < 1) {
        qw_errf(err, errcap, "MTP head declares no usable block_size");
        return false;
    }

    snprintf(path, sizeof path, "%s/model.safetensors", dir);
    const uint32_t before = e->tensors.count;
    if (!qw_load_shard(e, path, err, errcap)) return false;
    if (!qw_table_index(&e->tensors)) { qw_errf(err, errcap, "out of memory"); return false; }

    /* Fifteen tensors, no more and no less.  A head that gained or lost one is
     * not a head this code understands, and drafting from a partly bound one
     * would show up only as a mysteriously bad acceptance rate. */
    const uint32_t added = e->tensors.count - before;
    if (added != 15) {
        qw_errf(err, errcap, "MTP head has %u tensors, expected 15", added);
        return false;
    }

    const int32_t h  = c->hidden_size;
    const int32_t qd = c->num_attention_heads * c->head_dim;

    /* q_proj emits query and gate together, exactly as the base model's full
     * attention layers do (attn_output_gate). */
    if (!qw_bind_dense(e, &m->fc,        "fc.weight",                          2 * h, h, err, errcap)
     || !qw_bind_dense(e, &m->q_proj,    "layers.0.self_attn.q_proj.weight",   h, 2 * qd, err, errcap)
     || !qw_bind_dense(e, &m->k_proj,    "layers.0.self_attn.k_proj.weight",   h, c->num_key_value_heads * c->head_dim, err, errcap)
     || !qw_bind_dense(e, &m->v_proj,    "layers.0.self_attn.v_proj.weight",   h, c->num_key_value_heads * c->head_dim, err, errcap)
     || !qw_bind_dense(e, &m->o_proj,    "layers.0.self_attn.o_proj.weight",   qd, h, err, errcap)
     || !qw_bind_dense(e, &m->gate_proj, "layers.0.mlp.gate_proj.weight",      h, c->intermediate_size, err, errcap)
     || !qw_bind_dense(e, &m->up_proj,   "layers.0.mlp.up_proj.weight",        h, c->intermediate_size, err, errcap)
     || !qw_bind_dense(e, &m->down_proj, "layers.0.mlp.down_proj.weight",      c->intermediate_size, h, err, errcap))
        return false;

    m->pre_fc_norm_hidden    = qw_bind_mtp_norm(e, "pre_fc_norm_hidden.weight", h, err, errcap);
    m->pre_fc_norm_embedding = qw_bind_mtp_norm(e, "pre_fc_norm_embedding.weight", h, err, errcap);
    m->norm                  = qw_bind_mtp_norm(e, "norm.weight", h, err, errcap);
    m->input_layernorm       = qw_bind_mtp_norm(e, "layers.0.input_layernorm.weight", h, err, errcap);
    m->post_attention_layernorm =
        qw_bind_mtp_norm(e, "layers.0.post_attention_layernorm.weight", h, err, errcap);
    m->q_norm = qw_bind_mtp_norm(e, "layers.0.self_attn.q_norm.weight", c->head_dim, err, errcap);
    m->k_norm = qw_bind_mtp_norm(e, "layers.0.self_attn.k_norm.weight", c->head_dim, err, errcap);

    if (!m->pre_fc_norm_hidden || !m->pre_fc_norm_embedding || !m->norm
        || !m->input_layernorm || !m->post_attention_layernorm
        || !m->q_norm || !m->k_norm)
        return false;

    if (!qw_quantise_mtp(e, err, errcap)) return false;

    m->present = true;
    /* The bf16 originals stay mapped but are never touched again, so the
     * quantised copy is what the footprint should report. */
    e->weight_bytes += e->weight_bytes_mtp_q4;
    return true;
}

static bool qw_bind_weights(qwasar_engine *e, char *err, size_t errcap) {
    const qw_config *c = &e->config;
    const qw_shape  *s = &e->shape;

    if (!qw_bind_qlinear(e, &e->embed_tokens, "language_model.model.embed_tokens",
                         c->hidden_size, c->vocab_size, err, errcap)) return false;
    if (!c->tie_word_embeddings
        && !qw_bind_qlinear(e, &e->lm_head, "language_model.lm_head",
                            c->hidden_size, c->vocab_size, err, errcap)) return false;
    if (c->tie_word_embeddings) e->lm_head = e->embed_tokens;

    e->final_norm = qw_table_find(&e->tensors, "language_model.model.norm.weight");
    if (!e->final_norm) { qw_errf(err, errcap, "missing language_model.model.norm.weight"); return false; }
    e->weight_bytes_text += e->final_norm->nbytes;

    char p[256];
    for (int i = 0; i < c->num_hidden_layers; i++) {
        qw_layer *L = &e->layers[i];
        memset(L, 0, sizeof *L);
        L->is_linear_attn = qw_layer_is_linear(c, i);

        L->input_layernorm = qw_bind_norm(e,
            "language_model.model.layers.%d.input_layernorm.weight", i,
            c->hidden_size, err, errcap);
        if (!L->input_layernorm) return false;
        L->post_attention_layernorm = qw_bind_norm(e,
            "language_model.model.layers.%d.post_attention_layernorm.weight", i,
            c->hidden_size, err, errcap);
        if (!L->post_attention_layernorm) return false;

        if (L->is_linear_attn) {
            snprintf(p, sizeof p, "language_model.model.layers.%d.linear_attn.in_proj_qkv", i);
            if (!qw_bind_qlinear(e, &L->in_proj_qkv, p, c->hidden_size, s->conv_dim, err, errcap)) return false;
            snprintf(p, sizeof p, "language_model.model.layers.%d.linear_attn.in_proj_z", i);
            if (!qw_bind_qlinear(e, &L->in_proj_z, p, c->hidden_size, s->value_dim, err, errcap)) return false;
            snprintf(p, sizeof p, "language_model.model.layers.%d.linear_attn.in_proj_b", i);
            if (!qw_bind_qlinear(e, &L->in_proj_b, p, c->hidden_size, c->linear_num_value_heads, err, errcap)) return false;
            snprintf(p, sizeof p, "language_model.model.layers.%d.linear_attn.in_proj_a", i);
            if (!qw_bind_qlinear(e, &L->in_proj_a, p, c->hidden_size, c->linear_num_value_heads, err, errcap)) return false;
            snprintf(p, sizeof p, "language_model.model.layers.%d.linear_attn.out_proj", i);
            if (!qw_bind_qlinear(e, &L->out_proj, p, s->value_dim, c->hidden_size, err, errcap)) return false;

            L->conv1d  = qw_table_findf(&e->tensors, "language_model.model.layers.%d.linear_attn.conv1d.weight", i);
            L->A_log   = qw_table_findf(&e->tensors, "language_model.model.layers.%d.linear_attn.A_log", i);
            L->dt_bias = qw_table_findf(&e->tensors, "language_model.model.layers.%d.linear_attn.dt_bias", i);
            L->gdn_norm = qw_bind_norm(e,
                "language_model.model.layers.%d.linear_attn.norm.weight", i,
                c->linear_value_head_dim, err, errcap);
            if (!L->conv1d || !L->A_log || !L->dt_bias || !L->gdn_norm) {
                qw_errf(err, errcap, "layer %d: missing gated-delta tensors", i);
                return false;
            }
            /* [conv_dim, K, 1]: MLX stores the depthwise taps transposed, tap 0
             * oldest.  A [conv_dim, 1, K] here would mean an unsanitised
             * checkpoint and a silently wrong convolution. */
            if (L->conv1d->ndim != 3 || L->conv1d->shape[0] != s->conv_dim
                || L->conv1d->shape[1] != c->linear_conv_kernel_dim
                || L->conv1d->shape[2] != 1) {
                qw_errf(err, errcap,
                        "layer %d: conv1d.weight is [%lld,%lld,%lld], expected [%d,%d,1]", i,
                        (long long)L->conv1d->shape[0], (long long)L->conv1d->shape[1],
                        (long long)L->conv1d->shape[2], s->conv_dim, c->linear_conv_kernel_dim);
                return false;
            }
            e->weight_bytes_text += L->conv1d->nbytes + L->A_log->nbytes + L->dt_bias->nbytes;
        } else {
            snprintf(p, sizeof p, "language_model.model.layers.%d.self_attn.q_proj", i);
            if (!qw_bind_qlinear(e, &L->q_proj, p, c->hidden_size, s->q_proj_out, err, errcap)) return false;
            snprintf(p, sizeof p, "language_model.model.layers.%d.self_attn.k_proj", i);
            if (!qw_bind_qlinear(e, &L->k_proj, p, c->hidden_size, s->kv_dim, err, errcap)) return false;
            snprintf(p, sizeof p, "language_model.model.layers.%d.self_attn.v_proj", i);
            if (!qw_bind_qlinear(e, &L->v_proj, p, c->hidden_size, s->kv_dim, err, errcap)) return false;
            snprintf(p, sizeof p, "language_model.model.layers.%d.self_attn.o_proj", i);
            if (!qw_bind_qlinear(e, &L->o_proj, p, s->q_dim, c->hidden_size, err, errcap)) return false;

            L->q_norm = qw_bind_norm(e, "language_model.model.layers.%d.self_attn.q_norm.weight",
                                     i, c->head_dim, err, errcap);
            if (!L->q_norm) return false;
            L->k_norm = qw_bind_norm(e, "language_model.model.layers.%d.self_attn.k_norm.weight",
                                     i, c->head_dim, err, errcap);
            if (!L->k_norm) return false;
        }

        snprintf(p, sizeof p, "language_model.model.layers.%d.mlp.gate_proj", i);
        if (!qw_bind_qlinear(e, &L->gate_proj, p, c->hidden_size, c->intermediate_size, err, errcap)) return false;
        snprintf(p, sizeof p, "language_model.model.layers.%d.mlp.up_proj", i);
        if (!qw_bind_qlinear(e, &L->up_proj, p, c->hidden_size, c->intermediate_size, err, errcap)) return false;
        snprintf(p, sizeof p, "language_model.model.layers.%d.mlp.down_proj", i);
        if (!qw_bind_qlinear(e, &L->down_proj, p, c->intermediate_size, c->hidden_size, err, errcap)) return false;
    }

    /* The vision tower is unquantised and is not bound until Milestone 3; count
     * its bytes so --info reports the true resident footprint. */
    for (uint32_t i = 0; i < e->tensors.count; i++)
        if (!strncmp(e->tensors.v[i].name, "vision_tower.", 13))
            e->weight_bytes_vision += e->tensors.v[i].nbytes;

    e->weight_bytes = e->weight_bytes_text + e->weight_bytes_vision;
    return true;
}

/* ---- finding the model ----------------------------------------------------
 *
 * download_model.sh leaves a `qwasar-model` symlink beside the binaries, which
 * is what makes `./qwasar -p "Hello"` work with no arguments.  Two places are
 * searched because the two obvious ways to run this disagree about which one is
 * right: from a checkout the working directory is the project, but a binary
 * copied onto a PATH is run from anywhere, and only its own directory still
 * points at the download. */

#define QW_MODEL_LINK "qwasar-model"

static bool qw_is_model_dir(const char *dir) {
    char probe[1200];
    struct stat st;
    snprintf(probe, sizeof probe, "%s/config.json", dir);
    return stat(probe, &st) == 0 && S_ISREG(st.st_mode);
}

const char *qwasar_default_model_path(void) {
    static char buf[1200];

    /* An explicit setting is returned whether or not it exists.  The user said
     * where the model is; a failure should name their path rather than quietly
     * fall through and complain about a default they never mentioned. */
    const char *env = getenv("QWASAR_MODEL");
    if (env && *env) {
        snprintf(buf, sizeof buf, "%s", env);
        return buf;
    }

    if (qw_is_model_dir(QW_MODEL_LINK)) {
        snprintf(buf, sizeof buf, "%s", QW_MODEL_LINK);
        return buf;
    }

    char exe[1024];
    uint32_t cap = sizeof exe;
    if (_NSGetExecutablePath(exe, &cap) == 0) {
        char *slash = strrchr(exe, '/');
        if (slash) {
            *slash = 0;
            snprintf(buf, sizeof buf, "%s/%s", exe, QW_MODEL_LINK);
            if (qw_is_model_dir(buf)) return buf;
        }
    }
    return NULL;
}

/* ---- lifecycle ------------------------------------------------------------ */

qwasar_engine *qwasar_engine_load(const qwasar_options *opts, char *err, size_t errcap) {
    if (!opts || !opts->model_path) {
        qw_errf(err, errcap, "no model path given");
        return NULL;
    }
    if (!qw_gpu_init(err, errcap)) return NULL;

    qwasar_engine *e = calloc(1, sizeof *e);
    if (!e) { qw_errf(err, errcap, "out of memory"); return NULL; }
    e->opts = *opts;
    snprintf(e->model_path, sizeof e->model_path, "%s", opts->model_path);

    if (!qw_load_config(e, err, errcap))   goto fail;

    /* Context length is the caller's choice, bounded by what the model was
     * trained for.  The full 262144 would be a 16 GB KV cache and does not fit
     * alongside the weights, so the default is deliberately far lower. */
    e->context_size = opts->context_size > 0 ? opts->context_size : 32768;
    if (e->context_size > e->config.max_position_embeddings)
        e->context_size = e->config.max_position_embeddings;
    e->prefill_chunk = opts->prefill_chunk > 0 ? opts->prefill_chunk : 256;

    if (!qw_load_shards(e, err, errcap))   goto fail;
    if (!qw_table_index(&e->tensors))      { qw_errf(err, errcap, "out of memory"); goto fail; }
    if (!qw_bind_weights(e, err, errcap))  goto fail;
    if (opts->mtp_path && *opts->mtp_path
        && !qw_load_mtp(e, opts->mtp_path, err, errcap)) goto fail;
    return e;

fail:
    qwasar_engine_free(e);
    return NULL;
}

void qwasar_engine_free(qwasar_engine *e) {
    if (!e) return;
    for (int i = 0; i < e->n_shards; i++) {
        if (e->shards[i].buf)  qw_buf_free(e->shards[i].buf);
        if (e->shards[i].addr) munmap(e->shards[i].addr, e->shards[i].len);
    }
    qw_table_free(&e->tensors);
    qw_arena_free(&e->names);
    free(e);
}

/* ---- internal accessors ---------------------------------------------------
 *
 * The graph code and the tests need to see inside the engine; qwasar.h's
 * consumers do not.  These are declared in qwasar_model.h. */

const qw_config *qwasar_engine_config(const qwasar_engine *e) { return &e->config; }
const qw_qlinear *qwasar_engine_embed(const qwasar_engine *e) { return &e->embed_tokens; }
const qw_qlinear *qwasar_engine_head (const qwasar_engine *e) { return &e->lm_head; }
const qw_tensor  *qwasar_engine_final_norm(const qwasar_engine *e) { return e->final_norm; }
const qw_mtp     *qwasar_engine_mtp(const qwasar_engine *e) { return &e->mtp; }
int32_t qwasar_engine_context_size (const qwasar_engine *e) { return e->context_size; }
int32_t qwasar_engine_prefill_chunk(const qwasar_engine *e) { return e->prefill_chunk; }

int32_t qwasar_vocab_size(const qwasar_engine *e) { return e->config.vocab_size; }
int32_t qwasar_n_layers  (const qwasar_engine *e) { return e->config.num_hidden_layers; }

bool qwasar_is_eos(const qwasar_engine *e, int32_t token) {
    for (int32_t i = 0; i < e->config.n_eos; i++)
        if (e->config.eos_token_ids[i] == token) return true;
    return false;
}

const qw_shape  *qwasar_engine_shape (const qwasar_engine *e) { return &e->shape;  }

const qw_layer *qwasar_engine_layer(const qwasar_engine *e, int index) {
    if (index < 0 || index >= e->config.num_hidden_layers) return NULL;
    return &e->layers[index];
}

const qw_tensor *qwasar_engine_tensor(const qwasar_engine *e, const char *name) {
    return qw_table_find(&e->tensors, name);
}

const void *qw_tensor_data(const qw_tensor *t) {
    if (!t) return NULL;
    const char *base = qw_buf_contents(t->buf);
    return base ? base + t->offset : NULL;
}

qw_ref qw_tensor_ref(const qw_tensor *t) {
    qw_ref r = { t ? t->buf : NULL, t ? t->offset : 0 };
    return r;
}

/* ---- info ----------------------------------------------------------------- */

static double qw_gb(size_t bytes) { return (double)bytes / (1024.0 * 1024.0 * 1024.0); }

void qwasar_engine_print_info(const qwasar_engine *e, FILE *out) {
    const qw_config *c = &e->config;
    const qw_shape  *s = &e->shape;

    fprintf(out, "model      %s\n", e->model_path);
    fprintf(out, "device     %s  (working set %.1f GB, max buffer %.1f GB)\n",
            qw_gpu_name(), qw_gb(qw_gpu_working_set_limit()), qw_gb(qw_gpu_max_buffer_length()));
    fprintf(out, "shards     %d, %u tensors  (%.2f GB mapped, %.2f GB copied for alignment)\n",
            e->n_shards, e->tensors.count, qw_gb(e->bytes_mapped), qw_gb(e->bytes_copied));
    fprintf(out, "\n");

    fprintf(out, "text       hidden %d  layers %d  vocab %d  ffn %d  eps %.0e\n",
            c->hidden_size, c->num_hidden_layers, c->vocab_size,
            c->intermediate_size, (double)c->rms_norm_eps);
    fprintf(out, "schedule   full attention every %d layers -> %d full, %d gated-delta\n",
            c->full_attention_interval, s->n_full_attn_layers, s->n_linear_attn_layers);
    fprintf(out, "attention  %d q heads x %d dim, %d kv heads (gqa %d), output gate %s\n",
            c->num_attention_heads, c->head_dim, c->num_key_value_heads,
            s->gqa_factor, c->attn_output_gate ? "on" : "off");
    fprintf(out, "rope       theta %.0f  partial %.2f -> rotate %d of %d dims  "
                 "mrope %s [%d,%d,%d]\n",
            (double)c->rope_theta, (double)c->partial_rotary_factor,
            c->rotary_dim, c->head_dim,
            c->mrope_interleaved ? "interleaved" : "chunked",
            c->mrope_section[0], c->mrope_section[1], c->mrope_section[2]);
    fprintf(out, "delta-net  %d v heads x %d, %d k heads x %d (gqa %d), conv k=%d over %d ch\n",
            c->linear_num_value_heads, c->linear_value_head_dim,
            c->linear_num_key_heads, c->linear_key_head_dim,
            s->gdn_gqa_factor, c->linear_conv_kernel_dim, s->conv_dim);
    fprintf(out, "quant      MLX affine %d-bit, group %d\n",
            c->quant_bits, c->quant_group_size);

    if (c->has_vision)
        fprintf(out, "vision     %d blocks x %d, %d heads, patch %d (t%d), merge %d -> %d\n",
                c->vis_depth, c->vis_hidden_size, c->vis_num_heads, c->vis_patch_size,
                c->vis_temporal_patch_size, c->vis_spatial_merge_size, c->vis_out_hidden_size);

    fprintf(out, "\n");
    if (e->mtp.present)
        fprintf(out, "mtp head   1 full-attention layer, %.0f MB bf16 quantised to "
                     "%.0f MB 4-bit\n",
                qw_gb(e->weight_bytes_mtp) * 1024.0,
                qw_gb(e->weight_bytes_mtp_q4) * 1024.0);

    fprintf(out, "weights    %.2f GB text + %.2f GB vision",
            qw_gb(e->weight_bytes_text), qw_gb(e->weight_bytes_vision));
    if (e->mtp.present) fprintf(out, " + %.2f GB mtp", qw_gb(e->weight_bytes_mtp_q4));
    fprintf(out, " = %.2f GB\n", qw_gb(e->weight_bytes));

    /* Per-token cache cost.  The hybrid schedule is what makes long context
     * affordable: only the full-attention layers grow with position, and the
     * recurrent state is constant no matter how long the conversation runs. */
    size_t kv_per_token = (size_t)s->n_full_attn_layers * 2 * (size_t)s->kv_dim * sizeof(uint16_t);
    size_t ssm_state = (size_t)s->n_linear_attn_layers *
        ((size_t)c->linear_num_value_heads * c->linear_value_head_dim * c->linear_key_head_dim * 4
         + (size_t)(c->linear_conv_kernel_dim - 1) * s->conv_dim * 2);
    fprintf(out, "kv cache   %.0f KB/token (%.2f GB at 32K)\n",
            kv_per_token / 1024.0, qw_gb(kv_per_token * 32768));
    fprintf(out, "ssm state  %.0f MB, constant in context length\n",
            ssm_state / (1024.0 * 1024.0));

    fprintf(out, "\n");
    fprintf(out, "eos        ");
    for (int i = 0; i < c->n_eos; i++) fprintf(out, "%s%d", i ? ", " : "", c->eos_token_ids[i]);
    fprintf(out, "\n");
}
