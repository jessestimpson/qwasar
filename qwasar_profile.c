/* qwasar_profile -- context per session and the live-session budget, from the
 * model and the machine.
 *
 * Crucible's MemoryProfile.derive (PLAN.md 2.3), ported: the server is the
 * one process that knows how many sessions exist, so it is the one to size
 * them.  Reads config.json and the safetensors headers -- the same bytes the
 * engine's own memory check reads -- and touches no weights. */

#include "qwasar_sessions.h"
#include "qwasar_gpu.h"

#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <unistd.h>

/* Bytes of a shard the GPU will hold: every tensor but the host-only ones --
 * a file whose metadata places it on the cpu, and any tensor of the engram
 * table (`.ngram_embedding.`), which MLX's build mixes into model files. */
static bool shard_device_bytes(const char *path, uint64_t *bytes) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return false;
    uint64_t header_len = 0;
    struct stat st;
    if (fstat(fd, &st) != 0 || pread(fd, &header_len, 8, 0) != 8
        || header_len == 0 || header_len > (uint64_t)st.st_size - 8 || header_len > (512u << 20)) {
        close(fd);
        return false;
    }
    char *header = malloc(header_len + 1);
    if (!header || (uint64_t)pread(fd, header, header_len, 8) != header_len) {
        free(header); close(fd);
        return false;
    }
    close(fd);
    qj_doc doc;
    const bool ok = qj_parse(&doc, header, header_len);
    free(header);
    if (!ok) { qj_free(&doc); return false; }
    const qj_node *meta = qj_get(&doc, qj_root(&doc), "__metadata__");
    char placement[16] = "";
    if (meta) qj_str_copy(&doc, meta, "placement", placement, sizeof placement);
    if (!strcmp(placement, "cpu")) { qj_free(&doc); return true; }
    static const char mark[] = ".ngram_embedding.";
    for (const qj_node *m = qj_first(&doc, qj_root(&doc)); m; m = qj_next(&doc, m)) {
        if (m->type != QJ_OBJECT) continue;
        bool host = false;
        for (uint32_t i = 0; i + sizeof mark - 1 <= m->key_len; i++)
            if (!memcmp(doc.text + m->key_off + i, mark, sizeof mark - 1)) { host = true; break; }
        if (host) continue;
        const qj_node *offs = qj_get(&doc, m, "data_offsets");
        const qj_node *o0 = offs ? qj_idx(&doc, offs, 0) : NULL, *o1 = offs ? qj_idx(&doc, offs, 1) : NULL;
        if (o0 && o1 && o1->u.num > o0->u.num) *bytes += (uint64_t)(o1->u.num - o0->u.num);
    }
    qj_free(&doc);
    return true;
}

static bool resident_bytes(const char *dir, uint64_t *out) {
    DIR *d = opendir(dir);
    if (!d) return false;
    uint64_t total = 0;
    int shards = 0;
    bool ok = true;
    struct dirent *ent;
    while (ok && (ent = readdir(d))) {
        const size_t n = strlen(ent->d_name);
        if (n < 12 || strcmp(ent->d_name + n - 12, ".safetensors")) continue;
        char path[1400];
        snprintf(path, sizeof path, "%s/%s", dir, ent->d_name);
        ok = shard_device_bytes(path, &total);
        shards++;
    }
    closedir(d);
    if (!ok || shards == 0) return false;
    *out = total;
    return true;
}

/* Context stepped down to a multiple of 8192 -- a tidy number to show a
 * user -- and never past the model's window. */
static int32_t context_fitting(double budget, int sessions, uint64_t fixed,
                               int32_t ceiling, uint64_t per_token) {
    const double each = budget / sessions - (double)fixed;
    if (each <= 0) return 0;
    double raw = each / (double)per_token;
    if (raw > 2147483647.0) raw = 2147483647.0;
    int32_t stepped = ((int32_t)raw / 8192) * 8192;
    return stepped < ceiling ? stepped : ceiling;
}

bool qw_profile_derive(const char *model_path, uint64_t working_set, qw_profile *p,
                       char *err, size_t errcap) {
    memset(p, 0, sizeof *p);
    char path[1300];
    snprintf(path, sizeof path, "%s/config.json", model_path);
    qj_doc d;
    if (!qj_parse_file(&d, path)) {
        snprintf(err, errcap, "cannot read %s: %s", path, d.err);
        qj_free(&d);
        return false;
    }
    const qj_node *root = qj_root(&d);
    const qj_node *text = qj_get(&d, root, "text_config");
    if (!text) text = root;
    char type[32] = "";
    qj_str_copy(&d, root, "model_type", type, sizeof type);
    const bool flash = !strncmp(type, "qwen4_exp", 9);
    snprintf(p->family, sizeof p->family, "%s", flash ? "qwen4_exp" : "qwen3_5");
    p->max_ctx = (int32_t)qj_int_or(&d, text, "max_position_embeddings", 262144);
    if (p->max_ctx <= 0) p->max_ctx = 262144;
    qj_free(&d);

    /* Per token: the KV cache -- full-attention layers x KV heads x head dim
     * x (K+V) x fp16 -- plus, for Flash-Next, the indexer's key cache, its
     * block scores and the attention mask: 64 KB and 32 KB respectively.
     * Fixed: the delta layers' recurrent state and a prefill chunk's scratch,
     * 351 MB and ~1.1 GB (the allocations in qwasar_graph.c and
     * qwasar_flash_graph.c). */
    p->kv_per_token = flash ? 32 * 1024 : 64 * 1024;
    p->fixed = flash ? 1100000000ull : 351ull * 1024 * 1024;
    p->reserve = 0.85;

    if (!resident_bytes(model_path, &p->weights))
        p->weights = flash ? 79520000000ull : 16020000000ull;

    size_t len = sizeof p->physical;
    if (sysctlbyname("hw.memsize", &p->physical, &len, NULL, 0) != 0) p->physical = 0;
    /* Metal, brought up early (qw_gpu_init is idempotent; the engine load
     * finds it done), for the working set the engine itself budgets by. */
    char gerr[128];
    p->working_set = working_set ? working_set
                   : qw_gpu_init(gerr, sizeof gerr) ? qw_gpu_working_set_limit() : 0;
    if (!p->working_set) p->working_set = p->physical * 84 / 100;

    const double usable = (double)p->working_set * p->reserve;
    const double for_sessions = usable - (double)p->weights;
    if (for_sessions <= (double)p->fixed) {
        p->ctx = 4096;
        p->live = 1;
        snprintf(p->note, sizeof p->note,
                 "this machine cannot hold the weights and a usable session at once");
        return true;
    }

    /* Context first, then live sessions -- and another live session only
     * where it still leaves the model's whole window. */
    int live = 1;
    int32_t ctx = context_fitting(for_sessions, 1, p->fixed, p->max_ctx, p->kv_per_token);
    while (live < 4) {
        const int32_t c = context_fitting(for_sessions, live + 1, p->fixed, p->max_ctx, p->kv_per_token);
        if (c >= p->max_ctx) { live++; ctx = c; } else break;
    }
    if (ctx < 4096) ctx = 4096;
    p->ctx = ctx;
    p->live = live;
    return true;
}
