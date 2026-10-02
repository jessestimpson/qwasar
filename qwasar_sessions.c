/* qwasar_sessions -- the session store, the scheduler, and generation.
 *
 * What qwasar_server.c used to keep in one `server` struct -- one live
 * qwasar_session, its checkpoint mark, the engine lock -- generalised to
 * many named sessions and one anonymous one, with the policy API.md §6
 * states: one step at a time, a bounded live set with LRU parking,
 * checkpoints at boundaries, the token log after every step.
 *
 * Threads: a connection thread runs its own step end to end while it holds
 * the engine (qw_store_acquire).  The store's own lock covers the session
 * list and each session's fields; a session's event log has a lock and a
 * condition of its own so a reattaching reader never waits on the engine. */

#include "qwasar_sessions.h"

#include <math.h>

#include <dirent.h>
#include <errno.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mount.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

/* ---- log and clock --------------------------------------------------------- */

double qw_now(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + tv.tv_usec * 1e-6;
}

void qw_log(const char *fmt, ...) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    struct tm tm;
    localtime_r(&tv.tv_sec, &tm);
    char when[32], line[1024];
    strftime(when, sizeof when, "%Y-%m-%d %H:%M:%S", &tm);
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(line, sizeof line, fmt, ap);
    va_end(ap);
    fprintf(stderr, "[%s.%03d] %s\n", when, (int)(tv.tv_usec / 1000), line);
}

static atomic_bool g_stopping;
bool qw_store_stopping(void) { return atomic_load(&g_stopping); }

/* ---- structures ------------------------------------------------------------ */

typedef struct {
    int   step, seq;
    char *name;
    char *json;
} qw_event;

struct qw_sess {
    char     id[24];
    char    *system;
    char    *tools_json;        /* "[...]", the tools as given */
    int32_t  n_tools;
    qj_doc   tools_doc;         /* {"tools": [...]} for names and argument types */
    bool     has_tools_doc;
    bool     thinking;
    char     effort[8];
    char    *metadata;          /* JSON object text */
    char     model_id[32];
    int64_t  created;

    int32_t *tokens;            /* the timeline */
    int32_t  n_tokens, cap_tokens;
    int32_t  prefix_n;
    int      step;
    qw_sess_state state;
    char     last_stop[16];
    int64_t  last_at;
    qw_pending_call pending[QW_MAX_CALLS];
    int      n_pending;

    qwasar_session *h;          /* live handle, or NULL */
    double   last_used;
    int32_t  ckpt_n;            /* tokens covered by the last checkpoint this run wrote */
    /* Tokens the session's OWN checkpoint file covers, as far as this run
     * knows -- kept apart from ckpt_n, which the shared cache can satisfy:
     * a session whose state is only in the cache is one eviction from cold. */
    int32_t  own_n;
    bool     compat;

    /* The step in flight: its events, for the connection and for reattach. */
    qw_event *ev;
    int       n_ev, cap_ev;
    pthread_mutex_t ev_lock;
    pthread_cond_t  ev_cond;
    bool      step_open;
    int       seq;
    qw_emit_fn emit;
    void     *emit_ud;
    bool      peer_gone;
    atomic_bool cancel;
    /* An aside (qw_sess_aside): a step taken off the record and rolled back.
     * `aside` keeps its tokens out of the timeline; `aside_running` is what
     * a real step waits on, after setting `aside_cancel` to end it. */
    bool        aside, aside_running;
    atomic_bool aside_cancel;
    /* Prefill progress, rebased over the whole of a resume. */
    int32_t   prog_base, prog_total;
};

typedef struct waiter {
    qw_sess *s;
    struct waiter *next;
} waiter;

struct qw_store {
    qwasar_engine    *e;
    qwasar_tokenizer *tok;
    char     model_path[1024];
    char     model_id[32];
    qw_profile prof;
    int32_t  ctx;
    int      live_max;
    bool     no_cache, verbose;
    char     dir[1024];

    pthread_mutex_t lock;       /* the list and every session's fields */
    qw_sess **sess;
    int       n_sess, cap_sess;
    qw_sess  *compat;

    /* The engine: one holder, a FIFO of waiters. */
    pthread_mutex_t qlock;
    pthread_cond_t  qcond;
    waiter  *head, *tail;
    bool     busy;
    qw_sess *holder;
    /* The handle of the session last parked, kept for the next to take:
     * see handle_put.  The engine holder's, like every handle. */
    qwasar_session *spare;

    uint64_t rng;
    /* Measured rates, for resume estimates. */
    double   prefill_tps;       /* tokens/s, from real prefills */
    double   restore_bps;       /* bytes/s, from real restores */

    /* The compat front's last miss (-v). */
    int32_t  miss_at, miss_of;
    char     miss_had[160], miss_got[160];
};

/* ---- small helpers ----------------------------------------------------------- */

static void token_text(qw_store *st, const int32_t *toks, int32_t from, int32_t to,
                       char *out, size_t cap);


static char *xstrdup(const char *s) {
    if (!s) return NULL;
    size_t n = strlen(s);
    char *p = malloc(n + 1);
    if (p) memcpy(p, s, n + 1);
    return p;
}

static bool mkdir_p(const char *path) {
    char buf[1200];
    snprintf(buf, sizeof buf, "%s", path);
    for (char *p = buf + 1; *p; p++) {
        if (*p != '/') continue;
        *p = 0;
        if (mkdir(buf, 0755) != 0 && errno != EEXIST) return false;
        *p = '/';
    }
    return mkdir(buf, 0755) == 0 || errno == EEXIST;
}

static void new_id(char *out, size_t cap) {
    unsigned char r[8];
    arc4random_buf(r, sizeof r);
    snprintf(out, cap, "s_%02x%02x%02x%02x%02x%02x%02x%02x",
             r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]);
}

const char *qw_sess_state_name(qw_sess_state s) {
    switch (s) {
    case QW_SESS_IDLE: return "idle";
    case QW_SESS_QUEUED: return "queued";
    case QW_SESS_RUNNING: return "running";
    case QW_SESS_AWAITING: return "awaiting_tools";
    case QW_SESS_FULL: return "full";
    }
    return "idle";
}

static bool tokens_reserve(qw_sess *s, int32_t n) {
    if (s->n_tokens + n <= s->cap_tokens) return true;
    int32_t cap = s->cap_tokens ? s->cap_tokens : 1024;
    while (cap < s->n_tokens + n) cap *= 2;
    int32_t *p = realloc(s->tokens, (size_t)cap * sizeof *p);
    if (!p) return false;
    s->tokens = p;
    s->cap_tokens = cap;
    return true;
}

static bool tokens_append(qw_sess *s, const int32_t *t, int32_t n) {
    if (n <= 0) return true;
    if (!tokens_reserve(s, n)) return false;
    memcpy(s->tokens + s->n_tokens, t, (size_t)n * sizeof *t);
    s->n_tokens += n;
    return true;
}

/* ---- persistence ------------------------------------------------------------- */

static void sess_dir(const qw_store *st, const qw_sess *s, char *out, size_t cap) {
    snprintf(out, cap, "%s/sessions/%s", st->dir, s->id);
}

/* The session's own checkpoint: written when it is parked -- by request, by
 * eviction, at shutdown -- and on a long conversation's growth; read on
 * resume.  The session's to keep: nothing evicts it, and it goes when the
 * session does or when its checkpoint is dropped (qw_sess_drop_checkpoint). */
static void sess_ckpt_path(const qw_store *st, const qw_sess *s, char *out, size_t cap) {
    snprintf(out, cap, "%s/sessions/%s/checkpoint.bin", st->dir, s->id);
}

static uint64_t file_bytes(const char *path) {
    struct stat stt;
    return stat(path, &stt) == 0 ? (uint64_t)stt.st_size : 0;
}

/* Writes the session's own checkpoint from its live handle. */
static bool sess_save_own(qw_store *st, qw_sess *s, const char *why) {
    char path[1400], dir[1300], err[256];
    sess_dir(st, s, dir, sizeof dir);
    if (!mkdir_p(dir)) return false;
    sess_ckpt_path(st, s, path, sizeof path);
    const double t0 = qw_now();
    const bool saved = qwasar_session_save_file(s->h, st->e, path, err, sizeof err);
    if (saved) s->ckpt_n = s->own_n = qwasar_session_n_past(s->h);
    if (st->verbose || !saved)
        qw_log("  %s: checkpoint (%s) at %d tokens, %.0f MB in %.2fs%s%s", s->id, why,
               qwasar_session_n_past(s->h), file_bytes(path) / 1e6, qw_now() - t0,
               saved ? "" : " -- NOT SAVED: ", saved ? "" : err);
    return saved;
}

/* Writes `data` to `path` through a temporary, so a crash leaves the old
 * file or the new one and never half of either. */
static bool write_file(const char *path, const void *data, size_t n) {
    char tmp[1300];
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    FILE *f = fopen(tmp, "wb");
    if (!f) return false;
    const bool ok = n == 0 || fwrite(data, 1, n, f) == n;
    if (fclose(f) != 0 || !ok) { unlink(tmp); return false; }
    if (rename(tmp, path) != 0) { unlink(tmp); return false; }
    return true;
}

static void record_json(const qw_sess *s, str *b) {
    str_puts(b, "{\"id\": ");
    str_jsons(b, s->id);
    str_printf(b, ", \"created\": %lld, \"model_id\": ", (long long)s->created);
    str_jsons(b, s->model_id);
    str_puts(b, ", \"system\": ");
    str_jsons(b, s->system);
    str_puts(b, ", \"tools\": ");
    str_puts(b, s->tools_json ? s->tools_json : "[]");
    str_printf(b, ", \"thinking\": %s, \"effort\": \"%s\", \"metadata\": %s",
               s->thinking ? "true" : "false", s->effort, s->metadata ? s->metadata : "{}");
    str_printf(b, ", \"prefix_tokens\": %d, \"n_tokens\": %d, \"step\": %d, \"state\": \"%s\"",
               s->prefix_n, s->n_tokens, s->step,
               s->state == QW_SESS_AWAITING ? "awaiting_tools"
             : s->state == QW_SESS_FULL ? "full" : "idle");
    str_puts(b, ", \"last_stop\": ");
    if (s->last_stop[0]) str_jsons(b, s->last_stop); else str_puts(b, "null");
    str_printf(b, ", \"last_at\": %lld, \"pending\": [", (long long)s->last_at);
    for (int i = 0; i < s->n_pending; i++) {
        if (i) str_puts(b, ", ");
        str_puts(b, "{\"id\": ");
        str_jsons(b, s->pending[i].id);
        str_puts(b, ", \"name\": ");
        str_jsons(b, s->pending[i].name);
        str_puts(b, "}");
    }
    str_puts(b, "]}");
}

/* The record and the token log, after every step and on open. */
static bool sess_persist(qw_store *st, qw_sess *s, char *err, size_t errcap) {
    if (s->compat) return true;
    char dir[1300], path[1400];
    sess_dir(st, s, dir, sizeof dir);
    if (!mkdir_p(dir)) {
        snprintf(err, errcap, "cannot create %s: %s", dir, strerror(errno));
        return false;
    }
    snprintf(path, sizeof path, "%s/tokens.bin", dir);
    if (!write_file(path, s->tokens, (size_t)s->n_tokens * sizeof *s->tokens)) {
        snprintf(err, errcap, "cannot write %s: %s", path, strerror(errno));
        return false;
    }
    str b = { 0 };
    record_json(s, &b);
    snprintf(path, sizeof path, "%s/record.json", dir);
    const bool ok = write_file(path, b.p, b.len);
    if (!ok) snprintf(err, errcap, "cannot write %s: %s", path, strerror(errno));
    str_free(&b);
    return ok;
}

/* ---- sessions: construction -------------------------------------------------- */

static bool sess_set_tools(qw_sess *s, const char *tools_json, int32_t n_tools) {
    s->tools_json = xstrdup(tools_json ? tools_json : "[]");
    s->n_tools = n_tools;
    str doc = { 0 };
    str_puts(&doc, "{\"tools\": ");
    str_puts(&doc, s->tools_json);
    str_puts(&doc, "}");
    s->has_tools_doc = qj_parse(&s->tools_doc, doc.p, doc.len);
    str_free(&doc);
    return s->tools_json != NULL;
}

static qw_sess *sess_new(void) {
    qw_sess *s = calloc(1, sizeof *s);
    if (!s) return NULL;
    pthread_mutex_init(&s->ev_lock, NULL);
    pthread_cond_init(&s->ev_cond, NULL);
    snprintf(s->effort, sizeof s->effort, "xhigh");
    return s;
}

static void events_clear(qw_sess *s, int keep_from_step) {
    int w = 0;
    for (int i = 0; i < s->n_ev; i++) {
        if (s->ev[i].step >= keep_from_step) { s->ev[w++] = s->ev[i]; continue; }
        free(s->ev[i].name);
        free(s->ev[i].json);
    }
    s->n_ev = w;
}

static void sess_free(qw_sess *s) {
    if (!s) return;
    if (s->h) qwasar_session_free(s->h);
    events_clear(s, 1 << 30);
    free(s->ev);
    free(s->system);
    free(s->tools_json);
    if (s->has_tools_doc) qj_free(&s->tools_doc);
    free(s->metadata);
    free(s->tokens);
    pthread_mutex_destroy(&s->ev_lock);
    pthread_cond_destroy(&s->ev_cond);
    free(s);
}

static bool list_add(qw_store *st, qw_sess *s) {
    if (st->n_sess == st->cap_sess) {
        int cap = st->cap_sess ? st->cap_sess * 2 : 16;
        qw_sess **p = realloc(st->sess, (size_t)cap * sizeof *p);
        if (!p) return false;
        st->sess = p;
        st->cap_sess = cap;
    }
    st->sess[st->n_sess++] = s;
    return true;
}

/* The tools as the template wants them: one string per tool. */
static char **tools_split(const qj_doc *d, const qj_node *arr, int32_t *n) {
    *n = 0;
    const int32_t count = (int32_t)qj_count(arr);
    char **v = calloc((size_t)(count ? count : 1), sizeof *v);
    if (!v) return NULL;
    for (const qj_node *t = qj_first(d, arr); t; t = qj_next(d, t)) {
        str one = { 0 };
        str_node(&one, d, t);
        v[(*n)++] = one.p;
    }
    return v;
}

static void tools_free(char **v, int32_t n) {
    for (int32_t i = 0; i < n; i++) free(v[i]);
    free(v);
}

/* Renders the prefix -- the system turn alone, no generation prompt -- which
 * is a token prefix of every first turn (Crucible's PrefixSuite pins this
 * property; the first step checks it again before relying on it). */
static int32_t *render_prefix(qw_store *st, qw_sess *s, int32_t *n, char *err, size_t errcap) {
    char **tools = NULL;
    int32_t n_tools = 0;
    if (s->has_tools_doc) tools = tools_split(&s->tools_doc, qj_get(&s->tools_doc, qj_root(&s->tools_doc), "tools"), &n_tools);
    qwasar_message m = { .role = "system", .content = s->system ? s->system : "" };
    qwasar_chat_options o = {
        .enable_thinking = s->thinking, .reasoning_effort = s->effort,
        .add_generation_prompt = false,
        .tools = n_tools ? (const char *const *)tools : NULL, .n_tools = n_tools,
    };
    int32_t *p = qwasar_apply_chat_template(st->tok, &m, 1, &o, n, err, errcap);
    tools_free(tools, n_tools);
    return p;
}

qw_sess *qw_store_open_session(qw_store *st, const char *system,
                               const char *const *tools, int32_t n_tools,
                               bool thinking, const char *effort,
                               const char *metadata, char *err, size_t errcap) {
    if (!effort || (strcmp(effort, "low") && strcmp(effort, "medium") && strcmp(effort, "xhigh"))) {
        snprintf(err, errcap, "effort must be low, medium or xhigh");
        return NULL;
    }
    qw_sess *s = sess_new();
    if (!s) { snprintf(err, errcap, "out of memory"); return NULL; }
    new_id(s->id, sizeof s->id);
    s->system = xstrdup(system ? system : "");
    snprintf(s->effort, sizeof s->effort, "%s", effort);
    s->thinking = thinking;
    s->metadata = xstrdup(metadata && *metadata ? metadata : "{}");
    snprintf(s->model_id, sizeof s->model_id, "%s", st->model_id);
    s->created = (int64_t)time(NULL);
    {
        str arr = { 0 };
        str_puts(&arr, "[");
        for (int32_t i = 0; i < n_tools; i++) {
            if (i) str_puts(&arr, ", ");
            str_puts(&arr, tools[i]);
        }
        str_puts(&arr, "]");
        sess_set_tools(s, arr.p, n_tools);
        str_free(&arr);
        if (n_tools > 0 && !s->has_tools_doc) {
            snprintf(err, errcap, "tools are not valid JSON");
            sess_free(s);
            return NULL;
        }
    }
    int32_t n = 0;
    int32_t *p = render_prefix(st, s, &n, err, errcap);
    if (!p) { sess_free(s); return NULL; }
    free(p);
    s->prefix_n = n;
    if (n >= st->ctx) {
        snprintf(err, errcap, "the prefix is %d tokens and the context is %d", n, st->ctx);
        sess_free(s);
        return NULL;
    }
    if (!sess_persist(st, s, err, errcap)) { sess_free(s); return NULL; }

    pthread_mutex_lock(&st->lock);
    const bool ok = list_add(st, s);
    pthread_mutex_unlock(&st->lock);
    if (!ok) { snprintf(err, errcap, "out of memory"); sess_free(s); return NULL; }
    if (st->verbose) qw_log("  session %s opened: prefix %d tokens, %d tool%s, effort %s",
                            s->id, n, n_tools, n_tools == 1 ? "" : "s", s->effort);
    return s;
}

/* A session from its record and token log; NULL if either will not read. */
static qw_sess *sess_load(qw_store *st, const char *id) {
    char path[1400];
    snprintf(path, sizeof path, "%s/sessions/%s/record.json", st->dir, id);
    qj_doc d;
    if (!qj_parse_file(&d, path)) { qj_free(&d); return NULL; }
    const qj_node *root = qj_root(&d);
    qw_sess *s = sess_new();
    if (!s) { qj_free(&d); return NULL; }
    snprintf(s->id, sizeof s->id, "%s", id);
    s->created = qj_int_or(&d, root, "created", 0);
    qj_str_copy(&d, root, "model_id", s->model_id, sizeof s->model_id);
    s->system = qj_strdup(&d, qj_get(&d, root, "system"));
    if (!s->system) s->system = xstrdup("");
    s->thinking = qj_bool_or(&d, root, "thinking", true);
    if (!qj_str_copy(&d, root, "effort", s->effort, sizeof s->effort)) snprintf(s->effort, sizeof s->effort, "xhigh");
    {
        str t = { 0 };
        str_node(&t, &d, qj_get(&d, root, "tools"));
        sess_set_tools(s, t.p, (int32_t)qj_count(qj_get(&d, root, "tools")));
        str_free(&t);
        str m = { 0 };
        const qj_node *md = qj_get(&d, root, "metadata");
        if (md && md->type == QJ_OBJECT) str_node(&m, &d, md); else str_puts(&m, "{}");
        s->metadata = m.p;
    }
    s->prefix_n = (int32_t)qj_int_or(&d, root, "prefix_tokens", 0);
    s->step = (int)qj_int_or(&d, root, "step", 0);
    char state[24] = "";
    qj_str_copy(&d, root, "state", state, sizeof state);
    s->state = !strcmp(state, "awaiting_tools") ? QW_SESS_AWAITING
             : !strcmp(state, "full") ? QW_SESS_FULL : QW_SESS_IDLE;
    qj_str_copy(&d, root, "last_stop", s->last_stop, sizeof s->last_stop);
    s->last_at = qj_int_or(&d, root, "last_at", 0);
    for (const qj_node *p = qj_first(&d, qj_get(&d, root, "pending"));
         p && s->n_pending < QW_MAX_CALLS; p = qj_next(&d, p)) {
        qj_str_copy(&d, p, "id", s->pending[s->n_pending].id, sizeof s->pending[0].id);
        qj_str_copy(&d, p, "name", s->pending[s->n_pending].name, sizeof s->pending[0].name);
        s->n_pending++;
    }
    qj_free(&d);

    snprintf(path, sizeof path, "%s/sessions/%s/tokens.bin", st->dir, id);
    FILE *f = fopen(path, "rb");
    if (f) {
        fseek(f, 0, SEEK_END);
        long len = ftell(f);
        fseek(f, 0, SEEK_SET);
        if (len > 0 && tokens_reserve(s, (int32_t)(len / 4))) {
            s->n_tokens = (int32_t)fread(s->tokens, sizeof *s->tokens, (size_t)(len / 4), f);
        }
        fclose(f);
    }
    return s;
}

/* ---- the store --------------------------------------------------------------- */

static bool default_state_dir(char *out, size_t cap) {
    const char *env = getenv("QWASAR_STATE");
    if (env && *env) { snprintf(out, cap, "%s", env); return true; }
    const char *home = getenv("HOME");
    if (!home || !*home) return false;
    snprintf(out, cap, "%s/Library/Application Support/Qwasar", home);
    return true;
}

qw_store *qw_store_open(qwasar_engine *e, qwasar_tokenizer *tok, const char *model_path,
                        const qw_profile *p, const qw_store_opts *o, char *err, size_t errcap) {
    qw_store *st = calloc(1, sizeof *st);
    if (!st) { snprintf(err, errcap, "out of memory"); return NULL; }
    st->e = e;
    st->tok = tok;
    snprintf(st->model_path, sizeof st->model_path, "%s", model_path);
    snprintf(st->model_id, sizeof st->model_id, "%s", qwasar_model_id(e));
    st->prof = *p;
    st->ctx = o->ctx > 0 ? o->ctx : p->ctx;
    st->live_max = o->live > 0 ? o->live : (p->live > 0 ? p->live : 1);
    st->no_cache = o->no_cache;
    st->verbose = o->verbose;
    st->miss_at = -1;
    st->prefill_tps = !strcmp(p->family, "qwen4_exp") ? 400.0 : 32.0;
    st->restore_bps = 2e9;
    pthread_mutex_init(&st->lock, NULL);
    pthread_mutex_init(&st->qlock, NULL);
    pthread_cond_init(&st->qcond, NULL);

    if (o->state_dir && *o->state_dir) snprintf(st->dir, sizeof st->dir, "%s", o->state_dir);
    else if (!default_state_dir(st->dir, sizeof st->dir)) {
        snprintf(err, errcap, "no HOME and no --state-dir: nowhere to keep sessions");
        free(st);
        return NULL;
    }
    char sdir[1200];
    snprintf(sdir, sizeof sdir, "%s/sessions", st->dir);
    if (!mkdir_p(sdir)) {
        snprintf(err, errcap, "cannot create %s: %s", sdir, strerror(errno));
        free(st);
        return NULL;
    }

    st->compat = sess_new();
    if (!st->compat) { snprintf(err, errcap, "out of memory"); free(st); return NULL; }
    snprintf(st->compat->id, sizeof st->compat->id, "compat");
    st->compat->compat = true;
    snprintf(st->compat->model_id, sizeof st->compat->model_id, "%s", st->model_id);

    /* Every session on disk, offered cold or warm; nothing evaluated. */
    DIR *d = opendir(sdir);
    int loaded = 0;
    if (d) {
        struct dirent *ent;
        while ((ent = readdir(d))) {
            if (ent->d_name[0] != 's' || ent->d_name[1] != '_') continue;
            qw_sess *s = sess_load(st, ent->d_name);
            if (s && list_add(st, s)) loaded++;
            else sess_free(s);
        }
        closedir(d);
    }
    if (st->verbose || loaded)
        qw_log("%d session%s in %s", loaded, loaded == 1 ? "" : "s", st->dir);
    return st;
}

qwasar_engine    *qw_store_engine(const qw_store *st) { return st->e; }
qwasar_tokenizer *qw_store_tokenizer(const qw_store *st) { return st->tok; }
const qw_profile *qw_store_profile(const qw_store *st) { return &st->prof; }
const char       *qw_store_state_dir(const qw_store *st) { return st->dir; }
const char       *qw_store_model_path(const qw_store *st) { return st->model_path; }
void qw_store_seed(qw_store *st, uint64_t seed) {
    st->rng = seed ? seed : (uint64_t)qw_now() * 6364136223846793005ull + 1;
}
int32_t           qw_store_ctx(const qw_store *st) { return st->ctx; }
bool              qw_store_no_cache(const qw_store *st) { return st->no_cache; }
const char       *qw_sess_id(const qw_sess *s) { return s->id; }
qw_sess          *qw_store_compat(qw_store *st) { return st->compat; }

void qw_store_counts(qw_store *st, int *total, int *live, int *queued) {
    pthread_mutex_lock(&st->lock);
    int l = 0, q = 0;
    for (int i = 0; i < st->n_sess; i++) {
        if (st->sess[i]->h) l++;
        if (st->sess[i]->state == QW_SESS_QUEUED) q++;
    }
    if (st->compat->h) l++;
    *total = st->n_sess;
    *live = l;
    *queued = q;
    pthread_mutex_unlock(&st->lock);
}

qw_sess *qw_store_find(qw_store *st, const char *id) {
    qw_sess *found = NULL;
    pthread_mutex_lock(&st->lock);
    for (int i = 0; i < st->n_sess && !found; i++)
        if (!strcmp(st->sess[i]->id, id)) found = st->sess[i];
    pthread_mutex_unlock(&st->lock);
    return found;
}

int qw_store_list(qw_store *st, qw_sess **out, int cap) {
    pthread_mutex_lock(&st->lock);
    int n = 0;
    for (int i = st->n_sess - 1; i >= 0 && n < cap; i--) out[n++] = st->sess[i];
    /* Newest first: by creation time, which the list order approximates
     * except for sessions loaded from disk in directory order. */
    for (int i = 1; i < n; i++)
        for (int j = i; j > 0 && out[j]->created > out[j - 1]->created; j--) {
            qw_sess *t = out[j]; out[j] = out[j - 1]; out[j - 1] = t;
        }
    pthread_mutex_unlock(&st->lock);
    return n;
}

/* ---- the engine queue ---------------------------------------------------------
 *
 * A FIFO of waiters rather than a ticket, so a waiter that is cancelled can
 * leave from the middle.  Position is reported to the waiter as it changes. */

static int queue_position(const qw_store *st, const qw_sess *s) {
    int pos = 1;
    for (const waiter *w = st->head; w; w = w->next, pos++)
        if (w->s == s) return pos;
    return 0;
}

static void queue_remove(qw_store *st, const qw_sess *s) {
    waiter **pp = &st->head;
    while (*pp && (*pp)->s != s) pp = &(*pp)->next;
    if (!*pp) return;
    waiter *w = *pp;
    *pp = w->next;
    if (st->tail == w) {
        st->tail = NULL;
        for (waiter *q = st->head; q; q = q->next) st->tail = q;
    }
    free(w);
}

/* Emits an event of the step in flight: logged for reattach, sent to the
 * connection while it is still there.  A dead peer does not stop a step. */
static void sess_emit(qw_sess *s, const char *event, const char *json) {
    char id[32];
    pthread_mutex_lock(&s->ev_lock);
    snprintf(id, sizeof id, "%d.%d", s->step, ++s->seq);
    if (s->n_ev == s->cap_ev) {
        int cap = s->cap_ev ? s->cap_ev * 2 : 64;
        qw_event *p = realloc(s->ev, (size_t)cap * sizeof *p);
        if (p) { s->ev = p; s->cap_ev = cap; }
    }
    if (s->n_ev < s->cap_ev) {
        s->ev[s->n_ev].step = s->step;
        s->ev[s->n_ev].seq = s->seq;
        s->ev[s->n_ev].name = xstrdup(event);
        s->ev[s->n_ev].json = xstrdup(json);
        s->n_ev++;
    }
    const bool done = !strcmp(event, "done") || !strcmp(event, "error");
    if (done) s->step_open = false;
    pthread_cond_broadcast(&s->ev_cond);
    pthread_mutex_unlock(&s->ev_lock);
    if (s->emit && !s->peer_gone && !s->emit(s->emit_ud, id, event, json)) s->peer_gone = true;
}

static void sess_emitf(qw_sess *s, const char *event, const char *fmt, ...)
    __attribute__((format(printf, 3, 4)));
static void sess_emitf(qw_sess *s, const char *event, const char *fmt, ...) {
    str b = { 0 };
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    str_puts(&b, buf);
    sess_emit(s, event, b.p);
    str_free(&b);
}

/* Waits for the engine.  False if the session was cancelled while waiting. */
static bool engine_acquire(qw_store *st, qw_sess *s, bool report) {
    pthread_mutex_lock(&st->qlock);
    if (!st->busy && !st->head) {
        st->busy = true;
        st->holder = s;
        pthread_mutex_unlock(&st->qlock);
        return true;
    }
    waiter *w = calloc(1, sizeof *w);
    w->s = s;
    if (st->tail) st->tail->next = w; else st->head = w;
    st->tail = w;
    int last_pos = -1;
    for (;;) {
        if (atomic_load(&s->cancel)) {
            queue_remove(st, s);
            pthread_cond_broadcast(&st->qcond);
            pthread_mutex_unlock(&st->qlock);
            return false;
        }
        const int pos = queue_position(st, s);
        if (report && pos != last_pos) {
            pthread_mutex_unlock(&st->qlock);
            sess_emitf(s, "queued", "{\"position\": %d}", pos);
            pthread_mutex_lock(&st->qlock);
            last_pos = pos;
            continue;          /* the state may have moved while unlocked */
        }
        if (!st->busy && st->head && st->head->s == s) {
            queue_remove(st, s);
            st->busy = true;
            st->holder = s;
            pthread_mutex_unlock(&st->qlock);
            return true;
        }
        pthread_cond_wait(&st->qcond, &st->qlock);
    }
}

static void engine_release(qw_store *st) {
    pthread_mutex_lock(&st->qlock);
    st->busy = false;
    st->holder = NULL;
    pthread_cond_broadcast(&st->qcond);
    pthread_mutex_unlock(&st->qlock);
}

void qw_store_acquire(qw_store *st, qw_sess *s) {
    atomic_store(&s->cancel, false);
    engine_acquire(st, s, false);
}
void qw_store_release(qw_store *st, qw_sess *s) { (void)s; engine_release(st); }

/* ---- the live set ---------------------------------------------------------------
 *
 * Caller holds the engine: nothing else touches a handle meanwhile. */

/* A handle leaving the live set is kept, not freed, for the next session
 * to take.  A new handle's memory is committed on first write -- by the CPU
 * on a restore, and again by the GPU on the first step that reads it -- and
 * at a long context that is seconds, twice; a reset handle's pages are
 * already both.  One spare at most, and only ever made by a handle leaving
 * the live set and spent by the next one joining it, so live handles and
 * the spare together stay within live_max. */
static void handle_put(qw_store *st, qwasar_session *h) {
    if (!h) return;
    if (st->spare) qwasar_session_free(st->spare);
    st->spare = h;
}

static qwasar_session *handle_take(qw_store *st, char *err, size_t errcap) {
    qwasar_session *h = st->spare;
    st->spare = NULL;
    if (h) {
        char why[256];
        if (qwasar_session_reset(h, why, sizeof why)) return h;
        qw_log("  a kept handle did not reset (%s); allocating a new one", why);
        qwasar_session_free(h);
    }
    return qwasar_session_new(st->e, err, errcap);
}

/* Writes the session's checkpoint and gives up its handle.  Returns whether a
 * checkpoint now covers it (else it is cold, which for a session under the
 * store's floor of 256 tokens is cheap and expected). */
static bool park_locked(qw_store *st, qw_sess *s, const char *why) {
    if (!s->h) return false;
    bool saved = false;
    if (!st->no_cache && !s->compat) {
        /* A named session parks to its own file, whatever its length: the
         * state is the session's, and the shared cache's 6 GB budget would
         * evict it -- or, at a large context, could not hold it at all. */
        saved = s->own_n == qwasar_session_n_past(s->h) || sess_save_own(st, s, why);
    } else if (!st->no_cache) {
        /* The compat session's next prompt repeats the last one but not
         * necessarily the reply: its checkpoint is taken at the rewind point,
         * in the shared cache, since nothing owns an anonymous session. */
        qwasar_session_rewind_to_mark(s->h);
        if (qwasar_session_n_past(s->h) >= 256) {
            char err[256];
            const double t0 = qw_now();
            saved = qwasar_session_save(s->h, st->e, err, sizeof err);
            if (st->verbose)
                qw_log("  %s parked (%s): %d tokens %s in %.2fs%s%s", s->id, why,
                       qwasar_session_n_past(s->h), saved ? "checkpointed" : "not saved",
                       qw_now() - t0, saved ? "" : ": ", saved ? "" : err);
            if (saved) s->ckpt_n = qwasar_session_n_past(s->h);
        }
    }
    handle_put(st, s->h);
    s->h = NULL;
    return saved;
}

/* Makes room for `s` in the live set: while it is full, the least recently
 * used other session is parked. */
static void admit(qw_store *st, qw_sess *s) {
    if (s->h) return;
    for (;;) {
        pthread_mutex_lock(&st->lock);
        int live = st->compat->h && st->compat != s ? 1 : 0;
        qw_sess *lru = st->compat->h && st->compat != s ? st->compat : NULL;
        for (int i = 0; i < st->n_sess; i++) {
            qw_sess *o = st->sess[i];
            if (o == s || !o->h) continue;
            live++;
            if (!lru || o->last_used < lru->last_used) lru = o;
        }
        pthread_mutex_unlock(&st->lock);
        if (live < st->live_max || !lru) return;
        park_locked(st, lru, "evicted");
    }
}

/* ---- prefill progress ----------------------------------------------------------- */

static void progress_cb(void *ud, int32_t done, int32_t total) {
    qw_sess *s = ud;
    (void)total;
    if (s->prog_total < 2) return;
    sess_emitf(s, "prefill", "{\"done\": %d, \"total\": %d}", s->prog_base + done, s->prog_total);
}

static void progress_arm(qw_sess *s, int32_t base, int32_t total) {
    s->prog_base = base;
    s->prog_total = total;
}

/* Evaluates `tokens` on the live handle, progress rebased over the span. */
static const float *eval_span(qw_store *st, qw_sess *s, const int32_t *tokens, int32_t n,
                              const qwasar_image_input *images, int32_t n_images,
                              int32_t base, int32_t total, char *err, size_t errcap) {
    progress_arm(s, base, total);
    const double t0 = qw_now();
    const float *l = n_images > 0
        ? qwasar_session_eval_images(s->h, tokens, n, images, n_images, err, errcap)
        : qwasar_session_eval(s->h, tokens, n, err, errcap);
    const double dt = qw_now() - t0;
    if (l && n >= 64 && dt > 0) {
        /* A running estimate of what prefill costs here, for resume estimates. */
        const double tps = n / dt;
        st->prefill_tps = st->prefill_tps > 0 ? 0.7 * st->prefill_tps + 0.3 * tps : tps;
    }
    return l;
}

/* ---- generation ------------------------------------------------------------------ */

/* True if `p` (n bytes) could still become "NAME>\n" for one of the tools --
 * the newline included, because the tokenizer often spells ">\n" as one. */
static bool name_prefix_ok(const char *p, size_t n, const qw_genopts *go) {
    for (int i = 0; i < go->n_names; i++) {
        const size_t nl = strlen(go->names[i]);
        if (n <= nl + 2 && !memcmp(p, go->names[i], n < nl ? n : nl)
            && (n <= nl || p[nl] == '>')
            && (n <= nl + 1 || p[nl + 1] == '\n'))
            return true;
    }
    return false;
}

/* Bytes at the end of `p` that are the start of a UTF-8 sequence whose last
 * byte has not been generated yet. */
static size_t utf8_tail(const char *p, size_t n) {
    for (size_t k = 1; k <= 4 && k <= n; k++) {
        const unsigned char c = (unsigned char)p[n - k];
        if ((c & 0xC0) == 0x80) continue;
        const size_t need = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
        return need > k ? k : 0;
    }
    return 0;
}

/* Longest suffix of `p` that begins some stop sequence. */
static size_t stop_prefix_tail(const char *p, size_t n, const qw_genopts *go) {
    size_t best = 0;
    for (int s = 0; s < go->n_stops; s++) {
        const size_t sl = strlen(go->stops[s]);
        for (size_t k = sl - 1; k > best; k--)
            if (k <= n && !memcmp(p + n - k, go->stops[s], k)) { best = k; break; }
    }
    return best;
}

static long find_stop(const char *p, size_t n, size_t from, const qw_genopts *go, int *which) {
    long best = -1;
    for (int s = 0; s < go->n_stops; s++) {
        const size_t sl = strlen(go->stops[s]);
        for (size_t i = from; i + sl <= n && (best < 0 || (long)i < best); i++)
            if (!memcmp(p + i, go->stops[s], sl)) { best = (long)i; *which = s; break; }
    }
    return best;
}

typedef struct {
    str    shown;
    size_t sent;
} channel;

static void chan_send(channel *ch, bool reasoning, size_t upto, qw_delta_fn fn, void *ud) {
    if (upto <= ch->sent) return;
    if (fn) fn(ud, reasoning, ch->shown.p + ch->sent, upto - ch->sent);
    ch->sent = upto;
}

/* Starts the tool call the request insists on, by evaluating its opening as
 * though the model had written it. */
static const float *force_call(qw_store *st, qw_sess *s, bool after_think, const char *name,
                               int32_t call_open, qw_genres *out, char *err, size_t cap) {
    str tail = { 0 };
    str_puts(&tail, "\n<function=");
    if (name) { str_puts(&tail, name); str_puts(&tail, ">\n"); }

    int32_t n_lead = 0, n_tail = 0;
    int32_t *lead = after_think ? qwasar_encode(st->tok, "\n\n", &n_lead) : NULL;
    int32_t *tl = qwasar_encode(st->tok, tail.p, &n_tail);
    int32_t *ids = malloc(sizeof *ids * (size_t)(n_lead + 1 + n_tail));
    const float *logits = NULL;
    if (ids && tl) {
        int32_t n = 0;
        for (int32_t i = 0; i < n_lead; i++) ids[n++] = lead[i];
        ids[n++] = call_open;
        for (int32_t i = 0; i < n_tail; i++) ids[n++] = tl[i];
        str_puts(&out->text, "<tool_call>");
        str_puts(&out->text, tail.p);
        out->n_gen += n;
        logits = qwasar_session_eval(s->h, ids, n, err, cap);
        if (logits && !s->compat) tokens_append(s, ids, n);
    } else {
        snprintf(err, cap, "out of memory");
    }
    free(ids); free(lead); free(tl); str_free(&tail);
    return logits;
}

void qw_genres_free(qw_genres *g) { str_free(&g->text); str_free(&g->reasoning); }

bool qw_generate(qw_store *st, qw_sess *s, const float *logits, const qwasar_sampling *sp,
                 int32_t max_tokens, bool thinking, const qw_genopts *go,
                 qw_delta_fn on_delta, qw_token_fn on_token, void *ud, atomic_bool *cancel,
                 qw_genres *out, char *err, size_t cap) {
    memset(out, 0, sizeof *out);
    const int32_t vocab = qwasar_vocab_size(st->e);
    const int32_t think_close = qwasar_token_id(st->tok, "</think>");
    const int32_t call_open = qwasar_token_id(st->tok, "<tool_call>");
    bool reasoning = thinking;
    bool in_call = false, forced = false, ok = true;
    bool naming = false;
    str name = { 0 };
    channel rc = { 0 }, tc = { 0 };

    float *masked = NULL;
    if ((go->tools == QW_TOOLS_NONE && call_open >= 0)
        || (go->tools == QW_TOOLS_FORCE && !go->force_name)) {
        masked = malloc(sizeof *masked * (size_t)vocab);
        if (!masked) { snprintf(err, cap, "out of memory"); return false; }
    }

    for (int32_t i = 0; i < max_tokens; i++) {
        if (go->tools == QW_TOOLS_FORCE && !reasoning && !forced) {
            forced = in_call = true;
            logits = force_call(st, s, thinking, go->force_name, call_open, out, err, cap);
            if (!logits) { ok = false; break; }
            naming = !go->force_name;
        }
        const float *lp = logits;
        if (go->tools == QW_TOOLS_NONE && masked) {
            memcpy(masked, logits, sizeof *masked * (size_t)vocab);
            masked[call_open] = -1e30f;
            lp = masked;
        } else if (naming) {
            memcpy(masked, logits, sizeof *masked * (size_t)vocab);
            char buf[256];
            memcpy(buf, name.p ? name.p : "", name.len);
            for (int32_t t = 0; t < vocab; t++) {
                size_t tl = 0;
                bool sp_tok = false;
                const char *tb = qwasar_token_bytes(st->tok, t, &tl, &sp_tok);
                if (sp_tok || !tb || !tl || name.len + tl > sizeof buf) { masked[t] = -1e30f; continue; }
                memcpy(buf + name.len, tb, tl);
                if (!name_prefix_ok(buf, name.len + tl, go)) masked[t] = -1e30f;
            }
            lp = masked;
        }
        /* Shutting down, or cancelled: the reply ends here, as if cut by its
         * budget, so the engine is free. */
        if (atomic_load(&g_stopping)) break;
        if (cancel && atomic_load(cancel)) { out->cancelled = true; break; }
        int32_t next = qwasar_sample(lp, vocab, sp, &st->rng);
        if (i == 0) {
            /* What the first token was drawn from, for the log: a token the
             * filters (top-k, top-p) should never have let through shows up
             * as a probability no sampler could have produced. */
            float mx = lp[0];
            int32_t top = 0;
            for (int32_t v = 1; v < vocab; v++) if (lp[v] > mx) { mx = lp[v]; top = v; }
            double z = 0.0;
            for (int32_t v = 0; v < vocab; v++) z += exp((double)(lp[v] - mx));
            out->first_token = next;
            out->first_p = (float)(exp((double)(lp[next] - mx)) / z);
            out->first_top = top;
            out->first_top_p = (float)(1.0 / z);
        }
        if (qwasar_is_eos(st->e, next)) { out->hit_eos = true; break; }
        out->n_gen++;
        if (reasoning) out->n_reasoning++;

        size_t len = 0;
        bool special = false;
        const char *bytes = qwasar_token_bytes(st->tok, next, &len, &special);

        if (naming && bytes && len) {
            str_add(&name, bytes, len);
            if (memchr(name.p, '>', name.len)) naming = false;
        }

        if (next == think_close) {
            reasoning = false;
            chan_send(&rc, true, rc.shown.len, on_delta, ud);
        } else if (bytes && len) {
            /* A tool call opened while still reasoning: the model skipped its
             * </think> and went straight to the call.  <tool_call> is not
             * something reasoning writes, so it ends the reasoning here --
             * what came before stays reasoning, the call is parsed and run.
             * (Seen in a long session: the model's whole reasoning became
             * one stray word, copied step after step without its </think>,
             * and every call it wrote after that was lost.) */
            if (reasoning && next == call_open && go->tools != QW_TOOLS_NONE) {
                reasoning = false;
                chan_send(&rc, true, rc.shown.len, on_delta, ud);
                out->call_in_reasoning = true;
            }
            str_add(reasoning ? &out->reasoning : &out->text, bytes, len);
            if (next == call_open) {
                in_call = true;
                chan_send(&tc, false, tc.shown.len, on_delta, ud);
            }
            if (special) {
                /* structure, not content */
            } else if (reasoning) {
                str_add(&rc.shown, bytes, len);
                chan_send(&rc, true, rc.shown.len - utf8_tail(rc.shown.p, rc.shown.len),
                          on_delta, ud);
            } else if (!in_call) {
                str_add(&tc.shown, bytes, len);
                const long at = find_stop(tc.shown.p, tc.shown.len, tc.sent, go,
                                          &out->stop_index);
                if (at >= 0) {
                    chan_send(&tc, false, (size_t)at, on_delta, ud);
                    out->text.len = 0;
                    str_add(&out->text, tc.shown.p, (size_t)at);
                    out->hit_stop = true;
                    break;
                }
                size_t hold = stop_prefix_tail(tc.shown.p, tc.shown.len, go);
                const size_t u = utf8_tail(tc.shown.p, tc.shown.len);
                if (u > hold) hold = u;
                chan_send(&tc, false, tc.shown.len - hold, on_delta, ud);
            }
        }

        if (!reasoning && next != call_open && go->tools != QW_TOOLS_NONE
            && qw_tool_call_complete(out->text.p ? out->text.p : "", out->text.len)) {
            out->has_call = true;
            /* The token that completed the call -- </tool_call> -- is part of
             * the turn as the chat template writes it ("</function>\n
             * </tool_call><|im_end|>"), so it is evaluated and recorded like
             * any other before the step ends.  It used not to be: the step
             * stopped on it, the next render closed the turn, and every call
             * in a session's context lacked its closing tag -- a shape the
             * model never saw in training, repeated dozens of times a session. */
            logits = qwasar_session_eval(s->h, &next, 1, err, cap);
            if (!logits) { ok = false; break; }
            if (!s->compat && !s->aside) tokens_append(s, &next, 1);
            if (on_token) on_token(ud, out->n_gen, in_call, out->text.p ? out->text.p : "", out->text.len);
            break;
        }

        logits = qwasar_session_eval(s->h, &next, 1, err, cap);
        if (!logits) { ok = false; break; }
        if (!s->compat && !s->aside) tokens_append(s, &next, 1);
        if (on_token) on_token(ud, out->n_gen, in_call, out->text.p ? out->text.p : "", out->text.len);
    }

    if (ok) {
        chan_send(&rc, true, rc.shown.len - utf8_tail(rc.shown.p, rc.shown.len), on_delta, ud);
        if (!in_call && !out->hit_stop)
            chan_send(&tc, false, tc.shown.len - utf8_tail(tc.shown.p, tc.shown.len),
                      on_delta, ud);
        out->text.len -= utf8_tail(out->text.p, out->text.len);
        out->reasoning.len -= utf8_tail(out->reasoning.p, out->reasoning.len);
        if (out->text.p) out->text.p[out->text.len] = 0;
        if (out->reasoning.p) out->reasoning.p[out->reasoning.len] = 0;
    }
    free(masked);
    str_free(&name);
    str_free(&rc.shown);
    str_free(&tc.shown);
    return ok;
}

/* ---- tool arguments as JSON ------------------------------------------------------ */

/* Whether `tool` declares parameter `param` a string -- "type": "string", or
 * a type list of only "string" and "null".  Tools come in OpenAI's shape
 * (function.parameters) or Anthropic's (input_schema). */
static bool param_is_string(const qj_doc *d, const char *tool, const char *param) {
    if (!d) return false;
    const qj_node *tools = qj_get(d, qj_root(d), "tools");
    for (const qj_node *t = qj_first(d, tools); t; t = qj_next(d, t)) {
        const qj_node *fn = qj_get(d, t, "function");
        if (!fn) fn = t;
        if (!qj_str_eq(d, qj_get(d, fn, "name"), tool)) continue;
        const qj_node *schema = qj_get(d, fn, "parameters");
        if (!schema) schema = qj_get(d, t, "input_schema");
        const qj_node *props = qj_get(d, schema, "properties");
        const size_t pl = strlen(param);
        for (const qj_node *m = qj_first(d, props); m; m = qj_next(d, m)) {
            if (m->key_len != pl || memcmp(d->text + m->key_off, param, pl)) continue;
            const qj_node *type = qj_get(d, m, "type");
            if (qj_str_eq(d, type, "string")) return true;
            if (type && type->type == QJ_ARRAY) {
                bool is_str = false, other = false;
                for (const qj_node *e = qj_first(d, type); e; e = qj_next(d, e)) {
                    if (qj_str_eq(d, e, "string")) is_str = true;
                    else if (!qj_str_eq(d, e, "null")) other = true;
                }
                return is_str && !other;
            }
            return false;
        }
        return false;
    }
    return false;
}

/* Strict JSON number syntax: "007" or "1." are text, however a lenient
 * parser reads them. */
static bool json_number_syntax(const char *v) {
    const char *p = v;
    if (*p == '-') p++;
    if (*p == '0') p++;
    else if (*p >= '1' && *p <= '9') while (*p >= '0' && *p <= '9') p++;
    else return false;
    if (*p == '.') { p++; if (!(*p >= '0' && *p <= '9')) return false; while (*p >= '0' && *p <= '9') p++; }
    if (*p == 'e' || *p == 'E') {
        p++;
        if (*p == '+' || *p == '-') p++;
        if (!(*p >= '0' && *p <= '9')) return false;
        while (*p >= '0' && *p <= '9') p++;
    }
    return *p == 0;
}

/* A parameter its tool declares a string is a string, whatever it looks
 * like.  Otherwise a value that is, whole, a JSON number, boolean, object or
 * array goes out as JSON; anything else is a string. */
static void emit_arg_value(str *out, const char *v, bool is_string) {
    if (!is_string && v && *v) {
        qj_doc probe;
        if (qj_parse(&probe, v, strlen(v))) {
            const qj_node *r = qj_root(&probe);
            const bool json = (r->type == QJ_NUMBER && json_number_syntax(v))
                           || r->type == QJ_TRUE || r->type == QJ_FALSE
                           || r->type == QJ_OBJECT || r->type == QJ_ARRAY;
            qj_free(&probe);
            if (json) { str_puts(out, v); return; }
        } else {
            qj_free(&probe);
        }
    }
    str_jsons(out, v ? v : "");
}

void qw_args_json(str *out, const qw_tool_call *c, const qj_doc *tools) {
    str_puts(out, "{");
    for (int i = 0; i < c->n_params; i++) {
        if (i) str_puts(out, ", ");
        str_jsons(out, c->params[i].key);
        str_puts(out, ": ");
        emit_arg_value(out, c->params[i].value, param_is_string(tools, c->name, c->params[i].key));
    }
    str_puts(out, "}");
}

/* ---- steps ------------------------------------------------------------------------ */

/* What one step carries while it runs, for the callbacks. */
typedef struct {
    qw_store *st;
    qw_sess  *s;
    double    t_decode;
    double    t_first;
    int32_t   n_gen;            /* tokens taken so far, as on_token reports them */
    int32_t   reasoning_at;     /* n_gen when the last reasoning delta went out */
    int32_t   last_report;
    int32_t   last_rate_n;
    double    last_rate_t;
} step_ctx;

static void step_delta(void *ud, bool reasoning, const char *p, size_t n) {
    step_ctx *x = ud;
    if (x->t_first == 0) x->t_first = qw_now();
    str b = { 0 };
    str_puts(&b, "{\"text\": ");
    str_json(&b, p, n);
    if (reasoning) {
        /* A delta arrives before its token is counted, so the token that
         * completed it is the one after the last count. */
        const int32_t upto = x->n_gen + 1;
        str_printf(&b, ", \"tokens\": %d}", upto - x->reasoning_at);
        x->reasoning_at = upto;
    } else {
        str_puts(&b, "}");
    }
    sess_emit(x->s, reasoning ? "reasoning" : "text", b.p);
    str_free(&b);
}

/* The name and parameter keys of the call being written, from the text
 * after its opening tag, so a client can show that the step is alive. */
static void call_progress_json(str *b, const char *text, size_t len) {
    const char *open = NULL;
    for (const char *p = text; (p = strstr(p, "<tool_call>")); p += 11) open = p;
    (void)len;
    str_puts(b, "{\"name\": ");
    const char *fn = open ? strstr(open, "<function=") : NULL;
    if (fn) {
        const char *e = strchr(fn + 10, '>');
        if (e) str_json(b, fn + 10, (size_t)(e - fn - 10)); else str_puts(b, "null");
    } else {
        str_puts(b, "null");
    }
    str_puts(b, ", \"keys\": [");
    bool first = true;
    for (const char *p = fn ? fn : text; (p = strstr(p, "<parameter=")); p += 11) {
        const char *e = strchr(p + 11, '>');
        if (!e) break;
        if (!first) str_puts(b, ", ");
        str_json(b, p + 11, (size_t)(e - p - 11));
        first = false;
    }
    str_puts(b, "]");
}

static void step_token(void *ud, int32_t n_gen, bool in_call, const char *text, size_t len) {
    step_ctx *x = ud;
    if (x->t_first == 0) x->t_first = qw_now();
    x->n_gen = n_gen;
    if (n_gen - x->last_report < 8) return;
    x->last_report = n_gen;
    const double now = qw_now();
    const double elapsed = now - x->t_decode;
    const double dt = now - x->last_rate_t;
    const double inst = dt > 0 ? (n_gen - x->last_rate_n) / dt : 0;
    x->last_rate_n = n_gen;
    x->last_rate_t = now;
    sess_emitf(x->s, "context", "{\"used\": %d, \"limit\": %d}",
               qwasar_session_n_past(x->s->h), x->st->ctx);
    sess_emitf(x->s, "decode", "{\"generated\": %d, \"tokens_per_second\": %.2f, \"instantaneous\": %.2f}",
               n_gen, elapsed > 0 ? n_gen / elapsed : 0.0, inst);
    if (in_call) {
        str b = { 0 };
        call_progress_json(&b, text, len);
        str_printf(&b, ", \"tokens\": %d}", n_gen);
        sess_emit(x->s, "call_progress", b.p);
        str_free(&b);
    }
}

const char *qw_warmth_name(qw_warmth w) {
    return w == QW_WARMTH_LIVE ? "live" : w == QW_WARMTH_WARM ? "warm" : "cold";
}

static void end_step(qw_store *st, qw_sess *s, const char *stop, bool persist) {
    (void)st;
    snprintf(s->last_stop, sizeof s->last_stop, "%s", stop);
    s->last_at = (int64_t)time(NULL);
    if (persist) {
        char err[256];
        if (!sess_persist(st, s, err, sizeof err)) qw_log("  %s: %s", s->id, err);
    }
    pthread_mutex_lock(&s->ev_lock);
    s->step_open = false;
    /* The step's connection is gone when its handler returns: nothing may
     * emit to it after this.  (An aside's prefill once did, through the
     * progress callback, and wrote into a dead stack frame.) */
    s->emit = NULL;
    s->emit_ud = NULL;
    s->prog_total = 0;
    pthread_cond_broadcast(&s->ev_cond);
    pthread_mutex_unlock(&s->ev_lock);
}

static void emit_done(qw_store *st, qw_sess *s, const char *stop, int32_t prompt,
                      const qw_genres *g, double prefill_s, double decode_s, double first_s) {
    str b = { 0 };
    str_printf(&b, "{\"stop\": \"%s\", \"usage\": {\"prompt\": %d, \"generated\": %d, \"reasoning\": %d}, "
                   "\"timing\": {\"prefill_seconds\": %.3f, \"decode_seconds\": %.3f, \"first_token_seconds\": %.3f}, "
                   "\"speculation\": {\"rounds\": 0, \"committed\": 0}, "
                   "\"context\": {\"used\": %d, \"limit\": %d}, "
                   "\"warmth\": {\"state\": \"%s\", \"covered\": %d}}",
               stop, prompt, g ? g->n_gen : 0, g ? g->n_reasoning : 0,
               prefill_s, decode_s, first_s,
               s->h ? qwasar_session_n_past(s->h) : s->n_tokens, st->ctx,
               s->h ? "live" : "cold", s->n_tokens);
    sess_emit(s, "done", b.p);
    str_free(&b);
}

/* Refuses a step that cannot start, with the HTTP status that fits. */
static bool step_admissible(qw_store *st, qw_sess *s, bool is_continue,
                            int *status, char *err, size_t errcap) {
    if (strcmp(s->model_id, st->model_id)) {
        *status = 409;
        snprintf(err, errcap, "this session ran on %s and the server has %s loaded; "
                 "open a new session for it", s->model_id, st->model_id);
        return false;
    }
    switch (s->state) {
    case QW_SESS_FULL:
        *status = 409;
        snprintf(err, errcap, "the session's window is full; open a new session");
        return false;
    case QW_SESS_QUEUED: case QW_SESS_RUNNING:
        *status = 409;
        snprintf(err, errcap, "a step is already in flight on this session");
        return false;
    case QW_SESS_AWAITING:
        return true;
    case QW_SESS_IDLE:
        if (is_continue) {
            *status = 409;
            snprintf(err, errcap, "no tool calls are pending on this session");
            return false;
        }
        return true;
    }
    return true;
}

/* Begins a step: clears events older than the last step, opens the log. */
static void step_begin(qw_sess *s, qw_emit_fn emit, void *ud) {
    pthread_mutex_lock(&s->ev_lock);
    s->step++;
    events_clear(s, s->step - 1);
    s->seq = 0;
    s->step_open = true;
    pthread_mutex_unlock(&s->ev_lock);
    s->emit = emit;
    s->emit_ud = ud;
    s->peer_gone = false;
    atomic_store(&s->cancel, false);
}

/* Brings the session's handle to the end of its timeline, emitting `resume`
 * over the whole of what remains to evaluate (its own tail plus `n_new`).
 * `prefix` is the rendered prefix on a first step, for the shared store. */
static bool resume(qw_store *st, qw_sess *s, int32_t n_new,
                   const int32_t *prefix, int32_t prefix_n,
                   double *prefill_s, int32_t *evaluated, char *err, size_t errcap) {
    char e2[256];
    *evaluated = 0;
    if (s->h) {
        sess_emitf(s, "resume", "{\"from\": \"live\", \"restored\": %d, \"prefill\": %d}",
                   s->n_tokens, n_new);
        return true;
    }
    admit(st, s);
    s->h = handle_take(st, err, errcap);
    if (!s->h) return false;
    qwasar_session_set_progress(s->h, progress_cb, s);
    s->ckpt_n = 0;

    const int32_t *seq = s->n_tokens ? s->tokens : prefix;
    const int32_t seq_n = s->n_tokens ? s->n_tokens : prefix_n;
    int32_t covered = 0;
    const double t0 = qw_now();
    if (!st->no_cache && seq_n > 0) {
        /* Its own checkpoint, or the shared cache (which holds the prefix
         * every session of it shares) -- whichever covers more.  Both are
         * probed before either is read: a fresh handle takes one restore. */
        char own[1400];
        sess_ckpt_path(st, s, own, sizeof own);
        const int32_t by_file = qwasar_kv_probe_file(st->e, own, seq, seq_n);
        const int32_t by_cache = qwasar_kv_probe(st->e, seq, seq_n);
        if (by_file >= by_cache && by_file > 0) {
            covered = qwasar_session_restore_file(s->h, st->e, own, seq, seq_n);
            s->own_n = covered;
        } else if (by_cache > 0) {
            covered = qwasar_session_restore(s->h, st->e, seq, seq_n);
            s->own_n = by_file;
        }
    }
    const double dt = qw_now() - t0;
    if (covered > 0 && dt > 0) {
        const double bytes = 150e6 + (double)covered * (double)st->prof.kv_per_token;
        st->restore_bps = 0.7 * st->restore_bps + 0.3 * (bytes / dt);
        s->ckpt_n = covered;
    }
    const int32_t rest = seq_n - covered;
    sess_emitf(s, "resume", "{\"from\": \"%s\", \"restored\": %d, \"prefill\": %d%s}",
               covered > 0 ? "checkpoint" : "cold", covered, rest + n_new,
               s->n_tokens == 0 ? (covered >= prefix_n ? ", \"prefix_cached\": true"
                                                       : ", \"prefix_cached\": false") : "");
    if (st->verbose && covered > 0)
        qw_log("  %s: restored %d tokens from a checkpoint in %.2fs", s->id, covered, dt);

    if (rest > 0) {
        const double t1 = qw_now();
        if (!eval_span(st, s, seq + covered, rest, NULL, 0, 0, rest + n_new, err, errcap))
            return false;
        *prefill_s += qw_now() - t1;
        *evaluated = rest;
        /* The shared prefix, the first time a session evaluates it: every
         * later session of this prefix starts from it. */
        if (s->n_tokens == 0 && !st->no_cache && covered < prefix_n && prefix_n >= 256) {
            const double t2 = qw_now();
            if (qwasar_session_save(s->h, st->e, e2, sizeof e2)) {
                s->ckpt_n = prefix_n;
                if (st->verbose) qw_log("  %s: prefix checkpoint (%d tokens) written in %.2fs",
                                        s->id, prefix_n, qw_now() - t2);
            } else if (st->verbose) {
                qw_log("  %s: prefix not checkpointed: %s", s->id, e2);
            }
        }
    }
    if (s->n_tokens == 0) tokens_append(s, prefix, prefix_n);
    return true;
}

/* The common body of turn and continue: `fresh` is what this step adds, with
 * its images; for a first step `full` is the whole first turn and `fresh`
 * the part after the prefix. */
static bool run_step(qw_store *st, qw_sess *s, int32_t *fresh, int32_t n_fresh,
                     int32_t *prefix, int32_t prefix_n,
                     const qwasar_image_input *images, int32_t n_images,
                     const qwasar_sampling *sp, int32_t max_tokens,
                     qw_emit_fn emit, void *ud) {
    char err[512] = "";
    step_ctx x = { .st = st, .s = s };
    double prefill_s = 0;

    pthread_mutex_lock(&st->lock);
    s->state = QW_SESS_QUEUED;
    pthread_mutex_unlock(&st->lock);
    step_begin(s, emit, ud);

    if (!engine_acquire(st, s, true)) {
        pthread_mutex_lock(&st->lock);
        s->state = s->n_pending ? QW_SESS_AWAITING : QW_SESS_IDLE;
        pthread_mutex_unlock(&st->lock);
        emit_done(st, s, "cancelled", 0, NULL, 0, 0, 0);
        end_step(st, s, "cancelled", false);
        free(fresh); free(prefix);
        return true;
    }
    pthread_mutex_lock(&st->lock);
    s->state = QW_SESS_RUNNING;
    s->last_used = qw_now();
    pthread_mutex_unlock(&st->lock);

    /* Room: the step's new material must fit, with at least one token of
     * generation after it. */
    const int32_t base = s->n_tokens + prefix_n;
    if (base + n_fresh + 1 >= st->ctx) {
        pthread_mutex_lock(&st->lock);
        s->state = QW_SESS_FULL;
        pthread_mutex_unlock(&st->lock);
        sess_emitf(s, "context", "{\"used\": %d, \"limit\": %d}", base, st->ctx);
        emit_done(st, s, "context_full", 0, NULL, 0, 0, 0);
        end_step(st, s, "context_full", true);
        engine_release(st);
        free(fresh); free(prefix);
        return true;
    }

    st->rng = sp->seed ? sp->seed : (uint64_t)qw_now() * 6364136223846793005ull + 1;

    int32_t resumed = 0;
    bool ok = resume(st, s, n_fresh, prefix, prefix_n, &prefill_s, &resumed, err, sizeof err);
    const float *logits = NULL;
    if (ok) {
        const double t0 = qw_now();
        /* The resume's own span reported its part; this is the rest of it. */
        logits = eval_span(st, s, fresh, n_fresh, images, n_images,
                           resumed, resumed + n_fresh, err, sizeof err);
        prefill_s += qw_now() - t0;
        ok = logits != NULL;
        if (ok) tokens_append(s, fresh, n_fresh);
    }
    free(prefix);
    free(fresh);
    if (!ok) {
        str b = { 0 };
        str_puts(&b, "{\"message\": ");
        str_jsons(&b, err);
        str_puts(&b, "}");
        sess_emit(s, "error", b.p);
        str_free(&b);
        qw_log("  %s: prefill failed: %s", s->id, err);
        /* The handle may be part-way through something; drop it.  The
         * timeline still says what was evaluated up to the step. */
        if (s->h) { qwasar_session_free(s->h); s->h = NULL; }
        pthread_mutex_lock(&st->lock);
        s->state = s->n_pending ? QW_SESS_AWAITING : QW_SESS_IDLE;
        pthread_mutex_unlock(&st->lock);
        end_step(st, s, "error", true);
        engine_release(st);
        return true;
    }
    sess_emitf(s, "context", "{\"used\": %d, \"limit\": %d}", qwasar_session_n_past(s->h), st->ctx);

    /* Output fits in what the window has left. */
    const int32_t room = st->ctx - qwasar_session_n_past(s->h) - 1;
    if (max_tokens <= 0 || max_tokens > room) max_tokens = room;

    qw_genopts go;
    memset(&go, 0, sizeof go);
    go.tools = s->n_tools > 0 ? QW_TOOLS_AUTO : QW_TOOLS_NONE;
    qw_genres g;
    x.t_decode = qw_now();
    x.last_rate_t = x.t_decode;
    ok = qw_generate(st, s, logits, sp, max_tokens, s->thinking, &go,
                     step_delta, step_token, &x, &s->cancel, &g, err, sizeof err);
    const double decode_s = qw_now() - x.t_decode;
    const double first_s = x.t_first ? x.t_first - x.t_decode + prefill_s : prefill_s;

    const char *stop = "end_turn";
    s->n_pending = 0;
    if (!ok) {
        str b = { 0 };
        str_puts(&b, "{\"message\": ");
        str_jsons(&b, err);
        str_puts(&b, "}");
        sess_emit(s, "error", b.p);
        str_free(&b);
        qw_log("  %s: generation failed: %s", s->id, err);
        stop = "error";
    } else {
        qw_tool_calls calls;
        memset(&calls, 0, sizeof calls);
        int n_calls = 0;
        if (g.has_call) {
            char perr[256];
            n_calls = qw_tool_parse(g.text.p ? g.text.p : "", &calls, perr, sizeof perr);
            if (n_calls < 0) n_calls = 0;
        }
        for (int i = 0; i < n_calls && i < QW_MAX_CALLS; i++) {
            snprintf(s->pending[i].id, sizeof s->pending[i].id, "c_%d_%d", s->step, i + 1);
            snprintf(s->pending[i].name, sizeof s->pending[i].name, "%s", calls.calls[i].name);
            s->n_pending = i + 1;
            str b = { 0 };
            str_printf(&b, "{\"id\": \"%s\", \"name\": ", s->pending[i].id);
            str_jsons(&b, calls.calls[i].name);
            str_puts(&b, ", \"arguments\": ");
            qw_args_json(&b, &calls.calls[i], s->has_tools_doc ? &s->tools_doc : NULL);
            str_puts(&b, "}");
            sess_emit(s, "tool_call", b.p);
            str_free(&b);
        }
        qw_tool_calls_free(&calls);
        if (n_calls > 0) stop = "tool_calls";
        else if (g.cancelled) stop = "cancelled";
        else if (atomic_load(&g_stopping)) stop = "shutdown";
        else if (g.hit_eos || g.hit_stop) stop = "end_turn";
        else if (qwasar_session_n_past(s->h) >= st->ctx - 1) stop = "context_full";
        else stop = "length";
    }

    /* A long conversation leaves a checkpoint in its own file as it grows,
     * so a crash costs a bounded re-prefill.  Spaced by a quarter of the
     * conversation (at least 4K tokens) rather than every 4K: the file is
     * rewritten whole, and at 200K tokens it is gigabytes -- a crash then
     * costs at most a fifth of the conversation, and the writes stay few. */
    if (ok && !st->no_cache) {
        const int32_t grown = s->n_tokens - s->own_n;
        const int32_t span = s->own_n / 4 > 4096 ? s->own_n / 4 : 4096;
        if (grown >= span) sess_save_own(st, s, "growth");
    }

    pthread_mutex_lock(&st->lock);
    s->state = !strcmp(stop, "tool_calls") ? QW_SESS_AWAITING
             : !strcmp(stop, "context_full") ? QW_SESS_FULL : QW_SESS_IDLE;
    s->last_used = qw_now();
    pthread_mutex_unlock(&st->lock);

    if (ok && g.n_gen > 0 && (st->verbose || g.first_p < 1e-3f)) {
        char a[48], b[48];
        token_text(st, &g.first_token, 0, 1, a, sizeof a);
        token_text(st, &g.first_top, 0, 1, b, sizeof b);
        qw_log("  %s: step %d: first token %d '%s' p=%.4f (most likely %d '%s' p=%.4f)%s", s->id, s->step,
               g.first_token, a, g.first_p, g.first_top, b, g.first_top_p,
               g.first_p < 1e-3f ? " -- UNLIKELY: no top-k/top-p draw gives this; the logits or the sampler are suspect" : "");
    }
    if (g.call_in_reasoning)
        qw_log("  %s: step %d: a tool call opened inside the reasoning block; read as its end", s->id, s->step);
    if (st->verbose)
        qw_log("  %s: step %d: %d tokens prefilled in %.2fs, %d generated in %.2fs (%.1f tok/s), %s",
               s->id, s->step, n_fresh, prefill_s, g.n_gen, decode_s,
               decode_s > 0 ? g.n_gen / decode_s : 0.0, stop);
    /* usage.prompt is what this step evaluated: the fresh tokens, and any of
     * the timeline a checkpoint did not cover -- but never a prefix that was
     * read rather than prefilled. */
    if (ok) emit_done(st, s, stop, resumed + n_fresh, &g, prefill_s, decode_s, first_s);
    qw_genres_free(&g);
    end_step(st, s, stop, true);
    engine_release(st);
    return true;
}

/* Ends an aside running on `s`, if any, and waits for it to roll back: a
 * real step never waits behind one. */
static void aside_preempt(qw_sess *s) {
    pthread_mutex_lock(&s->ev_lock);
    if (s->aside_running) {
        atomic_store(&s->aside_cancel, true);
        while (s->aside_running) pthread_cond_wait(&s->ev_cond, &s->ev_lock);
    }
    pthread_mutex_unlock(&s->ev_lock);
}

/* The engine if it is free right now; an aside never queues. */
static bool engine_try_acquire(qw_store *st, qw_sess *s) {
    pthread_mutex_lock(&st->qlock);
    const bool got = !st->busy && !st->head;
    if (got) { st->busy = true; st->holder = s; }
    pthread_mutex_unlock(&st->qlock);
    return got;
}

typedef struct { qw_emit_fn emit; void *ud; int seq; bool gone; } aside_sink;

static void aside_delta(void *ud, bool reasoning, const char *p, size_t n) {
    aside_sink *k = ud;
    if (reasoning || n == 0 || k->gone) return;
    str b = { 0 };
    str_puts(&b, "{\"text\": ");
    char *t = malloc(n + 1);
    if (t) { memcpy(t, p, n); t[n] = 0; str_jsons(&b, t); free(t); }
    str_puts(&b, "}");
    char id[32];
    snprintf(id, sizeof id, "aside.%d", ++k->seq);
    if (!k->emit(k->ud, id, "text", b.p)) k->gone = true;
    str_free(&b);
}

static void chat_opts(const qw_sess *s, qwasar_chat_options *o) {
    memset(o, 0, sizeof *o);
    o->enable_thinking = s->thinking;
    o->reasoning_effort = s->effort;
    o->add_generation_prompt = true;
    o->n_tools = s->n_tools;
}

bool qw_sess_aside(qw_store *st, qw_sess *s, const char *text, int32_t max_tokens,
                   const qwasar_sampling *sp, qw_emit_fn emit, void *ud,
                   int *status, char *err, size_t errcap) {
    *status = 409;
    pthread_mutex_lock(&st->lock);
    const bool idle = s->state == QW_SESS_IDLE && !strcmp(s->model_id, st->model_id) && !s->compat;
    pthread_mutex_unlock(&st->lock);
    if (!idle) { snprintf(err, errcap, "an aside needs an idle session"); return false; }

    pthread_mutex_lock(&s->ev_lock);
    const bool already = s->aside_running;
    if (!already) { s->aside_running = true; atomic_store(&s->aside_cancel, false); }
    pthread_mutex_unlock(&s->ev_lock);
    if (already) { snprintf(err, errcap, "an aside is already running"); return false; }

    bool ok = false;
    if (!engine_try_acquire(st, s)) {
        snprintf(err, errcap, "the engine is busy");
        goto done;
    }
    /* Live only: an aside costs no resume, and needs a rewind point, which
     * a session with images in it cannot take. */
    if (!s->h || !qwasar_session_mark(s->h)) {
        snprintf(err, errcap, s->h ? "this session cannot take a rewind point (it has images)"
                                   : "the session is not in memory");
        engine_release(st);
        goto done;
    }
    {
        const int32_t base = qwasar_session_n_past(s->h);
        qwasar_chat_options o;
        chat_opts(s, &o);
        o.enable_thinking = false;            /* notes, not deliberation */
        int32_t n = 0;
        int32_t *ids = qwasar_render_user_turn(st->tok, text ? text : "", 0, false, &o, &n);
        if (max_tokens <= 0) max_tokens = 1024;
        if (!ids || base + n + max_tokens + 1 >= st->ctx) {
            free(ids);
            snprintf(err, errcap, ids ? "not enough room left in the window for an aside"
                                      : "cannot render the aside");
            engine_release(st);
            goto done;
        }
        *status = 200;
        aside_sink k = { emit, ud, 0, false };
        /* Its prefill reports to no one: the progress callback emits into
         * the session's step log and connection, which are not the aside's. */
        s->prog_total = 0;
        const double t0 = qw_now();
        char e2[256] = "";
        const float *logits = qwasar_session_eval(s->h, ids, n, e2, sizeof e2);
        free(ids);
        qw_genres g;
        memset(&g, 0, sizeof g);
        bool gen = false;
        if (logits) {
            qw_genopts go;
            memset(&go, 0, sizeof go);
            go.tools = QW_TOOLS_NONE;
            s->aside = true;
            gen = qw_generate(st, s, logits, sp, max_tokens, false, &go, aside_delta, NULL, &k,
                              &s->aside_cancel, &g, e2, sizeof e2);
            s->aside = false;
        }
        /* Back to where the session was: the rewind point is exactly the
         * timeline's end, so nothing the aside did remains.  Should it fail
         * to land there, the handle goes and the session resumes from disk
         * -- never from a state with the aside in it. */
        if (qwasar_session_rewind_to_mark(s->h) != s->n_tokens || qwasar_session_n_past(s->h) != base) {
            qw_log("  %s: aside did not rewind cleanly; dropping the handle", s->id);
            qwasar_session_free(s->h);
            s->h = NULL;
        }
        const char *stop = !logits || !gen ? "error"
                         : g.cancelled ? "cancelled" : (g.hit_eos || g.hit_stop) ? "end_turn" : "length";
        str b = { 0 };
        str_printf(&b, "{\"stop\": \"%s\", \"usage\": {\"prompt\": %d, \"generated\": %d}, "
                       "\"seconds\": %.3f}", stop, n, g.n_gen, qw_now() - t0);
        if (!k.gone) emit(ud, "aside.done", "done", b.p);
        str_free(&b);
        if (st->verbose)
            qw_log("  %s: aside: %d tokens in, %d out in %.2fs, %s; rolled back to %d",
                   s->id, n, g.n_gen, qw_now() - t0, stop, base);
        qw_genres_free(&g);
        engine_release(st);
        ok = true;
    }
done:
    pthread_mutex_lock(&s->ev_lock);
    s->aside_running = false;
    pthread_cond_broadcast(&s->ev_cond);
    pthread_mutex_unlock(&s->ev_lock);
    return ok;
}

bool qw_sess_turn(qw_store *st, qw_sess *s, const qw_turn *t,
                  qw_emit_fn emit, void *ud, int *status, char *err, size_t errcap) {
    *status = 400;
    aside_preempt(s);
    pthread_mutex_lock(&st->lock);
    const bool adm = step_admissible(st, s, false, status, err, errcap);
    pthread_mutex_unlock(&st->lock);
    if (!adm) return false;

    /* Attachments: encoded now, because the turn cannot be rendered until
     * the number of placeholder tokens is known. */
    qwasar_image_input images[8];
    int32_t n_images = 0, img_tokens = 0;
    bool is_video = false;
    for (int32_t i = 0; i < t->n_attachments; i++) {
        if (n_images >= 8) { snprintf(err, errcap, "at most 8 attachments per turn"); goto fail_images; }
        const qw_attachment *a = &t->attachments[i];
        const bool video = a->kind && !strcmp(a->kind, "video");
        if (i > 0 && video != is_video) {
            snprintf(err, errcap, "a turn may carry images or a video, not both");
            goto fail_images;
        }
        is_video = video;
        char ext[16] = "";
        if (a->media_type) {
            const char *slash = strchr(a->media_type, '/');
            if (slash) snprintf(ext, sizeof ext, "%s", slash + 1);
        }
        const bool ok = video
            ? qwasar_video_encode_memory(st->e, a->bytes, a->len, ext, &images[n_images], err, errcap)
            : qwasar_image_encode_memory(st->e, a->bytes, a->len, &images[n_images], err, errcap);
        if (!ok) goto fail_images;
        img_tokens += images[n_images].n_rows;
        n_images++;
    }

    {
        int32_t *fresh = NULL, *prefix = NULL;
        int32_t n_fresh = 0, prefix_n = 0;
        if (s->n_tokens <= s->prefix_n) {
            /* The first turn: rendered whole, then split.  A session holding
             * only its prefix -- a first step that failed after evaluating
             * it -- is the same case with the split already evaluated. */
            char **tools = NULL;
            int32_t n_tools = 0;
            if (s->has_tools_doc)
                tools = tools_split(&s->tools_doc, qj_get(&s->tools_doc, qj_root(&s->tools_doc), "tools"), &n_tools);
            qwasar_message m[2] = {
                { .role = "system", .content = s->system },
                { .role = "user", .content = t->text ? t->text : "",
                  .n_image_tokens = img_tokens, .vision_is_video = is_video },
            };
            qwasar_chat_options o;
            chat_opts(s, &o);
            o.tools = n_tools ? (const char *const *)tools : NULL;
            o.n_tools = n_tools;
            int32_t n_full = 0;
            int32_t *full = qwasar_apply_chat_template(st->tok, m, 2, &o, &n_full, err, errcap);
            tools_free(tools, n_tools);
            if (!full) goto fail_images;
            if (s->n_tokens == 0) {
                prefix = render_prefix(st, s, &prefix_n, err, errcap);
                if (!prefix) { free(full); goto fail_images; }
                if (n_full <= prefix_n || memcmp(full, prefix, (size_t)prefix_n * sizeof *full)) {
                    /* Not a prefix after all: never leave a recurrent state
                     * part-way through a system turn it cannot rewind out
                     * of.  The whole turn is evaluated at once, unshared. */
                    free(prefix);
                    prefix = NULL;
                    prefix_n = 0;
                }
            } else if (n_full <= s->n_tokens
                       || memcmp(full, s->tokens, (size_t)s->n_tokens * sizeof *full)) {
                free(full);
                snprintf(err, errcap, "the session's evaluated prefix is not a prefix of this turn");
                goto fail_images;
            }
            const int32_t skip = s->n_tokens + prefix_n;
            n_fresh = n_full - skip;
            fresh = malloc((size_t)n_fresh * sizeof *fresh);
            memcpy(fresh, full + skip, (size_t)n_fresh * sizeof *fresh);
            free(full);
        } else {
            qwasar_chat_options o;
            chat_opts(s, &o);
            fresh = qwasar_render_user_turn(st->tok, t->text ? t->text : "", img_tokens, is_video,
                                            &o, &n_fresh);
            if (!fresh) { snprintf(err, errcap, "cannot render the user turn"); goto fail_images; }
        }
        run_step(st, s, fresh, n_fresh, prefix, prefix_n, images, n_images,
                 &t->sampling, t->max_tokens, emit, ud);
    }
    for (int32_t i = 0; i < n_images; i++) qwasar_image_release(&images[i]);
    return true;

fail_images:
    for (int32_t i = 0; i < n_images; i++) qwasar_image_release(&images[i]);
    *status = 400;
    return false;
}

bool qw_sess_continue(qw_store *st, qw_sess *s, const qw_tool_result *results, int n,
                      const qwasar_sampling *sp, int32_t max_tokens,
                      qw_emit_fn emit, void *ud, int *status, char *err, size_t errcap) {
    *status = 400;
    pthread_mutex_lock(&st->lock);
    bool adm = step_admissible(st, s, true, status, err, errcap);
    if (adm && n != s->n_pending) {
        adm = false;
        *status = 400;
        snprintf(err, errcap, "%d result%s for %d pending call%s", n, n == 1 ? "" : "s",
                 s->n_pending, s->n_pending == 1 ? "" : "s");
    }
    const char **each = adm ? calloc((size_t)(n > 0 ? n : 1), sizeof *each) : NULL;
    if (adm && !each) { adm = false; *status = 500; snprintf(err, errcap, "out of memory"); }
    for (int i = 0; adm && i < n; i++) {
        if (strcmp(results[i].id, s->pending[i].id)) {
            adm = false;
            *status = 400;
            snprintf(err, errcap, "result %d is for %s; expected %s (results in the order the calls were made)",
                     i + 1, results[i].id, s->pending[i].id);
            break;
        }
        each[i] = results[i].content ? results[i].content : "";
    }
    pthread_mutex_unlock(&st->lock);
    if (!adm) { free(each); return false; }

    /* One <tool_response> per result, as the chat template renders
     * consecutive tool messages; joining them into one block is a shape the
     * model was never trained on. */
    qwasar_chat_options o;
    chat_opts(s, &o);
    int32_t n_fresh = 0;
    int32_t *fresh = qwasar_render_tool_results(st->tok, each, n, &o, &n_fresh);
    free(each);
    if (!fresh) { snprintf(err, errcap, "cannot render the tool result"); return false; }
    run_step(st, s, fresh, n_fresh, NULL, 0, NULL, 0, sp, max_tokens, emit, ud);
    return true;
}

/* ---- reattach, cancel, park, delete, describe -------------------------------------- */

bool qw_sess_reattach(qw_store *st, qw_sess *s, const char *last_id, qw_emit_fn emit, void *ud) {
    (void)st;
    int after_step = 0, after_seq = 0;
    if (last_id && sscanf(last_id, "%d.%d", &after_step, &after_seq) != 2) { after_step = 0; after_seq = 0; }
    pthread_mutex_lock(&s->ev_lock);
    if (s->n_ev == 0) { pthread_mutex_unlock(&s->ev_lock); return false; }
    int i = 0;
    bool alive = true;
    for (;;) {
        for (; i < s->n_ev && alive; i++) {
            const qw_event *e = &s->ev[i];
            if (e->step < after_step || (e->step == after_step && e->seq <= after_seq)) continue;
            char id[32];
            snprintf(id, sizeof id, "%d.%d", e->step, e->seq);
            char *name = xstrdup(e->name), *json = xstrdup(e->json);
            pthread_mutex_unlock(&s->ev_lock);
            alive = emit(ud, id, name, json);
            free(name); free(json);
            pthread_mutex_lock(&s->ev_lock);
        }
        if (!alive || !s->step_open) break;
        pthread_cond_wait(&s->ev_cond, &s->ev_lock);
    }
    pthread_mutex_unlock(&s->ev_lock);
    return true;
}

bool qw_sess_cancel(qw_store *st, qw_sess *s) {
    pthread_mutex_lock(&st->lock);
    const bool active = s->state == QW_SESS_QUEUED || s->state == QW_SESS_RUNNING;
    pthread_mutex_unlock(&st->lock);
    if (!active) return false;
    atomic_store(&s->cancel, true);
    pthread_mutex_lock(&st->qlock);
    pthread_cond_broadcast(&st->qcond);
    pthread_mutex_unlock(&st->qlock);
    return true;
}

bool qw_sess_park(qw_store *st, qw_sess *s, char *err, size_t errcap) {
    aside_preempt(s);
    pthread_mutex_lock(&st->lock);
    const bool busy = s->state == QW_SESS_QUEUED || s->state == QW_SESS_RUNNING;
    pthread_mutex_unlock(&st->lock);
    if (busy) { snprintf(err, errcap, "a step is running"); return false; }
    if (!s->h) return true;
    /* The engine, briefly: a save reads the handle's buffers.  Asked for,
     * a park gives the memory back: the handle park_locked keeps as the
     * spare -- right for an eviction, whose next resume reuses it -- is
     * freed, since a client asking to park is asking for its memory. */
    engine_acquire(st, s, false);
    const bool saved = park_locked(st, s, "parked");
    qwasar_session_free(st->spare);
    st->spare = NULL;
    engine_release(st);
    if (!saved && s->n_tokens >= 256 && !st->no_cache) {
        snprintf(err, errcap, "the checkpoint could not be written; the session is cold");
        return false;
    }
    return true;
}

bool qw_sess_purge(qw_store *st, qw_sess *s, char *err, size_t errcap) {
    aside_preempt(s);
    pthread_mutex_lock(&st->lock);
    const bool busy = s->state == QW_SESS_QUEUED || s->state == QW_SESS_RUNNING;
    pthread_mutex_unlock(&st->lock);
    if (busy) { snprintf(err, errcap, "a step is running"); return false; }
    /* Nothing is written: what is on disk stays as it is, and the memory
     * goes -- the handle freed rather than kept as the spare, and any spare
     * with it, since giving memory back is the point. */
    engine_acquire(st, s, false);
    if (s->h) {
        if (st->verbose)
            qw_log("  %s purged from memory at %d tokens; %d on disk", s->id,
                   qwasar_session_n_past(s->h), s->own_n > s->ckpt_n ? s->own_n : s->ckpt_n);
        qwasar_session_free(s->h);
        s->h = NULL;
    }
    qwasar_session_free(st->spare);
    st->spare = NULL;
    engine_release(st);
    return true;
}

bool qw_store_delete(qw_store *st, qw_sess *s, char *err, size_t errcap) {
    aside_preempt(s);
    pthread_mutex_lock(&st->lock);
    if (s->state == QW_SESS_QUEUED || s->state == QW_SESS_RUNNING) {
        pthread_mutex_unlock(&st->lock);
        snprintf(err, errcap, "a step is running; cancel it first");
        return false;
    }
    for (int i = 0; i < st->n_sess; i++)
        if (st->sess[i] == s) { memmove(st->sess + i, st->sess + i + 1, (size_t)(st->n_sess - i - 1) * sizeof *st->sess); st->n_sess--; break; }
    pthread_mutex_unlock(&st->lock);
    if (s->h) {
        engine_acquire(st, s, false);
        handle_put(st, s->h);
        s->h = NULL;
        engine_release(st);
    }
    char dir[1300], path[1400];
    sess_dir(st, s, dir, sizeof dir);
    snprintf(path, sizeof path, "%s/tokens.bin", dir); unlink(path);
    snprintf(path, sizeof path, "%s/checkpoint.bin", dir); unlink(path);
    snprintf(path, sizeof path, "%s/record.json", dir); unlink(path);
    rmdir(dir);
    /* The struct itself is kept: a list or describe on another thread may
     * still be reading it, and a session is a few hundred bytes plus its
     * token log.  Out of the list, it is unreachable from any request. */
    return true;
}

bool qw_sess_drop_checkpoint(qw_store *st, qw_sess *s, uint64_t *freed, char *err, size_t errcap) {
    pthread_mutex_lock(&st->lock);
    const bool busy = s->state == QW_SESS_QUEUED || s->state == QW_SESS_RUNNING;
    pthread_mutex_unlock(&st->lock);
    if (busy) { snprintf(err, errcap, "a step is running"); return false; }
    char path[1400];
    sess_ckpt_path(st, s, path, sizeof path);
    *freed = file_bytes(path);
    if (*freed && unlink(path) != 0) {
        snprintf(err, errcap, "cannot remove %s: %s", path, strerror(errno));
        return false;
    }
    /* A live session's state is in memory still; the next park writes it
     * again.  A parked one is cold now -- its tokens are all still here. */
    s->own_n = 0;
    if (s->h) s->ckpt_n = 0;
    if (st->verbose) qw_log("  %s: checkpoint dropped, %.0f MB freed", s->id, *freed / 1e6);
    return true;
}

void qw_store_disk(qw_store *st, uint64_t *sessions_bytes, uint64_t *cache_bytes,
                   uint64_t *free_bytes) {
    uint64_t total = 0;
    pthread_mutex_lock(&st->lock);
    for (int i = 0; i < st->n_sess; i++) {
        char path[1400];
        sess_ckpt_path(st, st->sess[i], path, sizeof path);
        total += file_bytes(path);
    }
    pthread_mutex_unlock(&st->lock);
    *sessions_bytes = total;
    int entries = 0;
    qwasar_kv_cache_stats(cache_bytes, &entries);
    struct statfs fs;
    *free_bytes = statfs(st->dir, &fs) == 0 ? (uint64_t)fs.f_bavail * fs.f_bsize : 0;
}

void qw_sess_info_get(qw_store *st, qw_sess *s, qw_sess_info *out) {
    memset(out, 0, sizeof *out);
    pthread_mutex_lock(&st->lock);
    out->id = s->id;
    out->metadata = s->metadata ? s->metadata : "{}";
    out->model_id = s->model_id;
    out->model_mismatch = strcmp(s->model_id, st->model_id) != 0;
    out->created = s->created;
    out->n_tokens = s->n_tokens;
    out->prefix_tokens = s->prefix_n;
    out->ctx = st->ctx;
    out->step = s->step;
    out->state = s->state;
    out->last_stop = s->last_stop[0] ? s->last_stop : NULL;
    out->last_at = s->last_at;
    out->pending = s->pending;
    out->n_pending = s->n_pending;
    const bool live = s->h != NULL;
    pthread_mutex_unlock(&st->lock);

    if (live) {
        char own[1400];
        sess_ckpt_path(st, s, own, sizeof own);
        out->checkpoint_bytes = file_bytes(own);
        out->warmth = QW_WARMTH_LIVE;
        out->covered = s->n_tokens;
        return;
    }
    int32_t covered = 0;
    char own[1400];
    sess_ckpt_path(st, s, own, sizeof own);
    out->checkpoint_bytes = file_bytes(own);
    if (!st->no_cache && s->n_tokens > 0) {
        const int32_t by_file = qwasar_kv_probe_file(st->e, own, s->tokens, s->n_tokens);
        const int32_t by_cache = qwasar_kv_probe(st->e, s->tokens, s->n_tokens);
        covered = by_file > by_cache ? by_file : by_cache;
    }
    out->covered = covered;
    out->warmth = covered > 0 ? QW_WARMTH_WARM : QW_WARMTH_COLD;
    const double read_s = covered > 0
        ? (150e6 + (double)covered * (double)st->prof.kv_per_token) / (st->restore_bps > 0 ? st->restore_bps : 2e9)
        : 0;
    const double prefill = (double)(s->n_tokens - covered) / (st->prefill_tps > 0 ? st->prefill_tps : 32.0);
    out->estimate_s = read_s + prefill;
}

/* ---- the compat front ------------------------------------------------------------ */

static void token_text(qw_store *st, const int32_t *toks, int32_t from, int32_t to,
                       char *out, size_t cap) {
    size_t o = 0;
    for (int32_t i = from < 0 ? 0 : from; i < to && o + 8 < cap; i++) {
        size_t len = 0;
        bool special = false;
        const char *b = qwasar_token_bytes(st->tok, toks[i], &len, &special);
        for (size_t k = 0; b && k < len && o + 8 < cap; k++) {
            const unsigned char ch = (unsigned char)b[k];
            if (ch == '\n')      { out[o++] = '\\'; out[o++] = 'n'; }
            else if (ch == '\t') { out[o++] = '\\'; out[o++] = 't'; }
            else if (ch < 0x20)  o += (size_t)snprintf(out + o, cap - o, "\\x%02x", ch);
            else                 out[o++] = (char)ch;
        }
    }
    out[o] = 0;
}

#define QW_SRV_CKPT_SPAN 4096

/* Evaluates tokens[from, n) on the compat session, stopping on the way at a
 * checkpoint boundary to write one, and leaving a rewind point one token
 * short of the end. */
static const float *compat_eval_from(qw_store *st, qw_sess *s, const int32_t *tokens,
                                     int32_t from, int32_t n, const qw_ckpt_marks *mk,
                                     char *err, size_t cap) {
    int32_t stops[2], ns = 0;
    if (mk && !st->no_cache) {
        if (mk->sys_n > from && mk->sys_n < n && mk->sys_n > s->ckpt_n) stops[ns++] = mk->sys_n;
        const int32_t last = ns ? stops[0] : s->ckpt_n;
        if (mk->hist_n > from && mk->hist_n < n && mk->hist_n - last >= QW_SRV_CKPT_SPAN)
            stops[ns++] = mk->hist_n;
    }
    for (int i = 0; i < ns; i++) {
        if (!qwasar_session_eval(s->h, tokens + from, stops[i] - from, err, cap)) return NULL;
        from = stops[i];
        char serr[256];
        const double t0 = qw_now();
        const bool saved = qwasar_session_save(s->h, st->e, serr, sizeof serr);
        if (saved) s->ckpt_n = from;
        if (st->verbose) {
            if (saved) qw_log("  checkpoint written at %d tokens in %.2fs", from, qw_now() - t0);
            else qw_log("  no checkpoint at %d tokens: %s", from, serr);
        }
    }
    if (n - from > 1) {
        if (!qwasar_session_eval(s->h, tokens + from, n - 1 - from, err, cap)) return NULL;
        from = n - 1;
    }
    qwasar_session_mark(s->h);
    return qwasar_session_eval(s->h, tokens + from, n - from, err, cap);
}

const float *qw_compat_prefill(qw_store *st, qw_sess *s, const int32_t *tokens, int32_t n,
                               const qwasar_image_input *images, int32_t n_images,
                               const qw_ckpt_marks *mk, int32_t *reused, const char **how,
                               char *err, size_t cap) {
    *reused = 0;
    *how = "a new session";
    s->last_used = qw_now();

    /* Images defeat prefix reuse, and silently: two pictures render to the
     * same run of placeholder tokens.  A request carrying them starts fresh. */
    if (n_images > 0) {
        handle_put(st, s->h);
        s->h = NULL;
        s->ckpt_n = 0;
        admit(st, s);
        *how = "a new session (images)";
        s->h = handle_take(st, err, cap);
        if (!s->h) return NULL;
        return qwasar_session_eval_images(s->h, tokens, n, images, n_images, err, cap);
    }

    int32_t live = s->h ? qwasar_session_common_prefix(s->h, tokens, n) : 0;
    if (live > 0 && live < n) {
        *reused = live;
        *how = "the live session";
        return compat_eval_from(st, s, tokens, live, n, mk, err, cap);
    }
    if (live == n && n > 0) {
        *reused = n;
        *how = "the live session, already at its end";
        const float *l = qwasar_session_logits(s->h);
        if (l) return l;
    }

    const int32_t back = s->h ? qwasar_session_rewind(s->h, tokens, n) : 0;
    if (back > 0) {
        *reused = back;
        *how = "the live session, rewound to the last prompt";
        return compat_eval_from(st, s, tokens, back, n, mk, err, cap);
    }

    st->miss_at = -1;
    if (s->h && st->verbose) {
        const int32_t *had = NULL;
        int32_t n_had = 0;
        const int32_t at = qwasar_session_divergence(s->h, tokens, n, &had, &n_had);
        if (had && n_had > 0) {
            st->miss_at = at;
            st->miss_of = n_had;
            token_text(st, had, at - 6, at + 14 < n_had ? at + 14 : n_had, st->miss_had, sizeof st->miss_had);
            token_text(st, tokens, at - 6, at + 14 < n ? at + 14 : n, st->miss_got, sizeof st->miss_got);
        }
    }

    handle_put(st, s->h);
    s->h = NULL;
    admit(st, s);
    s->h = handle_take(st, err, cap);
    if (!s->h) return NULL;

    /* A checkpoint covers at most n-1 tokens, so there is a token left to
     * evaluate for the logits. */
    const double t0 = qw_now();
    int32_t covered = st->no_cache ? 0 : qwasar_session_restore(s->h, st->e, tokens, n - 1);
    *reused = covered;
    s->ckpt_n = covered;
    if (covered > 0) {
        *how = "a disk checkpoint";
        if (st->verbose)
            qw_log("  restored %d tokens from a disk checkpoint in %.2fs", covered, qw_now() - t0);
    }
    return compat_eval_from(st, s, tokens, covered, n, mk, err, cap);
}

void qw_compat_last_miss(qw_store *st, int32_t *miss_at, int32_t *miss_of,
                         const char **had, const char **got) {
    *miss_at = st->miss_at;
    *miss_of = st->miss_of;
    *had = st->miss_had;
    *got = st->miss_got;
}

/* ---- shutdown ------------------------------------------------------------------- */

void qw_store_shutdown(qw_store *st) {
    atomic_store(&g_stopping, true);
    /* Wake anyone queued -- they leave, cancelled -- and any cancel-polling
     * step ends at its next token; then the engine is ours. */
    pthread_mutex_lock(&st->lock);
    for (int i = 0; i < st->n_sess; i++) atomic_store(&st->sess[i]->cancel, true);
    pthread_mutex_unlock(&st->lock);
    pthread_mutex_lock(&st->qlock);
    pthread_cond_broadcast(&st->qcond);
    pthread_mutex_unlock(&st->qlock);
    engine_acquire(st, st->compat, false);
    if (!st->no_cache) {
        for (int i = 0; i < st->n_sess; i++) {
            qw_sess *s = st->sess[i];
            if (!s->h) continue;
            if (s->n_tokens - s->own_n > 0) park_locked(st, s, "shutdown");
            else { qwasar_session_free(s->h); s->h = NULL; }
            char err[256];
            sess_persist(st, s, err, sizeof err);
        }
        if (st->compat->h) park_locked(st, st->compat, "shutdown");
    }
    qwasar_session_free(st->spare);
    st->spare = NULL;
    engine_release(st);
}
