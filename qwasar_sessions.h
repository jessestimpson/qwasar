#ifndef QWASAR_SESSIONS_H
#define QWASAR_SESSIONS_H

#include "qwasar.h"
#include "qwasar_http.h"
#include "qwasar_json.h"
#include "qwasar_toolcall.h"

#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>

/* The session store: the one place that owns qwasar_session handles.
 *
 * API.md §2 in code.  A session is a prefix (system prompt, tools, thinking,
 * effort -- fixed at open), a timeline (the tokens the model has evaluated,
 * kept here and rewritten to disk after every step), and a warmth (live in
 * the engine, warm on disk, or cold).  Steps run on the calling thread while
 * it holds the engine; the store queues callers, admits sessions to the live
 * set -- parking the least recently used when the set is full -- and resumes
 * a session from whatever it has before evaluating anything new.
 *
 * The OpenAI and Anthropic endpoints are a client too: one anonymous session
 * (qw_store_compat) that they match to a resent prompt by prefix, the way
 * the server always has.  It shares the queue and the live set with the
 * named ones, so a stateless client and the Session API compete fairly for
 * one engine rather than each assuming it has the machine. */

typedef struct qw_store qw_store;
typedef struct qw_sess  qw_sess;

/* ---- the profile ------------------------------------------------------------
 *
 * Context per session and the size of the live set, derived from the model
 * and the machine (Crucible's MemoryProfile, PLAN.md 2.3, ported here because
 * the server is the only process that knows how many sessions exist). */
typedef struct {
    char     family[16];        /* "qwen3_5" | "qwen4_exp" */
    uint64_t physical;          /* hw.memsize */
    uint64_t working_set;       /* Metal's recommendedMaxWorkingSetSize */
    uint64_t weights;           /* bytes the GPU holds: every tensor but the engram rows */
    uint64_t kv_per_token;      /* cache bytes a token costs, every cache included */
    uint64_t fixed;             /* per-session bytes that do not grow with context */
    double   reserve;           /* fraction of the working set spent; 0.85 */
    int32_t  max_ctx;           /* the model's trained window */
    int32_t  ctx;               /* what a session gets */
    int      live;              /* how many may be live at once */
    char     note[200];         /* "" or why the machine is short */
} qw_profile;

/* Reads config.json and the shard headers under `model_path`; touches no
 * weights.  `working_set` 0 means ask Metal. */
bool qw_profile_derive(const char *model_path, uint64_t working_set, qw_profile *out,
                       char *err, size_t errcap);

/* ---- the store ------------------------------------------------------------- */

typedef struct {
    const char *state_dir;      /* NULL: ~/Library/Application Support/Qwasar */
    int32_t     ctx;            /* 0: the profile's */
    int         live;           /* 0: the profile's */
    bool        no_cache;       /* neither read nor write disk checkpoints */
    bool        verbose;
} qw_store_opts;

/* The engine must already be loaded with `ctx` as its context (the store
 * cannot resize it), so the profile is derived first, the engine loaded at
 * its context, then the store opened.  Reads every session record under
 * state_dir; evaluates nothing. */
qw_store *qw_store_open(qwasar_engine *e, qwasar_tokenizer *tok, const char *model_path,
                        const qw_profile *p, const qw_store_opts *o, char *err, size_t errcap);

/* Shutdown: steps in flight end at their next token, every live session is
 * checkpointed, records are written.  Blocks until done. */
void qw_store_shutdown(qw_store *st);
/* True once shutdown has begun; the generation loop polls it. */
bool qw_store_stopping(void);

qwasar_engine    *qw_store_engine(const qw_store *st);
qwasar_tokenizer *qw_store_tokenizer(const qw_store *st);
const qw_profile *qw_store_profile(const qw_store *st);
const char       *qw_store_state_dir(const qw_store *st);
const char       *qw_store_model_path(const qw_store *st);
int32_t           qw_store_ctx(const qw_store *st);
bool              qw_store_no_cache(const qw_store *st);
/* Seeds the sampler for the next generation: 0 draws from the clock. */
void              qw_store_seed(qw_store *st, uint64_t seed);
void              qw_store_counts(qw_store *st, int *total, int *live, int *queued);

/* One timestamped line on stderr, the server's log. */
void qw_log(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
double qw_now(void);

/* ---- sessions -------------------------------------------------------------- */

typedef enum {
    QW_SESS_IDLE,
    QW_SESS_QUEUED,
    QW_SESS_RUNNING,
    QW_SESS_AWAITING,   /* the last step ended in tool calls; a continue is due */
    QW_SESS_FULL,       /* the window is used up */
} qw_sess_state;

typedef enum { QW_WARMTH_LIVE, QW_WARMTH_WARM, QW_WARMTH_COLD } qw_warmth;

const char *qw_sess_state_name(qw_sess_state s);   /* "idle", "awaiting_tools", ... */
const char *qw_warmth_name(qw_warmth w);          /* "live" | "warm" | "cold" */

typedef struct {
    char    id[24];
    char    name[64];
} qw_pending_call;

/* A snapshot for describe/list, taken under the store's lock; the strings
 * point into the session and are valid while the store lives. */
typedef struct {
    const char     *id;
    const char     *metadata;      /* JSON object text, "{}" when none */
    const char     *model_id;      /* the model that ran it */
    bool            model_mismatch;/* not the loaded model: steps refused */
    int64_t         created;
    int32_t         n_tokens;
    int32_t         prefix_tokens;
    int32_t         ctx;
    int             step;          /* steps taken so far */
    qw_sess_state   state;
    qw_warmth       warmth;
    int32_t         covered;       /* tokens the warmth covers */
    uint64_t        checkpoint_bytes; /* the session's own checkpoint on disk, or 0 */
    double          estimate_s;    /* to resume: a read, or a re-prefill */
    const char     *last_stop;     /* NULL before the first step */
    int64_t         last_at;
    const qw_pending_call *pending;
    int             n_pending;
} qw_sess_info;

/* Opens a session: renders the prefix and counts it, writes the record,
 * evaluates nothing.  `tools` are JSON objects, one per tool; `metadata` is
 * JSON object text or NULL.  Returns NULL with `err` on a prefix that will
 * not render or a store that cannot be written. */
qw_sess *qw_store_open_session(qw_store *st, const char *system,
                               const char *const *tools, int32_t n_tools,
                               bool thinking, const char *effort,
                               const char *metadata, char *err, size_t errcap);

qw_sess *qw_store_find(qw_store *st, const char *id);
/* Every session, newest first, into `out` (at most `cap`); returns the count. */
int      qw_store_list(qw_store *st, qw_sess **out, int cap);
/* Snapshot, probing the disk store for warmth (a directory scan). */
void     qw_sess_info_get(qw_store *st, qw_sess *s, qw_sess_info *out);
const char *qw_sess_id(const qw_sess *s);

/* A step's events, as the store emits them: the name and a JSON object.
 * The callback returns false when the peer is gone, which does not stop the
 * step -- a disconnect is not a cancel -- only the sending. */
typedef bool (*qw_emit_fn)(void *ud, const char *id, const char *event, const char *json);

typedef struct {
    const char *kind;          /* "image" | "video" */
    const char *media_type;    /* "image/png", "video/mp4", ... */
    const unsigned char *bytes;
    size_t      len;
} qw_attachment;

typedef struct {
    const char      *text;
    const qw_attachment *attachments;
    int32_t          n_attachments;
    qwasar_sampling  sampling;
    int32_t          max_tokens;   /* 0: the room the window has left */
} qw_turn;

typedef struct {
    const char *id;
    const char *content;
} qw_tool_result;

/* The two steps.  Each runs to completion on the calling thread: waits for
 * the engine (emitting `queued`), admits and resumes the session (`resume`,
 * `prefill`), evaluates and generates (the rest of API.md §5), ends with
 * `done` or `error`, and persists.  Returns false only when the step could
 * not start; `err` says why and which HTTP status fits (`status`). */
bool qw_sess_turn(qw_store *st, qw_sess *s, const qw_turn *t,
                  qw_emit_fn emit, void *ud, int *status, char *err, size_t errcap);
bool qw_sess_continue(qw_store *st, qw_sess *s, const qw_tool_result *results, int n,
                      const qwasar_sampling *sp, int32_t max_tokens,
                      qw_emit_fn emit, void *ud, int *status, char *err, size_t errcap);

/* Replays the buffered events of the step in flight (or the last one) after
 * `last_id` (NULL: all of them), then follows the live tail until the step
 * ends or `emit` returns false.  False when there is nothing to replay. */
bool qw_sess_reattach(qw_store *st, qw_sess *s, const char *last_id, qw_emit_fn emit, void *ud);

/* Ends the step in flight at its next token, or removes a queued one.
 * True if there was something to cancel. */
bool qw_sess_cancel(qw_store *st, qw_sess *s);
/* Checkpoints the session and frees its memory -- the handle, and the
 * store's spare.  (Eviction from the live set parks too, and keeps the
 * handle as the spare for the next resume.)  False with `err` while a step
 * runs or when nothing could be written (the session is then cold). */
bool qw_sess_park(qw_store *st, qw_sess *s, char *err, size_t errcap);
/* A step off the record: `text` as a user turn, answered with thinking off
 * and no tools, streamed to `emit` as `text` events and a final `done` --
 * then rolled back to the rewind point taken before it, so the session's
 * timeline, token log and checkpoints are exactly as they were.  For the
 * app's running notes, written while the user reads a reply.  Only on an
 * idle session in memory, only when the engine is free (it never queues),
 * and a turn, continue, park or delete on the session ends it at once.
 * False with `status`/`err` when it cannot start. */
bool qw_sess_aside(qw_store *st, qw_sess *s, const char *text, int32_t max_tokens,
                   const qwasar_sampling *sp, qw_emit_fn emit, void *ud,
                   int *status, char *err, size_t errcap);
/* Frees the session's memory without writing anything: the checkpoints it
 * already has on disk stay, and its warmth is whatever they cover.  The
 * store's spare handle goes too.  False with `err` while a step runs. */
bool qw_sess_purge(qw_store *st, qw_sess *s, char *err, size_t errcap);
/* Deletes the session's own checkpoint: a parked session becomes cold (its
 * tokens stay, and a resume re-prefills what the shared cache does not
 * cover); a live one is rewritten at its next park.  The one way disk used
 * by sessions is given back short of deleting them, and never done by the
 * server on its own.  False while a step runs. */
bool qw_sess_drop_checkpoint(qw_store *st, qw_sess *s, uint64_t *freed, char *err, size_t errcap);
/* Bytes in session checkpoints, in the shared prefix cache, and free on the
 * volume that holds the state directory. */
void qw_store_disk(qw_store *st, uint64_t *sessions_bytes, uint64_t *cache_bytes,
                   uint64_t *free_bytes);
/* Removes the session and everything on disk that is its own. */
bool qw_store_delete(qw_store *st, qw_sess *s, char *err, size_t errcap);

/* ---- the compat front's use of the store -------------------------------------
 *
 * The anonymous session, matched to resent prompts by prefix.  The caller
 * holds the engine around the whole of a request (qw_store_acquire /
 * qw_store_release), prefills with the ladder that has always served
 * stateless clients (the live session, its rewind point, a disk checkpoint,
 * a fresh session), then generates. */
qw_sess *qw_store_compat(qw_store *st);
void     qw_store_acquire(qw_store *st, qw_sess *s);
void     qw_store_release(qw_store *st, qw_sess *s);

typedef struct {
    int32_t sys_n;    /* end of the system prompt and tools: every conversation shares it */
    int32_t hist_n;   /* end of the last complete turn, before the generation prompt */
} qw_ckpt_marks;

const float *qw_compat_prefill(qw_store *st, qw_sess *s, const int32_t *tokens, int32_t n,
                               const qwasar_image_input *images, int32_t n_images,
                               const qw_ckpt_marks *mk, int32_t *reused, const char **how,
                               char *err, size_t errcap);
/* The last prompt that could not reuse the live session (-v): where it
 * departed, and the text on each side.  miss_at < 0 when there was none. */
void qw_compat_last_miss(qw_store *st, int32_t *miss_at, int32_t *miss_of,
                         const char **had, const char **got);

/* ---- generation -------------------------------------------------------------- */

#define QW_MAX_STOPS 16
#define QW_MAX_TOOLS 32

typedef enum {
    QW_TOOLS_AUTO,         /* the model decides */
    QW_TOOLS_NONE,         /* no call may start */
    QW_TOOLS_FORCE,        /* the answer is a call, to `force_name` if set */
} qw_tool_mode;

typedef struct {
    const char  *stops[QW_MAX_STOPS];
    int          n_stops;
    qw_tool_mode tools;
    const char  *force_name;
    /* The tool names a forced call without a name is constrained to. */
    const char  *names[QW_MAX_TOOLS];
    int          n_names;
} qw_genopts;

typedef struct {
    str     text;
    str     reasoning;
    int32_t n_gen;
    int32_t n_reasoning;   /* of n_gen, inside the reasoning block */
    bool    hit_eos;
    bool    hit_stop;      /* ended on one of the stop sequences */
    int     stop_index;    /* which one */
    bool    has_call;
    bool    cancelled;
    /* A tool call opened inside the reasoning block -- the model skipped its
     * </think> -- and was read as the end of it. */
    bool    call_in_reasoning;
} qw_genres;

/* A delta of reasoning or content, UTF-8 complete. */
typedef void (*qw_delta_fn)(void *ud, bool reasoning, const char *s, size_t n);
/* Every token, after it is taken: how many so far, whether a call is being
 * written, and the answer text accumulated so far (for a partial parse). */
typedef void (*qw_token_fn)(void *ud, int32_t n_gen, bool in_call, const char *text, size_t len);

/* One assistant turn on the session's live handle, from `logits`.  Stops at
 * end-of-turn, a stop sequence, a completed tool call, the budget, a cancel,
 * or shutdown.  Caller holds the engine. */
bool qw_generate(qw_store *st, qw_sess *s, const float *logits, const qwasar_sampling *sp,
                 int32_t max_tokens, bool thinking, const qw_genopts *go,
                 qw_delta_fn on_delta, qw_token_fn on_token, void *ud, atomic_bool *cancel,
                 qw_genres *out, char *err, size_t errcap);
void qw_genres_free(qw_genres *g);

/* Tool arguments as a JSON object.  `tools` is a document whose root has a
 * "tools" array (a request, or a session's), consulted for which parameters
 * are declared strings. */
void qw_args_json(str *out, const qw_tool_call *c, const qj_doc *tools);

#endif /* QWASAR_SESSIONS_H */
