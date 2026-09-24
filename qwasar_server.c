/* qwasar-server -- an OpenAI and Anthropic compatible HTTP front end.
 *
 * The API surface follows ds4-server so the same clients work against either.
 * What differs is underneath, and it is worth stating plainly: this engine
 * holds one session, because 48 of the model's 64 layers are recurrent and
 * their state cannot be forked cheaply the way a KV cache can. So requests are
 * served one at a time, and the concurrency features ds4-server has --
 * --batched-session, mixed prefill scheduling -- have no counterpart here.
 *
 * What does carry over is prefix reuse, which is what actually matters for
 * stateless clients: an agent that resends a growing conversation on every turn
 * continues from wherever the live session already is, and falls back to a disk
 * checkpoint when the live session has moved on to something else.  The server
 * writes those checkpoints itself: at the end of the system prompt, which every
 * conversation with that prompt and those tools shares, and at the last complete
 * turn of a long conversation (srv_eval_from). */

#include "qwasar.h"
#include "qwasar_json.h"
#include "qwasar_toolcall.h"

#include <arpa/inet.h>
#include <pthread.h>
#include <errno.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

/* The loaded model's id and display name (qwasar_model_id), set once the
 * model is loaded and read-only from then on. */
static const char *model_id = "";
static const char *model_name = "";

/* ---- growable text --------------------------------------------------------- */

typedef struct { char *p; size_t len, cap; } str;

static bool str_add(str *s, const char *d, size_t n) {
    if (s->len + n + 1 > s->cap) {
        size_t cap = s->cap ? s->cap * 2 : 1024;
        while (cap < s->len + n + 1) cap *= 2;
        char *p = realloc(s->p, cap);
        if (!p) return false;
        s->p = p; s->cap = cap;
    }
    memcpy(s->p + s->len, d, n);
    s->len += n;
    s->p[s->len] = 0;
    return true;
}
static bool str_puts(str *s, const char *t) { return t ? str_add(s, t, strlen(t)) : true; }
static void str_free(str *s) { free(s->p); s->p = NULL; s->len = s->cap = 0; }

static void str_printf(str *s, const char *fmt, ...) {
    char buf[2048];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    if (n > 0) str_add(s, buf, (size_t)n < sizeof buf ? (size_t)n : sizeof buf - 1);
}

/* Appends `t` as a JSON string, quotes included. */
static void str_json(str *s, const char *t, size_t n) {
    str_add(s, "\"", 1);
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)t[i];
        switch (c) {
        case '"':  str_puts(s, "\\\""); break;
        case '\\': str_puts(s, "\\\\"); break;
        case '\n': str_puts(s, "\\n");  break;
        case '\r': str_puts(s, "\\r");  break;
        case '\t': str_puts(s, "\\t");  break;
        case '\b': str_puts(s, "\\b");  break;
        case '\f': str_puts(s, "\\f");  break;
        default:
            /* Everything else goes through as UTF-8; only C0 needs escaping. */
            if (c < 0x20) str_printf(s, "\\u%04x", c);
            else str_add(s, (const char *)&c, 1);
        }
    }
    str_add(s, "\"", 1);
}
static void str_jsons(str *s, const char *t) { str_json(s, t ? t : "", t ? strlen(t) : 0); }

/* Re-serialises a parsed node.  Tool schemas arrive as JSON and have to reach
 * the model's prompt as JSON; the parser unescapes strings in place, so the
 * original bytes are gone by then and the node has to be written back out. */
static void str_node(str *s, const qj_doc *d, const qj_node *n) {
    if (!n) { str_puts(s, "null"); return; }
    switch (n->type) {
    case QJ_NULL:   str_puts(s, "null");  break;
    case QJ_TRUE:   str_puts(s, "true");  break;
    case QJ_FALSE:  str_puts(s, "false"); break;
    case QJ_NUMBER: {
        double v = n->u.num;
        if (v == (double)(long long)v) str_printf(s, "%lld", (long long)v);
        else str_printf(s, "%.17g", v);
        break;
    }
    case QJ_STRING: str_json(s, d->text + n->u.str.off, n->u.str.len); break;
    case QJ_ARRAY:
        str_puts(s, "[");
        for (const qj_node *c = qj_first(d, n); c; c = qj_next(d, c)) {
            if (c != qj_first(d, n)) str_puts(s, ", ");
            str_node(s, d, c);
        }
        str_puts(s, "]");
        break;
    case QJ_OBJECT:
        str_puts(s, "{");
        for (const qj_node *c = qj_first(d, n); c; c = qj_next(d, c)) {
            if (c != qj_first(d, n)) str_puts(s, ", ");
            str_json(s, d->text + c->key_off, c->key_len);
            str_puts(s, ": ");
            str_node(s, d, c);
        }
        str_puts(s, "}");
        break;
    }
}

/* ---- http ------------------------------------------------------------------ */

typedef struct {
    int   fd;
    bool  cors;
    bool  streaming;   /* headers already sent, body is chunked */
    bool  dead;        /* the peer went away */
    bool  anthropic;   /* errors in Anthropic's envelope rather than OpenAI's */
} conn;

static bool conn_write(conn *c, const char *data, size_t n) {
    if (c->dead) return false;
    while (n > 0) {
        ssize_t w = write(c->fd, data, n);
        if (w <= 0) {
            if (errno == EINTR) continue;
            c->dead = true;
            return false;
        }
        data += w;
        n -= (size_t)w;
    }
    return true;
}

static void conn_cors(conn *c, str *h) {
    if (!c->cors) return;
    str_puts(h, "Access-Control-Allow-Origin: *\r\n"
                "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
                "Access-Control-Allow-Headers: *\r\n");
}

static void http_send(conn *c, int status, const char *reason,
                      const char *ctype, const char *body, size_t len) {
    str h = { 0 };
    str_printf(&h, "HTTP/1.1 %d %s\r\n", status, reason);
    str_printf(&h, "Content-Type: %s\r\n", ctype);
    str_printf(&h, "Content-Length: %zu\r\n", len);
    conn_cors(c, &h);
    str_puts(&h, "Connection: keep-alive\r\n\r\n");
    conn_write(c, h.p, h.len);
    if (len) conn_write(c, body, len);
    str_free(&h);
}

/* Anthropic's error types, by status. */
static const char *anthropic_error_type(int status) {
    switch (status) {
    case 401: return "authentication_error";
    case 403: return "permission_error";
    case 404: return "not_found_error";
    case 413: return "request_too_large";
    case 429: return "rate_limit_error";
    case 503: case 529: return "overloaded_error";
    default:  return status >= 500 && status != 501 ? "api_error" : "invalid_request_error";
    }
}

/* {"type": "error", "error": {"type": ..., "message": ...}} -- the body of an
 * Anthropic error response, and the data of an `error` event mid-stream. */
static void anthropic_error_body(str *b, int status, const char *msg) {
    str_printf(b, "{\"type\": \"error\", \"error\": {\"type\": \"%s\", \"message\": ",
               anthropic_error_type(status));
    str_jsons(b, msg);
    str_puts(b, "}}");
}

/* The error envelope for whichever API the client speaks.  OpenAI's carries
 * message, type, and the param and code a client switches on -- each a string
 * or null, and always present. */
static void http_error_at(conn *c, int status, const char *reason, const char *msg,
                          const char *param, const char *code) {
    str b = { 0 };
    if (c->anthropic) {
        anthropic_error_body(&b, status, msg);
        http_send(c, status, reason, "application/json", b.p, b.len);
        str_free(&b);
        return;
    }
    str_puts(&b, "{\"error\": {\"message\": ");
    str_jsons(&b, msg);
    str_printf(&b, ", \"type\": \"%s\", \"param\": ",
               status >= 500 && status != 501 ? "server_error" : "invalid_request_error");
    if (param) str_jsons(&b, param); else str_puts(&b, "null");
    str_puts(&b, ", \"code\": ");
    if (code) str_jsons(&b, code); else str_puts(&b, "null");
    str_puts(&b, "}}");
    http_send(c, status, reason, "application/json", b.p, b.len);
    str_free(&b);
}

static void http_error(conn *c, int status, const char *reason, const char *msg) {
    http_error_at(c, status, reason, msg, NULL, NULL);
}

/* Server-sent events over chunked transfer, so the connection survives the
 * response and a client can reuse it. */
static void sse_begin(conn *c) {
    str h = { 0 };
    str_puts(&h, "HTTP/1.1 200 OK\r\n"
                 "Content-Type: text/event-stream\r\n"
                 "Cache-Control: no-cache\r\n"
                 "Transfer-Encoding: chunked\r\n");
    conn_cors(c, &h);
    str_puts(&h, "Connection: keep-alive\r\n\r\n");
    conn_write(c, h.p, h.len);
    str_free(&h);
    c->streaming = true;
}

static void sse_chunk(conn *c, const char *data, size_t n) {
    char head[32];
    int hn = snprintf(head, sizeof head, "%zx\r\n", n);
    conn_write(c, head, (size_t)hn);
    conn_write(c, data, n);
    conn_write(c, "\r\n", 2);
}

static void sse_event(conn *c, const char *event, const char *json) {
    str f = { 0 };
    if (event) str_printf(&f, "event: %s\n", event);
    str_printf(&f, "data: %s\n\n", json);
    sse_chunk(c, f.p, f.len);
    str_free(&f);
}

static void sse_end(conn *c) {
    conn_write(c, "0\r\n\r\n", 5);
    c->streaming = false;
}

/* ---- engine ---------------------------------------------------------------- */

typedef struct {
    qwasar_engine    *e;
    qwasar_tokenizer *tok;
    qwasar_session   *s;
    int32_t           ctx;
    int32_t           max_tokens;    /* output when a request names none; 0 = the context's room */
    bool              no_cache;
    int32_t           ckpt_n;        /* longest checkpoint the live session's prefix has */
    uint64_t          rng;
    bool              verbose;
    pthread_mutex_t   lock;          /* held for the whole of a completion */
    int               n_conns;       /* live connections, under conns_lock */
    pthread_mutex_t   conns_lock;
} server;

/* Where the prompt is worth a disk checkpoint (srv_eval_from). */
typedef struct {
    int32_t sys_n;    /* end of the system prompt and tools: every conversation shares it */
    int32_t hist_n;   /* end of the last complete turn, before the generation prompt */
} ckpt_marks;

/* A long conversation leaves a checkpoint at its last complete turn once this
 * many tokens have gone by since the last one it has, so a conversation the
 * live session was taken from -- by another client, a side request, a restart
 * -- resumes near where it was.  A Flash-Next checkpoint is ~127 MB whatever
 * its length, so not every turn. */
#define QW_SRV_CKPT_SPAN 4096

/* Evaluates tokens[from, n), stopping on the way at a checkpoint boundary
 * to write one.  Same total work either way; a boundary past the end, or
 * one already covered, is ignored. */
static const float *srv_eval_from(server *sv, const int32_t *tokens, int32_t from, int32_t n,
                                  const ckpt_marks *mk, char *err, size_t cap) {
    int32_t stops[2], ns = 0;
    if (mk && !sv->no_cache) {
        if (mk->sys_n > from && mk->sys_n < n && mk->sys_n > sv->ckpt_n) stops[ns++] = mk->sys_n;
        const int32_t last = ns ? stops[0] : sv->ckpt_n;
        if (mk->hist_n > from && mk->hist_n < n && mk->hist_n - last >= QW_SRV_CKPT_SPAN)
            stops[ns++] = mk->hist_n;
    }
    for (int i = 0; i < ns; i++) {
        if (!qwasar_session_eval(sv->s, tokens + from, stops[i] - from, err, cap)) return NULL;
        from = stops[i];
        char serr[256];
        struct timeval t0, t1;
        gettimeofday(&t0, NULL);
        const bool saved = qwasar_session_save(sv->s, sv->e, serr, sizeof serr);
        gettimeofday(&t1, NULL);
        if (saved) sv->ckpt_n = from;
        if (sv->verbose) {
            const double dt = (double)(t1.tv_sec - t0.tv_sec) + (t1.tv_usec - t0.tv_usec) * 1e-6;
            if (saved) fprintf(stderr, "  checkpoint at %d tokens (%.2fs)\n", from, dt);
            else fprintf(stderr, "  no checkpoint at %d tokens: %s\n", from, serr);
        }
    }
    return qwasar_session_eval(sv->s, tokens + from, n - from, err, cap);
}

/* Evaluates `tokens`, reusing whatever the live session already covers.
 *
 * Prefix reuse is all-or-nothing here.  A KV cache could be truncated back to
 * the first divergence, but the recurrent layers keep no per-position history,
 * so a session that has evaluated one wrong token is worthless for this prompt
 * and has to be replaced. */
static const float *srv_prefill(server *sv, const int32_t *tokens, int32_t n,
                                const qwasar_image_input *images, int32_t n_images,
                                const ckpt_marks *mk, int32_t *reused, char *err, size_t cap) {
    *reused = 0;

    /* Images defeat prefix reuse, and silently.  Two different pictures render
     * to the same run of <|image_pad|> tokens, so a token-sequence match can
     * say a prompt is a prefix of the live session when the pixels behind it
     * were something else entirely.  Nothing downstream would notice.  A
     * request carrying images therefore starts from a fresh session, which
     * costs a prefill and cannot be wrong.
     *
     * Fixing this properly means keying the match on a digest of the image
     * bytes as well as the tokens, which is worth doing when someone is holding
     * a conversation about a picture. */
    if (n_images > 0) {
        if (sv->s) qwasar_session_free(sv->s);
        sv->ckpt_n = 0;
        sv->s = qwasar_session_new(sv->e, err, cap);
        if (!sv->s) return NULL;
        return qwasar_session_eval_images(sv->s, tokens, n, images, n_images, err, cap);
    }

    int32_t live = sv->s ? qwasar_session_common_prefix(sv->s, tokens, n) : 0;
    if (live > 0 && live < n) {
        *reused = live;
        return srv_eval_from(sv, tokens, live, n, mk, err, cap);
    }
    if (live == n && n > 0) {
        /* The session is already sitting at the end of this prompt, so its last
         * logits are the ones we want.  Re-evaluating the final token would
         * append a duplicate instead of reproducing the step. */
        *reused = n;
        const float *l = qwasar_session_logits(sv->s);
        if (l) return l;
    }

    if (sv->s) qwasar_session_free(sv->s);
    sv->s = qwasar_session_new(sv->e, err, cap);
    if (!sv->s) return NULL;

    /* A checkpoint is asked to cover at most n-1 tokens.  Restoring the whole
     * prompt would leave the session with no logits and nothing left to
     * evaluate to produce them. */
    int32_t covered = sv->no_cache ? 0 : qwasar_session_restore(sv->s, sv->e, tokens, n - 1);
    *reused = covered;
    sv->ckpt_n = covered;
    return srv_eval_from(sv, tokens, covered, n, mk, err, cap);
}

/* How many leading tokens of `prompt` the first `n_msgs` messages render to
 * under `opts` -- 0 unless that rendering is a proper prefix of it. */
static int32_t srv_prefix_len(server *sv, const qwasar_message *msgs, int32_t n_msgs,
                              const qwasar_chat_options *opts, const int32_t *prompt, int32_t n) {
    char err[256];
    int32_t m = 0;
    int32_t *p = qwasar_apply_chat_template(sv->tok, msgs, n_msgs, opts, &m, err, sizeof err);
    if (!p) return 0;
    const bool ok = m > 0 && m < n && !memcmp(p, prompt, (size_t)m * sizeof *p);
    free(p);
    return ok ? m : 0;
}

typedef void (*delta_fn)(void *ud, bool reasoning, const char *s, size_t n);

typedef struct {
    str     text;
    str     reasoning;
    int32_t n_gen;
    bool    hit_eos;
    bool    hit_stop;      /* ended on one of the request's stop sequences */
    int     stop_index;    /* which one */
    bool    has_call;
} genres;

static void genres_free(genres *g) { str_free(&g->text); str_free(&g->reasoning); }

#define QW_MAX_STOPS 16

typedef enum {
    QW_TOOLS_AUTO,         /* the model decides */
    QW_TOOLS_NONE,         /* no call may start */
    QW_TOOLS_FORCE,        /* the answer is a call, to `force_name` if set */
} tool_mode;

#define QW_MAX_TOOLS 32

typedef struct {
    const char *stops[QW_MAX_STOPS];
    int         n_stops;
    tool_mode   tools;
    const char *force_name;
    /* The request's tool names.  A call forced without a name has its name
     * constrained to one of these, or the model is free to invent one. */
    const char *names[QW_MAX_TOOLS];
    int         n_names;
} genopts;

/* True if `p` (n bytes) could still become "NAME>\n" for one of the tools --
 * the newline included, because the tokenizer often spells ">\n" as one. */
static bool name_prefix_ok(const char *p, size_t n, const genopts *go) {
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
 * byte has not been generated yet.  A token can end partway through a
 * character, and those bytes cannot go into a JSON string on their own: a
 * delta carrying half an emoji is invalid UTF-8, which strict clients reject. */
static size_t utf8_tail(const char *p, size_t n) {
    for (size_t k = 1; k <= 4 && k <= n; k++) {
        const unsigned char c = (unsigned char)p[n - k];
        if ((c & 0xC0) == 0x80) continue;          /* continuation: keep looking */
        const size_t need = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
        return need > k ? k : 0;
    }
    return 0;
}

/* Longest suffix of `p` that begins some stop sequence.  Those bytes may yet
 * turn out to be the stop, and a stop sequence is never shown, so they wait. */
static size_t stop_prefix_tail(const char *p, size_t n, const genopts *go) {
    size_t best = 0;
    for (int s = 0; s < go->n_stops; s++) {
        const size_t sl = strlen(go->stops[s]);
        for (size_t k = sl - 1; k > best; k--)
            if (k <= n && !memcmp(p + n - k, go->stops[s], k)) { best = k; break; }
    }
    return best;
}

/* Earliest stop sequence in p[from..n), as an offset, or -1; *which says
 * which sequence it was. */
static long find_stop(const char *p, size_t n, size_t from, const genopts *go, int *which) {
    long best = -1;
    for (int s = 0; s < go->n_stops; s++) {
        const size_t sl = strlen(go->stops[s]);
        for (size_t i = from; i + sl <= n && (best < 0 || (long)i < best); i++)
            if (!memcmp(p + i, go->stops[s], sl)) { best = (long)i; *which = s; break; }
    }
    return best;
}

/* What one channel -- reasoning or content -- has produced, and how much of it
 * has gone out as deltas.  The gap is what is being held back. */
typedef struct {
    str    shown;
    size_t sent;
} channel;

static void chan_send(channel *ch, bool reasoning, size_t upto, delta_fn fn, void *ud) {
    if (upto <= ch->sent) return;
    if (fn) fn(ud, reasoning, ch->shown.p + ch->sent, upto - ch->sent);
    ch->sent = upto;
}

/* Starts the tool call the request's tool_choice insists on, by evaluating its
 * opening as though the model had written it:
 *
 *     <tool_call>\n<function=NAME>\n
 *
 * or up to "<function=" when any tool will do and the model picks the name.
 * After a reasoning block the model's own next move is "\n\n", so that goes in
 * first; with thinking off the template has already written it. */
static const float *srv_force_call(server *sv, bool after_think, const char *name,
                                   int32_t call_open, genres *out,
                                   char *err, size_t cap) {
    str tail = { 0 };
    str_puts(&tail, "\n<function=");
    if (name) { str_puts(&tail, name); str_puts(&tail, ">\n"); }

    int32_t n_lead = 0, n_tail = 0;
    int32_t *lead = after_think ? qwasar_encode(sv->tok, "\n\n", &n_lead) : NULL;
    int32_t *tl = qwasar_encode(sv->tok, tail.p, &n_tail);
    int32_t *ids = malloc(sizeof *ids * (size_t)(n_lead + 1 + n_tail));
    const float *logits = NULL;
    if (ids && tl) {
        int32_t n = 0;
        for (int32_t i = 0; i < n_lead; i++) ids[n++] = lead[i];
        ids[n++] = call_open;
        for (int32_t i = 0; i < n_tail; i++) ids[n++] = tl[i];
        /* The parser sees the call from its opening tag; the lead is only
         * whitespace between the reasoning and the call. */
        str_puts(&out->text, "<tool_call>");
        str_puts(&out->text, tail.p);
        out->n_gen += n;
        logits = qwasar_session_eval(sv->s, ids, n, err, cap);
    } else {
        snprintf(err, cap, "out of memory");
    }
    free(ids); free(lead); free(tl); str_free(&tail);
    return logits;
}

/* One assistant turn.  Stops at end-of-turn, a stop sequence, a completed tool
 * call, or the token budget.
 *
 * Deltas are not always sent the moment a token arrives.  Content is held back
 * while its tail could still be the start of a stop sequence or of a UTF-8
 * character, and nothing after <tool_call> is content at all: the call is
 * parsed and sent whole once it is complete. */
static bool srv_generate(server *sv, const float *logits, const qwasar_sampling *sp,
                         int32_t max_tokens, bool thinking, const genopts *go,
                         delta_fn on_delta, void *ud,
                         genres *out, char *err, size_t cap) {
    memset(out, 0, sizeof *out);
    const int32_t vocab = qwasar_vocab_size(sv->e);
    const int32_t think_close = qwasar_token_id(sv->tok, "</think>");
    const int32_t call_open = qwasar_token_id(sv->tok, "<tool_call>");
    /* The generation prompt leaves <think> open, so output starts as reasoning
     * and the model closes it.  With thinking disabled the template writes the
     * close itself, so the model never emits one -- and starting in reasoning
     * mode meant every answer came back as reasoning_content with content
     * null, which is a valid-looking response carrying nothing. */
    bool reasoning = thinking;
    bool in_call = false, forced = false, ok = true;
    bool naming = false;          /* choosing the name of a forced call */
    str name = { 0 };
    channel rc = { 0 }, tc = { 0 };

    /* tool_choice "none" is enforced by never letting <tool_call> be sampled.
     * The tools stay in the prompt, so it is identical to an "auto" request's
     * and prefix reuse carries across.  A large negative rather than -INFINITY
     * because the build uses -ffast-math, which assumes no infinities. */
    float *masked = NULL;
    if ((go->tools == QW_TOOLS_NONE && call_open >= 0)
        || (go->tools == QW_TOOLS_FORCE && !go->force_name)) {
        masked = malloc(sizeof *masked * (size_t)vocab);
        if (!masked) { snprintf(err, cap, "out of memory"); return false; }
    }

    for (int32_t i = 0; i < max_tokens; i++) {
        if (go->tools == QW_TOOLS_FORCE && !reasoning && !forced) {
            forced = in_call = true;
            logits = srv_force_call(sv, thinking, go->force_name, call_open, out, err, cap);
            if (!logits) { ok = false; break; }
            naming = !go->force_name;
        }
        const float *lp = logits;
        if (go->tools == QW_TOOLS_NONE && masked) {
            memcpy(masked, logits, sizeof *masked * (size_t)vocab);
            masked[call_open] = -1e30f;
            lp = masked;
        } else if (naming) {
            /* Only tokens that keep the name on the way to a real tool's can
             * be sampled.  A scan of the vocabulary per token, for the few
             * tokens a name takes. */
            memcpy(masked, logits, sizeof *masked * (size_t)vocab);
            char buf[256];
            memcpy(buf, name.p ? name.p : "", name.len);
            for (int32_t t = 0; t < vocab; t++) {
                size_t tl = 0;
                bool sp_tok = false;
                const char *tb = qwasar_token_bytes(sv->tok, t, &tl, &sp_tok);
                if (sp_tok || !tb || !tl || name.len + tl > sizeof buf) { masked[t] = -1e30f; continue; }
                memcpy(buf + name.len, tb, tl);
                if (!name_prefix_ok(buf, name.len + tl, go)) masked[t] = -1e30f;
            }
            lp = masked;
        }
        int32_t next = qwasar_sample(lp, vocab, sp, &sv->rng);
        if (qwasar_is_eos(sv->e, next)) { out->hit_eos = true; break; }
        out->n_gen++;

        size_t len = 0;
        bool special = false;
        const char *bytes = qwasar_token_bytes(sv->tok, next, &len, &special);

        if (naming && bytes && len) {
            str_add(&name, bytes, len);
            if (memchr(name.p, '>', name.len)) naming = false;
        }

        if (next == think_close) {
            reasoning = false;
            chan_send(&rc, true, rc.shown.len, on_delta, ud);
        } else if (bytes && len) {
            str_add(reasoning ? &out->reasoning : &out->text, bytes, len);
            if (next == call_open) {
                /* Whatever preceded the call is final now; the call itself is
                 * never content. */
                in_call = true;
                chan_send(&tc, false, tc.shown.len, on_delta, ud);
            }
            /* Control tokens are structure, not content: they belong in the
             * accumulated text the parser sees, never in a client delta. */
            if (special) {
                /* nothing to show */
            } else if (reasoning) {
                str_add(&rc.shown, bytes, len);
                chan_send(&rc, true, rc.shown.len - utf8_tail(rc.shown.p, rc.shown.len),
                          on_delta, ud);
            } else if (!in_call) {
                str_add(&tc.shown, bytes, len);
                const long at = find_stop(tc.shown.p, tc.shown.len, tc.sent, go,
                                          &out->stop_index);
                if (at >= 0) {
                    /* The stop sequence and everything after it are dropped,
                     * from the deltas and from the final text alike. */
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
            break;
        }

        logits = qwasar_session_eval(sv->s, &next, 1, err, cap);
        if (!logits) { ok = false; break; }
    }

    if (ok) {
        /* A held-back tail is sent now that nothing can complete it -- except
         * half a character, which the budget cut off and which is dropped from
         * the final text too. */
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

/* ---- request shapes -------------------------------------------------------- */

#define QW_MAX_MSGS 256
/* Images per request.  A cap exists because each one is a tower pass and a few
 * hundred kilobytes of rows, and because a request that wants more than this
 * is almost certainly a mistake. */
#define QW_MAX_IMAGES 8

typedef struct {
    qwasar_message msgs[QW_MAX_MSGS];
    int32_t        n;
    str            owned[QW_MAX_MSGS * 3];
    int32_t        n_owned;
    str            tools[32];
    const char    *tool_ptr[32];
    int32_t        n_tools;
    qwasar_image_input images[QW_MAX_IMAGES];
    int32_t        n_images;
} request;

static char *req_own(request *r, str *s) {
    if (r->n_owned >= (int32_t)(sizeof r->owned / sizeof *r->owned)) return (char *)"";
    r->owned[r->n_owned] = *s;
    memset(s, 0, sizeof *s);
    return r->owned[r->n_owned++].p ? r->owned[r->n_owned - 1].p : (char *)"";
}

static void req_free(request *r) {
    for (int32_t i = 0; i < r->n_owned; i++) str_free(&r->owned[i]);
    for (int32_t i = 0; i < r->n_tools; i++) str_free(&r->tools[i]);
    for (int32_t i = 0; i < r->n_images; i++) qwasar_image_release(&r->images[i]);
}

/* Flattens a message `content` field, which both APIs allow to be either a
 * plain string or an array of typed blocks. */
/* ---- images over the wire --------------------------------------------------
 *
 * Both APIs send an image as base64 inside the message content, OpenAI as a
 * data URL under `image_url` and Anthropic as a `source` block.  Neither ever
 * touches the filesystem, so this decodes into memory and hands the bytes
 * straight to the tower.
 *
 * A data URL's payload starts after the comma; a bare base64 string has no
 * comma, and starting at the beginning is the right answer for it. */
static int b64_value(unsigned char ch) {
    if (ch >= 'A' && ch <= 'Z') return ch - 'A';
    if (ch >= 'a' && ch <= 'z') return ch - 'a' + 26;
    if (ch >= '0' && ch <= '9') return ch - '0' + 52;
    if (ch == '+') return 62;
    if (ch == '/') return 63;
    return -1;                      /* padding and whitespace are skipped */
}

static unsigned char *b64_decode(const char *src, size_t n, size_t *out_len) {
    unsigned char *out = malloc(n / 4 * 3 + 4);
    if (!out) return NULL;
    size_t o = 0;
    uint32_t acc = 0;
    int bits = 0;
    for (size_t i = 0; i < n; i++) {
        const int v = b64_value((unsigned char)src[i]);
        if (v < 0) continue;
        acc = (acc << 6) | (uint32_t)v;
        bits += 6;
        if (bits >= 8) { bits -= 8; out[o++] = (unsigned char)(acc >> bits); }
    }
    *out_len = o;
    return out;
}

/* Finds the base64 payload of one content block, whichever API shaped it. */
static bool block_image_data(const qj_doc *d, const qj_node *b,
                             const char **data, size_t *len, char *ext, size_t ecap) {
    if (ext && ecap) ext[0] = 0;
    const qj_node *src = qj_get(d, b, "source");             /* Anthropic */
    const qj_node *n = src ? qj_get(d, src, "data") : NULL;
    if (!n) {                                                /* OpenAI */
        const qj_node *iu = qj_get(d, b, "image_url");
        if (!iu) iu = qj_get(d, b, "video_url");
        if (iu) n = (iu->type == QJ_STRING) ? iu : qj_get(d, iu, "url");
    }
    if (!n || n->type != QJ_STRING) return false;
    const char *p = d->text + n->u.str.off;
    size_t l = n->u.str.len;
    const void *comma = memchr(p, ',', l);
    if (comma) {
        /* "data:video/mp4;base64," -- the subtype is what AVFoundation needs
         * to pick a demuxer, so it is carried through rather than dropped. */
        const char *slash = memchr(p, '/', (size_t)((const char *)comma - p));
        if (slash && ext && ecap) {
            size_t k = 0;
            for (const char *q = slash + 1; q < (const char *)comma && k + 1 < ecap; q++) {
                if (*q == ';') break;
                ext[k++] = *q;
            }
            ext[k] = 0;
        }
        l -= (size_t)((const char *)comma - p) + 1;
        p = (const char *)comma + 1;
    }
    *data = p;
    *len = l;
    return l > 0;
}

/* Encodes every image in a content array, appending to the request's list.
 * Returns the number of <|image_pad|> tokens the message needs. */
static int32_t content_images(server *sv, request *r, const qj_doc *d,
                              const qj_node *n, bool *is_video,
                              char *err, size_t cap) {
    if (!n || n->type != QJ_ARRAY) return 0;
    int32_t tokens = 0;
    int32_t n_img = 0, n_vid = 0;
    for (const qj_node *b = qj_first(d, n); b; b = qj_next(d, b)) {
        const qj_node *t = qj_get(d, b, "type");
        const bool video = qj_str_eq(d, t, "video") || qj_str_eq(d, t, "video_url");
        const bool image = qj_str_eq(d, t, "image") || qj_str_eq(d, t, "image_url");
        if (!video && !image) continue;

        /* One kind per turn.  A message could carry both, but its placeholders
         * are a single run of one token, so mixing them would silently label
         * some of them wrongly -- and the model was trained to tell an image
         * from a video.  Refusing is the honest answer. */
        if (video) n_vid++; else n_img++;
        if (n_img > 0 && n_vid > 0) {
            snprintf(err, cap, "a message may carry images or a video, not both");
            return -1;
        }
        if (r->n_images >= QW_MAX_IMAGES) {
            snprintf(err, cap, "at most %d images per request", QW_MAX_IMAGES);
            return -1;
        }
        const char *data = NULL;
        size_t len = 0;
        char ext[16] = "";
        if (!block_image_data(d, b, &data, &len, ext, sizeof ext)) {
            snprintf(err, cap, "a %s block carries no base64 data",
                     video ? "video" : "image");
            return -1;
        }
        size_t raw_len = 0;
        unsigned char *raw = b64_decode(data, len, &raw_len);
        if (!raw) { snprintf(err, cap, "out of memory"); return -1; }
        const bool ok = video
            ? qwasar_video_encode_memory(sv->e, raw, raw_len, ext,
                                         &r->images[r->n_images], err, cap)
            : qwasar_image_encode_memory(sv->e, raw, raw_len,
                                         &r->images[r->n_images], err, cap);
        free(raw);
        if (!ok) return -1;
        tokens += r->images[r->n_images].n_rows;
        r->n_images++;
    }
    if (is_video) *is_video = n_vid > 0;
    return tokens;
}

static void content_text(const qj_doc *d, const qj_node *n, str *out) {
    if (!n) return;
    if (n->type == QJ_STRING) { str_add(out, d->text + n->u.str.off, n->u.str.len); return; }
    if (n->type != QJ_ARRAY) return;
    for (const qj_node *b = qj_first(d, n); b; b = qj_next(d, b)) {
        const qj_node *t = qj_get(d, b, "type");
        const qj_node *txt = qj_get(d, b, "text");
        if (txt && txt->type == QJ_STRING) {
            if (out->len) str_puts(out, "\n");
            str_add(out, d->text + txt->u.str.off, txt->u.str.len);
        } else if (qj_str_eq(d, t, "tool_result")) {
            const qj_node *c = qj_get(d, b, "content");
            if (out->len) str_puts(out, "\n");
            content_text(d, c, out);
        }
    }
}

/* Writes one <parameter=...> block, keeping JSON-typed values as JSON. */
static void xml_param(str *out, const qj_doc *d, const qj_node *v,
                      const char *key, size_t keylen) {
    str_puts(out, "<parameter=");
    str_add(out, key, keylen);
    str_puts(out, ">\n");
    if (v && v->type == QJ_STRING) str_add(out, d->text + v->u.str.off, v->u.str.len);
    else str_node(out, d, v);
    str_puts(out, "\n</parameter>\n");
}

/* Rebuilds the XML tool call the model originally emitted, from the normalised
 * JSON a client sends back. */
static void xml_from_openai_calls(str *out, const qj_doc *d, const qj_node *calls) {
    for (const qj_node *c = qj_first(d, calls); c; c = qj_next(d, c)) {
        const qj_node *fn = qj_get(d, c, "function");
        if (!fn) fn = c;
        const qj_node *name = qj_get(d, fn, "name");
        const qj_node *args = qj_get(d, fn, "arguments");
        if (!name) continue;

        str_puts(out, "<tool_call>\n<function=");
        str_add(out, d->text + name->u.str.off, name->u.str.len);
        str_puts(out, ">\n");

        /* OpenAI carries arguments as a JSON document inside a string. */
        if (args && args->type == QJ_STRING) {
            qj_doc inner;
            if (qj_parse(&inner, d->text + args->u.str.off, args->u.str.len)) {
                const qj_node *o = qj_root(&inner);
                for (const qj_node *m = qj_first(&inner, o); m; m = qj_next(&inner, m))
                    xml_param(out, &inner, m, inner.text + m->key_off, m->key_len);
            }
            qj_free(&inner);
        } else if (args) {
            for (const qj_node *m = qj_first(d, args); m; m = qj_next(d, m))
                xml_param(out, d, m, d->text + m->key_off, m->key_len);
        }
        str_puts(out, "</function>\n</tool_call>");
    }
}

static void xml_from_anthropic_blocks(str *out, const qj_doc *d, const qj_node *content) {
    if (!content || content->type != QJ_ARRAY) return;
    for (const qj_node *b = qj_first(d, content); b; b = qj_next(d, b)) {
        if (!qj_str_eq(d, qj_get(d, b, "type"), "tool_use")) continue;
        const qj_node *name = qj_get(d, b, "name");
        const qj_node *input = qj_get(d, b, "input");
        if (!name) continue;
        str_puts(out, "<tool_call>\n<function=");
        str_add(out, d->text + name->u.str.off, name->u.str.len);
        str_puts(out, ">\n");
        for (const qj_node *m = qj_first(d, input); m; m = qj_next(d, m))
            xml_param(out, d, m, d->text + m->key_off, m->key_len);
        str_puts(out, "</function>\n</tool_call>");
    }
}

/* ---- building the prompt ---------------------------------------------------- */

static bool collect_openai(server *sv, request *r, const qj_doc *d,
                           const qj_node *root, char *err, size_t cap) {
    const qj_node *tools = qj_get(d, root, "tools");
    for (const qj_node *t = qj_first(d, tools); t && r->n_tools < 32; t = qj_next(d, t)) {
        str j = { 0 };
        str_node(&j, d, t);   /* already the shape the template wants */
        r->tools[r->n_tools] = j;
        r->tool_ptr[r->n_tools] = j.p;
        r->n_tools++;
    }

    const qj_node *msgs = qj_get(d, root, "messages");
    for (const qj_node *m = qj_first(d, msgs); m && r->n < QW_MAX_MSGS; m = qj_next(d, m)) {
        const qj_node *role = qj_get(d, m, "role");
        if (!role || role->type != QJ_STRING) continue;

        str content = { 0 };
        const qj_node *cnode = qj_get(d, m, "content");
        content_text(d, cnode, &content);
        bool is_video = false;
        const int32_t img_tokens = content_images(sv, r, d, cnode, &is_video, err, cap);
        if (img_tokens < 0) { str_free(&content); return false; }

        const char *rname = "user";
        if (qj_str_eq(d, role, "system")) rname = "system";
        else if (qj_str_eq(d, role, "assistant")) rname = "assistant";
        else if (qj_str_eq(d, role, "tool")) rname = "tool";

        str calls = { 0 };
        const qj_node *tc = qj_get(d, m, "tool_calls");
        if (tc && tc->type == QJ_ARRAY) xml_from_openai_calls(&calls, d, tc);

        /* Reasoning is carried back when the client keeps it.  Without it the
         * replayed assistant turn cannot match what the session actually
         * generated, and prefix reuse is lost for the whole conversation. */
        str reasoning = { 0 };
        const qj_node *rc = qj_get(d, m, "reasoning_content");
        if (rc && rc->type == QJ_STRING) str_add(&reasoning, d->text + rc->u.str.off,
                                                 rc->u.str.len);

        r->msgs[r->n].role = rname;
        r->msgs[r->n].content = req_own(r, &content);
        r->msgs[r->n].n_image_tokens = img_tokens;
        r->msgs[r->n].vision_is_video = is_video;
        r->msgs[r->n].reasoning = reasoning.p ? req_own(r, &reasoning) : NULL;
        str_free(&reasoning);
        r->msgs[r->n].tool_calls = calls.p ? req_own(r, &calls) : NULL;
        str_free(&calls);
        r->n++;
    }
    return true;
}

static bool collect_anthropic(server *sv, request *r, const qj_doc *d,
                              const qj_node *root, char *err, size_t cap) {
    /* Anthropic tool schemas name the schema differently; rewrap them into the
     * shape the model's template documents. */
    const qj_node *tools = qj_get(d, root, "tools");
    for (const qj_node *t = qj_first(d, tools); t && r->n_tools < 32; t = qj_next(d, t)) {
        const qj_node *name = qj_get(d, t, "name");
        const qj_node *desc = qj_get(d, t, "description");
        const qj_node *schema = qj_get(d, t, "input_schema");
        if (!name) continue;

        str j = { 0 };
        str_puts(&j, "{\"type\": \"function\", \"function\": {\"name\": ");
        str_node(&j, d, name);
        if (desc) { str_puts(&j, ", \"description\": "); str_node(&j, d, desc); }
        str_puts(&j, ", \"parameters\": ");
        if (schema) str_node(&j, d, schema);
        else str_puts(&j, "{\"type\": \"object\", \"properties\": {}}");
        str_puts(&j, "}}");
        r->tools[r->n_tools] = j;
        r->tool_ptr[r->n_tools] = j.p;
        r->n_tools++;
    }

    const qj_node *sys = qj_get(d, root, "system");
    if (sys) {
        str content = { 0 };
        content_text(d, sys, &content);
        if (content.len) {
            r->msgs[r->n].role = "system";
            r->msgs[r->n].content = req_own(r, &content);
            r->msgs[r->n].reasoning = NULL;
            r->msgs[r->n].tool_calls = NULL;
            r->n++;
        }
        str_free(&content);
    }

    const qj_node *msgs = qj_get(d, root, "messages");
    for (const qj_node *m = qj_first(d, msgs); m && r->n < QW_MAX_MSGS; m = qj_next(d, m)) {
        const qj_node *role = qj_get(d, m, "role");
        const qj_node *content = qj_get(d, m, "content");
        const bool assistant = qj_str_eq(d, role, "assistant");

        /* A user turn carrying tool_result blocks is a tool response, which the
         * model's template renders as its own kind of turn. */
        bool is_tool_result = false;
        if (!assistant && content && content->type == QJ_ARRAY)
            for (const qj_node *b = qj_first(d, content); b; b = qj_next(d, b))
                if (qj_str_eq(d, qj_get(d, b, "type"), "tool_result")) is_tool_result = true;

        str text = { 0 };
        content_text(d, content, &text);

        str calls = { 0 };
        if (assistant) xml_from_anthropic_blocks(&calls, d, content);

        /* Anthropic keeps reasoning as its own block type; carrying it back is
         * what lets a replayed turn match the session and keep prefix reuse. */
        str reasoning = { 0 };
        if (assistant && content && content->type == QJ_ARRAY)
            for (const qj_node *b = qj_first(d, content); b; b = qj_next(d, b)) {
                if (!qj_str_eq(d, qj_get(d, b, "type"), "thinking")) continue;
                const qj_node *th = qj_get(d, b, "thinking");
                if (th && th->type == QJ_STRING)
                    str_add(&reasoning, d->text + th->u.str.off, th->u.str.len);
            }

        /* Only a user turn can carry an image; an assistant turn replaying one
         * would double-count its pad tokens. */
        bool is_video = false;
        const int32_t img_tokens = assistant ? 0
                                 : content_images(sv, r, d, content, &is_video, err, cap);
        if (img_tokens < 0) {
            str_free(&text); str_free(&reasoning); str_free(&calls);
            return false;
        }

        r->msgs[r->n].role = assistant ? "assistant" : (is_tool_result ? "tool" : "user");
        r->msgs[r->n].content = req_own(r, &text);
        r->msgs[r->n].n_image_tokens = img_tokens;
        r->msgs[r->n].vision_is_video = is_video;
        r->msgs[r->n].reasoning = reasoning.p ? req_own(r, &reasoning) : NULL;
        str_free(&reasoning);
        r->msgs[r->n].tool_calls = calls.p ? req_own(r, &calls) : NULL;
        str_free(&calls);
        r->n++;
    }
    return true;
}

/* Sampling knobs, with the model's own generation_config as the floor.  A knob
 * set explicitly in the request always wins, so temperature=0 is greedy all the
 * way through and a benchmark harness gets deterministic output. */
static void read_sampling(qwasar_sampling *sp, const qj_doc *d, const qj_node *root) {
    qwasar_sampling_defaults(sp);
    const qj_node *n;
    if ((n = qj_get(d, root, "temperature")) && n->type == QJ_NUMBER) sp->temperature = (float)n->u.num;
    if ((n = qj_get(d, root, "top_p")) && n->type == QJ_NUMBER) sp->top_p = (float)n->u.num;
    if ((n = qj_get(d, root, "top_k")) && n->type == QJ_NUMBER) sp->top_k = (int32_t)n->u.num;
    if ((n = qj_get(d, root, "min_p")) && n->type == QJ_NUMBER) sp->min_p = (float)n->u.num;
    if ((n = qj_get(d, root, "seed")) && n->type == QJ_NUMBER) sp->seed = (uint64_t)n->u.num;
}

static int32_t read_max_tokens(const qj_doc *d, const qj_node *root, int32_t dflt) {
    const char *keys[] = { "max_tokens", "max_completion_tokens", "max_output_tokens" };
    for (size_t i = 0; i < sizeof keys / sizeof *keys; i++) {
        const qj_node *n = qj_get(d, root, keys[i]);
        if (n && n->type == QJ_NUMBER && n->u.num > 0) return (int32_t)n->u.num;
    }
    return dflt;
}

/* The request options both APIs define beyond sampling -- stop sequences and
 * tool choice -- with the storage the genopts pointers refer to. */
typedef struct {
    genopts go;
    char    stop_store[QW_MAX_STOPS][128];
    char    name_store[128];
    char    names_store[QW_MAX_TOOLS][128];
    bool    include_usage;      /* stream_options.include_usage */
} req_opts;

/* Copies a JSON string into `dst`, false if it is not a string or too long. */
static bool copy_str(const qj_doc *d, const qj_node *n, char *dst, size_t cap) {
    if (!n || n->type != QJ_STRING || qj_strlen(n) >= cap) return false;
    memcpy(dst, qj_str(d, n), qj_strlen(n));
    dst[qj_strlen(n)] = 0;
    return true;
}

static bool request_has_tool(const qj_doc *d, const qj_node *root, const char *name) {
    const qj_node *tools = qj_get(d, root, "tools");
    for (const qj_node *t = qj_first(d, tools); t; t = qj_next(d, t)) {
        const qj_node *fn = qj_get(d, t, "function");
        if (qj_str_eq(d, qj_get(d, fn ? fn : t, "name"), name)) return true;
    }
    return false;
}

/* Collects the request's tool names, whichever API shaped the tools. */
static void read_tool_names(const qj_doc *d, const qj_node *root, req_opts *o) {
    const qj_node *tools = qj_get(d, root, "tools");
    for (const qj_node *t = qj_first(d, tools); t && o->go.n_names < QW_MAX_TOOLS;
         t = qj_next(d, t)) {
        const qj_node *fn = qj_get(d, t, "function");
        char *slot = o->names_store[o->go.n_names];
        if (copy_str(d, qj_get(d, fn ? fn : t, "name"), slot, sizeof o->names_store[0]) && *slot)
            o->go.names[o->go.n_names++] = slot;
    }
}

/* Reads a stop-sequence parameter: one string, or an array of them. */
static bool read_stops(const qj_doc *d, const qj_node *n, req_opts *o,
                       char *msg, size_t cap) {
    if (!n || n->type == QJ_NULL) return true;
    const qj_node *one = n->type == QJ_STRING ? n : NULL;
    if (!one && n->type != QJ_ARRAY) {
        snprintf(msg, cap, "stop sequences must be a string or an array of strings");
        return false;
    }
    for (const qj_node *s = one ? one : qj_first(d, n); s; s = one ? NULL : qj_next(d, s)) {
        if (o->go.n_stops >= QW_MAX_STOPS
            || !copy_str(d, s, o->stop_store[o->go.n_stops], sizeof o->stop_store[0])) {
            snprintf(msg, cap, "at most %d stop sequences of at most %zu bytes",
                     QW_MAX_STOPS, sizeof o->stop_store[0] - 1);
            return false;
        }
        if (o->stop_store[o->go.n_stops][0])           /* "" would stop at once */
            o->go.stops[o->go.n_stops] = o->stop_store[o->go.n_stops], o->go.n_stops++;
    }
    return true;
}

/* Anthropic's stop_sequences and tool_choice ({"type": "auto" | "any" |
 * "tool" | "none", "name"}).  Same contract as read_openai_opts. */
static bool read_anthropic_opts(const qj_doc *d, const qj_node *root, req_opts *o,
                                const char **param, char *msg, size_t cap) {
    memset(o, 0, sizeof *o);
    read_tool_names(d, root, o);
    if (!read_stops(d, qj_get(d, root, "stop_sequences"), o, msg, cap)) {
        *param = "stop_sequences";
        return false;
    }
    const qj_node *n = qj_get(d, root, "tool_choice");
    if (!n || n->type == QJ_NULL) return true;
    *param = "tool_choice";
    const qj_node *t = qj_get(d, n, "type");
    if (qj_str_eq(d, t, "auto"))      o->go.tools = QW_TOOLS_AUTO;
    else if (qj_str_eq(d, t, "none")) o->go.tools = QW_TOOLS_NONE;
    else if (qj_str_eq(d, t, "any"))  o->go.tools = QW_TOOLS_FORCE;
    else if (qj_str_eq(d, t, "tool")) {
        if (!copy_str(d, qj_get(d, n, "name"), o->name_store, sizeof o->name_store)) {
            snprintf(msg, cap, "tool_choice of type tool needs a name");
            return false;
        }
        if (!request_has_tool(d, root, o->name_store)) {
            snprintf(msg, cap, "tool_choice names '%s', which is not in tools", o->name_store);
            return false;
        }
        o->go.tools = QW_TOOLS_FORCE;
        o->go.force_name = o->name_store;
    } else {
        snprintf(msg, cap, "tool_choice.type must be auto, any, tool or none");
        return false;
    }
    if (o->go.tools == QW_TOOLS_FORCE && o->go.n_names == 0) {
        snprintf(msg, cap, "tool_choice requires a tool call but no tools were given");
        return false;
    }
    *param = NULL;
    return true;
}

/* Reads n, stop, tool_choice and stream_options.  On a request this server
 * cannot honour, returns false with the offending parameter in *param and the
 * reason in msg -- refusing is better than silently doing something else. */
static bool read_openai_opts(const qj_doc *d, const qj_node *root, req_opts *o,
                             const char **param, char *msg, size_t cap) {
    memset(o, 0, sizeof *o);
    const qj_node *n;

    /* One session means one continuation.  Producing n of them would take n
     * full prefills, since the session cannot be forked. */
    if ((n = qj_get(d, root, "n")) && n->type != QJ_NULL
        && !(n->type == QJ_NUMBER && n->u.num == 1)) {
        *param = "n";
        snprintf(msg, cap, "only n=1 is supported");
        return false;
    }

    if (!read_stops(d, qj_get(d, root, "stop"), o, msg, cap)) {
        *param = "stop";
        return false;
    }

    const qj_node *tools = qj_get(d, root, "tools");
    const bool have_tools = tools && tools->type == QJ_ARRAY && qj_count(tools) > 0;
    read_tool_names(d, root, o);
    if ((n = qj_get(d, root, "tool_choice")) && n->type != QJ_NULL) {
        *param = "tool_choice";
        if (qj_str_eq(d, n, "auto")) {
            o->go.tools = QW_TOOLS_AUTO;
        } else if (qj_str_eq(d, n, "none")) {
            o->go.tools = QW_TOOLS_NONE;
        } else if (qj_str_eq(d, n, "required")) {
            o->go.tools = QW_TOOLS_FORCE;
        } else if (n->type == QJ_OBJECT && qj_str_eq(d, qj_get(d, n, "type"), "function")) {
            if (!copy_str(d, qj_path(d, n, "function.name"), o->name_store, sizeof o->name_store)) {
                snprintf(msg, cap, "tool_choice names no function");
                return false;
            }
            if (!request_has_tool(d, root, o->name_store)) {
                snprintf(msg, cap, "tool_choice names '%s', which is not in tools", o->name_store);
                return false;
            }
            o->go.tools = QW_TOOLS_FORCE;
            o->go.force_name = o->name_store;
        } else {
            snprintf(msg, cap, "tool_choice must be \"none\", \"auto\", \"required\" "
                               "or a function");
            return false;
        }
        if (o->go.tools == QW_TOOLS_FORCE && !have_tools) {
            snprintf(msg, cap, "tool_choice requires a tool call but no tools were given");
            return false;
        }
        *param = NULL;
    }

    o->include_usage = qj_bool_or(d, root, "stream_options.include_usage", false);
    return true;
}

/* ---- response building ------------------------------------------------------ */

/* Tool arguments arrive from the model as text.  A value that is valid JSON on
 * its own is emitted as JSON so numbers and booleans survive the round trip;
 * anything else is emitted as a string, which is what it is. */
static void emit_arg_value(str *out, const char *v) {
    qj_doc probe;
    if (v && *v && qj_parse(&probe, v, strlen(v))) {
        const qj_node *r = qj_root(&probe);
        if (r->type == QJ_NUMBER || r->type == QJ_TRUE || r->type == QJ_FALSE
            || r->type == QJ_OBJECT || r->type == QJ_ARRAY) {
            str_puts(out, v);
            qj_free(&probe);
            return;
        }
    }
    qj_free(&probe);
    str_jsons(out, v);
}

static void emit_args_object(str *out, const qw_tool_call *c) {
    str_puts(out, "{");
    for (int i = 0; i < c->n_params; i++) {
        if (i) str_puts(out, ", ");
        str_jsons(out, c->params[i].key);
        str_puts(out, ": ");
        emit_arg_value(out, c->params[i].value);
    }
    str_puts(out, "}");
}

static void gen_id(char *out, size_t cap, const char *prefix) {
    static uint64_t counter;
    const uint64_t k = __atomic_add_fetch(&counter, 1, __ATOMIC_RELAXED);
    snprintf(out, cap, "%s%08llx%04llx", prefix,
             (unsigned long long)time(NULL), (unsigned long long)(k & 0xffff));
}

/* ---- streaming state -------------------------------------------------------- */

typedef struct {
    conn       *c;
    const char *id;
    long        created;
    bool        anthropic;
    /* Anthropic frames content as indexed blocks that must be opened and
     * closed, so a switch between thinking and text is a block boundary. */
    int         index;
    bool        open;
    bool        open_is_thinking;
    uint64_t    sig;              /* running hash of the open thinking block */
} stream_ctx;

/* A thinking block's signature.  Anthropic's is an opaque token clients must
 * hand back unchanged; nothing here verifies one, but the field is required,
 * so it carries a digest of the text -- stable, and different per block. */
#define QW_FNV_INIT 0xcbf29ce484222325ull
static uint64_t fnv1a(uint64_t h, const char *s, size_t n) {
    for (size_t i = 0; i < n; i++) { h ^= (unsigned char)s[i]; h *= 0x100000001b3ull; }
    return h;
}

static void oai_delta(stream_ctx *st, bool reasoning, const char *s, size_t n) {
    str b = { 0 };
    str_printf(&b, "{\"id\": \"%s\", \"object\": \"chat.completion.chunk\", "
                   "\"created\": %ld, \"model\": \"%s\", \"choices\": "
                   "[{\"index\": 0, \"delta\": {",
               st->id, st->created, model_id);
    str_puts(&b, reasoning ? "\"reasoning_content\": " : "\"content\": ");
    str_json(&b, s, n);
    str_puts(&b, "}, \"finish_reason\": null}]}");
    sse_event(st->c, NULL, b.p);
    str_free(&b);
}

static void ant_block_close(stream_ctx *st) {
    if (!st->open) return;
    str b = { 0 };
    if (st->open_is_thinking) {
        /* The signature arrives just before the block closes. */
        str_printf(&b, "{\"type\": \"content_block_delta\", \"index\": %d, \"delta\": "
                       "{\"type\": \"signature_delta\", \"signature\": \"%016llx\"}}",
                   st->index, (unsigned long long)st->sig);
        sse_event(st->c, "content_block_delta", b.p);
        b.len = 0;
    }
    str_printf(&b, "{\"type\": \"content_block_stop\", \"index\": %d}", st->index);
    sse_event(st->c, "content_block_stop", b.p);
    str_free(&b);
    st->open = false;
    st->index++;
}

static void ant_block_open(stream_ctx *st, bool thinking) {
    str b = { 0 };
    str_printf(&b, "{\"type\": \"content_block_start\", \"index\": %d, "
                   "\"content_block\": {\"type\": \"%s\", \"%s\": \"\"%s}}",
               st->index, thinking ? "thinking" : "text", thinking ? "thinking" : "text",
               thinking ? ", \"signature\": \"\"" : "");
    st->sig = QW_FNV_INIT;
    sse_event(st->c, "content_block_start", b.p);
    str_free(&b);
    st->open = true;
    st->open_is_thinking = thinking;
}

static void ant_delta(stream_ctx *st, bool reasoning, const char *s, size_t n) {
    if (!st->open || st->open_is_thinking != reasoning) {
        ant_block_close(st);
        ant_block_open(st, reasoning);
    }
    if (reasoning) st->sig = fnv1a(st->sig, s, n);
    str b = { 0 };
    str_printf(&b, "{\"type\": \"content_block_delta\", \"index\": %d, \"delta\": "
                   "{\"type\": \"%s\", \"%s\": ",
               st->index, reasoning ? "thinking_delta" : "text_delta",
               reasoning ? "thinking" : "text");
    str_json(&b, s, n);
    str_puts(&b, "}}");
    sse_event(st->c, "content_block_delta", b.p);
    str_free(&b);
}

static void on_delta(void *ud, bool reasoning, const char *s, size_t n) {
    stream_ctx *st = ud;
    if (st->c->dead) return;
    if (st->anthropic) ant_delta(st, reasoning, s, n);
    else               oai_delta(st, reasoning, s, n);
}

/* ---- endpoints -------------------------------------------------------------- */

/* A completion, or with `count_only` just the size of its prompt --
 * Anthropic's /v1/messages/count_tokens, which renders exactly what a
 * completion would and stops there. */
static void handle_completion(server *sv, conn *c, const qj_doc *d, bool anthropic,
                              bool count_only) {
    const qj_node *root = qj_root(d);

    request req;
    memset(&req, 0, sizeof req);
    char cerr[256] = "";
    const bool collected = anthropic
        ? collect_anthropic(sv, &req, d, root, cerr, sizeof cerr)
        : collect_openai(sv, &req, d, root, cerr, sizeof cerr);
    if (!collected) {
        http_error(c, 400, "Bad Request", cerr[0] ? cerr : "cannot read the request");
        req_free(&req);
        return;
    }
    if (req.n == 0) {
        http_error(c, 400, "Bad Request", "no messages");
        req_free(&req);
        return;
    }

    req_opts oo;
    const char *param = NULL;
    if (!(anthropic ? read_anthropic_opts(d, root, &oo, &param, cerr, sizeof cerr)
                    : read_openai_opts(d, root, &oo, &param, cerr, sizeof cerr))) {
        http_error_at(c, 400, "Bad Request", cerr, param, NULL);
        req_free(&req);
        return;
    }

    /* An Anthropic request ending in an assistant turn is a prefill: the reply
     * continues that turn instead of starting a new one. */
    const bool prefill = anthropic && !strcmp(req.msgs[req.n - 1].role, "assistant");

    qwasar_sampling sp;
    read_sampling(&sp, d, root);
    sv->rng = sp.seed ? sp.seed : (uint64_t)time(NULL) * 6364136223846793005ull + 1;

    int32_t max_tokens = read_max_tokens(d, root, sv->max_tokens);
    const qj_node *stream_n = qj_get(d, root, "stream");
    const bool stream = stream_n && stream_n->type == QJ_TRUE;

    /* Thinking is on unless a client turns it off.  Anthropic clients express
     * that as thinking.type = "disabled"; OpenAI ones have no standard field,
     * so an explicit enable_thinking is honoured as an extension. */
    bool thinking = true;
    const qj_node *th = qj_get(d, root, "thinking");
    if (th && qj_str_eq(d, qj_get(d, th, "type"), "disabled")) thinking = false;
    const qj_node *et = qj_get(d, root, "enable_thinking");
    if (et && et->type == QJ_FALSE) thinking = false;

    qwasar_chat_options chat = {
        .enable_thinking = thinking,
        .reasoning_effort = "xhigh",
        .add_generation_prompt = true,
        .tools = req.n_tools ? req.tool_ptr : NULL,
        .n_tools = req.n_tools,
        .continue_final_message = prefill,
    };
    char rerr[256];
    if (qj_str_copy(d, root, "reasoning_effort", rerr, sizeof rerr)
        && (!strcmp(rerr, "low") || !strcmp(rerr, "medium") || !strcmp(rerr, "xhigh")))
        chat.reasoning_effort = !strcmp(rerr, "low") ? "low"
                              : !strcmp(rerr, "medium") ? "medium" : "xhigh";

    char err[512] = "";
    int32_t n_prompt = 0;
    int32_t *prompt = qwasar_apply_chat_template(sv->tok, req.msgs, req.n, &chat,
                                                 &n_prompt, err, sizeof err);
    if (!prompt) { req_free(&req); http_error(c, 400, "Bad Request", err); return; }

    if (count_only) {
        char body[64];
        const int bl = snprintf(body, sizeof body, "{\"input_tokens\": %d}", n_prompt);
        http_send(c, 200, "OK", "application/json", body, (size_t)bl);
        free(prompt);
        req_free(&req);
        return;
    }

    /* Output fits in what the context has left: past it the cache has no row
     * for the next token, and generation would fail partway through a
     * response rather than end it with a length stop. */
    const int32_t room = sv->ctx - n_prompt;
    if (max_tokens <= 0 || max_tokens > room) max_tokens = room;

    if (n_prompt >= sv->ctx) {
        free(prompt);
        req_free(&req);
        http_error(c, 400, "Bad Request", "prompt exceeds the server's context size");
        return;
    }

    /* Checkpoint boundaries: the prompt rendered only as far as the system
     * messages (with the tools, which the template puts there), and as far as
     * the last complete turn.  Used only where the rendering really is a
     * prefix of the prompt. */
    ckpt_marks marks = { 0, 0 };
    if (!sv->no_cache && req.n_images == 0) {
        qwasar_chat_options upto = chat;
        upto.add_generation_prompt = false;
        upto.continue_final_message = false;
        int32_t n_sys = 0;
        while (n_sys < req.n && req.msgs[n_sys].role && !strcmp(req.msgs[n_sys].role, "system")) n_sys++;
        if (n_sys > 0) marks.sys_n = srv_prefix_len(sv, req.msgs, n_sys, &upto, prompt, n_prompt);
        if (!prefill) marks.hist_n = srv_prefix_len(sv, req.msgs, req.n, &upto, prompt, n_prompt);
    }

    int32_t reused = 0;
    /* The request outlives the prompt now: rendering turns its text into
     * tokens, but its image rows are what the prefill scatters in, so freeing
     * it here -- which is where it used to happen -- released them one call
     * before they were read. */
    const float *logits = srv_prefill(sv, prompt, n_prompt, req.images, req.n_images,
                                      &marks, &reused, err, sizeof err);
    req_free(&req);
    free(prompt);
    if (!logits) { http_error(c, 500, "Internal Server Error", err); return; }
    if (sv->verbose)
        fprintf(stderr, "  prompt %d tokens (%d reused)\n", n_prompt, reused);

    char id[64];
    gen_id(id, sizeof id, anthropic ? "msg_" : "chatcmpl-");
    const long created = (long)time(NULL);

    stream_ctx st = { .c = c, .id = id, .created = created, .anthropic = anthropic };

    if (stream) {
        sse_begin(c);
        if (anthropic) {
            str b = { 0 };
            str_printf(&b, "{\"type\": \"message_start\", \"message\": {\"id\": \"%s\", "
                           "\"type\": \"message\", \"role\": \"assistant\", \"model\": \"%s\", "
                           "\"content\": [], \"stop_reason\": null, \"stop_sequence\": null, "
                           "\"usage\": {\"input_tokens\": %d, \"output_tokens\": 0}}}",
                       id, model_id, n_prompt);
            sse_event(c, "message_start", b.p);
            str_free(&b);
        } else {
            str b = { 0 };
            str_printf(&b, "{\"id\": \"%s\", \"object\": \"chat.completion.chunk\", "
                           "\"created\": %ld, \"model\": \"%s\", \"choices\": [{\"index\": 0, "
                           "\"delta\": {\"role\": \"assistant\"}, \"finish_reason\": null}]}",
                       id, created, model_id);
            sse_event(c, NULL, b.p);
            str_free(&b);
        }
    }

    genres g;
    /* A prefilled turn already has its reasoning block closed, so the
     * continuation starts in the answer. */
    bool ok = srv_generate(sv, logits, &sp, max_tokens, thinking && !prefill, &oo.go,
                           stream ? on_delta : NULL, &st, &g, err, sizeof err);
    if (!ok) {
        if (!stream) {
            http_error(c, 500, "Internal Server Error", err);
        } else {
            /* Headers are gone, so the failure travels in the stream: an
             * `error` event for Anthropic, an error object for OpenAI. */
            str b = { 0 };
            if (anthropic) {
                ant_block_close(&st);
                anthropic_error_body(&b, 500, err);
                sse_event(c, "error", b.p);
            } else {
                str_puts(&b, "{\"error\": {\"message\": ");
                str_jsons(&b, err);
                str_puts(&b, ", \"type\": \"server_error\", \"param\": null, \"code\": null}}");
                sse_event(c, NULL, b.p);
            }
            str_free(&b);
            sse_end(c);
        }
        genres_free(&g);
        return;
    }

    /* Tool calls are recognised only once the block is complete, so they are
     * emitted whole rather than streamed argument by argument. */
    qw_tool_calls calls;
    memset(&calls, 0, sizeof calls);
    int n_calls = 0;
    if (g.has_call) {
        char perr[256];
        n_calls = qw_tool_parse(g.text.p ? g.text.p : "", &calls, perr, sizeof perr);
        if (n_calls < 0) n_calls = 0;
    }
    const char *visible = (n_calls > 0 && calls.preamble) ? calls.preamble
                        : (g.has_call ? "" : (g.text.p ? g.text.p : ""));
    const char *finish = n_calls > 0 ? "tool_calls"
                       : (g.hit_eos || g.hit_stop) ? "stop" : "length";
    const char *stop_reason = n_calls > 0 ? "tool_use"
                            : g.hit_stop ? "stop_sequence"
                            : g.hit_eos ? "end_turn" : "max_tokens";
    /* Anthropic names the sequence that matched, or null. */
    str stop_seq = { 0 };
    if (g.hit_stop) str_jsons(&stop_seq, oo.go.stops[g.stop_index]);
    else str_puts(&stop_seq, "null");

    if (stream) {
        if (anthropic) {
            ant_block_close(&st);
            for (int i = 0; i < n_calls; i++) {
                char tid[64];
                gen_id(tid, sizeof tid, "toolu_");
                str b = { 0 };
                str_printf(&b, "{\"type\": \"content_block_start\", \"index\": %d, "
                               "\"content_block\": {\"type\": \"tool_use\", \"id\": \"%s\", "
                               "\"name\": ", st.index, tid);
                str_jsons(&b, calls.calls[i].name);
                str_puts(&b, ", \"input\": {}}}");
                sse_event(c, "content_block_start", b.p);
                str_free(&b);

                str args = { 0 };
                emit_args_object(&args, &calls.calls[i]);
                str db = { 0 };
                str_printf(&db, "{\"type\": \"content_block_delta\", \"index\": %d, "
                                "\"delta\": {\"type\": \"input_json_delta\", "
                                "\"partial_json\": ", st.index);
                str_jsons(&db, args.p ? args.p : "{}");
                str_puts(&db, "}}");
                sse_event(c, "content_block_delta", db.p);
                str_free(&db);
                str_free(&args);

                str e = { 0 };
                str_printf(&e, "{\"type\": \"content_block_stop\", \"index\": %d}", st.index);
                sse_event(c, "content_block_stop", e.p);
                str_free(&e);
                st.index++;
            }
            str b = { 0 };
            str_printf(&b, "{\"type\": \"message_delta\", \"delta\": {\"stop_reason\": \"%s\", "
                           "\"stop_sequence\": %s}, \"usage\": {\"output_tokens\": %d}}",
                       stop_reason, stop_seq.p, g.n_gen);
            sse_event(c, "message_delta", b.p);
            str_free(&b);
            sse_event(c, "message_stop", "{\"type\": \"message_stop\"}");
        } else {
            for (int i = 0; i < n_calls; i++) {
                char tid[64];
                gen_id(tid, sizeof tid, "call_");
                str args = { 0 };
                emit_args_object(&args, &calls.calls[i]);
                str b = { 0 };
                str_printf(&b, "{\"id\": \"%s\", \"object\": \"chat.completion.chunk\", "
                               "\"created\": %ld, \"model\": \"%s\", \"choices\": [{\"index\": 0, "
                               "\"delta\": {\"tool_calls\": [{\"index\": %d, \"id\": \"%s\", "
                               "\"type\": \"function\", \"function\": {\"name\": ",
                           id, created, model_id, i, tid);
                str_jsons(&b, calls.calls[i].name);
                str_puts(&b, ", \"arguments\": ");
                str_jsons(&b, args.p ? args.p : "{}");
                str_puts(&b, "}}]}, \"finish_reason\": null}]}");
                sse_event(c, NULL, b.p);
                str_free(&b);
                str_free(&args);
            }
            /* Usage rides on the finishing chunk by default, which clients
             * written against other local servers expect.  A client that asks
             * with stream_options.include_usage gets the spec's shape instead:
             * a chunk of its own, with no choices, just before [DONE]. */
            str b = { 0 };
            str_printf(&b, "{\"id\": \"%s\", \"object\": \"chat.completion.chunk\", "
                           "\"created\": %ld, \"model\": \"%s\", \"choices\": [{\"index\": 0, "
                           "\"delta\": {}, \"finish_reason\": \"%s\"}]",
                       id, created, model_id, finish);
            if (oo.include_usage) {
                str_puts(&b, "}");
                sse_event(c, NULL, b.p);
                b.len = 0;
                str_printf(&b, "{\"id\": \"%s\", \"object\": \"chat.completion.chunk\", "
                               "\"created\": %ld, \"model\": \"%s\", \"choices\": []",
                           id, created, model_id);
            }
            str_printf(&b, ", \"usage\": {\"prompt_tokens\": %d, \"completion_tokens\": %d, "
                           "\"total_tokens\": %d}}",
                       n_prompt, g.n_gen, n_prompt + g.n_gen);
            sse_event(c, NULL, b.p);
            str_free(&b);
            sse_event(c, NULL, "[DONE]");
        }
        sse_end(c);
        qw_tool_calls_free(&calls);
        genres_free(&g);
        str_free(&stop_seq);
        return;
    }

    str b = { 0 };
    if (anthropic) {
        str_printf(&b, "{\"id\": \"%s\", \"type\": \"message\", \"role\": \"assistant\", "
                       "\"model\": \"%s\", \"content\": [", id, model_id);
        bool first = true;
        if (g.reasoning.len) {
            str_puts(&b, "{\"type\": \"thinking\", \"thinking\": ");
            str_jsons(&b, g.reasoning.p);
            str_printf(&b, ", \"signature\": \"%016llx\"}",
                       (unsigned long long)fnv1a(QW_FNV_INIT, g.reasoning.p, g.reasoning.len));
            first = false;
        }
        if (visible && *visible) {
            if (!first) str_puts(&b, ", ");
            str_puts(&b, "{\"type\": \"text\", \"text\": ");
            str_jsons(&b, visible);
            str_puts(&b, "}");
            first = false;
        }
        for (int i = 0; i < n_calls; i++) {
            char tid[64];
            gen_id(tid, sizeof tid, "toolu_");
            if (!first) str_puts(&b, ", ");
            str_printf(&b, "{\"type\": \"tool_use\", \"id\": \"%s\", \"name\": ", tid);
            str_jsons(&b, calls.calls[i].name);
            str_puts(&b, ", \"input\": ");
            emit_args_object(&b, &calls.calls[i]);
            str_puts(&b, "}");
            first = false;
        }
        str_printf(&b, "], \"stop_reason\": \"%s\", \"stop_sequence\": %s, "
                       "\"usage\": {\"input_tokens\": %d, \"output_tokens\": %d}}",
                   stop_reason, stop_seq.p, n_prompt, g.n_gen);
    } else {
        str_printf(&b, "{\"id\": \"%s\", \"object\": \"chat.completion\", \"created\": %ld, "
                       "\"model\": \"%s\", \"choices\": [{\"index\": 0, \"message\": "
                       "{\"role\": \"assistant\", \"refusal\": null, \"content\": ",
                   id, created, model_id);
        if (visible && *visible) str_jsons(&b, visible);
        else str_puts(&b, "null");
        if (g.reasoning.len) {
            str_puts(&b, ", \"reasoning_content\": ");
            str_jsons(&b, g.reasoning.p);
        }
        if (n_calls > 0) {
            str_puts(&b, ", \"tool_calls\": [");
            for (int i = 0; i < n_calls; i++) {
                char tid[64];
                gen_id(tid, sizeof tid, "call_");
                str args = { 0 };
                emit_args_object(&args, &calls.calls[i]);
                if (i) str_puts(&b, ", ");
                str_printf(&b, "{\"id\": \"%s\", \"type\": \"function\", \"function\": "
                               "{\"name\": ", tid);
                str_jsons(&b, calls.calls[i].name);
                str_puts(&b, ", \"arguments\": ");
                str_jsons(&b, args.p ? args.p : "{}");
                str_puts(&b, "}}");
                str_free(&args);
            }
            str_puts(&b, "]");
        }
        str_printf(&b, "}, \"logprobs\": null, \"finish_reason\": \"%s\"}], \"usage\": {\"prompt_tokens\": %d, "
                       "\"completion_tokens\": %d, \"total_tokens\": %d}}",
                   finish,
                   n_prompt, g.n_gen, n_prompt + g.n_gen);
    }
    http_send(c, 200, "OK", "application/json", b.p, b.len);
    str_free(&b);
    str_free(&stop_seq);
    qw_tool_calls_free(&calls);
    genres_free(&g);
}

static time_t started_at;     /* a model's `created`, which should not move */

/* The model list, in whichever API's shape the client speaks: the two share a
 * path but not a format, and Anthropic clients say who they are with an
 * anthropic-version header. */
static void handle_models(conn *c, bool single, bool anthropic) {
    str one = { 0 };
    if (anthropic) {
        char when[32];
        strftime(when, sizeof when, "%Y-%m-%dT%H:%M:%SZ", gmtime(&started_at));
        str_printf(&one, "{\"type\": \"model\", \"id\": \"%s\", "
                         "\"display_name\": \"%s\", \"created_at\": \"%s\"}",
                   model_id, model_name, when);
    } else {
        str_printf(&one, "{\"id\": \"%s\", \"object\": \"model\", \"created\": %ld, "
                         "\"owned_by\": \"qwasar\"}", model_id, (long)started_at);
    }
    str b = { 0 };
    if (single)
        str_puts(&b, one.p);
    else if (anthropic)
        str_printf(&b, "{\"data\": [%s], \"has_more\": false, \"first_id\": \"%s\", "
                       "\"last_id\": \"%s\"}", one.p, model_id, model_id);
    else
        str_printf(&b, "{\"object\": \"list\", \"data\": [%s]}", one.p);
    http_send(c, 200, "OK", "application/json", b.p, b.len);
    str_free(&one);
    str_free(&b);
}

/* ---- request loop ----------------------------------------------------------- */

typedef struct {
    char   method[8];
    char   path[512];
    size_t content_length;
    bool   keep_alive;
    bool   anthropic;      /* sent an anthropic-version header */
    bool   chunked;        /* Transfer-Encoding: chunked */
    bool   expect_continue;
    bool   too_large;      /* body over QW_MAX_BODY; left unread */
} http_req;

/* Base64 images and video make for big bodies, but not this big. */
#define QW_MAX_BODY ((size_t)256 << 20)

/* Reads until `carry` holds at least `want` bytes. */
static bool carry_fill(conn *c, str *carry, size_t want) {
    char buf[8192];
    while (carry->len < want) {
        ssize_t n = read(c->fd, buf, sizeof buf);
        if (n <= 0) return false;
        if (!str_add(carry, buf, (size_t)n)) return false;
    }
    return true;
}

/* Offset of the line ending at or after `pos` in `carry`, reading more as
 * needed; lines are CRLF-terminated but a bare LF is accepted. */
static bool carry_line(conn *c, str *carry, size_t pos, size_t *eol) {
    for (;;) {
        const char *nl = carry->len > pos ? memchr(carry->p + pos, '\n', carry->len - pos) : NULL;
        if (nl) { *eol = (size_t)(nl - carry->p); return true; }
        if (carry->len - pos > 4096) return false;       /* no chunk line is this long */
        if (!carry_fill(c, carry, carry->len + 1)) return false;
    }
}

/* Decodes a chunked body starting at carry[pos] into `body`, returning the
 * offset just past it.  Chunk extensions and trailers are read and dropped. */
static bool read_chunked(conn *c, str *carry, size_t pos, str *body, size_t *end) {
    for (;;) {
        size_t eol;
        if (!carry_line(c, carry, pos, &eol)) return false;
        char *stop = NULL;
        const unsigned long long sz = strtoull(carry->p + pos, &stop, 16);
        if (stop == carry->p + pos) return false;        /* not a chunk size */
        pos = eol + 1;
        if (sz == 0) {
            for (;;) {                                   /* trailers, then a blank line */
                if (!carry_line(c, carry, pos, &eol)) return false;
                const bool blank = eol == pos || (eol == pos + 1 && carry->p[pos] == '\r');
                pos = eol + 1;
                if (blank) break;
            }
            *end = pos;
            return true;
        }
        if (sz > QW_MAX_BODY || body->len + sz > QW_MAX_BODY) return false;
        if (!carry_fill(c, carry, pos + sz + 2)) return false;
        if (!str_add(body, carry->p + pos, (size_t)sz)) return false;
        pos += (size_t)sz;
        if (carry->p[pos] == '\r') pos++;
        if (carry->p[pos] != '\n') return false;
        pos++;
    }
}

/* Reads one request.  Returns false when the connection is finished or
 * malformed; `body` is left owning the payload. */
static bool read_request(conn *c, str *carry, http_req *r, str *body) {
    memset(r, 0, sizeof *r);
    r->keep_alive = true;

    char buf[8192];
    const char *hend = NULL;
    for (;;) {
        if (carry->len) {
            hend = strstr(carry->p, "\r\n\r\n");
            if (hend) break;
        }
        ssize_t n = read(c->fd, buf, sizeof buf);
        if (n <= 0) return false;
        if (!str_add(carry, buf, (size_t)n)) return false;
        if (carry->len > (8u << 20)) return false;   /* headers are not this big */
    }

    const size_t head_len = (size_t)(hend - carry->p) + 4;

    /* Request line. */
    const char *sp1 = memchr(carry->p, ' ', head_len);
    if (!sp1) return false;
    const char *sp2 = memchr(sp1 + 1, ' ', head_len - (size_t)(sp1 + 1 - carry->p));
    if (!sp2) return false;
    size_t ml = (size_t)(sp1 - carry->p), pl = (size_t)(sp2 - sp1 - 1);
    if (ml >= sizeof r->method || pl >= sizeof r->path) return false;
    memcpy(r->method, carry->p, ml); r->method[ml] = 0;
    memcpy(r->path, sp1 + 1, pl);    r->path[pl] = 0;

    /* Strip a query string; none of the endpoints take one. */
    char *q = strchr(r->path, '?');
    if (q) *q = 0;

    /* Headers, matched case-insensitively as HTTP requires.
     *
     * The walk runs to the end of the header block rather than to the blank
     * line that terminates it: that blank line's first CR is also the last
     * header's terminator, so stopping there drops the final header -- which is
     * routinely the Content-Length. */
    {
        const char *end = carry->p + head_len;
        const char *p = memchr(carry->p, '\n', head_len);
        while (p && p + 1 < end) {
            const char *line = p + 1;
            const char *nl = memchr(line, '\n', (size_t)(end - line));
            if (!nl) break;
            size_t len = (size_t)(nl - line);
            if (len && line[len - 1] == '\r') len--;
            if (len == 0) break;                 /* blank line: headers are done */

            if (len > 15 && !strncasecmp(line, "Content-Length:", 15)) {
                r->content_length = (size_t)strtoul(line + 15, NULL, 10);
            } else if (len > 18 && !strncasecmp(line, "anthropic-version:", 18)) {
                r->anthropic = true;
            } else if (len > 18 && !strncasecmp(line, "Transfer-Encoding:", 18)) {
                for (size_t i = 18; i + 7 <= len; i++)
                    if (!strncasecmp(line + i, "chunked", 7)) { r->chunked = true; break; }
            } else if (len > 7 && !strncasecmp(line, "Expect:", 7)) {
                for (size_t i = 7; i + 12 <= len; i++)
                    if (!strncasecmp(line + i, "100-continue", 12)) {
                        r->expect_continue = true;
                        break;
                    }
            } else if (len > 11 && !strncasecmp(line, "Connection:", 11)) {
                for (size_t i = 11; i + 5 <= len; i++)
                    if (!strncasecmp(line + i, "close", 5)) { r->keep_alive = false; break; }
            }
            p = nl;
        }
    }

    if (!r->chunked && r->content_length > QW_MAX_BODY) {
        /* Refused unread, and the connection with it: the body is still on
         * the wire, so nothing after it can be framed. */
        r->too_large = true;
        r->keep_alive = false;
        return true;
    }

    /* A client that asked first waits for the go-ahead before sending the
     * body -- curl does, for large uploads, and otherwise stalls a second. */
    if (r->expect_continue && (r->chunked || carry->len < head_len + r->content_length)) {
        static const char go[] = "HTTP/1.1 100 Continue\r\n\r\n";
        conn_write(c, go, sizeof go - 1);
    }

    /* Chunked framing takes precedence over Content-Length, as HTTP/1.1
     * requires; clients that stream their request body send it this way. */
    size_t consumed;
    if (r->chunked) {
        if (!read_chunked(c, carry, head_len, body, &consumed)) return false;
    } else {
        if (!carry_fill(c, carry, head_len + r->content_length)) return false;
        str_add(body, carry->p + head_len, r->content_length);
        consumed = head_len + r->content_length;
    }

    /* Keep anything belonging to the next pipelined request. */
    memmove(carry->p, carry->p + consumed, carry->len - consumed);
    carry->len -= consumed;
    carry->p[carry->len] = 0;
    return true;
}

static void serve(server *sv, conn *c) {
    str carry = { 0 };
    for (;;) {
        http_req r;
        str body = { 0 };
        if (!read_request(c, &carry, &r, &body)) { str_free(&body); break; }

        if (sv->verbose) fprintf(stderr, "%s %s\n", r.method, r.path);
        /* Errors take the shape of the API being spoken: the Messages API's
         * paths are Anthropic's, and elsewhere an anthropic-version header
         * says so -- except on OpenAI's own completions path. */
        const bool messages_api = !strncmp(r.path, "/v1/messages", 12);
        c->anthropic = messages_api
                    || (r.anthropic && strcmp(r.path, "/v1/chat/completions"));

        if (r.too_large) {
            http_error(c, 413, "Payload Too Large", "request body is too large");
        } else if (!strcmp(r.method, "OPTIONS")) {
            http_send(c, 204, "No Content", "text/plain", "", 0);
        } else if (!strcmp(r.method, "GET")
                   && (!strcmp(r.path, "/health") || !strcmp(r.path, "/"))) {
            const char *ok = "{\"status\": \"ok\"}";
            http_send(c, 200, "OK", "application/json", ok, strlen(ok));
        } else if (!strcmp(r.method, "GET") && !strcmp(r.path, "/v1/models")) {
            handle_models(c, false, r.anthropic);
        } else if (!strcmp(r.method, "GET") && !strncmp(r.path, "/v1/models/", 11)) {
            if (!strcmp(r.path + 11, model_id)) handle_models(c, true, r.anthropic);
            else http_error_at(c, 404, "Not Found", "no such model", "model", "model_not_found");
        } else if (!strcmp(r.method, "POST")
                   && (!strcmp(r.path, "/v1/chat/completions")
                       || !strcmp(r.path, "/v1/messages")
                       || !strcmp(r.path, "/v1/messages/count_tokens"))) {
            qj_doc d;
            if (!qj_parse(&d, body.p ? body.p : "", body.len)) {
                http_error(c, 400, "Bad Request", d.err);
            } else {
                /* The engine holds one session, so completions take turns.
                 * Everything else answers without waiting for them. */
                pthread_mutex_lock(&sv->lock);
                handle_completion(sv, c, &d, messages_api,
                                  !strcmp(r.path, "/v1/messages/count_tokens"));
                pthread_mutex_unlock(&sv->lock);
            }
            qj_free(&d);
        } else if (!strcmp(r.method, "POST")
                   && (!strcmp(r.path, "/v1/responses")
                       || !strcmp(r.path, "/v1/completions"))) {
            http_error(c, 501, "Not Implemented",
                       "qwasar-server implements /v1/chat/completions and /v1/messages; "
                       "/v1/responses and /v1/completions are not available yet");
        } else {
            http_error(c, 404, "Not Found", "no such endpoint");
        }

        str_free(&body);
        if (c->dead || !r.keep_alive) break;
    }
    str_free(&carry);
}

/* Each connection gets a thread, so a client holding a keep-alive connection
 * open -- every pooled HTTP client does -- cannot lock the others out.  Only
 * completions are serialised, by the engine lock in serve(). */
#define QW_MAX_CONNS 64
#define QW_IDLE_SECS 300

typedef struct {
    server *sv;
    conn    c;
} conn_job;

static void *conn_main(void *arg) {
    conn_job *job = arg;
    server *sv = job->sv;
    serve(sv, &job->c);
    close(job->c.fd);
    free(job);
    pthread_mutex_lock(&sv->conns_lock);
    sv->n_conns--;
    pthread_mutex_unlock(&sv->conns_lock);
    return NULL;
}

static void *stdin_watch(void *arg) {
    (void)arg;
    char buf[256];
    for (;;) {
        const ssize_t n = read(STDIN_FILENO, buf, sizeof buf);
        if (n == 0 || (n < 0 && errno != EINTR)) break;
    }
    fprintf(stderr, "qwasar-server: standard input closed; exiting\n");
    _exit(0);
}

static void usage(FILE *out) {
    fprintf(out,
        "qwasar-server -- OpenAI and Anthropic compatible HTTP API for Qwen3.8\n"
        "\n"
        "usage: qwasar-server [-m <model-dir>] [options]\n"
        "\n"
        "  -m, --model <dir>   model directory; default ./qwasar-model\n"
        "      --host <addr>   bind address (default 127.0.0.1)\n"
        "      --port <n>      port (default 8080)\n"
        "      --ctx <n>       context size in tokens (default 32768)\n"
        "      --max-tokens <n>\n"
        "                      output limit for requests that set none (default\n"
        "                      2048); 0 means whatever room the context has left.\n"
        "                      A request's own limit is capped to that room too.\n"
        "      --cors          emit Access-Control-Allow-* headers\n"
        "      --no-cache      do not use or write disk checkpoints\n"
        "      --exit-on-eof   exit when standard input closes, so a supervising\n"
        "                      app that holds the other end cannot be outlived\n"
        "  -v, --verbose       log requests\n"
        "  -h, --help          this message\n"
        "\n"
        "Endpoints:\n"
        "  GET  /health\n"
        "  GET  /v1/models\n"
        "  GET  /v1/models/{id}\n"
        "  POST /v1/chat/completions   OpenAI, streaming and not, with tools\n"
        "  POST /v1/messages           Anthropic, streaming and not, with tools\n"
        "  POST /v1/messages/count_tokens\n"
        "\n"
        "One request is served at a time: the model's recurrent layers hold a\n"
        "single session that cannot be forked. A client resending a growing\n"
        "conversation continues from wherever that session already is.\n");
}

static bool resolve_model(qwasar_options *opts, const char *prog) {
    if (opts->model_path) return true;
    opts->model_path = qwasar_default_model_path();
    if (opts->model_path) return true;
    fprintf(stderr,
        "%s: no model given and none found.\n"
        "\n"
        "Download it once:\n"
        "    ./download_model.sh model\n"
        "\n"
        "or point at an existing copy with -m <dir>, or set QWASAR_MODEL.\n", prog);
    return false;
}

int main(int argc, char **argv) {
    qwasar_options opts = { 0 };
    server sv = { 0 };
    sv.max_tokens = 2048;
    const char *host = "127.0.0.1";
    int port = 8080;
    bool cors = false;
    bool exit_on_eof = false;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if ((!strcmp(a, "-m") || !strcmp(a, "--model")) && i + 1 < argc) opts.model_path = argv[++i];
        else if (!strcmp(a, "--host") && i + 1 < argc) host = argv[++i];
        else if (!strcmp(a, "--port") && i + 1 < argc) port = atoi(argv[++i]);
        else if (!strcmp(a, "--ctx") && i + 1 < argc) opts.context_size = atoi(argv[++i]);
        else if (!strcmp(a, "--max-tokens") && i + 1 < argc) sv.max_tokens = atoi(argv[++i]);
        else if (!strcmp(a, "--cors")) cors = true;
        else if (!strcmp(a, "--no-cache")) sv.no_cache = true;
        else if (!strcmp(a, "--exit-on-eof")) exit_on_eof = true;
        else if (!strcmp(a, "-v") || !strcmp(a, "--verbose")) sv.verbose = true;
        else if (!strcmp(a, "-h") || !strcmp(a, "--help")) { usage(stdout); return 0; }
        else { fprintf(stderr, "qwasar-server: unknown argument '%s'\n\n", a); usage(stderr); return 2; }
    }
    if (!resolve_model(&opts, "qwasar-server")) return 2;

    /* A supervisor -- the menu bar app -- runs the server with a pipe on stdin
     * and never writes to it.  The pipe closes when the supervisor exits,
     * however it exits, and the server goes with it rather than holding the
     * port with nothing left to show that it is running.  Watched from the
     * start, so a supervisor that dies during the model load is noticed too. */
    if (exit_on_eof) {
        pthread_t t;
        if (pthread_create(&t, NULL, stdin_watch, NULL) == 0) pthread_detach(t);
    }

    /* A client that hangs up mid-stream would otherwise take the server with
     * it; write failures are detected and end the response instead. */
    signal(SIGPIPE, SIG_IGN);

    char err[512] = "";
    sv.e = qwasar_engine_load(&opts, err, sizeof err);
    if (!sv.e) { fprintf(stderr, "qwasar-server: %s\n", err); return 1; }
    sv.tok = qwasar_tokenizer_load(opts.model_path, err, sizeof err);
    if (!sv.tok) { fprintf(stderr, "qwasar-server: %s\n", err); return 1; }
    sv.ctx = opts.context_size > 0 ? opts.context_size : 32768;
    model_id = qwasar_model_id(sv.e);
    model_name = qwasar_model_name(sv.e);
    started_at = time(NULL);

    int ls = socket(AF_INET, SOCK_STREAM, 0);
    if (ls < 0) { perror("socket"); return 1; }
    int one = 1;
    setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof addr);
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port);
    if (inet_pton(AF_INET, host, &addr.sin_addr) != 1) {
        fprintf(stderr, "qwasar-server: bad bind address '%s'\n", host);
        return 1;
    }
    if (bind(ls, (struct sockaddr *)&addr, sizeof addr) != 0) { perror("bind"); return 1; }
    if (listen(ls, 16) != 0) { perror("listen"); return 1; }

    fprintf(stderr, "qwasar-server on http://%s:%d  (model %s, ctx %d)\n",
            host, port, model_id, sv.ctx);

    pthread_mutex_init(&sv.lock, NULL);
    pthread_mutex_init(&sv.conns_lock, NULL);
    pthread_attr_t detached;
    pthread_attr_init(&detached);
    pthread_attr_setdetachstate(&detached, PTHREAD_CREATE_DETACHED);

    for (;;) {
        int fd = accept(ls, NULL, NULL);
        if (fd < 0) { if (errno == EINTR) continue; perror("accept"); break; }
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
        /* A peer that goes quiet -- idle between requests, or not reading a
         * stream -- is dropped rather than keeping its thread for ever.  A
         * stalled reader would otherwise also hold the engine lock. */
        struct timeval idle = { .tv_sec = QW_IDLE_SECS };
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &idle, sizeof idle);
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &idle, sizeof idle);

        conn_job *job = calloc(1, sizeof *job);
        pthread_mutex_lock(&sv.conns_lock);
        const bool room = sv.n_conns < QW_MAX_CONNS;
        if (room && job) sv.n_conns++;
        pthread_mutex_unlock(&sv.conns_lock);
        if (!room || !job) {
            conn c = { .fd = fd, .cors = cors };
            http_error_at(&c, 503, "Service Unavailable", "too many open connections",
                          NULL, NULL);
            close(fd);
            free(job);
            continue;
        }
        job->sv = &sv;
        job->c = (conn){ .fd = fd, .cors = cors };
        pthread_t t;
        if (pthread_create(&t, &detached, conn_main, job) != 0) {
            pthread_mutex_lock(&sv.conns_lock);
            sv.n_conns--;
            pthread_mutex_unlock(&sv.conns_lock);
            close(fd);
            free(job);
        }
    }

    qwasar_session_free(sv.s);
    qwasar_tokenizer_free(sv.tok);
    qwasar_engine_free(sv.e);
    close(ls);
    return 0;
}
