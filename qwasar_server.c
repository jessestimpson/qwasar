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
 * checkpoint when the live session has moved on to something else. */

#include "qwasar.h"
#include "qwasar_json.h"
#include "qwasar_toolcall.h"

#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#define QW_MODEL_ID "qwen3.8-27b"

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

static void http_error(conn *c, int status, const char *reason, const char *msg) {
    str b = { 0 };
    str_puts(&b, "{\"error\": {\"message\": ");
    str_jsons(&b, msg);
    str_printf(&b, ", \"type\": \"invalid_request_error\", \"code\": %d}}", status);
    http_send(c, status, reason, "application/json", b.p, b.len);
    str_free(&b);
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
    bool              no_cache;
    uint64_t          rng;
    bool              verbose;
} server;

/* Evaluates `tokens`, reusing whatever the live session already covers.
 *
 * Prefix reuse is all-or-nothing here.  A KV cache could be truncated back to
 * the first divergence, but the recurrent layers keep no per-position history,
 * so a session that has evaluated one wrong token is worthless for this prompt
 * and has to be replaced. */
static const float *srv_prefill(server *sv, const int32_t *tokens, int32_t n,
                                const qwasar_image_input *images, int32_t n_images,
                                int32_t *reused, char *err, size_t cap) {
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
        sv->s = qwasar_session_new(sv->e, err, cap);
        if (!sv->s) return NULL;
        return qwasar_session_eval_images(sv->s, tokens, n, images, n_images, err, cap);
    }

    int32_t live = sv->s ? qwasar_session_common_prefix(sv->s, tokens, n) : 0;
    if (live > 0 && live < n) {
        *reused = live;
        return qwasar_session_eval(sv->s, tokens + live, n - live, err, cap);
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
    return qwasar_session_eval(sv->s, tokens + covered, n - covered, err, cap);
}

typedef void (*delta_fn)(void *ud, bool reasoning, const char *s, size_t n);

typedef struct {
    str     text;
    str     reasoning;
    int32_t n_gen;
    bool    hit_eos;
    bool    has_call;
} genres;

static void genres_free(genres *g) { str_free(&g->text); str_free(&g->reasoning); }

/* One assistant turn.  Stops at end-of-turn, a completed tool call, or the
 * token budget. */
static bool srv_generate(server *sv, const float *logits, const qwasar_sampling *sp,
                         int32_t max_tokens, bool thinking,
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

    for (int32_t i = 0; i < max_tokens; i++) {
        int32_t next = qwasar_sample(logits, vocab, sp, &sv->rng);
        if (qwasar_is_eos(sv->e, next)) { out->hit_eos = true; break; }
        out->n_gen++;

        size_t len = 0;
        bool special = false;
        const char *bytes = qwasar_token_bytes(sv->tok, next, &len, &special);

        if (next == think_close) {
            reasoning = false;
        } else if (bytes && len) {
            str_add(reasoning ? &out->reasoning : &out->text, bytes, len);
            /* Control tokens are structure, not content: they belong in the
             * accumulated text the parser sees, never in a client delta. */
            if (on_delta && !special) on_delta(ud, reasoning, bytes, len);
        }

        if (!reasoning && next != call_open
            && qw_tool_call_complete(out->text.p ? out->text.p : "", out->text.len)) {
            out->has_call = true;
            break;
        }

        logits = qwasar_session_eval(sv->s, &next, 1, err, cap);
        if (!logits) return false;
    }
    return true;
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
    snprintf(out, cap, "%s%08llx%04llx", prefix,
             (unsigned long long)time(NULL), (unsigned long long)(++counter & 0xffff));
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
} stream_ctx;

static void oai_delta(stream_ctx *st, bool reasoning, const char *s, size_t n) {
    str b = { 0 };
    str_printf(&b, "{\"id\": \"%s\", \"object\": \"chat.completion.chunk\", "
                   "\"created\": %ld, \"model\": \"%s\", \"choices\": "
                   "[{\"index\": 0, \"delta\": {",
               st->id, st->created, QW_MODEL_ID);
    str_puts(&b, reasoning ? "\"reasoning_content\": " : "\"content\": ");
    str_json(&b, s, n);
    str_puts(&b, "}, \"finish_reason\": null}]}");
    sse_event(st->c, NULL, b.p);
    str_free(&b);
}

static void ant_block_close(stream_ctx *st) {
    if (!st->open) return;
    str b = { 0 };
    str_printf(&b, "{\"type\": \"content_block_stop\", \"index\": %d}", st->index);
    sse_event(st->c, "content_block_stop", b.p);
    str_free(&b);
    st->open = false;
    st->index++;
}

static void ant_block_open(stream_ctx *st, bool thinking) {
    str b = { 0 };
    str_printf(&b, "{\"type\": \"content_block_start\", \"index\": %d, "
                   "\"content_block\": {\"type\": \"%s\", \"%s\": \"\"}}",
               st->index, thinking ? "thinking" : "text", thinking ? "thinking" : "text");
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

static void handle_completion(server *sv, conn *c, const qj_doc *d, bool anthropic) {
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

    qwasar_sampling sp;
    read_sampling(&sp, d, root);
    sv->rng = sp.seed ? sp.seed : (uint64_t)time(NULL) * 6364136223846793005ull + 1;

    const int32_t max_tokens = read_max_tokens(d, root, 2048);
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

    if (n_prompt >= sv->ctx) {
        free(prompt);
        req_free(&req);
        http_error(c, 400, "Bad Request", "prompt exceeds the server's context size");
        return;
    }

    int32_t reused = 0;
    /* The request outlives the prompt now: rendering turns its text into
     * tokens, but its image rows are what the prefill scatters in, so freeing
     * it here -- which is where it used to happen -- released them one call
     * before they were read. */
    const float *logits = srv_prefill(sv, prompt, n_prompt, req.images, req.n_images,
                                      &reused, err, sizeof err);
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
                       id, QW_MODEL_ID, n_prompt);
            sse_event(c, "message_start", b.p);
            str_free(&b);
        } else {
            str b = { 0 };
            str_printf(&b, "{\"id\": \"%s\", \"object\": \"chat.completion.chunk\", "
                           "\"created\": %ld, \"model\": \"%s\", \"choices\": [{\"index\": 0, "
                           "\"delta\": {\"role\": \"assistant\"}, \"finish_reason\": null}]}",
                       id, created, QW_MODEL_ID);
            sse_event(c, NULL, b.p);
            str_free(&b);
        }
    }

    genres g;
    bool ok = srv_generate(sv, logits, &sp, max_tokens, thinking,
                           stream ? on_delta : NULL, &st, &g, err, sizeof err);
    if (!ok) {
        if (stream) { ant_block_close(&st); sse_end(c); }
        else http_error(c, 500, "Internal Server Error", err);
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
                           "\"stop_sequence\": null}, \"usage\": {\"output_tokens\": %d}}",
                       n_calls > 0 ? "tool_use" : (g.hit_eos ? "end_turn" : "max_tokens"),
                       g.n_gen);
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
                           id, created, QW_MODEL_ID, i, tid);
                str_jsons(&b, calls.calls[i].name);
                str_puts(&b, ", \"arguments\": ");
                str_jsons(&b, args.p ? args.p : "{}");
                str_puts(&b, "}}]}, \"finish_reason\": null}]}");
                sse_event(c, NULL, b.p);
                str_free(&b);
                str_free(&args);
            }
            str b = { 0 };
            str_printf(&b, "{\"id\": \"%s\", \"object\": \"chat.completion.chunk\", "
                           "\"created\": %ld, \"model\": \"%s\", \"choices\": [{\"index\": 0, "
                           "\"delta\": {}, \"finish_reason\": \"%s\"}], "
                           "\"usage\": {\"prompt_tokens\": %d, \"completion_tokens\": %d, "
                           "\"total_tokens\": %d}}",
                       id, created, QW_MODEL_ID,
                       n_calls > 0 ? "tool_calls" : (g.hit_eos ? "stop" : "length"),
                       n_prompt, g.n_gen, n_prompt + g.n_gen);
            sse_event(c, NULL, b.p);
            str_free(&b);
            sse_event(c, NULL, "[DONE]");
        }
        sse_end(c);
        qw_tool_calls_free(&calls);
        genres_free(&g);
        return;
    }

    str b = { 0 };
    if (anthropic) {
        str_printf(&b, "{\"id\": \"%s\", \"type\": \"message\", \"role\": \"assistant\", "
                       "\"model\": \"%s\", \"content\": [", id, QW_MODEL_ID);
        bool first = true;
        if (g.reasoning.len) {
            str_puts(&b, "{\"type\": \"thinking\", \"thinking\": ");
            str_jsons(&b, g.reasoning.p);
            str_puts(&b, "}");
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
        str_printf(&b, "], \"stop_reason\": \"%s\", \"stop_sequence\": null, "
                       "\"usage\": {\"input_tokens\": %d, \"output_tokens\": %d}}",
                   n_calls > 0 ? "tool_use" : (g.hit_eos ? "end_turn" : "max_tokens"),
                   n_prompt, g.n_gen);
    } else {
        str_printf(&b, "{\"id\": \"%s\", \"object\": \"chat.completion\", \"created\": %ld, "
                       "\"model\": \"%s\", \"choices\": [{\"index\": 0, \"message\": "
                       "{\"role\": \"assistant\", \"content\": ", id, created, QW_MODEL_ID);
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
        str_printf(&b, "}, \"finish_reason\": \"%s\"}], \"usage\": {\"prompt_tokens\": %d, "
                       "\"completion_tokens\": %d, \"total_tokens\": %d}}",
                   n_calls > 0 ? "tool_calls" : (g.hit_eos ? "stop" : "length"),
                   n_prompt, g.n_gen, n_prompt + g.n_gen);
    }
    http_send(c, 200, "OK", "application/json", b.p, b.len);
    str_free(&b);
    qw_tool_calls_free(&calls);
    genres_free(&g);
}

static void handle_models(conn *c, bool single) {
    str b = { 0 };
    if (single)
        str_printf(&b, "{\"id\": \"%s\", \"object\": \"model\", \"created\": %ld, "
                       "\"owned_by\": \"qwasar\"}", QW_MODEL_ID, (long)time(NULL));
    else
        str_printf(&b, "{\"object\": \"list\", \"data\": [{\"id\": \"%s\", "
                       "\"object\": \"model\", \"created\": %ld, \"owned_by\": \"qwasar\"}]}",
                   QW_MODEL_ID, (long)time(NULL));
    http_send(c, 200, "OK", "application/json", b.p, b.len);
    str_free(&b);
}

/* ---- request loop ----------------------------------------------------------- */

typedef struct {
    char   method[8];
    char   path[512];
    size_t content_length;
    bool   keep_alive;
} http_req;

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
            } else if (len > 11 && !strncasecmp(line, "Connection:", 11)) {
                for (size_t i = 11; i + 5 <= len; i++)
                    if (!strncasecmp(line + i, "close", 5)) { r->keep_alive = false; break; }
            }
            p = nl;
        }
    }

    while (carry->len < head_len + r->content_length) {
        ssize_t n = read(c->fd, buf, sizeof buf);
        if (n <= 0) return false;
        if (!str_add(carry, buf, (size_t)n)) return false;
    }

    str_add(body, carry->p + head_len, r->content_length);

    /* Keep anything belonging to the next pipelined request. */
    size_t consumed = head_len + r->content_length;
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

        if (!strcmp(r.method, "OPTIONS")) {
            http_send(c, 204, "No Content", "text/plain", "", 0);
        } else if (!strcmp(r.method, "GET")
                   && (!strcmp(r.path, "/health") || !strcmp(r.path, "/"))) {
            const char *ok = "{\"status\": \"ok\"}";
            http_send(c, 200, "OK", "application/json", ok, strlen(ok));
        } else if (!strcmp(r.method, "GET") && !strcmp(r.path, "/v1/models")) {
            handle_models(c, false);
        } else if (!strcmp(r.method, "GET") && !strncmp(r.path, "/v1/models/", 11)) {
            handle_models(c, true);
        } else if (!strcmp(r.method, "POST")
                   && (!strcmp(r.path, "/v1/chat/completions")
                       || !strcmp(r.path, "/v1/messages"))) {
            qj_doc d;
            if (!qj_parse(&d, body.p ? body.p : "", body.len)) {
                http_error(c, 400, "Bad Request", d.err);
            } else {
                handle_completion(sv, c, &d, !strcmp(r.path, "/v1/messages"));
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
        "      --cors          emit Access-Control-Allow-* headers\n"
        "      --no-cache      do not use or write disk checkpoints\n"
        "  -v, --verbose       log requests\n"
        "  -h, --help          this message\n"
        "\n"
        "Endpoints:\n"
        "  GET  /health\n"
        "  GET  /v1/models\n"
        "  GET  /v1/models/{id}\n"
        "  POST /v1/chat/completions   OpenAI, streaming and not, with tools\n"
        "  POST /v1/messages           Anthropic, streaming and not, with tools\n"
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
    const char *host = "127.0.0.1";
    int port = 8080;
    bool cors = false;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if ((!strcmp(a, "-m") || !strcmp(a, "--model")) && i + 1 < argc) opts.model_path = argv[++i];
        else if (!strcmp(a, "--host") && i + 1 < argc) host = argv[++i];
        else if (!strcmp(a, "--port") && i + 1 < argc) port = atoi(argv[++i]);
        else if (!strcmp(a, "--ctx") && i + 1 < argc) opts.context_size = atoi(argv[++i]);
        else if (!strcmp(a, "--cors")) cors = true;
        else if (!strcmp(a, "--no-cache")) sv.no_cache = true;
        else if (!strcmp(a, "-v") || !strcmp(a, "--verbose")) sv.verbose = true;
        else if (!strcmp(a, "-h") || !strcmp(a, "--help")) { usage(stdout); return 0; }
        else { fprintf(stderr, "qwasar-server: unknown argument '%s'\n\n", a); usage(stderr); return 2; }
    }
    if (!resolve_model(&opts, "qwasar-server")) return 2;

    /* A client that hangs up mid-stream would otherwise take the server with
     * it; write failures are detected and end the response instead. */
    signal(SIGPIPE, SIG_IGN);

    char err[512] = "";
    sv.e = qwasar_engine_load(&opts, err, sizeof err);
    if (!sv.e) { fprintf(stderr, "qwasar-server: %s\n", err); return 1; }
    sv.tok = qwasar_tokenizer_load(opts.model_path, err, sizeof err);
    if (!sv.tok) { fprintf(stderr, "qwasar-server: %s\n", err); return 1; }
    sv.ctx = opts.context_size > 0 ? opts.context_size : 32768;

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
            host, port, QW_MODEL_ID, sv.ctx);

    for (;;) {
        int fd = accept(ls, NULL, NULL);
        if (fd < 0) { if (errno == EINTR) continue; perror("accept"); break; }
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
        conn c = { .fd = fd, .cors = cors };
        serve(&sv, &c);
        close(fd);
    }

    qwasar_session_free(sv.s);
    qwasar_tokenizer_free(sv.tok);
    qwasar_engine_free(sv.e);
    close(ls);
    return 0;
}
