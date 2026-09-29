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
 * turn of a long conversation (qwasar_sessions.c, the compat session).
 *
 * That anonymous session now sits beside the named ones of the Session API
 * (API.md, qwasar_api.c), in one store with one queue and one live set. */

#include "qwasar.h"
#include "qwasar_api.h"
#include "qwasar_http.h"
#include "qwasar_json.h"
#include "qwasar_sessions.h"
#include "qwasar_toolcall.h"

#include <arpa/inet.h>
#include <pthread.h>
#include <errno.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <stdatomic.h>
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

/* ---- engine ---------------------------------------------------------------- */

/* The engine, the sessions and the queue live in the store (qwasar_sessions.c);
 * the compat front keeps only what shapes its own requests. */
typedef struct {
    qw_store *st;
    int32_t   max_tokens;    /* output when a request names none; 0 = the context's room */
    bool      verbose;
    int       n_conns;       /* live connections, under conns_lock */
    pthread_mutex_t conns_lock;
} server;

static server     *g_srv;
static atomic_bool g_ready;      /* the model is loaded */
/* --token: every request but /health carries it as a bearer. */
static const char *g_token;

/* Tokens per second, 0 for no time. */
static double srv_rate(int32_t n, double secs) { return secs > 0 ? n / secs : 0.0; }

/* How many leading tokens of `prompt` the first `n_msgs` messages render to
 * under `opts` -- 0 unless that rendering is a proper prefix of it. */
static int32_t srv_prefix_len(server *sv, const qwasar_message *msgs, int32_t n_msgs,
                              const qwasar_chat_options *opts, const int32_t *prompt, int32_t n) {
    char err[256];
    int32_t m = 0;
    int32_t *p = qwasar_apply_chat_template(qw_store_tokenizer(sv->st), msgs, n_msgs, opts, &m,
                                            err, sizeof err);
    if (!p) return 0;
    const bool ok = m > 0 && m < n && !memcmp(p, prompt, (size_t)m * sizeof *p);
    free(p);
    return ok ? m : 0;
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
            ? qwasar_video_encode_memory(qw_store_engine(sv->st), raw, raw_len, ext,
                                         &r->images[r->n_images], err, cap)
            : qwasar_image_encode_memory(qw_store_engine(sv->st), raw, raw_len,
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
 * tool choice -- with the storage the qw_genopts pointers refer to. */
typedef struct {
    qw_genopts go;
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
    const double t_start = qw_now();
    qw_sess *cs = qw_store_compat(sv->st);

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
    qw_store_seed(sv->st, sp.seed);

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
    int32_t *prompt = qwasar_apply_chat_template(qw_store_tokenizer(sv->st), req.msgs, req.n, &chat,
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
    const int32_t room = qw_store_ctx(sv->st) - n_prompt;
    if (max_tokens <= 0 || max_tokens > room) max_tokens = room;

    if (n_prompt >= qw_store_ctx(sv->st)) {
        free(prompt);
        req_free(&req);
        http_error(c, 400, "Bad Request", "prompt exceeds the server's context size");
        return;
    }

    /* Checkpoint boundaries: the prompt rendered only as far as the system
     * messages (with the tools, which the template puts there), and as far as
     * the last complete turn.  Used only where the rendering really is a
     * prefix of the prompt. */
    qw_ckpt_marks marks = { 0, 0 };
    if (!qw_store_no_cache(sv->st) && req.n_images == 0) {
        qwasar_chat_options upto = chat;
        upto.add_generation_prompt = false;
        upto.continue_final_message = false;
        int32_t n_sys = 0;
        while (n_sys < req.n && req.msgs[n_sys].role && !strcmp(req.msgs[n_sys].role, "system")) n_sys++;
        if (n_sys > 0) marks.sys_n = srv_prefix_len(sv, req.msgs, n_sys, &upto, prompt, n_prompt);
        if (!prefill) marks.hist_n = srv_prefix_len(sv, req.msgs, req.n, &upto, prompt, n_prompt);
    }

    if (sv->verbose)
        qw_log("  %s request: %d message%s, %d tool%s, thinking %s, %s",
                anthropic ? "Anthropic" : "OpenAI", req.n, req.n == 1 ? "" : "s",
                req.n_tools, req.n_tools == 1 ? "" : "s", thinking ? "on" : "off",
                stream ? "streaming" : "not streaming");

    int32_t reused = 0;
    const char *how = "";
    /* The request outlives the prompt now: rendering turns its text into
     * tokens, but its image rows are what the prefill scatters in, so freeing
     * it here -- which is where it used to happen -- released them one call
     * before they were read. */
    const double t_prefill = qw_now();
    const float *logits = qw_compat_prefill(sv->st, cs, prompt, n_prompt, req.images, req.n_images,
                                            &marks, &reused, &how, err, sizeof err);
    int32_t miss_at = -1, miss_of = 0;
    const char *miss_had = "", *miss_got = "";
    qw_compat_last_miss(sv->st, &miss_at, &miss_of, &miss_had, &miss_got);
    if (sv->verbose && miss_at >= 0) {
        /* Which message the departure falls in: the first whose rendering
         * reaches past it. */
        int32_t msg = -1;
        qwasar_chat_options upto = chat;
        upto.add_generation_prompt = false;
        upto.continue_final_message = false;
        for (int32_t k = 1; k <= req.n && msg < 0; k++) {
            char e2[64];
            int32_t m = 0;
            int32_t *p = qwasar_apply_chat_template(qw_store_tokenizer(sv->st), req.msgs, k, &upto, &m, e2, sizeof e2);
            free(p);
            if (p && m > miss_at) msg = k - 1;
        }
        char where[96];
        if (msg >= 0)
            snprintf(where, sizeof where, "message %d of %d (%s)", msg + 1, req.n,
                     req.msgs[msg].role ? req.msgs[msg].role : "?");
        else
            snprintf(where, sizeof where, "after the last message");
        qw_log("  cannot reuse the live session (%d tokens): this prompt departs from it "
                "at token %d, in %s", miss_of, miss_at, where);
        qw_log("    session had: ...%s...", miss_had);
        qw_log("    prompt has:  ...%s...", miss_got);
    }
    req_free(&req);
    free(prompt);
    const double prefill_s = qw_now() - t_prefill;
    if (!logits) {
        qw_log("  prefill failed: %s", err);
        http_error(c, 500, "Internal Server Error", err);
        return;
    }
    if (sv->verbose) {
        /* A rate over a handful of tokens is mostly fixed cost; not shown. */
        const int32_t fresh = n_prompt - reused;
        char rate[32] = "";
        if (fresh >= 16) snprintf(rate, sizeof rate, " (%.0f tok/s)", srv_rate(fresh, prefill_s));
        if (reused > 0)
            qw_log("  prompt %d tokens: %d reused from %s, %d prefilled in %.2fs%s",
                    n_prompt, reused, how, fresh, prefill_s, rate);
        else
            qw_log("  prompt %d tokens: prefilled in %.2fs%s, from %s",
                    n_prompt, prefill_s, rate, how);
    }

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

    qw_genres g;
    /* A prefilled turn already has its reasoning block closed, so the
     * continuation starts in the answer. */
    const double t_decode = qw_now();
    bool ok = qw_generate(sv->st, cs, logits, &sp, max_tokens, thinking && !prefill, &oo.go,
                          stream ? on_delta : NULL, NULL, &st, NULL, &g, err, sizeof err);
    const double decode_s = qw_now() - t_decode;
    if (sv->verbose || !ok) {
        const char *why = !ok ? "failed" : g.has_call ? "a tool call"
                        : g.hit_stop ? "a stop sequence" : g.hit_eos ? "end of turn"
                        : qw_store_stopping() ? "shutdown" : "the output limit";
        qw_log("  reply %d tokens in %.2fs (%.1f tok/s), ended by %s%s%s; "
                "first token after %.2fs, request %.2fs",
                g.n_gen, decode_s, srv_rate(g.n_gen, decode_s), why,
                ok ? "" : ": ", ok ? "" : err, t_decode - t_start, qw_now() - t_start);
    }
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
        qw_genres_free(&g);
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
                qw_args_json(&args, &calls.calls[i], d);
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
                qw_args_json(&args, &calls.calls[i], d);
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
        qw_genres_free(&g);
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
            qw_args_json(&b, &calls.calls[i], d);
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
                qw_args_json(&args, &calls.calls[i], d);
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
    qw_genres_free(&g);
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

static void serve(server *sv, conn *c) {
    str carry = { 0 };
    for (;;) {
        http_req r;
        str body = { 0 };
        if (!read_request(c, &carry, &r, &body)) { str_free(&body); break; }

        if (sv->verbose && strcmp(r.path, "/health")) qw_log("%s %s", r.method, r.path);
        /* Errors take the shape of the API being spoken: the Messages API's
         * paths are Anthropic's, and elsewhere an anthropic-version header
         * says so -- except on OpenAI's own completions path. */
        const bool messages_api = !strncmp(r.path, "/v1/messages", 12);
        c->anthropic = messages_api
                    || (r.anthropic && strcmp(r.path, "/v1/chat/completions"));

        if (r.too_large) {
            http_error(c, 413, "Payload Too Large", "request body is too large");
        } else if (g_token && strcmp(r.path, "/health") && strcmp(r.method, "OPTIONS")
                   && strcmp(r.bearer, g_token)) {
            http_error(c, 401, "Unauthorized", "this server needs Authorization: Bearer <token>");
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
        } else if (qw_api_handle(sv->st, c, &r, &body)) {
            /* the Session API answered */
        } else if (!strcmp(r.method, "POST")
                   && (!strcmp(r.path, "/v1/chat/completions")
                       || !strcmp(r.path, "/v1/messages")
                       || !strcmp(r.path, "/v1/messages/count_tokens"))) {
            qj_doc d;
            if (!qj_parse(&d, body.p ? body.p : "", body.len)) {
                http_error(c, 400, "Bad Request", d.err);
            } else if (!strcmp(r.path, "/v1/messages/count_tokens")) {
                handle_completion(sv, c, &d, messages_api, true);
            } else {
                /* One step at a time on the engine: a completion waits its
                 * turn behind whatever session holds it. */
                qw_sess *cs = qw_store_compat(sv->st);
                qw_store_acquire(sv->st, cs);
                handle_completion(sv, c, &d, messages_api, false);
                qw_store_release(sv->st, cs);
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

/* A graceful stop -- SIGTERM, SIGINT, or the supervisor closing stdin --
 * writes every live session to disk before exiting, so the next run picks
 * them up where they were.  Once only; whichever cause comes first does it. */
static void srv_shutdown(const char *why) {
    static atomic_flag once = ATOMIC_FLAG_INIT;
    if (atomic_flag_test_and_set(&once)) {
        for (;;) pause();                  /* the first caller exits for us all */
    }
    qw_log("%s; exiting", why);
    server *sv = g_srv;
    if (!atomic_load(&g_ready) || !sv || !sv->st) _exit(0);
    /* Steps in flight end at their next token; every live session is
     * checkpointed and its record written (qw_store_shutdown). */
    qw_store_shutdown(sv->st);
    _exit(0);
}

static void *stdin_watch(void *arg) {
    (void)arg;
    char buf[256];
    for (;;) {
        const ssize_t n = read(STDIN_FILENO, buf, sizeof buf);
        if (n == 0 || (n < 0 && errno != EINTR)) break;
    }
    srv_shutdown("standard input closed");
    return NULL;
}

static void *shutdown_run(void *arg) {
    srv_shutdown(arg);
    return NULL;
}

/* SIGTERM and SIGINT, taken synchronously on a thread of their own (they are
 * blocked everywhere else).  The first starts the shutdown; a second, while
 * a large checkpoint is still being written, exits at once. */
static void *signal_watch(void *arg) {
    sigset_t *set = arg;
    bool first = true;
    for (;;) {
        int sig = 0;
        if (sigwait(set, &sig) != 0) continue;
        if (!first) _exit(128 + sig);
        first = false;
        pthread_t t;
        if (pthread_create(&t, NULL, shutdown_run,
                           (void *)(sig == SIGINT ? "interrupted" : "terminated")) == 0)
            pthread_detach(t);
        else
            _exit(0);
    }
    return NULL;
}

static void usage(FILE *out) {
    fprintf(out,
        "qwasar-server -- the Session API, and OpenAI and Anthropic compatible\n"
        "                 endpoints, for Qwen3.8 27B and Flash-Next\n"
        "\n"
        "usage: qwasar-server [-m <model-dir>] [options]\n"
        "\n"
        "  -m, --model <dir>   model directory; default ./qwasar-model\n"
        "      --host <addr>   bind address (default 127.0.0.1)\n"
        "      --port <n>      port (default 8080)\n"
        "      --ctx <n>       context size in tokens (default: what the machine\n"
        "                      holds beside the weights, up to the model's window)\n"
        "      --live <n>      sessions kept in memory at once (default: from the\n"
        "                      same arithmetic; 1 on a 32 GB machine)\n"
        "      --state-dir <d> where sessions live (default\n"
        "                      ~/Library/Application Support/Qwasar; $QWASAR_STATE)\n"
        "      --token <t>     require Authorization: Bearer <t> on every request\n"
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
        "  GET  /v1/server              the Session API (API.md): the model, the\n"
        "  POST /v1/sessions            machine, and sessions the server owns --\n"
        "  POST /v1/sessions/{id}/turn  a prefix fixed at open, deltas only,\n"
        "  ...                          prefill progress and warmth over SSE\n"
        "\n"
        "One step runs on the engine at a time; sessions queue for it.  A\n"
        "stateless client resending a growing conversation continues from\n"
        "wherever the anonymous session already is.\n");
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
    qw_store_opts so = { 0 };

    /* Stops are taken by a thread of their own from the first moment, so one
     * that arrives during the model load exits the same way as any other. */
    g_srv = &sv;
    static sigset_t stop_signals;
    sigemptyset(&stop_signals);
    sigaddset(&stop_signals, SIGTERM);
    sigaddset(&stop_signals, SIGINT);
    pthread_sigmask(SIG_BLOCK, &stop_signals, NULL);   /* before any thread starts */
    {
        pthread_t t;
        if (pthread_create(&t, NULL, signal_watch, &stop_signals) == 0) pthread_detach(t);
    }
    const char *host = "127.0.0.1";
    int port = 8080;
    bool cors = false;
    bool exit_on_eof = false;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if ((!strcmp(a, "-m") || !strcmp(a, "--model")) && i + 1 < argc) opts.model_path = argv[++i];
        else if (!strcmp(a, "--host") && i + 1 < argc) host = argv[++i];
        else if (!strcmp(a, "--port") && i + 1 < argc) port = atoi(argv[++i]);
        else if (!strcmp(a, "--ctx") && i + 1 < argc) so.ctx = atoi(argv[++i]);
        else if (!strcmp(a, "--live") && i + 1 < argc) so.live = atoi(argv[++i]);
        else if (!strcmp(a, "--state-dir") && i + 1 < argc) so.state_dir = argv[++i];
        else if (!strcmp(a, "--token") && i + 1 < argc) g_token = argv[++i];
        else if (!strcmp(a, "--max-tokens") && i + 1 < argc) sv.max_tokens = atoi(argv[++i]);
        else if (!strcmp(a, "--cors")) cors = true;
        else if (!strcmp(a, "--no-cache")) so.no_cache = true;
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
    /* The profile first: context is fixed when the engine loads, and the
     * profile is what decides it (API.md §4.1). */
    qw_profile prof;
    if (!qw_profile_derive(opts.model_path, 0, &prof, err, sizeof err)) {
        fprintf(stderr, "qwasar-server: %s\n", err);
        return 1;
    }
    opts.context_size = so.ctx > 0 ? so.ctx : prof.ctx;
    opts.verbose = sv.verbose;
    qw_log("loading %s", opts.model_path);
    const double t_load = qw_now();
    qwasar_engine *e = qwasar_engine_load(&opts, err, sizeof err);
    if (!e) { fprintf(stderr, "qwasar-server: %s\n", err); return 1; }
    qwasar_tokenizer *tok = qwasar_tokenizer_load(opts.model_path, err, sizeof err);
    if (!tok) { fprintf(stderr, "qwasar-server: %s\n", err); return 1; }
    /* What the engine took: the profile's context, or the override, never
     * past the model's window (the engine caps it the same way). */
    so.ctx = opts.context_size < prof.max_ctx ? opts.context_size : prof.max_ctx;
    so.verbose = sv.verbose;
    sv.st = qw_store_open(e, tok, opts.model_path, &prof, &so, err, sizeof err);
    if (!sv.st) { fprintf(stderr, "qwasar-server: %s\n", err); return 1; }
    model_id = qwasar_model_id(e);
    model_name = qwasar_model_name(e);
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

    char limit[64];
    if (sv.max_tokens > 0) snprintf(limit, sizeof limit, "%d tokens unless a request sets one", sv.max_tokens);
    else snprintf(limit, sizeof limit, "whatever the context has room for");
    qw_log("%s loaded in %.1fs; context %d tokens, %d live session%s, output limit %s, "
            "checkpoints %s, sessions in %s",
            model_name, qw_now() - t_load, qw_store_ctx(sv.st), prof.live, prof.live == 1 ? "" : "s",
            limit, so.no_cache ? "off" : "on", qw_store_state_dir(sv.st));
    qw_log("listening on http://%s:%d  (model %s)", host, port, model_id);

    pthread_mutex_init(&sv.conns_lock, NULL);
    atomic_store(&g_ready, true);
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

    qwasar_tokenizer_free(tok);
    qwasar_engine_free(e);
    close(ls);
    return 0;
}
