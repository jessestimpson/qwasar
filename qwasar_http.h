#ifndef QWASAR_HTTP_H
#define QWASAR_HTTP_H

#include "qwasar_json.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* The HTTP/1.1 core the server's fronts share: qwasar_server.c (the OpenAI
 * and Anthropic endpoints) and qwasar_api.c (the Session API).  Blocking
 * reads and writes on one connection per thread; nothing asynchronous. */

/* ---- growable text --------------------------------------------------------- */

typedef struct { char *p; size_t len, cap; } str;

bool str_add(str *s, const char *d, size_t n);
bool str_puts(str *s, const char *t);
void str_free(str *s);
/* Formats onto the end of `s`, whatever the length. */
void str_printf(str *s, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
/* Appends `t` as a JSON string, quotes included. */
void str_json(str *s, const char *t, size_t n);
void str_jsons(str *s, const char *t);
/* Re-serialises a parsed node as JSON. */
void str_node(str *s, const qj_doc *d, const qj_node *n);

/* ---- a connection ---------------------------------------------------------- */

typedef struct {
    int   fd;
    bool  cors;
    bool  streaming;   /* headers already sent, body is chunked */
    bool  dead;        /* the peer went away */
    bool  anthropic;   /* errors in Anthropic's envelope rather than OpenAI's */
} conn;

bool conn_write(conn *c, const char *data, size_t n);

void http_send(conn *c, int status, const char *reason,
               const char *ctype, const char *body, size_t len);
/* {"type": "error", "error": {"type": ..., "message": ...}} -- the body of an
 * Anthropic error response, and the data of an `error` event mid-stream. */
void anthropic_error_body(str *b, int status, const char *msg);
/* The error envelope for whichever API the connection speaks. */
void http_error_at(conn *c, int status, const char *reason, const char *msg,
                   const char *param, const char *code);
void http_error(conn *c, int status, const char *reason, const char *msg);

/* Server-sent events over chunked transfer, so the connection survives the
 * response and a client can reuse it. */
void sse_begin(conn *c);
void sse_chunk(conn *c, const char *data, size_t n);
void sse_event(conn *c, const char *event, const char *json);
/* The same with an `id:` line, so a client can resume with Last-Event-ID. */
void sse_event_id(conn *c, const char *id, const char *event, const char *json);
void sse_end(conn *c);

/* ---- requests -------------------------------------------------------------- */

typedef struct {
    char   method[8];
    char   path[512];
    size_t content_length;
    bool   keep_alive;
    bool   anthropic;      /* sent an anthropic-version header */
    bool   chunked;        /* Transfer-Encoding: chunked */
    bool   expect_continue;
    bool   too_large;      /* body over QW_MAX_BODY; left unread */
    char   bearer[128];    /* Authorization: Bearer <token>, or "" */
    char   last_event_id[32]; /* Last-Event-ID, for a reattaching stream, or "" */
} http_req;

/* Base64 images and video make for big bodies, but not this big. */
#define QW_MAX_BODY ((size_t)256 << 20)

/* Reads one request.  Returns false when the connection is finished or
 * malformed; `body` is left owning the payload. */
bool read_request(conn *c, str *carry, http_req *r, str *body);

/* ---- odds and ends --------------------------------------------------------- */

/* Base64 to bytes; padding and whitespace are skipped.  Caller frees. */
unsigned char *b64_decode(const char *src, size_t n, size_t *out_len);
/* A unique-enough id: prefix, the time, a counter. */
void gen_id(char *out, size_t cap, const char *prefix);

#endif /* QWASAR_HTTP_H */
