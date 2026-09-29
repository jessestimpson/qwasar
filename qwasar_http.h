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
/* A NUL-terminated copy of a string node, or NULL for anything else.
 * qj_str() points into the document and is NOT terminated -- the parser
 * unescapes in place and leaves the closing quote behind -- so a node used
 * as a C string goes through this.  Caller frees. */
char *qj_strdup(const qj_doc *d, const qj_node *n);

/* ---- a connection ---------------------------------------------------------- */

typedef struct {
    int   fd;
    bool  cors;
    bool  streaming;   /* headers already sent, body is chunked */
    bool  dead;        /* the peer went away */
    bool  anthropic;   /* errors in Anthropic's envelope rather than OpenAI's */
    int   err;         /* errno of the write that found the peer gone, or 0 */
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

/* ---- the client side --------------------------------------------------------
 *
 * For qwasar-agent, which speaks the Session API over the same code.  One
 * connection per request; a streaming response is read incrementally so the
 * caller can poll its own input between events. */

typedef struct {
    int    status;
    char   ctype[64];
    size_t content_length;
    bool   chunked;
    bool   close;          /* Connection: close, or no framing: read to EOF */
} http_resp;

/* Connects to host:port (an IPv4 literal or a name); -1 on failure. */
int  http_connect(const char *host, int port);
/* Writes one request.  `bearer` may be NULL; `extra` is raw header lines
 * ending in CRLF, or NULL. */
bool http_send_request(conn *c, const char *method, const char *path, const char *host,
                       const char *bearer, const char *extra, const char *body, size_t len);
/* Reads the status line and headers; the body is left in `carry`. */
bool http_read_response(conn *c, str *carry, http_resp *r);
/* Reads a whole body after http_read_response, framed however `r` says. */
bool http_read_body(conn *c, str *carry, const http_resp *r, str *body);
/* Everything at once: request, response, body.  Returns false when the
 * connection failed; the status is the server's otherwise. */
bool http_call(const char *host, int port, const char *bearer, const char *method,
               const char *path, const char *body, http_resp *r, str *out);

/* A server-sent event stream, decoded incrementally: feed it bytes off the
 * socket as they arrive, take events as they complete.  Chunked transfer is
 * undone here, so the caller reads the socket with plain read(). */
typedef struct {
    str    raw;        /* undecoded bytes, when chunked */
    str    text;       /* decoded stream, events consumed from its front */
    size_t remaining;  /* bytes left in the chunk being read */
    bool   chunked;
    bool   ended;      /* the terminating chunk arrived */
} sse_reader;

void sse_reader_init(sse_reader *rd, bool chunked);
void sse_reader_free(sse_reader *rd);
bool sse_reader_feed(sse_reader *rd, const char *bytes, size_t n);
/* The next complete event, or false.  `id` and `event` may come back empty;
 * `data` is the joined data lines.  The strings are the reader's until the
 * next call. */
bool sse_reader_next(sse_reader *rd, const char **id, const char **event, const char **data);

/* ---- odds and ends --------------------------------------------------------- */

/* Bytes to base64 (no line breaks).  Caller frees. */
char *b64_encode(const unsigned char *src, size_t n);

/* Base64 to bytes; padding and whitespace are skipped.  Caller frees. */
unsigned char *b64_decode(const char *src, size_t n, size_t *out_len);
/* A unique-enough id: prefix, the time, a counter. */
void gen_id(char *out, size_t cap, const char *prefix);

#endif /* QWASAR_HTTP_H */
