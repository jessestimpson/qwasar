/* qwasar_http -- the HTTP/1.1 core shared by the server's fronts.
 *
 * One reader, one writer, server-sent events over chunked transfer, a growable
 * string with JSON escaping, and base64.  Lifted verbatim from
 * qwasar_server.c so that the Session API (qwasar_api.c) and the OpenAI and
 * Anthropic endpoints speak HTTP the same way; nothing here knows what a
 * model is. */

#include "qwasar_http.h"

#include <arpa/inet.h>
#include <errno.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <stdarg.h>
#include <sys/socket.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <time.h>
#include <unistd.h>

/* ---- growable text --------------------------------------------------------- */

bool str_add(str *s, const char *d, size_t n) {
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
bool str_puts(str *s, const char *t) { return t ? str_add(s, t, strlen(t)) : true; }
void str_free(str *s) { free(s->p); s->p = NULL; s->len = s->cap = 0; }

/* Formats onto the end of `s`, whatever the length: on the stack when it
 * fits, else sized and formatted again on the heap.  (It once cut at 2 KB,
 * silently -- which truncated any SSE event carrying a large tool call.) */
void str_printf(str *s, const char *fmt, ...) {
    char buf[2048];
    va_list ap, again;
    va_start(ap, fmt);
    va_copy(again, ap);
    const int n = vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    if (n > 0 && (size_t)n < sizeof buf) {
        str_add(s, buf, (size_t)n);
    } else if (n > 0) {
        char *big = malloc((size_t)n + 1);
        if (big) {
            vsnprintf(big, (size_t)n + 1, fmt, again);
            str_add(s, big, (size_t)n);
            free(big);
        }
    }
    va_end(again);
}

/* Appends `t` as a JSON string, quotes included. */
void str_json(str *s, const char *t, size_t n) {
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
void str_jsons(str *s, const char *t) { str_json(s, t ? t : "", t ? strlen(t) : 0); }

/* Re-serialises a parsed node.  Tool schemas arrive as JSON and have to reach
 * the model's prompt as JSON; the parser unescapes strings in place, so the
 * original bytes are gone by then and the node has to be written back out. */
void str_node(str *s, const qj_doc *d, const qj_node *n) {
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

bool conn_write(conn *c, const char *data, size_t n) {
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

void http_send(conn *c, int status, const char *reason,
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
void anthropic_error_body(str *b, int status, const char *msg) {
    str_printf(b, "{\"type\": \"error\", \"error\": {\"type\": \"%s\", \"message\": ",
               anthropic_error_type(status));
    str_jsons(b, msg);
    str_puts(b, "}}");
}

/* The error envelope for whichever API the client speaks.  OpenAI's carries
 * message, type, and the param and code a client switches on -- each a string
 * or null, and always present. */
void http_error_at(conn *c, int status, const char *reason, const char *msg,
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

void http_error(conn *c, int status, const char *reason, const char *msg) {
    http_error_at(c, status, reason, msg, NULL, NULL);
}

/* Server-sent events over chunked transfer, so the connection survives the
 * response and a client can reuse it. */
void sse_begin(conn *c) {
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

void sse_chunk(conn *c, const char *data, size_t n) {
    char head[32];
    int hn = snprintf(head, sizeof head, "%zx\r\n", n);
    conn_write(c, head, (size_t)hn);
    conn_write(c, data, n);
    conn_write(c, "\r\n", 2);
}

void sse_event(conn *c, const char *event, const char *json) {
    str f = { 0 };
    if (event) { str_puts(&f, "event: "); str_puts(&f, event); str_puts(&f, "\n"); }
    str_puts(&f, "data: ");
    str_puts(&f, json);
    str_puts(&f, "\n\n");
    sse_chunk(c, f.p, f.len);
    str_free(&f);
}

void sse_event_id(conn *c, const char *id, const char *event, const char *json) {
    str f = { 0 };
    if (id) { str_puts(&f, "id: "); str_puts(&f, id); str_puts(&f, "\n"); }
    if (event) { str_puts(&f, "event: "); str_puts(&f, event); str_puts(&f, "\n"); }
    str_puts(&f, "data: ");
    str_puts(&f, json);
    str_puts(&f, "\n\n");
    sse_chunk(c, f.p, f.len);
    str_free(&f);
}

void sse_end(conn *c) {
    conn_write(c, "0\r\n\r\n", 5);
    c->streaming = false;
}

static int b64_value(unsigned char ch) {
    if (ch >= 'A' && ch <= 'Z') return ch - 'A';
    if (ch >= 'a' && ch <= 'z') return ch - 'a' + 26;
    if (ch >= '0' && ch <= '9') return ch - '0' + 52;
    if (ch == '+') return 62;
    if (ch == '/') return 63;
    return -1;                      /* padding and whitespace are skipped */
}

unsigned char *b64_decode(const char *src, size_t n, size_t *out_len) {
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

void gen_id(char *out, size_t cap, const char *prefix) {
    static uint64_t counter;
    const uint64_t k = __atomic_add_fetch(&counter, 1, __ATOMIC_RELAXED);
    snprintf(out, cap, "%s%08llx%04llx", prefix,
             (unsigned long long)time(NULL), (unsigned long long)(k & 0xffff));
}

/* ---- request loop ----------------------------------------------------------- */

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
bool read_request(conn *c, str *carry, http_req *r, str *body) {
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
            } else if (len > 14 && !strncasecmp(line, "Authorization:", 14)) {
                size_t i = 14;
                while (i < len && line[i] == ' ') i++;
                if (len - i > 7 && !strncasecmp(line + i, "Bearer ", 7)) {
                    i += 7;
                    size_t k = 0;
                    while (i < len && k + 1 < sizeof r->bearer && line[i] != ' ') r->bearer[k++] = line[i++];
                    r->bearer[k] = 0;
                }
            } else if (len > 14 && !strncasecmp(line, "Last-Event-ID:", 14)) {
                size_t i = 14;
                while (i < len && line[i] == ' ') i++;
                size_t k = 0;
                while (i < len && k + 1 < sizeof r->last_event_id && line[i] != ' ')
                    r->last_event_id[k++] = line[i++];
                r->last_event_id[k] = 0;
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

/* ---- the client side ---------------------------------------------------------- */

char *b64_encode(const unsigned char *src, size_t n) {
    static const char tab[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    char *out = malloc((n + 2) / 3 * 4 + 1);
    if (!out) return NULL;
    size_t o = 0;
    for (size_t i = 0; i < n; i += 3) {
        const uint32_t a = src[i], b = i + 1 < n ? src[i + 1] : 0, d = i + 2 < n ? src[i + 2] : 0;
        const uint32_t v = (a << 16) | (b << 8) | d;
        out[o++] = tab[(v >> 18) & 63];
        out[o++] = tab[(v >> 12) & 63];
        out[o++] = i + 1 < n ? tab[(v >> 6) & 63] : '=';
        out[o++] = i + 2 < n ? tab[v & 63] : '=';
    }
    out[o] = 0;
    return out;
}

int http_connect(const char *host, int port) {
    struct addrinfo hints, *res = NULL;
    memset(&hints, 0, sizeof hints);
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    char service[16];
    snprintf(service, sizeof service, "%d", port);
    if (getaddrinfo(host, service, &hints, &res) != 0) return -1;
    int fd = -1;
    for (struct addrinfo *ai = res; ai; ai = ai->ai_next) {
        fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) continue;
        if (connect(fd, ai->ai_addr, ai->ai_addrlen) == 0) break;
        close(fd);
        fd = -1;
    }
    freeaddrinfo(res);
    if (fd >= 0) {
        int one = 1;
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
        setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one);
    }
    return fd;
}

bool http_send_request(conn *c, const char *method, const char *path, const char *host,
                       const char *bearer, const char *extra, const char *body, size_t len) {
    str h = { 0 };
    str_printf(&h, "%s %s HTTP/1.1\r\nHost: %s\r\nConnection: close\r\n", method, path, host);
    if (bearer && *bearer) str_printf(&h, "Authorization: Bearer %s\r\n", bearer);
    if (extra) str_puts(&h, extra);
    if (body) str_printf(&h, "Content-Type: application/json\r\nContent-Length: %zu\r\n", len);
    str_puts(&h, "\r\n");
    bool ok = conn_write(c, h.p, h.len);
    if (ok && body && len) ok = conn_write(c, body, len);
    str_free(&h);
    return ok;
}

bool http_read_response(conn *c, str *carry, http_resp *r) {
    memset(r, 0, sizeof *r);
    r->close = true;
    const char *hend = NULL;
    for (;;) {
        if (carry->len && (hend = strstr(carry->p, "\r\n\r\n"))) break;
        if (carry->len > (1u << 20)) return false;
        char buf[8192];
        ssize_t n = read(c->fd, buf, sizeof buf);
        if (n <= 0) { if (n < 0 && errno == EINTR) continue; return false; }
        if (!str_add(carry, buf, (size_t)n)) return false;
    }
    const size_t head_len = (size_t)(hend - carry->p) + 4;
    if (sscanf(carry->p, "HTTP/1.%*d %d", &r->status) != 1) return false;
    const char *end = carry->p + head_len;
    const char *p = memchr(carry->p, '\n', head_len);
    while (p && p + 1 < end) {
        const char *line = p + 1;
        const char *nl = memchr(line, '\n', (size_t)(end - line));
        if (!nl) break;
        size_t len = (size_t)(nl - line);
        if (len && line[len - 1] == '\r') len--;
        if (len == 0) break;
        if (len > 15 && !strncasecmp(line, "Content-Length:", 15)) {
            r->content_length = (size_t)strtoul(line + 15, NULL, 10);
            r->close = false;
        } else if (len > 13 && !strncasecmp(line, "Content-Type:", 13)) {
            size_t i = 13;
            while (i < len && line[i] == ' ') i++;
            size_t k = 0;
            while (i < len && k + 1 < sizeof r->ctype) r->ctype[k++] = line[i++];
            r->ctype[k] = 0;
        } else if (len > 18 && !strncasecmp(line, "Transfer-Encoding:", 18)) {
            for (size_t i = 18; i + 7 <= len; i++)
                if (!strncasecmp(line + i, "chunked", 7)) { r->chunked = true; r->close = false; break; }
        }
        p = nl;
    }
    memmove(carry->p, carry->p + head_len, carry->len - head_len);
    carry->len -= head_len;
    carry->p[carry->len] = 0;
    return true;
}

bool http_read_body(conn *c, str *carry, const http_resp *r, str *body) {
    if (r->chunked) {
        size_t end = 0;
        if (!read_chunked(c, carry, 0, body, &end)) return false;
        return true;
    }
    if (!r->close) {
        if (!carry_fill(c, carry, r->content_length)) return false;
        return str_add(body, carry->p, r->content_length);
    }
    /* No framing: the body is everything to EOF. */
    str_add(body, carry->p, carry->len);
    char buf[8192];
    for (;;) {
        ssize_t n = read(c->fd, buf, sizeof buf);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return true;
        str_add(body, buf, (size_t)n);
    }
}

bool http_call(const char *host, int port, const char *bearer, const char *method,
               const char *path, const char *body, http_resp *r, str *out) {
    conn c = { .fd = http_connect(host, port) };
    if (c.fd < 0) return false;
    str carry = { 0 };
    bool ok = http_send_request(&c, method, path, host, bearer, NULL, body, body ? strlen(body) : 0)
           && http_read_response(&c, &carry, r)
           && http_read_body(&c, &carry, r, out);
    str_free(&carry);
    close(c.fd);
    return ok;
}

/* ---- server-sent events, decoded incrementally --------------------------------- */

void sse_reader_init(sse_reader *rd, bool chunked) {
    memset(rd, 0, sizeof *rd);
    rd->chunked = chunked;
}

void sse_reader_free(sse_reader *rd) {
    str_free(&rd->raw);
    str_free(&rd->text);
}

/* Undoes chunked framing from `raw` into `text`, as far as the bytes go. */
static void dechunk(sse_reader *rd) {
    size_t pos = 0;
    for (;;) {
        if (rd->remaining > 0) {
            const size_t have = rd->raw.len - pos;
            const size_t take = have < rd->remaining ? have : rd->remaining;
            str_add(&rd->text, rd->raw.p + pos, take);
            pos += take;
            rd->remaining -= take;
            if (rd->remaining > 0) break;
        }
        /* The CRLF that ends a chunk -- which the server writes separately
         * from the data, so a read may well stop between the two -- and any
         * blank line before the next size. */
        while (pos < rd->raw.len && (rd->raw.p[pos] == '\r' || rd->raw.p[pos] == '\n')) pos++;
        const char *nl = memchr(rd->raw.p + pos, '\n', rd->raw.len - pos);
        if (!nl) break;
        char *stop = NULL;
        const unsigned long long sz = strtoull(rd->raw.p + pos, &stop, 16);
        if (stop == rd->raw.p + pos) { rd->ended = true; break; }   /* not a chunk line */
        pos = (size_t)(nl - rd->raw.p) + 1;
        if (sz == 0) { rd->ended = true; break; }
        rd->remaining = (size_t)sz;
    }
    /* A chunk's data may straddle reads: keep what was not consumed.  A
     * partial size line is kept too, whole. */
    if (rd->remaining > 0 || pos > 0) {
        memmove(rd->raw.p, rd->raw.p + pos, rd->raw.len - pos);
        rd->raw.len -= pos;
        if (rd->raw.p) rd->raw.p[rd->raw.len] = 0;
    }
}

bool sse_reader_feed(sse_reader *rd, const char *bytes, size_t n) {
    if (!rd->chunked) return str_add(&rd->text, bytes, n);
    if (!str_add(&rd->raw, bytes, n)) return false;
    dechunk(rd);
    return true;
}

bool sse_reader_next(sse_reader *rd, const char **id, const char **event, const char **data) {
    static str s_id, s_event, s_data;
    /* An event ends at a blank line. */
    const char *p = rd->text.p;
    if (!p) return false;
    const char *end = strstr(p, "\n\n");
    if (!end) return false;
    s_id.len = s_event.len = s_data.len = 0;
    const char *line = p;
    while (line < end + 1) {
        const char *nl = memchr(line, '\n', (size_t)(end + 1 - line));
        if (!nl) break;
        size_t len = (size_t)(nl - line);
        if (len > 3 && !strncmp(line, "id:", 3)) {
            const char *v = line + 3; size_t vl = len - 3;
            if (vl && *v == ' ') { v++; vl--; }
            str_add(&s_id, v, vl);
        } else if (len > 6 && !strncmp(line, "event:", 6)) {
            const char *v = line + 6; size_t vl = len - 6;
            if (vl && *v == ' ') { v++; vl--; }
            str_add(&s_event, v, vl);
        } else if (len >= 5 && !strncmp(line, "data:", 5)) {
            const char *v = line + 5; size_t vl = len - 5;
            if (vl && *v == ' ') { v++; vl--; }
            if (s_data.len) str_add(&s_data, "\n", 1);
            str_add(&s_data, v, vl);
        }
        line = nl + 1;
    }
    const size_t consumed = (size_t)(end + 2 - p);
    memmove(rd->text.p, rd->text.p + consumed, rd->text.len - consumed);
    rd->text.len -= consumed;
    rd->text.p[rd->text.len] = 0;
    str_add(&s_id, "", 0); str_add(&s_event, "", 0); str_add(&s_data, "", 0);
    *id = s_id.p; *event = s_event.p; *data = s_data.p;
    return true;
}
