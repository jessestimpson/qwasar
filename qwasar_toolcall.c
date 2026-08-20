#include "qwasar_toolcall.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- tool calls ------------------------------------------------------------ */

static char *qw_dupn(const char *s, size_t n) {
    char *p = malloc(n + 1);
    if (!p) return NULL;
    memcpy(p, s, n);
    p[n] = 0;
    return p;
}

static const char *qw_find(const char *hay, const char *end, const char *needle) {
    size_t n = strlen(needle);
    if ((size_t)(end - hay) < n) return NULL;
    for (const char *p = hay; p + n <= end; p++)
        if (!memcmp(p, needle, n)) return p;
    return NULL;
}

bool qw_tool_call_complete(const char *text, size_t len) {
    return qw_find(text, text + len, "</tool_call>") != NULL;
}

/* Parses one <function=...>...</function> body into `call`. */
static bool qw_parse_function(const char *p, const char *end, qw_tool_call *call,
                              char *err, size_t errcap) {
    const char *fn = qw_find(p, end, "<function=");
    if (!fn) { snprintf(err, errcap, "tool call has no <function=...> block"); return false; }
    fn += strlen("<function=");

    const char *gt = memchr(fn, '>', (size_t)(end - fn));
    if (!gt) { snprintf(err, errcap, "unterminated <function= tag"); return false; }
    call->name = qw_dupn(fn, (size_t)(gt - fn));
    if (!call->name) { snprintf(err, errcap, "out of memory"); return false; }

    const char *cur = gt + 1;
    const char *fend = qw_find(cur, end, "</function>");
    if (!fend) fend = end;

    while (call->n_params < QW_MAX_PARAMS) {
        const char *ps = qw_find(cur, fend, "<parameter=");
        if (!ps) break;
        ps += strlen("<parameter=");
        const char *pgt = memchr(ps, '>', (size_t)(fend - ps));
        if (!pgt) { snprintf(err, errcap, "unterminated <parameter= tag"); return false; }

        /* The value begins after the newline that follows the opening tag and
         * ends at the newline before </parameter>, so a body that starts or
         * ends with blank lines keeps them. */
        const char *vs = pgt + 1;
        if (vs < fend && *vs == '\n') vs++;

        const char *pe = qw_find(vs, fend, "</parameter>");
        if (!pe) { snprintf(err, errcap, "parameter '%.*s' is not closed",
                            (int)(pgt - ps), ps); return false; }
        const char *ve = pe;
        if (ve > vs && ve[-1] == '\n') ve--;

        qw_tool_param *par = &call->params[call->n_params];
        par->key   = qw_dupn(ps, (size_t)(pgt - ps));
        par->value = qw_dupn(vs, (size_t)(ve - vs));
        if (!par->key || !par->value) { snprintf(err, errcap, "out of memory"); return false; }
        call->n_params++;

        cur = pe + strlen("</parameter>");
    }
    return true;
}

int qw_tool_parse(const char *text, qw_tool_calls *out, char *err, size_t errcap) {
    memset(out, 0, sizeof *out);
    if (!text) return 0;

    const char *end = text + strlen(text);
    const char *cur = text;

    const char *first = qw_find(cur, end, "<tool_call>");
    if (!first) return 0;

    /* Narration before the call is kept: the model is told it may reason in
     * natural language before a call, and that text is worth showing. */
    if (first > text) {
        const char *p = text, *q = first;
        while (p < q && (*p == '\n' || *p == ' ')) p++;
        while (q > p && (q[-1] == '\n' || q[-1] == ' ')) q--;
        if (q > p) out->preamble = qw_dupn(p, (size_t)(q - p));
    }

    while (out->n_calls < QW_MAX_CALLS) {
        const char *ts = qw_find(cur, end, "<tool_call>");
        if (!ts) break;
        ts += strlen("<tool_call>");

        const char *te = qw_find(ts, end, "</tool_call>");
        const char *body_end = te ? te : end;

        if (!qw_parse_function(ts, body_end, &out->calls[out->n_calls], err, errcap)) {
            qw_tool_calls_free(out);
            return -1;
        }
        out->n_calls++;
        cur = te ? te + strlen("</tool_call>") : end;
    }
    return out->n_calls;
}

void qw_tool_calls_free(qw_tool_calls *c) {
    if (!c) return;
    for (int i = 0; i < c->n_calls; i++) {
        free(c->calls[i].name);
        for (int j = 0; j < c->calls[i].n_params; j++) {
            free(c->calls[i].params[j].key);
            free(c->calls[i].params[j].value);
        }
    }
    free(c->preamble);
    memset(c, 0, sizeof *c);
}

const char *qw_tool_arg(const qw_tool_call *c, const char *key) {
    for (int i = 0; i < c->n_params; i++)
        if (!strcmp(c->params[i].key, key)) return c->params[i].value;
    return NULL;
}

/* ---- file editing ---------------------------------------------------------- */

const char *qw_edit_status_text(qw_edit_status s) {
    switch (s) {
    case QW_EDIT_OK:        return "ok";
    case QW_EDIT_NOT_FOUND: return "the old text was not found in the file";
    case QW_EDIT_AMBIGUOUS: return "the old text matches more than one place";
    case QW_EDIT_EMPTY_OLD: return "the old text is empty";
    case QW_EDIT_NOMEM:     return "out of memory";
    }
    return "unknown";
}

typedef struct { size_t off, len; } qw_span;   /* one line, excluding its newline */

/* Splits into lines on '\n'.  A trailing newline does not produce a final empty
 * line, so "a\nb\n" is two lines, matching how a person counts them. */
static size_t qw_split_lines(const char *s, size_t len, qw_span **out) {
    size_t cap = 64, n = 0;
    qw_span *v = malloc(cap * sizeof *v);
    if (!v) return 0;

    size_t start = 0;
    for (size_t i = 0; i <= len; i++) {
        if (i == len || s[i] == '\n') {
            if (i == len && start == len && n > 0) break;   /* trailing newline */
            if (n == cap) {
                cap *= 2;
                qw_span *nv = realloc(v, cap * sizeof *v);
                if (!nv) { free(v); return 0; }
                v = nv;
            }
            v[n].off = start;
            v[n].len = i - start;
            n++;
            start = i + 1;
        }
    }
    *out = v;
    return n;
}

static bool qw_line_eq(const char *a, qw_span sa, const char *b, qw_span sb) {
    return sa.len == sb.len && !memcmp(a + sa.off, b + sb.off, sa.len);
}

qw_edit_status qw_edit_apply(const char *content, size_t content_len,
                             const char *old_text, const char *new_text,
                             char **out, size_t *out_len, int *matches) {
    if (matches) *matches = 0;
    if (!old_text || !*old_text) return QW_EDIT_EMPTY_OLD;

    size_t old_len = strlen(old_text);
    /* A trailing newline on `old` is presentation, not content; drop it so the
     * model can quote a block with or without one and get the same result. */
    while (old_len > 0 && old_text[old_len - 1] == '\n') old_len--;
    if (old_len == 0) return QW_EDIT_EMPTY_OLD;

    qw_span *flines = NULL, *olines = NULL;
    size_t nf = qw_split_lines(content, content_len, &flines);
    size_t no = qw_split_lines(old_text, old_len, &olines);
    if (!flines || !olines || no == 0) {
        free(flines); free(olines);
        return QW_EDIT_NOMEM;
    }

    /* Match whole lines only.  Anchoring to line boundaries is what stops a
     * quoted fragment from matching the middle of a longer line and producing
     * a mangled result. */
    size_t found_at = 0;
    int n_found = 0;
    for (size_t i = 0; no <= nf && i + no <= nf; i++) {
        bool eq = true;
        for (size_t j = 0; j < no && eq; j++)
            eq = qw_line_eq(content, flines[i + j], old_text, olines[j]);
        if (eq) {
            if (n_found == 0) found_at = i;
            n_found++;
            if (n_found > 1) break;     /* ambiguity is decided; stop looking */
        }
    }
    if (matches) *matches = n_found;

    if (n_found == 0) { free(flines); free(olines); return QW_EDIT_NOT_FOUND; }
    if (n_found > 1)  { free(flines); free(olines); return QW_EDIT_AMBIGUOUS; }

    const size_t a = flines[found_at].off;
    size_t b = flines[found_at + no - 1].off + flines[found_at + no - 1].len;

    size_t new_len = new_text ? strlen(new_text) : 0;
    /* Deleting lines should remove them, not leave a blank one behind, so an
     * empty replacement takes the line terminator with it. */
    if (new_len == 0 && b < content_len && content[b] == '\n') b++;

    size_t total = a + new_len + (content_len - b);
    char *buf = malloc(total + 1);
    if (!buf) { free(flines); free(olines); return QW_EDIT_NOMEM; }

    memcpy(buf, content, a);
    if (new_len) memcpy(buf + a, new_text, new_len);
    memcpy(buf + a + new_len, content + b, content_len - b);
    buf[total] = 0;

    *out = buf;
    *out_len = total;
    free(flines);
    free(olines);
    return QW_EDIT_OK;
}
