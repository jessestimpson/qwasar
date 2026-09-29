/* qwasar_api -- the Session API's endpoints (API.md).
 *
 * Thin on purpose: requests are parsed into the store's types, the store
 * runs the step and emits events, this file writes them as server-sent
 * events.  Everything about sessions, warmth and the queue lives in
 * qwasar_sessions.c, where the compat endpoints share it. */

#include "qwasar_api.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* {"error": {"code": ..., "message": ...}} */
static void api_error(conn *c, int status, const char *code, const char *msg) {
    static const char *reasons[] = { "Bad Request", "Not Found", "Conflict",
                                     "Internal Server Error", "Service Unavailable" };
    const char *reason = status == 400 ? reasons[0] : status == 404 ? reasons[1]
                       : status == 409 ? reasons[2] : status == 503 ? reasons[4] : reasons[3];
    str b = { 0 };
    str_printf(&b, "{\"error\": {\"code\": \"%s\", \"message\": ", code);
    str_jsons(&b, msg);
    str_puts(&b, "}}");
    http_send(c, status, reason, "application/json", b.p, b.len);
    str_free(&b);
}

static void api_json(conn *c, int status, const str *b) {
    http_send(c, status, status == 201 ? "Created" : "OK", "application/json", b->p, b->len);
}

/* ---- describe ---------------------------------------------------------------- */

static void info_json(str *b, const qw_sess_info *i) {
    str_puts(b, "{\"id\": ");
    str_jsons(b, i->id);
    str_printf(b, ", \"metadata\": %s, \"created\": %lld, \"model\": ", i->metadata, (long long)i->created);
    str_jsons(b, i->model_id);
    if (i->model_mismatch) str_puts(b, ", \"model_mismatch\": true");
    str_printf(b, ", \"tokens\": %d, \"context\": %d, \"prefix_tokens\": %d, \"steps\": %d, "
                  "\"state\": \"%s\", \"warmth\": {\"state\": \"%s\", \"covered\": %d",
               i->n_tokens, i->ctx, i->prefix_tokens, i->step,
               qw_sess_state_name(i->state), qw_warmth_name(i->warmth), i->covered);
    if (i->warmth != QW_WARMTH_LIVE) str_printf(b, ", \"estimate_seconds\": %.1f", i->estimate_s);
    str_puts(b, "}");
    str_printf(b, ", \"checkpoint_bytes\": %llu", (unsigned long long)i->checkpoint_bytes);
    if (i->last_stop) {
        str_printf(b, ", \"last_step\": {\"stop\": \"%s\", \"at\": %lld, \"pending_calls\": [",
                   i->last_stop, (long long)i->last_at);
        for (int k = 0; k < i->n_pending; k++) {
            if (k) str_puts(b, ", ");
            str_printf(b, "{\"id\": \"%s\", \"name\": ", i->pending[k].id);
            str_jsons(b, i->pending[k].name);
            str_puts(b, "}");
        }
        str_puts(b, "]}");
    }
    str_puts(b, "}");
}

static void handle_server(qw_store *st, conn *c) {
    const qw_profile *p = qw_store_profile(st);
    qwasar_engine *e = qw_store_engine(st);
    int total = 0, live = 0, queued = 0;
    qw_store_counts(st, &total, &live, &queued);
    const bool vision = !strcmp(p->family, "qwen3_5");
    str b = { 0 };
    str_printf(&b, "{\"model\": {\"id\": \"%s\", \"name\": \"%s\", \"family\": \"%s\", \"path\": ",
               qwasar_model_id(e), qwasar_model_name(e), p->family);
    str_jsons(&b, qw_store_model_path(st));
    str_printf(&b, "}, \"context\": %d, \"live_sessions\": %d, "
                   "\"profile\": {\"physical_bytes\": %llu, \"working_set_bytes\": %llu, "
                   "\"weights_bytes\": %llu, \"kv_bytes_per_token\": %llu, "
                   "\"session_fixed_bytes\": %llu, \"reserve\": %.2f, \"max_context\": %d",
               qw_store_ctx(st), p->live,
               (unsigned long long)p->physical, (unsigned long long)p->working_set,
               (unsigned long long)p->weights, (unsigned long long)p->kv_per_token,
               (unsigned long long)p->fixed, p->reserve, p->max_ctx);
    if (p->note[0]) { str_puts(&b, ", \"note\": "); str_jsons(&b, p->note); }
    str_printf(&b, "}, \"capabilities\": {\"reasoning\": true, \"images\": %s, \"video\": %s, "
                   "\"speculation\": false, \"rewind\": false, \"fork\": false}, "
                   "\"sessions\": {\"total\": %d, \"live\": %d, \"queued\": %d}, \"state_dir\": ",
               vision ? "true" : "false", vision ? "true" : "false", total, live, queued);
    str_jsons(&b, qw_store_state_dir(st));
    uint64_t sb = 0, cb = 0, fb = 0;
    qw_store_disk(st, &sb, &cb, &fb);
    str_printf(&b, ", \"disk\": {\"sessions_bytes\": %llu, \"cache_bytes\": %llu, \"free_bytes\": %llu}}",
               (unsigned long long)sb, (unsigned long long)cb, (unsigned long long)fb);
    api_json(c, 200, &b);
    str_free(&b);
}

/* ---- open ---------------------------------------------------------------------- */

static void handle_open(qw_store *st, conn *c, const qj_doc *d) {
    const qj_node *root = qj_root(d);
    if (!root || root->type != QJ_OBJECT) { api_error(c, 400, "bad_request", "the body must be a JSON object"); return; }
    char *system = qj_strdup(d, qj_get(d, root, "system"));
    const qj_node *tools = qj_get(d, root, "tools");
    if (tools && tools->type != QJ_ARRAY) { api_error(c, 400, "bad_request", "tools must be an array"); return; }
    const int32_t n_tools = (int32_t)qj_count(tools);
    if (n_tools > QW_MAX_TOOLS) { api_error(c, 400, "bad_request", "too many tools"); return; }
    /* Each tool re-serialised: the parser unescaped the text in place. */
    str tool_text[QW_MAX_TOOLS];
    const char *tool_ptr[QW_MAX_TOOLS];
    int32_t k = 0;
    for (const qj_node *t = qj_first(d, tools); t && k < n_tools; t = qj_next(d, t), k++) {
        memset(&tool_text[k], 0, sizeof tool_text[k]);
        str_node(&tool_text[k], d, t);
        tool_ptr[k] = tool_text[k].p;
    }
    const bool thinking = qj_bool_or(d, root, "thinking", true);
    char effort[16] = "";
    if (!qj_str_copy(d, root, "effort", effort, sizeof effort)) snprintf(effort, sizeof effort, "xhigh");
    if (!strcmp(effort, "high")) snprintf(effort, sizeof effort, "xhigh");
    str meta = { 0 };
    const qj_node *md = qj_get(d, root, "metadata");
    if (md && md->type == QJ_OBJECT) str_node(&meta, d, md);

    char err[512];
    qw_sess *s = qw_store_open_session(st, system ? system : "", tool_ptr, n_tools,
                                       thinking, effort, meta.p, err, sizeof err);
    for (int32_t i = 0; i < k; i++) str_free(&tool_text[i]);
    str_free(&meta);
    free(system);
    if (!s) { api_error(c, 400, "bad_request", err); return; }

    qw_sess_info i;
    qw_sess_info_get(st, s, &i);
    str b = { 0 };
    str_printf(&b, "{\"id\": \"%s\", \"prefix_tokens\": %d, \"context\": %d, "
                   "\"warmth\": {\"state\": \"cold\", \"covered\": 0}, \"created\": %lld}",
               i.id, i.prefix_tokens, i.ctx, (long long)i.created);
    api_json(c, 201, &b);
    str_free(&b);
}

/* ---- steps ------------------------------------------------------------------------ */

/* The stream is begun by its first event, not before the step is admitted:
 * a refusal is an ordinary HTTP status (API.md §3), not an event. */
typedef struct { conn *c; bool begun; bool lost; } sse_sink;

static bool emit_sse(void *ud, const char *id, const char *event, const char *json) {
    sse_sink *k = ud;
    if (!k->begun) { sse_begin(k->c); k->begun = true; }
    sse_event_id(k->c, id, event, json);
    /* Logged whether or not -v: a client that stops hearing a step is the
     * one failure it cannot report itself.  The step goes on regardless, and
     * its events stay buffered for GET .../events with Last-Event-ID. */
    if (k->c->dead && !k->lost) {
        k->lost = true;
        qw_log("  the stream to the client was lost at event %s (%s): %s; "
               "the step continues, and its events can be reattached",
               id ? id : "?", event, k->c->err ? strerror(k->c->err) : "peer closed");
    }
    return !k->c->dead;
}

static void sink_close(sse_sink *k) {
    if (k->begun) sse_end(k->c);
}

static void read_sampling(const qj_doc *d, const qj_node *root, qwasar_sampling *sp) {
    qwasar_sampling_defaults(sp);
    const qj_node *s = qj_get(d, root, "sampling");
    if (!s) return;
    sp->temperature = (float)qj_num_or(d, s, "temperature", sp->temperature);
    sp->top_k = (int32_t)qj_int_or(d, s, "top_k", sp->top_k);
    sp->top_p = (float)qj_num_or(d, s, "top_p", sp->top_p);
    sp->min_p = (float)qj_num_or(d, s, "min_p", sp->min_p);
    sp->seed = (uint64_t)qj_int_or(d, s, "seed", 0);
}

static void handle_turn(qw_store *st, qw_sess *s, conn *c, const qj_doc *d) {
    const qj_node *root = qj_root(d);
    if (!root || root->type != QJ_OBJECT) { api_error(c, 400, "bad_request", "the body must be a JSON object"); return; }
    qw_turn t;
    memset(&t, 0, sizeof t);
    char *text = qj_strdup(d, qj_get(d, root, "text"));
    t.text = text;
    read_sampling(d, root, &t.sampling);
    t.max_tokens = (int32_t)qj_int_or(d, root, "max_tokens", 0);

    qw_attachment att[8];
    unsigned char *raw[8] = { 0 };
    char kinds[8][8], types[8][32];
    int32_t n_att = 0;
    const qj_node *images = qj_get(d, root, "images");
    for (const qj_node *im = qj_first(d, images); im; im = qj_next(d, im)) {
        if (n_att >= 8) { api_error(c, 400, "bad_request", "at most 8 attachments per turn"); goto out; }
        const qj_node *data = qj_get(d, im, "data");
        if (!data || data->type != QJ_STRING) { api_error(c, 400, "bad_request", "an attachment needs base64 data"); goto out; }
        size_t len = 0;
        raw[n_att] = b64_decode(d->text + data->u.str.off, data->u.str.len, &len);
        if (!raw[n_att]) { api_error(c, 500, "server_error", "out of memory"); goto out; }
        if (!qj_str_copy(d, im, "kind", kinds[n_att], sizeof kinds[0])) snprintf(kinds[n_att], sizeof kinds[0], "image");
        if (!qj_str_copy(d, im, "media_type", types[n_att], sizeof types[0])) types[n_att][0] = 0;
        att[n_att].kind = kinds[n_att];
        att[n_att].media_type = types[n_att][0] ? types[n_att] : NULL;
        att[n_att].bytes = raw[n_att];
        att[n_att].len = len;
        n_att++;
    }
    t.attachments = att;
    t.n_attachments = n_att;

    {
        int status = 400;
        char err[512] = "";
        sse_sink k = { c, false, false };
        if (!qw_sess_turn(st, s, &t, emit_sse, &k, &status, err, sizeof err))
            api_error(c, status, status == 409 ? "conflict" : "bad_request", err);
        sink_close(&k);
    }
out:
    for (int32_t i = 0; i < n_att; i++) free(raw[i]);
    free(text);
}

static void handle_continue(qw_store *st, qw_sess *s, conn *c, const qj_doc *d) {
    const qj_node *root = qj_root(d);
    if (!root || root->type != QJ_OBJECT) { api_error(c, 400, "bad_request", "the body must be a JSON object"); return; }
    const qj_node *results = qj_get(d, root, "results");
    if (!results || results->type != QJ_ARRAY) { api_error(c, 400, "bad_request", "results must be an array"); return; }
    qw_tool_result rs[QW_MAX_CALLS];
    char *owned[QW_MAX_CALLS * 2] = { 0 };
    int n = 0;
    bool bad = false;
    for (const qj_node *r = qj_first(d, results); r && !bad; r = qj_next(d, r)) {
        if (n >= QW_MAX_CALLS) { api_error(c, 400, "bad_request", "too many results"); bad = true; break; }
        owned[2 * n] = qj_strdup(d, qj_get(d, r, "id"));
        owned[2 * n + 1] = qj_strdup(d, qj_get(d, r, "content"));
        rs[n].id = owned[2 * n];
        rs[n].content = owned[2 * n + 1] ? owned[2 * n + 1] : "";
        if (!rs[n].id) { api_error(c, 400, "bad_request", "each result needs the call's id"); bad = true; break; }
        n++;
    }
    if (bad) { for (int i = 0; i < 2 * QW_MAX_CALLS; i++) free(owned[i]); return; }
    qwasar_sampling sp;
    read_sampling(d, root, &sp);
    const int32_t max_tokens = (int32_t)qj_int_or(d, root, "max_tokens", 0);

    int status = 400;
    char err[512] = "";
    sse_sink k = { c, false, false };
    if (!qw_sess_continue(st, s, rs, n, &sp, max_tokens, emit_sse, &k, &status, err, sizeof err))
        api_error(c, status, status == 409 ? "conflict" : "bad_request", err);
    sink_close(&k);
    for (int i = 0; i < 2 * QW_MAX_CALLS; i++) free(owned[i]);
}

/* ---- routing ---------------------------------------------------------------------- */

bool qw_api_handle(qw_store *st, conn *c, const http_req *r, const str *body) {
    if (!strcmp(r->path, "/v1/server")) {
        if (strcmp(r->method, "GET")) { api_error(c, 400, "bad_request", "GET /v1/server"); return true; }
        handle_server(st, c);
        return true;
    }
    if (strncmp(r->path, "/v1/sessions", 12)) return false;
    const char *rest = r->path + 12;

    if (!*rest || !strcmp(rest, "/")) {
        if (!strcmp(r->method, "GET")) {
            qw_sess *list[1024];
            const int n = qw_store_list(st, list, 1024);
            str b = { 0 };
            str_puts(&b, "{\"sessions\": [");
            for (int i = 0; i < n; i++) {
                qw_sess_info info;
                qw_sess_info_get(st, list[i], &info);
                if (i) str_puts(&b, ", ");
                info_json(&b, &info);
            }
            str_puts(&b, "]}");
            api_json(c, 200, &b);
            str_free(&b);
            return true;
        }
        if (!strcmp(r->method, "POST")) {
            qj_doc d;
            if (!qj_parse(&d, body->p ? body->p : "", body->len)) api_error(c, 400, "bad_request", d.err);
            else handle_open(st, c, &d);
            qj_free(&d);
            return true;
        }
        api_error(c, 400, "bad_request", "GET or POST /v1/sessions");
        return true;
    }
    if (*rest != '/') return false;
    rest++;
    char id[32];
    size_t k = 0;
    while (rest[k] && rest[k] != '/' && k + 1 < sizeof id) { id[k] = rest[k]; k++; }
    id[k] = 0;
    const char *verb = rest[k] == '/' ? rest + k + 1 : "";

    qw_sess *s = qw_store_find(st, id);
    if (!s) { api_error(c, 404, "not_found", "no such session"); return true; }

    if (!*verb) {
        if (!strcmp(r->method, "GET")) {
            qw_sess_info info;
            qw_sess_info_get(st, s, &info);
            str b = { 0 };
            info_json(&b, &info);
            api_json(c, 200, &b);
            str_free(&b);
            return true;
        }
        if (!strcmp(r->method, "DELETE")) {
            char err[256];
            if (!qw_store_delete(st, s, err, sizeof err)) api_error(c, 409, "conflict", err);
            else http_send(c, 204, "No Content", "text/plain", "", 0);
            return true;
        }
        api_error(c, 400, "bad_request", "GET or DELETE a session");
        return true;
    }

    const bool post = !strcmp(r->method, "POST");
    if (!strcmp(verb, "checkpoint") && !strcmp(r->method, "DELETE")) {
        uint64_t freed = 0;
        char err[256];
        if (!qw_sess_drop_checkpoint(st, s, &freed, err, sizeof err)) {
            api_error(c, strstr(err, "running") ? 409 : 500, strstr(err, "running") ? "conflict" : "server_error", err);
            return true;
        }
        str b = { 0 };
        str_printf(&b, "{\"freed_bytes\": %llu}", (unsigned long long)freed);
        api_json(c, 200, &b);
        str_free(&b);
        return true;
    }
    if (!strcmp(verb, "events") && !strcmp(r->method, "GET")) {
        sse_sink k = { c, false, false };
        if (!qw_sess_reattach(st, s, r->last_event_id[0] ? r->last_event_id : NULL, emit_sse, &k))
            api_error(c, 404, "not_found", "no step to reattach to");
        sink_close(&k);
        return true;
    }
    if (!post) { api_error(c, 400, "bad_request", "POST for a session verb"); return true; }

    if (!strcmp(verb, "cancel")) {
        const bool did = qw_sess_cancel(st, s);
        const char *b = did ? "{\"cancelled\": true}" : "{\"cancelled\": false}";
        http_send(c, 200, "OK", "application/json", b, strlen(b));
        return true;
    }
    if (!strcmp(verb, "park")) {
        char err[256];
        if (!qw_sess_park(st, s, err, sizeof err)) {
            api_error(c, strstr(err, "running") ? 409 : 500, strstr(err, "running") ? "conflict" : "server_error", err);
            return true;
        }
        qw_sess_info info;
        qw_sess_info_get(st, s, &info);
        str b = { 0 };
        str_printf(&b, "{\"warmth\": {\"state\": \"%s\", \"covered\": %d", qw_warmth_name(info.warmth), info.covered);
        if (info.warmth != QW_WARMTH_LIVE) str_printf(&b, ", \"estimate_seconds\": %.1f", info.estimate_s);
        str_puts(&b, "}}");
        api_json(c, 200, &b);
        str_free(&b);
        return true;
    }
    if (!strcmp(verb, "turn") || !strcmp(verb, "continue")) {
        qj_doc d;
        if (!qj_parse(&d, body->p ? body->p : "", body->len)) {
            api_error(c, 400, "bad_request", d.err);
        } else if (!strcmp(verb, "turn")) {
            handle_turn(st, s, c, &d);
        } else {
            handle_continue(st, s, c, &d);
        }
        qj_free(&d);
        return true;
    }
    api_error(c, 404, "not_found", "no such verb; turn, continue, events, cancel, park, checkpoint");
    return true;
}
