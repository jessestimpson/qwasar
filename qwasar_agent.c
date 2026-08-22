/* qwasar-agent -- an agentic loop on the qwasar engine.
 *
 * The conversation is append-only, which is not a simplification but a
 * requirement: 48 of this model's 64 layers carry recurrent state with no
 * per-position history, so a session can be extended but never rewound (see
 * qwasar.h).  An agent loop happens to be a natural fit -- every turn appends,
 * nothing edits history -- so each step feeds back exactly the tokens the model
 * just produced plus the rendered tool result, and the KV cache and recurrent
 * state carry forward untouched.  Re-rendering the whole conversation each turn
 * would be both slower and, for the recurrent half, wrong.
 *
 * Tools that only read run unattended.  Tools that write to the filesystem or
 * run commands ask first, unless --yes. */

#include "qwasar.h"
#include "qwasar_toolcall.h"
#include "qwasar_tui.h"

#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define AGENT_MAX_READ   (256 * 1024)
#define AGENT_MAX_OUTPUT (64 * 1024)

/* ---- tool definitions ------------------------------------------------------
 *
 * One JSON object per tool, embedded verbatim in the system turn.  The edit
 * description spells out the match rule because that rule is the tool's whole
 * contract: the model has to know that quoting too little will be rejected as
 * ambiguous rather than applied somewhere arbitrary. */
static const char *const AGENT_TOOLS[] = {
"{\"type\": \"function\", \"function\": {\"name\": \"read\", \"description\": "
"\"Read a file and return its exact contents, with no line numbers or other "
"decoration, so the text can be quoted back to edit.\", \"parameters\": {\"type\": "
"\"object\", \"properties\": {\"path\": {\"type\": \"string\", \"description\": "
"\"Path to the file.\"}}, \"required\": [\"path\"]}}}",

"{\"type\": \"function\", \"function\": {\"name\": \"write\", \"description\": "
"\"Create a file or replace its entire contents. Use edit for changes to an "
"existing file.\", \"parameters\": {\"type\": \"object\", \"properties\": {\"path\": "
"{\"type\": \"string\", \"description\": \"Path to the file.\"}, \"content\": "
"{\"type\": \"string\", \"description\": \"The complete new contents.\"}}, "
"\"required\": [\"path\", \"content\"]}}}",

"{\"type\": \"function\", \"function\": {\"name\": \"edit\", \"description\": "
"\"Replace a run of whole lines in a file. The old text must match complete "
"lines exactly once, including indentation; if it matches nowhere or in more "
"than one place the edit is refused and nothing changes. Quote enough "
"surrounding lines to be unique. To insert, set old to a unique nearby line and "
"new to that same line plus the addition. To delete, set new to an empty "
"string.\", \"parameters\": {\"type\": \"object\", \"properties\": {\"path\": "
"{\"type\": \"string\", \"description\": \"Path to the file.\"}, \"old\": {\"type\": "
"\"string\", \"description\": \"Exact existing lines to replace.\"}, \"new\": "
"{\"type\": \"string\", \"description\": \"Replacement lines.\"}}, \"required\": "
"[\"path\", \"old\", \"new\"]}}}",

"{\"type\": \"function\", \"function\": {\"name\": \"list\", \"description\": "
"\"List the entries of a directory.\", \"parameters\": {\"type\": \"object\", "
"\"properties\": {\"path\": {\"type\": \"string\", \"description\": \"Directory "
"path; defaults to the working directory.\"}}, \"required\": []}}}",

"{\"type\": \"function\", \"function\": {\"name\": \"grep\", \"description\": "
"\"Search files recursively for a regular expression and return matching lines "
"with their file and line number.\", \"parameters\": {\"type\": \"object\", "
"\"properties\": {\"pattern\": {\"type\": \"string\", \"description\": \"Extended "
"regular expression.\"}, \"path\": {\"type\": \"string\", \"description\": \"File "
"or directory to search; defaults to the working directory.\"}}, \"required\": "
"[\"pattern\"]}}}",

"{\"type\": \"function\", \"function\": {\"name\": \"bash\", \"description\": "
"\"Run a shell command and return its combined output and exit status.\", "
"\"parameters\": {\"type\": \"object\", \"properties\": {\"command\": {\"type\": "
"\"string\", \"description\": \"Command to run via /bin/sh.\"}}, \"required\": "
"[\"command\"]}}}",
};
#define AGENT_N_TOOLS ((int32_t)(sizeof AGENT_TOOLS / sizeof *AGENT_TOOLS))

/* ---- growable text --------------------------------------------------------- */

typedef struct { char *p; size_t len, cap; } str;

static bool str_add(str *s, const char *data, size_t n) {
    if (s->len + n + 1 > s->cap) {
        size_t cap = s->cap ? s->cap * 2 : 4096;
        while (cap < s->len + n + 1) cap *= 2;
        char *p = realloc(s->p, cap);
        if (!p) return false;
        s->p = p;
        s->cap = cap;
    }
    memcpy(s->p + s->len, data, n);
    s->len += n;
    s->p[s->len] = 0;
    return true;
}
static bool str_puts(str *s, const char *t) { return str_add(s, t, strlen(t)); }
static void str_free(str *s) { free(s->p); s->p = NULL; s->len = s->cap = 0; }

static void str_printf(str *s, const char *fmt, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    if (n > 0) str_add(s, buf, (size_t)n < sizeof buf ? (size_t)n : sizeof buf - 1);
}

/* ---- subprocess ------------------------------------------------------------
 *
 * argv is passed to execvp directly, so a pattern or path from the model is
 * never seen by a shell.  Only the bash tool goes through /bin/sh, and that one
 * asks for confirmation first. */
static bool run_capture(char *const argv[], str *out, int *status) {
    int fds[2];
    if (pipe(fds) != 0) return false;

    pid_t pid = fork();
    if (pid < 0) { close(fds[0]); close(fds[1]); return false; }
    if (pid == 0) {
        close(fds[0]);
        dup2(fds[1], STDOUT_FILENO);
        dup2(fds[1], STDERR_FILENO);
        close(fds[1]);
        execvp(argv[0], argv);
        _exit(127);
    }
    close(fds[1]);

    char buf[4096];
    ssize_t n;
    while ((n = read(fds[0], buf, sizeof buf)) > 0) {
        if (out->len < AGENT_MAX_OUTPUT) str_add(out, buf, (size_t)n);
    }
    close(fds[0]);

    int st = 0;
    waitpid(pid, &st, 0);
    *status = WIFEXITED(st) ? WEXITSTATUS(st) : -1;

    if (out->len >= AGENT_MAX_OUTPUT)
        str_puts(out, "\n[output truncated]");
    return true;
}

/* ---- tools ----------------------------------------------------------------- */

typedef struct {
    bool yes;          /* skip confirmations */
    bool show_think;
    int  max_steps;
    int  max_tokens;
    int  mtp_depth;    /* 0 = decode serially */
} agent_cfg;

/* Mutating tools ask before acting.  A refusal is reported back to the model as
 * a tool result rather than aborting, so it can choose something else. */
/* Asking mid-turn has to go through the same output path as everything else,
 * or the question lands on top of the pinned footer.  The reply is read with
 * the line editor for the same reason. */
static qw_tui *g_tui;

static bool confirm(const agent_cfg *cfg, const char *what, const char *detail) {
    if (cfg->yes) return true;
    const bool tty = tui_is_tty(g_tui);
    tui_printf(g_tui, "\n  %s%s%s %s\n", tty ? "\x1b[1;33m" : "", what,
               tty ? "\x1b[0m" : "", detail);
    char *ans = tui_readline(g_tui, "  proceed? [y/N] ");
    bool ok = ans && (ans[0] == 'y' || ans[0] == 'Y');
    free(ans);
    return ok;
}

static bool read_file(const char *path, str *out) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    char buf[8192];
    size_t n;
    while ((n = fread(buf, 1, sizeof buf, f)) > 0) {
        if (out->len >= AGENT_MAX_READ) { str_puts(out, "\n[file truncated]"); break; }
        str_add(out, buf, n);
    }
    fclose(f);
    return true;
}

static void tool_read(const qw_tool_call *c, str *result) {
    const char *path = qw_tool_arg(c, "path");
    if (!path) { str_puts(result, "error: read requires a path"); return; }
    if (!read_file(path, result))
        str_printf(result, "error: cannot read %s: %s", path, strerror(errno));
    else if (result->len == 0)
        str_puts(result, "[the file is empty]");
}

static void tool_write(const qw_tool_call *c, const agent_cfg *cfg, str *result) {
    const char *path = qw_tool_arg(c, "path");
    const char *content = qw_tool_arg(c, "content");
    if (!path || !content) { str_puts(result, "error: write requires path and content"); return; }

    char detail[512];
    snprintf(detail, sizeof detail, "write %zu bytes to %s", strlen(content), path);
    if (!confirm(cfg, "WRITE", detail)) {
        str_puts(result, "error: the user declined this write");
        return;
    }
    FILE *f = fopen(path, "wb");
    if (!f) { str_printf(result, "error: cannot open %s: %s", path, strerror(errno)); return; }
    size_t n = fwrite(content, 1, strlen(content), f);
    fclose(f);
    str_printf(result, "wrote %zu bytes to %s", n, path);
}

static void tool_edit(const qw_tool_call *c, const agent_cfg *cfg, str *result) {
    const char *path = qw_tool_arg(c, "path");
    const char *old  = qw_tool_arg(c, "old");
    const char *new  = qw_tool_arg(c, "new");
    if (!path || !old || !new) {
        str_puts(result, "error: edit requires path, old and new");
        return;
    }

    str content = { 0 };
    if (!read_file(path, &content)) {
        str_printf(result, "error: cannot read %s: %s", path, strerror(errno));
        str_free(&content);
        return;
    }

    char *edited = NULL;
    size_t edited_len = 0;
    int matches = 0;
    qw_edit_status st = qw_edit_apply(content.p ? content.p : "", content.len,
                                      old, new, &edited, &edited_len, &matches);
    str_free(&content);

    if (st != QW_EDIT_OK) {
        /* Told plainly, with the count, so the model knows whether to quote
         * more context or to go and look at the file again. */
        str_printf(result, "error: %s (%d matches). Nothing was changed. %s",
                   qw_edit_status_text(st), matches,
                   st == QW_EDIT_AMBIGUOUS
                       ? "Quote more surrounding lines to make it unique."
                       : "Read the file and quote the lines exactly, including indentation.");
        return;
    }

    char detail[512];
    snprintf(detail, sizeof detail, "replace lines in %s", path);
    if (!confirm(cfg, "EDIT", detail)) {
        free(edited);
        str_puts(result, "error: the user declined this edit");
        return;
    }

    FILE *f = fopen(path, "wb");
    if (!f) {
        str_printf(result, "error: cannot write %s: %s", path, strerror(errno));
        free(edited);
        return;
    }
    fwrite(edited, 1, edited_len, f);
    fclose(f);
    free(edited);
    str_printf(result, "edited %s", path);
}

static void tool_list(const qw_tool_call *c, str *result) {
    const char *path = qw_tool_arg(c, "path");
    char *argv[] = { "ls", "-lA", (char *)(path ? path : "."), NULL };
    int status = 0;
    if (!run_capture(argv, result, &status)) str_puts(result, "error: cannot run ls");
}

static void tool_grep(const qw_tool_call *c, str *result) {
    const char *pat = qw_tool_arg(c, "pattern");
    const char *path = qw_tool_arg(c, "path");
    if (!pat) { str_puts(result, "error: grep requires a pattern"); return; }
    char *argv[] = { "grep", "-rnE", "--", (char *)pat, (char *)(path ? path : "."), NULL };
    int status = 0;
    if (!run_capture(argv, result, &status)) { str_puts(result, "error: cannot run grep"); return; }
    if (status == 1 && result->len == 0) str_puts(result, "[no matches]");
}

static void tool_bash(const qw_tool_call *c, const agent_cfg *cfg, str *result) {
    const char *cmd = qw_tool_arg(c, "command");
    if (!cmd) { str_puts(result, "error: bash requires a command"); return; }
    if (!confirm(cfg, "RUN", cmd)) {
        str_puts(result, "error: the user declined to run this command");
        return;
    }
    char *argv[] = { "/bin/sh", "-c", (char *)cmd, NULL };
    int status = 0;
    if (!run_capture(argv, result, &status)) { str_puts(result, "error: cannot run the command"); return; }
    str_printf(result, "\n[exit status %d]", status);
}

static void dispatch(const qw_tool_call *c, const agent_cfg *cfg, str *result) {
    if      (!strcmp(c->name, "read"))  tool_read(c, result);
    else if (!strcmp(c->name, "write")) tool_write(c, cfg, result);
    else if (!strcmp(c->name, "edit"))  tool_edit(c, cfg, result);
    else if (!strcmp(c->name, "list"))  tool_list(c, result);
    else if (!strcmp(c->name, "grep"))  tool_grep(c, result);
    else if (!strcmp(c->name, "bash"))  tool_bash(c, cfg, result);
    else str_printf(result, "error: no tool named '%s'", c->name);
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ---- prefill progress ------------------------------------------------------
 *
 * Prompt processing is the one stretch of a turn with nothing to look at, and
 * on a long conversation it is the longest.  The rate is written into the
 * unfilled part of the bar, so the line stays one width whether or not there is
 * a number to show yet -- a trick worth stealing from ds4-agent. */

#define AGENT_BAR_WIDTH 28

/* Below this the prompt is processed faster than the eye can follow, and a bar
 * that appears and vanishes is worse than no bar.  A tool result is usually a
 * few dozen tokens; a conversation being reloaded is thousands. */
#define AGENT_BAR_MIN_TOKENS 128

/* ---- generation ------------------------------------------------------------ */

typedef struct {
    str      text;      /* everything after </think> */
    str      think;
    int32_t *tokens;    /* what the model produced, to feed straight back */
    int32_t  n_tokens;
    bool     hit_eos;
    /* Why the turn ended, which the caller has to be able to tell apart: a
     * finished answer and a truncated one look identical otherwise. */
    bool     hit_budget;
    bool     in_reasoning;   /* still inside <think> when it ended */
    int32_t  n_think;        /* tokens spent reasoning, which are not printed */
    bool     has_call;
} turn;

static void turn_free(turn *t) {
    str_free(&t->text);
    str_free(&t->think);
    free(t->tokens);
    memset(t, 0, sizeof *t);
}

static bool turn_push(turn *t, int32_t id) {
    int32_t *v = realloc(t->tokens, (size_t)(t->n_tokens + 1) * sizeof *v);
    if (!v) return false;
    t->tokens = v;
    t->tokens[t->n_tokens++] = id;
    return true;
}

static int32_t argmax(const float *v, int32_t n) {
    int32_t best = 0;
    for (int32_t i = 1; i < n; i++) if (v[i] > v[best]) best = i;
    return best;
}

/* Runs one assistant turn to completion: to a finished tool call, to
 * end-of-turn, or to the token budget. */
typedef struct {
    qwasar_engine       *e;
    qwasar_tokenizer    *tok;
    qwasar_session      *s;
    qwasar_chat_options  chat;
    agent_cfg            cfg;
    const char          *guidance;
    bool                 no_cache;

    qw_tui              *tui;
    /* Footer state.  Kept here rather than recomputed at each call site so the
     * status line reads the same whatever produced it. */
    int32_t              ctx_max;
    double               turn_started;
    int32_t              turn_tokens;
    int64_t              spec_rounds, spec_tokens;

    /* An image waiting to be attached to the next turn.  Encoded when it is
     * named rather than when it is used, because the prompt has to say how
     * many <|image_pad|> tokens it needs and that is a property of the patch
     * grid.  Consumed by the first eval of the turn and not by the tool-result
     * evals that follow it. */
    qwasar_image_input   img;
    int32_t              n_img_rows;
    bool                 img_is_video;
    /* Only the prefill start time is kept; the rate is derived from it. */
    double               prefill_started;
    bool                 interrupted;
} agent;

/* Renders token counts the way a person reads them. */
static void fmt_tokens(char *out, size_t cap, int32_t n) {
    if (n >= 10000) snprintf(out, cap, "%.1fk", (double)n / 1000.0);
    else snprintf(out, cap, "%d", n);
}

/* The one place the footer is composed, so every state looks alike:
 *
 *     ctx 1.2k/32k  ·  thinking 84 tokens  ·  5.8 t/s
 */
static void status_set(agent *a, const char *what) {
    if (!a->tui) return;
    char used[24], total[24];
    fmt_tokens(used, sizeof used, a->s ? qwasar_session_n_past(a->s) : 0);
    fmt_tokens(total, sizeof total, a->ctx_max);

    if (a->turn_tokens > 0) {
        double dt = now_sec() - a->turn_started;
        double tps = dt > 0.0 ? (double)a->turn_tokens / dt : 0.0;
        tui_status(a->tui, "ctx %s/%s  ·  %s %d tokens  ·  %.1f t/s",
                   used, total, what, a->turn_tokens, tps);
    } else {
        tui_status(a->tui, "ctx %s/%s  ·  %s", used, total, what);
    }
}

static void progress_cb(void *ud, int32_t done, int32_t total) {
    agent *a = ud;
    if (!a->tui || total < AGENT_BAR_MIN_TOKENS) return;
    if (done == 0) { a->prefill_started = now_sec(); return; }

    const double elapsed = now_sec() - a->prefill_started;
    const double tps = elapsed > 0.0 ? (double)done / elapsed : 0.0;

    /* The rate is written into the unfilled run of the bar rather than after
     * it, so the footer keeps one width whether or not there is a number yet. */
    char rate[24] = "";
    if (tps > 0.0) snprintf(rate, sizeof rate, " %.0f t/s ", tps);
    size_t rate_len = strlen(rate);

    const int filled = (int)(((long long)done * AGENT_BAR_WIDTH) / total);
    /* Near the end the bar overruns the rate; a clipped "17]" is worse than a
     * bar that simply finishes, so drop the number once it no longer fits. */
    if (rate_len + (size_t)filled > AGENT_BAR_WIDTH) rate_len = 0;
    char bar[AGENT_BAR_WIDTH * 4 + 1];
    size_t pos = 0;
    for (int i = 0; i < AGENT_BAR_WIDTH; i++) {
        if (i < filled) { memcpy(bar + pos, "\u25b6", 3); pos += 3; }
        else if (rate_len && (size_t)(i - filled) < rate_len) bar[pos++] = rate[i - filled];
        else { memcpy(bar + pos, "\u00b7", 2); pos += 2; }
    }
    bar[pos] = 0;

    char used[24], total_s[24];
    fmt_tokens(used, sizeof used, a->s ? qwasar_session_n_past(a->s) : 0);
    fmt_tokens(total_s, sizeof total_s, a->ctx_max);
    tui_status(a->tui, "ctx %s/%s  ·  prefill [%s] %d/%d %.0f%%",
               used, total_s, bar, done, total,
               100.0 * (double)done / (double)(total > 0 ? total : 1));
    tui_tick(a->tui);
}


/* Evaluates a prompt, using a disk checkpoint for whatever prefix it already
 * covers, and leaves a checkpoint at the system turn.
 *
 * The system prefix is the one span that is byte-identical on every run, and at
 * ~900 tokens it is most of a cold start.  Whole conversations are not saved
 * automatically: a checkpoint carries the recurrent state, which is ~149 MB
 * regardless of length, so saving every turn would fill the budget with
 * near-duplicates.  /save exists for when a conversation is worth keeping. */
static int32_t agent_prefill(agent *a, const int32_t *tokens, int32_t n,
                             char *err, size_t errcap) {
    /* Where the system turn ends inside this prompt.  Rendered here rather than
     * cached at startup because /effort rewrites the system turn, so a value
     * captured earlier can describe a prompt that no longer exists -- and one
     * that is longer than the current prompt makes the caller's suffix length
     * negative. */
    int32_t sys_n = 0;
    {
        qwasar_message sys = { "system", a->guidance, NULL, NULL, 0, false };
        qwasar_chat_options only = a->chat;
        only.add_generation_prompt = false;
        char serr[256];
        int32_t *p = qwasar_apply_chat_template(a->tok, &sys, 1, &only, &sys_n,
                                                serr, sizeof serr);
        free(p);
        if (!p) sys_n = 0;
    }
    if (sys_n > n) sys_n = n;

    int32_t covered = 0;
    if (!a->no_cache) {
        double t0 = now_sec();
        covered = qwasar_session_restore(a->s, a->e, tokens, n);
        if (covered > 0)
            tui_printf(a->tui, "  \x1b[2m[restored %d tokens from cache in %.2fs]\x1b[0m\n",
                       covered, now_sec() - t0);
    }

    /* Stop at the system boundary so a checkpoint can be left there, then
     * continue.  Same total work either way. */
    if (covered < sys_n) {
        if (!qwasar_session_eval(a->s, tokens + covered, sys_n - covered, err, errcap))
            return -1;
        covered = sys_n;
        if (!a->no_cache) {
            char serr[256];
            if (qwasar_session_save(a->s, a->e, serr, sizeof serr))
                tui_printf(a->tui, "  \x1b[2m[cached %d-token system prefix]\x1b[0m\n", sys_n);
        }
    }
    return covered > n ? n : covered;
}

static bool agent_open_session(agent *a, char *err, size_t errcap) {
    if (a->s) qwasar_session_free(a->s);
    a->s = qwasar_session_new(a->e, err, errcap);
    if (!a->s) return false;
    /* The callback reads the whole agent -- the tui, the session, the context
     * limit -- so the whole agent is what it gets.  Handing it a sub-object
     * compiles either way through the void *, and misreads the struct. */
    qwasar_session_set_progress(a->s, progress_cb, a);
    return true;
}

/* What ended the consumption of one token. */
typedef enum { TAKE_GO, TAKE_EOS, TAKE_CALL, TAKE_OOM } take_result;

/* Accumulates one generated token into the turn and shows it.
 *
 * Factored out because the speculative path settles several tokens in a single
 * pass and every one of them has to be treated exactly as a serially decoded
 * one would be -- the whole guarantee is that speculation changes nothing but
 * the number of forward passes. */
static take_result take_token(agent *a, turn *out, int32_t next,
                              bool *reasoning, bool *in_call,
                              int32_t think_close, int32_t call_open) {
    if (qwasar_is_eos(a->e, next)) { out->hit_eos = true; return TAKE_EOS; }
    if (!turn_push(out, next)) return TAKE_OOM;
    a->turn_tokens++;
    if (*reasoning) out->n_think++;

    size_t len = 0;
    bool special = false;
    const char *bytes = qwasar_token_bytes(a->tok, next, &len, &special);

    if (next == think_close) {
        *reasoning = false;
        if (a->cfg.show_think) tui_puts(a->tui, "\x1b[0m\n");
    } else if (bytes && len) {
        /* The call itself is markup, not prose.  It is still accumulated for
         * the parser, but echoing it would bury any narration the model wrote
         * first -- and it is told it may narrate before a call. */
        if (!*reasoning && next == call_open) *in_call = true;
        str_add(*reasoning ? &out->think : &out->text, bytes, len);
        if (!special && !*in_call && (!*reasoning || a->cfg.show_think)) {
            if (*reasoning && tui_is_tty(a->tui) && out->think.len == len)
                tui_puts(a->tui, "\x1b[2m");
            tui_out(a->tui, bytes, len);
        }
    }

    status_set(a, *reasoning ? "thinking" : (*in_call ? "calling" : "writing"));
    tui_tick(a->tui);

    if (!*reasoning && qw_tool_call_complete(out->text.p ? out->text.p : "",
                                             out->text.len)) {
        out->has_call = true;
        return TAKE_CALL;
    }
    return TAKE_GO;
}

static bool generate(agent *a, const int32_t *prompt, int32_t n_prompt,
                     turn *out, char *err, size_t errcap) {
    memset(out, 0, sizeof *out);

    status_set(a, "prefill");
    const float *logits;
    if (a->n_img_rows > 0) {
        logits = qwasar_session_eval_images(a->s, prompt, n_prompt, &a->img, 1,
                                            err, errcap);
        /* One turn's worth: the tool results that follow are text. */
        qwasar_image_release(&a->img);
        a->n_img_rows = 0;
    } else {
        logits = qwasar_session_eval(a->s, prompt, n_prompt, err, errcap);
    }
    if (!logits) return false;

    const int32_t vocab = qwasar_vocab_size(a->e);
    const int32_t think_close = qwasar_token_id(a->tok, "</think>");
    const int32_t call_open   = qwasar_token_id(a->tok, "<tool_call>");
    bool reasoning = true;
    bool in_call = false;

    a->turn_started = now_sec();
    a->turn_tokens = 0;
    a->interrupted = false;

    const bool spec = a->cfg.mtp_depth != 0 && qwasar_session_has_mtp(a->s);
    int32_t next = argmax(logits, vocab);
    int i = 0;

    while (i < a->cfg.max_tokens) {
        if (tui_interrupted(a->tui)) {
            a->interrupted = true;
            tui_newline(a->tui);
            tui_puts(a->tui, "  [interrupted]\n");
            break;
        }

        take_result r = take_token(a, out, next, &reasoning, &in_call,
                                   think_close, call_open);
        i++;
        if (r == TAKE_OOM) { snprintf(err, errcap, "out of memory"); return false; }
        if (r != TAKE_GO) break;

        if (spec) {
            /* One pass settles the whole drafted block.  The last committed
             * token takes the place `next` had: it is the one this round leaves
             * undecided, exactly as an ordinary step would. */
            int32_t blk[1 + QWASAR_MAX_DRAFT], got[1 + QWASAR_MAX_DRAFT];
            blk[0] = next;
            const int32_t want = a->cfg.mtp_depth > 0
                               ? a->cfg.mtp_depth
                               : qwasar_session_draft_depth(a->s);
            if (want == 0) {
                /* Not worth a round here; take the cheaper plain step. */
                logits = qwasar_session_eval(a->s, &next, 1, err, errcap);
                if (!logits) return false;
                next = argmax(logits, vocab);
                continue;
            }
            int32_t nd = qwasar_session_draft(a->s, next, blk + 1, want, err, errcap);
            if (nd < 0) return false;
            int32_t nc = qwasar_session_verify(a->s, blk, nd + 1, got, err, errcap);
            if (nc < 0) return false;
            a->spec_rounds++;
            a->spec_tokens += nc;

            bool stop = false;
            for (int32_t t = 0; t + 1 < nc && i < a->cfg.max_tokens; t++) {
                r = take_token(a, out, got[t], &reasoning, &in_call,
                               think_close, call_open);
                i++;
                if (r == TAKE_OOM) { snprintf(err, errcap, "out of memory"); return false; }
                if (r != TAKE_GO) { stop = true; break; }
            }
            if (stop) break;
            next = got[nc - 1];
            continue;
        }

        logits = qwasar_session_eval(a->s, &next, 1, err, errcap);
        if (!logits) return false;
        next = argmax(logits, vocab);
    }

    /* Reaching the bound is the one way this loop ends with nothing to show for
     * it, so it is recorded rather than inferred from an empty answer. */
    out->hit_budget  = (i >= a->cfg.max_tokens);
    out->in_reasoning = reasoning;
    if (tui_is_tty(a->tui)) tui_puts(a->tui, "\x1b[0m");
    return true;
}

/* ---- session ---------------------------------------------------------------- */

/* Runs one task to completion: generate, and while the model asks for a tool,
 * run it and hand the result back.  `prompt` is consumed. */
/* Tool calls are the part of a transcript a person scans for, so they get a
 * marker column and colour rather than being another paragraph of prose.  Long
 * values are elided on one line; the tool output that follows is what matters. */
static void show_tool_call(agent *a, const qw_tool_call *c) {
    const bool tty = tui_is_tty(a->tui);
    tui_printf(a->tui, "%s  %s%s", tty ? "\x1b[36m" : "", c->name, tty ? "\x1b[0m" : "");
    for (int j = 0; j < c->n_params; j++) {
        const char *v = c->params[j].value;
        size_t vl = strlen(v);
        /* A file body would swamp the line; its first line is enough to
         * recognise, and the edit either lands or reports why. */
        size_t show = vl;
        const char *nl = memchr(v, '\n', vl);
        if (nl) show = (size_t)(nl - v);
        if (show > 52) show = 52;
        tui_printf(a->tui, " %s%s=%s%.*s%s", tty ? "\x1b[2m" : "",
                   c->params[j].key, tty ? "\x1b[0m" : "", (int)show, v,
                   show < vl ? (tty ? "\x1b[2m...\x1b[0m" : "...") : "");
    }
    tui_puts(a->tui, "\n");
}

/* A few dimmed lines, not the whole payload: a 200-line file read would
 * otherwise push the conversation off the screen, and the model has the full
 * text regardless. */
#define TOOL_RESULT_LINES 3
#define TOOL_RESULT_COLS  72

static void show_tool_result(agent *a, const str *result) {
    const char *p = result->p ? result->p : "";
    size_t n = result->len;
    while (n && (p[n-1] == '\n' || p[n-1] == ' ')) n--;
    if (!n) { tui_puts(a->tui, "  \x1b[2m    (no output)\x1b[0m\n"); return; }

    size_t shown = 0, off = 0;
    while (off < n && shown < TOOL_RESULT_LINES) {
        const char *nl = memchr(p + off, '\n', n - off);
        size_t len = nl ? (size_t)(nl - (p + off)) : n - off;
        size_t cut = len > TOOL_RESULT_COLS ? TOOL_RESULT_COLS : len;
        tui_printf(a->tui, "  \x1b[2m    %.*s%s\x1b[0m\n",
                   (int)cut, p + off, cut < len ? "…" : "");
        off += len + (nl ? 1 : 0);
        shown++;
    }
    if (off < n) {
        size_t rest = 0;
        for (size_t i = off; i < n; i++) if (p[i] == '\n') rest++;
        tui_printf(a->tui, "  \x1b[2m    … %zu more line%s, %zu bytes\x1b[0m\n",
                   rest + 1, rest ? "s" : "", n);
    }
}

static bool agent_run(agent *a, int32_t *prompt, int32_t n_prompt,
                      char *err, size_t errcap) {
    int step = 0;
    for (;;) {
        turn t;
        if (!generate(a, prompt, n_prompt, &t, err, errcap)) {
            free(prompt);
            return false;
        }
        free(prompt);
        prompt = NULL;

        if (!t.has_call || a->interrupted) {
            tui_newline(a->tui);
            /* A turn cut off by the token budget used to end exactly like a
             * finished one: no message, and -- when the budget ran out inside
             * the reasoning block -- no output at all, because reasoning is not
             * printed.  From the outside that is generation stopping for no
             * reason, which is precisely what it looked like. */
            if (t.hit_budget) {
                if (t.in_reasoning)
                    tui_printf(a->tui, "  [stopped at the %d-token budget while still "
                               "reasoning, so there is no answer to show; raise it with "
                               "-n, or use /effort low]\n", a->cfg.max_tokens);
                else
                    tui_printf(a->tui, "  [stopped at the %d-token budget, %d of them "
                               "reasoning; raise it with -n]\n",
                               a->cfg.max_tokens, t.n_think);
            }
            turn_free(&t);
            status_set(a, a->interrupted ? "interrupted"
                                         : (t.hit_budget ? "truncated" : "done"));
            return true;
        }

        qw_tool_calls calls;
        char perr[256] = "";
        int n = qw_tool_parse(t.text.p ? t.text.p : "", &calls, perr, sizeof perr);

        str result = { 0 };
        if (n < 0) {
            str_printf(&result, "error: %s. Re-issue the call in the required format.", perr);
        } else if (n == 0) {
            str_puts(&result, "error: no tool call was found.");
        } else {
            for (int i = 0; i < n; i++) {
                tui_newline(a->tui);
                show_tool_call(a, &calls.calls[i]);
                status_set(a, calls.calls[i].name);
                tui_tick(a->tui);
                if (i) str_puts(&result, "\n");
                dispatch(&calls.calls[i], &a->cfg, &result);
                show_tool_result(a, &result);
            }
        }
        qw_tool_calls_free(&calls);
        turn_free(&t);

        if (++step >= a->cfg.max_steps) {
            tui_printf(a->tui, "  [stopped after %d tool calls]\n", step);
            str_free(&result);
            return true;
        }

        prompt = qwasar_render_tool_result(a->tok, result.p ? result.p : "",
                                           &a->chat, &n_prompt);
        str_free(&result);
        if (!prompt) {
            snprintf(err, errcap, "cannot render the tool result");
            return false;
        }
    }
}

/* ---- repl ------------------------------------------------------------------- */

static void repl_help(agent *a) {
    tui_puts(a->tui,"  /help            this message\n"
           "  /new             start a fresh conversation\n"
           "  /image <path>    attach an image to the next message\n"
           "  /video <path>    attach a video, sampled at 2 fps\n"
           "  /effort <level>  xhigh, medium or low\n"
           "  /think           show or hide the reasoning block\n"
           "  /yes             toggle asking before writes and commands\n"
           "  /ctx             context and disk cache usage\n"
           "  /save            checkpoint this conversation to disk\n"
           "  /quit            leave\n");
}

/* Returns false when the command asked to quit. */
static bool repl_command(agent *a, const char *line, bool *handled) {
    *handled = true;
    if (!strcmp(line, "/quit") || !strcmp(line, "/exit")) return false;
    if (!strcmp(line, "/help")) { repl_help(a); return true; }
    if (!strcmp(line, "/ctx")) {
        uint64_t bytes = 0;
        int entries = 0;
        qwasar_kv_cache_stats(&bytes, &entries);
        tui_printf(a->tui, "  %d tokens used  ·  disk cache %d entries, %.1f GB\n",
                   qwasar_session_n_past(a->s), entries, (double)bytes / 1e9);
        return true;
    }
    if (!strcmp(line, "/save")) {
        char serr[256];
        if (qwasar_session_save(a->s, a->e, serr, sizeof serr))
            tui_printf(a->tui, "  saved %d tokens\n", qwasar_session_n_past(a->s));
        else
            tui_printf(a->tui, "  not saved: %s\n", serr);
        return true;
    }
    if (!strncmp(line, "/image", 6) || !strncmp(line, "/video", 6)) {
        const bool as_video = line[1] == 'v';
        const char *path = line + 6;
        while (*path == ' ') path++;
        if (!*path) {
            tui_printf(a->tui, "  usage: %s <path>\n", as_video ? "/video" : "/image");
            return true;
        }
        /* Attaching costs a tower pass now rather than at send time, because
         * the message cannot be rendered until the token count is known. */
        qwasar_image_release(&a->img);
        a->n_img_rows = 0;
        a->img_is_video = false;
        status_set(a, "looking");
        tui_tick(a->tui);
        char ierr[256];
        const bool ok = as_video
            ? qwasar_video_encode(a->e, path, &a->img, ierr, sizeof ierr)
            : qwasar_image_encode(a->e, path, &a->img, ierr, sizeof ierr);
        if (!ok) {
            tui_printf(a->tui, "  %s\n", ierr);
        } else {
            a->n_img_rows = a->img.n_rows;
            a->img_is_video = as_video;
            tui_printf(a->tui, "  %dx%d%s -> %d tokens; it goes with your next message\n",
                       a->img.src_w, a->img.src_h,
                       a->img.grid_t > 1 ? ", multiple frames" : "", a->img.n_rows);
        }
        status_set(a, "ready");
        return true;
    }
    if (!strcmp(line, "/think")) {
        a->cfg.show_think = !a->cfg.show_think;
        tui_printf(a->tui, "  reasoning block %s\n", a->cfg.show_think ? "shown" : "hidden");
        return true;
    }
    if (!strcmp(line, "/yes")) {
        a->cfg.yes = !a->cfg.yes;
        tui_printf(a->tui, "  %s before writes and commands\n",
                   a->cfg.yes ? "no longer asking" : "asking");
        return true;
    }
    if (!strncmp(line, "/effort", 7)) {
        const char *lvl = line + 7;
        while (*lvl == ' ') lvl++;
        if (!strcmp(lvl, "xhigh") || !strcmp(lvl, "medium") || !strcmp(lvl, "low")) {
            a->chat.reasoning_effort = !strcmp(lvl, "low")    ? "low"
                                     : !strcmp(lvl, "medium") ? "medium" : "xhigh";
            /* The effort instruction lives in the system turn, which is already
             * in the cache; it only takes effect on a fresh conversation. */
            tui_printf(a->tui, "  effort will be %s from the next /new\n", a->chat.reasoning_effort);
        } else {
            tui_printf(a->tui, "  effort must be xhigh, medium or low\n");
        }
        return true;
    }
    if (!strcmp(line, "/new")) { *handled = false; return true; }   /* caller resets */
    tui_printf(a->tui, "  unknown command; try /help\n");
    return true;
}

static void usage(FILE *out) {
    fprintf(out,
        "qwasar-agent -- an agentic loop on Qwen3.8\n"
        "\n"
        "usage: qwasar-agent [-m <model-dir>] [options] [task...]\n"
        "\n"
        "With a task it runs once and exits.  With no task it opens a REPL.\n"
        "\n"
        "  -m, --model <dir>    model directory; default ./qwasar-model\n"
        "  -C, --chdir <dir>    work in this directory\n"
        "  -c, --context <n>    context size in tokens (default 32768)\n"
        "  -y, --yes            do not ask before writing files or running commands\n"
        "  -i, --interactive    open the REPL after running the task\n"
        "      --steps <n>      maximum tool calls per task (default 24)\n"
        "  -n, --predict <n>    maximum tokens per turn (default 8192)\n"
        "      --image <path>   an image for the first turn (jpeg, png, bmp, gif)\n"
        "      --video <path>   a video for the first turn, sampled at 2 fps\n"
        "      --mtp <dir>      draft head; speculative decoding, about 1.4x\n"
        "      --mtp-depth <n>  drafts per round; default adapts, 0 disables\n"
        "      --effort <lvl>   reasoning effort: xhigh (default), medium, low\n"
        "      --show-think     print the reasoning block\n"
        "      --no-cache       do not use or write disk checkpoints\n"
        "  -h, --help           this message\n"
        "\n"
        "Tools: read, write, edit, list, grep, bash.  Reading runs unattended;\n"
        "writing and running commands ask first unless --yes.\n"
        "If AGENT.md exists in the working directory it is added to the system\n"
        "prompt as project guidance.\n");
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
    const char *image_path = NULL;
    agent a = { 0 };
    /* The turn budget bounds a runaway generation; it is not meant to bound
     * ordinary work, and the two failure modes are not symmetric.  Too high
     * wastes time on a turn ctrl-C can stop; too low silently truncates the
     * answer, and at the default xhigh effort most of the budget goes to a
     * reasoning block the user never sees, so the truncation arrives with only
     * a few dozen visible tokens on screen.  8192 leaves several turns inside
     * the 32K context. */
    a.cfg = (agent_cfg){ .yes = false, .show_think = false, .max_steps = 24, .max_tokens = 8192, .mtp_depth = -1 };
    const char *effort = "xhigh";
    const char *workdir = NULL;
    bool interactive = false;
    str task = { 0 };

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if ((!strcmp(arg, "-m") || !strcmp(arg, "--model")) && i + 1 < argc) opts.model_path = argv[++i];
        else if ((!strcmp(arg, "-C") || !strcmp(arg, "--chdir")) && i + 1 < argc) workdir = argv[++i];
        else if ((!strcmp(arg, "-c") || !strcmp(arg, "--context")) && i + 1 < argc) opts.context_size = atoi(argv[++i]);
        else if (!strcmp(arg, "-y") || !strcmp(arg, "--yes")) a.cfg.yes = true;
        else if (!strcmp(arg, "-i") || !strcmp(arg, "--interactive")) interactive = true;
        else if (!strcmp(arg, "--image") && i + 1 < argc) image_path = argv[++i];
        else if (!strcmp(arg, "--video") && i + 1 < argc) { image_path = argv[++i]; a.img_is_video = true; }
        else if (!strcmp(arg, "--mtp") && i + 1 < argc) opts.mtp_path = argv[++i];
        else if (!strcmp(arg, "--mtp-depth") && i + 1 < argc) a.cfg.mtp_depth = atoi(argv[++i]);
        else if (!strcmp(arg, "--steps") && i + 1 < argc) a.cfg.max_steps = atoi(argv[++i]);
        else if ((!strcmp(arg, "-n") || !strcmp(arg, "--predict")) && i + 1 < argc) a.cfg.max_tokens = atoi(argv[++i]);
        else if (!strcmp(arg, "--effort") && i + 1 < argc) effort = argv[++i];
        else if (!strcmp(arg, "--show-think")) a.cfg.show_think = true;
        else if (!strcmp(arg, "--no-cache")) a.no_cache = true;
        else if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) { usage(stdout); return 0; }
        else if (arg[0] == '-') { fprintf(stderr, "qwasar-agent: unknown argument '%s'\n\n", arg); usage(stderr); return 2; }
        else { if (task.len) str_puts(&task, " "); str_puts(&task, arg); }
    }

    if (!resolve_model(&opts, "qwasar-agent")) return 2;
    if (workdir && chdir(workdir) != 0) {
        fprintf(stderr, "qwasar-agent: cannot enter %s: %s\n", workdir, strerror(errno));
        return 1;
    }
    if (!task.len) interactive = true;

    a.tui = tui_new();
    g_tui = a.tui;

    char err[512] = "";
    double t0 = now_sec();
    a.e = qwasar_engine_load(&opts, err, sizeof err);
    if (!a.e) { fprintf(stderr, "qwasar-agent: %s\n", err); return 1; }
    a.tok = qwasar_tokenizer_load(opts.model_path, err, sizeof err);
    if (!a.tok) { fprintf(stderr, "qwasar-agent: %s\n", err); return 1; }

    str guidance = { 0 };
    str_puts(&guidance,
             "You are qwasar-agent, working in the user's current directory. "
             "Use the tools to inspect and change real files. Prefer edit over "
             "write for existing files. Check your work when you are done.");
    str agentmd = { 0 };
    if (read_file("AGENT.md", &agentmd) && agentmd.len) {
        str_puts(&guidance, "\n\nProject guidance from AGENT.md:\n\n");
        str_add(&guidance, agentmd.p, agentmd.len);
    }
    str_free(&agentmd);

    a.ctx_max = opts.context_size > 0 ? opts.context_size : 32768;

    a.chat = (qwasar_chat_options){
        .enable_thinking = true,
        .reasoning_effort = effort,
        .add_generation_prompt = true,
        .tools = AGENT_TOOLS,
        .n_tools = AGENT_N_TOOLS,
    };
    a.guidance = guidance.p;

    if (!agent_open_session(&a, err, sizeof err)) {
        fprintf(stderr, "qwasar-agent: %s\n", err);
        return 1;
    }
    if (image_path) {
        const bool ok = a.img_is_video
            ? qwasar_video_encode(a.e, image_path, &a.img, err, sizeof err)
            : qwasar_image_encode(a.e, image_path, &a.img, err, sizeof err);
        if (!ok) { fprintf(stderr, "qwasar-agent: %s\n", err); return 1; }
        a.n_img_rows = a.img.n_rows;
        tui_printf(a.tui, "\x1b[2m%s %dx%d%s -> %d patches -> %d tokens\x1b[0m\n",
                   a.img_is_video ? "video" : "image", a.img.src_w, a.img.src_h,
                   a.img.grid_t > 1 ? " (multiple frames)" : "",
                   a.img.n_patches, a.img.n_rows);
    }

    /* Speculation is silent about what it does to the output -- nothing -- but
     * it is not silent about being on, because it changes the memory footprint
     * and the shape of a stall. */
    const bool spec_on = a.cfg.mtp_depth > 0 && qwasar_session_has_mtp(a.s);
    if (opts.mtp_path && !spec_on)
        tui_printf(a.tui, "\x1b[2mmtp head loaded but drafting is off\x1b[0m\n");
    tui_printf(a.tui, "\x1b[2mloaded in %.1fs  ·  %d tools  ·  %s%s\x1b[0m\n",
               now_sec() - t0, AGENT_N_TOOLS,
               a.cfg.yes ? "not asking before writes" : "asking before writes",
               spec_on ? "  ·  drafting ahead" : "");

    bool fresh = true;   /* the next turn must render the system prompt */
    int rc = 0;


    /* One-shot task, if given. */
    if (task.len) {
        qwasar_message msgs[2] = {
            { "system", a.guidance, NULL, NULL, 0, false },
            { "user",   task.p,     NULL, NULL, a.n_img_rows, a.img_is_video },
        };
        int32_t n = 0;
        int32_t *p = qwasar_apply_chat_template(a.tok, msgs, 2, &a.chat, &n, err, sizeof err);
        if (!p) { fprintf(stderr, "qwasar-agent: %s\n", err); return 1; }
        int32_t covered = agent_prefill(&a, p, n, err, sizeof err);
        if (covered < 0) { fprintf(stderr, "qwasar-agent: %s\n", err); free(p); return 1; }
        memmove(p, p + covered, (size_t)(n - covered) * sizeof *p);
        n -= covered;
        fresh = false;
        if (!agent_run(&a, p, n, err, sizeof err)) {
            fprintf(stderr, "qwasar-agent: %s\n", err);
            rc = 1;
        }
    }

    if (interactive && rc == 0) {
        static const char *const COMMANDS[] = {
            "/help", "/new", "/effort ", "/think", "/yes", "/ctx", "/save", "/quit", NULL
        };
        tui_set_commands(COMMANDS);

        char hist[1024] = "";
        const char *home = getenv("HOME");
        if (home) {
            snprintf(hist, sizeof hist, "%s/.cache/qwasar/history", home);
            tui_history_load(a.tui, hist);
        }

        if (tui_is_tty(a.tui)) {
            tui_printf(a.tui, "\n\x1b[1mqwasar-agent\x1b[0m  \x1b[2m%s  ·  "
                              "/help for commands  ·  ctrl-C interrupts\x1b[0m\n\n",
                       AGENT_N_TOOLS == 6 ? "6 tools" : "tools");
        } else {
            tui_puts(a.tui, "\nqwasar-agent. /help for commands, /quit to leave.\n\n");
        }

        for (;;) {
            a.turn_tokens = 0;
            status_set(&a, "ready");
            char *line = tui_readline(a.tui, "\x1b[1;32m>\x1b[0m ");
            if (!line) break;                        /* ctrl-D */
            if (!*line) { free(line); continue; }
            tui_history_add(a.tui, line);
            if (hist[0]) tui_history_save(a.tui, hist);

            if (line[0] == '/') {
                bool handled = true;
                bool keep = repl_command(&a, line, &handled);
                if (!keep) { free(line); break; }
                if (handled) { free(line); continue; }
                if (!agent_open_session(&a, err, sizeof err)) {
                    tui_printf(a.tui, "  %s\n", err);
                    free(line);
                    rc = 1;
                    break;
                }
                fresh = true;
                tui_puts(a.tui, "  new conversation\n");
                free(line);
                continue;
            }

            int32_t n = 0;
            int32_t *p;
            if (fresh) {
                qwasar_message msgs2[2] = {
                    { "system", a.guidance, NULL, NULL, 0, false },
                    { "user",   line,       NULL, NULL, a.n_img_rows, a.img_is_video },
                };
                p = qwasar_apply_chat_template(a.tok, msgs2, 2, &a.chat, &n, err, sizeof err);
                if (p) {
                    int32_t covered = agent_prefill(&a, p, n, err, sizeof err);
                    if (covered < 0) { free(p); p = NULL; }
                    else { memmove(p, p + covered, (size_t)(n - covered) * sizeof *p);
                           n -= covered; }
                }
                fresh = false;
            } else {
                p = qwasar_render_user_turn(a.tok, line, a.n_img_rows, a.img_is_video,
                                            &a.chat, &n);
                if (!p) snprintf(err, sizeof err, "cannot render the turn");
            }
            free(line);
            if (!p) { tui_printf(a.tui, "  %s\n", err); rc = 1; break; }

            if (!agent_run(&a, p, n, err, sizeof err)) {
                /* Running out of context ends a long conversation; it is not a
                 * crash, and /new carries on from there. */
                tui_printf(a.tui, "  %s\n", err);
                if (strstr(err, "context exhausted"))
                    tui_puts(a.tui, "  use /new to start over\n");
                else { rc = 1; break; }
            }
        }
    }

    tui_free(a.tui);
    str_free(&task);
    str_free(&guidance);
    qwasar_session_free(a.s);
    qwasar_tokenizer_free(a.tok);
    qwasar_engine_free(a.e);
    return rc;
}
