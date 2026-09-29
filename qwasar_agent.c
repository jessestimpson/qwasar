/* qwasar-agent -- an agentic loop on qwasar-server's Session API.
 *
 * The tools, the confirmations and the terminal are here; the model is not.
 * The agent opens a session on the server (API.md) -- its system prompt and
 * tools become the session's prefix, fixed for its life -- and from then on
 * sends only what is new: a user turn, or the results of the calls the model
 * asked for.  The server owns the conversation's state and says how warm it
 * is; the agent shows what the server streams: prefill progress, reasoning,
 * the answer, the calls.
 *
 * No server, no problem: when nothing answers on the port and a model can be
 * found, the agent starts qwasar-server itself, holding a pipe on its stdin
 * so the server goes when the agent does (the menu bar's lifeline, reused).
 *
 * Tools that only read run unattended.  Tools that write to the filesystem or
 * run commands ask first, unless --yes. */

#include "qwasar_http.h"
#include "qwasar_json.h"
#include "qwasar_toolcall.h"
#include "qwasar_tui.h"

#include <errno.h>
#include <fcntl.h>
#include <mach-o/dyld.h>
#include <poll.h>
#include <signal.h>
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
 * One JSON object per tool, sent to the server at open and rendered by it
 * into the system turn.  The edit description spells out the match rule
 * because that rule is the tool's whole contract: the model has to know that
 * quoting too little will be rejected as ambiguous rather than applied
 * somewhere arbitrary. */
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
"\"List a directory (ls -lA).\", \"parameters\": {\"type\": \"object\", "
"\"properties\": {\"path\": {\"type\": \"string\", \"description\": \"Directory; "
"defaults to the current one.\"}}, \"required\": []}}}",

"{\"type\": \"function\", \"function\": {\"name\": \"grep\", \"description\": "
"\"Search files recursively for an extended regular expression (grep -rnE). "
"Returns matching lines with file names and line numbers.\", \"parameters\": "
"{\"type\": \"object\", \"properties\": {\"pattern\": {\"type\": \"string\", "
"\"description\": \"The regular expression.\"}, \"path\": {\"type\": \"string\", "
"\"description\": \"File or directory to search; defaults to the current "
"directory.\"}}, \"required\": [\"pattern\"]}}}",

"{\"type\": \"function\", \"function\": {\"name\": \"bash\", \"description\": "
"\"Run a shell command and return its combined output and exit status. The user "
"is asked to approve each command first.\", \"parameters\": {\"type\": \"object\", "
"\"properties\": {\"command\": {\"type\": \"string\", \"description\": \"The "
"command line.\"}}, \"required\": [\"command\"]}}}",
};
#define AGENT_N_TOOLS ((int)(sizeof AGENT_TOOLS / sizeof *AGENT_TOOLS))

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
    bool    yes;          /* skip confirmations */
    bool    show_think;
    int     max_steps;
    int32_t max_tokens;   /* per step; the server caps it to the window's room */
    float   temperature;  /* < 0: the model's own defaults */
} agent_cfg;

static qw_tui *g_tui;

/* Mutating tools ask before acting.  A refusal is reported back to the model as
 * a tool result rather than aborting, so it can choose something else.  The
 * question goes through the TUI's own ask, so it lands above the footer and
 * a message typed ahead is not mistaken for the answer. */
static bool confirm(const agent_cfg *cfg, const char *what, const char *detail) {
    if (cfg->yes) return true;
    const bool tty = tui_is_tty(g_tui);
    tui_printf(g_tui, "\n  %s%s%s %s\n", tty ? "\x1b[1;33m" : "", what,
               tty ? "\x1b[0m" : "", detail);
    char *ans = tui_ask(g_tui, "  proceed? [y/N] ");
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

/* ---- the agent ---------------------------------------------------------------- */

/* A call the server reported, with what running it produced. */
typedef struct {
    char        id[24];
    qw_tool_call call;      /* strings owned here */
    str         result;
} pending_call;

typedef struct {
    char        host[256];
    int         port;
    const char *token;
    agent_cfg   cfg;
    const char *effort;
    char       *guidance;
    qw_tui     *tui;

    char        session[32];      /* the open session's id, or "" */
    int32_t     prefix_tokens;
    char        model_name[64];

    /* Footer state, from the server's events. */
    int32_t     ctx_used, ctx_max;
    double      turn_started;
    int32_t     turn_tokens;
    double      tps;
    double      prefill_started;
    int32_t     think_tokens;

    /* An attachment waiting for the next turn: base64, with its type. */
    char       *att_b64;
    char        att_type[32];
    bool        att_video;

    /* The step in flight. */
    pending_call calls[QW_MAX_CALLS];
    int          n_calls;
    char         stop[24];
    int32_t      generated;
    bool         interrupted;
    bool         think_open;      /* a dim reasoning run is on screen */

    /* A server this agent started, and the pipe that ends it. */
    pid_t       server_pid;
    int         lifeline;
    char        server_log[512];
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
    fmt_tokens(used, sizeof used, a->ctx_used);
    fmt_tokens(total, sizeof total, a->ctx_max);
    if (a->turn_tokens > 0)
        tui_status(a->tui, "ctx %s/%s  ·  %s %d tokens  ·  %.1f t/s",
                   used, total, what, a->turn_tokens, a->tps);
    else
        tui_status(a->tui, "ctx %s/%s  ·  %s", used, total, what);
    tui_tick(a->tui);
}

/* ---- prefill progress ------------------------------------------------------
 *
 * Prompt processing is the one stretch of a turn with nothing to look at, and
 * on a long conversation it is the longest.  The server reports it over the
 * whole of what it is evaluating; the rate is written into the unfilled part
 * of the bar, so the line stays one width whether or not there is a number
 * to show yet -- a trick worth stealing from ds4-agent. */

#define AGENT_BAR_WIDTH 28
/* Below this the prompt is processed faster than the eye can follow, and a
 * bar that appears and vanishes is worse than no bar. */
#define AGENT_BAR_MIN_TOKENS 128

static void show_prefill(agent *a, int32_t done, int32_t total) {
    if (!a->tui || total < AGENT_BAR_MIN_TOKENS) return;
    if (a->prefill_started == 0) a->prefill_started = now_sec();
    const double elapsed = now_sec() - a->prefill_started;
    const double tps = elapsed > 0.0 && done > 0 ? (double)done / elapsed : 0.0;

    char rate[24] = "";
    if (tps > 0.0) snprintf(rate, sizeof rate, " %.0f t/s ", tps);
    size_t rate_len = strlen(rate);
    const int filled = (int)(((long long)done * AGENT_BAR_WIDTH) / (total > 0 ? total : 1));
    if (rate_len + (size_t)filled > AGENT_BAR_WIDTH) rate_len = 0;
    char bar[AGENT_BAR_WIDTH * 4 + 1];
    size_t pos = 0;
    for (int i = 0; i < AGENT_BAR_WIDTH; i++) {
        if (i < filled) { memcpy(bar + pos, "▶", 3); pos += 3; }
        else if (rate_len && (size_t)(i - filled) < rate_len) bar[pos++] = rate[i - filled];
        else { memcpy(bar + pos, "·", 2); pos += 2; }
    }
    bar[pos] = 0;

    char used[24], total_s[24];
    fmt_tokens(used, sizeof used, a->ctx_used);
    fmt_tokens(total_s, sizeof total_s, a->ctx_max);
    tui_status(a->tui, "ctx %s/%s  ·  prefill [%s] %d/%d %.0f%%",
               used, total_s, bar, done, total,
               100.0 * (double)done / (double)(total > 0 ? total : 1));
    tui_tick(a->tui);
}

/* ---- talking to the server ------------------------------------------------------- */

/* One request with a JSON answer.  False when the server could not be
 * reached; otherwise `status` is the server's and `doc` holds the body when
 * it parsed (the caller frees it either way). */
static bool api(agent *a, const char *method, const char *path, const char *body,
                int *status, qj_doc *doc) {
    http_resp r;
    str out = { 0 };
    memset(doc, 0, sizeof *doc);
    if (!http_call(a->host, a->port, a->token, method, path, body, &r, &out)) {
        str_free(&out);
        return false;
    }
    *status = r.status;
    if (out.len) qj_parse(doc, out.p, out.len);
    str_free(&out);
    return true;
}

static void api_error(agent *a, const char *what, int status, const qj_doc *doc) {
    char *msg = qj_strdup(doc, qj_get(doc, qj_get(doc, qj_root(doc), "error"), "message"));
    tui_printf(a->tui, "  %s: %s (HTTP %d)\n", what, msg ? msg : "no reason given", status);
    free(msg);
}

static bool server_up(agent *a) {
    int status = 0;
    qj_doc d;
    const bool ok = api(a, "GET", "/health", NULL, &status, &d) && status == 200;
    qj_free(&d);
    return ok;
}

/* Where this binary is, for the server beside it. */
static void self_dir(char *out, size_t cap) {
    char path[1024];
    uint32_t n = sizeof path;
    if (_NSGetExecutablePath(path, &n) != 0) { snprintf(out, cap, "."); return; }
    char *real = realpath(path, NULL);
    snprintf(out, cap, "%s", real ? real : path);
    free(real);
    char *slash = strrchr(out, '/');
    if (slash) *slash = 0; else snprintf(out, cap, ".");
}

static bool is_model_dir(const char *p) {
    char cfg[1200];
    snprintf(cfg, sizeof cfg, "%s/config.json", p);
    return access(cfg, R_OK) == 0;
}

/* The model, when the agent has to start a server: -m, $QWASAR_MODEL, a
 * qwasar-model link in the working directory or beside the binary. */
static const char *resolve_model(const char *given, char *buf, size_t cap) {
    if (given && is_model_dir(given)) return given;
    const char *env = getenv("QWASAR_MODEL");
    if (env && *env && is_model_dir(env)) return env;
    if (is_model_dir("./qwasar-model")) return "./qwasar-model";
    char dir[1024];
    self_dir(dir, sizeof dir);
    snprintf(buf, cap, "%s/qwasar-model", dir);
    return is_model_dir(buf) ? buf : NULL;
}

/* Starts qwasar-server on our port with a lifeline: a pipe on its stdin
 * that closes when this process ends, however it ends, so the server does
 * not outlive the agent that started it.  Waits for the model to load. */
static bool server_start(agent *a, const char *model) {
    char dir[1024], bin[1100];
    self_dir(dir, sizeof dir);
    snprintf(bin, sizeof bin, "%s/qwasar-server", dir);
    if (access(bin, X_OK) != 0) snprintf(bin, sizeof bin, "qwasar-server");   /* PATH */

    const char *tmp = getenv("TMPDIR");
    snprintf(a->server_log, sizeof a->server_log, "%s/qwasar-server-%d.log",
             tmp && *tmp ? tmp : "/tmp", a->port);
    int fds[2];
    if (pipe(fds) != 0) return false;
    int log = open(a->server_log, O_WRONLY | O_CREAT | O_TRUNC, 0644);

    char port[16];
    snprintf(port, sizeof port, "%d", a->port);
    pid_t pid = fork();
    if (pid < 0) { close(fds[0]); close(fds[1]); if (log >= 0) close(log); return false; }
    if (pid == 0) {
        close(fds[1]);
        dup2(fds[0], STDIN_FILENO);
        close(fds[0]);
        if (log >= 0) { dup2(log, STDOUT_FILENO); dup2(log, STDERR_FILENO); close(log); }
        char *argv[] = { bin, "-m", (char *)model, "--port", port, "--exit-on-eof", "-v", NULL };
        execvp(bin, argv);
        _exit(127);
    }
    close(fds[0]);
    if (log >= 0) close(log);
    a->server_pid = pid;
    a->lifeline = fds[1];

    tui_printf(a->tui, "\x1b[2mstarting qwasar-server on port %d with %s  ·  log %s\x1b[0m\n",
               a->port, model, a->server_log);
    const double t0 = now_sec();
    for (;;) {
        int st = 0;
        if (waitpid(pid, &st, WNOHANG) == pid) {
            tui_printf(a->tui, "  qwasar-server exited (status %d); see %s\n",
                       WIFEXITED(st) ? WEXITSTATUS(st) : -1, a->server_log);
            a->server_pid = 0;
            return false;
        }
        if (server_up(a)) {
            tui_printf(a->tui, "\x1b[2mserver ready in %.1fs; it stops when this agent does\x1b[0m\n",
                       now_sec() - t0);
            return true;
        }
        if (now_sec() - t0 > 600) { tui_puts(a->tui, "  the server did not come up in ten minutes\n"); return false; }
        tui_status(a->tui, "loading the model  ·  %.0fs", now_sec() - t0);
        tui_tick(a->tui);
        usleep(250 * 1000);
    }
}

/* Reads what the server is, for the footer and the banner. */
static bool server_info(agent *a) {
    int status = 0;
    qj_doc d;
    if (!api(a, "GET", "/v1/server", NULL, &status, &d) || status != 200) { qj_free(&d); return false; }
    const qj_node *root = qj_root(&d);
    a->ctx_max = (int32_t)qj_int_or(&d, root, "context", 0);
    qj_str_copy(&d, root, "model.name", a->model_name, sizeof a->model_name);
    qj_free(&d);
    return true;
}

/* Opens a session: the guidance and the tools become its prefix. */
static bool open_session(agent *a, const char *title) {
    str b = { 0 };
    str_puts(&b, "{\"system\": ");
    str_jsons(&b, a->guidance);
    str_puts(&b, ", \"tools\": [");
    for (int i = 0; i < AGENT_N_TOOLS; i++) {
        if (i) str_puts(&b, ", ");
        str_puts(&b, AGENT_TOOLS[i]);
    }
    char cwd[1024];
    if (!getcwd(cwd, sizeof cwd)) snprintf(cwd, sizeof cwd, "?");
    str_printf(&b, "], \"thinking\": true, \"effort\": \"%s\", \"metadata\": {\"client\": \"qwasar-agent\", \"cwd\": ",
               a->effort);
    str_jsons(&b, cwd);
    str_puts(&b, ", \"title\": ");
    str_jsons(&b, title ? title : "");
    str_puts(&b, "}}");
    int status = 0;
    qj_doc d;
    const bool reached = api(a, "POST", "/v1/sessions", b.p, &status, &d);
    str_free(&b);
    if (!reached) { qj_free(&d); tui_puts(a->tui, "  the server went away\n"); return false; }
    if (status != 201) { api_error(a, "cannot open a session", status, &d); qj_free(&d); return false; }
    qj_str_copy(&d, qj_root(&d), "id", a->session, sizeof a->session);
    a->prefix_tokens = (int32_t)qj_int_or(&d, qj_root(&d), "prefix_tokens", 0);
    a->ctx_used = 0;
    qj_free(&d);
    return true;
}

/* Parks the session: keeps it warm on disk, frees the server's live slot. */
static void park_session(agent *a) {
    if (!a->session[0]) return;
    char path[96];
    snprintf(path, sizeof path, "/v1/sessions/%s/park", a->session);
    int status = 0;
    qj_doc d;
    api(a, "POST", path, "{}", &status, &d);
    qj_free(&d);
}

/* ---- showing calls ------------------------------------------------------------- */

/* Tool calls are the part of a transcript a person scans for, so they get a
 * marker column and colour rather than being another paragraph of prose.  Long
 * values are elided on one line; the tool output that follows is what matters. */
static void show_tool_call(agent *a, const qw_tool_call *c) {
    const bool tty = tui_is_tty(a->tui);
    tui_printf(a->tui, "%s  %s%s", tty ? "\x1b[36m" : "", c->name, tty ? "\x1b[0m" : "");
    for (int j = 0; j < c->n_params; j++) {
        const char *v = c->params[j].value;
        size_t vl = strlen(v);
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

#define TOOL_RESULT_LINES 3
#define TOOL_RESULT_COLS  72

/* A few dimmed lines, not the whole payload: the model has the full text
 * regardless. */
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

/* ---- one step ------------------------------------------------------------------- */

static void calls_free(agent *a) {
    for (int i = 0; i < a->n_calls; i++) {
        pending_call *p = &a->calls[i];
        free(p->call.name);
        for (int j = 0; j < p->call.n_params; j++) { free(p->call.params[j].key); free(p->call.params[j].value); }
        str_free(&p->result);
    }
    a->n_calls = 0;
}

static char *dupn(const char *s, size_t n) {
    char *p = malloc(n + 1);
    if (p) { memcpy(p, s, n); p[n] = 0; }
    return p;
}

/* A tool_call event into the pending list: arguments come as JSON, and the
 * tools take strings, so a non-string value is its JSON text. */
static void take_call(agent *a, const qj_doc *d) {
    if (a->n_calls >= QW_MAX_CALLS) return;
    pending_call *p = &a->calls[a->n_calls];
    memset(p, 0, sizeof *p);
    const qj_node *root = qj_root(d);
    qj_str_copy(d, root, "id", p->id, sizeof p->id);
    p->call.name = qj_strdup(d, qj_get(d, root, "name"));
    if (!p->call.name) p->call.name = dupn("", 0);
    const qj_node *args = qj_get(d, root, "arguments");
    for (const qj_node *m = qj_first(d, args); m && p->call.n_params < QW_MAX_PARAMS; m = qj_next(d, m)) {
        qw_tool_param *prm = &p->call.params[p->call.n_params++];
        prm->key = dupn(d->text + m->key_off, m->key_len);
        if (m->type == QJ_STRING) prm->value = dupn(d->text + m->u.str.off, m->u.str.len);
        else { str v = { 0 }; str_node(&v, d, m); prm->value = v.p ? v.p : dupn("", 0); }
    }
    a->n_calls++;
}

/* One event, shown. */
static void on_event(agent *a, const char *event, const char *data) {
    qj_doc d;
    if (!qj_parse(&d, data, strlen(data))) { qj_free(&d); return; }
    const qj_node *root = qj_root(&d);
    const bool tty = tui_is_tty(a->tui);

    if (!strcmp(event, "queued")) {
        char s[48];
        snprintf(s, sizeof s, "queued, position %lld", (long long)qj_int_or(&d, root, "position", 0));
        status_set(a, s);
    } else if (!strcmp(event, "resume")) {
        const int32_t restored = (int32_t)qj_int_or(&d, root, "restored", 0);
        char from[16] = "";
        qj_str_copy(&d, root, "from", from, sizeof from);
        if (!strcmp(from, "checkpoint"))
            tui_printf(a->tui, "  \x1b[2m[restored %d tokens from a checkpoint]\x1b[0m\n", restored);
        else if (!strcmp(from, "cold") && restored == 0 && a->ctx_used > 0)
            tui_printf(a->tui, "  \x1b[2m[re-evaluating %d tokens: no checkpoint covered this session]\x1b[0m\n",
                       (int32_t)qj_int_or(&d, root, "prefill", 0));
        a->prefill_started = 0;
        status_set(a, "prefill");
    } else if (!strcmp(event, "prefill")) {
        show_prefill(a, (int32_t)qj_int_or(&d, root, "done", 0), (int32_t)qj_int_or(&d, root, "total", 0));
    } else if (!strcmp(event, "context")) {
        a->ctx_used = (int32_t)qj_int_or(&d, root, "used", a->ctx_used);
    } else if (!strcmp(event, "reasoning")) {
        a->think_tokens += (int32_t)qj_int_or(&d, root, "tokens", 0);
        a->turn_tokens = a->generated + a->think_tokens;
        if (a->cfg.show_think) {
            char *t = qj_strdup(&d, qj_get(&d, root, "text"));
            if (t) {
                if (tty && !a->think_open) { tui_puts(a->tui, "\x1b[2m"); a->think_open = true; }
                tui_puts(a->tui, t);
                free(t);
            }
        }
        status_set(a, "thinking");
    } else if (!strcmp(event, "text")) {
        if (a->think_open) { tui_puts(a->tui, "\x1b[0m\n"); a->think_open = false; }
        char *t = qj_strdup(&d, qj_get(&d, root, "text"));
        if (t) { tui_puts(a->tui, t); free(t); }
        status_set(a, "writing");
    } else if (!strcmp(event, "decode")) {
        a->generated = (int32_t)qj_int_or(&d, root, "generated", a->generated);
        a->tps = qj_num_or(&d, root, "tokens_per_second", a->tps);
        a->turn_tokens = a->generated;
    } else if (!strcmp(event, "call_progress")) {
        char name[64] = "";
        qj_str_copy(&d, root, "name", name, sizeof name);
        char s[96];
        snprintf(s, sizeof s, "calling %s", name[0] ? name : "…");
        if (a->think_open) { tui_puts(a->tui, "\x1b[0m\n"); a->think_open = false; }
        status_set(a, s);
    } else if (!strcmp(event, "tool_call")) {
        if (a->think_open) { tui_puts(a->tui, "\x1b[0m\n"); a->think_open = false; }
        take_call(a, &d);
    } else if (!strcmp(event, "done")) {
        if (a->think_open) { tui_puts(a->tui, "\x1b[0m\n"); a->think_open = false; }
        qj_str_copy(&d, root, "stop", a->stop, sizeof a->stop);
        a->generated = (int32_t)qj_int_or(&d, root, "usage.generated", a->generated);
        a->think_tokens = (int32_t)qj_int_or(&d, root, "usage.reasoning", a->think_tokens);
        a->ctx_used = (int32_t)qj_int_or(&d, root, "context.used", a->ctx_used);
        const double ds = qj_num_or(&d, root, "timing.decode_seconds", 0);
        if (ds > 0) a->tps = a->generated / ds;
        a->turn_tokens = a->generated;
    } else if (!strcmp(event, "error")) {
        if (a->think_open) { tui_puts(a->tui, "\x1b[0m\n"); a->think_open = false; }
        char *m = qj_strdup(&d, qj_get(&d, root, "message"));
        tui_newline(a->tui);
        tui_printf(a->tui, "  server error: %s\n", m ? m : "?");
        free(m);
        snprintf(a->stop, sizeof a->stop, "error");
    }
    qj_free(&d);
}

static void cancel_step(agent *a) {
    char path[96];
    snprintf(path, sizeof path, "/v1/sessions/%s/cancel", a->session);
    int status = 0;
    qj_doc d;
    api(a, "POST", path, "{}", &status, &d);
    qj_free(&d);
}

/* Runs one step -- a turn or a continue -- and reads its stream to the end,
 * watching the keyboard between events so ctrl-C becomes a cancel.  On
 * return a->stop says how it ended and a->calls holds any calls. */
static bool run_step(agent *a, const char *verb, const char *body) {
    calls_free(a);
    a->stop[0] = 0;
    a->generated = 0;
    a->think_tokens = 0;
    a->turn_tokens = 0;
    a->tps = 0;
    a->think_open = false;
    a->turn_started = now_sec();

    char path[96];
    snprintf(path, sizeof path, "/v1/sessions/%s/%s", a->session, verb);
    conn c = { .fd = http_connect(a->host, a->port) };
    if (c.fd < 0) { tui_puts(a->tui, "  the server went away\n"); return false; }
    str carry = { 0 };
    http_resp r;
    if (!http_send_request(&c, "POST", path, a->host, a->token, NULL, body, strlen(body))
        || !http_read_response(&c, &carry, &r)) {
        tui_puts(a->tui, "  the server went away\n");
        close(c.fd); str_free(&carry);
        return false;
    }
    if (strncmp(r.ctype, "text/event-stream", 17)) {
        /* A refusal, as a status with a reason. */
        str out = { 0 };
        http_read_body(&c, &carry, &r, &out);
        qj_doc d;
        memset(&d, 0, sizeof d);
        if (out.len) qj_parse(&d, out.p, out.len);
        api_error(a, "the step was refused", r.status, &d);
        qj_free(&d);
        str_free(&out); str_free(&carry); close(c.fd);
        return false;
    }

    sse_reader rd;
    sse_reader_init(&rd, r.chunked);
    if (carry.len) sse_reader_feed(&rd, carry.p, carry.len);
    str_free(&carry);
    bool cancelled = false;
    bool done = false;
    while (!done) {
        const char *id, *event, *data;
        while (!done && sse_reader_next(&rd, &id, &event, &data)) {
            on_event(a, event, data);
            if (!strcmp(event, "done") || !strcmp(event, "error")) done = true;
        }
        if (done || rd.ended) break;

        struct pollfd fds[2] = { { c.fd, POLLIN, 0 }, { STDIN_FILENO, POLLIN, 0 } };
        const int nfds = tui_is_tty(a->tui) ? 2 : 1;
        const int pr = poll(fds, (nfds_t)nfds, 200);
        if (pr < 0 && errno != EINTR) break;
        if (nfds == 2 && (fds[1].revents & POLLIN)) tui_tick(a->tui);
        if (!cancelled && tui_interrupted(a->tui)) {
            cancelled = true;
            a->interrupted = true;
            cancel_step(a);
            status_set(a, "stopping");
        }
        if (fds[0].revents & (POLLIN | POLLHUP | POLLERR)) {
            char buf[16384];
            const ssize_t n = read(c.fd, buf, sizeof buf);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) break;
            sse_reader_feed(&rd, buf, (size_t)n);
        }
    }
    /* Anything that completed with the last bytes. */
    const char *id, *event, *data;
    while (!done && sse_reader_next(&rd, &id, &event, &data)) {
        on_event(a, event, data);
        if (!strcmp(event, "done") || !strcmp(event, "error")) done = true;
    }
    sse_reader_free(&rd);
    close(c.fd);
    if (!done) {
        tui_newline(a->tui);
        tui_puts(a->tui, "  the stream ended without a result\n");
        return false;
    }
    return strcmp(a->stop, "error") != 0;
}

/* The sampling and budget every step carries. */
static void step_options(agent *a, str *b) {
    str_printf(b, "\"max_tokens\": %d", a->cfg.max_tokens);
    if (a->cfg.temperature >= 0) str_printf(b, ", \"sampling\": {\"temperature\": %.3f}", (double)a->cfg.temperature);
}

/* Runs one task to completion: a turn, and while the model asks for tools,
 * run them and hand the results back. */
static bool agent_run(agent *a, const char *text) {
    str b = { 0 };
    str_puts(&b, "{\"text\": ");
    str_jsons(&b, text);
    if (a->att_b64) {
        str_printf(&b, ", \"images\": [{\"kind\": \"%s\", \"media_type\": \"%s\", \"data\": \"%s\"}]",
                   a->att_video ? "video" : "image", a->att_type, a->att_b64);
        free(a->att_b64);
        a->att_b64 = NULL;
    }
    str_puts(&b, ", ");
    step_options(a, &b);
    str_puts(&b, "}");
    a->interrupted = false;
    bool ok = run_step(a, "turn", b.p);
    str_free(&b);

    int step = 0;
    while (ok && !strcmp(a->stop, "tool_calls")) {
        str res = { 0 };
        str_puts(&res, "{\"results\": [");
        for (int i = 0; i < a->n_calls; i++) {
            pending_call *p = &a->calls[i];
            tui_newline(a->tui);
            show_tool_call(a, &p->call);
            status_set(a, p->call.name);
            dispatch(&p->call, &a->cfg, &p->result);
            show_tool_result(a, &p->result);
            if (i) str_puts(&res, ", ");
            str_printf(&res, "{\"id\": \"%s\", \"content\": ", p->id);
            str_jsons(&res, p->result.p ? p->result.p : "");
            str_puts(&res, "}");
        }
        str_puts(&res, "], ");
        step_options(a, &res);
        str_puts(&res, "}");
        if (++step >= a->cfg.max_steps) {
            tui_printf(a->tui, "  [stopped after %d tool calls]\n", step);
            str_free(&res);
            calls_free(a);
            return true;
        }
        ok = run_step(a, "continue", res.p);
        str_free(&res);
    }
    calls_free(a);
    tui_newline(a->tui);
    if (!ok) return false;

    if (!strcmp(a->stop, "cancelled") || a->interrupted) {
        tui_puts(a->tui, "  [interrupted]\n");
    } else if (!strcmp(a->stop, "length")) {
        if (a->generated > 0 && a->think_tokens >= a->generated)
            tui_printf(a->tui, "  [stopped at the %d-token budget while still reasoning, so there is "
                       "no answer to show; raise it with -n, or use /effort low]\n", a->cfg.max_tokens);
        else
            tui_printf(a->tui, "  [stopped at the %d-token budget, %d of them reasoning; raise it with -n]\n",
                       a->cfg.max_tokens, a->think_tokens);
    } else if (!strcmp(a->stop, "context_full")) {
        tui_puts(a->tui, "  [the context is full; /new starts over]\n");
    }
    status_set(a, a->interrupted ? "interrupted" : !strcmp(a->stop, "length") ? "truncated" : "done");
    return true;
}

/* ---- attachments ------------------------------------------------------------------ */

static const char *media_type(const char *path, bool video) {
    const char *dot = strrchr(path, '.');
    const char *ext = dot ? dot + 1 : "";
    if (video) return !strcasecmp(ext, "mov") ? "video/quicktime" : "video/mp4";
    if (!strcasecmp(ext, "png")) return "image/png";
    if (!strcasecmp(ext, "gif")) return "image/gif";
    if (!strcasecmp(ext, "bmp")) return "image/bmp";
    return "image/jpeg";
}

static bool attach(agent *a, const char *path, bool video) {
    str bytes = { 0 };
    FILE *f = fopen(path, "rb");
    if (!f) { tui_printf(a->tui, "  cannot read %s: %s\n", path, strerror(errno)); return false; }
    char buf[65536];
    size_t n;
    while ((n = fread(buf, 1, sizeof buf, f)) > 0) str_add(&bytes, buf, n);
    fclose(f);
    free(a->att_b64);
    a->att_b64 = b64_encode((const unsigned char *)bytes.p, bytes.len);
    snprintf(a->att_type, sizeof a->att_type, "%s", media_type(path, video));
    a->att_video = video;
    tui_printf(a->tui, "  %s (%zu bytes) goes with your next message\n", video ? "video" : "image", bytes.len);
    str_free(&bytes);
    return a->att_b64 != NULL;
}

/* ---- repl ------------------------------------------------------------------- */

static void repl_help(agent *a) {
    tui_puts(a->tui,"  /help            this message\n"
           "  /new             start a fresh conversation (this one is parked)\n"
           "  /sessions        this directory's conversations on the server\n"
           "  /image <path>    attach an image to the next message\n"
           "  /video <path>    attach a video\n"
           "  /effort <level>  xhigh, medium or low: a new conversation at that effort\n"
           "  /think           show or hide the reasoning block\n"
           "  /yes             toggle asking before writes and commands\n"
           "  /ctx             context used, and how warm the session is\n"
           "  /save            park: keep the conversation warm on disk\n"
           "  /quit            leave\n");
}

static void show_sessions(agent *a) {
    int status = 0;
    qj_doc d;
    if (!api(a, "GET", "/v1/sessions", NULL, &status, &d) || status != 200) {
        tui_puts(a->tui, "  cannot list sessions\n"); qj_free(&d); return;
    }
    char cwd[1024];
    if (!getcwd(cwd, sizeof cwd)) cwd[0] = 0;
    int shown = 0;
    for (const qj_node *s = qj_first(&d, qj_get(&d, qj_root(&d), "sessions")); s; s = qj_next(&d, s)) {
        char client[32] = "", scwd[1024] = "", title[64] = "", id[32] = "", warm[8] = "", state[24] = "";
        qj_str_copy(&d, s, "metadata.client", client, sizeof client);
        qj_str_copy(&d, s, "metadata.cwd", scwd, sizeof scwd);
        if (strcmp(client, "qwasar-agent") || strcmp(scwd, cwd)) continue;
        qj_str_copy(&d, s, "metadata.title", title, sizeof title);
        qj_str_copy(&d, s, "id", id, sizeof id);
        qj_str_copy(&d, s, "warmth.state", warm, sizeof warm);
        qj_str_copy(&d, s, "state", state, sizeof state);
        tui_printf(a->tui, "  %s%s  %6lld tokens  %-5s %s%s\n",
                   !strcmp(id, a->session) ? "* " : "  ", id,
                   (long long)qj_int_or(&d, s, "tokens", 0), warm,
                   title[0] ? title : "(untitled)", strcmp(state, "idle") ? "  [awaiting tools]" : "");
        shown++;
    }
    if (!shown) tui_puts(a->tui, "  no conversations for this directory\n");
    else tui_puts(a->tui, "  resume one with: qwasar-agent --resume <id>\n");
    qj_free(&d);
}

static void show_ctx(agent *a) {
    char path[96];
    snprintf(path, sizeof path, "/v1/sessions/%s", a->session);
    int status = 0;
    qj_doc d;
    if (!api(a, "GET", path, NULL, &status, &d) || status != 200) { qj_free(&d); return; }
    const qj_node *r = qj_root(&d);
    char warm[8] = "";
    qj_str_copy(&d, r, "warmth.state", warm, sizeof warm);
    tui_printf(a->tui, "  %lld of %lld tokens used  ·  session %s is %s%s\n",
               (long long)qj_int_or(&d, r, "tokens", 0), (long long)qj_int_or(&d, r, "context", 0),
               a->session, warm, !strcmp(warm, "live") ? " on the server" : " on disk");
    qj_free(&d);
}

/* Returns false when the command asked to quit; *reopen when a new session
 * is wanted. */
static bool repl_command(agent *a, const char *line, bool *reopen) {
    *reopen = false;
    if (!strcmp(line, "/quit") || !strcmp(line, "/exit")) return false;
    if (!strcmp(line, "/help")) { repl_help(a); return true; }
    if (!strcmp(line, "/ctx")) { show_ctx(a); return true; }
    if (!strcmp(line, "/sessions")) { show_sessions(a); return true; }
    if (!strcmp(line, "/save")) {
        park_session(a);
        tui_printf(a->tui, "  parked %s; it resumes from disk\n", a->session);
        return true;
    }
    if (!strncmp(line, "/image", 6) || !strncmp(line, "/video", 6)) {
        const bool as_video = line[1] == 'v';
        const char *path = line + 6;
        while (*path == ' ') path++;
        if (!*path) tui_printf(a->tui, "  usage: %s <path>\n", as_video ? "/video" : "/image");
        else attach(a, path, as_video);
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
            /* Effort is part of the prefix, so it is a new session. */
            a->effort = !strcmp(lvl, "low") ? "low" : !strcmp(lvl, "medium") ? "medium" : "xhigh";
            *reopen = true;
        } else {
            tui_printf(a->tui, "  effort must be xhigh, medium or low\n");
        }
        return true;
    }
    if (!strcmp(line, "/new")) { *reopen = true; return true; }
    tui_printf(a->tui, "  unknown command; try /help\n");
    return true;
}

static void usage(FILE *out) {
    fprintf(out,
        "qwasar-agent -- an agentic loop on qwasar-server\n"
        "\n"
        "usage: qwasar-agent [options] [task...]\n"
        "\n"
        "With a task it runs once and exits.  With no task it opens a REPL.\n"
        "Talks to qwasar-server's Session API; if nothing is listening it starts\n"
        "a server itself, which stops when the agent does.\n"
        "\n"
        "      --server <url>   the server (default $QWASAR_SERVER or http://127.0.0.1:8080)\n"
        "      --token <t>      its bearer token, if it needs one\n"
        "  -m, --model <dir>    model directory, for a server this agent starts\n"
        "      --resume <id>    continue a conversation (/sessions lists them; `last`\n"
        "                       is this directory's most recent)\n"
        "  -C, --chdir <dir>    work in this directory\n"
        "  -y, --yes            do not ask before writing files or running commands\n"
        "  -i, --interactive    open the REPL after running the task\n"
        "      --steps <n>      maximum tool calls per task (default 24)\n"
        "  -n, --predict <n>    maximum tokens per step (default 8192)\n"
        "      --temperature <t> sampling temperature (default: the model's; 0 is greedy)\n"
        "      --image <path>   an image for the first turn (jpeg, png, bmp, gif)\n"
        "      --video <path>   a video for the first turn\n"
        "      --effort <lvl>   reasoning effort: xhigh (default), medium, low\n"
        "      --show-think     print the reasoning block\n"
        "  -h, --help           this message\n"
        "\n"
        "Tools: read, write, edit, list, grep, bash.  Reading runs unattended;\n"
        "writing and running commands ask first unless --yes.\n"
        "If AGENT.md exists in the working directory it is added to the system\n"
        "prompt as project guidance.\n");
}

static bool parse_server(agent *a, const char *url) {
    const char *p = url;
    if (!strncmp(p, "http://", 7)) p += 7;
    else if (!strncmp(p, "https://", 8)) { fprintf(stderr, "qwasar-agent: https is not supported\n"); return false; }
    const char *colon = strrchr(p, ':');
    const char *slash = strchr(p, '/');
    size_t hl = colon ? (size_t)(colon - p) : slash ? (size_t)(slash - p) : strlen(p);
    if (hl == 0 || hl >= sizeof a->host) return false;
    memcpy(a->host, p, hl);
    a->host[hl] = 0;
    if (!strcmp(a->host, "localhost")) snprintf(a->host, sizeof a->host, "127.0.0.1");
    a->port = colon ? atoi(colon + 1) : 80;
    return a->port > 0;
}

/* The most recent of this directory's sessions, for --resume last. */
static bool resume_last(agent *a) {
    int status = 0;
    qj_doc d;
    if (!api(a, "GET", "/v1/sessions", NULL, &status, &d) || status != 200) { qj_free(&d); return false; }
    char cwd[1024];
    if (!getcwd(cwd, sizeof cwd)) cwd[0] = 0;
    bool found = false;
    for (const qj_node *s = qj_first(&d, qj_get(&d, qj_root(&d), "sessions")); s && !found; s = qj_next(&d, s)) {
        char client[32] = "", scwd[1024] = "";
        qj_str_copy(&d, s, "metadata.client", client, sizeof client);
        qj_str_copy(&d, s, "metadata.cwd", scwd, sizeof scwd);
        if (strcmp(client, "qwasar-agent") || strcmp(scwd, cwd)) continue;
        qj_str_copy(&d, s, "id", a->session, sizeof a->session);
        a->ctx_used = (int32_t)qj_int_or(&d, s, "tokens", 0);
        found = true;
    }
    qj_free(&d);
    return found;
}

int main(int argc, char **argv) {
    agent a;
    memset(&a, 0, sizeof a);
    /* The step budget bounds a runaway generation; it is not meant to bound
     * ordinary work.  8192 leaves several turns inside a 32K context; the
     * server caps it to the room the window has. */
    a.cfg = (agent_cfg){ .yes = false, .show_think = false, .max_steps = 24, .max_tokens = 8192,
                         .temperature = -1 };
    a.effort = "xhigh";
    a.lifeline = -1;
    const char *server = getenv("QWASAR_SERVER");
    const char *model = NULL, *workdir = NULL, *image_path = NULL, *resume = NULL;
    bool image_is_video = false, interactive = false;
    str task = { 0 };

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if ((!strcmp(arg, "-m") || !strcmp(arg, "--model")) && i + 1 < argc) model = argv[++i];
        else if (!strcmp(arg, "--server") && i + 1 < argc) server = argv[++i];
        else if (!strcmp(arg, "--token") && i + 1 < argc) a.token = argv[++i];
        else if (!strcmp(arg, "--resume") && i + 1 < argc) resume = argv[++i];
        else if ((!strcmp(arg, "-C") || !strcmp(arg, "--chdir")) && i + 1 < argc) workdir = argv[++i];
        else if (!strcmp(arg, "-y") || !strcmp(arg, "--yes")) a.cfg.yes = true;
        else if (!strcmp(arg, "-i") || !strcmp(arg, "--interactive")) interactive = true;
        else if (!strcmp(arg, "--image") && i + 1 < argc) image_path = argv[++i];
        else if (!strcmp(arg, "--video") && i + 1 < argc) { image_path = argv[++i]; image_is_video = true; }
        else if (!strcmp(arg, "--steps") && i + 1 < argc) a.cfg.max_steps = atoi(argv[++i]);
        else if ((!strcmp(arg, "-n") || !strcmp(arg, "--predict")) && i + 1 < argc) a.cfg.max_tokens = atoi(argv[++i]);
        else if (!strcmp(arg, "--temperature") && i + 1 < argc) a.cfg.temperature = (float)atof(argv[++i]);
        else if (!strcmp(arg, "--effort") && i + 1 < argc) a.effort = argv[++i];
        else if (!strcmp(arg, "--show-think")) a.cfg.show_think = true;
        else if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) { usage(stdout); return 0; }
        else if (arg[0] == '-') { fprintf(stderr, "qwasar-agent: unknown argument '%s'\n\n", arg); usage(stderr); return 2; }
        else { if (task.len) str_puts(&task, " "); str_puts(&task, arg); }
    }
    if (strcmp(a.effort, "xhigh") && strcmp(a.effort, "medium") && strcmp(a.effort, "low")) {
        fprintf(stderr, "qwasar-agent: effort must be xhigh, medium or low\n");
        return 2;
    }
    if (!parse_server(&a, server ? server : "http://127.0.0.1:8080")) {
        fprintf(stderr, "qwasar-agent: bad --server url\n");
        return 2;
    }
    if (workdir && chdir(workdir) != 0) {
        fprintf(stderr, "qwasar-agent: cannot enter %s: %s\n", workdir, strerror(errno));
        return 1;
    }
    if (!task.len) interactive = true;
    signal(SIGPIPE, SIG_IGN);

    a.tui = tui_new();
    g_tui = a.tui;

    /* A server, ours if there is none. */
    if (!server_up(&a)) {
        const bool local = !strcmp(a.host, "127.0.0.1");
        char mbuf[1200];
        const char *m = local ? resolve_model(model, mbuf, sizeof mbuf) : NULL;
        if (!local || !m) {
            fprintf(stderr, "qwasar-agent: nothing is listening at http://%s:%d%s\n", a.host, a.port,
                    local ? ", and no model was found to start a server with.\n"
                            "Start the Qwasar app, or pass -m <model-dir> (or set QWASAR_MODEL)"
                          : "");
            tui_free(a.tui);
            return 1;
        }
        if (!server_start(&a, m)) { tui_free(a.tui); return 1; }
    }
    if (!server_info(&a)) {
        fprintf(stderr, "qwasar-agent: the server at http://%s:%d does not speak the Session API\n", a.host, a.port);
        tui_free(a.tui);
        return 1;
    }

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
    a.guidance = guidance.p;

    if (resume) {
        if (!strcmp(resume, "last")) {
            if (!resume_last(&a)) { fprintf(stderr, "qwasar-agent: no conversation to resume here\n"); tui_free(a.tui); return 1; }
        } else {
            snprintf(a.session, sizeof a.session, "%s", resume);
        }
        char path[96];
        snprintf(path, sizeof path, "/v1/sessions/%s", a.session);
        int status = 0;
        qj_doc d;
        if (!api(&a, "GET", path, NULL, &status, &d) || status != 200) {
            fprintf(stderr, "qwasar-agent: no session %s on the server\n", a.session);
            qj_free(&d); tui_free(a.tui); return 1;
        }
        a.ctx_used = (int32_t)qj_int_or(&d, qj_root(&d), "tokens", 0);
        char warm[8] = "";
        qj_str_copy(&d, qj_root(&d), "warmth.state", warm, sizeof warm);
        qj_free(&d);
        tui_printf(a.tui, "\x1b[2mresuming %s: %d tokens, %s\x1b[0m\n", a.session, a.ctx_used, warm);
    } else if (!open_session(&a, task.len ? task.p : NULL)) {
        tui_free(a.tui);
        return 1;
    }
    if (image_path && !attach(&a, image_path, image_is_video)) { tui_free(a.tui); return 1; }

    tui_printf(a.tui, "\x1b[2m%s  ·  %d tools  ·  %d-token prefix  ·  %s\x1b[0m\n",
               a.model_name[0] ? a.model_name : "connected", AGENT_N_TOOLS, a.prefix_tokens,
               a.cfg.yes ? "not asking before writes" : "asking before writes");

    int rc = 0;
    if (task.len && !agent_run(&a, task.p)) rc = 1;

    if (interactive && rc == 0) {
        static const char *const COMMANDS[] = {
            "/help", "/new", "/sessions", "/effort ", "/think", "/yes", "/ctx", "/save",
            "/image ", "/video ", "/quit", NULL
        };
        tui_set_commands(COMMANDS);

        char hist[1024] = "";
        const char *home = getenv("HOME");
        if (home) {
            snprintf(hist, sizeof hist, "%s/.cache/qwasar/history", home);
            tui_history_load(a.tui, hist);
        }

        if (tui_is_tty(a.tui))
            tui_printf(a.tui, "\n\x1b[1mqwasar-agent\x1b[0m  \x1b[2m%d tools  ·  "
                              "/help for commands  ·  ctrl-C interrupts\x1b[0m\n\n", AGENT_N_TOOLS);
        else
            tui_puts(a.tui, "\nqwasar-agent. /help for commands, /quit to leave.\n\n");

        for (;;) {
            a.turn_tokens = 0;
            status_set(&a, "ready");
            char *line = tui_readline(a.tui, "\x1b[1;32m>\x1b[0m ");
            if (!line) break;                        /* ctrl-D */
            if (!*line) { free(line); continue; }
            tui_history_add(a.tui, line);
            if (hist[0]) tui_history_save(a.tui, hist);

            if (line[0] == '/') {
                bool reopen = false;
                const bool keep = repl_command(&a, line, &reopen);
                free(line);
                if (!keep) break;
                if (reopen) {
                    park_session(&a);
                    if (!open_session(&a, NULL)) { rc = 1; break; }
                    tui_printf(a.tui, "  new conversation at effort %s\n", a.effort);
                }
                continue;
            }
            const bool ok = agent_run(&a, line);
            free(line);
            if (!ok) {
                if (!server_up(&a)) { tui_puts(a.tui, "  the server is gone\n"); rc = 1; break; }
            }
        }
    }

    park_session(&a);
    tui_free(a.tui);
    str_free(&task);
    str_free(&guidance);
    free(a.att_b64);
    /* A server we started goes with us: closing the lifeline is enough, and
     * it checkpoints on the way out. */
    if (a.lifeline >= 0) close(a.lifeline);
    return rc;
}
