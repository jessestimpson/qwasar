#include "qwasar_tui.h"
#include "linenoise.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/select.h>
#include <time.h>
#include <unistd.h>

/* Rows reserved at the bottom: one for the prompt, one for the status. */
#define TUI_RESERVED 2
/* The footer is repainted at most this often.  Streaming can produce hundreds
 * of tokens a second and a footer that fast is unreadable as well as slow. */
#define TUI_REDRAW_INTERVAL 0.10

struct qw_tui {
    struct linenoiseState ls;
    char   buf[8192];
    char   prompt[96];
    char   status[512];

    bool   tty;
    bool   started;     /* a linenoise edit session is open */
    bool   hidden;      /* the footer is currently not on screen */
    bool   region;      /* the scroll region is installed */

    int    rows, cols;
    int    out_bottom;  /* last row of the scrolling output region */
    int    prompt_row;  /* first row of the reserved footer */
    int    out_col;     /* column the next output byte goes to, 0-based */
    bool   line_open;   /* output is mid-line, so a newline is owed */

    double last_draw;
};

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void wr(const char *s, size_t n) {
    while (n) {
        ssize_t k = write(STDOUT_FILENO, s, n);
        if (k <= 0) return;
        s += k;
        n -= (size_t)k;
    }
}
static void wrs(const char *s) { wr(s, strlen(s)); }

static void csi_move(int row, int col) {
    char b[32];
    int n = snprintf(b, sizeof b, "\x1b[%d;%dH", row, col);
    if (n > 0) wr(b, (size_t)n);
}

static bool term_size(int *rows, int *cols) {
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) != 0 || ws.ws_row == 0) return false;
    *rows = ws.ws_row;
    *cols = ws.ws_col;
    return true;
}

/* ---- scroll region ---------------------------------------------------------- */

/* Output always appends at the bottom row of the scroll region: the terminal
 * scrolls the region itself when a line ends there, so the append point never
 * moves.  Only the column has to be tracked.
 *
 * This replaces an earlier DECSC/DECRC scheme, which was wrong in a way that
 * only a real terminal showed: a saved cursor is an absolute screen cell, and
 * scrolling silently invalidates it, so two turns of output landed on the same
 * line and overwrote each other. */
static int advance_col(int col, const char *s, size_t n, int cols) {
    for (size_t i = 0; i < n; i++) {
        if (s[i] == '\x1b') {                     /* escapes occupy no cells */
            size_t j = i + 1;
            if (j < n && s[j] == '[') {
                j++;
                while (j < n && !(s[j] >= '@' && s[j] <= '~')) j++;
            }
            i = j;
            continue;
        }
        if (s[i] == '\n' || s[i] == '\r') { col = 0; continue; }
        if ((unsigned char)s[i] < 0x20) continue;
        /* UTF-8 continuation bytes are part of the cell already counted. */
        if (((unsigned char)s[i] & 0xC0) == 0x80) continue;
        col++;
        if (cols > 0 && col >= cols) col = 0;      /* the terminal wrapped */
    }
    return col;
}

static void region_set_margin(int bottom) {
    char b[32];
    int n = snprintf(b, sizeof b, "\x1b[1;%dr", bottom);
    if (n > 0) wr(b, (size_t)n);
}

static bool region_install(qw_tui *t) {
    if (!t->tty) return false;
    if (!term_size(&t->rows, &t->cols)) return false;
    /* Below this there is no room for a footer and a usable output area, so
     * fall back to plain scrolling rather than squeezing. */
    if (t->rows < 8 || t->cols < 20) return false;

    t->out_bottom = t->rows - TUI_RESERVED;
    t->prompt_row = t->out_bottom + 1;
    t->region = true;

    region_set_margin(t->out_bottom);

    /* Anything already on screen was printed before the region existed.  Open a
     * fresh line at the bottom of the output area rather than assuming the
     * cursor is already there, or the first streamed token overwrites the last
     * startup line. */
    csi_move(t->out_bottom, 1);
    wrs("\n");
    csi_move(t->out_bottom, 1);
    t->out_col = 0;
    return true;
}

static void region_remove(qw_tui *t) {
    if (!t->region) return;
    wrs("\x1b[0m\x1b[r");           /* reset attributes, drop the margin */
    csi_move(t->rows, 1);
    wrs("\r\x1b[0K");
    t->region = false;
}

/* Re-measures on resize.  A terminal that changed size has invalidated both the
 * margin and the saved cursor, so both are re-established. */
static void region_resync(qw_tui *t) {
    if (!t->region) return;
    int r = 0, c = 0;
    if (!term_size(&r, &c) || (r == t->rows && c == t->cols)) return;
    t->rows = r;
    t->cols = c;
    if (r < 8 || c < 20) { region_remove(t); return; }
    t->out_bottom = r - TUI_RESERVED;
    t->prompt_row = t->out_bottom + 1;
    region_set_margin(t->out_bottom);
    t->out_col = 0;
    t->ls.cols = (size_t)c;
    t->ls.oldrows = 0;
    t->ls.oldstatusrows = 0;
}

/* ---- footer ----------------------------------------------------------------- */

static void footer_clear(qw_tui *t) {
    for (int row = t->prompt_row; row <= t->rows; row++) {
        csi_move(row, 1);
        wrs("\r\x1b[0K");
    }
    /* linenoise tracks what it last painted relative to the cursor.  The rows
     * are owned here, so its bookkeeping is reset and the next show becomes a
     * plain write into them. */
    t->ls.oldrows = 0;
    t->ls.oldstatusrows = 0;
    t->ls.oldrpos = 1;
    t->ls.oldpos = t->ls.pos;
}

static void tui_hide(qw_tui *t) {
    if (!t->started || t->hidden) return;
    if (t->region) footer_clear(t);
    else           linenoiseHide(&t->ls);
    t->hidden = true;
}

static void tui_show(qw_tui *t) {
    if (!t->started || !t->hidden) return;
    if (t->region) csi_move(t->prompt_row, 1);
    /* Streamed output can leave SGR attributes set; linenoise assumes it starts
     * from normal. */
    wrs("\x1b[0m");
    linenoiseShow(&t->ls);
    t->hidden = false;
    t->last_draw = now_sec();
}

static void status_paint(qw_tui *t);

/* Puts the footer on screen and leaves the cursor in it, ready for typing.
 * The output position must already be stashed. */
static void footer_enter(qw_tui *t) {
    footer_clear(t);
    status_paint(t);
    csi_move(t->prompt_row, 1);
    wrs("\x1b[0m");
    linenoiseShow(&t->ls);
    t->hidden = false;
    t->last_draw = now_sec();
}

/* Paints the footer and returns the cursor to the output position. */
static void footer_repaint(qw_tui *t) {
    footer_clear(t);
    status_paint(t);
    csi_move(t->prompt_row, 1);
    wrs("\x1b[0m");
    linenoiseShow(&t->ls);
    csi_move(t->out_bottom, t->out_col + 1);   /* back to the append point */
}

/* The status row is painted here rather than handed to linenoise.
 *
 * Letting linenoise own both footer rows meant its relative-motion redraws and
 * its full-width status padding had to agree with the scroll region, and they
 * did not: status lines ended up scrolled into the transcript.  Owning the row
 * outright makes it absolute, truncated to the terminal width, and impossible
 * to wrap. */
static void status_paint(qw_tui *t) {
    if (!t->region) return;
    int row = t->prompt_row + 1;
    if (row > t->rows) return;
    csi_move(row, 1);
    wrs("\r\x1b[0K");
    if (!t->status[0]) return;

    /* One short of the width so writing the last cell cannot trigger a wrap,
     * which at the bottom of the screen would scroll everything. */
    int budget = t->cols - 1;
    wrs("\x1b[2m");
    int cells = 0;
    for (const char *p = t->status; *p && cells < budget; p++) {
        wr(p, 1);
        if (((unsigned char)*p & 0xC0) != 0x80) cells++;   /* count characters */
    }
    wrs("\x1b[0m");
}

static void write_plain(const char *s, size_t n);

/* ---- public ----------------------------------------------------------------- */

static const char *const *g_cmds;

static void complete_cb(const char *line, linenoiseCompletions *lc) {
    if (!g_cmds || line[0] != '/') return;
    size_t n = strlen(line);
    for (const char *const *c = g_cmds; *c; c++)
        if (!strncmp(*c, line, n)) linenoiseAddCompletion(lc, *c);
}

void tui_set_commands(const char *const *cmds) {
    g_cmds = cmds;
    linenoiseSetCompletionCallback(complete_cb);
}

qw_tui *tui_new(void) {
    qw_tui *t = calloc(1, sizeof *t);
    if (!t) return NULL;

    /* Both ends must be a terminal, and it must be able to report its size.
     * When TIOCGWINSZ fails, linenoise falls back to asking the terminal with
     * a cursor-position report and blocks forever if nothing answers -- which
     * is exactly what a pty with no window size does.  Refusing tty mode here
     * avoids the hang without forking linenoise. */
    int r = 0, c = 0;
    t->tty = isatty(STDIN_FILENO) && isatty(STDOUT_FILENO) && term_size(&r, &c);
    if (t->tty) linenoiseSetMultiLine(0);
    return t;
}

void tui_free(qw_tui *t) {
    if (!t) return;
    if (t->started) {
        tui_hide(t);
        linenoiseEditStop(&t->ls);
        t->started = false;
    }
    region_remove(t);
    free(t);
}

bool tui_is_tty(const qw_tui *t) { return t && t->tty; }
int  tui_width(const qw_tui *t) { return (t && t->cols > 0) ? t->cols : 80; }

char *tui_readline(qw_tui *t, const char *prompt) {
    if (!t->tty) {
        /* Piped input: read a line and echo the prompt so a transcript still
         * shows what was asked. */
        char *line = NULL;
        size_t cap = 0;
        ssize_t n = getline(&line, &cap, stdin);
        if (n <= 0) { free(line); return NULL; }
        while (n > 0 && (line[n-1] == '\n' || line[n-1] == '\r')) line[--n] = 0;
        /* Echo through the same filter as everything else, so a transcript
         * shows the prompt without its colour codes. */
        write_plain(prompt, strlen(prompt));
        write_plain(line, (size_t)n);
        fputc('\n', stdout);
        fflush(stdout);
        return line;
    }

    snprintf(t->prompt, sizeof t->prompt, "%s", prompt);

    if (!t->started) {
        /* The region goes in first so linenoise's opening draw can be placed
         * deliberately instead of wherever the cursor happened to be. */
        if (!region_install(t)) t->region = false;
        if (linenoiseEditStart(&t->ls, STDIN_FILENO, STDOUT_FILENO,
                               t->buf, sizeof t->buf, t->prompt) != 0) {
            t->tty = false;
            return tui_readline(t, prompt);
        }
        t->started = true;
        if (t->region) {
            /* That draw landed in the output area; erase it or the first
             * prompt is orphaned in the transcript. */
            csi_move(t->out_bottom, 1);
            wrs("\r\x1b[0K");
            t->out_col = 0;
        }
        t->hidden = true;
    } else if (t->region) {
        /* Anything that touches the edit buffer can trigger a refresh, and
         * linenoise draws wherever the cursor is.  Getting into the footer
         * first is what keeps a prompt from being painted into the transcript
         * and the output column from being left mid-line. */
        if (!t->hidden) tui_hide(t);          /* the stash stays valid */
        footer_clear(t);
        csi_move(t->prompt_row, 1);
        t->ls.prompt = t->prompt;
        t->ls.plen = strlen(t->prompt);
        linenoiseEditClear(&t->ls);
        t->hidden = true;
    } else {
        t->ls.prompt = t->prompt;
        t->ls.plen = strlen(t->prompt);
        linenoiseEditClear(&t->ls);
        t->hidden = true;
    }

    if (t->region) footer_enter(t);
    else tui_show(t);

    for (;;) {
        char *line = linenoiseEditFeed(&t->ls);
        if (line == linenoiseEditMore) { region_resync(t); continue; }
        /* Drop the submitted text before anything repaints, or the footer
         * spends the whole turn showing the line that was already sent.  The
         * cursor goes to the prompt row first because linenoise redraws
         * wherever it happens to be. */
        if (t->region) {
            csi_move(t->prompt_row, 1);
            linenoiseEditClear(&t->ls);
        }
        tui_hide(t);
        if (!line) return NULL;                 /* ctrl-D */
        char *copy = strdup(line);
        linenoiseFree(line);
        /* Echo the submitted line into the transcript.  The footer is
         * ephemeral, so without this the scrollback would be a list of answers
         * with nothing to say what was asked. */
        if (t->region && copy) {
            tui_puts(t, "\x1b[1;32m> \x1b[0m");
            tui_puts(t, copy);
            tui_puts(t, "\n");
        }
        return copy;
    }
}

/* Drops CSI sequences.  Callers write with colour unconditionally and this is
 * the single place that decides whether it reaches the destination, which is
 * far harder to get wrong than gating at every call site. */
static void write_plain(const char *s, size_t n) {
    size_t start = 0;
    for (size_t i = 0; i < n; i++) {
        if (s[i] != '\x1b') continue;
        if (i > start) fwrite(s + start, 1, i - start, stdout);
        size_t j = i + 1;
        if (j < n && s[j] == '[') {
            j++;
            while (j < n && !((s[j] >= '@' && s[j] <= '~'))) j++;
            if (j < n) j++;
        } else if (j < n) {
            j++;                       /* two-byte escape such as ESC 7 */
        }
        i = j - 1;
        start = j;
    }
    if (start < n) fwrite(s + start, 1, n - start, stdout);
}

void tui_out(qw_tui *t, const char *s, size_t n) {
    if (!n) return;
    /* Tracked on every path: callers rely on it to decide whether a newline is
     * owed, and getting it wrong on the plain path ran the next prompt onto the
     * end of the previous answer. */
    t->line_open = (s[n - 1] != '\n');
    if (!t->tty || !t->started) { write_plain(s, n); fflush(stdout); return; }
    if (!t->hidden) tui_hide(t);
    if (t->region) csi_move(t->out_bottom, t->out_col + 1);
    wr(s, n);
    if (t->region) t->out_col = advance_col(t->out_col, s, n, t->cols);
}

void tui_puts(qw_tui *t, const char *s) { if (s) tui_out(t, s, strlen(s)); }

void tui_printf(qw_tui *t, const char *fmt, ...) {
    char b[4096];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(b, sizeof b, fmt, ap);
    va_end(ap);
    if (n > 0) tui_out(t, b, (size_t)n < sizeof b ? (size_t)n : sizeof b - 1);
}

void tui_newline(qw_tui *t) { if (t->line_open) tui_out(t, "\n", 1); }

void tui_status(qw_tui *t, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(t->status, sizeof t->status, fmt, ap);
    va_end(ap);
    /* Painting is deferred to tui_tick so a fast stream cannot spend its time
     * redrawing a line nobody can read at that rate. */
}

void tui_status_clear(qw_tui *t) { t->status[0] = 0; }

void tui_tick(qw_tui *t) {
    if (!t->tty || !t->started || !t->region) return;
    if (now_sec() - t->last_draw < TUI_REDRAW_INTERVAL) return;
    region_resync(t);
    /* The cursor is already stashed, so the footer can be painted and the
     * output position restored without the caller noticing. */
    footer_repaint(t);
    t->last_draw = now_sec();
}

bool tui_interrupted(qw_tui *t) {
    if (!t->tty || !t->started) return false;

    bool hit = false;
    for (;;) {
        char buf[64];
        /* The terminal is in raw non-blocking-friendly mode here only because
         * linenoise put it there; a short read simply means nothing is
         * pending. */
        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(STDIN_FILENO, &fds);
        struct timeval tv = { 0, 0 };
        if (select(STDIN_FILENO + 1, &fds, NULL, NULL, &tv) <= 0) break;

        ssize_t n = read(STDIN_FILENO, buf, sizeof buf);
        if (n <= 0) break;

        /* Keep whatever was typed that is not the interrupt, so a user who
         * starts composing the next message while the model works does not
         * lose it. */
        char keep[64];
        size_t k = 0;
        for (ssize_t i = 0; i < n; i++) {
            if (buf[i] == 3) hit = true;
            else keep[k++] = buf[i];
        }
        if (k) linenoiseEditQueueInput(&t->ls, keep, k);
    }
    return hit;
}

void tui_history_load(qw_tui *t, const char *path) {
    if (t->tty) linenoiseHistoryLoad(path);
}
void tui_history_add(qw_tui *t, const char *line) {
    if (t->tty) linenoiseHistoryAdd(line);
}
void tui_history_save(qw_tui *t, const char *path) {
    if (t->tty) linenoiseHistorySave(path);
}
