#include "qwasar_tui.h"
#include "linenoise.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <sys/ioctl.h>
#include <sys/select.h>
#include <time.h>
#include <unistd.h>

/* Rows reserved at the bottom: one for the prompt, one for the status. */
#define TUI_RESERVED 2
/* The footer is repainted at most this often.  Streaming can produce hundreds
 * of tokens a second and a footer that fast is unreadable as well as slow. */
#define TUI_REDRAW_INTERVAL 0.10

/* Bytes held for the output row being built.  A row is at most `cols` cells,
 * but a cell can cost four bytes of UTF-8 and rows carry SGR sequences that
 * cost cells nothing, so the buffer is generous rather than exact; appends
 * that would overflow it force the row out early. */
#define TUI_ROW_BYTES 4096

/* Messages that may be composed ahead of the model.  A cap exists because each
 * one commits to a turn whose output nobody has seen yet; when it is reached
 * the editor simply keeps the text instead of accepting it, so a line is never
 * taken and then silently dropped. */
#define TUI_MAX_QUEUED 4

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

    /* The output row under construction, owned outright.
     *
     * An earlier version tracked only a column and appended at it, which meant
     * maintaining a model of where the terminal's cursor was by parsing the
     * bytes going past.  Every gap in that model desynchronised it permanently
     * for the rest of the row, and there were several: a tab moved the real
     * cursor to the next multiple of eight while the model advanced by nothing,
     * and CJK and emoji occupy two cells while the model counted one.  Streamed
     * output containing any of them started overwriting itself.
     *
     * Keeping the row's bytes instead removes the model.  Every write repaints
     * the whole row from column one, so the terminal's cursor is irrelevant --
     * it is told where to be, rather than guessed at. */
    char   row[TUI_ROW_BYTES];
    size_t row_len;
    int    row_cells;   /* display columns the row occupies */
    bool   line_open;   /* output is mid-line, so a newline is owed */

    /* Lines typed and submitted while the model was generating, waiting to be
     * returned by the next tui_readline calls, oldest first. */
    char  *queued[TUI_MAX_QUEUED];
    int    n_queued;

    double last_draw;
};

/* Set from the SIGWINCH handler; a resize invalidates the margin and both
 * footer rows, and nothing else notices until something is repainted. */
static volatile sig_atomic_t g_resized;
static void on_winch(int sig) { (void)sig; g_resized = 1; }

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
 * moves.  Only the row's contents have to be tracked, and they are tracked
 * exactly, because they are ours.
 *
 * This replaces two earlier schemes, both wrong in ways only a real terminal
 * showed.  DECSC/DECRC saved an absolute screen cell that scrolling silently
 * invalidated.  Column counting then replaced it and survived longer, but it
 * was still a model of the terminal's cursor inferred from a byte stream, and
 * tabs and double-width characters were enough to break it.
 *
 * ---- display width -------------------------------------------------------
 *
 * Only wrapping decisions depend on this, and autowrap is disabled inside the
 * region (region_install), so a wrong answer costs a row that breaks early or
 * late rather than a layout that comes apart.  The ranges are the standard
 * double-width blocks plus the emoji planes. */

static int cell_width(uint32_t cp) {
    if (cp == 0x200D) return 0;                        /* zero-width joiner */
    if (cp >= 0x0300 && cp <= 0x036F) return 0;        /* combining marks */
    if (cp >= 0xFE00 && cp <= 0xFE0F) return 0;        /* variation selectors */
    if (cp < 0x1100) return 1;
    if ((cp >= 0x1100 && cp <= 0x115F) ||              /* hangul jamo */
        (cp >= 0x2E80 && cp <= 0xA4CF) ||              /* CJK */
        (cp >= 0xAC00 && cp <= 0xD7A3) ||              /* hangul syllables */
        (cp >= 0xF900 && cp <= 0xFAFF) ||
        (cp >= 0xFE30 && cp <= 0xFE6F) ||
        (cp >= 0xFF00 && cp <= 0xFF60) ||              /* fullwidth forms */
        (cp >= 0xFFE0 && cp <= 0xFFE6) ||
        (cp >= 0x1F300 && cp <= 0x1FAFF) ||            /* emoji */
        (cp >= 0x20000 && cp <= 0x3FFFD))              /* CJK ext */
        return 2;
    return 1;
}

/* Decodes one UTF-8 character, returning its byte length.  Malformed input is
 * consumed one byte at a time as a single cell rather than rejected: this is
 * whatever a tool printed, and refusing to display it would be worse. */
static size_t utf8_next(const char *s, size_t n, uint32_t *cp) {
    unsigned char c = (unsigned char)s[0];
    size_t len = c < 0x80 ? 1 : (c & 0xE0) == 0xC0 ? 2
               : (c & 0xF0) == 0xE0 ? 3 : (c & 0xF8) == 0xF0 ? 4 : 1;
    if (len > n) len = 1;
    if (len == 1) { *cp = c; return 1; }
    uint32_t v = c & (0xFF >> (len + 1));
    for (size_t i = 1; i < len; i++) {
        if (((unsigned char)s[i] & 0xC0) != 0x80) { *cp = c; return 1; }
        v = (v << 6) | ((unsigned char)s[i] & 0x3F);
    }
    *cp = v;
    return len;
}

/* Repaints the row from column one and clears whatever used to follow it. */
static void row_paint(qw_tui *t) {
    csi_move(t->out_bottom, 1);
    if (t->row_len) wr(t->row, t->row_len);
    wrs("\x1b[0m\x1b[0K");
}

/* Paints the row and scrolls the region, leaving the next row empty. */
static void row_flush(qw_tui *t) {
    row_paint(t);
    wrs("\r\n");
    t->row_len = 0;
    t->row_cells = 0;
}

static void row_add(qw_tui *t, const char *s, size_t n) {
    if (t->row_len + n > sizeof t->row - 1) return;
    memcpy(t->row + t->row_len, s, n);
    t->row_len += n;
}

/* Appends output to the row, flushing whenever a row is complete.
 *
 * Everything that is not printable text or an SGR sequence is dropped.  Tool
 * output is whatever a command decided to print, and a stray cursor movement or
 * screen clear from inside it would leave the layout in a state nothing here
 * could recover from; colour is worth keeping and nothing else is. */
static void row_write(qw_tui *t, const char *s, size_t n) {
    for (size_t i = 0; i < n; ) {
        char c = s[i];

        if (c == '\x1b') {
            size_t j = i + 1;
            if (j < n && s[j] == '[') {
                j++;
                while (j < n && !(s[j] >= '@' && s[j] <= '~')) j++;
                if (j < n && s[j] == 'm') row_add(t, s + i, j - i + 1);
                i = (j < n) ? j + 1 : n;
            } else {
                i = (j < n) ? j + 1 : n;   /* two-byte escape, dropped */
            }
            continue;
        }

        if (c == '\n') { row_flush(t); i++; continue; }
        if (c == '\r') { t->row_len = 0; t->row_cells = 0; i++; continue; }
        if (c == '\t') {
            /* Expanded here so the terminal never sees one.  A tab is the
             * clearest case of a byte whose width depends on where it lands,
             * which is exactly what this design refuses to reason about. */
            int stop = (t->row_cells / 8 + 1) * 8;
            while (t->row_cells < stop && t->row_cells < t->cols) {
                row_add(t, " ", 1);
                t->row_cells++;
            }
            i++;
            if (t->row_cells >= t->cols) row_flush(t);
            continue;
        }
        if ((unsigned char)c < 0x20 || (unsigned char)c == 0x7F) { i++; continue; }

        uint32_t cp = 0;
        size_t len = utf8_next(s + i, n - i, &cp);
        int w = cell_width(cp);
        if (t->row_cells + w > t->cols) row_flush(t);
        row_add(t, s + i, len);
        t->row_cells += w;
        i += len;
        if (t->row_cells >= t->cols) row_flush(t);
    }
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

    /* Autowrap off, as a guard rather than a mechanism.  Rows are broken here,
     * at a width this code computes, so the terminal is never asked to wrap
     * one; turning the feature off means that if that width is ever wrong --
     * an emoji sequence measured badly, a script written right to left -- the
     * row is clipped instead of pushing the whole layout down a line. */
    wrs("\x1b[?7l");

    /* Anything already on screen was printed before the region existed.  Open a
     * fresh line at the bottom of the output area rather than assuming the
     * cursor is already there, or the first streamed token overwrites the last
     * startup line. */
    csi_move(t->out_bottom, 1);
    wrs("\n");
    csi_move(t->out_bottom, 1);
    t->row_len = 0;
    t->row_cells = 0;
    return true;
}

static void region_remove(qw_tui *t) {
    if (!t->region) return;
    wrs("\x1b[0m\x1b[?7h\x1b[r");   /* attributes, autowrap, margin */
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
    wrs("\x1b[?7l");
    /* The partial row was measured against the old width and the terminal has
     * already reflowed whatever was on screen.  Sending it out ends the
     * ambiguity in one line rather than carrying it forward. */
    if (t->row_len) row_flush(t);
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
/* Paints the footer and leaves the cursor in it.
 *
 * The cursor stays on the prompt row rather than being returned to the output
 * area, which is what lets someone type while the model is generating: the
 * caret sits where their text is going, and output repaints its own row from
 * column one so it never needs the cursor to be anywhere in particular. */
static void footer_repaint(qw_tui *t) {
    footer_clear(t);
    status_paint(t);
    csi_move(t->prompt_row, 1);
    wrs("\x1b[0m");
    linenoiseShow(&t->ls);
    t->hidden = false;
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
    if (t->tty) {
        linenoiseSetMultiLine(0);
        /* SA_RESTART so a resize does not turn into a short read somewhere
         * that is not expecting one; the flag is all the handler needs to
         * set. */
        struct sigaction sa;
        memset(&sa, 0, sizeof sa);
        sa.sa_handler = on_winch;
        sa.sa_flags = SA_RESTART;
        sigaction(SIGWINCH, &sa, NULL);
    }
    return t;
}

void tui_free(qw_tui *t) {
    if (!t) return;
    for (int i = 0; i < t->n_queued; i++) linenoiseFree(t->queued[i]);
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

/* Piped input: read a line and echo the prompt so a transcript still shows
 * what was asked. */
static char *readline_plain(const char *prompt) {
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

/* Resets the editor for a new read.  `keep` carries a half-typed message
 * across the reset -- composed while the model was writing, it belongs to the
 * next prompt, and the clear that resets linenoise's completion and fold
 * state used to drop it on the floor at every turn boundary. */
static void edit_reset(qw_tui *t, bool keep) {
    char held[sizeof t->buf];
    snprintf(held, sizeof held, "%s", keep ? t->buf : "");
    linenoiseEditClear(&t->ls);
    if (held[0]) {
        snprintf(t->buf, sizeof t->buf, "%s", held);
        t->ls.len = t->ls.pos = strlen(t->buf);
    }
}

/* The interactive read.  `prompt` becomes the footer's prompt for the
 * duration; whether it STAYS the footer's prompt afterwards is the caller's
 * business -- tui_readline leaves it, tui_ask restores what was there.
 * `keep_text` is whether a half-typed line survives into this read. */
static char *readline_tty(qw_tui *t, const char *prompt, bool take_queued, bool keep_text) {
    snprintf(t->prompt, sizeof t->prompt, "%s", prompt);

    /* Composed while the model was still writing, so it is already what the
     * user asked for next; making them press return again would be asking
     * twice.  Never for a question: a message typed ahead is an answer to
     * nothing but the prompt it was typed at. */
    if (take_queued && t->n_queued > 0) {
        char *line = t->queued[0];
        for (int i = 1; i < t->n_queued; i++) t->queued[i - 1] = t->queued[i];
        t->n_queued--;
        if (t->region) {
            row_write(t, prompt, strlen(prompt));
            row_write(t, line, strlen(line));
            row_write(t, "\n", 1);
            row_paint(t);
            footer_repaint(t);
        }
        return line;
    }

    if (!t->started) {
        /* The region goes in first so linenoise's opening draw can be placed
         * deliberately instead of wherever the cursor happened to be. */
        if (!region_install(t)) t->region = false;
        if (linenoiseEditStart(&t->ls, STDIN_FILENO, STDOUT_FILENO,
                               t->buf, sizeof t->buf, t->prompt) != 0) {
            t->tty = false;
            return readline_plain(prompt);
        }
        t->started = true;
        if (t->region) {
            /* That draw landed in the output area; erase it or the first
             * prompt is orphaned in the transcript. */
            csi_move(t->out_bottom, 1);
            wrs("\r\x1b[0K");
            t->row_len = 0;
            t->row_cells = 0;
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
        edit_reset(t, keep_text);
        t->hidden = true;
    } else {
        t->ls.prompt = t->prompt;
        t->ls.plen = strlen(t->prompt);
        edit_reset(t, keep_text);
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
        /* Echo the submitted line into the transcript, under the prompt it
         * answered.  The footer is ephemeral, so without this the scrollback
         * would be a list of answers with nothing to say what was asked. */
        if (t->region && copy) {
            tui_puts(t, prompt);
            tui_puts(t, copy);
            tui_puts(t, "\n");
        }
        return copy;
    }
}

char *tui_readline(qw_tui *t, const char *prompt) {
    if (!t->tty) return readline_plain(prompt);
    return readline_tty(t, prompt, true, true);
}

char *tui_ask(qw_tui *t, const char *prompt) {
    if (!t->tty) return readline_plain(prompt);

    /* What the footer showed before the question, and whatever was being
     * typed into it.  Both come back afterwards: the question is a detour,
     * not a change of prompt -- leaving its label in place was how
     * "proceed? [y/N]" used to sit in the footer for the rest of the turn,
     * long after it had been answered. */
    char prompt_before[sizeof t->prompt];
    char text_before[sizeof t->buf];
    snprintf(prompt_before, sizeof prompt_before, "%s", t->prompt);
    snprintf(text_before, sizeof text_before, "%s", t->started ? t->buf : "");

    char *line = readline_tty(t, prompt, false, false);

    snprintf(t->prompt, sizeof t->prompt, "%s", prompt_before);
    if (t->started) {
        t->ls.prompt = t->prompt;
        t->ls.plen = strlen(t->prompt);
        /* ls.buf IS t->buf; the answer was cleared out of it already. */
        snprintf(t->buf, sizeof t->buf, "%s", text_before);
        t->ls.len = t->ls.pos = strlen(t->buf);
        t->ls.oldpos = 0;
    }
    return line;
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
    if (!t->region) {
        /* A terminal too small for a reserved footer: there is no row to own
         * and no scroll region to protect, so the prompt is taken down for the
         * write the way it always was. */
        if (!t->hidden) tui_hide(t);
        wr(s, n);
        return;
    }

    if (g_resized) { g_resized = 0; region_resync(t); }

    row_write(t, s, n);
    row_paint(t);
    if (!t->row_len) t->line_open = false;   /* the row was flushed */

    /* The footer is never taken down for output any more.  It sits outside the
     * scroll region, so nothing written above can disturb it, and leaving it up
     * means the prompt and the caret stay where someone typing expects them. */
    csi_move(t->prompt_row, 1);
    linenoiseShow(&t->ls);
    t->hidden = false;
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
    if (g_resized) { g_resized = 0; region_resync(t); row_paint(t); }
    if (now_sec() - t->last_draw < TUI_REDRAW_INTERVAL) return;
    /* The cursor is already stashed, so the footer can be painted and the
     * output position restored without the caller noticing. */
    footer_repaint(t);
    t->last_draw = now_sec();
}

/* Feeds anything typed during generation straight into the line editor, so a
 * message can be composed -- and edited, and submitted -- while the model is
 * still writing.  Typing is echoed as it is entered rather than replayed later,
 * which is the difference between a prompt that works and one that only looks
 * like it does.
 *
 * A line finished here is held until the next tui_readline asks for one.
 *
 * Returns true if ctrl-C was pressed.  Raw mode clears ISIG, so it arrives as a
 * byte rather than a signal, and polling for it is both necessary and more
 * reliable than a handler. */
bool tui_interrupted(qw_tui *t) {
    if (!t->tty || !t->started) return false;

    bool hit = false;
    for (;;) {
        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(STDIN_FILENO, &fds);
        struct timeval tv = { 0, 0 };
        if (select(STDIN_FILENO + 1, &fds, NULL, NULL, &tv) <= 0) break;

        char buf[256];
        ssize_t n = read(STDIN_FILENO, buf, sizeof buf);
        if (n <= 0) break;

        /* Everything available goes into the queue in one go.  linenoise reads
         * an escape sequence a byte at a time and falls back to the file
         * descriptor when the queue runs dry, so feeding one byte per call
         * would block halfway through an arrow key. */
        if (linenoiseEditQueueInput(&t->ls, buf, (size_t)n) == -1) break;

        while (linenoiseEditQueuedInput(&t->ls) > 0) {
            /* linenoise redraws relative to the cursor, so it has to be on the
             * prompt row before it is given a chance to. */
            if (t->region) csi_move(t->prompt_row, 1);
            char *line = linenoiseEditFeed(&t->ls);
            if (line == linenoiseEditMore) continue;
            if (line == NULL) {
                /* ctrl-C sets EAGAIN; anything else here is end of input, and
                 * treating that as an interrupt is the safe reading. */
                hit = true;
                linenoiseEditClear(&t->ls);
                break;
            }
            if (t->n_queued < TUI_MAX_QUEUED) {
                t->queued[t->n_queued++] = line;
                linenoiseEditClear(&t->ls);
            } else {
                /* Leave it in the editor rather than accept and discard it.
                 * The text stays on the prompt row, which is its own
                 * explanation. */
                linenoiseFree(line);
            }
        }
    }
    return hit;
}

bool tui_has_queued_line(const qw_tui *t) { return t && t->n_queued > 0; }

void tui_history_load(qw_tui *t, const char *path) {
    if (t->tty) linenoiseHistoryLoad(path);
}
void tui_history_add(qw_tui *t, const char *line) {
    if (t->tty) linenoiseHistoryAdd(line);
}
void tui_history_save(qw_tui *t, const char *path) {
    if (t->tty) linenoiseHistorySave(path);
}
