#ifndef QWASAR_TUI_H
#define QWASAR_TUI_H

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>

/* Terminal UI for qwasar-agent.
 *
 * The layout is the one ds4-agent uses, and it is worth describing because it
 * is not the obvious one.  There is no alternate screen: everything stays in
 * the terminal's normal scrollback, so a finished session can be scrolled,
 * copied and searched like any other command output.
 *
 * What makes it feel like an application is a scroll region.  The bottom rows
 * are reserved for the prompt and a status footer; DECSTBM confines scrolling
 * to the rows above them, so streamed output scrolls naturally underneath a
 * prompt that never moves and never has to be redrawn per token.  That last
 * part matters over SSH, where a full redraw on every token is visible.
 *
 * Output owns its row rather than tracking a column.  The bytes of the row
 * being written are kept here and the whole row is repainted from column one on
 * every write, so the terminal's cursor position is never inferred -- it is
 * told.  Tabs are expanded here and double-width characters are measured here,
 * because both were enough to desynchronise a column counter and start
 * overwriting text mid-line.  Autowrap is disabled inside the region as a
 * guard: rows are broken at a width this code computes, and if that width is
 * ever wrong the row is clipped instead of the layout coming apart.
 *
 * The footer stays up while the model generates, and anything typed is fed
 * straight into the line editor, so a message can be composed, edited and
 * submitted mid-turn; submitted ones queue for the following turns.
 *
 * Everything degrades to plain reads and writes when stdout is not a terminal,
 * so piping the agent still works and produces clean text. */

typedef struct qw_tui qw_tui;

/* Returns a TUI in whichever mode the terminal supports.  Never fails: a pipe
 * or a terminal too small for a reserved footer yields a plain-IO fallback. */
qw_tui *tui_new(void);
void    tui_free(qw_tui *t);

bool    tui_is_tty(const qw_tui *t);
int     tui_width(const qw_tui *t);

/* Reads one line.  Returns NULL on end of input.  Caller frees. */
char   *tui_readline(qw_tui *t, const char *prompt);

/* Streams output into the scrolling region above the footer.  Safe to call as
 * often as one token at a time. */
void    tui_out(qw_tui *t, const char *s, size_t n);
void    tui_puts(qw_tui *t, const char *s);
void    tui_printf(qw_tui *t, const char *fmt, ...);

/* Replaces the footer text.  Cheap: the footer is only repainted when
 * tui_tick() decides enough time has passed. */
void    tui_status(qw_tui *t, const char *fmt, ...);
void    tui_status_clear(qw_tui *t);

/* Repaints the footer if it is due.  Call from inside a generation loop. */
void    tui_tick(qw_tui *t);

/* True if ctrl-C was pressed since the last call.
 *
 * Raw mode clears ISIG, so ctrl-C arrives as a byte on stdin rather than a
 * signal.  Polling for it is therefore both necessary and more reliable than a
 * handler.  Anything else typed while a turn runs is queued for the next
 * prompt instead of being dropped. */
bool    tui_interrupted(qw_tui *t);

/* True if a message was typed and submitted while the model was generating.
 * The next tui_readline returns it without waiting. */
bool    tui_has_queued_line(const qw_tui *t);

/* Forces the footer down so the next output starts on a fresh line. */
void    tui_newline(qw_tui *t);

/* History, persisted across runs. */
void    tui_history_load(qw_tui *t, const char *path);
void    tui_history_add(qw_tui *t, const char *line);
void    tui_history_save(qw_tui *t, const char *path);

/* Completion for the slash commands; `cmds` is NULL-terminated. */
void    tui_set_commands(const char *const *cmds);

#endif /* QWASAR_TUI_H */
