#ifndef QWASAR_TOOLCALL_H
#define QWASAR_TOOLCALL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Tool-call parsing and file editing for qwasar-agent.
 *
 * Both are pure string processing with no I/O, so they can be tested directly.
 * They are also the two places where a subtle mistake is least visible: a
 * mis-parsed argument silently calls the tool with the wrong value, and a
 * mis-applied edit silently corrupts a file. */

/* ---- tool calls ------------------------------------------------------------
 *
 * Qwen3.8 emits tool calls as XML rather than JSON, in the shape its own chat
 * template documents:
 *
 *     <tool_call>
 *     <function=NAME>
 *     <parameter=KEY>
 *     value, which may span lines
 *     </parameter>
 *     </function>
 *     </tool_call>
 *
 * Values are taken verbatim between the newline after <parameter=KEY> and the
 * newline before </parameter>, so leading and trailing blank lines in a file
 * body survive. A value containing a literal "\n</parameter>" line would
 * terminate early; that is inherent to the format the model was trained on. */

#define QW_MAX_PARAMS 16
#define QW_MAX_CALLS   8

typedef struct {
    char  *key;
    char  *value;
} qw_tool_param;

typedef struct {
    char          *name;
    qw_tool_param  params[QW_MAX_PARAMS];
    int            n_params;
} qw_tool_call;

typedef struct {
    qw_tool_call calls[QW_MAX_CALLS];
    int          n_calls;
    /* Text before the first <tool_call>, which the model may use to narrate the
     * call.  Owned by the parse; NULL if empty. */
    char        *preamble;
} qw_tool_calls;

/* Parses every tool call in `text`.  Returns the number found, or -1 on a
 * malformed call, with `err` describing what was wrong so the agent can hand
 * the message back to the model rather than failing the turn. */
int  qw_tool_parse(const char *text, qw_tool_calls *out, char *err, size_t errcap);
void qw_tool_calls_free(qw_tool_calls *c);

/* Argument lookup by name, or NULL. */
const char *qw_tool_arg(const qw_tool_call *c, const char *key);

/* True once `text` contains a complete tool call, so generation can stop at the
 * closing tag instead of running to the token limit. */
bool qw_tool_call_complete(const char *text, size_t len);

/* ---- file editing ----------------------------------------------------------
 *
 * Conventional line-anchored search and replace: `old` must match a run of
 * whole lines exactly once.  Nothing is guessed -- no fuzzy matching, no anchor
 * markers, no partial-line matches.  An edit either does what it says or it
 * fails and tells the model why. */

typedef enum {
    QW_EDIT_OK = 0,
    QW_EDIT_NOT_FOUND,     /* no run of lines matched */
    QW_EDIT_AMBIGUOUS,     /* matched in more than one place */
    QW_EDIT_EMPTY_OLD,
    QW_EDIT_NOMEM,
} qw_edit_status;

/* Replaces the unique line run matching `old` with `new`.
 *
 * On QW_EDIT_OK, *out is a newly allocated buffer of *out_len bytes that the
 * caller frees.  `matches` always reports how many places matched, which is
 * what the agent tells the model when the edit is rejected. */
qw_edit_status qw_edit_apply(const char *content, size_t content_len,
                             const char *old_text, const char *new_text,
                             char **out, size_t *out_len, int *matches);

const char *qw_edit_status_text(qw_edit_status s);

#endif /* QWASAR_TOOLCALL_H */
