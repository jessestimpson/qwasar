#ifndef QWASAR_H
#define QWASAR_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

/* Public engine boundary.
 *
 * `qwasar_engine` is the loaded model: immutable weights plus the compiled GPU
 * pipelines, shared by every session.  `qwasar_session` is one mutable
 * inference timeline -- it owns the KV cache for the 16 full-attention layers
 * and the recurrent state for the 48 gated-delta layers.
 *
 * Keep this header narrow: CLI, agent, and server code must never need to know
 * what a tensor looks like. */

typedef struct qwasar_engine  qwasar_engine;
typedef struct qwasar_session qwasar_session;

typedef struct {
    const char *model_path;   /* directory holding config.json + *.safetensors */
    int         context_size;  /* max tokens; 0 = engine default (32768) */
    int         prefill_chunk; /* tokens per forward pass; 0 = default (256) */
    bool        verbose;
} qwasar_options;

/* Loads config and binds weights.  Returns NULL and fills `err` on failure. */
qwasar_engine *qwasar_engine_load(const qwasar_options *opts, char *err, size_t errcap);
void           qwasar_engine_free(qwasar_engine *e);

/* Human-readable architecture and weight inventory. */
void qwasar_engine_print_info(const qwasar_engine *e, FILE *out);

int32_t qwasar_vocab_size(const qwasar_engine *e);
int32_t qwasar_n_layers(const qwasar_engine *e);
/* True if `token` ends generation. */
bool    qwasar_is_eos(const qwasar_engine *e, int32_t token);

/* ---- tokenizer ------------------------------------------------------------
 *
 * Decoding only, for now; BPE encoding lands with the chat template. */

typedef struct qwasar_tokenizer qwasar_tokenizer;

qwasar_tokenizer *qwasar_tokenizer_load(const char *model_path, char *err, size_t errcap);
void              qwasar_tokenizer_free(qwasar_tokenizer *t);
int32_t           qwasar_tokenizer_size(const qwasar_tokenizer *t);

/* Raw UTF-8 bytes for one token.  `special` reports control tokens such as
 * <|im_end|>, which callers usually suppress from display. */
const char *qwasar_token_bytes(const qwasar_tokenizer *t, int32_t id,
                               size_t *len, bool *special);

/* Id of a control token by its literal spelling, e.g. "<|im_end|>", or -1. */
int32_t qwasar_token_id(const qwasar_tokenizer *t, const char *literal);

/* Encodes plain text.  Control tokens are never produced, however the text is
 * spelled, so untrusted content cannot introduce a role marker.  Caller frees. */
int32_t *qwasar_encode(const qwasar_tokenizer *t, const char *text, int32_t *out_n);

/* ---- chat template --------------------------------------------------------- */

typedef struct {
    const char *role;       /* "system" | "user" | "assistant" | "tool" */
    const char *content;
    const char *reasoning;  /* assistant only; may be NULL */
} qwasar_message;

typedef struct {
    bool        enable_thinking;       /* default on for this model */
    const char *reasoning_effort;      /* "xhigh" (default), "medium", "low" */
    bool        add_generation_prompt;
    /* Tool definitions as JSON objects, one per tool.  When present the system
     * turn is rebuilt around them, as the model's own template does: the call
     * format is XML rather than JSON, and the format description is part of the
     * prompt the model was trained against. */
    const char *const *tools;
    int32_t            n_tools;
} qwasar_chat_options;

/* Renders the conversation as ChatML and encodes it.  Caller frees the result. */
int32_t *qwasar_apply_chat_template(const qwasar_tokenizer *t,
                                    const qwasar_message *msgs, int32_t n_msgs,
                                    const qwasar_chat_options *opts,
                                    int32_t *out_n, char *err, size_t errcap);

/* Continuations, for an agent loop that feeds back the tokens it generated
 * rather than re-rendering the whole conversation each turn.  Both close the
 * open assistant turn, add their own turn, and reopen a generation prompt.
 * Caller frees. */
int32_t *qwasar_render_tool_result(const qwasar_tokenizer *t, const char *result,
                                   const qwasar_chat_options *opts, int32_t *out_n);
int32_t *qwasar_render_user_turn(const qwasar_tokenizer *t, const char *text,
                                 const qwasar_chat_options *opts, int32_t *out_n);

/* ---- sessions -------------------------------------------------------------
 *
 * A session owns one inference timeline: the KV cache for the 16 full-attention
 * layers, and the recurrent conv and delta-rule state for the other 48.
 *
 * That second half is why a session is append-only.  A KV cache can be
 * truncated to any prefix, but a recurrent state keeps no per-position history,
 * so there is nothing to roll back to.  Extending a session is cheap; rewinding
 * it means starting over.  Callers that need to edit earlier turns must build a
 * fresh session and re-evaluate. */

qwasar_session *qwasar_session_new(qwasar_engine *e, char *err, size_t errcap);
void            qwasar_session_free(qwasar_session *s);

/* Appends `n` tokens and returns logits for the last one, valid until the next
 * eval on this session.  Returns NULL and fills `err` on failure. */
const float *qwasar_session_eval(qwasar_session *s, const int32_t *tokens, int32_t n,
                                 char *err, size_t errcap);

int32_t qwasar_session_n_past(const qwasar_session *s);

/* Diagnostic: record the residual stream for the final token after each of the
 * given layers, so a divergence can be located rather than just detected.
 * Pass n = 0 to turn off.  Returns false if the capture buffer cannot be sized. */
bool         qwasar_session_set_capture(qwasar_session *s, const int32_t *layers, int32_t n);
const float *qwasar_session_captured(const qwasar_session *s, int32_t which);

#endif /* QWASAR_H */
