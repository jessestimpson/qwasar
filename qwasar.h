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
    const char *mtp_path;     /* optional MTP draft head directory; NULL to skip */
    int         context_size;  /* max tokens; 0 = engine default (32768) */
    int         prefill_chunk; /* tokens per forward pass; 0 = default (256) */
    bool        verbose;
} qwasar_options;

/* Where the model is when the caller did not say: $QWASAR_MODEL if set, then a
 * `qwasar-model` directory beside the working directory or beside the running
 * binary -- which is what download_model.sh links.  Returns NULL if none of
 * them is a model directory.  The result points at static storage. */
const char *qwasar_default_model_path(void);

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
    /* Assistant tool calls, already rendered in the model's XML call format.
     *
     * A client replaying a conversation sends tool calls back as normalised
     * JSON, but the model has to see what it originally produced, control
     * tokens included.  Content goes through the encoder that never emits
     * control tokens; this field goes through the one that maps them, because
     * it is reconstructed by the server rather than supplied by a user. */
    const char *tool_calls;
    /* Image placeholders to emit before the content, for a user turn: one
     * <|vision_start|>, this many <|image_pad|>, one <|vision_end|>.
     *
     * Declared here rather than written into `content` on purpose.  The
     * content encoder does not honour control tokens -- that is what stops a
     * user's text from injecting them -- so an image has to be stated as
     * structure, and the count has to match the rows the vision tower
     * produced. */
    int32_t     n_image_tokens;
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

/* Logits from this session's last eval, or NULL if it has not run one.
 *
 * A caller that finds the session already sitting at the end of its prompt has
 * nothing left to evaluate, and must not re-run the final token: that would
 * append a duplicate rather than reproduce the step. */
const float *qwasar_session_logits(const qwasar_session *s);

/* Progress during prefill, reported once per chunk.  Prompt processing is the
 * one part of a turn with no visible output, and on a long prompt it is the
 * longest part, so callers that face a person should show it something.
 * Decoding does not report -- its progress is the text appearing. */
typedef void (*qwasar_progress_fn)(void *ud, int32_t done, int32_t total);
void qwasar_session_set_progress(qwasar_session *s, qwasar_progress_fn fn, void *ud);

/* ---- images ----------------------------------------------------------------
 *
 * An image is encoded once by the vision tower and then handed to a session
 * alongside the prompt that references it.  Its rows replace the embeddings of
 * the prompt's <|image_pad|> tokens, and its grid is what makes the three MRoPE
 * position axes diverge -- text advances all three together, an image gives its
 * tokens a frame, a row and a column, and the text after it resumes from one
 * past the largest. */
typedef struct qwasar_image_input {
    const float *rows;                /* [n_rows, hidden], from the tower */
    int32_t      n_rows;
    int32_t      grid_t, grid_h, grid_w;   /* in patches, before the 2x2 merge */
} qwasar_image_input;

/* Like qwasar_session_eval, but with images to place.  `tokens` must contain
 * exactly the pad tokens the grids account for. */
const float *qwasar_session_eval_images(qwasar_session *s, const int32_t *tokens,
                                        int32_t n, const qwasar_image_input *im,
                                        int32_t n_images, char *err, size_t errcap);

/* ---- speculative drafting -------------------------------------------------
 *
 * True when the engine was loaded with an MTP draft head.  The head proposes
 * tokens; it never decides one, so nothing here can change what the model
 * emits -- only how many forward passes it takes to emit it. */
bool qwasar_session_has_mtp(const qwasar_session *s);

/* Longest chained draft the head will produce in one round.  The head's own
 * config says it was trained for a block of 3; this is the ceiling on asking
 * for more than that. */
#define QWASAR_MAX_DRAFT 8

/* Drafts up to `n_draft` tokens for the positions after `emitted`, which must
 * be the token the session's last evaluation produced.  Writes them oldest
 * first and returns how many were produced, or -1 with `err` filled.
 *
 * The first draft costs a pass over the head; each one after it is chained from
 * the head's own output, so acceptance falls with depth.  Committed history is
 * maintained by the session itself: the head attends over the whole conversation
 * so far, which is worth roughly three times the acceptance rate of drafting
 * from a single position. */
int32_t qwasar_session_draft(qwasar_session *s, int32_t emitted,
                             int32_t *drafts, int32_t n_draft,
                             char *err, size_t errcap);

/* Evaluates `block` -- the token already decided, followed by its drafts -- in
 * one pass over the weights, commits the longest correct prefix, and rewinds
 * whatever was rejected.  Writes the committed tokens to `out` (at most
 * n_block of them) and returns how many, or -1 with `err` filled.
 *
 * At least one token is always committed: the pass computes the token that
 * follows the last accepted draft whether or not anything was accepted, so a
 * fully rejected round costs a forward and still advances like an ordinary
 * decode step.
 *
 * The emitted sequence is identical to greedy decoding without any of this.
 * That is the property the whole design exists to keep, and tests/test_verify
 * is what holds it. */
int32_t qwasar_session_verify(qwasar_session *s, const int32_t *block, int32_t n_block,
                              int32_t *out, char *err, size_t errcap);

/* How deep to draft the next round, from the acceptance this session has
 * actually seen and the measured price of a wider verify.  Returns 0 when
 * drafting is not worth it, which is a real answer: turning it on at all costs
 * a third of a decode step, so a stretch the head keeps getting wrong is
 * cheaper decoded serially.
 *
 * Updated by qwasar_session_verify, so it needs no bookkeeping from callers. */
int32_t qwasar_session_draft_depth(qwasar_session *s);

/* ---- sampling ---------------------------------------------------------------
 *
 * Filters apply in a fixed order: temperature, top-k, min-p, top-p.
 * `temperature = 0` means greedy and is exactly reproducible. */
typedef struct {
    float    temperature;
    int32_t  top_k;        /* 0 = off */
    float    top_p;        /* 1 = off */
    float    min_p;        /* 0 = off */
    uint64_t seed;
} qwasar_sampling;

/* The model's own generation_config: temp 1.0, top_k 20, top_p 0.95. */
void    qwasar_sampling_defaults(qwasar_sampling *sp);
int32_t qwasar_sample(const float *logits, int32_t n, const qwasar_sampling *sp,
                      uint64_t *rng);

/* How many leading tokens of `tokens` this session has already evaluated.
 *
 * A stateless client that resends a longer version of the same conversation can
 * continue from here instead of prefilling from zero.  Because the recurrent
 * layers cannot rewind, only a true prefix is reusable: the first differing
 * token makes everything after it worthless, and the caller must start a fresh
 * session. */
int32_t qwasar_session_common_prefix(const qwasar_session *s,
                                     const int32_t *tokens, int32_t n);

/* ---- disk checkpoints -------------------------------------------------------
 *
 * Saves the session so a later run can skip prefilling whatever prefix it
 * shares.  Because 48 of this model's 64 layers are recurrent, a checkpoint
 * carries their state as well as the KV cache, and can only be reused as a
 * strict prefix -- a recurrent state cannot be rewound the way a KV cache can
 * be truncated.
 *
 * qwasar_session_restore fills a *fresh* session from the longest cached prefix
 * of `tokens` and returns how many tokens it covered; the caller then evaluates
 * only the remainder.  Returns 0 when nothing matched, which is not an error. */
bool    qwasar_session_save(qwasar_session *s, const qwasar_engine *e,
                            char *err, size_t errcap);
int32_t qwasar_session_restore(qwasar_session *s, const qwasar_engine *e,
                               const int32_t *tokens, int32_t n);
void    qwasar_kv_cache_stats(uint64_t *bytes, int *entries);

/* Diagnostic: record the residual stream for the final token after each of the
 * given layers, so a divergence can be located rather than just detected.
 * Pass n = 0 to turn off.  Returns false if the capture buffer cannot be sized. */
bool         qwasar_session_set_capture(qwasar_session *s, const int32_t *layers, int32_t n);
const float *qwasar_session_captured(const qwasar_session *s, int32_t which);

#endif /* QWASAR_H */
