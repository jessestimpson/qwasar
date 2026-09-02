/* qwasar_session.h -- the session, shared between the graph files.
 *
 * Internal: qwasar_graph.c owns the 27B's forward and the machinery every
 * session has (chunked eval, the draft head, snapshots); qwasar_flash_graph.c
 * adds the Flash-Next family's forward over the same session.  Neither the
 * CLI nor the agent see this header. */
#ifndef QWASAR_SESSION_H
#define QWASAR_SESSION_H

#include "qwasar.h"
#include "qwasar_gpu.h"
#include "qwasar_model.h"

struct qw_flash_state;

/* Scratch buffers, named by role rather than pooled.  At the default chunk size
 * these total a couple of hundred megabytes against a 25 GB budget, and keeping
 * them distinct makes the forward pass readable and debuggable. */
struct qwasar_session {
    qwasar_engine  *e;
    const qw_config *cfg;
    const qw_shape  *shape;

    int32_t max_ctx;
    int32_t max_rows;     /* prefill chunk */
    int32_t n_past;

    /* Per-layer state, packed across layers of the same kind. */
    qw_buf kcache, vcache;   /* [n_full, kv_heads, max_ctx, head_dim] fp16 */
    qw_buf ssm_state;        /* [n_linear, hv, dv, dk] fp32 */
    qw_buf conv_state;       /* [n_linear, ksize-1, conv_dim] fp32 */
    int32_t *kind_index;     /* layer -> index among its own kind */

    /* rope tables */
    qw_buf rope_axis, rope_inv_freq, positions;

    /* activations */
    qw_buf tokens, h, hn, hn2;
    qw_buf qkv, z, a_proj, b_proj, g, beta;      /* gated delta */
    qw_buf gq, gk, gv, gdn_y, gdn_norm;
    qw_buf qg, q, gate, k, v, attn_out;          /* full attention */
    qw_buf mlp_gate, mlp_up, mlp_act;            /* mlp */
    qw_buf logits;

    /* ---- MTP draft head -------------------------------------------------
     *
     * The head runs only after a base forward has completed, so it borrows
     * that pass's scratch (hn, hn2, qg, q, gate, k, v, attn_out, mlp_*) and
     * needs its own storage only for what has to outlive a step.
     *
     * `mtp_hidden` is [1 + max_rows, hidden].  Row 0 is the PENDING slot and
     * the rest are the last forward's post-norm hidden states, and the offset
     * is the whole bookkeeping: head row p is
     *
     *     fused(embed(token_{p+1}), hidden_p)
     *
     * so a position's hidden state pairs with the NEXT token, and the last
     * position of any forward cannot be committed until that token exists.
     * It waits in row 0 for the following step to supply it. */
    /* Speculative verify: how many leading rows of the current forward should
     * have their recurrent state saved, and where.  Zero on every ordinary
     * pass, so nothing is written and nothing is allocated until a verify
     * needs it.  Layout is [row][layer], so committing a boundary is one
     * contiguous copy rather than a walk. */
    int32_t n_snap;
    qw_buf  ssm_snap, conv_snap, verify_logits;
    int32_t snap_capacity;   /* rows the snapshot buffers were sized for */
    bool    mtp_defer;       /* a verify hands its head upkeep to the next draft */

    /* Committed head rows a verify left owed, carried to the next draft so the
     * two become one pass over the head's weights instead of two.  Head row p
     * needs token p+1, and after a verify the last of those tokens is the one
     * the verify itself produced -- so the rows the verify could not commit and
     * the row the draft wants are contiguous, with contiguous tokens. */
    /* Adaptive depth.  `mtp_p[i]` estimates P(draft i accepted | 0..i-1 were),
     * so these are conditional and multiply into a reach.  `mtp_margin` is the
     * target's own top-2 logit gap at the boundary, which is evidence about the
     * very next token that no amount of history has. */
    double  mtp_p[QWASAR_MAX_DRAFT];
    double  mtp_margin;
    bool    mtp_p_seeded;

    bool    mtp_after_verify;   /* the owed rows' hidden start at mtp_hidden[1] */
    int32_t mtp_owed;           /* how many of them there are; may be zero */
    int32_t mtp_owed_tokens[QWASAR_MAX_DRAFT + 1];

    bool    mtp_on;
    qw_buf  mtp_kcache, mtp_vcache;   /* [kv_heads, max_ctx, head_dim] fp16 */
    int32_t mtp_n_past;               /* committed head rows */
    qw_buf  mtp_hidden;               /* [1 + max_rows, hidden] fp32 */
    bool    mtp_pending;              /* row 0 holds a hidden awaiting its token */
    int32_t mtp_pending_pos;
    qw_buf  mtp_tokens, mtp_positions;
    qw_buf  mtp_embed, mtp_fused, mtp_h, mtp_out, mtp_logits;

    /* Selection results, written by the GPU and read back once the command
     * buffer they were encoded into has completed.  `sel_scratch` is the
     * partial pass's workspace; neither is ever touched by the host between
     * dispatch and wait.  Four kilobytes, so they are not worth deferring. */
    qw_buf  sel_out, sel_scratch;

    /* Acceptance, counted per draft position.  Drafting is free of the
     * exactness surface, so these are the only way to know it is working. */
    int64_t mtp_drafted[QWASAR_MAX_DRAFT];
    int64_t mtp_accepted[QWASAR_MAX_DRAFT];

    qwasar_progress_fn progress;
    void              *progress_ud;

    /* ---- images ---------------------------------------------------------
     *
     * An image reaches the text model as a run of already-encoded rows that
     * replace the embeddings of its <|image_pad|> tokens, and as a stretch of
     * positions where the three MRoPE axes stop agreeing.  Both are prepared
     * for a whole prompt at once and consumed a chunk at a time, because the
     * positions of a later chunk depend on every image before it. */
    qw_buf   img_rows;        /* [total rows, hidden] fp32, all images */
    int32_t  n_img_rows;
    int32_t  img_cursor;      /* rows consumed, so a split run resumes right */
    int32_t *mrope;           /* [3, n] for the attached prompt, or NULL */
    int32_t  mrope_len;
    /* Position bookkeeping only matters once an image has been seen.  Until
     * then all three axes are the token index and n_past is the whole story,
     * which is why this could be deferred for three milestones. */
    bool     mrope_active;
    int32_t  mrope_next;      /* the position a following text token takes */

    /* Every token evaluated, in order.  Needed to key a disk checkpoint and to
     * tell how much of an incoming prompt a checkpoint already covers. */
    int32_t *history;
    int32_t  n_history;

    /* diagnostic capture (see qwasar_session_set_capture) */
    int32_t *capture_layers;
    int32_t  n_capture;
    qw_buf   capture;

    /* Flash-Next (qwen4_exp): the family's own state, NULL for the 27B. */
    struct qw_flash_state *flash;
};


/* A view `elems` floats into a buffer. */
qw_ref qw_off(qw_buf b, size_t elems);

/* Shared encoders. */
void qw_encode_qlinear(qw_cmd c, const qw_qlinear *ql, qw_ref out, qw_ref in, int32_t rows);
/* Gated DeltaNet for one layer, reading s->hn and writing s->hn2; uses the
 * family's output gate (silu for the 27B, sigmoid for qwen4_exp). */
void qw_encode_gated_delta_layer(qwasar_session *s, qw_cmd c, const qw_layer *L,
                                 int32_t li, int32_t rows);

/* The Flash-Next forward (qwasar_flash_graph.c). */
struct qw_flash_state *qw_flash_state_new(qwasar_session *s, char *err, size_t errcap);
void qw_flash_state_free(struct qw_flash_state *f);
/* Host-side work a chunk needs before encoding: the engram ids and rows. */
void qw_flash_prepare_chunk(qwasar_session *s, const int32_t *tokens, int32_t rows);
void qw_flash_encode_forward(qwasar_session *s, qw_cmd c, int32_t rows, bool want_logits);

#endif /* QWASAR_SESSION_H */
