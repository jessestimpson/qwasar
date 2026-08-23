/* Session state and the forward pass.
 *
 * One eval encodes the whole model -- 64 layers, roughly 700 dispatches -- into
 * a single command buffer and submits it once.  Metal's default compute encoder
 * dispatches serially with implicit memory coherence between commands, so the
 * data dependencies below need no explicit barriers; that is load-bearing, and
 * switching the encoder to a concurrent dispatch type would silently break
 * every one of them.
 *
 * Batching is uniform: every kernel takes a `rows` count, so prefill is the
 * same code path as decode with rows > 1.  That is why there is no separate
 * prefill graph. */

#include "qwasar.h"
#include "qwasar_gpu.h"
#include "qwasar_model.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void qw_gerrf(char *err, size_t cap, const char *fmt, ...) {
    if (!err || !cap) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, cap, fmt, ap);
    va_end(ap);
}

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
};

static bool qw_alloc_all(qwasar_session *s, char *err, size_t errcap) {
    const qw_config *c = s->cfg;
    const qw_shape  *sh = s->shape;
    const int32_t R = s->max_rows;

    struct { qw_buf *b; size_t n; const char *name; } want[] = {
        { &s->tokens,   (size_t)R * sizeof(int32_t),          "tokens" },
        { &s->positions,(size_t)3 * R * sizeof(int32_t),      "positions" },
        { &s->h,        (size_t)R * c->hidden_size * 4,       "h" },
        { &s->hn,       (size_t)R * c->hidden_size * 4,       "hn" },
        { &s->hn2,      (size_t)R * c->hidden_size * 4,       "hn2" },

        { &s->qkv,      (size_t)R * sh->conv_dim * 4,         "qkv" },
        { &s->z,        (size_t)R * sh->value_dim * 4,        "z" },
        { &s->a_proj,   (size_t)R * c->linear_num_value_heads * 4, "a" },
        { &s->b_proj,   (size_t)R * c->linear_num_value_heads * 4, "b" },
        { &s->g,        (size_t)R * c->linear_num_value_heads * 4, "g" },
        { &s->beta,     (size_t)R * c->linear_num_value_heads * 4, "beta" },
        { &s->gq,       (size_t)R * sh->key_dim * 4,          "gdn q" },
        { &s->gk,       (size_t)R * sh->key_dim * 4,          "gdn k" },
        { &s->gv,       (size_t)R * sh->value_dim * 4,        "gdn v" },
        { &s->gdn_y,    (size_t)R * sh->value_dim * 4,        "gdn y" },
        { &s->gdn_norm, (size_t)R * sh->value_dim * 4,        "gdn norm" },

        { &s->qg,       (size_t)R * sh->q_proj_out * 4,       "q|gate" },
        { &s->q,        (size_t)R * sh->q_dim * 4,            "q" },
        { &s->gate,     (size_t)R * sh->q_dim * 4,            "gate" },
        { &s->k,        (size_t)R * sh->kv_dim * 4,           "k" },
        { &s->v,        (size_t)R * sh->kv_dim * 4,           "v" },
        { &s->attn_out, (size_t)R * sh->q_dim * 4,            "attn out" },

        { &s->mlp_gate, (size_t)R * c->intermediate_size * 4, "mlp gate" },
        { &s->mlp_up,   (size_t)R * c->intermediate_size * 4, "mlp up" },
        { &s->mlp_act,  (size_t)R * c->intermediate_size * 4, "mlp act" },

        { &s->logits,   (size_t)c->vocab_size * 4,            "logits" },
    };

    for (size_t i = 0; i < sizeof want / sizeof *want; i++) {
        *want[i].b = qw_buf_alloc(want[i].n);
        if (!*want[i].b) {
            qw_gerrf(err, errcap, "cannot allocate %s scratch (%.1f MB)",
                     want[i].name, want[i].n / (1024.0 * 1024.0));
            return false;
        }
    }

    /* Cache and recurrent state.  These are the session's real memory cost:
     * the KV cache is allocated at full context length because the head-major
     * layout cannot grow, but a shared buffer only commits pages on write, so
     * the resident set still tracks how much context is actually used. */
    const size_t kv_per_layer = (size_t)c->num_key_value_heads * s->max_ctx
                              * c->head_dim * sizeof(uint16_t);
    s->kcache = qw_buf_alloc(kv_per_layer * sh->n_full_attn_layers);
    s->vcache = qw_buf_alloc(kv_per_layer * sh->n_full_attn_layers);

    const size_t ssm_per_layer = (size_t)c->linear_num_value_heads
                               * c->linear_value_head_dim * c->linear_key_head_dim * 4;
    const size_t conv_per_layer = (size_t)(c->linear_conv_kernel_dim - 1) * sh->conv_dim * 4;
    s->ssm_state  = qw_buf_alloc(ssm_per_layer * sh->n_linear_attn_layers);
    s->conv_state = qw_buf_alloc(conv_per_layer * sh->n_linear_attn_layers);

    if (!s->kcache || !s->vcache || !s->ssm_state || !s->conv_state) {
        qw_gerrf(err, errcap, "cannot allocate cache for %d tokens of context", s->max_ctx);
        return false;
    }
    /* A fresh session starts from a zero recurrent state; the KV cache needs no
     * clearing because attention only reads positions that were written. */
    memset(qw_buf_contents(s->ssm_state), 0, ssm_per_layer * sh->n_linear_attn_layers);
    memset(qw_buf_contents(s->conv_state), 0, conv_per_layer * sh->n_linear_attn_layers);

    /* The draft head, if one was loaded.  Its KV cache is a sixteenth of one
     * base attention layer's, and everything else it needs is a handful of
     * single rows. */
    const qw_mtp *m = qwasar_engine_mtp(s->e);
    if (m->present) {
        const size_t mtp_kv = (size_t)c->num_key_value_heads * s->max_ctx
                            * c->head_dim * sizeof(uint16_t);
        s->mtp_kcache = qw_buf_alloc(mtp_kv);
        s->mtp_vcache = qw_buf_alloc(mtp_kv);
        s->mtp_hidden = qw_buf_alloc((size_t)(1 + R) * c->hidden_size * 4);
        s->mtp_tokens = qw_buf_alloc((size_t)R * sizeof(int32_t));
        s->mtp_positions = qw_buf_alloc((size_t)3 * R * sizeof(int32_t));
        s->mtp_embed  = qw_buf_alloc((size_t)R * c->hidden_size * 4);
        s->mtp_fused  = qw_buf_alloc((size_t)R * 2 * c->hidden_size * 4);
        s->mtp_h      = qw_buf_alloc((size_t)R * c->hidden_size * 4);
        s->mtp_out    = qw_buf_alloc((size_t)R * c->hidden_size * 4);
        s->mtp_logits = qw_buf_alloc((size_t)c->vocab_size * 4);
        /* Sized for a full block so the verify can select every row at once;
         * the draft only ever asks for the first. */
        s->sel_out     = qw_buf_alloc((size_t)(QWASAR_MAX_DRAFT + 1) * sizeof(qw_cand));
        s->sel_scratch = qw_buf_alloc((size_t)(QWASAR_MAX_DRAFT + 1)
                                      * QW_SEL_TILES * sizeof(qw_cand));
        if (!s->mtp_kcache || !s->mtp_vcache || !s->mtp_hidden || !s->mtp_tokens
            || !s->mtp_positions || !s->mtp_embed || !s->mtp_fused || !s->mtp_h
            || !s->mtp_out || !s->mtp_logits || !s->sel_out || !s->sel_scratch) {
            qw_gerrf(err, errcap, "cannot allocate MTP head state");
            return false;
        }
        s->mtp_on = true;
    }

    /* rope tables */
    const int32_t nfreq = c->rotary_dim / 2;
    s->rope_axis     = qw_buf_alloc((size_t)nfreq);
    s->rope_inv_freq = qw_buf_alloc((size_t)nfreq * 4);
    if (!s->rope_axis || !s->rope_inv_freq) {
        qw_gerrf(err, errcap, "cannot allocate rope tables");
        return false;
    }
    qw_rope_tables(qw_buf_contents(s->rope_axis), qw_buf_contents(s->rope_inv_freq),
                   c->rotary_dim, c->rope_theta, c->mrope_section);
    return true;
}

qwasar_session *qwasar_session_new(qwasar_engine *e, char *err, size_t errcap) {
    qwasar_session *s = calloc(1, sizeof *s);
    if (!s) { qw_gerrf(err, errcap, "out of memory"); return NULL; }

    s->e     = e;
    s->cfg   = qwasar_engine_config(e);
    s->shape = qwasar_engine_shape(e);
    s->max_ctx  = qwasar_engine_context_size(e);
    s->max_rows = qwasar_engine_prefill_chunk(e);
    s->n_past   = 0;

    s->kind_index = calloc((size_t)s->cfg->num_hidden_layers, sizeof *s->kind_index);
    if (!s->kind_index) { qw_gerrf(err, errcap, "out of memory"); goto fail; }
    for (int32_t i = 0, lin = 0, full = 0; i < s->cfg->num_hidden_layers; i++)
        s->kind_index[i] = qw_layer_is_linear(s->cfg, i) ? lin++ : full++;

    if (!qw_alloc_all(s, err, errcap)) goto fail;
    return s;

fail:
    qwasar_session_free(s);
    return NULL;
}

void qwasar_session_free(qwasar_session *s) {
    if (!s) return;
    qw_buf *all[] = {
        &s->kcache, &s->vcache, &s->ssm_state, &s->conv_state,
        &s->rope_axis, &s->rope_inv_freq, &s->positions,
        &s->tokens, &s->h, &s->hn, &s->hn2,
        &s->qkv, &s->z, &s->a_proj, &s->b_proj, &s->g, &s->beta,
        &s->gq, &s->gk, &s->gv, &s->gdn_y, &s->gdn_norm,
        &s->qg, &s->q, &s->gate, &s->k, &s->v, &s->attn_out,
        &s->mlp_gate, &s->mlp_up, &s->mlp_act, &s->logits,
        &s->mtp_kcache, &s->mtp_vcache, &s->mtp_hidden, &s->mtp_tokens,
        &s->sel_out, &s->sel_scratch,
        &s->mtp_positions, &s->mtp_embed, &s->mtp_fused, &s->mtp_h,
        &s->mtp_out, &s->mtp_logits,
        &s->ssm_snap, &s->conv_snap, &s->verify_logits, &s->img_rows,
    };
    for (size_t i = 0; i < sizeof all / sizeof *all; i++) qw_buf_free(*all[i]);
    qw_buf_free(s->capture);
    free(s->capture_layers);
    free(s->history);
    free(s->mrope);
    free(s->kind_index);
    free(s);
}

int32_t qwasar_session_n_past(const qwasar_session *s) { return s ? s->n_past : 0; }

const float *qwasar_session_logits(const qwasar_session *s) {
    return (s && s->n_past > 0) ? (const float *)qw_buf_contents(s->logits) : NULL;
}

void qwasar_session_set_progress(qwasar_session *s, qwasar_progress_fn fn, void *ud) {
    if (!s) return;
    s->progress = fn;
    s->progress_ud = ud;
}

bool qwasar_session_set_capture(qwasar_session *s, const int32_t *layers, int32_t n) {
    free(s->capture_layers);
    s->capture_layers = NULL;
    qw_buf_free(s->capture);
    s->capture = NULL;
    s->n_capture = 0;
    if (n <= 0) return true;

    s->capture_layers = malloc((size_t)n * sizeof *s->capture_layers);
    s->capture = qw_buf_alloc((size_t)n * s->cfg->hidden_size * 4);
    if (!s->capture_layers || !s->capture) return false;
    memcpy(s->capture_layers, layers, (size_t)n * sizeof *layers);
    s->n_capture = n;
    return true;
}

const float *qwasar_session_captured(const qwasar_session *s, int32_t which) {
    if (!s->capture || which < 0 || which >= s->n_capture) return NULL;
    return (const float *)qw_buf_contents(s->capture)
         + (size_t)which * s->cfg->hidden_size;
}

/* ---- state serialisation ----------------------------------------------------
 *
 * Packed layout, in order:
 *
 *   KV     [n_full][kv_heads][n_tokens][head_dim] fp16, K then V per layer
 *   SSM    [n_linear][hv][dv][dk] fp32
 *   conv   [n_linear][ksize-1][conv_dim] fp32
 *
 * The KV cache is stored head-major with max_ctx as its stride, so packing
 * compacts each head's rows down to the tokens actually used.  That is what
 * makes a checkpoint portable across context sizes -- and it is also why the
 * recurrent halves dominate: SSM and conv are the same size no matter how short
 * the prefix is. */

static size_t qw_kv_row_bytes(const qwasar_session *s) {
    return (size_t)s->cfg->head_dim * sizeof(uint16_t);
}

size_t qw_session_state_bytes(const qwasar_session *s, int32_t n_tokens) {
    const qw_config *c = s->cfg;
    const qw_shape  *sh = s->shape;
    size_t kv = (size_t)sh->n_full_attn_layers * c->num_key_value_heads
              * (size_t)n_tokens * qw_kv_row_bytes(s) * 2;
    size_t ssm = (size_t)sh->n_linear_attn_layers * c->linear_num_value_heads
               * c->linear_value_head_dim * c->linear_key_head_dim * sizeof(float);
    size_t conv = (size_t)sh->n_linear_attn_layers
                * (size_t)(c->linear_conv_kernel_dim - 1) * sh->conv_dim * sizeof(float);
    return kv + ssm + conv;
}

bool qw_session_pack(const qwasar_session *s, void *dst, size_t cap) {
    const qw_config *c = s->cfg;
    const qw_shape  *sh = s->shape;
    const int32_t n = s->n_past;
    if (cap < qw_session_state_bytes(s, n)) return false;

    char *out = dst;
    const size_t row = qw_kv_row_bytes(s);
    const size_t used = (size_t)n * row;
    const size_t head_stride = (size_t)s->max_ctx * row;
    const size_t layer_stride = (size_t)c->num_key_value_heads * head_stride;

    const char *kc = qw_buf_contents(s->kcache);
    const char *vc = qw_buf_contents(s->vcache);
    for (int32_t l = 0; l < sh->n_full_attn_layers; l++)
        for (int32_t h = 0; h < c->num_key_value_heads; h++) {
            const size_t off = (size_t)l * layer_stride + (size_t)h * head_stride;
            memcpy(out, kc + off, used); out += used;
            memcpy(out, vc + off, used); out += used;
        }

    size_t ssm = (size_t)sh->n_linear_attn_layers * c->linear_num_value_heads
               * c->linear_value_head_dim * c->linear_key_head_dim * sizeof(float);
    memcpy(out, qw_buf_contents(s->ssm_state), ssm);
    out += ssm;

    size_t conv = (size_t)sh->n_linear_attn_layers
                * (size_t)(c->linear_conv_kernel_dim - 1) * sh->conv_dim * sizeof(float);
    memcpy(out, qw_buf_contents(s->conv_state), conv);
    return true;
}

bool qw_session_unpack(qwasar_session *s, const void *src, size_t len,
                       const int32_t *tokens, int32_t n_tokens) {
    const qw_config *c = s->cfg;
    const qw_shape  *sh = s->shape;
    if (n_tokens > s->max_ctx) return false;
    if (len != qw_session_state_bytes(s, n_tokens)) return false;

    const char *in = src;
    const size_t row = qw_kv_row_bytes(s);
    const size_t used = (size_t)n_tokens * row;
    const size_t head_stride = (size_t)s->max_ctx * row;
    const size_t layer_stride = (size_t)c->num_key_value_heads * head_stride;

    char *kc = qw_buf_contents(s->kcache);
    char *vc = qw_buf_contents(s->vcache);
    for (int32_t l = 0; l < sh->n_full_attn_layers; l++)
        for (int32_t h = 0; h < c->num_key_value_heads; h++) {
            const size_t off = (size_t)l * layer_stride + (size_t)h * head_stride;
            memcpy(kc + off, in, used); in += used;
            memcpy(vc + off, in, used); in += used;
        }

    size_t ssm = (size_t)sh->n_linear_attn_layers * c->linear_num_value_heads
               * c->linear_value_head_dim * c->linear_key_head_dim * sizeof(float);
    memcpy(qw_buf_contents(s->ssm_state), in, ssm);
    in += ssm;

    size_t conv = (size_t)sh->n_linear_attn_layers
                * (size_t)(c->linear_conv_kernel_dim - 1) * sh->conv_dim * sizeof(float);
    memcpy(qw_buf_contents(s->conv_state), in, conv);

    if (!s->history) s->history = malloc((size_t)s->max_ctx * sizeof *s->history);
    if (!s->history) return false;
    memcpy(s->history, tokens, (size_t)n_tokens * sizeof *s->history);
    s->n_history = n_tokens;
    s->n_past = n_tokens;
    return true;
}

int32_t qwasar_session_common_prefix(const qwasar_session *s,
                                     const int32_t *tokens, int32_t n) {
    if (!s || !s->history || !tokens) return 0;
    int32_t limit = s->n_history < n ? s->n_history : n;
    int32_t i = 0;
    while (i < limit && s->history[i] == tokens[i]) i++;
    /* Only a match covering the whole history is usable: the recurrent state
     * reflects every token evaluated, so a partial match cannot be truncated
     * back to the agreeing prefix. */
    return i == s->n_history ? i : 0;
}

const int32_t *qw_session_history(const qwasar_session *s, int32_t *n) {
    if (n) *n = s ? s->n_history : 0;
    return s ? s->history : NULL;
}

/* ---- the forward pass ------------------------------------------------------ */

static qw_ref qw_off(qw_buf b, size_t elems) { return qw_ref_at(b, elems * 4); }

static void qw_encode_qlinear(qw_cmd c, const qw_qlinear *ql, qw_ref out, qw_ref in,
                              int32_t rows) {
    qw_op_qmat_q4(c, out, in, qw_tensor_ref(ql->weight), qw_tensor_ref(ql->scales),
                  qw_tensor_ref(ql->biases), ql->in_features, ql->out_features, rows);
}

/* The same projection over a contiguous run of output rows, writing them at
 * `out`.  A 4-bit affine row is in_features/2 bytes of nibbles and
 * in_features/32 bytes each of scales and biases, all row-major, so a run of
 * rows is just three offsets and a smaller n -- no copy and no repack. */
static void qw_encode_qlinear_rows(qw_cmd c, const qw_qlinear *ql, qw_ref out, qw_ref in,
                                   int32_t rows, int32_t row0, int32_t n_rows) {
    const size_t wstride = (size_t)ql->in_features / 2;
    const size_t sstride = (size_t)ql->in_features / 32;
    qw_op_qmat_q4(c, out, in,
                  qw_ref_offset(qw_tensor_ref(ql->weight), (size_t)row0 * wstride),
                  qw_ref_offset(qw_tensor_ref(ql->scales), (size_t)row0 * sstride),
                  qw_ref_offset(qw_tensor_ref(ql->biases), (size_t)row0 * sstride),
                  ql->in_features, n_rows, rows);
}

static void qw_encode_gated_delta_layer(qwasar_session *s, qw_cmd c,
                                        const qw_layer *L, int32_t li, int32_t rows) {
    const qw_config *cfg = s->cfg;
    const qw_shape  *sh  = s->shape;
    const int32_t hv = cfg->linear_num_value_heads, hk = cfg->linear_num_key_heads;
    const int32_t dk = cfg->linear_key_head_dim, dv = cfg->linear_value_head_dim;

    const size_t conv_stride = (size_t)(cfg->linear_conv_kernel_dim - 1) * sh->conv_dim;
    const size_t ssm_stride  = (size_t)hv * dv * dk;

    qw_encode_qlinear(c, &L->in_proj_qkv, qw_ref_at(s->qkv, 0),    qw_ref_at(s->hn, 0), rows);
    qw_encode_qlinear(c, &L->in_proj_z,   qw_ref_at(s->z, 0),      qw_ref_at(s->hn, 0), rows);
    qw_encode_qlinear(c, &L->in_proj_a,   qw_ref_at(s->a_proj, 0), qw_ref_at(s->hn, 0), rows);
    qw_encode_qlinear(c, &L->in_proj_b,   qw_ref_at(s->b_proj, 0), qw_ref_at(s->hn, 0), rows);

    /* Causal conv writes back over qkv; the per-channel history lives in
     * conv_state and is advanced in place. */
    qw_op_conv1d_causal_silu(c, qw_ref_at(s->qkv, 0), qw_ref_at(s->qkv, 0),
                             qw_off(s->conv_state, conv_stride * li),
                             qw_tensor_ref(L->conv1d), sh->conv_dim, rows,
                             cfg->linear_conv_kernel_dim,
                             qw_off(s->conv_snap, conv_stride * li),
                             s->n_snap, (int32_t)(conv_stride * sh->n_linear_attn_layers));

    /* q | k | v are concatenated along the channel axis; de-stride them. */
    qw_op_slice_rows(c, qw_ref_at(s->gq, 0), qw_ref_at(s->qkv, 0), rows, sh->conv_dim,
                     0, sh->key_dim);
    qw_op_slice_rows(c, qw_ref_at(s->gk, 0), qw_ref_at(s->qkv, 0), rows, sh->conv_dim,
                     sh->key_dim, sh->key_dim);
    qw_op_slice_rows(c, qw_ref_at(s->gv, 0), qw_ref_at(s->qkv, 0), rows, sh->conv_dim,
                     2 * sh->key_dim, sh->value_dim);

    /* k = l2norm(k);  q = l2norm(q)/sqrt(dk).  Expressed as an unweighted RMS
     * norm with a trailing scale -- see metal/norm.metal. */
    const qw_ref no_weight = qw_ref_at(NULL, 0);
    qw_op_rms_norm(c, qw_ref_at(s->gq, 0), qw_ref_at(s->gq, 0), no_weight,
                   dk, rows * hk, 1e-6f, 1.0f / (float)dk);
    qw_op_rms_norm(c, qw_ref_at(s->gk, 0), qw_ref_at(s->gk, 0), no_weight,
                   dk, rows * hk, 1e-6f, 1.0f / sqrtf((float)dk));

    qw_op_gdn_gates(c, qw_ref_at(s->g, 0), qw_ref_at(s->beta, 0),
                    qw_ref_at(s->a_proj, 0), qw_ref_at(s->b_proj, 0),
                    qw_tensor_ref(L->A_log), qw_tensor_ref(L->dt_bias), hv, rows);

    qw_op_gated_delta(c, qw_ref_at(s->gdn_y, 0), qw_ref_at(s->gq, 0), qw_ref_at(s->gk, 0),
                      qw_ref_at(s->gv, 0), qw_ref_at(s->g, 0), qw_ref_at(s->beta, 0),
                      qw_off(s->ssm_state, ssm_stride * li), hk, hv, dk, dv, rows,
                      qw_off(s->ssm_snap, ssm_stride * li), s->n_snap,
                      (int32_t)(ssm_stride * sh->n_linear_attn_layers));

    /* Output norm is per value head and gated by silu(z). */
    qw_op_rms_norm_gated(c, qw_ref_at(s->gdn_norm, 0), qw_ref_at(s->gdn_y, 0),
                         qw_tensor_ref(L->gdn_norm), qw_ref_at(s->z, 0),
                         dv, rows * hv, cfg->rms_norm_eps, 1.0f);

    qw_encode_qlinear(c, &L->out_proj, qw_ref_at(s->hn2, 0), qw_ref_at(s->gdn_norm, 0), rows);
}

static void qw_encode_attention_layer(qwasar_session *s, qw_cmd c,
                                      const qw_layer *L, int32_t fi, int32_t rows) {
    const qw_config *cfg = s->cfg;
    const qw_shape  *sh  = s->shape;
    const int32_t hd = cfg->head_dim;
    const size_t kv_stride = (size_t)cfg->num_key_value_heads * s->max_ctx * hd;

    qw_encode_qlinear(c, &L->q_proj, qw_ref_at(s->qg, 0), qw_ref_at(s->hn, 0), rows);
    qw_encode_qlinear(c, &L->k_proj, qw_ref_at(s->k, 0),  qw_ref_at(s->hn, 0), rows);
    qw_encode_qlinear(c, &L->v_proj, qw_ref_at(s->v, 0),  qw_ref_at(s->hn, 0), rows);

    /* q_proj emits query and output gate interleaved per head. */
    qw_op_split_heads2(c, qw_ref_at(s->q, 0), qw_ref_at(s->gate, 0), qw_ref_at(s->qg, 0),
                       rows, cfg->num_attention_heads, hd);

    qw_op_rms_norm(c, qw_ref_at(s->q, 0), qw_ref_at(s->q, 0), qw_tensor_ref(L->q_norm),
                   hd, rows * cfg->num_attention_heads, cfg->rms_norm_eps, 1.0f);
    qw_op_rms_norm(c, qw_ref_at(s->k, 0), qw_ref_at(s->k, 0), qw_tensor_ref(L->k_norm),
                   hd, rows * cfg->num_key_value_heads, cfg->rms_norm_eps, 1.0f);

    qw_op_rope_partial(c, qw_ref_at(s->q, 0), qw_ref_at(s->positions, 0),
                       qw_ref_at(s->rope_axis, 0), qw_ref_at(s->rope_inv_freq, 0),
                       rows, cfg->num_attention_heads, hd, cfg->rotary_dim);
    qw_op_rope_partial(c, qw_ref_at(s->k, 0), qw_ref_at(s->positions, 0),
                       qw_ref_at(s->rope_axis, 0), qw_ref_at(s->rope_inv_freq, 0),
                       rows, cfg->num_key_value_heads, hd, cfg->rotary_dim);

    qw_op_kv_write(c, qw_ref_at(s->kcache, kv_stride * fi * sizeof(uint16_t)),
                   qw_ref_at(s->vcache, kv_stride * fi * sizeof(uint16_t)),
                   qw_ref_at(s->k, 0), qw_ref_at(s->v, 0),
                   rows, cfg->num_key_value_heads, hd, s->max_ctx, s->n_past);

    qw_op_attn_decode(c, qw_ref_at(s->attn_out, 0), qw_ref_at(s->q, 0),
                      qw_ref_at(s->kcache, kv_stride * fi * sizeof(uint16_t)),
                      qw_ref_at(s->vcache, kv_stride * fi * sizeof(uint16_t)),
                      rows, cfg->num_attention_heads, cfg->num_key_value_heads,
                      hd, s->max_ctx, s->n_past, 1.0f / sqrtf((float)hd));

    /* The output gate is what makes this "gated attention". */
    qw_op_mul_sigmoid(c, qw_ref_at(s->attn_out, 0), qw_ref_at(s->gate, 0),
                      rows * sh->q_dim);

    qw_encode_qlinear(c, &L->o_proj, qw_ref_at(s->hn2, 0), qw_ref_at(s->attn_out, 0), rows);
}

/* Encodes one chunk of `rows` tokens. */
/* ---- MTP draft head -------------------------------------------------------
 *
 * One full-attention layer with its own KV cache.  Structurally it is
 * qw_encode_attention_layer above -- the same gated output, the same q and k
 * norms, the same partial RoPE -- differing only in its own cache.  It arrives
 * bf16 and is quantised at load to the format the rest of the model uses, so it
 * runs on the same kernels (see qw_quantise_mtp).
 *
 * The head only PROPOSES.  Nothing it computes reaches an emitted token except
 * through the target's verification, so none of this sits on the exactness
 * surface: a bug here costs acceptance rate, not correctness.  Which is also
 * why it has to be measured.  There is no wrong answer for it to produce, only
 * a worse one, and nothing will report it. */

/* Fuses `rows` (hidden, next-token) pairs, runs them through the head's layer,
 * and appends their keys and values to the head cache at `mtp_n_past`.
 *
 * `hidden` is [rows, hidden_size] of post-norm backbone hidden states;
 * `mtp_embed` must already hold the embeddings of the paired next tokens and
 * `mtp_positions` their backbone positions.  The result lands in `mtp_out`,
 * which is what the base lm_head reads. */
static void qw_encode_mtp_rows(qwasar_session *s, qw_cmd c, qw_ref hidden,
                               int32_t rows, qw_ref pos) {
    const qw_config *cfg = s->cfg;
    const qw_shape  *sh  = s->shape;
    const qw_mtp    *m   = qwasar_engine_mtp(s->e);
    const int32_t hd = cfg->head_dim;

    /* [ norm_e(embed(next)) | norm_h(hidden) ] -- embedding half first.  That
     * is the opposite of the DeepSeek layout this head otherwise resembles;
     * reversed, it loads, runs at full speed, and drafts nonsense. */
    qw_op_rms_norm_concat(c, qw_ref_at(s->mtp_fused, 0),
                          qw_ref_at(s->mtp_embed, 0),
                          qw_tensor_ref(m->pre_fc_norm_embedding),
                          hidden, qw_tensor_ref(m->pre_fc_norm_hidden),
                          cfg->hidden_size, rows, cfg->rms_norm_eps);
    qw_encode_qlinear(c, &m->q_fc, qw_ref_at(s->mtp_h, 0), qw_ref_at(s->mtp_fused, 0), rows);

    qw_op_rms_norm(c, qw_ref_at(s->hn, 0), qw_ref_at(s->mtp_h, 0),
                   qw_tensor_ref(m->input_layernorm),
                   cfg->hidden_size, rows, cfg->rms_norm_eps, 1.0f);

    qw_encode_qlinear(c, &m->q_q, qw_ref_at(s->qg, 0), qw_ref_at(s->hn, 0), rows);
    qw_encode_qlinear(c, &m->q_k, qw_ref_at(s->k, 0),  qw_ref_at(s->hn, 0), rows);
    qw_encode_qlinear(c, &m->q_v, qw_ref_at(s->v, 0),  qw_ref_at(s->hn, 0), rows);

    qw_op_split_heads2(c, qw_ref_at(s->q, 0), qw_ref_at(s->gate, 0), qw_ref_at(s->qg, 0),
                       rows, cfg->num_attention_heads, hd);
    qw_op_rms_norm(c, qw_ref_at(s->q, 0), qw_ref_at(s->q, 0), qw_tensor_ref(m->q_norm),
                   hd, rows * cfg->num_attention_heads, cfg->rms_norm_eps, 1.0f);
    qw_op_rms_norm(c, qw_ref_at(s->k, 0), qw_ref_at(s->k, 0), qw_tensor_ref(m->k_norm),
                   hd, rows * cfg->num_key_value_heads, cfg->rms_norm_eps, 1.0f);

    qw_op_rope_partial(c, qw_ref_at(s->q, 0), pos,
                       qw_ref_at(s->rope_axis, 0), qw_ref_at(s->rope_inv_freq, 0),
                       rows, cfg->num_attention_heads, hd, cfg->rotary_dim);
    qw_op_rope_partial(c, qw_ref_at(s->k, 0), pos,
                       qw_ref_at(s->rope_axis, 0), qw_ref_at(s->rope_inv_freq, 0),
                       rows, cfg->num_key_value_heads, hd, cfg->rotary_dim);

    qw_op_kv_write(c, qw_ref_at(s->mtp_kcache, 0), qw_ref_at(s->mtp_vcache, 0),
                   qw_ref_at(s->k, 0), qw_ref_at(s->v, 0),
                   rows, cfg->num_key_value_heads, hd, s->max_ctx, s->mtp_n_past);
    qw_op_attn_decode(c, qw_ref_at(s->attn_out, 0), qw_ref_at(s->q, 0),
                      qw_ref_at(s->mtp_kcache, 0), qw_ref_at(s->mtp_vcache, 0),
                      rows, cfg->num_attention_heads, cfg->num_key_value_heads,
                      hd, s->max_ctx, s->mtp_n_past, 1.0f / sqrtf((float)hd));
    qw_op_mul_sigmoid(c, qw_ref_at(s->attn_out, 0), qw_ref_at(s->gate, 0),
                      rows * sh->q_dim);
    qw_encode_qlinear(c, &m->q_o, qw_ref_at(s->hn2, 0), qw_ref_at(s->attn_out, 0), rows);
    qw_op_add_inplace(c, qw_ref_at(s->mtp_h, 0), qw_ref_at(s->hn2, 0),
                      rows * cfg->hidden_size);

    qw_op_rms_norm(c, qw_ref_at(s->hn, 0), qw_ref_at(s->mtp_h, 0),
                   qw_tensor_ref(m->post_attention_layernorm),
                   cfg->hidden_size, rows, cfg->rms_norm_eps, 1.0f);
    qw_encode_qlinear(c, &m->q_gate, qw_ref_at(s->mlp_gate, 0), qw_ref_at(s->hn, 0), rows);
    qw_encode_qlinear(c, &m->q_up,   qw_ref_at(s->mlp_up, 0),   qw_ref_at(s->hn, 0), rows);
    qw_op_swiglu(c, qw_ref_at(s->mlp_act, 0), qw_ref_at(s->mlp_gate, 0),
                 qw_ref_at(s->mlp_up, 0), rows * cfg->intermediate_size);
    qw_encode_qlinear(c, &m->q_down, qw_ref_at(s->hn2, 0), qw_ref_at(s->mlp_act, 0), rows);
    qw_op_add_inplace(c, qw_ref_at(s->mtp_h, 0), qw_ref_at(s->hn2, 0),
                      rows * cfg->hidden_size);

    qw_op_rms_norm(c, qw_ref_at(s->mtp_out, 0), qw_ref_at(s->mtp_h, 0),
                   qw_tensor_ref(m->norm), cfg->hidden_size, rows,
                   cfg->rms_norm_eps, 1.0f);
}

/* Fills the head's position buffer.  Text-only positions are identical on all
 * three MRoPE axes, and the stride is the row count of this call. */
/* Writes one step's positions at `slot` (an int32 offset into mtp_positions).
 *
 * The axis stride is `rows`, so a step's region is 3*rows wide and steps of
 * different widths cannot share one.  Chaining a draft block needs a region
 * per step anyway: the host fills these while earlier steps are still queued,
 * and overwriting a region the GPU has not read yet would corrupt it. */
static void qw_mtp_positions(qwasar_session *s, int32_t slot,
                             int32_t first, int32_t rows) {
    int32_t *pv = (int32_t *)qw_buf_contents(s->mtp_positions) + slot;
    for (int axis = 0; axis < 3; axis++)
        for (int32_t r = 0; r < rows; r++) pv[axis * rows + r] = first + r;
}

/* Extends the head's committed history by every row whose next token is
 * already known -- which, after a forward over `rows` tokens, is all of them
 * but the last.
 *
 * This is the part that decides whether drafting is worth anything at all.  A
 * head asked to predict from a single position instead of the whole committed
 * prefix accepts about a quarter of the time rather than nine tenths, so the
 * history is not an optimisation and cannot be skipped; it is what the head
 * was trained to read.
 *
 * The bookkeeping follows from one invariant: head row p pairs the backbone's
 * hidden state at p with the embedding of token p+1.  A position's own next
 * token is therefore the thing that is missing at the end of every forward, so
 * the last row waits in `mtp_hidden` row 0 for the following call -- an eval
 * that supplies it as its first input token, or a draft that supplies the
 * token just emitted. */
static void qw_encode_mtp_upkeep(qwasar_session *s, qw_cmd c, int32_t rows) {
    const qw_config *cfg = s->cfg;
    const qw_qlinear *embed = qwasar_engine_embed(s->e);

    /* With a row pending, the tokens are this chunk's inputs and the hidden
     * states run from the pending row through all but this chunk's last.
     * Without one -- the first forward of a session -- there is no position
     * before the first, so the first token pairs with nothing. */
    const int32_t n_up = s->mtp_pending ? rows : rows - 1;
    if (n_up > 0) {
        const int32_t *tokens = qw_buf_contents(s->tokens);
        int32_t *tv = qw_buf_contents(s->mtp_tokens);
        memcpy(tv, tokens + (s->mtp_pending ? 0 : 1), (size_t)n_up * sizeof *tv);
        qw_mtp_positions(s, 0, s->mtp_n_past, n_up);

        qw_op_embed_q4(c, qw_ref_at(s->mtp_embed, 0), qw_ref_at(s->mtp_tokens, 0),
                       qw_tensor_ref(embed->weight), qw_tensor_ref(embed->scales),
                       qw_tensor_ref(embed->biases), cfg->hidden_size, n_up);
        qw_encode_mtp_rows(s, c,
                           qw_off(s->mtp_hidden,
                                  s->mtp_pending ? 0 : (size_t)cfg->hidden_size),
                           n_up, qw_ref_at(s->mtp_positions, 0));
        s->mtp_n_past += n_up;
    }

    /* Park this chunk's last hidden state in the pending slot. */
    qw_op_slice_rows(c, qw_ref_at(s->mtp_hidden, 0),
                     qw_off(s->mtp_hidden, (size_t)rows * cfg->hidden_size),
                     1, cfg->hidden_size, 0, cfg->hidden_size);
    s->mtp_pending = true;
}

static void qw_encode_forward(qwasar_session *s, qw_cmd c, int32_t rows, bool want_logits) {
    const qw_config *cfg = s->cfg;
    qwasar_engine *e = s->e;

    qw_op_embed_q4(c, qw_ref_at(s->h, 0), qw_ref_at(s->tokens, 0),
                   qw_tensor_ref(qwasar_engine_embed(e)->weight),
                   qw_tensor_ref(qwasar_engine_embed(e)->scales),
                   qw_tensor_ref(qwasar_engine_embed(e)->biases),
                   cfg->hidden_size, rows);

    /* Image rows replace what the embedding table produced for the pad tokens.
     * They arrive in runs, so this copies spans rather than scattering rows --
     * a chunk boundary can split a run, which is why the source cursor is
     * carried on the session rather than recomputed here. */
    if (s->n_img_rows > 0) {
        const int32_t *tv = qw_buf_contents(s->tokens);
        int32_t r = 0;
        while (r < rows) {
            const bool pad = tv[r] == cfg->image_token_id
                          || tv[r] == cfg->video_token_id;
            if (!pad) { r++; continue; }
            int32_t run = 0;
            while (r + run < rows && (tv[r + run] == cfg->image_token_id
                                   || tv[r + run] == cfg->video_token_id)) run++;
            if (s->img_cursor + run > s->n_img_rows) run = s->n_img_rows - s->img_cursor;
            if (run > 0)
                qw_op_slice_rows(c, qw_off(s->h, (size_t)r * cfg->hidden_size),
                                 qw_off(s->img_rows,
                                        (size_t)s->img_cursor * cfg->hidden_size),
                                 run, cfg->hidden_size, 0, cfg->hidden_size);
            s->img_cursor += run;
            r += run > 0 ? run : 1;
        }
    }

    for (int32_t i = 0; i < cfg->num_hidden_layers; i++) {
        const qw_layer *L = qwasar_engine_layer(e, i);

        qw_op_rms_norm(c, qw_ref_at(s->hn, 0), qw_ref_at(s->h, 0),
                       qw_tensor_ref(L->input_layernorm),
                       cfg->hidden_size, rows, cfg->rms_norm_eps, 1.0f);

        if (L->is_linear_attn) qw_encode_gated_delta_layer(s, c, L, s->kind_index[i], rows);
        else                   qw_encode_attention_layer  (s, c, L, s->kind_index[i], rows);

        qw_op_add_inplace(c, qw_ref_at(s->h, 0), qw_ref_at(s->hn2, 0),
                          rows * cfg->hidden_size);

        qw_op_rms_norm(c, qw_ref_at(s->hn, 0), qw_ref_at(s->h, 0),
                       qw_tensor_ref(L->post_attention_layernorm),
                       cfg->hidden_size, rows, cfg->rms_norm_eps, 1.0f);
        qw_encode_qlinear(c, &L->gate_proj, qw_ref_at(s->mlp_gate, 0), qw_ref_at(s->hn, 0), rows);
        qw_encode_qlinear(c, &L->up_proj,   qw_ref_at(s->mlp_up, 0),   qw_ref_at(s->hn, 0), rows);
        qw_op_swiglu(c, qw_ref_at(s->mlp_act, 0), qw_ref_at(s->mlp_gate, 0),
                     qw_ref_at(s->mlp_up, 0), rows * cfg->intermediate_size);
        qw_encode_qlinear(c, &L->down_proj, qw_ref_at(s->hn2, 0), qw_ref_at(s->mlp_act, 0), rows);

        qw_op_add_inplace(c, qw_ref_at(s->h, 0), qw_ref_at(s->hn2, 0),
                          rows * cfg->hidden_size);

        for (int32_t j = 0; j < s->n_capture; j++)
            if (s->capture_layers[j] == i)
                qw_op_slice_rows(c, qw_off(s->capture, (size_t)j * cfg->hidden_size),
                                 qw_off(s->h, (size_t)(rows - 1) * cfg->hidden_size),
                                 1, cfg->hidden_size, 0, cfg->hidden_size);
    }

    /* The draft head reads this same normalised hidden state, so the norm runs
     * on every chunk when a head is attached rather than only on the one that
     * produces logits. */
    if (!want_logits && !s->mtp_on) return;

    qw_op_rms_norm(c, qw_ref_at(s->hn, 0), qw_ref_at(s->h, 0),
                   qw_tensor_ref(qwasar_engine_final_norm(e)),
                   cfg->hidden_size, rows, cfg->rms_norm_eps, 1.0f);

    /* Rows 1.. of mtp_hidden hold this chunk's post-norm hidden states; row 0
     * is the pending slot the upkeep below reads and then refills. */
    if (s->mtp_on)
        qw_op_slice_rows(c, qw_off(s->mtp_hidden, (size_t)cfg->hidden_size),
                         qw_ref_at(s->hn, 0), rows, cfg->hidden_size,
                         0, cfg->hidden_size);

    /* Ordinarily only the final token's logits are used, and the head is the
     * single widest matvec in the model, so it runs on one row rather than
     * `rows`.  A verify wants all of them: that is what makes one pass over the
     * weights settle several tokens instead of one.  Either way it goes before
     * the draft head's own pass, which reuses `hn` as scratch. */
    if (s->mtp_defer)
        qw_encode_qlinear(c, qwasar_engine_head(e), qw_ref_at(s->verify_logits, 0),
                          qw_ref_at(s->hn, 0), rows);
    else if (want_logits)
        qw_encode_qlinear(c, qwasar_engine_head(e), qw_ref_at(s->logits, 0),
                          qw_off(s->hn, (size_t)(rows - 1) * cfg->hidden_size), 1);

    if (s->mtp_on && !s->mtp_defer) qw_encode_mtp_upkeep(s, c, rows);
}

/* ---- compact draft head -----------------------------------------------------
 *
 * Why the draft scores only part of the vocabulary, and why that is safe, is in
 * qwasar_model.h above QW_DRAFT_PREFIX.  What is here is only the mechanics. */

/* How many rows a draft scored, and how to read a winning row as a token id.
 * Zeroed prefix/tail_base is the identity, which is what a full score wants. */
typedef struct { int32_t scored, prefix, tail_base; } qw_draft_head;

/* Scores the kept rows into `out`, compacted: [0, prefix) then the tail. */
static qw_draft_head qw_encode_draft_head(qwasar_session *s, qw_cmd c,
                                          qw_ref out, qw_ref in) {
    const qw_qlinear *head  = qwasar_engine_head(s->e);
    const int32_t     vocab = s->cfg->vocab_size;

    /* A model whose vocabulary does not reach the cut scores all of it. */
    if (vocab <= QW_DRAFT_PREFIX || QW_DRAFT_TAIL_LO >= vocab) {
        qw_encode_qlinear(c, head, out, in, 1);
        return (qw_draft_head){ vocab, 0, 0 };
    }
    const int32_t tail = vocab - QW_DRAFT_TAIL_LO;
    qw_encode_qlinear_rows(c, head, out, in, 1, 0, QW_DRAFT_PREFIX);
    qw_encode_qlinear_rows(c, head, qw_ref_offset(out, (size_t)QW_DRAFT_PREFIX * 4), in,
                           1, QW_DRAFT_TAIL_LO, tail);
    return (qw_draft_head){ QW_DRAFT_PREFIX + tail, QW_DRAFT_PREFIX, QW_DRAFT_TAIL_LO };
}

/* ---- adaptive draft depth --------------------------------------------------
 *
 * Cost of one round at each depth, in decode steps, measured on this machine
 * over 200 tokens of prose, each depth bracketed by its own serial control so
 * that drift cancels per point:
 *
 *   depth 1  1.32      depth 2  1.40      depth 3  1.56      depth 4  3.08
 *
 * Index 0 is not drafting at all, which is a decode step exactly.  The step
 * from 0 to 1 is much the largest -- turning drafting on switches every
 * projection from the single-token kernel to the batched one and starts saving
 * rewind state -- so a round that will not accept its first draft is better off
 * not asking for one.  That discontinuity is the whole reason this rule beats a
 * constant.
 *
 * Re-measured after the compact draft head and after the CLI stopped leaving
 * drafting out of its decode timer.  The previous numbers (1.39 / 1.58 / 1.91)
 * were wrong in BOTH directions at once -- the timer understated every depth,
 * the full-vocabulary head overstated them -- and the errors did not cancel:
 * the marginal drafts at 2 and 3 are much cheaper than the table claimed, so
 * the rule had been declining depth it should have taken.
 *
 * These are properties of the kernels, not of the model, and they have to be
 * re-measured whenever the kernels move.  The comment above QW_QMVB_B says the
 * same thing about the width curve they come from. */
static const double QW_DEPTH_COST[QWASAR_MAX_DRAFT + 1] = {
    1.00, 1.32, 1.40, 1.56, 3.08, 3.35, 3.60, 3.85, 5.35
};

/* Depth 4 is where a verify needs a second batched-matvec block, and it is now
 * measured rather than guessed: 1.56 -> 3.08, a doubling for one more token.
 * That is a cliff, not a slope, so the search still stops at 3 -- reaching past
 * it would need an acceptance rate this model does not produce.  Depths 5-8 stay
 * extrapolated and deliberately pessimistic, with a second cliff at 8 where a
 * third block starts; nothing consults them while the cap is 3. */
#define QW_DEPTH_MEASURED 3

static void qw_mtp_seed(qwasar_session *s) {
    if (s->mtp_p_seeded) return;
    /* Optimistic and decaying, so the first rounds draft rather than spend ten
     * rounds learning that they could have.  Capped below 1 because uncapped
     * optimism keeps over-drafting on a prompt that never earns it. */
    for (int i = 0; i < QWASAR_MAX_DRAFT; i++) {
        double v = 0.85 * pow(0.98, (double)i);
        s->mtp_p[i] = v < 0.95 ? v : 0.95;
    }
    s->mtp_margin = -1.0;
    s->mtp_p_seeded = true;
}

int32_t qwasar_session_draft_depth(qwasar_session *s) {
    if (!s || !s->mtp_on) return 0;
    qw_mtp_seed(s);

    /* The top-2 gap caps the first position only: it is evidence about the very
     * next token, and says nothing about the one after it. */
    double conf = 1.0;
    if (s->mtp_margin >= 0.0) conf = 1.0 / (1.0 + exp(-s->mtp_margin / 2.0));

    int32_t best_d = 0;
    double best_rate = 1.0 / QW_DEPTH_COST[0];
    double expected = 1.0, reach = 1.0;

    for (int32_t d = 1; d <= QW_DEPTH_MEASURED; d++) {
        double p = s->mtp_p[d - 1];
        if (d == 1 && conf < p) p = conf;
        reach *= p;
        expected += reach;
        const double rate = expected / QW_DEPTH_COST[d];
        if (rate > best_rate) { best_rate = rate; best_d = d; }
    }
    return best_d;
}

bool qwasar_session_has_mtp(const qwasar_session *s) { return s && s->mtp_on; }

int32_t qwasar_session_draft(qwasar_session *s, int32_t emitted,
                             int32_t *drafts, int32_t n_draft,
                             char *err, size_t errcap) {
    if (!s || !s->mtp_on) { qw_gerrf(err, errcap, "no MTP draft head"); return -1; }
    if (!s->mtp_pending && !s->mtp_after_verify) {
        qw_gerrf(err, errcap, "nothing to draft from");
        return -1;
    }
    if (n_draft < 1) return 0;
    if (n_draft > QWASAR_MAX_DRAFT) n_draft = QWASAR_MAX_DRAFT;

    const qw_config  *cfg   = s->cfg;
    const qw_qlinear *embed = qwasar_engine_embed(s->e);

    /* Rows a verify left owed go in front of the first draft: same weights, one
     * pass instead of two.  They are contiguous with it by construction -- the
     * verify's own hidden states, and tokens the verify confirmed -- so the
     * whole thing is one run of rows starting at mtp_hidden row 1.  Without a
     * verify in front there is nothing owed and the draft reads the pending
     * slot, row 0, as it always did. */
    const bool    after_verify = s->mtp_after_verify;
    const int32_t lead  = after_verify ? s->mtp_owed : 0;
    const int32_t first = after_verify ? 1 : 0;
    s->mtp_after_verify = false;
    s->mtp_owed = 0;

    /* The first drafted row completes the pending position and is therefore
     * committed too: both of its inputs -- the backbone's hidden state and the
     * token just emitted -- are real.  Every row after it is built on a token
     * the head only guessed, so those are speculative and are dropped at the
     * end by rewinding the write cursor. */
    const int32_t committed = s->mtp_n_past + lead + 1;
    qw_ref  hidden = qw_off(s->mtp_hidden, (size_t)first * cfg->hidden_size);
    int32_t n = 0;

    /* The whole block goes into ONE command buffer.  Step n+1 needs step n's
     * token, which used to mean a sync per draft: the host read the id back
     * and wrote it into the token buffer for the next embedding gather.  Now
     * the selection kernel writes it there itself, so the dependency stays on
     * the device and the chain is just a sequence of dispatches.
     *
     * That is safe because the encoder is serial -- Metal orders each dispatch
     * after the last, so every shared scratch buffer behaves exactly as it did
     * when the steps were separate command buffers.  What is NOT automatic is
     * the buffers the host fills: it writes them all up front, while later
     * steps are still queued, so each step needs its own region rather than
     * reusing offset zero.  Token slots and position slots below are that. */
    const int32_t rows0 = lead + 1;

    /* Step 0's tokens are all known here: the rows a verify left owed, then
     * the token just emitted.  Every later step's token is written by the GPU
     * into the slot that step's gather reads. */
    int32_t *tv = qw_buf_contents(s->mtp_tokens);
    for (int32_t i = 0; i < lead; i++) tv[i] = s->mtp_owed_tokens[i];
    tv[rows0 - 1] = emitted;

    qw_cmd c = qw_cmd_begin();
    if (!c) { qw_gerrf(err, errcap, "cannot begin a command buffer"); return -1; }

    for (; n < n_draft; n++) {
        const int32_t rows = (n == 0) ? rows0 : 1;
        /* Step 0 owns [0, rows0); each later step owns one slot after it. */
        const int32_t tok_slot = (n == 0) ? 0 : rows0 + n - 1;
        /* Position regions are 3*rows wide, so they cannot overlap either. */
        const int32_t pos_slot = (n == 0) ? 0 : 3 * rows0 + 3 * (n - 1);

        qw_mtp_positions(s, pos_slot, s->mtp_n_past, rows);

        qw_op_embed_q4(c, qw_ref_at(s->mtp_embed, 0), qw_off(s->mtp_tokens, tok_slot),
                       qw_tensor_ref(embed->weight), qw_tensor_ref(embed->scales),
                       qw_tensor_ref(embed->biases), cfg->hidden_size, rows);
        qw_encode_mtp_rows(s, c, hidden, rows, qw_off(s->mtp_positions, pos_slot));
        /* The pending slot has to end up holding the last CONFIRMED hidden
         * state, which is the one the drafted row was built from.  It is
         * refilled after the rows above have read it, not before -- row 0 is
         * both the first owed row's input and the slot being overwritten. */
        if (n == 0 && after_verify)
            qw_op_slice_rows(c, qw_ref_at(s->mtp_hidden, 0),
                             qw_off(s->mtp_hidden,
                                    (size_t)(first + rows - 1) * cfg->hidden_size),
                             1, cfg->hidden_size, 0, cfg->hidden_size);
        /* Only the last row is a proposal; the ones in front of it are history
         * being caught up, and their outputs go nowhere. */
        const qw_draft_head dh =
            qw_encode_draft_head(s, c, qw_ref_at(s->mtp_logits, 0),
                                 qw_off(s->mtp_out, (size_t)(rows - 1) * cfg->hidden_size));
        /* Result to this step's own slot, so the host can read the whole block
         * after one wait; token to the slot the NEXT step gathers from.  The
         * mapping turns a compact row back into a token id on the device, so
         * the chained gather reads a real id. */
        const size_t cand = sizeof(qw_cand) / 4;   /* qw_off counts 4-byte units */
        qw_op_argmax_top2(c, qw_off(s->sel_out, (size_t)n * cand),
                          qw_off(s->sel_scratch, (size_t)n * QW_SEL_TILES * cand),
                          qw_ref_at(s->mtp_logits, 0), dh.scored, 1,
                          qw_off(s->mtp_tokens, rows0 + n),
                          dh.prefix, dh.tail_base);

        s->mtp_n_past += rows;
        /* Row 0 was just refilled when there were owed rows, so it still holds
         * a hidden state waiting for its token.  Without them nothing refills
         * it, and the next eval's upkeep is what will. */
        if (n == 0) s->mtp_pending = after_verify;

        hidden = qw_off(s->mtp_out, (size_t)(rows - 1) * cfg->hidden_size);
    }

    qw_cmd_wait(c);
    const char *cerr = qw_cmd_error(c);
    if (cerr) {
        qw_gerrf(err, errcap, "GPU error: %s", cerr);
        qw_cmd_free(c);
        return -1;
    }
    qw_cmd_free(c);

    const qw_cand *sel = qw_buf_contents(s->sel_out);
    for (int32_t i = 0; i < n_draft; i++) drafts[i] = (int32_t)sel[i].index;

    s->mtp_n_past = committed;
    return n;
}

/* Sizes the rewind buffers for a verify of `n_draft` drafts, once.
 *
 * Deliberately lazy: the recurrent state is 150 MB and a boundary needs a whole
 * copy of it, so a session that never drafts should not pay for the ability. */
static bool qw_snap_reserve(qwasar_session *s, int32_t n_draft,
                            char *err, size_t errcap) {
    if (s->snap_capacity >= n_draft) return true;
    const qw_config *c = s->cfg;
    const qw_shape  *sh = s->shape;

    qw_buf_free(s->ssm_snap);
    qw_buf_free(s->conv_snap);
    const size_t ssm_row  = (size_t)c->linear_num_value_heads * c->linear_value_head_dim
                          * c->linear_key_head_dim * 4 * sh->n_linear_attn_layers;
    const size_t conv_row = (size_t)(c->linear_conv_kernel_dim - 1) * sh->conv_dim * 4
                          * sh->n_linear_attn_layers;
    s->ssm_snap  = qw_buf_alloc(ssm_row  * (size_t)n_draft);
    s->conv_snap = qw_buf_alloc(conv_row * (size_t)n_draft);
    if (!s->ssm_snap || !s->conv_snap) {
        qw_gerrf(err, errcap, "cannot allocate %.0f MB of rewind state for depth %d",
                 (double)((ssm_row + conv_row) * (size_t)n_draft) / 1e6, n_draft);
        s->snap_capacity = 0;
        return false;
    }
    s->snap_capacity = n_draft;
    return true;
}

/* The shared verify pass.  `sp` NULL (or temperature 0) accepts by argmax
 * equality; otherwise by rejection sampling against the target's filtered
 * distribution, which needs the per-row logits the pass already keeps. */
static int32_t qw_verify(qwasar_session *s, const int32_t *block, int32_t n_block,
                         const qwasar_sampling *sp, uint64_t *rng,
                         int32_t *out, char *err, size_t errcap) {
    if (!s || !block || n_block < 2 || !out) {
        qw_gerrf(err, errcap, "nothing to verify");
        return -1;
    }
    const int32_t n_draft = n_block - 1;
    if (n_draft > QWASAR_MAX_DRAFT) { qw_gerrf(err, errcap, "draft too deep"); return -1; }
    if (!qw_snap_reserve(s, n_draft, err, errcap)) return -1;
    if (n_block > s->max_rows) { qw_gerrf(err, errcap, "draft exceeds the chunk size"); return -1; }
    if (s->n_past + n_block > s->max_ctx) {
        qw_gerrf(err, errcap, "context exhausted: %d + %d exceeds %d tokens",
                 s->n_past, n_block, s->max_ctx);
        return -1;
    }
    if (!s->verify_logits) {
        s->verify_logits = qw_buf_alloc((size_t)(QWASAR_MAX_DRAFT + 1)
                                        * (size_t)s->cfg->vocab_size * 4);
        if (!s->verify_logits) { qw_gerrf(err, errcap, "out of memory"); return -1; }
    }

    const int32_t base = s->n_past;

    memcpy(qw_buf_contents(s->tokens), block, (size_t)n_block * sizeof *block);
    int32_t *pos = qw_buf_contents(s->positions);
    for (int32_t axis = 0; axis < 3; axis++)
        for (int32_t r = 0; r < n_block; r++) pos[axis * n_block + r] = base + r;

    /* Every boundary but the last: if the whole block is accepted the live
     * state is already the right one. */
    s->n_snap = n_draft;
    s->mtp_defer = true;

    qw_cmd c = qw_cmd_begin();
    if (!c) { qw_gerrf(err, errcap, "cannot begin a command buffer"); return -1; }
    qw_encode_forward(s, c, n_block, true);
    /* Every row, not just the one that turns out to decide: which row that is
     * depends on results this pass has not produced yet, and selecting all of
     * them costs one dispatch over logits the GPU already holds. */
    qw_op_argmax_top2(c, qw_ref_at(s->sel_out, 0), qw_ref_at(s->sel_scratch, 0),
                      qw_ref_at(s->verify_logits, 0), s->cfg->vocab_size, n_block,
                      qw_ref_at(NULL, 0), 0, 0);
    qw_cmd_wait(c);
    const char *cerr = qw_cmd_error(c);
    s->n_snap = 0;
    s->mtp_defer = false;
    if (cerr) {
        qw_gerrf(err, errcap, "GPU error: %s", cerr);
        qw_cmd_free(c);
        return -1;
    }
    qw_cmd_free(c);

    /* Row i's logits predict the token after block[i], so they are what
     * block[i+1] guessed. */
    const qw_cand *sel = qw_buf_contents(s->sel_out);
    const bool sampled = sp && sp->temperature > 0.0f && rng;
    const float  *lg    = qw_buf_contents(s->verify_logits);
    const int32_t vocab = s->cfg->vocab_size;
    int32_t j = 0;
    if (!sampled) {
        /* Accept while the guesses match the argmax. */
        while (j < n_draft && (int32_t)sel[j].index == block[j + 1]) j++;
    } else {
        /* Accept draft block[j+1] with the probability the filtered target
         * distribution gives it.  The draft is a point mass (the head proposed
         * exactly one token), so this acceptance rule plus the residual
         * resample below emits the filtered distribution exactly. */
        while (j < n_draft) {
            const float p = qwasar_sample_prob(lg + (size_t)j * vocab, vocab,
                                               sp, block[j + 1]);
            if (qwasar_rng_uniform(rng) >= p) break;
            j++;
        }
    }

    /* The accepted drafts, plus the token row j decides.  That last one is
     * free: the pass computed row j's logits whether or not anything was
     * accepted, which is why even a fully rejected round still advances by
     * one.  Greedy takes the argmax; sampled takes the residual max(0, p - q)
     * renormalised on a rejection -- with a point-mass draft that is p with
     * the rejected token removed -- and a plain sample when every draft held. */
    for (int32_t i = 0; i < j; i++) out[i] = block[i + 1];
    if (!sampled) {
        out[j] = (int32_t)sel[j].index;
    } else if (j < n_draft) {
        out[j] = qwasar_sample_excluding(lg + (size_t)j * vocab, vocab,
                                         sp, rng, block[j + 1]);
    } else {
        out[j] = qwasar_sample(lg + (size_t)j * vocab, vocab, sp, rng);
    }
    s->mtp_margin = (double)sel[j].best - (double)sel[j].second;

    /* Update the acceptance estimates.  Position i was only put to the test if
     * everything before it was accepted, so only those are observations -- the
     * rest of the block was never reached and says nothing. */
    qw_mtp_seed(s);
    for (int32_t i = 0; i <= j && i < n_draft; i++) {
        const double hit = (i < j) ? 1.0 : 0.0;
        s->mtp_p[i] += 0.15 * (hit - s->mtp_p[i]);
    }

    if (j < n_draft) {
        /* Undo the rejected rows.  Attention only reads positions below n_past,
         * so its cache needs no work; the recurrent state was advanced in place
         * and has to be put back from the boundary the forward saved. */
        const qw_config *cfg = s->cfg;
        const qw_shape  *sh  = s->shape;
        const size_t ssm_row  = (size_t)cfg->linear_num_value_heads
                              * cfg->linear_value_head_dim * cfg->linear_key_head_dim
                              * 4 * sh->n_linear_attn_layers;
        const size_t conv_row = (size_t)(cfg->linear_conv_kernel_dim - 1)
                              * sh->conv_dim * 4 * sh->n_linear_attn_layers;
        memcpy(qw_buf_contents(s->ssm_state),
               (const char *)qw_buf_contents(s->ssm_snap) + ssm_row * (size_t)j, ssm_row);
        memcpy(qw_buf_contents(s->conv_state),
               (const char *)qw_buf_contents(s->conv_snap) + conv_row * (size_t)j, conv_row);
    }

    s->n_past = base + j + 1;
    if (s->history && s->n_history > s->n_past) s->n_history = s->n_past;

    /* The head's history can only take the confirmed rows.  Running that here
     * would be a second pass over the head's weights in the same round, so it
     * is handed to the next draft instead, which needs a pass anyway and can
     * put these rows in front of its own. */
    if (s->mtp_on) {
        /* Head row p pairs hidden_p with token_{p+1}.  The row for the position
         * before this block was already committed by the draft that produced
         * it, so what is owed is the accepted drafts' own rows -- positions
         * base..base+j-1, taking tokens block[1..j].  Their hidden states are
         * this pass's, which start at mtp_hidden row 1. */
        s->mtp_after_verify = true;
        s->mtp_owed = j;
        for (int32_t i = 0; i < j; i++) s->mtp_owed_tokens[i] = block[i + 1];
    }
    return j + 1;
}

int32_t qwasar_session_verify(qwasar_session *s, const int32_t *block, int32_t n_block,
                              int32_t *out, char *err, size_t errcap) {
    return qw_verify(s, block, n_block, NULL, NULL, out, err, errcap);
}

int32_t qwasar_session_verify_sampled(qwasar_session *s, const int32_t *block,
                                      int32_t n_block, const qwasar_sampling *sp,
                                      uint64_t *rng, int32_t *out,
                                      char *err, size_t errcap) {
    return qw_verify(s, block, n_block, sp, rng, out, err, errcap);
}

/* Commits head rows a verify left owed, when something other than a draft comes
 * next -- a tool result, a new turn.  Dropping them would leave a hole in the
 * head's history, which costs acceptance and reports nothing. */
static bool qw_mtp_flush_owed(qwasar_session *s, char *err, size_t errcap) {
    if (!s->mtp_on || !s->mtp_after_verify) return true;
    const qw_config  *cfg   = s->cfg;
    const qw_qlinear *embed = qwasar_engine_embed(s->e);
    const int32_t rows = s->mtp_owed;
    s->mtp_after_verify = false;
    s->mtp_owed = 0;
    if (rows == 0) {
        /* Nothing owed, but the pending slot still holds the state from before
         * the verify.  What belongs there is the verify's own last confirmed
         * hidden, which is row 1. */
        qw_cmd c0 = qw_cmd_begin();
        if (!c0) { qw_gerrf(err, errcap, "cannot begin a command buffer"); return false; }
        qw_op_slice_rows(c0, qw_ref_at(s->mtp_hidden, 0),
                         qw_off(s->mtp_hidden, (size_t)cfg->hidden_size),
                         1, cfg->hidden_size, 0, cfg->hidden_size);
        qw_cmd_wait(c0);
        const char *e0 = qw_cmd_error(c0);
        qw_cmd_free(c0);
        if (e0) { qw_gerrf(err, errcap, "GPU error: %s", e0); return false; }
        s->mtp_pending = true;
        return true;
    }

    int32_t *tv = qw_buf_contents(s->mtp_tokens);
    for (int32_t i = 0; i < rows; i++) tv[i] = s->mtp_owed_tokens[i];
    qw_mtp_positions(s, 0, s->mtp_n_past, rows);

    qw_cmd c = qw_cmd_begin();
    if (!c) { qw_gerrf(err, errcap, "cannot begin a command buffer"); return false; }
    qw_op_embed_q4(c, qw_ref_at(s->mtp_embed, 0), qw_ref_at(s->mtp_tokens, 0),
                   qw_tensor_ref(embed->weight), qw_tensor_ref(embed->scales),
                   qw_tensor_ref(embed->biases), cfg->hidden_size, rows);
    qw_encode_mtp_rows(s, c, qw_off(s->mtp_hidden, (size_t)cfg->hidden_size), rows,
                       qw_ref_at(s->mtp_positions, 0));
    qw_op_slice_rows(c, qw_ref_at(s->mtp_hidden, 0),
                     qw_off(s->mtp_hidden, (size_t)(rows + 1) * cfg->hidden_size),
                     1, cfg->hidden_size, 0, cfg->hidden_size);
    qw_cmd_wait(c);
    const char *cerr = qw_cmd_error(c);
    qw_cmd_free(c);
    if (cerr) { qw_gerrf(err, errcap, "GPU error: %s", cerr); return false; }
    s->mtp_n_past += rows;
    s->mtp_pending = true;
    return true;
}

/* ---- images ---------------------------------------------------------------
 *
 * Two things have to happen for an image, and they are unrelated to each other
 * except that both are driven by where its <|image_pad|> tokens sit.
 *
 * The embeddings are a substitution: the tower has already produced one row per
 * merged 2x2 block, and those rows replace what the embedding table produced
 * for the pad tokens.  Nothing else about the forward pass changes.
 *
 * The positions are the part that has been deferred since milestone 1.  Text
 * advances all three MRoPE axes together, which is why treating them as one
 * counter has been correct until now.  An image does not: its tokens take a
 * frame index, a row and a column, and the next text token resumes from one
 * past the largest of the three.  So a prompt containing an image ends at a
 * lower position than its token count -- a 16x16 grid is 64 tokens but advances
 * position by 8 -- and every position after it depends on that. */

bool qwasar_session_attach_images(qwasar_session *s, const qwasar_image_input *im,
                                  int32_t n_images, char *err, size_t errcap) {
    if (!s) return false;
    qw_buf_free(s->img_rows);
    s->img_rows = 0;
    s->n_img_rows = 0;
    if (n_images <= 0) return true;

    const int32_t hidden = s->cfg->hidden_size;
    int32_t total = 0;
    for (int32_t i = 0; i < n_images; i++) total += im[i].n_rows;

    s->img_rows = qw_buf_alloc((size_t)total * hidden * sizeof(float));
    if (!s->img_rows) { qw_gerrf(err, errcap, "cannot allocate %d image rows", total); return false; }
    float *dst = qw_buf_contents(s->img_rows);
    for (int32_t i = 0; i < n_images; i++) {
        memcpy(dst, im[i].rows, (size_t)im[i].n_rows * hidden * sizeof(float));
        dst += (size_t)im[i].n_rows * hidden;
    }
    s->n_img_rows = total;
    s->img_cursor = 0;
    return true;
}

/* Position triples for a whole prompt, following the reference's rope index.
 * Only called when images are attached; without them all three axes are the
 * running token index and no array is needed. */
bool qw_mrope_positions(const qw_config *c, const int32_t *tokens, int32_t n,
                        const qwasar_image_input *im, int32_t n_images,
                        int32_t start, int32_t *out, int32_t *out_next,
                        char *err, size_t errcap) {
    const int32_t merge = c->vis_spatial_merge_size;
    int32_t pos = start;
    int32_t at = 0, img = 0;

    while (at < n) {
        const bool is_pad = tokens[at] == c->image_token_id
                         || tokens[at] == c->video_token_id;
        if (img < n_images && is_pad) {
            const int32_t gt = im[img].grid_t;
            const int32_t gh = im[img].grid_h / merge, gw = im[img].grid_w / merge;
            const int32_t count = gt * gh * gw;
            if (at + count > n) {
                qw_gerrf(err, errcap, "image %d needs %d pad tokens, only %d remain",
                         img, count, n - at);
                return false;
            }
            for (int32_t t = 0; t < gt; t++)
                for (int32_t y = 0; y < gh; y++)
                    for (int32_t x = 0; x < gw; x++) {
                        const int32_t k = at + (t * gh + y) * gw + x;
                        out[0 * n + k] = pos + t;
                        out[1 * n + k] = pos + y;
                        out[2 * n + k] = pos + x;
                    }
            int32_t span = gt > gh ? gt : gh;
            if (gw > span) span = gw;
            pos += span;
            at  += count;
            img++;
            continue;
        }
        for (int a = 0; a < 3; a++) out[a * n + at] = pos;
        pos++;
        at++;
    }
    *out_next = pos;
    return true;
}

static bool qw_build_mrope(qwasar_session *s, const int32_t *tokens, int32_t n,
                           const qwasar_image_input *im, int32_t n_images,
                           char *err, size_t errcap) {
    free(s->mrope);
    s->mrope = malloc((size_t)3 * n * sizeof *s->mrope);
    if (!s->mrope) { qw_gerrf(err, errcap, "out of memory"); return false; }
    s->mrope_len = n;

    /* The prompt starts wherever the session already is; a conversation with
     * no images before this one counted its tokens one per position. */
    if (!s->mrope_active) { s->mrope_next = s->n_past; s->mrope_active = true; }
    return qw_mrope_positions(s->cfg, tokens, n, im, n_images, s->mrope_next,
                              s->mrope, &s->mrope_next, err, errcap);
}

const float *qwasar_session_eval_images(qwasar_session *s, const int32_t *tokens,
                                        int32_t n, const qwasar_image_input *im,
                                        int32_t n_images, char *err, size_t errcap) {
    if (n_images > 0) {
        if (!qwasar_session_attach_images(s, im, n_images, err, errcap)) return NULL;
        if (!qw_build_mrope(s, tokens, n, im, n_images, err, errcap)) return NULL;
    }
    const float *r = qwasar_session_eval(s, tokens, n, err, errcap);
    /* One prompt's worth: a following decode step is ordinary text again. */
    free(s->mrope);
    s->mrope = NULL;
    s->mrope_len = 0;
    return r;
}

const float *qwasar_session_eval(qwasar_session *s, const int32_t *tokens, int32_t n,
                                 char *err, size_t errcap) {
    if (!s || !tokens || n <= 0) {
        qw_gerrf(err, errcap, "nothing to evaluate");
        return NULL;
    }
    if (s->n_past + n > s->max_ctx) {
        qw_gerrf(err, errcap, "context exhausted: %d + %d exceeds %d tokens",
                 s->n_past, n, s->max_ctx);
        return NULL;
    }
    if (!qw_mtp_flush_owed(s, err, errcap)) return NULL;

    /* Only worth reporting when there is a prompt to grind through; a
     * single-token decode step would just flicker. */
    const bool report = s->progress && n > 1;
    if (report) s->progress(s->progress_ud, 0, n);

    for (int32_t done = 0; done < n; ) {
        const int32_t rows = (n - done < s->max_rows) ? n - done : s->max_rows;
        const bool last_chunk = (done + rows == n);

        memcpy(qw_buf_contents(s->tokens), tokens + done, (size_t)rows * sizeof(int32_t));

        if (s->n_history + rows <= s->max_ctx) {
            if (!s->history) s->history = malloc((size_t)s->max_ctx * sizeof *s->history);
            if (s->history) {
                memcpy(s->history + s->n_history, tokens + done,
                       (size_t)rows * sizeof *s->history);
                s->n_history += rows;
            }
        }

        /* Text advances all three MRoPE axes together, so the running index is
         * all three positions.  An image is what makes them diverge, and when
         * one is attached the whole prompt's triples were computed up front --
         * a later chunk's positions depend on every image before it. */
        int32_t *pos = qw_buf_contents(s->positions);
        const bool from_array = s->mrope && done + rows <= s->mrope_len;
        const int32_t base = s->mrope_active ? s->mrope_next : s->n_past;
        for (int32_t axis = 0; axis < 3; axis++)
            for (int32_t r = 0; r < rows; r++)
                pos[axis * rows + r] = from_array
                    ? s->mrope[axis * s->mrope_len + done + r]
                    : base + r;
        /* The array already accounts for the whole prompt, so only the plain
         * path advances the counter. */
        if (!from_array && s->mrope_active) s->mrope_next += rows;

        qw_cmd c = qw_cmd_begin();
        if (!c) { qw_gerrf(err, errcap, "cannot begin a command buffer"); return NULL; }
        qw_encode_forward(s, c, rows, last_chunk);
        qw_cmd_wait(c);
        const char *cerr = qw_cmd_error(c);
        if (cerr) {
            qw_gerrf(err, errcap, "GPU error: %s", cerr);
            qw_cmd_free(c);
            return NULL;
        }
        qw_cmd_free(c);

        s->n_past += rows;
        done += rows;
        if (report) s->progress(s->progress_ud, done, n);
    }
    return qw_buf_contents(s->logits);
}
