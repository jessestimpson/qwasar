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

    qwasar_progress_fn progress;
    void              *progress_ud;

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
    };
    for (size_t i = 0; i < sizeof all / sizeof *all; i++) qw_buf_free(*all[i]);
    qw_buf_free(s->capture);
    free(s->capture_layers);
    free(s->history);
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
                             cfg->linear_conv_kernel_dim);

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
                      qw_off(s->ssm_state, ssm_stride * li), hk, hv, dk, dv, rows);

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
static void qw_encode_forward(qwasar_session *s, qw_cmd c, int32_t rows, bool want_logits) {
    const qw_config *cfg = s->cfg;
    qwasar_engine *e = s->e;

    qw_op_embed_q4(c, qw_ref_at(s->h, 0), qw_ref_at(s->tokens, 0),
                   qw_tensor_ref(qwasar_engine_embed(e)->weight),
                   qw_tensor_ref(qwasar_engine_embed(e)->scales),
                   qw_tensor_ref(qwasar_engine_embed(e)->biases),
                   cfg->hidden_size, rows);

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

    if (!want_logits) return;

    /* Only the final token's logits are ever used, and the head is the single
     * widest matvec in the model, so it runs on one row rather than `rows`. */
    qw_op_rms_norm(c, qw_ref_at(s->hn, 0), qw_ref_at(s->h, 0),
                   qw_tensor_ref(qwasar_engine_final_norm(e)),
                   cfg->hidden_size, rows, cfg->rms_norm_eps, 1.0f);
    qw_encode_qlinear(c, qwasar_engine_head(e), qw_ref_at(s->logits, 0),
                      qw_off(s->hn, (size_t)(rows - 1) * cfg->hidden_size), 1);
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

        /* Text-only positions are identical on all three MRoPE axes; images are
         * what make them diverge, and that is Milestone 3's business. */
        int32_t *pos = qw_buf_contents(s->positions);
        for (int32_t axis = 0; axis < 3; axis++)
            for (int32_t r = 0; r < rows; r++)
                pos[axis * rows + r] = s->n_past + r;

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
