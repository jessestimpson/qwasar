/* qwasar_flash_graph.c -- the Flash-Next (qwen4_exp) forward, on Metal.
 *
 * PLAN-flash-next.md, Phases 3-5.  The same session, the same chunked eval,
 * the same command buffer discipline as qwasar_graph.c: one serial encoder,
 * every op a `rows`-wide batch, so prefill and decode are one code path.
 * What differs is the layer: four residual streams, a hyper-connection
 * around every sublayer, MoE in place of the dense MLP, QSA's indexer in
 * front of attention, and the engram layer once.  Every op here is held to
 * qwasar_flash_cpu.c by tests/test_flashnext.
 *
 * Two things are host-side by design: the engram hash and gather (a few
 * table rows per token off a cold mmap -- the table is never a GPU operand),
 * and nothing else.  Routing, selection and the expert matvecs all stay on
 * the GPU, in the simplest shapes that are right; the measurements that
 * would justify faster ones are Phase 6's. */

#include "qwasar_session.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct qw_flash_state {
    /* streams and mixers */
    qw_buf h4, n4, mixd, mixu, inj;
    /* MoE */
    qw_buf route_logits, route_idx, route_w, exp_gu, exp_act, exp_y;
    qw_buf sh_g, sh_u, sh_out, sh_gs;
    /* QSA */
    qw_buf iqk, iq, iscores, mask, ikeys;   /* ikeys: [n_full, max_ctx, d] fp32 */
    int32_t max_blocks;
    /* PLE */
    qw_buf ple_emb, ple_key, ple_keyn, ple_val, ple_qn, ple_gv, ple_gvn, ple_conv, ple_state;
    int32_t ple_state_len;
    int32_t last_eos;
    int32_t hist[8];
    /* diagnostics: stop the forward after this layer (-1: run it all) */
    int32_t dbg_stop_layer;
};

void qw_flash_debug_stop(qwasar_session *s, int32_t layer) {
    if (s && s->flash) s->flash->dbg_stop_layer = layer;
}
const float *qw_flash_debug_h4(const qwasar_session *s) {
    return (s && s->flash) ? (const float *)qw_buf_contents(s->flash->h4) : NULL;
}

static void *fcontents(qw_buf b) { return qw_buf_contents(b); }

struct qw_flash_state *qw_flash_state_new(qwasar_session *s, char *err, size_t errcap) {
    const qw_config *c = s->cfg;
    const qw_shape *sh = s->shape;
    const int32_t R = s->max_rows, H = c->hidden_size, S = c->hc_count, HH = sh->hc_hidden;
    const int32_t K = c->num_experts_per_tok, I = c->moe_intermediate_size;
    const int32_t SI = c->shared_expert_intermediate_size, E = c->num_experts;
    const int32_t nq = c->indexer_n_heads, d = c->indexer_head_dim;

    struct qw_flash_state *f = calloc(1, sizeof *f);
    if (!f) { qw_verrf(err, errcap, "out of memory"); return NULL; }
    f->max_blocks = s->max_ctx / c->indexer_compress_ratio;
    f->ple_state_len = (c->ple_conv_kernel_size - 1) * c->ngram_size;
    f->last_eos = -1;
    f->dbg_stop_layer = -1;
    for (int i = 0; i < 8; i++) f->hist[i] = -1;

    struct { qw_buf *b; size_t n; const char *name; } want[] = {
        { &f->h4,   (size_t)R * HH * 4, "h4" },
        { &f->n4,   (size_t)R * HH * 4, "n4" },
        { &f->mixd, (size_t)R * c->hc_lowrank * 4, "mix down" },
        { &f->mixu, (size_t)R * HH * 4, "mix up" },
        { &f->inj,  (size_t)R * S * 4, "inject" },
        { &f->route_logits, (size_t)R * E * 4, "router" },
        { &f->route_idx,    (size_t)R * K * 4, "route idx" },
        { &f->route_w,      (size_t)R * K * 4, "route w" },
        { &f->exp_gu,  (size_t)R * K * 2 * I * 4, "expert gate|up" },
        { &f->exp_act, (size_t)R * K * I * 4, "expert act" },
        { &f->exp_y,   (size_t)R * K * H * 4, "expert out" },
        { &f->sh_g,   (size_t)R * SI * 4, "shared gate" },
        { &f->sh_u,   (size_t)R * SI * 4, "shared up" },
        { &f->sh_out, (size_t)R * H * 4, "shared out" },
        { &f->sh_gs,  (size_t)R * 4, "shared scale" },
        { &f->iqk,     (size_t)R * (nq + 1) * d * 4, "index qk" },
        { &f->iq,      (size_t)R * nq * d * 4, "index q" },
        { &f->iscores, (size_t)R * (f->max_blocks > 0 ? f->max_blocks : 1) * 4, "index scores" },
        { &f->mask,    (size_t)R * s->max_ctx, "qsa mask" },
        { &f->ikeys,   (size_t)sh->n_full_attn_layers * s->max_ctx * d * 4, "index keys" },
    };
    for (size_t i = 0; i < sizeof want / sizeof *want; i++) {
        *want[i].b = qw_buf_alloc(want[i].n);
        if (!*want[i].b) {
            qw_verrf(err, errcap, "cannot allocate %s (%.1f MB)", want[i].name, want[i].n / 1048576.0);
            qw_flash_state_free(f);
            return NULL;
        }
    }
    if (c->ple_layer >= 0) {
        const int32_t E2 = c->ple_embed_dim;
        struct { qw_buf *b; size_t n; } pw[] = {
            { &f->ple_emb,  (size_t)R * E2 * 4 }, { &f->ple_key,  (size_t)R * HH * 4 },
            { &f->ple_keyn, (size_t)R * HH * 4 }, { &f->ple_val,  (size_t)R * H * 4 },
            { &f->ple_qn,   (size_t)R * HH * 4 }, { &f->ple_gv,   (size_t)R * HH * 4 },
            { &f->ple_gvn,  (size_t)R * HH * 4 }, { &f->ple_conv, (size_t)R * HH * 4 },
            { &f->ple_state, (size_t)f->ple_state_len * HH * 4 },
        };
        for (size_t i = 0; i < sizeof pw / sizeof *pw; i++) {
            *pw[i].b = qw_buf_alloc(pw[i].n);
            if (!*pw[i].b) {
                qw_verrf(err, errcap, "cannot allocate the engram scratch");
                qw_flash_state_free(f);
                return NULL;
            }
        }
        memset(fcontents(f->ple_state), 0, (size_t)f->ple_state_len * HH * 4);
    }
    return f;
}

void qw_flash_state_free(struct qw_flash_state *f) {
    if (!f) return;
    qw_buf *all[] = {
        &f->h4, &f->n4, &f->mixd, &f->mixu, &f->inj,
        &f->route_logits, &f->route_idx, &f->route_w, &f->exp_gu, &f->exp_act, &f->exp_y,
        &f->sh_g, &f->sh_u, &f->sh_out, &f->sh_gs,
        &f->iqk, &f->iq, &f->iscores, &f->mask, &f->ikeys,
        &f->ple_emb, &f->ple_key, &f->ple_keyn, &f->ple_val, &f->ple_qn, &f->ple_gv,
        &f->ple_gvn, &f->ple_conv, &f->ple_state,
    };
    for (size_t i = 0; i < sizeof all / sizeof *all; i++) qw_buf_free(*all[i]);
    free(f);
}

/* ---- the engram, host side ---------------------------------------------------
 *
 * The hash needs the token history and the EOS bookkeeping, and the gather
 * reads a few rows of a table that is never a GPU operand, so both happen
 * here, per chunk, before the command buffer is encoded. */
void qw_flash_prepare_chunk(qwasar_session *s, const int32_t *tokens, int32_t rows) {
    struct qw_flash_state *f = s->flash;
    const qw_config *c = s->cfg;
    if (c->ple_layer < 0) return;
    const qw_layer *L = qwasar_engine_layer(s->e, c->ple_layer);
    const qw_ple *P = L->ple;
    const int32_t HD = P->head_dim, E = c->ple_embed_dim;
    float *emb = fcontents(f->ple_emb);
    int64_t ids[QW_MAX_NGRAM_HEADS];

    for (int32_t r = 0; r < rows; r++) {
        const int32_t tok = tokens[r];
        const int32_t pos = s->n_past + r;
        for (int i = 7; i > 0; i--) f->hist[i] = f->hist[i - 1];
        f->hist[0] = tok;

        qw_ple_ids(P, c, f->hist, pos - 1 - f->last_eos, ids);
        for (int32_t h = 0; h < P->n_heads; h++) {
            const uint16_t *row = qw_ple_row(P, ids[h]);
            for (int32_t i = 0; i < HD; i++)
                emb[(size_t)r * E + (size_t)h * HD + i] = qw_bf16_to_f32_c(row[i]);
        }

        for (int32_t k = 0; k < c->n_eos; k++)
            if (tok == c->eos_token_ids[k]) { f->last_eos = pos; break; }
    }
}

/* ---- the pieces --------------------------------------------------------------- */

/* GatedResidual: streams -> normalised streams (n4), one mixed input (s->hn),
 * and the per-stream injection logits (inj) when the block has them. */
static void encode_hc(qwasar_session *s, qw_cmd c, const qw_hc *hc, int32_t rows) {
    struct qw_flash_state *f = s->flash;
    const qw_config *cfg = s->cfg;
    const int32_t H = cfg->hidden_size, S = cfg->hc_count, HH = s->shape->hc_hidden;

    qw_op_rms_norm_grouped(c, qw_ref_at(f->n4, 0), qw_ref_at(f->h4, 0), qw_tensor_ref(hc->hc_norm),
                           H, S, rows, cfg->rms_norm_eps);
    qw_encode_qlinear(c, &hc->mix_down, qw_ref_at(f->mixd, 0), qw_ref_at(f->n4, 0), rows);
    qw_op_silu_scale(c, qw_ref_at(f->mixd, 0), rows * cfg->hc_lowrank, 1.0f / (float)S);
    qw_encode_qlinear(c, &hc->mix_up, qw_ref_at(f->mixu, 0), qw_ref_at(f->mixd, 0), rows);
    qw_op_hc_mix(c, qw_ref_at(s->hn, 0), qw_ref_at(f->n4, 0), qw_ref_at(f->mixu, 0), rows, H, S);
    if (hc->block_inject)
        qw_op_dmat_bf16(c, qw_ref_at(f->inj, 0), qw_ref_at(f->n4, 0),
                        qw_tensor_ref(hc->block_inject), HH, S, rows);
}

static void encode_inject(qwasar_session *s, qw_cmd c, int32_t rows) {
    struct qw_flash_state *f = s->flash;
    qw_op_hc_inject(c, qw_ref_at(f->h4, 0), qw_ref_at(s->hn2, 0), qw_ref_at(f->inj, 0),
                    rows, s->cfg->hidden_size, s->cfg->hc_count);
}

/* The engram layer: h4 += gated value + silu(dilated conv(normed gated value)). */
static void encode_ple(qwasar_session *s, qw_cmd c, const qw_ple *P, int32_t rows) {
    struct qw_flash_state *f = s->flash;
    const qw_config *cfg = s->cfg;
    const int32_t H = cfg->hidden_size, S = cfg->hc_count, HH = s->shape->hc_hidden;

    qw_encode_qlinear(c, &P->key_proj, qw_ref_at(f->ple_key, 0), qw_ref_at(f->ple_emb, 0), rows);
    qw_op_rms_norm_grouped(c, qw_ref_at(f->ple_keyn, 0), qw_ref_at(f->ple_key, 0),
                           qw_tensor_ref(P->norm_key), H, S, rows, cfg->rms_norm_eps);
    qw_encode_qlinear(c, &P->value_proj, qw_ref_at(f->ple_val, 0), qw_ref_at(f->ple_emb, 0), rows);
    qw_op_rms_norm_grouped(c, qw_ref_at(f->ple_qn, 0), qw_ref_at(f->h4, 0),
                           qw_tensor_ref(P->norm_query), H, S, rows, cfg->rms_norm_eps);
    qw_op_ple_gate(c, qw_ref_at(f->ple_gv, 0), qw_ref_at(f->ple_keyn, 0), qw_ref_at(f->ple_qn, 0),
                   qw_ref_at(f->ple_val, 0), rows, H, S);
    qw_op_rms_norm_grouped(c, qw_ref_at(f->ple_gvn, 0), qw_ref_at(f->ple_gv, 0),
                           qw_tensor_ref(P->norm_conv), H, S, rows, cfg->rms_norm_eps);
    qw_op_conv1d_dilated_silu(c, qw_ref_at(f->ple_conv, 0), qw_ref_at(f->ple_gvn, 0),
                              qw_ref_at(f->ple_state, 0), qw_tensor_ref(P->conv1d),
                              HH, rows, cfg->ple_conv_kernel_size, cfg->ngram_size);
    qw_op_add_inplace(c, qw_ref_at(f->h4, 0), qw_ref_at(f->ple_gv, 0), rows * HH);
    qw_op_add_inplace(c, qw_ref_at(f->h4, 0), qw_ref_at(f->ple_conv, 0), rows * HH);
}

/* Gated attention with QSA in front of it.  Mirrors qw_encode_attention_layer
 * up to the cache write; the indexer then decides what the query may see. */
static void encode_qsa_layer(qwasar_session *s, qw_cmd c, const qw_layer *L, int32_t fi, int32_t rows) {
    struct qw_flash_state *f = s->flash;
    const qw_config *cfg = s->cfg;
    const qw_shape  *sh  = s->shape;
    const int32_t hd = cfg->head_dim;
    const int32_t nq = cfg->indexer_n_heads, d = cfg->indexer_head_dim;
    const size_t kv_stride = (size_t)cfg->num_key_value_heads * s->max_ctx * hd;

    qw_encode_qlinear(c, &L->q_proj, qw_ref_at(s->qg, 0), qw_ref_at(s->hn, 0), rows);
    qw_encode_qlinear(c, &L->k_proj, qw_ref_at(s->k, 0),  qw_ref_at(s->hn, 0), rows);
    qw_encode_qlinear(c, &L->v_proj, qw_ref_at(s->v, 0),  qw_ref_at(s->hn, 0), rows);
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

    /* The indexer: query heads normed and rotated at their own positions;
     * raw keys appended to this layer's key cache. */
    const qw_ref ikeys = qw_off(f->ikeys, (size_t)fi * s->max_ctx * d);
    qw_encode_qlinear(c, &L->indexer.qk_proj, qw_ref_at(f->iqk, 0), qw_ref_at(s->hn, 0), rows);
    qw_op_slice_rows(c, qw_ref_at(f->iq, 0), qw_ref_at(f->iqk, 0), rows, (nq + 1) * d, 0, nq * d);
    qw_op_rms_norm(c, qw_ref_at(f->iq, 0), qw_ref_at(f->iq, 0), qw_tensor_ref(L->indexer.q_norm),
                   d, rows * nq, cfg->rms_norm_eps, 1.0f);
    qw_op_rope_partial(c, qw_ref_at(f->iq, 0), qw_ref_at(s->positions, 0),
                       qw_ref_at(s->rope_axis, 0), qw_ref_at(s->rope_inv_freq, 0),
                       rows, nq, d, cfg->rotary_dim);
    qw_op_slice_rows(c, qw_ref_offset(ikeys, (size_t)s->n_past * d * 4), qw_ref_at(f->iqk, 0),
                     rows, (nq + 1) * d, nq * d, d);
    qw_op_qsa_scores(c, qw_ref_at(f->iscores, 0), qw_ref_at(f->iq, 0), ikeys,
                     qw_tensor_ref(L->indexer.k_norm), qw_ref_at(s->rope_inv_freq, 0),
                     rows, nq, d, cfg->indexer_compress_ratio, s->n_past, cfg->rotary_dim,
                     f->max_blocks, cfg->rms_norm_eps);
    qw_op_qsa_select(c, qw_ref_at(f->mask, 0), qw_ref_at(f->iscores, 0), rows,
                     cfg->indexer_compress_ratio, s->n_past,
                     cfg->indexer_budget / cfg->indexer_compress_ratio, s->max_ctx, f->max_blocks);

    qw_op_attn_masked(c, qw_ref_at(s->attn_out, 0), qw_ref_at(s->q, 0),
                      qw_ref_at(s->kcache, kv_stride * fi * sizeof(uint16_t)),
                      qw_ref_at(s->vcache, kv_stride * fi * sizeof(uint16_t)),
                      qw_ref_at(f->mask, 0),
                      rows, cfg->num_attention_heads, cfg->num_key_value_heads,
                      hd, s->max_ctx, s->n_past, 1.0f / sqrtf((float)hd));
    qw_op_mul_sigmoid(c, qw_ref_at(s->attn_out, 0), qw_ref_at(s->gate, 0), rows * sh->q_dim);
    qw_encode_qlinear(c, &L->o_proj, qw_ref_at(s->hn2, 0), qw_ref_at(s->attn_out, 0), rows);
}

/* The MoE block: router, routed experts, the gated shared expert; into hn2. */
static void encode_moe(qwasar_session *s, qw_cmd c, const qw_moe *M, int32_t rows) {
    struct qw_flash_state *f = s->flash;
    const qw_config *cfg = s->cfg;
    const int32_t H = cfg->hidden_size, K = cfg->num_experts_per_tok;
    const int32_t I = cfg->moe_intermediate_size, SI = cfg->shared_expert_intermediate_size;
    const int32_t pairs = rows * K;

    qw_op_dmat_bf16(c, qw_ref_at(f->route_logits, 0), qw_ref_at(s->hn, 0),
                    qw_tensor_ref(M->router), H, M->n_experts, rows);
    qw_op_moe_route(c, qw_ref_at(f->route_idx, 0), qw_ref_at(f->route_w, 0),
                    qw_ref_at(f->route_logits, 0), rows, M->n_experts, K, cfg->norm_topk_prob);
    qw_op_qmv_q4_bank(c, qw_ref_at(f->exp_gu, 0), qw_ref_at(s->hn, 0), qw_ref_at(f->route_idx, 0),
                      qw_tensor_ref(M->gate_up.weight), qw_tensor_ref(M->gate_up.scales),
                      qw_tensor_ref(M->gate_up.biases), H, 2 * I, pairs, K, false);
    qw_op_swiglu_split(c, qw_ref_at(f->exp_act, 0), qw_ref_at(f->exp_gu, 0), pairs, I);
    qw_op_qmv_q4_bank(c, qw_ref_at(f->exp_y, 0), qw_ref_at(f->exp_act, 0), qw_ref_at(f->route_idx, 0),
                      qw_tensor_ref(M->down.weight), qw_tensor_ref(M->down.scales),
                      qw_tensor_ref(M->down.biases), I, H, pairs, K, true);
    qw_op_moe_combine(c, qw_ref_at(s->hn2, 0), qw_ref_at(f->exp_y, 0), qw_ref_at(f->route_w, 0),
                      rows, K, H);

    qw_encode_qlinear(c, &M->sh_gate, qw_ref_at(f->sh_g, 0), qw_ref_at(s->hn, 0), rows);
    qw_encode_qlinear(c, &M->sh_up,   qw_ref_at(f->sh_u, 0), qw_ref_at(s->hn, 0), rows);
    qw_op_swiglu(c, qw_ref_at(f->sh_g, 0), qw_ref_at(f->sh_g, 0), qw_ref_at(f->sh_u, 0), rows * SI);
    qw_encode_qlinear(c, &M->sh_down, qw_ref_at(f->sh_out, 0), qw_ref_at(f->sh_g, 0), rows);
    qw_op_dmat_bf16(c, qw_ref_at(f->sh_gs, 0), qw_ref_at(s->hn, 0), qw_tensor_ref(M->sh_gate_w),
                    H, 1, rows);
    qw_op_scale_rows_sigmoid(c, qw_ref_at(f->sh_out, 0), qw_ref_at(f->sh_gs, 0), rows, H);
    qw_op_add_inplace(c, qw_ref_at(s->hn2, 0), qw_ref_at(f->sh_out, 0), rows * H);
}

/* ---- the forward --------------------------------------------------------------- */

void qw_flash_encode_forward(qwasar_session *s, qw_cmd c, int32_t rows, bool want_logits) {
    struct qw_flash_state *f = s->flash;
    const qw_config *cfg = s->cfg;
    qwasar_engine *e = s->e;
    const int32_t H = cfg->hidden_size, S = cfg->hc_count;

    qw_op_embed_q4(c, qw_ref_at(s->h, 0), qw_ref_at(s->tokens, 0),
                   qw_tensor_ref(qwasar_engine_embed(e)->weight),
                   qw_tensor_ref(qwasar_engine_embed(e)->scales),
                   qw_tensor_ref(qwasar_engine_embed(e)->biases), H, rows);
    qw_op_repeat_cols(c, qw_ref_at(f->h4, 0), qw_ref_at(s->h, 0), rows, H, S);

    for (int32_t i = 0; i < cfg->num_hidden_layers; i++) {
        const qw_layer *L = qwasar_engine_layer(e, i);
        if (L->ple) encode_ple(s, c, L->ple, rows);

        encode_hc(s, c, &L->attn_hc, rows);
        if (L->is_linear_attn) qw_encode_gated_delta_layer(s, c, L, s->kind_index[i], rows);
        else                   encode_qsa_layer(s, c, L, s->kind_index[i], rows);
        encode_inject(s, c, rows);

        encode_hc(s, c, &L->mlp_hc, rows);
        encode_moe(s, c, &L->moe, rows);
        encode_inject(s, c, rows);

        if (f->dbg_stop_layer == i) return;
    }

    if (!want_logits) return;
    /* The final mixer feeds lm_head directly: there is no norm between. */
    encode_hc(s, c, qwasar_engine_final_hc(e), rows);
    qw_encode_qlinear(c, qwasar_engine_head(e), qw_ref_at(s->logits, 0),
                      qw_off(s->hn, (size_t)(rows - 1) * H), 1);
}
