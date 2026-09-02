/* qwasar_flash_cpu.c -- the Flash-Next (qwen4_exp) family, as a scalar
 * reference forward over the engine's own bound tensors.
 *
 * PLAN-flash-next.md, Phase 2.  This is the twin every Metal op for the
 * family is checked against, written straight from the transformers source
 * with the same names in the same order, one token at a time -- the
 * recurrences are sequential anyway, and prefill-as-a-loop-of-decodes is
 * exactly the reference's own cached path, which the oracle showed agrees
 * with its prefill to 1e-6.
 *
 * What it holds per layer is what a session will have to hold: a Gated
 * DeltaNet state and conv window, or a KV cache plus the indexer's raw key
 * cache; and on the engram layer, the two previous tokens and the dilated
 * conv's window.  Everything is fp32.  The quantised weights are read
 * through the same dequantisation the kernels use, so a mismatch against
 * the oracle is a mismatch of mechanics, not of rounding. */

#include "qwasar_model.h"
#include "qwasar_gpu.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ---- small helpers --------------------------------------------------------- */

static float sigm(float x) { return 1.0f / (1.0f + expf(-x)); }
static float silu(float x) { return x * sigm(x); }

static const void *tdata(const qw_tensor *t) {
    return (const char *)qw_buf_contents(t->buf) + t->offset;
}

/* y[out] = W x for one quantised row block; `row0` offsets into a bank. */
static void qmv(float *y, const float *x, const qw_qlinear *q, int64_t row0, int32_t rows) {
    const int32_t k = q->in_features;
    const uint32_t *w = (const uint32_t *)tdata(q->weight) + (size_t)row0 * (k / 8);
    const uint16_t *s = (const uint16_t *)tdata(q->scales) + (size_t)row0 * (k / 64);
    const uint16_t *b = (const uint16_t *)tdata(q->biases) + (size_t)row0 * (k / 64);
    qw_cpu_qmv_q4(y, x, w, s, b, k, rows, 1);
}
static void qlin(float *y, const float *x, const qw_qlinear *q) { qmv(y, x, q, 0, q->out_features); }

/* y[n] = W[n,k] x, W in BF16. */
static void dmv(float *y, const float *x, const qw_tensor *w, int32_t k, int32_t n) {
    qw_cpu_dmv_bf16(y, x, (const uint16_t *)tdata(w), k, n, 1);
}

/* RMSNorm with a BF16 weight already carrying its +1. */
static void rms(float *y, const float *x, const qw_tensor *w, int32_t dim, float eps) {
    qw_cpu_rms_norm(y, x, w ? (const uint16_t *)tdata(w) : NULL, dim, 1, eps, 1.0f);
}

/* The hyper-connection norm: `groups` streams of `dim`, each normalised on
 * its own, then the full-width (+1) weight. */
static void grouped_rms(float *y, const float *x, const qw_tensor *w,
                        int32_t groups, int32_t dim, float eps) {
    const uint16_t *wp = (const uint16_t *)tdata(w);
    for (int32_t g = 0; g < groups; g++) {
        const float *xg = x + (size_t)g * dim;
        float *yg = y + (size_t)g * dim;
        float ss = 0.0f;
        for (int32_t i = 0; i < dim; i++) ss += xg[i] * xg[i];
        const float inv = 1.0f / sqrtf(ss / (float)dim + eps);
        for (int32_t i = 0; i < dim; i++)
            yg[i] = xg[i] * inv * qw_bf16_to_f32_c(wp[(size_t)g * dim + i]);
    }
}

/* FLA's l2norm: x / sqrt(sum x^2 + 1e-6). */
static void l2norm(float *x, int32_t n, float scale) {
    float ss = 0.0f;
    for (int32_t i = 0; i < n; i++) ss += x[i] * x[i];
    const float inv = scale / sqrtf(ss + 1e-6f);
    for (int32_t i = 0; i < n; i++) x[i] *= inv;
}

/* ---- the engram hash ------------------------------------------------------- */

#define SPLITMIX_GAMMA 0x9E3779B97F4A7C15ULL
#define SPLITMIX_M1    0xBF58476D1CE4E5B9ULL
#define SPLITMIX_M2    0x94D049BB133111EBULL

static uint64_t splitmix64(uint64_t v) {
    v += SPLITMIX_GAMMA;
    v = (v ^ (v >> 30)) * SPLITMIX_M1;
    v = (v ^ (v >> 27)) * SPLITMIX_M2;
    return v ^ (v >> 31);
}

void qw_ple_multipliers(int64_t out[8], int64_t unigram_vocab, int32_t ngram_size,
                        int32_t ple_layer_index, int32_t seed) {
    const int64_t max_long = INT64_MAX;
    const int64_t multiplier_max = max_long / (unigram_vocab > 1 ? unigram_vocab : 1);
    int64_t half = multiplier_max / 2;
    if (half < 1) half = 1;
    const uint64_t base = (uint64_t)((int64_t)seed + 10007LL * ple_layer_index);
    for (int32_t i = 0; i < ngram_size && i < 8; i++) {
        const uint64_t v = base + SPLITMIX_GAMMA * (uint64_t)(i + 1);
        /* Python's % on a non-negative left side is plain modulo. */
        out[i] = 2 * (int64_t)(splitmix64(v) % (uint64_t)half) + 1;
    }
}

/* torch.remainder: the result takes the divisor's sign, i.e. non-negative. */
static int64_t pymod(int64_t a, int64_t p) {
    int64_t r = a % p;
    return r < 0 ? r + p : r;
}

/* ---- state ----------------------------------------------------------------- */

typedef struct {
    /* linear attention */
    float *conv_state;   /* [K-1][conv_dim] */
    float *ssm;          /* [hv][dv][dk] */
    /* sparse attention */
    float *kc, *vc;      /* [max_ctx][kv_dim] */
    float *ikeys;        /* [max_ctx][index_head_dim], raw */
} ref_layer;

struct qw_flash_ref {
    const qwasar_engine *e;
    const qw_config *c;
    const qw_shape  *sh;
    int32_t max_ctx, n_past;
    ref_layer *L;
    /* engram: the two previous tokens and where the last EOS was */
    int32_t last_eos;            /* position of the last EOS token seen, or -1 */
    int32_t hist[8];             /* the most recent tokens, hist[0] newest */
    float  *ple_conv_state;      /* [state_len][hc_hidden], oldest first */
    int32_t ple_state_len;
    /* rope */
    uint8_t axis[128];
    float   inv_freq[128];
};

qw_flash_ref *qw_flash_ref_new(const qwasar_engine *e, int32_t max_ctx) {
    const qw_config *c = qwasar_engine_config(e);
    const qw_shape *sh = qwasar_engine_shape(e);
    if (c->family != QW_FAMILY_QWEN4_EXP) return NULL;

    qw_flash_ref *r = calloc(1, sizeof *r);
    r->e = e; r->c = c; r->sh = sh; r->max_ctx = max_ctx;
    r->L = calloc((size_t)c->num_hidden_layers, sizeof *r->L);
    r->last_eos = -1;
    for (int i = 0; i < 8; i++) r->hist[i] = -1;

    for (int i = 0; i < c->num_hidden_layers; i++) {
        ref_layer *l = &r->L[i];
        if (qw_layer_is_linear(c, i)) {
            l->conv_state = calloc((size_t)(c->linear_conv_kernel_dim - 1) * sh->conv_dim, sizeof(float));
            l->ssm = calloc((size_t)c->linear_num_value_heads * c->linear_value_head_dim
                            * c->linear_key_head_dim, sizeof(float));
        } else {
            l->kc = calloc((size_t)max_ctx * sh->kv_dim, sizeof(float));
            l->vc = calloc((size_t)max_ctx * sh->kv_dim, sizeof(float));
            l->ikeys = calloc((size_t)max_ctx * c->indexer_head_dim, sizeof(float));
        }
    }
    if (c->ple_layer >= 0) {
        r->ple_state_len = (c->ple_conv_kernel_size - 1) * c->ngram_size;
        r->ple_conv_state = calloc((size_t)r->ple_state_len * sh->hc_hidden, sizeof(float));
    }
    qw_rope_tables(r->axis, r->inv_freq, c->rotary_dim, c->rope_theta, c->mrope_section);
    return r;
}

void qw_flash_ref_free(qw_flash_ref *r) {
    if (!r) return;
    for (int i = 0; i < r->c->num_hidden_layers; i++) {
        free(r->L[i].conv_state); free(r->L[i].ssm);
        free(r->L[i].kc); free(r->L[i].vc); free(r->L[i].ikeys);
    }
    free(r->L); free(r->ple_conv_state); free(r);
}

/* ---- the pieces ------------------------------------------------------------ */

/* GatedResidual: normalise, mix the streams to one input, and (when the
 * weights exist) compute the per-stream injection weights.  `n` receives
 * the normalised streams because PLE and the mixers all read it. */
static void hc_mix(const qw_flash_ref *r, const qw_hc *hc, const float *h4,
                   float *n, float *x, float *inj) {
    const qw_config *c = r->c;
    const int32_t H = c->hidden_size, S = c->hc_count, HH = r->sh->hc_hidden;
    grouped_rms(n, h4, hc->hc_norm, S, H, c->rms_norm_eps);

    float *d = malloc((size_t)c->hc_lowrank * sizeof(float));
    float *m = malloc((size_t)HH * sizeof(float));
    qlin(d, n, &hc->mix_down);
    for (int32_t i = 0; i < c->hc_lowrank; i++) d[i] = silu(d[i] / (float)S);
    qlin(m, d, &hc->mix_up);
    for (int32_t i = 0; i < H; i++) {
        float acc = 0.0f;
        for (int32_t s = 0; s < S; s++) {
            const size_t j = (size_t)s * H + i;
            acc += sigm(m[j]) * n[j];
        }
        x[i] = acc / (float)S;
    }
    if (inj && hc->block_inject) {
        dmv(inj, n, hc->block_inject, HH, S);
        for (int32_t s = 0; s < S; s++) inj[s] = 2.0f * sigm(inj[s] / (float)S);
    }
    free(d); free(m);
}

static void hc_inject(const qw_flash_ref *r, float *h4, const float *out, const float *inj) {
    const int32_t H = r->c->hidden_size;
    for (int32_t s = 0; s < r->c->hc_count; s++)
        for (int32_t i = 0; i < H; i++) h4[(size_t)s * H + i] += out[i] * inj[s];
}

/* Gated DeltaNet for one token, in the order the 27B's twins already pin,
 * with this family's sigmoid output gate. */
static void gdn_step(qw_flash_ref *r, const qw_layer *L, ref_layer *l,
                     const float *x, float *out) {
    const qw_config *c = r->c;
    const qw_shape *sh = r->sh;
    const int32_t hk = c->linear_num_key_heads, hv = c->linear_num_value_heads;
    const int32_t dk = c->linear_key_head_dim, dv = c->linear_value_head_dim;

    float *qkv = malloc((size_t)sh->conv_dim * sizeof(float));
    float *mixed = malloc((size_t)sh->conv_dim * sizeof(float));
    float *z = malloc((size_t)sh->value_dim * sizeof(float));
    float *a = malloc((size_t)hv * sizeof(float));
    float *b = malloc((size_t)hv * sizeof(float));
    float *g = malloc((size_t)hv * sizeof(float));
    float *beta = malloc((size_t)hv * sizeof(float));
    float *y = malloc((size_t)sh->value_dim * sizeof(float));
    float *yn = malloc((size_t)sh->value_dim * sizeof(float));

    qlin(qkv, x, &L->in_proj_qkv);
    qlin(z, x, &L->in_proj_z);
    qlin(b, x, &L->in_proj_b);
    qlin(a, x, &L->in_proj_a);

    qw_cpu_conv1d_causal_silu(mixed, qkv, l->conv_state, (const uint16_t *)tdata(L->conv1d),
                              sh->conv_dim, 1, c->linear_conv_kernel_dim);
    float *q = mixed, *k = mixed + sh->key_dim, *v = mixed + 2 * sh->key_dim;
    for (int32_t h = 0; h < hk; h++) {
        l2norm(q + (size_t)h * dk, dk, 1.0f / sqrtf((float)dk));
        l2norm(k + (size_t)h * dk, dk, 1.0f);
    }
    qw_cpu_gdn_gates(g, beta, a, b, (const uint16_t *)tdata(L->A_log),
                     (const uint16_t *)tdata(L->dt_bias), hv, 1);
    qw_cpu_gated_delta(y, q, k, v, g, beta, l->ssm, hk, hv, dk, dv, 1);

    /* rmsnorm(y) * w, per value head, then the gate: sigmoid(z) here. */
    const uint16_t *nw = (const uint16_t *)tdata(L->gdn_norm);
    for (int32_t h = 0; h < hv; h++) {
        qw_cpu_rms_norm(yn + (size_t)h * dv, y + (size_t)h * dv, nw, dv, 1, c->rms_norm_eps, 1.0f);
        for (int32_t i = 0; i < dv; i++) {
            const float zz = z[(size_t)h * dv + i];
            yn[(size_t)h * dv + i] *= c->gdn_gate_sigmoid ? sigm(zz) : silu(zz);
        }
    }
    qlin(out, yn, &L->out_proj);

    free(qkv); free(mixed); free(z); free(a); free(b); free(g); free(beta); free(y); free(yn);
}

/* Which cached positions this query may attend to: QSA's block selection,
 * or every position when the budget covers them all.  `pos` is the query's
 * (0-based) position; positions 0..pos are visible.  Writes a 0/1 mask. */
static void qsa_select(qw_flash_ref *r, const qw_layer *L, ref_layer *l,
                       const float *x, int32_t pos, uint8_t *allowed) {
    const qw_config *c = r->c;
    const int32_t nq = c->indexer_n_heads, d = c->indexer_head_dim;
    const int32_t ratio = c->indexer_compress_ratio;
    const int32_t block_topk = c->indexer_budget / ratio;
    const int32_t rd = c->rotary_dim;

    float *qk = malloc((size_t)(nq + 1) * d * sizeof(float));
    qlin(qk, x, &L->indexer.qk_proj);
    float *q = qk;
    float *rawk = qk + (size_t)nq * d;
    memcpy(l->ikeys + (size_t)pos * d, rawk, (size_t)d * sizeof(float));

    float *qn = malloc((size_t)nq * d * sizeof(float));
    for (int32_t h = 0; h < nq; h++)
        rms(qn + (size_t)h * d, q + (size_t)h * d, L->indexer.q_norm, d, c->rms_norm_eps);
    int32_t p3[3] = { pos, pos, pos };
    qw_cpu_rope_partial(qn, p3, r->axis, r->inv_freq, 1, nq, d, rd);

    const int32_t visible = pos + 1;
    const int32_t n_blocks = visible / ratio;
    memset(allowed, 0, (size_t)visible);
    /* the tail: an incomplete trailing block is always attended */
    for (int32_t t = n_blocks * ratio; t < visible; t++) allowed[t] = 1;

    if (n_blocks > 0) {
        float *scores = malloc((size_t)n_blocks * sizeof(float));
        float *pooled = malloc((size_t)d * sizeof(float));
        float *kn = malloc((size_t)d * sizeof(float));
        for (int32_t bidx = 0; bidx < n_blocks; bidx++) {
            for (int32_t i = 0; i < d; i++) pooled[i] = 0.0f;
            for (int32_t t = 0; t < ratio; t++)
                for (int32_t i = 0; i < d; i++)
                    pooled[i] += l->ikeys[(size_t)(bidx * ratio + t) * d + i];
            for (int32_t i = 0; i < d; i++) pooled[i] /= (float)ratio;
            rms(kn, pooled, L->indexer.k_norm, d, c->rms_norm_eps);
            int32_t bp[3] = { bidx * ratio, bidx * ratio, bidx * ratio };
            qw_cpu_rope_partial(kn, bp, r->axis, r->inv_freq, 1, 1, d, rd);
            float sc = 0.0f;
            for (int32_t h = 0; h < nq; h++) {
                float dot = 0.0f;
                for (int32_t i = 0; i < d; i++) dot += qn[(size_t)h * d + i] * kn[i];
                if (dot > 0.0f) sc += dot;
            }
            scores[bidx] = sc / sqrtf((float)d);
        }
        const int32_t take = block_topk < n_blocks ? block_topk : n_blocks;
        for (int32_t n = 0; n < take; n++) {
            int32_t best = -1;
            for (int32_t bidx = 0; bidx < n_blocks; bidx++)
                if (scores[bidx] > -FLT_MAX && (best < 0 || scores[bidx] > scores[best])) best = bidx;
            if (best < 0) break;
            for (int32_t t = 0; t < ratio; t++) allowed[best * ratio + t] = 1;
            scores[best] = -FLT_MAX;
        }
        free(scores); free(pooled); free(kn);
    }
    free(qk); free(qn);
}

/* Gated attention with the QSA mask, for one token at `pos`. */
static void qsa_step(qw_flash_ref *r, const qw_layer *L, ref_layer *l,
                     const float *x, int32_t pos, float *out) {
    const qw_config *c = r->c;
    const qw_shape *sh = r->sh;
    const int32_t H = c->num_attention_heads, KVH = c->num_key_value_heads;
    const int32_t D = c->head_dim, gqa = H / KVH;

    float *qg = malloc((size_t)sh->q_proj_out * sizeof(float));
    float *k = malloc((size_t)sh->kv_dim * sizeof(float));
    float *v = malloc((size_t)sh->kv_dim * sizeof(float));
    float *q = malloc((size_t)sh->q_dim * sizeof(float));
    float *gate = malloc((size_t)sh->q_dim * sizeof(float));
    float *kn = malloc((size_t)sh->kv_dim * sizeof(float));
    float *att = malloc((size_t)sh->q_dim * sizeof(float));
    uint8_t *allowed = malloc((size_t)(pos + 1));

    qlin(qg, x, &L->q_proj);
    qlin(k, x, &L->k_proj);
    qlin(v, x, &L->v_proj);
    /* per head: [q(D) | gate(D)] */
    for (int32_t h = 0; h < H; h++) {
        rms(q + (size_t)h * D, qg + (size_t)h * 2 * D, L->q_norm, D, c->rms_norm_eps);
        memcpy(gate + (size_t)h * D, qg + (size_t)h * 2 * D + D, (size_t)D * sizeof(float));
    }
    for (int32_t h = 0; h < KVH; h++)
        rms(kn + (size_t)h * D, k + (size_t)h * D, L->k_norm, D, c->rms_norm_eps);
    int32_t p3[3] = { pos, pos, pos };
    qw_cpu_rope_partial(q,  p3, r->axis, r->inv_freq, 1, H,   D, c->rotary_dim);
    qw_cpu_rope_partial(kn, p3, r->axis, r->inv_freq, 1, KVH, D, c->rotary_dim);
    memcpy(l->kc + (size_t)pos * sh->kv_dim, kn, (size_t)sh->kv_dim * sizeof(float));
    memcpy(l->vc + (size_t)pos * sh->kv_dim, v,  (size_t)sh->kv_dim * sizeof(float));

    qsa_select(r, L, l, x, pos, allowed);

    const float scale = 1.0f / sqrtf((float)D);
    float *probs = malloc((size_t)(pos + 1) * sizeof(float));
    for (int32_t h = 0; h < H; h++) {
        const int32_t kvh = h / gqa;
        const float *qh = q + (size_t)h * D;
        float m = -FLT_MAX;
        for (int32_t t = 0; t <= pos; t++) {
            if (!allowed[t]) { probs[t] = -FLT_MAX; continue; }
            const float *kt = l->kc + (size_t)t * sh->kv_dim + (size_t)kvh * D;
            float s = 0.0f;
            for (int32_t i = 0; i < D; i++) s += qh[i] * kt[i];
            probs[t] = s * scale;
            if (probs[t] > m) m = probs[t];
        }
        float sum = 0.0f;
        for (int32_t t = 0; t <= pos; t++) {
            probs[t] = allowed[t] ? expf(probs[t] - m) : 0.0f;
            sum += probs[t];
        }
        float *o = att + (size_t)h * D;
        for (int32_t i = 0; i < D; i++) o[i] = 0.0f;
        for (int32_t t = 0; t <= pos; t++) {
            if (!allowed[t]) continue;
            const float *vt = l->vc + (size_t)t * sh->kv_dim + (size_t)kvh * D;
            const float p = probs[t] / sum;
            for (int32_t i = 0; i < D; i++) o[i] += p * vt[i];
        }
        for (int32_t i = 0; i < D; i++) o[i] *= sigm(gate[(size_t)h * D + i]);
    }
    qlin(out, att, &L->o_proj);

    free(qg); free(k); free(v); free(q); free(gate); free(kn); free(att); free(allowed); free(probs);
}

/* The MoE block: softmax router in fp32 over every expert, top-k
 * renormalised, the shared expert scaled by its own sigmoid gate. */
static void moe_step(qw_flash_ref *r, const qw_moe *M, const float *x, float *out) {
    const qw_config *c = r->c;
    const int32_t H = c->hidden_size, E = M->n_experts, K = c->num_experts_per_tok;
    const int32_t I = c->moe_intermediate_size, SI = c->shared_expert_intermediate_size;

    float *logits = malloc((size_t)E * sizeof(float));
    dmv(logits, x, M->router, H, E);
    float m = -FLT_MAX;
    for (int32_t e = 0; e < E; e++) if (logits[e] > m) m = logits[e];
    float sum = 0.0f;
    for (int32_t e = 0; e < E; e++) { logits[e] = expf(logits[e] - m); sum += logits[e]; }
    for (int32_t e = 0; e < E; e++) logits[e] /= sum;

    int32_t idx[64];
    float   w[64];
    float wsum = 0.0f;
    for (int32_t n = 0; n < K; n++) {
        int32_t best = -1;
        for (int32_t e = 0; e < E; e++)
            if (logits[e] >= 0.0f && (best < 0 || logits[e] > logits[best])) best = e;
        idx[n] = best; w[n] = logits[best]; wsum += w[n];
        logits[best] = -1.0f;
    }
    if (c->norm_topk_prob) for (int32_t n = 0; n < K; n++) w[n] /= wsum;

    for (int32_t i = 0; i < H; i++) out[i] = 0.0f;
    float *gu = malloc((size_t)2 * I * sizeof(float));
    float *act = malloc((size_t)I * sizeof(float));
    float *y = malloc((size_t)H * sizeof(float));
    for (int32_t n = 0; n < K; n++) {
        const int64_t e = idx[n];
        qmv(gu, x, &M->gate_up, e * (int64_t)(2 * I), 2 * I);
        for (int32_t i = 0; i < I; i++) act[i] = silu(gu[i]) * gu[I + i];
        qmv(y, act, &M->down, e * (int64_t)H, H);
        for (int32_t i = 0; i < H; i++) out[i] += w[n] * y[i];
    }

    /* the shared expert */
    float *sg = malloc((size_t)SI * sizeof(float));
    float *su = malloc((size_t)SI * sizeof(float));
    qlin(sg, x, &M->sh_gate);
    qlin(su, x, &M->sh_up);
    for (int32_t i = 0; i < SI; i++) sg[i] = silu(sg[i]) * su[i];
    qlin(y, sg, &M->sh_down);
    float g1;
    dmv(&g1, x, M->sh_gate_w, H, 1);
    const float gs = sigm(g1);
    for (int32_t i = 0; i < H; i++) out[i] += gs * y[i];

    free(logits); free(gu); free(act); free(y); free(sg); free(su);
}

/* The engram layer for one token.  `tok` is the current token, already
 * pushed onto the history; `pos` its position. */
static void ple_step(qw_flash_ref *r, const qw_ple *P, const float *h4, float *out) {
    const qw_config *c = r->c;
    const int32_t H = c->hidden_size, S = c->hc_count, HH = r->sh->hc_hidden;
    const int32_t E = c->ple_embed_dim, NH = P->n_heads, HD = P->head_dim;
    const int32_t per = c->heads_per_ngram;
    const int32_t pos = r->n_past;                 /* current position */
    const int32_t eos = c->eos_token_ids[0];

    /* shifted tokens: the token s back within the EOS-delimited segment */
    const int32_t in_seg = pos - 1 - r->last_eos;  /* position within segment */
    int64_t shifted[8];
    for (int32_t s = 0; s < c->ngram_size; s++)
        shifted[s] = (in_seg >= s) ? r->hist[s] : eos;

    /* hashed ids, one per head; heads for n-gram order n are contiguous */
    float *emb = malloc((size_t)E * sizeof(float));
    const uint16_t *table = (const uint16_t *)tdata(P->table);
    for (int32_t ng = 2; ng <= c->ngram_size; ng++) {
        int64_t mixed = (int64_t)((uint64_t)shifted[0] * (uint64_t)P->mult[0]);
        for (int32_t p = 1; p < ng; p++)
            mixed ^= (int64_t)((uint64_t)shifted[p] * (uint64_t)P->mult[p]);
        for (int32_t k = 0; k < per; k++) {
            const int32_t h = (ng - 2) * per + k;
            const int64_t id = pymod(mixed, P->head_size[h]) + P->head_off[h];
            for (int32_t i = 0; i < HD; i++)
                emb[(size_t)h * HD + i] = qw_bf16_to_f32_c(table[(size_t)id * HD + i]);
        }
    }
    (void)NH;

    float *key = malloc((size_t)HH * sizeof(float));
    float *keyn = malloc((size_t)HH * sizeof(float));
    float *value = malloc((size_t)H * sizeof(float));
    float *qn = malloc((size_t)HH * sizeof(float));
    float *gv = malloc((size_t)HH * sizeof(float));
    float *gvn = malloc((size_t)HH * sizeof(float));
    qlin(key, emb, &P->key_proj);
    grouped_rms(keyn, key, P->norm_key, S, H, c->rms_norm_eps);
    qlin(value, emb, &P->value_proj);
    grouped_rms(qn, h4, P->norm_query, S, H, c->rms_norm_eps);
    for (int32_t s = 0; s < S; s++) {
        float g = 0.0f;
        for (int32_t i = 0; i < H; i++) g += keyn[(size_t)s * H + i] * qn[(size_t)s * H + i];
        g /= sqrtf((float)H);
        float mag = fabsf(g); if (mag < 1e-6f) mag = 1e-6f;
        g = (g < 0 ? -1.0f : 1.0f) * sqrtf(mag);
        const float gs = sigm(g);
        for (int32_t i = 0; i < H; i++) gv[(size_t)s * H + i] = gs * value[i];
    }
    grouped_rms(gvn, gv, P->norm_conv, S, H, c->rms_norm_eps);

    /* dilated depthwise conv over time: tap j reads t - (K-1-j)*dilation.
     * The state holds the last state_len inputs, oldest first. */
    const int32_t K = c->ple_conv_kernel_size, dil = c->ngram_size, SL = r->ple_state_len;
    const uint16_t *cw = (const uint16_t *)tdata(P->conv1d);
    for (int32_t ch = 0; ch < HH; ch++) {
        float acc = 0.0f;
        for (int32_t j = 0; j < K; j++) {
            const int32_t back = (K - 1 - j) * dil;      /* 0 is the current input */
            const float xin = back == 0 ? gvn[ch]
                            : r->ple_conv_state[(size_t)(SL - back) * HH + ch];
            acc += xin * qw_bf16_to_f32_c(cw[(size_t)ch * K + j]);
        }
        out[ch] = gv[ch] + silu(acc);
    }
    /* shift the window and append this input */
    memmove(r->ple_conv_state, r->ple_conv_state + HH, (size_t)(SL - 1) * HH * sizeof(float));
    memcpy(r->ple_conv_state + (size_t)(SL - 1) * HH, gvn, (size_t)HH * sizeof(float));

    free(emb); free(key); free(keyn); free(value); free(qn); free(gv); free(gvn);
}

/* ---- the forward ----------------------------------------------------------- */

bool qw_flash_ref_forward(qw_flash_ref *r, const int32_t *tokens, int32_t n,
                          float *logits, float *hidden_last, char *err, size_t errcap) {
    const qw_config *c = r->c;
    const qw_shape *sh = r->sh;
    const qwasar_engine *e = r->e;
    const int32_t H = c->hidden_size, HH = sh->hc_hidden;
    if (r->n_past + n > r->max_ctx) {
        qw_verrf(err, errcap, "reference: %d tokens exceed the %d context", r->n_past + n, r->max_ctx);
        return false;
    }

    float *x   = malloc((size_t)H * sizeof(float));
    float *h4  = malloc((size_t)HH * sizeof(float));
    float *nrm = malloc((size_t)HH * sizeof(float));
    float *out = malloc((size_t)HH * sizeof(float));
    float inj[16];

    const qw_qlinear *emb = qwasar_engine_embed(e);
    for (int32_t t = 0; t < n; t++) {
        const int32_t tok = tokens[t];
        const int32_t pos = r->n_past;

        /* the token joins the history before the engram reads it */
        for (int i = 7; i > 0; i--) r->hist[i] = r->hist[i - 1];
        r->hist[0] = tok;

        qw_cpu_embed_q4(x, &tok, (const uint32_t *)tdata(emb->weight),
                        (const uint16_t *)tdata(emb->scales),
                        (const uint16_t *)tdata(emb->biases), H, 1);
        for (int32_t s = 0; s < c->hc_count; s++)
            memcpy(h4 + (size_t)s * H, x, (size_t)H * sizeof(float));

        for (int32_t i = 0; i < c->num_hidden_layers; i++) {
            const qw_layer *L = qwasar_engine_layer(e, i);
            ref_layer *l = &r->L[i];

            if (L->ple) {
                ple_step(r, L->ple, h4, out);
                for (int32_t j = 0; j < HH; j++) h4[j] += out[j];
            }

            hc_mix(r, &L->attn_hc, h4, nrm, x, inj);
            if (L->is_linear_attn) gdn_step(r, L, l, x, out);
            else                   qsa_step(r, L, l, x, pos, out);
            hc_inject(r, h4, out, inj);

            hc_mix(r, &L->mlp_hc, h4, nrm, x, inj);
            moe_step(r, &L->moe, x, out);
            hc_inject(r, h4, out, inj);

            if (hidden_last && t == n - 1)
                memcpy(hidden_last + (size_t)i * HH, h4, (size_t)HH * sizeof(float));
        }

        /* the final mixer, then lm_head -- no norm in between */
        hc_mix(r, qwasar_engine_final_hc(e), h4, nrm, x, NULL);
        qlin(logits + (size_t)t * c->vocab_size, x, qwasar_engine_head(e));

        /* the EOS bookkeeping for the NEXT token's engram context */
        for (int32_t k = 0; k < c->n_eos; k++)
            if (tok == c->eos_token_ids[k]) { r->last_eos = pos; break; }
        r->n_past++;
    }

    free(x); free(h4); free(nrm); free(out);
    return true;
}
