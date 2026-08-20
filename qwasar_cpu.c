/* Scalar fp32 reference implementations of the Metal kernels.
 *
 * These exist so every kernel has something to be wrong against.  They are
 * deliberately literal transcriptions of the format documentation -- no
 * cleverness, no vectorisation -- because their only job is to be obviously
 * correct.  Nothing on an inference path calls them. */

#include "qwasar_model.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

float qw_bf16_to_f32_c(uint16_t v) {
    /* bfloat16 is the top 16 bits of an IEEE float; widening is a shift. */
    union { uint32_t u; float f; } c = { .u = (uint32_t)v << 16 };
    return c.f;
}

void qw_cpu_dequant_row(float *out, const uint32_t *w, const uint16_t *scales,
                        const uint16_t *biases, int32_t k, int32_t row) {
    const int32_t words  = k / 8;
    const int32_t groups = k / 64;
    const uint32_t *wr = w + (size_t)row * words;
    const uint16_t *sr = scales + (size_t)row * groups;
    const uint16_t *br = biases + (size_t)row * groups;

    for (int32_t i = 0; i < k; i++) {
        /* Element i lives in word i/8 at bit 4*(i%8), and takes its scale and
         * bias from group i/64.  The nibble is unsigned; the zero point is
         * folded into the bias, so there is nothing to subtract. */
        uint32_t nib = (wr[i / 8] >> (4 * (i % 8))) & 0xF;
        float sc = qw_bf16_to_f32_c(sr[i / 64]);
        float bi = qw_bf16_to_f32_c(br[i / 64]);
        out[i] = sc * (float)nib + bi;
    }
}

void qw_cpu_qmv_q4(float *y, const float *x, const uint32_t *w,
                   const uint16_t *scales, const uint16_t *biases,
                   int32_t k, int32_t n, int32_t rows) {
    const int32_t words  = k / 8;
    const int32_t groups = k / 64;

    for (int32_t r = 0; r < rows; r++) {
        const float *xv = x + (size_t)r * k;
        float *yv = y + (size_t)r * n;
        for (int32_t o = 0; o < n; o++) {
            const uint32_t *wr = w + (size_t)o * words;
            const uint16_t *sr = scales + (size_t)o * groups;
            const uint16_t *br = biases + (size_t)o * groups;
            float acc = 0.0f;
            for (int32_t i = 0; i < k; i++) {
                uint32_t nib = (wr[i / 8] >> (4 * (i % 8))) & 0xF;
                float sc = qw_bf16_to_f32_c(sr[i / 64]);
                float bi = qw_bf16_to_f32_c(br[i / 64]);
                acc += (sc * (float)nib + bi) * xv[i];
            }
            yv[o] = acc;
        }
    }
}

/* silu(x) = x * sigmoid(x), in fp32 to match the kernel. */
static float qw_silu_c(float x) { return x / (1.0f + expf(-x)); }

void qw_cpu_rms_norm(float *y, const float *x, const uint16_t *w,
                     int32_t dim, int32_t rows, float eps, float out_scale) {
    for (int32_t r = 0; r < rows; r++) {
        const float *xr = x + (size_t)r * dim;
        float *yr = y + (size_t)r * dim;
        float ss = 0.0f;
        for (int32_t i = 0; i < dim; i++) ss += xr[i] * xr[i];
        float inv = 1.0f / sqrtf(ss / (float)dim + eps) * out_scale;
        for (int32_t i = 0; i < dim; i++) {
            float v = xr[i] * inv;
            if (w) v *= qw_bf16_to_f32_c(w[i]);
            yr[i] = v;
        }
    }
}

void qw_cpu_rms_norm_gated(float *y, const float *x, const uint16_t *w,
                           const float *gate, int32_t dim, int32_t rows,
                           float eps, float out_scale) {
    for (int32_t r = 0; r < rows; r++) {
        const float *xr = x + (size_t)r * dim;
        const float *gr = gate + (size_t)r * dim;
        float *yr = y + (size_t)r * dim;
        float ss = 0.0f;
        for (int32_t i = 0; i < dim; i++) ss += xr[i] * xr[i];
        float inv = 1.0f / sqrtf(ss / (float)dim + eps) * out_scale;
        for (int32_t i = 0; i < dim; i++) {
            float v = xr[i] * inv;
            if (w) v *= qw_bf16_to_f32_c(w[i]);
            /* silu is applied to the gate projection, not to the normalised
             * value -- see metal/norm.metal. */
            yr[i] = v * qw_silu_c(gr[i]);
        }
    }
}

void qw_cpu_swiglu(float *y, const float *gate, const float *up, int32_t n) {
    for (int32_t i = 0; i < n; i++) y[i] = qw_silu_c(gate[i]) * up[i];
}

static float qw_sigmoid_c(float x) { return 1.0f / (1.0f + expf(-x)); }
static float qw_softplus_c(float x) { return x > 20.0f ? x : log1pf(expf(x)); }

void qw_cpu_conv1d_causal_silu(float *y, const float *x, float *state,
                               const uint16_t *w, int32_t channels, int32_t rows,
                               int32_t ksize) {
    for (int32_t ch = 0; ch < channels; ch++) {
        /* win[0] is the oldest sample; the cached state holds the same order. */
        float win[8];
        for (int32_t j = 0; j < ksize - 1; j++)
            win[j] = state[(size_t)j * channels + ch];

        for (int32_t t = 0; t < rows; t++) {
            win[ksize - 1] = x[(size_t)t * channels + ch];
            float acc = 0.0f;
            for (int32_t j = 0; j < ksize; j++)
                acc += win[j] * qw_bf16_to_f32_c(w[(size_t)ch * ksize + j]);
            y[(size_t)t * channels + ch] = qw_silu_c(acc);
            for (int32_t j = 0; j < ksize - 1; j++) win[j] = win[j + 1];
        }
        for (int32_t j = 0; j < ksize - 1; j++)
            state[(size_t)j * channels + ch] = win[j];
    }
}

void qw_cpu_gdn_gates(float *g, float *beta, const float *a, const float *b,
                      const uint16_t *A_log, const uint16_t *dt_bias,
                      int32_t hv, int32_t rows) {
    for (int32_t i = 0; i < rows * hv; i++) {
        int32_t h = i % hv;
        float A  = qw_bf16_to_f32_c(A_log[h]);
        float dt = qw_bf16_to_f32_c(dt_bias[h]);
        g[i]    = expf(-expf(A) * qw_softplus_c(a[i] + dt));
        beta[i] = qw_sigmoid_c(b[i]);
    }
}

void qw_cpu_gated_delta(float *y, const float *q, const float *k, const float *v,
                        const float *g, const float *beta, float *state,
                        int32_t hk, int32_t hv, int32_t dk, int32_t dv, int32_t rows) {
    const int32_t gqa = hv / hk;
    for (int32_t t = 0; t < rows; t++) {
        for (int32_t h = 0; h < hv; h++) {
            const int32_t hkx = h / gqa;
            const float *qt = q + ((size_t)t * hk + hkx) * dk;
            const float *kt = k + ((size_t)t * hk + hkx) * dk;
            const float *vt = v + ((size_t)t * hv + h) * dv;
            float *yt = y + ((size_t)t * hv + h) * dv;
            float *S = state + (size_t)h * dv * dk;
            const float gt = g[(size_t)t * hv + h];
            const float bt = beta[(size_t)t * hv + h];

            for (int32_t r = 0; r < dv; r++) {
                float *Sr = S + (size_t)r * dk;
                /* decay, then read what the state already predicts for v[r] */
                float kv = 0.0f;
                for (int32_t c = 0; c < dk; c++) { Sr[c] *= gt; kv += Sr[c] * kt[c]; }
                /* delta rule: write only the residual, scaled by beta */
                float delta = (vt[r] - kv) * bt;
                float out = 0.0f;
                for (int32_t c = 0; c < dk; c++) {
                    Sr[c] += kt[c] * delta;
                    out += Sr[c] * qt[c];
                }
                yt[r] = out;
            }
        }
    }
}

/* Interleaved MRoPE frequency-to-axis assignment.
 *
 * Frequency j reads position axis `axis[j]`, cycling t, h, w so that the counts
 * land on mrope_section = [11, 11, 10] for a 64-dim rotation.  The reference
 * builds this by striding each non-time axis by 3 from its own offset and
 * stopping at section*3, which leaves the tail slots on the time axis. */
void qw_rope_tables(uint8_t *axis, float *inv_freq, int32_t rotary_dim,
                    float theta, const int32_t mrope_section[3]) {
    const int32_t half = rotary_dim / 2;
    for (int32_t j = 0; j < half; j++) {
        axis[j] = 0;
        inv_freq[j] = 1.0f / powf(theta, (float)(2 * j) / (float)rotary_dim);
    }
    for (int32_t d = 1; d <= 2; d++) {
        int32_t limit = mrope_section[d] * 3;
        if (limit > half) limit = half;
        for (int32_t idx = d; idx < limit; idx += 3) axis[idx] = (uint8_t)d;
    }
}

void qw_cpu_rope_partial(float *x, const int32_t *pos, const uint8_t *axis,
                         const float *inv_freq, int32_t rows, int32_t heads,
                         int32_t head_dim, int32_t rotary_dim) {
    const int32_t half = rotary_dim / 2;
    for (int32_t r = 0; r < rows; r++)
        for (int32_t h = 0; h < heads; h++) {
            float *xv = x + ((size_t)r * heads + h) * head_dim;
            for (int32_t j = 0; j < half; j++) {
                float angle = (float)pos[(size_t)axis[j] * rows + r] * inv_freq[j];
                float c = cosf(angle), s = sinf(angle);
                float x0 = xv[j], x1 = xv[j + half];
                xv[j]        = x0 * c - x1 * s;
                xv[j + half] = x1 * c + x0 * s;
            }
        }
}

void qw_cpu_embed_q4(float *y, const int32_t *tokens, const uint32_t *w,
                     const uint16_t *scales, const uint16_t *biases,
                     int32_t hidden, int32_t n_tokens) {
    for (int32_t t = 0; t < n_tokens; t++)
        qw_cpu_dequant_row(y + (size_t)t * hidden, w, scales, biases, hidden, tokens[t]);
}

/* The KV cache is fp16, so the reference must read exactly the values the
 * kernel reads -- comparing against unrounded fp32 would measure the cache
 * dtype rather than the kernel. */
float qw_f16_to_f32_c(uint16_t v) {
    __fp16 h;
    memcpy(&h, &v, sizeof h);
    return (float)h;
}

uint16_t qw_f32_to_f16_c(float v) {
    __fp16 h = (__fp16)v;
    uint16_t u;
    memcpy(&u, &h, sizeof u);
    return u;
}

void qw_cpu_kv_write(uint16_t *kc, uint16_t *vc, const float *k, const float *v,
                     int32_t rows, int32_t kv_heads, int32_t head_dim,
                     int32_t max_ctx, int32_t base_pos) {
    for (int32_t r = 0; r < rows; r++)
        for (int32_t h = 0; h < kv_heads; h++)
            for (int32_t i = 0; i < head_dim; i++) {
                size_t src = ((size_t)r * kv_heads + h) * head_dim + i;
                size_t dst = ((size_t)h * max_ctx + base_pos + r) * head_dim + i;
                kc[dst] = qw_f32_to_f16_c(k[src]);
                vc[dst] = qw_f32_to_f16_c(v[src]);
            }
}

void qw_cpu_attn_decode(float *out, const float *q, const uint16_t *kc,
                        const uint16_t *vc, int32_t rows, int32_t q_heads,
                        int32_t kv_heads, int32_t head_dim, int32_t max_ctx,
                        int32_t base_pos, float scale) {
    const int32_t gqa = q_heads / kv_heads;
    float *probs = malloc((size_t)(base_pos + rows) * sizeof(float));

    for (int32_t r = 0; r < rows; r++)
        for (int32_t h = 0; h < q_heads; h++) {
            const int32_t kvh = h / gqa;
            const int32_t n = base_pos + r + 1;      /* causal */
            const float *qv = q + ((size_t)r * q_heads + h) * head_dim;
            float *ov = out + ((size_t)r * q_heads + h) * head_dim;

            float m = -FLT_MAX;   /* see metal/attn.metal: -ffast-math forbids -INFINITY */
            for (int32_t t = 0; t < n; t++) {
                const uint16_t *kv = kc + ((size_t)kvh * max_ctx + t) * head_dim;
                float s = 0.0f;
                for (int32_t i = 0; i < head_dim; i++)
                    s += qv[i] * scale * qw_f16_to_f32_c(kv[i]);
                probs[t] = s;
                if (s > m) m = s;
            }
            float sum = 0.0f;
            for (int32_t t = 0; t < n; t++) { probs[t] = expf(probs[t] - m); sum += probs[t]; }

            for (int32_t i = 0; i < head_dim; i++) ov[i] = 0.0f;
            for (int32_t t = 0; t < n; t++) {
                const uint16_t *vv = vc + ((size_t)kvh * max_ctx + t) * head_dim;
                float p = probs[t] / sum;
                for (int32_t i = 0; i < head_dim; i++) ov[i] += p * qw_f16_to_f32_c(vv[i]);
            }
        }
    free(probs);
}
