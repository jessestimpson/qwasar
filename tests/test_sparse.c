/* Flash-Next's kernels (metal/sparse.metal) against scalar twins.
 *
 * Each op on random inputs -- and on the toy checkpoint's real tensors where
 * a weight layout is the thing to get wrong (expert banks, norm weights) --
 * compared to a scalar reimplementation of the same semantics as
 * qwasar_flash_cpu.c.  tests/test_flashnext holds the whole forward to the
 * oracle; this is where a kernel that is slightly wrong gets named. */

#include "qwasar.h"
#include "qwasar_gpu.h"
#include "qwasar_model.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;

#define CHECK(cond, ...) do { \
    if (!(cond)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
                   fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); fails++; } \
} while (0)

static void fill_random(float *v, size_t n, uint32_t seed) {
    uint32_t s = seed | 1u;
    for (size_t i = 0; i < n; i++) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        v[i] = (float)((int32_t)(s >> 8) - 8388608) / 8388608.0f;
    }
}

static double rel_l2(const float *a, const float *b, size_t n) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = (double)a[i] - (double)b[i];
        num += d * d;
        den += (double)b[i] * (double)b[i];
    }
    return den > 0.0 ? sqrt(num / den) : sqrt(num);
}

static float sigm(float x) { return 1.0f / (1.0f + expf(-x)); }
static float silu(float x) { return x * sigm(x); }
static uint16_t f2bf(float f) { union { float f; uint32_t u; } c = { f }; return (uint16_t)((c.u + 0x7FFF + ((c.u >> 16) & 1)) >> 16); }

static qw_buf mkbuf(size_t bytes) { qw_buf b = qw_buf_alloc(bytes); memset(qw_buf_contents(b), 0, bytes); return b; }

static void run(qw_cmd c, const char *label) {
    qw_cmd_wait(c);
    CHECK(qw_cmd_error(c) == NULL, "%s: %s", label, qw_cmd_error(c));
    qw_cmd_free(c);
}

static void report(const char *label, const float *got, const float *ref, size_t n, double tol) {
    double e = rel_l2(got, ref, n);
    printf("  %-28s rel L2 %.2e\n", label, e);
    CHECK(e < tol, "%s: rel L2 %.3g exceeds %.1g", label, e, tol);
}

/* ---- hyper-connections ---------------------------------------------------- */

static void test_hc(void) {
    const int32_t rows = 5, H = 64, S = 4, HH = S * H, R = 64;
    qw_buf h4 = mkbuf((size_t)rows * HH * 4), n4 = mkbuf((size_t)rows * HH * 4), x = mkbuf((size_t)rows * H * 4);
    qw_buf w = mkbuf((size_t)HH * 2), m = mkbuf((size_t)rows * HH * 4), inj = mkbuf((size_t)rows * S * 4);
    qw_buf out = mkbuf((size_t)rows * H * 4), d = mkbuf((size_t)rows * R * 4);
    float *h4p = qw_buf_contents(h4), *mp = qw_buf_contents(m), *injp = qw_buf_contents(inj);
    float *outp = qw_buf_contents(out), *dp = qw_buf_contents(d), *xp = qw_buf_contents(x);
    uint16_t *wp = qw_buf_contents(w);
    fill_random(h4p, (size_t)rows * HH, 11); fill_random(mp, (size_t)rows * HH, 12);
    fill_random(injp, (size_t)rows * S, 13); fill_random(outp, (size_t)rows * H, 14);
    fill_random(dp, (size_t)rows * R, 15); fill_random(xp, (size_t)rows * H, 16);
    for (int32_t i = 0; i < HH; i++) { float t; fill_random(&t, 1, 100 + i); wp[i] = f2bf(1.0f + 0.1f * t); }

    /* repeat_cols */
    qw_buf rep = mkbuf((size_t)rows * HH * 4);
    qw_cmd c = qw_cmd_begin();
    qw_op_repeat_cols(c, qw_ref_at(rep, 0), qw_ref_at(x, 0), rows, H, S);
    run(c, "repeat_cols");
    float *ref = malloc((size_t)rows * HH * 4);
    for (int32_t r = 0; r < rows; r++) for (int32_t s = 0; s < S; s++) for (int32_t i = 0; i < H; i++)
        ref[(size_t)r * HH + s * H + i] = xp[(size_t)r * H + i];
    report("repeat_cols", qw_buf_contents(rep), ref, (size_t)rows * HH, 1e-6);

    /* grouped norm */
    c = qw_cmd_begin();
    qw_op_rms_norm_grouped(c, qw_ref_at(n4, 0), qw_ref_at(h4, 0), qw_ref_at(w, 0), H, S, rows, 1e-6f);
    run(c, "rms_norm_grouped");
    for (int32_t r = 0; r < rows; r++) for (int32_t s = 0; s < S; s++) {
        const float *xg = h4p + (size_t)r * HH + s * H;
        float ss = 0.0f; for (int32_t i = 0; i < H; i++) ss += xg[i] * xg[i];
        const float inv = 1.0f / sqrtf(ss / H + 1e-6f);
        for (int32_t i = 0; i < H; i++) ref[(size_t)r * HH + s * H + i] = xg[i] * inv * qw_bf16_to_f32_c(wp[s * H + i]);
    }
    report("rms_norm_grouped", qw_buf_contents(n4), ref, (size_t)rows * HH, 1e-5);

    /* silu_scale */
    float *dref = malloc((size_t)rows * R * 4);
    for (size_t i = 0; i < (size_t)rows * R; i++) dref[i] = silu(dp[i] * 0.25f);
    c = qw_cmd_begin();
    qw_op_silu_scale(c, qw_ref_at(d, 0), rows * R, 0.25f);
    run(c, "silu_scale");
    report("silu_scale", dp, dref, (size_t)rows * R, 1e-6);

    /* hc_mix: x = mean_s sigmoid(m) * n4 */
    qw_buf mix = mkbuf((size_t)rows * H * 4);
    c = qw_cmd_begin();
    qw_op_hc_mix(c, qw_ref_at(mix, 0), qw_ref_at(n4, 0), qw_ref_at(m, 0), rows, H, S);
    run(c, "hc_mix");
    const float *n4p = qw_buf_contents(n4);
    float *mref = malloc((size_t)rows * H * 4);
    for (int32_t r = 0; r < rows; r++) for (int32_t i = 0; i < H; i++) {
        float acc = 0.0f;
        for (int32_t s = 0; s < S; s++) acc += sigm(mp[(size_t)r * HH + s * H + i]) * n4p[(size_t)r * HH + s * H + i];
        mref[(size_t)r * H + i] = acc / S;
    }
    report("hc_mix", qw_buf_contents(mix), mref, (size_t)rows * H, 1e-5);

    /* hc_inject */
    for (size_t i = 0; i < (size_t)rows * HH; i++) ref[i] = h4p[i];
    for (int32_t r = 0; r < rows; r++) for (int32_t s = 0; s < S; s++) {
        const float wgt = 2.0f * sigm(injp[(size_t)r * S + s] / S);
        for (int32_t i = 0; i < H; i++) ref[(size_t)r * HH + s * H + i] += outp[(size_t)r * H + i] * wgt;
    }
    c = qw_cmd_begin();
    qw_op_hc_inject(c, qw_ref_at(h4, 0), qw_ref_at(out, 0), qw_ref_at(inj, 0), rows, H, S);
    run(c, "hc_inject");
    report("hc_inject", h4p, ref, (size_t)rows * HH, 1e-5);

    /* ple_gate: per stream, sigmoid(sign-sqrt(key.q / sqrt(H))) * value */
    qw_buf gv = mkbuf((size_t)rows * HH * 4);
    c = qw_cmd_begin();
    qw_op_ple_gate(c, qw_ref_at(gv, 0), qw_ref_at(n4, 0), qw_ref_at(m, 0), qw_ref_at(x, 0), rows, H, S);
    run(c, "ple_gate");
    for (int32_t r = 0; r < rows; r++) for (int32_t s = 0; s < S; s++) {
        float g = 0.0f;
        for (int32_t i = 0; i < H; i++) g += n4p[(size_t)r * HH + s * H + i] * mp[(size_t)r * HH + s * H + i];
        g /= sqrtf((float)H);
        float mag = fabsf(g); if (mag < 1e-6f) mag = 1e-6f;
        g = (g < 0 ? -1.0f : 1.0f) * sqrtf(mag);
        for (int32_t i = 0; i < H; i++) ref[(size_t)r * HH + s * H + i] = sigm(g) * xp[(size_t)r * H + i];
    }
    report("ple_gate", qw_buf_contents(gv), ref, (size_t)rows * HH, 1e-5);

    free(ref); free(dref); free(mref);
    qw_buf_free(h4); qw_buf_free(n4); qw_buf_free(x); qw_buf_free(w); qw_buf_free(m); qw_buf_free(inj);
    qw_buf_free(out); qw_buf_free(d); qw_buf_free(rep); qw_buf_free(mix); qw_buf_free(gv);
}

/* ---- MoE -------------------------------------------------------------------- */

static void test_moe(qwasar_engine *e) {
    const qw_config *cfg = qwasar_engine_config(e);
    const qw_moe *M = &qwasar_engine_layer(e, 0)->moe;
    const int32_t rows = 7, E = M->n_experts, K = cfg->num_experts_per_tok, H = cfg->hidden_size;
    const int32_t I = cfg->moe_intermediate_size, pairs = rows * K;

    /* routing on random logits with a clear margin */
    qw_buf lg = mkbuf((size_t)rows * E * 4), idx = mkbuf((size_t)pairs * 4), w = mkbuf((size_t)pairs * 4);
    float *lgp = qw_buf_contents(lg);
    fill_random(lgp, (size_t)rows * E, 21);
    for (size_t i = 0; i < (size_t)rows * E; i++) lgp[i] *= 4.0f;
    qw_cmd c = qw_cmd_begin();
    qw_op_moe_route(c, qw_ref_at(idx, 0), qw_ref_at(w, 0), qw_ref_at(lg, 0), rows, E, K, true);
    run(c, "moe_route");
    const int32_t *idxp = qw_buf_contents(idx);
    const float *wp = qw_buf_contents(w);
    int32_t route_bad = 0;
    float *wref = malloc((size_t)pairs * 4);
    for (int32_t r = 0; r < rows; r++) {
        float m = -FLT_MAX, sum = 0.0f;
        for (int32_t k = 0; k < E; k++) if (lgp[r * E + k] > m) m = lgp[r * E + k];
        for (int32_t k = 0; k < E; k++) sum += expf(lgp[r * E + k] - m);
        float *p = malloc((size_t)E * 4);
        for (int32_t k = 0; k < E; k++) p[k] = expf(lgp[r * E + k] - m) / sum;
        float ws = 0.0f;
        for (int32_t n = 0; n < K; n++) {
            int32_t best = -1;
            for (int32_t k = 0; k < E; k++) if (p[k] >= 0 && (best < 0 || p[k] > p[best])) best = k;
            if (idxp[r * K + n] != best) route_bad++;
            wref[r * K + n] = p[best]; ws += p[best];
            p[best] = -1.0f;
        }
        for (int32_t n = 0; n < K; n++) wref[r * K + n] /= ws;
        free(p);
    }
    CHECK(route_bad == 0, "moe_route: %d index mismatches", route_bad);
    printf("  %-28s index mismatches %d\n", "moe_route (idx)", route_bad);
    report("moe_route (weights)", wp, wref, (size_t)pairs, 1e-5);

    /* the gate|up bank, then swiglu_split, then the down bank, then combine */
    qw_buf x = mkbuf((size_t)rows * H * 4), gu = mkbuf((size_t)pairs * 2 * I * 4);
    qw_buf act = mkbuf((size_t)pairs * I * 4), y = mkbuf((size_t)pairs * H * 4), out = mkbuf((size_t)rows * H * 4);
    float *xp = qw_buf_contents(x);
    fill_random(xp, (size_t)rows * H, 22);
    c = qw_cmd_begin();
    qw_op_qmv_q4_bank(c, qw_ref_at(gu, 0), qw_ref_at(x, 0), qw_ref_at(idx, 0),
                      qw_tensor_ref(M->gate_up.weight), qw_tensor_ref(M->gate_up.scales),
                      qw_tensor_ref(M->gate_up.biases), H, 2 * I, pairs, K, false);
    run(c, "qmv_q4_bank (gate_up)");
    float *guref = malloc((size_t)pairs * 2 * I * 4);
    const uint32_t *bw = qw_tensor_data(M->gate_up.weight);
    const uint16_t *bs = qw_tensor_data(M->gate_up.scales), *bb = qw_tensor_data(M->gate_up.biases);
    for (int32_t p = 0; p < pairs; p++) {
        const int32_t ex = idxp[p];
        qw_cpu_qmv_q4(guref + (size_t)p * 2 * I, xp + (size_t)(p / K) * H,
                      bw + (size_t)ex * 2 * I * (H / 8), bs + (size_t)ex * 2 * I * (H / 64),
                      bb + (size_t)ex * 2 * I * (H / 64), H, 2 * I, 1);
    }
    report("qmv_q4_bank (gate_up)", qw_buf_contents(gu), guref, (size_t)pairs * 2 * I, 1e-4);

    c = qw_cmd_begin();
    qw_op_swiglu_split(c, qw_ref_at(act, 0), qw_ref_at(gu, 0), pairs, I);
    run(c, "swiglu_split");
    const float *gup = qw_buf_contents(gu);
    float *actref = malloc((size_t)pairs * I * 4);
    for (int32_t p = 0; p < pairs; p++) for (int32_t i = 0; i < I; i++)
        actref[(size_t)p * I + i] = silu(gup[(size_t)p * 2 * I + i]) * gup[(size_t)p * 2 * I + I + i];
    report("swiglu_split", qw_buf_contents(act), actref, (size_t)pairs * I, 1e-5);

    c = qw_cmd_begin();
    qw_op_qmv_q4_bank(c, qw_ref_at(y, 0), qw_ref_at(act, 0), qw_ref_at(idx, 0),
                      qw_tensor_ref(M->down.weight), qw_tensor_ref(M->down.scales),
                      qw_tensor_ref(M->down.biases), I, H, pairs, K, true);
    run(c, "qmv_q4_bank (down)");
    const float *actp = qw_buf_contents(act);
    float *yref = malloc((size_t)pairs * H * 4);
    const uint32_t *dw = qw_tensor_data(M->down.weight);
    const uint16_t *ds = qw_tensor_data(M->down.scales), *db = qw_tensor_data(M->down.biases);
    for (int32_t p = 0; p < pairs; p++) {
        const int32_t ex = idxp[p];
        qw_cpu_qmv_q4(yref + (size_t)p * H, actp + (size_t)p * I,
                      dw + (size_t)ex * H * (I / 8), ds + (size_t)ex * H * (I / 64),
                      db + (size_t)ex * H * (I / 64), I, H, 1);
    }
    report("qmv_q4_bank (down)", qw_buf_contents(y), yref, (size_t)pairs * H, 1e-4);

    c = qw_cmd_begin();
    qw_op_moe_combine(c, qw_ref_at(out, 0), qw_ref_at(y, 0), qw_ref_at(w, 0), rows, K, H);
    run(c, "moe_combine");
    const float *yp = qw_buf_contents(y);
    float *oref = malloc((size_t)rows * H * 4);
    for (int32_t r = 0; r < rows; r++) for (int32_t i = 0; i < H; i++) {
        float acc = 0.0f;
        for (int32_t k = 0; k < K; k++) acc += wp[r * K + k] * yp[((size_t)r * K + k) * H + i];
        oref[(size_t)r * H + i] = acc;
    }
    report("moe_combine", qw_buf_contents(out), oref, (size_t)rows * H, 1e-5);

    /* scale_rows_sigmoid */
    qw_buf g = mkbuf((size_t)rows * 4);
    float *gp = qw_buf_contents(g);
    fill_random(gp, rows, 23);
    float *sref = malloc((size_t)rows * H * 4);
    const float *op = qw_buf_contents(out);
    for (int32_t r = 0; r < rows; r++) for (int32_t i = 0; i < H; i++) sref[(size_t)r * H + i] = op[(size_t)r * H + i] * sigm(gp[r]);
    c = qw_cmd_begin();
    qw_op_scale_rows_sigmoid(c, qw_ref_at(out, 0), qw_ref_at(g, 0), rows, H);
    run(c, "scale_rows_sigmoid");
    report("scale_rows_sigmoid", qw_buf_contents(out), sref, (size_t)rows * H, 1e-5);

    free(wref); free(guref); free(actref); free(yref); free(oref); free(sref);
    qw_buf_free(lg); qw_buf_free(idx); qw_buf_free(w); qw_buf_free(x); qw_buf_free(gu);
    qw_buf_free(act); qw_buf_free(y); qw_buf_free(out); qw_buf_free(g);
}

/* ---- the sigmoid-gated norm, and the dilated conv ---------------------------- */

static void test_gated_and_conv(void) {
    const int32_t rows = 6, dim = 128;
    qw_buf x = mkbuf((size_t)rows * dim * 4), gate = mkbuf((size_t)rows * dim * 4), y = mkbuf((size_t)rows * dim * 4);
    qw_buf w = mkbuf((size_t)dim * 2);
    float *xp = qw_buf_contents(x), *gp = qw_buf_contents(gate);
    uint16_t *wp = qw_buf_contents(w);
    fill_random(xp, (size_t)rows * dim, 31); fill_random(gp, (size_t)rows * dim, 32);
    for (int32_t i = 0; i < dim; i++) { float t; fill_random(&t, 1, 300 + i); wp[i] = f2bf(1.0f + 0.2f * t); }
    qw_cmd c = qw_cmd_begin();
    qw_op_rms_norm_gated_sigmoid(c, qw_ref_at(y, 0), qw_ref_at(x, 0), qw_ref_at(w, 0), qw_ref_at(gate, 0),
                                 dim, rows, 1e-6f, 1.0f);
    run(c, "rms_norm_gated_sigmoid");
    float *ref = malloc((size_t)rows * dim * 4);
    for (int32_t r = 0; r < rows; r++) {
        float ss = 0.0f; for (int32_t i = 0; i < dim; i++) ss += xp[(size_t)r * dim + i] * xp[(size_t)r * dim + i];
        const float inv = 1.0f / sqrtf(ss / dim + 1e-6f);
        for (int32_t i = 0; i < dim; i++)
            ref[(size_t)r * dim + i] = xp[(size_t)r * dim + i] * inv * qw_bf16_to_f32_c(wp[i]) * sigm(gp[(size_t)r * dim + i]);
    }
    report("rms_norm_gated_sigmoid", qw_buf_contents(y), ref, (size_t)rows * dim, 1e-5);
    free(ref);
    qw_buf_free(x); qw_buf_free(gate); qw_buf_free(y); qw_buf_free(w);

    /* dilated conv: channels C, two calls of rows each, state carried */
    const int32_t C = 96, K = 4, DIL = 3, SL = (K - 1) * DIL;
    qw_buf xin = mkbuf((size_t)2 * rows * C * 4), st = mkbuf((size_t)SL * C * 4), cw = mkbuf((size_t)C * K * 2);
    qw_buf yo = mkbuf((size_t)2 * rows * C * 4);
    float *xi = qw_buf_contents(xin);
    uint16_t *cwp = qw_buf_contents(cw);
    fill_random(xi, (size_t)2 * rows * C, 41);
    for (int32_t i = 0; i < C * K; i++) { float t; fill_random(&t, 1, 500 + i); cwp[i] = f2bf(t); }
    c = qw_cmd_begin();
    qw_op_conv1d_dilated_silu(c, qw_ref_at(yo, 0), qw_ref_at(xin, 0), qw_ref_at(st, 0), qw_ref_at(cw, 0), C, rows, K, DIL);
    run(c, "conv1d_dilated_silu (1)");
    c = qw_cmd_begin();
    qw_op_conv1d_dilated_silu(c, qw_ref_offset(qw_ref_at(yo, 0), (size_t)rows * C * 4),
                              qw_ref_offset(qw_ref_at(xin, 0), (size_t)rows * C * 4), qw_ref_at(st, 0), qw_ref_at(cw, 0),
                              C, rows, K, DIL);
    run(c, "conv1d_dilated_silu (2)");
    /* reference over the whole 2*rows sequence with zero history */
    float *cref = malloc((size_t)2 * rows * C * 4);
    for (int32_t ch = 0; ch < C; ch++)
        for (int32_t t = 0; t < 2 * rows; t++) {
            float acc = 0.0f;
            for (int32_t j = 0; j < K; j++) {
                const int32_t back = (K - 1 - j) * DIL;
                const float in = t - back >= 0 ? xi[(size_t)(t - back) * C + ch] : 0.0f;
                acc += in * qw_bf16_to_f32_c(cwp[ch * K + j]);
            }
            cref[(size_t)t * C + ch] = silu(acc);
        }
    report("conv1d_dilated_silu", qw_buf_contents(yo), cref, (size_t)2 * rows * C, 1e-5);
    free(cref);
    qw_buf_free(xin); qw_buf_free(st); qw_buf_free(cw); qw_buf_free(yo);
}

/* ---- QSA ---------------------------------------------------------------------- */

static void test_qsa(qwasar_engine *e) {
    const qw_config *cfg = qwasar_engine_config(e);
    const qw_layer *L = qwasar_engine_layer(e, 3);
    const int32_t nq = cfg->indexer_n_heads, d = cfg->indexer_head_dim, ratio = cfg->indexer_compress_ratio;
    const int32_t rows = 5, base_pos = 13, max_ctx = 64, max_blocks = max_ctx / ratio;
    const int32_t rd = cfg->rotary_dim;

    /* scores */
    qw_buf qn = mkbuf((size_t)rows * nq * d * 4), ik = mkbuf((size_t)max_ctx * d * 4);
    qw_buf sc = mkbuf((size_t)rows * max_blocks * 4), invf = mkbuf((size_t)(rd / 2) * 4);
    float *qp = qw_buf_contents(qn), *kp = qw_buf_contents(ik);
    fill_random(qp, (size_t)rows * nq * d, 51); fill_random(kp, (size_t)max_ctx * d, 52);
    uint8_t axis[64];
    qw_rope_tables(axis, qw_buf_contents(invf), rd, cfg->rope_theta, cfg->mrope_section);
    const float *inv_freq = qw_buf_contents(invf);
    qw_cmd c = qw_cmd_begin();
    qw_op_qsa_scores(c, qw_ref_at(sc, 0), qw_ref_at(qn, 0), qw_ref_at(ik, 0), qw_tensor_ref(L->indexer.k_norm),
                     qw_ref_at(invf, 0), rows, nq, d, ratio, base_pos, rd, max_blocks, cfg->rms_norm_eps);
    run(c, "qsa_scores");
    const uint16_t *kw = qw_tensor_data(L->indexer.k_norm);
    float *sref = calloc((size_t)rows * max_blocks, 4);
    const float *scp = qw_buf_contents(sc);
    double worst = 0.0;
    int32_t compared = 0;
    for (int32_t r = 0; r < rows; r++) {
        const int32_t n_blocks = (base_pos + r + 1) / ratio;
        for (int32_t b = 0; b < n_blocks; b++) {
            float pooled[256], kn[256];
            for (int32_t i = 0; i < d; i++) {
                float acc = 0.0f;
                for (int32_t t = 0; t < ratio; t++) acc += kp[(size_t)(b * ratio + t) * d + i];
                pooled[i] = acc / ratio;
            }
            qw_cpu_rms_norm(kn, pooled, kw, d, 1, cfg->rms_norm_eps, 1.0f);
            int32_t bp[3] = { b * ratio, b * ratio, b * ratio };
            qw_cpu_rope_partial(kn, bp, axis, inv_freq, 1, 1, d, rd);
            float s = 0.0f;
            for (int32_t h = 0; h < nq; h++) {
                float dot = 0.0f;
                for (int32_t i = 0; i < d; i++) dot += qp[((size_t)r * nq + h) * d + i] * kn[i];
                if (dot > 0) s += dot;
            }
            sref[(size_t)r * max_blocks + b] = s / sqrtf((float)d);
            const double err = fabs(scp[(size_t)r * max_blocks + b] - sref[(size_t)r * max_blocks + b]);
            if (err > worst) worst = err;
            compared++;
        }
    }
    printf("  %-28s max |diff| %.2e over %d (row, block) scores\n", "qsa_scores", worst, compared);
    CHECK(worst < 1e-4, "qsa_scores: max |diff| %.3g", worst);

    /* select on random distinct scores: top-k blocks plus the tail */
    qw_buf mask = mkbuf((size_t)rows * max_ctx);
    float *scr = qw_buf_contents(sc);
    fill_random(scr, (size_t)rows * max_blocks, 53);
    float *keep = malloc((size_t)rows * max_blocks * 4);
    memcpy(keep, scr, (size_t)rows * max_blocks * 4);
    const int32_t block_topk = 2;
    c = qw_cmd_begin();
    qw_op_qsa_select(c, qw_ref_at(mask, 0), qw_ref_at(sc, 0), rows, ratio, base_pos, block_topk, max_ctx, max_blocks);
    run(c, "qsa_select");
    const uint8_t *mp = qw_buf_contents(mask);
    int32_t mask_bad = 0;
    for (int32_t r = 0; r < rows; r++) {
        const int32_t n_keys = base_pos + r + 1, n_blocks = n_keys / ratio;
        uint8_t ref[64] = { 0 };
        for (int32_t t = n_blocks * ratio; t < n_keys; t++) ref[t] = 1;
        float *s = malloc((size_t)n_blocks * 4);
        memcpy(s, keep + (size_t)r * max_blocks, (size_t)n_blocks * 4);
        const int32_t take = block_topk < n_blocks ? block_topk : n_blocks;
        for (int32_t n = 0; n < take; n++) {
            int32_t best = -1;
            for (int32_t b = 0; b < n_blocks; b++) if (s[b] > -FLT_MAX && (best < 0 || s[b] > s[best])) best = b;
            for (int32_t t = 0; t < ratio; t++) ref[best * ratio + t] = 1;
            s[best] = -FLT_MAX;
        }
        for (int32_t t = 0; t < n_keys; t++) if ((mp[(size_t)r * max_ctx + t] != 0) != (ref[t] != 0)) mask_bad++;
        free(s);
    }
    printf("  %-28s mismatches %d\n", "qsa_select", mask_bad);
    CHECK(mask_bad == 0, "qsa_select: %d mask entries differ", mask_bad);

    /* masked attention against a scalar softmax over fp16-rounded K/V */
    const int32_t QH = cfg->num_attention_heads, KVH = cfg->num_key_value_heads, D = cfg->head_dim;
    const int32_t n_keys_max = base_pos + rows;
    qw_buf q = mkbuf((size_t)rows * QH * D * 4), k = mkbuf((size_t)n_keys_max * KVH * D * 4), v = mkbuf((size_t)n_keys_max * KVH * D * 4);
    qw_buf kc = mkbuf((size_t)KVH * max_ctx * D * 2), vc = mkbuf((size_t)KVH * max_ctx * D * 2), out = mkbuf((size_t)rows * QH * D * 4);
    float *qq = qw_buf_contents(q), *kk = qw_buf_contents(k), *vv = qw_buf_contents(v);
    fill_random(qq, (size_t)rows * QH * D, 61); fill_random(kk, (size_t)n_keys_max * KVH * D, 62); fill_random(vv, (size_t)n_keys_max * KVH * D, 63);
    qw_cpu_kv_write(qw_buf_contents(kc), qw_buf_contents(vc), kk, vv, n_keys_max, KVH, D, max_ctx, 0);
    const float scale = 1.0f / sqrtf((float)D);
    c = qw_cmd_begin();
    qw_op_attn_masked(c, qw_ref_at(out, 0), qw_ref_at(q, 0), qw_ref_at(kc, 0), qw_ref_at(vc, 0), qw_ref_at(mask, 0),
                      rows, QH, KVH, D, max_ctx, base_pos, scale);
    run(c, "attn_masked");
    float *aref = calloc((size_t)rows * QH * D, 4);
    const uint16_t *kcp = qw_buf_contents(kc), *vcp = qw_buf_contents(vc);
    float probs[64];
    for (int32_t r = 0; r < rows; r++) for (int32_t h = 0; h < QH; h++) {
        const int32_t kvh = h / (QH / KVH), n_keys = base_pos + r + 1;
        const float *qv = qq + ((size_t)r * QH + h) * D;
        float m = -FLT_MAX;
        for (int32_t t = 0; t < n_keys; t++) {
            if (!mp[(size_t)r * max_ctx + t]) { probs[t] = -FLT_MAX; continue; }
            float sdot = 0.0f;
            for (int32_t i = 0; i < D; i++) sdot += qv[i] * scale * qw_f16_to_f32_c(kcp[((size_t)kvh * max_ctx + t) * D + i]);
            probs[t] = sdot; if (sdot > m) m = sdot;
        }
        float sum = 0.0f;
        for (int32_t t = 0; t < n_keys; t++) { probs[t] = mp[(size_t)r * max_ctx + t] ? expf(probs[t] - m) : 0.0f; sum += probs[t]; }
        float *o = aref + ((size_t)r * QH + h) * D;
        for (int32_t t = 0; t < n_keys; t++) {
            if (!mp[(size_t)r * max_ctx + t]) continue;
            for (int32_t i = 0; i < D; i++) o[i] += probs[t] / sum * qw_f16_to_f32_c(vcp[((size_t)kvh * max_ctx + t) * D + i]);
        }
    }
    report("attn_masked", qw_buf_contents(out), aref, (size_t)rows * QH * D, 1e-4);

    free(sref); free(keep); free(aref);
    qw_buf_free(qn); qw_buf_free(ik); qw_buf_free(sc); qw_buf_free(invf); qw_buf_free(mask);
    qw_buf_free(q); qw_buf_free(k); qw_buf_free(v); qw_buf_free(kc); qw_buf_free(vc); qw_buf_free(out);
}

/* ---- the reused ops, at this family's shapes ---------------------------------
 *
 * The dense bf16 matvec served only the draft head before, at a width of
 * thousands; the mixers call it with n = 4 and the shared-expert gate with
 * n = 1.  The quantised matmul is called at the mixer's narrow shapes. */
static void test_reused(qwasar_engine *e) {
    static const struct { int32_t k, n, rows; } shapes[] = {
        { 64, 1, 1 }, { 64, 8, 1 }, { 256, 4, 1 }, { 256, 4, 5 }, { 64, 8, 7 }, { 2560, 512, 3 },
    };
    for (size_t i = 0; i < sizeof shapes / sizeof *shapes; i++) {
        const int32_t k = shapes[i].k, n = shapes[i].n, rows = shapes[i].rows;
        qw_buf x = mkbuf((size_t)rows * k * 4), w = mkbuf((size_t)n * k * 2), y = mkbuf((size_t)rows * n * 4);
        float *xp = qw_buf_contents(x);
        uint16_t *wp = qw_buf_contents(w);
        fill_random(xp, (size_t)rows * k, 71 + (uint32_t)i);
        for (int32_t j = 0; j < n * k; j++) { float t; fill_random(&t, 1, 900 + (uint32_t)j + 7 * (uint32_t)i); wp[j] = f2bf(t); }
        qw_cmd c = qw_cmd_begin();
        qw_op_dmat_bf16(c, qw_ref_at(y, 0), qw_ref_at(x, 0), qw_ref_at(w, 0), k, n, rows);
        run(c, "dmat_bf16");
        float *ref = malloc((size_t)rows * n * 4);
        qw_cpu_dmv_bf16(ref, xp, wp, k, n, rows);
        char label[64];
        snprintf(label, sizeof label, "dmat_bf16 k=%d n=%d rows=%d", k, n, rows);
        report(label, qw_buf_contents(y), ref, (size_t)rows * n, 1e-4);
        free(ref);
        qw_buf_free(x); qw_buf_free(w); qw_buf_free(y);
    }

    const qw_layer *L = qwasar_engine_layer(e, 0);
    const qw_qlinear *qs[] = { &L->attn_hc.mix_down, &L->attn_hc.mix_up, &L->in_proj_qkv, &L->moe.sh_gate };
    const char *names[] = { "mix_down", "mix_up", "in_proj_qkv", "sh_gate" };
    static const int32_t rowsets[] = { 1, 5, 32 };
    for (size_t i = 0; i < 4; i++)
        for (size_t ri = 0; ri < 3; ri++) {
            const qw_qlinear *q = qs[i];
            const int32_t rows = rowsets[ri], k = q->in_features, n = q->out_features;
            qw_buf x = mkbuf((size_t)rows * k * 4), y = mkbuf((size_t)rows * n * 4);
            float *xp = qw_buf_contents(x);
            fill_random(xp, (size_t)rows * k, 81 + (uint32_t)i);
            qw_cmd c = qw_cmd_begin();
            qw_op_qmat_q4(c, qw_ref_at(y, 0), qw_ref_at(x, 0), qw_tensor_ref(q->weight), qw_tensor_ref(q->scales),
                          qw_tensor_ref(q->biases), k, n, rows);
            run(c, "qmat_q4");
            float *ref = malloc((size_t)rows * n * 4);
            qw_cpu_qmv_q4(ref, xp, qw_tensor_data(q->weight), qw_tensor_data(q->scales), qw_tensor_data(q->biases), k, n, rows);
            char label[64];
            snprintf(label, sizeof label, "qmat_q4 %s rows=%d", names[i], rows);
            /* From QW_QMM_MIN_ROWS the tiled matmul runs, with fp16 operand
             * tiles by design (metal/qmm.metal): ~2e-4, not 1e-7. */
            report(label, qw_buf_contents(y), ref, (size_t)rows * n, rows >= QW_QMM_MIN_ROWS ? 1e-3 : 1e-4);
            free(ref);
            qw_buf_free(x); qw_buf_free(y);
        }
}

int main(void) {
    char err[512] = "";
    qwasar_options o = { .model_path = "tests/fixtures/flashnext-tiny-q4", .context_size = 256 };
    qwasar_engine *e = qwasar_engine_load(&o, err, sizeof err);
    if (!e) { fprintf(stderr, "cannot load the toy checkpoint: %s\n", err); return 1; }

    printf("== hyper-connections\n"); test_hc();
    printf("== moe\n");               test_moe(e);
    printf("== norm, conv\n");        test_gated_and_conv();
    printf("== qsa\n");               test_qsa(e);
    printf("== reused ops\n");        test_reused(e);

    qwasar_engine_free(e);
    if (fails) { fprintf(stderr, "%d failure(s)\n", fails); return 1; }
    printf("sparse: all checks pass\n");
    return 0;
}
