/* Flash-Next (qwen4_exp): the kernels the 27B never needed.
 *
 * PLAN-flash-next.md.  Everything the family adds over the 27B lives here --
 * the hyper-connection mixers, the MoE router and expert banks, QSA's block
 * indexer and masked attention, the engram layer's gate and dilated conv --
 * each one a direct transcription of qwasar_flash_cpu.c, which is the twin
 * tests/test_flashnext holds them to.  Correctness first: several of these
 * are the simplest shape that is right (the selection kernel in particular),
 * and the measurements that justify a faster one have not been taken yet.
 *
 * File order matters: the sources are concatenated alphabetically, and this
 * one relies on qw_row_sumsq from norm.metal and the helpers in common.metal. */

/* ---- hyper-connections ---------------------------------------------------- */

struct qw_hc_args { uint rows, H, S; };

/* h4[r, s*H + i] = x[r, i] for every stream s: the embedding, repeated. */
kernel void qw_repeat_cols(
    device const float *x  [[buffer(0)]],   /* [rows, H] */
    device       float *h4 [[buffer(1)]],   /* [rows, S*H] */
    constant qw_hc_args &a [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    const uint HH = a.S * a.H;
    if (gid >= a.rows * HH) return;
    const uint r = gid / HH, i = (gid % HH) % a.H;
    h4[gid] = x[r * a.H + i];
}

/* Grouped RMS norm: `groups` streams of `dim` per row, each normalised on
 * its own, then the full-width (+1) weight.  One threadgroup per stream. */
struct qw_gnorm_args { uint dim, groups, rows; float eps; };

kernel void qw_rms_norm_grouped(
    device const float  *x [[buffer(0)]],   /* [rows, groups*dim] */
    device const ushort *w [[buffer(1)]],   /* [groups*dim] bf16 */
    device       float  *y [[buffer(2)]],
    constant qw_gnorm_args &a [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint ntg  [[threads_per_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint nsg  [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float partial[32];
    const uint g = tgid % a.groups;
    device const float *xr = x + (ulong)tgid * a.dim;     /* row*groups + g */
    device       float *yr = y + (ulong)tgid * a.dim;
    device const ushort *wr = w + (ulong)g * a.dim;
    float sumsq = qw_row_sumsq(xr, a.dim, tid, ntg, sgid, nsg, lane, partial);
    const float inv = rsqrt(sumsq / float(a.dim) + a.eps);
    for (uint i = tid; i < a.dim; i += ntg) yr[i] = xr[i] * inv * qw_bf16_to_f32(wr[i]);
}

/* y = silu(y * scale), in place -- the mixer's `silu(down(n) / hc_count)`. */
struct qw_scale_args { uint n; float scale; };

kernel void qw_silu_scale(
    device float *y [[buffer(0)]],
    constant qw_scale_args &a [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < a.n) y[gid] = qw_silu(y[gid] * a.scale);
}

/* x[r, i] = mean over streams of sigmoid(m[r, s*H+i]) * n[r, s*H+i]. */
kernel void qw_hc_mix(
    device const float *n  [[buffer(0)]],   /* [rows, S*H] normalised streams */
    device const float *m  [[buffer(1)]],   /* [rows, S*H] pre-sigmoid mix weights */
    device       float *x  [[buffer(2)]],   /* [rows, H] */
    constant qw_hc_args &a [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= a.rows * a.H) return;
    const uint r = gid / a.H, i = gid % a.H;
    float acc = 0.0f;
    for (uint s = 0; s < a.S; ++s) {
        const ulong j = (ulong)r * a.S * a.H + (ulong)s * a.H + i;
        acc += qw_sigmoid(m[j]) * n[j];
    }
    x[gid] = acc / float(a.S);
}

/* h4[r, s*H+i] += out[r, i] * 2*sigmoid(inj[r, s] / S). */
kernel void qw_hc_inject(
    device       float *h4  [[buffer(0)]],   /* [rows, S*H] */
    device const float *out [[buffer(1)]],   /* [rows, H] */
    device const float *inj [[buffer(2)]],   /* [rows, S] raw block-inject logits */
    constant qw_hc_args &a  [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    const uint HH = a.S * a.H;
    if (gid >= a.rows * HH) return;
    const uint r = gid / HH, s = (gid % HH) / a.H, i = gid % a.H;
    const float w = 2.0f * qw_sigmoid(inj[r * a.S + s] / float(a.S));
    h4[gid] += out[r * a.H + i] * w;
}

/* ---- mixture of experts --------------------------------------------------- */

/* Softmax in fp32 over every expert, top-k, renormalised: one thread per
 * token.  K is small and E is a few hundred, so the scans are nothing. */
struct qw_route_args { uint rows, E, K, norm; };

kernel void qw_moe_route(
    device const float *logits [[buffer(0)]],   /* [rows, E] */
    device       int   *idx    [[buffer(1)]],   /* [rows, K] */
    device       float *w      [[buffer(2)]],   /* [rows, K] */
    constant qw_route_args &a  [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= a.rows) return;
    device const float *l = logits + (ulong)gid * a.E;
    float m = -FLT_MAX;
    for (uint e = 0; e < a.E; ++e) m = max(m, l[e]);
    float sum = 0.0f;
    for (uint e = 0; e < a.E; ++e) sum += exp(l[e] - m);

    int   chosen[32];
    float wsum = 0.0f;
    for (uint k = 0; k < a.K; ++k) {
        int best = -1;
        float bv = -1.0f;
        for (uint e = 0; e < a.E; ++e) {
            bool taken = false;
            for (uint j = 0; j < k; ++j) taken = taken || (chosen[j] == (int)e);
            if (taken) continue;
            const float p = exp(l[e] - m) / sum;
            if (p > bv) { bv = p; best = (int)e; }
        }
        chosen[k] = best;
        w[gid * a.K + k] = bv;
        wsum += bv;
    }
    for (uint k = 0; k < a.K; ++k) {
        idx[gid * a.K + k] = chosen[k];
        if (a.norm) w[gid * a.K + k] /= wsum;
    }
}

/* Matvec against an expert bank: pair p reads matrix idx[p] of the bank and
 * activation row p (x_by_pair) or p / K.  Otherwise qw_qmv_q4_g64 exactly:
 * one simdgroup per QW_QMV_ROWS output rows, lanes walking words. */
struct qw_bank_args { uint k, n, pairs, K, x_by_pair; };

kernel void qw_qmv_q4_bank(
    device const uint    *w       [[buffer(0)]],   /* [E, n, k/8] */
    device const ushort  *scales  [[buffer(1)]],   /* [E, n, k/64] */
    device const ushort  *biases  [[buffer(2)]],
    device const float   *x       [[buffer(3)]],   /* [rows or pairs, k] */
    device const int     *idx     [[buffer(4)]],   /* [pairs] */
    device       float   *y       [[buffer(5)]],   /* [pairs, n] */
    constant qw_bank_args &a      [[buffer(6)]],
    uint3 tgid  [[threadgroup_position_in_grid]],
    uint  sgid  [[simdgroup_index_in_threadgroup]],
    uint  nsg   [[simdgroups_per_threadgroup]],
    uint  lane  [[thread_index_in_simdgroup]])
{
    const uint words  = a.k / QW_QPER_WORD;
    const uint groups = a.k / QW_QGROUP;
    const uint row0 = (tgid.x * nsg + sgid) * QW_QMV_ROWS;
    if (row0 >= a.n) return;

    const uint p = tgid.y;
    const uint e = (uint)idx[p];
    const uint xr = a.x_by_pair ? p : p / a.K;
    device const float  *xv = x + (ulong)xr * a.k;
    device const uint   *we = w      + (ulong)e * a.n * words;
    device const ushort *se = scales + (ulong)e * a.n * groups;
    device const ushort *be = biases + (ulong)e * a.n * groups;

    float acc[QW_QMV_ROWS] = { 0.0f, 0.0f, 0.0f, 0.0f };
    for (uint wi = lane; wi < words; wi += 32) {
        const uint g = wi / QW_WORDS_PER_GROUP;
        float xs[QW_QPER_WORD];
#pragma unroll
        for (int j = 0; j < QW_QPER_WORD; ++j) xs[j] = xv[wi * QW_QPER_WORD + j];
#pragma unroll
        for (uint r = 0; r < QW_QMV_ROWS; ++r) {
            const uint n = row0 + r;
            if (n >= a.n) break;
            const uint  ww = we[(ulong)n * words + wi];
            const float sc = qw_bf16_to_f32(se[(ulong)n * groups + g]);
            const float bi = qw_bf16_to_f32(be[(ulong)n * groups + g]);
            float4 ev, od;
            qw_unpack8_affine(ww, sc * 255.0f, bi, &ev, &od);
#pragma unroll
            for (uint k = 0; k < 4; ++k) {
                acc[r] = fma(ev[k], xs[2 * k],     acc[r]);
                acc[r] = fma(od[k], xs[2 * k + 1], acc[r]);
            }
        }
    }
#pragma unroll
    for (uint r = 0; r < QW_QMV_ROWS; ++r) {
        const float v = simd_sum(acc[r]);
        const uint  n = row0 + r;
        if (lane == 0 && n < a.n) y[(ulong)p * a.n + n] = v;
    }
}

/* act[p, i] = silu(gu[p, i]) * gu[p, I + i]: the fused expert's two halves. */
struct qw_swiglu_split_args { uint pairs, I; };

kernel void qw_swiglu_split(
    device const float *gu  [[buffer(0)]],   /* [pairs, 2I] */
    device       float *act [[buffer(1)]],   /* [pairs, I] */
    constant qw_swiglu_split_args &a [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= a.pairs * a.I) return;
    const uint p = gid / a.I, i = gid % a.I;
    const float g = gu[(ulong)p * 2 * a.I + i], u = gu[(ulong)p * 2 * a.I + a.I + i];
    act[gid] = qw_silu(g) * u;
}

/* out[r, i] = sum_k w[r, k] * y[r*K + k, i]. */
struct qw_combine_args { uint rows, K, H; };

kernel void qw_moe_combine(
    device const float *y   [[buffer(0)]],   /* [rows*K, H] */
    device const float *w   [[buffer(1)]],   /* [rows, K] */
    device       float *out [[buffer(2)]],   /* [rows, H] */
    constant qw_combine_args &a [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= a.rows * a.H) return;
    const uint r = gid / a.H, i = gid % a.H;
    float acc = 0.0f;
    for (uint k = 0; k < a.K; ++k)
        acc = fma(w[r * a.K + k], y[((ulong)r * a.K + k) * a.H + i], acc);
    out[gid] = acc;
}

/* y[r, :] *= sigmoid(g[r]): the shared expert's scalar gate. */
struct qw_rowscale_args { uint rows, dim; };

kernel void qw_scale_rows_sigmoid(
    device       float *y [[buffer(0)]],   /* [rows, dim] */
    device const float *g [[buffer(1)]],   /* [rows] */
    constant qw_rowscale_args &a [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= a.rows * a.dim) return;
    y[gid] *= qw_sigmoid(g[gid / a.dim]);
}

/* ---- gated delta, this family's gate ---------------------------------------
 *
 * rms_norm(x, w) * sigmoid(gate): output_gate_type "sigmoid" where the 27B
 * used silu.  Same kernel shape as qw_rms_norm_gated. */
kernel void qw_rms_norm_gated_sigmoid(
    device const float   *x    [[buffer(0)]],
    device const ushort  *w    [[buffer(1)]],
    device const float   *gate [[buffer(2)]],
    device       float   *y    [[buffer(3)]],
    constant qw_norm_args &a   [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint ntg  [[threads_per_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint nsg  [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float partial[32];
    device const float *xr = x    + (ulong)tgid * a.dim;
    device const float *gr = gate + (ulong)tgid * a.dim;
    device       float *yr = y    + (ulong)tgid * a.dim;
    float sumsq = qw_row_sumsq(xr, a.dim, tid, ntg, sgid, nsg, lane, partial);
    const float inv = rsqrt(sumsq / float(a.dim) + a.eps) * a.out_scale;
    for (uint i = tid; i < a.dim; i += ntg) {
        float v = xr[i] * inv;
        if (a.has_weight) v *= qw_bf16_to_f32(w[i]);
        yr[i] = v * qw_sigmoid(gr[i]);
    }
}

/* ---- Qwen Sparse Attention -------------------------------------------------
 *
 * The indexer scores every complete block of `ratio` cached raw keys against
 * a query's normalised, rotated index heads; the top `block_topk` blocks plus
 * the incomplete tail are what attention may see.  One simdgroup per
 * (query, block): lanes own index-head dims lane, lane+32, ..., which puts a
 * rotary pair (j, j+32) in one lane -- so rotary_dim must be 64, and the
 * dispatcher refuses anything else. */
struct qw_qsa_score_args {
    uint rows, nq, d, ratio, base_pos, rotary_dim, max_blocks;
    float eps;
};

#define QW_QSA_MAXM 8   /* index head dims per lane: d <= 256 */

kernel void qw_qsa_scores(
    device const float  *qn       [[buffer(0)]],   /* [rows, nq, d] normed + rotated */
    device const float  *ikeys    [[buffer(1)]],   /* [max_ctx, d] raw keys */
    device const ushort *kw       [[buffer(2)]],   /* [d] bf16 k_layernorm (+1) */
    device const float  *inv_freq [[buffer(3)]],   /* [rotary_dim/2] */
    device       float  *scores   [[buffer(4)]],   /* [rows, max_blocks] */
    constant qw_qsa_score_args &a [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  nsg  [[simdgroups_per_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    const uint r = tgid.y;
    const uint b = tgid.x * nsg + sgid;
    const uint visible  = a.base_pos + r + 1;
    const uint n_blocks = visible / a.ratio;
    if (b >= n_blocks) return;
    const uint M = a.d / 32;

    /* pooled raw key, one lane strip at a time */
    float kn[QW_QSA_MAXM];
    float ss = 0.0f;
    for (uint m = 0; m < M; ++m) {
        const uint dim = lane + 32 * m;
        float acc = 0.0f;
        for (uint t = 0; t < a.ratio; ++t) acc += ikeys[(ulong)(b * a.ratio + t) * a.d + dim];
        kn[m] = acc / float(a.ratio);
        ss = fma(kn[m], kn[m], ss);
    }
    ss = simd_sum(ss);
    const float inv = rsqrt(ss / float(a.d) + a.eps);
    for (uint m = 0; m < M; ++m) kn[m] = kn[m] * inv * qw_bf16_to_f32(kw[lane + 32 * m]);

    /* rope on the first rotary_dim dims at the block's first position:
     * pairs (j, j+32) are lane strips 0 and 1. */
    {
        const float angle = float(b * a.ratio) * inv_freq[lane];
        const float c = cos(angle), s = sin(angle);
        const float x0 = kn[0], x1 = kn[1];
        kn[0] = x0 * c - x1 * s;
        kn[1] = x1 * c + x0 * s;
    }

    float score = 0.0f;
    for (uint h = 0; h < a.nq; ++h) {
        device const float *qh = qn + ((ulong)r * a.nq + h) * a.d;
        float dot = 0.0f;
        for (uint m = 0; m < M; ++m) dot = fma(qh[lane + 32 * m], kn[m], dot);
        dot = simd_sum(dot);
        if (dot > 0.0f) score += dot;
    }
    if (lane == 0) scores[(ulong)r * a.max_blocks + b] = score / sqrt(float(a.d));
}

/* Top-k blocks per query into a byte mask over cache positions, plus the
 * tail.  One threadgroup per query; k rounds of a parallel argmax, ties to
 * the lowest index, matching the reference.  O(k * n_blocks) per query --
 * the simplest correct shape, and the one to measure before replacing. */
struct qw_qsa_select_args { uint rows, ratio, base_pos, block_topk, max_ctx, max_blocks; };

kernel void qw_qsa_select(
    device       float *scores [[buffer(0)]],   /* [rows, max_blocks], consumed */
    device       uchar *mask   [[buffer(1)]],   /* [rows, max_ctx] */
    constant qw_qsa_select_args &a [[buffer(2)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint ntg  [[threads_per_threadgroup]])
{
    threadgroup float bv[256];
    threadgroup int   bi[256];

    const uint r = tgid;
    const uint n_keys = a.base_pos + r + 1;
    const uint n_blocks = n_keys / a.ratio;
    device float *sc = scores + (ulong)r * a.max_blocks;
    device uchar *mr = mask + (ulong)r * a.max_ctx;

    for (uint t = tid; t < n_keys; t += ntg) mr[t] = (t >= n_blocks * a.ratio) ? 1 : 0;
    threadgroup_barrier(mem_flags::mem_device);

    const uint take = min(a.block_topk, n_blocks);
    for (uint it = 0; it < take; ++it) {
        float lv = -FLT_MAX;
        int   li = -1;
        for (uint b = tid; b < n_blocks; b += ntg) {
            const float v = sc[b];
            if (v > lv || (v == lv && li >= 0 && (int)b < li)) { lv = v; li = (int)b; }
        }
        bv[tid] = lv; bi[tid] = li;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float gv = -FLT_MAX; int gi = -1;
            for (uint t = 0; t < ntg; ++t)
                if (bi[t] >= 0 && (bv[t] > gv || (bv[t] == gv && bi[t] < gi))) { gv = bv[t]; gi = bi[t]; }
            if (gi >= 0) {
                sc[gi] = -FLT_MAX;
                for (uint t = 0; t < a.ratio; ++t) mr[gi * a.ratio + t] = 1;
            }
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
}

/* qw_attn_decode with a per-(query, position) byte mask: a masked key
 * contributes nothing.  Identical otherwise, head_dim 256. */
kernel void qw_attn_masked(
    device const float   *q    [[buffer(0)]],
    device const half    *kc   [[buffer(1)]],
    device const half    *vc   [[buffer(2)]],
    device       float   *out  [[buffer(3)]],
    device const uchar   *mask [[buffer(4)]],   /* [rows, max_ctx] */
    constant qw_attn_args &a   [[buffer(5)]],
    uint tgid     [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]])
{
    threadgroup float tg_out[QW_ATTN_SIMDS * QW_ATTN_LANES];
    threadgroup float tg_max[QW_ATTN_SIMDS];
    threadgroup float tg_sum[QW_ATTN_SIMDS];

    const uint row = tgid / a.q_heads;
    const uint qh  = tgid % a.q_heads;
    const uint kvh = qh / a.gqa;
    const int n_keys = (int)(a.base_pos + row) + 1;
    device const uchar *mr = mask + (ulong)row * a.max_ctx;

    const device float *qp = q + ((ulong)row * a.q_heads + qh) * QW_ATTN_D + simd_lid * QW_ATTN_PER_THREAD;
    const device half  *kp = kc + ((ulong)kvh * a.max_ctx + simd_gid) * QW_ATTN_D + simd_lid * QW_ATTN_PER_THREAD;
    const device half  *vp = vc + ((ulong)kvh * a.max_ctx + simd_gid) * QW_ATTN_D + simd_lid * QW_ATTN_PER_THREAD;

    float qv[QW_ATTN_PER_THREAD];
    float acc[QW_ATTN_PER_THREAD];
#pragma unroll
    for (int i = 0; i < QW_ATTN_PER_THREAD; ++i) { qv[i] = a.scale * qp[i]; acc[i] = 0.0f; }

    float run_max = -FLT_MAX;
    float run_sum = 0.0f;
    for (int t = (int)simd_gid; t < n_keys; t += QW_ATTN_SIMDS) {
        if (mr[t]) {
            float score = 0.0f;
#pragma unroll
            for (int i = 0; i < QW_ATTN_PER_THREAD; ++i) score = fma(qv[i], float(kp[i]), score);
            score = simd_sum(score);
            const float new_max = max(run_max, score);
            const float factor  = exp(run_max - new_max);
            const float w       = exp(score - new_max);
            run_max = new_max;
            run_sum = run_sum * factor + w;
#pragma unroll
            for (int i = 0; i < QW_ATTN_PER_THREAD; ++i) acc[i] = fma(acc[i], factor, w * float(vp[i]));
        }
        kp += QW_ATTN_SIMDS * QW_ATTN_D;
        vp += QW_ATTN_SIMDS * QW_ATTN_D;
    }

    if (simd_lid == 0) { tg_max[simd_gid] = run_max; tg_sum[simd_gid] = run_sum; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float part_max = tg_max[simd_lid];
    const float glob_max = simd_max(part_max);
    const float rescale  = exp(part_max - glob_max);
    const float glob_sum = simd_sum(tg_sum[simd_lid] * rescale);
    for (int i = 0; i < QW_ATTN_PER_THREAD; ++i) {
        tg_out[simd_lid * QW_ATTN_LANES + simd_gid] = acc[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float v = simd_sum(tg_out[simd_gid * QW_ATTN_LANES + simd_lid] * rescale);
        acc[i] = glob_sum == 0.0f ? v : v / glob_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (simd_lid == 0) {
        device float *op = out + ((ulong)row * a.q_heads + qh) * QW_ATTN_D + simd_gid * QW_ATTN_PER_THREAD;
#pragma unroll
        for (int i = 0; i < QW_ATTN_PER_THREAD; ++i) op[i] = acc[i];
    }
}

/* ---- the engram layer ------------------------------------------------------ */

/* Per stream: g = key.query / sqrt(H), signed square root, sigmoid; the
 * stream's gated value is that times the shared value.  One simdgroup per
 * (row, stream). */
kernel void qw_ple_gate(
    device const float *keyn  [[buffer(0)]],   /* [rows, S*H] */
    device const float *qn    [[buffer(1)]],   /* [rows, S*H] */
    device const float *value [[buffer(2)]],   /* [rows, H] */
    device       float *gv    [[buffer(3)]],   /* [rows, S*H] */
    constant qw_hc_args &a    [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint r = tgid, s = sgid;
    if (s >= a.S) return;
    const ulong base = (ulong)r * a.S * a.H + (ulong)s * a.H;
    float dot = 0.0f;
    for (uint i = lane; i < a.H; i += 32) dot = fma(keyn[base + i], qn[base + i], dot);
    dot = simd_sum(dot);
    float g = dot / sqrt(float(a.H));
    const float mag = max(fabs(g), 1e-6f);
    g = (g < 0.0f ? -1.0f : 1.0f) * sqrt(mag);
    const float sg = qw_sigmoid(g);
    for (uint i = lane; i < a.H; i += 32) gv[base + i] = sg * value[(ulong)r * a.H + i];
}

/* Depthwise causal conv with dilation: kernel 4, dilation 3, so tap j reads
 * the input 3*(3-j) steps back.  The state holds the last nine inputs per
 * channel, oldest first.  One thread per channel, like qw_conv1d_causal_silu. */
#define QW_PLE_K   4
#define QW_PLE_DIL 3
#define QW_PLE_SL  ((QW_PLE_K - 1) * QW_PLE_DIL)   /* 9 */

struct qw_dconv_args { uint channels, rows; };

kernel void qw_conv1d_dilated_silu(
    device const float  *x     [[buffer(0)]],   /* [rows, channels] */
    device       float  *state [[buffer(1)]],   /* [SL, channels] */
    device const ushort *w     [[buffer(2)]],   /* [channels, K] bf16, tap 0 oldest */
    device       float  *y     [[buffer(3)]],   /* [rows, channels] */
    constant qw_dconv_args &a  [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= a.channels) return;
    float taps[QW_PLE_K];
#pragma unroll
    for (uint j = 0; j < QW_PLE_K; ++j) taps[j] = qw_bf16_to_f32(w[gid * QW_PLE_K + j]);
    float win[QW_PLE_SL];
#pragma unroll
    for (uint j = 0; j < QW_PLE_SL; ++j) win[j] = state[(ulong)j * a.channels + gid];

    for (uint t = 0; t < a.rows; ++t) {
        const float cur = x[(ulong)t * a.channels + gid];
        float acc = taps[QW_PLE_K - 1] * cur;
#pragma unroll
        for (uint j = 0; j < QW_PLE_K - 1; ++j)
            acc = fma(win[QW_PLE_SL - (QW_PLE_K - 1 - j) * QW_PLE_DIL], taps[j], acc);
        y[(ulong)t * a.channels + gid] = qw_silu(acc);
#pragma unroll
        for (uint j = 0; j < QW_PLE_SL - 1; ++j) win[j] = win[j + 1];
        win[QW_PLE_SL - 1] = cur;
    }
#pragma unroll
    for (uint j = 0; j < QW_PLE_SL; ++j) state[(ulong)j * a.channels + gid] = win[j];
}
