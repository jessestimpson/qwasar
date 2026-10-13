/* Full attention over the KV cache -- 16 of the model's 64 layers.
 *
 * Cache layout is head-major, [kv_heads, max_ctx, head_dim], so one head's keys
 * are contiguous.  Attention re-reads every key on every token, while writing
 * touches one position, so the read pattern is the one worth optimising for.
 *
 * Work assignment follows the shape that is standard for vector SDPA and is a
 * good fit here: one threadgroup per (token, query head), 32 simdgroups of 32
 * lanes.  Each lane owns head_dim/32 = 8 dimensions of the query and of the
 * output accumulator; each simdgroup strides through the keys 32 apart.  A
 * score is therefore one simd_sum across the lanes holding that key's
 * dimensions, and the running softmax lives in registers.
 *
 * The softmax is online (flash-style): scores are never materialised, so cost
 * is independent of context length beyond the unavoidable key traffic. */

#define QW_ATTN_D           256
#define QW_ATTN_LANES       32
#define QW_ATTN_SIMDS       32
#define QW_ATTN_PER_THREAD  (QW_ATTN_D / QW_ATTN_LANES)   /* 8 */

struct qw_attn_args {
    uint  rows;       /* query tokens in this step */
    uint  q_heads;
    uint  kv_heads;
    uint  gqa;        /* q_heads / kv_heads */
    uint  max_ctx;    /* cache stride */
    uint  base_pos;   /* cache position of row 0 */
    float scale;
};

kernel void qw_attn_decode(
    device const float   *q   [[buffer(0)]],   /* [rows, q_heads, D] fp32 */
    device const half    *kc  [[buffer(1)]],   /* [kv_heads, max_ctx, D] fp16 */
    device const half    *vc  [[buffer(2)]],   /* [kv_heads, max_ctx, D] fp16 */
    device       float   *out [[buffer(3)]],   /* [rows, q_heads, D] fp32 */
    constant qw_attn_args &a  [[buffer(4)]],
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

    /* Causal: this token sees every cached key up to and including its own
     * position.  With rows > 1 that makes the same kernel do prefill, one
     * query at a time -- correct, if not yet tiled. */
    const int n_keys = (int)(a.base_pos + row) + 1;

    const device float *qp = q + ((ulong)row * a.q_heads + qh) * QW_ATTN_D
                               + simd_lid * QW_ATTN_PER_THREAD;
    const device half  *kp = kc + ((ulong)kvh * a.max_ctx + simd_gid) * QW_ATTN_D
                               + simd_lid * QW_ATTN_PER_THREAD;
    const device half  *vp = vc + ((ulong)kvh * a.max_ctx + simd_gid) * QW_ATTN_D
                               + simd_lid * QW_ATTN_PER_THREAD;

    float qv[QW_ATTN_PER_THREAD];
    float acc[QW_ATTN_PER_THREAD];
#pragma unroll
    for (int i = 0; i < QW_ATTN_PER_THREAD; ++i) {
        qv[i]  = a.scale * qp[i];   /* fold the 1/sqrt(d) in once, not per key */
        acc[i] = 0.0f;
    }

    /* Not -INFINITY: fast-math permits assuming infinities never occur.
     * -FLT_MAX gives the same behaviour, since the first rescale factor
     * exp(-FLT_MAX - score) underflows to zero. */
    float run_max = -FLT_MAX;
    float run_sum = 0.0f;

    for (int t = (int)simd_gid; t < n_keys; t += QW_ATTN_SIMDS) {
        float score = 0.0f;
#pragma unroll
        for (int i = 0; i < QW_ATTN_PER_THREAD; ++i)
            score = fma(qv[i], float(kp[i]), score);
        score = simd_sum(score);

        /* Online softmax: rescale what we have rather than revisiting it. */
        const float new_max = max(run_max, score);
        const float factor  = exp(run_max - new_max);
        const float w       = exp(score - new_max);

        run_max = new_max;
        run_sum = run_sum * factor + w;

#pragma unroll
        for (int i = 0; i < QW_ATTN_PER_THREAD; ++i)
            acc[i] = fma(acc[i], factor, w * float(vp[i]));

        kp += QW_ATTN_SIMDS * QW_ATTN_D;
        vp += QW_ATTN_SIMDS * QW_ATTN_D;
    }

    /* Combine the simdgroups' partial softmaxes. */
    if (simd_lid == 0) {
        tg_max[simd_gid] = run_max;
        tg_sum[simd_gid] = run_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float part_max = tg_max[simd_lid];
    const float glob_max = simd_max(part_max);
    const float rescale  = exp(part_max - glob_max);
    const float glob_sum = simd_sum(tg_sum[simd_lid] * rescale);

    /* Each lane holds dims [simd_lid*8, +8) of its own simdgroup's partial
     * output; the reduction needs to sum across simdgroups instead.  Bouncing
     * one dimension at a time through threadgroup memory transposes lane and
     * simdgroup indices so simd_sum reduces along the right axis. */
    for (int i = 0; i < QW_ATTN_PER_THREAD; ++i) {
        tg_out[simd_lid * QW_ATTN_LANES + simd_gid] = acc[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float v = simd_sum(tg_out[simd_gid * QW_ATTN_LANES + simd_lid] * rescale);
        acc[i] = glob_sum == 0.0f ? v : v / glob_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (simd_lid == 0) {
        device float *op = out + ((ulong)row * a.q_heads + qh) * QW_ATTN_D
                               + simd_gid * QW_ATTN_PER_THREAD;
#pragma unroll
        for (int i = 0; i < QW_ATTN_PER_THREAD; ++i) op[i] = acc[i];
    }
}

/* Appends this step's keys and values to the cache, converting to fp16.
 *
 * fp16 halves the traffic the attention kernel above is bound by, and the
 * values being stored are post-norm and post-rope, so their range is tame. */
struct qw_kv_args {
    uint rows;
    uint kv_heads;
    uint head_dim;
    uint max_ctx;
    uint base_pos;
};

kernel void qw_kv_write(
    device const float  *k  [[buffer(0)]],   /* [rows, kv_heads, head_dim] */
    device const float  *v  [[buffer(1)]],
    device       half   *kc [[buffer(2)]],   /* [kv_heads, max_ctx, head_dim] */
    device       half   *vc [[buffer(3)]],
    constant qw_kv_args &a  [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint i = gid.x;
    if (i >= a.head_dim) return;

    const uint rh  = gid.y;                 /* flattened (row, kv head) */
    const uint row = rh / a.kv_heads;
    const uint h   = rh % a.kv_heads;

    const ulong src = (ulong)rh * a.head_dim + i;
    const ulong dst = ((ulong)h * a.max_ctx + a.base_pos + row) * a.head_dim + i;
    kc[dst] = half(k[src]);
    vc[dst] = half(v[src]);
}

/* ---- decode at depth: one read of each key for its whole GQA group ---------
 *
 * qw_attn_decode gives every query head its own threadgroup, so the six query
 * heads that share a KV head each walk that head's whole cache, and one token
 * is 24 threadgroups.  The repeated reads mostly hit cache -- it sustains
 * ~50-60 GB/s of unique KV -- but at depth it is still the larger share of a
 * token's attention: on an M4, 40 ms for the 16 layers at 32K, 87 ms at 64K.
 *
 * Here a threadgroup takes one KV head, all QW_ATTN_GQA query heads that read
 * it, and one slice of the keys; each of its QW_ATTN_GS simdgroups walks the
 * slice 32 dims to a lane as before, scoring a key against all the heads once
 * it is loaded.  Every simdgroup leaves a partial softmax -- its max, its sum,
 * its unnormalised output -- and qw_attn_merge combines them.  Splitting the
 * keys is what fills the GPU: four KV heads alone are four threadgroups.
 *
 * Measured on an M4, 16 layers, one token: 34 ms against 40 at 32K, 64 against
 * 87 at 64K -- but slower below ~20K keys (5.3 against 3.0 ms at 2K), where
 * the merge pass and six reductions a key cost more than the reads saved;
 * hence QW_ATTN_SPLIT_MIN_KEYS. */

/* Injected from qwasar_gpu.h by the runtime compile; the fallbacks serve
 * only `make check-metal`. */
#ifndef QW_ATTN_GQA
#define QW_ATTN_GQA 6
#define QW_ATTN_GS  8
#endif

struct qw_attn_split_args {
    uint  rows, q_heads, kv_heads, max_ctx, base_pos;
    uint  splits;      /* key slices */
    uint  per_split;   /* keys per slice */
    float scale;
};

kernel void qw_attn_decode_gqa(
    device const float   *q     [[buffer(0)]],   /* [rows, q_heads, D] */
    device const half    *kc    [[buffer(1)]],
    device const half    *vc    [[buffer(2)]],
    device       float   *pacc  [[buffer(3)]],   /* [rows, q_heads, parts, D] */
    device       float   *pstat [[buffer(4)]],   /* [rows, q_heads, parts, 2]: max, sum */
    constant qw_attn_split_args &a [[buffer(5)]],
    uint2 tgid     [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
    const uint split = tgid.x;
    const uint row = tgid.y / a.kv_heads, kvh = tgid.y % a.kv_heads;
    const uint parts = a.splits * QW_ATTN_GS;
    const uint part = split * QW_ATTN_GS + simd_gid;
    const int  last = (int)(a.base_pos + row);                  /* causal */
    const int  k_begin = (int)(split * a.per_split);
    const int  k_end = min((int)(k_begin + a.per_split), last + 1);

    float qv[QW_ATTN_GQA][QW_ATTN_PER_THREAD], acc[QW_ATTN_GQA][QW_ATTN_PER_THREAD];
    float run_max[QW_ATTN_GQA], run_sum[QW_ATTN_GQA];
#pragma unroll
    for (int h = 0; h < QW_ATTN_GQA; ++h) {
        const device float *qp = q + ((ulong)row * a.q_heads + kvh * QW_ATTN_GQA + h) * QW_ATTN_D
                                   + simd_lid * QW_ATTN_PER_THREAD;
#pragma unroll
        for (int i = 0; i < QW_ATTN_PER_THREAD; ++i) { qv[h][i] = a.scale * qp[i]; acc[h][i] = 0.0f; }
        run_max[h] = -FLT_MAX;
        run_sum[h] = 0.0f;
    }

    for (int t = k_begin + (int)simd_gid; t < k_end; t += QW_ATTN_GS) {
        const device half *kp = kc + ((ulong)kvh * a.max_ctx + t) * QW_ATTN_D + simd_lid * QW_ATTN_PER_THREAD;
        const device half *vp = vc + ((ulong)kvh * a.max_ctx + t) * QW_ATTN_D + simd_lid * QW_ATTN_PER_THREAD;
        float kv[QW_ATTN_PER_THREAD], vv[QW_ATTN_PER_THREAD];
#pragma unroll
        for (int i = 0; i < QW_ATTN_PER_THREAD; ++i) { kv[i] = float(kp[i]); vv[i] = float(vp[i]); }
#pragma unroll
        for (int h = 0; h < QW_ATTN_GQA; ++h) {
            float score = 0.0f;
#pragma unroll
            for (int i = 0; i < QW_ATTN_PER_THREAD; ++i) score = fma(qv[h][i], kv[i], score);
            score = simd_sum(score);
            const float new_max = max(run_max[h], score);
            const float factor  = exp(run_max[h] - new_max);
            const float w       = exp(score - new_max);
            run_max[h] = new_max;
            run_sum[h] = run_sum[h] * factor + w;
#pragma unroll
            for (int i = 0; i < QW_ATTN_PER_THREAD; ++i) acc[h][i] = fma(acc[h][i], factor, w * vv[i]);
        }
    }

#pragma unroll
    for (int h = 0; h < QW_ATTN_GQA; ++h) {
        const ulong idx = ((ulong)row * a.q_heads + kvh * QW_ATTN_GQA + h) * parts + part;
        device float *op = pacc + idx * QW_ATTN_D + simd_lid * QW_ATTN_PER_THREAD;
#pragma unroll
        for (int i = 0; i < QW_ATTN_PER_THREAD; ++i) op[i] = acc[h][i];
        if (simd_lid == 0) { pstat[idx * 2] = run_max[h]; pstat[idx * 2 + 1] = run_sum[h]; }
    }
}

/* Combines the split kernels' partial softmaxes (qw_attn_decode_gqa, and on
 * an M5 qw_attn_split_tensor and qw_attn_masked_split_tensor): one
 * threadgroup per (row, query head), a thread per output dimension.  A part
 * that saw no keys has sum 0 and drops out.
 *
 * Each part's weight exp(m_p - m) / sum is found once, by a thread a part, into
 * threadgroup memory; then every dimension's sum over the parts is loads that
 * do not wait on each other.  Walking the parts in one thread, max, then sum
 * and output, made each step wait on a device load: 8 us of a Flash-Next
 * layer's 29 at 37 parts. */
#define QW_MERGE_MAX 256   /* parts a pass; more take several */

inline float qw_attn_merge_one(device const float *pacc, device const float *pstat,
                               uint parts, uint tgid, uint d, threadgroup float *w,
                               threadgroup float *red) {
    const device float *st = pstat + (ulong)tgid * parts * 2;
    const device float *ac = pacc + (ulong)tgid * parts * QW_ATTN_D + d;
    /* The max over live parts, d threads striding the parts. */
    float m = -FLT_MAX;
    for (uint p = d; p < parts; p += QW_ATTN_D) if (st[p * 2 + 1] > 0.0f) m = max(m, st[p * 2]);
    m = simd_max(m);
    if (d % 32 == 0) red[d / 32] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m = red[0];
    for (uint i = 1; i < QW_ATTN_D / 32; ++i) m = max(m, red[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sum = 0.0f, o = 0.0f;
    for (uint p0 = 0; p0 < parts; p0 += QW_MERGE_MAX) {
        const uint n = min(parts - p0, (uint)QW_MERGE_MAX);
        for (uint p = d; p < n; p += QW_ATTN_D) {
            const float l = st[(p0 + p) * 2 + 1];
            w[p] = l > 0.0f ? exp(st[(p0 + p) * 2] - m) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint p = 0; p < n; ++p) {
            sum += st[(p0 + p) * 2 + 1] * w[p];
            o += ac[(ulong)(p0 + p) * QW_ATTN_D] * w[p];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return sum > 0.0f ? o / sum : 0.0f;
}

kernel void qw_attn_merge(
    device const float *pacc  [[buffer(0)]],
    device const float *pstat [[buffer(1)]],
    device       float *out   [[buffer(2)]],   /* [rows, q_heads, D] */
    constant uint      &parts [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint d    [[thread_index_in_threadgroup]])
{
    threadgroup float w[QW_MERGE_MAX], red[QW_ATTN_D / 32];
    out[(ulong)tgid * QW_ATTN_D + d] = qw_attn_merge_one(pacc, pstat, parts, tgid, d, w, red);
}

/* The same with the output gate applied as it is written, out = merged *
 * sigmoid(gate) -- qw_mul_sigmoid's step folded in. */
kernel void qw_attn_merge_gated(
    device const float *pacc  [[buffer(0)]],
    device const float *pstat [[buffer(1)]],
    device       float *out   [[buffer(2)]],   /* [rows, q_heads, D] */
    constant uint      &parts [[buffer(3)]],
    device const float *gate  [[buffer(4)]],   /* [rows, q_heads, D] */
    uint tgid [[threadgroup_position_in_grid]],
    uint d    [[thread_index_in_threadgroup]])
{
    threadgroup float w[QW_MERGE_MAX], red[QW_ATTN_D / 32];
    const ulong i = (ulong)tgid * QW_ATTN_D + d;
    out[i] = qw_attn_merge_one(pacc, pstat, parts, tgid, d, w, red) * qw_sigmoid(gate[i]);
}
