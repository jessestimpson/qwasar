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
