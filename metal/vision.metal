/* Kernels the vision tower needs and the text model does not.
 *
 * ---- 2D rotary position embedding -------------------------------------------
 *
 * Not the text model's RoPE, and sharing code with it would be a mistake: that
 * one is interleaved, partial, and multimodal over three axes; this one is
 * half-split, applies to the whole head, and encodes a patch's row and column.
 *
 * A head is 72 wide.  Its 36 angles are the row's 18 frequencies followed by
 * the column's 18, and the rotation pairs element i with element i + 36 -- the
 * halves of the head, not adjacent elements. */
kernel void qw_rope_2d(
    device       float *x      [[buffer(0)]],   /* strided [tokens, heads, dim] */
    device const float *angles [[buffer(1)]],   /* [tokens, dim/2] */
    constant uint4     &a      [[buffer(2)]],   /* tokens, heads, dim, token stride */
    uint3 gid [[thread_position_in_grid]])
{
    const uint half_dim = a.z / 2;
    if (gid.x >= half_dim || gid.y >= a.y || gid.z >= a.x) return;

    /* The stride is explicit because q and k live inside one fused
     * [tokens, 3, heads, dim] block, so a token's rows are not adjacent. */
    device float *h = x + (ulong)gid.z * a.w + (ulong)gid.y * a.z;
    const float ang = angles[(ulong)gid.z * half_dim + gid.x];
    const float c = precise::cos(ang), s = precise::sin(ang);

    const float lo = h[gid.x], hi = h[gid.x + half_dim];
    h[gid.x]            = lo * c - hi * s;
    h[gid.x + half_dim] = hi * c + lo * s;
}

/* Bidirectional attention over an image's patches.
 *
 * Every patch attends to every other one: no cache, no mask, no causality.
 * That makes it the simplest attention in the engine and also the only
 * quadratic one, so a threadgroup owns one (query, head) pair and streams the
 * keys past it, keeping the running softmax in registers rather than
 * materialising a row of scores.
 *
 * q, k and v arrive as one [tokens, 3, heads, dim] block, which is the layout
 * the fused qkv projection already produces. */
kernel void qw_vision_attn(
    device const float *qkv [[buffer(0)]],   /* [tokens, 3, heads, dim] */
    device       float *out [[buffer(1)]],   /* [tokens, heads, dim] */
    constant uint3     &a   [[buffer(2)]],   /* tokens, heads, dim */
    constant float     &scale [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid2 [[thread_position_in_threadgroup]],
    uint2 ntg2 [[threads_per_threadgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  nsg  [[simdgroups_per_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    const uint tid = tid2.x, ntg = ntg2.x;
    const uint tokens = a.x, heads = a.y, dim = a.z;
    const uint qi = tgid.x, hi = tgid.y;
    if (qi >= tokens || hi >= heads) return;

    threadgroup float red[64];
    threadgroup float qs[128];          /* the query, shared by every thread */
    threadgroup float acc[128];         /* the running weighted sum */
    threadgroup float st[2];            /* running max and running sum */

    const ulong qbase = ((ulong)qi * 3 + 0) * heads * dim + (ulong)hi * dim;
    for (uint i = tid; i < dim; i += ntg) { qs[i] = qkv[qbase + i]; acc[i] = 0.0f; }
    if (tid == 0) { st[0] = -3.0e38f; st[1] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint kj = 0; kj < tokens; kj++) {
        const ulong kbase = ((ulong)kj * 3 + 1) * heads * dim + (ulong)hi * dim;
        float dot = 0.0f;
        for (uint i = tid; i < dim; i += ntg) dot = fma(qs[i], qkv[kbase + i], dot);
        dot = simd_sum(dot);
        if (lane == 0) red[sgid] = dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgid == 0) {
            float v = (lane < nsg) ? red[lane] : 0.0f;
            v = simd_sum(v);
            if (lane == 0) red[0] = v * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        /* Online softmax: rescale what has been accumulated whenever a larger
         * score arrives, so no row of logits is ever stored. */
        const float score = red[0];
        const float m_old = st[0];
        const float m_new = max(m_old, score);
        const float rescale = precise::exp(m_old - m_new);
        const float w = precise::exp(score - m_new);

        const ulong vbase = ((ulong)kj * 3 + 2) * heads * dim + (ulong)hi * dim;
        for (uint i = tid; i < dim; i += ntg)
            acc[i] = fma(rescale, acc[i], w * qkv[vbase + i]);
        if (tid == 0) { st[0] = m_new; st[1] = fma(rescale, st[1], w); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inv = 1.0f / st[1];
    device float *o = out + ((ulong)qi * heads + hi) * dim;
    for (uint i = tid; i < dim; i += ntg) o[i] = acc[i] * inv;
}
