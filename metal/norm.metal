/* RMS normalisation.
 *
 * One threadgroup per row, so the same kernel serves every place the model
 * normalises: hidden states (dim 5120, one row per token), per-head Q/K norms
 * (dim 256, one row per token-head), and the gated-delta output norm
 * (dim 128, one row per token-value-head).
 *
 * `out_scale` exists because the gated-delta path wants an L2 norm rather than
 * an RMS norm, and the two differ only by a constant:
 *
 *     l2norm(x) = x / ||x||  =  rms_norm(x) / sqrt(dim)
 *
 * so `q = l2norm(q) / sqrt(Dk)` is expressed as out_scale = 1/Dk, exactly the
 * `inv_scale**2 * rms_norm(...)` the reference implementation writes. */

struct qw_norm_args {
    uint  dim;
    uint  rows;
    float eps;
    float out_scale;   /* applied after normalisation, before the weight */
    uint  has_weight;  /* 0 for the unweighted L2 normalisations */
};

/* Sum of squares across a threadgroup: simd reduction, then one simdgroup
 * folds the per-simdgroup partials. */
static inline float qw_row_sumsq(device const float *xr, uint dim,
                                 uint tid, uint ntg, uint sgid, uint nsg, uint lane,
                                 threadgroup float *partial) {
    float ss = 0.0f;
    for (uint i = tid; i < dim; i += ntg) {
        float v = xr[i];
        ss = fma(v, v, ss);
    }
    ss = simd_sum(ss);
    if (lane == 0) partial[sgid] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0) {
        float v = (lane < nsg) ? partial[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0) partial[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return partial[0];
}

kernel void qw_rms_norm(
    device const float   *x  [[buffer(0)]],   /* [rows, dim] */
    device const ushort  *w  [[buffer(1)]],   /* [dim] bf16; ignored if !has_weight */
    device       float   *y  [[buffer(2)]],   /* [rows, dim] */
    constant qw_norm_args &a [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint ntg  [[threads_per_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint nsg  [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float partial[32];

    device const float *xr = x + (ulong)tgid * a.dim;
    device       float *yr = y + (ulong)tgid * a.dim;

    float sumsq = qw_row_sumsq(xr, a.dim, tid, ntg, sgid, nsg, lane, partial);
    const float inv = rsqrt(sumsq / float(a.dim) + a.eps) * a.out_scale;

    for (uint i = tid; i < a.dim; i += ntg) {
        float v = xr[i] * inv;
        if (a.has_weight) v *= qw_bf16_to_f32(w[i]);
        yr[i] = v;
    }
}

/* The gated-delta output norm: rms_norm(x, w) * silu(gate).
 *
 * Note which operand is gated -- silu applies to the separate `z` projection,
 * not to the normalised value.  Getting that backwards produces plausible
 * garbage rather than an obvious failure. */
kernel void qw_rms_norm_gated(
    device const float   *x    [[buffer(0)]],   /* [rows, dim] */
    device const ushort  *w    [[buffer(1)]],   /* [dim] bf16 */
    device const float   *gate [[buffer(2)]],   /* [rows, dim] */
    device       float   *y    [[buffer(3)]],   /* [rows, dim] */
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
        yr[i] = v * qw_silu(gr[i]);
    }
}

/* Two RMS norms written straight into the concatenated layout the MTP head's
 * `fc` consumes:
 *
 *     y[t] = [ norm_e(e[t]) | norm_h(h[t]) ]        each half `dim` wide
 *
 * Embedding first, hidden second.  The order is the head's, not a convention,
 * and reversing it produces a head that runs and drafts nonsense.
 *
 * Fusing the pair saves nothing but a launch; what it really buys is that the
 * concatenation has no separate existence to get wrong.  One threadgroup per
 * token, the two halves reduced independently. */
kernel void qw_rms_norm_concat(
    device const float   *e  [[buffer(0)]],   /* [rows, dim] */
    device const ushort  *we [[buffer(1)]],   /* [dim] bf16 */
    device const float   *h  [[buffer(2)]],   /* [rows, dim] */
    device const ushort  *wh [[buffer(3)]],   /* [dim] bf16 */
    device       float   *y  [[buffer(4)]],   /* [rows, 2 * dim] */
    constant qw_norm_args &a [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint ntg  [[threads_per_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint nsg  [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float partial[32];

    device       float *yr = y + (ulong)tgid * 2 * a.dim;

    for (uint half_i = 0; half_i < 2; ++half_i) {
        device const float  *xr = (half_i == 0 ? e : h) + (ulong)tgid * a.dim;
        device const ushort *wr =  half_i == 0 ? we : wh;

        float sumsq = qw_row_sumsq(xr, a.dim, tid, ntg, sgid, nsg, lane, partial);
        const float inv = rsqrt(sumsq / float(a.dim) + a.eps);

        for (uint i = tid; i < a.dim; i += ntg)
            yr[half_i * a.dim + i] = xr[i] * inv * qw_bf16_to_f32(wr[i]);

        /* partial[] is reused by the second half's reduction. */
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
