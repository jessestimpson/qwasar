/* Quantised mat-vec / mat-mul: the decode hot path.
 *
 * Decoding one token reads every weight in the model exactly once, so this
 * kernel is a bandwidth problem, not an arithmetic one: ~14 GB moved per token
 * against ~120 GB/s.  Everything here is arranged around making the weight
 * reads coalesce and never re-reading a byte. */

struct qw_qmv_args {
    uint k;      /* input features  */
    uint n;      /* output features */
    uint rows;   /* tokens in this step */
};

/* Output rows handled per simdgroup.  Each activation element is fetched once
 * and reused across all of them, which is what keeps the x traffic negligible
 * next to the weight traffic. */
#define QW_QMV_ROWS 4

kernel void qw_qmv_q4_g64(
    device const uint    *w       [[buffer(0)]],   /* [n, k/8]  packed nibbles */
    device const ushort  *scales  [[buffer(1)]],   /* [n, k/64] bf16 */
    device const ushort  *biases  [[buffer(2)]],   /* [n, k/64] bf16 */
    device const float   *x       [[buffer(3)]],   /* [rows, k] */
    device       float   *y       [[buffer(4)]],   /* [rows, n] */
    constant qw_qmv_args &a       [[buffer(5)]],
    uint3 tgid  [[threadgroup_position_in_grid]],
    uint  sgid  [[simdgroup_index_in_threadgroup]],
    uint  nsg   [[simdgroups_per_threadgroup]],
    uint  lane  [[thread_index_in_simdgroup]])
{
    const uint words  = a.k / QW_QPER_WORD;     /* u32 words per weight row */
    const uint groups = a.k / QW_QGROUP;        /* scale/bias entries per row */

    /* Rows are assigned per simdgroup, so every divergence test below is
     * simdgroup-uniform and simd_sum() stays legal. */
    const uint row0 = (tgid.x * nsg + sgid) * QW_QMV_ROWS;
    if (row0 >= a.n) return;

    const uint b = tgid.y;
    device const float *xv = x + (ulong)b * a.k;

    float acc[QW_QMV_ROWS] = { 0.0f, 0.0f, 0.0f, 0.0f };

    /* Lane l walks words l, l+32, l+64 ...  Consecutive lanes therefore hold
     * consecutive words of the same weight row, so each iteration is one fully
     * coalesced 128-byte fetch per row. */
    for (uint wi = lane; wi < words; wi += 32) {
        const uint g = wi / QW_WORDS_PER_GROUP;

        float xs[QW_QPER_WORD];
#pragma unroll
        for (int j = 0; j < QW_QPER_WORD; ++j) xs[j] = xv[wi * QW_QPER_WORD + j];

#pragma unroll
        for (uint r = 0; r < QW_QMV_ROWS; ++r) {
            const uint n = row0 + r;
            if (n >= a.n) break;
            const uint  ww = w[(ulong)n * words + wi];
            const float sc = qw_bf16_to_f32(scales[(ulong)n * groups + g]);
            const float bi = qw_bf16_to_f32(biases[(ulong)n * groups + g]);
            acc[r] += qw_qdot8(ww, sc, bi, xs);
        }
    }

#pragma unroll
    for (uint r = 0; r < QW_QMV_ROWS; ++r) {
        const float v = simd_sum(acc[r]);
        const uint  n = row0 + r;
        if (lane == 0 && n < a.n) y[(ulong)b * a.n + n] = v;
    }
}
