/* Quantised mat-vec / mat-mul: the decode hot path.
 *
 * Decoding one token reads every weight in the model exactly once, so this
 * kernel is a bandwidth problem, not an arithmetic one: ~14 GB moved per token
 * against ~120 GB/s.  Everything here is arranged around making the weight
 * reads coalesce and never re-reading a byte.
 *
 * It dispatches one threadgroup row per token, which is right for decode and
 * badly wrong for prefill: N tokens would re-read the weights N times.  Prefill
 * uses qw_qmm_q4_g64 instead, which tiles over tokens. */

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
    constant qw_matmul_args &a    [[buffer(5)]],
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

/* Batched quantised matvec: B token rows against one pass over the weights.
 *
 * This is the kernel speculative decoding turns on, and neither existing path
 * could do its job.  Verifying B drafted tokens is only worth anything if it
 * costs less than B decode steps, and measured on gate_proj (5120 x 17408):
 *
 *   rows      qmv        qmm
 *      1   0.558 ms   4.508 ms
 *      4   2.092 ms   4.525 ms
 *      8   4.164 ms   4.576 ms
 *
 * qmv is exactly linear -- one threadgroup per token, every weight re-read per
 * token -- so it turns a width-8 verify into eight decode steps and there is no
 * speedup left to divide.  qmm is flat because it pads its token tile to
 * QW_QMM_BM and does 64 tokens of arithmetic whatever it was asked for: at 2.53
 * TFLOP/s it is running at full speed, on work nobody wanted.
 *
 * The gap between them is the whole opportunity.  One pass over this
 * projection's 50.1 MB at the measured 90 GB/s is 0.558 ms, so the ideal
 * width-8 kernel costs what a single token costs today.
 *
 * Getting there does not need the matrix units.  Arithmetic intensity here is
 * 4B FLOP per weight byte (2 FLOP per token per 4-bit weight), and this machine
 * needs 37 FLOP/byte before scalar fused multiply-add stops keeping up with
 * memory -- so anything up to B = 9 stays bandwidth-bound with ordinary FMAs.
 * At B = 8 that is 0.43 ms of arithmetic hiding under 0.558 ms of loads.  The
 * matrix units only start to matter once there are enough tokens to saturate
 * them, which is what QW_QMM_MIN_ROWS is for.
 *
 * So: one threadgroup per block of output rows, the token loop innermost, and a
 * weight word dequantised once and spent across every token before it is
 * dropped.  The token loop is unrolled to a compile-time B with dead lanes
 * predicated off rather than bounded dynamically -- a dynamic bound would push
 * `acc` out of registers and into scratch, which costs far more than the
 * wasted multiplies it saves. */

#ifndef QW_QMVB_B
#define QW_QMVB_B    8
#define QW_QMVB_ROWS 4
#endif

kernel void qw_qmvb_q4_g64(
    device const uint    *w       [[buffer(0)]],   /* [n, k/8]  packed nibbles */
    device const ushort  *scales  [[buffer(1)]],   /* [n, k/64] bf16 */
    device const ushort  *biases  [[buffer(2)]],   /* [n, k/64] bf16 */
    device const float   *x       [[buffer(3)]],   /* [rows, k] */
    device       float   *y       [[buffer(4)]],   /* [rows, n] */
    constant qw_matmul_args &a    [[buffer(5)]],
    uint3 tgid  [[threadgroup_position_in_grid]],
    uint  sgid  [[simdgroup_index_in_threadgroup]],
    uint  nsg   [[simdgroups_per_threadgroup]],
    uint  lane  [[thread_index_in_simdgroup]])
{
    const uint words  = a.k / QW_QPER_WORD;
    const uint groups = a.k / QW_QGROUP;

    const uint row0 = (tgid.x * nsg + sgid) * QW_QMVB_ROWS;
    if (row0 >= a.n) return;

    /* Out-of-range output rows and dead token lanes are handled by CLAMPING
     * their indices to something valid and discarding the result at the store,
     * never by bounding a loop at runtime.  Every index into `acc` and `wd`
     * below is therefore a compile-time constant, which is what keeps them in
     * registers: a dynamic bound makes the compiler spill both arrays to
     * scratch memory, and the first version of this kernel measured 3.6 ms
     * against qmv's 0.56 for exactly that reason -- six times slower while
     * doing strictly less work. */
    float acc[QW_QMVB_ROWS][QW_QMVB_B];
#pragma unroll
    for (uint r = 0; r < QW_QMVB_ROWS; ++r)
#pragma unroll
        for (uint b = 0; b < QW_QMVB_B; ++b) acc[r][b] = 0.0f;

    /* Lane l walks words l, l+32, ...  Consecutive lanes hold consecutive words
     * of the same weight row, so each iteration is one coalesced 128-byte fetch
     * per row -- the same access pattern the single-token kernel above relies
     * on, which is why this one inherits its bandwidth. */
    for (uint wi = lane; wi < words; wi += 32) {
        const uint g = wi / QW_WORDS_PER_GROUP;

        float wd[QW_QMVB_ROWS][QW_QPER_WORD];
#pragma unroll
        for (uint r = 0; r < QW_QMVB_ROWS; ++r) {
            const uint  n  = min(row0 + r, a.n - 1);
            const uint  ww = w[(ulong)n * words + wi];
            const float sc = qw_bf16_to_f32(scales[(ulong)n * groups + g]);
            const float bi = qw_bf16_to_f32(biases[(ulong)n * groups + g]);
#pragma unroll
            for (uint j = 0; j < QW_QPER_WORD; ++j)
                wd[r][j] = fma(sc, float((ww >> (4 * j)) & 0xF), bi);
        }

#pragma unroll
        for (uint b = 0; b < QW_QMVB_B; ++b) {
            /* A dead token lane re-reads a live token's activations, which is a
             * cache hit, and computes a real dot product that is thrown away. */
            device const float *xv = x + (ulong)min(b, a.rows - 1) * a.k
                                       + wi * QW_QPER_WORD;
            float xs[QW_QPER_WORD];
#pragma unroll
            for (uint j = 0; j < QW_QPER_WORD; ++j) xs[j] = xv[j];

#pragma unroll
            for (uint r = 0; r < QW_QMVB_ROWS; ++r)
#pragma unroll
                for (uint j = 0; j < QW_QPER_WORD; ++j)
                    acc[r][b] = fma(wd[r][j], xs[j], acc[r][b]);
        }
    }

#pragma unroll
    for (uint r = 0; r < QW_QMVB_ROWS; ++r) {
#pragma unroll
        for (uint b = 0; b < QW_QMVB_B; ++b) {
            const float v = simd_sum(acc[r][b]);
            if (lane == 0 && b < a.rows && row0 + r < a.n)
                y[(ulong)b * a.n + row0 + r] = v;
        }
    }
}
