/* Quantised matmul: the prefill counterpart to qw_qmv_q4_g64.
 *
 * Decode reads every weight once per token and is bandwidth-bound, so the
 * matvec kernel is the right shape there.  Prefill has many tokens sharing the
 * same weights, and reusing them is the whole game: dispatching the matvec once
 * per token re-reads all 15 GB of weights N times, which is why prefill was no
 * faster than decode before this kernel existed.
 *
 * Blocking:
 *
 *   Each threadgroup computes a BM x BN block of the output -- BM tokens by BN
 *   weight rows -- walking K in steps of BK.  Both operand tiles are staged in
 *   threadgroup memory, so a weight block is dequantised once and then reused
 *   by all BM tokens.  Each of the 256 threads keeps a TM x TN register tile,
 *   giving 16 fused multiply-adds per 8 threadgroup reads in the inner loop.
 *
 *   Tiles are stored K-major ([BK][BM], [BK][BN]) rather than row-major.  That
 *   makes the inner loop read runs of consecutive floats: lanes sharing a token
 *   index broadcast from one address, and lanes walking weight rows read
 *   consecutive addresses instead of colliding on one bank.
 *
 * Arithmetic intensity here is roughly 900 flop per byte of weight, so unlike
 * the matvec this kernel is compute-bound and the dequantisation cost is
 * amortised across the token tile. */

/* Tiling comes from qwasar_gpu.h, injected as preprocessor macros when the
 * library is compiled, so the host's dispatch geometry cannot drift from the
 * kernel's blocking.  The fallbacks below exist only so `make check-metal` can
 * compile this file standalone; they are never used by the engine. */
#ifndef QW_QMM_BM
#define QW_QMM_BM 64
#define QW_QMM_BN 64
#define QW_QMM_BK 32
#define QW_QMM_TM 4
#define QW_QMM_TN 4
#endif
#define QW_QMM_THREADS ((QW_QMM_BM / QW_QMM_TM) * (QW_QMM_BN / QW_QMM_TN))

kernel void qw_qmm_q4_g64(
    device const uint    *w       [[buffer(0)]],   /* [n, k/8]  packed nibbles */
    device const ushort  *scales  [[buffer(1)]],   /* [n, k/64] bf16 */
    device const ushort  *biases  [[buffer(2)]],   /* [n, k/64] bf16 */
    device const float   *x       [[buffer(3)]],   /* [rows, k] */
    device       float   *y       [[buffer(4)]],   /* [rows, n] */
    constant qw_matmul_args &a    [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])
{
    threadgroup float As[QW_QMM_BK][QW_QMM_BM];
    threadgroup float Bs[QW_QMM_BK][QW_QMM_BN];

    const uint words  = a.k / QW_QPER_WORD;
    const uint groups = a.k / QW_QGROUP;

    const uint row0 = tgid.y * QW_QMM_BM;   /* first token in this block */
    const uint col0 = tgid.x * QW_QMM_BN;   /* first weight row in this block */

    const uint tm = tid / (QW_QMM_BN / QW_QMM_TN);   /* 0..15, token sub-tile */
    const uint tn = tid % (QW_QMM_BN / QW_QMM_TN);   /* 0..15, weight sub-tile */

    float acc[QW_QMM_TM][QW_QMM_TN];
#pragma unroll
    for (uint i = 0; i < QW_QMM_TM; ++i)
#pragma unroll
        for (uint j = 0; j < QW_QMM_TN; ++j) acc[i][j] = 0.0f;

    for (uint k0 = 0; k0 < a.k; k0 += QW_QMM_BK) {

        /* Activations.  Consecutive threads take consecutive k for one token,
         * so each group of 32 reads one contiguous 128-byte span of x. */
        for (uint idx = tid; idx < QW_QMM_BM * QW_QMM_BK; idx += QW_QMM_THREADS) {
            const uint kk = idx % QW_QMM_BK;
            const uint mm = idx / QW_QMM_BK;
            const uint gm = row0 + mm;
            As[kk][mm] = (gm < a.rows) ? x[(ulong)gm * a.k + k0 + kk] : 0.0f;
        }

        /* Weights, one packed word (8 values) per thread.  A word never spans
         * two quantisation groups because 8 divides 64, so one scale and one
         * bias cover the whole word. */
        for (uint idx = tid; idx < QW_QMM_BN * (QW_QMM_BK / QW_QPER_WORD);
             idx += QW_QMM_THREADS) {
            const uint nn = idx / (QW_QMM_BK / QW_QPER_WORD);
            const uint wk = idx % (QW_QMM_BK / QW_QPER_WORD);
            const uint gn = col0 + nn;
            const uint gk = k0 + wk * QW_QPER_WORD;

            uint  ww = 0;
            float sc = 0.0f, bi = 0.0f;
            if (gn < a.n) {
                ww = w[(ulong)gn * words + gk / QW_QPER_WORD];
                const uint g = gk / QW_QGROUP;
                sc = qw_bf16_to_f32(scales[(ulong)gn * groups + g]);
                bi = qw_bf16_to_f32(biases[(ulong)gn * groups + g]);
            }
#pragma unroll
            for (uint j = 0; j < QW_QPER_WORD; ++j)
                Bs[wk * QW_QPER_WORD + j][nn] = fma(sc, float((ww >> (4 * j)) & 0xF), bi);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < QW_QMM_BK; ++kk) {
            float av[QW_QMM_TM], bv[QW_QMM_TN];
#pragma unroll
            for (uint i = 0; i < QW_QMM_TM; ++i) av[i] = As[kk][tm * QW_QMM_TM + i];
#pragma unroll
            for (uint j = 0; j < QW_QMM_TN; ++j) bv[j] = Bs[kk][tn * QW_QMM_TN + j];
#pragma unroll
            for (uint i = 0; i < QW_QMM_TM; ++i)
#pragma unroll
                for (uint j = 0; j < QW_QMM_TN; ++j)
                    acc[i][j] = fma(av[i], bv[j], acc[i][j]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = 0; i < QW_QMM_TM; ++i) {
        const uint gm = row0 + tm * QW_QMM_TM + i;
        if (gm >= a.rows) continue;
        for (uint j = 0; j < QW_QMM_TN; ++j) {
            const uint gn = col0 + tn * QW_QMM_TN + j;
            if (gn < a.n) y[(ulong)gm * a.n + gn] = acc[i][j];
        }
    }
}
