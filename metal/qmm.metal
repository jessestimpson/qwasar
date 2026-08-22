/* Quantised matmul: the prefill counterpart to qw_qmv_q4_g64.
 *
 * Decode reads every weight once per token and is bandwidth-bound, so the
 * matvec kernel is the right shape there.  Prefill has many tokens sharing the
 * same weights, and reusing them is the whole game: dispatching the matvec once
 * per token re-reads all 15 GB of weights N times, which is why prefill was no
 * faster than decode before this kernel existed.
 *
 * Each threadgroup computes a BM x BN block of the output -- BM tokens by BN
 * weight rows -- walking K in steps of BK, with both operand tiles staged in
 * threadgroup memory so a weight block is dequantised once and reused across
 * the whole token tile.
 *
 * The inner product runs on the GPU's 8x8 matrix units.  A register-tiled
 * version of this kernel held 1.3 TFLOP/s against a measured machine peak of
 * 3.33: with a 4x4 register tile it issues eight threadgroup loads for every
 * sixteen fused multiply-adds, and no rebalancing fixed that -- larger register
 * tiles starved the threadgroup of threads, smaller K steps multiplied the
 * barrier count.  simdgroup_multiply_accumulate performs an entire 8x8x8
 * product per instruction, which removes the load-to-arithmetic ratio as the
 * limit rather than tuning around it.
 *
 * Bs is stored K-major, [BK][BN], which is exactly B's natural [K][N] layout,
 * so its fragments load directly.  As is stored M-major, [BM][BK], which is A's
 * natural layout for the same reason -- neither fragment load has to transpose.
 *
 * (An earlier register-tiled version stored As K-major so the inner loop could
 * read float4 runs.  That reason disappeared with the matrix units, and keeping
 * it cost twice over: a transposing fragment load, and a strided threadgroup
 * write during staging.) */

kernel void qw_qmm_q4_g64(
    device const uint    *w       [[buffer(0)]],   /* [n, k/8]  packed nibbles */
    device const ushort  *scales  [[buffer(1)]],   /* [n, k/64] bf16 */
    device const ushort  *biases  [[buffer(2)]],   /* [n, k/64] bf16 */
    device const float   *x       [[buffer(3)]],   /* [rows, k] */
    device       float   *y       [[buffer(4)]],   /* [rows, n] */
    constant qw_matmul_args &a    [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]])
{
    /* One pool, used as the two operand tiles during the K loop and then as the
     * output tile once the loop is done.
     *
     * Operands are half, accumulators stay float.  Dequantised 4-bit weights
     * and post-norm activations both sit far inside half's range, and halving
     * the tiles halves the threadgroup traffic in the inner loop, which is
     * where this kernel spends its time.  The matrix units themselves are not
     * meaningfully faster for half on this hardware -- measured 16.5 against
     * 15.6 -- so the win is bandwidth, not arithmetic. */
    threadgroup float pool[QW_QMM_POOL_F];
    threadgroup half *As = (threadgroup half *)pool;                     /* [BM][BK] */
    threadgroup half *Bs = (threadgroup half *)pool + QW_QMM_BK * QW_QMM_BM;

    const uint words  = a.k / QW_QPER_WORD;
    const uint groups = a.k / QW_QGROUP;

    const uint row0 = tgid.y * QW_QMM_BM;   /* first token in this block */
    const uint col0 = tgid.x * QW_QMM_BN;   /* first weight row in this block */

    /* This simdgroup's corner of the output tile. */
    const uint sg_m = sgid / QW_QMM_SG_N;
    const uint sg_n = sgid % QW_QMM_SG_N;
    const uint m_base = sg_m * (QW_QMM_BM / QW_QMM_SG_M);
    const uint n_base = sg_n * (QW_QMM_BN / QW_QMM_SG_N);

    simdgroup_float8x8 acc[QW_QMM_FRAG_M][QW_QMM_FRAG_N];
#pragma unroll
    for (uint i = 0; i < QW_QMM_FRAG_M; ++i)
#pragma unroll
        for (uint j = 0; j < QW_QMM_FRAG_N; ++j)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint k0 = 0; k0 < a.k; k0 += QW_QMM_BK) {

        /* Activations.  Consecutive threads take consecutive k for one token,
         * so each group of 32 reads one contiguous 128-byte span of x. */
        for (uint idx = tid; idx < QW_QMM_BM * QW_QMM_BK; idx += QW_QMM_THREADS) {
            const uint kk = idx % QW_QMM_BK;
            const uint mm = idx / QW_QMM_BK;
            const uint gm = row0 + mm;
            /* Consecutive threads take consecutive k for one token, so both the
             * global read and the threadgroup write run contiguously. */
            As[mm * QW_QMM_BK + kk] = (gm < a.rows)
                                    ? half(x[(ulong)gm * a.k + k0 + kk]) : half(0);
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
                Bs[(wk * QW_QPER_WORD + j) * QW_QMM_BN + nn] =
                    half(fma(sc, float((ww >> (4 * j)) & 0xF), bi));
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint ks = 0; ks < QW_QMM_BK; ks += QW_SG_TILE) {
            simdgroup_half8x8 af[QW_QMM_FRAG_M], bf[QW_QMM_FRAG_N];

#pragma unroll
            for (uint i = 0; i < QW_QMM_FRAG_M; ++i)
                simdgroup_load(af[i],
                               As + (m_base + i * QW_SG_TILE) * QW_QMM_BK + ks,
                               QW_QMM_BK, 0, /*transpose=*/false);
#pragma unroll
            for (uint j = 0; j < QW_QMM_FRAG_N; ++j)
                simdgroup_load(bf[j], Bs + ks * QW_QMM_BN + n_base + j * QW_SG_TILE,
                               QW_QMM_BN, 0, /*transpose=*/false);

#pragma unroll
            for (uint i = 0; i < QW_QMM_FRAG_M; ++i)
#pragma unroll
                for (uint j = 0; j < QW_QMM_FRAG_N; ++j)
                    simdgroup_multiply_accumulate(acc[i][j], af[i], bf[j], acc[i][j]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    /* Reuse the operand pool as the output tile.  Every simdgroup has finished
     * reading it above, and the barrier ending the last K step is what makes
     * that safe. */
    threadgroup float *Cs = pool;    /* [BM][BN] */

#pragma unroll
    for (uint i = 0; i < QW_QMM_FRAG_M; ++i)
#pragma unroll
        for (uint j = 0; j < QW_QMM_FRAG_N; ++j)
            simdgroup_store(acc[i][j],
                            Cs + (m_base + i * QW_SG_TILE) * QW_QMM_BN
                               + n_base + j * QW_SG_TILE,
                            QW_QMM_BN, 0, /*transpose=*/false);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* Ragged tiles are handled here rather than by the fragment stores, which
     * always write a full 8x8. */
    for (uint idx = tid; idx < QW_QMM_BM * QW_QMM_BN; idx += QW_QMM_THREADS) {
        const uint mm = idx / QW_QMM_BN;
        const uint nn = idx % QW_QMM_BN;
        const uint gm = row0 + mm, gn = col0 + nn;
        if (gm < a.rows && gn < a.n) y[(ulong)gm * a.n + gn] = Cs[idx];
    }
}
