/* Dense bf16 matvec: the MTP draft head's eight projections.
 *
 * The base model is 4-bit throughout and so, now, is the draft head: it is
 * quantised at load, because a drafted token was reading 810 MB of head next to
 * 715 MB of output head and the head only proposes, so precision there is
 * purely an efficiency question.  Quantising cost nothing measurable --
 * acceptance is unchanged to the digit across ten configurations.
 *
 * This kernel is what remains, and it is not dead: it is the reference the
 * quantised head is checked against in tests/test_mtp, which is the only thing
 * that can catch a bad quantisation.  A head quantised wrongly does not fail,
 * it drafts worse.
 *
 * The blocking is qw_qmvb_q4_g64's, minus the dequantisation: one threadgroup
 * per block of output rows, a block of QW_QMVB_B tokens carried in registers,
 * and each weight read once and spent across every token.  Without nibbles to
 * unpack this kernel has half the ALU per byte of its quantised counterpart,
 * so it stays bandwidth-bound with room to spare.
 *
 * A lane reads 8 contiguous bf16 values -- 16 bytes -- and consecutive lanes
 * read consecutive chunks, so a simdgroup's fetch is 512 contiguous bytes of
 * one weight row.  Every in_features in this head is a multiple of 8. */

#define QW_DMV_PER_LANE 8

kernel void qw_dmvb_bf16(
    device const ushort  *w       [[buffer(0)]],   /* [n, k] bf16 */
    device const float   *x       [[buffer(1)]],   /* [rows, k] */
    device       float   *y       [[buffer(2)]],   /* [rows, n] */
    constant qw_matmul_args &a    [[buffer(3)]],
    uint3 tgid  [[threadgroup_position_in_grid]],
    uint  sgid  [[simdgroup_index_in_threadgroup]],
    uint  nsg   [[simdgroups_per_threadgroup]],
    uint  lane  [[thread_index_in_simdgroup]])
{
    const uint chunks = a.k / QW_DMV_PER_LANE;

    const uint row0 = (tgid.x * nsg + sgid) * QW_QMVB_ROWS;
    if (row0 >= a.n) return;

    /* As in qw_qmvb_q4_g64: out-of-range rows and dead token lanes are clamped
     * to something valid and discarded at the store, never bounded at runtime,
     * so every index into `acc` stays a compile-time constant and the array
     * stays in registers. */
    float acc[QW_QMVB_ROWS][QW_QMVB_B];
#pragma unroll
    for (uint r = 0; r < QW_QMVB_ROWS; ++r)
#pragma unroll
        for (uint b = 0; b < QW_QMVB_B; ++b) acc[r][b] = 0.0f;

    for (uint ci = lane; ci < chunks; ci += 32) {
        const uint k0 = ci * QW_DMV_PER_LANE;

        float wd[QW_QMVB_ROWS][QW_DMV_PER_LANE];
#pragma unroll
        for (uint r = 0; r < QW_QMVB_ROWS; ++r) {
            const uint n = min(row0 + r, a.n - 1);
            device const ushort *wp = w + (ulong)n * a.k + k0;
#pragma unroll
            for (uint j = 0; j < QW_DMV_PER_LANE; ++j)
                wd[r][j] = qw_bf16_to_f32(wp[j]);
        }

#pragma unroll
        for (uint b = 0; b < QW_QMVB_B; ++b) {
            device const float *xv = x + (ulong)min(b, a.rows - 1) * a.k + k0;
            float xs[QW_DMV_PER_LANE];
#pragma unroll
            for (uint j = 0; j < QW_DMV_PER_LANE; ++j) xs[j] = xv[j];

#pragma unroll
            for (uint r = 0; r < QW_QMVB_ROWS; ++r)
#pragma unroll
                for (uint j = 0; j < QW_DMV_PER_LANE; ++j)
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

/* Tiled bf16 matmul, for when the vision tower has more than a handful of
 * patches -- which is always.
 *
 * The batched matvec above reads the whole weight matrix once per block of
 * QW_QMVB_B tokens, which is right for a speculative verify of four and
 * catastrophic for an image: a 64x64 patch grid is 4096 tokens, so the tower's
 * 0.86 GB would be read a thousand times.  This is qw_qmm_q4_g64's structure
 * with the dequantisation removed -- same tiling, same matrix units, same
 * reason for storing As M-major -- staging bf16 straight into the operand tile.
 */
kernel void qw_dmm_bf16(
    device const ushort  *w  [[buffer(0)]],   /* [n, k] bf16 */
    device const float   *x  [[buffer(1)]],   /* [rows, k] */
    device       float   *y  [[buffer(2)]],   /* [rows, n] */
    constant qw_matmul_args &a [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]])
{
    threadgroup float pool[QW_QMM_POOL_F];
    threadgroup half *As = (threadgroup half *)pool;
    threadgroup half *Bs = (threadgroup half *)pool + QW_QMM_BK * QW_QMM_BM;

    const uint row0 = tgid.y * QW_QMM_BM;
    const uint col0 = tgid.x * QW_QMM_BN;

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
        for (uint idx = tid; idx < QW_QMM_BM * QW_QMM_BK; idx += QW_QMM_THREADS) {
            const uint kk = idx % QW_QMM_BK, mm = idx / QW_QMM_BK;
            const uint gm = row0 + mm;
            As[mm * QW_QMM_BK + kk] = (gm < a.rows)
                                    ? half(x[(ulong)gm * a.k + k0 + kk]) : half(0);
        }
        for (uint idx = tid; idx < QW_QMM_BN * QW_QMM_BK; idx += QW_QMM_THREADS) {
            const uint kk = idx % QW_QMM_BK, nn = idx / QW_QMM_BK;
            const uint gn = col0 + nn;
            /* Bs is K-major so its fragments load without transposing, which
             * costs a strided threadgroup write here and saves more there. */
            Bs[kk * QW_QMM_BN + nn] = (gn < a.n)
                ? half(qw_bf16_to_f32(w[(ulong)gn * a.k + k0 + kk])) : half(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint ks = 0; ks < QW_QMM_BK; ks += QW_SG_TILE) {
            simdgroup_half8x8 af[QW_QMM_FRAG_M], bf[QW_QMM_FRAG_N];
#pragma unroll
            for (uint i = 0; i < QW_QMM_FRAG_M; ++i)
                simdgroup_load(af[i], As + (m_base + i * QW_SG_TILE) * QW_QMM_BK + ks,
                               QW_QMM_BK, 0, false);
#pragma unroll
            for (uint j = 0; j < QW_QMM_FRAG_N; ++j)
                simdgroup_load(bf[j], Bs + ks * QW_QMM_BN + n_base + j * QW_SG_TILE,
                               QW_QMM_BN, 0, false);
#pragma unroll
            for (uint i = 0; i < QW_QMM_FRAG_M; ++i)
#pragma unroll
                for (uint j = 0; j < QW_QMM_FRAG_N; ++j)
                    simdgroup_multiply_accumulate(acc[i][j], af[i], bf[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float *Cs = pool;
#pragma unroll
    for (uint i = 0; i < QW_QMM_FRAG_M; ++i)
#pragma unroll
        for (uint j = 0; j < QW_QMM_FRAG_N; ++j)
            simdgroup_store(acc[i][j],
                            Cs + (m_base + i * QW_SG_TILE) * QW_QMM_BN
                               + n_base + j * QW_SG_TILE, QW_QMM_BN, 0, false);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = tid; idx < QW_QMM_BM * QW_QMM_BN; idx += QW_QMM_THREADS) {
        const uint mm = idx / QW_QMM_BN, nn = idx % QW_QMM_BN;
        const uint gm = row0 + mm, gn = col0 + nn;
        if (gm < a.rows && gn < a.n) y[(ulong)gm * a.n + gn] = Cs[idx];
    }
}
