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
