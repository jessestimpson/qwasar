/* Quantised matmul on Metal 4 tensor ops: two candidate replacements for
 * qw_qmm_q4_g64 on prefill, written to be measured against each other on an
 * M5, whose GPU cores have matrix units ("neural accelerators") that only
 * mpp::tensor_ops reaches.  PLAN-flash-next.md measured 52.5 TFLOP/s through
 * them on a dense half matmul of a delta projection's shape, against the
 * ~11.4 ceiling of the 8x8 simdgroup_matrix tile qw_qmm_q4_g64 is built on.
 *
 * The same source runs on M1-M4, where the compiler lowers the ops onto the
 * ordinary simdgroup matrix path: correct, and slower than qw_qmm_q4_g64.
 * Correctness is therefore testable anywhere (tests/test_tmm); speed is only
 * worth measuring on an M5.
 *
 * This file is its own library, compiled at Metal 4 (qwasar_metal.m,
 * qw_tmm_library) only when something asks for it, so a macOS or device
 * without tensor ops loses these kernels and nothing else.  It shares no
 * helpers with metal/: it does not see common.metal.
 *
 * Both compute y[r][n] = sum_k dequant(w)[n][k] * x[r][k] over a BM x BN
 * output tile per threadgroup -- BM tokens by BN weight rows -- with the
 * product held in a cooperative tensor across the whole K walk.  They differ
 * in where the 4-bit weights become numbers:
 *
 *   dequant  Unpacks a BN x BK weight tile to half in threadgroup memory
 *            (scale and bias applied), then one matmul2d per K step with x
 *            read straight from device memory as float.  The shape of
 *            qw_qmm_q4_g64 with the inner product swapped out; its risk is
 *            that the unpacking, hidden today behind a slow multiply, is what
 *            remains once the multiply is several times faster.
 *
 *   int4     Hands the packed nibbles to matmul2d as uint4b_format, read
 *            straight from the mmap'd weights with no staging at all.  The op
 *            has no notion of MLX's per-group scale and bias, so it runs once
 *            per quantisation group and the affine part is applied after:
 *
 *              y = sum_g  s[n][g] * (x_g . q_g)  +  b[n][g] * sum(x_g)
 *
 *            Nothing to unpack, but a matmul2d of K = G (32 or 64) per group,
 *            and a pass over the tile's accumulators per group to rescale.
 *            The row sums are taken over the same half values the op
 *            multiplies, so the bias term sees the operand the product saw.
 *
 * uint4b_format packs element 2i in the low nibble of byte i, which is MLX's
 * order (element i at bit 4*(i%8) of little-endian word i/8); tests/test_tmm
 * is what holds that, since nothing in the header says so. */

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

/* Injected from qwasar_gpu.h by the runtime compile; these serve only an
 * offline syntax check. */
#ifndef QW_TMM_BM
#define QW_TMM_BM 64
#define QW_TMM_BN 64
#define QW_TMM_BK 64
#define QW_TMM_SG 4
#endif

struct qw_tmm_args {
    uint k;      /* input features  */
    uint n;      /* output features */
    uint rows;   /* tokens */
};

typedef tensor<device float, dextents<int32_t, 2>, tensor_inline> qw_tmm_dev_f32;

inline float qw_tmm_bf16(ushort v) { return as_type<float>(uint(v) << 16); }

/* ---- dequant -------------------------------------------------------------- */

template <uint G>
kernel void qw_tmm_q4_dequant(
    device const uint   *w      [[buffer(0)]],   /* [n, k/8] */
    device const ushort *scales [[buffer(1)]],   /* [n, k/G] bf16 */
    device const ushort *biases [[buffer(2)]],   /* [n, k/G] bf16 */
    device       float  *x      [[buffer(3)]],   /* [rows, k] */
    device       float  *y      [[buffer(4)]],   /* [rows, n] */
    constant qw_tmm_args &a     [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])
{
    threadgroup half Ws[QW_TMM_BN * QW_TMM_BK];   /* [BN][BK], K contiguous */

    const uint row0 = tgid.y * QW_TMM_BM, col0 = tgid.x * QW_TMM_BN;
    const uint words = a.k / 8, groups = a.k / G;

    /* Extents run innermost first: x is [rows][k], so (k, rows). */
    qw_tmm_dev_f32 X(x, dextents<int32_t, 2>(a.k, a.rows));
    qw_tmm_dev_f32 Y(y, dextents<int32_t, 2>(a.n, a.rows));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>
        Wt(Ws, dextents<int32_t, 2>(QW_TMM_BK, QW_TMM_BN));

    /* x [BM x BK] times the tile transposed: the tile is [BN][BK], which is
     * B^T, so transpose_right and no copy. */
    constexpr auto desc = matmul2d_descriptor(QW_TMM_BM, QW_TMM_BN, QW_TMM_BK,
                                              false, true, false,
                                              matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<QW_TMM_SG>> op;

    auto x0 = X.slice(0, row0);
    auto acc = op.template get_destination_cooperative_tensor<decltype(x0), decltype(Wt), float>();
#pragma unroll
    for (uint16_t i = 0; i < acc.get_capacity(); ++i)
        if (acc.is_valid_element(i)) acc[i] = 0.0f;

    for (uint k0 = 0; k0 < a.k; k0 += QW_TMM_BK) {
        /* One packed word -- eight weights -- per step.  Consecutive threads
         * take consecutive words of one row, so the reads coalesce. */
        for (uint idx = tid; idx < QW_TMM_BN * QW_TMM_BK / 8; idx += QW_TMM_SG * 32) {
            const uint nn = idx / (QW_TMM_BK / 8), wi = idx % (QW_TMM_BK / 8);
            const uint gn = col0 + nn, kw = k0 / 8 + wi;
            threadgroup half *dst = Ws + nn * QW_TMM_BK + wi * 8;
            if (gn < a.n) {
                const uint word = w[(ulong)gn * words + kw];
                const uint g = kw * 8 / G;
                const float s = qw_tmm_bf16(scales[(ulong)gn * groups + g]);
                const float b = qw_tmm_bf16(biases[(ulong)gn * groups + g]);
#pragma unroll
                for (uint j = 0; j < 8; ++j)
                    dst[j] = half(fma(s, float((word >> (4 * j)) & 0xF), b));
            } else {
#pragma unroll
                for (uint j = 0; j < 8; ++j) dst[j] = half(0);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto xs = X.slice(k0, row0);
        op.run(xs, Wt, acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    /* The store clips to Y's extents, so ragged edges need nothing here. */
    auto ys = Y.slice(col0, row0);
    acc.store(ys);
}

/* ---- int4 ----------------------------------------------------------------- */

template <uint G>
kernel void qw_tmm_q4_int4(
    device const uint   *w      [[buffer(0)]],
    device const ushort *scales [[buffer(1)]],
    device const ushort *biases [[buffer(2)]],
    device       float  *x      [[buffer(3)]],
    device       float  *y      [[buffer(4)]],
    constant qw_tmm_args &a     [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])
{
    /* uint4b_format multiplies only against half, so x is staged. */
    threadgroup half  Xs[QW_TMM_BM * G];   /* [BM][G] */
    threadgroup float xsum[QW_TMM_BM];

    const uint row0 = tgid.y * QW_TMM_BM, col0 = tgid.x * QW_TMM_BN;
    const uint groups = a.k / G;

    /* A 4-bit tensor's data handle is a byte pointer; its extents and strides
     * count nibbles. */
    tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline>
        W((device uchar *)w, dextents<int32_t, 2>(a.k, a.n));
    qw_tmm_dev_f32 Y(y, dextents<int32_t, 2>(a.n, a.rows));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>
        Xt(Xs, dextents<int32_t, 2>(G, QW_TMM_BM));

    constexpr auto desc = matmul2d_descriptor(QW_TMM_BM, QW_TMM_BN, G, false, true, false,
                                              matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<QW_TMM_SG>> op;

    auto w0 = W.slice(0, col0);
    auto acc = op.template get_destination_cooperative_tensor<decltype(Xt), decltype(w0), float>();
    auto p   = op.template get_destination_cooperative_tensor<decltype(Xt), decltype(w0), float>();
#pragma unroll
    for (uint16_t i = 0; i < acc.get_capacity(); ++i)
        if (acc.is_valid_element(i)) acc[i] = 0.0f;

    for (uint g = 0; g < groups; ++g) {
        for (uint idx = tid; idx < QW_TMM_BM * G; idx += QW_TMM_SG * 32) {
            const uint mm = idx / G, kk = idx % G, gm = row0 + mm;
            Xs[idx] = gm < a.rows ? half(x[(ulong)gm * a.k + g * G + kk]) : half(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < QW_TMM_BM) {
            float s = 0.0f;
            for (uint kk = 0; kk < G; ++kk) s += float(Xs[tid * G + kk]);
            xsum[tid] = s;
        }
        auto wg = W.slice(g * G, col0);
        op.run(Xt, wg, p);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        /* p and acc come from the same op, so element i of each is the same
         * (n, m) of the tile. */
#pragma unroll
        for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
            if (!acc.is_valid_element(i)) continue;
            auto ix = acc.get_multidimensional_index(i);   /* (n, m) within the tile */
            const uint gn = col0 + ix[0];
            if (gn >= a.n) continue;
            const ulong sg = (ulong)gn * groups + g;
            acc[i] += qw_tmm_bf16(scales[sg]) * p[i] + qw_tmm_bf16(biases[sg]) * xsum[ix[1]];
        }
        /* Xs and xsum are rewritten next group. */
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    auto ys = Y.slice(col0, row0);
    acc.store(ys);
}

/* One instance per group size a model uses: the 27B and Qwen3.5 9B quantise
 * in groups of 64, MLX's Flash-Next build in groups of 32.  The int4 kernel
 * needs G at compile time -- it is the op's K -- so these are templates
 * rather than function constants, and the dequant kernel follows suit. */
#define QW_TMM_INSTANCE(kind, G)                                                       \
    template [[host_name("qw_tmm_q4_" #kind "_g" #G)]] kernel void qw_tmm_q4_##kind<G>( \
        device const uint *, device const ushort *, device const ushort *,             \
        device float *, device float *, constant qw_tmm_args &, uint2, uint);

QW_TMM_INSTANCE(dequant, 32)
QW_TMM_INSTANCE(dequant, 64)
QW_TMM_INSTANCE(int4, 32)
QW_TMM_INSTANCE(int4, 64)
