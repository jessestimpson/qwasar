/* Causal attention for prefill, tiled, on Metal 4 tensor ops.
 *
 * Prefill used to run qw_attn_decode once per query token: every query read
 * every key and value it can see straight from the cache, so a 256-token
 * chunk at a depth of 32K read ~200 GB of KV per attention layer, and the
 * 27B's prefill fell from 55 tok/s at the start of a prompt to 7 at 32K and
 * under 4 at 64K -- almost all of it this kernel.
 *
 * Here a threadgroup takes BQ query rows of one head and walks the keys BK at
 * a time, so each K and V block is read once for BQ queries:
 *
 *   S = Q K^T             matmul2d into threadgroup memory (BQ x BK, float)
 *   online softmax        per row: running max m and sum l, the block's
 *                         weights written over S in place, and the factor
 *                         alpha = exp(m_old - m_new) the output is rescaled by
 *   O = alpha O + P V     matmul2d accumulating into a cooperative tensor
 *
 * and O / l is stored at the end.  Q is read as float and the weights stay
 * float, so the only rounding beyond qw_attn_decode's is the order of the
 * sums: the cache is half either way.
 *
 * Causality: a query at row r sees keys [0, base_pos + r].  Blocks past a
 * tile's last row are never visited; within the diagonal block the excess is
 * masked to weight 0.  K and V are given extents of exactly the keys written,
 * so an op's edge reads past them are zeros, never stale cache. */

#ifndef QW_FA_BQ
#define QW_FA_BQ 32
#define QW_FA_BK 32
#define QW_FA_SG 4
#endif
#define QW_FA_D  256

struct qw_fa_args {
    uint  rows;       /* query tokens in this step */
    uint  q_heads;
    uint  kv_heads;
    uint  gqa;        /* q_heads / kv_heads */
    uint  max_ctx;    /* cache stride */
    uint  base_pos;   /* cache position of row 0 */
    float scale;
};

kernel void qw_attn_prefill_tensor(
    device float        *q   [[buffer(0)]],   /* [rows, q_heads, D] fp32 */
    device half         *kc  [[buffer(1)]],   /* [kv_heads, max_ctx, D] fp16 */
    device half         *vc  [[buffer(2)]],
    device float        *out [[buffer(3)]],   /* [rows, q_heads, D] fp32 */
    constant qw_fa_args &a   [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])
{
    threadgroup float S[QW_FA_BQ * QW_FA_BK];   /* scores, then weights, [BQ][BK] */
    threadgroup float m_run[QW_FA_BQ], l_run[QW_FA_BQ], alpha[QW_FA_BQ];

    const uint qh = tgid.y, kvh = qh / a.gqa;
    const uint r0 = tgid.x * QW_FA_BQ;
    const int  qstride = (int)(a.q_heads * QW_FA_D);
    const uint n_keys = a.base_pos + a.rows;
    /* The last key any row of this tile sees, plus one. */
    const uint k_end = a.base_pos + min(r0 + QW_FA_BQ, a.rows);

    /* One head's rows are strided by the whole row: (d, r) is at r*qstride + d. */
    tensor<device float, dextents<int32_t, 2>, tensor_inline>
        Q(q + qh * QW_FA_D, dextents<int32_t, 2>(QW_FA_D, a.rows), array<int32_t, 2>{1, qstride});
    tensor<device float, dextents<int32_t, 2>, tensor_inline>
        O_out(out + qh * QW_FA_D, dextents<int32_t, 2>(QW_FA_D, a.rows), array<int32_t, 2>{1, qstride});
    tensor<device half, dextents<int32_t, 2>, tensor_inline>
        K(kc + (ulong)kvh * a.max_ctx * QW_FA_D, dextents<int32_t, 2>(QW_FA_D, n_keys));
    tensor<device half, dextents<int32_t, 2>, tensor_inline>
        V(vc + (ulong)kvh * a.max_ctx * QW_FA_D, dextents<int32_t, 2>(QW_FA_D, n_keys));
    tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>
        St(S, dextents<int32_t, 2>(QW_FA_BK, QW_FA_BQ));

    /* K is [keys][D]: as the right operand of Q K^T that is B^T, so
     * transpose_right; V is [keys][D], B itself. */
    constexpr auto qk_desc = matmul2d_descriptor(QW_FA_BQ, QW_FA_BK, QW_FA_D, false, true, false,
                                                 matmul2d_descriptor::mode::multiply);
    constexpr auto pv_desc = matmul2d_descriptor(QW_FA_BQ, QW_FA_D, QW_FA_BK, false, false, false,
                                                 matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<qk_desc, execution_simdgroups<QW_FA_SG>> qk;
    matmul2d<pv_desc, execution_simdgroups<QW_FA_SG>> pv;

    auto q0 = Q.slice(0, r0);
    auto v0 = V.slice(0, 0);
    auto O = pv.template get_destination_cooperative_tensor<decltype(St), decltype(v0), float>();
#pragma unroll
    for (uint16_t i = 0; i < O.get_capacity(); ++i)
        if (O.is_valid_element(i)) O[i] = 0.0f;
    if (tid < QW_FA_BQ) { m_run[tid] = -FLT_MAX; l_run[tid] = 0.0f; }

    for (uint k0 = 0; k0 < k_end; k0 += QW_FA_BK) {
        auto kb = K.slice(0, k0);
        qk.run(q0, kb, St);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        /* One thread a row: the block's max, the weights, the rescale. */
        if (tid < QW_FA_BQ) {
            const uint r = r0 + tid;
            threadgroup float *row = S + tid * QW_FA_BK;
            if (r < a.rows) {
                /* Not -INFINITY: fast-math may assume there are none. */
                const int last = (int)(a.base_pos + r);     /* the last key this row sees */
                const int live = min((int)QW_FA_BK, last - (int)k0 + 1);
                float mx = m_run[tid];
                for (int j = 0; j < live; ++j) mx = max(mx, row[j] * a.scale);
                const float al = exp(m_run[tid] - mx);
                float sum = 0.0f;
                for (int j = 0; j < (int)QW_FA_BK; ++j) {
                    const float p = j < live ? exp(row[j] * a.scale - mx) : 0.0f;
                    row[j] = p;
                    sum += p;
                }
                m_run[tid] = mx;
                l_run[tid] = l_run[tid] * al + sum;
                alpha[tid] = al;
            } else {
                for (int j = 0; j < (int)QW_FA_BK; ++j) row[j] = 0.0f;
                alpha[tid] = 1.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

#pragma unroll
        for (uint16_t i = 0; i < O.get_capacity(); ++i) {
            if (!O.is_valid_element(i)) continue;
            auto ix = O.get_multidimensional_index(i);   /* (d, row) in the tile */
            O[i] *= alpha[ix[1]];
        }
        auto vb = V.slice(0, k0);
        pv.run(St, vb, O);
        /* S is written again next block. */
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

#pragma unroll
    for (uint16_t i = 0; i < O.get_capacity(); ++i) {
        if (!O.is_valid_element(i)) continue;
        const float l = l_run[O.get_multidimensional_index(i)[1]];
        O[i] = l > 0.0f ? O[i] / l : 0.0f;
    }
    auto o0 = O_out.slice(0, r0);
    O.store(o0);
}
