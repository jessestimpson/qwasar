/* Causal attention for prefill, tiled, on Metal 4 tensor ops.
 *
 * Prefill used to run qw_attn_decode once per query token: every query read
 * every key and value it can see straight from the cache, so a 256-token
 * chunk at a depth of 32K read ~200 GB of KV per attention layer, and the
 * 27B's prefill fell from 55 tok/s at the start of a prompt to 7 at 32K and
 * under 4 at 64K -- almost all of it this kernel.
 *
 * Here a threadgroup takes BQ query rows of one head and walks the keys BK at
 * a time (32 and 128; qwasar_gpu.h has the measurements), so each K and V
 * block is read once for BQ queries:
 *
 *   S = Q K^T             matmul2d into threadgroup memory (BQ x BK, float)
 *   online softmax        per row, over SG*32/BQ threads: running max m
 *                         and sum l, the block's
 *                         weights written over S in place, and the factor
 *                         alpha = exp(m_old - m_new) the output is rescaled by
 *   O = alpha O + P V     matmul2d accumulating into a cooperative tensor
 *
 * and O / l is stored at the end.  In the float form Q is read as float and
 * the weights stay float, so the only rounding beyond qw_attn_decode's is the
 * order of the sums: the cache is half either way.  The half form, for an M5,
 * is below.
 *
 * Causality: a query at row r sees keys [0, base_pos + r].  Blocks past a
 * tile's last row are never visited; within the diagonal block the excess is
 * masked to weight 0.  K and V are given extents of exactly the keys written,
 * so an op's edge reads past them are zeros, never stale cache. */

#ifndef QW_FA_BQ
#define QW_FA_BQ 32
#define QW_FA_BK 128
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

/* Two forms of one kernel.  The float one feeds Q and the softmax weights to
 * the ops as float: the M4's, where it matches qw_attn_decode to rounding
 * order.  The half one stages Q (scaled) and the weights as half -- the M5's
 * matrix units run half several times faster than float, 30.6 TFLOP/s
 * against 12-13 on the M5 Max -- at the precision of the operands qmm and the
 * expert kernels already round to, and finer than the bf16 the reference
 * runs this attention in.  Its tile is QW_FAH_BQ rows: Q and the weights
 * have to fit beside S. */
template <uint BQ, bool H>
kernel void qw_attn_prefill_impl(
    device float        *q   [[buffer(0)]],   /* [rows, q_heads, D] fp32 */
    device half         *kc  [[buffer(1)]],   /* [kv_heads, max_ctx, D] fp16 */
    device half         *vc  [[buffer(2)]],
    device float        *out [[buffer(3)]],   /* [rows, q_heads, D] fp32 */
    constant qw_fa_args &a   [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])
{
    /* Threads per row in the softmax step: all of them, BQ rows to SG*32. */
    constexpr uint SPLIT = (QW_FA_SG * 32) / BQ;
    threadgroup float S[BQ * QW_FA_BK];          /* scores, then (float) weights, [BQ][BK] */
    threadgroup half  Qs[H ? BQ * QW_FA_D : 1];  /* half: Q, scaled */
    threadgroup half  Ph[H ? BQ * QW_FA_BK : 1]; /* half: the weights */
    threadgroup float m_run[BQ], l_run[BQ], alpha[BQ];

    const uint qh = tgid.y, kvh = qh / a.gqa;
    const uint r0 = tgid.x * BQ;
    const int  qstride = (int)(a.q_heads * QW_FA_D);
    const uint n_keys = a.base_pos + a.rows;
    /* The last key any row of this tile sees, plus one. */
    const uint k_end = a.base_pos + min(r0 + BQ, a.rows);
    /* The half form folds the scale into Q. */
    const float sc = H ? 1.0f : a.scale;

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
        St(S, dextents<int32_t, 2>(QW_FA_BK, BQ));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>
        Qh(Qs, dextents<int32_t, 2>(QW_FA_D, BQ));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>
        Pt(Ph, dextents<int32_t, 2>(QW_FA_BK, BQ));

    /* K is [keys][D]: as the right operand of Q K^T that is B^T, so
     * transpose_right; V is [keys][D], B itself. */
    constexpr auto qk_desc = matmul2d_descriptor(BQ, QW_FA_BK, QW_FA_D, false, true, false,
                                                 matmul2d_descriptor::mode::multiply);
    constexpr auto pv_desc = matmul2d_descriptor(BQ, QW_FA_D, QW_FA_BK, false, false, false,
                                                 matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<qk_desc, execution_simdgroups<QW_FA_SG>> qk;
    matmul2d<pv_desc, execution_simdgroups<QW_FA_SG>> pv;

    if (H) {
        for (uint i = tid; i < BQ * QW_FA_D / 4; i += QW_FA_SG * 32) {
            const uint rr = i / (QW_FA_D / 4), d4 = i % (QW_FA_D / 4);
            const float4 v = r0 + rr < a.rows
                ? *(device const float4 *)(q + (ulong)(r0 + rr) * qstride + qh * QW_FA_D + d4 * 4) * a.scale
                : float4(0.0f);
            ((threadgroup half4 *)Qs)[i] = half4(v);
        }
    }
    auto q0 = Q.slice(0, r0);
    auto v0 = V.slice(0, 0);
    /* The accumulator's layout follows the weights' type. */
    using P_t = typename metal::conditional<H, decltype(Pt), decltype(St)>::type;
    auto O = pv.template get_destination_cooperative_tensor<P_t, decltype(v0), float>();
#pragma unroll
    for (uint16_t i = 0; i < O.get_capacity(); ++i)
        if (O.is_valid_element(i)) O[i] = 0.0f;
    if (tid < BQ) { m_run[tid] = -FLT_MAX; l_run[tid] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k0 = 0; k0 < k_end; k0 += QW_FA_BK) {
        auto kb = K.slice(0, k0);
        if constexpr (H) qk.run(Qh, kb, St);
        else             qk.run(q0, kb, St);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        /* SPLIT threads a row, adjacent lanes of one simdgroup, each taking
         * every SPLIT-th key: the block's max and sum are shuffled across the
         * group, and its first thread keeps the row's running state. */
        {
            const uint rr = tid / SPLIT, part = tid % SPLIT;
            const uint r = r0 + rr;
            threadgroup float *row = S + rr * QW_FA_BK;
            const bool valid = rr < BQ && r < a.rows;
            /* Not -INFINITY: fast-math may assume there are none. */
            const int live = valid ? min((int)QW_FA_BK, (int)(a.base_pos + r) - (int)k0 + 1) : 0;
            const float m_old = valid ? m_run[rr] : -FLT_MAX;
            float mx = m_old;
            for (int j = (int)part; j < live; j += SPLIT) mx = max(mx, row[j] * sc);
            for (uint o = 1; o < SPLIT; o <<= 1) mx = max(mx, simd_shuffle_xor(mx, (ushort)o));
            float sum = 0.0f;
            if (rr < BQ)
                for (int j = (int)part; j < (int)QW_FA_BK; j += SPLIT) {
                    const float p = j < live ? exp(row[j] * sc - mx) : 0.0f;
                    if (H) Ph[rr * QW_FA_BK + j] = half(p);
                    else   row[j] = p;
                    sum += p;
                }
            for (uint o = 1; o < SPLIT; o <<= 1) sum += simd_shuffle_xor(sum, (ushort)o);
            if (rr < BQ && part == 0) {
                const float al = valid ? exp(m_old - mx) : 1.0f;
                if (valid) { m_run[rr] = mx; l_run[rr] = l_run[rr] * al + sum; }
                alpha[rr] = al;
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
        if constexpr (H) pv.run(Pt, vb, O);
        else             pv.run(St, vb, O);
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

#ifndef QW_FAH_BQ
#define QW_FAH_BQ 16
#endif
#define QW_FA_INSTANCE(name, BQ, H)                                                       \
    template [[host_name(name)]] kernel void qw_attn_prefill_impl<BQ, H>(                \
        device float *, device half *, device half *, device float *,                     \
        constant qw_fa_args &, uint2, uint);
QW_FA_INSTANCE("qw_attn_prefill_tensor", QW_FA_BQ, false)
QW_FA_INSTANCE("qw_attn_prefill_tensor_half", QW_FAH_BQ, true)

/* ---- masked: prefill's sparse attention, on tensor ops ----------------------
 *
 * qw_attn_masked (metal/sparse.metal) over one row's selected positions,
 * which qw_qsa_select lists once per row for every head.  That kernel runs a
 * threadgroup per (row, query head), so each kv head's 2,048 selected keys
 * and values were read once for each of the gqa (12) query heads that share
 * them, by scalar FMAs.  Here a threadgroup takes (row, kv head): its query
 * heads are the rows of a QW_AM_BQ-row Q tile, and the selected keys are
 * gathered QW_AM_BK at a time into threadgroup memory --
 *
 *   S = Q K^T     matmul2d into threadgroup memory, an online softmax there
 *   O = alpha O + P V    with the same rows of V gathered into the same tile
 *
 * so each key and value is read once per row.  Q (pre-scaled) and P are
 * half, the cache already is; S, the softmax and O stay float. */

#ifndef QW_AM_BQ
#define QW_AM_BQ 16
#define QW_AM_BK 32
#define QW_AM_SG 8
#endif
#define QW_AM_T     (QW_AM_SG * 32)
#define QW_AM_SPLIT (QW_AM_T / QW_AM_BQ)   /* threads a row in the softmax */

struct qw_am_args {
    uint  rows, q_heads, kv_heads, gqa, max_ctx, base_pos;
    float scale;
    uint  list_cap;
};

kernel void qw_attn_masked_tensor(
    device const float  *q    [[buffer(0)]],   /* [rows, q_heads, D] fp32 */
    device const half   *kc   [[buffer(1)]],   /* [kv_heads, max_ctx, D] fp16 */
    device const half   *vc   [[buffer(2)]],
    device       float  *out  [[buffer(3)]],   /* [rows, q_heads, D] fp32 */
    device const uint   *list [[buffer(4)]],   /* [rows, 1 + list_cap]: count, positions */
    constant qw_am_args &a    [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])
{
    threadgroup half  Qs[QW_AM_BQ * QW_FA_D];    /* [BQ][D] */
    threadgroup half  KV[QW_AM_BK * QW_FA_D];    /* [BK][D]: keys, then values */
    threadgroup float S[QW_AM_BQ * QW_AM_BK];    /* [BQ][BK] */
    threadgroup half  P[QW_AM_BQ * QW_AM_BK];
    threadgroup float m_run[QW_AM_BQ], l_run[QW_AM_BQ], alpha[QW_AM_BQ];

    const uint row = tgid.x, kvh = tgid.y;
    device const uint *lr = list + (ulong)row * (1 + a.list_cap);
    const uint n_sel = lr[0];
    device const half *kbase = kc + (ulong)kvh * a.max_ctx * QW_FA_D;
    device const half *vbase = vc + (ulong)kvh * a.max_ctx * QW_FA_D;

    /* This kv head's query heads, scaled, as half; rows past gqa are zero. */
    device const float *qrow = q + ((ulong)row * a.q_heads + kvh * a.gqa) * QW_FA_D;
    for (uint i = tid; i < QW_AM_BQ * QW_FA_D / 4; i += QW_AM_T) {
        const uint h = i / (QW_FA_D / 4);
        const float4 v = h < a.gqa ? ((device const float4 *)qrow)[i] * a.scale : float4(0.0f);
        ((threadgroup half4 *)Qs)[i] = half4(v);
    }
    if (tid < QW_AM_BQ) { m_run[tid] = -FLT_MAX; l_run[tid] = 0.0f; }

    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>
        Qt(Qs, dextents<int32_t, 2>(QW_FA_D, QW_AM_BQ));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>
        KVt(KV, dextents<int32_t, 2>(QW_FA_D, QW_AM_BK));
    tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>
        St(S, dextents<int32_t, 2>(QW_AM_BK, QW_AM_BQ));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>
        Pt(P, dextents<int32_t, 2>(QW_AM_BK, QW_AM_BQ));
    constexpr auto qk_desc = matmul2d_descriptor(QW_AM_BQ, QW_AM_BK, QW_FA_D, false, true, false,
                                                 matmul2d_descriptor::mode::multiply);
    constexpr auto pv_desc = matmul2d_descriptor(QW_AM_BQ, QW_FA_D, QW_AM_BK, false, false, false,
                                                 matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<qk_desc, execution_simdgroups<QW_AM_SG>> qk;
    matmul2d<pv_desc, execution_simdgroups<QW_AM_SG>> pv;
    auto O = pv.template get_destination_cooperative_tensor<decltype(Pt), decltype(KVt), float>();
#pragma unroll
    for (uint16_t i = 0; i < O.get_capacity(); ++i)
        if (O.is_valid_element(i)) O[i] = 0.0f;

    /* Gathers BK selected rows of one cache into KV, eight halves a step;
     * slots past the list are zero. */
    auto gather = [&](device const half *base, uint j0) {
        for (uint i = tid; i < QW_AM_BK * QW_FA_D / 8; i += QW_AM_T) {
            const uint j = i / (QW_FA_D / 8), c = i % (QW_FA_D / 8);
            uint4 v = uint4(0);
            if (j0 + j < n_sel) v = ((device const uint4 *)(base + (ulong)lr[1 + j0 + j] * QW_FA_D))[c];
            ((threadgroup uint4 *)KV)[i] = v;
        }
    };

    for (uint j0 = 0; j0 < n_sel; j0 += QW_AM_BK) {
        gather(kbase, j0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        qk.run(Qt, KVt, St);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        /* Values into the same tile while the softmax runs on S. */
        gather(vbase, j0);
        {
            const uint rr = tid / QW_AM_SPLIT, part = tid % QW_AM_SPLIT;
            threadgroup float *srow = S + rr * QW_AM_BK;
            const bool valid = rr < a.gqa;
            const int live = valid ? min((int)QW_AM_BK, (int)n_sel - (int)j0) : 0;
            const float m_old = m_run[rr];
            float mx = m_old;
            for (int j = (int)part; j < live; j += QW_AM_SPLIT) mx = max(mx, srow[j]);
            for (uint o = 1; o < QW_AM_SPLIT; o <<= 1) mx = max(mx, simd_shuffle_xor(mx, (ushort)o));
            float sum = 0.0f;
            for (int j = (int)part; j < (int)QW_AM_BK; j += QW_AM_SPLIT) {
                const float p = j < live ? exp(srow[j] - mx) : 0.0f;
                P[rr * QW_AM_BK + j] = half(p);
                sum += p;
            }
            for (uint o = 1; o < QW_AM_SPLIT; o <<= 1) sum += simd_shuffle_xor(sum, (ushort)o);
            if (part == 0) {
                const float al = valid ? exp(m_old - mx) : 1.0f;
                if (valid) { m_run[rr] = mx; l_run[rr] = l_run[rr] * al + sum; }
                alpha[rr] = al;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

#pragma unroll
        for (uint16_t i = 0; i < O.get_capacity(); ++i) {
            if (!O.is_valid_element(i)) continue;
            O[i] *= alpha[O.get_multidimensional_index(i)[1]];
        }
        pv.run(Pt, KVt, O);
        /* KV and P are written again next block. */
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device float *orow = out + ((ulong)row * a.q_heads + kvh * a.gqa) * QW_FA_D;
#pragma unroll
    for (uint16_t i = 0; i < O.get_capacity(); ++i) {
        if (!O.is_valid_element(i)) continue;
        auto ix = O.get_multidimensional_index(i);   /* (d, head) */
        if ((uint)ix[1] >= a.gqa) continue;
        const float l = l_run[ix[1]];
        orow[(uint)ix[1] * QW_FA_D + ix[0]] = l > 0.0f ? O[i] / l : 0.0f;
    }
}

/* ---- split: decode at depth, on tensor ops ----------------------------------
 *
 * A token's attention is one query row: the per-query kernel's threadgroups
 * are (row, query head), 24 of them for the 27B -- too few to fill an M5 Max,
 * and each kv head's keys read once for each of its query heads.  Here a
 * threadgroup takes one slice of keys and one kv head, its query heads the
 * rows of a QW_AM_BQ-row Q tile as in qw_attn_masked_tensor, and walks the
 * slice QW_DS_BK keys a step with K and V read straight from the cache as
 * tensor operands, and Q and the weights float as in qw_attn_prefill_tensor.
 * Each slice leaves its unnormalised output and its
 * softmax's max and sum in the layout qw_attn_merge (metal/attn.metal)
 * combines: [rows, q_heads, parts, D] and [rows, q_heads, parts, 2]. */

#ifndef QW_DS_BK
#define QW_DS_BK 64
#define QW_DS_SG 4
#endif
#define QW_DS_T     (QW_DS_SG * 32)
#define QW_DS_SPLIT (QW_DS_T / QW_AM_BQ)

struct qw_ds_args {
    uint  rows, q_heads, kv_heads, gqa, max_ctx, base_pos;
    uint  splits, per_split;
    float scale;
};

kernel void qw_attn_split_tensor(
    device       float *q     [[buffer(0)]],   /* [rows, q_heads, D] fp32 */
    device       half  *kc    [[buffer(1)]],   /* [kv_heads, max_ctx, D] fp16 */
    device       half  *vc    [[buffer(2)]],
    device       float *pacc  [[buffer(3)]],   /* [rows, q_heads, splits, D] */
    device       float *pstat [[buffer(4)]],   /* [rows, q_heads, splits, 2]: max, sum */
    constant qw_ds_args &a    [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])
{
    threadgroup float S[QW_AM_BQ * QW_DS_BK];   /* scores, then weights, [BQ][BK] */
    threadgroup float m_run[QW_AM_BQ], l_run[QW_AM_BQ], alpha[QW_AM_BQ];

    const uint split = tgid.x;
    const uint row = tgid.y / a.kv_heads, kvh = tgid.y % a.kv_heads;
    const uint n_keys = a.base_pos + row + 1;   /* causal */
    const uint k_begin = split * a.per_split;
    const uint k_end = min(k_begin + a.per_split, n_keys);
    if (tid < QW_AM_BQ) { m_run[tid] = -FLT_MAX; l_run[tid] = 0.0f; }

    /* Q is float and its extents are this kv head's query heads, so the
     * tile's rows past them read as zeros.  K and V get extents of exactly
     * the keys this slice may see: reads past them are zeros, never another
     * slice's keys or stale cache. */
    tensor<device float, dextents<int32_t, 2>, tensor_inline>
        Q(q + ((ulong)row * a.q_heads + kvh * a.gqa) * QW_FA_D, dextents<int32_t, 2>(QW_FA_D, (int32_t)a.gqa));
    tensor<device half, dextents<int32_t, 2>, tensor_inline>
        K(kc + (ulong)kvh * a.max_ctx * QW_FA_D, dextents<int32_t, 2>(QW_FA_D, (int32_t)k_end));
    tensor<device half, dextents<int32_t, 2>, tensor_inline>
        V(vc + (ulong)kvh * a.max_ctx * QW_FA_D, dextents<int32_t, 2>(QW_FA_D, (int32_t)k_end));
    tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>
        St(S, dextents<int32_t, 2>(QW_DS_BK, QW_AM_BQ));
    constexpr auto qk_desc = matmul2d_descriptor(QW_AM_BQ, QW_DS_BK, QW_FA_D, false, true, false,
                                                 matmul2d_descriptor::mode::multiply);
    constexpr auto pv_desc = matmul2d_descriptor(QW_AM_BQ, QW_FA_D, QW_DS_BK, false, false, false,
                                                 matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<qk_desc, execution_simdgroups<QW_DS_SG>> qk;
    matmul2d<pv_desc, execution_simdgroups<QW_DS_SG>> pv;
    auto q0 = Q.slice(0, 0);
    auto v0 = V.slice(0, 0);
    auto O = pv.template get_destination_cooperative_tensor<decltype(St), decltype(v0), float>();
#pragma unroll
    for (uint16_t i = 0; i < O.get_capacity(); ++i)
        if (O.is_valid_element(i)) O[i] = 0.0f;

    for (uint k0 = k_begin; k0 < k_end; k0 += QW_DS_BK) {
        auto kb = K.slice(0, (int32_t)k0);
        qk.run(q0, kb, St);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            const uint rr = tid / QW_DS_SPLIT, part = tid % QW_DS_SPLIT;
            threadgroup float *srow = S + rr * QW_DS_BK;
            const bool valid = rr < a.gqa;
            const int live = valid ? min((int)QW_DS_BK, (int)(k_end - k0)) : 0;
            const float m_old = m_run[rr];
            float mx = m_old;
            for (int j = (int)part; j < live; j += QW_DS_SPLIT) mx = max(mx, srow[j] * a.scale);
            for (uint o = 1; o < QW_DS_SPLIT; o <<= 1) mx = max(mx, simd_shuffle_xor(mx, (ushort)o));
            float sum = 0.0f;
            for (int j = (int)part; j < (int)QW_DS_BK; j += QW_DS_SPLIT) {
                const float p = j < live ? exp(srow[j] * a.scale - mx) : 0.0f;
                srow[j] = p;
                sum += p;
            }
            for (uint o = 1; o < QW_DS_SPLIT; o <<= 1) sum += simd_shuffle_xor(sum, (ushort)o);
            if (part == 0) {
                const float al = valid ? exp(m_old - mx) : 1.0f;
                if (valid) { m_run[rr] = mx; l_run[rr] = l_run[rr] * al + sum; }
                alpha[rr] = al;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
        for (uint16_t i = 0; i < O.get_capacity(); ++i) {
            if (!O.is_valid_element(i)) continue;
            O[i] *= alpha[O.get_multidimensional_index(i)[1]];
        }
        auto vb = V.slice(0, (int32_t)k0);
        pv.run(St, vb, O);
        /* S is written again next block. */
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const ulong base = ((ulong)row * a.q_heads + kvh * a.gqa) * a.splits + split;
#pragma unroll
    for (uint16_t i = 0; i < O.get_capacity(); ++i) {
        if (!O.is_valid_element(i)) continue;
        auto ix = O.get_multidimensional_index(i);   /* (d, head) */
        if ((uint)ix[1] >= a.gqa) continue;
        pacc[(base + (ulong)ix[1] * a.splits) * QW_FA_D + ix[0]] = O[i];
    }
    if (tid < a.gqa) {
        const ulong idx = base + (ulong)tid * a.splits;
        pstat[idx * 2] = m_run[tid];
        pstat[idx * 2 + 1] = l_run[tid];
    }
}
