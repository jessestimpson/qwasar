/* Flash-Next (qwen4_exp): the kernels the 27B never needed.
 *
 * PLAN-flash-next.md.  Everything the family adds over the 27B lives here --
 * the hyper-connection mixers, the MoE router and expert banks, QSA's block
 * indexer and masked attention, the engram layer's gate and dilated conv --
 * each one a direct transcription of qwasar_flash_cpu.c, which is the twin
 * tests/test_flashnext holds them to.  Correctness first: several of these
 * are the simplest shape that is right (the selection kernel in particular),
 * and the measurements that justify a faster one have not been taken yet.
 *
 * File order matters: the sources are concatenated alphabetically, and this
 * one relies on qw_row_sumsq from norm.metal and the helpers in common.metal. */

/* ---- hyper-connections ---------------------------------------------------- */

struct qw_hc_args { uint rows, H, S; };

/* h4[r, s*H + i] = x[r, i] for every stream s: the embedding, repeated. */
kernel void qw_repeat_cols(
    device const float *x  [[buffer(0)]],   /* [rows, H] */
    device       float *h4 [[buffer(1)]],   /* [rows, S*H] */
    constant qw_hc_args &a [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    const uint HH = a.S * a.H;
    if (gid >= a.rows * HH) return;
    const uint r = gid / HH, i = (gid % HH) % a.H;
    h4[gid] = x[r * a.H + i];
}

/* Grouped RMS norm: `groups` streams of `dim` per row, each normalised on
 * its own, then the full-width (+1) weight.  One threadgroup per stream. */
struct qw_gnorm_args { uint dim, groups, rows; float eps; };

kernel void qw_rms_norm_grouped(
    device const float  *x [[buffer(0)]],   /* [rows, groups*dim] */
    device const ushort *w [[buffer(1)]],   /* [groups*dim] bf16 */
    device       float  *y [[buffer(2)]],
    constant qw_gnorm_args &a [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint ntg  [[threads_per_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint nsg  [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float partial[32];
    const uint g = tgid % a.groups;
    device const float *xr = x + (ulong)tgid * a.dim;     /* row*groups + g */
    device       float *yr = y + (ulong)tgid * a.dim;
    device const ushort *wr = w + (ulong)g * a.dim;
    float sumsq = qw_row_sumsq(xr, a.dim, tid, ntg, sgid, nsg, lane, partial);
    const float inv = rsqrt(sumsq / float(a.dim) + a.eps);
    for (uint i = tid; i < a.dim; i += ntg) yr[i] = xr[i] * inv * qw_bf16_to_f32(wr[i]);
}

/* y = silu(y * scale), in place -- the mixer's `silu(down(n) / hc_count)`. */
struct qw_scale_args { uint n; float scale; };

kernel void qw_silu_scale(
    device float *y [[buffer(0)]],
    constant qw_scale_args &a [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < a.n) y[gid] = qw_silu(y[gid] * a.scale);
}

/* x[r, i] = mean over streams of sigmoid(m[r, s*H+i]) * n[r, s*H+i]. */
kernel void qw_hc_mix(
    device const float *n  [[buffer(0)]],   /* [rows, S*H] normalised streams */
    device const float *m  [[buffer(1)]],   /* [rows, S*H] pre-sigmoid mix weights */
    device       float *x  [[buffer(2)]],   /* [rows, H] */
    constant qw_hc_args &a [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= a.rows * a.H) return;
    const uint r = gid / a.H, i = gid % a.H;
    float acc = 0.0f;
    for (uint s = 0; s < a.S; ++s) {
        const ulong j = (ulong)r * a.S * a.H + (ulong)s * a.H + i;
        acc += qw_sigmoid(m[j]) * n[j];
    }
    x[gid] = acc / float(a.S);
}

/* h4[r, s*H+i] += out[r, i] * 2*sigmoid(inj[r, s] / S). */
kernel void qw_hc_inject(
    device       float *h4  [[buffer(0)]],   /* [rows, S*H] */
    device const float *out [[buffer(1)]],   /* [rows, H] */
    device const float *inj [[buffer(2)]],   /* [rows, S] raw block-inject logits */
    constant qw_hc_args &a  [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    const uint HH = a.S * a.H;
    if (gid >= a.rows * HH) return;
    const uint r = gid / HH, s = (gid % HH) / a.H, i = gid % a.H;
    const float w = 2.0f * qw_sigmoid(inj[r * a.S + s] / float(a.S));
    h4[gid] += out[r * a.H + i] * w;
}

/* ---- mixture of experts --------------------------------------------------- */

/* Softmax in fp32 over every expert, top-k, renormalised.  One threadgroup
 * of QW_ROUTE_THREADS per token: the logits go to threadgroup memory, max and
 * sum are tree reductions, and each of the K picks is a parallel argmax --
 * highest probability, ties to the lowest expert index, the order a serial
 * scan with `>` would pick in.  (It was one thread per token, recomputing
 * every exp in every round: 72% of a decode step.) */
struct qw_route_args { uint rows, E, K, norm; };
#define QW_ROUTE_THREADS 256
#define QW_ROUTE_MAX_E   1024

kernel void qw_moe_route(
    device const float *logits [[buffer(0)]],   /* [rows, E] */
    device       int   *idx    [[buffer(1)]],   /* [rows, K] */
    device       float *w      [[buffer(2)]],   /* [rows, K] */
    constant qw_route_args &a  [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float p[QW_ROUTE_MAX_E];
    threadgroup float red[QW_ROUTE_THREADS / 32];
    threadgroup int   redi[QW_ROUTE_THREADS / 32];
    threadgroup float picked_w[32];
    threadgroup int   picked_i[32];
    const uint NSG = QW_ROUTE_THREADS / 32;

    device const float *l = logits + (ulong)tgid * a.E;

    /* max */
    float m = -FLT_MAX;
    for (uint e = tid; e < a.E; e += QW_ROUTE_THREADS) { const float v = l[e]; p[e] = v; m = max(m, v); }
    m = simd_max(m);
    if (lane == 0) red[sgid] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m = red[0];
    for (uint i = 1; i < NSG; ++i) m = max(m, red[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* sum of exps, then the probabilities */
    float sum = 0.0f;
    for (uint e = tid; e < a.E; e += QW_ROUTE_THREADS) { const float x = exp(p[e] - m); p[e] = x; sum += x; }
    sum = simd_sum(sum);
    if (lane == 0) red[sgid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum = 0.0f;
    for (uint i = 0; i < NSG; ++i) sum += red[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < a.E; e += QW_ROUTE_THREADS) p[e] = p[e] / sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* K rounds of argmax; a pick is struck out as -1, below any probability */
    for (uint k = 0; k < a.K; ++k) {
        float bv = -1.0f;
        int   bi = 0x7fffffff;
        for (uint e = tid; e < a.E; e += QW_ROUTE_THREADS) {
            const float v = p[e];
            if (v > bv || (v == bv && (int)e < bi)) { bv = v; bi = (int)e; }
        }
        const float sv = simd_max(bv);
        int si = simd_min(bv == sv ? bi : 0x7fffffff);
        if (lane == 0) { red[sgid] = sv; redi[sgid] = si; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float gv = red[0];
            int   gi = redi[0];
            for (uint i = 1; i < NSG; ++i)
                if (red[i] > gv || (red[i] == gv && redi[i] < gi)) { gv = red[i]; gi = redi[i]; }
            picked_w[k] = gv;
            picked_i[k] = gi;
            p[gi] = -1.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        float wsum = 0.0f;
        for (uint k = 0; k < a.K; ++k) wsum += picked_w[k];
        for (uint k = 0; k < a.K; ++k) {
            idx[tgid * a.K + k] = picked_i[k];
            w[tgid * a.K + k] = a.norm ? picked_w[k] / wsum : picked_w[k];
        }
    }
}

/* ---- grouped experts, for prefill ------------------------------------------
 *
 * A chunk of prefill routes rows*K (token, expert) pairs.  One matvec per pair
 * re-reads each expert's weights for every token routed to it -- half of a
 * prompt's GPU time.  Grouped, each expert's weights are read once per tile
 * of up to BM of its pairs, by the tiled matmul below.
 *
 * qw_moe_group sorts the pairs by expert (stably, so the order and the result
 * are deterministic) and cuts each expert's run into tiles.  It is one thread:
 * a few thousand pairs, once per layer per chunk, is tens of microseconds. */
struct qw_group_args { uint pairs, E, BM, max_tiles; };
#define QW_GROUP_THREADS 1024
#define QW_GROUP_MAX_E   1024

kernel void qw_moe_group(
    device const int *idx    [[buffer(0)]],   /* [pairs] expert of each pair */
    device       int *perm   [[buffer(1)]],   /* [pairs] out: pairs by expert */
    device       int *tiles  [[buffer(2)]],   /* out: [0] count, then (expert, start, len) */
    device       int *unused [[buffer(3)]],
    constant qw_group_args &a [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]],
    uint ntg [[threads_per_threadgroup]])
{
    /* Counted and scattered with threadgroup atomics.  The order of pairs
     * within one expert's run varies from run to run, and nothing depends on
     * it: every output row of a tile is its own dot product. */
    threadgroup atomic_int cnt[QW_GROUP_MAX_E];
    threadgroup int start[QW_GROUP_MAX_E];
    for (uint e = tid; e < a.E; e += ntg) atomic_store_explicit(&cnt[e], 0, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint p = tid; p < a.pairs; p += ntg)
        atomic_fetch_add_explicit(&cnt[idx[p]], 1, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        int running = 0, nt = 0;
        for (uint e = 0; e < a.E; ++e) {
            const int c = atomic_load_explicit(&cnt[e], memory_order_relaxed);
            for (int s = 0; s < c && nt < (int)a.max_tiles; s += (int)a.BM) {
                tiles[1 + nt * 3 + 0] = (int)e;
                tiles[1 + nt * 3 + 1] = running + s;
                tiles[1 + nt * 3 + 2] = min((int)a.BM, c - s);
                nt++;
            }
            start[e] = running;
            atomic_store_explicit(&cnt[e], running, memory_order_relaxed);
            running += c;
        }
        tiles[0] = nt;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint p = tid; p < a.pairs; p += ntg)
        perm[atomic_fetch_add_explicit(&cnt[idx[p]], 1, memory_order_relaxed)] = (int)p;
}

/* Tiling for the grouped matmul, injected from qwasar_gpu.h like the dense
 * one's.  Smaller in M than qmm's: a 256-token chunk spreads 2560 pairs over
 * 512 experts, ~5 each, and a 64-row tile would spend 90% of its matrix-unit
 * work on empty rows -- measured, it undid everything reading the weights
 * once had saved. */
#ifndef QW_GMM_BM
#define QW_GMM_BM   16
#define QW_GMM_BN   64
#define QW_GMM_BK   32
#define QW_GMM_SG_M 1
#define QW_GMM_SG_N 4
#endif
#define QW_GMM_THREADS (QW_GMM_SG_M * QW_GMM_SG_N * 32)
#define QW_GMM_FRAG_M  ((QW_GMM_BM / QW_GMM_SG_M) / QW_SG_TILE)
#define QW_GMM_FRAG_N  ((QW_GMM_BN / QW_GMM_SG_N) / QW_SG_TILE)
#define QW_GMM_POOL_HALF (QW_GMM_BK * (QW_GMM_BM + QW_GMM_BN))
#define QW_GMM_POOL_F    (QW_GMM_POOL_HALF / 2 > QW_GMM_BM * QW_GMM_BN \
                          ? QW_GMM_POOL_HALF / 2 : QW_GMM_BM * QW_GMM_BN)

/* The tiled matmul of metal/qmm.metal, over one expert's slice of a bank per
 * tile: rows gathered through `perm`, results scattered back to their pairs.
 * Everything between is qw_qmm_q4_g64's inner loop, unchanged. */
struct qw_gmm_args { uint k, n, K, x_by_pair; };

kernel void qw_qmm_q4_gather(
    device const uint    *wb      [[buffer(0)]],   /* bank [E, n, k/8] */
    device const ushort  *sb      [[buffer(1)]],   /* bank [E, n, k/G] bf16 */
    device const ushort  *bb      [[buffer(2)]],
    device const float   *x       [[buffer(3)]],   /* [rows or pairs, k] */
    device       float   *y       [[buffer(4)]],   /* [pairs, n] */
    constant qw_gmm_args &a       [[buffer(5)]],
    device const int     *perm    [[buffer(6)]],   /* [pairs], sorted by expert */
    device const int     *tiles   [[buffer(7)]],   /* [0] = count; then (expert, start, len) */
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
    threadgroup float pool[QW_GMM_POOL_F];
    threadgroup half *As = (threadgroup half *)pool;                     /* [BM][BK] */
    threadgroup half *Bs = (threadgroup half *)pool + QW_GMM_BK * QW_GMM_BM;

    const uint words  = a.k / QW_QPER_WORD;
    const uint groups = a.k / QW_QGROUP;

    /* This threadgroup's tile: up to BM pairs routed to one expert. */
    if ((int)tgid.y >= tiles[0]) return;
    device const int *tl = tiles + 1 + tgid.y * 3;
    const uint expert = (uint)tl[0], first = (uint)tl[1], len = (uint)tl[2];
    device const uint   *w      = wb + (ulong)expert * a.n * words;
    device const ushort *scales = sb + (ulong)expert * a.n * groups;
    device const ushort *biases = bb + (ulong)expert * a.n * groups;
    const uint col0 = tgid.x * QW_GMM_BN;   /* first weight row in this block */

    /* This simdgroup's corner of the output tile. */
    const uint sg_m = sgid / QW_GMM_SG_N;
    const uint sg_n = sgid % QW_GMM_SG_N;
    const uint m_base = sg_m * (QW_GMM_BM / QW_GMM_SG_M);
    const uint n_base = sg_n * (QW_GMM_BN / QW_GMM_SG_N);

    simdgroup_float8x8 acc[QW_GMM_FRAG_M][QW_GMM_FRAG_N];
#pragma unroll
    for (uint i = 0; i < QW_GMM_FRAG_M; ++i)
#pragma unroll
        for (uint j = 0; j < QW_GMM_FRAG_N; ++j)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint k0 = 0; k0 < a.k; k0 += QW_GMM_BK) {

        /* Activations.  Consecutive threads take consecutive k for one token,
         * so each group of 32 reads one contiguous 128-byte span of x. */
        for (uint idx = tid; idx < QW_GMM_BM * QW_GMM_BK; idx += QW_GMM_THREADS) {
            const uint kk = idx % QW_GMM_BK;
            const uint mm = idx / QW_GMM_BK;
            /* The pair's activation row: its own (after the expert's first
             * matmul) or its token's (x_by_pair off: row = pair / K). */
            half v = half(0);
            if (mm < len) {
                const uint p = (uint)perm[first + mm];
                const uint xr = a.x_by_pair ? p : p / a.K;
                v = half(x[(ulong)xr * a.k + k0 + kk]);
            }
            As[mm * QW_GMM_BK + kk] = v;
        }

        /* Weights, one packed word (8 values) per thread.  A word never spans
         * two quantisation groups because 8 divides 64, so one scale and one
         * bias cover the whole word. */
        for (uint idx = tid; idx < QW_GMM_BN * (QW_GMM_BK / QW_QPER_WORD);
             idx += QW_GMM_THREADS) {
            const uint nn = idx / (QW_GMM_BK / QW_QPER_WORD);
            const uint wk = idx % (QW_GMM_BK / QW_QPER_WORD);
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
                Bs[(wk * QW_QPER_WORD + j) * QW_GMM_BN + nn] =
                    half(fma(sc, float((ww >> (4 * j)) & 0xF), bi));
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint ks = 0; ks < QW_GMM_BK; ks += QW_SG_TILE) {
            simdgroup_half8x8 af[QW_GMM_FRAG_M], bf[QW_GMM_FRAG_N];

#pragma unroll
            for (uint i = 0; i < QW_GMM_FRAG_M; ++i)
                simdgroup_load(af[i],
                               As + (m_base + i * QW_SG_TILE) * QW_GMM_BK + ks,
                               QW_GMM_BK, 0, /*transpose=*/false);
#pragma unroll
            for (uint j = 0; j < QW_GMM_FRAG_N; ++j)
                simdgroup_load(bf[j], Bs + ks * QW_GMM_BN + n_base + j * QW_SG_TILE,
                               QW_GMM_BN, 0, /*transpose=*/false);

#pragma unroll
            for (uint i = 0; i < QW_GMM_FRAG_M; ++i)
#pragma unroll
                for (uint j = 0; j < QW_GMM_FRAG_N; ++j)
                    simdgroup_multiply_accumulate(acc[i][j], af[i], bf[j], acc[i][j]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    /* Reuse the operand pool as the output tile.  Every simdgroup has finished
     * reading it above, and the barrier ending the last K step is what makes
     * that safe. */
    threadgroup float *Cs = pool;    /* [BM][BN] */

#pragma unroll
    for (uint i = 0; i < QW_GMM_FRAG_M; ++i)
#pragma unroll
        for (uint j = 0; j < QW_GMM_FRAG_N; ++j)
            simdgroup_store(acc[i][j],
                            Cs + (m_base + i * QW_SG_TILE) * QW_GMM_BN
                               + n_base + j * QW_SG_TILE,
                            QW_GMM_BN, 0, /*transpose=*/false);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* Ragged tiles are handled here rather than by the fragment stores, which
     * always write a full 8x8. */
    for (uint idx = tid; idx < QW_GMM_BM * QW_GMM_BN; idx += QW_GMM_THREADS) {
        const uint mm = idx / QW_GMM_BN;
        const uint nn = idx % QW_GMM_BN;
        const uint gn = col0 + nn;
        if (mm < len && gn < a.n) y[(ulong)perm[first + mm] * a.n + gn] = Cs[idx];
    }
}

/* Matvec against an expert bank: pair p reads matrix idx[p] of the bank and
 * activation row p (x_by_pair) or p / K.  Otherwise qw_qmv_q4_g64 exactly:
 * one simdgroup per QW_QMV_ROWS output rows, lanes walking words. */
struct qw_bank_args { uint k, n, pairs, K, x_by_pair; };

kernel void qw_qmv_q4_bank(
    device const uint    *w       [[buffer(0)]],   /* [E, n, k/8] */
    device const ushort  *scales  [[buffer(1)]],   /* [E, n, k/64] */
    device const ushort  *biases  [[buffer(2)]],
    device const float   *x       [[buffer(3)]],   /* [rows or pairs, k] */
    device const int     *idx     [[buffer(4)]],   /* [pairs] */
    device       float   *y       [[buffer(5)]],   /* [pairs, n] */
    constant qw_bank_args &a      [[buffer(6)]],
    uint3 tgid  [[threadgroup_position_in_grid]],
    uint  sgid  [[simdgroup_index_in_threadgroup]],
    uint  nsg   [[simdgroups_per_threadgroup]],
    uint  lane  [[thread_index_in_simdgroup]])
{
    const uint words  = a.k / QW_QPER_WORD;
    const uint groups = a.k / QW_QGROUP;
    const uint row0 = (tgid.x * nsg + sgid) * QW_QMV_ROWS;
    if (row0 >= a.n) return;

    const uint p = tgid.y;
    const uint e = (uint)idx[p];
    const uint xr = a.x_by_pair ? p : p / a.K;
    device const float  *xv = x + (ulong)xr * a.k;
    device const uint   *we = w      + (ulong)e * a.n * words;
    device const ushort *se = scales + (ulong)e * a.n * groups;
    device const ushort *be = biases + (ulong)e * a.n * groups;

    float acc[QW_QMV_ROWS] = { 0.0f, 0.0f, 0.0f, 0.0f };
    for (uint wi = lane; wi < words; wi += 32) {
        const uint g = wi / QW_WORDS_PER_GROUP;
        float xs[QW_QPER_WORD];
#pragma unroll
        for (int j = 0; j < QW_QPER_WORD; ++j) xs[j] = xv[wi * QW_QPER_WORD + j];
#pragma unroll
        for (uint r = 0; r < QW_QMV_ROWS; ++r) {
            const uint n = row0 + r;
            if (n >= a.n) break;
            const uint  ww = we[(ulong)n * words + wi];
            const float sc = qw_bf16_to_f32(se[(ulong)n * groups + g]);
            const float bi = qw_bf16_to_f32(be[(ulong)n * groups + g]);
            float4 ev, od;
            qw_unpack8_affine(ww, sc * 255.0f, bi, &ev, &od);
#pragma unroll
            for (uint k = 0; k < 4; ++k) {
                acc[r] = fma(ev[k], xs[2 * k],     acc[r]);
                acc[r] = fma(od[k], xs[2 * k + 1], acc[r]);
            }
        }
    }
#pragma unroll
    for (uint r = 0; r < QW_QMV_ROWS; ++r) {
        const float v = simd_sum(acc[r]);
        const uint  n = row0 + r;
        if (lane == 0 && n < a.n) y[(ulong)p * a.n + n] = v;
    }
}

/* act[p, i] = silu(gu[p, i]) * gu[p, I + i]: the fused expert's two halves. */
struct qw_swiglu_split_args { uint pairs, I; };

kernel void qw_swiglu_split(
    device const float *gu  [[buffer(0)]],   /* [pairs, 2I] */
    device       float *act [[buffer(1)]],   /* [pairs, I] */
    constant qw_swiglu_split_args &a [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= a.pairs * a.I) return;
    const uint p = gid / a.I, i = gid % a.I;
    const float g = gu[(ulong)p * 2 * a.I + i], u = gu[(ulong)p * 2 * a.I + a.I + i];
    act[gid] = qw_silu(g) * u;
}

/* out[r, i] = sum_k w[r, k] * y[r*K + k, i]. */
struct qw_combine_args { uint rows, K, H; };

kernel void qw_moe_combine(
    device const float *y   [[buffer(0)]],   /* [rows*K, H] */
    device const float *w   [[buffer(1)]],   /* [rows, K] */
    device       float *out [[buffer(2)]],   /* [rows, H] */
    constant qw_combine_args &a [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= a.rows * a.H) return;
    const uint r = gid / a.H, i = gid % a.H;
    float acc = 0.0f;
    for (uint k = 0; k < a.K; ++k)
        acc = fma(w[r * a.K + k], y[((ulong)r * a.K + k) * a.H + i], acc);
    out[gid] = acc;
}

/* y[r, :] *= sigmoid(g[r]): the shared expert's scalar gate. */
struct qw_rowscale_args { uint rows, dim; };

kernel void qw_scale_rows_sigmoid(
    device       float *y [[buffer(0)]],   /* [rows, dim] */
    device const float *g [[buffer(1)]],   /* [rows] */
    constant qw_rowscale_args &a [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= a.rows * a.dim) return;
    y[gid] *= qw_sigmoid(g[gid / a.dim]);
}

/* ---- gated delta, this family's gate ---------------------------------------
 *
 * rms_norm(x, w) * sigmoid(gate): output_gate_type "sigmoid" where the 27B
 * used silu.  Same kernel shape as qw_rms_norm_gated. */
kernel void qw_rms_norm_gated_sigmoid(
    device const float   *x    [[buffer(0)]],
    device const ushort  *w    [[buffer(1)]],
    device const float   *gate [[buffer(2)]],
    device       float   *y    [[buffer(3)]],
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
        yr[i] = v * qw_sigmoid(gr[i]);
    }
}

/* ---- Qwen Sparse Attention -------------------------------------------------
 *
 * The indexer scores every complete block of `ratio` cached raw keys against
 * a query's normalised, rotated index heads; the top `block_topk` blocks plus
 * the incomplete tail are what attention may see.  One simdgroup per
 * (query, block): lanes own index-head dims lane, lane+32, ..., which puts a
 * rotary pair (j, j+32) in one lane -- so rotary_dim must be 64, and the
 * dispatcher refuses anything else. */
struct qw_qsa_score_args {
    uint rows, nq, d, ratio, base_pos, rotary_dim, max_blocks;
    float eps;
};

#define QW_QSA_MAXM 8   /* index head dims per lane: d <= 256 */

kernel void qw_qsa_scores(
    device const float  *qn       [[buffer(0)]],   /* [rows, nq, d] normed + rotated */
    device const float  *ikeys    [[buffer(1)]],   /* [max_ctx, d] raw keys */
    device const ushort *kw       [[buffer(2)]],   /* [d] bf16 k_layernorm (+1) */
    device const float  *inv_freq [[buffer(3)]],   /* [rotary_dim/2] */
    device       float  *scores   [[buffer(4)]],   /* [rows, max_blocks] */
    constant qw_qsa_score_args &a [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  nsg  [[simdgroups_per_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    const uint r = tgid.y;
    const uint b = tgid.x * nsg + sgid;
    const uint visible  = a.base_pos + r + 1;
    const uint n_blocks = visible / a.ratio;
    if (b >= n_blocks) return;
    const uint M = a.d / 32;

    /* pooled raw key, one lane strip at a time */
    float kn[QW_QSA_MAXM];
    float ss = 0.0f;
    for (uint m = 0; m < M; ++m) {
        const uint dim = lane + 32 * m;
        float acc = 0.0f;
        for (uint t = 0; t < a.ratio; ++t) acc += ikeys[(ulong)(b * a.ratio + t) * a.d + dim];
        kn[m] = acc / float(a.ratio);
        ss = fma(kn[m], kn[m], ss);
    }
    ss = simd_sum(ss);
    const float inv = rsqrt(ss / float(a.d) + a.eps);
    for (uint m = 0; m < M; ++m) kn[m] = kn[m] * inv * qw_bf16_to_f32(kw[lane + 32 * m]);

    /* rope on the first rotary_dim dims at the block's first position:
     * pairs (j, j+32) are lane strips 0 and 1. */
    {
        const float angle = float(b * a.ratio) * inv_freq[lane];
        const float c = cos(angle), s = sin(angle);
        const float x0 = kn[0], x1 = kn[1];
        kn[0] = x0 * c - x1 * s;
        kn[1] = x1 * c + x0 * s;
    }

    float score = 0.0f;
    for (uint h = 0; h < a.nq; ++h) {
        device const float *qh = qn + ((ulong)r * a.nq + h) * a.d;
        float dot = 0.0f;
        for (uint m = 0; m < M; ++m) dot = fma(qh[lane + 32 * m], kn[m], dot);
        dot = simd_sum(dot);
        if (dot > 0.0f) score += dot;
    }
    if (lane == 0) scores[(ulong)r * a.max_blocks + b] = score / sqrt(float(a.d));
}

/* Top-k blocks per query into a byte mask over cache positions, plus the
 * tail.  One threadgroup per query; k rounds of a parallel argmax, ties to
 * the lowest index, matching the reference.  O(k * n_blocks) per query --
 * the simplest correct shape, and the one to measure before replacing. */
struct qw_qsa_select_args { uint rows, ratio, base_pos, block_topk, max_ctx, max_blocks; };

/* Block selection: mark the `block_topk` best-scoring complete blocks (ties to
 * the lower index) and the incomplete tail as visible.
 *
 * A radix select, one threadgroup of QW_SEL_THREADS per query: scores map to
 * unsigned keys that order as the floats do, four 8-bit passes of a
 * threadgroup histogram narrow to the exact key T of the k-th best, and then
 * everything above T is taken and the rest of the budget filled from keys
 * equal to T in index order.  Four passes over the blocks, where the version
 * this replaced ran one round per selected block -- 512 of them per layer on
 * every step past the budget, 7 t/s at 4K context. */
#define QW_SEL_THREADS 1024

static inline uint qw_order_key(float v) {
    const uint u = as_type<uint>(v + 0.0f);          /* -0 -> +0 */
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

static inline void qw_sel_mark(device uchar *mr, uint b, uint ratio) {
    for (uint t = 0; t < ratio; ++t) mr[b * ratio + t] = 1;
}

kernel void qw_qsa_select(
    device       float *scores [[buffer(0)]],   /* [rows, max_blocks] */
    device       uchar *mask   [[buffer(1)]],   /* [rows, max_ctx] */
    constant qw_qsa_select_args &a [[buffer(2)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint ntg  [[threads_per_threadgroup]])
{
    threadgroup atomic_uint hist[256];
    threadgroup uint counts[QW_SEL_THREADS];
    threadgroup uint sh_prefix, sh_need;

    const uint r = tgid;
    const uint n_keys = a.base_pos + r + 1;
    const uint n_blocks = n_keys / a.ratio;
    device const float *sc = scores + (ulong)r * a.max_blocks;
    device uchar *mr = mask + (ulong)r * a.max_ctx;

    for (uint t = tid; t < n_keys; t += ntg) mr[t] = (t >= n_blocks * a.ratio) ? 1 : 0;
    threadgroup_barrier(mem_flags::mem_device);

    const uint take = min(a.block_topk, n_blocks);
    if (take == 0) return;
    if (take == n_blocks) {
        for (uint b = tid; b < n_blocks; b += ntg) qw_sel_mark(mr, b, a.ratio);
        return;
    }

    /* The k-th best key, eight bits at a time from the top. */
    uint prefix = 0, known = 0, need = take;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (uint i = tid; i < 256; i += ntg) atomic_store_explicit(&hist[i], 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint b = tid; b < n_blocks; b += ntg) {
            const uint key = qw_order_key(sc[b]);
            if ((key & known) == prefix)
                atomic_fetch_add_explicit(&hist[(key >> shift) & 255u], 1u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            uint above = 0;
            int d = 255;
            for (; d > 0; --d) {
                const uint h = atomic_load_explicit(&hist[d], memory_order_relaxed);
                if (above + h >= need) break;
                above += h;
            }
            sh_prefix = prefix | ((uint)d << shift);
            sh_need = need - above;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        prefix = sh_prefix;
        need = sh_need;
        known |= 255u << shift;
    }
    const uint T = prefix;           /* `need` of the keys equal to T are still to take */

    /* Everything above T; then the first `need` equal to T, by index.  Each
     * thread owns a contiguous run of blocks so index order is a prefix sum. */
    const uint per = (n_blocks + ntg - 1) / ntg;
    const uint lo = min(tid * per, n_blocks), hi = min(lo + per, n_blocks);
    uint eq = 0;
    for (uint b = lo; b < hi; ++b) {
        const uint key = qw_order_key(sc[b]);
        if (key > T) qw_sel_mark(mr, b, a.ratio);
        else if (key == T) eq++;
    }
    counts[tid] = eq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint run = 0;
        for (uint t = 0; t < ntg; ++t) { const uint c = counts[t]; counts[t] = run; run += c; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint rank = counts[tid];
    for (uint b = lo; b < hi && rank < need; ++b)
        if (qw_order_key(sc[b]) == T) { qw_sel_mark(mr, b, a.ratio); rank++; }
}

/* qw_attn_decode with a per-(query, position) byte mask: a masked key
 * contributes nothing.  Identical otherwise, head_dim 256. */
kernel void qw_attn_masked(
    device const float   *q    [[buffer(0)]],
    device const half    *kc   [[buffer(1)]],
    device const half    *vc   [[buffer(2)]],
    device       float   *out  [[buffer(3)]],
    device const uchar   *mask [[buffer(4)]],   /* [rows, max_ctx] */
    constant qw_attn_args &a   [[buffer(5)]],
    uint tgid     [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]])
{
    threadgroup float tg_out[QW_ATTN_SIMDS * QW_ATTN_LANES];
    threadgroup float tg_max[QW_ATTN_SIMDS];
    threadgroup float tg_sum[QW_ATTN_SIMDS];

    const uint row = tgid / a.q_heads;
    const uint qh  = tgid % a.q_heads;
    const uint kvh = qh / a.gqa;
    const int n_keys = (int)(a.base_pos + row) + 1;
    device const uchar *mr = mask + (ulong)row * a.max_ctx;

    const device float *qp = q + ((ulong)row * a.q_heads + qh) * QW_ATTN_D + simd_lid * QW_ATTN_PER_THREAD;
    const device half  *kp = kc + ((ulong)kvh * a.max_ctx + simd_gid) * QW_ATTN_D + simd_lid * QW_ATTN_PER_THREAD;
    const device half  *vp = vc + ((ulong)kvh * a.max_ctx + simd_gid) * QW_ATTN_D + simd_lid * QW_ATTN_PER_THREAD;

    float qv[QW_ATTN_PER_THREAD];
    float acc[QW_ATTN_PER_THREAD];
#pragma unroll
    for (int i = 0; i < QW_ATTN_PER_THREAD; ++i) { qv[i] = a.scale * qp[i]; acc[i] = 0.0f; }

    float run_max = -FLT_MAX;
    float run_sum = 0.0f;
    for (int t = (int)simd_gid; t < n_keys; t += QW_ATTN_SIMDS) {
        if (mr[t]) {
            float score = 0.0f;
#pragma unroll
            for (int i = 0; i < QW_ATTN_PER_THREAD; ++i) score = fma(qv[i], float(kp[i]), score);
            score = simd_sum(score);
            const float new_max = max(run_max, score);
            const float factor  = exp(run_max - new_max);
            const float w       = exp(score - new_max);
            run_max = new_max;
            run_sum = run_sum * factor + w;
#pragma unroll
            for (int i = 0; i < QW_ATTN_PER_THREAD; ++i) acc[i] = fma(acc[i], factor, w * float(vp[i]));
        }
        kp += QW_ATTN_SIMDS * QW_ATTN_D;
        vp += QW_ATTN_SIMDS * QW_ATTN_D;
    }

    if (simd_lid == 0) { tg_max[simd_gid] = run_max; tg_sum[simd_gid] = run_sum; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float part_max = tg_max[simd_lid];
    const float glob_max = simd_max(part_max);
    const float rescale  = exp(part_max - glob_max);
    const float glob_sum = simd_sum(tg_sum[simd_lid] * rescale);
    for (int i = 0; i < QW_ATTN_PER_THREAD; ++i) {
        tg_out[simd_lid * QW_ATTN_LANES + simd_gid] = acc[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float v = simd_sum(tg_out[simd_gid * QW_ATTN_LANES + simd_lid] * rescale);
        acc[i] = glob_sum == 0.0f ? v : v / glob_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (simd_lid == 0) {
        device float *op = out + ((ulong)row * a.q_heads + qh) * QW_ATTN_D + simd_gid * QW_ATTN_PER_THREAD;
#pragma unroll
        for (int i = 0; i < QW_ATTN_PER_THREAD; ++i) op[i] = acc[i];
    }
}

/* ---- the engram layer ------------------------------------------------------ */

/* Per stream: g = key.query / sqrt(H), signed square root, sigmoid; the
 * stream's gated value is that times the shared value.  One simdgroup per
 * (row, stream). */
kernel void qw_ple_gate(
    device const float *keyn  [[buffer(0)]],   /* [rows, S*H] */
    device const float *qn    [[buffer(1)]],   /* [rows, S*H] */
    device const float *value [[buffer(2)]],   /* [rows, H] */
    device       float *gv    [[buffer(3)]],   /* [rows, S*H] */
    constant qw_hc_args &a    [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint r = tgid, s = sgid;
    if (s >= a.S) return;
    const ulong base = (ulong)r * a.S * a.H + (ulong)s * a.H;
    float dot = 0.0f;
    for (uint i = lane; i < a.H; i += 32) dot = fma(keyn[base + i], qn[base + i], dot);
    dot = simd_sum(dot);
    float g = dot / sqrt(float(a.H));
    const float mag = max(fabs(g), 1e-6f);
    g = (g < 0.0f ? -1.0f : 1.0f) * sqrt(mag);
    const float sg = qw_sigmoid(g);
    for (uint i = lane; i < a.H; i += 32) gv[base + i] = sg * value[(ulong)r * a.H + i];
}

/* Depthwise causal conv with dilation: kernel 4, dilation 3, so tap j reads
 * the input 3*(3-j) steps back.  The state holds the last nine inputs per
 * channel, oldest first.  One thread per channel, like qw_conv1d_causal_silu. */
#define QW_PLE_K   4
#define QW_PLE_DIL 3
#define QW_PLE_SL  ((QW_PLE_K - 1) * QW_PLE_DIL)   /* 9 */

struct qw_dconv_args { uint channels, rows; };

kernel void qw_conv1d_dilated_silu(
    device const float  *x     [[buffer(0)]],   /* [rows, channels] */
    device       float  *state [[buffer(1)]],   /* [SL, channels] */
    device const ushort *w     [[buffer(2)]],   /* [channels, K] bf16, tap 0 oldest */
    device       float  *y     [[buffer(3)]],   /* [rows, channels] */
    constant qw_dconv_args &a  [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= a.channels) return;
    float taps[QW_PLE_K];
#pragma unroll
    for (uint j = 0; j < QW_PLE_K; ++j) taps[j] = qw_bf16_to_f32(w[gid * QW_PLE_K + j]);
    float win[QW_PLE_SL];
#pragma unroll
    for (uint j = 0; j < QW_PLE_SL; ++j) win[j] = state[(ulong)j * a.channels + gid];

    for (uint t = 0; t < a.rows; ++t) {
        const float cur = x[(ulong)t * a.channels + gid];
        float acc = taps[QW_PLE_K - 1] * cur;
#pragma unroll
        for (uint j = 0; j < QW_PLE_K - 1; ++j)
            acc = fma(win[QW_PLE_SL - (QW_PLE_K - 1 - j) * QW_PLE_DIL], taps[j], acc);
        y[(ulong)t * a.channels + gid] = qw_silu(acc);
#pragma unroll
        for (uint j = 0; j < QW_PLE_SL - 1; ++j) win[j] = win[j + 1];
        win[QW_PLE_SL - 1] = cur;
    }
#pragma unroll
    for (uint j = 0; j < QW_PLE_SL; ++j) state[(ulong)j * a.channels + gid] = win[j];
}
