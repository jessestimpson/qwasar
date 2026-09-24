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

        /* Activations first, held across every output row: a token's eight
         * values are read once and spent against all of them. */
        float xs[QW_QMVB_B][QW_QPER_WORD];
#pragma unroll
        for (uint b = 0; b < QW_QMVB_B; ++b) {
            /* A dead token lane re-reads a live token's activations, which is a
             * cache hit, and computes a real dot product that is thrown away. */
            device const float *xv = x + (ulong)min(b, a.rows - 1) * a.k
                                       + wi * QW_QPER_WORD;
#pragma unroll
            for (uint j = 0; j < QW_QPER_WORD; ++j) xs[b][j] = xv[j];
        }

#pragma unroll
        for (uint r = 0; r < QW_QMVB_ROWS; ++r) {
            const uint  n  = min(row0 + r, a.n - 1);
            const uint  ww = w[(ulong)n * words + wi];
            const float sc = qw_bf16_to_f32(scales[(ulong)n * groups + g]);
            const float bi = qw_bf16_to_f32(biases[(ulong)n * groups + g]);
            float4 ev, od;
            qw_unpack8_affine(ww, sc * 255.0f, bi, &ev, &od);

#pragma unroll
            for (uint b = 0; b < QW_QMVB_B; ++b)
#pragma unroll
                for (uint k = 0; k < 4; ++k) {
                    acc[r][b] = fma(ev[k], xs[b][2 * k],     acc[r][b]);
                    acc[r][b] = fma(od[k], xs[b][2 * k + 1], acc[r][b]);
                }
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

/* Split-K matvec, for one token against a matrix with few outputs.
 *
 * qw_qmv_q4_g64 gives each simdgroup its own output rows and walks the whole
 * input for each, so its parallelism is n / 4 simdgroups.  Flash-Next's decode
 * is full of matrices where that is a handful: the hyper-connection mixer's
 * 320 x 10240 ran as 10 threadgroups at 57 GB/s -- 3.5 ms of every token --
 * and the 48-, 512- and 640-row projections fared little better.  Here a
 * threadgroup owns QW_SK_ROWS rows and its simdgroups split the input between
 * them; partial sums meet in threadgroup memory.  The per-word arithmetic is
 * qmv's. */
#define QW_SK_ROWS 4

/* QW_SK_ROWS rows of y = W . x from row0, the rows' input split across the
 * threadgroup's simdgroups.  Shared by the dense, bank and fused kernels. */
static inline void qw_sk_rows(device const uint *w, device const ushort *scales,
                              device const ushort *biases, device const float *x,
                              device float *y, uint k, uint nrows, uint row0,
                              threadgroup float (*part)[QW_SK_ROWS],
                              uint sgid, uint nsg, uint lane)
{
    const uint words  = k / QW_QPER_WORD;
    const uint groups = k / QW_QGROUP;

    /* This simdgroup's run of words, the same run for every row. */
    const uint per = (words + nsg - 1) / nsg;
    const uint w0 = min(sgid * per, words), w1 = min(w0 + per, words);

    float acc[QW_SK_ROWS] = { 0.0f, 0.0f, 0.0f, 0.0f };
    for (uint wi = w0 + lane; wi < w1; wi += 32) {
        const uint g = wi / QW_WORDS_PER_GROUP;
        float xs[QW_QPER_WORD];
#pragma unroll
        for (int j = 0; j < QW_QPER_WORD; ++j) xs[j] = x[wi * QW_QPER_WORD + j];
#pragma unroll
        for (uint r = 0; r < QW_SK_ROWS; ++r) {
            const uint n = row0 + r;
            if (n >= nrows) break;
            const uint  ww = w[(ulong)n * words + wi];
            const float sc = qw_bf16_to_f32(scales[(ulong)n * groups + g]);
            const float bi = qw_bf16_to_f32(biases[(ulong)n * groups + g]);
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
    for (uint r = 0; r < QW_SK_ROWS; ++r) {
        const float v = simd_sum(acc[r]);
        if (lane == 0) part[sgid][r] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0 && lane < QW_SK_ROWS && row0 + lane < nrows) {
        float s = 0.0f;
        for (uint i = 0; i < nsg; ++i) s += part[i][lane];
        y[row0 + lane] = s;
    }
}

kernel void qw_qmv_q4_splitk(
    device const uint    *w       [[buffer(0)]],   /* [n, k/8] */
    device const ushort  *scales  [[buffer(1)]],   /* [n, k/G] */
    device const ushort  *biases  [[buffer(2)]],
    device const float   *x       [[buffer(3)]],   /* [1, k] */
    device       float   *y       [[buffer(4)]],   /* [1, n] */
    constant qw_matmul_args &a    [[buffer(5)]],
    uint3 tgid  [[threadgroup_position_in_grid]],
    uint  sgid  [[simdgroup_index_in_threadgroup]],
    uint  nsg   [[simdgroups_per_threadgroup]],
    uint  lane  [[thread_index_in_simdgroup]])
{
    threadgroup float part[32][QW_SK_ROWS];
    qw_sk_rows(w, scales, biases, x, y, a.k, a.n, tgid.x * QW_SK_ROWS, part, sgid, nsg, lane);
}

/* Up to QW_MULTI_MAX one-token matvecs sharing their input in one dispatch:
 * a gated-delta layer's four in-projections, an attention layer's q, k and
 * v, an MLP's gate and up.  Separate dispatches on the serial encoder each
 * drain before the next starts -- the small ones (a, b: 48 rows) leave the
 * GPU nearly idle for their whole duration.  A threadgroup takes rows of
 * whichever matrix its block falls in: QW_SK_ROWS of a narrow one, its input
 * split across the simdgroups, or QW_SK_ROWS per simdgroup of a wide one
 * (the `wide` bit), each row whole -- qw_qmv_q4_g64's plan, which a wide
 * matrix fills the GPU with anyway and which beats the split there. */
#define QW_MULTI_MAX 4
struct qw_multi_args {
    uint k;
    uint count;
    uint wide;                  /* bit i: part i one simdgroup per row block */
    uint n[QW_MULTI_MAX];
    uint end[QW_MULTI_MAX];     /* cumulative threadgroup blocks */
};

kernel void qw_qmv_q4_multi_splitk(
    device const float   *x       [[buffer(0)]],   /* [1, k] */
    constant qw_multi_args &a     [[buffer(1)]],
    device const uint    *w0 [[buffer(2)]],  device const ushort *s0 [[buffer(3)]],
    device const ushort  *b0 [[buffer(4)]],  device       float  *y0 [[buffer(5)]],
    device const uint    *w1 [[buffer(6)]],  device const ushort *s1 [[buffer(7)]],
    device const ushort  *b1 [[buffer(8)]],  device       float  *y1 [[buffer(9)]],
    device const uint    *w2 [[buffer(10)]], device const ushort *s2 [[buffer(11)]],
    device const ushort  *b2 [[buffer(12)]], device       float  *y2 [[buffer(13)]],
    device const uint    *w3 [[buffer(14)]], device const ushort *s3 [[buffer(15)]],
    device const ushort  *b3 [[buffer(16)]], device       float  *y3 [[buffer(17)]],
    uint3 tgid  [[threadgroup_position_in_grid]],
    uint  sgid  [[simdgroup_index_in_threadgroup]],
    uint  nsg   [[simdgroups_per_threadgroup]],
    uint  lane  [[thread_index_in_simdgroup]])
{
    threadgroup float part[32][QW_SK_ROWS];
    const uint blk = tgid.x;
    uint m = 0;
    while (m + 1 < a.count && blk >= a.end[m]) ++m;
    const uint local = blk - (m ? a.end[m - 1] : 0);
    /* Wide: each simdgroup its own rows, as a threadgroup of one. */
    const bool wide = (a.wide >> m) & 1u;
    const uint row0 = (wide ? local * nsg + sgid : local) * QW_SK_ROWS;
    threadgroup float (*pt)[QW_SK_ROWS] = wide ? part + sgid : part;
    const uint sg = wide ? 0u : sgid, ns = wide ? 1u : nsg;
    switch (m) {
    case 0:  qw_sk_rows(w0, s0, b0, x, y0, a.k, a.n[0], row0, pt, sg, ns, lane); break;
    case 1:  qw_sk_rows(w1, s1, b1, x, y1, a.k, a.n[1], row0, pt, sg, ns, lane); break;
    case 2:  qw_sk_rows(w2, s2, b2, x, y2, a.k, a.n[2], row0, pt, sg, ns, lane); break;
    default: qw_sk_rows(w3, s3, b3, x, y3, a.k, a.n[3], row0, pt, sg, ns, lane); break;
    }
}
