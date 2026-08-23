#ifndef QWASAR_GPU_H
#define QWASAR_GPU_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* GPU boundary used by qwasar.c.  The Metal implementation lives in
 * qwasar_metal.m; this header stays free of Objective-C so the engine
 * translation unit remains plain C99. */

/* Opaque handle to a device buffer.  Weight buffers wrap an mmap of a
 * safetensors shard with no copy, so a "buffer" is often 5 GB of file-backed
 * pages that the kernel faults in on first touch. */
typedef struct qw_buf_s *qw_buf;

/* Brings up the device, command queue, and shader library.  The library is
 * compiled from source embedded in the binary (see tools/bin2c.c) and cached
 * on disk keyed by a hash of that source, so only the first run pays.
 * Returns false and fills `err` on failure. */
bool qw_gpu_init(char *err, size_t errcap);
void qw_gpu_shutdown(void);

/* Device properties, for budgeting and for --info. */
const char *qw_gpu_name(void);
uint64_t    qw_gpu_working_set_limit(void);  /* recommendedMaxWorkingSetSize */
uint64_t    qw_gpu_max_buffer_length(void);

/* Wraps host memory without copying.  `ptr` must be page-aligned and stay
 * valid for the buffer's lifetime -- this is the mmap path. */
qw_buf qw_buf_wrap(void *ptr, size_t len);
/* Allocates shared (CPU+GPU visible) device memory. */
qw_buf qw_buf_alloc(size_t len);
void   qw_buf_free(qw_buf b);
void  *qw_buf_contents(qw_buf b);
size_t qw_buf_length(qw_buf b);

/* ---- dispatch -------------------------------------------------------------
 *
 * A qw_cmd is one command buffer under construction.  Ops append to it; nothing
 * runs until commit.  The graph encodes a whole forward pass into one command
 * buffer so a step costs a single CPU->GPU handoff. */

/* A tensor argument: a buffer plus a byte offset into it.  Weight buffers hold
 * an entire safetensors shard, so almost every argument is an interior view. */
typedef struct { qw_buf buf; size_t off; } qw_ref;

static inline qw_ref qw_ref_at(qw_buf b, size_t off) { qw_ref r = { b, off }; return r; }

/* A view further into an existing view; for walking blocks of a tensor. */
static inline qw_ref qw_ref_offset(qw_ref r, size_t off) { r.off += off; return r; }

typedef struct qw_cmd_s *qw_cmd;

qw_cmd qw_cmd_begin(void);
void   qw_cmd_commit(qw_cmd c);   /* submit, return immediately */
void   qw_cmd_wait(qw_cmd c);     /* submit and block until done */
void   qw_cmd_free(qw_cmd c);
/* Error string from the last completed command buffer, or NULL. */
const char *qw_cmd_error(qw_cmd c);

/* y[r][n] = sum_k dequant(w)[n][k] * x[r][k]
 *
 *   w       U32  [n, k/8]     MLX affine 4-bit, group 64
 *   scales  BF16 [n, k/64]
 *   biases  BF16 [n, k/64]
 *   x       F32  [rows, k]
 *   y       F32  [rows, n]
 *
 * `rows` is the token count: 1 while decoding, the chunk size while prefilling.
 *
 * qw_op_qmat_q4 is what the graph calls; it picks among the three
 * implementations by shape.  The matvec reads every weight once per token,
 * which is optimal at rows = 1 and linearly wasteful beyond it; the matmul
 * tiles over tokens so a weight block is read and dequantised once for the
 * whole tile, but pads its token tile to 64 and so wastes work on small counts;
 * the batched matvec covers the gap between them, reading the weights once for
 * a block of up to QW_QMVB_B tokens.  The others are exposed for tests and
 * benchmarks. */
void qw_op_qmat_q4(qw_cmd c, qw_ref y, qw_ref x,
                   qw_ref w, qw_ref scales, qw_ref biases,
                   int32_t k, int32_t n, int32_t rows);

void qw_op_qmv_q4(qw_cmd c, qw_ref y, qw_ref x,
                  qw_ref w, qw_ref scales, qw_ref biases,
                  int32_t k, int32_t n, int32_t rows);

void qw_op_qmm_q4(qw_cmd c, qw_ref y, qw_ref x,
                  qw_ref w, qw_ref scales, qw_ref biases,
                  int32_t k, int32_t n, int32_t rows);

/* y = x * w^T for an unquantised bf16 weight, blocked like qw_op_qmvb_q4.
 * Only the MTP draft head is dense. */
void qw_op_dmat_bf16(qw_cmd c, qw_ref y, qw_ref x, qw_ref w,
                     int32_t k, int32_t n, int32_t rows);

/* Dense bf16, tiled over tokens: the vision tower's shape.  Falls back to the
 * blocked matvec below the matmul's crossover. */
void qw_op_dmm_bf16(qw_cmd c, qw_ref y, qw_ref x, qw_ref w,
                    int32_t k, int32_t n, int32_t rows);

/* Evaluates `rows` tokens in ceil(rows / QW_QMVB_B) passes over the weights.
 * Correct for any row count; only worth calling below QW_QMM_MIN_ROWS. */
void qw_op_qmvb_q4(qw_cmd c, qw_ref y, qw_ref x,
                   qw_ref w, qw_ref scales, qw_ref biases,
                   int32_t k, int32_t n, int32_t rows);

/* Token count at or above which qw_op_qmat_q4 switches to the tiled matmul.
 *
 * The matmul pads its token tile to QW_QMM_BM and so costs the same for 1 token
 * as for 64; the batched matvec costs one flat block per QW_QMVB_B tokens.  On
 * gate_proj that is 4.7 ms against 0.85 ms per block, so the matmul wins from
 * the sixth block on.  down_proj crosses in the same place, which is why one
 * constant is enough.
 *
 * This was 16 when the choice was between the matmul and the single-token
 * matvec.  The batched matvec pushed it out to 21 by being faster than both
 * everywhere in between. */
#define QW_QMM_MIN_ROWS 21

/* Blocking for qw_qmvb_q4_g64, injected into the Metal compile alongside the
 * matmul's.  B is the token block: a whole block is evaluated against one pass
 * over the weights, so a speculative verify costs one flat block rather than
 * one decode step per token.
 *
 * Both numbers are measured, and the sweep is worth recording because the
 * plausible reasoning was wrong.  Arithmetic intensity here is 4B FLOP per
 * weight byte against a machine ratio of 37, which says B = 8 should still be
 * bandwidth-bound.  It was not, and the reason was that half the ALU went on
 * unpacking nibbles rather than on the dot product.  Measured cost per block of
 * B tokens on gate_proj, best of three, against 0.53 ms for one pass over its
 * weights:
 *
 *   B=4 ROWS=4  0.85 ms   <- shipped
 *   B=4 ROWS=2  1.13 ms
 *   B=5 ROWS=4  1.28 ms
 *   B=8 ROWS=4  1.78 ms
 *   B=4 ROWS=8  1.99 ms
 *
 * Those are from before qw_unpack8_affine cut the unpacking to a seventh of
 * what it was (metal/common.metal).  B = 4 now costs 0.61 ms, or 82 GB/s
 * against the single-token kernel's 94 -- close enough to bandwidth that the
 * shape of the sweep is unlikely to have changed, though only the winner was
 * re-measured.
 *
 * ROWS is output rows per simdgroup; ROWS * B accumulators have to stay in
 * registers, which is what caps both.  The tiled matmul was tried here too and
 * loses: at BM = 8 it costs 1.14 ms for the same work, because a narrow token
 * tile cannot amortise its threadgroup staging. */
#define QW_QMVB_B    4
#define QW_QMVB_ROWS 4

/* Tiling for qw_qmm_q4_g64.  These are the single source of truth: the host
 * derives its dispatch geometry from them and injects them into the Metal
 * compile as preprocessor macros, so the kernel and the launch cannot disagree.
 *
 * The output tile is BM tokens by BN weight rows, walked over K in steps of BK.
 * It is divided among an SG_M x SG_N grid of simdgroups, each of which owns a
 * (BM/SG_M) x (BN/SG_N) region and accumulates it in 8x8 matrix fragments.
 *
 *   threads per threadgroup = SG_M * SG_N * 32
 *   threadgroup memory      = max(BK * (BM + BN), BM * BN) * 4 bytes, under 32 KB
 *
 * BK must be a multiple of 8 and divide 64, so a fragment step stays inside one
 * quantisation group and a packed word never spans two. */
#define QW_QMM_BM 64
#define QW_QMM_BN 64
#define QW_QMM_BK 32
#define QW_QMM_SG_M 2
#define QW_QMM_SG_N 4
#define QW_QMM_THREADS (QW_QMM_SG_M * QW_QMM_SG_N * 32)

/* RMS norm over the last axis of an [rows, dim] fp32 buffer.
 *
 * `weight` may be a null ref, which is how the gated-delta path expresses an
 * unweighted normalisation.  `out_scale` is applied after normalising and
 * before the weight; it turns an RMS norm into the scaled L2 norm that path
 * wants (see metal/norm.metal). */
void qw_op_rms_norm(qw_cmd c, qw_ref y, qw_ref x, qw_ref weight,
                    int32_t dim, int32_t rows, float eps, float out_scale);

/* Two RMS norms into one [rows, 2 * dim] buffer: norm_e(e) then norm_h(h).
 * The MTP head's fusion step; the concatenation order is the head's. */
void qw_op_rms_norm_concat(qw_cmd c, qw_ref y, qw_ref e, qw_ref we,
                           qw_ref h, qw_ref wh, int32_t dim, int32_t rows,
                           float eps);

/* ---- vision tower ----------------------------------------------------------
 *
 * A second set of primitives: the tower is bf16 with biases and LayerNorm
 * throughout, where the text model is 4-bit, bias-free and RMSNorm. */

/* LayerNorm -- mean subtracted as well as scaled, unlike RMSNorm. */
void qw_op_layer_norm(qw_cmd c, qw_ref y, qw_ref x, qw_ref weight, qw_ref bias,
                      int32_t dim, int32_t rows, float eps);

/* GELU, tanh approximation, in place.  The definition this model trained on. */
void qw_op_gelu_tanh(qw_cmd c, qw_ref y, int32_t n);

/* Adds a bf16 bias to every row of a [rows, dim] buffer. */
void qw_op_add_bias(qw_cmd c, qw_ref y, qw_ref bias, int32_t dim, int32_t rows);

/* 2D rotary embedding over [tokens, heads, dim], half-split.  `angles` is
 * [tokens, dim/2]: a patch's row frequencies followed by its column's.  Shares
 * nothing with qw_op_rope_partial but the idea. */
void qw_op_rope_2d(qw_cmd c, qw_ref x, qw_ref angles,
                   int32_t tokens, int32_t heads, int32_t dim, int32_t stride);

/* Bidirectional attention over the patches of one frame.  `qkv` is
 * [tokens, 3, heads, dim] as the fused projection leaves it.  `segment` is how
 * many tokens a frame has: a patch never attends outside its own frame, and an
 * image is simply the one-segment case. */
void qw_op_vision_attn(qw_cmd c, qw_ref out, qw_ref qkv, int32_t tokens,
                       int32_t heads, int32_t dim, int32_t segment, float scale);

/* rms_norm(x, weight) * silu(gate) -- the gated-delta output norm. */
void qw_op_rms_norm_gated(qw_cmd c, qw_ref y, qw_ref x, qw_ref weight, qw_ref gate,
                          int32_t dim, int32_t rows, float eps, float out_scale);

/* y = silu(gate) * up, elementwise over n values. */
void qw_op_swiglu(qw_cmd c, qw_ref y, qw_ref gate, qw_ref up, int32_t n);

/* ---- gated delta ----------------------------------------------------------
 *
 * These carry recurrent state across calls, which is the structural difference
 * from every other op here: `state` is read and written in place, and a session
 * therefore cannot rewind its history the way a KV cache can. */

/* Causal depthwise conv over [rows, channels], with silu fused.
 * `state` is [ksize-1, channels] fp32, updated in place to the new tail. */
/* `snap` receives the state as it stood after each of the first `n_snap`
 * rows, `snap_stride` floats apart, so a speculative verify can be rewound to
 * any accepted prefix.  Pass n_snap = 0 to skip it. */
void qw_op_conv1d_causal_silu(qw_cmd c, qw_ref y, qw_ref x, qw_ref state,
                              qw_ref weight, int32_t channels, int32_t rows,
                              int32_t ksize, qw_ref snap, int32_t n_snap,
                              int32_t snap_stride);

/* g = exp(-exp(A_log) * softplus(a + dt_bias)),  beta = sigmoid(b). */
void qw_op_gdn_gates(qw_cmd c, qw_ref g, qw_ref beta, qw_ref a, qw_ref b,
                     qw_ref A_log, qw_ref dt_bias, int32_t n_v_heads, int32_t rows);

/* The delta-rule recurrence.  `state` is [hv, dv, dk] fp32, in and out.
 * q and k must already be l2-normalised. */
/* `snap` as above: the recurrent state after each of the first `n_snap`
 * timesteps.  The delta rule cannot be run backwards, so keeping the state on
 * the way past is the only way to undo a rejected draft. */
void qw_op_gated_delta(qw_cmd c, qw_ref y, qw_ref q, qw_ref k, qw_ref v,
                       qw_ref g, qw_ref beta, qw_ref state,
                       int32_t hk, int32_t hv, int32_t dk, int32_t dv, int32_t rows,
                       qw_ref snap, int32_t n_snap, int32_t snap_stride);

/* ---- rope, embedding, plumbing -------------------------------------------- */

/* Dequantises one embedding row per token into y[n_tokens, hidden]. */
void qw_op_embed_q4(qw_cmd c, qw_ref y, qw_ref tokens,
                    qw_ref w, qw_ref scales, qw_ref biases,
                    int32_t hidden, int32_t n_tokens);

/* Partial multimodal RoPE, applied in place to [rows, heads, head_dim].
 *
 *   pos       int32 [3, rows]            positions along t, h, w
 *   axis      uint8 [rotary_dim/2]       which axis each frequency reads
 *   inv_freq  f32   [rotary_dim/2]
 *
 * Build the last two with qw_rope_tables(); for text-only input all three
 * position rows are identical and the axis selection is a no-op. */
void qw_op_rope_partial(qw_cmd c, qw_ref x, qw_ref pos, qw_ref axis, qw_ref inv_freq,
                        int32_t rows, int32_t heads, int32_t head_dim, int32_t rotary_dim);

/* Fills the interleaved-MRoPE axis selector and the inverse frequencies.
 * Both arrays are rotary_dim/2 long. */
void qw_rope_tables(uint8_t *axis, float *inv_freq, int32_t rotary_dim,
                    float theta, const int32_t mrope_section[3]);

void qw_op_add_inplace(qw_cmd c, qw_ref y, qw_ref x, int32_t n);
void qw_op_mul_sigmoid(qw_cmd c, qw_ref y, qw_ref gate, int32_t n);

/* dst[r, 0:len] = src[r, offset:offset+len] -- de-strides a column range. */
void qw_op_slice_rows(qw_cmd c, qw_ref dst, qw_ref src,
                      int32_t rows, int32_t src_stride, int32_t offset, int32_t len);

/* Splits [rows, heads, 2*dim] into two contiguous [rows, heads, dim]. */
void qw_op_split_heads2(qw_cmd c, qw_ref a, qw_ref b, qw_ref src,
                        int32_t rows, int32_t heads, int32_t dim);

/* ---- full attention -------------------------------------------------------
 *
 * The KV cache is fp16 and head-major: [kv_heads, max_ctx, head_dim].  Because
 * the layout is head-major it must be allocated at its final context size up
 * front, which is free in practice -- a shared buffer only commits the pages
 * that are actually written. */

/* Appends `rows` tokens of K and V at cache position base_pos, converting to
 * fp16.  k and v are fp32 [rows, kv_heads, head_dim]. */
void qw_op_kv_write(qw_cmd c, qw_ref kcache, qw_ref vcache, qw_ref k, qw_ref v,
                    int32_t rows, int32_t kv_heads, int32_t head_dim,
                    int32_t max_ctx, int32_t base_pos);

/* Causal attention of `rows` queries against the cache.  Row r attends to cache
 * positions [0, base_pos + r].  q and out are fp32 [rows, q_heads, head_dim]. */
void qw_op_attn_decode(qw_cmd c, qw_ref out, qw_ref q, qw_ref kcache, qw_ref vcache,
                       int32_t rows, int32_t q_heads, int32_t kv_heads,
                       int32_t head_dim, int32_t max_ctx, int32_t base_pos,
                       float scale);

/* ---- token selection ------------------------------------------------------
 *
 * Argmax and runner-up per row of a logits block, on the GPU.  This replaces a
 * host scan of the whole vocabulary that ran after a full sync, once per draft
 * token; see metal/select.metal for why it is exactly equal to that scan.
 *
 * The result is small enough to read straight off the shared buffer once the
 * command buffer the op was encoded into has completed. */

/* Threadgroups the partial pass splits each row across.  Only needed here to
 * size the scratch buffer. */
#define QW_SEL_TILES 32

/* One row's answer.  `index` is the argmax; `best - second` is the margin to
 * the runner-up, which is what says whether the model was nearly undecided. */
typedef struct { float best, second; uint32_t index, pad; } qw_cand;

/* `logits` is fp32 [rows, n]; `out` is [rows] qw_cand; `scratch` is
 * [rows * QW_SEL_TILES] qw_cand of workspace.
 *
 * `token_out`, when it names a buffer, also receives the winning ids as int32
 * [rows].  Pointing it at the token buffer a later dispatch reads is what lets
 * a chain of dependent steps go into one command buffer without the host in
 * between; pass an empty ref when nothing needs that. */
void qw_op_argmax_top2(qw_cmd c, qw_ref out, qw_ref scratch, qw_ref logits,
                       int32_t n, int32_t rows, qw_ref token_out);

#endif /* QWASAR_GPU_H */
