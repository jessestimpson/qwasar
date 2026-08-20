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
 * qw_op_qmat_q4 is what the graph calls; it picks between the two
 * implementations by shape.  The matvec reads every weight once per token,
 * which is optimal at rows = 1 and quadratically wasteful beyond it; the matmul
 * tiles over tokens so a weight block is read and dequantised once for the
 * whole tile, but pads its token tile to 64 and so wastes work on small counts.
 * The others are exposed for tests and benchmarks. */
void qw_op_qmat_q4(qw_cmd c, qw_ref y, qw_ref x,
                   qw_ref w, qw_ref scales, qw_ref biases,
                   int32_t k, int32_t n, int32_t rows);

void qw_op_qmv_q4(qw_cmd c, qw_ref y, qw_ref x,
                  qw_ref w, qw_ref scales, qw_ref biases,
                  int32_t k, int32_t n, int32_t rows);

void qw_op_qmm_q4(qw_cmd c, qw_ref y, qw_ref x,
                  qw_ref w, qw_ref scales, qw_ref biases,
                  int32_t k, int32_t n, int32_t rows);

/* Token count at or above which qw_op_qmat_q4 switches to the tiled matmul.
 *
 * Measured: the matmul pads its token tile to QW_QMM_BM and so costs the same
 * for 1 token as for 64, roughly 2.6 s through the whole model, while the
 * matvec costs about 0.18 s per token.  They cross a little under 15. */
#define QW_QMM_MIN_ROWS 16

/* Tiling for qw_qmm_q4_g64.  These are the single source of truth: the host
 * derives its dispatch geometry from them and injects them into the Metal
 * compile as preprocessor macros, so the kernel and the launch cannot disagree.
 *
 *   threads per threadgroup = (BM/TM) * (BN/TN)
 *   threadgroup memory      = (BM + BN) * BK * 4 bytes, must stay under 32 KB
 *   inner-loop ALU:LDS      = TM*TN fused multiply-adds per TM+TN loads
 *
 * BK must divide 32 so a packed word never spans two quantisation groups. */
#define QW_QMM_BM 64
#define QW_QMM_BN 64
#define QW_QMM_BK 32
#define QW_QMM_TM 4
#define QW_QMM_TN 4
#define QW_QMM_THREADS ((QW_QMM_BM / QW_QMM_TM) * (QW_QMM_BN / QW_QMM_TN))

/* RMS norm over the last axis of an [rows, dim] fp32 buffer.
 *
 * `weight` may be a null ref, which is how the gated-delta path expresses an
 * unweighted normalisation.  `out_scale` is applied after normalising and
 * before the weight; it turns an RMS norm into the scaled L2 norm that path
 * wants (see metal/norm.metal). */
void qw_op_rms_norm(qw_cmd c, qw_ref y, qw_ref x, qw_ref weight,
                    int32_t dim, int32_t rows, float eps, float out_scale);

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
void qw_op_conv1d_causal_silu(qw_cmd c, qw_ref y, qw_ref x, qw_ref state,
                              qw_ref weight, int32_t channels, int32_t rows,
                              int32_t ksize);

/* g = exp(-exp(A_log) * softplus(a + dt_bias)),  beta = sigmoid(b). */
void qw_op_gdn_gates(qw_cmd c, qw_ref g, qw_ref beta, qw_ref a, qw_ref b,
                     qw_ref A_log, qw_ref dt_bias, int32_t n_v_heads, int32_t rows);

/* The delta-rule recurrence.  `state` is [hv, dv, dk] fp32, in and out.
 * q and k must already be l2-normalised. */
void qw_op_gated_delta(qw_cmd c, qw_ref y, qw_ref q, qw_ref k, qw_ref v,
                       qw_ref g, qw_ref beta, qw_ref state,
                       int32_t hk, int32_t hv, int32_t dk, int32_t dv, int32_t rows);

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

#endif /* QWASAR_GPU_H */
