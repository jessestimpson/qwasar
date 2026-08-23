#ifndef QWASAR_MODEL_H
#define QWASAR_MODEL_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#include "qwasar_gpu.h"

/* Parsed architecture and the resolved weight table.
 *
 * Internal to the engine (qwasar.c and the tests); not part of qwasar.h. */

#define QW_MAX_LAYERS      128
#define QW_MAX_VIS_BLOCKS   64
#define QW_MAX_SHARDS       16
#define QW_MAX_DIMS          6

/* ---- the draft head's vocabulary -------------------------------------------
 *
 * The MTP head's lm_head is the largest single read in a draft step: 715 MB of
 * the ~950 MB one costs, three quarters of it, and drafting is otherwise
 * already at about 80% of bandwidth (PLAN.md section 5).  The draft scores only
 * part of the vocabulary, which cuts that read proportionally.
 *
 * This cannot change what the model emits.  The draft only PROPOSES; every
 * proposal is checked against the full head by the verify, so a token the draft
 * can no longer reach is not mis-emitted -- it is rejected, and the round still
 * advances by the target's own answer.  The exactness surface is untouched, and
 * the only thing at risk is the acceptance rate.
 *
 * Two runs of rows are kept.  The prefix is a BPE-rank cut, which works because
 * a byte-level BPE vocabulary is ordered by merge priority and therefore
 * roughly by frequency.  The tail is every added special token -- 276 rows, too
 * cheap to be worth trimming, and losing the ability to propose an end-of-turn
 * marker would cost a rejection at every turn boundary.
 *
 * QW_DRAFT_TAIL_LO is the first added special token id.  The graph cannot reach
 * the tokenizer, so it is a constant; tests/test_select checks it against the
 * real one rather than trusting it.
 *
 * The prefix is overridable at build time (-DQW_DRAFT_PREFIX=n) because the
 * right cut is an empirical question and the sweep has to be repeatable.  A
 * value at or above the vocabulary disables the trim and scores everything,
 * which is how the baseline is built. */
#ifndef QW_DRAFT_PREFIX
#define QW_DRAFT_PREFIX   98304
#endif
#define QW_DRAFT_TAIL_LO  248044


typedef enum {
    QW_DT_UNKNOWN = 0,
    QW_DT_F32,
    QW_DT_F16,
    QW_DT_BF16,
    QW_DT_U32,   /* packed 4-bit quantised weights */
    QW_DT_I32,
    QW_DT_U8,
} qw_dtype;

const char *qw_dtype_name(qw_dtype d);
size_t      qw_dtype_size(qw_dtype d);

/* One tensor, addressed as an offset into the buffer wrapping its shard. */
typedef struct {
    const char *name;      /* points into the owned name arena */
    qw_buf      buf;
    size_t      offset;    /* byte offset within buf */
    size_t      nbytes;
    qw_dtype    dtype;
    int         ndim;
    int64_t     shape[QW_MAX_DIMS];
} qw_tensor;

/* A quantised linear: MLX affine, 4 bits, group_size 64.
 *   weight  U32  [out, in/8]   -- element i of a row lives at bit 4*(i%8) of word i/8
 *   scales  BF16 [out, in/64]
 *   biases  BF16 [out, in/64]
 * Dequantisation is  w = scale * nibble + bias,  with the nibble unsigned in
 * [0,15] and the zero-point folded into the bias. */
typedef struct {
    const qw_tensor *weight;
    const qw_tensor *scales;
    const qw_tensor *biases;
    int32_t in_features;
    int32_t out_features;
} qw_qlinear;

/* An unquantised projection.  Only the MTP draft head has these: the base
 * model is 4-bit throughout, but the head ships from its own repository in
 * bf16 and there is no reason to quantise it before its cost has been
 * measured (PLAN.md section 5, milestone 3). */
typedef struct {
    const qw_tensor *weight;    /* BF16 [out_features, in_features] */
    const qw_tensor *bias;      /* BF16 [out_features], or NULL */
    int32_t in_features;
    int32_t out_features;
} qw_dense;

/* LayerNorm, which scales and shifts.  The text model has none of these -- it
 * is RMSNorm throughout -- and the vision tower has nothing else. */
typedef struct {
    const qw_tensor *weight;
    const qw_tensor *bias;
} qw_lnorm;

typedef struct {
    bool is_linear_attn;   /* gated delta if true, full attention otherwise */

    const qw_tensor *input_layernorm;
    const qw_tensor *post_attention_layernorm;

    /* full attention */
    qw_qlinear q_proj, k_proj, v_proj, o_proj;
    const qw_tensor *q_norm, *k_norm;

    /* gated delta */
    qw_qlinear in_proj_qkv, in_proj_z, in_proj_b, in_proj_a, out_proj;
    const qw_tensor *conv1d;    /* BF16 [conv_dim, K, 1], tap 0 is oldest */
    const qw_tensor *A_log;     /* BF16 [num_v_heads] */
    const qw_tensor *dt_bias;   /* BF16 [num_v_heads] */
    const qw_tensor *gdn_norm;  /* BF16 [head_v_dim] -- note: NOT +1 offset */

    /* mlp */
    qw_qlinear gate_proj, up_proj, down_proj;
} qw_layer;

/* The multi-token-prediction draft head.
 *
 * One full-attention layer of exactly the shape the base model runs sixteen
 * of, fused onto the backbone by a concatenation and a single projection:
 *
 *     fused = fc([ norm_e(embed(next_token)) ; norm_h(hidden) ])
 *     out   = norm(layer(fused))
 *
 * and then the BASE model's lm_head reads `out`.  The head carries no
 * embedding table and no output projection of its own -- the head config's
 * `mtp_use_dedicated_embeddings: false` says so, and neither tensor is in its
 * repository.
 *
 * The concatenation order is embedding first, hidden second.  That is the
 * opposite of the DeepSeek layout it otherwise resembles, and getting it
 * backwards would produce a head that loads, runs, and drafts nonsense. */
typedef struct {
    bool present;

    const qw_tensor *pre_fc_norm_hidden;
    const qw_tensor *pre_fc_norm_embedding;
    qw_dense fc;                       /* [hidden, 2 * hidden] */

    const qw_tensor *input_layernorm;
    const qw_tensor *post_attention_layernorm;
    qw_dense q_proj, k_proj, v_proj, o_proj;
    const qw_tensor *q_norm, *k_norm;
    qw_dense gate_proj, up_proj, down_proj;

    /* The same eight matrices in 4-bit, which is what actually runs.  The bf16
     * originals above are only the quantiser's input. */
    bool     quantised;
    qw_buf   q4;
    qw_tensor  q4t[24];                /* three per matrix, backing the qlinears */
    qw_qlinear q_fc, q_q, q_k, q_v, q_o, q_gate, q_up, q_down;

    const qw_tensor *norm;             /* final norm, before the base lm_head */

    /* Draft tokens the head was trained to produce in one round; the head's own
     * config.json carries it.  Verify width is one more than this. */
    int32_t block_size;
} qw_mtp;

/* One block of the vision tower: pre-norm attention, pre-norm MLP, both
 * residual.  Attention is bidirectional over the image's patches, so there is
 * no cache and no mask -- the closest thing in the text model is prefill, and
 * even that is causal. */
typedef struct {
    qw_lnorm  norm1, norm2;
    qw_dense  qkv;              /* [3 * hidden, hidden], q|k|v concatenated */
    qw_dense  proj;
    qw_dense  fc1, fc2;         /* tanh-GELU between them */
} qw_vis_block;

typedef struct {
    bool present;

    /* A patch is temporal_patch_size x patch x patch x channels, flattened in
     * that order because that is how the conv weight was stored. */
    qw_dense  patch_embed;
    const qw_tensor *pos_embed;   /* BF16 [grid_side^2, hidden], interpolated */
    int32_t   grid_side;

    qw_vis_block blocks[QW_MAX_VIS_BLOCKS];

    /* The merger takes a 2x2 block of patches -- contiguous, because the token
     * order is merge-block order from the patch embedding on -- and projects
     * the concatenation into the text model's width. */
    qw_lnorm  merger_norm;
    qw_dense  merger_fc1, merger_fc2;
} qw_vision;

typedef struct {
    /* text */
    int32_t hidden_size;
    int32_t intermediate_size;
    int32_t num_hidden_layers;
    int32_t num_attention_heads;
    int32_t num_key_value_heads;
    int32_t head_dim;
    int32_t vocab_size;
    int32_t full_attention_interval;
    float   rms_norm_eps;
    bool    tie_word_embeddings;
    bool    attn_output_gate;

    /* gated delta */
    int32_t linear_num_value_heads;
    int32_t linear_num_key_heads;
    int32_t linear_key_head_dim;
    int32_t linear_value_head_dim;
    int32_t linear_conv_kernel_dim;

    /* rope */
    float   rope_theta;
    float   partial_rotary_factor;
    int32_t rotary_dim;          /* head_dim * partial_rotary_factor */
    int32_t mrope_section[3];
    bool    mrope_interleaved;
    int32_t max_position_embeddings;

    /* quantisation */
    int32_t quant_bits;
    int32_t quant_group_size;

    /* tokens */
    int32_t bos_token_id;
    int32_t eos_token_ids[8];
    int32_t n_eos;
    int32_t image_token_id, video_token_id;
    int32_t vision_start_token_id, vision_end_token_id;

    /* vision */
    bool    has_vision;
    int32_t vis_depth;
    int32_t vis_hidden_size;
    int32_t vis_intermediate_size;
    int32_t vis_num_heads;
    int32_t vis_patch_size;
    int32_t vis_temporal_patch_size;
    int32_t vis_spatial_merge_size;
    int32_t vis_in_channels;
    int32_t vis_out_hidden_size;
    int32_t vis_num_position_embeddings;
} qw_config;

/* Derived sizes used all over the graph; computed once by qw_config_derive(). */
typedef struct {
    int32_t n_full_attn_layers;
    int32_t n_linear_attn_layers;
    int32_t q_dim;          /* num_attention_heads * head_dim (query half only) */
    int32_t q_proj_out;     /* doubled when attn_output_gate */
    int32_t kv_dim;         /* num_key_value_heads * head_dim */
    int32_t gqa_factor;
    int32_t key_dim;        /* linear_num_key_heads   * linear_key_head_dim */
    int32_t value_dim;      /* linear_num_value_heads * linear_value_head_dim */
    int32_t conv_dim;       /* 2*key_dim + value_dim */
    int32_t gdn_gqa_factor;
} qw_shape;

void qw_config_derive(const qw_config *c, qw_shape *s);
static inline bool qw_layer_is_linear(const qw_config *c, int i) {
    return ((i + 1) % c->full_attention_interval) != 0;
}

/* ---- engine internals -----------------------------------------------------
 *
 * Exposed for the graph code and the tests, not for the CLI/agent -- those see
 * only qwasar.h. */

#include "qwasar.h"   /* for qwasar_engine */

const qw_config *qwasar_engine_config(const qwasar_engine *e);
const qw_shape  *qwasar_engine_shape (const qwasar_engine *e);
const qw_layer  *qwasar_engine_layer (const qwasar_engine *e, int index);
const qw_tensor *qwasar_engine_tensor(const qwasar_engine *e, const char *name);
const qw_qlinear *qwasar_engine_embed(const qwasar_engine *e);
const qw_qlinear *qwasar_engine_head (const qwasar_engine *e);
const qw_tensor  *qwasar_engine_final_norm(const qwasar_engine *e);

/* The MTP draft head; `present` is false unless --mtp named a head directory. */
const qw_mtp     *qwasar_engine_mtp(const qwasar_engine *e);

/* An image, already resized, normalised and cut into the patch sequence the
 * vision tower reads.  `patches` is [n_patches, patch_elems] fp32 in
 * merge-block order; see the comment at the top of qwasar_image.c. */
typedef struct {
    float  *patches;
    int32_t n_patches;
    int32_t patch_elems;
    int32_t grid_t, grid_h, grid_w;   /* in patches */
    int32_t src_w, src_h;             /* before resizing, for reporting */
} qw_image;

/* Frames decoded from a video file, all at the same size, tightly packed RGB8.
 * Sampling policy is the model's own: see qwasar_video.m. */
typedef struct {
    unsigned char **frames;
    int32_t         n_frames;
    int             width, height;
    double          seconds;
} qw_video;

bool qw_video_load(qw_video *v, const char *path, double fps,
                   int32_t min_frames, int32_t max_frames,
                   char *err, size_t errcap);
void qw_video_free(qw_video *v);

bool qw_image_load(qw_image *im, const char *path, const qw_config *c,
                   char *err, size_t errcap);
bool qw_image_load_memory(qw_image *im, const void *bytes, size_t len,
                          const qw_config *c, char *err, size_t errcap);
void qw_image_free(qw_image *im);
/* The video path: frames already decoded, all the same size. */
bool qw_image_from_frames(qw_image *im, unsigned char *const *frames,
                          int32_t n_frames, int32_t w, int32_t h,
                          const qw_config *c, char *err, size_t errcap);
void qw_video_fit(int32_t n_frames, int32_t w, int32_t h, int32_t factor,
                  int32_t *out_w, int32_t *out_h);

/* Runs the vision tower.  Returns [rows, out_hidden_size] fp32, caller frees;
 * one row per merged 2x2 block of patches. */
float *qwasar_encode_image(qwasar_engine *e, const qw_image *im,
                           int32_t *out_rows, char *err, size_t errcap);
void qw_verrf(char *err, size_t cap, const char *fmt, ...);
/* Exposed for tests: the resized dimensions this image would be given. */
void qw_image_fit(int32_t w, int32_t h, int32_t factor, int32_t *out_w, int32_t *out_h);

/* MRoPE position triples for a prompt: [3, n], axis-major.  Text advances all
 * three axes together; an image gives its tokens a frame, a row and a column,
 * and the text after it resumes from one past the largest of the three.
 * Exposed because it is pure index arithmetic with an exactly known answer. */
bool qw_mrope_positions(const qw_config *c, const int32_t *tokens, int32_t n,
                        const struct qwasar_image_input *im, int32_t n_images,
                        int32_t start, int32_t *out, int32_t *out_next,
                        char *err, size_t errcap);

/* The vision tower; `present` is false for a text-only checkpoint. */
const qw_vision  *qwasar_engine_vision(const qwasar_engine *e);
int32_t qwasar_engine_context_size (const qwasar_engine *e);
int32_t qwasar_engine_prefill_chunk(const qwasar_engine *e);

/* Host pointer to a tensor's bytes.  Valid for mapped and materialized shards
 * alike, since both live in shared (CPU-visible) device memory. */
const void *qw_tensor_data(const qw_tensor *t);
/* A GPU argument referring to the whole tensor. */
qw_ref      qw_tensor_ref(const qw_tensor *t);

/* ---- session state, for the disk cache -------------------------------------
 *
 * The layout lives here rather than in the kvstore so the file format never
 * needs to know how the graph arranges its buffers.  Packed state is
 * independent of the session's context size: the KV cache is compacted to the
 * tokens actually used, so a checkpoint written at one context length restores
 * into another. */

size_t  qw_session_state_bytes(const qwasar_session *s, int32_t n_tokens);
bool    qw_session_pack  (const qwasar_session *s, void *dst, size_t cap);
bool    qw_session_unpack(qwasar_session *s, const void *src, size_t len,
                          const int32_t *tokens, int32_t n_tokens);

/* The tokens this session has evaluated, in order. */
const int32_t *qw_session_history(const qwasar_session *s, int32_t *n);

/* ---- CPU reference ops ----------------------------------------------------
 *
 * Scalar fp32 twins of the Metal kernels, used by tests to prove the GPU path.
 * Correctness only -- these are not on any inference path. */

float qw_bf16_to_f32_c(uint16_t v);

void qw_cpu_qmv_q4(float *y, const float *x, const uint32_t *w,
                   const uint16_t *scales, const uint16_t *biases,
                   int32_t k, int32_t n, int32_t rows);

void qw_cpu_rms_norm(float *y, const float *x, const uint16_t *w,
                     int32_t dim, int32_t rows, float eps, float out_scale);
void qw_cpu_rms_norm_concat(float *y, const float *e, const uint16_t *we,
                            const float *h, const uint16_t *wh,
                            int32_t dim, int32_t rows, float eps);
void qw_cpu_dmv_bf16(float *y, const float *x, const uint16_t *w,
                     int32_t k, int32_t n, int32_t rows);
void qw_cpu_rms_norm_gated(float *y, const float *x, const uint16_t *w,
                           const float *gate, int32_t dim, int32_t rows,
                           float eps, float out_scale);
void qw_cpu_swiglu(float *y, const float *gate, const float *up, int32_t n);
void qw_cpu_layer_norm(float *y, const float *x, const uint16_t *w,
                       const uint16_t *b, int32_t dim, int32_t rows, float eps);
void qw_cpu_gelu_tanh(float *y, int32_t n);
void qw_cpu_rope_2d(float *x, const float *angles,
                    int32_t tokens, int32_t heads, int32_t dim, int32_t stride);
void qw_cpu_vision_attn(float *out, const float *qkv, int32_t tokens,
                        int32_t heads, int32_t dim, int32_t segment, float scale);

void qw_cpu_conv1d_causal_silu(float *y, const float *x, float *state,
                               const uint16_t *w, int32_t channels, int32_t rows,
                               int32_t ksize);
void qw_cpu_gdn_gates(float *g, float *beta, const float *a, const float *b,
                      const uint16_t *A_log, const uint16_t *dt_bias,
                      int32_t hv, int32_t rows);
void qw_cpu_gated_delta(float *y, const float *q, const float *k, const float *v,
                        const float *g, const float *beta, float *state,
                        int32_t hk, int32_t hv, int32_t dk, int32_t dv, int32_t rows);

void qw_cpu_rope_partial(float *x, const int32_t *pos, const uint8_t *axis,
                         const float *inv_freq, int32_t rows, int32_t heads,
                         int32_t head_dim, int32_t rotary_dim);
void qw_cpu_embed_q4(float *y, const int32_t *tokens, const uint32_t *w,
                     const uint16_t *scales, const uint16_t *biases,
                     int32_t hidden, int32_t n_tokens);

float    qw_f16_to_f32_c(uint16_t v);
uint16_t qw_f32_to_f16_c(float v);

void qw_cpu_kv_write(uint16_t *kc, uint16_t *vc, const float *k, const float *v,
                     int32_t rows, int32_t kv_heads, int32_t head_dim,
                     int32_t max_ctx, int32_t base_pos);
void qw_cpu_attn_decode(float *out, const float *q, const uint16_t *kc,
                        const uint16_t *vc, int32_t rows, int32_t q_heads,
                        int32_t kv_heads, int32_t head_dim, int32_t max_ctx,
                        int32_t base_pos, float scale);

/* Dequantises one full row of a quantised linear into `out` (length k). */
void qw_cpu_dequant_row(float *out, const uint32_t *w, const uint16_t *scales,
                        const uint16_t *biases, int32_t k, int32_t row);

#endif /* QWASAR_MODEL_H */
