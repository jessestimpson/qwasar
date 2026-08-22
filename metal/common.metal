/* Shared helpers.  These files are concatenated into one source string that is
 * embedded in the binary (tools/bin2c.c) and compiled at startup, so they must
 * not #include each other and must not repeat definitions. */

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>

using namespace metal;

/* ---- MLX affine 4-bit dequantisation -------------------------------------
 *
 *   weight  U32  [out, in/8]    element i of a row is at bit 4*(i%8) of word i/8
 *   scales  BF16 [out, in/64]
 *   biases  BF16 [out, in/64]
 *
 *   w[o][i] = scales[o][i/64] * nibble(o, i) + biases[o][i/64]
 *
 * The nibble is unsigned in [0,15]; there is no separate zero point, the bias
 * absorbs it.  Scales are signed, so a "min" of a group can be either end. */

#define QW_QBITS       4
#define QW_QGROUP      64
#define QW_QPER_WORD   8    /* 32 / QW_QBITS */
#define QW_WORDS_PER_GROUP (QW_QGROUP / QW_QPER_WORD)   /* 8 */

/* Unpacks the 8 nibbles of one word into floats, low nibble first. */
inline void qw_unpack8(uint w, thread float *out) {
#pragma unroll
    for (int j = 0; j < 8; ++j) out[j] = float((w >> (4 * j)) & 0xF);
}

/* Dequantises one packed word into two float4s, holding nibbles 0,2,4,6 and
 * 1,3,5,7 respectively -- `even[k]` is element 2k and `odd[k]` is element 2k+1.
 *
 * This is the whole quantised inner loop's cost, so it is worth the trick.  The
 * obvious form shifts, masks, converts and scales each nibble separately: four
 * operations per weight, thirty-two per word.  Masking the low nibble of every
 * byte instead leaves a value `unpack_unorm4x8_to_float` can convert four at a
 * time, and the 1/255 it divides by is folded into the caller's scale rather
 * than undone -- undoing it would reintroduce a rounding question for the sake
 * of a multiply.  Seven operations per word, for the same arithmetic.
 *
 * `scale255` must be the group's scale times 255. */
inline void qw_unpack8_affine(uint w, float scale255, float bias,
                              thread float4 *even, thread float4 *odd) {
    *even = fma(scale255, unpack_unorm4x8_to_float(w & 0x0F0F0F0Fu), bias);
    *odd  = fma(scale255, unpack_unorm4x8_to_float((w >> 4) & 0x0F0F0F0Fu), bias);
}

/* Dot product of 8 dequantised weights against 8 activations. */
inline float qw_qdot8(uint w, float scale, float bias, const thread float *x) {
    float acc = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j)
        acc += fma(scale, float((w >> (4 * j)) & 0xF), bias) * x[j];
    return acc;
}

/* ---- bfloat16 -------------------------------------------------------------
 *
 * Metal has a native `bfloat` on recent targets, but the weights arrive as raw
 * ushort in an mmap and we want the conversion to be explicit and identical to
 * what the CPU reference does. */

inline float qw_bf16_to_f32(ushort v) {
    return as_type<float>(uint(v) << 16);
}

inline ushort qw_f32_to_bf16(float v) {
    /* round-to-nearest-even, matching numpy/torch */
    uint u = as_type<uint>(v);
    uint rounded = u + 0x7FFF + ((u >> 16) & 1);
    return ushort(rounded >> 16);
}

/* ---- activations ---------------------------------------------------------- */

inline float qw_silu(float x) { return x / (1.0f + exp(-x)); }
inline float qw_sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }
/* softplus(x) = log1p(exp(x)), guarded so large x does not overflow exp. */
inline float qw_softplus(float x) {
    return x > 20.0f ? x : log(1.0f + exp(x));
}

/* Shared by both quantised matrix kernels: qw_qmv_q4_g64 (decode, one row per
 * dispatch) and qw_qmm_q4_g64 (prefill, tiled over rows).  One definition so
 * the two cannot drift apart. */
struct qw_matmul_args {
    uint k;      /* input features  */
    uint n;      /* output features */
    uint rows;   /* tokens in this step */
};

/* Smoke kernel: proves the library compiled and the queue dispatches. */
kernel void qw_probe(device float *out       [[buffer(0)]],
                     constant uint &n        [[buffer(1)]],
                     uint gid                [[thread_position_in_grid]]) {
    if (gid < n) out[gid] = qw_silu(float(gid));
}
