/* Token embedding lookup out of the quantised embedding table.
 *
 * The table is 248320 x 5120 in 4-bit, so it is never dequantised in bulk --
 * only the handful of rows a step actually needs.  One threadgroup per token. */

struct qw_embed_args {
    uint hidden;
    uint n_tokens;
};

kernel void qw_embed_q4(
    device const uint    *w      [[buffer(0)]],  /* [vocab, hidden/8] */
    device const ushort  *scales [[buffer(1)]],  /* [vocab, hidden/64] */
    device const ushort  *biases [[buffer(2)]],  /* [vocab, hidden/64] */
    device const int     *tokens [[buffer(3)]],  /* [n_tokens] */
    device       float   *y      [[buffer(4)]],  /* [n_tokens, hidden] */
    constant qw_embed_args &a    [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint ntg  [[threads_per_threadgroup]])
{
    const uint words  = a.hidden / QW_QPER_WORD;
    const uint groups = a.hidden / QW_QGROUP;
    const uint token  = (uint)tokens[tgid];

    device const uint   *wr = w      + (ulong)token * words;
    device const ushort *sr = scales + (ulong)token * groups;
    device const ushort *br = biases + (ulong)token * groups;
    device       float  *yr = y      + (ulong)tgid * a.hidden;

    /* One word (8 values) per thread per step. */
    for (uint wi = tid; wi < words; wi += ntg) {
        const uint  g  = wi / QW_WORDS_PER_GROUP;
        const uint  ww = wr[wi];
        const float sc = qw_bf16_to_f32(sr[g]);
        const float bi = qw_bf16_to_f32(br[g]);
#pragma unroll
        for (uint j = 0; j < QW_QPER_WORD; ++j)
            yr[wi * QW_QPER_WORD + j] = fma(sc, float((ww >> (4 * j)) & 0xF), bi);
    }
}
