/* Causal depthwise convolution over the gated-delta input projection.
 *
 * Every one of the 10240 mixed-QKV channels has its own 4-tap filter and its
 * own 3-element history, so this is 10240 independent 1-D convolutions.  One
 * thread owns one channel: it pulls the history into registers, walks the
 * timesteps, and writes the new history back -- which makes the in-place
 * update of `state` safe without any barrier, since no two threads touch the
 * same channel.
 *
 * The kernel size is a compile-time constant for the same reason the
 * gated-delta state dims are: a dynamic bound would put the sliding window in
 * scratch memory. */

#define QW_CONV_K 4

struct qw_conv_args {
    uint channels;
    uint rows;
    /* Speculative verify: the state as it stood after each of the first
     * `n_snap` rows, so a rejected draft can be undone.  The window lives in
     * registers for the whole call, so a snapshot is a store of what is already
     * there rather than a re-read. */
    uint n_snap;
    uint snap_stride;   /* floats between one row's snapshot and the next */
};

kernel void qw_conv1d_causal_silu(
    device const float   *x     [[buffer(0)]],  /* [rows, channels] */
    device       float   *state [[buffer(1)]],  /* [K-1, channels] fp32, in and out */
    device const ushort  *w     [[buffer(2)]],  /* [channels, K] bf16, tap 0 oldest */
    device       float   *y     [[buffer(3)]],  /* [rows, channels] */
    constant qw_conv_args &a    [[buffer(4)]],
    device       float   *snap  [[buffer(5)]],  /* [n_snap, K-1, channels] */
    uint gid [[thread_position_in_grid]])
{
    if (gid >= a.channels) return;

    float taps[QW_CONV_K];
#pragma unroll
    for (uint j = 0; j < QW_CONV_K; ++j)
        taps[j] = qw_bf16_to_f32(w[gid * QW_CONV_K + j]);

    /* win[0] is the oldest sample, matching the state layout the cache keeps. */
    float win[QW_CONV_K];
#pragma unroll
    for (uint j = 0; j < QW_CONV_K - 1; ++j)
        win[j] = state[(ulong)j * a.channels + gid];

    for (uint t = 0; t < a.rows; ++t) {
        win[QW_CONV_K - 1] = x[(ulong)t * a.channels + gid];

        float acc = 0.0f;
#pragma unroll
        for (uint j = 0; j < QW_CONV_K; ++j) acc = fma(win[j], taps[j], acc);

        /* silu is always applied here in this model, so it is fused rather
         * than costing another full pass over 10240 channels. */
        y[(ulong)t * a.channels + gid] = qw_silu(acc);

#pragma unroll
        for (uint j = 0; j < QW_CONV_K - 1; ++j) win[j] = win[j + 1];

        if (t < a.n_snap) {
            device float *sp = snap + (ulong)t * a.snap_stride;
#pragma unroll
            for (uint j = 0; j < QW_CONV_K - 1; ++j)
                sp[(ulong)j * a.channels + gid] = win[j];
        }
    }

#pragma unroll
    for (uint j = 0; j < QW_CONV_K - 1; ++j)
        state[(ulong)j * a.channels + gid] = win[j];
}
