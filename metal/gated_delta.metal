/* Gated DeltaNet: the recurrent half of the model, 48 of 64 layers.
 *
 * Per value head, the state is a [Dv, Dk] matrix carried across the whole
 * sequence.  For each timestep:
 *
 *     S      *= g                       decay, per head
 *     kv_mem  = S . k                   what the state already predicts
 *     delta   = (v - kv_mem) * beta     the correction to write
 *     S      += outer(delta, k)         delta rule
 *     y       = S . q                   read out
 *
 * The recurrence is serial in time and cannot be parallelised across it, so
 * parallelism comes from the 48 heads x 128 value rows, each owned by one
 * simdgroup.  A simdgroup's 32 lanes split the key dimension four ways, keeping
 * the entire [Dv, Dk] state resident in registers for the whole call -- the
 * state is only read from and written to device memory once per layer, not
 * once per timestep.
 *
 * The per-lane state array must be register-resident; a dynamic loop bound
 * would push it to scratch memory and cost far more than the arithmetic.  So
 * the key/value head dims are compile-time constants, checked against the
 * config at load.  qwasar is a single-model engine and this is where that
 * premise buys something real. */

#define QW_GDN_DK       128
#define QW_GDN_DV       128
#define QW_GDN_LANES    32
#define QW_GDN_PER_LANE (QW_GDN_DK / QW_GDN_LANES)   /* 4 */

struct qw_gdn_args {
    uint rows;   /* timesteps in this step: 1 decoding, N prefilling */
    uint hk;     /* key heads   */
    uint hv;     /* value heads */
    uint gqa;    /* hv / hk     */
    /* Speculative verify: the state as it stood after each of the first
     * `n_snap` timesteps.  A rejected draft has to be undone, and the delta
     * rule cannot be run backwards -- reversing S = g*S + outer(delta, k)
     * means dividing by a decayed g -- so the only way back is to have kept
     * the state on the way past.  The state is register-resident for the whole
     * call, so this is a store of what is already there. */
    uint n_snap;
    uint snap_stride;   /* floats between one timestep's snapshot and the next */
};

kernel void qw_gated_delta(
    device const float   *q     [[buffer(0)]],   /* [rows, hk, DK] l2-normalised */
    device const float   *k     [[buffer(1)]],   /* [rows, hk, DK] l2-normalised */
    device const float   *v     [[buffer(2)]],   /* [rows, hv, DV] */
    device const float   *g     [[buffer(3)]],   /* [rows, hv] decay */
    device const float   *beta  [[buffer(4)]],   /* [rows, hv] write strength */
    device       float   *state [[buffer(5)]],   /* [hv, DV, DK] fp32, in and out */
    device       float   *y     [[buffer(6)]],   /* [rows, hv, DV] */
    constant qw_gdn_args &a     [[buffer(7)]],
    device       float   *snap  [[buffer(8)]],   /* [n_snap, hv, DV, DK] fp32 */
    uint3 gid  [[thread_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]])
{
    const uint hv_idx = gid.z;
    const uint hk_idx = hv_idx / a.gqa;   /* grouped-query: 3 value heads per key head */
    const uint dv_idx = gid.y;

    const uint q_stride = a.hk * QW_GDN_DK;
    const uint v_stride = a.hv * QW_GDN_DV;

    device const float *qp = q    + hk_idx * QW_GDN_DK;
    device const float *kp = k    + hk_idx * QW_GDN_DK;
    device const float *vp = v    + hv_idx * QW_GDN_DV + dv_idx;
    device       float *yp = y    + hv_idx * QW_GDN_DV + dv_idx;
    device const float *gp = g    + hv_idx;
    device const float *bp = beta + hv_idx;

    /* This lane owns columns [PER_LANE*lane, PER_LANE*lane + PER_LANE) of row
     * dv_idx.  Consecutive lanes own consecutive columns, so both the state
     * load/store and every k/q read below coalesce. */
    device float *sp = state + ((ulong)hv_idx * QW_GDN_DV + dv_idx) * QW_GDN_DK
                             + QW_GDN_PER_LANE * lane;

    float st[QW_GDN_PER_LANE];
#pragma unroll
    for (uint i = 0; i < QW_GDN_PER_LANE; ++i) st[i] = sp[i];

    const uint col = QW_GDN_PER_LANE * lane;

    for (uint t = 0; t < a.rows; ++t) {
        const float gt = gp[0];
        const float bt = bp[0];

        float kv = 0.0f;
#pragma unroll
        for (uint i = 0; i < QW_GDN_PER_LANE; ++i) {
            st[i] *= gt;
            kv = fma(st[i], kp[col + i], kv);
        }
        kv = simd_sum(kv);

        /* Every lane derives the same delta; broadcasting it via simd_sum's
         * result is cheaper than a shuffle. */
        const float delta = (vp[0] - kv) * bt;

        float out = 0.0f;
#pragma unroll
        for (uint i = 0; i < QW_GDN_PER_LANE; ++i) {
            st[i] = fma(kp[col + i], delta, st[i]);
            out = fma(st[i], qp[col + i], out);
        }
        out = simd_sum(out);
        if (lane == 0) yp[0] = out;

        if (t < a.n_snap) {
            device float *dp = snap + (ulong)t * a.snap_stride
                             + ((ulong)hv_idx * QW_GDN_DV + dv_idx) * QW_GDN_DK
                             + QW_GDN_PER_LANE * lane;
#pragma unroll
            for (uint i = 0; i < QW_GDN_PER_LANE; ++i) dp[i] = st[i];
        }

        qp += q_stride; kp += q_stride;
        vp += v_stride; yp += v_stride;
        gp += a.hv;     bp += a.hv;
    }

#pragma unroll
    for (uint i = 0; i < QW_GDN_PER_LANE; ++i) sp[i] = st[i];
}

/* Decay and write-strength gates.
 *
 *   g    = exp(-exp(A_log) * softplus(a + dt_bias))
 *   beta = sigmoid(b)
 *
 * A_log and dt_bias are per value head; a and b come from their own tiny
 * projections.  Computed in fp32: g is a decay in (0,1) applied once per
 * timestep, so error here compounds along the whole sequence. */
struct qw_gdn_gate_args { uint rows; uint hv; };

kernel void qw_gdn_gates(
    device const float   *a_in    [[buffer(0)]],  /* [rows, hv] */
    device const float   *b_in    [[buffer(1)]],  /* [rows, hv] */
    device const ushort  *A_log   [[buffer(2)]],  /* [hv] bf16 */
    device const ushort  *dt_bias [[buffer(3)]],  /* [hv] bf16 */
    device       float   *g       [[buffer(4)]],  /* [rows, hv] */
    device       float   *beta    [[buffer(5)]],  /* [rows, hv] */
    constant qw_gdn_gate_args &p  [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.rows * p.hv) return;
    const uint h = gid % p.hv;
    const float A  = qw_bf16_to_f32(A_log[h]);
    const float dt = qw_bf16_to_f32(dt_bias[h]);
    g[gid]    = exp(-exp(A) * qw_softplus(a_in[gid] + dt));
    beta[gid] = qw_sigmoid(b_in[gid]);
}
