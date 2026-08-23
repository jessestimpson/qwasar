/* Argmax and runner-up over a logits row: the decode/draft selection step.
 *
 * Picking the next token used to be a host-side scan.  That was one full GPU
 * sync followed by the CPU walking 248,320 floats -- about a megabyte per row,
 * scalar, while the GPU sat idle.  Speculative decoding does it once per draft
 * token, so it was paid `depth` times a round on the critical path.
 *
 * The reduction runs in two stages because the draft path selects over a single
 * row.  One threadgroup per row would put a megabyte of reads on one core; the
 * partial pass spreads that row over QW_SEL_TILES threadgroups instead and a
 * second, trivial pass folds their candidates.
 *
 * EXACTNESS.  The host scan this replaces used strict `>`, so it kept the
 * LOWEST index among equal maxima, and its runner-up was the largest value
 * other than the winning occurrence.  qw_cand_merge reproduces both, and it is
 * associative and commutative, so every reduction order -- and therefore any
 * tile count or threadgroup size -- yields the token the serial scan would
 * have.  That is what lets this be a pure speedup rather than a change in
 * output, and tests/test_select.c checks it against the same serial scan. */

/* A running (best, runner-up) pair.  16 bytes so the host struct needs no
 * padding reasoning; `pad` is never read. */
struct qw_cand {
    float bv;    /* best value */
    float sv;    /* best value excluding one occurrence of bv */
    uint  bi;    /* index of the best value */
    uint  pad;
};

struct qw_sel_args {
    uint n;      /* vocabulary length */
    uint rows;
    uint tiles;
    uint write_token;   /* also publish the winning id as an int32 token */
    /* Compact-vocabulary mapping, for a head that scored only a prefix of the
     * rows plus a tail.  Row i below `prefix` is token i; at or above it, the
     * rows are the run starting at `tail_base`.  Zero for both is the identity,
     * so a full-vocabulary call needs no special case. */
    uint prefix;
    uint tail_base;
};

static inline uint qw_sel_token(uint bi, uint prefix, uint tail_base) {
    return bi < prefix ? bi : tail_base + (bi - prefix);
}

/* The identity: an empty range.  `sv <= bv` holds for every state built from
 * this one, which the equal-maxima branch below relies on. */
static inline qw_cand qw_cand_empty() {
    qw_cand c;
    c.bv = -INFINITY;
    c.sv = -INFINITY;
    c.bi = 0xFFFFFFFFu;
    c.pad = 0u;
    return c;
}

static inline void qw_cand_merge(thread qw_cand *a, qw_cand b) {
    if (b.bv > a->bv) {
        /* b wins; a's old best becomes a runner-up candidate. */
        a->sv = max(a->bv, b.sv);
        a->bv = b.bv;
        a->bi = b.bi;
    } else if (b.bv < a->bv) {
        a->sv = max(a->sv, b.bv);
    } else {
        /* Equal maxima.  The serial scan's strict `>` never replaced the
         * incumbent, so the lower index wins -- and the duplicate itself is
         * the runner-up, which is why a tie makes the margin zero. */
        a->bi = min(a->bi, b.bi);
        a->sv = a->bv;
    }
}

/* Folds one candidate per thread down to one per threadgroup, in `out[0]`. */
static inline qw_cand qw_cand_reduce(qw_cand c, uint sgid, uint nsg, uint lane,
                                     threadgroup qw_cand *partial) {
#pragma unroll
    for (uint off = 16; off > 0; off >>= 1) {
        qw_cand o;
        o.bv = simd_shuffle_down(c.bv, off);
        o.sv = simd_shuffle_down(c.sv, off);
        o.bi = simd_shuffle_down(c.bi, off);
        o.pad = 0u;
        qw_cand_merge(&c, o);
    }
    if (lane == 0) partial[sgid] = c;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0) {
        qw_cand v = (lane < nsg) ? partial[lane] : qw_cand_empty();
#pragma unroll
        for (uint off = 16; off > 0; off >>= 1) {
            qw_cand o;
            o.bv = simd_shuffle_down(v.bv, off);
            o.sv = simd_shuffle_down(v.sv, off);
            o.bi = simd_shuffle_down(v.bi, off);
            o.pad = 0u;
            qw_cand_merge(&v, o);
        }
        if (lane == 0) partial[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return partial[0];
}

/* Stage 1: grid (rows * tiles), flattened.  A 2-D grid would be the natural
 * shape, but Metal requires every attribute-declared input to be scalar or all
 * the same vector width, and the rest of these are scalars. */
kernel void qw_argmax_top2_partial(
    device const float    *logits  [[buffer(0)]],   /* [rows, n] */
    device       qw_cand  *partial [[buffer(1)]],   /* [rows, tiles] */
    constant qw_sel_args  &a       [[buffer(2)]],
    uint  tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_position_in_threadgroup]],
    uint  ntg  [[threads_per_threadgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  nsg  [[simdgroups_per_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    threadgroup qw_cand shared[32];

    const uint row  = tgid / a.tiles;
    const uint tile = tgid % a.tiles;

    /* Contiguous slices, so each threadgroup's reads coalesce within its own
     * span rather than interleaving with every other tile's. */
    const uint per   = (a.n + a.tiles - 1u) / a.tiles;
    const uint start = tile * per;
    const uint stop  = min(start + per, a.n);

    device const float *lr = logits + (ulong)row * a.n;

    qw_cand c = qw_cand_empty();
    for (uint i = start + tid; i < stop; i += ntg) {
        qw_cand e;
        e.bv = lr[i];
        e.sv = -INFINITY;
        e.bi = i;
        e.pad = 0u;
        qw_cand_merge(&c, e);
    }

    c = qw_cand_reduce(c, sgid, nsg, lane, shared);
    if (tid == 0) partial[(ulong)row * a.tiles + tile] = c;
}

/* Stage 2: grid (rows).  Folds one row's tiles into its final answer. */
kernel void qw_argmax_top2_final(
    device const qw_cand  *partial [[buffer(0)]],   /* [rows, tiles] */
    device       qw_cand  *out     [[buffer(1)]],   /* [rows] */
    device       int      *tokens  [[buffer(3)]],   /* [rows]; see write_token */
    constant qw_sel_args  &a       [[buffer(2)]],
    uint  tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_position_in_threadgroup]],
    uint  ntg  [[threads_per_threadgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  nsg  [[simdgroups_per_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    threadgroup qw_cand shared[32];

    device const qw_cand *pr = partial + (ulong)tgid * a.tiles;

    qw_cand c = qw_cand_empty();
    for (uint i = tid; i < a.tiles; i += ntg) qw_cand_merge(&c, pr[i]);

    c = qw_cand_reduce(c, sgid, nsg, lane, shared);
    if (tid == 0) {
        /* The host scan seeded its runner-up with -3e38 and never lowered it,
         * so restore that floor here.  Only reachable when a whole row is
         * -inf, which a matvec output is not; it keeps the two exactly equal
         * anyway rather than leaving a case to reason about later. */
        c.sv = max(c.sv, -3.0e38f);
        c.bi = qw_sel_token(c.bi, a.prefix, a.tail_base);
        out[tgid] = c;
        /* Publishing the id straight into the token buffer is what lets a
         * whole draft block be one command buffer: the next step's embedding
         * gather reads it here instead of the host reading it back and writing
         * it in again. */
        if (a.write_token) tokens[tgid] = int(c.bi);
    }
}
