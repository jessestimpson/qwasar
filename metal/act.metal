/* Elementwise activations. */

struct qw_binary_args { uint n; };

/* SwiGLU: silu(gate) * up, evaluated in fp32 regardless of storage dtype. */
kernel void qw_swiglu(
    device const float    *gate [[buffer(0)]],
    device const float    *up   [[buffer(1)]],
    device       float    *y    [[buffer(2)]],
    constant qw_binary_args &a  [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= a.n) return;
    y[gid] = qw_silu(gate[gid]) * up[gid];
}

/* Residual add: y += x. */
kernel void qw_add_inplace(
    device       float    *y   [[buffer(0)]],
    device const float    *x   [[buffer(1)]],
    constant qw_binary_args &a [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= a.n) return;
    y[gid] += x[gid];
}

/* y = x * sigmoid(gate) -- the full-attention output gate. */
kernel void qw_mul_sigmoid(
    device       float    *y    [[buffer(0)]],
    device const float    *gate [[buffer(1)]],
    constant qw_binary_args &a  [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= a.n) return;
    y[gid] *= qw_sigmoid(gate[gid]);
}

struct qw_split_args { uint rows; uint heads; uint dim; };

/* Splits [rows, heads, 2*dim] into two contiguous [rows, heads, dim] buffers.
 *
 * q_proj emits query and output-gate interleaved per head, so everything
 * downstream would otherwise need a stride.  One cheap pass here keeps the
 * norm, rope, and attention kernels reading plain contiguous arrays. */
kernel void qw_split_heads2(
    device const float    *src [[buffer(0)]],  /* [rows, heads, 2*dim] */
    device       float    *a   [[buffer(1)]],  /* [rows, heads, dim] -- first half */
    device       float    *b   [[buffer(2)]],  /* [rows, heads, dim] -- second half */
    constant qw_split_args &p  [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint i  = gid.x;                 /* index within a head's half */
    const uint rh = gid.y;                 /* flattened (row, head) */
    if (i >= p.dim) return;

    const ulong s = (ulong)rh * 2 * p.dim + i;
    const ulong d = (ulong)rh * p.dim + i;
    a[d] = src[s];
    b[d] = src[s + p.dim];
}

struct qw_slice_args { uint rows; uint src_stride; uint offset; uint len; };

/* Extracts a column range out of each row: dst[r, 0:len] = src[r, off:off+len].
 *
 * The gated-delta input projection produces q, k and v concatenated along the
 * channel axis, so with more than one token in flight each of them is strided
 * rather than contiguous.  Everything downstream wants plain [rows, ...]
 * arrays, and one cheap pass here is far simpler than teaching four kernels
 * about strides. */
kernel void qw_slice_rows(
    device const float    *src [[buffer(0)]],
    device       float    *dst [[buffer(1)]],
    constant qw_slice_args &a  [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint i = gid.x, r = gid.y;
    if (i >= a.len || r >= a.rows) return;
    dst[(ulong)r * a.len + i] = src[(ulong)r * a.src_stride + a.offset + i];
}
