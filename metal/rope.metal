/* Partial multimodal RoPE.
 *
 * Two things make this model's rotary embedding unusual:
 *
 *   - it is *partial*: head_dim is 256 but partial_rotary_factor is 0.25, so
 *     only the first 64 dims rotate and the remaining 192 pass through
 *     untouched;
 *   - it is *multimodal*: each of the 32 frequencies draws its position from
 *     one of three axes (time, height, width) chosen by `axis`, interleaved so
 *     the per-axis counts come out to mrope_section = [11, 11, 10].
 *
 * For text-only input all three axes carry the same position, so the selector
 * collapses to plain partial RoPE -- but the kernel is written for the general
 * case from the start, because the alternative is rewriting it when images
 * arrive, and images are exactly when the axes diverge.
 *
 * Rotation is half-split (rotate_half), not the traditional even/odd pairing:
 * dim i pairs with dim i + rotary_dim/2. */

struct qw_rope_args {
    uint rows;
    uint heads;
    uint head_dim;
    uint rotary_dim;   /* number of leading dims that rotate */
};

kernel void qw_rope_partial(
    device       float  *x        [[buffer(0)]],  /* [rows, heads, head_dim], in place */
    device const int    *pos      [[buffer(1)]],  /* [3, rows] -- t, h, w */
    device const uchar  *axis     [[buffer(2)]],  /* [rotary_dim/2] axis per frequency */
    device const float  *inv_freq [[buffer(3)]],  /* [rotary_dim/2] */
    constant qw_rope_args &a      [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint nfreq = a.rotary_dim / 2;   /* not `half`: that is a Metal type */
    const uint j    = gid.x;
    if (j >= nfreq) return;

    const uint rh  = gid.y;                 /* flattened (row, head) */
    const uint row = rh / a.heads;

    device float *xv = x + (ulong)rh * a.head_dim;

    const float angle = (float)pos[(ulong)axis[j] * a.rows + row] * inv_freq[j];
    const float c = cos(angle);
    const float s = sin(angle);

    const float x0 = xv[j];
    const float x1 = xv[j + nfreq];
    xv[j]        = x0 * c - x1 * s;
    xv[j + nfreq] = x1 * c + x0 * s;
}
