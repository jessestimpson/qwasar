/* Images in, patches out.
 *
 * The vision tower never sees an image.  It sees a sequence of patch vectors,
 * and the two things that have to be right here are the order those patches
 * come in and the order of the values inside each one.  Both are determined by
 * how the weights were stored, neither is checkable by looking at the output,
 * and getting either wrong produces an embedding that is confidently wrong.
 *
 * ---- the order of patches --------------------------------------------------
 *
 * Merge-block order.  The reference processor reshapes to
 *
 *     (t, tp, c, gh/m, m, p, gw/m, m, p)
 *
 * and transposes to (t, gh/m, gw/m, m, m, c, tp, p, p), so the sequence runs
 * block row, block column, then the merge block's own row and column.  The four
 * patches of a 2x2 block end up adjacent, which is what lets the merger read
 * them as one contiguous 4608-wide row instead of gathering.  Position
 * embeddings are permuted the same way before being added.
 *
 * ---- the order inside a patch ----------------------------------------------
 *
 * The processor emits C, T, H, W; the model immediately moves the channel axis
 * to the end and convolves with a kernel stored [out, T, H, W, C].  This writes
 * T, H, W, C directly, which is the same thing with one fewer transposition.
 *
 * A still image fills the temporal axis by repeating itself, which is what
 * temporal_patch_size = 2 means for a single frame. */

#include "qwasar_model.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_ONLY_BMP
#define STBI_ONLY_GIF
#include "vendor/stb_image.h"

/* Pixel budget, from the model's own preprocessor config.  Expressed in pixels
 * rather than in either dimension, because what the tower costs is patches. */
#define QW_IMG_MIN_PIXELS  (65536)
#define QW_IMG_MAX_PIXELS  (16777216)

/* Rounds a dimension to a multiple of `factor`, at least one factor. */
static int32_t qw_round_to(double v, int32_t factor) {
    int32_t n = (int32_t)llround(v / (double)factor) * factor;
    return n < factor ? factor : n;
}

/* The reference's smart_resize: keep the aspect ratio, land on a multiple of
 * patch * merge in both axes, and stay inside the pixel budget. */
void qw_image_fit(int32_t w, int32_t h, int32_t factor,
                  int32_t *out_w, int32_t *out_h) {
    int32_t bw = qw_round_to(w, factor), bh = qw_round_to(h, factor);
    const double area = (double)bw * (double)bh;

    if (area > (double)QW_IMG_MAX_PIXELS) {
        const double beta = sqrt(((double)w * (double)h) / (double)QW_IMG_MAX_PIXELS);
        bw = (int32_t)floor(w / beta / factor) * factor;
        bh = (int32_t)floor(h / beta / factor) * factor;
        if (bw < factor) bw = factor;
        if (bh < factor) bh = factor;
    } else if (area < (double)QW_IMG_MIN_PIXELS) {
        const double beta = sqrt((double)QW_IMG_MIN_PIXELS / ((double)w * (double)h));
        bw = (int32_t)ceil(w * beta / factor) * factor;
        bh = (int32_t)ceil(h * beta / factor) * factor;
    }
    *out_w = bw;
    *out_h = bh;
}

/* Separable resample: box-average when shrinking, bilinear when growing.
 *
 * Not the reference's resampler, which is torchvision's antialiased bicubic,
 * and the difference shows up in the pixels rather than in anything structural.
 * Matching it exactly is a filter-design exercise with no bearing on whether
 * the tower is right, so the tower is validated on pre-sized input instead and
 * this is left as a good resize rather than an identical one. */
static void qw_resample(const unsigned char *src, int32_t sw, int32_t sh,
                        float *dst, int32_t dw, int32_t dh) {
    const double xr = (double)sw / (double)dw, yr = (double)sh / (double)dh;
    for (int32_t y = 0; y < dh; y++) {
        const double y0 = y * yr, y1 = (y + 1) * yr;
        int32_t iy0 = (int32_t)y0, iy1 = (int32_t)ceil(y1);
        if (iy1 > sh) iy1 = sh;
        if (iy1 <= iy0) iy1 = iy0 + 1;
        for (int32_t x = 0; x < dw; x++) {
            const double x0 = x * xr, x1 = (x + 1) * xr;
            int32_t ix0 = (int32_t)x0, ix1 = (int32_t)ceil(x1);
            if (ix1 > sw) ix1 = sw;
            if (ix1 <= ix0) ix1 = ix0 + 1;

            double acc[3] = { 0, 0, 0 };
            double wsum = 0.0;
            for (int32_t sy = iy0; sy < iy1; sy++) {
                const double wy = fmin((double)sy + 1.0, y1) - fmax((double)sy, y0);
                if (wy <= 0.0) continue;
                for (int32_t sx = ix0; sx < ix1; sx++) {
                    const double wx = fmin((double)sx + 1.0, x1) - fmax((double)sx, x0);
                    if (wx <= 0.0) continue;
                    const unsigned char *p = src + ((size_t)sy * sw + sx) * 3;
                    const double wgt = wy * wx;
                    acc[0] += wgt * p[0]; acc[1] += wgt * p[1]; acc[2] += wgt * p[2];
                    wsum += wgt;
                }
            }
            float *o = dst + ((size_t)y * dw + x) * 3;
            for (int ch = 0; ch < 3; ch++)
                /* The model's mean and standard deviation are both 0.5, so
                 * normalisation is exactly this. */
                o[ch] = (float)(acc[ch] / wsum) / 127.5f - 1.0f;
        }
    }
}

/* Lays the normalised pixels out as patches, in the two orders above. */
static void qw_patchify(const float *px, int32_t w, const qw_config *c,
                        int32_t grid_h, int32_t grid_w, float *out) {
    const int32_t P = c->vis_patch_size, T = c->vis_temporal_patch_size;
    const int32_t M = c->vis_spatial_merge_size, C = c->vis_in_channels;
    const int32_t elems = T * P * P * C;
    int32_t n = 0;

    for (int32_t by = 0; by < grid_h / M; by++)
    for (int32_t bx = 0; bx < grid_w / M; bx++)
    for (int32_t my = 0; my < M; my++)
    for (int32_t mx = 0; mx < M; mx++) {
        const int32_t py = by * M + my, pxx = bx * M + mx;
        float *dst = out + (size_t)n++ * elems;
        for (int32_t t = 0; t < T; t++)
        for (int32_t dy = 0; dy < P; dy++)
        for (int32_t dx = 0; dx < P; dx++) {
            /* A still image repeats itself along the temporal axis. */
            const float *s = px + (((size_t)(py * P + dy) * w) + (pxx * P + dx)) * 3;
            for (int32_t ch = 0; ch < C; ch++) *dst++ = s[ch];
        }
    }
}

/* Shared by the path and memory entry points; takes ownership of `rgb`. */
static bool qw_image_finish(qw_image *im, unsigned char *rgb, int w, int h,
                            const qw_config *c, char *err, size_t errcap) {

    const int32_t factor = c->vis_patch_size * c->vis_spatial_merge_size;
    int32_t rw = 0, rh = 0;
    qw_image_fit(w, h, factor, &rw, &rh);

    float *px = malloc((size_t)rw * rh * 3 * sizeof *px);
    if (!px) { stbi_image_free(rgb); snprintf(err, errcap, "out of memory"); return false; }
    qw_resample(rgb, w, h, px, rw, rh);
    stbi_image_free(rgb);

    im->grid_t = 1;
    im->grid_h = rh / c->vis_patch_size;
    im->grid_w = rw / c->vis_patch_size;
    im->n_patches = im->grid_t * im->grid_h * im->grid_w;
    im->patch_elems = c->vis_temporal_patch_size * c->vis_patch_size
                    * c->vis_patch_size * c->vis_in_channels;
    im->src_w = w;
    im->src_h = h;

    im->patches = malloc((size_t)im->n_patches * im->patch_elems * sizeof *im->patches);
    if (!im->patches) { free(px); snprintf(err, errcap, "out of memory"); return false; }
    qw_patchify(px, rw, c, im->grid_h, im->grid_w, im->patches);
    free(px);
    return true;
}

bool qw_image_load(qw_image *im, const char *path, const qw_config *c,
                   char *err, size_t errcap) {
    memset(im, 0, sizeof *im);
    int w = 0, h = 0, comp = 0;
    unsigned char *rgb = stbi_load(path, &w, &h, &comp, 3);
    if (!rgb) {
        snprintf(err, errcap, "cannot read %s: %s", path, stbi_failure_reason());
        return false;
    }
    return qw_image_finish(im, rgb, w, h, c, err, errcap);
}

/* Images arriving over HTTP are base64 in a JSON body, so they never touch the
 * filesystem. */
bool qw_image_load_memory(qw_image *im, const void *bytes, size_t len,
                          const qw_config *c, char *err, size_t errcap) {
    memset(im, 0, sizeof *im);
    int w = 0, h = 0, comp = 0;
    unsigned char *rgb = stbi_load_from_memory(bytes, (int)len, &w, &h, &comp, 3);
    if (!rgb) {
        snprintf(err, errcap, "cannot decode image: %s", stbi_failure_reason());
        return false;
    }
    return qw_image_finish(im, rgb, w, h, c, err, errcap);
}

void qw_image_free(qw_image *im) {
    if (!im) return;
    free(im->patches);
    memset(im, 0, sizeof *im);
}
