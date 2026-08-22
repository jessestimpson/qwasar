/* The vision tower: patches in, text-model-width embeddings out.
 *
 * Runs once per image, before any text is evaluated, and produces one row per
 * merged 2x2 block of patches.  Those rows are then scattered into the token
 * embeddings wherever the prompt has an <|image_pad|>.
 *
 * It is a separate file because it is a separate model: 27 blocks of bf16
 * weights with biases, LayerNorm, tanh-GELU and bidirectional attention, none
 * of which the text side uses.  The only thing the two share is the buffer and
 * command plumbing in qwasar_gpu.h. */

#include "qwasar.h"
#include "qwasar_gpu.h"
#include "qwasar_model.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Angles for the 2D rotary embedding: for each token, its row's frequencies
 * followed by its column's.
 *
 * The token order is merge-block order (see qwasar_image.c), so a token's
 * position in the patch grid has to be reconstructed from its index rather than
 * read off it -- which is the whole reason this is computed here rather than
 * being a running counter. */
static void qw_vis_angles(float *out, const qw_config *c,
                          int32_t grid_h, int32_t grid_w, int32_t n_tokens) {
    const int32_t head_dim = c->vis_hidden_size / c->vis_num_heads;
    const int32_t nf = head_dim / 4;          /* frequencies per axis */
    const int32_t M  = c->vis_spatial_merge_size;

    float *inv = malloc((size_t)nf * sizeof *inv);
    for (int32_t i = 0; i < nf; i++)
        inv[i] = 1.0f / powf(10000.0f, (float)(2 * i) / (float)(head_dim / 2));

    int32_t n = 0;
    for (int32_t by = 0; by < grid_h / M && n < n_tokens; by++)
    for (int32_t bx = 0; bx < grid_w / M && n < n_tokens; bx++)
    for (int32_t my = 0; my < M; my++)
    for (int32_t mx = 0; mx < M; mx++) {
        const float row = (float)(by * M + my), col = (float)(bx * M + mx);
        float *a = out + (size_t)n++ * (head_dim / 2);
        for (int32_t i = 0; i < nf; i++)      a[i]      = row * inv[i];
        for (int32_t i = 0; i < nf; i++)      a[nf + i] = col * inv[i];
    }
    free(inv);
}

/* Bilinear interpolation of the learned position grid onto this image's, then
 * permuted into merge-block order to match the patches.
 *
 * Done on the CPU: it is a gather of four rows per token out of a 48x48 table,
 * which is nothing next to a single block of the tower, and it keeps the
 * awkward index arithmetic somewhere it can be read. */
static void qw_vis_pos_embed(float *out, const qw_vision *v, const qw_config *c,
                             int32_t grid_h, int32_t grid_w) {
    const int32_t H = c->vis_hidden_size, side = v->grid_side, M = c->vis_spatial_merge_size;
    const uint16_t *table = qw_tensor_data(v->pos_embed);

    /* Row-major first, because that is the order the interpolation is defined
     * in; the permutation into merge-block order happens on the way out. */
    for (int32_t by = 0; by < grid_h / M; by++)
    for (int32_t bx = 0; bx < grid_w / M; bx++)
    for (int32_t my = 0; my < M; my++)
    for (int32_t mx = 0; mx < M; mx++) {
        const int32_t gy = by * M + my, gx = bx * M + mx;
        /* linspace(0, side - 1, grid) -- the endpoints land on the table's. */
        const float fy = grid_h > 1 ? (float)gy * (side - 1) / (float)(grid_h - 1) : 0.0f;
        const float fx = grid_w > 1 ? (float)gx * (side - 1) / (float)(grid_w - 1) : 0.0f;
        int32_t y0 = (int32_t)fy, x0 = (int32_t)fx;
        int32_t y1 = y0 + 1 < side ? y0 + 1 : side - 1;
        int32_t x1 = x0 + 1 < side ? x0 + 1 : side - 1;
        const float dy = fy - (float)y0, dx = fx - (float)x0;

        const float w00 = (1 - dy) * (1 - dx), w01 = (1 - dy) * dx;
        const float w10 = dy * (1 - dx),       w11 = dy * dx;
        const uint16_t *r00 = table + (size_t)(y0 * side + x0) * H;
        const uint16_t *r01 = table + (size_t)(y0 * side + x1) * H;
        const uint16_t *r10 = table + (size_t)(y1 * side + x0) * H;
        const uint16_t *r11 = table + (size_t)(y1 * side + x1) * H;

        const int32_t idx = ((by * (grid_w / M) + bx) * M + my) * M + mx;
        float *o = out + (size_t)idx * H;
        for (int32_t i = 0; i < H; i++)
            o[i] = w00 * qw_bf16_to_f32_c(r00[i]) + w01 * qw_bf16_to_f32_c(r01[i])
                 + w10 * qw_bf16_to_f32_c(r10[i]) + w11 * qw_bf16_to_f32_c(r11[i]);
    }
}

/* y = x * W^T + b, the tower's only kind of projection. */
static void qw_encode_vlinear(qw_cmd c, const qw_dense *d, qw_ref out, qw_ref in,
                              int32_t rows) {
    qw_op_dmm_bf16(c, out, in, qw_tensor_ref(d->weight),
                   d->in_features, d->out_features, rows);
    if (d->bias) qw_op_add_bias(c, out, qw_tensor_ref(d->bias), d->out_features, rows);
}

struct qw_vis_run {
    qw_buf h, hn, qkv, attn, ff, angles, pos, merged;
};

static void qw_vis_free(struct qw_vis_run *r) {
    qw_buf *all[] = { &r->h, &r->hn, &r->qkv, &r->attn, &r->ff,
                      &r->angles, &r->pos, &r->merged };
    for (size_t i = 0; i < sizeof all / sizeof *all; i++) qw_buf_free(*all[i]);
}

float *qwasar_encode_image(qwasar_engine *e, const qw_image *im,
                           int32_t *out_rows, char *err, size_t errcap) {
    const qw_config *c = qwasar_engine_config(e);
    const qw_vision *v = qwasar_engine_vision(e);
    if (!v->present) { qw_verrf(err, errcap, "this checkpoint has no vision tower"); return NULL; }

    const int32_t n = im->n_patches;
    const int32_t H = c->vis_hidden_size;
    const int32_t heads = c->vis_num_heads, hd = H / heads;
    const int32_t M = c->vis_spatial_merge_size;
    const int32_t merged_rows = n / (M * M);
    const int32_t merged_w = H * M * M;

    struct qw_vis_run r = { 0 };
    r.h      = qw_buf_alloc((size_t)n * H * 4);
    r.hn     = qw_buf_alloc((size_t)n * H * 4);
    r.qkv    = qw_buf_alloc((size_t)n * 3 * H * 4);
    r.attn   = qw_buf_alloc((size_t)n * H * 4);
    r.ff     = qw_buf_alloc((size_t)n * c->vis_intermediate_size * 4);
    r.angles = qw_buf_alloc((size_t)n * (hd / 2) * 4);
    r.pos    = qw_buf_alloc((size_t)n * H * 4);
    r.merged = qw_buf_alloc((size_t)merged_rows * c->vis_out_hidden_size * 4);
    if (!r.h || !r.hn || !r.qkv || !r.attn || !r.ff || !r.angles || !r.pos || !r.merged) {
        qw_verrf(err, errcap, "cannot allocate vision scratch for %d patches", n);
        qw_vis_free(&r);
        return NULL;
    }

    /* The patch data itself goes in a buffer of its own: it is the one input
     * and it is written from the CPU. */
    qw_buf patches = qw_buf_alloc((size_t)n * im->patch_elems * 4);
    if (!patches) {
        qw_verrf(err, errcap, "out of memory");
        qw_vis_free(&r);
        return NULL;
    }
    memcpy(qw_buf_contents(patches), im->patches,
           (size_t)n * im->patch_elems * sizeof(float));
    qw_vis_angles(qw_buf_contents(r.angles), c, im->grid_h, im->grid_w, n);
    qw_vis_pos_embed(qw_buf_contents(r.pos), v, c, im->grid_h, im->grid_w);

    qw_cmd cmd = qw_cmd_begin();
    if (!cmd) {
        qw_verrf(err, errcap, "cannot begin a command buffer");
        qw_buf_free(patches);
        qw_vis_free(&r);
        return NULL;
    }

    qw_encode_vlinear(cmd, &v->patch_embed, qw_ref_at(r.h, 0), qw_ref_at(patches, 0), n);
    qw_op_add_inplace(cmd, qw_ref_at(r.h, 0), qw_ref_at(r.pos, 0), n * H);

    for (int32_t b = 0; b < c->vis_depth; b++) {
        const qw_vis_block *blk = &v->blocks[b];

        qw_op_layer_norm(cmd, qw_ref_at(r.hn, 0), qw_ref_at(r.h, 0),
                         qw_tensor_ref(blk->norm1.weight), qw_tensor_ref(blk->norm1.bias),
                         H, n, 1e-6f);
        qw_encode_vlinear(cmd, &blk->qkv, qw_ref_at(r.qkv, 0), qw_ref_at(r.hn, 0), n);

        /* q and k rotate; v does not.  They are interleaved in the fused
         * output as [token][qkv][head][dim], so each is a strided view. */
        qw_op_rope_2d(cmd, qw_ref_at(r.qkv, 0), qw_ref_at(r.angles, 0),
                      n, heads, hd, 3 * H);
        qw_op_rope_2d(cmd, qw_ref_at(r.qkv, (size_t)H * 4), qw_ref_at(r.angles, 0),
                      n, heads, hd, 3 * H);

        qw_op_vision_attn(cmd, qw_ref_at(r.attn, 0), qw_ref_at(r.qkv, 0),
                          n, heads, hd, 1.0f / sqrtf((float)hd));
        qw_encode_vlinear(cmd, &blk->proj, qw_ref_at(r.hn, 0), qw_ref_at(r.attn, 0), n);
        qw_op_add_inplace(cmd, qw_ref_at(r.h, 0), qw_ref_at(r.hn, 0), n * H);

        qw_op_layer_norm(cmd, qw_ref_at(r.hn, 0), qw_ref_at(r.h, 0),
                         qw_tensor_ref(blk->norm2.weight), qw_tensor_ref(blk->norm2.bias),
                         H, n, 1e-6f);
        qw_encode_vlinear(cmd, &blk->fc1, qw_ref_at(r.ff, 0), qw_ref_at(r.hn, 0), n);
        qw_op_gelu_tanh(cmd, qw_ref_at(r.ff, 0), n * c->vis_intermediate_size);
        qw_encode_vlinear(cmd, &blk->fc2, qw_ref_at(r.hn, 0), qw_ref_at(r.ff, 0), n);
        qw_op_add_inplace(cmd, qw_ref_at(r.h, 0), qw_ref_at(r.hn, 0), n * H);
    }

    /* The merger normalises each patch on its own and then reads four at a
     * time.  That reshape is free only because the token order puts a 2x2
     * block's four patches next to each other. */
    qw_op_layer_norm(cmd, qw_ref_at(r.hn, 0), qw_ref_at(r.h, 0),
                     qw_tensor_ref(v->merger_norm.weight),
                     qw_tensor_ref(v->merger_norm.bias), H, n, 1e-6f);
    qw_encode_vlinear(cmd, &v->merger_fc1, qw_ref_at(r.qkv, 0), qw_ref_at(r.hn, 0),
                      merged_rows);
    qw_op_gelu_tanh(cmd, qw_ref_at(r.qkv, 0), merged_rows * merged_w);
    qw_encode_vlinear(cmd, &v->merger_fc2, qw_ref_at(r.merged, 0), qw_ref_at(r.qkv, 0),
                      merged_rows);

    qw_cmd_wait(cmd);
    const char *cerr = qw_cmd_error(cmd);
    qw_cmd_free(cmd);
    qw_buf_free(patches);
    if (cerr) {
        qw_verrf(err, errcap, "GPU error: %s", cerr);
        qw_vis_free(&r);
        return NULL;
    }

    float *out = malloc((size_t)merged_rows * c->vis_out_hidden_size * sizeof *out);
    if (out) memcpy(out, qw_buf_contents(r.merged),
                    (size_t)merged_rows * c->vis_out_hidden_size * sizeof *out);
    qw_vis_free(&r);
    if (!out) { qw_verrf(err, errcap, "out of memory"); return NULL; }
    *out_rows = merged_rows;
    return out;
}


/* ---- the public entry point ------------------------------------------------
 *
 * Loading and encoding are one step for callers, because the two are only ever
 * useful together: the row count an image produces is what the prompt has to
 * declare, so nothing can be rendered until the tower has run. */
bool qwasar_image_encode(qwasar_engine *e, const char *path,
                         qwasar_image_input *out, char *err, size_t errcap) {
    memset(out, 0, sizeof *out);
    qw_image im;
    if (!qw_image_load(&im, path, qwasar_engine_config(e), err, errcap)) return false;

    int32_t rows = 0;
    float *emb = qwasar_encode_image(e, &im, &rows, err, errcap);
    if (!emb) { qw_image_free(&im); return false; }

    out->rows      = emb;
    out->n_rows    = rows;
    out->grid_t    = im.grid_t;
    out->grid_h    = im.grid_h;
    out->grid_w    = im.grid_w;
    out->src_w     = im.src_w;
    out->src_h     = im.src_h;
    out->n_patches = im.n_patches;
    qw_image_free(&im);
    return true;
}

bool qwasar_image_encode_memory(qwasar_engine *e, const void *bytes, size_t len,
                                qwasar_image_input *out, char *err, size_t errcap) {
    memset(out, 0, sizeof *out);
    qw_image im;
    if (!qw_image_load_memory(&im, bytes, len, qwasar_engine_config(e), err, errcap))
        return false;

    int32_t rows = 0;
    float *emb = qwasar_encode_image(e, &im, &rows, err, errcap);
    if (!emb) { qw_image_free(&im); return false; }

    out->rows      = emb;
    out->n_rows    = rows;
    out->grid_t    = im.grid_t;
    out->grid_h    = im.grid_h;
    out->grid_w    = im.grid_w;
    out->src_w     = im.src_w;
    out->src_h     = im.src_h;
    out->n_patches = im.n_patches;
    qw_image_free(&im);
    return true;
}

void qwasar_image_release(qwasar_image_input *in) {
    if (!in) return;
    free((void *)in->rows);
    memset(in, 0, sizeof *in);
}
