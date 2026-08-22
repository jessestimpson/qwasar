/* Shape and wiring checks for the vision tower.
 *
 * The tower is 333 tensors and every one of them is bf16 with a bias, which is
 * true of nothing in the text model, so the binder that reads them shares no
 * code with the one that reads the language weights and gets no safety from
 * it.  This is that safety: every dimension asserted against the config's own
 * numbers, so a checkpoint this code was not written for fails at load with a
 * tensor name rather than somewhere inside a kernel. */

#include "qwasar.h"
#include "qwasar_gpu.h"
#include "qwasar_model.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

float qw_bf16_to_f32_c(uint16_t v);

static uint32_t crc32_of(const unsigned char *p, size_t n, uint32_t crc) {
    for (size_t i = 0; i < n; i++) {
        crc ^= p[i];
        for (int k = 0; k < 8; k++) crc = (crc >> 1) ^ (0xEDB88320u & (~(crc & 1) + 1));
    }
    return crc;
}

static void write_chunk(FILE *f, const char *tag, const unsigned char *d, size_t n) {
    unsigned char len[4] = { (unsigned char)(n >> 24), (unsigned char)(n >> 16),
                             (unsigned char)(n >> 8), (unsigned char)n };
    fwrite(len, 1, 4, f);
    fwrite(tag, 1, 4, f);
    if (n) fwrite(d, 1, n, f);
    uint32_t crc = crc32_of((const unsigned char *)tag, 4, 0xFFFFFFFFu);
    crc = crc32_of(d, n, crc) ^ 0xFFFFFFFFu;
    unsigned char c[4] = { (unsigned char)(crc >> 24), (unsigned char)(crc >> 16),
                           (unsigned char)(crc >> 8), (unsigned char)crc };
    fwrite(c, 1, 4, f);
}

static void fill_random(float *v, size_t n, uint32_t seed) {
    uint32_t s = seed | 1u;
    for (size_t i = 0; i < n; i++) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        v[i] = (float)((int32_t)(s >> 8) - 8388608) / 8388608.0f;
    }
}



static int fails;

#define CHECK(cond, ...) do { \
    if (!(cond)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
                   fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); fails++; } \
} while (0)

/* Golden vector for the vision tower, produced by mlx-vlm on the synthetic
 * 64x64 image tests/mkimage builds, fed the patches qwasar itself produced so
 * that only the tower is under test and not the resampling.
 *
 * Recorded as per-row L2 norms plus the first row's leading values rather than
 * the whole 64 x 5120 output: that is 1.3 MB of floats to check in, and a norm
 * per row is sensitive to anything structural -- a permuted patch order, a
 * rope applied to the wrong half, a merger reading the wrong four patches --
 * while fitting on a screen. */
static const float VIS_GOLDEN_NORMS[64] = {
    66.94350f, 47.81256f, 27.54277f, 41.78676f, 33.50095f, 23.61590f,
    16.80412f, 123.79469f, 68.66905f, 43.06242f, 75.77682f, 73.98386f,
    33.93633f, 26.53344f, 65.50393f, 39.61759f, 105.97025f, 33.40952f,
    75.99689f, 32.54572f, 32.34419f, 39.44880f, 18.52782f, 28.65358f,
    30.00046f, 16.14200f, 38.90185f, 55.56018f, 47.17209f, 35.34398f,
    21.26126f, 42.34690f, 79.15764f, 34.44782f, 52.68167f, 66.42735f,
    109.05813f, 50.03310f, 43.45312f, 60.25219f, 48.06636f, 33.89858f,
    42.79483f, 38.76944f, 36.88052f, 84.12689f, 22.95619f, 56.14314f,
    94.41985f, 73.28897f, 19.76522f, 37.47570f, 36.07173f, 21.70023f,
    39.16454f, 60.75447f, 71.46056f, 97.42012f, 46.28175f, 17.69352f,
    48.15721f, 17.03474f, 62.36465f, 72.22750f
};

static const float VIS_GOLDEN_ROW0[16] = {
    0.678459f, 0.002669f, 0.031700f, -0.555264f, -1.219867f, -0.302966f,
    0.450130f, 0.235568f, -0.043961f, 0.776728f, -0.630046f, 0.536313f,
    -0.151452f, 0.052578f, 0.637941f, 0.354013f
};

/* Builds the image the golden was recorded against: a deterministic 64x64 PNG,
 * written rather than checked in so the test carries no binary. */
static bool write_test_png(const char *path) {
    /* Uncompressed deflate blocks -- no zlib dependency, and the image is
     * small enough that the size does not matter. */
    const int W = 64, H = 64;
    unsigned char *raw = malloc((size_t)H * (W * 3 + 1));
    size_t rn = 0;
    for (int y = 0; y < H; y++) {
        raw[rn++] = 0;
        for (int x = 0; x < W; x++) {
            raw[rn++] = (unsigned char)((x * 4) % 256);
            raw[rn++] = (unsigned char)((y * 4) % 256);
            raw[rn++] = (unsigned char)(((x ^ y) * 3) % 256);
        }
    }
    /* zlib stream: header, stored blocks, adler32. */
    size_t cap = rn + rn / 65535 * 5 + 64;
    unsigned char *z = malloc(cap);
    size_t zn = 0;
    z[zn++] = 0x78; z[zn++] = 0x01;
    size_t off = 0;
    while (off < rn) {
        size_t chunk = rn - off > 65535 ? 65535 : rn - off;
        z[zn++] = (off + chunk >= rn) ? 1 : 0;
        z[zn++] = (unsigned char)(chunk & 0xFF);
        z[zn++] = (unsigned char)(chunk >> 8);
        z[zn++] = (unsigned char)(~chunk & 0xFF);
        z[zn++] = (unsigned char)((~chunk >> 8) & 0xFF);
        memcpy(z + zn, raw + off, chunk);
        zn += chunk;
        off += chunk;
    }
    uint32_t a = 1, b = 0;
    for (size_t i = 0; i < rn; i++) { a = (a + raw[i]) % 65521; b = (b + a) % 65521; }
    const uint32_t adler = (b << 16) | a;
    z[zn++] = adler >> 24; z[zn++] = adler >> 16; z[zn++] = adler >> 8; z[zn++] = adler;

    FILE *f = fopen(path, "wb");
    if (!f) { free(raw); free(z); return false; }
    static const unsigned char sig[8] = { 0x89,'P','N','G','\r','\n',0x1A,'\n' };
    fwrite(sig, 1, 8, f);
    unsigned char ihdr[13] = { 0,0,0,(unsigned char)W, 0,0,0,(unsigned char)H, 8, 2, 0, 0, 0 };
    write_chunk(f, "IHDR", ihdr, sizeof ihdr);
    write_chunk(f, "IDAT", z, zn);
    write_chunk(f, "IEND", NULL, 0);
    fclose(f);
    free(raw); free(z);
    return true;
}

static void compare(const char *what, const float *got, const float *want,
                    size_t n, double tol) {
    double num = 0.0, den = 0.0, worst = 0.0;
    for (size_t i = 0; i < n; i++) {
        const double d = (double)got[i] - (double)want[i];
        num += d * d;
        den += (double)want[i] * (double)want[i];
        if (fabs(d) > worst) worst = fabs(d);
    }
    const double rel = den > 0.0 ? sqrt(num / den) : sqrt(num);
    CHECK(rel < tol, "%s: rel l2 %.3g exceeds %.3g", what, rel, tol);
    printf("  %-22s rel l2 %.2e  max abs %.2e\n", what, rel, worst);
}

static void check_dense(const char *what, const qw_dense *d,
                        int32_t in, int32_t out, bool want_bias) {
    CHECK(d->weight != NULL, "%s has no weight", what);
    CHECK(d->in_features == in && d->out_features == out,
          "%s is [%d -> %d], expected [%d -> %d]",
          what, d->in_features, d->out_features, in, out);
    CHECK(!want_bias || d->bias != NULL, "%s has no bias", what);
    if (d->bias) CHECK(d->bias->shape[0] == out, "%s bias is [%lld], expected [%d]",
                       what, (long long)d->bias->shape[0], out);
}

int main(int argc, char **argv) {
    const char *model = getenv("QWASAR_TEST_MODEL");
    if (argc > 1) model = argv[1];
    if (!model || !*model) {
        fprintf(stderr, "skip: set QWASAR_TEST_MODEL to a model directory\n");
        return 0;
    }

    char err[512];
    qwasar_options opts = { .model_path = model };
    qwasar_engine *e = qwasar_engine_load(&opts, err, sizeof err);
    if (!e) { fprintf(stderr, "load failed: %s\n", err); return 1; }

    const qw_config *c = qwasar_engine_config(e);
    const qw_vision *v = qwasar_engine_vision(e);

    if (!c->has_vision) {
        fprintf(stderr, "skip: checkpoint has no vision tower\n");
        qwasar_engine_free(e);
        return 0;
    }
    CHECK(v->present, "vision tower did not bind");
    if (!v->present) return 1;

    const int32_t h = c->vis_hidden_size;
    const int32_t patch = c->vis_temporal_patch_size * c->vis_patch_size
                        * c->vis_patch_size * c->vis_in_channels;
    const int32_t merged = h * c->vis_spatial_merge_size * c->vis_spatial_merge_size;

    /* A patch is temporal x spatial x spatial x channels, flattened.  Stating
     * it here means a checkpoint with a different patch geometry fails on this
     * line rather than by producing plausible nonsense. */
    check_dense("patch_embed", &v->patch_embed, patch, h, true);
    CHECK(patch == 1536, "a patch is %d values, expected 1536", patch);

    CHECK(v->pos_embed != NULL, "no position embedding");
    CHECK(v->grid_side * v->grid_side == c->vis_num_position_embeddings,
          "%d position embeddings is not the square of %d",
          c->vis_num_position_embeddings, v->grid_side);
    printf("  patch %d values -> %d, position grid %dx%d\n",
           patch, h, v->grid_side, v->grid_side);

    /* Attention head geometry has to divide, or the qkv split is meaningless. */
    CHECK(h % c->vis_num_heads == 0, "hidden %d does not divide into %d heads",
          h, c->vis_num_heads);
    const int32_t head_dim = h / c->vis_num_heads;
    CHECK(head_dim % 4 == 0, "head dim %d does not admit a half-split rope",
          head_dim);
    printf("  %d heads x %d\n", c->vis_num_heads, head_dim);

    for (int32_t i = 0; i < c->vis_depth; i++) {
        const qw_vis_block *b = &v->blocks[i];
        char what[64];
        snprintf(what, sizeof what, "block %d qkv", i);
        check_dense(what, &b->qkv, h, 3 * h, true);
        snprintf(what, sizeof what, "block %d proj", i);
        check_dense(what, &b->proj, h, h, true);
        snprintf(what, sizeof what, "block %d fc1", i);
        check_dense(what, &b->fc1, h, c->vis_intermediate_size, true);
        snprintf(what, sizeof what, "block %d fc2", i);
        check_dense(what, &b->fc2, c->vis_intermediate_size, h, true);
        CHECK(b->norm1.weight && b->norm1.bias && b->norm2.weight && b->norm2.bias,
              "block %d is missing a layer norm", i);
    }
    printf("  %d blocks bound, hidden %d, ffn %d\n",
           c->vis_depth, h, c->vis_intermediate_size);

    /* The merger normalises one patch and projects four, and those two widths
     * disagreeing by anything but the merge factor is a spatial-merge mistake. */
    CHECK(v->merger_norm.weight && v->merger_norm.weight->shape[0] == h,
          "merger norm is not [%d]", h);
    check_dense("merger fc1", &v->merger_fc1, merged, merged, true);
    check_dense("merger fc2", &v->merger_fc2, merged, c->vis_out_hidden_size, true);
    CHECK(merged == h * 4, "a 2x2 merge of %d should be %d, config says %d",
          h, h * 4, merged);
    CHECK(v->merger_fc2.out_features == c->hidden_size,
          "the merger emits %d, the text model takes %d",
          v->merger_fc2.out_features, c->hidden_size);
    printf("  merger %d -> %d, matching the text model's width\n",
           merged, v->merger_fc2.out_features);

    /* ---- the ops, each against its scalar twin --------------------------
     *
     * All five are new: the text model has no LayerNorm, no bias, no GELU, no
     * half-split rope and no attention without a cache.  None of them can be
     * checked by looking at an image. */
    {
        const int32_t tokens = 37, dim = h, heads = c->vis_num_heads;
        const int32_t hd = h / heads;

        qw_buf xb = qw_buf_alloc((size_t)tokens * dim * 4);
        qw_buf yb = qw_buf_alloc((size_t)tokens * dim * 4);
        float *want = malloc((size_t)tokens * dim * sizeof *want);
        float *x = qw_buf_contents(xb);
        fill_random(x, (size_t)tokens * dim, 0xB16B00B5u);

        /* LayerNorm, with the tower's own weights rather than synthetic ones:
         * a scale-and-shift bug hides when the scale is 1. */
        qw_cmd c1 = qw_cmd_begin();
        qw_op_layer_norm(c1, qw_ref_at(yb, 0), qw_ref_at(xb, 0),
                         qw_tensor_ref(v->blocks[0].norm1.weight),
                         qw_tensor_ref(v->blocks[0].norm1.bias), dim, tokens, 1e-6f);
        qw_cmd_wait(c1);
        CHECK(qw_cmd_error(c1) == NULL, "layer norm: %s", qw_cmd_error(c1));
        qw_cmd_free(c1);
        qw_cpu_layer_norm(want, x, qw_tensor_data(v->blocks[0].norm1.weight),
                          qw_tensor_data(v->blocks[0].norm1.bias), dim, tokens, 1e-6f);
        compare("layer norm", qw_buf_contents(yb), want, (size_t)tokens * dim, 2e-6);

        /* GELU in place. */
        memcpy(qw_buf_contents(yb), x, (size_t)tokens * dim * sizeof(float));
        memcpy(want, x, (size_t)tokens * dim * sizeof(float));
        qw_cmd c2 = qw_cmd_begin();
        qw_op_gelu_tanh(c2, qw_ref_at(yb, 0), tokens * dim);
        qw_cmd_wait(c2);
        qw_cmd_free(c2);
        qw_cpu_gelu_tanh(want, tokens * dim);
        compare("gelu tanh", qw_buf_contents(yb), want, (size_t)tokens * dim, 2e-6);

        /* Bias add. */
        memcpy(qw_buf_contents(yb), x, (size_t)tokens * dim * sizeof(float));
        qw_cmd c3 = qw_cmd_begin();
        qw_op_add_bias(c3, qw_ref_at(yb, 0), qw_tensor_ref(v->blocks[0].proj.bias),
                       dim, tokens);
        qw_cmd_wait(c3);
        qw_cmd_free(c3);
        {
            const uint16_t *bs = qw_tensor_data(v->blocks[0].proj.bias);
            for (int32_t r = 0; r < tokens; r++)
                for (int32_t i = 0; i < dim; i++)
                    want[(size_t)r * dim + i] = x[(size_t)r * dim + i]
                                              + qw_bf16_to_f32_c(bs[i]);
        }
        compare("add bias", qw_buf_contents(yb), want, (size_t)tokens * dim, 1e-6);

        free(want);
        qw_buf_free(xb);
        qw_buf_free(yb);

        /* Rope and attention share the fused [tokens, 3, heads, dim] layout,
         * so they are checked on it -- a stride mistake there is exactly the
         * kind of bug that survives a check on a tidier buffer. */
        const size_t qkv_n = (size_t)tokens * 3 * dim;
        qw_buf qb = qw_buf_alloc(qkv_n * 4);
        qw_buf ab = qw_buf_alloc((size_t)tokens * (hd / 2) * 4);
        qw_buf ob = qw_buf_alloc((size_t)tokens * dim * 4);
        float *qkv = qw_buf_contents(qb);
        fill_random(qkv, qkv_n, 0x5EED1234u);
        fill_random(qw_buf_contents(ab), (size_t)tokens * (hd / 2), 0xC0FFEEu);

        float *qref = malloc(qkv_n * sizeof *qref);
        memcpy(qref, qkv, qkv_n * sizeof *qref);

        qw_cmd c4 = qw_cmd_begin();
        qw_op_rope_2d(c4, qw_ref_at(qb, 0), qw_ref_at(ab, 0), tokens, heads, hd, 3 * dim);
        qw_cmd_wait(c4);
        CHECK(qw_cmd_error(c4) == NULL, "rope 2d: %s", qw_cmd_error(c4));
        qw_cmd_free(c4);
        qw_cpu_rope_2d(qref, qw_buf_contents(ab), tokens, heads, hd, 3 * dim);
        compare("rope 2d (q only)", qw_buf_contents(qb), qref, qkv_n, 2e-6);

        /* Only q should have moved; k and v sit at the same stride and must be
         * untouched, which is what proves the stride was applied. */
        {
            const float *got = qw_buf_contents(qb);
            double moved = 0.0;
            for (int32_t t = 0; t < tokens; t++)
                for (int32_t i = dim; i < 3 * dim; i++) {
                    const size_t o = (size_t)t * 3 * dim + i;
                    moved += fabs((double)got[o] - qref[o]);
                }
            CHECK(moved == 0.0, "rope touched k or v (sum |diff| %.3g)", moved);
        }

        qw_cmd c5 = qw_cmd_begin();
        qw_op_vision_attn(c5, qw_ref_at(ob, 0), qw_ref_at(qb, 0),
                          tokens, heads, hd, 1.0f / sqrtf((float)hd));
        qw_cmd_wait(c5);
        CHECK(qw_cmd_error(c5) == NULL, "vision attn: %s", qw_cmd_error(c5));
        qw_cmd_free(c5);
        float *aref = malloc((size_t)tokens * dim * sizeof *aref);
        qw_cpu_vision_attn(aref, qw_buf_contents(qb), tokens, heads, hd,
                           1.0f / sqrtf((float)hd));
        compare("vision attention", qw_buf_contents(ob), aref, (size_t)tokens * dim, 5e-6);

        free(qref); free(aref);
        qw_buf_free(qb); qw_buf_free(ab); qw_buf_free(ob);
    }

    /* ---- the whole tower, against the reference -------------------------
     *
     * The ops above being right does not make the tower right: patch order,
     * position interpolation and the merger's grouping are all wiring, and
     * wiring produces a confidently wrong answer rather than a wrong number.
     * This is the check that covers them, on a golden recorded from mlx-vlm. */
    {
        char png[1200];
        snprintf(png, sizeof png, "%s", "/tmp/qwasar-vision-test.png");
        if (!write_test_png(png)) {
            CHECK(false, "cannot write %s", png);
        } else {
            qw_image im;
            if (!qw_image_load(&im, png, c, err, sizeof err)) {
                CHECK(false, "image load: %s", err);
            } else {
                CHECK(im.grid_h == 16 && im.grid_w == 16,
                      "64x64 should be a 16x16 patch grid, got %dx%d",
                      im.grid_h, im.grid_w);
                CHECK(im.n_patches == 256, "expected 256 patches, got %d", im.n_patches);

                int32_t rows = 0;
                float *emb = qwasar_encode_image(e, &im, &rows, err, sizeof err);
                if (!emb) {
                    CHECK(false, "encode: %s", err);
                } else {
                    CHECK(rows == 64, "expected 64 merged rows, got %d", rows);
                    const int32_t d = c->vis_out_hidden_size;

                    double worst = 0.0;
                    int worst_row = -1;
                    for (int32_t r = 0; r < rows && r < 64; r++) {
                        double ss = 0.0;
                        for (int32_t i = 0; i < d; i++) {
                            const double vv = emb[(size_t)r * d + i];
                            ss += vv * vv;
                        }
                        const double got = sqrt(ss);
                        const double rel = fabs(got - VIS_GOLDEN_NORMS[r])
                                         / VIS_GOLDEN_NORMS[r];
                        if (rel > worst) { worst = rel; worst_row = r; }
                    }
                    /* The threshold is measured, not chosen.  Running the
                     * reference against itself with its weights promoted to
                     * fp32 moves its own row norms by up to 1.56e-2, and
                     * qwasar sits closer to that fp32 result (3.6e-3 overall)
                     * than the reference's own bf16 path does (3.7e-3).  So
                     * anything at this scale is the tower's precision, not its
                     * correctness, and a tighter bound would only fail on
                     * arithmetic order.  A structural mistake -- patches in the
                     * wrong order, a rope on the wrong half, a merger reading
                     * the wrong four -- lands orders of magnitude above it. */
                    CHECK(worst < 2.5e-2, "row norm %d differs by %.3g", worst_row, worst);
                    printf("  %-22s worst row norm %.2e vs mlx-vlm\n",
                           "tower golden", worst);

                    compare("tower row 0", emb, VIS_GOLDEN_ROW0, 16, 1e-2);
                    printf("  %-22s reference disagrees with itself by 1.6e-2\n",
                           "(for scale)");
                    free(emb);
                }
                qw_image_free(&im);
            }
            remove(png);
        }
    }

    /* ---- MRoPE ----------------------------------------------------------
     *
     * The one part of the image path that is pure index arithmetic, so it has
     * an exactly known answer rather than a tolerance.  It is also the part
     * that has been deferred since milestone 1, on the grounds that text
     * advances all three axes together and only an image makes them differ. */
    {
        const int32_t merge = c->vis_spatial_merge_size;
        const int32_t gh = 16, gw = 16;             /* patches */
        const int32_t mh = gh / merge, mw = gw / merge;   /* 8 x 8 = 64 tokens */
        const int32_t n_img = mh * mw;
        const int32_t pre = 5, post = 4;
        const int32_t n = pre + n_img + post;

        int32_t *toks = malloc((size_t)n * sizeof *toks);
        for (int32_t i = 0; i < n; i++) toks[i] = 1000 + i;
        for (int32_t i = 0; i < n_img; i++) toks[pre + i] = c->image_token_id;

        qwasar_image_input in = { NULL, n_img, 1, gh, gw };
        int32_t *pos = malloc((size_t)3 * n * sizeof *pos);
        int32_t next = -1;
        CHECK(qw_mrope_positions(c, toks, n, &in, 1, 0, pos, &next, err, sizeof err),
              "mrope: %s", err);

        /* Text before the image: all three axes agree and count up. */
        for (int32_t i = 0; i < pre; i++)
            CHECK(pos[i] == i && pos[n + i] == i && pos[2 * n + i] == i,
                  "text position %d is (%d,%d,%d), expected (%d,%d,%d)",
                  i, pos[i], pos[n + i], pos[2 * n + i], i, i, i);

        /* The image: frame constant, row and column walking the merged grid,
         * all offset by where the image starts. */
        for (int32_t y = 0; y < mh; y++)
            for (int32_t x = 0; x < mw; x++) {
                const int32_t k = pre + y * mw + x;
                CHECK(pos[k] == pre && pos[n + k] == pre + y && pos[2 * n + k] == pre + x,
                      "image token (%d,%d) is (%d,%d,%d), expected (%d,%d,%d)",
                      y, x, pos[k], pos[n + k], pos[2 * n + k], pre, pre + y, pre + x);
            }

        /* And the text after resumes from one past the largest axis, not from
         * one past the token count.  64 image tokens advance position by 8. */
        const int32_t resume = pre + mh;
        for (int32_t i = 0; i < post; i++) {
            const int32_t k = pre + n_img + i;
            CHECK(pos[k] == resume + i,
                  "text after the image is at %d, expected %d", pos[k], resume + i);
        }
        CHECK(next == resume + post, "next position is %d, expected %d",
              next, resume + post);
        printf("  %-22s %d image tokens advance position by %d\n",
               "mrope", n_img, mh);

        free(toks);
        free(pos);
    }

    qwasar_engine_free(e);
    qw_gpu_shutdown();

    if (fails) { fprintf(stderr, "%d check(s) failed\n", fails); return 1; }
    printf("ok: vision\n");
    return 0;
}
