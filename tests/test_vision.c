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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;

#define CHECK(cond, ...) do { \
    if (!(cond)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
                   fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); fails++; } \
} while (0)

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

    qwasar_engine_free(e);
    qw_gpu_shutdown();

    if (fails) { fprintf(stderr, "%d check(s) failed\n", fails); return 1; }
    printf("ok: vision\n");
    return 0;
}
