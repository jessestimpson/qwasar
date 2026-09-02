/* The Flash-Next (qwen4_exp) family, against the reference.
 *
 * Loads the toy checkpoint tools/flashnext_tiny.py built and
 * tools/flashnext_convert.py quantised, and holds the engine's CPU reference
 * forward to the logits `transformers` produced from the SAME dequantised
 * values (tools/flashnext_oracle.py).  Every feature of the real model is on
 * in the toy -- four residual streams, MoE with a shared expert, the sigmoid
 * DeltaNet gate, QSA selecting blocks past 11 visible tokens, one PLE layer
 * with an EOS mid-prompt -- so a mechanic that is wrong anywhere shows up as
 * a layer whose residual diverges, named by number.
 *
 * Also pins the engram hash: the multipliers derived here must equal the
 * buffers the checkpoint itself carries. */

#include "qwasar_model.h"
#include "qwasar_json.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;

#define CHECK(cond, ...) do { \
    if (!(cond)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
                   fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); fails++; } \
} while (0)

/* Reads a JSON array of numbers (or of arrays of numbers) into a flat buffer. */
static int32_t read_floats(const qj_doc *d, const qj_node *arr, float *out, int32_t cap) {
    int32_t n = 0;
    for (const qj_node *v = qj_first(d, arr); v; v = qj_next(d, v)) {
        if (v->type == QJ_ARRAY) {
            n += read_floats(d, v, out ? out + n : NULL, cap - n);
        } else if (v->type == QJ_NUMBER) {
            if (out && n < cap) out[n] = (float)v->u.num;
            n++;
        }
    }
    return n;
}

int main(void) {
    const char *dir = "tests/fixtures/flashnext-tiny-q4";
    char err[512] = "";

    qwasar_options o = { .model_path = dir, .context_size = 256 };
    qwasar_engine *e = qwasar_engine_load(&o, err, sizeof err);
    CHECK(e != NULL, "load %s: %s", dir, err);
    if (!e) return 1;
    const qw_config *c = qwasar_engine_config(e);
    CHECK(c->family == QW_FAMILY_QWEN4_EXP, "family");
    CHECK(c->ple_layer == 1, "ple_layer = %d, expected 1 (zero-indexed)", c->ple_layer);
    CHECK(qwasar_engine_shape(e)->hc_hidden == 256, "hc_hidden = %d", qwasar_engine_shape(e)->hc_hidden);
    CHECK(!qwasar_engine_layer(e, 3)->is_linear_attn && qwasar_engine_layer(e, 2)->is_linear_attn,
          "layer schedule from layer_types");

    /* The hash constants, against what flashnext_tiny.py printed from the
     * checkpoint's own buffers (vocab 256, seed 1234, ple layer index 0). */
    const qw_ple *P = qwasar_engine_layer(e, 1)->ple;
    CHECK(P != NULL, "layer 1 carries the PLE");
    if (P) {
        static const int64_t want_mult[3] = { 4295158889762567LL, 35454616657218897LL, 19285316470915801LL };
        static const int64_t want_size[16] = { 101, 103, 107, 109, 113, 127, 131, 137,
                                               139, 149, 151, 157, 163, 167, 173, 179 };
        for (int i = 0; i < 3; i++)
            CHECK(P->mult[i] == want_mult[i], "multiplier %d = %lld, want %lld", i,
                  (long long)P->mult[i], (long long)want_mult[i]);
        for (int i = 0; i < 16; i++)
            CHECK(P->head_size[i] == want_size[i], "head %d size %lld, want %lld", i,
                  (long long)P->head_size[i], (long long)want_size[i]);
        CHECK(P->head_off[15] == 2027, "head 15 offset %lld", (long long)P->head_off[15]);
        CHECK(P->table_rows == 2304, "table rows %lld", (long long)P->table_rows);
    }

    /* The oracle. */
    char path[512];
    snprintf(path, sizeof path, "%s/oracle.json", dir);
    qj_doc d;
    if (!qj_parse_file(&d, path)) { CHECK(false, "cannot parse %s: %s", path, d.err); return 1; }
    const qj_node *root = qj_root(&d);
    const qj_node *tk = qj_get(&d, root, "tokens");
    int32_t tokens[64];
    int32_t n = 0;
    for (const qj_node *v = qj_first(&d, tk); v && n < 64; v = qj_next(&d, v)) tokens[n++] = (int32_t)v->u.num;
    const int32_t vocab = (int32_t)qj_int_or(&d, root, "vocab", 0);
    CHECK(vocab == c->vocab_size, "vocab %d vs config %d", vocab, c->vocab_size);

    float *want = malloc((size_t)n * vocab * sizeof(float));
    int32_t got_n = read_floats(&d, qj_get(&d, root, "logits_full"), want, n * vocab);
    CHECK(got_n == n * vocab, "oracle logits: %d values, expected %d", got_n, n * vocab);
    const int32_t HH = qwasar_engine_shape(e)->hc_hidden;
    float *want_h = malloc((size_t)c->num_hidden_layers * HH * sizeof(float));
    int32_t got_h = read_floats(&d, qj_get(&d, root, "hidden_last_per_layer"), want_h,
                                c->num_hidden_layers * HH);
    CHECK(got_h == c->num_hidden_layers * HH, "oracle hidden: %d values", got_h);

    /* The reference, as one prefill. */
    qw_flash_ref *r = qw_flash_ref_new(e, 256);
    CHECK(r != NULL, "reference state");
    float *got = calloc((size_t)n * vocab, sizeof(float));
    float *got_hid = calloc((size_t)c->num_hidden_layers * HH, sizeof(float));
    bool ok = qw_flash_ref_forward(r, tokens, n, got, got_hid, err, sizeof err);
    CHECK(ok, "forward: %s", err);

    /* Per-layer residual of the last token: names the first layer that
     * diverges, which is the point of recording it. */
    for (int32_t i = 0; i < c->num_hidden_layers; i++) {
        float md = 0.0f, scale = 0.0f;
        for (int32_t j = 0; j < HH; j++) {
            const float a = got_hid[(size_t)i * HH + j], b = want_h[(size_t)i * HH + j];
            if (fabsf(a - b) > md) md = fabsf(a - b);
            if (fabsf(b) > scale) scale = fabsf(b);
        }
        const bool pass = md <= 1e-3f * (scale > 1.0f ? scale : 1.0f);
        printf("  layer %d (%s%s): last-token residual max |diff| %.2e (scale %.2f)%s\n", i,
               qwasar_engine_layer(e, i)->is_linear_attn ? "delta" : "qsa",
               qwasar_engine_layer(e, i)->ple ? "+ple" : "", md, scale, pass ? "" : "  <-- DIVERGES");
        CHECK(pass, "layer %d residual diverges", i);
    }

    /* Logits at every position, and the argmax, which is what decoding
     * actually consumes. */
    float md = 0.0f;
    int32_t argmax_mismatch = 0;
    for (int32_t t = 0; t < n; t++) {
        int32_t ag = 0, aw = 0;
        for (int32_t v = 0; v < vocab; v++) {
            const float a = got[(size_t)t * vocab + v], b = want[(size_t)t * vocab + v];
            if (fabsf(a - b) > md) md = fabsf(a - b);
            if (a > got[(size_t)t * vocab + ag]) ag = v;
            if (b > want[(size_t)t * vocab + aw]) aw = v;
        }
        if (ag != aw) argmax_mismatch++;
    }
    printf("  logits: max |diff| %.2e over %d positions, argmax mismatches %d\n", md, n, argmax_mismatch);
    CHECK(md < 2e-3f, "logits max |diff| %.3e", md);
    CHECK(argmax_mismatch == 0, "%d argmax mismatches", argmax_mismatch);

    /* And the same sequence split across two calls: the state must carry. */
    qw_flash_ref *r2 = qw_flash_ref_new(e, 256);
    float *got2 = calloc((size_t)n * vocab, sizeof(float));
    const int32_t cut = 17;
    ok = qw_flash_ref_forward(r2, tokens, cut, got2, NULL, err, sizeof err)
      && qw_flash_ref_forward(r2, tokens + cut, n - cut, got2 + (size_t)cut * vocab, NULL, err, sizeof err);
    CHECK(ok, "split forward: %s", err);
    float md2 = 0.0f;
    for (int32_t i = 0; i < n * vocab; i++)
        if (fabsf(got2[i] - got[i]) > md2) md2 = fabsf(got2[i] - got[i]);
    printf("  split at %d vs one call: max |diff| %.2e\n", cut, md2);
    CHECK(md2 < 1e-5f, "state does not carry across calls: %.3e", md2);

    qw_flash_ref_free(r); qw_flash_ref_free(r2);
    free(want); free(want_h); free(got); free(got_hid); free(got2);
    qj_free(&d);
    qwasar_engine_free(e);

    if (fails) { fprintf(stderr, "%d failure(s)\n", fails); return 1; }
    printf("flashnext: all checks pass\n");
    return 0;
}
