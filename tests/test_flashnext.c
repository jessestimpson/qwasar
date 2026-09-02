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

#include <float.h>
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
    const int32_t NL = c->num_hidden_layers;
    float *want_h = malloc((size_t)NL * n * HH * sizeof(float));
    int32_t got_h = read_floats(&d, qj_get(&d, root, "hidden_per_layer"), want_h, NL * n * HH);
    CHECK(got_h == NL * n * HH, "oracle hidden: %d values, expected %d", got_h, NL * n * HH);

    /* The reference, one token at a time so every position's residual is
     * seen -- a discrete flip (a routing tie, a block selection) shows at
     * one position and nowhere else. */
    qw_flash_ref *r = qw_flash_ref_new(e, 256);
    CHECK(r != NULL, "reference state");
    qw_flash_ref_debug(r, true);
    float *got = calloc((size_t)n * vocab, sizeof(float));
    float *got_hid = calloc((size_t)NL * n * HH, sizeof(float));
    float *one = calloc((size_t)NL * HH, sizeof(float));
    bool ok = true;
    for (int32_t t = 0; t < n && ok; t++) {
        ok = qw_flash_ref_forward(r, tokens + t, 1, got + (size_t)t * vocab, one, err, sizeof err);
        for (int32_t i = 0; i < NL; i++)
            memcpy(got_hid + ((size_t)i * n + t) * HH, one + (size_t)i * HH, (size_t)HH * sizeof(float));
    }
    CHECK(ok, "forward: %s", err);
    free(one);

    /* The discrete choices, exactly: which experts every token went to, and
     * which cache positions every query was allowed to see.  A residual that
     * differs at one position with these equal is a bug; with these unequal
     * it is a tie -- and the gap check below says a tie is the fixture's
     * fault, not the engine's. */
    {
        const int32_t K = c->num_experts_per_tok;
        int32_t route_bad = 0, mask_bad = 0;
        const qj_node *rl = qj_get(&d, root, "routes_per_layer");
        const qj_node *ml = qj_get(&d, root, "masks_per_layer");
        int32_t li = 0;
        for (const qj_node *lr = qj_first(&d, rl); lr && li < NL; lr = qj_next(&d, lr), li++) {
            int32_t t = 0;
            for (const qj_node *tr = qj_first(&d, lr); tr && t < n; tr = qj_next(&d, tr), t++) {
                int32_t want_k[64], nk = 0;
                for (const qj_node *v = qj_first(&d, tr); v && nk < 64; v = qj_next(&d, v)) want_k[nk++] = (int32_t)v->u.num;
                const int32_t *got_k = qw_flash_ref_routes(r, li, t);
                /* order-insensitive: top-k lists the same set either way */
                for (int32_t a = 0; a < K; a++) {
                    bool found = false;
                    for (int32_t b = 0; b < nk; b++) if (want_k[b] == got_k[a]) found = true;
                    if (!found) {
                        if (route_bad < 5) printf("  route mismatch: layer %d position %d expert %d not in the oracle's set\n", li, t, got_k[a]);
                        route_bad++;
                    }
                }
            }
        }
        li = 0;
        for (const qj_node *lm = qj_first(&d, ml); lm && li < NL; lm = qj_next(&d, lm), li++) {
            if (lm->type != QJ_ARRAY) continue;      /* null for linear layers */
            int32_t t = 0;
            for (const qj_node *tr = qj_first(&d, lm); tr && t < n; tr = qj_next(&d, tr), t++) {
                const uint8_t *got_m = qw_flash_ref_mask(r, li, t);
                int32_t p = 0;
                for (const qj_node *v = qj_first(&d, tr); v && p <= t; v = qj_next(&d, v), p++) {
                    const int32_t w = (int32_t)v->u.num;
                    if ((w != 0) != (got_m[p] != 0)) {
                        if (mask_bad < 5) printf("  mask mismatch: layer %d query %d key %d: oracle %d, reference %d (selection gap %.3e)\n",
                                                 li, t, p, w, got_m[p], qw_flash_ref_select_gap(r, li, t));
                        mask_bad++;
                    }
                }
            }
        }
        float min_gap = FLT_MAX;
        for (int32_t i = 0; i < NL; i++) {
            if (qwasar_engine_layer(e, i)->is_linear_attn) continue;
            for (int32_t t = 0; t < n; t++) {
                const float g = qw_flash_ref_select_gap(r, i, t);
                if (g < 1e-3f) printf("  near-tie: layer %d query %d gap %.3e\n", i, t, g);
                if (g < min_gap) min_gap = g;
            }
        }
        printf("  routing: %d expert choices differ; selection: %d mask entries differ; "
               "least decisive block cut %.3e\n", route_bad, mask_bad, min_gap);
        CHECK(min_gap > 1e-4f, "the fixture has a near-tie in block selection (gap %.3e); regenerate it", min_gap);
        CHECK(route_bad == 0, "%d expert choices differ from the oracle", route_bad);
        CHECK(mask_bad == 0, "%d QSA mask entries differ from the oracle", mask_bad);
    }

    /* Per-layer, per-position residuals: name the first place that diverges. */
    for (int32_t i = 0; i < NL; i++) {
        float md = 0.0f, scale = 0.0f;
        int32_t at = -1;
        for (int32_t t = 0; t < n; t++)
            for (int32_t j = 0; j < HH; j++) {
                const float a = got_hid[((size_t)i * n + t) * HH + j];
                const float b = want_h[((size_t)i * n + t) * HH + j];
                if (fabsf(a - b) > md) { md = fabsf(a - b); at = t; }
                if (fabsf(b) > scale) scale = fabsf(b);
            }
        const bool pass = md <= 1e-3f * (scale > 1.0f ? scale : 1.0f);
        printf("  layer %d (%s%s): residual max |diff| %.2e at position %d (scale %.2f)%s\n", i,
               qwasar_engine_layer(e, i)->is_linear_attn ? "delta" : "qsa",
               qwasar_engine_layer(e, i)->ple ? "+ple" : "", md, at, scale,
               pass ? "" : "  <-- DIVERGES");
        CHECK(pass, "layer %d residual diverges at position %d", i, at);
    }

    /* Logits at every position, and the argmax, which is what decoding
     * actually consumes. */
    float md = 0.0f;
    int32_t argmax_mismatch = 0, worst = -1;
    for (int32_t t = 0; t < n; t++) {
        int32_t ag = 0, aw = 0;
        float mt = 0.0f;
        for (int32_t v = 0; v < vocab; v++) {
            const float a = got[(size_t)t * vocab + v], b = want[(size_t)t * vocab + v];
            if (fabsf(a - b) > mt) mt = fabsf(a - b);
            if (a > got[(size_t)t * vocab + ag]) ag = v;
            if (b > want[(size_t)t * vocab + aw]) aw = v;
        }
        if (mt > md) { md = mt; worst = t; }
        if (ag != aw) argmax_mismatch++;
        if (mt > 2e-3f) printf("  position %d: max |diff| %.2e\n", t, mt);
    }
    printf("  logits: max |diff| %.2e (position %d) over %d positions, argmax mismatches %d\n",
           md, worst, n, argmax_mismatch);
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

    /* The Metal path (qwasar_flash_graph.c), held to the reference: every
     * position through a session one token at a time, and the same prompt
     * as one prefill.  fp16 KV cache and a different summation order buy a
     * small drift; a routing or selection flip would show as a large one. */
    {
        /* Layer by layer first: a fresh session per layer, stopped after it,
         * its residual streams read back after every token. */
        for (int32_t L = 0; L < NL; L++) {
            qwasar_session *ds = qwasar_session_new(e, err, sizeof err);
            if (!ds) { CHECK(false, "metal session: %s", err); break; }
            qw_flash_debug_stop(ds, L);
            float md = 0.0f, scale = 0.0f;
            int32_t at = -1;
            for (int32_t t = 0; t < n; t++) {
                if (!qwasar_session_eval(ds, tokens + t, 1, err, sizeof err)) { CHECK(false, "metal: %s", err); break; }
                const float *h4 = qw_flash_debug_h4(ds);
                for (int32_t j = 0; j < HH; j++) {
                    const float a = h4[j], b = got_hid[((size_t)L * n + t) * HH + j];
                    if (fabsf(a - b) > md) { md = fabsf(a - b); at = t; }
                    if (fabsf(b) > scale) scale = fabsf(b);
                }
            }
            printf("  metal layer %d (%s%s): residual max |diff| vs reference %.2e at position %d (scale %.2f)\n",
                   L, qwasar_engine_layer(e, L)->is_linear_attn ? "delta" : "qsa",
                   qwasar_engine_layer(e, L)->ple ? "+ple" : "", md, at, scale);
            qwasar_session_free(ds);
        }

        qwasar_session *ms = qwasar_session_new(e, err, sizeof err);
        CHECK(ms != NULL, "metal session: %s", err);
        if (ms) {
            float *mg = calloc((size_t)n * vocab, sizeof(float));
            bool mok = true;
            for (int32_t t = 0; t < n && mok; t++) {
                const float *lg = qwasar_session_eval(ms, tokens + t, 1, err, sizeof err);
                if (!lg) { mok = false; break; }
                memcpy(mg + (size_t)t * vocab, lg, (size_t)vocab * sizeof(float));
            }
            CHECK(mok, "metal decode: %s", err);
            float mmd = 0.0f, mscale = 0.0f;
            int32_t mworst = -1, margmax = 0;
            for (int32_t t = 0; t < n; t++) {
                int32_t ag = 0, aw = 0;
                for (int32_t v = 0; v < vocab; v++) {
                    const float a = mg[(size_t)t * vocab + v], b = got[(size_t)t * vocab + v];
                    if (fabsf(a - b) > mmd) { mmd = fabsf(a - b); mworst = t; }
                    if (fabsf(b) > mscale) mscale = fabsf(b);
                    if (a > mg[(size_t)t * vocab + ag]) ag = v;
                    if (b > got[(size_t)t * vocab + aw]) aw = v;
                }
                if (ag != aw) margmax++;
            }
            printf("  metal, token by token: max |diff| vs reference %.2e (position %d, logit scale %.2f), argmax mismatches %d\n",
                   mmd, mworst, mscale, margmax);
            CHECK(mmd <= 5e-3f * (mscale > 1.0f ? mscale : 1.0f), "metal decode diverges: %.3e", mmd);
            CHECK(margmax == 0, "metal decode: %d argmax mismatches", margmax);

            qwasar_session *ms2 = qwasar_session_new(e, err, sizeof err);
            const float *lg = ms2 ? qwasar_session_eval(ms2, tokens, n, err, sizeof err) : NULL;
            CHECK(lg != NULL, "metal prefill: %s", err);
            if (lg) {
                float pmd = 0.0f;
                for (int32_t v = 0; v < vocab; v++)
                    if (fabsf(lg[v] - got[(size_t)(n - 1) * vocab + v]) > pmd)
                        pmd = fabsf(lg[v] - got[(size_t)(n - 1) * vocab + v]);
                printf("  metal, one prefill: last-token max |diff| vs reference %.2e\n", pmd);
                CHECK(pmd <= 5e-3f * (mscale > 1.0f ? mscale : 1.0f), "metal prefill diverges: %.3e", pmd);
            }
            qwasar_session_free(ms2);
            qwasar_session_free(ms);
            free(mg);
        }
    }

    qw_flash_ref_free(r); qw_flash_ref_free(r2);
    free(want); free(want_h); free(got); free(got_hid); free(got2);
    qj_free(&d);
    qwasar_engine_free(e);

    if (fails) { fprintf(stderr, "%d failure(s)\n", fails); return 1; }
    printf("flashnext: all checks pass\n");
    return 0;
}
