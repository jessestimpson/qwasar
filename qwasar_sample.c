/* Token sampling.
 *
 * The filter chain runs in a fixed order: temperature, then top-k, then min-p,
 * then top-p.  Order is a real choice rather than a detail -- min-p thresholds
 * against the most likely token, so running it after top-k means the threshold
 * is computed from whatever top-k left, which is what callers expect from the
 * name.
 *
 * The full vocabulary is 248320 entries and sorting it costs more than a decode
 * step can spare, so nothing is sorted until the distribution has been pruned.
 * min-p prunes hardest and is cheap, so it runs as a threshold pass over the
 * raw probabilities first, and only the survivors -- usually a few dozen -- are
 * ever ordered.  With every filter disabled, no sort happens at all: the
 * cumulative walk over the unsorted distribution samples the same law. */

#include "qwasar.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* xoshiro-style splitmix64; small, seedable, and good enough for sampling. */
static uint64_t qw_rng_next(uint64_t *state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

static float qw_rng_float(uint64_t *state) {
    /* 24 bits of mantissa is plenty and keeps the result strictly below 1. */
    return (float)(qw_rng_next(state) >> 40) / 16777216.0f;
}

typedef struct { int32_t id; float p; } qw_cand;

static int qw_cand_desc(const void *a, const void *b) {
    float x = ((const qw_cand *)a)->p, y = ((const qw_cand *)b)->p;
    return x < y ? 1 : x > y ? -1 : 0;
}

void qwasar_sampling_defaults(qwasar_sampling *sp) {
    /* Matches the model's own generation_config. */
    sp->temperature = 1.0f;
    sp->top_k = 20;
    sp->top_p = 0.95f;
    sp->min_p = 0.0f;
    sp->seed = 0;
}

int32_t qwasar_sample(const float *logits, int32_t n, const qwasar_sampling *sp,
                      uint64_t *rng) {
    if (n <= 0) return 0;

    /* Greedy is not a special case of the chain -- it is what callers mean by
     * temperature 0, and it must be exactly reproducible. */
    if (!sp || sp->temperature <= 0.0f) {
        int32_t best = 0;
        for (int32_t i = 1; i < n; i++) if (logits[i] > logits[best]) best = i;
        return best;
    }

    float max_logit = logits[0];
    for (int32_t i = 1; i < n; i++) if (logits[i] > max_logit) max_logit = logits[i];

    const float inv_t = 1.0f / sp->temperature;
    const bool filtering = (sp->top_k > 0 && sp->top_k < n)
                        || (sp->top_p > 0.0f && sp->top_p < 1.0f)
                        || (sp->min_p > 0.0f);

    if (!filtering) {
        /* Sample straight from the unsorted distribution: two passes, no
         * allocation, no ordering. */
        double sum = 0.0;
        for (int32_t i = 0; i < n; i++) sum += exp((double)(logits[i] - max_logit) * inv_t);
        double target = (double)qw_rng_float(rng) * sum;
        for (int32_t i = 0; i < n; i++) {
            target -= exp((double)(logits[i] - max_logit) * inv_t);
            if (target <= 0.0) return i;
        }
        return n - 1;
    }

    /* Prune before ordering.  exp(x - max) is 1 at the mode, so a candidate is
     * kept when its unnormalised weight clears min_p; that is the same test as
     * against the normalised max probability, without needing the sum yet. */
    const float keep = sp->min_p > 0.0f ? sp->min_p : 0.0f;
    qw_cand *cand = malloc((size_t)n * sizeof *cand);
    if (!cand) {
        int32_t best = 0;
        for (int32_t i = 1; i < n; i++) if (logits[i] > logits[best]) best = i;
        return best;
    }

    int32_t m = 0;
    for (int32_t i = 0; i < n; i++) {
        float w = (float)exp((double)(logits[i] - max_logit) * inv_t);
        if (w < keep) continue;
        cand[m].id = i;
        cand[m].p = w;
        m++;
    }
    if (m == 0) { free(cand); return 0; }

    qsort(cand, (size_t)m, sizeof *cand, qw_cand_desc);

    if (sp->top_k > 0 && sp->top_k < m) m = sp->top_k;

    if (sp->top_p > 0.0f && sp->top_p < 1.0f) {
        /* Recompute the mass over what survived top-k, so top_p is a fraction
         * of the candidates actually in play. */
        double total = 0.0;
        for (int32_t i = 0; i < m; i++) total += cand[i].p;
        double want = total * (double)sp->top_p, acc = 0.0;
        int32_t cut = m;
        for (int32_t i = 0; i < m; i++) {
            acc += cand[i].p;
            if (acc >= want) { cut = i + 1; break; }
        }
        m = cut;
    }

    double total = 0.0;
    for (int32_t i = 0; i < m; i++) total += cand[i].p;
    double target = (double)qw_rng_float(rng) * total;
    int32_t chosen = cand[m - 1].id;
    for (int32_t i = 0; i < m; i++) {
        target -= cand[i].p;
        if (target <= 0.0) { chosen = cand[i].id; break; }
    }
    free(cand);
    return chosen;
}
