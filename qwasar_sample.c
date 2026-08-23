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

/* The filter chain, factored out because the sampled verify needs the same
 * pruned candidate set three ways: to sample from, to read one token's
 * probability out of, and to sample from with a token removed.  Fills `*out`
 * with candidates carrying unnormalised weights, ordered descending, and
 * returns how many survived -- or -1 when the scratch allocation failed, in
 * which case callers fall back to argmax exactly as qwasar_sample always has. */
static int32_t qw_filter(const float *logits, int32_t n, const qwasar_sampling *sp,
                         float max_logit, qw_cand **out) {
    const float inv_t = 1.0f / sp->temperature;
    /* Prune before ordering.  exp(x - max) is 1 at the mode, so a candidate is
     * kept when its unnormalised weight clears min_p; that is the same test as
     * against the normalised max probability, without needing the sum yet. */
    const float keep = sp->min_p > 0.0f ? sp->min_p : 0.0f;
    qw_cand *cand = malloc((size_t)n * sizeof *cand);
    *out = cand;
    if (!cand) return -1;

    int32_t m = 0;
    for (int32_t i = 0; i < n; i++) {
        float w = (float)exp((double)(logits[i] - max_logit) * inv_t);
        if (w < keep) continue;
        cand[m].id = i;
        cand[m].p = w;
        m++;
    }
    if (m == 0) return 0;

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
    return m;
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

    qw_cand *cand;
    int32_t m = qw_filter(logits, n, sp, max_logit, &cand);
    if (m < 0) {
        int32_t best = 0;
        for (int32_t i = 1; i < n; i++) if (logits[i] > logits[best]) best = i;
        return best;
    }
    if (m == 0) { free(cand); return 0; }

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

float qwasar_rng_uniform(uint64_t *state) { return qw_rng_float(state); }

/* Whether any filter is on, given `n` live logits.  Mirrors the test in
 * qwasar_sample so the streaming no-filter path stays shared behaviour. */
static bool qw_filtering(const qwasar_sampling *sp, int32_t n) {
    return (sp->top_k > 0 && sp->top_k < n)
        || (sp->top_p > 0.0f && sp->top_p < 1.0f)
        || (sp->min_p > 0.0f);
}

float qwasar_sample_prob(const float *logits, int32_t n,
                         const qwasar_sampling *sp, int32_t token) {
    if (n <= 0 || token < 0 || token >= n) return 0.0f;
    if (!sp || sp->temperature <= 0.0f) {
        /* Greedy is a point mass at the first argmax -- the same tie-break
         * qwasar_sample uses, so the two agree on which token has mass 1. */
        int32_t best = 0;
        for (int32_t i = 1; i < n; i++) if (logits[i] > logits[best]) best = i;
        return token == best ? 1.0f : 0.0f;
    }

    float max_logit = logits[0];
    for (int32_t i = 1; i < n; i++) if (logits[i] > max_logit) max_logit = logits[i];
    const float inv_t = 1.0f / sp->temperature;

    if (!qw_filtering(sp, n)) {
        double sum = 0.0;
        for (int32_t i = 0; i < n; i++)
            sum += exp((double)(logits[i] - max_logit) * inv_t);
        return sum > 0.0
            ? (float)(exp((double)(logits[token] - max_logit) * inv_t) / sum)
            : 0.0f;
    }

    qw_cand *cand;
    int32_t m = qw_filter(logits, n, sp, max_logit, &cand);
    if (m <= 0) { if (m == 0) free(cand); return 0.0f; }
    double total = 0.0, w = 0.0;
    for (int32_t i = 0; i < m; i++) {
        total += cand[i].p;
        if (cand[i].id == token) w = cand[i].p;
    }
    free(cand);
    return total > 0.0 ? (float)(w / total) : 0.0f;
}

int32_t qwasar_sample_excluding(const float *logits, int32_t n,
                                const qwasar_sampling *sp, uint64_t *rng,
                                int32_t banned) {
    if (n <= 0) return 0;
    if (!sp || sp->temperature <= 0.0f) {
        int32_t best = -1;
        for (int32_t i = 0; i < n; i++) {
            if (i == banned) continue;
            if (best < 0 || logits[i] > logits[best]) best = i;
        }
        return best >= 0 ? best : 0;
    }

    float max_logit = logits[0];
    for (int32_t i = 1; i < n; i++) if (logits[i] > max_logit) max_logit = logits[i];
    const float inv_t = 1.0f / sp->temperature;

    if (!qw_filtering(sp, n)) {
        double sum = 0.0;
        for (int32_t i = 0; i < n; i++) {
            if (i == banned) continue;
            sum += exp((double)(logits[i] - max_logit) * inv_t);
        }
        double target = (double)qw_rng_float(rng) * sum;
        int32_t last = banned == n - 1 ? n - 2 : n - 1;
        for (int32_t i = 0; i < n; i++) {
            if (i == banned) continue;
            target -= exp((double)(logits[i] - max_logit) * inv_t);
            if (target <= 0.0) return i;
        }
        return last >= 0 ? last : 0;
    }

    qw_cand *cand;
    int32_t m = qw_filter(logits, n, sp, max_logit, &cand);
    if (m < 0) return qwasar_sample_excluding(logits, n, NULL, rng, banned);
    double total = 0.0;
    int32_t live = 0;
    for (int32_t i = 0; i < m; i++) {
        if (cand[i].id == banned) continue;
        total += cand[i].p;
        live++;
    }
    if (live == 0 || total <= 0.0) {
        /* The filtered support was exactly the banned token.  Unreachable
         * from rejection (a token with mass 1 is never rejected), but a
         * caller can construct it; the least-wrong answer is the best token
         * the raw distribution has besides the ban. */
        free(cand);
        return qwasar_sample_excluding(logits, n, NULL, rng, banned);
    }
    double target = (double)qw_rng_float(rng) * total;
    int32_t chosen = -1;
    for (int32_t i = 0; i < m; i++) {
        if (cand[i].id == banned) continue;
        chosen = cand[i].id;
        target -= cand[i].p;
        if (target <= 0.0) break;
    }
    free(cand);
    return chosen;
}
