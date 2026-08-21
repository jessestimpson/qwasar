/* Sampling: the filter chain and its edges.
 *
 * Runs without a model, so it can afford to check distributions statistically
 * rather than just spot-checking a few draws. */

#include "qwasar.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;

#define CHECK(cond, ...) do { \
    if (!(cond)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
                   fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); fails++; } \
} while (0)

/* A distribution with a clear ordering and a long tail. */
static void make_logits(float *v, int32_t n) {
    for (int32_t i = 0; i < n; i++) v[i] = -0.02f * (float)i;
    v[7] = 5.0f;      /* mode */
    v[3] = 4.0f;
    v[9] = 3.0f;
    v[1] = 2.0f;
}

static void test_greedy(void) {
    const int32_t n = 5000;
    float *v = malloc((size_t)n * sizeof *v);
    make_logits(v, n);

    qwasar_sampling sp;
    qwasar_sampling_defaults(&sp);
    sp.temperature = 0.0f;

    uint64_t rng = 12345;
    for (int i = 0; i < 8; i++)
        CHECK(qwasar_sample(v, n, &sp, &rng) == 7, "greedy must always pick the mode");

    /* Greedy must not depend on the generator at all. */
    uint64_t a = 1, b = 999999;
    CHECK(qwasar_sample(v, n, &sp, &a) == qwasar_sample(v, n, &sp, &b),
          "greedy must not consume randomness");
    free(v);
    printf("  greedy                  picks the mode, ignores the seed\n");
}

static void test_top_k(void) {
    const int32_t n = 5000;
    float *v = malloc((size_t)n * sizeof *v);
    make_logits(v, n);

    qwasar_sampling sp;
    qwasar_sampling_defaults(&sp);
    sp.top_k = 2;
    sp.top_p = 1.0f;
    sp.min_p = 0.0f;

    uint64_t rng = 7;
    int seen[5000] = { 0 };
    for (int i = 0; i < 4000; i++) seen[qwasar_sample(v, n, &sp, &rng)]++;

    CHECK(seen[7] > 0 && seen[3] > 0, "both of the top 2 should appear");
    int others = 0;
    for (int32_t i = 0; i < n; i++) if (i != 7 && i != 3) others += seen[i];
    CHECK(others == 0, "top_k=2 admitted %d draws outside the top 2", others);
    /* The mode is e^1 times more likely than the runner-up. */
    double ratio = (double)seen[7] / (double)seen[3];
    CHECK(ratio > 2.0 && ratio < 3.5, "top-2 ratio %.2f, expected about e", ratio);
    printf("  top_k=2                 only ids 7 and 3, ratio %.2f (e = 2.72)\n", ratio);
    free(v);
}

static void test_top_p(void) {
    const int32_t n = 5000;
    float *v = malloc((size_t)n * sizeof *v);
    make_logits(v, n);

    qwasar_sampling sp;
    qwasar_sampling_defaults(&sp);
    sp.top_k = 0;
    sp.min_p = 0.0f;
    sp.top_p = 0.5f;   /* the mode alone carries more than half the mass here */

    uint64_t rng = 99;
    int outside = 0;
    for (int i = 0; i < 2000; i++) if (qwasar_sample(v, n, &sp, &rng) != 7) outside++;
    CHECK(outside == 0, "top_p=0.5 drew outside the nucleus %d times", outside);
    printf("  top_p=0.5               nucleus is the mode alone\n");
    free(v);
}

static void test_min_p(void) {
    const int32_t n = 5000;
    float *v = malloc((size_t)n * sizeof *v);
    make_logits(v, n);

    qwasar_sampling sp;
    qwasar_sampling_defaults(&sp);
    sp.top_k = 0;
    sp.top_p = 1.0f;
    sp.min_p = 0.1f;   /* keeps tokens within 10x of the mode */

    uint64_t rng = 4242;
    int seen[5000] = { 0 };
    for (int i = 0; i < 4000; i++) seen[qwasar_sample(v, n, &sp, &rng)]++;

    /* exp(5-5)=1, exp(4-5)=0.37, exp(3-5)=0.14 all clear 0.1; exp(2-5)=0.05
     * does not, and neither does anything in the tail. */
    CHECK(seen[7] && seen[3] && seen[9], "ids 7, 3 and 9 clear min_p and should appear");
    CHECK(seen[1] == 0, "id 1 is below min_p and must not be drawn");
    int others = 0;
    for (int32_t i = 0; i < n; i++)
        if (i != 7 && i != 3 && i != 9) others += seen[i];
    CHECK(others == 0, "min_p admitted %d draws it should have cut", others);
    printf("  min_p=0.1               keeps 7, 3, 9; cuts 1 and the tail\n");
    free(v);
}

static void test_unfiltered_is_still_the_right_law(void) {
    /* With no filters the sampler skips sorting entirely and walks the
     * unsorted distribution.  That shortcut has to produce the same law. */
    const int32_t n = 4;
    float v[4];
    v[0] = logf(0.1f); v[1] = logf(0.2f); v[2] = logf(0.3f); v[3] = logf(0.4f);

    qwasar_sampling sp = { .temperature = 1.0f, .top_k = 0, .top_p = 1.0f,
                           .min_p = 0.0f, .seed = 0 };
    uint64_t rng = 2024;
    int seen[4] = { 0 };
    const int draws = 200000;
    for (int i = 0; i < draws; i++) seen[qwasar_sample(v, n, &sp, &rng)]++;

    const float want[4] = { 0.1f, 0.2f, 0.3f, 0.4f };
    double worst = 0.0;
    for (int i = 0; i < 4; i++) {
        double got = (double)seen[i] / draws;
        double d = fabs(got - want[i]);
        if (d > worst) worst = d;
    }
    CHECK(worst < 0.01, "unfiltered draws deviate from the distribution by %.4f", worst);
    printf("  unfiltered              matches 0.1/0.2/0.3/0.4 within %.4f\n", worst);
}

static void test_seed_reproducibility(void) {
    const int32_t n = 5000;
    float *v = malloc((size_t)n * sizeof *v);
    make_logits(v, n);
    qwasar_sampling sp;
    qwasar_sampling_defaults(&sp);

    uint64_t a = 555, b = 555;
    bool same = true;
    for (int i = 0; i < 200; i++)
        if (qwasar_sample(v, n, &sp, &a) != qwasar_sample(v, n, &sp, &b)) same = false;
    CHECK(same, "the same seed must produce the same sequence");

    uint64_t c = 556;
    int diff = 0;
    a = 555;
    for (int i = 0; i < 200; i++)
        if (qwasar_sample(v, n, &sp, &a) != qwasar_sample(v, n, &sp, &c)) diff++;
    CHECK(diff > 0, "different seeds should diverge");
    printf("  seeding                 reproducible, and distinct seeds differ\n");
    free(v);
}

static void test_edges(void) {
    float one = 1.0f;
    qwasar_sampling sp;
    qwasar_sampling_defaults(&sp);
    uint64_t rng = 1;
    CHECK(qwasar_sample(&one, 1, &sp, &rng) == 0, "a single-token vocabulary");

    float v[3] = { 1.0f, 1.0f, 1.0f };
    sp.top_k = 100;              /* larger than the vocabulary */
    CHECK(qwasar_sample(v, 3, &sp, &rng) < 3, "top_k larger than n must be safe");

    sp.top_k = 0; sp.top_p = 0.0f;   /* degenerate nucleus */
    CHECK(qwasar_sample(v, 3, &sp, &rng) < 3, "top_p = 0 must still return a token");

    sp.top_p = 1.0f; sp.min_p = 2.0f;  /* threshold above any probability */
    CHECK(qwasar_sample(v, 3, &sp, &rng) < 3, "an impossible min_p must still return");
    printf("  edges                   n=1, top_k>n, top_p=0, min_p>1\n");
}

int main(void) {
    test_greedy();
    test_top_k();
    test_top_p();
    test_min_p();
    test_unfiltered_is_still_the_right_law();
    test_seed_reproducibility();
    test_edges();
    if (fails) { fprintf(stderr, "%d check(s) failed\n", fails); return 1; }
    printf("ok: sample\n");
    return 0;
}
