/* Time to first token and decode, against mlx-vlm (tools/bench/mlx_ttft.py
 * is its twin; docs/benchmarks.md has the comparison).  Loads the model,
 * warms up on a session of its own -- 2,600 tokens, past Flash-Next's
 * sparse-attention budget, and a few decode steps, so no kernel is compiled
 * in the timed part -- then a fresh session, with no checkpoint and no shared
 * prefix: TTFT is the prefill of the first N ids through the argmax of the
 * first token; then D greedy decode steps, end-of-text ignored.  One JSON
 * line on stdout.  The ids file is whitespace-separated token ids, the same
 * file for both engines.
 *
 *   tools/bench/ttftbench <model> <ids-file> <N> <D> */
#include "qwasar.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
static double now(void) { struct timeval tv; gettimeofday(&tv, NULL); return tv.tv_sec + tv.tv_usec * 1e-6; }
static int32_t argmax(const float *l, int32_t v) { int32_t b = 0; for (int32_t i = 1; i < v; i++) if (l[i] > l[b]) b = i; return b; }
int main(int argc, char **argv) {
    if (argc < 5) return 2;
    const int32_t N = atoi(argv[3]), D = atoi(argv[4]);
    FILE *f = fopen(argv[2], "r"); int32_t *ids = malloc(sizeof *ids * 200000); int32_t n = 0;
    while (n < 200000 && fscanf(f, "%d", &ids[n]) == 1) n++;
    fclose(f);
    if (n < N || n < 2600) { fprintf(stderr, "ids file too short\n"); return 2; }
    char err[512];
    const double t0 = now();
    qwasar_options o = { .model_path = argv[1], .context_size = N + D + 64 > 4096 ? N + D + 64 : 4096 };
    qwasar_engine *e = qwasar_engine_load(&o, err, sizeof err);
    if (!e) { fprintf(stderr, "%s\n", err); return 3; }
    const double load_s = now() - t0;
    const int32_t V = qwasar_vocab_size(e);
    /* warm-up: past the sparse-attention budget, and a few decode steps */
    qwasar_session *w = qwasar_session_new(e, err, sizeof err);
    const float *l = qwasar_session_eval(w, ids + 100000 % (n - 2600), 2600, err, sizeof err);
    for (int i = 0; i < 4 && l; i++) { int32_t t = argmax(l, V); l = qwasar_session_eval(w, &t, 1, err, sizeof err); }
    qwasar_session_free(w);
    /* timed, from a fresh session */
    qwasar_session *s = qwasar_session_new(e, err, sizeof err);
    const double a = now();
    l = qwasar_session_eval(s, ids, N, err, sizeof err);
    if (!l) { fprintf(stderr, "%s\n", err); return 4; }
    int32_t t = argmax(l, V);
    const double ttft = now() - a;
    int32_t *out = malloc(sizeof *out * (D + 1)); out[0] = t;
    const double b = now();
    for (int32_t i = 0; i < D; i++) {
        l = qwasar_session_eval(s, &t, 1, err, sizeof err);
        if (!l) { fprintf(stderr, "%s\n", err); return 5; }
        t = argmax(l, V); out[i + 1] = t;
    }
    const double dec = now() - b;
    printf("{\"engine\": \"qwasar\", \"n\": %d, \"d\": %d, \"load_s\": %.2f, \"ttft_s\": %.3f, "
           "\"prefill_tps\": %.1f, \"decode_tps\": %.2f, \"first\": [", N, D, load_s, ttft, N / ttft, D / dec);
    for (int i = 0; i < 16; i++) printf("%s%d", i ? ", " : "", out[i]);
    printf("]}\n");
    return 0;
}
