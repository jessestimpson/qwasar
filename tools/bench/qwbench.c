/* qwbench: one sweep point.  Loads the model, prefills exactly N tokens of
 * real text (from a corpus file, repeated as needed), timing every chunk, then
 * decodes exactly D tokens greedily at that depth, ignoring end-of-text.
 * Prints one JSON object on stdout.
 *
 * The JSON carries every chunk's completion time (so one long prompt gives the
 * prefill rate at every depth) and every decode token's time.
 *
 *   tools/bench/qwbench <model-dir> <corpus> <N> <D> */
#include "qwasar.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

static double now(void) { struct timeval tv; gettimeofday(&tv, NULL); return tv.tv_sec + tv.tv_usec * 1e-6; }

typedef struct { int32_t n; int32_t *done; double *t; } prog;
static void on_prog(void *ud, int32_t done, int32_t total) {
    (void)total;
    prog *p = ud;
    p->done[p->n] = done; p->t[p->n] = now(); p->n++;
}

int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: qwbench <model> <corpus> <N> <D>\n"); return 2; }
    const char *model = argv[1], *corpus = argv[2];
    const int32_t N = atoi(argv[3]), D = atoi(argv[4]);
    char err[512];

    FILE *f = fopen(corpus, "rb");
    if (!f) { perror(corpus); return 1; }
    fseek(f, 0, SEEK_END); long len = ftell(f); fseek(f, 0, SEEK_SET);
    char *text = malloc(len + 1);
    if (fread(text, 1, len, f) != (size_t)len) { fprintf(stderr, "short read\n"); return 1; }
    text[len] = 0; fclose(f);
    qwasar_tokenizer *tok = qwasar_tokenizer_load(model, err, sizeof err);
    if (!tok) { fprintf(stderr, "tokenizer: %s\n", err); return 1; }
    int32_t nc = 0;
    int32_t *ct = qwasar_encode(tok, text, &nc);
    if (!ct || nc < 1) { fprintf(stderr, "encode failed\n"); return 1; }
    int32_t *ids = malloc(sizeof *ids * N);
    for (int32_t i = 0; i < N; i++) ids[i] = ct[i % nc];

    const double t_load0 = now();
    qwasar_options opts = { .model_path = model, .context_size = N + D + 64 };
    qwasar_engine *e = qwasar_engine_load(&opts, err, sizeof err);
    if (!e) { printf("{\"n\": %d, \"error\": \"load: %s\"}\n", N, err); return 3; }
    qwasar_session *s = qwasar_session_new(e, err, sizeof err);
    if (!s) { printf("{\"n\": %d, \"error\": \"session: %s\"}\n", N, err); return 3; }
    const double load_s = now() - t_load0;
    const int32_t vocab = qwasar_vocab_size(e);

    prog p = { 0, malloc(sizeof(int32_t) * (N / 16 + 8)), malloc(sizeof(double) * (N / 16 + 8)) };
    qwasar_session_set_progress(s, on_prog, &p);
    const double t_pf0 = now();
    const float *logits = qwasar_session_eval(s, ids, N, err, sizeof err);
    const double t_pf1 = now();
    if (!logits) { printf("{\"n\": %d, \"error\": \"prefill: %s\"}\n", N, err); return 4; }
    qwasar_session_set_progress(s, NULL, NULL);

    double *dt = malloc(sizeof(double) * D);
    const double t_dc0 = now();
    for (int32_t i = 0; i < D; i++) {
        int32_t next = 0;
        for (int32_t v = 1; v < vocab; v++) if (logits[v] > logits[next]) next = v;
        const double a = now();
        logits = qwasar_session_eval(s, &next, 1, err, sizeof err);
        dt[i] = now() - a;
        if (!logits) { printf("{\"n\": %d, \"error\": \"decode: %s\"}\n", N, err); return 5; }
    }
    const double t_dc1 = now();

    printf("{\"n\": %d, \"d\": %d, \"load_s\": %.3f, \"t_prefill\": [%.3f, %.3f], \"t_decode\": [%.3f, %.3f], "
           "\"prefill_tps\": %.3f, \"decode_tps\": %.3f, \"chunks\": [",
           N, D, load_s, t_pf0, t_pf1, t_dc0, t_dc1, N / (t_pf1 - t_pf0), D / (t_dc1 - t_dc0));
    for (int32_t i = 0; i < p.n; i++)
        printf("%s[%d, %.3f]", i ? ", " : "", p.done[i], p.t[i] - t_pf0);
    printf("], \"decode_ms\": [");
    for (int32_t i = 0; i < D; i++) printf("%s%.1f", i ? ", " : "", dt[i] * 1e3);
    printf("]}\n");
    qwasar_session_free(s);
    qwasar_engine_free(e);
    return 0;
}
