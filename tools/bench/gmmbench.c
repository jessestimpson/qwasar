/* Prefill's grouped experts alone, on Flash-Next's shapes: 512 experts, top
 * 10, the gate/up projections (2560 -> 640) and the down (640 -> 2560), a
 * chunk's pairs routed uniformly at random.  Times qw_op_moe_group and the
 * three grouped matmuls of one layer, the simdgroup kernel (QWASAR_GMM=simd)
 * against the tensor-op one; the TFLOP/s counts only the useful rows.
 *
 *   tools/bench/gmmbench [rows ...]   default 256 1024 2048 */
#include "qwasar_gpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sys/time.h>
static double now(void) { struct timeval tv; gettimeofday(&tv, NULL); return tv.tv_sec + tv.tv_usec * 1e-6; }

typedef struct { qw_buf w, s, b; } bank;
static bank mkbank(int32_t E, int32_t n, int32_t k, int32_t G, uint32_t seed) {
    bank q = { qw_buf_alloc((size_t)E * n * k / 2), qw_buf_alloc((size_t)E * n * (k / G) * 2),
               qw_buf_alloc((size_t)E * n * (k / G) * 2) };
    uint32_t *w = qw_buf_contents(q.w);
    uint16_t *s = qw_buf_contents(q.s), *b = qw_buf_contents(q.b);
    for (size_t i = 0; i < (size_t)E * n * k / 8; i++) { seed = seed * 1664525u + 1013904223u; w[i] = seed; }
    for (size_t i = 0; i < (size_t)E * n * (k / G); i++) { s[i] = 0x3c00 >> 1; b[i] = 0xbc00 >> 1; }
    return q;
}

int main(int argc, char **argv) {
    char err[256];
    if (!qw_gpu_init(err, sizeof err)) { fprintf(stderr, "%s\n", err); return 1; }
    const int32_t E = 512, K = 10, H = 2560, I = 640, G = 32;
    bank gate = mkbank(E, I, H, G, 1), up = mkbank(E, I, H, G, 2), down = mkbank(E, H, I, G, 3);
    int32_t rows_list[16] = { 256, 1024, 2048 }, nr = 3;
    if (argc > 1) { nr = 0; for (int i = 1; i < argc && nr < 16; i++) rows_list[nr++] = atoi(argv[i]); }
    for (int ri = 0; ri < nr; ri++) {
        const int32_t rows = rows_list[ri], pairs = rows * K;
        qw_buf x = qw_buf_alloc((size_t)rows * H * 4), idx = qw_buf_alloc((size_t)pairs * 4),
               perm = qw_buf_alloc((size_t)pairs * 4),
               tiles = qw_buf_alloc((size_t)(1 + 3 * qw_moe_max_tiles(pairs, E)) * 4),
               cur = qw_buf_alloc((size_t)E * 4), g = qw_buf_alloc((size_t)pairs * I * 4),
               u = qw_buf_alloc((size_t)pairs * I * 4), y = qw_buf_alloc((size_t)pairs * H * 4);
        float *xp = qw_buf_contents(x);
        for (size_t i = 0; i < (size_t)rows * H; i++) xp[i] = (float)((i * 37) % 100) / 100.0f - 0.5f;
        int32_t *ip = qw_buf_contents(idx);
        uint32_t st = 7;
        for (int32_t t = 0; t < rows; t++)   /* K distinct experts a token */
            for (int32_t j = 0; j < K; j++) {
                int32_t e, dup;
                do {
                    st = st * 1664525u + 1013904223u; e = (int32_t)((st >> 8) % E); dup = 0;
                    for (int32_t q = 0; q < j; q++) dup |= ip[t * K + q] == e;
                } while (dup);
                ip[t * K + j] = e;
            }
        double best = 1e9;
        for (int it = 0; it < 6; it++) {
            qw_cmd c = qw_cmd_begin();
            double t0 = now();
            qw_op_moe_group(c, qw_ref_at(perm, 0), qw_ref_at(tiles, 0), qw_ref_at(cur, 0), qw_ref_at(idx, 0), pairs, E);
            qw_op_qmm_q4_gather(c, qw_ref_at(g, 0), qw_ref_at(x, 0), qw_ref_at(perm, 0), qw_ref_at(tiles, 0),
                                qw_ref_at(gate.w, 0), qw_ref_at(gate.s, 0), qw_ref_at(gate.b, 0), H, I, pairs, E, K, false, G);
            qw_op_qmm_q4_gather(c, qw_ref_at(u, 0), qw_ref_at(x, 0), qw_ref_at(perm, 0), qw_ref_at(tiles, 0),
                                qw_ref_at(up.w, 0), qw_ref_at(up.s, 0), qw_ref_at(up.b, 0), H, I, pairs, E, K, false, G);
            qw_op_qmm_q4_gather(c, qw_ref_at(y, 0), qw_ref_at(g, 0), qw_ref_at(perm, 0), qw_ref_at(tiles, 0),
                                qw_ref_at(down.w, 0), qw_ref_at(down.s, 0), qw_ref_at(down.b, 0), I, H, pairs, E, K, true, G);
            qw_cmd_wait(c);
            double t = now() - t0;
            const char *e2 = qw_cmd_error(c);
            if (e2) { fprintf(stderr, "GPU error: %s\n", e2); return 1; }
            qw_cmd_free(c);
            if (it && t < best) best = t;
        }
        const double flop = 2.0 * pairs * 3.0 * H * I;
        printf("rows %5d (%6d pairs): %7.3f ms a layer, %5.2f TFLOP/s, x48 layers %6.1f ms\n",
               rows, pairs, best * 1e3, flop / best / 1e12, best * 48e3);
        qw_buf_free(x); qw_buf_free(idx); qw_buf_free(perm); qw_buf_free(tiles); qw_buf_free(cur);
        qw_buf_free(g); qw_buf_free(u); qw_buf_free(y);
    }
    return 0;
}
