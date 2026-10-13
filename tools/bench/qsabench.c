/* Decode's sparse attention alone, at Flash-Next's shape: one token, 24
 * query heads on 2 kv heads, 256 wide, over a selection list of the budget's
 * 2,048 positions plus the tail, spread through a cache of the given depth;
 * 12 layers a token, as the model has, and 20 tokens to a command buffer so
 * that submission and the clock's ramp are not what is timed.  The op is the
 * layer's, qw_op_attn_masked_gated -- the attention and its output gate --
 * and the kernels whichever it takes (QWASAR_ATTN_MASKED=simd for the
 * simdgroup one).
 *
 *   tools/bench/qsabench [depth ...]   default 8192 32768 131072 */
#include "qwasar_gpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/time.h>
static double now(void) { struct timeval tv; gettimeofday(&tv, NULL); return tv.tv_sec + tv.tv_usec * 1e-6; }
int main(int argc, char **argv) {
    char err[256];
    if (!qw_gpu_init(err, sizeof err)) { fprintf(stderr, "%s\n", err); return 1; }
    const int qh = 24, kvh = 2, hd = 256, layers = 12, topk = 512, ratio = 4;
    const int cap = qw_qsa_list_cap(topk, ratio);
    int depths[16] = { 8192, 32768, 131072 }, nd = 3;
    if (argc > 1) { nd = 0; for (int i = 1; i < argc && nd < 16; i++) depths[nd++] = atoi(argv[i]); }
    for (int di = 0; di < nd; di++) {
        const int depth = depths[di], ctx = depth + 8;
        const size_t cn = (size_t)kvh * ctx * hd;
        qw_buf kc = qw_buf_alloc(cn * 2), vc = qw_buf_alloc(cn * 2), lb = qw_buf_alloc((size_t)(1 + cap) * 4);
        qw_buf q = qw_buf_alloc((size_t)qh * hd * 4), o = qw_buf_alloc((size_t)qh * hd * 4),
               g = qw_buf_alloc((size_t)qh * hd * 4);
        unsigned short *k = qw_buf_contents(kc), *v = qw_buf_contents(vc);
        for (size_t i = 0; i < cn; i++) { k[i] = 0x2c00 + (i * 7 % 512); v[i] = 0x2c00 + (i * 13 % 512); }
        float *qp = qw_buf_contents(q);
        for (int i = 0; i < qh * hd; i++) qp[i] = (float)((i * 37) % 100) / 100.0f - 0.5f;
        /* topk blocks of ratio keys, evenly spread, then the tail */
        uint32_t *l = qw_buf_contents(lb);
        int n = 0;
        for (int b = 0; b < topk; b++)
            for (int t = 0; t < ratio; t++) l[1 + n++] = (uint32_t)((size_t)b * (depth / ratio) / topk * ratio + t);
        for (int t = depth - 3; t <= depth; t++) l[1 + n++] = (uint32_t)t;
        l[0] = (uint32_t)n;
        double best = 1e9;
        for (int it = 0; it < 8; it++) {
            qw_cmd c = qw_cmd_begin();
            double t0 = now();
            for (int L = 0; L < layers * 20; L++)
                qw_op_attn_masked_gated(c, qw_ref_at(o, 0), qw_ref_at(g, 0), qw_ref_at(q, 0), qw_ref_at(kc, 0),
                                        qw_ref_at(vc, 0), qw_ref_at(lb, 0), 1, qh, kvh, hd, ctx, depth, 0.0625f,
                                        topk, ratio);
            qw_cmd_wait(c);
            double t = now() - t0;
            qw_cmd_free(c);
            if (it && t < best) best = t;
        }
        printf("depth %7d, %d keys: %6.3f ms a token (12 layers)\n", depth, n, best * 1e3 / 20);
        qw_buf_free(kc); qw_buf_free(vc); qw_buf_free(lb); qw_buf_free(q); qw_buf_free(o); qw_buf_free(g);
    }
    return 0;
}
