/* Prefill attention alone: one 256-row chunk (qw_op_attn_prefill) at several
 * depths, 16 layers to a command buffer as the 27B has, with the TFLOP/s it
 * sustains -- the number QW_FA_BQ/BK/SG were tuned on.
 *
 *   tools/bench/fabench */
#include "qwasar_gpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
static double now(void) { struct timeval tv; gettimeofday(&tv, NULL); return tv.tv_sec + tv.tv_usec * 1e-6; }
int main(void) {
    char err[256];
    if (!qw_gpu_init(err, sizeof err)) { fprintf(stderr, "%s\n", err); return 1; }
    const int qh = 24, kvh = 4, hd = 256, layers = 16, rows = 256;
    int depths[] = { 0, 4096, 16384, 32768 };
    for (int di = 0; di < 4; di++) {
        int base = depths[di], ctx = base + rows + 8;
        size_t cn = (size_t)kvh * ctx * hd;
        qw_buf kc = qw_buf_alloc(cn * 2), vc = qw_buf_alloc(cn * 2);
        qw_buf q = qw_buf_alloc((size_t)rows * qh * hd * 4), o = qw_buf_alloc((size_t)rows * qh * hd * 4);
        unsigned short *k = qw_buf_contents(kc), *v = qw_buf_contents(vc);
        for (size_t i = 0; i < cn; i++) { k[i] = 0x2c00 + (i * 7 % 512); v[i] = 0x2c00 + (i * 13 % 512); }
        float *qp = qw_buf_contents(q);
        for (size_t i = 0; i < (size_t)rows * qh * hd; i++) qp[i] = (float)((i * 37) % 100) / 100.0f - 0.5f;
        double best = 1e9;
        for (int it = 0; it < 4; it++) {
            qw_cmd c = qw_cmd_begin();
            double t0 = now();
            for (int l = 0; l < layers; l++)
                qw_op_attn_prefill(c, qw_ref_at(o, 0), qw_ref_at(q, 0), qw_ref_at(kc, 0), qw_ref_at(vc, 0),
                                   rows, qh, kvh, hd, ctx, base, 0.0625f);
            qw_cmd_wait(c);
            double t = now() - t0;
            qw_cmd_free(c);
            if (it && t < best) best = t;
        }
        double keys = base + rows / 2.0;   /* average keys a row sees */
        double flop = 4.0 * rows * qh * keys * hd * layers;
        printf("depth %6d: %8.1f ms per chunk (16 layers), %.2f TFLOP/s\n", base, best * 1e3, flop / best / 1e12);
        qw_buf_free(kc); qw_buf_free(vc); qw_buf_free(q); qw_buf_free(o);
    }
    return 0;
}
