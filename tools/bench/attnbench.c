/* Decode attention alone: one query token against a synthetic cache at
 * several depths, 16 layers to a command buffer as the 27B has, the
 * per-query kernel (qw_op_attn_decode) against the split one
 * (qw_op_attn_decode_split).  Where they cross is QW_ATTN_SPLIT_MIN_KEYS.
 *
 *   tools/bench/attnbench */
#include "qwasar_gpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
static double now(void) { struct timeval tv; gettimeofday(&tv, NULL); return tv.tv_sec + tv.tv_usec * 1e-6; }
int main(void) {
    char err[256];
    if (!qw_gpu_init(err, sizeof err)) { fprintf(stderr, "%s\n", err); return 1; }
    const int qh = 24, kvh = 4, hd = 256, layers = 16;
    int depths[] = { 2048, 8192, 16384, 20480, 24576, 32768, 65536, 131072, 262144 };
    for (int di = 0; di < (int)(sizeof depths / sizeof *depths); di++) {
        int ctx = depths[di] + 8;
        size_t cn = (size_t)kvh * ctx * hd;
        qw_buf kc = qw_buf_alloc(cn * 2), vc = qw_buf_alloc(cn * 2);
        qw_buf q = qw_buf_alloc((size_t)qh * hd * 4), o = qw_buf_alloc((size_t)qh * hd * 4);
        unsigned short *k = qw_buf_contents(kc), *v = qw_buf_contents(vc);
        for (size_t i = 0; i < cn; i++) { k[i] = 0x2c00 + (i * 7 % 512); v[i] = 0x2c00 + (i * 13 % 512); }
        float *qp = qw_buf_contents(q);
        for (int i = 0; i < qh * hd; i++) qp[i] = (float)((i * 37) % 100) / 100.0f - 0.5f;
        for (int kern = 0; kern < 3; kern++) {
            double best = 1e9;
            for (int it = 0; it < 6; it++) {
                qw_cmd c = qw_cmd_begin();
                double t0 = now();
                for (int l = 0; l < layers; l++) {
                    if (kern == 0) qw_op_attn_decode(c, qw_ref_at(o, 0), qw_ref_at(q, 0), qw_ref_at(kc, 0), qw_ref_at(vc, 0),
                                                     1, qh, kvh, hd, ctx, depths[di], 0.0625f);
                    else if (kern == 2) qw_op_attn_decode_tensor(c, qw_ref_at(o, 0), qw_ref_at(q, 0), qw_ref_at(kc, 0), qw_ref_at(vc, 0),
                                                 1, qh, kvh, hd, ctx, depths[di], 0.0625f);
                    else qw_op_attn_decode_split(c, qw_ref_at(o, 0), qw_ref_at(q, 0), qw_ref_at(kc, 0), qw_ref_at(vc, 0),
                                                 1, qh, kvh, hd, ctx, depths[di], 0.0625f);
                }
                qw_cmd_wait(c);
                double t = now() - t0;
                qw_cmd_free(c);
                if (it && t < best) best = t;
            }
            double gb = (double)depths[di] * kvh * hd * 2 * 2 * layers / 1e9;
            printf("depth %6d  %-10s %7.2f ms for 16 layers  (%.0f GB/s of unique KV)\n", depths[di],
                   kern == 2 ? "tensor" : kern ? "split" : "per-query", best * 1e3, gb / best);
        }
        qw_buf_free(kc); qw_buf_free(vc); qw_buf_free(q); qw_buf_free(o);
    }
    return 0;
}
