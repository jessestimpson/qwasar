/* Metal runtime for qwasar.
 *
 * Objective-C is confined to this file.  The kernel sources are embedded in the
 * binary as one string (see tools/bin2c.c and the generated
 * qwasar_metal_src.inc) and compiled at startup with newLibraryWithSource:.
 *
 * That choice is deliberate: compiling a .metallib at build time would require
 * the Metal toolchain, which is a separate multi-gigabyte Xcode component that
 * is not installed on a stock macOS box.  The in-framework compiler that
 * newLibraryWithSource: uses is always present.  We pay for it once per source
 * change by caching the resulting pipeline binaries in an MTLBinaryArchive
 * under ~/.cache/qwasar, keyed by a hash of the source. */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <CommonCrypto/CommonDigest.h>

#include "qwasar_gpu.h"

#include "qwasar_metal_src.inc"   /* static const char qwasar_metal_source[] */

static id<MTLDevice>       g_device;
static id<MTLCommandQueue> g_queue;
static id<MTLLibrary>      g_library;
static id<MTLBinaryArchive> g_archive;
static NSURL              *g_archive_url;
static NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *g_pipelines;
static char                g_device_name[128];
static bool                g_archive_dirty;

/* ---- init ----------------------------------------------------------------- */

static NSString *qw_source_hash(void) {
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(qwasar_metal_source, (CC_LONG)(sizeof qwasar_metal_source - 1), digest);
    char hex[2 * CC_SHA256_DIGEST_LENGTH + 1];
    for (int i = 0; i < CC_SHA256_DIGEST_LENGTH; i++) snprintf(hex + 2 * i, 3, "%02x", digest[i]);
    return [NSString stringWithUTF8String:hex];
}

/* ~/.cache/qwasar/<source-hash>.bin, created on demand.  Returns nil if the
 * cache directory cannot be made -- caching is an optimisation, never a
 * requirement. */
static NSURL *qw_archive_url(NSString *hash) {
    NSString *home = NSHomeDirectory();
    if (!home) return nil;
    NSString *dir = [home stringByAppendingPathComponent:@".cache/qwasar"];
    NSError *err = nil;
    if (![[NSFileManager defaultManager] createDirectoryAtPath:dir
                                  withIntermediateDirectories:YES
                                                   attributes:nil
                                                        error:&err]) return nil;
    return [NSURL fileURLWithPath:
        [dir stringByAppendingPathComponent:[hash stringByAppendingString:@".bin"]]];
}

bool qw_gpu_init(char *err, size_t errcap) {
    if (g_device) return true;

    @autoreleasepool {
        g_device = MTLCreateSystemDefaultDevice();
        if (!g_device) {
            snprintf(err, errcap, "no Metal device available");
            return false;
        }
        snprintf(g_device_name, sizeof g_device_name, "%s", [[g_device name] UTF8String]);

        g_queue = [g_device newCommandQueue];
        if (!g_queue) {
            snprintf(err, errcap, "cannot create Metal command queue");
            g_device = nil;
            return false;
        }

        NSError *nserr = nil;
        NSString *source = [NSString stringWithUTF8String:qwasar_metal_source];
        MTLCompileOptions *opts = [MTLCompileOptions new];
        /* Tiling constants come from qwasar_gpu.h so the kernel and the host's
         * dispatch geometry are derived from the same numbers. */
        opts.preprocessorMacros = @{
            @"QW_QMM_BM" : @(QW_QMM_BM),
            @"QW_QMM_BN" : @(QW_QMM_BN),
            @"QW_QMM_BK" : @(QW_QMM_BK),
            @"QW_QMM_SG_M" : @(QW_QMM_SG_M),
            @"QW_QMM_SG_N" : @(QW_QMM_SG_N),
            @"QW_QMVB_B" : @(QW_QMVB_B),
            @"QW_QMVB_ROWS" : @(QW_QMVB_ROWS),
        };
        g_library = [g_device newLibraryWithSource:source options:opts error:&nserr];
        if (!g_library) {
            snprintf(err, errcap, "Metal shader compilation failed: %s",
                     [[nserr localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return false;
        }

        g_pipelines = [NSMutableDictionary new];

        /* Pipeline binary cache.  A miss is not an error: we build the archive
         * as pipelines are created and serialise it at shutdown. */
        NSString *hash = qw_source_hash();
        g_archive_url = qw_archive_url(hash);
        MTLBinaryArchiveDescriptor *desc = [MTLBinaryArchiveDescriptor new];
        if (g_archive_url && [[NSFileManager defaultManager]
                                 fileExistsAtPath:[g_archive_url path]])
            desc.url = g_archive_url;
        nserr = nil;
        g_archive = [g_device newBinaryArchiveWithDescriptor:desc error:&nserr];
        if (!g_archive && desc.url) {
            /* Stale or corrupt cache -- start a fresh one. */
            desc.url = nil;
            nserr = nil;
            g_archive = [g_device newBinaryArchiveWithDescriptor:desc error:&nserr];
        }
        g_archive_dirty = false;
    }
    return true;
}

void qw_gpu_shutdown(void) {
    @autoreleasepool {
        if (g_archive && g_archive_url && g_archive_dirty) {
            NSError *err = nil;
            [g_archive serializeToURL:g_archive_url error:&err];
        }
        g_pipelines   = nil;
        g_archive     = nil;
        g_archive_url = nil;
        g_library     = nil;
        g_queue       = nil;
        g_device      = nil;
    }
}

const char *qw_gpu_name(void) { return g_device_name; }

uint64_t qw_gpu_working_set_limit(void) {
    return g_device ? (uint64_t)[g_device recommendedMaxWorkingSetSize] : 0;
}

uint64_t qw_gpu_max_buffer_length(void) {
    return g_device ? (uint64_t)[g_device maxBufferLength] : 0;
}

/* ---- buffers --------------------------------------------------------------
 *
 * qw_buf is an id<MTLBuffer> retained in a CFBridging box.  ARC is on for this
 * file, so handing an object pointer across the C boundary means transferring
 * a +1 retain and releasing it in qw_buf_free. */

qw_buf qw_buf_wrap(void *ptr, size_t len) {
    if (!g_device || !ptr || !len) return NULL;
    /* No copy: `ptr` is an mmap of a safetensors shard, which stays mapped for
     * the engine's lifetime.  Metal requires page alignment here, which mmap
     * guarantees. */
    id<MTLBuffer> b = [g_device newBufferWithBytesNoCopy:ptr
                                                  length:len
                                                 options:MTLResourceStorageModeShared
                                             deallocator:nil];
    return (qw_buf)CFBridgingRetain(b);
}

qw_buf qw_buf_alloc(size_t len) {
    if (!g_device || !len) return NULL;
    id<MTLBuffer> b = [g_device newBufferWithLength:len
                                            options:MTLResourceStorageModeShared];
    return (qw_buf)CFBridgingRetain(b);
}

void qw_buf_free(qw_buf b) {
    if (b) CFBridgingRelease((CFTypeRef)b);
}

void *qw_buf_contents(qw_buf b) {
    if (!b) return NULL;
    return [(__bridge id<MTLBuffer>)b contents];
}

size_t qw_buf_length(qw_buf b) {
    if (!b) return 0;
    return (size_t)[(__bridge id<MTLBuffer>)b length];
}

/* ---- pipelines ------------------------------------------------------------ */

static id<MTLComputePipelineState> qw_pipeline(NSString *name) {
    id<MTLComputePipelineState> ps = g_pipelines[name];
    if (ps) return ps;

    id<MTLFunction> fn = [g_library newFunctionWithName:name];
    if (!fn) {
        fprintf(stderr, "qwasar: Metal function %s not found\n", [name UTF8String]);
        return nil;
    }

    MTLComputePipelineDescriptor *desc = [MTLComputePipelineDescriptor new];
    desc.computeFunction = fn;
    /* Offering the archive lets Metal reuse a cached binary instead of
     * recompiling; on a miss it just builds normally and we add it below. */
    if (g_archive) desc.binaryArchives = @[ g_archive ];

    NSError *err = nil;
    ps = [g_device newComputePipelineStateWithDescriptor:desc
                                                 options:MTLPipelineOptionNone
                                              reflection:nil
                                                   error:&err];
    if (!ps) {
        fprintf(stderr, "qwasar: pipeline %s failed: %s\n",
                [name UTF8String], [[err localizedDescription] UTF8String]);
        return nil;
    }
    if (g_archive) {
        NSError *aerr = nil;
        if ([g_archive addComputePipelineFunctionsWithDescriptor:desc error:&aerr])
            g_archive_dirty = true;
    }
    g_pipelines[name] = ps;
    return ps;
}

/* ---- command buffers ------------------------------------------------------
 *
 * One qw_cmd wraps a command buffer and the single compute encoder that every
 * op appends to.  Keeping one encoder for a whole forward pass avoids
 * per-kernel encoder setup and lets Metal overlap dispatches that do not
 * conflict. */

struct qw_cmd_s {
    void *cb;   /* id<MTLCommandBuffer>, +1 retained */
    void *enc;  /* id<MTLComputeCommandEncoder>, +1 retained */
    char  err[256];
};

qw_cmd qw_cmd_begin(void) {
    if (!g_queue) return NULL;
    qw_cmd c = calloc(1, sizeof *c);
    if (!c) return NULL;
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        c->cb  = (void *)CFBridgingRetain(cb);
        c->enc = (void *)CFBridgingRetain(enc);
    }
    return c;
}

static void qw_cmd_end_encoding(qw_cmd c) {
    if (!c->enc) return;
    [(__bridge id<MTLComputeCommandEncoder>)c->enc endEncoding];
    CFBridgingRelease((CFTypeRef)c->enc);
    c->enc = NULL;
}

void qw_cmd_commit(qw_cmd c) {
    if (!c || !c->cb) return;
    qw_cmd_end_encoding(c);
    [(__bridge id<MTLCommandBuffer>)c->cb commit];
}

void qw_cmd_wait(qw_cmd c) {
    if (!c || !c->cb) return;
    qw_cmd_end_encoding(c);
    @autoreleasepool {
        id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)c->cb;
        [cb commit];
        [cb waitUntilCompleted];
        if ([cb error])
            snprintf(c->err, sizeof c->err, "%s",
                     [[[cb error] localizedDescription] UTF8String]);
    }
}

const char *qw_cmd_error(qw_cmd c) {
    return (c && c->err[0]) ? c->err : NULL;
}

void qw_cmd_free(qw_cmd c) {
    if (!c) return;
    qw_cmd_end_encoding(c);
    if (c->cb) CFBridgingRelease((CFTypeRef)c->cb);
    free(c);
}

/* ---- ops ------------------------------------------------------------------ */

static void qw_set(id<MTLComputeCommandEncoder> enc, qw_ref r, NSUInteger idx) {
    [enc setBuffer:(__bridge id<MTLBuffer>)r.buf offset:r.off atIndex:idx];
}

typedef struct { uint32_t k, n, rows; } qw_matmul_args;

void qw_op_qmv_q4(qw_cmd c, qw_ref y, qw_ref x,
                  qw_ref w, qw_ref scales, qw_ref biases,
                  int32_t k, int32_t n, int32_t rows) {
    if (!c || !c->enc) return;
    id<MTLComputePipelineState> ps = qw_pipeline(@"qw_qmv_q4_g64");
    if (!ps) return;

    id<MTLComputeCommandEncoder> enc = (__bridge id<MTLComputeCommandEncoder>)c->enc;
    [enc setComputePipelineState:ps];
    qw_set(enc, w, 0);
    qw_set(enc, scales, 1);
    qw_set(enc, biases, 2);
    qw_set(enc, x, 3);
    qw_set(enc, y, 4);
    qw_matmul_args args = { (uint32_t)k, (uint32_t)n, (uint32_t)rows };
    [enc setBytes:&args length:sizeof args atIndex:5];

    /* One simdgroup per QW_QMV_ROWS output rows; 8 simdgroups per threadgroup
     * is the usual sweet spot for a memory-bound kernel on Apple GPUs. */
    const NSUInteger kQmvRows = 4, nsg = 8;
    NSUInteger per_tg = nsg * kQmvRows;
    MTLSize grid = MTLSizeMake(((NSUInteger)n + per_tg - 1) / per_tg, (NSUInteger)rows, 1);
    MTLSize tg   = MTLSizeMake(32 * nsg, 1, 1);
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
}

void qw_op_qmm_q4(qw_cmd c, qw_ref y, qw_ref x,
                  qw_ref w, qw_ref scales, qw_ref biases,
                  int32_t k, int32_t n, int32_t rows) {
    if (!c || !c->enc) return;
    /* The tile walks K in steps of 32 with no remainder handling; every
     * projection in this model has k divisible by 64, so refuse loudly rather
     * than compute a truncated dot product. */
    if (k % QW_QMM_BK != 0) {
        fprintf(stderr, "qwasar: qmm needs k divisible by %d, got %d\n", QW_QMM_BK, k);
        return;
    }
    id<MTLComputePipelineState> ps = qw_pipeline(@"qw_qmm_q4_g64");
    if (!ps) return;

    id<MTLComputeCommandEncoder> enc = (__bridge id<MTLComputeCommandEncoder>)c->enc;
    [enc setComputePipelineState:ps];
    qw_set(enc, w, 0);
    qw_set(enc, scales, 1);
    qw_set(enc, biases, 2);
    qw_set(enc, x, 3);
    qw_set(enc, y, 4);
    qw_matmul_args args = { (uint32_t)k, (uint32_t)n, (uint32_t)rows };
    [enc setBytes:&args length:sizeof args atIndex:5];

    MTLSize grid = MTLSizeMake(((NSUInteger)n + QW_QMM_BN - 1) / QW_QMM_BN,
                               ((NSUInteger)rows + QW_QMM_BM - 1) / QW_QMM_BM, 1);
    [enc dispatchThreadgroups:grid
        threadsPerThreadgroup:MTLSizeMake(QW_QMM_THREADS, 1, 1)];
}

/* Blocks of QW_QMVB_B tokens, each block one pass over the weights.  The
 * remainder is a short block rather than a separate path: the kernel predicates
 * its dead token lanes off, and the arithmetic that wastes is arithmetic this
 * kernel has to spare. */
void qw_op_qmvb_q4(qw_cmd c, qw_ref y, qw_ref x,
                   qw_ref w, qw_ref scales, qw_ref biases,
                   int32_t k, int32_t n, int32_t rows) {
    if (!c || !c->enc) return;
    id<MTLComputePipelineState> ps = qw_pipeline(@"qw_qmvb_q4_g64");
    if (!ps) return;

    id<MTLComputeCommandEncoder> enc = (__bridge id<MTLComputeCommandEncoder>)c->enc;
    [enc setComputePipelineState:ps];
    qw_set(enc, w, 0);
    qw_set(enc, scales, 1);
    qw_set(enc, biases, 2);

    const NSUInteger nsg = 8;
    NSUInteger per_tg = nsg * QW_QMVB_ROWS;
    MTLSize grid = MTLSizeMake(((NSUInteger)n + per_tg - 1) / per_tg, 1, 1);
    MTLSize tg   = MTLSizeMake(32 * nsg, 1, 1);

    for (int32_t base = 0; base < rows; base += QW_QMVB_B) {
        int32_t block = rows - base;
        if (block > QW_QMVB_B) block = QW_QMVB_B;
        qw_set(enc, qw_ref_offset(x, (size_t)base * k * sizeof(float)), 3);
        qw_set(enc, qw_ref_offset(y, (size_t)base * n * sizeof(float)), 4);
        qw_matmul_args args = { (uint32_t)k, (uint32_t)n, (uint32_t)block };
        [enc setBytes:&args length:sizeof args atIndex:5];
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
    }
}

/* Dense bf16, blocked exactly like qw_op_qmvb_q4.  Only the MTP head uses it. */
void qw_op_dmat_bf16(qw_cmd c, qw_ref y, qw_ref x, qw_ref w,
                     int32_t k, int32_t n, int32_t rows) {
    if (!c || !c->enc) return;
    id<MTLComputePipelineState> ps = qw_pipeline(@"qw_dmvb_bf16");
    if (!ps) return;

    id<MTLComputeCommandEncoder> enc = (__bridge id<MTLComputeCommandEncoder>)c->enc;
    [enc setComputePipelineState:ps];
    qw_set(enc, w, 0);

    const NSUInteger nsg = 8;
    NSUInteger per_tg = nsg * QW_QMVB_ROWS;
    MTLSize grid = MTLSizeMake(((NSUInteger)n + per_tg - 1) / per_tg, 1, 1);
    MTLSize tg   = MTLSizeMake(32 * nsg, 1, 1);

    for (int32_t base = 0; base < rows; base += QW_QMVB_B) {
        int32_t block = rows - base;
        if (block > QW_QMVB_B) block = QW_QMVB_B;
        qw_set(enc, qw_ref_offset(x, (size_t)base * k * sizeof(float)), 1);
        qw_set(enc, qw_ref_offset(y, (size_t)base * n * sizeof(float)), 2);
        qw_matmul_args args = { (uint32_t)k, (uint32_t)n, (uint32_t)block };
        [enc setBytes:&args length:sizeof args atIndex:3];
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
    }
}

/* Blocks of QW_QMVB_B tokens below the matmul's crossover, tiles above it.
 * The vision tower is the only caller and is always above it in practice, but
 * a one-patch image should still work. */
void qw_op_dmm_bf16(qw_cmd c, qw_ref y, qw_ref x, qw_ref w,
                    int32_t k, int32_t n, int32_t rows) {
    if (!c || !c->enc) return;
    if (rows < QW_QMM_MIN_ROWS || k % QW_QMM_BK != 0) {
        qw_op_dmat_bf16(c, y, x, w, k, n, rows);
        return;
    }
    id<MTLComputePipelineState> ps = qw_pipeline(@"qw_dmm_bf16");
    if (!ps) return;
    id<MTLComputeCommandEncoder> enc = (__bridge id<MTLComputeCommandEncoder>)c->enc;
    [enc setComputePipelineState:ps];
    qw_set(enc, w, 0);
    qw_set(enc, x, 1);
    qw_set(enc, y, 2);
    qw_matmul_args args = { (uint32_t)k, (uint32_t)n, (uint32_t)rows };
    [enc setBytes:&args length:sizeof args atIndex:3];
    MTLSize grid = MTLSizeMake(((NSUInteger)n + QW_QMM_BN - 1) / QW_QMM_BN,
                               ((NSUInteger)rows + QW_QMM_BM - 1) / QW_QMM_BM, 1);
    [enc dispatchThreadgroups:grid
        threadsPerThreadgroup:MTLSizeMake(QW_QMM_THREADS, 1, 1)];
}

void qw_op_qmat_q4(qw_cmd c, qw_ref y, qw_ref x,
                   qw_ref w, qw_ref scales, qw_ref biases,
                   int32_t k, int32_t n, int32_t rows) {
    if (rows >= QW_QMM_MIN_ROWS && k % 32 == 0)
        qw_op_qmm_q4(c, y, x, w, scales, biases, k, n, rows);
    else if (rows > 1)
        qw_op_qmvb_q4(c, y, x, w, scales, biases, k, n, rows);
    else
        qw_op_qmv_q4(c, y, x, w, scales, biases, k, n, rows);
}

typedef struct { uint32_t dim, rows; float eps, out_scale; uint32_t has_weight; } qw_norm_args;

/* One threadgroup per row.  Width is capped at the row length rounded up to a
 * simd, so the dim-128 gated-delta norms do not launch 256 idle threads. */
static MTLSize qw_norm_threads(int32_t dim) {
    NSUInteger t = ((NSUInteger)dim + 31) / 32 * 32;
    if (t > 256) t = 256;
    if (t < 32)  t = 32;
    return MTLSizeMake(t, 1, 1);
}

void qw_op_rms_norm(qw_cmd c, qw_ref y, qw_ref x, qw_ref weight,
                    int32_t dim, int32_t rows, float eps, float out_scale) {
    if (!c || !c->enc) return;
    id<MTLComputePipelineState> ps = qw_pipeline(@"qw_rms_norm");
    if (!ps) return;

    id<MTLComputeCommandEncoder> enc = (__bridge id<MTLComputeCommandEncoder>)c->enc;
    [enc setComputePipelineState:ps];
    qw_set(enc, x, 0);
    /* An absent weight still needs a bound buffer; the kernel guards on the
     * has_weight flag rather than on the pointer. */
    qw_set(enc, weight.buf ? weight : x, 1);
    qw_set(enc, y, 2);
    qw_norm_args args = { (uint32_t)dim, (uint32_t)rows, eps, out_scale,
                          weight.buf ? 1u : 0u };
    [enc setBytes:&args length:sizeof args atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)rows, 1, 1)
        threadsPerThreadgroup:qw_norm_threads(dim)];
}

void qw_op_rms_norm_concat(qw_cmd c, qw_ref y, qw_ref e, qw_ref we,
                           qw_ref h, qw_ref wh, int32_t dim, int32_t rows,
                           float eps) {
    if (!c || !c->enc) return;
    id<MTLComputePipelineState> ps = qw_pipeline(@"qw_rms_norm_concat");
    if (!ps) return;

    id<MTLComputeCommandEncoder> enc = (__bridge id<MTLComputeCommandEncoder>)c->enc;
    [enc setComputePipelineState:ps];
    qw_set(enc, e, 0);
    qw_set(enc, we, 1);
    qw_set(enc, h, 2);
    qw_set(enc, wh, 3);
    qw_set(enc, y, 4);
    qw_norm_args args = { (uint32_t)dim, (uint32_t)rows, eps, 1.0f, 1u };
    [enc setBytes:&args length:sizeof args atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)rows, 1, 1)
        threadsPerThreadgroup:qw_norm_threads(dim)];
}

void qw_op_layer_norm(qw_cmd c, qw_ref y, qw_ref x, qw_ref weight, qw_ref bias,
                      int32_t dim, int32_t rows, float eps) {
    if (!c || !c->enc) return;
    id<MTLComputePipelineState> ps = qw_pipeline(@"qw_layer_norm");
    if (!ps) return;
    id<MTLComputeCommandEncoder> enc = (__bridge id<MTLComputeCommandEncoder>)c->enc;
    [enc setComputePipelineState:ps];
    qw_set(enc, x, 0);
    qw_set(enc, weight, 1);
    qw_set(enc, bias, 2);
    qw_set(enc, y, 3);
    qw_norm_args args = { (uint32_t)dim, (uint32_t)rows, eps, 1.0f, 1u };
    [enc setBytes:&args length:sizeof args atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)rows, 1, 1)
        threadsPerThreadgroup:qw_norm_threads(dim)];
}

void qw_op_gelu_tanh(qw_cmd c, qw_ref y, int32_t n) {
    if (!c || !c->enc) return;
    id<MTLComputePipelineState> ps = qw_pipeline(@"qw_gelu_tanh");
    if (!ps) return;
    id<MTLComputeCommandEncoder> enc = (__bridge id<MTLComputeCommandEncoder>)c->enc;
    [enc setComputePipelineState:ps];
    qw_set(enc, y, 0);
    uint32_t nn = (uint32_t)n;
    [enc setBytes:&nn length:sizeof nn atIndex:1];
    [enc dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
   threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

void qw_op_add_bias(qw_cmd c, qw_ref y, qw_ref bias, int32_t dim, int32_t rows) {
    if (!c || !c->enc || !bias.buf) return;
    id<MTLComputePipelineState> ps = qw_pipeline(@"qw_add_bias");
    if (!ps) return;
    id<MTLComputeCommandEncoder> enc = (__bridge id<MTLComputeCommandEncoder>)c->enc;
    [enc setComputePipelineState:ps];
    qw_set(enc, y, 0);
    qw_set(enc, bias, 1);
    uint32_t args[2] = { (uint32_t)dim, (uint32_t)rows };
    [enc setBytes:args length:sizeof args atIndex:2];
    [enc dispatchThreads:MTLSizeMake((NSUInteger)dim, (NSUInteger)rows, 1)
   threadsPerThreadgroup:MTLSizeMake(64, 4, 1)];
}

void qw_op_rope_2d(qw_cmd c, qw_ref x, qw_ref angles,
                   int32_t tokens, int32_t heads, int32_t dim, int32_t stride) {
    if (!c || !c->enc) return;
    id<MTLComputePipelineState> ps = qw_pipeline(@"qw_rope_2d");
    if (!ps) return;
    id<MTLComputeCommandEncoder> enc = (__bridge id<MTLComputeCommandEncoder>)c->enc;
    [enc setComputePipelineState:ps];
    qw_set(enc, x, 0);
    qw_set(enc, angles, 1);
    uint32_t args[4] = { (uint32_t)tokens, (uint32_t)heads, (uint32_t)dim,
                         (uint32_t)stride };
    [enc setBytes:args length:sizeof args atIndex:2];
    [enc dispatchThreads:MTLSizeMake((NSUInteger)(dim / 2), (NSUInteger)heads,
                                     (NSUInteger)tokens)
   threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}

void qw_op_vision_attn(qw_cmd c, qw_ref out, qw_ref qkv, int32_t tokens,
                       int32_t heads, int32_t dim, int32_t segment, float scale) {
    if (!c || !c->enc) return;
    id<MTLComputePipelineState> ps = qw_pipeline(@"qw_vision_attn");
    if (!ps) return;
    id<MTLComputeCommandEncoder> enc = (__bridge id<MTLComputeCommandEncoder>)c->enc;
    [enc setComputePipelineState:ps];
    qw_set(enc, qkv, 0);
    qw_set(enc, out, 1);
    uint32_t args[4] = { (uint32_t)tokens, (uint32_t)heads, (uint32_t)dim,
                         (uint32_t)segment };
    [enc setBytes:args length:sizeof args atIndex:2];
    [enc setBytes:&scale length:sizeof scale atIndex:3];
    /* One threadgroup per (query, head); the head dim is small, so a single
     * simdgroup covers it and the reduction stays inside one shuffle. */
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)tokens, (NSUInteger)heads, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}

void qw_op_rms_norm_gated(qw_cmd c, qw_ref y, qw_ref x, qw_ref weight, qw_ref gate,
                          int32_t dim, int32_t rows, float eps, float out_scale) {
    if (!c || !c->enc) return;
    id<MTLComputePipelineState> ps = qw_pipeline(@"qw_rms_norm_gated");
    if (!ps) return;

    id<MTLComputeCommandEncoder> enc = (__bridge id<MTLComputeCommandEncoder>)c->enc;
    [enc setComputePipelineState:ps];
    qw_set(enc, x, 0);
    qw_set(enc, weight.buf ? weight : x, 1);
    qw_set(enc, gate, 2);
    qw_set(enc, y, 3);
    qw_norm_args args = { (uint32_t)dim, (uint32_t)rows, eps, out_scale,
                          weight.buf ? 1u : 0u };
    [enc setBytes:&args length:sizeof args atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)rows, 1, 1)
        threadsPerThreadgroup:qw_norm_threads(dim)];
}

typedef struct { uint32_t n; } qw_binary_args;

typedef struct { uint32_t channels, rows, n_snap, snap_stride; } qw_conv_args;

void qw_op_conv1d_causal_silu(qw_cmd c, qw_ref y, qw_ref x, qw_ref state,
                              qw_ref weight, int32_t channels, int32_t rows,
                              int32_t ksize, qw_ref snap, int32_t n_snap,
                              int32_t snap_stride) {
    if (!c || !c->enc) return;
    /* The kernel unrolls a fixed 4-tap window (metal/conv1d.metal); refuse
     * rather than silently convolving the wrong support. */
    if (ksize != 4) {
        fprintf(stderr, "qwasar: conv kernel size %d unsupported (expected 4)\n", ksize);
        return;
    }
    id<MTLComputePipelineState> ps = qw_pipeline(@"qw_conv1d_causal_silu");
    if (!ps) return;

    id<MTLComputeCommandEncoder> enc = (__bridge id<MTLComputeCommandEncoder>)c->enc;
    [enc setComputePipelineState:ps];
    qw_set(enc, x, 0);
    qw_set(enc, state, 1);
    qw_set(enc, weight, 2);
    qw_set(enc, y, 3);
    qw_conv_args args = { (uint32_t)channels, (uint32_t)rows,
                          (uint32_t)n_snap, (uint32_t)snap_stride };
    [enc setBytes:&args length:sizeof args atIndex:4];
    /* Metal wants every argument bound whether or not the kernel reads it; the
     * count is what decides, so an unused snapshot aliases the state. */
    qw_set(enc, n_snap > 0 ? snap : state, 5);
    [enc dispatchThreads:MTLSizeMake((NSUInteger)channels, 1, 1)
   threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

typedef struct { uint32_t rows, hv; } qw_gdn_gate_args;

void qw_op_gdn_gates(qw_cmd c, qw_ref g, qw_ref beta, qw_ref a, qw_ref b,
                     qw_ref A_log, qw_ref dt_bias, int32_t n_v_heads, int32_t rows) {
    if (!c || !c->enc) return;
    id<MTLComputePipelineState> ps = qw_pipeline(@"qw_gdn_gates");
    if (!ps) return;

    id<MTLComputeCommandEncoder> enc = (__bridge id<MTLComputeCommandEncoder>)c->enc;
    [enc setComputePipelineState:ps];
    qw_set(enc, a, 0);
    qw_set(enc, b, 1);
    qw_set(enc, A_log, 2);
    qw_set(enc, dt_bias, 3);
    qw_set(enc, g, 4);
    qw_set(enc, beta, 5);
    qw_gdn_gate_args args = { (uint32_t)rows, (uint32_t)n_v_heads };
    [enc setBytes:&args length:sizeof args atIndex:6];
    [enc dispatchThreads:MTLSizeMake((NSUInteger)(rows * n_v_heads), 1, 1)
   threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
}

typedef struct { uint32_t rows, hk, hv, gqa, n_snap, snap_stride; } qw_gdn_args;

void qw_op_gated_delta(qw_cmd c, qw_ref y, qw_ref q, qw_ref k, qw_ref v,
                       qw_ref g, qw_ref beta, qw_ref state,
                       int32_t hk, int32_t hv, int32_t dk, int32_t dv, int32_t rows,
                       qw_ref snap, int32_t n_snap, int32_t snap_stride) {
    if (!c || !c->enc) return;
    /* The per-lane state array is sized at compile time; see the comment at the
     * top of metal/gated_delta.metal. */
    if (dk != 128 || dv != 128) {
        fprintf(stderr, "qwasar: gated-delta head dims %dx%d unsupported "
                        "(kernel is built for 128x128)\n", dk, dv);
        return;
    }
    id<MTLComputePipelineState> ps = qw_pipeline(@"qw_gated_delta");
    if (!ps) return;

    id<MTLComputeCommandEncoder> enc = (__bridge id<MTLComputeCommandEncoder>)c->enc;
    [enc setComputePipelineState:ps];
    qw_set(enc, q, 0);
    qw_set(enc, k, 1);
    qw_set(enc, v, 2);
    qw_set(enc, g, 3);
    qw_set(enc, beta, 4);
    qw_set(enc, state, 5);
    qw_set(enc, y, 6);
    qw_gdn_args args = { (uint32_t)rows, (uint32_t)hk, (uint32_t)hv,
                         (uint32_t)(hv / hk), (uint32_t)n_snap,
                         (uint32_t)snap_stride };
    [enc setBytes:&args length:sizeof args atIndex:7];
    qw_set(enc, n_snap > 0 ? snap : state, 8);

    /* One simdgroup per (head, value row): x spans the key dim, y the value
     * rows, z the heads.  A (32,4,1) threadgroup keeps each simdgroup's 32
     * lanes contiguous in x, which is what makes simd_sum reduce over the key
     * dimension. */
    [enc dispatchThreads:MTLSizeMake(32, (NSUInteger)dv, (NSUInteger)hv)
   threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
}

typedef struct { uint32_t hidden, n_tokens; } qw_embed_args;

void qw_op_embed_q4(qw_cmd c, qw_ref y, qw_ref tokens,
                    qw_ref w, qw_ref scales, qw_ref biases,
                    int32_t hidden, int32_t n_tokens) {
    if (!c || !c->enc) return;
    id<MTLComputePipelineState> ps = qw_pipeline(@"qw_embed_q4");
    if (!ps) return;

    id<MTLComputeCommandEncoder> enc = (__bridge id<MTLComputeCommandEncoder>)c->enc;
    [enc setComputePipelineState:ps];
    qw_set(enc, w, 0);
    qw_set(enc, scales, 1);
    qw_set(enc, biases, 2);
    qw_set(enc, tokens, 3);
    qw_set(enc, y, 4);
    qw_embed_args args = { (uint32_t)hidden, (uint32_t)n_tokens };
    [enc setBytes:&args length:sizeof args atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_tokens, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
}

typedef struct { uint32_t rows, heads, head_dim, rotary_dim; } qw_rope_args;

void qw_op_rope_partial(qw_cmd c, qw_ref x, qw_ref pos, qw_ref axis, qw_ref inv_freq,
                        int32_t rows, int32_t heads, int32_t head_dim, int32_t rotary_dim) {
    if (!c || !c->enc) return;
    id<MTLComputePipelineState> ps = qw_pipeline(@"qw_rope_partial");
    if (!ps) return;

    id<MTLComputeCommandEncoder> enc = (__bridge id<MTLComputeCommandEncoder>)c->enc;
    [enc setComputePipelineState:ps];
    qw_set(enc, x, 0);
    qw_set(enc, pos, 1);
    qw_set(enc, axis, 2);
    qw_set(enc, inv_freq, 3);
    qw_rope_args args = { (uint32_t)rows, (uint32_t)heads,
                          (uint32_t)head_dim, (uint32_t)rotary_dim };
    [enc setBytes:&args length:sizeof args atIndex:4];
    [enc dispatchThreads:MTLSizeMake((NSUInteger)(rotary_dim / 2),
                                     (NSUInteger)(rows * heads), 1)
   threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
}

void qw_op_add_inplace(qw_cmd c, qw_ref y, qw_ref x, int32_t n) {
    if (!c || !c->enc) return;
    id<MTLComputePipelineState> ps = qw_pipeline(@"qw_add_inplace");
    if (!ps) return;
    id<MTLComputeCommandEncoder> enc = (__bridge id<MTLComputeCommandEncoder>)c->enc;
    [enc setComputePipelineState:ps];
    qw_set(enc, y, 0);
    qw_set(enc, x, 1);
    qw_binary_args args = { (uint32_t)n };
    [enc setBytes:&args length:sizeof args atIndex:2];
    [enc dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
   threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

void qw_op_mul_sigmoid(qw_cmd c, qw_ref y, qw_ref gate, int32_t n) {
    if (!c || !c->enc) return;
    id<MTLComputePipelineState> ps = qw_pipeline(@"qw_mul_sigmoid");
    if (!ps) return;
    id<MTLComputeCommandEncoder> enc = (__bridge id<MTLComputeCommandEncoder>)c->enc;
    [enc setComputePipelineState:ps];
    qw_set(enc, y, 0);
    qw_set(enc, gate, 1);
    qw_binary_args args = { (uint32_t)n };
    [enc setBytes:&args length:sizeof args atIndex:2];
    [enc dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
   threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

typedef struct { uint32_t rows, src_stride, offset, len; } qw_slice_args;

void qw_op_slice_rows(qw_cmd c, qw_ref dst, qw_ref src,
                      int32_t rows, int32_t src_stride, int32_t offset, int32_t len) {
    if (!c || !c->enc) return;
    id<MTLComputePipelineState> ps = qw_pipeline(@"qw_slice_rows");
    if (!ps) return;
    id<MTLComputeCommandEncoder> enc = (__bridge id<MTLComputeCommandEncoder>)c->enc;
    [enc setComputePipelineState:ps];
    qw_set(enc, src, 0);
    qw_set(enc, dst, 1);
    qw_slice_args args = { (uint32_t)rows, (uint32_t)src_stride,
                           (uint32_t)offset, (uint32_t)len };
    [enc setBytes:&args length:sizeof args atIndex:2];
    [enc dispatchThreads:MTLSizeMake((NSUInteger)len, (NSUInteger)rows, 1)
   threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

typedef struct { uint32_t rows, heads, dim; } qw_split_args;

void qw_op_split_heads2(qw_cmd c, qw_ref a, qw_ref b, qw_ref src,
                        int32_t rows, int32_t heads, int32_t dim) {
    if (!c || !c->enc) return;
    id<MTLComputePipelineState> ps = qw_pipeline(@"qw_split_heads2");
    if (!ps) return;
    id<MTLComputeCommandEncoder> enc = (__bridge id<MTLComputeCommandEncoder>)c->enc;
    [enc setComputePipelineState:ps];
    qw_set(enc, src, 0);
    qw_set(enc, a, 1);
    qw_set(enc, b, 2);
    qw_split_args args = { (uint32_t)rows, (uint32_t)heads, (uint32_t)dim };
    [enc setBytes:&args length:sizeof args atIndex:3];
    [enc dispatchThreads:MTLSizeMake((NSUInteger)dim, (NSUInteger)(rows * heads), 1)
   threadsPerThreadgroup:MTLSizeMake(64, 4, 1)];
}

typedef struct { uint32_t rows, kv_heads, head_dim, max_ctx, base_pos; } qw_kv_args;

void qw_op_kv_write(qw_cmd c, qw_ref kcache, qw_ref vcache, qw_ref k, qw_ref v,
                    int32_t rows, int32_t kv_heads, int32_t head_dim,
                    int32_t max_ctx, int32_t base_pos) {
    if (!c || !c->enc) return;
    id<MTLComputePipelineState> ps = qw_pipeline(@"qw_kv_write");
    if (!ps) return;
    id<MTLComputeCommandEncoder> enc = (__bridge id<MTLComputeCommandEncoder>)c->enc;
    [enc setComputePipelineState:ps];
    qw_set(enc, k, 0);
    qw_set(enc, v, 1);
    qw_set(enc, kcache, 2);
    qw_set(enc, vcache, 3);
    qw_kv_args args = { (uint32_t)rows, (uint32_t)kv_heads, (uint32_t)head_dim,
                        (uint32_t)max_ctx, (uint32_t)base_pos };
    [enc setBytes:&args length:sizeof args atIndex:4];
    [enc dispatchThreads:MTLSizeMake((NSUInteger)head_dim,
                                     (NSUInteger)(rows * kv_heads), 1)
   threadsPerThreadgroup:MTLSizeMake(64, 4, 1)];
}

typedef struct {
    uint32_t rows, q_heads, kv_heads, gqa, max_ctx, base_pos;
    float    scale;
} qw_attn_args;

void qw_op_attn_decode(qw_cmd c, qw_ref out, qw_ref q, qw_ref kcache, qw_ref vcache,
                       int32_t rows, int32_t q_heads, int32_t kv_heads,
                       int32_t head_dim, int32_t max_ctx, int32_t base_pos,
                       float scale) {
    if (!c || !c->enc) return;
    /* The per-lane query and accumulator arrays are sized at compile time; see
     * the header comment in metal/attn.metal. */
    if (head_dim != 256) {
        fprintf(stderr, "qwasar: attention head_dim %d unsupported "
                        "(kernel is built for 256)\n", head_dim);
        return;
    }
    id<MTLComputePipelineState> ps = qw_pipeline(@"qw_attn_decode");
    if (!ps) return;

    id<MTLComputeCommandEncoder> enc = (__bridge id<MTLComputeCommandEncoder>)c->enc;
    [enc setComputePipelineState:ps];
    qw_set(enc, q, 0);
    qw_set(enc, kcache, 1);
    qw_set(enc, vcache, 2);
    qw_set(enc, out, 3);
    qw_attn_args args = { (uint32_t)rows, (uint32_t)q_heads, (uint32_t)kv_heads,
                          (uint32_t)(q_heads / kv_heads), (uint32_t)max_ctx,
                          (uint32_t)base_pos, scale };
    [enc setBytes:&args length:sizeof args atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)(rows * q_heads), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(32 * 32, 1, 1)];
}

void qw_op_swiglu(qw_cmd c, qw_ref y, qw_ref gate, qw_ref up, int32_t n) {
    if (!c || !c->enc) return;
    id<MTLComputePipelineState> ps = qw_pipeline(@"qw_swiglu");
    if (!ps) return;

    id<MTLComputeCommandEncoder> enc = (__bridge id<MTLComputeCommandEncoder>)c->enc;
    [enc setComputePipelineState:ps];
    qw_set(enc, gate, 0);
    qw_set(enc, up, 1);
    qw_set(enc, y, 2);
    qw_binary_args args = { (uint32_t)n };
    [enc setBytes:&args length:sizeof args atIndex:3];
    [enc dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
   threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

typedef struct { uint32_t n, rows, tiles, write_token, prefix, tail_base; } qw_sel_args;

void qw_op_argmax_top2(qw_cmd c, qw_ref out, qw_ref scratch, qw_ref logits,
                       int32_t n, int32_t rows, qw_ref token_out,
                       int32_t prefix, int32_t tail_base) {
    if (!c || !c->enc) return;
    id<MTLComputePipelineState> p1 = qw_pipeline(@"qw_argmax_top2_partial");
    id<MTLComputePipelineState> p2 = qw_pipeline(@"qw_argmax_top2_final");
    if (!p1 || !p2) return;

    id<MTLComputeCommandEncoder> enc = (__bridge id<MTLComputeCommandEncoder>)c->enc;
    qw_sel_args args = { (uint32_t)n, (uint32_t)rows, (uint32_t)QW_SEL_TILES,
                         token_out.buf ? 1u : 0u,
                         (uint32_t)prefix, (uint32_t)tail_base };

    [enc setComputePipelineState:p1];
    qw_set(enc, logits, 0);
    qw_set(enc, scratch, 1);
    [enc setBytes:&args length:sizeof args atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)rows * QW_SEL_TILES, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

    /* Same encoder, so the partial writes are visible to the fold: Metal
     * orders dispatches within one compute encoder. */
    [enc setComputePipelineState:p2];
    qw_set(enc, scratch, 0);
    qw_set(enc, out, 1);
    /* An absent destination still needs a bound buffer; the kernel guards on
     * the flag rather than the pointer, as the norm kernels do. */
    qw_set(enc, token_out.buf ? token_out : out, 3);
    [enc setBytes:&args length:sizeof args atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)rows, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

/* ---- Flash-Next (qwen4_exp) -- metal/sparse.metal --------------------------- */

typedef struct { uint32_t rows, H, S; } qw_hc_args;
typedef struct { uint32_t dim, groups, rows; float eps; } qw_gnorm_args;
typedef struct { uint32_t n; float scale; } qw_scale_args;
typedef struct { uint32_t rows, E, K, norm; } qw_route_args;
typedef struct { uint32_t k, n, pairs, K, x_by_pair; } qw_bank_args;
typedef struct { uint32_t pairs, I; } qw_swiglu_split_args;
typedef struct { uint32_t rows, K, H; } qw_combine_args;
typedef struct { uint32_t rows, dim; } qw_rowscale_args;
typedef struct { uint32_t rows, nq, d, ratio, base_pos, rotary_dim, max_blocks; float eps; } qw_qsa_score_args;
typedef struct { uint32_t rows, ratio, base_pos, block_topk, max_ctx, max_blocks; } qw_qsa_select_args;
typedef struct { uint32_t channels, rows; } qw_dconv_args;

/* The elementwise kernels share one launch shape. */
static void qw_dispatch_threads(id<MTLComputeCommandEncoder> enc, NSUInteger n) {
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

#define QW_BEGIN(name) \
    if (!c || !c->enc) return; \
    id<MTLComputePipelineState> ps = qw_pipeline(@name); \
    if (!ps) return; \
    id<MTLComputeCommandEncoder> enc = (__bridge id<MTLComputeCommandEncoder>)c->enc; \
    [enc setComputePipelineState:ps];

void qw_op_repeat_cols(qw_cmd c, qw_ref h4, qw_ref x, int32_t rows, int32_t H, int32_t S) {
    QW_BEGIN("qw_repeat_cols")
    qw_set(enc, x, 0); qw_set(enc, h4, 1);
    qw_hc_args args = { (uint32_t)rows, (uint32_t)H, (uint32_t)S };
    [enc setBytes:&args length:sizeof args atIndex:2];
    qw_dispatch_threads(enc, (NSUInteger)rows * H * S);
}

void qw_op_rms_norm_grouped(qw_cmd c, qw_ref y, qw_ref x, qw_ref weight,
                            int32_t dim, int32_t groups, int32_t rows, float eps) {
    QW_BEGIN("qw_rms_norm_grouped")
    qw_set(enc, x, 0); qw_set(enc, weight, 1); qw_set(enc, y, 2);
    qw_gnorm_args args = { (uint32_t)dim, (uint32_t)groups, (uint32_t)rows, eps };
    [enc setBytes:&args length:sizeof args atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)rows * groups, 1, 1)
        threadsPerThreadgroup:qw_norm_threads(dim)];
}

void qw_op_silu_scale(qw_cmd c, qw_ref y, int32_t n, float scale) {
    QW_BEGIN("qw_silu_scale")
    qw_set(enc, y, 0);
    qw_scale_args args = { (uint32_t)n, scale };
    [enc setBytes:&args length:sizeof args atIndex:1];
    qw_dispatch_threads(enc, (NSUInteger)n);
}

void qw_op_hc_mix(qw_cmd c, qw_ref x, qw_ref n, qw_ref m, int32_t rows, int32_t H, int32_t S) {
    QW_BEGIN("qw_hc_mix")
    qw_set(enc, n, 0); qw_set(enc, m, 1); qw_set(enc, x, 2);
    qw_hc_args args = { (uint32_t)rows, (uint32_t)H, (uint32_t)S };
    [enc setBytes:&args length:sizeof args atIndex:3];
    qw_dispatch_threads(enc, (NSUInteger)rows * H);
}

void qw_op_hc_inject(qw_cmd c, qw_ref h4, qw_ref out, qw_ref inj, int32_t rows, int32_t H, int32_t S) {
    QW_BEGIN("qw_hc_inject")
    qw_set(enc, h4, 0); qw_set(enc, out, 1); qw_set(enc, inj, 2);
    qw_hc_args args = { (uint32_t)rows, (uint32_t)H, (uint32_t)S };
    [enc setBytes:&args length:sizeof args atIndex:3];
    qw_dispatch_threads(enc, (NSUInteger)rows * H * S);
}

void qw_op_moe_route(qw_cmd c, qw_ref idx, qw_ref w, qw_ref logits,
                     int32_t rows, int32_t E, int32_t K, bool norm) {
    if (K > 32) { fprintf(stderr, "qwasar: moe top-k %d unsupported (max 32)\n", K); return; }
    QW_BEGIN("qw_moe_route")
    qw_set(enc, logits, 0); qw_set(enc, idx, 1); qw_set(enc, w, 2);
    qw_route_args args = { (uint32_t)rows, (uint32_t)E, (uint32_t)K, norm ? 1u : 0u };
    [enc setBytes:&args length:sizeof args atIndex:3];
    [enc dispatchThreads:MTLSizeMake((NSUInteger)rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}

void qw_op_qmv_q4_bank(qw_cmd c, qw_ref y, qw_ref x, qw_ref idx,
                       qw_ref w, qw_ref scales, qw_ref biases,
                       int32_t k, int32_t n, int32_t pairs, int32_t K, bool x_by_pair) {
    QW_BEGIN("qw_qmv_q4_bank")
    qw_set(enc, w, 0); qw_set(enc, scales, 1); qw_set(enc, biases, 2);
    qw_set(enc, x, 3); qw_set(enc, idx, 4); qw_set(enc, y, 5);
    qw_bank_args args = { (uint32_t)k, (uint32_t)n, (uint32_t)pairs, (uint32_t)K, x_by_pair ? 1u : 0u };
    [enc setBytes:&args length:sizeof args atIndex:6];
    const NSUInteger kRows = 4, nsg = 8, per_tg = nsg * kRows;
    [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)n + per_tg - 1) / per_tg, (NSUInteger)pairs, 1)
        threadsPerThreadgroup:MTLSizeMake(32 * nsg, 1, 1)];
}

void qw_op_swiglu_split(qw_cmd c, qw_ref act, qw_ref gu, int32_t pairs, int32_t I) {
    QW_BEGIN("qw_swiglu_split")
    qw_set(enc, gu, 0); qw_set(enc, act, 1);
    qw_swiglu_split_args args = { (uint32_t)pairs, (uint32_t)I };
    [enc setBytes:&args length:sizeof args atIndex:2];
    qw_dispatch_threads(enc, (NSUInteger)pairs * I);
}

void qw_op_moe_combine(qw_cmd c, qw_ref out, qw_ref y, qw_ref w, int32_t rows, int32_t K, int32_t H) {
    QW_BEGIN("qw_moe_combine")
    qw_set(enc, y, 0); qw_set(enc, w, 1); qw_set(enc, out, 2);
    qw_combine_args args = { (uint32_t)rows, (uint32_t)K, (uint32_t)H };
    [enc setBytes:&args length:sizeof args atIndex:3];
    qw_dispatch_threads(enc, (NSUInteger)rows * H);
}

void qw_op_scale_rows_sigmoid(qw_cmd c, qw_ref y, qw_ref g, int32_t rows, int32_t dim) {
    QW_BEGIN("qw_scale_rows_sigmoid")
    qw_set(enc, y, 0); qw_set(enc, g, 1);
    qw_rowscale_args args = { (uint32_t)rows, (uint32_t)dim };
    [enc setBytes:&args length:sizeof args atIndex:2];
    qw_dispatch_threads(enc, (NSUInteger)rows * dim);
}

void qw_op_rms_norm_gated_sigmoid(qw_cmd c, qw_ref y, qw_ref x, qw_ref weight, qw_ref gate,
                                  int32_t dim, int32_t rows, float eps, float out_scale) {
    QW_BEGIN("qw_rms_norm_gated_sigmoid")
    qw_set(enc, x, 0); qw_set(enc, weight.buf ? weight : x, 1); qw_set(enc, gate, 2); qw_set(enc, y, 3);
    qw_norm_args args = { (uint32_t)dim, (uint32_t)rows, eps, out_scale, weight.buf ? 1u : 0u };
    [enc setBytes:&args length:sizeof args atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)rows, 1, 1) threadsPerThreadgroup:qw_norm_threads(dim)];
}

void qw_op_qsa_scores(qw_cmd c, qw_ref scores, qw_ref qn, qw_ref ikeys, qw_ref k_norm,
                      qw_ref inv_freq, int32_t rows, int32_t nq, int32_t d, int32_t ratio,
                      int32_t base_pos, int32_t rotary_dim, int32_t max_blocks, float eps) {
    /* Lane strips of 32 put a rotary pair (j, j+32) in one lane; see the
     * kernel.  The real model is 64/128, and so is the toy. */
    if (rotary_dim != 64 || d % 32 != 0 || d > 256) {
        fprintf(stderr, "qwasar: QSA indexer shape (rotary %d, head %d) unsupported\n", rotary_dim, d);
        return;
    }
    QW_BEGIN("qw_qsa_scores")
    qw_set(enc, qn, 0); qw_set(enc, ikeys, 1); qw_set(enc, k_norm, 2); qw_set(enc, inv_freq, 3); qw_set(enc, scores, 4);
    qw_qsa_score_args args = { (uint32_t)rows, (uint32_t)nq, (uint32_t)d, (uint32_t)ratio,
                               (uint32_t)base_pos, (uint32_t)rotary_dim, (uint32_t)max_blocks, eps };
    [enc setBytes:&args length:sizeof args atIndex:5];
    const NSUInteger nsg = 8;
    const NSUInteger blocks = (NSUInteger)((base_pos + rows) / ratio);
    if (blocks == 0) return;
    [enc dispatchThreadgroups:MTLSizeMake((blocks + nsg - 1) / nsg, (NSUInteger)rows, 1)
        threadsPerThreadgroup:MTLSizeMake(32 * nsg, 1, 1)];
}

void qw_op_qsa_select(qw_cmd c, qw_ref mask, qw_ref scores, int32_t rows, int32_t ratio,
                      int32_t base_pos, int32_t block_topk, int32_t max_ctx, int32_t max_blocks) {
    QW_BEGIN("qw_qsa_select")
    qw_set(enc, scores, 0); qw_set(enc, mask, 1);
    qw_qsa_select_args args = { (uint32_t)rows, (uint32_t)ratio, (uint32_t)base_pos,
                                (uint32_t)block_topk, (uint32_t)max_ctx, (uint32_t)max_blocks };
    [enc setBytes:&args length:sizeof args atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

void qw_op_attn_masked(qw_cmd c, qw_ref out, qw_ref q, qw_ref kcache, qw_ref vcache, qw_ref mask,
                       int32_t rows, int32_t q_heads, int32_t kv_heads, int32_t head_dim,
                       int32_t max_ctx, int32_t base_pos, float scale) {
    if (head_dim != 256) {
        fprintf(stderr, "qwasar: attention head_dim %d unsupported (kernel is built for 256)\n", head_dim);
        return;
    }
    QW_BEGIN("qw_attn_masked")
    qw_set(enc, q, 0); qw_set(enc, kcache, 1); qw_set(enc, vcache, 2); qw_set(enc, out, 3); qw_set(enc, mask, 4);
    qw_attn_args args = { (uint32_t)rows, (uint32_t)q_heads, (uint32_t)kv_heads,
                          (uint32_t)(q_heads / kv_heads), (uint32_t)max_ctx, (uint32_t)base_pos, scale };
    [enc setBytes:&args length:sizeof args atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)(rows * q_heads), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(32 * 32, 1, 1)];
}

void qw_op_ple_gate(qw_cmd c, qw_ref gv, qw_ref keyn, qw_ref qn, qw_ref value,
                    int32_t rows, int32_t H, int32_t S) {
    QW_BEGIN("qw_ple_gate")
    qw_set(enc, keyn, 0); qw_set(enc, qn, 1); qw_set(enc, value, 2); qw_set(enc, gv, 3);
    qw_hc_args args = { (uint32_t)rows, (uint32_t)H, (uint32_t)S };
    [enc setBytes:&args length:sizeof args atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(32 * (NSUInteger)S, 1, 1)];
}

void qw_op_conv1d_dilated_silu(qw_cmd c, qw_ref y, qw_ref x, qw_ref state, qw_ref weight,
                               int32_t channels, int32_t rows, int32_t ksize, int32_t dilation) {
    if (ksize != 4 || dilation != 3) {
        fprintf(stderr, "qwasar: dilated conv %d/%d unsupported (kernel is built for 4/3)\n", ksize, dilation);
        return;
    }
    QW_BEGIN("qw_conv1d_dilated_silu")
    qw_set(enc, x, 0); qw_set(enc, state, 1); qw_set(enc, weight, 2); qw_set(enc, y, 3);
    qw_dconv_args args = { (uint32_t)channels, (uint32_t)rows };
    [enc setBytes:&args length:sizeof args atIndex:4];
    qw_dispatch_threads(enc, (NSUInteger)channels);
}
