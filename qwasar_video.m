/* Video frames, via AVFoundation.
 *
 * A video reaches the vision tower as a sequence of still frames, so the only
 * thing this file does is turn a file into evenly-spaced RGB bitmaps.  It is
 * Objective-C for the same reason qwasar_metal.m is: the platform's own decoder
 * is the one that already knows about every container and codec the machine can
 * play, and shelling out to ffmpeg would trade "one make, no dependencies" for
 * a binary the user has to install.
 *
 * Sampling follows the model's video_preprocessor_config: two frames a second,
 * at least four, at most 768.  The pair-up into temporal patches happens later,
 * in qwasar_image.c, which is also where the frame count is padded to even. */

#include "qwasar_model.h"

#import <AVFoundation/AVFoundation.h>
#import <CoreGraphics/CoreGraphics.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Draws one decoded frame into a tightly packed RGB8 buffer. */
static unsigned char *qw_cgimage_rgb(CGImageRef img, int w, int h) {
    unsigned char *rgba = calloc((size_t)w * h * 4, 1);
    if (!rgba) return NULL;
    CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();
    CGContextRef ctx = CGBitmapContextCreate(rgba, (size_t)w, (size_t)h, 8,
                                             (size_t)w * 4, cs,
                                             (CGBitmapInfo)kCGImageAlphaNoneSkipLast);
    CGColorSpaceRelease(cs);
    if (!ctx) { free(rgba); return NULL; }
    CGContextDrawImage(ctx, CGRectMake(0, 0, w, h), img);
    CGContextRelease(ctx);

    /* In place, forwards: the RGB triple for pixel i always starts at or before
     * where its RGBA quad does. */
    for (size_t i = 0; i < (size_t)w * h; i++) {
        rgba[i * 3 + 0] = rgba[i * 4 + 0];
        rgba[i * 3 + 1] = rgba[i * 4 + 1];
        rgba[i * 3 + 2] = rgba[i * 4 + 2];
    }
    return rgba;
}

void qw_video_free(qw_video *v) {
    if (!v) return;
    for (int32_t i = 0; i < v->n_frames; i++) free(v->frames[i]);
    free(v->frames);
    memset(v, 0, sizeof *v);
}

bool qw_video_load(qw_video *v, const char *path, double fps,
                   int32_t min_frames, int32_t max_frames,
                   char *err, size_t errcap) {
    memset(v, 0, sizeof *v);
    bool ok = false;
    @autoreleasepool {
        NSString *p = [NSString stringWithUTF8String:path];
        NSURL *url = [NSURL fileURLWithPath:p];
        AVURLAsset *asset = [AVURLAsset URLAssetWithURL:url options:nil];
        if (!asset) { snprintf(err, errcap, "cannot open %s", path); return false; }

        const CMTime dur = [asset duration];
        if (dur.timescale == 0 || CMTIME_IS_INDEFINITE(dur)) {
            snprintf(err, errcap, "%s has no readable duration", path);
            return false;
        }
        const double seconds = (double)dur.value / (double)dur.timescale;
        if (seconds <= 0.0) {
            snprintf(err, errcap, "%s is empty", path);
            return false;
        }

        int32_t want = (int32_t)(seconds * fps);
        if (want < min_frames) want = min_frames;
        if (want > max_frames) want = max_frames;

        AVAssetImageGenerator *gen =
            [AVAssetImageGenerator assetImageGeneratorWithAsset:asset];
        gen.appliesPreferredTrackTransform = YES;
        /* Exact frames: the default lets the generator return whatever keyframe
         * is nearby, which would sample the same picture several times over for
         * a video with sparse keyframes. */
        gen.requestedTimeToleranceBefore = kCMTimeZero;
        gen.requestedTimeToleranceAfter  = kCMTimeZero;

        v->frames = calloc((size_t)want, sizeof *v->frames);
        if (!v->frames) { snprintf(err, errcap, "out of memory"); return false; }

        for (int32_t i = 0; i < want; i++) {
            /* Sample the middle of each interval rather than its edge, so a
             * one-frame sample of a short clip is not its first black frame. */
            const double t = seconds * ((double)i + 0.5) / (double)want;
            CMTime at = CMTimeMakeWithSeconds(t, 600);
            NSError *e = nil;
            /* The synchronous call is deprecated in favour of an async one that
             * only exists from macOS 15.  Frames are wanted in order and one at
             * a time, so the asynchronous form would be a semaphore around the
             * same wait -- and would drop support for every earlier system. */
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            CGImageRef img = [gen copyCGImageAtTime:at actualTime:NULL error:&e];
#pragma clang diagnostic pop
            if (!img) {
                snprintf(err, errcap, "cannot decode %s at %.2fs: %s", path, t,
                         e ? [[e localizedDescription] UTF8String] : "unknown");
                qw_video_free(v);
                return false;
            }
            const int w = (int)CGImageGetWidth(img), h = (int)CGImageGetHeight(img);
            if (i == 0) { v->width = w; v->height = h; }
            unsigned char *rgb = qw_cgimage_rgb(img, v->width, v->height);
            CGImageRelease(img);
            if (!rgb) {
                snprintf(err, errcap, "out of memory decoding %s", path);
                qw_video_free(v);
                return false;
            }
            v->frames[i] = rgb;
            v->n_frames = i + 1;
        }
        v->seconds = seconds;
        ok = true;
    }
    return ok;
}
