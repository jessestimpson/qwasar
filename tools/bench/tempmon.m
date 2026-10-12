// tempmon: SoC temperatures without root, through IOKit's HID event system
// (the interface Activity-Monitor-style tools use).  Private API, declared here.
// Used by tools/bench/sweep.py to cool between runs and record the maxima.
//   tempmon            one line: epoch die_max dev_max battery_max thermal_state
// (die: "PMU tdie*", the SoC's die sensors; dev: "PMU tdev*"; tcal is a
// constant calibration value and is skipped.)
//   tempmon <secs>     a line every <secs> until killed
//   tempmon -l         list sensors
#import <Foundation/Foundation.h>
#include <IOKit/IOKitLib.h>
typedef struct __IOHIDEvent *IOHIDEventRef;
typedef struct __IOHIDServiceClient *IOHIDServiceClientRef;
typedef struct __IOHIDEventSystemClient *IOHIDEventSystemClientRef;
extern IOHIDEventSystemClientRef IOHIDEventSystemClientCreate(CFAllocatorRef);
extern int IOHIDEventSystemClientSetMatching(IOHIDEventSystemClientRef, CFDictionaryRef);
extern CFArrayRef IOHIDEventSystemClientCopyServices(IOHIDEventSystemClientRef);
extern IOHIDEventRef IOHIDServiceClientCopyEvent(IOHIDServiceClientRef, int64_t, int32_t, int64_t);
extern CFTypeRef IOHIDServiceClientCopyProperty(IOHIDServiceClientRef, CFStringRef);
extern double IOHIDEventGetFloatValue(IOHIDEventRef, int32_t);
#define kTemp 15
int main(int argc, char **argv) { @autoreleasepool {
    IOHIDEventSystemClientRef sys = IOHIDEventSystemClientCreate(kCFAllocatorDefault);
    NSDictionary *m = @{@"PrimaryUsagePage": @0xff00, @"PrimaryUsage": @5};
    IOHIDEventSystemClientSetMatching(sys, (__bridge CFDictionaryRef)m);
    bool list = argc > 1 && !strcmp(argv[1], "-l");
    double every = argc > 1 && !list ? atof(argv[1]) : 0;
    do {
        CFArrayRef svcs = IOHIDEventSystemClientCopyServices(sys);
        double die = -1, dev = -1, bat = -1;
        for (CFIndex i = 0; svcs && i < CFArrayGetCount(svcs); i++) {
            IOHIDServiceClientRef s = (IOHIDServiceClientRef)CFArrayGetValueAtIndex(svcs, i);
            NSString *name = CFBridgingRelease(IOHIDServiceClientCopyProperty(s, CFSTR("Product")));
            IOHIDEventRef ev = IOHIDServiceClientCopyEvent(s, kTemp, 0, 0);
            if (!ev) continue;
            double t = IOHIDEventGetFloatValue(ev, kTemp << 16);
            CFRelease(ev);
            if (t <= 0 || t > 150) continue;
            if (list) printf("%-24s %.1f\n", name.UTF8String ?: "?", t);
            if ([name containsString:@"tdie"] && t > die) die = t;
            if ([name containsString:@"tdev"] && t > dev) dev = t;
            if ([name containsString:@"battery"] && t > bat) bat = t;
        }
        if (svcs) CFRelease(svcs);
        if (!list) {
            printf("%.3f %.1f %.1f %.1f %ld\n", [NSDate date].timeIntervalSince1970, die, dev, bat,
                   (long)[NSProcessInfo processInfo].thermalState);
            fflush(stdout);
        }
        if (every > 0) [NSThread sleepForTimeInterval:every];
    } while (every > 0);
} return 0; }
