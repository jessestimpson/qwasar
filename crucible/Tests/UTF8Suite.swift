// UTF8Suite.swift -- token bytes reassembled into characters.
//
// PLAN.md 3.4: a single token is frequently half a codepoint. The C front ends
// write bytes straight to a terminal and get away with it; SwiftUI does not,
// and the failure is U+FFFD appearing mid-word in the transcript. This drives
// the assembler one byte at a time, which is the worst case the decode loop can
// produce, and checks that the output is exactly the input.

import Foundation
import CrucibleKit

enum UTF8Suite {
    static func run() -> Int {
        var f = 0

        let corpus = [
            "plain ascii",
            "日本語テスト",                       // 3-byte sequences
            "👩‍🚀🇯🇵🎉",                            // 4-byte, ZWJ, regional indicators
            "combining é́ ǫ̈ a\u{0301}",           // base + combining marks
            "math ∑∫≈∞",
            "mixed: ok 日本 👍 done",
            "",
        ]

        for s in corpus {
            var asm = UTF8Assembler()
            var out = ""
            for byte in Array(s.utf8) {
                var b = byte
                withUnsafePointer(to: &b) { p in
                    if let piece = asm.feed(UnsafeBufferPointer(start: p, count: 1)) { out += piece }
                }
            }
            if let piece = asm.flush() { out += piece }
            f += TestMain.check(out == s, "byte-at-a-time round trip: \(s.debugDescription)")
        }

        // Nothing may be emitted while a sequence is incomplete -- that is the
        // whole point, and a "helpful" partial decode is the bug.
        var asm = UTF8Assembler()
        let three = Array("日".utf8)          // 3 bytes
        var emitted = ""
        for (i, byte) in three.enumerated() {
            var b = byte
            let piece = withUnsafePointer(to: &b) { p in
                asm.feed(UnsafeBufferPointer(start: p, count: 1))
            }
            if i < 2 {
                f += TestMain.check(piece == nil, "holds an incomplete sequence at byte \(i + 1)")
            } else {
                emitted = piece ?? ""
            }
        }
        f += TestMain.check(emitted == "日", "emits the character once complete")

        // A malformed stream must not wedge the assembler forever.
        var bad = UTF8Assembler()
        var lone: UInt8 = 0x80                 // a continuation byte with no start
        let got = withUnsafePointer(to: &lone) { p in
            bad.feed(UnsafeBufferPointer(start: p, count: 1))
        }
        f += TestMain.check(got != nil, "releases an invalid lead byte instead of holding it")

        return f
    }
}
