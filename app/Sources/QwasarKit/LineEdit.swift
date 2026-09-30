// LineEdit.swift -- line-anchored search and replace, for host sessions.
//
// The same contract as the guest's Warden.Edit and the C agent's
// qw_edit_apply, because it is the contract the `edit` tool's description
// states and the model was promised:
//
//   * whole lines only -- a fragment of a line never matches;
//   * exactly once -- no match and two matches are both refusals;
//   * a trailing newline on `old` is presentation, not content;
//   * deleting takes the line's terminator with it;
//   * a file with no trailing newline neither gains nor loses one.
//
// Bytes, not Characters: the splice addresses the file's own bytes, and the
// comparison is exact.  Tests/HostToolsSuite.swift is Warden.Edit's cases.

import Foundation

public enum LineEdit {
    public enum Outcome: Equatable {
        case ok(String)
        case notFound
        case ambiguous
        case emptyOld
    }

    public static func apply(_ content: String, old: String, new: String) -> Outcome {
        var o = Array(old.utf8)
        while o.last == 0x0A { o.removeLast() }
        if o.isEmpty { return .emptyOld }

        let c = Array(content.utf8)
        let fileLines = lines(c)
        let oldLines = lines(o).map { Array(o[$0]) }
        let no = oldLines.count, nf = fileLines.count
        guard no <= nf else { return .notFound }

        var found: [Int] = []
        for i in 0...(nf - no) {
            var same = true
            for j in 0..<no where !c[fileLines[i + j]].elementsEqual(oldLines[j]) {
                same = false
                break
            }
            if same {
                found.append(i)
                if found.count > 1 { return .ambiguous }
            }
        }
        guard let at = found.first else { return .notFound }

        let a = fileLines[at].lowerBound
        var b = fileLines[at + no - 1].upperBound
        let n = Array(new.utf8)
        // An empty replacement takes the newline with it, so a deleted line
        // leaves nothing rather than a blank.
        if n.isEmpty, b < c.count, c[b] == 0x0A { b += 1 }
        return .ok(String(decoding: c[..<a] + n + c[b...], as: UTF8.self))
    }

    /// Lines as byte ranges, terminators excluded.  A trailing newline does
    /// not make a final empty line, so "a\nb\n" and "a\nb" both have two.
    static func lines(_ s: [UInt8]) -> [Range<Int>] {
        var out: [Range<Int>] = []
        var start = 0
        for (i, b) in s.enumerated() where b == 0x0A {
            out.append(start..<i)
            start = i + 1
        }
        if start < s.count { out.append(start..<s.count) }
        return out
    }
}

/// Edit's contract, as the tool describes it: `old_string` must occur exactly
/// once, as an exact substring, unless `replace_all`.
///
/// Two refinements on a plain replace, both from the whole-line contract
/// above, which the model was held to before:
///
///   * when the substring occurs more than once but exactly one occurrence is
///     a run of whole lines, that one is meant -- a quoted `}` or `return x;`
///     matches inside other lines, and the whole-line reading is the one the
///     model quoted;
///   * deleting a run of whole lines takes the newline after it, so a deleted
///     line leaves nothing rather than a blank.
public enum TextEdit {
    public enum Outcome: Equatable {
        case ok(String, replaced: Int)
        case notFound
        case ambiguous(Int)
        case emptyOld
        case unchanged
    }

    public static func apply(_ content: String, old: String, new: String,
                             replaceAll: Bool = false) -> Outcome {
        if old.isEmpty { return .emptyOld }
        if old == new { return .unchanged }
        let c = Array(content.utf8), o = Array(old.utf8), n = Array(new.utf8)
        let hits = occurrences(of: o, in: c)
        if hits.isEmpty {
            // A trailing newline on old is presentation, not content -- the
            // whole-line reading forgives it, so this does too.
            if old.hasSuffix("\n"), case .ok(let s) = LineEdit.apply(content, old: old, new: new) {
                return .ok(s, replaced: 1)
            }
            return .notFound
        }
        if replaceAll {
            var out: [UInt8] = []
            var at = 0
            for h in hits {
                out += c[at..<h]
                out += n
                at = h + o.count
            }
            out += c[at...]
            return .ok(String(decoding: out, as: UTF8.self), replaced: hits.count)
        }
        if hits.count > 1 {
            if case .ok(let s) = LineEdit.apply(content, old: old, new: new) { return .ok(s, replaced: 1) }
            return .ambiguous(hits.count)
        }
        let a = hits[0]
        var b = a + o.count
        let wholeLines = (a == 0 || c[a - 1] == 0x0A) && (b == c.count || c[b] == 0x0A)
        if n.isEmpty, wholeLines, b < c.count, c[b] == 0x0A { b += 1 }
        return .ok(String(decoding: c[..<a] + n + c[b...], as: UTF8.self), replaced: 1)
    }

    /// Non-overlapping occurrences, left to right.
    static func occurrences(of needle: [UInt8], in hay: [UInt8]) -> [Int] {
        guard !needle.isEmpty, needle.count <= hay.count else { return [] }
        var out: [Int] = []
        var i = 0
        let first = needle[0], last = hay.count - needle.count
        while i <= last {
            if hay[i] == first, hay[i..<(i + needle.count)].elementsEqual(needle) {
                out.append(i)
                i += needle.count
            } else {
                i += 1
            }
        }
        return out
    }
}
