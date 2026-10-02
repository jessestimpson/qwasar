// LineDiff.swift -- old and new text as a git-style line diff, for showing
// an Edit: the lines removed, the lines added, and the ones they share.
//
// A longest-common-subsequence over lines.  An edit's old_string and
// new_string are a few dozen lines at most, so the quadratic table is
// nothing; past a size where it would be, the diff degrades honestly to
// "all of this removed, all of that added" rather than stalling the window.

import Foundation

public enum LineDiff {
    public enum Line: Equatable, Sendable {
        case same(String)
        case removed(String)
        case added(String)
    }

    static let maxCells = 400_000

    public static func lines(_ s: String) -> [String] {
        if s.isEmpty { return [] }       // no text is no lines, not one empty one
        var l = s.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
        if s.hasSuffix("\n") { l.removeLast() }
        return l
    }

    public static func diff(old: String, new: String) -> [Line] {
        let a = lines(old), b = lines(new)
        if a.isEmpty { return b.map(Line.added) }
        if b.isEmpty { return a.map(Line.removed) }
        guard a.count * b.count <= maxCells else { return a.map(Line.removed) + b.map(Line.added) }

        // lcs[i][j]: the longest common subsequence of a[i...] and b[j...].
        let n = a.count, m = b.count
        var lcs = [[Int]](repeating: [Int](repeating: 0, count: m + 1), count: n + 1)
        for i in stride(from: n - 1, through: 0, by: -1) {
            for j in stride(from: m - 1, through: 0, by: -1) {
                lcs[i][j] = a[i] == b[j] ? lcs[i + 1][j + 1] + 1 : max(lcs[i + 1][j], lcs[i][j + 1])
            }
        }
        // Walked forwards, removals before additions where both are possible,
        // as git shows them.
        var out: [Line] = []
        var i = 0, j = 0
        while i < n || j < m {
            if i < n, j < m, a[i] == b[j] { out.append(.same(a[i])); i += 1; j += 1 }
            else if i < n, j == m || lcs[i + 1][j] >= lcs[i][j + 1] { out.append(.removed(a[i])); i += 1 }
            else { out.append(.added(b[j])); j += 1 }
        }
        return out
    }
}
