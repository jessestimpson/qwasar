// PathGuard.swift -- every path the model names, confined to the session root.
//
// M1's tools are read-only, so the worst a bad path can do here is disclose a
// file. That is still worth refusing, and more to the point this is the same
// check PLAN.md 7.4 step 8 applies to patch application later, where the stakes
// are writes. Getting it right against a read-only surface first, with its own
// adversarial test, is cheaper than getting it wrong against a writable one.
//
// The rule: after resolving symlinks, the path must be inside the root. Not
// "starts with the root as a string" -- that admits /root-evil for root /root.

import Foundation

public enum PathGuardError: Error, CustomStringConvertible, Equatable {
    case absolute(String)
    case escapesRoot(String)
    case notFound(String)
    case containsNUL

    public var description: String {
        switch self {
        case .absolute(let p):    return "absolute paths are not allowed: \(p)"
        case .escapesRoot(let p): return "path is outside the session directory: \(p)"
        case .notFound(let p):    return "no such file or directory: \(p)"
        case .containsNUL:        return "path contains a NUL byte"
        }
    }
}

public struct PathGuard: Sendable {
    /// Fully resolved, symlinks included. Everything is judged against this.
    public let root: URL

    public init(root: URL) {
        self.root = URL(fileURLWithPath: root.path).resolvingSymlinksInPath()
    }

    /// Resolves a model-supplied relative path, or throws.
    ///
    /// `mustExist` is false for paths that are allowed to be created later; in
    /// M1 everything must exist, but the parameter is here because M5 will need
    /// the other case and the containment rule must not fork.
    public func resolve(_ raw: String, mustExist: Bool = true) throws -> URL {
        if raw.utf8.contains(0) { throw PathGuardError.containsNUL }

        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.hasPrefix("/") || trimmed.hasPrefix("~") {
            throw PathGuardError.absolute(raw)
        }

        let candidate = root.appendingPathComponent(trimmed.isEmpty ? "." : trimmed)
        let resolved = URL(fileURLWithPath: candidate.path).resolvingSymlinksInPath()

        guard contains(resolved) else { throw PathGuardError.escapesRoot(raw) }
        if mustExist, !FileManager.default.fileExists(atPath: resolved.path) {
            throw PathGuardError.notFound(raw)
        }
        return resolved
    }

    /// Path containment by component, not by prefix string.
    ///
    /// A string prefix test says /Users/me/proj-evil is inside /Users/me/proj.
    /// Comparing path components does not.
    public func contains(_ url: URL) -> Bool {
        let r = root.standardized.pathComponents
        let u = url.standardized.pathComponents
        guard u.count >= r.count else { return false }
        return Array(u.prefix(r.count)) == r
    }

    /// How a path reads back to the model: relative to the root, always.
    /// Handing back an absolute path teaches it to use absolute paths.
    public func display(_ url: URL) -> String {
        let r = root.standardized.pathComponents
        let u = url.standardized.pathComponents
        guard u.count >= r.count, Array(u.prefix(r.count)) == r else { return url.path }
        let rest = u.dropFirst(r.count)
        return rest.isEmpty ? "." : rest.joined(separator: "/")
    }
}
