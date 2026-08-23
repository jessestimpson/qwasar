// ModelAccess.swift -- keeping access to a user-chosen folder across launches.
//
// PLAN.md 4.2: under App Sandbox there is no other way. The user picks the
// model directory once; a security-scoped bookmark is what makes that grant
// survive a relaunch, and `startAccessingSecurityScopedResource` is what makes
// it apply to this process. Discovering this after building the file layer is a
// bad day, so it is in M0.
//
// The engine mmaps ~16 GB out of this directory. That the sandbox permits it at
// all is half of the M0 gate.

import Foundation

public final class ModelAccess {
    private let defaultsKey = "dev.crucible.modelBookmark"
    private var accessing: URL?

    public private(set) var url: URL?

    public init() {}

    /// Restores a previously granted directory, or nil.
    @discardableResult
    public func restore() -> URL? {
        guard let data = UserDefaults.standard.data(forKey: defaultsKey) else { return nil }
        var stale = false
        guard let u = try? URL(resolvingBookmarkData: data,
                               options: .withSecurityScope,
                               relativeTo: nil,
                               bookmarkDataIsStale: &stale) else { return nil }
        if stale { _ = store(u) }   // re-mint rather than lose the grant
        return adopt(u) ? u : nil
    }

    /// Records a directory the user just chose.
    @discardableResult
    public func store(_ u: URL) -> Bool {
        guard let data = try? u.bookmarkData(options: .withSecurityScope,
                                             includingResourceValuesForKeys: nil,
                                             relativeTo: nil) else { return false }
        UserDefaults.standard.set(data, forKey: defaultsKey)
        return adopt(u)
    }

    private func adopt(_ u: URL) -> Bool {
        release()
        guard u.startAccessingSecurityScopedResource() else { return false }
        accessing = u
        url = u
        return true
    }

    public func release() {
        accessing?.stopAccessingSecurityScopedResource()
        accessing = nil
    }

    deinit { release() }

    /// A model directory is one with a config.json and at least one shard.
    /// Checked before load so a wrong pick fails in a millisecond with a clear
    /// message rather than several seconds into weight binding.
    public static func looksLikeModel(_ u: URL) -> Bool {
        let fm = FileManager.default
        guard fm.fileExists(atPath: u.appendingPathComponent("config.json").path) else { return false }
        let entries = (try? fm.contentsOfDirectory(atPath: u.path)) ?? []
        return entries.contains { $0.hasSuffix(".safetensors") }
    }
}
