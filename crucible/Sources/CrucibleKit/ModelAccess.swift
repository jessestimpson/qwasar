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
    private let defaultsKey: String
    private var accessing: URL?

    public private(set) var url: URL?

    /// The key is a parameter because there is more than one folder to keep:
    /// the model, and the MTP draft head, which lives wherever the user
    /// downloaded it. Under App Sandbox `~` is the container, so a path like
    /// `~/.cache/qwasar/mtp` is not something the app can simply open -- each
    /// folder needs its own grant, and its own bookmark to survive relaunch.
    public init(defaultsKey: String = "dev.crucible.modelBookmark") {
        self.defaultsKey = defaultsKey
    }

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

    /// An MTP draft head has the same shape as a model directory but is a
    /// single small BF16 head rather than the full stack -- ~810 MB against
    /// ~15 GB. The size is the discriminator, because a user who points this at
    /// the main model should be told so before the engine spends seconds
    /// finding out.
    public static func looksLikeDraftHead(_ u: URL) -> Bool {
        guard looksLikeModel(u) else { return false }
        let fm = FileManager.default
        let entries = (try? fm.contentsOfDirectory(atPath: u.path)) ?? []
        let bytes = entries
            .filter { $0.hasSuffix(".safetensors") }
            .compactMap { name -> UInt64? in
                let attrs = try? fm.attributesOfItem(atPath: u.appendingPathComponent(name).path)
                return (attrs?[.size] as? NSNumber)?.uint64Value
            }
            .reduce(0, +)
        return bytes > 0 && bytes < 4_000_000_000
    }

    /// Where the CLI and the tests keep it (`QWASAR_TEST_MTP`), for the
    /// picker's starting directory. Not readable from inside the sandbox
    /// without a grant -- this only saves the user some navigation.
    public static var conventionalDraftHead: URL {
        URL(fileURLWithPath: NSString(string: "~/.cache/qwasar/mtp").expandingTildeInPath)
    }
}
