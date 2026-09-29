// GuestImage.swift -- the golden image, and a copy-on-write clone per session.
//
// PLAN.md 6.3. This is the detail that makes "a VM per session" affordable:
// clonefile(2) on APFS shares every block until one side writes, so a 2 GB
// sparse image clones in microseconds and costs only the delta. Sessions get
// real isolation -- their own filesystem, their own installed packages -- for
// the price of what they actually change, and deleting a session is deleting
// the clone.
//
// The fallback to a full copy is not decoration. A non-APFS volume, or a
// container whose data lives on an external disk, silently turns "instant and
// free" into "2 GB per session", and the inspector says so rather than letting
// a user wonder where their disk went.

import Foundation

public struct GuestImage: Sendable {
    public enum CloneMethod: String, Sendable {
        case copyOnWrite   // clonefile(2): instant, pays only for changes
        case fullCopy      // not APFS: a real 2 GB per session
        case reused        // the session already had a disk, and it was kept
    }

    public struct CloneResult: Sendable {
        public let url: URL
        public let method: CloneMethod
        public let seconds: Double
    }

    public let golden: URL

    public init(golden: URL) { self.golden = golden }

    public var exists: Bool { FileManager.default.fileExists(atPath: golden.path) }

    public var bytes: UInt64 {
        UInt64((try? golden.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0)
    }

    /// Disk space actually consumed, which for a clone is only its divergence
    /// from the golden image. Reported in the inspector because it is the
    /// number a user needs to reason about ten sessions.
    public static func allocatedBytes(_ url: URL) -> UInt64 {
        var st = stat()
        guard stat(url.path, &st) == 0 else { return 0 }
        return UInt64(st.st_blocks) * 512
    }

    /// Clones the golden image for one session.
    public func clone(to destination: URL) throws -> CloneResult {
        guard exists else { throw SandboxError.noImage(golden.path) }
        let fm = FileManager.default
        try fm.createDirectory(at: destination.deletingLastPathComponent(),
                               withIntermediateDirectories: true)
        if fm.fileExists(atPath: destination.path) {
            try fm.removeItem(at: destination)
        }

        let t0 = Date()
        // clonefile refuses if the destination exists, which is why the removal
        // above is unconditional rather than best-effort.
        if clonefile(golden.path, destination.path, 0) == 0 {
            return CloneResult(url: destination, method: .copyOnWrite,
                               seconds: Date().timeIntervalSince(t0))
        }
        let cloneErrno = errno
        do {
            try fm.copyItem(at: golden, to: destination)
        } catch {
            throw SandboxError.noImage(
                "cannot clone \(golden.lastPathComponent): clonefile failed "
                + "(\(String(cString: strerror(cloneErrno)))) and the copy also failed: "
                + error.localizedDescription)
        }
        return CloneResult(url: destination, method: .fullCopy,
                           seconds: Date().timeIntervalSince(t0))
    }

    /// The disk for one session: cloned on first use, KEPT thereafter.
    ///
    /// This distinction is the difference between a resumable session and a
    /// session that silently starts over. `/work` and the git baseline that
    /// `propose` diffs against both live on this disk, so re-cloning it is not
    /// a cache miss -- it is the destruction of everything the model did. And
    /// `clone(to:)` removes its destination before calling clonefile(2), so
    /// there is no partial survival to fall back on.
    ///
    /// Deliberately not a flag on `clone`: the safe behaviour should be the one
    /// with the obvious name, and the destructive one should have to be asked
    /// for. `SandboxManager.discard(session:)` is the only caller that wants it.
    public func provision(at destination: URL) throws -> CloneResult {
        if FileManager.default.fileExists(atPath: destination.path) {
            return CloneResult(url: destination, method: .reused, seconds: 0)
        }
        return try clone(to: destination)
    }

    /// Removes a clone and the EFI variable store that travels with it.
    public static func discard(_ image: URL) {
        let fm = FileManager.default
        try? fm.removeItem(at: image)
        try? fm.removeItem(at: image.deletingPathExtension().appendingPathExtension("nvram"))
    }
}
