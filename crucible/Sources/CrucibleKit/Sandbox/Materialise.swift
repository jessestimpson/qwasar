// Materialise.swift -- the only thing that crosses back out of the sandbox.
//
// PLAN.md 7.4. Everything else in this design is about keeping the agent's work
// away from the user's files; this is the one deliberate exception, and it is
// built so that no part of it happens without a person having seen it.
//
// The order is the contract, and each step exists because of a specific way
// this can go wrong:
//
//   1. Every path is validated before anything is read or shown. The guest is
//      the one component whose output is derived from what a model wrote, so it
//      is the one component whose output is not trusted.
//   2. The user's tree is checked against the baseline the sandbox started
//      from. If a file moved underneath, that file is flagged and unchecked --
//      applying over an edit the user made themselves is the worst outcome here.
//   3. Nothing is written until an explicit approval names the files.
//   4. What is overwritten is copied aside first, so undo is a real thing and
//      not an apology.
//
// There is no `--yes`. PLAN.md 13: sandbox writes need no confirmation, and the
// boundary crossing never happens without one.

import Foundation
import CryptoKit
public struct ProposedChange: Sendable, Identifiable {
    public enum Status: String, Sendable {
        case added, modified, deleted
    }

    public var id: String { path }
    public let path: String
    public let status: Status
    /// Plain unified diff, for a person to read. Not used to apply anything.
    public let diff: String
    /// The resulting bytes. Absent for a deletion.
    public let content: Data?
    /// The guest's SHA-256 of the file as the sandbox first saw it.
    public let shaBefore: String?
    public let shaAfter: String?

    /// Set by verification, not by the guest.
    public var conflict: String?
    public var rejection: String?

    public var isApplicable: Bool { rejection == nil }

    /// Public so the adversarial suite can construct the cases that must be
    /// refused. Those cases are the point of this type existing.
    public init(path: String, status: Status, diff: String, content: Data?,
                shaBefore: String?, shaAfter: String?,
                conflict: String? = nil, rejection: String? = nil) {
        self.path = path
        self.status = status
        self.diff = diff
        self.content = content
        self.shaBefore = shaBefore
        self.shaAfter = shaAfter
        self.conflict = conflict
        self.rejection = rejection
    }
}

public struct Proposal: Sendable {
    public var changes: [ProposedChange]
    public var skipped: [(path: String, reason: String)]
    public var summary: String

    /// Nothing here can be applied. Distinguished from "no changes", because
    /// the two mean very different things to a user.
    public var isBlocked: Bool {
        !changes.isEmpty && changes.allSatisfy { !$0.isApplicable }
    }

    public init(changes: [ProposedChange],
                skipped: [(path: String, reason: String)] = [],
                summary: String = "") {
        self.changes = changes
        self.skipped = skipped
        self.summary = summary
    }
}

public enum MaterialiseError: Error, CustomStringConvertible {
    case guestRefused(String)
    case malformed(String)
    case noBaseline(String)

    public var description: String {
        switch self {
        case .guestRefused(let m): return m
        case .malformed(let m): return "the guest sent something unreadable: \(m)"
        case .noBaseline(let m): return m
        }
    }
}

public struct Materialiser: Sendable {
    let channel: VsockChannel
    let root: URL
    let guard_: PathGuard

    public init(channel: VsockChannel, projectRoot: URL) {
        self.channel = channel
        self.root = projectRoot
        self.guard_ = PathGuard(root: projectRoot)
    }

    // MARK: Proposing

    /// Asks the sandbox what changed, then validates every answer.
    public func propose(timeout: Int = 120) async throws -> Proposal {
        let r = try await channel.send(op: "propose", timeout: timeout)
        guard r.ok == true, let body = r.result else {
            let msg = r.error ?? "the guest gave no reason"
            if r.kind == "no_baseline" { throw MaterialiseError.noBaseline(msg) }
            throw MaterialiseError.guestRefused(msg)
        }
        return try verify(decode(body))
    }

    private func decode(_ body: String) throws -> Proposal {
        struct Wire: Decodable {
            struct Change: Decodable {
                let path: String
                let status: String
                let diff: String?
                let content: String?
                let sha_before: String?
                let sha_after: String?
            }
            struct Skip: Decodable { let path: String; let reason: String }
            let changes: [Change]
            let skipped: [Skip]
            let summary: String
        }

        guard let data = body.data(using: .utf8),
              let w = try? JSONDecoder().decode(Wire.self, from: data) else {
            throw MaterialiseError.malformed(String(body.prefix(200)))
        }

        let changes: [ProposedChange] = w.changes.map { c in
            ProposedChange(
                path: c.path,
                status: ProposedChange.Status(rawValue: c.status) ?? .modified,
                diff: c.diff ?? "",
                content: c.content.flatMap { Data(base64Encoded: $0) },
                shaBefore: c.sha_before,
                shaAfter: c.sha_after)
        }
        return Proposal(changes: changes,
                        skipped: w.skipped.map { ($0.path, $0.reason) },
                        summary: w.summary)
    }

    // MARK: Verification

    /// Rejects what must never be written, and flags what the user should look
    /// at twice. A rejection is final; a conflict is a warning that leaves the
    /// decision with the person.
    public func verify(_ proposal: Proposal) throws -> Proposal {
        var p = proposal
        for i in p.changes.indices {
            let c = p.changes[i]

            // Containment. The guest checks too (PLAN.md 7.4 step 8): both
            // sides check, because either being wrong is a bug and neither is a
            // reason to trust the other.
            let resolved: URL
            do {
                resolved = try guard_.resolve(c.path, mustExist: false)
            } catch {
                p.changes[i].rejection = "\(error)"
                continue
            }

            // A path that resolves inside the root but whose PARENT is a
            // symlink out of it would write outside anyway. Checked separately
            // because resolve() standardises the final path, not the walk.
            let parent = resolved.deletingLastPathComponent().resolvingSymlinksInPath()
            if !guard_.contains(parent) {
                p.changes[i].rejection = "its containing directory resolves outside the project"
                continue
            }

            if c.status != .deleted && c.content == nil {
                p.changes[i].rejection = "the guest sent no content for it"
                continue
            }

            // Baseline drift: has the user's own copy moved since the sandbox
            // took its snapshot?
            let onDisk = FileManager.default.contents(atPath: resolved.path)
            switch (c.status, c.shaBefore, onDisk) {
            case (.added, _, .some):
                p.changes[i].conflict = "this file did not exist when the session started, "
                                      + "and does now — applying replaces it"
            case (_, .some(let expected), .some(let data)):
                if Materialiser.sha256(data) != expected {
                    p.changes[i].conflict = "you have changed this file since the session "
                                          + "started — applying discards that"
                }
            case (.modified, _, .none), (.deleted, _, .none):
                p.changes[i].conflict = "this file no longer exists on disk"
            default:
                break
            }
        }
        return p
    }

    // MARK: Applying

    public struct ApplyResult: Sendable {
        public var applied: [String] = []
        public var failed: [(path: String, reason: String)] = []
        public var undoDirectory: URL?

        /// Reported honestly: a partial application is a partial application,
        /// never rounded up to success.
        public var isComplete: Bool { failed.isEmpty }
    }

    /// Writes the named changes, after copying aside whatever they overwrite.
    ///
    /// `paths` is what the user ticked. Anything not named is not touched, and
    /// anything rejected during verification is refused even if named.
    public func apply(_ proposal: Proposal, paths: Set<String>,
                      undoRoot: URL) throws -> ApplyResult {
        var result = ApplyResult()
        let chosen = proposal.changes.filter { paths.contains($0.path) && $0.isApplicable }
        guard !chosen.isEmpty else { return result }

        let stamp = ISO8601DateFormatter().string(from: Date())
            .replacingOccurrences(of: ":", with: "-")
        let undo = undoRoot.appendingPathComponent(stamp)
        try FileManager.default.createDirectory(at: undo, withIntermediateDirectories: true)
        result.undoDirectory = undo

        for c in chosen {
            do {
                let target = try guard_.resolve(c.path, mustExist: false)

                // Back up first, always, and before anything is destroyed.
                if let existing = FileManager.default.contents(atPath: target.path) {
                    let backup = undo.appendingPathComponent(c.path)
                    try FileManager.default.createDirectory(
                        at: backup.deletingLastPathComponent(), withIntermediateDirectories: true)
                    try existing.write(to: backup)
                }

                switch c.status {
                case .deleted:
                    try FileManager.default.removeItem(at: target)
                case .added, .modified:
                    guard let content = c.content else {
                        throw MaterialiseError.malformed("no content for \(c.path)")
                    }
                    try FileManager.default.createDirectory(
                        at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
                    // Atomic: a crash mid-write leaves the old file, not half a
                    // new one.
                    try content.write(to: target, options: .atomic)
                }
                result.applied.append(c.path)
            } catch {
                result.failed.append((c.path, "\(error)"))
            }
        }
        return result
    }

    /// Tells the sandbox that what it proposed is now the baseline, so the next
    /// proposal is a diff against what the user accepted rather than against
    /// the original.
    public func acceptBaseline(timeout: Int = 60) async throws {
        let r = try await channel.send(op: "accept_baseline", timeout: timeout)
        if r.ok != true {
            throw MaterialiseError.guestRefused(r.error ?? "re-baselining failed")
        }
    }

    /// Must match the guest's, which uses Erlang's :crypto.hash(:sha256, _).
    /// The comparison is the whole point of the baseline check, so the two
    /// spellings of "sha256" have to be the same function.
    public static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}
