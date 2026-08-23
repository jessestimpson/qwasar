// MaterialiseSuite.swift -- the boundary crossing, tested adversarially.
//
// PLAN.md 7.4 step 8 and PLAN.md 10. This is the only code in the project that
// writes to the user's own files, and its input is derived from what a model
// wrote inside a sandbox. So the tests are about refusal: what must never be
// written, and what must be flagged before a person decides.
//
// Each case asserts on the REASON, not merely that something failed. A path
// refused for the wrong reason is a rule that will admit the next variant.

import Foundation
import CrucibleKit

enum MaterialiseSuite {
    static func run() -> Int {
        var f = 0
        let fm = FileManager.default
        let tmp = fm.temporaryDirectory.appendingPathComponent("crucible-mat-\(UUID().uuidString)")
        let root = tmp.appendingPathComponent("project")
        let outside = tmp.appendingPathComponent("outside")
        try? fm.createDirectory(at: root.appendingPathComponent("sub"), withIntermediateDirectories: true)
        try? fm.createDirectory(at: outside, withIntermediateDirectories: true)
        try? "original\n".write(to: root.appendingPathComponent("a.txt"), atomically: true, encoding: .utf8)
        try? "secret\n".write(to: outside.appendingPathComponent("secret.txt"), atomically: true, encoding: .utf8)
        // A directory inside the root that is a symlink out of it: the case
        // where the final path looks contained and the walk is not.
        try? fm.createSymbolicLink(at: root.appendingPathComponent("escape"),
                                   withDestinationURL: outside)
        defer { try? fm.removeItem(at: tmp) }

        // Neither verify nor apply touches the channel.
        let m = Materialiser(channel: VsockChannel(fileDescriptor: -1), projectRoot: root)

        func change(_ path: String, _ status: ProposedChange.Status = .modified,
                    content: String? = "new\n", shaBefore: String? = nil) -> ProposedChange {
            ProposedChange(path: path, status: status, diff: "",
                           content: content.map { Data($0.utf8) },
                           shaBefore: shaBefore, shaAfter: nil)
        }

        func verified(_ cs: [ProposedChange]) -> [ProposedChange] {
            let p = Proposal(changes: cs, skipped: [], summary: "")
            return ((try? m.verify(p)) ?? p).changes
        }

        func refuses(_ c: ProposedChange, containing needle: String, _ label: String) -> Int {
            let r = verified([c])[0]
            guard let reason = r.rejection else {
                print("  FAIL \(label) -- accepted \(c.path.debugDescription)")
                return 1
            }
            if reason.lowercased().contains(needle.lowercased()) {
                return TestMain.check(true, label)
            }
            print("  FAIL \(label) -- refused as \(reason.debugDescription), wanted \(needle.debugDescription)")
            return 1
        }

        // ---- what must never be written ----
        f += refuses(change("/etc/hosts"), containing: "absolute", "an absolute path is refused")
        f += refuses(change("../outside/secret.txt"), containing: "outside",
                     "a parent traversal is refused")
        f += refuses(change("sub/../../outside/x"), containing: "outside",
                     "a traversal that first goes down is refused")
        f += refuses(change("escape/pwned.txt"), containing: "outside",
                     "a path whose parent directory symlinks out is refused")
        f += refuses(change("a.txt", content: nil), containing: "no content",
                     "a modification with no content is refused")

        // ---- what must be flagged, not refused ----
        let drifted = verified([change("a.txt", shaBefore: "deadbeef")])[0]
        f += TestMain.check(drifted.rejection == nil && drifted.conflict != nil,
                            "a file the user changed since the session started is flagged")

        let existing = verified([change("a.txt", .added)])[0]
        f += TestMain.check(existing.conflict != nil,
                            "an 'added' file that already exists is flagged")

        let missing = verified([change("gone.txt", .modified, shaBefore: nil)])[0]
        f += TestMain.check(missing.conflict != nil,
                            "a modification to a file that no longer exists is flagged")

        // A clean modification carries neither.
        let baseSha = Materialiser.sha256(Data("original\n".utf8))
        let clean = verified([change("a.txt", shaBefore: baseSha)])[0]
        f += TestMain.check(clean.rejection == nil && clean.conflict == nil,
                            "an unmodified file applies cleanly")

        // ---- applying ----
        let undo = tmp.appendingPathComponent("undo")
        let proposal = Proposal(changes: verified([
            change("a.txt", shaBefore: baseSha),
            change("sub/new.txt", .added, content: "created\n"),
            change("/etc/hosts"),
        ]), skipped: [], summary: "")

        // Everything named, including the refused one and one that was not
        // proposed at all.
        let all: Set<String> = ["a.txt", "sub/new.txt", "/etc/hosts", "never-mentioned.txt"]
        guard let r = try? m.apply(proposal, paths: all, undoRoot: undo) else {
            print("  FAIL apply threw"); return f + 1
        }

        f += TestMain.check(Set(r.applied) == ["a.txt", "sub/new.txt"],
                            "only applicable, named changes are written (\(r.applied.sorted()))")
        f += TestMain.check(
            (try? String(contentsOf: root.appendingPathComponent("a.txt"), encoding: .utf8)) == "new\n",
            "the file is written")
        f += TestMain.check(
            (try? String(contentsOf: root.appendingPathComponent("sub/new.txt"), encoding: .utf8)) == "created\n",
            "a new file is created, with its directory")
        f += TestMain.check(!fm.fileExists(atPath: "/etc/hosts.crucible-test"),
                            "the refused path wrote nothing")

        // The undo copy holds what was there before, which is the whole reason
        // it is taken before anything is destroyed.
        let backup = r.undoDirectory?.appendingPathComponent("a.txt")
        f += TestMain.check(
            backup.flatMap { try? String(contentsOf: $0, encoding: .utf8) } == "original\n",
            "the previous version is copied aside before overwriting")

        // Only ticked paths are touched: an untouched file stays untouched.
        let untouched = Proposal(changes: verified([change("a.txt", content: "should not appear\n",
                                                           shaBefore: nil)]),
                                 skipped: [], summary: "")
        _ = try? m.apply(untouched, paths: [], undoRoot: undo)
        f += TestMain.check(
            (try? String(contentsOf: root.appendingPathComponent("a.txt"), encoding: .utf8)) == "new\n",
            "an empty approval writes nothing")

        return f
    }
}
