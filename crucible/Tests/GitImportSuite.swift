// GitImportSuite.swift -- the crossing, judged by git itself.
//
// Spec 7.4a set the bar before the code existed: objects written by hand
// must survive `git fsck --strict`, log and merge like natives, and a
// concurrent edit must produce a proper line-level conflict. This suite
// replays that verification on every run, with the real git the test binary
// (unsandboxed, unlike the app) is allowed to spawn: one repo plays the
// host, a clone plays the guest, and GitImport carries the objects across.

import Foundation
import CrucibleKit

enum GitImportSuite {
    @discardableResult
    static func git(_ args: [String], in dir: URL) -> (out: String, code: Int32) {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/bin/git")
        p.arguments = args
        p.currentDirectoryURL = dir
        let pipe = Pipe()
        p.standardOutput = pipe; p.standardError = pipe
        try? p.run(); p.waitUntilExit()
        let out = String(data: pipe.fileHandleForReading.readDataToEndOfFile(),
                         encoding: .utf8) ?? ""
        return (out, p.terminationStatus)
    }

    static func gitData(_ args: [String], in dir: URL) -> Data {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/bin/git")
        p.arguments = args
        p.currentDirectoryURL = dir
        let pipe = Pipe()
        p.standardOutput = pipe
        try? p.run()
        let d = pipe.fileHandleForReading.readDataToEndOfFile()
        p.waitUntilExit()
        return d
    }

    static let ident = ["-c", "user.name=T", "-c", "user.email=t@t"]

    static func makeRepo(_ dir: URL, file: String, content: String) {
        try! FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        git(["init", "-q", "-b", "main"], in: dir)
        try! content.write(to: dir.appendingPathComponent(file), atomically: true, encoding: .utf8)
        git(["add", "-A"], in: dir)
        git(ident + ["commit", "-q", "-m", "base"], in: dir)
    }

    /// What the guest's export produces: every object base..HEAD, in store
    /// form, straight from the clone's own git.
    static func export(_ dir: URL, base: String) -> [(sha: String, type: String, data: Data)] {
        let tip = git(["rev-parse", "HEAD"], in: dir).out.trimmingCharacters(in: .whitespacesAndNewlines)
        let list = git(["rev-list", "--objects", "\(base)..\(tip)"], in: dir).out
        var out: [(String, String, Data)] = []
        for line in list.split(separator: "\n") {
            let sha = String(line.split(separator: " ").first!)
            let type = git(["cat-file", "-t", sha], in: dir).out.trimmingCharacters(in: .whitespacesAndNewlines)
            out.append((sha, type, gitData(["cat-file", type, sha], in: dir)))
        }
        return out
    }

    static func run() -> Int {
        var f = 0
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("crucible-git-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: base) }

        // Host repo, and a clone playing the guest's /work.
        let host = base.appendingPathComponent("host")
        makeRepo(host, file: "a.txt", content: "one\ntwo\nthree\n")
        let baseSha = git(["rev-parse", "HEAD"], in: host).out.trimmingCharacters(in: .whitespacesAndNewlines)
        let guest = base.appendingPathComponent("guest")
        git(["clone", "-q", host.path, guest.path], in: base)

        // The guest works: an edit, a new nested file, an empty file, a
        // deletion -- two commits, the reviewable series the spec wants.
        try! "one\ntwo!\nthree\n".write(to: guest.appendingPathComponent("a.txt"),
                                        atomically: true, encoding: .utf8)
        try! FileManager.default.createDirectory(at: guest.appendingPathComponent("sub"),
                                                 withIntermediateDirectories: true)
        try! "nested\n".write(to: guest.appendingPathComponent("sub/b.txt"),
                              atomically: true, encoding: .utf8)
        try! "".write(to: guest.appendingPathComponent("empty.txt"),
                      atomically: true, encoding: .utf8)
        git(["add", "-A"], in: guest)
        git(ident + ["commit", "-q", "-m", "edit and add"], in: guest)
        git(["rm", "-q", "a.txt"], in: guest)
        git(ident + ["commit", "-q", "-m", "remove a.txt"], in: guest)

        // Across the boundary: verify-and-write, then one ref.
        let objects = export(guest, base: baseSha)
        let tip = git(["rev-parse", "HEAD"], in: guest).out.trimmingCharacters(in: .whitespacesAndNewlines)
        do {
            let imp = try GitImport(repoRoot: host)
            for o in objects { try imp.write(sha: o.sha, type: o.type, content: o.data) }
            try imp.updateRef(branch: "crucible/test", to: tip)
        } catch {
            f += TestMain.check(false, "import threw: \(error)")
            return f
        }
        f += TestMain.check(objects.count >= 8, "the export carried a real object set (\(objects.count))")

        // git's own verdicts, the bar the spec set.
        let fsck = git(["fsck", "--strict"], in: host)
        f += TestMain.check(fsck.code == 0 && !fsck.out.contains("error"),
                            "git fsck --strict accepts every written object")
        let log = git(["log", "--oneline", "crucible/test"], in: host).out
        f += TestMain.check(log.contains("edit and add") && log.contains("remove a.txt"),
                            "the branch logs the guest's own commits")
        f += TestMain.check(git(["merge", "-q", "crucible/test"], in: host).code == 0,
                            "a clean merge merges")
        let merged = (try? String(contentsOf: host.appendingPathComponent("sub/b.txt"),
                                  encoding: .utf8)) ?? ""
        f += TestMain.check(merged == "nested\n"
                            && !FileManager.default.fileExists(atPath: host.appendingPathComponent("a.txt").path),
                            "the merge applied the edit, the add, and the delete")

        // A corrupted object is refused by name, and nothing is written.
        do {
            let imp = try GitImport(repoRoot: host)
            var bad = objects.first { $0.type == "blob" }!
            bad.data.append(Data("tamper".utf8))
            try imp.write(sha: bad.sha, type: bad.type, content: bad.data)
            f += TestMain.check(false, "a tampered object was accepted")
        } catch GitImportError.hashMismatch {
            f += TestMain.check(true, "a tampered object is refused: the hash is the verification")
        } catch {
            f += TestMain.check(false, "wrong refusal: \(error)")
        }

        // The concurrent-edit case: same line changed on both sides must be a
        // LINE-LEVEL conflict with markers -- the thing the byte-copy path
        // could only report as file-granularity drift.
        let host2 = base.appendingPathComponent("host2")
        makeRepo(host2, file: "c.txt", content: "alpha\nbeta\n")
        let base2 = git(["rev-parse", "HEAD"], in: host2).out.trimmingCharacters(in: .whitespacesAndNewlines)
        let guest2 = base.appendingPathComponent("guest2")
        git(["clone", "-q", host2.path, guest2.path], in: base)
        try! "alpha\nguest-beta\n".write(to: guest2.appendingPathComponent("c.txt"),
                                         atomically: true, encoding: .utf8)
        git(["add", "-A"], in: guest2)
        git(ident + ["commit", "-q", "-m", "guest edit"], in: guest2)
        try! "alpha\nhost-beta\n".write(to: host2.appendingPathComponent("c.txt"),
                                        atomically: true, encoding: .utf8)
        git(["add", "-A"], in: host2)
        git(ident + ["commit", "-q", "-m", "host edit"], in: host2)
        do {
            let imp = try GitImport(repoRoot: host2)
            for o in export(guest2, base: base2) { try imp.write(sha: o.sha, type: o.type, content: o.data) }
            try imp.updateRef(branch: "crucible/test", to:
                git(["rev-parse", "HEAD"], in: guest2).out.trimmingCharacters(in: .whitespacesAndNewlines))
        } catch { f += TestMain.check(false, "conflict-case import threw: \(error)") }
        let merge2 = git(ident + ["merge", "crucible/test"], in: host2)
        let conflicted = (try? String(contentsOf: host2.appendingPathComponent("c.txt"),
                                      encoding: .utf8)) ?? ""
        f += TestMain.check(merge2.code != 0 && conflicted.contains("<<<<<<<")
                            && conflicted.contains("guest-beta") && conflicted.contains("host-beta"),
                            "a concurrent edit is a line-level conflict with markers")

        // The summary parser feeds the sheet.
        let commitSha = objects.first { $0.type == "commit" }!
        let sum = GitImport.commitSummary(commitSha.data)
        f += TestMain.check(sum?.author == "T" && (sum?.message.isEmpty == false),
                            "commit summaries parse without spawning git")

        return f
    }
}
