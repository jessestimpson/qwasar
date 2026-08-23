// PathGuardSuite.swift -- the security boundary, tested adversarially.
//
// PLAN.md 10: "This is the security boundary; it gets the most tests in the
// project." In M1 the tools are read-only so the worst outcome is disclosure,
// but this is the same containment rule PLAN.md 7.4 step 8 will apply to patch
// application, where the outcome is a write outside the project. Getting it
// wrong there is the failure this whole architecture exists to prevent, so the
// rule is tested now, against a surface where a mistake is survivable.
//
// Each case asserts on the REJECTION REASON, not merely on failure. A path
// refused for the wrong reason is a rule that will admit the next variant.

import Foundation
import CrucibleKit

enum PathGuardSuite {
    static func run() -> Int {
        let fm = FileManager.default
        let tmp = fm.temporaryDirectory.appendingPathComponent("crucible-pg-\(UUID().uuidString)")
        let root = tmp.appendingPathComponent("project")
        let outside = tmp.appendingPathComponent("outside")
        try? fm.createDirectory(at: root.appendingPathComponent("sub"), withIntermediateDirectories: true)
        try? fm.createDirectory(at: outside, withIntermediateDirectories: true)
        try? "inside".write(to: root.appendingPathComponent("ok.txt"), atomically: true, encoding: .utf8)
        try? "secret".write(to: outside.appendingPathComponent("secret.txt"), atomically: true, encoding: .utf8)
        // A symlink that leaves the root: the case a string-prefix check misses.
        try? fm.createSymbolicLink(at: root.appendingPathComponent("escape"),
                                   withDestinationURL: outside)
        // A sibling whose name starts with the root's name: /proj vs /proj-evil.
        let sibling = tmp.appendingPathComponent("project-evil")
        try? fm.createDirectory(at: sibling, withIntermediateDirectories: true)
        try? "nope".write(to: sibling.appendingPathComponent("x.txt"), atomically: true, encoding: .utf8)
        defer { try? fm.removeItem(at: tmp) }

        let g = PathGuard(root: root)
        var f = 0

        func rejects(_ path: String, _ want: PathGuardError, _ label: String) -> Int {
            do {
                _ = try g.resolve(path)
                print("  FAIL \(label) -- accepted \(path.debugDescription)")
                return 1
            } catch let e as PathGuardError {
                if e == want { return TestMain.check(true, label) }
                print("  FAIL \(label) -- rejected as \(e), wanted \(want)")
                return 1
            } catch {
                print("  FAIL \(label) -- unexpected \(error)")
                return 1
            }
        }

        func accepts(_ path: String, _ label: String) -> Int {
            do { _ = try g.resolve(path); return TestMain.check(true, label) }
            catch { print("  FAIL \(label) -- rejected: \(error)"); return 1 }
        }

        f += accepts("ok.txt", "a plain file inside the root")
        f += accepts("sub", "a subdirectory")
        f += accepts(".", "the root itself")
        f += accepts("./ok.txt", "a leading ./")

        f += rejects("/etc/hosts", .absolute("/etc/hosts"), "an absolute path")
        f += rejects("~/.ssh/id_rsa", .absolute("~/.ssh/id_rsa"), "a tilde path")
        f += rejects("../outside/secret.txt", .escapesRoot("../outside/secret.txt"),
                     "a parent traversal")
        f += rejects("sub/../../outside/secret.txt", .escapesRoot("sub/../../outside/secret.txt"),
                     "a traversal that first goes down")
        f += rejects("escape/secret.txt", .escapesRoot("escape/secret.txt"),
                     "a symlink leaving the root")
        f += rejects("../project-evil/x.txt", .escapesRoot("../project-evil/x.txt"),
                     "a sibling sharing the root's name prefix")
        f += rejects("no-such-file.txt", .notFound("no-such-file.txt"),
                     "a path that does not exist")
        f += rejects("ok\u{0}.txt", .containsNUL, "a path containing NUL")

        // display() must never hand back an absolute path, or the model learns
        // to use them.
        let shown = g.display(root.appendingPathComponent("sub/deep.txt"))
        f += TestMain.check(shown == "sub/deep.txt", "display is relative to the root")

        return f
    }
}
