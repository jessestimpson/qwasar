// HostToolsSuite.swift -- a session's tools on this Mac.
//
// LineEdit is Warden.Edit's contract (the guest's), so its cases are
// test/unit/warden_edit_test.exs's, one for one.  Then the runner against a
// temp tree: the file tools, grep's exclusions, and bash -- exit status, the
// project as its directory, and a timeout that kills what the command started.

import Foundation
import QwasarKit

private func realpath(_ p: String) -> String {
    guard let r = Darwin.realpath(p, nil) else { return p }
    defer { free(r) }
    return String(cString: r)
}

enum HostToolsSuite {
    static func run() -> Int {
        var f = 0
        f += edits()
        f += textEdits()
        f += runner()
        f += prompt()
        f += environment()
        return f
    }

    private static let fileA = """
    int add(int a, int b) {
        return a + b;
    }

    int sub(int a, int b) {
        return a - b;
    }

    """

    private static func edits() -> Int {
        var f = 0
        func ok(_ r: LineEdit.Outcome, _ want: String, _ what: String) {
            f += TestMain.check(r == .ok(want), what)
        }
        ok(LineEdit.apply(fileA, old: "    return a + b;", new: "    return b + a;"),
           fileA.replacingOccurrences(of: "return a + b;", with: "return b + a;"), "edit: replaces a body")
        ok(LineEdit.apply(fileA, old: "int add(int a, int b) {\n    return a + b;", new: "int add(int x, int y) {\n    return x + y;"),
           fileA.replacingOccurrences(of: "int add(int a, int b) {\n    return a + b;", with: "int add(int x, int y) {\n    return x + y;"),
           "edit: replaces a multi-line run")
        f += TestMain.check({ if case .ok = LineEdit.apply(fileA, old: "    return a + b;\n", new: "    return 0;") { return true }; return false }(),
                            "edit: a trailing newline on old is presentation")
        f += TestMain.check(LineEdit.apply("x = 1;\ny = 2;\nx = 1;\n", old: "x = 1;", new: "x = 3;") == .ambiguous,
                            "edit: two identical lines are ambiguous")
        f += TestMain.check(LineEdit.apply(fileA, old: "}", new: "} /* end */") == .ambiguous,
                            "edit: a bare closing brace is ambiguous")
        f += TestMain.check(LineEdit.apply(fileA, old: "int mul(int a, int b) {", new: "x") == .notFound,
                            "edit: absent text")
        f += TestMain.check(LineEdit.apply(fileA, old: "return a + b", new: "return b + a") == .notFound,
                            "edit: a fragment of a line does not match")
        f += TestMain.check(LineEdit.apply(fileA, old: "return a + b;", new: "return b + a;") == .notFound,
                            "edit: indentation is content")
        ok(LineEdit.apply("a\nb\nc\n", old: "b", new: ""), "a\nc\n", "edit: deleting a line takes its newline")
        ok(LineEdit.apply("a\nb\nc\n", old: "a", new: "A"), "A\nb\nc\n", "edit: the first line")
        ok(LineEdit.apply("a\nb\nc\n", old: "c", new: "C"), "a\nb\nC\n", "edit: the last line")
        ok(LineEdit.apply("a\nb", old: "b", new: "B"), "a\nB", "edit: no trailing newline is kept so")
        ok(LineEdit.apply("a\nb\n", old: "a\nb", new: "x"), "x\n", "edit: the whole file")
        f += TestMain.check(LineEdit.apply(fileA, old: "", new: "x") == .emptyOld, "edit: an empty old is refused")
        ok(LineEdit.apply("a\nc\n", old: "a", new: "a\nb"), "a\nb\nc\n", "edit: inserting via a unique anchor")
        ok(LineEdit.apply("héllo\nwörld\n", old: "wörld", new: "world"), "héllo\nworld\n", "edit: multi-byte text")
        return f
    }

    private static func textEdits() -> Int {
        var f = 0
        func ok(_ r: TextEdit.Outcome, _ want: String, _ what: String) {
            f += TestMain.check(r == .ok(want, replaced: 1), what)
        }
        ok(TextEdit.apply("let x = 1\nlet y = 2\n", old: "x = 1", new: "x = 3"),
           "let x = 3\nlet y = 2\n", "Edit: a fragment of a line matches (substring contract)")
        f += TestMain.check(TextEdit.apply("a\nb\na\n", old: "zz", new: "q") == .notFound, "Edit: absent")
        f += TestMain.check(TextEdit.apply("f(a)\ng(a)\n", old: "(a)", new: "(b)") == .ambiguous(2),
                            "Edit: two substring matches and no whole-line one are ambiguous")
        ok(TextEdit.apply("  return x;\nreturn x;\n", old: "return x;", new: "return y;"),
           "  return x;\nreturn y;\n", "Edit: of several matches, the one whole line is meant")
        f += TestMain.check(TextEdit.apply("f(a)\ng(a)\n", old: "(a)", new: "(b)", replaceAll: true)
                            == .ok("f(b)\ng(b)\n", replaced: 2), "Edit: replace_all")
        ok(TextEdit.apply("a\nb\nc\n", old: "b", new: ""), "a\nc\n", "Edit: deleting a whole line takes its newline")
        ok(TextEdit.apply("abc\n", old: "b", new: ""), "ac\n", "Edit: deleting inside a line does not")
        f += TestMain.check(TextEdit.apply("x", old: "x", new: "x") == .unchanged, "Edit: old == new is refused")
        f += TestMain.check(TextEdit.apply("x", old: "", new: "y") == .emptyOld, "Edit: empty old is refused")
        ok(TextEdit.apply("a\nb\n", old: "b\n\n", new: "B"), "a\nB\n",
           "Edit: a trailing newline on old is forgiven, as before")
        return f
    }

    private static func runner() -> Int {
        var f = 0
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwasar-host-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: tmp.appendingPathComponent("src"),
                                                 withIntermediateDirectories: true)
        try? FileManager.default.createDirectory(at: tmp.appendingPathComponent("node_modules/x"),
                                                 withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmp) }
        try? "needle in src\n".write(to: tmp.appendingPathComponent("src/a.txt"), atomically: true, encoding: .utf8)
        try? "needle in deps\n".write(to: tmp.appendingPathComponent("node_modules/x/b.txt"), atomically: true, encoding: .utf8)
        // A .gitignore'd directory is left out by ripgrep; grep's fallback
        // leaves node_modules out by name.
        try? "node_modules/\n".write(to: tmp.appendingPathComponent(".gitignore"), atomically: true, encoding: .utf8)
        _ = try? Process.run(URL(fileURLWithPath: "/usr/bin/git"), arguments: ["-C", tmp.path, "init", "-q"]).waitUntilExit()

        let env = ShellEnvironment.resolve(in: tmp)
        let r = HostToolRunner(root: tmp, environment: env, timeout: 2, resultCap: 4096)
        func call(_ name: String, _ args: [String: String]) -> String {
            r.run(ToolCall(name: name, arguments: args))
        }
        print("  info ripgrep on PATH: \(r.kit.backend.hasRipgrep)")

        f += TestMain.check(r.schemas.count == 7, "seven core tools on the host")
        f += TestMain.check(call("Write", ["file_path": "new/dir/f.txt", "content": "one\ntwo\n"])
                                == "wrote 8 bytes (2 lines) to new/dir/f.txt", "Write creates directories and reports")
        f += TestMain.check(call("Read", ["file_path": "new/dir/f.txt"]) == "     1\tone\n     2\ttwo\n",
                            "Read numbers lines like cat -n")
        f += TestMain.check(call("Edit", ["file_path": "new/dir/f.txt", "old_string": "two", "new_string": "2"])
                                == "edited new/dir/f.txt"
                            && call("Read", ["file_path": "new/dir/f.txt"]) == "     1\tone\n     2\t2\n", "Edit applies")
        f += TestMain.check(call("Edit", ["file_path": "new/dir/f.txt", "old_string": "zzz", "new_string": "y"])
                                .hasPrefix("error: old_string was not found"), "an Edit refusal says why")
        let long = (1...5000).map { "line \($0)" }.joined(separator: "\n") + "\n"
        _ = call("Write", ["file_path": "long.txt", "content": long])
        let head = call("Read", ["file_path": "long.txt"])
        f += TestMain.check(head.hasPrefix("     1\tline 1\n") && head.contains("Continue with offset="),
                            "Read stops at the cap and says where to continue")
        f += TestMain.check(call("Read", ["file_path": "long.txt", "offset": "4999", "limit": "5"])
                                == "  4999\tline 4999\n  5000\tline 5000\n", "Read offset and limit")
        let abs = tmp.appendingPathComponent("src/a.txt").path
        f += TestMain.check(call("Read", ["file_path": abs]) == "     1\tneedle in src\n", "an absolute path is taken as given")
        f += TestMain.check(call("Read", ["file_path": "src"]).contains("is a directory"), "Read on a directory says so")

        let g = call("Grep", ["pattern": "needle"])
        f += TestMain.check(g == "src/a.txt", "Grep lists matching files, dependencies left out (\(g))")
        let gc = call("Grep", ["pattern": "NEEDLE\\s+in", "output_mode": "content", "-i": "true"])
        f += TestMain.check(gc.contains("src/a.txt:1:needle in src"), "Grep content, case-insensitive, \\s works (\(gc))")
        f += TestMain.check(call("Grep", ["pattern": "nothing-matches-this"]) == "[no matches]", "Grep: no matches")
        f += TestMain.check(call("Grep", ["pattern": "a(b"]).hasPrefix("error: the search failed"), "Grep: a bad pattern is an error")
        let gl = call("Glob", ["pattern": "**/*.txt"])
        f += TestMain.check(gl.contains("src/a.txt") && gl.contains("long.txt") && !gl.contains("node_modules"),
                            "Glob finds by pattern, ignored directories left out")

        f += TestMain.check(call("Bash", ["command": "echo out; echo err >&2; exit 3"]) == "[exit 3]\nout\nerr\n",
                            "Bash: exit status, stdout and stderr together")
        let t0 = Date()
        let slow = call("Bash", ["command": "sleep 30 & sleep 30; echo never", "timeout": "1500"])
        f += TestMain.check(slow.hasPrefix("[killed after 1s]") && Date().timeIntervalSince(t0) < 5,
                            "Bash: its timeout kills the command and what it started")
        f += TestMain.check(call("Bash", ["command": "yes | head -c 3000000"]).contains("bytes omitted"),
                            "Bash: long output keeps its head and tail")
        let todo = call("TodoWrite", ["todos": #"[{"content":"Read the parser","status":"completed"},{"content":"Fix it","status":"in_progress"}]"#])
        f += TestMain.check(todo == "task list updated (1 of 2 done):\n[x] Read the parser\n[~] Fix it", "TodoWrite")
        f += TestMain.check(call("elixir", ["code": "1"]).hasPrefix("error: no such tool"), "guest-only tools are refused")

        let ro = ToolRunner(root: tmp)
        f += TestMain.check(ro.run(ToolCall(name: "Read", arguments: ["file_path": "/etc/hosts"])).hasPrefix("error"),
                            "the read-only fallback stays inside the project")
        f += TestMain.check(ro.run(ToolCall(name: "Write", arguments: ["file_path": "x", "content": "y"]))
                                .contains("read-only"), "the read-only fallback does not write")
        return f
    }

    private static func prompt() -> Int {
        var f = 0
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent("qwasar-prompt-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: tmp.appendingPathComponent(".git"), withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmp) }
        try? "ref: refs/heads/feature/x\n".write(to: tmp.appendingPathComponent(".git/HEAD"), atomically: true, encoding: .utf8)
        try? "Run make test before finishing.\n".write(to: tmp.appendingPathComponent("AGENTS.md"), atomically: true, encoding: .utf8)
        let env = SystemPrompt.Environment.host(root: tmp)
        f += TestMain.check(env.isGitRepo && env.gitBranch == "feature/x", "the branch, read from .git/HEAD")
        let p = SystemPrompt.build(toolsDescription: "TOOLS", environment: env, projectRoot: tmp, projectPrompt: "")
        f += TestMain.check(p.contains("TOOLS") && p.contains("<env>\nWorking directory: \(tmp.path)"),
                            "the environment block names the working directory")
        f += TestMain.check(p.contains("# Project instructions (AGENTS.md)\n\nRun make test before finishing."),
                            "AGENTS.md is part of the prompt")
        f += TestMain.check(!p.contains("guidance for this project"), "an empty project prompt adds nothing")
        f += TestMain.check(SystemPrompt.build(toolsDescription: "", environment: env, projectRoot: nil,
                                               projectPrompt: "Be brief.").hasSuffix("Be brief."),
                            "the project's own prompt comes last")
        return f
    }

    private static func environment() -> Int {
        var f = 0
        let shell = ShellEnvironment.loginShell
        f += TestMain.check(FileManager.default.isExecutableFile(atPath: shell), "login shell found: \(shell)")
        let env = ShellEnvironment.resolve(in: URL(fileURLWithPath: NSHomeDirectory()))
        let path = env["PATH"] ?? ""
        f += TestMain.check(path.split(separator: ":").contains("/usr/bin"), "the resolved PATH includes the system's")
        f += TestMain.check(env["HOME"] == NSHomeDirectory(), "the resolved environment has HOME")
        print("  info PATH from \(shell): \(path.split(separator: ":").count) entries")
        return f
    }
}
