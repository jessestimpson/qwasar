// SystemPrompt.swift -- what a session's system turn says, after the tools.
//
// Shaped like the harness Qwen3.8 Flash-Next's agentic results were measured
// in (its model card: SWE-bench Pro in the Claude Code harness): who the agent
// is, an environment block, how to work, the project's own instruction files,
// then the project's prompt.  The wording is this project's.
//
// All of it is the session's prefix -- fixed when the session opens, and paid
// for once -- so nothing here may vary within a session.  The date does vary
// between days, which costs a new day's first session its shared-prefix
// checkpoint; knowing the date is worth that.

import Foundation

public enum SystemPrompt {

    /// Files a project keeps its instructions for agents in, read from its
    /// root in this order.  AGENT.md is qwasar-agent's; the others are the
    /// conventions other agents (and so the model) know.
    public static let instructionFiles = ["AGENTS.md", "CLAUDE.md", "QWEN.md", "AGENT.md"]
    static let maxInstructionBytes = 24 * 1024

    public struct Environment: Sendable {
        public var workingDirectory: String
        public var gitBranch: String?      // nil: not a git repository
        public var isGitRepo: Bool
        public var platform: String
        public var date: Date

        public init(workingDirectory: String, isGitRepo: Bool, gitBranch: String?,
                    platform: String, date: Date = Date()) {
            self.workingDirectory = workingDirectory
            self.isGitRepo = isGitRepo
            self.gitBranch = gitBranch
            self.platform = platform
            self.date = date
        }

        /// This Mac, for a session whose tools run on it.
        public static func host(root: URL) -> Environment {
            let (git, branch) = gitState(root)
            let v = ProcessInfo.processInfo.operatingSystemVersion
            return Environment(workingDirectory: root.path, isGitRepo: git, gitBranch: branch,
                               platform: "macOS \(v.majorVersion).\(v.minorVersion).\(v.patchVersion) (arm64), shell \(ShellEnvironment.loginShell)")
        }

        /// The sandbox's guest, whose /work is `root`.
        public static func guest(root: URL) -> Environment {
            let (git, branch) = gitState(root)
            return Environment(workingDirectory: "/work", isGitRepo: git, gitBranch: branch,
                               platform: "Linux (Alpine, arm64) in a VM with no network, /bin/sh")
        }
    }

    public static func build(toolsDescription: String, environment e: Environment,
                             projectRoot: URL?, projectPrompt: String) -> String {
        var parts: [String] = []
        parts.append("""
            You are a coding agent. You work in the user's project with the tools above: read and search the code, change it, run it, and verify what you did. Be direct and concise with the user; your final reply is what they read.
            """)
        parts.append(toolsDescription)

        let day = ISO8601DateFormatter.string(from: e.date, timeZone: .current, formatOptions: [.withFullDate])
        var env = "Working directory: \(e.workingDirectory)\n"
        env += "Is a git repository: \(e.isGitRepo ? "yes" : "no")"
        if let b = e.gitBranch { env += " (branch: \(b))" }
        env += "\nPlatform: \(e.platform)\nToday's date: \(day)"
        parts.append("# Environment\n\n<env>\n\(env)\n</env>")

        parts.append("""
            # How to work

            - Understand before you change anything: find the relevant code with Grep and Glob, and Read it. Do not guess at code you have not read.
            - For a task with several steps, keep a TodoWrite list and update it as you go.
            - Match the project: its style, structure, naming and conventions. Look at how nearby code does something, and which libraries it already uses, before adding your own way.
            - Keep changes to what the task needs. Do not refactor, rename or reformat code the task does not touch.
            - Verify your work: build it, run the relevant tests or the program, and read the output. If something fails, fix it, or say plainly what is still broken.
            - Make independent tool calls together, in one response -- several reads, several searches -- rather than one at a time.
            - Use Read, Edit, Write, Glob and Grep for files, not cat, sed, echo or find through Bash.
            - Version control is the user's: do not commit, push, branch or stage unless they ask.
            - When you are done, reply with a short summary: what you changed (with file paths), how you verified it, and anything left undone.
            """)

        if let root = projectRoot {
            for (name, text) in instructions(in: root) {
                parts.append("# Project instructions (\(name))\n\n" + text)
            }
        }

        let own = projectPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        if !own.isEmpty { parts.append("# The user's guidance for this project\n\n" + own) }
        return parts.joined(separator: "\n\n")
    }

    /// The project's instruction files, each capped, in `instructionFiles` order.
    public static func instructions(in root: URL) -> [(String, String)] {
        var out: [(String, String)] = []
        for name in instructionFiles {
            let url = root.appendingPathComponent(name)
            guard let data = try? Data(contentsOf: url), !data.isEmpty else { continue }
            var text = String(decoding: data.prefix(maxInstructionBytes), as: UTF8.self)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if data.count > maxInstructionBytes { text += "\n\n[\(name) continues; Read it for the rest]" }
            out.append((name, text))
        }
        return out
    }

    /// Whether `root` is a git repository, and its branch -- from .git/HEAD,
    /// without running git.
    static func gitState(_ root: URL) -> (Bool, String?) {
        let git = root.appendingPathComponent(".git")
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: git.path, isDirectory: &isDir) else { return (false, nil) }
        var headURL = git.appendingPathComponent("HEAD")
        if !isDir.boolValue,
           let s = try? String(contentsOf: git, encoding: .utf8),
           s.hasPrefix("gitdir: ") {
            // A worktree or submodule: .git is a file naming the real one.
            let dir = s.dropFirst(8).trimmingCharacters(in: .whitespacesAndNewlines)
            let base = dir.hasPrefix("/") ? URL(fileURLWithPath: dir) : root.appendingPathComponent(dir)
            headURL = base.appendingPathComponent("HEAD")
        }
        guard let head = try? String(contentsOf: headURL, encoding: .utf8) else { return (true, nil) }
        let t = head.trimmingCharacters(in: .whitespacesAndNewlines)
        if t.hasPrefix("ref: refs/heads/") { return (true, String(t.dropFirst(16))) }
        return (true, "detached at \(t.prefix(8))")
    }
}
