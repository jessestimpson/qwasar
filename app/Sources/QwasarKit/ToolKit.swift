// ToolKit.swift -- Read, Write, Edit, Glob, Grep, Bash and TodoWrite, written
// once, over a backend.
//
// A session's tools run on the user's Mac or in the sandbox's guest, and
// should behave identically in both: the same output format, the same edit
// contract, the same errors.  So the tools are here, and a backend supplies
// only what differs -- where a path points, how a file is read and written,
// and how a command runs: FileManager and posix_spawn on the Mac
// (HostBackend), four warden primitives in the guest (GuestBackend).
//
// Results are capped at `resultCap` bytes, and the cap is the model's: every
// byte of a result is prefilled before the model can react to it, which on
// the 27B (~32 tok/s) made 8 KB the most worth sending, and on Flash-Next
// (~465 tok/s) is a tenth of what is worth sending.

import Foundation

// MARK: - the backend

public enum PathKind: Equatable, Sendable {
    case file(size: Int)
    case directory
    case missing
}

public struct ExecResult: Sendable {
    public var status: Int32
    public var output: String
    public var timedOut: Bool
    public init(status: Int32, output: String, timedOut: Bool) {
        self.status = status; self.output = output; self.timedOut = timedOut
    }
}

public protocol ToolBackend: Sendable {
    /// Where relative paths resolve, and what paths are shown relative to.
    var workDir: String { get }
    /// Whether ripgrep is there; without it Grep and Glob fall back to grep
    /// and find, which the host may need and the guest never does.
    var hasRipgrep: Bool { get }
    /// A model-supplied path, made absolute, or nil if it cannot be used.
    func resolve(_ raw: String) -> String?
    func stat(_ path: String) -> Result<PathKind, ToolFailure>
    /// The file's full size and its first `maxBytes`, as text.
    func read(_ path: String, maxBytes: Int) -> Result<(size: Int, text: String), ToolFailure>
    func write(_ path: String, _ content: String) -> Result<Void, ToolFailure>
    /// `sh -c command` in the working directory.
    func exec(_ command: String, timeoutMS: Int, maxBytes: Int) -> Result<ExecResult, ToolFailure>
}

public struct ToolFailure: Error, Sendable {
    public var message: String
    public init(_ m: String) { message = m }
}

// MARK: - the tools

public struct ToolKit: Sendable {
    public let backend: ToolBackend
    /// Bytes a result may put in front of the model.
    public let resultCap: Int
    /// Bash's default timeout, in milliseconds.
    public let defaultTimeoutMS: Int

    /// Ceilings the tools share.  Raw reads stop here: the guest's frames are
    /// 8 MB, and a file larger than this is not one to read into a context.
    static let maxRawRead = 6 * 1024 * 1024
    static let defaultLines = 2000
    static let maxLineLength = 2000
    static let maxListed = 250

    public init(backend: ToolBackend, resultCap: Int, defaultTimeoutMS: Int = 120_000) {
        self.backend = backend
        self.resultCap = resultCap
        self.defaultTimeoutMS = defaultTimeoutMS
    }

    /// The result cap for a model: 64 KB for Flash-Next, 8 KB for the 27B.
    public static func resultCap(flashNext: Bool) -> Int { flashNext ? 64 * 1024 : 8 * 1024 }

    public func run(_ call: ToolCall) -> String {
        switch call.name {
        case "Read":      return read(call)
        case "Write":     return write(call)
        case "Edit":      return edit(call)
        case "Glob":      return glob(call)
        case "Grep":      return grep(call)
        case "Bash":      return bash(call)
        case "TodoWrite": return todo(call)
        // The app answers this one with a card; reaching here means no one
        // is there to (a headless run, a delegated remote model).
        case UserQuestions.toolName:
            return "error: there is no one to answer questions here. " + UserQuestions.skipped
        default:          return "error: no such tool: \(call.name)"
        }
    }

    // MARK: arguments

    /// The path argument under the name the schema gives it, or the older
    /// `path`, which a model will sometimes reach for.
    private func pathArg(_ call: ToolCall, _ key: String = "file_path") -> String? {
        call.argument(key) ?? call.argument("path")
    }

    private func intArg(_ call: ToolCall, _ key: String) -> Int? {
        guard let s = call.argument(key)?.trimmingCharacters(in: .whitespaces), !s.isEmpty else { return nil }
        return Int(s) ?? Double(s).map { Int($0) }
    }

    private func boolArg(_ call: ToolCall, _ key: String) -> Bool {
        ["true", "1", "yes"].contains(call.argument(key)?.lowercased() ?? "")
    }

    /// How a path reads back: relative to the working directory when inside it.
    func display(_ path: String) -> String {
        let root = backend.workDir.hasSuffix("/") ? backend.workDir : backend.workDir + "/"
        if path == backend.workDir { return "." }
        return path.hasPrefix(root) ? String(path.dropFirst(root.count)) : path
    }

    // MARK: Read

    private func read(_ call: ToolCall) -> String {
        guard let raw = pathArg(call), let path = backend.resolve(raw) else {
            return "error: Read requires file_path"
        }
        switch backend.stat(path) {
        case .failure(let f): return "error: \(f.message)"
        case .success(.missing): return "error: no such file: \(raw)"
        case .success(.directory): return "error: \(raw) is a directory; use Glob or Bash (ls) to list it"
        case .success(.file(let size)) where size > Self.maxRawRead:
            return "error: \(raw) is \(size) bytes, too large to read; use Grep to find what you need in it"
        case .success(.file): break
        }
        let got: (size: Int, text: String)
        switch backend.read(path, maxBytes: Self.maxRawRead) {
        case .failure(let f): return "error: \(f.message)"
        case .success(let r): got = r
        }
        if got.text.isEmpty { return "[the file is empty]" }

        var lines = got.text.split(separator: "\n", omittingEmptySubsequences: false)
        if got.text.hasSuffix("\n") { lines.removeLast() }
        let total = lines.count
        let start = max(1, intArg(call, "offset") ?? 1)
        let limit = max(1, intArg(call, "limit") ?? Self.defaultLines)
        if start > total { return "error: offset \(start) is past the end of the file (\(total) lines)" }

        var out = ""
        var shown = start - 1
        for i in (start - 1)..<min(total, start - 1 + limit) {
            var line = String(lines[i])
            if line.count > Self.maxLineLength {
                line = String(line.prefix(Self.maxLineLength)) + "… [line cut at \(Self.maxLineLength) characters]"
            }
            let numbered = String(format: "%6d\t", i + 1) + line + "\n"
            if out.utf8.count + numbered.utf8.count > resultCap { break }
            out += numbered
            shown = i + 1
        }
        if shown == start - 1 {
            return "error: line \(start) alone is larger than a result may be; use Grep or Bash (cut) on it"
        }
        if shown < total {
            out += "\n[showing lines \(start)–\(shown) of \(total). Continue with offset=\(shown + 1), "
                 + "or use Grep to find what you need.]"
        }
        return out
    }

    // MARK: Write

    private func write(_ call: ToolCall) -> String {
        guard let raw = pathArg(call), let path = backend.resolve(raw),
              let content = call.argument("content") else {
            return "error: Write requires file_path and content"
        }
        if case .success(.directory) = backend.stat(path) { return "error: \(raw) is a directory" }
        switch backend.write(path, content) {
        case .failure(let f): return "error: \(f.message)"
        case .success:
            let lines = content.split(separator: "\n", omittingEmptySubsequences: false).count
                      - (content.hasSuffix("\n") ? 1 : 0)
            return "wrote \(content.utf8.count) bytes (\(lines) lines) to \(display(path))"
        }
    }

    // MARK: Edit

    private func edit(_ call: ToolCall) -> String {
        guard let raw = pathArg(call), let path = backend.resolve(raw),
              let old = call.argument("old_string") ?? call.argument("old"),
              let new = call.argument("new_string") ?? call.argument("new") else {
            return "error: Edit requires file_path, old_string and new_string"
        }
        switch backend.stat(path) {
        case .success(.missing): return "error: no such file: \(raw). Use Write to create it."
        case .success(.directory): return "error: \(raw) is a directory"
        case .success(.file(let size)) where size > Self.maxRawRead: return "error: \(raw) is too large to edit here"
        case .failure(let f): return "error: \(f.message)"
        default: break
        }
        let content: String
        switch backend.read(path, maxBytes: Self.maxRawRead) {
        case .failure(let f): return "error: \(f.message)"
        case .success(let r): content = r.text
        }
        switch TextEdit.apply(content, old: old, new: new, replaceAll: boolArg(call, "replace_all")) {
        case .ok(let updated, let n):
            if case .failure(let f) = backend.write(path, updated) { return "error: \(f.message)" }
            return n == 1 ? "edited \(display(path))" : "edited \(display(path)): replaced \(n) occurrences"
        case .notFound:
            return "error: old_string was not found in \(raw). It must match the file exactly, "
                 + "indentation included, without the line numbers Read shows. Read the file again "
                 + "and copy the text."
        case .ambiguous(let n):
            return "error: old_string occurs \(n) times in \(raw). Include more surrounding lines "
                 + "to make it unique, or set replace_all to change every occurrence."
        case .emptyOld:
            return "error: old_string is empty. To create a file use Write; to insert, quote a nearby "
                 + "line as old_string and repeat it in new_string with the addition."
        case .unchanged:
            return "error: old_string and new_string are the same; nothing to change"
        }
    }

    // MARK: Glob and Grep

    /// Single-quoted for /bin/sh.
    static func quote(_ s: String) -> String { "'" + s.replacingOccurrences(of: "'", with: "'\\''") + "'" }

    private func searchRoot(_ call: ToolCall) -> (arg: String, shown: String)? {
        guard let raw = call.argument("path"), !raw.trimmingCharacters(in: .whitespaces).isEmpty else {
            return (".", ".")
        }
        guard let p = backend.resolve(raw) else { return nil }
        let rel = display(p)
        return (rel, rel)
    }

    private func glob(_ call: ToolCall) -> String {
        guard let pattern = call.argument("pattern"), !pattern.isEmpty else { return "error: Glob requires pattern" }
        guard let root = searchRoot(call) else { return "error: bad path" }
        let cmd: String
        if backend.hasRipgrep {
            cmd = "rg --files --color=never --sortr=modified --glob \(Self.quote(pattern)) -- \(Self.quote(root.arg))"
        } else {
            // find matches the whole path with -path when the pattern has a
            // slash, the name otherwise; ** is * to find, which crosses slashes.
            let test = pattern.contains("/")
                ? "-path \(Self.quote("*" + pattern.replacingOccurrences(of: "**/", with: "")))"
                : "-name \(Self.quote(pattern))"
            cmd = "find \(Self.quote(root.arg)) -type f \(test) -not -path '*/.git/*' "
                + "-not -path '*/node_modules/*' 2>/dev/null | sed 's|^\\./||'"
        }
        let r: ExecResult
        switch backend.exec(cmd, timeoutMS: 60_000, maxBytes: 1 << 20) {
        case .failure(let f): return "error: \(f.message)"
        case .success(let x): r = x
        }
        if r.timedOut { return "error: the search took more than 60s; narrow the pattern or the path" }
        let files = r.output.split(separator: "\n").map(String.init).filter { !$0.isEmpty }
            .map { $0.hasPrefix("./") ? String($0.dropFirst(2)) : $0 }
        if files.isEmpty { return "[no files match \(pattern)]" }
        var out = files.prefix(Self.maxListed).joined(separator: "\n")
        if files.count > Self.maxListed {
            out += "\n[\(files.count - Self.maxListed) more; narrow the pattern]"
        }
        return cap(out)
    }

    private func grep(_ call: ToolCall) -> String {
        guard let pattern = call.argument("pattern"), !pattern.isEmpty else { return "error: Grep requires pattern" }
        guard let root = searchRoot(call) else { return "error: bad path" }
        let mode = call.argument("output_mode") ?? "files_with_matches"
        let insensitive = boolArg(call, "-i")
        let context = intArg(call, "-C")
        let glob = call.argument("glob").flatMap { $0.isEmpty ? nil : $0 }

        var cmd: String
        if backend.hasRipgrep {
            cmd = "rg --color=never --no-heading --no-messages --max-columns=400 --max-columns-preview"
            switch mode {
            case "content": cmd += " -n" + (context.map { " -C \($0)" } ?? "")
            case "count":   cmd += " -c"
            default:        cmd += " -l --sortr=modified"
            }
            if insensitive { cmd += " -i" }
            if let glob { cmd += " --glob \(Self.quote(glob))" }
            cmd += " -e \(Self.quote(pattern)) -- \(Self.quote(root.arg))"
        } else {
            cmd = "grep -rEI --exclude-dir=.git --exclude-dir=node_modules --exclude-dir=.build"
            switch mode {
            case "content": cmd += " -n" + (context.map { " -C \($0)" } ?? "")
            case "count":   cmd += " -c"
            default:        cmd += " -l"
            }
            if insensitive { cmd += " -i" }
            if let glob { cmd += " --include=\(Self.quote((glob as NSString).lastPathComponent))" }
            cmd += " -e \(Self.quote(pattern)) -- \(Self.quote(root.arg))"
            if mode == "count" { cmd += " | grep -v ':0$'" }
        }
        let r: ExecResult
        switch backend.exec(cmd, timeoutMS: 60_000, maxBytes: 4 << 20) {
        case .failure(let f): return "error: \(f.message)"
        case .success(let x): r = x
        }
        if r.timedOut { return "error: the search took more than 60s; narrow it with path or glob" }
        // 0 found, 1 nothing found, 2 an error.  Unreadable files are kept
        // quiet, so a 2 with output is a bad pattern -- whose message is the
        // output, and is what the model needs to fix it.
        if r.status >= 2 {
            let msg = r.output.trimmingCharacters(in: .whitespacesAndNewlines)
            if msg.isEmpty || msg.contains("regex parse error") || msg.contains("error parsing")
                || !backend.hasRipgrep {
                return "error: the search failed" + (msg.isEmpty ? "; check the pattern" : ":\n" + String(msg.prefix(600)))
            }
        }
        let lines = r.output.split(separator: "\n", omittingEmptySubsequences: true)
            .map { String($0).hasPrefix("./") ? String(String($0).dropFirst(2)) : String($0) }
        if lines.isEmpty { return "[no matches]" }
        var out = lines.prefix(Self.maxListed).joined(separator: "\n")
        if lines.count > Self.maxListed {
            out += "\n[\(lines.count - Self.maxListed) more lines; narrow with path, glob or the pattern]"
        }
        return cap(out)
    }

    // MARK: Bash

    private func bash(_ call: ToolCall) -> String {
        guard let command = call.argument("command"), !command.isEmpty else { return "error: Bash requires command" }
        let timeout = min(600_000, max(1_000, intArg(call, "timeout") ?? defaultTimeoutMS))
        let r: ExecResult
        switch backend.exec(command, timeoutMS: timeout, maxBytes: 4 << 20) {
        case .failure(let f): return "error: \(f.message)"
        case .success(let x): r = x
        }
        if r.timedOut { return tail("[killed after \(timeout / 1000)s]\n" + r.output) }
        if r.status == 0 { return tail(r.output.isEmpty ? "[no output]" : r.output) }
        return tail("[exit \(r.status)]\n" + r.output)
    }

    // MARK: TodoWrite

    private func todo(_ call: ToolCall) -> String {
        guard let raw = call.argument("todos"), let data = raw.data(using: .utf8),
              let items = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] else {
            return "error: TodoWrite requires todos: an array of {content, status}"
        }
        if items.isEmpty { return "the task list is empty" }
        let lines = items.map { item -> String in
            let text = item["content"] as? String ?? "?"
            switch item["status"] as? String {
            case "completed":   return "[x] " + text
            case "in_progress": return "[~] " + text
            default:            return "[ ] " + text
            }
        }
        let done = items.filter { $0["status"] as? String == "completed" }.count
        return "task list updated (\(done) of \(items.count) done):\n" + lines.joined(separator: "\n")
    }

    // MARK: ceilings

    /// A list or a file: cut at a line boundary, and say so.
    func cap(_ s: String) -> String {
        let b = Array(s.utf8)
        guard b.count > resultCap else { return s }
        var head = b.prefix(resultCap)
        if let nl = head.lastIndex(of: 0x0A) { head = head.prefix(upTo: nl) }
        return String(decoding: head, as: UTF8.self) + "\n[cut at \(head.count) of \(b.count) bytes]"
    }

    /// Command output: its end is what matters -- the error, the summary -- so
    /// a long one keeps a little of its start and most of its end.
    func tail(_ s: String) -> String {
        let b = Array(s.utf8)
        guard b.count > resultCap else { return s }
        let keepHead = resultCap / 8, keepTail = resultCap - keepHead
        return String(decoding: b.prefix(keepHead), as: UTF8.self)
             + "\n[... \(b.count - keepHead - keepTail) bytes omitted ...]\n"
             + String(decoding: b.suffix(keepTail), as: UTF8.self)
    }
}
