// Tools.swift -- the M1 read-only tool implementations, host-side.
//
// Explicitly a stand-in. PLAN.md 12, M3 replaces every one of these with an
// Elixir implementation against /work inside the guest, and PLAN.md 13 says
// plainly that the host runs no model-requested tool. Until the sandbox exists
// these run here, and the compensating constraint is that they cannot change
// anything: no write, no edit, no shell.
//
// Every path goes through PathGuard, so the surface is confined to the
// session's working directory even though nothing here can write to it.

import Foundation
import CQwasar

// Ceilings, and they are far tighter than qwasar_agent.c's -- measured, not
// guessed.
//
// Prefill on this engine runs at ~32 tok/s (M1, measured; PLAN.md 2.5). Every
// tool result is prefilled before the model can react to it, so a result costs
// roughly **one second per 100 bytes**. The C agent's AGENT_MAX_READ of 256 KB
// would be forty minutes of staring at a progress bar for one `read`.
//
// The first agent run made the point: `read` on a 49 KB source file produced a
// 14,640-token prefill -- about seven and a half minutes, and 16% of a 90K
// context window, for one call.
//
// So results are capped at a size whose prefill cost is tolerable, and the
// truncation message tells the model what it can do instead. Being told the
// file is 49 KB and that grep exists is far more useful to it than receiving
// 49 KB.
private let maxResultBytes = 8 * 1024          // ~2.4k tokens, ~75s of prefill
private let maxGrepMatches = 60
private let maxListEntries = 300

public struct ToolCall: Sendable, Equatable {
    public var name: String
    public var arguments: [String: String]

    public init(name: String, arguments: [String: String]) {
        self.name = name
        self.arguments = arguments
    }

    public func argument(_ key: String) -> String? { arguments[key] }
}

public struct ToolRunner: Sendable {
    let guard_: PathGuard

    public init(root: URL) { self.guard_ = PathGuard(root: root) }

    /// Runs one call and returns the text the model will see.
    ///
    /// Errors are returned as results, never thrown. A refused tool has to
    /// reach the model as something it can react to -- the C agent does the
    /// same with a declined confirmation, and for the same reason: aborting the
    /// turn teaches it nothing.
    public func run(_ call: ToolCall) -> String {
        switch call.name {
        case "read": return read(call)
        case "list": return list(call)
        case "grep": return grep(call)
        default:
            return "error: no such tool: \(call.name). Available: "
                 + ToolSurface.names.sorted().joined(separator: ", ")
        }
    }

    // MARK: read

    private func read(_ call: ToolCall) -> String {
        guard let path = call.argument("path") else { return "error: read requires a path" }
        do {
            let url = try guard_.resolve(path)
            var isDir: ObjCBool = false
            FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir)
            if isDir.boolValue { return "error: \(path) is a directory; use list" }

            guard let data = FileManager.default.contents(atPath: url.path) else {
                return "error: cannot read \(path)"
            }
            if data.isEmpty { return "[the file is empty]" }
            if data.count > maxResultBytes {
                // Cut on a line boundary: half a line of C is worse than none,
                // and the model quotes what it reads.
                var head = data.prefix(maxResultBytes)
                if let nl = head.lastIndex(of: 0x0A) { head = head.prefix(upTo: nl) }
                let shownLines = head.filter { $0 == 0x0A }.count + 1
                let totalLines = data.filter { $0 == 0x0A }.count + 1
                return String(decoding: head, as: UTF8.self)
                     + "\n\n[truncated: showed \(shownLines) of \(totalLines) lines "
                     + "(\(head.count) of \(data.count) bytes). Reading the whole file "
                     + "would cost minutes of prompt processing. Use grep to find the "
                     + "lines you need.]"
            }
            return String(decoding: data, as: UTF8.self)
        } catch {
            return "error: \(error)"
        }
    }

    // MARK: list

    private func list(_ call: ToolCall) -> String {
        let path = call.argument("path") ?? "."
        do {
            let url = try guard_.resolve(path)
            let entries = try FileManager.default.contentsOfDirectory(
                at: url, includingPropertiesForKeys: [.isDirectoryKey, .fileSizeKey])
            if entries.isEmpty { return "[the directory is empty]" }

            var lines: [String] = []
            for e in entries.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
                if lines.count >= maxListEntries {
                    lines.append("[\(entries.count - maxListEntries) more entries]")
                    break
                }
                let v = try? e.resourceValues(forKeys: [.isDirectoryKey, .fileSizeKey])
                if v?.isDirectory == true {
                    lines.append("\(e.lastPathComponent)/")
                } else {
                    lines.append("\(e.lastPathComponent)  \(v?.fileSize ?? 0)")
                }
            }
            return cap(lines.joined(separator: "\n"))
        } catch {
            return "error: \(error)"
        }
    }

    /// The backstop. Every result passes through here, so no tool can put more
    /// than one cap's worth of prefill in front of the model however it is
    /// called.
    private func cap(_ s: String) -> String {
        let bytes = Array(s.utf8)
        guard bytes.count > maxResultBytes else { return s }
        var head = bytes.prefix(maxResultBytes)
        if let nl = head.lastIndex(of: 0x0A) { head = head.prefix(upTo: nl) }
        return String(decoding: head, as: UTF8.self)
             + "\n[truncated at \(head.count) of \(bytes.count) bytes]"
    }

    // MARK: grep

    private func grep(_ call: ToolCall) -> String {
        guard let pattern = call.argument("pattern") else {
            return "error: grep requires a pattern"
        }
        let path = call.argument("path") ?? "."

        // NSRegularExpression rather than spawning /usr/bin/grep. A subprocess
        // would inherit the sandbox and mostly work, but "mostly" is not a
        // property worth having in a tool the model relies on, and this stays
        // in-process where its limits are ours to state.
        guard let re = try? NSRegularExpression(pattern: pattern) else {
            return "error: not a valid regular expression: \(pattern)"
        }

        do {
            let root = try guard_.resolve(path)
            var out: [String] = []
            var scanned = 0

            for file in walk(root) {
                if out.count >= maxGrepMatches { break }
                guard let data = FileManager.default.contents(atPath: file.path) else { continue }
                // A NUL in the first 8 KB is the usual binary heuristic, and it
                // is what keeps a .safetensors shard out of the context window.
                if data.prefix(8192).contains(0) { continue }
                scanned += 1

                let text = String(decoding: data, as: UTF8.self)
                var lineNo = 0
                for line in text.split(separator: "\n", omittingEmptySubsequences: false) {
                    lineNo += 1
                    if out.count >= maxGrepMatches { break }
                    let s = String(line)
                    let r = NSRange(s.startIndex..., in: s)
                    if re.firstMatch(in: s, range: r) != nil {
                        let shown = s.count > 300 ? String(s.prefix(300)) + "…" : s
                        out.append("\(guard_.display(file)):\(lineNo):\(shown)")
                    }
                }
            }

            if out.isEmpty { return "[no matches in \(scanned) files]" }
            if out.count >= maxGrepMatches {
                out.append("[stopped at \(maxGrepMatches) matches; narrow the pattern]")
            }
            return cap(out.joined(separator: "\n"))
        } catch {
            return "error: \(error)"
        }
    }

    /// Files under a root, skipping the directories nobody means to search.
    private func walk(_ root: URL) -> [URL] {
        var isDir: ObjCBool = false
        FileManager.default.fileExists(atPath: root.path, isDirectory: &isDir)
        if !isDir.boolValue { return [root] }

        let skip: Set<String> = [".git", "node_modules", ".build", "build",
                                 "DerivedData", ".venv", "__pycache__"]
        var files: [URL] = []
        let e = FileManager.default.enumerator(at: root,
                                               includingPropertiesForKeys: [.isDirectoryKey],
                                               options: [.skipsHiddenFiles])
        while let u = e?.nextObject() as? URL {
            if skip.contains(u.lastPathComponent) { e?.skipDescendants(); continue }
            let v = try? u.resourceValues(forKeys: [.isDirectoryKey])
            if v?.isDirectory != true { files.append(u) }
        }
        return files
    }
}

// MARK: - Parsing the model's XML calls

/// Wraps qw_tool_parse, which is the C agent's parser for the XML call format
/// this model was trained on.
///
/// PLAN.md 3.5: reuse it. It is pure string processing with no I/O and is
/// already covered by tests/test_toolcall; reimplementing the format in Swift
/// would be duplicating a tested parser for no gain.
public enum ToolParser {

    public enum Result: Sendable {
        case none
        case calls([ToolCall], preamble: String?)
        case malformed(String)
    }

    public static func parse(_ text: String) -> Result {
        var out = qw_tool_calls()
        let (n, err) = withErrorBuffer { buf, cap in
            qw_tool_parse(text, &out, buf, cap)
        }
        defer { if n > 0 { qw_tool_calls_free(&out) } }

        if n < 0 { return .malformed(err.isEmpty ? "malformed tool call" : err) }
        if n == 0 { return .none }

        var calls: [ToolCall] = []
        withUnsafePointer(to: &out.calls) { tuple in
            tuple.withMemoryRebound(to: qw_tool_call.self, capacity: Int(QW_MAX_CALLS)) { arr in
                for i in 0..<Int(n) {
                    let c = arr[i]
                    guard let namePtr = c.name else { continue }
                    var args: [String: String] = [:]
                    withUnsafePointer(to: c.params) { ptuple in
                        ptuple.withMemoryRebound(to: qw_tool_param.self,
                                                 capacity: Int(QW_MAX_PARAMS)) { params in
                            for j in 0..<Int(c.n_params) {
                                guard let k = params[j].key, let v = params[j].value else { continue }
                                args[String(cString: k)] = String(cString: v)
                            }
                        }
                    }
                    calls.append(ToolCall(name: String(cString: namePtr), arguments: args))
                }
            }
        }
        let preamble = out.preamble.map { String(cString: $0) }
        return .calls(calls, preamble: preamble?.isEmpty == false ? preamble : nil)
    }

    /// What can be told about a call that is still being written.
    ///
    /// The C parser handles complete calls only, and a call takes a long time to
    /// write: `write` or `define` can run to hundreds of tokens, which at ~6
    /// tok/s is minutes during which the transcript shows nothing at all,
    /// because the markup is deliberately not echoed. The name and the parameter
    /// keys arrive early and are enough to say what is happening.
    ///
    /// Scanning from the last `<tool_call>` rather than the whole buffer: this
    /// runs during generation, and re-reading an accumulating string on every
    /// token is the kind of quadratic that only shows up on the longest calls --
    /// exactly the ones this exists to narrate.
    public static func partial(_ text: String) -> (name: String?, keys: [String]) {
        let scope = text.range(of: "<tool_call>", options: .backwards)
            .map { String(text[$0.upperBound...]) } ?? text
        var name: String?
        if let f = scope.range(of: "<function="),
           let close = scope[f.upperBound...].firstIndex(of: ">") {
            name = String(scope[f.upperBound..<close])
        }
        var keys: [String] = []
        var cursor = scope.startIndex
        while let p = scope.range(of: "<parameter=", range: cursor..<scope.endIndex) {
            guard let close = scope[p.upperBound...].firstIndex(of: ">") else { break }
            keys.append(String(scope[p.upperBound..<close]))
            cursor = close
        }
        return (name, keys)
    }

    /// True once the text holds a complete call, so generation can stop at the
    /// closing tag rather than running to the token budget.
    public static func isComplete(_ text: String) -> Bool {
        let bytes = Array(text.utf8)
        return bytes.withUnsafeBufferPointer { p in
            p.baseAddress.map { base in
                base.withMemoryRebound(to: CChar.self, capacity: p.count) {
                    qw_tool_call_complete($0, p.count)
                }
            } ?? false
        }
    }
}
