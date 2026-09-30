// ToolBackends.swift -- where ToolKit's tools act: this Mac, or the guest.

import Darwin
import Foundation

/// The user's own filesystem, as the user, with their shell environment.
///
/// `confined` keeps every path inside the working directory -- the read-only
/// stand-in a sandboxed session falls back to when its guest cannot start
/// promised that, and PathGuard is how it keeps the promise.
public struct HostBackend: ToolBackend {
    public let workDir: String
    public let environment: [String: String]
    public let confined: Bool
    public let hasRipgrep: Bool

    public init(root: URL, environment: [String: String], confined: Bool = false) {
        self.workDir = URL(fileURLWithPath: root.path).resolvingSymlinksInPath().path
        self.confined = confined
        var env = environment
        // No pagers, no colour, nothing waiting on a terminal that is not there.
        env["TERM"] = "dumb"
        env["PAGER"] = "cat"
        env["GIT_PAGER"] = "cat"
        env["NO_COLOR"] = "1"
        env["PWD"] = workDir
        self.environment = env
        let path = env["PATH"] ?? "/usr/bin:/bin"
        self.hasRipgrep = path.split(separator: ":").contains {
            FileManager.default.isExecutableFile(atPath: String($0) + "/rg")
        }
    }

    public func resolve(_ raw: String) -> String? {
        if raw.utf8.contains(0) { return nil }
        let t = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        let abs: String
        if t.isEmpty || t == "." { abs = workDir }
        else if t == "~" || t.hasPrefix("~/") { abs = (t as NSString).expandingTildeInPath }
        else if t.hasPrefix("/") { abs = t }
        else { abs = workDir + "/" + t }
        let std = URL(fileURLWithPath: abs).standardizedFileURL.path
        if confined {
            let g = PathGuard(root: URL(fileURLWithPath: workDir))
            guard g.contains(URL(fileURLWithPath: std).resolvingSymlinksInPath()) else { return nil }
        }
        return std
    }

    public func stat(_ path: String) -> Result<PathKind, ToolFailure> {
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: path, isDirectory: &isDir) else { return .success(.missing) }
        if isDir.boolValue { return .success(.directory) }
        let size = (try? FileManager.default.attributesOfItem(atPath: path))?[.size] as? Int ?? 0
        return .success(.file(size: size))
    }

    public func read(_ path: String, maxBytes: Int) -> Result<(size: Int, text: String), ToolFailure> {
        guard let h = FileHandle(forReadingAtPath: path) else { return .failure(ToolFailure("cannot read \(path)")) }
        defer { try? h.close() }
        let data = (try? h.read(upToCount: maxBytes)) ?? Data()
        if data.contains(0) { return .failure(ToolFailure("\(path) is a binary file")) }
        let size = (try? FileManager.default.attributesOfItem(atPath: path))?[.size] as? Int ?? data.count
        return .success((size, String(decoding: data, as: UTF8.self)))
    }

    public func write(_ path: String, _ content: String) -> Result<Void, ToolFailure> {
        if confined { return .failure(ToolFailure("this session is read-only")) }
        do {
            try FileManager.default.createDirectory(at: URL(fileURLWithPath: path).deletingLastPathComponent(),
                                                    withIntermediateDirectories: true)
            try Data(content.utf8).write(to: URL(fileURLWithPath: path))
            return .success(())
        } catch {
            return .failure(ToolFailure("cannot write \(path): \(error.localizedDescription)"))
        }
    }

    public func exec(_ command: String, timeoutMS: Int, maxBytes: Int) -> Result<ExecResult, ToolFailure> {
        let r = Spawn.run("/bin/sh", ["-c", command], cwd: URL(fileURLWithPath: workDir), env: environment,
                          timeout: TimeInterval(timeoutMS) / 1000, maxBytes: maxBytes)
        var out = String(decoding: r.output, as: UTF8.self)
        if r.truncated { out += "\n[output stopped at \(maxBytes / 1024 / 1024) MB]" }
        return .success(ExecResult(status: r.status, output: out, timedOut: r.timedOut))
    }
}

/// The sandbox's guest, through the warden's primitives.  Paths are the
/// guest's: /work is the project, and the warden refuses anything outside it.
public struct GuestBackend: ToolBackend {
    let channel: VsockChannel
    public let workDir = "/work"
    public let hasRipgrep = true

    public init(channel: VsockChannel) { self.channel = channel }

    public func resolve(_ raw: String) -> String? {
        if raw.utf8.contains(0) { return nil }
        let t = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if t.isEmpty || t == "." { return workDir }
        let abs = t.hasPrefix("/") ? t : workDir + "/" + t
        return URL(fileURLWithPath: abs).standardizedFileURL.path
    }

    public func stat(_ path: String) -> Result<PathKind, ToolFailure> {
        call("stat", ["path": .string(path)], timeout: 20).map { s in
            if s == "dir" { return .directory }
            if s.hasPrefix("file ") { return .file(size: Int(s.dropFirst(5)) ?? 0) }
            return .missing
        }
    }

    public func read(_ path: String, maxBytes: Int) -> Result<(size: Int, text: String), ToolFailure> {
        call("read_raw", ["path": .string(path), "max_bytes": .int(maxBytes)], timeout: 60).map { s in
            guard let nl = s.firstIndex(of: "\n") else { return (0, "") }
            return (Int(s[..<nl]) ?? 0, String(s[s.index(after: nl)...]))
        }
    }

    public func write(_ path: String, _ content: String) -> Result<Void, ToolFailure> {
        call("write", ["path": .string(path), "content": .string(content)], timeout: 60).map { _ in () }
    }

    public func exec(_ command: String, timeoutMS: Int, maxBytes: Int) -> Result<ExecResult, ToolFailure> {
        let r = call("exec", ["command": .string(command), "timeout_ms": .int(timeoutMS),
                              "max_bytes": .int(maxBytes)],
                     timeout: timeoutMS / 1000 + 30, timeoutIsResult: true)
        switch r {
        case .failure(let f) where f.message.hasPrefix("timeout:"):
            return .success(ExecResult(status: -1, output: "", timedOut: true))
        case .failure(let f): return .failure(f)
        case .success(let s):
            guard let nl = s.firstIndex(of: "\n") else { return .success(ExecResult(status: Int32(s) ?? -1, output: "", timedOut: false)) }
            return .success(ExecResult(status: Int32(s[..<nl]) ?? -1,
                                       output: String(s[s.index(after: nl)...]), timedOut: false))
        }
    }

    /// One warden op, synchronously -- ToolExecuting is synchronous, and the
    /// channel's reader is its own thread, so this cannot deadlock against it.
    private func call(_ op: String, _ args: [String: GuestValue], timeout: Int,
                      timeoutIsResult: Bool = false) -> Result<String, ToolFailure> {
        let box = GuestBox()
        let done = DispatchSemaphore(value: 0)
        Task.detached { [channel] in
            do {
                let r = try await channel.send(op: op, args: args, timeout: timeout)
                if r.ok == true { box.set(.success(r.result ?? "")) }
                else {
                    let kind = r.kind ?? "error"
                    box.set(.failure(ToolFailure("\(kind): \(r.error ?? "the guest gave no reason")")))
                }
            } catch {
                box.set(.failure(ToolFailure("\(error)")))
            }
            done.signal()
        }
        done.wait()
        let r = box.get()
        // The warden's own refusals read "<kind>: <reason>"; the model needs the reason.
        if case .failure(let f) = r, !timeoutIsResult || !f.message.hasPrefix("timeout:"),
           let colon = f.message.firstIndex(of: ":"),
           ["path", "io", "args", "binary", "isdir"].contains(String(f.message[..<colon])) {
            return .failure(ToolFailure(String(f.message[f.message.index(after: colon)...])
                .trimmingCharacters(in: .whitespaces)))
        }
        return r
    }
}

private final class GuestBox: @unchecked Sendable {
    private var value: Result<String, ToolFailure> = .failure(ToolFailure("no answer"))
    private let lock = NSLock()
    func set(_ v: Result<String, ToolFailure>) { lock.withLock { value = v } }
    func get() -> Result<String, ToolFailure> { lock.withLock { value } }
}
