// ToolBackends.swift -- where ToolKit's tools act: this Mac.

import Darwin
import Foundation

/// The user's own filesystem, as the user, with their shell environment.
public struct HostBackend: ToolBackend {
    public let workDir: String
    public let environment: [String: String]
    public let hasRipgrep: Bool

    public init(root: URL, environment: [String: String]) {
        self.workDir = URL(fileURLWithPath: root.path).resolvingSymlinksInPath().path
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
        return URL(fileURLWithPath: abs).standardizedFileURL.path
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
                          timeout: TimeInterval(timeoutMS) / 1000, maxBytes: maxBytes,
                          interrupt: .shared)
        var out = String(decoding: r.output, as: UTF8.self)
        if r.truncated { out += "\n[output stopped at \(maxBytes / 1024 / 1024) MB]" }
        return .success(ExecResult(status: r.status, output: out, timedOut: r.timedOut,
                                   interrupted: r.interrupted))
    }
}
