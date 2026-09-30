// HostToolRunner.swift -- tool calls, run directly on the user's Mac.
//
// A session is sandboxed or not, chosen when it is created.  This is "not":
// the core tools (ToolKit) against the user's own filesystem, as the user,
// with the user's shell environment (ShellEnvironment) -- so Bash finds what
// their Terminal finds: Homebrew, mise, cargo, the project's pinned
// toolchain.  Nothing is confined.  Relative paths are the project's;
// absolute and ~ paths are taken as given, because a session that can run any
// command gains nothing from a file tool that cannot follow it.

import Darwin
import Foundation

public struct HostToolRunner: ToolExecuting {
    public let kit: ToolKit
    public var root: String { kit.backend.workDir }

    /// `timeout` is Bash's default, in seconds; `resultCap` the model's
    /// (ToolKit.resultCap(flashNext:)).
    public init(root: URL, environment: [String: String], timeout: Int = 120,
                resultCap: Int = ToolKit.resultCap(flashNext: true)) {
        kit = ToolKit(backend: HostBackend(root: root, environment: environment),
                      resultCap: resultCap, defaultTimeoutMS: timeout * 1000)
    }

    public var schemas: [String] { ToolSurface.coreSchemas }

    public var environmentDescription: String {
        """
        # Where you are working

        Your tools run directly on the user's Mac, as the user -- there is no sandbox. Relative paths resolve against the project directory; absolute paths and ~ work too. Bash runs /bin/sh in the project directory with the user's login-shell environment, so their PATH, toolchains and network are all available.

        Every change is real and immediate: edits land in the user's files and commands act on their machine. Do not run destructive or irreversible commands -- deleting files the task does not need deleted, force-pushing, rewriting git history, installing or removing software system-wide -- unless the user asked for exactly that.
        """ + (kit.backend.hasRipgrep ? "" : """


        ripgrep is not installed here, so Grep runs grep -E: POSIX extended regular expressions, where \\d and (?i) do not work -- use [0-9] and the -i parameter.
        """)
    }

    public func run(_ call: ToolCall) -> String {
        guard ToolSurface.coreNames.contains(call.name) else {
            return "error: no such tool: \(call.name). Available: "
                 + ToolSurface.coreNames.sorted().joined(separator: ", ")
        }
        return kit.run(call)
    }
}

/// posix_spawn with its own process group, so a timeout kills the command
/// and everything it started -- a Process would leave the grandchildren.
enum Spawn {
    struct Result {
        var status: Int32 = -1
        var output = Data()
        var timedOut = false
        var truncated = false
    }

    static func run(_ exe: String, _ args: [String], cwd: URL, env: [String: String],
                    timeout: TimeInterval, maxBytes: Int) -> Result {
        var result = Result()
        var fds: [Int32] = [0, 0]
        guard pipe(&fds) == 0 else { return result }
        let (rd, wr) = (fds[0], fds[1])

        var fa: posix_spawn_file_actions_t? = nil
        posix_spawn_file_actions_init(&fa)
        defer { posix_spawn_file_actions_destroy(&fa) }
        posix_spawn_file_actions_addopen(&fa, 0, "/dev/null", O_RDONLY, 0)
        posix_spawn_file_actions_adddup2(&fa, wr, 1)
        posix_spawn_file_actions_adddup2(&fa, wr, 2)
        posix_spawn_file_actions_addclose(&fa, rd)
        posix_spawn_file_actions_addclose(&fa, wr)
        posix_spawn_file_actions_addchdir_np(&fa, cwd.path)

        var attr: posix_spawnattr_t? = nil
        posix_spawnattr_init(&attr)
        defer { posix_spawnattr_destroy(&attr) }
        posix_spawnattr_setflags(&attr, Int16(POSIX_SPAWN_SETPGROUP | POSIX_SPAWN_CLOEXEC_DEFAULT))
        posix_spawnattr_setpgroup(&attr, 0)

        let argv = ([exe] + args).map { strdup($0) } + [nil]
        let envp = env.map { strdup("\($0.key)=\($0.value)") } + [nil]
        defer { argv.forEach { free($0) }; envp.forEach { free($0) } }

        var pid: pid_t = 0
        let rc = posix_spawn(&pid, exe, &fa, &attr, argv, envp)
        close(wr)
        guard rc == 0 else {
            close(rd)
            result.output = Data("cannot run \(exe): \(String(cString: strerror(rc)))".utf8)
            return result
        }

        // The reader drains the pipe until EOF -- which a command that leaves
        // a background process holding it may never give -- so the wait is on
        // the process, and the reader gets a short grace after it exits.
        let box = SpawnBox()
        let readDone = DispatchSemaphore(value: 0)
        DispatchQueue.global().async {
            var buf = [UInt8](repeating: 0, count: 65536)
            while true {
                let n = Darwin.read(rd, &buf, buf.count)
                if n <= 0 { break }
                // Past the limit the bytes are dropped, not left in the
                // pipe: a writer blocked on a full pipe never exits.
                box.append(buf[0..<n], limit: maxBytes)
            }
            readDone.signal()
        }

        let exited = DispatchSemaphore(value: 0)
        let statusBox = SpawnBox()
        let child = pid
        DispatchQueue.global().async {
            var st: Int32 = 0
            while waitpid(child, &st, 0) == -1 && errno == EINTR {}
            statusBox.status = st
            exited.signal()
        }
        if exited.wait(timeout: .now() + timeout) == .timedOut {
            kill(-pid, SIGKILL)
            result.timedOut = true
            exited.wait()
        }
        _ = readDone.wait(timeout: .now() + 1)
        close(rd)

        let st = statusBox.status
        // WIFEXITED / WEXITSTATUS, which Swift does not import.
        result.status = (st & 0x7f) == 0 ? (st >> 8) & 0xff : 128 + (st & 0x7f)
        result.output = box.data
        result.truncated = box.truncated
        return result
    }
}

private final class SpawnBox: @unchecked Sendable {
    private let lock = NSLock()
    private var _data = Data()
    private var _truncated = false
    private var _status: Int32 = 0
    var data: Data { lock.withLock { _data } }
    var truncated: Bool { lock.withLock { _truncated } }
    var status: Int32 {
        get { lock.withLock { _status } }
        set { lock.withLock { _status = newValue } }
    }
    /// Keeps up to `limit` bytes; anything past it is counted as truncation.
    func append(_ bytes: ArraySlice<UInt8>, limit: Int) {
        lock.withLock {
            let room = limit - _data.count
            if bytes.count > room {
                _data.append(contentsOf: bytes.prefix(max(0, room)))
                _truncated = true
            } else {
                _data.append(contentsOf: bytes)
            }
        }
    }
}
