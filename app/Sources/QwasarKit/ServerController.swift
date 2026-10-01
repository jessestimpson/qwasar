// ServerController.swift -- qwasar-server as the app's helper process.
//
// Came from the menu bar app, unchanged in what it watches: the listen
// socket, probed once a second, is the truth about whether a client can
// connect; the process's own exit is caught the moment it happens; and a
// pipe on the server's stdin ends it whenever this app ends, however it ends.

import Darwin
import Foundation

/// What the menu bar shows.  Derived, never assumed: `listening` means a
/// connection to the port just succeeded, not that a process was launched.
public enum ServerState: Equatable, Sendable {
    case stopped
    case starting            // our server is running but not accepting yet: the model is loading
    case listening
    case stopping
    case portBusy            // something else is accepting on the port
    case failed(String)      // our server exited on its own; the reason from its log
}

/// Runs qwasar-server as a child process and watches its listen socket.
///
/// The state comes from two sources.  The socket is probed once a second --
/// a plain connect to 127.0.0.1 -- because that is the thing a client
/// depends on, and a process can be alive without accepting (loading the
/// model) or accepting without being ours (a server started by hand).  The
/// process's own exit is caught the moment it happens, so a crash turns the
/// indicator off without waiting for the next probe.
///
/// The server is started with --exit-on-eof and a pipe on its stdin that this
/// side never writes to.  Whatever ends this app -- Quit, a crash, kill -9 --
/// closes the pipe, and the server exits with it rather than holding the port
/// with no icon left to say so.
@MainActor
public final class ServerController {
    public static let defaultPort = 8080

    public var onChange: (() -> Void)?

    public private(set) var state: ServerState = .stopped {
        didSet { if state != oldValue { onChange?() } }
    }

    private var process: Process?
    private var lifeline: Pipe?
    private var stopRequested = false
    private var afterExit: [() -> Void] = []
    private var timer: Timer?

    public let logURL: URL = {
        let dir = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Logs/Qwasar Server", isDirectory: true)
        return dir.appendingPathComponent("server.log")
    }()

    public var isRunning: Bool { process != nil }

    public var port: Int {
        get {
            let p = UserDefaults.standard.integer(forKey: "port")
            return (1...65535).contains(p) ? p : Self.defaultPort
        }
        set { UserDefaults.standard.set(newValue, forKey: "port") }
    }

    /// Overrides for what the server would derive from the model and the
    /// machine (`--ctx`, `--live`).  nil -- the default -- lets it derive.
    public var contextOverride: Int? {
        get { let v = UserDefaults.standard.integer(forKey: "serverContext"); return v > 0 ? v : nil }
        set { UserDefaults.standard.set(newValue ?? 0, forKey: "serverContext") }
    }
    public var liveSessionsOverride: Int? {
        get { let v = UserDefaults.standard.integer(forKey: "serverLive"); return v > 0 ? v : nil }
        set { UserDefaults.standard.set(newValue ?? 0, forKey: "serverLive") }
    }

    public var apiURL: String { "http://127.0.0.1:\(port)/v1" }
    public var baseURL: URL { URL(string: "http://127.0.0.1:\(port)/")! }

    /// The model the server was last started with, for the menu.
    public private(set) var modelPath: String?
    /// Where the server keeps its sessions: beside the app's own store, in a
    /// directory of its own, so the two never write into one another.
    public var stateDir: URL?

    nonisolated public static func isModelFolder(_ path: String) -> Bool {
        FileManager.default.fileExists(atPath: (path as NSString).appendingPathComponent("config.json"))
    }

    // MARK: lifecycle

    public init() {
        let t = Timer(timeInterval: 1.0, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated { self?.refresh() }
        }
        RunLoop.main.add(t, forMode: .common)   // keeps ticking while the menu is open
        timer = t
        refresh()
    }

    public func start(model: String) {
        guard process == nil else { return }
        modelPath = model
        if Self.isListening(port: port) {
            state = .portBusy
            return
        }
        guard let binary = Bundle.main.url(forAuxiliaryExecutable: "qwasar-server") else {
            state = .failed("qwasar-server is missing from the app bundle")
            return
        }

        let log: FileHandle
        do {
            try FileManager.default.createDirectory(at: logURL.deletingLastPathComponent(),
                                                    withIntermediateDirectories: true)
            // The previous run's log is kept beside this one: it is where the
            // evidence is when something went wrong and the server was
            // restarted since -- which relaunching the app always does.
            let previous = logURL.deletingLastPathComponent().appendingPathComponent("server.previous.log")
            if FileManager.default.fileExists(atPath: logURL.path) {
                try? FileManager.default.removeItem(at: previous)
                try? FileManager.default.moveItem(at: logURL, to: previous)
            }
            FileManager.default.createFile(atPath: logURL.path, contents: nil)
            log = try FileHandle(forWritingTo: logURL)
        } catch {
            state = .failed("Cannot write the log: \(error.localizedDescription)")
            return
        }

        let p = Process()
        p.executableURL = binary
        // The server derives its context and live sessions from the model and
        // the machine (API.md §4.1) unless the user set them (contextOverride,
        // liveSessionsOverride -- the config session's server_context and
        // server_live_sessions).  --max-tokens 0 leaves a
        // compat request's output bounded only by the window's room.
        var args = ["-m", model, "--port", String(port), "--exit-on-eof", "--max-tokens", "0", "-v"]
        if let stateDir { args += ["--state-dir", stateDir.path] }
        if let c = contextOverride { args += ["--ctx", String(c)] }
        if let l = liveSessionsOverride { args += ["--live", String(l)] }
        p.arguments = args
        let pipe = Pipe()
        p.standardInput = pipe
        p.standardOutput = log
        p.standardError = log
        p.terminationHandler = { [weak self] proc in
            let status = proc.terminationStatus
            Task { @MainActor in self?.exited(status: status) }
        }
        do {
            try p.run()
        } catch {
            state = .failed("Could not launch: \(error.localizedDescription)")
            return
        }
        try? log.close()           // the child has its own descriptor now
        process = p
        lifeline = pipe
        stopRequested = false
        state = .starting
    }

    /// Asks the server to exit, and makes sure it does.  `then` runs once it
    /// has gone -- straight away if it was not running.  On the way out the
    /// server writes the live conversation to disk (a checkpoint the next run
    /// resumes from), which for a long conversation at a large context is
    /// gigabytes, hence the minute before it is killed.
    public func stop(then: (() -> Void)? = nil) {
        guard let p = process else { then?(); return }
        if let then { afterExit.append(then) }
        stopRequested = true
        state = .stopping
        p.terminate()                                   // SIGTERM
        try? lifeline?.fileHandleForWriting.close()     // and EOF, belt and braces
        let pid = p.processIdentifier
        Task { @MainActor [weak self] in
            try? await Task.sleep(for: .seconds(60))
            if let self, self.process === p { kill(pid, SIGKILL) }
        }
    }

    public func restart() {
        guard let model = modelPath else { return }
        stop { [weak self] in self?.start(model: model) }
    }

    private func exited(status: Int32) {
        process = nil
        lifeline = nil
        if stopRequested {
            state = .stopped
        } else {
            state = .failed(lastLogLine() ?? "exited with status \(status)")
        }
        stopRequested = false
        let pending = afterExit
        afterExit = []
        pending.forEach { $0() }
    }

    /// The indicator's source of truth: is the port accepting right now?
    public func refresh() {
        let up = Self.isListening(port: port)
        switch state {
        case .stopping:
            break                                   // settles when the process exits
        case .starting, .listening:
            if process != nil { state = up ? .listening : .starting }
        case .stopped, .portBusy:
            state = up ? .portBusy : .stopped
        case .failed:
            if up { state = .portBusy }             // keep the reason until something changes
        }
    }

    private func lastLogLine() -> String? {
        guard let text = try? String(contentsOf: logURL, encoding: .utf8) else { return nil }
        return text.split(whereSeparator: \.isNewline)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .last(where: { !$0.isEmpty })
            .map { line -> String in
                // Log lines carry a "[date time] " stamp; the reason is what follows.
                var text = line
                if text.hasPrefix("["), let close = text.firstIndex(of: "]") {
                    text = String(text[text.index(after: close)...]).trimmingCharacters(in: .whitespaces)
                }
                return text.replacingOccurrences(of: "qwasar-server: ", with: "")
            }
    }

    // MARK: the probe

    /// A non-blocking connect to 127.0.0.1:port.  Loopback answers at once --
    /// accepted or refused -- so the 250 ms bound only matters if the
    /// listener's backlog is full.
    nonisolated public static func isListening(port: Int) -> Bool {
        let fd = socket(AF_INET, SOCK_STREAM, 0)
        guard fd >= 0 else { return false }
        defer { close(fd) }
        _ = fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) | O_NONBLOCK)
        var one: Int32 = 1
        setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, socklen_t(MemoryLayout<Int32>.size))

        var addr = sockaddr_in()
        addr.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = in_port_t(UInt16(truncatingIfNeeded: port).bigEndian)
        addr.sin_addr.s_addr = inet_addr("127.0.0.1")
        let r = withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                connect(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        if r == 0 { return true }
        guard errno == EINPROGRESS else { return false }

        var pfd = pollfd(fd: fd, events: Int16(POLLOUT), revents: 0)
        guard poll(&pfd, 1, 250) == 1 else { return false }
        var err: Int32 = 0
        var len = socklen_t(MemoryLayout<Int32>.size)
        getsockopt(fd, SOL_SOCKET, SO_ERROR, &err, &len)
        return err == 0
    }
}
