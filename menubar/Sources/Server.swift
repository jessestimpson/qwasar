import Darwin
import Foundation

/// What the menu bar shows.  Derived, never assumed: `listening` means a
/// connection to the port just succeeded, not that a process was launched.
enum ServerState: Equatable {
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
final class ServerController {
    static let defaultPort = 8080

    var onChange: (() -> Void)?

    private(set) var state: ServerState = .stopped {
        didSet { if state != oldValue { onChange?() } }
    }

    private var process: Process?
    private var lifeline: Pipe?
    private var stopRequested = false
    private var afterExit: [() -> Void] = []
    private var timer: Timer?

    let logURL: URL = {
        let dir = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Logs/Qwasar Server", isDirectory: true)
        return dir.appendingPathComponent("server.log")
    }()

    var isRunning: Bool { process != nil }

    var port: Int {
        get {
            let p = UserDefaults.standard.integer(forKey: "port")
            return (1...65535).contains(p) ? p : Self.defaultPort
        }
        set { UserDefaults.standard.set(newValue, forKey: "port") }
    }

    var apiURL: String { "http://127.0.0.1:\(port)/v1" }

    // MARK: model

    /// The model folder: the one chosen in the menu, else the one the build
    /// found beside the checkout, else LM Studio's usual place.
    var modelPath: String? {
        var candidates: [String] = []
        if let chosen = UserDefaults.standard.string(forKey: "modelPath") { candidates.append(chosen) }
        if let built = Bundle.main.object(forInfoDictionaryKey: "QWDefaultModel") as? String,
           !built.isEmpty {
            candidates.append(built)
        }
        candidates.append(NSString(string:
            "~/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-4bit").expandingTildeInPath)
        return candidates.first(where: Self.isModelFolder)
    }

    func setModelPath(_ path: String) {
        UserDefaults.standard.set(path, forKey: "modelPath")
    }

    nonisolated static func isModelFolder(_ path: String) -> Bool {
        FileManager.default.fileExists(atPath: (path as NSString).appendingPathComponent("config.json"))
    }

    // MARK: lifecycle

    init() {
        let t = Timer(timeInterval: 1.0, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated { self?.refresh() }
        }
        RunLoop.main.add(t, forMode: .common)   // keeps ticking while the menu is open
        timer = t
        refresh()
    }

    func start() {
        guard process == nil else { return }
        guard let model = modelPath else {
            state = .failed("No model folder chosen")
            return
        }
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
            FileManager.default.createFile(atPath: logURL.path, contents: nil)   // truncates
            log = try FileHandle(forWritingTo: logURL)
        } catch {
            state = .failed("Cannot write the log: \(error.localizedDescription)")
            return
        }

        let p = Process()
        p.executableURL = binary
        // The whole context the model was trained for, and output limited only
        // by what that context has left -- the server's defaults (32K, and 2048
        // tokens for a request that names no limit) suit a terminal, not an
        // agent working through a codebase.  The KV cache is sized to the
        // context up front: at 262K about 8 GB for Flash-Next, 17 GB for the 27B.
        var args = ["-m", model, "--port", String(port), "--exit-on-eof", "--max-tokens", "0"]
        if let ctx = ModelCatalog.maxContext(of: model) { args += ["--ctx", String(ctx)] }
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
    func stop(then: (() -> Void)? = nil) {
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

    func restart() {
        stop { [weak self] in self?.start() }
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
    func refresh() {
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
            .map { $0.replacingOccurrences(of: "qwasar-server: ", with: "") }
    }

    // MARK: the probe

    /// A non-blocking connect to 127.0.0.1:port.  Loopback answers at once --
    /// accepted or refused -- so the 250 ms bound only matters if the
    /// listener's backlog is full.
    nonisolated static func isListening(port: Int) -> Bool {
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
