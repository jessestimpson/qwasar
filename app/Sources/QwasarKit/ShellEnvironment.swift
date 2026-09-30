// ShellEnvironment.swift -- the user's shell environment, for tools that run
// on the host.
//
// An app launched from the Finder or the Dock inherits launchd's environment,
// not the user's: PATH is /usr/bin:/bin:/usr/sbin:/sbin, and nothing their
// shell's startup files add -- Homebrew, mise, cargo, a project's toolchain --
// is on it.  The standard fix on macOS, and what editors that run a user's
// tools do, is to ask the user's own login shell: run it as a login,
// interactive shell (so it reads .zprofile and .zshrc, or bash's and fish's
// equivalents, exactly as a Terminal window would), have it print its
// environment, and use that.  Resolved in the project's directory, so a
// directory-aware activation like mise's picks the project's own versions.
//
// Resolved once per directory and cached: a login shell can take a second or
// more to start, and the environment does not change under a session.

import Darwin
import Foundation

public enum ShellEnvironment {

    /// The user's login shell, from the directory service -- what Terminal
    /// uses -- falling back to $SHELL, then zsh, the macOS default.
    public static var loginShell: String {
        if let pw = getpwuid(getuid()), let sh = pw.pointee.pw_shell {
            let s = String(cString: sh)
            if !s.isEmpty, FileManager.default.isExecutableFile(atPath: s) { return s }
        }
        if let s = ProcessInfo.processInfo.environment["SHELL"], !s.isEmpty { return s }
        return "/bin/zsh"
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var cache: [String: [String: String]] = [:]

    /// The environment a command in `directory` should run with.  Never
    /// fails: if the shell cannot be asked, the app's own environment with
    /// the system's standard PATH (path_helper's) stands in.
    public static func resolve(in directory: URL, timeout: TimeInterval = 15) -> [String: String] {
        let key = directory.standardizedFileURL.path
        if let hit = lock.withLock({ cache[key] }) { return hit }
        let env = ask(loginShell, in: directory, timeout: timeout) ?? fallback()
        lock.withLock { cache[key] = env }
        return env
    }

    /// Forgets what was resolved, so the next session sees edits to the
    /// user's startup files.
    public static func invalidate() { lock.withLock { cache.removeAll() } }

    // MARK: -

    private static let marker = "__QWASAR_ENV_BEGIN__"

    /// `<shell> -l -i -c 'printf marker; env -0'` from a clean environment.
    /// The marker skips anything the startup files print (a motd, a
    /// greeting); `env -0` is NUL-separated, so values with newlines survive.
    private static func ask(_ shell: String, in directory: URL, timeout: TimeInterval) -> [String: String]? {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: shell)
        p.arguments = ["-l", "-i", "-c", "printf '%s' \(marker); /usr/bin/env -0"]
        p.currentDirectoryURL = directory
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        let user = NSUserName()
        var base: [String: String] = [
            "HOME": home, "USER": user, "LOGNAME": user, "SHELL": shell,
            "TERM": "dumb", "PWD": directory.path,
        ]
        for k in ["TMPDIR", "LANG", "LC_ALL", "__CF_USER_TEXT_ENCODING", "SSH_AUTH_SOCK"] {
            if let v = ProcessInfo.processInfo.environment[k] { base[k] = v }
        }
        base["PATH"] = "/usr/bin:/bin:/usr/sbin:/sbin"
        p.environment = base
        let out = Pipe()
        p.standardOutput = out
        p.standardError = FileHandle.nullDevice
        p.standardInput = FileHandle.nullDevice

        let data = DataBox()
        let reader = DispatchSemaphore(value: 0)
        DispatchQueue.global().async {
            data.set(out.fileHandleForReading.readDataToEndOfFile())
            reader.signal()
        }
        do { try p.run() } catch { return nil }
        if reader.wait(timeout: .now() + timeout) == .timedOut {
            p.terminate()
            return nil
        }
        p.waitUntilExit()

        let raw = data.get()
        guard let m = raw.range(of: Data(marker.utf8)) else { return nil }
        var env: [String: String] = [:]
        for entry in raw[m.upperBound...].split(separator: 0) {
            guard let s = String(data: Data(entry), encoding: .utf8),
                  let eq = s.firstIndex(of: "=") else { continue }
            env[String(s[..<eq])] = String(s[s.index(after: eq)...])
        }
        // Shell-session bookkeeping that means nothing to a command run later.
        for k in ["SHLVL", "PWD", "OLDPWD", "_"] { env.removeValue(forKey: k) }
        return env["PATH"] == nil ? nil : env
    }

    /// The app's environment, with PATH as /usr/libexec/path_helper builds it
    /// for a login shell (/etc/paths and /etc/paths.d).
    private static func fallback() -> [String: String] {
        var env = ProcessInfo.processInfo.environment
        var path = ["/usr/local/bin", "/usr/bin", "/bin", "/usr/sbin", "/sbin"]
        var extra: [String] = []
        if let s = try? String(contentsOfFile: "/etc/paths", encoding: .utf8) {
            path = s.split(separator: "\n").map(String.init).filter { !$0.isEmpty }
        }
        if let d = try? FileManager.default.contentsOfDirectory(atPath: "/etc/paths.d") {
            for f in d.sorted() {
                if let s = try? String(contentsOfFile: "/etc/paths.d/" + f, encoding: .utf8) {
                    extra += s.split(separator: "\n").map(String.init).filter { !$0.isEmpty }
                }
            }
        }
        env["PATH"] = (path + extra + ["/opt/homebrew/bin"]).joined(separator: ":")
        return env
    }
}

private final class DataBox: @unchecked Sendable {
    private var value = Data()
    private let lock = NSLock()
    func set(_ d: Data) { lock.withLock { value = d } }
    func get() -> Data { lock.withLock { value } }
}
