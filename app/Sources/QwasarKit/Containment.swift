// Containment.swift -- Bash under Seatbelt.  Spec §16.5.
//
// ToolPolicy judges what a command says it will do; this bounds what it can
// do.  `make` runs whatever the Makefile says, and the model can write the
// Makefile first: no parser sees through that, so every command runs under
// sandbox-exec with a profile that lets it write only the project, the temp
// directories and the toolchains' caches, and reach no network but this
// Mac's own.  `rm -rf ~` hidden in a build script fails in the kernel.
//
// An approved question widens the profile by exactly what was approved --
// network for an approved curl, one path for an approved cp -- and the
// grant travels as a task-local, never as an argument the model could pass.
//
// Other processes are the hard part: a daemon that acts for a command acts
// outside its sandbox.  Measured with canaries (ContainmentSuite): launchd
// refuses jobs from a sandboxed caller, by `submit` or by `bootstrap` of a
// plist the command wrote; cfprefsd does not -- `defaults write` landed --
// so preference writes are denied; and Apple Events got through even with
// appleevent-send and the AE server's mach service denied, so the senders
// themselves (osascript, automator, shortcuts) may not be executed.  Any
// other program that sends Apple Events -- Python with PyObjC, say -- is
// not covered; the policy asks before the direct ones.

import Darwin
import Foundation

/// What an approved call may do beyond the default profile.
public struct ContainmentGrant: Sendable, Equatable, Hashable {
    public var network = false
    /// Paths it may write, besides the default writable set.
    public var paths: [String] = []
    /// Write anywhere the user can: an approved install, a write to a path
    /// computed at run time.
    public var anywhere = false
    /// Have other processes act for it, or act on them: send Apple Events
    /// (osascript, automator, shortcuts), write preferences through
    /// cfprefsd, signal processes outside its own sandbox.
    public var ipc = false

    public init(network: Bool = false, paths: [String] = [], anywhere: Bool = false, ipc: Bool = false) {
        self.network = network
        self.paths = paths
        self.anywhere = anywhere
        self.ipc = ipc
    }

    public static let none = ContainmentGrant()

    public static func + (a: ContainmentGrant, b: ContainmentGrant) -> ContainmentGrant {
        ContainmentGrant(network: a.network || b.network,
                         paths: a.paths + b.paths.filter { !a.paths.contains($0) },
                         anywhere: a.anywhere || b.anywhere, ipc: a.ipc || b.ipc)
    }
}

public enum ContainmentGrants {
    /// The grant for the command running on this task, set by the gate
    /// around an approved call.
    @TaskLocal public static var current = ContainmentGrant.none
}

public struct Containment: Sendable {
    public static let sandboxExec = "/usr/bin/sandbox-exec"
    public static var available: Bool { FileManager.default.isExecutableFile(atPath: sandboxExec) }

    /// Whether this process already runs in a sandbox -- the kernel's
    /// answer (sandbox_check on our own pid), not an environment variable
    /// anyone could set.  Profiles do not nest, so inside one, commands run
    /// plainly and the outer sandbox is what confines them: the case of
    /// `make test` run by a contained agent.
    public static let alreadySandboxed: Bool = {
        typealias Check = @convention(c) (pid_t, UnsafePointer<CChar>?, Int32) -> Int32
        guard let h = dlopen(nil, RTLD_NOW), let sym = dlsym(h, "sandbox_check") else { return false }
        return unsafeBitCast(sym, to: Check.self)(getpid(), nil, 0) == 1
    }()

    public let root: String
    /// Writable without asking: the root, scratch, caches.  Canonical.
    public let writable: [String]
    /// Not even readable -- the red-team box hides the real home, so no
    /// command can put a real secret in front of the model (§16.7.4) --
    /// except what `revealed` gives back.
    public let hidden: [String]
    public let revealed: [String]

    /// Under the home directory: where toolchains keep what they download
    /// and build.  Measured, not guessed: grown from the denials real builds
    /// hit (§16.5.1).
    public static let cachesInHome = [
        "Library/Caches", ".cache", ".npm", ".pnpm-store", "Library/pnpm", ".yarn", ".bun/install/cache",
        ".cargo/registry", ".cargo/git", "go/pkg/mod", "Library/Caches/go-build", ".gradle", ".m2/repository",
        ".swiftpm", "Library/org.swift.swiftpm", "Library/Developer/Xcode/DerivedData", ".hex", ".mix",
        ".ccache", ".pub-cache", ".nuget/packages", ".cocoapods",
    ]

    /// `scratch` nil: the per-user temporary directory's parent (so the
    /// compilers' caches beside $TMPDIR) and /tmp.
    public init(root: String, home: String = NSHomeDirectory(), scratch: [String]? = nil,
                hidden: [String] = [], revealed: [String] = []) {
        self.hidden = hidden.map { Paths.physical(Paths.lexical($0)) }
        self.revealed = revealed.map { Paths.physical(Paths.lexical($0)) }
        let r = Paths.physical(Paths.lexical(root))
        let h = Paths.physical(Paths.lexical(home))
        let s = scratch ?? [Paths.parent(Paths.physical(Paths.lexical(NSTemporaryDirectory()))), "/tmp"]
        self.root = r
        self.writable = [r] + s.map { Paths.physical(Paths.lexical($0)) }
            + Containment.cachesInHome.map { h + "/" + $0 }
    }

    /// SBPL, which wants its strings quoted the C way.
    static func quoted(_ s: String) -> String {
        "\"" + s.replacingOccurrences(of: "\\", with: "\\\\").replacingOccurrences(of: "\"", with: "\\\"") + "\""
    }

    /// Inside the project, but run later by something the user trusts and
    /// that runs unconfined: a script that writes these plants what the
    /// sandbox would not let it do itself.  git's config (core.fsmonitor,
    /// core.hooksPath) and hooks, in nested repositories too; direnv; the
    /// editor's tasks; CI.
    static let plantable = [".envrc", ".vscode", ".github/workflows", ".husky", ".pre-commit-config.yaml"]

    /// SBPL regexes are POSIX: every metacharacter in a path is escaped.
    static func regexEscaped(_ s: String) -> String {
        var out = ""
        for ch in s {
            if "\\.^$*+?()[]{}|/\"".contains(ch) { out.append("\\") }
            out.append(ch)
        }
        return out
    }

    /// The profile a command runs under.  In SBPL the last matching rule
    /// wins (pinned by the canary suite), so the order is the meaning: deny
    /// every write, allow the writable set, deny what is plantable inside
    /// it, then allow exactly what an approval granted.
    public func profile(_ grant: ContainmentGrant = .none) -> String {
        var p = ["(version 1)", "(allow default)"]
        if !hidden.isEmpty {
            p.append("(deny file-read* " + hidden.map { "(subpath \(Containment.quoted($0)))" }.joined(separator: " ") + ")")
            if !revealed.isEmpty {
                p.append("(allow file-read* " + revealed.map { "(subpath \(Containment.quoted($0)))" }.joined(separator: " ") + ")")
            }
        }
        if !grant.anywhere {
            p.append("(deny file-write*)")
            p.append("(allow file-write* " + writable.map { "(subpath \(Containment.quoted($0)))" }.joined(separator: " ")
                     + " (literal \"/dev/null\") (literal \"/dev/zero\") (literal \"/dev/tty\")"
                     + " (literal \"/dev/dtracehelper\") (regex #\"^/dev/fd/\") (regex #\"^/dev/ttys\"))")
            p.append("(deny file-write* (regex #\"^" + Containment.regexEscaped(root) + "/(.*/)?\\.git/(hooks|config)(/|$)\")"
                     + Containment.plantable.map { " (subpath \(Containment.quoted(root + "/" + $0)))" }.joined() + ")")
            if !grant.paths.isEmpty {
                p.append("(allow file-write* " + grant.paths.map {
                    "(subpath \(Containment.quoted(Paths.physical(Paths.lexical($0)))))"
                }.joined(separator: " ") + ")")
            }
        }
        if !grant.ipc {
            p.append("(deny signal)")
            p.append("(allow signal (target same-sandbox))")
            p.append("(deny user-preference-write)")
            p.append("(deny process-exec (literal \"/usr/bin/osascript\") (literal \"/usr/bin/automator\")"
                     + " (literal \"/usr/bin/shortcuts\"))")
        }
        if !grant.network {
            // "localhost" matches this Mac's own interface addresses as well
            // as loopback (ContainmentSuite measures it): this Mac, not the
            // network.
            p.append("(deny network-outbound)")
            p.append("(allow network-outbound (remote ip \"localhost:*\"))")
            p.append("(allow network-outbound (remote unix-socket))")
            // DNS is a channel too: name lookups go through this socket.
            p.append("(deny network-outbound (remote unix-socket (path-literal \"/private/var/run/mDNSResponder\")))")
        }
        return p.joined(separator: "\n")
    }

    // MARK: shims

    /// Where the shims live: the app's own store, which no contained command
    /// can write, so none can put its own `swiftc` here.
    public static var shimDirectory: URL { Store.defaultRoot.appendingPathComponent("shims", isDirectory: true) }

    /// swiftc runs its macro plugins under sandbox-exec, and a sandbox
    /// cannot be applied inside another (measured: "sandbox_apply: Operation
    /// not permitted", and every @Observable fails to expand).  So contained
    /// commands find these first on PATH: each runs the next real one with
    /// the compiler's own sandbox off -- the outer profile is the stricter of
    /// the two.  `xcrun swiftc` resolves the toolchain itself and is not
    /// covered.
    static let shims: [String: String] = [
        "swiftc": #"exec "$real" -disable-sandbox "$@""#,
        "swift": #"""
            case "$1" in
              build|test|run) sub=$1; shift; exec "$real" "$sub" --disable-sandbox "$@" ;;
              package) shift; exec "$real" package --disable-sandbox "$@" ;;
              *) exec "$real" "$@" ;;
            esac
            """#,
    ]

    static func shimScript(_ name: String, _ body: String) -> String {
        """
        #!/bin/sh
        # Written by Qwasar (spec §16.5): \(name), with the compiler's nested sandbox off,
        # because this command already runs in one.
        # Found by PATH, so $0 is "<its PATH entry>/\(name)": compared as text,
        # never by resolving directories -- under a profile that hides the
        # home directory, its ancestors cannot be resolved.
        self_dir=$(dirname "$0")
        real=
        IFS=:
        for d in $PATH; do
          [ "$d" = "$self_dir" ] && continue
          [ "$d/\(name)" = "$0" ] && continue
          if [ -x "$d/\(name)" ]; then real="$d/\(name)"; break; fi
        done
        unset IFS
        [ -n "$real" ] || { echo "\(name): command not found" >&2; exit 127; }
        \(body)

        """
    }

    /// Writes the shims if they are missing or stale; their directory, or
    /// nil if it cannot be written.
    public static func prepareShims(in dir: URL = shimDirectory) -> String? {
        let fm = FileManager.default
        do {
            try fm.createDirectory(at: dir, withIntermediateDirectories: true)
            for (name, body) in shims {
                let url = dir.appendingPathComponent(name)
                let script = shimScript(name, body)
                if (try? String(contentsOf: url, encoding: .utf8)) != script {
                    try script.write(to: url, atomically: true, encoding: .utf8)
                }
                try fm.setAttributes([.posixPermissions: 0o755], ofItemAtPath: url.path)
            }
            return Paths.physical(dir.path)
        } catch {
            return nil
        }
    }

    /// `/bin/sh -c command`, under the profile for `grant`.
    public func argv(_ command: String, grant: ContainmentGrant) -> (exe: String, args: [String]) {
        (Containment.sandboxExec, ["-p", profile(grant), "/bin/sh", "-c", command])
    }

    /// A line for the result when a failure looks like the profile's doing,
    /// so the model and the user both know why -- nil when it does not.
    public static func diagnosis(_ output: String, status: Int32, grant: ContainmentGrant) -> String? {
        guard status != 0 else { return nil }
        let o = output.lowercased()
        var lines: [String] = []
        if !grant.anywhere, o.contains("operation not permitted") || o.contains("read-only file system") {
            lines.append("[containment: commands may write only inside the project, the temp directories and "
                         + "toolchain caches; a write elsewhere fails with \"Operation not permitted\". If this "
                         + "needs to write elsewhere, say so to the user rather than working around it.]")
        }
        if !grant.network, ["could not resolve host", "couldn't resolve host", "nodename nor servname",
                            "name or service not known", "temporary failure in name resolution",
                            "network is unreachable", "no route to host", "getaddrinfo"].contains(where: o.contains) {
            lines.append("[containment: commands have no network unless the user approves one that needs it; "
                         + "a network command asks first.]")
        }
        return lines.isEmpty ? nil : lines.joined(separator: "\n")
    }
}
