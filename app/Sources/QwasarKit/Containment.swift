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
// What this does not confine: other processes.  A command that asks a
// daemon to act -- cfprefsd for `defaults write`, LaunchServices for
// `open`, Apple Events for osascript -- has the daemon do the writing,
// outside the sandbox.  The policy asks before each of those.

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

    public init(network: Bool = false, paths: [String] = [], anywhere: Bool = false) {
        self.network = network
        self.paths = paths
        self.anywhere = anywhere
    }

    public static let none = ContainmentGrant()

    public static func + (a: ContainmentGrant, b: ContainmentGrant) -> ContainmentGrant {
        ContainmentGrant(network: a.network || b.network,
                         paths: a.paths + b.paths.filter { !a.paths.contains($0) },
                         anywhere: a.anywhere || b.anywhere)
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
    public init(root: String, home: String = NSHomeDirectory(), scratch: [String]? = nil) {
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

    /// The profile a command runs under.  In SBPL the last matching rule
    /// wins, so each deny comes before the allows that carve it open, and
    /// the hooks deny comes after them all (pinned by the canary suite).
    public func profile(_ grant: ContainmentGrant = .none) -> String {
        var p = ["(version 1)", "(allow default)"]
        if !grant.anywhere {
            let paths = (writable + grant.paths.map { Paths.physical(Paths.lexical($0)) })
                .map { "(subpath \(Containment.quoted($0)))" }
            p.append("(deny file-write*)")
            p.append("(allow file-write* " + paths.joined(separator: " ")
                     + " (literal \"/dev/null\") (literal \"/dev/zero\") (literal \"/dev/tty\")"
                     + " (literal \"/dev/dtracehelper\") (regex #\"^/dev/fd/\") (regex #\"^/dev/ttys\"))")
            let hooks = root + "/.git/hooks"
            if !grant.paths.contains(where: { Paths.within(Paths.physical(Paths.lexical($0)), hooks) }) {
                p.append("(deny file-write* (subpath \(Containment.quoted(hooks))))")
            }
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
        self_dir=$(cd "$(dirname "$0")" && pwd -P)
        real=
        IFS=:
        for d in $PATH; do
          [ "$(cd "$d" 2>/dev/null && pwd -P)" = "$self_dir" ] && continue
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
