// ContainmentSuite.swift -- spec §16.7.3: Seatbelt, proven with canaries.
//
// Containment only proves itself by running commands, so these do -- and
// every one is harmless by construction: `touch` and `echo` inside a fresh
// temp tree, and `nc -z` against listeners this suite opens.  "Can it delete
// outside the root" is tested as "can it create outside the root": the
// profile's rule (file-write*) is the same.  Nothing is removed but the
// tree this suite made, no external host is contacted, and if sandbox-exec
// is missing the suite skips -- it never falls back to running unconfined.

import Darwin
import Foundation
import QwasarKit

enum ContainmentSuite {
    static func run() -> Int {
        guard Containment.available else {
            print("  skipped: \(Containment.sandboxExec) is missing")
            return 0
        }
        guard !Containment.alreadySandboxed else {
            // A contained agent running `make test`: profiles cannot nest,
            // so nothing here could be proven -- and nothing is claimed.
            print("  skipped: this process is already sandboxed, and profiles cannot nest")
            return 0
        }
        let fm = FileManager.default
        let t = URL(fileURLWithPath: fm.temporaryDirectory.path).resolvingSymlinksInPath()
            .appendingPathComponent("qwasar-contain-\(UUID().uuidString)")
        defer { try? fm.removeItem(at: t) }
        let proj = t.appendingPathComponent("proj"), home = t.appendingPathComponent("home"),
            outside = t.appendingPathComponent("outside")
        for d in [proj.appendingPathComponent(".git/hooks"), home, outside] {
            try? fm.createDirectory(at: d, withIntermediateDirectories: true)
        }
        // Scratch empty: the tree itself is under $TMPDIR, which is
        // otherwise writable.
        let c = Containment(root: proj.path, home: home.path, scratch: [])
        let backend = HostBackend(root: proj, environment: ["PATH": "/usr/bin:/bin", "HOME": home.path],
                                  containment: c)
        func sh(_ cmd: String, _ grant: ContainmentGrant = .none) -> ExecResult {
            ContainmentGrants.$current.withValue(grant) {
                (try? backend.exec(cmd, timeoutMS: 10_000, maxBytes: 1 << 16).get())
                    ?? ExecResult(status: -1, output: "", timedOut: false)
            }
        }
        func exists(_ u: URL) -> Bool { fm.fileExists(atPath: u.path) }
        var f = 0

        print("  -- writes")
        f += TestMain.check(sh("echo contained").output.hasPrefix("contained"), "a command runs under the profile")
        _ = sh("touch '\(proj.path)/canary'")
        f += TestMain.check(exists(proj.appendingPathComponent("canary")), "it writes inside the project")
        let r = sh("touch '\(outside.path)/canary'")
        f += TestMain.check(!exists(outside.appendingPathComponent("canary")) && r.status != 0,
                            "it cannot write outside the project")
        f += TestMain.check(r.output.contains("[containment:"), "and the result says containment blocked it")
        _ = sh("touch '\(proj.path)/.git/hooks/canary'")
        f += TestMain.check(!exists(proj.appendingPathComponent(".git/hooks/canary")),
                            "it cannot write the project's .git/hooks, inside the project or not")
        _ = sh("mkdir -p '\(home.path)/.cache/tool' && touch '\(home.path)/.cache/tool/canary'")
        f += TestMain.check(exists(home.appendingPathComponent(".cache/tool/canary")), "it writes a toolchain cache")
        _ = sh("touch '\(home.path)/.zshrc'")
        f += TestMain.check(!exists(home.appendingPathComponent(".zshrc")), "it cannot write the home directory")

        print("  -- grants widen by exactly what was approved")
        _ = sh("touch '\(outside.path)/one'", ContainmentGrant(paths: [outside.path + "/one"]))
        _ = sh("touch '\(outside.path)/two'", ContainmentGrant(paths: [outside.path + "/one"]))
        f += TestMain.check(exists(outside.appendingPathComponent("one")) && !exists(outside.appendingPathComponent("two")),
                            "a granted path is writable; its sibling is not")
        _ = sh("touch '\(proj.path)/.git/hooks/granted'", ContainmentGrant(paths: [proj.path + "/.git/hooks/granted"]))
        f += TestMain.check(exists(proj.appendingPathComponent(".git/hooks/granted")), "an approved hook write lands")
        _ = sh("touch '\(home.path)/anywhere'", ContainmentGrant(anywhere: true))
        f += TestMain.check(exists(home.appendingPathComponent("anywhere")), "an anywhere grant writes anywhere")

        print("  -- the network")
        guard let lo = Listener(address: "127.0.0.1") else { return f + TestMain.check(false, "a loopback listener") }
        f += TestMain.check(sh("nc -z -G 2 127.0.0.1 \(lo.port)").status == 0, "loopback is reachable")
        // Seatbelt's "localhost" is this Mac: its own interface addresses
        // match too (measured 2026-10-13), which is still not the network.
        if let lan = Listener.lanAddress(), let any = Listener(address: "0.0.0.0") {
            f += TestMain.check(sh("nc -z -G 2 \(lan) \(any.port)").status == 0,
                                "this Mac's own address (\(lan)) counts as localhost")
        }
        // 192.0.2.1 is TEST-NET-1 (RFC 5737): reserved for documentation,
        // owned by no one.  Under the profile the connect is refused before a
        // packet leaves; it is never attempted with the network granted.
        let off = sh("nc -v -z -G 2 192.0.2.1 80")
        f += TestMain.check(off.status != 0 && off.output.contains("Operation not permitted"),
                            "any other address is refused in the kernel, before a packet leaves")
        f += TestMain.check(sh("nc -U /private/var/run/mDNSResponder </dev/null").status != 0,
                            "the DNS resolver's socket is not reachable: no lookups to carry data out")
        f += TestMain.check(sh("nc -U /private/var/run/mDNSResponder </dev/null", ContainmentGrant(network: true)).status == 0,
                            "with a network grant it is")

        print("  -- through the gate")
        // The point of containment: a command the policy allows -- the
        // project's own script -- that does what the policy cannot see.
        try? "touch '\(outside.path)/sneaked'\n".write(to: proj.appendingPathComponent("build.sh"),
                                                       atomically: true, encoding: .utf8)
        let gate = GatedToolRunner(inner: KitOver(backend: backend), context: PolicyContext(root: proj.path,
                                   home: home.path, scratch: []), approver: { _ in .allow })
        _ = gate.run(ToolCall(name: "Bash", arguments: ["command": "sh build.sh"]))
        f += TestMain.check(!exists(outside.appendingPathComponent("sneaked")),
                            "an allowed project script cannot write outside the project")
        _ = gate.run(ToolCall(name: "Bash", arguments: ["command": "touch '\(outside.path)/approved'"]))
        f += TestMain.check(exists(outside.appendingPathComponent("approved")),
                            "an approved write outside runs, widened to that path by the gate")
        let m = sh("echo x", ContainmentGrant(network: true, paths: ["/x\"y"], anywhere: false))
        f += TestMain.check(m.status == 0, "a path with a quote in it does not break the profile")
        return f
    }

    struct KitOver: ToolExecuting {
        let kit: ToolKit
        init(backend: ToolBackend) { kit = ToolKit(backend: backend, resultCap: 1 << 16) }
        var schemas: [String] { [] }
        var environmentDescription: String { "" }
        func run(_ call: ToolCall) -> String { kit.run(call) }
    }

    /// A TCP listener this suite owns: the only thing its network checks
    /// ever connect to.
    final class Listener {
        let fd: Int32
        let port: Int

        init?(address: String) {
            let fd = socket(AF_INET, SOCK_STREAM, 0)
            guard fd >= 0 else { return nil }
            var on: Int32 = 1
            setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &on, socklen_t(MemoryLayout<Int32>.size))
            var a = sockaddr_in()
            a.sin_family = sa_family_t(AF_INET)
            a.sin_port = 0
            inet_pton(AF_INET, address, &a.sin_addr)
            let ok = withUnsafePointer(to: &a) {
                $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                    bind(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) == 0
                }
            }
            guard ok, listen(fd, 8) == 0 else { close(fd); return nil }
            var len = socklen_t(MemoryLayout<sockaddr_in>.size)
            _ = withUnsafeMutablePointer(to: &a) {
                $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { getsockname(fd, $0, &len) }
            }
            self.fd = fd
            port = Int(UInt16(bigEndian: a.sin_port))
        }

        deinit { close(fd) }

        /// This Mac's first non-loopback IPv4 address, if it has one.
        static func lanAddress() -> String? {
            var head: UnsafeMutablePointer<ifaddrs>?
            guard getifaddrs(&head) == 0 else { return nil }
            defer { freeifaddrs(head) }
            var p = head
            while let i = p {
                defer { p = i.pointee.ifa_next }
                guard let sa = i.pointee.ifa_addr, sa.pointee.sa_family == sa_family_t(AF_INET),
                      (i.pointee.ifa_flags & UInt32(IFF_LOOPBACK)) == 0,
                      (i.pointee.ifa_flags & UInt32(IFF_UP)) != 0 else { continue }
                var host = [CChar](repeating: 0, count: Int(NI_MAXHOST))
                if getnameinfo(sa, socklen_t(sa.pointee.sa_len), &host, socklen_t(host.count), nil, 0, NI_NUMERICHOST) == 0 {
                    return String(cString: host)
                }
            }
            return nil
        }
    }
}
