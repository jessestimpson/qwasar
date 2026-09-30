// TestMain.swift -- `make test`.
//
// The suites, in order of how much they cost to run:
//   pathguard  no I/O beyond a temp tree; the security boundary
//   host tools a session's tools on this Mac: edit's contract, the runner, the
//              user's shell environment (starts their login shell once)
//   client     pure; the Session API's event stream, decoded
//   store      persistence, including what survives a crash mid-turn
//   markdown   blocks out of Foundation's parse; the highlighter's reconstruction
//
// The chat template and the tool-call parser are the server's now, and are
// tested there (tests/test_tokenizer, tests/test_toolcall, tests/test_session_api.py).
//
// Everything here runs from a terminal, which is the parent tree's culture and
// the only way CI will ever exercise this.

import Foundation

@main
struct TestMain {
    static func main() {
        let args = CommandLine.arguments
        var failures = 0

        print("== pathguard");  failures += PathGuardSuite.run()
        print("== host tools"); failures += HostToolsSuite.run()
        print("== network");    failures += NetworkPolicySuite.run()
        print("== overlay");    failures += SandboxOverlaySuite.run()
        print("== delegation");  failures += EscalationSuite.run()
        print("== client");     failures += ClientSuite.run()
        print("== store");      failures += StoreSuite.run()
        print("== guestimage"); failures += GuestImageSuite.run()
        print("== markdown");   failures += MarkdownSuite.run(args)

        print("")
        if failures == 0 {
            print("all suites pass")
        } else {
            print("\(failures) failure(s)")
            exit(1)
        }
    }

    static func value(of flag: String, in args: [String]) -> String? {
        guard let i = args.firstIndex(of: flag), i + 1 < args.count else { return nil }
        return args[i + 1]
    }

    static func check(_ ok: Bool, _ what: String) -> Int {
        print(ok ? "  ok   \(what)" : "  FAIL \(what)")
        return ok ? 0 : 1
    }
}
