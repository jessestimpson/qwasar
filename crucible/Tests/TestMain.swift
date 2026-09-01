// TestMain.swift -- `make test`.
//
// The suites, in order of how much they cost to run:
//   pathguard  no I/O beyond a temp tree; the security boundary
//   store      persistence, including what survives a crash mid-turn
//   utf8       pure; the transcript's silent corruption bug
//   markdown   blocks out of Foundation's parse; the highlighter's reconstruction
//   prefix     loads the tokenizer; the invariant the KV checkpoint rests on
//   golden     loads the tokenizer (~1s); the highest-value test in the project
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
        print("== network");    failures += NetworkPolicySuite.run()
        print("== overlay");    failures += SandboxOverlaySuite.run()
        print("== delegation");  failures += EscalationSuite.run()
        print("== store");      failures += StoreSuite.run()
        print("== guestimage"); failures += GuestImageSuite.run()
        print("== utf8");       failures += UTF8Suite.run()
        print("== toolparse");  failures += ToolParserSuite.run()
        print("== markdown");   failures += MarkdownSuite.run(args)
        print("== prefix");     failures += PrefixSuite.run(args)
        print("== golden");     failures += GoldenSuite.run(args)

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
