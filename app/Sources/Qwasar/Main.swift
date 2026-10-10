// Main.swift -- entry point.
//
// An AppKit application with a status item, and a window it opens when asked
// (QwasarApp.swift).  `--agent` opens the window at launch; `--gate` runs the
// gates headless and prints a verdict, so a failure is a line of output
// rather than a screenshot.

import AppKit
import Foundation
import QwasarKit

@main
@MainActor
enum QwasarMain {
    static func main() {
        let args = CommandLine.arguments
        if args.contains("--gate") {
            // Nothing blocks the main thread: the gate's work is async.
            Task { @MainActor in
                let rc = await Gate.run(args)
                exit(rc)
            }
            RunLoop.main.run()
            return
        }

        let app = NSApplication.shared
        app.setActivationPolicy(.accessory)      // a menu bar item, until a window is asked for
        let delegate = QwasarAppDelegate(openAgentAtLaunch: args.contains("--agent"))
        app.delegate = delegate
        app.run()
    }

    static func value(of flag: String, in args: [String]) -> String? {
        guard let i = args.firstIndex(of: flag), i + 1 < args.count else { return nil }
        return args[i + 1]
    }
}
