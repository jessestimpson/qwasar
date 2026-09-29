// Main.swift -- entry point.
//
// An AppKit application with a status item, and a window it opens when asked
// (QwasarApp.swift).  `--agent` opens the window at launch; `--gate` runs the
// gates headless and prints a verdict, so a failure is a line of output
// rather than a screenshot; `--sandbox` boots a guest and talks to the warden.

import AppKit
import Foundation
import QwasarKit

@main
@MainActor
enum QwasarMain {
    static func main() {
        let args = CommandLine.arguments
        if args.contains("--sandbox") {
            // Virtualization needs a live main queue, so this mode keeps a run
            // loop turning and exits from inside the task rather than blocking
            // main on a semaphore (spec 3.4, fifth edge).
            let guestDir = URL(fileURLWithPath: value(of: "--guest", in: args) ?? "build/guest")
            let project = value(of: "--root", in: args).map { URL(fileURLWithPath: $0) }
            Task { @MainActor in
                let rc = await SandboxGate.run(guestDir: guestDir, projectDir: project)
                exit(rc)
            }
            RunLoop.main.run()
            return
        }
        if args.contains("--gate") {
            // NOTHING blocks the main thread in any mode: the VM probe has
            // main-queue affinity, and a blocked main deadlocks it silently.
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
