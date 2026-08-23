// Main.swift -- entry point.
//
// `--gate` runs M0's gate check headless and prints a verdict, so a failure is
// a line of output rather than a screenshot. It is the same GateCheck the
// window shows; there is no second implementation to drift.

import Foundation
import CrucibleKit

@main
struct CrucibleMain {
    static func main() {
        let args = CommandLine.arguments
        if args.contains("--sandbox") {
            // Virtualization needs a live main queue, so this mode keeps a run
            // loop turning and exits from inside the task rather than blocking
            // main on a semaphore (PLAN.md 3.4, fifth edge).
            let guestDir = URL(fileURLWithPath:
                value(of: "--guest", in: args) ?? "build/guest")
            let project = value(of: "--root", in: args).map { URL(fileURLWithPath: $0) }
            Task { @MainActor in
                let rc = await SandboxGate.run(guestDir: guestDir, projectDir: project)
                exit(rc)
            }
            RunLoop.main.run()
            return
        }

        if args.contains("--gate") {
            // Two phases, and the split is not cosmetic.
            //
            // Virtualization.framework binds a VZVirtualMachine to a serial
            // queue that defaults to the MAIN queue, and its initialiser
            // reaches that queue before returning. Constructing one from a
            // Task while the main thread sits in a semaphore deadlocks --
            // silently, with no output, which is exactly how it presented the
            // first time. So the entitlement and VM probes run on the main
            // thread, and only the engine load (which has no main-queue
            // affinity) is allowed to block it.
            gatePhase1()
            if let model = value(of: "--model", in: args) {
                let ask = value(of: "--prompt", in: args)
                // A live run loop, not a semaphore.
                //
                // Third time this trap has been sprung in this file, and the
                // rule is now unconditional: NOTHING blocks the main thread in
                // any mode. `main()` is implicitly @MainActor, so a plain Task
                // inherits main isolation; Task.detached escapes that, but the
                // work it does still reaches back to the main actor the moment
                // it touches Virtualization -- SandboxManager is @MainActor
                // because VZVirtualMachine has main-queue affinity (PLAN.md
                // 3.4). A blocked main thread deadlocks it silently: 0% CPU, no
                // output, indistinguishable from a slow model.
                Task.detached {
                    await gatePhase2(modelPath: model, prompt: ask)
                    exit(0)
                }
                RunLoop.main.run()
            } else {
                print("\n-- engine (the 16 GB mmap)")
                print("  skipped: pass --model <dir> to load")
                summarise(engineLoaded: nil)
            }
            return
        }
        CrucibleApp.main()
    }

    private static func value(of flag: String, in args: [String]) -> String? {
        guard let i = args.firstIndex(of: flag), i + 1 < args.count else { return nil }
        return args[i + 1]
    }

    nonisolated(unsafe) private static var report = GateReport()
    nonisolated(unsafe) private static var profile = MemoryProfile.derive()

    /// Main thread: profile, entitlements, and the VZVirtualMachine probe.
    private static func gatePhase1() {
        print("=== Crucible M0 gate ===")
        profile = MemoryProfile.derive()
        print("\n-- profile (PLAN.md 2.3)")
        print(profile.summary)

        report = GateCheck.run()
        print("\n-- entitlements and virtualization")
        for line in report.detail { print("  \(line)") }
        print("  verdict: \(report.virtualizationVerdict)")
    }

    /// Off-main: the engine load, which is the 16 GB mmap the gate is about.
    private static func gatePhase2(modelPath: String, prompt: String?) async {
        print("\n-- engine (the 16 GB mmap)")
        let p = profile
        let host = EngineHost()
        var loaded: EngineInfo?
        var failure: String?
        do { loaded = try await host.load(modelPath: modelPath, contextSize: p.contextSize) }
        catch { failure = String(describing: error) }

        if let i = loaded {
            print(String(format: "  loaded in %.1fs · %d layers · vocab %d · context %d",
                         i.loadSeconds, i.layers, i.vocabSize, i.contextSize))
            print(String(format: "  phys_footprint %.2f GB", Double(i.footprintBytes) / 1_073_741_824))
        } else {
            print("  FAILED: \(failure ?? "unknown")")
        }
        summarise(engineLoaded: loaded != nil)

        // --prefix: the system-prefix checkpoint, proven against the engine.
        //
        // Off by default because the cold half is a real ~2500-token prefill,
        // about eighty seconds. Worth its place in the gate anyway: this is an
        // optimisation whose failure mode is that everything still works and is
        // simply slower, which is exactly the kind of thing that stays broken.
        if loaded != nil, CommandLine.arguments.contains("--prefix") {
            print("\n-- system-prefix checkpoint")
            let runner: ToolExecuting = ToolRunner(root: URL(fileURLWithPath: "/"))
            let cfg = SessionConfig(system: runner.environmentDescription + "\n\n"
                                          + Project.defaultSystem)
            let probe = await host.probeSystemPrefix(config: cfg, runner: runner)
            if let e = probe.error {
                print("  FAILED: \(e)")
            } else {
                print(String(format: "  prefix: %d tokens", probe.prefixTokens))
                print(String(format: "  cold:   %.2fs, %d tokens prefilled",
                             probe.coldSeconds, probe.coldEvaluated))
                print(String(format: "  warm:   %.2fs, %d tokens restored from the cache",
                             probe.warmSeconds, probe.warmRestored))
                if probe.hit {
                    print(String(format: "  OK  the second session read the whole prefix (%.0fx)",
                                 probe.speedup))
                } else {
                    print("  NO  the second session did NOT hit the checkpoint — "
                        + "every new session is paying the full prefill")
                }
            }
        }

        guard loaded != nil, let prompt else { return }

        // A real agent turn, headless: session, tools, the loop. `--root`
        // scopes the read-only tool surface; without it the tools are pointed
        // at the current directory, which is what a person would expect from a
        // command line.
        let root = URL(fileURLWithPath: value(of: "--root", in: CommandLine.arguments)
                       ?? FileManager.default.currentDirectoryPath)
        print("\n-- agent turn")
        print("  root: \(root.path)")
        print("  > \(prompt)\n")

        // M3: the tools run in the guest, or -- with --no-sandbox -- in the
        // read-only host stand-in M1 shipped.
        let id = UUID()
        var runner: ToolExecuting = ToolRunner(root: root)
        var manager: SandboxManager?

        if !CommandLine.arguments.contains("--no-sandbox") {
            let guestDir = URL(fileURLWithPath:
                value(of: "--guest", in: CommandLine.arguments) ?? "build/guest")
            let m = await SandboxManager(guestDir: guestDir,
                                         stateDir: FileManager.default.temporaryDirectory
                                             .appendingPathComponent("crucible-m3"))
            do {
                let ready = try await m.start(session: id, projectRoot: root)
                print(String(format: "  sandbox: booted in %.2fs (%@ clone), tools run in /work",
                             ready.bootSeconds, ready.cloneMethod.rawValue))
                runner = SandboxToolRunner(channel: ready.channel)
                manager = m
            } catch {
                print("  sandbox unavailable (\(error)); falling back to the read-only host tools")
            }
        }
        defer { if let manager { Task { await manager.stopAll() } } }

        for await ev in host.openSession(id: id, runner: runner, config: SessionConfig(
            system: "You are a coding assistant working in \(root.lastPathComponent). "
                  + "Use the tools to inspect files before answering.",
            // 8192, not 2048.
            //
            // Measured: a `define` turn spent 2123 tokens reasoning about how
            // to write a small Elixir module and hit a 2048-token budget before
            // emitting the call at all. Writing code is a far more expensive
            // turn than calling a tool, and a budget sized for the latter
            // silently truncates the former -- the turn ends with no answer and
            // nothing to show for six minutes of decode.
            maxTokensPerTurn: 8192)) {
            if case .failed(let m) = ev { print("  session failed: \(m)"); return }
        }

        var sawText = false
        for await ev in host.send(id, text: prompt, cancelled: { false }) {
            switch ev {
            case .prefill(let done, let total):
                FileHandle.standardError.write("  prefill \(done)/\(total)\r".data(using: .utf8)!)
            case .context, .rate:
                break                                  // shown in the app's footer
            case .reasoning:
                break                                  // counted, not printed
            case .text(let t):
                if !sawText { print("  ", terminator: ""); sawText = true }
                print(t, terminator: ""); fflush(stdout)
            case .toolCall(let c):
                let args = c.arguments.map { "\($0.key)=\($0.value.prefix(40))" }
                                      .sorted().joined(separator: " ")
                print("\n  → \(c.name) \(args)")
            case .toolResult(let name, let r):
                let head = r.split(separator: "\n").prefix(3).joined(separator: "\n    ")
                print("    \(name): \(head)\(r.split(separator: "\n").count > 3 ? "\n    …" : "")")
                sawText = false
            case .note(let n):
                print("\n  [\(n)]")
            case .contextFull(let used, let limit):
                print("\n  [context full: \(used)/\(limit) — a successor session is needed]")
            case .turnFinished(let st):
                print("\n")
                print(String(format: "  %d prompt · %d generated (%d reasoning) · %d tool calls",
                             st.promptTokens, st.generatedTokens, st.reasoningTokens, st.toolCalls))
                print(String(format: "  prefill %.1fs (%.0f tok/s) · decode %.1fs (%.2f tok/s) · context %d/%d",
                             st.prefillSeconds, st.prefillTokensPerSecond,
                             st.decodeSeconds, st.tokensPerSecond,
                             st.contextUsed, st.contextLimit))
            case .failed(let m):
                print("\n  TURN FAILED: \(m)")
            }
        }
        let hist = await host.tokens(of: id)
        print("  history: \(hist.count) tokens (\(hist.count * 4) bytes to persist)")

        guard CommandLine.arguments.contains("--resume") else { return }

        // The restore path, end to end: close the session, throw its KV and
        // recurrent state away, reopen from the recorded tokens alone, and ask
        // a question only a session that remembers the first turn can answer.
        //
        // This is the failure the replay exists to prevent -- without it the
        // model starts over while the transcript still shows everything that
        // was said, and nothing in the UI would look wrong.
        print("\n-- close and resume from history")
        host.closeSession(id)
        let t0 = Date()
        var reopenFailed: String?
        for await ev in host.openSession(id: id, runner: runner, config: SessionConfig(
            system: "You are a coding assistant working in \(root.lastPathComponent). "
                  + "Use the tools to inspect files before answering.",
            maxTokensPerTurn: 8192), history: hist) {
            switch ev {
            case .prefill(let d, let t):
                FileHandle.standardError.write("  replay \(d)/\(t)\r".data(using: .utf8)!)
            case .note(let n): print("  [\(n)]")
            case .failed(let m): reopenFailed = m
            default: break
            }
        }
        if let reopenFailed { print("  RESUME FAILED: \(reopenFailed)"); return }
        print(String(format: "  rebuilt %d tokens in %.1fs", hist.count, Date().timeIntervalSince(t0)))

        let probe = "Without using any tools, name the file you read a moment ago. One sentence."
        print("  > \(probe)\n")
        var answer = ""
        for await ev in host.send(id, text: probe, cancelled: { false }) {
            switch ev {
            case .text(let t): answer += t; print(t, terminator: ""); fflush(stdout)
            case .toolCall(let c): print("\n  → \(c.name) (unexpected: it was told not to)")
            case .turnFinished(let st):
                print("\n")
                print(String(format: "  %d prompt · %d generated · context %d/%d",
                             st.promptTokens, st.generatedTokens, st.contextUsed, st.contextLimit))
            case .failed(let m): print("\n  FAILED: \(m)")
            default: break
            }
        }
        // A session that lost its history cannot name the file; one that kept
        // it says qwasar_toolcall.c. Reported rather than asserted, because the
        // model may phrase it any number of ways.
        let remembered = answer.lowercased().contains("toolcall")
        print("  resume verdict: \(remembered ? "REMEMBERS the first turn" : "does NOT appear to remember — check the replay")")
    }

    private static func summarise(engineLoaded: Bool?) {
        print("\n-- gate")
        let sandbox = report.sandboxed == true
        let virt = report.vmInstantiates
        line("App Sandbox active", sandbox)
        line("Virtualization entitlement usable", virt)
        switch engineLoaded {
        case .some(true):  line("16 GB mmap under those entitlements", true)
        case .some(false): line("16 GB mmap under those entitlements", false)
        case .none:        print("  ?  16 GB mmap: not tested")
        }
        let pass = sandbox && virt && engineLoaded == true
        print("\n  \(pass ? "GATE PASSES" : "gate incomplete or failed")")
    }

    private static func line(_ what: String, _ ok: Bool) {
        print("  \(ok ? "OK " : "NO ") \(what)")
    }
}
