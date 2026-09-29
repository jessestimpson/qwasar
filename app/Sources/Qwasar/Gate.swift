// Gate.swift -- the milestone gates, headless.
//
// Two halves.  The first is M0's: is the bundle sandboxed, does Virtualization
// instantiate a VM (GateCheck).  The second is the renovation's (PLAN-qwasar.md
// §4.2): can this sandboxed app run qwasar-server as its helper, with the
// helper reading a model folder the app was granted -- and then a real turn
// through it, tools in the guest when there is one.
//
//   Qwasar --gate                              probes only
//   Qwasar --gate --model <dir> [--prompt ..]  starts the helper on that model
//   Qwasar --gate --server <url> [--prompt ..] uses a server that is running
//   --root <dir> --no-sandbox --guest <dir>    as for the sandbox gate

import Foundation
import QwasarKit

@MainActor
enum Gate {
    static func run(_ args: [String]) async -> Int32 {
        print("=== Qwasar gate ===")
        let report = GateCheck.run()
        print("\n-- entitlements and virtualization")
        for line in report.detail { print("  \(line)") }
        print("  verdict: \(report.virtualizationVerdict)")

        let model = QwasarMain.value(of: "--model", in: args)
        let serverURL = QwasarMain.value(of: "--server", in: args)
        guard model != nil || serverURL != nil else {
            print("\n-- server\n  skipped: pass --model <dir> to start the helper, or --server <url>")
            return report.vmInstantiates ? 0 : 1
        }

        var controller: ServerController?
        let base: URL
        if let serverURL, let u = URL(string: serverURL) {
            base = u
        } else {
            let c = ServerController()
            c.port = 18200 + Int.random(in: 0..<200)
            c.stateDir = FileManager.default.temporaryDirectory.appendingPathComponent("qwasar-gate-\(UUID().uuidString)")
            print("\n-- the helper (qwasar-server, from the bundle)")
            print("  port \(c.port), state \(c.stateDir!.path)")
            c.start(model: model!)
            let t0 = Date()
            while c.state == .starting, Date().timeIntervalSince(t0) < 600 {
                try? await Task.sleep(for: .milliseconds(250))
            }
            switch c.state {
            case .listening: print(String(format: "  listening after %.1fs", Date().timeIntervalSince(t0)))
            case .failed(let why):
                print("  FAILED: \(why)")
                print("  (under App Sandbox the helper must inherit the app's grant to the model folder;")
                print("   see \(c.logURL.path))")
                return 1
            default: print("  did not come up: \(c.state)"); return 1
            }
            controller = c
            base = c.baseURL
        }
        defer { controller?.stop() }

        let client = QwasarClient(base: base)
        let info: ServerInfo
        do { info = try await client.serverInfo() } catch { print("  /v1/server: \(error)"); return 1 }
        print("  \(info.model.name) · context \(info.context) · \(info.live_sessions) live")
        print("  " + info.summary.replacingOccurrences(of: "\n", with: "\n  "))

        guard let prompt = QwasarMain.value(of: "--prompt", in: args) else {
            print("\n-- turn\n  skipped: pass --prompt to run one")
            return 0
        }
        let root = URL(fileURLWithPath: QwasarMain.value(of: "--root", in: args) ?? FileManager.default.currentDirectoryPath)
        print("\n-- agent turn")
        print("  root: \(root.path)")
        print("  > \(prompt)\n")

        var runner: ToolExecuting = ToolRunner(root: root)
        var sandboxes: SandboxManager?
        if !args.contains("--no-sandbox") {
            let guestDir = URL(fileURLWithPath: QwasarMain.value(of: "--guest", in: args) ?? "build/guest")
            let stateDir = FileManager.default.temporaryDirectory.appendingPathComponent("qwasar-gate-vm-\(UUID().uuidString)")
            let m = SandboxManager(guestDir: guestDir, stateDir: stateDir)
            sandboxes = m
            let id = UUID()
            do {
                let ready = try await m.start(session: id, projectRoot: root)
                print(String(format: "  sandbox: booted in %.2fs, tools run in /work", ready.bootSeconds))
                runner = SandboxToolRunner(channel: ready.channel)
            } catch {
                print("  sandbox unavailable (\(error)); falling back to the read-only host tools")
            }
        }
        defer { if let s = sandboxes { Task { await s.stopAll() } } }

        let opened: OpenedSession
        do {
            opened = try await client.open(system: runner.environmentDescription + "\n\n" + Project.defaultSystem,
                                           tools: runner.schemas, thinking: true, effort: "low",
                                           metadata: ["client": "gate"])
        } catch { print("  open: \(error)"); return 1 }
        print("  session \(opened.id): prefix \(opened.prefix_tokens) tokens")

        var stream = client.turn(opened.id, text: prompt, temperature: 0, maxTokens: 4096)
        var steps = 0
        var ok = false
        do {
            while steps < 8 {
                var calls: [(String, ToolCall)] = []
                var stop = ""
                var sawText = false
                for try await se in stream {
                    switch se.event {
                    case .resume(let from, let restored, let prefill, _):
                        print("  [resume: \(from), \(restored) restored, \(prefill) to prefill]")
                    case .prefill(let d, let t):
                        if t >= 128 { print("  [prefill \(d)/\(t)]") }
                    case .text(let t):
                        if !sawText { print("  ", terminator: ""); sawText = true }
                        print(t, terminator: ""); fflush(stdout)
                    case .toolCall(let id, let name, let args):
                        print("\n  → \(name) \(args)")
                        calls.append((id, ToolCall(name: name, arguments: args)))
                    case .done(let s, let prompt, let generated, let reasoning, let pf, let dec, _, _, _, let used, let limit, _):
                        stop = s
                        print(String(format: "\n  [%@: %d prompt · %d generated (%d reasoning) · prefill %.1fs · decode %.1fs (%.1f tok/s) · ctx %d/%d]",
                                     s, prompt, generated, reasoning, pf, dec, dec > 0 ? Double(generated) / dec : 0, used, limit))
                    case .error(let m):
                        print("\n  ERROR: \(m)"); return 1
                    default: break
                    }
                }
                steps += 1
                if stop == "tool_calls", !calls.isEmpty {
                    var results: [ToolResultPayload] = []
                    for (id, c) in calls {
                        let r = runner.run(c)
                        let head = r.split(separator: "\n").prefix(3).joined(separator: "\n    ")
                        print("    \(c.name): \(head)\(r.split(separator: "\n").count > 3 ? "\n    …" : "")")
                        results.append(ToolResultPayload(id: id, content: r))
                    }
                    stream = client.continueStep(opened.id, results: results, temperature: 0, maxTokens: 4096)
                    continue
                }
                ok = stop == "end_turn" || stop == "length"
                break
            }
        } catch { print("\n  TURN FAILED: \(error)"); return 1 }

        print("\n-- describe")
        if let d = try? await client.describe(opened.id) {
            print("  \(d.tokens) tokens · \(d.state) · \(d.warmth.state)")
        }
        _ = try? await client.park(opened.id)
        print("\n  \(ok ? "GATE PASSES" : "gate incomplete")")
        return ok ? 0 : 1
    }
}
