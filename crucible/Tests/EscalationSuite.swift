// EscalationSuite.swift -- delegate, against a scripted provider.
//
// Spec §15, E1. No network: a URLProtocol stub plays the OpenAI-compatible
// streaming endpoint, so the suite can script multi-turn conversations,
// costs, and failures deterministically. What it pins:
//
//   - the budget stops the NEXT request, never the one in flight (§15.3);
//   - the user's queued message continues the conversation, stop ends it,
//     silence ends it (§15.2);
//   - the key reaches the Authorization header and nothing else -- not the
//     schema, not the events, not the tool result (§15.4);
//   - an unconfigured policy advertises nothing.

import Foundation
import CrucibleKit

enum EscalationSuite {

    // MARK: the scripted provider

    final class Stub: URLProtocol {
        struct Script: @unchecked Sendable {
            /// Each entry answers one request: SSE deltas + a cost.
            var responses: [(deltas: [String], cost: Double)]
            var authSeen: [String] = []
        }
        nonisolated(unsafe) static var script = Script(responses: [])
        static let lock = NSLock()

        override class func canInit(with request: URLRequest) -> Bool { true }
        override class func canonicalRequest(for r: URLRequest) -> URLRequest { r }

        override func startLoading() {
            Self.lock.lock()
            if let auth = request.value(forHTTPHeaderField: "Authorization") {
                Self.script.authSeen.append(auth)
            }
            let entry = Self.script.responses.isEmpty
                ? (deltas: ["[script exhausted]"], cost: 0.0)
                : Self.script.responses.removeFirst()
            Self.lock.unlock()

            let http = HTTPURLResponse(url: request.url!, statusCode: 200,
                                       httpVersion: "HTTP/1.1",
                                       headerFields: ["Content-Type": "text/event-stream"])!
            client?.urlProtocol(self, didReceive: http, cacheStoragePolicy: .notAllowed)
            var body = ""
            for d in entry.deltas {
                body += "data: {\"choices\":[{\"delta\":{\"content\":\"\(d)\"}}]}\n\n"
            }
            body += "data: {\"choices\":[{\"delta\":{}}],\"usage\":{\"cost\":\(entry.cost)}}\n\n"
            body += "data: [DONE]\n\n"
            client?.urlProtocol(self, didLoad: Data(body.utf8))
            client?.urlProtocolDidFinishLoading(self)
        }

        override func stopLoading() {}
    }

    final class Events: @unchecked Sendable {
        private let lock = NSLock()
        private var list: [DelegationEvent] = []
        func add(_ e: DelegationEvent) { lock.withLock { list.append(e) } }
        var all: [DelegationEvent] { lock.withLock { list } }
        var endedReason: String? {
            for case .ended(let r, _) in all { return r }
            return nil
        }
    }

    struct Inner: ToolExecuting {
        var schemas: [String] { [] }
        var environmentDescription: String { "inner." }
        func run(_ call: ToolCall) -> String { "inner: \(call.name)" }
    }

    static func runner(models: [String] = ["stub/model-a", "stub/model-b"],
                       remaining: Double = 1.0, turn: Double = 1.0,
                       grace: Double = 0.05,
                       events: Events, mailbox: DelegationMailbox,
                       key: String? = "SECRET-KEY") -> DelegateToolRunner {
        DelegateToolRunner(inner: Inner(),
                           policy: EscalationPolicy(models: models,
                                                    sessionRemainingUSD: remaining,
                                                    turnBudgetUSD: turn,
                                                    graceSeconds: grace),
                           mailbox: mailbox,
                           emit: { events.add($0) },
                           baseURL: URL(string: "https://stub.invalid/api/v1")!,
                           keyProvider: { key })
    }

    static func run() -> Int {
        var f = 0
        let cfg = URLSessionConfiguration.ephemeral
        cfg.protocolClasses = [Stub.self]
        let saved = DelegateToolRunner.sessionConfiguration
        DelegateToolRunner.sessionConfiguration = cfg
        defer { DelegateToolRunner.sessionConfiguration = saved }

        func call(_ args: [String: String]) -> ToolCall {
            ToolCall(name: "delegate", arguments: args)
        }

        // A single consult: one response, then silence ends it.
        Stub.script = .init(responses: [(["Hello", " there"], 0.01)])
        var ev = Events(); var mb = DelegationMailbox()
        var out = runner(events: ev, mailbox: mb).run(call(["task": "greet"]))
        f += TestMain.check(out.contains("Hello there"), "the answer comes back")
        f += TestMain.check(out.contains("$0.0100"), "the header carries the cost")
        f += TestMain.check(ev.endedReason == "the remote model finished",
                            "silence after the grace window ends it")
        f += TestMain.check(!out.contains("SECRET-KEY"),
                            "the key is not in the tool result")
        f += TestMain.check(Stub.script.authSeen == ["Bearer SECRET-KEY"],
                            "the key went to the Authorization header, once")
        f += TestMain.check(!ev.all.contains { ev in
            if case .delta(let s) = ev { return s.contains("SECRET") } else { return false }
        }, "the key is in no event")

        // A queued user message continues the conversation; the second answer
        // is the one that returns; costs sum.
        Stub.script = .init(responses: [(["first"], 0.01), (["second"], 0.02)])
        ev = Events(); mb = DelegationMailbox()
        mb.post("go deeper")
        out = runner(events: ev, mailbox: mb).run(call(["task": "t"]))
        f += TestMain.check(out.contains("second") && !out.hasSuffix("first"),
                            "the user's message continued the conversation")
        f += TestMain.check(out.contains("$0.0300"), "costs sum across turns")
        f += TestMain.check(ev.all.contains { if case .userTurn = $0 { return true } else { return false } },
                            "the user turn is an event the card can render")

        // The budget stops the NEXT request: with the cap under the first
        // response's cost and a message queued, exactly one request is sent.
        Stub.script = .init(responses: [(["partial"], 0.01), (["never"], 0.01)])
        ev = Events(); mb = DelegationMailbox()
        mb.post("continue!")
        out = runner(turn: 0.005, events: ev, mailbox: mb).run(call(["task": "t"]))
        f += TestMain.check(out.contains("partial") && !out.contains("never"),
                            "the budget declined the next request, not the one in flight")
        f += TestMain.check(ev.endedReason == "the budget tripped", "and said so")
        f += TestMain.check(Stub.lock.withLock { Stub.script.responses.count } == 1,
                            "exactly one request reached the provider")

        // Stop ends it after the current response.
        Stub.script = .init(responses: [(["answer"], 0.01), (["never"], 0.01)])
        ev = Events(); mb = DelegationMailbox()
        mb.stop()
        out = runner(events: ev, mailbox: mb).run(call(["task": "t"]))
        f += TestMain.check(ev.endedReason == "the user ended it", "stop ends it")

        // Refusals, each for its stated reason.
        ev = Events(); mb = DelegationMailbox()
        out = runner(events: ev, mailbox: mb).run(call(["task": "t", "model": "other/model"]))
        f += TestMain.check(out.contains("not an allowed model"), "model allowlist holds")
        out = runner(events: ev, mailbox: mb, key: nil).run(call(["task": "t"]))
        f += TestMain.check(out.contains("no API key"), "a missing key is named")
        out = runner(remaining: 0, events: ev, mailbox: mb).run(call(["task": "t"]))
        f += TestMain.check(out.contains("budget is exhausted"), "an exhausted budget is named")

        // The schema: carries models and budgets, never the key; and the
        // overlay's empty-list semantics gate the whole feature.
        let schema = DelegateToolRunner.schema(policy: EscalationPolicy(
            models: ["a/b"], sessionRemainingUSD: 2.5, turnBudgetUSD: 1))
        f += TestMain.check((schema.contains("a/b") || schema.contains("a\\/b")) && schema.contains("2.50"),
                            "the schema names models and remaining budget")
        f += TestMain.check(!schema.contains("SECRET"), "and no secret")
        f += TestMain.check(SandboxSettings.resolve(
                global: SandboxOverlay(agentModels: ["x/y"]),
                project: nil,
                session: SandboxOverlay(agentModels: [])).agentModels.isEmpty,
            "a session's empty agent_models turns escalation OFF over a global grant")

        return f
    }
}
