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
        enum Reply: @unchecked Sendable {
            case text([String], cost: Double)
            /// A streamed tool call: the name in one fragment, the arguments
            /// split across two, the id in the first -- the shape providers
            /// actually send.
            case toolCall(name: String, args: String, cost: Double)
        }
        struct Script: @unchecked Sendable {
            var responses: [Reply]
            var authSeen: [String] = []
            /// Decoded request bodies, so tests can assert what went back.
            var requests: [[String: Any]] = []
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
            // URLProtocol surfaces the body as a stream.
            if let stream = request.httpBodyStream {
                stream.open()
                var data = Data()
                let buf = UnsafeMutablePointer<UInt8>.allocate(capacity: 65536)
                while stream.hasBytesAvailable {
                    let n = stream.read(buf, maxLength: 65536)
                    if n <= 0 { break }
                    data.append(buf, count: n)
                }
                buf.deallocate()
                stream.close()
                if let obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] {
                    Self.script.requests.append(obj)
                }
            }
            let entry: Reply = Self.script.responses.isEmpty
                ? .text(["[script exhausted]"], cost: 0.0)
                : Self.script.responses.removeFirst()
            Self.lock.unlock()

            let http = HTTPURLResponse(url: request.url!, statusCode: 200,
                                       httpVersion: "HTTP/1.1",
                                       headerFields: ["Content-Type": "text/event-stream"])!
            client?.urlProtocol(self, didReceive: http, cacheStoragePolicy: .notAllowed)
            var body = ""
            switch entry {
            case .text(let deltas, let cost):
                for d in deltas {
                    body += "data: {\"choices\":[{\"delta\":{\"content\":\"\(d)\"}}]}\n\n"
                }
                body += "data: {\"choices\":[{\"delta\":{}}],\"usage\":{\"cost\":\(cost)}}\n\n"
            case .toolCall(let name, let args, let cost):
                let mid = args.index(args.startIndex, offsetBy: args.count / 2)
                let a1 = String(args[..<mid]).replacingOccurrences(of: "\"", with: "\\\"")
                let a2 = String(args[mid...]).replacingOccurrences(of: "\"", with: "\\\"")
                body += "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_9\",\"function\":{\"name\":\"\(name)\",\"arguments\":\"\(a1)\"}}]}}]}\n\n"
                body += "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\(a2)\"}}]}}]}\n\n"
                body += "data: {\"choices\":[{\"delta\":{}}],\"usage\":{\"cost\":\(cost)}}\n\n"
            }
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

    final class CallLog: @unchecked Sendable {
        private let lock = NSLock()
        private var list: [ToolCall] = []
        func add(_ c: ToolCall) { lock.withLock { list.append(c) } }
        var all: [ToolCall] { lock.withLock { list } }
    }

    struct Inner: ToolExecuting {
        var log: CallLog? = nil
        var schemas: [String] {
            [#"{"type": "function", "function": {"name": "write", "description": "w", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}}"#]
        }
        var environmentDescription: String { "inner." }
        func run(_ call: ToolCall) -> String {
            log?.add(call)
            return "ok: \(call.name) \(call.arguments["path"] ?? "")"
        }
    }

    static func runner(models: [String] = ["stub/model-a", "stub/model-b"],
                       remaining: Double = 1.0, turn: Double = 1.0,
                       grace: Double = 0.05, log: CallLog? = nil,
                       events: Events, mailbox: DelegationMailbox,
                       key: String? = "SECRET-KEY",
                       records: (@Sendable (Int) -> DelegationRecord?)? = nil) -> DelegateToolRunner {
        DelegateToolRunner(inner: Inner(log: log),
                           policy: EscalationPolicy(models: models,
                                                    sessionRemainingUSD: remaining,
                                                    turnBudgetUSD: turn,
                                                    graceSeconds: grace),
                           mailbox: mailbox,
                           emit: { events.add($0) },
                           baseURL: URL(string: "https://stub.invalid/api/v1")!,
                           keyProvider: { key },
                           logProvider: records)
    }

    static func handoff(_ summary: String, cost: Double = 0.0) -> Stub.Reply {
        .toolCall(name: "handoff", args: "{\"summary\": \"\(summary)\"}", cost: cost)
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

        // The delegate's ONLY self-exit is handoff (spec 15.2): it cedes
        // with a summary, and the summary -- not a transcript -- is what the
        // local model receives.
        Stub.script = .init(responses: [handoff("done; detail in notes.txt", cost: 0.01)])
        var ev = Events(); var mb = DelegationMailbox()
        var out = runner(events: ev, mailbox: mb).run(call(["task": "greet"]))
        f += TestMain.check(out.contains("detail in notes.txt"), "the handoff comes back")
        f += TestMain.check(out.contains("$0.0100"), "the header carries the cost")
        f += TestMain.check(ev.endedReason == "the delegate ceded control",
                            "handoff is the exit, and says so")
        let req1 = Stub.lock.withLock { Stub.script.requests.first }
        let toolNames = (req1?["tools"] as? [[String: Any]])?.compactMap {
            ($0["function"] as? [String: Any])?["name"] as? String
        } ?? []
        f += TestMain.check(toolNames.contains("handoff"),
                            "the delegate is offered its handoff tool")
        f += TestMain.check(!out.contains("SECRET-KEY"),
                            "the key is not in the tool result")
        f += TestMain.check(Stub.script.authSeen == ["Bearer SECRET-KEY"],
                            "the key went to the Authorization header, once")
        f += TestMain.check(!ev.all.contains { ev in
            if case .delta(let s) = ev { return s.contains("SECRET") } else { return false }
        }, "the key is in no event")

        // Stopping WITHOUT handoff means waiting on the user -- unbounded:
        // a reply posted long after any old grace window continues it, and
        // the eventual handoff carries the cost of both turns.
        Stub.script = .init(responses: [.text(["which file?"], cost: 0.01),
                                        handoff("done", cost: 0.02)])
        ev = Events(); mb = DelegationMailbox()
        let late = mb
        Thread.detachNewThread {
            Thread.sleep(forTimeInterval: 0.4)
            late.post("the main one")
        }
        out = runner(events: ev, mailbox: mb).run(call(["task": "t"]))
        f += TestMain.check(ev.all.contains { if case .waiting = $0 { return true } else { return false } },
                            "the card is told the delegate is waiting")
        f += TestMain.check(out.contains("done") && ev.endedReason == "the delegate ceded control",
                            "the reply continued it and the handoff ended it")
        f += TestMain.check(out.contains("$0.0300"), "costs sum across turns")
        f += TestMain.check(ev.all.contains { if case .userTurn = $0 { return true } else { return false } },
                            "the user turn is an event the card can render")

        // Finish without a handoff: the last message stands in, marked.
        Stub.script = .init(responses: [.text(["half an answer"], cost: 0.0)])
        ev = Events(); mb = DelegationMailbox()
        let fin = mb
        Thread.detachNewThread {
            Thread.sleep(forTimeInterval: 0.2)
            fin.stop()
        }
        out = runner(events: ev, mailbox: mb).run(call(["task": "t"]))
        f += TestMain.check(ev.endedReason == "the user finished it"
                            && out.contains("no handoff was given")
                            && out.contains("half an answer"),
                            "Finish substitutes the last message, marked as such")

        // The budget stops the NEXT request: with the cap under the first
        // response's cost and a message queued, exactly one request is sent.
        Stub.script = .init(responses: [.text(["partial"], cost: 0.01), .text(["never"], cost: 0.01)])
        ev = Events(); mb = DelegationMailbox()
        mb.post("continue!")
        out = runner(turn: 0.005, events: ev, mailbox: mb).run(call(["task": "t"]))
        f += TestMain.check(out.contains("partial") && !out.contains("never")
                            && out.contains("no handoff was given"),
                            "the budget declined the next request, not the one in flight")
        f += TestMain.check(ev.endedReason == "the budget tripped", "and said so")
        f += TestMain.check(Stub.lock.withLock { Stub.script.responses.count } == 1,
                            "exactly one request reached the provider")

        // Finish queued before the first response still ends it.
        Stub.script = .init(responses: [.text(["answer"], cost: 0.01), .text(["never"], cost: 0.01)])
        ev = Events(); mb = DelegationMailbox()
        mb.stop()
        out = runner(events: ev, mailbox: mb).run(call(["task": "t"]))
        f += TestMain.check(ev.endedReason == "the user finished it", "Finish ends it")

        // Refusals, each for its stated reason.
        ev = Events(); mb = DelegationMailbox()
        out = runner(events: ev, mailbox: mb).run(call(["task": "t", "model": "other/model"]))
        f += TestMain.check(out.contains("not an allowed model"), "model allowlist holds")
        out = runner(events: ev, mailbox: mb, key: nil).run(call(["task": "t"]))
        f += TestMain.check(out.contains("no API key"), "a missing key is named")
        out = runner(remaining: 0, events: ev, mailbox: mb).run(call(["task": "t"]))
        f += TestMain.check(out.contains("delegation budget is exhausted"), "an exhausted budget is named")

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

        // delegation_log (spec 15.2): the local model pulls detail on demand
        // rather than having the transcript pushed at it.
        let record = DelegationRecord(model: "stub/model-a", task: "count things",
                                      log: "line one\nline two\nfindme here\nline four",
                                      ended: "the delegate ceded control")
        ev = Events(); mb = DelegationMailbox()
        let r = runner(events: ev, mailbox: mb,
                       records: { nth in nth == 1 ? record : nil })
        out = r.run(ToolCall(name: "delegation_log", arguments: [:]))
        f += TestMain.check(out.contains("1: line one") && out.contains("4: line four"),
                            "the log reads numbered")
        out = r.run(ToolCall(name: "delegation_log", arguments: ["grep": "findme"]))
        f += TestMain.check(out.contains("3: findme here") && !out.contains("line two"),
                            "grep narrows to matching lines")
        out = r.run(ToolCall(name: "delegation_log", arguments: ["nth": "2"]))
        f += TestMain.check(out.contains("no delegation #2"), "a missing delegation is named")
        f += TestMain.check(r.schemas.contains { $0.contains("\"delegation_log\"") },
                            "delegation_log is advertised with a provider")
        f += TestMain.check(!r.schemas.contains { $0.contains("\"handoff\"") },
                            "handoff is never advertised to the LOCAL model")

        // ---- E2: the remote agent works the sandbox by proxy ------------

        // The gate shape from spec §15.6: the remote model edits a file and
        // the result of that edit feeds back into its next request.
        Stub.script = .init(responses: [
            .toolCall(name: "write",
                      args: #"{"path": "notes.txt", "content": "hi"}"#,
                      cost: 0.01),
            handoff("wrote notes.txt; read it there", cost: 0.01),
        ])
        ev = Events(); mb = DelegationMailbox()
        let log = CallLog()
        out = runner(log: log, events: ev, mailbox: mb).run(call(["task": "edit the file"]))
        f += TestMain.check(log.all == [ToolCall(name: "write",
                                                 arguments: ["path": "notes.txt", "content": "hi"])],
                            "the remote tool call reached the inner executor, arguments decoded")
        f += TestMain.check(out.contains("wrote notes.txt; read it there"),
                            "and the handoff references the file rather than the content")
        f += TestMain.check(out.contains("write×1") && out.contains("re-read"),
                            "the result header tallies the tools so the local model knows /work moved")
        let second = Stub.lock.withLock { Stub.script.requests.last }
        let toolMsg = (second?["messages"] as? [[String: Any]])?.first {
            ($0["role"] as? String) == "tool"
        }
        f += TestMain.check((toolMsg?["content"] as? String) == "ok: write notes.txt"
                            && (toolMsg?["tool_call_id"] as? String) == "call_9",
                            "the tool result went back, correlated by id")
        let sentTools = Stub.lock.withLock { Stub.script.requests.first?["tools"] as? [[String: Any]] }
        let sentNames = (sentTools ?? []).compactMap { ($0["function"] as? [String: Any])?["name"] as? String }
        f += TestMain.check(Set(sentNames) == ["write", "handoff"],
                            "the remote model was offered the inner surface plus its handoff, and only those")
        f += TestMain.check(ev.all.contains { if case .toolCall(let n, _) = $0 { return n == "write" } else { return false } },
                            "the card hears the tool call")

        // Nested escalation is refused as a tool result, not executed.
        Stub.script = .init(responses: [
            .toolCall(name: "delegate", args: #"{"task": "recurse"}"#, cost: 0.0),
            handoff("fine", cost: 0.0),
        ])
        ev = Events(); mb = DelegationMailbox()
        _ = runner(events: ev, mailbox: mb).run(call(["task": "t"]))
        f += TestMain.check(ev.all.contains { ev in
            if case .toolResult(_, let r) = ev { return r.contains("nested escalation") } else { return false }
        }, "nested escalation is refused")

        // Long-horizon: NO step ceiling -- many cheap tool steps run to
        // completion; the governors are the budget (proven above) and Stop.
        Stub.script = .init(responses: [
            .toolCall(name: "write", args: #"{"path": "a", "content": "x"}"#, cost: 0.0),
            .toolCall(name: "write", args: #"{"path": "b", "content": "x"}"#, cost: 0.0),
            .toolCall(name: "write", args: #"{"path": "c", "content": "x"}"#, cost: 0.0),
            handoff("three files written", cost: 0.0),
        ])
        ev = Events(); mb = DelegationMailbox()
        let log2 = CallLog()
        out = runner(log: log2, events: ev, mailbox: mb).run(call(["task": "t"]))
        f += TestMain.check(log2.all.count == 3 && out.contains("three files written"),
                            "no step ceiling: a long tool run goes to completion")
        // Mid-work stop still cuts a long horizon short.
        // Deterministic mid-flight Finish: two tool rounds, then the delegate
        // stalls (plain text, no handoff) -- Finish lands during the wait.
        Stub.script = .init(responses: [
            .toolCall(name: "write", args: #"{"path": "a", "content": "x"}"#, cost: 0.0),
            .toolCall(name: "write", args: #"{"path": "b", "content": "x"}"#, cost: 0.0),
            .text(["thinking..."], cost: 0.0),
        ])
        ev = Events(); mb = DelegationMailbox()
        let log3 = CallLog()
        let stopper = mb
        Thread.detachNewThread {
            Thread.sleep(forTimeInterval: 0.3)
            stopper.stop()
        }
        _ = runner(log: log3, events: ev, mailbox: mb).run(call(["task": "t"]))
        f += TestMain.check(ev.endedReason == "the user finished it" && log3.all.count == 2,
                            "Finish ends a stalled long-horizon delegation after real work")

        return f
    }
}
