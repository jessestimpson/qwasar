// Escalation.swift -- `delegate`: a remote sub-agent, embedded and budgeted.
//
// Spec §15, milestones E1+E2: a remote sub-agent over an OpenAI-compatible
// streaming API (OpenRouter is the reference), executed by the HOST like
// `fetch`. E2's addition: the remote model gets the SAME tool surface the
// local one has (minus `delegate` itself -- nested escalation is refused),
// because the inner executor's schemas are already OpenAI function schemas
// and `ToolExecuting` already abstracts the guest: a remote tool call is
// translated, run through the same vsock into the same /work, and its
// result sent back. Serialisation with the local agent is by construction --
// the local turn blocks inside this tool call.
//
// The embedded-session contract: deltas, cost and completion stream out
// through an emit closure while the user's messages and stop requests come
// IN through a mailbox -- the sub-agent is steerable mid-flight by the
// person whose money it is spending. After each completed response the
// runner holds the conversation open for a grace window; a queued user
// message continues it, silence ends it.
//
// Budget (spec §15.3): enforced by declining to send the NEXT request -- a
// request in flight is already paid for. Cost comes from the provider's own
// usage accounting (`usage: {include: true}` on OpenRouter returns
// usage.cost in the final chunk).
//
// The key (spec §15.4): read from KeychainAccess at request time, attached
// by this file, and present nowhere else -- not in the schema, not in any
// event, not in an error. The base URL is NOT model- or config-addressable;
// it is fixed here, overridable only by host code (the test suite's stub).

import Foundation

public struct EscalationPolicy: Sendable {
    public var models: [String]
    /// Remaining for this session, already net of what earlier delegations
    /// spent; the caller derives it from the overlay budget and the record.
    public var sessionRemainingUSD: Double
    public var turnBudgetUSD: Double
    /// Seconds the conversation stays open for the user after each response.
    public var graceSeconds: Double

    public init(models: [String], sessionRemainingUSD: Double,
                turnBudgetUSD: Double, graceSeconds: Double = 10) {
        self.models = models
        self.sessionRemainingUSD = sessionRemainingUSD
        self.turnBudgetUSD = turnBudgetUSD
        self.graceSeconds = graceSeconds
    }

    public var isEnabled: Bool {
        !models.isEmpty && sessionRemainingUSD > 0 && turnBudgetUSD > 0
    }
}

/// What the UI hears about a running delegation. Every event carries enough
/// to render without reaching back into the runner.
public enum DelegationEvent: Sendable {
    case started(model: String, task: String)
    case delta(String)
    /// A user message accepted into the remote conversation.
    case userTurn(String)
    /// Cumulative cost for this delegation, USD.
    case cost(Double)
    /// The conversation is open for steering: the remote model has finished a
    /// response and the grace window is running. Typing holds it open.
    case waiting
    /// The remote model called a tool (E2); the result follows when it lands.
    case toolCall(name: String, arguments: String)
    case toolResult(name: String, result: String)
    case ended(reason: String, costUSD: Double)
}

/// The path INTO a live delegation: user messages and the stop request.
/// Posted from the main actor, drained by the runner on the engine queue.
public final class DelegationMailbox: @unchecked Sendable {
    private let lock = NSLock()
    private var queue: [String] = []
    private var stopped = false
    private var held = false
    private let signal = DispatchSemaphore(value: 0)

    public init() {}

    /// The user is composing: the grace window must not close mid-sentence.
    /// Set while the card's input is non-empty, cleared when it empties.
    public func hold(_ holding: Bool) {
        lock.withLock { held = holding }
        signal.signal()
    }

    var isHeld: Bool { lock.withLock { held } }

    public func post(_ text: String) {
        lock.withLock { queue.append(text) }
        signal.signal()
    }

    public func stop() {
        lock.withLock { stopped = true }
        signal.signal()
    }

    func drain() -> (messages: [String], stopped: Bool) {
        lock.withLock {
            let m = queue; queue = []
            return (m, stopped)
        }
    }

    /// Waits up to `seconds` for anything to arrive. Returns immediately if
    /// something already has.
    func wait(seconds: Double) {
        _ = signal.wait(timeout: .now() + seconds)
    }
}

// MARK: - The runner

public struct DelegateToolRunner: ToolExecuting {
    let inner: ToolExecuting
    let policy: EscalationPolicy
    let emit: @Sendable (DelegationEvent) -> Void
    let mailbox: DelegationMailbox
    /// Host-controlled only (spec §15.4): the default is the one provider;
    /// tests substitute a local stub. Never derived from a model or a config
    /// value.
    let baseURL: URL
    let keyProvider: @Sendable () -> String?

    public init(inner: ToolExecuting, policy: EscalationPolicy,
                mailbox: DelegationMailbox,
                emit: @escaping @Sendable (DelegationEvent) -> Void,
                baseURL: URL = URL(string: "https://openrouter.ai/api/v1")!,
                keyProvider: @escaping @Sendable () -> String? = { KeychainAccess.key() }) {
        self.inner = inner
        self.policy = policy
        self.mailbox = mailbox
        self.emit = emit
        self.baseURL = baseURL
        self.keyProvider = keyProvider
    }

    public var schemas: [String] { inner.schemas + [Self.schema(policy: policy)] }

    /// Built per session rather than frozen: the models, their availability
    /// and the remaining budget are facts the local model cannot weigh
    /// without being told (spec §15.1, §15.5).
    public static func schema(policy: EscalationPolicy) -> String {
        let models = policy.models.joined(separator: ", ")
        let desc = "Delegate a task to a more capable remote model and return its answer. "
            + "Available models: \(models). The first is the default. "
            + "THE REMOTE MODEL HAS YOUR TOOLS: the same read/write/edit/list/grep/bash "
            + "and the rest, acting on the same /work you act on. Changes it makes are "
            + "real -- expect files to differ afterwards, and read back anything you "
            + "depend on rather than assuming the state you left. Its tool activity is "
            + "summarised in the result only; the transcript shows the user the detail. "
            + "Budget: $\(String(format: "%.2f", policy.turnBudgetUSD)) per delegation, "
            + "$\(String(format: "%.2f", policy.sessionRemainingUSD)) remaining this session; "
            + "a delegation that trips its budget returns a partial answer marked as such. "
            + "The user can watch and steer the remote model while it works. "
            + "WHEN TO USE IT: a problem you have genuinely failed at (not merely found long), "
            + "a design question with real stakes, or a review before something irreversible. "
            + "Escalating work you could do yourself spends the user's money for nothing: "
            + "try first, and put what you tried in the task. "
            + "Pack the problem well -- the remote model sees ONLY what you send: "
            + "the task, and any code or context you quote into it."
        let schema: [String: Any] = ["type": "function", "function": [
            "name": "delegate",
            "description": desc,
            "parameters": ["type": "object", "properties": [
                "task": ["type": "string",
                         "description": "The brief: the problem, what you tried, what a good answer looks like."],
                "context": ["type": "string",
                            "description": "Supporting material -- code, errors, constraints -- quoted in full."],
                "model": ["type": "string",
                          "description": "One of the available models; omit for the default."],
            ], "required": ["task"]],
        ]]
        let data = try! JSONSerialization.data(withJSONObject: schema, options: [.sortedKeys])
        return String(decoding: data, as: UTF8.self)
    }

    public var environmentDescription: String {
        inner.environmentDescription + """


        Escalation: `delegate` hands a task to a more capable remote model (\(policy.models.joined(separator: ", "))) under a budget -- $\(String(format: "%.2f", policy.sessionRemainingUSD)) remains this session. The remote model works with YOUR tools on the SAME files: after a delegation, /work reflects whatever it did, so re-read anything you rely on. Reach for it at genuine capability walls, not for effort; say what you already tried.
        """
    }

    public func run(_ call: ToolCall) -> String {
        guard call.name == "delegate" else { return inner.run(call) }
        guard let task = call.arguments["task"], !task.isEmpty else {
            return "error: delegate requires a task"
        }
        let model = call.arguments["model"] ?? policy.models.first ?? ""
        guard policy.models.contains(model) else {
            return "error: \(model) is not an allowed model. Allowed: "
                 + policy.models.joined(separator: ", ")
        }
        guard let key = keyProvider() else {
            return "error: no API key is set. The user can add one via "
                 + "Crucible ▸ Set Escalation API Key…"
        }
        guard policy.isEnabled else { return "error: the escalation budget is exhausted" }

        var brief = task
        if let ctx = call.arguments["context"], !ctx.isEmpty {
            brief += "\n\n---\n\n" + ctx
        }

        // Sync over async, the SandboxToolRunner shape: the engine queue has
        // nothing else to do, and every await below is bounded by a timeout,
        // the budget, or the mailbox's stop.
        let box = ResultBox()
        let done = DispatchSemaphore(value: 0)
        let runner = self
        Task.detached {
            box.set(await runner.converse(model: model, brief: brief, key: key))
            done.signal()
        }
        done.wait()
        return box.get()
    }

    // MARK: The conversation loop

    private func converse(model: String, brief: String, key: String) async -> String {
        emit(.started(model: model, task: brief))
        var messages: [[String: Any]] = [["role": "user", "content": brief]]
        var cost = 0.0
        var toolSteps = 0
        var toolTally: [String: Int] = [:]
        var lastAnswer = ""
        var reason = "the remote model finished"
        let turnCap = min(policy.turnBudgetUSD, policy.sessionRemainingUSD)

        // E2: the remote model works with the SAME surface the local one has.
        // The inner schemas are already OpenAI function schemas; `delegate`
        // is not among them because the wrapper appends itself after inner --
        // which is exactly the "nested escalation refused" rule.
        let tools: [[String: Any]] = inner.schemas.compactMap {
            guard let d = $0.data(using: .utf8) else { return nil }
            return (try? JSONSerialization.jsonObject(with: d)) as? [String: Any]
        }

        loop: while true {
            // Spec §15.3: the ceiling is enforced by declining the NEXT
            // request; what is in flight is paid for.
            if cost >= turnCap { reason = "the budget tripped"; break }
            let r = await streamOnce(model: model, messages: messages,
                                     tools: tools, key: key)
            cost += r.cost
            emit(.cost(cost))
            if let err = r.error {
                if lastAnswer.isEmpty {
                    emit(.ended(reason: "failed: \(err)", costUSD: cost))
                    return "error: delegation failed: \(err)"
                }
                reason = "the request failed (\(err)); the last complete answer stands"
                break
            }

            if !r.toolCalls.isEmpty {
                // The remote model is working. Execute through the inner
                // chain -- the same vsock, the same guest, the same /work --
                // and feed the results back.
                var assistant: [String: Any] = ["role": "assistant",
                                                "content": r.text]
                assistant["tool_calls"] = r.toolCalls.map { tc in
                    ["id": tc.id, "type": "function",
                     "function": ["name": tc.name, "arguments": tc.arguments]]
                }
                messages.append(assistant)
                // No step ceiling, deliberately (spec §15.2): a delegation is
                // long-horizon by design, and its governors are the DOLLAR
                // budgets and the user's Stop -- both visible in the card,
                // where a looping agent is watched rather than guessed at.
                for tc in r.toolCalls {
                    toolSteps += 1
                    toolTally[tc.name, default: 0] += 1
                    emit(.toolCall(name: tc.name, arguments: tc.arguments))
                    let result: String
                    if tc.name == "delegate" {
                        result = "error: nested escalation is not available"
                    } else {
                        result = inner.run(ToolCall(name: tc.name,
                                                    arguments: Self.arguments(from: tc.arguments)))
                    }
                    emit(.toolResult(name: tc.name, result: result))
                    messages.append(["role": "tool", "tool_call_id": tc.id,
                                     "content": result])
                }
                // Steering lands between steps too -- no grace wait while the
                // model is mid-work, just a drain.
                let (queued, stopped) = mailbox.drain()
                if stopped { reason = "the user ended it"; break }
                for m in queued {
                    emit(.userTurn(m))
                    messages.append(["role": "user", "content": m])
                }
                continue
            }

            lastAnswer = r.text
            messages.append(["role": "assistant", "content": r.text])

            // The grace window: the conversation stays open briefly so the
            // user can continue it; a queued message continues, stop or
            // silence ends. While the user is TYPING the window does not
            // close -- a countdown that expires mid-sentence is worse than no
            // window -- bounded so an abandoned draft cannot hold the local
            // turn forever.
            var (queued, stopped) = mailbox.drain()
            if queued.isEmpty && !stopped {
                emit(.waiting)
                let deadline = Date().addingTimeInterval(180)
                while true {
                    mailbox.wait(seconds: policy.graceSeconds)
                    (queued, stopped) = mailbox.drain()
                    if !queued.isEmpty || stopped { break }
                    if !mailbox.isHeld || Date() > deadline { break }
                }
            }
            if stopped { reason = "the user ended it"; break }
            if queued.isEmpty { break }
            for m in queued {
                emit(.userTurn(m))
                messages.append(["role": "user", "content": m])
            }
        }

        emit(.ended(reason: reason, costUSD: cost))
        var header = "delegation to \(model): \(reason) · $\(String(format: "%.4f", cost))"
        if !toolTally.isEmpty {
            // The local model must know /work moved under it, per delegation
            // and concretely -- the schema's general warning is not evidence.
            let tally = toolTally.sorted { $0.key < $1.key }
                .map { "\($0.key)×\($0.value)" }.joined(separator: ", ")
            header += " · it used tools on /work: \(tally) -- re-read anything you depend on"
        }
        return header + "\n\n" + lastAnswer
    }

    /// OpenAI tool arguments arrive as a JSON object in a string; ToolCall
    /// carries [String: String]. Non-string values are re-encoded, which is
    /// the same convention the guest's own dispatch applies.
    static func arguments(from json: String) -> [String: String] {
        guard let d = json.data(using: .utf8),
              let obj = (try? JSONSerialization.jsonObject(with: d)) as? [String: Any]
        else { return ["input": json] }
        var out: [String: String] = [:]
        for (k, v) in obj {
            if let s = v as? String { out[k] = s }
            else if let dd = try? JSONSerialization.data(withJSONObject: v,
                                                         options: [.fragmentsAllowed]) {
                out[k] = String(decoding: dd, as: UTF8.self)
            }
        }
        return out
    }

    struct StreamResult {
        var text = ""
        var toolCalls: [(id: String, name: String, arguments: String)] = []
        var cost = 0.0
        var error: String?
    }

    /// One streaming completion: text deltas, streamed tool-call fragments
    /// assembled by index, and the provider's cost from the final chunk.
    private func streamOnce(model: String, messages: [[String: Any]],
                            tools: [[String: Any]], key: String) async -> StreamResult {
        var req = URLRequest(url: baseURL.appendingPathComponent("chat/completions"))
        req.httpMethod = "POST"
        req.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization")
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.timeoutInterval = 300
        var body: [String: Any] = ["model": model, "messages": messages,
                                   "stream": true,
                                   "usage": ["include": true]]
        if !tools.isEmpty { body["tools"] = tools }
        req.httpBody = try? JSONSerialization.data(withJSONObject: body)

        var r = StreamResult()
        // index -> (id, name, argument fragments); OpenAI streams a call's
        // name once and its arguments as pieces.
        var calls: [Int: (id: String, name: String, args: String)] = [:]
        do {
            let (bytes, response) = try await session().bytes(for: req)
            guard let http = response as? HTTPURLResponse else {
                r.error = "not an HTTP response"; return r
            }
            guard http.statusCode == 200 else {
                // The body may explain; read a little of it, never log the
                // request (it carries the Authorization header).
                var detail = ""
                for try await line in bytes.lines { detail += line; if detail.count > 300 { break } }
                r.error = "HTTP \(http.statusCode): \(detail.prefix(300))"; return r
            }
            for try await line in bytes.lines {
                guard line.hasPrefix("data:") else { continue }
                let payload = line.dropFirst(5).trimmingCharacters(in: .whitespaces)
                if payload == "[DONE]" { break }
                guard let d = payload.data(using: .utf8),
                      let obj = try? JSONSerialization.jsonObject(with: d) as? [String: Any]
                else { continue }
                if let choices = obj["choices"] as? [[String: Any]],
                   let delta = choices.first?["delta"] as? [String: Any] {
                    if let piece = delta["content"] as? String, !piece.isEmpty {
                        r.text += piece
                        emit(.delta(piece))
                    }
                    for tc in (delta["tool_calls"] as? [[String: Any]]) ?? [] {
                        let idx = tc["index"] as? Int ?? 0
                        var cur = calls[idx] ?? (id: "", name: "", args: "")
                        if let id = tc["id"] as? String { cur.id = id }
                        if let fn = tc["function"] as? [String: Any] {
                            if let n = fn["name"] as? String { cur.name += n }
                            if let a = fn["arguments"] as? String { cur.args += a }
                        }
                        calls[idx] = cur
                    }
                }
                if let usage = obj["usage"] as? [String: Any],
                   let c = usage["cost"] as? Double {
                    r.cost = c
                }
            }
            r.toolCalls = calls.sorted { $0.key < $1.key }.map {
                (id: $0.value.id.isEmpty ? "call_\($0.key)" : $0.value.id,
                 name: $0.value.name, arguments: $0.value.args)
            }
            return r
        } catch {
            r.error = (error as NSError).localizedDescription
            return r
        }
    }

    /// Overridable for tests via URLProtocol; public so the suite can install
    /// path; nonisolated static so the suite can install a stub.
    nonisolated(unsafe) public static var sessionConfiguration: URLSessionConfiguration = .ephemeral

    private func session() -> URLSession {
        URLSession(configuration: Self.sessionConfiguration)
    }
}

private final class ResultBox: @unchecked Sendable {
    private var value = ""
    private let lock = NSLock()
    func set(_ s: String) { lock.withLock { value = s } }
    func get() -> String { lock.withLock { value } }
}
