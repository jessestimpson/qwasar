// Escalation.swift -- `delegate`: a remote sub-agent, embedded and budgeted.
//
// Spec §15, milestone E1: text-in/text-out consultation against an
// OpenAI-compatible streaming API (OpenRouter is the reference), executed by
// the HOST like `fetch` -- the guest is not in the loop and stays
// network-less. The local turn blocks inside the tool call, which is the
// truthful state: the local model is waiting on the expensive one.
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
    case ended(reason: String, costUSD: Double)
}

/// The path INTO a live delegation: user messages and the stop request.
/// Posted from the main actor, drained by the runner on the engine queue.
public final class DelegationMailbox: @unchecked Sendable {
    private let lock = NSLock()
    private var queue: [String] = []
    private var stopped = false
    private let signal = DispatchSemaphore(value: 0)

    public init() {}

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


        Escalation: `delegate` hands a task to a more capable remote model (\(policy.models.joined(separator: ", "))) under a budget -- $\(String(format: "%.2f", policy.sessionRemainingUSD)) remains this session. Reach for it at genuine capability walls, not for effort; say what you already tried.
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
        var messages: [[String: String]] = [["role": "user", "content": brief]]
        var cost = 0.0
        var lastAnswer = ""
        var reason = "the remote model finished"
        let turnCap = min(policy.turnBudgetUSD, policy.sessionRemainingUSD)

        loop: while true {
            // Spec §15.3: the ceiling is enforced by declining the NEXT
            // request; what is in flight is paid for.
            if cost >= turnCap { reason = "the budget tripped"; break }
            let (text, spent, err) = await streamOnce(model: model, messages: messages, key: key)
            cost += spent
            emit(.cost(cost))
            if let err {
                if lastAnswer.isEmpty {
                    emit(.ended(reason: "failed: \(err)", costUSD: cost))
                    return "error: delegation failed: \(err)"
                }
                reason = "the request failed (\(err)); the last complete answer stands"
                break
            }
            lastAnswer = text
            messages.append(["role": "assistant", "content": text])

            // The grace window: the conversation stays open briefly so the
            // user can continue it; a queued message continues, stop or
            // silence ends.
            var (queued, stopped) = mailbox.drain()
            if queued.isEmpty && !stopped {
                mailbox.wait(seconds: policy.graceSeconds)
                (queued, stopped) = mailbox.drain()
            }
            if stopped { reason = "the user ended it"; break }
            if queued.isEmpty { break }
            for m in queued {
                emit(.userTurn(m))
                messages.append(["role": "user", "content": m])
            }
        }

        emit(.ended(reason: reason, costUSD: cost))
        let header = "delegation to \(model): \(reason) · $\(String(format: "%.4f", cost))"
        return header + "\n\n" + lastAnswer
    }

    /// One streaming completion. Returns the text, the cost the provider
    /// reported, and an error if the request failed.
    private func streamOnce(model: String, messages: [[String: String]],
                            key: String) async -> (String, Double, String?) {
        var req = URLRequest(url: baseURL.appendingPathComponent("chat/completions"))
        req.httpMethod = "POST"
        req.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization")
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.timeoutInterval = 300
        let body: [String: Any] = ["model": model, "messages": messages,
                                   "stream": true,
                                   "usage": ["include": true]]
        req.httpBody = try? JSONSerialization.data(withJSONObject: body)

        var text = ""
        var cost = 0.0
        do {
            let (bytes, response) = try await session().bytes(for: req)
            guard let http = response as? HTTPURLResponse else {
                return ("", 0, "not an HTTP response")
            }
            guard http.statusCode == 200 else {
                // The body may explain; read a little of it, never log the
                // request (it carries the Authorization header).
                var detail = ""
                for try await line in bytes.lines { detail += line; if detail.count > 300 { break } }
                return ("", 0, "HTTP \(http.statusCode): \(detail.prefix(300))")
            }
            for try await line in bytes.lines {
                guard line.hasPrefix("data:") else { continue }
                let payload = line.dropFirst(5).trimmingCharacters(in: .whitespaces)
                if payload == "[DONE]" { break }
                guard let d = payload.data(using: .utf8),
                      let obj = try? JSONSerialization.jsonObject(with: d) as? [String: Any]
                else { continue }
                if let choices = obj["choices"] as? [[String: Any]],
                   let delta = choices.first?["delta"] as? [String: Any],
                   let piece = delta["content"] as? String, !piece.isEmpty {
                    text += piece
                    emit(.delta(piece))
                }
                if let usage = obj["usage"] as? [String: Any],
                   let c = usage["cost"] as? Double {
                    cost = c
                }
            }
            return (text, cost, nil)
        } catch {
            return (text, cost, (error as NSError).localizedDescription)
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
