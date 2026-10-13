// ToolGate.swift -- the one choke point every tool call passes.  Spec §16.2.
//
// A GatedToolRunner wraps the executor, not the turn loop, because
// `delegate` calls its inner executor directly (Escalation.swift): a gate in
// AppState's loop would never see a remote model's calls.  Wrapping the
// executor covers the local turn, the delegated one and the headless gate
// with one piece of code.
//
// ToolPolicy decides; this enforces.  A refused or declined call is a tool
// result, never a failed turn -- the model is told why and chooses again,
// as the C agent's refused confirmations always were.

import Foundation

/// What the user is asked.
public struct ApprovalRequest: Sendable {
    public var call: ToolCall
    public var reasons: [String]
    public var high: Bool

    public init(call: ToolCall, reasons: [String], high: Bool) {
        self.call = call
        self.reasons = reasons
        self.high = high
    }
}

public enum ApprovalAnswer: Sendable, Equatable {
    case allow
    /// Allow, and stop asking about these reasons in this session.
    case allowForSession
    /// `reason`, if the user gave one, goes to the model.
    case decline(reason: String?)
}

/// `ask`, the default, shows the card; `trusted` runs what would have asked
/// and refuses only the catastrophic set.  A person's choice for a project
/// (spec §16.6), never the default.
public enum ToolApprovalMode: String, Codable, Sendable, CaseIterable {
    case ask, trusted
}

/// "Allow for this session": the reasons a user has approved, so a later
/// call flagged for the same reasons -- all of them -- runs without asking.
/// A call with a reason not yet approved still asks; widening is always the
/// user's, in the card.  Keyed by the policy's reason text, so a reason
/// reworded in a later version simply asks again: the safe direction.
public final class StandingApprovals: @unchecked Sendable {
    private let lock = NSLock()
    private var approved: Set<String>
    private let onAdd: @Sendable ([String]) -> Void

    /// `onAdd` hears each grant, to persist it.
    public init(_ initial: [String] = [], onAdd: @escaping @Sendable ([String]) -> Void = { _ in }) {
        approved = Set(initial)
        self.onAdd = onAdd
    }

    public var all: [String] { lock.withLock { approved.sorted() } }

    public func covers(_ reasons: [String]) -> Bool {
        lock.withLock { !reasons.isEmpty && reasons.allSatisfy(approved.contains) }
    }

    public func add(_ reasons: [String]) {
        let new: [String] = lock.withLock {
            let n = reasons.filter { !approved.contains($0) }
            approved.formUnion(n)
            return n
        }
        if !new.isEmpty { onAdd(new) }
    }
}

/// Asks the user.  Called off the main actor, from inside a tool call, and
/// may block until they answer.
public typealias Approver = @Sendable (ApprovalRequest) -> ApprovalAnswer

public struct GatedToolRunner: ToolExecuting {
    public let inner: ToolExecuting
    public let context: PolicyContext
    public let approver: Approver
    public let mode: ToolApprovalMode
    public let approvals: StandingApprovals
    /// A line for the transcript about every call that asked or was
    /// refused, and what became of it: nothing here is quiet.
    public let note: (@Sendable (String) -> Void)?

    public init(inner: ToolExecuting, context: PolicyContext, mode: ToolApprovalMode = .ask,
                approvals: StandingApprovals = StandingApprovals(),
                note: (@Sendable (String) -> Void)? = nil, approver: @escaping Approver) {
        self.inner = inner
        self.context = context
        self.mode = mode
        self.approvals = approvals
        self.note = note
        self.approver = approver
    }

    /// Fails closed: with no one to ask, every question is a no.
    public static let headless: Approver = { _ in
        .decline(reason: "this run is headless, so no one is here to approve it")
    }

    public var schemas: [String] { inner.schemas }

    public var environmentDescription: String {
        if mode == .trusted {
            return inner.environmentDescription + """


            A few commands are always refused, and the result says so: running as root, erasing disks, deleting the home directory or the project.
            """
        }
        return inner.environmentDescription + """


        Some tool calls ask the user first: deleting or changing things outside the project directory, using the network, rewriting git history, installing software, and commands this app does not recognise. If the user declines, the result says so and why; do not retry the same thing another way -- choose a different approach or ask them. A few commands are always refused (running as root, erasing disks, deleting the home directory or the project).
        """
    }

    public func run(_ call: ToolCall) -> String {
        switch ToolPolicy.classify(call, context) {
        case .allow:
            return inner.run(call)
        case .deny(let reason):
            note?("refused \(call.name): \(reason)")
            return "error: refused: \(reason). This is never run by the agent; "
                 + "if it is really needed, the user will run it themselves."
        case .ask(let reasons, let high, let grant):
            let what = reasons.joined(separator: "; ")
            // Approved, it runs with containment widened by exactly what it
            // was flagged for (spec §16.5) -- a task-local the model cannot
            // set, never an argument it could.
            let approved = { ContainmentGrants.$current.withValue(grant) { inner.run(call) } }
            if mode == .trusted {
                note?("ran without asking (tool_approval is trusted): \(what)")
                return approved()
            }
            if approvals.covers(reasons) {
                note?("ran on a standing approval: \(what)")
                return approved()
            }
            switch approver(ApprovalRequest(call: call, reasons: reasons, high: high)) {
            case .allow:
                note?("allowed once: \(what)")
                return approved()
            case .allowForSession:
                approvals.add(reasons)
                note?("allowed for this session: \(what)")
                return approved()
            case .decline(let why):
                note?("declined: \(what)" + (why.map { " -- \($0)" } ?? ""))
                return "error: the user declined this call (" + reasons.joined(separator: "; ") + ")."
                     + (why.map { " Their reason: " + $0 + (".!?".contains($0.last ?? ".") ? "" : ".") } ?? "")
                     + " Do not retry it another way; choose a different approach, or ask the user."
            }
        }
    }
}
