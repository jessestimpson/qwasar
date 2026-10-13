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
}

public enum ApprovalAnswer: Sendable, Equatable {
    case allow
    /// `reason`, if the user gave one, goes to the model.
    case decline(reason: String?)
}

/// Asks the user.  Called off the main actor, from inside a tool call, and
/// may block until they answer.
public typealias Approver = @Sendable (ApprovalRequest) -> ApprovalAnswer

public struct GatedToolRunner: ToolExecuting {
    public let inner: ToolExecuting
    public let context: PolicyContext
    public let approver: Approver

    public init(inner: ToolExecuting, context: PolicyContext, approver: @escaping Approver) {
        self.inner = inner
        self.context = context
        self.approver = approver
    }

    /// Fails closed: with no one to ask, every question is a no.
    public static let headless: Approver = { _ in
        .decline(reason: "this run is headless, so no one is here to approve it")
    }

    public var schemas: [String] { inner.schemas }

    public var environmentDescription: String {
        inner.environmentDescription + """


        Some tool calls ask the user first: deleting or changing things outside the project directory, using the network, rewriting git history, installing software, and commands this app does not recognise. If the user declines, the result says so and why; do not retry the same thing another way -- choose a different approach or ask them. A few commands are always refused (running as root, erasing disks, deleting the home directory or the project).
        """
    }

    public func run(_ call: ToolCall) -> String {
        switch ToolPolicy.classify(call, context) {
        case .allow:
            return inner.run(call)
        case .deny(let reason):
            return "error: refused: \(reason). This is never run by the agent; "
                 + "if it is really needed, the user will run it themselves."
        case .ask(let reasons, let high):
            switch approver(ApprovalRequest(call: call, reasons: reasons, high: high)) {
            case .allow:
                return inner.run(call)
            case .decline(let why):
                return "error: the user declined this call (" + reasons.joined(separator: "; ") + ")."
                     + (why.map { " Their reason: \($0)" } ?? "")
                     + " Do not retry it another way; choose a different approach, or ask the user."
            }
        }
    }
}
