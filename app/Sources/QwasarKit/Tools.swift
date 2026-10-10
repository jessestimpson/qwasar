// Tools.swift -- the calls a model makes.

import Foundation

public struct ToolCall: Sendable, Equatable {
    public var name: String
    public var arguments: [String: String]

    public init(name: String, arguments: [String: String]) {
        self.name = name
        self.arguments = arguments
    }

    public func argument(_ key: String) -> String? { arguments[key] }
}

/// Anything that can run a tool call for a session.
public protocol ToolExecuting: Sendable {
    func run(_ call: ToolCall) -> String
    /// Which tools the model should be told about. The surface and the
    /// executor have to agree, or the model is offered something that will
    /// come back "no such tool".
    var schemas: [String] { get }
    /// What the model is told about where it is working.
    ///
    /// This belongs to the executor and not to the project's settings, because
    /// it is a fact rather than a preference: whether the model can write, and
    /// what it writes to, is decided by which of these is running. A project's
    /// system prompt described a read-only world for three milestones after the
    /// tools stopped being read-only, and the model believed it -- correctly
    /// refusing to edit anything, and saying so. Derived from the executor, it
    /// cannot drift.
    var environmentDescription: String { get }
}
