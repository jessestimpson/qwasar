// SandboxToolRunner.swift -- tool calls, executed in the guest.
//
// PLAN.md 3.1: the host never runs a model-requested tool. This is what makes
// that true. A call the model produced is validated, sent over the vsock, and
// run by the warden against `/work` -- the user's own tree, mounted
// read-write (spec 7.4) -- so its writes are framework-confined to that one
// directory, and the real .git inside it is shadowed out of reach.
//
// The interface is deliberately synchronous. `Session.runTurn` runs on the
// engine queue and has nothing else to do while a tool executes, and making it
// async would mean suspending a loop whose whole contract is that it owns the
// engine for the duration of a turn. The wait is bounded, and the channel's
// reader lives on its own thread, so nothing here can deadlock against the
// engine.

import Foundation

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

extension ToolRunner: ToolExecuting {
    public var schemas: [String] { ToolSurface.readOnlySchemas }

    public var environmentDescription: String {
        """
        You are working directly against the user's own files, with READ-ONLY access. You can read files, list directories, and search with regular expressions. You cannot write, edit, or run commands, and there is no sandbox: investigate and explain, and describe any change you would make rather than attempting it.
        """
    }
}

public struct SandboxToolRunner: ToolExecuting {
    let channel: VsockChannel
    /// Per-call ceiling. Generous next to the tools' own timeouts, because a
    /// guest that has stopped answering is the case this covers -- not a slow
    /// `bash`, which the warden kills on its own and reports.
    let timeout: Int

    public init(channel: VsockChannel, timeout: Int = 180) {
        self.channel = channel
        self.timeout = timeout
    }

    public var schemas: [String] { ToolSurface.guestSchemas }

    public var environmentDescription: String {
        """
        You are working inside a sandbox: a virtual machine with no network. /work IS the user's real project directory, mounted read-write: every edit you make lands in their working tree immediately, and they may be editing files right beside you. Nothing else of their machine is reachable from here.

        Work directly. Make the change, then verify it by reading the file back or running a command. Do not ask permission to edit a file or run something -- editing the working tree is exactly what you are here for.

        Version control is the user's, not yours: they review your edits with their own tools and commit what they accept. Do not commit, branch, stage, or otherwise operate git -- your job is the files themselves.

        You can also extend yourself: `define` compiles and hot-loads an Elixir module, and one implementing the Crucible.Skill behaviour becomes a SKILL -- invokable through `invoke`, listed by `skills`, and owned by the project, so every session here has it. Worth doing for something you will need repeatedly; not worth it for one-off work, since writing a module costs far more than doing the task by hand.

        Paths are relative to the project root.
        """
    }

    public func run(_ call: ToolCall) -> String {
        guard ToolSurface.names.contains(call.name) else {
            return "error: no such tool: \(call.name). Available: "
                 + ToolSurface.names.sorted().joined(separator: ", ")
        }

        var args: [String: GuestValue] = [:]
        for (k, v) in call.arguments { args[k] = .string(v) }

        // Bridging async to sync on purpose (see the file comment). The
        // continuation is resumed by the channel's reader thread or by its own
        // timeout, never by this one, so the wait always ends.
        let box = ResultBox()
        let done = DispatchSemaphore(value: 0)
        Task.detached { [channel, timeout] in
            do {
                let r = try await channel.send(op: call.name, args: args, timeout: timeout)
                box.set(Self.render(r))
            } catch {
                // A failed tool is reported to the model as a tool result, not
                // as a failed turn -- the same principle the C agent applies to
                // a declined confirmation. It can then choose something else.
                box.set("error: \(error)")
            }
            done.signal()
        }
        done.wait()
        return box.get()
    }

    private static func render(_ r: GuestResponse) -> String {
        if r.ok == true { return r.result ?? "" }
        let kind = r.kind.map { "\($0): " } ?? ""
        return "error: \(kind)\(r.error ?? "the guest gave no reason")"
    }
}

/// A one-shot mailbox between the detached task and the waiting caller.
private final class ResultBox: @unchecked Sendable {
    private var value = ""
    private let lock = NSLock()
    func set(_ s: String) { lock.withLock { value = s } }
    func get() -> String { lock.withLock { value } }
}
