// ToolSession.swift -- what a session's tools remember between calls, and how
// the Stop button reaches a tool that is running.
//
// A runner is rebuilt for every turn (AppState.send), so anything that must
// outlive a turn -- what the model has already read -- lives here, owned by
// the session and handed to each turn's runner.

import Darwin
import Foundation

/// The reads a session has made, keyed by path and range, with a hash of what
/// each returned.  Reading the same range of an unchanged file again is an
/// error: its contents are already in the conversation, and a model told so
/// in a tool result learns, within the session, to use what it has.  A file
/// that changed -- by an Edit, a Write, a build, the user -- reads normally.
public final class ReadLedger: @unchecked Sendable {
    private let lock = NSLock()
    private var seen: [String: Int] = [:]

    public init() {}

    /// Records `output` as what `path`'s `range` read; true if exactly that
    /// was recorded before.
    func repeats(path: String, range: String, output: String) -> Bool {
        let key = path + "\u{0}" + range
        var h = Hasher()
        h.combine(output)
        let v = h.finalize()
        return lock.withLock {
            defer { seen[key] = v }
            return seen[key] == v
        }
    }

    /// Forgets `path`, so its next read goes through whatever it returns.
    func forget(path: String) {
        let prefix = path + "\u{0}"
        lock.withLock { seen = seen.filter { !$0.key.hasPrefix(prefix) } }
    }
}

/// Stop, as the tools hear it.  `interrupt()` runs every registered handler
/// -- a host command's process group killed, a guest command cancelled -- and
/// holds until `reset()`, so a call that starts after Stop does not run.
public final class ToolInterrupt: @unchecked Sendable {
    /// One per app: one turn runs at a time, and Stop stops it.
    public static let shared = ToolInterrupt()

    private let lock = NSLock()
    private var interrupted = false
    private var next = 0
    private var handlers: [Int: @Sendable () -> Void] = [:]

    public init() {}

    public var isInterrupted: Bool { lock.withLock { interrupted } }

    /// A new turn: Stop has not been pressed in it.
    public func reset() { lock.withLock { interrupted = false } }

    public func interrupt() {
        let hs: [@Sendable () -> Void] = lock.withLock {
            interrupted = true
            let h = Array(handlers.values)
            handlers = [:]
            return h
        }
        for h in hs { h() }
    }

    /// Registers `handler` for the work about to run; nil, and the handler is
    /// not kept, if Stop was already pressed -- the caller should not start.
    func register(_ handler: @escaping @Sendable () -> Void) -> Int? {
        lock.withLock {
            if interrupted { return nil }
            next += 1
            handlers[next] = handler
            return next
        }
    }

    func unregister(_ token: Int?) {
        guard let token else { return }
        _ = lock.withLock { handlers.removeValue(forKey: token) }
    }
}
