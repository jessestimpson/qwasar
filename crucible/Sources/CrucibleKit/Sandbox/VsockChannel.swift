// VsockChannel.swift -- framed JSON over the guest's control socket.
//
// PLAN.md 6.4. Four-byte big-endian length, then a JSON payload. That framing
// is not a preference: it is exactly what Erlang's `{packet, 4}` means on the
// port side, so the guest gets whole messages with no parsing of its own and
// this side gets a two-line read loop.
//
// Every request carries a host-assigned id and every response echoes it.
// Unsolicited events (logs, module-load notices) carry no id. Every request has
// a timeout, and a timeout is reported to the model as a tool result rather
// than as a failed turn -- the same principle the C agent applies to a declined
// confirmation: tell it, and let it choose something else.

import Foundation

/// An argument value. Not `[String: String]`: the guest guards on types --
/// `sleep` wants an integer and refuses a string that looks like one -- and a
/// protocol that can only carry strings pushes that coercion onto every op.
public enum GuestValue: Encodable, Sendable {
    case string(String)
    case int(Int)
    case bool(Bool)
    /// Nested, because `invoke` carries a tool's own arguments inside the
    /// envelope: `{"name": "grep_ast", "args": {...}}`.
    indirect case object([String: GuestValue])

    public func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .string(let v): try c.encode(v)
        case .int(let v):    try c.encode(v)
        case .bool(let v):   try c.encode(v)
        case .object(let v): try c.encode(v)
        }
    }
}

public struct GuestRequest: Encodable, Sendable {
    public var id: Int
    public var op: String
    public var tool: String?
    public var args: [String: GuestValue]?

    public init(id: Int, op: String, tool: String? = nil, args: [String: GuestValue]? = nil) {
        self.id = id
        self.op = op
        self.tool = tool
        self.args = args
    }
}

public struct GuestResponse: Decodable, Sendable {
    public var id: Int?
    public var ok: Bool?
    public var result: String?
    public var error: String?
    public var kind: String?
    public var took_ms: Int?
    /// Unsolicited events carry `event` instead of `id`.
    public var event: String?
    public var level: String?
    public var text: String?
}

public enum VsockError: Error, CustomStringConvertible {
    case closed
    case timedOut(Int)
    case oversized(Int)
    case malformed(String)

    public var description: String {
        switch self {
        case .closed:          return "the guest closed the control channel"
        case .timedOut(let s): return "the guest did not answer within \(s)s"
        case .oversized(let n): return "the guest sent an oversized frame (\(n) bytes)"
        case .malformed(let m): return "malformed frame from the guest: \(m)"
        }
    }
}

/// One control channel to one guest.
///
/// Deliberately not an actor: it owns a file descriptor and a reader thread,
/// and the reader must not be able to suspend. Access is serialised by a lock,
/// which is the smaller and more honest mechanism here.
public final class VsockChannel: @unchecked Sendable {
    /// A frame larger than this is a protocol error, not a big message. The
    /// guest never has a legitimate reason to send one -- tool results are
    /// capped far below it -- and accepting one would let a compromised guest
    /// exhaust the host's memory through the one channel it has.
    public static let maxFrameBytes = 8 * 1024 * 1024

    private let fd: Int32
    private let lock = NSLock()
    private var nextID = 1
    private var pending: [Int: (Result<GuestResponse, Error>) -> Void] = [:]
    private var events: ((GuestResponse) -> Void)?
    private var closed = false
    private var reader: Thread?

    public init(fileDescriptor: Int32) {
        self.fd = fileDescriptor
    }

    /// Scoped locking throughout: Swift 6 makes bare `lock()`/`unlock()`
    /// unavailable from async contexts, and rightly -- a suspension between the
    /// two is a deadlock waiting for a slow guest.
    private func locked<R>(_ body: () -> R) -> R { lock.withLock(body) }

    public func onEvent(_ handler: @escaping (GuestResponse) -> Void) {
        locked { events = handler }
    }

    public func start() {
        let t = Thread { [weak self] in self?.readLoop() }
        t.name = "dev.crucible.vsock"
        t.stackSize = 512 * 1024
        reader = t
        t.start()
    }

    public func close() {
        let waiting: [Int: (Result<GuestResponse, Error>) -> Void]? = locked {
            if closed { return nil }
            closed = true
            let w = pending
            pending = [:]
            return w
        }
        guard let waiting else { return }
        Darwin.close(fd)
        for (_, k) in waiting { k(.failure(VsockError.closed)) }
    }

    // MARK: Requests

    public func send(op: String, tool: String? = nil, args: [String: GuestValue]? = nil,
                     timeout: Int = 30) async throws -> GuestResponse {
        let id = locked { () -> Int in
            let n = nextID
            nextID += 1
            return n
        }

        let req = GuestRequest(id: id, op: op, tool: tool, args: args)
        let body = try JSONEncoder().encode(req)

        return try await withCheckedThrowingContinuation { cont in
            var resumed = false
            let resumeOnce: (Result<GuestResponse, Error>) -> Void = { r in
                guard !resumed else { return }
                resumed = true
                cont.resume(with: r)
            }

            let isClosed = locked { () -> Bool in
                if closed { return true }
                pending[id] = resumeOnce
                return false
            }
            if isClosed { resumeOnce(.failure(VsockError.closed)); return }

            // The deadline is enforced here rather than on the guest, because a
            // guest that has stopped answering is exactly the case a
            // guest-enforced timeout cannot cover.
            DispatchQueue.global().asyncAfter(deadline: .now() + .seconds(timeout)) { [weak self] in
                guard let self else { return }
                let k = self.locked { self.pending.removeValue(forKey: id) }
                k?(.failure(VsockError.timedOut(timeout)))
            }

            do { try writeFrame(body) }
            catch {
                _ = locked { pending.removeValue(forKey: id) }
                resumeOnce(.failure(error))
            }
        }
    }

    // MARK: Wire

    private func writeFrame(_ body: Data) throws {
        var header = withUnsafeBytes(of: UInt32(body.count).bigEndian) { Data($0) }
        header.append(body)
        try header.withUnsafeBytes { raw in
            var p = raw.bindMemory(to: UInt8.self).baseAddress!
            var n = raw.count
            while n > 0 {
                let w = write(fd, p, n)
                if w < 0 {
                    if errno == EINTR { continue }
                    throw VsockError.closed
                }
                if w == 0 { throw VsockError.closed }
                p += w
                n -= w
            }
        }
    }

    private func readExactly(_ n: Int) -> Data? {
        var out = Data(count: n)
        var got = 0
        while got < n {
            let r: Int = out.withUnsafeMutableBytes { raw in
                read(fd, raw.baseAddress!.advanced(by: got), n - got)
            }
            if r < 0 {
                if errno == EINTR { continue }
                return nil
            }
            if r == 0 { return nil }
            got += r
        }
        return out
    }

    private func readLoop() {
        let dec = JSONDecoder()
        while true {
            guard let head = readExactly(4) else { break }
            let len = Int(head.withUnsafeBytes { $0.loadUnaligned(as: UInt32.self).bigEndian })
            if len == 0 { continue }
            if len > Self.maxFrameBytes {
                // Cannot resynchronise a stream whose framing we no longer
                // trust, so the channel ends rather than guessing.
                fail(VsockError.oversized(len))
                break
            }
            guard let body = readExactly(len) else { break }
            guard let msg = try? dec.decode(GuestResponse.self, from: body) else {
                fail(VsockError.malformed(String(decoding: body.prefix(200), as: UTF8.self)))
                break
            }

            if let id = msg.id {
                locked { pending.removeValue(forKey: id) }?(.success(msg))
            } else {
                locked { events }?(msg)
            }
        }
        close()
    }

    private func fail(_ e: Error) {
        let waiting = locked { () -> [Int: (Result<GuestResponse, Error>) -> Void] in
            let w = pending
            pending = [:]
            return w
        }
        for (_, k) in waiting { k(.failure(e)) }
    }
}
