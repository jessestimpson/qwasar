// QwasarClient.swift -- the Session API (API.md), from Swift.
//
// The app no longer links the engine.  Sessions live on qwasar-server, which
// the app runs as a helper; this is how the window talks to them.  Thin by
// design: JSON in, events out.  The parsing of a server-sent event stream
// is a value type of its own (SSEParser) so it can be tested without a
// socket.

import Foundation

public enum ClientError: Error, CustomStringConvertible, Sendable {
    case unreachable(String)
    /// The server answered with a status and a reason (API.md §3).
    case refused(status: Int, code: String, message: String)
    case malformed(String)

    public var description: String {
        switch self {
        case .unreachable(let m): return "cannot reach the server: \(m)"
        case .refused(let s, _, let m): return "\(m) (HTTP \(s))"
        case .malformed(let m): return "unexpected answer from the server: \(m)"
        }
    }

    public var status: Int? { if case .refused(let s, _, _) = self { return s } else { return nil } }
}

// MARK: - What the server says about itself and its sessions

public struct ServerInfo: Decodable, Sendable {
    public struct Model: Decodable, Sendable {
        public var id: String
        public var name: String
        public var family: String
        public var path: String
    }
    public struct Profile: Decodable, Sendable {
        public var physical_bytes: UInt64
        public var working_set_bytes: UInt64
        public var weights_bytes: UInt64
        public var kv_bytes_per_token: UInt64
        public var session_fixed_bytes: UInt64
        public var reserve: Double
        public var max_context: Int
        public var note: String?
    }
    public struct Counts: Decodable, Sendable {
        public var total: Int
        public var live: Int
        public var queued: Int
    }
    public var model: Model
    public var context: Int
    public var live_sessions: Int
    public var profile: Profile
    public var capabilities: [String: Bool]
    public var sessions: Counts
    public var state_dir: String
    public struct Disk: Decodable, Sendable {
        public var sessions_bytes: UInt64
        public var cache_bytes: UInt64
        public var free_bytes: UInt64
    }
    /// Absent from a server older than M4.
    public var disk: Disk?

    /// The profile's arithmetic, as the toolbar's help shows it.
    public var summary: String {
        let gb = { (b: UInt64) in String(format: "%.2f GB", Double(b) / 1e9) }
        return """
        \(model.name) · physical \(gb(profile.physical_bytes)) · Metal working set \(gb(profile.working_set_bytes)) \
        · reserve \(Int(profile.reserve * 100))%
        weights \(gb(profile.weights_bytes)) · \(profile.kv_bytes_per_token / 1024) KB/token
        → \(live_sessions) live session\(live_sessions == 1 ? "" : "s") at \(context) tokens
        """ + (profile.note.map { "\n\($0)" } ?? "")
    }
}

public struct Warmth: Decodable, Sendable, Equatable {
    public var state: String          // "live" | "warm" | "cold"
    public var covered: Int
    public var estimate_seconds: Double?
}

public struct SessionInfo: Decodable, Sendable {
    public struct PendingCall: Decodable, Sendable { public var id: String; public var name: String }
    public struct LastStep: Decodable, Sendable {
        public var stop: String
        public var at: Int64
        public var pending_calls: [PendingCall]
    }
    public var id: String
    public var tokens: Int
    public var context: Int
    public var prefix_tokens: Int
    public var steps: Int
    public var state: String          // idle | queued | running | awaiting_tools | full
    public var warmth: Warmth
    public var model: String
    public var model_mismatch: Bool?
    public var metadata: [String: String]?
    public var last_step: LastStep?
    /// The session's own checkpoint on disk; absent from a server before M4.
    public var checkpoint_bytes: UInt64?
}

public struct OpenedSession: Decodable, Sendable {
    public var id: String
    public var prefix_tokens: Int
    public var context: Int
}

// MARK: - Events

public struct StepDone: Sendable {
    public var stop: String
    public var prompt = 0, generated = 0, reasoning = 0
    public var prefillSeconds = 0.0, decodeSeconds = 0.0, firstTokenSeconds = 0.0
    public var specRounds = 0, specCommitted = 0
    public var contextUsed = 0, contextLimit = 0
    public var warmth = "live"
}

public enum APIEvent: Sendable, Equatable {
    case queued(position: Int)
    case resume(from: String, restored: Int, prefill: Int, prefixCached: Bool?)
    case prefill(done: Int, total: Int)
    case context(used: Int, limit: Int)
    case reasoning(String, tokens: Int)
    case text(String)
    case decode(generated: Int, tokensPerSecond: Double, instantaneous: Double)
    case callProgress(name: String?, keys: [String], tokens: Int)
    case toolCall(id: String, name: String, arguments: [String: String])
    case done(stop: String, prompt: Int, generated: Int, reasoning: Int,
              prefillSeconds: Double, decodeSeconds: Double, firstTokenSeconds: Double,
              specRounds: Int, specCommitted: Int, contextUsed: Int, contextLimit: Int, warmth: String)
    case error(String)
    /// An event this client does not know; ignored by callers (API.md §3).
    case other(String)

    /// Decodes one event from its name and JSON data.
    public static func decode(name: String, data: String) -> APIEvent {
        guard let d = data.data(using: .utf8),
              let any = try? JSONSerialization.jsonObject(with: d),
              let o = any as? [String: Any] else { return .other(name) }
        func int(_ k: String) -> Int { (o[k] as? NSNumber)?.intValue ?? 0 }
        func dbl(_ k: String) -> Double { (o[k] as? NSNumber)?.doubleValue ?? 0 }
        func str(_ k: String) -> String { o[k] as? String ?? "" }
        switch name {
        case "queued": return .queued(position: int("position"))
        case "resume":
            return .resume(from: str("from"), restored: int("restored"), prefill: int("prefill"),
                           prefixCached: o["prefix_cached"] as? Bool)
        case "prefill": return .prefill(done: int("done"), total: int("total"))
        case "context": return .context(used: int("used"), limit: int("limit"))
        case "reasoning": return .reasoning(str("text"), tokens: int("tokens"))
        case "text": return .text(str("text"))
        case "decode":
            return .decode(generated: int("generated"), tokensPerSecond: dbl("tokens_per_second"),
                           instantaneous: dbl("instantaneous"))
        case "call_progress":
            return .callProgress(name: o["name"] as? String,
                                 keys: (o["keys"] as? [String]) ?? [], tokens: int("tokens"))
        case "tool_call":
            var args: [String: String] = [:]
            for (k, v) in (o["arguments"] as? [String: Any]) ?? [:] {
                if let s = v as? String { args[k] = s }
                else if let d = try? JSONSerialization.data(withJSONObject: v, options: [.fragmentsAllowed]) {
                    args[k] = String(decoding: d, as: UTF8.self)
                }
            }
            return .toolCall(id: str("id"), name: str("name"), arguments: args)
        case "done":
            let usage = o["usage"] as? [String: Any] ?? [:]
            let timing = o["timing"] as? [String: Any] ?? [:]
            let spec = o["speculation"] as? [String: Any] ?? [:]
            let ctx = o["context"] as? [String: Any] ?? [:]
            let warmth = o["warmth"] as? [String: Any] ?? [:]
            func n(_ m: [String: Any], _ k: String) -> Int { (m[k] as? NSNumber)?.intValue ?? 0 }
            func f(_ m: [String: Any], _ k: String) -> Double { (m[k] as? NSNumber)?.doubleValue ?? 0 }
            return .done(stop: str("stop"), prompt: n(usage, "prompt"), generated: n(usage, "generated"),
                         reasoning: n(usage, "reasoning"),
                         prefillSeconds: f(timing, "prefill_seconds"), decodeSeconds: f(timing, "decode_seconds"),
                         firstTokenSeconds: f(timing, "first_token_seconds"),
                         specRounds: n(spec, "rounds"), specCommitted: n(spec, "committed"),
                         contextUsed: n(ctx, "used"), contextLimit: n(ctx, "limit"),
                         warmth: warmth["state"] as? String ?? "live")
        case "error": return .error(str("message"))
        default: return .other(name)
        }
    }
}

/// An event with its id (`step.seq`), which is what reattaching needs.
public struct StreamEvent: Sendable {
    public var id: String
    public var event: APIEvent
}

/// Server-sent events, line by line.  Feed each line without its terminator;
/// an empty line completes an event.
public struct SSEParser: Sendable {
    public var id = "", event = "", data = ""
    private var hasData = false

    public init() {}

    /// Returns the completed event on a blank line, else nil.
    public mutating func feed(line: String) -> (id: String, event: String, data: String)? {
        if line.isEmpty {
            guard hasData else { return nil }
            let out = (id, event, data)
            id = ""; event = ""; data = ""; hasData = false
            return out
        }
        guard let colon = line.firstIndex(of: ":") else { return nil }
        let key = line[..<colon]
        var value = line[line.index(after: colon)...]
        if value.hasPrefix(" ") { value = value.dropFirst() }
        switch key {
        case "id": id = String(value)
        case "event": event = String(value)
        case "data":
            if hasData { data += "\n" }
            data += value
            hasData = true
        default: break
        }
        return nil
    }
}

// MARK: - The client

public struct Attachment: Sendable {
    public var kind: String        // "image" | "video"
    public var mediaType: String
    public var data: Data
    public init(kind: String, mediaType: String, data: Data) {
        self.kind = kind; self.mediaType = mediaType; self.data = data
    }
}

public struct ToolResultPayload: Sendable {
    public var id: String
    public var content: String
    public init(id: String, content: String) { self.id = id; self.content = content }
}

public final class QwasarClient: Sendable {
    public let base: URL
    public let token: String?
    private let session: URLSession

    public init(base: URL, token: String? = nil) {
        self.base = base
        self.token = token
        let cfg = URLSessionConfiguration.ephemeral
        // A step can run for as long as a turn takes; the request timeout is
        // between bytes, and prefill of a long tool result is quiet for a while.
        cfg.timeoutIntervalForRequest = 600
        cfg.timeoutIntervalForResource = 24 * 3600
        session = URLSession(configuration: cfg)
    }

    private func request(_ method: String, _ path: String, body: Any? = nil,
                         headers: [String: String] = [:]) throws -> URLRequest {
        var r = URLRequest(url: base.appendingPathComponent(path))
        r.httpMethod = method
        if let token { r.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization") }
        for (k, v) in headers { r.setValue(v, forHTTPHeaderField: k) }
        if let body {
            r.httpBody = try JSONSerialization.data(withJSONObject: body)
            r.setValue("application/json", forHTTPHeaderField: "Content-Type")
        }
        return r
    }

    /// One request with a JSON answer.  A non-2xx answer is a ClientError.refused.
    private func call<T: Decodable>(_ method: String, _ path: String, body: Any? = nil,
                                    as type: T.Type) async throws -> T {
        let (data, resp) = try await data(method, path, body: body)
        _ = resp
        do { return try JSONDecoder().decode(T.self, from: data) }
        catch { throw ClientError.malformed("\(path): \(error)") }
    }

    private func data(_ method: String, _ path: String, body: Any? = nil) async throws -> (Data, HTTPURLResponse) {
        let req = try request(method, path, body: body)
        let (data, resp): (Data, URLResponse)
        do { (data, resp) = try await session.data(for: req) }
        catch { throw ClientError.unreachable(error.localizedDescription) }
        guard let http = resp as? HTTPURLResponse else { throw ClientError.malformed("not HTTP") }
        if http.statusCode >= 300 { throw Self.refusal(http.statusCode, data) }
        return (data, http)
    }

    private static func refusal(_ status: Int, _ data: Data) -> ClientError {
        if let o = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let e = o["error"] as? [String: Any] {
            return .refused(status: status, code: e["code"] as? String ?? "",
                            message: e["message"] as? String ?? "")
        }
        return .refused(status: status, code: "", message: String(decoding: data, as: UTF8.self))
    }

    public func health() async -> Bool {
        (try? await data("GET", "health")) != nil
    }

    public func serverInfo() async throws -> ServerInfo {
        try await call("GET", "v1/server", as: ServerInfo.self)
    }

    public func open(system: String, tools: [String], thinking: Bool, effort: String,
                     metadata: [String: String]) async throws -> OpenedSession {
        // Tools are JSON already; they go in as objects, not strings.
        let toolObjects: [Any] = try tools.map { s in
            guard let d = s.data(using: .utf8), let o = try? JSONSerialization.jsonObject(with: d) else {
                throw ClientError.malformed("a tool schema is not JSON")
            }
            return o
        }
        let body: [String: Any] = ["system": system, "tools": toolObjects, "thinking": thinking,
                                   "effort": effort, "metadata": metadata]
        return try await call("POST", "v1/sessions", body: body, as: OpenedSession.self)
    }

    public func describe(_ id: String) async throws -> SessionInfo {
        try await call("GET", "v1/sessions/\(id)", as: SessionInfo.self)
    }

    public func list() async throws -> [SessionInfo] {
        struct L: Decodable { var sessions: [SessionInfo] }
        return try await call("GET", "v1/sessions", as: L.self).sessions
    }

    public func cancel(_ id: String) async throws {
        _ = try await data("POST", "v1/sessions/\(id)/cancel", body: [String: Any]())
    }

    public func park(_ id: String) async throws -> Warmth {
        struct P: Decodable { var warmth: Warmth }
        return try await call("POST", "v1/sessions/\(id)/park", body: [String: Any](), as: P.self).warmth
    }

    /// Gives the session's disk back: it becomes cold, its conversation kept.
    public func dropCheckpoint(_ id: String) async throws -> UInt64 {
        struct D: Decodable { var freed_bytes: UInt64 }
        return try await call("DELETE", "v1/sessions/\(id)/checkpoint", as: D.self).freed_bytes
    }

    public func delete(_ id: String) async throws {
        _ = try await data("DELETE", "v1/sessions/\(id)")
    }

    /// A user turn.  Events arrive as the server produces them; the stream
    /// ends after `done` or `error`.  A refusal before the stream starts
    /// throws ClientError.refused.
    public func turn(_ id: String, text: String, attachments: [Attachment] = [],
                     temperature: Float? = nil, maxTokens: Int = 0) -> AsyncThrowingStream<StreamEvent, Error> {
        var body: [String: Any] = ["text": text]
        if maxTokens > 0 { body["max_tokens"] = maxTokens }
        if let temperature { body["sampling"] = ["temperature": temperature] }
        if !attachments.isEmpty {
            body["images"] = attachments.map { ["kind": $0.kind, "media_type": $0.mediaType,
                                                 "data": $0.data.base64EncodedString()] }
        }
        return stream("v1/sessions/\(id)/turn", body: body)
    }

    public func continueStep(_ id: String, results: [ToolResultPayload], temperature: Float? = nil,
                             maxTokens: Int = 0) -> AsyncThrowingStream<StreamEvent, Error> {
        var body: [String: Any] = ["results": results.map { ["id": $0.id, "content": $0.content] }]
        if maxTokens > 0 { body["max_tokens"] = maxTokens }
        if let temperature { body["sampling"] = ["temperature": temperature] }
        return stream("v1/sessions/\(id)/continue", body: body)
    }

    /// Reattaches to the step in flight after `lastEventID` (API.md §4.6).
    public func events(_ id: String, after lastEventID: String?) -> AsyncThrowingStream<StreamEvent, Error> {
        stream("v1/sessions/\(id)/events", body: nil, method: "GET",
               headers: lastEventID.map { ["Last-Event-ID": $0] } ?? [:])
    }

    private func stream(_ path: String, body: Any?, method: String = "POST",
                        headers: [String: String] = [:]) -> AsyncThrowingStream<StreamEvent, Error> {
        // The request is built here, before the task: URLRequest is Sendable
        // and the JSON body is not.
        let req: URLRequest
        do { req = try request(method, path, body: body, headers: headers) }
        catch { return AsyncThrowingStream { $0.finish(throwing: error) } }
        let session = self.session
        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let (bytes, resp) = try await session.bytes(for: req)
                    guard let http = resp as? HTTPURLResponse else { throw ClientError.malformed("not HTTP") }
                    if !(http.value(forHTTPHeaderField: "Content-Type") ?? "").hasPrefix("text/event-stream") {
                        var buf = Data()
                        for try await b in bytes { buf.append(b) }
                        throw QwasarClient.refusal(http.statusCode, buf)
                    }
                    // Lines split here, byte by byte, rather than with
                    // AsyncBytes.lines: an event ends at an EMPTY line, and
                    // that sequence does not deliver empty lines, so no event
                    // ever completed and a whole turn streamed into silence.
                    var parser = SSEParser()
                    var line: [UInt8] = []
                    var finished = false
                    for try await b in bytes {
                        if b != 0x0A { line.append(b); continue }
                        if line.last == 0x0D { line.removeLast() }
                        let text = String(decoding: line, as: UTF8.self)
                        line.removeAll(keepingCapacity: true)
                        if let (id, name, data) = parser.feed(line: text) {
                            let ev = APIEvent.decode(name: name, data: data)
                            continuation.yield(StreamEvent(id: id, event: ev))
                            if case .done = ev { finished = true }
                            if case .error = ev { finished = true }
                            if finished { break }
                        }
                    }
                    continuation.finish()
                } catch is CancellationError {
                    continuation.finish()
                } catch let e as ClientError {
                    continuation.finish(throwing: e)
                } catch {
                    continuation.finish(throwing: ClientError.unreachable(error.localizedDescription))
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}
