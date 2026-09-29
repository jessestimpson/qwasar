// NetworkTool.swift -- `fetch`, executed by the HOST under host-side policy.
//
// PLAN.md 8.3. The guest keeps zero network devices; this never talks to it.
// The policy lives here, in code the agent cannot reach: everything in the
// guest -- including the warden, against root -- is inside the blast radius,
// so a guardrail in the guest would be advisory. A guardrail here is not.
//
// The rules, and every one is enforced in this file:
//   GET only, https only, port 443 only, no userinfo, no IP literals unless
//   explicitly listed, allowlisted hosts only, redirects re-checked hop by
//   hop, response bytes capped DURING download, and a timeout.

import Foundation

/// A project's network policy. An empty allowlist is "network off", which is
/// the default and means `fetch` is not even advertised.
public struct NetworkPolicy: Sendable, Equatable {
    /// Exact hostnames, lowercased. `*.example.com` admits any subdomain
    /// (but not example.com itself -- list both to have both).
    public var allowlist: [String]
    public var maxResponseBytes: Int
    public var timeoutSeconds: Double

    public init(allowlist: [String], maxResponseBytes: Int = 256 * 1024,
                timeoutSeconds: Double = 30) {
        self.allowlist = allowlist.map { $0.lowercased() }
        self.maxResponseBytes = maxResponseBytes
        self.timeoutSeconds = timeoutSeconds
    }

    public var isEnabled: Bool { !allowlist.isEmpty }

    /// Why `url` may not be fetched, or nil if it may. A reason rather than a
    /// Bool, because the reason is the tool result: the model can only choose
    /// something else if it is told what was wrong.
    public func refusal(for url: URL) -> String? {
        guard let comps = URLComponents(url: url, resolvingAgainstBaseURL: true) else {
            return "not a parseable URL"
        }
        guard comps.scheme?.lowercased() == "https" else {
            return "only https is allowed"
        }
        if comps.user != nil || comps.password != nil {
            return "userinfo in URLs is not allowed"
        }
        if let port = comps.port, port != 443 {
            return "only port 443 is allowed"
        }
        guard let host = comps.host?.lowercased(), !host.isEmpty else {
            return "the URL has no host"
        }
        if hostAllowed(host) { return nil }
        // An IP literal that is not explicitly listed gets the more useful
        // message: names are what the allowlist is for.
        if host.allSatisfy({ $0.isNumber || $0 == "." }) || host.contains(":") {
            return "IP-literal hosts are not allowed unless explicitly listed"
        }
        return "\(host) is not in this project's allowlist"
    }

    func hostAllowed(_ host: String) -> Bool {
        for entry in allowlist {
            if entry.hasPrefix("*.") {
                // The suffix keeps its leading dot, so "evilexample.com"
                // cannot satisfy "*.example.com" -- and the bare domain does
                // not either; list it separately to have it.
                if host.hasSuffix(String(entry.dropFirst(1))) { return true }
            } else if host == entry {
                return true
            }
        }
        return false
    }
}

/// Wraps any executor with the host-side `fetch` tool. Everything else is
/// delegated untouched, so the sandbox path stays exactly what it was.
public struct NetworkToolRunner: ToolExecuting {
    let inner: ToolExecuting
    let policy: NetworkPolicy

    public init(inner: ToolExecuting, policy: NetworkPolicy) {
        self.inner = inner
        self.policy = policy
    }

    public var schemas: [String] { inner.schemas + [ToolSurface.fetchSchema] }

    public var environmentDescription: String {
        inner.environmentDescription + """


        Network: `fetch` performs an HTTPS GET, executed by the host under a per-project allowlist -- allowed hosts: \(policy.allowlist.joined(separator: ", ")). A refused fetch names the host; the user can add it to the allowlist. Responses are text, capped at \(policy.maxResponseBytes / 1024) KB.
        """
    }

    public func run(_ call: ToolCall) -> String {
        guard call.name == "fetch" else { return inner.run(call) }
        guard let raw = call.arguments["url"], !raw.isEmpty else {
            return "error: fetch requires a url"
        }
        guard let url = URL(string: raw) else { return "error: not a parseable URL" }
        if let why = policy.refusal(for: url) { return "error: fetch refused: \(why)" }

        // Sync over async, the same shape SandboxToolRunner uses and for the
        // same reason: the engine queue has nothing else to do while a tool
        // runs, and the wait below always ends -- the delegate's byte cap or
        // the timeout cancels anything the server will not.
        let box = FetchBox()
        let done = DispatchSemaphore(value: 0)
        let policy = self.policy
        Task.detached {
            box.set(await Self.fetch(url, policy: policy))
            done.signal()
        }
        done.wait()
        return box.get()
    }

    static func fetch(_ url: URL, policy: NetworkPolicy) async -> String {
        let guardDelegate = FetchGuard(policy: policy)
        let cfg = URLSessionConfiguration.ephemeral
        cfg.timeoutIntervalForRequest = policy.timeoutSeconds
        cfg.timeoutIntervalForResource = policy.timeoutSeconds
        cfg.httpCookieAcceptPolicy = .never
        let session = URLSession(configuration: cfg, delegate: guardDelegate, delegateQueue: nil)
        defer { session.finishTasksAndInvalidate() }

        var req = URLRequest(url: url)
        req.httpMethod = "GET"
        do {
            let (data, response) = try await session.data(for: req)
            guard let http = response as? HTTPURLResponse else {
                return "error: not an HTTP response"
            }
            let type = (http.value(forHTTPHeaderField: "Content-Type") ?? "").lowercased()
            let textual = type.hasPrefix("text/") || type.contains("json")
                || type.contains("xml") || type.contains("javascript")
                || type.isEmpty
            let capped = data.prefix(policy.maxResponseBytes)
            guard textual, let body = String(data: capped, encoding: .utf8) else {
                return "HTTP \(http.statusCode) · \(data.count) bytes of \(type.isEmpty ? "unknown type" : type) — binary content is not delivered; fetch is for text"
            }
            let truncated = data.count > policy.maxResponseBytes
                ? "\n\n[truncated at \(policy.maxResponseBytes / 1024) KB of \(data.count) bytes]" : ""
            return "HTTP \(http.statusCode) · \(type.isEmpty ? "?" : type) · \(data.count) bytes\n\n\(body)\(truncated)"
        } catch let e as FetchGuard.Refused {
            return "error: fetch refused: \(e.reason)"
        } catch {
            let ns = error as NSError
            if let refused = ns.userInfo[NSUnderlyingErrorKey] as? FetchGuard.Refused {
                return "error: fetch refused: \(refused.reason)"
            }
            return "error: \(ns.localizedDescription)"
        }
    }
}

/// The policy's runtime half: redirects re-checked hop by hop, and the byte
/// cap enforced while the body arrives rather than after it has.
final class FetchGuard: NSObject, URLSessionTaskDelegate, URLSessionDataDelegate, @unchecked Sendable {
    struct Refused: Error { let reason: String }
    private let policy: NetworkPolicy
    private var received = 0

    init(policy: NetworkPolicy) { self.policy = policy }

    func urlSession(_ session: URLSession, task: URLSessionTask,
                    willPerformHTTPRedirection response: HTTPURLResponse,
                    newRequest request: URLRequest,
                    completionHandler: @escaping (URLRequest?) -> Void) {
        // A redirect is how an allowed host becomes a proxy for an arbitrary
        // one, so every hop faces the same policy the first URL did.
        if let u = request.url, policy.refusal(for: u) == nil {
            completionHandler(request)
        } else {
            task.cancel()
            completionHandler(nil)
        }
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask,
                    didReceive data: Data) {
        received += data.count
        // A generous multiple of the cap: the result is truncated anyway;
        // this exists so a bottomless response cannot fill memory.
        if received > policy.maxResponseBytes * 4 { dataTask.cancel() }
    }
}

private final class FetchBox: @unchecked Sendable {
    private var value = ""
    private let lock = NSLock()
    func set(_ s: String) { lock.withLock { value = s } }
    func get() -> String { lock.withLock { value } }
}
