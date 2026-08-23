// SandboxSettings.swift -- sandbox configuration, in three layers.
//
// PLAN.md 8.5. One overlay shape at every layer -- global, project, session --
// with every field optional, and one resolution rule: **field-wise, most
// specific non-nil wins** (session, then project, then global, then the
// built-in default). Nothing merges: a session that sets `networkAllowlist`
// REPLACES the project's list rather than adding to it, because "replace" is
// a rule a person can predict from the value they typed, and "merge" is a
// rule they have to go and check.

import Foundation

/// One layer of sandbox configuration. All fields optional; nil means "this
/// layer has no opinion" and resolution falls through to the next.
public struct SandboxOverlay: Codable, Sendable, Equatable, Hashable {
    /// Hosts `fetch` may reach (PLAN.md 8.3). Empty array is an OPINION --
    /// network explicitly off at this layer -- where nil is silence.
    public var networkAllowlist: [String]?
    public var guestMemoryMB: Int?
    public var guestCPUs: Int?
    /// Per-tool-call ceiling, for a guest that has stopped answering.
    public var toolTimeoutSeconds: Int?
    /// `fetch` response cap.
    public var fetchMaxKB: Int?

    public init(networkAllowlist: [String]? = nil, guestMemoryMB: Int? = nil,
                guestCPUs: Int? = nil, toolTimeoutSeconds: Int? = nil,
                fetchMaxKB: Int? = nil) {
        self.networkAllowlist = networkAllowlist
        self.guestMemoryMB = guestMemoryMB
        self.guestCPUs = guestCPUs
        self.toolTimeoutSeconds = toolTimeoutSeconds
        self.fetchMaxKB = fetchMaxKB
    }

    public var isEmpty: Bool {
        networkAllowlist == nil && guestMemoryMB == nil && guestCPUs == nil
            && toolTimeoutSeconds == nil && fetchMaxKB == nil
    }
}

/// The keys a config session can address, with their parsing -- one place, so
/// the tool, the renderer and the tests agree on what exists.
public enum SandboxKey: String, CaseIterable, Sendable {
    case networkAllowlist = "network_allowlist"
    case guestMemoryMB = "guest_memory_mb"
    case guestCPUs = "guest_cpus"
    case toolTimeoutSeconds = "tool_timeout_seconds"
    case fetchMaxKB = "fetch_max_kb"

    public var doc: String {
        switch self {
        case .networkAllowlist:
            return "hosts `fetch` may reach; comma-separated, `*.host` for subdomains, empty string for explicitly OFF"
        case .guestMemoryMB: return "guest VM memory, MB (default 2048)"
        case .guestCPUs: return "guest VM CPUs (default 2)"
        case .toolTimeoutSeconds: return "per-tool-call ceiling, seconds (default 180)"
        case .fetchMaxKB: return "fetch response cap, KB (default 256)"
        }
    }

    /// Applies a textual value to `overlay`, or says why it cannot.
    public func set(_ value: String, on overlay: inout SandboxOverlay) -> String? {
        func int(_ range: ClosedRange<Int>) -> Int? {
            guard let n = Int(value.trimmingCharacters(in: .whitespaces)),
                  range.contains(n) else { return nil }
            return n
        }
        switch self {
        case .networkAllowlist:
            let hosts = value.split(separator: ",")
                .map { $0.trimmingCharacters(in: .whitespaces).lowercased() }
                .filter { !$0.isEmpty }
            overlay.networkAllowlist = hosts     // [] = explicitly off
        case .guestMemoryMB:
            guard let n = int(512...65536) else { return "guest_memory_mb needs an integer in 512...65536" }
            overlay.guestMemoryMB = n
        case .guestCPUs:
            guard let n = int(1...16) else { return "guest_cpus needs an integer in 1...16" }
            overlay.guestCPUs = n
        case .toolTimeoutSeconds:
            guard let n = int(10...3600) else { return "tool_timeout_seconds needs an integer in 10...3600" }
            overlay.toolTimeoutSeconds = n
        case .fetchMaxKB:
            guard let n = int(1...10240) else { return "fetch_max_kb needs an integer in 1...10240" }
            overlay.fetchMaxKB = n
        }
        return nil
    }

    public func clear(on overlay: inout SandboxOverlay) {
        switch self {
        case .networkAllowlist: overlay.networkAllowlist = nil
        case .guestMemoryMB: overlay.guestMemoryMB = nil
        case .guestCPUs: overlay.guestCPUs = nil
        case .toolTimeoutSeconds: overlay.toolTimeoutSeconds = nil
        case .fetchMaxKB: overlay.fetchMaxKB = nil
        }
    }

    public func value(in overlay: SandboxOverlay) -> String? {
        switch self {
        case .networkAllowlist: return overlay.networkAllowlist.map { $0.isEmpty ? "(explicitly off)" : $0.joined(separator: ", ") }
        case .guestMemoryMB: return overlay.guestMemoryMB.map(String.init)
        case .guestCPUs: return overlay.guestCPUs.map(String.init)
        case .toolTimeoutSeconds: return overlay.toolTimeoutSeconds.map(String.init)
        case .fetchMaxKB: return overlay.fetchMaxKB.map(String.init)
        }
    }
}

/// The resolved settings a session actually runs with, and where each field
/// came from -- provenance is what makes the overlay model inspectable rather
/// than something the user reverse-engineers from behaviour.
public struct SandboxSettings: Sendable, Equatable {
    public var networkAllowlist: [String]
    public var guestMemoryMB: Int
    public var guestCPUs: Int
    public var toolTimeoutSeconds: Int
    public var fetchMaxKB: Int

    public static let defaults = SandboxSettings(networkAllowlist: [],
                                                 guestMemoryMB: 2048,
                                                 guestCPUs: 2,
                                                 toolTimeoutSeconds: 180,
                                                 fetchMaxKB: 256)

    public enum Layer: String, Sendable { case session, project, global, builtin = "built-in" }

    /// Field-wise: session, then project, then global, then the default.
    public static func resolve(global: SandboxOverlay?, project: SandboxOverlay?,
                               session: SandboxOverlay?) -> SandboxSettings {
        func pick<T>(_ path: (SandboxOverlay) -> T?, _ fallback: T) -> T {
            session.flatMap(path) ?? project.flatMap(path) ?? global.flatMap(path) ?? fallback
        }
        return SandboxSettings(
            networkAllowlist: pick({ $0.networkAllowlist }, defaults.networkAllowlist),
            guestMemoryMB: pick({ $0.guestMemoryMB }, defaults.guestMemoryMB),
            guestCPUs: pick({ $0.guestCPUs }, defaults.guestCPUs),
            toolTimeoutSeconds: pick({ $0.toolTimeoutSeconds }, defaults.toolTimeoutSeconds),
            fetchMaxKB: pick({ $0.fetchMaxKB }, defaults.fetchMaxKB))
    }

    /// Which layer decided `key`, for display.
    public static func provenance(of key: SandboxKey, global: SandboxOverlay?,
                                  project: SandboxOverlay?,
                                  session: SandboxOverlay?) -> Layer {
        if let s = session, key.value(in: s) != nil { return .session }
        if let p = project, key.value(in: p) != nil { return .project }
        if let g = global, key.value(in: g) != nil { return .global }
        return .builtin
    }
}
