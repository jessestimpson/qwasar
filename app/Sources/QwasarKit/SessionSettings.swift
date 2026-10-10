// SessionSettings.swift -- session configuration, in three layers.
//
// PLAN.md 8.5. One overlay shape at every layer -- global, project, session --
// with every field optional, and one resolution rule: **field-wise, most
// specific non-nil wins** (session, then project, then global, then the
// built-in default). Nothing merges: a session that sets `agentModels`
// REPLACES the project's list rather than adding to it, because "replace" is
// a rule a person can predict from the value they typed, and "merge" is a
// rule they have to go and check.
//
// These were the sandbox's settings, when sessions could run their tools in a
// VM; the guest's own keys (memory, CPUs, network, fetch) went with it.  Old
// overlay JSON still decodes: the keys it no longer has are ignored.

import Foundation

/// One layer of configuration. All fields optional; nil means "this layer
/// has no opinion" and resolution falls through to the next.
public struct SettingsOverlay: Codable, Sendable, Equatable, Hashable {
    /// Bash's default timeout, seconds.
    public var toolTimeoutSeconds: Int?
    /// Delegation (spec §15): remote model ids `delegate` may use. Empty is
    /// an OPINION -- delegation explicitly off -- where nil is silence.
    /// (Swift property names keep the older `agent` prefix so stored overlay
    /// JSON continues to decode; the user-facing key is `delegate_models`.)
    public var agentModels: [String]?
    /// Ceiling per session, summed across its delegations, USD.
    public var agentBudgetUSD: Double?
    /// Ceiling per single delegation, USD.
    public var agentTurnBudgetUSD: Double?

    public init(toolTimeoutSeconds: Int? = nil, agentModels: [String]? = nil,
                agentBudgetUSD: Double? = nil, agentTurnBudgetUSD: Double? = nil) {
        self.toolTimeoutSeconds = toolTimeoutSeconds
        self.agentModels = agentModels
        self.agentBudgetUSD = agentBudgetUSD
        self.agentTurnBudgetUSD = agentTurnBudgetUSD
    }

    public var isEmpty: Bool {
        toolTimeoutSeconds == nil && agentModels == nil && agentBudgetUSD == nil
            && agentTurnBudgetUSD == nil
    }
}

/// The keys a config session can address, with their parsing -- one place, so
/// the tool, the renderer and the tests agree on what exists.
public enum SettingsKey: String, CaseIterable, Sendable {
    case toolTimeoutSeconds = "tool_timeout_seconds"
    case agentModels = "delegate_models"
    case agentBudgetUSD = "delegate_budget_usd"
    case agentTurnBudgetUSD = "delegate_turn_budget_usd"

    public var doc: String {
        switch self {
        case .toolTimeoutSeconds: return "Bash's default timeout, seconds (default 180)"
        case .agentModels:
            return "remote model ids `delegate` may use (spec §15); comma-separated, empty string for explicitly OFF"
        case .agentBudgetUSD: return "delegation ceiling per session, USD (default 5.00)"
        case .agentTurnBudgetUSD: return "delegation ceiling per single delegation, USD (default 1.00)"
        }
    }

    /// Applies a textual value to `overlay`, or says why it cannot.
    public func set(_ value: String, on overlay: inout SettingsOverlay) -> String? {
        switch self {
        case .toolTimeoutSeconds:
            guard let n = Int(value.trimmingCharacters(in: .whitespaces)), (10...3600).contains(n) else {
                return "tool_timeout_seconds needs an integer in 10...3600"
            }
            overlay.toolTimeoutSeconds = n
        case .agentModels:
            let ids = value.split(separator: ",")
                .map { $0.trimmingCharacters(in: .whitespaces) }
                .filter { !$0.isEmpty }
            overlay.agentModels = ids                // [] = explicitly off
        case .agentBudgetUSD:
            guard let d = Double(value.trimmingCharacters(in: .whitespaces)),
                  (0...1000).contains(d) else { return "delegate_budget_usd needs a number in 0...1000" }
            overlay.agentBudgetUSD = d
        case .agentTurnBudgetUSD:
            guard let d = Double(value.trimmingCharacters(in: .whitespaces)),
                  (0...1000).contains(d) else { return "delegate_turn_budget_usd needs a number in 0...1000" }
            overlay.agentTurnBudgetUSD = d
        }
        return nil
    }

    public func clear(on overlay: inout SettingsOverlay) {
        switch self {
        case .toolTimeoutSeconds: overlay.toolTimeoutSeconds = nil
        case .agentModels: overlay.agentModels = nil
        case .agentBudgetUSD: overlay.agentBudgetUSD = nil
        case .agentTurnBudgetUSD: overlay.agentTurnBudgetUSD = nil
        }
    }

    public func value(in overlay: SettingsOverlay) -> String? {
        switch self {
        case .toolTimeoutSeconds: return overlay.toolTimeoutSeconds.map(String.init)
        case .agentModels: return overlay.agentModels.map { $0.isEmpty ? "(explicitly off)" : $0.joined(separator: ", ") }
        case .agentBudgetUSD: return overlay.agentBudgetUSD.map { String(format: "%.2f", $0) }
        case .agentTurnBudgetUSD: return overlay.agentTurnBudgetUSD.map { String(format: "%.2f", $0) }
        }
    }
}

/// The resolved settings a session actually runs with, and where each field
/// came from -- provenance is what makes the overlay model inspectable rather
/// than something the user reverse-engineers from behaviour.
public struct SessionSettings: Sendable, Equatable {
    public var toolTimeoutSeconds: Int
    public var agentModels: [String]
    public var agentBudgetUSD: Double
    public var agentTurnBudgetUSD: Double

    public static let defaults = SessionSettings(toolTimeoutSeconds: 180,
                                                 agentModels: [],
                                                 agentBudgetUSD: 5.0,
                                                 agentTurnBudgetUSD: 1.0)

    public enum Layer: String, Sendable { case session, project, global, builtin = "built-in" }

    /// Field-wise: session, then project, then global, then the default.
    public static func resolve(global: SettingsOverlay?, project: SettingsOverlay?,
                               session: SettingsOverlay?) -> SessionSettings {
        func pick<T>(_ path: (SettingsOverlay) -> T?, _ fallback: T) -> T {
            session.flatMap(path) ?? project.flatMap(path) ?? global.flatMap(path) ?? fallback
        }
        return SessionSettings(
            toolTimeoutSeconds: pick({ $0.toolTimeoutSeconds }, defaults.toolTimeoutSeconds),
            agentModels: pick({ $0.agentModels }, defaults.agentModels),
            agentBudgetUSD: pick({ $0.agentBudgetUSD }, defaults.agentBudgetUSD),
            agentTurnBudgetUSD: pick({ $0.agentTurnBudgetUSD }, defaults.agentTurnBudgetUSD))
    }

    /// Which layer decided `key`, for display.
    public static func provenance(of key: SettingsKey, global: SettingsOverlay?,
                                  project: SettingsOverlay?,
                                  session: SettingsOverlay?) -> Layer {
        if let s = session, key.value(in: s) != nil { return .session }
        if let p = project, key.value(in: p) != nil { return .project }
        if let g = global, key.value(in: g) != nil { return .global }
        return .builtin
    }
}
