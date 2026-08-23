// ConfigTools.swift -- the Crucible Config project's tool surface.
//
// PLAN.md 8.5. Sessions in the config project manage Crucible itself, so
// their tools run on the HOST with no sandbox -- but "unsandboxed" is not
// "unbounded": the surface is three purpose-built config operations, not a
// shell. The model can read and write sandbox configuration at its three
// layers and nothing else; there is no path from here to the filesystem, the
// network, or another project's files.
//
// Mutations go through AppState on the main actor -- the same single write
// path the UI uses -- so the sidebar, the store and a running config session
// can never disagree about what the configuration is.

import Foundation
import CrucibleKit

/// What a config tool call needs answered. Executed on the main actor by
/// AppState; the runner only parses and renders.
enum ConfigOp {
    case show
    case set(scope: String, target: String?, key: String, value: String)
    case clear(scope: String, target: String?, key: String?)
}

struct ConfigToolRunner: ToolExecuting {
    /// Bridges to the main actor. The engine queue blocks here for the
    /// duration of a config edit, which is microseconds of dictionary work --
    /// and main never blocks on the engine queue, so the sync cannot deadlock.
    let perform: @Sendable (ConfigOp) -> String

    static let showSchema = #"""
    {"type": "function", "function": {"name": "config_show", "description": "Show Crucible's sandbox configuration: the global layer, every project's layer, every session overlay, and the resolved effective settings with the layer each value came from. Call this before changing anything.", "parameters": {"type": "object", "properties": {}, "required": []}}}
    """#

    static let setSchema = #"""
    {"type": "function", "function": {"name": "config_set", "description": "Set one sandbox configuration key at one layer. Resolution is field-wise: session overrides project overrides global overrides the built-in default -- setting a value REPLACES what lower layers said for that key, never merges. Changes apply when a session is next opened (a live session keeps the settings it booted with). Keys: network_allowlist (comma-separated hosts, `*.host` for subdomains, empty string for explicitly OFF), guest_memory_mb, guest_cpus, tool_timeout_seconds, fetch_max_kb.", "parameters": {"type": "object", "properties": {"scope": {"type": "string", "description": "global, project, or session."}, "target": {"type": "string", "description": "Project name or session title/id; required for project and session scope."}, "key": {"type": "string", "description": "One of the keys above."}, "value": {"type": "string", "description": "The value, as text."}}, "required": ["scope", "key", "value"]}}}
    """#

    static let clearSchema = #"""
    {"type": "function", "function": {"name": "config_clear", "description": "Clear one key at one layer (so resolution falls through to the next layer), or clear a whole layer by omitting the key.", "parameters": {"type": "object", "properties": {"scope": {"type": "string", "description": "global, project, or session."}, "target": {"type": "string", "description": "Project name or session title/id; required for project and session scope."}, "key": {"type": "string", "description": "Omit to clear the whole layer."}}, "required": ["scope"]}}}
    """#

    var schemas: [String] { [Self.showSchema, Self.setSchema, Self.clearSchema] }

    var environmentDescription: String {
        """
        You are the Crucible Config session. Your tools run on the host, with no sandbox, and manage Crucible's own configuration -- nothing else. There is no file access and no shell here.

        Sandbox configuration has three layers: global, per-project, per-session. Resolution is field-wise and the most specific non-nil value wins: session > project > global > built-in default. Setting a key at a layer REPLACES lower layers' value for that key; clearing it lets resolution fall through. Changes take effect when a session is next opened.

        Keys: \(SandboxKey.allCases.map { "\($0.rawValue) — \($0.doc)" }.joined(separator: "; ")).

        Start with config_show. Change only what the user asked for, and say what changed at which layer.
        """
    }

    func run(_ call: ToolCall) -> String {
        let a = call.arguments
        switch call.name {
        case "config_show":
            return perform(.show)
        case "config_set":
            guard let scope = a["scope"], let key = a["key"], let value = a["value"] else {
                return "error: config_set needs scope, key and value"
            }
            return perform(.set(scope: scope, target: a["target"], key: key, value: value))
        case "config_clear":
            guard let scope = a["scope"] else { return "error: config_clear needs a scope" }
            return perform(.clear(scope: scope, target: a["target"], key: a["key"]))
        default:
            return "error: no such tool: \(call.name). Available: config_show, config_set, config_clear"
        }
    }
}

// MARK: - The main-actor half

extension AppState {
    /// The runner for config-project sessions. A fresh value per open, but
    /// the closure always reads live state, so it cannot go stale.
    func configToolRunner() -> ConfigToolRunner {
        ConfigToolRunner { op in
            DispatchQueue.main.sync {
                MainActor.assumeIsolated { self.performConfig(op) }
            }
        }
    }

    func performConfig(_ op: ConfigOp) -> String {
        switch op {
        case .show:
            return renderConfig()
        case .set(let scope, let target, let key, let value):
            guard let k = SandboxKey(rawValue: key) else {
                return "error: unknown key \(key). Keys: "
                     + SandboxKey.allCases.map(\.rawValue).joined(separator: ", ")
            }
            return mutateOverlay(scope: scope, target: target) { overlay in
                k.set(value, on: &overlay)
            }
        case .clear(let scope, let target, let key):
            if let key {
                guard let k = SandboxKey(rawValue: key) else { return "error: unknown key \(key)" }
                return mutateOverlay(scope: scope, target: target) { overlay in
                    k.clear(on: &overlay); return nil
                }
            }
            return mutateOverlay(scope: scope, target: target) { overlay in
                overlay = SandboxOverlay(); return nil
            }
        }
    }

    /// One mutation path for all three layers. The edit closure returns an
    /// error string or nil; the layer is persisted only on success.
    private func mutateOverlay(scope: String, target: String?,
                               _ edit: (inout SandboxOverlay) -> String?) -> String {
        switch scope.lowercased() {
        case "global":
            var o = globalSandbox ?? SandboxOverlay()
            if let err = edit(&o) { return "error: \(err)" }
            globalSandbox = o.isEmpty ? nil : o
            store?.saveGlobalOverlay(globalSandbox)
            return "ok: global layer is now \(describe(globalSandbox))"

        case "project":
            guard let target else { return "error: project scope needs a target" }
            guard let i = projects.firstIndex(where: {
                $0.name.lowercased() == target.lowercased() || $0.id.uuidString == target.uppercased()
            }) else {
                return "error: no project named \(target). Projects: "
                     + projects.map(\.name).joined(separator: ", ")
            }
            if projects[i].isConfig { return "error: the config project has no sandbox to configure" }
            var o = projects[i].overlay
            if let err = edit(&o) { return "error: \(err)" }
            projects[i].sandbox = o.isEmpty ? nil : o
            projects[i].networkAllowlist = nil    // folded into the overlay now
            store?.saveProjects(projects)
            return "ok: project \(projects[i].name) layer is now \(describe(projects[i].sandbox))"

        case "session":
            guard let target else { return "error: session scope needs a target" }
            let matches = sessions.enumerated().filter {
                $0.element.title.lowercased() == target.lowercased()
                    || $0.element.id.uuidString == target.uppercased()
            }
            guard matches.count == 1, let (i, _) = matches.first else {
                return matches.isEmpty
                    ? "error: no session titled \(target); config_show lists them"
                    : "error: \(matches.count) sessions share that title; use the id"
            }
            var o = sessions[i].sandbox ?? SandboxOverlay()
            if let err = edit(&o) { return "error: \(err)" }
            sessions[i].sandbox = o.isEmpty ? nil : o
            store?.save(sessions[i])
            return "ok: session \(sessions[i].title) layer is now \(describe(sessions[i].sandbox))"
                 + (sessions[i].id == liveSessionID
                    ? " (it is live; the change applies when it next opens)" : "")

        default:
            return "error: scope must be global, project, or session"
        }
    }

    private func describe(_ o: SandboxOverlay?) -> String {
        guard let o, !o.isEmpty else { return "empty (falls through)" }
        let parts = SandboxKey.allCases.compactMap { k in
            k.value(in: o).map { "\(k.rawValue)=\($0)" }
        }
        return parts.joined(separator: ", ")
    }

    private func renderConfig() -> String {
        var out = ["defaults: " + SandboxKey.allCases.map { k in
            "\(k.rawValue)=\(k.value(in: defaultsAsOverlay) ?? "?")"
        }.joined(separator: ", ")]
        out.append("global: \(describe(globalSandbox))")
        for p in projects where !p.isConfig {
            out.append("project \(p.name) [\(p.id.uuidString)]: \(describe(p.sandbox ?? (p.overlay.isEmpty ? nil : p.overlay)))")
            for s in sessions where s.projectID == p.id {
                let eff = SandboxSettings.resolve(global: globalSandbox,
                                                  project: p.overlay,
                                                  session: s.sandbox)
                let prov = SandboxKey.allCases.map { k in
                    "\(k.rawValue) ← \(SandboxSettings.provenance(of: k, global: globalSandbox, project: p.overlay, session: s.sandbox).rawValue)"
                }.joined(separator: ", ")
                out.append("  session \(s.title) [\(s.id.uuidString)]: layer \(describe(s.sandbox))")
                out.append("    effective: network=[\(eff.networkAllowlist.joined(separator: ", "))] "
                         + "memory=\(eff.guestMemoryMB)MB cpus=\(eff.guestCPUs) "
                         + "timeout=\(eff.toolTimeoutSeconds)s fetch_cap=\(eff.fetchMaxKB)KB")
                out.append("    provenance: \(prov)")
            }
        }
        return out.joined(separator: "\n")
    }

    private var defaultsAsOverlay: SandboxOverlay {
        let d = SandboxSettings.defaults
        return SandboxOverlay(networkAllowlist: d.networkAllowlist,
                              guestMemoryMB: d.guestMemoryMB, guestCPUs: d.guestCPUs,
                              toolTimeoutSeconds: d.toolTimeoutSeconds,
                              fetchMaxKB: d.fetchMaxKB)
    }
}
