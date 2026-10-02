// ConfigTools.swift -- the Qwasar Config project's tool surface.
//
// PLAN.md 8.5. Sessions in the config project manage Qwasar itself, so
// their tools run on the HOST with no sandbox -- but "unsandboxed" is not
// "unbounded": the surface is three purpose-built config operations, not a
// shell. The model can read and write the app's settings (the server's port,
// model, context, live sessions and running state; start at login; the
// checkpoint budget), each project's (default effort, its own prompt), and
// the sandbox configuration at its three layers -- and nothing else; there is
// no path from here to the filesystem, the network, or a project's files.
// The one secret, the delegation API key, stays the user's to enter.
//
// Mutations go through AppState on the main actor -- the same single write
// path the UI uses -- so the sidebar, the store and a running config session
// can never disagree about what the configuration is.

import Foundation
import QwasarKit
import ServiceManagement

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
    {"type": "function", "function": {"name": "config_show", "description": "Show all of Qwasar's configuration: the app's own settings (the server's port, model, context and live sessions, whether it runs, start at login, the checkpoint disk budget), each project's settings, and the sandbox layers -- global, every project, every session -- with the resolved effective values and the layer each came from. Call this before changing anything.", "parameters": {"type": "object", "properties": {}, "required": []}}}
    """#

    static let setSchema = #"""
    {"type": "function", "function": {"name": "config_set", "description": "Set one configuration key. scope \"app\" is the app itself: server_port (1-65535), server_model (\"flash-next\", \"27b\", or a model folder's path), server_context (tokens, or \"auto\"), server_live_sessions (a number, or \"auto\"), server_running (true or false), start_at_login (true or false), checkpoint_disk_budget_gb (a number), running_notes (true or false: whether the model writes its working notes, off the record, after each reply). A server_port, server_model, server_context, server_live_sessions or server_running change restarts or stops the server -- after your reply finishes, since you are running on it. scope \"project\" (target: its name) also takes default_effort (low, medium, xhigh, or \"auto\") and system_prompt (the project's own guidance for its sessions; empty to remove). Sandbox keys apply at scope global, project or session; resolution is field-wise, session over project over global over the built-in default, and setting a value REPLACES what lower layers said for that key. They apply when a session is next opened. Sandbox keys: network_allowlist (comma-separated hosts, `*.host` for subdomains, empty string for explicitly OFF), guest_memory_mb, guest_cpus, tool_timeout_seconds, fetch_max_kb, delegate_models (comma-separated remote model ids, empty string for explicitly OFF), delegate_budget_usd, delegate_turn_budget_usd.", "parameters": {"type": "object", "properties": {"scope": {"type": "string", "description": "app, global, project, or session."}, "target": {"type": "string", "description": "Project name or session title/id; required for project and session scope."}, "key": {"type": "string", "description": "One of the keys above."}, "value": {"type": "string", "description": "The value, as text."}}, "required": ["scope", "key", "value"]}}}
    """#

    static let clearSchema = #"""
    {"type": "function", "function": {"name": "config_clear", "description": "Clear one key so it falls back to its default -- for app keys the built-in default (server_port 8080; server_context, server_live_sessions and checkpoint_disk_budget_gb back to automatic; start_at_login off, running_notes on), for sandbox keys the next layer down -- or clear a whole sandbox layer by omitting the key.", "parameters": {"type": "object", "properties": {"scope": {"type": "string", "description": "app, global, project, or session."}, "target": {"type": "string", "description": "Project name or session title/id; required for project and session scope."}, "key": {"type": "string", "description": "The key to clear; omit to clear a whole sandbox layer."}}, "required": ["scope"]}}}
    """#

    var schemas: [String] { [Self.showSchema, Self.setSchema, Self.clearSchema] }

    var environmentDescription: String {
        """
        You are the Qwasar Config session. Your tools run on the host, with no sandbox, and manage Qwasar's own configuration -- nothing else. There is no file access and no shell here.

        Three kinds of setting:
        - The app's own (scope app): the local server -- its port, which model it runs, its context size and how many sessions it keeps in memory (both derived from the model and the machine unless set), whether it runs at all -- plus start at login and the disk budget for parked sessions' checkpoints. Changing the server's port, model, context, live sessions or running state restarts or stops it once your reply is done, because you are running on it: tell the user that, and do not expect to see the result in this turn.
        - A project's (scope project): its default reasoning effort for new sessions, and its own guidance prompt.
        - Sandbox settings, in three layers -- global, per project, per session -- resolved field-wise, the most specific non-nil value winning: session > project > global > built-in default. Setting a key at a layer REPLACES lower layers' value for that key; clearing it lets resolution fall through. They take effect when a session is next opened, and apply to sandboxed sessions (network and fetch), or to all sessions (tool timeout, delegation).

        Sandbox keys: \(SandboxKey.allCases.map { "\($0.rawValue) — \($0.doc)" }.joined(separator: "; ")).

        Two things you cannot do, and what to tell the user instead: the delegation API key is entered by the user through the app menu — Qwasar ▸ Set Delegation API Key… — and lands in the macOS Keychain; you can report whether one is set (config_show shows it) but never read or write it. Delegation needs both that key AND delegate_models granted at some layer, which IS yours to set. And a model folder must hold a Qwen3.8 27B or Flash-Next 4-bit MLX model; you can name one by path, but not download one.

        Start with config_show. Change only what the user asked for, and say what changed and when it takes effect.
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
            if scope.lowercased() == "app" { return setApp(key: key, value: value) }
            if let r = setProjectKey(target: target, key: key, value: value, scope: scope) { return r }
            guard let k = SandboxKey(rawValue: key) else {
                return "error: unknown key \(key). Keys: "
                     + SandboxKey.allCases.map(\.rawValue).joined(separator: ", ")
            }
            return mutateOverlay(scope: scope, target: target) { overlay in
                k.set(value, on: &overlay)
            }
        case .clear(let scope, let target, let key):
            if scope.lowercased() == "app" {
                guard let key else { return "error: name the app key to reset" }
                return clearApp(key: key)
            }
            if let key, let r = setProjectKey(target: target, key: key, value: nil, scope: scope) { return r }
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

    // MARK: the app's own settings

    /// The keys scope "app" takes.
    static let appKeys = ["server_port", "server_model", "server_context", "server_live_sessions",
                          "server_running", "start_at_login", "checkpoint_disk_budget_gb",
                          "running_notes"]

    private func setApp(key: String, value raw: String) -> String {
        let value = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        let auto = ["auto", "default", ""].contains(value.lowercased())
        func bool() -> Bool? {
            switch value.lowercased() {
            case "true", "yes", "on", "1": return true
            case "false", "no", "off", "0": return false
            default: return nil
            }
        }
        func restarting(_ what: String) -> String {
            guard server.isRunning || pendingServerStart else {
                return "ok: \(what). The server is stopped; it uses this when it next starts."
            }
            requestServerRestart()
            return "ok: \(what). The server restarts when this reply finishes -- this session included, "
                 + "which then resumes from its checkpoint."
        }
        switch key {
        case "server_port":
            guard let p = Int(value), (1...65535).contains(p) else { return "error: server_port must be 1-65535" }
            if p == server.port && pendingPort == nil { return "ok: the server already uses port \(p)" }
            if server.isRunning || pendingServerStart { pendingPort = p } else { server.port = p }
            return restarting("server_port is \(p) (API at http://127.0.0.1:\(p)/v1)")
        case "server_model":
            let path: String
            switch value.lowercased() {
            case "flash-next", "flashnext", "qwen3.8-flash-next":
                guard let p = ModelLibrary.path(for: .flashNext) else {
                    return "error: Flash-Next has not been set up here; give its folder's path"
                }
                path = p
            case "27b", "dense", "qwen3.8-27b":
                guard let p = ModelLibrary.path(for: .dense) else {
                    return "error: the 27B has not been set up here; give its folder's path"
                }
                path = p
            default:
                path = value
            }
            if let err = setModel(path: path) { return "error: \(err)" }
            return restarting("server_model is \(modelPath ?? path)")
        case "server_context":
            if auto { server.contextOverride = nil; return restarting("server_context is automatic") }
            guard let n = Int(value), n >= 4096 else { return "error: server_context must be at least 4096 tokens, or auto" }
            server.contextOverride = n
            return restarting("server_context is \(n) tokens")
        case "server_live_sessions":
            if auto { server.liveSessionsOverride = nil; return restarting("server_live_sessions is automatic") }
            guard let n = Int(value), (1...16).contains(n) else { return "error: server_live_sessions must be 1-16, or auto" }
            server.liveSessionsOverride = n
            return restarting("server_live_sessions is \(n)")
        case "server_running":
            guard let on = bool() else { return "error: server_running is true or false" }
            if on {
                if server.isRunning { return "ok: the server is already running" }
                startServer()
                return "ok: starting the server"
            }
            if !server.isRunning { return "ok: the server is already stopped" }
            pendingServerStop = true
            return "ok: the server stops when this reply finishes. Nothing can answer until it is "
                 + "started again -- from the menu bar's Start Server."
        case "start_at_login":
            guard let on = bool() else { return "error: start_at_login is true or false" }
            do {
                if on { try SMAppService.mainApp.register() } else { try SMAppService.mainApp.unregister() }
            } catch {
                return "error: could not change start at login: \(error.localizedDescription)"
            }
            return "ok: start_at_login is \(on)"
        case "checkpoint_disk_budget_gb":
            guard let gb = Double(value), gb > 0 else { return "error: checkpoint_disk_budget_gb must be a positive number" }
            diskBudgetBytes = UInt64(gb * 1e9)
            return "ok: the checkpoint disk budget is \(formatBytes(diskBudgetBytes)); "
                 + "checkpoints use \(formatBytes(sessionsDiskBytes)). Nothing is dropped automatically."
        case "running_notes":
            guard let on = bool() else { return "error: running_notes is true or false" }
            runningNotesEnabled = on
            return on ? "ok: running notes are on -- written off the record after each reply, while the user reads"
                      : "ok: running notes are off; Continue in a New Session will write them when it is used"
        default:
            return "error: unknown app key \(key). App keys: " + Self.appKeys.joined(separator: ", ")
        }
    }

    private func clearApp(key: String) -> String {
        switch key {
        case "server_port": return setApp(key: key, value: String(ServerController.defaultPort))
        case "server_context", "server_live_sessions": return setApp(key: key, value: "auto")
        case "start_at_login": return setApp(key: key, value: "false")
        case "running_notes": return setApp(key: key, value: "true")
        case "checkpoint_disk_budget_gb":
            let free = (serverInfo?.disk?.free_bytes ?? 0) + (serverInfo?.disk?.sessions_bytes ?? 0)
            diskBudgetBytes = free / 4
            return "ok: the checkpoint disk budget is back to a quarter of the free disk: \(formatBytes(diskBudgetBytes))"
        case "server_model", "server_running":
            return "error: \(key) has no default to go back to; set it instead"
        default:
            return "error: unknown app key \(key). App keys: " + Self.appKeys.joined(separator: ", ")
        }
    }

    /// default_effort and system_prompt, at scope project; nil when `key` is
    /// not one of them (a sandbox key, handled by the layers).
    private func setProjectKey(target: String?, key: String, value: String?, scope: String) -> String? {
        guard ["default_effort", "system_prompt"].contains(key) else { return nil }
        guard scope.lowercased() == "project" else { return "error: \(key) is a project setting; use scope project" }
        guard let target, let i = projects.firstIndex(where: {
            $0.name.lowercased() == target.lowercased() || $0.id.uuidString == target.uppercased()
        }), !projects[i].isConfig else {
            return "error: no project named \(target ?? "(none)"). Projects: "
                 + projects.filter { !$0.isConfig }.map(\.name).joined(separator: ", ")
        }
        switch key {
        case "default_effort":
            let v = (value ?? "auto").trimmingCharacters(in: .whitespaces).lowercased()
            if ["auto", "default", ""].contains(v) {
                projects[i].defaultEffort = nil
            } else {
                guard let e = ReasoningEffort(rawValue: v == "high" ? "xhigh" : v) else {
                    return "error: default_effort is low, medium, xhigh, or auto"
                }
                projects[i].defaultEffort = e
            }
            store?.saveProjects(projects)
            return "ok: new sessions in \(projects[i].name) start at \(defaultEffort(for: projects[i]).rawValue) effort "
                 + "(a session's effort is fixed once it starts)"
        default:
            projects[i].systemPrompt = value ?? ""
            store?.saveProjects(projects)
            return "ok: \(projects[i].name)'s guidance is "
                 + (projects[i].systemPrompt.isEmpty ? "removed" : "set")
                 + ". It applies to sessions opened from now on; open ones keep the prompt they started with."
        }
    }

    private func renderApp() -> [String] {
        let running: String
        switch server.state {
        case .listening: running = "listening"
        case .starting: running = "starting"
        case .stopped: running = "stopped"
        case .stopping: running = "stopping"
        case .portBusy: running = "port busy (another program holds it)"
        case .failed(let why): running = "failed: \(why)"
        }
        let derived = serverInfo.map { "derived: \($0.context) tokens, \($0.live_sessions) live" } ?? "derived when it starts"
        let login = SMAppService.mainApp.status == .enabled
        var out = ["app:"]
        out.append("  server_running: \(server.isRunning) (\(running)); API \(server.apiURL)")
        out.append("  server_port: \(server.port)" + (pendingPort.map { " (becomes \($0) at the restart)" } ?? ""))
        out.append("  server_model: \(modelPath ?? "none chosen")"
                   + (serverInfo.map { " (\($0.model.name))" } ?? ""))
        let known = ModelFamily.allCases.compactMap { f in ModelLibrary.path(for: f).map { "\(f.title) at \($0)" } }
        out.append("  models set up: " + (known.isEmpty ? "none" : known.joined(separator: "; ")))
        out.append("  server_context: " + (server.contextOverride.map { "\($0) tokens" } ?? "auto (\(derived))"))
        out.append("  server_live_sessions: " + (server.liveSessionsOverride.map(String.init) ?? "auto"))
        out.append("  start_at_login: \(login)")
        out.append("  running_notes: \(runningNotesEnabled)")
        out.append("  checkpoint_disk_budget_gb: \(String(format: "%.1f", Double(diskBudgetBytes) / 1e9)) "
                   + "(checkpoints use \(formatBytes(sessionsDiskBytes)))")
        if pendingServerRestart { out.append("  (a server restart is pending: after the current reply)") }
        if pendingServerStop { out.append("  (a server stop is pending: after the current reply)") }
        return out
    }

    private func renderConfig() -> String {
        var out = renderApp()
        for p in projects where !p.isConfig {
            out.append("project \(p.name): default_effort=\(p.defaultEffort?.rawValue ?? "auto (\(defaultEffort(for: p).rawValue))"), "
                       + "system_prompt=" + (p.systemPrompt.isEmpty ? "(none)" : "\"\(p.systemPrompt.prefix(200))\""))
        }
        out.append("sandbox layers:")
        out += ["defaults: " + SandboxKey.allCases.map { k in
            "\(k.rawValue)=\(k.value(in: defaultsAsOverlay) ?? "?")"
        }.joined(separator: ", ")]
        out.append("global: \(describe(globalSandbox))")
        // Presence only, by design (spec §15.4): there is no operation that
        // returns the key.
        out.append("delegation API key: \(KeychainAccess.status())")
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
                         + "timeout=\(eff.toolTimeoutSeconds)s fetch_cap=\(eff.fetchMaxKB)KB "
                         + "agent=[\(eff.agentModels.joined(separator: ", "))] "
                         + String(format: "spent=$%.4f of $%.2f", s.spentUSD ?? 0, eff.agentBudgetUSD))
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
