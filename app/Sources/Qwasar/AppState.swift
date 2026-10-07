// AppState.swift -- the harness, on the main actor.
//
// The model runs in qwasar-server, the app's helper (ServerController); this
// type talks to it over the Session API (QwasarClient) and folds what it
// streams into observable state.  Nothing here touches an engine.
//
// A session on the server is opened the first time a record is sent to and
// kept for the record's life (SessionRecord.serverSessionID): the project's
// prompt and the tool schemas are its prefix, every later message is a delta,
// and the server owns liveness and warmth (API.md).  The tools still run
// here -- in the guest when there is one -- on each tool_call the server
// streams, and their results go back as a continue.

import SwiftUI
import AppKit
import QwasarKit

@MainActor
@Observable
final class AppState {
    enum Phase: Equatable {
        /// The server is not listening: stopped, failed, or no model chosen.
        case serverDown(String)
        case loading(String)
        case ready
        case opening
        case generating
        case failed(String)
    }

    // The server
    var phase: Phase = .serverDown("stopped")
    /// qwasar-server, run as this app's helper.
    let server = ServerController()
    /// What /v1/server said the last time the server came up: the model,
    /// the machine's profile, the context every session gets.
    var serverInfo: ServerInfo?
    /// The model folder the server runs, granted through a security-scoped
    /// bookmark (ModelAccess) that the helper inherits.
    private(set) var modelPath: String?
    /// For the status item, which redraws on every change here.
    var onServerChange: (() -> Void)?
    private var client: QwasarClient? {
        server.state == .listening ? QwasarClient(base: server.baseURL) : nil
    }
    let gate = GateCheck.run()

    // Data
    var projects: [Project] = []
    var sessions: [SessionRecord] = []
    var selectedSessionID: UUID?
    /// What the window shows for the selected session: its saved transcript,
    /// and the turn in flight's items when the turn is THIS session's.
    ///
    /// Computed rather than stored, because a turn belongs to the session it
    /// started in, not to whatever is selected while it runs.  When it was a
    /// stored array, the turn streamed into it, and selecting another session
    /// mid-turn swapped the array underneath: the first session's tokens
    /// rendered in the second, and -- since streamed text appended to the
    /// visible tail -- went missing from the first session's saved transcript.
    var transcript: [TranscriptItem] {
        isTurnSelected ? savedTranscript + pendingItems : savedTranscript
    }
    /// The selected session's transcript as it is on disk.
    private var savedTranscript: [TranscriptItem] = []
    /// The session the turn in flight belongs to, from the moment send()
    /// takes it until its items are persisted.  Everything the turn writes
    /// -- items, the pending call, the meters -- is that session's.
    private(set) var turnSessionID: UUID?
    var isTurnSelected: Bool { turnSessionID != nil && turnSessionID == selectedSessionID }
    /// The turn's session, for a view that is showing a different one.
    var turnSessionTitle: String? {
        turnSessionID.flatMap { id in sessions.first { $0.id == id }?.title }
    }
    /// The turn session's context, kept apart from the displayed meter so
    /// that a selection elsewhere does not overwrite it and a selection back
    /// restores it.
    private var turnContext: (used: Int, limit: Int) = (0, 0)

    // Turn state
    var draft = ""
    /// Counts messages sent, so the transcript can snap to its end on each
    /// one: sending is looking at the bottom, wherever you had scrolled to.
    private(set) var sentCount = 0
    var prefillDone = 0
    var prefillTotal = 0
    /// Live, reported by the session rather than guessed at. Zero means there is
    /// nothing to show -- no live session, or one that has not evaluated yet.
    var contextUsed = 0
    var contextLimit = 0
    /// Live decode rate, reported by the session. Zero when nothing is
    /// generating, which is how the footer knows not to show it.
    var tokensPerSecond = 0.0
    /// Rate over roughly the last second, next to the turn average above --
    /// the two diverge whenever the phase changes (sampled reasoning vs
    /// speculative answer), which is exactly when a single number misleads.
    var instantaneousTokensPerSecond = 0.0
    var generatedThisTurn = 0
    var liveSessionID: UUID?

    private let access = ModelAccess()
    private(set) var store: Store?
    /// One guest per session, booted lazily (PLAN.md 6.5). Absent until the
    /// image has been built, in which case the session falls back to the
    /// read-only host tools and the UI says so.
    private var sandboxes: SandboxManager?
    var sandboxStatus: String?
    /// The project whose network allowlist is being edited, when the sheet is
    /// up. Set only from the UI -- no tool result or model output can reach it.
    var networkEditing: Project?
    /// The global sandbox layer (PLAN.md 8.5), loaded once and written only
    /// through performConfig -- the config session's single mutation path.
    var globalSandbox: SandboxOverlay?

    // Delegation (spec §15). The live delegation is what the embedded card
    // renders; the mailbox is the path INTO it. Both exist only while a
    // delegation runs.
    struct LiveDelegation {
        var model: String
        var task: String
        var log: String = ""
        var costUSD: Double = 0
        var ended: String?
        /// The grace window is running: the conversation is open for
        /// steering and will close shortly unless the user types.
        var waiting = false
    }
    var liveDelegation: LiveDelegation?
    /// The session the live delegation belongs to; its card shows there only.
    private(set) var delegationSessionID: UUID?
    var liveDelegationHere: LiveDelegation? {
        delegationSessionID == selectedSessionID ? liveDelegation : nil
    }
    var delegationDraft = ""
    var showingAPIKeySheet = false
    /// The user-initiated delegation sheet (spec §15).
    var showingDelegateSheet = false
    /// A finished user-initiated delegation's answer, waiting to ride along
    /// with the user's next message so the local model sees it. Discardable.
    var pendingHandoff: String?

    // Parking (spec 4.4). `warmTokens` is how many of each session's tokens a
    // checkpoint on disk covers, VERIFIED by probing the store -- never
    // remembered from history, because the LRU evicts and an indicator that
    // reflected the past would turn eviction into a mystery slowdown.
    var warmTokens: [UUID: Int] = [:]
    /// What the server says about each record's session -- warmth, resume
    /// estimate, checkpoint size -- keyed by the record, refreshed with
    /// warmTokens.
    var serverSessions: [UUID: SessionInfo] = [:]

    // Disk (M4).  The server keeps a checkpoint per parked session and never
    // spends one on its own; the budget is the app's, and so is the choice.
    /// Bytes the records' checkpoints use, as last reported.
    var sessionsDiskBytes: UInt64 {
        serverSessions.values.reduce(0) { $0 + ($1.checkpoint_bytes ?? 0) }
    }
    /// Set once, the first time the server reports free space: a quarter of
    /// it (spec 4.4).  Changeable in the disk sheet.
    var diskBudgetBytes: UInt64 = UInt64(UserDefaults.standard.double(forKey: "diskBudgetBytes")) {
        didSet { UserDefaults.standard.set(Double(diskBudgetBytes), forKey: "diskBudgetBytes") }
    }
    var overDiskBudget: Bool { diskBudgetBytes > 0 && sessionsDiskBytes > diskBudgetBytes }
    var showingDiskSheet = false
    private var delegationMailbox: DelegationMailbox?
    /// Set while the app is quitting and the guests are being flushed.
    var shuttingDown = false
    /// Something the ENGINE has to say, which can happen with no session open:
    /// a draft head accepted, removed, or refused. Cleared when it is read by a
    /// reload finishing.
    var engineNote: String?

    /// The call currently being written, if any. Replaced by a real ToolCard
    /// the moment the call parses.
    var pendingCall: (name: String?, keys: [String], tokens: Int)?
    /// The source of the define whose result has not landed yet (tools run
    /// one at a time, so one slot suffices). Captured from the CALL, because
    /// the result reports what loaded but not what was sent.
    private var pendingDefineSource: String?
    private var projectAccess: [UUID: URL] = [:]
    private var cancelFlag = CancelFlag()
    /// Items produced by the turn in flight, appended to the log when it ends.
    private var pendingItems: [TranscriptItem] = []

    var selectedSession: SessionRecord? {
        sessions.first { $0.id == selectedSessionID }
    }

    init() {
        store = try? Store()
        if let s = store {
            let guestDir = Bundle.main.resourceURL?.appendingPathComponent("guest")
                ?? URL(fileURLWithPath: "build/guest")
            sandboxes = SandboxManager(guestDir: guestDir, stateDir: s.root)
        }
        projects = store?.loadProjects() ?? []
        sessions = store?.loadSessions() ?? []
        globalSandbox = store?.loadGlobalOverlay()
        // The config project (PLAN.md 8.5): built in, fixed id, synthesized
        // when absent so it exists on first launch and after any store reset.
        if !projects.contains(where: \.isConfig) {
            projects.append(Project.configProject())
        }
        // The built-in project's name and prompt are the app's, not the
        // user's: a store written by an older build is brought up to date.
        if let i = projects.firstIndex(where: \.isConfig) {
            let current = Project.configProject()
            projects[i].name = current.name
            projects[i].systemPrompt = current.systemPrompt
        }
        migrateSystemPrompts()
        resolveProjectRoots()
        // The model folder's grant, from the last time it was chosen.  The
        // server starts on it when the app asks (QwasarApp), not here.
        if let u = access.restore() { modelPath = u.path }
        server.stateDir = store?.root.appendingPathComponent("server", isDirectory: true)
        server.onChange = { [weak self] in self?.serverChanged() }
    }

    // MARK: The server and its model

    /// Chooses a model folder and restarts the server on it.  Under App
    /// Sandbox the grant is the bookmark ModelAccess keeps, which the helper
    /// inherits; it is remembered per family (ModelLibrary) so the Model menu
    /// switches without the panel next time.
    func chooseModel(startingAt: String? = nil) {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.showsHiddenFiles = true
        panel.canCreateDirectories = false
        panel.message = "Choose a model directory — Qwen3.5 9B, Qwen3.8 27B or Qwen3.8 "
                      + "Flash-Next (config.json + *.safetensors)."
        panel.prompt = "Use Model"
        if let startingAt { panel.directoryURL = URL(fileURLWithPath: startingAt) }
        guard panel.runModal() == .OK, let u = panel.url else { return }
        let resolved = u.resolvingSymlinksInPath()
        guard ModelAccess.looksLikeModel(resolved) else {
            engineNote = "\(resolved.lastPathComponent) has no config.json and *.safetensors"
            return
        }
        guard let family = ModelCatalog.family(of: resolved.path) else {
            engineNote = "\(resolved.lastPathComponent) is not a model the server runs "
                       + "(Qwen3.5 9B, Qwen3.8 27B or Flash-Next, 4-bit)"
            return
        }
        guard access.store(resolved) else {
            engineNote = "could not hold a security-scoped grant for that folder"
            return
        }
        if let data = access.bookmark { ModelLibrary.remember(data, as: family) }
        modelPath = resolved.path
        restartServer()
    }

    /// The Model menu's choice: a folder already granted for its family is
    /// used at once; any other needs one click in the panel, which is what
    /// grants it (the app cannot read a folder it was not handed).
    /// Set when a change needs the server restarted (a config session's
    /// port, model, context or live sessions); done when the turn in flight
    /// ends, or at once if there is none.
    var pendingServerRestart = false
    /// The same, for a stop -- which a config session cannot do to itself
    /// mid-reply either.
    var pendingServerStop = false
    /// Whether the server is coming up, which counts as running for a
    /// change that needs it restarted.
    var pendingServerStart: Bool { if case .starting = server.state { return true }; return false }

    /// A new port, held until the restart: changed at once, the menu bar's
    /// probe would watch an empty port while the old one is still answering.
    var pendingPort: Int?

    func requestServerRestart() {
        if phase == .generating { pendingServerRestart = true } else { applyPendingPort(); restartServer() }
    }

    private func applyPendingPort() {
        if let p = pendingPort { server.port = p; pendingPort = nil }
    }

    /// A model folder by path, without the panel -- the app is not App
    /// Sandboxed, so naming a folder is enough to read it.  Returns an error,
    /// or nil once the model is chosen (the restart is requested separately).
    func setModel(path raw: String) -> String? {
        let u = URL(fileURLWithPath: (raw as NSString).expandingTildeInPath).resolvingSymlinksInPath()
        guard ModelAccess.looksLikeModel(u) else { return "\(u.path) has no config.json and *.safetensors" }
        guard let family = ModelCatalog.family(of: u.path) else {
            return "\(u.path) is not a model the server runs "
                 + "(Qwen3.5 9B, Qwen3.8 27B or Flash-Next, 4-bit MLX)"
        }
        guard access.store(u) else { return "could not keep a bookmark for \(u.path)" }
        if let data = access.bookmark { ModelLibrary.remember(data, as: family) }
        modelPath = u.path
        return nil
    }

    func useModel(path: String) {
        guard let family = ModelCatalog.family(of: path) else { chooseModel(startingAt: path); return }
        if let data = ModelLibrary.bookmark(for: family),
           ModelLibrary.path(for: family).map({ URL(fileURLWithPath: $0).resolvingSymlinksInPath().path })
               == URL(fileURLWithPath: path).resolvingSymlinksInPath().path,
           let u = access.use(bookmark: data) {
            modelPath = u.path
            restartServer()
        } else {
            chooseModel(startingAt: path)
        }
    }

    var knownModels: [ModelFamily] {
        ModelFamily.allCases.filter { ModelLibrary.bookmark(for: $0) != nil }
    }

    /// The family of the model the server runs, from its own report.
    var activeFamily: ModelFamily? {
        serverInfo.flatMap { ModelFamily(modelID: $0.model.id) }
    }

    var speedNote: String { activeFamily?.speedNote ?? "" }

    func startServer() {
        guard let modelPath else { phase = .serverDown("no model chosen"); return }
        server.start(model: modelPath)
        serverChanged()
    }

    func stopServer() {
        Task {
            await settleTurn()
            liveSessionID = nil
            server.stop()
        }
    }

    /// Stops and starts, waiting out a turn in flight and the server's own
    /// checkpoint on the way out.
    func restartServer() {
        guard let modelPath else { return }
        Task {
            await settleTurn()
            liveSessionID = nil
            if server.isRunning {
                await withCheckedContinuation { (c: CheckedContinuation<Void, Never>) in
                    server.stop { c.resume() }
                }
            }
            server.start(model: modelPath)
            serverChanged()
        }
    }

    /// The server's state, into the phase -- and /v1/server read once it is
    /// up, which is when the context every new session gets is known.
    private func serverChanged() {
        onServerChange?()
        switch server.state {
        case .listening:
            if serverInfo == nil {
                phase = .loading("reading the server")
                Task {
                    guard let client else { return }
                    do {
                        serverInfo = try await client.serverInfo()
                        if diskBudgetBytes == 0, let d = serverInfo?.disk, d.free_bytes > 0 {
                            diskBudgetBytes = (d.free_bytes + d.sessions_bytes) / 4
                        }
                        if case .generating = phase {} else { phase = .ready }
                        engineNote = nil
                        select(selectedSessionID)
                        refreshWarm()
                    } catch {
                        phase = .failed("the server at \(server.apiURL) does not speak the Session API: \(error)")
                    }
                }
            } else if case .generating = phase {} else if case .opening = phase {} else {
                phase = .ready
            }
        case .starting:
            serverInfo = nil
            phase = .loading("loading the model")
        case .stopping:
            phase = .loading("stopping the server")
        case .stopped:
            serverInfo = nil
            liveSessionID = nil
            phase = .serverDown(modelPath == nil ? "no model chosen" : "server stopped")
        case .portBusy:
            serverInfo = nil
            phase = .serverDown("port \(server.port) is in use by another program")
        case .failed(let why):
            serverInfo = nil
            liveSessionID = nil
            phase = .serverDown(why)
        }
    }

    /// Winds down a running turn before something that needs the engine
    /// queue idle: sets the cancel flag (the decode loop polls it every
    /// token) and waits for `send()` to finish persisting the turn. Bounded,
    /// because a tool call in flight is not interruptible and the caller
    /// must not hang behind it; `send()` guards its own persistence against
    /// a session closed out from under it.
    private func settleTurn(within seconds: Double = 8) async {
        guard phase == .generating else { return }
        interrupt()
        let deadline = Date().addingTimeInterval(seconds)
        while phase == .generating, Date() < deadline {
            try? await Task.sleep(for: .milliseconds(50))
        }
    }

    // MARK: Projects

    func addProject() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        // Also shown here: a dotfiles repository, ~/.config, or anything under
        // a hidden directory is a perfectly ordinary thing to want to work on.
        panel.showsHiddenFiles = true
        panel.message = "Choose a project directory. Sessions start in it; a sandboxed session can see nothing else."
        panel.prompt = "Add Project"
        guard panel.runModal() == .OK, let u = panel.url else { return }
        let resolved = u.resolvingSymlinksInPath()
        guard let bookmark = try? resolved.bookmarkData(options: .withSecurityScope,
                                                        includingResourceValuesForKeys: nil,
                                                        relativeTo: nil) else {
            phase = .failed("could not hold a security-scoped grant for that folder")
            return
        }
        var p = Project(name: resolved.lastPathComponent, rootBookmark: bookmark)
        _ = resolved.startAccessingSecurityScopedResource()
        p.resolvedRoot = resolved
        projectAccess[p.id] = resolved
        projects.append(p)
        store?.saveProjects(projects)
        newSession(in: p)
    }

    // MARK: Delegation (spec §15)

    /// Observable mirror of "a key exists" -- never the key itself, which
    /// goes straight to the Keychain and nowhere else (spec §15.4).
    var hasAPIKey = KeychainAccess.hasKey

    func setAPIKey(_ key: String) {
        if !KeychainAccess.set(key) {
            // A refused write must not present as "absent" three layers
            // later; say so where the user just acted.
            engineNote = "the Keychain refused the key: \(KeychainAccess.status())"
        }
        hasAPIKey = KeychainAccess.hasKey
        if hasAPIKey { engineNote = "escalation API key stored" }
    }

    func removeAPIKey() {
        KeychainAccess.remove()
        hasAPIKey = false
    }

    func sendToDelegation() {
        let text = delegationDraft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, let mailbox = delegationMailbox else { return }
        delegationDraft = ""
        mailbox.hold(false)
        mailbox.post(text)
        liveDelegation?.waiting = false
    }

    /// Called as the card's input changes: a non-empty draft holds the grace
    /// window open, an emptied one releases it.
    func delegationTyping() {
        delegationMailbox?.hold(!delegationDraft.trimmingCharacters(in: .whitespaces).isEmpty)
    }

    func stopDelegation() { delegationMailbox?.stop() }

    /// Whether the selected session could delegate right now: models granted,
    /// a key present, budget remaining, and no delegation already live.
    var canDelegate: Bool {
        guard liveDelegation == nil, hasAPIKey,
              let rec = selectedSession, let p = project(of: rec), !p.isConfig
        else { return false }
        let s = SandboxSettings.resolve(global: globalSandbox, project: p.overlay,
                                        session: rec.sandbox)
        return !s.agentModels.isEmpty
            && max(0, s.agentBudgetUSD - (rec.spentUSD ?? 0)) > 0
    }

    var delegationModels: [String] {
        guard let rec = selectedSession, let p = project(of: rec) else { return [] }
        return SandboxSettings.resolve(global: globalSandbox, project: p.overlay,
                                       session: rec.sandbox).agentModels
    }

    /// A user-initiated delegation (spec §15): the user pushes the problem up
    /// without waiting for the local model to decide. Interrupts a running
    /// turn -- which is the point, since a model visibly stuck mid-reasoning
    /// is the case this exists for. IDENTICAL to a model-initiated delegation
    /// in every respect: the remote model gets the same sandboxed tool chain
    /// against the same /work (the warden serialises tool requests, so a turn
    /// still winding down cannot collide, only queue).
    func startDelegation(task: String, model: String?, includeContext: Bool) {
        guard canDelegate, let rec = selectedSession, let p = project(of: rec),
              !task.trimmingCharacters(in: .whitespaces).isEmpty else { return }
        if case .generating = phase, turnSessionID == rec.id { interrupt() }

        let settings = SandboxSettings.resolve(global: globalSandbox, project: p.overlay,
                                               session: rec.sandbox)
        let remaining = max(0, settings.agentBudgetUSD - (rec.spentUSD ?? 0))
        var brief = task
        if includeContext {
            let ctx = recentContext()
            if !ctx.isEmpty {
                brief += "\n\n---\n\nRecent conversation, for context:\n\n" + ctx
            }
        }

        // The same tool chain a session's local agent gets: the sandbox over
        // the session's own vsock channel when the guest is up, the read-only
        // host tools when it is not, network per the same resolved policy.
        var inner: ToolExecuting
        if !rec.isSandboxed, let root = root(of: rec) {
            inner = HostToolRunner(root: root, environment: ShellEnvironment.resolve(in: root),
                                   timeout: settings.toolTimeoutSeconds, resultCap: resultCap)
        } else if let channel = sandboxes?.channel(for: rec.id) {
            inner = SandboxToolRunner(channel: channel,
                                      timeout: settings.toolTimeoutSeconds, resultCap: resultCap)
        } else if let root = root(of: rec) {
            inner = ToolRunner(root: root, resultCap: resultCap)
        } else {
            return
        }
        if rec.isSandboxed, !settings.networkAllowlist.isEmpty {
            inner = NetworkToolRunner(inner: inner,
                                      policy: NetworkPolicy(allowlist: settings.networkAllowlist,
                                                            maxResponseBytes: settings.fetchMaxKB * 1024))
        }

        let mailbox = DelegationMailbox()
        delegationMailbox = mailbox
        let sid = rec.id
        let runner = DelegateToolRunner(
            inner: inner,
            policy: EscalationPolicy(models: settings.agentModels,
                                     sessionRemainingUSD: remaining,
                                     turnBudgetUSD: settings.agentTurnBudgetUSD),
            mailbox: mailbox,
            emit: { ev in
                DispatchQueue.main.async {
                    MainActor.assumeIsolated { self.handleDelegation(ev, session: sid) }
                }
            },
            logProvider: makeLogProvider())
        var args = ["task": brief]
        if let model { args["model"] = model }
        Task.detached {
            let out = runner.run(ToolCall(name: "delegate", arguments: args))
            DispatchQueue.main.async {
                MainActor.assumeIsolated {
                    if !out.hasPrefix("error:") { self.pendingHandoff = out }
                }
            }
        }
    }

    /// The tail of the conversation, for the delegation brief: the last user
    /// message and everything the local model produced after it -- which,
    /// when it is stuck, is exactly the reasoning worth showing the expert.
    private func recentContext(cap: Int = 6000) -> String {
        var parts: [String] = []
        for item in transcript.reversed() {
            switch item.kind {
            case .user(let t):
                parts.append("USER: \(t)")
                return String(parts.reversed().joined(separator: "\n\n").suffix(cap))
            case .assistant(let t): parts.append("LOCAL MODEL: \(t)")
            case .reasoning(let t): parts.append("LOCAL MODEL (reasoning): \(t)")
            case .tool(let n, _, let r):
                parts.append("TOOL \(n): \((r ?? "").prefix(400))")
            default: break
            }
        }
        return String(parts.reversed().joined(separator: "\n\n").suffix(cap))
    }


    private func handleDelegation(_ ev: DelegationEvent, session sid: UUID) {
        switch ev {
        case .started(let model, let task):
            liveDelegation = LiveDelegation(model: model, task: task)
            delegationSessionID = sid
        case .delta(let piece):
            liveDelegation?.log += piece
            liveDelegation?.waiting = false
        case .waiting:
            liveDelegation?.waiting = true
        case .userTurn(let text):
            liveDelegation?.log += "\n\n**you:** \(text)\n\n"
        case .cost(let usd):
            liveDelegation?.costUSD = usd
        case .toolCall(let name, let args):
            if name == "define",
               let d = args.data(using: .utf8),
               let obj = (try? JSONSerialization.jsonObject(with: d)) as? [String: Any],
               let src = obj["source"] as? String {
                pendingDefineSource = src
            }
            // Compact: the card is a window, not a full transcript; the
            // arguments are truncated the way the local pending-call row is.
            let brief = args.count > 120 ? String(args.prefix(120)) + "…" : args
            liveDelegation?.log += "\n\n`→ \(name) \(brief)`\n"
        case .toolResult(let name, let result):
            if name == "define", let src = pendingDefineSource {
                pendingDefineSource = nil
                recordDefine(source: src, result: result)
            }
            let first = result.split(separator: "\n").first.map(String.init) ?? ""
            let brief = first.count > 160 ? String(first.prefix(160)) + "…" : first
            liveDelegation?.log += "`← \(name): \(brief)`\n\n"
        case .ended(let reason, let usd):
            // The sub-session becomes a transcript item so a parked session
            // replays it readable (spec §15.2), and the spend persists so the
            // budget survives a relaunch (spec §15.3).
            if let live = liveDelegation {
                let item = TranscriptItem(.delegation(model: live.model, task: live.task,
                                                      log: live.log, costUSD: usd,
                                                      ended: reason))
                // A tool-driven delegation ends inside its session's turn and
                // rides that turn's persistence; a user-driven one can end
                // while idle, or while another session's turn runs.
                if turnSessionID == sid {
                    pendingItems.append(item)
                } else {
                    store?.appendTranscript(sid, [item])
                    if selectedSessionID == sid { savedTranscript.append(item) }
                }
            }
            if let i = sessions.firstIndex(where: { $0.id == sid }) {
                sessions[i].spentUSD = (sessions[i].spentUSD ?? 0) + usd
                store?.save(sessions[i])
            }
            liveDelegation = nil
            delegationSessionID = nil
        }
    }

    /// The single write path for a project's network grant (PLAN.md 8.3).
    /// Takes effect when a session is next opened; the live session keeps the
    /// surface it was prefilled with, because the tool list is the system turn.
    func setNetworkAllowlist(_ p: Project, hosts: [String]) {
        guard let i = projects.firstIndex(where: { $0.id == p.id }) else { return }
        var o = projects[i].overlay
        o.networkAllowlist = hosts.isEmpty ? nil : hosts
        projects[i].sandbox = o.isEmpty ? nil : o
        projects[i].networkAllowlist = nil   // legacy field, folded in
        store?.saveProjects(projects)
    }

    func removeProject(_ p: Project) {
        guard !p.isConfig else { return }   // the config project is part of the app
        for s in sessions where s.projectID == p.id { store?.delete(s.id) }
        sessions.removeAll { $0.projectID == p.id }
        projectAccess[p.id]?.stopAccessingSecurityScopedResource()
        projectAccess[p.id] = nil
        projects.removeAll { $0.id == p.id }
        store?.saveProjects(projects)
    }

    /// A project that never customised its system prompt takes the current
    /// default. One that did is left alone -- the user's words are theirs.
    private func migrateSystemPrompts() {
        var changed = false
        for i in projects.indices
        where Project.supersededSystems.contains(projects[i].systemPrompt) {
            projects[i].systemPrompt = Project.defaultSystem
            changed = true
        }
        if changed { store?.saveProjects(projects) }
    }

    private func resolveProjectRoots() {
        for i in projects.indices where !projects[i].isConfig {
            var stale = false
            guard let u = try? URL(resolvingBookmarkData: projects[i].rootBookmark,
                                   options: .withSecurityScope, relativeTo: nil,
                                   bookmarkDataIsStale: &stale) else { continue }
            if u.startAccessingSecurityScopedResource() {
                projects[i].resolvedRoot = u
                projectAccess[projects[i].id] = u
            }
        }
    }

    func project(of s: SessionRecord) -> Project? {
        projects.first { $0.id == s.projectID }
    }

    func sessions(in p: Project) -> [SessionRecord] {
        sessions.filter { $0.projectID == p.id }.sorted { $0.createdAt > $1.createdAt }
    }

    // MARK: Sessions

    /// Bytes a tool result may put in front of the model: what its prefill
    /// rate makes worth sending (ToolKit).
    var resultCap: Int { ToolKit.resultCap(flashNext: activeFamily == .flashNext) }

    /// What a new session in `p` starts at: the project's choice if it made
    /// one, else the model's.  Flash-Next's card is plain that lower effort in
    /// agent work costs more than it saves -- "insufficient analysis, more
    /// failures, and repeated retries" -- so it gets its template's default,
    /// xhigh.  The 27B, at ~6 tokens a second, stays at medium, and so does
    /// the 9B: medium is the effort with no instruction, which is Qwen3.5's
    /// own template, and its reasoning runs long enough unprompted.
    func defaultEffort(for p: Project) -> ReasoningEffort {
        p.defaultEffort ?? (activeFamily == .flashNext ? .xhigh : .medium)
    }

    /// A new session in `p`.  Its tools run on this Mac unless `sandboxed`,
    /// and which is fixed for its life: the tool surface is part of the
    /// session's prefix.
    func newSession(in p: Project, sandboxed: Bool = false) {
        let r = SessionRecord(projectID: p.id, contextSize: Int32(serverInfo?.context ?? 0),
                              storedEffort: defaultEffort(for: p),
                              tools: sandboxed ? .sandbox : .host)
        sessions.insert(r, at: 0)
        store?.save(r)
        select(r.id)
    }

    func deleteSession(_ id: UUID) {
        if liveSessionID == id { liveSessionID = nil }
        // The server's session goes with the record; its checkpoint too.
        if let sid = sessions.first(where: { $0.id == id })?.serverSessionID, let client {
            Task { try? await client.delete(sid) }
        }
        if let sandboxes { Task { await sandboxes.discard(session: id) } }
        store?.delete(id)
        sessions.removeAll { $0.id == id }
        if selectedSessionID == id { select(sessions.first?.id) }
    }

    func select(_ id: UUID?) {
        selectedSessionID = id
        loadTranscript()
        // The meter is the selected session's: the turn's live figure when
        // the turn is this session's, its record's otherwise.  Showing the
        // previous one's figure against a newly selected session would be a
        // precise number about the wrong thing.
        if let id, id == turnSessionID {
            contextUsed = turnContext.used
            contextLimit = turnContext.limit
        } else {
            let rec = sessions.first { $0.id == id }
            contextUsed = rec?.tokenCount ?? 0
            contextLimit = Int(rec?.contextSize ?? 0)
        }
    }

    private func loadTranscript() {
        guard let id = selectedSessionID else { savedTranscript = []; return }
        savedTranscript = store?.loadTranscript(id) ?? []
    }

    /// A session's whole transcript, whether or not it is selected: saved,
    /// plus the turn's items when the turn is its.
    private func fullTranscript(of id: UUID) -> [TranscriptItem] {
        let saved = id == selectedSessionID ? savedTranscript : (store?.loadTranscript(id) ?? [])
        return id == turnSessionID ? saved + pendingItems : saved
    }

    /// Working directory for a session: the project root plus its subpath.
    func root(of s: SessionRecord) -> URL? {
        guard let p = project(of: s), let base = p.resolvedRoot else { return nil }
        return s.workingSubpath.isEmpty ? base : base.appendingPathComponent(s.workingSubpath)
    }

    var isLiveSelected: Bool { liveSessionID != nil && liveSessionID == selectedSessionID }

    /// True while the selected session can still change its effort for free.
    ///
    /// Effort rewrites the system turn, and the system turn is the prefix of
    /// everything (PLAN.md 2.2) -- so once a session has evaluated anything,
    /// changing it would mean re-prefilling the whole conversation. Before the
    /// first message it costs nothing.
    var canChangeEffort: Bool {
        guard let s = selectedSession else { return false }
        return s.tokenCount == 0
    }

    /// Sets the effort for the selected session if it has not started, and
    /// always for the project, so the next session starts where you left it.
    func setEffort(_ e: ReasoningEffort) {
        guard let rec = selectedSession else { return }

        if let pi = projects.firstIndex(where: { $0.id == rec.projectID }) {
            projects[pi].defaultEffort = e
            store?.saveProjects(projects)
        }

        guard canChangeEffort else { return }
        if let si = sessions.firstIndex(where: { $0.id == rec.id }) {
            sessions[si].storedEffort = e
            store?.save(sessions[si])
            // A server session opened at the old effort holds a prefix
            // rendered at it, so it is discarded; the next send opens a new one.
            if let sid = sessions[si].serverSessionID, let client {
                sessions[si].serverSessionID = nil
                store?.save(sessions[si])
                Task { try? await client.delete(sid) }
            }
            if liveSessionID == rec.id { liveSessionID = nil }
        }
    }

    // MARK: Quitting

    /// Flushes everything that lives outside this process before it exits.
    ///
    /// The guest's disk is the model's work. A VM that is killed with its host
    /// never runs the `sync; poweroff` its init traps, so anything still in the
    /// guest's page cache is lost -- and until this existed, quitting Crucible
    /// did exactly that, every time. `stopAll` sends the stop request and waits
    /// out the grace period, which is what gets the guest to flush.
    ///
    /// The engine checkpoint is the cheap half and is here for a different
    /// reason: nothing is lost without it, but a long session re-prefills from
    /// zero on the next open.
    func shutdown() async {
        shuttingDown = true
        // AppKit has stopped delivering events by now (terminateLater), so a
        // turn still generating would sit in front of the checkpoint on the
        // serial engine queue for as long as it runs -- minutes, for an
        // agent loop -- with the app beach-balling the whole way, until the
        // user force-quits and the guest loses its page cache. Interrupt it
        // first; it stops at the next token.
        await settleTurn()
        if let sandboxes { await sandboxes.stopAll() }
        // The server checkpoints every live session on its way out; the
        // controller allows it a minute for a long conversation.
        if server.isRunning {
            await withCheckedContinuation { (c: CheckedContinuation<Void, Never>) in
                server.stop { c.resume() }
            }
        }
    }

    // MARK: Sending

    func send() {
        guard case .ready = phase, let rec = selectedSession else { return }
        guard let p = project(of: rec) else { return }
        // The config project has no folder; its tools are the host's own
        // (PLAN.md 8.5). Every other project needs its root back.
        let root = root(of: rec)
        if root == nil && !p.isConfig {
            phase = .failed("this project's folder is no longer reachable — re-add it")
            return
        }
        let text = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        draft = ""

        sentCount += 1
        // The turn is this record's from here until its items are persisted,
        // whatever is selected in the meantime.
        turnSessionID = rec.id
        pendingItems = []
        turnContext = (rec.tokenCount, Int(rec.contextSize))

        Task {
            // However it ends -- persisted, refused, failed to open -- the
            // turn stops being anyone's.
            defer { turnSessionID = nil }
            guard let client, let sidx = sessions.firstIndex(where: { $0.id == rec.id }) else {
                phase = .failed("the server is not running"); return
            }
            // Switching sessions: the record's own server session is what
            // resumes -- the server holds or restores it -- so nothing here
            // replays anything.  The incumbent's guest keeps running until
            // it is parked or evicted.
            let switching = liveSessionID != rec.id
            if switching { phase = .opening; prefillDone = 0; prefillTotal = 0 }

            var runner: ToolExecuting = ToolRunner(root: root ?? URL(fileURLWithPath: "/"))
            // Whether the tools run in the guest, for the prompt's environment.
            var toolsInGuest = false
            do {
                // A session's tools run on this Mac, or -- when it was created
                // sandboxed -- in the guest.  A sandboxed session with no
                // guest still works, read-only, against the real tree; the
                // header says which world it is in, because "can this change
                // my files" is not a detail to leave implicit.
                if p.isConfig {
                    // The config project (PLAN.md 8.5): host-side config
                    // tools, no folder, no VM, no network wrapper.
                    runner = configToolRunner()
                    sandboxStatus = "config session — host config tools, no sandbox"
                } else {
                    let root = root!    // guarded at the top of send()
                    // The settings this session runs with: session over
                    // project over global over the defaults (PLAN.md 8.5),
                    // resolved once at open and fixed for the boot.
                    let settings = SandboxSettings.resolve(global: globalSandbox,
                                                           project: p.overlay,
                                                           session: rec.sandbox)
                    runner = ToolRunner(root: root, resultCap: resultCap)
                    if !rec.isSandboxed {
                        // On this Mac, with the user's own shell environment,
                        // resolved in the project (a login shell can take a
                        // second or two, so off the main actor).
                        let env = await Task.detached { ShellEnvironment.resolve(in: root) }.value
                        runner = HostToolRunner(root: root, environment: env,
                                                timeout: settings.toolTimeoutSeconds,
                                                resultCap: resultCap)
                        sandboxStatus = "on this Mac · \(URL(fileURLWithPath: ShellEnvironment.loginShell).lastPathComponent) environment"
                    } else if let sandboxes, sandboxes.isAvailable {
                        do {
                            let ready = try await sandboxes.start(session: rec.id,
                                                                  projectRoot: root,
                                                                  settings: settings)
                            runner = SandboxToolRunner(channel: ready.channel,
                                                       timeout: settings.toolTimeoutSeconds,
                                                       resultCap: resultCap)
                            toolsInGuest = true
                            sandboxStatus = String(format: "sandboxed · booted in %.1fs",
                                                   ready.bootSeconds)
                            // The project's tools, into this guest (spec
                            // 7.2): what one session defined, every sibling
                            // has from its first token.
                            await replaySkillLibrary(p, channel: ready.channel)
                        } catch {
                            sandboxStatus = "read-only — the sandbox did not start"
                        }
                    } else {
                        sandboxStatus = "read-only — no guest image (run `make guest`)"
                    }
                    // Network, when the resolved settings grant any (PLAN.md
                    // 8.3). The fetch tool is the HOST's -- the wrapper
                    // answers it itself and delegates everything else -- so
                    // the guest stays exactly as network-less as before.
                    let net = settings.networkAllowlist
                    if rec.isSandboxed, !net.isEmpty {
                        runner = NetworkToolRunner(
                            inner: runner,
                            policy: NetworkPolicy(allowlist: net,
                                                  maxResponseBytes: settings.fetchMaxKB * 1024))
                        sandboxStatus = (sandboxStatus ?? "") + " · net: \(net.count) host\(net.count == 1 ? "" : "s")"
                    }
                    // Delegation (spec §15): advertised only when models are
                    // granted, a key exists, and budget remains -- the same
                    // absent-means-absent rule fetch follows.
                    let remaining = max(0, settings.agentBudgetUSD - (rec.spentUSD ?? 0))
                    if !settings.agentModels.isEmpty, KeychainAccess.hasKey, remaining > 0 {
                        let mailbox = DelegationMailbox()
                        delegationMailbox = mailbox
                        let sid = rec.id
                        runner = DelegateToolRunner(
                            inner: runner,
                            policy: EscalationPolicy(models: settings.agentModels,
                                                     sessionRemainingUSD: remaining,
                                                     turnBudgetUSD: settings.agentTurnBudgetUSD),
                            mailbox: mailbox,
                            emit: { ev in
                                DispatchQueue.main.async {
                                    MainActor.assumeIsolated {
                                        self.handleDelegation(ev, session: sid)
                                    }
                                }
                            },
                            logProvider: makeLogProvider())
                        sandboxStatus = (sandboxStatus ?? "")
                            + String(format: " · delegate: $%.2f", remaining)
                    }
                }

            }

            // The server's session for this record: opened once, with the
            // executor's description of the environment and the project's
            // own prompt as its prefix (spec 4.1 -- the executor states the
            // environment, the project its preferences).
            var sid = sessions[sidx].serverSessionID
            if sid == nil, sessions[sidx].tokenCount > 0 {
                // A conversation from before the server owned sessions: its
                // tokens were never the server's, so a new server session
                // would start from nothing under a transcript that shows
                // everything -- the model silently forgetting.  Readable,
                // and closed (PLAN-qwasar.md §7, question 4).
                phase = .ready
                engineNote = "this session is from before the server; it can be read but not continued -- start a new one"
                return
            }
            if sid == nil {
                phase = .opening
                do {
                    let o = try await client.open(
                        system: p.isConfig
                            ? runner.environmentDescription + "\n\n" + p.systemPrompt
                            : SystemPrompt.build(toolsDescription: runner.environmentDescription,
                                                 environment: toolsInGuest ? .guest(root: root!) : .host(root: root!),
                                                 projectRoot: root, projectPrompt: p.systemPrompt),
                        tools: runner.schemas, thinking: true, effort: rec.effort.rawValue,
                        metadata: ["client": "qwasar-app", "project": p.name,
                                   "record": rec.id.uuidString, "title": rec.title])
                    sid = o.id
                    sessions[sidx].serverSessionID = o.id
                    sessions[sidx].contextSize = Int32(o.context)
                    store?.save(sessions[sidx])
                    appendItem(TranscriptItem(.note("session opened on the server: "
                        + "\(o.prefix_tokens)-token prefix, \(o.context)-token window")))
                } catch {
                    phase = .failed("cannot open a session: \(error)")
                    return
                }
            }
            guard let sid else { return }
            liveSessionID = rec.id
            turnContext.limit = Int(sessions[sidx].contextSize)
            if isTurnSelected { contextLimit = turnContext.limit }

            phase = .generating
            cancelFlag.clear()
            prefillDone = 0; prefillTotal = 0

            var promptText = text
            if let handoff = pendingHandoff {
                // The harness's voice, marked as such -- the convention the
                // model knows from the harness it was measured in.
                promptText = "<system-reminder>\nThe user asked a remote model about this; its "
                           + "answer follows.\n\n" + handoff + "\n</system-reminder>\n\n" + text
                pendingHandoff = nil
                appendItem(TranscriptItem(.note("the delegation result was attached to this message")))
            }
            // A successor's first message carries its ancestor's notes.
            if sessions[sidx].ancestorID != nil, sessions[sidx].tokenCount == 0,
               let notes = sessions[sidx].notes, !notes.isEmpty {
                promptText = "<system-reminder>\nThis session continues an earlier one that ran out "
                           + "of context. Its working notes follow; treat them as what you already "
                           + "know, and re-read files rather than trusting them where it matters."
                           + "\n\n<notes>\n" + notes + "\n</notes>\n</system-reminder>\n\n" + promptText
            }
            appendItem(TranscriptItem(.user(text)))

            // The turn: a step, and while the model asks for tools, run them
            // here and continue.  Per-step budget by model, as before.
            // Flash-Next at xhigh reasons at length before it acts; its card
            // asks for generous output room, so a step gets 64K there.  The
            // 9B's reasoning alone often passes 4K, and at ~15x the 27B's rate
            // 16K costs it what 4K costs the 27B.
            let budget: Int
            switch activeFamily {
            case .flashNext: budget = rec.effort == .xhigh ? 65_536 : 32_768
            case .nineB: budget = 16_384
            default: budget = 4096
            }
            var stats = TurnStats()
            stats.contextLimit = contextLimit
            var stream = client.turn(sid, text: promptText, maxTokens: budget)
            var steps = 0
            var ended = false
            // A guard against a runaway loop, not a working limit: a long
            // task on Flash-Next's window runs well past the 24 this was.
            // At the cap the model is asked to sum up rather than cut off.
            let maxSteps = 200
            var wrappingUp = false
            // The window's fill, as last reminded about: a long turn is told
            // at 75% and at 90%, once each (the in-turn half of running
            // notes -- notes are taken when a turn ends, so a turn that
            // never ends is asked to find a place to).
            var remindedFill = 0.0
            // The last event heard, and how often this step's stream has had
            // to be picked up again.
            var lastEventID: String?
            var reattaches = 0
            var calls: [(String, ToolCall)] = []
            do {
                while !ended {
                    var stop = ""
                    for try await se in stream {
                        if !se.id.isEmpty { lastEventID = se.id }
                        let ev = se.event
                        switch ev {
                        case .queued(let pos):
                            apply(.note("waiting for the engine (position \(pos))"))
                        case .resume(let from, let restored, let prefill, let cached):
                            if from == "checkpoint" {
                                apply(.note(cached == true
                                    ? "reused the \(restored)-token system prefix from the server's cache"
                                    : "restored \(restored) tokens from a checkpoint; \(prefill) to evaluate"))
                            } else if from == "cold", restored == 0, prefill > 64 {
                                apply(.note("evaluating \(prefill) tokens"))
                            }
                        case .prefill(let d, let t): apply(.prefill(done: d, total: t))
                        case .context(let u, let l): apply(.context(used: u, limit: l))
                        case .reasoning(let t, let n): apply(.reasoning(t, tokens: n))
                        case .text(let t): apply(.text(t))
                        case .decode(let g, let tps, let inst):
                            apply(.rate(generated: g, tokensPerSecond: tps, instantaneous: inst))
                        case .callProgress(let name, let keys, let n):
                            apply(.toolCallProgress(name: name, keys: keys, tokens: n))
                        case .toolCall(let id, let name, let args):
                            let c = ToolCall(name: name, arguments: args)
                            calls.append((id, c))
                            apply(.toolCall(c))
                        case .done(let s, let prompt, let generated, let reasoning, let pf, let dec, _,
                                   let rounds, let committed, let used, let limit, _):
                            stop = s
                            stats.promptTokens += prompt
                            stats.generatedTokens += generated
                            stats.reasoningTokens += reasoning
                            stats.prefillSeconds += pf
                            stats.decodeSeconds += dec
                            stats.specRounds += rounds
                            stats.specCommitted += committed
                            stats.contextUsed = used
                            stats.contextLimit = limit
                            apply(.context(used: used, limit: limit))
                        case .error(let m):
                            apply(.failed(m)); ended = true
                        case .other: break
                        }
                    }
                    if ended { break }
                    if stop.isEmpty {
                        // The stream ended without the step's `done`: the
                        // connection was lost, not the turn.  The server
                        // carries on and keeps the step's events, so pick
                        // them up after the last one heard (API.md 4.6).
                        // Treating this as the model finishing is what left
                        // a written tool call waiting, unrun, with nothing on
                        // screen to say so.
                        if reattaches < 3 {
                            reattaches += 1
                            apply(.note("lost the stream from the server mid-step; picking it up again"))
                            stream = client.events(sid, after: lastEventID)
                            continue
                        }
                        apply(.failed("the stream from the server kept ending before the step did; "
                                    + "the server may still hold the step's result -- send a message to continue"))
                        break
                    }
                    reattaches = 0
                    switch stop {
                    case "tool_calls" where wrappingUp:
                        // Asked to sum up, it asked for tools instead.  They
                        // are not run: the session is left awaiting them,
                        // which a next message closes (API.md 4.5).
                        apply(.note("the model asked for \(calls.count) more tool call(s) after the cap; "
                                  + "not run -- send a message to carry on"))
                        calls = []
                        ended = true
                    case "tool_calls":
                        var results: [ToolResultPayload] = []
                        let executor = runner      // a let: a captured var is not sendable
                        for (id, c) in calls {
                            // The tool runs off the main actor: a call into the
                            // guest can take seconds, and the window must not.
                            // The model's questions are the user's to answer:
                            // a card in the transcript, and the turn waits.
                            let r = c.name == UserQuestions.toolName
                                ? await askUser(c)
                                : await Task.detached { @Sendable in executor.run(c) }.value
                            apply(.toolResult(name: c.name, result: r))
                            results.append(ToolResultPayload(id: id, content: r))
                            stats.toolCalls += 1
                        }
                        calls = []
                        steps += 1
                        if steps >= maxSteps, !results.isEmpty {
                            // The results still go -- nothing ran for
                            // nothing -- with the request riding on the last.
                            results[results.count - 1].content += "\n\n<system-reminder>\nThis turn "
                                + "has made \(steps) rounds of tool calls, the most it may. Make no "
                                + "more tool calls. Summarize what you have learned so far, what you "
                                + "have changed, and what remains to be done, so the user can decide "
                                + "how to continue.\n</system-reminder>"
                            apply(.note("reached \(steps) rounds of tool calls; asking the model to summarize"))
                            stats.hitStepCap = true
                            wrappingUp = true
                        } else if !results.isEmpty, stats.contextLimit > 0 {
                            let fill = Double(stats.contextUsed) / Double(stats.contextLimit)
                            if let mark = [0.9, 0.75].first(where: { fill >= $0 && remindedFill < $0 }) {
                                remindedFill = mark
                                results[results.count - 1].content += "\n\n<system-reminder>\nThe context "
                                    + "window is \(Int(fill * 100))% used (\(stats.contextUsed) of "
                                    + "\(stats.contextLimit) tokens). At the next natural milestone, stop "
                                    + "and report to the user, so that working notes can be taken -- the "
                                    + "work may need to continue in a fresh session.\n</system-reminder>"
                                apply(.note("the context is \(Int(fill * 100))% full; the model was told"))
                            }
                        }
                        stream = client.continueStep(sid, results: results, maxTokens: budget)
                    case "cancelled":
                        stats.interrupted = true; ended = true
                    case "length":
                        stats.hitBudget = true
                        apply(.note("stopped at the \(budget)-token budget"))
                        ended = true
                    case "context_full":
                        apply(.contextFull(used: stats.contextUsed, limit: stats.contextLimit))
                        ended = true
                    default:
                        stats.hitEOS = true; ended = true
                    }
                }
            } catch {
                apply(.failed(String(describing: error)))
            }
            apply(.turnFinished(stats))

            // Persist the completed turn, then the record.  Turn granularity
            // is deliberate: a crash mid-generation loses this turn and
            // nothing else (spec 4.2).
            store?.appendTranscript(rec.id, pendingItems)
            if selectedSessionID == rec.id { savedTranscript += pendingItems }
            pendingItems = []
            if let i = sessions.firstIndex(where: { $0.id == rec.id }) {
                if stats.contextUsed > 0 { sessions[i].tokenCount = stats.contextUsed }
                sessions[i].state = .live
                if let id = serverInfo?.model.id { sessions[i].modelID = id }
                if sessions[i].title == "New session" {
                    sessions[i].title = String(text.prefix(48))
                }
                store?.save(sessions[i])
            }
            prefillTotal = 0
            tokensPerSecond = 0
            instantaneousTokensPerSecond = 0
            if case .generating = phase { phase = .ready }
            refreshWarm()
            // A restart a config session asked for waits for that session's
            // own reply: the server it would restart is the one answering.
            // Notes, while the user reads: off the record, so they cost
            // the conversation nothing.  Only after a turn that ended on its
            // own, in a session long enough to be worth summarizing.
            if !p.isConfig, runningNotesEnabled, !stats.interrupted, stats.contextUsed >= 4096,
               let i = sessions.firstIndex(where: { $0.id == rec.id }),
               (sessions[i].notesTokens ?? 0) < stats.contextUsed {
                let id = rec.id
                Task { await takeNotes(id) }
            }
            if pendingServerStop {
                pendingServerStop = false
                pendingServerRestart = false
                stopServer()
            } else if pendingServerRestart {
                pendingServerRestart = false
                applyPendingPort()
                restartServer()
            }
        }
    }

    /// Stops the turn at its next token: the server's cancel, and the flag
    /// the delegation code watches.
    func interrupt() {
        cancelFlag.set()
        if pendingQuestion != nil { answerQuestions(nil) }
        guard let rec = liveSessionID.flatMap({ id in sessions.first { $0.id == id } }),
              let sid = rec.serverSessionID, let client else { return }
        Task { try? await client.cancel(sid) }
    }

    /// Completed delegations of the turn's session, newest first, for the
    /// local model's delegation_log tool (spec 15.2). Reads the live
    /// transcript, so a delegation finished a moment ago is inspectable in
    /// the same turn.
    private func delegationRecord(_ nth: Int) -> DelegationRecord? {
        // The turn's session: it is the local model in that turn asking.
        guard let id = turnSessionID ?? selectedSessionID else { return nil }
        let all = fullTranscript(of: id).compactMap { item -> DelegationRecord? in
            if case .delegation(let m, let t, let l, _, let e) = item.kind {
                return DelegationRecord(model: m, task: t, log: l, ended: e)
            }
            return nil
        }
        guard nth >= 1, nth <= all.count else { return nil }
        return all[all.count - nth]
    }

    private func makeLogProvider() -> @Sendable (Int) -> DelegationRecord? {
        { nth in
            DispatchQueue.main.sync {
                MainActor.assumeIsolated { self.delegationRecord(nth) }
            }
        }
    }

    // MARK: The project skill library (spec 7.2)

    /// A successful define becomes project property: source keyed by module,
    /// replayed into every sibling session's guest at open. Helper modules
    /// are kept too, for the warden's own reason -- later modules may depend
    /// on them.
    private func recordDefine(source: String, result: String) {
        guard !result.hasPrefix("error:") else { return }
        let first = result.split(separator: "\n").first.map(String.init) ?? ""
        guard let colon = first.firstIndex(of: ":"), first.contains("exports") else { return }
        let module = String(first[..<colon])
        var toolName: String?
        if let r = result.range(of: #"registered as the skill \"([^\"]+)\""#,
                                options: .regularExpression) {
            toolName = String(result[r]).split(separator: "\"").dropFirst().first.map(String.init)
        }
        guard let rec = selectedSession,
              let i = projects.firstIndex(where: { $0.id == rec.projectID }),
              !projects[i].isConfig else { return }
        projects[i].recordDefine(module: module, skillName: toolName, source: source)
        store?.saveProjects(projects)
    }

    /// Replays the project's library into a freshly booted guest, in
    /// definition order. Failures are noted, not fatal: a module that no
    /// longer compiles should not hold the session hostage.
    private func replaySkillLibrary(_ p: Project, channel: VsockChannel) async {
        guard let lib = p.skillLibrary, !lib.isEmpty else { return }
        var loaded = 0, failed = 0
        for skill in lib {
            do {
                let r = try await channel.send(op: "define",
                                               args: ["source": .string(skill.source),
                                                      "force": .string("true")],
                                               timeout: 60)
                if r.ok == true { loaded += 1 } else { failed += 1 }
            } catch { failed += 1 }
        }
        if loaded > 0 {
            sandboxStatus = (sandboxStatus ?? "") + " · \(loaded) skill\(loaded == 1 ? "" : "s")"
        }
        if failed > 0 {
            engineNote = "\(failed) skill\(failed == 1 ? "" : "s") failed to reload; define again to update"
        }
    }

    // MARK: Parking (spec 4.4)

    /// Re-reads every session's warmth from the server (spec 4.4: the
    /// indicator claims only what the store just verified -- the server
    /// probes its checkpoints on every describe).
    func refreshWarm() {
        guard let client else { warmTokens = [:]; serverSessions = [:]; return }
        let byServer = Dictionary(uniqueKeysWithValues:
            sessions.compactMap { r in r.serverSessionID.map { ($0, r.id) } })
        Task {
            guard let list = try? await client.list() else { return }
            var fresh: [UUID: Int] = [:]
            var infos: [UUID: SessionInfo] = [:]
            for info in list {
                if let rid = byServer[info.id] {
                    fresh[rid] = info.warmth.covered
                    infos[rid] = info
                }
            }
            warmTokens = fresh
            serverSessions = infos
        }
    }

    /// Gives a session's disk back (M4): its checkpoint goes, the
    /// conversation stays, and opening it again re-evaluates it.  The user's
    /// choice, never the app's or the server's.
    func dropCheckpoint(_ id: UUID) {
        guard let sid = sessions.first(where: { $0.id == id })?.serverSessionID,
              turnSessionID != id, let client else { return }
        Task {
            do {
                let freed = try await client.dropCheckpoint(sid)
                engineNote = String(format: "freed %.1f GB; the session is kept and rebuilds when opened",
                                    Double(freed) / 1e9)
            } catch {
                engineNote = "could not drop the checkpoint: \(error)"
            }
            refreshWarm()
        }
    }

    // MARK: The model's questions (AskUserQuestion)

    /// Questions waiting on the user, shown as a card in the transcript.
    struct PendingQuestion {
        var questions: [UserQuestion]
        var resume: CheckedContinuation<String, Never>
    }
    private(set) var pendingQuestion: PendingQuestion?
    var pendingQuestions: [UserQuestion]? { pendingQuestion?.questions }

    /// Shows the questions and waits for the answer -- however long that
    /// takes: the engine is idle meanwhile, and the user may be thinking.
    private func askUser(_ c: ToolCall) async -> String {
        switch UserQuestions.parse(c) {
        case .failure(let f):
            return "error: \(f.message)"
        case .success(let qs):
            NSApp.requestUserAttention(.informationalRequest)
            return await withCheckedContinuation { k in
                pendingQuestion = PendingQuestion(questions: qs, resume: k)
            }
        }
    }

    /// The card's answer: per question, the chosen labels (and any words of
    /// the user's own); nil when the user skipped, or stopped the turn.
    func answerQuestions(_ answers: [String: [String]]?) {
        guard let p = pendingQuestion else { return }
        pendingQuestion = nil
        p.resume.resume(returning: answers.map { UserQuestions.answerText(p.questions, $0) }
                                   ?? UserQuestions.skipped)
    }

    // MARK: Running notes and successors

    /// Off by the config session's running_notes; on by default.
    var runningNotesEnabled: Bool {
        get { UserDefaults.standard.object(forKey: "runningNotes") as? Bool ?? true }
        set { UserDefaults.standard.set(newValue, forKey: "runningNotes") }
    }
    /// The session whose notes are being written right now, for the header.
    var notesInProgress: UUID?

    static let notesPrompt = """
        <system-reminder>
        This is a note-taking step. It is not shown to the user, and it is undone afterwards: nothing you write here stays in this conversation.

        Update the working notes for this session. They are what a fresh session would start from if this work had to continue elsewhere -- so they must stand on their own, for a reader who has seen nothing else. Be concrete: names, paths, commands, numbers. At most about 600 words, in this shape:

        ## Goal
        ## Decisions, and why
        ## Tried and did not work
        ## Current state -- files changed, what works, what does not
        ## Next steps

        Reply with the notes only, in Markdown.
        </system-reminder>
        """

    /// Writes the session's notes in an aside: the server answers the notes
    /// prompt with thinking off and rolls the session back, so this costs
    /// the conversation nothing and the user's next message waits for none
    /// of it (the server ends an aside when a real step arrives).  Returns
    /// whether notes were written.
    @discardableResult
    func takeNotes(_ id: UUID) async -> Bool {
        guard notesInProgress == nil, let client,
              let i = sessions.firstIndex(where: { $0.id == id }),
              let sid = sessions[i].serverSessionID else { return false }
        notesInProgress = id
        defer { notesInProgress = nil }
        var text = Self.notesPrompt
        if let old = sessions[i].notes, !old.isEmpty {
            text += "\n\nThe current notes, to replace -- keep what is still true:\n\n<notes>\n"
                  + old + "\n</notes>"
        }
        let seen = sessions[i].tokenCount
        var out = ""
        var stop = ""
        do {
            for try await se in client.aside(sid, text: text, maxTokens: 1200) {
                switch se.event {
                case .text(let t): out += t
                case .done(let s, _, _, _, _, _, _, _, _, _, _, _): stop = s
                default: break
                }
            }
        } catch {
            return false      // refused: busy, not in memory, or images -- next time
        }
        let notes = out.trimmingCharacters(in: .whitespacesAndNewlines)
        guard stop == "end_turn" || stop == "length", !notes.isEmpty,
              let j = sessions.firstIndex(where: { $0.id == id }) else { return false }
        sessions[j].notes = notes
        sessions[j].notesAt = Date()
        sessions[j].notesTokens = seen
        store?.save(sessions[j])
        return true
    }

    /// How full the selected session's window is, 0...1.
    var contextFill: Double {
        contextLimit > 0 ? Double(contextUsed) / Double(contextLimit) : 0
    }

    /// Whether to offer Continue in a New Session: a window 85% used, or full.
    var offerSuccessor: Bool {
        guard let rec = selectedSession, rec.successorID == nil,
              let p = project(of: rec), !p.isConfig else { return false }
        return contextFill >= 0.85 || rec.state == .archived
            || transcript.contains { if case .contextFull = $0.kind { return true }; return false }
    }

    /// Continues `id` in a new session: brings its notes up to date when
    /// there is room to, opens a successor in the same project with the same
    /// placement and effort, seeded with those notes, and unloads and
    /// archives the old one.  The successor's first message carries the
    /// notes; the old session stays readable.
    func continueInNewSession(_ id: UUID) {
        guard phase != .generating, let i = sessions.firstIndex(where: { $0.id == id }),
              let p = project(of: sessions[i]) else { return }
        Task {
            if (sessions[i].notesTokens ?? 0) < sessions[i].tokenCount {
                engineNote = "writing notes for the new session…"
                await takeNotes(id)
            }
            guard let k = sessions.firstIndex(where: { $0.id == id }) else { return }
            let old = sessions[k]
            var notes = old.notes ?? ""
            if notes.isEmpty {
                // No room left to write any: the last thing the model told
                // the user is the best summary there is.
                notes = "No working notes were taken. The model's last reply in that session was:\n\n"
                      + (fullTranscript(of: id).last { if case .assistant = $0.kind { return true }; return false }
                            .map { if case .assistant(let t) = $0.kind { return t }; return "" } ?? "(none)")
            }
            var next = SessionRecord(projectID: p.id, title: Self.successorTitle(old.title),
                                     contextSize: Int32(serverInfo?.context ?? Int(old.contextSize)),
                                     storedEffort: old.storedEffort ?? defaultEffort(for: p),
                                     tools: old.placement)
            next.ancestorID = old.id
            next.notes = notes
            next.notesAt = Date()
            next.notesTokens = 0
            sessions.insert(next, at: 0)
            sessions[k + 1].successorID = next.id
            sessions[k + 1].state = .archived
            store?.save(sessions[k + 1])
            store?.save(next)
            if isInMemory(old.id) { unload(old.id) }
            let opening = TranscriptItem(.note("continues “\(old.title)”: its notes go with your first "
                                               + "message (Notes, above, shows them)"))
            store?.appendTranscript(next.id, [opening])
            select(next.id)
            engineNote = nil
        }
    }

    static func successorTitle(_ t: String) -> String {
        if let r = t.range(of: #" \(continued( \d+)?\)$"#, options: .regularExpression) {
            let base = String(t[..<r.lowerBound])
            let n = Int(t[r].filter(\.isNumber)) ?? 1
            return "\(base) (continued \(n + 1))"
        }
        return t + " (continued)"
    }

    /// Whether the server holds `id` in memory -- by its own report, or
    /// because it is this window's live session.
    func isInMemory(_ id: UUID) -> Bool {
        liveSessionID == id || serverSessions[id]?.warmth.state == "live"
    }

    /// Takes a session out of memory (spec 4.4's one user verb).  `save`
    /// writes its checkpoint first, so the next message resumes with
    /// nothing to re-read; without it nothing is written, the memory goes at
    /// once, and the next message re-reads whatever the disk does not cover.
    /// Either way the server frees the memory outright, and a sandboxed
    /// session's VM stops -- its disk survives and reboots on the next open.
    func unload(_ id: UUID, save: Bool = true) {
        guard turnSessionID != id || phase != .generating else { return }
        Task {
            if let sid = sessions.first(where: { $0.id == id })?.serverSessionID, let client {
                do {
                    let w = try await client.park(sid, save: save)
                    let total = sessions.first(where: { $0.id == id })?.tokenCount ?? 0
                    engineNote = w.covered >= total
                        ? "unloaded from memory; its checkpoint on disk covers all of it"
                        : "unloaded from memory without saving; \(w.covered) of \(total) tokens "
                          + "are on disk, and the rest is re-read when it is next used"
                } catch {
                    engineNote = "could not unload the session: \(error)"
                }
            }
            if let sandboxes { await sandboxes.stop(session: id) }
            if liveSessionID == id { liveSessionID = nil; sandboxStatus = nil }
            if let i = sessions.firstIndex(where: { $0.id == id }) {
                sessions[i].state = .closed
                store?.save(sessions[i])
            }
            refreshWarm()
        }
    }

    // MARK: Refreshing the .git shadow (spec 7.4)

    /// Whether Refresh Git makes sense right now: a git project, its
    /// sandbox up, nothing generating. The refresh needs the channel (to
    /// arm the re-seed) and ends in a park, so both gates are hard.
    var canRefreshGit: Bool {
        guard phase != .generating, let rec = selectedSession,
              sandboxes?.channel(for: rec.id) != nil,
              let root = root(of: rec) else { return false }
        var isDir: ObjCBool = false
        return FileManager.default.fileExists(atPath: root.appendingPathComponent(".git").path,
                                              isDirectory: &isDir) && isDir.boolValue
    }

    /// Re-seeds the sandbox's private .git copy from the repository's
    /// current state. A running guest cannot see the real .git -- the bind
    /// mount is the point -- so refresh is a reboot: arm the re-seed (the
    /// guest drops its seed stamp), then park; the next message boots a
    /// guest whose mount-work re-seeds while nothing but init is running.
    /// The shadow's private commits are discarded; the user's tree and
    /// real .git are untouched.
    func refreshGit() {
        guard canRefreshGit, let rec = selectedSession,
              let channel = sandboxes?.channel(for: rec.id) else { return }
        Task {
            let r = try? await channel.send(op: "git_refresh", timeout: 30)
            guard r?.ok == true else {
                engineNote = "refresh failed: \(r?.error ?? "the guest gave no reason")"
                return
            }
            let item = TranscriptItem(.note("git refresh armed: the sandbox reboots on "
                + "your next message and re-seeds its private .git copy from the "
                + "repository's current state. Your files and your real .git are untouched."))
            if selectedSessionID == rec.id { savedTranscript.append(item) }
            store?.appendTranscript(rec.id, [item])
            if liveSessionID == rec.id {
                unload(rec.id)
            } else if let sandboxes {
                await sandboxes.stop(session: rec.id)
            }
        }
    }

    /// Appends to the turn in flight.  Never to what is on screen: the view
    /// derives that (`transcript`), and shows it only in the turn's session.
    private func appendItem(_ i: TranscriptItem) {
        pendingItems.append(i)
    }

    /// Appends into the turn's tail item when the kind matches, so streaming
    /// produces one paragraph rather than one item per token (spec 5.3).
    private func appendStreaming(_ s: String, reasoning: Bool, tokens: Int = 0) {
        let matches: Bool
        switch pendingItems.last?.kind {
        case .reasoning: matches = reasoning
        case .assistant: matches = !reasoning
        default: matches = false
        }
        guard matches else {
            appendItem(TranscriptItem(reasoning ? .reasoning(s) : .assistant(s),
                                      tokens: tokens > 0 ? tokens : nil))
            return
        }
        pendingItems[pendingItems.count - 1].append(s, tokens: tokens)
    }

    // MARK: coalescing the stream
    //
    // A token arrives as up to three events -- its text, the decode rate, the
    // context meter -- at ~60 tokens a second on Flash-Next, and each one that
    // touched observed state was a SwiftUI transaction over the whole
    // transcript.  So the high-rate events are buffered and applied together
    // at most every `streamInterval`; anything else flushes the buffer first,
    // so order is kept exactly.

    private static let streamInterval: Duration = .milliseconds(50)
    private var streamBuffer: [SessionEvent] = []
    private var streamFlushScheduled = false

    private func apply(_ ev: SessionEvent) {
        switch ev {
        case .text, .reasoning, .rate, .context, .prefill, .toolCallProgress:
            // Consecutive deltas of the same kind merge; for the others only
            // the latest value matters, so a newer one replaces the older.
            switch (streamBuffer.last, ev) {
            case (.text(let a)?, .text(let b)):
                streamBuffer[streamBuffer.count - 1] = .text(a + b)
            case (.reasoning(let a, let n)?, .reasoning(let b, let m)):
                streamBuffer[streamBuffer.count - 1] = .reasoning(a + b, tokens: n + m)
            default:
                if let i = streamBuffer.lastIndex(where: { Self.sameGauge($0, ev) }) {
                    streamBuffer.remove(at: i)
                }
                streamBuffer.append(ev)
            }
            if !streamFlushScheduled {
                streamFlushScheduled = true
                Task { @MainActor [weak self] in
                    try? await Task.sleep(for: Self.streamInterval)
                    self?.flushStream()
                }
            }
        default:
            flushStream()
            applyNow(ev)
        }
    }

    /// Gauges: events whose latest value supersedes the earlier ones.
    private static func sameGauge(_ a: SessionEvent, _ b: SessionEvent) -> Bool {
        switch (a, b) {
        case (.rate, .rate), (.context, .context), (.prefill, .prefill),
             (.toolCallProgress, .toolCallProgress): return true
        default: return false
        }
    }

    private func flushStream() {
        streamFlushScheduled = false
        guard !streamBuffer.isEmpty else { return }
        let batch = streamBuffer
        streamBuffer = []
        for ev in batch { applyNow(ev) }
    }

    private func applyNow(_ ev: SessionEvent) {
        switch ev {
        case .prefill(let done, let total):
            // Only while there is still prompt left to read. The engine reports
            // once per chunk, and the last report is `done == total` -- so
            // keying the bar off "a prefill was reported" left it pinned at 100%
            // for the whole decode phase, which is most of a turn.
            if done < total {
                prefillDone = done
                prefillTotal = total
            } else {
                prefillTotal = 0
            }
        case .context(let used, let limit):
            turnContext = (used, limit)
            if isTurnSelected {
                contextUsed = used
                contextLimit = limit
            }
        case .rate(let generated, let rate, let inst):
            generatedThisTurn = generated
            tokensPerSecond = rate
            instantaneousTokensPerSecond = inst
        case .reasoning(let s, let n):
            // The first token is the definitive signal that reading is over.
            // A tool result starts a new prefill and the bar comes back.
            prefillTotal = 0
            appendStreaming(s, reasoning: true, tokens: n)
        case .text(let s):
            prefillTotal = 0
            appendStreaming(s, reasoning: false)
        case .toolCallProgress(let name, let keys, let n):
            prefillTotal = 0
            pendingCall = (name, keys, n)
        case .toolCall(let c):
            pendingCall = nil
            if c.name == "define" { pendingDefineSource = c.arguments["source"] }
            appendItem(TranscriptItem(.tool(name: c.name, arguments: c.arguments, result: nil)))
        case .toolResult(let name, let r):
            if name == "define", let src = pendingDefineSource {
                pendingDefineSource = nil
                recordDefine(source: src, result: r)
            }
            // Fill the open card rather than adding a second item.
            if let i = pendingItems.lastIndex(where: {
                if case .tool(let n, _, let res) = $0.kind { return n == name && res == nil }
                return false
            }), case .tool(let n, let a, _) = pendingItems[i].kind {
                pendingItems[i].kind = .tool(name: n, arguments: a, result: r)
            } else {
                appendItem(TranscriptItem(.tool(name: name, arguments: [:], result: r)))
            }
        case .note(let n):
            prefillTotal = 0
            appendItem(TranscriptItem(.note(n)))
        case .contextFull(let used, let limit):
            prefillTotal = 0
            pendingCall = nil
            appendItem(TranscriptItem(.contextFull(used: used, limit: limit)))
        case .turnFinished(let st):
            prefillTotal = 0
            tokensPerSecond = 0
            instantaneousTokensPerSecond = 0
            pendingCall = nil
            appendItem(TranscriptItem(.footer(st)))
        case .failed(let m):
            tokensPerSecond = 0
            instantaneousTokensPerSecond = 0
            pendingCall = nil
            // Every terminal event clears it. A bar left running after a turn
            // ended is the failure this whole change is about, and there is more
            // than one way for a turn to end.
            prefillTotal = 0
            appendItem(TranscriptItem(.note("failed: \(m)")))
            phase = .ready
        }
    }
}

/// A cancellation flag the engine thread polls once per token. The C agent's
/// equivalent is `tui_interrupted`.
final class CancelFlag: @unchecked Sendable {
    private var value = false
    private let lock = NSLock()
    var isSet: Bool { lock.lock(); defer { lock.unlock() }; return value }
    func set() { lock.lock(); value = true; lock.unlock() }
    func clear() { lock.lock(); value = false; lock.unlock() }
}

