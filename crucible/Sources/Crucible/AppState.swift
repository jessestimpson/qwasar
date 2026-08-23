// AppState.swift -- the harness, on the main actor.
//
// The engine runs on its own thread (CrucibleKit/EngineHost). This type is the
// only thing that touches both worlds, and only in one direction: events arrive
// from an AsyncStream and are folded into observable state. Nothing here calls
// into the engine except through EngineHost's async API.
//
// PLAN.md 4.3: at most one session is live, because on this machine's profile
// that is all that fits. Selecting a different session closes the incumbent and
// opens the new one. Until M6 that means re-prefilling rather than restoring
// from a checkpoint, which is slow and correct; the UI says which it is doing.

import SwiftUI
import AppKit
import CrucibleKit

@MainActor
@Observable
final class AppState {
    enum Phase: Equatable {
        case needsModel
        case loading(String)
        case ready
        case opening
        case generating
        case failed(String)
    }

    // Engine
    var phase: Phase = .needsModel
    let profile = MemoryProfile.derive()
    let gate = GateCheck.run()
    var engineInfo: EngineInfo?

    // Data
    var projects: [Project] = []
    var sessions: [SessionRecord] = []
    var selectedSessionID: UUID?
    var transcript: [TranscriptItem] = []

    // Turn state
    var draft = ""
    var prefillDone = 0
    var prefillTotal = 0
    /// Live, reported by the session rather than guessed at. Zero means there is
    /// nothing to show -- no live session, or one that has not evaluated yet.
    var contextUsed = 0
    var contextLimit = 0
    /// Live decode rate, reported by the session. Zero when nothing is
    /// generating, which is how the footer knows not to show it.
    var tokensPerSecond = 0.0
    var generatedThisTurn = 0
    var liveSessionID: UUID?

    // Materialisation (PLAN.md 7.4). Nothing here writes anything; `applyApproved`
    // is the only thing that does, and only for paths the user ticked.
    var showingApproval = false
    var isProposing = false
    var proposal: Proposal?
    var proposalError: String?
    var approvedPaths: Set<String> = []
    var lastUndoDirectory: URL?

    private let engine = EngineHost()
    private let access = ModelAccess()
    private var store: Store?
    /// One guest per session, booted lazily (PLAN.md 6.5). Absent until the
    /// image has been built, in which case the session falls back to the
    /// read-only host tools and the UI says so.
    private var sandboxes: SandboxManager?
    var sandboxStatus: String?
    /// Set while the app is quitting and the guests are being flushed.
    var shuttingDown = false
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
        migrateSystemPrompts()
        resolveProjectRoots()
        if let u = access.restore() { Task { await load(u) } }
    }

    // MARK: Model

    func chooseModel() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        // Hidden files shown, because model weights nearly always live in a
        // dot-directory: ~/.lmstudio, ~/.cache/huggingface, ~/.ollama. A picker
        // that hides them makes the common case unreachable, and ⌘⇧. is not
        // something a user should have to know to complete the first step.
        panel.showsHiddenFiles = true
        panel.canCreateDirectories = false
        panel.message = "Choose the Qwen3.8 model directory (config.json + *.safetensors)."
        panel.prompt = "Use Model"
        guard panel.runModal() == .OK, let u = panel.url else { return }
        let resolved = u.resolvingSymlinksInPath()
        guard ModelAccess.looksLikeModel(resolved) else {
            phase = .failed("\(resolved.lastPathComponent) has no config.json and *.safetensors")
            return
        }
        guard access.store(resolved) else {
            phase = .failed("could not hold a security-scoped grant for that folder")
            return
        }
        Task { await load(resolved) }
    }

    private func load(_ u: URL) async {
        phase = .loading("binding weights")
        do {
            engineInfo = try await engine.load(modelPath: u.path,
                                               contextSize: profile.contextSize)
            phase = .ready
        } catch {
            phase = .failed(String(describing: error))
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
        panel.message = "Choose a project directory. Sessions can read inside it and nowhere else."
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

    func removeProject(_ p: Project) {
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
        for i in projects.indices {
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

    func newSession(in p: Project) {
        let r = SessionRecord(projectID: p.id, contextSize: profile.contextSize,
                              storedEffort: p.effort)
        sessions.insert(r, at: 0)
        store?.save(r)
        select(r.id)
    }

    func deleteSession(_ id: UUID) {
        if liveSessionID == id { engine.closeSession(id); liveSessionID = nil }
        if let sandboxes { Task { await sandboxes.discard(session: id) } }
        store?.delete(id)
        sessions.removeAll { $0.id == id }
        if selectedSessionID == id { selectedSessionID = sessions.first?.id; loadTranscript() }
    }

    func select(_ id: UUID?) {
        selectedSessionID = id
        loadTranscript()
        // The meter belongs to the live session. Showing the previous one's
        // figure against a newly selected session would be a precise number
        // about the wrong thing.
        if id != liveSessionID {
            let rec = sessions.first { $0.id == id }
            contextUsed = rec?.tokenCount ?? 0
            contextLimit = Int(rec?.contextSize ?? 0)
        }
    }

    private func loadTranscript() {
        guard let id = selectedSessionID else { transcript = []; return }
        transcript = store?.loadTranscript(id) ?? []
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
            // A session that was opened but never used still holds a system
            // turn rendered at the old effort, so it has to be discarded.
            if liveSessionID == rec.id {
                engine.closeSession(rec.id)
                liveSessionID = nil
            }
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
        if let id = liveSessionID { await engine.closeAndCheckpoint(id) }
        if let sandboxes { await sandboxes.stopAll() }
    }

    /// Whether quitting has anything to wait for.
    var needsShutdown: Bool { liveSessionID != nil || sandboxes != nil }

    // MARK: Sending

    func send() {
        guard case .ready = phase, let rec = selectedSession else { return }
        guard let root = root(of: rec) else {
            phase = .failed("this project's folder is no longer reachable — re-add it")
            return
        }
        guard let p = project(of: rec) else { return }
        let text = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        draft = ""

        Task {
            // One live session. Switching closes the incumbent, which on this
            // profile is the only way to afford the new one (PLAN.md 2.3) --
            // and reopening replays the history, because the transcript on
            // screen and the tokens in the KV cache have to be the same
            // conversation. Without the replay the model would silently start
            // over while the window still showed everything that was said.
            if liveSessionID != rec.id {
                if let old = liveSessionID { engine.closeSession(old) }
                phase = .opening
                prefillDone = 0; prefillTotal = 0
                let history = store?.loadTokens(rec.id) ?? []
                var failed: String?

                // The tools run in the guest when there is one. Without an
                // image the session still works, read-only, against the real
                // tree -- and the header says which world it is in, because
                // "can this change my files" is not a detail to leave implicit.
                var runner: ToolExecuting = ToolRunner(root: root)
                if let sandboxes, sandboxes.isAvailable {
                    do {
                        let ready = try await sandboxes.start(session: rec.id, projectRoot: root)
                        runner = SandboxToolRunner(channel: ready.channel)
                        sandboxStatus = String(format: "sandboxed · booted in %.1fs",
                                               ready.bootSeconds)
                    } catch {
                        sandboxStatus = "read-only — the sandbox did not start"
                    }
                } else {
                    sandboxStatus = "read-only — no guest image (run `make guest`)"
                }

                for await ev in engine.openSession(id: rec.id, runner: runner,
                                                   config: SessionConfig(system: p.systemPrompt,
                                                                         effort: rec.effort),
                                                   history: history) {
                    switch ev {
                    case .prefill(let d, let t): prefillDone = d; prefillTotal = t
                    case .failed(let m): failed = m
                    default: break
                    }
                }
                if let failed { phase = .failed(failed); return }
                liveSessionID = rec.id
                prefillTotal = 0
            }

            phase = .generating
            cancelFlag.clear()
            prefillDone = 0; prefillTotal = 0
            pendingItems = []

            appendItem(TranscriptItem(.user(text)))

            let flag = cancelFlag
            for await ev in engine.send(rec.id, text: text, cancelled: { flag.isSet }) {
                apply(ev)
            }

            // Persist the completed turn, then the record. Turn granularity is
            // deliberate: a crash mid-generation loses this turn and nothing
            // else (PLAN.md 4.2).
            store?.appendTranscript(rec.id, pendingItems)
            pendingItems = []
            let toks = await engine.tokens(of: rec.id)
            store?.saveTokens(rec.id, toks)
            if let i = sessions.firstIndex(where: { $0.id == rec.id }) {
                sessions[i].tokenCount = toks.count
                sessions[i].state = .live
                if sessions[i].title == "New session" {
                    sessions[i].title = String(text.prefix(48))
                }
                store?.save(sessions[i])
            }
            prefillTotal = 0
            tokensPerSecond = 0
            if case .generating = phase { phase = .ready }
        }
    }

    func interrupt() { cancelFlag.set() }

    // MARK: Materialisation

    /// Whether there is a sandbox to ask. Reviewing changes is meaningless
    /// against the read-only host tools, which cannot have made any.
    var canReviewChanges: Bool {
        guard let rec = selectedSession else { return false }
        return sandboxes?.channel(for: rec.id) != nil
    }

    func reviewChanges() {
        guard let rec = selectedSession,
              let root = root(of: rec),
              let channel = sandboxes?.channel(for: rec.id) else { return }

        proposal = nil
        proposalError = nil
        approvedPaths = []
        isProposing = true
        showingApproval = true

        Task {
            let m = Materialiser(channel: channel, projectRoot: root)
            do {
                let p = try await m.propose()
                proposal = p
                // Pre-ticked, except anything the user should look at twice.
                // A conflict starts off because applying over an edit they made
                // themselves is the worst outcome this sheet can produce.
                approvedPaths = Set(p.changes
                    .filter { $0.isApplicable && $0.conflict == nil }
                    .map(\.path))
            } catch {
                proposalError = "\(error)"
            }
            isProposing = false
        }
    }

    func toggleApproval(_ c: ProposedChange) {
        guard c.isApplicable else { return }
        if approvedPaths.contains(c.path) { approvedPaths.remove(c.path) }
        else { approvedPaths.insert(c.path) }
    }

    func applyApproved() async {
        guard let rec = selectedSession,
              let root = root(of: rec),
              let channel = sandboxes?.channel(for: rec.id),
              let p = proposal, let store else { return }

        let m = Materialiser(channel: channel, projectRoot: root)
        let undoRoot = store.root
            .appendingPathComponent("sessions/\(rec.id.uuidString)/undo")

        do {
            let r = try m.apply(p, paths: approvedPaths, undoRoot: undoRoot)
            lastUndoDirectory = r.undoDirectory

            // Reported honestly: a partial application is a partial application.
            var note = "applied \(r.applied.count) of \(approvedPaths.count) file(s)"
            if !r.failed.isEmpty {
                note += " — failed: " + r.failed.map { "\($0.path) (\($0.reason))" }
                                              .joined(separator: "; ")
            }
            if let dir = r.undoDirectory {
                note += ". The previous versions are in \(dir.lastPathComponent)."
            }
            recordNote(note, for: rec.id)

            // The sandbox re-baselines only on what actually landed, so the
            // next proposal is a diff against what the user accepted.
            if r.isComplete { try? await m.acceptBaseline() }
        } catch {
            recordNote("could not apply: \(error)", for: rec.id)
        }
        proposal = nil
        approvedPaths = []
    }

    /// Puts an outcome in the transcript, because applying a change to the
    /// user's files is part of the session's history and not a passing alert.
    private func recordNote(_ text: String, for id: UUID) {
        let item = TranscriptItem(.note(text))
        if selectedSessionID == id { transcript.append(item) }
        store?.appendTranscript(id, [item])
    }

    private func appendItem(_ i: TranscriptItem) {
        transcript.append(i)
        pendingItems.append(i)
    }

    /// Appends into the tail item when the kind matches, so streaming produces
    /// one paragraph rather than one item per token (PLAN.md 5.3).
    ///
    /// The pending copy is found by id, not by position. The two arrays do not
    /// stay in lockstep -- a reloaded transcript starts non-empty while pending
    /// starts empty -- and indexing pending by its own tail would append this
    /// turn's text onto whatever happened to be last.
    private func appendStreaming(_ s: String, reasoning: Bool, tokens: Int = 0) {
        let matches: Bool
        switch transcript.last?.kind {
        case .reasoning: matches = reasoning
        case .assistant: matches = !reasoning
        default: matches = false
        }
        guard matches, let tail = transcript.last else {
            appendItem(TranscriptItem(reasoning ? .reasoning(s) : .assistant(s),
                                      tokens: tokens > 0 ? tokens : nil))
            return
        }
        transcript[transcript.count - 1].append(s, tokens: tokens)
        if let j = pendingItems.firstIndex(where: { $0.id == tail.id }) {
            pendingItems[j].append(s, tokens: tokens)
        }
    }

    private func apply(_ ev: SessionEvent) {
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
            contextUsed = used
            contextLimit = limit
        case .rate(let generated, let rate):
            generatedThisTurn = generated
            tokensPerSecond = rate
        case .reasoning(let s, let n):
            // The first token is the definitive signal that reading is over.
            // A tool result starts a new prefill and the bar comes back.
            prefillTotal = 0
            appendStreaming(s, reasoning: true, tokens: n)
        case .text(let s):
            prefillTotal = 0
            appendStreaming(s, reasoning: false)
        case .toolCall(let c):
            appendItem(TranscriptItem(.tool(name: c.name, arguments: c.arguments, result: nil)))
        case .toolResult(let name, let r):
            // Fill the open card rather than adding a second item.
            if let i = transcript.lastIndex(where: {
                if case .tool(let n, _, let res) = $0.kind { return n == name && res == nil }
                return false
            }), case .tool(let n, let a, _) = transcript[i].kind {
                transcript[i].kind = .tool(name: n, arguments: a, result: r)
                if let j = pendingItems.lastIndex(where: { $0.id == transcript[i].id }) {
                    pendingItems[j].kind = transcript[i].kind
                }
            } else {
                appendItem(TranscriptItem(.tool(name: name, arguments: [:], result: r)))
            }
        case .note(let n):
            prefillTotal = 0
            appendItem(TranscriptItem(.note(n)))
        case .contextFull(let used, let limit):
            prefillTotal = 0
            appendItem(TranscriptItem(.contextFull(used: used, limit: limit)))
        case .turnFinished(let st):
            prefillTotal = 0
            tokensPerSecond = 0
            appendItem(TranscriptItem(.footer(st)))
        case .failed(let m):
            tokensPerSecond = 0
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
