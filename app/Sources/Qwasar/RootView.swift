// RootView.swift -- the coding agent's window.
//
// Spec 5.1: NavigationSplitView, the shape every macOS user already knows.
// Sidebar of projects and their sessions; the transcript in the middle; the
// composer pinned below it.  Hosted in an NSWindow by QwasarApp.swift.

import SwiftUI
import QwasarKit

struct RootView: View {
    @Bindable var state: AppState

    var body: some View {
        VStack(spacing: 0) {
            NavigationSplitView {
                Sidebar(state: state)
                    .navigationSplitViewColumnWidth(min: 220, ideal: 260)
            } detail: {
                if state.selectedSession != nil {
                    SessionView(state: state)
                } else {
                    EmptyPane(state: state)
                }
            }
            .toolbar { StatusToolbar(state: state) }
            .sheet(isPresented: $state.showingAPIKeySheet) { APIKeySheet(state: state) }

            // Spans the window, not the detail pane: what the engine is doing
            // is a property of the application, and there is only ever one
            // session doing it (PLAN.md 2.1).
            StatusFooter(state: state)
        }
        .animation(.easeInOut(duration: 0.15), value: state.prefillTotal > 0)
    }
}

// MARK: - Delegation API key (spec §15.4)

struct APIKeySheet: View {
    @Bindable var state: AppState
    @State private var key = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Delegation API key").font(.headline)
            Text("An OpenRouter (or OpenAI-compatible) key. It is stored in "
                 + "the macOS Keychain and attached to requests by the app "
                 + "alone — no model, tool, or config session can read it.")
                .font(.caption).foregroundStyle(.secondary)
            SecureField("sk-or-…", text: $key)
                .textFieldStyle(.roundedBorder)
            Text("Delegation also needs models granted: in a Qwasar Config "
                 + "session, `config_set` the `delegate_models` key at the "
                 + "layer you want.")
                .font(.caption2).foregroundStyle(.tertiary)
            HStack {
                Spacer()
                Button("Cancel") { state.showingAPIKeySheet = false }
                Button("Save") {
                    state.setAPIKey(key)
                    state.showingAPIKeySheet = false
                }
                .keyboardShortcut(.defaultAction)
                .disabled(key.trimmingCharacters(in: .whitespaces).isEmpty)
            }
        }
        .padding(16)
        .frame(width: 420)
    }
}

// MARK: - Network allowlist

/// The one place network gets granted (PLAN.md 8.3): a person, per project,
/// host by host. Nothing the model does can open this sheet or grow the list.
struct NetworkSheet: View {
    @Bindable var state: AppState
    let project: Project
    @State private var text = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Network access — \(project.name)").font(.headline)
            Text("One host per line (e.g. hexdocs.pm, *.github.io). Empty means "
                 + "network OFF, which is the default. `fetch` is HTTPS GET only, "
                 + "run by the app under this list — the sandbox itself still has "
                 + "no network device.")
                .font(.caption).foregroundStyle(.secondary)
            Text("Plainly: with any host granted, a prompt injection in a file "
                 + "the model reads could encode project contents into request "
                 + "URLs to that host. Leave this empty for confidential work.")
                .font(.caption).foregroundStyle(.orange)
            TextEditor(text: $text)
                .font(.body.monospaced())
                .frame(minHeight: 120)
                .overlay(RoundedRectangle(cornerRadius: 4).stroke(.quaternary))
            Text("Applies when a session is next opened; changing the tool "
                 + "surface re-prefills that session once.")
                .font(.caption2).foregroundStyle(.tertiary)
            HStack {
                Spacer()
                Button("Cancel") { state.networkEditing = nil }
                Button("Save") {
                    let hosts = text.split(whereSeparator: \.isNewline)
                        .map { $0.trimmingCharacters(in: .whitespaces).lowercased() }
                        .filter { !$0.isEmpty }
                    state.setNetworkAllowlist(project, hosts: hosts)
                    state.networkEditing = nil
                }
                .keyboardShortcut(.defaultAction)
            }
        }
        .padding(16)
        .frame(width: 440)
        .onAppear { text = (project.overlay.networkAllowlist ?? []).joined(separator: "\n") }
    }
}

// MARK: - Sidebar

struct Sidebar: View {
    @Bindable var state: AppState

    var body: some View {
        List(selection: Binding(
            get: { state.selectedSessionID },
            set: { state.select($0) }
        )) {
            ForEach(state.projects) { project in
                Section {
                    ForEach(state.sessions(in: project)) { s in
                        SessionRow(session: s, isLive: state.liveSessionID == s.id,
                                   info: state.serverSessions[s.id])
                            .tag(s.id)
                            .contextMenu {
                                if state.isInMemory(s.id), s.serverSessionID != nil {
                                    // Option skips the checkpoint write:
                                    // the memory goes at once, and the next
                                    // message re-reads what the disk lacks.
                                    Button("Unload from Memory") {
                                        state.unload(s.id, save: !NSEvent.modifierFlags.contains(.option))
                                    }
                                    .disabled(state.turnSessionID == s.id && state.phase == .generating)
                                    .help("Saves its checkpoint and frees its memory; the next "
                                          + "message resumes from disk. Hold Option to free it "
                                          + "without saving: faster now, and the turns since its "
                                          + "last checkpoint are re-read next time.")
                                }
                                if let b = state.serverSessions[s.id]?.checkpoint_bytes, b > 0,
                                   state.turnSessionID != s.id {
                                    Button("Drop Checkpoint (\(formatBytes(b)))") { state.dropCheckpoint(s.id) }
                                        .help("Frees its disk. The conversation is kept; opening it "
                                              + "again re-evaluates it.")
                                }
                                Button("Delete Session", role: .destructive) {
                                    state.deleteSession(s.id)
                                }
                            }
                    }
                    if project.isConfig {
                        Button {
                            state.newSession(in: project)
                        } label: {
                            Label("New Session", systemImage: "plus").font(.caption)
                        }
                        .buttonStyle(.plain)
                        .foregroundStyle(.secondary)
                    } else {
                        // Click for a session on this Mac; the arrow offers a
                        // sandboxed one.  Which it is cannot change later.
                        Menu {
                            Button("New Session") { state.newSession(in: project) }
                            Button("New Sandboxed Session") {
                                state.newSession(in: project, sandboxed: true)
                            }
                        } label: {
                            Label("New Session", systemImage: "plus").font(.caption)
                        } primaryAction: {
                            state.newSession(in: project)
                        }
                        .menuStyle(.borderlessButton)
                        .fixedSize()
                        .foregroundStyle(.secondary)
                        .help("A new session whose tools run on this Mac, with your shell "
                              + "environment. From the arrow: one whose tools run in a VM "
                              + "with no network, seeing only this folder.")
                    }
                } header: {
                    HStack {
                        Text(project.name)
                        Spacer()
                        if project.isConfig {
                            Image(systemName: "gearshape")
                                .foregroundStyle(.secondary)
                                .help("Built in. Sessions here manage Qwasar's "
                                      + "configuration -- the server, the projects, "
                                      + "the sandbox -- with host-side tools: no "
                                      + "folder, no shell.")
                        } else if project.resolvedRoot == nil {
                            Image(systemName: "exclamationmark.triangle.fill")
                                .foregroundStyle(.orange)
                                .help("This folder is no longer reachable. Re-add the project.")
                        }
                    }
                    .contextMenu {
                        if !project.isConfig {
                            Button("Network…") { state.networkEditing = project }
                            Button("Remove Project", role: .destructive) {
                                state.removeProject(project)
                            }
                        }
                    }
                }
            }
        }
        .listStyle(.sidebar)
        .sheet(item: $state.networkEditing) { p in
            NetworkSheet(state: state, project: p)
        }
        .sheet(isPresented: $state.showingDiskSheet) { DiskSheet(state: state) }
        .safeAreaInset(edge: .bottom) {
            VStack(spacing: 6) {
                if state.overDiskBudget {
                    // Said, never acted on: which sessions to let go cold is
                    // the user's choice (spec 4.4, API.md 4.9).
                    Button { state.showingDiskSheet = true } label: {
                        Label("Parked sessions use \(formatBytes(state.sessionsDiskBytes)) of "
                              + "your \(formatBytes(state.diskBudgetBytes)) — Manage…",
                              systemImage: "externaldrive.badge.exclamationmark")
                            .font(.caption)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(.orange)
                }
                Button {
                    state.addProject()
                } label: {
                    Label("Add Project…", systemImage: "folder.badge.plus")
                        .frame(maxWidth: .infinity)
                }
            }
            .padding(8)
        }
    }
}

struct SessionRow: View {
    let session: SessionRecord
    let isLive: Bool
    /// What the server says about this record's session, or nil (none yet,
    /// or the server is down).  Warm means a checkpoint the server just
    /// verified covers the whole session (spec 4.4: the indicator claims only
    /// what is verified), and the estimate is the server's, from its measured
    /// read and prefill rates -- right for either model.
    var info: SessionInfo? = nil

    /// Three states, each with the number that is its meaning: live holds
    /// memory; parked-warm resumes as a read; parked-cold pays a re-prefill.
    private var symbol: (name: String, tint: Color) {
        if isLive { return ("circle.fill", Color.accentColor) }
        if isWarm { return ("circle.lefthalf.filled", Color.accentColor.opacity(0.7)) }
        return ("circle", Color.secondary)
    }

    private var isWarm: Bool {
        guard let w = info?.warmth else { return false }
        return w.state == "live" || (w.state == "warm" && w.covered >= (info?.tokens ?? 0))
    }

    private var closedForGood: Bool { session.serverSessionID == nil && session.tokenCount > 0 }

    private var detail: String? {
        guard session.tokenCount > 0 else { return nil }
        var parts = ["\(session.tokenCount) / \(session.contextSize) tokens"]
        if closedForGood {
            parts.append("read-only")
        } else if !isLive, let s = info?.warmth.estimate_seconds {
            let secs = Int(s.rounded())
            let t = secs < 90 ? "~\(max(secs, 1))s" : "~\((secs + 30) / 60)m"
            parts.append(isWarm ? "resumes in \(t)" : "rebuilds in \(t)")
        }
        if let b = info?.checkpoint_bytes, b > 0 { parts.append(formatBytes(b)) }
        return parts.joined(separator: " · ")
    }

    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: symbol.name)
                .font(.system(size: 7))
                .foregroundStyle(symbol.tint)
            VStack(alignment: .leading, spacing: 1) {
                HStack(spacing: 4) {
                    Text(session.title).lineLimit(1)
                    if session.isSandboxed {
                        Image(systemName: "shippingbox")
                            .font(.caption2).foregroundStyle(.secondary)
                            .help("Sandboxed: its tools run in a VM with no network.")
                    }
                }
                if let detail {
                    Text(detail).font(.caption2).foregroundStyle(.secondary)
                }
            }
        }
        .help(closedForGood ? "From before the server owned sessions: readable, not continuable."
              : isLive ? "Live: holds its share of the working set."
              : isWarm ? "Parked, warm: its checkpoint on disk covers the whole session."
                       : "Parked, cold: no checkpoint covers it; opening re-evaluates it.")
    }
}

/// Decimal GB/MB, as the budget and the server report them.
func formatBytes(_ b: UInt64) -> String {
    b >= 1_000_000_000 ? String(format: "%.1f GB", Double(b) / 1e9)
                       : String(format: "%.0f MB", Double(b) / 1e6)
}

/// The disk the parked sessions use, and the budget for it (M4).  Lists every
/// session with a checkpoint, least recently used first, each with a Drop:
/// the conversation is kept, it just rebuilds when opened.  Nothing here
/// happens without a click.
struct DiskSheet: View {
    @Bindable var state: AppState
    @State private var budgetGB: Double = 0

    private var rows: [(SessionRecord, SessionInfo)] {
        state.sessions.compactMap { r in
            guard let i = state.serverSessions[r.id], (i.checkpoint_bytes ?? 0) > 0 else { return nil }
            return (r, i)
        }
        .sorted { ($0.1.last_step?.at ?? 0) < ($1.1.last_step?.at ?? 0) }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Session checkpoints").font(.headline)
            Text("A parked session keeps its state on disk so it resumes in seconds. "
                 + "Dropping a checkpoint frees that space; the conversation is kept, and "
                 + "opening it again re-evaluates it. Nothing is dropped unless you do it here.")
                .font(.caption).foregroundStyle(.secondary)
            HStack {
                Text("Using \(formatBytes(state.sessionsDiskBytes))")
                    .foregroundStyle(state.overDiskBudget ? .orange : .primary)
                Spacer()
                Text("Budget")
                TextField("GB", value: $budgetGB, format: .number.precision(.fractionLength(0)))
                    .frame(width: 60).textFieldStyle(.roundedBorder)
                Text("GB")
            }
            .font(.callout)
            List {
                ForEach(rows, id: \.0.id) { r, i in
                    HStack {
                        VStack(alignment: .leading) {
                            Text(r.title).lineLimit(1)
                            Text("\(i.tokens) tokens" + (i.last_step.map {
                                " · last used " + Date(timeIntervalSince1970: TimeInterval($0.at))
                                    .formatted(.relative(presentation: .named)) } ?? ""))
                                .font(.caption2).foregroundStyle(.secondary)
                        }
                        Spacer()
                        Text(formatBytes(i.checkpoint_bytes ?? 0)).monospacedDigit()
                        Button("Drop") { state.dropCheckpoint(r.id) }
                            .disabled(state.turnSessionID == r.id)
                    }
                }
            }
            .frame(minHeight: 180)
            HStack {
                Spacer()
                Button("Done") {
                    if budgetGB > 0 { state.diskBudgetBytes = UInt64(budgetGB * 1e9) }
                    state.showingDiskSheet = false
                }
                .keyboardShortcut(.defaultAction)
            }
        }
        .padding(16)
        .frame(width: 520)
        .onAppear {
            budgetGB = (Double(state.diskBudgetBytes) / 1e9).rounded()
            state.refreshWarm()
        }
    }
}

// MARK: - Empty state

struct EmptyPane: View {
    @Bindable var state: AppState

    var body: some View {
        VStack(spacing: 14) {
            Image(systemName: "square.stack.3d.up")
                .font(.system(size: 40)).foregroundStyle(.tertiary)
            if state.projects.isEmpty {
                Text("Add a project to begin.").font(.title3)
                Text("A project is a folder. Sessions can read inside it and nowhere else.")
                    .font(.callout).foregroundStyle(.secondary)
                Button("Add Project…") { state.addProject() }
            } else {
                Text("Select or create a session.").foregroundStyle(.secondary)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

// MARK: - Toolbar

struct StatusToolbar: ToolbarContent {
    @Bindable var state: AppState

    var body: some ToolbarContent {
        ToolbarItem(placement: .status) {
            HStack(spacing: 10) {
                switch state.phase {
                case .serverDown(let why):
                    if state.modelPath == nil {
                        Button("Choose Model…") { state.chooseModel() }
                    } else {
                        Label(why, systemImage: "exclamationmark.triangle")
                            .font(.caption).foregroundStyle(.secondary).lineLimit(1)
                        Button("Start Server") { state.startServer() }
                    }
                case .loading(let what):
                    ProgressView().controlSize(.small)
                    Text(what).font(.caption).foregroundStyle(.secondary)
                case .opening:
                    ProgressView().controlSize(.small)
                    Text("resuming session").font(.caption).foregroundStyle(.secondary)
                case .failed(let m):
                    Label(m, systemImage: "exclamationmark.triangle")
                        .font(.caption).foregroundStyle(.red).lineLimit(1)
                case .ready, .generating:
                    if let n = state.engineNote {
                        Label(n, systemImage: "info.circle")
                            .font(.caption).foregroundStyle(.secondary).lineLimit(1)
                    } else if let i = state.serverInfo {
                        Text("\(i.model.name) · \(i.context) ctx · \(i.live_sessions) live")
                            .font(.caption).foregroundStyle(.secondary)
                            .help(i.summary)
                    }
                }
            }
        }
    }
}
