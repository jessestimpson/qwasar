// CrucibleApp.swift -- the M1 window.
//
// PLAN.md 5.1: NavigationSplitView, the shape every macOS user already knows.
// Sidebar of projects and their sessions; the transcript in the middle; the
// composer pinned below it. The inspector is M6 and is deliberately absent
// rather than stubbed.

import SwiftUI
import CrucibleKit

struct CrucibleApp: App {
    @State private var state = AppState()
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate

    var body: some Scene {
        Window("Crucible", id: "main") {
            RootView(state: state)
                .frame(minWidth: 900, minHeight: 620)
                // The delegate is constructed by AppKit before any of this
                // exists, so it is handed the state rather than owning it.
                .onAppear { delegate.state = state }
        }
        .defaultSize(width: 1180, height: 820)
        .commands {
            CommandGroup(after: .newItem) {
                Button("New Project…") { state.addProject() }
                    .keyboardShortcut("n", modifiers: [.command, .shift])
            }
            CommandGroup(after: .appSettings) {
                // Speculation needs a folder the sandbox cannot reach on its
                // own, so it is a grant the user makes once, like the model.
                Button(state.hasDraftHead
                       ? "Replace MTP Draft Head…" : "Add MTP Draft Head…") {
                    state.chooseDraftHead()
                }
                if state.hasDraftHead {
                    Button("Remove MTP Draft Head") { state.forgetDraftHead() }
                }
                Divider()
                // Escalation (spec §15.4): the key is entered here, lands in
                // the Keychain, and is never seen again -- not by the config
                // project, not by any model, not by this UI.
                Button(state.hasAPIKey ? "Replace Escalation API Key…"
                                       : "Set Escalation API Key…") {
                    state.showingAPIKeySheet = true
                }
                if state.hasAPIKey {
                    Button("Remove Escalation API Key") { state.removeAPIKey() }
                }
            }
        }
    }
}

/// Holds termination until the guests have flushed.
///
/// A VM killed along with its host never runs the `sync; poweroff` that its init
/// traps, so whatever is still in the guest's page cache goes with it -- and the
/// guest disk is where the model's work lives. Quitting used to do precisely
/// that, silently, every time.
///
/// `terminateLater` is the only hook macOS gives for asynchronous cleanup on
/// quit: AppKit stops, waits, and resumes when `reply(toApplicationShouldTerminate:)`
/// says so. Bounded by the sandbox's own five-second grace period, so a guest
/// that will not stop cannot wedge the quit.
final class AppDelegate: NSObject, NSApplicationDelegate {
    @MainActor var state: AppState?

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        MainActor.assumeIsolated {
            guard let state, state.needsShutdown else { return .terminateNow }
            Task {
                await state.shutdown()
                NSApp.reply(toApplicationShouldTerminate: true)
            }
            return .terminateLater
        }
    }
}

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

// MARK: - Escalation API key (spec §15.4)

struct APIKeySheet: View {
    @Bindable var state: AppState
    @State private var key = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Escalation API key").font(.headline)
            Text("An OpenRouter (or OpenAI-compatible) key. It is stored in "
                 + "the macOS Keychain and attached to requests by the app "
                 + "alone — no model, tool, or config session can read it.")
                .font(.caption).foregroundStyle(.secondary)
            SecureField("sk-or-…", text: $key)
                .textFieldStyle(.roundedBorder)
            Text("Escalation also needs models granted: in a Crucible Config "
                 + "session, `config_set` the `agent_models` key at the layer "
                 + "you want.")
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
                        SessionRow(session: s, isLive: state.liveSessionID == s.id)
                            .tag(s.id)
                            .contextMenu {
                                Button("Delete Session", role: .destructive) {
                                    state.deleteSession(s.id)
                                }
                            }
                    }
                    Button {
                        state.newSession(in: project)
                    } label: {
                        Label("New Session", systemImage: "plus").font(.caption)
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(.secondary)
                } header: {
                    HStack {
                        Text(project.name)
                        Spacer()
                        if project.isConfig {
                            Image(systemName: "gearshape")
                                .foregroundStyle(.secondary)
                                .help("Built in. Sessions here manage Crucible's "
                                      + "configuration with host-side tools — no "
                                      + "folder, no sandbox.")
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
        .safeAreaInset(edge: .bottom) {
            Button {
                state.addProject()
            } label: {
                Label("Add Project…", systemImage: "folder.badge.plus")
                    .frame(maxWidth: .infinity)
            }
            .padding(8)
        }
    }
}

struct SessionRow: View {
    let session: SessionRecord
    let isLive: Bool

    var body: some View {
        HStack(spacing: 6) {
            // PLAN.md 5.1: filled means the session holds its share of the
            // working set; hollow means it is on disk and the first message
            // will pay to rebuild it.
            Image(systemName: isLive ? "circle.fill" : "circle")
                .font(.system(size: 7))
                .foregroundStyle(isLive ? Color.accentColor : Color.secondary)
            VStack(alignment: .leading, spacing: 1) {
                Text(session.title).lineLimit(1)
                if session.tokenCount > 0 {
                    Text("\(session.tokenCount) / \(session.contextSize) tokens")
                        .font(.caption2).foregroundStyle(.secondary)
                }
            }
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

    private static let specOnHelp =
        "\n\nSpeculative decoding is on: an MTP draft head proposes tokens and "
        + "the model verifies them, so the text is distributed exactly as "
        + "decoding one at a time and takes fewer passes to produce."
    private static let speedUpHelp =
        "Enable speculative decoding: point Crucible at the MTP draft head "
        + "(./download_model.sh mtp-head, or an existing ~/.cache/qwasar/mtp). "
        + "Decoding gets roughly 1.5x faster; the context window shrinks to "
        + "pay for the head's weights and cache. The grant is remembered "
        + "across launches."

    var body: some ToolbarContent {
        ToolbarItem(placement: .status) {
            HStack(spacing: 10) {
                switch state.phase {
                case .needsModel:
                    Button("Choose Model…") { state.chooseModel() }
                case .loading(let what):
                    ProgressView().controlSize(.small)
                    Text(what).font(.caption).foregroundStyle(.secondary)
                case .opening:
                    ProgressView().controlSize(.small)
                    Text("rebuilding session").font(.caption).foregroundStyle(.secondary)
                case .failed(let m):
                    Label(m, systemImage: "exclamationmark.triangle")
                        .font(.caption).foregroundStyle(.red).lineLimit(1)
                case .ready, .generating:
                    if let n = state.engineNote {
                        Label(n, systemImage: "info.circle")
                            .font(.caption).foregroundStyle(.secondary).lineLimit(1)
                    } else if let i = state.engineInfo {
                        let label = "\(i.contextSize) ctx · "
                                  + "\(state.profile.liveSessions) live"
                                  + (i.mtpActive ? " · spec" : "")
                        Text(label)
                            .font(.caption).foregroundStyle(.secondary)
                            .help(state.profile.summary
                                  + (i.mtpActive ? Self.specOnHelp : ""))
                        // Speculation is worth ~1.5x on decode and was buried
                        // in the app menu; a one-time grant deserves a
                        // one-click home where the eye already checks status.
                        // Gone once granted -- the ctx label gains "spec".
                        if !i.mtpActive {
                            Button {
                                state.chooseDraftHead()
                            } label: {
                                Label("Speed Up…", systemImage: "hare")
                                    .font(.caption)
                            }
                            .buttonStyle(.borderless)
                            .help(Self.speedUpHelp)
                        }
                    }
                }
            }
        }
    }
}
