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

            // Spans the window, not the detail pane: what the engine is doing
            // is a property of the application, and there is only ever one
            // session doing it (PLAN.md 2.1).
            StatusFooter(state: state)
        }
        .animation(.easeInOut(duration: 0.15), value: state.prefillTotal > 0)
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
                        if project.resolvedRoot == nil {
                            Image(systemName: "exclamationmark.triangle.fill")
                                .foregroundStyle(.orange)
                                .help("This folder is no longer reachable. Re-add the project.")
                        }
                    }
                    .contextMenu {
                        Button("Remove Project", role: .destructive) {
                            state.removeProject(project)
                        }
                    }
                }
            }
        }
        .listStyle(.sidebar)
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
                        Text("\(i.contextSize) ctx · \(state.profile.liveSessions) live"
                             + (i.mtpActive ? " · spec" : ""))
                            .font(.caption).foregroundStyle(.secondary)
                            .help(state.profile.summary
                                  + (i.mtpActive
                                     ? "\n\nSpeculative decoding is on: an MTP draft head "
                                       + "proposes tokens and the model verifies them, so the "
                                       + "text is identical to decoding one at a time and takes "
                                       + "fewer passes to produce."
                                     : "\n\nSpeculative decoding is off. Add an MTP draft head "
                                       + "from the app menu to trade some context for speed."))
                    }
                }
            }
        }
    }
}
