// ApprovalSheet.swift -- the boundary crossing, with a person in it.
//
// PLAN.md 7.4 step 9. The diff is deliberately plain for now: the point of this
// milestone is that the machinery is correct and that nothing is written
// without an explicit, per-file decision. Making it pretty is a later job and
// changes nothing about whether it is safe.
//
// There is no "apply all and don't ask again". The whole architecture exists to
// put a human at this one point.

import SwiftUI
import CrucibleKit

struct ApprovalSheet: View {
    @Bindable var state: AppState
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()

            if let p = state.proposal {
                if p.changes.isEmpty {
                    empty("The sandbox has made no changes to this project.")
                } else {
                    List {
                        ForEach(p.changes) { change in
                            ChangeRow(change: change,
                                      isSelected: state.approvedPaths.contains(change.path),
                                      toggle: { state.toggleApproval(change) })
                        }
                        if !p.skipped.isEmpty {
                            Section("Not included") {
                                ForEach(p.skipped, id: \.path) { s in
                                    // Stated, never silent. A proposal that
                                    // quietly omitted a file would be the worst
                                    // failure this sheet could have.
                                    Text("\(s.path) — \(s.reason)")
                                        .font(.caption).foregroundStyle(.secondary)
                                }
                            }
                        }
                    }
                    .listStyle(.inset)
                }
            } else if state.isProposing {
                empty("Asking the sandbox what changed…")
            } else if let e = state.proposalError {
                empty(e)
            }

            Divider()
            footer
        }
        .frame(minWidth: 720, minHeight: 460)
    }

    private var header: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text("Review changes").font(.headline)
                Text(state.proposal?.summary ?? "—")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
            if let p = state.proposal, !p.changes.isEmpty {
                Text("^[\(state.approvedPaths.count) file](inflect: true) selected")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
        .padding(14)
    }

    private var footer: some View {
        HStack {
            if let dir = state.lastUndoDirectory {
                Label("previous version saved", systemImage: "clock.arrow.circlepath")
                    .font(.caption).foregroundStyle(.secondary)
                    .help(dir.path)
            }
            Spacer()
            Button("Close") { dismiss() }
                .keyboardShortcut(.cancelAction)
            Button("Apply Selected") {
                Task { await state.applyApproved(); dismiss() }
            }
            .keyboardShortcut(.defaultAction)
            .disabled(state.approvedPaths.isEmpty)
        }
        .padding(14)
    }

    private func empty(_ text: String) -> some View {
        VStack {
            Spacer()
            Text(text).foregroundStyle(.secondary).multilineTextAlignment(.center)
                .padding()
            Spacer()
        }
        .frame(maxWidth: .infinity)
    }
}

private struct ChangeRow: View {
    let change: ProposedChange
    let isSelected: Bool
    let toggle: () -> Void

    @State private var expanded = false

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 10) {
                Toggle("", isOn: Binding(get: { isSelected }, set: { _ in toggle() }))
                    .labelsHidden()
                    .disabled(!change.isApplicable)

                Image(systemName: icon).foregroundStyle(tint).font(.caption)
                Text(change.path).font(.system(.body, design: .monospaced))
                Text(change.status.rawValue).font(.caption).foregroundStyle(.secondary)
                Spacer()
                Button(expanded ? "hide" : "diff") { expanded.toggle() }
                    .font(.caption).buttonStyle(.plain).foregroundStyle(.tint)
            }

            // A rejection is final and says why; a conflict leaves the decision
            // with the person but starts unticked.
            if let r = change.rejection {
                Label("refused: \(r)", systemImage: "hand.raised.fill")
                    .font(.caption).foregroundStyle(.red)
            } else if let c = change.conflict {
                Label(c, systemImage: "exclamationmark.triangle.fill")
                    .font(.caption).foregroundStyle(.orange)
            }

            if expanded {
                ScrollView(.horizontal) {
                    Text(change.diff.isEmpty ? "(no textual diff — binary or new file)"
                                             : change.diff)
                        .font(.system(.caption, design: .monospaced))
                        .textSelection(.enabled)
                }
                .frame(maxHeight: 260)
                .padding(8)
                .background(Color.secondary.opacity(0.07), in: .rect(cornerRadius: 6))
            }
        }
        .padding(.vertical, 4)
        .opacity(change.isApplicable ? 1 : 0.55)
    }

    private var icon: String {
        switch change.status {
        case .added: return "plus.circle"
        case .modified: return "pencil.circle"
        case .deleted: return "minus.circle"
        }
    }

    private var tint: Color {
        switch change.status {
        case .added: return .green
        case .modified: return .accentColor
        case .deleted: return .red
        }
    }
}


// MARK: - The git crossing (spec 7.4a)

/// The result of a crossing: a real branch in the user's real repo. The
/// destructive step -- changing files -- belongs to their own `git merge`,
/// so this sheet informs and hands over rather than asking permission.
struct GitCrossingSheet: View {
    @Bindable var state: AppState

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            if let x = state.gitCrossing {
                Label("Branch updated: \(x.branch)", systemImage: "arrow.triangle.branch")
                    .font(.headline)
                if x.commits.isEmpty && x.objectCount == 0 {
                    Text("Nothing new to cross — the sandbox matches the base.")
                        .foregroundStyle(.secondary)
                } else {
                    Text("\(x.objectCount) objects verified and written; nothing outside "
                         + "`.git/objects` and this one ref was touched. Your files "
                         + "change only when you merge.")
                        .font(.caption).foregroundStyle(.secondary)
                    ScrollView {
                        VStack(alignment: .leading, spacing: 4) {
                            ForEach(Array(x.commits.enumerated()), id: \.offset) { _, c in
                                HStack(alignment: .top, spacing: 6) {
                                    Text(c.sha.prefix(8))
                                        .font(.caption.monospaced()).foregroundStyle(.tertiary)
                                    VStack(alignment: .leading, spacing: 1) {
                                        Text(c.message).font(.caption)
                                        Text(c.author).font(.caption2).foregroundStyle(.tertiary)
                                    }
                                }
                            }
                        }.frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .frame(minHeight: 60, maxHeight: 180)
                    HStack(spacing: 6) {
                        Text("git merge \(x.branch)")
                            .font(.callout.monospaced())
                            .padding(6)
                            .background(RoundedRectangle(cornerRadius: 4).fill(.quaternary.opacity(0.5)))
                        Button {
                            NSPasteboard.general.clearContents()
                            NSPasteboard.general.setString("git merge \(x.branch)", forType: .string)
                        } label: { Image(systemName: "doc.on.doc") }
                        .buttonStyle(.borderless)
                        .help("Copy. Review first with `git log -p \(x.branch)`; conflicts "
                              + "resolve with your own mergetool, line by line.")
                    }
                }
            }
            HStack { Spacer(); Button("Done") { state.showingGitCrossing = false } }
        }
        .padding(16)
        .frame(width: 460)
    }
}

/// Spec 7.4a's session-start question, asked only when the tree is actually
/// dirty: should the agent see your uncommitted work at all?
struct DirtyPromptSheet: View {
    @Bindable var state: AppState

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Label("This project has uncommitted changes", systemImage: "pencil.line")
                .font(.headline)
            Text("The sandbox received a copy of the tree as it stands. Should "
                 + "this session work on your uncommitted changes, or from the "
                 + "last commit?")
                .font(.caption).foregroundStyle(.secondary)
            HStack {
                Spacer()
                Button("From last commit") { state.resolveDirtyChoice("exclude") }
                    .help("The copy is reset to HEAD; your uncommitted work never "
                          + "enters the sandbox.")
                Button("Include my changes") { state.resolveDirtyChoice("include") }
                    .keyboardShortcut(.defaultAction)
                    .help("Your changes are committed first, visibly yours, so a "
                          + "merge never buries them under the agent's name.")
            }
        }
        .padding(16)
        .frame(width: 440)
    }
}
