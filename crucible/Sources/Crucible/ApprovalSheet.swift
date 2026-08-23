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
