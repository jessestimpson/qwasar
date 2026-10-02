// SessionView.swift -- the transcript and the composer.
//
// PLAN.md 5.2: items, not a text stream. A tool call is a card, reasoning is a
// fold, and the turn footer states what the turn cost. PLAN.md 5.4: prefill is
// a determinate bar with real numbers, because a cold prompt is the longest
// part of a turn and a spinner during it reads as a hang.

import AppKit
import SwiftUI
import QwasarKit

struct SessionView: View {
    @Bindable var state: AppState

    var body: some View {
        VStack(spacing: 0) {
            SessionHeader(state: state)
            Divider()
            ScrollView {
                // Not lazy.  A LazyVStack estimates the heights of rows it
                // has not built and corrects them as it builds them, so the
                // content height moves whenever the view scrolls -- and
                // following the tail is scrolling.  Estimate, scroll,
                // correct, scroll: a hang report caught that loop pinning
                // the main thread for 108 s with nothing arriving.  Real
                // heights only change when content does.
                VStack(alignment: .leading, spacing: 16) {
                    ForEach(state.transcript) { item in
                        if state.isTurnSelected, let qs = state.pendingQuestions,
                           case .tool(let name, _, nil) = item.kind, name == UserQuestions.toolName {
                            // The model is waiting on the user: its questions,
                            // answerable in place.
                            QuestionCard(questions: qs) { state.answerQuestions($0) }
                                .id(item.id)
                        } else {
                            // Only the tail item can be mid-generation, and
                            // only then is a fence possibly still open.
                            TranscriptRow(item: item,
                                          isStreaming: state.phase == .generating
                                                       && state.isTurnSelected
                                                       && item.id == state.transcript.last?.id)
                                .id(item.id)
                        }
                    }
                    if state.isTurnSelected, let p = state.pendingCall {
                        PendingCallRow(name: p.name, keys: p.keys, tokens: p.tokens)
                    }
                    if let d = state.liveDelegationHere {
                        DelegationCard(model: d.model, task: d.task, log: d.log,
                                       costUSD: d.costUSD, ended: d.ended,
                                       waiting: d.waiting, state: state)
                    }
                }
                .padding(20)
                .frame(maxWidth: .infinity, alignment: .leading)
                // Follows the tail while the view is at the bottom, and
                // leaves it alone once the user scrolls up.  Re-pinned when
                // a session is opened or a message is sent from it.
                .background(BottomFollower(repin: [AnyHashable(state.selectedSessionID),
                                                   AnyHashable(state.sentCount)]))
            }
            Divider()
            if state.pendingHandoff != nil {
                HStack(spacing: 6) {
                    Image(systemName: "arrow.down.forward.circle")
                    Text("the delegation's answer will accompany your next message")
                        .font(.caption)
                    Spacer()
                    Button { state.pendingHandoff = nil } label: {
                        Image(systemName: "xmark.circle.fill")
                    }
                    .buttonStyle(.plain)
                    .help("Discard: the local model will not see the answer")
                }
                .padding(.horizontal, 20).padding(.vertical, 6)
                .foregroundStyle(.secondary)
                .background(Color.accentColor.opacity(0.06))
            }
            if state.offerSuccessor {
                SuccessorBanner(state: state)
            }
            Composer(state: state)
        }
        .sheet(isPresented: $state.showingDelegateSheet) {
            DelegateSheet(state: state)
        }
    }
}

struct SessionHeader: View {
    @Bindable var state: AppState

    var body: some View {
        HStack(spacing: 8) {
            if let s = state.selectedSession {
                VStack(alignment: .leading, spacing: 1) {
                    Text(s.title).font(.headline).lineLimit(1)
                    if let root = state.root(of: s) {
                        Text(root.path).font(.caption).foregroundStyle(.secondary).lineLimit(1)
                    }
                }
                Spacer()
                if state.canDelegate {
                    Button {
                        state.showingDelegateSheet = true
                    } label: {
                        Label(state.phase == .generating ? "Delegate (stops the turn)…"
                                                         : "Delegate…",
                              systemImage: "arrow.up.forward.circle")
                    }
                    .help("Hand this problem to a more capable remote model -- the "
                          + "same delegation the local model can start itself: same "
                          + "tools, same /work, same budget. If the local model is "
                          + "mid-turn -- stuck in a bad line of reasoning, say -- "
                          + "delegating interrupts it. The result also rides along "
                          + "with your next message.")
                }
                if state.canRefreshGit {
                    Button {
                        state.refreshGit()
                    } label: {
                        Label("Refresh Git", systemImage: "arrow.clockwise")
                    }
                    .help("Re-seed the sandbox's private .git copy from this "
                          + "repository's current state. Parks the session; your next "
                          + "message reboots the sandbox (~2s) with fresh history. "
                          + "Your files and your real .git are untouched.")
                }
                if state.notesInProgress == s.id {
                    HStack(spacing: 4) {
                        ProgressView().controlSize(.mini)
                        Text("taking notes…").font(.caption).foregroundStyle(.secondary)
                    }
                    .help("Writing this session's running notes while you read -- off the "
                          + "record, so they cost the conversation nothing. Sending a message "
                          + "stops it at once.")
                }
                if let notes = s.notes, !notes.isEmpty {
                    NotesButton(notes: notes, at: s.notesAt, tokens: s.notesTokens,
                                inherited: s.ancestorID != nil && s.tokenCount == 0)
                }
                if let status = state.sandboxStatus {
                    Label(status, systemImage: status.hasPrefix("sandboxed")
                          ? "shield.lefthalf.filled" : status.hasPrefix("on this Mac") ? "laptopcomputer" : "eye")
                        .font(.caption)
                        .foregroundStyle(status.hasPrefix("sandboxed") ? .green : .secondary)
                        .help(status.hasPrefix("sandboxed")
                              ? "Tools run in a VM with no network, editing this project's working tree directly — your git shows the edits as uncommitted changes, and your real .git is shadowed out of the VM's reach. Review and commit with your own git."
                              : status.hasPrefix("on this Mac")
                                ? "Tools run directly on your Mac, as you, with your login shell's environment."
                                : "Tools can read your files but cannot change anything.")
                }
            }
        }
        .padding(.horizontal, 20).padding(.vertical, 10)
    }
}

/// An Edit as `git diff` shows one: the file, then its lines removed (red,
/// `-`), added (green, `+`) and unchanged around them.
struct EditDiffView: View {
    let path: String
    let old: String
    let new: String
    let replaceAll: Bool

    var body: some View {
        let lines = LineDiff.diff(old: old, new: new)
        let removed = lines.filter { if case .removed = $0 { return true }; return false }.count
        let added = lines.filter { if case .added = $0 { return true }; return false }.count
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 8) {
                Text(path).font(.system(.caption, design: .monospaced)).bold()
                    .lineLimit(1).truncationMode(.head).textSelection(.enabled)
                Text("−\(removed)").font(.caption.monospacedDigit()).foregroundStyle(.red)
                Text("+\(added)").font(.caption.monospacedDigit()).foregroundStyle(.green)
                if replaceAll {
                    Text("every occurrence").font(.caption2).foregroundStyle(.secondary)
                        .padding(.horizontal, 5).padding(.vertical, 1)
                        .background(Color.secondary.opacity(0.12), in: .capsule)
                }
            }
            VStack(alignment: .leading, spacing: 0) {
                ForEach(Array(lines.enumerated()), id: \.offset) { _, line in
                    let (mark, text, tint): (String, String, Color?) = {
                        switch line {
                        case .same(let t):    return (" ", t, nil)
                        case .removed(let t): return ("-", t, .red)
                        case .added(let t):   return ("+", t, .green)
                        }
                    }()
                    HStack(alignment: .top, spacing: 6) {
                        Text(mark).foregroundStyle(tint ?? .secondary)
                        Text(text.isEmpty ? " " : text)
                            .foregroundStyle(tint == nil ? Color.secondary : Color.primary)
                            .fixedSize(horizontal: false, vertical: true)
                        Spacer(minLength: 0)
                    }
                    .font(.system(.caption, design: .monospaced))
                    .padding(.horizontal, 6).padding(.vertical, 1)
                    .background(tint.map { $0.opacity(0.12) } ?? Color.clear)
                }
            }
            .textSelection(.enabled)
            .clipShape(.rect(cornerRadius: 4))
            .overlay(RoundedRectangle(cornerRadius: 4).stroke(Color.secondary.opacity(0.15)))
        }
    }
}

/// The model's questions, answerable in place: Claude Code's question card.
/// Each question has its header chip, its options -- radio buttons, or
/// checkboxes when several may apply -- each with what it means, and an
/// "Other" field for an answer of the user's own.  Skip lets the model go
/// on with its own judgment.
struct QuestionCard: View {
    let questions: [UserQuestion]
    let answer: ([String: [String]]?) -> Void

    @State private var chosen: [String: Set<String>] = [:]
    @State private var other: [String: String] = [:]

    private func answered(_ q: UserQuestion) -> Bool {
        !(chosen[q.question] ?? []).isEmpty
            || !(other[q.question] ?? "").trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Label(questions.count == 1 ? "The model has a question" : "The model has \(questions.count) questions",
                  systemImage: "questionmark.bubble")
                .font(.headline)
            ForEach(questions) { q in
                VStack(alignment: .leading, spacing: 6) {
                    HStack(spacing: 8) {
                        if !q.header.isEmpty {
                            Text(q.header).font(.caption.weight(.semibold))
                                .padding(.horizontal, 7).padding(.vertical, 2)
                                .background(Color.accentColor.opacity(0.15), in: .capsule)
                        }
                        Text(q.question).font(.callout.weight(.medium))
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    ForEach(q.options, id: \.label) { o in
                        let on = chosen[q.question]?.contains(o.label) == true
                        Button {
                            var set = chosen[q.question] ?? []
                            if q.multiSelect {
                                if on { set.remove(o.label) } else { set.insert(o.label) }
                            } else {
                                set = [o.label]
                                other[q.question] = ""
                            }
                            chosen[q.question] = set
                        } label: {
                            HStack(alignment: .firstTextBaseline, spacing: 8) {
                                Image(systemName: q.multiSelect ? (on ? "checkmark.square.fill" : "square")
                                                                : (on ? "largecircle.fill.circle" : "circle"))
                                    .foregroundStyle(on ? Color.accentColor : .secondary)
                                VStack(alignment: .leading, spacing: 1) {
                                    Text(o.label).font(.callout)
                                    if !o.description.isEmpty {
                                        Text(o.description).font(.caption).foregroundStyle(.secondary)
                                            .fixedSize(horizontal: false, vertical: true)
                                    }
                                }
                                Spacer(minLength: 0)
                            }
                            .padding(.vertical, 5).padding(.horizontal, 8)
                            .background(on ? Color.accentColor.opacity(0.08) : Color.clear, in: .rect(cornerRadius: 6))
                            .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                    }
                    HStack(spacing: 8) {
                        Image(systemName: "pencil").foregroundStyle(.secondary)
                        TextField("Other…", text: Binding(
                            get: { other[q.question] ?? "" },
                            set: { v in
                                other[q.question] = v
                                if !q.multiSelect, !v.isEmpty { chosen[q.question] = [] }
                            }))
                            .textFieldStyle(.roundedBorder)
                    }
                    .padding(.horizontal, 8)
                }
            }
            HStack {
                Spacer()
                Button("Skip") { answer(nil) }
                    .help("Let the model go on with its own judgment; it states its assumptions.")
                Button("Answer") {
                    var out: [String: [String]] = [:]
                    for q in questions {
                        var a = q.options.map(\.label).filter { chosen[q.question]?.contains($0) == true }
                        let o = (other[q.question] ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
                        if !o.isEmpty { a.append(o) }
                        out[q.question] = a
                    }
                    answer(out)
                }
                .keyboardShortcut(.defaultAction)
                .disabled(!questions.allSatisfy(answered))
            }
        }
        .padding(14)
        .background(Color.accentColor.opacity(0.05), in: .rect(cornerRadius: 10))
        .overlay(RoundedRectangle(cornerRadius: 10).stroke(Color.accentColor.opacity(0.35)))
    }
}

/// Offered at 85% of the window, and when it is full: continue the work in a
/// new session, seeded with this one's notes (spec 2.4).
struct SuccessorBanner: View {
    @Bindable var state: AppState

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: "arrow.triangle.branch")
            Text(state.contextFill >= 1 || state.selectedSession.map { s in
                    state.transcript.contains { if case .contextFull = $0.kind { return true }; return false }
                 } == true
                 ? "The context is full."
                 : "The context is \(Int(state.contextFill * 100))% used.")
                .font(.caption)
            Spacer()
            Button("Continue in a New Session") {
                if let id = state.selectedSessionID { state.continueInNewSession(id) }
            }
            .disabled(state.phase == .generating || state.notesInProgress != nil)
            .help("Brings this session's notes up to date, opens a new session seeded "
                  + "with them, and unloads this one. It stays readable in the sidebar.")
        }
        .padding(.horizontal, 20).padding(.vertical, 6)
        .foregroundStyle(.orange)
        .background(Color.orange.opacity(0.07))
    }
}

/// The session's running notes, in a popover: what a successor would start
/// from, as of when they were last written.
struct NotesButton: View {
    let notes: String
    let at: Date?
    let tokens: Int?
    let inherited: Bool
    @State private var showing = false

    var body: some View {
        Button { showing.toggle() } label: {
            Label("Notes", systemImage: "note.text")
        }
        .help(inherited ? "The notes this session was started from."
                        : "This session's running notes -- what a new session would start from.")
        .popover(isPresented: $showing, arrowEdge: .bottom) {
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text(inherited ? "Carried over from the earlier session" : "Running notes")
                        .font(.headline)
                    Spacer()
                    Button {
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(notes, forType: .string)
                    } label: { Image(systemName: "doc.on.doc") }
                    .buttonStyle(.borderless).help("Copy")
                }
                if let at, !inherited {
                    Text("written \(at.formatted(.relative(presentation: .named)))"
                         + (tokens.map { ", covering \($0) tokens" } ?? ""))
                        .font(.caption).foregroundStyle(.secondary)
                }
                ScrollView {
                    MarkdownView(source: notes)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .frame(width: 520, height: 420)
            }
            .padding(14)
        }
    }
}

/// A full context is the end of a session (PLAN.md 2.4), so how close it is
/// belongs on screen rather than in a log.
/// Decode rate while a turn is in flight.
///
/// Present only while something is generating. A rate that lingers after a turn
/// is a number about the past pretending to be about the present.
struct RateReadout: View {
    /// The turn's average.
    let rate: Double
    /// Over roughly the last second. Shown first because it is the number
    /// that answers "what is it doing NOW" -- the two diverge whenever the
    /// decode regime changes, sampled reasoning versus speculative answer.
    let instantaneous: Double
    let generated: Int
    /// What speed to expect from the model that is running -- ~6 tok/s and
    /// ~68 tok/s are both healthy, for different models.
    let speedNote: String

    var body: some View {
        HStack(spacing: 4) {
            Image(systemName: "gauge.with.needle").font(.caption2)
            Text(String(format: "%.1f tok/s", instantaneous))
                .font(.caption).monospacedDigit()
            Text(String(format: "· %.1f avg", rate))
                .font(.caption).monospacedDigit().foregroundStyle(.tertiary)
            Text("· \(generated)")
                .font(.caption).monospacedDigit().foregroundStyle(.tertiary)
        }
        .foregroundStyle(.secondary)
        .help("Current decode rate, then the turn's average, then \(generated) "
              + "tokens generated this turn. " + speedNote)
    }
}

/// The reasoning effort this session is running at, and where it can be changed.
///
/// Changing it is free before the first message and impossible after, because
/// effort rewrites the system turn and the system turn is the prefix of
/// everything (PLAN.md 2.2). Rather than offer a control that silently costs a
/// full re-prefill, the menu is disabled once the session has evaluated
/// anything -- and says why -- while still setting the project's default so the
/// next session starts where you left it.
struct EffortControl: View {
    @Bindable var state: AppState

    private var current: ReasoningEffort { state.selectedSession?.effort ?? .medium }

    /// What a NEW session in this project would start at.
    ///
    /// Shown because choosing an effort sets it, and until this existed that
    /// was invisible: the checkmark tracks the session, so a change made on a
    /// session that had already started altered the project and displayed
    /// nothing at all. One pick, and every later session in the project
    /// silently started somewhere else.
    private var projectDefault: ReasoningEffort? {
        guard let rec = state.selectedSession else { return nil }
        return state.projects.first { $0.id == rec.projectID }.map(state.defaultEffort(for:))
    }

    var body: some View {
        if state.selectedSession != nil {
            Menu {
                ForEach(ReasoningEffort.allCases, id: \.self) { e in
                    Button {
                        state.setEffort(e)
                    } label: {
                        // Two different facts, so two different marks: the tick
                        // is what THIS session is running at and cannot change
                        // once it has started; the note is what the next one
                        // will start at.
                        if e == projectDefault, e != current {
                            Text("\(e.label) — default for new sessions")
                        } else if e == projectDefault {
                            Label("\(e.label) — default for new sessions",
                                  systemImage: "checkmark")
                        } else if e == current {
                            Label(e.label, systemImage: "checkmark")
                        } else {
                            Text(e.label)
                        }
                    }
                }
            } label: {
                HStack(spacing: 4) {
                    Image(systemName: "brain").font(.caption2)
                    Text("effort \(current.label)").font(.caption)
                }
                .foregroundStyle(.secondary)
            }
            .menuStyle(.borderlessButton)
            .fixedSize()
            // Both branches say that the choice is sticky, because it is in
            // both. The changeable branch used to say only "changing it now is
            // free", which is true of the SESSION and quietly untrue of the
            // project -- one pick on a fresh session set every later session's
            // starting effort, with nothing on screen to say so.
            .help(state.canChangeEffort
                  ? "How long this model reasons before answering. Free to change now, "
                    + "because this session has not evaluated anything yet — and it also "
                    + "becomes the default for new sessions in this project."
                  : "This session is running at \(current.label). Effort rewrites the system "
                    + "prompt, which every later turn is built on, so it is fixed once a "
                    + "session starts. Choosing another sets the default for the next one, "
                    + "and leaves this session where it is.")
        }
    }
}

struct ContextMeter: View {
    let used: Int
    let limit: Int

    private var fraction: Double { limit > 0 ? Double(used) / Double(limit) : 0 }

    // Amber past 85%: a full window is the END of a session (PLAN.md 2.4), not
    // a degradation, so the warning has to arrive while there is still room to
    // act on it.
    private var tint: Color {
        switch fraction {
        case ..<0.85: return .accentColor
        case ..<0.95: return .orange
        default: return .red
        }
    }

    var body: some View {
        HStack(spacing: 6) {
            Text("context")
                .font(.caption).foregroundStyle(.secondary)
            ProgressView(value: min(fraction, 1))
                .frame(width: 90)
                .tint(tint)
            Text("\(Int((fraction * 100).rounded()))%")
                .font(.caption).monospacedDigit().foregroundStyle(.secondary)
                .frame(minWidth: 30, alignment: .trailing)
            Text("\(used) / \(limit)")
                .font(.caption).monospacedDigit().foregroundStyle(.tertiary)
        }
        .help("\(used) of \(limit) tokens used. A full context ends the session; "
              + "a successor session carries a summary forward.")
    }
}

/// PLAN.md 5.4: a determinate bar with real numbers and an ETA, not a spinner.
///
/// It lives in the window's footer rather than in the transcript. Prefill is a
/// property of what the engine is doing, not an event in the conversation:
/// putting it inline meant it scrolled with the text, appeared between a tool
/// call and its result, and left a permanent artefact in the log of a finished
/// turn. A footer is where a status that comes and goes belongs.
struct PrefillBar: View {
    let done: Int
    let total: Int
    /// What this prefill is for. A turn is reading the prompt; reopening a
    /// session is replaying one. Same work, same bar, and the wait is worth
    /// naming differently because the reasons a person waits are different.
    var label: String = "reading the prompt"

    /// Measured on the target host (PLAN.md 2.5). Used only to estimate the
    /// wait; nothing depends on it being exact.
    private static let tokensPerSecond = 32.0

    private var remaining: Int { max(0, total - done) }

    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: "text.book.closed")
                .font(.caption)
                .foregroundStyle(.secondary)
            Text(label)
                .font(.caption)
                .foregroundStyle(.secondary)
            ProgressView(value: Double(done), total: Double(max(total, 1)))
                .frame(maxWidth: 220)
            Text("\(done) / \(total)\(eta)")
                .font(.caption).monospacedDigit()
                .foregroundStyle(.secondary)
        }
    }

    private var eta: String {
        guard remaining > 60 else { return "" }
        let s = Double(remaining) / Self.tokensPerSecond
        return s < 90 ? String(format: " · about %.0fs left", s)
                      : String(format: " · about %.0f min left", s / 60)
    }
}

/// The window's footer: what the engine is doing, when it is doing something.
///
/// Deliberately absent when there is nothing to say. A status bar that is always
/// there is a status bar nobody reads, and the two things worth interrupting a
/// reader for -- a long prompt being read, and a session being rebuilt -- are
/// both transient.
struct StatusFooter: View {
    @Bindable var state: AppState

    /// Absent when there is nothing to say. A status bar that is always there
    /// is a status bar nobody reads, and the two things worth interrupting a
    /// reader for -- a long prompt being read, and a session being rebuilt --
    /// are both transient.
    private var isPrefilling: Bool {
        state.isTurnSelected && state.prefillTotal > 0 && state.prefillDone < state.prefillTotal
    }

    private var hasContext: Bool { state.contextLimit > 0 && state.contextUsed > 0 }

    private var isVisible: Bool {
        switch state.phase {
        case .opening, .loading: return true
        default: return isPrefilling || hasContext
        }
    }

    var body: some View {
        if isVisible {
            VStack(spacing: 0) {
                Divider()
                HStack(spacing: 16) {
                    content
                    Spacer(minLength: 12)
                    if state.isTurnSelected, state.tokensPerSecond > 0 {
                        RateReadout(rate: state.tokensPerSecond,
                                    instantaneous: state.instantaneousTokensPerSecond,
                                    generated: state.generatedThisTurn,
                                    speedNote: state.speedNote)
                    }
                    EffortControl(state: state)
                    // Right-aligned and persistent, because it is state rather
                    // than an event: the transient messages come and go on the
                    // left, and this stays put so the eye knows where to find it.
                    if hasContext {
                        ContextMeter(used: state.contextUsed, limit: state.contextLimit)
                    }
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 6)
            }
            .background(.bar)
            .transition(.move(edge: .bottom).combined(with: .opacity))
        }
    }

    @ViewBuilder private var content: some View {
        switch state.phase {
        case .opening:
            // A rebuild IS a prefill -- `restore` reads whatever the checkpoint
            // covers and then evaluates the remainder, which is the same work,
            // reported through the same events. So once it starts reporting,
            // show the bar rather than a spinner that says nothing about how
            // long this will take.
            //
            // The spinner still has a job: it covers the part before the first
            // report -- the sandbox boot and the checkpoint read, neither of
            // which has a denominator -- and the case where the checkpoint
            // covered everything and there is nothing left to prefill.
            if isPrefilling {
                PrefillBar(done: state.prefillDone, total: state.prefillTotal,
                           label: "rebuilding this session")
            } else {
                HStack(spacing: 10) {
                    ProgressView().controlSize(.small)
                    Text("rebuilding this session — restoring what it already evaluated")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }

        case .loading(let what):
            HStack(spacing: 10) {
                ProgressView().controlSize(.small)
                Text(what)
                    .font(.caption).foregroundStyle(.secondary)
            }

        default:
            if isPrefilling {
                PrefillBar(done: state.prefillDone, total: state.prefillTotal)
            } else {
                EmptyView()
            }
        }
    }
}

// MARK: - Rows

/// Reasoning, collapsed by default.
///
/// Hand-rolled rather than a `DisclosureGroup`, which on macOS reserves
/// vertical padding under its label whether or not it is open. Stacked on the
/// transcript's own 16pt row spacing that left a caption-height row sitting in
/// roughly twice its own height of blank space, every turn -- and reasoning
/// appears before nearly every assistant message, so it read as a gap rather
/// than as a heading.
///
/// Collapsed, this is exactly the label. Open, the body is 6pt under it.
private struct ReasoningBlock: View {
    let text: String
    let tokens: Int?
    @State private var expanded = false

    var body: some View {
        VStack(alignment: .leading, spacing: expanded ? 6 : 0) {
            Button {
                withAnimation(.easeInOut(duration: 0.15)) { expanded.toggle() }
            } label: {
                HStack(spacing: 4) {
                    Image(systemName: "chevron.right")
                        .font(.caption2)
                        .rotationEffect(.degrees(expanded ? 90 : 0))
                    // Tokens, not characters: tokens are what this cost, in
                    // time and in context. Characters are an artefact of the
                    // encoding. Older transcripts have no count and say so by
                    // omission.
                    Label(tokens.map { "reasoning · \($0) tokens" } ?? "reasoning",
                          systemImage: "brain")
                }
                .font(.caption)
                .foregroundStyle(.secondary)
                .contentShape(.rect)     // the whole strip is the hit target
            }
            .buttonStyle(.plain)

            if expanded {
                Text(text)
                    .font(.system(.callout, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
    }
}

struct TranscriptRow: View {
    let item: TranscriptItem
    /// True only for the item currently being generated. Code blocks use it to
    /// decide whether an unhinted fence can be language-detected yet.
    var isStreaming: Bool = false

    var body: some View {
        switch item.kind {
        case .user(let t):
            HStack {
                Spacer(minLength: 60)
                Text(t)
                    .textSelection(.enabled)
                    .padding(.horizontal, 12).padding(.vertical, 8)
                    .background(Color.accentColor.opacity(0.14), in: .rect(cornerRadius: 10))
            }

        case .assistant(let t):
            // Markdown, because the model writes markdown (PLAN.md 5.6). User
            // turns, reasoning and tool results deliberately do not get this:
            // each would be claiming a formatting intent that is not in the
            // source.
            MarkdownView(source: t, isStreaming: isStreaming)

        case .reasoning(let t):
            // The model always reasons, so this is a first-class item with its
            // own affordance rather than an oddity to hide (PLAN.md 5.2).
            ReasoningBlock(text: t, tokens: item.tokens)

        case .tool(let name, let args, let result):
            ToolCard(name: name, arguments: args, result: result)

        case .note(let n):
            Label(n, systemImage: "info.circle")
                .font(.caption).foregroundStyle(.secondary)

        case .contextFull(let used, let limit):
            Label("Context is full (\(used)/\(limit)). This session cannot continue — "
                  + "a successor session carrying a summary is the way forward.",
                  systemImage: "exclamationmark.octagon")
                .font(.callout).foregroundStyle(.orange)

        case .footer(let s):
            Text(footerText(s))
                .font(.caption).monospacedDigit().foregroundStyle(.tertiary)

        case .delegation(let model, let task, let log, let cost, let ended):
            DelegationCard(model: model, task: task, log: log,
                           costUSD: cost, ended: ended, state: nil)
        }
    }

    private func footerText(_ s: TurnStats) -> String {
        var parts: [String] = [
            "\(s.generatedTokens) tokens",
            "\(s.reasoningTokens) reasoning",
            String(format: "%.1f tok/s", s.tokensPerSecond),
            String(format: "prefill %d tok in %.1fs (%.0f tok/s)",
                   s.promptTokens, s.prefillSeconds, s.prefillTokensPerSecond),
        ]
        // Only when a head is loaded and it actually ran: "1.0 tok/round" on a
        // serial turn would be noise, not information.
        if s.specRounds > 0 {
            parts.append(String(format: "spec %.2f tok/round over %d",
                                s.tokensPerRound, s.specRounds))
        }
        if s.toolCalls > 0 { parts.append("\(s.toolCalls) tool calls") }
        if s.interrupted { parts.append("interrupted") }
        if s.hitBudget { parts.append("budget reached") }
        if s.hitStepCap { parts.append("step cap") }
        parts.append("ctx \(s.contextUsed)/\(s.contextLimit)")
        return parts.joined(separator: " · ")
    }
}

/// A call the model is still writing.
///
/// The markup of a call is never echoed -- it would bury whatever narration came
/// before it -- and a `write` or a `define` runs to hundreds of tokens, which at
/// ~6 tok/s is minutes. Without this the transcript shows nothing at all for the
/// longest stretch of many turns, and a person cannot tell a working model from
/// a wedged one.
///
/// It says only what can be known early and honestly: the function name and the
/// parameter keys, which arrive in the first few tokens, and a count that keeps
/// moving. The values are not shown, because the finished ToolCard shows them
/// and showing them twice would be worse than showing them once.
struct PendingCallRow: View {
    let name: String?
    let keys: [String]
    let tokens: Int

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            ProgressView().controlSize(.small).scaleEffect(0.7)
            Text(name ?? "tool call")
                .font(.callout.weight(.medium).monospaced())
                .foregroundStyle(name == nil ? .secondary : .primary)
            if !keys.isEmpty {
                Text(keys.joined(separator: " · "))
                    .font(.caption).foregroundStyle(.secondary)
            }
            Spacer(minLength: 8)
            Text("\(tokens) tokens")
                .font(.caption.monospacedDigit()).foregroundStyle(.tertiary)
        }
        .padding(.horizontal, 12).padding(.vertical, 8)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.accentColor.opacity(0.07), in: .rect(cornerRadius: 8))
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .strokeBorder(Color.accentColor.opacity(0.22), lineWidth: 1))
    }
}

/// PLAN.md 5.2: a card, with long values elided to one line and the result
/// collapsed past three lines -- the same cut the C agent's TOOL_RESULT_LINES
/// makes, for the same reason.
struct ToolCard: View {
    let name: String
    let arguments: [String: String]
    let result: String?

    @State private var expanded = false
    /// The call in full: every argument, whole, selectable.
    @State private var showCall = false

    private var resultLines: [Substring] { (result ?? "").split(separator: "\n", omittingEmptySubsequences: false) }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            // The model's own words for what the call is for, when it gave
            // some (Bash's `description`), are the headline: "Bash · List
            // the project's files" reads at a glance where the command does
            // not.  The arguments follow, quieter, on their own line.
            HStack(spacing: 8) {
                Image(systemName: icon).foregroundStyle(.secondary).font(.caption)
                Text(name).font(.system(.callout, design: .monospaced)).bold()
                if let purpose {
                    Text("·").foregroundStyle(.tertiary)
                    Text(purpose).font(.callout).lineLimit(2)
                } else if !showCall {
                    argumentLine
                        .contentShape(Rectangle())
                        .onTapGesture { showCall = true }
                }
                Spacer()
                if result == nil { ProgressView().controlSize(.small) }
                if !shownArguments.isEmpty {
                    Button { showCall.toggle() } label: {
                        Image(systemName: showCall ? "chevron.up" : "chevron.down")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    .buttonStyle(.plain)
                    .help(showCall ? "Hide the call" : "Show the whole call")
                }
            }
            if showCall, name == "Edit", let old = arguments["old_string"] ?? arguments["old"],
               let new = arguments["new_string"] ?? arguments["new"] {
                EditDiffView(path: arguments["file_path"] ?? arguments["path"] ?? "",
                             old: old, new: new,
                             replaceAll: arguments["replace_all"] == "true")
                    .padding(.leading, 22)
            } else if showCall {
                VStack(alignment: .leading, spacing: 6) {
                    ForEach(shownArguments, id: \.key) { k, v in
                        VStack(alignment: .leading, spacing: 2) {
                            if shownArguments.count > 1 || k != "command" {
                                Text(k).font(.caption2).foregroundStyle(.tertiary)
                            }
                            Text(v)
                                .font(.system(.caption, design: .monospaced))
                                .textSelection(.enabled)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .padding(6)
                                .background(Color.secondary.opacity(0.08), in: .rect(cornerRadius: 4))
                        }
                    }
                }
                .padding(.leading, 22)
            } else if purpose != nil, !shownArguments.isEmpty {
                argumentLine.padding(.leading, 22)
                    .contentShape(Rectangle())
                    .onTapGesture { showCall = true }
            }
            if let result, !result.isEmpty {
                Text(expanded ? result : resultLines.prefix(3).joined(separator: "\n"))
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                if resultLines.count > 3 {
                    Button(expanded ? "show less" : "\(resultLines.count - 3) more lines") {
                        expanded.toggle()
                    }
                    .font(.caption).buttonStyle(.plain).foregroundStyle(.tint)
                }
            }
        }
        .padding(10)
        .background(Color.secondary.opacity(0.07), in: .rect(cornerRadius: 8))
    }

    /// What the call is for, in the model's words, if it said.
    private var purpose: String? {
        if name == UserQuestions.toolName {
            let n = UserQuestions.parse(ToolCall(name: name, arguments: arguments)).map(\.count)
            if case .success(let k) = n { return k == 1 ? "asked a question" : "asked \(k) questions" }
        }
        guard let d = arguments["description"]?.trimmingCharacters(in: .whitespacesAndNewlines),
              !d.isEmpty else { return nil }
        return d
    }

    /// Every argument but the description, which has its own place.
    private var shownArguments: [(key: String, value: String)] {
        arguments.filter { $0.key != "description" }.sorted { $0.key < $1.key }
    }

    private var argumentLine: some View {
        HStack(spacing: 8) {
            ForEach(shownArguments, id: \.key) { k, v in
                Text("\(k)=\(oneLine(v))")
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(.secondary).lineLimit(1)
            }
        }
    }

    private var icon: String {
        switch name {
        case "Read": return "doc.text"
        case "Write", "Edit": return "pencil"
        case "Glob": return "folder"
        case "Grep": return "magnifyingglass"
        case "Bash": return "terminal"
        case "TodoWrite": return "checklist"
        case UserQuestions.toolName: return "questionmark.bubble"
        default:     return "wrench.and.screwdriver"
        }
    }

    private func oneLine(_ v: String) -> String {
        let first = v.split(separator: "\n", maxSplits: 1).first.map(String.init) ?? v
        return first.count > 52 ? String(first.prefix(52)) + "…" : first
    }
}

// MARK: - Composer

/// Shift-Return in the composer is a new line, where Return sends.
///
/// The field's own newline is Option-Return; Shift-Return is what chat boxes
/// have taught everyone.  SwiftUI's onKeyPress does not reliably see Return
/// while a TextField's field editor has it, so this watches the window's key
/// events instead -- only while the composer has focus -- and hands the field
/// editor the same action Option-Return would, which inserts at the cursor.
@MainActor
final class ShiftReturnNewline {
    var active = false
    private var monitor: Any?

    func install() {
        guard monitor == nil else { return }
        monitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [weak self] e in
            guard let self, self.active,
                  e.keyCode == 36 || e.keyCode == 76,               // Return, keypad Enter
                  e.modifierFlags.intersection(.deviceIndependentFlagsMask) == .shift,
                  let editor = e.window?.firstResponder as? NSTextView, editor.isFieldEditor
            else { return e }
            editor.insertNewlineIgnoringFieldEditor(nil)
            return nil
        }
    }

    func remove() {
        if let monitor { NSEvent.removeMonitor(monitor) }
        monitor = nil
    }
}

struct Composer: View {
    @Bindable var state: AppState
    @FocusState private var focused: Bool
    @State private var shiftReturn = ShiftReturnNewline()

    var body: some View {
        HStack(alignment: .bottom, spacing: 8) {
            TextField("Ask about this project…", text: $state.draft, axis: .vertical)
                .lineLimit(1...6)
                .textFieldStyle(.plain)
                .padding(8)
                .background(Color.secondary.opacity(0.08), in: .rect(cornerRadius: 8))
                .onSubmit { state.send() }
                .focused($focused)
                .onChange(of: focused) { _, f in shiftReturn.active = f }
                .onAppear { shiftReturn.install() }
                .onDisappear { shiftReturn.remove() }
                .disabled(state.phase == .generating)

            if state.phase == .generating, !state.isTurnSelected {
                // One engine, one turn: this session waits for the other.
                Text("busy with “\(state.turnSessionTitle ?? "another session")”")
                    .font(.caption).foregroundStyle(.secondary)
                    .help("The model is working in another session. Select it to watch "
                          + "or stop the turn; this one can send when it finishes.")
            } else if state.phase == .generating {
                Button(role: .destructive) {
                    state.interrupt()
                } label: {
                    Label("Stop", systemImage: "stop.fill")
                }
                .keyboardShortcut(".", modifiers: .command)
            } else {
                Button {
                    state.send()
                } label: {
                    Label("Send", systemImage: "arrow.up.circle.fill")
                }
                .keyboardShortcut(.return, modifiers: .command)
                .disabled(state.phase != .ready || state.draft.trimmingCharacters(in: .whitespaces).isEmpty)
            }
        }
        .padding(12)
    }
}


// MARK: - Delegation (spec §15.2)

/// The embedded sub-session: a transcript item with an inside. Live (state
/// non-nil) it streams, meters cost, and takes input -- the paid model is
/// steerable by the person paying. Completed, it renders the same card from
/// the transcript record, read-only.
struct DelegationCard: View {
    let model: String
    let task: String
    let log: String
    let costUSD: Double
    let ended: String?
    var waiting = false
    let state: AppState?

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 6) {
                Image(systemName: "arrow.up.forward.circle")
                    .foregroundStyle(ended == nil ? Color.accentColor : Color.secondary)
                Text(model).font(.caption.bold())
                if ended == nil { ProgressView().controlSize(.mini) }
                Spacer()
                Text(String(format: "$%.4f", costUSD))
                    .font(.caption).monospacedDigit()
                    .foregroundStyle(.secondary)
                    .help("What this delegation has cost so far. The budget "
                          + "stops the next request, never the one in flight.")
                if ended == nil, let state {
                    Button("Finish") { state.stopDelegation() }
                        .controlSize(.small)
                        .help("End the delegation now. Without a handoff from the "
                              + "delegate, its last message stands in for one.")
                }
            }
            Text(task).font(.caption).foregroundStyle(.secondary).lineLimit(3)
            Divider()
            if log.isEmpty && ended == nil {
                Text("waiting for the remote model…")
                    .font(.caption).foregroundStyle(.tertiary)
            } else {
                MarkdownView(source: log, isStreaming: ended == nil)
            }
            if let ended {
                Label(ended, systemImage: "flag.checkered")
                    .font(.caption2).foregroundStyle(.tertiary)
            } else if let state {
                if waiting {
                    Label("the delegate is waiting on you — reply below, or Finish to "
                          + "end the delegation",
                          systemImage: "hourglass")
                        .font(.caption2).foregroundStyle(.orange)
                }
                // The input INTO the delegation. Bound to its own draft, so
                // the composer below stays the local model's.
                HStack(spacing: 6) {
                    TextField("steer the remote model…",
                              text: Binding(get: { state.delegationDraft },
                                            set: { state.delegationDraft = $0
                                                   state.delegationTyping() }))
                        .textFieldStyle(.roundedBorder)
                        .font(.caption)
                        .onSubmit { state.sendToDelegation() }
                    Button("Send") { state.sendToDelegation() }
                        .controlSize(.small)
                        .disabled(state.delegationDraft.trimmingCharacters(in: .whitespaces).isEmpty)
                }
            }
        }
        .padding(10)
        .background(RoundedRectangle(cornerRadius: 8).fill(Color.accentColor.opacity(0.06)))
        .overlay(RoundedRectangle(cornerRadius: 8)
            .stroke(ended == nil ? Color.accentColor.opacity(0.4) : Color.secondary.opacity(0.2)))
    }
}


// MARK: - Delegate (spec §15)

/// The user delegates without waiting for the local model to decide --
/// IDENTICAL to the model's own delegate tool: same remote agent, same
/// sandboxed tools, same /work, same budget. The answer additionally rides
/// along with the user's next message so the local model sees it.
struct DelegateSheet: View {
    @Bindable var state: AppState
    /// Prefilled so the common case is two clicks: with the conversation
    /// tail included by default, "continue" plus the context IS the brief --
    /// the remote model picks up exactly where the local one is stuck.
    @State private var task = "Continue with the task at hand."
    @State private var model: String = ""
    @State private var includeContext = true

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Delegate to a remote model").font(.headline)
            if state.phase == .generating {
                Label("the local model is mid-turn; delegating interrupts it",
                      systemImage: "exclamationmark.triangle")
                    .font(.caption).foregroundStyle(.orange)
            }
            TextEditor(text: $task)
                .font(.body)
                .frame(minHeight: 90)
                .overlay(RoundedRectangle(cornerRadius: 4).stroke(.quaternary))
                .overlay(alignment: .topLeading) {
                    if task.isEmpty {
                        Text("What should the remote model figure out?")
                            .foregroundStyle(.tertiary).padding(6)
                            .allowsHitTesting(false)
                    }
                }
            Text("Replace the default with specifics when you have them; with "
                 + "the conversation included below, \"continue\" is often enough.")
                .font(.caption2).foregroundStyle(.tertiary)
            Toggle("Include the recent conversation — the last message and "
                   + "everything the local model produced since, reasoning included",
                   isOn: $includeContext)
                .font(.caption)
            Picker("Model", selection: $model) {
                ForEach(state.delegationModels, id: \.self) { Text($0).tag($0) }
            }
            .pickerStyle(.menu)
            Text("The remote model works with the session's own tools on the "
                 + "same /work — watch and steer it as it goes. Its answer is "
                 + "also attached to your next message so the local model sees it.")
                .font(.caption2).foregroundStyle(.tertiary)
            HStack {
                Spacer()
                Button("Cancel") { state.showingDelegateSheet = false }
                Button("Delegate") {
                    state.startDelegation(task: task, model: model.isEmpty ? nil : model,
                                          includeContext: includeContext)
                    state.showingDelegateSheet = false
                }
                .keyboardShortcut(.defaultAction)
                .disabled(task.trimmingCharacters(in: .whitespaces).isEmpty)
            }
        }
        .padding(16)
        .frame(width: 480)
        .onAppear { model = state.delegationModels.first ?? "" }
    }
}
