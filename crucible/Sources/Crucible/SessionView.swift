// SessionView.swift -- the transcript and the composer.
//
// PLAN.md 5.2: items, not a text stream. A tool call is a card, reasoning is a
// fold, and the turn footer states what the turn cost. PLAN.md 5.4: prefill is
// a determinate bar with real numbers, because a cold prompt is the longest
// part of a turn and a spinner during it reads as a hang.

import SwiftUI
import CrucibleKit

struct SessionView: View {
    @Bindable var state: AppState

    var body: some View {
        VStack(spacing: 0) {
            SessionHeader(state: state)
            Divider()
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 16) {
                        ForEach(state.transcript) { item in
                            // Only the tail item can be mid-generation, and
                            // only then is a fence possibly still open.
                            TranscriptRow(item: item,
                                          isStreaming: state.phase == .generating
                                                       && item.id == state.transcript.last?.id)
                                .id(item.id)
                        }
                        if let p = state.pendingCall {
                            PendingCallRow(name: p.name, keys: p.keys, tokens: p.tokens)
                        }
                        if let d = state.liveDelegation {
                            DelegationCard(model: d.model, task: d.task, log: d.log,
                                           costUSD: d.costUSD, ended: d.ended,
                                           waiting: d.waiting, state: state)
                        }
                        Color.clear.frame(height: 1).id("bottom")
                    }
                    .padding(20)
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
                .onChange(of: state.transcript.count) {
                    withAnimation { proxy.scrollTo("bottom", anchor: .bottom) }
                }
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
            Composer(state: state)
        }
        .sheet(isPresented: $state.showingApproval) {
            ApprovalSheet(state: state)
        }
        .sheet(isPresented: $state.showingEscalateSheet) {
            EscalateSheet(state: state)
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
                if state.canEscalate {
                    Button {
                        state.showingEscalateSheet = true
                    } label: {
                        Label(state.phase == .generating ? "Escalate (stops the turn)…"
                                                         : "Escalate…",
                              systemImage: "arrow.up.forward.circle")
                    }
                    .help("Hand this problem to a more capable remote model, under "
                          + "the session's budget. If the local model is mid-turn -- "
                          + "stuck in a bad line of reasoning, say -- escalating "
                          + "interrupts it. The result rides along with your next "
                          + "message.")
                }
                if state.canReviewChanges {
                    Button {
                        state.reviewChanges()
                    } label: {
                        Label("Review Changes…", systemImage: "arrow.left.arrow.right")
                    }
                    .help("See what the sandbox changed, and choose what to apply to your "
                          + "own files. Nothing is written until you approve it.")
                }
                if let status = state.sandboxStatus {
                    Label(status, systemImage: status.hasPrefix("sandboxed")
                          ? "shield.lefthalf.filled" : "eye")
                        .font(.caption)
                        .foregroundStyle(status.hasPrefix("sandboxed") ? .green : .secondary)
                        .help(status.hasPrefix("sandboxed")
                              ? "Tools run on a copy of this folder inside a VM with no network. Nothing they do touches your files."
                              : "Tools can read your files but cannot change anything.")
                }
            }
        }
        .padding(.horizontal, 20).padding(.vertical, 10)
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
              + "tokens generated this turn. This is a dense 27B model; about "
              + "6 tok/s is the serial bandwidth ceiling — higher means "
              + "speculation is paying.")
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
        return state.projects.first { $0.id == rec.projectID }?.effort
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
        state.prefillTotal > 0 && state.prefillDone < state.prefillTotal
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
                    if state.tokensPerSecond > 0 {
                        RateReadout(rate: state.tokensPerSecond,
                                    instantaneous: state.instantaneousTokensPerSecond,
                                    generated: state.generatedThisTurn)
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
                Text("\(what) — about 16 GB, mapped once")
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

    private var resultLines: [Substring] { (result ?? "").split(separator: "\n", omittingEmptySubsequences: false) }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Image(systemName: icon).foregroundStyle(.secondary).font(.caption)
                Text(name).font(.system(.callout, design: .monospaced)).bold()
                ForEach(arguments.sorted(by: { $0.key < $1.key }), id: \.key) { k, v in
                    Text("\(k)=\(oneLine(v))")
                        .font(.system(.caption, design: .monospaced))
                        .foregroundStyle(.secondary).lineLimit(1)
                }
                Spacer()
                if result == nil { ProgressView().controlSize(.small) }
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

    private var icon: String {
        switch name {
        case "read": return "doc.text"
        case "list": return "folder"
        case "grep": return "magnifyingglass"
        default:     return "wrench.and.screwdriver"
        }
    }

    private func oneLine(_ v: String) -> String {
        let first = v.split(separator: "\n", maxSplits: 1).first.map(String.init) ?? v
        return first.count > 52 ? String(first.prefix(52)) + "…" : first
    }
}

// MARK: - Composer

struct Composer: View {
    @Bindable var state: AppState

    var body: some View {
        HStack(alignment: .bottom, spacing: 8) {
            TextField("Ask about this project…", text: $state.draft, axis: .vertical)
                .lineLimit(1...6)
                .textFieldStyle(.plain)
                .padding(8)
                .background(Color.secondary.opacity(0.08), in: .rect(cornerRadius: 8))
                .onSubmit { state.send() }
                .disabled(state.phase == .generating)

            if state.phase == .generating {
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
                    Button("Stop") { state.stopDelegation() }
                        .controlSize(.small)
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
                    Label("open for steering — closes in a few seconds unless you type",
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


// MARK: - Escalate this (spec §15)

/// The user pushes a problem up without waiting for the local model to
/// decide. Consult-only; the answer rides along with the next message.
struct EscalateSheet: View {
    @Bindable var state: AppState
    @State private var task = ""
    @State private var model: String = ""
    @State private var includeContext = true

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Escalate to a remote model").font(.headline)
            if state.phase == .generating {
                Label("the local model is mid-turn; escalating interrupts it",
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
            Toggle("Include the recent conversation — the last message and "
                   + "everything the local model produced since, reasoning included",
                   isOn: $includeContext)
                .font(.caption)
            Picker("Model", selection: $model) {
                ForEach(state.escalationModels, id: \.self) { Text($0).tag($0) }
            }
            .pickerStyle(.menu)
            Text("You can watch and steer it while it works; its answer is "
                 + "attached to your next message so the local model sees it.")
                .font(.caption2).foregroundStyle(.tertiary)
            HStack {
                Spacer()
                Button("Cancel") { state.showingEscalateSheet = false }
                Button("Escalate") {
                    state.startEscalation(task: task, model: model.isEmpty ? nil : model,
                                          includeContext: includeContext)
                    state.showingEscalateSheet = false
                }
                .keyboardShortcut(.defaultAction)
                .disabled(task.trimmingCharacters(in: .whitespaces).isEmpty)
            }
        }
        .padding(16)
        .frame(width: 480)
        .onAppear { model = state.escalationModels.first ?? "" }
    }
}
