// MarkdownView.swift -- the model's text, rendered.
//
// PLAN.md 5.6. Foundation parses; MarkdownParser reassembles runs into blocks;
// this turns blocks into views. Nothing here parses anything.
//
// The streaming rule lives in `CodeBlock` and is the part worth reading twice.

import SwiftUI
import CrucibleKit
import AppKit

struct MarkdownView: View {
    let source: String
    /// True while this text is still being generated. Only the code blocks care,
    /// and only for the case where the fence named no language.
    var isStreaming: Bool = false

    private var blocks: [MarkdownBlock] { MarkdownParser.blocks(source) }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(Array(blocks.enumerated()), id: \.offset) { _, block in
                row(block)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    @ViewBuilder private func row(_ b: MarkdownBlock) -> some View {
        switch b.kind {
        case .paragraph:
            styled(b.text)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)

        case .heading(let level):
            styled(b.text)
                .font(.system(size: headingSize(level), weight: .semibold))
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.top, 4)

        case .codeBlock(let language):
            CodeBlock(code: b.plain, language: language, isStreaming: isStreaming)

        case .listItem(let ordinal, let depth):
            HStack(alignment: .firstTextBaseline, spacing: 6) {
                Text(ordinal.map { "\($0)." } ?? "•")
                    .font(.body.monospacedDigit())
                    .foregroundStyle(.secondary)
                    .frame(minWidth: 16, alignment: .trailing)
                styled(b.text)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .padding(.leading, CGFloat(depth) * 18)

        case .blockQuote:
            HStack(alignment: .top, spacing: 8) {
                Rectangle().fill(.tertiary).frame(width: 3)
                styled(b.text)
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }

        case .thematicBreak:
            Divider().padding(.vertical, 2)

        case .table(let headerRows):
            TableBlock(rows: b.cells, headerRows: headerRows)
        }
    }

    private func headingSize(_ level: Int) -> CGFloat {
        switch level {
        case 1: return 22
        case 2: return 19
        case 3: return 17
        default: return 15
        }
    }

    /// Inline intents become concrete styling.
    ///
    /// Done explicitly rather than by handing the AttributedString to `Text` and
    /// hoping: `inlinePresentationIntent` is not reliably honoured, and silently
    /// unstyled bold is the kind of thing nobody notices is broken.
    @ViewBuilder private func styled(_ a: AttributedString) -> some View {
        a.runs.reduce(Text("")) { acc, run in
            var t = Text(AttributedString(a[run.range]))
            let intent = run.inlinePresentationIntent ?? []
            if intent.contains(.stronglyEmphasized) { t = t.bold() }
            if intent.contains(.emphasized) { t = t.italic() }
            if intent.contains(.strikethrough) { t = t.strikethrough() }
            if intent.contains(.code) {
                t = t.font(.system(.body, design: .monospaced))
                    .foregroundColor(CodePalette.inlineCode)
            }
            if run.link != nil { t = t.foregroundColor(.accentColor).underline() }
            return acc + t
        }
    }
}

// MARK: - Tables

private struct TableBlock: View {
    let rows: [[AttributedString]]
    let headerRows: Int

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            Grid(alignment: .leading, horizontalSpacing: 14, verticalSpacing: 5) {
                ForEach(Array(rows.enumerated()), id: \.offset) { i, cells in
                    GridRow {
                        ForEach(Array(cells.enumerated()), id: \.offset) { _, cell in
                            Text(cell)
                                .font(i < headerRows ? .body.weight(.semibold) : .body)
                                .textSelection(.enabled)
                        }
                    }
                    if i == headerRows - 1 { Divider().gridCellUnsizedAxes(.horizontal) }
                }
            }
            .padding(8)
        }
        .background(CodePalette.surface, in: .rect(cornerRadius: 6))
    }
}

// MARK: - Code

/// A fenced block, highlighted as it arrives.
///
/// PLAN.md 5.6, measured rather than assumed:
///
///   * **Highlight only up to the last newline.** A half-typed token would flash
///     its colours as it completes (`fun` -> `func`). Excluding the in-progress
///     line also makes the rest stable: across docstrings, block comments,
///     heredocs, multiline strings, sigils and raw strings, not one settled line
///     changed colour as the block grew.
///   * **A complete block ends with a newline**, so the same rule highlights all
///     of it once the fence closes -- no separate "finished" path.
///   * **No language hint means wait.** hljs must guess, and its guess on a
///     fragment is unstable: python -> cpp -> stata on one block, elixir -> ruby
///     on another. A flip recolours everything, so detection runs once, at the
///     end.
///   * Cost is not a constraint: re-highlighting the whole prefix on every line
///     is 0.07%-0.6% of a core, because decode runs at ~6 tok/s. The guard is
///     for absurd blocks only.
private struct CodeBlock: View {
    let code: String
    let language: String?
    let isStreaming: Bool

    /// Past this, per-line re-highlighting stops being free and the block waits
    /// for its close.
    private static let perLineLimit = 2000

    @State private var copied = false

    private var lineCount: Int { code.reduce(0) { $1 == "\n" ? $0 + 1 : $0 } }

    /// Everything up to and including the last newline; the rest is still being
    /// typed. A finished block ends in a newline, so this is all of it.
    private var settled: (done: String, pending: String) {
        guard let last = code.lastIndex(of: "\n") else { return ("", code) }
        let cut = code.index(after: last)
        return (String(code[code.startIndex..<cut]), String(code[cut...]))
    }

    private var highlighted: (runs: [HighlightedRun], trailing: String)? {
        guard lineCount <= Self.perLineLimit || !isStreaming else { return nil }
        let (done, pending) = settled

        if let language, !language.isEmpty {
            guard !done.isEmpty,
                  let runs = Highlighter.shared.highlight(done, language: language) else { return nil }
            return (runs, pending)
        }
        // No hint: detect, but only once the block is finished.
        guard !isStreaming,
              let got = Highlighter.shared.highlightDetecting(code) else { return nil }
        return (got.runs, "")
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header
            ScrollView(.horizontal, showsIndicators: false) {
                body_
                    .font(.system(.callout, design: .monospaced))
                    .textSelection(.enabled)
                    .padding(.horizontal, 10)
                    .padding(.bottom, 8)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .background(CodePalette.surface, in: .rect(cornerRadius: 6))
        .overlay(RoundedRectangle(cornerRadius: 6).strokeBorder(CodePalette.border, lineWidth: 1))
    }

    @ViewBuilder private var body_: some View {
        if let (runs, trailing) = highlighted {
            let coloured = runs.reduce(Text("")) { acc, run in
                acc + Text(run.text).foregroundColor(CodePalette.color(run.kind))
            }
            // The in-progress line, uncoloured until it is a line.
            coloured + Text(trailing).foregroundColor(CodePalette.color(.plain))
        } else {
            Text(code).foregroundColor(CodePalette.color(.plain))
        }
    }

    private var header: some View {
        HStack(spacing: 6) {
            if let language, !language.isEmpty {
                Text(language).font(.caption2.weight(.medium)).foregroundStyle(.secondary)
            }
            Spacer(minLength: 0)
            Button {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(code, forType: .string)
                copied = true
                Task { try? await Task.sleep(for: .seconds(1.4)); copied = false }
            } label: {
                Label(copied ? "Copied" : "Copy",
                      systemImage: copied ? "checkmark" : "doc.on.doc")
                    .font(.caption2)
                    .labelStyle(.titleAndIcon)
            }
            .buttonStyle(.plain)
            .foregroundStyle(.secondary)
            .help("Copy this block")
        }
        .padding(.horizontal, 10)
        .padding(.top, 6)
        .padding(.bottom, 4)
    }
}

// MARK: - Palette

/// Token colours, as semantic pairs rather than one theme.
///
/// hljs ships stylesheets; none of them follow the app into dark mode, so the
/// class names map here instead. Deliberately few distinctions: colouring each
/// of hljs's ~40 classes separately produces a rainbow rather than a reading
/// aid.
enum CodePalette {
    static let surface = Color(nsColor: .textBackgroundColor).opacity(0.55)
    static let border = Color(nsColor: .separatorColor)
    static let inlineCode = Color(nsColor: .systemPink)

    static func color(_ kind: TokenKind) -> Color {
        switch kind {
        case .plain:       return Color(nsColor: .labelColor)
        case .keyword:     return Color(nsColor: .systemPink)
        case .string:      return Color(nsColor: .systemRed)
        case .number:      return Color(nsColor: .systemOrange)
        case .comment:     return Color(nsColor: .secondaryLabelColor)
        case .type:        return Color(nsColor: .systemTeal)
        case .function:    return Color(nsColor: .systemBlue)
        case .variable:    return Color(nsColor: .systemPurple)
        case .constant:    return Color(nsColor: .systemOrange)
        case .operator:    return Color(nsColor: .labelColor).opacity(0.75)
        case .punctuation: return Color(nsColor: .tertiaryLabelColor)
        case .meta:        return Color(nsColor: .systemBrown)
        case .attribute:   return Color(nsColor: .systemIndigo)
        case .tag:         return Color(nsColor: .systemGreen)
        case .link:        return Color(nsColor: .linkColor)
        case .emphasis:    return Color(nsColor: .labelColor)
        }
    }
}
