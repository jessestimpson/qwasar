// Markdown.swift -- the model's text, as blocks.
//
// PLAN.md 5.6. No dependency: Foundation parses CommonMark *and* GFM, and
// `interpretedSyntax: .full` returns the block structure as `presentationIntent`
// attributes on the runs. The work is not parsing, it is REASSEMBLY -- inline
// formatting splits one paragraph into a run per span, and the only way back is
// the identity carried on each run's innermost intent.
//
// No SwiftUI here on purpose: this is where the fixtures point, and a suite that
// has to stand up a view to check that a fenced block kept its language hint is a
// suite nobody runs.

import Foundation

public struct MarkdownBlock: Sendable {
    public enum Kind: Sendable, Equatable {
        case paragraph
        case heading(level: Int)
        /// `language` is the fence's info string, when it gave one.
        case codeBlock(language: String?)
        /// `ordinal` is nil for a bullet. `depth` is 0 for a top-level list.
        case listItem(ordinal: Int?, depth: Int)
        case blockQuote
        case thematicBreak
        /// Rows of cells, header rows first. `headerRows` says how many lead.
        case table(headerRows: Int)
    }

    public var kind: Kind
    /// Inline formatting preserved: bold, italic, code, links, strikethrough.
    /// Empty for `.thematicBreak` and `.table`, whose content is in `cells`.
    public var text: AttributedString
    /// Row-major, for `.table` only.
    public var cells: [[AttributedString]]

    public init(kind: Kind, text: AttributedString = "", cells: [[AttributedString]] = []) {
        self.kind = kind
        self.text = text
        self.cells = cells
    }

    /// The plain characters, for a code block or any other consumer that wants
    /// the source rather than the presentation.
    public var plain: String { String(text.characters) }
}

public enum MarkdownParser {

    /// Parses `source` into blocks, in document order.
    ///
    /// Never throws and never returns empty for non-empty input: a document that
    /// cannot be parsed comes back as one paragraph holding the raw text, which
    /// is exactly what the transcript rendered before any of this existed. A
    /// formatting feature must not be able to lose the model's words.
    public static func blocks(_ source: String) -> [MarkdownBlock] {
        guard !source.isEmpty else { return [] }

        let opts = AttributedString.MarkdownParsingOptions(
            allowsExtendedAttributes: true,
            interpretedSyntax: .full,
            failurePolicy: .returnPartiallyParsedIfPossible)

        guard let parsed = try? AttributedString(markdown: source, options: opts) else {
            return [MarkdownBlock(kind: .paragraph, text: AttributedString(source))]
        }

        // --- reassemble runs into blocks ---------------------------------
        //
        // Consecutive runs sharing the identity of their INNERMOST intent are
        // one block. Two adjacent paragraphs get different identities; six runs
        // of one paragraph share theirs.
        struct Raw {
            var identity: Int
            var components: [PresentationIntent.IntentType]
            var text: AttributedString
        }
        var raws: [Raw] = []
        for run in parsed.runs {
            let comps = run.presentationIntent?.components ?? []
            let identity = comps.first?.identity ?? -1
            let slice = AttributedString(parsed[run.range])
            if var last = raws.last, last.identity == identity, !raws.isEmpty {
                last.text.append(slice)
                raws[raws.count - 1] = last
            } else {
                raws.append(Raw(identity: identity, components: comps, text: slice))
            }
        }

        // --- classify, and fold table cells back into tables --------------
        var out: [MarkdownBlock] = []
        var pendingCells: [(row: Int, column: Int, header: Bool, text: AttributedString)] = []

        func flushTable() {
            guard !pendingCells.isEmpty else { return }
            let headerRows = Set(pendingCells.filter(\.header).map(\.row)).count
            let rowIndices = pendingCells.map(\.row)
            var rows: [[AttributedString]] = []
            for r in (rowIndices.min() ?? 0)...(rowIndices.max() ?? 0) {
                let cells = pendingCells.filter { $0.row == r }.sorted { $0.column < $1.column }
                if !cells.isEmpty { rows.append(cells.map(\.text)) }
            }
            out.append(MarkdownBlock(kind: .table(headerRows: headerRows), cells: rows))
            pendingCells = []
        }

        for raw in raws {
            let trimmed = trimTrailingNewline(raw.text)

            // A table cell's intent stack is cell -> row -> table; the row tells
            // us where it goes and whether it is a header.
            if let cellColumn = raw.components.compactMap(tableColumn).first {
                let (row, isHeader) = tableRow(raw.components)
                pendingCells.append((row, cellColumn, isHeader, trimmed))
                continue
            }
            flushTable()

            guard let inner = raw.components.first else {
                if !isBlank(trimmed) { out.append(MarkdownBlock(kind: .paragraph, text: trimmed)) }
                continue
            }

            switch inner.kind {
            case .codeBlock(let hint):
                // Deliberately NOT trimmed: a code block's trailing newline is
                // how the renderer tells a finished line from one still being
                // typed (PLAN.md 5.6).
                out.append(MarkdownBlock(kind: .codeBlock(language: normalise(hint)),
                                         text: raw.text))
            case .header(let level):
                out.append(MarkdownBlock(kind: .heading(level: level), text: trimmed))
            case .thematicBreak:
                out.append(MarkdownBlock(kind: .thematicBreak))
            case .blockQuote:
                out.append(MarkdownBlock(kind: .blockQuote, text: trimmed))
            case .paragraph:
                // A paragraph inside a list item is the list item's content.
                if let (ordinal, depth) = listContext(raw.components) {
                    out.append(MarkdownBlock(kind: .listItem(ordinal: ordinal, depth: depth),
                                             text: trimmed))
                } else if raw.components.contains(where: { if case .blockQuote = $0.kind { return true }; return false }) {
                    out.append(MarkdownBlock(kind: .blockQuote, text: trimmed))
                } else if !isBlank(trimmed) {
                    out.append(MarkdownBlock(kind: .paragraph, text: trimmed))
                }
            default:
                if !isBlank(trimmed) { out.append(MarkdownBlock(kind: .paragraph, text: trimmed)) }
            }
        }
        flushTable()
        return out
    }

    // ---- intent inspection -------------------------------------------------

    /// `(ordinal, depth)` when this run sits inside a list item. `ordinal` is
    /// nil for a bullet; depth counts enclosing lists from zero.
    private static func listContext(_ comps: [PresentationIntent.IntentType]) -> (Int?, Int)? {
        var ordinal: Int?
        var isItem = false
        var ordered = false
        var depth = -1
        for c in comps {
            switch c.kind {
            case .listItem(let n): if !isItem { ordinal = n; isItem = true }
            case .orderedList: ordered = true; depth += 1
            case .unorderedList: depth += 1
            default: break
            }
        }
        guard isItem else { return nil }
        return (ordered ? ordinal : nil, max(0, depth))
    }

    private static func tableColumn(_ c: PresentationIntent.IntentType) -> Int? {
        if case .tableCell(let col) = c.kind { return col }
        return nil
    }

    private static func tableRow(_ comps: [PresentationIntent.IntentType]) -> (Int, Bool) {
        for c in comps {
            switch c.kind {
            case .tableHeaderRow: return (0, true)
            case .tableRow(let r): return (r + 1, false)
            default: continue
            }
        }
        return (0, false)
    }

    /// Fence info strings arrive as written: `Swift`, `objective-c`, `sh`.
    /// Lowercased and trimmed here so the highlighter's lookup is not the place
    /// that has to know about case.
    private static func normalise(_ hint: String?) -> String? {
        guard let h = hint?.trimmingCharacters(in: .whitespaces).lowercased(), !h.isEmpty else {
            return nil
        }
        return h
    }

    private static func trimTrailingNewline(_ a: AttributedString) -> AttributedString {
        var out = a
        while let last = out.characters.last, last == "\n" || last == "\r" {
            out.removeSubrange(out.index(beforeCharacterIndex: out.endIndex)..<out.endIndex)
        }
        return out
    }

    private static func isBlank(_ a: AttributedString) -> Bool {
        String(a.characters).trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }
}

private extension AttributedString {
    func index(beforeCharacterIndex i: AttributedString.Index) -> AttributedString.Index {
        characters.index(before: i)
    }
}
