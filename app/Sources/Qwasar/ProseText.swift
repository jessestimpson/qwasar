// ProseText.swift -- a run of prose as one selectable text view.
//
// SwiftUI cannot select across `Text` views: `.textSelection(.enabled)` makes
// each one selectable on its own, so a transcript rendered one `Text` per
// paragraph highlighted a paragraph at a time and a drag across three of
// them selected one.  Here a run of prose blocks -- paragraphs, headings,
// list items, quotes -- is one attributed string in one NSTextView, and
// selection, copy and Find work the way they do in any Mac app.
//
// Code blocks and tables are not prose and stay cards of their own (a copy
// button, horizontal scrolling); MarkdownView splits a message at them.

import AppKit
import QwasarKit
import SwiftUI

struct ProseText: NSViewRepresentable {
    let text: NSAttributedString

    func makeNSView(context: Context) -> NSTextView {
        // TextKit 1: its layout manager answers "how tall at this width".
        let tv = NSTextView(usingTextLayoutManager: false)
        tv.isEditable = false
        tv.isSelectable = true
        tv.drawsBackground = false
        tv.isRichText = true
        tv.textContainerInset = .zero
        tv.textContainer?.lineFragmentPadding = 0
        tv.textContainer?.widthTracksTextView = true
        tv.isVerticallyResizable = false
        tv.isHorizontallyResizable = false
        tv.linkTextAttributes = [.foregroundColor: NSColor.linkColor,
                                 .underlineStyle: NSUnderlineStyle.single.rawValue,
                                 .cursor: NSCursor.pointingHand]
        tv.textStorage?.setAttributedString(text)
        return tv
    }

    func updateNSView(_ tv: NSTextView, context: Context) {
        // Unchanged text keeps the selection; streaming text replaces it.
        if let ts = tv.textStorage, !ts.isEqual(to: text) { ts.setAttributedString(text) }
    }

    func sizeThatFits(_ proposal: ProposedViewSize, nsView tv: NSTextView, context: Context) -> CGSize? {
        guard let lm = tv.layoutManager, let tc = tv.textContainer else { return nil }
        let w = proposal.width.flatMap { $0.isFinite && $0 > 0 ? $0 : nil } ?? 600
        tc.containerSize = NSSize(width: w, height: .greatestFiniteMagnitude)
        lm.ensureLayout(for: tc)
        return CGSize(width: w, height: ceil(lm.usedRect(for: tc).height))
    }
}

/// Prose blocks as one attributed string, styled as MarkdownView styled them.
enum ProseBuilder {
    static let bodySize = NSFont.systemFontSize          // SwiftUI's .body on macOS
    static let blockSpacing: CGFloat = 8                 // MarkdownView's VStack spacing

    static func isProse(_ k: MarkdownBlock.Kind) -> Bool {
        switch k {
        case .paragraph, .heading, .listItem, .blockQuote: return true
        default: return false
        }
    }

    static func build(_ blocks: [MarkdownBlock]) -> NSAttributedString {
        let out = NSMutableAttributedString()
        for (i, b) in blocks.enumerated() {
            let para = NSMutableParagraphStyle()
            para.paragraphSpacing = i < blocks.count - 1 ? blockSpacing : 0
            var base = NSFont.systemFont(ofSize: bodySize)
            var color = NSColor.labelColor
            var marker: String?

            switch b.kind {
            case .heading(let level):
                let size: CGFloat = level == 1 ? 22 : level == 2 ? 19 : level == 3 ? 17 : 15
                base = NSFont.systemFont(ofSize: size, weight: .semibold)
                para.paragraphSpacingBefore = i > 0 ? 4 : 0
            case .listItem(let ordinal, let depth):
                let indent = CGFloat(depth) * 18
                let hang: CGFloat = 22
                marker = ordinal.map { "\($0).\t" } ?? "•\t"
                para.firstLineHeadIndent = indent
                para.headIndent = indent + hang
                para.tabStops = [NSTextTab(textAlignment: .left, location: indent + hang)]
                para.defaultTabInterval = hang
            case .blockQuote:
                para.firstLineHeadIndent = 12
                para.headIndent = 12
                color = .secondaryLabelColor
            default:
                break
            }

            if let marker {
                out.append(NSAttributedString(string: marker, attributes: [
                    .font: NSFont.monospacedDigitSystemFont(ofSize: bodySize, weight: .regular),
                    .foregroundColor: NSColor.secondaryLabelColor, .paragraphStyle: para]))
            }
            out.append(inline(b.text, base: base, color: color, para: para))
            if i < blocks.count - 1 {
                out.append(NSAttributedString(string: "\n", attributes: [.font: base, .paragraphStyle: para]))
            }
        }
        return out
    }

    /// Inline intents to concrete attributes, explicitly (as MarkdownView did:
    /// inlinePresentationIntent is not reliably honoured on its own).
    private static func inline(_ a: AttributedString, base: NSFont, color: NSColor,
                               para: NSParagraphStyle) -> NSAttributedString {
        let out = NSMutableAttributedString()
        let fm = NSFontManager.shared
        for run in a.runs {
            let s = String(a[run.range].characters)
            let intent = run.inlinePresentationIntent ?? []
            var font = base
            var attrs: [NSAttributedString.Key: Any] = [.foregroundColor: color, .paragraphStyle: para]
            if intent.contains(.stronglyEmphasized) { font = fm.convert(font, toHaveTrait: .boldFontMask) }
            if intent.contains(.emphasized) { font = fm.convert(font, toHaveTrait: .italicFontMask) }
            if intent.contains(.code) {
                font = NSFont.monospacedSystemFont(ofSize: base.pointSize - 1, weight: .regular)
                attrs[.foregroundColor] = NSColor(CodePalette.inlineCode)
            }
            if intent.contains(.strikethrough) { attrs[.strikethroughStyle] = NSUnderlineStyle.single.rawValue }
            if let link = run.link { attrs[.link] = link }
            attrs[.font] = font
            out.append(NSAttributedString(string: s, attributes: attrs))
        }
        return out
    }
}
