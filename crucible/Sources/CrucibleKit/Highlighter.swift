// Highlighter.swift -- syntax colouring for fenced code blocks.
//
// PLAN.md 5.6: highlight.js, vendored, run in JavaScriptCore -- a system
// framework, so there is still nothing for a user to install. 192 languages,
// ~53 ms to load once, ~0.06 ms per line to highlight.
//
// The output contract is deliberately narrow. hljs returns HTML; this returns
// (text, kind) runs whose texts concatenate back to the input, EXACTLY. Nothing
// downstream ever sees markup, and the reconstruction is checked rather than
// assumed -- if the scan does not rebuild the original byte for byte, the whole
// result is discarded and the block renders plain. A wrong colour is a cosmetic
// bug; altered code in a transcript is a lie about what the model wrote.

import Foundation
import JavaScriptCore

/// What a run of code is, reduced to what a palette needs to distinguish.
///
/// hljs emits ~40 class names of varying specificity; colouring each one
/// separately produces a rainbow rather than a reading aid. These are the
/// distinctions that carry meaning at a glance.
public enum TokenKind: String, Sendable, CaseIterable {
    case plain, keyword, string, number, comment, type, function, variable
    case constant, `operator`, punctuation, meta, attribute, tag, link, emphasis
}

public struct HighlightedRun: Sendable, Equatable {
    public let text: String
    public let kind: TokenKind
    public init(text: String, kind: TokenKind) {
        self.text = text
        self.kind = kind
    }
}

public final class Highlighter: @unchecked Sendable {

    public static let shared = Highlighter()

    private let lock = NSLock()
    private var context: JSContext?
    private var loadFailed = false
    private var languages: Set<String> = []

    /// The last result, and what produced it.
    ///
    /// The streaming rule says a block re-highlights once per completed line,
    /// but SwiftUI decides how often it evaluates `body` -- which during
    /// generation is every streamed delta, many times per line, for a result
    /// that cannot have changed. One entry is enough: a transcript highlights
    /// one growing block at a time, and every repeat within a line hits it.
    private var memoKey: String?
    private var memoValue: [HighlightedRun]?

    /// Where the bundle lives. `Bundle.main` in the app; the tests pass the
    /// vendored path directly so they do not need an .app to run.
    private let explicitBundleURL: URL?

    public init(bundleURL: URL? = nil) {
        self.explicitBundleURL = bundleURL
    }

    // MARK: Loading

    private func ready() -> JSContext? {
        lock.lock()
        defer { lock.unlock() }
        if let c = context { return c }
        if loadFailed { return nil }

        guard let url = explicitBundleURL ?? Self.defaultBundleURL(),
              let js = try? String(contentsOf: url, encoding: .utf8),
              let ctx = JSContext() else {
            loadFailed = true
            return nil
        }
        var failure: String?
        ctx.exceptionHandler = { _, e in failure = e?.toString() }
        ctx.evaluateScript(js)
        guard failure == nil, ctx.objectForKeyedSubscript("hljs") != nil else {
            loadFailed = true
            return nil
        }
        if let list = ctx.evaluateScript("hljs.listLanguages()")?.toArray() as? [String] {
            languages = Set(list.map { $0.lowercased() })
        }
        context = ctx
        return ctx
    }

    private static func defaultBundleURL() -> URL? {
        Bundle.main.url(forResource: "highlight.bundle", withExtension: "js")
    }

    /// Whether hljs knows this language, including through its aliases. Used to
    /// decide between "highlight it" and "leave it plain" without paying for a
    /// failed highlight first.
    public func knows(_ language: String) -> Bool {
        guard let ctx = ready() else { return false }
        let name = language.lowercased()
        // Under the lock throughout: `languages` is written by ready() and read
        // here, and JSContext is not safe to touch from two threads. Every call
        // today comes from the main thread, which makes an unlocked read benign
        // and wrong -- the kind of thing that stays correct until it doesn't.
        lock.lock()
        defer { lock.unlock() }
        if languages.contains(name) { return true }
        return ctx.objectForKeyedSubscript("hljs")?
            .invokeMethod("getLanguage", withArguments: [name])?.isUndefined == false
    }

    public var isAvailable: Bool { ready() != nil }

    // MARK: Highlighting

    /// Colours `code` as `language`. Returns nil when it cannot be done at all
    /// -- no bundle, unknown language, a JS exception, or a scan that did not
    /// reproduce the input -- and the caller renders plain monospace, which is
    /// what the transcript did before this existed.
    public func highlight(_ code: String, language: String) -> [HighlightedRun]? {
        guard !code.isEmpty, let ctx = ready(), knows(language) else { return nil }
        lock.lock()
        defer { lock.unlock() }

        let key = language + "\u{0}" + code
        if key == memoKey { return memoValue }

        var failure: String?
        ctx.exceptionHandler = { _, e in failure = e?.toString() }
        // ignoreIllegals, because a block that is still being typed is often not
        // yet valid in its own grammar, and hljs's default is to throw on that.
        let result = ctx.objectForKeyedSubscript("hljs")?
            .invokeMethod("highlight", withArguments: [code, ["language": language,
                                                              "ignoreIllegals": true]])
        guard failure == nil,
              let html = result?.objectForKeyedSubscript("value")?.toString() else { return nil }
        let runs = scan(html, mustReproduce: code)
        memoKey = key
        memoValue = runs
        return runs
    }

    /// Detects the language and highlights. Only for a block whose fence gave no
    /// hint, and only once it is complete: hljs's guess on a fragment is
    /// unstable, and a flip recolours the whole block (PLAN.md 5.6).
    public func highlightDetecting(_ code: String) -> (language: String?, runs: [HighlightedRun])? {
        guard !code.isEmpty, let ctx = ready() else { return nil }
        lock.lock()
        defer { lock.unlock() }

        var failure: String?
        ctx.exceptionHandler = { _, e in failure = e?.toString() }
        let result = ctx.objectForKeyedSubscript("hljs")?
            .invokeMethod("highlightAuto", withArguments: [code])
        guard failure == nil,
              let html = result?.objectForKeyedSubscript("value")?.toString(),
              let runs = scan(html, mustReproduce: code) else { return nil }
        let lang = result?.objectForKeyedSubscript("language")?.toString()
        return (lang == "undefined" ? nil : lang, runs)
    }

    // MARK: The span scanner

    /// Turns hljs's HTML into runs, without an HTML parser.
    ///
    /// `NSAttributedString(html:)` is the obvious shortcut and is the wrong one:
    /// slow, it drags in a full parser, and it means handing model output to
    /// something whose job is to interpret markup. hljs's output is a known,
    /// narrow shape -- nested `<span class="hljs-x">` and five entities -- so it
    /// is scanned directly.
    ///
    /// Nesting resolves innermost-wins: `<span class="string">"a <span
    /// class="subst">x</span>"</span>` makes `x` a variable and the quotes a
    /// string, which is what a reader expects to see.
    func scan(_ html: String, mustReproduce code: String) -> [HighlightedRun]? {
        var runs: [HighlightedRun] = []
        var stack: [TokenKind] = []
        var current = ""
        var currentKind = TokenKind.plain
        var rebuilt = ""

        func emit() {
            guard !current.isEmpty else { return }
            rebuilt += current
            if let last = runs.last, last.kind == currentKind {
                runs[runs.count - 1] = HighlightedRun(text: last.text + current, kind: currentKind)
            } else {
                runs.append(HighlightedRun(text: current, kind: currentKind))
            }
            current = ""
        }

        var i = html.startIndex
        while i < html.endIndex {
            let c = html[i]
            if c == "<" {
                // `</span>` or `<span class="hljs-...">`. Anything else is not
                // something hljs emits, so refuse rather than guess.
                if html[i...].hasPrefix("</span>") {
                    emit()
                    if !stack.isEmpty { stack.removeLast() }
                    currentKind = stack.last ?? .plain
                    i = html.index(i, offsetBy: 7)
                    continue
                }
                guard html[i...].hasPrefix("<span class=\""),
                      let close = html[i...].firstIndex(of: ">") else { return nil }
                let open = html.index(i, offsetBy: 13)
                let quoted = html[open..<close]
                guard quoted.hasSuffix("\"") else { return nil }
                let classes = quoted.dropLast()
                emit()
                let kind = Self.kind(forClasses: String(classes))
                stack.append(kind)
                currentKind = kind
                i = html.index(after: close)
                continue
            }
            if c == "&" {
                if let (decoded, next) = Self.entity(html, at: i) {
                    current.append(decoded)
                    i = next
                    continue
                }
            }
            current.append(c)
            i = html.index(after: i)
        }
        emit()

        // The check that makes this safe to ship. If the scan did not rebuild
        // the model's code exactly, everything is discarded.
        guard rebuilt == code else { return nil }
        return runs
    }

    private static func entity(_ s: String, at i: String.Index) -> (Character, String.Index)? {
        let table: [(String, Character)] = [
            ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
            ("&quot;", "\""), ("&#x27;", "'"), ("&#39;", "'"),
        ]
        for (needle, ch) in table where s[i...].hasPrefix(needle) {
            return (ch, s.index(i, offsetBy: needle.count))
        }
        return nil
    }

    /// hljs class names -> the palette's distinctions.
    ///
    /// A class attribute can carry several names; the first recognised one wins.
    /// Anything unrecognised is `.plain` rather than a fallback colour, so a new
    /// hljs class shows as uncoloured text instead of as something misleading.
    static func kind(forClasses classes: String) -> TokenKind {
        for raw in classes.split(separator: " ") {
            let name = raw.hasPrefix("hljs-") ? String(raw.dropFirst(5)) : String(raw)
            switch name {
            case "keyword", "selector-tag", "literal-keyword": return .keyword
            case "string", "regexp", "char", "char.escape", "template-tag", "quote": return .string
            case "number", "literal": return .number
            case "comment", "doctag": return .comment
            case "type", "class", "title.class", "title.class.inherited",
                 "built_in", "builtin-name": return .type
            case "title", "title.function", "function", "section": return .function
            case "variable", "template-variable", "subst", "params",
                 "variable.language", "variable.constant": return .variable
            case "symbol", "bullet": return .constant
            case "operator": return .operator
            case "punctuation": return .punctuation
            case "meta", "meta.prompt", "meta keyword", "meta string": return .meta
            case "attr", "attribute", "property", "selector-attr",
                 "selector-class", "selector-id", "selector-pseudo": return .attribute
            case "tag", "name": return .tag
            case "link": return .link
            case "emphasis", "strong", "formula": return .emphasis
            default: continue
            }
        }
        return .plain
    }
}
