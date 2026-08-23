// MarkdownSuite.swift -- blocks out, and colour that does not lie.
//
// PLAN.md 5.6. Two things here can break silently and neither is visible in a
// screenshot: Foundation's parse changing shape under us, and the span scanner
// reconstructing something other than what the model wrote.

import Foundation
import CrucibleKit

enum MarkdownSuite {
    static func run(_ args: [String]) -> Int {
        var f = 0
        f += blocks()
        f += highlighting()
        return f
    }

    // MARK: Parsing

    private static func blocks() -> Int {
        var f = 0

        let doc = """
        # Title

        A paragraph with **bold**, *italic*, `code` and a [link](https://x.com).

        ```swift
        func f() -> Int { 42 }
        ```

        - bullet one
        - bullet two

        1. first
        2. second

        > quoted

        ---

        | a | b |
        |---|---|
        | 1 | 2 |
        """
        let b = MarkdownParser.blocks(doc)

        func kinds(_ blocks: [MarkdownBlock]) -> [String] {
            blocks.map { blk in
                switch blk.kind {
                case .paragraph: return "p"
                case .heading(let l): return "h\(l)"
                case .codeBlock(let lang): return "code(\(lang ?? "-"))"
                case .listItem(let o, _): return o == nil ? "li" : "li\(o!)"
                case .blockQuote: return "quote"
                case .thematicBreak: return "hr"
                case .table(let h): return "table(\(h))"
                }
            }
        }
        let got = kinds(b)
        f += check(got.first == "h1", "a heading comes out as a heading (got \(got.first ?? "-"))")
        f += check(got.contains("code(swift)"),
                   "a fenced block keeps its language hint (got \(got))")
        f += check(got.filter { $0 == "li" }.count == 2, "two bullets")
        f += check(got.contains("li1") && got.contains("li2"), "ordered items keep their ordinals")
        f += check(got.contains("quote"), "a block quote")
        f += check(got.contains("hr"), "a thematic break")
        f += check(got.contains(where: { $0.hasPrefix("table(") }), "a table")

        // The code block's text is the code, not the fence, and it keeps the
        // trailing newline the streaming rule depends on.
        if let code = b.first(where: { if case .codeBlock = $0.kind { return true }; return false }) {
            f += check(code.plain == "func f() -> Int { 42 }\n",
                       "the code block is exactly the code, newline included "
                       + "(got \(code.plain.debugDescription))")
        } else {
            f += check(false, "there is a code block")
        }

        // Inline formatting survives as intents rather than as literal asterisks.
        if let para = b.first(where: { $0.kind == .paragraph }) {
            let text = para.plain
            f += check(!text.contains("**") && !text.contains("`"),
                       "inline markers are consumed, not shown (got \(text.debugDescription))")
            let hasBold = para.text.runs.contains {
                ($0.inlinePresentationIntent ?? []).contains(.stronglyEmphasized)
            }
            f += check(hasBold, "bold survives as an intent")
            f += check(para.text.runs.contains { $0.link != nil }, "the link survives")
        } else {
            f += check(false, "there is a paragraph")
        }

        // Degradation: nothing may lose the model's words.
        let plain = "just some text with no markup at all"
        f += check(MarkdownParser.blocks(plain).first?.plain == plain,
                   "plain text round-trips")
        f += check(MarkdownParser.blocks("").isEmpty, "empty input gives no blocks")

        // An unterminated fence is the normal state of a streaming block.
        let open = "before\n\n```python\nx = 1\n"
        let ob = MarkdownParser.blocks(open)
        f += check(ob.contains { if case .codeBlock = $0.kind { return true }; return false },
                   "an unclosed fence still parses as a code block")

        return f
    }

    // MARK: Highlighting

    private static func highlighting() -> Int {
        var f = 0
        // The vendored bundle, straight from the tree: the tests must not need
        // an .app to have been built.
        let url = URL(fileURLWithPath: "vendor/highlight.js/highlight.bundle.js")
        guard FileManager.default.fileExists(atPath: url.path) else {
            print("  SKIP no highlight bundle (run: sh vendor/highlight.js/build.sh)")
            return 0
        }
        let hl = Highlighter(bundleURL: url)
        f += check(hl.isAvailable, "the bundle loads")
        guard hl.isAvailable else { return f }

        // The languages this project actually contains, plus the ones whose
        // multi-line constructs are what separate a grammar from a regex.
        let samples: [(String, String)] = [
            ("swift", "func f(_ x: Int) -> String { return \"v=\\(x)\" } // note"),
            ("elixir", "defmodule A do\n  def run(%{\"k\" => v}), do: {:ok, ~s(sigil #{v})}\nend"),
            ("erlang", "-module(a).\nf(X) when X > 0 -> {ok, X}."),
            ("c", "int main(void){ /* c */ char *s = \"hi\\n\"; return 0x2A; }"),
            ("objectivec", "@implementation Foo\n- (void)bar { NSLog(@\"x\"); }\n@end"),
            ("bash", "cat <<'EOF'\nnot $expanded\nEOF"),
            ("python", "def f(x: int) -> str:\n    return f'{x!r}'  # comment"),
            ("json", "{\"a\": [1, 2.5, null, true]}"),
            ("rust", "let s = r#\"raw \"quoted\" string\"#;"),
            ("makefile", "all: foo\n\t$(CC) -o $@ $<"),
        ]
        for (lang, code) in samples {
            guard let runs = hl.highlight(code, language: lang) else {
                f += check(false, "\(lang) highlights"); continue
            }
            // The property that makes this safe to ship at all.
            let rebuilt = runs.map(\.text).joined()
            f += check(rebuilt == code, "\(lang): the runs rebuild the code exactly")
            f += check(runs.contains { $0.kind != .plain }, "\(lang): something is coloured")
        }

        // Characters that become entities in hljs's output must come back as
        // themselves. If this regresses, the transcript shows `&lt;` where the
        // model wrote `<` -- a lie about the code, not a cosmetic bug.
        let tricky = "if (a < b && c > d) { s = \"x&y\"; t = 'q'; }"
        if let runs = hl.highlight(tricky, language: "c") {
            f += check(runs.map(\.text).joined() == tricky,
                       "entities decode back to the original characters")
        } else {
            f += check(false, "the entity sample highlights")
        }

        // Failure paths return nil, and the caller renders plain.
        f += check(hl.highlight("x", language: "not-a-language") == nil,
                   "an unknown language is refused rather than guessed")
        f += check(hl.knows("swift") && !hl.knows("definitely-not-a-language"),
                   "known languages are reported accurately")

        // The streaming property: highlighting every growing prefix must leave
        // every SETTLED line identical to its rendering in the finished block.
        // The fixtures are constructs that span lines, because they are the only
        // ones that could break it.
        let spanning: [(String, String)] = [
            ("python", "x = 1\n\"\"\"\ndoc\nlines\n\"\"\"\ny = 2"),
            ("c", "int a;\n/* a block\n   comment */\nint b;"),
            ("bash", "echo hi\ncat <<'EOF'\nnot $expanded\nEOF\necho bye"),
            ("swift", "let a = 1\nlet s = \"\"\"\nmulti\n\"\"\"\nlet b = 2"),
            ("elixir", "x = 1\n~s\"\"\"\nheredoc\n\"\"\"\ny = 2"),
            ("rust", "let a = 1;\nlet s = r#\"raw\nstring\"#;\nlet b = 2;"),
        ]
        for (lang, code) in spanning {
            let lines = code.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
            guard let finalRuns = hl.highlight(code, language: lang) else {
                f += check(false, "\(lang) spanning sample highlights"); continue
            }
            let finalLines = renderLines(finalRuns)
            var stable = true
            for k in 1..<lines.count {
                let prefix = lines[0..<k].joined(separator: "\n")
                guard let runs = hl.highlight(prefix, language: lang) else { continue }
                let got = renderLines(runs)
                // Compare only the settled lines -- all but the one being typed.
                for i in 0..<max(0, k - 1) where i < got.count && i < finalLines.count {
                    if got[i] != finalLines[i] { stable = false }
                }
            }
            f += check(stable, "\(lang): settled lines never change colour as the block grows")
        }

        return f
    }

    /// Per-line "colour signature": what a settled line looks like, so two
    /// renderings can be compared without caring about run boundaries.
    private static func renderLines(_ runs: [HighlightedRun]) -> [String] {
        var out: [String] = [""]
        for run in runs {
            let parts = run.text.split(separator: "\n", omittingEmptySubsequences: false)
            for (i, part) in parts.enumerated() {
                if i > 0 { out.append("") }
                if !part.isEmpty { out[out.count - 1] += "\(run.kind.rawValue):\(part)|" }
            }
        }
        return out
    }

    private static func check(_ ok: Bool, _ what: String) -> Int {
        print(ok ? "  ok   \(what)" : "  FAIL \(what)")
        return ok ? 0 : 1
    }
}
