// ShellLexer.swift -- enough of sh's grammar to see the commands in a
// command line, for ToolPolicy (spec §16.3).
//
// Not a shell: nothing is expanded but $HOME and ~, which a path check needs
// and whose values are known.  Every other expansion -- $VAR, $(...),
// backticks, $'...' -- marks its word as not literal, so the policy treats
// it as unknown rather than guessing.  The bodies of command substitutions
// are handed back to be classified as commands in their own right, because
// they run.

import Foundation

struct ShellWord: Equatable {
    var text: String
    /// No expansion the lexer could not resolve.
    var literal = true
    /// An unquoted * ? [ or {.
    var glob = false
    /// Any part was quoted or escaped.
    var quoted = false
}

struct ShellRedirect: Equatable {
    var op: String
    var target: ShellWord
}

struct ShellCommand: Equatable {
    var words: [ShellWord] = []
    var redirects: [ShellRedirect] = []
    /// Not first in its pipeline: its stdin is another command's output.
    var pipedInto = false
}

struct ShellParse {
    var commands: [ShellCommand] = []
    /// The bodies of $(...), `...`, <(...) and >(...), and of unquoted
    /// heredocs' substitutions.
    var substitutions: [String] = []
    var error: String?
}

enum ShellLexer {
    static func parse(_ source: String, home: String) -> ShellParse {
        var l = Lexer(c: Array(source), home: home)
        l.run()
        return l.out
    }
}

private struct Lexer {
    let c: [Character]
    let home: String
    var i = 0
    var out = ShellParse()
    var cur = ShellCommand()
    var expecting: String?
    var heredocs: [(delim: String, strip: Bool, quoted: Bool)] = []

    init(c: [Character], home: String) {
        self.c = c
        self.home = home
    }

    func peek(_ k: Int) -> Character? { i + k < c.count ? c[i + k] : nil }

    static func ends(_ ch: Character?) -> Bool {
        guard let ch else { return true }
        return " \t\n;&|()<>".contains(ch)
    }

    mutating func run() {
        while i < c.count {
            let ch = c[i]
            switch ch {
            case " ", "\t", "\r":
                i += 1
            case "\\" where peek(1) == "\n":
                i += 2
            case "\n":
                i += 1
                end(pipe: false)
                readHeredocs()
            case "#":
                while i < c.count, c[i] != "\n" { i += 1 }
            case ";":
                i += peek(1) == ";" ? 2 : 1
                end(pipe: false)
            case "&":
                if peek(1) == "&" { i += 2; end(pipe: false) }
                else if peek(1) == ">" { redirect() }
                else { i += 1; end(pipe: false) }
            case "|":
                if peek(1) == "|" { i += 2; end(pipe: false) }
                else { i += peek(1) == "&" ? 2 : 1; end(pipe: true) }
            case "(", ")":
                i += 1
                end(pipe: false)
            case "<", ">":
                if peek(1) == "(" {
                    i += 1
                    out.substitutions.append(balanced())
                    add(ShellWord(text: "", literal: false))
                } else {
                    redirect()
                }
            default:
                let w = word()
                // `2>file`, `2>&1`: digits run into a redirection are its fd.
                if !w.quoted, w.literal, !w.text.isEmpty, w.text.allSatisfy(\.isASCII),
                   w.text.allSatisfy(\.isNumber), let n = peek(0), n == "<" || n == ">",
                   peek(1) != "(" {
                    redirect()
                } else {
                    add(w)
                }
            }
        }
        if expecting != nil { out.error = "a redirection with no target" }
        end(pipe: false)
    }

    mutating func end(pipe: Bool) {
        if !cur.words.isEmpty || !cur.redirects.isEmpty { out.commands.append(cur) }
        cur = ShellCommand()
        cur.pipedInto = pipe
    }

    mutating func add(_ w: ShellWord) {
        if let op = expecting {
            expecting = nil
            cur.redirects.append(ShellRedirect(op: op, target: w))
            if op == "<<" || op == "<<-" {
                heredocs.append((w.text, op == "<<-", w.quoted))
            }
        } else {
            cur.words.append(w)
        }
    }

    mutating func redirect() {
        for op in ["&>>", "&>", "<<<", "<<-", "<<", "<>", "<&", ">>", ">|", ">&", "<", ">"] {
            let ops = Array(op)
            if i + ops.count <= c.count, Array(c[i..<(i + ops.count)]) == ops {
                i += ops.count
                if expecting != nil { out.error = "a redirection with no target" }
                expecting = op
                return
            }
        }
        i += 1
    }

    /// Heredoc bodies start on the line after the redirection; they are
    /// data, but an unquoted delimiter lets $(...) in the body run.
    mutating func readHeredocs() {
        let pending = heredocs
        heredocs = []
        for h in pending {
            var body = ""
            var found = false
            while i < c.count {
                var line = ""
                while i < c.count, c[i] != "\n" { line.append(c[i]); i += 1 }
                if i < c.count { i += 1 }
                let test = h.strip ? String(line.drop(while: { $0 == "\t" })) : line
                if test == h.delim { found = true; break }
                body += line + "\n"
            }
            if !found { out.error = "a heredoc with no end (\(h.delim))" }
            if !h.quoted { scanSubstitutions(body) }
        }
    }

    /// The substitutions in text read as if double-quoted.
    mutating func scanSubstitutions(_ text: String) {
        var inner = Lexer(c: Array(text), home: home)
        var sink = ""
        var literal = true
        while inner.i < inner.c.count {
            switch inner.c[inner.i] {
            case "\\": inner.i += 2
            case "$": inner.dollar(&sink, &literal)
            case "`": inner.backtick()
            default: inner.i += 1
            }
        }
        out.substitutions += inner.out.substitutions
        if let e = inner.out.error { out.error = e }
    }

    mutating func word() -> ShellWord {
        var w = ShellWord(text: "")
        let start = i
        while i < c.count {
            let ch = c[i]
            if Lexer.ends(ch) { break }
            switch ch {
            case "\\":
                w.quoted = true
                if let n = peek(1) {
                    if n != "\n" { w.text.append(n) }
                    i += 2
                } else {
                    i += 1
                }
            case "'":
                w.quoted = true
                i += 1
                while i < c.count, c[i] != "'" { w.text.append(c[i]); i += 1 }
                if i >= c.count { out.error = "an unterminated quote" }
                i += 1
            case "\"":
                w.quoted = true
                i += 1
                doubleQuoted(&w)
            case "$":
                dollar(&w.text, &w.literal)
            case "`":
                backtick()
                w.literal = false
            case "~" where i == start:
                if peek(1) == "/" || Lexer.ends(peek(1)) {
                    w.text += home
                } else {
                    w.text.append("~")
                    w.literal = false       // ~user
                }
                i += 1
            case "*", "?", "[", "{":
                w.glob = true
                w.text.append(ch)
                i += 1
            default:
                w.text.append(ch)
                i += 1
            }
        }
        return w
    }

    mutating func doubleQuoted(_ w: inout ShellWord) {
        while i < c.count {
            let ch = c[i]
            switch ch {
            case "\"":
                i += 1
                return
            case "\\":
                if let n = peek(1), "$`\"\\\n".contains(n) {
                    if n != "\n" { w.text.append(n) }
                    i += 2
                } else {
                    w.text.append("\\")
                    i += 1
                }
            case "$":
                dollar(&w.text, &w.literal)
            case "`":
                backtick()
                w.literal = false
            default:
                w.text.append(ch)
                i += 1
            }
        }
        out.error = "an unterminated quote"
    }

    /// At a `$`.
    mutating func dollar(_ text: inout String, _ literal: inout Bool) {
        let n = peek(1)
        if n == "(" {
            i += 1
            if peek(1) == "(" {
                _ = balanced()              // arithmetic: nothing runs
            } else {
                out.substitutions.append(balanced())
            }
            literal = false
        } else if n == "{" {
            var k = i + 2
            var name = ""
            while k < c.count, c[k] != "}" { name.append(c[k]); k += 1 }
            if k >= c.count { out.error = "an unterminated ${" }
            i = k + 1
            if name == "HOME" { text += home } else { literal = false }
        } else if n == "'" {
            i += 2
            while i < c.count, c[i] != "'" { i += c[i] == "\\" ? 2 : 1 }
            i += 1
            literal = false
        } else if let n, n.isLetter || n == "_" {
            var k = i + 1
            var name = ""
            while k < c.count, c[k].isLetter || c[k].isNumber || c[k] == "_" { name.append(c[k]); k += 1 }
            i = k
            if name == "HOME" { text += home } else { literal = false }
        } else if let n, "0123456789?@*#$!-".contains(n) {
            i += 2
            literal = false
        } else {
            text.append("$")
            i += 1
        }
    }

    /// At a `(`: its body, through the matching `)`.
    mutating func balanced() -> String {
        let open = i
        var depth = 0
        var k = i
        while k < c.count {
            let ch = c[k]
            if ch == "\\" { k += 2; continue }
            if ch == "'" {
                k += 1
                while k < c.count, c[k] != "'" { k += 1 }
                k += 1
                continue
            }
            if ch == "\"" {
                k += 1
                while k < c.count, c[k] != "\"" { k += c[k] == "\\" ? 2 : 1 }
                k += 1
                continue
            }
            if ch == "(" { depth += 1 }
            if ch == ")" {
                depth -= 1
                if depth == 0 {
                    i = k + 1
                    return String(c[(open + 1)..<k])
                }
            }
            k += 1
        }
        out.error = "an unterminated ("
        i = c.count
        return String(c[min(open + 1, c.count)...])
    }

    /// At a backtick.
    mutating func backtick() {
        var body = ""
        var k = i + 1
        while k < c.count, c[k] != "`" {
            if c[k] == "\\", k + 1 < c.count { body.append(c[k + 1]); k += 2; continue }
            body.append(c[k])
            k += 1
        }
        if k >= c.count { out.error = "an unterminated backtick" }
        i = k + 1
        out.substitutions.append(body)
    }
}
