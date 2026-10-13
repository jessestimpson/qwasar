// ToolPolicy.swift -- which tool calls run, which ask the user first, and
// which are refused.  Spec §16.3.
//
// A pure function of the call and where the session works: nothing here
// runs anything, and the only I/O is realpath/lstat on the paths a call
// names, so a symlink in the project cannot point a write somewhere else.
// Every decision is testable with strings (ToolPolicySuite).
//
// The rule that matters most is the last one: a command this file does not
// recognise ASKS.  A denylist is a game the model wins by finding a synonym;
// with "unknown means ask", a missed pattern costs one card.  And parsing
// judges what a command says it will do, not what it can do -- `make` runs
// whatever the Makefile says.  Containment (§16.5) bounds that; this does not.

import Darwin
import Foundation

public enum Verdict: Sendable, Equatable {
    case allow
    /// The user decides.  `high` marks a card worth reading twice; `grant`
    /// is what the call needs beyond containment's default, if approved.
    case ask(reasons: [String], high: Bool, grant: ContainmentGrant)
    /// Refused without asking: the catastrophic set (§16.9).
    case deny(reason: String)

    static func asking(_ reason: String, high: Bool = false, grant: ContainmentGrant = .none) -> Verdict {
        .ask(reasons: [reason], high: high, grant: grant)
    }

    /// The worse of two; two asks keep both reasons.
    static func + (a: Verdict, b: Verdict) -> Verdict {
        switch (a, b) {
        case (.deny, _): return a
        case (_, .deny): return b
        case (.allow, _): return b
        case (_, .allow): return a
        case let (.ask(r1, h1, g1), .ask(r2, h2, g2)):
            return .ask(reasons: r1 + r2.filter { !r1.contains($0) }, high: h1 || h2, grant: g1 + g2)
        }
    }

    static func += (a: inout Verdict, b: Verdict) { a = a + b }
}

/// Where a session works, made canonical once.
public struct PolicyContext: Sendable {
    public let root: String
    public let home: String
    /// Outside the root, where writing is still not worth a question.
    public let scratch: [String]

    public init(root: String, home: String = NSHomeDirectory(), scratch: [String]? = nil) {
        self.root = Paths.physical(Paths.lexical(root))
        self.home = Paths.physical(Paths.lexical(home))
        self.scratch = (scratch ?? [NSTemporaryDirectory(), "/tmp"]).map { Paths.physical(Paths.lexical($0)) }
    }
}

public enum ToolPolicy {

    public static func classify(_ call: ToolCall, _ ctx: PolicyContext) -> Verdict {
        switch call.name {
        case "Write", "Edit":
            // A call with no usable path is the tool's to refuse.
            guard let raw = call.argument("file_path") ?? call.argument("path"),
                  let abs = HostBackend.absolutePath(raw, workDir: ctx.root, home: ctx.home)
            else { return .allow }
            return writing(Paths.physical(abs), ctx)
        case "Bash":
            guard let command = call.argument("command") else { return .allow }
            return shell(command, ctx, cwd: ctx.root, depth: 0)
        default:
            // Read, Glob, Grep, TodoWrite, AskUserQuestion; an unknown name
            // is the executor's to refuse.
            return .allow
        }
    }

    // MARK: - secrets (§16.4)

    /// The secrets `call` would read, as shown to the user; empty for most
    /// calls.  Reading is harmless on its own -- inference is local -- and
    /// this is what makes a later outbound call worth a closer look.  A path
    /// the shell computes is not seen: this is a tripwire, not a boundary.
    public static func secretsRead(_ call: ToolCall, _ ctx: PolicyContext) -> [String] {
        switch call.name {
        case "Read", "Grep":
            guard let raw = call.argument("file_path") ?? call.argument("path"),
                  let abs = HostBackend.absolutePath(raw, workDir: ctx.root, home: ctx.home) else { return [] }
            let p = Paths.physical(abs)
            return isSecret(p, ctx) || (call.name == "Grep" && holdsSecrets(p, ctx)) ? [shown(p, ctx)] : []
        case "Bash":
            guard let command = call.argument("command") else { return [] }
            return shellSecrets(command, ctx, depth: 0)
        default:
            return []
        }
    }

    static func shellSecrets(_ source: String, _ ctx: PolicyContext, depth: Int) -> [String] {
        guard depth <= 6 else { return [] }
        let parse = ShellLexer.parse(source, home: ctx.home)
        var out: [String] = []
        func add(_ s: String) { if !out.contains(s) { out.append(s) } }
        var cwd: String? = ctx.root
        for cmd in parse.commands {
            let words = cmd.words
            let name = words.first.map { Paths.base($0.text).lowercased() } ?? ""
            if name == "cd", let t = words.dropFirst().first { cwd = path(t, cwd) }
            if name == "security", let sub = words.dropFirst().first?.text,
               sub.hasPrefix("find-") || sub == "dump-keychain" || sub == "export" {
                add("the keychain")
            }
            if (name == "printenv" || name == "env") && words.count == 1 { add("the shell environment") }
            // Names and metadata, not contents.
            if metadataOnly.contains(name) { continue }
            for w in words.dropFirst() + cmd.redirects.filter({ $0.op == "<" }).map(\.target) {
                // `-d@file`, `--data=@file`: curl's way of sending a file.
                var w = w
                if w.text.hasPrefix("@") {
                    w.text = String(w.text.dropFirst())
                } else if let at = w.text.firstIndex(of: "@"), w.text.hasPrefix("-") || w.text.contains("=@") {
                    w.text = String(w.text[w.text.index(after: at)...])
                } else if w.text.hasPrefix("-") { continue }
                guard let p = path(w, cwd) else { continue }
                if isSecret(p, ctx) || (recursiveReaders.contains(name) && holdsSecrets(p, ctx)) { add(shown(p, ctx)) }
            }
        }
        for s in parse.substitutions { shellSecrets(s, ctx, depth: depth + 1).forEach(add) }
        return out
    }

    /// Commands that read a whole directory when given one.
    static let recursiveReaders: Set<String> = ["grep", "egrep", "fgrep", "rg", "ag", "ack", "tar", "bsdtar", "gtar",
                                                "zip", "cp", "rsync", "ditto", "scp", "cat", "base64", "xxd", "strings"]

    static let metadataOnly: Set<String> = ["ls", "stat", "test", "[", "[[", "file", "du", "find", "tree", "realpath",
                                            "readlink", "basename", "dirname", "mkdir", "touch", "chmod", "cd",
                                            "pushd", "mdls", "xattr", "wc"]

    /// Directories whose every file is a credential.
    static let secretDirsInHome = [".ssh", ".aws", ".gnupg", "Library/Keychains", ".config/gh", ".kube"]
    static let secretFilesInHome = [".netrc", ".git-credentials", ".npmrc", ".pypirc", ".cargo/credentials",
                                    ".cargo/credentials.toml", ".docker/config.json", ".config/hub"]

    static func isSecret(_ p: String, _ ctx: PolicyContext) -> Bool {
        let name = Paths.base(p).lowercased()
        if Paths.within(p, ctx.home + "/.ssh") {
            return !(name.hasSuffix(".pub") || ["known_hosts", "known_hosts.old", "config",
                                                "authorized_keys", ".ssh"].contains(name))
        }
        if secretDirsInHome.dropFirst().contains(where: { p != ctx.home + "/" + $0 && Paths.within(p, ctx.home + "/" + $0) }) {
            return true
        }
        if secretFilesInHome.contains(where: { p.lowercased() == (ctx.home + "/" + $0).lowercased() }) { return true }
        if name == ".env" || (name.hasPrefix(".env.")
                              && !["example", "sample", "template", "dist", "defaults"].contains(where: name.hasSuffix)) {
            return true
        }
        let ext = (name as NSString).pathExtension
        if ["pem", "key", "p12", "pfx", "keystore", "jks", "ppk", "asc", "gpg"].contains(ext) { return true }
        if ["id_rsa", "id_ed25519", "id_ecdsa", "id_dsa"].contains(name) { return true }
        return ["credentials.json", "secrets.json", "secrets.yml", "secrets.yaml", ".htpasswd",
                "service-account.json", "client_secret.json"].contains(name)
    }

    /// A directory a recursive read would pull credentials out of.
    static func holdsSecrets(_ p: String, _ ctx: PolicyContext) -> Bool {
        secretDirsInHome.contains { p.lowercased() == (ctx.home + "/" + $0).lowercased() }
            || p.lowercased() == ctx.home.lowercased()
    }

    // MARK: - paths

    static let devices: Set<String> = ["/dev/null", "/dev/stdout", "/dev/stderr", "/dev/tty", "/dev/zero"]

    /// Under the home directory: writes worth reading twice.
    static let sensitiveInHome = [
        ".ssh", ".aws", ".gnupg", ".kube", ".docker", ".netrc", ".git-credentials", ".gitconfig",
        ".config/gh", ".config/git", ".config/fish", ".npmrc", ".pypirc", ".cargo/credentials",
        ".cargo/credentials.toml", ".zshrc", ".zprofile", ".zshenv", ".zlogin", ".zlogout",
        ".bashrc", ".bash_profile", ".bash_login", ".profile", "Library/LaunchAgents",
        "Library/Keychains",
    ]
    static let sensitiveSystem = ["/etc", "/private/etc", "/Library", "/System", "/usr", "/bin", "/sbin",
                                  "/Applications", "/private/var/root"]

    /// Deleting these, or everything in them, is never the task.
    static let systemTops: Set<String> = ["/", "/System", "/Library", "/Applications", "/Users", "/usr", "/bin",
                                          "/sbin", "/etc", "/var", "/private", "/private/etc", "/private/var",
                                          "/opt", "/Volumes", "/cores", "/dev", "/tmp", "/private/tmp"]

    /// Inside the project, but run later by something the user trusts.
    static func executesLater(_ rel: String) -> String? {
        let parts = rel.split(separator: "/").map { $0.lowercased() }
        if parts.contains(".git") { return "writes inside .git, whose hooks and config git runs" }
        if parts.first == ".envrc" { return "writes .envrc, which direnv runs" }
        if parts.first == ".husky" { return "writes a git hook (.husky)" }
        if parts.first == ".github", parts.dropFirst().first == "workflows" {
            return "writes a CI workflow, which runs with the repository's secrets"
        }
        if parts == [".vscode", "tasks.json"] { return "writes .vscode/tasks.json, which the editor runs" }
        if parts == [".pre-commit-config.yaml"] { return "writes the pre-commit hooks" }
        return nil
    }

    static func shown(_ p: String, _ ctx: PolicyContext) -> String {
        if p == ctx.root { return "the project directory" }
        if Paths.within(p, ctx.root) { return String(p.dropFirst(ctx.root.count + 1)) }
        if Paths.within(p, ctx.home) { return "~" + p.dropFirst(ctx.home.count) }
        return p
    }

    /// A write to `path`, canonical.
    static func writing(_ path: String, _ ctx: PolicyContext) -> Verdict {
        if devices.contains(path) || path.hasPrefix("/dev/fd/") { return .allow }
        if path.hasPrefix("/dev/") { return .deny(reason: "writes to a device (\(path))") }
        if Paths.within(path, ctx.root) {
            if path == ctx.root { return .allow }
            let rel = String(path.dropFirst(ctx.root.count + 1))
            return executesLater(rel).map { .asking($0, grant: ContainmentGrant(paths: [path])) } ?? .allow
        }
        if ctx.scratch.contains(where: { Paths.within(path, $0) }) { return .allow }
        let sensitive = sensitiveInHome.contains { Paths.within(path, ctx.home + "/" + $0) }
            || sensitiveSystem.contains { Paths.within(path, $0) }
        return .asking("writes outside the project: \(shown(path, ctx))", high: sensitive,
                       grant: ContainmentGrant(paths: [path]))
    }

    /// A write to a path a command computes at run time.
    static let unknownWrite = Verdict.asking("writes to a path computed when it runs",
                                             grant: ContainmentGrant(anywhere: true))

    static func writing(_ path: String?, _ ctx: PolicyContext) -> Verdict {
        path.map { writing($0, ctx) } ?? unknownWrite
    }

    /// Removing (or recursively changing) `path` would be a catastrophe.
    static func catastrophic(_ path: String, glob: Bool, _ ctx: PolicyContext) -> Bool {
        if systemTops.contains(path) || path == ctx.home { return true }
        if Paths.within(ctx.root, path), path != ctx.root { return true }     // an ancestor
        if path == ctx.root { return !glob }                                    // the project itself
        if path == Paths.parent(ctx.home) { return true }
        return false
    }

    // MARK: - Bash

    static func shell(_ source: String, _ ctx: PolicyContext, cwd: String?, depth: Int) -> Verdict {
        if depth > 6 { return .asking("commands nested too deeply to check") }
        if forkBomb(source) { return .deny(reason: "a fork bomb") }
        let p = ShellLexer.parse(source, home: ctx.home)
        var v = Verdict.allow
        if let e = p.error { v += .asking("could not be read as a command: \(e)") }
        var dir = cwd
        for cmd in p.commands { v += command(cmd, ctx, cwd: &dir, depth: depth) }
        for s in p.substitutions { v += shell(s, ctx, cwd: cwd, depth: depth + 1) }
        return v
    }

    static func forkBomb(_ s: String) -> Bool {
        let compact = s.filter { !$0.isWhitespace }
        guard let def = compact.range(of: "(){") else { return false }
        // `name(){ ... name|name& ...}` -- the function pipes itself.
        let name = compact[..<def.lowerBound].split(whereSeparator: { ";&|".contains($0) }).last.map(String.init) ?? ""
        return !name.isEmpty && compact.contains(name + "|" + name)
    }

    static let keywordsDropped: Set<String> = ["then", "else", "do", "elif", "if", "while", "until", "!", "{", "}",
                                              "done", "fi", "esac", "time"]
    static let headers: Set<String> = ["for", "case", "select", "function"]

    static func command(_ cmd: ShellCommand, _ ctx: PolicyContext, cwd: inout String?, depth: Int) -> Verdict {
        var v = Verdict.allow
        for r in cmd.redirects { v += redirect(r, ctx, cwd: cwd) }
        var words = cmd.words
        while let w = words.first, !w.quoted, isAssignment(w.text) { words.removeFirst() }
        while let w = words.first, !w.quoted, keywordsDropped.contains(w.text) { words.removeFirst() }
        if let w = words.first, !w.quoted, headers.contains(w.text) { return v }
        if words.isEmpty { return v }
        let stdin = cmd.redirects.last { $0.op == "<" }?.target
        // A heredoc or here-string feeds text in, as a pipe does.
        let fed = cmd.pipedInto || cmd.redirects.contains { ["<<", "<<-", "<<<"].contains($0.op) }
        return v + simple(words, ctx, cwd: &cwd, piped: fed, stdin: stdin, depth: depth)
    }

    static func isAssignment(_ s: String) -> Bool {
        guard let eq = s.firstIndex(of: "="), eq != s.startIndex else { return false }
        let name = s[..<eq]
        return name.first.map { $0.isLetter || $0 == "_" } == true
            && name.allSatisfy { $0.isLetter || $0.isNumber || $0 == "_" }
    }

    static func redirect(_ r: ShellRedirect, _ ctx: PolicyContext, cwd: String?) -> Verdict {
        switch r.op {
        case ">", ">>", ">|", "&>", "&>>", "<>":
            return writing(path(r.target, cwd), ctx)
        case ">&":
            let t = r.target.text
            if r.target.literal, t == "-" || (!t.isEmpty && t.allSatisfy(\.isNumber)) { return .allow }
            return writing(path(r.target, cwd), ctx)
        default:
            return .allow       // reading
        }
    }

    /// A word as a canonical path; a glob's directory up to its first
    /// wildcard.  nil if the shell computes it.
    static func path(_ w: ShellWord, _ cwd: String?) -> String? {
        guard w.literal, !w.text.isEmpty else { return nil }
        var t = w.text
        if w.glob {
            let parts = t.split(separator: "/", omittingEmptySubsequences: false)
            let cut = parts.firstIndex { $0.contains(where: { "*?[{".contains($0) }) } ?? parts.count
            t = parts[..<cut].joined(separator: "/")
            if t.isEmpty { t = w.text.hasPrefix("/") ? "/" : "." }
        }
        let abs: String
        if t.hasPrefix("/") { abs = t }
        else if let cwd { abs = cwd + "/" + t }
        else { return nil }
        return Paths.physical(abs)
    }

    // MARK: - one command

    /// The words of one simple command, assignments and keywords gone.
    static func simple(_ words: [ShellWord], _ ctx: PolicyContext, cwd: inout String?,
                       piped: Bool, stdin: ShellWord?, depth: Int) -> Verdict {
        var w = words
        // Wrappers run the command after them.
        while let first = w.first {
            guard first.literal else { return .asking("runs a command whose name is computed when it runs") }
            switch Paths.base(first.text).lowercased() {
            case "sudo", "doas", "su", "pkexec", "run0":
                return .deny(reason: "runs as another user (\(first.text))")
            case "env":
                w.removeFirst()
                while let a = w.first, a.text.hasPrefix("-") || isAssignment(a.text) {
                    if a.text == "-S" || a.text.hasPrefix("--split-string") {
                        return .asking("env -S splits a string into a command")
                    }
                    w.removeFirst()
                    if a.text == "-u" || a.text == "-P" || a.text == "-C" { if !w.isEmpty { w.removeFirst() } }
                }
                if w.isEmpty { return .allow }
            case "command", "builtin":
                if w.count > 1, w[1].text == "-v" || w[1].text == "-V" { return .allow }
                w.removeFirst()
                while let a = w.first, a.text == "-p" { w.removeFirst() }
            case "nohup", "exec":
                w.removeFirst()
            case "nice":
                w.removeFirst()
                if w.first?.text == "-n" { w.removeFirst(min(2, w.count)) }
                else if let a = w.first, a.text.hasPrefix("-") { w.removeFirst() }
            case "timeout", "gtimeout":
                w.removeFirst()
                while let a = w.first, a.text.hasPrefix("-") {
                    w.removeFirst()
                    if a.text == "-s" || a.text == "-k" { if !w.isEmpty { w.removeFirst() } }
                }
                if !w.isEmpty { w.removeFirst() }       // the duration
            case "caffeinate":
                w.removeFirst()
                while let a = w.first, a.text.hasPrefix("-") {
                    w.removeFirst()
                    if a.text == "-t" || a.text == "-w" { if !w.isEmpty { w.removeFirst() } }
                }
            case "stdbuf":
                w.removeFirst()
                while let a = w.first, a.text.hasPrefix("-") { w.removeFirst() }
            case "xcrun":
                w.removeFirst()
                while let a = w.first, a.text.hasPrefix("-") {
                    if ["-f", "--find", "--show-sdk-path", "--show-sdk-version", "--show-sdk-platform-path",
                        "--version"].contains(a.text) { return .allow }
                    w.removeFirst()
                    if ["--sdk", "-sdk", "--toolchain", "-toolchain"].contains(a.text) { if !w.isEmpty { w.removeFirst() } }
                }
            default:
                return named(w, ctx, cwd: &cwd, piped: piped, stdin: stdin, depth: depth)
            }
            if w.isEmpty { return .allow }
        }
        return .allow
    }

    /// Non-option arguments, skipping the values of `valued` options; `--`
    /// ends the options.
    static func operands(_ args: ArraySlice<ShellWord>, valued: Set<String> = []) -> [ShellWord] {
        var out: [ShellWord] = []
        var it = args.makeIterator()
        var options = true
        while let a = it.next() {
            if options, a.text == "--" { options = false; continue }
            if options, a.text.hasPrefix("-"), a.text.count > 1 {
                if valued.contains(a.text) { _ = it.next() }
                continue
            }
            out.append(a)
        }
        return out
    }

    /// Single-letter flags (from `-abc` bundles) and long options.
    static func flags(_ args: ArraySlice<ShellWord>) -> Set<String> {
        var out: Set<String> = []
        for a in args {
            if a.text == "--" { break }
            if a.text.hasPrefix("--") { out.insert(String(a.text.split(separator: "=")[0])) }
            else if a.text.hasPrefix("-"), a.text.count > 1 { for ch in a.text.dropFirst() { out.insert(String(ch)) } }
        }
        return out
    }

    static func value(of option: String, in args: ArraySlice<ShellWord>) -> ShellWord? {
        var it = args.makeIterator()
        while let a = it.next() {
            if a.text == option { return it.next() }
            if a.text.hasPrefix(option + "=") {
                var w = a
                w.text = String(a.text.dropFirst(option.count + 1))
                return w
            }
        }
        return nil
    }

    static let readOnly: Set<String> = [
        "ls", "cat", "head", "tail", "wc", "grep", "egrep", "fgrep", "rg", "ag", "ack", "diff", "cmp", "comm",
        "file", "which", "whereis", "type", "echo", "printf", "pwd", "true", "false", "test", "[", "[[", ":",
        "basename", "dirname", "realpath", "readlink", "stat", "du", "df", "date", "printenv", "uname", "whoami",
        "id", "groups", "hostname", "sort", "uniq", "cut", "tr", "tac", "rev", "nl", "fold", "column", "paste",
        "join", "expand", "unexpand", "fmt", "jq", "yq", "less", "more", "tree", "od", "xxd", "hexdump",
        "strings", "shasum", "sha1sum", "sha256sum", "sha512sum", "md5", "md5sum", "cksum", "base64", "seq",
        "sleep", "wait", "export", "set", "unset", "shift", "read", "exit", "return", "local", "declare",
        "typeset", "unalias", "hash", "ulimit", "umask", "trap", "getconf", "locale", "sw_vers",
        "system_profiler", "vm_stat", "ps", "pgrep", "top", "iostat", "mdfind", "mdls", "lipo", "otool", "nm",
        "objdump", "size", "dwarfdump", "mktemp", "popd", "dirs", "history", "lsof", "nproc", "arch", "tput",
        "sum", "split", "csplit", "iconv", "look", "cal", "bc", "dc", "expr", "let", "units", "plutil",
        "codesign", "spctl-assess", "dsymutil", "atos", "leaks", "heap", "vmmap", "sample", "footprint",
        "hyperfine", "lldb", "gdb", "shellcheck", "tokei", "cloc", "scc", "fd", "bat", "delta", "sqlite3",
    ]

    /// Build and test tools that run the project's own code -- which is the
    /// job, and why containment exists.
    static let builders: Set<String> = [
        "make", "gmake", "cmake", "ninja", "meson", "xcodebuild", "gradle", "gradlew", "mvn", "ant", "bazel",
        "bazelisk", "buck", "buck2", "just", "task", "rake", "pytest", "py.test", "tox", "nox", "ruff", "mypy",
        "pyright", "black", "isort", "flake8", "pylint", "eslint", "prettier", "tsc", "jest", "vitest", "mocha",
        "rspec", "phpunit", "swiftlint", "swiftformat", "clang-format", "rustfmt", "gofmt", "goimports",
        "golangci-lint", "cppcheck", "clang-tidy", "rebar3", "erl", "erlc", "elixir", "elixirc", "iex", "ghc",
        "stack", "cabal", "dune", "opam-exec", "ctest", "cpack", "xctest", "swift-format", "deno-test",
        "strip", "install_name_tool", "ar", "ranlib", "ld", "lld", "as", "xcrun-simctl",
    ]

    /// Compilers, judged by where -o puts their output.
    static let compilers: Set<String> = ["cc", "gcc", "g++", "c++", "clang", "clang++", "swiftc", "rustc",
                                         "zig", "javac", "kotlinc", "nvcc", "metal", "metallib", "tcc"]

    static let network: Set<String> = [
        "curl", "wget", "nc", "ncat", "netcat", "socat", "ssh", "scp", "sftp", "ftp", "tftp", "telnet",
        "mail", "mailx", "sendmail", "aria2c", "http", "https", "xh", "httpie", "ping", "ping6", "dig",
        "nslookup", "host", "traceroute", "gh", "glab", "aws", "gcloud", "gsutil", "az", "kubectl", "helm",
        "docker", "podman", "terraform", "heroku", "fly", "flyctl", "vercel", "netlify", "wrangler", "ngrok",
        "tailscale", "nmap", "mosh", "sshpass", "ssh-copy-id", "rclone", "s3cmd",
    ]

    static let cloudCLIs: Set<String> = ["gh", "glab", "aws", "gcloud", "gsutil", "az", "kubectl", "helm", "docker",
                                         "podman", "terraform", "heroku", "fly", "flyctl", "vercel", "netlify",
                                         "wrangler", "ngrok", "tailscale", "rclone", "s3cmd"]

    /// Approved, these run with the network; installing ones may also write
    /// wherever they install.
    static let net = ContainmentGrant(network: true)
    static let install = ContainmentGrant(network: true, anywhere: true)

    static let catastrophicNames: Set<String> = ["shutdown", "reboot", "halt", "fdisk", "gpt", "pdisk", "asr",
                                                 "csrutil", "nvram", "bless", "newfs", "mkfs"]

    static let systemSettings: Set<String> = ["systemsetup", "pmset", "networksetup", "scutil", "spctl", "tccutil",
                                              "dscl", "sysadminctl", "fdesetup", "profiles", "softwareupdate",
                                              "kextload", "kextunload", "kmutil", "dseditgroup", "createhomedir"]

    static let shells: Set<String> = ["sh", "bash", "zsh", "dash", "ksh", "fish", "csh", "tcsh"]
    static let interpreters: Set<String> = ["python", "python2", "python3", "node", "nodejs", "perl", "ruby", "php",
                                            "lua", "luajit", "rscript", "julia", "tclsh", "pwsh", "groovy",
                                            "jshell", "scala", "elixir-script"]

    static func named(_ w: [ShellWord], _ ctx: PolicyContext, cwd: inout String?,
                      piped: Bool, stdin: ShellWord?, depth: Int) -> Verdict {
        let word0 = w[0]
        let args = w[1...]
        let full = word0.text
        // APFS finds /bin/rm for `RM`, so the tables are matched lowercased.
        var name = Paths.base(full).lowercased()
        if name.hasPrefix("python3.") || name.hasPrefix("python2.") { name = "python3" }
        if name.hasPrefix("mkfs") || name.hasPrefix("newfs") { return .deny(reason: "formats a disk (\(name))") }

        // A program by path: the project's own, or one elsewhere.
        if full.contains("/") {
            let p = path(word0, cwd)
            if let p, Paths.within(p, ctx.root) { return .allow }
            let system = full.hasPrefix("/bin/") || full.hasPrefix("/usr/bin/") || full.hasPrefix("/sbin/")
                || full.hasPrefix("/usr/sbin/") || full.hasPrefix("/usr/local/bin/") || full.hasPrefix("/opt/homebrew/bin/")
            if !system { return .asking("runs a program outside the project: \(full)") }
        }

        if catastrophicNames.contains(name) { return .deny(reason: "\(name) is never run by the agent") }
        if readOnly.contains(name) { return readOnlyCase(name, args, ctx, cwd: cwd) }
        if builders.contains(name) { return builderCase(name, args, ctx, cwd: cwd) }
        if compilers.contains(name) {
            for opt in ["-o", "--output", "-femit-bin"] {
                if let o = value(of: opt, in: args) { return writing(path(o, cwd), ctx) }
            }
            return .allow
        }
        if network.contains(name) {
            // The ssh family writes known_hosts; the cloud CLIs keep their
            // own config and credentials -- approved, they write where they
            // must.
            let grant = ["ssh", "scp", "sftp", "ssh-copy-id", "mosh", "sshpass"].contains(name)
                ? ContainmentGrant(network: true, paths: [ctx.home + "/.ssh"])
                : cloudCLIs.contains(name) ? ContainmentGrant(network: true, anywhere: true)
                : ContainmentGrant(network: true)
            return .asking("uses the network (\(name))", grant: grant)
        }
        if systemSettings.contains(name) {
            return .asking("changes system settings (\(name))", high: true, grant: ContainmentGrant(anywhere: true))
        }
        if shells.contains(name) { return shellCase(name, args, ctx, cwd: cwd, piped: piped, stdin: stdin, depth: depth) }
        if interpreters.contains(name) {
            return interpreterCase(name, args, ctx, cwd: cwd, piped: piped, stdin: stdin)
        }

        switch name {
        case "cd", "pushd":
            let target = operands(args).first
            if let target {
                cwd = target.text == "-" ? nil : path(target, cwd)
            } else {
                cwd = ctx.home
            }
            return .allow

        case "rm", "rmdir", "unlink", "shred", "srm":
            let f = flags(args)
            let recursive = name == "rm" && (f.contains("r") || f.contains("R") || f.contains("--recursive"))
            return deleting(operands(args), ctx, cwd: cwd, recursive: recursive, verb: name)

        case "mv":
            // Every operand: the sources leave, the destination is written.
            let ops = operands(args, valued: ["-t", "--target-directory", "-S", "--suffix"])
            let target = value(of: "-t", in: args) ?? value(of: "--target-directory", in: args)
            var v = Verdict.allow
            for (k, o) in ops.enumerated() {
                let p = path(o, cwd)
                let source = target != nil || k < ops.count - 1
                if source, let p, catastrophic(p, glob: o.glob, ctx) {
                    return .deny(reason: "moves \(shown(p, ctx)) away")
                }
                v += writing(p, ctx)
            }
            if let target { v += writing(path(target, cwd), ctx) }
            return v

        case "cp", "ln", "install", "ditto", "gcp":
            if let t = value(of: "-t", in: args) ?? value(of: "--target-directory", in: args) {
                return writing(path(t, cwd), ctx)
            }
            let ops = operands(args, valued: ["-S", "--suffix", "-m", "-o", "-g", "-B"])
            guard let dest = ops.last, ops.count >= (name == "ln" ? 1 : 2) else { return .allow }
            return writing(path(dest, cwd), ctx)

        case "rsync":
            if args.contains(where: { $0.text.contains(":") || $0.text.hasPrefix("rsync://") }) {
                return .asking("copies over the network (rsync)", grant: ContainmentGrant(network: true, anywhere: true))
            }
            guard let dest = operands(args, valued: ["-e", "--rsh", "--exclude", "--include", "--filter",
                                                     "-f", "--files-from", "--exclude-from"]).last
            else { return .allow }
            var v = writing(path(dest, cwd), ctx)
            if flags(args).contains(where: { $0.hasPrefix("--delete") || $0 == "--remove-source-files" }) {
                v += .asking("rsync deletes files")
            }
            return v

        case "touch", "mkdir", "tee", "truncate", "mkfifo":
            return operands(args, valued: ["-m", "-s", "--size", "-r", "-d", "-t"])
                .reduce(Verdict.allow) { $0 + writing(path($1, cwd), ctx) }

        case "chmod", "chown", "chgrp", "chflags":
            let ops = operands(args)
            let recursive = flags(args).contains("R")
            var v = Verdict.allow
            for o in ops.dropFirst() {
                let p = path(o, cwd)
                if recursive, let p, catastrophic(p, glob: o.glob, ctx) {
                    return .deny(reason: "\(name) -R on \(shown(p, ctx))")
                }
                v += writing(p, ctx)
            }
            return v

        case "xattr":
            let f = flags(args)
            if f.isDisjoint(with: ["d", "c", "w", "r"]) { return .allow }
            if args.contains(where: { $0.text == "com.apple.quarantine" }) {
                return .asking("removes the quarantine flag, which lets a downloaded program run unchecked", high: true)
            }
            return operands(args).dropFirst(f.contains("w") ? 2 : (f.contains("d") ? 1 : 0))
                .reduce(Verdict.allow) { $0 + writing(path($1, cwd), ctx) }

        case "sed", "gsed":
            let f = flags(args)
            guard f.contains("i") || f.contains("--in-place") else { return .allow }
            let scripted = args.contains { $0.text == "-e" || $0.text == "-f" || $0.text.hasPrefix("--expression") }
            let ops = operands(args, valued: ["-e", "-f", "--expression", "--file"])
                .filter { !$0.text.hasPrefix("-i") }
            return ops.dropFirst(scripted ? 0 : 1).reduce(Verdict.allow) { $0 + writing(path($1, cwd), ctx) }

        case "awk", "gawk", "nawk", "mawk":
            if args.contains(where: { $0.text == "-i" || $0.text.hasPrefix("--include") }) {
                return .asking("awk edits files in place")
            }
            let prog = operands(args, valued: ["-F", "-v", "-f"]).first?.text ?? ""
            if prog.contains("system") || prog.contains("|") || prog.contains(">") {
                return .asking("the awk program runs commands or writes files")
            }
            return .allow

        case "dd":
            for a in args where a.text.hasPrefix("of=") {
                var o = a
                o.text = String(a.text.dropFirst(3))
                return writing(path(o, cwd), ctx)
            }
            return .allow

        case "tar", "bsdtar", "gtar":
            let first = args.first?.text ?? ""
            let bundle = first.hasPrefix("-") ? "" : first
            let f = flags(args)
            let extract = bundle.contains("x") || f.contains("x") || f.contains("--extract") || f.contains("--get")
            let create = bundle.contains("c") || f.contains("c") || f.contains("--create")
            if extract { return writing(value(of: "-C", in: args).map { path($0, cwd) } ?? cwd, ctx) }
            if create, let a = value(of: "-f", in: args) ?? value(of: "--file", in: args) {
                return writing(path(a, cwd), ctx)
            }
            if create, bundle.contains("f") {
                return writing(operands(args.dropFirst()).first.map { path($0, cwd) } ?? nil, ctx)
            }
            return .allow

        case "unzip":
            if args.contains(where: { $0.text == "-l" || $0.text == "-t" || $0.text == "-v" || $0.text == "-p" }) {
                return .allow
            }
            return writing(value(of: "-d", in: args).map { path($0, cwd) } ?? cwd, ctx)

        case "zip", "gzip", "gunzip", "bzip2", "bunzip2", "xz", "unxz", "zstd", "lz4":
            if flags(args).contains("c") || flags(args).contains("--stdout") || flags(args).contains("l") { return .allow }
            return operands(args).reduce(Verdict.allow) { $0 + writing(path($1, cwd), ctx) }

        case "find":
            return findCase(args, ctx, cwd: cwd, depth: depth)

        case "xargs":
            var rest = args
            var replace: String?
            while let a = rest.first, a.text.hasPrefix("-") {
                rest = rest.dropFirst()
                if ["-I", "-J"].contains(a.text) { replace = rest.first?.text; rest = rest.dropFirst() }
                else if ["-n", "-L", "-P", "-s", "-E", "-d", "-a", "-R", "-S"].contains(a.text) { rest = rest.dropFirst() }
                else if a.text.hasPrefix("-I") { replace = String(a.text.dropFirst(2)) }
            }
            if rest.isEmpty { return .allow }               // echo
            // Its arguments arrive on stdin: unknown, as a computed word is.
            var sub = rest.map { word -> ShellWord in
                guard let r = replace, word.text.contains(r) else { return word }
                return ShellWord(text: word.text, literal: false)
            }
            if replace == nil { sub.append(ShellWord(text: "", literal: false)) }
            var d = cwd
            return simple(sub, ctx, cwd: &d, piped: false, stdin: nil, depth: depth + 1)

        case "git":
            return gitCase(args, ctx, cwd: cwd)

        case "eval":
            return .asking("eval runs text as a command")
        case "source", ".":
            guard let s = operands(args).first, let p = path(s, cwd), Paths.within(p, ctx.root) else {
                return .asking("sources a script from outside the project")
            }
            return .allow
        case "osascript":
            return .asking("runs AppleScript, which can drive other apps", high: true)
        case "open":
            return .asking("opens an app, file or URL")
        case "kill", "pkill", "killall":
            return .asking("stops processes (\(name))")
        case "crontab":
            return args.contains { $0.text == "-l" } ? .allow
                : .asking("changes scheduled jobs (crontab)", high: true, grant: ContainmentGrant(anywhere: true))
        case "defaults":
            let sub = args.first?.text ?? ""
            return ["read", "read-type", "domains", "find", "help"].contains(sub)
                ? .allow : .asking("changes app preferences (defaults \(sub))")
        case "security":
            return .asking("uses the keychain (security)", high: true)
        case "launchctl":
            let sub = args.first?.text ?? ""
            if ["list", "print", "print-disabled", "blame", "getenv", "version", "help", "procinfo",
                "dumpstate"].contains(sub) { return .allow }
            if ["load", "unload", "bootout", "bootstrap", "remove", "disable", "enable", "kickstart", "kill",
                "submit", "stop", "start", "reboot", "setenv", "unsetenv", "config"].contains(sub) {
                return .deny(reason: "launchctl \(sub) changes what macOS runs")
            }
            return .asking("launchctl \(sub)", high: true)
        case "diskutil":
            let sub = args.first?.text ?? ""
            if ["list", "info", "activity", "listFilesystems"].contains(sub) { return .allow }
            let lower = sub.lowercased()
            if lower.hasPrefix("erase") || lower.hasPrefix("partition") || lower.hasPrefix("zero")
                || lower.hasPrefix("secureerase") || lower == "reformat" || lower == "randomdisk"
                || (lower == "apfs" && args.dropFirst().first.map {
                    ["deletecontainer", "deletevolume", "erasevolume", "deletesnapshot"].contains($0.text.lowercased())
                } == true) {
                return .deny(reason: "diskutil \(sub) erases a disk")
            }
            return .asking("diskutil \(sub)", high: true)
        case "sysctl":
            return args.contains { $0.text == "-w" || $0.text.contains("=") }
                ? .asking("changes kernel settings (sysctl)", high: true) : .allow
        case "brew":
            let sub = args.first?.text ?? ""
            return ["list", "ls", "info", "search", "--prefix", "--cellar", "--repository", "--version", "config",
                    "deps", "uses", "leaves", "outdated", "desc", "doctor", "--env", "commands", "help"].contains(sub)
                ? .allow : .asking("changes Homebrew's installs (brew \(sub))", high: true, grant: install)
        case "port", "mas":
            return .asking("installs or removes software (\(name))", high: true, grant: install)
        case "npm", "pnpm", "yarn", "bun":
            return nodePackages(name, args)
        case "npx", "pnpx", "bunx":
            return .asking("downloads and runs a package (\(name))", grant: net)
        case "pip", "pip3", "pipx":
            return pip(name, args)
        case "uv":
            let sub = args.first?.text ?? ""
            if sub == "pip" { return pip("uv pip", args.dropFirst()) }
            return ["run", "tree", "--version", "version", "help"].contains(sub)
                ? .allow : .asking("changes Python packages or uses the network (uv \(sub))", grant: install)
        case "poetry", "pdm", "hatch", "pipenv", "conda", "mamba":
            let sub = args.first?.text ?? ""
            return ["run", "show", "check", "env", "--version", "version", "list", "info", "shell", "test"].contains(sub)
                ? .allow : .asking("changes Python packages or uses the network (\(name) \(sub))", grant: install)
        case "gem":
            let sub = args.first?.text ?? ""
            return ["list", "env", "which", "contents", "--version", "help", "search", "info"].contains(sub)
                ? .allow : .asking("changes Ruby gems (gem \(sub))", grant: install)
        case "bundle", "bundler":
            let sub = args.first?.text ?? ""
            return ["exec", "list", "show", "check", "--version", "info", "config"].contains(sub)
                ? .allow : .asking("changes Ruby gems or uses the network (bundle \(sub))", grant: install)
        case "cargo":
            let sub = args.first?.text ?? ""
            return ["build", "b", "check", "c", "test", "t", "run", "r", "clippy", "fmt", "doc", "bench", "tree",
                    "metadata", "--version", "-V", "version", "help", "clean", "nextest", "expand", "new", "init",
                    "locate-project", "pkgid", "verify-project"].contains(sub)
                ? .allow : .asking("changes Rust installs or uses the network (cargo \(sub))", grant: install)
        case "go":
            let sub = args.first?.text ?? ""
            if sub == "env", args.contains(where: { $0.text == "-w" || $0.text == "-u" }) {
                return .asking("changes Go's settings (go env -w)", grant: ContainmentGrant(anywhere: true))
            }
            if sub == "mod" {
                let m = args.dropFirst().first?.text ?? ""
                return ["graph", "why", "verify", "edit", "init"].contains(m)
                    ? .allow : .asking("uses the network (go mod \(m))", grant: net)
            }
            return ["build", "test", "vet", "run", "fmt", "env", "version", "list", "doc", "help", "generate",
                    "clean", "tool"].contains(sub)
                ? .allow : .asking("changes Go installs or uses the network (go \(sub))", grant: install)
        case "swift":
            let sub = args.first?.text ?? ""
            if sub == "package" {
                let m = args.dropFirst().first?.text ?? ""
                return ["describe", "dump-package", "show-dependencies", "init", "clean", "compute-checksum",
                        "dump-symbol-graph", "diagnose-api-breaking-changes", "plugin"].contains(m)
                    ? .allow : .asking("uses the network (swift package \(m))", grant: net)
            }
            if ["build", "test", "run", "--version", "-version", "format", "symbolgraph-extract", "demangle",
                "help", "repl", ""].contains(sub) { return .allow }
            if sub.hasSuffix(".swift") { return script(sub, args, ctx, cwd: cwd) }
            return .asking("swift \(sub)")
        case "mise", "asdf", "rtx":
            let sub = args.first?.text ?? ""
            return ["exec", "x", "run", "r", "ls", "list", "current", "which", "where", "env", "doctor", "version",
                    "--version", "help", "tasks", "outdated"].contains(sub)
                ? .allow : .asking("changes installed toolchains (\(name) \(sub))", grant: install)
        case "mix":
            let sub = args.first?.text ?? ""
            return sub.hasPrefix("deps.get") || sub.hasPrefix("deps.update") || sub.hasPrefix("hex") || sub.hasPrefix("archive.install")
                || sub.hasPrefix("local.")
                ? .asking("uses the network (mix \(sub))", grant: install) : .allow
        case "dotnet":
            let sub = args.first?.text ?? ""
            return ["restore", "add", "tool", "workload", "nuget", "new"].contains(sub)
                ? .asking("uses the network (dotnet \(sub))", grant: install) : .allow
        default:
            return .asking("\(name) is not a command this app knows to be safe")
        }
    }

    // MARK: - cases

    static func readOnlyCase(_ name: String, _ args: ArraySlice<ShellWord>, _ ctx: PolicyContext,
                             cwd: String?) -> Verdict {
        if name == "sort", let o = value(of: "-o", in: args) ?? value(of: "--output", in: args) {
            return writing(path(o, cwd), ctx)
        }
        if name == "split" || name == "csplit" { return writing(cwd, ctx) }
        if name == "plutil", args.contains(where: { ["-convert", "-insert", "-replace", "-remove"].contains($0.text) }) {
            let ops = operands(args, valued: ["-convert", "-o", "-e", "-insert", "-replace", "-remove", "-extract",
                                              "-r", "-s"])
            let out = value(of: "-o", in: args)
            return writing(out.map { path($0, cwd) } ?? ops.last.map { path($0, cwd) } ?? nil, ctx)
        }
        if name == "codesign" || name == "strip" {
            return operands(args, valued: ["-s", "--sign", "-i", "--identifier", "--entitlements", "-r",
                                           "--requirements", "--options", "-o"])
                .reduce(Verdict.allow) { $0 + writing(path($1, cwd), ctx) }
        }
        if name == "sqlite3" {
            return operands(args).first.map { writing(path($0, cwd), ctx) } ?? .allow
        }
        return .allow
    }

    static func builderCase(_ name: String, _ args: ArraySlice<ShellWord>, _ ctx: PolicyContext,
                            cwd: String?) -> Verdict {
        if args.contains(where: { $0.text == "install" || $0.text == "--install" || $0.text == "uninstall" }) {
            return .asking("\(name) install writes outside the project", grant: ContainmentGrant(anywhere: true))
        }
        // Run in another directory, it builds someone else's code.
        for opt in ["-C", "--directory", "-chdir", "--project-dir"] {
            if let d = value(of: opt, in: args) {
                guard let p = path(d, cwd), Paths.within(p, ctx.root) else {
                    return .asking("\(name) runs in a directory outside the project")
                }
            }
        }
        if name == "cmake", let b = value(of: "-B", in: args) { return writing(path(b, cwd), ctx) }
        guard let cwd, Paths.within(cwd, ctx.root) || ctx.scratch.contains(where: { Paths.within(cwd, $0) }) else {
            return .asking("\(name) runs outside the project")
        }
        return .allow
    }

    /// `filtered`: only what a find expression selects, so never the
    /// catastrophic refusal -- but still a question.
    static func deleting(_ targets: [ShellWord], _ ctx: PolicyContext, cwd: String?,
                         recursive: Bool, verb: String, filtered: Bool = false) -> Verdict {
        var v = Verdict.allow
        for t in targets {
            guard let p = path(t, cwd) else {
                v += .asking("\(verb)\(recursive ? " -r" : "") on a path computed when it runs", high: recursive,
                             grant: ContainmentGrant(anywhere: true))
                continue
            }
            if !filtered, catastrophic(p, glob: t.glob, ctx) && (recursive || t.glob) {
                return .deny(reason: "would delete \(t.glob ? "everything in " : "")\(shown(p, ctx))")
            }
            let inside = Paths.within(p, ctx.root) || ctx.scratch.contains { Paths.within(p, $0) }
            if !inside {
                v += .asking("deletes outside the project: \(shown(p, ctx))", high: true,
                             grant: ContainmentGrant(paths: [p]))
            } else if Paths.within(p, ctx.root), let why = executesLater(String(p.dropFirst(min(p.count, ctx.root.count + 1)))) {
                v += .asking(why.replacingOccurrences(of: "writes", with: "deletes"), grant: ContainmentGrant(paths: [p]))
            } else if recursive {
                v += .asking("deletes recursively: \(shown(p, ctx))", high: p == ctx.root)
            }
        }
        if targets.isEmpty, recursive { v += .asking("\(verb) -r with nothing named") }
        return v
    }

    static func shellCase(_ name: String, _ args: ArraySlice<ShellWord>, _ ctx: PolicyContext, cwd: String?,
                          piped: Bool, stdin: ShellWord?, depth: Int) -> Verdict {
        var it = args.makeIterator()
        while let a = it.next() {
            if a.text == "-c" || (a.text.hasPrefix("-") && !a.text.hasPrefix("--") && a.text.contains("c")) {
                guard let body = it.next() else { return .allow }
                guard body.literal else { return .asking("\(name) -c runs a command computed when it runs") }
                return shell(body.text, ctx, cwd: cwd, depth: depth + 1)
            }
            if ["-o", "+o", "-O", "+O", "--rcfile", "--init-file"].contains(a.text) { _ = it.next(); continue }
            if a.text.hasPrefix("-") || a.text.hasPrefix("+") { continue }
            return script(a.text, args, ctx, cwd: cwd, word: a)
        }
        if let stdin { return script(stdin.text, args, ctx, cwd: cwd, word: stdin) }
        return piped ? .asking("pipes text into a shell (\(name))", high: true) : .allow
    }

    /// The short flags that take a program as their argument.
    static let inlineFlags: [String: Set<String>] = [
        "python": ["c"], "python2": ["c"], "python3": ["c"], "node": ["e", "p"], "nodejs": ["e", "p"],
        "perl": ["e", "E", "i"], "ruby": ["e"], "php": ["r", "R", "B", "E"], "lua": ["e"], "luajit": ["e"],
        "rscript": ["e"], "julia": ["e", "E"], "pwsh": ["c", "C", "e"],
    ]

    static func interpreterCase(_ name: String, _ args: ArraySlice<ShellWord>, _ ctx: PolicyContext, cwd: String?,
                                piped: Bool, stdin: ShellWord?) -> Verdict {
        var it = args.makeIterator()
        while let a = it.next() {
            let t = a.text
            let bundle = t.hasPrefix("-") && !t.hasPrefix("--") ? Set(t.dropFirst().map(String.init)) : []
            if !bundle.isDisjoint(with: inlineFlags[name] ?? ["e", "c"])
                || ["--eval", "--print", "--command"].contains(where: { t.hasPrefix($0) }) {
                return .asking("runs inline \(name) code")
            }
            if t == "-m", name.hasPrefix("python") {
                let module = it.next()?.text ?? ""
                if module == "pip" { return pip("pip", ArraySlice(Array(it))) }
                let safe: Set<String> = ["pytest", "unittest", "venv", "mypy", "black", "ruff", "json.tool",
                                         "py_compile", "compileall", "doctest", "pyflakes", "pylint", "isort",
                                         "coverage", "cProfile", "timeit", "pdb", "trace", "tabnanny", "flake8"]
                return safe.contains(module) ? .allow : .asking("runs the Python module \(module)")
            }
            if t.hasPrefix("-") { continue }
            return script(t, args, ctx, cwd: cwd, word: a)
        }
        if let stdin { return script(stdin.text, args, ctx, cwd: cwd, word: stdin) }
        return piped ? .asking("pipes text into an interpreter (\(name))", high: true) : .allow
    }

    /// Runs the script `file`: the project's own is the job.
    static func script(_ file: String, _ args: ArraySlice<ShellWord>, _ ctx: PolicyContext, cwd: String?,
                       word: ShellWord? = nil) -> Verdict {
        let w = word ?? ShellWord(text: file)
        guard let p = path(w, cwd), Paths.within(p, ctx.root) || ctx.scratch.contains(where: { Paths.within(p, $0) })
        else { return .asking("runs a script from outside the project: \(file)") }
        return .allow
    }

    static func findCase(_ args: ArraySlice<ShellWord>, _ ctx: PolicyContext, cwd: String?, depth: Int) -> Verdict {
        var roots: [ShellWord] = []
        var i = args.startIndex
        while i < args.endIndex, !args[i].text.hasPrefix("-"), args[i].text != "(", args[i].text != "!" {
            roots.append(args[i])
            i += 1
        }
        if roots.isEmpty { roots = [ShellWord(text: ".")] }
        let rest = args[i...]
        let unfiltering: Set<String> = ["-delete", "-depth", "-d", "-print", "-print0", "-xdev", "-mount", "-H", "-L",
                                        "-P", "-follow"]
        var filtered = false
        var inExec = false
        for a in rest {
            if inExec { if a.text == ";" || a.text == "+" { inExec = false }; continue }
            if ["-exec", "-execdir", "-ok", "-okdir"].contains(a.text) { inExec = true; continue }
            if !unfiltering.contains(a.text) { filtered = true }
        }
        var v = Verdict.allow
        if rest.contains(where: { $0.text == "-delete" }) {
            v += deleting(roots, ctx, cwd: cwd, recursive: true, verb: "find -delete", filtered: filtered)
        }
        for opt in ["-fprint", "-fprint0", "-fprintf", "-fls"] {
            if let o = value(of: opt, in: rest) { v += writing(path(o, cwd), ctx) }
        }
        // -exec and friends: the command, with {} a path under the roots.
        var j = rest.startIndex
        while j < rest.endIndex {
            if ["-exec", "-execdir", "-ok", "-okdir"].contains(rest[j].text) {
                var sub: [ShellWord] = []
                j += 1
                while j < rest.endIndex, rest[j].text != ";", rest[j].text != "+" {
                    var w = rest[j]
                    if w.text.contains("{}") {
                        let base = roots.count == 1 ? roots[0].text : nil
                        if let base, roots[0].literal {
                            w.text = w.text.replacingOccurrences(of: "{}", with: base)
                        } else {
                            w.literal = false
                        }
                    }
                    sub.append(w)
                    j += 1
                }
                // A deleting command over everything find selects -- the
                // roots included -- is find -delete by another name.
                if let n = sub.first.map({ Paths.base($0.text) }),
                   ["rm", "rmdir", "unlink", "shred", "srm"].contains(n) {
                    v += deleting(roots, ctx, cwd: cwd, recursive: true, verb: "find -exec \(n)", filtered: filtered)
                } else if !sub.isEmpty {
                    var d = cwd
                    v += simple(sub, ctx, cwd: &d, piped: false, stdin: nil, depth: depth + 1)
                }
            }
            j += 1
        }
        return v
    }

    static func gitCase(_ args: ArraySlice<ShellWord>, _ ctx: PolicyContext, cwd: String?) -> Verdict {
        var rest = args
        var dir = cwd
        var v = Verdict.allow
        let safeConfig = ["color.", "core.quotepath", "user.name", "user.email", "advice.", "init.defaultbranch",
                          "log.", "diff.noprefix", "diff.color", "status.", "column.ui"]
        while let a = rest.first, a.text.hasPrefix("-") {
            rest = rest.dropFirst()
            switch a.text {
            case "-C":
                if let d = rest.first { dir = path(d, dir); rest = rest.dropFirst() }
            case "-c":
                if let kv = rest.first {
                    let key = kv.text.lowercased()
                    if !safeConfig.contains(where: { key.hasPrefix($0) }) {
                        v += .asking("sets git configuration that can run commands (\(kv.text))")
                    }
                    rest = rest.dropFirst()
                }
            case "--exec-path", "--git-dir", "--work-tree", "--namespace":
                rest = rest.dropFirst()
                v += .asking("git with \(a.text)")
            default:
                if a.text.hasPrefix("--exec-path=") || a.text.hasPrefix("--git-dir=") || a.text.hasPrefix("--work-tree=") {
                    v += .asking("git with \(a.text)")
                }
            }
        }
        guard let subWord = rest.first else { return v }
        let sub = subWord.text
        let sargs = rest.dropFirst()
        let f = flags(sargs)
        let outside = dir.map { !Paths.within($0, ctx.root) } ?? true

        let readOnly: Set<String> = ["status", "log", "diff", "show", "blame", "annotate", "shortlog", "describe",
                                     "rev-parse", "rev-list", "ls-files", "ls-tree", "cat-file", "merge-base",
                                     "name-rev", "show-ref", "for-each-ref", "grep", "whatchanged", "count-objects",
                                     "verify-commit", "verify-tag", "check-ignore", "check-attr", "help", "version",
                                     "var", "range-diff", "cherry", "show-branch", "diff-tree", "diff-files",
                                     "diff-index", "fsck", "--version", "--help"]
        if readOnly.contains(sub) { return v }

        let mutating: Verdict
        switch sub {
        case "add", "mv", "rm", "switch", "merge", "cherry-pick", "revert", "init", "apply", "am", "format-patch",
             "notes", "commit", "stash", "branch", "tag", "worktree", "bisect", "reflog", "restore", "checkout",
             "reset", "clean", "config", "remote", "gc", "archive", "sparse-checkout", "update-index", "mergetool":
            mutating = gitMutating(sub, sargs, f, ctx, cwd: dir)
        case "push":
            return v + .asking("pushes to a remote (git push)", high: true, grant: net)
        case "fetch", "pull", "clone", "ls-remote", "submodule", "lfs":
            if sub == "submodule", let m = sargs.first?.text, ["status", "summary"].contains(m) { return v }
            if sub == "submodule", sargs.first?.text == "foreach" { return v + .asking("git submodule foreach runs a command") }
            return v + .asking("uses the network (git \(sub))", grant: net)
        case "rebase":
            mutating = .asking("rewrites history (git rebase)")
        case "filter-branch", "filter-repo", "replace":
            mutating = .asking("rewrites history (git \(sub))", high: true)
        case "update-ref", "prune", "symbolic-ref":
            mutating = .asking("changes git refs directly (git \(sub))")
        default:
            return v + .asking("git \(sub) is not a git command this app knows to be safe")
        }
        if outside, mutating == .allow, sub != "init", sub != "clone" {
            return v + .asking("changes a repository outside the project (git \(sub))")
        }
        return v + mutating
    }

    static func gitMutating(_ sub: String, _ a: ArraySlice<ShellWord>, _ f: Set<String>, _ ctx: PolicyContext,
                            cwd: String?) -> Verdict {
        let ops = operands(a)
        switch sub {
        case "init":
            return ops.first.map { writing(path($0, cwd), ctx) } ?? .allow
        case "commit":
            return f.contains("--amend") ? .asking("rewrites the last commit (git commit --amend)") : .allow
        case "stash":
            let m = ops.first?.text ?? ""
            return ["drop", "clear"].contains(m) ? .asking("discards stashed changes (git stash \(m))", high: true) : .allow
        case "branch":
            if f.contains("D") || (f.contains("d") && (f.contains("f") || f.contains("--force"))) {
                return .asking("deletes a branch that may not be merged (git branch -D)")
            }
            return .allow
        case "tag":
            return f.contains("d") || f.contains("--delete") ? .asking("deletes a tag") : .allow
        case "worktree":
            let m = ops.first?.text ?? ""
            if m == "list" { return .allow }
            if m == "add", let p = ops.dropFirst().first { return writing(path(p, cwd), ctx) }
            return .asking("git worktree \(m)")
        case "bisect":
            return ops.first?.text == "run" ? .asking("git bisect run runs a command") : .allow
        case "reflog":
            let m = ops.first?.text ?? ""
            return ["expire", "delete"].contains(m) ? .asking("discards history (git reflog \(m))", high: true) : .allow
        case "restore":
            let staged = f.contains("S") || f.contains("--staged")
            let worktree = f.contains("W") || f.contains("--worktree")
            return staged && !worktree ? .allow : .asking("discards uncommitted changes (git restore)", high: true)
        case "checkout":
            if f.contains("f") || f.contains("--force") || a.contains(where: { $0.text == "--" })
                || ops.contains(where: { $0.text == "." || $0.glob }) {
                return .asking("discards uncommitted changes (git checkout)", high: true)
            }
            return .allow
        case "reset":
            if f.contains("--hard") || f.contains("--merge") || f.contains("--keep") {
                return .asking("discards uncommitted changes (git reset --hard)", high: true)
            }
            return .allow
        case "clean":
            if f.contains("n") || f.contains("--dry-run") { return .allow }
            return .asking("deletes untracked files (git clean)", high: true)
        case "config":
            let reads = ["--get", "--get-all", "--get-regexp", "--list", "l", "--show-origin", "--show-scope",
                         "--get-urlmatch", "--name-only"]
            let writes = ["--unset", "--unset-all", "--add", "--replace-all", "--edit", "e", "--rename-section",
                          "--remove-section"]
            let global = f.contains("--global") || f.contains("--system")
            if f.contains(where: { writes.contains($0) }) || (!f.contains(where: { reads.contains($0) }) && ops.count >= 2) {
                return .asking("changes git configuration, which can make git run commands", high: global)
            }
            return .allow
        case "remote":
            let m = ops.first?.text ?? ""
            return ["", "show", "get-url"].contains(m) ? .allow : .asking("changes git remotes (git remote \(m))")
        case "gc":
            return a.contains(where: { $0.text.hasPrefix("--prune") }) ? .asking("git gc --prune discards unreachable commits") : .allow
        case "archive":
            return value(of: "-o", in: a).map { writing(path($0, cwd), ctx) }
                ?? value(of: "--output", in: a).map { writing(path($0, cwd), ctx) } ?? .allow
        case "mergetool":
            return .asking("git mergetool runs an external tool")
        default:
            return .allow
        }
    }

    static func nodePackages(_ name: String, _ args: ArraySlice<ShellWord>) -> Verdict {
        if flags(args).contains("g") || flags(args).contains("--global") {
            return .asking("installs packages globally (\(name) -g)", high: true, grant: install)
        }
        let sub = args.first?.text ?? (name == "yarn" ? "install" : "")
        let network: Set<String> = ["install", "i", "ci", "add", "update", "up", "upgrade", "remove", "rm",
                                    "uninstall", "un", "link", "publish", "exec", "dlx", "create", "init", "x",
                                    "unpublish", "deprecate", "login", "adduser", "outdated", "audit", "view",
                                    "info", "pack", "version", "dedupe", "prune", "rebuild", "set", "config",
                                    "patch", "import", "fetch"]
        if network.contains(sub) {
            if sub == "config", args.dropFirst().first.map({ ["get", "list", "ls"].contains($0.text) }) == true { return .allow }
            return .asking("changes dependencies or uses the network (\(name) \(sub))", grant: net)
        }
        let known: Set<String> = ["test", "t", "tst", "run", "run-script", "start", "stop", "restart", "ls", "list",
                                  "why", "explain", "help", "root", "prefix", "bin", "--version", "-v", "build"]
        // yarn, pnpm and bun run a script by its bare name.
        return known.contains(sub) || name != "npm" ? .allow : .asking("\(name) \(sub)")
    }

    static func pip(_ name: String, _ args: ArraySlice<ShellWord>) -> Verdict {
        let sub = args.first?.text ?? ""
        return ["list", "show", "freeze", "check", "--version", "-V", "help", "inspect", "config"].contains(sub)
            ? .allow : .asking("changes Python packages or uses the network (\(name) \(sub))", grant: install)
    }
}

// MARK: - paths, physically

enum Paths {
    /// `.` and `..` removed by spelling alone.
    static func lexical(_ p: String) -> String {
        var out: [Substring] = []
        for part in p.split(separator: "/") {
            if part == "." { continue }
            if part == ".." { if !out.isEmpty { out.removeLast() }; continue }
            out.append(part)
        }
        return "/" + out.joined(separator: "/")
    }

    /// Where `abs` really points: the deepest existing prefix through
    /// realpath -- so a symlink, and `..` after one, resolve as the kernel
    /// resolves them -- and the rest by spelling.
    static func physical(_ abs: String) -> String {
        let parts = abs.split(separator: "/", omittingEmptySubsequences: true)
        var n = parts.count
        while n >= 0 {
            let prefix = "/" + parts[0..<n].joined(separator: "/")
            if let r = Darwin.realpath(prefix, nil) {
                defer { free(r) }
                let base = String(cString: r)
                let tail = parts[n...].joined(separator: "/")
                return lexical(tail.isEmpty ? base : base + "/" + tail)
            }
            n -= 1
        }
        return lexical(abs)
    }

    /// `p` is `dir` or under it; APFS is case-insensitive, and so is this.
    static func within(_ p: String, _ dir: String) -> Bool {
        let a = p.lowercased(), d = dir.lowercased()
        if d == "/" { return true }
        return a == d || a.hasPrefix(d + "/")
    }

    static func parent(_ p: String) -> String {
        let s = (p as NSString).deletingLastPathComponent
        return s.isEmpty ? "/" : s
    }

    static func base(_ p: String) -> String {
        p.split(separator: "/").last.map(String.init) ?? p
    }
}
