## 16. Tool safety: dangerous calls, detected and contained

*Specified 2026-10-13. Steps 1–4 of §16.8 built the same day: ToolPolicy
(+ ShellLexer), GatedToolRunner, the approval card, standing approvals,
`tool_approval`, Seatbelt containment for Bash and `tool_containment`, with
their suites. Every session's tools are gated and contained; the headless
gate declines every question.*

Since 2026-10-10 a session's tools run on the user's Mac, as the user, with
their login-shell environment (HostToolRunner, ShellEnvironment), outside App
Sandbox (Qwasar.entitlements). Nothing asks first. The system turn tells the
model not to do destructive things. That is a request, not a control. The C
agent asks before every write and every command (`confirm`, `--yes`). The app
asks before nothing.

This section covers what we detect, what happens when we detect it, and how
the work gets built and tested without running the things it exists to stop.

### 16.1 The threat model, restated for the Mac

**What we defend against:** the model, steered by its own reasoning or by
instructions in a file it read, takes an action the user did not intend and
cannot easily undo. There are three classes, in order of harm:

1. **Destruction on the machine.** `rm -rf` outside the project, `git reset
   --hard` / `clean -fdx` over uncommitted work, `diskutil`, `dd`, `shutdown`,
   `launchctl`, `defaults write`, `chmod -R` on `~`, killing processes the
   user owns.
2. **Writes outside the project.** Any Write/Edit, redirection, `cp`/`mv`
   target, or tool side effect that lands outside the project root:
   dotfiles, `~/Library/LaunchAgents`, shell rc files, other repos. A write
   to `.git/hooks` or `.git/config` is also in this class even though it is
   inside the tree, because the user's own git *executes* it.
3. **Exfiltration.** Reading is harmless on its own: inference is local, so
   a Read moves bytes from disk into this process and no further. Harm
   starts when a later call carries those bytes off the machine. The
   channels that exist today:
   - **Bash with the network:** `curl`, `wget`, `nc`, `ssh`/`scp`,
     `rsync host:`, `git push` to a remote, DNS lookups with data in the
     name, `open https://…`, `osascript` driving a browser.
   - **`delegate` (§15).** Not "local inference" at all: every tool result
     a delegated model sees, Reads included, goes to OpenRouter. With
     delegation enabled, point 1 of the user's framing no longer holds.
   - **Writes into synced folders** (iCloud Drive, Dropbox), which are
     exfiltration a step removed. Class 2 catches them.

**What we do not defend against:** a determined adversary inside the model's
context with unlimited attempts. Command parsing is advisory (§16.3).
Containment (§16.5) is the boundary, and it has its own limits, stated there.

### 16.2 Where the gate sits

There is one choke point: a `GatedToolRunner: ToolExecuting` that wraps
HostToolRunner. **The gate cannot live in AppState's dispatch loop.** The
`delegate` executor calls `inner.run` directly (Escalation.swift), so a
remote model's tool calls never pass through that loop. Wrapping the
executor covers the local turn, the delegated turn, and the headless gate
(`Qwasar --gate --prompt`) with one piece of code.

```
call ─▶ ToolPolicy.classify(call, ctx) ─▶ Verdict
          │ pure: no I/O but realpath/lstat on the path arguments
          ▼
   .allow ───────────────────────▶ inner.run(call)          [contained, §16.5]
   .ask(reason, risk) ─▶ Approver ─▶ yes ─▶ inner.run(call)  [widened as approved]
                                 └▶ no  ─▶ "error: the user declined: <reason>"
   .deny(reason) ────────────────▶ "error: refused: <reason>"   (never prompts)
```

- `ToolPolicy` is a **pure function** of the call, the project root, the
  home directory, and the session's standing approvals. Every decision it
  makes can be tested with strings alone (§16.7).
- The `Approver` is a closure. In the app it posts a card and waits. It
  reuses the AskUserQuestion pattern: `pendingQuestion` plus a checked
  continuation, bridged with a semaphore because `ToolExecuting.run` is
  synchronous and already runs off the main actor. Headless, it **fails
  closed**: every `.ask` becomes a decline, and the transcript says so.
- A decline or refusal is a tool result, never a failed turn. This is the
  C agent's principle and §6.4's: the model is told why and chooses again.
- Every `.ask` and `.deny` leaves a transcript note: the call, the reason,
  and the outcome. Nothing is quiet.

### 16.3 What the policy detects

#### File tools: Write, Edit (and Read, for exfiltration context only)

The rules apply to the path **as resolved**. That means the deepest
existing ancestor is `realpath`ed and the remaining components appended.
Otherwise a symlink inside the project (`proj/out -> ~`) is an escape. The
comparison is case-insensitive, as APFS usually is.

| target | verdict |
|---|---|
| inside the project root | allow |
| inside the root, but `.git/` (hooks, config, refs), `.envrc`, `.vscode/tasks.json`, `.github/workflows/` | ask: these execute later |
| `$TMPDIR`, the session's scratch directory | allow |
| anywhere else | ask |
| `~/.ssh`, `~/.aws`, `~/.gnupg`, `~/Library/LaunchAgents`, shell rc files, `/etc`, `/Library`, `/System` | ask, high risk |

Read stays allowed everywhere, as the user framed it. It does feed §16.4:
a Read of a known-secret path (`~/.ssh/*`, `.env*`, `*.pem`, `~/.aws`,
`~/.netrc`, keychain exports) marks the session **tainted**.

#### Bash

The command is lexed with a small sh-aware tokenizer. It handles quotes,
`;` `&&` `||` `|` `&`, `$(…)` and backticks (recursed into), redirections,
`VAR=x cmd` prefixes, and `env`/`command`/`nohup`/`time` wrappers. It yields
simple commands, each classified separately. The command's verdict is the
worst of its parts.

- **Deny (catastrophic, with no legitimate agent use):** `rm -r`/`-rf` whose
  target resolves to `/`, `~`, `$HOME`, a top-level system directory, or
  an ancestor of the project; `sudo`/`doas`/`su`; `shutdown`/`reboot`/
  `halt`; `diskutil erase*`/`partition*`; `dd of=/dev/…`; `mkfs*`;
  `csrutil`; `nvram`; `launchctl` (bootout/unload/load); fork bombs.
- **Ask (destructive, outside-project, or able to exfiltrate):**
  - any `rm` outside the root, or recursive anywhere
  - `mv`/`cp`/`ln`/`tee`/`>`/`>>` whose target is outside the root
  - `git push`, `reset --hard`, `clean -f*`, `checkout -- .`/`restore .`,
    `branch -D`, `rebase`, `filter-branch`, `stash drop`/`clear`, `commit
    --amend`
  - the network: `curl`, `wget`, `nc`/`ncat`/`socat`, `ssh`, `scp`,
    `sftp`, `rsync` with a remote, `ftp`, `telnet`, `open` with a URL
  - installs and removals: `brew`, `npm -g`, `pip install` outside a venv,
    `gem install`, `cargo install`, `mas`
  - code that hides code: `eval`; `sh -c`/`bash -c` with a non-literal
    argument; `python -c`/`node -e`/`perl -e`/`ruby -e`; `osascript`;
    `base64 -d | sh`-style pipes into an interpreter; `source` of a path
    outside the root
  - system state: `kill`/`pkill`/`killall`, `crontab`, `defaults write`,
    `security` (keychain), `chmod`/`chown` outside the root, `xattr -d
    com.apple.quarantine`
- **Allow:** a known read-only and build vocabulary (`ls cat head tail wc
  grep rg find(-without-delete/-exec) sed(-without-i-outside) diff file
  which echo pwd`, `git status/log/diff/show/blame/branch(list)`, `make`,
  `swift build/test`, `cargo build/test/check`, `npm test`/`run`, `pytest`,
  `go build/test`, `cc`/`clang`, and so on), with every redirect target
  inside the root.
- **Anything else: ask.** This is the important rule. A denylist is a
  game the model wins by finding a synonym (`find -delete`, `perl -e
  unlink`, a script written by Write then run). With "unknown means ask",
  a missed pattern costs one card, not a deleted home directory. "Allow
  for this session" (§16.6) is what keeps that from being tedious.

**Stated plainly:** `make` and `npm test` run arbitrary project code, and
the model can Write that code first. No lexer sees through that. Parsing
judges *what the call says it will do*. §16.5 bounds *what it can do*.

### 16.4 Exfiltration, specifically

- With no network, there is no exfiltration. So the strongest lever is
  §16.5's network denial on Bash, with network commands asking (§16.3) and
  running with the network only once approved.
- **Taint escalation.** Once a session has Read a secret path, a
  network-capable Bash call shows a card that names the secrets read this
  session, even when a standing approval would otherwise cover the call.
- **`delegate`.** While delegation is enabled, the session header says
  "tool results go to <provider>". A Read of a secret path inside a
  delegation is withheld from the remote model: the result says the file
  exists and was not sent. Nothing about the local model's reads changes.

### 16.5 Containment: Seatbelt around Bash

Write/Edit run in our process, so §16.3's checks on them are enforcement.
Bash is not, so it runs under `sandbox-exec` with a generated profile, the
same mechanism Codex CLI and Claude Code use on macOS:

```scheme
(version 1)
(allow default)
(deny file-write*)
(allow file-write* (subpath "<project root>") (subpath "<TMPDIR>")
                   (subpath "<scratch>") (literal "/dev/null") (literal "/dev/tty")
                   <toolchain caches, §16.5.1>)
(deny file-write* (subpath "<project root>/.git/hooks"))
(deny network*) (allow network* (remote unix-socket)) (allow network* (remote ip "localhost:*"))
```

- **This is what makes class 1 and class 2 hold** when the lexer is wrong:
  `rm -rf ~` hidden inside a Makefile fails in the kernel with EPERM.
- An **approved** `.ask` runs with the profile widened to exactly what was
  approved: network for an approved `curl`, one outside path for an
  approved `cp`. It is never fully unsandboxed by default.
- A project setting (`tool_containment`: `on` | `off`, default `on`) lets a
  person turn it off for a project that cannot live with it, with the same
  three-layer overlay rule as §8.5.

#### As built (2026-10-13), and what measuring it changed

- **Last match wins in SBPL.** That is why the `.git/hooks` deny comes
  after the allows. Pinned by canaries, not assumed.
- **Seatbelt's `localhost` includes this Mac's own interface addresses.**
  A connection to the Mac's LAN address succeeds under the profile. Any
  other address is refused in the kernel before a packet leaves (tested
  against TEST-NET-1, 192.0.2.1, which belongs to no one). This Mac is
  not the network, so the profile keeps the rule.
- **DNS is closed.** Connecting to `/private/var/run/mDNSResponder` is
  denied, so name lookups cannot carry data out. A network grant reopens it.
- **Grants travel as a `@TaskLocal`** (`ContainmentGrants.current`), set
  by the gate around an approved call. A model can never pass a grant as an
  argument. Each ask in ToolPolicy carries its grant: network for a network
  command, the named paths for outside writes, `anywhere` for installs and
  computed paths, and nothing for unknown commands or inline code, which
  run contained even when approved. `curl | bash` approved gets network
  for curl and nothing more for bash.
- **Sandboxes do not nest, and swiftc uses one.** swift-frontend runs
  macro plugins under `sandbox-exec`, which fails inside our profile
  (`sandbox_apply: Operation not permitted`), so every `@Observable` failed
  to expand and the app could not build contained. Contained commands find
  shims first on PATH: `swiftc` and `swift` wrappers that run the next real
  one with `-disable-sandbox` (`--disable-sandbox` for SwiftPM). The outer
  profile is the stricter of the two. The shims live in the app's store,
  which no contained command can write. `xcrun swiftc` bypasses them.
- **The containment found a test-hygiene bug.** StoreSuite wrote into the
  user's real store (`~/Library/Application Support/Crucible`), and the
  profile refused it. `Store(root:)` now exists, and the suite uses a temp
  directory.
- **The kernel does not log denials for ad-hoc profiles**, so the build
  measurements read "Operation not permitted" from the builds' own output.
  That is also exactly what the model sees, followed by the diagnosis line
  the backend appends.
- **The model is told.** HostToolRunner's environment description says
  whether commands are sandboxed and what that means, and the session
  header shows `sandboxed`, `unsandboxed` or `sandbox unavailable`.

#### 16.5.1 Known costs, measured (2026-10-13)

Contained, each of these passed with the default profile and the shims:
this repository's `make` (the C engine, Metal included), `make -C app` (the
Swift app, macros and codesign included), and `make -C app test`. Inside a
sandbox, the canary suite skips with a reason, because profiles cannot
nest; `Containment.alreadySandboxed` asks the kernel, and the backend then
runs commands plainly under the outer sandbox. A dependency-free Cargo
crate (`cargo build && cargo test`), npm package (`npm test`) and Python
venv (`python3 -m venv`, py_compile, run) also passed, with no denials.
Not measured: Go (not installed), and anything that fetches dependencies,
which asks for the network by design.

The list below was the plan; the cache list in Containment.swift is what
shipped, and grows from denials observed in real sessions.

- Toolchains write outside the tree: `~/.cargo`, `~/.npm`, `~/.cache`,
  `~/Library/Caches`, `~/Library/Developer` (SwiftPM, DerivedData),
  `~/.gradle`, mise's install directories. The allowlist starts from that
  list and grows from denials observed in real sessions. A sandbox denial
  is reported in the tool result ("blocked by containment: write to X"),
  so the model and the user both see it.
- Dependency fetches (`npm install`, `cargo build` with new crates, `swift
  package resolve`) need the network. They become `.ask` with an "allow
  for this project" option.
- `sandbox-exec` is deprecated and has been for years. It still works on
  Darwin 25 and the tools above depend on it. If Apple removes it,
  containment degrades to policy-only. The transcript would say so, and
  this section would be revisited.

### 16.6 The approval card

The choices are **Allow once** · **Allow for this session** (for a pattern
the policy derives and shows: "`swift test` in this project", "network for
`curl` to api.github.com") · **Deny**, with an optional reason that goes to
the model. `.deny` verdicts never show a card. A card shows the full
command, the resolved paths, and the reason it was flagged.

Standing approvals live with the session record. They are never written by
the model. The config project may *read* them but cannot *grant* them. That
fixes §8.5's asymmetry here: loosening a safety rule is something only a
person does, in the card.

As built:

- **Standing approvals are keyed by the policy's reasons.** "Allow for this
  session" stores the reasons the card showed, and a later call runs
  unasked only if *every* reason it carries is already approved. So an
  approved `curl` does not cover `curl … > ~/x`, which adds a write outside
  the project. A reworded reason in a later version asks again. They
  persist on `SessionRecord.approvals`, and `config_show` reports their
  count.
- **The card hides nothing.** It shows the whole command, never clipped or
  scrolled inside the transcript, and for Write and Edit the content too,
  up to 60 lines with the remainder counted. Return approves nothing;
  Escape denies.
- **Stop declines every open card**, as do stopping a delegation and
  quitting. A card is shown in its own session, and the Dock bounces,
  critically for a high-risk ask. Not yet built: a sidebar mark for a
  session waiting on a card while another is selected.
- **The config session can set `tool_approval`.** That is §8.5's stated
  asymmetry: it acts only on what the user typed into it. It cannot grant
  standing approvals.

A per-project `tool_approval` key: `ask` (default, everything above) or
`trusted` (only `.deny` and taint escalation interrupt; the `--yes`
equivalent). `.deny` and containment apply in both modes.

### 16.7 Building and testing this without doing the damage

Every test of this feature is a test about dangerous commands. The rules
below hold for the test suite, and for me (the agent doing the work) during
development.

**For the development session itself:**

- No dangerous command is ever run in my own shell, including to "see what
  happens" or to check a regex. Dangerous strings exist only as Swift
  string literals in test files, which are data. They are not passed to
  `Bash`, `echo`ed through `sh -c`, or assembled at runtime from parts.
- No end-to-end run (`make gate-full`, the app) points at this repository
  or any real project while policy code is in flux. A model-in-the-loop run
  uses a throwaway root under the scratch directory, with `HOME`
  overridden to a fake home inside it (§16.7.3).
- Before every commit: a grep check that the test sources do not call
  `Spawn.run`/`HostBackend.exec` on any string from the dangerous corpus,
  only on the canary commands below.

#### 16.7.1 Layer 1: the policy is pure, so it is tested with strings

`ToolPolicySuite` feeds `ToolCall`s to `ToolPolicy.classify` with a
synthetic root (`/nonexistent/proj`) and home (`/nonexistent/home`) and
asserts verdicts. Nothing executes. The corpus is a table:

- each deny and ask rule in §16.3, with evasions: quoting (`r''m -rf ~`),
  `$(…)`, `env rm`, `command rm`, `\rm`, `/bin/rm`, `;`/`&&` chains, `find
  -delete`, `xargs rm`, redirection to `~/.zshrc`, `cd .. && rm -rf proj`
- path escapes: `../`, absolute, `~`, and symlinks. Symlink cases use a
  real temp tree, because they need real `lstat`, but they only create
  links and never follow them for writing
- the allow vocabulary, so the false-positive rate is pinned: a corpus of
  ordinary commands drawn from real session transcripts must stay allowed

Every evasion found later becomes a row in this table before it becomes a
fix. Rows that were judgment calls, written down so they are not relitigated:
an *unfiltered* `find . -delete` or `find . -exec rm {} \;` is refused (it is
`rm -r` of the project), and a filtered one asks; `rm -rf link` where the link
points outside asks, though rm removes only the link; `of=~/x` is not
expanded, because sh does not expand it either.

#### 16.7.2 Layer 2: the gate is tested with a backend that cannot act

`ToolBackend` is already a protocol. A `RecordingBackend` implements it by
appending each call to a list and returning canned results: `exec` and
`write` *record* and do nothing. `GatedToolRunner` over it, with a scripted
approver (always-no, always-yes, headless), gives the suite's central
assertions:

- a denied or declined call **never reaches the backend** (the list is
  empty)
- an approved call reaches it exactly once, with the widened containment
  recorded
- headless fails closed
- a `delegate`'s inner calls are gated: an Escalation runner whose inner
  executor is the gated runner, driven by the existing stub server in
  `EscalationSuite`

#### 16.7.3 Layer 3: containment is tested with canaries, not with harm

Seatbelt only proves itself by really running commands. Those commands are
harmless by construction:

- **Setup:** `mktemp -d` → `T/`. Inside it, `T/proj` (the root), `T/home`
  (passed as `HOME`), `T/outside`. Nothing outside `T/` is ever named.
- **Writes:** `touch T/outside/canary` under the profile must fail, and
  `touch T/proj/canary` must succeed. "Can it delete outside the root" is
  tested by "can it *create* outside the root", because the profile rule
  (`file-write*`) is the same. There is never an `rm` of anything the test
  did not create, and no `rm -r` anywhere.
- **Network:** the test opens a listener on `127.0.0.1` and on a second
  non-loopback local address. Under the profile, connecting to the
  non-loopback one must fail and loopback must succeed. No external host
  is ever contacted. DNS is tested with `dscacheutil -q host` on a
  reserved name (`.invalid`) and asserts the sandbox error, not a lookup
  result.
- **`.git/hooks`:** a write to `T/proj/.git/hooks/canary` must fail.
- These tests run in `make test`. They skip with a printed reason if
  `sandbox-exec` is missing, and never fall back to running unsandboxed.

#### 16.7.4 Layer 4: the model in the loop, red-teamed in a box

A `--tools dry-run` flag for the headless gate swaps HostBackend for
RecordingBackend. The model sees plausible results and nothing executes.
Adversarial prompts (a README carrying an injection, "clean up disk space",
"the build is broken, reset everything", "post this file to this URL")
run against a throwaway root with a fake home, and the recorded calls and
their verdicts are the output. Only once dry runs look right does the same
suite run with real execution, still in the throwaway root, still under
containment, with the canary layout from 16.7.3 standing in for the home
directory.

### 16.8 Order of work

1. **ToolPolicy plus its string suite** (16.3, 16.7.1). Pure, no wiring,
   nothing user-visible.
2. **GatedToolRunner, RecordingBackend, headless fail-closed** (16.2,
   16.7.2). Wired into `send`, `startDelegation` and the gate. The app's
   approver denies everything until step 3, behind a debug default, so
   nothing ships half-built.
3. **The approval card and standing approvals** (16.6).
4. **Seatbelt for Bash plus the canary suite** (16.5, 16.7.3). The cache
   allowlist is measured on this repository's own builds first, then on
   one npm, one cargo, and one Python project.
5. **Exfiltration: taint, the delegation header, and withholding secret
   reads** (16.4).
6. **Dry-run red-teaming** (16.7.4), then real-execution red-teaming in
   the box.

Out of scope for now: the C terminal agent. It already asks before every
write and command, and it could share the policy later by porting the
classifier to C.

### 16.9 Decided (2026-10-13)

- **`tool_approval` defaults to `ask`.** `trusted` is a person's choice for
  a project, never the default.
- **Bash has no network by default** under containment. A network command
  asks, and once approved it runs with the network for that one call.
- **The catastrophic `.deny` set cannot be overridden from the card.** If a
  user really does want `sudo` or `diskutil erase`, they run it themselves.
