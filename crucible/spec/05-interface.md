## 5. The interface

### 5.1 Shape

`NavigationSplitView`, three columns, the shape every macOS user already knows:

- **Sidebar** — Projects, each expanding to its Sessions. Session rows carry a
  state dot: hot (filled), parked (hollow), queued (pulsing), awaiting approval
  (amber). Badge for unread completion when the app is backgrounded.
- **Content** — the transcript.
- **Inspector** (toggleable, ⌥⌘I) — the sandbox: VM state, `/work` tree, the
  registry of agent-defined tools, and a live log tail from the warden. This
  panel is how a person understands what the agent did to itself, and it is not
  a debug affordance — it is the product.

### 5.2 The transcript

Items, not a text stream:

| item | rendering |
|---|---|
| user turn | plain, with attachments (images/video — the runtime already supports both) |
| reasoning | collapsed by default, dimmed, expandable; token count shown when collapsed |
| assistant text | markdown, streamed |
| tool call | a card: tool name, arguments (long values elided to one line), status |
| tool result | inside the call's card, collapsed past 3 lines — the same cut `TOOL_RESULT_LINES` makes in the TUI, for the same reason |
| module load | a distinct card: module name, diff against the previous version, purge outcome (§7.3) |
| patch proposal | a card that opens the approval sheet (§7.4) |
| turn end | dim footnote: tokens, tok/s, spec acceptance, context used |

The model **always** reasons — this is a thinking model with `enable_thinking`
on by default — so reasoning is a first-class item with its own affordance, not
an oddity to hide.

### 5.3 Streaming

The engine thread produces `Delta` values; the view consumes an `AsyncStream`.
Two rules:

- ~~**Coalesce at ~30 Hz.**~~ **Not built, deliberately (M1).** At a measured
  5.8 tok/s decode the raw event rate is about six a second, and prefill reports
  once per 256-token chunk — roughly once every eight seconds. There is nothing
  to coalesce, and a timer that batches six events a second would be machinery
  earning nothing. Streaming appends into the tail transcript item, so SwiftUI
  re-lays out one row rather than the list.

  Revisit when M7 turns on speculation: a verify settling four tokens in one
  pass is the first thing here that produces a burst.
- **Emit on UTF-8 boundaries only** (§3.4).

### 5.3a A call being written is not a stall

The markup of a tool call is never echoed. That is right — a call is markup
rather than prose, and printing it would bury whatever narration the model wrote
before it (§3.5). But it left the *longest* stretch of many turns showing nothing
at all: a `write` or a `define` runs to hundreds of tokens, which at ~6 tok/s is
minutes of a window that looks identical to a wedged one.

Two separate faults, found together:

**Nothing was emitted while `inCall` was true.** No text, no reasoning — the only
events were the periodic context and rate updates, and those carry no sign that
anything is being *built*.

**The rate itself reported in bursts.** It fired on `tokens.count % 64`, which is
eleven seconds of silence between updates at this decode speed, and worse once
speculation landed: a round advances the count by up to nine, so the modulo can
step straight over its own trigger and go quiet for a hundred tokens. It is now
a delta — every 8 tokens, about a second — which no stride can skip.

What is shown is what can be known early and honestly. The name arrives inside
the first line of the call (measured: 28 characters) and the parameter keys
follow one at a time, so `ToolParser.partial` reads those from the tail of the
buffer and the row says *what* is being built and how far along it is. The
values are not shown, because the finished ToolCard shows them and showing them
twice is worse than showing them once.

The property that matters is asserted rather than assumed: feeding **every
prefix** of a real call, no prefix may report a name other than the true one or
invent a key that is not the call's. Saying nothing yet is honest; saying
`bash` while the model writes `write` is not.

### 5.4 Prefill progress is not a spinner

`qwasar_session_set_progress` reports once per chunk, and the C agent's own
comment explains why it exists: prompt processing is the one part of a turn with
no visible output, and on a long prompt it is the longest part. A cold 8K-token
prompt is tens of seconds.

So: a real determinate bar with token counts and an ETA derived from the
observed chunk rate, plus the reason — `restored 6144 from cache`,
`prefilling 2048`. The C agent prints exactly this and it is the difference
between "thinking" and "hung". Suppress it below 128 tokens (`AGENT_BAR_MIN_TOKENS`),
because a bar that flashes is worse than none.

### 5.6 Markdown in the transcript, without a dependency

The assistant's text is rendered today as `Text(t)` — the raw characters,
asterisks and backticks and all. A model that writes fenced code blocks and
bulleted lists into a window that shows them literally is being misread by its
own interface.

The instinct is to reach for a Swift package. **It is not needed, and the reason
is worth writing down so nobody adds one later.** Foundation parses CommonMark
*and* GFM already, and `AttributedString.MarkdownParsingOptions` with
`interpretedSyntax: .full` returns the block structure as `presentationIntent`
attributes on the runs. Verified against a real parse rather than assumed:

    [header 1]                        Heading one
    [paragraph inline:bold]           bold
    [paragraph inline:code]           inline code
    [paragraph link]                  link
    [codeBlock 'swift']               func f() -> Int { 42 }
    [paragraph listItem 1 unorderedList]  first item
    [paragraph listItem 2 orderedList]    ordered two
    [paragraph blockQuote]            a block quote
    [tableCell 0 tableHeaderRow table [.left, .left]]  a

Everything a coding assistant emits, including the one that matters most —
**fenced code blocks carry their language hint**, `codeBlock 'swift'`, so the
label and any later highlighting have something to key on. Tables arrive with
per-column alignment.

So the parse is free and the work is **rendering**: walk the runs, group
consecutive ones by the identity of their innermost block, and emit a view per
block. Grouping by `presentationIntent.components.first?.identity` reassembles a
paragraph that inline formatting split into six runs, which is the only subtle
part of the traversal.

That matters more here than in most apps. This project links a C engine, builds
its own guest image and signs its own bundle from a Makefile with no
`.xcodeproj` and no SwiftPM manifest (§3.3). Adding a package would mean adopting
SwiftPM for the whole app or vendoring a renderer — a structural change to the
build, to render text the system already knows how to parse.

**Cost, measured:** a 4 KB document parses in **2.0 ms**. Streaming re-renders on
every delta at ~6 tok/s, so a full re-parse per token is about 1% of a core and
needs no incremental machinery. If a very long message makes that visible, the
fix is to re-parse only the tail block, not to cache the whole tree.

**An unclosed fence during streaming is not a bug to fix.** A code block being
typed has no closing ``` yet; `failurePolicy: .returnPartiallyParsedIfPossible`
renders it as a code block that grows, which is what the user wants to see
anyway.

#### What renders, and what deliberately does not

  - **Assistant text** — markdown. This is the whole point.
  - **Reasoning** — raw and monospaced, as now. It is a view into what the model
    was doing, not a document it wrote for a reader, and formatting it would
    imply an intent that is not there.
  - **User turns** — plain. The user typed those characters and should see them.
  - **Tool results** — raw and monospaced. They are program output; asterisks in
    a grep result are asterisks.

#### Code blocks earn their own affordances

Long lines **scroll inside the block**, never widen the transcript — a window
that grows horizontally because the model emitted one long line is a window that
is now wrong for everything else in it. The language, when the fence gave one, is
shown. A copy button, because copying a snippet out of a transcript is the single
most common thing a person does with one.

#### Syntax highlighting: highlight.js, in JavaScriptCore

Doing this properly means a grammar per language; doing it badly means a regex
that mis-colours the user's own code. So it is neither hand-rolled nor a package
dependency — it is **highlight.js, vendored, running in JavaScriptCore**, which
is a system framework. Nothing for a user to install, which is the bar the rest
of the project holds.

Both halves of that already have precedent here. `vendor/stb_image.h` is
third-party source carried in the tree. `tools/bin2c` embeds the Metal kernels
as a string precisely so the shipping binary needs no separately-installed
toolchain. This is the same trade for the same reason, with the pieces kept
separate and reviewable and the single artefact generated by `build.sh` rather
than committed.

Measured before choosing it, not after: **192 languages register, the bundle
evaluates in 53 ms once, and a code block highlights in ~1.4 ms.** Swift,
Elixir, Erlang, C, Objective-C, bash, Python, JSON, Rust and Makefile all come
out right — including Elixir sigils with interpolation, bash heredocs and Rust
raw strings, which are exactly the cases the hand-rolled alternative gets wrong.

Three decisions inside it matter more than the choice of engine:

**The spans are scanned into ranges here; nothing ever interprets model output
as markup.** hljs returns `<span class="hljs-keyword">…</span>` with entity
escapes, and the obvious shortcut — `NSAttributedString(html:)` — is slow, drags
in a full HTML parser, and means handing untrusted model output to something
whose job is to interpret markup. A ~60-line scanner turns the spans into
`(range, class)` pairs and decodes the five entities hljs emits. Nothing else.

**A block with a language hint highlights on every completed line; a block
without one waits for the fence to close.** Both halves of that were measured,
and the first overturned an earlier draft of this section which said to wait for
the close in all cases.

*Cost is not the constraint.* Highlighting is linear at ~0.06 ms/line, and
re-highlighting the whole prefix on each new line is quadratic — but decode runs
at ~6 tok/s, so the denominator is enormous:

| block | total CPU re-highlighting every line | over | share of one core |
|---|---|---|---|
| 37 lines | 51 ms | ~74 s of streaming | 0.07% |
| 109 lines | 380 ms | ~218 s | 0.17% |
| 361 lines | 4.5 s | ~722 s | 0.62% |

The earlier reasoning — "1.4 ms per token is real work" — was per-token cost
without the rate it is divided by. A guard still belongs at a few thousand lines,
where the quadratic finally bites, and past it the block waits for its close.

*Stability is the constraint, and it has a sharp edge.* Re-highlighting a growing
prefix is safe **only if the in-progress tail line is excluded**. Highlight
everything up to the last newline and render the partial line plain: across
Python docstrings, C block comments, bash heredocs, Swift multiline strings,
Elixir sigil heredocs and Rust raw strings — every construct that spans lines and
could plausibly be read differently in fragment — **not one settled line ever
changed colour**. A half-typed token would flash, which is why the tail is
excluded rather than included.

*Auto-detection is the exception, and it is not close.* With no language hint,
hljs must guess, and its guess is unstable on a fragment. Measured across four
blocks, **every one flipped**:

    python -> cpp -> cpp -> cpp -> stata
    livescript -> livescript -> swift -> swift
    elixir -> ruby -> ruby -> ruby -> ruby
    fortran -> pgsql -> pgsql -> pgsql

A language flip recolours the entire block, not one line. So a bare fence stays
plain monospace until it closes, and is detected and highlighted exactly once.
The model writes the language on its fences nearly always, so this is the
uncommon path — and when it is taken, plain-until-done is honest rather than
wrong.

**The class names map to our palette, not hljs's stylesheet.** A theme that does
not follow the app into dark mode is worse than no theme. Unknown language, no
language hint, or hljs throwing all fall back to plain monospace — which is what
the transcript does today, so the failure mode is the current behaviour rather
than a broken one.

### 5.5 Session creation

New Session asks two things: which project, and which directory inside it. The
second defaults to the project root and is a folder picker constrained to the
root — a session cannot escape its project, and the sandbox will not carry
anything the picker did not select.

---
