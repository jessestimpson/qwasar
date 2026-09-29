## 3. Architecture

### 3.1 Processes

```
┌─ Crucible.app (one process) ──────────────────────────────────┐
│                                                               │
│  SwiftUI  ──▶ SessionStore (@Observable, MainActor)            │
│                    │                                          │
│                    │  AsyncStream<Delta> / continuations       │
│                    ▼                                          │
│  ┌─ EngineHost ─ serial DispatchQueue, one thread ─────────┐  │
│  │   qwasar_engine  (immutable after load, ~16 GB mmap)     │  │
│  │   qwasar_session × live-session budget (§2.3; 1 here)    │  │
│  │   the agent loop: prefill, decode, MTP verify, parse     │  │
│  └──────────────────────────────────────────────────────────┘  │
│                    │                                          │
│                    │  ToolRequest / ToolResult (JSON)          │
│                    ▼                                          │
│  ┌─ SandboxHost ─ one per session ─────────────────────────┐  │
│  │   VZVirtualMachine  ·  vsock  ·  virtiofs                │  │
│  └──────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────┘
                     │ AF_VSOCK, port 1024
┌────────────────────▼──────── guest: Alpine arm64 ─────────────┐
│  warden    (Elixir)  immutable control plane, owns the bridge  │
│     │ Erlang distribution over loopback                        │
│  workspace (Elixir)  the agent's node — hot-loadable, restarts │
│  /work   virtiofs, read-write: the user's OWN tree — edits     │
│          land there live, as uncommitted changes               │
│  /var/crucible/git   guest disk: a private copy of .git,       │
│          bind-mounted over /work/.git every boot (§7.4)        │
└────────────────────────────────────────────────────────────────┘
```

Two rules this diagram encodes:

- The **host never runs a tool**. Not `bash`, not `write`. If the model wants a
  command run, it runs in the guest.
- The **guest never writes the real `.git`**. The working tree is the agent's
  to edit — that is the product — but hooks, config, and history sit behind
  the bind-mounted shadow, out of reach of everything the tools do (§7.4).
- **Two nodes, not one**: warden must survive anything the agent does, so
  the agent's code never runs on the node that owns the wire (§7.3).

### 3.2 Directory layout

```
crucible/
  PLAN.md                     this document
  Makefile                    builds, signs, tests (§3.3 -- no .xcodeproj)
  Crucible.entitlements       what ships
  Crucible.gate.entitlements  TEST ONLY: adds read-only path exceptions so the
                              gate and the agent loop are provable headless
  Resources/Info.plist
  Sources/
    CQwasar/module.modulemap  two headers, deliberately (§3.3)
    CrucibleKit/              all logic; no SwiftUI -- enforced by being its own
                              Swift module, not by convention
      Bridge.swift            the five sharp edges (§3.4), each handled once
      Tokenizer.swift         the tokenizer alone, no engine: a golden test
                              costs a JSON parse, not 16 GB of mmap
      ChatTemplate.swift      qwasar_message marshalling + the continuations
      EngineHost.swift        the serial queue, engine lifecycle, session map
      Session.swift           one qwasar_session; generate/parse/dispatch/continue
      Tools.swift             the M1 read-only tools + the qw_tool_parse bridge
      ToolSurface.swift       schemas, verbatim from qwasar_agent.c
      PathGuard.swift         path confinement (§7.4 step 8, rehearsed early)
      MemoryProfile.swift     §2.3, made executable
      Model.swift  Store.swift  ModelAccess.swift  Diagnostics.swift  GateCheck.swift
    Crucible/                 the app: SwiftUI and AppKit only
      Main.swift              entry point; --gate runs the milestone gates headless
      CrucibleApp.swift       window, sidebar, toolbar
      SessionView.swift       transcript, tool cards, prefill bar, composer
      AppState.swift          the one place both worlds meet
  Tests/
    gen_golden.c              the C oracle for the chat template
    golden.tsv                committed goldens
    TestMain.swift            `make test`
    GoldenSuite / PathGuardSuite / StoreSuite / UTF8Suite
```

Still to come, in the milestone that needs them: `Sandbox/` (M2-M5),
`Materialise.swift` (M5), and `crucible-cli` (M6, when there is a scripted
session worth driving).

### 3.3 Linking the C runtime

Add one target to the parent `Makefile`:

```make
libqwasar.a: $(CORE_OBJS)
	ar rcs $@ $^
```

`CORE_OBJS` is already exactly the right set — engine, graph, kvstore,
tokenizer, toolcall, sample, json, cpu, image, video, vision, metal — and
excludes `qwasar_cli.o`, `qwasar_agent.o`, `qwasar_server.o`, `qwasar_tui.o`,
and `linenoise.o`, which are the three front ends and their terminal
plumbing. Nothing in the app should ever pull those in; if a symbol from
`qwasar_tui.o` is needed, that is a bug in the app, not a missing dependency.

**And there is no `.xcodeproj`.** This section originally assumed an Xcode app
target with a Run Script phase shelling out to `make`. On contact the whole
application builds with `swiftc` and a hand-assembled bundle, in a `Makefile`
short enough to read in one screen — which is both less machinery and a better
fit for the parent's rule that the build is one `make` with no codegen you have
to know about. `crucible/Makefile` does four things:

1. `make -C .. libqwasar.a`, with `MACOSX_DEPLOYMENT_TARGET` passed through
   (see below). The C tree keeps its own build; nothing here learns about
   `bin2c` or the embedded Metal source.
2. Compiles `CrucibleKit` as a **separate Swift module** into a static archive.
   That is what makes "CrucibleKit contains no SwiftUI" (§3.2) a compiler error
   rather than a convention — SwiftUI is linked only into the app target.
3. Compiles the app against it and links: `-lqwasar`, the parent's five
   frameworks, plus `Virtualization`, `SwiftUI`, `AppKit`, `Security`.
4. Assembles `Crucible.app`, copies `Info.plist`, and `codesign`s with
   `Crucible.entitlements` and the hardened runtime.

**Deployment target, measured.** The parent's `CFLAGS` carry no
`-mmacosx-version-min`, so clang defaults to the host SDK — 26.0 here — and the
linker correctly complains that an app claiming a macOS 14 floor is linking
objects that require 26. The engine compiles clean at 14.0, so `crucible/Makefile`
passes `MACOSX_DEPLOYMENT_TARGET=14.0` through to the parent make. This is a
widening change and affects no machine that could run the parent binaries
anyway, but it is a change to shared objects and is recorded here rather than
left as a surprise.

The `CQwasar` module map is deliberately narrow — the same discipline the
parent applies to `qwasar.h`:

```
module CQwasar {
    header "qwasar.h"
    header "qwasar_toolcall.h"
    export *
}
```

`qwasar_json.h`, `qwasar_gpu.h`, and `qwasar_model.h` are **not** exported.
Swift code that wants to parse JSON uses `Codable`. Swift code that wants to
know what a tensor looks like is doing something wrong.

**Metal kernels need no special handling.** They are embedded in
`qwasar_metal_src.inc` and compiled at runtime, so the `.app` is self-contained
and nothing has to be copied into `Resources/`. The pipeline binary archive
that `qwasar_metal.m` maintains will land in the app's container rather than
the CLI's; that is fine, and it means first launch pays the kernel compile once.

### 3.4 The five sharp edges of the bridge

These are the places where a Swift/C mistake is silent rather than loud. All
five are handled exactly once, in `CrucibleKit/Bridge.swift`, so no call site
has to remember. The fifth was added in M0, after it cost two debugging cycles.

**Logits are borrowed, not owned.** `qwasar_session_eval` returns a pointer
"valid until the next eval on this session". Sample from it immediately or copy.
Never store it, never let it cross a suspension point.

```swift
let logits = qwasar_session_eval(s, tokens.baseAddress, Int32(tokens.count),
                                 &err, err.count)
guard let logits else { throw EngineError(err) }
let next = argmax(logits, vocabSize)   // before anything else touches `s`
```

**`qwasar_encode` returns malloc'd memory.** So do
`qwasar_apply_chat_template`, `qwasar_render_tool_result`, and
`qwasar_render_user_turn`. Every one is a `free()` the caller owes. Wrap each in
a Swift function that copies into `[Int32]` and frees in a `defer`, and never
let a raw pointer escape that function.

**The progress callback fires on the engine thread.** It is a C function
pointer with a `void *`, so it cannot capture:

```swift
let cb: qwasar_progress_fn = { ud, done, total in
    guard let ud else { return }
    Unmanaged<ProgressBox>.fromOpaque(ud)
        .takeUnretainedValue()
        .report(done: Int(done), total: Int(total))   // hops to MainActor itself
}
qwasar_session_set_progress(s, cb, Unmanaged.passUnretained(box).toOpaque())
```

`passUnretained` means the box must outlive the session. Own it on the session
wrapper, and clear the callback in `deinit` before `qwasar_session_free`.

**Autorelease pools.** `qwasar_metal.m` is ARC and `qw_cmd_begin` uses
`@autoreleasepool` internally, but a decode loop that runs for thousands of
iterations inside one Swift call still accumulates whatever the Metal layer
autoreleases at the outer scope. Wrap the loop body:

```swift
for step in 0..<maxTokens {
    try autoreleasepool { ... }
}
```

**The main actor is not a place to wait.** Not a C boundary at all, but it bit
twice in M0's first hour and both times presented as a silent hang with no
output:

- `VZVirtualMachine` binds to a serial queue that **defaults to the main
  queue**, and its initialiser reaches that queue before returning. Constructing
  one from a `Task` while the main thread blocks on a semaphore deadlocks.
- Swift's `@main` entry point is implicitly `@MainActor`, so a plain
  `Task { … }` created there **inherits main-actor isolation** and cannot run
  while the main thread is blocked. `Task.detached` does not inherit it.

It happened a **third** time in M3, in the agent path rather than the gate path:
`SandboxManager` is `@MainActor` because `VZVirtualMachine` has main-queue
affinity, so the moment a detached task tried to boot a guest it needed a main
thread that was sitting in a semaphore. Ten minutes at 0% CPU with 5.1 GB
resident — indistinguishable, from outside, from a slow model.

So the rule is now unconditional and is not a guideline about Virtualization:

> **Nothing blocks the main thread, in any mode, ever.** Not the GUI, not the
> gate, not a headless driver. Every entry point keeps a live run loop and exits
> from inside its task.

`Task.detached` escapes main-actor *inheritance*, which is necessary and not
sufficient: the work it does still reaches back to the main actor the moment it
touches anything with main-queue affinity. Escaping the isolation does not
escape the dependency. The engine queue is off-main by construction (§2.1),
which is why generation was never affected — it was always the driver around it.

**Token bytes are not strings.** `qwasar_token_bytes` returns raw UTF-8 that may
end mid-codepoint — a single token is frequently half a character. Accumulate
into a `[UInt8]` buffer and emit `String(decoding:as:)` only on a valid boundary,
or the transcript will show replacement characters mid-word. The C TUI gets away
with writing bytes straight to a terminal; SwiftUI will not.

### 3.5 What we reuse from `qwasar_agent.c`, and what we do not

**Reuse, through the C bridge, unchanged** (all three now in
`CrucibleKit/Tools.swift`):

- `qw_tool_parse` / `qw_tool_calls_free` / `qw_tool_arg` — the XML tool-call
  parser. Pure string processing, no I/O, already covered by
  `tests/test_toolcall`. Reimplementing this in Swift would be duplicating a
  tested parser for the format the model was trained on. Don't.
- `qw_tool_call_complete` — the early-stop predicate that ends generation at the
  closing tag instead of running to the token budget.
- The **shape** of `generate()`: the reasoning/text split at `</think>`, the
  `in_call` suppression so tool markup is not echoed as prose, and the
  draft/verify structure in the speculative path. `Speculation.swift` will mirror
  it line for line when M7 adds it, because the guarantee it protects —
  identical output with or without the MTP head — is worth preserving exactly.

  One detail from `take_token` that looks like an oversight and is not,
  reproduced deliberately in `Session.generate`: when a tool call completes, the
  loop breaks **before evaluating** the token that completed it. That token is
  accumulated for the parser but never enters the KV cache — which is right,
  because the continuation renderer closes the assistant turn and the closing
  tag is markup the model does not need to see again. Diverging here would put a
  token in the cache the C agent never puts there.

**Do not reuse:**

- `qw_edit_apply`. File editing now happens inside the guest, in Elixir, against
  `/work`. The host's only write path is patch application (§7.4), which has a
  different and stricter contract.
- Everything in `agent_cfg`, `dispatch`, `tool_bash`, `tool_write`,
  `confirm`. These run tools on the host. That is precisely what Crucible does
  not do.
- `qwasar_tui.c` and `linenoise`. Obviously.

---
