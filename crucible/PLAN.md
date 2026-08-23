# Crucible — a native macOS coding harness on the qwasar runtime

`Crucible` is a SwiftUI macOS application that links `libqwasar` **directly** —
no HTTP, no `qwasar-server`, no subprocess — and gives the model an agent loop
whose tools do not run on your machine. They run on a **BEAM (Erlang/Elixir)
node inside a macOS Virtualization.framework guest**, where the model is allowed
to hot-load Elixir and Erlang modules and thereby rewrite its own tool surface
mid-task. The session's working directory is copied in; nothing comes back out
except a patch, and only with your approval.

> **Codename.** `Crucible` is provisional — a vessel you melt things in, sealed
> from the room. Rename before the first commit that ships a bundle identifier.

Two things make this different from a normal harness, and both come from the
same idea:

1. **The engine is in-process.** `qwasar_session_eval` is called from the app's
   own address space. There is no serialisation boundary, no token re-encoding,
   no localhost socket, and — the part that actually matters — no reason to
   throw away a session's KV cache and recurrent state between turns.
2. **The tools are in a VM, and the model owns them.** A conventional harness
   ships a fixed tool set written by the harness author. Here the fixed set is
   ten calls wide and one of them is `invoke`; everything else the model needs,
   it writes in Elixir, hot-loads into a live node, and calls. The blast radius
   of that freedom is exactly one virtual machine with no network device.

---

## 1. Scope

### 1.1 What we are building

- A standard, double-clickable, notarisable `.app`. `Crucible.app` in
  `/Applications`, launched from the Dock, quit with ⌘Q. Not a CLI with a
  window bolted on.
- A chat harness with **Projects** (a name, a root directory, settings) and
  **Sessions** (a conversation, an initial working directory inside a project,
  a sandbox, a transcript that survives relaunch).
- An agent loop written in Swift on top of `qwasar.h` and `qwasar_toolcall.h`.
- A per-session Linux guest running two BEAM nodes, reachable over vsock.
- A self-modification tool surface: define an Elixir module, load it into the
  running node, register it as a tool, call it on the next step.
- A materialisation step: diff the sandbox against its baseline, show the user,
  apply to the real tree on approval, with undo.

### 1.2 What we are not building

- **Not a server.** `qwasar-server` already exists and stays. Crucible does not
  speak HTTP, does not expose a port, and does not need one.
- **Not multi-model.** Same position as the engine: one model, done properly.
- **Not iOS/Catalyst.** Virtualization.framework is macOS-only on Apple silicon,
  and so is the whole premise.
- **Not a replacement for `qwasar-agent`.** The TUI agent stays as the small,
  readable reference loop. Crucible is the one with a sandbox and a scheduler.
- **No Python, no C++.** Inherited from the parent tree (§8) and unchanged.

### 1.3 Requirements

| | |
|---|---|
| Host | Apple silicon Mac, 32 GB recommended (see §2.3) |
| macOS | 14.0 minimum; developed on 26.x |
| Build | Xcode 16+, Swift 6 language mode, `cc` for the C tree |
| Model | The same `qwasar-model` directory the CLI uses; ~16 GB, not bundled |
| Guest | Alpine arm64 + OTP 27 + Elixir 1.18, ~320 MB image (see §6.2) |

---

## 2. Three constraints the runtime imposes, and what they cost

Everything unusual in this design traces back to one of these. Read this
section before §3; the architecture is downstream of it.

### 2.1 The engine serialises. All of it.

`qwasar_metal.m` holds `g_device`, `g_queue`, `g_library`, and a lazily
populated `g_pipelines` dictionary in file statics with no lock. `qwasar.h`
mentions no threading contract. `qwasar_server.c` says it outright in its
header comment: requests are *served one at a time*.

Command buffers themselves are per-call (`qw_cmd_begin` allocates one and
`qw_cmd_commit` submits it), so there is no hidden "current encoder" global.
What is global is the pipeline cache, and it is mutated on first use of each
kernel.

**Consequence.** Every `qwasar_*` call in the process must be *serialised*.
Thread affinity is not required — Metal does not care which thread encodes —
but overlap is forbidden. This is not a per-session lock: two sessions
evaluating concurrently would race the same pipeline dictionary.

**Design response.** One serial `DispatchQueue` owns the engine and every
session. It is a real queue, not a Swift `actor`:

> A Swift actor serialises correctly, but the generation loop is a multi-second
> blocking call sequence, and blocking a cooperative-pool thread starves the
> concurrency runtime. The engine gets its own thread; Swift concurrency talks
> to it through `AsyncStream` and `withCheckedContinuation`.

### 2.2 Sessions are append-only and cannot be rewound

From `qwasar.h`: 48 of 64 layers carry recurrent state with no per-position
history. A KV cache can be truncated; a delta-rule state cannot. So:

- **Editing an earlier turn is a full re-prefill.** There is no "retry from
  message 4". A retry is a new session that re-evaluates the prefix.
- **Changing the system turn is a full re-prefill**, because the system turn is
  the prefix of everything. This one has teeth: the obvious design for a
  self-modifying agent is "the model defines a tool, and the tool list in the
  system prompt grows". That design costs a complete re-prefill of the entire
  conversation *every time the model writes a tool* — the exact operation the
  product exists to make cheap.

  This is why the tool surface in §7.1 is **fixed and closed**, with a generic
  `invoke` dispatcher. The registry of agent-written tools is communicated in
  *tool results*, which are appended, not in the system turn, which is not.
  The single most important design decision in this document falls straight
  out of a constraint in the runtime's header file.

- **Prefix reuse is the mitigation and it is already built.**
  `qwasar_session_common_prefix`, `qwasar_session_save`, and
  `qwasar_session_restore` exist for precisely this. §4.4 uses all three — and
  §2.3 promotes them from an optimisation into the mechanism the scheduler is
  built on wherever the profile allows fewer live sessions than the user has.
- **And a full context window is the end of a session.** Not an error, not a
  degradation — the end. What happens next is §2.4.

### 2.3 The machine picks the profile

**Decision: the app derives a memory profile from the host at first launch, and
the profile decides how much context a session gets and how many sessions can be
live at once.** Not a constant, not "as much as possible", and not a slider the
user is expected to understand unaided. A 32 GB M4 and a 128 GB M5 Max are
different machines and should behave differently without anyone editing a plist.

Three facts set the arithmetic:

- **The model's ceiling is 262144** (`max_position_embeddings`), and the KV cache
  costs **64 KB/token** — 16 full-attention layers × 4 KV heads × 256 × 2 × fp16.
  The full window is 17.18 GB of KV, which on a 32 GB machine does not fit
  alongside 16.02 GB of weights under any arrangement.
- **The KV cache is allocated eagerly.** `qwasar_session_new` sizes it from
  `max_ctx` at creation — `qw_buf_alloc(kv_per_layer * n_full_attn_layers)` in
  `qwasar_graph.c`, where `kv_per_layer` is `kv_heads × max_ctx × head_dim`.
  There is no growth-on-demand, so a one-message conversation costs exactly what
  a full one costs. Context is chosen when a session is born and never changes.
- **Everything competes for the same unified memory.** Metal's
  `recommendedMaxWorkingSetSize` (26.8 GB on the target host, ~84% of physical)
  bounds the GPU-resident half — weights, KV, activation scratch. The guest VMs,
  the app, and macOS draw on the same pool from the other side. A profile that
  respects only one of the two is wrong.

#### The budget

```
GPU-resident   =  weights 16.02 GB              (15.1 text + 0.92 vision)
                + live sessions × session(ctx)
   session(ctx) =  0.35 GB fixed                (151 MB SSM/conv + ~200 MB scratch)
                +  ctx × 64 KB                  (KV)
                +  ctx ×  4 KB                  (MTP draft head KV, if loaded)

   held under   0.85 × recommendedMaxWorkingSetSize

System         =  weights (same allocation, unified)
                + running VMs × 2.0 GB
                + macOS and the app, ~3.0 GB
```

The 0.85 is a reserve, not a rounding error. `recommendedMaxWorkingSetSize` is
advisory — exceeding it does not fail an allocation, it starts evicting GPU
resources to swap, which on a 16 GB weight set is indistinguishable from the
application hanging. Leaving 15% is much cheaper than finding the cliff.

#### The tiers

Derived at first launch, shown to the user with this arithmetic visible, and
adjustable:

| host | live sessions | context | MTP | why |
|---|---|---|---|---|
| 24 GB | 1 | 8K | off | marginal — 17 GB of usable working set against 16 GB of weights. Runs; barely. Warn at launch. |
| **32 GB (target)** | **1** | **90112 — measured** | **off** | 22.78 usable − 16.02 = 6.76 GB for one session. A second does not fit at any context worth having. |
| 48 GB | 2 | 131072 (128K) | off | 34 − 16.02 = 18 GB; two sessions at 8.4 GB each. |
| 64 GB | 2 | 196608 (192K) | on | 46 − 16.02 = 30 GB; two at 12.9 GB, and the MTP head's 0.8 GB is finally affordable. |
| **128 GB (M5 Max)** | **4** | **262144 (max)** | **on** | 91 − 16.02 = 75 GB; four full-window sessions at 17.5 GB each. The model's ceiling, at last. |
| 256 GB+ | 4 | 262144 | on | capped deliberately — see below |

On the 32 GB target this reads exactly as expected: **one active session, and a
context comfortably below the model's maximum.** Well below the ~160K that would
technically fit, because the difference would be bought out of the reserve that
keeps the machine from swapping its weights.

> **Measured, M0.** `MemoryProfile.derive()` on the target host reports
> `recommendedMaxWorkingSetSize = 26,800,603,136` bytes and resolves
> **90112 tokens, 1 live session, MTP off**. This document first predicted 98304
> from rougher arithmetic; the code is right and the table above now carries the
> measured number. Every other row is still a prediction.
>
> **Confirmed under load, M1.** A 90112-token session allocates and runs: 24.6 GB
> wired, the machine stays responsive, and prefill is only 3.6% slower than at
> 32K (§2.5). The tier table survives contact with a real allocator, which was
> M1's gate.
>
> **Units.** Metal reports bytes and this document's tables are decimal GB:
> 26,800,603,136 bytes is 26.80 GB and 24.96 GiB. M0's first summary printed the
> GiB figure under a GB label, which is how a budget quietly gains 7%. Decimal
> GB throughout, and the code says so where it formats.

#### Why the live-session count is capped at 4

**More live sessions never buy throughput.** §2.1 serialises every `qwasar_*`
call onto one queue; exactly one session can be *running* whatever the profile
says. Live count buys one thing only — the elimination of a park and a restore
when the user switches — and past a handful of sessions that is a diminishing
convenience bought with gigabytes. So the tier table caps at 4 even where memory
would allow more, and the surplus on a very large machine goes to full context
instead, which is the resource that actually extends what a session can do.

#### What follows from this

- **Parking is the mechanism, not an optimisation** (§4.4). On the 32 GB target
  every session switch is a park and a restore, and the machinery has to be good
  enough to make that ordinary. On a 128 GB machine parking is rarer but never
  absent — a fifth session still parks the least recently used.
- **Context is per-session and immutable**, recorded in
  `SessionRecord.contextSize`. A session created under one profile and resumed
  under another — a smaller machine, or a profile the user tightened — must be
  *detected* and offered a re-prefill at the new size, never silently misread.
- **The profile is visible and adjustable**, in Settings, with the arithmetic
  and the consequence of each change spelled out ("2 live sessions at 128K, or 1
  at 192K"). A user who wants to trade context for switching speed should be
  able to, and should be able to see what they are trading.
- ~~**It needs one accessor the app cannot reach.**~~ **Withdrawn, M0.** This
  proposed adding a working-set accessor to `qwasar.h`, on the grounds that
  `qw_gpu_working_set_limit()` lives in a header the module map excludes. The
  ask is unnecessary: the app links Metal already, and
  `MTLDevice.recommendedMaxWorkingSetSize` read from Swift is the same number
  from the same device. **No change to the parent tree is needed for the
  profile.** §4.4's checkpoint API remains the only outstanding ask.
- **A full context window is still the end of a session** — sooner on a 32 GB
  machine than on a 128 GB one, which makes §2.4 a routine event on the target
  host rather than an edge case.

**These numbers are derived, not measured.** They come from the parent tree's
measured weight sizes and the KV arithmetic, and every one of them is a
prediction. M1's gate is where they meet a real allocator.

### 2.4 Compaction is a new session, and the plan says so

**Decision: there is no in-place compaction, and none will be attempted.** A
compaction event — automatic at a context threshold, or user-invoked — creates a
**successor session**: the model writes a handoff summary, and a fresh
`qwasar_session` is prefilled with a new system turn plus that summary. The old
session is archived, not mutated.

This is not a limitation being worked around; it is the only thing the runtime
permits. Every other harness's compaction quietly rewrites history in the middle
of a KV cache, and 48 of this model's layers have no history to rewrite. Stating
it as a product concept rather than an implementation detail is what makes it
honest in the UI: a successor session is a *visible* new row in the sidebar,
linked to its parent, with the summary as its first item. The user can read the
handoff and open the ancestor. Nothing is silently lost, because nothing is
silently anything.

Consequences worth building for:

- The successor prefills from cold. At 96K on the target host — more on a larger
  machine — that is the single most expensive operation in the application, and
  §5.4's progress reporting is what makes it survivable.
- Compaction is **more frequent on smaller machines**, because the profile gave
  them less context (§2.3). The flow is therefore designed against the 32 GB
  case, where it is routine, not against the 128 GB case, where it is rare.
- The sandbox **is inherited, not recreated**: same VM, same `/work`, same module
  manifest and registry (§7.3). The conversation restarts; the agent's
  accumulated tooling does not. This is a significant part of the argument for
  putting the tools in a VM in the first place — the expensive state is not in
  the context window.
- The handoff summary is generated by the model on the outgoing session, with a
  prompt that names what matters: the task, decisions made, what was tried and
  failed, the state of `/work`, and the tools it wrote.

### 2.5 And the speed ceiling, restated

Dense 27B on ~120 GB/s: **~8 tok/s absolute, 5.5–7 tok/s realistic**. No UI
work changes this. What the UI *can* change is whether the user is looking at a
blank pane while it happens, which is why prefill progress (§5.4) is a
first-class element and not a spinner.

Prefill is compute-bound and has more headroom, but §2.3 moves it to the centre
of the experience: a cold session at profile context is the longest-running
operation in the application by an order of magnitude, and it gets longer on the
machines that can afford more of it.

#### Prefill is the agent loop's real cost, and it is worse than decode

**Measured, M1, on the target host:**

| | |
|---|---|
| prefill, 4118-token prompt, 32K context | **33.0 tok/s** |
| prefill, same prompt, 90112 context | **31.8 tok/s** |
| decode | 5.6–5.8 tok/s |

Two things fall out, and the second is the one that shapes the product.

**Context size costs almost nothing.** 3.6% between a 2 GB KV cache and a 5.9 GB
one. The §2.3 profile's aggressive context is not paid for in throughput, which
is the outcome that decision needed and did not have evidence for until now.

**But every tool result is prefilled before the model can react to it**, and at
~32 tok/s that is roughly **one second per 100 bytes of tool output**. In a
tool-using loop this dominates everything: decode produces a few hundred tokens
per turn at 5.8 tok/s, while a single file read can produce fifteen thousand at
32 tok/s.

The first agent run made it concrete. `read` on a 49 KB source file produced a
**14,640-token prefill — about seven and a half minutes**, and 16% of a 90K
window, for one call. `qwasar_agent.c`'s `AGENT_MAX_READ` is 256 KB; at this
rate that ceiling is forty minutes.

Three consequences, all now built:

- **Tool results are capped at 8 KB** (~2.4k tokens, ~75 s), with a truncation
  message that states the file's real size and points at `grep`. Telling the
  model a file is 49 KB and that it can search it is worth far more to it than
  handing it 49 KB.
- **The prefill bar carries an ETA** (§5.4), because a seventy-five-second wait
  with no output is the failure mode this measurement predicts.
- **It is an argument for the architecture, not against it.** §7's whole premise
  is an agent that writes its own tools. On an engine where returning bytes is
  the expensive act, a tool that extracts the three functions the model actually
  wants beats one that returns the file — and only the model knows which three.
  Prefill cost is the strongest case in this document for `define` and `invoke`.

**A note on the parent's number.** `PLAN.md` in the parent tree records prefill
at 42.6 tok/s at a 256-token chunk. Measured here it is 32–33 with an ordinary
desktop load on the machine. Not a contradiction worth chasing — the shape of
every conclusion above is the same at either figure — but the app's ETA uses the
measured one. Three things keep it off
the critical path — the shared system-prefix checkpoint (§4.4), park/restore
instead of re-prefill (§4.4), and never re-rendering a conversation that can be
continued with `qwasar_render_tool_result`.

---

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
│  │   VZVirtualMachine  ·  vsock  ·  virtiofs (read-only)    │  │
│  └──────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────┘
                     │ AF_VSOCK, port 1024
┌────────────────────▼──────── guest: Alpine arm64 ─────────────┐
│  warden    (Elixir)  immutable control plane, owns the bridge  │
│     │ Erlang distribution over loopback                        │
│  workspace (Elixir)  the agent's node — hot-loadable, restarts │
│  sim       (ephemeral) per-simulation, eta-instrumented (§9.3) │
│  /base   virtiofs, read-only, the project as it was            │
│  /work   guest disk, writable, where everything happens        │
└────────────────────────────────────────────────────────────────┘
```

Two rules this diagram encodes:

- The **host never runs a tool**. Not `bash`, not `write`. If the model wants a
  command run, it runs in the guest.
- The **guest never writes the host**. `/base` is mounted read-only and `/work`
  is the guest's own disk. The only path back is a patch through §7.4.
- **Three nodes, not one**, and the split is load-bearing in both directions:
  warden must survive anything the agent does, and eta's simulator must be able
  to stop the world without stopping the bridge (§9.3).

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

## 4. Sessions, projects, and the scheduler

### 4.1 The data model

```swift
struct Project: Codable, Identifiable {
    let id: UUID
    var name: String
    var rootBookmark: Data        // security-scoped, resolved at launch
    var defaultContextSize: Int    // 32_768
    var sandboxNetworking: Bool    // default false; see §8.3
    var excludeGlobs: [String]     // beyond .gitignore
}

struct SessionRecord: Codable, Identifiable {
    let id: UUID
    var projectID: UUID
    var title: String              // model-generated after the first exchange
    var workingSubpath: String     // relative to the project root, "" = root
    var createdAt: Date
    var state: SessionState        // .live, .parked, .rebuilding, .archived
    var tokens: [Int32]            // FULL history — 4 bytes/token, see §4.4
    var transcript: [TranscriptItem]
    var moduleManifest: [AgentModule]   // §7.3
    var pendingPatch: PatchProposal?

    // Compaction chain (§2.4). A successor inherits the sandbox, not the context.
    var ancestorID: UUID?
    var successorID: UUID?
    var handoffSummary: String?    // the first item of a successor session

    // Resolved once at creation from the active profile (§2.3) and recorded
    // rather than recomputed: a session created on a 128 GB machine and resumed
    // on a 32 GB one must be detected and offered a re-prefill, never silently
    // misread. Immutable for the life of the session — the KV cache was sized
    // from it at qwasar_session_new and cannot grow.
    var contextSize: Int32
}
```

`tokens` is the load-bearing field. Everything about parking, resuming, and
crash recovery works because the complete token sequence is cheap to keep: a
96K-context session is 384 KB of `Int32`, against a checkpoint three orders of
magnitude larger. The transcript is for humans; the
token array is the truth, and it is what `qwasar_session_restore` is fed.

### 4.2 Storage

`~/Library/Application Support/Crucible/`

```
projects.json                      the Project list
sessions/<uuid>/
  record.json                      SessionRecord minus tokens
  tokens.bin                       raw Int32 LE
  transcript.jsonl                 append-only; survives a crash mid-turn
  checkpoint.bin                   parked session state — up to ~8.2 GB (§4.4)
  baseline.json                    path → sha256 + mode, at sandbox creation
  disk.img                         APFS clone of the golden guest image
  modules/                         agent-written source, one file per module
  traces/<id>.trace                eta simulation traces, replayable (§9)
  undo/<n>/                        pre-application snapshot of touched files
images/
  golden-<version>.img             the guest image, cloned per session
```

`checkpoint.bin` dominates this directory and is the reason session deletion is
a real command with a real confirmation rather than a swipe. `modules/` is
redundant with `record.json`'s manifest and exists anyway, because the agent's
own source is the artefact a user is most likely to want to read, diff, or steal
for a real project.

`transcript.jsonl` is append-only on purpose: a crash during generation should
lose the current turn, not the conversation. Same reasoning as the session
itself being append-only, one layer up.

Project roots are reached through **security-scoped bookmarks**, resolved at
launch and held with `startAccessingSecurityScopedResource()` for as long as the
project is open. The model directory gets the same treatment. Under App Sandbox
(§8.1) there is no other way to keep access to a user-chosen folder across
launches, and discovering that after building the whole file layer is a bad day.

### 4.3 The scheduler

One engine, one queue, and **as many live sessions as the profile allows** —
which on the target host is one, and on a 128 GB machine is four (§2.3). The
scheduler is written for the general case and degrades cleanly to the
single-session one, because writing it the other way round means writing it
twice.

```
     user sends ──▶ [ run queue ]
                          │
                          ▼   one RUNNING, ever (§2.1)
   .parked ──restore──▶ .live ──▶ .running ──▶ .idle
      ▲                   │                      │
      └──── park ─────────┘                      └──▶ .awaitingApproval (§7.4)
            when the live set is over budget
```

- `.live` means the session holds its share of the working set. The live set is
  bounded by the profile; admitting a session over the bound parks the
  least-recently-used one first.
- `.running` is one session at a time regardless of how many are live, because
  §2.1 says so. **Live count buys switch latency, never throughput** — the UI
  must not imply otherwise, and there is no "run these two in parallel".
- A session in `.queued` shows its position **and, if parked, its restore cost**.
  A parked session at profile context is seconds of disk before its first token,
  and saying so up front is the difference between a considered wait and a
  suspected hang.
- **On a one-session profile, switching costs a park and a restore.** So the UI
  does not switch on hover, on selection, or on anything speculative: reading a
  parked session's transcript does not restore it, sending a message does. On a
  multi-session profile the same interaction is instant, and the UI says which
  world it is in rather than being uniformly cautious.
- The user can cancel a queued session without touching the running one.
- **Interrupt** sets an atomic flag the generation loop polls once per token —
  the same contract `tui_interrupted` has in the C agent. It stops at a token
  boundary, keeps everything generated so far in the session (it was evaluated;
  it cannot be un-evaluated), and marks the turn interrupted.

### 4.4 Parking, and the checkpoint API it needs

Parking is how the profile is enforced (§2.3), and on a one-session host it is
what every switch costs — so the bar on it is high. The runtime has the right
primitives and the wrong policy around them.

**Park:**
1. `qwasar_session_save(s, e, ...)` — writes the KV cache *and* the 151 MB
   recurrent state.
2. `qwasar_session_free(s)` — reclaims the session's entire share of the
   working set, KV cache included.
3. Keep `tokens` in the record. Stop the VM (§6.5).

**Resume:**
1. `qwasar_session_new(e, ...)`.
2. `covered = qwasar_session_restore(s, e, tokens, tokens.count)`.
3. `qwasar_session_eval(s, tokens + covered, count - covered, ...)` for whatever
   the checkpoint did not cover.

`qwasar_session_restore` fills a fresh session from the longest cached *prefix*
and returns how many tokens it covered; a return of 0 is not an error, it is a
cold resume. Because the recurrent half cannot rewind, only a true prefix is
reusable — fine here, since a parked session's tokens are by construction the
prefix of the same session's tokens on resume.

#### The parent tree needs one change, and this is it

`qwasar_session_save` does not write a file the caller names. It writes into a
**hash-keyed LRU cache** under `$HOME`, and `qwasar_kvstore.c` ends every save
with `qw_kv_evict(dir, QW_KV_DEFAULT_BUDGET)` where

```c
#define QW_KV_DEFAULT_BUDGET (6ull * 1024ull * 1024ull * 1024ull)
```

**6 GB, hardcoded, no environment override.** A parked session at 96K is ~6.3 GB
and at the full 262144 is ~17 GB. Every park would meet or exceed the entire
store budget in a single entry — evicting
every other session's checkpoint, and then plausibly itself. The existing design
is correct for what it was built for (an opportunistic prefix cache with a
149 MB floor and a `QW_KV_MIN_TOKENS` threshold) and simply is not a place to
store session state the application owns.

So Crucible asks the parent tree for a narrow addition, in the spirit of the
existing header:

```c
/* Explicit-path checkpoints, for a caller that owns the file's lifetime.
 *
 * qwasar_session_save() manages a shared LRU cache keyed by token hash, which
 * is right for a prefix that many sessions rediscover.  An application parking
 * a live session needs the opposite contract: a named file, no eviction, and a
 * restore that either matches or fails loudly. */
bool    qwasar_session_save_file(qwasar_session *s, const qwasar_engine *e,
                                 const char *path, char *err, size_t errcap);
int32_t qwasar_session_restore_file(qwasar_session *s, const qwasar_engine *e,
                                    const char *path,
                                    const int32_t *tokens, int32_t n);
```

Both are thin wrappers over the serialisation `qwasar_kvstore.c` already does —
the header format, the `dims[8]` shape check that rejects a checkpoint from a
different quantisation, and the stored-token comparison that turns a hash
collision into a miss rather than corruption all stay exactly as they are. What
changes is who names the file and who deletes it.

**The hash-keyed cache still earns its keep**, for one thing: the shared system
prefix. `agent_prefill` in the C agent evaluates to the end of the system turn,
saves, then continues; every session shares that prefix, so it is cached once
and every new session starts warm. Because §2.2 makes the system turn constant
across the whole app, that single entry serves every session ever created. Copy
the trick verbatim — and it comfortably fits the 6 GB budget, because it is one
entry of ~149 MB.

#### Disk is now a first-class budget

Ten parked sessions is **63 GB** at the 32 GB host's 96K profile, and **172 GB**
at a large machine's full window. That is not a footnote:

- Checkpoints live under `sessions/<uuid>/checkpoint.bin`, so deleting a session
  reclaims its space and the accounting is obvious in Finder.
- A per-application quota, set in Settings, defaults to 25% of free space at
  first launch. Exceeding it prompts; it never silently deletes a session's
  state, because that state is not reconstructible without a full re-prefill.
- A session whose checkpoint is evicted or missing is not lost — it falls back to
  re-prefilling from `tokens`. Slow, correct, and clearly labelled in the UI as
  *rebuilding*.
- Checkpoint size is proportional to `n_past`, not to `context`. A young session
  parks in a fraction of a second. Report the real number, not the worst case.

**Measure, don't assume.** Park and restore times at multi-gigabyte checkpoint
sizes are dominated by disk bandwidth and are exactly the kind of number the
parent tree records rather than estimates. The full-profile figures get measured
in M6; here is the first real data point, from M1:

> A 3656-token session — a checkpoint of roughly 383 MB, being 149 MB of
> recurrent state plus 234 MB of KV — **restored in 0.2 s**, against the
> ~2 minutes a cold re-prefill of the same history would have cost at the
> measured prefill rate (§2.5). Extrapolating linearly, a full 90112-token
> session is about 6.05 GB and roughly 3 s.
>
> Two things this settles. `qwasar_session_save`/`restore` work correctly when
> driven from Swift, verified behaviourally rather than structurally: the
> session was closed, its state freed, reopened from the recorded tokens alone,
> and asked a question only a session that still remembered the first turn could
> answer. It answered. And the continuation path is right — the follow-up turn
> prefilled **29 tokens**, not the whole conversation.
>
> It also confirms that checkpointing after every turn is worth doing
> unconditionally at M1 sizes. What it does not settle is the 6 GB LRU budget:
> at 383 MB this session fits comfortably, and at full context it would not.

#### The budget, settled — and the policy was wrong

Read off a real cache after a fortnight's use, the budget question answered
itself, in the opposite direction to the guess above:

```
tokens=  1397   file= 236.9MB   hits=  0
tokens=  2563   file= 309.8MB   hits=  0
tokens=  3656   file= 378.1MB   hits=  1
tokens=  3686   file= 380.0MB   hits=  0
tokens=  4085   file= 405.0MB   hits=  0
tokens=  8691   file= 692.8MB   hits=  0
tokens= 18335   file=1295.6MB   hits=  0
tokens= 19084   file=1342.4MB   hits=  0

9 checkpoints, 5.30 GiB of 6.00 GiB — one hit between them.
```

Checkpointing after every turn does not produce *a* checkpoint per session. It
produces a **staircase**: one file per turn, each a strict superset of the last,
each superseding it and none of them removed. Two files here are 1.3 GB apiece —
44% of the budget — and the whole 5.30 GiB was read once.

Those files also give the size law directly. 236.9 MB at 1397 tokens and
309.8 MB at 2563 fit **64 KB/token on a 149.6 MB fixed floor**, matching the
recurrent-state figure in `qwasar_kvstore.c` to a decimal place. So the cost of
a checkpoint is knowable in advance, and the fixed floor is what makes many
small ones a bad trade.

The policy is now:

- **The system prefix, always.** Measured on the M4 surface, the system turn is
  **2491 tokens — 99.6% of a first turn**, because the twelve tool schemas are
  rendered into it. It is identical for every session of a project at a given
  effort, so at the ~32 tok/s of §2.5 it is *~78 s paid again on every new
  session*, and — since compaction is a successor session (§2.4) — on every
  compaction. One ~305 MB checkpoint buys all of them. `Session.primeSystemPrefix`
  evaluates it alone, saves, and then evaluates the rest of the turn on top.
- **On close, once.** A session earns exactly one checkpoint, taken when it has
  stopped growing. This is what makes reopening a read rather than a re-prefill,
  and the 19084-token session above is precisely the case that justifies it: ten
  minutes of prefill, or three seconds.
- **Nothing per turn.**

LRU then protects the right thing without being told to: the prefix is hit by
every session, so its `last_used` stays fresh and it survives; a stale
whole-session snapshot is not, and goes first.

Two properties make the prefix scheme safe rather than merely fast, and both are
asserted in `Tests/PrefixSuite.swift` (tokenizer only, so it runs in `make test`):

1. **The system turn is a strict token prefix of the first turn** — not a
   similar string, the same leading ids. If it were not, priming would leave the
   recurrent state part-way through a system turn it could never rewind out of
   (§2.2). Checked across every effort, both tool surfaces, and four message
   shapes.
2. **Anything that changes the model's behaviour changes the prefix**: effort,
   the tool surface, the system prompt. The store compares its stored tokens
   before unpacking, so a hit is by construction the right state — but a knob
   that changed behaviour *without* changing the prefix would put one checkpoint
   behind two different system turns, and nothing downstream could tell.

The suite turned up one thing worth recording: **`thinking` does not affect the
system turn when tools are present.** With tools the template rebuilds the system
turn around them and renders the same tokens either way. Harmless for the cache —
identical tokens are identical state — but it is a property of the template
rather than a decision made here, so it is pinned by a test.

`make gate-prefix` proves the round trip against the real engine: prime a cold
session, prime a second, and require that the second read the whole prefix off
disk. It is a separate target because the cold half is a real ~2500-token
prefill. It exists because this optimisation fails *silently* — a checkpoint
never written and a checkpoint read every time look identical from the outside,
only slower, which is exactly how the staircase above went unnoticed.

---

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

## 6. The sandbox

### 6.1 Framework choices, stated plainly

- **`VZVirtualMachine`** with `VZEFIBootLoader` and a raw disk image. EFI boot
  rather than `VZLinuxBootLoader` + kernel/initrd, because it lets the guest own
  its own kernel updates and lets `mkimage.sh` produce a single artefact.
- **`VZVirtioBlockDeviceConfiguration`** — the per-session disk (§6.3).
- **`VZVirtioFileSystemDeviceConfiguration`** with a `VZSharedDirectory` marked
  **read-only**, tag `base`. This is the project tree going in. Read-only is
  enforced by the framework, not by our code, and that is why we use it rather
  than streaming a tarball.
- **`VZVirtioSocketDeviceConfiguration`** — vsock. The control channel.
- **`VZVirtioConsoleDeviceSerialPortConfiguration`** — boot log to a file, for
  the inspector and for support.
- **No `VZNetworkDeviceConfiguration` at all**, by default. Not a firewall rule,
  not a policy toggle the guest could talk its way around: the VM is configured
  without a network device, so there is nothing to configure. This is the single
  strongest security property in the design and it costs one omitted line.
- **`VZVirtualMachineConfiguration.validateWithError`** before every boot. It
  catches misconfiguration with a real message instead of a runtime trap.

Memory: 2 GB per VM by default, 1 CPU less than `VZVirtualMachineConfiguration
.maximumAllowedCPUCount` capped at 4 — the engine needs the cores more than the
guest does, and the guest is mostly waiting on the model.

### 6.2 The golden image

`Guest/mkimage.sh` builds it, reproducibly, from an Alpine arm64 minirootfs:

```
alpine base + busybox                 ~  10 MB
linux-virt kernel + modules           ~  70 MB
mise + erlang 27 (source-built)       ~ 160 MB
mise + elixir 1.18 (precompiled)      ~  30 MB
node, python3, git, ripgrep (apk)     ~ 180 MB
warden (compiled) + vsock_port        ~   2 MB
--------------------------------------------
rootfs 479 MB · disk image 540 MB allocated, 3 GB apparent
```

#### Runtimes come from mise, not from apk

The operating system is Alpine's. The **language runtimes are mise's**, and the
split is deliberate:

- **apk cannot supply what this design needs.** Alpine's `elixir` package
  depends on `erlang26` and *conflicts* with `erlang27`, so the two cannot be
  installed together at all — and OTP 27 is not negotiable, because §6.4 decodes
  JSON with its built-in `:json` module. That is what forced the question.
- **The guest exists to work on someone's project, and projects pin versions.**
  mise reads `.tool-versions` and `mise.toml` out of `/work`, so the agent gets
  the runtime the project actually asks for rather than whatever the image
  happened to ship.
- **Adding a language becomes one word.** `$RUNTIMES` in the Dockerfile is the
  entire interface for giving the sandbox Go, Rust, or another Erlang.

**Measured:** erlang@27.3.4.16 is a source build and takes **1m39s**;
elixir@1.18.4-otp-27 is precompiled `.beam` files and takes **1.2s**. Both are
cached in a Docker layer, so they cost that once per pinned version.

**And a measured constraint: this guest is musl, and mise's catalogue is largely
glibc.** `node@22` fails outright — Node ships no musl binaries, so mise falls
back to a source build the image is not equipped for. So mise handles what mise
can handle here, and the musl-native runtimes (node, python3) come from apk.
Moving the guest to a glibc base would open the whole mise catalogue; that trade
is recorded here rather than made silently, and it is the first thing to
reconsider if the agent starts wanting runtimes Alpine does not package.

#### The control plane does not go through mise

`warden` runs on a **concrete path** — `/opt/mise/installs/erlang/27.3.4.16/bin/erl`,
resolved once at build time and symlinked — not through `mise exec`.

§7.3 requires that warden survive anything the agent does. If it started through
mise, an agent that edited a mise config, or simply switched the project's
Erlang version, would take the control plane down with it, and init's restart
loop would spin forever restarting something that can no longer start. mise is
what the agent and the project use. The control plane's runtime is infrastructure
and is pinned.

An Erlang built from source bakes its `ROOTDIR` into the `erl` script, so mise
installs into `/opt/mise` in the *builder* and the tree is copied across
wholesale — built at the path it will live at, rather than relocated afterwards.

Alpine rather than Debian: musl, no systemd, and a boot-to-BEAM path measured in
hundreds of milliseconds — **0.56 s to warden-ready, with all of the above in
the image**. `git` is in the image because the diff engine is git (§7.4) — that
is the one dependency worth its size.

OTP 27 specifically, for two reasons beyond currency: its built-in `json`
module removes the last dependency from the guest (§6.4), and eta is developed
against modern OTP. One language and one build tool across the guest — warden,
workspace, and the DST profile are all `mix` (§9.1). `eta` and the DST-built support beams are the only test
machinery in a shipping artefact anywhere in this design, and they are there
because §9.3's sim node is a runtime feature, not a test.

Init is not OpenRC. `/sbin/init` is a ~40-line shell script that mounts
`/proc`, `/sys`, `/base` (virtiofs), starts the warden, and execs. Target: **VM
boot to warden-ready under 2 seconds.** Every second here is a second a user
waits before the model can do anything, on every resume.

**Shipping it.** The image goes in `Contents/Resources/` compressed, or is
fetched on first launch. Decided in §14 — 320 MB is right at the boundary where
bundling stops being polite.

### 6.3 Per-session disks are APFS clones

```swift
// Copy-on-write. Instant, and zero additional blocks until written.
clonefile(goldenPath, sessionDiskPath, 0)
```

This is the detail that makes "a VM per session" affordable. A 2 GB sparse image
clones in microseconds and consumes only what the session actually writes.
Sessions get real isolation — their own filesystem, their own installed
packages, their own everything — at the cost of the delta. Deleting a session is
deleting the clone.

Fall back to a plain copy if `clonefile` fails (non-APFS volume), and say so in
the inspector rather than silently taking 2 GB per session.

### 6.4 The vsock channel

Host side: `VZVirtioSocketDevice.connect(toPort: 1024)` yields a
`VZVirtioSocketConnection` with a real file descriptor. Wrap it in a
`DispatchIO` or `NWConnection`-style reader.

Guest side is the interesting half, because **the BEAM cannot open an AF_VSOCK
socket.** `gen_tcp` has no vsock support and `socat` is a 400 KB dependency for
one pipe. So `Guest/vsock_port/vsock_port.c` is a ~120-line port program:

```c
/* AF_VSOCK listener ↔ stdio, spoken as an Erlang port with {packet, 4}.
 * The BEAM opens it with open_port({spawn_executable, ...},
 * [{packet, 4}, binary, exit_status]) and gets framed messages with no
 * parsing of its own. */
```

Framing is 4-byte big-endian length, which is exactly what `{packet, 4}` means
on the Erlang side and what a two-line read loop means on the Swift side.
Payloads are **JSON**, decoded on the guest with OTP 27's built-in `:json`
module — no Jason, no dependency, no mix deps to vendor into the image.

```
host → guest   {"id": 41, "op": "invoke", "tool": "grep_ast", "args": {...}}
guest → host   {"id": 41, "ok": true,  "result": "...", "took_ms": 12}
guest → host   {"id": 41, "ok": false, "error": "no such tool: grep_ast",
                "kind": "unknown_tool"}
guest → host   {"event": "log", "level": "warn", "text": "..."}   // unsolicited
```

Every request carries a host-assigned `id`; every response echoes it. Unsolicited
events (logs, module-load notices, progress) carry no `id`. The host times out
every request — default 30 s, `shell` 300 s — and a timeout is reported to the
model as a tool result, never as a failed turn. Same principle the C agent
applies to a refused confirmation: the model gets told and chooses again.

### 6.5 VM lifecycle

- **Created** lazily, on the session's first tool call — not on session creation.
  A conversation that never touches a file never boots a VM.
- **Parked** with the session (§4.4): `stop()`, keep the clone, keep the module
  manifest. Resume re-boots and replays the manifest (§7.3).
- **Capped**: `maxRunningVMs`, default 2, derived from physical memory. In
  practice the live session's VM plus at most one lingering VM, since only one
  session is live at a time (§4.3) — a VM outlives its session's parking only
  long enough to finish an in-flight `propose`.
- **Crash**: `VZVirtualMachineDelegate.guestDidStop` / `didStopWithError`. The
  session goes to `.parked` with a transcript item saying so; the next tool call
  re-boots and replays. The model sees a tool result explaining that the sandbox
  restarted and that in-memory state is gone but `/work` survived — because it
  did, it is on the disk.

---

## 7. The agent, and how it rewrites itself

### 7.1 The fixed tool surface

Twelve tools. This list is **frozen at build time** and never changes at
runtime, for the reason in §2.2: it is the system turn, and the system turn is
the prefix of everything. The count is allowed to grow during design and is
frozen at M4; what must never happen is a tool appearing because the model asked
for one.

| tool | runs where | notes |
|---|---|---|
| `read(path)` | guest `/work` | |
| `write(path, content)` | guest `/work` | no confirmation — it is a sandbox |
| `edit(path, old, new)` | guest `/work` | line-anchored, same contract as `qw_edit_apply`, reimplemented in Elixir |
| `list(path)` | guest `/work` | |
| `grep(pattern, path)` | guest `/work` | ripgrep |
| `shell(command)` | guest `/work` | unconfirmed; no network, no host |
| `elixir(code)` | guest workspace node | evaluate, with a persistent binding |
| `define(source)` | guest workspace node | compile + hot-load a module (§7.2) |
| `tools()` | guest warden | the current registry of agent-defined tools |
| `invoke(name, args)` | guest workspace node | call one of them (§7.3) |
| `simulate(harness, opts)` | guest sim node | run agent code under deterministic simulation (§9.4) |
| `replay(trace)` | guest sim node | re-run one exact schedule, to verify a fix (§9.4) |

Six of these are the C agent's own tools with their descriptions kept nearly
verbatim — those descriptions were written against this model's training and the
`edit` one in particular spells out its match rule because that rule *is* the
contract. Do not paraphrase them for the sake of paraphrasing them.

The last six are the new idea, and they are where all the leverage is: four let
the agent change what it can do, and two let it find out whether the change is
correct **before** it commits to it.

### 7.2 `define` — compiling into a live node

```
define(source: """
  defmodule ASTGrep do
    @behaviour Crucible.Tool
    def name, do: "grep_ast"
    def schema, do: %{...}
    def run(%{"pattern" => p}), do: ...
  end
""")
```

What happens, in order, all inside the guest:

1. **Warden receives it.** Warden does not compile it. Warden `:erpc.call`s the
   *workspace* node, under a timeout, in a linked task.
2. **Compile.** `Code.compile_string/2` for Elixir; for Erlang source,
   `:erl_scan` → `:erl_parse` → `:compile.forms/2`. Compilation errors and
   warnings come back as text and become the tool result. A module that does not
   compile changes nothing — this is a pure function up to this point.
3. **Purge, carefully.** The BEAM holds two versions of a module: current and
   old. A third load purges old, and *purging kills any process still running
   old code*. So warden calls `:code.soft_purge/1` first. If it returns `false`,
   the load is **refused** and the tool result names the processes still
   executing the previous version. The model then decides — kill them, wait, or
   accept the loss with a `force: true` argument. Silently killing a process the
   agent started three steps ago is the kind of failure that costs an hour of
   confusion, and it is entirely avoidable.
4. **Offer simulation.** If the compiled module spawns processes, implements
   `gen_server`/`gen_statem`/`supervisor`, or sends messages, warden says so in
   the tool result and names `simulate` (§9.4). With `verify: "SomeHarness"` the
   load is *gated* on the simulation passing. This is the step the whole of §9
   exists to make possible: the agent can find out whether concurrent code is
   correct before it is running in the node it depends on.
5. **Load.** `:code.load_binary(mod, ~c"crucible://#{name}.ex", bin)` on the
   workspace node.
6. **Register.** If the module implements `Crucible.Tool`, warden adds it to the
   registry: name, schema, source, load timestamp, version counter.
7. **Report.** The tool result names the module, the functions it exports, and
   whether it registered as a tool. The host also emits a *module load* card
   into the transcript (§5.2) with a diff against the previous version, because
   the agent changing its own behaviour is the single most interesting thing
   that happens in this application and it should not be buried in a tool result.

### 7.3 Three nodes, and why the warden is untouchable

```
warden  (Elixir, :warden@guest)      workspace (Elixir, :workspace@guest)
  Warden.Bridge    vsock port          Crucible.Tool  (behaviour)
  Warden.Registry  tool manifest       Crucible.Shell (eval binding)
  Warden.Fs        /work primitives    Crucible.Fs    (the six file tools)
  Warden.Workspace spawn / monitor     ...every module the agent writes
  Warden.Diff      git plumbing
  Warden.Sim       simulation driver ──▶ sim (:sim_N@guest)
                                           ephemeral, eta-instrumented,
                                           discarded after each run (§9.3)
```

They are **separate OS processes**, connected by Erlang distribution over
loopback inside the guest with a shared cookie. Not one node with three
supervision trees. Three reasons, and each one alone would be sufficient:

1. A module the agent hot-loads can crash the scheduler, exhaust the atom table,
   or take down the node. If that node also owns the vsock bridge, the harness
   goes silent and the user sees a hang.
2. `eta_sched` **serialises every process in the VM and takes over the clock**.
   A simulation running on the workspace node would freeze the tools, and one
   running on warden would freeze the bridge. Simulation needs its own node
   because deterministic simulation is, by construction, hostile to anything
   else trying to make progress (§9.3).
3. The node boundary is exactly where `eta_net` injects faults. Warden's
   handling of a dead, slow, or partitioned workspace is the most failure-prone
   code in the guest, and putting workspace on the far side of distribution is
   what makes that code **testable** rather than merely careful (§9.2).

**Warden is Elixir, on eta's `elixir` branch.** An earlier draft made it Erlang
because eta instrumented via `parse_transform` and its Elixir support was
limited. The `elixir` branch replaces that with explicit `use Eta` macros (§9.1),
which removes the reason — so the guest is one language and one build tool, and
the same `mix` invocation builds warden, workspace, and the DST profile.

The three justifications above are unaffected by that choice: they are about
process boundaries, not syntax. And the Erlang fallback stays written down
rather than forgotten, because the branch is best-effort by its author's own
description: if it proves too thin, warden moves to Erlang and `main`, the node
topology does not change, and §9.5 is the review where that call gets made.

So:

- **Warden owns the wire and cannot be modified.** `define` requests naming a
  `Warden.*` module are rejected by name, and warden's code path is never
  `erpc`'d into. It is also the only node whose code is in the image and signed
  with it.
- **Workspace is disposable.** Warden monitors it. If it dies, warden restarts
  it and **replays the module manifest** in load order, then reports to the
  host, which reports to the model as a tool result: *the workspace restarted;
  N modules were reloaded; process state was lost.* The agent is told the truth
  and can recover. That sentence is a promise, and §9.2 is how it is kept.
- **Sim nodes are cattle.** Started per `simulate` call with `-noinput`, given
  the modules under test via eta's `preload`, killed when the run returns or the
  budget expires. A simulation that hangs, deadlocks, or exhausts memory takes
  nothing with it.
- **The manifest is the durable artefact.** It is sent to the host on every
  change and stored in `SessionRecord.moduleManifest`. That is what makes
  self-modification survive parking, a VM crash, and an app relaunch. A session
  resumed tomorrow has the tools it wrote today.

`invoke(name, args)` is a warden call: look up the name in the registry, `erpc`
into workspace's `mod.run(args)` under a timeout, marshal the return. A tool that
throws returns its exception and stacktrace as the tool result. A tool that
hangs is killed at the timeout and says so.

### 7.4a Leaning into git — the crossing as an object exchange

§7.4 below describes what M5 built: a baseline manifest, per-file SHAs, drift
detection, size caps, and a host that writes bytes under approval. It works, and
it is on a path that ends badly.

Look at what it has already grown: baseline hashes per file, a *conflict* state
distinct from a *rejection*, an added-file-now-exists case, a
modification-to-a-deleted-file case, a backup directory, a re-baseline step. Each
one is a thing git does properly. Keep going and the next asks are conflict
resolution, then "take only these two files", then "the model made three logical
changes, split them" — **merge, cherry-pick and rebase, arriving one bad
reimplementation at a time**.

The 2 MB/6 MB caps are the tell. There is no principled reason for them except
that whole file bodies are being shipped; git would send a few KB of compressed
delta for the same change.

#### The reframe

**The VM boundary is a network boundary, and git is already a distributed VCS.**
Two repositories that cannot share a filesystem, exchanging work, is the problem
git was built to solve. The guest already holds a full clone — `.git` is copied
in with everything else — it simply is not being treated as one.

So the crossing stops being a file copy:

    guest commits  ->  sends (sha, type, content) for BASE..TIP
                   ->  host writes loose objects + one ref

and the user gets a **real branch in their real repo**. Merge, rebase,
cherry-pick, stash, `git log -p`, blame, their own mergetool and their own
config — none of it written here.

A `git worktree` would be the obvious thing to reach for and does not work: `git
worktree add` writes a `.git` *file* pointing at an absolute path inside the main
repository's `.git/worktrees/`, so objects, refs and index all stay in a repo the
worktree must be able to reach and write. The guest has no path to the host's
`.git` — the share is unmounted before the agent runs, which is the property the
entire sandbox rests on (§8.2). Restoring a writable mount to enable worktrees
would trade away exactly what the sandbox is for. A *clone* exchanging *objects*
is the right primitive for this topology, and it is the one git was designed
around.

#### No libgit2, and no child process

The host cannot spawn `git`: security-scoped access does not survive into a child
process, which is why `git apply` was rejected for §7.4 in the first place.
libgit2 was the obvious answer and is **not needed**.

A loose object is `<type> <len>\0<content>`, zlib-deflated, stored at
`.git/objects/<sha[0:2]>/<sha[2:]>` where the sha names the *uncompressed* form.
That is zlib, SHA-1, and a file write — `Insecure.SHA1` from CryptoKit and `-lz`,
a system library. Nothing to vendor, nothing for a user to install, which is the
same bar the rest of this project holds (`make`, and nothing else).

Verified before writing this section, against a real repository: objects written
by hand, `git fsck --strict` clean, `git log`/`git diff` correct across a
modification, an addition, a deletion and a nested path, `git merge` clean. And
with a concurrent edit to the same line on the host, `git merge` produced a
proper **line-level** conflict with markers — where §7.4 can only detect drift at
*file* granularity and offer a checkbox.

**The hash is the verification, and it is free.** A wrong type, a wrong length or
a truncated transfer all land as a refusal rather than as a corrupt repository.
That is a stronger guarantee than the byte path has, not a weaker one.

**The import is additive.** Objects, plus one ref. Never HEAD, never the index,
never a branch the user owns, never a working file. The worst outcome of a bug is
unreferenced objects that `git gc` collects. The destructive step — changing
files — moves to the user's own `git merge`, which they already trust and which
already handles conflicts.

#### Decisions taken

**Refs land at `refs/heads/crucible/<slug>`.** A real branch: it shows in `git
branch`, in every GUI, and merges by short name. The hidden `refs/crucible/*`
namespace avoids clutter at the cost of being forgettable and unmergeable without
typing a full ref, which is the wrong trade for something the user is meant to
act on. Stale ones are deleted like any other branch.

**The agent makes its own commits.** It already has git through `bash`; the
system prompt will say so. A reviewable series beats one flat blob, and it is
what makes cherry-picking a *part* of the work possible at all. The harness
commits any uncommitted remainder at proposal time so nothing is lost.

**A dirty worktree is the user's choice, at session start.** If the project has
uncommitted changes when a session begins, they are copied into `/work` like
everything else — and would end up inside the agent's commit, so merging would
commit the user's own work under the agent's name. Rather than pick a policy, the
session-creation flow asks, because the real question underneath is *should the
agent see my uncommitted work at all* and both answers are right sometimes:

  - **Include** — `/work` is the working tree as it stands, and the guest commits
    that state first, on its own, as "local changes before this session". The
    user's work stays visibly theirs in the log. Right when they are mid-change
    and want help with it.
  - **Exclude** — `/work` is reset to HEAD after the copy, so the uncommitted
    changes never enter the sandbox at all. Right when they want the agent
    working from a known, committed base.

The prompt appears only when the tree is actually dirty, and only for git
projects.

#### What this deletes

The honest test of the idea. Of the ~936 lines across `Materialise.swift`,
`materialise.ex`, `ApprovalSheet.swift` and their suites, roughly two thirds go:
the per-file SHA baselining, drift detection, conflict and rejection states, size
caps, the backup directory, the re-baseline step, and most of the approval
sheet's per-file machinery. A large net removal is the signal that this is the
right shape.

What remains is §7.4 itself, **demoted to the fallback** for projects that are not
git repositories. That case is genuinely simple — no history to reconcile — and
is where a byte copy under approval is the proportionate answer.

#### Risks worth naming

  - **It writes into the user's `.git`.** Confined to new objects and
    `refs/heads/crucible/*`; `git fsck` is the check, and it runs in the tests.
  - **`.git` is copied into the guest**, so its size is a boot cost. 6.2 MB for
    this repository, which is nothing; a repository with large history is a case
    to measure rather than assume.
  - **The agent can rewrite history inside the guest.** Harmless: only objects
    reachable from the proposed tip are exported, and import cannot move a ref
    the user owns.
  - **Submodules and LFS are untested.** Gitlinks and LFS pointers should
    transfer as the SHAs and pointer files they are, which is correct, but that
    is reasoning rather than evidence.

### 7.4 Materialisation — the only thing that crosses back

**At sandbox creation:**

1. Host walks the session's working directory, honouring `.gitignore` plus
   `Project.excludeGlobs`, capped at a configurable size (default 512 MB) and
   file count. What was excluded is *stated*, in a transcript item — an agent
   that cannot see `node_modules` should know it cannot see `node_modules`.
2. Host records `baseline.json`: every included path → SHA-256 + mode.
3. `/base` is mounted read-only over virtiofs. The guest's init does
   `cp -a /base/. /work/` — a local copy inside the guest, fast, and after it the
   host share is not touched again.
4. Guest runs, with a **separate git dir so a real repo is never disturbed**:
   ```
   git --git-dir=/work/.crucible-git --work-tree=/work init
   git --git-dir=/work/.crucible-git --work-tree=/work add -A
   git --git-dir=/work/.crucible-git --work-tree=/work commit -m baseline
   ```
   Git is the diff engine because it already solves renames, modes, binary
   files, and patch application, and because the guest image is carrying it
   anyway.

**When the model calls `propose` (or the user asks for changes):**

5. Guest produces `git diff --binary baseline` plus a structured summary: files
   added / modified / deleted / renamed, insertions, deletions.
6. Patch and summary come over vsock.
7. **Host verifies before showing anything.** For every path the patch touches,
   re-hash the current file on disk and compare to `baseline.json`. Any mismatch
   means the real tree changed while the agent worked; the file is flagged
   *conflicted* in the sheet and defaults to unchecked.
8. **Every path is validated** — this is the security boundary, so it is
   explicit and it is tested adversarially (§10):
   - reject absolute paths;
   - reject any path whose normalised form escapes the session root;
   - reject paths traversing a symlink that leaves the root;
   - reject `.git/` unless the user opted in;
   - reject the session root itself being a symlink resolved differently than at
     baseline.
9. **The approval sheet.** Unified diff per file, syntax-highlighted,
   per-file checkboxes, Apply Selected / Reject / Open in External Diff. The
   patch is *not* applied by anything the model can call. There is no `--yes`
   flag for this, at any level, ever. The C agent has `--yes` for host writes;
   Crucible's equivalent is that writes inside the sandbox need no confirmation
   at all — the confirmation moved to the boundary where it means something.
10. **Apply.** Snapshot the touched files into `undo/<n>/` first. Then
    `git apply --3way` if the project is a git repo, else write files with an
    atomic rename per file. Report per-file success; a partial application is
    reported as a partial application, not as success.
11. **Undo** restores from the snapshot. It is a menu item and a transcript
    affordance, not a hidden folder.
12. The guest re-baselines: the applied state becomes the new baseline commit,
    so the next proposal is a diff against what the user actually accepted.

---

## 8. Security

### 8.1 Entitlements

```xml
<key>com.apple.security.app-sandbox</key>              <true/>
<key>com.apple.security.virtualization</key>           <true/>
<key>com.apple.security.files.user-selected.read-write</key> <true/>
<key>com.apple.security.files.bookmarks.app-scope</key> <true/>
```

Notes that matter:

- `com.apple.security.virtualization` is required and is available with an
  ordinary Developer ID — it is **not** one of the restricted entitlements that
  needs a request to Apple.
- `com.apple.vm.networking` **is** restricted and needs Apple's approval. We do
  not want it: bridged networking is exactly the capability this design is built
  to withhold. NAT (`VZNATNetworkDeviceAttachment`) needs no special entitlement
  if §8.3's escape hatch is ever enabled.
- **App Sandbox plus Virtualization plus a 16 GB mmap is the combination to
  prove first.** It is a Milestone 1 gate (§12), not a Milestone 6 discovery.
  If App Sandbox turns out to fight the guest disk or the virtiofs share,
  the fallback is Hardened Runtime without App Sandbox — still notarisable,
  strictly worse, and a decision to make early and deliberately.

### 8.2 The threat model, stated

**What we defend against:** a model — steered by its own reasoning, or by
instructions embedded in a file it reads — taking an action the user did not
intend on the user's machine.

**How:** it has no ability to. The tool implementations do not exist on the
host. `shell` runs in a guest with no network device, whose only view of the
user's files is a read-only mount of one directory the user picked, and whose
only channel out is a patch the user reads and approves file by file.

Prompt injection therefore degrades from *arbitrary code execution on the
developer's laptop* to *a bad patch, shown to a human, in a diff view*. That is
the entire argument for the architecture, and it is worth stating in one
sentence because everything expensive in this document is bought with it.

**What we do not defend against:**

- A guest-to-host escape through Virtualization.framework itself. We inherit
  Apple's hypervisor boundary and do not second-guess it.
- A user approving a malicious patch. The sheet's job is to make the change
  legible; it cannot make the user read it.
- Model output that is simply wrong. Different problem.

**The attack surface we actually own** is the vsock protocol and the patch
applier. Both are small, both are pure functions over untrusted input, and both
are tested adversarially. Warden validates every host message; the host
validates every guest path (§7.4 step 8).

### 8.3 The networking escape hatch

Some tasks need the network — installing a dependency, reading a doc. So
`Project.sandboxNetworking` exists, defaults **off**, and when a user turns it
on:

- The VM is reconfigured with a NAT attachment. This requires a **VM restart**,
  which is stated in the toggle, because reconfiguration is not live.
- Every session in that project shows a persistent badge in the transcript
  header, and the transcript records the moment it was enabled with a
  timestamped item.
- It never turns itself on. No tool can request it. It is a settings toggle a
  human flips.

### 8.4 The model directory

16 GB of weights are mmapped from a user-selected folder held by bookmark. If
the folder disappears between launches, fail with a clear message and a picker,
not a crash — `qwasar_engine_load` returning `NULL` with a filled `err` is a
normal, expected outcome and the UI should treat it as one.

---

## 9. Deterministic simulation testing with `eta`

[`eta`](https://github.com/jessestimpson/eta) is a BEAM deterministic simulation
testing framework: it serialises the scheduler, virtualises the clock, records
every scheduling decision, replays a recorded trace exactly, and shrinks a
failing trace to a minimal counterexample. No patched BEAM, no NIF.

**We start on the `elixir` branch, and eta is an experiment with a defined
exit** (§9.5). That framing is deliberate and it constrains the integration:
nothing in the shipping guest may depend on eta being present, and removing it
must cost a build flag rather than a rewrite. The good news is that the `elixir`
branch's design gives that almost for free (§9.1).

It appears in this plan twice, for two different reasons, and they should not be
confused:

- **§9.2 — as how the warden is built.** The guest's control plane is a
  concurrency-and-time argument, and DST is how such an argument is settled
  rather than hoped for.
- **§9.4 — as a tool handed to the model.** The agent writes concurrent Elixir
  and Erlang into a live node. Giving it a way to *understand code before it
  loads it* turns hot-loading from a stunt into an engineering practice. This is
  the more speculative of the two, and it is also the more interesting.

### 9.1 What eta provides, and what the `elixir` branch changes

On `main`, instrumentation is a `parse_transform` — invisible, total, and
Erlang-only. On the **`elixir` branch** it is explicit macros instead:

> "Elixir is supported: `use Eta` in a module provides explicit macros
> (`Eta.send_after/3`, `Eta.cast/2`, `Eta.call/3`, `Eta.send/2`, etc.) that
> expand to `:eta_*` calls when `:eta` is a compile-time dependency and to
> stdlib calls otherwise — no parse_transform required."

Plus `@eta_observe` for state extraction, `Eta.log/1` for the unified timeline, a
`@before_compile` hook that wraps `gen_server` callbacks to publish state and
route replies (the Elixir-AST equivalent of `eta_transform`'s observe and
reply-routing passes), and a `mix.exs` so the whole thing is an ordinary
dependency.

**One non-obvious thing about consuming it**, found by reading its `mix.exs`
rather than by an afternoon of confusion. eta's Elixir entry point is compiled
only in its own `:test` environment:

```elixir
defp elixirc_paths(:test), do: ["lib", "test/support"]
defp elixirc_paths(_),     do: []
defp erlc_options(:test),  do: [:debug_info, {:d, :DST}]
```

Mix builds dependencies in `:prod` unless told otherwise, so a plain dependency
on eta yields the Erlang library **without the `Eta` macro module** — and
`use Eta` fails with a module-not-found that says nothing about why. The
dependency has to name the environment, which also turns on the `DST` define
that the header hangs on:

```elixir
{:eta, github: "jessestimpson/eta", branch: "elixir", env: :test, only: :dst}
```

Worth raising upstream: the package's own `files` list ships `lib`, so the
intent is clearly for consumers to have `Eta`, and `elixirc_paths` is what stops
them.

Three consequences, and the third is the one to watch:

1. **Warden can be Elixir after all.** §7.3 previously chose Erlang solely
   because eta's Elixir support was limited. On this branch that reason is gone,
   and one language plus one build tool across the whole guest is worth having.
   The Erlang fallback stays documented, because the branch is best-effort.
2. **Removal is a dependency toggle.** The macros expand to stdlib calls when
   `:eta` is not a compile-time dependency, which is exactly the containment
   §9.5 needs. The shipping warden build simply does not list eta, and the same
   source compiles to `Process.send_after/3` and `GenServer.cast/2`.
3. **Instrumentation is now opt-in per call site, and that is a real caveat.**
   A `parse_transform` catches every timer and every message in the module. An
   explicit macro catches only the ones you wrote as macros. **A hand-written
   `Process.send_after/3` is invisible to the simulator** — it reintroduces real
   time and real nondeterminism into a run that reports itself as deterministic,
   and it does so silently. This is the single most likely way for §9 to produce
   a confident, meaningless green result.

   The branch's own documentation names the sharpest instance, and it is a
   language construct rather than a function call, so no lint on module names
   will catch it:

   > **`receive ... after` is not virtualized.** The parse transform rewrites
   > this language construct; without it, a `receive ... after` timeout runs on
   > the real clock.

   So the rule for warden is concrete: **no `receive ... after`, anywhere.** A
   process that wants a deadline sends itself a message with `Eta.send_after/3`
   and matches it in a plain `receive`. That is a rule about how the control
   plane is written, not a build setting, and it is the first thing to check
   when a simulation says a timeout never fired.

   For warden, the answer is discipline plus a lint: a `dst` -profile compile
   check that fails the build on a raw `Process.send_after`, `send/2`,
   `GenServer.call/3`, or `:timer.*` in any module that `use Eta`. For
   *agent-written* code, the answer is in §9.4, and it is harder, because the
   model will reach for the stdlib call by default.

The framework's own documentation is candid that "Elixir support works and is
tested, but continued development focuses on Erlang". Treat Erlang as the
supported path for anything that must not break, which is why §9.5 exists.

#### The harness contract

A system under test supplies six required callbacks (plus two optional) through
the `eta_harness` behaviour:

```erlang
-callback init(Seed :: integer(), Config :: map()) -> {ok, state()}.
-callback processes(state())                       -> [pid()].
-callback generate(state(), rand:state())          -> {op(), rand:state()}.
-callback execute(op(), state())                   -> state().
-callback check(state())          -> ok | {violation, violation()}.
-callback terminate(state())                       -> ok.
%% optional
-callback labels(state())         -> #{pid() => term()}.
-callback check_final(settled(), state()) -> ok | {violation, violation()}.
```

and is driven by

```erlang
#{outcome := Outcome, trace := Trace} =
    eta_run:run(warden_harness, #{seed => 7, max_ops => 25,
                                  max_steps => 20000, preload => [warden]}),

#{trace := Minimal, verified := true} =
    eta_shrink:shrink(warden_harness, Trace, #{seed => 7, max_ops => 25,
                                               preload => [warden]}).
```

The modules that matter here:

| module | what it buys this project |
|---|---|
| `eta_sched` | one process runs at a time, chosen by a seeded RNG; every interleaving is reachable and reproducible |
| `eta_time` | virtual clock that jumps to the next deadline — **a 30-second warden timeout costs zero wall-clock**, which is the difference between testing timeout behaviour and not testing it |
| `eta_run` | drive a harness; returns outcome plus a trace |
| `eta_shrink` | delta-debug a failing trace to its essential decisions |
| `eta_log` | correlates scheduling with custom events on one timeline — a failure reads as a narrative, not a decision vector |
| `eta_observe` | inspects suspended process state; the only safe way for an invariant to read a halted system |
| `eta_net` | injects message-delivery faults across simulated nodes, plus `nodedown`/`nodeup` and monitor cleanup |
| `eta_transform` | the `parse_transform` that makes all of it work |

**One documented constraint still shapes the architecture: one simulation per
VM.** `eta_time` and `eta_log` use named ETS tables, so runs must be serial.
This is why simulation gets its own ephemeral node (§9.3) rather than sharing
the workspace — and why that decision survives the branch change intact.

Two pitfalls the docs call out, which become part of the `simulate` tool's
description verbatim, because a tool's contract belongs in its description:
**`execute/2` must not block**, and **`check/1` must not call into scheduled
processes**. Both fail silently, which is exactly the kind of rule a model needs
told rather than left to infer.

### 9.2 Building the warden under simulation

The warden's job is a list of things that are hard for the same reason: they are
about *when*, not about *what*.

- A request arrives on vsock; a response must be sent exactly once, with the
  right `id`, even if the workspace answers after the timeout fired.
- The workspace dies mid-`erpc`. The pending request must fail, the monitor must
  be cleaned up, the restart must happen, and the manifest must be replayed **in
  load order**.
- Two `define` calls race. The registry must not end up with two modules
  claiming one tool name.
- The VM is stopped while a `propose` is streaming a 200 MB patch.
- A soft-purge refusal (§7.2, step 3) races the process that was holding old
  code exiting on its own.

Written by hand and tested with ExUnit, every one of these is a `Process.sleep`
and a hope. Under eta they are ordinary test cases.

**The harness.** `warden_harness` implements the eight callbacks over a warden
instance plus a *simulated* workspace node:

```erlang
%% generate/2 draws from this op set:
{host_request, Id, Op}          %% a tool call arrives on the bridge
{host_cancel, Id}               %% the user interrupted
{ws_reply, Id, Result}          %% the workspace answers
{ws_crash, Reason}              %% it does not
{ws_slow, Millis}               %% it answers after the deadline
{net_partition, Duration}       %% eta_net drops distribution traffic
{define, Module, Source}        %% a hot-load lands mid-flight
{advance_time, Millis}          %% eta_time jumps to the next deadline
{vm_stop}                       %% the host pulls the plug
```

**The invariants** — `check/1` after every step, `check_final/2` once settled:

1. Every `id` receives **exactly one** terminal response. Never zero, never two.
2. No response is delivered as `ok` after its request has been reported as timed
   out. This is the classic late-reply bug and it is invisible without a
   controlled clock.
3. The registry has no duplicate tool names, and every registered name resolves
   to a loaded module.
4. After a workspace restart, the replayed manifest is order-preserving and
   contains exactly the modules that were loaded before the crash.
5. No monitor outlives the request that created it. (`eta_observe` reads this
   directly out of the suspended warden state.)
6. Warden never blocks on the workspace: the bridge remains responsive to a
   `cancel` while an `invoke` is outstanding.

Invariant 6 is the one that justifies the whole architecture, and until eta it
was an assertion in a design document rather than a property of the code.

**Build integration**, corrected by contact. The intended shape was that warden
modules `use Eta` unconditionally and the macros expand to the stdlib when eta
is absent. **That does not work, and the reason is worth writing down:** `use Eta`
requires the `Eta` module to *exist* at compile time, and eta compiles its Elixir
entry point only in its own `:test` environment (§9.1). A shipping build with no
eta dependency therefore cannot `use Eta` at all, however cleverly the macros
inside it are written. The first attempt failed with
`module Eta is not loaded and could not be found`.

So warden carries a seam of its own — `Warden.Sim`, the Elixir analogue of the
wrapper header eta's README prescribes for Erlang consumers:

```elixir
defmacro __using__(_) do
  if Code.ensure_loaded?(Eta) do
    quote do: (use Eta; require Eta; alias Eta, as: Sim)
  else
    quote do: (require Warden.Sim.Stdlib; alias Warden.Sim.Stdlib, as: Sim)
  end
end
```

Warden writes `Sim.send_after/3`, `Sim.label/1`, `Sim.log/1`. With eta on the
code path those are eta's; without it they are `Process.send_after/3` and, for
the observation hooks, nothing at all. **One source tree, two compilations, and
the shipping image carries no trace of eta** — which is the property §9.5's exit
depends on.

```elixir
defp deps(:dst), do: [{:eta, github: "jessestimpson/eta", branch: "elixir",
                       env: :test, runtime: false}]
defp deps(_),    do: []
```

One trap inside the trap: `Sim.*` are **macros**, so the aliased module needs a
`require`. Without it the calls compile as remote function calls and fail at
runtime with `Warden.Sim.Stdlib.label/1 is undefined or private`, which is a
confusing way to be told you forgot a `require`.

`make image` builds the default env. The DST build is a host-side test artefact
and never reaches `mkimage.sh`. The one exception is §9.3, where the sim node
deliberately ships DST-built beams.

**And the lint ships with it.** §9.1's third consequence — a raw
`Process.send_after` silently opting out of determinism — is caught by a compile
check in the `dst` env that fails on stdlib timing and messaging calls inside any
module that `use Eta`. Without it, warden's simulation coverage decays quietly
every time someone writes the obvious thing.

### 9.3 The sim node

`eta_sched` suspends every process in the VM and resumes one at a time.
`eta_time` replaces the clock. Neither is something to do to a node that is also
answering a user's tool calls or holding the vsock bridge open.

So `simulate` starts a **fresh, ephemeral node**:

```
warden_sim:run(Harness, Opts) ─▶ spawn `erl -noinput -sname sim_N`
                                 with the DST code path and eta on it
                              ─▶ preload the agent's modules (eta's own
                                 `preload` option does exactly this)
                              ─▶ eta_run:run/2 under a wall-clock budget
                              ─▶ collect outcome + trace, kill the node
```

Properties that fall out of this, all of them wanted:

- A simulation cannot hang the workspace, and a workspace tool call cannot
  perturb a simulation. eta's "one simulation per VM" constraint is satisfied
  per *node*, so it never becomes a global lock on the session.
- A run that deadlocks or exhausts memory kills one throwaway OS process.
- The sim node's code path is DST-built; the workspace's is not. The agent's
  module is compiled twice, once for each — and the fact that instrumentation is
  a compile-time transform is what makes that honest rather than a different
  program.
- Concurrent `simulate` calls are serialised by warden with a queue depth of
  one and a stated wait, because a session at 128K context is already the only
  thing running (§4.3) and a second simulator competing for the guest's two
  cores helps nobody.

The image carries eta plus the DST-built beams under `/opt/sim/lib`, reachable
only by a sim node's `ERL_LIBS` — a few megabytes on top of §6.2's budget, and
the only reason a testing framework ships in a production artefact anywhere in
this design. The build prints both, so the separation is visible rather than
asserted:

```
prod build carries: warden
sim node carries:   eta warden
```

### 9.4 `simulate` and `replay` — handing DST to the model

This is the part that is genuinely new, and it is worth being precise about what
it does and does not claim.

**The claim.** An agent that writes concurrent code into a node it depends on
should be able to answer "is this correct?" with evidence rather than with
confidence. Deterministic simulation is the only technique that answers that
question for concurrency in bounded time, and eta makes it a function call.

**The flow.**

```
define(source: "-module(build_queue). ...")            % compile only, no load
  ↳ "compiled. exports start_link/1, submit/2. spawns processes —
     consider simulate/2 before loading."

define(source: "-module(build_queue_harness). ...")    % the eta harness
  ↳ "compiled. implements eta_harness."

simulate(harness: "build_queue_harness", seed: 7, max_ops: 25)
  ↳ VIOLATION after 3 ops (shrunk from 21):
      1. submit(job_a)      [pid<0.312.0> queue]
      2. worker crashes     [pid<0.315.0> worker_1]
      3. submit(job_b)      -> job_a never re-queued
    invariant: every submitted job reaches done or failed
    trace: t_7f3a  (replay with replay/1)

define(source: "-module(build_queue). ... % monitor workers")
replay(trace: "t_7f3a")
  ↳ PASS. 3 ops, same schedule, invariant held.

define(source: "...", verify: "build_queue_harness")   % now load it
  ↳ simulated 25 ops across 12 seeds, 0 violations. loaded and registered.
```

The two things that make this useful rather than noisy:

- **Shrinking is not optional.** A raw failing trace is thousands of scheduling
  decisions and would consume the context window to no purpose. Warden runs
  `eta_shrink:shrink/3` automatically on any violation and returns **only** the
  minimised trace, rendered through `eta_log` as the narrative above. The model
  sees three steps, not twenty thousand.
- **`replay` is the verification story.** A concurrency fix normally cannot be
  verified — you re-run and it does not fail, which proves nothing. Here the
  exact failing schedule is a stored artefact, so "I fixed it" becomes a
  checkable claim. Traces are persisted host-side under
  `sessions/<uuid>/traces/`, survive parking and VM restarts, and are shown in
  the inspector.

**Where it will disappoint, stated now.** The model has to write an
`eta_harness`, and a harness is a small formal specification: an op generator
and an invariant. Models are markedly better at writing implementations than at
writing the invariants that would catch their own mistakes, so the likely failure
mode is not a broken simulator but a **vacuous harness** — one whose `check/1`
returns `ok` unconditionally, or whose `generate/2` never produces the
interleaving that matters. Three mitigations, in order of confidence:

1. Ship **harness templates** in the guest for the shapes an agent actually
   writes — a queue, a supervised worker pool, a cache with TTL, a retry loop.
   `simulate` names the closest template in its result when no harness is given.
2. Warden **rejects a vacuous harness**: if `check/1` never inspects state, or
   if a control run against a deliberately broken reference module fails to
   produce a violation, the harness is reported as not discriminating. A harness
   that cannot fail is worse than no harness because it produces a green result.
3. Report **coverage honestly**: ops executed, distinct schedules explored,
   seeds tried. `0 violations in 12 seeds` is a much weaker and much more useful
   statement than `PASS`.
4. **Detect uninstrumented calls and say so.** This is the agent-facing form of
   §9.1's third consequence, and it is the one most likely to bite. A model
   writing Elixir will reach for `Process.send_after/3`, not `Eta.send_after/3`,
   and the resulting run is partly real-time and wholly misreported. So warden
   scans the module under test for stdlib timing and messaging calls before
   simulating, and the result leads with them:

   ```
   simulate(harness: "build_queue_harness")
     ↳ 2 calls outside the simulation:
         build_queue.ex:14  Process.send_after/3  → use Eta.send_after/3
         build_queue.ex:31  GenServer.call/3      → use Eta.call/3
       these run in real time and are NOT explored by the scheduler.
       0 violations in 12 seeds — but coverage is partial.
   ```

   The `define` tool's description states the macro requirement up front, and
   the harness templates use the macros throughout, because the most reliable
   way to get a model to write `Eta.send_after/3` is to show it in the template
   it is copying.

**This is measured, not assumed.** M4's gate (§11) covers it: does the agent,
given these tools, actually catch a planted concurrency bug in code it wrote?
eta's own repository ships a two-phase-commit example with a planted bug and a
leader-election example with time-dependent correctness — those are the fixtures,
and they are a fair test precisely because they were not written for this.

**And it is optional, always.** `simulate` is a tool the model may call, not a
gate it must pass, except when the model itself passes `verify:`. An agent that
wants to hot-load a two-line pure function should not have to write a harness,
and a design that made it do so would simply be worked around.

#### Determinism has a precondition, and it is that the code is already loaded

The DST suite was **flaky: 12 failures in 30 runs**, on a hard-coded `1..64`
seed range. It reported a real-looking violation — `{:unanswered, [...]}`, ids
that never got a response — which is exactly what a genuine bridge defect would
look like. It was not one.

The shape of the failures gave it away. Always **exactly one** seed, always one
of the **first three**, and each seed failed *identically* when it failed. So the
seeds were individually deterministic; what varied was which one was unlucky.
Running a single seed twelve times in one process settled it:

```
SEQ seed1: 1:unanswered/11  2:ok  3:ok  4:ok  5:ok  6:ok  7:ok  8:ok  9:ok  10:ok
```

Only the **first `:eta_run.run` in a process** differs. Diffing `:code.all_loaded()`
across the first two runs says why: **165 modules load during run 1, zero during
run 2** — warden's, eta's own, parts of Elixir, and a good deal of OTP.

A module load is a blocking call into `code_server`, a process the scheduler does
not control and cannot interleave. A scheduled process that touches an unloaded
module therefore stops for a reason the seed did not choose, and the interleaving
diverges from the one the seed selected. **The first run measures the code server;
only the runs after it measure the system.**

Two fixes, in `test/dst/`:

1. **A warm-up run, discarded**, plus one `inspect/2` call — the `Inspect`
   protocol's implementations arrive lazily and the bridge reaches for them only
   on the `bad_reply` path, which a single warm-up seed may not take. Six modules
   were still landing mid-sweep before that line existed.
2. **The precondition is asserted rather than assumed.** The sweep snapshots
   `:code.all_loaded()` before and after and fails, *naming the modules*, if any
   arrived in between. A seed that ran while a module loaded produced a result —
   pass or fail — that means nothing, and saying so is better than flaking with a
   violation that reads as a real defect.

**40 runs, 40 passes** afterwards. And the suite is still discriminating: removing
the `{:error, msg}` clause from `Bridge.respond/3` — the original bug — is caught
immediately, with the CaseClauseError named.

Separately, `generate/2` ended with `Enum.random/1` for the `bad_reply` shape.
That draws from the **process dictionary's** `:rand` state, seeded from real
entropy, rather than from the state eta threads through the generator. Fixed by
threading the draw. This was worse than a second source of flakiness: `shrink`
and `replay` both re-run `generate/2`, so either would have drawn a *different*
shape than the trace it was trying to reproduce. **A generator that cannot be
replayed takes the whole point of the framework with it**, and nothing would have
reported that — the shrink would simply have produced a trace that did not fail.

The general lesson is the one §9.5 should weigh: eta's guarantee is conditional on
the system being fully loaded and fully instrumented, and neither condition
announces itself when broken. Both failures here were silent, both produced
plausible-looking output, and both were found only by asking whether the result
was reproducible rather than whether it was green.

This bears directly on §7.2. `define` **loads code at runtime, by design** — so any
simulation running while the agent compiles a module is, for that moment, not
deterministic. The warm-up trick does not help there, because the load is the
point. If the model is to trust `simulate` on code it has just written, `define`
and `simulate` have to be ordered so that no load happens inside a run.

#### Open: most tools will touch the filesystem, and eta cannot see it

The `define` schema now says plainly what the tool is *for* — moving work off
the token stream, since generation at ~6 tok/s is by far the slowest thing the
model does. A tool that reads, transforms or checks files under `/work` does in
one `invoke` what would otherwise cost hundreds of generated tokens and several
round trips through `read` and `edit`. That is the case worth writing a module
for; a pure helper used once is cheaper inline with `elixir`.

Which means **the expected tool takes a path and touches the disk, and is not a
pure function of its arguments** — and that changes the testability profile
this section assumed. eta gives determinism over *scheduling*: message order,
timer expiry, process interleaving. It says nothing about the filesystem. A
harness for a path-taking tool currently has three options, none good:

- run against a real temp tree, which reintroduces exactly the nondeterminism
  eta exists to remove, and makes a shrunk trace unreplayable;
- confine the tool to a pure core with I/O at the edges, which is the right
  advice and which the model will not reliably follow;
- test nothing but the pure part, which is how a harness ends up vacuous —
  the failure already seen once in M4, where a harness asserted nothing for its
  entire first life and only a planted bug revealed it.

So: **eta likely needs a simulated filesystem** — an in-memory tree the sim node
owns, with `File`/`Path` calls routed through it under simulation and passing
through untouched in production, in the same shape as `eta_time` and `eta_net`.
That would make a file-touching tool as replayable as a concurrent one, and
would let `simulate` shrink a failure involving a missing file, a permission
error, or a partial write.

Not scoped here, and deliberately not built before the §9.5 review: it is a real
extension to eta itself, and whether it is worth writing depends on the answer
to "did DST earn its place", not the other way round. Recorded so the review has
it. If the answer is that eta goes, this goes with it.

### 9.5 The experiment, and how it ends

eta is **an experiment with a defined exit**, and saying so changes how it is
integrated rather than how seriously it is taken. Three requirements make the
exit cheap:

1. **No shipping dependency.** Warden's default build has no `:eta` dep and its
   macros expand to stdlib calls (§9.2). Removing eta removes a dev dependency
   and a test directory. The production guest is byte-identical either way.
2. **No architectural dependency.** The three-node split (§7.3) is justified by
   isolation and by the bridge staying responsive, both of which hold with no
   simulator anywhere. Deleting the sim node deletes a node, not a design.
3. **The tool surface does not shrink.** This one is a trap worth naming: §2.2
   freezes the tool list because it is the system turn, so **removing
   `simulate` and `replay` later would change the system turn and re-prefill
   every session in the application.** They therefore stay in the surface
   permanently, and if eta goes they return `"deterministic simulation is not
   available in this build"` — which the model handles the way it handles any
   other tool result. Withdrawing a tool is more expensive than never shipping
   it; ship these two knowing that.

**The review is at the end of M4**, against three questions:

| question | how it is answered |
|---|---|
| Did DST make warden materially better? | Count the invariant violations §9.2's harnesses caught that a conventional test suite would have missed. Zero is a real answer and it means eta was not worth it for warden. |
| Can the model use it? | M4 gate (b): does the agent catch a planted concurrency bug given `define` + `simulate` + `replay`? |
| Is the `elixir` branch holding up? | Bugs hit, time lost, how much of warden had to be shaped around best-effort support. |

Four outcomes, all of them acceptable:

- **Keep both.** The intended result.
- **Keep it for warden, drop it for the agent.** Likely if the model cannot write
  a discriminating harness. `simulate` degrades to a warden-authored harness
  library the agent selects from — and `replay` keeps its value regardless, since
  a stored failing schedule is useful even when the model did not write the spec
  that found it.
- **Move warden to Erlang and `main`.** If the `elixir` branch costs more than it
  saves. The node topology is unchanged; workspace stays Elixir; §9.2's harnesses
  port with the language.
- **Rip it out.** Warden keeps its stdlib-expanded macros, the sim node goes, the
  two tools stay and report unavailable, and this section becomes a record of
  what was tried. That is a normal outcome for an experiment and it costs a
  build flag.

## 10. Correctness

The parent tree's rule is CPU reference twins and golden vectors against
mlx-vlm. The app's equivalents:

**Bridge fidelity — the highest-value tests.** For a fixed conversation, the
Swift `ChatTemplate` marshalling must produce byte-identical token arrays to
`qwasar_apply_chat_template` driven from C. Generate goldens with a small C
harness in `Tests/`, commit them, and compare. A silently different system turn
is a silently different model, and it would be invisible until quality dropped
for no reason anyone could name.

**Round-trip.** `encode` → `token_bytes` → UTF-8 assembly must reproduce the
input for a corpus that includes emoji, CJK, combining marks, and a token that
splits a codepoint. This is the §3.4 bug, caught by a test rather than by a user
seeing `\u{FFFD}`.

**Speculation equivalence.** Same property `tests/test_verify` protects, one
level up: a session with MTP enabled and one without must emit identical token
sequences from the same prompt at `temperature = 0`. If Crucible's Swift
draft/verify loop breaks this, it is broken.

**Patch application, adversarially.** A table-driven suite over hostile patch
paths: `../../etc/hosts`, `/etc/hosts`, `a/../../b`, a path through a symlink
that leaves the root, a path with a NUL, a path that differs only by Unicode
normalisation, `.git/hooks/pre-commit`, a rename whose destination escapes. Each
must be **rejected**, and the test asserts on the rejection reason, not just on
failure. This is the security boundary; it gets the most tests in the project.

**Baseline drift.** Apply a patch after mutating the working tree underneath;
assert the conflict is detected and the file defaults to unchecked.

**Guest, warden.** Not ExUnit — `eta`. Every property listed in §9.2 is a DST
harness run across a seed sweep in CI, with `eta_shrink` on any failure and the
minimised trace committed as a regression fixture. Registry duplication, purge
refusal races, workspace restart with manifest replay, late replies after a
timeout, and monitor leaks are all *when* bugs, and a `Process.sleep`-based test
for any of them is a test that passes for the wrong reason.

**Guest, workspace.** ExUnit, ordinarily: the six file tools against a fixture
`/work`, the `Crucible.Tool` behaviour, JSON marshalling, malformed and
oversized frames. This half is logic, not timing, and does not need a simulator.

**The `simulate` tool itself.** Two tests that matter more than they look:
a harness that cannot fail must be *reported* as non-discriminating (§9.4,
mitigation 2), and a violation must come back shrunk — assert on the step count
of the returned trace, not just on the violation, because an unshrunk trace is a
context-window denial-of-service dressed as a passing feature.

**`crucible-cli`.** A headless driver that loads the engine, runs one scripted
session end-to-end against a fixture project, and asserts on the final patch.
This is the CI target, and it is the only way to test the whole thing without a
UI harness. It matches the parent's culture: everything is runnable from a
terminal.

**Context ceiling.** A session driven to exactly `context` tokens must produce a
compaction offer (§2.4) and never an engine error. Drive it with a scripted
session in `crucible-cli` against a small context override, so the test costs
seconds rather than a 128K prefill.

**What is not automatable, and therefore gets measured by hand:** VM boot time,
first-launch kernel compile, cold 128K prefill, and — the two numbers §4.4 owes
this document — park and restore wall-clock at 8 GB. Record them here the way the
parent records its throughput numbers, and update them when they move.

---

## 11. Interaction risks worth naming now

**A generic `invoke` may cost tool-calling accuracy.** §2.2 forces the fixed
surface, but a model trained to call `grep_ast(pattern: ...)` directly may call
`invoke(name: "grep_ast", args: "{...}")` less reliably — nested JSON inside an
XML parameter is a format it has seen less of. **Measure this in Milestone 4**
against a small suite of agent-written tools. If accuracy is bad, the fallback
is to accept a re-prefill when the tool set changes, amortised by only allowing
tool-surface changes at user-turn boundaries and by the system-prefix
checkpoint. Do not assume; measure.

**Sandboxed work can drift from a moving tree.** Long sessions against a tree
the user is also editing will conflict. §7.4 step 7 detects it; it does not
prevent it. Consider a periodic re-baseline offer for sessions older than an
hour.

**The model must understand that it is in a box.** The system turn has to say
so, concretely: `/work` is a copy, changes are not real until proposed and
approved, there is no network, the sandbox may restart. A model that thinks it
edited the user's file and then does not propose the patch has done nothing at
all.

**VM boot is on the critical path of the first tool call.** Two seconds is fine.
Ten is not, and it is easy to reach ten by putting OpenRC in the image. Boot
time is a tracked number.

---

## 12. Milestones

Ordered so that every gate that could invalidate the design is crossed early.

### M0 — It links, it loads, it streams *(done)*

`libqwasar.a`, the `CQwasar` module map, a signed `Crucible.app` that loads the
model and streams a prompt. Built, and the gate passes.

**Measured on the target host** (M4 / 32 GB / macOS 26.5.1 / Swift 6.3):

```
$ make gate-full

-- profile (PLAN.md 2.3)
physical 34.36 GB · Metal working set 26.80 GB · reserve 85%
weights 16.02 GB · per session 6.27 GB
→ 1 live session at 90112 tokens, MTP off

-- entitlements and virtualization
  home: /Users/…/Library/Containers/dev.crucible.Crucible/Data
  sandbox: granted
  virtualization: granted
  config: validates (1 cpu, 4 MB, no network device)
  VZVirtualMachine: instantiated, state 0

-- engine (the 16 GB mmap)
  loaded in 5.5s · 64 layers · vocab 248320 · context 90112
  phys_footprint 5.06 GB

-- gate
  OK  App Sandbox active
  OK  Virtualization entitlement usable
  OK  16 GB mmap under those entitlements
  GATE PASSES

-- generation
  73 prompt · 159 generated (103 reasoning) · prefill 11.41s · decode 26.81s · 5.93 tok/s
  ended: EOS
```

**5.93 tok/s** sits inside the parent's predicted 5.5–7 band, from Swift, with no
speculation and no tuning — so nothing about being in an app costs throughput.
`phys_footprint` of 5.06 GB immediately after a load of 16 GB of weights is
`mmap` doing its job: pages fault in as the forward pass touches them.

**What the gate turned up:**

- **Ad-hoc signing is enough.** `com.apple.security.virtualization` is granted
  and `VZVirtualMachine` instantiates with `codesign -s -`. No Developer ID was
  needed to cross this gate, which was the open risk in §8.1.
- **The sandbox is genuinely enforcing.** With the shipping entitlements and no
  file grant, the load fails with `cannot open …/config.json`. That failure is
  the proof, and the `gate-full` target re-signs with a read-only temporary
  exception purely so the third leg is provable from a terminal — the shipping
  path is still the user-selected folder and a security-scoped bookmark.
- Three design changes came out of building it, each recorded where it belongs:
  no `.xcodeproj` (§3.3), the deployment-target correction (§3.3), the withdrawn
  `qwasar.h` accessor (§2.3), and a fifth sharp edge on the bridge (§3.4).

**Not done in M0, deliberately:** the golden-vector test for the chat template
(§10) — M0 proves the template *runs*, not that it is byte-identical to the C
path, and that test is M1's first task rather than its last.

### M1 — The harness, at profile context *(in progress)*
Projects, sessions, persistence, transcript, reasoning fold, streaming, prefill
bar, interrupt. Tools are *host-side and read-only* (`read`, `list`, `grep`)
purely as a stand-in. A usable read-only coding assistant that never writes
anything. Ship-quality UI from here on; do not defer it.

**Built and verified:**

- **The golden-vector suite passes**, 15 cases, byte-identical to
  `qwasar_apply_chat_template` driven from C — including the tool-schema system
  turn, the three reasoning-effort levels (which are genuinely different: 49, 23
  and 61 tokens), a conversation replaying a tool call, and content that spells
  `<|im_start|>` to prove the injection-safe encoder is the one being used.
  `Tests/gen_golden.c` is the oracle; `Tests/golden.tsv` is committed.
- **39 tests total** under `make test`: the goldens, an adversarial `PathGuard`
  suite that asserts on rejection *reasons* rather than on failure, and a UTF-8
  reassembly suite that drives the assembler one byte at a time.
- **The agent loop works**: multi-step tool use with narration between calls,
  the XML parser reused from `qw_tool_parse`, early stop at the closing tag, and
  continuation via `qwasar_render_tool_result` rather than re-rendering.
- **Prefill economics measured** (§2.5), which changed the tool caps from the C
  agent's 256 KB to 8 KB and put an ETA on the progress bar.

**An end-to-end agent turn, measured**, sandboxed and signed, against this
repository:

```
> What does qw_edit_apply do when the old text matches in more than one place?
  Use grep to find it, then read only what you need.

  → grep pattern=qw_edit_apply
  → read path=qwasar_toolcall.c

  …returns QW_EDIT_AMBIGUOUS … the moment a second match is found it breaks
  out of the loop — "ambiguity is decided; stop looking."

  3352 prompt · 728 generated (389 reasoning) · 2 tool calls
  prefill 110.7s (30 tok/s) · decode 150.0s (4.85 tok/s) · context 4078/90112
  history: 4078 tokens (16312 bytes to persist)
```

The answer is correct against `qwasar_toolcall.c:200-215`, comment quoted
verbatim. Four and a half minutes for that turn, and §2.5 says where it went:
**110 s of it was prefill, 150 s decode** — a ratio no amount of UI work
changes, and the reason the tool caps landed where they did.

**Not verified, and it should be said rather than implied:** the two NSOpenPanel
flows — choosing the model, adding a project — cannot be driven from a terminal,
and this session had no screen access to drive them by hand. What is confirmed
about the window is that it launches, lays out (a 268 px sidebar against an
1180 px detail pane), creates its store in the sandbox container, and logs no
faults. The panels themselves, and everything downstream of them in the UI, are
unexercised.

**Session switching is correct, not merely deferred.** An earlier cut closed the
old session and opened a fresh one, which silently restarted the conversation
while the transcript on screen still showed everything that had been said —
nothing in the UI would have looked wrong. Reopening now replays the recorded
tokens, restoring from the checkpoint cache where it can (§4.4: 0.2 s for a
3656-token session) and re-prefilling where it cannot.

**Still open in M1:** the compaction successor flow (§2.4) is detected and
reported but not yet offered as an action, and M6 still owns the explicit-path
checkpoint API that makes parking reliable at full context.

The profile lands here, not in M6: context runs at whatever §2.3 derives for the
host, and the live-session bound is enforced from the first session. Everything
downstream — park-on-switch, the disk quota, the compaction successor flow — is a
consequence of those two numbers, and building against a comfortable 32K first
would mean discovering all of it twice.

**Gate: §2.3's tier table survives contact with a real allocator.** Its numbers
are derived, not measured. On the 32 GB target that means a session at 98304
allocates, prefills cold, and leaves the machine responsive with a VM running —
and if it does not, the reserve moves and the table is rewritten here before
anything else is built on it.

### M2 — A VM that answers, and a warden that is tested *(done)*
Golden image, `clonefile` per session, boot, vsock port program, warden replying
to a ping. No agent involvement.

**eta lands here, with the first line of warden code**, on the `elixir` branch.
The `:dst` mix env, `use Eta` throughout warden, the uninstrumented-call lint,
and a harness covering invariants 1, 2 and 6 from §9.2 — not a later hardening
pass, because the bridge's request/response correlation and timeout handling are
exactly the code that is cheap to write correctly under simulation and expensive
to repair after it ships in an image.

This is also the first honest read on the branch. If `use Eta` fights warden's
supervision tree, that is a §9.5 signal arriving early and cheaply, while warden
is small enough to move to Erlang in a day.
**Gates: boot-to-warden-ready under 2 s; the warden harness green across a
64-seed sweep in CI; the default (non-DST) warden build has no `:eta` dependency
in its lock file.**

**Measured, and the first gate passes:**

```
$ make guest && make sandbox

-- disk
  cloned disk.img in 0.0001s (copyOnWrite)
  golden allocates 209 MB, the clone 209 MB
-- configuration
  2 cpu · 2048 MB · 1 disk · 1 share · 1 vsock · 1 entropy
  network devices: 0 — the guest has no way out
-- boot
  started in 0.09s
  vsock connected after 0.55s (5 attempts)
-- warden
  ping → pong (3 ms)
  info → otp=27 erts=15.2.7.4 elixir=1.18.4 kernel=6.12.103-0-virt
         work_files=23 schedulers=2 memory_mb=29
  list . → 3 entries in /work
  list ../../etc → refused: path escapes /work
  unknown op → answered: no such op: nonsense

  OK  guest boots with no network device
  OK  vsock control channel connects
  OK  warden ready in 0.55s (target < 2s)
  OK  warden answers correctly
  GATE PASSES
```

**boot-to-warden-ready: 0.55 s.** The guest image is 209 MB allocated, and the
per-session clone is instant and free, as §6.3 claimed.

#### Four things contact changed

- **VZLinuxBootLoader, not EFI.** §6.1 chose EFI so the guest could own its
  kernel updates and the image could be one artefact. On contact that buys a
  GPT, an ESP, a FAT filesystem and a bootloader to install into it, all to
  arrive at a kernel the host already has. `VZLinuxBootLoader` takes vmlinuz and
  initramfs directly, which makes the disk a bare ext4 filesystem with no
  partition table at all. The kernel is now versioned with the image built
  against it, which is the more honest arrangement anyway.
- **The kernel has to be unwrapped.** Alpine's aarch64 kernel is an EFI *zboot*
  image — `MZ`, then `zimg`, then a gzip payload. Virtualization.framework wants
  the raw arm64 `Image` with `ARM\x64` at offset 0x38, and handing it the
  wrapper fails at `start()` with *"Internal Virtualization error"* and not one
  word more. `mkimage.sh` unwraps it, and `SandboxHost` checks the magic so the
  message says what is actually wrong.
- **Elixir cannot come from apk.** Alpine's `elixir` package depends on
  `erlang26` and conflicts with `erlang27`, and OTP 27 is not negotiable here
  (§6.4 decodes JSON with its built-in `:json` module). So Erlang comes from apk
  and Elixir from its own project's precompiled `elixir-otp-27` release, which
  is nothing but `.beam` files.
- **`VZVirtioSocketConnection` owns its file descriptor** and closes it on
  deallocation. Returning the fd and letting the connection go out of scope
  produces a socket that connects and instantly hangs up — indistinguishable,
  from either end, from the guest closing the channel.

#### The five seconds, and where they actually were

Boot-to-warden-ready was **5.68 s** at first, against a 2 s target. Three
guesses were wrong before a measurement was right, and they are recorded so they
are not retried:

| suspected | result |
|---|---|
| the warden being compiled from source on every boot | precompiled to `.beam`; 5.52 → 5.39 s. Not it. |
| no entropy device, so the BEAM blocks seeding | added `VZVirtioEntropyDeviceConfiguration`; 5.39 → 5.49 s. Not it. |
| `virtio_rng` not loaded, so the device was inert | isolated afterwards: no effect on its own. Not it. |

The measurement that settled it was a probe in the guest's init: **a bare
`erl -noshell -eval 'halt(0).'` took 5.14 s.** Elixir added 0.09 s on top. The
BEAM resolves the local hostname while starting, and with no `/etc/hosts` entry
for it and no nameserver to ask, that lookup waits out the resolver's timeout.
Two lines of `/etc/hosts` took a bare `erl` from **5.14 s to 0.10 s**.

The lesson is the one the parent tree keeps relearning: it looked like a slow
runtime, and it was a name lookup. The entropy device stays — a guest with no
randomness source is a bad foundation — but it stays labelled as not having been
the problem.

#### The warden is now a supervised mix project

Not a script. `Warden.Application` supervises `Warden.Bridge` (the port, the
framing, and the id correlation §9.2's invariants are about) with
`Warden.Dispatch` as the pure request→reply half, which is what lets a
simulation drive the contract without a socket in the way. `Warden.Sim` is the
eta seam.

Three more bugs contact produced, each a silent failure of a different kind:

| symptom | cause |
|---|---|
| guest booted, warden never spoke, restart loop never fired | `application:ensure_all_started` returned `{error, {logger, …}}` and the `-eval` ignored it, then blocked on `receive`. Only `elixir/ebin` was on the path; Elixir ships `logger` as a separate OTP application. `ERL_LIBS` at the Elixir `lib` root fixes it, and the eval now pattern-matches `{ok, _}` so a failure crashes loudly. |
| `Warden.Sim.Stdlib.label/1 is undefined` | macros need `require`, not just `alias`. |
| `runtimes=none` from a guest that has them | PID 1 inherits an almost-empty `PATH`, so the warden's `mise` lookup failed and was reported as "no runtimes". Init sets `PATH` explicitly and the warden calls mise by absolute path. |

#### The simulation gate, and what it caught

Both M2 gates are green. `make guest` now refuses to produce an image unless the
control plane is fully instrumented and its simulation passes:

```
lint: warden is fully instrumented
1 test, 0 failures          ← 64 seeds, 0.3s wall clock
prod build carries: warden  ← asserted: no eta in what ships
```

**0.3 seconds for 64 simulations**, each with one-second timeouts in it. That is
`eta_time` doing exactly what it promises: a deadline costs a run nothing.

The bridge had to be rewritten first. It called `Warden.Dispatch.run/1` inline,
which meant **one slow op stalled the whole control channel** — invariant 6
violated by construction, and no test it had could have shown it, because every
op it had was instant. It is now non-blocking: requests go to a long-lived
`Warden.Worker`, deadlines are `Sim.send_after/3` timers, and a late reply is
dropped rather than delivered. Demonstrated on the real guest, not only in
simulation:

```
  ping during a 3s request → pong after 1ms
  the slow request finished → slept (3003 ms)
```

A long-lived worker rather than a task per request, deliberately: eta's
scheduler controls a fixed process set, so a design that spawned per request
would be a design the simulator cannot see.

#### The harness was vacuous, and only planting a bug showed it

This is §9.4's mitigation 2 turned on its author, and it was needed.

The first suite passed 64 seeds immediately. Suspicious of a green that arrived
too easily, the timeout path was deliberately broken — the handler answered the
host but forgot to retire the request, so a late reply would answer the same id
twice. **The suite stayed green.** So did a second plant. The harness was
checking nothing at all.

The cause: `eta_observe.read/1` returned `:undefined`, so every invariant that
read observed state short-circuited to `:ok`. And the reason for *that* is worth
recording, because it will catch anyone writing Elixir against this framework:

> **eta's observe pass wraps a module only when `:gen_server in @behaviour` —
> the Erlang atom.** Elixir's `use GenServer` sets `@behaviour GenServer`, the
> Elixir module. So an ordinary Elixir GenServer gets no observability, no
> warning, and no error: `read/1` simply returns `:undefined` for ever.

The fix is one line per module, `@behaviour :gen_server` alongside
`use GenServer`. Worth raising upstream: eta's `__before_compile__` could accept
`GenServer` as well as `:gen_server` and this class of silent no-op would not
exist. As it stands the failure mode is a green suite that means nothing, which
is the single worst outcome a testing framework can produce.

With observation working, the same planted bug is caught and shrunk:

```
seed 1: {:violation, {:responded_twice, [3]}}
minimal trace (5 steps):
[op: {:request, 20, "ping"}, step: 0, clock: 1000, step: 1, step: 0]
```

Five steps: a request, a scheduling step, the clock jumping to the deadline, two
more steps. That is the narrative §9.4 promises, and it is what makes a
violation actionable rather than a wall of decisions.

**Every green result in this project's guest is now backed by a demonstrated
failure.** A harness that has never been shown to fail is not evidence.

#### The lint

`Guest/warden/lint_sim.sh`, run as a build step. It rejects
`Process.send_after`, `Process.cancel_timer`, `Process.sleep`, `GenServer.cast`,
`GenServer.call`, `:timer.*`, and — with a small awk state machine rather than a
grep, because it is a language construct — `receive ... after`. `lib/warden/sim.ex`
is exempt, being the seam itself.

Verified in both directions: silent on the real code, and on a planted module it
reports both the timer and the `receive ... after`, naming the line the receive
opened at.

### M3 — Tools move into the guest *(done)*
The six file/shell tools reimplemented in Elixir against `/work`, the virtiofs
baseline copy, the tool surface swapped over. The agent now works entirely in the
sandbox and nothing it does is visible to the user's tree. Useless and safe —
which is the correct order.

~~Warden's workspace lifecycle arrives with it, and so do §9.2's invariants 3, 4
and 5.~~ **Moved to M4, and the reason is that they cannot be tested here.**
Invariants 3, 4 and 5 are about the *registry* and the *module manifest* — no
duplicate tool names, order-preserving replay after a workspace restart, no
monitor outliving its request. All three are properties of things that do not
exist until the agent can define a tool, which is M4. Writing a harness for them
at M3 would mean inventing a manifest to replay, and a simulation over invented
state proves something about the invention.

What M3 *does* own from §9.2 is already in place: invariants 1, 2 and 6, and the
bridge rewrite they forced.

#### Built and verified

- **The six tools in Elixir against `/work`**: `read`, `write`, `edit`, `list`,
  `grep`, `bash`. Every path goes through the guest's own confinement check —
  the other half of the rule the host also enforces (§7.4 step 8), because
  either side being wrong is a bug and neither is a reason to trust the other.
- **`edit` matches `qw_edit_apply` exactly.** Its description is the C agent's,
  verbatim, and the golden vectors pin that description into the system turn —
  so the model is told the C implementation's contract and the guest has to *be*
  it. `test/unit/warden_edit_test.exs` is the C suite's cases ported one for
  one: whole-line anchoring, ambiguity refused rather than guessed, a trailing
  newline on `old` treated as presentation, deletion taking its line terminator,
  a file without a trailing newline neither gaining nor losing one. 15 cases,
  all green.
- **29 guest tests** under `MIX_ENV=test`, split from the simulation suite by
  `test_paths` because they are different kinds of thing: the unit suite is
  logic and runs anywhere, the DST suite needs eta on the code path and only
  the `:dst` environment has it.
- **The tool surface grew from three to six**, which changes the system turn, so
  `Tests/golden.tsv` was regenerated deliberately and both paths re-verified.
  845 tokens now, against 515. §2.2 freezes this list at M4; until then a change
  here is a `make golden` and a read of the diff.
- **`SandboxToolRunner`** on the host: a tool call the model produced is
  validated, sent over vsock, and run by the warden. The host runs nothing.
  `ToolRunner` (M1's read-only host stand-in) stays behind the same protocol for
  a session with no guest image, and the session header says which world it is
  in — "can this change my files" is not a thing to leave implicit.

#### The milestone, demonstrated

```
> Use the edit tool to make greet() return an f-string instead of
  concatenation, then read the file back to confirm.

  sandbox: booted in 0.58s (copyOnWrite clone), tools run in /work

  → grep pattern=def greet      greet.py:1:def greet(name):
  → read path=greet.py          def greet(name):  return "Hello, " + name
  → edit path=greet.py          edited greet.py
  → read path=greet.py          def greet(name):  return f"Hello, {name}"

  Done. `greet()` in `greet.py` now returns an f-string. I read the file back
  to confirm — the change is in place, and `farewell()` was left untouched
  since you only asked about `greet()`.

  1035 prompt · 366 generated (140 reasoning) · 4 tool calls
```

And the user's file, afterwards:

```
$ shasum build/m3proj/greet.py
dca28cb792b72c8d50b1ae7d96d097c11c1c65b6      ← byte-identical, before and after
```

The agent searched, read, edited, and verified. Its account of what it did is
accurate: the file *does* now contain the f-string — in `/work`, the guest's own
copy, on a disk that is a copy-on-write clone of a 540 MB image. The user's tree
was mounted read-only, copied once at boot, and unmounted before the warden
started.

That gap between "the agent correctly edited the file" and "nothing on your disk
changed" is the entire milestone. **M5 is what closes it, under approval, and
nothing before M5 can.**

#### The workspace node, and a correction to §7.3

The agent's node is up, reachable, and — the part that matters for M4 —
survivable:

```
  workspace → up node=workspace@localhost otp=27 restarts=0 uptime_s=0
  eval 1..10 |> Enum.sum() → 55
  ping with the workspace dead → pong
  workspace recovered → up ... restarts=1 uptime_s=0 last_exit=1
```

The node is killed, the bridge answers a `ping` anyway, and the node comes back
without anything asking it to. That is the property M4 depends on, because
agent-written code crashes, and it is proven by causing the crash rather than by
reasoning about the supervisor.

**§7.3's table was wrong about where the file tools live.** It put `Crucible.Fs`
— the six of them — on the workspace node. They are in warden, and the reason is
recovery: *an agent that has broken its own workspace must still be able to read
and edit files in order to fix it.* A recovery path that runs on the thing that
broke is not a recovery path. So:

- **warden** — the bridge, the six file tools, the workspace lifecycle, and
  (M4) the registry, because the manifest is durable state the agent must not be
  able to corrupt by crashing.
- **workspace** — the agent's own modules, the eval shell, and whatever `invoke`
  dispatches into.

Two bugs, both instructive:

**Erlang distribution needs loopback, even between two nodes on one machine.**
The guest is configured with no network device at all (§8.2), and `lo` was down,
so `epmd` failed to listen and every warden start died with
`Protocol 'inet_tcp': register/listen error: enetunreach` — a restart loop with
no other symptom. `ip link set lo up` fixes it, and it does *not* weaken the
security property: there is still no interface for anything to leave by. This is
a socket between two processes that happens to speak TCP.

**A hardcoded node name named a node that did not exist.** `-sname workspace`
takes its host part from whatever the resolver calls the machine, and with
`127.0.0.1 localhost crucible` in `/etc/hosts` that is `localhost` — not the
hostname init had just set. The symptom was a workspace that started perfectly
and was never reachable. It is now derived from warden's own node, which is
correct by construction since the same init starts both.

### M4 — Self-modification, DST in the agent's hands, and the eta review *(gate (a) passed)*

**`define` → `tools` → `invoke`, and it survives the node dying:**

```
  define → WordCount: exports __info__/1, name/0, run/1, schema/0
           registered as the tool "wordcount" — call it with invoke
  tools  → wordcount  (WordCount v1) %{"args" => ["path"],
           "description" => "counts words in a file under /work"}
  invoke wordcount → 3 words
  a name clash does not hijack the tool → loaded, invoke still gives 3 words
  the tool survives its node dying → invoke works again after the workspace
                                     restarted
```

The last line is the one that matters. The workspace is killed, warden replays
the manifest it holds, and the agent's own tool answers again — because the
registry lives in warden and not on the node that crashes (§7.3). Invariants 3
and 4 from §9.2, demonstrated by causing the failure rather than reasoning about
the supervisor.

#### Three bugs, and one of them is about eta

**The bridge died of a reply shape.** `Crucible.Loader.invoke/2` returns
`{:error, message}`; `respond/3` matched only `{:ok, _}` and `{:error, _, _}`.
The `CaseClauseError` killed the bridge inside `handle_cast`, and **the request
that caused it received no response at all** — invariant 1 broken, not by a race
but by a shape. It now has a clause for the two-tuple and a catch-all, because
the bridge must not be able to die of something a tool returned.

**The simulation did not catch it**, and the reason is worth stating plainly: a
harness can only find what its `generate/2` can express, and mine only ever
produced well-formed ops. That is not a failing of eta — it is the vacuous-
harness problem from §9.4 in a subtler form. The generator now emits `bad_reply`
ops with four malformed shapes, and re-planting the bug proves the coverage is
real:

```
seed 1: {:violation, {:unanswered, [20, 19, 18, 17, 16, 14, 13, 11, 8, 7, 3]}}
```

Every request after the bridge died. **The lesson generalises: after fixing a
bug the simulation missed, extend the generator, then re-plant to prove it would
now be caught.** A harness that grows only when a bug is found is a harness that
is always one class of bug behind.

One rough edge worth recording: `eta_shrink` left that trace at **69 steps** for
a violation reproducible in about two. eta's own docs name a plausible cause —
step renumbering after deletion leaves seemingly unnecessary operations in place.
See §9.4's simulation results before drawing a conclusion from it, though: with a
well-formed harness, shrinking produced **11 verified steps**, so 69 is not the
general case.

#### `simulate`, working

An ephemeral node per run, the agent's modules compiled onto it, eta driving
them, the node killed when the run returns. Demonstrated on a module with a
genuine timing bug — a flush that adds its pending batch to the total and
forgets to clear it, so a second flush counts the same increments again:

```
  define Tally → ok (flagged concurrent, simulate suggested)
  define TallyHarness → ok
  simulate TallyHarness →
      VIOLATION on seed 1: {:violation, {:overcounted, %{total: 83, added: 39}}}
      (8 of 8 seeds failed)
      minimal trace (11 steps, verified still fails):
        op    {:add, 2}
        step  0
        op    :advance_time
        step  0
        op    {:add, 5}
        step  0
        op    :advance_time
        ...
```

**Eleven steps, verified.** Add two, flush, add five, flush — and the second
flush counts the two again. That is the narrative §9.4 promised, and it is
short enough to hand to a model.

Three things this cost, and each is now built into the tool rather than left as
lore:

- **The registry had to become a module registry, not a tool registry.** It only
  recorded modules that implemented `Crucible.Tool`, so the harness — which by
  definition is not a tool — never reached the sim node, and every seed failed
  with `:undef` before executing an op. Replay had the same hole for helper
  modules.
- **A run that did not happen was reported as a run with no violations.** Eight
  seeds raised before executing anything, failed to match `%{outcome: _}`, and
  the sweep printed *"0 violations in 8 seeds"*. That is the vacuous green this
  section exists to prevent, produced by the reporting layer rather than by the
  harness. Errors are now a hard failure reported on their own, and `simulate`
  says plainly that nothing was explored.
- **`already_started` is the first trap any harness author meets** — including
  the author of this document, twice. eta calls `init/2` once per run and many
  times while shrinking, so a harness that starts a process under a registered
  name works exactly once. `simulate` now recognises the error and answers with
  the remedy: start it unregistered and keep the pid in the harness state. The
  same treatment is given to a harness module that is not on the sim node.

Every one of those was a way to get a confident, meaningless result. That is
what this machinery is *for*, so it is fitting that building it produced three
of them.

#### The surface is frozen at twelve

`read write edit list grep bash · elixir define tools invoke simulate replay`

§2.2 is why this is a freeze rather than a list: the surface is the system turn,
and the system turn is the prefix of everything, so a thirteenth tool
re-prefills every session in the application. One of the twelve is `invoke`,
which is how the agent gets an unbounded number of its own without any of this
changing.

The system turn is now **1793 tokens**, up from 845 at M3 and 515 at M1 — about
56 s of prefill at the measured rate, paid once and then served from the shared
system-prefix checkpoint (§4.4).

The six new descriptions carry their contracts the way `edit`'s carries its
match rule. `simulate`'s in particular names the three rules whose violation
fails silently — verbatim from eta's documentation, and from having broken all
three here: `execute/2` must not block, `check/1` must not call into a scheduled
process, and anything `init/2` starts must be **unregistered**. It closes with
the sentence this whole section is about: *a run reporting no violations is not
evidence until you have broken the code deliberately and seen it go red.*

**A live bug the gate could not have caught.** The schema's name is `bash` — the
C agent's, pinned by the goldens — and the guest's op was `shell`. The gate
called `shell` directly, so it passed; the *model* would have called `bash` and
been told no such op existed. It surfaced within a minute of a real model
touching the surface. Dispatch now answers to both, and the lesson is that a
gate exercising the wire does not exercise the surface the model is given.

#### The surface the model sees is not the surface the gate tested

Gate (a) put a real model in front of the twelve tools for the first time, and
it found three bugs in under ten minutes — all of the same shape, and all
invisible to a gate that speaks to the wire directly.

| the model called | the guest implemented | what the model was told |
|---|---|---|
| `bash` | `shell` | *no such op: bash* |
| `elixir` | `eval` | *no such op: elixir* |
| `read` with no args | `read` **with** args | *no such op: read* |

The third is the worst, because the diagnosis is wrong rather than merely
unhelpful — and **the model acts on the diagnosis.** Told that `elixir` did not
exist, it went looking for a JSON library in the project. Told that a tool it
had just been handed did not exist, it reasonably concluded the environment was
different from its description.

The fix is a check rather than three fixes: the gate now iterates
`ToolSurface.names` — the names the *model* is given — and asserts every one
reaches a handler. It found five more the moment it was written.

**And one that is not a naming slip but a format limit.** `invoke` carries a
tool's own arguments inside the envelope, and this model's tool-call format is
XML with no nested objects. So `args` arrives as a JSON *string*, and the first
invoke failed with `no function clause matching in Linecount.run/1` — the
argument having been wrapped as `%{"input" => "{\"path\": ...}"}`, which tells
the model nothing about what went wrong. Warden now decodes a JSON string into a
map, and passes a non-JSON string through under `"input"` for a tool that takes
a bare argument.

This is the concrete form of the risk §11 raised about the generic `invoke`
dispatcher: not that the model calls it *less reliably*, but that nesting
arguments inside a format that has no nesting is lossy, and the loss lands as a
confusing error inside the agent's own code.

#### A `define` turn is not a tool-calling turn

Gate (a), first attempt, and the failure is worth more than a pass would have
been:

```
  → list path=.        greet.py  141
  → read path=greet.py
  [stopped at the 2048-token budget while still reasoning,
   so there is no answer to show]

  1914 prompt · 2173 generated (2123 reasoning) · 2 tool calls
  prefill 54.0s (35 tok/s) · decode 363.9s (5.97 tok/s)
```

**2123 tokens of reasoning before it could write a small Elixir module**, against
a 2048-token budget — so six minutes of decode produced nothing at all. Writing
code is a far more expensive turn than calling a tool, and a budget sized for the
latter truncates the former in the one place the truncation is invisible: inside
the reasoning block, which is not shown. M1's "stopped at the budget while still
reasoning" message is the only reason this was diagnosable at all.

The budget is now 8192. The general point is a design one: this engine reasons at
length, self-modification is the most reasoning-heavy thing the agent does, and at
~6 tok/s a `define` turn costs minutes. That is a real cost of §7.2 and it belongs
beside its benefits.

#### Gate (a): passed

> *Write yourself a tool called linecount that takes a path and returns how many
> lines that file has, then use it on greet.py.*

```
  → list · read · bash (wc -l, to know the answer first)
  → define    syntax error        -> fixed it
  → define    registered, warning -> cleaned it up
  → invoke    :enoent             -> the tool's cwd is not the shell's; used /work
  → invoke    :binary.match/3     -> unsupported here; switched to String.split
  → invoke    verified against three cases of its own devising:
              greet.py 8 (matching wc -l), no-trailing-newline 3, empty 0
  → bash      removed its own temp files

  greet.py has 8 lines.

  2669 prompt · 6037 generated (4151 reasoning) · 15 tool calls
  prefill 106.2s · decode 1335.4s (4.52 tok/s) · context 8691/90112
```

**The generic `invoke` dispatcher did not cost tool-calling accuracy.** That was
§11's stated risk and the reason gate (a) existed; it is not borne out. What
*did* cost accuracy was the argument-nesting bug above — a defect in the wire,
not in the model's grasp of the surface.

Three things in that transcript are worth more than the pass:

- **It iterated on its own code from diagnostics.** A syntax error, a
  version-specific warning, a wrong working directory, and an unsupported stdlib
  call — each corrected from the message it got back. That is the whole argument
  for compile errors and warnings being tool results rather than failures
  (§7.2 step 2).
- **It verified its own tool** against a file with no trailing newline and an
  empty file, without being asked, and stated the counting rule it had chosen.
- **Then it cleaned up after itself.**

**The honest headline is the cost.** Twenty-two minutes of decode for one
self-modification task, at 4.52 tok/s with 4151 tokens of reasoning. §7's premise
is that an agent writing its own tools beats one that re-reads files, and prefill
economics (§2.5) support that — but a `define` cycle is minutes, not seconds, and
a design expecting several per task is a design expecting the user to wait an
hour. This is the strongest argument yet for the MTP draft head (§12, M7), and
for `define` being for tools that will be *reused* rather than for one-off work.

**`erl -noshell` is not enough for a spawned node.** A node started as an Erlang
port has its stdin wired to a pipe, and `-noshell` alone leaves the BEAM reading
it, so the node exits with status 1 and no diagnostic. The workspace came up,
announced itself, and died 330 ms later, three times, before happening to
survive. `-noinput` is the fix, and §9.3 already specified it for the sim node —
it was simply not carried across to this one.
`elixir`, `define`, `tools`, `invoke`, the three-node split, soft-purge refusal,
workspace restart with replay, module-load transcript cards. Then `simulate` and
`replay`: the sim node, automatic shrinking, trace persistence, harness
templates, and the vacuous-harness check.
**Gates: (a) the tool-calling accuracy measurement in §11; (b) the agent, given
`define` + `simulate` + `replay`, catches a planted concurrency bug — eta's own
two-phase-commit and leader-election examples are the fixtures (§9.4).**

Gate (b) is the one that decides whether §9.4 is a feature or a paragraph, and
**M4 ends with the §9.5 review** — keep both, keep it for warden only, move to
Erlang and `main`, or rip it out. All four are acceptable outcomes and the
integration is built so that each costs a build flag rather than a rewrite. Note
the one asymmetry: `simulate` and `replay` stay in the tool surface whatever the
verdict, because withdrawing a tool re-prefills every session (§9.5, point 3).

### M5 — Materialisation
Baseline manifest, git diff over the wire, path validation, the approval sheet,
apply, undo, re-baseline. The product becomes useful here. The adversarial patch
suite lands with it, not after.

### M5g — Leaning into git *(§7.4a)*

Replaces M5's transport with an object exchange. Sequenced so that the only part
carrying real risk — writing into the user's repository — lands first, alone, and
fully testable before anything depends on it.

**G0 — the object store, host side.** `GitObjectStore`: is this path a git
repository, and which object format; write a loose object from `(type, content)`
*verified by hash*; create a ref. Nothing else. Useless without G1 and harmless
until then, which is the point: it can be exercised to exhaustion against scratch
repositories with `git fsck --strict` as the oracle.

Adversarial suite, because this writes into something the user cannot easily
reconstruct: a hash that does not match its content is refused; a sha containing
a path separator or `..` is refused; an object that already exists is left alone
rather than rewritten; a read-only or absent `.git` fails cleanly; a sha256
repository is handled rather than silently written as sha1. Every case ends with
`git fsck --strict`.

**G1 — the exporter, guest side.** `Warden.Git`: detect the repository, resolve
the baseline, commit any uncommitted remainder on `crucible/<slug>`, enumerate
`BASE..TIP` and stream each object as `(sha, type, content)`. `git cat-file`
rather than reading `.git/objects` directly, so an object that has been packed —
by the agent running `gc`, or by git's own auto-gc — is exported the same way as
a loose one.

**G2 — the crossing.** Replaces `propose`/`apply`. The 2 MB / 6 MB caps go: what
crosses is compressed deltas of what changed, not whole file bodies. A budget
still exists, but as a guard against absurdity rather than as a routine limit,
and exceeding it is an error rather than a silent skip.

**G3 — review.** One approval for the import, not a checkbox per file, because
the import is additive and the destructive step is the user's own merge. The
sheet shows the commits and their diff for reading, names the branch, and says
what to run. `ApprovalSheet` loses most of its machinery.

**G4 — the fallback.** §7.4's byte path, kept and demoted, for projects that are
not git repositories. Plus the session-start prompt for a dirty worktree
(include as its own commit, or reset `/work` to HEAD).

**The gate.** A session in a real repository: the agent makes two commits, the
user has concurrently changed one of the same lines on the host, and the import
produces a branch that `git merge` reports as a **line-level conflict** with
markers. That is the whole thesis in one run — git doing the part that §7.4 was
starting to reimplement — and it fails if any link is wrong.

### M5t — Markdown in the transcript *(§5.6)* — **built**

No dependency: Foundation parses CommonMark and GFM, and the work is a block
renderer over `presentationIntent` runs.

`MarkdownBlocks` — parse once per message, group runs by innermost block
identity, emit a view per block: paragraph, heading (by level), fenced code,
list (ordered and unordered, with nesting), block quote, table. Inline bold,
italic, code and links come straight through as an `AttributedString` that
`Text` already renders.

Code blocks get horizontal scrolling **inside** the block, the language label
when the fence carried one, and a copy button.

`Highlighter` — a JSContext holding `vendor/highlight.js/highlight.bundle.js`,
built lazily on first use (53 ms) and held for the process; `highlight(code,
language)` returning `(range, class)` pairs from a span scanner, never from an
HTML parser. Class names resolve through a palette that tracks light and dark.
Every failure path — no hint, unknown language, a JS exception — returns no
ranges, and the block renders as plain monospace.

Called on each **completed line** of a block whose fence named a language,
against the prefix up to the last newline, with the partial tail line rendered
plain; once at close for a block that named none. Guarded above a few thousand
lines, where the quadratic re-highlight stops being free.

`make` runs `vendor/highlight.js/build.sh` and stages the bundle into
`Crucible.app/Contents/Resources/`, the way it already stages the guest image.
Offline: the pieces are vendored, only the concatenation is generated.

Applies to assistant text only. Reasoning, user turns and tool results stay raw
for the reasons in §5.6 — each would be claiming a formatting intent that is not
in the source.

The check is a fixture suite rather than a screenshot: a document exercising every
block kind, asserting the blocks that come out — count, kind, order, and the
language hint on a fenced block. It fails if Foundation's parse changes shape
under us, which is the only thing here that can break silently.

The highlighter gets its own: the bundle loads, `listLanguages()` returns 192,
and a sample per language this project actually uses produces spans — Elixir
sigils, bash heredocs and Rust raw strings among them, because those are what
distinguish a real grammar from a regex. Plus the failure paths: an unknown
language name, and code containing `<`, `&` and `"` coming back with its
characters intact rather than as entities.

And the property the streaming rule depends on, asserted rather than trusted:
for a block with a language hint, highlighting every growing prefix leaves every
**settled** line byte-identical to its rendering in the finished block. The
fixtures are the constructs that span lines — docstring, block comment, heredoc,
multiline string, sigil, raw string — because those are the only ones that could
break it.

### M6 — Scheduling, parking, and the compaction successor
The single-live-session hand-off, park/restore over the explicit-path checkpoint
API (§4.4), the parent-tree change that API needs, VM parking, the disk quota,
and the compaction successor flow (§2.4) end to end — summary generation,
sandbox inheritance, the linked pair in the sidebar. The sandbox inspector.

This is where more than one session becomes real, and where §2.3's decision is
paid for. **Gate: park and restore of an 8 GB checkpoint measured and written
into §4.4; a session driven to full context produces a successor rather than an
error.**

### M7 — The rest
Vision (drag an image or a video into the composer — the runtime already does
both, and it is nearly free). MTP speculation in the Swift loop, evaluated
against its 4 KB/token cost, which §2.3 already prices into the profile — on
the target host it is off, and on a 64 GB machine it is affordable.
Model-generated session titles. Export a session, and its traces, as markdown.

---

## 13. Rules for this codebase

Inherited from the parent, which inherited them from ds4, plus what this app adds:

- **No Python, no C++.** Unchanged.
- **Keep `CQwasar` narrow.** `qwasar.h` and `qwasar_toolcall.h`. Nothing else is
  exported to Swift, ever. If Swift needs to know something about a tensor, the
  answer is a new function in `qwasar.h`, not a wider module map.
- **Never call `qwasar_*` off the engine queue.** One queue, no exceptions, and
  a debug assertion on the queue's specific key in every wrapper.
- **The profile decides; nothing is hardcoded.** Context size and live-session
  count come from §2.3's derivation at launch. No constant in the source, no
  code path that assumes one live session, and no feature that only works on a
  big machine. The 32 GB target is the case to write against; the 128 GB case is
  the one to not break.
- **Live count is never parallelism.** One session runs at a time (§2.1). Any UI
  or API that implies otherwise is wrong.
- **Warden is tested under `eta`, and never depends on it.** A new warden
  behaviour about timing, ordering, or failure arrives with a harness property,
  not a `Process.sleep` in a test — and the default build's lock file must stay
  free of `:eta`. Modules that `use Eta` use its macros for every timer and
  message; the lint enforces it, because a stdlib call is a silent hole in the
  simulation (§9.1).
- **The host runs no model-requested tool.** If a feature seems to need one,
  it does not; it needs a guest tool and a patch.
- **No `--yes` for materialisation.** Sandbox writes are unconfirmed; the
  boundary crossing never is.
- **Correctness before speed**, and never keep a faster path with unexplained
  drift. Same rule, one layer up: the speculation equivalence test in §10 is the
  same promise `tests/test_verify` makes.
- **The C tree is built by `make`.** Xcode shells out to it. Do not teach Xcode
  about `bin2c`.
- **Comment the mechanics, not the syntax.** Session lifetimes, the purge rules,
  the checkpoint prefix contract, and the path validation are the four places a
  reader needs the *why*. Everything else should be obvious.
- **A green simulation reports its coverage.** Seeds, ops, schedules explored.
  `PASS` on its own is a claim this project does not make.
- **Every tool result is capped, and says so.** Prefill runs at ~32 tok/s
  (§2.5), so a result costs about a second per 100 bytes. A tool that can return
  an unbounded amount of text is a tool that can spend ten minutes of the user's
  time on one call. Cap it, and make the truncation message tell the model what
  to do instead — that message is part of the tool's contract, like `edit`'s
  match rule.
- Small, sharp, readable. No slop.

---

## 14. Open questions

- **Bundle the guest image or download it?** 320 MB is right at the line. A
  download needs a hosting story, a signature check, and a first-run wait;
  bundling makes the `.app` ~350 MB and the notarisation upload slow. Leaning
  bundle, for offline-first symmetry with the rest of the project.
- **VM per session, or per project?** Per session is isolated and costs
  `clonefile` (nothing) plus RAM when running (a lot). Per project shares state
  between conversations, which is sometimes what you want and sometimes a
  disaster. Currently: per session, with §6.5's cap. Revisit after M6.
- ~~**Elixir, or Erlang alone?**~~ **Elixir throughout the guest**, on eta's
  `elixir` branch (§7.3, §9.1). The open part is now how well that branch holds
  up — it is best-effort by its author's description — and §9.5 is the scheduled
  review rather than a question left hanging. Fallback: warden to Erlang and
  `main`, topology unchanged.
- **Should agent tools be able to call the model?** A `Crucible.LLM.complete/1`
  available to hot-loaded modules would let the agent build sub-agents. It also
  re-enters the engine from inside a tool call, on a session that is mid-turn,
  on a queue that is already busy — and §2.1 says one at a time. It would need a
  second session and a budget. Deferred, deliberately, and noted here so it is
  not designed in by accident.
- ~~**Context management at 32K.**~~ **Decided: a machine-derived profile, and
  compaction is a successor session** (§2.3, §2.4). What remains open is the
  *policy*: at
  what fraction of the window does the app offer compaction, does it ever compact
  without asking, and does the handoff summary get written by the outgoing
  session (cheap, in-context) or by a second pass over the transcript (better,
  and it costs a whole prefill). Leaning: offer at 85%, never automatic, summary
  from the outgoing session.
- **Does the agent know how much context it has left?** It should — a model
  budgeting its own remaining window behaves differently, and better, than one
  that hits the wall. But the only appendable channel is a tool result, so this
  means every tool result carries a context footer, which costs tokens on every
  call to reduce the chance of wasting all of them. Measure the token cost
  before committing to it.
- **How much of the profile should the user be able to override?** Context and
  live-session count are the two dials, and they trade against each other
  legibly. Whether to expose the 0.85 working-set reserve is a different
  question: it is the one number standing between the machine and swapping 16 GB
  of weights, and a user who lowers it gets a mysteriously unusable app. Leaning:
  expose the two dials, keep the reserve internal, and let the dials refuse
  combinations that would breach it.
- **Should `simulate` gate `define` by default for concurrent code?** Currently
  no: the model opts in with `verify:`. The argument for defaulting on is that
  the failure it prevents — a bad module loaded into the node the session depends
  on — is exactly the failure the three-node split exists to contain, and
  containment is not correctness. The argument against is that a mandatory
  harness for a five-line `GenServer` is the kind of friction agents route
  around. Revisit after M4's gate (b).
- **KV dtype.** The parent defers fp8 until there is a quality benchmark. Same
  answer here, but the app is where long sessions will actually make it hurt.
