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
