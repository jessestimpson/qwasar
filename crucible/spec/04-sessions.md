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
  **~2200 tokens — 99.6% of a first turn**, because the ten tool schemas are
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

#### The UX, settled: there is no save verb

*Designed 2026-08-23, ahead of M6 building it.*

A checkpoint buys **speed, not safety**. The transcript and the token history
always persist; any session can be rebuilt by re-prefill. A "Save checkpoint"
button would assert the opposite contract — that forgetting it loses work —
manufacturing anxiety about precisely the thing the architecture guarantees.
And no user can be asked to weigh "is this 150 MB–8 GB write worth it now";
that is what the budget above is for. So checkpoints are autosave-shaped:
automatic at the moments that already contain a pause, visible only as the
answer to the one question the user actually has — *if I come back, how long
until this session is warm?*

**Automatic, at natural boundaries** (each throttled by the existing
`lastCheckpoint` guard, so an unchanged session never rewrites gigabytes):

- **On session switch** — the moment that today costs a full re-prefill, and
  most of M6's value. The write may overlap the incoming session's own
  restore/prefill; measure before assuming the overlap hides it (house rule).
- **On quit** — already built (`shutdown()`).
- **After the system-prefix prime** — already built.
- **After a few minutes idle.** The person who walks away mid-project and
  returns tomorrow is who checkpoints exist for, and idle is when a
  multi-second write is invisible.

This refines "on close, once" above from a single moment to a set of
boundary moments; the one-live-checkpoint-per-session accounting and the
staircase prohibition stand unchanged.

**One user verb, and it is not "save" — it is Park.** A context-menu action
on the session row meaning "I am done here for now; keep it warm and free
the live slot." That decision — which session deserves the slot — is genuinely
the user's; the checkpoint it implies is mechanism and goes unmentioned. No
dialog, no filename: checkpoints are not documents.

**The honest indicator is where the UX actually lives.** The sidebar's
live/parked circle gains a distinction with a number, in the
measured-figures tradition:

| state | shows |
|---|---|
| ● live | holds its share of the working set |
| ◐ parked, warm | "resumes in ~2 s" — checkpoint verified ON DISK now |
| ○ parked, cold | "rebuilds in ~4 min" — token count ÷ the measured ~32 tok/s |

"Verified on disk" is load-bearing: the LRU evicts, and an indicator that
reflected history rather than the store would turn eviction into a mystery
slowdown. Showing the estimate is what converts the checkpoint from invisible
plumbing into visible value — it *is* the user-comprehensible meaning of one.

**Deliberately absent:** per-turn autosave (the staircase above, measured and
rejected); confirmation dialogs; any cache-management UI in the session view.
If "clear checkpoints" is ever needed it is a config-project operation, not a
button next to someone's work.
