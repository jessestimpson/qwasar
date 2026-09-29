# The renovation: one server, one app, one API

*Written 2026-09-29. Nothing here is started. The wire contract it builds is
`API.md`; this file is the plan -- what changes in each program, in what
order, with what gate, and which decisions are not the plan's to make.*

Three moves, in dependency order:

1. **`qwasar-server` gains the Session API** (`API.md`): sessions the server
   owns, deltas from the client, prefill progress over SSE, warmth reported
   and guaranteed. The OpenAI and Anthropic endpoints stay.
2. **`qwasar-agent` becomes a client of it.** The smallest program that
   exercises the whole API, so it goes first among the clients and proves the
   contract before the GUI depends on it.
3. **Crucible and the menu bar become one app, `Qwasar`.** The menu bar that
   runs the server, and -- when asked -- the coding agent's window, which
   talks to the server over the API instead of linking the engine.

The engine (`qwasar.h`) is not renovated. It gains one narrow thing (§2.6)
and otherwise is used as it is.

## 1. What this buys, stated up front

- **The cache stops being a guess.** Today the server infers a conversation
  from a resent prompt and rewinds when the reply came back different. Under
  the API the client never sends history, so the only miss left is one the
  server chose (eviction), and it says so before paying it.
- **One engine process, however many fronts.** Crucible loads its own 16--80
  GB of weights and the menu bar's server loads another; they cannot run at
  once, and the constraint is a memory-check refusal rather than a design.
  After this, the weights are loaded once, and the CLI, the GUI and a
  third-party client on the compat endpoints share them under one scheduler.
- **The sandbox loses its heaviest tenant.** Crucible's App Sandbox exists for
  the VM and the project-folder grants. The engine never needed to be inside
  it, and the model-folder grant (`ModelAccess`, a security-scoped bookmark
  to mmap weights) was a workaround for having put it there.
- **The memory profile is derived once, where the sessions are.** Crucible's
  `MemoryProfile` (context per model per machine, live-session budget) moves
  into the server, which is the only place that knows how many sessions exist.
- **Crucible's session code is deleted, not ported.** `EngineHost`, `Session`
  (817 lines), `Bridge`, `Tokenizer`, `ChatTemplate`, `MemoryProfile`,
  `ModelFamily`, `ModelAccess` and the golden/prefix suites all go, replaced by
  a client (~400 lines of Swift) whose events are the API's events. The
  `SessionEvent` enum Crucible has today was designed around what a person at
  the window needs to see; `API.md` §5 is that enum, so the port is a rename.

## 2. The server

### 2.1 Files

`qwasar_server.c` is 2,500 lines: an HTTP core, the OpenAI and Anthropic
request shapes, and the completion handler around `srv_prefill`/`srv_generate`.
The API is a second front on the same engine, so the core is shared and the
fronts are separate files:

```
qwasar_http.c/.h      the HTTP/1.1 reader, response writer, SSE framing, the
                      growable `str`, JSON string escaping, bearer auth
                      (lifted from qwasar_server.c, unchanged in behaviour)
qwasar_sessions.c/.h  the session store and the scheduler (§2.2, §2.3): the
                      one place that owns qwasar_session handles, the live
                      set, the queue, the token logs, the checkpoints
qwasar_api.c          the Session API endpoints and the event stream (API.md)
qwasar_server.c       main, the compat endpoints, and the request loop that
                      routes to either front
```

`srv_generate` -- the decode loop with its stop-sequence and UTF-8 holdback,
tool-call detection and deltas -- moves to `qwasar_sessions.c` as the one
generation routine both fronts call, with the delta callback grown to the
event vocabulary. The compat handler becomes a client of the store too: an
anonymous session that it matches by prefix the way it does today (the
`common_prefix`/`rewind`/`restore` ladder in `srv_prefill`), so those clients
lose nothing and share the queue.

### 2.2 The store

```
<state_dir>/
  server.json                    schema version, model id last loaded
  sessions/<id>/
    record.json                  metadata, prefix hash, created, last step,
                                 pending calls, state
    tokens.bin                   int32 LE, the timeline; rewritten after
                                 every step (a few hundred KB at most)
    checkpoint.bin               the parked state (§2.6), when parked
```

The current and last step's events are kept in memory, for reattach; a
restart ends any step, so there is nothing to reattach to across one.

`state_dir` defaults to `~/Library/Application Support/Qwasar` (the merged
app's name; a `--state-dir` flag and `QWASAR_STATE` override it, and the
tests point it at a temp directory). The shared-prefix LRU cache stays where
it is (`~/.cache/qwasar/kv`, `qwasar_kvstore.c`), doing the job it was built
for: a prefix many sessions rediscover.

On start the server reads every `record.json` and offers those sessions
`cold` or `warm`; it evaluates nothing until a step arrives. A session whose
record names a different model id than the one loaded is listed with
`"model_mismatch": true` and refuses steps with `409` -- its tokens are the
same vocabulary (both models share the tokenizer) and it *could* be replayed
under the other model, but that is the client's decision to make explicitly
by opening a new session, not something the server does on its own.

### 2.3 The scheduler and the profile

One engine thread (the existing `sv->lock` becomes a worker with a queue).
Steps, compat requests and parks are jobs; the thread runs one at a time and
the connection threads block on their job's events. `queued` events go out as
the position changes.

The profile is Crucible's `MemoryProfile.derive`, ported to C once and run at
load: `qw_gpu_working_set_limit()` for the working set (the engine has it),
`hw.memsize` for physical, the model's family for KV bytes per token and
fixed session bytes (`crucible/Sources/CrucibleKit/ModelFamily.swift` has both
sets of numbers with their derivations), 85% reserve, context stepped to 8192
and capped at the model's trained window, live sessions preferred only where
they still leave the full window. The result is what `/v1/server` reports.
`--ctx` remains as an override and the menu bar stops passing it.

The live set: `live_sessions` handles at most. A step on a non-live session
admits it, parking the least recently used if the set is full. Parking writes
the session's checkpoint (§2.6) and frees the handle. Compat's anonymous
session occupies one slot like any other and is parked like any other.

### 2.4 Steps, exactly

`turn` and `continue` are the two continuation renderers the engine already
has, driven the way `qwasar_agent.c` and Crucible's `Session.swift` both
drive them today:

- first step of a session: `qwasar_apply_chat_template([system, user])`,
  split at the prefix (`ChatTemplate.systemPrefix`'s trick: render the system
  turn alone with no generation prompt; assert it is a token prefix of the
  whole; evaluate it; save it to the shared store; evaluate the rest);
- a later `turn`: `qwasar_render_user_turn` -- which closes any open
  assistant turn, so a `turn` on an `awaiting_tools` session is legal and
  documented, if unkind to the model;
- `continue`: results joined in call order with `"\n"` into one
  `qwasar_render_tool_result`, as both clients do now.

Images go through `qwasar_image_encode_memory` and `qwasar_session_eval_images`
as the compat path does. The rewind point (`qwasar_session_mark`) is not
taken for API sessions: nothing ever resends a prompt, so there is nothing to
rewind to.

### 2.5 Tests

`tests/test_session_api.py` beside the two compat suites, on the same
`qwasar_api.py` harness: open, a turn with a forced-by-prompt tool call,
`continue`, `describe` warmth before and after `park`, reattach with
`Last-Event-ID` after closing the socket mid-stream, `cancel`, `409` on a
second `continue`, a second server process started on the same state dir
finding the session `warm`, the compat endpoints still passing while an API
session is live. The prefill-progress events are asserted to span the whole
resume, not a chunk, on a session made cold by deleting its checkpoint.

These need a model. The toy fixtures (`tests/fixtures`, both Flash-Next toy
formats) load in the engine and generate, so the suite can run on a toy when
`QWASAR_TEST_MODEL` names one -- which makes it runnable while the real model
is busy elsewhere, and in CI. Whether a toy is allowed to touch Metal on this
machine while the other agent measures is a question for the owner (§7).

### 2.6 What the engine is asked for

One addition, the one Crucible's spec asked for a month ago and did not get
(`crucible/spec/04-sessions.md` §4.4):

```c
/* Explicit-path checkpoints, for a caller that owns the file's lifetime:
 * the session store's park.  The same serialisation as the hash-keyed
 * cache -- header, dims check, stored tokens -- with the caller naming the
 * file and no eviction. */
bool    qwasar_session_save_file(qwasar_session *s, const qwasar_engine *e,
                                 const char *path, char *err, size_t errcap);
int32_t qwasar_session_restore_file(qwasar_session *s, const qwasar_engine *e,
                                    const char *path, const int32_t *tokens, int32_t n);
```

Without it, parking goes through the 6 GB LRU store and a 262K session's
checkpoint (~8 GB for Flash-Next, ~17 GB for the 27B) evicts everything
including itself. Milestone 1 ships on the LRU store anyway -- it is correct,
merely budget-bound -- and milestone 4 switches parking to explicit files once
the engine has them. This is the other agent's file (`qwasar_kvstore.c`) and
is requested, not taken.

## 3. `qwasar-agent` as a client

The agent keeps its tools, its confirmations, its TUI and its REPL, and loses
the engine. `qwasar_agent.c` links `qwasar_http.c` (the same reader/writer,
now used as a client) and `qwasar_json.c`, and the loop becomes: `turn` →
read events → run the calls → `continue` → … The TUI's prefill bar is driven
by `prefill` events, the status line by `decode`, the reasoning toggle by
`reasoning`.

- `-m <model>` goes; `--server <url>` arrives, default `http://127.0.0.1:8080`.
- **No server, no problem.** If nothing answers on the port and a model can be
  resolved (`-m`, `$QWASAR_MODEL`, `./qwasar-model`), the agent launches
  `qwasar-server` from beside its own binary with `--exit-on-eof` on a pipe it
  holds -- the menu bar's lifeline, reused -- waits for `/health`, and
  proceeds. `qwasar-agent "fix the tests"` keeps working on a machine that
  never ran the app.
- REPL: `/new` opens a session; `/effort` opens one at that effort; `/ctx`
  is `describe`; `/save` is `park`; `/image` and `/video` attach to the next
  `turn`; `/yes`, `/think`, `/help`, `/quit` are unchanged. `-n` is
  `max_tokens` per step; `--steps` stays the client's cap.
- Ctrl-C calls `cancel` and waits for `done`; a second Ctrl-C leaves.
- The agent's own `AGENT.md` guidance and system prompt render on the server
  now: the prefix is what `open` says it is.

`qwasar` (the plain CLI in `qwasar_cli.c`) is not touched: one-shot generation
with the engine in-process is a different program with a different reason to
exist, and it is the tool the engine's own tests and measurements use.

## 4. `Qwasar.app`

### 4.1 One bundle, two faces

```
app/                                   was crucible/ (git mv; history kept)
  Sources/Qwasar/                      the app: menu bar + window
    Main.swift                         NSApplication, .accessory policy
    StatusItem.swift                   the menu bar (from menubar/App.swift,
                                       Icon.swift): server state, port, model,
                                       Open Coding Agent, Start at Login
    ServerController.swift             runs qwasar-server (from menubar/)
    CrucibleApp.swift, SessionView…    the window, as they are
    AppState.swift                     the harness, on the client
  Sources/QwasarKit/                   was CrucibleKit: the sandbox, the
                                       tools, the store, delegation, markdown,
                                       config -- and QwasarClient.swift
  Guest/  Tests/  spec/  Resources/    as they are
```

`LSUIElement` is true: the app is a menu bar item with no Dock icon. **Open
Coding Agent** (⌘N from the status menu, or `open -a Qwasar --args --agent`)
sets the activation policy to `.regular`, shows the window and the Dock icon;
closing the last window returns to `.accessory`. Quitting from the window
quits the app, server and all, after the guests flush and the server writes
its checkpoints -- the two shutdown sequences that exist today, run in order.

The server is the helper it is today (`qwasar-server` in `Contents/MacOS`,
lifeline pipe, `--exit-on-eof`, log in `~/Library/Logs`), started when the
app starts. The window needs it running: if the model is not loaded the
composer says so and offers Start Server, rather than loading a model of its
own.

### 4.2 The sandbox question

Crucible is App-Sandboxed for the VM and the project-folder bookmarks; the
menu bar is not sandboxed and the server reads model folders freely. Merged,
the server is a child of a sandboxed process, and a child inherits the
sandbox. Two ways out, in order of preference:

1. **The helper carries `com.apple.security.inherit`.** A child process with
   that entitlement runs inside the parent's sandbox *including* the
   extensions the parent has been granted -- so the app grants the model
   folder once (the bookmark it already keeps) and the server, spawned while
   the grant is active, can mmap it. This is the documented pattern for
   command-line helpers in sandboxed apps. It is also the M0 gate of this
   work, exactly as "App Sandbox plus Virtualization plus a 16 GB mmap" was
   Crucible's (`spec/08-security.md` §8.1): prove it first, headless, in the
   signed bundle.
2. **Hardened Runtime without App Sandbox**, the fallback that spec already
   names as strictly worse and still notarisable. The VM, the tools and the
   `.git` shadow -- the security that matters (§8.2 of the spec) -- do not
   depend on App Sandbox; the project-folder grants do, and become ordinary
   paths.

The decision is made at the gate, not before.

### 4.3 What changes in the Swift

| today | after |
|---|---|
| `EngineHost` (queue, load, sessions), `Session` (the loop), `Bridge`, `Tokenizer`, `ChatTemplate`, `MemoryProfile`, `ModelFamily`, `ModelAccess`, `Diagnostics`, `GateCheck`'s mmap half, `CQwasar` module map | **deleted**; the engine is not linked |
| `SessionEvent` | the API's events, decoded from SSE by `QwasarClient` |
| `AppState.send()`: open-or-reuse the live session, replay history, run the turn | `open` once per `SessionRecord` (the id stored on the record), `turn`, run tools in the guest on each `tool_call`, `continue`; the server owns liveness |
| `SessionRecord.tokens` / `tokens.bin` in the app's store | gone: the server keeps the timeline. The transcript stays -- it is for people |
| warm/cold indicators from `qwasar_kv_probe` | from `describe`/`list` warmth, same numbers, same honesty |
| `park` → `closeAndCheckpoint` | `POST park` |
| effort locked once evaluated | unchanged in meaning: a session's effort is its prefix; a new effort is a new session |
| the config project's host tools, delegation, `fetch`, skills, the guest | unchanged: they are the client's tools, run on `tool_call`, answered by `continue` |
| `Tests/GoldenTests`, `PrefixSuite` | deleted: the template is rendered on the server and tested there (`tests/test_tokenizer`, the API suite) |
| `make gate-full` etc. | start a server on a temp state dir and drive it; `--gate` needs no model folder grant |

The model menu (the menu bar's `ModelCatalog` and Crucible's `ModelFamily`
were converging on the same code) becomes one thing that scans folders and
restarts the server on a choice; `/v1/server` reports what loaded.

### 4.4 Delegation and the config project

Untouched in design. A delegation is a remote agent with the same tool chain
(`spec/15-delegation.md`); the local model's side of it is a `tool_call`
like any other. The config project's sessions are API sessions whose tools
are the host's config tools -- `open` with those schemas, run on the client.

## 5. Milestones and gates

**M0 -- design.** This document and `API.md` reviewed; the decisions in §7
made. *Gate: the owner's word.*

**M1 -- the server speaks the API.** `qwasar_http.c`, `qwasar_sessions.c`,
`qwasar_api.c`; the profile in C; the store with token logs and reattach;
the queue; parking through the LRU store; compat endpoints moved onto the
store. *Gate: `tests/test_session_api.py` passes on a toy and on the real
model; `make test-api` (the compat suites) still passes; a Goose session on
the compat endpoint runs at the speed it did.*

**M2 -- `qwasar-agent` on the API.** The C client, server autostart, the
REPL mapped. *Gate: `qwasar-agent -i` through a multi-tool task against a
server it started itself, with the prefill bar and status line driven by
events; the same task through a server the menu bar started.*

**M3 -- `Qwasar.app`.** `git mv crucible app`, the menu bar merged in,
`QwasarClient`, the deletions of §4.3, the sandbox gate of §4.2. *Gate:
`make -C app gate-full` runs a sandboxed turn through the server helper; the
app's `make test` passes with the engine-dependent suites gone; sessions
made before the merge are handled as §7 question 4 decides.*

**M4 -- parking as designed.** `qwasar_session_save_file`/`restore_file`
from the engine; per-session `checkpoint.bin`; a disk budget with the
prompt-never-delete rule from `spec/04-sessions.md`. *Gate: park and resume
of a 200K-token Flash-Next session measured and written into this file.*

### 2.7 M1, built -- 2026-09-29

`qwasar_http.c` (lifted), `qwasar_sessions.c` (the store: records, token
logs, the FIFO engine queue with `queued` positions, LRU parking through the
hash-keyed store, resume with progress over the whole span, the generation
loop moved from the server, the compat session on the same ladder it always
had), `qwasar_profile.c` (the arithmetic, from the shard headers and Metal),
`qwasar_api.c` (the endpoints and events).  `--state-dir`, `--live`,
`--token` added; `--ctx` now overrides a derived default.

`tests/test_session_api.py` runs its own servers on a temp state directory
and a temp HOME (so the toy's checkpoints stay out of `~/.cache/qwasar/kv`).
On `tests/fixtures/flashnext-tiny-q4`, with the tokenizer
`tools/toy_tokenizer.py` writes for it: **17 tests, 1.4 s** -- open,
describe, list, refusals as statuses, a cold first turn whose `prefill`
events span the whole resume, a live second turn, the shared prefix read
by the next session, park → warm → resume from the checkpoint, cancel,
reattach with `Last-Event-ID` (contiguous ids), delete, `context_full`
decided before evaluation, the compat endpoint sharing the engine, and a
restart on the same state directory finding the session and resuming it.
Two skip on the toy: the tool-call path (`tool_call` events, `continue`),
which needs a model that writes a call, and the window filling under
generation.  The OpenAI and Anthropic suites on the toy fail identically
before and after the change (74 pass, 34 are the toy's inability to follow
an instruction), so the moved compat path behaves as it did.

**Not yet run against the real model** -- the rest of the M1 gate waits for
the engine to be free: `make test-api` with `QWASAR_TEST_MODEL` at a real
folder (which runs the tool-call test), and a Goose session on the compat
endpoint at the speed it had.

### 3.1 M2, built -- 2026-09-29

`qwasar_http.c` gained its client half -- connect, request, response, an SSE
reader that undoes chunked framing incrementally so the agent polls its own
keyboard between events -- and `qwasar_agent.c` was rewritten around it.
The tools, confirmations, TUI presentation and REPL are as they were; the
engine, tokenizer, session, generation and MTP code are gone, and the agent
links `qwasar_toolcall.o`, `qwasar_json.o`, `qwasar_http.o` and the TUI
only: 172 KB, no Metal.  `--server`, `--token`, `--resume <id|last>`,
`--temperature` arrived; `-c`, `--mtp`, `--mtp-depth`, `--no-cache` went
with the engine.  Ctrl-C is a `cancel` on a second connection; the stream is
read to its `done`.  Sessions carry `metadata {client, cwd, title}` so
`/sessions` and `--resume last` find this directory's own.  A one-shot run
parks its session on the way out.

Autostart as planned: no listener on a loopback port plus a resolvable model
(`-m`, `$QWASAR_MODEL`, `./qwasar-model`, `qwasar-model` beside the binary)
starts `qwasar-server` from beside the binary with `--exit-on-eof` on a pipe
the agent holds; the log goes to `$TMPDIR/qwasar-server-<port>.log`.

Tests (in `tests/test_session_api.py`, on the toy): a one-shot task against
the suite's server leaves one parked session for its directory, `--resume
last` grows it; no server and no model is refused with the reason; a server
the agent starts serves the task and is gone once the agent is.  The toy's
tokenizer now trains 142 merges on the template's and the agent's own text
(`tools/toy_tokenizer.py`), which is what fits the agent's 3.3K-token prefix
in the toy's 4K window.  **The M2 gate's interactive run on the real model
waits for the engine to be free**, as M1's does.

### 4.5 M3, built -- 2026-09-29

`git mv crucible app`; the menu bar app's four files moved in; the modules
are `Qwasar` and `QwasarKit`.  Deleted, not ported: `EngineHost`, `Session`,
`Bridge`, `Tokenizer`, `ChatTemplate`, `MemoryProfile`, `ModelFamily`, the
`CQwasar` module map, the C tool-call parser and UTF-8 assembler (the server
guarantees whole characters), and the golden, prefix, tool-parser and UTF-8
suites.  Added: `QwasarClient` (the API, URLSession, ~400 lines, with a pure
`SSEParser` and event decoder that `ClientSuite` pins), `Events.swift` (the
`SessionEvent` the window already consumed, `TurnStats`, `ReasoningEffort`,
unchanged so old transcripts decode), `ServerController` (from the menu bar,
now taking its model and state directory from the app), `StatusItem` (the
menu bar item, plus **Open Coding Agent**), `QwasarApp` (an AppKit delegate:
status item, main menu, the window on demand -- `.accessory` until a window
opens, `.regular` while one is), `Gate` (the gates against the helper).
`AppState.send()` opens the record's server session once -- the executor's
environment description and the project's prompt as its prefix, which the
old app never composed -- then `turn`, runs each `tool_call` off the main
actor, `continue`s.  The server keeps its sessions in the app's own store
under `server/`, beside the app's records.

**The sandbox question (§4.2) is answered: App Sandbox stays.** The helper
carries `com.apple.security.inherit`; the app gained `network.server`.
`make gate-full` on Flash-Next, in the signed sandboxed bundle: the helper
listened after 23 s and reported the right profile (79.52 GB resident, 262K,
1 live); the guest booted in 0.49 s; a session opened with a 2,193-token
prefix; the model called `list` and `bash` in the guest and answered;
`describe` said idle, live.  On the toy the same path runs in 1 s, and its
ten-tool prefix (5,240 toy tokens) does not fit the toy's window -- the
real model is the gate.

Two things are for an interactive run: the model-folder grant through the
panel (the gate uses a static path exception; the shipping path is a
bookmark the helper inherits, the same mechanism), and the project folder
mounted into the guest (`/work` was empty in the gate, whose read-only
exception cannot be shared read-write; a panel grant is).  One bug the gate
found and fixed on the way: `URLSession.AsyncBytes.lines` does not deliver
empty lines, and an empty line is what ends an SSE event, so the first
Swift stream heard nothing; lines are split from the bytes directly now.
And one oddity recorded, not explained: under the sandbox the toy's shard
headers summed to zero resident bytes where the real model's summed right;
a zero now falls back to the known figure with a log line.

### M4, built and gated -- 2026-09-29

The engine gained `qwasar_session_save_file`, `qwasar_session_restore_file`
and `qwasar_kv_probe_file` (§2.6); the store parks named sessions to
`sessions/<id>/checkpoint.bin` (by request, eviction, shutdown, and on growth
by a quarter of the conversation, at least 4K tokens) and resumes from
whichever of that file and the shared prefix cache covers more; sessions
report `checkpoint_bytes`, the server a `disk` block, and
`DELETE /v1/sessions/{id}/checkpoint` drops one.  The app shows sizes, keeps a
budget (a quarter of free space at first launch) and lets the user choose
what to drop; nothing is dropped automatically.

**The gate, on Flash-Next, M5 Max 128 GB** (262,144-token window, one live
session; 720 KB of the repository's own source as one user turn; a private
HOME and state directory):

| | |
|---|---|
| first turn, cold | 192,441 tokens prefilled in 640 s (300 tok/s average, 430 at the start, 303 at the end) |
| its checkpoint | **6.33 GB** (127 MB fixed + ~32 KB/token, as §2.3 predicts) |
| park | 0.10 s -- the step had already written the file as its growth checkpoint, and a park over a current file writes nothing |
| resume, same server | 192,442 restored, 19 to prefill; resume event at 5.5 s, first token at **11.1 s** |
| shutdown | 3.0 s, rewriting 6.33 GB for the 19 new tokens (≥2.1 GB/s) |
| restart with the shared cache deleted | warm, 192,462 of 192,462 covered, from the session's own file alone |
| resume, new server | resume event at 8.3 s, first token at **16.4 s** |
| drop | freed 6.33 GB; the live session stays live and is rewritten at its next park |

Against 640 s to rebuild cold: **58x** in the same process, **39x** after a
restart.  The file's page cache was likely still warm across the restart
(written 30 s before); a cold read from disk is not measured -- `purge`
needs root.

**Two findings.**  (1) The first evaluation after a restore costs about as
much as the restore itself: 5.5 s for 19 tokens in the same process, 8.0 s
after the restart.  A normal 19-token step at this context is well under a
second, so this is the unpacked state's first use on the GPU -- 6 GB of
shared buffers faulted in -- not the attention.  Since measured
(`QWASAR_TEST_RESTORE_TIMING` in tests/test_kvstore, 131K tokens, 4.35 GB):
into a new session the restore took 1.1--7.4 s and its first step 1.5--8.2 s,
varying run to run; into a reset session (`qwasar_session_reset`, which keeps
a session's memory) 1.9 s and 0.22 s, steady, against 0.13 s for a normal
step.  The server now keeps the last parked handle and resets it for the
next resume.  Still open: the first resume after a restart, which has no
handle to reuse; and the restore's read through a temporary buffer.
(2) The server's resume estimate said **3.2 s** where the truth was 11--16 s:
it models the read and the uncovered prefill, not that first-use cost.  It
should learn it from measured resumes, as it already learns the read rate.

## 6. Working beside the engine

Another agent owns the engine and the server today. This plan touches the
server in M1, so the split is by file, agreed before M1 starts:

- **This work creates** `qwasar_http.c/.h`, `qwasar_sessions.c/.h`,
  `qwasar_api.c`, `tests/test_session_api.py`, and edits `qwasar_server.c`
  only to lift code out of it and route to the new front. Those lifts are
  mechanical and land as their own commit so the engine work rebases over one
  move rather than many.
- **The engine work owns** everything else in the C tree, and is asked for
  §2.6 when convenient.
- The Swift trees (`crucible/` → `app/`, `menubar/`) are this work's alone.

Nothing in M1 runs the model until the owner says the engine is free; the
toy question (§7) decides whether the API suite can run before then.

## 7. Decisions for the owner

1. **The API's names.** `turn`/`continue`/`park` and the event names in
   `API.md` §5 -- read them as the person who will type them into the CLI's
   help and the app's tooltips. Renames are free now and expensive after M2.
2. **Toys on Metal.** May the API suite run against a toy fixture while the
   real model is in use by the other agent? It is a few MB of weights and
   seconds of GPU, but it is the GPU.
3. **App Sandbox or not** (§4.2) -- decided at the M3 gate: kept.  The
   inheriting helper loads the real model inside the sandbox (§4.5).
4. **Old Crucible sessions.** The app's existing sessions have tokens and
   transcripts but no server session. Options: (a) re-send each old
   session's history as its first `turn` when it is next used -- a one-time
   re-prefill, and the model sees its own past as a quoted block, which is not
   the same conversation; (b) an import endpoint that accepts a token
   sequence once (`POST /v1/sessions` with `"tokens": [...]`, refused unless
   the tokens begin with the session's own rendered prefix) -- exact, but an
   endpoint that takes history, which §5.2 of the API says there is none of;
   (c) leave them readable and closed. The plan assumes (c) unless told
   otherwise; (b) is a day's work and a permanent exception.
5. **Where `qwasar-agent` autostarts from** (§3): beside its own binary, or
   `$QWASAR_SERVER`, or refuse and print the command. The plan assumes
   beside-its-own-binary with a printed line saying so.
6. **The default effort for a new project** in the app: medium was chosen for
   6 tok/s; at Flash-Next's ~68 tok/s high may be the right default. Not part
   of this renovation, but the moment the server reports which model runs is
   the moment to make the default depend on it.
