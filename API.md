# The Qwasar Session API

*Version 1. Designed 2026-09-29. This is the wire contract; the plan that
builds it, and the programs on either side of it, are in `PLAN-qwasar.md`.*

An HTTP API for a local inference server that **owns the conversation**. A
client opens a session, states its prefix once, and from then on sends only
what is new: a user message, or the results of the tool calls it was asked to
run. The server holds the model's state for that session, reports how warm it
is, streams progress through the one silent stretch of a turn (prefill), and
never has to guess whether a request continues a conversation it has seen.

It is written for `qwasar-server` and the two models it runs, and stated in
terms that hold for any autoregressive model with a cache -- so that an
implementation for another model reads this document and disagrees with
nothing in it. Where a statement is specific to qwasar it says so.

## 1. Why not the OpenAI shape

The OpenAI chat API is stateless: the client owns the history and resends all
of it with every request, and the server matches the new prompt against
whatever it still holds. That shape is easy to adopt and it works against this
server today (`/v1/chat/completions` and `/v1/messages` stay, for the clients
that speak them). But it works *against* the cache, in three ways this
project has measured rather than supposed:

- **The prefix is the client's to rewrite, and clients rewrite it.** A system
  prompt edited between requests -- a timestamp, a tool added, a "current
  file" line -- makes every token after it a miss. Goose and opencode both do
  this, and the server's log now names the message it happened in.
- **The reply comes back different from how it left.** Reasoning a client
  drops, a stop sequence, trimmed whitespace, a retry: the server's live
  session has moved past the prompt, and it takes a rewind point one token
  short of every prompt's end (`qwasar_session_mark`) to cope. Machinery that
  exists only to tolerate the client's forgetfulness.
- **Nothing is said.** A request cannot ask whether its conversation is warm,
  is not told how much was reused, and sees no progress while a 40K-token
  prompt is prefilled. To a person at a keyboard that is a hang.

The closest prior art is OpenAI's Responses API (`previous_response_id`),
which chains requests server-side. It still lets every request restate its
instructions, and its cache is a hint. This API makes the prefix a property
of the session, the timeline append-only, and the cache state a reported
fact.

## 2. The model

Three words carry the design.

**Prefix.** What the model sees before the first user message: the system
prompt, the tool definitions, and the settings that render into that turn
(for qwasar: thinking on or off, and the reasoning effort). It is stated when
the session is opened and **cannot be changed**. A different prefix is a
different session. This is the whole answer to §1's first point: there is no
request that can rewrite the prefix, so no client can miss on it by accident.

**Timeline.** The sequence the model has evaluated: the prefix, then user
turns, assistant turns and tool results, in order. It only grows. The client
never sends any part of it back; it sends the next piece. The server keeps
the token sequence -- it is the truth of the session, and it is small (4
bytes a token) next to the state it describes.

**Warmth.** Where the session's state is right now, which decides what the
next step costs:

| warmth | meaning | next step costs |
|---|---|---|
| `live` | in the engine's memory | only the new tokens |
| `warm` | a checkpoint on disk covers it, verified now | a read, then the new tokens |
| `cold` | only the token log | a re-prefill of what no checkpoint covers |

Warmth is reported, not promised: `warm` means the server just checked the
file. The guarantee the API makes is narrower and firmer than "always cached":
**a step never evaluates a token the session already holds, and whatever it
does evaluate beyond the new material is announced before it starts.** A miss
can come from the server's own memory management, never from how the client
formatted a request, because the client formats nothing.

**A step** is one call to the model: a `turn` (a user message) or a
`continue` (tool results). Each streams events and ends with a `done` that
says why. The agent loop -- generate, run tools, feed results back -- is the
client's, one step at a time, because the tools are the client's: a VM, a
working directory, a person confirming a write.

## 3. Transport

- HTTP/1.1, JSON request and response bodies, UTF-8.
- Streaming steps answer with `text/event-stream` (SSE). Every event has an
  `id`, an `event` name and one JSON object as `data`. Text deltas are
  complete UTF-8: a token that ends mid-character is held until the next.
- Bound to `127.0.0.1` unless told otherwise. When bound wider, or when
  `--token` is given, every request carries `Authorization: Bearer <token>`.
- Errors are `{"error": {"code": "...", "message": "..."}}` with the HTTP
  status that fits: `400` a malformed request, `404` no such session, `409` a
  session in a state that refuses the call (`full`; `continue` when no calls
  are pending; `park` while a step runs), `503` the model is not loaded yet,
  `500` the engine failed. Inside a stream, a failure is an
  `error` event followed by the end of the stream.
- Nothing is versioned in headers. `/v1/` is the version; changes are
  additive (new fields, new events, new capability flags), and a change that
  cannot be additive is `/v2/`. Clients ignore events and fields they do not
  know.

## 4. Endpoints

```
GET    /v1/server                      the model, the machine, capabilities
POST   /v1/sessions                    open
GET    /v1/sessions                    list
GET    /v1/sessions/{id}               describe
POST   /v1/sessions/{id}/turn          a user message            -> stream
POST   /v1/sessions/{id}/continue      tool results              -> stream
GET    /v1/sessions/{id}/events        reattach to the step in flight -> stream
POST   /v1/sessions/{id}/cancel        end the step at the next token
POST   /v1/sessions/{id}/park          free the live slot, keep it warm
DELETE /v1/sessions/{id}/checkpoint    give its disk back; it becomes cold, not gone
DELETE /v1/sessions/{id}               forget it, state and log
POST   /v1/sessions/{id}/aside         a step off the record    -> stream
```

The existing `GET /health`, `GET /v1/models`, `POST /v1/chat/completions`,
`POST /v1/messages` and `POST /v1/messages/count_tokens` remain as they are.

### 4.1 `GET /v1/server`

What a client needs before opening anything.

```json
{
  "model": {"id": "qwen3.8-flash-next", "name": "Qwen3.8 Flash-Next",
            "family": "qwen4_exp", "path": "/…/Qwen3.8-Flash-Next-MLX-4bit"},
  "context": 262144,
  "live_sessions": 1,
  "profile": {"physical_bytes": 147578404864, "working_set_bytes": 115448000000,
              "weights_bytes": 79520000000, "kv_bytes_per_token": 32768,
              "session_fixed_bytes": 1100000000, "reserve": 0.85},
  "capabilities": {"reasoning": true, "images": true, "video": true,
                   "speculation": false, "rewind": false, "fork": false},
  "sessions": {"total": 7, "live": 1, "queued": 0},
  "state_dir": "/Users/…/Library/Application Support/Qwasar",
  "disk": {"sessions_bytes": 7312000000, "cache_bytes": 412000000, "free_bytes": 812000000000}
}
```

`context` is the window every session gets, and `live_sessions` how many may
be live at once. Both are derived by the server from the model and the
machine (`PLAN-qwasar.md` §2.3 carries the arithmetic over from Crucible),
not passed in by a client. `capabilities` is how the API generalises: a
model with no reasoning channel says so and never emits `reasoning`; a
server for a model whose cache can be truncated may say `"rewind": true` and
accept the reserved field in §5.1. Nothing else in this document changes.

### 4.2 `POST /v1/sessions` -- open

```json
{
  "system": "Investigate before answering…",
  "tools": [ {"type": "function", "function": {"name": "read", "…": "…"}} ],
  "thinking": true,
  "effort": "medium",
  "metadata": {"client": "crucible", "project": "…", "title": "…"}
}
```

`system` and `tools` are the prefix. `tools` are OpenAI-shaped function
schemas, verbatim (the same objects the compat endpoints take); how they are
rendered to the model, and how the model's calls are parsed back into JSON,
is the server's business. `thinking` and `effort` are here rather than on a
turn because for this model they render into the system turn: they are part
of the prefix or they are nothing. `metadata` is an opaque object the server
stores and lists, for a client with several sessions to tell them apart; it
has no meaning to the server.

Answer, `201`:

```json
{"id": "s_6f2a…", "prefix_tokens": 2214, "context": 262144,
 "warmth": {"state": "cold", "covered": 0}, "created": 1790000000}
```

Opening is cheap: the prefix is rendered and counted, nothing is evaluated.
The first step evaluates it -- and, for a prefix the server has seen before
in any session, reads it from the shared checkpoint the server keeps for
exactly that (the `resume` event says which). `prefix_tokens` is the number
to show a user: on this model with ten tool schemas it is ~2200, and it is
the same for every session of a project at a given effort.

### 4.3 `GET /v1/sessions/{id}` -- describe

```json
{"id": "s_6f2a…", "metadata": {"…": "…"}, "created": 1790000000,
 "tokens": 18335, "context": 262144, "prefix_tokens": 2214,
 "state": "idle",
 "warmth": {"state": "warm", "covered": 18335, "estimate_seconds": 2.1},
 "checkpoint_bytes": 719000000,
 "last_step": {"stop": "tool_calls", "at": 1790000420,
               "pending_calls": [{"id": "c_1", "name": "read"}]}}
```

`state` is one of `idle`, `queued`, `running`, `awaiting_tools` (the last
step ended in tool calls and no `continue` has come), `full` (the window is
used up; only `describe` and `DELETE` work). `warmth` is verified at the
moment of the call -- the server probes its store -- and `estimate_seconds`
is the resume cost from measured rates: a checkpoint read at the disk's
speed, or a re-prefill at the model's measured prefill rate for whatever is
not covered.

`GET /v1/sessions` returns `{"sessions": [ …describe objects… ]}`, newest
first, warmth included (the probe is a directory scan and a token compare per
session).

### 4.4 `POST /v1/sessions/{id}/turn` -- a user message

```json
{
  "text": "What does qw_edit_apply do?",
  "images": [ {"kind": "image", "media_type": "image/png", "data": "<base64>"} ],
  "sampling": {"temperature": 1.0, "top_k": 20, "top_p": 0.95, "min_p": 0, "seed": 0},
  "max_tokens": 32768
}
```

`images` (kind `image` or `video`, base64 bytes) are encoded by the server
and placed in the turn; a model without the capability refuses them with
`400`. `sampling` is per step because it does not touch the cache; absent
fields take the model's own generation defaults, `seed` 0 means the clock.
`max_tokens` bounds this step's generation (reasoning included); absent
means the room the window has left.

The answer is the event stream of §5.

### 4.5 `POST /v1/sessions/{id}/continue` -- tool results

```json
{"results": [ {"id": "c_1", "content": "…file contents…"},
              {"id": "c_2", "content": "error: no such file"} ]}
```

One result per call the last step emitted, matched by `id`, in the order
they were emitted; a missing or unknown id is `400`. The server renders them
as the model's tool turn and the model continues. On a session that is not
`awaiting_tools`, `409`.

A `turn` on an `awaiting_tools` session is allowed: the assistant turn is
closed as it stands and the user's message follows, so the model sees a call
it never got an answer to. A client that abandons a call should prefer a
`continue` whose content says why -- the model reasons better about an error
than a silence -- but the API does not force it.

### 4.6 `GET /v1/sessions/{id}/events` -- reattach

A dropped connection is not a cancel: the model is still generating into the
session, and the tokens are the session's whether or not anyone is reading.
A client that lost its stream reconnects here with `Last-Event-ID` set to the
last id it saw and receives every event after it, then the live tail. The
server keeps the events of the step in flight and of the most recent
finished step; older ones are gone (`404`). Without `Last-Event-ID`, the
whole of the current or last step is replayed.

### 4.7 `POST /v1/sessions/{id}/cancel`

Ends the step in flight at the next token, with `done.stop = "cancelled"`.
What was generated stays in the session -- it was evaluated, and this model
cannot un-evaluate -- so the next `turn` continues after it. A queued step
is removed from the queue instead. `200` with `{"cancelled": true|false}`.

### 4.8 `POST /v1/sessions/{id}/park`

"I am done here for now." The session's state is written to disk and its
memory freed -- its live slot, and anything the server would otherwise
keep for the next resume; `describe` then reports `warm`. (When the server
parks a session itself, to make room in the live set, it may keep that
memory for the next session it resumes.) The one verb a user should
ever see about the cache; everything else the server does on its own (§6).
`200` with the new warmth, or `409` while a step is running.

`{"save": false}` frees the memory without writing anything: whatever
checkpoints the session already has on disk stay as they are, and its
warmth is what they cover -- `warm` if they cover all of it, else `cold`
with `covered` saying how much a resume will restore. For a client that
wants the memory back now and will pay a re-prefill later, rather than wait
for a checkpoint write.

### 4.9 `DELETE /v1/sessions/{id}/checkpoint`

Deletes the session's own checkpoint and answers `{"freed_bytes": n}`. A
parked session becomes `cold`: its timeline is intact, and the next step
re-prefills whatever the shared prefix cache does not cover. A live session
is unaffected until it is next parked, which writes the checkpoint again.
`409` while a step runs. This is how a client gives disk back without losing
a conversation; `checkpoint_bytes` on each session and `/v1/server`'s `disk`
are what it decides by. The server never does this on its own.

### 4.10 `DELETE /v1/sessions/{id}`

Removes the session, its token log, its checkpoint and its events. `204`.
The shared prefix checkpoint is not the session's and stays.

### 4.11 `POST /v1/sessions/{id}/aside` -- a step off the record

```json
{"text": "Update your notes: …", "max_tokens": 1024, "sampling": {"temperature": 0.7}}
```

A user turn the session answers and then forgets. The server takes a rewind
point at the end of the timeline, evaluates `text` as a user turn with
thinking off and no tools, streams the answer as `text` events and a `done`
(`{"stop", "usage": {"prompt", "generated"}, "seconds"}`), and rolls the
session back to the rewind point: its timeline, token log and checkpoints
are exactly what they were, and the next real step continues as if the
aside had never run. For work that belongs beside a conversation rather
than in it -- the app's running notes, written while the user reads.

It runs only on an `idle` session that is `live`, only when the engine is
free (an aside never queues), and never in a session whose timeline holds
images (no rewind point can be taken there); otherwise `409`. A `turn`,
`park` or `DELETE` on the session ends a running aside at its next token,
rolls it back, and proceeds -- an aside never delays a real step by more
than that. qwasar-specific: it rests on the engine's rewind point
(`qwasar_session_mark`), which copies the recurrent state; a server for a
pure-attention model would truncate its cache instead.

## 5. The event stream

Events, in the order a step can produce them. `id` is `<step>.<n>`: the
step's number within the session, then the event's number within the step,
so `Last-Event-ID` (§4.6) is unambiguous.

| event | data | when |
|---|---|---|
| `queued` | `{"position": 1}` | another step holds the engine; repeated as the position changes |
| `resume` | `{"from": "live"\|"checkpoint"\|"cold", "restored": 18335, "prefill": 412, "prefix_cached": true}` | once, before any evaluation: what the server holds and what it is about to prefill |
| `prefill` | `{"done": 256, "total": 412}` | during prefill, once per chunk, over the whole outstanding span |
| `context` | `{"used": 18747, "limit": 262144}` | after prefill, and every few tokens of decode |
| `reasoning` | `{"text": "The user wants…", "tokens": 3}` | a delta of the reasoning block, with the tokens that produced it |
| `text` | `{"text": "qw_edit_apply is…"}` | a delta of the answer |
| `decode` | `{"generated": 128, "tokens_per_second": 61.2, "instantaneous": 66.0}` | every few tokens: the turn's average and the last second's rate |
| `call_progress` | `{"name": "write", "keys": ["path"], "tokens": 214}` | while a tool call is being written: markup is never streamed as text, and this is the sign the step is alive |
| `tool_call` | `{"id": "c_1", "name": "write", "arguments": {"path": "a.c", "content": "…"}}` | one per call, once the block is complete and parsed |
| `done` | see below | last, always |
| `error` | `{"message": "…"}` | instead of `done` when the engine failed |

`resume` is the guarantee of §2 made visible. `from: "live"` with
`restored` equal to the session's length is the common case and costs
nothing; `from: "checkpoint"` names a read; `from: "cold"` names a
re-prefill, and the `prefill` events that follow run over
`prefill` tokens, not over the current 1024-token chunk -- the bar a person
sees runs from zero to the whole cost once.

`tool_call.arguments` is an object. A parameter the tool's schema declares a
string is always a string; other values are raw JSON where the model wrote a
complete JSON value, else a string. Calls are emitted whole rather than
argument by argument, because a call is markup rather than prose and a
half-parsed one is not something a client can act on.

`done`:

```json
{"stop": "end_turn",
 "usage": {"prompt": 412, "generated": 1523, "reasoning": 1180},
 "timing": {"prefill_seconds": 0.9, "decode_seconds": 24.1,
            "first_token_seconds": 1.1},
 "speculation": {"rounds": 0, "committed": 0},
 "context": {"used": 20270, "limit": 262144},
 "warmth": {"state": "live", "covered": 20270}}
```

`stop` is one of:

| stop | meaning | session state after |
|---|---|---|
| `end_turn` | the model finished | `idle` |
| `tool_calls` | the model asked for tools; `tool_call` events preceded this | `awaiting_tools` |
| `length` | this step's `max_tokens` | `idle` |
| `cancelled` | `cancel` was called | `idle` |
| `context_full` | the window is used up (before evaluating, if the new material would not fit; or during decode at its end) | `full` |
| `shutdown` | the server is stopping; the session is checkpointed on the way out | `idle` |

`usage.prompt` counts only what this step evaluated -- a prefix read from a
checkpoint is not "prompt tokens", and folding it in would report a prefill
speed the machine cannot do.

### 5.1 Reserved

`turn.truncate_to` (a token count) for a server whose `capabilities.rewind`
is true: drop the timeline past that point before appending. Refused with
`400` where the capability is false. Named here so that a KV-only
implementation has a place for the one thing it can do that this model
cannot, without inventing a second API.

### 5.2 What is deliberately absent

- **History in a request.** There is no field for it. A client that wants a
  different past opens a session that has it.
- **Editing the prefix.** No `PATCH`. Effort, tools and the system prompt
  change on the next session, not this one.
- **Stop sequences and `tool_choice`.** Both exist on the compat endpoints
  for clients written to them. An agent that owns its loop needs neither.
- **A save verb.** `park` frees a slot; checkpoints are the server's
  mechanism, taken when they cost nothing to wait for (§6).

## 6. The server's side of the bargain

What a conforming server does with warmth, stated so a client can rely on it.

- **One step at a time.** The engine runs one thing. Steps from any session
  -- and requests on the compat endpoints, which are treated as one anonymous
  session -- wait in one queue and say so with `queued`.
- **The live set is bounded**, by `live_sessions` from `/v1/server`. A step
  on a session that is not live makes it live; if that exceeds the bound, the
  least recently used live session is parked first, which is a checkpoint
  write. The step's `resume` event reports its own resume, and the parked
  session's `describe` reports `warm` from then on.
- **Checkpoints are taken at boundaries, never per token and never per
  turn**: the shared prefix, the first time a session evaluates it (every
  later session of that prefix starts from it); a session as it is parked or
  evicted; every session on shutdown; and a long conversation every ~4K
  tokens past its last checkpoint, so that a session lost to a crash resumes
  near where it was rather than from the prefix.
- **The token log is written after every step**, so a session survives a
  server restart at worst `cold`, never lost.
- **Disk is the client's to budget, and the server never spends a session's
  state behind its back.** Shared prefixes live in an LRU store with a
  budget of its own. A session's checkpoint is a file of its own, written
  when it is parked (by request, by eviction from the live set, at shutdown)
  and as a long conversation grows (each time it has grown by a quarter,
  at least 4K tokens); nothing evicts it, and it goes when the session is
  deleted or its checkpoint dropped (§4.9). A resume reads whichever of the
  two covers more.

## 7. Two flows

A first turn on a project whose prefix another session has already paid for:

```
POST /v1/sessions            {"system": …, "tools": […], "effort": "medium"}
201  {"id": "s_1", "prefix_tokens": 2214, "warmth": {"state": "cold", "covered": 0}}

POST /v1/sessions/s_1/turn   {"text": "List the files."}
event: resume    {"from": "checkpoint", "restored": 2214, "prefill": 9, "prefix_cached": true}
event: context   {"used": 2223, "limit": 262144}
event: reasoning {"text": "I should call list", "tokens": 5}
event: call_progress {"name": "list", "keys": ["path"], "tokens": 14}
event: tool_call {"id": "c_1", "name": "list", "arguments": {"path": "."}}
event: done      {"stop": "tool_calls", "usage": {"prompt": 9, "generated": 19, "reasoning": 5}, …}

POST /v1/sessions/s_1/continue  {"results": [{"id": "c_1", "content": "README.md\nMakefile"}]}
event: resume    {"from": "live", "restored": 2242, "prefill": 11}
event: text      {"text": "Two files: …"}
event: done      {"stop": "end_turn", …}
```

Coming back to a parked conversation of 18K tokens the next day, on a
machine that ran something else in between:

```
GET  /v1/sessions/s_1
200  {"tokens": 18335, "state": "idle", "warmth": {"state": "warm", "covered": 18335, "estimate_seconds": 2.1}}

POST /v1/sessions/s_1/turn   {"text": "Now the tests."}
event: resume    {"from": "checkpoint", "restored": 18335, "prefill": 8}
event: text      …
```

Nothing in the second flow re-sent the first. That is the API.
