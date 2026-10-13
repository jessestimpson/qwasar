# qwasar-server

`make` in the repository root builds three programs: `./qwasar`,
`./qwasar-agent`, and `./qwasar-server`. (`make test` runs the unit and
golden-vector suites; some need a model.)

```
qwasar-server --port 8080
```

The server speaks the standard APIs — **OpenAI** (`/v1/chat/completions`)
and **Anthropic** (`/v1/messages`), streaming and not, with tools — so the
clients you already use work against it:

```
GET  /health
GET  /v1/models
GET  /v1/models/{id}
POST /v1/chat/completions   OpenAI, streaming and not, with tools
POST /v1/messages           Anthropic, streaming and not, with tools
POST /v1/messages/count_tokens
```

and its own **Session API** ([API.md](../API.md)), which the app and
`qwasar-agent` use.

## Why the Session API exists

> **Most clients are poor stewards of the KV cache.** Those APIs are
> stateless: the client resends the whole conversation every time, and the
> server can reuse its cached state only for the part that comes back
> byte-for-byte the same. Many clients change something early — a timestamp
> or a "current file" line in the system prompt, a tool added mid-session,
> the reasoning they were sent and dropped — and everything after the first
> difference is evaluated again. On a model whose recurrent layers cannot
> rewind, that is a re-prefill from that point: seconds at a few thousand
> tokens, **minutes** at a hundred thousand (192K tokens took 640 s on
> Flash-Next). Run the server with `-v` and its log names the message where
> each request diverged. If your client keeps missing, that is the client;
> the app and `qwasar-agent` don't, because they use the Session API.

The **Session API** ([API.md](../API.md)) is the server's own, built so a
client cannot waste the cache: a session's system prompt and tools are fixed
when it is opened, every later request carries only what is new (a message,
or tool results), and the server reports how warm each session is and
streams prefill progress. Sessions live on the server, survive its restarts,
and park to checkpoints on disk.

## The OpenAI and Anthropic endpoints

Both compat endpoints take `temperature`, `top_p`, `top_k`, `min_p`,
`top_n_sigma`, `thinking_budget` (Anthropic's `thinking.budget_tokens` too),
`presence_penalty`, `seed`, `max_tokens`, `stream`, and `tools`; the default
sampling is the model's own generation config, plus a presence penalty of 1.5
for the 9B (0 for the others). The OpenAI one also honours `stop`,
`tool_choice` (`none`, `auto`, `required`, or a named function), and
`stream_options.include_usage`; `n` other than 1 is refused with a 400. The
Anthropic one honours `stop_sequences`, `tool_choice` (`auto`, `any`,
`tool`, `none`), and prefill — a conversation ending in an assistant turn is
continued. `/v1/models` answers in Anthropic's shape to clients that send
`anthropic-version`, and errors on the Messages API use Anthropic's
envelope. Reasoning comes back as `reasoning_content` (OpenAI) or `thinking`
blocks (Anthropic) — **send it back** with the next request, or the prefix
stops matching at the first reply. `--cors` for browser clients; `--host
0.0.0.0` (with `--token`) for remote machines.

**One step at a time** — the engine runs one thing, and most of each model's
layers are recurrent, whose state cannot be forked the way a KV cache can.
Connections are served concurrently, so an idle keep-alive client does not
lock others out; their requests queue. The compat endpoints share one
anonymous session, which continues from wherever it already is when a
request extends it.

Images come in through both APIs (OpenAI `image_url` data URLs, Anthropic
base64 `source` blocks); video as an OpenAI-shaped `video_url` block, base64
only — the server never fetches paths or URLs on a request's behalf. A message
may carry images or a video, not both. A request with an image starts a fresh
session rather than risking a stale prefix match (two different pictures render
to identical placeholder tokens).

`/v1/responses` and `/v1/completions` return 501. `qwasar-server --help`
lists the options, including `--ctx`, `--live` (sessions held in memory at
once), and `--state-dir`.

## Checkpoints

The server checkpoints at boundaries, never per token: the shared system
prefix the first time a project evaluates it (every later session of that
prefix starts from it), a session as it is parked or evicted from the live
set, every session at shutdown, and a long conversation as it grows. A
session's checkpoint is its own file, deleted only when the session is or
when you drop it; shared prefixes live in an LRU store under
`~/.cache/qwasar/kv` with a 6 GB budget.

Checkpoints are large because the recurrent state is a fixed size whatever
the length — 147 MB for the 27B, so an 874-token checkpoint is ~214 MB — which
drives the cache design: few large entries, taken at boundaries rather than
per turn. Restoring that 874-token checkpoint took 0.02 s on an M4. On
Flash-Next a 192,441-token session parked to a 6.33 GB checkpoint and resumed
to its first new token in 11 s, or 16 s after a server restart, against 640 s
to rebuild it ([PLAN-qwasar.md](../PLAN-qwasar.md) M4).

## Testing it

`make test-api` starts the server and checks it against the Session API, the
OpenAI and the Anthropic specs (Python standard library only;
`QWASAR_SERVER_URL` targets a running one; `make test-api-toy` runs the
Session API suite on a toy model in seconds). With the `openai` or
`anthropic` package installed, the official clients are exercised too.
