# qwasar

A small native inference engine for **Qwen3.8 27B** and **Qwen3.8
Flash-Next** on macOS Metal, written in C (with Objective-C only where Metal
requires it). It runs one model family, end to end: weight loading, tokenizer,
chat template, Metal kernels, KV and recurrent state, a disk cache, an HTTP
server, a terminal coding agent, and a macOS app with a sandboxed one — all
in one tree, no Python anywhere in the engine's build or runtime.

```
$ qwasar -p "Name three prime numbers, with one sentence on why each is prime."

2 is prime because its only positive divisors are 1 and itself.
3 is prime because it cannot be divided evenly by any whole number other than 1 and 3.
5 is prime because its only positive divisors are 1 and 5.
```

The project is modelled on [ds4](https://github.com/antirez/ds4) (DwarfStar4)
and borrows its shape: a self-contained binary, abstractions built for this one
model rather than for generality, and an agent that ships in the same repo.

**Why these models?** Qwen3.8 27B is genuinely good, fits in 4-bit on a 32 GB
Mac, and is a hybrid recurrent/attention model — different enough from a plain
transformer that implementing it properly beats bolting it onto a generic
runner. Flash-Next is the same family grown into a 125B mixture of experts
with 6B active per token: the same tokenizer, template and recurrent layers,
~80 GB resident, decoding at ~68 t/s on a 128 GB M5 Max against the 27B's ~6
on a 32 GB M4 ([PLAN-flash-next.md](PLAN-flash-next.md)). One code path serves both, driven
by the model's config. And an engine small enough to hold in your head is
easier to make fast, and easier to read.

| | Qwen3.8 27B | Qwen3.8 Flash-Next |
|---|---|---|
| Architecture | dense, 64 layers | MoE, 125B total / 6B active, 48 layers |
| On disk (4-bit MLX) | ~16 GB | ~111 GB |
| Held in memory | ~16 GB | ~80 GB (the engram table stays on disk) |
| Smallest Mac | 32 GB | 128 GB |
| Decode | ~6 t/s on an M4 (1.5x with the draft head) | ~68 t/s on an M5 Max |
| Download | `./download_model.sh model` | `./download_model.sh flash-next` |

**Status:** beta, and young. Text, images, video, the app, the agent, the
server, and the disk cache all work and are tested against the real models.
Expect rough edges — see [What is not implemented](#what-is-not-implemented).

## Which model?

Both run from the same binaries; you choose by which weights you download,
and can keep both and switch.

* **Qwen3.8 27B (dense)** if you have a 32–64 GB Mac. Everything works,
  including images and video, but at ~6 tokens a second a long agent task is
  something you leave running. Its context is sized to what the machine
  holds beside the weights.
* **Flash-Next (MoE)** if you have a 128 GB Mac. ~68 tokens a second on an
  M5 Max against the 27B's ~6 on an M4, the full 262K window, and a 32K-token budget per step — the model the
  agent is at its best on. It takes ~80 GB of memory while it runs, so on a
  128 GB machine it is most of what you have.

## AI full disclosure

**This software was written almost entirely by Claude Opus 5**, directed,
reviewed, and measured by the human author. If you would rather not use
AI-written code, this is not the project for you. If you would: every number in
this file was measured on the machine described beside it, and where a
measurement contradicted an assumption, the code follows the measurement.
Several such cases — including wrong first answers — are recorded in the
`PLAN*.md` files.

---

# Getting started

## Requirements

* An Apple Silicon Mac. The 27B was developed and measured on an M4 with
  32 GB; Flash-Next on an M5 Max with 128 GB. macOS 14 or later.
* Xcode command line tools — `cc`, `swiftc`, Foundation, and Metal.
* For the app's sandbox, which is built natively on the first run:
  `python3`, [mise](https://mise.jdx.dev) (it supplies pinned Erlang, Elixir
  and Zig), and `brew install e2fsprogs`. No Docker, no Linux.
* Free memory for the model — ~16 GB for the 27B, ~80 GB for Flash-Next —
  plus a few GB for cache and context. The engine checks before it maps
  anything, and refuses a model that does not fit in the memory free at the
  time rather than squeezing the machine into swap.

You do **not** need the Metal Toolchain: kernels are embedded as source and
compiled at startup. (If you have it, `make check-metal` uses it as a fast
offline lint.)

## Get the weights

**The 27B** — about 16 GB:

```
./download_model.sh model
```

**Flash-Next** — about 111 GB, and a 128 GB Mac to run it:

```
./download_model.sh flash-next --default
```

`model` links `./qwasar-model`, which is where every binary looks by default.
`flash-next` links `./qwasar-flash-next`; with `--default` it points
`./qwasar-model` at it too. Without `--default`, pass it explicitly
(`-m ./qwasar-flash-next`). On a machine with too little memory the script
refuses before downloading (add `--force` to fetch it for another Mac), and
it checks the disk has room first.

Both are resumable (re-run after an interruption) and pinned to a tested
revision, and only the files the engine reads are fetched. Add `--verify` to
check SHA-256 digests. `./download_model.sh mtp-head` adds the 27B's optional
draft head for speculative decoding (below); `all` is the 27B and its head.
Downloads go to `./models` (`QWASAR_MODEL_DIR` moves them).

**Already have a model?** qwasar reads the MLX 4-bit conversions directly — an
LM Studio or Hugging Face copy works as is. Point at it with `-m <dir>`, set
`QWASAR_MODEL`, or symlink it to `./qwasar-model`. It must be **4-bit MLX
affine**: group 64 for the 27B (lmstudio-community's conversion), group 32 for
Flash-Next (mlx-community's, whose 8-bit router gates and 4-bit engram table
the loader handles). FP8, GGUF, BF16, or 6- and 8-bit conversions will not
load (see PLAN.md §1.2). The engine tells the two models apart from
`config.json`, not the folder name.

## Run it: the app

The suggested way to run qwasar:

```
cd app && make run
```

That builds the engine's server, the app around it, and — the first time —
the Linux guest its tools run in (a few minutes, downloads cached after),
then opens **Qwasar.app**. It lives in the menu bar; **Open Coding Agent**
(⌘N) is the window. On first launch, choose the model folder when asked,
add a project folder, and type. [app/README.md](app/README.md) has the
details and what to expect.

The app is opinionated, and the opinions are the point of it:

* **The KV cache is something you can see and predict.** The server owns
  each conversation, so nothing is ever re-sent or re-prefilled by
  accident. Every session says whether it is live, warm on disk, or cold,
  what resuming it will cost, and how much disk its checkpoint holds; every
  prefill shows its progress; nothing is evicted behind your back.
* **The model's tools run in a VM with no network device.** Your project
  folder is the only thing it can see, your real `.git` is out of its
  reach, and network access is off unless you grant a project specific
  hosts — which the app, not the guest, then fetches from.

[What the app provides, and why](app/README.md#what-the-app-provides-and-why)
explains each choice.

## Run it: the server

`make` in this directory builds three programs: `./qwasar`, `./qwasar-agent`,
and `./qwasar-server`. (`make test` runs the unit and golden-vector suites;
some need a model.)

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

The **Session API** ([API.md](API.md)) is the server's own, built so a
client cannot waste the cache: a session's system prompt and tools are fixed
when it is opened, every later request carries only what is new (a message,
or tool results), and the server reports how warm each session is and
streams prefill progress. Sessions live on the server, survive its restarts,
and park to checkpoints on disk.

Both compat endpoints take `temperature`, `top_p`, `top_k`, `min_p`, `seed`,
`max_tokens`, `stream`, and `tools`; the default sampling is the model's own
generation config. The OpenAI one also honours `stop`, `tool_choice` (`none`,
`auto`, `required`, or a named function), and
`stream_options.include_usage`; `n` other than 1 is refused with a 400. The
Anthropic one honours `stop_sequences`, `tool_choice` (`auto`, `any`,
`tool`, `none`), and prefill — a conversation ending in an assistant turn is
continued. `/v1/models` answers in Anthropic's shape to clients that send
`anthropic-version`, and errors on the Messages API use Anthropic's
envelope. Reasoning comes back as `reasoning_content` (OpenAI) or `thinking`
blocks (Anthropic) — **send it back** with the next request, or the prefix
stops matching at the first reply. `--cors` for browser clients; `--host
0.0.0.0` (with `--token`) for remote machines.

**One step at a time** — the engine runs one thing, and 48 of the 27B's 64
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

`make test-api` starts the server and checks it against the Session API, the
OpenAI and the Anthropic specs (Python standard library only;
`QWASAR_SERVER_URL` targets a running one; `make test-api-toy` runs the
Session API suite on a toy model in seconds). With the `openai` or
`anthropic` package installed, the official clients are exercised too.

## Run it: the terminal agent

```
qwasar-agent -C ~/src/project "fix the bug in stats.c and rebuild"
```

```
  read path=stats.c
  edit path=stats.c old=        if (v[i] < best) best = v[i];  new=        if (v[i] > b...
  bash command=cc -o stats stats.c && ./stats

Fixed. The comparison `v[i] < best` was tracking the minimum instead of the
maximum. Rebuilt and ran: mean=2.80 max=5.00
```

With no task it opens a REPL. Six tools (`read`, `write`, `edit`, `list`,
`grep`, `bash`); commands `/help`, `/new`, `/sessions`, `/effort`, `/think`,
`/yes`, `/ctx`, `/save`, `/quit`; `/image` and `/video` attach media
mid-conversation.

**The agent is a Session API client.** Its system prompt and tools become a
session the server owns, and every turn sends only what is new. If nothing
is listening on the port it starts a server itself (`-m` or `$QWASAR_MODEL`
names the model), which stops when the agent does. Conversations persist on
the server: `--resume last` continues this directory's most recent one,
`/sessions` lists them, `/save` parks the current one warm on disk. The
agent links no engine: it is 170 KB and starts instantly.

Unlike the app, **its tools run on your machine**, as you. Worth knowing:

* **Reads run unattended; writes and commands ask first** (unless `--yes`).
  A declined action is reported back to the model so it can try something else.
* You can **type the next message while the model is still writing**; it runs
  when the turn finishes. Ctrl-C interrupts. No alternate screen — the
  transcript scrolls and copies like normal terminal output.
* `edit` is line-anchored search and replace: the quoted text must match a run
  of whole lines exactly once, or the edit is refused. No fuzzy matching.
* If `AGENT.md` exists in the working directory it is added to the system
  prompt as project guidance.
* A task stops after 24 rounds of tool calls (`--steps` changes it).

## Run it: the CLI

```
qwasar -p "..."                  # generate
qwasar --image <path> -p "..."   # with an image
qwasar --video <path> -p "..."   # with a video
qwasar -s "..." -p "..."         # with a system message
qwasar -p "..." --show-think     # print the reasoning block too
qwasar -p "..." --no-think       # skip reasoning entirely
qwasar -p "..." --effort low     # xhigh (default) | medium | low
qwasar --info                    # device, shards, architecture, memory
```

**Reasoning is on by default** and at the default effort the model thinks at
length — often hundreds of tokens before the visible answer starts. Reasoning
counts against the `-n` budget (default 512) but is not printed, so a turn can
look short while having spent most of its budget thinking; if it's cut off, it
says so. For short factual questions, `--effort low` is usually what you want.

## Images and video

```
$ qwasar --image circle.png --no-think -p "Describe this image in one short sentence."
image 224x224 -> 256 patches -> 64 tokens in 2.1s
A solid blue circle centered on a white background.
```

jpeg, png, bmp, and gif, via a vendored stb_image. The vision tower is 27
blocks of bf16, validated against mlx-vlm at rel L2 5.5e-3 on identical patches
— closer to the fp32 reference than the reference's own bf16 path.

```
$ qwasar --video digits.mp4 --no-think -p "List every digit you see, in order."
video 224x224 -> 4 frame groups -> 784 patches -> 196 tokens in 3.1s
1, 2, 3, 4
```

Frames come from AVFoundation, so anything the Mac can play works. Sampling is
the model's own: two frames a second, pixel budget shared across the clip.

---

# The model, briefly

Qwen3.8 27B is a hybrid: **every fourth layer is full attention, the other 48
are Gated DeltaNet**, a recurrent linear-attention layer with a fixed-size
fp32 state. Attention is output-gated, and RoPE is partial (64 of 256 dims) and
multimodal (three interleaved position axes). The interesting parts are each
about a page of C, with a scalar CPU reference twin beside them.

Two practical consequences:

* **Long context is cheap.** Only 16 layers pay per-token KV, so the cache is
  64 KB/token (2 GB at 32K) and the recurrent state is a constant 147 MB.
* **A session is append-only.** Recurrent state cannot be rewound, only
  extended. That is why the disk cache reuses only strict prefixes, and why the
  Session API has no way to send history: a conversation only grows.

Flash-Next keeps the same hybrid of recurrent and full-attention layers (48
layers, every fourth full attention) and replaces the dense MLP with a
mixture of experts: 125B parameters, 6B active per token, 32 KB of cache per
token. Both consequences above hold for it too.

---

# Performance

## Qwen3.8 27B

All numbers from one machine: **MacBook Air, Apple M4**, 10 CPU / 10 GPU cores,
32 GB, macOS 26.5.1, fanless. Measured 2026-08-23, after the compact draft
head, the re-measured depth table, and the decode-timer fix (earlier readmes
carried figures whose speculative decode excluded drafting time; these do not).

Prefill in 256-token chunks; decode over 24 greedy tokens at the stated depth:

| Context | Prefill | Decode |
| ---: | ---: | ---: |
| 506 | 43.1 t/s | 6.32 t/s |
| 2007 | 41.0 t/s | 6.21 t/s |
| 4002 | 31.3 t/s | 4.98 t/s |

Serial decode is at the memory-bandwidth roof: a dense 27B reads all 14.95 GB
of weights per token, which at ~120 GB/s caps out around 8 t/s by arithmetic.
The 4K row ran directly after two minutes of continuous prefill on a fanless
chassis, so it carries thermal load the short rows do not; the depth cost
itself is small, which is the hybrid schedule earning its keep.

Prefill reaches ~80% of MLX's quantised matmul throughput (2.25 vs 2.73–2.82
TFLOP/s on identical shapes), which is the honest target. The optimization
history is in `PLAN.md`.

**Speculative decoding gets past the bandwidth roof.** The model ships a
one-layer MTP draft head (`./download_model.sh mtp-head`, quantised to 4-bit at
load):

```
qwasar --mtp ./qwasar-mtp --spec -p "..."
```

**1.5x on prose, sustained.** Three alternating serial/speculative pairs of
200 tokens, so both share thermal state: serial 5.41 / 5.64 / 5.63 t/s,
speculative 8.32 / 8.42 / 8.38 t/s — ratios 1.54, 1.49, 1.49, with 2.58
tokens committed per round at mean depth 2.63 and drafting costing 1.3 s of a
24 s run. The stability is new: before the compact draft head and the
re-measured depth table this faded from 1.65x toward 1.4x as the chassis
warmed, and the 1.65x itself came from a timer that excluded drafting.

Draft depth adapts per round from observed acceptance; `--mtp-depth <n>`
overrides, `0` disables. The target verifies every draft, so output is
guaranteed identical to greedy decoding — `tests/test_verify` pins that
exactly. For sampling callers there is `qwasar_session_verify_sampled`, a
rejection-sampling verify whose output is distributed exactly as serial
sampling, for embedders; the CLI decodes greedily and stays on the exact
verify. The server does not run a draft head yet, so the app and the agent
decode serially. Details — the pruned draft vocabulary, the depth model, the
depth-4 ceiling — are in `PLAN.md`.

Startup: engine load 6–9 s. Restoring an 874-token checkpoint takes 0.02 s;
checkpoints are large (~214 MB at that length) because the recurrent state
is a fixed 147 MB whatever the length, which drives the cache design: few
large entries, taken at boundaries rather than per turn.

## Flash-Next

Measured on an **M5 Max with 128 GB**: ~68 t/s decode and ~465 t/s prefill
on short prompts; engine load ~22 s. At long context prefill slows as
attention grows — 192,441 tokens took 640 s, about 300 t/s — which is what
makes the cache matter: that session parked to a 6.33 GB checkpoint and
resumed to its first new token in 11 s, or 16 s after a server restart,
against 640 s to rebuild it ([PLAN-qwasar.md](PLAN-qwasar.md) M4).

## Checkpoints

The server checkpoints at boundaries, never per token: the shared system
prefix the first time a project evaluates it (every later session of that
prefix starts from it), a session as it is parked or evicted from the live
set, every session at shutdown, and a long conversation as it grows. A
session's checkpoint is its own file, deleted only when the session is or
when you drop it; shared prefixes live in an LRU store under
`~/.cache/qwasar/kv` with a 6 GB budget.

---

# Correctness

* Every Metal kernel has a **scalar fp32 CPU twin**, tested against **real
  model weights** (synthetic weights would miss quantisation-layout misreads).
* `tests/test_forward.c` replays golden activations from mlx-vlm and requires
  the argmax and all five top-5 ranks to match exactly, reporting per-layer
  drift so a divergence is located, not just detected.
* The gated-delta recurrence is **bit-identical** between streaming and batched
  execution — the prefill/decode seam.
* Attention provably ignores cache slots past its position (the test poisons
  471 of them and requires unchanged output).
* A restored session produces **bit-identical** next-token logits.
* The tokenizer matches the reference on 24 cases and all 8 chat template
  renderings.

Logit L2 is deliberately *not* what the tests lean on: the bf16 reference
disagrees with itself by 7.4e-2 relative when only accumulation order changes.
That reasoning is written down in `PLAN.md` so it doesn't get "fixed" into a
false failure later.

---

# Layout

```
qwasar.h            public engine boundary -- no tensor internals escape it
qwasar.c            config, safetensors mmap, weight table
qwasar_graph.c      session state and the forward pass (the 27B)
qwasar_flash_graph.c  the Flash-Next forward pass, on Metal
qwasar_flash_cpu.c  its scalar reference twin
qwasar_kvstore.c    disk checkpoints of session state
qwasar_tokenizer.c  byte-level BPE, plus the ChatML template
qwasar_toolcall.c   tool-call parsing and the line-anchored edit matcher
qwasar_cpu.c        scalar fp32 reference twins for every kernel
qwasar_metal.m      Metal runtime: device, library, pipelines, dispatch
qwasar_vision.c     the vision tower; qwasar_image.c and qwasar_video.m decode
qwasar_sample.c     temperature, top-k, top-p and min-p sampling
qwasar_cli.c        ./qwasar
qwasar_server.c     ./qwasar-server, and its OpenAI and Anthropic endpoints
qwasar_api.c        the Session API's endpoints (API.md)
qwasar_sessions.c   the session store, the scheduler, and generation
qwasar_profile.c    context size and live sessions, from the model and machine
qwasar_http.c       the HTTP/1.1 core the server's fronts share
qwasar_agent.c      ./qwasar-agent: a Session API client, its tools, the REPL
metal/*.metal       kernels
tests/              unit, golden-vector and API regression
tools/              build helper; dev-only fixture generators (never built)
linenoise.c         vendored line editing (BSD-2, antirez)
app/                Qwasar.app: the menu bar item and the coding agent's window
```

The plans carry the design, the measurements, and — deliberately — the
things that were tried and did not work, so they are not retried: `PLAN.md`
for the engine and the 27B, `PLAN-flash-next.md` for Flash-Next,
`PLAN-qwasar.md` for the Session API and the app's move onto it, and
[app/spec/](app/spec/README.md) for the app.

Conventions, inherited from ds4: no Python, no C++, correctness before speed,
mmap-backed loading, comments that explain *why*, and a narrow `qwasar.h` — the
CLI and agent never learn what a tensor is.

---

# What is not implemented

* **Sampling in `qwasar`.** The server samples (and so does the agent,
  through it; `--temperature 0` makes it greedy); the plain CLI is still
  greedy.
* **`/v1/responses`, `/v1/completions`, and concurrent steps** in the
  server (requests queue for the one engine).
* **A draft head for Flash-Next**, and speculative decoding in the server.
* **NFC normalisation** in the tokenizer (a no-op for ASCII and
  already-normalised text).
* **Todo tracking and a `glob` tool** in the agent.
* **Multi-GPU, CUDA, distributed inference.** Not planned. This targets one Mac.

Known rough edges: shards whose tensors are off 4-byte alignment are copied
at load (one for the 27B, most of Flash-Next's); the first step after resuming a session in a freshly started server
is slower than later ones; the prefill progress bar is chunky (one frame per 256 tokens); and
`qmm` sits at 80% of MLX's throughput with no cheap way found yet to close the
gap.

---

# Thanks

* **[ds4](https://github.com/antirez/ds4) and Salvatore Sanfilippo** — qwasar
  exists because ds4 showed what a single-model native engine looks like done
  with care. The build story, CPU reference twins, engine/session boundary,
  in-tree agent, and disk cache are all borrowed. `linenoise.c` is vendored
  from that tree under its own BSD-2 licence.
* **[MLX](https://github.com/ml-explore/mlx) and
  [mlx-vlm](https://github.com/Blaizzy/mlx-vlm)** — the reference
  implementation this engine was written and validated against, the source of
  the 4-bit format it reads, and the throughput target its kernels are measured
  against (still ahead).
* **The [Qwen3.8 MTP challenge](https://github.com/Layr-Labs/qwen-3.8-mtp-challenge)**
  — a public leaderboard for speculative decoding on this exact model, whose
  GPU token-selection shape qwasar follows. Two of their findings not yet acted
  on are recorded in `PLAN.md`.
* **The Qwen team**, for open weights worth building for, and a model card
  precise enough to reimplement from.

# License

MIT — see [LICENSE](LICENSE). Everything qwasar builds on is permissively
licensed; [THIRD-PARTY.md](THIRD-PARTY.md) records each dependency and its
terms. Model weights are not part of this repository and carry their own
licence from the Qwen team.
