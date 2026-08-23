# qwasar

A small native inference engine for **Qwen3.8 27B** on macOS Metal, written in
C (with Objective-C only where Metal requires it). It runs one model, end to
end: weight loading, tokenizer, chat template, Metal kernels, KV and recurrent
state, a disk cache, a coding agent, and an HTTP server — all in one tree, one
`make`, no Python anywhere in the build or runtime.

```
$ qwasar -p "Name three prime numbers, with one sentence on why each is prime."

2 is prime because its only positive divisors are 1 and itself.
3 is prime because it cannot be divided evenly by any whole number other than 1 and 3.
5 is prime because its only positive divisors are 1 and 5.
```

The project is modelled on [ds4](https://github.com/antirez/ds4) (DwarfStar4)
and borrows its shape: a self-contained binary, abstractions built for this one
model rather than for generality, and an agent that ships in the same repo.

**Why this model?** Qwen3.8 27B is genuinely good, fits in 4-bit on a 32 GB
Mac, and is a hybrid recurrent/attention model — different enough from a plain
transformer that implementing it properly beats bolting it onto a generic
runner. And an engine small enough to hold in your head (~5,600 lines of
C/ObjC/Metal for the engine, ~7,200 with the CLI and agent) is easier to make
fast, and easier to read.

**Status:** beta, and young. Text, images, video, the agent, the server, and
the disk cache all work and are tested against the real model. Expect rough
edges — see [What is not implemented](#what-is-not-implemented).

## AI full disclosure

**This software was written almost entirely by Claude Opus 5**, directed,
reviewed, and measured by the human author. If you would rather not use
AI-written code, this is not the project for you. If you would: every number in
this file was measured on the machine described below, and where a measurement
contradicted an assumption, the code follows the measurement. Several such
cases — including wrong first answers — are recorded in `PLAN.md`.

---

# Getting started

## Requirements

* An Apple Silicon Mac. Developed and measured on an M4 with 32 GB.
* Xcode command line tools — just `cc`, Foundation, and Metal.
* ~16 GB of free RAM for the model, plus a few GB for cache and context.

You do **not** need the Metal Toolchain: kernels are embedded as source and
compiled at startup. (If you have it, `make check-metal` uses it as a fast
offline lint.)

## Build

```
make
```

That produces `./qwasar`, `./qwasar-agent`, and `./qwasar-server`.
`make test` runs the unit and golden-vector suites (needs the model).

## Get the weights

```
./download_model.sh model
```

About 16 GB, resumable (re-run after an interruption), pinned to a tested
revision. Add `--verify` to check SHA-256 digests. It links `./qwasar-model`,
which is where every binary looks by default, so afterwards:

```
./qwasar -p "Hello"
```

**Already have the model?** qwasar reads the MLX 4-bit conversion directly — an
LM Studio copy works as is. Point at it with `-m <dir>`, set `QWASAR_MODEL`, or
symlink it to `./qwasar-model`. Note it must be the **4-bit MLX affine, group
64** conversion; 8-bit or 6-bit will not load (see PLAN.md §1.2).

`qwasar --info` prints what it found: device, shards, architecture, memory.

## Run

```
qwasar -p "..."                  # generate
qwasar --image <path> -p "..."   # with an image
qwasar --video <path> -p "..."   # with a video
qwasar -s "..." -p "..."         # with a system message
qwasar -p "..." --show-think     # print the reasoning block too
qwasar -p "..." --no-think       # skip reasoning entirely
qwasar -p "..." --effort low     # xhigh (default) | medium | low
```

**Reasoning is on by default** and at the default effort the model thinks at
length — often hundreds of tokens before the visible answer starts. Reasoning
counts against the `-n` budget (default 512) but is not printed, so a turn can
look short while having spent most of its budget thinking; if it's cut off, it
says so. For short factual questions and tool work, `--effort low` is usually
what you want.

## The agent

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
`grep`, `bash`); commands `/help`, `/new`, `/effort`, `/think`, `/yes`, `/ctx`,
`/save`, `/quit`; `/image` and `/video` attach media mid-conversation.

Worth knowing:

* **Reads run unattended; writes and commands ask first** (unless `--yes`).
  A declined action is reported back to the model so it can try something else.
* You can **type the next message while the model is still writing**; it runs
  when the turn finishes. Ctrl-C interrupts. No alternate screen — the
  transcript scrolls and copies like normal terminal output.
* `edit` is line-anchored search and replace: the quoted text must match a run
  of whole lines exactly once, or the edit is refused. No fuzzy matching.
* If `AGENT.md` exists in the working directory it is added to the system
  prompt as project guidance.
* Prompt processing is checkpointed to `~/.cache/qwasar/kv`, so the system
  prefix prefills once and restores on later runs (`--no-cache` to disable).

## The server

An OpenAI- and Anthropic-compatible HTTP API:

```
qwasar-server --port 8080
```

```
GET  /health
GET  /v1/models
GET  /v1/models/{id}
POST /v1/chat/completions   OpenAI, streaming and not, with tools
POST /v1/messages           Anthropic, streaming and not, with tools
```

Both completion endpoints take `temperature`, `top_p`, `top_k`, `min_p`,
`seed`, `max_tokens`, `stream`, and `tools`. Reasoning comes back as
`reasoning_content` (OpenAI) or `thinking` blocks (Anthropic). `--cors` for
browser clients; `--host 0.0.0.0` for remote machines.

**One request at a time** — 48 of the 64 layers are recurrent and their state
cannot be forked the way a KV cache can. What does work is **prefix reuse**: a
stateless client resending a growing conversation continues from wherever the
live session already is — *provided it sends the assistant's reasoning back*
(as `reasoning_content` or a `thinking` block). Without that, every request
prefills from zero.

Images come in through both APIs (OpenAI `image_url` data URLs, Anthropic
base64 `source` blocks); video as an OpenAI-shaped `video_url` block, base64
only — the server never fetches paths or URLs on a request's behalf. A message
may carry images or a video, not both. A request with an image starts a fresh
session rather than risking a stale prefix match (two different pictures render
to identical placeholder tokens).

`/v1/responses` and `/v1/completions` return 501.

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
  agent feeds back the tokens it generated rather than re-rendering the
  conversation.

---

# Performance

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
qwasar-agent --mtp ./qwasar-mtp        # on by default once the head is given
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
sampling (Crucible uses it); the CLI and agent decode greedily and stay on
the exact verify. Details — the pruned draft vocabulary, the depth model, the
depth-4 ceiling — are in `PLAN.md`.

Startup: engine load 6–9 s, agent cold start 46.7 s, 21.1 s with the system
prefix cached, and restoring an 874-token checkpoint takes 0.02 s. Checkpoints
are large (~214 MB) because the recurrent state dominates, which drives the
cache design: few large entries, whole conversations saved only on `/save`.

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
qwasar_graph.c      session state and the forward pass
qwasar_kvstore.c    disk checkpoints of session state
qwasar_tokenizer.c  byte-level BPE, plus the ChatML template
qwasar_toolcall.c   tool-call parsing and the line-anchored edit matcher
qwasar_cpu.c        scalar fp32 reference twins for every kernel
qwasar_metal.m      Metal runtime: device, library, pipelines, dispatch
qwasar_agent.c      the agent loop, its tools, and the REPL
qwasar_server.c     the HTTP API
qwasar_sample.c     temperature, top-k, top-p and min-p sampling
metal/*.metal       kernels
tests/              unit and golden-vector regression
tools/              build helper; dev-only fixture generators (never built)
linenoise.c         vendored line editing (BSD-2, antirez)
```

`PLAN.md` carries the design, the measurements, the roadmap, and — deliberately
— the things that were tried and did not work, so they are not retried.

Conventions, inherited from ds4: no Python, no C++, correctness before speed,
mmap-backed loading, comments that explain *why*, and a narrow `qwasar.h` — the
CLI and agent never learn what a tensor is.

---

# What is not implemented

* **Sampling in the CLI.** The server samples; `qwasar` and `qwasar-agent` are
  still greedy.
* **`/v1/responses`, `/v1/completions`, and concurrent requests** in the
  server.
* **NFC normalisation** in the tokenizer (a no-op for ASCII and
  already-normalised text).
* **Todo tracking and a `glob` tool** in the agent.
* **Multi-GPU, CUDA, distributed inference.** Not planned. This targets one Mac.

Known rough edges: one shard costs a 4.99 GB copy at load due to its data
alignment; the prefill progress bar is chunky (one frame per 256 tokens); and
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
