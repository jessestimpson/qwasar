# qwasar

A small native inference engine for **Qwen3.8 27B** and **Qwen3.8
Flash-Next** on macOS Metal — and, for 16 GB Macs, **Qwen3.5 9B** — written
in C (with Objective-C only where Metal requires it). It runs one model family, end to end: weight loading, tokenizer,
chat template, Metal kernels, KV and recurrent state, a disk cache, an HTTP
server, a terminal coding agent, and a native macOS app.

```
$ ./qwasar -p "Name three prime numbers, with one sentence on why each is prime."

2 is prime because its only positive divisors are 1 and itself.
3 is prime because it cannot be divided evenly by any whole number other than 1 and 3.
5 is prime because its only positive divisors are 1 and 5.
```

The project is modelled on [ds4](https://github.com/antirez/ds4) (DwarfStar4):
a self-contained binary, abstractions built for this one model rather than for
generality, and an agent that ships in the same repo.

**Why C and Metal?** qwasar was written with the starting assumption that a narrow scope
is the easiest path to optimal token throughput. Specifically,
if we own all the kernels, aggressive fusion becomes a natural evolution of
the codebase rather than a endeavor in building higher level APIs for
general purpose use.

**Why these models?** The Qwen3.8 family seems pretty usable for agentic tasks,
with Qwen3.8-Flash-Next fitting well on a 128 GB mac.

## AI disclosure

**This software was written almost entirely by Claude Opus 5/5.5**, directed,
reviewed, and measured by the human author. If you would rather not use
AI-written code, this is not the project for you.

## Human Readability

We intend to keep this README AI-generated-but-human-readable. The rest of the
documentation is probably irksome. You'll likely have a better experience asking
an LLM about details of the project instead of reading top to bottom.

---

# Which model, on which Mac?

| | Qwen3.5 9B | Qwen3.8 27B | Qwen3.8 Flash-Next |
|---|---|---|---|
| Architecture | dense, 32 layers | dense, 64 layers | MoE, 125B total / 6B active, 48 layers |
| On disk (4-bit MLX) | ~6 GB | ~16 GB | ~111 GB |
| Held in memory | ~6 GB (~8 GB with a 32K context) | ~16 GB | ~80 GB (the engram table stays on disk) |
| Smallest Mac | 16 GB | 32 GB | 128 GB |
| Download | `./download_model.sh 9b` | `./download_model.sh 27b` | `./download_model.sh flash-next` |

What to expect, by machine, on prompts of up to 4K tokens:

| Mac | Qwen3.5 9B | Qwen3.8 27B | Qwen3.8 Flash-Next |
|---|---|---|---|
| 16–24 GB | fits; not yet measured | — | — |
| M4 MacBook Air, 32 GB | fits; not yet measured | prefill ~31–43 t/s, decode ~5–6 t/s (~8 with the draft head) | — |
| M5 Max, 128 GB | prefill ~2,100 t/s, decode ~88 t/s | prefill ~670 t/s, decode ~26 t/s | prefill ~1,200 t/s, decode ~60–66 t/s |

Decode on a dense model is bound by memory bandwidth — the 27B reads all of
its 14.95 GB of weights for every token — so it scales with the machine's
bandwidth far more than with its GPU. Long contexts are cheap: three of every
four layers are recurrent, with a fixed-size state, so only a quarter pay
per-token KV cache (64 KB a token for the 27B, 2 GB at 32K). The 27B's M4
figures predate its October 2026 kernels; see [Performance](#performance).

All three run from the same binaries; you choose by which weights you
download, and can keep several and switch. Not sure? Run
`./download_model.sh` with no arguments and it suggests one for your Mac's
memory.

* **Qwen3.5 9B (dense)** if you have a 16–24 GB Mac. Everything works — the
  app, the agent, tools, images — and it is quick, but it is a smaller model
  of the previous generation: weaker on long agent tasks, and its reasoning
  can circle on a trivial prompt (the reference implementation does the
  same). It gets a presence penalty of 1.5 by default, as Qwen recommends,
  which helps; `--effort low`, or `--no-think`, keeps it brief.
* **Qwen3.8 27B (dense)** if you have a 32–64 GB Mac. Everything works,
  including images and video, but at ~6 tokens a second on an M4 a long
  agent task is something you leave running (an M5 Max decodes it at ~26).
  Its context is sized to what the machine holds beside the weights.
* **Flash-Next (MoE)** if you have a 128 GB Mac. ~60 tokens a second on an
  M5 Max, the full 262K window, and a 32K-token budget per step — the model the
  agent is at its best on. It takes ~80 GB of memory while it runs, so on a
  128 GB machine it is most of what you have.

---

# Getting started

## Requirements

* An Apple Silicon Mac. The 27B was developed and measured on an M4 with
  32 GB; Flash-Next on an M5 Max with 128 GB. The 9B is checked against the
  reference implementation and measured on the M5 Max, but has not yet been
  run on a 16 GB machine. macOS 14 or later.
* Xcode command line tools — `cc`, `swiftc`, Foundation, and Metal. Not the
  Metal Toolchain: kernels are embedded as source and compiled at startup.
* Free memory for the model — ~6 GB for the 9B, ~16 GB for the 27B, ~80 GB
  for Flash-Next — plus a few GB for cache and context. The engine checks before it maps
  anything, and refuses a model that does not fit in the memory free at the
  time rather than squeezing the machine into swap.

## Get the weights

**Which one fits?** With no arguments the script reads this Mac's memory and
suggests a model, with the command to fetch it:

```
./download_model.sh
```

Or name one — **`9b`** (about 6 GB, for a 16 GB Mac), **`27b`** (about
16 GB, for a 32 GB Mac), or **`flash-next`** (about 111 GB, for a 128 GB
Mac):

```
./download_model.sh 27b
```

Each target links `./qwasar-<target>` beside the binaries: `./qwasar-9b`,
`./qwasar-27b`, `./qwasar-flash-next`. The binaries use `./qwasar-model` when
`-m` is absent; the first model you download becomes it, and `--default`
moves it to another (`./download_model.sh flash-next --default`). Otherwise
pass one explicitly (`-m ./qwasar-27b`). `27b-mtp` adds the 27B's optional
draft head for speculative decoding, and several targets can go in one
command (`./download_model.sh 27b 27b-mtp`). On a machine with too little
memory the script refuses before downloading (add `--force` to fetch it for
another Mac), and it checks the disk has room first.

All are resumable (re-run after an interruption) and pinned to a tested
revision, and only the files the engine reads are fetched. Add `--verify` to
check SHA-256 digests. Downloads go to `./models` (`QWASAR_MODEL_DIR` moves
them).

**Already have a model?** qwasar reads the MLX 4-bit conversions directly — an
LM Studio or Hugging Face copy works as is. Point at it with `-m <dir>`, set
`QWASAR_MODEL`, or symlink it to `./qwasar-model`. It must be **4-bit MLX
affine**: group 64 for the 27B (lmstudio-community's conversion) and the 9B
(mlx-community's), group 32 for Flash-Next (mlx-community's). Other formats
— FP8, GGUF, BF16, 6- and 8-bit — will not load (PLAN.md §1.2). The other
small Qwen3.5 models (4B, 2B, 0.8B) share the 9B's format but are untested.

## Run it: the app

The suggested way to run qwasar:

```
cd app && make run
```

That builds the engine's server and the app around it, then opens
**Qwasar.app**. It lives in the menu bar; **Open Coding Agent**
(⌘N) launches the UI Agent. On first launch, choose the model folder when asked,
add a project folder, and send your first message. `make install` puts it in
`/Applications`. [app/README.md](app/README.md) has the
details and what to expect.

* **The KV cache is something you can see and predict.** The server owns
  each conversation, so nothing is ever re-sent or re-prefilled by
  accident. Every session says whether it is live, warm on disk, or cold,
  what resuming it will cost, and how much disk its checkpoint holds; every
  prefill shows its progress; nothing is evicted behind your back.
* **The model's tools run on your Mac**, as you, with your login shell's
  environment — your PATH, your toolchains — and Stop kills a running
  command, not just the generation.

[What the app provides, and why](app/README.md#what-the-app-provides-and-why)
explains each choice.

## Run it: the server

`make` in this directory builds `./qwasar`, `./qwasar-agent`, and
`./qwasar-server`.

```
./qwasar-server --port 8080
```

It speaks **OpenAI** (`/v1/chat/completions`) and **Anthropic**
(`/v1/messages`), streaming and not, with tools, so the clients you already
use work against it — and its own **Session API** ([API.md](API.md)), which
keeps each conversation on the server so no client can waste the cache by
resending it. [docs/server.md](docs/server.md) has the endpoints, the
parameters, why stateless clients so often miss the cache on a recurrent
model, and how checkpoints work.

## Run it: the terminal agent

```
./qwasar-agent -C ~/src/project "fix the bug in stats.c and rebuild"
```

```
  read path=stats.c
  edit path=stats.c old=        if (v[i] < best) best = v[i];  new=        if (v[i] > b...
  bash command=cc -o stats stats.c && ./stats

Fixed. The comparison `v[i] < best` was tracking the minimum instead of the
maximum. Rebuilt and ran: mean=2.80 max=5.00
```

With no task it opens a REPL. It is a Session API client: it starts a server
if none is running, and its conversations persist there (`--resume last`).
Reads run unattended; writes and commands ask first.
[docs/terminal-agent.md](docs/terminal-agent.md) has the tools, the
commands, and the rest.

## Run it: the CLI

```
./qwasar -p "..."                  # generate
./qwasar --image <path> -p "..."   # with an image
./qwasar --video <path> -p "..."   # with a video
./qwasar -s "..." -p "..."         # with a system message
./qwasar -p "..." --show-think     # print the reasoning block too
./qwasar -p "..." --no-think       # skip reasoning entirely
./qwasar -p "..." --effort low     # xhigh (default) | medium | low
./qwasar --info                    # device, shards, architecture, memory
```

**Reasoning is on by default** and at the default effort the model thinks at
length — often hundreds of tokens before the visible answer starts. Reasoning
counts against the `-n` budget (default 512) but is not printed, so a turn can
look short while having spent most of its budget thinking; if it's cut off, it
says so. For short factual questions, `--effort low` is usually what you want.

## Images and video

```
$ ./qwasar --image circle.png --no-think -p "Describe this image in one short sentence."
image 224x224 -> 1 frame group -> 256 patches -> 64 tokens in 0.10s
A solid blue circle centered on a white background.
```

jpeg, png, bmp, and gif, via a vendored stb_image.

```
$ ./qwasar --video digits.mp4 --no-think -p "List every digit you see, in order."
video 224x224 -> 4 frame groups -> 784 patches -> 196 tokens in 0.41s
1, 2, 3, 4
```

Frames come from AVFoundation, so anything the Mac can play works. Sampling is
the model's own: two frames a second, pixel budget shared across the clip.

---

# Performance

How each number was measured, and what is behind it, is in
[docs/benchmarks.md](docs/benchmarks.md).

## On an M5 Max

**MacBook Pro, Apple M5 Max**, 128 GB, 2026-10-12: each point a fresh
process that prefills exactly N tokens, then decodes 128 greedy tokens at
that depth, started from a cooled die; medians of two runs.

| Model | N | Prefill | Decode |
|---|---:|---:|---:|
| Qwen3.5 9B | 4,096 | 2,095 t/s | 88.0 t/s |
| | 16,384 | 1,928 t/s | 75.5 t/s |
| | 32,768 | 1,707 t/s | 68.1 t/s |
| Qwen3.8 27B | 4,096 | 666 t/s | 26.1 t/s |
| | 16,384 | 548 t/s | 23.1 t/s |
| | 32,768 | 494 t/s | 21.0 t/s |
| Qwen3.8 Flash-Next | 4,096 | 1,199 t/s | 65.8 t/s |
| | 16,384 | 1,056 t/s | 58.1 t/s |
| | 32,768 | 954 t/s | 57.3 t/s |

The M5's GPU has matrix units that only Metal 4 tensor ops reach; on an M5 the
engine takes kernels built for them, which took prefill at 4K from 247 to 639
t/s on the 27B, 849 to 2,017 on the 9B, and 512 to 1,180 on Flash-Next.

## Flash-Next against mlx-vlm

The same 4-bit weights, the same machine, 2026-10-13: a coding agent's first
prompt from a cold session — no checkpoint, no prompt cache, no shared prefix
on either side. 4,872 tokens is what Qwasar.app's agent sends first; 16,384
is a heavier agent's. Time to first token runs through the first token;
decode is 256 tokens after it. Two runs each, from a cooled die.

| Prompt | | qwasar | mlx-vlm `main` | mlx-vlm `d1bd74ed` |
|---|---|---:|---:|---:|
| 4,872 tokens | time to first token | 4.20 / 4.34 s | 4.43 / 4.66 s | 4.05 / 3.91 s |
| | prefill | 1,159 / 1,122 t/s | 1,100 / 1,045 t/s | 1,204 / 1,246 t/s |
| | decode | 62.9 / 62.2 t/s | 36.6 / 38.8 t/s | 29.3 / 30.2 t/s |
| 16,384 tokens | time to first token | 16.6 / 16.0 s | 16.5 / 16.9 s | 15.9 / 15.5 s |
| | prefill | 988 / 1,025 t/s | 993 / 968 t/s | 1,031 / 1,057 t/s |
| | decode | 57.1 / 57.9 t/s | 35.8 / 36.8 t/s | 29.5 / 29.4 t/s |

Prefill is level with current mlx-vlm (`main`, `e928101a`); decode is
1.6–1.7× it. Loading is not in these numbers: ~22 s for qwasar against ~7 s
for mlx-vlm, once per process.

## Qwen3.8 27B on an M4

**MacBook Air, Apple M4**, 10 CPU / 10 GPU cores, 32 GB, fanless,
2026-08-23. Prefill in 256-token chunks; decode over 24 greedy tokens at the
stated depth:

| Context | Prefill | Decode |
| ---: | ---: | ---: |
| 506 | 43.1 t/s | 6.32 t/s |
| 2007 | 41.0 t/s | 6.21 t/s |
| 4002 | 31.3 t/s | 4.98 t/s |

Decode is at the memory-bandwidth roof (~120 GB/s, ~8 t/s by arithmetic).
The 27B's MTP draft head gets past it — **1.5× on prose, sustained**, with
output identical to greedy decoding:

```
./qwasar -m ./qwasar-27b --mtp ./qwasar-27b-mtp --spec -p "..."
```

These rows predate the October 2026 kernels — a tensor-op matmul, which alone
took a 2,069-token prefill from 34–38 to 41–50 t/s, and tiled prefill
attention — and have not been re-swept.

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
  the 4-bit format it reads, and the yardstick its speed is measured by.
* **The [Qwen3.8 MTP challenge](https://github.com/Layr-Labs/qwen-3.8-mtp-challenge)**
  — a public leaderboard for speculative decoding on this exact model, whose
  GPU token-selection shape qwasar follows.

# License

MIT — see [LICENSE](LICENSE). Everything qwasar builds on is permissively
licensed; [THIRD-PARTY.md](THIRD-PARTY.md) records each dependency and its
terms. Model weights are not part of this repository and carry their own
licence from the Qwen team.
