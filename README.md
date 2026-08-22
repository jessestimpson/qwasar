# qwasar

**qwasar** is a small native inference engine for **Qwen3.8 27B** on macOS
Metal. It is written in C, with Objective-C only where Metal requires it, and it
is deliberately narrow: not a general model runner, but one model implemented
end to end. Weight loading, the tokenizer, the chat template, the Metal kernels,
the KV and recurrent state, the disk cache, and the coding agent are built and
tested together.

It is modelled on [ds4](https://github.com/antirez/ds4) (DwarfStar4), and takes
its shape from it: no Python in the build or the runtime, one `make` to a
self-contained binary, abstractions built for inference rather than for
supporting many architectures, and an agent shipped in the same tree.

```
$ qwasar -p "Name three prime numbers, with one sentence on why each is prime."

2 is prime because its only positive divisors are 1 and itself.
3 is prime because it cannot be divided evenly by any whole number other than 1 and 3.
5 is prime because its only positive divisors are 1 and 5.
```

## What you can do with this

* Run a capable 27B model locally on an ordinary Mac, from a binary you can read
  end to end. The engine is about 5,600 lines of C, Objective-C, and Metal;
  7,200 with the CLI and the agent, and 2,000 more of tests.
* Use `qwasar-agent` to let the model read and edit real files, search a tree,
  and run commands, either as a one-shot task or in a REPL.
* Read a complete, working implementation of a **hybrid recurrent/attention**
  model. The interesting parts of this architecture — the gated delta rule, the
  output-gated attention, partial multimodal RoPE — are each a page of C with a
  scalar reference twin beside them.
* Take the Metal kernels as a starting point. They are commented for why, not
  what, and each has a CPU reference and a test that runs both against real
  weights.

## Motivations

* Qwen3.8 27B is a genuinely good model that fits in 4-bit on a 32 GB machine.
* A hybrid model with 48 recurrent layers behaves differently enough from a
  plain transformer — in memory, in caching, in what a session can and cannot do
  — that implementing it properly is worth doing rather than bolting onto a
  generic runner.
* An engine narrow enough to hold in your head is easier to make fast than a
  general one, and easier to hand to a coding agent to make faster still.

# AI full disclosure

**This software was written almost entirely by Claude Opus 5**, working from
direction, review, and hardware provided by the human author. The design
decisions, the measurements, the kernel work, and the prose in this file are all
model output, shaped by a human deciding what to build, what to reject, and what
to measure next.

We say this plainly because it shaped the result. If you would rather not use
AI-written code, this is not the project for you. If you would, then the
disclosure matters in the other direction too: everything here that claims a
number was measured on the machine described below, and where a measurement
contradicted an assumption, the code and the notes follow the measurement. There
are several such cases recorded in `PLAN.md`, including a few where the first
answer was wrong.

## Thanks

**[ds4](https://github.com/antirez/ds4) and Salvatore Sanfilippo.** qwasar
exists because ds4 showed what a single-model native inference engine looks
like when it is done with care. The shape of this project is borrowed
throughout: the build story, the CPU reference twins, the engine/session
boundary, the agent living in the same tree, the disk KV cache, and the prefill
progress bar. Where qwasar departs — line-anchored edits instead of `[upto]`
anchors, a different checkpoint design — it is because this model forced it, not
because the original was wrong. `linenoise.c` is vendored from that tree, under
its own BSD-2 licence and with its notices intact.

**[MLX](https://github.com/ml-explore/mlx) and
[mlx-vlm](https://github.com/Blaizzy/mlx-vlm).** The reference implementation of
Qwen3.8 in mlx-vlm is what made this possible at all: it is the specification
this engine was written against, and the oracle every layer was validated
against. The 4-bit affine quantisation format qwasar reads is MLX's. MLX's own
quantised matmul is the throughput target the kernels here are measured against,
and it is still ahead.

**The Qwen team**, for open weights worth building for, and for a model card and
chat template precise enough to reimplement from.

## Status

Beta, and young. Text generation, the tokenizer, the chat template, the agent,
and the disk cache all work and are covered by tests that run against the real
model. **Vision is not implemented** — the tower is parsed and reported but not
executed. Expect rough edges, and see *What is not implemented* below.

---

# Getting started

## Requirements

* An Apple Silicon Mac. Developed and measured on an M4 with 32 GB.
* Xcode command line tools. That is all: `cc`, Foundation, and Metal.
* About 16 GB of free RAM for the model, plus a few GB for cache and context.

You do **not** need the Metal Toolchain component. Kernels are embedded in the
binary as source and compiled at startup, so the binary is self-contained and
builds on a machine that has never opened Xcode. If you happen to have the
toolchain installed, `make check-metal` will use it as a fast offline lint.

## qwasar-server

An OpenAI and Anthropic compatible HTTP API, following ds4-server so the same
clients work against either.

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

Both completion endpoints accept `temperature`, `top_p`, `top_k`, `min_p`,
`seed`, `max_tokens`, `stream`, and `tools`; reasoning is returned as
`reasoning_content` on the OpenAI side and as `thinking` blocks on the
Anthropic side. `--cors` adds `Access-Control-Allow-*` headers for browser
clients. `--host 0.0.0.0` is required for remote machines to connect.

**One request is served at a time.** ds4-server has `--batched-session` and a
mixed prefill scheduler; qwasar has no counterpart, because 48 of this model's
64 layers are recurrent and their state cannot be forked the way a KV cache can.

What does carry over is prefix reuse, which is what matters for agent clients: a
stateless client resending a growing conversation continues from wherever the
live session already is. **This only works if the client sends the assistant's
reasoning back** — as `reasoning_content` or as a `thinking` block. Without it
the replayed turn cannot match what the session generated, and every request
prefills from zero. With it, a second turn of a short conversation reused 80 of
its 98 tokens.

`/v1/responses` and `/v1/completions` are not implemented and return 501.

## Build

```
make
```

That is the whole thing. It produces `./qwasar`, `./qwasar-agent` and
`./qwasar-server`.

```
make test          # unit and golden-vector suites; needs the model
make check-metal   # optional offline kernel lint; needs the Metal Toolchain
```

## Weights

```
./download_model.sh model
```

About 16 GB. It fetches only the six files qwasar reads — `config.json`, the
safetensors index, the three shards and `tokenizer.json` — checks their sizes,
and links `./qwasar-model`, which is where every binary looks when `-m` is
absent. So after it finishes:

```
./qwasar -p "Hello"
```

Downloads resume; run the same command again after an interruption. Add
`--verify` to check every file's SHA-256 against the digest Hugging Face
reports. The repository is pinned to a revision, so a re-run fetches the bytes
this was tested against rather than whatever `main` has become.

**Already have the model?** qwasar reads the MLX 4-bit conversion directly, so
an LM Studio copy works as is — point at it with `-m <dir>`, set
`QWASAR_MODEL`, or link it yourself:

```
ln -s ~/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-4bit qwasar-model
```

Binaries look for the model in `$QWASAR_MODEL`, then `./qwasar-model`, then a
`qwasar-model` beside the executable — so a binary copied onto your `PATH`
still finds the weights it was downloaded next to.

**4-bit only.** MLX affine, group 64. The kernels are built around that one
format deliberately (PLAN.md §1.2); an 8-bit or 6-bit conversion of the same
model will not load.

Start by confirming qwasar agrees with you about what it found:

```
qwasar --info
```

```
model      qwasar-model
device     Apple M4  (working set 25.0 GB, max buffer 18.7 GB)
shards     3, 2180 tensors  (9.96 GB mapped, 4.99 GB copied for alignment)

text       hidden 5120  layers 64  vocab 248320  ffn 17408  eps 1e-06
schedule   full attention every 4 layers -> 16 full, 48 gated-delta
attention  24 q heads x 256 dim, 4 kv heads (gqa 6), output gate on
rope       theta 10000000  partial 0.25 -> rotate 64 of 256 dims  mrope interleaved [11,11,10]
delta-net  48 v heads x 128, 16 k heads x 128 (gqa 3), conv k=4 over 10240 ch
quant      MLX affine 4-bit, group 64

weights    14.09 GB text + 0.86 GB vision = 14.95 GB
kv cache   64 KB/token (2.00 GB at 32K)
ssm state  147 MB, constant in context length
```

## Run

```
qwasar -p "..."                  # generate
qwasar -p "..." --show-think     # include the reasoning block
qwasar -p "..." --no-think       # skip reasoning entirely
qwasar -s "..." -p "..."         # with a system message
qwasar -p "..." --effort low     # xhigh (default) | medium | low
```

**Reasoning is on by default for this model**, and at the default `xhigh`
effort it thinks at length — often past a few hundred tokens before the answer
starts. The reasoning block is consumed but not printed unless you ask. Two
consequences worth knowing on your first run: `-n` defaults to 512 for a reason,
and `--effort low` is usually what you want for short factual questions and for
tool work.

If a turn stops before it has finished, it says so and names the budget it hit.
That matters more than it sounds: reasoning tokens count against `-n` but are
not printed, so a turn can spend almost all of its budget thinking and then be
cut off after a few dozen visible tokens.

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

With no task it opens a REPL instead:

```
$ qwasar-agent -C ~/src/project
qwasar-agent. /help for commands, /quit to leave.

> how many lines in words.txt?
  bash command=wc -l words.txt
`words.txt` has **3 lines**.
```

Commands: `/help`, `/new`, `/effort`, `/think`, `/yes`, `/ctx`, `/save`,
`/quit`. Six tools: `read`, `write`, `edit`, `list`, `grep`, `bash`.

The prompt is pinned to the bottom of the terminal with a status footer under
it, and stays live while the model writes: **you can type, edit and send the
next message mid-turn**, and it runs when the current one finishes. Ctrl-C
interrupts. There is no alternate screen, so the transcript scrolls, copies and
searches like any other command output.

**Reading runs unattended. Writing files and running commands ask first**,
unless you pass `--yes`. Tool arguments never reach a shell except for `bash`
itself — `list` and `grep` pass argv to `execvp` directly. A declined action is
reported back to the model as a tool result, so it can choose something else
rather than the turn failing.

If `AGENT.md` exists in the working directory it is added to the system prompt
as project guidance.

`edit` is conventional line-anchored search and replace: the quoted text must
match a run of **whole lines exactly once**, or the edit is refused and nothing
changes. There are no anchor markers and no fuzzy matching. Quoting a bare `}`
fails outright rather than landing in an arbitrary function.

Prompt processing is checkpointed to `~/.cache/qwasar/kv`, so the system prefix
is prefilled once and restored on later runs. `--no-cache` turns that off.

---

# The model

Qwen3.8 27B is not a plain transformer, and most of what is interesting about
implementing it follows from that.

**64 layers in a hybrid schedule.** Every fourth layer is full attention; the
other 48 are **Gated DeltaNet**, a recurrent linear-attention layer. So 16
layers keep a KV cache and 48 keep a fixed-size recurrent state.

**The recurrent half is a delta rule.** Per value head it carries a
`[128, 128]` state matrix, and per token: decay it, read what it already
predicts for the incoming value, write back only the residual scaled by a
learned gate, then read out with the query. The state is fp32 and updated in
place.

**Attention is output-gated.** `q_proj` is twice as wide as you would expect;
half of it is a gate, and the attention result is multiplied by its sigmoid
before `o_proj`. Q and K carry per-head RMS norms.

**RoPE is partial and multimodal.** `head_dim` is 256 but only the first 64 dims
rotate, and each of the 32 frequencies draws its position from one of three axes
interleaved so the section counts land on `[11, 11, 10]`. For text all three
axes agree and it collapses to ordinary partial RoPE.

Two practical consequences fall out of the recurrent half:

**Long context is unusually cheap.** Only a quarter of the layers pay per-token
KV, so the cache is 64 KB/token — 2 GB at 32K, where a comparable
all-attention model would be four times that. The recurrent state is 147 MB no
matter how long the conversation runs.

**A session is append-only.** A KV cache can be truncated to any prefix; a
recurrent state keeps no per-position history and cannot be rewound. Extending
a session is cheap, rewinding means re-evaluating. This is visible in the API,
it is why the disk cache can only reuse strict prefixes, and it is why the agent
feeds back the tokens it generated rather than re-rendering the conversation.

---

# Benchmarks

Early numbers, one machine. Everything below was measured on:

> **MacBook Air, Apple M4** — 10 CPU cores, 10 GPU cores, 32 GB unified memory,
> macOS 26.5.1. Metal reports a 25.0 GB working set and an 18.7 GB maximum
> buffer. Model is `Qwen3.8-27B-MLX-4bit`, 14.95 GB of weights.

**This is a fanless machine**, so it is fair to ask how much of this survives
sustained load. Measured, over six consecutive matmul benchmark runs — a few
minutes of near-continuous GPU work — throughput went from 2243.9 to 2212.4
GFLOP/s. That is a **1.4% drift**, so the figures below are close to sustained
for runs of this length rather than a cold-start best case.

Longer than a few minutes is untested. A cooled chassis would presumably hold
the numbers indefinitely, and might start higher.

## Throughput

Prefill in 256-token chunks; decode measured over 24 greedy tokens at the stated
depth.

| Context | Prefill | Decode |
| ---: | ---: | ---: |
| 512 | 42.6 t/s | 5.71 t/s |
| 2048 | 41.1 t/s | 5.53 t/s |
| 4096 | 36.4 t/s | 5.21 t/s |

**Serial decode is at the memory bandwidth roof and will not go much faster on
this machine.** This is a dense 27B: every decoded token reads every weight. At
~120 GB/s, 14.95 GB per token is an 8.0 t/s ceiling by arithmetic, not by
implementation quality. Kernel work here is about reaching 6 rather than sitting
at 2.

**Speculative decoding is the way past that roof, and it works.** The model
ships a one-layer MTP draft head, published separately because merging it breaks
Python loaders; `./download_model.sh mtp-head` fetches it.

```
qwasar --mtp ./qwasar-mtp --mtp-depth 3 --spec -p "..."
qwasar-agent --mtp ./qwasar-mtp        # on by default once the head is given
```

Measured over 200 tokens of prose, alternating with a serial control so the two
share thermal state: **1.4x on prose and 2.1x on predictable text**, cooler
being better. The spread is not noise. Speculation trades memory traffic for
arithmetic — one pass over the weights, four rows of everything else — and
throttling takes arithmetic away first, so the verify costs 1.45 decode steps on
a cool machine and 1.70 on a warm one while serial decode barely moves. The head proposes and the target disposes, so this cannot change what
the model writes, and `tests/test_verify` holds that line exactly: the emitted
sequence must equal greedy decoding token for token, at every depth, including
on a prompt where two thirds of the rounds are rewound.

**The depth adapts.** How many tokens to draft is chosen per round from the
acceptance the session has actually seen and a measured price per depth, capped
by the target's own top-2 logit gap at the boundary — a near-tie is exactly when
a draft is about to be wrong. It settles around 2.2 on prose and 2.9 on
predictable text, and it will choose *not* to draft at all on a stretch the head
keeps getting wrong, because turning drafting on costs a third of a decode step
before it returns anything.

That beats any constant: fixed depth 2 gives up 13% on predictable text, fixed
depth 3 gives up 2% on prose. Pass `--mtp-depth <n>` to override, or `0` to
turn drafting off.

Depth 4 is a hard ceiling regardless. A fourth draft makes the verify five rows
wide, which needs two batched-matvec blocks and costs twice as much for one more
token.

Decode holding up across context — 5.71 to 5.21 from 512 to 4096 — is the
hybrid schedule earning its keep. Only 16 layers grow with position.

## Where prefill went

Prefill started at 7 t/s and took three rounds of work to reach 42.6.

| | `qmm` | prefill |
| --- | ---: | ---: |
| matvec, one dispatch per token | — | 7.0 t/s |
| tiled matmul, registers | 1.30 TFLOP/s | 26 t/s |
| inner product on the 8×8 matrix units | 1.83 TFLOP/s | 35 t/s |
| half operand tiles, A stored M-major | **2.25 TFLOP/s** | **42.6 t/s** |

For scale: a pure-FMA kernel measures this machine's fp32 peak at **3.33
TFLOP/s**, and **MLX's own quantised matmul sustains 2.73–2.82 TFLOP/s** on
identical shapes. qwasar is at roughly 80% of MLX, which is the honest target —
not the hardware peak, and definitely not the 15.6 TFLOP/s a naive synthetic
benchmark first reported before the loop-invariant product was noticed being
hoisted.

## Startup and the disk cache

| | |
| --- | ---: |
| engine load | 6–9 s |
| first forward pass | 4.8 s |
| agent cold start, empty cache | 46.7 s |
| agent start, system prefix cached | **21.1 s** |
| restoring 874 tokens from a checkpoint | **0.02 s** |

Load time is dominated by a 4.99 GB copy: one safetensors shard in this
checkpoint starts its data section at an odd byte, so it cannot be mapped and
read as packed 32-bit words. The other two shards are mapped zero-copy. The
first forward pass then faults in ~10 GB of mapped weights, which the CLI pays
during load so its reported prefill rate is the steady-state one.

Checkpoints are large: 214 MB for an 874-token prefix, of which only 56 MB is
KV. The rest is the recurrent state, which is the same size for any prefix
length. That single fact drives the cache design — few large entries, a
256-token minimum, and whole conversations saved only on `/save`.

## Memory

| | |
| --- | ---: |
| language model weights, 4-bit | 14.09 GB |
| vision tower, bf16 (loaded, not yet used) | 0.86 GB |
| KV cache | 64 KB/token — 2 GB at 32K |
| recurrent state | 147 MB, constant |

---

# Correctness

Every Metal kernel has a **scalar fp32 CPU twin** and a test that runs both
against **real weights from the model**. Synthetic weights would not catch a
misread of the quantisation layout, which is the failure this most needs to
catch.

On top of that, `tests/test_forward.c` replays golden activations captured from
mlx-vlm through the C engine. It requires the **argmax and all five top-5 ranks
to match the reference exactly**, and reports the per-layer drift so a
divergence is located rather than merely detected.

A few properties are pinned directly because an aggregate error would hide them:

* The gated-delta recurrence is **bit-identical** between streaming and batched
  execution. That is the prefill/decode seam, and a mismatch would mean the
  model answers differently depending on how a prompt was chunked.
* Attention provably ignores cache entries past its own position — the test
  poisons 471 slots beyond the query and requires the output to be unchanged.
* A session restored from disk produces **bit-identical** logits for its next
  token. Anything else would mean part of the recurrent state did not survive a
  restart.
* The tokenizer reproduces the reference's ids on 24 cases and all 8 chat
  template renderings exactly.

**On tolerances.** The reference runs bf16 activations, and disagrees with
*itself* by 7.4e-2 relative on logits when only the accumulation order changes.
So logit L2 is a weak signal here and is not what the tests lean on; argmax,
top-5 ordering, and a smooth per-layer drift curve are. This is written down in
`PLAN.md` next to the numbers, because it is the kind of thing that otherwise
gets "fixed" into a false failure later.

```
make test
```

```
ok: json
ok: toolcall
ok: tokenizer
ok: qmv
ok: ops
ok: gdn
ok: attn
ok: kvstore
ok: forward
```

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

---

# What is not implemented

Stated plainly, because a README that only lists what works is not much use:

* **Vision.** The tower is parsed, sized, and reported by `--info`, and its
  weights are loaded, but it is not executed. Image input does nothing yet. This
  is the largest remaining piece.
* **Sampling in the CLI.** `qwasar-server` samples with temperature, top-k,
  top-p and min-p; `qwasar` and `qwasar-agent` are still greedy.
* **`/v1/responses` and `/v1/completions`**, which ds4-server has. They return
  501. So do concurrent requests: qwasar serves one at a time.
* **NFC normalisation** in the tokenizer. A no-op for ASCII and already-normalised
  text; decomposed input would tokenize differently from the reference.
* **Todo tracking and a `glob` tool** in the agent.
* **Multi-GPU, CUDA, distributed inference.** Not planned. This targets one Mac.

## Known rough edges

* One shard of the standard checkpoint costs a 4.99 GB copy at load because of
  where its data section starts. A disk-cached aligned repack would remove it
  and is not written yet.
* The prefill progress bar advances once per 256-token chunk, so a short prompt
  gets four frames. Finer granularity means splitting the command buffer
  mid-chunk, which costs throughput.
* `qmm` is at 80% of MLX's throughput. Threadgroup memory bounds occupancy, but
  every way of freeing it so far gives up coalesced output writes and loses more
  than it gains.

---

# License

MIT — see [LICENSE](LICENSE).

Everything qwasar builds on is permissively licensed and MIT-compatible: MLX and
mlx-vlm are MIT, ds4 is MIT, and linenoise is BSD-2-Clause.
[THIRD-PARTY.md](THIRD-PARTY.md) records each of them, what its terms require,
and — for the ones that require nothing — why the acknowledgement is there
anyway.

Model weights are not part of this repository and carry their own licence from
the Qwen team.

---

# Notes on the code

A few conventions, inherited from ds4 and worth knowing before reading:

* **No Python** in the build or the runtime. `tools/` contains dev-only fixture
  generators that are never required to build, test, or run.
* **No C++.**
* Correctness before speed. A faster path with unexplained drift does not ship.
* Loading is mmap-backed. A 15 GB model is not eagerly copied.
* Comments explain **why** — a shape, an ordering, a cache boundary, a memory
  policy — and live beside the code rather than in a separate document.
* `qwasar.h` stays narrow. CLI and agent code never learn what a tensor is.
