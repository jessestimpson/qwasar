# qwasar

A single-model inference engine for **Qwen3.8 27B** on macOS Metal, written in
C with Objective-C only where Metal requires it.

Modelled on [ds4](https://github.com/antirez/ds4): no Python in the build or the
runtime, one `make` away from a binary, abstractions shaped around *inference*
rather than around supporting many architectures.

```
$ qwasar -m ~/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-4bit \
      -p "Name three prime numbers, with one sentence on why each is prime."

2 is prime because its only positive divisors are 1 and itself.
3 is prime because it cannot be divided evenly by any whole number other than 1 and 3.
5 is prime because its only positive divisors are 1 and 5.

prefill 66 tokens | decode 157 tokens (5.82 tok/s)
```

## qwasar-agent

An agentic loop on the same engine, with six tools: `read`, `write`, `edit`,
`list`, `grep`, `bash`.

```
$ qwasar-agent -m <model-dir> -C ~/src/proj \
      "maxval in stats.c returns the wrong answer. Fix it, rebuild, and check."

  read path=stats.c
  edit path=stats.c old=        if (v[i] < best) best = v[i]; new=        if (v[i] > b...
  bash command=cc -o stats stats.c && ./stats

Fixed. The comparison `v[i] < best` was tracking the minimum instead of the
maximum. Rebuilt and ran: mean=2.80 max=5.00
```

With no task it opens a REPL instead, with history and `/help`, `/new`,
`/effort`, `/think`, `/yes`, `/ctx`, `/quit`. Prompt processing shows a progress
bar, since it is the one part of a turn with nothing to look at:

```
prefill [▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶ 24 tok/s ··] 512/886 58%
```

Reading runs unattended; writing files and running commands ask first, unless
`--yes`. If `AGENT.md` exists in the working directory it is added to the system
prompt as project guidance.

`edit` uses conventional line-anchored replacement: the quoted text must match a
run of whole lines **exactly once**, or the edit is refused and nothing changes.
There are no anchor markers and no fuzzy matching — an edit either does what it
says or tells the model why it could not.

## Build

```
make
```

That is the whole story: `cc`, Foundation, and Metal. Metal kernels under
`metal/` are embedded into the binary as source (`tools/bin2c`) and compiled at
startup, so **the Metal Toolchain component is not required** — the binary is
self-contained and relocatable.

```
make test          # unit and golden-vector suites (needs a model)
make check-metal   # optional offline kernel lint (needs the Metal Toolchain)
```

## Run

```
qwasar -m <model-dir> --info                     # architecture + weight inventory
qwasar -m <model-dir> -p "..."                   # generate
qwasar -m <model-dir> -p "..." --show-think      # include the reasoning block
qwasar -m <model-dir> -p "..." --no-think        # skip reasoning entirely
qwasar -m <model-dir> -s "..." -p "..."          # with a system message
qwasar -m <model-dir> -p "..." --effort low      # xhigh (default) | medium | low
```

Reasoning is on by default for this model, and the reasoning block is consumed
but not printed unless you ask for it.

Weights are the MLX 4-bit conversion, e.g.
`lmstudio-community/Qwen3.8-27B-MLX-4bit`. Nothing needs converting; qwasar reads
the safetensors shards directly and mmaps them.

## What this model is

Not a plain transformer. 64 layers in a hybrid schedule — **48 Gated DeltaNet
(recurrent) layers and 16 full-attention layers**, one full-attention every
fourth. Attention is output-gated with per-head Q/K norms, `head_dim` 256, and
only the first 64 dims rotated (partial multimodal RoPE). Dense SwiGLU MLP, no
experts. There is also a vision tower, not yet implemented.

One consequence is visible in the API: the 48 recurrent layers carry state with
no per-position history, so **a session is append-only**. A KV cache can be
truncated to any prefix; a recurrent state cannot. Extending a session is cheap,
rewinding it means re-evaluating.

## Layout

```
qwasar.h            public engine boundary -- no tensor internals escape it
qwasar.c            config, safetensors mmap, weight table
qwasar_graph.c      session state and the forward pass
qwasar_tokenizer.c  byte-level BPE, plus the ChatML template
qwasar_toolcall.c   tool-call parsing and the line-anchored edit matcher
qwasar_agent.c      the agent loop, its tools, and the REPL
linenoise.c         vendored line editing (BSD-2, antirez)
qwasar_unicode.inc  generated codepoint tables for the pre-tokenizer
qwasar_metal.m      Metal runtime: device, library, pipelines, dispatch
qwasar_cpu.c        scalar fp32 reference twins for every kernel
metal/*.metal       kernels
tests/              unit + golden-vector regression
tools/              build helper; dev-only golden-vector generator (not built)
```

`PLAN.md` carries the design, the measurements, and the roadmap.

## Correctness

Every kernel has a scalar fp32 CPU twin and a test that runs both against real
model weights — synthetic weights would not catch a misread of the quantisation
layout. On top of that:

- the full forward pass matches the mlx-vlm reference's **argmax and all five
  top-5 ranks exactly**;
- the tokenizer reproduces the reference's ids on 24 cases and the chat template
  on all 6, exactly;
- the gated-delta recurrence is **bit-identical** between streaming and batched
  execution, which is the prefill/decode seam;
- attention provably ignores cache entries past its own position.

One deliberate divergence: `qwasar_encode` never emits control tokens, however
the input spells them. The reference splits input on added tokens, which would
let message content forge a `<|im_start|>` role boundary. The template emits
control tokens by id instead.

## Status

Milestone 1 is complete: text in, text out, verified against the reference.

Decode runs at 5.8 tok/s on an M4 — the memory-bandwidth roof for a dense 27B at
~120 GB/s (`PLAN.md` §2). Prefill is currently no faster, because the matvec
re-reads the weights once per token; a tiled `qmm` is the next significant win.

Prefill runs at 35 tok/s after two rounds of work on the quantised matmul: a
tiled version, then moving its inner product onto the GPU's 8x8 matrix units.
That is 55% of this machine's measured fp32 peak.

Next: vision, and half-precision operand tiles in the matmul.
