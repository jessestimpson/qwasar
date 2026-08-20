# qwasar

A single-model inference engine for **Qwen3.8 27B** on macOS Metal, written in
C with Objective-C only where Metal requires it.

Modelled on [ds4](https://github.com/antirez/ds4): no Python in the build or the
runtime, one `make` away from a binary, abstractions shaped around *inference*
rather than around supporting many architectures.

```
$ qwasar -m ~/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-4bit -n 120 \
      --tokens 248045,846,198,657,2250,9944,4947,13,248046,198,248045,74455,198,248068,198

The user asks me to name three prime numbers. Prime numbers are numbers greater
than 1 that have no positive divisors other than 1 and themselves...
</think>

Here are three prime numbers:

1. **2** (the only even prime)
2. **7**
3. **13**

prefill 15 tokens | decode 109 tokens (5.83 tok/s)
```

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
qwasar -m <model-dir> --info                  # parsed architecture + weight inventory
qwasar -m <model-dir> --tokens <ids> -n 120   # generate
```

Weights are the MLX 4-bit conversion, e.g.
`lmstudio-community/Qwen3.8-27B-MLX-4bit`. Nothing needs converting; qwasar reads
the safetensors shards directly and mmaps them.

BPE *encoding* is not wired up yet, so prompts are currently given as token ids.
Decoding works, so generated text streams normally.

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
qwasar_tokenizer.c  byte-level BPE (decoding)
qwasar_metal.m      Metal runtime: device, library, pipelines, dispatch
qwasar_cpu.c        scalar fp32 reference twins for every kernel
metal/*.metal       kernels
tests/              unit + golden-vector regression
tools/              build helper; dev-only golden-vector generator (not built)
```

`PLAN.md` carries the design, the measurements, and the roadmap.

## Status

Text generation works and matches the reference implementation's argmax and
top-5 ordering exactly. Decode runs at 5.8 tok/s on an M4, which is the
bandwidth roof for a dense 27B at ~120 GB/s — see `PLAN.md` §2.

Next: BPE encoding and the chat template, then a tiled `qmm` for prefill, then
the agent, then vision.
