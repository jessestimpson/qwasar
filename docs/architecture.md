# How qwasar is built

## The model, briefly

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
token. Its full-attention layers are sparse: past a 2,048-token budget each
query attends to the blocks an indexer selects. Both consequences above hold
for it too.

Qwen3.5 9B is the 27B's layout at a smaller width: 32 layers, every fourth
full attention, hidden size 4096. Its cache is 32 KB/token (1 GB at 32K) and
its recurrent state a constant 49 MB.

## Correctness

* Every Metal kernel has a **scalar fp32 CPU twin**, tested against **real
  model weights** (synthetic weights would miss quantisation-layout misreads).
* `tests/test_forward.c` replays golden activations from mlx-vlm and requires
  the argmax and all five top-5 ranks to match exactly, reporting per-layer
  drift so a divergence is located, not just detected.
* `tests/test_flashnext_real.c` holds Flash-Next to mlx-vlm on the same 4-bit
  weights, and with `QWASAR_FLASHNEXT_CPU` to the engine's own fp32 CPU
  reference token by token.
* The gated-delta recurrence is **bit-identical** between streaming and batched
  execution — the prefill/decode seam.
* Attention provably ignores cache slots past its position (the test poisons
  471 of them and requires unchanged output).
* A restored session produces **bit-identical** next-token logits.
* The tokenizer matches the reference on 24 cases and all 8 chat template
  renderings.
* The vision tower (27 blocks of bf16) is validated against mlx-vlm at rel L2
  5.5e-3 on identical patches — closer to the fp32 reference than the
  reference's own bf16 path.

Logit L2 is deliberately *not* what the tests lean on: the bf16 reference
disagrees with itself by 7.4e-2 relative when only accumulation order changes.
That reasoning is written down in `PLAN.md` so it doesn't get "fixed" into a
false failure later.

Greedy outputs from qwasar and mlx-vlm agree for a while and then part. Two
things legitimately separate them: mlx-vlm computes in BF16 and qwasar in
fp32, and on Flash-Next, when the 10th and 11th of 512 experts score within
~1e-7 of each other, summation order decides which one a token routes to.
Where the two were compared position by position, the Metal path matched the
fp32 CPU reference to ~1e-4 of the logit scale until such a tie.

## Layout

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
qwasar_sample.c     temperature, top-k, top-p, min-p and top-n-sigma sampling
qwasar_cli.c        ./qwasar
qwasar_server.c     ./qwasar-server, and its OpenAI and Anthropic endpoints
qwasar_api.c        the Session API's endpoints (API.md)
qwasar_sessions.c   the session store, the scheduler, and generation
qwasar_profile.c    context size and live sessions, from the model and machine
qwasar_http.c       the HTTP/1.1 core the server's fronts share
qwasar_agent.c      ./qwasar-agent: a Session API client, its tools, the REPL
metal/*.metal       kernels
metal4/*.metal      tensor-op kernels (Metal 4), built on first use
tests/              unit, golden-vector and API regression
tools/              build helper, benchmarks; dev-only fixture generators
linenoise.c         vendored line editing (BSD-2, antirez)
app/                Qwasar.app: the menu bar item and the coding agent's window
docs/               the server, the terminal agent, benchmarks, this file
```

The plans carry the design, the measurements, and — deliberately — the
things that were tried and did not work, so they are not retried: `PLAN.md`
for the engine and the 27B, `PLAN-flash-next.md` for Flash-Next,
`PLAN-qwasar.md` for the Session API and the app's move onto it, and
[app/spec/](../app/spec/README.md) for the app. The commit messages carry the
measurements behind each optimisation.

Conventions, inherited from ds4: no Python, no C++, correctness before speed,
mmap-backed loading, comments that explain *why*, and a narrow `qwasar.h` — the
CLI and agent never learn what a tensor is.

You do not need the Metal Toolchain to build: kernels are embedded as source
and compiled at startup. If you have it, `make check-metal` uses it as a fast
offline lint.
