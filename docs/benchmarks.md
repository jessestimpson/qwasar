# Benchmarks, in detail

The README carries the tables; this file carries how they were measured and
what is behind them. The commit messages carry the step-by-step history of
each optimisation.

## Measuring on a Mac

Laptops throttle. On the M5 Max MacBook Pro, back-to-back runs of identical
code drift by ~10% and more, in prefill and decode alike; one uncooled A/B
once showed a 2.4×-faster attention kernel as a regression. So every number
here starts from a die cooled to 42–45 C (`tools/bench/tempmon` reads it
without root; `tools/bench/sweep.py` waits for it), and A/B comparisons
alternate the two sides. `QWASAR_PROFILE=1` breaks the command buffer at
every mark, so it runs below the power cap: use it to find where time goes,
not to size an end-to-end win.

The tools, all under `tools/bench/` (`make bench` builds them):

| Tool | Measures |
|---|---|
| `sweep.py` | unattended prefill/decode sweeps of `qwbench`, with cooldowns and temperatures |
| `qwbench` | one point: load, prefill exactly N tokens (every chunk timed), decode D |
| `ttftbench`, `mlx_ttft.py` | time to first token and decode, qwasar and mlx-vlm, same token ids |
| `fabench`, `attnbench` | prefill and decode attention alone, by depth |
| `gmmbench`, `qsabench` | Flash-Next's grouped experts and decode sparse attention alone |
| `tempmon` | SoC temperatures |

## M5 Max

**MacBook Pro, Apple M5 Max**, 128 GB, 2026-10-12: `sweep.py`, each point a
fresh process that prefills exactly N tokens of this repository's sources,
then decodes 128 greedy tokens at that depth; medians of two runs. The table
is in the README.

The M5's GPU cores carry matrix units that only Metal 4 tensor ops reach, and
on an M5 (GPU family Apple10) the engine takes kernels built for them: the
quantised matmul (`metal4/qmm_tensor.metal`, 35–40 TFLOP/s on the 27B's MLP
shapes against 10–11 for the simdgroup kernel), Flash-Next's grouped experts
and sparse attention, prefill attention with half operands, and a split-key
decode attention that reads the cache at 445–480 GB/s from 64K keys. Against
the same machine before them, in back-to-back pairs: prefill at 4K 247 → 639
t/s for the 27B, 849 → 2,017 for the 9B, 512 → 1,180 for Flash-Next, and
Flash-Next at 32K 399 → 944. The 27B's decode at 4K, 26 t/s, is at the
bandwidth roof: 14.95 GB of weights a token. Earlier GPUs keep the kernels
tuned for the M4.

The 9B has not yet been measured on a 16 GB Mac, where the memory bandwidth —
and so the decode rate — is several times lower. Against the mlx-lm reference
its per-layer error matches the 27B's (0.5% after the first layer), and
greedy decoding agrees token for token until the first near-tie.

## Flash-Next against mlx-vlm

On the same 4-bit weights, the same machine, 2026-10-13: a coding agent's
first prompt, from a cold session — no checkpoint, no prompt cache, no shared
prefix on either side. Two sizes: 4,872 tokens, what Qwasar.app's agent
sends first (its system prompt and tools are 4,846), and 16,384, a heavier
agent's. Time to first token is the prefill through the first token's
argmax; decode is 256 greedy tokens after it. Both engines warm up
in-process first, both take the same token ids, every run starts from a die
cooled to ~44.5 C; two runs each. The table is in the README.

mlx-vlm `main` is commit `e928101a` with mlx 0.32.3; `d1bd74ed` (mlx 0.32.2)
is the commit `tests/fixtures` were recorded against. mlx-vlm runs as it ships
— BF16 activations, 2,048-token prefill steps — and qwasar in fp32 with
1,024-token chunks. Loading is outside the numbers: ~22 s for qwasar (most of
it copying shards to align them) against ~7 s for mlx-vlm, once per process.
The harnesses are `ttftbench` and `mlx_ttft.py` (run it from an mlx-vlm
install of its own; `tools/venv` is pinned to the fixtures' commit); the ids
were `qwasar.c`, `qwasar_graph.c`, `qwasar_metal.m`, `qwasar_sessions.c`,
`README.md` and `PLAN.md`, concatenated and encoded by the model's tokenizer.

**Why decode differs.** Profiled on one decode token at 4.9K tokens of
context, with mlx-vlm's engram lookup stubbed (the real one, a CPU read of a
30 GB on-disk table that waits on the GPU, adds 1–4 ms): mlx-vlm's step is
~6,980 MLX operations — 662 separate quantised matmuls among them — which
Python builds in 3.5 ms and MLX's runtime encodes in ~20 ms of CPU, ~2.9 µs
each; the token is CPU-bound at ~25 ms. qwasar issues ~1,100 dispatches at
~2.5 µs of CPU each, its CPU is busy 2.7 ms of a 16.7 ms token, and the GPU is
the limit. Prefill is GPU-bound on both, and level.

## Qwen3.8 27B on an M4

All numbers from one machine: **MacBook Air, Apple M4**, 10 CPU / 10 GPU
cores, 32 GB, macOS 26.5.1, fanless. Measured 2026-08-23, after the compact
draft head, the re-measured depth table, and the decode-timer fix (earlier
readmes carried figures whose speculative decode excluded drafting time; these
do not). Prefill in 256-token chunks, decode over 24 greedy tokens at the
stated depth; the table is in the README.

Serial decode is at the memory-bandwidth roof: a dense 27B reads all 14.95 GB
of weights per token, which at ~120 GB/s caps out around 8 t/s by arithmetic.
The 4K row ran directly after two minutes of continuous prefill on a fanless
chassis, so it carries thermal load the short rows do not; the depth cost
itself is small, which is the hybrid schedule earning its keep.

These rows predate the M4 work of October 2026: the quantised matmul on
tensor ops (2.9–3.0 TFLOP/s on the 27B's MLP shapes against 1.7–2.0 for the
simdgroup kernel, and against MLX's 2.73–2.82 measured in `PLAN.md`), the
tiled prefill attention and the split decode attention. The commit messages
carry their measurements; the table has not been re-swept since.

Engine load 6–9 s.

### Speculative decoding

The model ships a one-layer MTP draft head (`./download_model.sh 27b-mtp`,
quantised to 4-bit at load):

```
qwasar -m ./qwasar-27b --mtp ./qwasar-27b-mtp --spec -p "..."
```

**1.5× on prose, sustained.** Three alternating serial/speculative pairs of
200 tokens, so both share thermal state: serial 5.41 / 5.64 / 5.63 t/s,
speculative 8.32 / 8.42 / 8.38 t/s — ratios 1.54, 1.49, 1.49, with 2.58
tokens committed per round at mean depth 2.63 and drafting costing 1.3 s of a
24 s run. The stability is new: before the compact draft head and the
re-measured depth table this faded from 1.65× toward 1.4× as the chassis
warmed, and the 1.65× itself came from a timer that excluded drafting.

Draft depth adapts per round from observed acceptance; `--mtp-depth <n>`
overrides, `0` disables. The target verifies every draft, so output is
guaranteed identical to greedy decoding — `tests/test_verify` pins that
exactly. For sampling callers there is `qwasar_session_verify_sampled`, a
rejection-sampling verify whose output is distributed exactly as serial
sampling, for embedders; the CLI decodes greedily and stays on the exact
verify. The server does not run a draft head yet, so the app and the agent
decode serially. Details — the pruned draft vocabulary, the depth model, the
depth-4 ceiling — are in `PLAN.md`.
