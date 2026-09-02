# qwasar × Qwen3.8-Flash-Next (`qwen4_exp`) — the port, planned

*Written 2026-08-31. Nothing here is started. Facts below marked "measured"
came from the HF repo and config; facts marked "verify" are Phase 0's job —
this plan is honest about which is which.*

`Qwen/Qwen3.8-Flash-Next` is a 125B-total / **6B-activated** MoE with a 51B
n-gram embedding table beside it (131 shards, ~360 GB BF16 — the arithmetic
says ~180B parameters on disk, which is 125 + 51 with rounding). It keeps the
two things this engine's hardest code implements — the 3:1 hybrid schedule
(`full_attention_interval: 4`) and **Gated DeltaNet at exactly our geometry**
(16 key heads × 128, 48 value heads × 128, conv 4) — and swaps everything
around them: MoE MLPs for dense, "Qwen Sparse Attention" for full attention,
an engram table at the embedding, 48 layers at hidden 2560, the same 248320
vocab, 262K native context, and an MTP layer trained in.

## 0. The identity question, answered first

PLAN.md opens with "a single-model inference engine … abstractions shaped
around inference rather than around supporting many models." This port does
not repeal that rule; it rewords it: **a single-family engine.** `qwen3_5`
and `qwen4_exp` share the tokenizer, the vocab, the DeltaNet layer, the
hybrid schedule, the partial-RoPE convention, the MTP idea, and the session
model. One code path, hparams-driven (they already are — `is_linear_attn`,
`linear_num_key_heads` and friends live in the config struct today, not in
constants). What we refuse is generality beyond the family: no layer
registry, no plugin architecture, no third model until one earns it.

The 27B stays supported and gated: it is the only model the M4 Air can run,
and its goldens are the regression net under every shared kernel we touch.

## 1. Hardware, and who does what

Two machines, two roles, both load-bearing:

| | M4 Air 32 GB (~120 GB/s) | M5 Max 128 GB (~550 GB/s class) |
|---|---|---|
| 27B dense (today) | runs, 6–8 t/s | runs, faster, uninteresting |
| Flash-Next weights @4-bit (~70 GB transformer) | **does not fit, any quant** | resident, ~55 GB headroom |
| engram table @4-bit (~29 GB) | n/a | mmap, cold — lookups touch a few rows/token |
| KV @ 262K (12 QSA layers × 2 KV heads × 256, fp16 = **24 KB/token**) | n/a | 6.4 GB at full native context |
| DeltaNet state (36 layers × 48×128×128 fp32) | n/a | ~113 MB, constant |
| role | **CPU reference twins** on synthetic shapes; 27B regression | the model, the goldens, every measurement |

The decode ceiling, stated the way §2 of PLAN.md states it so nobody is
surprised later: ~6B active × ~0.56 B/param (4-bit affine, group 64) ≈
**3.4 GB read per token**, which at ~550 GB/s is ~160 t/s by arithmetic.
We will not see that: expert gather is scattered reads, eleven 640-wide
matvecs per layer sit far from the matmul roof, and the router and engram
add latency floors per layer. **30–60 t/s is the honest target band**, and
even the bottom of it is 5× the 27B on the Air. Prefill gains twice: ~4.5×
fewer FLOPs per token (6B vs 27B active) on a much larger GPU.

Logistics before any code: ~360 GB download on the Max, quantised locally to
MLX affine 4-bit (mlx_lm convert if it supports the arch by then — a
`tools/` job, dev-only Python as the rules allow — else our own quantiser
grows an export mode). Budget ~500 GB of free disk for the conversion window.

## 2. What transfers, what is new

**Transfers unchanged or config-only:** Gated DeltaNet kernel and its fp32
state machinery (identical head geometry; only `hidden_size` feeding the
projections changes) · hybrid layer schedule · partial RoPE (0.25 = 64 of
256, same convention; `rope_theta` 1e7, same) · tokenizer (same 248320
vocab; **verify** merges.txt byte-identical) · sampling and the
rejection-sampling verify · sessions, append-only, prefix checkpoints ·
mmap loading, shape validation at load · the agent, the server, Crucible.

**New, in effort order:**

1. **MoE** — the rock. Router (top-10 of 512 + 1 shared expert, expert
   intermediate 640), quantised expert banks, Metal gather-matmul. Decode is
   the easy half (one token × eleven small FFNs); prefill dispatch is where
   the engineering lives (hundreds of tokens fanning across hundreds of
   experts — expert-major loop over a token-sorted index beats
   token-major gather; measure, don't assume). Router details (softmax
   before or after top-k, `norm_topk_prob`, shared-expert gating) — verify
   in Phase 0 from the `transformers` modeling code.
2. **Qwen Sparse Attention** — block selection under a 2048-token budget at
   micro-block granularity, MQA 24 Q / 2 KV × 256. Written down now so the
   temptation is pre-refused: **running these layers dense is not a
   correctness shortcut.** A model trained through sparse selection produces
   different outputs under full attention; the selection kernel is the
   layer, not an optimisation of it. Exact scoring/selection — verify.
3. **The engram table** — bigram/trigram hash → gather a few rows from a
   20M-entry × 51B-param table → gated-residual merge (card: 4 branches,
   bottleneck 320 — verify mechanics). Mechanically small, and the table
   never needs residency (cold mmap; a few rows per token off NVMe is
   microseconds). The risk is not performance, it is the **hash**: get one
   detail of the n-gram hashing wrong and the model degrades silently with
   nothing to catch it but vectors. Pin it token-by-token before anything
   downstream.
4. **Plumbing** — `qwen4_exp` config keys and tensor-name map over 131
   shards; per-layer-type graph assembly; the loader's shape validation
   extended to expert banks ("where a mis-parsed config shows up first"
   stays true here). New chat template (8.95 KB jinja) → new golden set.
5. **Vision** — it is a VLM again. **Deferred exactly as Milestone 1
   deferred it**: text-only first, MRoPE machinery already exists, the
   tower waits until the language model holds goldens.

## 3. Phases, each with a gate

**Phase 0 — recon, oracle, weights.** Read the `transformers`
`modeling_qwen4_exp` source end to end and write the answers into this file:
router math · QSA selection · engram hash and merge · gated-residual shape ·
whether the MTP layer's weights are in the main shards · minimum
transformers version · whether mlx-lm has the arch (it is the preferred
oracle; transformers-on-CPU is the fallback oracle, slow but sufficient for
short vectors). Download and quantise on the Max. *Gate: every "verify" in
this document replaced by a fact with a source, and a quantised model on
disk.*

**Phase 1 — loader.** Config parse, tensor inventory, layer schedule,
mmap policy split (resident transformer / cold engram), shape validation.
*Gate: the loader walks all 131-shard-worth of quantised tensors, validates
every dimension against config arithmetic, evaluates nothing.*

**Phase 2 — CPU reference twins** (M4 Air work, synthetic shapes): MoE
block, QSA (selection included), engram lookup + merge, plus captured-
activation comparisons for the reused DeltaNet at the new hidden size.
Per-op twins only — PLAN.md §7's judgment against a whole-model CPU path
goes double at 125B. *Gate: each twin matches reference tensors captured
from the oracle on real shapes, to stated tolerance, checked in.*

**Phase 3 — Metal MoE.** Decode path first, then batched prefill dispatch.
Measure read volume per token against the 3.4 GB arithmetic — the gap IS
the expert-gather inefficiency, and it gets a number, not an adjective.
*Gate: layer output matches the twin; decode read volume within 1.5× of
active bytes.*

**Phase 4 — QSA on Metal.** Prefill (scores → top-k blocks → attention over
the budget) and decode. *Gate: matches the twin, including selection
indices on a fixed input.*

**Phase 5 — engram in C.** The hash, the cold-mmap gathers, the merge.
Measure page-fault latency under a real decode; add a small hot-row cache
only if the measurement demands it. *Gate: token-for-token merge output
matches the oracle on a corpus that exercises the hash edges (unicode,
byte-fallback-adjacent tokens, sequence starts).*

**Phase 6 — end to end.** Golden vectors from the new chat template; greedy
token-for-token match against the oracle for N tokens at temperature 0;
then the M5 Max sweep in PLAN.md §5's format (prefill/decode by context,
thermal noted). Context default: KV is 24 KB/token, so 128K fits in 3.2 GB
— the 27B's 32K conservatism does not carry over; pick from the measured
memory table. *Gate: greedy match, plus a measured table in this file.*

**Phase 7 — speculation.** Verify the MTP layer's weights exist, wire the
existing draft/verify, re-measure the depth table from scratch — at 6B
active the serial rate is already high and drafting may no longer earn its
rounds. If it does not, it ships disabled with the measurement written
down; the 27B keeps it regardless. *Gate: a table like §5's, and a
keep/disable decision citing it.*

**Phase 8 — Crucible.** `MemoryProfile.derive` learns the Max and the new
cost model (weights resident, KV cheap, contexts huge); model choice per
project; goldens per model_type; the session/parking/checkpoint machinery
is untouched by construction (recurrent state + KV both still exist, both
still append-only). The economics flip — prefill stops dominating turns —
is the payoff Crucible was designed to be patient about; nothing in its
spec assumed 6 t/s, it only endured it.

Order is dependency order, but Phases 2–5 interleave freely: twins are Air
work, kernels are Max work, and each twin unblocks its kernel.

## 4. Risks, named

- **The engram hash** (silent quality rot; caught only by Phase 5's gate —
  which is why that gate is token-for-token, not perplexity-shaped).
- **QSA fidelity** (the dense-fallback temptation; pre-refused above).
- **4-bit experts** — 640-wide experts quantise into few groups each;
  quality damage is plausible. If goldens drift, try 6-bit experts +
  4-bit elsewhere before concluding anything (mixed precision per tensor
  class, not a new format).
- **Oracle availability** — everything above assumes a trustworthy
  reference implementation; if `transformers` support is younger than it
  looks, Phase 0 gets longer, not skipped.
- **The ceiling estimate** — ~550 GB/s for the M5 Max is a class guess
  until measured on the machine; the first Phase 6 number to record is
  actual achievable bandwidth (the 27B's §2 did exactly this for the Air).
- **Vision scope creep** — deferred means deferred; the tower has its own
  milestone when the text model holds.
