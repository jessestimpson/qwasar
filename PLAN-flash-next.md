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

## Phase 0 — answered (2026-09-02, from source)

Every "verify" above, resolved against `transformers` main
(`modular_qwen4_exp.py`, which composes `qwen3_5`, `qwen3_5_moe` and
`qwen3_next`), the repo's `config.json`, and the safetensors headers of all
131 shards (fetched by HTTP range, no weights downloaded). The M4 Air did all
of this; nothing below needed the model.

**The oracle and the weights.**
- `transformers` supports `qwen4_exp` natively; the checkpoint says
  `transformers_version 5.8.0.dev0`, so the oracle is **main**, not a
  release. Installed in `tools/venv` (dev-only, never required to build).
- **mlx-lm has no `qwen4_exp`** (404 on 2026-09-02). There will be no
  MLX-sanitised checkpoint to lean on: `tools/flashnext_convert.py` is ours,
  HF BF16 in, MLX-affine 4-bit (group 64) out, in the engine's own tensor
  naming, so the existing quantised kernels and loader conventions carry.
- 180.0B parameters on disk in BF16 (359,999,963,128 bytes): **125.1B text
  model** plus the engram table, plus a vision tower and the MTP layer.
  Everything is BF16 except three I64 buffers per PLE layer, which are
  derived from config and need not be loaded.
- **The MTP layer's weights ARE in the main shards** (`mtp.*`): one QSA
  layer with MoE and hyper-connections, `fc_embedding` [2560,2560],
  `fc_hidden` [2560,2560], `pre_fc_norm_embedding` [2560],
  `pre_fc_norm_hidden` [10240], its own `hyper_connection_mixer`.
  **Open**: how 10240 becomes 2560 before `fc_hidden` is not in
  `transformers` (MTP is not part of its forward); Phase 7 needs Qwen's own
  MTP code. Everything else about the head is a normal decoder layer.

**Tokenizer and template.** Vocab (248,044 + 33 added), merges (247,587)
and the chat template are **identical** to the 27B's, byte for byte modulo
JSON formatting and a trailing newline. One real change: the pre-tokenizer
split regex is `[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+` and
` ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*` — combining marks (`\p{M}`) now attach to
letters instead of splitting off. A config-driven switch in
`qwasar_tokenizer.c`, and a golden case with combining marks.
`generation_config` is the 27B's exactly (eos 248046/248044, temp 1.0,
top-k 20, top-p 0.95).

**The residual stream is four streams** (`hc_count: 4`). The token
embedding is repeated ×4 into a 10240-wide residual; there is **no
`input_layernorm`, no `post_attention_layernorm`, and no final `norm`** —
none exist in the checkpoint. Every sublayer is wrapped by a
`GatedResidual` (`attn_hyper_connection`, `mlp_hyper_connection`):

    n   = grouped_rmsnorm(h4)                  # 4 groups of 2560, (1+w)
    m   = sigmoid(up(silu(down(n) / 4)))       # 10240→320→10240, sigmoid
    x   = mean over streams of (m ⊙ n)         # the sublayer's 2560 input
    inj = 2·sigmoid(block_inject(n) / 4)       # [4], one weight per stream
    h4 += out ⊗ inj                            # inject the 2560 output

and the model ends with `hyper_connection_mixer` — the same mixer without
`block_inject` — whose 2560 output goes **straight to `lm_head`**.
Per-token cost of the mixers: ~13M MACs per layer, an expert's worth.

**Layer schedule.** 36 `linear_attention` + 12 `qwen_sparse_attention`
(the config spells the latter "full_attention"; the modular code renames
it). PLE lives on **one** layer, zero-indexed 1 (`ple_layer_ids: [2]` is
one-indexed).

**Gated DeltaNet**: geometry identical to the 27B (16 K × 128, 48 V × 128,
conv 4; `in_proj_qkv` [10240,2560], `in_proj_z` [6144,2560], `in_proj_b/a`
[48,2560], `out_proj` [2560,6144], `conv1d` [10240,1,4]). **One change**:
`output_gate_type: "sigmoid"` — the gated norm is `rmsnorm(y)·sigmoid(z)`,
not `·silu(z)`. `mamba_ssm_dtype: float32`, as we already do.

**MoE** (`Qwen3NextSparseMoeBlock`, verbatim):
- router: `probs = softmax(W·x)` in **float over all 512**; top-10;
  `norm_topk_prob: true` → the ten are renormalised to sum 1; cast to the
  activation dtype. `gate.weight` [512,2560].
- experts fused: `gate_up_proj` **[512, 1280, 2560]** (gate is the first
  640 rows of each expert, up the second), `down_proj` **[512, 2560, 640]**;
  `silu(gate)·up`, then down, scaled by the routing weight, summed.
- shared expert: an ordinary 640-wide SwiGLU MLP, scaled by
  `sigmoid(shared_expert_gate · x)` (`shared_expert_gate.weight` [1,2560]);
  `out = routed + gated_shared`.

**Qwen Sparse Attention** = the 27B's gated attention plus an indexer.
- attention: `q_proj` [12288,2560] (per head `[q(256) | gate(256)]`,
  output × `sigmoid(gate)` — exactly the 27B's `attn_output_gate` layout),
  `k_proj`/`v_proj` [512,2560] (2 KV heads), `o_proj` [2560,6144],
  `q_norm`/`k_norm` [256] (+1), partial RoPE 64 of 256, θ=1e7,
  interleaved MRoPE [11,11,10] (text-only collapses, as before).
- indexer: `index_qk_proj` [640,2560] = 4 query heads × 128 + **1** key
  head × 128; `q_layernorm`/`k_layernorm` [128] (+1). Per query: q →
  norm → RoPE on its first 64 dims at the query's position. Keys are cached
  **raw** (pre-norm, pre-RoPE), one 128-vector per token. The visible
  tokens are grouped into blocks of `indexer_compress_ratio = 4`
  consecutive tokens; per block, key = RoPE(norm(mean_fp32(4 raw keys)))
  at the **block's first position**; score = Σ_heads relu(q_h·k) / √128;
  the top `2048/4 = 512` blocks are selected, and the **tail** — the
  leftover <4 tokens of an incomplete block — is always included.
  Attention runs over selected ∪ tail via the mask. **Corollary**: with
  ≤ 2051 visible tokens every block is selected, so QSA is *exactly* dense
  causal attention below that — the dense path is the correct oracle for
  short prompts and wrong beyond 2K, which is where the selection test
  must live.

**PLE — the engram table**, exactly:
- `ngram_size 3` → context 2 previous tokens; `heads_per_ngram 8` → 16
  heads (8 bigram, 8 trigram); `head_dim = ple_embed_dim/16 = 160`.
- head `h` (global index `h`, one PLE layer) has vocab size = the
  `(h+1)`-th prime after 19,999,999; offsets cumulative; total padded up
  to a multiple of 128 = **320,001,536 rows × 160** = 51.2B params, stored
  as 128 shards `ngram_embedding.shard_i.weight` [2500012,160],
  concatenated along dim 0.
- multipliers `m[0..2]`: `base = 1234 + 10007·0`; `v_i = (base +
  γ·(i+1)) mod 2⁶⁴` with γ = 0x9E3779B97F4A7C15; `m_i = 2·(splitmix64(v_i)
  mod half) + 1`, `half = max(1, ((2⁶³−1) // 248320) // 2)`; splitmix64
  is the standard finalizer (0xBF58476D1CE4E5B9, 0x94D049BB133111EB,
  shifts 30/27/31). The checkpoint stores the result too (I64 [3]) — the
  Max cross-checks our C against it.
- token context is **EOS-segmented**: `shift_s(t)` = the token `s` back
  within the same segment, else `eos` (248044); a fresh sequence starts
  with two virtual `eos`. `mixed_2 = t₀·m₀ ⊕ t₁·m₁`, `mixed_3 = mixed_2 ⊕
  t₂·m₂` in wrapping int64; id = `remainder(mixed, size_h) + offset_h`
  with **torch remainder semantics** (non-negative). Gather 16 rows of
  160, flatten to 2560.
- the layer: `key = grouped_norm(key_proj(e))` [4×2560], `value =
  value_proj(e)` [2560], `qn = grouped_norm(h4)`; per stream `g =
  (key_s·qn_s)/√2560`, `g = sign(g)·√max(|g|,1e-6)`; `gv_s = sigmoid(g)·
  value`; `out = gv + silu(dilated_depthwise_conv1d(grouped_norm(gv)))`
  with kernel 4, dilation 3, 10240 groups, state length 9;
  `h4 += out`, applied **before** the attention hyper-connection of layer
  1. Weights: `key_proj` [10240,2560], `value_proj` [2560,2560], three
  [10240] norms, `conv1d` [10240,1,4].

**Vision**: 27 blocks, hidden 1152, MLP 4304, out 2560 — the 27B's tower
shape. Deferred, as planned.

**RMSNorm conventions.** All `Qwen4ExpTextRMSNorm` (q/k norms, indexer
norms, every `hc_norm`, PLE norms) are `(1 + w)`; `linear_attn.norm` is
the gated norm and is **not** (+1) — same split as the 27B. The HF
checkpoint stores raw weights; the converter applies the +1 where MLX would
have, so the engine's existing convention holds.

## Status — 2026-09-02, on the M4 Air

**Phases 1–5 met, on the toy.** With no real weights on the machine, the
whole port was built against `tests/fixtures/flashnext-tiny-q4`: a seeded
random Qwen4Exp at the real model's head geometry (256-dim attention heads,
128×128 delta heads, 128-dim index heads, 64 rotary dims — the production
kernels are compiled for those) with every feature on. Two twins hold it:

- `tests/test_flashnext`: the loader binds every tensor with shape checks;
  the engram hash constants equal the checkpoint's own buffers; the CPU
  reference (`qwasar_flash_cpu.c`) matches `transformers` to **5e-7 per
  layer at every position** and 1e-6 on logits, with every expert choice
  and every QSA mask entry equal; and the **Metal forward matches the
  reference to ~1e-4** (fp16 KV cache rounding) token-by-token and as one
  prefill, with a per-layer debug stop that names a divergence by layer.
- `tests/test_sparse`: all sixteen new kernels (`metal/sparse.metal`) and
  the reused dense/quantised matmuls at the family's shapes, each against
  scalar code, ~1e-7.

Three things learned that the plan did not know:

- **The l2norm epsilon convention differs between the two families'
  oracles.** mlx-vlm (the 27B's) puts eps inside the mean; transformers
  (this family's) puts it on the sum of squares. Identical on the 27B's
  vectors, 0.7% on the toy's small ones. The same kernel expresses either
  (`eps/dk`), and each family gets its own oracle's; the 27B path is
  unchanged.
- **Selection ties are real.** A block's index score is a sum of ReLU'd
  dots, exactly zero with probability 2⁻ʰ per block, and `torch.topk`
  breaks ties in an unspecified order. The test measures the cut's gap and
  refuses an indecisive fixture; the toy has 16 index heads for that
  reason. In production a tie at the cut will resolve differently from
  torch, and nothing can be done about that but knowing it.
- **`save_pretrained` writes experts per-expert, the real checkpoint
  fused.** The converter accepts both and emits banks.

**What the Metal path does not do yet**, in the order it will matter:

1. *Speed.* `qw_qsa_select` is k rounds of a parallel argmax per query
   (O(k·n_blocks): ~33M ops per query per layer at 262K context) and the
   expert matvec is per (token, slot) pair with no weight reuse across
   tokens. Both are the simplest correct shapes; both need the Max's
   numbers before a replacement is designed (§3, Phase 6).
2. *Checkpoints.* The indexer key cache and the engram state have no
   place in the kvstore format; `qwasar_session_save/restore` refuse the
   family. Sessions work, parking's warm resume does not.
3. *Speculation, vision, images*: refused with a message (Phases 7, and
   the deferred tower).
4. *The tokenizer's `\p{M}` regex switch* and the sharded converter for
   a 360 GB input are the two remaining Air-side items before the Max can
   load anything.

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
