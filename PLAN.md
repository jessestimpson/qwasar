# qwasar — a Qwen3.8 inference engine for macOS Metal

`qwasar` is a single-model inference engine in C (plus Objective-C where Metal
requires it) for **Qwen3.8 27B**, the hybrid linear/full-attention VLM whose
config identifies itself as `model_type: qwen3_5` /
`Qwen3_5ForConditionalGeneration`.

It is modelled on **ds4** (DwarfStar4): no Python in the build or the runtime,
one `make` away from a binary, abstractions shaped around *inference* rather
than around *supporting many models*, and an agent (`qwasar-agent`) shipped in
the same tree.

> Naming: the request said "Qwen3.8 26B". The local weights are
> `Qwen3.8-27B-MLX-4bit` and the parameter count works out to ~27B, so this
> document uses 27B. Same model.

---

## 1. What we are implementing

### 1.1 Weights

`~/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-4bit` — **usable as-is,
nothing further to download.** 16.05 GB across 3 safetensors shards, 2180
tensors, `__metadata__ = {"format": "mlx"}`.

Two properties of this particular repo matter and are easy to get wrong:

- It is **MLX-sanitised**. Keys are already `language_model.model.…` /
  `vision_tower.…`; `conv1d.weight` is already transposed to `[C, K, 1]`; and
  the RMSNorm weights **already have the `+1.0` offset applied** (measured:
  `input_layernorm` mean 0.967, `q_norm` mean 1.230 — i.e. centred on 1, not on
  0). Do **not** re-apply the offset that `mlx_vlm/models/qwen3_5/qwen3_5.py`
  applies to raw HF checkpoints.
- The **language model is 4-bit quantised; the vision tower is not.** Vision
  weights are plain BF16.

### 1.2 Quantisation format (MLX affine)

`bits=4`, `group_size=64`, `mode=affine`. Per quantised linear:

| tensor | dtype | shape |
|---|---|---|
| `.weight` | `U32` | `[out, in/8]` — 8 nibbles per word, element `i` at bit `4*(i%8)` |
| `.scales` | `BF16` | `[out, in/64]` |
| `.biases` | `BF16` | `[out, in/64]` |

Dequantisation is `w = scale * q + bias` with `q` the raw unsigned nibble in
`[0, 15]`. Scales are signed. There is no zero-point subtraction — the bias
carries it.

### 1.3 Architecture

```
hidden 5120 · 64 layers · vocab 248320 (untied lm_head) · rms_eps 1e-6
```

**Layer schedule** — `full_attention_interval = 4`, so layer `i` is full
attention iff `(i+1) % 4 == 0`. That gives **16 full-attention layers**
(3, 7, 11, …, 63) and **48 gated-delta (linear attention) layers**. Every layer
has the same dense SwiGLU MLP; there are no experts.

**Full attention** (`self_attn`), gated in the Qwen3-Next style:
- `head_dim = 256`, 24 query heads, 4 KV heads (GQA factor 6).
- `q_proj: 5120 → 12288` = `24 * 256 * 2`. Split per head into **query** and
  **output gate**; the attention result is multiplied by `sigmoid(gate)` before
  `o_proj`. This is `attn_output_gate: true`.
- Per-head RMSNorm on Q and K (`q_norm`, `k_norm`, dim 256).
- **Partial MRoPE**: `partial_rotary_factor = 0.25` → only the **first 64 of
  256** head dims are rotated, `rope_theta = 1e7`, `mrope_section = [11,11,10]`,
  `mrope_interleaved: true`.

  Interleaved MRoPE means frequency index `j` of 32 draws its position from axis
  `j % 3` (t, h, w) — with the tail landing on `[…, t, h]` so the section counts
  come out to 11/11/10. Rotation is **half-split** (`rotate_half`), not the
  traditional even/odd pairing: `cos/sin` are `concat([f, f])` over 32
  frequencies. **For text-only input all three axes carry the same position**,
  so MRoPE collapses to plain partial RoPE — Milestone 1 does not need the
  interleaving at all.

**Gated DeltaNet** (`linear_attn`), 48 layers:
- 16 key heads × 128, 48 value heads × 128 (GQA factor 3), conv kernel 4.
- `in_proj_qkv: 5120 → 10240` (= `2*16*128 + 48*128`), `in_proj_z: 5120 → 6144`,
  `in_proj_b: 5120 → 48`, `in_proj_a: 5120 → 48`.
- Causal depthwise conv1d over the 10240 mixed-QKV channels, then `silu`.
- Normalisation: `k = l2norm(k)`, `q = l2norm(q) / sqrt(128)`. (The MLX code
  writes this as `inv_scale**2 * rms_norm(q, None)`, which is the same thing
  since `l2norm(x) = rms_norm(x) / sqrt(D)`.)
- Gates: `g = exp(-exp(A_log) * softplus(a + dt_bias))`, `beta = sigmoid(b)`,
  both per value-head, computed in fp32.
- Recurrence, state `S[Hv, Dv, Dk]` in fp32:
  ```
  S      *= g
  kv_mem  = S · k
  delta   = (v - kv_mem) * beta
  S      += outer(delta, k)
  y       = S · q
  ```
- Output: `out_proj(rms_norm_gated(y, z))` where the gated norm is
  `rms_norm(y, w) * silu(z)` per value-head (dim 128). Note `linear_attn.norm`
  is the **one** RMSNorm in the model that does *not* carry the `+1` convention.

**MLP**: `down(silu(gate(x)) * up(x))`, `5120 → 17408 → 5120`.

**Vision tower** (Phase 6): patch 16, temporal patch 2, 27 blocks, hidden 1152,
16 heads (72/head), plain GELU-tanh MLP (4304, *not* SwiGLU), LayerNorm with
bias, learned `pos_embed [2304, 1152]` (48×48, interpolated per image),
`spatial_merge_size 2` and a merger `4608 → 4608 → 5120`. `deepstack` is
explicitly disabled for this model.

**MTP**: `mtp_num_hidden_layers: 1`. The draft head is a separate shard and is
**not present** in these weights. Out of scope until speculative decoding.

### 1.4 Tokenizer & chat template

GPT-2 style byte-level BPE, 248044 merges-vocab + 33 added special tokens
(248044–248076). NFC normaliser, the standard GPT-4 split regex, `ignore_merges:
false`, no byte fallback. BOS/pad `<|endoftext|>` = 248044; stop tokens are
**248046 (`<|im_end|>`) and 248044**.

The chat template is ChatML with two Qwen3.8-specific wrinkles:
- **Reasoning is on by default.** The generation prompt ends with
  `<|im_start|>assistant\n<think>\n`, and a system message carrying a
  `reasoning_effort` instruction (`xhigh` default) is *synthesised* even when the
  caller supplies no system message.
- **Tool calls are XML, not JSON**:
  ```
  <tool_call>
  <function=NAME>
  <parameter=KEY>
  value
  </parameter>
  </function>
  </tool_call>
  ```
  Tool *definitions* are still JSON inside a `<tools>` block. This shapes
  `qwasar-agent`'s parser.

---

## 2. Target machine and the honest performance ceiling

Measured on this host:

```
Apple M4 · 10 CPU cores · 10 GPU cores · 32 GB unified · macOS 26.5.1
Metal recommendedMaxWorkingSetSize = 26.80 GB
Metal maxBufferLength              = 20.10 GB
```

**The Metal toolchain is installed, but the build does not depend on it.**
`xcodebuild -downloadComponent MetalToolchain` fails at Apple's asset catalog
(`Failed fetching catalog for assetType com.apple.MobileAsset.MetalToolchain`,
`RequestedBuild = 17F113`) — not a network, proxy, or licence problem, all three
were checked — but the same component installs cleanly through **Xcode →
Settings → Components**. `xcrun metal` now reports 32023.883.

The build still compiles kernels at runtime (§3.1) and must keep doing so: that
is what makes the binary self-contained and installable on a machine without the
component. What the toolchain buys is a **lint**, exposed as `make check-metal`:
it concatenates `metal/*.metal` and runs the offline compiler over them in about
a second. Two real errors during Phase 1 — a helper used before its definition
because the wildcard sorted `act.metal` ahead of `common.metal`, and a local
named `half`, which is a Metal builtin type — surfaced only when a test loaded
15 GB of weights and failed at model-load time. `check-metal` catches that class
instantly. It is a convenience, never a build step.

Runtime introspection remains the occupancy signal: a built
`MTLComputePipelineState` reports `maxTotalThreadsPerThreadgroup`,
`threadExecutionWidth`, and `staticThreadgroupMemoryLength`. `qw_qmv_q4_g64`
reports 1024 / 32 / 0 B — the full 1024 means register pressure is not capping
occupancy. Instruments (`xctrace`) covers timeline profiling.

Also verified: `newLibraryWithSource:` compiles and dispatches without the
component, and `newBufferWithBytesNoCopy:` binds a 5.34 GB `mmap` of a shard
directly.

### Memory budget

| | |
|---|---|
| language model weights (4-bit) | ~15.1 GB |
| vision tower (BF16) | ~0.92 GB |
| SSM state (48 × `[48,128,128]` fp32 + conv) | 151 MB, **constant in context length** |
| KV cache (16 layers × 4 heads × 256 × 2 × fp16) | **64 KB/token** → 2 GB @ 32K, 8 GB @ 128K |

Text-only at 32K context lands around 17.5 GB against a 26.8 GB budget. Roomy.
The hybrid design is what makes long context affordable here: only a quarter of
the layers pay per-token KV.

### Decode speed ceiling — read this before setting expectations

This is a **dense** 27B. Every decoded token reads every weight. M4 (base, not
Pro/Max) has ~120 GB/s of memory bandwidth, so:

```
14.95 GB / 120 GB/s  ≈  8.0 tok/s   absolute roof
realistic, at 70-85% of peak bandwidth  ≈  5.5-7 tok/s
```

No amount of kernel work moves that roof — it is a bandwidth identity, not an
efficiency problem. Optimisation work (Phase 7) is about *reaching* 6-7 tok/s
rather than sitting at 2. Prefill is compute-bound and has much more headroom.

**Measured, first cut.** `qw_qmv_q4_g64` as first written (§3.5, no tuning)
sustains **82.8 GB/s** on layer 0's `gate_proj` — 69% of peak, implying ~5.5
tok/s whole-model decode. That lands inside the predicted band before any
optimisation, so Phase 7's job is the last ~30%, not a rescue.

If interactive speed matters more than this model specifically, the lever is a
smaller/sparser model or a Pro/Max/Ultra host — worth knowing now rather than
after Phase 7.

---

## 3. Design

### 3.1 Build: one `make`, no Python, no Metal toolchain

ds4 concatenates `metal/*.metal` off disk at startup and calls
`newLibraryWithSource:`. That has the great property of **not requiring the
Metal toolchain**, but it makes the binary depend on its source tree at runtime.

qwasar keeps the runtime compile and drops the dependency: `make` runs a
6-line `bin2c` C program over `metal/*.metal` to generate `qwasar_metal_src.inc`
(a string literal), which is compiled into the binary. At startup we compile that
embedded source and cache the resulting `MTLBinaryArchive` under
`~/.cache/qwasar/<sha256-of-source>.metallib`, so only the first run pays.

Result: `cc`, Foundation, Metal. No Xcode component download, no Python, no
codegen step the user has to know about, and a single relocatable binary.
`DS4`-style env overrides (`QWASAR_METAL_<FILE>_SOURCE`) stay available for
kernel iteration without rebuilding.

### 3.2 Files

```
qwasar.h              public engine boundary (engine + session; no tensor internals)
qwasar.c              config, safetensors mmap, weight table, tokenizer,
                      chat template, session lifecycle, graph scheduling
qwasar_gpu.h          GPU-facing interface used by qwasar.c
qwasar_metal.m        Metal runtime: device, library, pipelines, buffers, encode
metal/*.metal         kernels
qwasar_cli.c          REPL + one-shot CLI
qwasar_agent.c        agentic loop, tool dispatch, XML tool-call parser
tests/                unit + golden-vector regression
tools/                DEV ONLY — not part of the build; generates golden vectors
                      from the mlx-vlm reference (see §4)
```

### 3.3 Weight loading

`mmap` each shard `PROT_READ|MAP_PRIVATE`, wrap the whole shard in one
`newBufferWithBytesNoCopy:` (shards are 5.34 GB, well under the 20.1 GB cap),
and address every tensor as `(buffer, byte_offset)`. Zero copy, zero eager read;
the page cache does the work and resident set grows as layers are first touched.

**Alignment, and why one shard is an exception.** Packed 4-bit words are read as
`uint` and scales as `ushort`, so tensor addresses must be at least 4-byte
aligned. Every safetensors tensor offset here is 32-byte aligned *relative to
its shard's data section* — but the data section itself starts right after a
JSON header of arbitrary length, and nothing makes that aligned. Measured on
this checkpoint:

| shard | data section starts at | `% 16` | |
|---|---|---|---|
| 1 | 104752 | 0 | zero-copy |
| 2 | 94284 | 12 | zero-copy (4-aligned, enough for `uint`) |
| 3 | 80323 | 3 | **misaligned — 4.99 GB copied into aligned memory at load** |

No mapping trick fixes this: `mmap` only controls the address modulo the page
size, so the byte offset within a word is a property of the file. So the loader
decides per shard — wrap when aligned, copy the data section once when not — and
`--info` reports the split (currently 9.96 GB mapped, 4.99 GB copied). Load is
still well under a second warm.

Phase 7's vectorised `uint4` loads will want 16-byte alignment, which shard 2
also fails. The answer then is a **disk-cached aligned repack** under
`~/.cache/qwasar` — mapped zero-copy forever after the first run — not more
resident copies. That repack is also the natural place to hang layout changes
(interleaving scales with weights, pre-swizzling for the matvec), so it earns
its keep rather than existing only to fix alignment.

The parsed safetensors header becomes a flat name→`{buffer, offset, dtype,
shape}` table. Layer weights are resolved once at load into a
`qwasar_layer[64]` array of direct pointers so the hot path never does a string
lookup.

### 3.4 Execution model

Whole-model graph, ds4-style: one command buffer encodes every layer for a step,
committed once. No per-op CPU round-trip, no synchronisation inside a step.
Scratch activations live in a small ring of reused device buffers sized at load.

Two step shapes:
- **decode** — `L = 1`. Gated-delta runs its per-token recurrence kernel;
  attention runs a decode flash kernel against the KV cache.
- **prefill** — `L = N`. Gated-delta runs the chunked scan; attention runs a
  causal tiled flash kernel. Chunked to bound scratch memory.

### 3.5 Kernels

| kernel | notes |
|---|---|
| `qmv_q4_g64` | 4-bit affine mat-vec. **The** decode kernel — ~90% of decode time |
| `qmm_q4_g64` | 4-bit affine mat-mul, simdgroup-tiled, for prefill |
| `get_rows_q4` | embedding lookup with inline dequant |
| `rms_norm` / `rms_norm_gated` | plain, and the per-head `* silu(z)` variant |
| `rope_partial_mrope` | rotate first 64 of 256 dims, half-split |
| `flash_attn_decode` / `flash_attn_prefill` | GQA 24/4, head_dim 256, output-gated |
| `kv_write` | append K/V to cache |
| `swiglu` | `silu(a) * b` |
| `dw_conv1d_causal` | depthwise K=4 over 10240 channels, with state |
| `gated_delta_step` | fp32 recurrence; grid `(32, Dv, Hv)`, 4 elems/thread |
| `gated_delta_chunked` | prefill scan (Phase 3) |
| `sample` | temperature / top-k / top-p / min-p, on-GPU argmax fast path |

`head_dim = 256` is unusually large and will drive flash-attention tiling
choices; the M4's 32 KB threadgroup memory holds only 32 fp16 K-vectors of that
width, so the K/V tile is the thing to tune.

### 3.6 Public API

Narrow, ds4-shaped — CLI, agent, and (later) server all sit on it:

```c
qwasar_engine  *qwasar_engine_load(const qwasar_options *opts);
qwasar_session *qwasar_session_new(qwasar_engine *e);
int  qwasar_session_sync(qwasar_session *s, const int *tokens, int n, ...);
int  qwasar_session_sample(qwasar_session *s, const qwasar_sampling *sp);
void qwasar_session_free(qwasar_session *s);
```

`sync` takes a full token prefix and decides for itself whether to reuse, extend,
or rebuild. Callers never see a tensor.

**Prefix reuse is a hybrid-model design constraint worth stating up front.** A
pure-attention engine can truncate a KV cache to any prefix length. Here, the 48
gated-delta layers carry a *recurrent* state with no per-position history — you
cannot rewind it. So a session can extend its prefix cheaply but **cannot
rewind** without re-prefilling. Editing an earlier turn is a full re-prefill.
`qwasar_session_sync` will state this in its contract, and the agent will be
written to append-only. Checkpointing the SSM state (151 MB, context-independent)
at turn boundaries is the mitigation, and is cheap — Phase 5.

---

## 4. Correctness strategy

Two independent oracles, because a 27B model gives no useful signal from
eyeballing output:

1. **Per-op CPU reference in C.** Every kernel has a scalar fp32 twin in
   `qwasar.c` and a test in `tests/` that runs both on random input and compares.
   This catches packing, indexing, and layout bugs — the overwhelming majority.
2. **Golden vectors from mlx-vlm.** `tools/` (dev-only, never built by `make`)
   drives the reference implementation in `reference/mlx-vlm` and dumps hidden
   states after selected layers plus final logits for fixed prompts. A test
   replays them through qwasar and asserts agreement. This catches architecture
   misreadings that a per-op test cannot.

   The venv is already built and working: `reference/mlx-vlm/.venv` (mlx 0.32.1).

Bring-up order is layer-by-layer against oracle 2: embedding → layer 0
(gated-delta) → layer 3 (full attention) → 4 layers → 64 layers → logits. A
hybrid model has two very different layer types and diverging early is the
expected failure mode, so the first gated-delta layer and the first attention
layer each get their own checkpoint.

**Tolerances, measured rather than guessed.** The reference keeps activations
in bf16. Running the same 17-token prompt through it batched versus one token at
a time -- identical arithmetic, different accumulation order -- moves its own
logits by **7.4e-2** relative L2, and reorders its own top-3. That is the noise
floor, and it means **logit L2 is a weak signal**: qwasar sits at 4.8e-2 against
the batched reference, i.e. closer to it than the reference's own stepwise run
is.

So correctness is judged on three things instead:

1. **argmax and top-5 order** must match exactly. They do.
2. **The per-layer drift curve must be smooth.** Layer 0 lands at 3.5e-3, about
   one bf16 rounding (eps = 3.9e-3), and grows to 4.6e-2 by layer 63 with no
   step change -- and critically no discontinuity between gated-delta layers and
   attention layers, which are entirely separate code.
3. **Per-op agreement with the CPU twins**, which is where real tolerances live:
   fp32 rounding, ~1e-7 to 1e-8.

---

## 5. Milestones

### Milestone 1 — inference works at all *(the current goal)*

Text-only, greedy, correct. Speed is explicitly not a goal here beyond "not
absurd".

- **Phase 0 — foundation.** Makefile + `bin2c` + embedded-source Metal library
  with on-disk pipeline cache. Safetensors mmap loader, config parser, weight
  table, no-copy `MTLBuffer` binding. `qwasar --model … --info` prints the
  parsed architecture and a weight inventory. *Proves the load path and the
  build story before any math exists.*
- **Phase 1 — kernels + CPU twins.** All decode-path kernels from §3.5 plus
  their scalar references and unit tests. `qmv_q4_g64` first and validated
  hardest — everything downstream is meaningless if dequant is wrong.
- **Phase 2 — decode graph.** Assemble the 64-layer forward for `L=1`. Validate
  layer-by-layer against golden vectors. **Prefill is a sequential loop over the
  decode path** at this stage — correct, slow, and it gets us to end-to-end
  generation without the chunked scan.
- **Phase 3 — real prefill.** Chunked gated-delta scan and tiled causal flash
  attention for `L=N`. Same golden vectors, now at batch. This is where prompt
  processing stops being unusable.
- **Phase 4 — usable CLI.** Byte-level BPE tokenizer (hand-written pre-tokenizer
  state machine over the GPT-4 split regex, with a checked-in Unicode
  `\p{L}`/`\p{N}` table so the build stays Python-free), the ChatML template
  including the `<think>` and reasoning-effort behaviour, sampling
  (temperature/top-k/top-p/min-p; the config's own defaults are
  `temp 1.0, top_k 20, top_p 0.95`), streaming output with thinking-block
  handling, and a linenoise REPL.

**Milestone 1 is done when** `./qwasar -m <model> -p "..."` streams a coherent
answer and the golden-vector test passes end to end.

---

### Status

**Milestone 1 is complete.** Phases 0-4 are done; the model takes text and
returns text:

```
$ qwasar -m <model> -p "Name three prime numbers, with one sentence on why each is prime."
2 is prime because its only positive divisors are 1 and itself.
3 is prime because it cannot be divided evenly by any whole number other than 1 and 3.
5 is prime because its only positive divisors are 1 and 5.
```

All seven test suites pass. The golden-vector replay matches the reference's
**argmax and all five top-5 ranks exactly**, with a smooth per-layer drift curve
(§4). The tokenizer matches the reference on 24 encode cases and all 6 chat
template renderings, exactly.

The pre-tokenizer is a hand-written state machine over the model's split
pattern, with checked-in `\p{L}` / `\p{N}` / `\s` range tables generated once by
`tools/gen_unicode.py` -- so the build stays Python-free.

**One deliberate divergence from the reference:** `qwasar_encode` never emits
control tokens, however the input spells them. HF's tokenizer splits input on
added tokens, which means user text containing `<|im_start|>` becomes a real
role boundary. The template emits control tokens by id, so message content
cannot forge one. Known gap in the other direction: the NFC normalizer is not
applied, which is a no-op for ASCII and already-normalised text.

### Measured performance

Separating one-time costs from steady state, on the M4:

| | |
|---|---|
| engine load | 8.6 s — dominated by the 4.99 GB alignment copy (§3.3) |
| first forward pass | 4.8 s — faulting 9.96 GB of mapped weights in; the CLI now pays this during load so its prefill figure is the steady-state one |
| **decode** | **5.8 tok/s** steady (0.17 s/token) |
| **prefill** | **35 tok/s** at a 256-token chunk (7.0 before `qmm`, 26 register-tiled) |

Decode landed inside the 5.5-7 band predicted from bandwidth, first try.

**Prefill was the largest single gap; `qw_qmm_q4_g64` closed most of it.**
`qw_qmv_q4_g64` dispatches one row per token, so an N-token prefill re-read all
15 GB of weights N times. The tiled matmul stages a 64x64 output block through
threadgroup memory, so each weight block is fetched and dequantised once and
reused across the whole token tile. Measured: **7 -> 35 tok/s at a 256-token
chunk, 5x.** `qw_op_qmat_q4` picks between the two by row count.

The crossover is not where intuition puts it. The matmul pads its token tile to
64, so it costs the same for 8 tokens as for 64 -- about 2.0 s through the whole
model -- while the matvec costs ~0.18 s per token. They cross just under 12, so
`QW_QMM_MIN_ROWS` is 16. An earlier guess of 6 made 8-token prefills *slower*
than before.

### Where the remaining prefill headroom is

The first version was register-tiled: a 4x4 accumulator per thread, eight
threadgroup loads for every sixteen fused multiply-adds. It held **1.2-1.33
TFLOP/s** on every projection shape and every row count. A pure-FMA kernel
measures this machine's fp32 peak at **3.33 TFLOP/s**, so that was 39%, and the
flatness across shapes said the limit was kernel structure rather than any
shape or tail effect.

Rebalancing the tile did not help, which is what ruled out the load-to-arithmetic
ratio as something tunable:

| change | result |
|---|---|
| `float4` threadgroup loads in the inner loop | 1235 vs 1239 GFLOP/s -- the compiler already vectorised |
| `BM=128, BN=64, TM=8, TN=4` (ALU:LDS 2.7) | +5% at 256 tokens, but half the throughput at 64 |
| `BM=BN=64, TM=TN=8` (ALU:LDS 4.0, 64 threads) | 3.9 tok/s -- two simdgroups per threadgroup starves the core |
| `BM=BN=128, BK=16` | far worse; halving BK doubles the barrier count |

Moving the inner product onto the **8x8 matrix units** did: one
`simdgroup_multiply_accumulate` performs an entire 8x8x8 product, which removes
the ratio as a limit instead of tuning around it. **1.83 TFLOP/s, 55% of peak**,
and prefill went 26 -> 35 tok/s.

Remaining headroom is the other 45%. The next lever is holding the operand tiles
in `half` rather than `float`: Apple's matrix units run fp16 inputs at a higher
rate, and halving the tiles would allow `BK=64` at the same 16 KB, halving the
barrier count too. Both operands are well inside fp16 range -- dequantised
4-bit weights and post-norm activations -- but it changes the numerics, so it
needs re-validation against the CPU reference rather than being assumed safe.

### Milestone 2 — `qwasar-agent` *(working)*

Agentic loop on the same engine, with six tools: `read`, `write`, `edit`,
`list`, `grep`, `bash`. Reading runs unattended; writing files and running
commands ask first unless `--yes`. `AGENT.md` in the working directory is
folded into the system prompt.

**The append-only constraint turned out to cost nothing here.** An agent loop
only ever appends, so each step feeds back exactly the tokens the model just
produced plus the rendered tool result, and the KV cache and recurrent state
carry forward untouched. No SSM checkpointing was needed. Re-rendering the
conversation each turn — the obvious implementation — would have been both
slower and, for the recurrent half, wrong.

Verified end to end on real tasks: given a C file with an inverted comparison in
a `maxval` function, it read the file, diagnosed the bug, applied a one-line
`edit`, rebuilt with `bash`, and confirmed the output. The edit was minimal and
indentation-preserving.

The REPL is in: with no task argument it opens a prompt with linenoise history
and `/help`, `/new`, `/effort`, `/think`, `/yes`, `/ctx`, `/quit`. Prefill shows
a progress bar in ds4-agent's style, with the rate written into the unfilled run
of the bar so the line keeps one width. It draws only on a terminal, and only
above 128 tokens -- a tool result is a few dozen tokens and a bar that appears
and vanishes is worse than none.

Still to do: todo tracking, glob, and interrupting a running turn.

### Disk KV cache

Checkpoints live in `~/.cache/qwasar/kv`, keyed by a hash of their token
sequence, evicted LRU against a 6 GB budget. A checkpoint is written to a
temporary file and renamed, so it is either complete or absent -- a half-written
one would restore silently corrupt state.

**Measured on a cold agent start:** 46.7 s to 21.1 s. The restore itself reads
874 tokens of state in **0.02 s**, against roughly 25 s to prefill them.

Two things differ from ds4's `ds4_kvstore.c`, both forced by the architecture:

- **A checkpoint is not just the KV cache.** Forty-eight of sixty-four layers
  are recurrent, so the conv and delta-rule state travel with it. Those are the
  same size for a short prefix as for a long one -- about 149 MB -- so every
  checkpoint has a large fixed floor and a shallow slope. An 874-token
  checkpoint is 214 MB, of which only 56 MB is KV. That is why the store keeps
  few large entries rather than ds4's many small ones, why there is a
  256-token minimum, and why whole conversations are saved only on `/save`
  rather than after every turn.
- **Prefix-only reuse is mandatory, not an optimisation.** A KV cache can be
  truncated to any length; a recurrent state keeps no per-position history and
  cannot be rewound. A checkpoint is usable exactly when its tokens are a prefix
  of the incoming prompt. ds4 keys on a hash of the rendered byte prefix for the
  same reason, so that part of the design carried over unchanged.

The agent checkpoints the **system prefix** automatically, since that span is
identical on every run and is most of a cold start. It stops the first eval at
the system boundary to leave the checkpoint exactly there, which costs nothing:
the same tokens get evaluated either way.

Not yet borrowed from ds4: hit-weighted eviction with a six-hour half-life
(plain LRU for now), and saving on shutdown.

**What the test pins is indistinguishability, not speed.** A session continued
from disk must produce exactly what the original would have produced next, so
the test evaluates one more token on both paths and requires the logits to be
**bit-identical** -- rel l2 exactly 0. Anything else would mean some part of the
recurrent state did not survive the round trip, which would surface as a model
that answers differently after a restart. The negative cases matter as much: a
prompt differing one token into the prefix must miss, a prompt shorter than the
checkpoint must miss, and a truncated file must be rejected rather than
restored as garbage.

### Milestone 3 — vision

Patch embedding, the 27-block SigLIP-style tower with 2D RoPE, spatial-merge
merger, `<|image_pad|>` scatter into the token embeddings, and — only now
necessary — **real MRoPE**, since image tokens are what make the three position
axes diverge. Image loading in C (stb_image, vendored).

### Milestone 4 — performance

Only after correctness is locked and a benchmark exists. In expected order of
payoff:

1. **`half` operand tiles in `qmm`** — the matmul now runs on the 8x8 matrix
   units at 55% of measured peak. fp16 tiles would raise the matrix-unit rate
   and allow `BK=64` at the same threadgroup memory, halving the barriers.
   Needs re-validation against the CPU reference, since it changes numerics.
2. `qmv_q4_g64` — the whole decode budget, already at 69% of peak bandwidth.
   Vectorised `uint4` loads, scales/biases resident in registers, tuned
   rows-per-threadgroup. Target: ≥85% of measured `memcpy` bandwidth.
3. Command-buffer shape — fewer, larger encodes; kill any residual per-layer
   CPU sync.
4. Kernel fusion — norm+matvec, SwiGLU into the up/gate matvecs, gate-sigmoid
   into `o_proj`.
5. `gated_delta_step` — 48 layers × a serial recurrence is the second-largest
   decode cost. Keep the fp32 state in registers across the whole kernel.
6. Flash attention tiling for `head_dim = 256`.
7. Startup: the alignment copy and the 4.8 s first-touch fault storm are 13 s of
   dead time per process. A disk-cached aligned repack (§3.3) removes both.

A `qwasar-bench` binary lands at the start of this milestone, not the end.

### Later (not planned in detail)

`qwasar-server` (OpenAI/Anthropic-compatible HTTP), speculative decoding via the
MTP head (needs a shard we do not have), disk KV/SSM checkpoints.

---

## 6. Rules for this codebase

Inherited from ds4's `AGENT.md`, because they are the reason ds4 reads well:

- **No Python** in the build or the runtime. `tools/` is dev-only and never
  required to build, test, or run.
- **No C++.**
- Correctness before speed. Never keep a faster path with unexplained drift.
- mmap-backed loading; do not eagerly copy 16 GB.
- Comment the model mechanics, cache lifetimes, and memory policy where they are
  not obvious locally. Prefer a comment beside the code over a design document.
- Keep `qwasar.h` narrow. CLI/agent code must not learn tensor internals.
- No permanent semantic variants behind flags. Diagnostic switches are fine.
- Small, sharp, readable. No slop.

## 7. Open questions

- **Whether to keep a whole-model CPU reference path.** ds4 has one, but at 27B
  it would be minutes per token and ds4's own notes warn about macOS kernel VM
  failures with very large mappings. Current plan: per-op CPU twins only, with
  mlx-vlm as the whole-model oracle. Revisit if debugging demands it.
- **KV cache dtype.** fp16 at 64 KB/token is the plan. fp8 would halve long-context
  cost; defer until there is a quality benchmark to measure the damage.
- **Context limit for Milestone 1.** `max_position_embeddings` is 262144, which
  would be a 16 GB KV cache and does not fit alongside the weights. Default to
  32K and make it an explicit flag.
