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

### Milestone 1 — inference works at all *(done)*

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
| **prefill** | **42.6 tok/s** at a 256-token chunk (7.0 before `qmm`) |

Decode landed inside the 5.5-7 band predicted from bandwidth, first try.

**Prefill was the largest single gap; `qw_qmm_q4_g64` closed most of it.**
`qw_qmv_q4_g64` dispatches one row per token, so an N-token prefill re-read all
15 GB of weights N times. The tiled matmul stages a 64x64 output block through
threadgroup memory, so each weight block is fetched and dequantised once and
reused across the whole token tile. Measured: **7 -> 42.6 tok/s at a 256-token
chunk, 6.1x.** `qw_op_qmat_q4` picks between the two by row count.

The crossover is not where intuition puts it. The matmul pads its token tile to
64, so it costs the same for 8 tokens as for 64 -- about 2.0 s through the whole
model -- while the matvec costs ~0.18 s per token. They cross just under 12, so
`QW_QMM_MIN_ROWS` is 16. An earlier guess of 6 made 8-token prefills *slower*
than before.

### Where prefill time goes, and what moved it

`qmm` is essentially all of prefill, so its throughput is the number that
matters. Three rounds of work took it from **1.30 to 2.25 TFLOP/s**, and prefill
from 7 to 42.6 tok/s overall.

**Getting the denominator right mattered more than any single change.** Two
measurements were wrong first:

- A synthetic `simdgroup_matrix` benchmark reported **15.6 TFLOP/s**, which
  would have put the kernel at 12% of peak and implied enormous headroom. It is
  not believable: it exceeds this machine's scalar FMA peak by 4.7x, and M4 has
  no dedicated matrix hardware for it to come from. The loop's operands were
  loop-invariant and the product was being hoisted.
- MLX's own quantised matmul first measured at **86 TFLOP/s**, which is the
  classic lazy-evaluation mistake: each iteration overwrote the previous result
  and only the last was ever forced.

Forcing evaluation per iteration gives the real figure, and it is the right
target because it is the same operation on the same shapes from a heavily tuned
library: **MLX sustains 2.73-2.82 TFLOP/s.** qwasar is now at 2.25, or 80% of
that. The scalar FMA peak of 3.33 TFLOP/s turns out to be roughly the right
ceiling after all -- MLX reaches 84% of it.

What helped:

| change | result |
|---|---|
| operand tiles in `half`, accumulators still `float` | 1.83 → 2.17 (+19%) |
| store A M-major so its fragments load without transposing | 2.17 → 2.25 (+3.5%) |

What did not, and is recorded so it is not retried:

| change | result |
|---|---|
| `BK=64` (half the barriers) | 2159 vs 2161 -- barriers were never the cost |
| 8 KB pool, results stored straight to device | 2245 → 2018; a fragment store writes eight rows strided by the output width, and losing coalesced writes cost more than the occupancy gained |
| 8 KB pool, results staged in row bands | 2245 → 2054; the band split serialises the stores |
| simdgroup grid 2x2 or 1x4 (better load:MAC ratio, 128 threads) | 1950 and 1874 -- thread count beats the ratio |

**Threadgroup memory is the occupancy constraint, but 16 KB is a local optimum
rather than a monotonic one.** Padding the pool to 28 KB costs 38% of
throughput; shrinking it to 8 KB also loses, because the only ways to free that
space give up coalesced output writes. That trade is the thing to attack next if
this is revisited.

Half operands cost about two decimal digits per matmul -- worst relative error
goes from 1.4e-7 to 1.4e-5 -- and **nothing measurable at the model level**:
end-to-end logits moved from 4.777e-2 to 4.769e-2 against the reference, with
argmax and all five top-5 ranks unchanged. That is the check that licenses the
precision drop; the per-op tolerance alone would not.

---

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

The REPL runs on a scroll region, following ds4-agent: no alternate screen, so
the transcript stays in normal scrollback, with the bottom two rows reserved for
a prompt and a status footer that output scrolls underneath.

Two things about that were only discoverable by rendering the escape stream
through a real terminal emulator rather than reasoning about it:

- **DECSC/DECRC cannot hold the output position.** A saved cursor is an absolute
  screen cell, and scrolling the region silently invalidates it, so two turns of
  output landed on the same line and overwrote each other. The append point is
  now tracked as a column on a fixed row, which scrolling cannot invalidate.
- **linenoise must not own the status row.** Its relative-motion redraws and its
  full-width status padding both fight the region, and status lines ended up
  scrolled into the transcript. The prompt row is still linenoise's; the status
  row is painted directly, absolutely, and truncated a column short of the
  width so it can never wrap.

Still to do: todo tracking and glob.

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

### Milestone 3 — speculative decoding with the MTP head *(next)*

**Why this is next, and why it is ahead of vision.** Decode sits at 5.8 tok/s
against a measured bandwidth roof of 8.0 (§2): 14.95 GB of weights are read to
produce one token, and no kernel can be written that reads less. Every item in
the performance milestone below chases the gap to that roof; speculative
decoding is the only thing that moves the roof, because it amortises one pass
over the weights across several accepted tokens. It also needs no new kernels.

#### What the head actually is

Qwen3.8 ships an MTP draft head, DeepSeek-style. The base config already
declares it -- `mtp_num_hidden_layers: 1`, `mtp_use_dedicated_embeddings:
false` -- even in the checkpoint that omits the weights. Read from the upstream
`Qwen/Qwen3.8-27B` index and safetensors header, it is fifteen tensors:

| tensor | shape | what it is |
|---|---|---|
| `mtp.pre_fc_norm_hidden` / `..._embedding` | [5120] each | RMSNorm on the two inputs |
| `mtp.fc` | [5120, 10240] | projects `[norm(h) ; norm(embed(next))]` back to one hidden |
| `mtp.layers.0.self_attn.{q,k,v,o}_proj` | [12288, 5120], [1024, 5120], [1024, 5120], [5120, 6144] | one **full-attention** layer |
| `mtp.layers.0.self_attn.{q,k}_norm` | [256] | per-head RMSNorm |
| `mtp.layers.0.mlp.{gate,up,down}_proj` | [17408, 5120] ×2, [5120, 17408] | SwiGLU |
| `mtp.layers.0.{input,post_attention}_layernorm`, `mtp.norm` | [5120] | norms |

Those shapes say more than the names do. `q_proj` at 12288 = 24 × 256 × 2 is the
same query-plus-gate packing the base model's full-attention layers use;
`k`/`v` at 1024 = 4 × 256 is the same GQA-6. `mtp_use_dedicated_embeddings:
false` and the absence of any `mtp.embed_tokens` or `mtp.lm_head` mean it reuses
the base embedding table and the base output head.

**So the draft head is one more full-attention layer of exactly the shape we
already run sixteen of, and every kernel it needs exists.** The only new op is a
concat and one matmul. That is the reason this is a smaller build than vision
despite being worth more.

It is 425M parameters, 849 MB in bf16. That is *not* the cost of a drafted
token, which was the first thing this plan got wrong: a draft also has to run
the **`lm_head` readout over 248,320 rows**, ~0.65 GiB at 4-bit, which is
larger than a 4-bit head body. Layr Labs' harness measures the whole draft step
at **h ≈ 0.18 verify-forwards**, and prices its schedule off that constant.

#### Getting the weights

They are not in the MLX 4-bit checkpoint: mlx-vlm's `sanitize` drops every
`mtp.` key, which its own `test_qwen3_5_mtp_sanitize.py` pins as intended
behaviour, so every MLX conversion downstream of it is missing them. The reason
is worse than tidiness -- mlx-lm's `qwen3_5` `sanitize` treats the presence of
any `mtp.` key as a signal to **+1-shift every trunk norm weight**, which
corrupts the model. That is why heads ship as separate trees rather than merged
checkpoints, and it is a hazard for anything that reads a merged directory
through the Python stack. qwasar's loader is its own and is unaffected.

No fetch tool is needed. `EigenLabs/Qwen3.8-27B-MTP-bf16` publishes exactly this
head as four files -- `model.safetensors` at 849,400,347 bytes, plus a config
and index that exist only as compatibility assertions -- with **bare** tensor
names (`fc.*`, `norm.weight`, `pre_fc_norm_*`, `layers.0.*`), no `mtp.` prefix
and no tokenizer. `--mtp <dir>` loads it, following ds4's flag of the same name.

Run it in bf16 first: it avoids writing a quantiser. A 4-bit repack is a known
quantity rather than a hope -- the previous generation's head was published as
`mlx-community/Qwen3.6-27B-MTP-4bit`, 258 MB, its 8 matrices appearing as
weight/scales/biases triples beside the 7 norms for 31 tensors -- so if the
draft cost shows up in a measurement, the target to hit is 258 MB.

#### The blocker: qwasar cannot yet evaluate 2-8 tokens for one weight read

This is the finding that reorders the milestone, and it comes from qwasar's own
kernels rather than from the model.

Speculation pays for exactly one reason: K+1 tokens are verified in a single
pass over the weights. qwasar has no kernel that does that at speculative
widths.

- `qmv` dispatches **one threadgroup row per token** and its own header says
  what that means: "badly wrong for prefill: N tokens would re-read the weights
  N times". A width-3 verify through `qmv` costs three decode steps. There is no
  speedup to divide.
- `qmm` pads its token tile to `QW_QMM_BM` and so "costs the same for 1 token as
  for 64" -- about 14.4 token-equivalents (§3.5). Worse at these widths, which
  is why `QW_QMM_MIN_ROWS` is 16 in the first place.

So the enabling work is a **batched matvec**: `qmv` with the token loop hoisted
inside the weight walk, accumulating `[QW_QMV_ROWS][B]` for B ≤ 8 so each weight
byte is read once for the whole block. Register pressure is the design
constraint and `QW_QMV_ROWS` may have to fall to pay for B. Nothing else in the
engine needs this kernel; MTP is entirely gated behind it, and it should be
built and measured **before** any head is downloaded.

Its width curve must then be measured rather than assumed flat. Layr Labs found
theirs is not: the step into verify width 6 cost 27.3 ms against 13.4 ms into
width 5, a dispatch-shape cliff, and they cap depth at 7 and segment wider
rounds into <= 5-row launches because of it.

#### The head drafts from committed history, not from one position

The largest measured effect in the whole design, and the one this plan had
wrong by omission. The head has its own KV cache, and what goes in it decides
whether any of this works:

> accept **0.903** with history vs **0.262** without.

A head that drafts from ~one position of context is not worth running. So the
head keeps **one persistent KV cache**: the prompt is streamed into it once, and
each *committed* token's fused row is appended, so it attends over the whole
committed prefix. Layout invariant: head position `p` holds
`fused(embed(token_{p+1}), hidden_p)` -- the hidden state at a position pairs
with the *next* token.

Speculative rows must never survive a round; only committed ones are appended.
History upkeep folds into the next draft forward as extra leading rows, so the
head weights are still read once per drafting round.

Because the head only *proposes*, all of this sits outside the exactness
surface: a better or worse draft changes the accept rate and never an emitted
token.

#### Rewinding a model that cannot be rewound

For the sixteen full-attention layers, discarding a rejected draft is a cursor
rewind. For the forty-eight Gated DeltaNet layers it is the same property that
shaped the disk cache: the state is advanced in place with no per-position
history, and it is not invertible -- undoing `S = g·S + outer(delta, k)` means
dividing by a decayed `g`.

Two mechanisms exist and the choice is settled by measurement, not taste:

- **Replay.** Snapshot once before the verify, restore on rejection, then re-run
  the accepted prefix.
- **Per-boundary checkpoints.** Materialise the state at every possible accept
  boundary during the verify pass; committing is then a pointer swap.

Layr Labs ships checkpoints and quantifies the difference: a reject "pays a
repair forward" under replay, and their break-even head-cost ratio moves from
**0.43 to ~0.20** with checkpoints -- the mechanism roughly halves the price of
drafting. Take checkpoints.

**But not for free, and this plan claimed free.** The earlier draft of this
section argued the copies cost nothing because the recurrence already streams
the whole state every step. That is true of a naive kernel and false of ours:
`qw_gated_delta` deliberately holds the entire `[Dv, Dk]` state **in registers
for the duration of the call** and touches device memory once per layer, not
once per timestep -- the comment at the top of `metal/gated_delta.metal` says
so, and it is why the kernel is fast. Writing a checkpoint per boundary adds
traffic the kernel currently does not have: 157 MB per boundary (SSM 48 layers x
48 value heads x 128 x 128 x fp32 = 151 MB, conv 5.9 MB), ~1.3 ms, straight out
of registers at the right timestep. At K=3 that is ~4 ms against a ~172 ms
round: cheap, worth it, and not zero. Say the real number.

The disk checkpoint format does not change -- only the committed state is ever
persisted -- so existing cache entries stay valid.

#### The schedule is where the speedup actually lives

The number that should govern expectations: on Layr Labs' ranked runner, an
**unmodified depth-2 tree medians ~0.994** -- six tenths of a percent *slower*
than serial decode. Tuned trees on the same harness and the same weights median
**~2.8-3.3x**. Naive block MTP nets nothing on this model; essentially all of
the win is in deciding how deep to draft and in making the drafting path cheap.

So a fixed depth is the wrong shape. Their scheduler, which is the design to
follow:

- Per-position acceptance EMAs, `p[i] = P(draft i accepted | 0..<i accepted)`,
  alpha 0.15 (half-life ~9 rounds, which converges well inside a 512-token
  window), seeded with an optimistic decaying prior `0.85 * 0.98^i` capped at
  0.95. The cap is load-bearing: uncapped optimism over-drafts on hard prompts.
- A marginal rule -- draft one more only while expected committed tokens per
  unit round time still rises -- against a measured per-depth price, not a
  constant.
- A confidence gate from **the target's own top-2 logit margin** at the
  boundary: `p := min(p, sigmoid(margin/2))`. This is the disciplined version of
  ds4's `--mtp-margin`: a near-tied argmax is exactly when a draft is about to
  be wrong, and the target already computed the evidence.

qwasar's own price constant has to be fitted here, not copied: h ≈ 0.18 is
measured against their kernels and their 4-bit head, and both differ.

#### Acceptance

- **Greedy:** accept while the drafted token equals the argmax. Exact by
  construction.
- **Sampled:** modified rejection sampling, which preserves the target
  distribution exactly. Do not ship an "accept if close enough" rule; a sampler
  that silently changes the distribution is the unexplained drift §6 forbids.

#### How this is tested

The invariant is unusually strong and unusually cheap: **with drafting on, the
generated token sequence must be identical to greedy decoding with drafting
off.** Not close -- identical. Any bug in the checkpoints, the KV rewind, the
head history, or the acceptance rule breaks it immediately, and it is the same
gate the challenge harness makes absolute.

Under it: a CPU twin for the concat-and-`fc` step, the only new op; a boundary
test that accepts *j* of *K* drafts and compares the resulting state against a
plain run of the same tokens; and a batched-`qmv` test against the existing
per-token path.

One caveat worth carrying: a near-tie argmax can diverge between Apple Silicon
generations even for correct code. A token-identity failure needs the same run
on the serial path before it is called a regression.

#### Order of work

1. Batched `qmv` for B <= 8, measured against the per-token path. *(done)*
2. Head load (`--mtp`), bf16, with the concat-and-`fc` CPU twin. *(done)*
3. Persistent committed-history head cache, and acceptance per position. *(done)*
4. Per-boundary GDN checkpoints and KV rewind; the token-identity gate. *(done)*
5. Fixed depth 3, measured end to end against serial on the same thermal
   footing -- the paired back-to-back method §2 already uses. *(done)*
6. Only then the adaptive schedule, refitting the price constant from qwasar's
   own width curve.

#### Measured: what the head actually accepts

`--mtp-depth N` drafts and compares without rolling anything back, so the target
decides every token exactly as it would have. Verified: output is byte-identical
with and without it. Position *i* is counted only when every draft before it was
accepted, so these are conditional probabilities, which is what a depth schedule
needs.

| depth | tokens/round | d0 | d1 | d2 | d3 | d4 | d5 |
|---|---|---|---|---|---|---|---|
| 1 | 0.99 | 87% | | | | | |
| 2 | 1.83 | 87% | 74% | | | | |
| 3 | 2.64 | 96% | 74% | 77% | | | |
| 4 | 2.75 | 83% | 66% | 84% | 88% | | |
| 6 | 3.41 | 82% | 83% | 68% | 46%* | 60% | 50% |

\* small samples past depth 3; a hundred generated tokens is 29 rounds.

This is far better than the prior warranted, and it settles the pre-norm vs
post-norm contradiction between the two references empirically: **post-norm**.
A first-draft acceptance of 83-96% is not what a head reading the wrong hidden
state produces, and neither is it what a reversed concatenation produces.

Drafting cost lands where the arithmetic said it would: ~18 ms per draft against
a ~178 ms decode step, or **h ≈ 0.10**, which is the head's 849 MB plus the
lm_head's 715 MB against the backbone's 14.95 GB. Layr Labs measured 0.18 on
their stack; the difference is theirs is a 4-bit head with chained-launch
overhead, and ours has not paid for its round trips yet.

At depth 3 that projects to a round of 3 x 18 ms of drafting plus one verify.
If the verify of four tokens costs ~1.15 decode steps, the round is ~258 ms for
2.64 tokens, against 178 ms per token serial: **~1.8x**. That number is a
projection until step 4 exists -- there is no verify yet, only the acceptance it
would have had.

#### Measured: end to end, with the verify built

Steps 4 and 5 are done. The rewind is per-boundary state snapshots taken inside
the two recurrent kernels: both already hold their state in registers for the
whole call, so saving it after each of the first `n_draft` timesteps is a store
of what is there rather than a re-read. `tests/test_verify` is the gate, and it
holds: **the emitted token sequence is identical to greedy decoding at every
depth 1-4, on both a predictable prompt and open prose.** The prose case rewinds
23 of 29 rounds, so the identity is evidence about the rewind and not just about
lucky drafts.

200 tokens of prose, paired against serial on the same machine:

200 tokens of prose, alternating with a serial control so the two share thermal
state:

| pair | serial | speculative (depth 3) | ratio | verify, in decode steps |
|---|---|---|---|---|
| 1 (cool) | 6.13 tok/s | 8.92 tok/s | **1.46x** | 1.45 |
| 2 | 6.01 | 7.66 | 1.28x | 1.70 |
| 3 | 5.48 | 7.10 | 1.30x | 1.69 |

**1.3x on a warm machine, 1.45x on a cool one**, and the spread is not noise:
the verify costs 1.45 decode steps cool and 1.70 warm, while serial decode moves
much less. Speculation trades memory traffic for arithmetic -- one pass over the
weights, four rows of everything else -- and arithmetic is what throttling takes
away first. So the ALU in the verify path is worth more than its steady-state
share suggests: it is also the part that evaporates under sustained load.

(An unpaired measurement earlier in the same session read 1.42x. It was
flattered by measuring serial while cool and speculative while warm, which is
the argument for pairing rather than for the higher number.)

Where the verify's time goes, measured rather than modelled: **98% of it is
GPU.** Over 78 rounds, the CPU side is 0.12 s of argmax over the block's logits
and 0.29 s of rewind across 51 rewinds -- the 144 MB state copy costs 5.7 ms and
happens on two rounds in three. The remaining gap between 1.45 decode steps and
the 1.15 the batched matvec alone would cost is everything that scales with
width rather than with weights: attention reading the cache four times, and the
recurrence stepping four times through 48 layers. That last one is inherent to
speculating on a hybrid model, and is the reason a verify here cannot get as
close to one decode step as it would on a pure-attention architecture.

The first measurement of this was 1.2x, with a verify costing 1.52 decode steps
against the ~1.05 its weight traffic allows. That gap was `qmvb` running at 59
GB/s against the single-token kernel's 90, and the cause was nibble unpacking:
four operations per weight against eight fused multiply-adds of actual work.
`qw_unpack8_affine` masks the low nibble of every byte so
`unpack_unorm4x8_to_float` can convert four at a time, and folds the 1/255 into
the scale rather than undoing it -- seven operations per word instead of
thirty-two. `qmvb` went to 82 GB/s and a width-4 verify from 1.52 decode steps
to 1.15.

It is the same arithmetic: the per-op error floor is unchanged at 1e-8,
end-to-end logits moved 4.777e-2 to 4.776e-2 against the reference, and top-5
order and the token-identity gate both hold.

What remains is not the kernel. Drafting is 17% of a round, and two things
would cut it: quantising the head to 4-bit, which is a published artifact
rather than a research question, and folding the history upkeep into the first
draft's rows so the head's weights are read once per round instead of twice.

**Depth 4 is a cliff, and a predicted one.** Verify width 5 needs two `QW_QMVB_B`
blocks, so the pass costs twice what width 4 costs -- measured at 3.7 tok/s
before the unpacking work, against depth 3's 6.7. The batched matvec's block
size is therefore a hard cap on useful draft depth, at `QW_QMVB_B - 1 = 3`. This
is the same shape of finding Layr Labs report at their own width 6, from the
same cause, and it is the argument for the scheduler knowing the width curve
rather than a constant.

#### Step 6: the depth adapts

A fixed depth is wrong on one prompt or the other, and measurably so. Costed
per round on this machine, against a paired serial control:

| depth | prose: tokens/round, rate | primes: tokens/round, rate |
|---|---|---|
| 2 | 2.32, 7.63 tok/s | 3.00, 10.34 tok/s |
| 3 | 2.60, 7.47 tok/s | 3.90, 11.92 tok/s |
| adaptive | 2.40 (mean depth 2.18) | 3.88 (mean depth 2.94) |

Fixed 3 gives up 2% on prose; fixed 2 gives up 13% on the predictable prompt.
The rule lands within measurement noise of the better one in both cases without
being told which it is facing, and the mean depths show it is actually reacting
rather than averaging.

The rule is expected committed tokens per unit round time, maximised over depth:

* Per-position acceptance as an EMA, alpha 0.15, seeded optimistically at
  `0.85 * 0.98^i` and capped at 0.95. These are conditional -- position *i* is
  only an observation when everything before it was accepted -- so they
  multiply into a reach.
* A measured price per depth, `QW_DEPTH_COST`, in decode steps: 1.39, 1.58,
  1.91. Index 0 is a plain decode step at 1.00, and **the step from 0 to 1 is
  the largest one**, because turning drafting on switches every projection to
  the batched kernel and starts saving rewind state. That discontinuity is why
  this beats a constant: a stretch the head keeps getting wrong is genuinely
  cheaper decoded serially, and the rule will choose that.
* A cap on the first position from the target's own top-2 logit gap,
  `sigmoid(margin/2)`. A near-tie is exactly when a draft is about to be wrong,
  and the target has already computed the evidence. It applies to the first
  position only: it says nothing about the token after next.

`tests/test_verify` runs the adaptive schedule as one of its depths, which is
the only case that mixes plain steps in among the rounds -- and therefore the
only one that exercises flushing head rows a verify left owed. On the prose
prompt it drops to depth 0 for two thirds of the tokens, and the emitted
sequence is still identical.

Speedups end to end, against paired serial controls: **1.4x on prose, 2.1x on
predictable text.** Both are worse on a warm machine, for the reason above.

#### The head is quantised at load

A drafted token was reading 810 MB of bf16 head beside 715 MB of the base
model's output head. The head only proposes, so precision there is purely an
efficiency question -- quantisation error can change which token it suggests,
never which one is emitted. It is now converted to the same MLX affine 4-bit
format as the rest of the model at load: **810 MB to 228 MB**, about a second of
startup, and drafting fell from 4.0 s to 2.1 s over 200 tokens.

It cost nothing measurable in acceptance. Tokens per round is **identical to the
digit** across all ten configurations `tests/test_verify` runs -- an argmax over
248,320 rows is not close enough to be moved by four-bit noise except on
near-ties, and near-ties are the rounds that were going to be rejected anyway.

The quantiser needed a test of its own, because nothing else can catch it: a
badly quantised head does not fail, it drafts worse. Comparing its output
against the bf16 path is too weak -- four bits over a group of 64 costs about
14% relative error in a matvec, which swamps most mistakes. The check is
structural instead: **every weight must come back within half a quantisation
step of where it started**, which holds for any correct affine quantisation and
fails immediately for a nibble packed at the wrong offset, a group stride off by
one, or a scale paired with the wrong row. Measured worst case: 1.000 half-steps.

The bf16 matmul kernel stays, as the reference that check runs against.

#### Where the floor is now

The verify's remaining cost above one decode step is work that scales with
width rather than with weights -- attention reading its cache four times, and
the recurrence stepping four times through forty-eight layers. The second is a
property of the architecture, not of this code: a pure-attention model of the
same size would get its verify much closer to a single decode step. That is the
bill for the same hybrid design that makes this model's disk cache cheap.

*Sources for the measured figures in this section: Layr Labs'
`qwen-3.8-mtp-challenge` (MIT), kept under `reference/`. The numbers are theirs
and were measured on an M5 against MLX Swift; every one of them has to be
re-measured here before it is trusted. The kernel facts are qwasar's own.*

### Milestone 4 — vision *(done)*

The tower is 333 tensors, all **bf16 and all with biases**, which is the first
thing to notice: nothing in the text model is either. So this milestone is not
mostly about vision, it is about a second set of primitives -- LayerNorm rather
than RMSNorm, dense matmuls with a bias, tanh-GELU, and bidirectional attention
over a few thousand patches rather than causal attention over a cache.

Read off the weights rather than the docs:

| | |
|---|---|
| `patch_embed.proj.weight` | [1152, 2, 16, 16, 3] -- a patch is 2 frames x 16x16 x 3, flattened to 1536 |
| `pos_embed.weight` | [2304, 1152] -- a learned 48x48 grid, bilinearly interpolated to the image's |
| 27 x `blocks.N` | `norm1`, `attn.qkv` [3456, 1152], `attn.proj`, `norm2`, `mlp.linear_fc1` [4304, 1152], `mlp.linear_fc2` |
| `merger` | `norm` [1152], `linear_fc1` [4608, 4608], `linear_fc2` [5120, 4608] |

16 heads of 72, and `deepstack_visual_indexes` is empty, so no deepstack path.

#### The order the patches arrive in

The one detail that will silently produce nonsense. Position embeddings are
interpolated in row-major grid order and then permuted into **merge-block
order** -- `(t, h/2, 2, w/2, 2, D)` transposed to put the 2x2 block's four
patches next to each other -- before being added to the patch embeddings. So
every token from the patch embedding on lives in merge-block order, which is
also what makes the merger's reshape to 4608 a contiguous view rather than a
gather. The preprocessing has to emit patches in that order, and within a patch
the element order is T, H, W, C, because that is the layout the conv weight was
stored in.

#### Stages, each with something that can be checked

1. **Config, binding, shapes.** Every tensor asserted against the config's own
   dimensions at load, the way the MTP head is. Cheap, and it is what catches a
   checkpoint that is not the one this code was written for.
2. **Preprocess and embed.** Image loading (stb_image, vendored), resize to a
   multiple of 32, normalise to [-1, 1], patchify in merge-block order, then the
   patch projection and the interpolated position embedding. Checked against a
   golden vector from mlx-vlm, because the patch order is exactly the kind of
   thing that is wrong in a way arithmetic tolerances do not reveal.
3. **The tower.** LayerNorm, tanh-GELU, 2D RoPE, bidirectional attention. Each
   op against a CPU twin, then the 27 blocks against the reference.
4. **Merger, scatter, MRoPE.** The 2x2 merge into 5120, scattering into the text
   embeddings at `<|image_pad|>`, and real MRoPE -- image tokens are what finally
   make the three position axes diverge, which is why it could be deferred until
   now.

#### What it does

```
$ qwasar --image circle.png --no-think -p "Describe this image in one short sentence."
A solid blue circle centered on a white background.
```

224x224 becomes a 16x16 patch grid, 256 patches, 64 tokens after the merge, in
about 2 s. The same prompt without an image answers "A person is sitting on a
bench in a park, reading a book", which is the control worth keeping: it shows
the model is reading pixels rather than guessing from the question.

Against mlx-vlm on the same patches the tower is **rel l2 5.5e-3**, and that
number had to be calibrated before it meant anything. Promoting the reference's
own weights to fp32 moves its row norms by up to 1.56e-2, and qwasar sits closer
to that fp32 result (3.6e-3) than the reference's bf16 path does (3.7e-3). So
the tower is at least as accurate as what it is checked against. The first
threshold set here was tighter than the reference's own reproducibility --
exactly the mistake made with the logits tolerance in milestone 1.

#### Two things a tolerance cannot check

Both are wiring, and wiring produces a confident wrong answer rather than a
wrong number, so both are asserted structurally instead.

**Image tokens are declared, not written.** The tokenizer deliberately does not
honour control tokens inside message content -- that is what stops user text
from injecting them -- so a message carries `n_image_tokens` and the template
expands the pads itself. Smuggling `<|image_pad|>` through the content string
silently tokenizes as literal text, which was the first thing tried and the
first thing that failed.

**MRoPE has an exactly known answer**, so it gets an exact test rather than a
tolerance: a 16x16 grid is 64 tokens that advance position by 8, with the frame
axis constant and the row and column axes walking the merged grid. The text
after an image resumes from one past the largest axis, not from one past the
token count.

#### The agent and the server take images too

`qwasar-agent --image <path>` for the first turn, `/image <path>` to attach one
mid-conversation. The server accepts both APIs' shapes -- OpenAI's `image_url`
data URLs and Anthropic's base64 `source` blocks -- decoded straight from the
JSON body without touching the filesystem.

**Images defeat prefix reuse, silently, so the server does not attempt it.**
Two different pictures render to the same run of `<|image_pad|>` tokens, so a
token-sequence match can say a prompt continues the live session when the pixels
behind it were something else entirely. A request with images therefore starts
from a fresh session. Fixing it properly means keying the match on a digest of
the image bytes as well as the tokens, which is worth doing when someone is
holding a conversation about a picture.

Wiring this up surfaced a bug that had nothing to do with images: with thinking
disabled, every answer came back as `reasoning_content` with `content` null.
The generation prompt normally leaves `<think>` open and the model closes it,
but when thinking is off the template writes the close itself, so the model
never emits one and the classifier never left reasoning mode. A valid-looking
response carrying nothing in the field clients read.

#### Video

```
$ qwasar --video digits.mp4 --no-think -p "List every digit you see, in order."
video 224x224 -> 4 frame groups -> 784 patches -> 196 tokens in 3.1s
1, 2, 3, 4
```

Frames come from **AVFoundation**, which is the platform's own decoder and
therefore already knows every container the machine can play. Shelling out to
ffmpeg would have traded "one make, no dependencies" for a binary the user has
to install. Sampling is the model's own policy, from its video preprocessor
config: two frames a second, at least four, at most 768, with the resolution
budget shared across all of them -- a long clip is many small frames, because
what the tower costs is patches.

Frames pair into temporal patches, so the count is padded up to even by
repeating the last one, and `grid_t` is half the frame count.

**A video is not a taller image, and getting that wrong is subtle.** Three
things repeat per frame rather than spanning the clip: the interpolated
position grid, the rope angles, and -- the one that actually broke -- the
attention. The reference builds `cu_seqlens` by repeating `h * w` once per
frame group, so a patch never attends outside its own frame; time is carried by
MRoPE on the text side instead. An image is the single-segment case of that,
which is why the mistake was invisible until there was a second frame:
**grid_t = 1 matched the reference at 4.5e-4 while grid_t = 2 was off by 0.45.**

Once fixed, four frames land at 1.2e-2 -- and the calibration matters again.
At sixteen patches a frame the reference disagrees with its own fp32 self by
**1.38e-2**, while qwasar sits at **1.65e-3** against that fp32 result. Closer
to the reference than the reference is to itself, for the second time.

`tests/test_vision` now carries a four-frame golden and an exact MRoPE check
for `grid_t > 1`: four frames of an 8x8 grid are 256 tokens that advance
position by 8, because the resume point is one past the largest axis and not
past the token count.

The server takes video as well, as an OpenAI-shaped `video_url` block. Two
decisions there worth stating:

* **Base64 only, never a path or an http URL.** A server that fetched whatever
  a request named would read the machine's filesystem or the network on a
  stranger's behalf, which is not a feature anyone asked for.
* **A message carries images or a video, not both.** Its placeholders are a
  single run of one token, so a mixture would silently label some of them as
  the wrong kind, and the model was trained to tell them apart. Refusing with a
  clear message beats a subtly worse answer.

AVFoundation reads assets rather than buffers, so a posted video is written to
an unlinked temporary file first. The extension is load-bearing: AVFoundation
picks its demuxer from it, and an extensionless file is reported as having no
duration rather than as unreadable. The subtype from the data URL supplies it.

There is no frame-level timestamp text; this model's template does not ask for
any.

#### 2D RoPE, which is not the text model's

Half-split rather than interleaved, and unrelated to the partial multimodal RoPE
the text side uses. The frequency table is 18 entries over `head_dim/2`; a
token's angles are its row's 18 followed by its column's 18, tiled to 72, and
applied with `rotate_half`. Sharing code with `qw_op_rope_partial` would be a
mistake -- the two agree on nothing but the name.

### Milestone 5 — performance

Only after correctness is locked and a benchmark exists. In expected order of
payoff:

1. **`qmm`'s output-store trade** — the kernel sits at 80% of MLX's throughput
   on identical shapes. Threadgroup memory bounds occupancy, but every way of
   freeing it so far gives up coalesced output writes and loses more than it
   gains. Breaking that trade is the remaining ~20%.
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

Todo tracking and a `glob` tool in the agent, `/v1/responses` and
`/v1/completions` in the server, concurrent requests, NFC normalisation in the
tokenizer, a disk-cached aligned repack of the misaligned shard (§3.3).

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
