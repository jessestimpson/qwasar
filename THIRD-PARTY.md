# Third-party notices

qwasar is MIT licensed (see [LICENSE](LICENSE)).  This file records everything
in the tree that came from somewhere else, what its terms are, and — where the
terms do not strictly require anything — why the acknowledgement is here anyway.

---

## linenoise — BSD-2-Clause

`linenoise.c` and `linenoise.h` are vendored unmodified.  They provide line
editing and history for the `qwasar-agent` REPL.

* Upstream: <https://github.com/antirez/linenoise>
* Obtained via the ds4 tree.

BSD-2-Clause requires the notice below to appear both in source redistributions
(satisfied by the file headers, which are intact) **and in the documentation
accompanying a binary redistribution** — which is what this file is for.

```
Copyright (c) 2010-2023, Salvatore Sanfilippo <antirez at gmail dot com>
Copyright (c) 2010-2013, Pieter Noordhuis <pcnoordhuis at gmail dot com>

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

 *  Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.

 *  Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

---

## MLX — MIT (Apple Inc.)

No MLX code is copied and qwasar does not link against it, but two Metal kernels
were written with MLX's open in front of them and follow their structure closely
enough to say so:

* **`metal/gated_delta.metal`** takes its work assignment from MLX's gated-delta
  kernel: one simdgroup per (value head, value row), the 32 lanes splitting the
  key dimension, the recurrent state held in registers across the whole call,
  and the `simd_sum` reductions for `kv_mem` and the readout.
* **`metal/attn.metal`** follows MLX's vector-SDPA shape: simdgroups striding the
  keys, the online softmax kept in registers, and the transpose-through-
  threadgroup-memory trick that lets one `simd_sum` reduce across simdgroups
  instead of within one.

MLX is MIT licensed, so nothing obliges this note or the copyright line in
`LICENSE`.  Both are there because reading that code is a large part of why
these kernels work, and saying so is more useful to a reader than pretending
otherwise.

MLX also defines the **4-bit affine quantisation format** qwasar reads directly
from the safetensors shards, and MLX's own quantised matmul is the throughput
target the kernels here are measured against.

* Upstream: <https://github.com/ml-explore/mlx>

---

## mlx-vlm — MIT (Prince Canuma)

Not vendored, and not required to build or run qwasar.  It is the reference
implementation of Qwen3.8 that this engine was written against, and the oracle
every layer is validated against: `tools/dump_golden.py` and
`tools/dump_tokens.py` drive it to produce the fixtures in `tests/`, and
`tests/test_forward.c` and `tests/test_tokenizer.c` replay those through the C
engine.

The fixtures themselves (`tests/golden.bin`, `tests/tokens.json`) are numbers
produced by running the Qwen3.8 weights, not mlx-vlm source.

* Upstream: <https://github.com/Blaizzy/mlx-vlm>

---

## ds4 (DwarfStar4) — MIT

Not vendored beyond linenoise, which is antirez's own work and listed above.

ds4 is the project qwasar is modelled on, and the debt is structural rather than
textual: the build story, the scalar CPU reference twins beside every kernel, the
narrow engine/session boundary, the agent living in the same tree, the disk KV
cache, and the prefill progress bar all come from reading it.  No ds4 source was
copied.

* Upstream: <https://github.com/antirez/ds4>

---

## Layr Labs `qwen-3.8-mtp-challenge` — MIT, studied, not used

MIT, Copyright (c) 2026 Layr Labs, Inc.

A Swift/MLX benchmark harness for this exact model's MTP head.  Nothing from it
is compiled into qwasar and no code was copied; it is cited in PLAN.md §5
Milestone 3 for measured facts that would otherwise have cost weeks to learn --
that the head must draft from committed history (accept 0.903 with, 0.262
without), that per-boundary recurrent checkpoints roughly halve the price of
drafting against replay, that an untuned depth-2 configuration lands slightly
below serial, and that verify cost is not flat in width.

Those numbers were measured on other hardware against another runtime and are
recorded as prior expectations to re-measure, not as qwasar results.

* Upstream: <https://github.com/Layr-Labs/qwen-3.8-mtp-challenge>

---

## Qwen3.8 chat template — from the model repository

`qwasar_tokenizer.c` embeds the tool-calling format description from the model's
`chat_template.jinja` **verbatim**, including the `<IMPORTANT>` reminder block.

This is not stylistic.  The model was trained conditioned on those exact bytes;
paraphrasing them would hand it a prompt it has never seen, and would measurably
change its behaviour.  The text is reproduced as data the model requires, from
the weights distribution it ships with.

Model weights are not included in this repository and are governed by their own
licence from the Qwen team.

---

## Unicode Character Database 16.0.0

`qwasar_unicode.inc` contains codepoint range tables for the `\p{L}`, `\p{N}`
and `\s` classes used by the BPE pre-tokenizer.  It is generated by
`tools/gen_unicode.py` from Python's `unicodedata` module, which derives from the
Unicode Character Database.

The tables are character-property data rather than code, regenerable from
`tools/gen_unicode.py`, and checked in only so that building qwasar never
requires Python.

* Terms: <https://www.unicode.org/license.txt>
