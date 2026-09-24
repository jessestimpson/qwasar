#!/usr/bin/env python3
"""mlx-vlm's outputs on the real Flash-Next MLX weights, for the engine to match.

Dev-only (PLAN.md 6).  PLAN-flash-next.md expected no oracle for the real
model on this machine: transformers cannot hold the BF16 checkpoint in
128 GB.  mlx-vlm can hold the 4-bit build, which is the build the engine
loads, so the two can be compared on the very same weights.

For a few prompts -- prose, code, and text that is not English, so the
tokenizer's newer paths and the engram's hash see varied ids -- this records,
per position, mlx-vlm's top logits from one prefill, and a greedy
continuation decoded through its cache.  mlx-vlm runs as it ships: BF16
activations, so its logits carry BF16's rounding and the comparison is made
at that tolerance (tests/test_flashnext_real.c).

The engram table is left on disk: mlx-vlm's own external-PLE view (hard
links, no copy) reads its rows by mmap instead of wiring 30 GB of them.

    tools/venv/bin/python tools/flashnext_real_oracle.py \\
        models/Qwen3.8-Flash-Next-MLX-4bit tests/fixtures/flashnext-real-oracle.json
"""
import json, os, shutil, sys, time
from pathlib import Path

import mlx.core as mx
from tokenizers import Tokenizer

from mlx_vlm.models.qwen4_exp.ple_storage import prepare_external_ple_model
from mlx_vlm.utils import load_model

PROMPTS = [
    "The history of the printing press begins in the fifteenth century, when",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "Paris est la capitale de la France. 東京は日本の首都です。Berlin ist",
]
TOP = 10
CONTINUE = 32

import argparse
from mlx.utils import tree_map
ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("dst")
ap.add_argument("--fp32", action="store_true",
                help="widen every BF16 parameter first, so mlx-vlm computes in fp32 as the "
                     "engine does (~15 GB more): the reference for telling rounding from a bug")
args = ap.parse_args()
src = Path(args.src).resolve()
dst = Path(args.dst)
view = src.parent / (src.name + ".mlx-external-ple")
if not view.exists():
    prepare_external_ple_model(src, view)

tok = Tokenizer.from_file(str(src / "tokenizer.json"))
t0 = time.time()
model = load_model(view)
if args.fp32:
    model.language_model.update(tree_map(
        lambda a: a.astype(mx.float32) if a.dtype == mx.bfloat16 else a,
        model.language_model.parameters()))
lm = model.language_model
print(f"loaded in {time.time() - t0:.1f}s")

out = []
for text in PROMPTS:
    ids = tok.encode(text, add_special_tokens=False).ids
    logits = lm(mx.array([ids])).logits[0].astype(mx.float32)
    top_ids = mx.argsort(-logits, axis=-1)[:, :TOP]
    top_vals = mx.take_along_axis(logits, top_ids, axis=-1)

    cache = lm.make_cache()
    step = lm(mx.array([ids]), cache=cache).logits[0, -1]
    cont = []
    for _ in range(CONTINUE):
        nxt = int(mx.argmax(step).item())
        cont.append(nxt)
        step = lm(mx.array([[nxt]]), cache=cache).logits[0, -1]
    print(f"{text[:40]!r}... -> {tok.decode(cont)!r}")
    out.append({
        "text": text,
        "tokens": ids,
        "top_ids": top_ids.tolist(),
        "top_logits": top_vals.tolist(),
        "greedy": cont,
    })

json.dump({"source": "mlx-vlm d1bd74ed, " + ("fp32" if args.fp32 else "BF16") + " activations",
           "model": src.name,
           "prompts": out}, open(dst, "w"))
print(f"wrote {dst}")
