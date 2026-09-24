#!/usr/bin/env python3
"""A toy Flash-Next checkpoint in MLX's format, and mlx-vlm's outputs for it.

Dev-only (PLAN.md 6).  The mlx-community build of Qwen3.8-Flash-Next differs
from our converter's output in ways the engine has to get exactly right, and
none of them can be checked against the 111 GB model alone:

  * groups of 32, not 64, for almost every quantised tensor;
  * the router and shared-expert gate 8-bit (group 64), the injection
    weights 4-bit -- tensors the graph reads as BF16;
  * gate and up as two expert banks (`switch_mlp`) instead of one;
  * the engram table 4-bit in groups of 32, as `shards.N`;
  * the (1+w) norm gains stored centred at zero, not with the +1 folded in.

So this builds the toy flashnext_tiny.py makes (with 32-wide engram rows,
which group-32 quantisation needs), converts it with mlx-vlm's OWN code --
the model class, its sanitize(), its quantisation predicate, quantize_model
at group 32, save_weights -- the path the real build took, and then loads the
result back through mlx-vlm and records what it computes:

  * logits at every position from one prefill, and fed one token at a time
    through the cache (the decode path);
  * the 4-stream residual after every decoder layer, from the prefill.

The engine is held to these in tests/test_flashnext.  mlx-vlm is run in
fp32 (its BF16 parameters widened, which is exact), so the comparison
measures the engine and not the reference's rounding.

    tools/venv/bin/python tools/flashnext_mlx_toy.py tests/fixtures/flashnext-tiny-mlx
"""
import argparse, json, os, re, shutil, subprocess, sys, tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_map

from mlx_vlm.models.qwen4_exp import Model, ModelConfig
from mlx_vlm.models.qwen4_exp import language as lang
from mlx_vlm.quant_utils import quantize_model
from mlx_vlm.utils import load_model, save_config, save_weights
from safetensors.torch import load_file as load_pt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("out", nargs="?", default="tests/fixtures/flashnext-tiny-mlx")
ap.add_argument("--prompt-seed", type=int, default=11,
                help="the prompt, drawn as tools/flashnext_oracle.py draws it; a prompt can "
                     "land on a near-tie in the indexer's block scores, which tests/test_flashnext "
                     "refuses -- the answer is another seed")
args = ap.parse_args()

work = tempfile.mkdtemp(prefix="flashnext-mlx-toy-")
hf = os.path.join(work, "hf")
subprocess.run([sys.executable, os.path.join(ROOT, "tools/flashnext_tiny.py"), hf, "1",
                "--ple-embed-dim", "512"], check=True, stdout=subprocess.DEVNULL)
text_cfg = json.load(open(os.path.join(hf, "config.json")))

# ---- the HF toy, as the VLM checkpoint mlx-vlm converts -----------------------
# mlx-vlm's qwen4_exp is the vision-language model; the text model sits under
# model.language_model.  A vision tower is required by the class and unused by
# anything here; the smallest one it accepts is left at its random init.
raw = {}
for f in sorted(os.listdir(hf)):
    if f.endswith(".safetensors"):
        for k, v in load_pt(os.path.join(hf, f)).items():
            raw[k] = mx.array(v.float().numpy()).astype(mx.bfloat16) if v.is_floating_point() \
                else mx.array(v.numpy())
# save_pretrained writes the expert banks one expert at a time; the released
# checkpoint, and so mlx-vlm's sanitize(), has them fused.
per_expert = {}
for k in list(raw):
    m = re.match(r"^(.*\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$", k)
    if m:
        per_expert.setdefault(m.group(1), {}).setdefault(int(m.group(2)), {})[m.group(3)] = raw.pop(k)
for base, ex in per_expert.items():
    n = max(ex) + 1
    raw[base + ".gate_up_proj"] = mx.stack([mx.concatenate([ex[e]["gate_proj"], ex[e]["up_proj"]], 0)
                                            for e in range(n)])
    raw[base + ".down_proj"] = mx.stack([ex[e]["down_proj"] for e in range(n)])

weights = {}
for k, v in raw.items():
    if k.startswith("model."):
        k = "model.language_model." + k[len("model."):]
    # One engram part is saved unindexed; mlx-vlm names parts by number.
    k = k.replace(".ngram_embedding.weight", ".ngram_embedding.shard_0.weight")
    weights[k] = v

vlm_cfg = {
    "model_type": "qwen4_exp",
    "text_config": dict(text_cfg, model_type="qwen4_exp_text"),
    "vision_config": {"model_type": "qwen4_exp", "depth": 1, "hidden_size": 32,
                      "intermediate_size": 64, "num_heads": 2, "in_channels": 3,
                      "patch_size": 16, "spatial_merge_size": 2, "temporal_patch_size": 2,
                      "out_hidden_size": text_cfg["hidden_size"], "num_position_embeddings": 16},
    "eos_token_id": text_cfg["eos_token_id"],
    "vocab_size": text_cfg["vocab_size"],
}
model = Model(ModelConfig.from_dict(vlm_cfg))
weights = model.sanitize(weights)
lm_params = {k for k, _ in tree_flatten(model.parameters()) if not k.startswith("vision_tower.")}
missing = sorted(lm_params - set(weights))
if missing:
    sys.exit(f"the toy does not cover the MLX model: missing {missing[:8]}")
model.load_weights(list(weights.items()), strict=False)

# ---- quantise and save, as mlx_vlm.convert does ------------------------------
model, qcfg = quantize_model(model, vlm_cfg, 32, 4, quant_predicate=model.quant_predicate)
out = args.out
if os.path.exists(out):
    shutil.rmtree(out)
save_weights(out, model)
save_config(qcfg, os.path.join(out, "config.json"))
shutil.copy(os.path.join(hf, "generation_config.json"), out)
q = qcfg["quantization"]
print(f"saved {out}: default {({k: v for k, v in q.items() if not isinstance(v, dict)})}, "
      f"{sum(isinstance(v, dict) for v in q.values())} per-module overrides")

# ---- mlx-vlm's outputs for it, loaded back from disk ------------------------
model = load_model(Path(out))
model.update(tree_map(lambda a: a.astype(mx.float32) if a.dtype == mx.bfloat16 else a,
                      model.parameters()))
lm = model.language_model
# 32 tokens with a repeat (so n-grams recur) and an EOS at 13 (so the engram's
# segment reset shows), past the tiny indexer's budget so QSA selects.
import torch
g = torch.Generator().manual_seed(args.prompt_seed)
tokens = torch.randint(0, vlm_cfg["vocab_size"] - 8, (32,), generator=g).tolist()
tokens[3:6] = tokens[20:23]
eos = text_cfg["eos_token_id"]
tokens[13] = eos if isinstance(eos, int) else eos[0]

captured = []
orig = lang.Qwen4ExpDecoderLayer.__call__
def capture(self, *a, **k):
    h = orig(self, *a, **k)
    captured.append(h)
    return h
lang.Qwen4ExpDecoderLayer.__call__ = capture
full = lm(mx.array([tokens])).logits[0].astype(mx.float32)
lang.Qwen4ExpDecoderLayer.__call__ = orig
hidden = [np.array(h[0].astype(mx.float32)).tolist() for h in captured]

cache = lm.make_cache()
step = []
for t in tokens:
    step.append(lm(mx.array([[t]]), cache=cache).logits[0, -1].astype(mx.float32))
step = mx.stack(step)
print(f"prefill vs step-by-step max |diff| = {mx.abs(full - step).max().item():.3e}")

json.dump({
    "tokens": tokens,
    "vocab": vlm_cfg["vocab_size"],
    "logits_full": np.array(full).tolist(),
    "logits_step": np.array(step).tolist(),
    "hidden_per_layer": hidden,
    "source": "mlx-vlm, fp32, on this directory's own weights",
}, open(os.path.join(out, "oracle.json"), "w"))
print(f"wrote {out}/oracle.json: {len(tokens)} tokens, {len(hidden)} layers")
shutil.rmtree(work)
