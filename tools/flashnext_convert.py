#!/usr/bin/env python3
"""HF Qwen4-Exp (Flash-Next) checkpoint -> qwasar's on-disk format.

Dev-only (PLAN.md 6). mlx-lm has no qwen4_exp, so there is no sanitised
checkpoint to lean on the way there was for the 27B; this does what the MLX
converter did there, in the engine's own naming:

  * every linear is MLX-affine 4-bit, group 64: nibble-packed U32 weights
    [out, in/8] plus BF16 scales and biases [out, in/64], w = s*q + b.
    Expert banks stay 3-D, quantised along the input axis, so expert e is a
    contiguous slice of the bank. Router gates, block-inject weights and
    every 1-D tensor stay BF16 -- the router in particular decides top-k,
    and 4-bit there would be a quality trade nobody has measured.
  * `(1 + w)` RMSNorms get the +1 folded in, as MLX did; the gated
    `linear_attn.norm` does not, as before.
  * the depthwise conv taps are transposed to [C, K, 1], as MLX did.
  * the PLE n-gram table stays in row shards and BF16, in files of its own
    marked `placement: cpu` that the engine maps without a GPU buffer: its
    rows are 160 wide, not a multiple of 64, it is read a few rows per token
    off a cold mmap, and at 102 GB it could not be one device buffer anyway.
  * the output is split into files under --shard-bytes with an index, since
    Metal caps a single buffer far below the model's size.
  * the I64 hash buffers are dropped; the engine derives them from config
    and the Max cross-checks the derivation against them.

Two outputs: the engine directory, and (with --dequant) an HF checkpoint
holding exactly the values the engine will see, dequantised, so the oracle
and the engine can be compared to fp32 tolerance rather than to
quantisation noise. Names inside the HF copy are the originals.

    tools/venv/bin/python tools/flashnext_convert.py <hf_dir> <out_dir> [--dequant <hf_out_dir>]
"""
import sys, os, json, re, glob, argparse
import torch
from safetensors.torch import load_file, save_file

GROUP = 64
BITS = 4

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("dst")
ap.add_argument("--dequant", default=None)
ap.add_argument("--shard-bytes", type=int, default=8 * 1024 ** 3,
                help="split the output into files of at most this many bytes (Metal caps one buffer well below the model)")
ap.add_argument("--engram-parts", type=int, default=0,
                help="re-split the engram table into this many row shards (0: keep the checkpoint's)")
args = ap.parse_args()

cfg = json.load(open(os.path.join(args.src, "config.json")))
tc = cfg.get("text_config", cfg)
if cfg.get("model_type", tc.get("model_type", "")) not in ("qwen4_exp", "qwen4_exp_text"):
    sys.exit(f"not a qwen4_exp checkpoint: {cfg.get('model_type')}")

# ---- the tensors, all of them, in memory one shard at a time ----------------
tensors = {}
for shard in sorted(glob.glob(os.path.join(args.src, "*.safetensors"))):
    tensors.update(load_file(shard))
print(f"read {len(tensors)} tensors")


def quantize(w: torch.Tensor):
    """MLX affine: per group of 64 along the last axis, q in [0,15]."""
    w = w.float()
    *lead, n = w.shape
    assert n % GROUP == 0, f"in-dim {n} not a multiple of {GROUP}"
    g = w.reshape(*lead, n // GROUP, GROUP)
    wmax = g.amax(-1, keepdim=True)
    wmin = g.amin(-1, keepdim=True)
    scale = (wmax - wmin) / (2 ** BITS - 1)
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    # Round-trip through BF16 so the scale and bias the engine reads are
    # exactly the ones used to pick the nibbles.
    scale = scale.to(torch.bfloat16).float()
    bias = wmin.to(torch.bfloat16).float()
    q = torch.clamp(torch.round((g - bias) / scale), 0, 15).to(torch.int64)
    deq = (scale * q + bias).reshape(*lead, n)
    q = q.reshape(*lead, n // 8, 8)
    packed = torch.zeros(*lead, n // 8, dtype=torch.int64)
    for i in range(8):
        packed |= q[..., i] << (4 * i)
    packed = packed.to(torch.int32)          # safetensors has no U32; bits are bits
    return (packed.contiguous(),
            scale.reshape(*lead, n // GROUP).to(torch.bfloat16).contiguous(),
            bias.reshape(*lead, n // GROUP).to(torch.bfloat16).contiguous(),
            deq)


# ---- naming ------------------------------------------------------------------
# HF: model.language_model.layers.N... (VLM) or model.layers.N... (text-only)
# Ours: language_model.model.layers.N...  (the 27B's MLX layout, kept)
def rename(name: str) -> str:
    if name.startswith("model.language_model."):
        return "language_model.model." + name[len("model.language_model."):]
    if name.startswith("model.visual."):
        return "vision_tower." + name[len("model.visual."):]
    if name.startswith("model."):
        return "language_model.model." + name[len("model."):]
    if name == "lm_head.weight":
        return "language_model.lm_head.weight"
    return name                                # mtp.*


PLUS_ONE_NORMS = re.compile(
    r"(\.q_norm|\.k_norm|\.q_layernorm|\.k_layernorm|\.hc_norm|\.norm_key|\.norm_query"
    r"|\.norm_conv|pre_fc_norm_hidden|pre_fc_norm_embedding)\.weight$")
KEEP_BF16 = re.compile(
    r"(\.mlp\.gate\.weight|\.shared_expert_gate\.weight|\.block_inject_weight\.weight"
    r"|\.conv1d\.weight|\.A_log|\.dt_bias|ngram_embedding)")
DROP = re.compile(r"(layer_multipliers|ngram_heads_vocab_sizes|ngram_heads_offsets)$")
NGRAM_SHARD = re.compile(r"^(.*\.ngram_embedding)\.shard_(\d+)\.weight$")
PER_EXPERT = re.compile(r"^(.*\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")

# transformers saves the expert banks in one of two layouts: fused 3-D
# (`experts.gate_up_proj` [E, 2I, H], `experts.down_proj` [E, H, I] -- the
# real checkpoint) or one 2-D tensor per expert (`experts.N.gate_proj` --
# what save_pretrained wrote for the toy).  The engine wants banks; stack.
per_expert = {}
for name in list(tensors):
    m = PER_EXPERT.match(name)
    if m:
        per_expert.setdefault(m.group(1), {}).setdefault(int(m.group(2)), {})[m.group(3)] = tensors.pop(name)
for base, experts in per_expert.items():
    n = max(experts) + 1
    gate_up = torch.stack([torch.cat([experts[e]["gate_proj"], experts[e]["up_proj"]], dim=0)
                           for e in range(n)])
    down = torch.stack([experts[e]["down_proj"] for e in range(n)])
    tensors[base + ".gate_up_proj"] = gate_up
    tensors[base + ".down_proj"] = down
    print(f"stacked {n} experts under {base}: gate_up {tuple(gate_up.shape)}, down {tuple(down.shape)}")

out = {}
deq = {}
ngram_parts = {}
q_bytes = 0
for name in sorted(tensors):
    t = tensors[name]
    if DROP.search(name):
        deq[name] = t                        # the oracle's copy keeps its buffers
        continue
    m = NGRAM_SHARD.match(name)
    if m:
        ngram_parts.setdefault(m.group(1), []).append((int(m.group(2)), t))
        continue
    new = rename(name)
    if new.startswith("vision_tower."):
        out[new] = t.to(torch.bfloat16)      # unquantised, like the 27B's tower
        deq[name] = t
        continue
    if t.ndim == 1 or KEEP_BF16.search(name):
        v = t.to(torch.bfloat16)
        if name.endswith(".conv1d.weight"):
            v = v.transpose(1, 2).contiguous()          # [C,1,K] -> [C,K,1]
        elif PLUS_ONE_NORMS.search(name):
            v = (t.float() + 1.0).to(torch.bfloat16)
            # The fold rounds (1+w) to bf16; the oracle must apply the SAME
            # rounded weight, so its raw w is the folded value minus one.
            deq[name] = v.float() - 1.0
            out[new] = v
            continue
        out[new] = v
        deq[name] = t
        continue
    # everything else is a linear (2-D) or an expert bank (3-D): quantise
    packed, scales, biases, dq = quantize(t)
    base = new[:-len(".weight")] if new.endswith(".weight") else new
    out[base + ".weight"] = packed
    out[base + ".scales"] = scales
    out[base + ".biases"] = biases
    deq[name] = dq                            # fp32: exactly what the engine computes
    q_bytes += packed.numel() * 4 + scales.numel() * 2 + biases.numel() * 2

# The engram table stays in row shards, BF16, in files of its own that the
# engine maps without a GPU buffer: it is read by row on the host, and at
# 102 GB it could not be one device buffer anyway.
engram = {}
for base, parts in ngram_parts.items():
    parts.sort()
    for i, p in parts:
        deq[f"{base}.shard_{i}.weight"] = p
    if args.engram_parts > 0:
        table = torch.cat([p for _, p in parts], dim=0)
        rows = table.shape[0]
        per = (rows + args.engram_parts - 1) // args.engram_parts
        parts = [(i, table[i * per:(i + 1) * per]) for i in range(args.engram_parts) if i * per < rows]
    for i, p in parts:
        engram[f"{rename(base)}.shard_{i}.weight"] = p.to(torch.bfloat16).contiguous()
    print(f"engram table {base}: {sum(p.shape[0] for _, p in parts)} rows in {len(parts)} shards")


def nbytes(t):
    return t.numel() * t.element_size()


def pack(tensors, limit):
    """Greedy split of a name->tensor dict into files under `limit` bytes,
    one tensor per file when a tensor alone exceeds it."""
    files, cur, cur_b = [], {}, 0
    for name in sorted(tensors):
        b = nbytes(tensors[name])
        if cur and cur_b + b > limit:
            files.append(cur); cur, cur_b = {}, 0
        cur[name] = tensors[name]; cur_b += b
    if cur: files.append(cur)
    return files


os.makedirs(args.dst, exist_ok=True)
weight_map = {}
model_files = pack(out, args.shard_bytes)
for i, group in enumerate(model_files):
    fname = f"model-{i + 1:05d}-of-{len(model_files):05d}.safetensors"
    save_file(group, os.path.join(args.dst, fname), metadata={"format": "qwasar"})
    for k in group: weight_map[k] = fname
engram_files = pack(engram, args.shard_bytes) if engram else []
for i, group in enumerate(engram_files):
    fname = f"engram-{i + 1:05d}-of-{len(engram_files):05d}.safetensors"
    save_file(group, os.path.join(args.dst, fname), metadata={"format": "qwasar", "placement": "cpu"})
    for k in group: weight_map[k] = fname
json.dump({"metadata": {"total_size": sum(nbytes(t) for t in out.values()) + sum(nbytes(t) for t in engram.values())},
           "weight_map": weight_map},
          open(os.path.join(args.dst, "model.safetensors.index.json"), "w"), indent=1)
print(f"wrote {len(model_files)} model file(s) and {len(engram_files)} engram file(s)")
qcfg = dict(cfg)
qcfg["quantization"] = {"bits": BITS, "group_size": GROUP, "mode": "affine"}
json.dump(qcfg, open(os.path.join(args.dst, "config.json"), "w"), indent=2)
for extra in ("tokenizer.json", "tokenizer_config.json", "generation_config.json",
              "chat_template.jinja", "merges.txt", "vocab.json"):
    p = os.path.join(args.src, extra)
    if os.path.exists(p):
        os.system(f"cp '{p}' '{args.dst}/'")
print(f"wrote {args.dst}: {len(out)} tensors, {q_bytes / 1e6:.1f} MB quantised")

if args.dequant:
    # Unstack the banks again so the dequantised checkpoint has exactly the
    # layout it was read in; from_pretrained maps either, but only names it
    # has seen before are guaranteed not to report as missing.
    for base in per_expert:
        gu = deq.pop(base + ".gate_up_proj")
        dn = deq.pop(base + ".down_proj")
        inter = gu.shape[1] // 2
        for e in range(gu.shape[0]):
            deq[f"{base}.{e}.gate_proj.weight"] = gu[e, :inter]
            deq[f"{base}.{e}.up_proj.weight"] = gu[e, inter:]
            deq[f"{base}.{e}.down_proj.weight"] = dn[e]
    os.makedirs(args.dequant, exist_ok=True)
    save_file({k: v.contiguous() for k, v in deq.items()},
              os.path.join(args.dequant, "model.safetensors"), metadata={"format": "pt"})
    dcfg = dict(cfg)
    dcfg["dtype"] = "float32"           # the dequantised values are fp32 and must stay so
    json.dump(dcfg, open(os.path.join(args.dequant, "config.json"), "w"), indent=2)
    print(f"wrote {args.dequant}: the same values, dequantised, HF names")
