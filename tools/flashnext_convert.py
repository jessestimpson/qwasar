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
  * the I64 hash buffers are kept as they are.  The engine derives the hash
    from config and, when these are present, refuses to load if its
    derivation differs from the checkpoint's own -- the one place the real
    hash constants can be checked.

It streams.  The real input is 360 GB and the machine converting it has
128 GB, so tensors are opened lazily (safetensors' safe_open), quantised one
at a time -- on --device mps when there is one -- and written out as each
output file fills; nothing is held but the output file being filled.

With --fetch REPO it also fits the disk.  The input is not downloaded
first: shards are fetched a few ahead of the one being converted, each is
converted into output files of its own, and deleted.  Peak disk is the
output (~172 GB) plus the shards in flight, instead of input plus output
(~530 GB).  Progress is recorded per input shard, so an interrupted run
picks up where it stopped.  Checkpoint tensors never span files, and the
real checkpoint's expert banks are fused, so every shard converts alone.

    tools/venv/bin/python tools/flashnext_convert.py <work_dir> <out_dir> \\
        --fetch Qwen/Qwen3.8-Flash-Next --device mps [--prefetch 3]

Two outputs: the engine directory, and (with --dequant; toy inputs only, it
is fp32 and one file) an HF checkpoint holding exactly the values the
engine will see, so the oracle and the engine can be compared to fp32
tolerance rather than to quantisation noise. Names inside it are the
originals.

    tools/venv/bin/python tools/flashnext_convert.py <hf_dir> <out_dir> \\
        [--dequant <hf_out_dir>] [--shard-bytes N] [--engram-parts P] [--device mps]
"""
import sys, os, json, re, glob, argparse, shutil
import torch
from safetensors import safe_open
from safetensors.torch import save_file

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
ap.add_argument("--device", default="cpu", help="where to quantise: cpu, or mps on Apple silicon")
ap.add_argument("--fetch", default=None, metavar="REPO",
                help="download from this Hugging Face repo into src, converting and deleting shard by shard")
ap.add_argument("--prefetch", type=int, default=3, help="with --fetch: shards downloaded ahead")
ap.add_argument("--shards", default=None,
                help="with --fetch: convert only these input shards (comma-separated numbers), for a trial")
args = ap.parse_args()
dev = torch.device(args.device)

EXTRAS = ("tokenizer.json", "tokenizer_config.json", "generation_config.json",
          "chat_template.jinja", "merges.txt", "vocab.json")

if args.fetch:
    from huggingface_hub import hf_hub_download
    os.makedirs(args.src, exist_ok=True)
    for f in ("config.json", "model.safetensors.index.json") + EXTRAS:
        try:
            hf_hub_download(args.fetch, f, local_dir=args.src)
        except Exception as ex:                          # not every repo has every extra
            if f in ("config.json", "model.safetensors.index.json"):
                raise
            print(f"  (no {f}: {type(ex).__name__})")

cfg = json.load(open(os.path.join(args.src, "config.json")))
tc = cfg.get("text_config", cfg)
if cfg.get("model_type", tc.get("model_type", "")) not in ("qwen4_exp", "qwen4_exp_text"):
    sys.exit(f"not a qwen4_exp checkpoint: {cfg.get('model_type')}")

# ---- lazy handles: which input file holds which tensor -----------------------
handles = {}
where = {}
if not args.fetch:
    for shard in sorted(glob.glob(os.path.join(args.src, "*.safetensors"))):
        h = safe_open(shard, framework="pt")
        handles[shard] = h
        for k in h.keys():
            where[k] = shard
    print(f"{len(where)} tensors in {len(handles)} input file(s)")


def get(name):
    return handles[where[name]].get_tensor(name)


def quantize(w):
    """MLX affine: per group of 64 along the last axis, q in [0,15]."""
    w = w.to(dev).float()
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
    q = torch.clamp(torch.round((g - bias) / scale), 0, 15).to(torch.int32)
    deq = (scale * q + bias).reshape(*lead, n).cpu() if args.dequant else None
    q = q.reshape(*lead, n // 8, 8)
    packed = torch.zeros(*lead, n // 8, dtype=torch.int32, device=dev)
    for i in range(8):
        packed |= q[..., i] << (4 * i)   # safetensors has no U32; bits are bits
    return (packed.cpu().contiguous(),
            scale.reshape(*lead, n // GROUP).to(torch.bfloat16).cpu().contiguous(),
            bias.reshape(*lead, n // GROUP).to(torch.bfloat16).cpu().contiguous(),
            deq)


# ---- naming ------------------------------------------------------------------
# HF: model.language_model.layers.N... (VLM) or model.layers.N... (text-only)
# Ours: language_model.model.layers.N...  (the 27B's MLX layout, kept)
def rename(name):
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
    r"|\.conv1d\.weight|\.A_log|\.dt_bias)")
HASH_BUFFERS = re.compile(r"(layer_multipliers|ngram_heads_vocab_sizes|ngram_heads_offsets)$")
NGRAM_SHARD = re.compile(r"^(.*\.ngram_embedding)\.shard_(\d+)\.weight$")
PER_EXPERT = re.compile(r"^(.*\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")

# transformers saves the expert banks in one of two layouts: fused 3-D
# (`experts.gate_up_proj` [E, 2I, H], `experts.down_proj` [E, H, I] -- the
# real checkpoint) or one 2-D tensor per expert (`experts.N.gate_proj` --
# what save_pretrained writes for the toy).  The engine wants banks, so the
# per-expert layout is stacked on the way through under the fused names.
per_expert = {}
ngram_parts = {}
names = []
for name in sorted(where):
    m = PER_EXPERT.match(name)
    if m:
        per_expert.setdefault(m.group(1), {}).setdefault(int(m.group(2)), {})[m.group(3)] = name
        continue
    m = NGRAM_SHARD.match(name)
    if m:
        ngram_parts.setdefault(m.group(1), []).append((int(m.group(2)), name))
        continue
    names.append(name)
for base in per_expert:
    names += [base + ".gate_up_proj", base + ".down_proj"]
names.sort()


def load(name):
    """The tensor under a (possibly fused) name."""
    for base, experts in per_expert.items():
        n = max(experts) + 1
        if name == base + ".gate_up_proj":
            return torch.stack([torch.cat([get(experts[e]["gate_proj"]), get(experts[e]["up_proj"])], 0)
                                for e in range(n)])
        if name == base + ".down_proj":
            return torch.stack([get(experts[e]["down_proj"]) for e in range(n)])
    return get(name)


def nbytes(t):
    return t.numel() * t.element_size()


class Writer:
    """Fills output files up to --shard-bytes and saves each as it fills."""
    def __init__(self, prefix, metadata):
        self.prefix, self.metadata = prefix, metadata
        self.cur, self.cur_b, self.done = {}, 0, []

    def add(self, name, t):
        b = nbytes(t)
        if self.cur and self.cur_b + b > args.shard_bytes:
            self.flush()
        self.cur[name] = t
        self.cur_b += b

    def flush(self):
        if not self.cur:
            return
        fname = f"{self.prefix}-{len(self.done) + 1:05d}.safetensors"
        save_file(self.cur, os.path.join(args.dst, fname), metadata=self.metadata)
        print(f"  {fname}: {len(self.cur)} tensors, {self.cur_b / 1e9:.2f} GB")
        self.done.append((fname, list(self.cur)))
        self.cur, self.cur_b = {}, 0

    def finish(self):
        """Saves the last file and renames them all NNNNN-of-MMMMM."""
        self.flush()
        total = len(self.done)
        final = []
        for i, (fname, keys) in enumerate(self.done):
            new = f"{self.prefix}-{i + 1:05d}-of-{total:05d}.safetensors"
            os.rename(os.path.join(args.dst, fname), os.path.join(args.dst, new))
            final.append((new, keys))
        return final


os.makedirs(args.dst, exist_ok=True)
deq = {}
q_bytes = 0


def emit(name, t, add):
    """Converts one checkpoint tensor and hands the results to `add`."""
    global q_bytes
    new = rename(name)
    if HASH_BUFFERS.search(name):
        add(new, t.contiguous())                              # I64, as stored
        if args.dequant: deq[name] = t
        return
    if new.startswith("vision_tower."):
        add(new, t.to(torch.bfloat16).contiguous())           # unquantised, like the 27B's tower
        if args.dequant: deq[name] = t
        return
    if t.ndim == 1 or KEEP_BF16.search(name):
        v = t.to(torch.bfloat16)
        if name.endswith(".conv1d.weight"):
            v = v.transpose(1, 2)                             # [C,1,K] -> [C,K,1]
        elif PLUS_ONE_NORMS.search(name):
            v = (t.float() + 1.0).to(torch.bfloat16)
            # The fold rounds (1+w) to bf16; the oracle must apply the SAME
            # rounded weight, so its raw w is the folded value minus one.
            if args.dequant: deq[name] = v.float() - 1.0
            add(new, v.contiguous())
            return
        add(new, v.contiguous())
        if args.dequant: deq[name] = t
        return
    # everything else is a linear (2-D) or an expert bank (3-D): quantise
    packed, scales, biases, dq = quantize(t)
    base = new[:-len(".weight")] if new.endswith(".weight") else new
    add(base + ".weight", packed)
    add(base + ".scales", scales)
    add(base + ".biases", biases)
    if args.dequant: deq[name] = dq                           # fp32: exactly what the engine computes
    q_bytes += nbytes(packed) + nbytes(scales) + nbytes(biases)


def write_index_and_config(files):
    weight_map = {}
    for fname in files:
        with safe_open(os.path.join(args.dst, fname), framework="pt") as h:
            for k in h.keys():
                weight_map[k] = fname
    total = sum(os.path.getsize(os.path.join(args.dst, f)) for f in files)
    json.dump({"metadata": {"total_size": total}, "weight_map": weight_map},
              open(os.path.join(args.dst, "model.safetensors.index.json"), "w"), indent=1)
    qcfg = dict(cfg)
    qcfg["quantization"] = {"bits": BITS, "group_size": GROUP, "mode": "affine"}
    json.dump(qcfg, open(os.path.join(args.dst, "config.json"), "w"), indent=2)
    for extra in EXTRAS:
        p = os.path.join(args.src, extra)
        if os.path.exists(p):
            shutil.copy(p, args.dst)
    return total


if args.fetch:
    # ---- shard by shard: fetch ahead, convert, delete ------------------------
    import concurrent.futures, time
    index = json.load(open(os.path.join(args.src, "model.safetensors.index.json")))
    inputs = sorted(set(index["weight_map"].values()))
    if any(PER_EXPERT.match(k) for k in index["weight_map"]):
        sys.exit("--fetch needs fused expert banks; this checkpoint stores experts one by one")
    progress_path = os.path.join(args.dst, "convert-progress.json")
    done = json.load(open(progress_path))["done"] if os.path.exists(progress_path) else []
    todo = [f for f in inputs if f not in done]
    if args.shards:
        want = {int(x) for x in args.shards.split(",")}
        todo = [f for f in todo if int(re.search(r"-(\d+)-of-", f).group(1)) in want]
    print(f"{len(inputs)} input shards, {len(done)} already converted, {len(todo)} to go")

    def fetch(f):
        return hf_hub_download(args.fetch, f, local_dir=args.src)

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.prefetch))
    pending = {}
    started = time.time()
    for i, f in enumerate(todo):
        for g in todo[i:i + args.prefetch + 1]:
            if g not in pending:
                pending[g] = pool.submit(fetch, g)
        path = pending.pop(f).result()
        stem = f[:-len(".safetensors")]
        model_t, engram_t = {}, {}
        with safe_open(path, framework="pt") as h:
            for name in sorted(h.keys()):
                m = NGRAM_SHARD.match(name)
                if m:
                    engram_t[f"{rename(m.group(1))}.shard_{int(m.group(2))}.weight"] = \
                        h.get_tensor(name).to(torch.bfloat16).contiguous()
                    continue
                emit(name, h.get_tensor(name), lambda k, v: model_t.__setitem__(k, v))
        # Written under a temporary name and renamed, so a file that exists is
        # a file that is whole; the progress record comes last of all.
        for prefix, tensors, meta in (("model", model_t, {"format": "qwasar"}),
                                      ("engram", engram_t, {"format": "qwasar", "placement": "cpu"})):
            if not tensors:
                continue
            out = os.path.join(args.dst, f"{prefix}-{stem}.safetensors")
            save_file(tensors, out + ".part", metadata=meta)
            os.replace(out + ".part", out)
        done.append(f)
        json.dump({"done": done}, open(progress_path + ".part", "w"))
        os.replace(progress_path + ".part", progress_path)
        os.remove(path)
        el = time.time() - started
        print(f"[{len(done)}/{len(inputs)}] {f}: {len(model_t)} model + {len(engram_t)} engram tensors, "
              f"{el / 60:.1f} min elapsed, ~{el / (i + 1) * (len(todo) - i - 1) / 60:.0f} min to go", flush=True)
        del model_t, engram_t
    pool.shutdown()
    if args.shards:
        print("trial run: no index written")
        sys.exit(0)
    files = sorted(f for f in os.listdir(args.dst)
                   if f.endswith(".safetensors") and f.startswith(("model-", "engram-")))
    total = write_index_and_config(files)
    print(f"wrote {args.dst}: {len(files)} files, {total / 1e9:.1f} GB")
    sys.exit(0)

model_w = Writer("model", {"format": "qwasar"})
for name in names:
    emit(name, load(name), model_w.add)
model_files = model_w.finish()

# The engram table: row shards, BF16, in host-only files.
engram_w = Writer("engram", {"format": "qwasar", "placement": "cpu"})
for base, parts in ngram_parts.items():
    parts.sort()
    if args.dequant:
        for i, n in parts: deq[n] = get(n)
    if args.engram_parts > 0:
        table = torch.cat([get(n) for _, n in parts], dim=0)
        rows = table.shape[0]
        per = (rows + args.engram_parts - 1) // args.engram_parts
        pieces = [(i, table[i * per:(i + 1) * per]) for i in range(args.engram_parts) if i * per < rows]
    else:
        pieces = [(i, get(n)) for i, n in parts]
    for i, p in pieces:
        engram_w.add(f"{rename(base)}.shard_{i}.weight", p.to(torch.bfloat16).contiguous())
    print(f"engram table {base}: {sum(p.shape[0] for _, p in pieces)} rows in {len(pieces)} shards")
engram_files = engram_w.finish()

write_index_and_config([f for f, _ in model_files + engram_files])
print(f"wrote {args.dst}: {len(model_files)} model file(s), {len(engram_files)} engram file(s), "
      f"{q_bytes / 1e9:.3f} GB quantised")

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
