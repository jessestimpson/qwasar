"""tools/bench/ttftbench's twin on mlx-vlm, as it ships: generate_step, BF16
activations, its default 2048-token prefill steps.  Loads the model (the
engram table left on disk, as tools/flashnext_real_oracle.py does), warms up
on 2,600 tokens and a few decode steps, then times a fresh prompt cache --
generate_step makes one when none is passed, and its prefix cache (APC) lives
only in the server and batch paths: TTFT is the time to the first yielded
token, then D more greedy tokens.  One JSON line.

Dev-only, and best run from an mlx-vlm install of its own rather than
tools/venv, which is pinned to the commit tests/fixtures were recorded on:

    python -m venv /tmp/mlxvlm && /tmp/mlxvlm/bin/pip install \
        git+https://github.com/Blaizzy/mlx-vlm.git mlx
    /tmp/mlxvlm/bin/python tools/bench/mlx_ttft.py <model> <ids-file> <N> <D>
"""
import json, sys, time
from pathlib import Path
import mlx.core as mx
from mlx_vlm.models.qwen4_exp.ple_storage import prepare_external_ple_model
from mlx_vlm.utils import load_model
from mlx_vlm.generate.ar import generate_step

src = Path(sys.argv[1]).resolve(); N = int(sys.argv[3]); D = int(sys.argv[4])
ids = [int(x) for x in open(sys.argv[2]).read().split()]
view = src.parent / (src.name + ".mlx-external-ple")
if not view.exists():
    prepare_external_ple_model(src, view)
t0 = time.time()
model = load_model(view)
mx.eval(model.parameters())
load_s = time.time() - t0

def run(prompt, d):
    x = mx.array([prompt])
    a = time.time(); first = None; toks = []
    for i, (tok, _) in enumerate(generate_step(x, model, None, None, max_tokens=d + 1, temperature=0.0)):
        toks.append(int(tok))
        if i == 0:
            first = time.time()
    end = time.time()
    return first - a, end - first, toks

w0 = 100000 % (len(ids) - 2600)
run(ids[w0:w0 + 2600], 4)          # warm-up
mx.clear_cache()
ttft, dec, toks = run(ids[:N], D)
print(json.dumps({"engine": "mlx-vlm", "n": N, "d": D, "load_s": round(load_s, 2), "ttft_s": round(ttft, 3),
                  "prefill_tps": round(N / ttft, 1), "decode_tps": round(D / dec, 2), "first": toks[:16]}))
