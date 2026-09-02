#!/usr/bin/env python3
"""Reference outputs for a Qwen4-Exp checkpoint, from `transformers` itself.

Dev-only (PLAN.md 6). Runs the HF model -- ideally the dequantised copy the
converter wrote, so the values match what the engine holds -- on a fixed
token sequence and records what the engine must reproduce:

  * logits at every position from one prefill (fp32);
  * logits from the same sequence fed one token at a time through the
    cache, which is the decode path and exercises every recurrent state;
  * the 4-stream hidden state of the last token after each decoder layer,
    so a mismatch can be localised to a layer rather than guessed at.

The prompt is long enough for the QSA indexer to select blocks, and it
contains an EOS mid-sequence so the PLE's segment-reset shows.

    tools/venv/bin/python tools/flashnext_oracle.py <hf_dir> <out.json> [prompt_seed]
"""
import sys, json
import torch
from transformers import Qwen4ExpForCausalLM, Qwen4ExpTextConfig

src, dst = sys.argv[1], sys.argv[2]
prompt_seed = int(sys.argv[3]) if len(sys.argv) > 3 else 7
torch.manual_seed(0)
model = Qwen4ExpForCausalLM.from_pretrained(src, dtype=torch.float32, attn_implementation="eager")
model.eval()
cfg = model.config
eos = cfg.eos_token_id if isinstance(cfg.eos_token_id, int) else cfg.eos_token_id[0]

# 32 tokens: a repeating structure (so n-grams recur), an EOS at 13, and
# enough length past the 11-visible-token threshold for the tiny indexer's
# budget of 8 to leave blocks unselected.
# The seed is a parameter because a prompt can land on a tie in the
# indexer's block scores, which torch.topk breaks in an unspecified order;
# tests/test_flashnext refuses such a fixture, and the answer is a new seed.
g = torch.Generator().manual_seed(prompt_seed)
tokens = torch.randint(0, cfg.vocab_size - 8, (32,), generator=g).tolist()
tokens[3:6] = tokens[20:23]
tokens[13] = eos
ids = torch.tensor([tokens])

# The residual after each decoder layer, captured by hook: the model's own
# `hidden_states` output swaps the last layer's residual for the post-mixer
# state, which is not the same tensor.
per_layer = [None] * len(model.model.layers)      # [layer][T][hc*hidden]
routes = [None] * len(model.model.layers)         # [layer][T][K] expert ids
def capture(i):
    def hook(_m, _inp, out):
        per_layer[i] = (out[0] if isinstance(out, tuple) else out)[0].float().tolist()
    return hook
def capture_route(i):
    def hook(_m, _inp, out):
        routes[i] = out[2].reshape(-1, out[2].shape[-1]).tolist()   # (logits, scores, indices)
    return hook
masks = [None] * len(model.model.layers)          # [layer][T][T] 0/1, QSA layers only
def capture_mask(i):
    def hook(_m, _inp, out):
        m = out[0, 0]                                   # [T, kv_len], bool or additive float
        m = (m == 0) if m.is_floating_point() else m
        masks[i] = m.to(torch.int32).tolist()
    return hook
handles = []
for i, layer in enumerate(model.model.layers):
    handles.append(layer.register_forward_hook(capture(i)))
    handles.append(layer.mlp.gate.register_forward_hook(capture_route(i)))
    if hasattr(layer, "self_attn"):
        handles.append(layer.self_attn.indexer.register_forward_hook(capture_mask(i)))

with torch.no_grad():
    full = model(ids, use_cache=False)
    logits_full = full.logits[0].float()
    # The hooks have done their job; the decode loop below must not
    # overwrite the prefill's captures with single-row ones.
    for h in handles:
        h.remove()
    print(f"captured {len(per_layer)} layers x {len(per_layer[0])} positions x {len(per_layer[0][0])}, "
          f"routes {len(routes[0])} x {len(routes[0][0])}")

    step_logits = []
    past = None
    for i, t in enumerate(tokens):
        o = model(torch.tensor([[t]]), past_key_values=past, use_cache=True)
        past = o.past_key_values
        step_logits.append(o.logits[0, -1].float())
    logits_step = torch.stack(step_logits)

diff = (logits_full - logits_step).abs().max().item()
print(f"prefill vs step-by-step max |diff| = {diff:.3e}")

json.dump({
    "tokens": tokens,
    "vocab": cfg.vocab_size,
    "logits_full": logits_full.tolist(),
    "logits_step": logits_step.tolist(),
    "hidden_per_layer": per_layer,
    "routes_per_layer": routes,
    "masks_per_layer": masks,
}, open(dst, "w"))
print(f"wrote {dst}: {len(tokens)} tokens x {cfg.vocab_size} vocab, {len(per_layer)} layers")
