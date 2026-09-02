#!/usr/bin/env python3
"""A toy Qwen4-Exp (Flash-Next family) checkpoint, for the engine's tests.

Dev-only (PLAN.md 6: tools/ is never required to build, test, or run).
Builds a seeded random `Qwen4ExpForCausalLM` at a size the M4 Air runs in
milliseconds, with every architectural feature of the real model switched
on -- four residual streams, MoE with a shared expert, Gated DeltaNet with
the sigmoid gate, QSA with an indexer whose budget is small enough that a
32-token prompt actually SELECTS blocks, and one PLE layer with a hashed
n-gram table -- and saves it as an HF safetensors checkpoint.

Every "in" dimension is a multiple of 64 because that is the engine's
quantisation group, the head geometry is the real model's because the
production attention and delta-rule kernels are compiled for it, and the QSA budget is 8 tokens (2 blocks of 4) so
sparse selection engages after 11 visible tokens.

    tools/venv/bin/python tools/flashnext_tiny.py tests/fixtures/flashnext-tiny
"""
import sys, json, os
import torch
from transformers import Qwen4ExpTextConfig, Qwen4ExpForCausalLM

out = sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/flashnext-tiny"
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1

cfg = Qwen4ExpTextConfig(
    vocab_size=256,
    hidden_size=64,
    num_hidden_layers=8,
    full_attention_interval=4,          # 3 linear : 1 sparse, twice
    num_attention_heads=4,
    num_key_value_heads=1,
    head_dim=256,                       # the production attention kernel is built for 256
    rms_norm_eps=1e-6,
    hidden_act="silu",
    output_gate_type="sigmoid",
    attention_bias=False,
    # gated delta: 2 key heads x 128, 4 value heads x 128, conv 4
    linear_num_key_heads=2,
    linear_key_head_dim=128,            # and the delta-rule kernel for 128 x 128
    linear_num_value_heads=4,
    linear_value_head_dim=128,
    linear_conv_kernel_dim=4,
    # moe: 8 experts, top 2, 64-wide experts and shared expert
    num_experts=8,
    num_experts_per_tok=2,
    moe_intermediate_size=64,
    shared_expert_intermediate_size=64,
    norm_topk_prob=True,
    # hyper-connections
    hc_count=4,
    hc_lowrank=64,
    # QSA indexer: 16 q heads x 128, 1 k head, budget 8 tokens in blocks of 4.
    # Sixteen rather than the real model's four: a block's score is a sum of
    # ReLU'd dots, so with h heads it is exactly zero with probability 2^-h
    # and two such blocks at the cut are a tie torch.topk breaks in an
    # unspecified order.  At 2^-16 the fixture is decisive.
    indexer_n_heads=16,
    indexer_kv_heads=1,
    indexer_head_dim=128,
    indexer_budget=8,
    indexer_compress_ratio=4,
    # PLE on layer 2 (one-indexed): 16 heads x 4 dims, tiny prime vocabs
    ple_layer_ids=[2],
    ple_embed_dim=64,
    ple_conv_kernel_size=4,
    ngram_size=3,
    heads_per_ngram=8,
    ngram_vocab_size_base=101,
    make_ngram_vocab_size_divisible_by=128,
    seed=1234,
    split_ngram_parts=1,
    # rope: partial 0.25 of 256 = 64 dims, interleaved mrope like the real one
    max_position_embeddings=4096,
    rope_parameters={"rope_type": "default", "rope_theta": 10000000.0,
                     "partial_rotary_factor": 0.25,
                     "mrope_section": [11, 11, 10], "mrope_interleaved": True},
    tie_word_embeddings=False,
    bos_token_id=250,
    eos_token_id=251,
    pad_token_id=250,
    dtype="bfloat16",
)

torch.manual_seed(seed)
model = Qwen4ExpForCausalLM(cfg)
# Random init leaves the (1+w) norms at exactly 1 and the PLE conv at zero,
# which would hide a wrong norm or a wrong conv. Perturb every parameter so
# nothing in the model is an identity.
with torch.no_grad():
    for name, p in model.named_parameters():
        if p.dtype.is_floating_point:
            p.add_(torch.randn_like(p) * 0.05)
    # Decisive routing and selection.  With every weight small and random the
    # router's probabilities are near-uniform and the indexer's block scores
    # near-tied, so top-k flips on rounding noise between two correct
    # implementations -- a fixture that fails for reasons that are not bugs.
    for name, p in model.named_parameters():
        if name.endswith("mlp.gate.weight") or name.endswith("index_qk_proj.weight"):
            p.mul_(30.0)
model = model.to(torch.bfloat16)
model.eval()

os.makedirs(out, exist_ok=True)
model.save_pretrained(out, safe_serialization=True)
cfg.save_pretrained(out)

n = sum(p.numel() for p in model.parameters())
print(f"saved {out}: {n} parameters")
for name, p in model.named_parameters():
    print(f"  {str(p.dtype):15s} {str(list(p.shape)):24s} {name}")
for name, b in model.named_buffers():
    if "multipliers" in name or "vocab_sizes" in name or "offsets" in name:
        print(f"  buffer {name} = {b.tolist()}")
