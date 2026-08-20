#!/usr/bin/env python3
"""Dump reference activations for qwasar's regression test.

DEV ONLY.  Not part of the build and not required to run qwasar -- it drives the
mlx-vlm reference implementation in reference/mlx-vlm to produce golden values
that tests/test_forward.c replays through the C engine.

  reference/mlx-vlm/.venv/bin/python tools/dump_golden.py <model-dir> <out.bin>

Layout (little endian):
  magic "QWGOLD01", int32 n_tokens, int32 hidden, int32 vocab, int32 n_capture
  int32 tokens[n_tokens]
  int32 capture_layer[n_capture]
  float32 hidden_state[n_capture][hidden]     -- last token, after that layer
  float32 final_norm[hidden]                  -- last token
  float32 logits[vocab]                       -- last token
"""
import struct
import sys

import mlx.core as mx
from mlx_vlm import load

PROMPT = "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n<think>\n"
CAPTURE = [0, 1, 2, 3, 7, 31, 63]


def main(model_dir, out_path):
    model, processor = load(model_dir)
    lm = model.language_model
    tok = processor.tokenizer

    ids = tok.encode(PROMPT)
    x = mx.array([ids])

    sink = []
    cache = lm.make_cache()
    normed = lm.model(x, cache=cache, capture_layer_ids=CAPTURE, hidden_sink=sink)
    logits = lm.lm_head(normed)
    mx.eval(normed, logits, *sink)

    hidden = normed.shape[-1]
    vocab = logits.shape[-1]
    print(f"tokens={len(ids)} hidden={hidden} vocab={vocab} captured={len(sink)}")

    def f32(a):
        return mx.array(a).astype(mx.float32).tolist()

    with open(out_path, "wb") as f:
        f.write(b"QWGOLD01")
        f.write(struct.pack("<4i", len(ids), hidden, vocab, len(CAPTURE)))
        f.write(struct.pack(f"<{len(ids)}i", *ids))
        f.write(struct.pack(f"<{len(CAPTURE)}i", *CAPTURE))
        for h in sink:
            f.write(struct.pack(f"<{hidden}f", *f32(h[0, -1])))
        f.write(struct.pack(f"<{hidden}f", *f32(normed[0, -1])))
        f.write(struct.pack(f"<{vocab}f", *f32(logits[0, -1])))

    last = logits[0, -1].astype(mx.float32)
    top = mx.argsort(-last)[:5].tolist()
    print("top5:", [(t, tok.decode([t]), round(float(last[t]), 3)) for t in top])
    print("wrote", out_path)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
