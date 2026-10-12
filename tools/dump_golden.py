#!/usr/bin/env python3
"""Dump reference activations for qwasar's regression test.

DEV ONLY.  Not part of the build and not required to run qwasar -- it drives the
mlx-vlm reference implementation in reference/mlx-vlm to produce golden values
that tests/test_forward.c replays through the C engine.

  reference/mlx-vlm/.venv/bin/python tools/dump_golden.py <model-dir> <out.bin> [long]

tests/golden.bin is the short prompt.  tests/golden_long.bin is `long`: a
prompt over 256 tokens, so the engine prefills it in two chunks, both wider
than QW_QMM_MIN_ROWS -- the short one never leaves the batched matvec and so
never reaches a prefill matmul at all.

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
PROMPT_LONG = (
    "<|im_start|>user\n"
    "Read the passage and answer the question after it.\n\n"
    "The lighthouse at the end of the northern breakwater was built in 1874 from "
    "granite quarried on the island itself.  For its first forty years it burned "
    "colza oil, then paraffin, and in 1921 it was converted to acetylene, which "
    "allowed the keepers to be withdrawn; a single attendant rowed out once a week "
    "to change the cylinders.  Electricity arrived only in 1958, by a cable laid "
    "along the breakwater during the same works that widened the harbour mouth.  "
    "The original Fresnel lens, a second-order apparatus made in Paris, was removed "
    "in 1987 and is now displayed in the town museum, where visitors can stand "
    "inside it.  Its replacement is a rotating beacon of the ordinary kind, visible "
    "for eighteen nautical miles in clear weather, flashing white twice every ten "
    "seconds.  The tower itself is twenty-three metres tall and painted in red and "
    "white bands, a scheme adopted in 1902 after a collier mistook it in fog for "
    "the plain white tower on the headland to the south and ran aground on the "
    "shoal between them.  No lives were lost, but the cargo was, and the inquiry "
    "that followed recommended the bands, a fog bell, and a second light on the "
    "shoal itself; only the first two were carried out.  The fog bell was replaced "
    "by a diaphone in 1931, and the diaphone was switched off for good in 2005.\n\n"
    "Question: list, in order, every fuel or power source the light has used, and "
    "say which change let the keepers leave.<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n"
)
CAPTURE = [0, 1, 2, 3, 7, 31, 63]


def main(model_dir, out_path, which="short"):
    model, processor = load(model_dir)
    lm = model.language_model
    tok = processor.tokenizer

    ids = tok.encode(PROMPT_LONG if which == "long" else PROMPT)
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
    main(*sys.argv[1:4])
