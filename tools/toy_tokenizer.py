#!/usr/bin/env python3
"""A tokenizer.json for the toy Flash-Next checkpoints (tests/fixtures/).

Dev-only (PLAN.md 6).  The toys have a 256-entry vocabulary and no tokenizer,
which is fine for the kernel tests that feed them ids -- and useless for the
server, whose chat template needs <|im_start|>, <think> and friends.  This
writes a byte-level BPE tokenizer in the real one's format that fits the toy:
printable ASCII, tab, newline and carriage return one token each, a handful
of merges so the merge loop runs, and the control tokens the template resolves
at ids under 256, with <|im_end|> at the toy config's eos_token_id (251).

Non-ASCII input has no byte tokens and is dropped by the encoder; the API
tests stay in ASCII.  Standard library only.

    python3 tools/toy_tokenizer.py tests/fixtures/flashnext-tiny-q4 tests/fixtures/flashnext-tiny-mlx
"""
import json
import sys


def byte_to_unicode():
    """GPT-2's byte-level mapping: printable Latin-1 maps to itself, the rest
    to code points from 256 up, in order."""
    keep = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    table = {b: chr(b) for b in keep}
    n = 0
    for b in range(256):
        if b not in table:
            table[b] = chr(256 + n)
            n += 1
    return table


def build():
    u = byte_to_unicode()
    vocab = {}
    for b in [9, 10, 13] + list(range(0x20, 0x7F)):
        vocab[u[b]] = len(vocab)
    # A few merges, so the merge loop is exercised on the way through the
    # template's own text ("system", "user", newlines).
    merges = [["s", "y"], ["e", "m"], ["u", "s"], ["e", "r"], ["t", "h"], [u[10], u[10]]]
    for l, r in merges:
        vocab[l + r] = len(vocab)
    controls = {
        240: "<|im_start|>", 241: "<think>", 242: "</think>", 243: "<tool_call>",
        244: "</tool_call>", 245: "<tool_response>", 246: "</tool_response>",
        247: "<|vision_start|>", 248: "<|vision_end|>", 249: "<|image_pad|>",
        250: "<|endoftext|>", 251: "<|im_end|>", 252: "<|video_pad|>",
    }
    assert max(vocab.values()) < min(controls)
    return {
        "version": "1.0",
        "normalizer": None,
        "pre_tokenizer": None,
        "added_tokens": [
            {"id": i, "content": s, "single_word": False, "lstrip": False,
             "rstrip": False, "normalized": False, "special": True}
            for i, s in sorted(controls.items())
        ],
        "model": {"type": "BPE", "vocab": vocab, "merges": merges},
    }


if __name__ == "__main__":
    for d in sys.argv[1:] or ["tests/fixtures/flashnext-tiny-q4", "tests/fixtures/flashnext-tiny-mlx"]:
        with open(f"{d}/tokenizer.json", "w") as f:
            json.dump(build(), f, indent=1, ensure_ascii=False)
        print(f"wrote {d}/tokenizer.json")
