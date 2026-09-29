#!/usr/bin/env python3
"""A tokenizer.json for the toy Flash-Next checkpoints (tests/fixtures/).

Dev-only (PLAN.md 6).  The toys have a 256-entry vocabulary and no tokenizer,
which is fine for the kernel tests that feed them ids -- and useless for the
server, whose chat template needs <|im_start|>, <think> and friends.  This
writes a byte-level BPE tokenizer in the real one's format that fits the toy:
printable ASCII, tab, newline and carriage return one token each, the control
tokens the template resolves at ids under 256 (<|im_end|> at the toy config's
eos_token_id, 251), and every id left over spent on merges trained on the text
the toy will meet -- the template's own strings, the agent's tool schemas --
so a real prefix fits the toy's 4096-token window.

Non-ASCII input has no byte tokens and is dropped by the encoder; the API
tests stay in ASCII.  Standard library only.

    python3 tools/toy_tokenizer.py tests/fixtures/flashnext-tiny-q4 tests/fixtures/flashnext-tiny-mlx
"""
import json
import os
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


def corpus(root):
    """What the toy will be asked to encode: the template's own text (its
    string literals in qwasar_tokenizer.c), the agent's tool schemas and
    guidance (qwasar_agent.c), and some plain English.  The merges trained on
    it are what keep a real prefix inside the toy's 4096-token window."""
    import re
    parts = []
    for name in ("qwasar_tokenizer.c", "qwasar_agent.c"):
        try:
            with open(os.path.join(root, name), encoding="utf-8", errors="replace") as f:
                src = f.read()
        except OSError:
            continue
        for lit in re.findall(r'"((?:[^"\\]|\\.)*)"', src):
            lit = lit.encode().decode("unicode_escape", errors="replace")
            if len(lit) > 3:
                parts.append(lit)
    parts.append("You are a careful assistant who reads before answering. " * 3)
    parts.append("The quick brown fox jumps over the lazy dog. List the files. Read the file and "
                 "return its contents. Say hello. Once more, with feeling: what does this do?")
    return "\n".join(parts)


def pieces(text):
    """Roughly the engine's pre-tokenizer: a word with its leading space,
    runs of punctuation, runs of whitespace.  Merges never cross a piece."""
    import re
    return re.findall(r" ?[A-Za-z]+| ?[0-9]+| ?[^A-Za-z0-9\s]+|\s+", text)


def train_merges(text, u, budget):
    """Greedy byte-pair merging: the most frequent adjacent pair, `budget`
    times.  Symbols are byte-level strings, as the vocabulary spells them."""
    from collections import Counter
    words = Counter(pieces(text))
    seqs = {w: [u[b] for b in w.encode("utf-8") if b in u] for w in words}
    merges = []
    for _ in range(budget):
        pairs = Counter()
        for w, n in words.items():
            s = seqs[w]
            for a, b in zip(s, s[1:]):
                pairs[(a, b)] += n
        if not pairs:
            break
        (a, b), n = pairs.most_common(1)[0]
        if n < 2:
            break
        merges.append([a, b])
        ab = a + b
        for w in seqs:
            s = seqs[w]
            i, out = 0, []
            while i < len(s):
                if i + 1 < len(s) and s[i] == a and s[i + 1] == b:
                    out.append(ab)
                    i += 2
                else:
                    out.append(s[i])
                    i += 1
            seqs[w] = out
    return merges


def build(root):
    u = byte_to_unicode()
    # Only ASCII has byte tokens: the toy's 256 ids cannot hold all 256
    # bytes and the control tokens too.
    keep = {b: u[b] for b in [9, 10, 13] + list(range(0x20, 0x7F))}
    vocab = {}
    for b in keep:
        vocab[keep[b]] = len(vocab)
    controls = {
        240: "<|im_start|>", 241: "<think>", 242: "</think>", 243: "<tool_call>",
        244: "</tool_call>", 245: "<tool_response>", 246: "</tool_response>",
        247: "<|vision_start|>", 248: "<|vision_end|>", 249: "<|image_pad|>",
        250: "<|endoftext|>", 251: "<|im_end|>", 252: "<|video_pad|>",
    }
    budget = min(controls) - len(vocab)
    merges = train_merges(corpus(root), keep, budget)
    for l, r in merges:
        if l + r not in vocab:
            vocab[l + r] = len(vocab)
    assert max(vocab.values()) < min(controls), max(vocab.values())
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
    }, len(merges)


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tok, n_merges = build(root)
    for d in sys.argv[1:] or ["tests/fixtures/flashnext-tiny-q4", "tests/fixtures/flashnext-tiny-mlx"]:
        with open(os.path.join(root, d, "tokenizer.json"), "w") as f:
            json.dump(tok, f, indent=1, ensure_ascii=False)
        print(f"wrote {d}/tokenizer.json: {len(tok['model']['vocab'])} entries, {n_merges} merges")
