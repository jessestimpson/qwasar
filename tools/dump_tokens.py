#!/usr/bin/env python3
"""Dump reference tokenizations for tests/tokens.json.

DEV ONLY.  Not part of the build.

  reference/mlx-vlm/.venv/bin/python tools/dump_tokens.py <model-dir> <out.json>
"""
import json
import sys

from mlx_vlm import load

CASES = [
    "",
    "hello",
    " hello",
    "hello world",
    "Hello, World!",
    "I don't think it's Bob's.",
    "THEY'RE HERE",
    "2+2=4",
    "1234567890",
    "  leading and  double  spaces  ",
    "trailing   ",
    "line one\nline two\n\n\nline three",
    "tabs\tand\tmore\ttabs",
    "\n\n  \n\t\n",
    "def foo(x): return x ** 2  # comment",
    "https://example.com/a/b?c=d&e=f",
    "café naïve résumé",
    "日本語のテキストです",
    "Пример текста на русском",
    "emoji 🎉 and 🚀 mixed with text",
    "mixed 漢字 and ASCII 123 together",
    "<|im_start|>not actually special<|im_end|>",
    "a" * 200,
    "The quick brown fox jumps over the lazy dog. " * 5,
]

CHATS = [
    {
        "name": "simple",
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "kwargs": {"add_generation_prompt": True},
    },
    {
        "name": "with_system",
        "messages": [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "Hi"},
        ],
        "kwargs": {"add_generation_prompt": True},
    },
    {
        "name": "no_thinking",
        "messages": [{"role": "user", "content": "Hi"}],
        "kwargs": {"add_generation_prompt": True, "enable_thinking": False},
    },
    {
        "name": "low_effort",
        "messages": [{"role": "user", "content": "Hi"}],
        "kwargs": {"add_generation_prompt": True, "reasoning_effort": "low"},
    },
    {
        "name": "medium_effort",
        "messages": [{"role": "user", "content": "Hi"}],
        "kwargs": {"add_generation_prompt": True, "reasoning_effort": "medium"},
    },
    {
        "name": "multi_turn",
        "messages": [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer", "reasoning_content": "thinking here"},
            {"role": "user", "content": "Second question"},
        ],
        "kwargs": {"add_generation_prompt": True},
    },
]


def main(model_dir, out_path):
    _, processor = load(model_dir)
    tok = processor.tokenizer

    out = {"cases": [], "chats": []}
    for text in CASES:
        out["cases"].append({"text": text, "ids": tok.encode(text)})

    for chat in CHATS:
        ids = tok.apply_chat_template(chat["messages"], tokenize=True, **chat["kwargs"])
        # Newer transformers hands back a BatchEncoding rather than a bare list.
        if hasattr(ids, "keys") and "input_ids" in ids:
            ids = ids["input_ids"]
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        while isinstance(ids, list) and len(ids) == 1 and isinstance(ids[0], list):
            ids = ids[0]
        out["chats"].append({"name": chat["name"], "ids": list(map(int, ids))})
        print(f"  {chat['name']:16s} {len(ids)} tokens")

    with open(out_path, "w") as f:
        json.dump(out, f)
    print(f"wrote {out_path}: {len(out['cases'])} cases, {len(out['chats'])} chats")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
