#!/usr/bin/env python3
"""Reference tokenizations for text with combining marks (tests/tokens_marks.json).

DEV ONLY.  Flash-Next's tokenizer differs from the 27B's in exactly one
place: its split regex lets combining marks (\\p{M}) ride with the letters
they attach to.  Same vocab, same merges.  These cases are the ones where
that shows, dumped with the `tokenizers` library from a tokenizer.json.

    tools/venv/bin/python tools/dump_tokens_marks.py <tokenizer.json> <out.json>

Run it against both tokenizers to see the ids differ; the fixture checked in
is from Flash-Next's.
"""
import json
import sys
from tokenizers import Tokenizer

CASES = [
    "école",                              # e + combining acute
    "Việt Nam",                     # decomposed Vietnamese
    "नमस्ते दुनिया",                            # Devanagari vowel signs, virama
    "สวัสดีครับ",                                # Thai vowels and tone marks
    "مَرْحَبًا بِالعَالَم",                     # Arabic with harakat
    "שָׁלוֹם",                                   # Hebrew points
    "বাংলা ভাষা",                                 # Bengali
    "தமிழ் மொழி",                                 # Tamil
    "❤️ and 👍🏽 and 👨‍👩‍👧",                     # variation selector, modifier, ZWJ
    "äb̀c",                          # marks between plain letters
    "x́ 1́ -́",                  # a mark after a letter, a digit, punctuation
    "́leading mark",
    "mixed café é ascii 123",
    "\u1112\u1161\u11ab\u1100\u1173\u11af",        # 한글 as decomposed jamo
    "한글 composed",
    "A\u030a ring, e\u0323\u0302 dot-then-hat, e\u0302\u0323 hat-then-dot",   # ordering
    "ﬁ ligature and Ⅻ stay as they are",                   # compatibility forms: not NFC's business
]

tok = Tokenizer.from_file(sys.argv[1])
cases = [{"text": t, "ids": tok.encode(t, add_special_tokens=False).ids} for t in CASES]
json.dump({"cases": cases}, open(sys.argv[2], "w"), ensure_ascii=False, indent=0)
print(f"wrote {sys.argv[2]}: {len(cases)} cases")
for c in cases:
    print(f"  {c['text']!r}: {c['ids']}")
