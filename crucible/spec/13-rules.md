## 13. Rules for this codebase

Inherited from the parent, which inherited them from ds4, plus what this app adds:

- **No Python, no C++.** Unchanged.
- **Keep `CQwasar` narrow.** `qwasar.h` and `qwasar_toolcall.h`. Nothing else is
  exported to Swift, ever. If Swift needs to know something about a tensor, the
  answer is a new function in `qwasar.h`, not a wider module map.
- **Never call `qwasar_*` off the engine queue.** One queue, no exceptions, and
  a debug assertion on the queue's specific key in every wrapper.
- **The profile decides; nothing is hardcoded.** Context size and live-session
  count come from §2.3's derivation at launch. No constant in the source, no
  code path that assumes one live session, and no feature that only works on a
  big machine. The 32 GB target is the case to write against; the 128 GB case is
  the one to not break.
- **Live count is never parallelism.** One session runs at a time (§2.1). Any UI
  or API that implies otherwise is wrong.
- **Warden deadlines are messages, never clock races.** No `receive ... after`
  anywhere in warden: a deadline is a `Process.send_after` matched by identity,
  and a test of timing behaviour causes the failure (kill the node, outlive the
  deadline) rather than sleeping and hoping (§10's invariants).
- **The host runs no model-requested tool.** If a feature seems to need one,
  it does not; it needs a guest tool and a patch.
- **No `--yes` for materialisation.** Sandbox writes are unconfirmed; the
  boundary crossing never is.
- **Correctness before speed**, and never keep a faster path with unexplained
  drift. Same rule, one layer up: the speculation equivalence test in §10 is the
  same promise `tests/test_verify` makes.
- **The C tree is built by `make`.** Xcode shells out to it. Do not teach Xcode
  about `bin2c`.
- **Comment the mechanics, not the syntax.** Session lifetimes, the purge rules,
  the checkpoint prefix contract, and the path validation are the four places a
  reader needs the *why*. Everything else should be obvious.
- **A green simulation reports its coverage.** Seeds, ops, schedules explored.
  `PASS` on its own is a claim this project does not make.
- **Every tool result is capped, and says so.** Prefill runs at ~32 tok/s
  (§2.5), so a result costs about a second per 100 bytes. A tool that can return
  an unbounded amount of text is a tool that can spend ten minutes of the user's
  time on one call. Cap it, and make the truncation message tell the model what
  to do instead — that message is part of the tool's contract, like `edit`'s
  match rule.
- Small, sharp, readable. No slop.

---
