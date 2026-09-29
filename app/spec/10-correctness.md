## 10. Correctness

The parent tree's rule is CPU reference twins and golden vectors against
mlx-vlm. The app's equivalents:

**Bridge fidelity — the highest-value tests.** For a fixed conversation, the
Swift `ChatTemplate` marshalling must produce byte-identical token arrays to
`qwasar_apply_chat_template` driven from C. Generate goldens with a small C
harness in `Tests/`, commit them, and compare. A silently different system turn
is a silently different model, and it would be invisible until quality dropped
for no reason anyone could name.

**Round-trip.** `encode` → `token_bytes` → UTF-8 assembly must reproduce the
input for a corpus that includes emoji, CJK, combining marks, and a token that
splits a codepoint. This is the §3.4 bug, caught by a test rather than by a user
seeing `\u{FFFD}`.

**Speculation equivalence.** Same property `tests/test_verify` protects, one
level up: a session with MTP enabled and one without must emit identical token
sequences from the same prompt at `temperature = 0`. If Crucible's Swift
draft/verify loop breaks this, it is broken.

**The `.git` vault.** `test_mount_work.sh` pins the guest contract that
decides whether the user's repository can be damaged: the shadow is seeded
once per disk and never re-seeded over the agent's commits, the bind mount
is asked for on **every** boot with the shadow in place first, and non-git
or gitfile projects are left entirely alone. It gates the image build. The
sandbox gate then proves the mounted result end to end from the host side:
a guest write is in the mounted tree; a guest commit and a guest-written
hook leave the real `.git` without a new byte; HEAD never moves. This is
the security boundary; it gets the most tests in the project.

**Guest, warden — the control-plane invariants.** Numbered, because code
comments cite them by number:

1. Every `id` receives **exactly one** terminal response. Never zero, never two.
2. No response is delivered as `ok` after its request has been reported as
   timed out.
3. The registry has no duplicate tool names, and every registered name
   resolves to a loaded module.
4. After a workspace restart, the replayed manifest is order-preserving and
   contains exactly the modules that were loaded before the crash.
5. No monitor outlives the request that created it.
6. Warden never blocks on the workspace: the bridge remains responsive to a
   `cancel` while an `invoke` is outstanding.

Invariants 1, 2, 4 and 6 are held by the warden unit suite and the sandbox
gate (which causes the failures — a killed workspace, a request that outlives
its deadline — rather than reasoning about them). These are *when* bugs, so
the design rule that keeps them testable stands: no `receive ... after`
anywhere in warden — every deadline is a `Process.send_after` matched by
message identity, so a late reply is told apart from a timeout by identity
rather than by racing the clock.

**Guest, workspace.** ExUnit, ordinarily: the six file tools against a fixture
`/work`, the `Crucible.Skill` behaviour, JSON marshalling, malformed and
oversized frames. This half is logic, not timing, and does not need a simulator.

**`crucible-cli`.** A headless driver that loads the engine, runs one scripted
session end-to-end against a fixture project, and asserts on the final patch.
This is the CI target, and it is the only way to test the whole thing without a
UI harness. It matches the parent's culture: everything is runnable from a
terminal.

**Context ceiling.** A session driven to exactly `context` tokens must produce a
compaction offer (§2.4) and never an engine error. Drive it with a scripted
session in `crucible-cli` against a small context override, so the test costs
seconds rather than a 128K prefill.

**What is not automatable, and therefore gets measured by hand:** VM boot time,
first-launch kernel compile, cold 128K prefill, and — the two numbers §4.4 owes
this document — park and restore wall-clock at 8 GB. Record them here the way the
parent records its throughput numbers, and update them when they move.

---
