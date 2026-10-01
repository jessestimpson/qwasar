# The Qwasar.app spec

The design of the coding agent's window of Qwasar.app — called Crucible when
most of this was written, and still so named below — one topic per file, numbered so that section
references stay stable: a citation like `PLAN.md 2.2` or `§7.3` in code
comments resolves to the file whose name carries that number (`02-…` §2.2,
`07-…` §7.3). The numbering has one deliberate gap — §9 was the eta
deterministic-simulation experiment, removed with its feature; its surviving
invariants live in §10.

| file | subject |
|---|---|
| [00-overview.md](00-overview.md) | what Crucible is, and the two ideas it runs on |
| [01-scope.md](01-scope.md) | scope |
| [02-constraints.md](02-constraints.md) | the three runtime constraints and the memory model |
| [03-architecture.md](03-architecture.md) | processes, modules, directory layout |
| [04-sessions.md](04-sessions.md) | sessions, projects, the scheduler, checkpoints |
| [05-interface.md](05-interface.md) | the window and the transcript |
| [06-sandbox.md](06-sandbox.md) | the guest: image, boot, disks, the native build |
| [07-agent.md](07-agent.md) | the tool surface, self-modification, the two nodes, the direct tree and the `.git` vault |
| [08-security.md](08-security.md) | threat model, network (`fetch`), sandbox configuration layers, the config project |
| [10-correctness.md](10-correctness.md) | the test strategy and the control-plane invariants |
| [11-risks.md](11-risks.md) | interaction risks worth naming |
| [12-roadmap.md](12-roadmap.md) | what is built, what is next (histories: git log) |
| [13-rules.md](13-rules.md) | rules for this codebase |
| [14-open-questions.md](14-open-questions.md) | open questions |
| [15-delegation.md](15-delegation.md) | delegation: remote sub-agents, embedded, budgeted, steerable |

House rule, carried over from when this was one file: design decisions carry
their measurements, and a measurement that contradicts an assumption wins.

**Since 2026-10-01 §2.4's successor exists, with running notes.** The
handoff is not written at the end: the outgoing session keeps notes as it
goes, in asides the server rolls back (API.md §4.11), after each reply.
The 85% offer, never automatic, is §14's leaning, built.

**Since 2026-09-30 the tool surface is Claude Code's** — Read, Write, Edit,
Glob, Grep, Bash, TodoWrite, written once in `ToolKit` over a host or guest
backend — and the system prompt is built by `SystemPrompt`, to match the
harness Qwen3.8 Flash-Next's agentic results were measured in. §7.1's frozen
C-agent surface is history.

**Since 2026-09-30 the sandbox is a per-session choice.** A new session's
tools run on the user's Mac (`HostToolRunner`, with their login shell's
environment); a sandboxed one's run in the guest as described below. The
app is no longer App Sandboxed (§8's threat model applies to sandboxed
sessions). See [`../README.md`](../README.md#2-you-choose-where-the-tools-run-per-session).

**Since 2026-09-29 the engine is out of process.** The app runs
`qwasar-server` as a helper and talks to it over the Session API
([`../../API.md`](../../API.md)); the renovation that did this, and what moved
where, is [`../../PLAN-qwasar.md`](../../PLAN-qwasar.md). Where a section
below says the engine is in-process, or sizes memory in the app, or parks
and restores sessions itself, read it as the server's job now: the
invariants (append-only sessions, a prefix that is exactly the system turn,
warmth claimed only when verified) are unchanged and are enforced there.
