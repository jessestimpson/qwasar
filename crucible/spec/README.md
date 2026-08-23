# The Crucible spec

The design of Crucible, one topic per file, numbered so that section
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
| [07-agent.md](07-agent.md) | the tool surface, self-modification, the two nodes, materialisation |
| [08-security.md](08-security.md) | threat model, network (`fetch`), sandbox configuration layers, the config project |
| [10-correctness.md](10-correctness.md) | the test strategy and the control-plane invariants |
| [11-risks.md](11-risks.md) | interaction risks worth naming |
| [12-roadmap.md](12-roadmap.md) | what is built, what is next (histories: git log) |
| [13-rules.md](13-rules.md) | rules for this codebase |
| [14-open-questions.md](14-open-questions.md) | open questions |
| [15-delegation.md](15-delegation.md) | delegation: remote sub-agents, embedded, budgeted, steerable |

House rule, carried over from when this was one file: design decisions carry
their measurements, and a measurement that contradicts an assumption wins.
