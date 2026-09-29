## 14. Open questions

- **Bundle the guest image or download it?** 320 MB is right at the line. A
  download needs a hosting story, a signature check, and a first-run wait;
  bundling makes the `.app` ~350 MB and the notarisation upload slow. Leaning
  bundle, for offline-first symmetry with the rest of the project.
- **VM per session, or per project?** Per session is isolated and costs
  `clonefile` (nothing) plus RAM when running (a lot). Per project shares state
  between conversations, which is sometimes what you want and sometimes a
  disaster. Currently: per session, with §6.5's cap. Revisit after M6.
- ~~**Elixir, or Erlang alone?**~~ **Elixir throughout the guest** (§7.3):
  one language and one build tool for warden and workspace, settled and no
  longer open.
- **Should agent tools be able to call the model?** A `Crucible.LLM.complete/1`
  available to hot-loaded modules would let the agent build sub-agents. It also
  re-enters the engine from inside a tool call, on a session that is mid-turn,
  on a queue that is already busy — and §2.1 says one at a time. It would need a
  second session and a budget. Deferred, deliberately, and noted here so it is
  not designed in by accident.
- ~~**Context management at 32K.**~~ **Decided: a machine-derived profile, and
  compaction is a successor session** (§2.3, §2.4). What remains open is the
  *policy*: at
  what fraction of the window does the app offer compaction, does it ever compact
  without asking, and does the handoff summary get written by the outgoing
  session (cheap, in-context) or by a second pass over the transcript (better,
  and it costs a whole prefill). Leaning: offer at 85%, never automatic, summary
  from the outgoing session.
- **Does the agent know how much context it has left?** It should — a model
  budgeting its own remaining window behaves differently, and better, than one
  that hits the wall. But the only appendable channel is a tool result, so this
  means every tool result carries a context footer, which costs tokens on every
  call to reduce the chance of wasting all of them. Measure the token cost
  before committing to it.
- **How much of the profile should the user be able to override?** Context and
  live-session count are the two dials, and they trade against each other
  legibly. Whether to expose the 0.85 working-set reserve is a different
  question: it is the one number standing between the machine and swapping 16 GB
  of weights, and a user who lowers it gets a mysteriously unusable app. Leaning:
  expose the two dials, keep the reserve internal, and let the dials refuse
  combinations that would breach it.
- **KV dtype.** The parent defers fp8 until there is a quality benchmark. Same
  answer here, but the app is where long sessions will actually make it hurt.
