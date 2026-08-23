## 15. Escalation: remote sub-agents, embedded and budgeted

*E1 and E2 built 2026-08-23; E3 not started. This section is the design;
the milestones are at the end of it.*

The premise: the local model is cheap, private, and always available, and it
is a 27B. Some problems deserve more model than that. So a session can
**escalate** — open a sub-agent against an OpenAI-compatible API (OpenRouter
is the reference target, so one key reaches many models) — under a budget the
sandbox configuration defines, with the API key invisible to every model
involved, and with the sub-agent running as an **embedded interactive
session inside the main transcript** rather than the fire-and-forget
subprocess most subagent tools are.

The shape in one sentence: *the local model decides when a problem is worth
paying for; the user watches the paid model work, can steer it mid-flight,
and the local model gets the result — all inside one conversation.*

### 15.1 The tool, and who calls it

One new tool on the sandboxed surface, advertised only when escalation is
configured (the `fetch` pattern from §8.3 — an unconfigured project's system
turn is unchanged):

```
delegate(task, context?, model?)
```

- `task` is the brief. `context` is optional supporting material the local
  model chooses to include — it is the local model's job to pack the problem,
  which is itself a skill worth prompting for.
- `model` picks from the configured allowlist; absent means the list's first
  entry. The tool description names the models AND their prices per million
  tokens, because the local model cannot weigh a trade it cannot see.
- The tool description states when to reach for it: a problem the local model
  has failed at twice, a design question with real stakes, a review before
  something irreversible. Escalation-as-first-resort burns budget on work the
  local model could do; the description says so plainly.

The HOST executes it, like `fetch`: the request never transits the guest, and
the guest stays network-less. The local session's turn blocks in the tool
call for the duration — the engine has nothing else to do, and "the local
model is waiting on the expensive model" is exactly the truthful state.

### 15.2 The embedded session

A delegation is a transcript item with an inside: a **sub-session card**
that streams the remote model's output (reasoning collapsed, same as local),
shows its tool calls as cards, carries a running cost meter — and has its
own input field.

- **The user can type into a live delegation.** A message typed there goes to
  the remote conversation as a user turn. This is the difference from every
  spawn-and-wait subagent design: the expensive model is steerable while it
  spends, by the person whose money it is spending. Watching a paid model
  head down the wrong alley with no way to intervene is the failure this
  exists to prevent.
- **The remote agent gets the same sandbox, by proxy.** Its tool calls
  (OpenAI tool-calling, the local agent's own schemas passed through) are
  executed by the host through the same vsock channel into the same guest,
  against the same `/work`. The sandbox already contains the blast radius of an untrusted
  agent — that argument does not care which model the agent is. Tool access
  is serialised with the local agent's by construction, since the local turn
  is blocked inside `delegate`.
- **Ending.** The delegation ends when the remote model stops, the budget
  trips, or the user stops it — and nothing else, by explicit decision:
  no wall-clock ceiling, no tool-step ceiling, no token cap. A delegation is
  **long-horizon by design**; its governors are the dollar budgets and the
  person watching the card, both of which see a looping agent rather than
  guessing at one. (E2 briefly shipped a 32-step ceiling; it was removed —
  a step count is a proxy for cost, and the cost is already governed
  directly.) Whatever ended it, the local model receives a tool result: the remote model's final
  message, plus a header stating model, tokens, cost, and how it ended.
  A budget-tripped delegation is reported as exactly that — the local model
  can act on a partial answer knowing it is partial.
- **Persistence.** The sub-session is part of the transcript record; a parked
  session replays with its delegations intact and readable. Delegations do
  not survive an app restart as *live* things — they are turns, not
  processes.

### 15.3 The budget is configuration, and it overlays

Three new `SandboxOverlay` keys, resolving exactly as §8.5 resolves
everything (field-wise, most specific non-nil wins, replace never merge):

| key | meaning |
|---|---|
| `agent_models` | allowlisted model ids, comma-separated; empty = escalation explicitly OFF, nil = fall through |
| `agent_budget_usd` | ceiling per session, summed across its delegations |
| `agent_turn_budget_usd` | ceiling per single delegation |

The same semantics carry the same weight: a global grant with a per-session
empty `agent_models` is how one confidential session opts out; a project
layer replaces the global model list rather than merging with it. The config
project manages these with the tools it already has — `config_set project
qwasar agent_models "anthropic/claude-...,openai/gpt-..."` — and
`config_show` reports provenance like any other key.

Cost accounting is the host's: OpenRouter returns token usage per response
and publishes per-model pricing; the host accumulates cost per delegation and
per session, streams it into the card's meter, and hard-stops at the
ceiling by declining to send the next request (a request in flight is paid
for; the meter says so). Actual spend is recorded on the SessionRecord, so
`config_show` can answer "what has this session cost".

### 15.4 The key never meets a model

The API key is the one secret in the system, and every path from it to a
model is closed by construction rather than by filtering:

- **Storage**: macOS Keychain, entered once through a host UI panel ("Set
  API key…", the model-directory pattern). Never in the store, never in
  `sandbox.json`, never in any overlay — the config project can *see whether*
  a key is set, never the key, because the config tools have no operation
  that returns it.
- **Use**: attached to requests by host code only, at one pinned base URL
  from the app's own configuration. The URL is NOT model-addressable: neither
  `delegate` nor any config key takes an endpoint, because "send my
  credentials to an attacker-chosen URL" is the exfiltration this closes.
  Changing providers is a host UI action.
- **Exposure**: never in a prompt, never in a tool result, never in the
  transcript, never in the guest (nothing of this transits the guest), and
  redacted from logs. The remote model is treated as no more trusted than
  the local one: it authenticates nothing and is told nothing worth
  stealing.
- What the remote model CAN exfiltrate is what the local one could with
  `fetch`: project contents, to wherever the provider is. That is §8.3's
  confidentiality trade again, wearing a different hat, and it is stated the
  same way — escalation off (the default) keeps today's guarantees; on, the
  project's contents reach the configured provider and no one else.

### 15.5 What the local model is told

The escalation prompt fragment (rendered into the system turn only when
configured, like the fetch fragment) carries three things: the models and
their prices, the budget remaining at session open, and the judgment rule —
escalate on genuine capability walls, not on effort. The interesting
research question this feature exists to explore is whether a 27B can learn
*calibrated self-doubt*: knowing which problems are beyond it cheaply enough
to be worth 1.5 t/s of overhead. The transcript makes that measurable —
every delegation records what the local model had tried first, and a
delegation whose result the local model could have produced is visible waste
with a dollar figure on it.

### 15.6 Milestones

- **E1 — consult.** *Built.* Text-in/text-out delegation, no remote tools:
  budget keys, keychain panel, cost metering, the embedded card with live
  streaming and mid-flight user input. `EscalationSuite` scripts the provider
  through a URLProtocol stub and pins the budget rule, the steering
  semantics, and the key's absence from every model-visible surface. One
  semantic E1 settled that the design left open: after each completed
  response the conversation holds open for a **grace window** (10 s) —
  a queued user message continues it, stop or silence ends it — because
  "ends the instant the model stops" and "waits forever" both fail the
  person mid-sentence. Field use immediately found the window's weak
  spot: reading the answer takes longer than 10 s, so a single-response
  delegation offered no real chance to steer. So **typing holds the
  window open** (the card's non-empty draft, bounded at 180 s so an
  abandoned draft cannot hold the local turn forever), and the card says
  the window exists while it runs — an invisible countdown is not an
  opportunity.
- **E2 — the remote agent works.** *Built.* The inner executor's surface
  proxied to the remote model -- its schemas are already OpenAI function
  schemas, so the proxy is a translation loop: streamed tool-call fragments
  assembled, executed through the same chain into the same `/work`, results
  fed back correlated by id. Three decisions E2 settled: the remote model
  gets the *whole* inner surface, not just the guest ten -- if the project
  granted `fetch`, the remote agent fetches under the same allowlist,
  because "the same rules as the local agent" is the statement a person can
  reason about; and nested `delegate` is refused as a tool result (the
  wrapper appends itself after inner, so it is never advertised, and a model
  that guesses the name is told no). Steering drains between tool steps
  too -- no grace wait while the model is mid-work. A tool-step ceiling
  shipped here and was later removed at the user's direction (see §15.2's
  Ending): long horizon is the point, and the budget governs cost directly
  where a step count only guessed at it. All pinned by
  `EscalationSuite` against the scripted provider, including the E2 gate
  shape: the remote model writes a file and its result returns in the next
  request.
- **E3 — judgment.** Prompt-fragment tuning for when to escalate, spend
  reporting in `config_show`, and a measured week of use: delegations that
  paid vs. delegations the local model could have done, with dollar figures.

Open questions, recorded now: whether `delegate` should be callable by the
remote agent (nested escalation — priced trees of models; deferred, refused
in E1/E2); and whether budget exhaustion should park the session or merely
disable the tool (disable, initially — a session that stops mid-thought
because money ran out is worse than one that says so and continues locally).

**"Escalate this" — the user's own affordance, built.** A header button, live
whenever the session could escalate — including *mid-turn*, which is the
design driver: a model visibly stuck in a bad line of reasoning is exactly
when the person watching knows before the model does. Escalating then
interrupts the turn (the button says so). The sheet takes the brief, a model
pick, and — on by default — the conversation tail: the last user message and
everything the local model produced since, *reasoning included*, because
"here is what it tried" is most of what the expert needs. The delegation runs
in the same card with the same steering and the same budget accounting; it
is **consult-only** (no remote tools — the session's tool chain belongs to
the local turn, whose state mid-interrupt is exactly what should not be
driven around). The answer then rides along with the user's next message —
shown as a discardable chip above the composer, attached to the prompt but
not duplicated in the display — so the local model sees it as context, which
closes the loop: the user escalates, the expert answers, the local model
continues with the answer in hand.
