## 7. The agent, and how it rewrites itself

### 7.1 The fixed tool surface

Twelve tools. This list is **frozen at build time** and never changes at
runtime, for the reason in §2.2: it is the system turn, and the system turn is
the prefix of everything. The count is allowed to grow during design and is
frozen at M4; what must never happen is a tool appearing because the model asked
for one.

| tool | runs where | notes |
|---|---|---|
| `read(path)` | guest `/work` | |
| `write(path, content)` | guest `/work` | no confirmation — it is a sandbox |
| `edit(path, old, new)` | guest `/work` | line-anchored, same contract as `qw_edit_apply`, reimplemented in Elixir |
| `list(path)` | guest `/work` | |
| `grep(pattern, path)` | guest `/work` | ripgrep |
| `shell(command)` | guest `/work` | unconfirmed; no network, no host |
| `elixir(code)` | guest workspace node | evaluate, with a persistent binding |
| `define(source)` | guest workspace node | compile + hot-load a module (§7.2) |
| `skills()` | guest warden | the registry of SKILLS — capabilities the agent wrote, as distinct from these fixed tools |
| `invoke(name, args)` | guest workspace node | call one of them (§7.3) |

Six of these are the C agent's own tools with their descriptions kept nearly
verbatim — those descriptions were written against this model's training and the
`edit` one in particular spells out its match rule because that rule *is* the
contract. Do not paraphrase them for the sake of paraphrasing them.

The last four are the new idea, and they are where all the leverage is: they
let the agent change what it can do, and find out whether the change is
correct **before** it commits to it.

### 7.2 `define` — compiling a SKILL into a live node

**The vocabulary, deliberate:** the ten in-context schemas are *tools* —
given, fixed, selected by the model from its list. What `define` produces is
a **skill** — made, open-ended, reached through `invoke`, owned by the
project. Overloading "tool" for both confused the model and the user alike
about which list a thing lived on; the two words now carry the distinction
everywhere: schemas, briefings, ops (`skills`), the behaviour
(`Crucible.Skill`), the library. Pre-release, so no legacy aliases survive.


```
define(source: """
  defmodule ASTGrep do
    @behaviour Crucible.Skill
    def name, do: "grep_ast"
    def schema, do: %{...}
    def run(%{"pattern" => p}), do: ...
  end
""")
```

What happens, in order, all inside the guest:

1. **Warden receives it.** Warden does not compile it. Warden `:erpc.call`s the
   *workspace* node, under a timeout, in a linked task.
2. **Compile.** `Code.compile_string/2` for Elixir; for Erlang source,
   `:erl_scan` → `:erl_parse` → `:compile.forms/2`. Compilation errors and
   warnings come back as text and become the tool result. A module that does not
   compile changes nothing — this is a pure function up to this point.
3. **Purge, carefully.** The BEAM holds two versions of a module: current and
   old. A third load purges old, and *purging kills any process still running
   old code*. So warden calls `:code.soft_purge/1` first. If it returns `false`,
   the load is **refused** and the tool result names the processes still
   executing the previous version. The model then decides — kill them, wait, or
   accept the loss with a `force: true` argument. Silently killing a process the
   agent started three steps ago is the kind of failure that costs an hour of
   confusion, and it is entirely avoidable.
4. **Flag concurrency.** If the compiled module spawns processes, implements
   `gen_server`/`gen_statem`/`supervisor`, or sends messages, warden says so
   in the tool result: correctness there is about ordering, and a module that
   loads green has proven nothing about it.
5. **Load.** `:code.load_binary(mod, ~c"crucible://#{name}.ex", bin)` on the
   workspace node.
6. **Register.** If the module implements `Crucible.Skill`, warden adds it to the
   registry: name, schema, source, load timestamp, version counter.
7. **Report.** The tool result names the module, the functions it exports, and
   whether it registered as a tool. The host also emits a *module load* card
   into the transcript (§5.2) with a diff against the previous version, because
   the agent changing its own behaviour is the single most interesting thing
   that happens in this application and it should not be buried in a tool result.

### 7.3 Two nodes, and why the warden is untouchable

```
warden  (Elixir, :warden@guest)      workspace (Elixir, :workspace@guest)
  Warden.Bridge    vsock port          Crucible.Skill (behaviour)
  Warden.Registry  tool manifest       Crucible.Shell (eval binding)
  Warden.Fs        /work primitives    Crucible.Fs    (the six file tools)
  Warden.Workspace spawn / monitor     ...every module the agent writes
```

They are **separate OS processes**, connected by Erlang distribution over
loopback inside the guest with a shared cookie. Not one node with three
supervision trees. Three reasons, and each one alone would be sufficient:

1. A module the agent hot-loads can crash the scheduler, exhaust the atom table,
   or take down the node. If that node also owns the vsock bridge, the harness
   goes silent and the user sees a hang.
2. `eta_sched` **serialises every process in the VM and takes over the clock**.
   A simulation running on the workspace node would freeze the tools, and one
   running on warden would freeze the bridge.
3. Warden's handling of a dead, slow, or partitioned workspace is the most
   failure-prone code in the guest, and putting workspace on the far side of
   distribution is what keeps that boundary explicit rather than accidental.

So:

- **Warden owns the wire and cannot be modified.** `define` requests naming a
  `Warden.*` module are rejected by name, and warden's code path is never
  `erpc`'d into. It is also the only node whose code is in the image and signed
  with it.
- **Workspace is disposable.** Warden monitors it. If it dies, warden restarts
  it and **replays the module manifest** in load order, then reports to the
  host, which reports to the model as a tool result: *the workspace restarted;
  N modules were reloaded; process state was lost.* The agent is told the truth
  and can recover. That sentence is a promise, and the unit suite plus the
  sandbox gate are how it is kept (§10 lists the invariants).
- **The skill library is the durable artefact, and it belongs to the PROJECT.**
  The host captures every successful `define` from its own event stream --
  the call carries the source, the result names the module -- and stores it
  as `Project.skillLibrary`: source keyed by module, first-definition order
  preserved (later modules may reference earlier ones), helper modules kept
  for the warden's own reason. At every session open, the library replays
  into the freshly booted guest before the first token, so a tool written
  once reaches every sibling session and survives parking, a VM stop, and an
  app relaunch. An earlier draft scoped this to `SessionRecord` -- but a tool
  is project knowledge, not VM state: the session whose guest happened to
  compile it first has no better claim to it than its siblings. Defines made
  by a delegated remote agent are captured the same way; a module that no
  longer compiles on replay is noted and skipped, never fatal. Removing or
  editing a library entry is a future config-project operation.

`invoke(name, args)` is a warden call: look up the name in the registry, `erpc`
into workspace's `mod.run(args)` under a timeout, marshal the return. A tool that
throws returns its exception and stacktrace as the tool result. A tool that
hangs is killed at the timeout and says so.

### 7.4 The tree, direct — and the vault over `.git` *(rebuilt 2026-08-31)*

*Third design at this number, each recorded in the git log. M5's
materialisation moved bytes under approval; the shared-workspace design
moved the tree to a host copy and history across a verified object
crossing. Living with the copy showed the remaining seam: the workspace and
the project were still two directories, so "the changes are right there"
was true of the copy and false of the project, and the crossing existed
only to bridge a gap the copy itself had created. Pre-release, so this
replaces both wholesale.*

**`/work` is the user's project directory**, mounted read-write into the
guest. There is no copy, no seeding, no export, no import, no `crucible/*`
branch — the agent's edit lands in the user's working tree as it happens,
where their own git shows it as an ordinary uncommitted change. Review is
`git diff` in their own repo; acceptance is their own `git commit`; undo is
their own `git checkout`. The integration surface is zero new concepts.

What survives of the second design is its best idea, applied at the only
boundary left: **the real `.git` is never guest-writable.**

```
host   project/                 the user's own tree — the agent edits it live
         src/…                  uncommitted changes, in their own git status
         .git/                  REAL: hooks, config, history — see below
guest  /work                    the same directory, virtiofs, read-write
       /var/crucible/git        the SHADOW: a private copy of .git on the VM
                                disk, bind-mounted over /work/.git every boot
```

At first boot, `mount-work` copies the real `.git` (visible through the
mount) onto the VM disk; on **every** boot, before the warden starts, it
bind-mounts that shadow over `/work/.git`. Git inside the guest is
completely ordinary — log, blame, diff, even commits, all against the
private copy — while the mountpoint physically stands between the agent and
the user's real `.git`: nothing written through `/work/.git` can reach it,
and even `rm -rf /work/.git/*` empties only the shadow. Hooks, config,
filters — the things the user's own git *executes* — cannot be planted.
The gate proves it from the host side: the guest commits and writes a hook,
and the real `.git` must not gain a byte.

Four consequences, stated:

- **The shadow is a snapshot, refreshed on request.** The agent's view of
  history is the session's first boot — until the user clicks **Refresh
  Git**: the `git_refresh` op drops the seed stamp (the one thing a running
  guest can do about a `.git` it cannot see), the session parks, and the
  next message boots a guest whose `mount-work` re-seeds from the repo's
  current state while nothing but init is running. Refresh discards the
  shadow's private commits and never touches the tree or the real `.git`;
  the transcript notes it, and the button is disabled mid-turn (a reboot
  under a running tool call would read as a crashed sandbox). The rejected
  alternative — unmounting the shadow on a live VM to copy fresh bytes —
  would put agent code and the real `.git` alive at the same time, which is
  the invariant, so it stays rejected.
- **The model is briefed away from git entirely, for now.** Version control
  is the user's: they review edits with their own tools and commit what
  they accept; the model's job is the files. The shadow's purpose is
  therefore pure enforcement — briefings are advisory and injection ignores
  them, so the guest's git *working* against a private copy is what makes a
  disobedient `git` command harmless rather than merely discouraged.
- **A non-git project gets nothing** — no shadow, no repo, and git in the
  guest says "not a repository", which is the truth. A `.git` *file* (a
  worktree or submodule checkout) is treated the same way, because the real
  git dir it points to is not reachable and must not be guessed at.
- **Sessions of one project share the tree.** Two sessions no longer have
  two copies; §4.3's one-live-session rule is what keeps them from typing
  over each other, and it is now load-bearing for the tree, not just for
  memory.

#### What this deletes

The entire crossing: `GitImport` and its suite (the loose-object writer,
the zlib framing, the SHA verification), `Warden.Git` and the
`git_info`/`git_export`/`git_objects` ops, the crossing sheet, the
automatic per-turn export, the `crucible/*` ref namespace, and the branch
walkthrough. Also the host-side workspace seeder and its suite, the
`gitseed` share, and the *Open Workspace* affordance — the workspace is the
project; the user already has it open.

#### Risks worth naming

- **The write boundary moved from the repo to the working tree.** A bad or
  injected agent can now vandalise uncommitted files directly. For tracked
  files the user's own git is the recovery (`git status` shows the damage,
  `git checkout` reverts it); **untracked files have no such net** — an
  overwritten untracked file is gone. This is the trade the direct tree
  buys collaboration with, chosen deliberately; §8.2 carries the honest
  threat-model statement.
- **The bind mount is load-bearing.** If it ever failed to attach on a git
  project, the real `.git` would be exposed; `mount-work` seeds the shadow
  *before* mounting, warns loudly on failure, and the gate asserts the
  mounted result. A failed seed still mounts an empty shadow — protection
  over functionality.
- **virtiofs write performance**, unchanged from the second design: builds
  that hammer the tree do it over the share; the escape hatch is build
  output on guest-local `/tmp`, and the number to watch is a real build's
  wall clock.
- **Concurrent edits are last-write-wins at the file level.** Now genuinely
  two hands on one checkout — the user's own working tree. Same statement,
  sharper teeth.

**Correctness bar:** `test_mount_work.sh` pins the guest contract — shadow
seeded once per disk and never re-seeded over the agent's commits *unless
the stamp was dropped* (the refresh path, which must discard private
commits and re-stamp), the bind mount asked for on every boot, non-git and
gitfile projects left alone — and gates the image build. The sandbox gate
proves the mounted result end to end: a guest write visible in the host
tree, a guest commit and a guest-written hook absent from the real `.git`,
HEAD unmoved, and `git_refresh` actually dropping the stamp.

### 7.5 Sampling, and speculation under it

Crucible samples by default — temperature 1.0, top-k 20, top-p 0.95, the
model's own `generation_config`, via `qwasar_sample` — because Qwen's guidance
for thinking-mode models warns against greedy decoding: long reasoning chains
under argmax are prone to repetition loops, and a harness whose every turn
opens with hundreds of reasoning tokens is the worst case for that failure,
not a neutral one. `temperature = 0` remains exactly greedy, and the headless
gates pin it, because a gate is a measurement and two runs of it must be
comparable. A nonzero `seed` reproduces a sampled run exactly.

**Speculation now runs under sampling, by rejection** — the Leviathan et al.
/ Chen et al. scheme, implemented as `qwasar_session_verify_sampled` in the
engine. The old conflict was a contract: `qwasar_session_verify` promises the
greedy sequence exactly and keeps that promise by comparing the target's
argmax against the draft, which under sampling is simply the wrong operation.
The sampled verify replaces the comparison with an acceptance test: draft `x`
is accepted with probability `min(1, p(x)/q(x))` against the target's
*filtered* distribution `p` (temperature, top-k, min-p, top-p — the same
chain serial sampling applies), the first rejection is replaced by a sample
from the normalised residual `max(0, p − q)`, and a fully accepted block ends
with a plain sample from the final row. Because the draft head proposes
deterministically, `q` is a point mass and the scheme simplifies: accept `x`
with probability `p(x)`, resample from `p` with `x` removed on a miss. What
comes out is distributed **exactly** as serial sampling — the law that
replaces greedy equality.

What made it cheap, against the section's earlier cost estimate: the verify
pass already materialises full logits for every row (`s->verify_logits`, in
unified memory), so the target probabilities were lying there all along; and
the point-mass draft means `p_draft` never needs to be reported at all. The
CPU-side acceptance reads a few rows of 248K floats per round — noise next to
a pass over 15 GB of weights.

How it is held, in three layers:

- **The law, without a model.** `tests/test_sample` runs the accept/resample
  rule 400K times against brute-forced filtered distributions and requires
  total variation under 0.005 — including a draft the filters cut, which must
  be rejected always and still leave the residual equal to the distribution.
- **The engine path.** `tests/test_verify` proves temperature 0 through the
  sampled entry point is token-for-token the greedy verify, and that a
  seeded sampled run reproduces exactly — any state the rewind corrupted
  would surface as the second run diverging.
- **The greedy contract is untouched**: `qwasar_session_verify` still exists,
  still exact, still pinned at every depth.

Two things this deliberately leaves alone. The adaptive depth model is
calibrated on greedy acceptance; sampled acceptance is lower (accepting the
mode with probability `p(mode)` rather than certainty), so the controller
will settle shallower — which is correct behaviour, just tuned by observation
rather than re-derivation, and the acceptance estimator adapts on its own.
And the C-tree CLI and agent stay on the greedy verify, because they decode
greedily to begin with.

Crucible calls the sampled verify unconditionally: at the gates' temperature
0 it *is* the greedy verify, and at the app's defaults speculation now covers
the whole turn — the reasoning block included, which is most of an agent
turn's tokens and is exactly what the phase split (this section's previous
design, now superseded) could not reach.

---
