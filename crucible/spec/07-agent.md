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
  Warden.Diff      git plumbing
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

### 7.4a Leaning into git — the crossing as an object exchange

§7.4 below describes what M5 built: a baseline manifest, per-file SHAs, drift
detection, size caps, and a host that writes bytes under approval. It works, and
it is on a path that ends badly.

Look at what it has already grown: baseline hashes per file, a *conflict* state
distinct from a *rejection*, an added-file-now-exists case, a
modification-to-a-deleted-file case, a backup directory, a re-baseline step. Each
one is a thing git does properly. Keep going and the next asks are conflict
resolution, then "take only these two files", then "the model made three logical
changes, split them" — **merge, cherry-pick and rebase, arriving one bad
reimplementation at a time**.

The 2 MB/6 MB caps are the tell. There is no principled reason for them except
that whole file bodies are being shipped; git would send a few KB of compressed
delta for the same change.

#### The reframe

**The VM boundary is a network boundary, and git is already a distributed VCS.**
Two repositories that cannot share a filesystem, exchanging work, is the problem
git was built to solve. The guest already holds a full clone — `.git` is copied
in with everything else — it simply is not being treated as one.

So the crossing stops being a file copy:

    guest commits  ->  sends (sha, type, content) for BASE..TIP
                   ->  host writes loose objects + one ref

and the user gets a **real branch in their real repo**. Merge, rebase,
cherry-pick, stash, `git log -p`, blame, their own mergetool and their own
config — none of it written here.

A `git worktree` would be the obvious thing to reach for and does not work: `git
worktree add` writes a `.git` *file* pointing at an absolute path inside the main
repository's `.git/worktrees/`, so objects, refs and index all stay in a repo the
worktree must be able to reach and write. The guest has no path to the host's
`.git` — the share is unmounted before the agent runs, which is the property the
entire sandbox rests on (§8.2). Restoring a writable mount to enable worktrees
would trade away exactly what the sandbox is for. A *clone* exchanging *objects*
is the right primitive for this topology, and it is the one git was designed
around.

#### No libgit2, and no child process

The host cannot spawn `git`: security-scoped access does not survive into a child
process, which is why `git apply` was rejected for §7.4 in the first place.
libgit2 was the obvious answer and is **not needed**.

A loose object is `<type> <len>\0<content>`, zlib-deflated, stored at
`.git/objects/<sha[0:2]>/<sha[2:]>` where the sha names the *uncompressed* form.
That is zlib, SHA-1, and a file write — `Insecure.SHA1` from CryptoKit and `-lz`,
a system library. Nothing to vendor, nothing for a user to install, which is the
same bar the rest of this project holds (`make`, and nothing else).

Verified before writing this section, against a real repository: objects written
by hand, `git fsck --strict` clean, `git log`/`git diff` correct across a
modification, an addition, a deletion and a nested path, `git merge` clean. And
with a concurrent edit to the same line on the host, `git merge` produced a
proper **line-level** conflict with markers — where §7.4 can only detect drift at
*file* granularity and offer a checkbox.

**The hash is the verification, and it is free.** A wrong type, a wrong length or
a truncated transfer all land as a refusal rather than as a corrupt repository.
That is a stronger guarantee than the byte path has, not a weaker one.

**The import is additive.** Objects, plus one ref. Never HEAD, never the index,
never a branch the user owns, never a working file. The worst outcome of a bug is
unreferenced objects that `git gc` collects. The destructive step — changing
files — moves to the user's own `git merge`, which they already trust and which
already handles conflicts.

#### Decisions taken

**Refs land at `refs/heads/crucible/<slug>`.** A real branch: it shows in `git
branch`, in every GUI, and merges by short name. The hidden `refs/crucible/*`
namespace avoids clutter at the cost of being forgettable and unmergeable without
typing a full ref, which is the wrong trade for something the user is meant to
act on. Stale ones are deleted like any other branch.

**The agent makes its own commits.** It already has git through `bash`; the
system prompt will say so. A reviewable series beats one flat blob, and it is
what makes cherry-picking a *part* of the work possible at all. The harness
commits any uncommitted remainder at proposal time so nothing is lost.

**A dirty worktree is the user's choice, at session start.** If the project has
uncommitted changes when a session begins, they are copied into `/work` like
everything else — and would end up inside the agent's commit, so merging would
commit the user's own work under the agent's name. Rather than pick a policy, the
session-creation flow asks, because the real question underneath is *should the
agent see my uncommitted work at all* and both answers are right sometimes:

  - **Include** — `/work` is the working tree as it stands, and the guest commits
    that state first, on its own, as "local changes before this session". The
    user's work stays visibly theirs in the log. Right when they are mid-change
    and want help with it.
  - **Exclude** — `/work` is reset to HEAD after the copy, so the uncommitted
    changes never enter the sandbox at all. Right when they want the agent
    working from a known, committed base.

The prompt appears only when the tree is actually dirty, and only for git
projects.

#### What this deletes

The honest test of the idea. Of the ~936 lines across `Materialise.swift`,
`materialise.ex`, `ApprovalSheet.swift` and their suites, roughly two thirds go:
the per-file SHA baselining, drift detection, conflict and rejection states, size
caps, the backup directory, the re-baseline step, and most of the approval
sheet's per-file machinery. A large net removal is the signal that this is the
right shape.

What remains is §7.4 itself, **demoted to the fallback** for projects that are not
git repositories. That case is genuinely simple — no history to reconcile — and
is where a byte copy under approval is the proportionate answer.

#### Risks worth naming

  - **It writes into the user's `.git`.** Confined to new objects and
    `refs/heads/crucible/*`; `git fsck` is the check, and it runs in the tests.
  - **`.git` is copied into the guest**, so its size is a boot cost. 6.2 MB for
    this repository, which is nothing; a repository with large history is a case
    to measure rather than assume.
  - **The agent can rewrite history inside the guest.** Harmless: only objects
    reachable from the proposed tip are exported, and import cannot move a ref
    the user owns.
  - **Submodules and LFS are untested.** Gitlinks and LFS pointers should
    transfer as the SHAs and pointer files they are, which is correct, but that
    is reasoning rather than evidence.

### 7.4 Materialisation — the only thing that crosses back

**At sandbox creation:**

1. Host walks the session's working directory, honouring `.gitignore` plus
   `Project.excludeGlobs`, capped at a configurable size (default 512 MB) and
   file count. What was excluded is *stated*, in a transcript item — an agent
   that cannot see `node_modules` should know it cannot see `node_modules`.
2. Host records `baseline.json`: every included path → SHA-256 + mode.
3. `/base` is mounted read-only over virtiofs. The guest's init does
   `cp -a /base/. /work/` — a local copy inside the guest, fast, and after it the
   host share is not touched again.
4. Guest runs, with a **separate git dir so a real repo is never disturbed**:
   ```
   git --git-dir=/work/.crucible-git --work-tree=/work init
   git --git-dir=/work/.crucible-git --work-tree=/work add -A
   git --git-dir=/work/.crucible-git --work-tree=/work commit -m baseline
   ```
   Git is the diff engine because it already solves renames, modes, binary
   files, and patch application, and because the guest image is carrying it
   anyway.

**When the model calls `propose` (or the user asks for changes):**

5. Guest produces `git diff --binary baseline` plus a structured summary: files
   added / modified / deleted / renamed, insertions, deletions.
6. Patch and summary come over vsock.
7. **Host verifies before showing anything.** For every path the patch touches,
   re-hash the current file on disk and compare to `baseline.json`. Any mismatch
   means the real tree changed while the agent worked; the file is flagged
   *conflicted* in the sheet and defaults to unchecked.
8. **Every path is validated** — this is the security boundary, so it is
   explicit and it is tested adversarially (§10):
   - reject absolute paths;
   - reject any path whose normalised form escapes the session root;
   - reject paths traversing a symlink that leaves the root;
   - reject `.git/` unless the user opted in;
   - reject the session root itself being a symlink resolved differently than at
     baseline.
9. **The approval sheet.** Unified diff per file, syntax-highlighted,
   per-file checkboxes, Apply Selected / Reject / Open in External Diff. The
   patch is *not* applied by anything the model can call. There is no `--yes`
   flag for this, at any level, ever. The C agent has `--yes` for host writes;
   Crucible's equivalent is that writes inside the sandbox need no confirmation
   at all — the confirmation moved to the boundary where it means something.
10. **Apply.** Snapshot the touched files into `undo/<n>/` first. Then
    `git apply --3way` if the project is a git repo, else write files with an
    atomic rename per file. Report per-file success; a partial application is
    reported as a partial application, not as success.
11. **Undo** restores from the snapshot. It is a menu item and a transcript
    affordance, not a hidden folder.
12. The guest re-baselines: the applied state becomes the new baseline commit,
    so the next proposal is a diff against what the user actually accepted.

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
