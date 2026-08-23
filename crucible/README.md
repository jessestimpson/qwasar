# Crucible

A native macOS coding harness that links the [qwasar](../README.md) runtime
directly — no HTTP, no server, no subprocess — and runs the agent's tools inside
a Virtualization.framework guest where the model can rewrite its own tooling in
Elixir.

Two things make it different, and they come from the same idea:

* **The engine is in-process.** `qwasar_session_eval` is called from the app's
  own address space. No serialisation boundary, no localhost socket, and no
  reason to throw away a session's KV cache and recurrent state between turns.
* **The tools are in a VM, and the model owns them.** A conventional harness
  ships a fixed tool set written by its author. Here the fixed set is twelve
  calls wide and one of them is `invoke`; everything else the model needs, it
  writes in Elixir, hot-loads into a live node, and calls. The blast radius of
  that freedom is one virtual machine with no network device.

The design, the measurements, and the decisions behind both are in
[PLAN.md](PLAN.md).

## Status

Milestones 0 through 5 are built and gated, plus markdown rendering with syntax
highlighting. The thing to understand before using it:

> **Nothing the agent does reaches your files.** Your project folder is mounted
> read-only, copied into the guest at boot, and unmounted. The agent reads,
> writes, edits and runs commands against *that copy*. It will tell you —
> accurately — that it edited a file, and your file will be byte-identical.

**Review Changes…** in the session header is how work crosses back: the sandbox
is diffed against the baseline it booted with, every file is shown with its diff,
and nothing is written until you tick it and apply. Whatever gets overwritten is
copied aside first. There is no "apply all and don't ask again" — the whole
architecture exists to put a person at this one point.

Also not built: session parking to explicit checkpoints (M6 — switching sessions
currently re-prefills, helped by the engine's own LRU cache), vision, and
speculative decoding (M7). Planned but not started: replacing the file-copy
patch-back with a real git branch you can merge (PLAN.md 7.4a).

## Requirements

| | |
|---|---|
| Machine | Apple silicon. Developed and measured on an M4 with 32 GB |
| macOS | 14.0 minimum; developed on 26 |
| Build | Xcode command line tools, and Docker **only** to build the guest image |
| Model | The same `qwasar-model` directory the CLI uses; ~16 GB, not bundled |
| Disk | ~530 MB for the guest image, plus the app |

Docker is a build-time dependency and nothing else. The shipped app never needs
it, and neither does anything at runtime.

## Build and run

```sh
make guest && make run
```

`make guest` builds the Linux guest image — Alpine, OTP 27 and Elixir via mise,
the warden, and the vsock bridge. It takes about ten minutes the first time and
is cached afterwards. Docker Desktop must be running for that step.

`make run` builds the app, signs it, stages the guest image into the bundle, and
opens it. After the first time, `make run` on its own is enough.

The result is `build/Crucible.app`, which is self-contained apart from the model
weights. Drag it to `/Applications` if you like.

## First launch

1. **Choose Model…** in the toolbar. Pick the directory holding `config.json`
   and the `*.safetensors` shards. Binding the weights takes about nine seconds,
   after which the toolbar reads `90112 ctx · 1 live` — the context and
   live-session budget derived from your machine (see PLAN.md §2.3).
2. **Add Project…** at the foot of the sidebar. The folder you pick is the only
   thing a session can see. A session is created for it automatically.
3. Type, and **⌘↵**.

The session header should say **`sandboxed · booted in 0.6s`**. If it says
*read-only*, the guest image was not staged — run `make guest`, then `make`.

## What to expect

* **~6 tokens a second.** This is a dense 27B model on ~120 GB/s of memory
  bandwidth; the ceiling is a bandwidth identity, not an efficiency problem.
* **A pause before the first token, once.** The system turn is about 2500 tokens
  — 99.6% of a first turn, because the twelve tool schemas render into it — so a
  cold project spends roughly a minute reading its own prompt. It is then
  checkpointed, and every later session in that project restores it instead of
  re-reading it. The footer shows a real count and an estimate.
* **Formatted replies.** Markdown renders: headings, lists, tables, quotes, and
  fenced code blocks with syntax highlighting for ~200 languages, a language
  label and a copy button. A block highlights as each line completes rather than
  all at once at the end. Reasoning, your own turns and tool results stay raw —
  formatting them would claim an intent that is not in the source.
* **A lot of reasoning.** The block is collapsed by default; click to expand. A
  turn where the model writes itself a tool ran 4151 reasoning tokens.
* **⌘.** stops a turn at the next token.
* **Switching sessions is not free.** Only one session is live at a time on a
  32 GB machine, so switching frees one and rebuilds the other. Reading a
  parked session's transcript costs nothing; sending it a message pays.

## The sandbox

The guest is configured with **no network device at all** — not a firewall rule
the guest might talk its way around, but an absence. It has:

* a copy-on-write clone of the guest disk, per session;
* your project folder, mounted read-only, copied to `/work` at boot and then
  unmounted;
* one vsock channel to the host, and nothing else.

So a prompt injection in a file the model reads degrades from *arbitrary code
execution on your laptop* to *a bad edit inside a VM you can throw away*. That
is the argument for the whole design.

## Headless

Every gate the project uses is runnable from a terminal, which is also the
fastest way to try it without the GUI:

```sh
make gate-full ROOT_DIR=/path/to/project PROMPT="what does qw_edit_apply do?"
```

```sh
make sandbox        # boot a guest and exercise the twelve tools end to end
make test           # host suite: goldens, path confinement, UTF-8, persistence, KV prefix
```

## Make targets

| target | what it does |
|---|---|
| `make` | build and sign `build/Crucible.app` |
| `make run` | build, then launch it |
| `make guest` | build the guest image (needs Docker) |
| `make test` | the host test suite |
| `make golden` | regenerate the chat-template goldens — read the diff |
| `make sandbox` | boot a guest and run the sandbox gate |
| `make gate-full` | load the model and run one agent turn, headless |
| `make gate-prefix` | prove the system-prefix KV checkpoint against the engine (~80s) |
| `make clean` | remove build products, keeping the guest image |
| `make clean-guest` | remove the guest image too |

`make guest` grows Docker's build cache by several GB per run. If a rebuild
fails with *no space left on device*, run `docker builder prune -af`; mkimage
warns you once the cache passes 40 GB.

## What is not verified

**The two file-picker flows have never been exercised.** Choosing the model and
adding a project both go through `NSOpenPanel`, which cannot be driven from a
terminal, and everything downstream of them in the UI is untested with it. What
is confirmed is that the app launches, lays out, creates its store in the
sandbox container, finds the staged guest image, and logs no faults.

Everything else in this README was measured rather than assumed, and the numbers
that back it are in [PLAN.md](PLAN.md).

## AI disclosure

Like the engine it sits on, this was written almost entirely by Claude Opus 5,
working from direction and review by the human author. Where a measurement
contradicted an assumption, the code and the notes follow the measurement —
PLAN.md records several such cases, including the ones where the first answer
was wrong.
