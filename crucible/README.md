# Crucible

A native macOS coding harness that links the [qwasar](../README.md) runtime
directly — no HTTP, no server, no subprocess — and runs the agent's tools
inside a Virtualization.framework guest where the model can write its own
tooling in Elixir.

Two things make it different, and they come from the same idea:

* **The engine is in-process.** `qwasar_session_eval` is called from the app's
  own address space — no serialisation boundary, and no reason to throw away a
  session's KV cache and recurrent state between turns.
* **The tools live in a VM, and the model owns them.** The fixed tool set is
  twelve calls wide and one of them is `invoke`; everything else the model
  needs, it writes in Elixir, hot-loads into a live node, and calls. The blast
  radius of that freedom is one virtual machine with no network device.

The design, measurements, and decisions are in [PLAN.md](PLAN.md).

## The one thing to understand first

> **Nothing the agent does reaches your files.** Your project folder is
> mounted read-only, copied into the guest at boot, and unmounted. The agent
> works against *that copy* — it will tell you, accurately, that it edited a
> file, and your file will be byte-identical.

Work crosses back through **Review Changes…** in the session header: the
sandbox is diffed against its boot baseline, every file is shown with its
diff, and nothing is written until you tick it and apply (overwritten files
are copied aside first). There is deliberately no "apply all and don't ask
again" — the whole architecture exists to put a person at this one point.

The guest has **no network device at all** — an absence, not a firewall rule.
It gets a copy-on-write clone of the guest disk per session, the read-only
project copy, and one vsock channel to the host. So a prompt injection in a
file the model reads degrades from *arbitrary code execution on your laptop*
to *a bad edit inside a VM you can throw away*.

**Status:** milestones 0–5 built and gated, plus markdown rendering with
syntax highlighting. Not yet built: session parking to explicit checkpoints
(M6), vision and speculative decoding (M7), and replacing the file-copy
patch-back with a real git branch you can merge (PLAN.md 7.4a).

## Requirements

| | |
|---|---|
| Machine | Apple silicon. Developed and measured on an M4 with 32 GB |
| macOS | 14.0 minimum; developed on 26 |
| Build | Xcode command line tools, and Docker **only** to build the guest image |
| Model | The same `qwasar-model` directory the CLI uses; ~16 GB, not bundled |
| Disk | ~530 MB for the guest image, plus the app |

Docker is build-time only; the shipped app never needs it.

## Build and run

```sh
make guest && make run
```

`make guest` builds the Linux guest image (Alpine, OTP 27 and Elixir, the
warden, the vsock bridge) — about ten minutes the first time, cached after,
Docker Desktop running. `make run` builds the app, signs it, stages the guest
image, and opens it; after the first time, `make run` alone is enough. The
result is `build/Crucible.app`, self-contained apart from the weights.

Then, on first launch:

1. **Choose Model…** in the toolbar — the directory holding `config.json` and
   the safetensors shards. Binding takes ~9 s, after which the toolbar shows
   the context and live-session budget derived from your machine (e.g.
   `90112 ctx · 1 live`; see PLAN.md §2.3).
2. **Add Project…** at the foot of the sidebar. That folder is the only thing
   a session can see; a session is created automatically.
3. Type, and **⌘↵**.

The session header should say **`sandboxed · booted in 0.6s`**. If it says
*read-only*, the guest image was not staged — run `make guest`, then `make`.

## What to expect

* **~6 tokens a second.** A dense 27B on ~120 GB/s of memory bandwidth; the
  ceiling is a bandwidth identity, not an efficiency problem.
* **A pause before the first token, once per project.** The system turn is
  ~2500 tokens (the twelve tool schemas render into it), so a cold project
  spends roughly a minute reading its own prompt. It is then checkpointed and
  restored by every later session in that project.
* **Formatted replies.** Markdown renders — headings, lists, tables, fenced
  code with syntax highlighting for ~200 languages and a copy button,
  highlighting as each line completes. Reasoning, your turns, and tool results
  stay raw.
* **A lot of reasoning**, collapsed by default; click to expand. A turn where
  the model wrote itself a tool ran 4151 reasoning tokens.
* **Sampled reasoning, speculative answers.** Crucible samples with the
  model's own generation_config (temperature 1.0, top-k 20, top-p 0.95) —
  Qwen's own guidance, since thinking-mode models are prone to repetition
  loops under greedy decoding — and, when a draft head is loaded, switches to
  greedy decoding with speculation at the `</think>` boundary, where the text
  is structured enough to draft well. The headless gates pin temperature 0 so
  runs stay comparable. Design and trade-offs: PLAN.md 7.5.
* **⌘.** stops a turn at the next token.
* **Switching sessions is not free.** Only one session is live at a time on a
  32 GB machine; switching re-prefills (helped by the engine's LRU cache).
  Reading a parked transcript costs nothing; sending it a message pays.

## Headless and make targets

Every gate runs from a terminal, which is also the fastest way to try the
system without the GUI:

```sh
make gate-full ROOT_DIR=/path/to/project PROMPT="what does qw_edit_apply do?"
```

| target | what it does |
|---|---|
| `make` | build and sign `build/Crucible.app` |
| `make run` | build, then launch it |
| `make guest` | build the guest image (needs Docker) |
| `make test` | host suite: goldens, path confinement, UTF-8, persistence, KV prefix |
| `make golden` | regenerate the chat-template goldens — read the diff |
| `make sandbox` | boot a guest and exercise the twelve tools end to end |
| `make gate-full` | load the model and run one agent turn, headless |
| `make gate-prefix` | prove the system-prefix KV checkpoint against the engine (~80s) |
| `make clean` | remove build products, keeping the guest image |
| `make clean-guest` | remove the guest image too |

`make guest` grows Docker's build cache by several GB per run; if a rebuild
hits *no space left on device*, run `docker builder prune -af`.

## What is not verified

**The two file-picker flows have never been exercised** — choosing the model
and adding a project go through `NSOpenPanel`, which cannot be driven from a
terminal. What is confirmed is that the app launches, lays out, creates its
store, finds the staged guest image, and logs no faults.

Everything else here was measured rather than assumed; the numbers are in
[PLAN.md](PLAN.md).

## AI disclosure

Like the engine it sits on, this was written almost entirely by Claude Opus 5,
working from direction and review by the human author. Where a measurement
contradicted an assumption, the code and the notes follow the measurement;
PLAN.md records several such cases, including ones where the first answer was
wrong.
