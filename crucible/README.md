# Crucible

A native macOS coding harness that links the [qwasar](../README.md) runtime
directly — no HTTP, no server, no subprocess — and runs the agent's tools
inside a Virtualization.framework guest where the model can write its own
tooling in Elixir.

Two things make it different, and they come from the same idea:

* **The engine is in-process.** `qwasar_session_eval` is called from the app's
  own address space — no serialisation boundary, and no reason to throw away a
  session's KV cache and recurrent state between turns.
* **The tools live in a VM, and the model grows its own.** The fixed tool
  set is ten calls wide, and one of them is `invoke`: everything else the
  model needs, it writes in Elixir as a **skill** — hot-loaded into a live
  node, owned by the project, replayed into every sibling session. The blast
  radius of that freedom is one virtual machine with no network device.

The design lives in [the spec](spec/README.md), one topic per file; the measurements that shaped it are in the git history.

## The one thing to understand first

> **Nothing the agent does reaches your files.** Your project folder is
> mounted read-only, copied into the guest at boot, and unmounted. The agent
> works against *that copy* — it will tell you, accurately, that it edited a
> file, and your file will be byte-identical.

Work crosses back through **Review Changes…** in the session header. For a
git project it crosses as git: the agent's commits arrive as verified
objects plus one branch — `crucible/<session>` — and nothing else in your
repository is touched; your files change only when *you* `git merge`, with
your own tools and your own conflict resolution. (A dirty tree at session
start gets one question: include your uncommitted changes, or work from
HEAD.) Non-git projects keep the per-file approval sheet: every file shown
with its diff, nothing written until you tick and apply. Either way the
architecture puts a person at this one point.

The guest has **no network device at all** — an absence, not a firewall rule.
It gets a copy-on-write clone of the guest disk per session, the read-only
project copy, and one vsock channel to the host. So a prompt injection in a
file the model reads degrades from *arbitrary code execution on your laptop*
to *a bad edit inside a VM you can throw away*.

**Sandbox settings overlay in three layers** — global, per-project,
per-session; field-wise, the most specific value wins, replace not merge
(spec §8.5). The built-in **Crucible Config** project manages them
conversationally: its sessions run host-side config tools (`config_show`,
`config_set`, `config_clear`) with no sandbox — and no shell, no file
access, no network; the special project's whole reach is the configuration
itself.

**Network is opt-in, per project, and never touches the guest.** Right-click
a project → **Network…** to grant a host allowlist; the model then gets a
`fetch` tool (HTTPS GET only, capped, logged in the transcript) that the
*app* executes under that list — the sandbox stays network-less either way.
Stated plainly, because it is the one real trade in this design: with any
host granted, code sandboxing is unchanged, but *confidentiality* is not —
a prompt injection could encode project contents into request URLs to an
allowed host. The default is off, and for confidential work it should stay
off. The reasoning is spec §8.3.

**Status:** milestones 0–5 built and gated, plus markdown with syntax
highlighting, session parking with verified warm/cold indicators, delegation
(remote sub-agents under a budget), the config project, project-owned
skills, and the git crossing (agent work arrives as a real branch you
merge). Not yet built: `crucible-cli` and the inspector (M6 remainder), and
vision (M7). The full map is
[spec/12-roadmap.md](spec/12-roadmap.md).

## Requirements

| | |
|---|---|
| Machine | Apple silicon. Developed and measured on an M4 with 32 GB |
| macOS | 14.0 minimum; developed on 26 |
| Build | Xcode command line tools, `mise` (erlang/elixir/zig pins), `brew install e2fsprogs` |
| Model | The same `qwasar-model` directory the CLI uses; ~16 GB, not bundled |
| Disk | ~530 MB for the guest image, plus the app |

There is no Docker and no Linux anywhere in the build: the guest image is
assembled natively on macOS. Alpine packages are tarballs, BEAM bytecode is
portable, and `mke2fs -d` builds an ext4 image without root — see
`Guest/mkimage.sh` for the whole story.

## Build and run

```sh
make guest && make run
```

`make guest` assembles the Linux guest image natively — Alpine packages
fetched and untarred, OTP 27 from Alpine's own `erlang27` package, Elixir as
the precompiled `-otp-27` release, the warden compiled on the host's pinned
toolchain, the initramfs and ext4 image built by `Guest/mkrootfs.py`. A
couple of minutes on first run, and downloads are cached after. `make run`
builds the app, signs it, stages the guest image, and opens it; after the
first time, `make run` alone is enough. The result is `build/Crucible.app`,
self-contained apart from the weights.

Then, on first launch:

1. **Choose Model…** in the toolbar — the directory holding `config.json` and
   the safetensors shards. Binding takes ~9 s, after which the toolbar shows
   the context and live-session budget derived from your machine (e.g.
   `90112 ctx · 1 live`; see spec §2.3).
2. **Add Project…** at the foot of the sidebar. That folder is the only thing
   a session can see; a session is created automatically.
3. Type, and **⌘↵**.

The session header should say **`sandboxed · booted in 0.6s`**. If it says
*read-only*, the guest image was not staged — run `make guest`, then `make`.

## What to expect

* **~6 tokens a second.** A dense 27B on ~120 GB/s of memory bandwidth; the
  ceiling is a bandwidth identity, not an efficiency problem.
* **A pause before the first token, once per project.** The system turn is
  ~2200 tokens (the ten tool schemas render into it), so a cold project
  spends roughly a minute reading its own prompt. It is then checkpointed and
  restored by every later session in that project.
* **Formatted replies.** Markdown renders — headings, lists, tables, fenced
  code with syntax highlighting for ~200 languages and a copy button,
  highlighting as each line completes. Reasoning, your turns, and tool results
  stay raw.
* **A lot of reasoning**, collapsed by default; click to expand. A turn where
  the model wrote itself a tool ran 4151 reasoning tokens.
* **Sampled output, speculation included.** Crucible samples with the model's
  own generation_config (temperature 1.0, top-k 20, top-p 0.95) — Qwen's own
  guidance, since thinking-mode models are prone to repetition loops under
  greedy decoding. With a draft head loaded, speculation runs under sampling
  too: the engine verifies drafts by rejection sampling, so the output is
  distributed exactly as serial sampling would produce. The headless gates
  pin temperature 0 so runs stay comparable. Design: spec §7.5.
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
| `make guest` | build the guest image, natively — no Docker |
| `make test` | host suite: goldens, path confinement, UTF-8, persistence, KV prefix |
| `make golden` | regenerate the chat-template goldens — read the diff |
| `make sandbox` | boot a guest and exercise the ten tools end to end |
| `make gate-full` | load the model and run one agent turn, headless |
| `make gate-prefix` | prove the system-prefix KV checkpoint against the engine (~80s) |
| `make clean` | remove build products, keeping the guest image |
| `make clean-guest` | remove the guest image too |

## What is not verified

**The two file-picker flows have never been exercised** — choosing the model
and adding a project go through `NSOpenPanel`, which cannot be driven from a
terminal. What is confirmed is that the app launches, lays out, creates its
store, finds the staged guest image, and logs no faults.

Everything else here was measured rather than assumed; the numbers live with
the designs they justified, in [the spec](spec/README.md) and the git history.

## AI disclosure

Like the engine it sits on, this was written almost entirely by Claude Opus 5,
working from direction and review by the human author. Where a measurement
contradicted an assumption, the code and the notes follow the measurement;
the git history records several such cases, including ones where the first answer was
wrong.
