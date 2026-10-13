# Qwasar.app

One macOS app with two faces: a menu bar item that runs
[`qwasar-server`](../README.md) and shows whether its port is up, and — when
asked — the coding agent's window, which talks to that server over the
[Session API](../API.md) and runs the agent's tools on your Mac. It was
Crucible and the Qwasar Server menu bar app; the two merged when the server
grew an API built for this ([PLAN-qwasar.md](../PLAN-qwasar.md)). A few
names inside still say Crucible — the built-in config project, the state
folder — and mean this app.

## Build and run

From the repository root:

```sh
cd app && make run
```

That builds `qwasar-server` in the parent tree and the app around it, then
opens `build/Qwasar.app`. `make agent` does the same and opens the coding
agent's window straight away.

| | |
|---|---|
| Machine | Apple silicon. The 9B fits a 16 GB Mac; the 27B runs on an M4 with 32 GB; Flash-Next needs ~80 GB resident, so a 128 GB machine |
| macOS | 14.0 minimum; developed on 26 |
| Build | Xcode command line tools |
| Model | Qwen3.5 9B, Qwen3.8 27B or Qwen3.8 Flash-Next, 4-bit MLX — the folders `../download_model.sh` fetches (run it bare for a suggestion by memory); not bundled |
| Disk | the app, plus session checkpoints (a budget you set) |

On first launch:

1. **Choose a model** when asked — the folder holding `config.json` and the
   safetensors shards: the 9B, the 27B or Flash-Next, told apart by the config. The
   server starts on it (the Q in the menu bar pulses amber while the model
   loads, ~10–25 s), and the window's toolbar then shows the model and the
   context and live-session budget the server derived for this machine
   (e.g. `Qwen3.8 Flash-Next · 262144 ctx · 1 live`). Each model folder is
   remembered once granted, so the menu bar's **Model** submenu switches
   between them by restarting the server.
2. **Open Coding Agent** (⌘N), then **Add Project…** at the foot of the
   sidebar. A session is created in it; **New Session** makes more.
3. Type, and **⌘↵**. Return sends; Shift-Return is a new line.

The session header says where its tools run: **`on this Mac · zsh
environment`**.

The menu bar item reads the server's state from its socket, probed once a
second: a **plain Q** is listening, a **pulsing amber dot** is loading the
model, a **red dot** means the server failed or another program holds the
port, a **faded Q** is stopped. Its menu has Open Coding Agent, Copy API
URL, Start/Stop, Port…, Model…, Thinking for API Clients, Start at Login,
Open Server Log, and Quit. **Thinking for API Clients** sets whether the
OpenAI and Anthropic endpoints reason when a request does not say — off
restarts the server with `--no-think`, and a request can still ask either
way; the coding agent's own sessions choose for themselves. The server is a helper
inside the bundle, serving this window, `qwasar-agent` in a terminal, and
anything on the OpenAI or Anthropic endpoints at once; it runs on a pipe the
app holds, so it cannot outlive the app, even after a crash or `kill -9`.

## What the app provides, and why

The app is an opinionated coding agent. It does not try to be a general
chat client or to fit every workflow; it makes two things right that
general-purpose agents leave to chance — **what the model's cache costs
you**, and **what the model's tools can reach** — and accepts the
constraints that come with them.

### 1. The KV cache is transparent and predictable

At a long context, the cache *is* the performance. Flash-Next evaluates a
192K-token conversation in about eleven minutes; resuming the same
conversation from its checkpoint takes about eleven seconds. A client that
throws the cache away by accident — by resending a conversation that
differs by one early token — turns the second number into the first, and
most clients do, silently (see the warning in the
[top-level README](../README.md#run-it-the-server)). So:

* **The server owns the conversation.** A session's system prompt, tools
  and reasoning effort are its prefix, fixed when it opens; every later
  message sends only what is new. There is no request the app could format
  differently and miss on, because it formats nothing but the new message.
* **Every session says where its state is.** The sidebar marks each one
  **live** (in memory; the next message costs only itself), **warm** (a
  checkpoint on disk covers all of it — verified by the server, not
  assumed), or **cold** (it will be re-evaluated), with the server's
  estimate of what resuming costs (`resumes in ~11s`, `rebuilds in ~9m`)
  and the size of its checkpoint.
* **Nothing is silent.** A resume says what it restored and what is left to
  evaluate; prefill shows a progress bar; the context meter shows how full
  the window is.
* **Nothing is evicted behind your back.** **Session Checkpoints…** in the
  app menu lists every checkpoint with its size and when it was last used,
  shows the total against a disk budget you set, and drops one only when
  you click Drop. A dropped session keeps its conversation; it just
  rebuilds when opened. Only the shared system-prefix cache is managed for
  you, in a fixed budget of its own.
* **Changes that would cost a re-prefill are not offered as if they were
  free.** Reasoning effort renders into the prefix, so it is chosen before
  the first message and fixed after it (the menu says so, and sets the
  project's default for the next session). A session only grows — the
  recurrent layers cannot rewind — so there is no editing an earlier turn;
  a different past is a new session.

* **Running notes, and a successor when the window fills.** A session
  cannot be compacted in place — its recurrent layers keep no history to
  cut — so when the window runs out the work continues in a new session.
  To make that cheap and good, the model keeps working notes as it goes:
  after each reply, while you read, the app has it update them in an
  *aside* — a step the server runs and then rolls back, so the notes cost
  the conversation nothing, and a message you send stops it at once. The
  header's **Notes** shows them. At 85% of the window (and when it is full)
  the work goes on in a new session by itself: the notes are brought up to
  date, a successor opens in the same project, with the same placement and
  effort, its first message carrying them, and the old session is unloaded
  and kept, marked as continued. A message sent to a session past the mark
  goes to its successor; a long turn that crosses it at a tool call is
  asked to sum up, and the successor is told to carry on. **Continue in a
  New Session** does the same by hand. `running_notes` in the config
  session turns the asides off; `auto_compact` moves the mark, or turns it
  off and leaves only the offer.

The cost of this is the working set: one session is live at a time on a
32 GB machine and under Flash-Next. Sending a message to another session
parks the live one to disk (a checkpoint write) and resumes the other (a
read) — seconds, shown as they happen, rather than a re-prefill.

### 2. The tools run on your Mac

A session runs its tools directly on your Mac: `Read`, `Write`,
`Edit`, `Glob`, `Grep`, `Bash` and `TodoWrite`, as you, in the project folder, with your
login shell's environment — the app starts your shell as a login shell once
per project, the way Terminal would, and takes its PATH and variables, so
Homebrew, mise, cargo and the project's own pinned toolchain are all there.
Relative paths are the project's; absolute paths work too. Nothing is
confined. The session is told so, and told not to do anything destructive or
irreversible you did not ask for — but that is an instruction, not a wall:
a coding agent reads files it did not write, and any of them can carry
instructions aimed at the model. Review its changes with your own
`git diff`. **⌘.** stops a turn, and kills a command that is running.

The app is not App Sandboxed: everything a sandboxed app starts inherits
its sandbox, which would leave a session unable to reach your files or your
toolchains.

(Sessions could once run their tools in a Linux VM instead. That is gone;
a session created that way is marked with a box in the sidebar, can be
read, and continues only through **Continue in a New Session**, on this
Mac.)

### 3. Aligned with how the model was trained### 4. Aligned with how the model was trained, and turns that end cleanly

Qwen3.8 Flash-Next's agentic results are reported in the Claude Code harness
(its model card), and its chat template fixes the format of tool calls and
results. The app follows both, rather than a format of its own:

* **The tools are named and shaped as in that harness**: `Read` with
  numbered lines and `offset`/`limit`; `Edit` with `old_string`,
  `new_string` and `replace_all`, matching an exact substring once; `Grep`
  with ripgrep syntax and an `output_mode`; `Glob`; `Bash` with a
  `timeout`; `TodoWrite` for a task list. Results may be 64 KB on
  Flash-Next (8 KB on the 27B, whose prefill is ten times slower, and on
  the 9B, whose window on a 16 GB Mac is small).
* **The system prompt has the harness's shape**: an environment block
  (working directory, git branch, platform, date), how to work (read before
  changing, match the project, verify, summarize), then the project's own
  `AGENTS.md`, `CLAUDE.md` or `QWEN.md` if it keeps one.
* **Tool results are rendered as the template renders them** — each in its
  own `<tool_response>`, several to a turn — and the app's own interjections
  are marked as `<system-reminder>`s.
* **Effort defaults to xhigh on Flash-Next**, with 64K tokens a step: its
  card says lower effort in agent work costs more in retries than it saves.
  The 27B stays at medium, with 4K tokens a step; the 9B at medium too (its
  own template's default) with 16K, since its reasoning alone often passes
  4K and it runs ~15x faster.


* Sampling is the model's own generation config (temperature 1.0, top-k 20,
  top-p 0.95) — Qwen's guidance for thinking models, which loop under
  greedy decoding — and on the 9B a presence penalty of 1.5, which Qwen
  recommends for its smaller models. The headless gates pin temperature 0 so runs compare.
* A turn runs as many rounds of tool calls as the task takes, up to 200. At
  that cap the last results are still delivered, with a request to stop and
  summarize what was learned, what changed and what remains — so a long
  task ends in a report you can act on, not mid-thought. **⌘.** stops a turn
  at the next token.

### Also in the box

* **Configuration by conversation.** The built-in **Qwasar Config** project
  manages everything the menus and sheets do, by asking: the server's port,
  its model, its context size and live sessions, starting and stopping it,
  start at login, the checkpoint disk budget; each project's default effort
  and guidance prompt; and the session settings (Bash's timeout,
  delegation), which layer global →
  project → session, the most specific value winning field by field. Its
  sessions run host-side config tools — no shell, no file access, no
  network; their whole reach is the configuration. A change that restarts
  the server waits until the reply that made it is finished. The delegation
  API key is the one setting it cannot touch: you enter that yourself.
* **Delegation.** A session can hand a sub-task to a remote model, under a
  dollar budget, if you have set an API key (**Set Delegation API Key…**,
  stored in the Keychain) and granted models in the config. Off otherwise.
* **Formatted replies.** Markdown renders — headings, lists, tables, fenced
  code with syntax highlighting for ~200 languages and a copy button.
  Reasoning is collapsed by default; your turns and tool results stay raw.

## What to expect

* **~6 tokens a second on the 27B**; **~68 on Flash-Next** and **~90 on the
  9B** on an M5 Max (~465 and 200–400 tok/s prefill on short prompts; the
  9B has not been measured on a 16 GB machine, whose memory bandwidth is
  several times lower). The 27B's ceiling is a memory
  bandwidth identity, not an efficiency problem. At 32 KB of cache per
  token Flash-Next gets the full 262K window, and each step may generate up
  to 64K tokens at xhigh (32K otherwise) against the 27B's 4K.
* **A pause before the first token, once per project and day.** The system
  turn is a few thousand tokens (the tool schemas render into it), so a project's first
  session spends a while reading its own prompt. The server then keeps it,
  and every later session in that project starts from it.
* **A lot of reasoning.** One measured turn of the 27B ran 4,151 reasoning
  tokens.
* **Sessions survive restarts.** Quitting checkpoints the live session, and
  every session's conversation is kept on the server, so nothing is lost to
  a restart and a checkpointed session resumes warm. The first resume in a
  freshly started server is slower than later ones.

## Headless and make targets

Every gate runs from a terminal, which is also the fastest way to try the
system without the GUI:

```sh
make gate-full ROOT_DIR=/path/to/project PROMPT="what does qw_edit_apply do?"
```

| target | what it does |
|---|---|
| `make` | build and sign `build/Qwasar.app` |
| `make install` | ... and copy it to `/Applications` (`INSTALL_DIR=` elsewhere); it still uses this checkout's models |
| `make uninstall` | remove it from `/Applications`; sessions and settings are kept |
| `make run` | build, then launch it |
| `make agent` | the same, with the coding agent's window open |
| `make test` | the host suite: the tools, settings, delegation, persistence |
| `make gate` | start the server on the model, headless |
| `make gate-full` | ... and run one agent turn through it |
| `make clean` | remove build products |

The design lives in [the spec](spec/README.md), one topic per file; the
measurements that shaped it are in the git history. Sections of the spec
that describe an in-process engine describe what the server now does, in
the same terms; the API is the contract between them.

## What is not verified

**The two file-picker flows are exercised only by hand** — choosing the
model and adding a project go through `NSOpenPanel`, which cannot be driven
from a terminal.

## AI disclosure

Like the engine it sits on, this was written almost entirely by Claude Opus 5,
working from direction and review by the human author. Where a measurement
contradicted an assumption, the code and the notes follow the measurement;
the git history records several such cases, including ones where the first
answer was wrong.
