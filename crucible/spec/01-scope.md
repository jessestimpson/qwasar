## 1. Scope

### 1.1 What we are building

- A standard, double-clickable, notarisable `.app`. `Crucible.app` in
  `/Applications`, launched from the Dock, quit with ⌘Q. Not a CLI with a
  window bolted on.
- A chat harness with **Projects** (a name, a root directory, settings) and
  **Sessions** (a conversation, an initial working directory inside a project,
  a sandbox, a transcript that survives relaunch).
- An agent loop written in Swift on top of `qwasar.h` and `qwasar_toolcall.h`.
- A per-session Linux guest running two BEAM nodes, reachable over vsock.
- A self-modification tool surface: define an Elixir module, load it into the
  running node, register it as a tool, call it on the next step.
- Direct collaboration on the user's own tree: the agent edits the working
  copy live, the user reviews and commits with their own git, and the real
  `.git` stays behind a guest-side shadow it can never write (§7.4).

### 1.2 What we are not building

- **Not a server.** `qwasar-server` already exists and stays. Crucible does not
  speak HTTP, does not expose a port, and does not need one.
- **Not multi-model.** Same position as the engine: one model, done properly.
- **Not iOS/Catalyst.** Virtualization.framework is macOS-only on Apple silicon,
  and so is the whole premise.
- **Not a replacement for `qwasar-agent`.** The TUI agent stays as the small,
  readable reference loop. Crucible is the one with a sandbox and a scheduler.
- **No Python, no C++.** Inherited from the parent tree (§8) and unchanged.

### 1.3 Requirements

| | |
|---|---|
| Host | Apple silicon Mac, 32 GB recommended (see §2.3) |
| macOS | 14.0 minimum; developed on 26.x |
| Build | Xcode 16+, Swift 6 language mode, `cc` for the C tree |
| Model | The same `qwasar-model` directory the CLI uses; ~16 GB, not bundled |
| Guest | Alpine arm64 + OTP 27 + Elixir 1.18, ~320 MB image (see §6.2) |

---
