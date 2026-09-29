# Crucible — a native macOS coding harness on the qwasar runtime

`Crucible` is a SwiftUI macOS application that links `libqwasar` **directly** —
no HTTP, no `qwasar-server`, no subprocess — and gives the model an agent loop
whose tools do not run on your machine. They run on a **BEAM (Erlang/Elixir)
node inside a macOS Virtualization.framework guest**, where the model is allowed
to hot-load Elixir and Erlang modules and thereby rewrite its own tool surface
mid-task. The session works directly in your own working tree — edits land
live, as ordinary uncommitted changes your git shows you — while your real
`.git` (hooks, config, history) sits behind a bind-mounted shadow the guest
can never write. You review with `git diff` and commit when you say.

> **Codename.** `Crucible` is provisional — a vessel you melt things in, sealed
> from the room. Rename before the first commit that ships a bundle identifier.

Two things make this different from a normal harness, and both come from the
same idea:

1. **The engine is in-process.** `qwasar_session_eval` is called from the app's
   own address space. There is no serialisation boundary, no token re-encoding,
   no localhost socket, and — the part that actually matters — no reason to
   throw away a session's KV cache and recurrent state between turns.
2. **The tools are in a VM, and the model owns them.** A conventional harness
   ships a fixed tool set written by the harness author. Here the fixed set is
   ten calls wide and one of them is `invoke`; everything else the model needs,
   it writes in Elixir, hot-loads into a live node, and calls. The blast radius
   of that freedom is exactly one virtual machine with no network device.

---
