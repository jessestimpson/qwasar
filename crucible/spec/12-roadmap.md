# 12. Roadmap

The milestone *histories* — what each one built, the bugs it hit, the numbers
it measured — live in this repository's git log (they were this file, once).
What a spec needs is the map: what exists, and what is next.

## Built and gated

| milestone | what it proved |
|---|---|
| M0 | entitlements, virtualization probe, the 16 GB mmap inside App Sandbox |
| M1 | the window: sidebar, transcript, composer; read-only host tools |
| M2 | the guest: boot under 2 s, vsock bridge, warden, path confinement |
| M3 | the tools in the guest: `/work` copy, seed-work contract, six file tools |
| M4 | self-modification: `elixir`/`define`/`tools`/`invoke`, workspace crash + replay; tool surface frozen at ten |
| M5 | materialisation: baseline, propose, the approval sheet, undo |
| — | markdown transcript with syntax highlighting |
| — | sampling with the rejection-sampling verify (speculation under sampling) |
| — | the native macOS image build — no Docker, no Linux (§6.2) |
| — | host-mediated network: `fetch` under a per-project allowlist (§8.3) |
| — | sandbox configuration in three layers + the config project (§8.5) |
| — | delegation E1+E2: budgeted, embedded, steerable; the remote agent drives the sandbox by proxy; one feature under one name, user- or model-initiated (§15) |

## Next

- **M6 — session parking.** Checkpoints so switching sessions restores
  instead of re-prefilling — the UX is settled ahead of the build (§4.4: no
  save verb, boundary-moment autosave, a Park action, warm/cold indicators
  with time estimates); `crucible-cli` for headless end-to-end runs; the
  inspector.
- **M7 — vision, and speculation in the app's gates.** Images into the
  session (§2's mrope machinery is already in the engine); `gate-mtp`
  promoted into CI.
- **§7.4a — the git crossing.** Replace the file-copy patch-back with a real
  branch the user merges; §7.4 demoted to the fallback for non-git projects.
- **§15 — delegation.** E1 and E2 built; next E3 (judgment: prompt-fragment
  tuning and a measured week of use).

## Retired

Deterministic simulation testing with `eta` — built through M4, reviewed
against its own exit criteria, removed 2026-08-23 with its two tools and its
sim node. The control-plane invariants it checked outlived it (§10); the full
record is in the git history of `PLAN.md`.
