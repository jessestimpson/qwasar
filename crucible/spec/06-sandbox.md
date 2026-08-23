## 6. The sandbox

### 6.1 Framework choices, stated plainly

- **`VZVirtualMachine`** with `VZEFIBootLoader` and a raw disk image. EFI boot
  rather than `VZLinuxBootLoader` + kernel/initrd, because it lets the guest own
  its own kernel updates and lets `mkimage.sh` produce a single artefact.
- **`VZVirtioBlockDeviceConfiguration`** — the per-session disk (§6.3).
- **`VZVirtioFileSystemDeviceConfiguration`** with a `VZSharedDirectory` marked
  **read-only**, tag `base`. This is the project tree going in. Read-only is
  enforced by the framework, not by our code, and that is why we use it rather
  than streaming a tarball.
- **`VZVirtioSocketDeviceConfiguration`** — vsock. The control channel.
- **`VZVirtioConsoleDeviceSerialPortConfiguration`** — boot log to a file, for
  the inspector and for support.
- **No `VZNetworkDeviceConfiguration` at all**, by default. Not a firewall rule,
  not a policy toggle the guest could talk its way around: the VM is configured
  without a network device, so there is nothing to configure. This is the single
  strongest security property in the design and it costs one omitted line.
- **`VZVirtualMachineConfiguration.validateWithError`** before every boot. It
  catches misconfiguration with a real message instead of a runtime trap.

Memory: 2 GB per VM by default, 1 CPU less than `VZVirtualMachineConfiguration
.maximumAllowedCPUCount` capped at 4 — the engine needs the cores more than the
guest does, and the guest is mostly waiting on the model.

### 6.2 The golden image

`Guest/mkimage.sh` builds it **natively on macOS** — no Docker, no Linux
anywhere in the build. What made that possible is that every reason the old
containerised build needed a Linux userland turned out to dissolve:

- **Alpine packages are tarballs.** `mkrootfs.py` reads APKINDEX, resolves the
  dependency closure itself, verifies and untars the packages into a plain
  directory tree. The one apk trigger that matters — busybox's applet symlink
  farm — is one idempotent line in `crucible-init` at boot instead.
- **OTP 27 is Alpine's own `erlang27` package** (3.23 finally carries it), so
  the 1m39s source build is simply gone. Elixir is the precompiled `-otp-27`
  release: pure BEAM, portable, pinned by sha256.
- **The warden is compiled on the host** with mise's pinned
  `erlang@27.3.4` + `elixir@1.18.4-otp-27` — BEAM bytecode does not care which
  OS compiled it, only which OTP. The `:work_fs` unit tests are excluded on
  the host (they write under the real `/work`); `make sandbox` exercises the
  real tools in the real guest, which is the stronger claim anyway.
- **`vsock_port` is cross-compiled with zig** (`-target aarch64-linux-musl
  -static`), a self-contained toolchain already on the machine.
- **The initramfs is ours**: a generated ~15-line `/init`, `busybox.static`,
  and the module closure for virtio/ext4 computed from `modules.dep` (with
  `modules.builtin` consulted, because linux-virt moves drivers across that
  line between releases). 1.1 MB, against mkinitfs's 6.4.
- **The ext4 image comes from `mke2fs -d`** (brew e2fsprogs): populated from
  the directory tree with no root, no loop devices, no Linux.

```
alpine base + busybox                 ~  10 MB
linux-virt kernel + modules           ~  70 MB
erlang 27 (Alpine apk)                ~ 100 MB
elixir 1.18 (precompiled, otp-27)     ~  30 MB
node, python3, git, ripgrep (apk)     ~ 180 MB
warden (compiled) + vsock_port        ~   2 MB
--------------------------------------------
rootfs 249 MB · disk image ~270 MB allocated, 3 GB apparent
```

Host build dependencies, the whole list: `python3`, `mise` (the erlang,
elixir and zig pins), and `brew install e2fsprogs`. Measured: a warm rebuild
is under a minute; downloads are cached in `build/guest-cache`.

Two details that cost an afternoon each, written down so they are not paid
again: the kernel silently ignores a malformed initramfs and falls back to
mounting root itself, so a cpio header miscount presents as a boot panic
three layers away; and remounting `/` read-write needs `/proc/mounts`, so the
initramfs *moves* `/proc` into the new root rather than unmounting it.

#### mise is gone from the guest, and the reason is honest

The old image shipped mise so the agent could get "the runtime the project
pins". But the sandbox has **no network device** — mise could never install
anything in there; it could only ever select among runtimes already baked in,
which is what plain `/usr/bin` paths do with less machinery. The runtimes are
the image's, reported by `info` as `runtimes=erlang@27,elixir@...`, and adding
a language is adding its package to `PACKAGES` in `mkrootfs.py`.

The musl constraint stands unchanged: runtimes must be Alpine packages (or
portable bytecode, as Elixir is). Moving to a glibc base to widen the
catalogue remains the first thing to reconsider if the agent starts wanting
runtimes Alpine does not package.

#### The control plane's runtime is a concrete path

`warden` runs on `/usr/bin/erl` — the erlang27 package's own binary — with
`ERL_LIBS` from `/etc/crucible-erl-libs`. §7.3 requires that warden survive
anything the agent does; nothing the agent edits can move a package-installed
`/usr/bin/erl`, which is the same property the old mise-bypassing symlinks
bought, with fewer moving parts.

Alpine rather than Debian: musl, no systemd, and a boot-to-BEAM path measured
in hundreds of milliseconds — **0.54 s to warden-ready with the native-built
image**, matching the containerised build it replaced. `git` is in the image
because the diff engine is git (§7.4) — that is the one dependency worth its
size.

OTP 27 specifically, for two reasons beyond currency: its built-in `json`
module removes the last dependency from the guest (§6.4). One language and
one build tool across the guest — warden and workspace are both `mix`.

Init is not OpenRC. `/sbin/init` is a ~40-line shell script that mounts
`/proc`, `/sys`, `/base` (virtiofs), starts the warden, and execs. Target: **VM
boot to warden-ready under 2 seconds.** Every second here is a second a user
waits before the model can do anything, on every resume.

**Shipping it.** The image goes in `Contents/Resources/` compressed, or is
fetched on first launch. Decided in §14 — 270 MB is right at the boundary where
bundling stops being polite.

### 6.3 Per-session disks are APFS clones

```swift
// Copy-on-write. Instant, and zero additional blocks until written.
clonefile(goldenPath, sessionDiskPath, 0)
```

This is the detail that makes "a VM per session" affordable. A 2 GB sparse image
clones in microseconds and consumes only what the session actually writes.
Sessions get real isolation — their own filesystem, their own installed
packages, their own everything — at the cost of the delta. Deleting a session is
deleting the clone.

Fall back to a plain copy if `clonefile` fails (non-APFS volume), and say so in
the inspector rather than silently taking 2 GB per session.

### 6.4 The vsock channel

Host side: `VZVirtioSocketDevice.connect(toPort: 1024)` yields a
`VZVirtioSocketConnection` with a real file descriptor. Wrap it in a
`DispatchIO` or `NWConnection`-style reader.

Guest side is the interesting half, because **the BEAM cannot open an AF_VSOCK
socket.** `gen_tcp` has no vsock support and `socat` is a 400 KB dependency for
one pipe. So `Guest/vsock_port/vsock_port.c` is a ~120-line port program:

```c
/* AF_VSOCK listener ↔ stdio, spoken as an Erlang port with {packet, 4}.
 * The BEAM opens it with open_port({spawn_executable, ...},
 * [{packet, 4}, binary, exit_status]) and gets framed messages with no
 * parsing of its own. */
```

Framing is 4-byte big-endian length, which is exactly what `{packet, 4}` means
on the Erlang side and what a two-line read loop means on the Swift side.
Payloads are **JSON**, decoded on the guest with OTP 27's built-in `:json`
module — no Jason, no dependency, no mix deps to vendor into the image.

```
host → guest   {"id": 41, "op": "invoke", "tool": "grep_ast", "args": {...}}
guest → host   {"id": 41, "ok": true,  "result": "...", "took_ms": 12}
guest → host   {"id": 41, "ok": false, "error": "no such tool: grep_ast",
                "kind": "unknown_tool"}
guest → host   {"event": "log", "level": "warn", "text": "..."}   // unsolicited
```

Every request carries a host-assigned `id`; every response echoes it. Unsolicited
events (logs, module-load notices, progress) carry no `id`. The host times out
every request — default 30 s, `shell` 300 s — and a timeout is reported to the
model as a tool result, never as a failed turn. Same principle the C agent
applies to a refused confirmation: the model gets told and chooses again.

### 6.5 VM lifecycle

- **Created** lazily, on the session's first tool call — not on session creation.
  A conversation that never touches a file never boots a VM.
- **Parked** with the session (§4.4): `stop()`, keep the clone, keep the module
  manifest. Resume re-boots and replays the manifest (§7.3).
- **Capped**: `maxRunningVMs`, default 2, derived from physical memory. In
  practice the live session's VM plus at most one lingering VM, since only one
  session is live at a time (§4.3) — a VM outlives its session's parking only
  long enough to finish an in-flight `propose`.
- **Crash**: `VZVirtualMachineDelegate.guestDidStop` / `didStopWithError`. The
  session goes to `.parked` with a transcript item saying so; the next tool call
  re-boots and replays. The model sees a tool result explaining that the sandbox
  restarted and that in-memory state is gone but `/work` survived — because it
  did, it is on the disk.

---

### 6.6 Provisioning: packages as configuration, never as images

*Specified 2026-08-23. Not started.*

The gap this closes: the guest has no network **by design**, so the agent can
never install anything, and the golden image's package list is a constant in
`mkrootfs.py` that only a developer edits. Real projects need system
components — a compiler, a database client, an image library — and today the
answer is "rebuild the image," which is no answer for a user.

**The user's primitive is a package list, not an image.** One new overlay
key, with §8.5's semantics exactly:

| key | meaning |
|---|---|
| `guest_packages` | Alpine package names, comma-separated; a session's list REPLACES the project's; empty = explicitly none, nil = fall through |

Set it the way everything else is set — conversationally in the config
project ("give this project rust and cmake"), at whichever layer fits. The
agent participates through the loop that already exists: a missing command is
a tool result, the model says what it needs, the user tells the config agent,
the next session open has it. No new machinery surfaces the request because
requests already have a surface.

#### The mechanism: a provision overlay over virtiofs

Two facts the codebase already established do all the work. Alpine packages
are tarballs, and the HOST knows how to resolve and extract them natively —
that is precisely what `mkrootfs.py` does at image-build time. And the guest
already has a boot-time ingestion path — virtiofs, copy, unmount — the
`/base` pattern whose security argument (§8.2) is unchanged by a second
read-only share.

1. **Resolve and cache, host-side.** The dependency closure is computed
   against APKINDEX (the ~100-line resolver from `mkrootfs.py`, ported to
   Swift — the app already holds the outgoing-network entitlement, and the
   Alpine mirror URLs are pinned host-side, not configurable, the §8.3
   rule). Packages are verified against the index checksums and extracted
   into a **content-addressed cache**: one directory per package-set hash,
   in Application Support, built once, shared across projects. Packages the
   golden image already carries are excluded from the closure.
2. **Overlay at boot.** The cache directory rides into the guest as a second
   read-only virtiofs share (`/provision`); `crucible-init` copies it into
   `/` once per session disk — a marker file, the seed-work pattern — and
   unmounts. Session disks persist, so later boots skip the copy. apk
   install scripts do not run; the busybox precedent (§6.2) says how to
   handle the rare package that needs one, and most need nothing.
3. **Say what happened.** The session header carries it — `· +14 packages` —
   and the first boot of a new set shows "provisioning" during the fetch.

**Failure is named, never fatal.** An unknown package is reported at session
open by name ("no Alpine package `gcc-13` — did you mean `gcc`?"),
and the session boots without it. A network failure during fetch degrades the
same way: the session runs on the golden image, the note says which packages
are missing and why.

#### What this deliberately is not

- **Not per-project images.** Derived images would multiply kernel-matching
  liabilities, cost gigabytes per variant, and rebuild on every golden
  update; an overlay directory is content-addressed bytes that compose with
  ANY golden. No ext4 surgery from macOS either (possible via debugfs,
  miserable, and unnecessary).
- **Not guest-side installation.** Nothing inside the VM fetches anything;
  the no-network property survives untouched. The host fetches, from pinned
  URLs, and the guest receives files — the same trust shape as `fetch`.
- **Not a general software manager.** Alpine/musl packages only (~20k of
  them), stated plainly in the config agent's briefing. The recorded escape
  hatch for glibc-only software or services needing real post-install setup
  is a custom golden: `make guest` with a `PACKAGES` override — a build-time
  operation for power users, and the image it produces still composes with
  every project's overlay.

**Deferred, deliberately:** version-pinned language runtimes
(`guest_runtimes` fetching Node 20 vs 24, Python 3.11 vs 3.12 from pinned
sources). Alpine's shipped versions cover v1; pinning is a second key and a
second source of truth, and it waits until the plain package path has been
lived with.

**Correctness bar, set now:** the resolver port is pinned against
`mkrootfs.py`'s closures (same inputs, same package sets); the gate boots a
session with a provisioned package and runs its binary; a bogus package name
must produce the named note and a working session.
