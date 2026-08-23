## 8. Security

### 8.1 Entitlements

```xml
<key>com.apple.security.app-sandbox</key>              <true/>
<key>com.apple.security.virtualization</key>           <true/>
<key>com.apple.security.files.user-selected.read-write</key> <true/>
<key>com.apple.security.files.bookmarks.app-scope</key> <true/>
```

Notes that matter:

- `com.apple.security.virtualization` is required and is available with an
  ordinary Developer ID — it is **not** one of the restricted entitlements that
  needs a request to Apple.
- `com.apple.vm.networking` **is** restricted and needs Apple's approval. We do
  not want it: bridged networking is exactly the capability this design is built
  to withhold. NAT (`VZNATNetworkDeviceAttachment`) needs no special entitlement
  if §8.3's escape hatch is ever enabled.
- **App Sandbox plus Virtualization plus a 16 GB mmap is the combination to
  prove first.** It is a Milestone 1 gate (§12), not a Milestone 6 discovery.
  If App Sandbox turns out to fight the guest disk or the virtiofs share,
  the fallback is Hardened Runtime without App Sandbox — still notarisable,
  strictly worse, and a decision to make early and deliberately.

### 8.2 The threat model, stated

**What we defend against:** a model — steered by its own reasoning, or by
instructions embedded in a file it reads — taking an action the user did not
intend on the user's machine.

**How:** it has no ability to. The tool implementations do not exist on the
host. `shell` runs in a guest with no network device, whose only view of the
user's files is a read-only mount of one directory the user picked, and whose
only channel out is a patch the user reads and approves file by file.

Prompt injection therefore degrades from *arbitrary code execution on the
developer's laptop* to *a bad patch, shown to a human, in a diff view*. That is
the entire argument for the architecture, and it is worth stating in one
sentence because everything expensive in this document is bought with it.

**What we do not defend against:**

- A guest-to-host escape through Virtualization.framework itself. We inherit
  Apple's hypervisor boundary and do not second-guess it.
- A user approving a malicious patch. The sheet's job is to make the change
  legible; it cannot make the user read it.
- Model output that is simply wrong. Different problem.

**The attack surface we actually own** is the vsock protocol and the patch
applier. Both are small, both are pure functions over untrusted input, and both
are tested adversarially. Warden validates every host message; the host
validates every guest path (§7.4 step 8).

### 8.3 Network: host-mediated egress, never a NIC

Some tasks need the network — reading a doc, checking an API. An earlier
draft of this section sketched a NAT device behind a settings toggle. That
design is rejected, and the reason generalises: **any in-guest network is
policed by in-guest code, and everything in the guest is inside the blast
radius.** The agent's own code runs in an Erlang VM (the workspace node), so
"network for the BEAM only" grants network to the agent by definition; and
the agent has root, so nftables rules, a de-privileged socket owner, even the
warden's own binaries on disk are advisory the moment they stand between the
agent and something it was told to want. Today none of that matters because
there is nothing inside the guest worth protecting — the security story is
that nothing gets out. A NIC would make warden integrity security-critical,
which is exactly the property root cannot be made to respect.

So the guest keeps **zero network devices, permanently**, and network exists
only as `fetch`: a tool the **host** executes, under host-side policy the
agent cannot reach. The guest is not in the loop at all — the call goes
model → host, and the host answers it like any tool result.

The policy, all of it enforced in `NetworkPolicy` on the host:

- **Default off.** A project with an empty allowlist has no `fetch` in its
  tool surface at all — the system turn is unchanged, nothing is advertised
  that will be refused, and the network-off project keeps today's guarantees
  exactly. Granting domains changes the surface, which re-prefills that
  project once (the disk-cached prefix makes it one cold start).
- **A per-project domain allowlist**, edited by a person in the app, stored
  with the project. An entry matches its exact host; a `*.example.com` entry
  matches subdomains. Nothing the model does can grow the list; a refused
  fetch is a tool result naming the domain, so the user can decide.
- **GET only, https only, port 443 only,** no userinfo, no IP literals
  unless explicitly listed. Redirects are re-checked against the allowlist
  hop by hop — a 302 to an unlisted host fails the request, because a
  redirect is the classic way an allowed domain becomes a proxy for an
  arbitrary one.
- **A response byte cap** (256 KB), enforced during download rather than
  after, and a timeout. Bodies come back as text in the tool result;
  binary content is reported, not delivered.
- **Every request is a tool call in the transcript**: URL, outcome, size.
  There is no quiet path.

#### What this keeps, and what it breaks — stated for the README

Execution sandboxing is kept intact: code in the guest still cannot open a
socket, scan, listen, or exfiltrate on its own — the only egress is a request
the host chooses to perform. What is broken, by construction and not by
implementation, is **perfect confidentiality**: an outbound channel, however
mediated, is a channel. A prompt injection in a file the model reads could
previously send nothing anywhere; with `fetch` granted it can encode project
contents into request URLs aimed at allowed hosts. The allowlist narrows the
recipients and the log makes it visible; nothing closes the channel while it
exists. Fetched content is also new injection input, so the loop can
self-amplify. The honest posture: for projects where confidentiality is the
point, the answer is the default — leave the list empty. The README says
this in as many words rather than hiding it in a settings tooltip.

Future, deliberately not in v1: a package-mirror proxy (hex/npm read-only)
for dependency installation — most of the remaining utility at a fraction of
the general-egress risk — and binary delivery into `/work` for fetched
archives.

### 8.4 The model directory

16 GB of weights are mmapped from a user-selected folder held by bookmark. If
the folder disappears between launches, fail with a clear message and a picker,
not a crash — `qwasar_engine_load` returning `NULL` with a filled `err` is a
normal, expected outcome and the UI should treat it as one.

### 8.5 Sandbox configuration: three layers, and the config project

Sandbox settings — the network allowlist (§8.3), guest memory and CPUs, the
tool-call timeout, the fetch cap — apply at three layers: **global**
(`sandbox.json` in the store), **per-project** (`Project.sandbox`), and
**per-session** (`SessionRecord.sandbox`). Every layer is the same
`SandboxOverlay` shape with every field optional, and there is exactly one
resolution rule: **field-wise, most specific non-nil wins** — session, then
project, then global, then the built-in default.

Two consequences of that rule are load-bearing and are pinned by
`SandboxOverlaySuite`:

- **Replace, never merge.** A session that sets `network_allowlist` replaces
  the project's list rather than adding to it, because "replace" is a rule a
  person can predict from the value they typed and "merge" is a rule they
  have to go and check.
- **Empty is an opinion; nil is silence.** An empty network list at the
  session layer turns network OFF over a global grant — which is how a
  confidential session opts out of a permissive global without touching it.
  Provenance (which layer decided each field) is part of the API, so
  `config_show` can say where a value came from rather than leaving the user
  to reverse-engineer it from behaviour.

Settings are resolved once, at session open, and fixed for the boot — the
tool surface is the system turn (§2.2), so a live session keeps what it was
prefilled with and changes apply at the next open.

**The config project.** A built-in project (fixed id, synthesized at launch,
not removable) named *Crucible Config*, whose sessions manage this
configuration conversationally. Its tools run on the HOST with no sandbox —
but *unsandboxed is not unbounded*: the surface is three purpose-built
operations, `config_show` / `config_set` / `config_clear`, not a shell.
There is no path from a config session to the filesystem, the network, or
another project's files; the blast radius of the special project is the
configuration itself, which is precisely its job. Mutations go through the
same main-actor write path the UI uses, so a running config session, the
sidebar and the store can never disagree.

One deliberate asymmetry: the config session **can grow a project's network
allowlist**, which §8.3 says only a person may do. The chain is still a
person — a config session acts only on what the user typed into it, and its
transcript shows every change — but text a user pastes into a config session
is trusted the way text typed into the sheet is, so the same social-
engineering caution applies. The config project itself has no sandbox and no
fetch: nothing a *project's* compromised agent produces can reach a config
session's input except through the user choosing to paste it.

---
