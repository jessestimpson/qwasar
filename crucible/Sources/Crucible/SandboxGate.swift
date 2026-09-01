// SandboxGate.swift -- M2's gate, headless.
//
// PLAN.md 12, M2: "boot-to-warden-ready under 2 s". So it is timed, not
// asserted vaguely, and the whole chain is exercised: clone the golden image,
// boot it with no network device, connect the vsock, and hold a conversation
// with the warden.
//
// Everything Virtualization touches runs on the main actor (PLAN.md 3.4, fifth
// edge). The caller keeps a run loop alive so the main queue can actually do
// that work, rather than blocking it and wondering why nothing happens.

import Foundation
import CrucibleKit

@MainActor
enum SandboxGate {

    static func run(guestDir: URL, projectDir: URL?) async -> Int32 {
        print("=== Crucible M2 gate — the sandbox ===")

        let fm = FileManager.default
        let golden = guestDir.appendingPathComponent("disk.img")
        let kernel = guestDir.appendingPathComponent("Image")   // unwrapped by mkimage.sh
        let initramfs = guestDir.appendingPathComponent("initramfs")

        for f in [golden, kernel, initramfs] where !fm.fileExists(atPath: f.path) {
            print("  missing \(f.path)")
            print("  run `make guest` first")
            return 1
        }

        // --- clone --------------------------------------------------------
        let scratch = fm.temporaryDirectory
            .appendingPathComponent("crucible-m2-\(UUID().uuidString)")
        try? fm.createDirectory(at: scratch, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: scratch) }

        let image = GuestImage(golden: golden)
        let disk = scratch.appendingPathComponent("disk.img")
        guard let clone = try? image.clone(to: disk) else {
            print("  clone failed"); return 1
        }
        print(String(format: "\n-- disk\n  cloned %@ in %.4fs (%@)",
                     golden.lastPathComponent, clone.seconds, clone.method.rawValue))
        print("  golden allocates \(GuestImage.allocatedBytes(golden) / 1_000_000) MB, "
              + "the clone \(GuestImage.allocatedBytes(disk) / 1_000_000) MB")

        // --- the stand-in project -----------------------------------------
        //
        // The gate plays the user: a temp copy of `projectDir` with a
        // hand-built minimal .git (HEAD on an unborn main, empty objects and
        // refs -- exactly what `git init` leaves, written with FileManager
        // because the gate binary is sandboxed and spawns nothing). The
        // guest must seed its shadow from it, bind-mount the shadow over
        // /work/.git, and leave this real .git untouched by everything the
        // tools do -- which the checks below read back from the host side.
        var project: URL?
        if let projectDir {
            do {
                let p = scratch.appendingPathComponent("project")
                try fm.copyItem(at: projectDir, to: p)
                let git = p.appendingPathComponent(".git")
                try fm.createDirectory(at: git.appendingPathComponent("objects"),
                                       withIntermediateDirectories: true)
                try fm.createDirectory(at: git.appendingPathComponent("refs/heads"),
                                       withIntermediateDirectories: true)
                try fm.createDirectory(at: git.appendingPathComponent("hooks"),
                                       withIntermediateDirectories: true)
                try "ref: refs/heads/main\n"
                    .write(to: git.appendingPathComponent("HEAD"),
                           atomically: true, encoding: .utf8)
                try "[core]\n\trepositoryformatversion = 0\n\tbare = false\n"
                    .write(to: git.appendingPathComponent("config"),
                           atomically: true, encoding: .utf8)
                project = p
                print("\n-- stand-in project")
                print("  copied \(projectDir.lastPathComponent), gave it a real .git (unborn main)")
            } catch {
                print("  fixture FAILED: \(error)"); return 1
            }
        }

        // --- boot ---------------------------------------------------------
        let console = scratch.appendingPathComponent("console.log")
        var cfg = SandboxConfig(diskImage: disk, kernel: kernel, initramfs: initramfs,
                                consoleLog: console, workDirectory: project)
        cfg.memoryBytes = 2 * 1024 * 1024 * 1024
        cfg.cpuCount = 2

        let host = SandboxHost(config: cfg)
        print("\n-- configuration")
        do {
            let vzc = try host.makeConfiguration()
            print("  \(vzc.cpuCount) cpu · \(vzc.memorySize / 1_048_576) MB "
                  + "· \(vzc.storageDevices.count) disk "
                  + "· \(vzc.directorySharingDevices.count) share "
                  + "· \(vzc.socketDevices.count) vsock"
                  + "· \(vzc.entropyDevices.count) entropy")
            // The security property PLAN.md 8.2 rests on, asserted rather than
            // assumed: there is no network device because none was configured.
            print("  network devices: \(vzc.networkDevices.count) "
                  + (vzc.networkDevices.isEmpty ? "— the guest has no way out" : "— WRONG, expected none"))
            if !vzc.networkDevices.isEmpty { return 1 }
        } catch {
            print("  FAILED: \(error)"); return 1
        }

        print("\n-- boot")
        let t0 = Date()
        do { try await host.start() }
        catch { print("  FAILED: \(error)"); dumpConsole(console); return 1 }
        print(String(format: "  started in %.2fs", Date().timeIntervalSince(t0)))

        // --- connect ------------------------------------------------------
        //
        // The warden comes up when it comes up; the host retries rather than
        // assuming a boot order it does not control.
        var fd: Int32 = -1
        var attempts = 0
        while Date().timeIntervalSince(t0) < 30 {
            attempts += 1
            if let got = try? await host.connectControl() { fd = got; break }
            try? await Task.sleep(for: .milliseconds(100))
        }
        guard fd >= 0 else {
            print("  FAILED: no vsock connection after 30s (\(attempts) attempts)")
            dumpConsole(console)
            await host.stop()
            return 1
        }
        let ready = Date().timeIntervalSince(t0)
        print(String(format: "  vsock connected after %.2fs (%d attempts)", ready, attempts))

        // --- talk to the warden -------------------------------------------
        let channel = VsockChannel(fileDescriptor: fd)
        channel.onEvent { ev in
            if let t = ev.text { print("  [guest \(ev.level ?? "log")] \(t)") }
        }
        channel.start()

        var failures = 0
        print("\n-- warden")
        do {
            let pong = try await channel.send(op: "ping", timeout: 10)
            let ok = pong.ok == true && pong.result == "pong"
            print("  ping → \(pong.result ?? pong.error ?? "?") (\(pong.took_ms ?? 0) ms)")
            if !ok { failures += 1 }

            let info = try await channel.send(op: "info", timeout: 10)
            print("  info → \(info.result ?? info.error ?? "?")")
            if info.ok != true { failures += 1 }

            if let project {
                let ls = try await channel.send(op: "list", args: ["path": .string(".")], timeout: 10)
                let n = (ls.result ?? "").split(separator: "\n").count
                print("  list . → \(n) entries in /work")
                if ls.ok != true || n == 0 { failures += 1 }

                // The containment rule, from the guest's own side.
                let esc = try await channel.send(op: "list", args: ["path": .string("../../etc")], timeout: 10)
                let refused = esc.ok == false && esc.kind == "path"
                print("  list ../../etc → \(refused ? "refused: \(esc.error ?? "")" : "NOT REFUSED")")
                if !refused { failures += 1 }

                // The six tools, end to end over the wire, against the user's
                // own tree (spec 7.4): every write lands in the mounted
                // project as it happens, and the real .git must not gain a
                // byte -- both read back from the host side below.
                failures += try await exerciseTools(channel, project: project)
            }

            // The agent's node (PLAN.md 7.3): reachable, usable, and — the
            // part that matters — survivable. A workspace crash must leave the
            // control plane answering and bring the node back, because M4 puts
            // agent-written code on it and agent-written code crashes.
            let ws = try await channel.send(op: "workspace", timeout: 15)
            print("  workspace → \(ws.result ?? ws.error ?? "?")")
            if !(ws.result ?? "").hasPrefix("up") { failures += 1 }

            let ev = try await channel.send(op: "eval",
                                            args: ["code": .string("1..10 |> Enum.sum()")],
                                            timeout: 20)
            print("  eval 1..10 |> Enum.sum() → \(ev.result ?? ev.error ?? "?")")
            if ev.result != "55" { failures += 1 }

            _ = try await channel.send(op: "workspace_kill", timeout: 15)
            // The bridge must still answer instantly with the node gone.
            let afterKill = try await channel.send(op: "ping", timeout: 5)
            print("  ping with the workspace dead → \(afterKill.result ?? "?")")
            if afterKill.ok != true { failures += 1 }

            // And it must come back on its own, without anything asking.
            var recovered = false
            for _ in 0..<25 {
                try await Task.sleep(for: .milliseconds(400))
                let s = try await channel.send(op: "workspace", timeout: 15)
                if (s.result ?? "").hasPrefix("up") {
                    print("  workspace recovered → \(s.result ?? "")")
                    recovered = true
                    break
                }
            }
            if !recovered { print("  workspace DID NOT recover"); failures += 1 }

            // Every advertised tool must be dispatchable.
            //
            // Two bugs of this exact shape reached a real model before this
            // check existed: the schema said `bash` while the guest op was
            // `shell`, and the schema said `elixir` while the op was `eval`. In
            // both cases the gate passed, because the gate called the op names
            // directly rather than the names the MODEL is given. A tool the
            // model is told it has and cannot call is worse than no tool.
            print("  every advertised tool is dispatchable:")
            var missing: [String] = []
            for name in ToolSurface.names.sorted() {
                let probe = try await channel.send(op: name, timeout: 20)
                if probe.kind == "unknown_op" { missing.append(name) }
            }
            if missing.isEmpty {
                print("      all \(ToolSurface.names.count) reach a handler")
            } else {
                print("      NOT DISPATCHABLE: \(missing.joined(separator: ", "))")
                failures += 1
            }

            // Self-modification (PLAN.md 7.2): the agent writes a tool,
            // hot-loads it, and calls it — with no restart and no change to the
            // tool surface, which is what makes it free (§2.2).
            failures += try await exerciseSelfModification(channel)

            // While a slow request is outstanding, the channel must still
            // answer. The
            // bridge used to call dispatch inline, which meant one slow op
            // stalled everything -- and no test it had could have shown that,
            // because every op it had was instant.
            async let slow = channel.send(op: "sleep", args: ["ms": .int(3000)], timeout: 15)
            try await Task.sleep(for: .milliseconds(150))
            let t0 = Date()
            let mid = try await channel.send(op: "ping", timeout: 5)
            let midMs = Date().timeIntervalSince(t0) * 1000
            let responsive = mid.ok == true && midMs < 1000
            print(String(format: "  ping during a 3s request → %@ after %.0fms",
                         mid.result ?? "?", midMs))
            if !responsive { failures += 1 }
            let slept = try await slow
            print("  the slow request finished → \(slept.result ?? slept.error ?? "?") "
                  + "(\(slept.took_ms ?? 0) ms)")
            if slept.ok != true { failures += 1 }

            // An unknown op must be answered, not dropped: silence would make
            // the host wait out a timeout for a mistake it could be told about.
            let bogus = try await channel.send(op: "nonsense", timeout: 10)
            let told = bogus.ok == false && bogus.kind == "unknown_op"
            print("  unknown op → \(told ? "answered: \(bogus.error ?? "")" : "NOT ANSWERED PROPERLY")")
            if !told { failures += 1 }
        } catch {
            print("  FAILED: \(error)")
            failures += 1
        }

        channel.close()
        print("\n-- shutdown")
        await host.stop()
        print("  stopped")

        print("\n-- gate")
        line("guest boots with no network device", true)
        line("vsock control channel connects", true)
        line(String(format: "warden ready in %.2fs (target < 2s)", ready), ready < 2.0)
        line("warden answers correctly", failures == 0)
        let pass = failures == 0 && ready < 2.0
        print("\n  \(pass ? "GATE PASSES" : "gate incomplete — see above")")
        // Always, not only on failure: the guest's boot timeline is the number
        // this milestone is about, and hiding it on success is how a regression
        // from 0.6 s back to 5 s goes unnoticed.
        dumpConsole(console)
        return pass ? 0 : 1
    }


    /// The six file tools, over the wire, against the mounted tree.
    /// Written as a round trip -- create, search, read, edit, verify --
    /// because each tool checked in isolation proves less than the sequence an
    /// agent actually performs. `project` is the HOST side of the same
    /// directory, read directly: writes must be there, and the real .git
    /// must not be.
    private static func exerciseTools(_ channel: VsockChannel, project: URL) async throws -> Int {
        var bad = 0
        func check(_ label: String, _ ok: Bool, _ detail: String) {
            print("  \(label) → \(detail)")
            if !ok { bad += 1 }
        }

        let body = "alpha\nbeta\ngamma\n"
        let w = try await channel.send(op: "write",
                                       args: ["path": .string("m3probe.txt"),
                                              "content": .string(body)], timeout: 10)
        check("write", w.ok == true, w.result ?? w.error ?? "?")

        let g = try await channel.send(op: "grep",
                                       args: ["pattern": .string("bet[a]"),
                                              "path": .string(".")], timeout: 15)
        let foundRelative = (g.result ?? "").contains("m3probe.txt:2:beta")
        check("grep", g.ok == true && foundRelative, (g.result ?? g.error ?? "?")
              .split(separator: "\n").first.map(String.init) ?? "?")

        let e = try await channel.send(op: "edit",
                                       args: ["path": .string("m3probe.txt"),
                                              "old": .string("beta"),
                                              "new": .string("BETA")], timeout: 10)
        check("edit", e.ok == true, e.result ?? e.error ?? "?")

        let r = try await channel.send(op: "read",
                                       args: ["path": .string("m3probe.txt")], timeout: 10)
        check("read", r.result == "alpha\nBETA\ngamma\n", (r.result ?? "?").debugDescription)

        // Ambiguity is refused rather than guessed -- the contract the model was
        // given, inherited from qw_edit_apply and pinned by the edit suite.
        _ = try await channel.send(op: "write",
                                   args: ["path": .string("dup.txt"),
                                          "content": .string("x = 1;\ny = 2;\nx = 1;\n")],
                                   timeout: 10)
        let amb = try await channel.send(op: "edit",
                                         args: ["path": .string("dup.txt"),
                                                "old": .string("x = 1;"),
                                                "new": .string("x = 3;")], timeout: 10)
        check("edit (ambiguous)", amb.ok == false && amb.kind == "ambiguous",
              amb.error ?? "NOT REFUSED")

        let sh = try await channel.send(op: "shell",
                                        args: ["command": .string("pwd && wc -l < m3probe.txt")],
                                        timeout: 20)
        let inWork = (sh.result ?? "").contains("/work")
        check("shell", sh.ok == true && inWork,
              (sh.result ?? sh.error ?? "?").replacingOccurrences(of: "\n", with: " "))

        // The property everything above now rests on (spec 7.4): the guest's
        // write is on the HOST, in the user's tree, as it happens. One
        // guest-side sync first, so the check tests the sharing rather than
        // racing the page cache.
        _ = try await channel.send(op: "shell",
                                   args: ["command": .string("sync")], timeout: 20)
        let hostCopy = try? String(contentsOf: project.appendingPathComponent("m3probe.txt"),
                                   encoding: .utf8)
        check("a guest write is visible on the host", hostCopy == "alpha\nBETA\ngamma\n",
              hostCopy.map { "host sees \($0.count) bytes" } ?? "NOT ON THE HOST")

        // The vault (spec 7.4): git inside the guest is a live, private copy
        // of the project's history -- usable enough to commit to -- while
        // the REAL .git on the host must not gain a single byte from any of
        // it, hooks included. This is the property the bind mount buys, and
        // the reason a compromised agent cannot plant anything the user's
        // own git would execute.
        let gs = try await channel.send(op: "shell",
                                        args: ["command": .string("cd /work && git status --porcelain >/dev/null && echo repo-ok")],
                                        timeout: 30)
        check("the guest has a working repo (the shadow)",
              (gs.result ?? "").contains("repo-ok"),
              (gs.result ?? gs.error ?? "?").trimmingCharacters(in: .whitespacesAndNewlines))

        let commit = try await channel.send(op: "shell",
                                            args: ["command": .string("cd /work && git add -A && git commit -qm probe && echo hook > .git/hooks/gate-probe && sync && git rev-parse --short HEAD")],
                                            timeout: 30)
        check("the agent commits into the shadow", commit.ok == true,
              (commit.result ?? commit.error ?? "?").trimmingCharacters(in: .whitespacesAndNewlines))

        let realGit = project.appendingPathComponent(".git")
        let objects = (try? FileManager.default
            .contentsOfDirectory(atPath: realGit.appendingPathComponent("objects").path)) ?? []
        check("the real .git gained no objects", objects.isEmpty,
              objects.isEmpty ? "objects/ still empty" : "\(objects.count) entries appeared")
        let hookLanded = FileManager.default
            .fileExists(atPath: realGit.appendingPathComponent("hooks/gate-probe").path)
        check("a guest-written hook never reaches the real .git", !hookLanded,
              hookLanded ? "IT LANDED" : "hooks/ untouched")
        let head = try? String(contentsOf: realGit.appendingPathComponent("HEAD"),
                               encoding: .utf8)
        check("the real HEAD never moved", head == "ref: refs/heads/main\n",
              head?.trimmingCharacters(in: .whitespacesAndNewlines) ?? "?")

        // Refresh Git's guest half (spec 7.4): the op drops the seed stamp,
        // arming a re-seed for the next boot. The re-seed behaviour itself
        // is pinned by test_mount_work.sh; here the op must answer and the
        // stamp must actually be gone.
        let refresh = try await channel.send(op: "git_refresh", timeout: 15)
        check("git_refresh arms a re-seed", refresh.ok == true,
              refresh.result ?? refresh.error ?? "?")
        let stamp = try await channel.send(op: "shell",
                                           args: ["command": .string("test -f /var/lib/crucible/git-seeded && echo present || echo gone")],
                                           timeout: 15)
        check("the seed stamp is dropped", (stamp.result ?? "").contains("gone"),
              (stamp.result ?? stamp.error ?? "?").trimmingCharacters(in: .whitespacesAndNewlines))

        // And the baked-in BEAM is actually usable by the agent.
        let elixir = try await channel.send(op: "shell",
                                            args: ["command": .string("elixir -e 'IO.puts(:erlang.system_info(:otp_release))'")],
                                            timeout: 30)
        check("shell (guest runtime)", elixir.ok == true && (elixir.result ?? "").contains("27"),
              (elixir.result ?? elixir.error ?? "?").trimmingCharacters(in: .whitespacesAndNewlines))

        return bad
    }


    /// `define` → `tools` → `invoke`, then a crash and a replay.
    ///
    /// The last part is the one that matters. A tool the agent wrote must
    /// survive the node it lives on dying, because agent-written code crashes
    /// that node — and the manifest lives in warden precisely so it can be
    /// replayed (PLAN.md 7.3).
    private static func exerciseSelfModification(_ channel: VsockChannel) async throws -> Int {
        var bad = 0
        func check(_ label: String, _ ok: Bool, _ detail: String) {
            print("  \(label) → \(detail)")
            if !ok { bad += 1 }
        }

        // A Swift *raw* string: Elixir's #{} interpolation is not a Swift escape
        // sequence, and in a plain multiline literal it is a compile error.
        let source = #"""
        defmodule WordCount do
          @behaviour Crucible.Skill
          def name, do: "wordcount"
          def schema, do: %{"description" => "counts words in a file under /work",
                            "args" => ["path"]}
          def run(%{"path" => path}) do
            full = Path.expand(path, "/work")
            if String.starts_with?(full, "/work") do
              case File.read(full) do
                {:ok, text} -> {:ok, "#{length(String.split(text))} words"}
                {:error, r} -> {:error, "cannot read: #{r}"}
              end
            else
              {:error, "outside /work"}
            end
          end
        end
        """#

        let d = try await channel.send(op: "define",
                                       args: ["source": .string(source)], timeout: 60)
        check("define", d.ok == true && (d.result ?? "").contains("wordcount"),
              (d.result ?? d.error ?? "?").split(separator: "\n").first.map(String.init) ?? "?")

        let t = try await channel.send(op: "skills", timeout: 15)
        check("skills", (t.result ?? "").contains("wordcount"), t.result ?? t.error ?? "?")

        let i = try await channel.send(op: "invoke",
                                       args: ["name": .string("wordcount"),
                                              "args": .object(["path": .string("m3probe.txt")])],
                                       timeout: 30)
        check("invoke wordcount", i.ok == true && (i.result ?? "").contains("3 words"),
              i.result ?? i.error ?? "?")

        // Two modules cannot answer to one name.
        let clash = try await channel.send(op: "define", args: ["source": .string(#"""
        defmodule OtherCounter do
          @behaviour Crucible.Skill
          def name, do: "wordcount"
          def run(_), do: {:ok, "nope"}
        end
        """#)], timeout: 60)
        // The define itself succeeds (the module loads); the registry refuses
        // the name, and `tools` must still resolve to the original.
        let stillOriginal = try await channel.send(op: "invoke",
                                                   args: ["name": .string("wordcount"),
                                                          "args": .object(["path": .string("m3probe.txt")])],
                                                   timeout: 30)
        check("a name clash does not hijack the tool",
              (stillOriginal.result ?? "").contains("3 words"),
              "\(clash.ok == true ? "loaded" : "refused"), invoke still gives "
              + (stillOriginal.result ?? "?"))

        // Now kill the node the tool lives on, and require it back.
        _ = try await channel.send(op: "workspace_kill", timeout: 15)
        var replayed = false
        for _ in 0..<30 {
            try await Task.sleep(for: .milliseconds(400))
            let r = try await channel.send(op: "invoke",
                                           args: ["name": .string("wordcount"),
                                                  "args": .object(["path": .string("m3probe.txt")])],
                                           timeout: 30)
            if (r.result ?? "").contains("3 words") { replayed = true; break }
        }
        check("the tool survives its node dying", replayed,
              replayed ? "invoke works again after the workspace restarted"
                       : "NOT REPLAYED")
        return bad
    }


    private static func line(_ what: String, _ ok: Bool) {
        print("  \(ok ? "OK " : "NO ") \(what)")
    }

    private static func dumpConsole(_ url: URL) {
        guard let text = try? String(contentsOf: url, encoding: .utf8), !text.isEmpty else {
            print("\n  (the guest console is empty — it did not get far enough to say anything)")
            return
        }
        print("\n-- guest console")
        for line in text.split(separator: "\n").suffix(40) { print("  \(line)") }
    }
}
