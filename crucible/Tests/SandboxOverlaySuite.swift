// SandboxOverlaySuite.swift -- the overlay rule, pinned.
//
// PLAN.md 8.5: field-wise resolution, most specific non-nil wins, replace not
// merge. The distinction that carries the most weight is nil-versus-empty on
// the network list -- silence falls through, an empty list is an explicit OFF
// that beats a global grant -- so that is tested by name.

import Foundation
import CrucibleKit

enum SandboxOverlaySuite {
    static func run() -> Int {
        var f = 0

        let global = SandboxOverlay(networkAllowlist: ["hexdocs.pm"], guestMemoryMB: 4096)
        let project = SandboxOverlay(guestCPUs: 4)
        let session = SandboxOverlay(toolTimeoutSeconds: 60)

        let r = SandboxSettings.resolve(global: global, project: project, session: session)
        f += TestMain.check(r.networkAllowlist == ["hexdocs.pm"], "global fills what nothing above set")
        f += TestMain.check(r.guestMemoryMB == 4096, "global memory survives silent upper layers")
        f += TestMain.check(r.guestCPUs == 4, "project cpus override the default")
        f += TestMain.check(r.toolTimeoutSeconds == 60, "session timeout wins")
        f += TestMain.check(r.fetchMaxKB == 256, "an unset field everywhere is the built-in default")

        // Replace, not merge -- and empty is an opinion where nil is silence.
        let sOff = SandboxOverlay(networkAllowlist: [])
        let off = SandboxSettings.resolve(global: global, project: nil, session: sOff)
        f += TestMain.check(off.networkAllowlist.isEmpty,
                            "a session's empty list turns network OFF over a global grant")
        let pList = SandboxOverlay(networkAllowlist: ["a.dev"])
        let repl = SandboxSettings.resolve(global: global, project: pList, session: nil)
        f += TestMain.check(repl.networkAllowlist == ["a.dev"],
                            "a project list replaces the global list, never merges")

        // Provenance names the deciding layer.
        f += TestMain.check(SandboxSettings.provenance(of: .networkAllowlist, global: global,
                                                       project: nil, session: sOff) == .session,
                            "provenance: the explicit OFF is the session's")
        f += TestMain.check(SandboxSettings.provenance(of: .guestMemoryMB, global: global,
                                                       project: project, session: session) == .global,
                            "provenance: memory came from global")
        f += TestMain.check(SandboxSettings.provenance(of: .fetchMaxKB, global: global,
                                                       project: project, session: session) == .builtin,
                            "provenance: the untouched field is built-in")

        // The keys parse, refuse out-of-range, and clear back to silence.
        var o = SandboxOverlay()
        f += TestMain.check(SandboxKey.guestCPUs.set("4", on: &o) == nil && o.guestCPUs == 4,
                            "guest_cpus parses")
        f += TestMain.check(SandboxKey.guestCPUs.set("99", on: &o) != nil && o.guestCPUs == 4,
                            "out-of-range is refused and changes nothing")
        f += TestMain.check(SandboxKey.networkAllowlist.set(" A.dev , *.B.io ", on: &o) == nil
                            && o.networkAllowlist == ["a.dev", "*.b.io"],
                            "network list splits, trims and folds case")
        f += TestMain.check(SandboxKey.networkAllowlist.set("", on: &o) == nil
                            && o.networkAllowlist == [],
                            "empty string sets the explicit OFF")
        SandboxKey.guestCPUs.clear(on: &o)
        SandboxKey.networkAllowlist.clear(on: &o)
        f += TestMain.check(o.isEmpty, "clearing every set key leaves an empty layer")

        // The legacy per-project network field folds into the overlay.
        var legacy = Project(name: "x", rootBookmark: Data())
        legacy.networkAllowlist = ["old.dev"]
        f += TestMain.check(legacy.overlay.networkAllowlist == ["old.dev"],
                            "a pre-overlay network field reads as the project layer")
        legacy.sandbox = SandboxOverlay(networkAllowlist: ["new.dev"])
        f += TestMain.check(legacy.overlay.networkAllowlist == ["new.dev"],
                            "the overlay wins over the legacy field once set")

        // The skill library (spec 7.2): upsert by module, order preserved.
        var proj = Project(name: "p", rootBookmark: Data())
        proj.recordDefine(module: "Helper", skillName: nil, source: "v1")
        proj.recordDefine(module: "Counter", skillName: "count", source: "c1")
        proj.recordDefine(module: "Helper", skillName: nil, source: "v2")
        f += TestMain.check(proj.skillLibrary?.map(\.module) == ["Helper", "Counter"],
                            "a redefine keeps its place in definition order")
        f += TestMain.check(proj.skillLibrary?.first?.source == "v2",
                            "and carries the newest source")
        f += TestMain.check(proj.skillLibrary?.last?.skillName == "count",
                            "the invoke name rides along")

        // The config project is what it claims.
        f += TestMain.check(Project.configProject().isConfig, "the config project knows itself")

        return f
    }
}
