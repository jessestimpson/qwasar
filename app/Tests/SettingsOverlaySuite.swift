// SettingsOverlaySuite.swift -- the overlay rule, pinned.
//
// PLAN.md 8.5: field-wise resolution, most specific non-nil wins, replace not
// merge. The distinction that carries the most weight is nil-versus-empty on
// the delegate list -- silence falls through, an empty list is an explicit
// OFF that beats a global grant -- so that is tested by name.

import Foundation
import QwasarKit

enum SettingsOverlaySuite {
    static func run() -> Int {
        var f = 0

        let global = SettingsOverlay(agentModels: ["remote/a"], agentBudgetUSD: 9)
        let project = SettingsOverlay(agentTurnBudgetUSD: 2)
        let session = SettingsOverlay(toolTimeoutSeconds: 60)

        let r = SessionSettings.resolve(global: global, project: project, session: session)
        f += TestMain.check(r.agentModels == ["remote/a"], "global fills what nothing above set")
        f += TestMain.check(r.agentBudgetUSD == 9, "a global budget survives silent upper layers")
        f += TestMain.check(r.agentTurnBudgetUSD == 2, "the project's turn budget overrides the default")
        f += TestMain.check(r.toolTimeoutSeconds == 60, "session timeout wins")
        let bare = SessionSettings.resolve(global: nil, project: nil, session: nil)
        f += TestMain.check(bare == SessionSettings.defaults, "an unset field everywhere is the built-in default")

        // Replace, not merge -- and empty is an opinion where nil is silence.
        let sOff = SettingsOverlay(agentModels: [])
        let off = SessionSettings.resolve(global: global, project: nil, session: sOff)
        f += TestMain.check(off.agentModels.isEmpty,
                            "a session's empty list turns delegation OFF over a global grant")
        let pList = SettingsOverlay(agentModels: ["remote/b"])
        let repl = SessionSettings.resolve(global: global, project: pList, session: nil)
        f += TestMain.check(repl.agentModels == ["remote/b"],
                            "a project list replaces the global list, never merges")

        // Provenance names the deciding layer.
        f += TestMain.check(SessionSettings.provenance(of: .agentModels, global: global,
                                                       project: nil, session: sOff) == .session,
                            "provenance: the explicit OFF is the session's")
        f += TestMain.check(SessionSettings.provenance(of: .agentBudgetUSD, global: global,
                                                       project: project, session: session) == .global,
                            "provenance: the budget came from global")
        f += TestMain.check(SessionSettings.provenance(of: .toolTimeoutSeconds, global: global,
                                                       project: project, session: nil) == .builtin,
                            "provenance: the untouched field is built-in")

        // The keys parse, refuse out-of-range, and clear back to silence.
        var o = SettingsOverlay()
        f += TestMain.check(SettingsKey.toolTimeoutSeconds.set("300", on: &o) == nil && o.toolTimeoutSeconds == 300,
                            "tool_timeout_seconds parses")
        f += TestMain.check(SettingsKey.toolTimeoutSeconds.set("5", on: &o) != nil && o.toolTimeoutSeconds == 300,
                            "out-of-range is refused and changes nothing")
        f += TestMain.check(SettingsKey.agentModels.set(" a/x , b/y ", on: &o) == nil
                            && o.agentModels == ["a/x", "b/y"],
                            "the model list splits and trims")
        f += TestMain.check(SettingsKey.agentModels.set("", on: &o) == nil && o.agentModels == [],
                            "empty string sets the explicit OFF")
        f += TestMain.check(SettingsKey.topNSigma.set("1", on: &o) == nil && o.topNSigma == 1,
                            "top_n_sigma parses")
        f += TestMain.check(SettingsKey.topNSigma.set("-1", on: &o) != nil && o.topNSigma == 1,
                            "a negative top_n_sigma is refused and changes nothing")
        let sigmaOff = SessionSettings.resolve(global: SettingsOverlay(topNSigma: 1), project: nil,
                                               session: SettingsOverlay(topNSigma: 0))
        f += TestMain.check(sigmaOff.topNSigma == 0, "a session's 0 turns top_n_sigma off over a global 1")
        f += TestMain.check(SessionSettings.defaults.topNSigma == 0,
                            "top_n_sigma is off unless set: the model's own sampling")
        f += TestMain.check(QwasarClient.sampling(temperature: nil, topNSigma: 0) == nil,
                            "nothing set sends no sampling object")
        f += TestMain.check(QwasarClient.sampling(temperature: nil, topNSigma: 1.5)?["top_n_sigma"] as? Double == 1.5
                            && QwasarClient.sampling(temperature: nil, topNSigma: 1.5)?["temperature"] == nil,
                            "top_n_sigma alone sends only top_n_sigma")
        f += TestMain.check(SettingsKey.exploreNudgeAfter.set("6", on: &o) == nil && o.exploreNudgeAfter == 6,
                            "explore_nudge_after parses")
        f += TestMain.check(SessionSettings.defaults.toolApproval == .ask, "tool_approval defaults to ask")
        f += TestMain.check(SettingsKey.toolApproval.set("Trusted", on: &o) == nil && o.toolApproval == .trusted
                            && SettingsKey.toolApproval.set("yolo", on: &o) != nil && o.toolApproval == .trusted,
                            "tool_approval parses ask/trusted and refuses anything else")
        f += TestMain.check(SessionSettings.resolve(global: SettingsOverlay(toolApproval: .trusted), project: nil,
                                                    session: SettingsOverlay(toolApproval: .ask)).toolApproval == .ask,
                            "a session's ask overrides a global trusted")
        SettingsKey.toolApproval.clear(on: &o)
        f += TestMain.check(SessionSettings.defaults.exploreNudgeAfter == 10,
                            "the exploration nudge is on by default, at 10")

        // The nudge: reads and searches count, a change ends the streak, Bash
        // is neither, and a round crossing the mark gets one reminder.
        var n = ExplorationNudge(after: 3)
        n.record("Read"); n.record("Grep")
        f += TestMain.check(n.reminder() == nil, "no reminder below the mark")
        n.record("Bash"); n.record("TodoWrite")
        f += TestMain.check(n.streak == 2 && n.reminder() == nil, "Bash and TodoWrite neither count nor reset")
        n.record("Glob"); n.record("Read"); n.record("Read")
        let first = n.reminder()
        f += TestMain.check(first?.contains("5 reads and searches this turn, and no edit yet") == true,
                            "the reminder names the streak before any edit")
        f += TestMain.check(n.reminder() == nil, "one reminder per crossing, not per call")
        n.record("Read"); n.record("Read")
        f += TestMain.check(n.reminder() == nil, "the next is `after` more reads away")
        n.record("Read")
        f += TestMain.check(n.reminder() != nil, "and then it comes")
        n.record("Edit")
        f += TestMain.check(n.streak == 0 && n.reminder() == nil, "an edit ends the streak")
        n.record("Read"); n.record("Read"); n.record("Grep")
        f += TestMain.check(n.reminder()?.contains("3 reads and searches since your last edit") == true,
                            "after an edit, the count restarts and the wording says so")
        var quiet = ExplorationNudge(after: 0)
        for _ in 0..<50 { quiet.record("Read") }
        f += TestMain.check(quiet.reminder() == nil, "0 is off")

        SettingsKey.toolTimeoutSeconds.clear(on: &o)
        SettingsKey.agentModels.clear(on: &o)
        SettingsKey.topNSigma.clear(on: &o)
        f += TestMain.check(SettingsKey.thinkingBudget.set("4096", on: &o) == nil && o.thinkingBudget == 4096,
                            "thinking_budget parses")
        f += TestMain.check(SettingsKey.thinkingBudget.set("12", on: &o) != nil && o.thinkingBudget == 4096,
                            "a thinking_budget too small to think in is refused")
        let budgeted = QwasarClient.sampling(temperature: nil, topNSigma: 0, thinkingBudget: 2048)
        f += TestMain.check(budgeted?["thinking_budget"] as? Int == 2048,
                            "a thinking budget rides in the step's sampling")
        SettingsKey.exploreNudgeAfter.clear(on: &o)
        SettingsKey.thinkingBudget.clear(on: &o)
        f += TestMain.check(o.isEmpty, "clearing every set key leaves an empty layer")

        // Records from when sessions could run in a VM still decode: the
        // sandbox's own keys are ignored, and the session is marked as such.
        let oldProject = #"{"id":"6F1D2C3B-0000-4000-8000-000000000001","name":"p","rootBookmark":"","systemPrompt":"","#
            + #""networkAllowlist":["old.dev"],"skillLibrary":[],"#
            + #""sandbox":{"networkAllowlist":["a.dev"],"guestCPUs":4,"toolTimeoutSeconds":90}}"#
        let p = try? JSONDecoder().decode(Project.self, from: Data(oldProject.utf8))
        f += TestMain.check(p?.overlay.toolTimeoutSeconds == 90,
                            "an old project decodes, its surviving keys kept")
        let oldSession = #"{"id":"6F1D2C3B-0000-4000-8000-000000000002","projectID":"6F1D2C3B-0000-4000-8000-000000000001","#
            + #""title":"t","workingSubpath":"","createdAt":0,"state":"closed","contextSize":1,"tokenCount":0,"#
            + #""tools":"sandbox"}"#
        let s = try? JSONDecoder().decode(SessionRecord.self, from: Data(oldSession.utf8))
        f += TestMain.check(s?.ranInSandbox == true, "an old sandboxed session decodes, marked as one")

        // The config project is what it claims.
        f += TestMain.check(Project.configProject().isConfig, "the config project knows itself")

        return f
    }
}
