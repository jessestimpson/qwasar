// StoreSuite.swift -- persistence, including the property it exists for.
//
// PLAN.md 4.2: transcript.jsonl is append-only so that a crash during
// generation loses the current turn and not the conversation. That is a claim
// about a failure, so it is tested by simulating one: write two turns, append a
// third that is never completed, and assert the first two survive intact.

import Foundation
import CrucibleKit

enum StoreSuite {
    static func run() -> Int {
        var f = 0
        guard let store = try? Store() else {
            print("  FAIL cannot open the store"); return 1
        }

        let projectID = UUID()
        var rec = SessionRecord(projectID: projectID, title: "suite", contextSize: 90112)
        defer { store.delete(rec.id) }

        store.save(rec)
        let reloaded = store.loadSessions().first { $0.id == rec.id }
        f += TestMain.check(reloaded?.title == "suite", "session record round-trips")
        f += TestMain.check(reloaded?.contextSize == 90112, "context size is persisted")

        // Two complete turns.
        let turn1 = [TranscriptItem(.user("hello")),
                     TranscriptItem(.assistant("hi")),
                     TranscriptItem(.footer(TurnStats()))]
        let turn2 = [TranscriptItem(.user("list files")),
                     TranscriptItem(.tool(name: "list", arguments: ["path": "."],
                                          result: "a.txt\nb.txt")),
                     TranscriptItem(.assistant("two files"))]
        store.appendTranscript(rec.id, turn1)
        store.appendTranscript(rec.id, turn2)

        var items = store.loadTranscript(rec.id)
        f += TestMain.check(items.count == 6, "both turns are readable (\(items.count)/6)")

        if case .tool(let n, let a, let r) = items[4].kind {
            f += TestMain.check(n == "list" && a["path"] == "." && r == "a.txt\nb.txt",
                                "a tool card round-trips with its result")
        } else {
            f += TestMain.check(false, "a tool card round-trips with its result")
        }

        // A crash mid-turn: a truncated final line, as an interrupted write
        // leaves behind.
        let f2 = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Crucible/sessions/\(rec.id.uuidString)/transcript.jsonl")
        if let h = try? FileHandle(forWritingTo: f2) {
            _ = try? h.seekToEnd()
            try? h.write(contentsOf: Data(#"{"id":"broken","at":0,"kind":"#.utf8))
            try? h.close()
        }
        items = store.loadTranscript(rec.id)
        f += TestMain.check(items.count == 6,
                            "a truncated trailing line loses only itself (\(items.count)/6)")

        // Tokens are the truth; they must survive exactly.
        let toks: [Int32] = [1, 2, 3, 248_319, -1, 0]
        store.saveTokens(rec.id, toks)
        f += TestMain.check(store.loadTokens(rec.id) == toks, "tokens round-trip exactly")

        rec.tokenCount = toks.count
        store.save(rec)
        f += TestMain.check(store.diskBytes(rec.id) > 0, "disk usage is reportable")

        store.delete(rec.id)
        f += TestMain.check(store.loadSessions().first { $0.id == rec.id } == nil,
                            "delete removes the session")
        return f
    }
}
