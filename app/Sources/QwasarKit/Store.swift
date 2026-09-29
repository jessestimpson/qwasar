// Store.swift -- persistence, PLAN.md 4.2.
//
// Plain files under Application Support, which under App Sandbox means the
// app's container. Two deliberate choices:
//
//   transcript.jsonl is APPEND-ONLY. A crash during generation should lose the
//   current turn, not the conversation -- the same reasoning as the session
//   itself being append-only, one layer up.
//
//   tokens.bin is raw little-endian Int32 rather than JSON, because it is the
//   one field that is large and is never read by a human.

import Foundation

public final class Store: @unchecked Sendable {
    public let root: URL

    public init() throws {
        let base = FileManager.default.urls(for: .applicationSupportDirectory,
                                            in: .userDomainMask)[0]
        root = base.appendingPathComponent("Crucible", isDirectory: true)
        try FileManager.default.createDirectory(at: root.appendingPathComponent("sessions"),
                                                withIntermediateDirectories: true)
    }

    private var projectsURL: URL { root.appendingPathComponent("projects.json") }
    private func dir(_ id: UUID) -> URL {
        root.appendingPathComponent("sessions/\(id.uuidString)", isDirectory: true)
    }

    // MARK: Projects

    public func loadProjects() -> [Project] {
        guard let d = try? Data(contentsOf: projectsURL) else { return [] }
        return (try? JSONDecoder().decode([Project].self, from: d)) ?? []
    }

    public func saveProjects(_ p: [Project]) {
        let e = JSONEncoder()
        e.outputFormatting = [.prettyPrinted, .sortedKeys]
        try? e.encode(p).write(to: projectsURL, options: .atomic)
    }

    // MARK: Global sandbox configuration (PLAN.md 8.5)

    private var globalOverlayURL: URL { root.appendingPathComponent("sandbox.json") }

    public func loadGlobalOverlay() -> SandboxOverlay? {
        guard let d = try? Data(contentsOf: globalOverlayURL) else { return nil }
        return try? JSONDecoder().decode(SandboxOverlay.self, from: d)
    }

    public func saveGlobalOverlay(_ o: SandboxOverlay?) {
        guard let o, !o.isEmpty else {
            try? FileManager.default.removeItem(at: globalOverlayURL)
            return
        }
        let e = JSONEncoder()
        e.outputFormatting = [.prettyPrinted, .sortedKeys]
        try? e.encode(o).write(to: globalOverlayURL, options: .atomic)
    }

    // MARK: Sessions

    public func loadSessions() -> [SessionRecord] {
        let sessionsDir = root.appendingPathComponent("sessions")
        let entries = (try? FileManager.default.contentsOfDirectory(at: sessionsDir,
                                                                    includingPropertiesForKeys: nil)) ?? []
        var out: [SessionRecord] = []
        for e in entries {
            let f = e.appendingPathComponent("record.json")
            guard let d = try? Data(contentsOf: f),
                  let r = try? JSONDecoder().decode(SessionRecord.self, from: d) else { continue }
            out.append(r)
        }
        return out.sorted { $0.createdAt > $1.createdAt }
    }

    public func save(_ r: SessionRecord) {
        try? FileManager.default.createDirectory(at: dir(r.id), withIntermediateDirectories: true)
        let e = JSONEncoder()
        e.outputFormatting = [.prettyPrinted, .sortedKeys]
        try? e.encode(r).write(to: dir(r.id).appendingPathComponent("record.json"),
                               options: .atomic)
    }

    public func delete(_ id: UUID) {
        try? FileManager.default.removeItem(at: dir(id))
    }

    // MARK: Transcript

    public func loadTranscript(_ id: UUID) -> [TranscriptItem] {
        let f = dir(id).appendingPathComponent("transcript.jsonl")
        guard let text = try? String(contentsOf: f, encoding: .utf8) else { return [] }
        let dec = JSONDecoder()
        return text.split(separator: "\n").compactMap { line in
            guard let d = line.data(using: .utf8) else { return nil }
            return try? dec.decode(TranscriptItem.self, from: d)
        }
    }

    /// Appends completed items. Called once per turn, not once per token: a
    /// partial turn is exactly what a crash is allowed to lose.
    public func appendTranscript(_ id: UUID, _ items: [TranscriptItem]) {
        guard !items.isEmpty else { return }
        try? FileManager.default.createDirectory(at: dir(id), withIntermediateDirectories: true)
        let f = dir(id).appendingPathComponent("transcript.jsonl")
        let enc = JSONEncoder()
        var blob = Data()
        for i in items {
            guard let d = try? enc.encode(i) else { continue }
            blob.append(d)
            blob.append(0x0A)
        }
        if let h = try? FileHandle(forWritingTo: f) {
            defer { try? h.close() }
            _ = try? h.seekToEnd()
            try? h.write(contentsOf: blob)
        } else {
            try? blob.write(to: f, options: .atomic)
        }
    }

    public func rewriteTranscript(_ id: UUID, _ items: [TranscriptItem]) {
        let f = dir(id).appendingPathComponent("transcript.jsonl")
        try? FileManager.default.removeItem(at: f)
        appendTranscript(id, items)
    }

    // MARK: Tokens

    public func loadTokens(_ id: UUID) -> [Int32] {
        let f = dir(id).appendingPathComponent("tokens.bin")
        guard let d = try? Data(contentsOf: f) else { return [] }
        return d.withUnsafeBytes { raw in
            Array(raw.bindMemory(to: Int32.self))
        }
    }

    public func saveTokens(_ id: UUID, _ tokens: [Int32]) {
        try? FileManager.default.createDirectory(at: dir(id), withIntermediateDirectories: true)
        let d = tokens.withUnsafeBufferPointer { Data(buffer: $0) }
        try? d.write(to: dir(id).appendingPathComponent("tokens.bin"), options: .atomic)
    }

    public func diskBytes(_ id: UUID) -> UInt64 {
        let e = FileManager.default.enumerator(at: dir(id), includingPropertiesForKeys: [.fileSizeKey])
        var total: UInt64 = 0
        while let u = e?.nextObject() as? URL {
            total += UInt64((try? u.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0)
        }
        return total
    }
}
