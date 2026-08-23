// GitImport.swift -- the git crossing, host half (spec 7.4a).
//
// The guest exports objects; this writes them into the user's own repository
// as LOOSE OBJECTS plus one ref, and nothing else -- never HEAD, never the
// index, never a working file, never a branch the user owns. The worst
// outcome of a bug is unreferenced objects that `git gc` collects; the
// destructive step, changing files, stays in the user's own `git merge`.
//
// No libgit2 and no child process (security-scoped access does not survive
// into one). A loose object is `<type> <len>\0<content>`, zlib-deflated,
// stored at .git/objects/<sha[0:2]>/<sha[2:]> where the sha names the
// UNCOMPRESSED form -- so the hash is the verification, and it is free: a
// wrong type, a wrong length, or a truncated transfer lands as a refusal,
// never as a corrupt repository.
//
// zlib framing by hand: Apple's Compression gives raw DEFLATE, so this adds
// the two-byte header and the adler32 of the uncompressed stream, which is
// all the zlib container is.

import Foundation
import Compression
import CryptoKit

public enum GitImportError: Error, CustomStringConvertible {
    case hashMismatch(claimed: String, actual: String)
    case badType(String)
    case notARepository(String)
    case writeFailed(String)

    public var description: String {
        switch self {
        case .hashMismatch(let c, let a): return "object \(c) hashed to \(a); refused"
        case .badType(let t): return "unknown object type \(t)"
        case .notARepository(let p): return "\(p) is not a git repository"
        case .writeFailed(let m): return m
        }
    }
}

public struct GitImport {
    let gitDir: URL

    public init(repoRoot: URL) throws {
        let g = repoRoot.appendingPathComponent(".git")
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: g.path, isDirectory: &isDir),
              isDir.boolValue else {
            throw GitImportError.notARepository(repoRoot.path)
        }
        self.gitDir = g
    }

    static let types: Set<String> = ["blob", "tree", "commit", "tag"]

    /// Verifies that `content` really is the object `sha` claims, then writes
    /// it loose. Idempotent: an object already present is left alone (its
    /// name is its content, so there is nothing to update).
    public func write(sha: String, type: String, content: Data) throws {
        guard Self.types.contains(type) else { throw GitImportError.badType(type) }
        var raw = Data("\(type) \(content.count)\u{0}".utf8)
        raw.append(content)
        let actual = Insecure.SHA1.hash(data: raw).map { String(format: "%02x", $0) }.joined()
        guard actual == sha.lowercased() else {
            throw GitImportError.hashMismatch(claimed: sha, actual: actual)
        }

        let dir = gitDir.appendingPathComponent("objects/\(actual.prefix(2))")
        let dest = dir.appendingPathComponent(String(actual.dropFirst(2)))
        if FileManager.default.fileExists(atPath: dest.path) { return }
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        let tmp = dir.appendingPathComponent(".tmp-\(UUID().uuidString)")
        try Self.zlib(raw).write(to: tmp, options: [])
        do {
            try FileManager.default.moveItem(at: tmp, to: dest)
        } catch {
            // A concurrent writer beat us to an identical file; that is fine.
            try? FileManager.default.removeItem(at: tmp)
            if !FileManager.default.fileExists(atPath: dest.path) {
                throw GitImportError.writeFailed("\(dest.path): \(error)")
            }
        }
    }

    /// Points `refs/heads/<branch>` at `sha`. A loose ref overrides
    /// packed-refs, so this needs no packed-refs handling.
    public func updateRef(branch: String, to sha: String) throws {
        let ref = gitDir.appendingPathComponent("refs/heads/\(branch)")
        try FileManager.default.createDirectory(at: ref.deletingLastPathComponent(),
                                                withIntermediateDirectories: true)
        try Data((sha + "\n").utf8).write(to: ref, options: .atomic)
    }

    /// First message line and author of a commit object, for the summary the
    /// crossing sheet shows -- parsed here because the host never spawns git.
    public static func commitSummary(_ content: Data) -> (message: String, author: String)? {
        guard let text = String(data: content, encoding: .utf8),
              let split = text.range(of: "\n\n") else { return nil }
        let message = text[split.upperBound...]
            .split(separator: "\n", maxSplits: 1).first.map(String.init) ?? ""
        var author = "?"
        for line in text[..<split.lowerBound].split(separator: "\n") {
            if line.hasPrefix("author ") {
                let a = line.dropFirst(7)
                author = a.range(of: " <").map { String(a[..<$0.lowerBound]) } ?? String(a)
            }
        }
        return (message, author)
    }

    // MARK: zlib

    /// zlib container: 0x78 0x01 header, raw DEFLATE, adler32 (big-endian) of
    /// the uncompressed input. `git fsck --strict` accepts this framing.
    static func zlib(_ data: Data) -> Data {
        var out = Data([0x78, 0x01])
        out.append(deflate(data))
        var adler = adler32(data).bigEndian
        withUnsafeBytes(of: &adler) { out.append(contentsOf: $0) }
        return out
    }

    static func deflate(_ data: Data) -> Data {
        // An empty input is a legal object (the empty blob is famous); the
        // encoder refuses it, so its two-byte fixed-Huffman form is spelled.
        guard !data.isEmpty else { return Data([0x03, 0x00]) }
        let cap = data.count + data.count / 2 + 256
        var dst = Data(count: cap)
        let n = dst.withUnsafeMutableBytes { d in
            data.withUnsafeBytes { s in
                compression_encode_buffer(d.bindMemory(to: UInt8.self).baseAddress!, cap,
                                          s.bindMemory(to: UInt8.self).baseAddress!, data.count,
                                          nil, COMPRESSION_ZLIB)
            }
        }
        if n > 0 { return dst.prefix(n) }
        // Incompressible and over the scratch estimate: store it raw in
        // 64 KB stored-blocks, which is always valid DEFLATE.
        var out = Data()
        var i = 0
        while i < data.count {
            let chunk = data[i..<min(i + 65535, data.count)]
            let final: UInt8 = chunk.endIndex == data.count ? 1 : 0
            out.append(final)
            var len = UInt16(chunk.count).littleEndian
            withUnsafeBytes(of: &len) { out.append(contentsOf: $0) }
            var nlen = (~UInt16(chunk.count)).littleEndian
            withUnsafeBytes(of: &nlen) { out.append(contentsOf: $0) }
            out.append(chunk)
            i = chunk.endIndex
        }
        return out
    }

    static func adler32(_ data: Data) -> UInt32 {
        var a: UInt32 = 1, b: UInt32 = 0
        for byte in data {
            a = (a &+ UInt32(byte)) % 65521
            b = (b &+ a) % 65521
        }
        return (b << 16) | a
    }
}
