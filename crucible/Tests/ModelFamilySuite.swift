// ModelFamilySuite.swift -- telling the two models apart, and sizing each.
//
// Pure: a synthetic model folder in a temp directory (a config.json and a
// safetensors file that is all header), and the profile arithmetic with the
// machine passed in. No tokenizer, no engine -- so the 128 GB numbers are
// checkable on a 32 GB laptop.

import Foundation
import CrucibleKit

enum ModelFamilySuite {
    static func run() -> Int {
        var f = 0
        let fm = FileManager.default
        let dir = fm.temporaryDirectory.appendingPathComponent("crucible-mf-\(UUID())")
        try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: dir) }

        func config(_ json: String) {
            try? json.write(to: dir.appendingPathComponent("config.json"),
                            atomically: true, encoding: .utf8)
        }
        /// A shard with the given tensors (name -> byte length) and no data:
        /// only the header is ever read.
        func shard(_ name: String, _ tensors: [(String, Int)], placement: String? = nil) {
            var entries: [String] = []
            if let placement { entries.append("\"__metadata__\":{\"placement\":\"\(placement)\"}") }
            var off = 0
            for (n, len) in tensors {
                entries.append("\"\(n)\":{\"dtype\":\"U32\",\"shape\":[\(len / 4)],"
                             + "\"data_offsets\":[\(off),\(off + len)]}")
                off += len
            }
            let header = Data(("{" + entries.joined(separator: ",") + "}").utf8)
            var len = UInt64(header.count).littleEndian
            var d = Data(bytes: &len, count: 8)
            d.append(header)
            try? d.write(to: dir.appendingPathComponent(name))
        }

        // --- Flash-Next, MLX's layout: engram rows mixed into model files ---
        config("""
        {"model_type":"qwen4_exp","quantization":{"bits":4,"group_size":32},
         "text_config":{"max_position_embeddings":262144}}
        """)
        shard("model-00001-of-00002.safetensors", [
            ("language_model.model.layers.0.mlp.gate.weight", 1000),
            ("language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.0.weight", 50_000),
            ("language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.0.scales", 7_000),
        ])
        shard("model-00002-of-00002.safetensors", [("lm_head.weight", 2000)])
        // Our converter's layout: a whole file placed on the host.
        shard("engram-00001.safetensors", [("x.weight", 90_000)], placement: "cpu")

        do {
            let m = try ModelInspection.inspect(dir)
            f += TestMain.check(m.family == .flashNext, "qwen4_exp is Flash-Next")
            f += TestMain.check(m.maxContext == 262_144, "context from text_config")
            f += TestMain.check(m.weightsBytes == 3000,
                                "engram rows and cpu-placed files are not resident "
                                + "(\(m.weightsBytes.map(String.init) ?? "nil")/3000)")
        } catch {
            f += TestMain.check(false, "Flash-Next inspects (\(error))")
        }

        // --- the 27B, flat config ---
        try? fm.removeItem(at: dir.appendingPathComponent("engram-00001.safetensors"))
        config("""
        {"model_type":"qwen3_5","max_position_embeddings":131072,
         "quantization":{"bits":4,"group_size":64}}
        """)
        if let m = try? ModelInspection.inspect(dir) {
            f += TestMain.check(m.family == .dense, "qwen3_5 is the 27B")
            f += TestMain.check(m.maxContext == 131_072, "context from the top level")
        } else {
            f += TestMain.check(false, "the 27B inspects")
        }

        // --- refusals, before any binding ---
        config(#"{"model_type":"llama","quantization":{"bits":4,"group_size":64}}"#)
        f += TestMain.check(refusal(dir) == .unsupportedType("llama"), "another model_type is refused")
        config(#"{"model_type":"qwen3_5","quantization":{"bits":8,"group_size":64}}"#)
        f += TestMain.check(refusal(dir) == .unsupportedQuantisation, "8-bit weights are refused")
        config(#"{"model_type":"qwen4_exp"}"#)
        f += TestMain.check(refusal(dir) == .unsupportedQuantisation, "BF16 (no quantization) is refused")
        try? fm.removeItem(at: dir.appendingPathComponent("config.json"))
        f += TestMain.check(refusal(dir) == .noConfig, "no config.json is refused")

        // --- the profile, per model, per machine ---
        // The M4 Air's measured working set; the Max's is an assumption of
        // the same ~84% of 128 GB, which is what the fallback uses too.
        let air: (ws: UInt64, phys: UInt64) = (26_800_603_136, 32 << 30)
        let max: (ws: UInt64, phys: UInt64) = (115_448_000_000, 128 << 30)
        func p(_ fam: ModelFamily, _ m: (ws: UInt64, phys: UInt64), mtp: Bool = false) -> MemoryProfile {
            MemoryProfile.derive(family: fam, weightsBytes: fam.fallbackWeightsBytes,
                                 maxContext: 262_144, mtpAvailable: mtp,
                                 workingSetBytes: m.ws, physicalBytes: m.phys)
        }

        let d = p(.dense, air)
        f += TestMain.check(d.contextSize == 90_112 && d.liveSessions == 1,
                            "27B on the Air: 90112 ctx, 1 live, as before (\(d.contextSize), \(d.liveSessions))")
        let dm = p(.dense, air, mtp: true)
        f += TestMain.check(dm.mtpEnabled && dm.contextSize == 73_728,
                            "27B on the Air with a head: 73728 ctx, as before (\(dm.contextSize))")
        let fa = p(.flashNext, air)
        f += TestMain.check(!fa.note.isEmpty, "Flash-Next on the Air says it does not fit")
        let fx = p(.flashNext, max, mtp: true)
        f += TestMain.check(fx.contextSize == 262_144, "Flash-Next on the Max: full 262K (\(fx.contextSize))")
        f += TestMain.check(!fx.mtpEnabled, "Flash-Next never enables the 27B's draft head")
        f += TestMain.check(fx.note.isEmpty, "Flash-Next on the Max fits")
        let dx = p(.dense, max)
        f += TestMain.check(dx.contextSize == 262_144 && dx.liveSessions == 4,
                            "27B on the Max: full 262K, 4 live (\(dx.contextSize), \(dx.liveSessions))")
        let short = MemoryProfile.derive(family: .dense, weightsBytes: ModelFamily.dense.fallbackWeightsBytes,
                                         maxContext: 32_768, mtpAvailable: false,
                                         workingSetBytes: max.ws, physicalBytes: max.phys)
        f += TestMain.check(short.contextSize == 32_768, "never more context than the config trained for")

        // --- ids round-trip with the engine's spelling ---
        f += TestMain.check(ModelFamily(modelID: "qwen3.8-flash-next") == .flashNext
                            && ModelFamily(modelID: "qwen3.8-27b") == .dense,
                            "engine ids map to families")
        return f
    }

    private static func refusal(_ dir: URL) -> ModelInspection.Refusal? {
        do { _ = try ModelInspection.inspect(dir); return nil }
        catch let r as ModelInspection.Refusal { return r }
        catch { return nil }
    }
}
