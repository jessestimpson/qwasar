// ModelFamily.swift -- which of the engine's two models a folder holds, and
// what each one costs.
//
// The engine is a single-FAMILY engine (PLAN-flash-next.md §0): Qwen3.8 27B
// (`qwen3_5`, dense) and Qwen3.8 Flash-Next (`qwen4_exp`, a 125B MoE with 6B
// active). They share the tokenizer, the chat template and the session model,
// so everything above the engine -- transcripts, token histories, the agent
// loop -- carries over unchanged. What does not carry over is the arithmetic:
// weights, KV per token, fixed per-session state, and how fast a turn goes.
// That is what this file holds, so MemoryProfile can size a session for the
// model actually chosen rather than for the 27B it was written around.
//
// Recognised from config.json the way the engine and the menu bar recognise
// them, not by folder name -- a download is called whatever it was called.
// Everything here reads JSON and safetensors HEADERS only; nothing is mapped.

import Foundation

public enum ModelFamily: String, Sendable, CaseIterable, Codable {
    case dense        // Qwen3.8 27B: model_type qwen3_5
    case flashNext    // Qwen3.8 Flash-Next: model_type qwen4_exp

    /// The display name, matching `qwasar_model_name`.
    public var title: String {
        switch self {
        case .dense: return "Qwen3.8 27B"
        case .flashNext: return "Qwen3.8 Flash-Next"
        }
    }

    /// The engine's id for it, matching `qwasar_model_id`.
    public var id: String {
        switch self {
        case .dense: return "qwen3.8-27b"
        case .flashNext: return "qwen3.8-flash-next"
        }
    }

    public init?(modelID: String) {
        guard let f = ModelFamily.allCases.first(where: { $0.id == modelID }) else { return nil }
        self = f
    }

    /// KV and every other cache sized by context, per token.
    ///
    /// Dense: 16 full-attention layers x 4 KV heads x 256 x (K+V) x fp16.
    /// Flash-Next: 12 QSA layers x 2 KV heads x 256 x (K+V) x fp16 = 24 KB,
    /// plus the indexer's raw key cache (12 x 128 x fp32 = 6 KB), its block
    /// scores and its attention mask (1 KB each, being one prefill chunk of
    /// rows wide) -- 32 KB, half the dense model's.
    public var kvBytesPerToken: UInt64 {
        switch self {
        case .dense: return 64 * 1024
        case .flashNext: return 32 * 1024
        }
    }

    /// Per-session memory that does not grow with context.
    ///
    /// Dense: 151 MB of SSM/conv state plus ~200 MB of activation scratch.
    /// Flash-Next: 113 MB of delta state, and scratch for a 1024-row prefill
    /// chunk (the engine's default for this family) through four residual
    /// streams, the MoE and the engram -- ~1 GB, from the allocation sizes in
    /// qwasar_graph.c and qwasar_flash_graph.c.
    public var sessionFixedBytes: UInt64 {
        switch self {
        case .dense: return 351 * 1024 * 1024
        case .flashNext: return 1_100_000_000
        }
    }

    /// Resident weights, when the headers cannot be read to say exactly.
    /// Dense: 15.1 GB of 4-bit text weights plus a 0.92 GB vision tower.
    /// Flash-Next: 79.5 GB (74 GiB) from mlx-community's headers; the engram
    /// table stays on disk.
    public var fallbackWeightsBytes: UInt64 {
        switch self {
        case .dense: return 16_020_000_000
        case .flashNext: return 79_520_000_000
        }
    }

    /// Whether the engine can speculate with a draft head for this model.
    /// The 27B's head is a separate download; Flash-Next's MTP layer is in its
    /// own shards but not wired yet (PLAN-flash-next.md, Phase 7), and the
    /// engine refuses a head for it outright.
    public var supportsDraftHead: Bool { self == .dense }

    /// The per-turn generation budget.
    ///
    /// Dense: 4096, because at ~6 tok/s that is already eleven minutes, and a
    /// budget is how a reasoning loop that is going nowhere gets stopped.
    /// Flash-Next decodes ~68 tok/s on an M5 Max, so the same wall-clock bound
    /// is ten times the tokens; 32K lets it reason at length and still stops a
    /// runaway in about eight minutes.
    public var turnBudget: Int {
        switch self {
        case .dense: return 4096
        case .flashNext: return 32_768
        }
    }

    /// One line on what speed to expect, for the rate meter's help.
    public var speedNote: String {
        switch self {
        case .dense:
            return "This is a dense 27B model; about 6 tok/s is the serial bandwidth "
                 + "ceiling on a 32 GB M4 — higher means speculation is paying."
        case .flashNext:
            return "This is Flash-Next, a mixture of experts with 6B active; about "
                 + "68 tok/s on an M5 Max, a little less past 2K tokens of context."
        }
    }
}

/// What a model folder holds, read from its config and shard headers.
public struct ModelInspection: Sendable, Equatable {
    public var family: ModelFamily
    /// max_position_embeddings, from text_config where the config nests one.
    public var maxContext: Int32
    /// Bytes the GPU will hold: every tensor but the host-only engram rows.
    /// nil when the headers could not be read.
    public var weightsBytes: UInt64?

    public var residentWeightsBytes: UInt64 { weightsBytes ?? family.fallbackWeightsBytes }

    /// Why a folder is not a model the engine can load, or nil if it is.
    public enum Refusal: Error, CustomStringConvertible, Equatable {
        case noConfig
        case unsupportedType(String)
        case unsupportedQuantisation

        public var description: String {
            switch self {
            case .noConfig:
                return "no readable config.json"
            case .unsupportedType(let t):
                return "model_type \"\(t)\" is neither Qwen3.8 27B (qwen3_5) nor "
                     + "Qwen3.8 Flash-Next (qwen4_exp)"
            case .unsupportedQuantisation:
                return "the engine runs 4-bit weights in groups of 32 or 64; this is "
                     + "some other quantisation (an FP8 or BF16 download?)"
            }
        }
    }

    /// Reads `dir/config.json` and the shard headers. Throws a Refusal for
    /// anything the engine would refuse, so a wrong pick fails in a
    /// millisecond with a reason rather than seconds into binding.
    public static func inspect(_ dir: URL) throws -> ModelInspection {
        guard let data = try? Data(contentsOf: dir.appendingPathComponent("config.json")),
              let json = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
        else { throw Refusal.noConfig }

        let type = json["model_type"] as? String ?? ""
        let family: ModelFamily
        switch type {
        case "qwen3_5": family = .dense
        case "qwen4_exp": family = .flashNext
        case "qwen4_exp_text" where json["text_config"] == nil: family = .flashNext
        default: throw Refusal.unsupportedType(type)
        }

        let q = (json["quantization"] ?? json["quantization_config"]) as? [String: Any]
        guard let bits = q?["bits"] as? Int, bits == 4,
              let group = q?["group_size"] as? Int, group == 32 || group == 64
        else { throw Refusal.unsupportedQuantisation }

        let text = json["text_config"] as? [String: Any]
        let n = (text?["max_position_embeddings"] ?? json["max_position_embeddings"]) as? Int
        let maxContext = Int32(clamping: (n ?? 0) > 0 ? n! : 262_144)

        return ModelInspection(family: family, maxContext: maxContext,
                               weightsBytes: deviceBytes(in: dir))
    }

    /// Sum of tensor sizes across the folder's shards, less the host-only
    /// ones -- the engine's own rule (qw_shard_device_bytes): a file whose
    /// metadata says `placement: cpu` is skipped whole, and any tensor named
    /// `*.ngram_embedding.*` is skipped wherever it sits, since MLX's build
    /// mixes the engram shards into ordinary model files.
    static func deviceBytes(in dir: URL) -> UInt64? {
        let fm = FileManager.default
        guard let names = try? fm.contentsOfDirectory(atPath: dir.path) else { return nil }
        let shards = names.filter { $0.hasSuffix(".safetensors") }
        guard !shards.isEmpty else { return nil }
        var total: UInt64 = 0
        for name in shards {
            guard let n = shardDeviceBytes(dir.appendingPathComponent(name)) else { return nil }
            total += n
        }
        return total
    }

    static func shardDeviceBytes(_ url: URL) -> UInt64? {
        guard let h = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? h.close() }
        guard let lenData = try? h.read(upToCount: 8), lenData.count == 8 else { return nil }
        let len = lenData.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }.littleEndian
        guard len > 0, len < 512 * 1024 * 1024,
              let header = try? h.read(upToCount: Int(len)), header.count == Int(len),
              let json = (try? JSONSerialization.jsonObject(with: header)) as? [String: Any]
        else { return nil }

        if let meta = json["__metadata__"] as? [String: Any],
           meta["placement"] as? String == "cpu" { return 0 }

        var bytes: UInt64 = 0
        for (name, v) in json where name != "__metadata__" {
            guard !name.contains(".ngram_embedding."),
                  let t = v as? [String: Any],
                  let offs = t["data_offsets"] as? [NSNumber], offs.count == 2 else { continue }
            let a = offs[0].uint64Value, b = offs[1].uint64Value
            if b > a { bytes += b - a }
        }
        return bytes
    }
}
