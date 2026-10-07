// ModelCatalog.swift -- which model a folder holds, and where model folders are.
//
// Recognised from config.json the way the engine recognises them, not by
// folder name.  Came from the menu bar app; the window's Model menu and the
// status item's are the same menu now.

import Foundation

/// The models qwasar-server runs, told apart by their config.json the way
/// the engine tells them apart -- not by folder name, which is whatever the
/// download happened to be called.  Smallest first, the Model menu's order.
public enum ModelFamily: String, CaseIterable, Sendable, Codable {
    case nineB        // Qwen3.5 9B: model_type qwen3_5, hidden_size 4096
    case dense        // Qwen3.8 27B: model_type qwen3_5, hidden_size 5120
    case flashNext    // Qwen3.8 Flash-Next: model_type qwen4_exp

    /// The engine's id for it (`qwasar_model_id`, /v1/server's model.id).
    public var id: String {
        switch self {
        case .nineB: "qwen3.5-9b"
        case .dense: "qwen3.8-27b"
        case .flashNext: "qwen3.8-flash-next"
        }
    }

    public init?(modelID: String) {
        guard let f = ModelFamily.allCases.first(where: { $0.id == modelID }) else { return nil }
        self = f
    }

    /// One line on what speed to expect, for the rate meter's help.
    public var speedNote: String {
        switch self {
        case .nineB:
            "This is Qwen3.5 9B, a dense model; about 90 tok/s on an M5 Max, "
            + "several times less on a base-model Mac's memory bandwidth."
        case .dense:
            "This is a dense 27B model; about 6 tok/s is the serial bandwidth "
            + "ceiling on a 32 GB M4 — higher means speculation is paying."
        case .flashNext:
            "This is Flash-Next, a mixture of experts with 6B active; about "
            + "68 tok/s on an M5 Max, a little less past 2K tokens of context."
        }
    }

    public var title: String {
        switch self {
        case .nineB: "Qwen3.5 9B"
        case .dense: "Qwen3.8 27B"
        case .flashNext: "Qwen3.8 Flash-Next"
        }
    }

    /// What loading it asks of the machine, for the menu's tooltip.
    public var note: String {
        switch self {
        case .nineB: "Dense, for 16 GB Macs; ~6 GB on disk, ~8 GB in memory with a 32K "
                   + "context.  The previous Qwen generation: quick, weaker on long tasks."
        case .dense: "Dense; ~18 GB on disk, nearly all of it held in memory."
        case .flashNext: "Mixture of experts; ~104 GB on disk, ~75 GB held in memory "
                       + "(its engram table stays on disk).  The server will not load it "
                       + "without that much memory free."
        }
    }
}

public struct FoundModel: Sendable {
    public let family: ModelFamily
    public let path: String
}

/// Finds model folders the server can load.
public enum ModelCatalog {
    /// The family of the model in `path`, or nil if the engine would refuse it:
    /// an unsupported model_type, or anything but 4-bit weights in groups of
    /// 32 or 64 (an FP8 or BF16 download of the same model, say).  The 9B and
    /// the 27B share a model_type and are told apart by width, as the engine
    /// tells them apart; Qwen3.5's 4B, 2B and 0.8B would load too, but are
    /// untested, so they are not offered.
    public static func family(of path: String) -> ModelFamily? {
        let url = URL(fileURLWithPath: path).appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: url),
              let json = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
        else { return nil }

        let type = json["model_type"] as? String ?? ""
        let family: ModelFamily
        switch type {
        case "qwen3_5":
            let text = json["text_config"] as? [String: Any]
            switch text?["hidden_size"] as? Int {
            case 4096: family = .nineB
            case 5120: family = .dense
            default: return nil
            }
        case "qwen4_exp": family = .flashNext
        case "qwen4_exp_text" where json["text_config"] == nil: family = .flashNext
        default: return nil
        }

        let q = (json["quantization"] ?? json["quantization_config"]) as? [String: Any]
        guard let bits = q?["bits"] as? Int, bits == 4,
              let group = q?["group_size"] as? Int, group == 32 || group == 64
        else { return nil }
        return family
    }

    /// The longest context the model was trained for: max_position_embeddings,
    /// from text_config where the config nests one.
    public static func maxContext(of path: String) -> Int? {
        let url = URL(fileURLWithPath: path).appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: url),
              let json = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
        else { return nil }
        let text = json["text_config"] as? [String: Any]
        let n = (text?["max_position_embeddings"] ?? json["max_position_embeddings"]) as? Int
        return n.flatMap { $0 > 0 ? $0 : nil }
    }

    /// Every loadable model folder under the usual places, the checkout's
    /// first: `models/` beside it (recorded at build time, since the installed
    /// app does not sit there), LM Studio's library, and the Hugging Face
    /// cache.  `extra` are folders to consider as well, such as the current
    /// choice.  Symlinked duplicates appear once.
    public static func scan(extra: [String]) -> [FoundModel] {
        let fm = FileManager.default
        let home = fm.homeDirectoryForCurrentUser.path
        var candidates = extra

        func children(_ dir: String) -> [String] {
            ((try? fm.contentsOfDirectory(atPath: dir)) ?? [])
                .filter { !$0.hasPrefix(".") }
                .sorted()
                .map { (dir as NSString).appendingPathComponent($0) }
        }

        if let dir = Bundle.main.object(forInfoDictionaryKey: "QWModelsDir") as? String, !dir.isEmpty {
            candidates += children(dir)
        }
        if let built = Bundle.main.object(forInfoDictionaryKey: "QWDefaultModel") as? String,
           !built.isEmpty {
            candidates += children((built as NSString).deletingLastPathComponent)
        }
        for org in children(home + "/.lmstudio/models") { candidates += children(org) }
        for repo in children(home + "/.cache/huggingface/hub") where
            (repo as NSString).lastPathComponent.hasPrefix("models--") {
            candidates += children(repo + "/snapshots")
        }

        var seen = Set<String>()
        var found: [FoundModel] = []
        for path in candidates {
            let real = URL(fileURLWithPath: path).resolvingSymlinksInPath().path
            guard !seen.contains(real) else { continue }
            seen.insert(real)
            if let family = family(of: path) { found.append(FoundModel(family: family, path: path)) }
        }
        return found
    }
}
