import Foundation

/// The two models qwasar-server runs, told apart by their config.json the way
/// the engine tells them apart -- not by folder name, which is whatever the
/// download happened to be called.
enum ModelFamily: CaseIterable, Sendable {
    case dense        // Qwen3.8 27B: model_type qwen3_5
    case flashNext    // Qwen3.8 Flash-Next: model_type qwen4_exp

    var title: String {
        switch self {
        case .dense: "Qwen3.8 27B"
        case .flashNext: "Qwen3.8 Flash-Next"
        }
    }

    /// What loading it asks of the machine, for the menu's tooltip.
    var note: String {
        switch self {
        case .dense: "Dense; ~18 GB on disk, nearly all of it held in memory."
        case .flashNext: "Mixture of experts; ~104 GB on disk, ~75 GB held in memory "
                       + "(its engram table stays on disk).  The server will not load it "
                       + "without that much memory free."
        }
    }
}

struct FoundModel: Sendable {
    let family: ModelFamily
    let path: String
}

/// Finds model folders the server can load.
enum ModelCatalog {
    /// The family of the model in `path`, or nil if the engine would refuse it:
    /// an unsupported model_type, or anything but 4-bit weights in groups of
    /// 32 or 64 (an FP8 or BF16 download of the same model, say).
    static func family(of path: String) -> ModelFamily? {
        let url = URL(fileURLWithPath: path).appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: url),
              let json = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
        else { return nil }

        let type = json["model_type"] as? String ?? ""
        let family: ModelFamily
        switch type {
        case "qwen3_5": family = .dense
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
    static func maxContext(of path: String) -> Int? {
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
    static func scan(extra: [String]) -> [FoundModel] {
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
