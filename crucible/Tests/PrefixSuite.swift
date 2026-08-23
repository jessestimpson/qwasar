// PrefixSuite.swift -- the invariant the checkpoint prefix rests on.
//
// Session.primeSystemPrefix evaluates the system turn ALONE, checkpoints it,
// and then evaluates the rest of the first turn on top. That is only sound if
// the system turn is a true TOKEN prefix of the first turn -- not a similar
// string, not the same text, the same leading ids. Forty-eight of the model's
// sixty-four layers are recurrent (PLAN.md 2.2), so if the split point is wrong
// there is no way back: the state cannot be rewound to try again, and the
// damage is a silently mis-primed conversation rather than a crash.
//
// Both sides render through ChatTemplate.systemPrefix, so this cannot pass by
// agreeing with a copy of the session's own logic.
//
// Tokenizer only, so it costs about a second and runs in `make test`.

import Foundation
import CrucibleKit

enum PrefixSuite {
    static func run(_ args: [String]) -> Int {
        let modelPath = TestMain.value(of: "--model", in: args)
            ?? ProcessInfo.processInfo.environment["QWASAR_TEST_MODEL"]
            ?? (NSHomeDirectory() + "/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-4bit")

        guard let tok = try? Tokenizer(modelPath: modelPath) else {
            print("  SKIP no tokenizer at \(modelPath)")
            return 0
        }

        var failures = 0
        let efforts: [ReasoningEffort] = [.low, .medium, .xhigh]
        // Both executors, because the surface is part of the prefix and the
        // read-only runner offers three tools where the sandbox offers twelve.
        let surfaces: [(String, [String])] = [
            ("guest", ToolSurface.guestSchemas),
            ("read-only", ToolSurface.readOnlySchemas),
        ]
        let system = SandboxToolRunner(channel: VsockChannel(fileDescriptor: -1))
            .environmentDescription + "\n\n" + Project.defaultSystem

        // The messages a first turn actually starts with, including the shapes
        // that have historically broken tokenizers: empty-ish, multibyte, and
        // long enough to cross a prefill chunk.
        let firsts = [
            "hi",
            "what does Session.swift do?",
            "explain — with the em dash — what 🚀 does",
            String(repeating: "the quick brown fox ", count: 200),
        ]

        for (label, tools) in surfaces {
            for effort in efforts {
                guard let prefix = try? ChatTemplate.systemPrefix(
                        system, tokenizer: tok, thinking: true,
                        effort: effort, tools: tools) else {
                    failures += TestMain.check(false, "\(label)/\(effort.rawValue): renders")
                    continue
                }

                // Below the engine's floor a checkpoint is refused outright, so
                // the whole optimisation would be dead code.
                failures += TestMain.check(prefix.count >= 256,
                    "\(label)/\(effort.rawValue): prefix is \(prefix.count) tokens, over the 256 floor")

                for msg in firsts {
                    guard let full = try? ChatTemplate.render(
                            [.system(system), .user(msg)], tokenizer: tok,
                            thinking: true, effort: effort, tools: tools) else {
                        failures += TestMain.check(false, "\(label): renders first turn")
                        continue
                    }
                    let isPrefix = full.count > prefix.count
                        && Array(full.prefix(prefix.count)) == prefix
                    failures += TestMain.check(isPrefix,
                        "\(label)/\(effort.rawValue): system turn is a strict token prefix "
                        + "of a \(msg.count)-char first turn")
                    if !isPrefix {
                        let i = zip(prefix, full).enumerated()
                            .first { $0.element.0 != $0.element.1 }?.offset
                        print("       diverges at token \(i.map(String.init) ?? "end")"
                            + " (prefix \(prefix.count), full \(full.count))")
                    }
                }
            }
        }

        // The point of the exercise: almost the whole first turn is the prefix.
        // If this collapses, something moved the tool schemas out of the system
        // turn and the checkpoint stopped being worth its ~272 MB.
        if let prefix = try? ChatTemplate.systemPrefix(system, tokenizer: tok, thinking: true,
                                                       effort: .medium, tools: ToolSurface.guestSchemas),
           let full = try? ChatTemplate.render([.system(system), .user("hi")], tokenizer: tok,
                                               thinking: true, effort: .medium,
                                               tools: ToolSurface.guestSchemas) {
            let share = Double(prefix.count) / Double(full.count)
            failures += TestMain.check(share > 0.9, String(
                format: "prefix covers %.1f%% of the first turn (%d of %d tokens)",
                share * 100, prefix.count, full.count))
        }

        // What a checkpoint may and may not be reused for.
        //
        // The store compares its STORED tokens against the leading tokens of
        // the request before unpacking anything, so a hit is by construction
        // the right state and a partial hit just leaves more to evaluate. The
        // hazard is upstream of that: a knob that changes how the model behaves
        // but does NOT change the prefix would mean one checkpoint standing in
        // for two different system turns, and nothing downstream could tell.
        func prefixOf(_ sys: String, _ effort: ReasoningEffort, _ tools: [String]) -> [Int32]? {
            try? ChatTemplate.systemPrefix(sys, tokenizer: tok, thinking: true,
                                           effort: effort, tools: tools)
        }
        guard let base = prefixOf(system, .medium, ToolSurface.guestSchemas) else {
            return failures + TestMain.check(false, "renders the baseline prefix")
        }
        let mustDiffer: [(String, [Int32]?)] = [
            ("effort", prefixOf(system, .xhigh, ToolSurface.guestSchemas)),
            ("the tool surface", prefixOf(system, .medium, ToolSurface.readOnlySchemas)),
            ("the system prompt", prefixOf(system + " Be terse.", .medium, ToolSurface.guestSchemas)),
        ]
        for (what, other) in mustDiffer {
            guard let b = other else {
                failures += TestMain.check(false, "renders a variant of \(what)"); continue
            }
            let shares = (b.count >= base.count && Array(b.prefix(base.count)) == base)
                      || (base.count >= b.count && Array(base.prefix(b.count)) == b)
            failures += TestMain.check(!shares,
                "changing \(what) changes the prefix, so it cannot reuse this checkpoint")
        }

        // `thinking` is the exception, and it is recorded rather than asserted
        // away: with tools present the template rebuilds the system turn around
        // them and renders the SAME tokens either way. That is safe here --
        // identical tokens mean identical state, so the hit is correct -- but it
        // is a property of the template, not a decision this cache made, and if
        // it ever changes the checkpoint story changes with it.
        if let off = try? ChatTemplate.render([.system(system)], tokenizer: tok,
                                              thinking: false, effort: .medium,
                                              addGenerationPrompt: false,
                                              tools: ToolSurface.guestSchemas) {
            failures += TestMain.check(off == base,
                "thinking does not affect the system turn when tools are present "
                + "(same tokens, so a shared checkpoint is the same state)")
        }

        return failures
    }
}
