// Events.swift -- what a turn tells the window, and what a finished one cost.
//
// These used to be produced by an in-process engine loop (Session.swift, now
// gone); they are produced from the server's events (QwasarClient) instead,
// and the window consumes them unchanged.  API.md §5 was written from this
// enum, so the translation is a rename.

import Foundation

/// The reasoning effort a session runs at.  Part of the prefix (the template
/// renders it into the system turn), so fixed for a session's life; the raw
/// values are the server's and are persisted in records.
public enum ReasoningEffort: String, Sendable, CaseIterable, Codable {
    case low, medium, xhigh

    /// How it reads to a person.  The model's own default is xhigh.
    public var label: String {
        switch self {
        case .low: return "low"
        case .medium: return "medium"
        case .xhigh: return "high"
        }
    }
}

public enum SessionEvent: Sendable {
    case prefill(done: Int, total: Int)
    /// How much of the window this session has consumed, as it consumes it.
    case context(used: Int, limit: Int)
    /// Decode rate: the turn's average, and the last second's.
    case rate(generated: Int, tokensPerSecond: Double, instantaneous: Double)
    /// Reasoning text, with how many tokens produced it.
    case reasoning(String, tokens: Int)
    case text(String)
    /// A call being written, before it is complete.
    case toolCallProgress(name: String?, keys: [String], tokens: Int)
    case toolCall(ToolCall)
    case toolResult(name: String, result: String)
    case note(String)
    case turnFinished(TurnStats)
    case contextFull(used: Int, limit: Int)
    case failed(String)
}

/// What one user turn cost, as the footer shows it.  The field names are
/// persisted in transcript.jsonl, so they stay what they were.
public struct TurnStats: Sendable, Codable {
    public var promptTokens = 0
    public var generatedTokens = 0
    public var reasoningTokens = 0
    public var toolCalls = 0
    public var prefillSeconds = 0.0
    public var decodeSeconds = 0.0
    public var contextUsed = 0
    public var contextLimit = 0
    /// Speculation, when the server runs a draft head.
    public var specRounds = 0
    public var specCommitted = 0
    public var tokensPerRound: Double {
        specRounds > 0 ? Double(specCommitted) / Double(specRounds) : 0
    }

    public var hitEOS = false
    public var hitBudget = false
    public var hitStepCap = false
    public var interrupted = false
    public var stoppedInReasoning = false

    public var tokensPerSecond: Double {
        decodeSeconds > 0 ? Double(generatedTokens) / decodeSeconds : 0
    }

    public var prefillTokensPerSecond: Double {
        prefillSeconds > 0 ? Double(promptTokens) / prefillSeconds : 0
    }

    public init() {}
}
