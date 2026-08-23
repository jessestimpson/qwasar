// ChatTemplate.swift -- marshalling Swift messages into qwasar_message.
//
// PLAN.md 10 makes this the highest-value thing to test: a silently different
// system turn is a silently different model, invisible until quality drops for
// no reason anyone can name. So this file does the marshalling and nothing
// else, and the golden-vector test compares its output byte for byte against
// qwasar_apply_chat_template driven from C.
//
// Note what the header says about the two encoders: `content` goes through an
// encoder that never emits control tokens -- which is what stops a user's text
// from injecting a role marker -- while `tool_calls` goes through one that maps
// them, because it is reconstructed rather than supplied. Images are declared
// as a count, not written into content, for the same reason.

import Foundation
import CQwasar

public struct ChatMessage {
    public var role: String
    public var content: String
    public var reasoning: String?
    public var toolCalls: String?
    public var imageTokens: Int32
    public var visionIsVideo: Bool

    public init(role: String, content: String, reasoning: String? = nil,
                toolCalls: String? = nil, imageTokens: Int32 = 0,
                visionIsVideo: Bool = false) {
        self.role = role
        self.content = content
        self.reasoning = reasoning
        self.toolCalls = toolCalls
        self.imageTokens = imageTokens
        self.visionIsVideo = visionIsVideo
    }

    public static func system(_ s: String) -> ChatMessage { .init(role: "system", content: s) }
    public static func user(_ s: String) -> ChatMessage { .init(role: "user", content: s) }
    public static func assistant(_ s: String) -> ChatMessage { .init(role: "assistant", content: s) }
}

public enum ReasoningEffort: String, Sendable, CaseIterable, Codable {
    case low, medium, xhigh

    /// How it reads to a person. The model's own default is xhigh, and it is
    /// not a small difference: the goldens record system turns of 49, 23 and 61
    /// tokens for low, medium and xhigh -- three genuinely different prompts,
    /// not one prompt with a dial.
    public var label: String {
        switch self {
        case .low: return "low"
        case .medium: return "medium"
        case .xhigh: return "high"
        }
    }
}

public enum ChatTemplate {

    /// Renders a conversation as ChatML and encodes it.
    ///
    /// The returned array is a copy; the engine's malloc'd buffer is freed
    /// before this returns (PLAN.md 3.4).
    ///
    /// The defaults here mirror the C API's documented ones -- `qwasar.h` says
    /// xhigh -- and must keep doing so, because `Tests/golden.tsv` compares this
    /// path against `qwasar_apply_chat_template` driven from C with those same
    /// defaults. What the *application* starts a session at is a separate
    /// decision and lives in `Project.effort`; it is currently medium.
    public static func render(_ msgs: [ChatMessage],
                              tokenizer: Tokenizer,
                              thinking: Bool = true,
                              effort: ReasoningEffort = .xhigh,
                              addGenerationPrompt: Bool = true,
                              tools: [String] = []) throws -> [Int32] {
        let arena = CStringArena()

        var cmsgs: [qwasar_message] = msgs.map { m in
            var c = qwasar_message()
            c.role            = arena.dup(m.role)
            c.content         = arena.dup(m.content)
            c.reasoning       = arena.dup(m.reasoning)
            c.tool_calls      = arena.dup(m.toolCalls)
            c.n_image_tokens  = m.imageTokens
            c.vision_is_video = m.visionIsVideo
            return c
        }

        // Tool schemas are a `const char *const *`; the arena keeps the strings
        // alive and this keeps the array of pointers alive across the call.
        var toolPtrs: [UnsafePointer<CChar>?] = tools.map { arena.dup($0) }

        var opts = qwasar_chat_options()
        opts.enable_thinking       = thinking
        opts.reasoning_effort      = arena.dup(effort.rawValue)
        opts.add_generation_prompt = addGenerationPrompt
        opts.n_tools               = Int32(tools.count)

        var count: Int32 = 0
        let (owned, err): (UnsafeMutablePointer<Int32>?, String) = withErrorBuffer { buf, cap in
            toolPtrs.withUnsafeMutableBufferPointer { tp -> UnsafeMutablePointer<Int32>? in
                opts.tools = tools.isEmpty ? nil : UnsafePointer(tp.baseAddress)
                return cmsgs.withUnsafeMutableBufferPointer { mp in
                    withUnsafePointer(to: &opts) { op in
                        qwasar_apply_chat_template(tokenizer.handle, mp.baseAddress, Int32(mp.count),
                                                   op, &count, buf, cap)
                    }
                }
            }
        }

        guard let tokens = takeOwnedTokens(owned, count: count) else {
            throw EngineError.template(err.isEmpty ? "unknown" : err)
        }
        return tokens
    }

    /// Plain text, encoded. Control tokens are never produced however the text
    /// is spelled, which is the property that makes untrusted content safe here.
    public static func encode(_ text: String, tokenizer: Tokenizer) -> [Int32] {
        var n: Int32 = 0
        let p = qwasar_encode(tokenizer.handle, text, &n)
        return takeOwnedTokens(p, count: n) ?? []
    }

    /// Decodes a token sequence for display or for a golden-vector comparison.
    /// Special tokens are rendered in their literal spelling so a rendered
    /// prompt can be read and diffed.
    public static func decode(_ tokens: [Int32], tokenizer: Tokenizer,
                              includeSpecial: Bool = true) -> String {
        var asm = UTF8Assembler()
        var out = ""
        for t in tokens {
            tokenizer.withBytes(of: t) { bytes, special in
                if special && !includeSpecial { return }
                if let s = asm.feed(bytes) { out += s }
            }
        }
        if let s = asm.flush() { out += s }
        return out
    }

    /// Continuations, for a loop that feeds back what it generated rather than
    /// re-rendering the conversation each turn. Both close the open assistant
    /// turn, add their own, and reopen a generation prompt.
    /// The system turn on its own, ending exactly where the user turn begins.
    ///
    /// Rendered with no generation prompt, which is what makes it a true token
    /// prefix of the first turn rather than merely a similar string. Session's
    /// checkpoint priming splits on this, and `PrefixSuite` asserts the prefix
    /// property against it -- one definition, so a template change cannot make
    /// the test agree with a session that has drifted.
    public static func systemPrefix(_ system: String, tokenizer: Tokenizer,
                                    thinking: Bool, effort: ReasoningEffort,
                                    tools: [String]) throws -> [Int32] {
        try render([.system(system)], tokenizer: tokenizer, thinking: thinking,
                   effort: effort, addGenerationPrompt: false, tools: tools)
    }

    public static func renderToolResult(_ result: String, tokenizer: Tokenizer,
                                        thinking: Bool = true,
                                        effort: ReasoningEffort = .xhigh,
                                        toolCount: Int = 0) -> [Int32]? {
        let arena = CStringArena()
        var opts = qwasar_chat_options()
        opts.enable_thinking = thinking
        opts.reasoning_effort = arena.dup(effort.rawValue)
        opts.add_generation_prompt = true
        opts.n_tools = Int32(toolCount)
        var n: Int32 = 0
        let p = withUnsafePointer(to: &opts) {
            qwasar_render_tool_result(tokenizer.handle, result, $0, &n)
        }
        return takeOwnedTokens(p, count: n)
    }

    public static func renderUserTurn(_ text: String, tokenizer: Tokenizer,
                                      imageTokens: Int32 = 0, isVideo: Bool = false,
                                      thinking: Bool = true,
                                      effort: ReasoningEffort = .xhigh,
                                      toolCount: Int = 0) -> [Int32]? {
        let arena = CStringArena()
        var opts = qwasar_chat_options()
        opts.enable_thinking = thinking
        opts.reasoning_effort = arena.dup(effort.rawValue)
        opts.add_generation_prompt = true
        opts.n_tools = Int32(toolCount)
        var n: Int32 = 0
        let p = withUnsafePointer(to: &opts) {
            qwasar_render_user_turn(tokenizer.handle, text, imageTokens, isVideo, $0, &n)
        }
        return takeOwnedTokens(p, count: n)
    }
}
