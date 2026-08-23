// Session.swift -- one inference timeline, and the agent loop over it.
//
// PLAN.md 2.2: a session is append-only. Forty-eight of the model's sixty-four
// layers carry recurrent state with no per-position history, so it can be
// extended but never rewound. An agent loop is a natural fit -- every turn
// appends, nothing edits history -- so each step feeds back exactly the tokens
// the model produced plus the rendered tool result, and the caches carry
// forward untouched. Re-rendering the conversation each turn would be both
// slower and, for the recurrent half, wrong.
//
// The shape of runTurn() is qwasar_agent.c's generate() + agent_run(), and the
// details that look incidental are not: EOS is never appended (the continuation
// renderers close the open assistant turn), tool-call markup is accumulated but
// not echoed, and generation stops at the closing tag rather than the budget.
//
// Everything here runs on the engine queue. Nothing in this file is safe to
// call from anywhere else.

import Foundation
import CQwasar

public enum SessionEvent: Sendable {
    case prefill(done: Int, total: Int)
    /// How much of the window this session has consumed, as it consumes it.
    ///
    /// Emitted rather than inferred: a caller can only guess at context from
    /// what it has seen streamed, and a guess in a precision display is worse
    /// than no display. The session knows exactly, so it says so -- and because
    /// a full window is the end of a session (PLAN.md 2.4) rather than a
    /// degradation, watching it fill is something a user needs to be able to do.
    case context(used: Int, limit: Int)
    /// Decode rate, as it is happening.
    ///
    /// Reported rather than derived: a caller counting streamed characters is
    /// measuring the wrong thing (a token is one to several bytes, and the
    /// reasoning block is not shown at all), and at ~6 tok/s a wrong rate is
    /// obviously wrong to the person watching it.
    /// `tokensPerSecond` is the whole turn's average; `instantaneous` covers
    /// only the tokens since the previous report, so it is the number that
    /// moves when the phase changes -- sampled reasoning versus speculative
    /// answer are different regimes, and averaging across them hides both.
    case rate(generated: Int, tokensPerSecond: Double, instantaneous: Double)
    /// Reasoning text, with how many TOKENS produced it.
    ///
    /// One delta is not one token: the UTF-8 assembler holds bytes back until a
    /// character completes, so a single emission can cover several. The count
    /// travels with the text because only the decode loop knows it.
    case reasoning(String, tokens: Int)
    case text(String)
    /// A call being written, before it is complete.
    ///
    /// The markup itself is never echoed (it would bury any narration the model
    /// wrote first), which left the longest part of many turns looking like a
    /// hang. This says what is being built and how far along it is.
    case toolCallProgress(name: String?, keys: [String], tokens: Int)
    case toolCall(ToolCall)
    case toolResult(name: String, result: String)
    case note(String)                 // budget reached, step cap, malformed call
    case turnFinished(TurnStats)
    case contextFull(used: Int, limit: Int)
    case failed(String)
}

public struct TurnStats: Sendable, Codable {
    public var promptTokens = 0
    public var generatedTokens = 0
    public var reasoningTokens = 0
    public var toolCalls = 0
    public var prefillSeconds = 0.0
    public var decodeSeconds = 0.0
    public var contextUsed = 0
    public var contextLimit = 0
    /// Speculation, when a draft head is loaded.
    ///
    /// Two numbers rather than an acceptance rate, because the rate is only
    /// meaningful with the round count beside it: 2.0 tokens a round over four
    /// rounds says nothing, and over four hundred says the head is earning its
    /// memory.
    public var specRounds = 0
    public var specCommitted = 0
    /// Tokens committed per speculative round. 1.0 means the head proposed
    /// nothing useful and the rounds cost a forward pass each; the engine's own
    /// draft_depth backs off to zero when that persists.
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

    /// Prefill rate, which in a tool-using loop is the number that decides how
    /// a turn feels: every tool result is prefilled before the model can react
    /// to it (PLAN.md 2.5, measured at ~32 tok/s).
    public var prefillTokensPerSecond: Double {
        prefillSeconds > 0 ? Double(promptTokens) / prefillSeconds : 0
    }

    public init() {}
}

public struct SessionConfig: Sendable {
    public var system: String
    public var thinking: Bool
    public var effort: ReasoningEffort
    public var maxTokensPerTurn: Int
    public var maxToolSteps: Int
    /// Sampling. The defaults are the model's own generation_config -- the
    /// same values `qwasar_sampling_defaults` encodes -- because Qwen's
    /// guidance for thinking-mode models warns against greedy decoding: long
    /// reasoning chains under argmax are prone to repetition loops.
    /// `temperature = 0` means greedy and is exactly reproducible, which is
    /// what the headless gates want.
    public var temperature: Float
    public var topK: Int32
    public var topP: Float
    public var minP: Float
    /// 0 means seed from the clock at turn start; nonzero is reproducible.
    public var seed: UInt64

    public init(system: String,
                thinking: Bool = true,
                effort: ReasoningEffort = .medium,
                maxTokensPerTurn: Int = 4096,
                maxToolSteps: Int = 24,
                temperature: Float = 1.0,
                topK: Int32 = 20,
                topP: Float = 0.95,
                minP: Float = 0,
                seed: UInt64 = 0) {
        self.system = system
        self.thinking = thinking
        self.effort = effort
        self.maxTokensPerTurn = maxTokensPerTurn
        self.maxToolSteps = maxToolSteps
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.minP = minP
        self.seed = seed
    }
}

/// One live `qwasar_session` plus the history needed to rebuild it.
///
/// `tokens` is the load-bearing field: it is exactly the sequence this session
/// has evaluated, in order, and it is what a later restore is fed. At 4 bytes a
/// token a full 90K session is 360 KB, so keeping all of it is free next to the
/// ~6 GB of state it describes.
final class LiveSession {
    let handle: OpaquePointer
    let tokenizer: Tokenizer
    let engine: OpaquePointer
    let contextLimit: Int32
    var config: SessionConfig
    var runner: ToolExecuting
    var tokens: [Int32] = []
    var started = false
    /// The system turn, once this session's caches hold it.
    ///
    /// Kept as tokens rather than as a count so that a first turn which primed
    /// the prefix and then failed can be retried safely: the retry re-renders
    /// the conversation and checks that what it rendered still BEGINS with what
    /// was evaluated, rather than trusting that nothing moved underneath it.
    var primedPrefix: [Int32] = []
    /// Length of the last checkpoint written, so the same state is not saved
    /// twice. A save is several hundred megabytes; a redundant one is a
    /// second of the engine queue and a slot in the LRU budget.
    private var lastCheckpoint = 0
    /// Prefill chunk used when replaying history. Matches the engine's own
    /// default so the progress callback fires at the same cadence as a normal
    /// prefill and the bar behaves identically.
    static let replayChunk = 256

    /// Held for the life of the session because the C side keeps an unretained
    /// pointer to it (PLAN.md 3.4).
    var progressBox: ProgressBox?

    init(handle: OpaquePointer, tokenizer: Tokenizer, engine: OpaquePointer,
         contextLimit: Int32, config: SessionConfig, runner: ToolExecuting) {
        self.handle = handle
        self.tokenizer = tokenizer
        self.engine = engine
        self.contextLimit = contextLimit
        self.config = config
        self.runner = runner
    }

    deinit {
        qwasar_session_set_progress(handle, nil, nil)
        qwasar_session_free(handle)
    }

    var nPast: Int32 { qwasar_session_n_past(handle) }

    // MARK: Rebuilding a session from its history

    /// Restores this session to the end of `history`.
    ///
    /// PLAN.md 2.2: only a true prefix is reusable, which is exactly what a
    /// session's own recorded history is. `qwasar_session_restore` fills a
    /// fresh session from the longest cached prefix and returns how many tokens
    /// it covered -- 0 is a cold rebuild, not an error -- and the remainder is
    /// evaluated here.
    ///
    /// Until M6 this is the whole story, and it is slow: a cold rebuild of a
    /// 4000-token session is about two minutes at the measured prefill rate
    /// (PLAN.md 2.5). The checkpoint below is what usually saves it, and the
    /// explicit-path API in PLAN.md 4.4 is what will make it reliable.
    func restore(history: [Int32], emit: @escaping (SessionEvent) -> Void) -> Bool {
        guard !history.isEmpty else { return true }

        installProgress(emit)

        let covered = history.withUnsafeBufferPointer { p in
            Int(qwasar_session_restore(handle, engine, p.baseAddress, Int32(p.count)))
        }
        tokens = Array(history.prefix(covered))
        if covered > 0 { emit(.note("restored \(covered) tokens from the cache")) }

        var i = covered
        while i < history.count {
            let end = min(i + Self.replayChunk, history.count)
            // Same trap as the prefix: without this the engine reports 0/256
            // per chunk and the bar never moves through a replay that can take
            // minutes.
            progressBox?.span = (base: i, total: history.count)
            let ok: Bool = autoreleasepool {
                let (logits, err) = withErrorBuffer { buf, cap in
                    history[i..<end].withUnsafeBufferPointer { p in
                        qwasar_session_eval(handle, p.baseAddress, Int32(p.count), buf, cap)
                    }
                }
                guard logits != nil else { emit(.failed("rebuilding the session: \(err)")); return false }
                return true
            }
            guard ok else { progressBox?.span = nil; return false }
            tokens.append(contentsOf: history[i..<end])
            emit(.prefill(done: end, total: history.count))
            i = end
        }
        progressBox?.span = nil

        // A session with history has, by construction, already been given its
        // system turn -- so the next user message is a continuation, not a
        // fresh conversation.
        started = true
        return true
    }

    /// Writes the session to the engine's checkpoint cache so the next rebuild
    /// is a read rather than a re-prefill.
    ///
    /// Best-effort on purpose: the cache is hash-keyed with a 6 GB LRU budget
    /// (PLAN.md 4.4), so a large session's checkpoint may be evicted or may not
    /// fit at all. Failing to cache is not a failure of the turn.
    ///
    /// Called at two points and no others: at the system-turn boundary, and
    /// when the session closes. It used to run at the end of every completed
    /// turn, which sounds harmless and was not -- each turn wrote a fresh
    /// several-hundred-megabyte snapshot of the same conversation, one strict
    /// superset of the last. A real cache, read off this machine, held nine
    /// such files: 5.30 GiB of a 6 GiB budget, ONE hit between them, and no
    /// room left for the ~272 MB system prefix that every session would have
    /// hit. Frequency was the bug, not the caching.
    /// Returns nil on success, or why it did not save.
    ///
    /// A reason rather than a Bool, and reported rather than dropped: a
    /// checkpoint that is never written is invisible from the outside -- every
    /// session simply stays slow -- and that is exactly how the per-turn policy
    /// went wrong for a fortnight without anyone noticing.
    @discardableResult
    func checkpoint() -> String? {
        guard tokens.count >= Self.minCheckpointTokens else {
            return "only \(tokens.count) tokens, under the engine's \(Self.minCheckpointTokens) floor"
        }
        guard tokens.count != lastCheckpoint else { return nil }
        lastCheckpoint = tokens.count
        let (ok, err) = withErrorBuffer { buf, cap in
            qwasar_session_save(handle, engine, buf, cap)
        }
        return ok ? nil : (err.isEmpty ? "the engine did not say" : err)
    }

    /// The engine's own floor (QW_KV_MIN_TOKENS). Below it a checkpoint is
    /// refused, because the ~149 MB of recurrent state is the same size for ten
    /// tokens as for ten thousand.
    static let minCheckpointTokens = 256

    // MARK: The system prefix

    /// Puts the system turn into the session on its own -- from the checkpoint
    /// cache when it is there, leaving a checkpoint behind when it is not.
    ///
    /// Measured on this surface: the system turn is 1954 tokens of the 1970 in
    /// a first turn, 99.2%, because the twelve tool schemas are rendered into
    /// it. At the 31.8 tok/s of PLAN.md 2.5 that is about a minute of prefill,
    /// and it is IDENTICAL for every session of a project at a given effort --
    /// so it is paid again on every new session and, since compaction is a
    /// successor session (PLAN.md 2.4), on every compaction too. One ~272 MB
    /// checkpoint (149.6 MB fixed floor plus 64 KB/token, fitted from real
    /// files) buys all of them.
    ///
    /// This does not change what the model sees. The first turn is rendered
    /// exactly as before and then SPLIT; the only thing that moves is where the
    /// eval boundary falls. The split point is checked to be a true prefix of
    /// what was rendered and the whole optimisation is skipped if it is not,
    /// because a recurrent state cannot be rewound if this is wrong (PLAN.md
    /// 2.2) -- there would be no way back from a half-evaluated system turn.
    ///
    /// Effort is part of the prefix and not of the key, which is the right way
    /// round: the store compares stored tokens before use, so a medium
    /// checkpoint offered to an xhigh session is a miss rather than a
    /// corruption. Context size is not part of it at all -- the payload is
    /// stored densely and unpacked against the target's stride -- so one prefix
    /// serves a 32K session and a 90K one.
    ///
    /// Returns how many leading tokens of `full` the session already holds; 0
    /// when nothing was primed, which is not an error. nil means the session is
    /// no longer usable and the turn must be abandoned.
    func primeSystemPrefix(full: [Int32], stats: inout TurnStats,
                           emit: @escaping (SessionEvent) -> Void) -> Int? {
        // A first turn that primed the prefix and then failed -- a full window,
        // a cancelled render -- leaves the system turn evaluated with `started`
        // still false, so the next attempt renders the whole conversation
        // again. Evaluating that on top would give the model two system turns.
        if !primedPrefix.isEmpty {
            guard full.count > primedPrefix.count,
                  Array(full.prefix(primedPrefix.count)) == primedPrefix else {
                emit(.failed("the system turn changed after it was evaluated; "
                           + "this session has to be started again"))
                return nil
            }
            return primedPrefix.count
        }

        guard nPast == 0, tokens.isEmpty else { return 0 }

        // No generation prompt, so it ends exactly where the user turn begins.
        guard let prefix = try? ChatTemplate.systemPrefix(config.system,
                                                         tokenizer: tokenizer,
                                                         thinking: config.thinking,
                                                         effort: config.effort,
                                                         tools: runner.schemas),
              prefix.count >= Self.minCheckpointTokens,
              full.count > prefix.count,
              Array(full.prefix(prefix.count)) == prefix
        else { return 0 }

        // Longest cached prefix of `prefix`, across every checkpoint the store
        // holds. 0 is a cold start, not a failure.
        let covered = prefix.withUnsafeBufferPointer { p in
            Int(qwasar_session_restore(handle, engine, p.baseAddress, Int32(p.count)))
        }
        tokens = Array(prefix.prefix(covered))

        if covered == prefix.count {
            // Deliberately NOT added to stats.promptTokens. Nothing was
            // prefilled, and folding a free 1954 tokens into the rate would
            // report a prefill speed the machine cannot do.
            primedPrefix = prefix
            lastCheckpoint = covered
            emit(.note("reused the \(covered)-token system prefix from the cache"))
            emit(.context(used: Int(nPast), limit: Int(contextLimit)))
            return covered
        }

        // ONE eval, not a loop of chunked ones. The engine chunks internally
        // and reports progress across whatever it was handed, so a single call
        // gives a bar that runs 0 -> 2315; nine calls of 256 give "0/256" nine
        // times, which is what a person reads as hung. There is nothing to gain
        // from chunking here: it is one contiguous block either way.
        let rest = Array(prefix[covered...])
        let t0 = Date()
        let ok: Bool = autoreleasepool {
            let (logits, err) = withErrorBuffer { buf, cap in
                rest.withUnsafeBufferPointer { p in
                    qwasar_session_eval(handle, p.baseAddress, Int32(p.count), buf, cap)
                }
            }
            guard logits != nil else {
                emit(.failed("evaluating the system turn: \(err)")); return false
            }
            return true
        }
        guard ok else { return nil }
        tokens.append(contentsOf: rest)
        stats.promptTokens += rest.count
        stats.prefillSeconds += Date().timeIntervalSince(t0)
        primedPrefix = prefix
        emit(.context(used: Int(nPast), limit: Int(contextLimit)))

        // Said out loud, because otherwise this is a silent minute followed by
        // silent instant sessions, and neither is attributable to anything.
        if let why = checkpoint() {
            emit(.note("could not cache the system prefix: \(why). "
                     + "Every new session will pay this prefill again."))
        } else {
            emit(.note(String(format: "evaluated the %d-token system prefix in %.0fs and cached it; "
                            + "later sessions in this project start from it",
                              prefix.count, Date().timeIntervalSince(t0))))
        }
        return prefix.count
    }

    private func installProgress(_ emit: @escaping (SessionEvent) -> Void) {
        guard progressBox == nil else { return }
        // PLAN.md 5.4 / AGENT_BAR_MIN_TOKENS: a bar that flashes is worse than
        // none, so short prefills report nothing.
        let box = ProgressBox { done, total in
            if total >= 128 { emit(.prefill(done: done, total: total)) }
        }
        progressBox = box
        qwasar_session_set_progress(handle, progressTrampoline,
                                    Unmanaged.passUnretained(box).toOpaque())
    }

    // MARK: One user turn, to completion

    func runTurn(userText: String,
                 cancelled: @escaping @Sendable () -> Bool,
                 emit: @escaping (SessionEvent) -> Void) {
        var stats = TurnStats()
        stats.contextLimit = Int(contextLimit)

        // Progress is wired once per session, not per turn: the callback reads
        // nothing turn-specific, and re-registering it each turn would mean
        // re-boxing a pointer the C side already holds.
        installProgress(emit)

        // --- the prompt for this step -------------------------------------
        //
        // First turn renders the whole conversation, because the system turn
        // and the tool schemas have to exist. Every turn after it renders only
        // the continuation, which is the entire point of an append-only
        // session.
        var prompt: [Int32]
        let wasFirstTurn = !started
        do {
            if wasFirstTurn {
                let msgs: [ChatMessage] = [.system(config.system), .user(userText)]
                prompt = try ChatTemplate.render(msgs, tokenizer: tokenizer,
                                                 thinking: config.thinking,
                                                 effort: config.effort,
                                                 tools: runner.schemas)
                // The system turn is the same ~1954 tokens for every session of
                // this project at this effort, so it comes out of the
                // checkpoint cache when it can. What is left to evaluate here
                // is the user's own message: about sixteen tokens.
                guard let primed = primeSystemPrefix(full: prompt, stats: &stats,
                                                     emit: emit) else { return }
                prompt = Array(prompt.dropFirst(primed))
            } else {
                guard let p = ChatTemplate.renderUserTurn(userText, tokenizer: tokenizer,
                                                          thinking: config.thinking,
                                                          effort: config.effort,
                                                          toolCount: runner.schemas.count) else {
                    emit(.failed("cannot render the user turn")); return
                }
                prompt = p
            }
        } catch {
            emit(.failed(String(describing: error))); return
        }

        var step = 0
        while true {
            if Int(nPast) + prompt.count >= Int(contextLimit) {
                // PLAN.md 2.4: a full context is the end of a session, not an
                // error. The caller offers a successor; the engine is never
                // asked to do something it cannot.
                emit(.contextFull(used: Int(nPast), limit: Int(contextLimit)))
                stats.contextUsed = Int(nPast)
                emit(.turnFinished(stats))
                return
            }

            guard let turn = generate(prompt: prompt, stats: &stats,
                                      cancelled: cancelled, emit: emit) else { return }
            // Only now. `started` means "this session has a system turn in its
            // KV cache", and rendering one is not evaluating one: a template
            // failure, or a context check that fires first, would otherwise
            // leave the session claiming a prefix it never got, and every later
            // turn would be a continuation of nothing.
            started = true

            if !turn.hasCall || turn.interrupted {
                if stats.hitBudget {
                    emit(.note(turn.inReasoning
                        ? "stopped at the \(config.maxTokensPerTurn)-token budget while still reasoning, so there is no answer to show"
                        : "stopped at the \(config.maxTokensPerTurn)-token budget"))
                }
                stats.contextUsed = Int(nPast)
                emit(.turnFinished(stats))
                return
            }

            // --- dispatch ---------------------------------------------------
            var result = ""
            switch ToolParser.parse(turn.text) {
            case .malformed(let why):
                result = "error: \(why). Re-issue the call in the required format."
                emit(.toolResult(name: "(malformed)", result: result))
            case .none:
                result = "error: no tool call was found."
                emit(.toolResult(name: "(none)", result: result))
            case .calls(let calls, _):
                for (i, c) in calls.enumerated() {
                    emit(.toolCall(c))
                    let r = runner.run(c)
                    emit(.toolResult(name: c.name, result: r))
                    if i > 0 { result += "\n" }
                    result += r
                    stats.toolCalls += 1
                }
            }

            step += 1
            if step >= config.maxToolSteps {
                emit(.note("stopped after \(step) tool calls"))
                stats.hitStepCap = true
                stats.contextUsed = Int(nPast)
                emit(.turnFinished(stats))
                return
            }

            guard let next = ChatTemplate.renderToolResult(result, tokenizer: tokenizer,
                                                           thinking: config.thinking,
                                                           effort: config.effort,
                                                           toolCount: runner.schemas.count) else {
                emit(.failed("cannot render the tool result")); return
            }
            prompt = next
        }
    }

    // MARK: One assistant turn

    private struct Turn {
        var text = ""
        var hasCall = false
        var inReasoning = true
        var interrupted = false
    }

    private func generate(prompt: [Int32], stats: inout TurnStats,
                          cancelled: @escaping @Sendable () -> Bool,
                          emit: @escaping (SessionEvent) -> Void) -> Turn? {
        var turn = Turn()
        stats.promptTokens += prompt.count

        let t0 = Date()
        var next: Int32 = 0
        let vocab = qwasar_vocab_size(engine)

        var sp = qwasar_sampling(temperature: config.temperature,
                                 top_k: config.topK,
                                 top_p: config.topP,
                                 min_p: config.minP,
                                 seed: config.seed)
        // Same seeding as qwasar-server: an explicit seed reproduces exactly,
        // 0 draws from the clock. The rng state lives for the turn.
        var rng: UInt64 = sp.seed != 0
            ? sp.seed
            : UInt64(bitPattern: Int64(Date().timeIntervalSince1970 * 1000))
              &* 6364136223846793005 &+ 1

        // The phase split (PLAN.md 7.5). Sampling is what the reasoning block
        // needs -- long greedy chains loop -- and speculation is what the
        // answer needs: tool-call markup is low-entropy, drafts well, and at
        // these settings sampling nearly always picks the argmax token there
        // anyway. `qwasar_session_verify`'s contract is greedy equivalence, so
        // the two cannot overlap; they alternate on the `</think>` boundary
        // instead. With no draft head there is no speedup to buy, so sampling
        // simply runs the whole turn.
        var greedySP = qwasar_sampling(temperature: 0, top_k: 0, top_p: 1,
                                       min_p: 0, seed: 0)
        var reasoning = true
        let hasMTP = qwasar_session_has_mtp(handle)
        func pick(_ logits: UnsafePointer<Float>) -> Int32 {
            hasMTP && !reasoning
                ? qwasar_sample(logits, vocab, &greedySP, &rng)
                : qwasar_sample(logits, vocab, &sp, &rng)
        }

        let ok: Bool = autoreleasepool {
            let (logits, err) = withErrorBuffer { buf, cap in
                prompt.withUnsafeBufferPointer { p in
                    qwasar_session_eval(handle, p.baseAddress, Int32(p.count), buf, cap)
                }
            }
            guard let logits else { emit(.failed("evaluation failed: \(err)")); return false }
            tokens.append(contentsOf: prompt)
            emit(.context(used: Int(nPast), limit: Int(contextLimit)))
            next = pick(logits)               // borrowed; consumed before anything else
            return true
        }
        guard ok else { return nil }
        stats.prefillSeconds += Date().timeIntervalSince(t0)

        let thinkClose = tokenizer.id(of: "</think>")
        let callOpen = tokenizer.id(of: "<tool_call>")
        var inCall = false
        var textAsm = UTF8Assembler()
        var thinkAsm = UTF8Assembler()
        // Tokens consumed since the assembler last produced anything, so the
        // count that travels with a delta is the number that produced it.
        var thinkPending = 0
        let tDecode = Date()
        var i = 0
        // The template writes "\n\n" after </think>, so the first delta of an
        // answer is blank lines -- two empty rows under the reasoning block,
        // every turn. Suppressed in the TRANSCRIPT only: `turn.text` still gets
        // the bytes verbatim, because the tool parser reads that and must see
        // what the model actually produced.
        var textStarted = false
        var thinkStarted = false

        /// Drops whitespace at the very start of a stream, once.
        ///
        /// Returns nil for a delta that is still entirely whitespace, so the
        /// caller holds its pending token count rather than attributing it to a
        /// delta that was never shown.
        func opening(_ s: String, _ started: inout Bool) -> String? {
            if started { return s }
            let t = s.drop(while: \.isWhitespace)
            if t.isEmpty { return nil }
            started = true
            return String(t)
        }

        /// One token, taken. Extracted because speculation commits several at a
        /// time and every one of them has to travel the same path -- the EOS
        /// check, the reasoning switch, the assemblers, the tool-call test. A
        /// drafted token that skipped any of these would be a token the
        /// transcript never showed or the parser never saw.
        enum Take { case go, stop }
        func take(_ tok: Int32) -> Take {
            if qwasar_is_eos(engine, tok) { stats.hitEOS = true; return .stop }

            i += 1
            stats.generatedTokens += 1
            if reasoning { stats.reasoningTokens += 1 }

            if tok == thinkClose {
                reasoning = false
            } else {
                if !reasoning && tok == callOpen {
                    inCall = true
                    // Immediately, not at the next report: the switch from prose
                    // to markup is the moment the screen would otherwise go
                    // quiet, and a second of nothing there reads as a stall.
                    emit(.toolCallProgress(name: nil, keys: [], tokens: i))
                }
                tokenizer.withBytes(of: tok) { buf, special in
                    if reasoning {
                        thinkPending += 1
                        if let s = thinkAsm.feed(buf),
                           let out = opening(s, &thinkStarted) {
                            emit(.reasoning(out, tokens: thinkPending))
                            thinkPending = 0
                        }
                    } else {
                        // The call itself is markup, not prose. It is still
                        // accumulated for the parser, but echoing it would bury
                        // any narration the model wrote first -- and it is told
                        // it may narrate before a call.
                        turn.text += String(decoding: buf, as: UTF8.self)
                        if !special && !inCall {
                            if let s = textAsm.feed(buf),
                               let out = opening(s, &textStarted) { emit(.text(out)) }
                        }
                    }
                }
            }

            if !reasoning && ToolParser.isComplete(turn.text) {
                turn.hasCall = true
                return .stop
            }
            return .go
        }

        /// Progress reporting, on the same cadence whether a step committed one
        /// token or five.
        ///
        /// A DELTA rather than `count % 64`, for two reasons. A speculative
        /// round advances the count by up to nine, so a modulo test can step
        /// straight over its own trigger and go silent for a hundred tokens.
        /// And 64 tokens is eleven seconds at ~6 tok/s -- long enough that a
        /// turn with no text to show reads as a stall. Eight is about a second
        /// and costs one event.
        var lastReported = 0
        var lastRateTokens = 0
        var lastRateTime = tDecode
        func report(force: Bool = false) {
            guard force || tokens.count - lastReported >= 8 else { return }
            lastReported = tokens.count
            emit(.context(used: Int(nPast), limit: Int(contextLimit)))
            let now = Date()
            let elapsed = now.timeIntervalSince(tDecode)
            if elapsed > 0 {
                // The window since the last report -- about a second, so it
                // tracks the current regime rather than the turn's history.
                let dt = now.timeIntervalSince(lastRateTime)
                let inst = dt > 0 ? Double(i - lastRateTokens) / dt
                                  : Double(i) / elapsed
                lastRateTokens = i
                lastRateTime = now
                emit(.rate(generated: i, tokensPerSecond: Double(i) / elapsed,
                           instantaneous: inst))
            }
            // While a call is being written there is nothing else on screen, so
            // this is the only sign the turn is alive.
            if inCall {
                let (name, keys) = ToolParser.partial(turn.text)
                emit(.toolCallProgress(name: name, keys: keys, tokens: i))
            }
        }

        // Speculation, when the engine was given a draft head. The head only
        // ever PROPOSES: `qwasar_session_verify` commits the longest correct
        // prefix and rewinds the rest, so the emitted sequence is identical to
        // decoding serially. That property is the whole point, and it is held by
        // tests/test_verify in the C tree rather than assumed here.
        //
        // That contract is GREEDY equivalence, so drafting runs only in the
        // greedy phase: the whole turn at temperature 0, otherwise from the
        // `</think>` boundary on -- checked per iteration, because take() is
        // what flips `reasoning`. Everything a round commits, including the
        // boundary token verify leaves undecided, is argmax, which is exactly
        // what pick() decodes in that phase.
        var blk = [Int32](repeating: 0, count: Int(QWASAR_MAX_DRAFT) + 1)
        var got = [Int32](repeating: 0, count: Int(QWASAR_MAX_DRAFT) + 1)

        while i < config.maxTokensPerTurn {
            if cancelled() { turn.interrupted = true; stats.interrupted = true; break }
            // The pre-turn check bounds the prompt; this bounds the generation.
            // Without it a long answer walks into the engine's own limit and
            // comes back as an evaluation failure, which is a true statement
            // and a useless one -- the session is full, not broken.
            if nPast >= contextLimit - 1 {
                emit(.contextFull(used: Int(nPast), limit: Int(contextLimit)))
                break
            }

            if take(next) == .stop { break }

            // --- speculative step -----------------------------------------
            if hasMTP && (config.temperature <= 0 || !reasoning) {
                // Depth 0 is a real answer, not an error: the head has been
                // wrong often enough here that a round costs more than it saves,
                // and the engine says so from measured acceptance.
                let want = qwasar_session_draft_depth(handle)
                if want > 0 {
                    blk[0] = next
                    let (nd, dErr): (Int32, String) = withErrorBuffer { buf, cap in
                        blk.withUnsafeMutableBufferPointer { b in
                            qwasar_session_draft(handle, next, b.baseAddress! + 1, want, buf, cap)
                        }
                    }
                    guard nd >= 0 else { emit(.failed("drafting failed: \(dErr)")); return nil }

                    let (nc, vErr): (Int32, String) = withErrorBuffer { buf, cap in
                        blk.withUnsafeBufferPointer { bp in
                            got.withUnsafeMutableBufferPointer { gp in
                                qwasar_session_verify(handle, bp.baseAddress, nd + 1,
                                                      gp.baseAddress, buf, cap)
                            }
                        }
                    }
                    guard nc > 0 else { emit(.failed("verify failed: \(vErr)")); return nil }

                    // What the session actually committed: the token that was
                    // already decided, then every draft it accepted -- which are
                    // exactly the outputs bar the last. The last is what this
                    // round leaves undecided, taking `next`'s place.
                    tokens.append(next)
                    if nc > 1 { tokens.append(contentsOf: got[0..<Int(nc) - 1]) }
                    stats.specRounds += 1
                    stats.specCommitted += Int(nc)
                    report()

                    var stopped = false
                    for t in 0..<(Int(nc) - 1) {
                        if i >= config.maxTokensPerTurn { stopped = true; break }
                        if take(got[t]) == .stop { stopped = true; break }
                    }
                    if stopped { break }
                    next = got[Int(nc) - 1]
                    continue
                }
            }

            let stepOK: Bool = autoreleasepool {
                let (logits, err) = withErrorBuffer { buf, cap in
                    withUnsafePointer(to: next) { p in
                        qwasar_session_eval(handle, p, 1, buf, cap)
                    }
                }
                guard let logits else { emit(.failed("evaluation failed: \(err)")); return false }
                tokens.append(next)
                // Often enough that the meter moves visibly at ~6 tok/s, rare
                // enough to cost nothing.
                report()
                // pick(), not sp: this step also serves the answer phase when
                // draft_depth backs off to 0, and that phase is greedy.
                next = pick(logits)
                return true
            }
            guard stepOK else { return nil }
        }

        stats.decodeSeconds += Date().timeIntervalSince(tDecode)
        if i >= config.maxTokensPerTurn { stats.hitBudget = true }
        turn.inReasoning = reasoning
        stats.stoppedInReasoning = reasoning

        if let s = thinkAsm.flush() { emit(.reasoning(s, tokens: thinkPending)) }
        if let s = textAsm.flush() { emit(.text(s)) }
        emit(.context(used: Int(nPast), limit: Int(contextLimit)))
        return turn
    }
}
