// EngineHost.swift -- the one thread that is allowed to touch the engine.
//
// PLAN.md 2.1: qwasar_metal.m keeps g_device, g_queue, g_library and a lazily
// populated g_pipelines dictionary in file statics with no lock, and
// qwasar_server.c says outright that requests are served one at a time. So
// every qwasar_* call in this process must be serialised.
//
// It is a DispatchQueue rather than a Swift actor on purpose. An actor would
// serialise correctly, but a decode step is a multi-second blocking sequence of
// C calls, and blocking a cooperative-pool thread starves the concurrency
// runtime. The engine gets its own thread; Swift concurrency reaches it through
// continuations and AsyncStream.
//
// Thread affinity is not required -- Metal does not care which thread encodes,
// and command buffers here are per-call. Serialisation is. The assertion in
// `assertOnEngineQueue` is what keeps that true as the app grows.

import Foundation
import CQwasar

public enum EngineError: Error, CustomStringConvertible {
    case load(String)
    case eval(String)
    case template(String)
    case noModel
    case notLoaded

    public var description: String {
        switch self {
        case .load(let m):     return "cannot load the model: \(m)"
        case .eval(let m):     return "evaluation failed: \(m)"
        case .template(let m): return "cannot render the prompt: \(m)"
        case .noModel:         return "no model directory selected"
        case .notLoaded:       return "the engine is not loaded"
        }
    }
}

/// What a loaded engine can tell us about itself.
public struct EngineInfo: Sendable {
    public let modelPath: String
    public let vocabSize: Int32
    public let layers: Int32
    public let contextSize: Int32
    public let loadSeconds: Double
    public let footprintBytes: UInt64
    /// Whether a draft head was loaded and the engine will speculate.
    ///
    /// Reported from the engine rather than from what we asked for: a head that
    /// failed to bind leaves a working engine that simply decodes serially, and
    /// a UI that claimed otherwise would be lying about why it is slow.
    public let mtpActive: Bool
}

public final class EngineHost: @unchecked Sendable {
    private let queue = DispatchQueue(label: "dev.crucible.engine", qos: .userInitiated)
    private static let key = DispatchSpecificKey<Bool>()

    private var engine: OpaquePointer?
    private var tokenizer: Tokenizer?
    /// Why speculation is off, when it was asked for and did not happen.
    public private(set) var mtpDropped: String?
    private var sessions: [UUID: LiveSession] = [:]
    private(set) public var info: EngineInfo?

    public init() {
        queue.setSpecific(key: Self.key, value: true)
    }

    /// Trips in debug builds if engine state is touched from anywhere else.
    /// PLAN.md 13: one queue, no exceptions.
    @inline(__always)
    static func assertOnEngineQueue() {
        assert(DispatchQueue.getSpecific(key: key) == true,
               "qwasar_* called off the engine queue")
    }

    // MARK: Lifecycle

    /// Loads config and binds weights. Roughly 16 GB of mmap; several seconds.
    ///
    /// `contextSize` comes from the caller because PLAN.md 2.3 makes it a
    /// property of the machine's profile, not of the engine: the KV cache is
    /// allocated eagerly at `qwasar_session_new` from whatever is resolved here.
    public func load(modelPath: String, contextSize: Int32,
                     mtpPath: String? = nil) async throws -> EngineInfo {
        try await withCheckedThrowingContinuation { cont in
            queue.async {
                do { cont.resume(returning: try self.loadSync(modelPath, contextSize, mtpPath)) }
                catch { cont.resume(throwing: error) }
            }
        }
    }

    private func loadSync(_ modelPath: String, _ contextSize: Int32,
                          _ mtpPath: String?) throws -> EngineInfo {
        Self.assertOnEngineQueue()
        if engine != nil { throw EngineError.load("already loaded") }

        let t0 = Date()
        let arena = CStringArena()
        var opts = qwasar_options()
        opts.model_path    = arena.dup(modelPath)
        // The MTP draft head, when the user has granted one. It proposes tokens
        // and never decides one, so this cannot change what the model emits --
        // only how many forward passes it takes to emit it (qwasar.h).
        opts.mtp_path      = mtpPath.map { arena.dup($0) } ?? nil
        opts.context_size  = contextSize
        opts.prefill_chunk = 0              // engine default
        opts.verbose       = false

        var (e, err) = withErrorBuffer { buf, cap in
            withUnsafePointer(to: &opts) { qwasar_engine_load($0, buf, cap) }
        }

        // A draft head that will not bind fails the WHOLE load -- qwasar.c goes
        // to `fail` and returns NULL -- so without this a bad or mismatched head
        // presents as "cannot load the model", which is both alarming and
        // wrong. Speculation is an optimisation; losing it must never cost the
        // application. Retry once without it, and say what happened.
        var mtpActive = mtpPath != nil
        if e == nil, mtpPath != nil {
            mtpDropped = "the draft head did not load (\(err)); decoding serially"
            mtpActive = false
            opts.mtp_path = nil
            (e, err) = withErrorBuffer { buf, cap in
                withUnsafePointer(to: &opts) { qwasar_engine_load($0, buf, cap) }
            }
        }
        guard let e else { throw EngineError.load(err) }
        engine = e

        do { tokenizer = try Tokenizer(modelPath: modelPath) }
        catch {
            qwasar_engine_free(e); engine = nil
            throw error
        }

        let i = EngineInfo(modelPath: modelPath,
                           vocabSize: qwasar_vocab_size(e),
                           layers: qwasar_n_layers(e),
                           contextSize: contextSize,
                           loadSeconds: Date().timeIntervalSince(t0),
                           footprintBytes: Diagnostics.physFootprint(),
                           mtpActive: mtpActive)
        info = i
        return i
    }

    public func unload() {
        queue.sync {
            Self.assertOnEngineQueue()
            sessions.removeAll()
            tokenizer = nil
            if let e = engine { qwasar_engine_free(e); engine = nil }
            info = nil
        }
    }

    // MARK: Sessions

    /// Opens a session, replaying `history` into it.
    ///
    /// One `qwasar_session` is allocated here, and PLAN.md 2.3 is why that is a
    /// heavyweight act: the KV cache is sized from the profile context at
    /// creation, not grown on demand, so an empty conversation costs exactly
    /// what a full one costs.
    ///
    /// Returns a stream rather than completing silently because reopening a
    /// session with history is not instant -- it is a checkpoint read, or a
    /// re-prefill of everything said so far. The caller shows that happening.
    public func openSession(id: UUID, runner: ToolExecuting, config: SessionConfig,
                            history: [Int32] = []) -> AsyncStream<SessionEvent> {
        AsyncStream { continuation in
            queue.async {
                Self.assertOnEngineQueue()
                guard let e = self.engine, let tok = self.tokenizer else {
                    continuation.yield(.failed(String(describing: EngineError.notLoaded)))
                    continuation.finish(); return
                }
                if self.sessions[id] != nil { continuation.finish(); return }

                let (sp, err) = withErrorBuffer { buf, cap in qwasar_session_new(e, buf, cap) }
                guard let handle = sp else {
                    continuation.yield(.failed("cannot open a session: \(err)"))
                    continuation.finish(); return
                }
                let s = LiveSession(handle: handle, tokenizer: tok, engine: e,
                                    contextLimit: self.info?.contextSize ?? 0,
                                    config: config, runner: runner)
                // Registered before the replay so a failure leaves a session
                // that can be closed rather than a leaked handle.
                self.sessions[id] = s

                if !s.restore(history: history, emit: { continuation.yield($0) }) {
                    self.sessions[id] = nil
                }
                continuation.finish()
            }
        }
    }

    /// Frees a session's state, checkpointing it on the way out.
    ///
    /// The tokens survive on the caller's record either way, so a reopen is
    /// always possible; the checkpoint is what makes it a read rather than a
    /// re-prefill of the whole conversation -- ten minutes, for the 19084-token
    /// session found in a real cache.
    ///
    /// Once, here, rather than after every turn. Per-turn checkpointing wrote a
    /// fresh snapshot each time, every one a strict superset of the last, and
    /// filled 5.30 GiB of the 6 GiB budget with nine files that were read once
    /// between them. A session earns exactly one checkpoint, taken when it has
    /// stopped growing; LRU then keeps whichever are actually read, which is
    /// what protects the system prefix -- the one entry hit constantly.
    public func closeSession(_ id: UUID) {
        queue.async {
            Self.assertOnEngineQueue()
            self.sessions[id]?.checkpoint()
            self.sessions[id] = nil
        }
    }

    /// `closeSession`, but waits for the checkpoint to be on disk.
    ///
    /// The fire-and-forget version is right everywhere except one place: on the
    /// way out of the application, where the process may be gone before the
    /// queue gets to it. Nothing the USER wrote is at risk either way -- the
    /// transcript and the token history are already persisted -- but a session
    /// that quits without its checkpoint re-prefills its whole history on the
    /// next open, and for a long session that is minutes.
    public func closeAndCheckpoint(_ id: UUID) async {
        await withCheckedContinuation { (c: CheckedContinuation<Void, Never>) in
            queue.async {
                Self.assertOnEngineQueue()
                self.sessions[id]?.checkpoint()
                self.sessions[id] = nil
                c.resume()
            }
        }
    }

    public func isOpen(_ id: UUID) async -> Bool {
        await withCheckedContinuation { cont in
            queue.async { cont.resume(returning: self.sessions[id] != nil) }
        }
    }

    /// Runs one user turn to completion, streaming events as they happen.
    public func send(_ id: UUID, text: String,
                     cancelled: @escaping @Sendable () -> Bool) -> AsyncStream<SessionEvent> {
        AsyncStream { continuation in
            queue.async {
                Self.assertOnEngineQueue()
                guard let s = self.sessions[id] else {
                    continuation.yield(.failed("session is not open"))
                    continuation.finish()
                    return
                }
                s.runTurn(userText: text, cancelled: cancelled,
                          emit: { continuation.yield($0) })
                continuation.finish()
            }
        }
    }

    // MARK: The system-prefix checkpoint, proven

    /// What one run of `probeSystemPrefix` found.
    public struct PrefixProbe: Sendable {
        public var prefixTokens = 0
        public var coldSeconds = 0.0
        /// Tokens the cold run actually had to prefill. Equal to
        /// `prefixTokens` on a genuinely empty cache.
        public var coldEvaluated = 0
        public var warmSeconds = 0.0
        /// Tokens the second run got from the checkpoint rather than the GPU.
        public var warmRestored = 0
        public var error: String?

        public var hit: Bool { error == nil && prefixTokens > 0 && warmRestored == prefixTokens }
        public var speedup: Double { warmSeconds > 0 ? coldSeconds / warmSeconds : 0 }
    }

    /// Primes the system prefix twice, in two separate sessions, and reports
    /// whether the second one came out of the checkpoint cache.
    ///
    /// This exists because the failure it guards against is silent. The
    /// optimisation is invisible when it works and invisible when it does not:
    /// a session whose checkpoint is never written, or never matched, behaves
    /// exactly like one whose checkpoint is read every time, only slower. That
    /// is how per-turn checkpointing came to fill 5.30 GiB of a 6 GiB budget
    /// for one hit without anyone noticing. So it is asserted rather than
    /// assumed, against the real engine.
    ///
    /// Generates nothing -- it is a prefill and a file read, no sampling -- but
    /// the cold half is a real ~2500-token prefill and costs about eighty
    /// seconds. It also leaves one ~280 MB checkpoint behind, keyed on whatever
    /// system prompt it was handed; LRU reclaims it.
    public func probeSystemPrefix(config: SessionConfig,
                                  runner: ToolExecuting) async -> PrefixProbe {
        await withCheckedContinuation { cont in
            self.queue.async {
                Self.assertOnEngineQueue()
                var r = PrefixProbe()
                guard let e = self.engine, let tok = self.tokenizer else {
                    r.error = String(describing: EngineError.notLoaded)
                    cont.resume(returning: r); return
                }

                // One turn per pass, in its own session, so the second starts
                // from n_past == 0 the way a new session really does.
                func pass(_ label: String) -> (primed: Int, seconds: Double, evaluated: Int)? {
                    let (sp, err) = withErrorBuffer { b, c in qwasar_session_new(e, b, c) }
                    guard let h = sp else { r.error = "\(label): \(err)"; return nil }
                    // LiveSession owns the handle and frees it when it goes.
                    let s = LiveSession(handle: h, tokenizer: tok, engine: e,
                                        contextLimit: self.info?.contextSize ?? 0,
                                        config: config, runner: runner)
                    guard let full = try? ChatTemplate.render(
                            [.system(config.system), .user("probe")], tokenizer: tok,
                            thinking: config.thinking, effort: config.effort,
                            tools: runner.schemas) else {
                        r.error = "\(label): cannot render the first turn"; return nil
                    }
                    var stats = TurnStats()
                    let t0 = Date()
                    guard let primed = s.primeSystemPrefix(full: full, stats: &stats,
                                                           emit: { _ in }) else {
                        r.error = "\(label): priming failed"; return nil
                    }
                    return (primed, Date().timeIntervalSince(t0), stats.promptTokens)
                }

                guard let cold = pass("cold") else { cont.resume(returning: r); return }
                r.prefixTokens = cold.primed
                r.coldSeconds  = cold.seconds
                r.coldEvaluated = cold.evaluated

                guard let warm = pass("warm") else { cont.resume(returning: r); return }
                r.warmSeconds  = warm.seconds
                // Whatever it did not have to prefill, it read.
                r.warmRestored = warm.primed - warm.evaluated
                cont.resume(returning: r)
            }
        }
    }

    /// The session's full token history, for persistence and for a later
    /// restore (PLAN.md 4.1).
    public func tokens(of id: UUID) async -> [Int32] {
        await withCheckedContinuation { cont in
            queue.async { cont.resume(returning: self.sessions[id]?.tokens ?? []) }
        }
    }
}
