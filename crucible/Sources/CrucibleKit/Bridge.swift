// Bridge.swift -- the four sharp edges of calling qwasar.h from Swift.
//
// PLAN.md 3.4 names them; this file is where each one is handled exactly once,
// so no call site has to remember:
//
//   1. Logits are borrowed, valid only until the next eval on that session.
//   2. Several qwasar_* functions return malloc'd arrays the caller owes free().
//   3. The progress callback is a C function pointer and cannot capture.
//   4. Token bytes are raw UTF-8 and routinely split a codepoint in half.
//
// Everything here is deliberately small and free of engine knowledge.

import Foundation
import CQwasar

// MARK: - Error buffers

/// Calls a qwasar_* function that reports failure through a `char *err` buffer.
///
/// The engine's convention is a caller-supplied buffer plus a capacity, filled
/// only on failure. Wrapping it here means no call site allocates its own
/// buffer, forgets the capacity, or reads the buffer when the call succeeded
/// (where it holds whatever was there before).
func withErrorBuffer<R>(_ body: (UnsafeMutablePointer<CChar>, Int) -> R) -> (result: R, error: String) {
    var buf = [CChar](repeating: 0, count: 512)
    let r = buf.withUnsafeMutableBufferPointer { p in
        body(p.baseAddress!, p.count)
    }
    let n = buf.firstIndex(of: 0) ?? buf.count
    return (r, String(decoding: buf[0..<n].map { UInt8(bitPattern: $0) }, as: UTF8.self))
}

// MARK: - Owned arrays

/// Copies a malloc'd `int32_t *` returned by the engine into a Swift array and
/// frees it.
///
/// `qwasar_encode`, `qwasar_apply_chat_template`, `qwasar_render_tool_result`
/// and `qwasar_render_user_turn` all return memory the caller owes free(). This
/// is the only place that debt is settled, and no raw pointer escapes it.
func takeOwnedTokens(_ p: UnsafeMutablePointer<Int32>?, count: Int32) -> [Int32]? {
    guard let p else { return nil }
    defer { free(p) }
    guard count > 0 else { return [] }
    return Array(UnsafeBufferPointer(start: p, count: Int(count)))
}

// MARK: - C strings for struct fields

/// Holds strdup'd copies alive for the duration of a call that takes
/// `const char *` struct fields, then frees them.
///
/// qwasar_message is four string pointers; building one from Swift Strings
/// without this leaks, dangles, or both.
final class CStringArena {
    private var owned: [UnsafeMutablePointer<CChar>] = []

    func dup(_ s: String?) -> UnsafePointer<CChar>? {
        guard let s else { return nil }
        guard let p = strdup(s) else { return nil }
        owned.append(p)
        return UnsafePointer(p)
    }

    deinit { for p in owned { free(p) } }
}

// MARK: - UTF-8 reassembly

/// Reassembles whole characters from token bytes.
///
/// A single token is frequently half a codepoint -- routinely so for CJK and
/// emoji, and for any multi-byte character the BPE merges split. The C agent
/// writes bytes straight to a terminal and gets away with it; SwiftUI does not,
/// and the failure mode is U+FFFD appearing mid-word in the transcript.
///
/// Feed every token's bytes; take whatever completes. Anything incomplete is
/// held until the next token supplies the rest.
public struct UTF8Assembler {
    private var pending: [UInt8] = []

    public init() {}

    /// Appends raw bytes and returns whatever now forms complete characters.
    public mutating func feed(_ bytes: UnsafeBufferPointer<UInt8>) -> String? {
        pending.append(contentsOf: bytes)
        let n = Self.completeCount(pending)
        guard n > 0 else { return nil }
        let out = String(decoding: pending[0..<n], as: UTF8.self)
        pending.removeFirst(n)
        return out
    }

    /// Anything still held at the end of a turn, decoded as best it can be.
    /// A non-empty result here means the model stopped mid-character, which is
    /// worth showing rather than swallowing.
    public mutating func flush() -> String? {
        guard !pending.isEmpty else { return nil }
        let out = String(decoding: pending, as: UTF8.self)
        pending.removeAll()
        return out
    }

    /// How many leading bytes form complete UTF-8 sequences.
    ///
    /// Walks back from the end over continuation bytes to find the last start
    /// byte, and asks whether its sequence finished. An invalid start byte is
    /// released rather than held forever -- a malformed stream must not wedge
    /// the assembler.
    private static func completeCount(_ b: [UInt8]) -> Int {
        guard !b.isEmpty else { return 0 }
        var i = b.count - 1
        let floor = max(0, b.count - 4)
        while i > floor && (b[i] & 0xC0) == 0x80 { i -= 1 }
        let start = b[i]
        let need: Int
        switch start {
        case 0x00...0x7F: need = 1
        case 0xC0...0xDF: need = 2
        case 0xE0...0xEF: need = 3
        case 0xF0...0xF7: need = 4
        default:          need = 1   // continuation or invalid: do not hold it
        }
        return (i + need <= b.count) ? b.count : i
    }
}

// MARK: - Progress

/// The far side of `qwasar_session_set_progress`.
///
/// The C callback is a bare function pointer with a `void *`, so it cannot
/// capture. This box is what the `void *` points at. It is passed unretained,
/// so whoever sets the callback must outlive the session -- see Session.swift,
/// which owns its box and clears the callback before freeing the session.
///
/// The callback fires on the engine thread, during prefill, between chunks.
final class ProgressBox {
    private let sink: (Int, Int) -> Void

    /// Rebases the engine's per-CALL progress onto a larger span.
    ///
    /// `qwasar_session_eval(s, p, n)` reports `(done, n)` for the call it is
    /// inside -- `n` is the size of THAT call, not of whatever the caller
    /// thinks it is doing. So a 2315-token prefill split into nine 256-token
    /// calls reports "0/256" nine times over, and the bar sits at zero through
    /// the longest, quietest part of a turn while looking hung.
    ///
    /// A caller driving a sequence of evals that are really one prefill sets
    /// this to the whole of it, and the reports come out rebased. A caller
    /// making a single eval leaves it nil and the engine's own totals are
    /// already right.
    var span: (base: Int, total: Int)?

    init(_ sink: @escaping (Int, Int) -> Void) { self.sink = sink }

    func report(done: Int, total: Int) {
        if let s = span { sink(s.base + done, s.total) } else { sink(done, total) }
    }
}

let progressTrampoline: qwasar_progress_fn = { ud, done, total in
    guard let ud else { return }
    Unmanaged<ProgressBox>.fromOpaque(ud).takeUnretainedValue()
        .report(done: Int(done), total: Int(total))
}
