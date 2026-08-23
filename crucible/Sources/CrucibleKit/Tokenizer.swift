// Tokenizer.swift -- the tokenizer, on its own.
//
// Deliberately independent of the engine: qwasar_tokenizer_load takes a model
// path, not a qwasar_engine, so anything that only needs to render or decode
// text costs a JSON parse rather than 16 GB of mmap. The golden-vector test
// (Tests/) depends on that, and so does any future prompt inspection in the UI.
//
// Not tied to the engine queue either. The tokenizer is read-only after load
// and touches no Metal state, so it is the one part of the C API that does not
// have to be serialised with everything else.

import Foundation
import CQwasar

public final class Tokenizer: @unchecked Sendable {
    let handle: OpaquePointer

    public init(modelPath: String) throws {
        let (h, err) = withErrorBuffer { buf, cap in
            qwasar_tokenizer_load(modelPath, buf, cap)
        }
        guard let h else { throw EngineError.load("tokenizer: \(err)") }
        handle = h
    }

    deinit { qwasar_tokenizer_free(handle) }

    public var size: Int32 { qwasar_tokenizer_size(handle) }

    /// Id of a control token by its literal spelling, e.g. "<|im_end|>", or -1.
    public func id(of literal: String) -> Int32 { qwasar_token_id(handle, literal) }

    /// Raw UTF-8 for one token. `special` reports control tokens, which callers
    /// usually suppress from display. The bytes are the engine's own storage --
    /// valid for the life of the tokenizer, not copied.
    public func withBytes<R>(of id: Int32, _ body: (UnsafeBufferPointer<UInt8>, Bool) -> R) -> R? {
        var len = 0
        var special = false
        guard let p = qwasar_token_bytes(handle, id, &len, &special), len > 0 else { return nil }
        return p.withMemoryRebound(to: UInt8.self, capacity: len) {
            body(UnsafeBufferPointer(start: $0, count: len), special)
        }
    }
}
