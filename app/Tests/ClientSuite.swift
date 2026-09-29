// ClientSuite.swift -- the Session API's stream, decoded without a socket.
//
// SSEParser and APIEvent.decode are what stand between the server's bytes and
// the window's events; both are pure, so they are pinned here with the
// framing the server actually writes (API.md §5), including the two things a
// naive parser gets wrong: data split across lines, and events it does not
// know, which must be ignored rather than end the stream.

import Foundation
import QwasarKit

enum ClientSuite {
    static func run() -> Int {
        var f = 0
        var p = SSEParser()
        let lines = [
            "id: 1.1", "event: resume",
            "data: {\"from\": \"checkpoint\", \"restored\": 2214, \"prefill\": 9, \"prefix_cached\": true}",
            "",
            "id: 1.2", "event: text", "data: {\"text\":", "data: \"two lines\"}", "",
            ": a comment line", "",
            "id: 1.3", "event: brand_new_thing", "data: {\"x\": 1}", "",
            "id: 1.4", "event: tool_call",
            "data: {\"id\": \"c_1_1\", \"name\": \"write\", \"arguments\": {\"path\": \"a.c\", \"count\": 3, \"flag\": true}}",
            "",
            "id: 1.5", "event: done",
            "data: {\"stop\": \"tool_calls\", \"usage\": {\"prompt\": 9, \"generated\": 19, \"reasoning\": 5}, "
            + "\"timing\": {\"prefill_seconds\": 0.9, \"decode_seconds\": 0.3, \"first_token_seconds\": 1.0}, "
            + "\"speculation\": {\"rounds\": 0, \"committed\": 0}, \"context\": {\"used\": 2242, \"limit\": 262144}, "
            + "\"warmth\": {\"state\": \"live\", \"covered\": 2242}}",
            "",
        ]
        var events: [(String, APIEvent)] = []
        for l in lines {
            if let (id, name, data) = p.feed(line: l) { events.append((id, APIEvent.decode(name: name, data: data))) }
        }
        f += TestMain.check(events.count == 5, "five events out of the framing (\(events.count))")
        f += TestMain.check(events.map(\.0) == ["1.1", "1.2", "1.3", "1.4", "1.5"], "ids carried through")
        f += TestMain.check(events[0].1 == .resume(from: "checkpoint", restored: 2214, prefill: 9, prefixCached: true),
                            "resume decodes, prefix_cached included")
        f += TestMain.check(events[1].1 == .text("two lines"), "data lines join (with a newline, between JSON tokens)")
        f += TestMain.check(events[2].1 == .other("brand_new_thing"), "an unknown event is passed as other, not dropped")
        if case .toolCall(let id, let name, let args) = events[3].1 {
            f += TestMain.check(id == "c_1_1" && name == "write", "tool_call id and name")
            f += TestMain.check(args["path"] == "a.c", "a string argument is itself")
            f += TestMain.check(args["count"] == "3" && args["flag"] == "true", "JSON-typed arguments become their text")
        } else {
            f += TestMain.check(false, "tool_call decodes")
        }
        if case .done(let stop, let prompt, let gen, let reas, let pf, let dec, _, _, _, let used, let limit, let warmth) = events[4].1 {
            f += TestMain.check(stop == "tool_calls" && prompt == 9 && gen == 19 && reas == 5, "done usage")
            f += TestMain.check(pf == 0.9 && dec == 0.3 && used == 2242 && limit == 262144 && warmth == "live", "done timing, context, warmth")
        } else {
            f += TestMain.check(false, "done decodes")
        }
        // A blank line with nothing pending is not an event.
        var q = SSEParser()
        f += TestMain.check(q.feed(line: "") == nil && q.feed(line: "event: x") == nil && q.feed(line: "") == nil,
                            "no data, no event")
        return f
    }
}
