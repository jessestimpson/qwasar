// ToolParserSuite.swift -- what can be told about a call still being written.
//
// `ToolParser.partial` runs during generation, on text that is by definition
// incomplete, and drives what the user is told is happening. The failure that
// matters is not "no name yet" -- that is honest -- but a name that is WRONG,
// because the row would then confidently narrate the wrong tool.

import Foundation
import CrucibleKit

enum ToolParserSuite {
    static func run() -> Int {
        var f = 0

        let full = """
        <tool_call>
        <function=write>
        <parameter=path>
        src/main.c
        </parameter>
        <parameter=content>
        int main(void){ return 0; }
        </parameter>
        </function>
        </tool_call>
        """

        let (name, keys) = ToolParser.partial(full)
        f += check(name == "write", "the function name is read (got \(name ?? "nil"))")
        f += check(keys == ["path", "content"],
                   "parameter keys, in order (got \(keys))")

        // Nothing yet: honest, not wrong.
        let (n0, k0) = ToolParser.partial("<tool_call>\n<func")
        f += check(n0 == nil && k0.isEmpty, "an unfinished tag yields nothing")
        let (n1, _) = ToolParser.partial("here is some prose with no call at all")
        f += check(n1 == nil, "prose yields no name")

        // The streaming property. Every prefix must give either nothing or the
        // right answer -- never a different name, and never a key that is not
        // a real key of this call.
        var wrongName = 0
        var wrongKey = 0
        var firstNameAt: Int?
        let chars = Array(full)
        for k in 1...chars.count {
            let (nm, ks) = ToolParser.partial(String(chars[0..<k]))
            if let nm {
                if nm != "write" { wrongName += 1 }
                if firstNameAt == nil { firstNameAt = k }
            }
            for key in ks where !["path", "content"].contains(key) { wrongKey += 1 }
            if ks.count > 2 { wrongKey += 1 }
        }
        f += check(wrongName == 0, "no prefix ever reports the wrong name (\(wrongName) did)")
        f += check(wrongKey == 0, "no prefix ever invents a key (\(wrongKey) did)")
        // It has to arrive early enough to be worth showing: the name sits in
        // the first line, well inside the first few tokens.
        f += check((firstNameAt ?? .max) < 40,
                   "the name is known within \(firstNameAt ?? -1) characters")

        // A second call in the same turn must not report the first one's name.
        let two = full + "\n<tool_call>\n<function=bash>\n<parameter=cmd>\n"
        let (n2, k2) = ToolParser.partial(two)
        f += check(n2 == "bash", "a later call wins over an earlier one (got \(n2 ?? "nil"))")
        f += check(k2 == ["cmd"], "and carries only its own keys (got \(k2))")

        // The complete-call detector still agrees with itself.
        f += check(ToolParser.isComplete(full), "a full call reads as complete")
        f += check(!ToolParser.isComplete(String(full.dropLast(20))),
                   "a truncated call does not")

        return f
    }

    private static func check(_ ok: Bool, _ what: String) -> Int {
        print(ok ? "  ok   \(what)" : "  FAIL \(what)")
        return ok ? 0 : 1
    }
}
