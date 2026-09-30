// Tools.swift -- the calls a model makes, and the read-only fallback.
//
// A sandboxed session whose guest cannot start still works, against the real
// tree, but only to look: Read, Glob and Grep, confined by PathGuard to the
// project directory, and nothing that writes or runs.  The session header
// says so, and so does the model's environment description.

import Foundation

public struct ToolCall: Sendable, Equatable {
    public var name: String
    public var arguments: [String: String]

    public init(name: String, arguments: [String: String]) {
        self.name = name
        self.arguments = arguments
    }

    public func argument(_ key: String) -> String? { arguments[key] }
}

public struct ToolRunner: ToolExecuting {
    let kit: ToolKit

    public init(root: URL, resultCap: Int = ToolKit.resultCap(flashNext: true)) {
        kit = ToolKit(backend: HostBackend(root: root, environment: ["PATH": "/usr/bin:/bin:/usr/sbin:/sbin"],
                                           confined: true),
                      resultCap: resultCap)
    }

    public var schemas: [String] { ToolSurface.readOnlySchemas }

    public var environmentDescription: String {
        """
        # Where you are working

        You are working directly against the user's own files, with READ-ONLY access: Read, Glob and Grep, inside the project directory. You cannot write, edit, or run commands. Investigate and explain, and describe any change you would make rather than attempting it.
        """
    }

    public func run(_ call: ToolCall) -> String {
        guard ToolSurface.readOnlyNames.contains(call.name) else {
            return "error: \(call.name) is not available: this session is read-only. Available: "
                 + ToolSurface.readOnlyNames.sorted().joined(separator: ", ")
        }
        return kit.run(call)
    }
}
