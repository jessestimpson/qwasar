// Model.swift -- projects, sessions, transcripts.
//
// PLAN.md 4.1. Two fields carry more weight than they look:
//
//   SessionRecord.tokens      the FULL evaluated history. It is the truth; the
//                             transcript is for humans. At 4 bytes a token a
//                             90K session is 360 KB, so keeping all of it is
//                             free next to the state it describes, and it is
//                             what a restore is fed.
//   SessionRecord.contextSize resolved once at creation from the profile and
//                             never changed -- the KV cache was sized from it
//                             and cannot grow. A session created on a large
//                             machine and reopened on a small one has to be
//                             detected, not silently misread.

import Foundation

public struct Project: Codable, Identifiable, Sendable, Hashable {
    public var id: UUID
    public var name: String
    /// Security-scoped bookmark. Under App Sandbox this is the only way a
    /// user-chosen folder survives a relaunch (PLAN.md 4.2).
    public var rootBookmark: Data
    /// Resolved at launch; not persisted, because a path is not a grant.
    public var resolvedRoot: URL?
    public var systemPrompt: String
    /// The effort new sessions in this project start at.
    ///
    /// Optional so that a record written before this existed still decodes;
    /// `effort` resolves the absence to the model's own default.
    public var defaultEffort: ReasoningEffort?


    // The application's default is MEDIUM, not the engine's xhigh.
    //
    // Measured reason: at ~6 tok/s reasoning is the dominant cost of a turn --
    // gate (a) spent 4151 tokens of it on one task (PLAN.md 12, M4) -- and the
    // model reasons at length whatever it is asked. xhigh remains one menu item
    // away for work that earns it.
    //
    // Note this is a PRODUCT choice and lives here, not in ChatTemplate, whose
    // defaults deliberately mirror the C API's documented ones so the golden
    // vectors keep comparing like with like.
    public var effort: ReasoningEffort { defaultEffort ?? .medium }

    public init(id: UUID = UUID(), name: String, rootBookmark: Data,
                resolvedRoot: URL? = nil, systemPrompt: String = Project.defaultSystem,
                defaultEffort: ReasoningEffort? = nil) {
        self.id = id
        self.name = name
        self.rootBookmark = rootBookmark
        self.resolvedRoot = resolvedRoot
        self.systemPrompt = systemPrompt
        self.defaultEffort = defaultEffort
    }

    enum CodingKeys: String, CodingKey { case id, name, rootBookmark, systemPrompt, defaultEffort }

    /// The project's own instructions. NOT a description of the environment --
    /// that is the executor's to state (see `ToolExecuting.environmentDescription`),
    /// because it is a fact about what is running rather than a preference.
    public static let defaultSystem = """
        Investigate before answering: read the code rather than guessing at it, \
        and quote what you found. Prefer a small, verifiable change to a large \
        speculative one, and say plainly when something is not what it looked \
        like.
        """

    /// What the default said until M4, when the tools had been writable for two
    /// milestones and this had not been updated. Recognised so that a project
    /// which never customised its prompt is migrated rather than left telling
    /// the model it cannot edit anything.
    public static let supersededSystems: Set<String> = [
        """
        You are a coding assistant. You can read files, list directories, and \
        search with regular expressions, all scoped to the session's working \
        directory. You cannot write, edit, or run commands. Inspect the code \
        before answering rather than guessing, and quote what you found.
        """
    ]
}

public enum SessionState: String, Codable, Sendable {
    case live, closed, archived
}

public struct SessionRecord: Codable, Identifiable, Sendable {
    public var id: UUID
    public var projectID: UUID
    public var title: String
    /// Relative to the project root. "" is the root itself. A session cannot
    /// escape its project.
    public var workingSubpath: String
    public var createdAt: Date
    public var state: SessionState
    public var contextSize: Int32
    public var tokenCount: Int
    /// Compaction chain (PLAN.md 2.4). A successor inherits the sandbox, not
    /// the context.
    public var ancestorID: UUID?
    public var successorID: UUID?
    /// Fixed for the life of the session once it has evaluated anything.
    ///
    /// Effort rewrites the SYSTEM turn (the goldens measure three different
    /// prompts), and the system turn is the prefix of everything (PLAN.md 2.2).
    /// Changing it on a session that has already been evaluated would mean
    /// re-prefilling the entire conversation, so it is settable only before the
    /// first message and inherited from the project after that.
    public var storedEffort: ReasoningEffort?

    public var effort: ReasoningEffort { storedEffort ?? .medium }

    public init(id: UUID = UUID(), projectID: UUID, title: String = "New session",
                workingSubpath: String = "", createdAt: Date = Date(),
                state: SessionState = .closed, contextSize: Int32,
                tokenCount: Int = 0, ancestorID: UUID? = nil, successorID: UUID? = nil,
                storedEffort: ReasoningEffort? = nil) {
        self.id = id
        self.projectID = projectID
        self.title = title
        self.workingSubpath = workingSubpath
        self.createdAt = createdAt
        self.state = state
        self.contextSize = contextSize
        self.tokenCount = tokenCount
        self.ancestorID = ancestorID
        self.successorID = successorID
        self.storedEffort = storedEffort
    }
}

public struct TranscriptItem: Codable, Identifiable, Sendable {
    public var id: UUID
    public var at: Date
    public var kind: Kind
    /// Tokens that produced this item, where the number is known.
    ///
    /// A field on the struct rather than a payload on `Kind`, deliberately:
    /// adding an associated value to an enum case changes how it encodes, and
    /// `loadTranscript` drops items it cannot decode -- so an existing
    /// conversation would quietly lose its reasoning blocks. An added optional
    /// on a struct decodes from records written before it existed.
    public var tokens: Int?

    public enum Kind: Codable, Sendable {
        case user(String)
        case reasoning(String)
        case assistant(String)
        /// One card: the call and, once it lands, its result. Kept together
        /// because that is how a person reads it (PLAN.md 5.2).
        case tool(name: String, arguments: [String: String], result: String?)
        case note(String)
        case footer(TurnStats)
        case contextFull(used: Int, limit: Int)
    }

    public init(_ kind: Kind, id: UUID = UUID(), at: Date = Date(), tokens: Int? = nil) {
        self.id = id
        self.at = at
        self.kind = kind
        self.tokens = tokens
    }

    /// Streaming appends into the tail item rather than making one per token.
    public mutating func append(_ s: String, tokens n: Int = 0) {
        switch kind {
        case .reasoning(let t): kind = .reasoning(t + s)
        case .assistant(let t): kind = .assistant(t + s)
        default: return
        }
        if n > 0 { self.tokens = (self.tokens ?? 0) + n }
    }
}
