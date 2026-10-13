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

    /// This project's layer of session settings (PLAN.md 8.5).  Named for
    /// the sandbox it once configured; the stored key stays so records decode.
    public var sandbox: SettingsOverlay?

    /// The project layer as resolution sees it.
    public var overlay: SettingsOverlay { sandbox ?? SettingsOverlay() }

    public init(id: UUID = UUID(), name: String, rootBookmark: Data,
                resolvedRoot: URL? = nil, systemPrompt: String = Project.defaultSystem,
                defaultEffort: ReasoningEffort? = nil,
                sandbox: SettingsOverlay? = nil) {
        self.id = id
        self.name = name
        self.rootBookmark = rootBookmark
        self.resolvedRoot = resolvedRoot
        self.systemPrompt = systemPrompt
        self.defaultEffort = defaultEffort
        self.sandbox = sandbox
    }

    enum CodingKeys: String, CodingKey {
        case id, name, rootBookmark, systemPrompt, defaultEffort, sandbox
    }

    // MARK: The config project (PLAN.md 8.5)

    /// A fixed id, so the special project is the same one across launches and
    /// its sessions persist like any other's.
    public static let configProjectID = UUID(uuidString: "C0F16000-0000-4000-8000-000000000001")!

    public var isConfig: Bool { id == Project.configProjectID }

    /// The built-in project whose sessions manage Qwasar itself: host-side
    /// config tools, no folder.
    public static let configProjectName = "Qwasar Config"
    public static func configProject() -> Project {
        Project(id: configProjectID, name: configProjectName, rootBookmark: Data(),
                systemPrompt: """
                You manage Qwasar's configuration through the config tools. \
                Show the current state before changing it, change only what was \
                asked, and state what you changed, where, and when it takes effect.
                """)
    }

    /// The project's own instructions. NOT a description of the environment --
    /// that is the executor's to state (see `ToolExecuting.environmentDescription`),
    /// because it is a fact about what is running rather than a preference.
    /// Empty: how to work is the harness's (SystemPrompt), and a project's
    /// own guidance is added under it when the user writes some -- or when
    /// the project keeps an AGENTS.md, CLAUDE.md or QWEN.md.
    public static let defaultSystem = ""

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

/// Where a session's tools run.  Only on the user's Mac now; `sandbox` is a
/// session from when they could run in a Linux VM -- its context speaks of
/// /work and tools that are gone, so it can be read but not continued.
public enum ToolPlacement: String, Codable, Sendable {
    /// Directly on the user's Mac, with their shell environment
    /// (HostToolRunner).
    case host
    /// In the VM that was removed.  Kept so those records still decode.
    case sandbox
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
    /// Compaction chain (PLAN.md 2.4).
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
    /// This session's layer of settings (PLAN.md 8.5): the most specific
    /// overlay, consulted first. Optional so old records decode.
    public var sandbox: SettingsOverlay?
    /// What this session's delegations have cost (spec §15.3), summed. The
    /// escalation budget is enforced against this, so it persists.
    public var spentUSD: Double?
    /// The engine's id for the model that last ran this session
    /// (`qwasar_model_id`). Optional so old records decode; they were all the 27B.
    public var modelID: String?
    /// The server's session for this record (API.md), opened on first send
    /// and kept for the record's life.  nil for a record from before the
    /// server owned sessions: readable, and closed (PLAN-qwasar.md §7).
    public var serverSessionID: String?
    /// Where this session's tools run.  Optional only so a record without it
    /// decodes, as host.
    public var tools: ToolPlacement?
    /// The session's running notes: what a fresh session would start from
    /// if this one had to continue elsewhere.  Written off the record, while
    /// the user reads a reply (AppState.takeNotes); a successor starts with
    /// its ancestor's.  `notesTokens` is how much of the conversation they
    /// had seen.
    public var notes: String?
    public var notesAt: Date?
    public var notesTokens: Int?
    /// "Allow for this session" (spec §16.6): the policy reasons the user
    /// approved.  Only the approval card adds to it; nil decodes old records.
    public var approvals: [String]?
    /// Secrets this session's tools have read (spec §16.4), as shown: its
    /// model still holds them when the session reopens, so they persist.
    public var secretsRead: [String]?

    public var effort: ReasoningEffort { storedEffort ?? .medium }
    public var placement: ToolPlacement { tools ?? .host }
    /// A session whose tools ran in the removed VM: readable, not continued.
    public var ranInSandbox: Bool { placement == .sandbox }

    public init(id: UUID = UUID(), projectID: UUID, title: String = "New session",
                workingSubpath: String = "", createdAt: Date = Date(),
                state: SessionState = .closed, contextSize: Int32,
                tokenCount: Int = 0, ancestorID: UUID? = nil, successorID: UUID? = nil,
                storedEffort: ReasoningEffort? = nil, sandbox: SettingsOverlay? = nil,
                tools: ToolPlacement = .host) {
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
        self.sandbox = sandbox
        self.tools = tools
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
        /// A completed delegation (spec §15.2): the sub-session, kept in the
        /// transcript so a parked session replays it readable. `log` is the
        /// rendered sub-conversation, `ended` how it finished.
        case delegation(model: String, task: String, log: String,
                        costUSD: Double, ended: String)
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
