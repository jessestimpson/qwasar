// ExplorationNudge.swift -- a reminder when a turn reads and reads and does
// not act.
//
// Flash-Next on a coding task tends to keep reading and searching well past
// the point where it knows what to change.  This is documented of agents in
// general -- "analysis paralysis", the commonest overthinking pattern across
// 4,018 SWE-bench Verified trajectories (Cuadron et al., arXiv 2502.08235),
// where more of it went with fewer tasks solved -- and Qwen3-Coder-Next's
// report (arXiv 2603.00729) says its agent RL lengthened trajectories with
// nothing shaping their efficiency, so it is a trained-in tendency rather than
// something an engine setting removes.
//
// What helps without training is feedback in the trajectory.  SWE-PRM (arXiv
// 2509.02360) raised SWE-bench Verified from 40.0% to 50.6% with mid-run
// feedback, and found that naming the inefficiency beat prescribing the next
// action.  So this names it -- how many reads and searches since the last
// change -- and leaves the call to the model, including that the task may be a
// question, which needs no edit at all.
//
// Counted: Read, Grep, Glob.  A change -- Edit, Write -- ends the streak.
// Bash is neither: it runs a build or a test as often as it looks at
// something, and running the check is what the system prompt asks for.

import Foundation

public struct ExplorationNudge: Sendable {
    public static let exploring: Set<String> = ["Read", "Grep", "Glob"]
    public static let changing: Set<String> = ["Edit", "Write"]

    /// Reads and searches before the first reminder, and between later ones;
    /// 0 is off.
    public let after: Int
    /// Reads and searches since the last change, or since the turn began.
    public private(set) var streak = 0
    /// Whether any change has been made this turn -- the wording differs.
    public private(set) var changed = false
    private var lastReminded = 0

    public init(after: Int) { self.after = max(0, after) }

    public mutating func record(_ toolName: String) {
        if Self.exploring.contains(toolName) {
            streak += 1
        } else if Self.changing.contains(toolName) {
            streak = 0
            lastReminded = 0
            changed = true
        }
    }

    /// The reminder due after this round of tool calls, or nil.  Due when the
    /// streak has passed `after` more reads since the last one; a round of
    /// several parallel reads that crosses the mark gets one reminder, not
    /// one per read.
    public mutating func reminder() -> String? {
        guard after > 0, streak >= lastReminded + after else { return nil }
        lastReminded = streak
        let since = changed ? "since your last edit" : "this turn, and no edit yet"
        return "<system-reminder>\n\(streak) reads and searches \(since). If you already know where "
            + "the change goes and roughly what it is, make it now and let building or testing show "
            + "what is missing; read further only for a specific question you can name. If the task "
            + "is a question rather than a change, answer it.\n</system-reminder>"
    }
}
