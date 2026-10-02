// UserQuestions.swift -- AskUserQuestion: the model asks, the user picks.
//
// A model given an underspecified request explores every reading of it in
// its reasoning -- thousands of tokens on what one question would settle.
// This is the question, shaped as Claude Code's tool for it is: one to four
// questions, each a short header, a question, and two to four options with
// a description of what each means; multi-select where the choices are not
// exclusive; and always a free-text "Other", so the options never box the
// user in.  The app shows them as a card in the transcript and the turn
// waits for the answer -- the call is the app's, not the tools' backend.

import Foundation

public struct UserQuestion: Sendable, Equatable, Identifiable {
    public struct Option: Sendable, Equatable {
        public var label: String
        public var description: String
    }
    public var question: String
    public var header: String
    public var options: [Option]
    public var multiSelect: Bool
    public var id: String { question }
}

public enum UserQuestions {
    public static let toolName = "AskUserQuestion"

    /// The call's `questions`, or why they cannot be shown.  Lenient where
    /// leniency is free (a missing header, a missing description), strict
    /// where the card would not work (no question, fewer than two options).
    public static func parse(_ call: ToolCall) -> Result<[UserQuestion], ToolFailure> {
        guard let raw = call.argument("questions"), let data = raw.data(using: .utf8),
              let items = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] else {
            return .failure(ToolFailure("AskUserQuestion requires questions: an array of "
                                        + "{question, header, options: [{label, description}], multiSelect}"))
        }
        guard (1...4).contains(items.count) else {
            return .failure(ToolFailure("ask between one and four questions at a time"))
        }
        var out: [UserQuestion] = []
        for item in items {
            guard let q = (item["question"] as? String)?.trimmingCharacters(in: .whitespacesAndNewlines),
                  !q.isEmpty else { return .failure(ToolFailure("every question needs its question text")) }
            let opts = (item["options"] as? [[String: Any]] ?? []).compactMap { o -> UserQuestion.Option? in
                guard let l = (o["label"] as? String)?.trimmingCharacters(in: .whitespacesAndNewlines),
                      !l.isEmpty else { return nil }
                return .init(label: l, description: (o["description"] as? String) ?? "")
            }
            guard (2...4).contains(opts.count) else {
                return .failure(ToolFailure("“\(q)” needs two to four options (the user can always "
                                            + "answer in their own words as well)"))
            }
            let multi = (item["multiSelect"] as? Bool) ?? ((item["multiSelect"] as? String) == "true")
            out.append(UserQuestion(question: q, header: String(((item["header"] as? String) ?? "").prefix(12)),
                                    options: opts, multiSelect: multi))
        }
        if Set(out.map(\.question)).count != out.count {
            return .failure(ToolFailure("each question must be different"))
        }
        return .success(out)
    }

    /// The tool result: each question with what the user chose.
    public static func answerText(_ questions: [UserQuestion], _ answers: [String: [String]]) -> String {
        let lines = questions.map { q -> String in
            let a = (answers[q.question] ?? []).filter { !$0.isEmpty }
            return "“\(q.question)” — " + (a.isEmpty ? "(no answer)" : a.joined(separator: "; "))
        }
        return "The user answered:\n" + lines.joined(separator: "\n")
             + "\n\nContinue with these answers in mind."
    }

    public static let skipped = "The user chose not to answer these questions. Proceed with your best "
                              + "judgment, and state the assumptions you made in your reply."
}
