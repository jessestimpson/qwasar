// ToolSurface.swift -- what the model is told it can do.
//
// The tool names, parameter names and contracts are Claude Code's -- Read,
// Write, Edit, Glob, Grep, Bash, TodoWrite -- because Qwen3.8 Flash-Next's
// agentic results are reported in that harness (its model card: SWE-bench
// Pro "Claude Code harness"), so these are the calls it is most practised at.
// The descriptions are this project's own.  They were the C agent's six
// until 2026-09-30; that list was frozen because the surface is part of every
// session's prefix, and it changed when the app started from a clean slate.
//
// A session gets the core tools, run on the user's Mac (HostToolRunner).

import Foundation

public enum ToolSurface {

    public static let readSchema = #"""
    {"type": "function", "function": {"name": "Read", "description": "Read a file. Output is numbered like `cat -n`: each line is its line number, a tab, then the line exactly as it is in the file. The number and the tab are not part of the file -- never include them in old_string for Edit. By default up to 2000 lines from the start; for a long file pass offset and limit to read the part you need, and use Grep to find it first. Lines longer than 2000 characters are cut.", "parameters": {"type": "object", "properties": {"file_path": {"type": "string", "description": "The file to read. Absolute, or relative to the project directory."}, "offset": {"type": "integer", "description": "The line number to start at (1-based). Only for files too long to read at once."}, "limit": {"type": "integer", "description": "How many lines to read. Only for files too long to read at once."}}, "required": ["file_path"]}}}
    """#
    public static let writeSchema = #"""
    {"type": "function", "function": {"name": "Write", "description": "Write a file, creating it and any missing directories, or replacing it entirely if it exists. Prefer Edit for changing an existing file; use Write for new files or complete rewrites.", "parameters": {"type": "object", "properties": {"file_path": {"type": "string", "description": "The file to write. Absolute, or relative to the project directory."}, "content": {"type": "string", "description": "The complete contents of the file."}}, "required": ["file_path", "content"]}}}
    """#
    public static let editSchema = #"""
    {"type": "function", "function": {"name": "Edit", "description": "Replace exact text in a file. old_string must appear in the file exactly -- same characters, same indentation, no line-number prefixes from Read -- and exactly once, unless replace_all is true. If it is not found, or found more than once, nothing changes and you are told why: include more surrounding lines to make it unique. To delete, set new_string to an empty string.", "parameters": {"type": "object", "properties": {"file_path": {"type": "string", "description": "The file to edit. Absolute, or relative to the project directory."}, "old_string": {"type": "string", "description": "The exact text to replace."}, "new_string": {"type": "string", "description": "The text to replace it with. Must differ from old_string."}, "replace_all": {"type": "boolean", "description": "Replace every occurrence of old_string rather than exactly one. Default false."}}, "required": ["file_path", "old_string", "new_string"]}}}
    """#
    public static let globSchema = #"""
    {"type": "function", "function": {"name": "Glob", "description": "Find files by name with a glob pattern such as \"**/*.swift\" or \"src/**/test_*.py\". Returns matching paths, most recently modified first. Files ignored by .gitignore, and hidden files, are left out.", "parameters": {"type": "object", "properties": {"pattern": {"type": "string", "description": "The glob pattern to match file paths against."}, "path": {"type": "string", "description": "The directory to search in. Defaults to the project directory."}}, "required": ["pattern"]}}}
    """#
    public static let grepSchema = #"""
    {"type": "function", "function": {"name": "Grep", "description": "Search file contents with a regular expression (ripgrep syntax: \\d, \\w, \\s, (?i) and the like all work). Respects .gitignore. Choose the output with output_mode: \"files_with_matches\" (the default) lists matching files; \"content\" shows matching lines with their line numbers; \"count\" gives matches per file.", "parameters": {"type": "object", "properties": {"pattern": {"type": "string", "description": "The regular expression to search for."}, "path": {"type": "string", "description": "File or directory to search. Defaults to the project directory."}, "glob": {"type": "string", "description": "Only search files matching this glob, e.g. \"*.ts\" or \"src/**/*.c\"."}, "output_mode": {"type": "string", "enum": ["files_with_matches", "content", "count"], "description": "What to return. Default files_with_matches."}, "-i": {"type": "boolean", "description": "Case-insensitive search."}, "-C": {"type": "integer", "description": "Lines of context around each match, with output_mode content."}}, "required": ["pattern"]}}}
    """#
    public static let bashSchema = #"""
    {"type": "function", "function": {"name": "Bash", "description": "Run a shell command with /bin/sh in the project directory and return its combined output and exit status. Use it to build, run tests, run git read-only commands, and use the project's own tools. Prefer Read, Edit, Write, Glob and Grep for files: they are faster and their results are exact. Commands that wait for input will hang until the timeout.", "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "The command to run."}, "description": {"type": "string", "description": "In a few words, what the command does."}, "timeout": {"type": "integer", "description": "Milliseconds before the command is killed. Default 120000, at most 600000."}}, "required": ["command"]}}}
    """#
    public static let askSchema = #"""
    {"type": "function", "function": {"name": "AskUserQuestion", "description": "Ask the user to settle a decision that is theirs to make, before you commit to it: what to build, which of several reasonable approaches, how far the change should reach, a preference the request does not state. The questions appear in the app as a card with your options as choices; the user picks, or answers in their own words, and their answers come back as the result. Ask when the request is ambiguous or underspecified and the code cannot tell you -- early, and in one call, rather than exploring every reading of the request in your reasoning. Do not ask what you can find out by reading the code or running something, and do not ask for permission to proceed with a reasonable default. Offer concrete options; put the one you recommend first and end its label with \"(Recommended)\". The user can always choose \"Other\" and write their own answer, so do not add an \"Other\" option yourself.", "parameters": {"type": "object", "properties": {"questions": {"type": "array", "description": "One to four questions.", "minItems": 1, "maxItems": 4, "items": {"type": "object", "properties": {"question": {"type": "string", "description": "The complete question, ending in a question mark."}, "header": {"type": "string", "description": "A very short label for the question, at most 12 characters, e.g. \"Approach\" or \"Scope\"."}, "options": {"type": "array", "description": "Two to four distinct choices.", "minItems": 2, "maxItems": 4, "items": {"type": "object", "properties": {"label": {"type": "string", "description": "The choice, in one to five words."}, "description": {"type": "string", "description": "What choosing it means: its trade-off or consequence."}}, "required": ["label", "description"]}}, "multiSelect": {"type": "boolean", "description": "True when more than one choice may apply; false (the default) when they are exclusive."}}, "required": ["question", "header", "options", "multiSelect"]}}}, "required": ["questions"]}}}
    """#
    public static let todoSchema = #"""
    {"type": "function", "function": {"name": "TodoWrite", "description": "Keep a short task list for work with several steps, and update it as you go: mark an item in_progress when you start it and completed as soon as it is done. Each call replaces the whole list. Skip it for a single simple step.", "parameters": {"type": "object", "properties": {"todos": {"type": "array", "description": "The complete, updated list.", "items": {"type": "object", "properties": {"content": {"type": "string", "description": "The task, as an imperative (\"Fix the parser\")."}, "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}}, "required": ["content", "status"]}}}, "required": ["todos"]}}}
    """#

    /// The core surface, in order -- and the order is part of the system turn.
    public static let coreSchemas: [String] = [
        readSchema, writeSchema, editSchema, globSchema, grepSchema, bashSchema, todoSchema,
        askSchema,
    ]
    /// AskUserQuestion is among them, though no backend runs it: the app
    /// answers it with a card in the transcript (UserQuestions).
    public static let coreNames: Set<String> = ["Read", "Write", "Edit", "Glob", "Grep", "Bash", "TodoWrite",
                                                "AskUserQuestion"]

    /// Tools that change files. Rendered differently in the transcript,
    /// because a change is what a reader scans for.
    public static let mutating: Set<String> = ["Write", "Edit", "Bash"]
}
