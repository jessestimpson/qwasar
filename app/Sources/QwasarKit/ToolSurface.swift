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
// A session on the user's Mac gets the seven core tools.  A sandboxed one gets
// them too -- implemented once, in ToolKit, over the guest's primitives -- plus
// the four that only make sense inside the VM: elixir, define, skills, invoke.
// `fetch` rides alongside when a sandboxed project allows hosts.

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

    // The sandbox's own: the warden's workspace node and its skills.

    public static let elixirSchema = #"""
    {"type": "function", "function": {"name": "elixir", "description": "Evaluate Elixir on the workspace node and return the value. State does not persist between calls; define a module if you need something to survive. Use this to try an expression, not to do work a tool should do.", "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "Elixir expression to evaluate."}}, "required": ["code"]}}}
    """#
    public static let defineSchema = #"""
    {"type": "function", "function": {"name": "define", "description": "Compile and hot-load an Elixir or Erlang module into the workspace node, creating a SKILL: a capability you wrote, distinct from these fixed tools, invokable from now on -- and owned by the PROJECT, so every session of this project has it from its first token.\n\nWHAT IT IS FOR. Moving work off the token stream. A skill that reads, transforms or checks files under /work does in one invoke what would otherwise cost hundreds of generated tokens and several round trips through read and edit -- and generation is by far the slowest thing you do. So expect most skills to take PATHS and act on files, rather than to be pure functions of their arguments. Good candidates: a search across the project that returns only the part that matters; a check that has to run over many files; a transform you will apply more than once. A pure helper used once is cheaper written inline with elixir than defined here.\n\nTHE SKILL CONTRACT. A module becomes invokable if and only if it does all of this:\n  @behaviour Crucible.Skill         -- required; registration keys off this exact attribute\n  def name, do: \"my_skill\"          -- required; THIS string is what invoke takes, not the module name\n  def run(args), do: {:ok, \"text\"}  -- required; args is a map with STRING keys\n  def schema, do: %{...}            -- optional; documentation returned by skills, never enforced\n\nrun/1 must return {:ok, binary} or {:error, binary}. Any other shape -- including {:ok, 42}, a bare string, or :ok -- is inspected and handed back as ok text, which is almost never what you meant. Exceptions are caught and returned as an error with a stacktrace, so do not wrap the body in a rescue.\n\nA complete, loadable example:\ndefmodule LineCount do\n  @behaviour Crucible.Skill\n  def name, do: \"line_count\"\n  def schema, do: %{\"description\" => \"Count the lines in a file under /work\", \"args\" => [\"path\"]}\n  def run(%{\"path\" => p}) do\n    case File.read(Path.join(\"/work\", p)) do\n      {:ok, s} -> {:ok, Integer.to_string(length(String.split(s, \"\\n\")))}\n      {:error, e} -> {:error, \"cannot read #{p}: #{inspect(e)}\"}\n    end\n  end\nend\n\nSkill names are unique across modules: a second module claiming a name already in use is refused, and the refusal names the holder. Redefining the SAME module replaces it and bumps its version. The load is refused if a process is still running the previous version, and that refusal names those processes; stop them or pass force to accept losing them. Modules run on the workspace node, not the warden's, and are replayed there in definition order if it crashes.", "parameters": {"type": "object", "properties": {"source": {"type": "string", "description": "Complete module source."}, "force": {"type": "boolean", "description": "Load even if processes are running the old version, killing them."}}, "required": ["source"]}}}
    """#
    public static let skillsSchema = #"""
    {"type": "function", "function": {"name": "skills", "description": "List the skills you have defined, with their module, version and schema. Skills are capabilities you wrote with define -- distinct from these fixed tools -- and they belong to the project: every session of this project has them.", "parameters": {"type": "object", "properties": {}, "required": []}}}
    """#
    public static let invokeSchema = #"""
    {"type": "function", "function": {"name": "invoke", "description": "Call a skill you defined with define. `name` is the string that skill's name/0 returns, which is not the module name -- call skills to see the names. `args` is handed to its run/1 as a map with string keys; a bare string that is not JSON arrives as %{\"input\" => that_string}.", "parameters": {"type": "object", "properties": {"name": {"type": "string", "description": "The skill name, as returned by skills."}, "args": {"type": "object", "description": "Arguments passed to the skill's run/1."}}, "required": ["name"]}}}
    """#

    /// `fetch` is executed by the HOST under per-project policy (PLAN.md 8.3),
    /// offered only to a sandboxed session whose project allows hosts, and never
    /// reaches the guest at all.
    public static let fetchSchema = #"""
    {"type": "function", "function": {"name": "fetch", "description": "HTTPS GET a URL and return the response as text. Executed by the host, outside the sandbox, under a per-project allowlist of hosts -- a fetch to a host not on the list is refused and the refusal names the host, so say which host you need and why if the user should add one. GET only; responses are capped and binary content is not delivered.", "parameters": {"type": "object", "properties": {"url": {"type": "string", "description": "The https URL to fetch."}}, "required": ["url"]}}}
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

    /// A sandboxed session's: the core, then the guest's own four.
    public static let sandboxSchemas: [String] = coreSchemas + [
        elixirSchema, defineSchema, skillsSchema, invokeSchema,
    ]
    public static let guestNames: Set<String> = ["elixir", "define", "skills", "invoke"]

    /// A sandboxed session whose guest did not start: looking, not touching.
    public static let readOnlySchemas: [String] = [readSchema, globSchema, grepSchema, askSchema]
    public static let readOnlyNames: Set<String> = ["Read", "Glob", "Grep", "AskUserQuestion"]

    /// The warden ops the host calls, which the sandbox gate checks are all
    /// dispatchable: the primitives ToolKit is written over, and the guest's
    /// own tools under their own names.
    public static let wardenOps: Set<String> = [
        "read_raw", "stat", "write", "exec", "elixir", "define", "skills", "invoke",
    ]

    /// Tools that change files or the agent itself. Rendered differently in the
    /// transcript, because "it rewrote its own tooling" is what a reader scans for.
    public static let mutating: Set<String> = ["Write", "Edit", "Bash", "define"]
}
