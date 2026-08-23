// ToolSurface.swift -- what the model is told it can do.
//
// PLAN.md 7.1, and this list is now FROZEN. §2.2 is why: the tool surface is the
// system turn, and the system turn is the prefix of everything, so adding a tool
// after this point re-prefills every session in the application. Ten tools, and
// one of them is `invoke` -- which is how the agent gets an unbounded number of
// its own without any of this changing. (It was twelve: `simulate` and `replay`
// went with a retired experiment (spec 12), whose removal was the one sanctioned
// change to a frozen list -- and it cost exactly the re-prefill the freeze
// exists to prevent, once, on the next session of each project.)
//
// The first six are the C agent's, verbatim, because those descriptions were
// written against this model's training. The last four are new, and each carries
// its contract in its description for the same reason `edit`'s spells out its
// match rule: a rule the model is not told is a rule it will break.
// `Tests/golden.tsv` pins all ten; `Tests/gen_golden.c` carries the same
// list, and the golden suite failing is how the two are kept identical.

import Foundation

public enum ToolSurface {

    public static let readSchema = #"""
    {"type": "function", "function": {"name": "read", "description": "Read a file and return its exact contents, with no line numbers or other decoration, so the text can be quoted back to edit.", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "Path to the file."}}, "required": ["path"]}}}
    """#

    public static let writeSchema = #"""
    {"type": "function", "function": {"name": "write", "description": "Create a file or replace its entire contents. Use edit for changes to an existing file.", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "Path to the file."}, "content": {"type": "string", "description": "The complete new contents."}}, "required": ["path", "content"]}}}
    """#

    public static let editSchema = #"""
    {"type": "function", "function": {"name": "edit", "description": "Replace a run of whole lines in a file. The old text must match complete lines exactly once, including indentation; if it matches nowhere or in more than one place the edit is refused and nothing changes. Quote enough surrounding lines to be unique. To insert, set old to a unique nearby line and new to that same line plus the addition. To delete, set new to an empty string.", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "Path to the file."}, "old": {"type": "string", "description": "Exact existing lines to replace."}, "new": {"type": "string", "description": "Replacement lines."}}, "required": ["path", "old", "new"]}}}
    """#

    public static let listSchema = #"""
    {"type": "function", "function": {"name": "list", "description": "List the entries of a directory.", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "Directory path; defaults to the working directory."}}, "required": []}}}
    """#

    public static let grepSchema = #"""
    {"type": "function", "function": {"name": "grep", "description": "Search files recursively for a regular expression and return matching lines with their file and line number.", "parameters": {"type": "object", "properties": {"pattern": {"type": "string", "description": "Extended regular expression."}, "path": {"type": "string", "description": "File or directory to search; defaults to the working directory."}}, "required": ["pattern"]}}}
    """#

    public static let bashSchema = #"""
    {"type": "function", "function": {"name": "bash", "description": "Run a shell command and return its combined output and exit status.", "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "Command to run via /bin/sh."}}, "required": ["command"]}}}
    """#

    public static let elixirSchema = #"""
    {"type": "function", "function": {"name": "elixir", "description": "Evaluate Elixir on the workspace node and return the value. State does not persist between calls; define a module if you need something to survive. Use this to try an expression, not to do work a tool should do.", "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "Elixir expression to evaluate."}}, "required": ["code"]}}}
    """#

    public static let defineSchema = #"""
    {"type": "function", "function": {"name": "define", "description": "Compile and hot-load an Elixir or Erlang module into the workspace node, changing what you can do without restarting anything.\n\nWHAT IT IS FOR. Moving work off the token stream. A tool that reads, transforms or checks files under /work does in one invoke what would otherwise cost hundreds of generated tokens and several round trips through read and edit -- and generation is by far the slowest thing you do. So expect most tools to take PATHS and act on files, rather than to be pure functions of their arguments. Good candidates: a search across the project that returns only the part that matters; a check that has to run over many files; a transform you will apply more than once. A pure helper used once is cheaper written inline with elixir than defined here.\n\nTHE TOOL CONTRACT. A module becomes callable through invoke if and only if it does all of this:\n  @behaviour Crucible.Tool          -- required; registration keys off this exact attribute\n  def name, do: \"my_tool\"           -- required; THIS string is what invoke takes, not the module name\n  def run(args), do: {:ok, \"text\"}  -- required; args is a map with STRING keys\n  def schema, do: %{...}            -- optional; documentation returned by tools, never enforced\n\nrun/1 must return {:ok, binary} or {:error, binary}. Any other shape -- including {:ok, 42}, a bare string, or :ok -- is inspected and handed back as ok text, which is almost never what you meant. Exceptions are caught and returned as an error with a stacktrace, so do not wrap the body in a rescue.\n\nA complete, loadable example:\ndefmodule LineCount do\n  @behaviour Crucible.Tool\n  def name, do: \"line_count\"\n  def schema, do: %{\"description\" => \"Count the lines in a file under /work\", \"args\" => [\"path\"]}\n  def run(%{\"path\" => p}) do\n    case File.read(Path.join(\"/work\", p)) do\n      {:ok, s} -> {:ok, Integer.to_string(length(String.split(s, \"\\n\")))}\n      {:error, e} -> {:error, \"cannot read #{p}: #{inspect(e)}\"}\n    end\n  end\nend\n\nTool names are unique across modules: a second module claiming a name already in use is refused, and the refusal names the holder. Redefining the SAME module replaces it and bumps its version. The load is refused if a process is still running the previous version, and that refusal names those processes; stop them or pass force to accept losing them. Modules run on the workspace node, not the warden's, and are replayed there in definition order if it crashes.", "parameters": {"type": "object", "properties": {"source": {"type": "string", "description": "Complete module source."}, "force": {"type": "boolean", "description": "Load even if processes are running the old version, killing them."}}, "required": ["source"]}}}
    """#

    public static let toolsSchema = #"""
    {"type": "function", "function": {"name": "tools", "description": "List the tools you have defined, with their module, version and schema.", "parameters": {"type": "object", "properties": {}, "required": []}}}
    """#

    public static let invokeSchema = #"""
    {"type": "function", "function": {"name": "invoke", "description": "Call a tool you defined with define. `name` is the string that tool's name/0 returns, which is not the module name -- call tools to see the names. `args` is handed to its run/1 as a map with string keys; a bare string that is not JSON arrives as %{\"input\" => that_string}.", "parameters": {"type": "object", "properties": {"name": {"type": "string", "description": "The tool name, as returned by tools."}, "args": {"type": "object", "description": "Arguments passed to the tool's run/1."}}, "required": ["name"]}}}
    """#



    /// `fetch` is NOT part of the frozen guest surface: it is executed by the
    /// HOST under per-project policy (PLAN.md 8.3), appears only when the
    /// project's allowlist is non-empty, and never reaches the guest at all.
    /// The golden suite pins the ten guest schemas; this one rides alongside.
    public static let fetchSchema = #"""
    {"type": "function", "function": {"name": "fetch", "description": "HTTPS GET a URL and return the response as text. Executed by the host, outside the sandbox, under a per-project allowlist of hosts -- a fetch to a host not on the list is refused and the refusal names the host, so say which host you need and why if the user should add one. GET only; responses are capped and binary content is not delivered.", "parameters": {"type": "object", "properties": {"url": {"type": "string", "description": "The https URL to fetch."}}, "required": ["url"]}}}
    """#

    /// The frozen surface, in order -- and the order is part of the system turn.
    public static let guestSchemas: [String] = [
        readSchema, writeSchema, editSchema, listSchema, grepSchema, bashSchema,
        elixirSchema, defineSchema, toolsSchema, invokeSchema,
    ]

    /// M1's host-side stand-in: the three that cannot change anything. Kept for
    /// a session with no sandbox.
    public static let readOnlySchemas: [String] = [readSchema, listSchema, grepSchema]

    public static let names: Set<String> = [
        "read", "write", "edit", "list", "grep", "bash",
        "elixir", "define", "tools", "invoke",
    ]
    public static let readOnlyNames: Set<String> = ["read", "list", "grep"]

    /// Tools that change the guest's copy or the agent itself. Rendered
    /// differently in the transcript, because "it rewrote its own tooling" is
    /// what a reader scans for.
    public static let mutating: Set<String> = ["write", "edit", "bash", "define"]
}
