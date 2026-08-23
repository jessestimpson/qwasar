defmodule Crucible.Tool do
  @moduledoc """
  What a tool the agent writes for itself looks like.

  PLAN.md 7.2. A module implementing this behaviour, hot-loaded into the
  workspace node, becomes callable through `invoke` on the very next step — no
  restart, and no change to the tool surface the model is shown, which is what
  makes it free (§2.2: the surface is the system turn, and the system turn is
  the prefix of everything).

      defmodule ASTGrep do
        @behaviour Crucible.Tool
        def name, do: "grep_ast"
        def schema, do: %{"description" => "...", "args" => ["pattern"]}
        def run(%{"pattern" => p}), do: {:ok, "..."}
      end

  `schema/0` is documentation for the model rather than something enforced: the
  registry hands it back through `tools()`, and the model decides what to pass.
  Validating it here would mean inventing a schema language the model would then
  have to satisfy, which buys nothing an error message does not.
  """

  @callback name() :: String.t()
  @callback schema() :: map()
  @callback run(args :: map()) :: {:ok, String.t()} | {:error, String.t()}

  @optional_callbacks schema: 0
end
