defmodule Crucible.Skill do
  @moduledoc """
  What a SKILL looks like: a capability the agent writes for itself.

  Spec 7.2. Named apart from the ten in-context tools deliberately -- those
  are given, fixed, and selected by the model from its schema list; a skill
  is made, open-ended, reached through `invoke`, and owned by the PROJECT
  (the host replays the project's skill library into every session's guest).

  A module implementing this behaviour, hot-loaded into the workspace node,
  becomes invokable on the very next step -- no restart, and no change to the
  tool surface the model is shown, which is what makes it free (spec 2.2).

      defmodule ASTGrep do
        @behaviour Crucible.Skill
        def name, do: "grep_ast"
        def schema, do: %{"description" => "...", "args" => ["pattern"]}
        def run(%{"pattern" => p}), do: {:ok, "..."}
      end

  `schema/0` is documentation for the model rather than something enforced:
  the registry hands it back through `skills()`, and the model decides what
  to pass. Validating it here would mean inventing a schema language the
  model would then have to satisfy, which buys nothing an error message does
  not.
  """

  @callback name() :: String.t()
  @callback schema() :: map()
  @callback run(args :: map()) :: {:ok, String.t()} | {:error, String.t()}

  @optional_callbacks schema: 0
end
