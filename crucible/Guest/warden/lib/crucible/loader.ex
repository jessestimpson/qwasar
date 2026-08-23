defmodule Crucible.Loader do
  @moduledoc """
  Compiling and loading the agent's own code, on the agent's own node.

  This module runs on the **workspace** node — warden `erpc`s into it. That
  boundary is the whole safety story of PLAN.md 7.2: compilation happens where a
  bad module can only hurt the agent, and warden never evaluates anything the
  model wrote.

  The steps are in this order for a reason. Compilation is a pure function up to
  the point of loading, so a module that does not compile changes nothing. The
  purge check comes before the load, because the BEAM keeps exactly two versions
  of a module and a third load *kills any process still running the oldest* —
  silently. An agent that started a process three steps ago and then reloads its
  module twice would lose it with no explanation, so warden refuses instead and
  says which processes are in the way.
  """

  @doc """
  Compiles source without loading it. Returns the modules it would define, plus
  any warnings, or the compiler's own message.
  """
  def compile(source) when is_binary(source) do
    {result, diagnostics} =
      Code.with_diagnostics(fn ->
        try do
          {:ok, Code.compile_string(source, "crucible://agent.ex")}
        rescue
          e -> {:error, Exception.message(e)}
        catch
          kind, reason -> {:error, Exception.format(kind, reason, [])}
        end
      end)

    warnings =
      diagnostics
      |> Enum.filter(&(&1.severity == :warning))
      |> Enum.map(&"#{&1.message} (line #{inspect(&1.position)})")

    case result do
      {:ok, mods} -> {:ok, mods, warnings}
      {:error, msg} -> {:error, msg, warnings}
    end
  end

  @doc """
  Whether every module can be replaced without killing a running process.

  `:code.soft_purge/1` returns false when a process is still executing the old
  version. That is the case worth stopping on, so this reports which modules are
  blocked and by whom rather than forcing it.
  """
  def purge_check(modules) do
    blocked =
      for mod <- modules,
          :erlang.check_old_code(mod),
          not :code.soft_purge(mod) do
        {mod, holders(mod)}
      end

    if blocked == [], do: :ok, else: {:blocked, blocked}
  end

  defp holders(mod) do
    Process.list()
    |> Enum.filter(&(:erlang.check_process_code(&1, mod) == true))
    |> Enum.map(&inspect/1)
  end

  @doc "Loads compiled modules. Purging is the caller's decision, not ours."
  def load(mods) do
    results =
      for {mod, bin} <- mods do
        _ = :code.purge(mod)

        case :code.load_binary(mod, ~c"crucible://agent.ex", bin) do
          {:module, ^mod} -> {:ok, mod}
          {:error, reason} -> {:error, mod, reason}
        end
      end

    case Enum.filter(results, &match?({:error, _, _}, &1)) do
      [] -> {:ok, Enum.map(results, fn {:ok, m} -> m end)}
      errs -> {:error, errs}
    end
  end

  @doc """
  What a loaded module offers: whether it is a tool, and what it exports.

  Reported back to the model because "it compiled" is not the interesting fact —
  "it registered as `grep_ast` and exports run/1" is.
  """
  def describe(mod) do
    exports = mod.module_info(:exports) |> Enum.reject(&(elem(&1, 0) == :module_info))

    behaviours =
      mod.module_info(:attributes)
      |> Keyword.get_values(:behaviour)
      |> List.flatten()

    tool? = Crucible.Tool in behaviours and function_exported?(mod, :name, 0)

    %{
      module: inspect(mod),
      exports: Enum.map(exports, fn {f, a} -> "#{f}/#{a}" end),
      tool: tool?,
      tool_name: if(tool?, do: mod.name(), else: nil),
      schema: if(tool? and function_exported?(mod, :schema, 0), do: mod.schema(), else: %{}),
      # A module that spawns, or implements a behaviour with callbacks, is code
      # whose correctness is about *when* rather than *what* -- the model is
      # told so before it depends on code a green load has proven nothing about.
      concurrent: concurrent?(mod, behaviours)
    }
  end

  defp concurrent?(mod, behaviours) do
    gen = Enum.any?(behaviours, &(&1 in [:gen_server, :gen_statem, :supervisor, GenServer]))
    spawns = function_exported?(mod, :start_link, 1) or function_exported?(mod, :start_link, 0)
    gen or spawns
  end

  @doc "Calls a registered tool. Runs here, on the agent's node."
  def invoke(mod, args) do
    case mod.run(args) do
      {:ok, out} when is_binary(out) -> {:ok, out}
      {:error, msg} -> {:error, to_string(msg)}
      other -> {:ok, inspect(other, pretty: true, limit: 50)}
    end
  rescue
    e -> {:error, "the tool raised: " <> Exception.message(e) <> "\n" <> Exception.format_stacktrace(__STACKTRACE__)}
  catch
    kind, reason -> {:error, Exception.format(kind, reason, __STACKTRACE__)}
  end
end
