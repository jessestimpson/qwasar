defmodule Warden.Define do
  @moduledoc """
  `define` — the agent changing what it can do.

  PLAN.md 7.2, step for step, and the order is the contract:

    1. **Warden does not compile it.** It `erpc`s the workspace node, which is
       where a bad module can only hurt the agent.
    2. **Compile.** Errors and warnings become the tool result. A module that
       does not compile changes nothing.
    3. **Purge, carefully.** `:code.soft_purge/1` first; if a process is still
       running the previous version the load is *refused* and the result names
       the processes. Silently killing a process the agent started three steps
       ago is the kind of failure that costs an hour of confusion.
    4. **Flag concurrency**: a module that spawns or implements an OTP
       behaviour is called out, because a green load proves nothing about
       ordering.
    5. **Load**, and **register** if it implements `Crucible.Tool`.
    6. **Report** what it now exports and whether it became a tool.
  """

  @compile_timeout 30_000

  def run(source, opts \\ []) do
    force = Keyword.get(opts, :force, false)
    node = Warden.Workspace.node_name()

    with {:ok, mods, warnings} <- compile(node, source),
         :ok <- purge(node, mods, force),
         {:ok, loaded} <- load(node, mods) do
      described = Enum.map(loaded, &describe(node, &1))
      register(source, described)
      {:ok, report(described, warnings)}
    end
  end

  defp compile(node, source) do
    case erpc(node, Crucible.Loader, :compile, [source], @compile_timeout) do
      {:ok, mods, warnings} ->
        {:ok, mods, warnings}

      # Tagged apart from the compiler's own {:error, msg, warnings}, which has
      # the same shape. Without the tag the first clause matched everything and
      # a dead workspace was reported as a compile error, with the erpc reason
      # in the position where the model expected a compiler message.
      {:erpc_failed, kind, msg} ->
        {:error, kind, msg}

      {:error, msg, warnings} ->
        {:error, "compile",
         msg <> if(warnings == [], do: "", else: "\n\nwarnings:\n" <> Enum.join(warnings, "\n"))}
    end
  end

  defp purge(_node, _mods, true), do: :ok

  defp purge(node, mods, false) do
    names = Enum.map(mods, &elem(&1, 0))

    case erpc(node, Crucible.Loader, :purge_check, [names], 10_000) do
      :ok ->
        :ok

      {:erpc_failed, kind, msg} ->
        {:error, kind, msg}

      {:blocked, blocked} ->
        detail =
          Enum.map_join(blocked, "; ", fn {mod, pids} ->
            "#{inspect(mod)} is still running in #{Enum.join(pids, ", ")}"
          end)

        {:error, "purge_blocked",
         "refused: #{detail}. Loading now would kill those processes. " <>
           "Stop them first, or pass force to accept the loss."}

      {:error, kind, msg} ->
        {:error, kind, msg}
    end
  end

  defp load(node, mods) do
    case erpc(node, Crucible.Loader, :load, [mods], 10_000) do
      {:ok, loaded} -> {:ok, loaded}
      {:erpc_failed, kind, msg} -> {:error, kind, msg}
      {:error, errs} -> {:error, "load", inspect(errs)}
    end
  end

  defp describe(node, mod) do
    case erpc(node, Crucible.Loader, :describe, [mod], 10_000) do
      %{} = d -> d
      _ -> %{module: inspect(mod), exports: [], tool: false, tool_name: nil, schema: %{}, concurrent: false}
    end
  end

  # Every defined module, not only the ones that became tools.
  #
  # The registry started as a tool registry, and that was wrong: a workspace
  # restart must replay helper modules too, not only the ones that became
  # tools -- a helper a tool depends on that is absent from the manifest is
  # an :undef after the first crash.
  defp register(source, described) do
    for d <- described do
      Warden.Registry.put(%{
        module: d.module,
        tool_name: d.tool_name,
        schema: d.schema,
        source: source,
        concurrent: d.concurrent,
        loaded_at: System.system_time(:second),
        version: 0
      })
    end
  end

  defp report(described, warnings) do
    lines =
      Enum.map(described, fn d ->
        base = "#{d.module}: exports #{Enum.join(d.exports, ", ")}"

        base =
          if d.tool,
            do: base <> "\n  registered as the tool #{inspect(d.tool_name)} — call it with invoke",
            else: base <> "\n  not a Crucible.Tool, so it is loaded but not callable through invoke"

        # PLAN.md 7.2 step 4: concurrent code is where a green result means
        # least, so say so where the model will see it.
        if d.concurrent do
          base <>
            "\n  this module spawns or implements an OTP behaviour, so its " <>
            "correctness is about ordering — test it under load before you " <>
            "depend on it."
        else
          base
        end
      end)

    body = Enum.join(lines, "\n")
    if warnings == [], do: body, else: body <> "\n\nwarnings:\n" <> Enum.join(warnings, "\n")
  end

  # erpc that reports a dead node as a tool result rather than raising through
  # the bridge. spec 10 invariant 6: warden never blocks on the workspace,
  # and never dies with it either.
  defp erpc(node, mod, fun, args, timeout) do
    :erpc.call(node, mod, fun, args, timeout)
  catch
    :error, {:erpc, :noconnection} ->
      {:erpc_failed, "workspace_down",
       "the workspace node is not reachable. It restarts on its own; try again."}

    :error, {:erpc, :timeout} ->
      {:erpc_failed, "timeout", "the workspace did not answer within #{timeout}ms"}

    kind, reason ->
      {:erpc_failed, "erpc", Exception.format(kind, reason, [])}
  end
end
