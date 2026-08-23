defmodule Warden.Simulate do
  @moduledoc """
  Deterministic simulation, handed to the model (PLAN.md 9.3, 9.4).

  An ephemeral node per run, and that is not tidiness. `eta_sched` suspends
  every process in the VM and `eta_time` replaces the clock — neither is
  something to do to a node that is also answering tool calls or holding the
  vsock open. So a simulation gets its own `sim_N` node, the agent's modules are
  compiled onto it, eta drives them, and the node is killed when the run
  returns. eta's "one simulation per VM" constraint is then satisfied per
  *node*, and never becomes a lock on the session.

  Two things are reported that a bare pass/fail would not be, and both are
  §9.4's argument that a green result must state its own weakness:

    * **the shrunk trace**, not the raw one — a violation is useless to the
      model at twenty thousand scheduling decisions;
    * **eta's own `audit/1`**, which reports determinism holes the run itself
      noticed: cold code loads, stray timers, sends the scheduler did not own.
      A run that audits dirty is a run whose green means less, and the model is
      told so rather than left to assume.
  """

  @sim_lib "/opt/sim/lib"
  @boot_timeout_ms 15_000
  @run_timeout_ms 120_000

  def available?, do: File.dir?(@sim_lib)

  @doc """
  Runs `harness` under eta, on a fresh node, with the agent's own modules on it.

  `sources` is the manifest, in definition order, so the harness can refer to
  the modules it is a harness for.
  """
  def run(harness, sources, opts) do
    if available?() do
      seeds = Map.get(opts, :seeds, [1, 2, 3, 4, 5, 6, 7, 8])
      with {:ok, node, port} <- start_node(),
           :ok <- load_sources(node, sources) do
        try do
          sweep(node, harness, seeds, opts)
        after
          stop_node(node, port)
        end
      end
    else
      {:error, "unavailable",
       "this build has no simulation support: /opt/sim/lib is absent. " <>
         "eta was removed, or the image predates it (PLAN.md 9.5)."}
    end
  end

  @doc "Re-runs one exact schedule, so a fix is a checkable claim."
  def replay(harness, sources, trace, opts) do
    if available?() do
      with {:ok, node, port} <- start_node(),
           :ok <- load_sources(node, sources) do
        try do
          case erpc(node, :eta_run, :replay, [module(harness), trace, erl_opts(opts)], @run_timeout_ms) do
            %{outcome: :ok} = r -> {:ok, "PASS — the same schedule, no violation.\n" <> audit_line(node, r)}
            %{outcome: outcome} = r -> {:ok, "STILL FAILS: #{inspect(outcome)}\n" <> audit_line(node, r)}
            {:error, kind, msg} -> {:error, kind, msg}
            other -> {:error, "replay", inspect(other)}
          end
        after
          stop_node(node, port)
        end
      end
    else
      {:error, "unavailable", "this build has no simulation support"}
    end
  end

  # ---- the sweep -----------------------------------------------------------

  defp sweep(node, harness, seeds, opts) do
    mod = module(harness)

    results =
      Enum.map(seeds, fn seed ->
        o = erl_opts(Map.put(opts, :seed, seed))
        {seed, erpc(node, :eta_run, :run, [mod, o], @run_timeout_ms)}
      end)

    # A run that did not happen is not a run with no violations.
    #
    # An earlier cut folded errored runs into the success path -- they simply
    # failed to match `%{outcome: _}` -- and reported "0 violations in 8 seeds"
    # for eight seeds that had all raised `:undef` before executing a single
    # op. That is the vacuous green this whole section exists to prevent, and it
    # came from the reporting layer rather than from the harness. Errors are now
    # a hard failure, reported first and on their own.
    errored = for {seed, r} <- results, not is_map(r) or not Map.has_key?(r, :outcome), do: {seed, r}

    if errored != [] do
      {_seed, first} = hd(errored)

      {:error, "sim_failed",
       "the simulation did not run: #{length(errored)} of #{length(seeds)} seeds errored " <>
         "before completing.\n\n#{describe_error(first)}\n\n" <>
         "Nothing was explored, so this is neither a pass nor a violation."}
    else
      sweep_results(node, mod, results, seeds, opts)
    end
  end

  defp describe_error({:error, kind, msg}), do: hint(kind <> ": " <> msg)
  defp describe_error(other), do: hint(inspect(other, limit: 20))

  # Error text with the remedy attached, for the traps a harness author hits
  # first. The `already_started` one is not exotic: eta calls `init/2` once per
  # run, and shrinking runs dozens of them, so any harness that starts a process
  # under a fixed name works exactly once and then fails in a way that says
  # nothing about the harness.
  defp hint(msg) do
    cond do
      String.contains?(msg, "already_started") ->
        msg <>
          "\n\n  Your harness starts a process under a registered name. eta calls " <>
          "init/2 once per run, and shrinking runs it many times, so the second " <>
          "one fails with :already_started.\n" <>
          "  Start it unregistered -- :gen_server.start_link(__MODULE__, [], []) " <>
          "with no {:local, _} -- and keep the pid in the harness state."

      String.contains?(msg, ":undef") and String.contains?(msg, "init") ->
        msg <>
          "\n\n  The harness module is not on the simulation node. Only modules " <>
          "defined through `define` are carried there; define the harness too."

      true ->
        msg
    end
  end

  defp sweep_results(node, mod, results, seeds, opts) do
    failures = for {seed, %{outcome: out} = r} <- results, out != :ok, do: {seed, out, r}

    case failures do
      [] ->
        audits = Enum.map(results, fn {_s, r} -> erpc(node, :eta_run, :audit, [r]) end)
        dirty = audits |> Enum.reject(&(&1 == :ok)) |> Enum.uniq() |> Enum.take(3)

        {:ok,
         "0 violations in #{length(seeds)} seeds (#{ops(opts)} ops each).\n" <>
           if(dirty == [],
             do: "audit clean: the scheduler owned every decision.",
             else:
               "AUDIT NOT CLEAN — this green means less than it looks:\n  " <>
                 inspect(dirty, limit: 10, printable_limit: 400) <>
                 "\n  A run with determinism holes did not explore what it claims to."
           ) <>
           "\n\nA harness that has never been shown to fail is not evidence. " <>
           "Break the code deliberately and confirm this goes red."}

      [{seed, outcome, result} | _] ->
        shrunk = shrink(node, mod, result, Map.put(opts, :seed, seed))
        {:ok,
         "VIOLATION on seed #{seed}: #{inspect(outcome)}\n" <>
           "(#{length(failures)} of #{length(seeds)} seeds failed)\n\n" <>
           shrunk}
    end
  end

  # Shrinking is not optional. A raw trace is thousands of decisions and would
  # consume the context window to no purpose (PLAN.md 9.4).
  defp shrink(node, mod, %{trace: trace}, opts) do
    case erpc(node, :eta_shrink, :shrink, [mod, trace, erl_opts(opts)], @run_timeout_ms) do
      %{trace: minimal} = r ->
        verified = Map.get(r, :verified, false)

        "minimal trace (#{length(minimal)} steps, " <>
          "#{if verified, do: "verified still fails", else: "NOT verified"}):\n" <>
          render(minimal) <>
          "\n\nreplay it with: replay(trace: <the id above>)"

      other ->
        # Say why. Swallowing the reason here is the same mistake as reporting
        # a run that never happened as a pass: the model is told less than is
        # known, and cannot act on what it is not told.
        "could not shrink (#{describe_error(other)}); the raw trace is " <>
          "#{length(trace)} steps:\n" <> render(Enum.take(trace, 24)) <>
          if(length(trace) > 24, do: "\n  ... #{length(trace) - 24} more", else: "")
    end
  end

  defp render(trace) do
    trace
    |> Enum.map(fn
      {:op, op} -> "  op    #{inspect(op)}"
      {:step, id} -> "  step  #{inspect(id)}"
      {:clock, t} -> "  clock #{t}"
      other -> "  #{inspect(other)}"
    end)
    |> Enum.join("\n")
  end

  defp audit_line(node, result) do
    case erpc(node, :eta_run, :audit, [result]) do
      :ok -> "audit clean."
      other -> "audit: #{inspect(other)} — the run had determinism holes."
    end
  end

  # ---- the node ------------------------------------------------------------

  defp start_node do
    n = System.unique_integer([:positive])
    [_, host] = node() |> Atom.to_string() |> String.split("@", parts: 2)
    name = "sim_#{n}"
    sim_node = :"#{name}@#{host}"
    libs = String.trim(File.read!("/etc/crucible-erl-libs"))

    port =
      Port.open({:spawn_executable, System.find_executable("erl") || "/usr/bin/erl"}, [
        {:args, ["-noshell", "-noinput", "-sname", name, "-boot", "no_dot_erlang",
                 "-eval", "application:ensure_all_started(elixir), 'Elixir.Crucible.Workspace':main()."]},
        {:env, [{~c"ERL_LIBS", String.to_charlist("#{@sim_lib}:#{libs}")}]},
        :binary,
        :exit_status
      ])

    case await(sim_node, @boot_timeout_ms) do
      :ok -> {:ok, sim_node, port}
      :timeout ->
        Port.close(port)
        {:error, "sim_boot", "the simulation node did not start within #{@boot_timeout_ms}ms"}
    end
  end

  defp await(node, remaining) when remaining > 0 do
    case :erpc.call(node, Crucible.Workspace, :ping, [], 1_000) do
      {:pong, _, _} -> :ok
      _ -> retry(node, remaining)
    end
  catch
    _, _ -> retry(node, remaining)
  end

  defp await(_node, _), do: :timeout

  defp retry(node, remaining) do
    # There is no simulation here to virtualize: this module is what *starts*
    # one, and until the node answers there is nothing to schedule.
    Process.sleep(150)  # lint:real-clock polling a real OS node into existence
    await(node, remaining - 150)
  end

  defp stop_node(node, port) do
    try do
      :erpc.cast(node, :erlang, :halt, [0])
    catch
      _, _ -> :ok
    end

    try do
      Port.close(port)
    rescue
      _ -> :ok
    end
  end

  # The agent's modules, compiled onto the sim node in definition order. They
  # are compiled here rather than copied as beams so that anything that
  # `use Eta` picks up the instrumented macros -- the sim node has eta on its
  # path and the workspace does not.
  defp load_sources(_node, []), do: :ok

  defp load_sources(node, sources) do
    Enum.reduce_while(sources, :ok, fn src, _ ->
      case erpc(node, Crucible.Loader, :compile, [src], 30_000) do
        {:ok, mods, _warnings} ->
          case erpc(node, Crucible.Loader, :load, [mods], 10_000) do
            {:ok, _} -> {:cont, :ok}
            other -> {:halt, {:error, "sim_load", inspect(other)}}
          end

        {:error, msg, _} ->
          {:halt, {:error, "sim_compile", msg}}

        other ->
          {:halt, {:error, "sim_compile", inspect(other)}}
      end
    end)
  end

  defp module(name) do
    String.to_atom("Elixir." <> String.replace_prefix(name, "Elixir.", ""))
  end

  defp ops(opts), do: Map.get(opts, :max_ops, 25)

  defp erl_opts(opts) do
    %{
      seed: Map.get(opts, :seed, 1),
      max_ops: Map.get(opts, :max_ops, 25),
      max_steps: Map.get(opts, :max_steps, 20_000)
    }
  end

  defp erpc(node, mod, fun, args, timeout \\ 15_000) do
    :erpc.call(node, mod, fun, args, timeout)
  catch
    :error, {:erpc, :noconnection} -> {:error, "sim_down", "the simulation node died"}
    :error, {:erpc, :timeout} -> {:error, "timeout", "the simulation exceeded #{timeout}ms"}
    kind, reason -> {:error, "sim", Exception.format(kind, reason, [])}
  end
end
