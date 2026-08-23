defmodule Warden.Workspace do
  @moduledoc """
  Starts the agent's node, watches it, and brings it back.

  PLAN.md 7.3, and the reason it is a separate OS process rather than a
  supervision tree inside warden: a module the agent hot-loads can crash the
  scheduler, exhaust the atom table, or take the node down. If that node also
  owned the vsock bridge, the harness would go silent and the user would see a
  hang.

  ## What lives where, corrected

  §7.3's table put `Crucible.Fs` — the six file tools — on the workspace node.
  That is wrong, and M3 moves them to warden: **an agent that has broken its own
  workspace must still be able to read and edit files in order to fix it.** A
  recovery path that runs on the thing that broke is not a recovery path. So:

    * **warden** — the bridge, the six file tools, this lifecycle, and (M4) the
      registry, because the manifest is durable state the agent must not be able
      to corrupt by crashing.
    * **workspace** — the agent's own modules, the eval shell, and whatever
      `invoke` dispatches into.

  ## Restarts

  Exponential backoff, capped. A workspace that cannot start is a fact to report
  to the model, not something to hammer: the host is told through an unsolicited
  event, and the next `eval` says plainly that the node is down.
  """
  use Warden.Sim, gen_server: true

  @eta_observe :all

  @backoff_ms [200, 500, 1_000, 2_000, 5_000]

  # Derived from warden's own node, never hardcoded.
  #
  # `-sname workspace` takes its host part from whatever the resolver calls this
  # machine, and with `127.0.0.1 localhost crucible` in /etc/hosts that is
  # `localhost`, not the hostname init set. A hardcoded `workspace@crucible`
  # therefore names a node that does not exist, and the symptom is a workspace
  # that starts perfectly and is never reachable. Both nodes are started by the
  # same init on the same host, so warden's own suffix is the right one by
  # construction.
  def node_name do
    [_, host] = node() |> Atom.to_string() |> String.split("@", parts: 2)
    :"workspace@#{host}"
  end

  defstruct port: nil, restarts: 0, last_exit: nil, started_at: nil, enabled: true

  # ---- api -----------------------------------------------------------------

  def start_link(opts \\ []), do: start_named(__MODULE__, opts)

  defp start_named(module, opts) do
    case Keyword.fetch(opts, :name) do
      {:ok, nil} -> GenServer.start_link(module, opts)
      {:ok, name} -> GenServer.start_link(module, opts, name: name)
      :error -> GenServer.start_link(module, opts, name: module)
    end
  end

  @doc "Whether the node answers right now, with how long it took."
  def status(server \\ __MODULE__), do: Sim.call(server, :status, 10_000)

  @doc """
  Evaluates Elixir on the workspace node.

  The seed of M4's `elixir` tool. It runs `erpc` under a timeout, so a crash
  there is a failed call here rather than a warden that waits for ever — which
  is invariant 6 applied to the node boundary the whole design turns on.
  """
  def eval(server \\ __MODULE__, code, timeout_ms),
    do: Sim.call(server, {:eval, code, timeout_ms}, timeout_ms + 5_000)

  @doc "Calls a registered tool on the agent's node."
  def invoke(module, args, timeout_ms) do
    mod = String.to_existing_atom("Elixir." <> String.replace_prefix(module, "Elixir.", ""))

    :erpc.call(node_name(), Crucible.Loader, :invoke, [mod, args], timeout_ms)
  catch
    :error, {:erpc, :noconnection} ->
      {:error, "workspace_down", "the workspace node is not reachable; it restarts on its own"}

    :error, {:erpc, :timeout} ->
      {:error, "timeout", "the tool ran for #{timeout_ms}ms without returning"}

    kind, reason ->
      {:error, "invoke", Exception.format(kind, reason, [])}
  end

  # ---- callbacks -----------------------------------------------------------

  @impl :gen_server
  def init(opts) do
    Sim.label(:warden_workspace)
    Process.flag(:trap_exit, true)
    enabled = Keyword.get(opts, :enabled, true)
    state = %__MODULE__{enabled: enabled}
    {:ok, if(enabled, do: spawn_node(state), else: state)}
  end

  @impl :gen_server
  def handle_call(:status, _from, state) do
    {:reply, describe(state), state}
  end

  def handle_call({:eval, code, timeout_ms}, _from, state) do
    reply =
      try do
        case :erpc.call(node_name(), Code, :eval_string, [code], timeout_ms) do
          {value, _binding} -> {:ok, inspect(value, pretty: true, limit: 50)}
        end
      catch
        :error, {:erpc, :noconnection} ->
          {:error, "workspace_down", "the workspace node is not reachable"}

        :error, {:erpc, :timeout} ->
          {:error, "timeout", "the workspace did not answer within #{timeout_ms}ms"}

        kind, reason ->
          {:error, "eval", Exception.format(kind, reason, [])}
      end

    {:reply, reply, state}
  end

  def handle_call(_other, _from, state), do: {:reply, {:error, :unknown_call}, state}

  @impl :gen_server
  def handle_info({port, {:exit_status, status}}, %{port: port} = state) do
    # The node died. Report it, then bring it back -- and say how many times,
    # because a workspace that keeps dying is information the model needs.
    log("workspace exited with #{status}; restarting")
    state = %{state | port: nil, restarts: state.restarts + 1, last_exit: status}
    Sim.send_after(self(), :respawn, backoff(state.restarts))
    {:noreply, state}
  end

  def handle_info({_port, {:data, data}}, state) do
    # The node's stderr, forwarded to the console so a startup failure there is
    # visible in the guest log rather than swallowed.
    IO.write(:stderr, data)
    {:noreply, state}
  end

  def handle_info(:respawn, state), do: {:noreply, spawn_node(state)}

  # After a restart the node is empty: every module the agent defined lives only
  # in warden's registry now. Replaying it is what makes self-modification
  # survive a crash (PLAN.md 7.3), and it has to happen before the model's next
  # `invoke`, not when something notices the tool is missing.
  def handle_info(:verify_up, state) do
    reachable =
      try do
        match?({:pong, _, _}, :erpc.call(node_name(), Crucible.Workspace, :ping, [], 2_000))
      catch
        _, _ -> false
      end

    cond do
      not reachable ->
        Sim.send_after(self(), :verify_up, 300)
        {:noreply, state}

      state.restarts > 0 ->
        {:noreply, replay(state)}

      true ->
        {:noreply, state}
    end
  end

  def handle_info({:EXIT, _port, _reason}, state), do: {:noreply, state}
  def handle_info(_other, state), do: {:noreply, state}

  @impl :gen_server
  def handle_cast(_msg, state), do: {:noreply, state}

  # ---- the node ------------------------------------------------------------

  defp spawn_node(%{enabled: false} = state), do: state

  defp spawn_node(state) do
    erl = System.find_executable("erl") || "/usr/bin/erl"
    libs = String.trim(File.read!("/etc/crucible-erl-libs"))

    port =
      Port.open({:spawn_executable, erl}, [
        {:args,
         [
           # -noinput, not just -noshell. A node started as an Erlang port has
           # its stdin wired to a pipe, and `-noshell` alone leaves the BEAM
           # reading it -- so the node exits the moment that stdin does anything
           # it does not like, with status 1 and no diagnostic whatsoever. The
           # symptom was a workspace that came up, announced itself, and died
           # 330ms later, three times, before happening to survive.
           "-noshell",
           "-noinput",
           "-sname", "workspace",
           "-boot", "no_dot_erlang",
           "-pa", "/opt/warden/ebin",
           "-eval",
           "application:ensure_all_started(elixir), 'Elixir.Crucible.Workspace':main()."
         ]},
        {:env, [{~c"ERL_LIBS", String.to_charlist(libs)}]},
        :binary,
        :exit_status
        # NOT :stderr_to_stdout: the node's own diagnostics go straight to the
        # guest console, where a startup failure is readable. Routing them
        # through this process meant an exit reason arrived as {:data, ...}
        # after the {:exit_status, ...} that ended the port, and was lost.
      ])

    log("workspace starting (#{node_name()})")
    Sim.send_after(self(), :verify_up, 300)
    %{state | port: port, started_at: System.monotonic_time(:millisecond)}
  rescue
    e ->
      log("cannot start the workspace: #{Exception.message(e)}")
      state
  end

  # Replay in definition order: a module defined later may call one defined
  # earlier, and compilation order is the only ordering information there is
  # (PLAN.md 9.2, invariant 4).
  defp replay(state) do
    sources = Warden.Registry.manifest()

    if sources == [] do
      state
    else
      results = Enum.map(sources, &Warden.Define.run(&1, force: true))
      failed = Enum.count(results, &match?({:error, _, _}, &1))
      n = Warden.Registry.note_replay()

      log(
        "replayed #{length(sources)} module(s) after restart " <>
          "(#{failed} failed, replay ##{n})"
      )

      state
    end
  rescue
    e ->
      log("replay failed: #{Exception.message(e)}")
      state
  end

  defp backoff(n), do: Enum.at(@backoff_ms, min(n - 1, length(@backoff_ms) - 1))

  defp describe(state) do
    reachable =
      try do
        case :erpc.call(node_name(), Crucible.Workspace, :ping, [], 2_000) do
          {:pong, n, otp} -> "up node=#{n} otp=#{otp}"
        end
      catch
        _, _ -> "down"
      end

    uptime =
      case state.started_at do
        nil -> 0
        t -> div(System.monotonic_time(:millisecond) - t, 1000)
      end

    "#{reachable} restarts=#{state.restarts} uptime_s=#{uptime}" <>
      if(state.last_exit, do: " last_exit=#{state.last_exit}", else: "")
  end

  defp log(msg) do
    up =
      case File.read("/proc/uptime") do
        {:ok, s} -> s |> String.split() |> hd()
        _ -> "?"
      end

    IO.puts(:stderr, "[#{up}] warden: #{msg}")
  end
end
