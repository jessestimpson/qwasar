defmodule Warden.Worker do
  @moduledoc """
  Where a request actually runs, so that the bridge never has to wait for one.

  PLAN.md 9.2, invariant 6: *warden never blocks on the workspace — the bridge
  remains responsive to a cancel while an invoke is outstanding.* At M2 there is
  no workspace yet, but the shape has to be right now: `Warden.Bridge` used to
  call `Warden.Dispatch.run/1` inline, which meant a single slow op stalled the
  whole control channel — and no amount of testing would have found that,
  because every op it had was instant.

  A long-lived process rather than a task per request, deliberately. eta's
  scheduler controls a fixed set of processes (`processes/1` in the harness), so
  a design that spawned per request would be a design the simulator cannot see.
  M4's `invoke` gets its concurrency from the workspace node on the far side of
  distribution, not from spawning here.
  """
  use GenServer

  defstruct pending: %{}

  def start_link(opts \\ []), do: start_named(__MODULE__, opts)

  # `name: nil` means "do not register", which is what a simulation needs: each
  # seed starts a fresh instance, and a module-name registration would make the
  # second one fail with {:error, {:already_started, _}}. Omitting :name keeps
  # the ordinary singleton behaviour the supervisor relies on.
  defp start_named(module, opts) do
    case Keyword.fetch(opts, :name) do
      {:ok, nil} -> GenServer.start_link(module, opts)
      {:ok, name} -> GenServer.start_link(module, opts, name: name)
      :error -> GenServer.start_link(module, opts, name: module)
    end
  end


  @doc "Runs `req` for `id` and casts the reply back to `reply_to`."
  def run(server, id, req, reply_to), do: GenServer.cast(server, {:run, id, req, reply_to})

  @impl GenServer
  def init(_opts) do
    {:ok, %__MODULE__{}}
  end

  @impl GenServer
  def handle_cast({:run, id, req, reply_to}, state) do
    # `sleep` exists so a test can make a request outlive its deadline, which
    # is the only way to exercise the rule that a late reply must not be
    # delivered as success after the request was already reported timed out.
    case req do
      %{"op" => "sleep", "args" => %{"ms" => ms}} when is_integer(ms) ->
        Process.send_after(self(), {:wake, id, reply_to}, ms)
        {:noreply, %{state | pending: Map.put(state.pending, id, reply_to)}}

      # A reply shape the bridge does not expect. This exists because the
      # bridge once died of one: a tool returned {:error, msg} where respond/3
      # matched only {:ok, _} and {:error, _, _}, the CaseClauseError killed the
      # bridge inside handle_cast, and the request that caused it got NO
      # response -- invariant 1 broken by a shape rather than by a race.
      #
      # The simulation did not catch it, because generate/2 only ever produced
      # well-formed ops. A harness can only find what its generator can express,
      # so the generator now expresses this.
      %{"op" => "bad_reply", "args" => %{"shape" => shape}} ->
        reply =
          case shape do
            "two_tuple" -> {:error, "a two-tuple error"}
            "bare" -> :not_a_tuple
            "nested" -> {:ok, {:unexpected, [1, 2, 3]}}
            _ -> {:ok, 12_345}
          end

        GenServer.cast(reply_to, {:done, id, reply})
        {:noreply, state}

      _ ->
        reply = Warden.Dispatch.run(req)
        GenServer.cast(reply_to, {:done, id, reply})
        {:noreply, state}
    end
  end

  def handle_cast(_other, state), do: {:noreply, state}

  # Required by :gen_server.
  @impl GenServer
  def handle_call(:health, _from, state),
    do: {:reply, %{pending: map_size(state.pending)}, state}

  def handle_call(_other, _from, state), do: {:reply, {:error, :unknown_call}, state}

  @impl GenServer
  def handle_info({:wake, id, reply_to}, state) do
    GenServer.cast(reply_to, {:done, id, {:ok, "slept"}})
    {:noreply, %{state | pending: Map.delete(state.pending, id)}}
  end

  def handle_info(_other, state), do: {:noreply, state}
end
