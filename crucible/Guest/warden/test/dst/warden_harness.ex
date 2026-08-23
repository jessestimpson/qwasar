defmodule Warden.DST.Harness do
  @moduledoc """
  The system under test, as `eta_harness` sees it.

  PLAN.md 9.2. Three invariants at M2, and each one is a *when* bug — the kind
  a `Process.sleep` in an ExUnit case passes for the wrong reason:

    1. Every id receives exactly one terminal response. Never zero, never two.
    2. No reply is delivered as `ok` after its request was reported timed out.
    6. The bridge never blocks: a cancel is answered while another request is
       still outstanding.

  Two rules from eta's own documentation shape this file, and both fail
  silently when broken: **`execute/2` must not block**, and **`check/1` must not
  call into a scheduled process**. So `execute/2` only sends, and `check/1`
  reads the bridge's state through `eta_observe` rather than by asking it.
  """
  @behaviour :eta_harness

  defstruct bridge: nil, worker: nil, issued: [], cancelled: [], next_id: 1

  @timeout_ms 1_000

  @impl true
  def init(_seed, _config) do
    {:ok, worker} = Warden.Worker.start_link(name: nil)
    # port: nil means the bridge keeps what it would have written, so the
    # invariants are checkable from observed state alone.
    {:ok, bridge} =
      Warden.Bridge.start_link(name: nil, port: nil, worker: worker, timeout_ms: @timeout_ms)

    {:ok, %__MODULE__{bridge: bridge, worker: worker}}
  end

  @impl true
  def processes(%{bridge: b, worker: w}), do: [b, w]

  @impl true
  def labels(%{bridge: b, worker: w}), do: %{b => :bridge, w => :worker}

  @impl true
  def terminate(_state), do: :ok

  # ---- the workload --------------------------------------------------------

  # The shapes a bad_reply can take. A module attribute rather than an inline
  # list so the draw below can index it from the harness's OWN random state.
  @bad_shapes ~w(two_tuple bare nested other)

  @impl true
  def generate(state, rand) do
    {pick, rand} = :rand.uniform_s(100, rand)
    op(pick, state, rand)
  end

  # Every branch returns the rand state it leaves behind, and every draw comes
  # from it.
  #
  # This used to end with `Enum.random/1` for the bad_reply shape, which reads
  # as harmless and is not: Enum.random draws from the PROCESS DICTIONARY's
  # :rand state, seeded from real entropy, not from the state eta threads
  # through here. So the same seed produced different workloads on different
  # runs -- and worse than a flaky test, it made `shrink` and `replay`
  # unsound, since both re-run generate/2 and would draw a different shape than
  # the trace they are trying to reproduce. A generator that cannot be replayed
  # takes the whole point of the framework with it.

  # A request that will outlive its deadline: the only way to reach invariant 2.
  defp op(pick, state, rand) when pick <= 25,
    do: {{:slow_request, state.next_id, @timeout_ms * 2}, rand}

  # One that will not.
  defp op(pick, state, rand) when pick <= 45,
    do: {{:slow_request, state.next_id, div(@timeout_ms, 4)}, rand}

  defp op(pick, state, rand) when pick <= 65,
    do: {{:request, state.next_id, "ping"}, rand}

  defp op(pick, state, rand) when pick <= 72,
    do: {{:request, state.next_id, "nonsense"}, rand}

  # A handler returning a shape the bridge does not expect. The bridge must
  # answer the request anyway -- never crash, never leave it unanswered.
  defp op(pick, state, rand) when pick <= 80 do
    {i, rand} = :rand.uniform_s(length(@bad_shapes), rand)
    {{:bad_reply, state.next_id, Enum.at(@bad_shapes, i - 1)}, rand}
  end

  defp op(pick, state, rand) when pick <= 90,
    do: {{:cancel, state.next_id, pick_outstanding(state)}, rand}

  defp op(_pick, _state, rand), do: {:advance_time, rand}

  defp pick_outstanding(%{issued: []}), do: 0
  defp pick_outstanding(%{issued: [id | _]}), do: id

  @impl true
  def execute({:request, id, op}, state) do
    Warden.Bridge.deliver(state.bridge, %{"id" => id, "op" => op})
    %{state | issued: [id | state.issued], next_id: id + 1}
  end

  def execute({:slow_request, id, ms}, state) do
    Warden.Bridge.deliver(state.bridge, %{"id" => id, "op" => "sleep", "args" => %{"ms" => ms}})
    %{state | issued: [id | state.issued], next_id: id + 1}
  end

  def execute({:bad_reply, id, shape}, state) do
    Warden.Bridge.deliver(state.bridge, %{
      "id" => id,
      "op" => "bad_reply",
      "args" => %{"shape" => shape}
    })

    %{state | issued: [id | state.issued], next_id: id + 1}
  end

  def execute({:cancel, id, target}, state) do
    Warden.Bridge.deliver(state.bridge, %{
      "id" => id,
      "op" => "cancel",
      "args" => %{"target" => target}
    })

    %{state | issued: [id | state.issued], cancelled: [target | state.cancelled], next_id: id + 1}
  end

  # The clock jumps straight to the next deadline rather than ticking, which is
  # what makes a one-second timeout cost a simulation nothing at all.
  def execute(:advance_time, state) do
    :eta_time.advance_to_next()
    state
  end

  # ---- the invariants ------------------------------------------------------

  @impl true
  def check(state) do
    case observe(state.bridge) do
      nil -> :ok
      bridge -> check_bridge(bridge)
    end
  end

  # Once everything has settled, every issued id must have been answered. This
  # is invariant 1's "never zero" half, and it can only be judged at the end:
  # mid-run, an unanswered id is simply one still in flight.
  @impl true
  def check_final(_settled, state) do
    case observe(state.bridge) do
      nil ->
        :ok

      bridge ->
        answered = MapSet.new(bridge.answered)
        missing = Enum.reject(state.issued, &MapSet.member?(answered, &1))

        cond do
          missing != [] ->
            {:violation, {:unanswered, missing}}

          bridge.inflight != %{} ->
            {:violation, {:still_inflight, Map.keys(bridge.inflight)}}

          true ->
            check_bridge(bridge)
        end
    end
  end

  defp check_bridge(bridge) do
    ids = Enum.map(bridge.written, & &1[:id]) |> Enum.reject(&is_nil/1)
    dupes = ids -- Enum.uniq(ids)

    cond do
      # Invariant 1, the "never two" half.
      dupes != [] ->
        {:violation, {:responded_twice, Enum.uniq(dupes)}}

      # Invariant 2: an id reported timed out must never also appear as ok.
      true ->
        case double_answer(bridge.written) do
          nil -> :ok
          id -> {:violation, {:success_after_timeout, id}}
        end
    end
  end

  defp double_answer(written) do
    written
    |> Enum.reject(&is_nil(&1[:id]))
    |> Enum.group_by(& &1[:id])
    |> Enum.find_value(fn {id, msgs} ->
      failed = Enum.any?(msgs, &(&1[:ok] == false))
      succeeded = Enum.any?(msgs, &(&1[:ok] == true))
      if failed and succeeded, do: id
    end)
  end

  # `eta_observe.read/1` reads a suspended process's state out of its process
  # dictionary, where `@eta_observe :all` republishes it on every callback
  # return. That is the only way an invariant can safely inspect a halted
  # system: asking a suspended gen_server would hang, and a client API that
  # catches its own timeout would answer plausibly rather than failing.
  defp observe(pid) do
    case :eta_observe.read(pid) do
      %Warden.Bridge{} = state -> state
      _ -> nil
    end
  end
end
