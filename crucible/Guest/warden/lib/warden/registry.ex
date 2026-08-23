defmodule Warden.Registry do
  @moduledoc """
  Everything the agent has defined, kept where the agent cannot lose it.

  PLAN.md 7.3: the manifest is the durable artefact. It lives in **warden**, not
  in the workspace, and that placement is the whole point — the workspace is the
  node that crashes, and a registry stored there would vanish with exactly the
  code it was meant to make recoverable.

  Two invariants it exists to hold (PLAN.md 9.2, 3 and 4):

    * **No two modules claim the same tool name.** A second `define` for an
      existing name replaces it and bumps a version; two *different* modules
      claiming one name is refused, because `invoke` would then be ambiguous and
      the model would have no way to say which it meant.
    * **Replay is order-preserving.** A workspace restart reloads every module
      in the order it was first defined, because later modules may reference
      earlier ones and compilation order is the only ordering information there
      is.
  """
  use Warden.Sim, gen_server: true

  @eta_observe :all

  defstruct entries: %{}, order: [], replays: 0

  # ---- api -----------------------------------------------------------------

  def start_link(opts \\ []) do
    case Keyword.fetch(opts, :name) do
      {:ok, nil} -> GenServer.start_link(__MODULE__, opts)
      {:ok, name} -> GenServer.start_link(__MODULE__, opts, name: name)
      :error -> GenServer.start_link(__MODULE__, opts, name: __MODULE__)
    end
  end

  def put(server \\ __MODULE__, entry), do: Sim.call(server, {:put, entry}, 10_000)
  def lookup(server \\ __MODULE__, tool_name), do: Sim.call(server, {:lookup, tool_name}, 10_000)
  def list(server \\ __MODULE__), do: Sim.call(server, :list, 10_000)
  def manifest(server \\ __MODULE__), do: Sim.call(server, :manifest, 10_000)
  def note_replay(server \\ __MODULE__), do: Sim.call(server, :note_replay, 10_000)

  # ---- callbacks -----------------------------------------------------------

  @impl :gen_server
  def init(_opts) do
    Sim.label(:warden_registry)
    {:ok, %__MODULE__{}}
  end

  @impl :gen_server
  def handle_call({:put, entry}, _from, state) do
    key = entry.module

    conflict =
      entry.tool_name &&
        Enum.find(state.entries, fn {m, e} ->
          m != key and e.tool_name == entry.tool_name
        end)

    case conflict do
      {other, _} ->
        {:reply,
         {:error,
          "the tool name #{inspect(entry.tool_name)} is already claimed by #{other}. " <>
            "Two modules cannot answer to one name; rename one of them."}, state}

      nil ->
        prior = Map.get(state.entries, key)
        version = if prior, do: prior.version + 1, else: 1
        entry = Map.put(entry, :version, version)

        order = if prior, do: state.order, else: state.order ++ [key]
        entries = Map.put(state.entries, key, entry)
        {:reply, {:ok, version}, %{state | entries: entries, order: order}}
    end
  end

  def handle_call({:lookup, tool_name}, _from, state) do
    found =
      Enum.find_value(state.entries, fn {_m, e} ->
        if e.tool_name == tool_name, do: e
      end)

    {:reply, found, state}
  end

  def handle_call(:list, _from, state) do
    {:reply, Enum.map(state.order, &Map.fetch!(state.entries, &1)), state}
  end

  # In definition order, which is what a replay has to follow: a module defined
  # later may call one defined earlier.
  def handle_call(:manifest, _from, state) do
    {:reply, Enum.map(state.order, &Map.fetch!(state.entries, &1).source), state}
  end

  def handle_call(:note_replay, _from, state),
    do: {:reply, state.replays + 1, %{state | replays: state.replays + 1}}

  def handle_call(_other, _from, state), do: {:reply, {:error, :unknown_call}, state}

  @impl :gen_server
  def handle_cast(_msg, state), do: {:noreply, state}

  @impl :gen_server
  def handle_info(_msg, state), do: {:noreply, state}
end
