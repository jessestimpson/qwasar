defmodule Warden.Bridge do
  @moduledoc """
  The vsock control channel: one framed request in, exactly one response out.

  PLAN.md 6.4. The framing is Erlang's own — `{:packet, 4}` writes a four-byte
  big-endian length ahead of each message on the port side and strips it on the
  way in — so `vsock_port` stays a byte pipe and nothing here parses a length.

  Two rules keep the channel alive under load:

    * **Nothing here blocks.** A request is handed to `Warden.Worker` and the
      bridge goes straight back to its mailbox, so a slow op cannot stall the
      channel.
    * **Every deadline is a `Process.send_after/3`** matched in a plain
      `receive`, never a `receive ... after`, so a late reply is told apart
      from a timeout by message identity rather than by racing the clock.
  """
  use GenServer

  @default_timeout_ms 30_000

  defstruct port: nil,
            worker: Warden.Worker,
            timeout_ms: @default_timeout_ms,
            # id => %{started_at, timer}
            inflight: %{},
            # ids already answered, in order. The invariant "exactly one
            # terminal response per id" is checkable from this alone, which is
            # why it is state rather than a side effect.
            answered: [],
            # ids reported timed out. A reply arriving after this must be
            # dropped, not delivered as success (invariant 2).
            expired: MapSet.new(),
            # What was written, when there is no real port. Simulation only.
            written: []

  # ---- api -----------------------------------------------------------------

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


  @doc "Injects a frame as though it arrived from the host. Simulation only."
  def deliver(server, frame), do: Kernel.send(server, {:frame, frame})

  # ---- callbacks -----------------------------------------------------------

  @impl GenServer
  def init(opts) do

    port =
      case Keyword.fetch(opts, :port) do
        {:ok, given} -> given
        :error -> open_vsock(Keyword.get(opts, :vsock_port, "1024"))
      end

    state = %__MODULE__{
      port: port,
      worker: Keyword.get(opts, :worker, Warden.Worker),
      timeout_ms: Keyword.get(opts, :timeout_ms, @default_timeout_ms)
    }

    log("bridge open on OTP #{:erlang.system_info(:otp_release)}")
    {:ok, state}
  end

  @impl GenServer
  def handle_info({port, {:data, frame}}, %{port: port} = state),
    do: {:noreply, handle_frame(frame, state)}

  # A frame injected by a simulation, with no real port behind it.
  def handle_info({:frame, frame}, state), do: {:noreply, handle_frame(frame, state)}

  def handle_info({port, {:exit_status, status}}, %{port: port} = state) do
    log("bridge exited with #{status}")
    {:stop, {:bridge_exit, status}, state}
  end

  # The worker finished. If the request already expired, this is a late reply
  # and must be dropped: the host has been told it timed out, and telling it
  # afterwards that the same id succeeded is the classic bug invariant 2 exists
  # to catch. It is invisible without a clock the test controls.
  def handle_info({:"$gen_cast", {:done, id, reply}}, state), do: {:noreply, done(state, id, reply)}

  def handle_info({:timeout_fired, id}, state) do
    if Map.has_key?(state.inflight, id) do
      state = %{state | expired: MapSet.put(state.expired, id)}
      {:noreply,
       respond(state, id, {:error, "timeout", "the request exceeded #{state.timeout_ms}ms"})}
    else
      {:noreply, state}
    end
  end

  def handle_info({:EXIT, _port, :normal}, state), do: {:noreply, state}

  def handle_info(other, state) do
    log("unexpected: #{inspect(other)}")
    {:noreply, state}
  end

  @impl GenServer
  def handle_cast({:done, id, reply}, state), do: {:noreply, done(state, id, reply)}
  def handle_cast(_other, state), do: {:noreply, state}

  # Required by :gen_server, and genuinely useful: a health probe that does not
  # go through the wire.
  @impl GenServer
  def handle_call(:health, _from, state),
    do: {:reply, %{inflight: map_size(state.inflight), answered: length(state.answered)}, state}

  def handle_call(_other, _from, state), do: {:reply, {:error, :unknown_call}, state}

  # ---- the contract --------------------------------------------------------

  # PLAN.md 9.2, invariant 1: every id receives exactly one terminal response.
  # Never zero — an op this warden does not understand is ANSWERED, because
  # silence would leave the host waiting out a timeout for a mistake it could
  # have been told about at once. Never two — which is what `answered` and the
  # deletion from `inflight` together guarantee.
  defp handle_frame(frame, state) do
    case decode(frame) do
      {:ok, %{"id" => id, "op" => "cancel", "args" => %{"target" => target}}}
      when is_integer(id) and is_integer(target) ->
        # Cancelling is itself a request and gets its own response, which is how
        # a simulation proves the channel stayed responsive while something else
        # was outstanding.
        state = cancel(state, target)
        start(state, id, %{"op" => "__cancel_ack"})

      {:ok, %{"id" => id} = req} when is_integer(id) ->
        start(state, id, req)

      {:ok, _} ->
        emit(state, "warn", "request without an integer id was ignored")

      {:error, why} ->
        emit(state, "error", "undecodable frame: #{why}")
    end
  end

  defp start(state, id, req) do
    cond do
      Map.has_key?(state.inflight, id) ->
        # A repeated id is the host's bug, and answering it twice would be ours.
        emit(state, "warn", "duplicate in-flight id #{id} ignored")

      id in state.answered ->
        emit(state, "warn", "id #{id} was already answered")

      req["op"] == "__cancel_ack" ->
        respond(%{state | inflight: Map.put(state.inflight, id, entry(state, id))}, id,
                {:ok, "cancelled"})

      true ->
        state = %{state | inflight: Map.put(state.inflight, id, entry(state, id))}
        Warden.Worker.run(state.worker, id, req, self())
        state
    end
  end

  defp entry(state, id) do
    %{
      started_at: System.monotonic_time(:millisecond),
      timer: Process.send_after(self(), {:timeout_fired, id}, state.timeout_ms)
    }
  end

  defp done(state, id, reply) do
    cond do
      MapSet.member?(state.expired, id) ->
        # Late. The host already has its answer.
        emit(state, "warn", "dropped a late reply for id #{id}")

      Map.has_key?(state.inflight, id) ->
        respond(state, id, reply)

      true ->
        state
    end
  end

  defp cancel(state, target) do
    case Map.fetch(state.inflight, target) do
      {:ok, _} ->
        state = %{state | expired: MapSet.put(state.expired, target)}
        respond(state, target, {:error, "cancelled", "cancelled by the host"})

      :error ->
        state
    end
  end

  defp respond(state, id, reply) do
    {entry, inflight} = Map.pop(state.inflight, id)
    if entry, do: Process.cancel_timer(entry.timer)

    took =
      case entry do
        nil -> 0
        %{started_at: t} -> System.monotonic_time(:millisecond) - t
      end

    payload =
      case reply do
        {:ok, result} when is_binary(result) ->
          %{id: id, ok: true, result: result, took_ms: took}

        {:error, kind, msg} ->
          %{id: id, ok: false, error: to_string(msg), kind: to_string(kind), took_ms: took}

        # A two-tuple error, which is what a tool's own {:error, message} looks
        # like by the time it reaches here. This clause exists because its
        # absence took the entire control channel down: `respond/3` raised a
        # CaseClauseError inside handle_cast, the bridge terminated, and the
        # request that caused it received NO response at all -- invariant 1
        # broken by a reply shape rather than by a race.
        {:error, msg} ->
          %{id: id, ok: false, error: to_string(msg), kind: "tool", took_ms: took}

        # And a catch-all, because the bridge must not be able to die of
        # something a tool returned. Anything unrecognised is reported as what
        # it is rather than crashed on.
        other ->
          %{id: id, ok: false, kind: "malformed_reply",
            error: "the handler returned an unexpected shape: " <> inspect(other),
            took_ms: took}
      end

    %{state | inflight: inflight, answered: [id | state.answered]}
    |> write(payload)
  end

  # Unsolicited events carry no id, which is how the host tells them from
  # replies without needing a second channel.
  defp emit(state, level, text), do: write(state, %{event: "log", level: level, text: text})

  # When there is no port, what would have been written is kept instead, so a
  # test can assert on it.
  defp write(%{port: nil} = state, payload),
    do: %{state | written: state.written ++ [payload]}

  defp write(%{port: port} = state, payload) when is_port(port) do
    Port.command(port, :erlang.iolist_to_binary(:json.encode(payload)))
    state
  end

  defp write(%{port: pid} = state, payload) when is_pid(pid) do
    Kernel.send(pid, {:wrote, payload})
    %{state | written: state.written ++ [payload]}
  end

  defp decode(frame) when is_map(frame), do: {:ok, frame}

  defp decode(frame) do
    case :json.decode(frame) do
      map when is_map(map) -> {:ok, map}
      other -> {:error, "expected an object, got #{inspect(other)}"}
    end
  rescue
    e -> {:error, Exception.message(e)}
  end

  defp open_vsock(vsock_port) do
    Port.open({:spawn_executable, "/usr/local/bin/vsock_port"}, [
      {:args, [vsock_port]},
      {:packet, 4},
      :binary,
      :exit_status
      # NOT :stderr_to_stdout. The port's stdout IS the framed protocol, so
      # merging stderr into it feeds vsock_port's own log lines to the frame
      # parser as a four-byte length.
    ])
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
