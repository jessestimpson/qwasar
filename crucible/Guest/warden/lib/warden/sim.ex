defmodule Warden.Sim do
  @moduledoc """
  The seam between warden and `eta`, so that removing eta costs a dependency
  rather than a rewrite (PLAN.md 9.5).

  eta's README prescribes a wrapper header for Erlang consumers:

      -ifdef(DST).
      -include_lib("eta/include/eta.hrl").
      -else.
      -define(ETA_LABEL(Label), ok).
      -endif.

  This is the Elixir equivalent, and it exists because `use Eta` cannot itself
  be the seam: the `Eta` module has to be present for `use Eta` to compile at
  all, and eta compiles its Elixir entry point only in its own `:test`
  environment. A shipping build that has no eta dependency therefore cannot
  `use Eta`, however cleverly the macros inside it are written.

  So warden writes `Sim.send_after/3` rather than `Eta.send_after/3` or
  `Process.send_after/3`, and this decides which one that becomes:

    * with eta on the code path — the simulation build — it is `Eta.*`, and the
      scheduler and clock are under the test's control;
    * without it — what ships — it is the stdlib, with no trace of eta in the
      image or the lock file.

  ## The rule this does not enforce

  `receive ... after` is **not** virtualized by eta's Elixir macros; the parse
  transform rewrites that construct and the macros cannot. A deadline written
  that way runs on the real clock inside a simulation that reports itself
  deterministic. So warden contains no `receive ... after` anywhere: a process
  that wants a deadline sends itself a message with `Sim.send_after/3` and
  matches it in a plain `receive`.
  """

  @doc false
  defmacro __using__(opts) do
    gen = Keyword.get(opts, :gen_server, false)
    quote do
      unquote(sim_alias())
      unquote(if gen, do: gen_server_shim())
    end
  end

  # `@behaviour :gen_server` -- the ERLANG atom -- rather than `use GenServer`.
  #
  # eta's observe pass only wraps a module when `:gen_server in @behaviour`, and
  # Elixir's `use GenServer` sets `@behaviour GenServer`, the Elixir module.
  # Declaring both silences nothing and warns about conflicting callbacks, so
  # this declares the one that matters and supplies the one thing `use
  # GenServer` was wanted for.
  defp gen_server_shim do
    quote do
      @behaviour :gen_server

      def child_spec(arg) do
        %{id: __MODULE__, start: {__MODULE__, :start_link, [arg]}, type: :worker}
      end

      defoverridable child_spec: 1
    end
  end

  defp sim_alias do
    if Code.ensure_loaded?(Eta) do
      quote do
        use Eta
        # `require` is not optional: Sim.* are macros, and an aliased module's
        # macros are invisible without it -- the call compiles as a remote
        # function and fails at runtime with "undefined or private", which is a
        # confusing way to be told you forgot a require.
        require Eta
        alias Eta, as: Sim
      end
    else
      quote do
        # persist: true, or the compiler warns that an attribute it can see set
        # is never read -- which in a build with no eta is exactly true and
        # exactly not worth saying four times.
        Module.register_attribute(__MODULE__, :eta_observe, persist: true)
        require Warden.Sim.Stdlib
        alias Warden.Sim.Stdlib, as: Sim
      end
    end
  end

  @doc "True when this build is instrumented. Reported, never branched on."
  def instrumented?, do: Code.ensure_loaded?(:eta_time)

  defmodule Stdlib do
    @moduledoc """
    What `Sim.*` means when eta is absent: the ordinary thing, with the
    observation hooks compiled away to nothing.
    """

    defmacro send_after(dest, msg, time), do: quote(do: Process.send_after(unquote(dest), unquote(msg), unquote(time)))
    defmacro send(dest, msg), do: quote(do: Kernel.send(unquote(dest), unquote(msg)))
    defmacro cast(server, msg), do: quote(do: GenServer.cast(unquote(server), unquote(msg)))
    defmacro call(server, msg, timeout), do: quote(do: GenServer.call(unquote(server), unquote(msg), unquote(timeout)))
    defmacro spawn(fun), do: quote(do: Kernel.spawn(unquote(fun)))
    defmacro cancel_timer(ref), do: quote(do: Process.cancel_timer(unquote(ref)))

    # Observation is a simulation concept. Outside one it costs nothing, and
    # must not be allowed to cost anything -- these are the calls that would
    # otherwise be sprinkled through the control plane for the benefit of a
    # test framework that is not there.
    defmacro label(_name), do: quote(do: :ok)
    defmacro log(_event), do: quote(do: :ok)
  end
end
