defmodule Warden.MixProject do
  use Mix.Project

  # The guest's control plane.
  #
  # PLAN.md 7.3: warden owns the wire and is the one thing in the guest the
  # agent cannot modify. PLAN.md 9.5: eta is an experiment with a defined exit,
  # so it must be possible to remove it by deleting a dependency rather than by
  # rewriting anything.
  #
  # Three environments, and the difference between them is the whole point:
  #
  #   prod  what ships in the image. No :eta in the lock file at all. `use Eta`
  #         still compiles -- its macros expand to Process.send_after/2,
  #         GenServer.cast/2 and friends when :eta is absent, which is what
  #         makes the exit cheap.
  #   dst   the simulation build. :eta present, so the same macros expand to
  #         :eta_* calls and the scheduler and clock are under test control.
  #   test  ordinary ExUnit, for the parts that are logic rather than timing.
  def project do
    [
      app: :warden,
      version: "0.1.0",
      elixir: "~> 1.17",
      start_permanent: Mix.env() == :prod,
      elixirc_paths: elixirc_paths(Mix.env()),
      test_paths: test_paths(Mix.env()),
      deps: deps(Mix.env()),
      aliases: aliases()
    ]
  end

  def application do
    [extra_applications: [:logger], mod: {Warden.Application, []}]
  end

  defp elixirc_paths(:dst), do: ["lib", "test/dst"]
  defp elixirc_paths(_), do: ["lib"]

  # The two suites are different kinds of thing and must not run together.
  #
  # `test/unit` is ordinary ExUnit over the parts that are logic -- the edit
  # matcher, path confinement -- and runs anywhere. `test/dst` is the
  # deterministic simulation and needs eta on the code path, which only the
  # :dst environment has. Running the simulation in :test fails with an
  # undefined :eta_run rather than with anything informative.
  defp test_paths(:dst), do: ["test/dst"]
  defp test_paths(_), do: ["test/unit"]

  # `env: :test` is not a typo and not a workaround for our convenience.
  # eta compiles its Elixir entry point -- the `Eta` macro module -- only in its
  # own :test environment, and Mix builds dependencies in :prod unless told
  # otherwise. Without this the dependency yields the Erlang library with no
  # `Eta` module, and `use Eta` fails with a module-not-found that explains
  # nothing. The same environment turns on the DST define its header hangs on.
  defp deps(:dst) do
    [{:eta, github: "jessestimpson/eta", branch: "elixir", env: :test, runtime: false}]
  end

  defp deps(_), do: []

  defp aliases do
    [
      # The simulation suite. Named rather than left to `mix test` so that
      # running it in the wrong environment is a typo you cannot make.
      dst: ["cmd MIX_ENV=dst mix test --no-start"]
    ]
  end
end
