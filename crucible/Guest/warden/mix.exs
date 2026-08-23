defmodule Warden.MixProject do
  use Mix.Project

  # The guest's control plane.
  #
  # PLAN.md 7.3: warden owns the wire and is the one thing in the guest the
  # agent cannot modify. It once carried a deterministic-simulation build
  # (eta, PLAN.md §9); that experiment ended and was removed exactly the way
  # its seam promised -- by deleting the dependency.
  def project do
    [
      app: :warden,
      version: "0.1.0",
      elixir: "~> 1.17",
      start_permanent: Mix.env() == :prod,
      test_paths: ["test/unit"],
      deps: []
    ]
  end

  def application do
    [extra_applications: [:logger], mod: {Warden.Application, []}]
  end
end
