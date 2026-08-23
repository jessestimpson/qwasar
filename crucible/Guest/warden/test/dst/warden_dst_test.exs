defmodule Warden.DSTTest do
  @moduledoc """
  The control plane's real test suite (PLAN.md 9.2, 10).

  A seed sweep, not a single run: one seed proves one interleaving, and the
  point of a scheduler under test control is that the others are reachable too.
  Any violation is shrunk before it is reported, because an unshrunk trace is
  thousands of scheduling decisions and no help at all.
  """
  use ExUnit.Case, async: false

  @moduletag timeout: 600_000

  @seeds 1..64
  @opts %{max_ops: 25, max_steps: 20_000, preload: [:warden]}

  # Determinism has a precondition, and it is not stated anywhere in eta: the
  # code must already be loaded.
  #
  # The first :eta_run.run in a process loads ~165 modules -- warden's, eta's
  # own, parts of Elixir and a good deal of OTP. A module load is a blocking
  # call into `code_server`, a process the scheduler does not control and cannot
  # interleave, so a scheduled process that touches an unloaded module stops for
  # a reason the seed did not choose. The first run is therefore measuring the
  # code server; only the ones after it are measuring the system.
  #
  # Untreated this was a ~40% failure rate over the sweep (12 of 30 runs here),
  # always exactly one seed and always one of the first few -- whichever seed
  # happened to be running while the last modules came in. It read as "the
  # bridge is subtly broken" and was nothing of the kind.
  #
  # This matters beyond the test. PLAN.md 7.2 exists to load code at runtime, so
  # any simulation running while `define` compiles a module is, for that moment,
  # not deterministic.
  # In the test process, not in setup_all: eta's log is one per VM and is held
  # by whichever process is driving, so a warm-up run from a setup_all process
  # -- which outlives the block -- makes every later run fail with
  # {:log_in_use, ...} instead of running.
  test "the bridge's contract holds across a seed sweep" do
    # One throwaway run loads warden, eta and most of OTP.
    _ = :eta_run.run(Warden.DST.Harness, Map.put(@opts, :seed, 0))

    # And one inspect, because the Inspect protocol's implementations arrive
    # lazily and the bridge reaches for them only on the bad_reply path -- a
    # branch a single warm-up seed may not take. Six modules were still landing
    # mid-sweep without this.
    _ = inspect({:a, [1, 2], 3}, pretty: true, limit: 40)

    loaded_before = MapSet.new(:code.all_loaded(), fn {m, _} -> m end)

    failures =
      Enum.reduce(@seeds, [], fn seed, acc ->
        result = :eta_run.run(Warden.DST.Harness, Map.put(@opts, :seed, seed))

        case result do
          %{outcome: :ok} ->
            acc

          %{outcome: outcome, trace: trace} ->
            minimal =
              try do
                %{trace: t} =
                  :eta_shrink.shrink(Warden.DST.Harness, trace, Map.put(@opts, :seed, seed))

                t
              rescue
                _ -> trace
              end

            [{seed, outcome, minimal} | acc]
        end
      end)

    if failures != [] do
      report =
        Enum.map_join(failures, "\n\n", fn {seed, outcome, trace} ->
          """
          seed #{seed}: #{inspect(outcome)}
          minimal trace (#{length(List.wrap(trace))} steps):
          #{inspect(trace, pretty: true, limit: 40)}
          """
        end)

      flunk("#{length(failures)} of #{Enum.count(@seeds)} seeds violated the contract\n\n#{report}")
    end

    # The precondition, checked after the fact rather than assumed.
    #
    # A module arriving mid-sweep means some seed ran against a scheduler that
    # was not in sole control, and its result -- pass OR fail -- says nothing.
    # Better to fail with the reason than to flake with a violation that looks
    # like a real defect.
    arrived =
      :code.all_loaded()
      |> MapSet.new(fn {m, _} -> m end)
      |> MapSet.difference(loaded_before)

    if MapSet.size(arrived) > 0 do
      flunk("""
      #{MapSet.size(arrived)} module(s) were loaded during the sweep:

        #{arrived |> Enum.sort() |> inspect(limit: 30)}

      A module load blocks on code_server, which the scheduler does not control,
      so whichever seed was running at that moment did not get the interleaving
      its seed chose -- and its result, pass or fail, means nothing. Reach for
      those modules in the warm-up above, then run this again.
      """)
    end
  end
end
