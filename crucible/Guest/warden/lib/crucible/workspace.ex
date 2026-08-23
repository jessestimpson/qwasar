defmodule Crucible.Workspace do
  @moduledoc """
  The agent's node.

  PLAN.md 7.3: warden owns the wire and cannot be modified; this is the other
  half — the node the agent *can* modify, on the far side of Erlang
  distribution, where a module it hot-loads (M4) can crash the scheduler or
  exhaust the atom table without the control plane noticing anything worse than
  a `nodedown`.

  At M3 it does nothing but exist and be reachable. That is the point: the
  lifecycle has to be right before there is anything valuable running on it,
  and `Warden.Workspace` is what proves it comes back.

  It blocks on a plain `receive` with no `after`. A timer here would be a
  real-clock wait that eta cannot virtualize, and this process is meant to do
  precisely nothing until someone `erpc`s into it.
  """

  def main do
    :global.register_name(:crucible_workspace, self())
    IO.puts(:stderr, "[workspace] up on #{node()}")
    receive do
      :never -> :ok
    end
  end

  @doc "Reachability, cheap enough to call on every status check."
  def ping, do: {:pong, node(), :erlang.system_info(:otp_release)}
end
