defmodule Warden.Application do
  @moduledoc """
  The guest's control plane, as a supervision tree.

  PLAN.md 7.3: warden owns the wire. It must survive anything the agent does,
  which begins with it being supervised rather than being a script that returns.

  Four children, and the order matters: the registry before the workspace
  (which replays from it after a crash), and both before the bridge, which hands
  work to all of them. `:one_for_one`
  means neither can take the channel down -- the bridge stays up, its in-flight
  requests time out and are answered, and the host is told rather than left
  waiting.

  The parts that will make this interesting (the workspace node, its monitor,
  the registry) are M3 and M4; a supervisor with placeholders in it is harder to
  read than one that grows.
  """
  use Application

  @impl true
  def start(_type, _args) do
    children =
      if Application.get_env(:warden, :start_bridge, true) do
        [Warden.Worker, Warden.Registry, Warden.Workspace, Warden.Bridge]
      else
        # Tests start their own bridge with their own port, so the
        # application must not race it to the vsock.
        []
      end

    Supervisor.start_link(children, strategy: :one_for_one, name: Warden.Supervisor)
  end
end
