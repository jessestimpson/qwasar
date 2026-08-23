import Config

# The bridge opens a vsock port that exists only inside the guest. Tests run in
# a container, and a simulation starts its own bridge with a fake port and its
# own worker (PLAN.md 9.3), so the application must not race it to the socket.
#
# Only :prod -- what actually ships -- brings the bridge up on boot.
config :warden, start_bridge: config_env() == :prod
