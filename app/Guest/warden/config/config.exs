import Config

# The bridge opens a vsock port that exists only inside the guest, and tests
# start their own bridge with a fake port, so the application must not race
# them to the socket. Only :prod -- what ships -- brings the bridge up on boot.
config :warden, start_bridge: config_env() == :prod
