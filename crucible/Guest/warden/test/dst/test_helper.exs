# The control plane's tests run serially, always.
#
# eta's `eta_time` and `eta_log` use named ETS tables, so only one simulation
# can exist in a VM at a time (PLAN.md 9.1). `async: false` on each case is not
# enough on its own -- the seed sweep inside a single test is already serial,
# but two test files would not be.
ExUnit.start(max_cases: 1, timeout: 600_000)
