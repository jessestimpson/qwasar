#!/bin/sh
# lint_sim.sh -- the rule that keeps the simulation honest.
#
# PLAN.md 9.1: on eta's `elixir` branch, instrumentation is explicit macros
# rather than a parse transform, so it catches only the call sites you wrote as
# macros. A hand-written `Process.send_after/3` puts real time back into a run
# that reports itself deterministic, silently. And `receive ... after` cannot be
# virtualized at all -- eta's own documentation says so -- so a deadline written
# that way runs on the real clock inside the simulation.
#
# Neither failure announces itself. Both make a green suite mean nothing. So
# they are a build error.
#
# lib/warden/sim.ex is exempt: it IS the seam, and the stdlib calls in it are
# the thing being switched.

set -eu
cd "$(dirname "$0")"

FAIL=0
FILES=$(find lib -name '*.ex' ! -name 'sim.ex')

# Exemptions are per line and must say why:
#
#     Process.sleep(150)   # lint:real-clock polling a real OS node
#
# Not per file. A file-level exemption is how a lint rots -- everything drifts
# into the exempt file and nobody notices. A line-level one is visible at the
# call site, has to be justified where it is read, and is COUNTED below so they
# cannot quietly multiply.
EXEMPT='# lint:real-clock'

flag() {
    pattern="$1"; label="$2"
    # shellcheck disable=SC2086
    hits=$(grep -nE "$pattern" $FILES 2>/dev/null | grep -vF "$EXEMPT" || true)
    if [ -n "$hits" ]; then
        echo "lint: $label -- use the Sim.* equivalent, or mark the line"
        echo "      '$EXEMPT <why>' if it is genuinely outside any simulation"
        echo "$hits" | sed 's/^/      /'
        FAIL=1
    fi
}

flag '(^|[^.[:alnum:]_])Process\.(send_after|cancel_timer|sleep)\(' 'real-clock timer'
flag '(^|[^.[:alnum:]_])GenServer\.(cast|call)\('                   'uninstrumented message'
flag '(^|[^.[:alnum:]_]):timer\.'                                   'real-clock timer'

# `receive ... after`, which is a language construct rather than a call, so it
# needs more than a grep. `try ... after` uses the same keyword and is fine.
recv=$(awk '
    /(^|[^a-zA-Z_])receive([^a-zA-Z_]|$)/ { inrecv = 1; start = FNR }
    inrecv && /^[[:space:]]*after([^a-zA-Z_]|$)/ {
        printf "%s:%d: receive ... after (opened at line %d)\n", FILENAME, FNR, start
        inrecv = 0
    }
    /^[[:space:]]*end([^a-zA-Z_]|$)/ { inrecv = 0 }
' $FILES 2>/dev/null || true)

if [ -n "$recv" ]; then
    echo "lint: receive ... after is NOT virtualized by eta's Elixir macros"
    echo "$recv" | sed 's/^/      /'
    echo "      send yourself a message with Sim.send_after/3 and match it in a plain receive"
    FAIL=1
fi

# shellcheck disable=SC2086
N_EXEMPT=$(grep -cF "$EXEMPT" $FILES 2>/dev/null | awk -F: '{s+=$2} END {print s+0}')

if [ "$FAIL" = 0 ]; then
    if [ "$N_EXEMPT" = 0 ]; then
        echo "lint: warden is fully instrumented"
    else
        echo "lint: warden is instrumented, with $N_EXEMPT declared real-clock exemption(s):"
        # shellcheck disable=SC2086
        grep -nF "$EXEMPT" $FILES | sed 's/^/      /'
    fi
else
    exit 1
fi
