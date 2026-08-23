#!/bin/sh
# test_seed_work.sh -- does an hour of the model's work survive a restart?
#
# The host half of this has its own test (GuestImageSuite: the session's disk is
# reused, not re-cloned). This is the guest half, and it is the one where the
# distinction between "the file is there" and "the file is what I left it as"
# decides the answer: `cp -a /base/. /work/` ADDS files and OVERWRITES changed
# ones, so a presence check passes while every edit is silently reverted.
#
# So every assertion here reads content back.
set -u

FAILS=0
ok()   { echo "  ok   $1"; }
bad()  { echo "  FAIL $1"; FAILS=$((FAILS + 1)); }
is()   { # is <label> <expected> <actual>
    if [ "$2" = "$3" ]; then ok "$1"; else bad "$1 (expected '$2', got '$3')"; fi
}
gone() { if [ ! -e "$2" ]; then ok "$1"; else bad "$1 ($2 still exists)"; fi }

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

export WORK="$TMP/work" BASE="$TMP/base" SEED="$TMP/var/seeded"
export BASE_MOUNT=true BASE_UMOUNT=true          # the test populates $BASE itself
log() { :; }                                     # quiet; the assertions are the output

# Overridable so the build can point this at the copy it just INSTALLED into
# the rootfs, rather than at the one next to this file. Testing what ships.
SEED_WORK=${SEED_WORK:-$(dirname "$0")/seed-work}
. "$SEED_WORK"

# The user's project, as it arrives over virtiofs.
mkdir -p "$BASE/src"
echo "original" > "$BASE/src/main.c"
echo "keep me"  > "$BASE/README"

echo "== first boot seeds /work"
seed_work
is  "the project is copied"              "original" "$(cat "$WORK/src/main.c" 2>/dev/null)"
is  "every file comes across"            "keep me"  "$(cat "$WORK/README" 2>/dev/null)"
[ -d "$WORK/.crucible-git" ] && ok "a baseline is recorded" || bad "a baseline is recorded"
[ -f "$SEED" ] && ok "the seed marker is written" || bad "the seed marker is written"

echo "== the model works"
echo "EDITED BY THE MODEL" > "$WORK/src/main.c"     # modified  <- cp -a reverts this
echo "new file"            > "$WORK/src/extra.c"    # added
rm -f "$WORK/README"                                # deleted   <- cp -a restores this

echo "== second boot keeps it"
seed_work
is   "a MODIFIED file keeps the model's content"  "EDITED BY THE MODEL" "$(cat "$WORK/src/main.c" 2>/dev/null)"
is   "an ADDED file survives"                     "new file"            "$(cat "$WORK/src/extra.c" 2>/dev/null)"
gone "a DELETED file stays deleted"               "$WORK/README"

echo "== a disk from before the marker is adopted, not re-seeded"
rm -f "$SEED"                                       # older disk: baseline, no marker
seed_work
is   "the model's content is still there"  "EDITED BY THE MODEL" "$(cat "$WORK/src/main.c" 2>/dev/null)"
gone "the deleted file is still deleted"   "$WORK/README"
[ -f "$SEED" ] && ok "the marker is written on adoption" || bad "the marker is written on adoption"

echo "== a half-seeded /work is seeded again, not preserved"
# Baseline never got recorded, so the marker was never written. The next boot
# must copy rather than treat a partial tree as the model's work.
rm -rf "$TMP/work" "$TMP/var"
export WORK="$TMP/work2" SEED="$TMP/var2/seeded"
mkdir -p "$WORK" && echo "partial" > "$WORK/stale"
seed_work
is "an unmarked, unbaselined /work is re-seeded" "original" "$(cat "$WORK/src/main.c" 2>/dev/null)"

echo "== with no share attached, nothing is destroyed"
export WORK="$TMP/work3" SEED="$TMP/var3/seeded" BASE_MOUNT=false
mkdir -p "$WORK" && echo "precious" > "$WORK/held"
seed_work
is "an unmountable share leaves /work alone" "precious" "$(cat "$WORK/held" 2>/dev/null)"

echo ""
if [ "$FAILS" = 0 ]; then
    echo "seed-work: all checks pass"
else
    echo "seed-work: $FAILS failure(s)"
    exit 1
fi
