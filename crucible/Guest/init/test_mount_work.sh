#!/bin/sh
# test_mount_work.sh -- can the user's repository be damaged?
#
# mount-work is the code that stands between the agent and the real .git:
# it seeds the shadow ONCE per disk and swings the bind mount on EVERY boot.
# This runs on the build host (macOS sh + git), so the bind mount itself is
# a recorded invocation rather than a real mount -- what Linux does with the
# mount is the kernel's contract; that we ASK for it, every boot, with the
# shadow seeded first, is ours, and it is what this file pins. The guest
# gate proves the mounted result end to end.
set -u

FAILS=0
ok()   { echo "  ok   $1"; }
bad()  { echo "  FAIL $1"; FAILS=$((FAILS + 1)); }
is()   { # is <label> <expected> <actual>
    if [ "$2" = "$3" ]; then ok "$1"; else bad "$1 (expected '$2', got '$3')"; fi
}

command -v git >/dev/null 2>&1 || { echo "test_mount_work: git is required"; exit 1; }

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
log() { :; }                                     # quiet; the assertions are the output

# The bind mount, recorded instead of performed.
SHADOW_CALLS="$TMP/shadow-calls"
shadow_mark() { echo called >> "$SHADOW_CALLS"; }
shadow_count() { wc -l < "$SHADOW_CALLS" 2>/dev/null | tr -d ' '; }

# Overridable so the build can point this at the copy it just INSTALLED into
# the rootfs, rather than at the one next to this file. Testing what ships.
MOUNT_WORK=${MOUNT_WORK:-$(dirname "$0")/mount-work}

export WORK="$TMP/work" GITDIR="$TMP/var/crucible/git" STAMP="$TMP/var/lib/seeded"
export WORK_MOUNT=true SHADOW_MOUNT=shadow_mark
: > "$SHADOW_CALLS"

# A real project with real history, playing the user's tree behind the mount.
mkdir -p "$WORK"
( cd "$WORK" && git init -q -b main \
    && echo "from the user" > file.txt && git add -A \
    && git -c user.name=u -c user.email=u@u commit -qm "user commit" )
USER_HEAD=$(git -C "$WORK" rev-parse HEAD)

. "$MOUNT_WORK"

echo "== first boot: the shadow is seeded and the mount is asked for"
mount_work
is "the shadow carries the project's history" "$USER_HEAD" \
   "$(git --git-dir="$GITDIR" rev-parse HEAD 2>/dev/null)"
[ -f "$STAMP" ] && ok "the stamp is written" || bad "the stamp is written"
is "the bind mount is invoked" "1" "$(shadow_count)"

echo "== the agent commits into the shadow; a reboot keeps it and re-mounts"
echo "EDITED BY THE MODEL" > "$WORK/file.txt"
git --git-dir="$GITDIR" --work-tree="$WORK" add -A
git --git-dir="$GITDIR" --work-tree="$WORK" \
    -c user.name=t -c user.email=t@t commit -qm "agent work"
AGENT_TIP=$(git --git-dir="$GITDIR" rev-parse HEAD)
mount_work
is "the shadow is NOT re-seeded over the agent's commits" "$AGENT_TIP" \
   "$(git --git-dir="$GITDIR" rev-parse HEAD 2>/dev/null)"
is "the bind mount is invoked on EVERY boot" "2" "$(shadow_count)"
is "the real .git never moved" "$USER_HEAD" "$(git -C "$WORK" rev-parse HEAD)"

echo "== a half-seeded shadow is seeded again, not adopted"
rm -rf "$GITDIR"                                  # the crash: stamp says yes, shadow gone
mount_work
is "the shadow is seeded again" "$USER_HEAD" \
   "$(git --git-dir="$GITDIR" rev-parse HEAD 2>/dev/null)"
is "the mount is still asked for" "3" "$(shadow_count)"

echo "== a dropped stamp (git_refresh) re-seeds, discarding private commits"
git --git-dir="$GITDIR" --work-tree="$WORK" \
    -c user.name=t -c user.email=t@t commit -qam "private checkpoint"
rm -f "$STAMP"                                    # what the git_refresh op does
mount_work
is "the shadow is back at the repo's state" "$USER_HEAD" \
   "$(git --git-dir="$GITDIR" rev-parse HEAD 2>/dev/null)"
[ -f "$STAMP" ] && ok "the stamp is rewritten" || bad "the stamp is rewritten"
is "the mount follows the refresh" "4" "$(shadow_count)"

echo "== a project with no repository gets no shadow and no mount"
export WORK="$TMP/plain" GITDIR="$TMP/var2/git" STAMP="$TMP/var2/seeded"
mkdir -p "$WORK" && echo "just a file" > "$WORK/a.txt"
: > "$SHADOW_CALLS"
mount_work
[ ! -e "$GITDIR" ] && ok "no shadow is created" || bad "no shadow is created"
is "no bind mount is attempted" "0" "$(shadow_count)"

echo "== a .git FILE (worktree/submodule checkout) is treated as non-git"
export WORK="$TMP/wt" GITDIR="$TMP/var3/git" STAMP="$TMP/var3/seeded"
mkdir -p "$WORK" && echo "gitdir: /somewhere/else" > "$WORK/.git"
: > "$SHADOW_CALLS"
mount_work
[ ! -e "$GITDIR" ] && ok "no shadow for a gitfile checkout" || bad "no shadow for a gitfile checkout"
is "no mount over a gitfile" "0" "$(shadow_count)"
is "the gitfile is left alone" "gitdir: /somewhere/else" "$(cat "$WORK/.git")"

echo ""
if [ "$FAILS" = 0 ]; then
    echo "mount-work: all checks pass"
else
    echo "mount-work: $FAILS failure(s)"
    exit 1
fi
