#!/bin/sh
# mkimage.sh -- builds the Crucible guest image, natively on macOS.
#
# PLAN.md 6.2. Three outputs, into ../build/guest/:
#
#   Image       the raw arm64 kernel (vmlinuz kept beside it, unwrapped here)
#   initramfs   built by mkrootfs.py: busybox.static + the virtio/ext4 modules
#   disk.img    a 3 GB sparse ext4 root filesystem
#
# No Docker, no Linux. Alpine packages are tarballs and BEAM files are
# portable, which between them dissolve every reason the old build needed a
# Linux userland:
#
#   - the rootfs is apk packages fetched and untarred by mkrootfs.py;
#   - OTP 27 is Alpine's own erlang27 package -- no source build at all;
#   - Elixir is the precompiled -otp-27 release, pure BEAM;
#   - the warden is compiled HERE, on the host's pinned Elixir -- BEAM code
#     does not care which OS compiled it, only which OTP;
#   - vsock_port is cross-compiled with zig (a self-contained musl toolchain);
#   - the ext4 image comes from mke2fs -d, which needs no root and no Linux.
#
# Host dependencies, checked below: python3, mise (erlang/elixir/zig/rebar
# pins), and e2fsprogs from brew. That is the whole list.
set -eu

HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${OUT:-$HERE/../build/guest}"
CACHE="${CACHE:-$HERE/../build/guest-cache}"

# The toolchain pins. erlang/elixir MUST match the erlang27 package and the
# elixir zip mkrootfs.py installs -- the warden's .beam files are compiled
# here and run there.
ERLANG_PIN="erlang@27.3.4"
ELIXIR_PIN="elixir@1.18.4-otp-27"
ZIG_PIN="zig@0.15.2"
# rebar3 is an escript -- BEAM bytecode, like Elixir -- so the host's own
# mise install runs as is on the guest's erlang27.  REBAR_PIN= leaves it out.
REBAR_PIN="${REBAR_PIN-rebar@3.24.0}"
# Beyond the defaults: more Alpine packages, and a folder copied over the
# root filesystem as is.  Both are read when the image is built.
#   GUEST_PACKAGES="go rust"   space- or comma-separated Alpine package names
#   GUEST_OVERLAY=<dir>        default Guest/overlay, used when it exists
GUEST_OVERLAY="${GUEST_OVERLAY-$HERE/overlay}"

fail() { echo "mkimage: $*" >&2; exit 1; }

command -v python3 >/dev/null || fail "python3 is required"
command -v mise >/dev/null || fail "mise is required (https://mise.jdx.dev); it supplies the pinned erlang, elixir and zig"
for p in /opt/homebrew/opt/e2fsprogs/sbin /usr/local/opt/e2fsprogs/sbin; do
    [ -x "$p/mke2fs" ] && MKE2FS_OK=1
done
[ "${MKE2FS_OK:-}" = 1 ] || fail "mke2fs not found; brew install e2fsprogs"

echo "mkimage: building the guest image (native, no Docker)"
mkdir -p "$OUT" "$CACHE"

# ---- vsock_port: one static arm64-musl binary -------------------------------
echo "mkimage: vsock_port (zig cc, aarch64-linux-musl, static)"
mise exec "$ZIG_PIN" -- zig cc -target aarch64-linux-musl -static -O2 \
    -Wall -Wextra -Werror \
    -o "$CACHE/vsock_port" "$HERE/vsock_port/vsock_port.c" \
    || fail "zig cross-compile failed; mise install $ZIG_PIN"

# ---- mount-work: the gate that decides whether the user's repo is safe ------
echo "mkimage: mount-work contract"
MOUNT_WORK="$HERE/init/mount-work" sh "$HERE/init/test_mount_work.sh" \
    || fail "mount-work failed its contract; the image could expose the real .git"

# ---- warden: tested, then compiled, on the host's pinned toolchain ----------
#
# MIX_ENV=test excludes :work_fs -- those tests write under /work, which only
# exists in the guest; `make sandbox` exercises the real tools there. The
# .beam files produced here run unmodified on the guest's erlang27: BEAM
# bytecode is portable across OSes within an OTP major.
echo "mkimage: warden (mise: $ERLANG_PIN $ELIXIR_PIN)"
(
    cd "$HERE/warden"
    mise exec "$ERLANG_PIN" "$ELIXIR_PIN" -- sh -c '
        set -eu
        erl -noshell -eval "io:put_chars(erlang:system_info(otp_release)), halt(0)." | grep -q 27 \
            || { echo "mkimage: host OTP is not 27; the guest runs erlang27" >&2; exit 1; }
        MIX_ENV=test mix test --no-start --exclude work_fs
        MIX_ENV=prod mix compile
    '
) || fail "warden build failed"

# ---- rebar3, from the host's mise --------------------------------------------
REBAR3=""
if [ -n "$REBAR_PIN" ]; then
    echo "mkimage: rebar3 (mise: $REBAR_PIN)"
    mise install "$REBAR_PIN" >/dev/null 2>&1 || fail "mise install $REBAR_PIN failed"
    REBAR3="$(mise where "$REBAR_PIN" 2>/dev/null)/bin/rebar3"
    head -c 32 "$REBAR3" 2>/dev/null | grep -q escript \
        || fail "$REBAR3 is not an escript; only a portable rebar3 can go in the guest"
fi

# ---- the image itself -------------------------------------------------------
OUT="$OUT" CACHE="$CACHE" \
    VSOCK_PORT="$CACHE/vsock_port" \
    WARDEN_EBIN="$HERE/warden/_build/prod/lib/warden" \
    REBAR3="$REBAR3" GUEST_PACKAGES="${GUEST_PACKAGES:-}" GUEST_OVERLAY="$GUEST_OVERLAY" \
    python3 "$HERE/mkrootfs.py"

echo "mkimage: done"
ls -lh "$OUT"
