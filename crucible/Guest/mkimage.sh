#!/bin/sh
# mkimage.sh -- builds the Crucible guest image.
#
# PLAN.md 6.2. Three outputs, into ../build/guest/:
#
#   vmlinuz     the Alpine linux-virt kernel
#   initramfs   built against the rootfs's own modules
#   disk.img    a 2 GB sparse ext4 root filesystem
#
# VZLinuxBootLoader takes the kernel and initramfs from the host, so the disk
# needs no partition table and no bootloader -- it is a root filesystem and
# nothing else.
#
# Docker is a BUILD dependency only. Building a Linux rootfs needs Linux; the
# shipped app never needs Docker and neither does anything at runtime. This runs
# once, and its output is cloned per session (PLAN.md 6.3).

set -eu

HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${OUT:-$HERE/../build/guest}"
ALPINE="${ALPINE:-3.22}"
TAG="crucible-guest:${ALPINE}"

if ! docker info >/dev/null 2>&1; then
    echo "mkimage: the Docker daemon is not reachable." >&2
    echo "         Start Docker Desktop and try again; it is needed only to build" >&2
    echo "         this image, never to run Crucible." >&2
    exit 1
fi

# Docker's build cache grows by several GB per rebuild of the mise layers, and
# a full cache fails with "no space left on device" in the middle of mkfs --
# which reads as a broken image build rather than a full disk. Warn early.
CACHE_GB=$(docker system df --format '{{.Type}} {{.Size}}' 2>/dev/null \
    | awk '$1 == "Build" { gsub(/GB/, "", $2); print int($2) }')
if [ -n "${CACHE_GB:-}" ] && [ "$CACHE_GB" -gt 40 ]; then
    echo "mkimage: docker build cache is ${CACHE_GB}GB; run 'docker builder prune -af'" >&2
fi

echo "mkimage: building $TAG (alpine $ALPINE, arm64)"
docker build \
    --platform linux/arm64 \
    --build-arg "ALPINE=$ALPINE" \
    --target artifacts \
    --progress plain \
    -t "$TAG" \
    "$HERE"

mkdir -p "$OUT"
echo "mkimage: extracting to $OUT"

# `docker build --target artifacts` leaves a scratch image holding exactly the
# three files; --output would be cleaner but needs BuildKit, which this Docker
# is old enough to lack. Create-and-copy works everywhere.
CID="$(docker create --platform linux/arm64 "$TAG" /nonexistent)"
trap 'docker rm -f "$CID" >/dev/null 2>&1 || true' EXIT
for f in vmlinuz initramfs disk.img; do
    docker cp "$CID:/$f" "$OUT/$f"
done

# Alpine's aarch64 kernel is an EFI zboot image -- "MZ" then "zimg" then a
# gzip payload -- which Virtualization.framework cannot boot. VZLinuxBootLoader
# on Apple silicon wants the raw arm64 `Image`, the one with ARM\x64 at offset
# 0x38. So unwrap it here, at build time, and fail loudly if the shape ever
# changes rather than handing the framework something it rejects with
# "Internal Virtualization error" and no further detail.
echo "mkimage: unwrapping the EFI zboot kernel"
python3 - "$OUT/vmlinuz" "$OUT/Image" <<'KERNEL_PY'
import struct, gzip, lzma, sys
src, dst = sys.argv[1], sys.argv[2]
d = open(src, "rb").read()
if d[:2] != b"MZ" or d[4:8] != b"zimg":
    sys.exit("mkimage: %s is not an EFI zboot image (magic %s)" % (src, d[:8].hex()))
off, size = struct.unpack_from("<II", d, 8)
comp = d[24:36].split(b"\0")[0].decode()
payload = d[off:off + size]
if comp == "gzip":
    raw = gzip.decompress(payload)
elif comp in ("lzma", "xz"):
    raw = lzma.decompress(payload)
else:
    sys.exit("mkimage: unsupported zboot compression %r" % comp)
if raw[0x38:0x3c] != b"ARM\x64":
    sys.exit("mkimage: the decompressed payload is not an arm64 Image")
open(dst, "wb").write(raw)
print("  vmlinuz -> Image (%s, %.1f MB)" % (comp, len(raw) / 1e6))
KERNEL_PY

# `docker cp` writes every hole out as real zeros, so a 257 MB filesystem
# arrives as a 2 GB file. That matters twice: it is the golden image's cost on
# disk, and PLAN.md 6.3's whole argument is that a session clone is cheap
# because the image is mostly holes. Punch them back.
echo "mkimage: re-sparsifying disk.img"
python3 - "$OUT/disk.img" <<'PY'
import os, sys
path = sys.argv[1]
block = 1 << 20
tmp = path + ".sparse"
with open(path, "rb") as src, open(tmp, "wb") as dst:
    zero = b"\0" * block
    while True:
        buf = src.read(block)
        if not buf:
            break
        if buf == zero[:len(buf)]:
            dst.seek(len(buf), os.SEEK_CUR)      # a hole, not a write
        else:
            dst.write(buf)
    dst.truncate(os.path.getsize(path))
os.replace(tmp, path)
st = os.stat(path)
print("  apparent %.2f GB, allocated %.0f MB"
      % (st.st_size / 1e9, st.st_blocks * 512 / 1e6))
PY

echo "mkimage: done"
ls -lh "$OUT"
echo
echo "  disk.img apparent: $(stat -f '%z' "$OUT/disk.img" | awk '{printf "%.2f GB", $1/1e9}')"
echo "  disk.img on disk:  $(du -h "$OUT/disk.img" | cut -f1)"
