#!/usr/bin/env python3
"""mkrootfs.py -- assembles the Crucible guest image on macOS, no Linux anywhere.

PLAN.md 6.2. Driven by mkimage.sh, which compiles the pieces that need real
toolchains (vsock_port via zig, the warden via the host's pinned Elixir) and
hands their paths in. This script does everything a Linux builder used to:

  - resolves and downloads Alpine packages by reading APKINDEX directly --
    an .apk is gzip-concatenated tar, so extraction is stdlib work;
  - lays the rootfs out as a plain directory tree;
  - builds the initramfs itself: a ~20-line /init, busybox.static, and the
    module closure for virtio_blk+ext4 computed from modules.dep;
  - unwraps the EFI zboot kernel to the raw arm64 Image VZLinuxBootLoader wants;
  - populates a sparse ext4 disk with mke2fs -d (no root, no loop devices).

What apk's install scripts would have done and this does not: the busybox
symlink farm (crucible-init runs `/bin/busybox --install -s` at boot instead),
and device nodes (devtmpfs provides them). File ownership in the image is the
build user's, not root's -- everything in the guest runs as root, which reads
any file regardless, and nothing in the image relies on setuid.
"""

import gzip
import hashlib
import io
import os
import re
import shutil
import stat
import struct
import subprocess
import sys
import tarfile
import urllib.request
import zipfile

ALPINE = "v3.23"
ARCH = "aarch64"
MIRROR = "https://dl-cdn.alpinelinux.org/alpine"
REPOS = ["main", "community"]

# The operating system, the tools the agent uses to look at code, and the
# BEAM the control plane and workspace run on. erlang27 rather than a source
# build: Alpine 3.23 carries OTP 27 as a package, which is what made this
# whole no-Linux build possible. busybox-suid is gone with mise -- nothing in
# the guest runs as a non-root user, so setuid serves no one.
PACKAGES = [
    "alpine-baselayout", "alpine-release", "busybox", "musl",
    "linux-virt",
    "libstdc++", "ncurses-libs", "openssl", "zlib", "libcrypto3", "libssl3",
    "git", "ripgrep", "nodejs", "npm", "python3",
    "e2fsprogs-extra",
    "erlang27",
]
# For the initramfs only; never lands in the rootfs.
INITRAMFS_PACKAGES = ["busybox-static"]

# Elixir is precompiled BEAM, portable across OSes; the -otp-27 build matches
# the erlang27 package and the host toolchain that compiles the warden.
ELIXIR_URL = "https://github.com/elixir-lang/elixir/releases/download/v1.18.4/elixir-otp-27.zip"
ELIXIR_SHA256 = "5be18f35e329f7c5914a80dd9f323d7bbb144616df1ed16f6f0862a1900b4bb5"

# Modules the initramfs must carry to mount root: virtio transport, the block
# device, and ext4. The dependency closure is computed from modules.dep, not
# guessed. virtiofs/vsock/rng load later from the rootfs (crucible-init).
INITRAMFS_MODULES = ["virtio_pci", "virtio_blk", "ext4"]

DISK_SIZE = "3G"


def log(msg):
    print(f"mkrootfs: {msg}", flush=True)


def fetch(url, dest, sha256=None):
    if os.path.exists(dest):
        if sha256 is None or file_sha256(dest) == sha256:
            return
        os.unlink(dest)
    log(f"fetch {url}")
    tmp = dest + ".part"
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    if sha256 is not None:
        got = file_sha256(tmp)
        if got != sha256:
            sys.exit(f"mkrootfs: {url}: sha256 {got}, expected {sha256}")
    os.replace(tmp, dest)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---- APKINDEX ---------------------------------------------------------------

def parse_index(data):
    """APKINDEX records -> {name: {V, D, p, C, repo}}, plus a provides map."""
    pkgs, provides = {}, {}
    for rec in data.decode().split("\n\n"):
        fields = {}
        for line in rec.split("\n"):
            if len(line) > 2 and line[1] == ":":
                fields.setdefault(line[0], []).append(line[2:])
        if "P" not in fields:
            continue
        name = fields["P"][0]
        pkgs[name] = fields
        provides.setdefault(name, name)
        for p in fields.get("p", [""])[0].split():
            provides.setdefault(p.split("=")[0], name)
    return pkgs, provides


def resolve(roots, pkgs, provides):
    """Transitive dependency closure. Version constraints are not solved --
    each Alpine release carries one version of everything, which is the pin."""
    want, order = set(), []

    def add(name):
        if name in want:
            return
        if name not in pkgs:
            sys.exit(f"mkrootfs: unknown package {name!r}")
        want.add(name)
        for dep in pkgs[name].get("D", [""])[0].split():
            if dep.startswith("!"):
                continue                      # a conflict, not a dependency
            dep = re.split(r"[<>=~]", dep)[0]
            if dep == "/bin/sh":
                dep = "busybox"
            target = provides.get(dep)
            if target is None:
                sys.exit(f"mkrootfs: nothing provides {dep!r} (wanted by {name})")
            add(target)
        order.append(name)

    for r in roots:
        add(r)
    return order


# ---- extraction -------------------------------------------------------------

def extract_apk(path, rootfs):
    """An .apk is concatenated gzip streams of cut tars: signature, control,
    data. gzip.decompress handles the concatenation; the result reads as one
    tar. Control entries all start with '.' and are skipped."""
    with open(path, "rb") as f:
        blob = gzip.decompress(f.read())
    with tarfile.open(fileobj=io.BytesIO(blob), ignore_zeros=True) as tf:
        for m in tf:
            name = m.name.lstrip("./")
            if not name or name.startswith("."):
                continue
            if ".." in name.split("/"):
                sys.exit(f"mkrootfs: refusing path {m.name!r} in {path}")
            dest = os.path.join(rootfs, name)
            if m.isdir():
                os.makedirs(dest, exist_ok=True)
                os.chmod(dest, m.mode)
            elif m.issym():
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                if os.path.lexists(dest):
                    os.unlink(dest)
                os.symlink(m.linkname, dest)
            elif m.islnk():
                src = os.path.join(rootfs, m.linkname.lstrip("./"))
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                if os.path.lexists(dest):
                    os.unlink(dest)
                os.link(src, dest)
            elif m.isfile():
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                if os.path.lexists(dest):
                    os.unlink(dest)
                with tf.extractfile(m) as src, open(dest, "wb") as out:
                    shutil.copyfileobj(src, out)
                os.chmod(dest, m.mode)
            # devices and fifos: devtmpfs provides them at boot


# ---- initramfs --------------------------------------------------------------

def module_closure(moddir, roots):
    """Load order for `roots` from modules.dep: dependencies before dependents,
    each path once. A root the kernel builds in (modules.builtin) needs no
    loading at all -- linux-virt moves drivers in and out of the kernel across
    releases, and this build must not care which side virtio_pci landed on."""
    deps = {}
    with open(os.path.join(moddir, "modules.dep")) as f:
        for line in f:
            path, _, rest = line.partition(":")
            deps[path.strip()] = rest.split()

    def base(p):
        return os.path.basename(p).split(".ko")[0].replace("-", "_")

    by_name = {base(p): p for p in deps}
    builtin = set()
    with open(os.path.join(moddir, "modules.builtin")) as f:
        for line in f:
            builtin.add(base(line.strip()))

    order, seen = [], set()

    def add(path):
        if path in seen:
            return
        seen.add(path)
        for d in deps.get(path, []):
            add(d)
        order.append(path)

    for r in roots:
        name = r.replace("-", "_")
        if name in by_name:
            add(by_name[name])
        elif name in builtin:
            log(f"initramfs: {r} is built into this kernel")
        else:
            sys.exit(f"mkrootfs: module {r} in neither modules.dep nor modules.builtin")
    return order


def cpio_newc(entries):
    """entries: [(path, mode, data_or_None_for_dir)] -> newc cpio archive."""
    out = io.BytesIO()
    ino = 0

    def emit(name, mode, data):
        nonlocal ino
        ino += 1
        body = data or b""
        # newc: magic then 13 8-digit hex fields -- ino, mode, uid, gid,
        # nlink, mtime, filesize, devmajor, devminor, rdevmajor, rdevminor,
        # namesize, check. The kernel silently ignores a malformed archive
        # and falls back to mounting root itself, so a field miscount here
        # presents as a boot panic three layers away.
        fields = [ino, mode, 0, 0, 1, 0, len(body), 0, 0, 0, 0, len(name) + 1, 0]
        hdr = ("070701" + "".join(f"{v:08x}" for v in fields)).encode()
        rec = hdr + name.encode() + b"\0"
        rec += b"\0" * ((4 - len(rec) % 4) % 4)
        rec += body
        rec += b"\0" * ((4 - len(rec) % 4) % 4)
        out.write(rec)

    for path, mode, data in entries:
        emit(path, mode, data)
    emit("TRAILER!!!", 0, b"")
    return out.getvalue()


def build_initramfs(rootfs, busybox_static, out_path):
    kver = os.listdir(os.path.join(rootfs, "lib/modules"))[0]
    moddir = os.path.join(rootfs, "lib/modules", kver)
    modules = module_closure(moddir, INITRAMFS_MODULES)
    log(f"initramfs: kernel {kver}, {len(modules)} modules")

    insmod = []
    entries = [("dev", 0o40755, None), ("proc", 0o40755, None),
               ("sys", 0o40755, None), ("mnt", 0o40755, None),
               ("bin", 0o40755, None), ("lib", 0o40755, None),
               ("lib/modules", 0o40755, None)]
    with open(busybox_static, "rb") as f:
        entries.append(("bin/busybox", 0o100755, f.read()))

    for rel in modules:
        # Alpine compresses modules; the kernel loads plain .ko from anywhere,
        # so decompress at build time and insmod needs no compression support.
        src = os.path.join(moddir, rel)
        name = os.path.basename(rel)
        if name.endswith(".gz"):
            data, name = gzip.decompress(open(src, "rb").read()), name[:-3]
        elif name.endswith(".xz"):
            import lzma
            data, name = lzma.decompress(open(src, "rb").read()), name[:-3]
        else:
            data = open(src, "rb").read()
        entries.append((f"lib/modules/{name}", 0o100644, data))
        insmod.append(f"bb insmod /lib/modules/{name}")

    init = "\n".join([
        "#!/bin/busybox sh",
        "# Generated by mkrootfs.py. Mount root, hand over. Nothing else.",
        "bb() { /bin/busybox \"$@\"; }",
        "bb mount -t devtmpfs devtmpfs /dev",
        "bb mount -t proc proc /proc",
        *insmod,
        'root=/dev/vda; init=/sbin/crucible-init; flag=rw',
        'for a in $(bb cat /proc/cmdline); do case "$a" in',
        '  root=*) root=${a#root=};; init=*) init=${a#init=};; ro) flag=ro;;',
        "esac; done",
        'bb mount -o "$flag" -t ext4 "$root" /mnt || { echo "initramfs: cannot mount $root"; bb sh; }',
        # /proc moves across rather than being unmounted: the first thing the
        # real init does is remount and busybox mount needs /proc/mounts to
        # know what / is.
        "bb mount -o move /proc /mnt/proc",
        "bb umount /dev",
        # exec resolves a real path, not a shell function.
        'exec /bin/busybox switch_root /mnt "$init"',
        "",
    ])
    entries.append(("init", 0o100755, init.encode()))

    with open(out_path, "wb") as f:
        f.write(gzip.compress(cpio_newc(entries), 6))
    log(f"initramfs: {os.path.getsize(out_path) / 1e6:.1f} MB")


# ---- kernel -----------------------------------------------------------------

def unwrap_zboot(src, dst):
    """Alpine's aarch64 kernel is an EFI zboot wrapper, which
    Virtualization.framework cannot boot; VZLinuxBootLoader wants the raw
    arm64 Image (ARM\\x64 at 0x38)."""
    d = open(src, "rb").read()
    if d[:2] != b"MZ" or d[4:8] != b"zimg":
        sys.exit(f"mkrootfs: {src} is not an EFI zboot image ({d[:8].hex()})")
    off, size = struct.unpack_from("<II", d, 8)
    comp = d[24:36].split(b"\0")[0].decode()
    payload = d[off:off + size]
    if comp == "gzip":
        raw = gzip.decompress(payload)
    elif comp in ("lzma", "xz"):
        import lzma
        raw = lzma.decompress(payload)
    else:
        sys.exit(f"mkrootfs: unsupported zboot compression {comp!r}")
    if raw[0x38:0x3c] != b"ARM\x64":
        sys.exit("mkrootfs: decompressed payload is not an arm64 Image")
    open(dst, "wb").write(raw)
    log(f"kernel: vmlinuz -> Image ({comp}, {len(raw) / 1e6:.1f} MB)")


# ---- disk -------------------------------------------------------------------

def find_mke2fs():
    for p in ["/opt/homebrew/opt/e2fsprogs/sbin", "/usr/local/opt/e2fsprogs/sbin",
              "/opt/homebrew/sbin", "/usr/local/sbin"]:
        for tool in ["mke2fs"]:
            c = os.path.join(p, tool)
            if os.path.exists(c):
                return p
    sys.exit("mkrootfs: mke2fs not found; brew install e2fsprogs")


def make_disk(rootfs, out_path):
    sbin = find_mke2fs()
    if os.path.exists(out_path):
        os.unlink(out_path)
    subprocess.run([os.path.join(sbin, "mke2fs"), "-F", "-q", "-L", "crucible",
                    "-t", "ext4", "-d", rootfs, out_path, DISK_SIZE], check=True)
    fsck = os.path.join(sbin, "e2fsck")
    if os.path.exists(fsck):
        subprocess.run([fsck, "-fp", out_path], check=False)
    sparsify(out_path)
    st = os.stat(out_path)
    log(f"disk: apparent {st.st_size / 1e9:.2f} GB, "
        f"allocated {st.st_blocks * 512 / 1e6:.0f} MB")


def sparsify(path):
    """PLAN.md 6.3's argument is that a session clone is cheap because the
    image is mostly holes; make sure they are holes."""
    block = 1 << 20
    tmp = path + ".sparse"
    zero = b"\0" * block
    with open(path, "rb") as src, open(tmp, "wb") as dst:
        while True:
            buf = src.read(block)
            if not buf:
                break
            if buf == zero[: len(buf)]:
                dst.seek(len(buf), os.SEEK_CUR)
            else:
                dst.write(buf)
        dst.truncate(os.path.getsize(path))
    os.replace(tmp, path)


# ---- layout -----------------------------------------------------------------

def install_elixir(cache, rootfs):
    dest = os.path.join(cache, "elixir-otp-27.zip")
    fetch(ELIXIR_URL, dest, ELIXIR_SHA256)
    target = os.path.join(rootfs, "usr/lib/elixir")
    shutil.rmtree(target, ignore_errors=True)
    with zipfile.ZipFile(dest) as z:
        for m in z.infolist():
            out = z.extract(m, target)
            # zipfile drops the unix mode; the launchers must be executable.
            mode = (m.external_attr >> 16) & 0o7777
            if mode:
                os.chmod(out, mode)
            elif m.filename.startswith("bin/"):
                os.chmod(out, 0o755)
    for b in ["elixir", "elixirc", "mix", "iex"]:
        link = os.path.join(rootfs, "usr/bin", b)
        if os.path.lexists(link):
            os.unlink(link)
        os.symlink(f"../lib/elixir/bin/{b}", link)
    # ERL_LIBS for the warden's `erl -pa` start (crucible-init) and for the
    # workspace node: elixir, logger, mix and the rest are separate OTP apps.
    with open(os.path.join(rootfs, "etc/crucible-erl-libs"), "w") as f:
        f.write("/usr/lib/elixir/lib\n")


def install_local(rootfs, guest_dir, vsock_port, warden_build):
    sbin = os.path.join(rootfs, "sbin")
    os.makedirs(sbin, exist_ok=True)
    for script in ["crucible-init", "seed-work"]:
        dst = os.path.join(sbin, script)
        shutil.copy(os.path.join(guest_dir, "init", script), dst)
        os.chmod(dst, 0o755)

    dst = os.path.join(rootfs, "usr/local/bin/vsock_port")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy(vsock_port, dst)
    os.chmod(dst, 0o755)

    warden = os.path.join(rootfs, "opt/warden")
    shutil.rmtree(warden, ignore_errors=True)
    os.makedirs(warden)
    shutil.copytree(os.path.join(warden_build, "ebin"), os.path.join(warden, "ebin"))
    shutil.copytree(os.path.join(guest_dir, "warden/lib"), os.path.join(warden, "lib"))
    shutil.copy(os.path.join(guest_dir, "warden/mix.exs"), warden)

    # The agent commits its work (spec 7.4a), and git refuses to commit
    # without an identity; the answer must not depend on the model
    # remembering to configure one.
    with open(os.path.join(rootfs, "root/.gitconfig"), "w") as f:
        f.write("[user]\n\tname = Crucible Agent\n\temail = agent@crucible.invalid\n"
                "[safe]\n\tdirectory = /work\n")

    # The one apk trigger this build cares about, done by hand: a shell at
    # /bin/sh so crucible-init can start at all. crucible-init's first act is
    # `/bin/busybox --install -s`, which builds the rest of the applet farm.
    sh = os.path.join(rootfs, "bin/sh")
    if not os.path.lexists(sh):
        os.symlink("/bin/busybox", sh)


def main():
    guest_dir = os.path.dirname(os.path.abspath(__file__))
    out = os.environ.get("OUT") or os.path.join(guest_dir, "../build/guest")
    cache = os.environ.get("CACHE") or os.path.join(guest_dir, "../build/guest-cache")
    vsock_port = os.environ["VSOCK_PORT"]
    warden_build = os.environ["WARDEN_EBIN"]
    rootfs = os.path.join(cache, "rootfs")

    os.makedirs(out, exist_ok=True)
    os.makedirs(cache, exist_ok=True)
    shutil.rmtree(rootfs, ignore_errors=True)
    os.makedirs(rootfs)

    pkgs, provides = {}, {}
    urls = {}
    for repo in REPOS:
        idx = os.path.join(cache, f"APKINDEX.{repo}.tar.gz")
        fetch(f"{MIRROR}/{ALPINE}/{repo}/{ARCH}/APKINDEX.tar.gz", idx)
        with tarfile.open(idx) as tf:
            data = tf.extractfile("APKINDEX").read()
        p, pr = parse_index(data)
        for name, rec in p.items():
            if name not in pkgs:
                pkgs[name] = rec
                urls[name] = f"{MIRROR}/{ALPINE}/{repo}/{ARCH}"
        for k, v in pr.items():
            provides.setdefault(k, v)

    order = resolve(PACKAGES, pkgs, provides)
    log(f"{len(order)} packages: {' '.join(order)}")
    for name in order:
        ver = pkgs[name]["V"][0]
        apk = os.path.join(cache, f"{name}-{ver}.apk")
        fetch(f"{urls[name]}/{name}-{ver}.apk", apk)
        extract_apk(apk, rootfs)

    install_elixir(cache, rootfs)
    install_local(rootfs, guest_dir, vsock_port, warden_build)

    # busybox-static, for the initramfs only.
    bb = pkgs["busybox-static"]
    apk = os.path.join(cache, f"busybox-static-{bb['V'][0]}.apk")
    fetch(f"{urls['busybox-static']}/busybox-static-{bb['V'][0]}.apk", apk)
    tmp = os.path.join(cache, "busybox-static-root")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    extract_apk(apk, tmp)
    build_initramfs(rootfs, os.path.join(tmp, "bin/busybox.static"),
                    os.path.join(out, "initramfs"))

    vmlinuz = os.path.join(rootfs, "boot/vmlinuz-virt")
    shutil.copy(vmlinuz, os.path.join(out, "vmlinuz"))
    unwrap_zboot(vmlinuz, os.path.join(out, "Image"))
    # /boot in the image is dead weight once the host holds the kernel.
    shutil.rmtree(os.path.join(rootfs, "boot"), ignore_errors=True)

    make_disk(rootfs, os.path.join(out, "disk.img"))

    log("guest image contents:")
    rel = os.path.join(rootfs, 'etc/alpine-release')
    log(f"  alpine:  {open(rel).read().strip() if os.path.exists(rel) else '?'}")
    log(f"  kernel:  {os.listdir(os.path.join(rootfs, 'lib/modules'))[0]}")
    log(f"  erlang:  {pkgs['erlang27']['V'][0]},  elixir: {ELIXIR_URL.rsplit('/v', 1)[1].split('/')[0]}-otp-27")
    du = subprocess.run(["du", "-sh", rootfs], capture_output=True, text=True)
    log(f"  rootfs:  {du.stdout.split()[0]}")


if __name__ == "__main__":
    main()
