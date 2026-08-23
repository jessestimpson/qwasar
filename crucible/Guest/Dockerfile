# The Crucible guest image.
#
# PLAN.md 6.2. Built in a container because building a Linux rootfs needs Linux;
# the shipped app never needs Docker, and neither does anything at runtime. This
# is a build-time step whose only output is three files.
#
# Alpine for the operating system, mise for the language runtimes. The split is
# deliberate and is explained where mise appears below.

ARG ALPINE=3.22
ARG MISE=2026.8.10
# Runtimes mise manages, pinned. One line to edit to add a language.
#
# MEASURED CONSTRAINT: this guest is musl, and mise's catalogue is largely
# glibc. erlang builds from source and works (1m39s); elixir is precompiled
# .beam files and works (1.2s); node@22 FAILS, because Node ships no musl
# binaries and mise falls back to a source build that needs a toolchain this
# image does not carry. So mise handles what mise can handle here, and the
# musl-native runtimes come from apk below. Moving the guest to a glibc base
# would make the whole mise catalogue available; that trade is recorded in
# PLAN.md 6.2 rather than made silently.
ARG RUNTIMES="erlang@27.3.4.16 elixir@1.18.4-otp-27"

FROM alpine:${ALPINE} AS builder
ARG ALPINE
ARG MISE
ARG RUNTIMES

# Builder tools only. None of this reaches the guest, except the runtimes mise
# builds, which are copied in wholesale.
#
# The long list is what kerl needs to build OTP from source: mise's erlang
# backend is a source build, measured at 1m39s on this machine. It is cached in
# a layer, so it costs that once per pinned version.
RUN apk add --no-cache \
        e2fsprogs \
        gcc musl-dev linux-headers \
        mkinitfs \
        build-base autoconf ncurses-dev openssl-dev zlib-dev libxslt perl \
        bash git \
        xz curl unzip

# ---- mise, and the runtimes it manages --------------------------------------
#
# Language runtimes come from mise rather than from apk, for three reasons that
# all point the same way:
#
#   - Alpine's `elixir` package depends on erlang26 and CONFLICTS with erlang27,
#     so apk cannot give us OTP 27 and Elixir together at all. That is what
#     forced the question.
#   - The guest exists to work on the user's project, and projects pin their own
#     runtime versions. mise reads .tool-versions and mise.toml out of /work, so
#     the agent gets the version the project actually asks for.
#   - Adding Go, Rust or another Node becomes one word in $RUNTIMES rather than
#     a hunt for an Alpine package that carries it.
#
# Installed into /opt/mise in the BUILDER and copied across, rather than
# installed into $ROOTFS directly: an Erlang built from source bakes its ROOTDIR
# into the `erl` script, so it has to be built at the path it will live at.
RUN set -eux; \
    curl -fsSL -o /usr/local/bin/mise \
      "https://github.com/jdx/mise/releases/download/v${MISE}/mise-v${MISE}-linux-arm64-musl"; \
    chmod +x /usr/local/bin/mise; \
    mise --version

ENV MISE_DATA_DIR=/opt/mise \
    MISE_CONFIG_DIR=/etc/mise \
    MISE_CACHE_DIR=/tmp/mise-cache \
    MISE_YES=1

RUN set -eux; \
    mkdir -p /etc/mise; \
    { echo "[tools]"; \
      for t in $RUNTIMES; do echo "\"${t%@*}\" = \"${t#*@}\""; done; } \
      > /etc/mise/config.toml; \
    cat /etc/mise/config.toml; \
    mise install; \
    mise ls --installed

# ---- the guest rootfs -------------------------------------------------------
#
# apk --root builds it as a directory tree; no chroot, no privileges, no loop
# devices. The keys and repository list have to be planted first or apk has
# nothing to verify signatures against.
ENV ROOTFS=/rootfs
RUN mkdir -p $ROOTFS/etc/apk \
 && cp -r /etc/apk/keys $ROOTFS/etc/apk/ \
 && printf '%s\n' \
      "https://dl-cdn.alpinelinux.org/alpine/v${ALPINE}/main" \
      "https://dl-cdn.alpinelinux.org/alpine/v${ALPINE}/community" \
      > $ROOTFS/etc/apk/repositories

# The operating system and the tools the agent uses to look at code. No language
# runtimes here -- those are mise's job. The -libs packages are what the
# mise-built Erlang links against at runtime.
RUN apk add --root $ROOTFS --initdb --no-cache \
        alpine-baselayout busybox busybox-suid musl \
        linux-virt mkinitfs \
        libstdc++ ncurses-libs openssl zlib libcrypto3 libssl3 \
        git ripgrep \
        nodejs npm python3 \
        e2fsprogs-extra

# mise and its runtimes, moved into the guest at the path they were built for.
RUN set -eux; \
    cp -a /opt/mise $ROOTFS/opt/mise; \
    mkdir -p $ROOTFS/etc/mise $ROOTFS/etc/profile.d; \
    cp /etc/mise/config.toml $ROOTFS/etc/mise/config.toml; \
    cp /usr/local/bin/mise $ROOTFS/usr/local/bin/mise; \
    printf 'export MISE_DATA_DIR=/opt/mise\nexport MISE_CONFIG_DIR=/etc/mise\nexport PATH="/opt/mise/shims:$PATH"\n' \
      > $ROOTFS/etc/profile.d/mise.sh; \
    mise reshim 2>/dev/null || true; \
    cp -a /opt/mise/shims $ROOTFS/opt/mise/shims 2>/dev/null || true

# Stable shims for the runtime the CONTROL PLANE uses, resolved once, here, to a
# concrete path.
#
# PLAN.md 7.3: warden must survive anything the agent does. If it started
# through `mise exec`, an agent that edited a mise config -- or simply switched
# the project's Erlang version -- would take the control plane down with it, and
# the init loop would spin restarting something that can no longer start. So the
# warden's runtime is pinned to a path at build time. mise stays what the agent
# and the project use.
RUN set -eux; \
    ERL_ROOT="$(ls -d /opt/mise/installs/erlang/*/ | head -1)"; \
    ELIXIR_ROOT="$(ls -d /opt/mise/installs/elixir/*/ | head -1)"; \
    echo "control-plane erlang: $ERL_ROOT"; \
    echo "control-plane elixir: $ELIXIR_ROOT"; \
    for b in erl erlc escript epmd; do ln -sf "${ERL_ROOT}bin/$b" "$ROOTFS/usr/bin/$b"; done; \
    for b in elixir elixirc mix iex; do ln -sf "${ELIXIR_ROOT}bin/$b" "$ROOTFS/usr/bin/$b"; done; \
    echo "${ELIXIR_ROOT}lib" > $ROOTFS/etc/crucible-erl-libs; \
    ls -l $ROOTFS/usr/bin/erl $ROOTFS/usr/bin/elixir

# ---- vsock bridge -----------------------------------------------------------
#
# Static so it has no opinion about which libc the rootfs ended up with.
COPY vsock_port/vsock_port.c /tmp/vsock_port.c
RUN cc -O2 -Wall -Wextra -Werror -static -o $ROOTFS/usr/local/bin/vsock_port /tmp/vsock_port.c \
 && ls -l $ROOTFS/usr/local/bin/vsock_port

# ---- init and warden --------------------------------------------------------
COPY init/crucible-init $ROOTFS/sbin/crucible-init
COPY init/seed-work     $ROOTFS/sbin/seed-work
COPY init/test_seed_work.sh /tmp/test_seed_work.sh
COPY warden/            /src/warden/
RUN chmod +x $ROOTFS/sbin/crucible-init

# Does an hour of the model's work survive a restart?
#
# The one piece of guest logic that can destroy something the user cannot get
# back, so it is a build gate rather than a comment. Runs against a temp tree --
# no VM needed -- and asserts CONTENT, because `cp -a /base/. /work/` adds files
# and overwrites changed ones: a test that only checked for a file's presence
# would pass while every edit the model made was silently reverted.
RUN SEED_WORK=$ROOTFS/sbin/seed-work sh /tmp/test_seed_work.sh

# The warden: linted, simulated, then compiled.
#
# The order is the point. `make guest` cannot produce an image unless the
# control plane is fully instrumented (lint_sim.sh) and its deterministic
# simulation is green across a seed sweep (PLAN.md 9.2). A control plane that
# ships without those having passed is a control plane nobody checked.
#
# MIX_ENV=prod is what ships, and the build asserts it carries no eta at all --
# which is the property PLAN.md 9.5's exit depends on.
ENV PATH="/opt/mise/installs/elixir/1.18.4-otp-27/bin:/opt/mise/installs/erlang/27.3.4.16/bin:${PATH}" \
    MIX_HOME=/tmp/mix HEX_HOME=/tmp/hex
RUN set -eux; \
    cd /src/warden; \
    mix local.hex --force >/dev/null; \
    mix local.rebar --force >/dev/null; \
    \
    : '--- the control plane must be fully instrumented ---'; \
    sh lint_sim.sh; \
    \
    : '--- the logic: the edit contract, confinement, ceilings ---'; \
    MIX_ENV=test mix test --no-start; \
    \
    : '--- and the simulation must be green before anything ships ---'; \
    MIX_ENV=dst mix deps.get >/dev/null; \
    MIX_ENV=dst mix test --no-start; \
    \
    : '--- what ships: no eta anywhere in it ---'; \
    MIX_ENV=prod mix compile; \
    if [ -d _build/prod/lib/eta ]; then \
      echo "prod build contains eta; the seam is broken" >&2; exit 1; \
    fi; \
    echo "prod build carries: $(ls _build/prod/lib | tr '\n' ' ')"; \
    \
    : '--- the sim node DOES ship eta, and only the sim node ---'; \
    mkdir -p $ROOTFS/opt/sim/lib; \
    for app in _build/dst/lib/*; do \
      mkdir -p "$ROOTFS/opt/sim/lib/$(basename $app)"; \
      cp -a "$app/ebin" "$ROOTFS/opt/sim/lib/$(basename $app)/ebin"; \
    done; \
    echo "sim node carries: $(ls $ROOTFS/opt/sim/lib | tr '\n' ' ')"; \
    mkdir -p $ROOTFS/opt/warden; \
    cp -a _build/prod/lib/warden/ebin $ROOTFS/opt/warden/ebin; \
    cp -a lib mix.exs $ROOTFS/opt/warden/; \
    ls $ROOTFS/opt/warden/ebin

# ---- initramfs --------------------------------------------------------------
#
# virtio_blk and ext4 are modules in linux-virt (measured), so root cannot be
# mounted without one. Built against the rootfs's own modules rather than the
# builder's, or it would load drivers that are not there.
RUN set -eux; \
    KVER="$(ls $ROOTFS/lib/modules | head -1)"; \
    echo "kernel: $KVER"; \
    mkinitfs -b $ROOTFS -F "base virtio ext4 virtiofs" -o /tmp/initramfs "$KVER"; \
    cp $ROOTFS/boot/vmlinuz-virt /tmp/vmlinuz

# ---- the disk ---------------------------------------------------------------
#
# A bare ext4 filesystem, no partition table: VZLinuxBootLoader takes the kernel
# and initramfs from the host, so the disk only has to be a root filesystem.
# Sparse; PLAN.md 6.3 clones it per session and pays only for the delta.
RUN set -eux; \
    truncate -s 3G /tmp/disk.img; \
    mkfs.ext4 -F -q -L crucible -d $ROOTFS /tmp/disk.img; \
    e2fsck -fp /tmp/disk.img || true

RUN set -eux; \
    echo "=== guest image contents"; \
    echo "alpine:   $(cat $ROOTFS/etc/alpine-release)"; \
    echo "kernel:   $(ls $ROOTFS/lib/modules)"; \
    echo "mise:     $(mise --version)"; \
    echo "mise:     $RUNTIMES"; \
    echo "apk:      $(chroot $ROOTFS /usr/bin/node --version 2>/dev/null || echo node?) $(chroot $ROOTFS /usr/bin/python3 --version 2>/dev/null || echo python?)"; \
    echo "rootfs:   $(du -sh $ROOTFS | cut -f1)"; \
    echo "disk:     $(du -h /tmp/disk.img | cut -f1) allocated"

# ---- output -----------------------------------------------------------------
FROM scratch AS artifacts
COPY --from=builder /tmp/vmlinuz    /vmlinuz
COPY --from=builder /tmp/initramfs  /initramfs
COPY --from=builder /tmp/disk.img   /disk.img
