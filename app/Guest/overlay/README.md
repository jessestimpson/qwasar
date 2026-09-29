# The guest overlay

Everything in this folder except this README is copied over the guest's
root filesystem when the image is built, as is: `usr/local/bin/mytool` here
lands at `/usr/local/bin/mytool` in the sandbox, which is on its PATH.
Modes and symlinks are kept. The folder is git-ignored apart from this file,
so what you put here is yours.

The guest is **Alpine Linux on arm64**, with no network device. What goes
here must run there without fetching anything:

- **Statically linked Linux arm64 binaries** (most Go and Zig tools, many
  Rust ones built for `aarch64-unknown-linux-musl`), or ones built against
  musl.
- **Portable bytecode and scripts**: escripts and `.beam` files for OTP 27,
  shell, Python 3 and Node scripts.
- **Not** macOS binaries (your `~/.local/share/mise` installs, Homebrew),
  and not glibc builds (most prebuilt Linux downloads): neither runs here.

Alpine packages are easier when one exists: `GUEST_PACKAGES="go rust"`
when building the image. The overlay may not replace the init, the
`mount-work` gate, the warden or `vsock_port`; the build refuses.

Rebuild after changing anything here:

    make guest && make run

`GUEST_OVERLAY=<dir>` uses another folder instead of this one.
