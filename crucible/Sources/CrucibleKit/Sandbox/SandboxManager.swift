// SandboxManager.swift -- a guest per session, and the channel to it.
//
// PLAN.md 6.5: the VM is created lazily, on the session's first tool call, so a
// conversation that never touches a file never boots one. It is capped, because
// a running guest is ~2 GB of the same unified memory the model wants (§2.3).
//
// Everything Virtualization touches is main-actor work (§3.4, fifth edge). The
// channel that comes back is not: it owns a file descriptor and a reader
// thread, so the engine queue can use it without hopping actors.

import Foundation

@MainActor
public final class SandboxManager {
    public struct Ready: Sendable {
        public let channel: VsockChannel
        public let bootSeconds: Double
        public let cloneMethod: GuestImage.CloneMethod
    }

    private struct Live {
        let host: SandboxHost
        let channel: VsockChannel
        let scratch: URL
    }

    private var live: [UUID: Live] = [:]
    private let guestDir: URL
    private let stateDir: URL
    public var maxRunning = 2

    public init(guestDir: URL, stateDir: URL) {
        self.guestDir = guestDir
        self.stateDir = stateDir
    }

    public var isAvailable: Bool {
        let fm = FileManager.default
        return ["disk.img", "Image", "initramfs"].allSatisfy {
            fm.fileExists(atPath: guestDir.appendingPathComponent($0).path)
        }
    }

    public func channel(for id: UUID) -> VsockChannel? { live[id]?.channel }

    /// Boots a guest for `id`, sharing `projectRoot` into it read-only.
    ///
    /// The guest copies that share into `/work` and unmounts it, so the tools
    /// operate on the copy and the user's tree is never writable from inside.
    public func start(session id: UUID, projectRoot: URL) async throws -> Ready {
        if let existing = live[id] {
            return Ready(channel: existing.channel, bootSeconds: 0, cloneMethod: .copyOnWrite)
        }
        guard isAvailable else {
            throw SandboxError.noImage("\(guestDir.path) — run `make guest`")
        }
        if live.count >= maxRunning, let victim = live.keys.first {
            await stop(session: victim)
        }

        let scratch = stateDir.appendingPathComponent("sandboxes/\(id.uuidString)")
        try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)

        // A session's disk is the model's work: /work lives on it, and so does
        // the git baseline that `propose` diffs against. It is CLONED ONCE, on
        // the session's first boot, and reused for every boot after that.
        //
        // Re-cloning here unconditionally -- which is what this did -- silently
        // destroyed everything the model had done, on every app restart and on
        // every eviction under `maxRunning`. `GuestImage.clone` removes its
        // destination before calling clonefile(2), so there was not even a
        // partial survival: the disk was deleted and replaced with a pristine
        // one. `discard(session:)` remains the only way to lose a sandbox, and
        // it is called only when a person deletes the session.
        let disk = scratch.appendingPathComponent("disk.img")
        let image = GuestImage(golden: guestDir.appendingPathComponent("disk.img"))
        let clone = try image.provision(at: disk)

        var config = SandboxConfig(diskImage: clone.url,
                                   kernel: guestDir.appendingPathComponent("Image"),
                                   initramfs: guestDir.appendingPathComponent("initramfs"),
                                   consoleLog: scratch.appendingPathComponent("console.log"),
                                   baseDirectory: projectRoot)
        config.memoryBytes = 2 * 1024 * 1024 * 1024
        config.cpuCount = 2

        let host = SandboxHost(config: config)
        let t0 = Date()
        try await host.start()

        // The warden comes up when it comes up; the host retries rather than
        // assuming a boot order it does not control. Measured at ~0.55s.
        var fd: Int32 = -1
        while Date().timeIntervalSince(t0) < 30 {
            if let got = try? await host.connectControl() { fd = got; break }
            try? await Task.sleep(for: .milliseconds(100))
        }
        guard fd >= 0 else {
            await host.stop()
            throw SandboxError.start("the warden did not answer within 30s")
        }

        let channel = VsockChannel(fileDescriptor: fd)
        channel.start()
        live[id] = Live(host: host, channel: channel, scratch: scratch)

        return Ready(channel: channel,
                     bootSeconds: Date().timeIntervalSince(t0),
                     cloneMethod: clone.method)
    }

    public func stop(session id: UUID) async {
        guard let l = live.removeValue(forKey: id) else { return }
        l.channel.close()
        await l.host.stop()
    }

    /// Removes a session's guest disk. Its own command, because it is the one
    /// action here that destroys work the model did.
    public func discard(session id: UUID) async {
        await stop(session: id)
        let scratch = stateDir.appendingPathComponent("sandboxes/\(id.uuidString)")
        try? FileManager.default.removeItem(at: scratch)
    }

    public func stopAll() async {
        for id in live.keys { await stop(session: id) }
    }
}
