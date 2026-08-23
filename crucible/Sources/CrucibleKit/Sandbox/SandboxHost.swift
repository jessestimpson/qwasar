// SandboxHost.swift -- one guest VM per session.
//
// PLAN.md 6.1. The configuration is as much about what is absent as what is
// present: there is no VZNetworkDeviceConfiguration, so the guest has no
// network device to configure rather than a firewall rule it might talk its way
// around. That omission is the strongest security property in the design and it
// costs one line not written.
//
// Everything in Virtualization.framework has main-queue affinity. PLAN.md 3.4's
// fifth sharp edge was learned here: a VZVirtualMachine constructed off the
// main queue while the main thread waits deadlocks silently. So this type is
// @MainActor, without exception, and the long-running work it triggers happens
// on the guest's side of the wire rather than on this thread.

import Foundation
import Virtualization

public enum SandboxError: Error, CustomStringConvertible {
    case noEntitlement
    case configuration(String)
    case noImage(String)
    case start(String)
    case notRunning

    public var description: String {
        switch self {
        case .noEntitlement:      return "this build lacks com.apple.security.virtualization"
        case .configuration(let m): return "invalid VM configuration: \(m)"
        case .noImage(let p):     return "no guest image at \(p)"
        case .start(let m):       return "the guest did not start: \(m)"
        case .notRunning:         return "the guest is not running"
        }
    }
}

public struct SandboxConfig: Sendable {
    /// Per-VM memory. PLAN.md 2.3 prices this against the model's share of a
    /// unified pool, so it is small on purpose.
    public var memoryBytes: UInt64 = 2 * 1024 * 1024 * 1024
    public var cpuCount: Int = 2
    /// The session's working directory, mounted read-only as `base`. The guest
    /// copies it into its own disk and never writes here.
    public var baseDirectory: URL?
    /// Copy-on-write clone of the golden image (PLAN.md 6.3).
    public var diskImage: URL
    /// Kernel and initramfs, taken from the host rather than the disk.
    public var kernel: URL
    public var initramfs: URL
    /// Boot log, for the inspector and for support.
    public var consoleLog: URL

    /// virtio_blk and ext4 are modules in Alpine's linux-virt (measured while
    /// building the image), so root cannot be mounted without the initramfs,
    /// and the console is hvc0 because CONFIG_VIRTIO_CONSOLE is built in.
    public var kernelCommandLine =
        "root=/dev/vda rw rootfstype=ext4 console=hvc0 init=/sbin/crucible-init quiet"

    public init(diskImage: URL, kernel: URL, initramfs: URL,
                consoleLog: URL, baseDirectory: URL? = nil) {
        self.diskImage = diskImage
        self.kernel = kernel
        self.initramfs = initramfs
        self.consoleLog = consoleLog
        self.baseDirectory = baseDirectory
    }
}

@MainActor
public final class SandboxHost: NSObject {
    public private(set) var config: SandboxConfig
    private var vm: VZVirtualMachine?
    /// The live control connection.
    ///
    /// Held because VZVirtioSocketConnection OWNS its file descriptor and
    /// closes it on deallocation. Returning the fd and letting the connection
    /// go out of scope produces a socket that connects and then immediately
    /// hangs up -- which reads, from both ends, exactly like the guest closing
    /// the channel.
    private var control: VZVirtioSocketConnection?
    private var stopWaiters: [StopWaiter] = []

    /// Resumes at most once, whichever arrives first: the guest actually
    /// stopping, or the grace period expiring.
    private final class StopWaiter {
        private var cont: CheckedContinuation<Void, Never>?
        init(_ c: CheckedContinuation<Void, Never>) { cont = c }
        func fire() {
            guard let c = cont else { return }
            cont = nil
            c.resume()
        }
    }

    /// The port the guest's warden listens on (PLAN.md 6.4).
    public static let controlPort: UInt32 = 1024

    public enum State: Sendable, Equatable {
        case stopped
        case starting
        case running
        case failed(String)
    }
    public private(set) var state: State = .stopped

    public init(config: SandboxConfig) {
        self.config = config
        super.init()
    }

    // MARK: Configuration

    /// Builds and validates the configuration without starting anything.
    ///
    /// Separate from `start()` because `validate()` gives a real message for a
    /// misconfiguration where starting gives a trap, and because the M2 gate
    /// wants to check the shape of what we ship without booting it.
    public func makeConfiguration() throws -> VZVirtualMachineConfiguration {
        guard Diagnostics.hasVirtualization else { throw SandboxError.noEntitlement }
        let fm = FileManager.default
        guard fm.fileExists(atPath: config.diskImage.path) else {
            throw SandboxError.noImage(config.diskImage.path)
        }

        let c = VZVirtualMachineConfiguration()
        c.cpuCount = max(1, min(config.cpuCount,
                                VZVirtualMachineConfiguration.maximumAllowedCPUCount))
        c.memorySize = min(max(config.memoryBytes,
                               VZVirtualMachineConfiguration.minimumAllowedMemorySize),
                           VZVirtualMachineConfiguration.maximumAllowedMemorySize)

        // A kernel and an initramfs, not EFI.
        //
        // PLAN.md 6.1 chose VZEFIBootLoader so the guest could own its own
        // kernel updates and the image could be one artefact. On contact that
        // buys a GPT, an ESP, a FAT filesystem and a bootloader to install into
        // it -- all to arrive at a kernel the host already has on disk.
        // VZLinuxBootLoader takes vmlinuz and initramfs directly, which makes
        // the disk a bare ext4 root filesystem with no partition table at all.
        // Simpler to build, simpler to read, and the guest's kernel is now
        // versioned with the image that was built against it, which is the more
        // honest arrangement anyway.
        guard fm.fileExists(atPath: config.kernel.path) else {
            throw SandboxError.noImage(config.kernel.path)
        }
        // The framework wants a raw arm64 Image. Alpine ships an EFI zboot
        // wrapper, and handing that to VZLinuxBootLoader fails at start() with
        // "Internal Virtualization error" and nothing else -- an hour of
        // guessing. Checked here, where the message can say what is wrong.
        try checkKernelShape(config.kernel)
        let loader = VZLinuxBootLoader(kernelURL: config.kernel)
        loader.initialRamdiskURL = config.initramfs
        loader.commandLine = config.kernelCommandLine
        c.bootLoader = loader

        // Disk: the per-session APFS clone.
        do {
            let attachment = try VZDiskImageStorageDeviceAttachment(url: config.diskImage,
                                                                    readOnly: false)
            c.storageDevices = [VZVirtioBlockDeviceConfiguration(attachment: attachment)]
        } catch {
            throw SandboxError.configuration("disk: \(error.localizedDescription)")
        }

        // The project, read-only. Enforced by the framework rather than by our
        // code, which is why this is a share and not a copy over the wire.
        if let base = config.baseDirectory {
            let share = VZSingleDirectoryShare(directory: VZSharedDirectory(url: base, readOnly: true))
            let fsDevice = VZVirtioFileSystemDeviceConfiguration(tag: "base")
            fsDevice.share = share
            c.directorySharingDevices = [fsDevice]
        }

        // The control channel.
        c.socketDevices = [VZVirtioSocketDeviceConfiguration()]

        // Entropy. A guest with no randomness source is a bad thing to build
        // on, so it gets a device and the init loads virtio_rng for it.
        //
        // Recorded so it is not retried: this was the first suspect for a
        // five-second BEAM startup and it was the wrong one. The cause was a
        // hostname lookup with no /etc/hosts entry; see Guest/init/crucible-init.
        c.entropyDevices = [VZVirtioEntropyDeviceConfiguration()]

        // Console to a file, so a boot failure is readable after the fact.
        let console = VZVirtioConsoleDeviceSerialPortConfiguration()
        FileManager.default.createFile(atPath: config.consoleLog.path, contents: nil)
        if let out = try? FileHandle(forWritingTo: config.consoleLog) {
            console.attachment = VZFileHandleSerialPortAttachment(fileHandleForReading: nil,
                                                              fileHandleForWriting: out)
        }
        c.serialPorts = [console]

        // NO network device. Not a policy, not a rule -- an absence.
        // PLAN.md 8.2 rests on this line not existing.

        do { try c.validate() }
        catch { throw SandboxError.configuration(error.localizedDescription) }
        return c
    }


    /// arm64 Linux images carry the bytes 41 52 4D 64 at offset 0x38. An EFI
    /// zboot kernel carries "MZ"/"zimg" and a compressed payload instead;
    /// Guest/mkimage.sh unwraps it at build time.
    private func checkKernelShape(_ url: URL) throws {
        guard let h = try? FileHandle(forReadingFrom: url) else { return }
        defer { try? h.close() }
        try? h.seek(toOffset: 0x38)
        let magic = (try? h.read(upToCount: 4)) ?? Data()
        guard magic != Data([0x41, 0x52, 0x4D, 0x64]) else { return }
        let hex = magic.map { String(format: "%02x", $0) }.joined()
        throw SandboxError.configuration(
            "\(url.lastPathComponent) is not a raw arm64 Image (0x38 = \(hex)). "
            + "An EFI zboot kernel has to be unwrapped first; Guest/mkimage.sh does that.")
    }

    // MARK: Lifecycle

    public func start() async throws {
        guard vm == nil else { return }
        let c = try makeConfiguration()
        state = .starting

        let machine = VZVirtualMachine(configuration: c)
        machine.delegate = self
        vm = machine

        do {
            try await machine.start()
            state = .running
        } catch {
            state = .failed(error.localizedDescription)
            vm = nil
            throw SandboxError.start(error.localizedDescription)
        }
    }

    /// Stops the guest, politely first and then not.
    ///
    /// The wait is bounded on purpose: `requestStop()` is a request, and a
    /// guest whose init does not handle it -- which is every guest until its
    /// init learns to -- simply keeps running. An unbounded wait here turns
    /// that into a hung host, which is exactly how it presented the first time.
    public func stop(gracePeriod: Duration = .seconds(5)) async {
        guard let machine = vm else { return }

        if machine.canRequestStop {
            try? machine.requestStop()
            let seconds = Double(gracePeriod.components.seconds)
            await withCheckedContinuation { (c: CheckedContinuation<Void, Never>) in
                let waiter = StopWaiter(c)
                stopWaiters.append(waiter)
                DispatchQueue.main.asyncAfter(deadline: .now() + seconds) { waiter.fire() }
            }
        }
        if let m = vm, m.canStop { try? await m.stop() }
        control = nil
        vm = nil
        state = .stopped
        resumeStopWaiters()
    }

    /// Opens the control channel. The guest must already be listening on
    /// `controlPort`; a connection attempt before the warden is up fails, which
    /// is why the caller retries rather than assuming boot order.
    public func connectControl() async throws -> Int32 {
        guard let machine = vm, state == .running else { throw SandboxError.notRunning }
        guard let socketDevice = machine.socketDevices.first as? VZVirtioSocketDevice else {
            throw SandboxError.configuration("no virtio socket device")
        }
        let connection = try await socketDevice.connect(toPort: Self.controlPort)
        control = connection
        return connection.fileDescriptor
    }

    /// Closes the control connection without stopping the guest.
    public func disconnectControl() {
        control?.close()
        control = nil
    }
}

// Isolated to the main actor, which is a statement of fact rather than a way
// to quiet the compiler: a VZVirtualMachine delivers its delegate callbacks on
// the queue it was created with, and that queue is the main one (PLAN.md 3.4,
// fifth edge). The SDK does not annotate the protocol, so Swift 6 needs telling.
extension SandboxHost: VZVirtualMachineDelegate {
    // `nonisolated` plus assumeIsolated rather than an isolated conformance:
    // the SDK does not annotate the protocol, and this spells out what is
    // actually true instead of asking the compiler to take it on trust. If the
    // assumption ever stops holding it traps here, loudly, rather than becoming
    // a race.
    public nonisolated func guestDidStop(_ virtualMachine: VZVirtualMachine) {
        MainActor.assumeIsolated {
            state = .stopped
            vm = nil
            resumeStopWaiters()
        }
    }

    public nonisolated func virtualMachine(_ virtualMachine: VZVirtualMachine,
                                           didStopWithError error: Error) {
        MainActor.assumeIsolated { handleStop(error: error) }
    }

    private func handleStop(error: Error) {
        // PLAN.md 6.5: a crashed guest is reported, not hidden. The session
        // parks, and the next tool call re-boots and replays the manifest.
        state = .failed(error.localizedDescription)
        vm = nil
        resumeStopWaiters()
    }

    fileprivate func resumeStopWaiters() {
        let waiting = stopWaiters
        stopWaiters = []
        for w in waiting { w.fire() }
    }
}
