// GateCheck.swift -- M0's gate, run at launch and reported in the UI.
//
// PLAN.md 12, M0: "App Sandbox + com.apple.security.virtualization + a 16 GB
// mmap coexist in a signed bundle. If this fails, the entitlement plan changes,
// and it must fail here rather than at M5."
//
// So it is a real check with a real verdict, not a comment. Four probes, in the
// order that makes a failure diagnosable:
//
//   1. Is the bundle signed, and with which entitlements? Read them off the
//      running task rather than trusting the plist we shipped.
//   2. Does Virtualization.framework let us build and validate a configuration?
//   3. Does it let us INSTANTIATE a VZVirtualMachine? This is the step that
//      actually consults the entitlement, and it is attempted only when probe 1
//      says the entitlement is present -- without it the framework terminates
//      the process rather than returning an error, which would take the gate
//      report down with it.
//   4. Does the engine load, i.e. does a ~16 GB mmap survive the sandbox?
//      Answered by EngineHost, not here, because it takes seconds.

import Foundation
import Virtualization

public struct GateReport: Sendable {
    public var sandboxed: Bool?
    public var virtualization: Bool?
    public var configValidates: Bool = false
    public var vmInstantiates: Bool = false
    public var detail: [String] = []

    public init() {}

    public var virtualizationVerdict: String {
        if vmInstantiates { return "VZVirtualMachine instantiated" }
        if virtualization != true { return "entitlement absent" }
        return "entitlement present, instantiation not proven"
    }
}

public enum GateCheck {

    public static func run() -> GateReport {
        var r = GateReport()
        r.sandboxed = Diagnostics.entitlement("com.apple.security.app-sandbox")
        r.virtualization = Diagnostics.entitlement("com.apple.security.virtualization")

        r.detail.append("home: \(Diagnostics.homeDirectory)")
        r.detail.append("sandbox: \(describe(r.sandboxed))")
        r.detail.append("virtualization: \(describe(r.virtualization))")

        // A minimal but genuinely valid configuration: EFI bootloader with a
        // fresh variable store, the framework's own minimum memory, one CPU.
        // No network device -- PLAN.md 6.1 omits it by construction, and the
        // gate should probe the shape we actually ship.
        let config = VZVirtualMachineConfiguration()
        config.cpuCount = 1
        config.memorySize = VZVirtualMachineConfiguration.minimumAllowedMemorySize

        let storeURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("crucible-gate-efi-\(UUID().uuidString)")
        do {
            let loader = VZEFIBootLoader()
            loader.variableStore = try VZEFIVariableStore(creatingVariableStoreAt: storeURL)
            config.bootLoader = loader
            try config.validate()
            r.configValidates = true
            r.detail.append("config: validates (\(config.cpuCount) cpu, "
                            + "\(config.memorySize / 1_048_576) MB, no network device)")
        } catch {
            r.detail.append("config: \(error.localizedDescription)")
        }
        defer { try? FileManager.default.removeItem(at: storeURL) }

        // Only now, and only if the entitlement is really there.
        if r.configValidates, r.virtualization == true {
            let vm = VZVirtualMachine(configuration: config)
            r.vmInstantiates = true
            r.detail.append("VZVirtualMachine: instantiated, state \(vm.state.rawValue)")
        } else if r.configValidates {
            r.detail.append("VZVirtualMachine: not attempted -- entitlement absent, and "
                            + "the framework terminates the process rather than failing")
        }

        return r
    }

    private static func describe(_ b: Bool?) -> String {
        switch b {
        case .some(true):  return "granted"
        case .some(false): return "present but false"
        case .none:        return "absent (unsigned, or not in the entitlements)"
        }
    }
}
