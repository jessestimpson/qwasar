// Diagnostics.swift -- what the process actually costs, and what it is allowed
// to do. Both are M0 gate evidence rather than niceties.

import Foundation
import Security

public enum Diagnostics {

    /// Real memory footprint as the kernel accounts it -- the number that
    /// matters when 16 GB of weights are mmapped and the question is whether
    /// they are resident or being paged.
    public static func physFootprint() -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return kr == KERN_SUCCESS ? UInt64(info.phys_footprint) : 0
    }

    /// Reads an entitlement off the running process.
    ///
    /// This is the only way to state, rather than assume, that a signed bundle
    /// actually carries the entitlements its plist asked for. An ad-hoc
    /// signature grants them for local use; a missing one shows up here as nil
    /// rather than as a mystery at first use.
    public static func entitlement(_ key: String) -> Bool? {
        guard let task = SecTaskCreateFromSelf(nil) else { return nil }
        guard let value = SecTaskCopyValueForEntitlement(task, key as CFString, nil) else { return nil }
        return (value as? Bool) ?? false
    }

    public static var isSandboxed: Bool {
        entitlement("com.apple.security.app-sandbox") == true
    }

    public static var hasVirtualization: Bool {
        entitlement("com.apple.security.virtualization") == true
    }

    /// The app's own container, which under App Sandbox is where writes land
    /// whatever path the code asks for.
    public static var homeDirectory: String {
        NSHomeDirectory()
    }
}
