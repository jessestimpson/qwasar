// KeychainAccess.swift -- the API key, held where no model can reach it.
//
// Spec §15.4: the key is the one secret in the system. It lives in the macOS
// Keychain, is entered once through a host UI panel, and is read only by the
// host code that attaches it to a request. Nothing here is reachable from a
// tool call: the config tools can ask `hasKey`, never `key`.

import Foundation
import Security

public enum KeychainAccess {
    static let service = "dev.crucible.escalation"
    static let account = "api-key"

    public static var hasKey: Bool { key() != nil }

    /// The last status each operation returned, for surfacing failures --
    /// ad-hoc signing plus App Sandbox is exactly where keychain calls fail
    /// quietly, and a swallowed OSStatus here presents three layers away as
    /// "the key is absent" right after the user typed it.
    nonisolated(unsafe) public private(set) static var lastStatus: OSStatus = errSecSuccess

    /// Host code only. Never place the result anywhere a model reads:
    /// not a prompt, not a tool result, not a transcript, not a log.
    public static func key() -> String? {
        let q: [String: Any] = [kSecClass as String: kSecClassGenericPassword,
                                kSecAttrService as String: service,
                                kSecAttrAccount as String: account,
                                kSecReturnData as String: true]
        var out: CFTypeRef?
        lastStatus = SecItemCopyMatching(q as CFDictionary, &out)
        guard lastStatus == errSecSuccess,
              let d = out as? Data, let s = String(data: d, encoding: .utf8),
              !s.isEmpty else { return nil }
        return s
    }

    /// A one-line report for the UI and config_show: set, or absent with the
    /// OSStatus that made it so.
    public static func status() -> String {
        if hasKey { return "set (in the Keychain)" }
        if lastStatus == errSecItemNotFound { return "absent" }
        let msg = SecCopyErrorMessageString(lastStatus, nil).map { $0 as String }
            ?? "OSStatus \(lastStatus)"
        return "absent — the keychain read failed: \(msg) (\(lastStatus))"
    }

    @discardableResult
    public static func set(_ key: String) -> Bool {
        remove()
        let trimmed = key.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return false }
        let q: [String: Any] = [kSecClass as String: kSecClassGenericPassword,
                                kSecAttrService as String: service,
                                kSecAttrAccount as String: account,
                                kSecValueData as String: Data(trimmed.utf8)]
        lastStatus = SecItemAdd(q as CFDictionary, nil)
        return lastStatus == errSecSuccess
    }

    public static func remove() {
        let q: [String: Any] = [kSecClass as String: kSecClassGenericPassword,
                                kSecAttrService as String: service,
                                kSecAttrAccount as String: account]
        SecItemDelete(q as CFDictionary)
    }
}
