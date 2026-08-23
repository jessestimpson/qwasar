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

    /// Host code only. Never place the result anywhere a model reads:
    /// not a prompt, not a tool result, not a transcript, not a log.
    public static func key() -> String? {
        let q: [String: Any] = [kSecClass as String: kSecClassGenericPassword,
                                kSecAttrService as String: service,
                                kSecAttrAccount as String: account,
                                kSecReturnData as String: true]
        var out: CFTypeRef?
        guard SecItemCopyMatching(q as CFDictionary, &out) == errSecSuccess,
              let d = out as? Data, let s = String(data: d, encoding: .utf8),
              !s.isEmpty else { return nil }
        return s
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
        return SecItemAdd(q as CFDictionary, nil) == errSecSuccess
    }

    public static func remove() {
        let q: [String: Any] = [kSecClass as String: kSecClassGenericPassword,
                                kSecAttrService as String: service,
                                kSecAttrAccount as String: account]
        SecItemDelete(q as CFDictionary)
    }
}
