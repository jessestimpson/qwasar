// NetworkPolicySuite.swift -- the allowlist, tested adversarially.
//
// PLAN.md 8.3: the policy is the security boundary for egress, and like
// PathGuardSuite it asserts on the REASON a URL was refused, because "refused"
// alone would pass a policy that refuses for the wrong reason and admits for
// the wrong reason too. Every case here is a real evasion shape: scheme games,
// lookalike hosts, userinfo tricks, ports, IP literals.

import Foundation
import CrucibleKit

enum NetworkPolicySuite {
    static func run() -> Int {
        var fails = 0
        func check(_ label: String, _ ok: Bool, _ detail: String = "") {
            fails += TestMain.check(ok, detail.isEmpty ? label : "\(label) \(detail)")
        }
        let p = NetworkPolicy(allowlist: ["hexdocs.pm", "*.github.io", "1.2.3.4"])

        func refusal(_ s: String) -> String? {
            guard let u = URL(string: s) else { return "unparseable" }
            return p.refusal(for: u)
        }

        // What the list admits.
        check("exact host", refusal("https://hexdocs.pm/elixir/") == nil)
        check("exact host, case-folded", refusal("https://HexDocs.PM/x") == nil)
        check("wildcard subdomain", refusal("https://a.github.io/repo") == nil)
        check("deep subdomain", refusal("https://a.b.github.io/") == nil)
        check("explicit port 443", refusal("https://hexdocs.pm:443/x") == nil)
        check("listed IP literal", refusal("https://1.2.3.4/x") == nil)

        // What it must refuse, for the right reason.
        func refused(_ label: String, _ url: String, containing want: String) {
            let r = refusal(url)
            check(label, r?.contains(want) == true,
                  r?.contains(want) == true ? "" : "got \(r ?? "nil (ADMITTED)")")
        }
        refused("http downgrade", "http://hexdocs.pm/x", containing: "https")
        refused("unlisted host", "https://example.com/", containing: "allowlist")
        refused("lookalike suffix", "https://evilgithub.io/", containing: "allowlist")
        refused("lookalike prefix", "https://hexdocs.pm.evil.com/", containing: "allowlist")
        refused("bare wildcard base", "https://github.io/", containing: "allowlist")
        refused("userinfo trick", "https://hexdocs.pm@evil.com/", containing: "userinfo")
        refused("explicit userinfo", "https://user:pw@hexdocs.pm/", containing: "userinfo")
        refused("odd port", "https://hexdocs.pm:8443/x", containing: "port")
        refused("unlisted IP literal", "https://4.3.2.1/", containing: "IP-literal")
        refused("no host", "https:///path", containing: "host")
        check("file scheme", refusal("file:///etc/passwd") != nil)
        check("empty allowlist is off",
              NetworkPolicy(allowlist: []).isEnabled == false)

        // The wrapper advertises fetch only through NetworkToolRunner, so a
        // policy-off project never carries the schema. The composition rule,
        // pinned here so the wiring cannot silently regress.
        check("fetch schema names fetch", ToolSurface.fetchSchema.contains("\"fetch\""))
        check("fetch is not in the frozen guest surface",
              !ToolSurface.guestSchemas.contains(ToolSurface.fetchSchema))

        return fails
    }
}
