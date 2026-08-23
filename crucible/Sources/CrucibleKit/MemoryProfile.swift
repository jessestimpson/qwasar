// MemoryProfile.swift -- PLAN.md 2.3, made executable.
//
// The profile decides two numbers: how much context a session gets, and how
// many sessions may be live at once. Both are derived from the machine at
// launch, and both are shown to the user with the arithmetic that produced
// them. Nothing here is a constant except the model's own ceiling and the
// reserve.
//
// NOTE ON A PLAN CHANGE: PLAN.md 2.3 proposed adding a working-set accessor to
// qwasar.h, on the grounds that qw_gpu_working_set_limit() lives in a header the
// module map excludes. That ask is unnecessary -- the app links Metal already,
// and MTLDevice.recommendedMaxWorkingSetSize is the same number from the same
// device. No change to the parent tree is needed for this.

import Foundation
import Metal

public struct MemoryProfile: Sendable, Equatable {
    public var contextSize: Int32
    public var liveSessions: Int
    public var mtpEnabled: Bool

    /// Everything that went into the decision, so the UI can show its working.
    public var workingSetBytes: UInt64
    public var physicalBytes: UInt64
    public var reserveFraction: Double
    public var weightsBytes: UInt64
    public var perSessionBytes: UInt64
    public var note: String

    public static let modelMaxContext: Int32 = 262_144

    /// KV is 64 KB/token: 16 full-attention layers x 4 KV heads x 256 x 2 x fp16.
    public static let kvBytesPerToken: UInt64 = 64 * 1024
    /// The MTP draft head keeps its own single-layer KV cache.
    public static let mtpBytesPerToken: UInt64 = 4 * 1024
    /// The head's own weights: a single BF16 layer, measured at 810 MB on disk.
    public static let mtpWeightsBytes: UInt64 = 850_000_000
    /// Below this, the context the head costs is worth more than the speed it
    /// buys, and it stays off however much memory is free.
    public static let mtpContextFloor: Int32 = 32_768
    /// 151 MB of SSM/conv state plus ~200 MB of activation scratch, both
    /// independent of context length.
    public static let sessionFixedBytes: UInt64 = 351 * 1024 * 1024
    /// 15.1 GB of 4-bit text weights plus a 0.92 GB BF16 vision tower.
    public static let weightsBytesDefault: UInt64 = 16_020_000_000

    /// Exceeding recommendedMaxWorkingSetSize does not fail an allocation, it
    /// starts evicting GPU resources to swap -- which on a 16 GB weight set is
    /// indistinguishable from the app hanging. 15% is much cheaper than finding
    /// that cliff.
    public static let reserve = 0.85

    /// Live count is capped even where memory allows more: one session runs at a
    /// time whatever the profile says (PLAN.md 2.1), so extra live sessions buy
    /// switch latency and nothing else.
    public static let maxLiveSessions = 4

    /// `mtpAvailable` says whether a draft head has been granted; without one
    /// there is nothing to enable, so the memory question does not arise.
    public static func derive(weightsBytes: UInt64 = weightsBytesDefault,
                              mtpAvailable: Bool = false) -> MemoryProfile {
        let physical = ProcessInfo.processInfo.physicalMemory
        let device = MTLCreateSystemDefaultDevice()
        let workingSet = UInt64(device?.recommendedMaxWorkingSetSize ?? (physical * 84 / 100))

        let usable = Double(workingSet) * reserve
        let forSessions = usable - Double(weightsBytes)

        guard forSessions > Double(sessionFixedBytes) else {
            return MemoryProfile(contextSize: 4096, liveSessions: 1, mtpEnabled: false,
                                 workingSetBytes: workingSet, physicalBytes: physical,
                                 reserveFraction: reserve, weightsBytes: weightsBytes,
                                 perSessionBytes: sessionFixedBytes,
                                 note: "This machine cannot hold the weights and a "
                                     + "usable session at once. Expect swapping.")
        }

        // The draft head is a real trade, not a free win: 811 MB of weights plus
        // 4 KB/token of its own cache, paid out of the same budget the context
        // comes from. On a 32 GB machine that is 1.14 GB of a 6.30 GB session
        // budget -- about 12% of the context -- bought with faster decode.
        //
        // So it is enabled when a head is actually present and the context that
        // survives is still worth having. The previous rule (`forSessions >
        // 24 GB`) was a guess made while nothing loaded the head at all, and it
        // said no on every machine that could comfortably have said yes.
        let mtp = mtpAvailable
            && forSessions > Double(mtpWeightsBytes + sessionFixedBytes)
            && contextFitting(budget: forSessions - Double(mtpWeightsBytes),
                              sessions: 1,
                              perToken: kvBytesPerToken + mtpBytesPerToken) >= mtpContextFloor
        let perToken = kvBytesPerToken + (mtp ? mtpBytesPerToken : 0)
        // The head's weights come out of the same budget the context does.
        let budget = mtp ? forSessions - Double(mtpWeightsBytes) : forSessions

        // Prefer context first, then extra live sessions, but never more than
        // the cap and never more context than the model has positions for.
        var live = 1
        var context = contextFitting(budget: budget, sessions: 1, perToken: perToken)

        while live < maxLiveSessions {
            let next = live + 1
            let c = contextFitting(budget: budget, sessions: next, perToken: perToken)
            // Only take another live session if the context it leaves is still
            // the model's full window -- context is the resource that extends
            // what a session can do; live count only removes a park and a
            // restore.
            if c >= modelMaxContext { live = next; context = c } else { break }
        }

        let per = sessionFixedBytes + UInt64(context) * perToken
        return MemoryProfile(contextSize: context, liveSessions: live, mtpEnabled: mtp,
                             workingSetBytes: workingSet, physicalBytes: physical,
                             reserveFraction: reserve, weightsBytes: weightsBytes,
                             perSessionBytes: per,
                             note: "")
    }

    private static func contextFitting(budget: Double, sessions: Int, perToken: UInt64) -> Int32 {
        let each = budget / Double(sessions) - Double(sessionFixedBytes)
        guard each > 0 else { return 0 }
        let raw = each / Double(perToken)
        // Round down to a multiple of 8192: a tidy number to show a user, and
        // well below any allocator granularity that would matter.
        let stepped = (Int32(min(raw, Double(Int32.max))) / 8192) * 8192
        return min(stepped, modelMaxContext)
    }

    /// Decimal GB throughout, because that is what Metal reports and what
    /// PLAN.md's tables are written in: recommendedMaxWorkingSetSize is
    /// 26,800,603,136 bytes, which is 26.80 GB and 24.96 GiB. Printing one and
    /// labelling it the other is how a budget quietly gains 7%.
    public var summary: String {
        let gb = { (b: UInt64) in String(format: "%.2f GB", Double(b) / 1e9) }
        return """
        physical \(gb(physicalBytes)) · Metal working set \(gb(workingSetBytes)) \
        · reserve \(Int(reserveFraction * 100))%
        weights \(gb(weightsBytes)) · per session \(gb(perSessionBytes))
        → \(liveSessions) live session\(liveSessions == 1 ? "" : "s") \
        at \(contextSize) tokens, MTP \(mtpEnabled ? "on" : "off")
        """
    }
}
