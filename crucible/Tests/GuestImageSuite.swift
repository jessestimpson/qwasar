// GuestImageSuite.swift -- the claim in PLAN.md 6.3, checked.
//
// "A 2 GB sparse image clones in microseconds and consumes only what the
// session actually writes." That is a factual claim about APFS, the whole
// argument for a VM per session, and it is cheap to verify -- so it is.

import Foundation
import CrucibleKit

enum GuestImageSuite {
    static func run() -> Int {
        var f = 0
        let fm = FileManager.default
        let tmp = fm.temporaryDirectory.appendingPathComponent("crucible-img-\(UUID().uuidString)")
        try? fm.createDirectory(at: tmp, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: tmp) }

        // A 256 MB sparse file, standing in for the golden image.
        let golden = tmp.appendingPathComponent("golden.img")
        guard fm.createFile(atPath: golden.path, contents: nil),
              let h = try? FileHandle(forWritingTo: golden) else {
            print("  FAIL cannot create a test image"); return 1
        }
        try? h.truncate(atOffset: 256 * 1024 * 1024)
        // truncate(atOffset:) also moves the file offset, so the marker has to
        // be seeked back into place -- without this the bytes land at 256 MB
        // and the golden's head is zeros.
        try? h.seek(toOffset: 0)
        try? h.write(contentsOf: Data(repeating: 0xAB, count: 1024 * 1024))
        try? h.close()

        let image = GuestImage(golden: golden)
        f += TestMain.check(image.exists, "the golden image is found")

        let dest = tmp.appendingPathComponent("session/disk.img")
        guard let r = try? image.clone(to: dest) else {
            print("  FAIL clone threw"); return f + 1
        }
        f += TestMain.check(r.method == .copyOnWrite,
                            "clones copy-on-write on APFS (got \(r.method.rawValue))")
        f += TestMain.check(r.seconds < 0.25,
                            String(format: "the clone is effectively instant (%.4fs)", r.seconds))

        // The apparent size matches; the allocated size does not.
        let apparent = UInt64((try? dest.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0)
        f += TestMain.check(apparent == image.bytes, "the clone has the same apparent size")

        let allocated = GuestImage.allocatedBytes(dest)
        f += TestMain.check(allocated < 8 * 1024 * 1024,
                            "the clone allocates almost nothing until written "
                            + "(\(allocated / 1024) KB)")

        // Writing to the clone must not disturb the golden image -- that is the
        // isolation the whole scheme depends on.
        if let hc = try? FileHandle(forWritingTo: dest) {
            try? hc.seek(toOffset: 4096)
            try? hc.write(contentsOf: Data(repeating: 0x5A, count: 4096))
            try? hc.close()
        }
        let goldenHead = (try? FileHandle(forReadingFrom: golden))
            .flatMap { h -> Data? in defer { try? h.close() }
                       try? h.seek(toOffset: 4096); return try? h.read(upToCount: 8) }
        f += TestMain.check(goldenHead == Data(repeating: 0xAB, count: 8),
                            "writing the clone leaves the golden image untouched")

        // --- a session's disk is kept, not re-cloned -----------------------
        //
        // The failure this pins is the worst kind the project can have: the
        // model works for an hour, the app restarts, and the work is gone with
        // nothing on screen to say anything happened. `start(session:)` used to
        // clone unconditionally, and `clone(to:)` deletes its destination
        // first, so a restart replaced the session's disk with a pristine one.
        let work = fm.temporaryDirectory
            .appendingPathComponent("crucible-provision-\(UUID().uuidString)")
        try? fm.createDirectory(at: work, withIntermediateDirectories: true)
        let seedURL = work.appendingPathComponent("golden.img")
        let sessionDisk = work.appendingPathComponent("session/disk.img")
        try? Data("PRISTINE".utf8).write(to: seedURL)
        let seeded = GuestImage(golden: seedURL)

        if let first = try? seeded.provision(at: sessionDisk) {
            f += TestMain.check(first.method != .reused,
                                "first boot clones the golden image (\(first.method.rawValue))")
            // Stand in for an hour of the model's work.
            try? Data("THE MODEL'S WORK".utf8).write(to: sessionDisk)

            if let second = try? seeded.provision(at: sessionDisk) {
                f += TestMain.check(second.method == .reused,
                                    "second boot reuses the session's disk")
            } else {
                f += TestMain.check(false, "provision succeeds on a second boot")
            }
            let after = (try? Data(contentsOf: sessionDisk))
                .map { String(decoding: $0, as: UTF8.self) }
            f += TestMain.check(after == "THE MODEL'S WORK",
                                "the session's work survives a restart "
                                + "(found \(after ?? "nothing"))")

            // The destructive path stays available to whoever asks for it by
            // name -- discard(session:) does, on an explicit delete.
            _ = try? seeded.clone(to: sessionDisk)
            let wiped = (try? Data(contentsOf: sessionDisk))
                .map { String(decoding: $0, as: UTF8.self) }
            f += TestMain.check(wiped == "PRISTINE",
                                "clone(to:) still replaces the disk when asked directly")
        } else {
            f += TestMain.check(false, "provision clones on first use")
        }
        try? fm.removeItem(at: work)

        GuestImage.discard(dest)
        f += TestMain.check(!fm.fileExists(atPath: dest.path), "discard removes the clone")
        return f
    }
}
