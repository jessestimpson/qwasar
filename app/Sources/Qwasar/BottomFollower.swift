// BottomFollower.swift -- a transcript that follows its tail only while you do.
//
// At the bottom, new content arrives and the view stays at the bottom.
// Scrolled up, new content arrives below and what is on screen does not move.
// A session starts at the bottom: until you scroll up, it follows.
//
// SwiftUI on macOS 14 cannot tell those apart: its geometry arrives as one
// number, and "the content grew" and "the user scrolled" both move the tail
// relative to the viewport.  AppKit can.  The ScrollView is an NSScrollView,
// whose clip view reports scrolling (bounds changes) separately from growth.
// So whether to follow is decided only when the view scrolls with nothing else
// changing, and acted on whenever the content or the viewport changes size.
//
// Three things make that hold in the real window rather than only in a test:
//
//   - Growth is seen from this view's own frame.  It is the background of the
//     scroll view's content, so it is exactly the content's size, whatever
//     SwiftUI does with the document view behind it.
//   - A scroll that coincides with a change of content height or viewport size
//     is the system's -- a LazyVStack correcting an estimated row height, a
//     footer appearing, the window resizing -- and never unpins.  Only a scroll
//     with nothing else changing is the user's.
//   - The scroll view is looked for again whenever this view moves or lays
//     out, not only when it first enters a window: SwiftUI can put the view in
//     a window before it is inside the scroll view.
//
// QWASAR_DEBUG_FOLLOW=1 in the environment logs attachment and every change of
// mind, for the case this still gets wrong.

import AppKit
import SwiftUI

struct BottomFollower: NSViewRepresentable {
    /// Changes when the view should snap to the bottom and follow, whatever
    /// it was doing: a message sent, a different session selected.
    var repin: AnyHashable

    func makeNSView(context: Context) -> FollowerView { FollowerView() }

    func updateNSView(_ v: FollowerView, context: Context) {
        v.attach()
        if v.repinToken != repin {
            v.repinToken = repin
            v.repin()
        }
    }
}

final class FollowerView: NSView {
    var repinToken: AnyHashable?

    /// Within this many points of the end counts as at the end: a trackpad
    /// flick rarely lands exactly, and "almost at the bottom" means "follow".
    private let slack: CGFloat = 32
    private weak var scrollView: NSScrollView?
    private var pinned = true
    /// Set while this view scrolls, so its own scroll is not read as the user's.
    private var scrolling = false
    /// What the last scroll event saw, to tell a user's scroll (only the origin
    /// moved) from the system's (the content or the viewport changed size).
    private var lastContentHeight: CGFloat = -1
    private var lastViewport: CGSize = .zero
    /// Main-thread only; read by deinit, which Swift cannot prove is.
    nonisolated(unsafe) private var observers: [NSObjectProtocol] = []
    private let debug = ProcessInfo.processInfo.environment["QWASAR_DEBUG_FOLLOW"] != nil

    private func log(_ s: @autoclosure () -> String) {
        if debug { NSLog("BottomFollower: %@", s()) }
    }

    override func viewDidMoveToWindow() { super.viewDidMoveToWindow(); attach() }
    override func viewDidMoveToSuperview() { super.viewDidMoveToSuperview(); attach() }
    override func layout() { super.layout(); attach() }

    /// This view is the content's background, so its size is the content's.
    override func setFrameSize(_ newSize: NSSize) {
        let grewOrShrank = abs(newSize.height - frame.height) > 0.5
        super.setFrameSize(newSize)
        if grewOrShrank { contentChanged() }
    }

    deinit {
        observers.forEach(NotificationCenter.default.removeObserver)
    }

    func attach() {
        guard window != nil, let sv = enclosingScrollView, sv !== scrollView else { return }
        observers.forEach(NotificationCenter.default.removeObserver)
        observers = []
        scrollView = sv
        let clip = sv.contentView
        clip.postsBoundsChangedNotifications = true
        clip.postsFrameChangedNotifications = true
        observers.append(NotificationCenter.default.addObserver(
            forName: NSView.boundsDidChangeNotification, object: clip, queue: .main
        ) { [weak self] _ in
            MainActor.assumeIsolated { self?.scrolled() }
        })
        // The viewport itself resizing: a footer appearing, the window.
        observers.append(NotificationCenter.default.addObserver(
            forName: NSView.frameDidChangeNotification, object: clip, queue: .main
        ) { [weak self] _ in
            MainActor.assumeIsolated { self?.contentChanged() }
        })
        pinned = true
        snapshot()
        log("attached to \(type(of: sv)); pinned")
        DispatchQueue.main.async { [weak self] in self?.toBottom() }
    }

    /// Snap to the end and follow from here on.
    func repin() {
        pinned = true
        log("re-pinned")
        DispatchQueue.main.async { [weak self] in self?.toBottom() }
    }

    // MARK: the decision, and the action

    private var contentHeight: CGFloat {
        scrollView?.documentView?.frame.height ?? frame.height
    }

    private func snapshot() {
        lastContentHeight = contentHeight
        lastViewport = scrollView?.contentView.bounds.size ?? .zero
    }

    /// The view scrolled.  If only the origin moved, it was the user, and
    /// whether to follow is now whether they left it at the end.  If the
    /// content or the viewport changed size too, it was the system, and a
    /// pinned view goes back to the end.
    private func scrolled() {
        guard !scrolling, let sv = scrollView else { return }
        let sameContent = abs(contentHeight - lastContentHeight) < 0.5
        let sameViewport = sv.contentView.bounds.size == lastViewport
        snapshot()
        if sameContent && sameViewport {
            let was = pinned
            pinned = distanceFromBottom() <= slack
            if was != pinned { log(pinned ? "user scrolled to the end: following" : "user scrolled up: holding") }
        } else if pinned {
            follow()
        }
    }

    /// The content or the viewport changed size.  Follow if pinned -- once,
    /// after this layout pass, however many changes it made; otherwise leave
    /// the clip where it is, which on a top-anchored document keeps what is on
    /// screen exactly where it was.
    private func contentChanged() {
        snapshot()
        guard pinned, !followScheduled else { return }
        followScheduled = true
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.followScheduled = false
            if self.pinned { self.follow() }
        }
    }

    // A follow that makes the content change size makes another follow, and
    // if the two never agree -- a layout that settles differently for every
    // scroll position -- that is a loop with no input to end it.  More than
    // this many follows in a second and following pauses briefly, then snaps
    // once: the view ends at the bottom and the main thread is let go.
    private var followScheduled = false
    private var recentFollows: [Date] = []
    private var pausedUntil = Date.distantPast
    private let maxFollowsPerSecond = 30

    private func follow() {
        let now = Date()
        if now < pausedUntil { return }
        recentFollows = recentFollows.filter { now.timeIntervalSince($0) < 1 }
        recentFollows.append(now)
        if recentFollows.count > maxFollowsPerSecond {
            log("following \(recentFollows.count) times a second; pausing")
            recentFollows = []
            pausedUntil = now.addingTimeInterval(0.5)
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.55) { [weak self] in
                guard let self, self.pinned else { return }
                self.toBottom()
            }
            return
        }
        toBottom()
    }

    private func distanceFromBottom() -> CGFloat {
        guard let sv = scrollView, let doc = sv.documentView else { return 0 }
        let clip = sv.contentView.bounds
        return doc.isFlipped ? doc.frame.height - clip.maxY : clip.minY - doc.frame.minY
    }

    private func toBottom() {
        guard let sv = scrollView, let doc = sv.documentView else { return }
        let clip = sv.contentView
        let y = doc.isFlipped ? max(0, doc.frame.height - clip.bounds.height) : doc.frame.minY
        guard abs(clip.bounds.origin.y - y) > 0.5 else { return }
        scrolling = true
        clip.scroll(to: NSPoint(x: clip.bounds.origin.x, y: y))
        sv.reflectScrolledClipView(clip)
        scrolling = false
        snapshot()
    }
}
