// BottomFollower.swift -- a transcript that follows its tail only while you do.
//
// At the bottom, new content arrives and the view stays at the bottom.
// Scrolled up, new content arrives below and what is on screen does not move.
//
// SwiftUI on macOS 14 cannot tell those apart: its geometry arrives as one
// number, and "the content grew" and "the user scrolled" both move the tail
// relative to the viewport.  AppKit can.  The ScrollView is an NSScrollView,
// whose clip view reports *scrolling* (bounds changes) separately from the
// document reports *growth* (frame changes).  So whether to follow is decided
// only when the view scrolls, and acted on only when the content grows.
//
// Placed as the background of the scroll view's content, where its NSView
// finds the enclosing NSScrollView once it is in a window.

import AppKit
import SwiftUI

struct BottomFollower: NSViewRepresentable {
    /// Changes when the view should snap to the bottom and follow, whatever
    /// it was doing: a message sent, a different session selected.
    var repin: AnyHashable

    func makeNSView(context: Context) -> FollowerView { FollowerView() }

    func updateNSView(_ v: FollowerView, context: Context) {
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
    /// Main-thread only; read by deinit, which Swift cannot prove is.
    nonisolated(unsafe) private var observers: [NSObjectProtocol] = []

    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        attach()
    }

    deinit {
        observers.forEach(NotificationCenter.default.removeObserver)
    }

    private func attach() {
        guard window != nil, let sv = enclosingScrollView, sv !== scrollView else { return }
        observers.forEach(NotificationCenter.default.removeObserver)
        observers = []
        scrollView = sv
        let clip = sv.contentView
        clip.postsBoundsChangedNotifications = true
        observers.append(NotificationCenter.default.addObserver(
            forName: NSView.boundsDidChangeNotification, object: clip, queue: .main
        ) { [weak self] _ in
            MainActor.assumeIsolated { self?.scrolled() }
        })
        if let doc = sv.documentView {
            doc.postsFrameChangedNotifications = true
            observers.append(NotificationCenter.default.addObserver(
                forName: NSView.frameDidChangeNotification, object: doc, queue: .main
            ) { [weak self] _ in
                MainActor.assumeIsolated { self?.grew() }
            })
        }
        // A session opens at its end.
        pinned = true
        DispatchQueue.main.async { [weak self] in self?.toBottom() }
    }

    /// Snap to the end and follow from here on.
    func repin() {
        pinned = true
        DispatchQueue.main.async { [weak self] in self?.toBottom() }
    }

    // MARK: the decision, and the action

    /// The view scrolled.  If it was the user, whether to follow is now
    /// whether they left it at the end.
    private func scrolled() {
        guard !scrolling else { return }
        pinned = distanceFromBottom() <= slack
    }

    /// The content changed size.  Follow if pinned; otherwise leave the clip
    /// where it is, which on a top-anchored document keeps what is on screen
    /// exactly where it was.
    private func grew() {
        if pinned { toBottom() }
    }

    private func distanceFromBottom() -> CGFloat {
        guard let sv = scrollView, let doc = sv.documentView else { return 0 }
        let clip = sv.contentView.bounds
        if doc.isFlipped {
            return doc.frame.height - clip.maxY
        } else {
            return clip.minY - doc.frame.minY
        }
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
    }
}
