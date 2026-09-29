// QwasarApp.swift -- one app, two faces (PLAN-qwasar.md §4.1).
//
// A menu bar item that runs qwasar-server (StatusItem.swift), and the coding
// agent's window, opened on request: Open Coding Agent, ⌘N, or `--agent`.
// While a window is open the app is an ordinary one, with a Dock icon and a
// main menu; when the last window closes it goes back to being an item in
// the menu bar and the server keeps running.
//
// The window is an NSWindow hosting the SwiftUI RootView -- AppKit owns the
// window and the menus because it is AppKit that decides when there is a
// window at all.

import AppKit
import QwasarKit
import SwiftUI

@MainActor
final class QwasarAppDelegate: NSObject, NSApplicationDelegate, NSWindowDelegate {
    private let openAgentAtLaunch: Bool
    private var state: AppState!
    private var statusItem: StatusItemController!
    private var window: NSWindow?

    init(openAgentAtLaunch: Bool) {
        self.openAgentAtLaunch = openAgentAtLaunch
    }

    func applicationDidFinishLaunching(_ note: Notification) {
        state = AppState()
        statusItem = StatusItemController(state: state) { [weak self] in self?.openAgent() }
        installMainMenu()
        if state.modelPath == nil {
            // Nothing to run yet: the first thing to do is pick a model.
            NSApp.activate()
            state.chooseModel()
        }
        state.startServer()
        if openAgentAtLaunch { openAgent() }
    }

    /// Opens the window, or brings it forward.
    func openAgent() {
        if window == nil {
            let w = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1180, height: 820),
                             styleMask: [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView],
                             backing: .buffered, defer: false)
            w.title = "Qwasar"
            w.setFrameAutosaveName("QwasarMain")
            w.minSize = NSSize(width: 900, height: 620)
            w.isReleasedWhenClosed = false
            w.contentViewController = NSHostingController(rootView: RootView(state: state))
            w.delegate = self
            w.center()
            window = w
        }
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
        window?.makeKeyAndOrderFront(nil)
    }

    func windowWillClose(_ notification: Notification) {
        // Back to the menu bar.  The window is kept (its state is the app's),
        // the Dock icon goes.
        DispatchQueue.main.async { NSApp.setActivationPolicy(.accessory) }
    }

    /// Quitting from the window quits the app, server and all.  Held until
    /// the guests have flushed and the server has checkpointed: `terminateLater`
    /// is the only hook macOS gives for asynchronous cleanup on quit.
    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        guard let state else { return .terminateNow }
        Task {
            await state.shutdown()
            NSApp.reply(toApplicationShouldTerminate: true)
        }
        return .terminateLater
    }

    // MARK: the main menu

    /// The menus a window needs: the app's, File, Edit (so text fields cut,
    /// copy and paste), Window.  What the status item offers is not repeated.
    private func installMainMenu() {
        let main = NSMenu()

        let appMenu = NSMenu()
        appMenu.addItem(withTitle: "About Qwasar", action: #selector(NSApplication.orderFrontStandardAboutPanel(_:)), keyEquivalent: "")
        appMenu.addItem(.separator())
        appMenu.addItem(withTitle: "Set Delegation API Key…", action: #selector(setAPIKey), keyEquivalent: "").target = self
        appMenu.addItem(withTitle: "Remove Delegation API Key", action: #selector(removeAPIKey), keyEquivalent: "").target = self
        appMenu.addItem(.separator())
        appMenu.addItem(withTitle: "Hide Qwasar", action: #selector(NSApplication.hide(_:)), keyEquivalent: "h")
        appMenu.addItem(withTitle: "Quit Qwasar", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        let appItem = NSMenuItem(); appItem.submenu = appMenu; main.addItem(appItem)

        let file = NSMenu(title: "File")
        file.addItem(withTitle: "Open Coding Agent", action: #selector(openAgentAction), keyEquivalent: "n").target = self
        let np = file.addItem(withTitle: "New Project…", action: #selector(newProject), keyEquivalent: "N")
        np.keyEquivalentModifierMask = [.command, .shift]
        np.target = self
        file.addItem(.separator())
        file.addItem(withTitle: "Close", action: #selector(NSWindow.performClose(_:)), keyEquivalent: "w")
        let fileItem = NSMenuItem(); fileItem.submenu = file; main.addItem(fileItem)

        let edit = NSMenu(title: "Edit")
        edit.addItem(withTitle: "Undo", action: Selector(("undo:")), keyEquivalent: "z")
        edit.addItem(withTitle: "Redo", action: Selector(("redo:")), keyEquivalent: "Z")
        edit.addItem(.separator())
        edit.addItem(withTitle: "Cut", action: #selector(NSText.cut(_:)), keyEquivalent: "x")
        edit.addItem(withTitle: "Copy", action: #selector(NSText.copy(_:)), keyEquivalent: "c")
        edit.addItem(withTitle: "Paste", action: #selector(NSText.paste(_:)), keyEquivalent: "v")
        edit.addItem(withTitle: "Select All", action: #selector(NSText.selectAll(_:)), keyEquivalent: "a")
        let editItem = NSMenuItem(); editItem.submenu = edit; main.addItem(editItem)

        let win = NSMenu(title: "Window")
        win.addItem(withTitle: "Minimize", action: #selector(NSWindow.performMiniaturize(_:)), keyEquivalent: "m")
        win.addItem(withTitle: "Zoom", action: #selector(NSWindow.performZoom(_:)), keyEquivalent: "")
        let winItem = NSMenuItem(); winItem.submenu = win; main.addItem(winItem)
        NSApp.windowsMenu = win

        NSApp.mainMenu = main
    }

    @objc private func openAgentAction() { openAgent() }
    @objc private func newProject() { openAgent(); state.addProject() }
    @objc private func setAPIKey() { openAgent(); state.showingAPIKeySheet = true }
    @objc private func removeAPIKey() { state.removeAPIKey() }
}
