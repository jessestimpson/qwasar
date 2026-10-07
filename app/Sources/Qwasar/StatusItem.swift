// StatusItem.swift -- the menu bar item: the server's state at a glance, and
// its menu.  Came from the menu bar app; the one addition is Open Coding
// Agent, which is the window (QwasarApp.swift).
//
// Everything shown is read from the state at the moment it changes or the
// menu opens.  The indicator is derived, never assumed: `listening` means a
// connection to the port just succeeded.

import AppKit
import QwasarKit
import ServiceManagement

@MainActor
final class StatusItemController: NSObject, NSMenuDelegate {
    private let state: AppState
    private let openAgent: () -> Void
    private var item: NSStatusItem!
    private let glyph = StatusIcon.glyph()
    private var dot: DotView!
    private var pulse: Timer?

    private let statusLine = NSMenuItem(title: "", action: nil, keyEquivalent: "")
    private let agentItem = NSMenuItem(title: "Open Coding Agent", action: #selector(openWindow), keyEquivalent: "n")
    private let copyItem = NSMenuItem(title: "Copy API URL", action: #selector(copyURL), keyEquivalent: "c")
    private let toggleItem = NSMenuItem(title: "", action: #selector(toggle), keyEquivalent: "s")
    private let portItem = NSMenuItem(title: "", action: #selector(choosePort), keyEquivalent: "")
    private let modelItem = NSMenuItem(title: "", action: nil, keyEquivalent: "")
    private let modelMenu = NSMenu()
    private let thinkingItem = NSMenuItem(title: "Thinking for API Clients", action: #selector(toggleThinking),
                                          keyEquivalent: "")
    private let loginItem = NSMenuItem(title: "Start at Login", action: #selector(toggleLogin), keyEquivalent: "")
    /// Loadable model folders found on disk, rescanned each time the menu opens.
    private var models: [FoundModel] = []

    init(state: AppState, openAgent: @escaping () -> Void) {
        self.state = state
        self.openAgent = openAgent
        super.init()
        item = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        if let button = item.button {
            button.image = glyph.plain
            button.imagePosition = .imageOnly
            let d = glyph.dotRadius * 2
            dot = DotView(frame: NSRect(x: 0, y: 0, width: d, height: d))
            button.addSubview(dot)
        }
        let menu = NSMenu()
        menu.delegate = self
        menu.autoenablesItems = false
        statusLine.isEnabled = false
        for i in [agentItem, copyItem, toggleItem, portItem, thinkingItem, loginItem] { i.target = self }
        modelMenu.autoenablesItems = false
        modelItem.submenu = modelMenu
        menu.addItem(statusLine)
        menu.addItem(agentItem)
        menu.addItem(.separator())
        menu.addItem(copyItem)
        menu.addItem(toggleItem)
        menu.addItem(portItem)
        menu.addItem(modelItem)
        menu.addItem(thinkingItem)
        menu.addItem(withTitle: "Open Server Log", action: #selector(openLog), keyEquivalent: "l").target = self
        menu.addItem(loginItem)
        menu.addItem(.separator())
        menu.addItem(withTitle: "Quit Qwasar", action: #selector(quit), keyEquivalent: "q").target = self
        item.menu = menu

        state.onServerChange = { [weak self] in self?.render() }
        render()
        DispatchQueue.main.async { [weak self] in self?.render() }   // once the button has its size
    }

    func menuNeedsUpdate(_ menu: NSMenu) {
        guard menu === item.menu else { return }
        models = ModelCatalog.scan(extra: state.modelPath.map { [$0] } ?? [])
        state.server.refresh()
        render()
    }

    // MARK: drawing

    func render() {
        let server = state.server
        let port = server.port
        var color: StatusIcon.Dot = .none
        var dim = false, pulsing = false
        let text: String

        switch server.state {
        case .listening:
            text = state.serverInfo.map { "\($0.model.name) on port \(port)" } ?? "Listening on port \(port)"
        case .starting:
            color = .amber; pulsing = true
            text = "Loading the model — port \(port) not open yet"
        case .stopping:
            color = .amber; pulsing = true
            text = "Stopping…"
        case .stopped:
            dim = true
            text = state.modelPath == nil ? "Stopped — no model chosen" : "Stopped"
        case .portBusy:
            color = .red
            text = "Port \(port) is in use by another program"
        case .failed(let why):
            color = .red
            text = "Stopped: \(why)"
        }

        if let button = item.button {
            button.image = color == .none ? glyph.plain : glyph.badged
            button.appearsDisabled = dim
            button.toolTip = "Qwasar — \(text)"
            button.setAccessibilityLabel("Qwasar: \(text)")
            placeDot(in: button)
        }
        dot.color = color.color
        setPulsing(pulsing)

        statusLine.title = text
        copyItem.isEnabled = server.state == .listening
        copyItem.toolTip = server.apiURL
        let running = server.isRunning
        toggleItem.title = running ? "Stop Server" : "Start Server"
        toggleItem.isEnabled = server.state != .stopping && (running || state.modelPath != nil)
        portItem.title = "Port: \(port)…"
        renderModels()
        thinkingItem.state = server.apiThinking ? .on : .off
        thinkingItem.toolTip = "Whether the OpenAI and Anthropic endpoints reason before they answer "
                             + "when a request does not say (off starts the server with --no-think).  "
                             + "The coding agent's own sessions are not affected.  "
                             + (server.isRunning ? "Changing it restarts the server." : "")
        loginItem.state = SMAppService.mainApp.status == .enabled ? .on : .off
    }

    /// "Model: …" and its submenu: one entry per supported model, each the
    /// folder that would be used for it -- the current one if it is that
    /// model, else the one the app was granted, else the first found -- then
    /// any other folder by hand.  Choosing one restarts the server on it.
    private func renderModels() {
        let current = state.modelPath
        let currentFamily = current.flatMap(ModelCatalog.family(of:))
        modelItem.title = "Model: " + (currentFamily?.title
            ?? current.map { ($0 as NSString).lastPathComponent } ?? "none")
        modelItem.toolTip = current

        modelMenu.removeAllItems()
        for family in ModelFamily.allCases {
            let folder = family == currentFamily ? current
                : ModelLibrary.path(for: family)
                ?? models.first(where: { $0.family == family })?.path
            let entry = NSMenuItem(title: family.title, action: #selector(selectModel(_:)),
                                   keyEquivalent: "")
            entry.target = self
            if let folder {
                entry.representedObject = folder
                entry.state = family == currentFamily ? .on : .off
                entry.toolTip = "\(folder)\n\(family.note)"
            } else {
                entry.title = "\(family.title) — not found"
                entry.isEnabled = false
                entry.toolTip = "No \(family.title) folder in the checkout's models/, LM Studio's "
                              + "models or the Hugging Face cache.  Other Folder… can point at one."
            }
            modelMenu.addItem(entry)
        }
        modelMenu.addItem(.separator())
        let other = NSMenuItem(title: "Other Folder…", action: #selector(chooseModel), keyEquivalent: "")
        other.target = self
        other.state = current != nil && currentFamily == nil ? .on : .off
        modelMenu.addItem(other)
    }

    private func placeDot(in button: NSStatusBarButton) {
        let b = button.bounds, img = glyph.plain.size
        let x = (b.width - img.width) / 2 + glyph.dotCenter.x
        var y = (b.height - img.height) / 2 + glyph.dotCenter.y
        if button.isFlipped { y = b.height - y }
        let r = glyph.dotRadius
        dot.frame = NSRect(x: x - r, y: y - r, width: r * 2, height: r * 2)
    }

    /// A slow fade of the dot while the server is up but its port is not:
    /// loading takes long enough that a static "almost" reads as stuck.
    private func setPulsing(_ on: Bool) {
        if !on {
            pulse?.invalidate(); pulse = nil
            dot.alphaValue = 1
            return
        }
        guard pulse == nil else { return }
        let t = Timer(timeInterval: 0.7, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated {
                guard let dot = self?.dot else { return }
                NSAnimationContext.runAnimationGroup { ctx in
                    ctx.duration = 0.6
                    dot.animator().alphaValue = dot.alphaValue < 1 ? 1 : 0.25
                }
            }
        }
        RunLoop.main.add(t, forMode: .common)
        pulse = t
    }

    // MARK: actions

    @objc private func openWindow() { openAgent() }

    @objc private func copyURL() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(state.server.apiURL, forType: .string)
    }

    @objc private func toggle() {
        if state.server.isRunning { state.stopServer() } else { state.startServer() }
    }

    @objc private func choosePort() {
        let server = state.server
        let field = NSTextField(string: String(server.port))
        field.frame = NSRect(x: 0, y: 0, width: 120, height: 24)
        let alert = NSAlert()
        alert.messageText = "Port"
        alert.informativeText = "The port qwasar-server listens on, on 127.0.0.1."
        alert.accessoryView = field
        alert.addButton(withTitle: server.isRunning ? "Restart on This Port" : "Save")
        alert.addButton(withTitle: "Cancel")
        alert.window.initialFirstResponder = field
        NSApp.activate()
        guard alert.runModal() == .alertFirstButtonReturn else { return }
        guard let p = Int(field.stringValue.trimmingCharacters(in: .whitespaces)),
              (1...65535).contains(p) else {
            showError("“\(field.stringValue)” is not a port number.")
            return
        }
        guard p != server.port else { return }
        let wasRunning = server.isRunning
        server.port = p
        if wasRunning { state.restartServer() } else { server.refresh() }
        render()
    }

    /// The server reads it at launch, so a running one is restarted -- after
    /// the turn in flight, if the coding agent is mid-reply.
    @objc private func toggleThinking() {
        let server = state.server
        server.apiThinking.toggle()
        if server.isRunning || state.pendingServerStart { state.requestServerRestart() }
        render()
    }

    @objc private func selectModel(_ sender: NSMenuItem) {
        guard let path = sender.representedObject as? String, path != state.modelPath else { return }
        state.useModel(path: path)
        render()
    }

    @objc private func chooseModel() {
        NSApp.activate()
        state.chooseModel()
        render()
    }

    @objc private func openLog() {
        if FileManager.default.fileExists(atPath: state.server.logURL.path) {
            NSWorkspace.shared.open(state.server.logURL)
        } else {
            showError("There is no log yet; one is written each time the server starts.")
        }
    }

    @objc private func toggleLogin() {
        do {
            if SMAppService.mainApp.status == .enabled {
                try SMAppService.mainApp.unregister()
            } else {
                try SMAppService.mainApp.register()
            }
        } catch {
            showError("Could not change Start at Login: \(error.localizedDescription)")
        }
        render()
    }

    @objc private func quit() { NSApp.terminate(nil) }

    private func showError(_ text: String) {
        let alert = NSAlert()
        alert.messageText = "Qwasar"
        alert.informativeText = text
        NSApp.activate()
        alert.runModal()
    }
}
