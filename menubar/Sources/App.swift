import AppKit
import ServiceManagement

@main
@MainActor
enum Main {
    static let delegate = AppDelegate()     // NSApplication holds its delegate weakly

    static func main() {
        let app = NSApplication.shared
        app.setActivationPolicy(.accessory)  // menu bar only: no Dock icon, no app menu
        app.delegate = delegate
        app.run()
    }
}

/// The status item and its menu.  Everything shown is read from the
/// ServerController's state at the moment it changes or the menu opens.
@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate, NSMenuDelegate {
    private let server = ServerController()
    private var item: NSStatusItem!
    private let glyph = StatusIcon.glyph()
    private var dot: DotView!
    private var pulse: Timer?

    private let statusLine = NSMenuItem(title: "", action: nil, keyEquivalent: "")
    private let copyItem = NSMenuItem(title: "Copy API URL", action: #selector(copyURL), keyEquivalent: "c")
    private let toggleItem = NSMenuItem(title: "", action: #selector(toggle), keyEquivalent: "s")
    private let portItem = NSMenuItem(title: "", action: #selector(choosePort), keyEquivalent: "")
    private let modelItem = NSMenuItem(title: "", action: #selector(chooseModel), keyEquivalent: "")
    private let loginItem = NSMenuItem(title: "Start at Login", action: #selector(toggleLogin), keyEquivalent: "")

    func applicationDidFinishLaunching(_ note: Notification) {
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
        for i in [copyItem, toggleItem, portItem, modelItem, loginItem] { i.target = self }
        menu.addItem(statusLine)
        menu.addItem(copyItem)
        menu.addItem(.separator())
        menu.addItem(toggleItem)
        menu.addItem(portItem)
        menu.addItem(modelItem)
        menu.addItem(withTitle: "Open Log", action: #selector(openLog), keyEquivalent: "l").target = self
        menu.addItem(loginItem)
        menu.addItem(.separator())
        menu.addItem(withTitle: "Quit Qwasar Server", action: #selector(quit), keyEquivalent: "q").target = self
        item.menu = menu

        server.onChange = { [weak self] in self?.render() }
        render()
        DispatchQueue.main.async { [weak self] in self?.render() }   // once the button has its size

        if server.modelPath == nil { chooseModel() }
        server.start()
    }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        guard server.isRunning else { return .terminateNow }
        server.stop { NSApp.reply(toApplicationShouldTerminate: true) }
        return .terminateLater
    }

    func menuNeedsUpdate(_ menu: NSMenu) {
        server.refresh()
        render()
    }

    // MARK: drawing

    private func render() {
        let port = server.port
        var color: StatusIcon.Dot = .none
        var dim = false, pulsing = false
        let text: String

        switch server.state {
        case .listening:
            text = "Listening on port \(port)"
        case .starting:
            color = .amber; pulsing = true
            text = "Loading model — port \(port) not open yet"
        case .stopping:
            color = .amber; pulsing = true
            text = "Stopping…"
        case .stopped:
            dim = true
            text = "Stopped"
        case .portBusy:
            color = .red
            text = "Port \(port) is in use by another program"
        case .failed(let why):
            color = .red
            text = "Stopped: \(why)"
        }

        if let button = item.button {
            // All well is the plain Q; anything else to say is the dot.
            button.image = color == .none ? glyph.plain : glyph.badged
            button.appearsDisabled = dim
            button.toolTip = "Qwasar Server — \(text)"
            button.setAccessibilityLabel("Qwasar Server: \(text)")
            placeDot(in: button)
        }
        dot.color = color.color
        setPulsing(pulsing)

        statusLine.title = text
        copyItem.isEnabled = server.state == .listening
        copyItem.toolTip = server.apiURL
        let running = server.isRunning
        toggleItem.title = running ? "Stop Server" : "Start Server"
        toggleItem.isEnabled = server.state != .stopping
        portItem.title = "Port: \(port)…"
        let model = server.modelPath.map { ($0 as NSString).lastPathComponent } ?? "none"
        modelItem.title = "Model: \(model)…"
        modelItem.toolTip = server.modelPath
        loginItem.state = SMAppService.mainApp.status == .enabled ? .on : .off
    }

    /// Puts the dot where the Q's cut-out is.  The button draws its image
    /// centred, so the spot follows from the image's place; the button may be
    /// flipped, the glyph's coordinates never are.
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

    @objc private func copyURL() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(server.apiURL, forType: .string)
    }

    @objc private func toggle() {
        if server.isRunning { server.stop() } else { server.start() }
    }

    @objc private func choosePort() {
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
        if wasRunning { server.restart() } else { server.refresh() }
        render()
    }

    @objc private func chooseModel() {
        let panel = NSOpenPanel()
        panel.message = "Choose the model folder (the one containing config.json)."
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        if let current = server.modelPath { panel.directoryURL = URL(fileURLWithPath: current) }
        NSApp.activate()
        guard panel.runModal() == .OK, let url = panel.url else { return }
        guard ServerController.isModelFolder(url.path) else {
            showError("\(url.lastPathComponent) has no config.json, so it is not a model folder.")
            return
        }
        server.setModelPath(url.path)
        if server.isRunning { server.restart() }
        render()
    }

    @objc private func openLog() {
        if FileManager.default.fileExists(atPath: server.logURL.path) {
            NSWorkspace.shared.open(server.logURL)
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

    @objc private func quit() {
        NSApp.terminate(nil)
    }

    private func showError(_ text: String) {
        let alert = NSAlert()
        alert.messageText = "Qwasar Server"
        alert.informativeText = text
        NSApp.activate()
        alert.runModal()
    }
}
