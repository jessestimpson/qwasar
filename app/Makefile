# Qwasar -- the menu bar app that runs qwasar-server, and the coding agent's
# window, which talks to it (PLAN-qwasar.md §4).
#
# No .xcodeproj: swiftc and a hand-assembled bundle, readable as a text file.
# The app links no engine.  qwasar-server is built by the parent tree and
# copied into the bundle as a helper.
#
#   make          build and sign Qwasar.app
#   make install  ... and copy it to /Applications (INSTALL_DIR)
#   make uninstall
#   make run      build, then launch it (the menu bar item)
#   make agent    ... with the coding agent's window open
#   make gate     the helper on $(MODEL), headless
#   make gate-full  ... and one agent turn through it
#   make test     the host test suite
#   make clean

ROOT      := $(abspath ..)
BUILD     := build
APP       := $(BUILD)/Qwasar.app
BIN       := $(APP)/Contents/MacOS/Qwasar
HELPER    := $(APP)/Contents/MacOS/qwasar-server

# macOS 14 is the deployment floor.  Built and run on 26.
DEPLOY    ?= 14.0
MODEL     ?= $(shell [ -e $(ROOT)/qwasar-model ] && cd $(ROOT)/qwasar-model && pwd -P)
MODELS_DIR ?= $(shell [ -d $(ROOT)/models ] && cd $(ROOT)/models && pwd -P)
TARGET    ?= arm64-apple-macos$(DEPLOY)
SWIFT     ?= swiftc
SWIFTFLAGS ?= -O -target $(TARGET) -swift-version 6 -parse-as-library

# Subdirectories included: a flat wildcard silently drops new files rather
# than failing to build.
KIT_SRCS  := $(wildcard Sources/QwasarKit/*.swift) \
             $(wildcard Sources/QwasarKit/*/*.swift)
APP_SRCS  := $(wildcard Sources/Qwasar/*.swift) \
             $(wildcard Sources/Qwasar/*/*.swift)
KITLIB    := $(BUILD)/libQwasarKit.a

FRAMEWORKS := -framework Foundation -framework SwiftUI \
              -framework AppKit -framework Security -framework JavaScriptCore \
              -framework ServiceManagement
LDLIBS    := $(FRAMEWORKS)

# The entitlements the shipping app carries, and the helper's.  Neither is
# App Sandboxed: a session's tools run as the user, which a sandbox would
# forbid (Qwasar.entitlements).
ENTITLEMENTS        ?= Qwasar.entitlements
HELPER_ENTITLEMENTS ?= Helper.entitlements

.PHONY: all help run agent gate gate-full test clean server install uninstall

all: $(APP)

help:
	@echo "Qwasar build targets (see README.md):"
	@echo "  make              build and sign build/Qwasar.app"
	@echo "  make install      ... and copy it to $(INSTALL_DIR)"
	@echo "  make uninstall    remove it from $(INSTALL_DIR)"
	@echo "  make run          build, then launch it"
	@echo "  make agent        build, then launch it with the coding agent's window"
	@echo "  make test         the host test suite"
	@echo "  make gate         start the helper on the model, headless"
	@echo "  make gate-full    ... and run one agent turn through it"
	@echo "  make clean        remove build products"
	@echo ""
	@echo "To start:   make run"

# The server, from the parent tree, with the same deployment floor as the
# Swift so the bundle honours the minimum it claims.
server:
	@MACOSX_DEPLOYMENT_TARGET=$(DEPLOY) $(MAKE) -C $(ROOT) qwasar-server

# QwasarKit is a separate module: "QwasarKit contains no SwiftUI" (spec 3.2)
# is a compiler error rather than a convention.
$(KITLIB): $(KIT_SRCS)
	@mkdir -p $(BUILD)
	$(SWIFT) $(SWIFTFLAGS) -module-name QwasarKit \
	    -emit-module -emit-module-path $(BUILD)/QwasarKit.swiftmodule \
	    -emit-library -static -o $@ $(KIT_SRCS)

# The icon, drawn by Tools/AppIcon.swift (the menu bar's Q on a tile) and
# packed by iconutil: no binary asset in the tree.
ICON_TOOL := $(BUILD)/AppIcon
ICNS      := $(BUILD)/AppIcon.icns

$(ICON_TOOL): Tools/AppIcon.swift
	@mkdir -p $(BUILD)
	$(SWIFT) -O -target $(TARGET) $< -o $@

$(ICNS): $(ICON_TOOL)
	@rm -rf $(BUILD)/AppIcon.iconset
	@$(ICON_TOOL) $(BUILD)/AppIcon.iconset
	iconutil -c icns $(BUILD)/AppIcon.iconset -o $@
	@rm -rf $(BUILD)/AppIcon.iconset

# highlight.js, concatenated from the vendored core and language modules.
HLJS := vendor/highlight.js/highlight.bundle.js

$(HLJS): vendor/highlight.js/core.min.js vendor/highlight.js/build.sh
	@sh vendor/highlight.js/build.sh

$(BIN): $(KITLIB) $(APP_SRCS)
	@mkdir -p $(APP)/Contents/MacOS $(APP)/Contents/Resources
	$(SWIFT) $(SWIFTFLAGS) -I $(BUILD) $(APP_SRCS) -o $@ \
	    -L$(BUILD) -lQwasarKit $(LDLIBS)

$(APP): $(BIN) server Resources/Info.plist $(ENTITLEMENTS) $(HELPER_ENTITLEMENTS) $(HLJS) $(ICNS)
	@cp Resources/Info.plist $(APP)/Contents/Info.plist
	@cp $(ICNS) $(APP)/Contents/Resources/AppIcon.icns
	@plutil -replace QWDefaultModel -string "$(MODEL)" $(APP)/Contents/Info.plist
	@plutil -replace QWModelsDir -string "$(MODELS_DIR)" $(APP)/Contents/Info.plist
	@cp $(HLJS) $(APP)/Contents/Resources/highlight.bundle.js
	@# A guest image staged by an older build has nothing left to use it.
	@rm -rf $(APP)/Contents/Resources/guest
	@# The helper, then the app: signing the bundle seals what is inside it.
	@cp $(ROOT)/qwasar-server $(HELPER)
	codesign --force --sign - --entitlements $(HELPER_ENTITLEMENTS) \
	         --options runtime --timestamp=none $(HELPER)
	codesign --force --sign - --entitlements $(ENTITLEMENTS) \
	         --options runtime --timestamp=none $(APP)
	@codesign -d --entitlements - $(APP) 2>&1 | sed 's/^/  /'
	@# Finder and the Dock cache icons by bundle; a new date makes them look again.
	@touch $(APP)
	@echo "built $(APP)"

run:
	@$(MAKE) --no-print-directory $(APP)
	open $(APP)

# Installed, it is the same bundle: it still finds this checkout's models
# (QWDefaultModel and QWModelsDir in its Info.plist), and the Model menu can
# point it anywhere else.  ditto keeps the signature intact; a running copy
# is not replaced underneath itself.
INSTALL_DIR ?= /Applications

install: $(APP)
	@if pgrep -xq Qwasar; then \
	    echo "Qwasar is running; quit it (menu bar Q > Quit) and run make install again." >&2; \
	    exit 1; \
	fi
	rm -rf "$(INSTALL_DIR)/Qwasar.app"
	ditto $(APP) "$(INSTALL_DIR)/Qwasar.app"
	@echo "installed $(INSTALL_DIR)/Qwasar.app"
	@echo "  its default model: $$(plutil -extract QWDefaultModel raw "$(INSTALL_DIR)/Qwasar.app/Contents/Info.plist")"

uninstall:
	@if pgrep -xq Qwasar; then \
	    echo "Qwasar is running; quit it first." >&2; exit 1; \
	fi
	rm -rf "$(INSTALL_DIR)/Qwasar.app"
	@echo "removed $(INSTALL_DIR)/Qwasar.app (sessions and settings are kept)"

agent:
	@$(MAKE) --no-print-directory $(APP)
	open $(APP) --args --agent

# The helper started on $(MODEL), headless; gate-full adds one agent turn
# through it, its tools on this Mac in $(ROOT_DIR).
PROMPT ?= List the files here, then say in one sentence what this project is.
ROOT_DIR ?= .
gate: $(APP)
	@$(BIN) --gate --model "$(MODEL)" || true

gate-full: $(APP)
	@$(BIN) --gate --model "$(MODEL)" --prompt "$(PROMPT)" --root "$(abspath $(ROOT_DIR))" $(EXTRA) || true

# ---- tests -------------------------------------------------------------------

TEST_SRCS := $(wildcard Tests/*.swift)

$(BUILD)/qwasar-test: $(KITLIB) $(TEST_SRCS)
	$(SWIFT) $(SWIFTFLAGS) -I $(BUILD) $(TEST_SRCS) -o $@ \
	    -L$(BUILD) -lQwasarKit $(LDLIBS)

test: $(BUILD)/qwasar-test
	@./$(BUILD)/qwasar-test

clean:
	rm -rf $(BUILD)/Qwasar.app $(BUILD)/*.a $(BUILD)/*.swiftmodule $(BUILD)/*.swiftdoc \
	       $(BUILD)/*.swiftsourceinfo $(BUILD)/*.abi.json $(BUILD)/qwasar-test \
	       $(ICON_TOOL) $(ICNS) $(BUILD)/AppIcon.iconset
