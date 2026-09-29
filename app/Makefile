# Qwasar -- the menu bar app that runs qwasar-server, and the coding agent's
# window, which talks to it (PLAN-qwasar.md §4).
#
# No .xcodeproj: swiftc and a hand-assembled bundle, readable as a text file.
# The app links no engine.  qwasar-server is built by the parent tree and
# copied into the bundle as a helper, signed to inherit the app's sandbox.
#
#   make          build and sign Qwasar.app
#   make run      build, then launch it (the menu bar item)
#   make agent    build, then launch it with the coding agent's window open
#   make gate     the entitlement and VM probes, headless
#   make gate-full  ... and the helper on $(MODEL), and one agent turn through it
#   make test     the host test suite
#   make guest    the guest image (native macOS build; no Docker)
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

FRAMEWORKS := -framework Foundation -framework Virtualization -framework SwiftUI \
              -framework AppKit -framework Security -framework JavaScriptCore \
              -framework ServiceManagement
LDLIBS    := $(FRAMEWORKS)

# The entitlements the shipping app carries, and the helper's.  If the M3
# gate finds the helper cannot inherit the model-folder grant, both become
# Hardened-Runtime-only files here and nothing else changes (PLAN-qwasar.md 4.2).
ENTITLEMENTS        ?= Qwasar.entitlements
HELPER_ENTITLEMENTS ?= Helper.entitlements

.PHONY: all help run agent gate gate-full sandbox test guest clean clean-guest server

all: $(APP)

help:
	@echo "Qwasar build targets (see README.md):"
	@echo "  make              build and sign build/Qwasar.app"
	@echo "  make run          build, then launch it"
	@echo "  make agent        build, then launch it with the coding agent's window"
	@echo "  make guest        build the guest image (native macOS build; no Docker)"
	@echo "  make test         the host test suite"
	@echo "  make sandbox      boot a guest and run the sandbox gate"
	@echo "  make gate         the entitlement and VM probes"
	@echo "  make gate-full    start the helper on the model and run one agent turn, headless"
	@echo "  make clean        remove build products, keeping the guest image"
	@echo "  make clean-guest  remove the guest image too"
	@echo ""
	@echo "First run:  make guest && make run"

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

# highlight.js, concatenated from the vendored core and language modules.
HLJS := vendor/highlight.js/highlight.bundle.js

$(HLJS): vendor/highlight.js/core.min.js vendor/highlight.js/build.sh
	@sh vendor/highlight.js/build.sh

$(BIN): $(KITLIB) $(APP_SRCS)
	@mkdir -p $(APP)/Contents/MacOS $(APP)/Contents/Resources
	$(SWIFT) $(SWIFTFLAGS) -I $(BUILD) $(APP_SRCS) -o $@ \
	    -L$(BUILD) -lQwasarKit $(LDLIBS)

$(APP): $(BIN) server Resources/Info.plist $(ENTITLEMENTS) $(HELPER_ENTITLEMENTS) $(HLJS)
	@cp Resources/Info.plist $(APP)/Contents/Info.plist
	@plutil -replace QWDefaultModel -string "$(MODEL)" $(APP)/Contents/Info.plist
	@plutil -replace QWModelsDir -string "$(MODELS_DIR)" $(APP)/Contents/Info.plist
	@cp $(HLJS) $(APP)/Contents/Resources/highlight.bundle.js
	@# The guest image, staged where the app looks for it.  Without it the app
	@# falls back to the read-only host tools and says so in the session header.
	@if [ -f $(BUILD)/guest/disk.img ]; then \
	    mkdir -p $(APP)/Contents/Resources/guest; \
	    for f in disk.img Image initramfs; do \
	        cp -c $(BUILD)/guest/$$f $(APP)/Contents/Resources/guest/$$f 2>/dev/null \
	          || cp $(BUILD)/guest/$$f $(APP)/Contents/Resources/guest/$$f; \
	    done; \
	    echo "  guest image staged into the bundle"; \
	else \
	    echo "  NO guest image -- run 'make guest'; sessions will be read-only"; \
	fi
	@# The helper, then the app: signing the bundle seals what is inside it.
	@cp $(ROOT)/qwasar-server $(HELPER)
	codesign --force --sign - --entitlements $(HELPER_ENTITLEMENTS) \
	         --options runtime --timestamp=none $(HELPER)
	codesign --force --sign - --entitlements $(ENTITLEMENTS) \
	         --options runtime --timestamp=none $(APP)
	@codesign -d --entitlements - $(APP) 2>&1 | sed 's/^/  /'
	@echo "built $(APP)"

run: $(APP)
	open $(APP)

agent: $(APP)
	open $(APP) --args --agent

gate: $(APP)
	@$(BIN) --gate || true

# The full gate: the bundle re-signed with the gate entitlements (a read-only
# exception for the model folder, since the shipping grant comes from a
# person clicking a panel), the helper started on $(MODEL), one agent turn
# through it with the tools in the guest.
PROMPT ?= List the files here, then say in one sentence what this project is.
ROOT_DIR ?= Guest/warden
gate-full: $(APP)
	codesign --force --sign - --entitlements Qwasar.gate.entitlements \
	         --options runtime --timestamp=none $(APP)
	@$(BIN) --gate --model "$(MODEL)" --prompt "$(PROMPT)" --root "$(abspath $(ROOT_DIR))" \
	        --guest "$(abspath $(BUILD)/guest)" $(EXTRA) || true
	@echo "re-signing with the shipping entitlements"
	@codesign --force --sign - --entitlements $(ENTITLEMENTS) \
	          --options runtime --timestamp=none $(APP) 2>/dev/null

# ---- tests -------------------------------------------------------------------

TEST_SRCS := $(wildcard Tests/*.swift)

$(BUILD)/qwasar-test: $(KITLIB) $(TEST_SRCS)
	$(SWIFT) $(SWIFTFLAGS) -I $(BUILD) $(TEST_SRCS) -o $@ \
	    -L$(BUILD) -lQwasarKit $(LDLIBS)

test: $(BUILD)/qwasar-test
	@./$(BUILD)/qwasar-test

# M2's gate: clone the image, boot it, and talk to the warden over vsock.
SANDBOX_ROOT ?= Guest/warden
sandbox: $(APP)
	@codesign --force --sign - --entitlements Qwasar.gate.entitlements \
	          --options runtime --timestamp=none $(APP)
	@$(BIN) --sandbox --guest "$(abspath $(BUILD)/guest)" \
	        --root "$(abspath $(SANDBOX_ROOT))" || true
	@codesign --force --sign - --entitlements $(ENTITLEMENTS) \
	          --options runtime --timestamp=none $(APP) 2>/dev/null

# ---- guest image -------------------------------------------------------------
guest:
	@Guest/mkimage.sh

clean:
	rm -rf $(BUILD)/Qwasar.app $(BUILD)/*.a $(BUILD)/*.swiftmodule $(BUILD)/*.swiftdoc \
	       $(BUILD)/*.swiftsourceinfo $(BUILD)/*.abi.json $(BUILD)/qwasar-test

clean-guest:
	rm -rf $(BUILD)/guest
