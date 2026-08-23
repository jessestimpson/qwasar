# Crucible -- a native macOS coding harness on the qwasar runtime.
#
# No .xcodeproj.  PLAN.md 3.3 assumed an Xcode app target shelling out to make
# for the C; on contact it turned out the whole thing builds with swiftc and a
# hand-assembled bundle, which is both shorter and readable as a text file.  The
# parent tree's rule -- one `make`, no codegen you have to know about -- applies
# to the app as well, so this is the build.
#
#   make          build and sign Crucible.app
#   make run      build, then launch it
#   make gate     build, then run the M0 gate check headless and print it
#   make clean

ROOT      := $(abspath ..)
BUILD     := build
APP       := $(BUILD)/Crucible.app
BIN       := $(APP)/Contents/MacOS/Crucible

# macOS 14 is the deployment floor (PLAN.md 1.3).  Built and run on 26.
DEPLOY    ?= 14.0
MODEL     ?= $(HOME)/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-4bit
TARGET    ?= arm64-apple-macos$(DEPLOY)
SWIFT     ?= swiftc
SWIFTFLAGS ?= -O -target $(TARGET) -swift-version 6 -parse-as-library

# The module map lives in Sources/CQwasar and names the engine headers by a path
# relative to itself.  -I puts it on the Clang importer's search path.
INCLUDES  := -I Sources/CQwasar

# Subdirectories included: CrucibleKit grew a Sandbox/ in M2, and a flat
# wildcard silently drops new files rather than failing to build.
KIT_SRCS  := $(wildcard Sources/CrucibleKit/*.swift) \
             $(wildcard Sources/CrucibleKit/*/*.swift)
APP_SRCS  := $(wildcard Sources/Crucible/*.swift) \
             $(wildcard Sources/Crucible/*/*.swift)
KITLIB    := $(BUILD)/libCrucibleKit.a

# LDLIBS mirrors the parent Makefile's, plus Virtualization for the sandbox and
# the SwiftUI/AppKit pair for the window.  -lqwasar is the static library the
# parent builds from CORE_OBJS only; if a link error ever names a symbol from
# qwasar_tui.o or linenoise.o, the app has reached for a front end and the fix
# is in the app, not here.
FRAMEWORKS := -framework Foundation -framework Metal -framework AVFoundation \
              -framework CoreMedia -framework CoreGraphics -framework Virtualization \
              -framework SwiftUI -framework AppKit -framework Security \
              -framework JavaScriptCore
LDLIBS    := -L$(ROOT) -lqwasar -lm $(FRAMEWORKS)

.PHONY: all help run gate gate-prefix gate-mtp gate-full sandbox test golden guest clean clean-guest lib

all: $(APP)

help:
	@echo "Crucible build targets (see README.md):"
	@echo "  make              build and sign build/Crucible.app"
	@echo "  make run          build, then launch it"
	@echo "  make guest        build the guest image (native macOS build; no Docker)"
	@echo "  make test         the host test suite"
	@echo "  make golden       regenerate the chat-template goldens -- read the diff"
	@echo "  make sandbox      boot a guest and run the sandbox gate"
	@echo "  make gate-full    load the model and run one agent turn, headless"
	@echo "  make gate-prefix  prove the system-prefix KV checkpoint (~80s)"
	@echo "  make gate-mtp     load the MTP draft head and run one speculative turn"
	@echo "  make clean        remove build products, keeping the guest image"
	@echo "  make clean-guest  remove the guest image too"
	@echo ""
	@echo "First run:  make guest && make run"

# The C tree keeps its own build.  Xcode never learns about bin2c or the
# embedded Metal source, and neither does this file.
# MACOSX_DEPLOYMENT_TARGET is passed through so the C objects carry the same
# floor as the Swift.  Without it clang defaults to the host SDK (26.0 here) and
# the app claims a macOS 14 minimum it cannot honour -- the linker says so, and
# it is right.  The engine compiles clean at 14.0; this is a widening change and
# the parent's own binaries are unaffected on any machine that could run them.
lib:
	@MACOSX_DEPLOYMENT_TARGET=$(DEPLOY) $(MAKE) -C $(ROOT) libqwasar.a

# CrucibleKit is a separate module, not just a directory.  That is what makes
# "CrucibleKit contains no SwiftUI" (PLAN.md 3.2) a compiler error rather than a
# convention -- SwiftUI is linked only into the app target below.
$(KITLIB): lib $(KIT_SRCS) Sources/CQwasar/module.modulemap
	@mkdir -p $(BUILD)
	$(SWIFT) $(SWIFTFLAGS) $(INCLUDES) -module-name CrucibleKit \
	    -emit-module -emit-module-path $(BUILD)/CrucibleKit.swiftmodule \
	    -emit-library -static -o $@ $(KIT_SRCS)

# highlight.js, concatenated from the vendored core and language modules.
#
# Generated rather than committed, for the reason tools/bin2c gives for the Metal
# kernels: the pieces stay reviewable and diffable, and the single artefact is
# built.  Offline -- everything it needs is in vendor/.
HLJS := vendor/highlight.js/highlight.bundle.js

$(HLJS): vendor/highlight.js/core.min.js vendor/highlight.js/build.sh
	@sh vendor/highlight.js/build.sh

$(BIN): $(KITLIB) $(APP_SRCS)
	@mkdir -p $(APP)/Contents/MacOS $(APP)/Contents/Resources
	$(SWIFT) $(SWIFTFLAGS) $(INCLUDES) -I $(BUILD) $(APP_SRCS) -o $@ \
	    -L$(BUILD) -lCrucibleKit $(LDLIBS)

$(APP): $(BIN) Resources/Info.plist Crucible.entitlements $(HLJS)
	@cp Resources/Info.plist $(APP)/Contents/Info.plist
	@# Syntax highlighting (PLAN.md 5.6).  Loaded at runtime by JavaScriptCore;
	@# without it code blocks render as plain monospace, which is what the
	@# transcript did before, so a missing bundle degrades rather than breaks.
	@cp $(HLJS) $(APP)/Contents/Resources/highlight.bundle.js
	@echo "  highlight.js staged ($$(wc -c < $(HLJS) | tr -d ' ') bytes)"
	@# The guest image, staged where the app looks for it.
	@#
	@# Without this the app finds no image, falls back to the read-only host
	@# tools, and says so in the session header -- correct on a machine that has
	@# never run `make guest`, and quietly wrong on one that has. APFS clones the
	@# file, so it is instant and costs no additional disk.
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
	@# Ad-hoc signing.  There is no Developer ID on this machine, and ad-hoc is
	@# enough to make the entitlements real for local use -- which is exactly
	@# what the M0 gate needs to find out.  Notarisation is a later problem.
	codesign --force --sign - \
	         --entitlements Crucible.entitlements \
	         --options runtime \
	         --timestamp=none \
	         $(APP)
	@codesign -d --entitlements - $(APP) 2>&1 | sed 's/^/  /'
	@echo "built $(APP)"

run: $(APP)
	open $(APP)

# Headless gate: the same GateCheck the app runs, with no window, so a failure
# is a line of output rather than a screenshot.
gate: $(APP)
	@$(BIN) --gate $(if $(MODEL),--model "$(MODEL)") $(EXTRA) || true

# The system-prefix checkpoint, proven end to end against the engine: prime a
# cold session, then a second one, and require that the second read the whole
# prefix off disk.  Adds ~80s to the gate (the cold prefill), which is why it is
# its own target rather than part of `gate`.
# Speculative decoding, proven against the engine.  Loads the MTP draft head,
# runs one real turn, and reports tokens committed per round -- the number that
# says whether the head is earning the context it costs.
MTP ?= $(HOME)/.cache/qwasar/mtp/Qwen3.8-27B-MTP-bf16
gate-mtp: $(APP)
	codesign --force --sign - --entitlements Crucible.gate.entitlements \
	         --options runtime --timestamp=none $(APP)
	@$(BIN) --gate --model "$(MODEL)" --mtp "$(MTP)" \
	        --prompt "$(PROMPT)" --root "$(ROOT_DIR)" --no-sandbox || true
	@echo "re-signing with the shipping entitlements"
	@codesign --force --sign - --entitlements Crucible.entitlements \
	          --options runtime --timestamp=none $(APP) 2>/dev/null

gate-prefix: $(APP)
	codesign --force --sign - --entitlements Crucible.gate.entitlements \
	         --options runtime --timestamp=none $(APP)
	@$(BIN) --gate --model "$(MODEL)" --prefix || true
	@echo "re-signing with the shipping entitlements"
	@codesign --force --sign - --entitlements Crucible.entitlements \
	          --options runtime --timestamp=none $(APP) 2>/dev/null

# The full M0 gate, provable from a terminal.
#
# Same bundle, re-signed with Crucible.gate.entitlements -- which is
# Crucible.entitlements plus a read-only temporary exception for the model
# folder, because the shipping grant comes from a person clicking an NSOpenPanel
# and that cannot be scripted.  Everything the gate actually tests -- sandbox
# active, virtualization usable, 16 GB mmap alongside both -- is unchanged.
PROMPT ?= Name three prime numbers, with one sentence on why each is prime.
ROOT_DIR ?= $(ROOT)
gate-full: $(APP)
	codesign --force --sign - --entitlements Crucible.gate.entitlements \
	         --options runtime --timestamp=none $(APP)
	@$(BIN) --gate --model "$(MODEL)" --prompt "$(PROMPT)" --root "$(ROOT_DIR)" \
	        --guest "$(abspath $(BUILD)/guest)" $(EXTRA) || true
	@echo "re-signing with the shipping entitlements"
	@codesign --force --sign - --entitlements Crucible.entitlements \
	          --options runtime --timestamp=none $(APP) 2>/dev/null

# ---- tests -------------------------------------------------------------------
#
# The golden-vector suite (PLAN.md 10) is the highest-value test here: it proves
# CrucibleKit's chat template marshalling produces byte-identical token arrays to
# qwasar_apply_chat_template driven from C.  Tests/gen_golden.c is the oracle and
# Tests/golden.tsv is committed, so a change in the engine's template shows up as
# a failing test rather than as drift nobody notices.
#
# It loads only the tokenizer, so it runs in about a second.

CFRAMEWORKS := -framework Foundation -framework Metal -framework AVFoundation \
               -framework CoreMedia -framework CoreGraphics

# Regenerated from the Swift source every time, so the oracle cannot be
# compiled against a stale copy of the tool descriptions.
Tests/toolschemas.inc: Sources/CrucibleKit/ToolSurface.swift tools/schemas2c.py
	@python3 tools/schemas2c.py > $@
	@echo "  regenerated $@ from ToolSurface.swift"

$(BUILD)/gen_golden: lib Tests/gen_golden.c Tests/toolschemas.inc
	@mkdir -p $(BUILD)
	MACOSX_DEPLOYMENT_TARGET=$(DEPLOY) $(CC) -O2 -std=c99 -I$(ROOT) -o $@ \
	    -ITests Tests/gen_golden.c $(ROOT)/libqwasar.a -lm -pthread $(CFRAMEWORKS)

TEST_SRCS := $(wildcard Tests/*.swift)

$(BUILD)/crucible-test: $(KITLIB) $(TEST_SRCS)
	$(SWIFT) $(SWIFTFLAGS) $(INCLUDES) -I $(BUILD) $(TEST_SRCS) -o $@ \
	    -L$(BUILD) -lCrucibleKit $(LDLIBS)

test: $(BUILD)/crucible-test
	@./$(BUILD)/crucible-test --model "$(MODEL)" --golden Tests/golden.tsv

# Regenerates the goldens from the C oracle.  Run this ONLY when the engine's
# template changed on purpose, and read the diff before committing it.
golden: $(BUILD)/gen_golden
	./$(BUILD)/gen_golden "$(MODEL)" > Tests/golden.tsv
	@echo "regenerated Tests/golden.tsv -- read the diff"

# M2's gate: clone the image, boot it, and talk to the warden over vsock.
#
# Signed with the gate entitlements for the same reason `gate-full` is: during
# development the guest image lives in the repository, and a sandboxed app can
# only reach it through a grant.  The shipping app will carry the image in its
# own bundle and copy it into its container on first run, at which point no
# exception is needed -- see PLAN.md 6.2.
# The gate's project is the warden's own source, not $(ROOT): the repo root
# carries build/ -- several GB of guest image and cache -- and the raw share
# has no exclusion list, so copying it into a 3 GB guest disk is ENOSPC
# reported three layers away as a cookie-file write failure.
SANDBOX_ROOT ?= Guest/warden
sandbox: $(APP)
	@codesign --force --sign - --entitlements Crucible.gate.entitlements \
	          --options runtime --timestamp=none $(APP)
	@$(BIN) --sandbox --guest "$(abspath $(BUILD)/guest)" \
	        --root "$(abspath $(SANDBOX_ROOT))" || true
	@codesign --force --sign - --entitlements Crucible.entitlements \
	          --options runtime --timestamp=none $(APP) 2>/dev/null

# ---- guest image -------------------------------------------------------------
#
# Built once, cloned per session (PLAN.md 6.3).  Built natively on macOS --
# Alpine packages are tarballs and BEAM is portable, so no Docker and no
# Linux anywhere (PLAN.md 6.2, and Guest/mkimage.sh for the how).
guest:
	@Guest/mkimage.sh

clean:
	rm -rf $(BUILD)

# The guest image is expensive to rebuild and is not a build artefact in the
# ordinary sense, so `clean` leaves it alone and this removes it explicitly.
clean-guest:
	rm -rf $(BUILD)/guest
