# Qwasar Server -- a menu bar app that runs qwasar-server and shows, in the
# menu bar, whether its port is accepting connections.
#
# Built like crucible/: swiftc and a hand-assembled bundle, no .xcodeproj.
# The server binary is copied into the bundle, so the app stands alone.
#
#   make           build build/QwasarServer.app
#   make run       build, then launch it
#   make install   copy it to /Applications
#   make clean

ROOT      := $(abspath ..)
BUILD     := build
APP       := $(BUILD)/QwasarServer.app
BIN       := $(APP)/Contents/MacOS/QwasarServer
HELPER    := $(APP)/Contents/MacOS/qwasar-server

DEPLOY    ?= 14.0
TARGET    ?= arm64-apple-macos$(DEPLOY)
SWIFT     ?= swiftc
SWIFTFLAGS ?= -O -target $(TARGET) -swift-version 6 -parse-as-library

SRCS      := $(wildcard Sources/*.swift)

# The model the app offers until one is chosen in its menu: wherever the
# checkout's qwasar-model link points, resolved now, because the bundle will
# not be sitting beside it.
MODEL     ?= $(shell [ -e $(ROOT)/qwasar-model ] && cd $(ROOT)/qwasar-model && pwd -P)

.PHONY: all run install clean server

all: $(APP)

# The parent tree keeps its own build.
server:
	@MACOSX_DEPLOYMENT_TARGET=$(DEPLOY) $(MAKE) -C $(ROOT) qwasar-server

$(BIN): $(SRCS)
	@mkdir -p $(APP)/Contents/MacOS $(APP)/Contents/Resources
	$(SWIFT) $(SWIFTFLAGS) $(SRCS) -o $@ -framework AppKit -framework ServiceManagement

$(APP): $(BIN) server Resources/Info.plist
	@cp Resources/Info.plist $(APP)/Contents/Info.plist
	@plutil -replace QWDefaultModel -string "$(MODEL)" $(APP)/Contents/Info.plist
	@cp $(ROOT)/qwasar-server $(HELPER)
	@# Ad-hoc signing, helper first: signing the bundle seals what is inside it.
	@codesign --force --sign - --options runtime --timestamp=none $(HELPER)
	@codesign --force --sign - --options runtime --timestamp=none $(APP)
	@echo "built $(APP)$(if $(MODEL), (default model $(MODEL)))"

run: $(APP)
	open $(APP)

install: $(APP)
	@rm -rf "/Applications/Qwasar Server.app"
	ditto $(APP) "/Applications/Qwasar Server.app"
	@echo "installed /Applications/Qwasar Server.app"

clean:
	rm -rf $(BUILD)
