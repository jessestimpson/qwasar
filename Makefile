# qwasar -- Qwen3.8 inference on macOS Metal.
#
# One `make`.  No Python, no Metal toolchain, no codegen the user has to know
# about.  Metal kernels under metal/ are turned into an embedded C string by
# tools/bin2c and compiled at runtime; see the comment at the top of
# qwasar_metal.m for why that is the right trade here.

UNAME_S := $(shell uname -s)
ifneq ($(UNAME_S),Darwin)
$(error qwasar targets macOS with Metal)
endif

CC       ?= cc
DEBUG    ?= -g
CFLAGS   ?= -O3 -ffast-math $(DEBUG) -mcpu=native -Wall -Wextra -std=c99
OBJCFLAGS ?= -O3 -ffast-math $(DEBUG) -mcpu=native -Wall -Wextra -fobjc-arc
LDLIBS   ?= -lm -pthread -framework Foundation -framework Metal

# common.metal defines the shared helpers every other kernel file uses, and the
# files are concatenated into a single translation unit, so it must come first.
METAL_SRCS := metal/common.metal \
              $(filter-out metal/common.metal,$(sort $(wildcard metal/*.metal)))
CORE_OBJS  := qwasar.o qwasar_graph.o qwasar_tokenizer.o qwasar_json.o qwasar_cpu.o qwasar_metal.o

.PHONY: all clean test help check-metal

all: qwasar

help:
	@echo "qwasar build targets:"
	@echo "  make          build ./qwasar"
	@echo "  make test     build and run tests"
	@echo "  make clean    remove build outputs"
	@echo "  make check-metal  offline kernel syntax check (needs the Metal Toolchain)"

qwasar: qwasar_cli.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

# Kernel sources are embedded rather than read from disk at runtime, so the
# binary stays relocatable and cannot silently pick up a stale metal/ tree.
tools/bin2c: tools/bin2c.c
	$(CC) -O2 -std=c99 -o $@ $<

qwasar_metal_src.inc: tools/bin2c $(METAL_SRCS) Makefile
	./tools/bin2c qwasar_metal_source $(METAL_SRCS) > $@

qwasar_metal.o: qwasar_metal.m qwasar_gpu.h qwasar_metal_src.inc
	$(CC) $(OBJCFLAGS) -c -o $@ $<

qwasar.o:      qwasar.c qwasar.h qwasar_gpu.h qwasar_json.h qwasar_model.h
qwasar_graph.o: qwasar_graph.c qwasar.h qwasar_gpu.h qwasar_model.h
qwasar_tokenizer.o: qwasar_tokenizer.c qwasar.h qwasar_json.h qwasar_unicode.inc
qwasar_cpu.o:  qwasar_cpu.c qwasar_model.h
qwasar_json.o: qwasar_json.c qwasar_json.h
qwasar_cli.o:  qwasar_cli.c qwasar.h qwasar_gpu.h

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

QWASAR_TEST_MODEL ?= $(HOME)/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-4bit

TESTS := tests/test_json tests/test_tokenizer tests/test_qmv tests/test_ops \
         tests/test_gdn tests/test_attn tests/test_forward

tests/test_json: tests/test_json.c qwasar_json.o qwasar_json.h
	$(CC) $(CFLAGS) -I. -o $@ tests/test_json.c qwasar_json.o $(LDLIBS)

tests/test_qmv: tests/test_qmv.c $(CORE_OBJS) qwasar.h qwasar_gpu.h qwasar_model.h
	$(CC) $(CFLAGS) -I. -o $@ tests/test_qmv.c $(CORE_OBJS) $(LDLIBS)

tests/test_ops: tests/test_ops.c $(CORE_OBJS) qwasar.h qwasar_gpu.h qwasar_model.h
	$(CC) $(CFLAGS) -I. -o $@ tests/test_ops.c $(CORE_OBJS) $(LDLIBS)

tests/test_gdn: tests/test_gdn.c $(CORE_OBJS) qwasar.h qwasar_gpu.h qwasar_model.h
	$(CC) $(CFLAGS) -I. -o $@ tests/test_gdn.c $(CORE_OBJS) $(LDLIBS)

tests/test_attn: tests/test_attn.c $(CORE_OBJS) qwasar.h qwasar_gpu.h qwasar_model.h
	$(CC) $(CFLAGS) -I. -o $@ tests/test_attn.c $(CORE_OBJS) $(LDLIBS)

tests/test_forward: tests/test_forward.c $(CORE_OBJS) qwasar.h qwasar_gpu.h qwasar_model.h
	$(CC) $(CFLAGS) -I. -o $@ tests/test_forward.c $(CORE_OBJS) $(LDLIBS)

tests/test_tokenizer: tests/test_tokenizer.c $(CORE_OBJS) qwasar.h qwasar_json.h
	$(CC) $(CFLAGS) -I. -o $@ tests/test_tokenizer.c $(CORE_OBJS) $(LDLIBS)

test: $(TESTS)
	@for t in $(TESTS); do echo "== $$t"; \
		QWASAR_TEST_MODEL="$(QWASAR_TEST_MODEL)" ./$$t || exit 1; done

# Optional offline syntax check for the kernels.
#
# The shipping build never needs the Metal Toolchain -- kernels are compiled at
# runtime -- but when the component is installed this catches a bad kernel in a
# second instead of at model-load time, several gigabytes into a test run.
check-metal:
	@if xcrun -sdk macosx metal --version >/dev/null 2>&1; then \
		cat $(METAL_SRCS) > .metal_check.metal; \
		xcrun -sdk macosx metal -Wall -Wextra -ffast-math \
			-c .metal_check.metal -o /dev/null && echo "metal: kernels ok"; \
		rm -f .metal_check.metal; \
	else \
		echo "metal: toolchain not installed; skipping (runtime compile still validates)"; \
	fi

clean:
	rm -f qwasar *.o tools/bin2c qwasar_metal_src.inc .metal_check.metal $(TESTS) tests/*.o
