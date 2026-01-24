# FLUX.2 klein 4B - Pure C Inference Engine
# Makefile

CC = gcc
CFLAGS_BASE = -Wall -Wextra -O3 -march=native -ffast-math
LDFLAGS = -lm

# Platform detection
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

# Source files
SRCS = flux.c flux_kernels.c flux_tokenizer.c flux_vae.c flux_transformer.c flux_sample.c flux_image.c flux_safetensors.c flux_qwen3.c flux_qwen3_tokenizer.c terminals.c
OBJS = $(SRCS:.c=.o)
CLI_SRCS = flux_cli.c linenoise.c
CLI_OBJS = $(CLI_SRCS:.c=.o)
MAIN = main.c
TARGET = flux
LIB = libflux.a

# Debug build flags
DEBUG_CFLAGS = -Wall -Wextra -g -O0 -DDEBUG -fsanitize=address

.PHONY: all clean debug lib install info test pngtest help generic blas mps

# Default: show available targets
all: help

help:
	@echo "FLUX.2 klein 4B - Build Targets"
	@echo ""
	@echo "Choose a backend:"
	@echo "  make generic  - Pure C, no dependencies (slow)"
	@echo "  make blas     - With BLAS acceleration (~30x faster)"
ifeq ($(UNAME_S),Darwin)
ifeq ($(UNAME_M),arm64)
	@echo "  make mps      - Apple Silicon with Metal GPU (fastest)"
endif
endif
ifneq ($(shell command -v hipcc 2> /dev/null),)
	@echo "  make rocm     - AMD GPU with ROCm/HIP (fast)"
endif
	@echo ""
	@echo "Other targets:"
	@echo "  make clean    - Remove build artifacts"
	@echo "  make test     - Run inference test"
	@echo "  make pngtest  - Compare PNG load on compressed image"
	@echo "  make info     - Show build configuration"
	@echo "  make lib      - Build static library"
	@echo ""
	@echo "Example: make mps && ./flux -d flux-klein-model -p \"a cat\" -o cat.png"

# ======================================================================
# Backend: generic (pure C, no BLAS)
# ======================================================================
generic: CFLAGS = $(CFLAGS_BASE) -DGENERIC_BUILD
generic: clean $(TARGET)
	@echo ""
	@echo "Built with GENERIC backend (pure C, no BLAS)"
	@echo "This will be slow but has zero dependencies."

# ======================================================================
# Backend: blas (Accelerate on macOS, OpenBLAS on Linux)
# ======================================================================
ifeq ($(UNAME_S),Darwin)
blas: CFLAGS = $(CFLAGS_BASE) -DUSE_BLAS -DACCELERATE_NEW_LAPACK
blas: LDFLAGS += -framework Accelerate
else
blas: CFLAGS = $(CFLAGS_BASE) -DUSE_BLAS -DUSE_OPENBLAS -I/usr/include/openblas
blas: LDFLAGS += -lopenblas
endif
blas: clean $(TARGET)
	@echo ""
	@echo "Built with BLAS backend (~30x faster than generic)"

# ======================================================================
# Backend: mps (Apple Silicon Metal GPU)
# ======================================================================
ifeq ($(UNAME_S),Darwin)
ifeq ($(UNAME_M),arm64)
MPS_CFLAGS = $(CFLAGS_BASE) -DUSE_BLAS -DUSE_METAL -DACCELERATE_NEW_LAPACK
MPS_OBJCFLAGS = $(MPS_CFLAGS) -fobjc-arc
MPS_LDFLAGS = $(LDFLAGS) -framework Accelerate -framework Metal -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph -framework Foundation

mps: clean mps-build
	@echo ""
	@echo "Built with MPS backend (Metal GPU acceleration)"

mps-build: $(SRCS:.c=.mps.o) $(CLI_SRCS:.c=.mps.o) flux_metal.o main.mps.o
	$(CC) $(MPS_CFLAGS) -o $(TARGET) $^ $(MPS_LDFLAGS)

%.mps.o: %.c flux.h flux_kernels.h
	$(CC) $(MPS_CFLAGS) -c -o $@ $<

# Embed Metal shader source as C array (runtime compilation, no Metal toolchain needed)
flux_shaders_source.h: flux_shaders.metal
	xxd -i $< > $@

flux_metal.o: flux_metal.m flux_metal.h flux_shaders_source.h
	$(CC) $(MPS_OBJCFLAGS) -c -o $@ $<

else
mps:
	@echo "Error: MPS backend requires Apple Silicon (arm64)"
	@exit 1
endif
else
mps:
	@echo "Error: MPS backend requires macOS"
	@exit 1
endif

# ======================================================================
# Backend: rocm (AMD GPUs with ROCm/HIP)
# ======================================================================
ifeq ($(shell command -v hipcc 2> /dev/null),)
rocm:
	@echo "Error: ROCm/hipcc not found. Install ROCm toolkit first."
	@echo "Visit: https://rocm.docs.amd.com/"
	@exit 1
else
ROCM_PATH ?= /opt/rocm
HIPCC = hipcc
GPU_ARCH := $(shell rocminfo 2>/dev/null | grep -o 'gfx[0-9]*' | head -1)
ifeq ($(GPU_ARCH),)
GPU_ARCH = native
endif

ROCM_CFLAGS = $(CFLAGS_BASE) -DUSE_BLAS -DUSE_ROCM -I$(ROCM_PATH)/include -I/usr/include/openblas
ROCM_LDFLAGS = $(LDFLAGS) -L$(ROCM_PATH)/lib -lopenblas -lhipblas -lamdhip64
HIPCC_FLAGS = -O3 -fPIC -std=c++17 --offload-arch=$(GPU_ARCH)

# ROCm object files
ROCM_HIP_OBJS = rocm/flux_rocm.o rocm/flux_rocm_kernels.o

rocm: clean rocm-build
	@echo ""
	@echo "Built with ROCm backend (AMD GPU acceleration)"
	@echo "GPU Architecture: $(GPU_ARCH)"

rocm-build: $(SRCS:.c=.rocm.o) $(CLI_SRCS:.c=.rocm.o) $(ROCM_HIP_OBJS) main.rocm.o
	$(CC) $(ROCM_CFLAGS) -o $(TARGET) $^ $(ROCM_LDFLAGS) -lstdc++

%.rocm.o: %.c flux.h flux_kernels.h
	$(CC) $(ROCM_CFLAGS) -c -o $@ $<

rocm/flux_rocm.o: rocm/flux_rocm.cpp rocm/flux_rocm.h
	$(HIPCC) $(HIPCC_FLAGS) -c -o $@ $<

rocm/flux_rocm_kernels.o: rocm/flux_rocm_kernels.hip rocm/flux_rocm.h
	$(HIPCC) $(HIPCC_FLAGS) -c -o $@ $<

endif

# ======================================================================
# Build rules
# ======================================================================
$(TARGET): $(OBJS) $(CLI_OBJS) main.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

lib: $(LIB)

$(LIB): $(OBJS)
	ar rcs $@ $^

%.o: %.c flux.h flux_kernels.h
	$(CC) $(CFLAGS) -c -o $@ $<

# Debug build
debug: CFLAGS = $(DEBUG_CFLAGS)
debug: LDFLAGS += -fsanitize=address
debug: clean $(TARGET)

# ======================================================================
# Test and utilities
# ======================================================================
test:
	@python3 run_test.py --flux-binary ./$(TARGET)

test-quick:
	@python3 run_test.py --flux-binary ./$(TARGET) --quick

pngtest:
	@echo "Running PNG compression compare test..."
	@$(CC) $(CFLAGS_BASE) -I. png_compare.c flux_image.c -lm -o /tmp/flux_png_compare
	@/tmp/flux_png_compare images/woman_with_sunglasses.png images/woman_with_sunglasses_compressed2.png
	@/tmp/flux_png_compare images/cat_uncompressed.png images/cat_compressed.png
	@rm -f /tmp/flux_png_compare
	@echo "PNG TEST PASSED"

install: $(TARGET) $(LIB)
	install -d /usr/local/bin
	install -d /usr/local/lib
	install -d /usr/local/include
	install -m 755 $(TARGET) /usr/local/bin/
	install -m 644 $(LIB) /usr/local/lib/
	install -m 644 flux.h /usr/local/include/
	install -m 644 flux_kernels.h /usr/local/include/

clean:
	rm -f $(OBJS) $(CLI_OBJS) *.mps.o flux_metal.o main.o $(TARGET) $(LIB)
	rm -f flux_shaders_source.h
	rm -f *.rocm.o rocm/*.o rocm/*.a

info:
	@echo "Platform: $(UNAME_S) $(UNAME_M)"
	@echo "Compiler: $(CC)"
	@echo ""
	@echo "Available backends for this platform:"
	@echo "  generic - Pure C (always available)"
ifeq ($(UNAME_S),Darwin)
	@echo "  blas    - Apple Accelerate"
ifeq ($(UNAME_M),arm64)
	@echo "  mps     - Metal GPU (recommended)"
endif
else
	@echo "  blas    - OpenBLAS (requires libopenblas-dev)"
endif

# ======================================================================
# Dependencies
# ======================================================================
flux.o: flux.c flux.h flux_kernels.h flux_safetensors.h flux_qwen3.h
flux_kernels.o: flux_kernels.c flux_kernels.h
flux_tokenizer.o: flux_tokenizer.c flux.h
flux_vae.o: flux_vae.c flux.h flux_kernels.h
flux_transformer.o: flux_transformer.c flux.h flux_kernels.h
flux_sample.o: flux_sample.c flux.h flux_kernels.h
flux_image.o: flux_image.c flux.h
flux_safetensors.o: flux_safetensors.c flux_safetensors.h
flux_qwen3.o: flux_qwen3.c flux_qwen3.h flux_safetensors.h
flux_qwen3_tokenizer.o: flux_qwen3_tokenizer.c flux_qwen3.h
terminals.o: terminals.c terminals.h flux.h
flux_cli.o: flux_cli.c flux_cli.h flux.h linenoise.h terminals.h
linenoise.o: linenoise.c linenoise.h
main.o: main.c flux.h flux_kernels.h flux_cli.h terminals.h
