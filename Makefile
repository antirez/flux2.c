# FLUX.2 klein 4B - Pure C Inference Engine
# Makefile

CC = gcc
CFLAGS_BASE = -Wall -Wextra -O3 -march=native -ffast-math -fopenmp
LDFLAGS = -lm -fopenmp

# Platform detection
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

# Source files
SRCS = flux.c flux_kernels.c flux_tokenizer.c flux_vae.c flux_transformer.c flux_sample.c flux_image.c flux_safetensors.c flux_qwen3.c flux_qwen3_tokenizer.c
OBJS = $(SRCS:.c=.o)
MAIN = main.c
TARGET = flux
LIB = libflux.a

# Debug build flags
DEBUG_CFLAGS = -Wall -Wextra -g -O0 -DDEBUG -fsanitize=address

.PHONY: all clean debug lib install info test help generic blas mps

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
	@echo "  make rocm     - AMD GPU with ROCm/rocBLAS (GPU acceleration)"
endif
	@echo ""
	@echo "Other targets:"
	@echo "  make clean    - Remove build artifacts"
	@echo "  make test     - Run inference test"
	@echo "  make info     - Show build configuration"
	@echo "  make lib      - Build static library"
	@echo ""
	@echo "Example: make mps && ./flux -d flux-klein-model -p \"a cat\" -o cat.png"

# =============================================================================
# Backend: generic (pure C, no BLAS)
# =============================================================================
generic: CFLAGS = $(CFLAGS_BASE) -DGENERIC_BUILD
generic: clean $(TARGET)
	@echo ""
	@echo "Built with GENERIC backend (pure C, no BLAS)"
	@echo "This will be slow but has zero dependencies."

# =============================================================================
# Backend: blas (Accelerate on macOS, OpenBLAS on Linux)
# =============================================================================
ifeq ($(UNAME_S),Darwin)
blas: CFLAGS = $(CFLAGS_BASE) -DUSE_BLAS -DACCELERATE_NEW_LAPACK
blas: LDFLAGS += -framework Accelerate
else
blas: CFLAGS = $(CFLAGS_BASE) -DUSE_BLAS -DUSE_OPENBLAS
blas: LDFLAGS += -lopenblas
endif
blas: clean $(TARGET)
	@echo ""
	@echo "Built with BLAS backend (~30x faster than generic)"

# =============================================================================
# Backend: mps (Apple Silicon Metal GPU)
# =============================================================================
ifeq ($(UNAME_S),Darwin)
ifeq ($(UNAME_M),arm64)
MPS_CFLAGS = $(CFLAGS_BASE) -DUSE_BLAS -DUSE_METAL -DACCELERATE_NEW_LAPACK
MPS_OBJCFLAGS = $(MPS_CFLAGS) -fobjc-arc
MPS_LDFLAGS = $(LDFLAGS) -framework Accelerate -framework Metal -framework MetalPerformanceShaders -framework Foundation

mps: clean mps-build
	@echo ""
	@echo "Built with MPS backend (Metal GPU acceleration)"

mps-build: $(SRCS:.c=.mps.o) flux_metal.o main.mps.o
	$(CC) $(MPS_CFLAGS) -o $(TARGET) $^ $(MPS_LDFLAGS)

%.mps.o: %.c flux.h flux_kernels.h
	$(CC) $(MPS_CFLAGS) -c -o $@ $<

flux_metal.o: flux_metal.m flux_metal.h
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

# =============================================================================
# Backend: rocm (AMD GPUs with custom CBLAS wrapper)
# =============================================================================
ifeq ($(shell command -v hipcc 2> /dev/null),)
rocm:
	@echo "Error: ROCm/hipcc not found. Install ROCm toolkit first."
	@echo "Visit: https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html"
	@exit 1
else
ROCM_PATH ?= $(shell hipcc --version 2>/dev/null | grep -q "HIP" && echo /opt/rocm || echo "")
ifeq ($(ROCM_PATH),)
ROCM_PATH = /opt/rocm
endif

# Use custom CBLAS wrapper for rocBLAS
rocm: CFLAGS = $(CFLAGS_BASE) -DUSE_BLAS -DUSE_ROCM -D__HIP_PLATFORM_AMD__ -I$(ROCM_PATH)/include
rocm: LDFLAGS += -L$(ROCM_PATH)/lib -lrocblas -lamdhip64 -lm
rocm: clean rocblas_cblas_wrapper.o flux_kernels_rocm.o $(OBJS) main.o
	$(CC) $(CFLAGS) -o $(TARGET) rocblas_cblas_wrapper.o flux_kernels_rocm.o $(OBJS) main.o $(LDFLAGS)
	@echo ""
	@echo "Built with rocBLAS + custom CBLAS wrapper (GPU)"
	@echo "ROCm path: $(ROCM_PATH)"

rocblas_cblas_wrapper.o: rocblas_cblas_wrapper.c
	$(CC) $(CFLAGS) -c -o $@ $<

flux_kernels_rocm.o: flux_kernels_rocm.cpp
	hipcc -O3 -fPIC --offload-arch=native -c -o $@ $<
endif

# =============================================================================
# Build rules
# =============================================================================
$(TARGET): $(OBJS) main.o
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

# =============================================================================
# Test and utilities
# =============================================================================
TEST_PROMPT = "A fluffy orange cat sitting on a windowsill"
test: $(TARGET)
	@echo "Running inference test..."
	@./$(TARGET) -d flux-klein-model -p $(TEST_PROMPT) --seed 42 --steps 1 -o /tmp/flux_test_output.png -W 64 -H 64
	@python3 -c "\
import numpy as np; \
from PIL import Image; \
ref = np.array(Image.open('test_vectors/reference_1step_64x64_seed42.png')); \
test = np.array(Image.open('/tmp/flux_test_output.png')); \
diff = np.abs(ref.astype(float) - test.astype(float)); \
print(f'Max diff: {diff.max()}, Mean diff: {diff.mean():.4f}'); \
exit(0 if diff.max() < 2 else 1)"
	@rm -f /tmp/flux_test_output.png
	@echo "TEST PASSED"

install: $(TARGET) $(LIB)
	install -d /usr/local/bin
	install -d /usr/local/lib
	install -d /usr/local/include
	install -m 755 $(TARGET) /usr/local/bin/
	install -m 644 $(LIB) /usr/local/lib/
	install -m 644 flux.h /usr/local/include/
	install -m 644 flux_kernels.h /usr/local/include/

clean:
	rm -f $(OBJS) *.mps.o flux_metal.o main.o $(TARGET) $(LIB)

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

# =============================================================================
# Dependencies
# =============================================================================
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
main.o: main.c flux.h flux_kernels.h
