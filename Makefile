# FLUX.2 klein 4B - Pure C Inference Engine
# Makefile

CC = gcc
CFLAGS_BASE = -Wall -Wextra -O3 -march=native -ffast-math
LDFLAGS = -lm

# Platform detection
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

# Source files
SRCS = flux.c flux_kernels.c flux_tokenizer.c flux_vae.c flux_transformer.c flux_sample.c flux_image.c jpeg.c flux_safetensors.c flux_qwen3.c flux_qwen3_tokenizer.c terminals.c
OBJS = $(SRCS:.c=.o)
CLI_SRCS = flux_cli.c linenoise.c embcache.c
CLI_OBJS = $(CLI_SRCS:.c=.o)
MAIN = main.c
TARGET = flux
LIB = libflux.a

# Debug build flags
DEBUG_CFLAGS = -Wall -Wextra -g -O0 -DDEBUG -fsanitize=address

.PHONY: all clean debug lib install info test pngtest help generic blas mps cuda

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
else
	@echo "  make cuda     - NVIDIA GPU with CUDA/cuBLAS (fastest)"
endif
	@echo ""
	@echo "Other targets:"
	@echo "  make clean    - Remove build artifacts"
	@echo "  make test     - Run inference test"
	@echo "  make pngtest  - Compare PNG load on compressed image"
	@echo "  make info     - Show build configuration"
	@echo "  make lib      - Build static library"
	@echo ""
ifeq ($(UNAME_S),Darwin)
	@echo "Example: make mps && ./flux -d flux-klein-model -p \"a cat\" -o cat.png"
else
	@echo "Example: make cuda && ./flux -d flux-klein-model -p \"a cat\" -o cat.png"
endif

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
blas: CFLAGS = $(CFLAGS_BASE) -DUSE_BLAS -DUSE_OPENBLAS -I/usr/include/openblas
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

# =============================================================================
# Backend: cuda (NVIDIA GPU with CUDA/cuBLAS)
# =============================================================================
# CUDA Toolkit paths - adjust if needed
CUDA_PATH ?= /usr/local/cuda
NVCC = $(CUDA_PATH)/bin/nvcc

# Detect CUDA availability
CUDA_AVAILABLE := $(shell which $(NVCC) 2>/dev/null)

ifdef CUDA_AVAILABLE
CUDA_CFLAGS = $(CFLAGS_BASE) -DUSE_CUDA -DUSE_BLAS -I$(CUDA_PATH)/include
CUDA_NVCCFLAGS = -O3 -use_fast_math --compiler-options "$(CFLAGS_BASE)"
CUDA_LDFLAGS = $(LDFLAGS) -L$(CUDA_PATH)/lib64 -lcudart -lcublas -lopenblas -lstdc++

# Auto-detect GPU architecture from installed GPU, fallback to multi-arch fat binary
DETECTED_COMPUTE := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
ifneq ($(DETECTED_COMPUTE),)
    CUDA_ARCH ?= sm_$(DETECTED_COMPUTE)
    CUDA_NVCCFLAGS += -arch=$(CUDA_ARCH)
else
    # Fat binary: Turing (RTX 2080), Ampere (RTX 3090), Ada (RTX 4090), Hopper (H100), Blackwell (RTX 5090)
    CUDA_NVCCFLAGS += -gencode arch=compute_75,code=sm_75 \
                      -gencode arch=compute_86,code=sm_86 \
                      -gencode arch=compute_89,code=sm_89 \
                      -gencode arch=compute_90,code=sm_90 \
                      -gencode arch=compute_120,code=sm_120
endif

cuda: clean cuda-build
	@echo ""
	@echo "Built with CUDA backend (NVIDIA GPU acceleration)"
	@echo "Using GPU architecture: $(CUDA_ARCH)"

cuda-build: $(SRCS:.c=.cuda.o) $(CLI_SRCS:.c=.cuda.o) flux_cuda.o main.cuda.o
	$(CC) $(CUDA_CFLAGS) -o $(TARGET) $^ $(CUDA_LDFLAGS)

%.cuda.o: %.c flux.h flux_kernels.h
	$(CC) $(CUDA_CFLAGS) -c -o $@ $<

flux_cuda.o: flux_cuda.cu flux_cuda.h
	$(NVCC) $(CUDA_NVCCFLAGS) -c -o $@ $<

else
cuda:
	@echo "Error: CUDA toolkit not found"
	@echo "Please install CUDA toolkit and ensure nvcc is in PATH"
	@echo "Or set CUDA_PATH environment variable"
	@exit 1
endif

# =============================================================================
# Build rules
# =============================================================================
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

# =============================================================================
# Test and utilities
# =============================================================================
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
	rm -f $(OBJS) $(CLI_OBJS) *.mps.o *.cuda.o flux_metal.o flux_cuda.o main.o $(TARGET) $(LIB)
	rm -f flux_shaders_source.h

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
ifdef CUDA_AVAILABLE
	@echo "  cuda    - NVIDIA GPU (requires CUDA toolkit)"
endif
endif

# =============================================================================
# Dependencies
# =============================================================================
flux.o: flux.c flux.h flux_kernels.h flux_safetensors.h flux_qwen3.h
flux_kernels.o: flux_kernels.c flux_kernels.h flux_cuda.h
flux_tokenizer.o: flux_tokenizer.c flux.h
flux_vae.o: flux_vae.c flux.h flux_kernels.h
flux_transformer.o: flux_transformer.c flux.h flux_kernels.h
flux_sample.o: flux_sample.c flux.h flux_kernels.h
flux_image.o: flux_image.c flux.h
flux_safetensors.o: flux_safetensors.c flux_safetensors.h
flux_qwen3.o: flux_qwen3.c flux_qwen3.h flux_safetensors.h
flux_qwen3_tokenizer.o: flux_qwen3_tokenizer.c flux_qwen3.h
terminals.o: terminals.c terminals.h flux.h
flux_cli.o: flux_cli.c flux_cli.h flux.h flux_qwen3.h embcache.h linenoise.h terminals.h
linenoise.o: linenoise.c linenoise.h
embcache.o: embcache.c embcache.h
flux_cuda.o: flux_cuda.cu flux_cuda.h
main.o: main.c flux.h flux_kernels.h flux_cli.h terminals.h
