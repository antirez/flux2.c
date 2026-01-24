# ROCm Backend Build
# 
# Usage:
#   make test        - Build and run all tests
#   make test-basic  - Build and run infrastructure test only
#   make test-kernel - Build and run kernel test
#   make clean       - Clean build artifacts

HIPCC = hipcc
CXXFLAGS = -O3 -fPIC -std=c++17
LDFLAGS = -lhipblas

# Auto-detect GPU architecture
GPU_ARCH := $(shell rocminfo 2>/dev/null | grep -o 'gfx[0-9]*' | head -1)
ifeq ($(GPU_ARCH),)
GPU_ARCH = native
endif
HIPCC_FLAGS = --offload-arch=$(GPU_ARCH)

.PHONY: all test test-basic test-kernel clean info

all: test_rocm test_kernels

info:
	@echo "GPU Architecture: $(GPU_ARCH)"
	@echo "HIPCC: $(HIPCC)"
	@hipcc --version

# Infrastructure test (BLAS only)
test_rocm: test_rocm.cpp flux_rocm.cpp flux_rocm.h
	$(HIPCC) $(CXXFLAGS) $(HIPCC_FLAGS) -o $@ test_rocm.cpp flux_rocm.cpp $(LDFLAGS)

# Kernel test (includes custom kernels)
test_kernels: test_kernels.cpp flux_rocm.cpp flux_rocm_kernels.hip flux_rocm.h
	$(HIPCC) $(CXXFLAGS) $(HIPCC_FLAGS) -o $@ test_kernels.cpp flux_rocm.cpp flux_rocm_kernels.hip $(LDFLAGS)

test: test_rocm test_kernels
	@echo "=== Running infrastructure test ==="
	./test_rocm
	@echo ""
	@echo "=== Running kernel test ==="
	./test_kernels

test-basic: test_rocm
	./test_rocm

test-kernel: test_kernels
	./test_kernels

# Build library objects (for integration)
flux_rocm.o: flux_rocm.cpp flux_rocm.h
	$(HIPCC) $(CXXFLAGS) $(HIPCC_FLAGS) -c -o $@ flux_rocm.cpp

flux_rocm_kernels.o: flux_rocm_kernels.hip flux_rocm.h
	$(HIPCC) $(CXXFLAGS) $(HIPCC_FLAGS) -c -o $@ flux_rocm_kernels.hip

flux_rocm_compat.o: flux_rocm_compat.cpp flux_rocm_compat.h flux_rocm.h
	$(HIPCC) $(CXXFLAGS) $(HIPCC_FLAGS) -c -o $@ flux_rocm_compat.cpp

# Full library for integration
lib: flux_rocm.o flux_rocm_kernels.o flux_rocm_compat.o
	ar rcs libflux_rocm.a flux_rocm.o flux_rocm_kernels.o flux_rocm_compat.o
	@echo "Built libflux_rocm.a"

clean:
	rm -f test_rocm test_kernels *.o *.a
