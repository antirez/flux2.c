# JPEG Decoder Test Suite Makefile

CC = cc
CFLAGS = -Wall -Wextra -O2 -I..
CFLAGS_DEBUG = -Wall -Wextra -g -O0 -I..

TEST_BINARY = test_jpeg
TEST_SOURCE = test.c
FUZZ_BINARY = fuzz_jpeg
FUZZ_SOURCE = fuzz.c

.PHONY: all test clean debug fuzz

# Default: build and run tests
all: test

# Build test binary
$(TEST_BINARY): $(TEST_SOURCE) ../jpeg.h ../png.h
	$(CC) $(CFLAGS) -o $@ $(TEST_SOURCE)

# Run tests
test: $(TEST_BINARY)
	@./$(TEST_BINARY)

# Build with debug symbols
debug: $(TEST_SOURCE) ../jpeg.h ../png.h
	$(CC) $(CFLAGS_DEBUG) -o $(TEST_BINARY) $(TEST_SOURCE)

# Run tests with verbose output
verbose: $(TEST_BINARY)
	@./$(TEST_BINARY) -v

# Build fuzzer
$(FUZZ_BINARY): $(FUZZ_SOURCE) ../jpeg.h
	$(CC) $(CFLAGS) -o $@ $(FUZZ_SOURCE)

# Build fuzzer with AddressSanitizer
fuzz_jpeg_asan: $(FUZZ_SOURCE) ../jpeg.h
	$(CC) -Wall -Wextra -g -O1 -fsanitize=address -I.. -o $@ $(FUZZ_SOURCE)

# Run fuzzer (default 100k iterations)
# Usage: make fuzz [FUZZ_ITER=N] [FUZZ_SEED=N]
FUZZ_ITER ?= 100000
FUZZ_SEED ?=
fuzz: $(FUZZ_BINARY)
	@./$(FUZZ_BINARY) $(FUZZ_ITER) $(FUZZ_SEED)

# Run fuzzer with AddressSanitizer (slower but catches more bugs)
fuzz-asan: fuzz_jpeg_asan
	@./fuzz_jpeg_asan $(FUZZ_ITER) $(FUZZ_SEED)

# Clean build artifacts
clean:
	rm -f $(TEST_BINARY) $(FUZZ_BINARY) fuzz_jpeg_asan
	rm -rf *.dSYM
	rm -f crash_*.jpg fuzz.jpg

# Help
help:
	@echo "JPEG Decoder Test Suite"
	@echo ""
	@echo "Targets:"
	@echo "  all     - Build and run tests (default)"
	@echo "  test    - Build and run tests"
	@echo "  debug   - Build with debug symbols"
	@echo "  verbose - Run tests with verbose output"
	@echo "  fuzz    - Run mutation fuzzer (FUZZ_ITER=N, FUZZ_SEED=N)"
	@echo "  clean   - Remove build artifacts"
	@echo "  help    - Show this help"
