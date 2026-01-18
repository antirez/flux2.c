# FLUX.2 klein 4B - Pure C Inference Engine
# Makefile

CC = gcc
CFLAGS = -Wall -Wextra -O3 -march=native -ffast-math
LDFLAGS = -lm

# Platform-specific BLAS support
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    # macOS: Use Accelerate framework for BLAS
    CFLAGS += -DACCELERATE_NEW_LAPACK
    LDFLAGS += -framework Accelerate
endif
ifeq ($(UNAME_S),Linux)
    # Linux: Use OpenBLAS if available (compile with USE_OPENBLAS=1)
    ifdef USE_OPENBLAS
        CFLAGS += -DUSE_OPENBLAS
        LDFLAGS += -lopenblas
    endif
endif

# Debug build
DEBUG_CFLAGS = -Wall -Wextra -g -O0 -DDEBUG -fsanitize=address

# Source files
SRCS = flux.c flux_kernels.c flux_tokenizer.c flux_vae.c flux_transformer.c flux_sample.c flux_image.c flux_safetensors.c
OBJS = $(SRCS:.c=.o)

# Main program
MAIN = main.c
TARGET = flux

# Library
LIB = libflux.a

.PHONY: all clean debug lib install

all: $(TARGET)

$(TARGET): $(OBJS) $(MAIN:.c=.o)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

lib: $(LIB)

$(LIB): $(OBJS)
	ar rcs $@ $^

%.o: %.c flux.h flux_kernels.h
	$(CC) $(CFLAGS) -c -o $@ $<

debug: CFLAGS = $(DEBUG_CFLAGS)
debug: LDFLAGS += -fsanitize=address
debug: clean $(TARGET)

# Install to /usr/local
install: $(TARGET) $(LIB)
	install -d /usr/local/bin
	install -d /usr/local/lib
	install -d /usr/local/include
	install -m 755 $(TARGET) /usr/local/bin/
	install -m 644 $(LIB) /usr/local/lib/
	install -m 644 flux.h /usr/local/include/
	install -m 644 flux_kernels.h /usr/local/include/

clean:
	rm -f $(OBJS) $(MAIN:.c=.o) $(TARGET) $(LIB)

# Dependencies
flux.o: flux.c flux.h flux_kernels.h flux_safetensors.h
flux_kernels.o: flux_kernels.c flux_kernels.h
flux_tokenizer.o: flux_tokenizer.c flux.h
flux_vae.o: flux_vae.c flux.h flux_kernels.h
flux_transformer.o: flux_transformer.c flux.h flux_kernels.h
flux_sample.o: flux_sample.c flux.h flux_kernels.h
flux_image.o: flux_image.c flux.h
flux_safetensors.o: flux_safetensors.c flux_safetensors.h
main.o: main.c flux.h flux_kernels.h

help:
	@echo "FLUX.2 Makefile targets:"
	@echo "  all       - Build the flux executable (default)"
	@echo "  lib       - Build static library libflux.a"
	@echo "  debug     - Build with debug symbols and sanitizers"
	@echo "  install   - Install to /usr/local"
	@echo "  clean     - Remove build artifacts"
	@echo ""
	@echo "Usage:"
	@echo "  ./flux -d model/ -p \"prompt\" -o output.png"
