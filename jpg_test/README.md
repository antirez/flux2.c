# JPEG Decoder Test Suite

Test suite for the standalone `jpeg.h` single-header JPEG decoder library.

## Test Images

Test images are from various sources with permissive licenses:

| File | Format | Dimensions | Source |
|------|--------|------------|--------|
| testorig.jpg | Baseline 4:2:2 | 227x149 | libjpeg-turbo (BSD) |
| testimgint.jpg | Baseline 4:2:2 | 227x149 | libjpeg-turbo (BSD) |
| testorig_444.jpg | Baseline 4:4:4 | 227x149 | Generated |
| testorig_420.jpg | Baseline 4:2:0 | 227x149 | Generated |
| testorig_prog.jpg | Progressive | 227x149 | Generated |
| sample.jpg | Baseline | 290x442 | libvips (LGPL) |
| hopper.jpg | Baseline | 128x128 | Pillow (HPND) |
| wizard.jpg | Baseline | 264x351 | ImageMagick (Apache) |
| cd1.1.jpg | Grayscale | 531x373 | libvips (LGPL) |
| cd2.1.jpg | Grayscale | 531x373 | libvips (LGPL) |
| monkey_prog.jpg | Progressive | 149x227 | Generated from libjpeg-turbo |
| gray_prog.jpg | Progressive Grayscale | 227x149 | Generated |
| testorig.ppm | PPM (RGB) | 227x149 | libjpeg-turbo (BSD) |
| testorig.pgm | PGM (Grayscale) | 227x149 | libjpeg-turbo (BSD) |
| monkey16.ppm | PPM (RGB) | 149x227 | libjpeg-turbo (BSD) |

## Running Tests

```bash
make            # Build and run tests
make clean      # Remove build artifacts
make debug      # Build with debug symbols
make fuzz       # Run mutation fuzzer (FUZZ_ITER=N, FUZZ_SEED=N)
make fuzz-asan  # Run fuzzer with AddressSanitizer
```

## Test Coverage

1. **Baseline JPEG Loading** - RGB images with various subsampling (4:4:4, 4:2:2, 4:2:0)
2. **Grayscale JPEG Loading** - Single-channel images
3. **Progressive JPEG Loading** - Multi-scan progressive images
4. **Reference Comparison** - Compare decoded pixels against original PPM source
5. **Memory Management** - Load/free stress test, clone functionality
6. **API Tests** - `jpeg_load_mem()`, PNG roundtrip
7. **Error Handling** - Invalid files, truncated data, wrong magic bytes

## Fuzzer

The mutation-based fuzzer (`fuzz.c`) tests decoder robustness by:
- Loading valid JPEG images
- Applying random mutations (bit flips, byte changes, insertions, deletions, truncation)
- Attempting to decode mutated data
- Using `fork()` to isolate crashes and continue testing

Usage:
```bash
make fuzz FUZZ_ITER=100000           # Run 100k iterations
make fuzz FUZZ_ITER=10000 FUZZ_SEED=42  # Reproducible run with fixed seed
make fuzz-asan FUZZ_ITER=10000       # With AddressSanitizer
```

## License

Test images are from projects with permissive licenses:
- libjpeg-turbo: BSD-style (IJG + Modified BSD)
- libvips: LGPL-2.1
- Pillow: HPND (Historical Permission Notice and Disclaimer)
- ImageMagick: Apache-2.0

See respective project repositories for full license terms.
