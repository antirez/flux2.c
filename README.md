# FLUX.2-klein-4B Pure C Implementation

This program generates images from text prompts (and optionally from other images) using the FLUX.2-klein-4B model from Black Forest Labs. It can be used as a library as well, and is implemented entirely in C, with zero external dependencies beyond the C standard library. MPS and BLAS acceleration are optional but recommended.

## Quick Start

```bash
# Build (choose your backend)
make mps       # Apple Silicon (fastest)
# or: make blas    # Intel Mac / Linux with OpenBLAS
# or: make generic # Pure C, no dependencies

# Download the model (~16GB)
pip install huggingface_hub
python download_model.py

# Generate an image
./flux -d flux-klein-model -p "A woman wearing sunglasses" -o output.png
```

That's it. No Python runtime or CUDA toolkit required at inference time.

## Example Output

![Woman with sunglasses](images/woman_with_sunglasses.png)

*Generated with: `./flux -d flux-klein-model -p "A picture of a woman in 1960 America. Sunglasses. ASA 400 film. Black and White." -W 512 -H 512 -o woman.png`*

### Image-to-Image Example

![antirez to drawing](images/antirez_to_drawing.png)

*Generated with: `./flux -i antirez.png -o antirez_to_drawing.png -p "make it a drawing" -d flux-klein-model`*

## Features

- **Zero dependencies**: Pure C implementation, works standalone. BLAS optional for ~30x speedup (Apple Accelerate on macOS, OpenBLAS on Linux)
- **Metal GPU acceleration**: Automatic on Apple Silicon Macs
- **Text-to-image**: Generate images from text prompts
- **Image-to-image**: Transform existing images guided by prompts
- **Integrated text encoder**: Qwen3-4B encoder built-in, no external embedding computation needed
- **Memory efficient**: Automatic encoder release after encoding (~8GB freed)
- **Low memory mode**: `--mmap` flag enables on-demand weight loading, reducing peak memory from ~16GB to ~4-5GB for 16GB RAM systems

## Usage

### Text-to-Image

```bash
./flux -d flux-klein-model -p "A fluffy orange cat sitting on a windowsill" -o cat.png
```

### Image-to-Image

Transform an existing image based on a prompt:

```bash
./flux -d flux-klein-model -p "oil painting style" -i photo.png -o painting.png
```

FLUX.2 uses **in-context conditioning** for image-to-image generation. Unlike traditional approaches that add noise to the input image, FLUX.2 passes the reference image as additional tokens that the model can attend to during generation. This means:

- The model "sees" your input image and uses it as a reference
- The prompt describes what you want the output to look like
- Results tend to preserve the composition while applying the described transformation

**Tips for good results:**
- Use descriptive prompts that describe the desired output, not instructions
- Good: `"oil painting of a woman with sunglasses, impressionist style"`
- Less good: `"make it an oil painting"` (instructional prompts may work less well)

### Command Line Options

**Required:**
```
-d, --dir PATH        Path to model directory
-p, --prompt TEXT     Text prompt for generation
-o, --output PATH     Output image path (.png or .ppm)
```

**Generation options:**
```
-W, --width N         Output width in pixels (default: 256)
-H, --height N        Output height in pixels (default: 256)
-s, --steps N         Sampling steps (default: 4)
-g, --guidance N      Guidance scale (default: 1.0)
-S, --seed N          Random seed for reproducibility
```

**Image-to-image options:**
```
-i, --input PATH      Input image for img2img
```

**Output options:**
```
-q, --quiet           Silent mode, no output
-v, --verbose         Show detailed config and timing info
```

**Other options:**
```
-m, --mmap            Low memory mode (load weights on-demand, slower)
-e, --embeddings PATH Load pre-computed text embeddings (advanced)
-h, --help            Show help
```

### Reproducibility

The seed is always printed to stderr, even when random:
```
$ ./flux -d flux-klein-model -p "a landscape" -o out.png
Seed: 1705612345
out.png
```

To reproduce the same image, use the printed seed:
```
$ ./flux -d flux-klein-model -p "a landscape" -o out.png -S 1705612345
```

### PNG Metadata

Generated PNG images include metadata with the seed and model information, so you can always recover the seed even if you didn't save the terminal output:

```bash
# Using exiftool
exiftool image.png | grep flux

# Using Python/PIL
python3 -c "from PIL import Image; print(Image.open('image.png').info)"

# Using ImageMagick
identify -verbose image.png | grep -A1 "Properties:"
```

The following metadata fields are stored:
- `flux:seed` - The random seed used for generation
- `flux:model` - The model name (FLUX.2-klein-4B)
- `Software` - Program identifier

## Building

Choose a backend when building:

```bash
make            # Show available backends
make generic    # Pure C, no dependencies (slow)
make blas       # BLAS acceleration (~30x faster)
make mps        # Apple Silicon Metal GPU (fastest, macOS only)
```

**Recommended:**
- macOS Apple Silicon: `make mps`
- macOS Intel: `make blas`
- Linux with OpenBLAS: `make blas`
- Linux without OpenBLAS: `make generic`

For `make blas` on Linux, install OpenBLAS first:
```bash
# Ubuntu/Debian
sudo apt install libopenblas-dev

# Fedora
sudo dnf install openblas-devel
```

Other targets:
```bash
make clean      # Clean build artifacts
make info       # Show available backends for this platform
```

## Testing

Run the test suite to verify your build produces correct output:

```bash
make test        # Run all tests (64x64 and 512x512)
make test-quick  # Run only the quick 64x64 test
```

The tests compare generated images against reference images in `test_vectors/`. A test passes if the maximum pixel difference is ≤2 (to allow for minor floating-point variations across platforms).

**Test cases:**
| Test | Size | Steps | Purpose |
|------|------|-------|---------|
| Quick | 64×64 | 2 | Fast sanity check (~20s) |
| Full | 512×512 | 4 | Validates larger resolutions (~40s) |

You can also run the test script directly for more options:
```bash
python3 run_test.py --help
python3 run_test.py --quick          # Quick test only
python3 run_test.py --flux-binary ./flux --model-dir /path/to/model
```

## Model Download

The model weights are downloaded from HuggingFace:

```bash
pip install huggingface_hub
python download_model.py
```

This downloads approximately 16GB to `./flux-klein-model`:
- VAE (~300MB)
- Transformer (~4GB)
- Qwen3-4B Text Encoder (~8GB)
- Tokenizer

## Technical Details

### Model Architecture

**FLUX.2-klein-4B** is a rectified flow transformer optimized for fast inference:

| Component | Architecture |
|-----------|-------------|
| Transformer | 5 double blocks + 20 single blocks, 3072 hidden dim, 24 attention heads |
| VAE | AutoencoderKL, 128 latent channels, 8x spatial compression |
| Text Encoder | Qwen3-4B, 36 layers, 2560 hidden dim |

**Inference steps**: This is a distilled model that produces good results with exactly 4 sampling steps.

### Memory Requirements

| Phase | Memory |
|-------|--------|
| Text encoding | ~8GB (encoder weights) |
| Diffusion | ~8GB (transformer ~4GB + VAE ~300MB + activations) |
| Peak | ~16GB (if encoder not released) |

The text encoder is automatically released after encoding, reducing peak memory during diffusion. If you generate multiple images with different prompts, the encoder reloads automatically.

### Low Memory Inference

For systems with limited RAM (16GB or less), the `--mmap` flag enables memory-mapped weight loading:

```bash
./flux -d flux-klein-model -p "A cat" -o cat.png --mmap
```

**How it works:** Instead of loading all model weights into RAM upfront, `--mmap` keeps the safetensors files memory-mapped and loads weights on-demand:

- **Text encoder (Qwen3):** Each of the 36 transformer layers (~400MB each) is loaded, processed, and immediately freed. Only ~2GB stays resident instead of ~8GB.
- **Denoising transformer:** Each of the 5 double-blocks (~300MB) and 20 single-blocks (~150MB) is loaded on-demand and freed after use. Only ~200MB of shared weights stays resident instead of ~4GB.

This reduces peak memory from ~16GB to ~4-5GB, making inference possible on systems with only 16GB of RAM (tested on Linux).

**Backend compatibility:**
- `make generic` - Works
- `make blas` - Works
- `make mps` - Works, but less beneficial since MPS already uses bf16 weights on GPU (no expansion to float32), so memory pressure is lower

**Trade-off:** Inference is slower with `--mmap` due to repeated disk I/O and weight conversion. Use it only when you don't have enough RAM for normal operation.

### How Fast Is It?

Benchmarks on **Apple M3 Max** (128GB RAM), generating a 4-step image.

At this point the MPS inference of this project is rougly as fast as the PyTorch higly optimized implementation for MPS using the official Flux.2 pipeline. The difference in time is mostly due to the fact that the weights loading code is not as optimized as it could.

**Important:** Previous benchmarks in this README were misleading - they compared C timings (which included model loading) against reference timings (which excluded loading and used warmup runs). The table below shows fair "cold start" benchmarks where all implementations include model loading time and no warmup:

| Size | C (MPS) | C (BLAS) | PyTorch (MPS) |
|------|---------|----------|---------------|
| 256x256 | 23s | 16s | 11s |
| 512x512 | 18s | 46s | 13s |
| 1024x1024 | 32s | 173s | 25s |

**Notes:**
- All times measured as wall clock, including model loading, no warmup. The Python script is measured NOT including the huge time it takes to import the libraries, to be as fair as possible.
- The C MPS implementation uses bf16 weights on GPU with optimized batch processing. The C BLAS implementation uses float32 throughout. The C BLAS optimization can be faster with `--mmap` since it avoids the costly loading process.
- For batch generation (multiple images), only the denoising time matters after the first image. The MPS backend is competitive with the PyTorch engine.
- The `make generic` backend (pure C, no BLAS) is approximately 30x slower than BLAS and not included in benchmarks. It can be used just for fun if you don't have any other way to run it.

### Resolution Limits

**Maximum resolution**: 1024x1024 pixels. Higher resolutions require prohibitive memory for the attention mechanisms.

**Minimum resolution**: 64x64 pixels.

Dimensions should be multiples of 16 (the VAE downsampling factor).

## C Library API

The library can be integrated into your own C/C++ projects. Link against `libflux.a` and include `flux.h`.

### Text-to-Image Generation

Here's a complete program that generates an image from a text prompt:

```c
#include "flux.h"
#include <stdio.h>

int main(void) {
    /* Load the model. This loads VAE, transformer, and text encoder. */
    flux_ctx *ctx = flux_load_dir("flux-klein-model");
    if (!ctx) {
        fprintf(stderr, "Failed to load model: %s\n", flux_get_error());
        return 1;
    }

    /* Configure generation parameters. Start with defaults and customize. */
    flux_params params = FLUX_PARAMS_DEFAULT;
    params.width = 512;
    params.height = 512;
    params.seed = 42;  /* Use -1 for random seed */

    /* Generate the image. This handles text encoding, diffusion, and VAE decode. */
    flux_image *img = flux_generate(ctx, "A fluffy orange cat in a sunbeam", &params);
    if (!img) {
        fprintf(stderr, "Generation failed: %s\n", flux_get_error());
        flux_free(ctx);
        return 1;
    }

    /* Save to file. Format is determined by extension (.png or .ppm). */
    flux_image_save(img, "cat.png");
    printf("Saved cat.png (%dx%d)\n", img->width, img->height);

    /* Clean up */
    flux_image_free(img);
    flux_free(ctx);
    return 0;
}
```

Compile with:
```bash
gcc -o myapp myapp.c -L. -lflux -lm -framework Accelerate  # macOS
gcc -o myapp myapp.c -L. -lflux -lm -lopenblas              # Linux
```

### Image-to-Image Transformation

Transform an existing image guided by a text prompt using in-context conditioning:

```c
#include "flux.h"
#include <stdio.h>

int main(void) {
    flux_ctx *ctx = flux_load_dir("flux-klein-model");
    if (!ctx) return 1;

    /* Load the input image */
    flux_image *photo = flux_image_load("photo.png");
    if (!photo) {
        fprintf(stderr, "Failed to load image\n");
        flux_free(ctx);
        return 1;
    }

    /* Set up parameters. Output size defaults to input size. */
    flux_params params = FLUX_PARAMS_DEFAULT;
    params.seed = 123;

    /* Transform the image - describe the desired output */
    flux_image *painting = flux_img2img(ctx, "oil painting of the scene, impressionist style",
                                         photo, &params);
    flux_image_free(photo);  /* Done with input */

    if (!painting) {
        fprintf(stderr, "Transformation failed: %s\n", flux_get_error());
        flux_free(ctx);
        return 1;
    }

    flux_image_save(painting, "painting.png");
    printf("Saved painting.png\n");

    flux_image_free(painting);
    flux_free(ctx);
    return 0;
}

### Generating Multiple Images

When generating multiple images with different seeds but the same prompt, you can avoid reloading the text encoder:

```c
flux_ctx *ctx = flux_load_dir("flux-klein-model");
flux_params params = FLUX_PARAMS_DEFAULT;
params.width = 256;
params.height = 256;

/* Generate 5 variations with different seeds */
for (int i = 0; i < 5; i++) {
    flux_set_seed(1000 + i);

    flux_image *img = flux_generate(ctx, "A mountain landscape at sunset", &params);

    char filename[64];
    snprintf(filename, sizeof(filename), "landscape_%d.png", i);
    flux_image_save(img, filename);
    flux_image_free(img);
}

flux_free(ctx);
```

Note: The text encoder (~8GB) is automatically released after the first generation to save memory. It reloads automatically if you use a different prompt.

### Error Handling

All functions that can fail return NULL on error. Use `flux_get_error()` to get a description:

```c
flux_ctx *ctx = flux_load_dir("nonexistent-model");
if (!ctx) {
    fprintf(stderr, "Error: %s\n", flux_get_error());
    /* Prints something like: "Failed to load VAE - cannot generate images" */
    return 1;
}
```

### API Reference

**Core functions:**
```c
flux_ctx *flux_load_dir(const char *model_dir);   /* Load model, returns NULL on error */
void flux_free(flux_ctx *ctx);                     /* Free all resources */

flux_image *flux_generate(flux_ctx *ctx, const char *prompt, const flux_params *params);
flux_image *flux_img2img(flux_ctx *ctx, const char *prompt, const flux_image *input,
                          const flux_params *params);
```

**Image handling:**
```c
flux_image *flux_image_load(const char *path);     /* Load PNG or PPM */
int flux_image_save(const flux_image *img, const char *path);  /* 0=success, -1=error */
int flux_image_save_with_seed(const flux_image *img, const char *path, int64_t seed);  /* Save with metadata */
flux_image *flux_image_resize(const flux_image *img, int new_w, int new_h);
void flux_image_free(flux_image *img);
```

**Utilities:**
```c
void flux_set_seed(int64_t seed);                  /* Set RNG seed for reproducibility */
const char *flux_get_error(void);                  /* Get last error message */
void flux_release_text_encoder(flux_ctx *ctx);     /* Manually free ~8GB (optional) */
```

### Parameters

```c
typedef struct {
    int width;              /* Output width in pixels (default: 256) */
    int height;             /* Output height in pixels (default: 256) */
    int num_steps;          /* Denoising steps, use 4 for klein (default: 4) */
    float guidance_scale;   /* CFG scale, use 1.0 for klein (default: 1.0) */
    int64_t seed;           /* Random seed, -1 for random (default: -1) */
    float strength;         /* img2img only: 0.0-1.0 (default: 0.75) */
} flux_params;

/* Initialize with sensible defaults */
#define FLUX_PARAMS_DEFAULT { 256, 256, 4, 1.0f, -1, 0.75f }
```

## Debugging

### Comparing with Python Reference

When debugging img2img issues, the `--debug-py` flag allows you to run the C implementation with exact inputs saved from a Python reference script. This isolates whether differences are due to input preparation (VAE encoding, text encoding, noise generation) or the transformer itself.

**Setup:**

1. Set up the Python environment:
```bash
python -m venv flux_env
source flux_env/bin/activate
pip install torch diffusers transformers safetensors einops huggingface_hub
```

2. Clone the flux2 reference (for the model class):
```bash
git clone https://github.com/black-forest-labs/flux flux2
```

3. Run the Python debug script to save inputs:
```bash
python debug/debug_img2img_compare.py
```

This saves to `/tmp/`:
- `py_noise.bin` - Initial noise tensor
- `py_ref_latent.bin` - VAE-encoded reference image
- `py_text_emb.bin` - Text embeddings from Qwen3

4. Run C with the same inputs:
```bash
./flux -d flux-klein-model --debug-py -W 256 -H 256 --steps 4 -o /tmp/c_debug.png
```

5. Compare the outputs visually or numerically.

**What this helps diagnose:**
- If C and Python produce identical outputs with identical inputs, any differences in normal operation are due to input preparation (VAE, text encoder, RNG)
- If outputs differ even with identical inputs, the issue is in the transformer or sampling implementation

### Debug Scripts

The `debug/` directory contains Python scripts for comparing C and Python implementations:

- `debug_img2img_compare.py` - Full img2img comparison with step-by-step statistics
- `debug_rope_img2img.py` - Verify RoPE position encoding matches between C and Python

## License

MIT
