# FLUX.2-klein-4B Pure C Implementation

This program generates images from text prompts (and optionally from other images) using the FLUX.2-klein-4B model from Black Forest Labs. It can be used as a library as well, and is implemented entirely in C, with zero external dependencies beyond the C standard library. MPS and BLAS acceleration are optional but recommended.

## An experiment in AI code generation and open source software

I (the human here, Salvatore) wanted to test code generation with a more ambitious task, over the weekend. This is the result. It is my first open source project where I wrote zero lines of code. I believe that inference systems not using the Python stack (which I do not appreciate) are a way to free open models usage and make AI more accessible. There is already a project [doing the inference of diffusion models in C / C++](https://github.com/leejet/stable-diffusion.cpp) that supports multiple models, and is based on GGML. I wanted to see if, with the assistance of modern AI, I could reproduce this work in a more concise way, from scratch, in a weekend. Looks like it is possible.

This code base was written with Claude Code, using the Claude Max plan, the small one of ~80 euros per month. I almost reached the limits but this plan was definitely sufficient for such a large task, which was surprising. In order to simplify the usage of this software, no quantization is used, nor do you need to convert the model. It runs directly with the safetensors model as input, using floats.

Even if the code was generated using AI, my help in steering towards the right design, implementation choices, and correctness has been vital during the development. I learned quite a few things about working with non trivial projects and AI.

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

That's it. No Python runtime, no PyTorch, no CUDA toolkit required at inference time.

## Example Output

![Woman with sunglasses](images/woman_with_sunglasses.png)

*Generated with: `./flux -d flux-klein-model -p "A picture of a woman in 1960 America. Sunglasses. ASA 400 film. Black and White." -W 250 -H 250 -o /tmp/woman.png`, and later processed with image to image generation via `./flux -d flux-klein-model -i /tmp/woman.png -o /tmp/woman2.png -p "oil painting of woman with sunglasses" -v -H 256 -W 256`*

## Features

- **Zero dependencies**: Pure C implementation, works standalone. BLAS optional for ~30x speedup (Apple Accelerate on macOS, OpenBLAS on Linux)
- **Metal GPU acceleration**: Automatic on Apple Silicon Macs
- **Text-to-image**: Generate images from text prompts
- **Image-to-image**: Transform existing images guided by prompts
- **Integrated text encoder**: Qwen3-4B encoder built-in, no external embedding computation needed
- **Memory efficient**: Automatic encoder release after encoding (~8GB freed)

## Usage

### Text-to-Image

```bash
./flux -d flux-klein-model -p "A fluffy orange cat sitting on a windowsill" -o cat.png
```

### Image-to-Image

Transform an existing image based on a prompt:

```bash
./flux -d flux-klein-model -p "oil painting style" -i photo.png -o painting.png -t 0.7
```

The `-t` (strength) parameter controls how much the image changes:
- `0.0` = no change (output equals input)
- `1.0` = full generation (input only provides composition hint)
- `0.7` = good balance for style transfer

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
-S, --seed N          Random seed for reproducibility
```

**Image-to-image options:**
```
-i, --input PATH      Input image for img2img
-t, --strength N      How much to change the image, 0.0-1.0 (default: 0.75)
```

**Output options:**
```
-q, --quiet           Silent mode, no output
-v, --verbose         Show detailed config and timing info
```

**Other options:**
```
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
make test       # Run reference image test
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

### How Fast Is It?

Benchmarks on **Apple M3 Max** (128GB RAM), generating a 4-step image:

| Size | C (MPS) | C (BLAS) | C (Generic) | PyTorch (MPS) |
|------|---------|----------|-------------|---------------|
| 512x512 | 49.6s | 51.9s | - | 5.4s |
| 256x256 | 32.4s | 29.7s | - | 3.0s |
| 64x64 | 25.0s | 23.5s | 605.6s | 2.2s |

**Notes:**
- The C implementation uses float32 throughout, while PyTorch uses bfloat16 with highly optimized MPS kernels. The next step of this project is likely to implement such an optimization, in order to reach similar speed, or at least try to approach it.
- The generic (pure C) backend is extremely slow and only practical for testing at small sizes.
- Times include text encoding, denoising (4 steps), and VAE decode.

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

Transform an existing image guided by a text prompt. The `strength` parameter controls how much the image changes:

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
    params.strength = 0.7;  /* 0.0 = no change, 1.0 = full regeneration */
    params.seed = 123;

    /* Transform the image */
    flux_image *painting = flux_img2img(ctx, "oil painting, impressionist style",
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
```

**Strength values:**
- `0.3` - Subtle style transfer, preserves most details
- `0.5` - Moderate transformation
- `0.7` - Strong transformation, good for style transfer
- `0.9` - Almost complete regeneration, keeps only composition

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

## License

MIT
