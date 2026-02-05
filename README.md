# FLUX.2-klein-4B Pure C Implementation

This program generates images from text prompts (and optionally from other images) using the [FLUX.2-klein-4B model](https://bfl.ai/models/flux-2-klein) from [Black Forest Labs](https://bfl.ai/). It can be used as a library as well, and is implemented entirely in C, with zero external dependencies beyond the C standard library. MPS and BLAS acceleration are optional but recommended.

Supported models:

- Flux.2 4B Klein distilled model (4 steps, auto guidance set to 1, very fast).
- Flux.2 4B Klein base model. (50 steps for max quality, or less. Classifier-Free Diffusion Guidance, much slower but more generation variety).

## Quick Start

```bash
# Build (choose your backend)
make mps       # Apple Silicon (fastest)
# or: make blas    # Intel Mac / Linux with OpenBLAS
# or: make generic # Pure C, no dependencies

# Download the model (~16GB) - pick one:
./download_model.sh                      # using curl
# or: pip install huggingface_hub && python download_model.py

# Generate an image
./flux -d flux-klein-model -p "A woman wearing sunglasses" -o output.png
```

If you want to try the base model, instead of the distilled one (much slower, higher quality), use the following instructions. Use 10 steps if your computer is quite slow, instead of the default of 50, it will still work well enough to test it (10 seconds to generate a 256x256 image on a MacBook M3 Max).
```
./download_model.sh --base
# or: pip install huggingface_hub && python download_model.py --base
./flux -d flux-klein-base-model -p "A woman wearing sunglasses" -o output.png
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
- **Metal GPU acceleration**: Automatic on Apple Silicon Macs. Performance matches PyTorch's optimized MPS pipeline
- **Runs where Python can't**: Memory-mapped weights (default) enable inference on 8GB RAM systems where the Python ML stack cannot run FLUX.2 at all
- **Text-to-image**: Generate images from text prompts
- **Image-to-image**: Transform existing images guided by prompts
- **Multi-reference**: Combine multiple reference images (e.g., `-i car.png -i beach.png` for "car on beach")
- **Integrated text encoder**: Qwen3-4B encoder built-in, no external embedding computation needed
- **Memory efficient**: Automatic encoder release after encoding (~8GB freed)
- **Memory-mapped weights**: Enabled by default. Reduces peak memory from ~16GB to ~4-5GB. Fastest mode on MPS; BLAS users with plenty of RAM may prefer `--no-mmap` for faster inference
- **Size-independent seeds**: Same seed produces similar compositions at different resolutions. Explore at 256×256, then render at 512×512 with the same seed
- **Terminal image display**: watch the resulting image without leaving your terminal (Ghostty, Kitty, iTerm2, or Konsole).

### Terminal Image Display

![Kitty protocol example](images/kitty-example.png)

Display generated images directly in your terminal with `--show`, or watch the denoising process step-by-step with `--show-steps`:

```bash
# Display final image in terminal (auto-detects Kitty/Ghostty/iTerm2/Konsole)
./flux -d flux-klein-model -p "a cute robot" -o robot.png --show

# Display each denoising step (slower, but interesting to watch)
./flux -d flux-klein-model -p "a cute robot" -o robot.png --show-steps
```

Requires a terminal supporting the [Kitty graphics protocol](https://sw.kovidgoyal.net/kitty/graphics-protocol/) (such as [Kitty](https://sw.kovidgoyal.net/kitty/) or [Ghostty](https://ghostty.org/)), [iTerm2](https://iterm2.com/), or [Konsole](https://konsole.kde.org/). Terminal type is auto-detected from environment variables.

Use `--zoom N` to adjust the display size (default: 2 for Retina displays, use 1 for non-HiDPI screens).

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

**Super Resolution:** Since the reference image can be a different size than the output, you can use img2img for upscaling:

```bash
./flux -d flux-klein-model -i small.png -W 1024 -H 1024 -o big.png -p "Create an exact copy of the input image."
```

The model will generate a higher-resolution version while preserving the composition and details of the input.

### Multi-Reference Generation

Combine elements from multiple reference images:

```bash
./flux -d flux-klein-model -i car.png -i beach.png -p "a sports car on the beach" -o result.png
```

Each reference image is encoded separately and passed to the transformer with different positional embeddings (T=10, T=20, T=30, ...). The model attends to all references during generation, allowing it to combine elements from each.

**Example:**
- Reference 1: A red sports car
- Reference 2: A tropical beach with palm trees
- Prompt: "combine the two images"
- Result: A red sports car on a tropical beach

You can specify up to 16 reference images with multiple `-i` flags. The prompt guides how the references are combined.

### Interactive CLI Mode

Start without `-p` to enter interactive mode:

```bash
./flux -d flux-klein-model
```

Generate images by typing prompts. Each image gets a `$N` reference ID:

```
flux> a red sports car
Done -> /tmp/flux-.../image-0001.png (ref $0)

flux> a tropical beach
Done -> /tmp/flux-.../image-0002.png (ref $1)

flux> $0 $1 combine them
Generating 256x256 (multi-ref, 2 images)...
Done -> /tmp/flux-.../image-0003.png (ref $2)
```

**Prompt syntax:**
- `prompt` - text-to-image
- `512x512 prompt` - set size inline
- `$ prompt` - img2img with last image
- `$N prompt` - img2img with reference $N
- `$0 $3 prompt` - multi-reference (combine images)

**Commands:** `!help`, `!save`, `!load`, `!seed`, `!size`, `!steps`, `!guidance`, `!linear`, `!power`, `!explore`, `!show`, `!quit`

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
-s, --steps N         Sampling steps (default: auto, 4 distilled / 50 base)
-S, --seed N          Random seed for reproducibility
-g, --guidance N      CFG guidance scale (default: auto, 1.0 distilled / 4.0 base)
    --linear          Use linear timestep schedule (see below)
    --power           Use power curve timestep schedule (see below)
    --power-alpha N   Set power schedule exponent (default: 2.0)
    --base            Force base model mode (undistilled, CFG enabled)
```

**Image-to-image options:**
```
-i, --input PATH      Reference image (can be specified multiple times)
```

**Output options:**
```
-q, --quiet           Silent mode, no output
-v, --verbose         Show detailed config and timing info
    --show            Display image in terminal (auto-detects Kitty/Ghostty/iTerm2/Konsole)
    --show-steps      Display each denoising step (slower)
    --zoom N          Terminal image zoom factor (default: 2 for Retina)
```

**Other options:**
```
-m, --mmap            Memory-mapped weights (default, fastest on MPS)
    --no-mmap         Disable mmap, load all weights upfront
-e, --embeddings PATH Load pre-computed text embeddings (advanced)
-h, --help            Show help
```

## Reproducibility

The seed is always printed to stderr, even when random:
```
$ ./flux -d flux-klein-model -p "a landscape" -o out.png
Seed: 1705612345
...
Saving... out.png 256x256 (0.1s)
```

To reproduce the same image, use the printed seed:
```
$ ./flux -d flux-klein-model -p "a landscape" -o out.png -S 1705612345
```

## PNG Metadata

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
make test        # Run all 3 tests
make test-quick  # Run only the quick 64x64 test
```

The tests compare generated images against reference images in `test_vectors/`. A test passes if the maximum pixel difference is within tolerance (to allow for minor floating-point variations across platforms).

**Test cases:**
| Test | Size | Steps | Purpose |
|------|------|-------|---------|
| Quick | 64×64 | 2 | Fast txt2img sanity check |
| Full | 512×512 | 4 | Validates txt2img at larger resolution |
| img2img | 256×256 | 4 | Validates image-to-image transformation |

You can also run the test script directly for more options:
```bash
python3 run_test.py --help
python3 run_test.py --quick          # Quick test only
python3 run_test.py --flux-binary ./flux --model-dir /path/to/model
```

## Model Download

Download model weights from HuggingFace using one of these methods:

**Distilled model** (~16GB, fast 4-step inference):
```bash
./download_model.sh                      # using curl
# or: python download_model.py           # using huggingface_hub
```

**Base model** (~16GB, 50-step inference with CFG, higher quality):
```bash
./download_model.sh --base
# or: python download_model.py --base
```

The distilled model downloads to `./flux-klein-model`, the base model to `./flux-klein-base-model`. Both contain:
- VAE (~300MB)
- Transformer (~4GB)
- Qwen3-4B Text Encoder (~8GB)
- Tokenizer

## How Fast Is It?

Benchmarks on **Apple M3 Max** (128GB RAM), distilled model (4 steps).

The MPS implementation matches the PyTorch optimized pipeline performance, providing better speed for small image sizes.

| Size | C (MPS) | PyTorch (MPS) |
|------|---------|---------------|
| 256x256 | 5.6s | 11s |
| 512x512 | 9.1s | 13s |
| 1024x1024 | 26s | 25s |

**Notes:**
- All times measured as wall clock, including model loading, no warmup. PyTorch times exclude library import overhead (~5-10s) to be fair.
- The base model is roughly 25x slower (50 steps × 2 passes per step vs 4 steps × 1 pass). It actually produces acceptable results even with 10 steps, so you can tune quality/time. The 25x figure is not exactly accurate because it only covers the denoising steps: text encoding and VAE use the same time for both the models, however such steps are a minor percentage of the generation time.
- The C BLAS backend (CPU) is not shown.
- The `make generic` backend (pure C, no BLAS) is approximately 30x slower than BLAS and not included in benchmarks.
- The fastest implementation for Metal remains [the Draw Things app](https://drawthings.ai/) that can produce a 1024x1024 image in just 14 seconds!

## Resolution Limits

**Maximum resolution**: 1792x1792 pixels. The model produces good results up to this size; beyond this resolution image quality degrades significantly (this is a model limitation, not an implementation issue).

**Minimum resolution**: 64x64 pixels.

Dimensions should be multiples of 16 (the VAE downsampling factor).

## Model Architecture

Both models share the same architecture, a rectified flow transformer:

| Component | Architecture |
|-----------|-------------|
| Transformer | 5 double blocks + 20 single blocks, 3072 hidden dim, 24 attention heads |
| VAE | AutoencoderKL, 128 latent channels, 8x spatial compression |
| Text Encoder | Qwen3-4B, 36 layers, 2560 hidden dim |

The models differ in inference:

| | Distilled | Base |
|---|-----------|------|
| Steps | 4 | 50 (default) |
| CFG guidance | 1.0 (none) | 4.0 (default) |
| Passes per step | 1 | 2 (conditioned + unconditioned) |

The model type is autodetected from `model_index.json` in the model directory. Use `--base` to force base model mode if autodetection fails.

**Classifier-Free Guidance (CFG)**: The base model runs the transformer twice per step — once with an empty prompt (unconditioned) and once with the real prompt (conditioned). The final velocity is `v = v_uncond + guidance * (v_cond - v_uncond)`. This makes each step ~2x slower than the distilled model, and the base model needs ~12x more steps, making it roughly 25x slower overall.

## Timestep Schedule

By default, denoising uses a **shifted sigmoid** timestep schedule (matching the official BFL implementation). This schedule concentrates most steps in the high-noise regime and rushes through the detail-forming region near t=0. For the 4 steps distilled model, this is definitely the way to go, and changing scheduler will produce proor results.

### Linear scheduling

However, for the base model, the shifted sigmoid shceduler may look extremely unbalanced — for example at 10 steps, the first 5 steps cover only 12% of the denoising trajectory while the last 5 steps cover 88%. Still, the base model works well with this scheduler, but after some testing I decided to add the `--linear` flag in order to switch to a uniform timestep schedule, where each step covers an equal portion of the trajectory. This sometimes produces better results with the **base model**, at least more realistic looking results, especially at reduced step count (10 steps, for instance, which is 1/5 of execution time compared to the default 50 steps), since the linear schedule avoids the huge final steps that the shifted sigmoid creates, and this alters the generation in a significant way, often in interesting ways.

### Power curve scheduler

The `--power` flag provides a middle ground: a power curve schedule (`t = 1 - (i/n)^α`) that is denser at the start and sparser at the end, but less extreme than the shifted sigmoid. The default exponent is 2.0 (quadratic); use `--power-alpha` to adjust it (1.0 = linear, higher = more front-loaded).

```bash
# Base model with 10 steps and linear schedule
./flux -d flux-klein-base-model -p "a cat" -o cat.png -s 10 --linear

# Base model with power schedule (quadratic by default)
./flux -d flux-klein-base-model -p "a cat" -o cat.png -s 10 --power

# Power schedule with custom exponent
./flux -d flux-klein-base-model -p "a cat" -o cat.png -s 10 --power-alpha 1.5
```

In interactive CLI mode, toggle with `!linear` or `!power [alpha]`.

**Note**: for the distilled model (4 steps), the shifted sigmoid schedule is part of the distillation training, so alternative schedules are not recommended.

If you have a terminal supporting the iTerm2 or Kitty terminal graphics protocols, it is strongly suggested to test the different schedulers with --show and --show-steps options. It is quite an experience to see the denoising process happening in different ways.

## Memory Requirements

With mmap (default):

| Phase | Memory |
|-------|--------|
| Text encoding | ~2GB (layers loaded on-demand) |
| Diffusion | ~1-2GB (blocks loaded on-demand) |
| Peak | ~4-5GB |

With `--no-mmap` (all weights in RAM):

| Phase | Memory |
|-------|--------|
| Text encoding | ~8GB (encoder weights) |
| Diffusion | ~8GB (transformer ~4GB + VAE ~300MB + activations) |
| Peak | ~16GB (if encoder not released) |

The text encoder is automatically released after encoding, reducing peak memory during diffusion. If you generate multiple images with different prompts, the encoder reloads automatically.

## Memory-Mapped Weights (Default)

Memory-mapped weight loading is enabled by default. Use `--no-mmap` to disable and load all weights upfront.

```bash
./flux -d flux-klein-model -p "A cat" -o cat.png           # mmap (default)
./flux -d flux-klein-model -p "A cat" -o cat.png --no-mmap # load all upfront
```

**How it works:** Instead of loading all model weights into RAM upfront, mmap keeps the safetensors files memory-mapped and loads weights on-demand:

- **Text encoder (Qwen3):** Each of the 36 transformer layers (~400MB each) is loaded, processed, and immediately freed. Only ~2GB stays resident instead of ~8GB.
- **Denoising transformer:** Each of the 5 double-blocks (~300MB) and 20 single-blocks (~150MB) is loaded on-demand and freed after use. Only ~200MB of shared weights stays resident instead of ~4GB.

This reduces peak memory from ~16GB to ~4-5GB, making inference possible on 16GB RAM systems where the Python ML stack cannot run FLUX.2 at all.

**Performance varies by backend:**

- **MPS (Apple Silicon):** mmap is the **fastest** mode. The model stores weights in bf16 format, and MPS uses them directly via zero-copy pointers into the memory-mapped region. No conversion overhead, and the kernel handles paging efficiently.

- **BLAS (CPU):** mmap is **slightly slower** but uses much less RAM. BLAS requires f32 weights, so each block must be converted from bf16→f32 on every step (25 blocks × 4 steps = 100 conversions). With `--no-mmap`, this conversion happens once at startup. **Recommendation:** If you have 32GB+ RAM and use BLAS, try `--no-mmap` for faster inference. If RAM is limited, mmap lets you run at all.

- **Generic (pure C):** Same tradeoffs as BLAS, but slower overall.

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
```

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
flux_image *flux_image_load(const char *path);     /* Load PNG, JPEG, or PPM */
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
int flux_is_distilled(flux_ctx *ctx);              /* 1 = distilled, 0 = base */
void flux_set_base_mode(flux_ctx *ctx);            /* Force base model mode */
```

### Parameters

```c
typedef struct {
    int width;              /* Output width in pixels (default: 256) */
    int height;             /* Output height in pixels (default: 256) */
    int num_steps;          /* Denoising steps, 0 = auto (4 distilled, 50 base) */
    int64_t seed;           /* Random seed, -1 for random (default: -1) */
    float guidance;         /* CFG guidance scale, 0 = auto (1.0 distilled, 4.0 base) */
    int linear_schedule;    /* Use linear timestep schedule (0 = shifted sigmoid) */
    int power_schedule;     /* Use power curve timestep schedule */
    float power_alpha;      /* Exponent for power schedule (default: 2.0) */
} flux_params;

/* Initialize with sensible defaults (auto steps and guidance from model type) */
#define FLUX_PARAMS_DEFAULT { 256, 256, 0, -1, 0.0f, 0, 0, 2.0f }
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
