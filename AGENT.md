This is a C implementation of two image synthesis model families:
- Flux.2 Klein (4B and 9B variants)
- Z-Image-Turbo (6B)

The project is called "Iris" (from the Greek goddess of the rainbow).
The Flux models are created by Black Forest Labs.
Z-Image-Turbo is published as `Tongyi-MAI/Z-Image-Turbo`.

Model type and architecture are autodetected from model metadata/config files.
Do not rely on hardcoded dimensions when a config value is available.

# Naming Convention

- **`iris_`** prefix for all shared/generic identifiers
- **`_flux`** postfix on internal functions/types specific to the Flux model family
- **`_zimage`** postfix on internal functions/types specific to Z-Image
- **No postfix** on public API functions (they route internally by model type)
- **Unchanged**: `qwen3_*`, `safetensors_*`, `zi_*` internal helpers (component-level namespaces)

# Supported Model Variants

Flux variants:
- **4B Distilled** (`flux-klein-4b`): 4 steps, no CFG, fast.
- **4B Base** (`flux-klein-4b-base`): 50 steps default, CFG, higher quality but much slower.
- **9B Distilled** (`flux-klein-9b`): 4 steps, larger model, higher quality. Non-commercial license.
- **9B Base** (`flux-klein-9b-base`): 50 steps default, CFG. Non-commercial license.

Z-Image variant:
- **Z-Image-Turbo** (`zimage-turbo`): distilled S3-DiT model, default 8 NFE (9 scheduler values), guidance 0.0.

# High-Level Capabilities

- Flux supports both txt2img and img2img with text conditioning.
- Z-Image currently supports txt2img path in this codebase.
- Text embeddings are generated via Qwen3.
- VAE is used for latent/image conversion in both families.

# File Structure

```
iris.c                    - Main library (model load, generation routing)
iris_transformer_flux.c   - Flux diffusion transformer (MMDiT)
iris_transformer_zimage.c - Z-Image transformer (S3-DiT)
iris_sample.c             - Sampling/denoising loops (Euler ODE)
iris_qwen3.c              - Qwen3 text encoder
iris_qwen3_tokenizer.c    - BPE tokenizer
iris_vae.c                - VAE encoder/decoder
iris_kernels.c            - CPU kernels (softmax, RMSNorm, etc.)
iris_metal.m              - Metal GPU acceleration runtime
iris_metal.h              - Metal/GPU API surface
iris_shaders.metal        - Metal compute kernels
iris_safetensors.c        - Weight loading
iris_image.c              - Image I/O (PNG/PPM/JPEG)
png.c                     - PNG encoder/decoder
jpeg.c                    - JPEG decoder
iris_cli.c                - Interactive CLI mode (REPL)
embcache.c                - Embedding cache (4-bit quantized)
linenoise.c               - Line editing library
terminals.c               - Terminal handling
main.c                    - CLI entry point
```

# Build Targets

This project implements three targets:
- MPS: Apple Silicon GPU path.
- BLAS: optimized CPU inference via BLAS/OpenBLAS. On Windows (MSYS2
  UCRT64) this is the only supported target; it builds without the
  interactive REPL.
- generic: pure C fallback, very slow.

# Windows Notes

Build from an MSYS2 UCRT64 shell:

    pacman -S mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-openblas make
    make blas

Platform shims, all guarded by `#ifdef _WIN32`:
- `iris_safetensors.c`: file mapping via CreateFileMapping/MapViewOfFile
  instead of mmap. The view keeps its own references, so the file and
  mapping handles are closed right after mapping, mirroring the POSIX
  code that closes the fd after mmap. Release goes through the
  `iris_unmap()` helper.
- `iris_transformer_flux.c`: GetSystemInfo instead of sysconf for the CPU
  count. MinGW ships unistd.h but not sysconf.
- `terminals.c`: the iTerm2 temp-file path returns -1. MinGW has no
  mkstemps(), and iTerm2 is never detected on Windows anyway.
- Config, tokenizer and index files are opened in binary mode ("rb"):
  text mode would translate CRLF and break sized reads.

Not built on Windows: the interactive REPL (`iris_cli.c`, `linenoise.c`,
`embcache.c` need termios). Requesting interactive mode prints an error
and exits. Terminal image previews do not render.

Note that `make pngtest` still fails on its first image pair: upstream
compares `woman_with_sunglasses.png` (512x512) against
`woman_with_sunglasses_compressed2.png` (256x256). The `cat_*` pair is
correctly matched and passes.

## Windows performance

Measured on a Xeon W-2125 (4 cores / 8 threads, 32 GB), flux-klein-4b,
512x512 at 4 steps. **Use the mmap default and cap BLAS at one thread per
physical core** -- that combination is 2.7x faster than the opposite one:

    ./iris.exe -d flux-klein-4b --blas-threads 4 -p "..." -o out.png

| Config                    | 512x512 | 256x256 |
|---------------------------|---------|---------|
| mmap + 4 threads          | 219s    | 138s    |
| --no-mmap + 4 threads     |         | 238s    |
| --no-mmap + 8 threads     | 592s    | 293s    |

Two results worth keeping:
- **`--no-mmap` is slower here**, unlike what the README suggests for
  machines with spare RAM. Copying ~15 GB of weights into RAM costs 26s
  for the transformer alone and doubles text encoding; a single
  generation never earns that back. It may still pay off in a long
  interactive session, which this build does not have.
- **8 threads lose to 4.** The 4 physical cores already saturate the
  vector units, so hyperthreading only adds contention with the
  head-parallel attention threads.

Neither switch changes the output: mmap vs --no-mmap and 4 vs 8 threads
produce bit-identical images (mean_diff 0.000000).

## Windows test results

`run_test.py` cannot be used as-is on hardware this slow: it kills each
test after 300s, and the 512x512 case needs more. The three cases were
run by hand instead, all passing well inside the threshold of 20:

| Test                          | mean_diff | upstream expects |
|-------------------------------|-----------|------------------|
| 64x64, 2 steps, seed 42       | 1.14      | ~3.4             |
| 512x512, 4 steps, seed 123    | 2.12      | ~1.7             |
| img2img 256x256, seed 456     | 8.29      | 6-17             |

# Development Rules

- No additional project dependencies. Acceptable external deps are BLAS/OpenBLAS and Metal/MPS from macOS.
- Reject tiny speed gains that add complexity; prefer substantial wins.
- Always test code modifications with `make test`.
- Once changes are validated, commit them.
- Never add or commit unrelated unstaged files.
- Keep code simple and understandable; leave no dead code.
- If you optimize one backend, verify others were not regressed.
- Stick to standard C; avoid compiler-specific tricks/pragmas unless strictly required.

# How To Run

Flux examples:

    ./iris -d flux-klein-4b -p "a cat and a dog playing" -o /tmp/test.png
    ./iris -d flux-klein-4b-base -p "a cat and a dog playing" -o /tmp/test.png
    ./iris -d flux-klein-9b -p "a cat and a dog playing" -o /tmp/test.png
    ./iris -d flux-klein-9b-base -p "a cat and a dog playing" -o /tmp/test.png

Z-Image example:

    ./iris -d zimage-turbo -p "a fish" -o /tmp/zimage.png

On Windows the binary is `iris.exe`:

    ./iris.exe -d flux-klein-4b -p "a cat and a dog playing" -o test.png

If model weights are missing, use the download script only after user approval.

# Python Reference Implementations

For parity checks/debugging:

Flux references:
- Python venv in `./flux_env/`
- Official Flux Python code in `./flux2/`

Z-Image references:
- `flux_env/lib/python3.12/site-packages/diffusers/models/transformers/transformer_z_image.py`
- `flux_env/lib/python3.12/site-packages/diffusers/pipelines/z_image/pipeline_z_image.py`
- `flux_env/lib/python3.12/site-packages/diffusers/schedulers/scheduling_flow_match_euler_discrete.py`
- `flux_env/lib/python3.12/site-packages/diffusers/models/autoencoders/autoencoder_kl.py`

Rules:
- Never add/commit `flux_env/` or `flux2/`.
- If missing, ask user before recreating/downloading.

# Debugging Notes

- Reusable debug scripts belong in `./debug`.
- One-off throwaway debugging tools should be created in `/tmp` and discarded.
- JPEG code has dedicated tests/tools in `./jpg_test`.

# Flux Pipeline Overview

```
1. Text Encoding:    prompt -> Qwen3 -> [512, text_dim] embeddings
2. Latent Init:      random noise [H/16, W/16, 128]
3. Denoising Loop:   double blocks -> single blocks -> final layer -> velocity
4. VAE Decode:       latents -> VAE decoder -> RGB image

img2img: VAE-encode reference images and pass as extra tokens (in-context conditioning).

Base CFG: run transformer twice per step (empty prompt + conditioned prompt):
v = v_uncond + guidance * (v_cond - v_uncond)
```

# Flux Key Architecture Constants

Read from runtime config; reference values:

Flux 4B transformer:
- hidden=3072
- heads=24
- head_dim=128
- mlp=9216
- double_blocks=5
- single_blocks=20
- text_dim=7680

Flux 9B transformer:
- hidden=4096
- heads=32
- head_dim=128
- mlp=12288
- double_blocks=8
- single_blocks=24
- text_dim=12288

Qwen3 (4B/9B encoders used for Flux):
- 4B: hidden=2560, q_heads=32, kv_heads=8, layers=36, output_dim=7680
- 9B: hidden=4096, q_heads=32, kv_heads=8, layers=36, output_dim=12288

Shared:
- latent_ch=128 (Flux latent after patchification)
- head_dim=128

# Flux Critical Implementation Details

- Concatenation order for attention is `[TEXT, IMAGE]`, not `[IMAGE, TEXT]`.
- AdaLN formula is `out = (1 + scale) * norm(x) + shift`.
- Final layer modulation split is `(scale, shift)`, not `(shift, scale)`.
- RoPE pair rotation is:
  - `out0 = cos * x0 - sin * x1`
  - `out1 = cos * x1 + sin * x0`

# Flux RoPE (Rotary Position Embedding)

4-axis RoPE with 32 dims per axis (128 total):
- Axis 0 (dims 0..31): T position
- Axis 1 (dims 32..63): H position
- Axis 2 (dims 64..95): W position
- Axis 3 (dims 96..127): L sequence index

Token usage:
- Image tokens: use H/W axes, T/L identity.
- Text tokens: use L axis only.
- Reference image tokens (img2img): same as image plus T offset (10, 20, 30...).

# Flux Timestep Embedding

1. Scale timestep by 1000.
2. Sinusoidal embedding (128 freqs -> 256 dims).
3. MLP: linear(256->hidden) + SiLU + linear(hidden->hidden).

# Flux Text Encoder (Qwen3)

Chat template:
```
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
<think>

</think>

```

Flux extraction: concatenate layers 8, 17, 26 (0-indexed):
- 4B: [seq, 7680]
- 9B: [seq, 12288]

# Flux VAE

- Latent space: 32 channels, 16x spatial compression.
- Patchify (encode): [B, 32, H/8, W/8] -> [B, 128, H/16, W/16]
- Unpatchify (decode): [B, 128, H/16, W/16] -> [B, 32, H/8, W/8]
- Channel multipliers: [1,2,4,4] -> [128,256,512,512]

# Flux Double Block Flow

Input: `img_hidden [img_seq, hidden]`, `txt_hidden [txt_seq, hidden]`

1. AdaLN normalize both streams (shift1, scale1)
2. Stream-specific Q/K/V projections
3. QK per-head RMSNorm
4. RoPE (image axes 1/2, text axis 3)
5. Joint attention over concatenated KV
6. Output projection + gate1
7. Residual add
8. AdaLN normalize (shift2, scale2)
9. SiLU-gated MLP
10. gate2 + residual add

Per-stream modulation params: shift1, scale1, gate1, shift2, scale2, gate2.

# Flux Single Block Flow

Input: concatenated `[txt_hidden, img_hidden]`

1. AdaLN normalize (shift/scale from t_emb)
2. Fused QKV+MLP projection -> [Q,K,V,gate,up]
3. QK normalization
4. RoPE (text uses axis 3, image uses axes 1/2)
5. Full self-attention
6. SwiGLU: `silu(gate) * up`
7. Concat attention output + MLP output
8. Output projection
9. Gated residual add

# Z-Image Pipeline Overview

```
1. Text Encoding:    prompt -> Qwen3 hidden_states[-2] -> [seq, 2560]
2. Latent Init:      random noise [16, H/8, W/8] then patch-space processing
3. Denoising Loop:   noise_refiner(2) -> context_refiner(2) -> main layers(30)
4. Final Layer:      norm + modulation + projection to patch channels
5. Unpatchify:       [num_patches, 64] -> [16, H/8, W/8]
6. VAE Decode:       latents -> image
```

Z-Image uses sequence order `[IMAGE_TOKENS | CAPTION_TOKENS]` in its transformer path.

# Z-Image Architecture (S3-DiT)

Overview:
- Model: Z-Image-Turbo (6B)
- Architecture: S3-DiT single-stream style with refiners
- Steps: default 8 NFE (scheduler array has 9 sigma values)
- guidance: 0.0 (no CFG)

Transformer config (reference):
- dim=3840
- n_heads=30
- n_kv_heads=30
- head_dim=128
- n_layers=30
- n_refiner_layers=2 (noise + context)
- in_channels=16
- patch_size=2
- cap_feat_dim=2560
- axes_dims=[32,48,48]
- rope_theta=256.0
- ffn_hidden=10240

# Z-Image Text Encoder

Uses the same Qwen3 4B base encoder family, but extraction differs from Flux:
- extract only `hidden_states[-2]` (layer 34/36)
- output shape [seq, 2560]

Chat template is the same as Flux Qwen3 template.

# Z-Image VAE Notes

Z-Image VAE is Flux-like architecture with different latent settings:
- latent_channels=16 (Flux uses 32)
- scaling_factor=0.3611
- shift_factor=0.1159
- patchification for transformer input uses 2x2 over 16ch -> 64-dim patch vector

Decode uses:
- `latents = (latents / scaling_factor) + shift_factor`, then decode.

# Z-Image Block Structure

With modulation (noise_refiner + main layers):
- modulation vector split into: scale_msa, gate_msa, scale_mlp, gate_mlp
- gates go through tanh
- normalization is scaled (no additive shift in block modulation path)
- attention and FFN residuals are gated

Without modulation (context_refiner):
- no adaLN modulation
- standard residual attention + FFN structure

# Z-Image RoPE

3-axis RoPE, head_dim=128 split as:
- Axis T: 32 dims
- Axis H: 48 dims
- Axis W: 48 dims

Theta is 256.0.
Caption and image position-id construction must match CPU/GPU path exactly.

# Z-Image Scheduler

Default scheduler follows official diffusers FlowMatch Euler behavior (static shift).
Important points:
- terminal sigma is appended as 0
- implementation should match official sigma construction semantics
- explicit `--linear` or `--power` options override default schedule for parity/testing

# Z-Image Critical Implementation Details

- Transformer sequence order is `[IMAGE | CAPTION]` in Z-Image path.
- Step previews must decode correctly patchified latent representation (`patch_size=2`),
  otherwise shown step resolution/content can be misleading.
- Caption padding and position-id alignment must match between CPU and MPS paths.
- Keep immutable/static and dynamic matrix paths separate in MPS GEMM caching.

# Z-Image Runtime Lessons (Keep)

- Keep transformer loaded across generations in interactive workflows; avoid load/free per image.
- Precompute per-step modulation once and reuse across modulated blocks.
- Keep GPU fallback policy inside the zImage GPU forward path; avoid immediate full CPU fallback.
- Run zImage final layer on GPU path where possible; read back only what is needed for decode.
- Run image/caption embedding projections on GPU path in MPS mode.
- Warm bf16 weight caches at load time to reduce first-step jitter in steady-state usage.
- Keep embedding-cache path wired for repeated prompts in distilled interactive mode.
- Maintain zImage-specific denoising timing counters for targeted profiling.

# Z-Image Known Pitfalls (Historical Bugs)

1. **MPS SGEMM B-cache misuse caused VAE decode corruption (hue/border artifacts).**
   - Root cause: generic SGEMM cached matrix B by pointer, but VAE attention K/V are dynamic temporaries.
   - Fix: split API paths:
     - `iris_metal_sgemm`: generic path, no B-pointer cache
     - `iris_metal_sgemm_cached`: explicit static-weight cached path
   - Static weight call sites (for example linear layers) use cached variant only.

2. **Scheduler parity bug with official Python implementation.**
   - Z-Image sigma schedule must follow official diffusers FlowMatch behavior.
   - Incorrect sigma handling can waste steps and make show-steps output confusing.

3. **Step-preview mismatch bug.**
   - Z-Image preview decode must account for patch-size transformation, otherwise preview steps do not match final decode behavior.

4. **CPU/GPU position-id mismatch under padded caption sequences.**
   - Using non-padded cap length in one path and padded length in another changes RoPE indexing and output quality.

5. **RMSNorm temporary weight caching pitfalls.**
   - Temporary fused norm weights must not be pointer-cached as immutable weights.

# Test Commands

Basic verification:
```bash
make test
```

Manual Flux sanity:
```bash
./iris -d flux-klein-4b -p "A fluffy orange cat sitting on a windowsill" \
  --seed 42 --steps 2 -o /tmp/iris_test.png -W 64 -H 64
```

Manual Z-Image sanity:
```bash
./iris -d zimage-turbo -p "a fish" --seed 43 --steps 8 -o /tmp/zimage_test.png
```

# Flux Known Pitfalls (Historical Bugs)

1. **Unified RoPE kernel indexing**: GPU must use consecutive pairs `(d, d+1)`, not axis-half indexing.
2. **GPU caching of timestep params**: step-dependent shift/scale/gate must not be cached as static weights.
3. **CLI mode CFG routing**: base models in interactive mode must go through `iris_generate()` (CFG-aware), not distilled-only embedding path.
