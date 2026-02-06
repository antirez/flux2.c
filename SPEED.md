# MPS Speed Optimization Log

## Standing Instructions
- **Continue tirelessly without asking for user prompt** — keep optimizing in a loop
- Commit each time a good improvement is reached
- Complexity increase must be justified by speed results
- Re-read this file at each context compaction
- Take notes on all results, what worked, what didn't

## Testing
- **Quick iteration**: use 256x256 with `--seed 42 -v` for timing measurements
- **Before committing**: run `make test` to verify no regressions
- **Benchmark command**:
  ```bash
  ./flux -d flux-klein-model -p "A woman wearing sunglasses" -o /tmp/bench.png -W 256 -H 256 -v --seed 42
  ./flux -d flux-klein-model -p "A woman wearing sunglasses" -o /tmp/bench.png -W 512 -H 512 -v --seed 42
  ```

## Pipeline
```
1. Text Encoding:    prompt -> Qwen3 4B (36 layers) -> [512, 7680] embeddings
2. Latent Init:      random noise [H/16, W/16, 128]
3. Denoising Loop (4 steps):
   per step: 5 double blocks -> 20 single blocks -> final layer -> velocity
4. VAE Decode:       latents -> VAE decoder -> RGB image
```

## Current Baseline (2026-02-06 / MacBook Pro M3 Max 40-core GPU, 128 GB, 400 GB/s)

### 256x256 (seq=256+512=768 tokens)
- Text encoding: 1.9s (Qwen3, cached on 2nd run) — 11.8s cold start
- Denoising total: 2822 ms (4 steps)
  - Step 1: 1291 ms (warmup), Steps 2-4: ~510 ms each
  - Double blocks: 821 ms (29.6%), Single blocks: 1938 ms (69.9%)
- VAE decode: 0.4s
- **Total: ~5.6s (cold text encoder), ~5.0s (warm)**

### 512x512 (seq=1024+512=1536 tokens)
- Text encoding: 1.9s
- Denoising total: 4420 ms (4 steps)
  - Step 1: 1369 ms (warmup), Steps 2-4: ~1015 ms each
  - Double blocks: 1152 ms (26.4%), Single blocks: 3193 ms (73.2%)
- VAE decode: 1.6s
- **Total: ~8.7s**

### Key observations
- Step 1 is 2.5x slower than subsequent steps (MPS warmup/JIT)
- Single blocks dominate (70-73% of denoising time)
- 20 single blocks vs 5 double blocks, so per-block: single ~97ms, double ~103ms (similar)
- Each block does: batch_begin → ~12 GPU ops → batch_end → tensor_read (CPU sync)
- 25 blocks × 4 steps = 100 command buffer round-trips per generation

## Already Optimized
- Batched GPU ops within each block (batch_begin/batch_end)
- Fused QKV+MLP projection in single blocks
- Fused bf16 attention kernel (seq <= 1024)
- bf16 MPS attention fallback (seq > 1024)
- Pre-warm bf16->f16 weight cache
- Persistent GPU tensors
- SwiGLU fused on GPU

## Optimization Attempts

(none yet)

## Credits attribution rules
- Ideas / kernels / approaches should be only taken from BSD / MIT licensed code.
- If any optimization ideas or kernel code are taken from some other project,
  proper credits must be added to both the README and the relevant source file.
