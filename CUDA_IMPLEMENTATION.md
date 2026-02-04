# CUDA Implementation Notes

This document describes the CUDA acceleration layer for flux2.c.

---

## Files Modified for CUDA

| File | Changes |
|------|---------|
| `flux_cuda.cu` | Main CUDA implementation (kernels, cuBLAS, tensor pool) |
| `flux_cuda.h` | Public API declarations |
| `flux_transformer.c` | CUDA paths for double/single blocks, BF16 weight loading |
| `flux_vae.c` | CUDA conv2d for VAE decoder |
| `flux_qwen3.c` | CUDA causal attention for text encoder |
| `Makefile` | `make cuda` target with nvcc compilation |

---

## Architecture Overview

### GPU Acceleration Strategy

1. **Weights stay on GPU** - BF16 weights are uploaded once and cached
2. **Activations in tensor pool** - Reusable GPU buffers avoid malloc/free
3. **Minimal CPU↔GPU transfers** - Only upload inputs, download outputs
4. **cuBLAS for matmuls** - Uses tensor cores via `cublasGemmEx`

### Key Data Structures

```c
// Tensor pool - reusable GPU buffers
g_tensor_pool[64]  // Pool of GPU allocations
flux_cuda_tensor_get(size)    // Acquire buffer
flux_cuda_tensor_release(id)  // Release buffer

// Weight cache - permanent GPU storage for weights
g_weight_cache[2048]  // CPU ptr → GPU ptr mapping
weight_cache_get()    // Lookup cached weight
weight_cache_add()    // Upload and cache weight
```

---

## CUDA Kernels

### Transformer Operations

| Kernel | Purpose |
|--------|---------|
| `k_silu` | SiLU activation |
| `k_silu_mul` | Fused SiLU + elementwise multiply (SwiGLU) |
| `k_mul` | Elementwise multiply |
| `k_gated_add` | Gated residual: `out += gate * x` |
| `k_split_fused` | Split fused QKV+MLP projection |
| `k_concat` | Concatenate attention + MLP outputs |
| `k_rms_norm` | RMSNorm |
| `k_qk_rms_norm` | Fused Q/K normalization |
| `k_adaln_norm` | AdaLN modulation |
| `k_softmax` | Row-wise softmax |
| `k_softmax_attention` | Fused attention softmax with scale |

### RoPE Kernels

| Kernel | Purpose |
|--------|---------|
| `k_rope_2d` | 2D RoPE for transformer (4 axes) |
| `k_rope_2d_offset` | RoPE with sequence offset |

### VAE Kernels

| Kernel | Purpose |
|--------|---------|
| `k_im2col` | im2col for conv2d |
| `k_add_bias_conv` | Add bias after convolution |

### Text Encoder Kernels

| Kernel | Purpose |
|--------|---------|
| `k_causal_softmax` | Causal attention with mask |
| `k_bf16_to_f32` | BF16→F32 conversion on GPU |

### Utility Kernels

| Kernel | Purpose |
|--------|---------|
| `k_transpose_shd_to_hsd` | Transpose [seq,heads,dim] → [heads,seq,dim] |
| `k_transpose_hsd_to_shd` | Transpose [heads,seq,dim] → [seq,heads,dim] |

---

## BF16 Weight Handling

### mmap Mode
- Weights read directly from mmap'd safetensors file as BF16
- Pointers are stable (point into mmap region)
- Weight cache **enabled** - weights uploaded once, cached permanently
- `g_weight_cache_disabled = 0`

### no-mmap Mode
- Weights copied via `safetensors_get_bf16()` into malloc'd buffers
- Pointers may be reused after free
- Weight cache **disabled** - weights uploaded fresh each time
- `g_weight_cache_disabled = 1`

### BF16→F32 Conversion
```c
flux_cuda_sgemm_gpu_bf16()  // For mmap with cache
// 1. Check cache for existing F32 conversion
// 2. If miss: upload BF16, convert to F32 on GPU, cache result
// 3. Run cuBLAS sgemm with F32 weights
```

---

## Transformer Forward Paths

### Double Blocks (`double_block_forward_cuda`)
1. Upload img/txt hidden states to GPU
2. AdaLN modulation (fused for all streams)
3. QKV projection via `flux_cuda_sgemm_gpu_bf16`
4. Q/K normalization + RoPE
5. Joint attention via `flux_cuda_joint_attention_t`
6. Output projection + gated residual
7. MLP (SwiGLU) + gated residual
8. Download results to CPU

### Single Blocks (`single_block_forward_cuda_chained`)
- **Chained execution** - hidden state stays on GPU across all 20 blocks
- AdaLN vectors pre-computed once for all blocks
- Only final result downloaded to CPU

---

## Attention Implementation

### Joint Attention (Double Blocks)
```
img_out = softmax(img_Q @ cat(img_K, txt_K)^T) @ cat(img_V, txt_V)
txt_out = softmax(txt_Q @ cat(img_K, txt_K)^T) @ cat(img_V, txt_V)
```
- Uses `flux_cuda_joint_attention_t`
- Batched cuBLAS gemm for Q@K^T and scores@V

### Causal Attention (Qwen3 Text Encoder)
- GQA with 32 query heads, 8 KV heads (4:1 ratio)
- Causal mask + attention mask
- Uses `flux_cuda_causal_attention`

---

## Performance Characteristics

### Typical 1024×1024 @ 4 steps (RTX PRO 6000 Blackwell)

| Phase | Time | Notes |
|-------|------|-------|
| Text encoding | ~3s | Qwen3 36 layers, CUDA attention |
| Denoising | ~7s | 5 double + 20 single blocks |
| VAE decode | ~3.5s | CUDA conv2d |
| **Total** | ~14s | |

### Memory Usage
- Transformer weights: ~8GB (BF16)
- Qwen3 weights: ~8GB (F32, loaded per-layer in mmap mode)
- Activations: ~2GB peak
- Weight cache: Grows to ~4GB for transformer

---

## Build Instructions

```bash
# Build with CUDA support
make cuda

# Requirements:
# - CUDA toolkit (nvcc)
# - cuBLAS
# - OpenBLAS (for CPU fallback)

# GPU architecture auto-detected, or override:
make cuda CUDA_ARCH=sm_89  # Ada
make cuda CUDA_ARCH=sm_120 # Blackwell
```

---

## Debugging

### Enable verbose output
```c
// In flux_cuda.cu, uncomment:
// #define CUDA_DEBUG
```

### Check GPU memory
```bash
nvidia-smi --query-gpu=memory.used,memory.free --format=csv -l 1
```

### Verify correctness
```bash
# Generate with CPU reference
./flux_cpu -d model -p "test" -o ref.png --seed 42

# Generate with CUDA
./flux -d model -p "test" -o cuda.png --seed 42

# Compare (should be nearly identical, small FP differences OK)
```

---

## Known Limitations

1. **No Flash Attention** - Using standard cuBLAS attention
2. **No FP16 compute** - All compute in FP32 (weights can be BF16)
3. **Single GPU only** - No multi-GPU support
4. **No dynamic batching** - Single image at a time

---

## Future Optimizations

- [ ] Flash Attention 2 integration
- [ ] FP16 compute path for Ampere+
- [ ] Persistent kernel for single blocks
- [ ] CUDA graphs for reduced launch overhead
