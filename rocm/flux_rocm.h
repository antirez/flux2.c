/*
 * FLUX ROCm Acceleration
 *
 * GPU-accelerated operations using AMD ROCm/HIP.
 * Mirrors flux_metal.h API for drop-in replacement.
 */

#ifndef FLUX_ROCM_H
#define FLUX_ROCM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================
 * Initialization / Cleanup
 * ======================================================================== */

/*
 * Initialize ROCm acceleration.
 * Returns 1 on success, 0 if ROCm is not available.
 * Safe to call multiple times.
 */
int flux_rocm_init(void);

/*
 * Check if ROCm acceleration is available and initialized.
 */
int flux_rocm_available(void);

/*
 * Cleanup ROCm resources.
 */
void flux_rocm_cleanup(void);

/*
 * Reset GPU state (clear caches/pools).
 * Call between independent inference phases.
 */
void flux_rocm_reset(void);

/*
 * Synchronize all pending GPU operations.
 */
void flux_rocm_sync(void);

/*
 * Get GPU memory usage info (for debugging).
 */
size_t flux_rocm_memory_used(void);

/* ========================================================================
 * GPU Tensor API
 * 
 * Unlike Apple Silicon's unified memory, discrete GPUs require explicit
 * memory management. These tensors live on GPU and are transferred
 * explicitly when needed.
 * ======================================================================== */

/*
 * Opaque handle to a GPU-resident tensor.
 */
typedef struct flux_rocm_tensor *flux_rocm_tensor_t;

/*
 * Create a GPU tensor and copy data from CPU.
 * Returns NULL on failure.
 */
flux_rocm_tensor_t flux_rocm_tensor_create(const float *data, size_t num_elements);

/*
 * Create an uninitialized GPU tensor (for output buffers).
 */
flux_rocm_tensor_t flux_rocm_tensor_alloc(size_t num_elements);

/*
 * Create a persistent GPU tensor (won't return to pool on free).
 */
flux_rocm_tensor_t flux_rocm_tensor_alloc_persistent(size_t num_elements);

/*
 * Mark tensor as persistent.
 */
void flux_rocm_tensor_set_persistent(flux_rocm_tensor_t tensor, int persistent);

/*
 * Copy tensor data back to CPU.
 * Blocks until GPU operations complete.
 */
void flux_rocm_tensor_read(flux_rocm_tensor_t tensor, float *out, size_t num_elements);

/*
 * Copy data from CPU to tensor.
 */
void flux_rocm_tensor_write(flux_rocm_tensor_t tensor, const float *data, size_t num_elements);

/*
 * Release a GPU tensor (returns to pool if not persistent).
 */
void flux_rocm_tensor_free(flux_rocm_tensor_t tensor);

/*
 * Get tensor element count.
 */
size_t flux_rocm_tensor_size(flux_rocm_tensor_t tensor);

/*
 * Check if tensor is bf16 format.
 */
int flux_rocm_tensor_is_bf16(flux_rocm_tensor_t tensor);

/*
 * Allocate bf16 tensor (half the memory of f32).
 */
flux_rocm_tensor_t flux_rocm_tensor_alloc_bf16(size_t num_elements);

/*
 * Get raw GPU pointer (for passing to BLAS/kernels).
 * WARNING: Pointer is only valid while tensor exists.
 */
void *flux_rocm_tensor_gpu_ptr(flux_rocm_tensor_t tensor);

/* ========================================================================
 * BLAS Operations (via hipBLAS/rocBLAS)
 * ======================================================================== */

/*
 * GPU matrix multiplication.
 * C[M,N] = alpha * A[M,K] @ B[K,N] + beta * C[M,N]
 *
 * Data must already be on GPU (use tensor API).
 */
void flux_rocm_sgemm(int transpose_a, int transpose_b,
                     int M, int N, int K,
                     float alpha,
                     const float *A, int lda,
                     const float *B, int ldb,
                     float beta,
                     float *C, int ldc);

/*
 * GPU matrix multiplication with bf16 weights.
 * A is f32, B is bf16, C is f32.
 */
void flux_rocm_sgemm_bf16(int transpose_a, int transpose_b,
                          int M, int N, int K,
                          float alpha,
                          const float *A, int lda,
                          const uint16_t *B_bf16, int ldb,
                          float beta,
                          float *C, int ldc);

/*
 * Batched GPU matrix multiplication.
 */
void flux_rocm_sgemm_batch(int transpose_a, int transpose_b,
                           int M, int N, int K,
                           float alpha,
                           const float *A, int lda, int stride_a,
                           const float *B, int ldb, int stride_b,
                           float beta,
                           float *C, int ldc, int stride_c,
                           int batch_count);

/* ========================================================================
 * Linear Layers (GEMM wrappers for neural network use)
 * ======================================================================== */

/*
 * Linear layer: out = x @ W^T
 * x: [seq_len, in_dim]
 * W: [out_dim, in_dim]
 * Returns new GPU tensor with result.
 */
flux_rocm_tensor_t flux_rocm_linear(flux_rocm_tensor_t x,
                                    const float *W,
                                    int seq_len, int in_dim, int out_dim);

/*
 * Linear layer with bf16 weights.
 */
flux_rocm_tensor_t flux_rocm_linear_bf16(flux_rocm_tensor_t x,
                                         const uint16_t *W_bf16,
                                         int seq_len, int in_dim, int out_dim);

/* ========================================================================
 * Element-wise Operations (Custom HIP Kernels)
 * ======================================================================== */

/*
 * RMSNorm: out = x * rsqrt(mean(x^2) + eps) * weight
 * x: [seq_len, hidden], weight: [hidden]
 */
void flux_rocm_rms_norm(float *out, const float *x, const float *weight,
                        int seq_len, int hidden, float eps);

/*
 * QK RMSNorm (in-place on Q and K).
 * q, k: [seq, heads*head_dim]
 * q_weight, k_weight: [head_dim]
 */
void flux_rocm_qk_rms_norm(float *q, float *k,
                           const float *q_weight, const float *k_weight,
                           int seq, int heads, int head_dim, float eps);

/*
 * AdaLN: out = (1 + scale) * layernorm(x) + shift
 * x: [seq_len, hidden], shift/scale: [hidden]
 */
void flux_rocm_adaln_norm(float *out, const float *x,
                          const float *shift, const float *scale,
                          int seq_len, int hidden, float eps);

/*
 * SiLU activation (in-place): x = x * sigmoid(x)
 */
void flux_rocm_silu(float *x, int n);

/*
 * SiLU with multiply (SwiGLU, in-place): gate = silu(gate) * up
 */
void flux_rocm_silu_mul(float *gate, const float *up, int n);

/*
 * Softmax (row-wise, in-place).
 * x: [rows, cols]
 */
void flux_rocm_softmax(float *x, int rows, int cols);

/*
 * Gated add: out += gate * proj
 */
void flux_rocm_gated_add(float *out, const float *gate,
                         const float *proj, int seq, int hidden);

/* ========================================================================
 * RoPE (Rotary Position Embeddings)
 * ======================================================================== */

/*
 * 2D RoPE (in-place).
 * x: [seq, heads*head_dim]
 * cos_freq, sin_freq: [seq, axis_dim]
 */
void flux_rocm_rope_2d(float *x, const float *cos_freq, const float *sin_freq,
                       int seq, int heads, int head_dim, int axis_dim);

/*
 * Unified RoPE for text+image.
 * Different frequencies for text (0 to img_offset) and image portions.
 */
void flux_rocm_rope_unified(float *q, float *k,
                            const float *txt_cos, const float *txt_sin,
                            const float *img_cos, const float *img_sin,
                            int seq, int img_offset, int heads, int head_dim, int axis_dim);

/* ========================================================================
 * Attention
 * ======================================================================== */

/*
 * Fused attention (non-causal, for transformer).
 * Works on [seq, hidden] layout.
 * Returns 1 on success, 0 on failure.
 */
int flux_rocm_attention_fused(float *out,
                              const float *Q, const float *K, const float *V,
                              int seq_q, int seq_k, int num_heads, int head_dim,
                              float scale);

/*
 * Causal attention (for text encoder).
 * Supports GQA (num_q_heads > num_kv_heads).
 * Returns 1 on success, 0 on failure.
 */
int flux_rocm_causal_attention(float *out,
                               const float *Q, const float *K, const float *V,
                               const int *attention_mask,
                               int seq, int num_q_heads, int num_kv_heads,
                               int head_dim, float scale);

/* ========================================================================
 * Tensor Operations (GPU tensor API)
 * ======================================================================== */

/* AdaLN on GPU tensors */
void flux_rocm_gpu_adaln_norm(flux_rocm_tensor_t out, flux_rocm_tensor_t x,
                              const float *shift, const float *scale,
                              int seq, int hidden, float eps);

/* QK RMSNorm on GPU tensors (in-place) */
void flux_rocm_gpu_qk_rms_norm(flux_rocm_tensor_t q, flux_rocm_tensor_t k,
                               const float *q_weight, const float *k_weight,
                               int seq, int heads, int head_dim, float eps);

/* RoPE unified on GPU tensors */
void flux_rocm_gpu_rope_unified(flux_rocm_tensor_t q, flux_rocm_tensor_t k,
                                const float *txt_cos, const float *txt_sin,
                                const float *img_cos, const float *img_sin,
                                int seq, int img_offset, int heads, int head_dim, int axis_dim);

/* SiLU multiply on GPU tensors */
void flux_rocm_gpu_silu_mul(flux_rocm_tensor_t gate, flux_rocm_tensor_t up, int n);

/* Gated add on GPU tensors */
void flux_rocm_gpu_gated_add(flux_rocm_tensor_t out, const float *gate,
                             flux_rocm_tensor_t proj, int seq, int hidden);

/* Fused attention on GPU tensors */
int flux_rocm_gpu_attention_fused(flux_rocm_tensor_t out,
                                  flux_rocm_tensor_t Q, flux_rocm_tensor_t K, flux_rocm_tensor_t V,
                                  int seq_q, int seq_k, int num_heads, int head_dim, float scale);

/* Split fused QKV+MLP output */
void flux_rocm_gpu_split_qkv_mlp(flux_rocm_tensor_t fused,
                                 flux_rocm_tensor_t q, flux_rocm_tensor_t k, flux_rocm_tensor_t v,
                                 flux_rocm_tensor_t gate, flux_rocm_tensor_t up,
                                 int seq, int hidden, int mlp_hidden);

/* Concatenate attention and MLP outputs */
void flux_rocm_gpu_concat_attn_mlp(flux_rocm_tensor_t attn, flux_rocm_tensor_t mlp,
                                   flux_rocm_tensor_t out, int seq, int hidden, int mlp_hidden);

/* Linear layer on GPU tensors - returns new tensor */
flux_rocm_tensor_t flux_rocm_gpu_linear(flux_rocm_tensor_t x, const float *W,
                                        int seq_len, int in_dim, int out_dim);

/* ========================================================================
 * Double Block Support (GPU tensor operations)
 * ======================================================================== */

/* RoPE 2D on single tensor (for separate img/txt RoPE) */
void flux_rocm_gpu_rope_2d(flux_rocm_tensor_t x,
                           const float *cos_freq, const float *sin_freq,
                           int seq, int heads, int head_dim);

/* Concatenate K or V tensors from img and txt streams
 * Order: [txt, img] (matches Python Flux2 implementation)
 */
void flux_rocm_gpu_concat_tensors(flux_rocm_tensor_t txt, flux_rocm_tensor_t img,
                                  flux_rocm_tensor_t out,
                                  int txt_seq, int img_seq, int hidden);

/* SwiGLU FFN: out = down(silu(gate(x)) * up(x))
 * All three projections executed on GPU
 */
flux_rocm_tensor_t flux_rocm_gpu_swiglu_ffn(flux_rocm_tensor_t x,
                                            const float *gate_weight,
                                            const float *up_weight,
                                            const float *down_weight,
                                            int seq, int hidden, int mlp_hidden);

/* ========================================================================
 * Batch/Chain API (for operation fusion)
 * ======================================================================== */

/*
 * Begin batch mode - operations queued but not executed.
 */
void flux_rocm_batch_begin(void);

/*
 * End batch mode - execute all queued operations.
 */
void flux_rocm_batch_end(void);

/*
 * Check if in batch mode.
 */
int flux_rocm_in_batch(void);

/*
 * Begin operation chain (shared stream, minimal sync).
 */
void flux_rocm_chain_begin(void);

/*
 * End operation chain.
 */
void flux_rocm_chain_end(void);

/*
 * Check if in chain mode.
 */
int flux_rocm_in_chain(void);

/* ========================================================================
 * Utility
 * ======================================================================== */

/*
 * Check if custom HIP kernels are available.
 */
int flux_rocm_kernels_available(void);

/*
 * Pre-warm bf16 conversion cache for weights.
 */
void flux_rocm_warmup_bf16(const uint16_t *bf16_weights, size_t num_elements);

/*
 * Check if bf16 pipeline is available.
 */
int flux_rocm_bf16_available(void);

#ifdef __cplusplus
}
#endif

#endif /* FLUX_ROCM_H */
