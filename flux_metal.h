/*
 * FLUX Metal Acceleration
 *
 * GPU-accelerated matrix operations using Apple Metal Performance Shaders.
 * Provides significant speedup on Apple Silicon Macs.
 */

#ifndef FLUX_METAL_H
#define FLUX_METAL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Initialize Metal acceleration.
 * Returns 1 on success, 0 if Metal is not available.
 * Safe to call multiple times.
 */
int flux_metal_init(void);

/*
 * Check if Metal acceleration is available and initialized.
 */
int flux_metal_available(void);

/*
 * Cleanup Metal resources.
 */
void flux_metal_cleanup(void);

/*
 * Reset all GPU state (caches, pools, pending commands).
 * Call this between independent inference phases (e.g., after text encoding,
 * before loading transformer) to ensure clean GPU state.
 * Device and pipelines are preserved; only data buffers are cleared.
 */
void flux_metal_reset(void);
void flux_metal_rope_cache_begin(void);

/* Debug: Clear only specific caches (for isolating issues) */
void flux_metal_clear_weight_cache_only(void);
void flux_metal_clear_bf16_cache_only(void);
void flux_metal_clear_f16_cache_only(void);
void flux_metal_clear_activation_pool_only(void);

/*
 * GPU-accelerated matrix multiplication using MPS.
 * C[M,N] = alpha * A[M,K] @ B[K,N] + beta * C[M,N]
 *
 * transpose_a: if non-zero, use A^T
 * transpose_b: if non-zero, use B^T
 */
void flux_metal_sgemm(int transpose_a, int transpose_b,
                      int M, int N, int K,
                      float alpha,
                      const float *A, int lda,
                      const float *B, int ldb,
                      float beta,
                      float *C, int ldc);

/*
 * GPU-accelerated matrix multiplication with bf16 weights.
 * C[M,N] = alpha * A[M,K] @ B[K,N] + beta * C[M,N]
 *
 * A is f32, B is bf16 (weights), C is f32
 * This provides 2x memory bandwidth improvement for weight-bound operations.
 */
void flux_metal_sgemm_bf16(int transpose_a, int transpose_b,
                           int M, int N, int K,
                           float alpha,
                           const float *A, int lda,
                           const uint16_t *B_bf16, int ldb,
                           float beta,
                           float *C, int ldc);

/*
 * 2D convolution using MPSGraph (NCHW/OIHW, explicit padding).
 * Returns 1 on success, 0 on failure.
 */
int flux_metal_conv2d(float *out, const float *in,
                      const float *weight, const float *bias,
                      int batch, int in_ch, int out_ch,
                      int H, int W, int kH, int kW,
                      int stride, int padding);

/*
 * Batch matrix multiplication on GPU.
 * Performs batch_count independent matrix multiplications.
 */
void flux_metal_sgemm_batch(int transpose_a, int transpose_b,
                            int M, int N, int K,
                            float alpha,
                            const float *A, int lda, int stride_a,
                            const float *B, int ldb, int stride_b,
                            float beta,
                            float *C, int ldc, int stride_c,
                            int batch_count);

/*
 * Synchronize GPU operations (wait for completion).
 */
void flux_metal_sync(void);

/*
 * Begin a batch of GPU operations.
 * Operations after this call are encoded but not executed until flux_metal_end_batch().
 * This eliminates per-operation sync overhead.
 */
void flux_metal_begin_batch(void);

/*
 * End a batch of GPU operations.
 * Commits all encoded operations and waits for completion.
 */
void flux_metal_end_batch(void);

/*
 * Check if currently in batch mode.
 */
int flux_metal_in_batch(void);

/*
 * Get GPU memory usage info (for debugging).
 */
size_t flux_metal_memory_used(void);

/* ========================================================================
 * GPU Tensor API - Keep activations on GPU between operations
 * ======================================================================== */

/*
 * Opaque handle to a GPU-resident tensor.
 * Tensors are backed by pooled Metal buffers with shared storage mode,
 * allowing zero-copy access from both CPU and GPU on Apple Silicon.
 */
typedef struct flux_gpu_tensor *flux_gpu_tensor_t;

/*
 * Create a GPU tensor from CPU data.
 * Data is copied to GPU (or just referenced in shared memory mode).
 * Returns NULL on failure.
 */
flux_gpu_tensor_t flux_gpu_tensor_create(const float *data, size_t num_elements);

/*
 * Create an uninitialized GPU tensor (for output buffers).
 */
flux_gpu_tensor_t flux_gpu_tensor_alloc(size_t num_elements);

/*
 * Create a persistent GPU tensor that won't be released back to the pool.
 * Use this for tensors that need to stay on GPU between operations.
 * Call flux_gpu_tensor_free() when completely done with the tensor.
 */
flux_gpu_tensor_t flux_gpu_tensor_alloc_persistent(size_t num_elements);

/*
 * Mark an existing tensor as persistent (won't return to pool on free).
 */
void flux_gpu_tensor_set_persistent(flux_gpu_tensor_t tensor, int persistent);

/*
 * Copy tensor data back to CPU.
 * Waits for any pending GPU operations on this tensor.
 */
void flux_gpu_tensor_read(flux_gpu_tensor_t tensor, float *out);

/*
 * Copy data from CPU to tensor.
 * Waits for any pending GPU operations on this tensor first.
 */
void flux_gpu_tensor_write(flux_gpu_tensor_t tensor, const float *data);

/*
 * Get direct pointer to tensor data (shared memory mode).
 * WARNING: Caller must ensure no GPU operations are pending on this tensor.
 * On Apple Silicon unified memory, this provides zero-copy access.
 */
float *flux_gpu_tensor_data(flux_gpu_tensor_t tensor);

/*
 * Release a GPU tensor back to the pool.
 */
void flux_gpu_tensor_free(flux_gpu_tensor_t tensor);

/*
 * Get tensor element count.
 */
size_t flux_gpu_tensor_size(flux_gpu_tensor_t tensor);

/*
 * Check if tensor is in bf16/f16 format.
 */
int flux_gpu_tensor_is_f16(flux_gpu_tensor_t tensor);

/* ========================================================================
 * GPU Operations on Tensors - Operations that keep data on GPU
 * ======================================================================== */

/*
 * Linear layer on GPU: out = x @ W^T + b (if b != NULL)
 * x: [seq_len, in_dim]
 * W: [out_dim, in_dim]
 * b: [out_dim] (can be NULL)
 * out: [seq_len, out_dim]
 *
 * Returns a new GPU tensor with the result.
 * Does NOT sync - GPU operation is queued.
 */
flux_gpu_tensor_t flux_gpu_linear(flux_gpu_tensor_t x,
                                   const float *W, const float *b,
                                   int seq_len, int in_dim, int out_dim);

/*
 * Linear layer on GPU with bf16 weights: out = x @ W^T
 * x: [seq_len, in_dim] (f32 on GPU)
 * W_bf16: [out_dim, in_dim] (bf16, converted to f16 internally)
 * out: [seq_len, out_dim] (f32)
 * Returns a new GPU tensor with the result.
 */
flux_gpu_tensor_t flux_gpu_linear_bf16(flux_gpu_tensor_t x,
                                        const uint16_t *W_bf16,
                                        int seq_len, int in_dim, int out_dim);

/*
 * GPU linear with bf16 weights - writes to pre-allocated output tensor.
 * Same as flux_gpu_linear_bf16 but allows reusing output buffers to avoid allocation overhead.
 * out: [seq_len, out_dim] (f32, pre-allocated)
 * Returns 1 on success, 0 on failure.
 */
int flux_gpu_linear_bf16_into(flux_gpu_tensor_t out,
                              flux_gpu_tensor_t x,
                              const uint16_t *W_bf16,
                              int seq_len, int in_dim, int out_dim);

/*
 * GPU linear with bf16 weights - outputs bf16 tensor for full bf16 pipeline.
 * Uses native MPSDataTypeBFloat16.
 */
flux_gpu_tensor_t flux_gpu_linear_bf16_bf16out(flux_gpu_tensor_t x,
                                               const uint16_t *W_bf16,
                                               int seq_len, int in_dim, int out_dim);

/*
 * Allocate bf16 GPU tensor (uses half the memory of f32).
 */
flux_gpu_tensor_t flux_gpu_tensor_alloc_f16(size_t num_elements);

/*
 * BFloat16 MPS attention for bf16 GPU tensors.
 * Uses native MPSDataTypeBFloat16.
 * Q, K, V, out must all be bf16 tensors (is_f16 = 1).
 * Returns 1 on success, 0 on failure.
 */
int flux_gpu_attention_mps_bf16(flux_gpu_tensor_t out,
                                flux_gpu_tensor_t Q, flux_gpu_tensor_t K, flux_gpu_tensor_t V,
                                int seq_q, int seq_k, int num_heads, int head_dim, float scale);

/*
 * BFloat16 attention with f32 tensor interface.
 * Takes f32 GPU tensors, converts to bf16, does bf16 attention, converts back.
 * Provides 2x memory bandwidth savings while keeping rest of pipeline in f32.
 * Returns 1 on success, 0 on failure.
 */
int flux_gpu_attention_bf16(flux_gpu_tensor_t out,
                            flux_gpu_tensor_t Q, flux_gpu_tensor_t K, flux_gpu_tensor_t V,
                            int seq_q, int seq_k, int num_heads, int head_dim, float scale);

/*
 * Sync all pending GPU operations.
 * Call this before reading tensor data or at step boundaries.
 */
void flux_gpu_sync(void);

/*
 * Begin a batch of GPU operations.
 * Operations are encoded but not executed until flux_gpu_batch_end().
 */
void flux_gpu_batch_begin(void);

/*
 * End batch and execute all queued operations.
 */
void flux_gpu_batch_end(void);

/* ========================================================================
 * GPU Operation Chains - Keep data on GPU between operations
 * ======================================================================== */

/*
 * Begin an operation chain. Operations within a chain:
 * - Share the same command buffer (reduced dispatch overhead)
 * - Keep intermediate results on GPU (no CPU round-trips)
 * - Only sync at chain end
 * Must be ended with flux_gpu_chain_end().
 */
void flux_gpu_chain_begin(void);

/*
 * End an operation chain and execute all queued operations.
 * Results stay in GPU tensors until explicitly read.
 */
void flux_gpu_chain_end(void);

/*
 * Check if currently in chain mode.
 */
int flux_gpu_in_chain(void);

/* ========================================================================
 * GPU Tensor Operations - Keep data on GPU between operations
 * These functions operate on GPU tensors and keep data on GPU.
 * Use flux_gpu_batch_begin/end to batch operations efficiently.
 * ======================================================================== */

/* AdaLN normalization on GPU: out = (1 + scale) * norm(x) + shift */
void flux_gpu_adaln_norm(flux_gpu_tensor_t out, flux_gpu_tensor_t x,
                         const float *shift, const float *scale,
                         int seq, int hidden, float eps);

/* QK RMSNorm on GPU: applies RMSNorm to Q and K in-place */
void flux_gpu_qk_rms_norm(flux_gpu_tensor_t q, flux_gpu_tensor_t k,
                          const float *q_weight, const float *k_weight,
                          int seq, int heads, int head_dim, float eps);

/* RoPE 2D on GPU: applies rotary position embeddings in-place */
void flux_gpu_rope_2d(flux_gpu_tensor_t x, const float *cos_freq, const float *sin_freq,
                      int seq, int heads, int head_dim, int axis_dim);

/* Unified RoPE for text+image: applies different frequencies to text/image portions */
void flux_gpu_rope_unified(flux_gpu_tensor_t q, flux_gpu_tensor_t k,
                           const float *txt_cos, const float *txt_sin,
                           const float *img_cos, const float *img_sin,
                           int seq, int img_offset, int heads, int head_dim, int axis_dim);

/* SiLU multiply on GPU: gate = silu(gate) * up */
void flux_gpu_silu_mul(flux_gpu_tensor_t gate, flux_gpu_tensor_t up, int n);

/* Gated add on GPU: out += gate * proj */
void flux_gpu_gated_add(flux_gpu_tensor_t out, const float *gate,
                        flux_gpu_tensor_t proj, int seq, int hidden);

/* Split fused QKV+MLP output into separate tensors */
void flux_gpu_split_qkv_mlp(flux_gpu_tensor_t fused,
                            flux_gpu_tensor_t q, flux_gpu_tensor_t k, flux_gpu_tensor_t v,
                            flux_gpu_tensor_t gate, flux_gpu_tensor_t up,
                            int seq, int hidden, int mlp_hidden);

/* Concatenate attention and MLP outputs */
void flux_gpu_concat_attn_mlp(flux_gpu_tensor_t attn, flux_gpu_tensor_t mlp,
                              flux_gpu_tensor_t out, int seq, int hidden, int mlp_hidden);

/* Fused attention on GPU tensors (no transpose needed) */
int flux_gpu_attention_fused(flux_gpu_tensor_t out,
                             flux_gpu_tensor_t Q, flux_gpu_tensor_t K, flux_gpu_tensor_t V,
                             int seq_q, int seq_k, int num_heads, int head_dim, float scale);

/* Native BF16 attention on GPU tensors (all tensors must be bf16 format).
 * Uses bf16 compute shaders with f32 accumulation for numerical stability.
 * Returns 1 on success, 0 if tensors are not bf16 or shaders unavailable.
 */
int flux_gpu_attention_bf16_native(flux_gpu_tensor_t out,
                                    flux_gpu_tensor_t Q, flux_gpu_tensor_t K, flux_gpu_tensor_t V,
                                    int seq_q, int seq_k, int num_heads, int head_dim, float scale);

/* Truly fused BF16 attention on GPU tensors - no intermediate score storage.
 * Uses custom Metal kernel with bf16 I/O and f32 internal computation.
 * Returns 1 on success, 0 if tensors are not bf16, seq_k > 1024, or shaders unavailable.
 */
int flux_gpu_attention_fused_bf16(flux_gpu_tensor_t out,
                                   flux_gpu_tensor_t Q, flux_gpu_tensor_t K, flux_gpu_tensor_t V,
                                   int seq_q, int seq_k, int num_heads, int head_dim, float scale);

/* ========================================================================
 * BF16 GPU Tensor Operations
 * All operations work on bf16 tensors (is_f16 = 1) with f32 internal computation.
 * ======================================================================== */

/* BF16 AdaLN normalization */
void flux_gpu_adaln_norm_bf16(flux_gpu_tensor_t out, flux_gpu_tensor_t x,
                               flux_gpu_tensor_t shift_bf16, flux_gpu_tensor_t scale_bf16,
                               int seq, int hidden, float eps);

/* BF16 QK RMSNorm (in-place) */
void flux_gpu_qk_rms_norm_bf16(flux_gpu_tensor_t q, flux_gpu_tensor_t k,
                                flux_gpu_tensor_t q_weight_bf16, flux_gpu_tensor_t k_weight_bf16,
                                int seq, int heads, int head_dim, float eps);

/* BF16 per-head RMSNorm (single tensor, for GQA with different Q/K head counts) */
int flux_gpu_head_rms_norm_bf16(flux_gpu_tensor_t x, flux_gpu_tensor_t weight_bf16,
                                 int seq, int heads, int head_dim, float eps);

/* BF16 RMS Norm: out = rms_norm(x) * weight */
void flux_gpu_rms_norm_bf16(flux_gpu_tensor_t out, flux_gpu_tensor_t x,
                             flux_gpu_tensor_t weight, int seq, int hidden, float eps);

/* BF16 element-wise add: out = a + b */
void flux_gpu_add_bf16(flux_gpu_tensor_t out, flux_gpu_tensor_t a, flux_gpu_tensor_t b, int n);

/* BF16 SiLU multiply: gate = silu(gate) * up */
void flux_gpu_silu_mul_bf16(flux_gpu_tensor_t gate, flux_gpu_tensor_t up, int n);

/* BF16 Gated add: out += gate * proj */
void flux_gpu_gated_add_bf16(flux_gpu_tensor_t out, flux_gpu_tensor_t gate_bf16,
                              flux_gpu_tensor_t proj, int seq, int hidden);

/* BF16 RoPE unified (text + image) */
void flux_gpu_rope_unified_bf16(flux_gpu_tensor_t q, flux_gpu_tensor_t k,
                                 const float *txt_cos, const float *txt_sin,
                                 const float *img_cos, const float *img_sin,
                                 int seq, int img_offset, int heads, int head_dim, int axis_dim);

/* BF16 RoPE 2D (single stream) */
void flux_gpu_rope_2d_bf16(flux_gpu_tensor_t x,
                            const float *cos_freq, const float *sin_freq,
                            int seq, int heads, int head_dim, int axis_dim);

/* BF16 Causal Attention with GQA support (for text encoder)
 * Q: [seq, num_q_heads * head_dim] (bf16)
 * K, V: [seq, num_kv_heads * head_dim] (bf16)
 * out: [seq, num_q_heads * head_dim] (bf16)
 * attention_mask: [seq] - 1 for valid, 0 for padding (can be NULL)
 * Supports GQA where num_q_heads > num_kv_heads.
 * Returns 1 on success, 0 on failure.
 */
int flux_gpu_causal_attention_bf16(flux_gpu_tensor_t out,
                                    flux_gpu_tensor_t Q, flux_gpu_tensor_t K, flux_gpu_tensor_t V,
                                    const int *attention_mask,
                                    int seq, int num_q_heads, int num_kv_heads,
                                    int head_dim, float scale);

/* BF16 RoPE for text encoder (Qwen3 style)
 * Q: [seq, num_q_heads * head_dim] (bf16) - modified in-place
 * K: [seq, num_kv_heads * head_dim] (bf16) - modified in-place
 * cos_cache, sin_cache: [seq, head_dim/2] (f32) - precomputed
 */
void flux_gpu_rope_text_bf16(flux_gpu_tensor_t q, flux_gpu_tensor_t k,
                              const float *cos_cache, const float *sin_cache,
                              int seq, int num_q_heads, int num_kv_heads, int head_dim);

/* Concatenate two bf16 sequences along seq dimension */
void flux_gpu_concat_seq_bf16(flux_gpu_tensor_t out,
                               flux_gpu_tensor_t a, flux_gpu_tensor_t b,
                               int seq_a, int seq_b, int hidden);

/* Slice a bf16 sequence along seq dimension */
void flux_gpu_slice_seq_bf16(flux_gpu_tensor_t out,
                              flux_gpu_tensor_t in,
                              int seq_out, int hidden, int start);

/* BF16 Split QKV+MLP output */
void flux_gpu_split_qkv_mlp_bf16(flux_gpu_tensor_t fused,
                                  flux_gpu_tensor_t q, flux_gpu_tensor_t k, flux_gpu_tensor_t v,
                                  flux_gpu_tensor_t gate, flux_gpu_tensor_t up,
                                  int seq, int hidden, int mlp_hidden);

/* BF16 Concat attention + MLP outputs */
void flux_gpu_concat_attn_mlp_bf16(flux_gpu_tensor_t attn, flux_gpu_tensor_t mlp,
                                    flux_gpu_tensor_t out, int seq, int hidden, int mlp_hidden);

/* Convert f32 GPU tensor to bf16 (returns new tensor) */
flux_gpu_tensor_t flux_gpu_tensor_f32_to_bf16(flux_gpu_tensor_t f32_tensor);

/* Convert bf16 GPU tensor to f32 (returns new tensor) */
flux_gpu_tensor_t flux_gpu_tensor_bf16_to_f32(flux_gpu_tensor_t bf16_tensor);

/* BF16 native linear layer (all bf16) */
flux_gpu_tensor_t flux_gpu_linear_bf16_native(flux_gpu_tensor_t x,
                                               const uint16_t *W_bf16,
                                               int seq_len, int in_dim, int out_dim);

/* BF16 native linear layer writing into a pre-allocated bf16 output tensor.
 * Returns 1 on success, 0 on failure. */
int flux_gpu_linear_bf16_native_into(flux_gpu_tensor_t out,
                                     flux_gpu_tensor_t x,
                                     const uint16_t *W_bf16,
                                     int seq_len, int in_dim, int out_dim);

/* BF16 Transpose for attention: [seq, heads*head_dim] -> [heads, seq, head_dim] */
void flux_gpu_transpose_to_heads_bf16(flux_gpu_tensor_t in, flux_gpu_tensor_t out,
                                       int seq, int heads, int head_dim);

/* BF16 Transpose for attention output: [heads, seq, head_dim] -> [seq, heads*head_dim] */
void flux_gpu_transpose_from_heads_bf16(flux_gpu_tensor_t in, flux_gpu_tensor_t out,
                                         int seq, int heads, int head_dim);

/*
 * GPU-accelerated scaled dot-product attention.
 * Computes attention for all heads in a single GPU batch.
 *
 * Q, K, V are in [heads, seq_q/seq_k, head_dim] layout (already transposed)
 * scores_scratch must be pre-allocated: [heads * seq_q * seq_k] floats
 * out will be [heads, seq_q, head_dim]
 *
 * This does: out = softmax(Q @ K^T * scale) @ V
 * Softmax is done on CPU (between two GPU batches).
 */
void flux_metal_attention(float *out,
                          const float *Q, const float *K, const float *V,
                          float *scores_scratch,
                          int heads, int seq_q, int seq_k, int head_dim,
                          float scale);

/*
 * Half-precision version of flux_metal_attention.
 * Same interface but uses f16 MPS matmuls internally for ~2x bandwidth savings.
 * Takes f32 inputs, converts to f16, computes attention, converts back to f32.
 * scores_scratch is unused (kept for interface compatibility).
 */
void flux_metal_attention_bf16(float *out,
                               const float *Q, const float *K, const float *V,
                               float *scores_scratch,
                               int heads, int seq_q, int seq_k, int head_dim,
                               float scale);

/*
 * GPU-accelerated causal attention for text encoder (Qwen3).
 * Processes all heads in parallel on GPU with causal masking.
 * Supports GQA (Grouped Query Attention) where Q heads > KV heads.
 *
 * Q: [seq, num_q_heads * head_dim] - query tensor
 * K: [seq, num_kv_heads * head_dim] - key tensor (may have fewer heads)
 * V: [seq, num_kv_heads * head_dim] - value tensor
 * out: [seq, num_q_heads * head_dim] - output tensor
 * attention_mask: [seq] - 1 for valid tokens, 0 for padding (can be NULL)
 *
 * This does: out = softmax(Q @ K^T * scale + causal_mask + attn_mask) @ V
 * All operations are fused in a single GPU kernel.
 * Returns 1 on success, 0 on failure (falls back to CPU).
 */
int flux_metal_causal_attention(float *out,
                                 const float *Q, const float *K, const float *V,
                                 const int *attention_mask,
                                 int seq, int num_q_heads, int num_kv_heads,
                                 int head_dim, float scale);

/*
 * Fused non-causal attention for transformer.
 * Works directly on [seq, hidden] layout without transpose.
 * Supports different Q and K/V sequence lengths (for joint attention).
 *
 * This does: out = softmax(Q @ K^T * scale) @ V
 * All operations are fused in a single GPU kernel.
 * Returns 1 on success, 0 on failure (falls back to CPU).
 */
int flux_metal_attention_fused(float *out,
                               const float *Q, const float *K, const float *V,
                               int seq_q, int seq_k, int num_heads, int head_dim,
                               float scale);

/* ========================================================================
 * GPU Compute Shaders - Element-wise operations on GPU
 * ======================================================================== */

/*
 * Initialize compute shaders from .metal file.
 * Called automatically by flux_metal_init() if shader file exists.
 * Returns 1 on success, 0 on failure.
 */
int flux_metal_init_shaders(void);

/*
 * GPU-accelerated RMSNorm.
 * out[i] = x[i] * rsqrt(mean(x^2) + eps) * weight[i]
 * x: [seq_len, hidden], weight: [hidden], out: [seq_len, hidden]
 */
void flux_metal_rms_norm(float *out, const float *x, const float *weight,
                         int seq_len, int hidden, float eps);

/*
 * GPU-accelerated QK RMSNorm (in-place).
 * Normalizes Q and K separately for each head.
 * q, k: [seq, heads*head_dim] (modified in-place)
 * q_weight, k_weight: [head_dim]
 */
void flux_metal_qk_rms_norm(float *q, float *k,
                            const float *q_weight, const float *k_weight,
                            int seq, int heads, int head_dim, float eps);

/*
 * GPU-accelerated LayerNorm + AdaLN modulation.
 * out = (1 + scale) * layernorm(x) + shift
 * x: [seq_len, hidden], shift/scale: [hidden]
 */
void flux_metal_adaln_norm(float *out, const float *x,
                           const float *shift, const float *scale,
                           int seq_len, int hidden, float eps);

/*
 * GPU-accelerated SiLU activation (in-place).
 * x = x * sigmoid(x)
 */
void flux_metal_silu(float *x, int n);

/*
 * GPU-accelerated SiLU with multiply (SwiGLU style, in-place).
 * gate = silu(gate) * up
 */
void flux_metal_silu_mul(float *gate, const float *up, int n);

/*
 * GPU-accelerated softmax (row-wise, in-place).
 * x: [rows, cols], softmax applied to each row
 */
void flux_metal_softmax(float *x, int rows, int cols);

/*
 * GPU-accelerated 2D RoPE (in-place).
 * x: [seq, heads*head_dim]
 * cos_freq, sin_freq: [seq, head_dim]
 */
void flux_metal_rope_2d(float *x, const float *cos_freq, const float *sin_freq,
                        int seq, int heads, int head_dim, int axis_dim);

/*
 * Check if compute shaders are available.
 */
int flux_metal_shaders_available(void);

/*
 * Pre-warm the bf16->f16 conversion cache for a weight tensor.
 * Call this during model loading to avoid conversion overhead during inference.
 * This converts bf16 weights to f16 and caches the result.
 */
void flux_metal_warmup_bf16(const uint16_t *bf16_weights, size_t num_elements);

/* ========================================================================
 * Native BF16 Pipeline API
 *
 * These functions work with native bf16 GPU buffers to implement a full
 * bf16 pipeline. All operations keep data in bf16
 * with f32 accumulation internally for numerical stability.
 *
 * To use this API from C code, include flux_metal.h and link with flux_metal.m
 * The MTLBuffer pointers should be obtained from flux_gpu_tensor via
 * flux_gpu_tensor_get_buffer() or created directly using Metal API.
 * ======================================================================== */

/*
 * Check if bf16 pipeline is available (all required shaders loaded).
 */
int flux_bf16_pipeline_available(void);

#ifdef __OBJC__
#import <Metal/Metal.h>

/*
 * Native BF16 attention (no conversion overhead).
 * All buffers contain bf16 data, f32 accumulation happens internally.
 * Q: [heads, seq_q, head_dim] (bf16)
 * K: [heads, seq_k, head_dim] (bf16)
 * V: [heads, seq_k, head_dim] (bf16)
 * out: [heads, seq_q, head_dim] (bf16)
 */
void flux_metal_attention_bf16_native(id<MTLBuffer> bufQ, id<MTLBuffer> bufK,
                                       id<MTLBuffer> bufV, id<MTLBuffer> bufOut,
                                       int heads, int seq_q, int seq_k, int head_dim,
                                       float scale);

/* Convert f32 GPU buffer to bf16 */
void flux_bf16_convert_f32_to_bf16(id<MTLBuffer> input_f32, id<MTLBuffer> output_bf16, int n);

/* Convert bf16 GPU buffer to f32 */
void flux_bf16_convert_bf16_to_f32(id<MTLBuffer> input_bf16, id<MTLBuffer> output_f32, int n);

/* RMSNorm on bf16 buffers */
void flux_bf16_rms_norm(id<MTLBuffer> out, id<MTLBuffer> x, id<MTLBuffer> weight,
                         int seq_len, int hidden, float eps);

/* QK RMSNorm on bf16 buffers (in-place) */
void flux_bf16_qk_rms_norm(id<MTLBuffer> q, id<MTLBuffer> k,
                            id<MTLBuffer> q_weight, id<MTLBuffer> k_weight,
                            int seq, int heads, int head_dim, float eps);

/* SiLU on bf16 buffer (in-place) */
void flux_bf16_silu(id<MTLBuffer> x, int n);

/* SiLU with multiply on bf16 buffers: gate = silu(gate) * up */
void flux_bf16_silu_mul(id<MTLBuffer> gate, id<MTLBuffer> up, int n);

/* RoPE on bf16 buffer (frequencies are f32) */
void flux_bf16_rope_unified(id<MTLBuffer> x,
                             const float *txt_cos, const float *txt_sin,
                             const float *img_cos, const float *img_sin,
                             int seq, int img_offset, int heads, int head_dim);

#endif /* __OBJC__ */

#ifdef __cplusplus
}
#endif

#endif /* FLUX_METAL_H */
