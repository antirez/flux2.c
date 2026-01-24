/*
 * FLUX CUDA Acceleration
 *
 * GPU-accelerated matrix operations using NVIDIA CUDA and cuBLAS.
 * Provides significant speedup on NVIDIA GPUs.
 *
 * Inspired by stable-diffusion.cpp's GGML CUDA backend, but standalone.
 */

#ifndef FLUX_CUDA_H
#define FLUX_CUDA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Initialize CUDA acceleration.
 * Returns 1 on success, 0 if CUDA is not available.
 * Safe to call multiple times.
 */
int flux_cuda_init(void);

/*
 * Check if CUDA acceleration is available and initialized.
 */
int flux_cuda_available(void);

/*
 * Cleanup CUDA resources.
 */
void flux_cuda_cleanup(void);

/*
 * Reset all GPU state (caches, pools, pending commands).
 * Call this between independent inference phases.
 */
void flux_cuda_reset(void);

/*
 * GPU-accelerated matrix multiplication using cuBLAS.
 * C[M,N] = alpha * A[M,K] @ B[K,N] + beta * C[M,N]
 *
 * transpose_a: if non-zero, use A^T
 * transpose_b: if non-zero, use B^T
 */
void flux_cuda_sgemm(int transpose_a, int transpose_b,
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
void flux_cuda_sgemm_bf16(int transpose_a, int transpose_b,
                          int M, int N, int K,
                          float alpha,
                          const float *A, int lda,
                          const uint16_t *B_bf16, int ldb,
                          float beta,
                          float *C, int ldc);

/*
 * 2D convolution using cuDNN (if available) or im2col+cuBLAS.
 * Returns 1 on success, 0 on failure.
 */
int flux_cuda_conv2d(float *out, const float *in,
                     const float *weight, const float *bias,
                     int batch, int in_ch, int out_ch,
                     int H, int W, int kH, int kW,
                     int stride, int padding);

/*
 * Batch matrix multiplication on GPU.
 * Performs batch_count independent matrix multiplications.
 */
void flux_cuda_sgemm_batch(int transpose_a, int transpose_b,
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
void flux_cuda_sync(void);

/*
 * Begin a batch of GPU operations.
 * Operations after this call are queued but not executed until flux_cuda_end_batch().
 */
void flux_cuda_begin_batch(void);

/*
 * End a batch of GPU operations.
 * Executes all queued operations and waits for completion.
 */
void flux_cuda_end_batch(void);

/*
 * Check if currently in batch mode.
 */
int flux_cuda_in_batch(void);

/*
 * Get GPU memory usage info (for debugging).
 */
size_t flux_cuda_memory_used(void);

/*
 * Fused attention on GPU.
 * Computes attention for all heads in a single GPU batch.
 *
 * Q, K, V are in [seq, heads*head_dim] layout
 * out: [seq_q, heads * head_dim]
 *
 * This does: out = softmax(Q @ K^T * scale) @ V
 * Returns 1 on success, 0 on failure (falls back to CPU).
 */
int flux_cuda_attention_fused(float *out,
                              const float *Q, const float *K, const float *V,
                              int seq_q, int seq_k, int num_heads, int head_dim,
                              float scale);

/*
 * GPU-accelerated causal attention for text encoder.
 * Supports GQA (Grouped Query Attention).
 * Returns 1 on success, 0 on failure.
 */
int flux_cuda_causal_attention(float *out,
                               const float *Q, const float *K, const float *V,
                               const int *attention_mask,
                               int seq, int num_q_heads, int num_kv_heads,
                               int head_dim, float scale);

/*
 * Check if compute kernels are available.
 */
int flux_cuda_kernels_available(void);

/*
 * Get CUDA device name for display.
 */
const char* flux_cuda_device_name(void);

/*
 * Get CUDA compute capability.
 */
int flux_cuda_compute_capability(void);

/* ========================================================================
 * GPU Tensor Pool - Keep data on GPU between operations
 * ======================================================================== */

/*
 * Get a GPU tensor from the pool (allocates if needed).
 * Returns tensor ID or -1 on error.
 */
int flux_cuda_tensor_get(size_t size_bytes);

/*
 * Release a tensor back to the pool.
 */
void flux_cuda_tensor_release(int tensor_id);

/*
 * Get raw GPU pointer for a tensor.
 */
float* flux_cuda_tensor_ptr(int tensor_id);

/*
 * Upload CPU data to a GPU tensor.
 */
void flux_cuda_tensor_upload(int tensor_id, const float *data, size_t size);

/*
 * Download GPU tensor data to CPU.
 */
void flux_cuda_tensor_download(int tensor_id, float *data, size_t size);
void flux_cuda_memcpy_d2d(int dst_id, size_t dst_offset, int src_id, size_t src_offset, size_t size);

/*
 * GPU-to-GPU sgemm. A_id and C_id are tensor IDs, B is weight pointer.
 * Returns C_id on success, -1 on error.
 */
int flux_cuda_sgemm_gpu(int ta, int tb, int M, int N, int K,
                        float alpha, int A_id, int lda,
                        const float *B, int ldb,
                        float beta, int C_id, int ldc);

/*
 * GPU-to-GPU sgemm with bf16 weights. Converts bf16→f32 on GPU then matmul.
 * A_id and C_id are tensor IDs, B_bf16 is bf16 weight pointer.
 * Returns C_id on success, -1 on error.
 */
int flux_cuda_sgemm_gpu_bf16(int ta, int tb, int M, int N, int K,
                              float alpha, int A_id, int lda,
                              const uint16_t *B_bf16, int ldb,
                              float beta, int C_id, int ldc);

/* GPU Tensor operations - work directly on GPU tensors */
void flux_cuda_gated_add_t(int out_id, const float *gate, int x_id, int seq, int hidden);
void flux_cuda_split_fused_t(int fused_id, int q_id, int k_id, int v_id,
                             int gate_id, int up_id, int seq, int h, int mlp);
void flux_cuda_concat_t(int concat_id, int attn_id, int mlp_id, int seq, int h, int mlp);
void flux_cuda_silu_t(int tensor_id, int n);
void flux_cuda_mul_t(int a_id, int b_id, int n);
void flux_cuda_adaln_t(int out_id, int x_id, const float *shift, const float *scale,
                       int seq, int hid, float eps);
void flux_cuda_qk_norm_t(int q_id, int k_id, const float *qw, const float *kw,
                         int seq, int heads, int hdim, float eps);
/* RoPE 2D full head_dim version - uses tensor pool */
void flux_cuda_rope_2d_full_t(int x_id, const float *cos_f, const float *sin_f,
                               int seq, int heads, int hdim);
void flux_cuda_rope_t(int x_id, const float *cos_f, const float *sin_f,
                      int seq, int heads, int hdim, int axis_dim);
void flux_cuda_rope_offset_t(int x_id, const float *cos_f, const float *sin_f,
                              int seq_len, int seq_offset, int heads, int hdim, int axis_dim);

/* GPU attention using batched cuBLAS gemm
 * Q,K,V,out are tensor IDs with layout [seq, heads, hdim]
 * Returns 1 on success, 0 on failure
 */
int flux_cuda_attention_t(int out_id, int q_id, int k_id, int v_id,
                          int seq, int heads, int hdim, float scale);

/* Joint attention for double blocks - img and txt Q attend to concatenated K,V */
int flux_cuda_joint_attention_t(int img_out_id, int txt_out_id,
                                 int img_q_id, int txt_q_id,
                                 int cat_k_id, int cat_v_id,
                                 int img_seq, int txt_seq, int heads, int hdim, float scale);

/*
 * Clear the GPU weight cache.
 * Must be called when weights are freed/reallocated (mmap mode).
 */
void flux_cuda_weight_cache_clear(void);

/*
 * Disable/enable the GPU weight cache.
 * Call with disable=1 for mmap mode (weights change addresses).
 */
void flux_cuda_weight_cache_disable(int disable);

#ifdef __cplusplus
}
#endif

#endif /* FLUX_CUDA_H */
