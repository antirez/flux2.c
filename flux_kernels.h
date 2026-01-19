/*
 * FLUX Math Kernels - Header
 *
 * Low-level math operations for the FLUX inference engine.
 * All operations work on float32 tensors in row-major order.
 */

#ifndef FLUX_KERNELS_H
#define FLUX_KERNELS_H

#include <stddef.h>
#include <stdint.h>

/* ========================================================================
 * Basic Operations
 * ======================================================================== */

/* Element-wise operations */
void flux_add(float *out, const float *a, const float *b, int n);
void flux_add_scalar(float *out, const float *a, float s, int n);
void flux_sub(float *out, const float *a, const float *b, int n);
void flux_mul(float *out, const float *a, const float *b, int n);
void flux_mul_scalar(float *out, const float *a, float s, int n);
void flux_div(float *out, const float *a, const float *b, int n);

/* In-place variants */
void flux_add_inplace(float *a, const float *b, int n);
void flux_mul_inplace(float *a, const float *b, int n);
void flux_scale_inplace(float *a, float s, int n);

/* Accumulate: a += scale * b */
void flux_axpy(float *a, float scale, const float *b, int n);

/* ========================================================================
 * Matrix Operations
 * ======================================================================== */

/*
 * General matrix multiplication: C = A @ B
 * A: [M, K], B: [K, N], C: [M, N]
 */
void flux_matmul(float *C, const float *A, const float *B,
                 int M, int K, int N);

/*
 * Matrix multiplication with transposed B: C = A @ B^T
 * A: [M, K], B: [N, K], C: [M, N]
 */
void flux_matmul_t(float *C, const float *A, const float *B,
                   int M, int K, int N);

/*
 * Batched matrix multiplication: C[b] = A[b] @ B[b]
 */
void flux_batched_matmul(float *C, const float *A, const float *B,
                         int batch, int M, int K, int N);

/*
 * Linear layer: y = x @ W^T + b (if b != NULL)
 * x: [seq_len, in_dim], W: [out_dim, in_dim], b: [out_dim], y: [seq_len, out_dim]
 */
void flux_linear(float *y, const float *x, const float *W, const float *b,
                 int seq_len, int in_dim, int out_dim);

/*
 * Linear layer without bias
 */
void flux_linear_nobias(float *y, const float *x, const float *W,
                        int seq_len, int in_dim, int out_dim);

/*
 * Linear layer without bias using bf16 weights
 * x: [seq_len, in_dim] (f32), W: [out_dim, in_dim] (bf16), y: [seq_len, out_dim] (f32)
 * Provides 2x memory bandwidth improvement for weight-bound operations.
 */
void flux_linear_nobias_bf16(float *y, const float *x, const uint16_t *W_bf16,
                             int seq_len, int in_dim, int out_dim);

/* ========================================================================
 * GPU Batch Operations
 * These functions allow batching multiple GPU operations to reduce sync overhead.
 * On non-GPU builds, these are no-ops.
 * ======================================================================== */

/*
 * Begin a batch of GPU operations.
 * Operations after this call are queued but not executed until flux_gpu_end_batch().
 * NOTE: Only use for INDEPENDENT operations (outputs don't feed into subsequent inputs).
 */
void flux_gpu_begin_batch(void);

/*
 * End a batch of GPU operations.
 * Executes all queued operations and waits for completion.
 */
void flux_gpu_end_batch(void);

/*
 * Check if GPU batch mode is currently active.
 */
int flux_gpu_in_batch(void);

/* ========================================================================
 * Convolution Operations
 * ======================================================================== */

/*
 * 2D Convolution: out = conv2d(in, weight, bias)
 * in: [batch, in_ch, H, W]
 * weight: [out_ch, in_ch, kH, kW]
 * bias: [out_ch] (can be NULL)
 * out: [batch, out_ch, outH, outW]
 */
void flux_conv2d(float *out, const float *in, const float *weight, const float *bias,
                 int batch, int in_ch, int out_ch, int H, int W,
                 int kH, int kW, int stride, int padding);

/*
 * Transposed 2D Convolution (for upsampling)
 */
void flux_conv2d_transpose(float *out, const float *in, const float *weight, const float *bias,
                           int batch, int in_ch, int out_ch, int H, int W,
                           int kH, int kW, int stride, int padding, int output_padding);

/*
 * Depthwise separable convolution
 */
void flux_conv2d_depthwise(float *out, const float *in, const float *weight, const float *bias,
                           int batch, int channels, int H, int W,
                           int kH, int kW, int stride, int padding);

/* ========================================================================
 * Normalization
 * ======================================================================== */

/*
 * Layer Normalization
 * x: [seq_len, hidden], gamma/beta: [hidden]
 */
void flux_layer_norm(float *out, const float *x, const float *gamma, const float *beta,
                     int seq_len, int hidden, float eps);

/*
 * RMS Normalization (no mean centering, no bias)
 * x: [seq_len, hidden], weight: [hidden]
 */
void flux_rms_norm(float *out, const float *x, const float *weight,
                   int seq_len, int hidden, float eps);

/*
 * Group Normalization
 * x: [batch, channels, H, W], gamma/beta: [channels]
 */
void flux_group_norm(float *out, const float *x, const float *gamma, const float *beta,
                     int batch, int channels, int H, int W, int num_groups, float eps);

/*
 * Batch Normalization (inference mode with running stats)
 * x: [batch, channels, H, W]
 */
void flux_batch_norm(float *out, const float *x,
                     const float *running_mean, const float *running_var,
                     const float *gamma, const float *beta,
                     int batch, int channels, int H, int W, float eps);

/* ========================================================================
 * Activation Functions
 * ======================================================================== */

/* GELU activation (approximate) */
void flux_gelu(float *x, int n);

/* SiLU / Swish activation: x * sigmoid(x) */
void flux_silu(float *x, int n);

/* SwiGLU: gate * silu(x) where input is [x, gate] concatenated */
void flux_swiglu(float *out, const float *x, const float *gate, int n);

/* Softmax over last dimension */
void flux_softmax(float *x, int rows, int cols);

/* Sigmoid */
void flux_sigmoid(float *x, int n);

/* Tanh */
void flux_tanh(float *x, int n);

/* ========================================================================
 * Attention Operations
 * ======================================================================== */

/*
 * Scaled dot-product attention
 * Q: [batch, heads, seq_q, head_dim]
 * K: [batch, heads, seq_k, head_dim]
 * V: [batch, heads, seq_k, head_dim]
 * out: [batch, heads, seq_q, head_dim]
 * scale: typically 1/sqrt(head_dim)
 */
void flux_attention(float *out, const float *Q, const float *K, const float *V,
                    int batch, int heads, int seq_q, int seq_k, int head_dim,
                    float scale);

/*
 * Attention with mask
 * mask: [seq_q, seq_k] or NULL, 0 = attend, -inf = mask out
 */
void flux_attention_masked(float *out, const float *Q, const float *K, const float *V,
                           const float *mask,
                           int batch, int heads, int seq_q, int seq_k, int head_dim,
                           float scale);

/*
 * Flash attention - memory-efficient tiled attention.
 * Uses online softmax to avoid materializing O(n²) attention matrix.
 * Memory: O(seq_q + tile_size²) instead of O(seq_q × seq_k).
 *
 * Works on [seq, heads*head_dim] layout (same as transformer tensors).
 * Q: [seq_q, heads * head_dim]
 * K: [seq_k, heads * head_dim]
 * V: [seq_k, heads * head_dim]
 * out: [seq_q, heads * head_dim]
 */
void flux_flash_attention(float *out, const float *Q, const float *K, const float *V,
                          int seq_q, int seq_k, int heads, int head_dim, float scale);

/*
 * Apply rotary position embeddings (RoPE)
 * x: [batch, seq, heads, head_dim]
 * freqs: [seq, head_dim/2, 2] (cos, sin pairs)
 */
void flux_apply_rope(float *x, const float *freqs,
                     int batch, int seq, int heads, int head_dim);

/*
 * Compute RoPE frequencies
 * pos: position indices [seq]
 * freqs: output [seq, dim/2, 2]
 */
void flux_compute_rope_freqs(float *freqs, const int *pos, int seq, int dim, float theta);

/* ========================================================================
 * Pooling and Reshape
 * ======================================================================== */

/* Average pooling 2D */
void flux_avgpool2d(float *out, const float *in,
                    int batch, int channels, int H, int W,
                    int kH, int kW, int stride, int padding);

/* Max pooling 2D */
void flux_maxpool2d(float *out, const float *in,
                    int batch, int channels, int H, int W,
                    int kH, int kW, int stride, int padding);

/* Upsample with nearest neighbor */
void flux_upsample_nearest(float *out, const float *in,
                           int batch, int channels, int H, int W,
                           int scale_h, int scale_w);

/* Upsample with bilinear interpolation */
void flux_upsample_bilinear(float *out, const float *in,
                            int batch, int channels, int H, int W,
                            int out_H, int out_W);

/* Patchify: [B, C, H, W] -> [B, C*p*p, H/p, W/p] */
void flux_patchify(float *out, const float *in,
                   int batch, int channels, int H, int W, int patch_size);

/* Unpatchify: [B, C*p*p, H, W] -> [B, C, H*p, W*p] */
void flux_unpatchify(float *out, const float *in,
                     int batch, int channels, int H, int W, int patch_size);

/* ========================================================================
 * Random Number Generation
 * ======================================================================== */

/* Initialize RNG with seed */
void flux_rng_seed(uint64_t seed);

/* Generate uniform random [0, 1) */
float flux_random_uniform(void);

/* Generate standard normal using Box-Muller */
float flux_random_normal(void);

/* Fill tensor with random normal values */
void flux_randn(float *out, int n);

/* Fill tensor with uniform random [0, 1) */
void flux_rand(float *out, int n);

/* ========================================================================
 * Utility Functions
 * ======================================================================== */

/* Copy tensor */
void flux_copy(float *dst, const float *src, int n);

/* Fill tensor with value */
void flux_fill(float *x, float val, int n);

/* Sum all elements */
float flux_sum(const float *x, int n);

/* Mean of all elements */
float flux_mean(const float *x, int n);

/* Variance of all elements */
float flux_var(const float *x, int n);

/* L2 norm */
float flux_norm(const float *x, int n);

/* Dot product */
float flux_dot(const float *a, const float *b, int n);

/* Clamp values */
void flux_clamp(float *x, float min_val, float max_val, int n);

/* Transpose 2D: [M, N] -> [N, M] */
void flux_transpose(float *out, const float *in, int M, int N);

/* Reshape (just changes interpretation, no data movement) */
/* ... handled at higher level ... */

/* ========================================================================
 * Progress Callbacks
 * ======================================================================== */

/* Substep types during transformer forward pass */
typedef enum {
    FLUX_SUBSTEP_DOUBLE_BLOCK,   /* Double-stream block completed */
    FLUX_SUBSTEP_SINGLE_BLOCK,   /* Single-stream block completed */
    FLUX_SUBSTEP_FINAL_LAYER,    /* Final layer completed */
} flux_substep_type_t;

/*
 * Substep callback - called during transformer forward pass.
 * type: which operation completed
 * index: 0-based index of this substep within its type
 * total: total count for this substep type
 */
typedef void (*flux_substep_callback_t)(flux_substep_type_t type, int index, int total);

/*
 * Step callback - called at sampling step boundaries.
 * step: current step (1-based), or 0 to indicate sampling is starting
 * total: total number of steps
 */
typedef void (*flux_step_callback_t)(int step, int total);

/* Global callback pointers - set by caller before inference */
extern flux_substep_callback_t flux_substep_callback;
extern flux_step_callback_t flux_step_callback;

/*
 * Phase callback - called at major phase boundaries.
 * phase: descriptive name ("encoding text", "decoding image", etc.)
 * done: 0 when starting, 1 when finished
 */
typedef void (*flux_phase_callback_t)(const char *phase, int done);
extern flux_phase_callback_t flux_phase_callback;

#endif /* FLUX_KERNELS_H */
