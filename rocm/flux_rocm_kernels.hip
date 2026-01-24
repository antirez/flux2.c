/*
 * FLUX ROCm Custom Kernels
 *
 * HIP kernels for operations that aren't covered by hipBLAS.
 * These are the building blocks for transformer inference.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <math.h>
#include <float.h>

/* ========================================================================
 * Helper Macros
 * ======================================================================== */

/* RDNA uses 32-wide wavefronts (not 64 like GCN/CDNA) */
#define WARP_SIZE 32
#define MAX_THREADS 256
#define MAX_WARPS (MAX_THREADS / WARP_SIZE)

/* Warp-level reduction */
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down(val, offset));
    }
    return val;
}

/* Block-level reduction using shared memory */
__device__ float block_reduce_sum(float val) {
    __shared__ float shared[MAX_WARPS];
    
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    
    /* Warp-level reduction */
    val = warp_reduce_sum(val);
    
    /* Store warp results */
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    /* Final reduction in first warp */
    if (threadIdx.x < num_warps) {
        val = shared[threadIdx.x];
    } else {
        val = 0.0f;
    }
    
    if (wid == 0) {
        val = warp_reduce_sum(val);
    }
    
    return val;
}

__device__ float block_reduce_max(float val) {
    __shared__ float shared[MAX_WARPS];
    
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    
    val = warp_reduce_max(val);
    
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    if (threadIdx.x < num_warps) {
        val = shared[threadIdx.x];
    } else {
        val = -FLT_MAX;
    }
    
    if (wid == 0) {
        val = warp_reduce_max(val);
    }
    
    return val;
}

/* ========================================================================
 * SiLU Activation: x * sigmoid(x)
 * ======================================================================== */

__global__ void kernel_silu(float *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float val = x[i];
        x[i] = val / (1.0f + expf(-val));
    }
}

__global__ void kernel_silu_mul(float *gate, const float *up, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = gate[i];
        float u = up[i];
        gate[i] = (g / (1.0f + expf(-g))) * u;
    }
}

/* ========================================================================
 * RMSNorm: out = x * rsqrt(mean(x^2) + eps) * weight
 * ======================================================================== */

__global__ void kernel_rms_norm(float *out, const float *x, const float *weight,
                                int seq_len, int hidden, float eps) {
    /* One block per sequence position */
    int seq_idx = blockIdx.x;
    if (seq_idx >= seq_len) return;
    
    const float *x_row = x + seq_idx * hidden;
    float *out_row = out + seq_idx * hidden;
    
    /* Compute sum of squares */
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float val = x_row[i];
        sum_sq += val * val;
    }
    sum_sq = block_reduce_sum(sum_sq);
    
    /* Broadcast RMS to all threads */
    __shared__ float s_rms_inv;
    if (threadIdx.x == 0) {
        s_rms_inv = rsqrtf(sum_sq / hidden + eps);
    }
    __syncthreads();
    
    /* Apply normalization and weight */
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        out_row[i] = x_row[i] * s_rms_inv * weight[i];
    }
}

/* ========================================================================
 * QK RMSNorm: Per-head normalization for Q and K
 * q, k: [seq, heads * head_dim]
 * weight: [head_dim]
 * ======================================================================== */

__global__ void kernel_qk_rms_norm(float *q, float *k,
                                   const float *q_weight, const float *k_weight,
                                   int seq, int heads, int head_dim, float eps) {
    /* One block per (seq_idx, head_idx) */
    int idx = blockIdx.x;
    int total = seq * heads;
    if (idx >= total) return;
    
    int seq_idx = idx / heads;
    int head_idx = idx % heads;
    
    int offset = seq_idx * (heads * head_dim) + head_idx * head_dim;
    float *q_head = q + offset;
    float *k_head = k + offset;
    
    /* RMSNorm for Q */
    float sum_sq_q = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float val = q_head[i];
        sum_sq_q += val * val;
    }
    sum_sq_q = block_reduce_sum(sum_sq_q);
    
    __shared__ float s_rms_inv_q;
    if (threadIdx.x == 0) {
        s_rms_inv_q = rsqrtf(sum_sq_q / head_dim + eps);
    }
    __syncthreads();
    
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        q_head[i] = q_head[i] * s_rms_inv_q * q_weight[i];
    }
    __syncthreads();
    
    /* RMSNorm for K */
    float sum_sq_k = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float val = k_head[i];
        sum_sq_k += val * val;
    }
    sum_sq_k = block_reduce_sum(sum_sq_k);
    
    __shared__ float s_rms_inv_k;
    if (threadIdx.x == 0) {
        s_rms_inv_k = rsqrtf(sum_sq_k / head_dim + eps);
    }
    __syncthreads();
    
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        k_head[i] = k_head[i] * s_rms_inv_k * k_weight[i];
    }
}

/* ========================================================================
 * AdaLN: out = (1 + scale) * layernorm(x) + shift
 * ======================================================================== */

__global__ void kernel_adaln_norm(float *out, const float *x,
                                  const float *shift, const float *scale,
                                  int seq_len, int hidden, float eps) {
    int seq_idx = blockIdx.x;
    if (seq_idx >= seq_len) return;
    
    const float *x_row = x + seq_idx * hidden;
    float *out_row = out + seq_idx * hidden;
    
    /* Compute mean */
    float sum = 0.0f;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        sum += x_row[i];
    }
    sum = block_reduce_sum(sum);
    
    __shared__ float s_mean;
    if (threadIdx.x == 0) {
        s_mean = sum / hidden;
    }
    __syncthreads();
    
    /* Compute variance */
    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float diff = x_row[i] - s_mean;
        var_sum += diff * diff;
    }
    var_sum = block_reduce_sum(var_sum);
    
    __shared__ float s_var_inv;
    if (threadIdx.x == 0) {
        s_var_inv = rsqrtf(var_sum / hidden + eps);
    }
    __syncthreads();
    
    /* Apply: out = (1 + scale) * norm(x) + shift */
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float norm_val = (x_row[i] - s_mean) * s_var_inv;
        out_row[i] = (1.0f + scale[i]) * norm_val + shift[i];
    }
}

/* ========================================================================
 * Softmax (row-wise)
 * x: [rows, cols]
 * ======================================================================== */

__global__ void kernel_softmax(float *x, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    
    float *row_ptr = x + row * cols;
    
    /* Find max for numerical stability */
    float max_val = -FLT_MAX;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        max_val = fmaxf(max_val, row_ptr[i]);
    }
    max_val = block_reduce_max(max_val);
    
    __shared__ float s_max;
    if (threadIdx.x == 0) s_max = max_val;
    __syncthreads();
    
    /* Compute exp(x - max) and sum */
    float sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float val = expf(row_ptr[i] - s_max);
        row_ptr[i] = val;
        sum += val;
    }
    sum = block_reduce_sum(sum);
    
    __shared__ float s_sum_inv;
    if (threadIdx.x == 0) s_sum_inv = 1.0f / sum;
    __syncthreads();
    
    /* Normalize */
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        row_ptr[i] *= s_sum_inv;
    }
}

/* ========================================================================
 * Gated Add: out += gate * proj
 * ======================================================================== */

__global__ void kernel_gated_add(float *out, const float *gate,
                                 const float *proj, int seq, int hidden) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq * hidden;
    if (idx >= total) return;
    
    int h = idx % hidden;
    out[idx] += gate[h] * proj[idx];
}

/* ========================================================================
 * RoPE (Rotary Position Embeddings)
 * 
 * x: [seq, heads * head_dim]
 * cos_freq, sin_freq: [seq, axis_dim] - precomputed frequencies
 * 
 * RoPE rotates pairs of adjacent elements:
 *   x_new[2i]   = x[2i] * cos - x[2i+1] * sin
 *   x_new[2i+1] = x[2i] * sin + x[2i+1] * cos
 * ======================================================================== */

__global__ void kernel_rope_2d(float *x, const float *cos_freq, const float *sin_freq,
                               int seq, int heads, int head_dim, int axis_dim) {
    /* Each thread handles one rotation pair */
    int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pairs_per_head = axis_dim / 2;
    int total_pairs = seq * heads * pairs_per_head;
    
    if (pair_idx >= total_pairs) return;
    
    int p = pair_idx % pairs_per_head;
    int tmp = pair_idx / pairs_per_head;
    int head = tmp % heads;
    int s = tmp / heads;
    
    /* Position in x tensor */
    int x_base = s * (heads * head_dim) + head * head_dim + p * 2;
    
    /* Position in frequency tensor */
    int freq_idx = s * axis_dim + p * 2;
    
    float x0 = x[x_base];
    float x1 = x[x_base + 1];
    float cos_val = cos_freq[freq_idx / 2];  /* cos is stored per pair */
    float sin_val = sin_freq[freq_idx / 2];
    
    x[x_base]     = x0 * cos_val - x1 * sin_val;
    x[x_base + 1] = x0 * sin_val + x1 * cos_val;
}

/* Unified RoPE for text + image (different frequencies) */
__global__ void kernel_rope_unified(float *q, float *k,
                                    const float *txt_cos, const float *txt_sin,
                                    const float *img_cos, const float *img_sin,
                                    int seq, int img_offset, int heads, int head_dim, int axis_dim) {
    int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pairs_per_head = axis_dim / 2;
    int total_pairs = seq * heads * pairs_per_head;
    
    if (pair_idx >= total_pairs) return;
    
    int p = pair_idx % pairs_per_head;
    int tmp = pair_idx / pairs_per_head;
    int head = tmp % heads;
    int s = tmp / heads;
    
    /* Select frequency table based on position */
    int is_img = (s >= img_offset);
    int local_s = is_img ? (s - img_offset) : s;
    
    const float *cos_tbl = is_img ? img_cos : txt_cos;
    const float *sin_tbl = is_img ? img_sin : txt_sin;
    
    int freq_idx = local_s * pairs_per_head + p;
    float cos_val = cos_tbl[freq_idx];
    float sin_val = sin_tbl[freq_idx];
    
    /* Apply to Q */
    int q_base = s * (heads * head_dim) + head * head_dim + p * 2;
    float q0 = q[q_base];
    float q1 = q[q_base + 1];
    q[q_base]     = q0 * cos_val - q1 * sin_val;
    q[q_base + 1] = q0 * sin_val + q1 * cos_val;
    
    /* Apply to K */
    int k_base = q_base;  /* Same layout */
    float k0 = k[k_base];
    float k1 = k[k_base + 1];
    k[k_base]     = k0 * cos_val - k1 * sin_val;
    k[k_base + 1] = k0 * sin_val + k1 * cos_val;
}

/* ========================================================================
 * Transpose Kernels for Attention
 * ======================================================================== */

/* Transpose from [seq, heads*head_dim] to [heads, seq, head_dim] */
__global__ void kernel_transpose_to_heads(float *out, const float *in,
                                          int seq, int heads, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq * heads * head_dim;
    if (idx >= total) return;
    
    int d = idx % head_dim;
    int tmp = idx / head_dim;
    int s = tmp % seq;
    int h = tmp / seq;
    
    /* out[h, s, d] = in[s, h*head_dim + d] */
    int in_idx = s * (heads * head_dim) + h * head_dim + d;
    out[idx] = in[in_idx];
}

/* Transpose from [heads, seq, head_dim] to [seq, heads*head_dim] */
__global__ void kernel_transpose_from_heads(float *out, const float *in,
                                            int seq, int heads, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq * heads * head_dim;
    if (idx >= total) return;
    
    int d = idx % head_dim;
    int tmp = idx / head_dim;
    int s = tmp % seq;
    int h = tmp / seq;
    
    /* out[s, h*head_dim + d] = in[h, s, d] */
    int out_idx = s * (heads * head_dim) + h * head_dim + d;
    int in_idx = h * (seq * head_dim) + s * head_dim + d;
    out[out_idx] = in[in_idx];
}

/* ========================================================================
 * C Interface Wrappers
 * ======================================================================== */

extern "C" {

static hipStream_t get_stream();  /* Defined in flux_rocm.cpp */

void flux_rocm_kernel_silu(float *x, int n, hipStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    kernel_silu<<<blocks, threads, 0, stream>>>(x, n);
}

void flux_rocm_kernel_silu_mul(float *gate, const float *up, int n, hipStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    kernel_silu_mul<<<blocks, threads, 0, stream>>>(gate, up, n);
}

void flux_rocm_kernel_rms_norm(float *out, const float *x, const float *weight,
                               int seq_len, int hidden, float eps, hipStream_t stream) {
    int threads = min(256, hidden);
    kernel_rms_norm<<<seq_len, threads, 0, stream>>>(out, x, weight, seq_len, hidden, eps);
}

void flux_rocm_kernel_qk_rms_norm(float *q, float *k,
                                  const float *q_weight, const float *k_weight,
                                  int seq, int heads, int head_dim, float eps, hipStream_t stream) {
    int threads = min(256, head_dim);
    int blocks = seq * heads;
    kernel_qk_rms_norm<<<blocks, threads, 0, stream>>>(q, k, q_weight, k_weight, seq, heads, head_dim, eps);
}

void flux_rocm_kernel_adaln_norm(float *out, const float *x,
                                 const float *shift, const float *scale,
                                 int seq_len, int hidden, float eps, hipStream_t stream) {
    int threads = min(256, hidden);
    kernel_adaln_norm<<<seq_len, threads, 0, stream>>>(out, x, shift, scale, seq_len, hidden, eps);
}

void flux_rocm_kernel_softmax(float *x, int rows, int cols, hipStream_t stream) {
    int threads = min(256, cols);
    kernel_softmax<<<rows, threads, 0, stream>>>(x, rows, cols);
}

void flux_rocm_kernel_gated_add(float *out, const float *gate, const float *proj,
                                int seq, int hidden, hipStream_t stream) {
    int total = seq * hidden;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    kernel_gated_add<<<blocks, threads, 0, stream>>>(out, gate, proj, seq, hidden);
}

void flux_rocm_kernel_rope_2d(float *x, const float *cos_freq, const float *sin_freq,
                              int seq, int heads, int head_dim, int axis_dim, hipStream_t stream) {
    int pairs_per_head = axis_dim / 2;
    int total_pairs = seq * heads * pairs_per_head;
    int threads = 256;
    int blocks = (total_pairs + threads - 1) / threads;
    kernel_rope_2d<<<blocks, threads, 0, stream>>>(x, cos_freq, sin_freq, seq, heads, head_dim, axis_dim);
}

void flux_rocm_kernel_rope_unified(float *q, float *k,
                                   const float *txt_cos, const float *txt_sin,
                                   const float *img_cos, const float *img_sin,
                                   int seq, int img_offset, int heads, int head_dim, int axis_dim,
                                   hipStream_t stream) {
    int pairs_per_head = axis_dim / 2;
    int total_pairs = seq * heads * pairs_per_head;
    int threads = 256;
    int blocks = (total_pairs + threads - 1) / threads;
    kernel_rope_unified<<<blocks, threads, 0, stream>>>(q, k, txt_cos, txt_sin, img_cos, img_sin,
                                                        seq, img_offset, heads, head_dim, axis_dim);
}

void flux_rocm_kernel_transpose_to_heads(float *out, const float *in,
                                         int seq, int heads, int head_dim,
                                         hipStream_t stream) {
    int total = seq * heads * head_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    kernel_transpose_to_heads<<<blocks, threads, 0, stream>>>(out, in, seq, heads, head_dim);
}

void flux_rocm_kernel_transpose_from_heads(float *out, const float *in,
                                           int seq, int heads, int head_dim,
                                           hipStream_t stream) {
    int total = seq * heads * head_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    kernel_transpose_from_heads<<<blocks, threads, 0, stream>>>(out, in, seq, heads, head_dim);
}

} /* extern "C" */
