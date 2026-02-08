/*
 * FLUX CUDA Attention Helpers
 *
 * CUDA implementation of attention operations used by the CUDA backend.
 * Keeps Q/K/V, attention scores, softmax, and output projection on GPU.
 */

#include "flux_cuda.h"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cublasLt.h>

#include <stdio.h>
#include <stdlib.h>

/* -------------------------------------------------------------------------
 * Global CUDA state (single-process singleton)
 * ------------------------------------------------------------------------- */

static cublasHandle_t g_cublas = NULL;
static cublasLtHandle_t g_cublas_lt = NULL;
static cudaStream_t g_stream = NULL;

static float *g_d_q = NULL;
static float *g_d_k = NULL;
static float *g_d_v = NULL;
static float *g_d_scores = NULL;
static float *g_d_out = NULL;
static float *g_d_q_shd = NULL;
static float *g_d_k_shd = NULL;
static float *g_d_v_shd = NULL;
static float *g_d_out_shd = NULL;
static __nv_bfloat16 *g_d_q_bf16 = NULL;
static __nv_bfloat16 *g_d_k_bf16 = NULL;
static int *g_d_mask = NULL;
static void *g_d_workspace = NULL;

static size_t g_d_q_bytes = 0;
static size_t g_d_k_bytes = 0;
static size_t g_d_v_bytes = 0;
static size_t g_d_scores_bytes = 0;
static size_t g_d_out_bytes = 0;
static size_t g_d_q_shd_bytes = 0;
static size_t g_d_k_shd_bytes = 0;
static size_t g_d_v_shd_bytes = 0;
static size_t g_d_out_shd_bytes = 0;
static size_t g_d_q_bf16_bytes = 0;
static size_t g_d_k_bf16_bytes = 0;
static size_t g_d_mask_bytes = 0;
static size_t g_d_workspace_bytes = 0;

static int g_cuda_ready = -1;
static int g_warned = 0;
static int g_flash_mode = -1;

static void flux_cuda_warn_once(const char *msg) {
    if (!g_warned) {
        fprintf(stderr, "%s\n", msg);
        g_warned = 1;
    }
}

static int flux_cuda_flash_enabled(void) {
    if (g_flash_mode == -1) {
        g_flash_mode = getenv("FLUX_CUDA_FLASH_ATTN") ? 1 : 0;
    }
    return g_flash_mode;
}

static void flux_cuda_cleanup(void) {
    if (g_d_q) cudaFree(g_d_q);
    if (g_d_k) cudaFree(g_d_k);
    if (g_d_v) cudaFree(g_d_v);
    if (g_d_scores) cudaFree(g_d_scores);
    if (g_d_out) cudaFree(g_d_out);
    if (g_d_q_shd) cudaFree(g_d_q_shd);
    if (g_d_k_shd) cudaFree(g_d_k_shd);
    if (g_d_v_shd) cudaFree(g_d_v_shd);
    if (g_d_out_shd) cudaFree(g_d_out_shd);
    if (g_d_q_bf16) cudaFree(g_d_q_bf16);
    if (g_d_k_bf16) cudaFree(g_d_k_bf16);
    if (g_d_mask) cudaFree(g_d_mask);
    if (g_d_workspace) cudaFree(g_d_workspace);

    g_d_q = g_d_k = g_d_v = g_d_scores = g_d_out = NULL;
    g_d_q_shd = g_d_k_shd = g_d_v_shd = g_d_out_shd = NULL;
    g_d_q_bf16 = g_d_k_bf16 = NULL;
    g_d_mask = NULL;
    g_d_workspace = NULL;

    g_d_q_bytes = g_d_k_bytes = g_d_v_bytes = 0;
    g_d_scores_bytes = g_d_out_bytes = 0;
    g_d_q_shd_bytes = g_d_k_shd_bytes = g_d_v_shd_bytes = g_d_out_shd_bytes = 0;
    g_d_q_bf16_bytes = g_d_k_bf16_bytes = 0;
    g_d_mask_bytes = 0;
    g_d_workspace_bytes = 0;

    if (g_cublas) {
        cublasDestroy(g_cublas);
        g_cublas = NULL;
    }
    if (g_cublas_lt) {
        cublasLtDestroy(g_cublas_lt);
        g_cublas_lt = NULL;
    }
    g_stream = NULL;
}

static int flux_cuda_ensure_init(void) {
    if (g_cuda_ready != -1) {
        return g_cuda_ready;
    }

    g_cuda_ready = 0;
    int device_count = 0;
    cudaError_t cuda_err = cudaGetDeviceCount(&device_count);
    if (cuda_err != cudaSuccess || device_count <= 0) {
        return 0;
    }

    if (cublasCreate(&g_cublas) != CUBLAS_STATUS_SUCCESS) {
        flux_cuda_warn_once("CUDA attention: failed to create cuBLAS handle");
        flux_cuda_cleanup();
        return 0;
    }
    if (cublasLtCreate(&g_cublas_lt) != CUBLAS_STATUS_SUCCESS) {
        flux_cuda_warn_once("CUDA attention: failed to create cuBLASLt handle");
        flux_cuda_cleanup();
        return 0;
    }
    if (cublasSetPointerMode(g_cublas, CUBLAS_POINTER_MODE_HOST) != CUBLAS_STATUS_SUCCESS) {
        flux_cuda_warn_once("CUDA attention: failed to configure cuBLAS pointer mode");
        flux_cuda_cleanup();
        return 0;
    }
    if (cublasSetStream(g_cublas, g_stream) != CUBLAS_STATUS_SUCCESS) {
        flux_cuda_warn_once("CUDA attention: failed to configure cuBLAS stream");
        flux_cuda_cleanup();
        return 0;
    }

    atexit(flux_cuda_cleanup);
    g_cuda_ready = 1;
    return 1;
}

int flux_cuda_ops_set_stream(void *stream_handle) {
    g_stream = (cudaStream_t)stream_handle;
    if (!flux_cuda_ensure_init()) return 0;
    return cublasSetStream(g_cublas, g_stream) == CUBLAS_STATUS_SUCCESS;
}

static int flux_cuda_ensure_buffer(void **buf, size_t *cap_bytes, size_t need_bytes) {
    if (*cap_bytes >= need_bytes) {
        return 1;
    }

    size_t new_cap = need_bytes;
    if (*cap_bytes > 0) {
        new_cap = *cap_bytes;
        while (new_cap < need_bytes) {
            new_cap *= 2;
        }
    }

    void *new_buf = NULL;
    if (cudaMalloc(&new_buf, new_cap) != cudaSuccess) {
        flux_cuda_warn_once("CUDA attention: device allocation failed");
        return 0;
    }

    if (*buf) cudaFree(*buf);
    *buf = new_buf;
    *cap_bytes = new_cap;
    return 1;
}

/* -------------------------------------------------------------------------
 * CUDA kernels
 * ------------------------------------------------------------------------- */

__global__ static void f32_to_bf16_kernel(const float *in, __nv_bfloat16 *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2bfloat16(in[i]);
}

/* Row-wise masked softmax for attention scores.
 * scores: [rows, cols]
 * row_local = row % seq_q for causal masking in batched mode.
 */
__global__ static void masked_softmax_kernel(float *scores,
                                             const int *mask,
                                             int rows, int cols,
                                             int seq_q,
                                             int causal,
                                             int use_mask) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int row_local = row % seq_q;
    float *row_ptr = scores + (size_t)row * cols;

    extern __shared__ float sh[];

    float local_max = -1e30f;
    int local_valid = 0;
    for (int c = tid; c < cols; c += blockDim.x) {
        int valid = 1;
        if (causal && c > row_local) valid = 0;
        if (use_mask && mask[c] == 0) valid = 0;
        if (valid) {
            float v = row_ptr[c];
            local_max = fmaxf(local_max, v);
            local_valid = 1;
        }
    }

    sh[tid] = local_valid ? local_max : -1e30f;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sh[tid] = fmaxf(sh[tid], sh[tid + stride]);
        }
        __syncthreads();
    }

    float max_val = sh[0];

    float local_sum = 0.0f;
    for (int c = tid; c < cols; c += blockDim.x) {
        int valid = 1;
        if (causal && c > row_local) valid = 0;
        if (use_mask && mask[c] == 0) valid = 0;

        float e = 0.0f;
        if (valid) {
            e = expf(row_ptr[c] - max_val);
        }
        row_ptr[c] = e;
        local_sum += e;
    }

    sh[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sh[tid] += sh[tid + stride];
        }
        __syncthreads();
    }

    float sum_val = sh[0];
    if (sum_val <= 0.0f) {
        for (int c = tid; c < cols; c += blockDim.x) {
            row_ptr[c] = 0.0f;
        }
        return;
    }

    float inv_sum = 1.0f / sum_val;
    for (int c = tid; c < cols; c += blockDim.x) {
        row_ptr[c] *= inv_sum;
    }
}

/* Experimental flash-attention style fused kernel for SHD layout.
 * Computes out = softmax(scale * q @ k^T) @ v without materializing scores.
 * Constraints: no causal/mask, head_dim <= 256. */
__global__ static void flash_attn_shd_kernel(float *out,
                                             const float *q,
                                             const float *k,
                                             const float *v,
                                             int heads, int seq_q, int seq_k,
                                             int head_dim, float scale) {
    int row_head = blockIdx.x;            /* [0, heads * seq_q) */
    int h = row_head / seq_q;
    int qi = row_head % seq_q;
    int tid = threadIdx.x;

    if (h >= heads || qi >= seq_q) return;

    const float *q_row = q + ((size_t)qi * heads + h) * head_dim;
    float out_acc = 0.0f;

    extern __shared__ float sh[];
    float *sh_dot = sh;
    __shared__ float running_max;
    __shared__ float running_sum;
    __shared__ float corr;
    __shared__ float weight;
    __shared__ int new_max;

    if (tid == 0) {
        running_max = -1e30f;
        running_sum = 0.0f;
        corr = 1.0f;
        weight = 0.0f;
        new_max = 0;
    }
    __syncthreads();

    for (int kj = 0; kj < seq_k; kj++) {
        float partial = 0.0f;
        if (tid < head_dim) {
            partial = q_row[tid] * k[((size_t)kj * heads + h) * head_dim + tid];
        }
        sh_dot[tid] = partial;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                sh_dot[tid] += sh_dot[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0) {
            float score = sh_dot[0] * scale;
            if (score > running_max) {
                corr = expf(running_max - score);
                running_sum = running_sum * corr + 1.0f;
                running_max = score;
                weight = 1.0f;
                new_max = 1;
            } else {
                corr = 1.0f;
                weight = expf(score - running_max);
                running_sum += weight;
                new_max = 0;
            }
        }
        __syncthreads();

        if (tid < head_dim) {
            float vv = v[((size_t)kj * heads + h) * head_dim + tid];
            if (new_max) {
                out_acc = out_acc * corr + vv;
            } else {
                out_acc += weight * vv;
            }
        }
        __syncthreads();
    }

    if (tid < head_dim) {
        float inv = (running_sum > 0.0f) ? (1.0f / running_sum) : 0.0f;
        out[((size_t)qi * heads + h) * head_dim + tid] = out_acc * inv;
    }
}

/* AdaLN: out = (1 + scale) * LN(x) + shift, with shift/scale shared across seq. */
__global__ static void adaln_norm_kernel(float *out, const float *x,
                                         const float *shift, const float *scale,
                                         int seq, int hidden, float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= seq) return;

    const float *x_row = x + (size_t)row * hidden;
    float *o_row = out + (size_t)row * hidden;

    extern __shared__ float sh[];
    float *sh_sum = sh;
    float *sh_sq = sh + blockDim.x;

    float local_sum = 0.0f;
    float local_sq = 0.0f;
    for (int i = tid; i < hidden; i += blockDim.x) {
        float v = x_row[i];
        local_sum += v;
        local_sq += v * v;
    }
    sh_sum[tid] = local_sum;
    sh_sq[tid] = local_sq;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sh_sum[tid] += sh_sum[tid + stride];
            sh_sq[tid] += sh_sq[tid + stride];
        }
        __syncthreads();
    }

    float mean = sh_sum[0] / (float)hidden;
    float var = sh_sq[0] / (float)hidden - mean * mean;
    if (var < 0.0f) var = 0.0f;
    float inv = rsqrtf(var + eps);

    for (int i = tid; i < hidden; i += blockDim.x) {
        float norm = (x_row[i] - mean) * inv;
        o_row[i] = (1.0f + scale[i]) * norm + shift[i];
    }
}

/* Split fused projection output into Q/K/V + MLP gate/up. */
__global__ static void split_qkv_mlp_kernel(const float *fused,
                                            float *q, float *k, float *v,
                                            float *gate, float *up,
                                            int seq, int hidden, int mlp_hidden) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= seq) return;

    int fused_dim = hidden * 3 + mlp_hidden * 2;
    const float *src = fused + (size_t)row * fused_dim;

    float *q_row = q + (size_t)row * hidden;
    float *k_row = k + (size_t)row * hidden;
    float *v_row = v + (size_t)row * hidden;
    for (int i = tid; i < hidden; i += blockDim.x) {
        q_row[i] = src[i];
        k_row[i] = src[hidden + i];
        v_row[i] = src[hidden * 2 + i];
    }

    float *g_row = gate + (size_t)row * mlp_hidden;
    float *u_row = up + (size_t)row * mlp_hidden;
    int off = hidden * 3;
    for (int i = tid; i < mlp_hidden; i += blockDim.x) {
        g_row[i] = src[off + i];
        u_row[i] = src[off + mlp_hidden + i];
    }
}

/* Per-head RMSNorm for Q/K with shared head weights. */
__global__ static void qk_rms_norm_kernel(float *q, float *k,
                                          const float *q_weight, const float *k_weight,
                                          int rows, int head_dim, float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= rows) return;

    float *q_row = q + (size_t)row * head_dim;
    float *k_row = k + (size_t)row * head_dim;

    extern __shared__ float sh[];
    float *sh_q = sh;
    float *sh_k = sh + blockDim.x;

    float sq_q = 0.0f;
    float sq_k = 0.0f;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float qv = q_row[d];
        float kv = k_row[d];
        sq_q += qv * qv;
        sq_k += kv * kv;
    }
    sh_q[tid] = sq_q;
    sh_k[tid] = sq_k;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sh_q[tid] += sh_q[tid + stride];
            sh_k[tid] += sh_k[tid + stride];
        }
        __syncthreads();
    }

    float inv_q = rsqrtf(sh_q[0] / (float)head_dim + eps);
    float inv_k = rsqrtf(sh_k[0] / (float)head_dim + eps);

    for (int d = tid; d < head_dim; d += blockDim.x) {
        q_row[d] = q_row[d] * inv_q * q_weight[d];
        k_row[d] = k_row[d] * inv_k * k_weight[d];
    }
}

/* Apply unified RoPE for text (prefix) + image (suffix) tokens. */
__global__ static void rope_unified_kernel(float *q, float *k,
                                           const float *txt_cos, const float *txt_sin,
                                           const float *img_cos, const float *img_sin,
                                           int seq, int img_offset,
                                           int heads, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pairs = head_dim / 2;
    int total = seq * heads * pairs;
    if (idx >= total) return;

    int pair = idx % pairs;
    int t = idx / pairs;
    int h = t % heads;
    int s = t / heads;
    int d = pair * 2;

    const float *cos_row;
    const float *sin_row;
    if (s < img_offset) {
        cos_row = txt_cos + (size_t)s * head_dim;
        sin_row = txt_sin + (size_t)s * head_dim;
    } else {
        int img_s = s - img_offset;
        cos_row = img_cos + (size_t)img_s * head_dim;
        sin_row = img_sin + (size_t)img_s * head_dim;
    }

    size_t base = ((size_t)s * heads + h) * head_dim;
    float *qv = q + base;
    float *kv = k + base;

    float c = cos_row[d];
    float sn = sin_row[d];

    float q0 = qv[d];
    float q1 = qv[d + 1];
    qv[d] = q0 * c - q1 * sn;
    qv[d + 1] = q1 * c + q0 * sn;

    float k0 = kv[d];
    float k1 = kv[d + 1];
    kv[d] = k0 * c - k1 * sn;
    kv[d + 1] = k1 * c + k0 * sn;
}

__global__ static void silu_mul_kernel(float *gate, const float *up, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i];
    float silu = g / (1.0f + expf(-g));
    gate[i] = silu * up[i];
}

__global__ static void concat_attn_mlp_kernel(const float *attn, const float *mlp, float *out,
                                              int seq, int hidden, int mlp_hidden) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row_dim = hidden + mlp_hidden;
    int total = seq * row_dim;
    if (idx >= total) return;

    int row = idx / row_dim;
    int col = idx % row_dim;
    if (col < hidden) {
        out[idx] = attn[(size_t)row * hidden + col];
    } else {
        out[idx] = mlp[(size_t)row * mlp_hidden + (col - hidden)];
    }
}

__global__ static void concat_seq_kernel(float *out, const float *a, const float *b,
                                         int seq_a, int seq_b, int hidden) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = (seq_a + seq_b) * hidden;
    if (idx >= total) return;

    int row = idx / hidden;
    int col = idx % hidden;
    if (row < seq_a) {
        out[idx] = a[(size_t)row * hidden + col];
    } else {
        int rb = row - seq_a;
        out[idx] = b[(size_t)rb * hidden + col];
    }
}

__global__ static void gated_add_kernel(float *hidden, const float *gate, const float *proj,
                                        int seq, int hidden_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq * hidden_dim;
    if (idx >= total) return;
    int col = idx % hidden_dim;
    hidden[idx] += gate[col] * proj[idx];
}

__global__ static void im2col_nchw_rows_kernel(float *col, const float *in,
                                                int in_ch, int H, int W,
                                                int kH, int kW, int stride, int padding,
                                                int outW, int row_offset, int tile_h,
                                                int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tile_pixels = tile_h * outW;
    int total = tile_pixels * K;
    if (idx >= total) return;

    int pix = idx / K;
    int k = idx % K;

    int oh_rel = pix / outW;
    int ow = pix % outW;
    int oh = row_offset + oh_rel;

    int ic = k / (kH * kW);
    int rem = k % (kH * kW);
    int kh = rem / kW;
    int kw = rem % kW;

    int ih = oh * stride - padding + kh;
    int iw = ow * stride - padding + kw;

    float v = 0.0f;
    if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
        v = in[(size_t)ic * H * W + (size_t)ih * W + iw];
    }

    col[idx] = v;
}

__global__ static void add_bias_rows_kernel(float *rows, const float *bias,
                                            int rows_count, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows_count * cols;
    if (idx >= total) return;
    int col = idx % cols;
    rows[idx] += bias[col];
}

__global__ static void rows_to_nchw_tile_kernel(float *out, const float *rows,
                                                int out_ch, int outH, int outW,
                                                int row_offset, int tile_h) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tile_pixels = tile_h * outW;
    int total = out_ch * tile_pixels;
    if (idx >= total) return;

    int oc = idx / tile_pixels;
    int pix = idx % tile_pixels;
    int oh = row_offset + (pix / outW);
    int ow = pix % outW;

    out[(size_t)oc * outH * outW + (size_t)oh * outW + ow] = rows[(size_t)pix * out_ch + oc];
}

__global__ static void nchw_to_rows_kernel(float *rows, const float *x,
                                           int channels, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int seq = H * W;
    int total = seq * channels;
    if (idx >= total) return;

    int s = idx / channels;
    int c = idx % channels;
    rows[idx] = x[(size_t)c * seq + s];
}

__global__ static void rows_to_nchw_kernel(float *x, const float *rows,
                                           int channels, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int seq = H * W;
    int total = seq * channels;
    if (idx >= total) return;

    int s = idx / channels;
    int c = idx % channels;
    x[(size_t)c * seq + s] = rows[idx];
}

__global__ static void group_norm_nchw_kernel(float *out, const float *x,
                                              const float *gamma, const float *beta,
                                              int batch, int channels, int spatial,
                                              int num_groups, int channels_per_group,
                                              float eps) {
    int bg = blockIdx.x;
    int b = bg / num_groups;
    int g = bg % num_groups;
    if (b >= batch) return;

    int c_start = g * channels_per_group;
    int total = channels_per_group * spatial;
    int tid = threadIdx.x;

    extern __shared__ float sh[];
    float *sh_sum = sh;
    float *sh_sumsq = sh + blockDim.x;

    float local_sum = 0.0f;
    float local_sumsq = 0.0f;

    for (int idx = tid; idx < total; idx += blockDim.x) {
        int c = c_start + idx / spatial;
        int s = idx % spatial;
        float v = x[((size_t)b * channels + c) * spatial + s];
        local_sum += v;
        local_sumsq += v * v;
    }

    sh_sum[tid] = local_sum;
    sh_sumsq[tid] = local_sumsq;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sh_sum[tid] += sh_sum[tid + stride];
            sh_sumsq[tid] += sh_sumsq[tid + stride];
        }
        __syncthreads();
    }

    float mean = sh_sum[0] / (float)total;
    float var = sh_sumsq[0] / (float)total - mean * mean;
    float inv_std = rsqrtf(var + eps);

    for (int idx = tid; idx < total; idx += blockDim.x) {
        int c = c_start + idx / spatial;
        int s = idx % spatial;
        size_t off = ((size_t)b * channels + c) * spatial + s;
        float v = (x[off] - mean) * inv_std;
        out[off] = gamma[c] * v + beta[c];
    }
}

__global__ static void silu_kernel(float *x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float v = x[idx];
    x[idx] = v / (1.0f + expf(-v));
}

__global__ static void add_inplace_kernel(float *a, const float *b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    a[idx] += b[idx];
}

__global__ static void upsample_nearest2x_nchw_kernel(float *out, const float *in,
                                                      int channels, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int outH = H * 2;
    int outW = W * 2;
    int total = channels * outH * outW;
    if (idx >= total) return;

    int c = idx / (outH * outW);
    int rem = idx % (outH * outW);
    int oh = rem / outW;
    int ow = rem % outW;

    int ih = oh >> 1;
    int iw = ow >> 1;

    out[(size_t)c * outH * outW + (size_t)oh * outW + ow] =
        in[(size_t)c * H * W + (size_t)ih * W + iw];
}

/* -------------------------------------------------------------------------
 * GEMM helpers (row-major wrappers)
 * ------------------------------------------------------------------------- */

/* Row-major scores = scale * q @ k^T
 * q: [seq_q, head_dim], k: [seq_k, head_dim], scores: [seq_q, seq_k]
 */
static int flux_cuda_qk_matmul_f32(float *d_scores,
                                   const float *d_q,
                                   const float *d_k,
                                   int seq_q, int seq_k, int head_dim,
                                   float scale) {
    const float beta = 0.0f;
    cublasStatus_t st = cublasSgemm(g_cublas,
                                    CUBLAS_OP_T, CUBLAS_OP_N,
                                    seq_k, seq_q, head_dim,
                                    &scale,
                                    d_k, head_dim,
                                    d_q, head_dim,
                                    &beta,
                                    d_scores, seq_k);
    return st == CUBLAS_STATUS_SUCCESS;
}

/* Row-major out = scores @ v
 * scores: [seq_q, seq_k], v: [seq_k, head_dim], out: [seq_q, head_dim]
 */
static int flux_cuda_pv_matmul_f32(float *d_out,
                                   const float *d_scores,
                                   const float *d_v,
                                   int seq_q, int seq_k, int head_dim) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasStatus_t st = cublasSgemm(g_cublas,
                                    CUBLAS_OP_N, CUBLAS_OP_N,
                                    head_dim, seq_q, seq_k,
                                    &alpha,
                                    d_v, head_dim,
                                    d_scores, seq_k,
                                    &beta,
                                    d_out, head_dim);
    return st == CUBLAS_STATUS_SUCCESS;
}

/* Row-major batched scores = scale * q @ k^T
 * q/k/scores are head-major: [heads, seq, dim]
 */
static int flux_cuda_qk_matmul_f32_batched(float *d_scores,
                                           const float *d_q,
                                           const float *d_k,
                                           int heads, int seq_q, int seq_k, int head_dim,
                                           float scale) {
    const float beta = 0.0f;
    long long stride_q = (long long)seq_q * head_dim;
    long long stride_k = (long long)seq_k * head_dim;
    long long stride_scores = (long long)seq_q * seq_k;

    cublasStatus_t st = cublasSgemmStridedBatched(g_cublas,
                                                  CUBLAS_OP_T, CUBLAS_OP_N,
                                                  seq_k, seq_q, head_dim,
                                                  &scale,
                                                  d_k, head_dim, stride_k,
                                                  d_q, head_dim, stride_q,
                                                  &beta,
                                                  d_scores, seq_k, stride_scores,
                                                  heads);
    return st == CUBLAS_STATUS_SUCCESS;
}

/* Row-major batched out = scores @ v */
static int flux_cuda_pv_matmul_f32_batched(float *d_out,
                                           const float *d_scores,
                                           const float *d_v,
                                           int heads, int seq_q, int seq_k, int head_dim) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    long long stride_v = (long long)seq_k * head_dim;
    long long stride_scores = (long long)seq_q * seq_k;
    long long stride_out = (long long)seq_q * head_dim;

    cublasStatus_t st = cublasSgemmStridedBatched(g_cublas,
                                                  CUBLAS_OP_N, CUBLAS_OP_N,
                                                  head_dim, seq_q, seq_k,
                                                  &alpha,
                                                  d_v, head_dim, stride_v,
                                                  d_scores, seq_k, stride_scores,
                                                  &beta,
                                                  d_out, head_dim, stride_out,
                                                  heads);
    return st == CUBLAS_STATUS_SUCCESS;
}

/* Row-major batched scores = scale * q @ k^T
 * q/k are sequence-major [seq, heads, dim] interleaved by head.
 * scores are contiguous head-major [heads, seq_q, seq_k].
 */
static int flux_cuda_qk_matmul_f32_batched_shd(float *d_scores,
                                               const float *d_q_shd,
                                               const float *d_k_shd,
                                               int heads, int seq_q, int seq_k,
                                               int hidden, int head_dim,
                                               float scale) {
    const float beta = 0.0f;
    long long stride_qk = (long long)head_dim;            /* interleaved heads */
    long long stride_scores = (long long)seq_q * seq_k;   /* contiguous per head */

    cublasStatus_t st = cublasSgemmStridedBatched(g_cublas,
                                                  CUBLAS_OP_T, CUBLAS_OP_N,
                                                  seq_k, seq_q, head_dim,
                                                  &scale,
                                                  d_k_shd, hidden, stride_qk,
                                                  d_q_shd, hidden, stride_qk,
                                                  &beta,
                                                  d_scores, seq_k, stride_scores,
                                                  heads);
    return st == CUBLAS_STATUS_SUCCESS;
}

/* Row-major batched out = scores @ v
 * scores are contiguous head-major [heads, seq_q, seq_k].
 * v/out are sequence-major [seq, heads, dim] interleaved by head.
 */
static int flux_cuda_pv_matmul_f32_batched_shd(float *d_out_shd,
                                               const float *d_scores,
                                               const float *d_v_shd,
                                               int heads, int seq_q, int seq_k,
                                               int hidden, int head_dim) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    long long stride_v_out = (long long)head_dim;         /* interleaved heads */
    long long stride_scores = (long long)seq_q * seq_k;   /* contiguous per head */

    cublasStatus_t st = cublasSgemmStridedBatched(g_cublas,
                                                  CUBLAS_OP_N, CUBLAS_OP_N,
                                                  head_dim, seq_q, seq_k,
                                                  &alpha,
                                                  d_v_shd, hidden, stride_v_out,
                                                  d_scores, seq_k, stride_scores,
                                                  &beta,
                                                  d_out_shd, hidden, stride_v_out,
                                                  heads);
    return st == CUBLAS_STATUS_SUCCESS;
}

/* BF16 tensor-core QK path via cuBLASLt (single-head). */
static int flux_cuda_qk_matmul_bf16_lt(float *d_scores,
                                       const float *d_q,
                                       const float *d_k,
                                       __nv_bfloat16 *d_q_bf16,
                                       __nv_bfloat16 *d_k_bf16,
                                       int seq_q, int seq_k, int head_dim,
                                       float scale) {
    int q_elems = seq_q * head_dim;
    int k_elems = seq_k * head_dim;

    int threads = 256;
    int q_blocks = (q_elems + threads - 1) / threads;
    int k_blocks = (k_elems + threads - 1) / threads;
    f32_to_bf16_kernel<<<q_blocks, threads, 0, g_stream>>>(d_q, d_q_bf16, q_elems);
    f32_to_bf16_kernel<<<k_blocks, threads, 0, g_stream>>>(d_k, d_k_bf16, k_elems);
    if (cudaGetLastError() != cudaSuccess) {
        return 0;
    }

    cublasLtMatmulDesc_t op_desc = NULL;
    cublasLtMatrixLayout_t a_layout = NULL;
    cublasLtMatrixLayout_t b_layout = NULL;
    cublasLtMatrixLayout_t c_layout = NULL;
    cublasLtMatmulPreference_t pref = NULL;
    cublasLtMatmulHeuristicResult_t heuristic;
    int num_results = 0;

    cublasOperation_t trans_a = CUBLAS_OP_N;
    cublasOperation_t trans_b = CUBLAS_OP_T;
    cublasLtOrder_t order = CUBLASLT_ORDER_ROW;

#ifdef CUBLAS_COMPUTE_32F_FAST_16BF
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F_FAST_16BF;
#else
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
#endif

    cublasStatus_t st = cublasLtMatmulDescCreate(&op_desc, compute_type, CUDA_R_32F);
    if (st != CUBLAS_STATUS_SUCCESS) goto fail;

    st = cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                        &trans_a, sizeof(trans_a));
    if (st != CUBLAS_STATUS_SUCCESS) goto fail;
    st = cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB,
                                        &trans_b, sizeof(trans_b));
    if (st != CUBLAS_STATUS_SUCCESS) goto fail;

    st = cublasLtMatrixLayoutCreate(&a_layout, CUDA_R_16BF, seq_q, head_dim, head_dim);
    if (st != CUBLAS_STATUS_SUCCESS) goto fail;
    st = cublasLtMatrixLayoutCreate(&b_layout, CUDA_R_16BF, seq_k, head_dim, head_dim);
    if (st != CUBLAS_STATUS_SUCCESS) goto fail;
    st = cublasLtMatrixLayoutCreate(&c_layout, CUDA_R_32F, seq_q, seq_k, seq_k);
    if (st != CUBLAS_STATUS_SUCCESS) goto fail;

    st = cublasLtMatrixLayoutSetAttribute(a_layout, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                          &order, sizeof(order));
    if (st != CUBLAS_STATUS_SUCCESS) goto fail;
    st = cublasLtMatrixLayoutSetAttribute(b_layout, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                          &order, sizeof(order));
    if (st != CUBLAS_STATUS_SUCCESS) goto fail;
    st = cublasLtMatrixLayoutSetAttribute(c_layout, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                          &order, sizeof(order));
    if (st != CUBLAS_STATUS_SUCCESS) goto fail;

    st = cublasLtMatmulPreferenceCreate(&pref);
    if (st != CUBLAS_STATUS_SUCCESS) goto fail;

    st = cublasLtMatmulPreferenceSetAttribute(pref,
                                              CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                              &g_d_workspace_bytes, sizeof(g_d_workspace_bytes));
    if (st != CUBLAS_STATUS_SUCCESS) goto fail;

    st = cublasLtMatmulAlgoGetHeuristic(g_cublas_lt, op_desc,
                                        a_layout, b_layout, c_layout, c_layout,
                                        pref, 1, &heuristic, &num_results);
    if (st != CUBLAS_STATUS_SUCCESS || num_results == 0) goto fail;

    {
        float beta = 0.0f;
        st = cublasLtMatmul(g_cublas_lt, op_desc,
                            &scale,
                            d_q_bf16, a_layout,
                            d_k_bf16, b_layout,
                            &beta,
                            d_scores, c_layout,
                            d_scores, c_layout,
                            &heuristic.algo,
                            g_d_workspace, g_d_workspace_bytes,
                            g_stream);
        if (st != CUBLAS_STATUS_SUCCESS) goto fail;
    }

    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(c_layout);
    cublasLtMatrixLayoutDestroy(b_layout);
    cublasLtMatrixLayoutDestroy(a_layout);
    cublasLtMatmulDescDestroy(op_desc);
    return 1;

fail:
    if (pref) cublasLtMatmulPreferenceDestroy(pref);
    if (c_layout) cublasLtMatrixLayoutDestroy(c_layout);
    if (b_layout) cublasLtMatrixLayoutDestroy(b_layout);
    if (a_layout) cublasLtMatrixLayoutDestroy(a_layout);
    if (op_desc) cublasLtMatmulDescDestroy(op_desc);
    return 0;
}

/* -------------------------------------------------------------------------
 * Public API
 * ------------------------------------------------------------------------- */

int flux_cuda_attention_single(float *out,
                               const float *q, const float *k, const float *v,
                               int seq_q, int seq_k, int head_dim,
                               float scale, int causal,
                               const int *attention_mask,
                               int prefer_bf16) {
    if (!out || !q || !k || !v) return 0;
    if (seq_q <= 0 || seq_k <= 0 || head_dim <= 0) return 0;
    if (!flux_cuda_ensure_init()) return 0;

    size_t q_bytes = (size_t)seq_q * head_dim * sizeof(float);
    size_t k_bytes = (size_t)seq_k * head_dim * sizeof(float);
    size_t v_bytes = (size_t)seq_k * head_dim * sizeof(float);
    size_t scores_bytes = (size_t)seq_q * seq_k * sizeof(float);
    size_t out_bytes = (size_t)seq_q * head_dim * sizeof(float);
    size_t q_bf16_bytes = (size_t)seq_q * head_dim * sizeof(__nv_bfloat16);
    size_t k_bf16_bytes = (size_t)seq_k * head_dim * sizeof(__nv_bfloat16);
    size_t mask_bytes = (size_t)seq_k * sizeof(int);
    size_t workspace_target = (size_t)8 * 1024 * 1024;

    if (!flux_cuda_ensure_buffer((void **)&g_d_q, &g_d_q_bytes, q_bytes) ||
        !flux_cuda_ensure_buffer((void **)&g_d_k, &g_d_k_bytes, k_bytes) ||
        !flux_cuda_ensure_buffer((void **)&g_d_v, &g_d_v_bytes, v_bytes) ||
        !flux_cuda_ensure_buffer((void **)&g_d_scores, &g_d_scores_bytes, scores_bytes) ||
        !flux_cuda_ensure_buffer((void **)&g_d_out, &g_d_out_bytes, out_bytes) ||
        !flux_cuda_ensure_buffer((void **)&g_d_q_bf16, &g_d_q_bf16_bytes, q_bf16_bytes) ||
        !flux_cuda_ensure_buffer((void **)&g_d_k_bf16, &g_d_k_bf16_bytes, k_bf16_bytes) ||
        !flux_cuda_ensure_buffer((void **)&g_d_workspace, &g_d_workspace_bytes, workspace_target)) {
        return 0;
    }

    if (attention_mask) {
        if (!flux_cuda_ensure_buffer((void **)&g_d_mask, &g_d_mask_bytes, mask_bytes)) {
            return 0;
        }
        if (cudaMemcpy(g_d_mask, attention_mask, mask_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            return 0;
        }
    }

    if (cudaMemcpy(g_d_q, q, q_bytes, cudaMemcpyHostToDevice) != cudaSuccess) return 0;
    if (cudaMemcpy(g_d_k, k, k_bytes, cudaMemcpyHostToDevice) != cudaSuccess) return 0;
    if (cudaMemcpy(g_d_v, v, v_bytes, cudaMemcpyHostToDevice) != cudaSuccess) return 0;

    int ok = 0;
    if (prefer_bf16) {
        ok = flux_cuda_qk_matmul_bf16_lt(g_d_scores, g_d_q, g_d_k,
                                         g_d_q_bf16, g_d_k_bf16,
                                         seq_q, seq_k, head_dim, scale);
    }
    if (!ok) {
        ok = flux_cuda_qk_matmul_f32(g_d_scores, g_d_q, g_d_k, seq_q, seq_k, head_dim, scale);
    }
    if (!ok) return 0;

    {
        int threads = 256;
        int rows = seq_q;
        size_t shmem = (size_t)threads * sizeof(float);
        masked_softmax_kernel<<<rows, threads, shmem, g_stream>>>(
            g_d_scores, g_d_mask, rows, seq_k, seq_q,
            causal ? 1 : 0, attention_mask ? 1 : 0
        );
        if (cudaGetLastError() != cudaSuccess) return 0;
    }

    if (!flux_cuda_pv_matmul_f32(g_d_out, g_d_scores, g_d_v, seq_q, seq_k, head_dim)) {
        return 0;
    }

    if (cudaMemcpy(out, g_d_out, out_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        return 0;
    }

    return 1;
}

int flux_cuda_attention_batched(float *out,
                                const float *q, const float *k, const float *v,
                                int heads, int seq_q, int seq_k, int head_dim,
                                float scale, int causal,
                                const int *attention_mask) {
    if (!out || !q || !k || !v) return 0;
    if (heads <= 0 || seq_q <= 0 || seq_k <= 0 || head_dim <= 0) return 0;
    if (!flux_cuda_ensure_init()) return 0;

    size_t q_bytes = (size_t)heads * seq_q * head_dim * sizeof(float);
    size_t k_bytes = (size_t)heads * seq_k * head_dim * sizeof(float);
    size_t v_bytes = (size_t)heads * seq_k * head_dim * sizeof(float);
    size_t scores_bytes = (size_t)heads * seq_q * seq_k * sizeof(float);
    size_t out_bytes = (size_t)heads * seq_q * head_dim * sizeof(float);
    size_t mask_bytes = (size_t)seq_k * sizeof(int);

    if (!flux_cuda_ensure_buffer((void **)&g_d_q, &g_d_q_bytes, q_bytes) ||
        !flux_cuda_ensure_buffer((void **)&g_d_k, &g_d_k_bytes, k_bytes) ||
        !flux_cuda_ensure_buffer((void **)&g_d_v, &g_d_v_bytes, v_bytes) ||
        !flux_cuda_ensure_buffer((void **)&g_d_scores, &g_d_scores_bytes, scores_bytes) ||
        !flux_cuda_ensure_buffer((void **)&g_d_out, &g_d_out_bytes, out_bytes)) {
        return 0;
    }

    if (attention_mask) {
        if (!flux_cuda_ensure_buffer((void **)&g_d_mask, &g_d_mask_bytes, mask_bytes)) {
            return 0;
        }
        if (cudaMemcpy(g_d_mask, attention_mask, mask_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            return 0;
        }
    }

    if (cudaMemcpy(g_d_q, q, q_bytes, cudaMemcpyHostToDevice) != cudaSuccess) return 0;
    if (cudaMemcpy(g_d_k, k, k_bytes, cudaMemcpyHostToDevice) != cudaSuccess) return 0;
    if (cudaMemcpy(g_d_v, v, v_bytes, cudaMemcpyHostToDevice) != cudaSuccess) return 0;

    if (!flux_cuda_qk_matmul_f32_batched(g_d_scores, g_d_q, g_d_k,
                                         heads, seq_q, seq_k, head_dim, scale)) {
        return 0;
    }

    {
        int threads = 256;
        int rows = heads * seq_q;
        size_t shmem = (size_t)threads * sizeof(float);
        masked_softmax_kernel<<<rows, threads, shmem, g_stream>>>(
            g_d_scores, g_d_mask, rows, seq_k, seq_q,
            causal ? 1 : 0, attention_mask ? 1 : 0
        );
        if (cudaGetLastError() != cudaSuccess) return 0;
    }

    if (!flux_cuda_pv_matmul_f32_batched(g_d_out, g_d_scores, g_d_v,
                                         heads, seq_q, seq_k, head_dim)) {
        return 0;
    }

    if (cudaMemcpy(out, g_d_out, out_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        return 0;
    }

    return 1;
}

int flux_cuda_attention_batched_shd(float *out,
                                    const float *q, const float *k, const float *v,
                                    int heads, int seq_q, int seq_k, int head_dim,
                                    float scale, int causal,
                                    const int *attention_mask) {
    if (!out || !q || !k || !v) return 0;
    if (heads <= 0 || seq_q <= 0 || seq_k <= 0 || head_dim <= 0) return 0;
    if (!flux_cuda_ensure_init()) return 0;

    size_t q_shd_bytes = (size_t)seq_q * heads * head_dim * sizeof(float);
    size_t k_shd_bytes = (size_t)seq_k * heads * head_dim * sizeof(float);
    size_t v_shd_bytes = (size_t)seq_k * heads * head_dim * sizeof(float);
    size_t out_shd_bytes = (size_t)seq_q * heads * head_dim * sizeof(float);
    size_t scores_bytes = (size_t)heads * seq_q * seq_k * sizeof(float);
    size_t mask_bytes = (size_t)seq_k * sizeof(int);
    int hidden = heads * head_dim;

    if (!flux_cuda_ensure_buffer((void **)&g_d_q_shd, &g_d_q_shd_bytes, q_shd_bytes) ||
        !flux_cuda_ensure_buffer((void **)&g_d_k_shd, &g_d_k_shd_bytes, k_shd_bytes) ||
        !flux_cuda_ensure_buffer((void **)&g_d_v_shd, &g_d_v_shd_bytes, v_shd_bytes) ||
        !flux_cuda_ensure_buffer((void **)&g_d_out_shd, &g_d_out_shd_bytes, out_shd_bytes) ||
        !flux_cuda_ensure_buffer((void **)&g_d_scores, &g_d_scores_bytes, scores_bytes)) {
        return 0;
    }

    if (attention_mask) {
        if (!flux_cuda_ensure_buffer((void **)&g_d_mask, &g_d_mask_bytes, mask_bytes)) {
            return 0;
        }
        if (cudaMemcpy(g_d_mask, attention_mask, mask_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            return 0;
        }
    }

    if (cudaMemcpy(g_d_q_shd, q, q_shd_bytes, cudaMemcpyHostToDevice) != cudaSuccess) return 0;
    if (cudaMemcpy(g_d_k_shd, k, k_shd_bytes, cudaMemcpyHostToDevice) != cudaSuccess) return 0;
    if (cudaMemcpy(g_d_v_shd, v, v_shd_bytes, cudaMemcpyHostToDevice) != cudaSuccess) return 0;

    if (!flux_cuda_qk_matmul_f32_batched_shd(g_d_scores, g_d_q_shd, g_d_k_shd,
                                             heads, seq_q, seq_k, hidden, head_dim, scale)) {
        return 0;
    }

    {
        int threads = 256;
        int rows = heads * seq_q;
        size_t shmem = (size_t)threads * sizeof(float);
        masked_softmax_kernel<<<rows, threads, shmem, g_stream>>>(
            g_d_scores, g_d_mask, rows, seq_k, seq_q,
            causal ? 1 : 0, attention_mask ? 1 : 0
        );
        if (cudaGetLastError() != cudaSuccess) return 0;
    }

    if (!flux_cuda_pv_matmul_f32_batched_shd(g_d_out_shd, g_d_scores, g_d_v_shd,
                                             heads, seq_q, seq_k, hidden, head_dim)) {
        return 0;
    }

    if (cudaMemcpy(out, g_d_out_shd, out_shd_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        return 0;
    }

    return 1;
}

int flux_cuda_attention_batched_shd_device(float *d_out,
                                           const float *d_q, const float *d_k, const float *d_v,
                                           int heads, int seq_q, int seq_k, int head_dim,
                                           float scale, int causal,
                                           const int *attention_mask) {
    if (!d_out || !d_q || !d_k || !d_v) return 0;
    if (heads <= 0 || seq_q <= 0 || seq_k <= 0 || head_dim <= 0) return 0;
    if (!flux_cuda_ensure_init()) return 0;

    size_t scores_bytes = (size_t)heads * seq_q * seq_k * sizeof(float);
    size_t mask_bytes = (size_t)seq_k * sizeof(int);
    int hidden = heads * head_dim;

    if (!causal && !attention_mask && flux_cuda_flash_enabled() && head_dim > 0 && head_dim <= 256) {
        int threads = 1;
        while (threads < head_dim) threads <<= 1;
        if (threads < 32) threads = 32;
        if (threads > 256) threads = 256;
        int blocks = heads * seq_q;
        size_t shmem = (size_t)threads * sizeof(float);
        flash_attn_shd_kernel<<<blocks, threads, shmem, g_stream>>>(
            d_out, d_q, d_k, d_v, heads, seq_q, seq_k, head_dim, scale
        );
        return cudaGetLastError() == cudaSuccess;
    }

    if (!flux_cuda_ensure_buffer((void **)&g_d_scores, &g_d_scores_bytes, scores_bytes)) {
        return 0;
    }

    if (attention_mask) {
        if (!flux_cuda_ensure_buffer((void **)&g_d_mask, &g_d_mask_bytes, mask_bytes)) {
            return 0;
        }
        if (cudaMemcpy(g_d_mask, attention_mask, mask_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            return 0;
        }
    }

    if (!flux_cuda_qk_matmul_f32_batched_shd(g_d_scores, d_q, d_k,
                                             heads, seq_q, seq_k, hidden, head_dim, scale)) {
        return 0;
    }

    {
        int threads = 256;
        int rows = heads * seq_q;
        size_t shmem = (size_t)threads * sizeof(float);
        masked_softmax_kernel<<<rows, threads, shmem, g_stream>>>(
            g_d_scores, g_d_mask, rows, seq_k, seq_q,
            causal ? 1 : 0, attention_mask ? 1 : 0
        );
        if (cudaGetLastError() != cudaSuccess) return 0;
    }

    if (!flux_cuda_pv_matmul_f32_batched_shd(d_out, g_d_scores, d_v,
                                             heads, seq_q, seq_k, hidden, head_dim)) {
        return 0;
    }

    return 1;
}

int flux_cuda_adaln_norm_device(float *d_out, const float *d_x,
                                const float *d_shift, const float *d_scale,
                                int seq, int hidden, float eps) {
    if (!d_out || !d_x || !d_shift || !d_scale || seq <= 0 || hidden <= 0) return 0;
    if (!flux_cuda_ensure_init()) return 0;

    int threads = 256;
    size_t shmem = (size_t)threads * 2 * sizeof(float);
    adaln_norm_kernel<<<seq, threads, shmem, g_stream>>>(d_out, d_x, d_shift, d_scale, seq, hidden, eps);
    return cudaGetLastError() == cudaSuccess;
}

int flux_cuda_split_qkv_mlp_device(const float *d_fused,
                                   float *d_q, float *d_k, float *d_v,
                                   float *d_gate, float *d_up,
                                   int seq, int hidden, int mlp_hidden) {
    if (!d_fused || !d_q || !d_k || !d_v || !d_gate || !d_up) return 0;
    if (seq <= 0 || hidden <= 0 || mlp_hidden <= 0) return 0;
    if (!flux_cuda_ensure_init()) return 0;

    int threads = 256;
    split_qkv_mlp_kernel<<<seq, threads, 0, g_stream>>>(d_fused, d_q, d_k, d_v, d_gate, d_up,
                                           seq, hidden, mlp_hidden);
    return cudaGetLastError() == cudaSuccess;
}

int flux_cuda_qk_rms_norm_device(float *d_q, float *d_k,
                                 const float *d_q_weight, const float *d_k_weight,
                                 int seq, int heads, int head_dim, float eps) {
    if (!d_q || !d_k || !d_q_weight || !d_k_weight) return 0;
    if (seq <= 0 || heads <= 0 || head_dim <= 0) return 0;
    if (!flux_cuda_ensure_init()) return 0;

    int rows = seq * heads;
    int threads = 128;
    size_t shmem = (size_t)threads * 2 * sizeof(float);
    qk_rms_norm_kernel<<<rows, threads, shmem, g_stream>>>(d_q, d_k, d_q_weight, d_k_weight,
                                                 rows, head_dim, eps);
    return cudaGetLastError() == cudaSuccess;
}

int flux_cuda_rope_unified_device(float *d_q, float *d_k,
                                  const float *d_txt_cos, const float *d_txt_sin,
                                  const float *d_img_cos, const float *d_img_sin,
                                  int seq, int img_offset, int heads, int head_dim) {
    if (!d_q || !d_k || !d_txt_cos || !d_txt_sin || !d_img_cos || !d_img_sin) return 0;
    if (seq <= 0 || img_offset < 0 || img_offset > seq || heads <= 0 || head_dim <= 0) return 0;
    if (!flux_cuda_ensure_init()) return 0;

    int pairs = head_dim / 2;
    int total = seq * heads * pairs;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    rope_unified_kernel<<<blocks, threads, 0, g_stream>>>(d_q, d_k,
                                             d_txt_cos, d_txt_sin,
                                             d_img_cos, d_img_sin,
                                             seq, img_offset, heads, head_dim);
    return cudaGetLastError() == cudaSuccess;
}

int flux_cuda_silu_mul_device(float *d_gate, const float *d_up, int n) {
    if (!d_gate || !d_up || n <= 0) return 0;
    if (!flux_cuda_ensure_init()) return 0;

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    silu_mul_kernel<<<blocks, threads, 0, g_stream>>>(d_gate, d_up, n);
    return cudaGetLastError() == cudaSuccess;
}

int flux_cuda_concat_attn_mlp_device(const float *d_attn, const float *d_mlp,
                                     float *d_out, int seq, int hidden, int mlp_hidden) {
    if (!d_attn || !d_mlp || !d_out) return 0;
    if (seq <= 0 || hidden <= 0 || mlp_hidden <= 0) return 0;
    if (!flux_cuda_ensure_init()) return 0;

    int total = seq * (hidden + mlp_hidden);
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    concat_attn_mlp_kernel<<<blocks, threads, 0, g_stream>>>(d_attn, d_mlp, d_out, seq, hidden, mlp_hidden);
    return cudaGetLastError() == cudaSuccess;
}

int flux_cuda_gated_add_device(float *d_hidden, const float *d_gate,
                               const float *d_proj, int seq, int hidden) {
    if (!d_hidden || !d_gate || !d_proj) return 0;
    if (seq <= 0 || hidden <= 0) return 0;
    if (!flux_cuda_ensure_init()) return 0;

    int total = seq * hidden;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    gated_add_kernel<<<blocks, threads, 0, g_stream>>>(d_hidden, d_gate, d_proj, seq, hidden);
    return cudaGetLastError() == cudaSuccess;
}

int flux_cuda_concat_seq_device(float *d_out, const float *d_a, const float *d_b,
                                int seq_a, int seq_b, int hidden) {
    if (!d_out || !d_a || !d_b) return 0;
    if (seq_a < 0 || seq_b < 0 || hidden <= 0) return 0;
    if (!flux_cuda_ensure_init()) return 0;

    int total = (seq_a + seq_b) * hidden;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    concat_seq_kernel<<<blocks, threads, 0, g_stream>>>(d_out, d_a, d_b, seq_a, seq_b, hidden);
    return cudaGetLastError() == cudaSuccess;
}

int flux_cuda_im2col_nchw_rows_device(float *d_col, const float *d_in,
                                      int in_ch, int H, int W,
                                      int kH, int kW, int stride, int padding,
                                      int outH, int outW,
                                      int row_offset, int tile_h) {
    if (!d_col || !d_in) return 0;
    if (in_ch <= 0 || H <= 0 || W <= 0 || kH <= 0 || kW <= 0 || stride <= 0) return 0;
    if (outH <= 0 || outW <= 0 || row_offset < 0 || tile_h <= 0 || row_offset + tile_h > outH) return 0;
    if (!flux_cuda_ensure_init()) return 0;

    int K = in_ch * kH * kW;
    int tile_pixels = tile_h * outW;
    int total = tile_pixels * K;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    im2col_nchw_rows_kernel<<<blocks, threads, 0, g_stream>>>(d_col, d_in,
                                                 in_ch, H, W,
                                                 kH, kW, stride, padding,
                                                 outW, row_offset, tile_h, K);
    return cudaGetLastError() == cudaSuccess;
}

int flux_cuda_add_bias_rows_device(float *d_rows, const float *d_bias,
                                   int rows, int cols) {
    if (!d_rows || !d_bias) return 0;
    if (rows <= 0 || cols <= 0) return 0;
    if (!flux_cuda_ensure_init()) return 0;

    int total = rows * cols;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    add_bias_rows_kernel<<<blocks, threads, 0, g_stream>>>(d_rows, d_bias, rows, cols);
    return cudaGetLastError() == cudaSuccess;
}

int flux_cuda_rows_to_nchw_tile_device(float *d_out, const float *d_rows,
                                       int out_ch, int outH, int outW,
                                       int row_offset, int tile_h) {
    if (!d_out || !d_rows) return 0;
    if (out_ch <= 0 || outH <= 0 || outW <= 0 || row_offset < 0 || tile_h <= 0) return 0;
    if (row_offset + tile_h > outH) return 0;
    if (!flux_cuda_ensure_init()) return 0;

    int tile_pixels = tile_h * outW;
    int total = out_ch * tile_pixels;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    rows_to_nchw_tile_kernel<<<blocks, threads, 0, g_stream>>>(d_out, d_rows, out_ch, outH, outW,
                                                  row_offset, tile_h);
    return cudaGetLastError() == cudaSuccess;
}

int flux_cuda_nchw_to_rows_device(float *d_rows, const float *d_x,
                                  int channels, int H, int W) {
    if (!d_rows || !d_x) return 0;
    if (channels <= 0 || H <= 0 || W <= 0) return 0;
    if (!flux_cuda_ensure_init()) return 0;

    int total = channels * H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    nchw_to_rows_kernel<<<blocks, threads, 0, g_stream>>>(d_rows, d_x, channels, H, W);
    return cudaGetLastError() == cudaSuccess;
}

int flux_cuda_rows_to_nchw_device(float *d_x, const float *d_rows,
                                  int channels, int H, int W) {
    if (!d_x || !d_rows) return 0;
    if (channels <= 0 || H <= 0 || W <= 0) return 0;
    if (!flux_cuda_ensure_init()) return 0;

    int total = channels * H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    rows_to_nchw_kernel<<<blocks, threads, 0, g_stream>>>(d_x, d_rows, channels, H, W);
    return cudaGetLastError() == cudaSuccess;
}

int flux_cuda_group_norm_nchw_device(float *d_out, const float *d_x,
                                     const float *d_gamma, const float *d_beta,
                                     int batch, int channels, int H, int W,
                                     int num_groups, float eps) {
    if (!d_out || !d_x || !d_gamma || !d_beta) return 0;
    if (batch <= 0 || channels <= 0 || H <= 0 || W <= 0 || num_groups <= 0) return 0;
    if (channels % num_groups != 0) return 0;
    if (!flux_cuda_ensure_init()) return 0;

    int channels_per_group = channels / num_groups;
    int spatial = H * W;
    int threads = 256;
    int blocks = batch * num_groups;
    size_t shmem = (size_t)threads * 2 * sizeof(float);
    group_norm_nchw_kernel<<<blocks, threads, shmem, g_stream>>>(d_out, d_x, d_gamma, d_beta,
                                                       batch, channels, spatial,
                                                       num_groups, channels_per_group,
                                                       eps);
    return cudaGetLastError() == cudaSuccess;
}

int flux_cuda_silu_device(float *d_x, int n) {
    if (!d_x || n <= 0) return 0;
    if (!flux_cuda_ensure_init()) return 0;

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    silu_kernel<<<blocks, threads, 0, g_stream>>>(d_x, n);
    return cudaGetLastError() == cudaSuccess;
}

int flux_cuda_add_inplace_device(float *d_a, const float *d_b, int n) {
    if (!d_a || !d_b || n <= 0) return 0;
    if (!flux_cuda_ensure_init()) return 0;

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add_inplace_kernel<<<blocks, threads, 0, g_stream>>>(d_a, d_b, n);
    return cudaGetLastError() == cudaSuccess;
}

int flux_cuda_upsample_nearest2x_nchw_device(float *d_out, const float *d_in,
                                             int channels, int H, int W) {
    if (!d_out || !d_in) return 0;
    if (channels <= 0 || H <= 0 || W <= 0) return 0;
    if (!flux_cuda_ensure_init()) return 0;

    int outH = H * 2;
    int outW = W * 2;
    int total = channels * outH * outW;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    upsample_nearest2x_nchw_kernel<<<blocks, threads, 0, g_stream>>>(d_out, d_in, channels, H, W);
    return cudaGetLastError() == cudaSuccess;
}
