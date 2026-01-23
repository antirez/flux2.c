/*
 * FLUX CUDA Acceleration - Implementation
 *
 * GPU-accelerated operations using NVIDIA CUDA and cuBLAS.
 * Inspired by ggml-cuda from stable-diffusion.cpp, but standalone.
 */

#include "flux_cuda.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ========================================================================
 * Error Handling Macros
 * ======================================================================== */

#define CUDA_CHECK(err) do { \
    cudaError_t e = (err); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        return; \
    } \
} while(0)

#define CUDA_CHECK_RET(err, ret) do { \
    cudaError_t e = (err); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        return ret; \
    } \
} while(0)

#define CUBLAS_CHECK(err) do { \
    cublasStatus_t e = (err); \
    if (e != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, (int)e); \
        return; \
    } \
} while(0)

/* ========================================================================
 * Global State
 * ======================================================================== */

static int g_initialized = 0;
static int g_available = 0;
static cublasHandle_t g_cublas = NULL;
static cudaStream_t g_stream = NULL;
static int g_batch_mode = 0;
static char g_device_name[256] = "Unknown";
static int g_compute_cap = 0;

/* ========================================================================
 * Weight Cache - Keep weights on GPU permanently
 * ======================================================================== */

#define WEIGHT_CACHE_SIZE 2048

typedef struct {
    const void *cpu_ptr;  /* Key: CPU address of weight */
    void *gpu_ptr;        /* Value: GPU copy */
    size_t size;
} weight_cache_entry_t;

static weight_cache_entry_t g_weight_cache[WEIGHT_CACHE_SIZE];
static int g_weight_cache_count = 0;

static void* weight_cache_get(const void *cpu_ptr) {
    for (int i = 0; i < g_weight_cache_count; i++) {
        if (g_weight_cache[i].cpu_ptr == cpu_ptr) {
            return g_weight_cache[i].gpu_ptr;
        }
    }
    return NULL;
}

static void* weight_cache_add(const void *cpu_ptr, size_t size) {
    if (g_weight_cache_count >= WEIGHT_CACHE_SIZE) return NULL;

    void *gpu_ptr = NULL;
    if (cudaMalloc(&gpu_ptr, size) != cudaSuccess) return NULL;
    if (cudaMemcpy(gpu_ptr, cpu_ptr, size, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(gpu_ptr);
        return NULL;
    }

    g_weight_cache[g_weight_cache_count].cpu_ptr = cpu_ptr;
    g_weight_cache[g_weight_cache_count].gpu_ptr = gpu_ptr;
    g_weight_cache[g_weight_cache_count].size = size;
    g_weight_cache_count++;

    return gpu_ptr;
}

static void weight_cache_clear(void) {
    for (int i = 0; i < g_weight_cache_count; i++) {
        if (g_weight_cache[i].gpu_ptr) cudaFree(g_weight_cache[i].gpu_ptr);
    }
    g_weight_cache_count = 0;
    memset(g_weight_cache, 0, sizeof(g_weight_cache));
}

/* ========================================================================
 * Scratch Buffers - Reusable GPU memory for activations
 * ======================================================================== */

static float *g_scratch_A = NULL;
static float *g_scratch_C = NULL;
static size_t g_scratch_A_size = 0;
static size_t g_scratch_C_size = 0;

static float* ensure_scratch(float **buf, size_t *current, size_t needed) {
    if (*current >= needed) return *buf;
    if (*buf) cudaFree(*buf);
    if (cudaMalloc((void**)buf, needed) != cudaSuccess) {
        *buf = NULL;
        *current = 0;
        return NULL;
    }
    *current = needed;
    return *buf;
}

static void free_scratch(void) {
    if (g_scratch_A) { cudaFree(g_scratch_A); g_scratch_A = NULL; g_scratch_A_size = 0; }
    if (g_scratch_C) { cudaFree(g_scratch_C); g_scratch_C = NULL; g_scratch_C_size = 0; }
}

/* ========================================================================
 * GPU Tensor Pool - Keep activations on GPU between operations
 * ======================================================================== */

#define GPU_TENSOR_POOL_SIZE 64

static struct {
    float *ptr;
    size_t size;
    int in_use;
} g_tensor_pool[GPU_TENSOR_POOL_SIZE];

int flux_cuda_tensor_get(size_t size) {
    if (!g_available) return -1;

    /* Find existing free tensor that fits */
    for (int i = 0; i < GPU_TENSOR_POOL_SIZE; i++) {
        if (!g_tensor_pool[i].in_use && g_tensor_pool[i].size >= size) {
            g_tensor_pool[i].in_use = 1;
            return i;
        }
    }

    /* Find empty slot and allocate */
    for (int i = 0; i < GPU_TENSOR_POOL_SIZE; i++) {
        if (g_tensor_pool[i].ptr == NULL) {
            if (cudaMalloc((void**)&g_tensor_pool[i].ptr, size) != cudaSuccess) {
                return -1;
            }
            g_tensor_pool[i].size = size;
            g_tensor_pool[i].in_use = 1;
            return i;
        }
    }

    return -1;  /* Pool full */
}

void flux_cuda_tensor_release(int id) {
    if (id >= 0 && id < GPU_TENSOR_POOL_SIZE) {
        g_tensor_pool[id].in_use = 0;
    }
}

float* flux_cuda_tensor_ptr(int id) {
    if (id < 0 || id >= GPU_TENSOR_POOL_SIZE) return NULL;
    return g_tensor_pool[id].ptr;
}

void flux_cuda_tensor_upload(int id, const float *data, size_t size) {
    if (id < 0 || id >= GPU_TENSOR_POOL_SIZE || !g_tensor_pool[id].ptr) return;
    cudaMemcpyAsync(g_tensor_pool[id].ptr, data, size, cudaMemcpyHostToDevice, g_stream);
}

void flux_cuda_tensor_download(int id, float *data, size_t size) {
    if (id < 0 || id >= GPU_TENSOR_POOL_SIZE || !g_tensor_pool[id].ptr) return;
    cudaMemcpyAsync(data, g_tensor_pool[id].ptr, size, cudaMemcpyDeviceToHost, g_stream);
    cudaStreamSynchronize(g_stream);
}

static void free_tensor_pool(void) {
    for (int i = 0; i < GPU_TENSOR_POOL_SIZE; i++) {
        if (g_tensor_pool[i].ptr) {
            cudaFree(g_tensor_pool[i].ptr);
            g_tensor_pool[i].ptr = NULL;
            g_tensor_pool[i].size = 0;
            g_tensor_pool[i].in_use = 0;
        }
    }
}

/* ========================================================================
 * Kernel Constants
 * ======================================================================== */

#define WARP_SIZE 32
#define BLOCK_1D 256
#define BLOCK_NORM 256

/* ========================================================================
 * Initialization
 * ======================================================================== */

int flux_cuda_init(void) {
    if (g_initialized) return g_available;
    g_initialized = 1;

    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        fprintf(stderr, "CUDA: No devices found\n");
        return 0;
    }

    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return 0;

    snprintf(g_device_name, sizeof(g_device_name), "%s", prop.name);
    g_compute_cap = prop.major * 10 + prop.minor;

    printf("CUDA: %s (SM %d.%d, %zu MB)\n", prop.name, prop.major, prop.minor,
           prop.totalGlobalMem / (1024 * 1024));

    if (cublasCreate(&g_cublas) != CUBLAS_STATUS_SUCCESS) return 0;
    if (cudaStreamCreate(&g_stream) != cudaSuccess) {
        cublasDestroy(g_cublas);
        return 0;
    }

    cublasSetStream(g_cublas, g_stream);
    if (g_compute_cap >= 70) cublasSetMathMode(g_cublas, CUBLAS_TF32_TENSOR_OP_MATH);

    g_available = 1;
    return 1;
}

int flux_cuda_available(void) { return g_available; }
const char* flux_cuda_device_name(void) { return g_device_name; }
int flux_cuda_compute_capability(void) { return g_compute_cap; }
int flux_cuda_kernels_available(void) { return g_available; }

void flux_cuda_cleanup(void) {
    weight_cache_clear();
    free_scratch();
    free_tensor_pool();
    if (g_stream) { cudaStreamDestroy(g_stream); g_stream = NULL; }
    if (g_cublas) { cublasDestroy(g_cublas); g_cublas = NULL; }
    g_available = 0;
    g_initialized = 0;
}

void flux_cuda_reset(void) {
    if (g_available) cudaStreamSynchronize(g_stream);
}

void flux_cuda_sync(void) {
    if (g_available) cudaStreamSynchronize(g_stream);
}

void flux_cuda_begin_batch(void) { g_batch_mode = 1; }
void flux_cuda_end_batch(void) { g_batch_mode = 0; flux_cuda_sync(); }
int flux_cuda_in_batch(void) { return g_batch_mode; }
size_t flux_cuda_memory_used(void) { return 0; }

/* ========================================================================
 * CUDA Kernels
 * ======================================================================== */

__global__ void k_silu(float *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        x[i] = v / (1.0f + expf(-v));
    }
}

__global__ void k_silu_mul(float *gate, const float *up, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = gate[i];
        gate[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}

__global__ void k_gelu(float *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        float inner = 0.7978845608f * (v + 0.044715f * v * v * v);
        x[i] = 0.5f * v * (1.0f + tanhf(inner));
    }
}

__global__ void k_add(float *a, const float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}

__global__ void k_mul(float *a, const float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] *= b[i];
}

__global__ void k_scale(float *a, float s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] *= s;
}

/* Gated residual: out[i] += gate[i % hidden] * x[i] */
__global__ void k_gated_add(float *out, const float *gate, const float *x,
                            int seq, int hidden) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq * hidden;
    if (i < total) {
        int h = i % hidden;
        out[i] += gate[h] * x[i];
    }
}

/* Split fused output: [seq, fused_dim] -> Q,K,V,gate,up
 * fused_dim = h*3 + mlp*2, layout per row: [Q(h), K(h), V(h), gate(mlp), up(mlp)]
 */
__global__ void k_split_fused(const float *fused, float *q, float *k, float *v,
                               float *gate, float *up,
                               int seq, int h, int mlp) {
    int s = blockIdx.x;
    if (s >= seq) return;

    int fused_dim = h * 3 + mlp * 2;
    const float *row = fused + s * fused_dim;

    /* Each thread handles multiple elements */
    for (int i = threadIdx.x; i < h; i += blockDim.x) {
        q[s * h + i] = row[i];
        k[s * h + i] = row[h + i];
        v[s * h + i] = row[h * 2 + i];
    }
    for (int i = threadIdx.x; i < mlp; i += blockDim.x) {
        gate[s * mlp + i] = row[h * 3 + i];
        up[s * mlp + i] = row[h * 3 + mlp + i];
    }
}

/* Concat: [attn_out, mlp_out] -> concat
 * concat layout per row: [attn(h), mlp(mlp)]
 */
__global__ void k_concat(float *concat, const float *attn, const float *mlp_out,
                         int seq, int h, int mlp) {
    int s = blockIdx.x;
    if (s >= seq) return;

    int concat_dim = h + mlp;
    float *out_row = concat + s * concat_dim;

    for (int i = threadIdx.x; i < h; i += blockDim.x) {
        out_row[i] = attn[s * h + i];
    }
    for (int i = threadIdx.x; i < mlp; i += blockDim.x) {
        out_row[h + i] = mlp_out[s * mlp + i];
    }
}

__global__ void k_rms_norm(float *out, const float *x, const float *w,
                            int seq, int hid, float eps) {
    int row = blockIdx.x;
    if (row >= seq) return;

    const float *xr = x + row * hid;
    float *outr = out + row * hid;

    __shared__ float ssum[BLOCK_NORM];
    float sum = 0.0f;
    for (int i = threadIdx.x; i < hid; i += blockDim.x) {
        float v = xr[i];
        sum += v * v;
    }
    ssum[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) ssum[threadIdx.x] += ssum[threadIdx.x + s];
        __syncthreads();
    }

    float rms = rsqrtf(ssum[0] / hid + eps);
    for (int i = threadIdx.x; i < hid; i += blockDim.x) {
        outr[i] = xr[i] * rms * w[i];
    }
}

__global__ void k_softmax(float *x, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    float *xr = x + row * cols;
    __shared__ float smax[BLOCK_NORM], ssum[BLOCK_NORM];

    float mx = -INFINITY;
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        mx = fmaxf(mx, xr[i]);
    smax[threadIdx.x] = mx;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smax[threadIdx.x] = fmaxf(smax[threadIdx.x], smax[threadIdx.x + s]);
        __syncthreads();
    }
    mx = smax[0];

    float sm = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float e = expf(xr[i] - mx);
        xr[i] = e;
        sm += e;
    }
    ssum[threadIdx.x] = sm;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) ssum[threadIdx.x] += ssum[threadIdx.x + s];
        __syncthreads();
    }
    sm = ssum[0];

    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        xr[i] /= sm;
}

__global__ void k_qk_rms_norm(float *q, float *k, const float *qw, const float *kw,
                               int seq, int heads, int hdim, float eps) {
    int idx = blockIdx.x;
    int s = idx / heads, h = idx % heads;
    if (s >= seq) return;

    float *qh = q + s * heads * hdim + h * hdim;
    float *kh = k + s * heads * hdim + h * hdim;

    __shared__ float sq[BLOCK_NORM], sk[BLOCK_NORM];
    float sumq = 0, sumk = 0;
    for (int i = threadIdx.x; i < hdim; i += blockDim.x) {
        sumq += qh[i] * qh[i];
        sumk += kh[i] * kh[i];
    }
    sq[threadIdx.x] = sumq;
    sk[threadIdx.x] = sumk;
    __syncthreads();

    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (threadIdx.x < st) {
            sq[threadIdx.x] += sq[threadIdx.x + st];
            sk[threadIdx.x] += sk[threadIdx.x + st];
        }
        __syncthreads();
    }

    float rmsq = rsqrtf(sq[0] / hdim + eps);
    float rmsk = rsqrtf(sk[0] / hdim + eps);

    for (int i = threadIdx.x; i < hdim; i += blockDim.x) {
        qh[i] = qh[i] * rmsq * qw[i];
        kh[i] = kh[i] * rmsk * kw[i];
    }
}

__global__ void k_adaln_norm(float *out, const float *x, const float *shift,
                              const float *scale, int seq, int hid, float eps) {
    int row = blockIdx.x;
    if (row >= seq) return;

    const float *xr = x + row * hid;
    float *outr = out + row * hid;

    __shared__ float smean[BLOCK_NORM], svar[BLOCK_NORM];
    float sm = 0, sv = 0;
    for (int i = threadIdx.x; i < hid; i += blockDim.x) sm += xr[i];
    smean[threadIdx.x] = sm;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smean[threadIdx.x] += smean[threadIdx.x + s];
        __syncthreads();
    }
    float mean = smean[0] / hid;

    for (int i = threadIdx.x; i < hid; i += blockDim.x) {
        float d = xr[i] - mean;
        sv += d * d;
    }
    svar[threadIdx.x] = sv;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) svar[threadIdx.x] += svar[threadIdx.x + s];
        __syncthreads();
    }
    float rstd = rsqrtf(svar[0] / hid + eps);

    for (int i = threadIdx.x; i < hid; i += blockDim.x) {
        float norm = (xr[i] - mean) * rstd;
        outr[i] = (1.0f + scale[i]) * norm + shift[i];
    }
}

__global__ void k_rope_2d(float *x, const float *cos_f, const float *sin_f,
                           int seq, int heads, int hdim, int axis_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq * heads * (axis_dim / 2);
    if (idx >= total) return;

    int s = idx / (heads * (axis_dim / 2));
    int rem = idx % (heads * (axis_dim / 2));
    int h = rem / (axis_dim / 2);
    int p = rem % (axis_dim / 2);

    int freq_idx = s * (axis_dim / 2) + p;
    float c = cos_f[freq_idx], sn = sin_f[freq_idx];

    int base = s * heads * hdim + h * hdim + p * 2;
    float x0 = x[base], x1 = x[base + 1];
    x[base] = x0 * c - x1 * sn;
    x[base + 1] = x0 * sn + x1 * c;
}

/* RoPE with sequence offset - applies to x starting at seq_offset
 * x layout: [total_seq, heads, head_dim]
 * cos/sin layout: [seq_len, head_dim] (full head_dim, not axis_dim)
 */
__global__ void k_rope_2d_offset(float *x, const float *cos_f, const float *sin_f,
                                  int seq_len, int seq_offset, int heads, int hdim, int axis_dim) {
    (void)axis_dim;  /* Not used - we apply to all head_dim pairs */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * heads * (hdim / 2);
    if (idx >= total) return;

    int s = idx / (heads * (hdim / 2));
    int rem = idx % (heads * (hdim / 2));
    int h = rem / (hdim / 2);
    int d = rem % (hdim / 2);  /* pair index: 0..63 for hdim=128 */

    /* cos/sin index: [s, d*2] */
    int freq_idx = s * hdim + d * 2;
    float c = cos_f[freq_idx];
    float sn = sin_f[freq_idx];

    /* x index with offset */
    int base = (seq_offset + s) * heads * hdim + h * hdim + d * 2;
    float x0 = x[base], x1 = x[base + 1];

    /* Complex rotation: (x0 + i*x1) * (cos + i*sin) */
    x[base] = x0 * c - x1 * sn;
    x[base + 1] = x1 * c + x0 * sn;  /* Note: x1*cos + x0*sin, not x0*sin + x1*cos */
}

/* ========================================================================
 * cuBLAS Matrix Multiplication
 * ======================================================================== */

void flux_cuda_sgemm(int ta, int tb, int M, int N, int K,
                     float alpha, const float *A, int lda,
                     const float *B, int ldb, float beta, float *C, int ldc) {
    if (!g_available) return;

    size_t szA = (size_t)(ta ? K * M : M * K) * sizeof(float);
    size_t szB = (size_t)(tb ? N * K : K * N) * sizeof(float);
    size_t szC = (size_t)M * N * sizeof(float);

    /* A = activations, use scratch buffer */
    float *dA = ensure_scratch(&g_scratch_A, &g_scratch_A_size, szA);
    if (!dA) return;
    CUDA_CHECK(cudaMemcpyAsync(dA, A, szA, cudaMemcpyHostToDevice, g_stream));

    /* B = weights, check cache first */
    float *dB = (float*)weight_cache_get(B);
    if (!dB) {
        dB = (float*)weight_cache_add(B, szB);
        if (!dB) return;  /* Cache full and can't allocate */
    }

    /* C = output, use scratch buffer */
    float *dC = ensure_scratch(&g_scratch_C, &g_scratch_C_size, szC);
    if (!dC) return;
    if (beta != 0.0f) CUDA_CHECK(cudaMemcpyAsync(dC, C, szC, cudaMemcpyHostToDevice, g_stream));

    cublasOperation_t opA = ta ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = tb ? CUBLAS_OP_T : CUBLAS_OP_N;

    CUBLAS_CHECK(cublasSgemm(g_cublas, opB, opA, N, M, K, &alpha,
                             dB, ldb, dA, lda, &beta, dC, ldc));

    CUDA_CHECK(cudaMemcpyAsync(C, dC, szC, cudaMemcpyDeviceToHost, g_stream));
    if (!g_batch_mode) cudaStreamSynchronize(g_stream);
}

/* GPU-to-GPU sgemm: works on tensor IDs, no CPU copies! */
int flux_cuda_sgemm_gpu(int ta, int tb, int M, int N, int K,
                        float alpha, int A_id, int lda,
                        const float *B, int ldb,
                        float beta, int C_id, int ldc) {
    if (!g_available) return -1;

    float *dA = flux_cuda_tensor_ptr(A_id);
    float *dC = flux_cuda_tensor_ptr(C_id);
    if (!dA || !dC) return -1;

    /* B = weights, check cache */
    size_t szB = (size_t)(tb ? N * K : K * N) * sizeof(float);
    float *dB = (float*)weight_cache_get(B);
    if (!dB) {
        dB = (float*)weight_cache_add(B, szB);
        if (!dB) return -1;
    }

    cublasOperation_t opA = ta ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = tb ? CUBLAS_OP_T : CUBLAS_OP_N;

    cublasStatus_t err = cublasSgemm(g_cublas, opB, opA, N, M, K, &alpha,
                                      dB, ldb, dA, lda, &beta, dC, ldc);
    return (err == CUBLAS_STATUS_SUCCESS) ? C_id : -1;
}

void flux_cuda_sgemm_bf16(int ta, int tb, int M, int N, int K,
                          float alpha, const float *A, int lda,
                          const uint16_t *B_bf16, int ldb,
                          float beta, float *C, int ldc) {
    if (!g_available) return;

    /* Convert bf16 to f32 */
    size_t szB = (size_t)(tb ? N * K : K * N);
    float *B_f32 = (float *)malloc(szB * sizeof(float));
    if (!B_f32) return;

    for (size_t i = 0; i < szB; i++) {
        uint32_t bits = ((uint32_t)B_bf16[i]) << 16;
        memcpy(&B_f32[i], &bits, sizeof(float));
    }

    flux_cuda_sgemm(ta, tb, M, N, K, alpha, A, lda, B_f32, ldb, beta, C, ldc);
    free(B_f32);
}

void flux_cuda_sgemm_batch(int ta, int tb, int M, int N, int K,
                           float alpha, const float *A, int lda, int strideA,
                           const float *B, int ldb, int strideB,
                           float beta, float *C, int ldc, int strideC, int batch) {
    for (int b = 0; b < batch; b++) {
        flux_cuda_sgemm(ta, tb, M, N, K, alpha,
                        A + b * strideA, lda, B + b * strideB, ldb,
                        beta, C + b * strideC, ldc);
    }
}

/* ========================================================================
 * C API Wrappers for Kernels
 * ======================================================================== */

void flux_cuda_silu(float *x, int n) {
    if (!g_available) return;
    float *dx; size_t sz = n * sizeof(float);
    CUDA_CHECK(cudaMalloc(&dx, sz));
    CUDA_CHECK(cudaMemcpyAsync(dx, x, sz, cudaMemcpyHostToDevice, g_stream));
    int blk = (n + BLOCK_1D - 1) / BLOCK_1D;
    k_silu<<<blk, BLOCK_1D, 0, g_stream>>>(dx, n);
    CUDA_CHECK(cudaMemcpyAsync(x, dx, sz, cudaMemcpyDeviceToHost, g_stream));
    if (!g_batch_mode) cudaStreamSynchronize(g_stream);
    cudaFree(dx);
}

void flux_cuda_gelu(float *x, int n) {
    if (!g_available) return;
    float *dx; size_t sz = n * sizeof(float);
    CUDA_CHECK(cudaMalloc(&dx, sz));
    CUDA_CHECK(cudaMemcpyAsync(dx, x, sz, cudaMemcpyHostToDevice, g_stream));
    int blk = (n + BLOCK_1D - 1) / BLOCK_1D;
    k_gelu<<<blk, BLOCK_1D, 0, g_stream>>>(dx, n);
    CUDA_CHECK(cudaMemcpyAsync(x, dx, sz, cudaMemcpyDeviceToHost, g_stream));
    if (!g_batch_mode) cudaStreamSynchronize(g_stream);
    cudaFree(dx);
}

void flux_cuda_silu_mul(float *gate, const float *up, int n) {
    if (!g_available) return;
    float *dg, *du; size_t sz = n * sizeof(float);
    CUDA_CHECK(cudaMalloc(&dg, sz)); CUDA_CHECK(cudaMalloc(&du, sz));
    CUDA_CHECK(cudaMemcpyAsync(dg, gate, sz, cudaMemcpyHostToDevice, g_stream));
    CUDA_CHECK(cudaMemcpyAsync(du, up, sz, cudaMemcpyHostToDevice, g_stream));
    int blk = (n + BLOCK_1D - 1) / BLOCK_1D;
    k_silu_mul<<<blk, BLOCK_1D, 0, g_stream>>>(dg, du, n);
    CUDA_CHECK(cudaMemcpyAsync(gate, dg, sz, cudaMemcpyDeviceToHost, g_stream));
    if (!g_batch_mode) cudaStreamSynchronize(g_stream);
    cudaFree(dg); cudaFree(du);
}

void flux_cuda_add_inplace(float *a, const float *b, int n) {
    if (!g_available) return;
    float *da, *db; size_t sz = n * sizeof(float);
    CUDA_CHECK(cudaMalloc(&da, sz)); CUDA_CHECK(cudaMalloc(&db, sz));
    CUDA_CHECK(cudaMemcpyAsync(da, a, sz, cudaMemcpyHostToDevice, g_stream));
    CUDA_CHECK(cudaMemcpyAsync(db, b, sz, cudaMemcpyHostToDevice, g_stream));
    k_add<<<(n + BLOCK_1D - 1) / BLOCK_1D, BLOCK_1D, 0, g_stream>>>(da, db, n);
    CUDA_CHECK(cudaMemcpyAsync(a, da, sz, cudaMemcpyDeviceToHost, g_stream));
    if (!g_batch_mode) cudaStreamSynchronize(g_stream);
    cudaFree(da); cudaFree(db);
}

void flux_cuda_mul_inplace(float *a, const float *b, int n) {
    if (!g_available) return;
    float *da, *db; size_t sz = n * sizeof(float);
    CUDA_CHECK(cudaMalloc(&da, sz)); CUDA_CHECK(cudaMalloc(&db, sz));
    CUDA_CHECK(cudaMemcpyAsync(da, a, sz, cudaMemcpyHostToDevice, g_stream));
    CUDA_CHECK(cudaMemcpyAsync(db, b, sz, cudaMemcpyHostToDevice, g_stream));
    k_mul<<<(n + BLOCK_1D - 1) / BLOCK_1D, BLOCK_1D, 0, g_stream>>>(da, db, n);
    CUDA_CHECK(cudaMemcpyAsync(a, da, sz, cudaMemcpyDeviceToHost, g_stream));
    if (!g_batch_mode) cudaStreamSynchronize(g_stream);
    cudaFree(da); cudaFree(db);
}

void flux_cuda_scale_inplace(float *a, float s, int n) {
    if (!g_available) return;
    float *da; size_t sz = n * sizeof(float);
    CUDA_CHECK(cudaMalloc(&da, sz));
    CUDA_CHECK(cudaMemcpyAsync(da, a, sz, cudaMemcpyHostToDevice, g_stream));
    k_scale<<<(n + BLOCK_1D - 1) / BLOCK_1D, BLOCK_1D, 0, g_stream>>>(da, s, n);
    CUDA_CHECK(cudaMemcpyAsync(a, da, sz, cudaMemcpyDeviceToHost, g_stream));
    if (!g_batch_mode) cudaStreamSynchronize(g_stream);
    cudaFree(da);
}

void flux_cuda_rms_norm(float *out, const float *x, const float *w,
                        int seq, int hid, float eps) {
    if (!g_available) return;
    float *dout, *dx, *dw;
    size_t szx = (size_t)seq * hid * sizeof(float), szw = hid * sizeof(float);
    CUDA_CHECK(cudaMalloc(&dout, szx)); CUDA_CHECK(cudaMalloc(&dx, szx)); CUDA_CHECK(cudaMalloc(&dw, szw));
    CUDA_CHECK(cudaMemcpyAsync(dx, x, szx, cudaMemcpyHostToDevice, g_stream));
    CUDA_CHECK(cudaMemcpyAsync(dw, w, szw, cudaMemcpyHostToDevice, g_stream));
    k_rms_norm<<<seq, BLOCK_NORM, 0, g_stream>>>(dout, dx, dw, seq, hid, eps);
    CUDA_CHECK(cudaMemcpyAsync(out, dout, szx, cudaMemcpyDeviceToHost, g_stream));
    if (!g_batch_mode) cudaStreamSynchronize(g_stream);
    cudaFree(dout); cudaFree(dx); cudaFree(dw);
}

void flux_cuda_softmax(float *x, int rows, int cols) {
    if (!g_available) return;
    float *dx; size_t sz = (size_t)rows * cols * sizeof(float);
    CUDA_CHECK(cudaMalloc(&dx, sz));
    CUDA_CHECK(cudaMemcpyAsync(dx, x, sz, cudaMemcpyHostToDevice, g_stream));
    k_softmax<<<rows, BLOCK_NORM, 0, g_stream>>>(dx, rows, cols);
    CUDA_CHECK(cudaMemcpyAsync(x, dx, sz, cudaMemcpyDeviceToHost, g_stream));
    if (!g_batch_mode) cudaStreamSynchronize(g_stream);
    cudaFree(dx);
}

void flux_cuda_qk_rms_norm(float *q, float *k, const float *qw, const float *kw,
                           int seq, int heads, int hdim, float eps) {
    if (!g_available) return;
    float *dq, *dk, *dqw, *dkw;
    size_t szqk = (size_t)seq * heads * hdim * sizeof(float), szw = hdim * sizeof(float);
    CUDA_CHECK(cudaMalloc(&dq, szqk)); CUDA_CHECK(cudaMalloc(&dk, szqk));
    CUDA_CHECK(cudaMalloc(&dqw, szw)); CUDA_CHECK(cudaMalloc(&dkw, szw));
    CUDA_CHECK(cudaMemcpyAsync(dq, q, szqk, cudaMemcpyHostToDevice, g_stream));
    CUDA_CHECK(cudaMemcpyAsync(dk, k, szqk, cudaMemcpyHostToDevice, g_stream));
    CUDA_CHECK(cudaMemcpyAsync(dqw, qw, szw, cudaMemcpyHostToDevice, g_stream));
    CUDA_CHECK(cudaMemcpyAsync(dkw, kw, szw, cudaMemcpyHostToDevice, g_stream));
    k_qk_rms_norm<<<seq * heads, BLOCK_NORM, 0, g_stream>>>(dq, dk, dqw, dkw, seq, heads, hdim, eps);
    CUDA_CHECK(cudaMemcpyAsync(q, dq, szqk, cudaMemcpyDeviceToHost, g_stream));
    CUDA_CHECK(cudaMemcpyAsync(k, dk, szqk, cudaMemcpyDeviceToHost, g_stream));
    if (!g_batch_mode) cudaStreamSynchronize(g_stream);
    cudaFree(dq); cudaFree(dk); cudaFree(dqw); cudaFree(dkw);
}

void flux_cuda_adaln_norm(float *out, const float *x, const float *shift,
                          const float *scale, int seq, int hid, float eps) {
    if (!g_available) return;
    float *dout, *dx, *dsh, *dsc;
    size_t szx = (size_t)seq * hid * sizeof(float), szm = hid * sizeof(float);
    CUDA_CHECK(cudaMalloc(&dout, szx)); CUDA_CHECK(cudaMalloc(&dx, szx));
    CUDA_CHECK(cudaMalloc(&dsh, szm)); CUDA_CHECK(cudaMalloc(&dsc, szm));
    CUDA_CHECK(cudaMemcpyAsync(dx, x, szx, cudaMemcpyHostToDevice, g_stream));
    CUDA_CHECK(cudaMemcpyAsync(dsh, shift, szm, cudaMemcpyHostToDevice, g_stream));
    CUDA_CHECK(cudaMemcpyAsync(dsc, scale, szm, cudaMemcpyHostToDevice, g_stream));
    k_adaln_norm<<<seq, BLOCK_NORM, 0, g_stream>>>(dout, dx, dsh, dsc, seq, hid, eps);
    CUDA_CHECK(cudaMemcpyAsync(out, dout, szx, cudaMemcpyDeviceToHost, g_stream));
    if (!g_batch_mode) cudaStreamSynchronize(g_stream);
    cudaFree(dout); cudaFree(dx); cudaFree(dsh); cudaFree(dsc);
}

void flux_cuda_rope_2d(float *x, const float *cos_f, const float *sin_f,
                       int seq, int heads, int hdim, int axis_dim) {
    if (!g_available) return;
    float *dx, *dc, *ds;
    size_t szx = (size_t)seq * heads * hdim * sizeof(float);
    size_t szf = (size_t)seq * (axis_dim / 2) * sizeof(float);
    CUDA_CHECK(cudaMalloc(&dx, szx)); CUDA_CHECK(cudaMalloc(&dc, szf)); CUDA_CHECK(cudaMalloc(&ds, szf));
    CUDA_CHECK(cudaMemcpyAsync(dx, x, szx, cudaMemcpyHostToDevice, g_stream));
    CUDA_CHECK(cudaMemcpyAsync(dc, cos_f, szf, cudaMemcpyHostToDevice, g_stream));
    CUDA_CHECK(cudaMemcpyAsync(ds, sin_f, szf, cudaMemcpyHostToDevice, g_stream));
    int total = seq * heads * (axis_dim / 2);
    k_rope_2d<<<(total + BLOCK_1D - 1) / BLOCK_1D, BLOCK_1D, 0, g_stream>>>(dx, dc, ds, seq, heads, hdim, axis_dim);
    CUDA_CHECK(cudaMemcpyAsync(x, dx, szx, cudaMemcpyDeviceToHost, g_stream));
    if (!g_batch_mode) cudaStreamSynchronize(g_stream);
    cudaFree(dx); cudaFree(dc); cudaFree(ds);
}

/* ========================================================================
 * GPU Tensor Operations - Work on tensors already on GPU
 * ======================================================================== */

void flux_cuda_gated_add_t(int out_id, const float *gate, int x_id, int seq, int hidden) {
    if (!g_available) return;
    float *d_out = flux_cuda_tensor_ptr(out_id);
    float *d_x = flux_cuda_tensor_ptr(x_id);
    if (!d_out || !d_x) return;

    /* Upload gate (small: just hidden floats) */
    float *d_gate;
    size_t gate_sz = hidden * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_gate, gate_sz));
    CUDA_CHECK(cudaMemcpyAsync(d_gate, gate, gate_sz, cudaMemcpyHostToDevice, g_stream));

    int total = seq * hidden;
    k_gated_add<<<(total + BLOCK_1D - 1) / BLOCK_1D, BLOCK_1D, 0, g_stream>>>(d_out, d_gate, d_x, seq, hidden);

    cudaFree(d_gate);
}

void flux_cuda_split_fused_t(int fused_id, int q_id, int k_id, int v_id,
                             int gate_id, int up_id, int seq, int h, int mlp) {
    if (!g_available) return;
    float *d_fused = flux_cuda_tensor_ptr(fused_id);
    float *d_q = flux_cuda_tensor_ptr(q_id);
    float *d_k = flux_cuda_tensor_ptr(k_id);
    float *d_v = flux_cuda_tensor_ptr(v_id);
    float *d_gate = flux_cuda_tensor_ptr(gate_id);
    float *d_up = flux_cuda_tensor_ptr(up_id);
    if (!d_fused || !d_q || !d_k || !d_v || !d_gate || !d_up) return;

    k_split_fused<<<seq, BLOCK_1D, 0, g_stream>>>(d_fused, d_q, d_k, d_v, d_gate, d_up, seq, h, mlp);
}

void flux_cuda_concat_t(int concat_id, int attn_id, int mlp_id, int seq, int h, int mlp) {
    if (!g_available) return;
    float *d_concat = flux_cuda_tensor_ptr(concat_id);
    float *d_attn = flux_cuda_tensor_ptr(attn_id);
    float *d_mlp = flux_cuda_tensor_ptr(mlp_id);
    if (!d_concat || !d_attn || !d_mlp) return;

    k_concat<<<seq, BLOCK_1D, 0, g_stream>>>(d_concat, d_attn, d_mlp, seq, h, mlp);
}

void flux_cuda_silu_t(int tensor_id, int n) {
    if (!g_available) return;
    float *d = flux_cuda_tensor_ptr(tensor_id);
    if (!d) return;
    k_silu<<<(n + BLOCK_1D - 1) / BLOCK_1D, BLOCK_1D, 0, g_stream>>>(d, n);
}

void flux_cuda_mul_t(int a_id, int b_id, int n) {
    if (!g_available) return;
    float *d_a = flux_cuda_tensor_ptr(a_id);
    float *d_b = flux_cuda_tensor_ptr(b_id);
    if (!d_a || !d_b) return;
    k_mul<<<(n + BLOCK_1D - 1) / BLOCK_1D, BLOCK_1D, 0, g_stream>>>(d_a, d_b, n);
}

void flux_cuda_adaln_t(int out_id, int x_id, const float *shift, const float *scale,
                       int seq, int hid, float eps) {
    if (!g_available) return;
    float *d_out = flux_cuda_tensor_ptr(out_id);
    float *d_x = flux_cuda_tensor_ptr(x_id);
    if (!d_out || !d_x) return;

    /* Upload shift/scale (small) */
    float *d_sh, *d_sc;
    size_t sz = hid * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_sh, sz)); CUDA_CHECK(cudaMalloc(&d_sc, sz));
    CUDA_CHECK(cudaMemcpyAsync(d_sh, shift, sz, cudaMemcpyHostToDevice, g_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_sc, scale, sz, cudaMemcpyHostToDevice, g_stream));

    /* Ensure previous uploads are complete */
    cudaStreamSynchronize(g_stream);

    k_adaln_norm<<<seq, BLOCK_NORM, 0, g_stream>>>(d_out, d_x, d_sh, d_sc, seq, hid, eps);

    cudaStreamSynchronize(g_stream);  /* Wait for kernel completion before freeing */
    cudaFree(d_sh); cudaFree(d_sc);
}

void flux_cuda_qk_norm_t(int q_id, int k_id, const float *qw, const float *kw,
                         int seq, int heads, int hdim, float eps) {
    if (!g_available) return;
    float *d_q = flux_cuda_tensor_ptr(q_id);
    float *d_k = flux_cuda_tensor_ptr(k_id);
    if (!d_q || !d_k) return;

    /* Upload weights */
    float *d_qw, *d_kw;
    size_t sz = hdim * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_qw, sz)); CUDA_CHECK(cudaMalloc(&d_kw, sz));
    CUDA_CHECK(cudaMemcpyAsync(d_qw, qw, sz, cudaMemcpyHostToDevice, g_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_kw, kw, sz, cudaMemcpyHostToDevice, g_stream));

    k_qk_rms_norm<<<seq * heads, BLOCK_NORM, 0, g_stream>>>(d_q, d_k, d_qw, d_kw, seq, heads, hdim, eps);

    cudaFree(d_qw); cudaFree(d_kw);
}

void flux_cuda_rope_t(int x_id, const float *cos_f, const float *sin_f,
                      int seq, int heads, int hdim, int axis_dim) {
    if (!g_available) return;
    float *d_x = flux_cuda_tensor_ptr(x_id);
    if (!d_x) return;

    size_t szf = (size_t)seq * (axis_dim / 2) * sizeof(float);
    float *d_c, *d_s;
    CUDA_CHECK(cudaMalloc(&d_c, szf)); CUDA_CHECK(cudaMalloc(&d_s, szf));
    CUDA_CHECK(cudaMemcpyAsync(d_c, cos_f, szf, cudaMemcpyHostToDevice, g_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_s, sin_f, szf, cudaMemcpyHostToDevice, g_stream));

    int total = seq * heads * (axis_dim / 2);
    k_rope_2d<<<(total + BLOCK_1D - 1) / BLOCK_1D, BLOCK_1D, 0, g_stream>>>(d_x, d_c, d_s, seq, heads, hdim, axis_dim);

    cudaFree(d_c); cudaFree(d_s);
}

/* RoPE with offset - applies to portion of tensor starting at seq_offset */
void flux_cuda_rope_offset_t(int x_id, const float *cos_f, const float *sin_f,
                              int seq_len, int seq_offset, int heads, int hdim, int axis_dim) {
    if (!g_available) return;
    float *d_x = flux_cuda_tensor_ptr(x_id);
    if (!d_x) return;

    /* cos/sin are [seq_len, hdim] */
    size_t szf = (size_t)seq_len * hdim * sizeof(float);
    float *d_c, *d_s;
    CUDA_CHECK(cudaMalloc(&d_c, szf)); CUDA_CHECK(cudaMalloc(&d_s, szf));
    CUDA_CHECK(cudaMemcpyAsync(d_c, cos_f, szf, cudaMemcpyHostToDevice, g_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_s, sin_f, szf, cudaMemcpyHostToDevice, g_stream));

    int total = seq_len * heads * (hdim / 2);
    k_rope_2d_offset<<<(total + BLOCK_1D - 1) / BLOCK_1D, BLOCK_1D, 0, g_stream>>>(
        d_x, d_c, d_s, seq_len, seq_offset, heads, hdim, axis_dim);

    cudaFree(d_c); cudaFree(d_s);
}

/* ========================================================================
 * Attention and Conv2D - Fall back to CPU for now
 * ======================================================================== */

int flux_cuda_conv2d(float *out, const float *in, const float *weight, const float *bias,
                     int batch, int in_ch, int out_ch, int H, int W, int kH, int kW,
                     int stride, int padding) {
    (void)out; (void)in; (void)weight; (void)bias;
    (void)batch; (void)in_ch; (void)out_ch; (void)H; (void)W; (void)kH; (void)kW;
    (void)stride; (void)padding;
    return 0;  /* Fall back to CPU */
}

int flux_cuda_attention_fused(float *out, const float *Q, const float *K, const float *V,
                              int seq_q, int seq_k, int num_heads, int head_dim, float scale) {
    (void)out; (void)Q; (void)K; (void)V;
    (void)seq_q; (void)seq_k; (void)num_heads; (void)head_dim; (void)scale;
    return 0;  /* Fall back to CPU */
}

int flux_cuda_causal_attention(float *out, const float *Q, const float *K, const float *V,
                               const int *attention_mask, int seq, int num_q_heads,
                               int num_kv_heads, int head_dim, float scale) {
    (void)out; (void)Q; (void)K; (void)V; (void)attention_mask;
    (void)seq; (void)num_q_heads; (void)num_kv_heads; (void)head_dim; (void)scale;
    return 0;  /* Fall back to CPU */
}

/* ========================================================================
 * GPU Tensor Attention - operates on tensor IDs
 * Q,K,V layout: [seq, heads, head_dim] (packed as [seq, hidden])
 * Uses cuBLAS batched gemm for all heads in parallel
 * ======================================================================== */

/* Transpose kernel: [seq, heads, hdim] -> [heads, seq, hdim] */
__global__ void k_transpose_shd_to_hsd(float *out, const float *in,
                                        int seq, int heads, int hdim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq * heads * hdim;
    if (idx >= total) return;

    int s = idx / (heads * hdim);
    int rem = idx % (heads * hdim);
    int h = rem / hdim;
    int d = rem % hdim;

    /* in: [s, h, d], out: [h, s, d] */
    int out_idx = h * seq * hdim + s * hdim + d;
    out[out_idx] = in[idx];
}

/* Transpose kernel: [heads, seq, hdim] -> [seq, heads, hdim] */
__global__ void k_transpose_hsd_to_shd(float *out, const float *in,
                                        int seq, int heads, int hdim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq * heads * hdim;
    if (idx >= total) return;

    int h = idx / (seq * hdim);
    int rem = idx % (seq * hdim);
    int s = rem / hdim;
    int d = rem % hdim;

    /* in: [h, s, d], out: [s, h, d] */
    int out_idx = s * heads * hdim + h * hdim + d;
    out[out_idx] = in[idx];
}

/* Softmax per row for attention scores [heads, seq_q, seq_k] */
__global__ void k_softmax_attention(float *scores, int heads, int seq_q, int seq_k, float scale) {
    int idx = blockIdx.x;  /* One block per row */
    if (idx >= heads * seq_q) return;

    float *row = scores + idx * seq_k;

    /* Scale and find max */
    __shared__ float smax[256];
    float mx = -INFINITY;
    for (int i = threadIdx.x; i < seq_k; i += blockDim.x) {
        row[i] *= scale;
        mx = fmaxf(mx, row[i]);
    }
    smax[threadIdx.x] = mx;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smax[threadIdx.x] = fmaxf(smax[threadIdx.x], smax[threadIdx.x + s]);
        __syncthreads();
    }
    mx = smax[0];

    /* Exp and sum */
    __shared__ float ssum[256];
    float sm = 0;
    for (int i = threadIdx.x; i < seq_k; i += blockDim.x) {
        float e = expf(row[i] - mx);
        row[i] = e;
        sm += e;
    }
    ssum[threadIdx.x] = sm;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) ssum[threadIdx.x] += ssum[threadIdx.x + s];
        __syncthreads();
    }
    sm = ssum[0];

    /* Normalize */
    for (int i = threadIdx.x; i < seq_k; i += blockDim.x) {
        row[i] /= sm;
    }
}

int flux_cuda_attention_t(int out_id, int q_id, int k_id, int v_id,
                          int seq, int heads, int hdim, float scale) {
    if (!g_available) return 0;

    float *d_q = flux_cuda_tensor_ptr(q_id);
    float *d_k = flux_cuda_tensor_ptr(k_id);
    float *d_v = flux_cuda_tensor_ptr(v_id);
    float *d_out = flux_cuda_tensor_ptr(out_id);
    if (!d_q || !d_k || !d_v || !d_out) return 0;

    size_t sz_qkv = (size_t)seq * heads * hdim * sizeof(float);
    size_t sz_scores = (size_t)heads * seq * seq * sizeof(float);

    /* Use tensor pool for transposed buffers */
    int t_qt = flux_cuda_tensor_get(sz_qkv);
    int t_kt = flux_cuda_tensor_get(sz_qkv);
    int t_vt = flux_cuda_tensor_get(sz_qkv);
    int t_ot = flux_cuda_tensor_get(sz_qkv);
    int t_scores = flux_cuda_tensor_get(sz_scores);

    if (t_qt < 0 || t_kt < 0 || t_vt < 0 || t_ot < 0 || t_scores < 0) {
        flux_cuda_tensor_release(t_qt); flux_cuda_tensor_release(t_kt);
        flux_cuda_tensor_release(t_vt); flux_cuda_tensor_release(t_ot);
        flux_cuda_tensor_release(t_scores);
        return 0;
    }

    float *d_qt = flux_cuda_tensor_ptr(t_qt);
    float *d_kt = flux_cuda_tensor_ptr(t_kt);
    float *d_vt = flux_cuda_tensor_ptr(t_vt);
    float *d_ot = flux_cuda_tensor_ptr(t_ot);
    float *d_scores = flux_cuda_tensor_ptr(t_scores);

    int total = seq * heads * hdim;

    /* Transpose Q,K,V from [seq, heads, hdim] to [heads, seq, hdim] */
    k_transpose_shd_to_hsd<<<(total + BLOCK_1D - 1) / BLOCK_1D, BLOCK_1D, 0, g_stream>>>(d_qt, d_q, seq, heads, hdim);
    k_transpose_shd_to_hsd<<<(total + BLOCK_1D - 1) / BLOCK_1D, BLOCK_1D, 0, g_stream>>>(d_kt, d_k, seq, heads, hdim);
    k_transpose_shd_to_hsd<<<(total + BLOCK_1D - 1) / BLOCK_1D, BLOCK_1D, 0, g_stream>>>(d_vt, d_v, seq, heads, hdim);

    /* Batched GEMM: scores = Q @ K^T for all heads */
    float alpha = 1.0f, beta = 0.0f;
    long long strideQ = seq * hdim;
    long long strideK = seq * hdim;
    long long strideS = seq * seq;

    cublasSgemmStridedBatched(g_cublas,
        CUBLAS_OP_T, CUBLAS_OP_N,
        seq, seq, hdim,
        &alpha,
        d_kt, hdim, strideK,
        d_qt, hdim, strideQ,
        &beta,
        d_scores, seq, strideS,
        heads);

    /* Softmax with scale */
    k_softmax_attention<<<heads * seq, 256, 0, g_stream>>>(d_scores, heads, seq, seq, scale);

    /* Batched GEMM: out = scores @ V for all heads */
    long long strideV = seq * hdim;
    long long strideO = seq * hdim;

    cublasSgemmStridedBatched(g_cublas,
        CUBLAS_OP_N, CUBLAS_OP_N,
        hdim, seq, seq,
        &alpha,
        d_vt, hdim, strideV,
        d_scores, seq, strideS,
        &beta,
        d_ot, hdim, strideO,
        heads);

    /* Transpose output back */
    k_transpose_hsd_to_shd<<<(total + BLOCK_1D - 1) / BLOCK_1D, BLOCK_1D, 0, g_stream>>>(d_out, d_ot, seq, heads, hdim);

    flux_cuda_tensor_release(t_qt); flux_cuda_tensor_release(t_kt);
    flux_cuda_tensor_release(t_vt); flux_cuda_tensor_release(t_ot);
    flux_cuda_tensor_release(t_scores);
    return 1;
}

/* Joint attention for double blocks: Q attends to concatenated K,V
 * img_q: [img_seq, heads, hdim], txt_q: [txt_seq, heads, hdim]
 * cat_k, cat_v: [total_seq, heads, hdim] where total_seq = txt_seq + img_seq
 * Returns 1 on success
 */
int flux_cuda_joint_attention_t(int img_out_id, int txt_out_id,
                                 int img_q_id, int txt_q_id,
                                 int cat_k_id, int cat_v_id,
                                 int img_seq, int txt_seq, int heads, int hdim, float scale) {
    if (!g_available) return 0;

    int total_seq = img_seq + txt_seq;

    float *d_img_q = flux_cuda_tensor_ptr(img_q_id);
    float *d_txt_q = flux_cuda_tensor_ptr(txt_q_id);
    float *d_cat_k = flux_cuda_tensor_ptr(cat_k_id);
    float *d_cat_v = flux_cuda_tensor_ptr(cat_v_id);
    float *d_img_out = flux_cuda_tensor_ptr(img_out_id);
    float *d_txt_out = flux_cuda_tensor_ptr(txt_out_id);

    if (!d_img_q || !d_txt_q || !d_cat_k || !d_cat_v || !d_img_out || !d_txt_out) return 0;

    /* Use tensor pool for transposed buffers */
    size_t sz_img_q = (size_t)img_seq * heads * hdim * sizeof(float);
    size_t sz_txt_q = (size_t)txt_seq * heads * hdim * sizeof(float);
    size_t sz_cat = (size_t)total_seq * heads * hdim * sizeof(float);
    size_t sz_img_scores = (size_t)heads * img_seq * total_seq * sizeof(float);
    size_t sz_txt_scores = (size_t)heads * txt_seq * total_seq * sizeof(float);

    int t_img_qt = flux_cuda_tensor_get(sz_img_q);
    int t_txt_qt = flux_cuda_tensor_get(sz_txt_q);
    int t_cat_kt = flux_cuda_tensor_get(sz_cat);
    int t_cat_vt = flux_cuda_tensor_get(sz_cat);
    int t_img_ot = flux_cuda_tensor_get(sz_img_q);
    int t_txt_ot = flux_cuda_tensor_get(sz_txt_q);
    int t_img_scores = flux_cuda_tensor_get(sz_img_scores);
    int t_txt_scores = flux_cuda_tensor_get(sz_txt_scores);

    if (t_img_qt < 0 || t_txt_qt < 0 || t_cat_kt < 0 || t_cat_vt < 0 ||
        t_img_ot < 0 || t_txt_ot < 0 || t_img_scores < 0 || t_txt_scores < 0) {
        flux_cuda_tensor_release(t_img_qt); flux_cuda_tensor_release(t_txt_qt);
        flux_cuda_tensor_release(t_cat_kt); flux_cuda_tensor_release(t_cat_vt);
        flux_cuda_tensor_release(t_img_ot); flux_cuda_tensor_release(t_txt_ot);
        flux_cuda_tensor_release(t_img_scores); flux_cuda_tensor_release(t_txt_scores);
        return 0;
    }

    float *d_img_qt = flux_cuda_tensor_ptr(t_img_qt);
    float *d_txt_qt = flux_cuda_tensor_ptr(t_txt_qt);
    float *d_cat_kt = flux_cuda_tensor_ptr(t_cat_kt);
    float *d_cat_vt = flux_cuda_tensor_ptr(t_cat_vt);
    float *d_img_ot = flux_cuda_tensor_ptr(t_img_ot);
    float *d_txt_ot = flux_cuda_tensor_ptr(t_txt_ot);
    float *d_img_scores = flux_cuda_tensor_ptr(t_img_scores);
    float *d_txt_scores = flux_cuda_tensor_ptr(t_txt_scores);

    /* Transpose all inputs */
    int img_total = img_seq * heads * hdim;
    int txt_total = txt_seq * heads * hdim;
    int cat_total = total_seq * heads * hdim;

    k_transpose_shd_to_hsd<<<(img_total + BLOCK_1D - 1) / BLOCK_1D, BLOCK_1D, 0, g_stream>>>(d_img_qt, d_img_q, img_seq, heads, hdim);
    k_transpose_shd_to_hsd<<<(txt_total + BLOCK_1D - 1) / BLOCK_1D, BLOCK_1D, 0, g_stream>>>(d_txt_qt, d_txt_q, txt_seq, heads, hdim);
    k_transpose_shd_to_hsd<<<(cat_total + BLOCK_1D - 1) / BLOCK_1D, BLOCK_1D, 0, g_stream>>>(d_cat_kt, d_cat_k, total_seq, heads, hdim);
    k_transpose_shd_to_hsd<<<(cat_total + BLOCK_1D - 1) / BLOCK_1D, BLOCK_1D, 0, g_stream>>>(d_cat_vt, d_cat_v, total_seq, heads, hdim);

    float alpha = 1.0f, beta = 0.0f;

    /* Image attention: img_Q @ cat_K^T -> [heads, img_seq, total_seq] */
    cublasSgemmStridedBatched(g_cublas,
        CUBLAS_OP_T, CUBLAS_OP_N,
        total_seq, img_seq, hdim,
        &alpha,
        d_cat_kt, hdim, (long long)total_seq * hdim,
        d_img_qt, hdim, (long long)img_seq * hdim,
        &beta,
        d_img_scores, total_seq, (long long)img_seq * total_seq,
        heads);

    /* Softmax for image scores */
    k_softmax_attention<<<heads * img_seq, 256, 0, g_stream>>>(d_img_scores, heads, img_seq, total_seq, scale);

    /* Image output: scores @ cat_V -> [heads, img_seq, hdim] */
    cublasSgemmStridedBatched(g_cublas,
        CUBLAS_OP_N, CUBLAS_OP_N,
        hdim, img_seq, total_seq,
        &alpha,
        d_cat_vt, hdim, (long long)total_seq * hdim,
        d_img_scores, total_seq, (long long)img_seq * total_seq,
        &beta,
        d_img_ot, hdim, (long long)img_seq * hdim,
        heads);

    /* Text attention: txt_Q @ cat_K^T -> [heads, txt_seq, total_seq] */
    cublasSgemmStridedBatched(g_cublas,
        CUBLAS_OP_T, CUBLAS_OP_N,
        total_seq, txt_seq, hdim,
        &alpha,
        d_cat_kt, hdim, (long long)total_seq * hdim,
        d_txt_qt, hdim, (long long)txt_seq * hdim,
        &beta,
        d_txt_scores, total_seq, (long long)txt_seq * total_seq,
        heads);

    /* Softmax for text scores */
    k_softmax_attention<<<heads * txt_seq, 256, 0, g_stream>>>(d_txt_scores, heads, txt_seq, total_seq, scale);

    /* Text output: scores @ cat_V -> [heads, txt_seq, hdim] */
    cublasSgemmStridedBatched(g_cublas,
        CUBLAS_OP_N, CUBLAS_OP_N,
        hdim, txt_seq, total_seq,
        &alpha,
        d_cat_vt, hdim, (long long)total_seq * hdim,
        d_txt_scores, total_seq, (long long)txt_seq * total_seq,
        &beta,
        d_txt_ot, hdim, (long long)txt_seq * hdim,
        heads);

    /* Transpose outputs back */
    k_transpose_hsd_to_shd<<<(img_total + BLOCK_1D - 1) / BLOCK_1D, BLOCK_1D, 0, g_stream>>>(d_img_out, d_img_ot, img_seq, heads, hdim);
    k_transpose_hsd_to_shd<<<(txt_total + BLOCK_1D - 1) / BLOCK_1D, BLOCK_1D, 0, g_stream>>>(d_txt_out, d_txt_ot, txt_seq, heads, hdim);

    flux_cuda_tensor_release(t_img_qt); flux_cuda_tensor_release(t_txt_qt);
    flux_cuda_tensor_release(t_cat_kt); flux_cuda_tensor_release(t_cat_vt);
    flux_cuda_tensor_release(t_img_ot); flux_cuda_tensor_release(t_txt_ot);
    flux_cuda_tensor_release(t_img_scores); flux_cuda_tensor_release(t_txt_scores);
    return 1;
}
