/*
 * FLUX ROCm Acceleration - Core Implementation
 *
 * GPU infrastructure for AMD GPUs using HIP/ROCm.
 * Key design: keep data on GPU, minimize PCIe transfers.
 */

#include "flux_rocm.h"
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <unordered_map>
#include <mutex>

/* ========================================================================
 * Error Handling
 * ======================================================================== */

#define HIP_CHECK(call) do { \
    hipError_t err = (call); \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP Error: %s at %s:%d\n", \
                hipGetErrorString(err), __FILE__, __LINE__); \
        return; \
    } \
} while(0)

#define HIP_CHECK_RET(call, ret) do { \
    hipError_t err = (call); \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP Error: %s at %s:%d\n", \
                hipGetErrorString(err), __FILE__, __LINE__); \
        return (ret); \
    } \
} while(0)

#define HIPBLAS_CHECK(call) do { \
    hipblasStatus_t status = (call); \
    if (status != HIPBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "hipBLAS Error: %d at %s:%d\n", \
                (int)status, __FILE__, __LINE__); \
    } \
} while(0)

/* ========================================================================
 * Global State
 * ======================================================================== */

static int g_initialized = 0;
static hipDevice_t g_device = 0;
static hipStream_t g_stream = nullptr;
static hipblasHandle_t g_hipblas = nullptr;

/* Memory tracking */
static size_t g_memory_used = 0;
static std::mutex g_mutex;

/* ========================================================================
 * GPU Tensor Implementation
 * ======================================================================== */

struct flux_rocm_tensor {
    void *data;           /* GPU pointer */
    size_t num_elements;
    size_t bytes;
    int is_bf16;          /* 1 if bf16, 0 if f32 */
    int persistent;       /* Don't return to pool */
    int ref_count;
};

/* Simple buffer pool - maps size to list of free buffers */
static std::unordered_map<size_t, std::vector<void*>> g_buffer_pool;
static std::mutex g_pool_mutex;

static void* pool_alloc(size_t bytes) {
    std::lock_guard<std::mutex> lock(g_pool_mutex);
    
    auto it = g_buffer_pool.find(bytes);
    if (it != g_buffer_pool.end() && !it->second.empty()) {
        void *ptr = it->second.back();
        it->second.pop_back();
        return ptr;
    }
    
    /* Allocate new buffer */
    void *ptr = nullptr;
    hipError_t err = hipMalloc(&ptr, bytes);
    if (err != hipSuccess) {
        fprintf(stderr, "hipMalloc failed for %zu bytes: %s\n", 
                bytes, hipGetErrorString(err));
        return nullptr;
    }
    
    g_memory_used += bytes;
    return ptr;
}

static void pool_free(void *ptr, size_t bytes) {
    if (!ptr) return;
    
    std::lock_guard<std::mutex> lock(g_pool_mutex);
    g_buffer_pool[bytes].push_back(ptr);
}

static void pool_clear() {
    std::lock_guard<std::mutex> lock(g_pool_mutex);
    
    for (auto &kv : g_buffer_pool) {
        for (void *ptr : kv.second) {
            hipFree(ptr);
            g_memory_used -= kv.first;
        }
    }
    g_buffer_pool.clear();
}

/* ========================================================================
 * Weight Cache - Keep model weights on GPU
 * ======================================================================== */

struct weight_cache_entry {
    void *gpu_ptr;
    size_t bytes;
    uint64_t content_hash;  /* Hash of first N bytes to detect content changes */
};

/* Simple hash of first N bytes to detect if tensor content changed.
 * Needed for mmap mode where same address can hold different weights. */
static uint64_t compute_content_hash(const void *data, size_t bytes) {
    const uint8_t *p = (const uint8_t *)data;
    /* Sample first 64 bytes, last 64 bytes, and a few in the middle */
    size_t sample_size = 64;
    uint64_t hash = bytes;  /* Include size in hash */
    
    /* Hash first bytes */
    size_t n = (bytes < sample_size) ? bytes : sample_size;
    for (size_t i = 0; i < n; i++) {
        hash = hash * 31 + p[i];
    }
    
    /* Hash some middle bytes if tensor is large enough */
    if (bytes > sample_size * 3) {
        size_t mid = bytes / 2;
        for (size_t i = 0; i < sample_size; i++) {
            hash = hash * 31 + p[mid + i];
        }
    }
    
    /* Hash last bytes */
    if (bytes > sample_size) {
        for (size_t i = 0; i < sample_size; i++) {
            hash = hash * 31 + p[bytes - sample_size + i];
        }
    }
    
    return hash;
}

static std::unordered_map<const void*, weight_cache_entry> g_weight_cache;
static std::mutex g_weight_mutex;

/* Get or upload weight tensor to GPU */
/* Exposed for compat layer */
void* get_cached_weight(const void *cpu_ptr, size_t bytes) {
    std::lock_guard<std::mutex> lock(g_weight_mutex);
    
    /* Compute content hash to detect if data changed (needed for mmap mode
     * where same virtual address can hold different weights) */
    uint64_t hash = compute_content_hash(cpu_ptr, bytes);
    
    auto it = g_weight_cache.find(cpu_ptr);
    if (it != g_weight_cache.end()) {
        /* Check if this is the same tensor (size AND content must match) */
        if (it->second.bytes == bytes && it->second.content_hash == hash) {
            return it->second.gpu_ptr;
        }
        /* Different tensor at same address - evict old one */
        hipFree(it->second.gpu_ptr);
        g_memory_used -= it->second.bytes;
        g_weight_cache.erase(it);
    }
    
    /* Upload to GPU */
    void *gpu_ptr = nullptr;
    hipError_t err = hipMalloc(&gpu_ptr, bytes);
    if (err != hipSuccess) {
        fprintf(stderr, "Weight cache: hipMalloc failed for %zu bytes\n", bytes);
        return nullptr;
    }
    
    err = hipMemcpy(gpu_ptr, cpu_ptr, bytes, hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        fprintf(stderr, "Weight cache: hipMemcpy failed\n");
        hipFree(gpu_ptr);
        return nullptr;
    }
    
    g_weight_cache[cpu_ptr] = {gpu_ptr, bytes, hash};
    g_memory_used += bytes;
    
    return gpu_ptr;
}

static void weight_cache_clear() {
    std::lock_guard<std::mutex> lock(g_weight_mutex);
    
    for (auto &kv : g_weight_cache) {
        hipFree(kv.second.gpu_ptr);
        g_memory_used -= kv.second.bytes;
    }
    g_weight_cache.clear();
}

/* ========================================================================
 * Initialization / Cleanup
 * ======================================================================== */

extern "C" int flux_rocm_init(void) {
    if (g_initialized) return 1;
    
    /* Check for HIP devices */
    int device_count = 0;
    hipError_t err = hipGetDeviceCount(&device_count);
    if (err != hipSuccess || device_count == 0) {
        fprintf(stderr, "ROCm: No HIP devices found\n");
        return 0;
    }
    
    /* Use first device */
    err = hipSetDevice(0);
    if (err != hipSuccess) {
        fprintf(stderr, "ROCm: Failed to set device\n");
        return 0;
    }
    
    /* Get device info */
    hipDeviceProp_t props;
    hipGetDeviceProperties(&props, 0);
    fprintf(stderr, "ROCm: Using %s (%s, %zu MB VRAM)\n",
            props.name, props.gcnArchName, props.totalGlobalMem / (1024*1024));
    
    /* Create stream */
    err = hipStreamCreate(&g_stream);
    if (err != hipSuccess) {
        fprintf(stderr, "ROCm: Failed to create stream\n");
        return 0;
    }
    
    /* Initialize hipBLAS */
    hipblasStatus_t blas_status = hipblasCreate(&g_hipblas);
    if (blas_status != HIPBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "ROCm: Failed to initialize hipBLAS\n");
        hipStreamDestroy(g_stream);
        return 0;
    }
    
    /* Associate hipBLAS with our stream */
    hipblasSetStream(g_hipblas, g_stream);
    
    g_initialized = 1;
    fprintf(stderr, "ROCm: Acceleration enabled\n");
    return 1;
}

extern "C" int flux_rocm_available(void) {
    return g_initialized;
}

extern "C" void flux_rocm_cleanup(void) {
    if (!g_initialized) return;
    
    pool_clear();
    weight_cache_clear();
    
    if (g_hipblas) {
        hipblasDestroy(g_hipblas);
        g_hipblas = nullptr;
    }
    
    if (g_stream) {
        hipStreamDestroy(g_stream);
        g_stream = nullptr;
    }
    
    g_initialized = 0;
}

extern "C" void flux_rocm_reset(void) {
    pool_clear();
    /* Note: Don't clear weight cache - weights should persist */
}

extern "C" void flux_rocm_sync(void) {
    if (g_stream) {
        hipStreamSynchronize(g_stream);
    }
}

extern "C" size_t flux_rocm_memory_used(void) {
    return g_memory_used;
}

/* ========================================================================
 * GPU Tensor API
 * ======================================================================== */

extern "C" flux_rocm_tensor_t flux_rocm_tensor_create(const float *data, size_t num_elements) {
    if (!g_initialized) return nullptr;
    
    flux_rocm_tensor *t = new flux_rocm_tensor;
    t->num_elements = num_elements;
    t->bytes = num_elements * sizeof(float);
    t->is_bf16 = 0;
    t->persistent = 0;
    t->ref_count = 1;
    
    t->data = pool_alloc(t->bytes);
    if (!t->data) {
        delete t;
        return nullptr;
    }
    
    hipError_t err = hipMemcpyAsync(t->data, data, t->bytes, 
                                     hipMemcpyHostToDevice, g_stream);
    if (err != hipSuccess) {
        pool_free(t->data, t->bytes);
        delete t;
        return nullptr;
    }
    
    return t;
}

extern "C" flux_rocm_tensor_t flux_rocm_tensor_alloc(size_t num_elements) {
    if (!g_initialized) return nullptr;
    
    flux_rocm_tensor *t = new flux_rocm_tensor;
    t->num_elements = num_elements;
    t->bytes = num_elements * sizeof(float);
    t->is_bf16 = 0;
    t->persistent = 0;
    t->ref_count = 1;
    
    t->data = pool_alloc(t->bytes);
    if (!t->data) {
        delete t;
        return nullptr;
    }
    
    return t;
}

extern "C" flux_rocm_tensor_t flux_rocm_tensor_alloc_persistent(size_t num_elements) {
    flux_rocm_tensor_t t = flux_rocm_tensor_alloc(num_elements);
    if (t) t->persistent = 1;
    return t;
}

extern "C" void flux_rocm_tensor_set_persistent(flux_rocm_tensor_t tensor, int persistent) {
    if (tensor) tensor->persistent = persistent;
}

extern "C" void flux_rocm_tensor_read(flux_rocm_tensor_t tensor, float *out, size_t num_elements) {
    if (!tensor || !out) return;
    
    size_t bytes = num_elements * sizeof(float);
    if (bytes > tensor->bytes) bytes = tensor->bytes;
    
    hipMemcpyAsync(out, tensor->data, bytes, hipMemcpyDeviceToHost, g_stream);
    hipStreamSynchronize(g_stream);  /* Must sync for CPU to see data */
}

extern "C" void flux_rocm_tensor_write(flux_rocm_tensor_t tensor, const float *data, size_t num_elements) {
    if (!tensor || !data) return;
    
    size_t bytes = num_elements * sizeof(float);
    if (bytes > tensor->bytes) bytes = tensor->bytes;
    
    hipMemcpyAsync(tensor->data, data, bytes, hipMemcpyHostToDevice, g_stream);
}

extern "C" void flux_rocm_tensor_free(flux_rocm_tensor_t tensor) {
    if (!tensor) return;
    
    if (!tensor->persistent) {
        pool_free(tensor->data, tensor->bytes);
    }
    delete tensor;
}

extern "C" size_t flux_rocm_tensor_size(flux_rocm_tensor_t tensor) {
    return tensor ? tensor->num_elements : 0;
}

extern "C" int flux_rocm_tensor_is_bf16(flux_rocm_tensor_t tensor) {
    return tensor ? tensor->is_bf16 : 0;
}

extern "C" void *flux_rocm_tensor_gpu_ptr(flux_rocm_tensor_t tensor) {
    return tensor ? tensor->data : nullptr;
}

extern "C" flux_rocm_tensor_t flux_rocm_tensor_alloc_bf16(size_t num_elements) {
    if (!g_initialized) return nullptr;
    
    flux_rocm_tensor *t = new flux_rocm_tensor;
    t->num_elements = num_elements;
    t->bytes = num_elements * sizeof(uint16_t);  /* bf16 = 2 bytes */
    t->is_bf16 = 1;
    t->persistent = 0;
    t->ref_count = 1;
    
    t->data = pool_alloc(t->bytes);
    if (!t->data) {
        delete t;
        return nullptr;
    }
    
    return t;
}

/* ========================================================================
 * BLAS Operations (hipBLAS)
 * ======================================================================== */

/* Internal helper: get GPU pointer for data (from cache or tensor) */
static float* get_gpu_ptr(const float *cpu_or_gpu, size_t bytes, bool is_weight) {
    /* TODO: Detect if pointer is already on GPU */
    /* For now, assume weights need caching, activations are already GPU tensors */
    if (is_weight) {
        return (float*)get_cached_weight(cpu_or_gpu, bytes);
    }
    return (float*)cpu_or_gpu;  /* Assume already on GPU */
}

extern "C" void flux_rocm_sgemm(int transpose_a, int transpose_b,
                                int M, int N, int K,
                                float alpha,
                                const float *A, int lda,
                                const float *B, int ldb,
                                float beta,
                                float *C, int ldc) {
    if (!g_initialized || !g_hipblas) return;
    
    /* hipBLAS uses column-major, so we swap A and B */
    hipblasOperation_t opA = transpose_b ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    hipblasOperation_t opB = transpose_a ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    
    /* Note: For row-major C = A @ B, we compute C^T = B^T @ A^T in col-major */
    HIPBLAS_CHECK(hipblasSgemm(g_hipblas,
                               opA, opB,
                               N, M, K,
                               &alpha,
                               B, ldb,
                               A, lda,
                               &beta,
                               C, ldc));
}

extern "C" void flux_rocm_sgemm_batch(int transpose_a, int transpose_b,
                                      int M, int N, int K,
                                      float alpha,
                                      const float *A, int lda, int stride_a,
                                      const float *B, int ldb, int stride_b,
                                      float beta,
                                      float *C, int ldc, int stride_c,
                                      int batch_count) {
    if (!g_initialized || !g_hipblas) return;
    
    hipblasOperation_t opA = transpose_b ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    hipblasOperation_t opB = transpose_a ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    
    HIPBLAS_CHECK(hipblasSgemmStridedBatched(g_hipblas,
                                             opA, opB,
                                             N, M, K,
                                             &alpha,
                                             B, ldb, stride_b,
                                             A, lda, stride_a,
                                             &beta,
                                             C, ldc, stride_c,
                                             batch_count));
}

/* ========================================================================
 * Linear Layer
 * ======================================================================== */

extern "C" flux_rocm_tensor_t flux_rocm_linear(flux_rocm_tensor_t x,
                                               const float *W,
                                               int seq_len, int in_dim, int out_dim) {
    if (!g_initialized || !x) return nullptr;
    
    /* Allocate output */
    flux_rocm_tensor_t out = flux_rocm_tensor_alloc(seq_len * out_dim);
    if (!out) return nullptr;
    
    /* Get weight on GPU (cached) */
    float *W_gpu = (float*)get_cached_weight(W, out_dim * in_dim * sizeof(float));
    if (!W_gpu) {
        flux_rocm_tensor_free(out);
        return nullptr;
    }
    
    /* out = x @ W^T */
    /* x: [seq, in_dim], W: [out_dim, in_dim], out: [seq, out_dim] */
    flux_rocm_sgemm(0, 1,  /* no trans A, trans B */
                    seq_len, out_dim, in_dim,
                    1.0f,
                    (float*)x->data, in_dim,
                    W_gpu, in_dim,
                    0.0f,
                    (float*)out->data, out_dim);
    
    return out;
}

extern "C" flux_rocm_tensor_t flux_rocm_linear_bf16(flux_rocm_tensor_t x,
                                                    const uint16_t *W_bf16,
                                                    int seq_len, int in_dim, int out_dim) {
    /* TODO: Implement bf16 GEMM using hipblasGemmEx or custom kernel */
    /* For now, fall back to f32 by converting weights */
    fprintf(stderr, "ROCm: bf16 linear not yet implemented, using f32 fallback\n");
    
    /* This is inefficient but gets us started */
    /* In production, use hipblasGemmEx with HIPBLAS_R_16BF */
    return nullptr;
}

/* ========================================================================
 * Batch/Chain API (Placeholder)
 * ======================================================================== */

static int g_in_batch = 0;
static int g_in_chain = 0;

extern "C" void flux_rocm_batch_begin(void) { g_in_batch = 1; }
extern "C" void flux_rocm_batch_end(void) { 
    g_in_batch = 0; 
    flux_rocm_sync();
}
extern "C" int flux_rocm_in_batch(void) { return g_in_batch; }

extern "C" void flux_rocm_chain_begin(void) { g_in_chain = 1; }
extern "C" void flux_rocm_chain_end(void) { 
    g_in_chain = 0; 
    flux_rocm_sync();
}
extern "C" int flux_rocm_in_chain(void) { return g_in_chain; }

/* ========================================================================
 * Stub implementations (to be filled in)
 * ======================================================================== */

extern "C" int flux_rocm_kernels_available(void) {
    return g_initialized;  /* Will be true once we add kernels */
}

extern "C" void flux_rocm_warmup_bf16(const uint16_t *bf16_weights, size_t num_elements) {
    /* Pre-upload bf16 weights to GPU */
    get_cached_weight(bf16_weights, num_elements * sizeof(uint16_t));
}

extern "C" int flux_rocm_bf16_available(void) {
    /* RDNA 3+ has good bf16 support */
    return g_initialized;
}

/* ========================================================================
 * Kernel declarations (from flux_rocm_kernels.hip)
 * ======================================================================== */

extern "C" {
void flux_rocm_kernel_silu(float *x, int n, hipStream_t stream);
void flux_rocm_kernel_silu_mul(float *gate, const float *up, int n, hipStream_t stream);
void flux_rocm_kernel_rms_norm(float *out, const float *x, const float *weight,
                               int seq_len, int hidden, float eps, hipStream_t stream);
void flux_rocm_kernel_qk_rms_norm(float *q, float *k,
                                  const float *q_weight, const float *k_weight,
                                  int seq, int heads, int head_dim, float eps, hipStream_t stream);
void flux_rocm_kernel_adaln_norm(float *out, const float *x,
                                 const float *shift, const float *scale,
                                 int seq_len, int hidden, float eps, hipStream_t stream);
void flux_rocm_kernel_softmax(float *x, int rows, int cols, hipStream_t stream);
void flux_rocm_kernel_gated_add(float *out, const float *gate, const float *proj,
                                int seq, int hidden, hipStream_t stream);
void flux_rocm_kernel_rope_2d(float *x, const float *cos_freq, const float *sin_freq,
                              int seq, int heads, int head_dim, int axis_dim, hipStream_t stream);
void flux_rocm_kernel_rope_unified(float *q, float *k,
                                   const float *txt_cos, const float *txt_sin,
                                   const float *img_cos, const float *img_sin,
                                   int seq, int img_offset, int heads, int head_dim, int axis_dim,
                                   hipStream_t stream);
void flux_rocm_kernel_transpose_to_heads(float *out, const float *in,
                                         int seq, int heads, int head_dim,
                                         hipStream_t stream);
void flux_rocm_kernel_transpose_from_heads(float *out, const float *in,
                                           int seq, int heads, int head_dim,
                                           hipStream_t stream);
}

/* Helper to get the global stream */
hipStream_t get_stream() { return g_stream; }

/* ========================================================================
 * Kernel Wrappers
 * ======================================================================== */

extern "C" void flux_rocm_rms_norm(float *out, const float *x, const float *weight,
                                   int seq_len, int hidden, float eps) {
    if (!g_initialized) return;
    flux_rocm_kernel_rms_norm(out, x, weight, seq_len, hidden, eps, g_stream);
}

extern "C" void flux_rocm_qk_rms_norm(float *q, float *k,
                                      const float *q_weight, const float *k_weight,
                                      int seq, int heads, int head_dim, float eps) {
    if (!g_initialized) return;
    flux_rocm_kernel_qk_rms_norm(q, k, q_weight, k_weight, seq, heads, head_dim, eps, g_stream);
}

extern "C" void flux_rocm_adaln_norm(float *out, const float *x,
                                     const float *shift, const float *scale,
                                     int seq_len, int hidden, float eps) {
    if (!g_initialized) return;
    flux_rocm_kernel_adaln_norm(out, x, shift, scale, seq_len, hidden, eps, g_stream);
}

extern "C" void flux_rocm_silu(float *x, int n) {
    if (!g_initialized) return;
    flux_rocm_kernel_silu(x, n, g_stream);
}

extern "C" void flux_rocm_silu_mul(float *gate, const float *up, int n) {
    if (!g_initialized) return;
    flux_rocm_kernel_silu_mul(gate, up, n, g_stream);
}

extern "C" void flux_rocm_softmax(float *x, int rows, int cols) {
    if (!g_initialized) return;
    flux_rocm_kernel_softmax(x, rows, cols, g_stream);
}

extern "C" void flux_rocm_gated_add(float *out, const float *gate,
                                    const float *proj, int seq, int hidden) {
    if (!g_initialized) return;
    flux_rocm_kernel_gated_add(out, gate, proj, seq, hidden, g_stream);
}

extern "C" void flux_rocm_rope_2d(float *x, const float *cos_freq, const float *sin_freq,
                                  int seq, int heads, int head_dim, int axis_dim) {
    if (!g_initialized) return;
    flux_rocm_kernel_rope_2d(x, cos_freq, sin_freq, seq, heads, head_dim, axis_dim, g_stream);
}

extern "C" void flux_rocm_rope_unified(float *q, float *k,
                                       const float *txt_cos, const float *txt_sin,
                                       const float *img_cos, const float *img_sin,
                                       int seq, int img_offset, int heads, int head_dim, int axis_dim) {
    if (!g_initialized) return;
    flux_rocm_kernel_rope_unified(q, k, txt_cos, txt_sin, img_cos, img_sin,
                                  seq, img_offset, heads, head_dim, axis_dim, g_stream);
}

/*
 * Attention using batched GEMM.
 * 
 * Q, K, V: [seq, num_heads * head_dim] (input layout)
 * out: [seq, num_heads * head_dim]
 * 
 * Internally we transpose to [num_heads, seq, head_dim] for batched ops.
 */
extern "C" int flux_rocm_attention_fused(float *out,
                                         const float *Q, const float *K, const float *V,
                                         int seq_q, int seq_k, int num_heads, int head_dim,
                                         float scale) {
    if (!g_initialized || !g_hipblas) return 0;
    
    int hidden = num_heads * head_dim;
    
    /* Allocate workspace for transposed tensors and scores */
    size_t qkv_size = num_heads * seq_q * head_dim;
    size_t scores_size = num_heads * seq_q * seq_k;
    
    /* OOM guard: scores matrix can explode at large sizes
     * E.g., 1024x1024 image: seq~16k, scores~24*16k*16k = 6.6B floats = 26GB
     * Fall back to CPU path if scores would exceed 2GB */
    size_t scores_bytes = scores_size * sizeof(float);
    const size_t MAX_SCORES_BYTES = 2ULL * 1024 * 1024 * 1024;  /* 2GB limit */
    if (scores_bytes > MAX_SCORES_BYTES) {
        return 0;  /* Fall back to CPU */
    }
    
    float *Q_t = (float*)pool_alloc(qkv_size * sizeof(float));
    float *K_t = (float*)pool_alloc(num_heads * seq_k * head_dim * sizeof(float));
    float *V_t = (float*)pool_alloc(num_heads * seq_k * head_dim * sizeof(float));
    float *scores = (float*)pool_alloc(scores_size * sizeof(float));
    float *out_t = (float*)pool_alloc(qkv_size * sizeof(float));
    
    if (!Q_t || !K_t || !V_t || !scores || !out_t) {
        if (Q_t) pool_free(Q_t, qkv_size * sizeof(float));
        if (K_t) pool_free(K_t, num_heads * seq_k * head_dim * sizeof(float));
        if (V_t) pool_free(V_t, num_heads * seq_k * head_dim * sizeof(float));
        if (scores) pool_free(scores, scores_size * sizeof(float));
        if (out_t) pool_free(out_t, qkv_size * sizeof(float));
        return 0;
    }
    
    /* GPU Transpose: [seq, heads*head_dim] -> [heads, seq, head_dim] */
    flux_rocm_kernel_transpose_to_heads(Q_t, Q, seq_q, num_heads, head_dim, g_stream);
    flux_rocm_kernel_transpose_to_heads(K_t, K, seq_k, num_heads, head_dim, g_stream);
    flux_rocm_kernel_transpose_to_heads(V_t, V, seq_k, num_heads, head_dim, g_stream);
    
    /* Batched GEMM 1: scores = Q @ K^T
     * Q_t: [heads, seq_q, head_dim]
     * K_t: [heads, seq_k, head_dim]
     * scores: [heads, seq_q, seq_k]
     */
    float alpha = scale;
    float beta = 0.0f;
    
    HIPBLAS_CHECK(hipblasSgemmStridedBatched(g_hipblas,
        HIPBLAS_OP_T, HIPBLAS_OP_N,  /* K^T @ Q in col-major = Q @ K^T in row-major */
        seq_k, seq_q, head_dim,
        &alpha,
        K_t, head_dim, seq_k * head_dim,
        Q_t, head_dim, seq_q * head_dim,
        &beta,
        scores, seq_k, seq_q * seq_k,
        num_heads));
    
    /* Softmax on scores */
    flux_rocm_kernel_softmax(scores, num_heads * seq_q, seq_k, g_stream);
    
    /* Batched GEMM 2: out = scores @ V
     * scores: [heads, seq_q, seq_k]
     * V_t: [heads, seq_k, head_dim]
     * out_t: [heads, seq_q, head_dim]
     */
    alpha = 1.0f;
    HIPBLAS_CHECK(hipblasSgemmStridedBatched(g_hipblas,
        HIPBLAS_OP_N, HIPBLAS_OP_N,  /* V @ scores^T in col-major = scores @ V in row-major */
        head_dim, seq_q, seq_k,
        &alpha,
        V_t, head_dim, seq_k * head_dim,
        scores, seq_k, seq_q * seq_k,
        &beta,
        out_t, head_dim, seq_q * head_dim,
        num_heads));
    
    /* GPU Transpose output back: [heads, seq, head_dim] -> [seq, heads*head_dim] */
    flux_rocm_kernel_transpose_from_heads(out, out_t, seq_q, num_heads, head_dim, g_stream);
    
    /* Cleanup */
    pool_free(Q_t, qkv_size * sizeof(float));
    pool_free(K_t, num_heads * seq_k * head_dim * sizeof(float));
    pool_free(V_t, num_heads * seq_k * head_dim * sizeof(float));
    pool_free(scores, scores_size * sizeof(float));
    pool_free(out_t, qkv_size * sizeof(float));
    
    return 1;
}

/* ========================================================================
 * GPU Tensor Wrapper Operations
 * These operate on flux_rocm_tensor_t, keeping data on GPU throughout.
 * ======================================================================== */

extern "C" void flux_rocm_gpu_adaln_norm(flux_rocm_tensor_t out, flux_rocm_tensor_t x,
                                         const float *shift, const float *scale,
                                         int seq, int hidden, float eps) {
    if (!g_initialized || !out || !x) return;
    
    /* Upload shift/scale to GPU - DON'T cache since they change per forward pass */
    size_t param_size = hidden * sizeof(float);
    float *shift_gpu = (float*)pool_alloc(param_size);
    float *scale_gpu = (float*)pool_alloc(param_size);
    
    if (!shift_gpu || !scale_gpu) {
        if (shift_gpu) pool_free(shift_gpu, param_size);
        if (scale_gpu) pool_free(scale_gpu, param_size);
        return;
    }
    
    hipMemcpyAsync(shift_gpu, shift, param_size, hipMemcpyHostToDevice, g_stream);
    hipMemcpyAsync(scale_gpu, scale, param_size, hipMemcpyHostToDevice, g_stream);
    
    flux_rocm_kernel_adaln_norm((float*)out->data, (float*)x->data,
                                shift_gpu, scale_gpu,
                                seq, hidden, eps, g_stream);
    
    pool_free(shift_gpu, param_size);
    pool_free(scale_gpu, param_size);
}

extern "C" void flux_rocm_gpu_qk_rms_norm(flux_rocm_tensor_t q, flux_rocm_tensor_t k,
                                          const float *q_weight, const float *k_weight,
                                          int seq, int heads, int head_dim, float eps) {
    if (!g_initialized || !q || !k) return;
    
    /* Upload weights to GPU (they're small, head_dim size) - cache OK since they're model weights */
    float *q_w_gpu = (float*)get_cached_weight(q_weight, head_dim * sizeof(float));
    float *k_w_gpu = (float*)get_cached_weight(k_weight, head_dim * sizeof(float));
    if (!q_w_gpu || !k_w_gpu) return;
    
    flux_rocm_kernel_qk_rms_norm((float*)q->data, (float*)k->data,
                                 q_w_gpu, k_w_gpu,
                                 seq, heads, head_dim, eps, g_stream);
}

extern "C" void flux_rocm_gpu_rope_unified(flux_rocm_tensor_t q, flux_rocm_tensor_t k,
                                           const float *txt_cos, const float *txt_sin,
                                           const float *img_cos, const float *img_sin,
                                           int seq, int img_offset, int heads, int head_dim, int axis_dim) {
    if (!g_initialized || !q || !k) return;
    
    /* Upload RoPE frequencies to GPU - DON'T cache since they change per forward pass */
    size_t txt_size = img_offset * head_dim * sizeof(float);
    size_t img_size = (seq - img_offset) * head_dim * sizeof(float);
    
    float *txt_cos_gpu = (float*)pool_alloc(txt_size);
    float *txt_sin_gpu = (float*)pool_alloc(txt_size);
    float *img_cos_gpu = (float*)pool_alloc(img_size);
    float *img_sin_gpu = (float*)pool_alloc(img_size);
    
    if (!txt_cos_gpu || !txt_sin_gpu || !img_cos_gpu || !img_sin_gpu) {
        if (txt_cos_gpu) pool_free(txt_cos_gpu, txt_size);
        if (txt_sin_gpu) pool_free(txt_sin_gpu, txt_size);
        if (img_cos_gpu) pool_free(img_cos_gpu, img_size);
        if (img_sin_gpu) pool_free(img_sin_gpu, img_size);
        return;
    }
    
    hipMemcpyAsync(txt_cos_gpu, txt_cos, txt_size, hipMemcpyHostToDevice, g_stream);
    hipMemcpyAsync(txt_sin_gpu, txt_sin, txt_size, hipMemcpyHostToDevice, g_stream);
    hipMemcpyAsync(img_cos_gpu, img_cos, img_size, hipMemcpyHostToDevice, g_stream);
    hipMemcpyAsync(img_sin_gpu, img_sin, img_size, hipMemcpyHostToDevice, g_stream);
    
    flux_rocm_kernel_rope_unified((float*)q->data, (float*)k->data,
                                  txt_cos_gpu, txt_sin_gpu,
                                  img_cos_gpu, img_sin_gpu,
                                  seq, img_offset, heads, head_dim, axis_dim, g_stream);
    
    pool_free(txt_cos_gpu, txt_size);
    pool_free(txt_sin_gpu, txt_size);
    pool_free(img_cos_gpu, img_size);
    pool_free(img_sin_gpu, img_size);
}

extern "C" void flux_rocm_gpu_silu_mul(flux_rocm_tensor_t gate, flux_rocm_tensor_t up, int n) {
    if (!g_initialized || !gate || !up) return;
    flux_rocm_kernel_silu_mul((float*)gate->data, (float*)up->data, n, g_stream);
}

extern "C" void flux_rocm_gpu_gated_add(flux_rocm_tensor_t out, const float *gate,
                                        flux_rocm_tensor_t proj, int seq, int hidden) {
    if (!g_initialized || !out || !proj) return;
    
    /* Gate is small (hidden size), upload to GPU - DON'T cache since it changes */
    size_t gate_size = hidden * sizeof(float);
    float *gate_gpu = (float*)pool_alloc(gate_size);
    if (!gate_gpu) return;
    
    hipMemcpyAsync(gate_gpu, gate, gate_size, hipMemcpyHostToDevice, g_stream);
    
    flux_rocm_kernel_gated_add((float*)out->data, gate_gpu, (float*)proj->data,
                               seq, hidden, g_stream);
    
    pool_free(gate_gpu, gate_size);
}

extern "C" int flux_rocm_gpu_attention_fused(flux_rocm_tensor_t out,
                                             flux_rocm_tensor_t Q, flux_rocm_tensor_t K, flux_rocm_tensor_t V,
                                             int seq_q, int seq_k, int num_heads, int head_dim, float scale) {
    if (!g_initialized || !out || !Q || !K || !V) return 0;
    
    return flux_rocm_attention_fused((float*)out->data,
                                     (float*)Q->data, (float*)K->data, (float*)V->data,
                                     seq_q, seq_k, num_heads, head_dim, scale);
}

extern "C" void flux_rocm_gpu_split_qkv_mlp(flux_rocm_tensor_t fused,
                                            flux_rocm_tensor_t q, flux_rocm_tensor_t k, flux_rocm_tensor_t v,
                                            flux_rocm_tensor_t gate, flux_rocm_tensor_t up,
                                            int seq, int hidden, int mlp_hidden) {
    if (!g_initialized || !fused || !q || !k || !v || !gate || !up) return;
    
    int fused_dim = hidden * 3 + mlp_hidden * 2;
    float *src = (float*)fused->data;
    
    for (int s = 0; s < seq; s++) {
        float *row = src + s * fused_dim;
        hipMemcpyAsync((float*)q->data + s * hidden, row, hidden * sizeof(float),
                       hipMemcpyDeviceToDevice, g_stream);
        hipMemcpyAsync((float*)k->data + s * hidden, row + hidden, hidden * sizeof(float),
                       hipMemcpyDeviceToDevice, g_stream);
        hipMemcpyAsync((float*)v->data + s * hidden, row + hidden * 2, hidden * sizeof(float),
                       hipMemcpyDeviceToDevice, g_stream);
        hipMemcpyAsync((float*)gate->data + s * mlp_hidden, row + hidden * 3, mlp_hidden * sizeof(float),
                       hipMemcpyDeviceToDevice, g_stream);
        hipMemcpyAsync((float*)up->data + s * mlp_hidden, row + hidden * 3 + mlp_hidden, mlp_hidden * sizeof(float),
                       hipMemcpyDeviceToDevice, g_stream);
    }
}

extern "C" void flux_rocm_gpu_concat_attn_mlp(flux_rocm_tensor_t attn, flux_rocm_tensor_t mlp,
                                              flux_rocm_tensor_t out, int seq, int hidden, int mlp_hidden) {
    if (!g_initialized || !attn || !mlp || !out) return;
    
    int out_dim = hidden + mlp_hidden;
    
    for (int s = 0; s < seq; s++) {
        hipMemcpyAsync((float*)out->data + s * out_dim, 
                       (float*)attn->data + s * hidden, hidden * sizeof(float),
                       hipMemcpyDeviceToDevice, g_stream);
        hipMemcpyAsync((float*)out->data + s * out_dim + hidden,
                       (float*)mlp->data + s * mlp_hidden, mlp_hidden * sizeof(float),
                       hipMemcpyDeviceToDevice, g_stream);
    }
}

extern "C" flux_rocm_tensor_t flux_rocm_gpu_linear(flux_rocm_tensor_t x,
                                                   const float *W,
                                                   int seq_len, int in_dim, int out_dim) {
    if (!g_initialized || !x) return nullptr;
    
    flux_rocm_tensor_t out = flux_rocm_tensor_alloc(seq_len * out_dim);
    if (!out) return nullptr;
    
    float *W_gpu = (float*)get_cached_weight(W, out_dim * in_dim * sizeof(float));
    if (!W_gpu) {
        flux_rocm_tensor_free(out);
        return nullptr;
    }
    
    flux_rocm_sgemm(0, 1,
                    seq_len, out_dim, in_dim,
                    1.0f,
                    (float*)x->data, in_dim,
                    W_gpu, in_dim,
                    0.0f,
                    (float*)out->data, out_dim);
    
    return out;
}

/* ========================================================================
 * Double Block Support Functions
 * ======================================================================== */

extern "C" void flux_rocm_gpu_rope_2d(flux_rocm_tensor_t x,
                                      const float *cos_freq, const float *sin_freq,
                                      int seq, int heads, int head_dim) {
    if (!g_initialized || !x) return;
    
    /* Upload RoPE frequencies to GPU - DON'T cache since they change */
    size_t freq_size = seq * head_dim * sizeof(float);
    float *cos_gpu = (float*)pool_alloc(freq_size);
    float *sin_gpu = (float*)pool_alloc(freq_size);
    
    if (!cos_gpu || !sin_gpu) {
        if (cos_gpu) pool_free(cos_gpu, freq_size);
        if (sin_gpu) pool_free(sin_gpu, freq_size);
        return;
    }
    
    hipMemcpyAsync(cos_gpu, cos_freq, freq_size, hipMemcpyHostToDevice, g_stream);
    hipMemcpyAsync(sin_gpu, sin_freq, freq_size, hipMemcpyHostToDevice, g_stream);
    
    flux_rocm_kernel_rope_2d((float*)x->data, cos_gpu, sin_gpu,
                             seq, heads, head_dim, head_dim, g_stream);
    
    pool_free(cos_gpu, freq_size);
    pool_free(sin_gpu, freq_size);
}

extern "C" void flux_rocm_gpu_concat_tensors(flux_rocm_tensor_t txt, flux_rocm_tensor_t img,
                                             flux_rocm_tensor_t out,
                                             int txt_seq, int img_seq, int hidden) {
    if (!g_initialized || !txt || !img || !out) return;
    
    /* Copy txt first, then img (matches Python Flux2: [txt, img]) */
    hipMemcpyAsync((float*)out->data, (float*)txt->data, 
                   txt_seq * hidden * sizeof(float),
                   hipMemcpyDeviceToDevice, g_stream);
    hipMemcpyAsync((float*)out->data + txt_seq * hidden, (float*)img->data,
                   img_seq * hidden * sizeof(float),
                   hipMemcpyDeviceToDevice, g_stream);
}

extern "C" flux_rocm_tensor_t flux_rocm_gpu_swiglu_ffn(flux_rocm_tensor_t x,
                                                       const float *gate_weight,
                                                       const float *up_weight,
                                                       const float *down_weight,
                                                       int seq, int hidden, int mlp_hidden) {
    if (!g_initialized || !x) return nullptr;
    
    /* Allocate intermediate tensors */
    flux_rocm_tensor_t gate = flux_rocm_tensor_alloc(seq * mlp_hidden);
    flux_rocm_tensor_t up = flux_rocm_tensor_alloc(seq * mlp_hidden);
    flux_rocm_tensor_t out = flux_rocm_tensor_alloc(seq * hidden);
    
    if (!gate || !up || !out) {
        if (gate) flux_rocm_tensor_free(gate);
        if (up) flux_rocm_tensor_free(up);
        if (out) flux_rocm_tensor_free(out);
        return nullptr;
    }
    
    /* Get cached weights */
    float *gate_w_gpu = (float*)get_cached_weight(gate_weight, mlp_hidden * hidden * sizeof(float));
    float *up_w_gpu = (float*)get_cached_weight(up_weight, mlp_hidden * hidden * sizeof(float));
    float *down_w_gpu = (float*)get_cached_weight(down_weight, hidden * mlp_hidden * sizeof(float));
    
    if (!gate_w_gpu || !up_w_gpu || !down_w_gpu) {
        flux_rocm_tensor_free(gate);
        flux_rocm_tensor_free(up);
        flux_rocm_tensor_free(out);
        return nullptr;
    }
    
    /* gate = x @ gate_weight.T, up = x @ up_weight.T */
    flux_rocm_sgemm(0, 1, seq, mlp_hidden, hidden,
                    1.0f, (float*)x->data, hidden,
                    gate_w_gpu, hidden,
                    0.0f, (float*)gate->data, mlp_hidden);
    
    flux_rocm_sgemm(0, 1, seq, mlp_hidden, hidden,
                    1.0f, (float*)x->data, hidden,
                    up_w_gpu, hidden,
                    0.0f, (float*)up->data, mlp_hidden);
    
    /* SwiGLU: gate = silu(gate) * up */
    flux_rocm_kernel_silu_mul((float*)gate->data, (float*)up->data, seq * mlp_hidden, g_stream);
    
    /* out = gate @ down_weight.T */
    flux_rocm_sgemm(0, 1, seq, hidden, mlp_hidden,
                    1.0f, (float*)gate->data, mlp_hidden,
                    down_w_gpu, mlp_hidden,
                    0.0f, (float*)out->data, hidden);
    
    /* Cleanup intermediates */
    flux_rocm_tensor_free(gate);
    flux_rocm_tensor_free(up);
    
    return out;
}
