/*
 * FLUX Math Kernels - Implementation
 *
 * Math operations for FLUX inference.
 * Uses Metal/MPS on Apple Silicon, BLAS otherwise.
 */

#include "flux_kernels.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <errno.h>

/* Use Metal for GPU acceleration on Apple Silicon */
#ifdef USE_METAL
#include "flux_metal.h"
#endif

/* Use BLAS for matrix operations when enabled via Makefile */
#ifdef USE_BLAS
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#endif

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

/* Minimum matrix size to use Metal GPU (smaller matrices are faster on CPU) */
#define MIN_GPU_ELEMENTS (512 * 512)

#ifdef USE_CUDA
/* Default minimum GEMM op-count (M*K*N) to dispatch to CUDA.
 * Tunable with FLUX_CUDA_MIN_OPS env var.
 * Lowered after CUDA-attention batching benchmarks (2M was consistently faster
 * than 8M-20M on FLUX.2-klein 256x256 runs). */
#define MIN_CUDA_OPS_DEFAULT ((size_t)2 * 1024 * 1024)
#define CUDA_WEIGHT_CACHE_DEFAULT_MB 1024ULL
#define CUDA_WEIGHT_CACHE_MAX_ENTRIES 256
#define CUDA_WEIGHT_CACHE_MIN_BYTES ((size_t)1 * 1024 * 1024)

static cublasHandle_t g_cuda_handle = NULL;
static cudaStream_t g_cuda_stream = NULL;
static float *g_cuda_a = NULL;
static float *g_cuda_b = NULL;
static float *g_cuda_c = NULL;
static uint16_t *g_cuda_b_bf16 = NULL;
static size_t g_cuda_a_bytes = 0;
static size_t g_cuda_b_bytes = 0;
static size_t g_cuda_c_bytes = 0;
static size_t g_cuda_b_bf16_bytes = 0;
static int g_cuda_available = -1;
static int g_cuda_warned = 0;
static size_t g_cuda_min_ops = 0;
static int g_cuda_bf16_linear_ok = -1;
static int g_cuda_tf32_mode = -1;
static int g_cuda_weight_cache_configured = 0;
static size_t g_cuda_weight_cache_limit_bytes = 0;
static size_t g_cuda_weight_cache_used_bytes = 0;
static uint64_t g_cuda_weight_cache_tick = 1;

typedef struct {
    const void *host_ptr;
    void *device_ptr;
    size_t bytes;
    int k;
    int n;
    int transpose_b;
    int data_type;                 /* 0=f32, 1=bf16 */
    uint64_t fingerprint;
    uint64_t last_use;
} cuda_weight_cache_entry_t;

static cuda_weight_cache_entry_t g_cuda_weight_cache[CUDA_WEIGHT_CACHE_MAX_ENTRIES];

static void flux_cuda_warn_once(const char *msg) {
    if (!g_cuda_warned) {
        fprintf(stderr, "%s\n", msg);
        g_cuda_warned = 1;
    }
}

static size_t flux_cuda_get_min_ops(void) {
    if (g_cuda_min_ops != 0) {
        return g_cuda_min_ops;
    }

    g_cuda_min_ops = MIN_CUDA_OPS_DEFAULT;
    const char *env = getenv("FLUX_CUDA_MIN_OPS");
    if (!env || !*env) {
        return g_cuda_min_ops;
    }

    errno = 0;
    char *end = NULL;
    unsigned long long parsed = strtoull(env, &end, 10);
    if (errno == 0 && end && *end == '\0' && parsed > 0ULL) {
        g_cuda_min_ops = (size_t)parsed;
    } else {
        flux_cuda_warn_once("CUDA: invalid FLUX_CUDA_MIN_OPS value, using default threshold");
    }

    return g_cuda_min_ops;
}

static int flux_cuda_tf32_enabled(void) {
    if (g_cuda_tf32_mode != -1) {
        return g_cuda_tf32_mode;
    }
    g_cuda_tf32_mode = getenv("FLUX_CUDA_NO_TF32") ? 0 : 1;
    return g_cuda_tf32_mode;
}

static uint64_t flux_cuda_weight_fingerprint(const void *data, size_t bytes) {
    if (!data || bytes == 0) {
        return 0;
    }

    const uint8_t *p = (const uint8_t *)data;
    uint64_t h = 1469598103934665603ULL ^ (uint64_t)bytes;
    const int samples = 16;

    for (int i = 0; i < samples; i++) {
        size_t idx = (size_t)i * (bytes - 1) / (size_t)(samples - 1);
        h ^= (uint64_t)p[idx];
        h *= 1099511628211ULL;
    }
    return h;
}

static uint64_t flux_cuda_next_cache_tick(void) {
    g_cuda_weight_cache_tick++;
    if (g_cuda_weight_cache_tick == 0) {
        g_cuda_weight_cache_tick = 1;
    }
    return g_cuda_weight_cache_tick;
}

static void flux_cuda_weight_cache_clear_entry(int idx) {
    cuda_weight_cache_entry_t *e = &g_cuda_weight_cache[idx];
    if (e->device_ptr) {
        cudaFree(e->device_ptr);
    }
    if (g_cuda_weight_cache_used_bytes >= e->bytes) {
        g_cuda_weight_cache_used_bytes -= e->bytes;
    } else {
        g_cuda_weight_cache_used_bytes = 0;
    }
    memset(e, 0, sizeof(*e));
}

static void flux_cuda_weight_cache_clear_all(void) {
    for (int i = 0; i < CUDA_WEIGHT_CACHE_MAX_ENTRIES; i++) {
        flux_cuda_weight_cache_clear_entry(i);
    }
    g_cuda_weight_cache_used_bytes = 0;
}

static int flux_cuda_weight_cache_find_lru(void) {
    int lru_idx = -1;
    uint64_t oldest = UINT64_MAX;

    for (int i = 0; i < CUDA_WEIGHT_CACHE_MAX_ENTRIES; i++) {
        cuda_weight_cache_entry_t *e = &g_cuda_weight_cache[i];
        if (!e->device_ptr) continue;
        if (e->last_use < oldest) {
            oldest = e->last_use;
            lru_idx = i;
        }
    }

    return lru_idx;
}

static int flux_cuda_weight_cache_ensure_space(size_t need_bytes) {
    if (need_bytes > g_cuda_weight_cache_limit_bytes) {
        return 0;
    }

    while (g_cuda_weight_cache_used_bytes + need_bytes > g_cuda_weight_cache_limit_bytes) {
        int lru_idx = flux_cuda_weight_cache_find_lru();
        if (lru_idx < 0) {
            return 0;
        }
        flux_cuda_weight_cache_clear_entry(lru_idx);
    }

    return 1;
}

static int flux_cuda_weight_cache_find(const void *host_ptr,
                                       int k, int n, int transpose_b,
                                       int data_type, uint64_t fingerprint) {
    for (int i = 0; i < CUDA_WEIGHT_CACHE_MAX_ENTRIES; i++) {
        cuda_weight_cache_entry_t *e = &g_cuda_weight_cache[i];
        if (!e->device_ptr) continue;
        if (e->host_ptr == host_ptr &&
            e->k == k &&
            e->n == n &&
            e->transpose_b == transpose_b &&
            e->data_type == data_type) {
            if (e->fingerprint == fingerprint) {
                e->last_use = flux_cuda_next_cache_tick();
                return i;
            }
            /* Same address reused for different content. Evict stale entry. */
            flux_cuda_weight_cache_clear_entry(i);
            return -1;
        }
    }
    return -1;
}

static int flux_cuda_weight_cache_select_slot(void) {
    for (int i = 0; i < CUDA_WEIGHT_CACHE_MAX_ENTRIES; i++) {
        if (!g_cuda_weight_cache[i].device_ptr) {
            return i;
        }
    }

    int lru_idx = flux_cuda_weight_cache_find_lru();
    if (lru_idx >= 0) {
        flux_cuda_weight_cache_clear_entry(lru_idx);
    }
    return lru_idx;
}

static void *flux_cuda_weight_cache_insert(const void *host_ptr,
                                            int k, int n, int transpose_b, int data_type,
                                            uint64_t fingerprint,
                                            size_t bytes) {
    if (bytes > g_cuda_weight_cache_limit_bytes || g_cuda_weight_cache_limit_bytes == 0) {
        return NULL;
    }

    if (!flux_cuda_weight_cache_ensure_space(bytes)) {
        return NULL;
    }

    int slot = flux_cuda_weight_cache_select_slot();
    if (slot < 0) {
        return NULL;
    }

    void *device_ptr = NULL;
    if (cudaMalloc(&device_ptr, bytes) != cudaSuccess) {
        return NULL;
    }
    if (cudaMemcpy(device_ptr, host_ptr, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(device_ptr);
        return NULL;
    }

    cuda_weight_cache_entry_t *e = &g_cuda_weight_cache[slot];
    e->host_ptr = host_ptr;
    e->device_ptr = device_ptr;
    e->bytes = bytes;
    e->k = k;
    e->n = n;
    e->transpose_b = transpose_b;
    e->data_type = data_type;
    e->fingerprint = fingerprint;
    e->last_use = flux_cuda_next_cache_tick();
    g_cuda_weight_cache_used_bytes += bytes;
    return device_ptr;
}

static void flux_cuda_weight_cache_configure(void) {
    if (g_cuda_weight_cache_configured) {
        return;
    }
    g_cuda_weight_cache_configured = 1;

    size_t limit_bytes = (size_t)(CUDA_WEIGHT_CACHE_DEFAULT_MB * 1024ULL * 1024ULL);
    const char *env = getenv("FLUX_CUDA_WEIGHT_CACHE_MB");

    if (env && *env) {
        errno = 0;
        char *end = NULL;
        unsigned long long parsed = strtoull(env, &end, 10);
        if (errno == 0 && end && *end == '\0') {
            if (parsed == 0ULL) {
                g_cuda_weight_cache_limit_bytes = 0;
                return;
            }
            if (parsed > (unsigned long long)(SIZE_MAX / (1024ULL * 1024ULL))) {
                limit_bytes = SIZE_MAX;
            } else {
                limit_bytes = (size_t)parsed * 1024ULL * 1024ULL;
            }
        } else {
            flux_cuda_warn_once("CUDA: invalid FLUX_CUDA_WEIGHT_CACHE_MB value, using default cache size");
        }
    } else {
        size_t free_mem = 0, total_mem = 0;
        if (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess && free_mem > 0) {
            size_t auto_limit = free_mem / 4;   /* Keep cache under 25% of currently free VRAM. */
            size_t max_default = (size_t)(CUDA_WEIGHT_CACHE_DEFAULT_MB * 1024ULL * 1024ULL);
            if (auto_limit < CUDA_WEIGHT_CACHE_MIN_BYTES) auto_limit = CUDA_WEIGHT_CACHE_MIN_BYTES;
            if (auto_limit < max_default) limit_bytes = auto_limit;
        }
    }

    g_cuda_weight_cache_limit_bytes = limit_bytes;
}

static void flux_cuda_cleanup_impl(void) {
    flux_cuda_weight_cache_clear_all();

    if (g_cuda_a) cudaFree(g_cuda_a);
    if (g_cuda_b) cudaFree(g_cuda_b);
    if (g_cuda_c) cudaFree(g_cuda_c);
    if (g_cuda_b_bf16) cudaFree(g_cuda_b_bf16);
    g_cuda_a = g_cuda_b = g_cuda_c = NULL;
    g_cuda_b_bf16 = NULL;
    g_cuda_a_bytes = g_cuda_b_bytes = g_cuda_c_bytes = 0;
    g_cuda_b_bf16_bytes = 0;

    if (g_cuda_handle) {
        cublasDestroy(g_cuda_handle);
        g_cuda_handle = NULL;
    }
}

static int flux_cuda_ensure_initialized(void) {
    if (g_cuda_available != -1) {
        return g_cuda_available;
    }

    g_cuda_available = 0;

    int device_count = 0;
    cudaError_t cuda_err = cudaGetDeviceCount(&device_count);
    if (cuda_err != cudaSuccess || device_count <= 0) {
        return 0;
    }

    if (cublasCreate(&g_cuda_handle) != CUBLAS_STATUS_SUCCESS) {
        flux_cuda_warn_once("CUDA: failed to initialize cuBLAS, falling back to CPU/BLAS");
        flux_cuda_cleanup_impl();
        return 0;
    }

    if (cublasSetPointerMode(g_cuda_handle, CUBLAS_POINTER_MODE_HOST) != CUBLAS_STATUS_SUCCESS) {
        flux_cuda_warn_once("CUDA: failed to configure cuBLAS pointer mode, falling back to CPU/BLAS");
        flux_cuda_cleanup_impl();
        return 0;
    }
    if (cublasSetStream(g_cuda_handle, g_cuda_stream) != CUBLAS_STATUS_SUCCESS) {
        flux_cuda_warn_once("CUDA: failed to configure cuBLAS stream, falling back to CPU/BLAS");
        flux_cuda_cleanup_impl();
        return 0;
    }

    flux_cuda_weight_cache_configure();
    atexit(flux_cuda_cleanup_impl);
    g_cuda_available = 1;
    return 1;
}

static int flux_cuda_ensure_buffer(float **buffer, size_t *capacity_bytes, size_t needed_bytes) {
    if (*capacity_bytes >= needed_bytes) {
        return 1;
    }

    size_t new_capacity = needed_bytes;
    if (*capacity_bytes > 0) {
        new_capacity = *capacity_bytes;
        while (new_capacity < needed_bytes) {
            new_capacity *= 2;
        }
    }

    float *new_buffer = NULL;
    if (cudaMalloc((void **)&new_buffer, new_capacity) != cudaSuccess) {
        flux_cuda_warn_once("CUDA: device allocation failed, falling back to CPU/BLAS");
        return 0;
    }

    if (*buffer) {
        cudaFree(*buffer);
    }
    *buffer = new_buffer;
    *capacity_bytes = new_capacity;
    return 1;
}

int flux_cuda_linear_set_stream(void *stream_handle) {
#ifdef USE_CUDA
    g_cuda_stream = (cudaStream_t)stream_handle;
    if (!flux_cuda_ensure_initialized()) return 0;
    return cublasSetStream(g_cuda_handle, g_cuda_stream) == CUBLAS_STATUS_SUCCESS;
#else
    (void)stream_handle;
    return 0;
#endif
}

static int flux_cuda_ensure_buffer_bf16(uint16_t **buffer,
                                        size_t *capacity_bytes,
                                        size_t needed_bytes) {
    if (*capacity_bytes >= needed_bytes) {
        return 1;
    }

    size_t new_capacity = needed_bytes;
    if (*capacity_bytes > 0) {
        new_capacity = *capacity_bytes;
        while (new_capacity < needed_bytes) {
            new_capacity *= 2;
        }
    }

    uint16_t *new_buffer = NULL;
    if (cudaMalloc((void **)&new_buffer, new_capacity) != cudaSuccess) {
        flux_cuda_warn_once("CUDA: bf16 device allocation failed, falling back to CPU/BLAS");
        return 0;
    }

    if (*buffer) {
        cudaFree(*buffer);
    }
    *buffer = new_buffer;
    *capacity_bytes = new_capacity;
    return 1;
}

static int flux_cuda_probe_bf16_linear(void) {
    if (!flux_cuda_ensure_initialized()) {
        return 0;
    }

    float h_a[16];
    uint16_t h_b[16];
    float h_c[16];
    for (int i = 0; i < 16; i++) {
        float a = 0.01f * (float)(i + 1);
        h_a[i] = a;
        uint32_t bits;
        memcpy(&bits, &a, sizeof(bits));
        h_b[i] = (uint16_t)(bits >> 16);  /* f32 -> bf16 */
        h_c[i] = 0.0f;
    }

    float *d_a = NULL;
    uint16_t *d_b = NULL;
    float *d_c = NULL;
    size_t a_bytes = sizeof(h_a);
    size_t b_bytes = sizeof(h_b);
    size_t c_bytes = sizeof(h_c);

    if (cudaMalloc((void **)&d_a, a_bytes) != cudaSuccess ||
        cudaMalloc((void **)&d_b, b_bytes) != cudaSuccess ||
        cudaMalloc((void **)&d_c, c_bytes) != cudaSuccess) {
        if (d_a) cudaFree(d_a);
        if (d_b) cudaFree(d_b);
        if (d_c) cudaFree(d_c);
        return 0;
    }

    int ok = 1;
    if (cudaMemcpy(d_a, h_a, a_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_b, h_b, b_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        ok = 0;
        goto done;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
#ifdef CUBLAS_COMPUTE_32F_FAST_16BF
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F_FAST_16BF;
#else
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
#endif
    cublasStatus_t st = cublasGemmEx(g_cuda_handle,
                                     CUBLAS_OP_T, CUBLAS_OP_N,
                                     4, 4, 4,
                                     &alpha,
                                     d_b, CUDA_R_16BF, 4,
                                     d_a, CUDA_R_32F, 4,
                                     &beta,
                                     d_c, CUDA_R_32F, 4,
                                     compute_type, CUBLAS_GEMM_DEFAULT);
    if (st != CUBLAS_STATUS_SUCCESS) {
        ok = 0;
        goto done;
    }

    if (cudaMemcpy(h_c, d_c, c_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        ok = 0;
    }

done:
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return ok;
}

/* Row-major SGEMM wrapper.
 * Computes C = A @ B (transpose_b=0) or C = A @ B^T (transpose_b=1). */
static int flux_cuda_sgemm_rowmajor(float *C, const float *A, const float *B,
                                    int M, int K, int N, int transpose_b,
                                    int cache_b) {
    if (!flux_cuda_ensure_initialized()) {
        return 0;
    }

    size_t a_bytes = (size_t)M * K * sizeof(float);
    size_t b_bytes = transpose_b
        ? (size_t)N * K * sizeof(float)
        : (size_t)K * N * sizeof(float);
    size_t c_bytes = (size_t)M * N * sizeof(float);

    if (!flux_cuda_ensure_buffer(&g_cuda_a, &g_cuda_a_bytes, a_bytes) ||
        !flux_cuda_ensure_buffer(&g_cuda_c, &g_cuda_c_bytes, c_bytes)) {
        return 0;
    }

    cudaError_t cuda_err = cudaMemcpy(g_cuda_a, A, a_bytes, cudaMemcpyHostToDevice);
    if (cuda_err != cudaSuccess) {
        flux_cuda_warn_once("CUDA: host-to-device copy failed, falling back to CPU/BLAS");
        return 0;
    }

    const float *d_b = NULL;
    if (cache_b && g_cuda_weight_cache_limit_bytes > 0 && b_bytes >= CUDA_WEIGHT_CACHE_MIN_BYTES) {
        uint64_t fingerprint = flux_cuda_weight_fingerprint(B, b_bytes);
        int cache_idx = flux_cuda_weight_cache_find(B, K, N, transpose_b, 0, fingerprint);
        if (cache_idx >= 0) {
            d_b = (const float *)g_cuda_weight_cache[cache_idx].device_ptr;
        } else {
            d_b = (const float *)flux_cuda_weight_cache_insert(B, K, N, transpose_b,
                                                               0, fingerprint, b_bytes);
        }
    }

    if (!d_b) {
        if (!flux_cuda_ensure_buffer(&g_cuda_b, &g_cuda_b_bytes, b_bytes)) {
            return 0;
        }
        cuda_err = cudaMemcpy(g_cuda_b, B, b_bytes, cudaMemcpyHostToDevice);
        if (cuda_err != cudaSuccess) {
            flux_cuda_warn_once("CUDA: host-to-device copy failed, falling back to CPU/BLAS");
            return 0;
        }
        d_b = g_cuda_b;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasStatus_t status;
    cublasOperation_t op_a = transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    int lda = transpose_b ? K : N;

    /* Prefer TF32 tensor-core GEMM on supported GPUs.
     * Disable with FLUX_CUDA_NO_TF32=1 for exact-fp32 fallback testing. */
    if (flux_cuda_tf32_enabled()) {
#ifdef CUBLAS_COMPUTE_32F_FAST_TF32
        cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;
#else
        cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
#endif
        status = cublasGemmEx(g_cuda_handle,
                              op_a, CUBLAS_OP_N,
                              N, M, K,
                              &alpha,
                              d_b, CUDA_R_32F, lda,
                              g_cuda_a, CUDA_R_32F, K,
                              &beta,
                              g_cuda_c, CUDA_R_32F, N,
                              compute_type, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    } else {
        status = CUBLAS_STATUS_NOT_SUPPORTED;
    }

    if (status != CUBLAS_STATUS_SUCCESS) {
        /* Row-major A[M,K] @ B[*,K] -> column-major mapping for cuBLAS */
        status = cublasSgemm(g_cuda_handle,
                             op_a, CUBLAS_OP_N,
                             N, M, K,
                             &alpha,
                             d_b, lda,
                             g_cuda_a, K,
                             &beta,
                             g_cuda_c, N);
    }

    if (status != CUBLAS_STATUS_SUCCESS) {
        flux_cuda_warn_once("CUDA: cuBLAS SGEMM failed, falling back to CPU/BLAS");
        return 0;
    }

    cuda_err = cudaMemcpy(C, g_cuda_c, c_bytes, cudaMemcpyDeviceToHost);
    if (cuda_err != cudaSuccess) {
        flux_cuda_warn_once("CUDA: device-to-host copy failed, falling back to CPU/BLAS");
        return 0;
    }

    return 1;
}

/* Row-major SGEMM wrapper with device input/output and host-side weight matrix.
 * d_C = d_A @ B (transpose_b=0) or d_A @ B^T (transpose_b=1)
 * d_A/d_C are device pointers, B is host pointer (cached on device when possible). */
static int flux_cuda_sgemm_rowmajor_device_weight(float *d_C,
                                                  const float *d_A,
                                                  const float *B,
                                                  int M, int K, int N,
                                                  int transpose_b,
                                                  int cache_b) {
    if (!flux_cuda_ensure_initialized()) {
        return 0;
    }
    if (!d_C || !d_A || !B || M <= 0 || K <= 0 || N <= 0) {
        return 0;
    }

    size_t b_bytes = transpose_b
        ? (size_t)N * K * sizeof(float)
        : (size_t)K * N * sizeof(float);

    const float *d_b = NULL;
    if (cache_b && g_cuda_weight_cache_limit_bytes > 0 && b_bytes >= CUDA_WEIGHT_CACHE_MIN_BYTES) {
        uint64_t fingerprint = flux_cuda_weight_fingerprint(B, b_bytes);
        int cache_idx = flux_cuda_weight_cache_find(B, K, N, transpose_b, 0, fingerprint);
        if (cache_idx >= 0) {
            d_b = (const float *)g_cuda_weight_cache[cache_idx].device_ptr;
        } else {
            d_b = (const float *)flux_cuda_weight_cache_insert(B, K, N, transpose_b,
                                                               0, fingerprint, b_bytes);
        }
    }

    if (!d_b) {
        if (!flux_cuda_ensure_buffer(&g_cuda_b, &g_cuda_b_bytes, b_bytes)) {
            return 0;
        }
        if (cudaMemcpy(g_cuda_b, B, b_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            flux_cuda_warn_once("CUDA: host-to-device copy failed, falling back to CPU/BLAS");
            return 0;
        }
        d_b = g_cuda_b;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasOperation_t op_a = transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    int lda = transpose_b ? K : N;

    cublasStatus_t status = CUBLAS_STATUS_NOT_SUPPORTED;
    if (flux_cuda_tf32_enabled()) {
#ifdef CUBLAS_COMPUTE_32F_FAST_TF32
        cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;
#else
        cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
#endif
        status = cublasGemmEx(g_cuda_handle,
                              op_a, CUBLAS_OP_N,
                              N, M, K,
                              &alpha,
                              d_b, CUDA_R_32F, lda,
                              d_A, CUDA_R_32F, K,
                              &beta,
                              d_C, CUDA_R_32F, N,
                              compute_type, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }

    if (status != CUBLAS_STATUS_SUCCESS) {
        status = cublasSgemm(g_cuda_handle,
                             op_a, CUBLAS_OP_N,
                             N, M, K,
                             &alpha,
                             d_b, lda,
                             d_A, K,
                             &beta,
                             d_C, N);
    }

    if (status != CUBLAS_STATUS_SUCCESS) {
        flux_cuda_warn_once("CUDA: cuBLAS SGEMM failed, falling back to CPU/BLAS");
        return 0;
    }

    return 1;
}

/* Row-major SGEMM wrapper with bf16 weights:
 * C = A @ B (transpose_b=0) or C = A @ B^T (transpose_b=1)
 * A/C are f32, B is bf16 (uint16 storage). */
static int flux_cuda_sgemm_rowmajor_bf16_weight(float *C,
                                                const float *A,
                                                const uint16_t *B_bf16,
                                                int M, int K, int N,
                                                int transpose_b,
                                                int cache_b) {
    if (!flux_cuda_ensure_initialized()) {
        return 0;
    }

    size_t a_bytes = (size_t)M * K * sizeof(float);
    size_t b_bytes = transpose_b
        ? (size_t)N * K * sizeof(uint16_t)
        : (size_t)K * N * sizeof(uint16_t);
    size_t c_bytes = (size_t)M * N * sizeof(float);

    if (!flux_cuda_ensure_buffer(&g_cuda_a, &g_cuda_a_bytes, a_bytes) ||
        !flux_cuda_ensure_buffer(&g_cuda_c, &g_cuda_c_bytes, c_bytes)) {
        return 0;
    }

    cudaError_t cuda_err = cudaMemcpy(g_cuda_a, A, a_bytes, cudaMemcpyHostToDevice);
    if (cuda_err != cudaSuccess) {
        flux_cuda_warn_once("CUDA: bf16 host-to-device copy failed, falling back to CPU/BLAS");
        return 0;
    }

    const uint16_t *d_b = NULL;
    if (cache_b && g_cuda_weight_cache_limit_bytes > 0 && b_bytes >= CUDA_WEIGHT_CACHE_MIN_BYTES) {
        uint64_t fingerprint = flux_cuda_weight_fingerprint(B_bf16, b_bytes);
        int cache_idx = flux_cuda_weight_cache_find(B_bf16, K, N, transpose_b, 1, fingerprint);
        if (cache_idx >= 0) {
            d_b = (const uint16_t *)g_cuda_weight_cache[cache_idx].device_ptr;
        } else {
            d_b = (const uint16_t *)flux_cuda_weight_cache_insert(B_bf16, K, N, transpose_b,
                                                                  1, fingerprint, b_bytes);
        }
    }

    if (!d_b) {
        if (!flux_cuda_ensure_buffer_bf16(&g_cuda_b_bf16, &g_cuda_b_bf16_bytes, b_bytes)) {
            return 0;
        }
        cuda_err = cudaMemcpy(g_cuda_b_bf16, B_bf16, b_bytes, cudaMemcpyHostToDevice);
        if (cuda_err != cudaSuccess) {
            flux_cuda_warn_once("CUDA: bf16 host-to-device copy failed, falling back to CPU/BLAS");
            return 0;
        }
        d_b = g_cuda_b_bf16;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
#ifdef CUBLAS_COMPUTE_32F_FAST_16BF
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F_FAST_16BF;
#else
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
#endif

    cublasStatus_t status;
    if (transpose_b) {
        status = cublasGemmEx(g_cuda_handle,
                              CUBLAS_OP_T, CUBLAS_OP_N,
                              N, M, K,
                              &alpha,
                              d_b, CUDA_R_16BF, K,
                              g_cuda_a, CUDA_R_32F, K,
                              &beta,
                              g_cuda_c, CUDA_R_32F, N,
                              compute_type, CUBLAS_GEMM_DEFAULT);
    } else {
        status = cublasGemmEx(g_cuda_handle,
                              CUBLAS_OP_N, CUBLAS_OP_N,
                              N, M, K,
                              &alpha,
                              d_b, CUDA_R_16BF, N,
                              g_cuda_a, CUDA_R_32F, K,
                              &beta,
                              g_cuda_c, CUDA_R_32F, N,
                              compute_type, CUBLAS_GEMM_DEFAULT);
    }

    if (status != CUBLAS_STATUS_SUCCESS) {
        flux_cuda_warn_once("CUDA: bf16 cuBLAS GEMM failed, falling back to CPU/BLAS");
        return 0;
    }

    cuda_err = cudaMemcpy(C, g_cuda_c, c_bytes, cudaMemcpyDeviceToHost);
    if (cuda_err != cudaSuccess) {
        flux_cuda_warn_once("CUDA: bf16 device-to-host copy failed, falling back to CPU/BLAS");
        return 0;
    }

    return 1;
}

int flux_cuda_bf16_linear_available(void) {
    if (g_cuda_bf16_linear_ok != -1) {
        return g_cuda_bf16_linear_ok;
    }

    g_cuda_bf16_linear_ok = flux_cuda_probe_bf16_linear() ? 1 : 0;
    return g_cuda_bf16_linear_ok;
}
#endif

#ifndef USE_CUDA
int flux_cuda_bf16_linear_available(void) {
    return 0;
}
#endif

int flux_cuda_linear_nobias_device(float *d_y, const float *d_x, const float *W,
                                   int seq_len, int in_dim, int out_dim) {
#ifdef USE_CUDA
    return flux_cuda_sgemm_rowmajor_device_weight(d_y, d_x, W,
                                                  seq_len, in_dim, out_dim,
                                                  1, 1);
#else
    (void)d_y; (void)d_x; (void)W; (void)seq_len; (void)in_dim; (void)out_dim;
    return 0;
#endif
}

/* Progress callbacks - set by caller before inference */
flux_substep_callback_t flux_substep_callback = NULL;
flux_step_callback_t flux_step_callback = NULL;
flux_phase_callback_t flux_phase_callback = NULL;
flux_step_image_callback_t flux_step_image_callback = NULL;
void *flux_step_image_vae = NULL;
flux_text_progress_callback_t flux_text_progress_callback = NULL;
flux_vae_progress_callback_t flux_vae_progress_callback = NULL;
int flux_verbose = 0;

/* ========================================================================
 * Random Number Generator (xoshiro256**)
 * ======================================================================== */

static uint64_t rng_state[4] = {
    0x853c49e6748fea9bULL,
    0xda3e39cb94b95bdbULL,
    0x647c4677a2884327ULL,
    0xc6e7918d2e2969f5ULL
};

static inline uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static uint64_t xoshiro256ss(void) {
    const uint64_t result = rotl(rng_state[1] * 5, 7) * 9;
    const uint64_t t = rng_state[1] << 17;
    rng_state[2] ^= rng_state[0];
    rng_state[3] ^= rng_state[1];
    rng_state[1] ^= rng_state[2];
    rng_state[0] ^= rng_state[3];
    rng_state[2] ^= t;
    rng_state[3] = rotl(rng_state[3], 45);
    return result;
}

void flux_rng_seed(uint64_t seed) {
    /* SplitMix64 to initialize state from seed */
    for (int i = 0; i < 4; i++) {
        seed += 0x9e3779b97f4a7c15ULL;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        rng_state[i] = z ^ (z >> 31);
    }
}

float flux_random_uniform(void) {
    return (xoshiro256ss() >> 11) * (1.0 / 9007199254740992.0);
}

float flux_random_normal(void) {
    /* Box-Muller transform */
    float u1 = flux_random_uniform();
    float u2 = flux_random_uniform();
    /* Avoid log(0) */
    while (u1 == 0.0f) u1 = flux_random_uniform();
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265358979323846f * u2);
}

void flux_randn(float *out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = flux_random_normal();
    }
}

void flux_rand(float *out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = flux_random_uniform();
    }
}

/* ========================================================================
 * Basic Element-wise Operations
 * ======================================================================== */

void flux_add(float *out, const float *a, const float *b, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = a[i] + b[i];
    }
}

void flux_add_inplace(float *a, const float *b, int n) {
    for (int i = 0; i < n; i++) {
        a[i] += b[i];
    }
}

void flux_mul_inplace(float *a, const float *b, int n) {
    for (int i = 0; i < n; i++) {
        a[i] *= b[i];
    }
}

void flux_axpy(float *a, float scale, const float *b, int n) {
    for (int i = 0; i < n; i++) {
        a[i] += scale * b[i];
    }
}

/* ========================================================================
 * Matrix Operations
 * ======================================================================== */

void flux_matmul(float *C, const float *A, const float *B,
                 int M, int K, int N) {
    /* C[M,N] = A[M,K] @ B[K,N] */

#ifdef USE_METAL
    size_t matrix_elements = (size_t)M * N;
    if (flux_metal_available() && matrix_elements >= MIN_GPU_ELEMENTS) {
        flux_metal_sgemm(0, 0,  /* no transpose */
                         M, N, K,
                         1.0f,
                         A, K,
                         B, N,
                         0.0f,
                         C, N);
        return;
    }
#endif

#ifdef USE_CUDA
    size_t matrix_ops_cuda = (size_t)M * K * N;
    if (matrix_ops_cuda >= flux_cuda_get_min_ops() &&
        flux_cuda_sgemm_rowmajor(C, A, B, M, K, N, 0, 0)) {
        return;
    }
#endif

#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K,
                1.0f, A, K, B, N,
                0.0f, C, N);
#else
    /* Fallback: naive implementation */
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = sum;
        }
    }
#endif
}

void flux_matmul_t(float *C, const float *A, const float *B,
                   int M, int K, int N) {
    /* C[M,N] = A[M,K] @ B[N,K]^T */

#ifdef USE_METAL
    size_t matrix_elements = (size_t)M * N;
    if (flux_metal_available() && matrix_elements >= MIN_GPU_ELEMENTS) {
        flux_metal_sgemm(0, 1,  /* no transpose A, transpose B */
                         M, N, K,
                         1.0f,
                         A, K,
                         B, K,
                         0.0f,
                         C, N);
        return;
    }
#endif

#ifdef USE_CUDA
    size_t matrix_ops_cuda = (size_t)M * K * N;
    if (matrix_ops_cuda >= flux_cuda_get_min_ops() &&
        flux_cuda_sgemm_rowmajor(C, A, B, M, K, N, 1, 0)) {
        return;
    }
#endif

#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, N, K,
                1.0f, A, K, B, K,
                0.0f, C, N);
#else
    /* Fallback: naive implementation */
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[m * K + k] * B[n * K + k];
            }
            C[m * N + n] = sum;
        }
    }
#endif
}

void flux_linear(float *y, const float *x, const float *W, const float *b,
                 int seq_len, int in_dim, int out_dim) {
    /* y[seq, out] = x[seq, in] @ W[out, in]^T + b[out] */

#ifdef USE_METAL
    /* Use Metal GPU for large matrices */
    size_t matrix_elements = (size_t)seq_len * out_dim;
    if (flux_metal_available() && matrix_elements >= MIN_GPU_ELEMENTS) {
        /* Metal sgemm: C = alpha * A @ B^T
         * A[M, K] = x[seq_len, in_dim]
         * B[N, K] = W[out_dim, in_dim] (transposed)
         * C[M, N] = y[seq_len, out_dim]
         */
        flux_metal_sgemm(0, 1,  /* no transpose A, transpose B */
                         seq_len, out_dim, in_dim,
                         1.0f,
                         x, in_dim,
                         W, in_dim,
                         0.0f,
                         y, out_dim);

        /* Add bias if present */
        if (b != NULL) {
            for (int s = 0; s < seq_len; s++) {
                for (int o = 0; o < out_dim; o++) {
                    y[s * out_dim + o] += b[o];
                }
            }
        }
        return;
    }
#endif

#ifdef USE_CUDA
    size_t matrix_ops_cuda = (size_t)seq_len * in_dim * out_dim;
    if (matrix_ops_cuda >= flux_cuda_get_min_ops() &&
        flux_cuda_sgemm_rowmajor(y, x, W, seq_len, in_dim, out_dim, 1, 1)) {
        if (b != NULL) {
            for (int s = 0; s < seq_len; s++) {
                for (int o = 0; o < out_dim; o++) {
                    y[s * out_dim + o] += b[o];
                }
            }
        }
        return;
    }
#endif

#ifdef USE_BLAS
    /* Use BLAS sgemm: C = alpha * A @ B^T + beta * C
     * A[M, K] = x[seq_len, in_dim]
     * B[N, K] = W[out_dim, in_dim]
     * C[M, N] = y[seq_len, out_dim]
     */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                seq_len, out_dim, in_dim,
                1.0f, x, in_dim, W, in_dim,
                0.0f, y, out_dim);

    /* Add bias if present */
    if (b != NULL) {
        for (int s = 0; s < seq_len; s++) {
            for (int o = 0; o < out_dim; o++) {
                y[s * out_dim + o] += b[o];
            }
        }
    }
#else
    /* Fallback: naive implementation */
    for (int s = 0; s < seq_len; s++) {
        const float *x_row = x + s * in_dim;
        float *y_row = y + s * out_dim;
        for (int o = 0; o < out_dim; o++) {
            const float *w_row = W + o * in_dim;
            float sum = (b != NULL) ? b[o] : 0.0f;
            for (int i = 0; i < in_dim; i++) {
                sum += x_row[i] * w_row[i];
            }
            y_row[o] = sum;
        }
    }
#endif
}

void flux_linear_nobias(float *y, const float *x, const float *W,
                        int seq_len, int in_dim, int out_dim) {
    flux_linear(y, x, W, NULL, seq_len, in_dim, out_dim);
}

void flux_linear_nobias_bf16(float *y, const float *x, const uint16_t *W_bf16,
                             int seq_len, int in_dim, int out_dim) {
    /* y[seq, out] = x[seq, in] @ W[out, in]^T */

#ifdef USE_METAL
    /* Use Metal GPU for bf16 matmul - provides 2x memory bandwidth */
    size_t matrix_elements = (size_t)seq_len * out_dim;
    if (flux_metal_available() && matrix_elements >= MIN_GPU_ELEMENTS) {
        /* Metal bf16 sgemm: C = alpha * A @ B^T
         * A[M, K] = x[seq_len, in_dim] (f32)
         * B[N, K] = W[out_dim, in_dim] (bf16, transposed)
         * C[M, N] = y[seq_len, out_dim] (f32)
         */
        flux_metal_sgemm_bf16(0, 1,  /* no transpose A, transpose B */
                              seq_len, out_dim, in_dim,
                              1.0f,
                              x, in_dim,
                              W_bf16, in_dim,
                              0.0f,
                              y, out_dim);
        return;
    }
#endif

#ifdef USE_CUDA
    size_t matrix_ops_cuda = (size_t)seq_len * in_dim * out_dim;
    if (matrix_ops_cuda >= flux_cuda_get_min_ops() &&
        flux_cuda_sgemm_rowmajor_bf16_weight(y, x, W_bf16,
                                             seq_len, in_dim, out_dim,
                                             1, 1)) {
        return;
    }
#endif

    /* Fallback: convert bf16 to f32 and use regular linear */
    float *W_f32 = (float *)malloc((size_t)out_dim * in_dim * sizeof(float));
    if (!W_f32) return;

    /* Convert bf16 to f32 */
    for (int i = 0; i < out_dim * in_dim; i++) {
        uint32_t f32_bits = ((uint32_t)W_bf16[i]) << 16;
        memcpy(&W_f32[i], &f32_bits, sizeof(float));
    }

    flux_linear_nobias(y, x, W_f32, seq_len, in_dim, out_dim);
    free(W_f32);
}

/* ========================================================================
 * GPU Batch Operations
 * ======================================================================== */

void flux_gpu_begin_batch(void) {
#ifdef USE_METAL
    flux_metal_begin_batch();
#endif
}

void flux_gpu_end_batch(void) {
#ifdef USE_METAL
    flux_metal_end_batch();
#endif
}

/* ========================================================================
 * Convolution Operations
 * ======================================================================== */

void flux_conv2d(float *out, const float *in, const float *weight, const float *bias,
                 int batch, int in_ch, int out_ch, int H, int W,
                 int kH, int kW, int stride, int padding) {
    int outH = (H + 2 * padding - kH) / stride + 1;
    int outW = (W + 2 * padding - kW) / stride + 1;

#ifdef USE_BLAS
    /* im2col + BLAS optimization with tiling for large convolutions */
    size_t col_size = (size_t)in_ch * kH * kW * outH * outW;
    size_t max_col_size = (size_t)256 * 1024 * 1024;  /* 1GB limit */

    /* For large convolutions, process in row tiles */
    int tile_rows = outH;
    if (col_size > max_col_size) {
        /* Calculate how many rows we can process at once */
        size_t row_size = (size_t)in_ch * kH * kW * outW;
        tile_rows = (int)(max_col_size / row_size);
        if (tile_rows < 1) tile_rows = 1;
    }

    size_t tile_col_size = (size_t)in_ch * kH * kW * tile_rows * outW;
    float *col = malloc(tile_col_size * sizeof(float));
    if (!col) {
        goto naive_fallback;
    }

    for (int b = 0; b < batch; b++) {
        const float *in_b = in + b * in_ch * H * W;
        float *out_b = out + b * out_ch * outH * outW;

        /* Process in tiles of rows */
        for (int tile_start = 0; tile_start < outH; tile_start += tile_rows) {
            int tile_end = tile_start + tile_rows;
            if (tile_end > outH) tile_end = outH;
            int tile_h = tile_end - tile_start;
            int tile_pixels = tile_h * outW;

            /* im2col for this tile: col[in_ch*kH*kW, tile_pixels] */
            int col_row = 0;
            for (int ic = 0; ic < in_ch; ic++) {
                for (int kh = 0; kh < kH; kh++) {
                    for (int kw = 0; kw < kW; kw++) {
                        for (int oh = tile_start; oh < tile_end; oh++) {
                            for (int ow = 0; ow < outW; ow++) {
                                int ih = oh * stride - padding + kh;
                                int iw = ow * stride - padding + kw;
                                int col_idx = col_row * tile_pixels + (oh - tile_start) * outW + ow;
                                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                    col[col_idx] = in_b[ic * H * W + ih * W + iw];
                                } else {
                                    col[col_idx] = 0.0f;
                                }
                            }
                        }
                        col_row++;
                    }
                }
            }

            /* BLAS sgemm: tmp[out_ch, tile_pixels] = weight[out_ch, K] @ col[K, tile_pixels]
             * where K = in_ch * kH * kW */
            int K = in_ch * kH * kW;

            /* Allocate temporary contiguous buffer for tile output */
            float *tmp = malloc((size_t)out_ch * tile_pixels * sizeof(float));
            if (!tmp) {
                free(col);
                goto naive_fallback;
            }

            /* sgemm: tmp[out_ch, tile_pixels] = weight[out_ch, K] @ col[K, tile_pixels] */
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        out_ch, tile_pixels, K,
                        1.0f, weight, K,
                        col, tile_pixels,
                        0.0f, tmp, tile_pixels);

            /* Scatter tile output to correct positions in out_b */
            for (int oc = 0; oc < out_ch; oc++) {
                float *out_tile = out_b + oc * outH * outW + tile_start * outW;
                float *tmp_row = tmp + oc * tile_pixels;
                memcpy(out_tile, tmp_row, tile_pixels * sizeof(float));
            }

            free(tmp);
        }

        /* Add bias */
        if (bias != NULL) {
            for (int oc = 0; oc < out_ch; oc++) {
                float b_val = bias[oc];
                float *out_ch_ptr = out_b + oc * outH * outW;
                for (int i = 0; i < outH * outW; i++) {
                    out_ch_ptr[i] += b_val;
                }
            }
        }
    }

    free(col);
    return;

naive_fallback:
#endif
    /* Naive implementation (fallback) */
    for (int b = 0; b < batch; b++) {
        for (int oc = 0; oc < out_ch; oc++) {
            for (int oh = 0; oh < outH; oh++) {
                for (int ow = 0; ow < outW; ow++) {
                    float sum = (bias != NULL) ? bias[oc] : 0.0f;

                    for (int ic = 0; ic < in_ch; ic++) {
                        for (int kh = 0; kh < kH; kh++) {
                            for (int kw = 0; kw < kW; kw++) {
                                int ih = oh * stride - padding + kh;
                                int iw = ow * stride - padding + kw;

                                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                    int in_idx = b * in_ch * H * W + ic * H * W + ih * W + iw;
                                    int w_idx = oc * in_ch * kH * kW + ic * kH * kW + kh * kW + kw;
                                    sum += in[in_idx] * weight[w_idx];
                                }
                            }
                        }
                    }

                    int out_idx = b * out_ch * outH * outW + oc * outH * outW + oh * outW + ow;
                    out[out_idx] = sum;
                }
            }
        }
    }
}

/* ========================================================================
 * Normalization
 * ======================================================================== */

void flux_rms_norm(float *out, const float *x, const float *weight,
                   int seq_len, int hidden, float eps) {
#ifdef USE_METAL
    /* Use GPU for RMSNorm only for very large tensors
     * The CPU-GPU sync overhead usually outweighs benefits for smaller ops */
    size_t elements = (size_t)seq_len * hidden;
    if (flux_metal_shaders_available() && elements >= 1024 * 1024) {
        flux_metal_rms_norm(out, x, weight, seq_len, hidden, eps);
        return;
    }
#endif

    for (int s = 0; s < seq_len; s++) {
        const float *x_row = x + s * hidden;
        float *out_row = out + s * hidden;

        /* Compute RMS */
        float sum_sq = 0.0f;
        for (int i = 0; i < hidden; i++) {
            sum_sq += x_row[i] * x_row[i];
        }
        float rms = sqrtf(sum_sq / hidden + eps);
        float rms_inv = 1.0f / rms;

        /* Normalize and scale */
        for (int i = 0; i < hidden; i++) {
            out_row[i] = x_row[i] * rms_inv * weight[i];
        }
    }
}

void flux_group_norm(float *out, const float *x, const float *gamma, const float *beta,
                     int batch, int channels, int H, int W, int num_groups, float eps) {
    int channels_per_group = channels / num_groups;
    int spatial = H * W;

    for (int b = 0; b < batch; b++) {
        for (int g = 0; g < num_groups; g++) {
            int c_start = g * channels_per_group;
            int c_end = c_start + channels_per_group;

            float mean = 0.0f;
            int count = 0;
            for (int c = c_start; c < c_end; c++) {
                for (int i = 0; i < spatial; i++) {
                    int idx = b * channels * spatial + c * spatial + i;
                    mean += x[idx];
                    count++;
                }
            }
            mean /= count;

            float var = 0.0f;
            for (int c = c_start; c < c_end; c++) {
                for (int i = 0; i < spatial; i++) {
                    int idx = b * channels * spatial + c * spatial + i;
                    float diff = x[idx] - mean;
                    var += diff * diff;
                }
            }
            var /= count;

            float std_inv = 1.0f / sqrtf(var + eps);

            for (int c = c_start; c < c_end; c++) {
                for (int i = 0; i < spatial; i++) {
                    int idx = b * channels * spatial + c * spatial + i;
                    float norm = (x[idx] - mean) * std_inv;
                    out[idx] = gamma[c] * norm + beta[c];
                }
            }
        }
    }
}

void flux_batch_norm(float *out, const float *x,
                     const float *running_mean, const float *running_var,
                     const float *gamma, const float *beta,
                     int batch, int channels, int H, int W, float eps) {
    int spatial = H * W;

    for (int c = 0; c < channels; c++) {
        float mean = running_mean[c];
        float var = running_var[c];
        float std_inv = 1.0f / sqrtf(var + eps);
        float g = (gamma != NULL) ? gamma[c] : 1.0f;
        float b_val = (beta != NULL) ? beta[c] : 0.0f;

        for (int n = 0; n < batch; n++) {
            for (int i = 0; i < spatial; i++) {
                int idx = n * channels * spatial + c * spatial + i;
                out[idx] = g * (x[idx] - mean) * std_inv + b_val;
            }
        }
    }
}

/* ========================================================================
 * Activation Functions
 * ======================================================================== */

void flux_silu(float *x, int n) {
#ifdef USE_METAL
    /* Use GPU for very large arrays (overhead not worth it for small ones) */
    if (flux_metal_shaders_available() && n >= 4 * 1024 * 1024) {
        flux_metal_silu(x, n);
        return;
    }
#endif

    for (int i = 0; i < n; i++) {
        float val = x[i];
        x[i] = val / (1.0f + expf(-val));
    }
}

/* Fused SiLU(gate) * up in a single pass. */
void flux_silu_mul(float *gate, const float *up, int n) {
#ifdef USE_METAL
    if (flux_metal_shaders_available() && n >= 4 * 1024 * 1024) {
        flux_metal_silu_mul(gate, up, n);
        return;
    }
#endif
    for (int i = 0; i < n; i++) {
        float v = gate[i];
        gate[i] = (v / (1.0f + expf(-v))) * up[i];
    }
}

void flux_softmax(float *x, int rows, int cols) {
#ifdef USE_METAL
    /* Use GPU only for very large softmax operations
     * Sync overhead usually dominates for smaller ops */
    if (flux_metal_shaders_available() && (size_t)rows * cols >= 4 * 1024 * 1024) {
        flux_metal_softmax(x, rows, cols);
        return;
    }
#endif

    for (int r = 0; r < rows; r++) {
        float *row = x + r * cols;

        /* Find max for numerical stability */
        float max_val = row[0];
        for (int c = 1; c < cols; c++) {
            if (row[c] > max_val) max_val = row[c];
        }

        /* Compute exp and sum */
        float sum = 0.0f;
        for (int c = 0; c < cols; c++) {
            row[c] = expf(row[c] - max_val);
            sum += row[c];
        }

        /* Normalize */
        float inv_sum = 1.0f / sum;
        for (int c = 0; c < cols; c++) {
            row[c] *= inv_sum;
        }
    }
}

/* ========================================================================
 * Attention Operations
 * ======================================================================== */

void flux_attention(float *out, const float *Q, const float *K, const float *V,
                    int batch, int heads, int seq_q, int seq_k, int head_dim,
                    float scale) {
    /* Allocate attention scores */
    float *scores = (float *)malloc(seq_q * seq_k * sizeof(float));

    for (int b = 0; b < batch; b++) {
        for (int h = 0; h < heads; h++) {
            const float *q = Q + (b * heads + h) * seq_q * head_dim;
            const float *k = K + (b * heads + h) * seq_k * head_dim;
            const float *v = V + (b * heads + h) * seq_k * head_dim;
            float *o = out + (b * heads + h) * seq_q * head_dim;

            /* scores = Q @ K^T * scale */
            for (int i = 0; i < seq_q; i++) {
                for (int j = 0; j < seq_k; j++) {
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; d++) {
                        dot += q[i * head_dim + d] * k[j * head_dim + d];
                    }
                    scores[i * seq_k + j] = dot * scale;
                }
            }

            /* softmax */
            flux_softmax(scores, seq_q, seq_k);

            /* out = scores @ V */
            for (int i = 0; i < seq_q; i++) {
                for (int d = 0; d < head_dim; d++) {
                    float sum = 0.0f;
                    for (int j = 0; j < seq_k; j++) {
                        sum += scores[i * seq_k + j] * v[j * head_dim + d];
                    }
                    o[i * head_dim + d] = sum;
                }
            }
        }
    }

    free(scores);
}

/* ========================================================================
 * Flash Attention - Memory-Efficient Tiled Attention
 *
 * Uses online softmax algorithm to compute attention without materializing
 * the full [seq_q, seq_k] attention matrix. Reduces memory from O(n²) to O(n).
 *
 * Algorithm (for each query position):
 * 1. Initialize: max_score = -inf, sum = 0, output = 0
 * 2. For each key/value block:
 *    - Compute local scores = Q @ K^T * scale
 *    - Update running max and sum with correction factors
 *    - Accumulate weighted values into output
 * 3. Normalize: output /= sum
 *
 * Reference: "FlashAttention: Fast and Memory-Efficient Exact Attention"
 * ======================================================================== */

/*
 * Flash attention for a single head.
 * Q: [seq_q, head_dim], K: [seq_k, head_dim], V: [seq_k, head_dim]
 * out: [seq_q, head_dim]
 * Uses O(head_dim) working memory per query instead of O(seq_k).
 */
static void flash_attention_head(float *out,
                                  const float *Q, const float *K, const float *V,
                                  int seq_q, int seq_k, int head_dim, float scale) {
    /* Process each query position independently */
    for (int i = 0; i < seq_q; i++) {
        const float *q_row = Q + i * head_dim;
        float *o_row = out + i * head_dim;

        /* Running statistics for online softmax */
        float max_score = -1e30f;  /* Large negative value (avoid -INFINITY with -ffast-math) */
        float sum_exp = 0.0f;

        /* Initialize output to zero */
        for (int d = 0; d < head_dim; d++) {
            o_row[d] = 0.0f;
        }

        /* Iterate over all key/value positions */
        for (int j = 0; j < seq_k; j++) {
            const float *k_row = K + j * head_dim;
            const float *v_row = V + j * head_dim;

            /* Compute attention score: Q[i] · K[j] * scale */
            float score = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                score += q_row[d] * k_row[d];
            }
            score *= scale;

            /* Online softmax update */
            if (score > max_score) {
                /* New maximum found - rescale previous accumulations */
                float correction = expf(max_score - score);
                sum_exp = sum_exp * correction + 1.0f;
                for (int d = 0; d < head_dim; d++) {
                    o_row[d] = o_row[d] * correction + v_row[d];
                }
                max_score = score;
            } else {
                /* Score is less than current max */
                float weight = expf(score - max_score);
                sum_exp += weight;
                for (int d = 0; d < head_dim; d++) {
                    o_row[d] += weight * v_row[d];
                }
            }
        }

        /* Normalize by sum */
        float inv_sum = 1.0f / sum_exp;
        for (int d = 0; d < head_dim; d++) {
            o_row[d] *= inv_sum;
        }
    }
}

/*
 * Flash attention with BLAS-optimized tiling.
 * Processes queries in tiles for better cache utilization.
 * Uses BLAS for tile-level matrix operations when available.
 *
 * Q: [seq_q, head_dim], K: [seq_k, head_dim], V: [seq_k, head_dim]
 * out: [seq_q, head_dim]
 * tile_scores: scratch buffer of size [q_tile_size, k_tile_size]
 */
static void flash_attention_head_tiled(float *out,
                                        const float *Q, const float *K, const float *V,
                                        int seq_q, int seq_k, int head_dim, float scale,
                                        float *tile_scores, int q_tile_size, int k_tile_size) {
    /* Per-query running statistics: max_score[seq_q], sum_exp[seq_q] */
    float *max_scores = (float *)malloc(seq_q * sizeof(float));
    float *sum_exps = (float *)malloc(seq_q * sizeof(float));

    /* Initialize */
    for (int i = 0; i < seq_q; i++) {
        max_scores[i] = -1e30f;  /* Large negative value (avoid -INFINITY with -ffast-math) */
        sum_exps[i] = 0.0f;
    }
    memset(out, 0, seq_q * head_dim * sizeof(float));

    /* Process in tiles over K/V dimension */
    for (int k_start = 0; k_start < seq_k; k_start += k_tile_size) {
        int k_end = (k_start + k_tile_size < seq_k) ? k_start + k_tile_size : seq_k;
        int k_len = k_end - k_start;

        /* Process in tiles over Q dimension */
        for (int q_start = 0; q_start < seq_q; q_start += q_tile_size) {
            int q_end = (q_start + q_tile_size < seq_q) ? q_start + q_tile_size : seq_q;
            int q_len = q_end - q_start;

            const float *Q_tile = Q + q_start * head_dim;
            const float *K_tile = K + k_start * head_dim;
            const float *V_tile = V + k_start * head_dim;
            float *out_tile = out + q_start * head_dim;

            /* Compute tile scores: Q_tile @ K_tile^T * scale */
#ifdef USE_BLAS
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        q_len, k_len, head_dim,
                        scale, Q_tile, head_dim, K_tile, head_dim,
                        0.0f, tile_scores, k_tile_size);
#else
            for (int qi = 0; qi < q_len; qi++) {
                for (int ki = 0; ki < k_len; ki++) {
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; d++) {
                        dot += Q_tile[qi * head_dim + d] * K_tile[ki * head_dim + d];
                    }
                    tile_scores[qi * k_tile_size + ki] = dot * scale;
                }
            }
#endif

            /* Online softmax update for this tile */
            for (int qi = 0; qi < q_len; qi++) {
                int i = q_start + qi;
                float *score_row = tile_scores + qi * k_tile_size;
                float *o_row = out_tile + qi * head_dim;

                /* Find max in this tile */
                float tile_max = score_row[0];
                for (int ki = 1; ki < k_len; ki++) {
                    if (score_row[ki] > tile_max) tile_max = score_row[ki];
                }

                /* Compute correction factors */
                float old_max = max_scores[i];
                float new_max = (tile_max > old_max) ? tile_max : old_max;

                /* Rescale old accumulations if needed */
                if (old_max > -1e29f) {  /* Check if we have prior accumulations */
                    float correction = expf(old_max - new_max);
                    sum_exps[i] *= correction;
                    for (int d = 0; d < head_dim; d++) {
                        o_row[d] *= correction;
                    }
                }

                /* Accumulate this tile's contribution */
                for (int ki = 0; ki < k_len; ki++) {
                    float weight = expf(score_row[ki] - new_max);
                    sum_exps[i] += weight;
                    const float *v_row = V_tile + ki * head_dim;
                    for (int d = 0; d < head_dim; d++) {
                        o_row[d] += weight * v_row[d];
                    }
                }

                max_scores[i] = new_max;
            }
        }
    }

    /* Final normalization */
    for (int i = 0; i < seq_q; i++) {
        float inv_sum = 1.0f / sum_exps[i];
        float *o_row = out + i * head_dim;
        for (int d = 0; d < head_dim; d++) {
            o_row[d] *= inv_sum;
        }
    }

    free(max_scores);
    free(sum_exps);
}

/*
 * Flash attention for multi-head attention.
 * Works on [seq, heads*head_dim] layout (same as transformer tensors).
 *
 * Q: [seq_q, heads * head_dim]
 * K: [seq_k, heads * head_dim]
 * V: [seq_k, heads * head_dim]
 * out: [seq_q, heads * head_dim]
 *
 * Memory usage: O(seq_q + tile_size²) instead of O(seq_q * seq_k)
 */
void flux_flash_attention(float *out, const float *Q, const float *K, const float *V,
                          int seq_q, int seq_k, int heads, int head_dim, float scale) {
    /* Tile sizes for cache efficiency */
    int q_tile_size = 32;  /* Process 32 queries at a time */
    int k_tile_size = 64;  /* Process 64 keys at a time */

    /* Allocate tile scratch buffer */
    float *tile_scores = (float *)malloc(q_tile_size * k_tile_size * sizeof(float));

    /* Process each head */
    for (int h = 0; h < heads; h++) {
        const float *Q_head = Q + h * head_dim;
        const float *K_head = K + h * head_dim;
        const float *V_head = V + h * head_dim;
        float *out_head = out + h * head_dim;

        /* Stride between consecutive positions for this head */
        int hidden = heads * head_dim;

        /* For small sequences, use simple non-tiled version */
        if (seq_q <= 64 && seq_k <= 128) {
            /* Extract head data into contiguous buffers */
            float *Q_contig = (float *)malloc(seq_q * head_dim * sizeof(float));
            float *K_contig = (float *)malloc(seq_k * head_dim * sizeof(float));
            float *V_contig = (float *)malloc(seq_k * head_dim * sizeof(float));
            float *out_contig = (float *)malloc(seq_q * head_dim * sizeof(float));

            for (int i = 0; i < seq_q; i++) {
                for (int d = 0; d < head_dim; d++) {
                    Q_contig[i * head_dim + d] = Q_head[i * hidden + d];
                }
            }
            for (int j = 0; j < seq_k; j++) {
                for (int d = 0; d < head_dim; d++) {
                    K_contig[j * head_dim + d] = K_head[j * hidden + d];
                    V_contig[j * head_dim + d] = V_head[j * hidden + d];
                }
            }

            flash_attention_head(out_contig, Q_contig, K_contig, V_contig,
                                 seq_q, seq_k, head_dim, scale);

            /* Copy back with stride */
            for (int i = 0; i < seq_q; i++) {
                for (int d = 0; d < head_dim; d++) {
                    out_head[i * hidden + d] = out_contig[i * head_dim + d];
                }
            }

            free(Q_contig);
            free(K_contig);
            free(V_contig);
            free(out_contig);
        } else {
            /* For larger sequences, use tiled version with strided access */
            /* Extract head data into contiguous buffers for BLAS efficiency */
            float *Q_contig = (float *)malloc(seq_q * head_dim * sizeof(float));
            float *K_contig = (float *)malloc(seq_k * head_dim * sizeof(float));
            float *V_contig = (float *)malloc(seq_k * head_dim * sizeof(float));
            float *out_contig = (float *)malloc(seq_q * head_dim * sizeof(float));

            for (int i = 0; i < seq_q; i++) {
                for (int d = 0; d < head_dim; d++) {
                    Q_contig[i * head_dim + d] = Q_head[i * hidden + d];
                }
            }
            for (int j = 0; j < seq_k; j++) {
                for (int d = 0; d < head_dim; d++) {
                    K_contig[j * head_dim + d] = K_head[j * hidden + d];
                    V_contig[j * head_dim + d] = V_head[j * hidden + d];
                }
            }

            flash_attention_head_tiled(out_contig, Q_contig, K_contig, V_contig,
                                        seq_q, seq_k, head_dim, scale,
                                        tile_scores, q_tile_size, k_tile_size);

            /* Copy back with stride */
            for (int i = 0; i < seq_q; i++) {
                for (int d = 0; d < head_dim; d++) {
                    out_head[i * hidden + d] = out_contig[i * head_dim + d];
                }
            }

            free(Q_contig);
            free(K_contig);
            free(V_contig);
            free(out_contig);
        }
    }

    free(tile_scores);
}

void flux_apply_rope(float *x, const float *freqs,
                     int batch, int seq, int heads, int head_dim) {
    /* x: [batch, seq, heads, head_dim]
     * freqs: [seq, head_dim/2, 2] (cos, sin)
     * Apply rotary embedding to pairs of dimensions */

    int half_dim = head_dim / 2;

    for (int b = 0; b < batch; b++) {
        for (int s = 0; s < seq; s++) {
            for (int h = 0; h < heads; h++) {
                float *vec = x + ((b * seq + s) * heads + h) * head_dim;

                for (int d = 0; d < half_dim; d++) {
                    float cos_val = freqs[s * half_dim * 2 + d * 2];
                    float sin_val = freqs[s * half_dim * 2 + d * 2 + 1];

                    float x0 = vec[d];
                    float x1 = vec[d + half_dim];

                    vec[d] = x0 * cos_val - x1 * sin_val;
                    vec[d + half_dim] = x0 * sin_val + x1 * cos_val;
                }
            }
        }
    }
}

void flux_compute_rope_freqs(float *freqs, const int *pos, int seq, int dim, float theta) {
    int half_dim = dim / 2;

    for (int s = 0; s < seq; s++) {
        float p = (float)pos[s];
        for (int d = 0; d < half_dim; d++) {
            float freq = 1.0f / powf(theta, (float)(2 * d) / (float)dim);
            float angle = p * freq;
            freqs[s * half_dim * 2 + d * 2] = cosf(angle);
            freqs[s * half_dim * 2 + d * 2 + 1] = sinf(angle);
        }
    }
}

/* ========================================================================
 * Pooling and Reshape
 * ======================================================================== */

void flux_upsample_nearest(float *out, const float *in,
                           int batch, int channels, int H, int W,
                           int scale_h, int scale_w) {
    int outH = H * scale_h;
    int outW = W * scale_w;

    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channels; c++) {
            for (int oh = 0; oh < outH; oh++) {
                for (int ow = 0; ow < outW; ow++) {
                    int ih = oh / scale_h;
                    int iw = ow / scale_w;
                    int in_idx = b * channels * H * W + c * H * W + ih * W + iw;
                    int out_idx = b * channels * outH * outW + c * outH * outW + oh * outW + ow;
                    out[out_idx] = in[in_idx];
                }
            }
        }
    }
}

void flux_patchify(float *out, const float *in,
                   int batch, int channels, int H, int W, int patch_size) {
    /* [B, C, H, W] -> [B, C*p*p, H/p, W/p] */
    int p = patch_size;
    int outH = H / p;
    int outW = W / p;
    int out_ch = channels * p * p;

    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channels; c++) {
            for (int ph = 0; ph < outH; ph++) {
                for (int pw = 0; pw < outW; pw++) {
                    for (int pi = 0; pi < p; pi++) {
                        for (int pj = 0; pj < p; pj++) {
                            int ih = ph * p + pi;
                            int iw = pw * p + pj;
                            int in_idx = b * channels * H * W + c * H * W + ih * W + iw;

                            int out_c = c * p * p + pi * p + pj;
                            int out_idx = b * out_ch * outH * outW + out_c * outH * outW + ph * outW + pw;
                            out[out_idx] = in[in_idx];
                        }
                    }
                }
            }
        }
    }
}

void flux_unpatchify(float *out, const float *in,
                     int batch, int channels, int H, int W, int patch_size) {
    /* [B, C*p*p, H, W] -> [B, C, H*p, W*p] */
    int p = patch_size;
    int in_ch = channels * p * p;
    int outH = H * p;
    int outW = W * p;

    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channels; c++) {
            for (int ph = 0; ph < H; ph++) {
                for (int pw = 0; pw < W; pw++) {
                    for (int pi = 0; pi < p; pi++) {
                        for (int pj = 0; pj < p; pj++) {
                            int in_c = c * p * p + pi * p + pj;
                            int in_idx = b * in_ch * H * W + in_c * H * W + ph * W + pw;

                            int oh = ph * p + pi;
                            int ow = pw * p + pj;
                            int out_idx = b * channels * outH * outW + c * outH * outW + oh * outW + ow;
                            out[out_idx] = in[in_idx];
                        }
                    }
                }
            }
        }
    }
}

/* ========================================================================
 * Utility Functions
 * ======================================================================== */

void flux_copy(float *dst, const float *src, int n) {
    memcpy(dst, src, n * sizeof(float));
}
