/*
 * FLUX ROCm Compatibility Layer - Implementation
 *
 * CPU-pointer wrappers for ROCm operations.
 * These handle the GPU memory management automatically.
 */

#include "flux_rocm_compat.h"
#include "flux_rocm.h"
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <unordered_map>
#include <mutex>

/* External functions from flux_rocm.cpp */
extern hipStream_t get_stream();

/* ========================================================================
 * Activation Cache
 * 
 * For activations that we see repeatedly (same size), we can reuse buffers.
 * This avoids malloc/free overhead for every operation.
 * ======================================================================== */

struct activation_cache {
    void *d_A;
    void *d_C;
    size_t A_bytes;
    size_t C_bytes;
};

static activation_cache g_act_cache = {nullptr, nullptr, 0, 0};
static std::mutex g_act_mutex;

static void* get_activation_buffer(size_t bytes, void **cache_ptr, size_t *cache_size) {
    if (*cache_ptr && *cache_size >= bytes) {
        return *cache_ptr;
    }
    
    if (*cache_ptr) {
        hipFree(*cache_ptr);
    }
    
    void *ptr = nullptr;
    if (hipMalloc(&ptr, bytes) != hipSuccess) {
        fprintf(stderr, "ROCm compat: failed to allocate %zu bytes\n", bytes);
        return nullptr;
    }
    
    *cache_ptr = ptr;
    *cache_size = bytes;
    return ptr;
}

/* ========================================================================
 * Weight Cache (reuse from flux_rocm.cpp via extern)
 * ======================================================================== */

/* Forward declaration - implemented in flux_rocm.cpp */
extern void* get_cached_weight(const void *cpu_ptr, size_t bytes);

/* ========================================================================
 * CPU-Pointer SGEMM
 *
 * C[M,N] = alpha * A[M,K] @ B[K,N] + beta * C[M,N]
 *
 * Strategy:
 * - A (activations): Upload, compute, may be reused
 * - B (weights): Use weight cache (uploaded once)
 * - C (output): Allocate, compute, download
 * ======================================================================== */

extern "C" void flux_rocm_sgemm_cpu(int transpose_a, int transpose_b,
                                     int M, int N, int K,
                                     float alpha,
                                     const float *A, int lda,
                                     const float *B, int ldb,
                                     float beta,
                                     float *C, int ldc) {
    if (!flux_rocm_available()) return;
    
    hipStream_t stream = get_stream();
    
    /* Calculate sizes */
    size_t A_rows = transpose_a ? K : M;
    size_t A_cols = transpose_a ? M : K;
    size_t B_rows = transpose_b ? N : K;
    size_t B_cols = transpose_b ? K : N;
    
    size_t A_bytes = A_rows * A_cols * sizeof(float);
    size_t B_bytes = B_rows * B_cols * sizeof(float);
    size_t C_bytes = M * N * sizeof(float);
    
    /* Get GPU buffers */
    std::lock_guard<std::mutex> lock(g_act_mutex);
    
    void *d_A = get_activation_buffer(A_bytes, &g_act_cache.d_A, &g_act_cache.A_bytes);
    void *d_C = get_activation_buffer(C_bytes, &g_act_cache.d_C, &g_act_cache.C_bytes);
    
    if (!d_A || !d_C) return;
    
    /* B is typically weights - use cache */
    void *d_B = get_cached_weight(B, B_bytes);
    if (!d_B) return;
    
    /* Upload A */
    hipMemcpyAsync(d_A, A, A_bytes, hipMemcpyHostToDevice, stream);
    
    /* Upload C if beta != 0 */
    if (beta != 0.0f) {
        hipMemcpyAsync(d_C, C, C_bytes, hipMemcpyHostToDevice, stream);
    }
    
    /* Run SGEMM on GPU */
    flux_rocm_sgemm(transpose_a, transpose_b,
                    M, N, K,
                    alpha,
                    (float*)d_A, lda,
                    (float*)d_B, ldb,
                    beta,
                    (float*)d_C, ldc);
    
    /* Download C */
    hipMemcpyAsync(C, d_C, C_bytes, hipMemcpyDeviceToHost, stream);
    hipStreamSynchronize(stream);
}

/* ========================================================================
 * CPU-Pointer RMSNorm
 * ======================================================================== */

extern "C" void flux_rocm_rms_norm_cpu(float *out, const float *x, const float *weight,
                                        int seq_len, int hidden, float eps) {
    if (!flux_rocm_available()) return;
    
    hipStream_t stream = get_stream();
    
    size_t data_bytes = seq_len * hidden * sizeof(float);
    size_t weight_bytes = hidden * sizeof(float);
    
    /* Allocate GPU memory */
    void *d_x = nullptr, *d_out = nullptr;
    hipMalloc(&d_x, data_bytes);
    hipMalloc(&d_out, data_bytes);
    
    /* Weight uses cache */
    void *d_weight = get_cached_weight(weight, weight_bytes);
    
    if (!d_x || !d_out || !d_weight) {
        if (d_x) hipFree(d_x);
        if (d_out) hipFree(d_out);
        return;
    }
    
    /* Upload */
    hipMemcpyAsync(d_x, x, data_bytes, hipMemcpyHostToDevice, stream);
    
    /* Compute */
    flux_rocm_rms_norm((float*)d_out, (float*)d_x, (float*)d_weight, seq_len, hidden, eps);
    
    /* Download */
    hipMemcpyAsync(out, d_out, data_bytes, hipMemcpyDeviceToHost, stream);
    hipStreamSynchronize(stream);
    
    hipFree(d_x);
    hipFree(d_out);
}

/* ========================================================================
 * CPU-Pointer SiLU
 * ======================================================================== */

extern "C" void flux_rocm_silu_cpu(float *x, int n) {
    if (!flux_rocm_available()) return;
    
    hipStream_t stream = get_stream();
    size_t bytes = n * sizeof(float);
    
    void *d_x = nullptr;
    hipMalloc(&d_x, bytes);
    if (!d_x) return;
    
    hipMemcpyAsync(d_x, x, bytes, hipMemcpyHostToDevice, stream);
    flux_rocm_silu((float*)d_x, n);
    hipMemcpyAsync(x, d_x, bytes, hipMemcpyDeviceToHost, stream);
    hipStreamSynchronize(stream);
    
    hipFree(d_x);
}

/* ========================================================================
 * CPU-Pointer Softmax
 * ======================================================================== */

extern "C" void flux_rocm_softmax_cpu(float *x, int rows, int cols) {
    if (!flux_rocm_available()) return;
    
    hipStream_t stream = get_stream();
    size_t bytes = rows * cols * sizeof(float);
    
    void *d_x = nullptr;
    hipMalloc(&d_x, bytes);
    if (!d_x) return;
    
    hipMemcpyAsync(d_x, x, bytes, hipMemcpyHostToDevice, stream);
    flux_rocm_softmax((float*)d_x, rows, cols);
    hipMemcpyAsync(x, d_x, bytes, hipMemcpyDeviceToHost, stream);
    hipStreamSynchronize(stream);
    
    hipFree(d_x);
}
