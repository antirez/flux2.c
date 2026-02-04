/*
 * Simple ROCm infrastructure test
 * 
 * Build: hipcc -O2 -o test_rocm test_rocm.cpp flux_rocm.cpp -lhipblas
 * Run: ./test_rocm
 */

#include "flux_rocm.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>

/* Helper to get float pointer from tensor */
static float* gpu_ptr(flux_rocm_tensor_t t) {
    return (float*)flux_rocm_tensor_gpu_ptr(t);
}

/* Simple SGEMM correctness test */
static int test_sgemm() {
    printf("Testing SGEMM... ");
    
    const int M = 64, N = 128, K = 256;
    
    /* Allocate CPU memory */
    float *A = (float*)malloc(M * K * sizeof(float));
    float *B = (float*)malloc(K * N * sizeof(float));
    float *C_gpu = (float*)malloc(M * N * sizeof(float));
    float *C_cpu = (float*)malloc(M * N * sizeof(float));
    
    /* Initialize with simple pattern */
    for (int i = 0; i < M * K; i++) A[i] = (float)(i % 7) / 7.0f;
    for (int i = 0; i < K * N; i++) B[i] = (float)(i % 11) / 11.0f;
    
    /* CPU reference: C = A @ B^T (to match our SGEMM call) */
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[j * K + k];  /* B^T means B[j,k] = B[j*K+k] */
            }
            C_cpu[i * N + j] = sum;
        }
    }
    
    /* GPU: Upload data */
    flux_rocm_tensor_t t_A = flux_rocm_tensor_create(A, M * K);
    flux_rocm_tensor_t t_B = flux_rocm_tensor_create(B, K * N);
    flux_rocm_tensor_t t_C = flux_rocm_tensor_alloc(M * N);
    
    if (!t_A || !t_B || !t_C) {
        printf("FAIL (tensor alloc)\n");
        return 0;
    }
    
    /* Run SGEMM: C = A @ B^T */
    flux_rocm_sgemm(0, 1,  /* no trans A, trans B */
                    M, N, K,
                    1.0f,
                    gpu_ptr(t_A), K,
                    gpu_ptr(t_B), K,
                    0.0f,
                    gpu_ptr(t_C), N);
    
    /* Read back */
    flux_rocm_tensor_read(t_C, C_gpu, M * N);
    
    /* Compare */
    float max_diff = 0;
    for (int i = 0; i < M * N; i++) {
        float diff = fabsf(C_gpu[i] - C_cpu[i]);
        if (diff > max_diff) max_diff = diff;
    }
    
    flux_rocm_tensor_free(t_A);
    flux_rocm_tensor_free(t_B);
    flux_rocm_tensor_free(t_C);
    free(A); free(B); free(C_gpu); free(C_cpu);
    
    if (max_diff < 1e-4) {
        printf("PASS (max diff: %.2e)\n", max_diff);
        return 1;
    } else {
        printf("FAIL (max diff: %.2e)\n", max_diff);
        return 0;
    }
}

/* SGEMM performance test */
static void test_sgemm_perf() {
    printf("\nSGEMM Performance Test:\n");
    
    /* Test sizes similar to transformer ops */
    struct { int M, N, K; const char *desc; } tests[] = {
        {768, 3072, 3072, "Linear (seq=768, hidden=3072)"},
        {768, 9216, 3072, "FFN up (seq=768, 3072->9216)"},
        {768, 3072, 9216, "FFN down (seq=768, 9216->3072)"},
        {256, 3072, 3072, "Linear (seq=256)"},
        {1024, 3072, 3072, "Linear (seq=1024)"},
    };
    
    for (auto &t : tests) {
        int M = t.M, N = t.N, K = t.K;
        
        /* Allocate */
        flux_rocm_tensor_t A = flux_rocm_tensor_alloc(M * K);
        flux_rocm_tensor_t B = flux_rocm_tensor_alloc(K * N);
        flux_rocm_tensor_t C = flux_rocm_tensor_alloc(M * N);
        
        if (!A || !B || !C) {
            printf("  %s: SKIP (alloc failed)\n", t.desc);
            continue;
        }
        
        /* Warmup */
        flux_rocm_sgemm(0, 1, M, N, K, 1.0f,
                        gpu_ptr(A), K,
                        gpu_ptr(B), K,
                        0.0f,
                        gpu_ptr(C), N);
        flux_rocm_sync();
        
        /* Benchmark */
        const int iters = 50;
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < iters; i++) {
            flux_rocm_sgemm(0, 1, M, N, K, 1.0f,
                            gpu_ptr(A), K,
                            gpu_ptr(B), K,
                            0.0f,
                            gpu_ptr(C), N);
        }
        flux_rocm_sync();
        
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count() / iters;
        
        /* Compute TFLOPS */
        double flops = 2.0 * M * N * K;  /* 2 ops per multiply-add */
        double tflops = (flops / 1e12) / (ms / 1000.0);
        
        printf("  %s: %.2f ms (%.1f TFLOPS)\n", t.desc, ms, tflops);
        
        flux_rocm_tensor_free(A);
        flux_rocm_tensor_free(B);
        flux_rocm_tensor_free(C);
    }
}

/* Linear layer test */
static int test_linear() {
    printf("Testing Linear layer... ");
    
    const int seq = 64, in_dim = 256, out_dim = 512;
    
    float *x = (float*)malloc(seq * in_dim * sizeof(float));
    float *W = (float*)malloc(out_dim * in_dim * sizeof(float));
    float *out_cpu = (float*)malloc(seq * out_dim * sizeof(float));
    float *out_gpu = (float*)malloc(seq * out_dim * sizeof(float));
    
    /* Initialize */
    for (int i = 0; i < seq * in_dim; i++) x[i] = (float)(i % 13) / 13.0f;
    for (int i = 0; i < out_dim * in_dim; i++) W[i] = (float)(i % 17) / 17.0f;
    
    /* CPU reference: out = x @ W^T */
    for (int s = 0; s < seq; s++) {
        for (int o = 0; o < out_dim; o++) {
            float sum = 0;
            for (int i = 0; i < in_dim; i++) {
                sum += x[s * in_dim + i] * W[o * in_dim + i];
            }
            out_cpu[s * out_dim + o] = sum;
        }
    }
    
    /* GPU */
    flux_rocm_tensor_t t_x = flux_rocm_tensor_create(x, seq * in_dim);
    flux_rocm_tensor_t t_out = flux_rocm_linear(t_x, W, seq, in_dim, out_dim);
    
    if (!t_out) {
        printf("FAIL (linear returned null)\n");
        flux_rocm_tensor_free(t_x);
        free(x); free(W); free(out_cpu); free(out_gpu);
        return 0;
    }
    
    flux_rocm_tensor_read(t_out, out_gpu, seq * out_dim);
    
    /* Compare */
    float max_diff = 0;
    for (int i = 0; i < seq * out_dim; i++) {
        float diff = fabsf(out_gpu[i] - out_cpu[i]);
        if (diff > max_diff) max_diff = diff;
    }
    
    flux_rocm_tensor_free(t_x);
    flux_rocm_tensor_free(t_out);
    free(x); free(W); free(out_cpu); free(out_gpu);
    
    if (max_diff < 1e-4) {
        printf("PASS (max diff: %.2e)\n", max_diff);
        return 1;
    } else {
        printf("FAIL (max diff: %.2e)\n", max_diff);
        return 0;
    }
}

/* Memory test */
static void test_memory() {
    printf("\nMemory Management Test:\n");
    
    size_t initial = flux_rocm_memory_used();
    printf("  Initial memory: %zu bytes\n", initial);
    
    /* Allocate some tensors */
    const int N = 10;
    flux_rocm_tensor_t tensors[N];
    for (int i = 0; i < N; i++) {
        tensors[i] = flux_rocm_tensor_alloc(1024 * 1024);  /* 4MB each */
    }
    
    size_t after_alloc = flux_rocm_memory_used();
    printf("  After allocating %d x 4MB tensors: %zu bytes (%.1f MB)\n", 
           N, after_alloc, after_alloc / (1024.0 * 1024.0));
    
    /* Free half */
    for (int i = 0; i < N/2; i++) {
        flux_rocm_tensor_free(tensors[i]);
        tensors[i] = nullptr;
    }
    
    /* Memory should stay same (pooled) */
    size_t after_free = flux_rocm_memory_used();
    printf("  After freeing %d tensors (pooled): %zu bytes\n", N/2, after_free);
    
    /* Allocate same size - should reuse */
    for (int i = 0; i < N/2; i++) {
        tensors[i] = flux_rocm_tensor_alloc(1024 * 1024);
    }
    
    size_t after_reuse = flux_rocm_memory_used();
    printf("  After re-allocating (should reuse): %zu bytes\n", after_reuse);
    
    /* Cleanup */
    for (int i = 0; i < N; i++) {
        if (tensors[i]) flux_rocm_tensor_free(tensors[i]);
    }
    
    /* Reset pool */
    flux_rocm_reset();
    size_t after_reset = flux_rocm_memory_used();
    printf("  After reset: %zu bytes\n", after_reset);
}

int main() {
    printf("=== FLUX ROCm Infrastructure Test ===\n\n");
    
    /* Initialize */
    if (!flux_rocm_init()) {
        printf("Failed to initialize ROCm\n");
        return 1;
    }
    
    printf("Memory used after init: %zu bytes\n\n", flux_rocm_memory_used());
    
    /* Run tests */
    int passed = 0, total = 0;
    
    total++; if (test_sgemm()) passed++;
    total++; if (test_linear()) passed++;
    
    test_sgemm_perf();
    test_memory();
    
    printf("\n=== Results: %d/%d tests passed ===\n", passed, total);
    
    /* Cleanup */
    flux_rocm_cleanup();
    
    return (passed == total) ? 0 : 1;
}
