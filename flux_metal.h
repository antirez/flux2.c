/*
 * FLUX Metal Acceleration
 *
 * GPU-accelerated matrix operations using Apple Metal Performance Shaders.
 * Provides significant speedup on Apple Silicon Macs.
 */

#ifndef FLUX_METAL_H
#define FLUX_METAL_H

#include <stddef.h>

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
 * Get GPU memory usage info (for debugging).
 */
size_t flux_metal_memory_used(void);

#ifdef __cplusplus
}
#endif

#endif /* FLUX_METAL_H */
