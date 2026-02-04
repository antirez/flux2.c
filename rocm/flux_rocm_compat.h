/*
 * FLUX ROCm Compatibility Layer
 *
 * Drop-in wrappers that match flux_metal.h function signatures.
 * This allows flux_kernels.c to use ROCm with minimal changes.
 */

#ifndef FLUX_ROCM_COMPAT_H
#define FLUX_ROCM_COMPAT_H

#include "flux_rocm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Alias init/available/cleanup to match Metal naming */
#define flux_metal_init flux_rocm_init
#define flux_metal_available flux_rocm_available
#define flux_metal_cleanup flux_rocm_cleanup
#define flux_metal_sync flux_rocm_sync

/* Alias SGEMM operations */
#define flux_metal_sgemm flux_rocm_sgemm
#define flux_metal_sgemm_bf16 flux_rocm_sgemm_bf16
#define flux_metal_sgemm_batch flux_rocm_sgemm_batch

/* Alias kernel operations */
#define flux_metal_rms_norm flux_rocm_rms_norm
#define flux_metal_silu flux_rocm_silu
#define flux_metal_softmax flux_rocm_softmax
#define flux_metal_shaders_available flux_rocm_kernels_available

/* Batch/chain operations */
#define flux_metal_begin_batch flux_rocm_batch_begin
#define flux_metal_end_batch flux_rocm_batch_end

/*
 * CPU-pointer wrapper for SGEMM.
 * Handles upload to GPU, compute, download automatically.
 * Uses weight caching for B matrix (typically the model weights).
 */
void flux_rocm_sgemm_cpu(int transpose_a, int transpose_b,
                         int M, int N, int K,
                         float alpha,
                         const float *A, int lda,
                         const float *B, int ldb,
                         float beta,
                         float *C, int ldc);

/*
 * CPU-pointer wrapper for RMSNorm.
 */
void flux_rocm_rms_norm_cpu(float *out, const float *x, const float *weight,
                            int seq_len, int hidden, float eps);

/*
 * CPU-pointer wrapper for SiLU.
 */
void flux_rocm_silu_cpu(float *x, int n);

/*
 * CPU-pointer wrapper for Softmax.
 */
void flux_rocm_softmax_cpu(float *x, int rows, int cols);

#ifdef __cplusplus
}
#endif

#endif /* FLUX_ROCM_COMPAT_H */
