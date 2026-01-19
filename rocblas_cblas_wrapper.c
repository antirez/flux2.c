/*
 * CBLAS to rocBLAS Wrapper
 * Provides CBLAS interface for rocBLAS GPU acceleration with HIP memory
 * management
 */

#include <cblas.h>
#include <hip/hip_runtime_api.h>
#include <rocblas/rocblas.h>
#include <stdio.h>
#include <stdlib.h>

/* Minimum matrix size to use GPU (smaller matrices are faster on CPU) */
#define MIN_GPU_ELEMENTS (512 * 512)

static rocblas_handle handle = NULL;

__attribute__((constructor)) static void init_rocblas(void) {
  if (rocblas_create_handle(&handle) != rocblas_status_success) {
    fprintf(stderr, "Warning: Failed to initialize rocBLAS handle\n");
    handle = NULL;
  }
}

__attribute__((destructor)) static void cleanup_rocblas(void) {
  if (handle) {
    rocblas_destroy_handle(handle);
    handle = NULL;
  }
}

void cblas_sgemm(const CBLAS_ORDER Order, const CBLAS_TRANSPOSE TransA,
                 const CBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const float alpha, const float *A, const int lda,
                 const float *B, const int ldb, const float beta, float *C,
                 const int ldc) {
  if (!handle || Order != CblasRowMajor) {
    fprintf(stderr, "rocBLAS wrapper: invalid parameters\n");
    return;
  }

  /* NOTE: MIN_GPU_ELEMENTS threshold removed - all matrices use GPU
   * Small matrices may have overhead, but ensures computation happens
   * TODO: Implement CPU fallback for small matrices if performance is poor */
  // size_t matrix_elements = (size_t)M * N;
  // if (matrix_elements < MIN_GPU_ELEMENTS) {
  //   return;  // This was leaving C unmodified!
  // }

  /* Allocate GPU memory */
  float *d_A = NULL, *d_B = NULL, *d_C = NULL;
  size_t size_A = (size_t)M * K * sizeof(float);
  size_t size_B = (size_t)K * N * sizeof(float);
  size_t size_C = (size_t)M * N * sizeof(float);

  hipError_t hip_status;

  hip_status = hipMalloc((void **)&d_A, size_A);
  if (hip_status != hipSuccess) {
    fprintf(stderr, "rocBLAS wrapper: hipMalloc failed for A: %s\n",
            hipGetErrorString(hip_status));
    return;
  }

  hip_status = hipMalloc((void **)&d_B, size_B);
  if (hip_status != hipSuccess) {
    fprintf(stderr, "rocBLAS wrapper: hipMalloc failed for B: %s\n",
            hipGetErrorString(hip_status));
    hipFree(d_A);
    return;
  }

  hip_status = hipMalloc((void **)&d_C, size_C);
  if (hip_status != hipSuccess) {
    fprintf(stderr, "rocBLAS wrapper: hipMalloc failed for C: %s\n",
            hipGetErrorString(hip_status));
    hipFree(d_A);
    hipFree(d_B);
    return;
  }

  /* Copy data from CPU to GPU */
  hip_status = hipMemcpy(d_A, A, size_A, hipMemcpyHostToDevice);
  if (hip_status != hipSuccess) {
    fprintf(stderr, "rocBLAS wrapper: hipMemcpy failed for A: %s\n",
            hipGetErrorString(hip_status));
    goto cleanup;
  }

  hip_status = hipMemcpy(d_B, B, size_B, hipMemcpyHostToDevice);
  if (hip_status != hipSuccess) {
    fprintf(stderr, "rocBLAS wrapper: hipMemcpy failed for B: %s\n",
            hipGetErrorString(hip_status));
    goto cleanup;
  }

  /* If beta != 0, we need to copy C to GPU first */
  if (beta != 0.0f) {
    hip_status = hipMemcpy(d_C, C, size_C, hipMemcpyHostToDevice);
    if (hip_status != hipSuccess) {
      fprintf(stderr, "rocBLAS wrapper: hipMemcpy failed for C: %s\n",
              hipGetErrorString(hip_status));
      goto cleanup;
    }
  }

  /* Convert CBLAS transpose to rocBLAS transpose */
  rocblas_operation transA_op = (TransA == CblasTrans)
                                    ? rocblas_operation_transpose
                                    : rocblas_operation_none;
  rocblas_operation transB_op = (TransB == CblasTrans)
                                    ? rocblas_operation_transpose
                                    : rocblas_operation_none;

  /* Execute rocBLAS operation on GPU
   * rocBLAS uses column-major, so we swap A and B and transpose */
  rocblas_status status =
      rocblas_sgemm(handle, transB_op, transA_op, N, M, K, &alpha, d_B, ldb,
                    d_A, lda, &beta, d_C, ldc);
  if (status != rocblas_status_success) {
    fprintf(stderr, "rocBLAS wrapper: rocblas_sgemm failed with status %d\n",
            status);
    goto cleanup;
  }

  /* Copy result back from GPU to CPU */
  hip_status = hipMemcpy(C, d_C, size_C, hipMemcpyDeviceToHost);
  if (hip_status != hipSuccess) {
    fprintf(stderr, "rocBLAS wrapper: hipMemcpy back failed for C: %s\n",
            hipGetErrorString(hip_status));
  }

cleanup:
  /* Free GPU memory */
  if (d_A)
    hipFree(d_A);
  if (d_B)
    hipFree(d_B);
  if (d_C)
    hipFree(d_C);
}
