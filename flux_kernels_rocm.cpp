/*
 * flux_kernels_rocm.cpp
 * HIP/ROCm kernels for FLUX VAE (Full GPU path)
 */

#include <hip/hip_runtime.h>
#include <math.h>
#include <rocblas/rocblas.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" {

/* Helper to check HIP errors */
#define HIP_CHECK(call)                                                        \
  do {                                                                         \
    hipError_t err = call;                                                     \
    if (err != hipSuccess) {                                                   \
      fprintf(stderr, "HIP Error: %s at %s:%d\n", hipGetErrorString(err),      \
              __FILE__, __LINE__);                                             \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

/* ========================================================================
 * HIP Kernels (Internal)
 * ======================================================================== */

/* SILU Activation: x / (1 + exp(-x)) */
__global__ void k_silu(float *x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float val = x[i];
    x[i] = val / (1.0f + expf(-val));
  }
}

/* Element-wise Add: out = a + b */
__global__ void k_add(float *out, const float *a, const float *b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = a[i] + b[i];
  }
}

/* Element-wise Add In-place: a += b */
__global__ void k_add_inplace(float *a, const float *b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    a[i] += b[i];
  }
}

/* Upsample Nearest: 2x2 */
__global__ void k_upsample_nearest(float *out, const float *in, int channels,
                                   int inH, int inW, int scale) {
  int outH = inH * scale;
  int outW = inW * scale;
  int total = channels * outH * outW;

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < total) {
    int ow = i % outW;
    int oh = (i / outW) % outH;
    int c = i / (outW * outH);

    int iw = ow / scale;
    int ih = oh / scale;
    int in_idx = c * inH * inW + ih * inW + iw;

    out[i] = in[in_idx];
  }
}

/* Group Norm Kernel (Naive implementation for simplicity) */
/* One block per group/channel? For now, simple parallel reduction might be
   complex. Let's do a multi-pass approach or simple block-per-group if spatial
   is small. Given VAE spatial can be large (1024x1024), we need efficient
   reduction. For MVP: Single kernel using atomicAdd or multiple kernels.
   Simpler approach: Calculate Mean/Var on CPU (copy small data) or use
   optimized kernel. Let's write a robust kernel for [B, G, C_per_G, HW].
 */
__global__ void k_group_norm_stats(const float *x, float *mean, float *var,
                                   int batch, int groups,
                                   int channels_per_group, int spatial) {
  /* Each thread handles some spatial elements and accumulates to shared memory?
     Too complex for rapid dev. Let's do a grid-stride loop or simple reduction.
     For implementation speed, we'll assign one block per (batch, group).
  */
  int b = blockIdx.x;
  int g = blockIdx.y;
  int items_per_group = channels_per_group * spatial;
  int block_start = b * groups * items_per_group + g * items_per_group;

  extern __shared__ float sdata[];
  float *s_mean = sdata;
  float *s_var = sdata + blockDim.x;

  float local_sum = 0.0f;
  float local_sq_sum = 0.0f;

  for (int i = threadIdx.x; i < items_per_group; i += blockDim.x) {
    float val = x[block_start + i];
    local_sum += val;
    local_sq_sum += val * val;
  }

  /* Reduction in shared memory */
  s_mean[threadIdx.x] = local_sum;
  s_var[threadIdx.x] = local_sq_sum;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      s_mean[threadIdx.x] += s_mean[threadIdx.x + s];
      s_var[threadIdx.x] += s_var[threadIdx.x + s];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    int idx = b * groups + g;
    float final_mean = s_mean[0] / items_per_group;
    mean[idx] = final_mean;
    var[idx] = (s_var[0] / items_per_group) - (final_mean * final_mean);
  }
}

__global__ void k_group_norm_apply(float *out, const float *x,
                                   const float *mean, const float *var,
                                   const float *gamma, const float *beta,
                                   int batch, int groups,
                                   int channels_per_group, int spatial,
                                   float eps) {
  int total = batch * groups * channels_per_group * spatial;
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < total) {
    /* reverse index mapping */
    int sp_idx = i % spatial;
    int tmp = i / spatial;
    int c_pg = tmp % channels_per_group;
    tmp = tmp / channels_per_group;
    int g = tmp % groups;
    int b = tmp / groups;

    int c = g * channels_per_group + c_pg; // absolute channel index
    int stat_idx = b * groups + g;

    float m = mean[stat_idx];
    float v = var[stat_idx];
    float std_inv = rsqrtf(v + eps);

    float g_val = (gamma) ? gamma[c] : 1.0f;
    float b_val = (beta) ? beta[c] : 0.0f;

    float val = x[i];
    out[i] = g_val * (val - m) * std_inv + b_val;
  }
}

/* ========================================================================
 * C Interface Wrappers
 * ======================================================================== */

void *rocm_malloc(size_t size) {
  void *ptr;
  HIP_CHECK(hipMalloc(&ptr, size));
  return ptr;
}

void rocm_free(void *ptr) {
  if (ptr)
    HIP_CHECK(hipFree(ptr));
}

void rocm_memcpy_h2d(void *dst, const void *src, size_t size) {
  HIP_CHECK(hipMemcpy(dst, src, size, hipMemcpyHostToDevice));
}

void rocm_memcpy_d2h(void *dst, const void *src, size_t size) {
  HIP_CHECK(hipMemcpy(dst, src, size, hipMemcpyDeviceToHost));
}

void rocm_memset(void *dst, int val, size_t size) {
  HIP_CHECK(hipMemset(dst, val, size));
}

void rocm_silu(float *x, int n) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  k_silu<<<blocks, threads>>>(x, n);
  HIP_CHECK(hipGetLastError());
}

void rocm_add(float *out, const float *a, const float *b, int n) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  k_add<<<blocks, threads>>>(out, a, b, n);
}

void rocm_add_inplace(float *a, const float *b, int n) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  k_add_inplace<<<blocks, threads>>>(a, b, n);
}

void rocm_upsample_nearest(float *out, const float *in, int batch, int channels,
                           int H, int W, int scale) {
  int total_elements = batch * channels * H * scale * W * scale;
  int threads = 256;
  int blocks = (total_elements + threads - 1) / threads;

  /* Simplified kernel call: treating batch*channels as 'channels' dimension */
  k_upsample_nearest<<<blocks, threads>>>(out, in, batch * channels, H, W,
                                          scale);
}

void rocm_group_norm(float *out, const float *x, const float *gamma,
                     const float *beta, int batch, int channels, int H, int W,
                     int num_groups, float eps) {

  /* 1. Calculate Mean/Var per group */
  float *d_mean, *d_var;
  HIP_CHECK(hipMalloc(&d_mean, batch * num_groups * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_var, batch * num_groups * sizeof(float)));

  int threads = 256;
  dim3 grid(batch, num_groups);
  int shared_mem = 2 * threads * sizeof(float);

  k_group_norm_stats<<<grid, threads, shared_mem>>>(
      x, d_mean, d_var, batch, num_groups, channels / num_groups, H * W);

  /* 2. Apply Normalization */
  int total = batch * channels * H * W;
  int blocks = (total + threads - 1) / threads;
  k_group_norm_apply<<<blocks, threads>>>(out, x, d_mean, d_var, gamma, beta,
                                          batch, num_groups,
                                          channels / num_groups, H * W, eps);

  HIP_CHECK(hipFree(d_mean));
  HIP_CHECK(hipFree(d_var));
}

/* ========================================================================
 * GPU Convolution Support
 * ======================================================================== */

/* Device-side im2col kernel */
__global__ void k_im2col(const float *in, float *col, int in_ch, int H, int W,
                         int kH, int kW, int stride, int padding, int outH,
                         int outW) {
  int total = in_ch * outH * outW;
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < total) {
    /* i maps to (ic, oh, ow) */
    int ow = i % outW;
    int tmp = i / outW;
    int oh = tmp % outH;
    int ic = tmp / outH;

    /* Calculate base input offset */
    int base_col_idx = (ic * outH * outW + oh * outW + ow) * (kH * kW);

    /* Unroll kernel loop in registers if possible, or just loop */
    /* Note: original im2col layout is col[in_ch*kH*kW, outH*outW] for GEMM KxN
     */
    /* Here we write to col[in_ch, kH, kW, outH, outW] flattened? */
    /* Standard im2col for GEMM (patches as columns):
       Output matrix is [in_ch*kH*kW, outH*outW] */

    /* We are parallelizing over output pixels (columns of the result matrix) */
    /* For a given pixel (oh, ow) and channel ic, we copy the patch */

    /* Wait, standard im2col makes a tall matrix where each column is a patch.
       Rows are (ic, kh, kw).
       Col index is (oh, ow).
    */

    /* Let's strictly follow the layout needed for rocblas_sgemm */
    /* Weights: [out_ch, in_ch*kH*kW] */
    /* Input Col: [in_ch*kH*kW, outH*outW] */
    /* Gemm: Weights * Col = [out_ch, outH*outW] */

    /* We need to write to Col.
       Col has dimensions (K, N) where K = in_ch*kH*kW, N = outH*outW.
       The pixel (oh, ow) corresponds to column 'j' = oh*outW + ow.
       The row 'r' corresponds to (ic, kh, kw).
       Address = r * N + j
    */

    int col_idx_base = oh * outW + ow; /* The column index in the matrix */
    int N = outH * outW;

    for (int kh = 0; kh < kH; kh++) {
      for (int kw = 0; kw < kW; kw++) {
        int ih = oh * stride - padding + kh;
        int iw = ow * stride - padding + kw;

        /* Row index in the column matrix */
        int row_idx = ic * kH * kW + kh * kW + kw;

        int dest_idx = row_idx * N + col_idx_base;

        if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
          int src_idx = ic * H * W + ih * W + iw;
          col[dest_idx] = in[src_idx];
        } else {
          col[dest_idx] = 0.0f;
        }
      }
    }
  }
}

/* Add bias kernel */
__global__ void k_add_bias(float *out, const float *bias, int n_out,
                           int spatial) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n_out * spatial) {
    int c = i / spatial;
    out[i] += bias[c];
  }
}

static rocblas_handle handle = NULL;

void rocm_init_context() {
  if (!handle)
    rocblas_create_handle(&handle);
}

void rocm_conv2d(float *d_out, const float *d_in, const float *d_weight,
                 const float *d_bias, int batch, int in_ch, int out_ch, int H,
                 int W, int kH, int kW, int stride, int padding) {
  if (!handle)
    rocm_init_context();

  int outH = (H + 2 * padding - kH) / stride + 1;
  int outW = (W + 2 * padding - kW) / stride + 1;

  /* Calculate temporary column buffer size */
  size_t col_elements = (size_t)in_ch * kH * kW * outH * outW;

  /* Allocate device temporary memory */
  /* Note: In a real efficient engine, this should be pre-allocated workspace */
  float *d_col;
  HIP_CHECK(hipMalloc(&d_col, col_elements * sizeof(float)));

  /* Run per batch */
  for (int b = 0; b < batch; b++) {
    const float *in_b = d_in + b * in_ch * H * W;
    float *out_b = d_out + b * out_ch * outH * outW;

    /* 1. Im2Col Kernel */
    /* Threads: one per output pixel per input channel? No, one per output COL
     * is better? */
    /* Let's launch threads covering (in_ch, outH, outW) */
    /* This simplifies implementation but each thread loops 3x3 */
    int total_threads = in_ch * outH * outW;
    int blocks = (total_threads + 255) / 256;
    k_im2col<<<blocks, 256>>>(in_b, d_col, in_ch, H, W, kH, kW, stride, padding,
                              outH, outW);

    /* 2. GEMM: Weight [Out_Ch, K] * Col [K, Pixels] = Out [Out_Ch, Pixels] */
    int m = out_ch;
    int n = outH * outW;
    int k = in_ch * kH * kW;

    const float alpha = 1.0f;
    const float beta = 0.0f;

    /* Weights are assumed row-major [Out_Ch, K] */
    /* rocBLAS is col-major.
       C = A * B
       To get C (row-major), we compute C^T = B^T * A^T
       A (Weights) row-major -> A^T col-major
       B (Col) row-major -> B^T col-major

       This is confusing. Let's use standard trick:
       We want C = A * B.
       Pass B as A to rocBLAS, Pass A as B to rocBLAS.
       Result is C^T (col-major) which is C (row-major).

       So we compute Col * Weights^T ?? No.

       Let's stick to standard row-major logic mapping:
       sgemm(N, M, K, args...)
       N = number of columns of C
       M = number of rows of C

       We want C [M, N].
       rocblas_sgemm(handle, transB, transA, N, M, K, ...)
         Pass B (Col) as first matrix (ldb = N)
         Pass A (Weight) as second matrix (lda = K)
         Result C (ldc = N)

       Wait, leading dim of RowMajor(MxN) is N.
    */

    rocblas_status status = rocblas_sgemm(
        handle,
        rocblas_operation_none,    /* B (Col) is not transposed effectively? */
        rocblas_operation_none,    /* A (Weight) is not transposed? */
        n, m, k, &alpha, d_col, n, /* "A" = Col [K, N], lda=N */
        d_weight, k,               /* "B" = Wgt [M, K], ldb=K */
        &beta, out_b, n            /* "C" = Out [M, N], ldc=N */
    );

    if (status != rocblas_status_success) {
      fprintf(stderr, "ROCBLAS Error in conv2d: %d\n", status);
    }

    /* 3. Add Bias */
    if (d_bias) {
      int total_out = out_ch * outH * outW;
      int b_blocks = (total_out + 255) / 256;
      k_add_bias<<<b_blocks, 256>>>(out_b, d_bias, out_ch, outH * outW);
    }
  }

  HIP_CHECK(hipFree(d_col));
}

} /* extern "C" */
