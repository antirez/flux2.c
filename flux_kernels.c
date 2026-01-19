/*
 * FLUX Math Kernels - Implementation
 *
 * Math operations for FLUX inference.
 * Uses Metal/MPS on Apple Silicon, BLAS otherwise.
 */

#include "flux_kernels.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Use Metal for GPU acceleration on Apple Silicon */
#ifdef USE_METAL
#include "flux_metal.h"
#endif

/* Use BLAS for matrix operations when enabled via Makefile */
#ifdef USE_BLAS
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
/* OpenBLAS or rocBLAS-backed CBLAS */
#include <cblas.h>
#endif
#endif

/* Minimum matrix size to use GPU (smaller matrices are faster on CPU) */
#define MIN_GPU_ELEMENTS (512 * 512)
#define MAX_IM2COL_MEM (512L * 1024 * 1024) /* 512MB threshold for fallback */

/* Progress callbacks - set by caller before inference */
flux_substep_callback_t flux_substep_callback = NULL;
flux_step_callback_t flux_step_callback = NULL;
flux_phase_callback_t flux_phase_callback = NULL;

/* ========================================================================
 * Random Number Generator (xoshiro256**)
 * ======================================================================== */

static uint64_t rng_state[4] = {0x853c49e6748fea9bULL, 0xda3e39cb94b95bdbULL,
                                0x647c4677a2884327ULL, 0xc6e7918d2e2969f5ULL};

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
  while (u1 == 0.0f)
    u1 = flux_random_uniform();
  return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265358979323846f * u2);
}

void flux_randn(float *out, int n) {
  /* Random generation is sequential due to state update */
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
#pragma omp parallel for
  for (int i = 0; i < n; i++) {
    out[i] = a[i] + b[i];
  }
}

void flux_add_scalar(float *out, const float *a, float s, int n) {
#pragma omp parallel for
  for (int i = 0; i < n; i++) {
    out[i] = a[i] + s;
  }
}

void flux_sub(float *out, const float *a, const float *b, int n) {
#pragma omp parallel for
  for (int i = 0; i < n; i++) {
    out[i] = a[i] - b[i];
  }
}

void flux_mul(float *out, const float *a, const float *b, int n) {
#pragma omp parallel for
  for (int i = 0; i < n; i++) {
    out[i] = a[i] * b[i];
  }
}

void flux_mul_scalar(float *out, const float *a, float s, int n) {
#pragma omp parallel for
  for (int i = 0; i < n; i++) {
    out[i] = a[i] * s;
  }
}

void flux_div(float *out, const float *a, const float *b, int n) {
#pragma omp parallel for
  for (int i = 0; i < n; i++) {
    out[i] = a[i] / b[i];
  }
}

void flux_add_inplace(float *a, const float *b, int n) {
#pragma omp parallel for
  for (int i = 0; i < n; i++) {
    a[i] += b[i];
  }
}

void flux_mul_inplace(float *a, const float *b, int n) {
#pragma omp parallel for
  for (int i = 0; i < n; i++) {
    a[i] *= b[i];
  }
}

void flux_scale_inplace(float *a, float s, int n) {
#pragma omp parallel for
  for (int i = 0; i < n; i++) {
    a[i] *= s;
  }
}

void flux_axpy(float *a, float scale, const float *b, int n) {
#pragma omp parallel for
  for (int i = 0; i < n; i++) {
    a[i] += scale * b[i];
  }
}

/* ========================================================================
 * Matrix Operations
 * ======================================================================== */

void flux_matmul(float *C, const float *A, const float *B, int M, int K,
                 int N) {
  /* C[M,N] = A[M,K] @ B[K,N] */

#ifdef USE_METAL
  size_t matrix_elements = (size_t)M * N;
  if (flux_metal_available() && matrix_elements >= MIN_GPU_ELEMENTS) {
    flux_metal_sgemm(0, 0, /* no transpose */
                     M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
    return;
  }
#endif

#ifdef USE_BLAS
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, A, K, B,
              N, 0.0f, C, N);
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

void flux_matmul_t(float *C, const float *A, const float *B, int M, int K,
                   int N) {
  /* C[M,N] = A[M,K] @ B[N,K]^T */

#ifdef USE_METAL
  size_t matrix_elements = (size_t)M * N;
  if (flux_metal_available() && matrix_elements >= MIN_GPU_ELEMENTS) {
    flux_metal_sgemm(0, 1, /* no transpose A, transpose B */
                     M, N, K, 1.0f, A, K, B, K, 0.0f, C, N);
    return;
  }
#endif

#ifdef USE_BLAS
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0f, A, K, B,
              K, 0.0f, C, N);
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

void flux_batched_matmul(float *C, const float *A, const float *B, int batch,
                         int M, int K, int N) {
  int a_stride = M * K;
  int b_stride = K * N;
  int c_stride = M * N;

  for (int b = 0; b < batch; b++) {
    flux_matmul(C + b * c_stride, A + b * a_stride, B + b * b_stride, M, K, N);
  }
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
    flux_metal_sgemm(0, 1, /* no transpose A, transpose B */
                     seq_len, out_dim, in_dim, 1.0f, x, in_dim, W, in_dim, 0.0f,
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

#ifdef USE_BLAS
  /* Use BLAS sgemm: C = alpha * A @ B^T + beta * C
   * A[M, K] = x[seq_len, in_dim]
   * B[N, K] = W[out_dim, in_dim]
   * C[M, N] = y[seq_len, out_dim]
   */
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, seq_len, out_dim, in_dim,
              1.0f, x, in_dim, W, in_dim, 0.0f, y, out_dim);

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

void flux_linear_nobias(float *y, const float *x, const float *W, int seq_len,
                        int in_dim, int out_dim) {
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
    flux_metal_sgemm_bf16(0, 1, /* no transpose A, transpose B */
                          seq_len, out_dim, in_dim, 1.0f, x, in_dim, W_bf16,
                          in_dim, 0.0f, y, out_dim);
    return;
  }
#endif

  /* Fallback: convert bf16 to f32 and use regular linear */
  float *W_f32 = (float *)malloc((size_t)out_dim * in_dim * sizeof(float));
  if (!W_f32)
    return;

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

int flux_gpu_in_batch(void) {
#ifdef USE_METAL
  return flux_metal_in_batch();
#else
  return 0;
#endif
}

/* ========================================================================
 * Convolution Operations
 * ======================================================================== */

#ifdef USE_BLAS
/* im2col: Extract patches from input image into columns for BLAS matmul */
static void im2col(const float *in, float *col, int in_ch, int H, int W, int kH,
                   int kW, int stride, int padding, int outH, int outW) {
/* Parallelized im2col implementation */
#pragma omp parallel for collapse(2)
  for (int ic = 0; ic < in_ch; ic++) {
    for (int kh = 0; kh < kH; kh++) {
      for (int kw = 0; kw < kW; kw++) {

        /* Calculate col_row index directly */
        int col_row = ic * kH * kW + kh * kW + kw;

        for (int oh = 0; oh < outH; oh++) {
          for (int ow = 0; ow < outW; ow++) {
            int ih = oh * stride - padding + kh;
            int iw = ow * stride - padding + kw;
            int col_idx = col_row * (outH * outW) + oh * outW + ow;

            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
              col[col_idx] = in[ic * H * W + ih * W + iw];
            } else {
              col[col_idx] = 0.0f;
            }
          }
        }
      }
    }
  }
}
#endif /* USE_BLAS */

void flux_conv2d(float *out, const float *in, const float *weight,
                 const float *bias, int batch, int in_ch, int out_ch, int H,
                 int W, int kH, int kW, int stride, int padding) {
  int outH = (H + 2 * padding - kH) / stride + 1;
  int outW = (W + 2 * padding - kW) / stride + 1;

#ifdef USE_BLAS
  /* im2col + BLAS optimization */
  size_t col_size_elements = (size_t)in_ch * kH * kW * outH * outW;

  /* Fallback to naive if allocation would be too large (>MAX_IM2COL_MEM) to
   * avoid OOM */
  if (col_size_elements * sizeof(float) > MAX_IM2COL_MEM) {
    goto naive_fallback;
  }

  float *col = malloc(col_size_elements * sizeof(float));
  if (!col) {
    /* Fallback to naive if allocation fails */
    goto naive_fallback;
  }

  for (int b = 0; b < batch; b++) {
    const float *in_b = in + b * in_ch * H * W;
    float *out_b = out + b * out_ch * outH * outW;

    /* im2col: col[in_ch*kH*kW, outH*outW] */
    im2col(in_b, col, in_ch, H, W, kH, kW, stride, padding, outH, outW);

    /* BLAS: out[out_ch, outH*outW] = weight[out_ch, in_ch*kH*kW] @
     * col[in_ch*kH*kW, outH*outW] */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, out_ch, outH * outW,
                in_ch * kH * kW, 1.0f, weight, in_ch * kH * kW, col,
                outH * outW, 0.0f, out_b, outH * outW);

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
/* Parallelize across batch and output channels */
#pragma omp parallel for collapse(2)
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
                  int w_idx =
                      oc * in_ch * kH * kW + ic * kH * kW + kh * kW + kw;
                  sum += in[in_idx] * weight[w_idx];
                }
              }
            }
          }

          int out_idx =
              b * out_ch * outH * outW + oc * outH * outW + oh * outW + ow;
          out[out_idx] = sum;
        }
      }
    }
  }
}

void flux_conv2d_transpose(float *out, const float *in, const float *weight,
                           const float *bias, int batch, int in_ch, int out_ch,
                           int H, int W, int kH, int kW, int stride,
                           int padding, int output_padding) {
  int outH = (H - 1) * stride - 2 * padding + kH + output_padding;
  int outW = (W - 1) * stride - 2 * padding + kW + output_padding;

  /* Initialize output with bias */
  for (int b = 0; b < batch; b++) {
    for (int oc = 0; oc < out_ch; oc++) {
      float b_val = (bias != NULL) ? bias[oc] : 0.0f;
      for (int oh = 0; oh < outH; oh++) {
        for (int ow = 0; ow < outW; ow++) {
          int idx =
              b * out_ch * outH * outW + oc * outH * outW + oh * outW + ow;
          out[idx] = b_val;
        }
      }
    }
  }

  /* Gather-based convolution (Transposed) - Optimized */
  /* Replaces Scatter to fix race conditions AND improve cache locality */

#pragma omp parallel for collapse(2)
  for (int b = 0; b < batch; b++) {
    for (int oc = 0; oc < out_ch; oc++) {
      float bias_val = (bias != NULL) ? bias[oc] : 0.0f;

      for (int oh = 0; oh < outH; oh++) {
        for (int ow = 0; ow < outW; ow++) {

          float sum = bias_val;

          /* Gather contribution from relevant input pixels */
          /* oh = ih * stride - padding + kh */

          /* Iterate over kernel: kh, kw */
          for (int kh = 0; kh < kH; kh++) {
            /* Solve for ih: ih = (oh + padding - kh) / stride */
            int num_h = oh + padding - kh;
            if (num_h % stride != 0)
              continue;
            int ih = num_h / stride;
            if (ih < 0 || ih >= H)
              continue;

            for (int kw = 0; kw < kW; kw++) {
              int num_w = ow + padding - kw;
              if (num_w % stride != 0)
                continue;
              int iw = num_w / stride;
              if (iw < 0 || iw >= W)
                continue;

              /* Input exists. Sum over all input channels */
              /* In: [b, ic, H, W] */
              /* Weight: [ic, oc, kH, kW] */

              const float *in_ptr = in + b * in_ch * H * W + ih * W + iw;
              const float *w_ptr = weight + oc * kH * kW + kh * kW + kw;
              int in_step = H * W; // stride for ic
              int w_step =
                  out_ch * kH * kW; // stride for ic in weight [in, out, k, k]

              for (int ic = 0; ic < in_ch; ic++) {
                sum += (*in_ptr) * (*w_ptr);
                in_ptr += in_step;
                w_ptr += w_step;
              }
            }
          }

          /* Write result */
          out[b * out_ch * outH * outW + oc * outH * outW + oh * outW + ow] =
              sum;
        }
      }
    }
  }
}

void flux_conv2d_depthwise(float *out, const float *in, const float *weight,
                           const float *bias, int batch, int channels, int H,
                           int W, int kH, int kW, int stride, int padding) {
  int outH = (H + 2 * padding - kH) / stride + 1;
  int outW = (W + 2 * padding - kW) / stride + 1;

  for (int b = 0; b < batch; b++) {
#pragma omp parallel for collapse(2)
    for (int c = 0; c < channels; c++) {
      for (int oh = 0; oh < outH; oh++) {
        for (int ow = 0; ow < outW; ow++) {
          float sum = (bias != NULL) ? bias[c] : 0.0f;

          for (int kh = 0; kh < kH; kh++) {
            for (int kw = 0; kw < kW; kw++) {
              int ih = oh * stride - padding + kh;
              int iw = ow * stride - padding + kw;

              if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                int in_idx = b * channels * H * W + c * H * W + ih * W + iw;
                int w_idx = c * kH * kW + kh * kW + kw;
                sum += in[in_idx] * weight[w_idx];
              }
            }
          }

          int out_idx =
              b * channels * outH * outW + c * outH * outW + oh * outW + ow;
          out[out_idx] = sum;
        }
      }
    }
  }
}

/* ========================================================================
 * Normalization
 * ======================================================================== */

void flux_layer_norm(float *out, const float *x, const float *gamma,
                     const float *beta, int seq_len, int hidden, float eps) {
  for (int s = 0; s < seq_len; s++) {
    const float *x_row = x + s * hidden;
    float *out_row = out + s * hidden;

    /* Compute mean */
    float mean = 0.0f;
    for (int i = 0; i < hidden; i++) {
      mean += x_row[i];
    }
    mean /= hidden;

    /* Compute variance */
    float var = 0.0f;
    for (int i = 0; i < hidden; i++) {
      float diff = x_row[i] - mean;
      var += diff * diff;
    }
    var /= hidden;

    /* Normalize and scale */
    float std_inv = 1.0f / sqrtf(var + eps);
    for (int i = 0; i < hidden; i++) {
      float norm = (x_row[i] - mean) * std_inv;
      out_row[i] = gamma[i] * norm + beta[i];
    }
  }
}

void flux_rms_norm(float *out, const float *x, const float *weight, int seq_len,
                   int hidden, float eps) {
#ifdef USE_METAL
  /* Use GPU for RMSNorm only for very large tensors
   * The CPU-GPU sync overhead usually outweighs benefits for smaller ops */
  size_t elements = (size_t)seq_len * hidden;
  if (flux_metal_shaders_available() && elements >= 1024 * 1024) {
    flux_metal_rms_norm(out, x, weight, seq_len, hidden, eps);
    return;
  }
#endif

#pragma omp parallel for
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

void flux_group_norm(float *out, const float *x, const float *gamma,
                     const float *beta, int batch, int channels, int H, int W,
                     int num_groups, float eps) {
  int channels_per_group = channels / num_groups;
  int spatial = H * W;

#pragma omp parallel for collapse(2)
  for (int b = 0; b < batch; b++) {
    for (int g = 0; g < num_groups; g++) {
      int c_start = g * channels_per_group;
      int c_end = c_start + channels_per_group;

      /* Compute mean and variance for this group */
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

      /* Normalize and scale */
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

void flux_batch_norm(float *out, const float *x, const float *running_mean,
                     const float *running_var, const float *gamma,
                     const float *beta, int batch, int channels, int H, int W,
                     float eps) {
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

void flux_gelu(float *x, int n) {
#pragma omp parallel for
  for (int i = 0; i < n; i++) {
    float val = x[i];
    /* GELU(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) */
    x[i] = 0.5f * val *
           (1.0f + tanhf(0.7978845608f * (val + 0.044715f * val * val * val)));
  }
}

void flux_silu(float *x, int n) {
#ifdef USE_METAL
  /* Use GPU for very large arrays (overhead not worth it for small ones) */
  if (flux_metal_shaders_available() && n >= 4 * 1024 * 1024) {
    flux_metal_silu(x, n);
    return;
  }
#endif

#pragma omp parallel for
  for (int i = 0; i < n; i++) {
    float val = x[i];
    x[i] = val / (1.0f + expf(-val));
  }
}

void flux_swiglu(float *out, const float *x, const float *gate, int n) {
#pragma omp parallel for
  for (int i = 0; i < n; i++) {
    float g = gate[i];
    float silu_g = g / (1.0f + expf(-g));
    out[i] = x[i] * silu_g;
  }
}

void flux_softmax(float *x, int rows, int cols) {
#ifdef USE_METAL
  /* Use GPU only for very large softmax operations
   * Sync overhead usually dominates for smaller ops */
  if (flux_metal_shaders_available() &&
      (size_t)rows * cols >= 4 * 1024 * 1024) {
    flux_metal_softmax(x, rows, cols);
    return;
  }
#endif

#pragma omp parallel for
  for (int r = 0; r < rows; r++) {
    float *row = x + r * cols;

    /* Find max for numerical stability */
    float max_val = row[0];
    for (int c = 1; c < cols; c++) {
      if (row[c] > max_val)
        max_val = row[c];
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

void flux_sigmoid(float *x, int n) {
#pragma omp parallel for
  for (int i = 0; i < n; i++) {
    x[i] = 1.0f / (1.0f + expf(-x[i]));
  }
}

void flux_tanh(float *x, int n) {
#pragma omp parallel for
  for (int i = 0; i < n; i++) {
    x[i] = tanhf(x[i]);
  }
}

/* ========================================================================
 * Attention Operations
 * ========================================================================
 */

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

void flux_attention_masked(float *out, const float *Q, const float *K,
                           const float *V, const float *mask, int batch,
                           int heads, int seq_q, int seq_k, int head_dim,
                           float scale) {
  float *scores = (float *)malloc(seq_q * seq_k * sizeof(float));

  for (int b = 0; b < batch; b++) {
    for (int h = 0; h < heads; h++) {
      const float *q = Q + (b * heads + h) * seq_q * head_dim;
      const float *k = K + (b * heads + h) * seq_k * head_dim;
      const float *v = V + (b * heads + h) * seq_k * head_dim;
      float *o = out + (b * heads + h) * seq_q * head_dim;

      /* scores = Q @ K^T * scale + mask */
      for (int i = 0; i < seq_q; i++) {
        for (int j = 0; j < seq_k; j++) {
          float dot = 0.0f;
          for (int d = 0; d < head_dim; d++) {
            dot += q[i * head_dim + d] * k[j * head_dim + d];
          }
          float m = (mask != NULL) ? mask[i * seq_k + j] : 0.0f;
          scores[i * seq_k + j] = dot * scale + m;
        }
      }

      flux_softmax(scores, seq_q, seq_k);

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
 * the full [seq_q, seq_k] attention matrix. Reduces memory from O(n²) to
 * O(n).
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
 * ========================================================================
 */

/*
 * Flash attention for a single head.
 * Q: [seq_q, head_dim], K: [seq_k, head_dim], V: [seq_k, head_dim]
 * out: [seq_q, head_dim]
 * Uses O(head_dim) working memory per query instead of O(seq_k).
 */
static void flash_attention_head(float *out, const float *Q, const float *K,
                                 const float *V, int seq_q, int seq_k,
                                 int head_dim, float scale) {
  /* Process each query position independently */
  for (int i = 0; i < seq_q; i++) {
    const float *q_row = Q + i * head_dim;
    float *o_row = out + i * head_dim;

    /* Running statistics for online softmax */
    float max_score = -1e30f; /* Large negative value (avoid -INFINITY with
                                 -ffast-math) */
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
static void flash_attention_head_tiled(float *out, const float *Q,
                                       const float *K, const float *V,
                                       int seq_q, int seq_k, int head_dim,
                                       float scale, float *tile_scores,
                                       int q_tile_size, int k_tile_size) {
  /* Per-query running statistics: max_score[seq_q], sum_exp[seq_q] */
  float *max_scores = (float *)malloc(seq_q * sizeof(float));
  float *sum_exps = (float *)malloc(seq_q * sizeof(float));

  /* Initialize */
  for (int i = 0; i < seq_q; i++) {
    max_scores[i] = -1e30f; /* Large negative value (avoid -INFINITY with
                               -ffast-math) */
    sum_exps[i] = 0.0f;
  }
  memset(out, 0, seq_q * head_dim * sizeof(float));

  /* Process in tiles over K/V dimension */
  for (int k_start = 0; k_start < seq_k; k_start += k_tile_size) {
    int k_end = (k_start + k_tile_size < seq_k) ? k_start + k_tile_size : seq_k;
    int k_len = k_end - k_start;

    /* Process in tiles over Q dimension */
    for (int q_start = 0; q_start < seq_q; q_start += q_tile_size) {
      int q_end =
          (q_start + q_tile_size < seq_q) ? q_start + q_tile_size : seq_q;
      int q_len = q_end - q_start;

      const float *Q_tile = Q + q_start * head_dim;
      const float *K_tile = K + k_start * head_dim;
      const float *V_tile = V + k_start * head_dim;
      float *out_tile = out + q_start * head_dim;

      /* Compute tile scores: Q_tile @ K_tile^T * scale */
#ifdef USE_BLAS
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, q_len, k_len,
                  head_dim, scale, Q_tile, head_dim, K_tile, head_dim, 0.0f,
                  tile_scores, k_tile_size);
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
          if (score_row[ki] > tile_max)
            tile_max = score_row[ki];
        }

        /* Compute correction factors */
        float old_max = max_scores[i];
        float new_max = (tile_max > old_max) ? tile_max : old_max;

        /* Rescale old accumulations if needed */
        if (old_max > -1e29f) { /* Check if we have prior accumulations */
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
void flux_flash_attention(float *out, const float *Q, const float *K,
                          const float *V, int seq_q, int seq_k, int heads,
                          int head_dim, float scale) {
  /* Tile sizes for cache efficiency */
  int q_tile_size = 32; /* Process 32 queries at a time */
  int k_tile_size = 64; /* Process 64 keys at a time */

  /* Allocate tile scratch buffer */
  float *tile_scores =
      (float *)malloc(q_tile_size * k_tile_size * sizeof(float));

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

      flash_attention_head(out_contig, Q_contig, K_contig, V_contig, seq_q,
                           seq_k, head_dim, scale);

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
                                 seq_q, seq_k, head_dim, scale, tile_scores,
                                 q_tile_size, k_tile_size);

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

void flux_apply_rope(float *x, const float *freqs, int batch, int seq,
                     int heads, int head_dim) {
  /* x: [batch, seq, heads, head_dim]
   * freqs: [seq, head_dim/2, 2] (cos, sin)
   * Apply rotary embedding to pairs of dimensions */

  int half_dim = head_dim / 2;

#pragma omp parallel for collapse(3)
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

void flux_compute_rope_freqs(float *freqs, const int *pos, int seq, int dim,
                             float theta) {
  int half_dim = dim / 2;

#pragma omp parallel for
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
 * ========================================================================
 */

void flux_avgpool2d(float *out, const float *in, int batch, int channels, int H,
                    int W, int kH, int kW, int stride, int padding) {
  int outH = (H + 2 * padding - kH) / stride + 1;
  int outW = (W + 2 * padding - kW) / stride + 1;

  for (int b = 0; b < batch; b++) {
#pragma omp parallel for collapse(2)
    for (int c = 0; c < channels; c++) {
      for (int oh = 0; oh < outH; oh++) {
        for (int ow = 0; ow < outW; ow++) {
          float sum = 0.0f;
          int count = 0;

          for (int kh = 0; kh < kH; kh++) {
            for (int kw = 0; kw < kW; kw++) {
              int ih = oh * stride - padding + kh;
              int iw = ow * stride - padding + kw;

              if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                int idx = b * channels * H * W + c * H * W + ih * W + iw;
                sum += in[idx];
                count++;
              }
            }
          }

          int out_idx =
              b * channels * outH * outW + c * outH * outW + oh * outW + ow;
          out[out_idx] = sum / count;
        }
      }
    }
  }
}

void flux_maxpool2d(float *out, const float *in, int batch, int channels, int H,
                    int W, int kH, int kW, int stride, int padding) {
  int outH = (H + 2 * padding - kH) / stride + 1;
  int outW = (W + 2 * padding - kW) / stride + 1;

  for (int b = 0; b < batch; b++) {
#pragma omp parallel for collapse(2)
    for (int c = 0; c < channels; c++) {
      for (int oh = 0; oh < outH; oh++) {
        for (int ow = 0; ow < outW; ow++) {
          float max_val = -1e30f;

          for (int kh = 0; kh < kH; kh++) {
            for (int kw = 0; kw < kW; kw++) {
              int ih = oh * stride - padding + kh;
              int iw = ow * stride - padding + kw;

              if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                int idx = b * channels * H * W + c * H * W + ih * W + iw;
                if (in[idx] > max_val)
                  max_val = in[idx];
              }
            }
          }

          int out_idx =
              b * channels * outH * outW + c * outH * outW + oh * outW + ow;
          out[out_idx] = max_val;
        }
      }
    }
  }
}

void flux_upsample_nearest(float *out, const float *in, int batch, int channels,
                           int H, int W, int scale_h, int scale_w) {
  int outH = H * scale_h;
  int outW = W * scale_w;

  for (int b = 0; b < batch; b++) {
#pragma omp parallel for collapse(2)
    for (int c = 0; c < channels; c++) {
      for (int oh = 0; oh < outH; oh++) {
        for (int ow = 0; ow < outW; ow++) {
          int ih = oh / scale_h;
          int iw = ow / scale_w;
          int in_idx = b * channels * H * W + c * H * W + ih * W + iw;
          int out_idx =
              b * channels * outH * outW + c * outH * outW + oh * outW + ow;
          out[out_idx] = in[in_idx];
        }
      }
    }
  }
}

void flux_upsample_bilinear(float *out, const float *in, int batch,
                            int channels, int H, int W, int out_H, int out_W) {
  float scale_h = (float)H / out_H;
  float scale_w = (float)W / out_W;

  for (int b = 0; b < batch; b++) {
#pragma omp parallel for collapse(2)
    for (int c = 0; c < channels; c++) {
      for (int oh = 0; oh < out_H; oh++) {
        for (int ow = 0; ow < out_W; ow++) {
          float ih = (oh + 0.5f) * scale_h - 0.5f;
          float iw = (ow + 0.5f) * scale_w - 0.5f;

          int ih0 = (int)floorf(ih);
          int iw0 = (int)floorf(iw);
          int ih1 = ih0 + 1;
          int iw1 = iw0 + 1;

          float h_weight = ih - ih0;
          float w_weight = iw - iw0;

          ih0 = (ih0 < 0) ? 0 : (ih0 >= H) ? H - 1 : ih0;
          ih1 = (ih1 < 0) ? 0 : (ih1 >= H) ? H - 1 : ih1;
          iw0 = (iw0 < 0) ? 0 : (iw0 >= W) ? W - 1 : iw0;
          iw1 = (iw1 < 0) ? 0 : (iw1 >= W) ? W - 1 : iw1;

          int base = b * channels * H * W + c * H * W;
          float v00 = in[base + ih0 * W + iw0];
          float v01 = in[base + ih0 * W + iw1];
          float v10 = in[base + ih1 * W + iw0];
          float v11 = in[base + ih1 * W + iw1];

          float val = v00 * (1 - h_weight) * (1 - w_weight) +
                      v01 * (1 - h_weight) * w_weight +
                      v10 * h_weight * (1 - w_weight) +
                      v11 * h_weight * w_weight;

          int out_idx = b * channels * out_H * out_W + c * out_H * out_W +
                        oh * out_W + ow;
          out[out_idx] = val;
        }
      }
    }
  }
}

void flux_patchify(float *out, const float *in, int batch, int channels, int H,
                   int W, int patch_size) {
  /* [B, C, H, W] -> [B, C*p*p, H/p, W/p] */
  int p = patch_size;
  int outH = H / p;
  int outW = W / p;
  int out_ch = channels * p * p;

  for (int b = 0; b < batch; b++) {
#pragma omp parallel for collapse(2)
    for (int c = 0; c < channels; c++) {
      for (int ph = 0; ph < outH; ph++) {
        for (int pw = 0; pw < outW; pw++) {
          for (int pi = 0; pi < p; pi++) {
            for (int pj = 0; pj < p; pj++) {
              int ih = ph * p + pi;
              int iw = pw * p + pj;
              int in_idx = b * channels * H * W + c * H * W + ih * W + iw;

              int out_c = c * p * p + pi * p + pj;
              int out_idx = b * out_ch * outH * outW + out_c * outH * outW +
                            ph * outW + pw;
              out[out_idx] = in[in_idx];
            }
          }
        }
      }
    }
  }
}

void flux_unpatchify(float *out, const float *in, int batch, int channels,
                     int H, int W, int patch_size) {
  /* [B, C*p*p, H, W] -> [B, C, H*p, W*p] */
  int p = patch_size;
  int in_ch = channels * p * p;
  int outH = H * p;
  int outW = W * p;

  for (int b = 0; b < batch; b++) {
#pragma omp parallel for collapse(2)
    for (int c = 0; c < channels; c++) {
      for (int ph = 0; ph < H; ph++) {
        for (int pw = 0; pw < W; pw++) {
          for (int pi = 0; pi < p; pi++) {
            for (int pj = 0; pj < p; pj++) {
              int in_c = c * p * p + pi * p + pj;
              int in_idx = b * in_ch * H * W + in_c * H * W + ph * W + pw;

              int oh = ph * p + pi;
              int ow = pw * p + pj;
              int out_idx =
                  b * channels * outH * outW + c * outH * outW + oh * outW + ow;
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
 * ========================================================================
 */

void flux_copy(float *dst, const float *src, int n) {
  memcpy(dst, src, n * sizeof(float));
}

void flux_fill(float *x, float val, int n) {
  for (int i = 0; i < n; i++) {
    x[i] = val;
  }
}

float flux_sum(const float *x, int n) {
  float sum = 0.0f;
  for (int i = 0; i < n; i++) {
    sum += x[i];
  }
  return sum;
}

float flux_mean(const float *x, int n) { return flux_sum(x, n) / n; }

float flux_var(const float *x, int n) {
  float mean = flux_mean(x, n);
  float var = 0.0f;
  for (int i = 0; i < n; i++) {
    float diff = x[i] - mean;
    var += diff * diff;
  }
  return var / n;
}

float flux_norm(const float *x, int n) {
  float sum = 0.0f;
  for (int i = 0; i < n; i++) {
    sum += x[i] * x[i];
  }
  return sqrtf(sum);
}

float flux_dot(const float *a, const float *b, int n) {
  float sum = 0.0f;
  for (int i = 0; i < n; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

void flux_clamp(float *x, float min_val, float max_val, int n) {
  for (int i = 0; i < n; i++) {
    if (x[i] < min_val)
      x[i] = min_val;
    else if (x[i] > max_val)
      x[i] = max_val;
  }
}

void flux_transpose(float *out, const float *in, int M, int N) {
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      out[n * M + m] = in[m * N + n];
    }
  }
}
