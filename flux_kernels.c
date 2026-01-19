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

  /* Fallback to naive if allocation would be too large (>512MB) to avoid OOM */
  if (col_size_elements * sizeof(float) > 512 * 1024 * 1024) {
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
 * ======================================================================== */

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
 * ======================================================================== */

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
