/*
 * Qwen3 Text Encoder Implementation
 *
 * Implements Qwen3-4B model for text encoding in FLUX image generation.
 * - 36 transformer layers
 * - 2560 hidden dimension
 * - GQA with 32 query heads and 8 KV heads
 * - RoPE positional embeddings
 * - SwiGLU MLP
 */

#include "flux_qwen3.h"
#include "flux_safetensors.h"
#include "flux_kernels.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Use BLAS for matrix operations when enabled via Makefile */
#ifdef USE_BLAS
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#endif

/* Use Metal for GPU acceleration */
#ifdef USE_METAL
#include "flux_metal.h"
#endif
#ifdef USE_CUDA
#include "flux_cuda.h"
#endif

/* Minimum matrix size for GPU acceleration.
 * Using 10M threshold keeps text encoder on CPU (Accelerate BLAS), which is
 * faster and avoids GPU memory pressure on 16GB systems. Text encoder weights
 * are only used once per generation, so GPU caching provides no benefit.
 * Fixes issue #9: SIGKILL on 16GB Metal systems during text encoding. */
#define QWEN3_MIN_GPU_ELEMENTS (10 * 1024 * 1024)

/* Maximum number of safetensors shards (Qwen3-8B may have >2) */
#define QWEN3_MAX_SHARDS 16

/* ========================================================================
 * Data Structures
 * ======================================================================== */

typedef struct {
    float *q_proj_weight;     /* [num_heads * head_dim, hidden] = [4096, 2560] */
    float *k_proj_weight;     /* [num_kv_heads * head_dim, hidden] = [1024, 2560] */
    float *v_proj_weight;     /* [num_kv_heads * head_dim, hidden] = [1024, 2560] */
    float *o_proj_weight;     /* [hidden, num_heads * head_dim] = [2560, 4096] */
    float *q_norm_weight;     /* [head_dim] = [128] */
    float *k_norm_weight;     /* [head_dim] = [128] */
    /* BF16 weight pointers (for GPU path) */
    uint16_t *q_proj_weight_bf16;
    uint16_t *k_proj_weight_bf16;
    uint16_t *v_proj_weight_bf16;
    uint16_t *o_proj_weight_bf16;
    uint16_t *q_norm_weight_bf16;  /* [head_dim] = [128] */
    uint16_t *k_norm_weight_bf16;  /* [head_dim] = [128] */
} qwen3_attention_t;

typedef struct {
    float *gate_proj_weight;  /* [intermediate, hidden] = [9728, 2560] */
    float *up_proj_weight;    /* [intermediate, hidden] = [9728, 2560] */
    float *down_proj_weight;  /* [hidden, intermediate] = [2560, 9728] */
    /* BF16 weight pointers (for GPU path) */
    uint16_t *gate_proj_weight_bf16;
    uint16_t *up_proj_weight_bf16;
    uint16_t *down_proj_weight_bf16;
} qwen3_mlp_t;

typedef struct {
    float *input_layernorm_weight;      /* [hidden] */
    float *post_attention_layernorm_weight;  /* [hidden] */
    qwen3_attention_t attn;
    qwen3_mlp_t mlp;
    /* BF16 layer norm weights (for GPU path) - unused currently, kept for future */
    uint16_t *input_layernorm_weight_bf16;
    uint16_t *post_attention_layernorm_weight_bf16;
} qwen3_layer_t;

struct qwen3_model {
    /* Architecture (from config.json) */
    int hidden_size;
    int intermediate_size;
    int num_heads;
    int num_kv_heads;
    int head_dim;
    int vocab_size;
    float rope_theta;
    int text_dim;             /* 3 * hidden_size */

    /* Embedding layer */
    float *embed_tokens;      /* [vocab_size, hidden] */

    /* Transformer layers */
    qwen3_layer_t *layers;    /* [num_layers] */
    int num_layers;

    /* Final layer norm */
    float *norm_weight;       /* [hidden] */

    /* RoPE precomputed */
    float *rope_cos;          /* [max_seq_len, head_dim/2] */
    float *rope_sin;          /* [max_seq_len, head_dim/2] */

    /* Working memory */
    float *hidden_state;      /* [seq_len, hidden] */
    float *residual;          /* [seq_len, hidden] */
    float *q_buf;             /* [seq_len, num_heads * head_dim] */
    float *k_buf;             /* [seq_len, num_kv_heads * head_dim] */
    float *v_buf;             /* [seq_len, num_kv_heads * head_dim] */
    float *attn_scores;       /* [num_heads, seq_len, seq_len] */
    float *attn_out;          /* [seq_len, num_heads * head_dim] */
    float *mlp_gate;          /* [seq_len, intermediate] */
    float *mlp_up;            /* [seq_len, intermediate] */
    float *mlp_out;           /* [seq_len, hidden] */
    float *norm_buf;          /* [seq_len, hidden] */

    /* Output layers storage (for extracting layers 9, 18, 27) */
    float *layer_outputs[3];  /* [seq_len, hidden] each */

    /* Pre-allocated attention work buffers (avoid per-call allocation) */
    float *attn_q_head;       /* [seq_len, head_dim] */
    float *attn_v_head;       /* [seq_len, head_dim] */
    float *attn_out_head;     /* [seq_len, head_dim] */
    /* CUDA batched attention staging buffers [num_heads, seq_len, head_dim] */
    float *attn_q_hsd;
    float *attn_k_hsd;
    float *attn_v_hsd;
    float *attn_out_hsd;

    /* Mmap mode: keep safetensors files open, load layer weights on-demand */
    int use_mmap;
    safetensors_file_t *sf_files[QWEN3_MAX_SHARDS];
    int num_sf_files;

    /* BF16 GPU acceleration */
    int use_bf16;
};

/* Forward declarations for mmap streaming mode */
static int load_layer_weights(qwen3_layer_t *layer, safetensors_file_t **files,
                              int num_files, int layer_idx);
#ifdef USE_METAL
static int load_layer_weights_small_f32(qwen3_layer_t *layer, safetensors_file_t **files,
                                        int num_files, int layer_idx);
static int load_layer_weights_bf16(qwen3_layer_t *layer, safetensors_file_t **files,
                                   int num_files, int layer_idx);
#endif
static void free_layer_weights(qwen3_layer_t *layer);

/* ========================================================================
 * Basic Operations
 * ======================================================================== */

static void qwen3_linear(float *y, const float *x, const float *W,
                         int seq_len, int in_dim, int out_dim) {
    /* y[seq, out] = x[seq, in] @ W[out, in]^T */
#ifdef USE_METAL
    /* Use GPU for large matrices */
    size_t matrix_elements = (size_t)seq_len * out_dim;
    if (flux_metal_available() && matrix_elements >= QWEN3_MIN_GPU_ELEMENTS) {
        flux_metal_sgemm(0, 1,  /* no transpose A, transpose B */
                         seq_len, out_dim, in_dim,
                         1.0f,
                         x, in_dim,
                         W, in_dim,
                         0.0f,
                         y, out_dim);
        return;
    }
#endif

#if defined(USE_BLAS) || defined(USE_CUDA)
    /* Route through shared GEMM so CUDA builds can use GPU linear kernels. */
    flux_matmul_t(y, x, W, seq_len, in_dim, out_dim);
#else
    for (int s = 0; s < seq_len; s++) {
        for (int o = 0; o < out_dim; o++) {
            float sum = 0.0f;
            for (int i = 0; i < in_dim; i++) {
                sum += x[s * in_dim + i] * W[o * in_dim + i];
            }
            y[s * out_dim + o] = sum;
        }
    }
#endif
}

static void qwen3_rms_norm(float *out, const float *x, const float *weight,
                           int seq_len, int hidden, float eps) {
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

/* Per-head RMS norm for Q/K normalization */
static void qwen3_head_rms_norm(float *out, const float *x, const float *weight,
                                int seq_len, int num_heads, int head_dim, float eps) {
    for (int s = 0; s < seq_len; s++) {
        for (int h = 0; h < num_heads; h++) {
            const float *x_head = x + s * num_heads * head_dim + h * head_dim;
            float *out_head = out + s * num_heads * head_dim + h * head_dim;

            /* Compute RMS for this head */
            float sum_sq = 0.0f;
            for (int i = 0; i < head_dim; i++) {
                sum_sq += x_head[i] * x_head[i];
            }
            float rms = sqrtf(sum_sq / head_dim + eps);
            float rms_inv = 1.0f / rms;

            /* Normalize and scale */
            for (int i = 0; i < head_dim; i++) {
                out_head[i] = x_head[i] * rms_inv * weight[i];
            }
        }
    }
}

static void qwen3_softmax(float *x, int len) {
    float max_val = x[0];
    for (int i = 1; i < len; i++) {
        if (x[i] > max_val) max_val = x[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < len; i++) {
        x[i] = fast_expf(x[i] - max_val);
        sum += x[i];
    }

    float inv_sum = 1.0f / sum;
    for (int i = 0; i < len; i++) {
        x[i] *= inv_sum;
    }
}

/* ========================================================================
 * RoPE (Rotary Position Embedding)
 * ======================================================================== */

static void compute_rope_freqs(float *cos_out, float *sin_out,
                               int max_seq_len, int head_dim, float theta) {
    int half_dim = head_dim / 2;

    for (int pos = 0; pos < max_seq_len; pos++) {
        for (int i = 0; i < half_dim; i++) {
            float freq = 1.0f / powf(theta, (float)(2 * i) / head_dim);
            float angle = pos * freq;
            cos_out[pos * half_dim + i] = cosf(angle);
            sin_out[pos * half_dim + i] = sinf(angle);
        }
    }
}

static void apply_rope(float *q, float *k, const float *cos_cache, const float *sin_cache,
                       int seq_len, int num_q_heads, int num_kv_heads, int head_dim) {
    int half_dim = head_dim / 2;

    /* Apply RoPE to Q */
    for (int s = 0; s < seq_len; s++) {
        const float *cos_row = cos_cache + s * half_dim;
        const float *sin_row = sin_cache + s * half_dim;

        for (int h = 0; h < num_q_heads; h++) {
            float *q_head = q + s * num_q_heads * head_dim + h * head_dim;

            for (int i = 0; i < half_dim; i++) {
                float x0 = q_head[i];
                float x1 = q_head[i + half_dim];
                float cos_val = cos_row[i];
                float sin_val = sin_row[i];

                q_head[i] = x0 * cos_val - x1 * sin_val;
                q_head[i + half_dim] = x0 * sin_val + x1 * cos_val;
            }
        }
    }

    /* Apply RoPE to K */
    for (int s = 0; s < seq_len; s++) {
        const float *cos_row = cos_cache + s * half_dim;
        const float *sin_row = sin_cache + s * half_dim;

        for (int h = 0; h < num_kv_heads; h++) {
            float *k_head = k + s * num_kv_heads * head_dim + h * head_dim;

            for (int i = 0; i < half_dim; i++) {
                float x0 = k_head[i];
                float x1 = k_head[i + half_dim];
                float cos_val = cos_row[i];
                float sin_val = sin_row[i];

                k_head[i] = x0 * cos_val - x1 * sin_val;
                k_head[i + half_dim] = x0 * sin_val + x1 * cos_val;
            }
        }
    }
}

/* ========================================================================
 * Attention
 * ======================================================================== */

static void qwen3_attention_forward(qwen3_model_t *model, qwen3_layer_t *layer,
                                    int seq_len, const int *attention_mask) {
    int num_heads = model->num_heads;
    int num_kv_heads = model->num_kv_heads;
    int head_dim = model->head_dim;
    int hidden = model->hidden_size;
    int kv_dim = num_kv_heads * head_dim;
    int q_dim = num_heads * head_dim;
    float scale = 1.0f / sqrtf((float)head_dim);

    /* Q, K, V projections */
    qwen3_linear(model->q_buf, model->norm_buf, layer->attn.q_proj_weight,
                 seq_len, hidden, q_dim);
    qwen3_linear(model->k_buf, model->norm_buf, layer->attn.k_proj_weight,
                 seq_len, hidden, kv_dim);
    qwen3_linear(model->v_buf, model->norm_buf, layer->attn.v_proj_weight,
                 seq_len, hidden, kv_dim);

    /* Q/K RMS normalization (per-head) */
    qwen3_head_rms_norm(model->q_buf, model->q_buf, layer->attn.q_norm_weight,
                        seq_len, num_heads, head_dim, QWEN3_RMS_NORM_EPS);
    qwen3_head_rms_norm(model->k_buf, model->k_buf, layer->attn.k_norm_weight,
                        seq_len, num_kv_heads, head_dim, QWEN3_RMS_NORM_EPS);

    /* Apply RoPE */
    apply_rope(model->q_buf, model->k_buf, model->rope_cos, model->rope_sin,
               seq_len, num_heads, num_kv_heads, head_dim);

#ifdef USE_METAL
    /* Try GPU-accelerated causal attention for all heads in parallel.
     * The GPU kernel uses both causal masking and attention mask.
     * This ensures exact parity with CPU implementation. */
    if (flux_metal_available()) {
        if (flux_metal_causal_attention(model->attn_out,
                                         model->q_buf, model->k_buf, model->v_buf,
                                         attention_mask,
                                         seq_len, num_heads, num_kv_heads,
                                         head_dim, scale)) {
            /* GPU attention succeeded - skip to output projection */
            goto output_proj;
        }
    }
#endif

    /* CPU fallback: compute attention for each head with GQA
     * Use BLAS for Q@K^T and scores@V matrix multiplications */
    {
        int heads_per_kv = num_heads / num_kv_heads;

#ifdef USE_CUDA
        /* CUDA batched GQA path: one attention launch for all query heads.
         * Pack Q/K/V from [seq, hidden] strided layout to [heads, seq, head_dim]. */
        if (model->attn_q_hsd && model->attn_k_hsd && model->attn_v_hsd && model->attn_out_hsd) {
            float *q_hsd = model->attn_q_hsd;
            float *k_hsd = model->attn_k_hsd;
            float *v_hsd = model->attn_v_hsd;
            float *out_hsd = model->attn_out_hsd;

            for (int h = 0; h < num_heads; h++) {
                int kv_h = h / heads_per_kv;
                float *q_h = q_hsd + (size_t)h * seq_len * head_dim;
                float *k_h = k_hsd + (size_t)h * seq_len * head_dim;
                float *v_h = v_hsd + (size_t)h * seq_len * head_dim;
                for (int s = 0; s < seq_len; s++) {
                    const float *q_src = model->q_buf + s * q_dim + h * head_dim;
                    const float *k_src = model->k_buf + s * kv_dim + kv_h * head_dim;
                    const float *v_src = model->v_buf + s * kv_dim + kv_h * head_dim;
                    memcpy(q_h + s * head_dim, q_src, head_dim * sizeof(float));
                    memcpy(k_h + s * head_dim, k_src, head_dim * sizeof(float));
                    memcpy(v_h + s * head_dim, v_src, head_dim * sizeof(float));
                }
            }

            if (flux_cuda_attention_batched(out_hsd, q_hsd, k_hsd, v_hsd,
                                            num_heads, seq_len, seq_len, head_dim,
                                            scale, 1, attention_mask)) {
                /* Unpack back to [seq, heads * head_dim]. */
                for (int h = 0; h < num_heads; h++) {
                    const float *out_h = out_hsd + (size_t)h * seq_len * head_dim;
                    for (int s = 0; s < seq_len; s++) {
                        float *dst = model->attn_out + s * q_dim + h * head_dim;
                        memcpy(dst, out_h + s * head_dim, head_dim * sizeof(float));
                    }
                }
                goto output_proj;
            }
        }
#endif

        for (int h = 0; h < num_heads; h++) {
            int kv_h = h / heads_per_kv;  /* Which KV head to use */
            float *scores = model->attn_scores + h * seq_len * seq_len;

            /* Q accessed directly with strided lda (avoids copy)
             * Q[s,d] = q_buf[s * q_dim + h * head_dim + d] */
            const float *q_strided = model->q_buf + h * head_dim;

            /* K accessed directly with strided lda + CblasTrans (avoids transpose)
             * K[s,d] = k_buf[s * kv_dim + kv_h * head_dim + d] */
            const float *k_strided = model->k_buf + kv_h * head_dim;

            /* scores = scale * Q @ K^T using strided BLAS */
#ifdef USE_BLAS
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        seq_len, seq_len, head_dim,
                        scale, q_strided, q_dim, k_strided, kv_dim,
                        0.0f, scores, seq_len);
#else
            /* Fallback: naive matmul */
            for (int i = 0; i < seq_len; i++) {
                for (int j = 0; j < seq_len; j++) {
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; d++) {
                        dot += q_strided[i * q_dim + d] * k_strided[j * kv_dim + d];
                    }
                    scores[i * seq_len + j] = dot * scale;
                }
            }
#endif

            /* Apply causal mask and attention mask, then softmax */
            for (int i = 0; i < seq_len; i++) {
                for (int j = 0; j < seq_len; j++) {
                    if (j > i) {
                        scores[i * seq_len + j] = -1e9f;
                    }
                    if (attention_mask && attention_mask[j] == 0) {
                        scores[i * seq_len + j] = -1e9f;
                    }
                }
                qwen3_softmax(scores + i * seq_len, seq_len);
            }

            /* V can be accessed directly with strided lda (avoids copy)
             * V[s,d] = v_buf[s * kv_dim + kv_h * head_dim + d] */
            const float *v_strided = model->v_buf + kv_h * head_dim;

            /* Output can be written directly with strided ldc (avoids copy)
             * out[s,d] = attn_out[s * q_dim + h * head_dim + d] */
            float *out_strided = model->attn_out + h * head_dim;

            /* out = scores @ V using strided BLAS (avoids V copy and output copy)
             * scores: [seq_len, seq_len], V: [seq_len, head_dim] with ldb=kv_dim */
#ifdef USE_BLAS
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        seq_len, head_dim, seq_len,
                        1.0f, scores, seq_len, v_strided, kv_dim,
                        0.0f, out_strided, q_dim);
#else
            for (int i = 0; i < seq_len; i++) {
                for (int d = 0; d < head_dim; d++) {
                    float sum = 0.0f;
                    for (int j = 0; j < seq_len; j++) {
                        sum += scores[i * seq_len + j] * v_strided[j * kv_dim + d];
                    }
                    out_strided[i * q_dim + d] = sum;
                }
            }
#endif
        }
    }

    /* Work buffers are pre-allocated in model, no free needed */

#if defined(USE_METAL) || defined(USE_CUDA)
output_proj:
#endif
    /* Output projection */
    qwen3_linear(model->hidden_state, model->attn_out, layer->attn.o_proj_weight,
                 seq_len, q_dim, hidden);
}

/* ========================================================================
 * MLP (SwiGLU)
 * ======================================================================== */

static void qwen3_mlp_forward(qwen3_model_t *model, qwen3_layer_t *layer, int seq_len) {
    int hidden = model->hidden_size;
    int intermediate = model->intermediate_size;

    /* Gate and Up projections */
    qwen3_linear(model->mlp_gate, model->norm_buf, layer->mlp.gate_proj_weight,
                 seq_len, hidden, intermediate);
    qwen3_linear(model->mlp_up, model->norm_buf, layer->mlp.up_proj_weight,
                 seq_len, hidden, intermediate);

    /* SwiGLU: silu(gate) * up - fused for better performance */
    flux_silu_mul(model->mlp_gate, model->mlp_up, seq_len * intermediate);

    /* Down projection */
    qwen3_linear(model->mlp_out, model->mlp_gate, layer->mlp.down_proj_weight,
                 seq_len, intermediate, hidden);
}

/* ========================================================================
 * Transformer Layer
 * ======================================================================== */

static void qwen3_layer_forward(qwen3_model_t *model, qwen3_layer_t *layer,
                                int seq_len, const int *attention_mask) {
    int hidden = model->hidden_size;

    /* Save residual */
    memcpy(model->residual, model->hidden_state, seq_len * hidden * sizeof(float));

    /* Pre-attention LayerNorm */
    qwen3_rms_norm(model->norm_buf, model->hidden_state, layer->input_layernorm_weight,
                   seq_len, hidden, QWEN3_RMS_NORM_EPS);

    /* Self-attention */
    qwen3_attention_forward(model, layer, seq_len, attention_mask);

    /* Residual connection */
    for (int i = 0; i < seq_len * hidden; i++) {
        model->hidden_state[i] += model->residual[i];
    }

    /* Save residual */
    memcpy(model->residual, model->hidden_state, seq_len * hidden * sizeof(float));

    /* Pre-MLP LayerNorm */
    qwen3_rms_norm(model->norm_buf, model->hidden_state, layer->post_attention_layernorm_weight,
                   seq_len, hidden, QWEN3_RMS_NORM_EPS);

    /* MLP */
    qwen3_mlp_forward(model, layer, seq_len);

    /* Residual connection */
    for (int i = 0; i < seq_len * hidden; i++) {
        model->hidden_state[i] = model->residual[i] + model->mlp_out[i];
    }
}

#ifdef USE_METAL
/* ========================================================================
 * BF16 GPU-Accelerated Layer Forward
 * Uses GPU for linear layers, keeps attention/norm on CPU for simplicity.
 * ======================================================================== */

/* Helper to convert f32 array to bf16 GPU tensor */
static flux_gpu_tensor_t f32_to_bf16_tensor(const float *data, int n) {
    flux_gpu_tensor_t f32_tensor = flux_gpu_tensor_create(data, n);
    if (!f32_tensor) return NULL;
    flux_gpu_tensor_t bf16_tensor = flux_gpu_tensor_f32_to_bf16(f32_tensor);
    flux_gpu_tensor_free(f32_tensor);
    return bf16_tensor;
}

/* Helper to read bf16 GPU tensor back to f32 array */
static void bf16_tensor_to_f32(flux_gpu_tensor_t bf16_tensor, float *out) {
    flux_gpu_tensor_t f32_tensor = flux_gpu_tensor_bf16_to_f32(bf16_tensor);
    if (f32_tensor) {
        flux_gpu_tensor_read(f32_tensor, out);
        flux_gpu_tensor_free(f32_tensor);
    }
}

/* Convert bf16 value to f32 */
static inline float bf16_to_f32_val(uint16_t bf16) {
    uint32_t f32_bits = ((uint32_t)bf16) << 16;
    float result;
    memcpy(&result, &f32_bits, sizeof(float));
    return result;
}

/* Helper to create bf16 GPU tensor from bf16 CPU data (for small tensors like norm weights) */
static flux_gpu_tensor_t bf16_ptr_to_bf16_tensor(const uint16_t *bf16_data, int n) {
    /* Convert bf16->f32 on CPU, then f32->bf16 on GPU */
    float *f32_tmp = malloc(n * sizeof(float));
    if (!f32_tmp) return NULL;
    for (int i = 0; i < n; i++) {
        f32_tmp[i] = bf16_to_f32_val(bf16_data[i]);
    }
    flux_gpu_tensor_t result = f32_to_bf16_tensor(f32_tmp, n);
    free(f32_tmp);
    return result;
}

/* GPU-accelerated MLP using bf16 weights */
static void qwen3_mlp_forward_bf16(qwen3_model_t *model, qwen3_layer_t *layer, int seq_len) {
    int hidden = model->hidden_size;
    int intermediate = model->intermediate_size;
    int n = seq_len * intermediate;

    /* Convert input to bf16 tensor on GPU */
    flux_gpu_tensor_t x = f32_to_bf16_tensor(model->norm_buf, seq_len * hidden);
    if (!x) {
        qwen3_mlp_forward(model, layer, seq_len);
        return;
    }

    /* Gate and Up projections on GPU */
    flux_gpu_tensor_t gate = flux_gpu_linear_bf16_native(x, layer->mlp.gate_proj_weight_bf16,
                                                          seq_len, hidden, intermediate);
    flux_gpu_tensor_t up = flux_gpu_linear_bf16_native(x, layer->mlp.up_proj_weight_bf16,
                                                        seq_len, hidden, intermediate);
    flux_gpu_tensor_free(x);

    if (!gate || !up) {
        if (gate) flux_gpu_tensor_free(gate);
        if (up) flux_gpu_tensor_free(up);
        qwen3_mlp_forward(model, layer, seq_len);
        return;
    }

    /* SwiGLU: silu(gate) * up on GPU */
    flux_gpu_silu_mul_bf16(gate, up, n);
    flux_gpu_tensor_free(up);

    /* Down projection on GPU */
    flux_gpu_tensor_t out = flux_gpu_linear_bf16_native(gate, layer->mlp.down_proj_weight_bf16,
                                                         seq_len, intermediate, hidden);
    flux_gpu_tensor_free(gate);

    if (!out) {
        qwen3_mlp_forward(model, layer, seq_len);
        return;
    }

    /* Read result back to CPU */
    bf16_tensor_to_f32(out, model->mlp_out);
    flux_gpu_tensor_free(out);
}

/* GPU-accelerated attention using bf16 weights for projections */
static void qwen3_attention_forward_bf16(qwen3_model_t *model, qwen3_layer_t *layer,
                                          int seq_len, const int *attention_mask) {
    int num_heads = model->num_heads;
    int num_kv_heads = model->num_kv_heads;
    int head_dim = model->head_dim;
    int hidden = model->hidden_size;
    int kv_dim = num_kv_heads * head_dim;
    int q_dim = num_heads * head_dim;
    float scale = 1.0f / sqrtf((float)head_dim);

    /* Convert input to bf16 tensor */
    flux_gpu_tensor_t x = f32_to_bf16_tensor(model->norm_buf, seq_len * hidden);
    if (!x) {
        qwen3_attention_forward(model, layer, seq_len, attention_mask);
        return;
    }

    /* Q, K, V projections on GPU */
    flux_gpu_tensor_t q = flux_gpu_linear_bf16_native(x, layer->attn.q_proj_weight_bf16,
                                                       seq_len, hidden, q_dim);
    flux_gpu_tensor_t k = flux_gpu_linear_bf16_native(x, layer->attn.k_proj_weight_bf16,
                                                       seq_len, hidden, kv_dim);
    flux_gpu_tensor_t v = flux_gpu_linear_bf16_native(x, layer->attn.v_proj_weight_bf16,
                                                       seq_len, hidden, kv_dim);
    flux_gpu_tensor_free(x);

    if (!q || !k || !v) {
        if (q) flux_gpu_tensor_free(q);
        if (k) flux_gpu_tensor_free(k);
        if (v) flux_gpu_tensor_free(v);
        qwen3_attention_forward(model, layer, seq_len, attention_mask);
        return;
    }

    /* Try full bf16 pipeline: Q/K norm, RoPE, and attention all on GPU */
    flux_gpu_tensor_t attn_out = flux_gpu_tensor_alloc_f16(seq_len * q_dim);
    if (attn_out && layer->attn.q_norm_weight_bf16 && layer->attn.k_norm_weight_bf16) {
        /* Get bf16 weight tensors for Q/K norm */
        flux_gpu_tensor_t q_norm_w = bf16_ptr_to_bf16_tensor(layer->attn.q_norm_weight_bf16, head_dim);
        flux_gpu_tensor_t k_norm_w = bf16_ptr_to_bf16_tensor(layer->attn.k_norm_weight_bf16, head_dim);

        if (q_norm_w && k_norm_w) {

            /* Q/K RMS normalization on GPU - separate calls for GQA (different head counts) */
            int q_norm_ok = flux_gpu_head_rms_norm_bf16(q, q_norm_w, seq_len, num_heads, head_dim, QWEN3_RMS_NORM_EPS);
            int k_norm_ok = flux_gpu_head_rms_norm_bf16(k, k_norm_w, seq_len, num_kv_heads, head_dim, QWEN3_RMS_NORM_EPS);

            flux_gpu_tensor_free(q_norm_w);
            flux_gpu_tensor_free(k_norm_w);

            if (q_norm_ok && k_norm_ok) {
                /* Apply RoPE - GPU bf16 */
                flux_gpu_rope_text_bf16(q, k, model->rope_cos, model->rope_sin,
                                         seq_len, num_heads, num_kv_heads, head_dim);

                /* GPU causal attention (bf16) */
                if (flux_gpu_causal_attention_bf16(attn_out, q, k, v, attention_mask,
                                                    seq_len, num_heads, num_kv_heads,
                                                    head_dim, scale)) {
                    flux_gpu_tensor_free(q);
                    flux_gpu_tensor_free(k);
                    flux_gpu_tensor_free(v);

                    /* Output projection on GPU - input already bf16 */
                    flux_gpu_tensor_t out = flux_gpu_linear_bf16_native(attn_out, layer->attn.o_proj_weight_bf16,
                                                                         seq_len, q_dim, hidden);
                    flux_gpu_tensor_free(attn_out);

                    if (out) {
                        bf16_tensor_to_f32(out, model->hidden_state);
                        flux_gpu_tensor_free(out);
                        return;  /* Success - full bf16 pipeline */
                    }
                }
            }
        } else {
            /* q_norm_w or k_norm_w allocation failed */
            if (q_norm_w) flux_gpu_tensor_free(q_norm_w);
            if (k_norm_w) flux_gpu_tensor_free(k_norm_w);
        }
        /* GPU bf16 path failed - free everything and use full CPU fallback */
        flux_gpu_tensor_free(q);
        flux_gpu_tensor_free(k);
        flux_gpu_tensor_free(v);
        flux_gpu_tensor_free(attn_out);
        qwen3_attention_forward(model, layer, seq_len, attention_mask);
        return;
    }

    /* Fallback: No attn_out or no bf16 norm weights - use CPU path */
    if (attn_out) flux_gpu_tensor_free(attn_out);
    bf16_tensor_to_f32(q, model->q_buf);
    bf16_tensor_to_f32(k, model->k_buf);
    bf16_tensor_to_f32(v, model->v_buf);
    flux_gpu_tensor_free(q);
    flux_gpu_tensor_free(k);
    flux_gpu_tensor_free(v);

    /* Q/K RMS normalization (per-head) - CPU */
    qwen3_head_rms_norm(model->q_buf, model->q_buf, layer->attn.q_norm_weight,
                        seq_len, num_heads, head_dim, QWEN3_RMS_NORM_EPS);
    qwen3_head_rms_norm(model->k_buf, model->k_buf, layer->attn.k_norm_weight,
                        seq_len, num_kv_heads, head_dim, QWEN3_RMS_NORM_EPS);

    /* Apply RoPE - CPU */
    apply_rope(model->q_buf, model->k_buf, model->rope_cos, model->rope_sin,
               seq_len, num_heads, num_kv_heads, head_dim);

    /* GPU causal attention (f32) */
    if (!flux_metal_causal_attention(model->attn_out,
                                      model->q_buf, model->k_buf, model->v_buf,
                                      attention_mask,
                                      seq_len, num_heads, num_kv_heads,
                                      head_dim, scale)) {
        qwen3_attention_forward(model, layer, seq_len, attention_mask);
        return;
    }

    /* Output projection on GPU */
    flux_gpu_tensor_t attn = f32_to_bf16_tensor(model->attn_out, seq_len * q_dim);
    if (!attn) {
        qwen3_linear(model->hidden_state, model->attn_out, layer->attn.o_proj_weight,
                     seq_len, q_dim, hidden);
        return;
    }

    flux_gpu_tensor_t out = flux_gpu_linear_bf16_native(attn, layer->attn.o_proj_weight_bf16,
                                                         seq_len, q_dim, hidden);
    flux_gpu_tensor_free(attn);

    if (!out) {
        qwen3_linear(model->hidden_state, model->attn_out, layer->attn.o_proj_weight,
                     seq_len, q_dim, hidden);
        return;
    }

    bf16_tensor_to_f32(out, model->hidden_state);
    flux_gpu_tensor_free(out);
}

/* GPU-accelerated layer forward */
static void qwen3_layer_forward_bf16(qwen3_model_t *model, qwen3_layer_t *layer,
                                      int seq_len, const int *attention_mask) {
    int hidden = model->hidden_size;

    /* Check if we have bf16 weights */
    if (!layer->attn.q_proj_weight_bf16 || !layer->mlp.gate_proj_weight_bf16) {
        qwen3_layer_forward(model, layer, seq_len, attention_mask);
        return;
    }

    /* Save residual */
    memcpy(model->residual, model->hidden_state, seq_len * hidden * sizeof(float));

    /* Pre-attention LayerNorm - CPU (GPU conversion overhead not worth it) */
    qwen3_rms_norm(model->norm_buf, model->hidden_state, layer->input_layernorm_weight,
                   seq_len, hidden, QWEN3_RMS_NORM_EPS);

    /* Self-attention with GPU projections */
    qwen3_attention_forward_bf16(model, layer, seq_len, attention_mask);

    /* Residual connection */
    for (int i = 0; i < seq_len * hidden; i++) {
        model->hidden_state[i] += model->residual[i];
    }

    /* Save residual */
    memcpy(model->residual, model->hidden_state, seq_len * hidden * sizeof(float));

    /* Pre-MLP LayerNorm - CPU (GPU conversion overhead not worth it) */
    qwen3_rms_norm(model->norm_buf, model->hidden_state, layer->post_attention_layernorm_weight,
                   seq_len, hidden, QWEN3_RMS_NORM_EPS);

    /* MLP with GPU */
    qwen3_mlp_forward_bf16(model, layer, seq_len);

    /* Residual connection */
    for (int i = 0; i < seq_len * hidden; i++) {
        model->hidden_state[i] = model->residual[i] + model->mlp_out[i];
    }
}

/* ========================================================================
 * Fully GPU-Resident Forward Pass
 *
 * Keeps hidden state on GPU (bf16) across all layers, eliminating the
 * 72 CPU-GPU syncs (2 per layer × 36 layers) of the mixed path above.
 * Only one GPU sync at the end for all 27 needed layers.
 * ======================================================================== */

/* GPU-only attention: bf16 tensor in, bf16 tensor out.
 * Returns O projection output or NULL on failure. */
static flux_gpu_tensor_t qwen3_attention_gpu(qwen3_model_t *model,
                                              qwen3_layer_t *layer,
                                              flux_gpu_tensor_t norm_out,
                                              int seq_len,
                                              const int *attention_mask) {
    int num_heads = model->num_heads;
    int num_kv_heads = model->num_kv_heads;
    int head_dim = model->head_dim;
    int hidden = model->hidden_size;
    int q_dim = num_heads * head_dim;
    int kv_dim = num_kv_heads * head_dim;
    float scale = 1.0f / sqrtf((float)head_dim);

    /* Q, K, V projections (bf16 → bf16) */
    flux_gpu_tensor_t q = flux_gpu_linear_bf16_native(norm_out, layer->attn.q_proj_weight_bf16,
                                                       seq_len, hidden, q_dim);
    flux_gpu_tensor_t k = flux_gpu_linear_bf16_native(norm_out, layer->attn.k_proj_weight_bf16,
                                                       seq_len, hidden, kv_dim);
    flux_gpu_tensor_t v = flux_gpu_linear_bf16_native(norm_out, layer->attn.v_proj_weight_bf16,
                                                       seq_len, hidden, kv_dim);
    if (!q || !k || !v) goto fail_qkv;

    /* Q/K RMS normalization on GPU */
    if (layer->attn.q_norm_weight_bf16 && layer->attn.k_norm_weight_bf16) {
        flux_gpu_tensor_t q_norm_w = bf16_ptr_to_bf16_tensor(layer->attn.q_norm_weight_bf16, head_dim);
        flux_gpu_tensor_t k_norm_w = bf16_ptr_to_bf16_tensor(layer->attn.k_norm_weight_bf16, head_dim);
        if (!q_norm_w || !k_norm_w) {
            if (q_norm_w) flux_gpu_tensor_free(q_norm_w);
            if (k_norm_w) flux_gpu_tensor_free(k_norm_w);
            goto fail_qkv;
        }
        flux_gpu_head_rms_norm_bf16(q, q_norm_w, seq_len, num_heads, head_dim, QWEN3_RMS_NORM_EPS);
        flux_gpu_head_rms_norm_bf16(k, k_norm_w, seq_len, num_kv_heads, head_dim, QWEN3_RMS_NORM_EPS);
        flux_gpu_tensor_free(q_norm_w);
        flux_gpu_tensor_free(k_norm_w);
    }

    /* RoPE */
    flux_gpu_rope_text_bf16(q, k, model->rope_cos, model->rope_sin,
                             seq_len, num_heads, num_kv_heads, head_dim);

    /* Causal attention */
    flux_gpu_tensor_t attn_out = flux_gpu_tensor_alloc_f16(seq_len * q_dim);
    if (!attn_out) goto fail_qkv;
    if (!flux_gpu_causal_attention_bf16(attn_out, q, k, v, attention_mask,
                                        seq_len, num_heads, num_kv_heads,
                                        head_dim, scale)) {
        flux_gpu_tensor_free(attn_out);
        goto fail_qkv;
    }
    flux_gpu_tensor_free(q);
    flux_gpu_tensor_free(k);
    flux_gpu_tensor_free(v);

    /* O projection */
    flux_gpu_tensor_t out = flux_gpu_linear_bf16_native(attn_out, layer->attn.o_proj_weight_bf16,
                                                         seq_len, q_dim, hidden);
    flux_gpu_tensor_free(attn_out);
    return out;

fail_qkv:
    if (q) flux_gpu_tensor_free(q);
    if (k) flux_gpu_tensor_free(k);
    if (v) flux_gpu_tensor_free(v);
    return NULL;
}

/* GPU-only MLP: bf16 tensor in, bf16 tensor out.
 * Returns down projection output or NULL on failure. */
static flux_gpu_tensor_t qwen3_mlp_gpu(qwen3_model_t *model, qwen3_layer_t *layer,
                                         flux_gpu_tensor_t norm_out,
                                         int seq_len) {
    int hidden = model->hidden_size;
    int intermediate = model->intermediate_size;

    flux_gpu_tensor_t gate = flux_gpu_linear_bf16_native(norm_out, layer->mlp.gate_proj_weight_bf16,
                                                          seq_len, hidden, intermediate);
    flux_gpu_tensor_t up = flux_gpu_linear_bf16_native(norm_out, layer->mlp.up_proj_weight_bf16,
                                                        seq_len, hidden, intermediate);
    if (!gate || !up) {
        if (gate) flux_gpu_tensor_free(gate);
        if (up) flux_gpu_tensor_free(up);
        return NULL;
    }

    flux_gpu_silu_mul_bf16(gate, up, seq_len * intermediate);
    flux_gpu_tensor_free(up);

    flux_gpu_tensor_t out = flux_gpu_linear_bf16_native(gate, layer->mlp.down_proj_weight_bf16,
                                                         seq_len, intermediate, hidden);
    flux_gpu_tensor_free(gate);
    return out;
}

/* Fully GPU-resident forward pass.
 * hidden_state must already be set (from embedding lookup).
 * Fills model->layer_outputs[0..2] with layers 8, 17, 26 outputs.
 * Only processes layers 0..26 (layers 27-35 are unused by output).
 * Returns 1 on success, 0 on failure. */
static int qwen3_forward_gpu(qwen3_model_t *model, int seq_len, const int *attention_mask) {
    int hidden = model->hidden_size;

    /* Upload hidden state to GPU as bf16 */
    flux_gpu_batch_begin();
    flux_gpu_tensor_t hidden_gpu = f32_to_bf16_tensor(model->hidden_state, seq_len * hidden);
    if (!hidden_gpu) {
        flux_gpu_batch_end();
        return 0;
    }

    /* Allocate tensors for saved layer outputs */
    flux_gpu_tensor_t saved[3] = {NULL, NULL, NULL};
    for (int i = 0; i < 3; i++) {
        saved[i] = flux_gpu_tensor_alloc_f16(seq_len * hidden);
        if (!saved[i]) {
            for (int j = 0; j <= i; j++) if (saved[j]) flux_gpu_tensor_free(saved[j]);
            flux_gpu_tensor_free(hidden_gpu);
            flux_gpu_batch_end();
            return 0;
        }
    }

    int ok = 1;

    for (int layer_idx = 0; layer_idx <= QWEN3_OUTPUT_LAYER_3; layer_idx++) {
        qwen3_layer_t *layer = &model->layers[layer_idx];

        /* Load weights on demand (mmap mode) */
        if (model->use_mmap) {
            if (load_layer_weights_small_f32(layer, model->sf_files, model->num_sf_files, layer_idx) != 0) {
                ok = 0; break;
            }
            load_layer_weights_bf16(layer, model->sf_files, model->num_sf_files, layer_idx);
        }

        if (!layer->attn.q_proj_weight_bf16 || !layer->mlp.gate_proj_weight_bf16) {
            ok = 0; break;
        }

        /* Input RMS norm */
        flux_gpu_tensor_t norm_w = f32_to_bf16_tensor(layer->input_layernorm_weight, hidden);
        flux_gpu_tensor_t norm_out = flux_gpu_tensor_alloc_f16(seq_len * hidden);
        if (!norm_w || !norm_out) {
            if (norm_w) flux_gpu_tensor_free(norm_w);
            if (norm_out) flux_gpu_tensor_free(norm_out);
            ok = 0; break;
        }
        flux_gpu_rms_norm_bf16(norm_out, hidden_gpu, norm_w, seq_len, hidden, QWEN3_RMS_NORM_EPS);
        flux_gpu_tensor_free(norm_w);

        /* Attention */
        flux_gpu_tensor_t attn_out = qwen3_attention_gpu(model, layer, norm_out, seq_len, attention_mask);
        flux_gpu_tensor_free(norm_out);
        if (!attn_out) { ok = 0; break; }

        /* Residual: hidden += attn_out */
        flux_gpu_add_bf16(hidden_gpu, hidden_gpu, attn_out, seq_len * hidden);
        flux_gpu_tensor_free(attn_out);

        /* Post-attention RMS norm */
        flux_gpu_tensor_t post_norm_w = f32_to_bf16_tensor(layer->post_attention_layernorm_weight, hidden);
        norm_out = flux_gpu_tensor_alloc_f16(seq_len * hidden);
        if (!post_norm_w || !norm_out) {
            if (post_norm_w) flux_gpu_tensor_free(post_norm_w);
            if (norm_out) flux_gpu_tensor_free(norm_out);
            ok = 0; break;
        }
        flux_gpu_rms_norm_bf16(norm_out, hidden_gpu, post_norm_w, seq_len, hidden, QWEN3_RMS_NORM_EPS);
        flux_gpu_tensor_free(post_norm_w);

        /* MLP */
        flux_gpu_tensor_t mlp_out = qwen3_mlp_gpu(model, layer, norm_out, seq_len);
        flux_gpu_tensor_free(norm_out);
        if (!mlp_out) { ok = 0; break; }

        /* Residual: hidden += mlp_out */
        flux_gpu_add_bf16(hidden_gpu, hidden_gpu, mlp_out, seq_len * hidden);
        flux_gpu_tensor_free(mlp_out);

        /* Free layer weights (mmap mode) */
        if (model->use_mmap) {
            free_layer_weights(layer);
        }

        /* Save output at extraction layers (GPU blit copy) */
        if (layer_idx == QWEN3_OUTPUT_LAYER_1)
            flux_gpu_copy_bf16(saved[0], hidden_gpu, seq_len * hidden);
        else if (layer_idx == QWEN3_OUTPUT_LAYER_2)
            flux_gpu_copy_bf16(saved[1], hidden_gpu, seq_len * hidden);
        else if (layer_idx == QWEN3_OUTPUT_LAYER_3)
            flux_gpu_copy_bf16(saved[2], hidden_gpu, seq_len * hidden);

        if (flux_text_progress_callback)
            flux_text_progress_callback(layer_idx, model->num_layers);
    }

    if (!ok) {
        flux_gpu_batch_end();
        for (int i = 0; i < 3; i++) if (saved[i]) flux_gpu_tensor_free(saved[i]);
        flux_gpu_tensor_free(hidden_gpu);
        return 0;
    }

    /* Convert saved bf16 → f32 on GPU (still within batch) */
    flux_gpu_tensor_t saved_f32[3] = {NULL, NULL, NULL};
    for (int i = 0; i < 3; i++)
        saved_f32[i] = flux_gpu_tensor_bf16_to_f32(saved[i]);

    /* Execute everything in one GPU sync */
    flux_gpu_batch_end();

    /* Signal full completion for progress display */
    if (flux_text_progress_callback)
        flux_text_progress_callback(model->num_layers - 1, model->num_layers);

    /* Read f32 results to CPU */
    int read_ok = 1;
    for (int i = 0; i < 3; i++) {
        if (saved_f32[i]) {
            flux_gpu_tensor_read(saved_f32[i], model->layer_outputs[i]);
            flux_gpu_tensor_free(saved_f32[i]);
        } else {
            read_ok = 0;
        }
        flux_gpu_tensor_free(saved[i]);
    }
    flux_gpu_tensor_free(hidden_gpu);

    return read_ok;
}

#endif /* USE_METAL */

/* ========================================================================
 * Forward Pass
 * ======================================================================== */

float *qwen3_forward(qwen3_model_t *model, const int *input_ids,
                     const int *attention_mask, int seq_len) {
    int hidden = model->hidden_size;
    float *output;

    /* Embedding lookup */
    for (int s = 0; s < seq_len; s++) {
        int token_id = input_ids[s];
        if (token_id >= 0 && token_id < model->vocab_size) {
            memcpy(model->hidden_state + s * hidden,
                   model->embed_tokens + token_id * hidden,
                   hidden * sizeof(float));
        } else {
            /* Unknown token - use zeros */
            memset(model->hidden_state + s * hidden, 0, hidden * sizeof(float));
        }
    }

    /* Run through transformer layers */
#ifdef USE_METAL
    /* Try fully GPU-resident path: 1 sync instead of 72, skips unneeded layers */
    if (model->use_bf16 && flux_metal_available() && seq_len <= 512) {
        if (qwen3_forward_gpu(model, seq_len, attention_mask))
            goto concatenate;
        /* GPU path failed, fall through to mixed CPU/GPU path.
         * hidden_state is unmodified (GPU worked on a copy). */
    }

    /* Start batch mode to reduce GPU sync overhead between layers */
    int batch_mode = model->use_bf16 && flux_metal_available();
    if (batch_mode) {
        flux_gpu_batch_begin();
    }
#endif

    for (int layer_idx = 0; layer_idx < model->num_layers; layer_idx++) {
        /* In mmap mode, load layer weights on-demand */
        if (model->use_mmap) {
#ifdef USE_METAL
            if (model->use_bf16) {
                /* Load only small f32 weights (layer norms) + bf16 projection weights */
                if (load_layer_weights_small_f32(&model->layers[layer_idx], model->sf_files, model->num_sf_files, layer_idx) != 0) {
                    fprintf(stderr, "Failed to load layer %d small weights\n", layer_idx);
#ifdef USE_METAL
                    if (batch_mode) flux_gpu_batch_end();
#endif
                    return NULL;
                }
                load_layer_weights_bf16(&model->layers[layer_idx], model->sf_files, model->num_sf_files, layer_idx);
            } else
#endif
            {
                if (load_layer_weights(&model->layers[layer_idx], model->sf_files, model->num_sf_files, layer_idx) != 0) {
                    fprintf(stderr, "Failed to load layer %d weights\n", layer_idx);
                    return NULL;
                }
            }
        }

#ifdef USE_METAL
        if (model->use_bf16 && flux_metal_available()) {
            qwen3_layer_forward_bf16(model, &model->layers[layer_idx], seq_len, attention_mask);
        } else
#endif
        {
            qwen3_layer_forward(model, &model->layers[layer_idx], seq_len, attention_mask);
        }

        /* In mmap mode, free layer weights after use */
        if (model->use_mmap) {
            free_layer_weights(&model->layers[layer_idx]);
        }

        /* Save output at extraction layers (9, 18, 27) */
        if (layer_idx == QWEN3_OUTPUT_LAYER_1) {
            memcpy(model->layer_outputs[0], model->hidden_state, seq_len * hidden * sizeof(float));
        } else if (layer_idx == QWEN3_OUTPUT_LAYER_2) {
            memcpy(model->layer_outputs[1], model->hidden_state, seq_len * hidden * sizeof(float));
        } else if (layer_idx == QWEN3_OUTPUT_LAYER_3) {
            memcpy(model->layer_outputs[2], model->hidden_state, seq_len * hidden * sizeof(float));
        }

        /* Progress callback */
        if (flux_text_progress_callback)
            flux_text_progress_callback(layer_idx, model->num_layers);
    }

#ifdef USE_METAL
    /* End batch mode */
    if (batch_mode) {
        flux_gpu_batch_end();
    }
#endif

    /* Concatenate outputs from layers 8, 17, 26 -> [seq_len, 7680] */
#ifdef USE_METAL
concatenate: (void)0; /* label needs a statement; can't precede a declaration in C */
#endif
    int text_dim = model->text_dim;
    output = malloc(seq_len * text_dim * sizeof(float));
    if (!output) return NULL;

    for (int s = 0; s < seq_len; s++) {
        memcpy(output + s * text_dim,
               model->layer_outputs[0] + s * hidden,
               hidden * sizeof(float));
        memcpy(output + s * text_dim + hidden,
               model->layer_outputs[1] + s * hidden,
               hidden * sizeof(float));
        memcpy(output + s * text_dim + 2 * hidden,
               model->layer_outputs[2] + s * hidden,
               hidden * sizeof(float));
    }

    return output;
}

/* ========================================================================
 * Model Loading
 * ======================================================================== */

/* Helper to load a tensor from safetensors files */
static float *load_tensor(safetensors_file_t **files, int num_files, const char *name) {
    for (int f = 0; f < num_files; f++) {
        const safetensor_t *t = safetensors_find(files[f], name);
        if (t) {
            return safetensors_get_f32(files[f], t);
        }
    }
    fprintf(stderr, "Error: required tensor not found: %s\n", name);
    return NULL;
}

#ifdef USE_METAL
/* Helper to load bf16 tensor directly (zero-copy from mmap region) */
static uint16_t *load_tensor_bf16(safetensors_file_t **files, int num_files, const char *name) {
    for (int f = 0; f < num_files; f++) {
        const safetensor_t *t = safetensors_find(files[f], name);
        if (t && safetensor_is_bf16(t)) {
            return safetensors_get_bf16_direct(files[f], t);
        }
    }
    return NULL;  /* Not found or not bf16 - fall back to f32 */
}

/* Load only small f32 weights for bf16 path (layer norms and q/k norms) */
static int load_layer_weights_small_f32(qwen3_layer_t *layer, safetensors_file_t **files,
                                        int num_files, int layer_idx) {
    char name[256];

    /* Input layernorm */
    snprintf(name, sizeof(name), "model.layers.%d.input_layernorm.weight", layer_idx);
    layer->input_layernorm_weight = load_tensor(files, num_files, name);

    /* Post-attention layernorm */
    snprintf(name, sizeof(name), "model.layers.%d.post_attention_layernorm.weight", layer_idx);
    layer->post_attention_layernorm_weight = load_tensor(files, num_files, name);

    /* Q/K norm */
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_norm.weight", layer_idx);
    layer->attn.q_norm_weight = load_tensor(files, num_files, name);

    snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_norm.weight", layer_idx);
    layer->attn.k_norm_weight = load_tensor(files, num_files, name);

    return (layer->input_layernorm_weight && layer->post_attention_layernorm_weight &&
            layer->attn.q_norm_weight && layer->attn.k_norm_weight) ? 0 : -1;
}
#endif

static int load_layer_weights(qwen3_layer_t *layer, safetensors_file_t **files,
                              int num_files, int layer_idx) {
    char name[256];

    /* Input layernorm */
    snprintf(name, sizeof(name), "model.layers.%d.input_layernorm.weight", layer_idx);
    layer->input_layernorm_weight = load_tensor(files, num_files, name);

    /* Post-attention layernorm */
    snprintf(name, sizeof(name), "model.layers.%d.post_attention_layernorm.weight", layer_idx);
    layer->post_attention_layernorm_weight = load_tensor(files, num_files, name);

    /* Attention weights */
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.weight", layer_idx);
    layer->attn.q_proj_weight = load_tensor(files, num_files, name);

    snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_proj.weight", layer_idx);
    layer->attn.k_proj_weight = load_tensor(files, num_files, name);

    snprintf(name, sizeof(name), "model.layers.%d.self_attn.v_proj.weight", layer_idx);
    layer->attn.v_proj_weight = load_tensor(files, num_files, name);

    snprintf(name, sizeof(name), "model.layers.%d.self_attn.o_proj.weight", layer_idx);
    layer->attn.o_proj_weight = load_tensor(files, num_files, name);

    /* Q/K norm */
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_norm.weight", layer_idx);
    layer->attn.q_norm_weight = load_tensor(files, num_files, name);

    snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_norm.weight", layer_idx);
    layer->attn.k_norm_weight = load_tensor(files, num_files, name);

    /* MLP weights */
    snprintf(name, sizeof(name), "model.layers.%d.mlp.gate_proj.weight", layer_idx);
    layer->mlp.gate_proj_weight = load_tensor(files, num_files, name);

    snprintf(name, sizeof(name), "model.layers.%d.mlp.up_proj.weight", layer_idx);
    layer->mlp.up_proj_weight = load_tensor(files, num_files, name);

    snprintf(name, sizeof(name), "model.layers.%d.mlp.down_proj.weight", layer_idx);
    layer->mlp.down_proj_weight = load_tensor(files, num_files, name);

    /* Check that all required tensors were loaded */
    if (!layer->input_layernorm_weight || !layer->post_attention_layernorm_weight ||
        !layer->attn.q_proj_weight || !layer->attn.k_proj_weight ||
        !layer->attn.v_proj_weight || !layer->attn.o_proj_weight ||
        !layer->attn.q_norm_weight || !layer->attn.k_norm_weight ||
        !layer->mlp.gate_proj_weight || !layer->mlp.up_proj_weight ||
        !layer->mlp.down_proj_weight) {
        return -1;
    }

    return 0;
}

#ifdef USE_METAL
/* Load bf16 weights for a layer (GPU acceleration path).
 * Returns 1 if all bf16 weights loaded successfully, 0 otherwise.
 * bf16 pointers are direct into mmap region - do NOT free them. */
static int load_layer_weights_bf16(qwen3_layer_t *layer, safetensors_file_t **files,
                                    int num_files, int layer_idx) {
    char name[256];

    /* Attention weights */
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.weight", layer_idx);
    layer->attn.q_proj_weight_bf16 = load_tensor_bf16(files, num_files, name);

    snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_proj.weight", layer_idx);
    layer->attn.k_proj_weight_bf16 = load_tensor_bf16(files, num_files, name);

    snprintf(name, sizeof(name), "model.layers.%d.self_attn.v_proj.weight", layer_idx);
    layer->attn.v_proj_weight_bf16 = load_tensor_bf16(files, num_files, name);

    snprintf(name, sizeof(name), "model.layers.%d.self_attn.o_proj.weight", layer_idx);
    layer->attn.o_proj_weight_bf16 = load_tensor_bf16(files, num_files, name);

    /* Q/K norm weights (small but needed for GPU path) */
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_norm.weight", layer_idx);
    layer->attn.q_norm_weight_bf16 = load_tensor_bf16(files, num_files, name);

    snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_norm.weight", layer_idx);
    layer->attn.k_norm_weight_bf16 = load_tensor_bf16(files, num_files, name);

    /* MLP weights */
    snprintf(name, sizeof(name), "model.layers.%d.mlp.gate_proj.weight", layer_idx);
    layer->mlp.gate_proj_weight_bf16 = load_tensor_bf16(files, num_files, name);

    snprintf(name, sizeof(name), "model.layers.%d.mlp.up_proj.weight", layer_idx);
    layer->mlp.up_proj_weight_bf16 = load_tensor_bf16(files, num_files, name);

    snprintf(name, sizeof(name), "model.layers.%d.mlp.down_proj.weight", layer_idx);
    layer->mlp.down_proj_weight_bf16 = load_tensor_bf16(files, num_files, name);

    /* Check if all large weights loaded as bf16 */
    return (layer->attn.q_proj_weight_bf16 && layer->attn.k_proj_weight_bf16 &&
            layer->attn.v_proj_weight_bf16 && layer->attn.o_proj_weight_bf16 &&
            layer->mlp.gate_proj_weight_bf16 && layer->mlp.up_proj_weight_bf16 &&
            layer->mlp.down_proj_weight_bf16);
}
#endif

/* Free a single layer's weights (used in mmap streaming mode) */
static void free_layer_weights(qwen3_layer_t *layer) {
    free(layer->input_layernorm_weight);
    free(layer->post_attention_layernorm_weight);
    free(layer->attn.q_proj_weight);
    free(layer->attn.k_proj_weight);
    free(layer->attn.v_proj_weight);
    free(layer->attn.o_proj_weight);
    free(layer->attn.q_norm_weight);
    free(layer->attn.k_norm_weight);
    free(layer->mlp.gate_proj_weight);
    free(layer->mlp.up_proj_weight);
    free(layer->mlp.down_proj_weight);
    /* Note: bf16 pointers are direct to mmap region, don't free them */
    memset(layer, 0, sizeof(*layer));
}

/* ========================================================================
 * Qwen3 Config Parsing and Dynamic Shard Loading
 * ======================================================================== */

/* Parse text_encoder/config.json to get architecture dimensions.
 * Sets model fields. Returns 0 on success, -1 on failure. */
static int parse_qwen3_config(const char *model_dir, qwen3_model_t *model) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/config.json", model_dir);

    FILE *f = fopen(path, "r");
    if (!f) return -1;

    char buf[8192];
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    buf[n] = '\0';
    fclose(f);

    char *p;
    int hidden = 0, intermediate = 0, num_heads = 0, num_kv_heads = 0;
    int head_dim = 0, vocab_size = 0, num_layers = 0;
    float rope_theta = 0;

    if ((p = strstr(buf, "\"hidden_size\""))) {
        if ((p = strchr(p, ':'))) hidden = atoi(p + 1);
    }
    if ((p = strstr(buf, "\"intermediate_size\""))) {
        if ((p = strchr(p, ':'))) intermediate = atoi(p + 1);
    }
    if ((p = strstr(buf, "\"num_attention_heads\""))) {
        if ((p = strchr(p, ':'))) num_heads = atoi(p + 1);
    }
    if ((p = strstr(buf, "\"num_key_value_heads\""))) {
        if ((p = strchr(p, ':'))) num_kv_heads = atoi(p + 1);
    }
    if ((p = strstr(buf, "\"head_dim\""))) {
        if ((p = strchr(p, ':'))) head_dim = atoi(p + 1);
    }
    if ((p = strstr(buf, "\"vocab_size\""))) {
        if ((p = strchr(p, ':'))) vocab_size = atoi(p + 1);
    }
    if ((p = strstr(buf, "\"num_hidden_layers\""))) {
        if ((p = strchr(p, ':'))) num_layers = atoi(p + 1);
    }
    if ((p = strstr(buf, "\"rope_theta\""))) {
        if ((p = strchr(p, ':'))) rope_theta = atof(p + 1);
    }

    if (hidden <= 0 || num_heads <= 0) return -1;

    model->hidden_size = hidden;
    model->intermediate_size = intermediate > 0 ? intermediate : 9728;
    model->num_heads = num_heads;
    model->num_kv_heads = num_kv_heads > 0 ? num_kv_heads : 8;
    model->head_dim = head_dim > 0 ? head_dim : 128;
    model->vocab_size = vocab_size > 0 ? vocab_size : QWEN3_VOCAB_SIZE;
    model->num_layers = num_layers > 0 ? num_layers : 36;
    model->rope_theta = rope_theta > 0 ? rope_theta : QWEN3_ROPE_THETA;
    model->text_dim = 3 * hidden;

    return 0;
}

/* Set default Qwen3-4B architecture values. */
static void qwen3_set_defaults(qwen3_model_t *model) {
    model->hidden_size = 2560;
    model->intermediate_size = 9728;
    model->num_heads = 32;
    model->num_kv_heads = 8;
    model->head_dim = 128;
    model->vocab_size = QWEN3_VOCAB_SIZE;
    model->num_layers = 36;
    model->rope_theta = QWEN3_ROPE_THETA;
    model->text_dim = 3 * 2560;
}

/* Open all safetensors shards dynamically by reading model.safetensors.index.json.
 * Returns number of files opened, 0 on failure. */
static int open_safetensors_shards(const char *model_dir,
                                    safetensors_file_t **files, int max_files) {
    char path[1024];

    /* First try: read the index JSON to discover shard filenames */
    snprintf(path, sizeof(path), "%s/model.safetensors.index.json", model_dir);
    FILE *f = fopen(path, "r");
    if (f) {
        /* Read the whole index file */
        fseek(f, 0, SEEK_END);
        long fsize = ftell(f);
        fseek(f, 0, SEEK_SET);
        char *buf = malloc(fsize + 1);
        if (!buf) { fclose(f); return 0; }
        fread(buf, 1, fsize, f);
        buf[fsize] = '\0';
        fclose(f);

        /* Collect unique shard filenames from weight_map values.
         * They look like: "model-00001-of-00003.safetensors" */
        char shard_names[QWEN3_MAX_SHARDS][128];
        int num_shards = 0;

        char *p = buf;
        while ((p = strstr(p, "model-")) != NULL) {
            /* Find the end of the filename */
            char *end = strstr(p, ".safetensors");
            if (!end) { p++; continue; }
            end += strlen(".safetensors");

            int len = (int)(end - p);
            if (len >= 128) { p = end; continue; }

            /* Check if we already have this shard */
            char name[128];
            memcpy(name, p, len);
            name[len] = '\0';

            int found = 0;
            for (int i = 0; i < num_shards; i++) {
                if (strcmp(shard_names[i], name) == 0) {
                    found = 1;
                    break;
                }
            }
            if (!found && num_shards < max_files && num_shards < QWEN3_MAX_SHARDS) {
                strcpy(shard_names[num_shards], name);
                num_shards++;
            }
            p = end;
        }
        free(buf);

        if (num_shards > 0) {
            for (int i = 0; i < num_shards; i++) {
                snprintf(path, sizeof(path), "%s/%s", model_dir, shard_names[i]);
                files[i] = safetensors_open(path);
                if (!files[i]) {
                    fprintf(stderr, "qwen3: failed to open shard %s\n", path);
                    for (int j = 0; j < i; j++) safetensors_close(files[j]);
                    return 0;
                }
            }
            return num_shards;
        }
    }

    /* Fallback: try model-00001-of-00002.safetensors pattern */
    char path1[512], path2[512];
    snprintf(path1, sizeof(path1), "%s/model-00001-of-00002.safetensors", model_dir);
    snprintf(path2, sizeof(path2), "%s/model-00002-of-00002.safetensors", model_dir);

    files[0] = safetensors_open(path1);
    files[1] = safetensors_open(path2);
    if (files[0] && files[1]) return 2;

    if (files[0]) safetensors_close(files[0]);
    if (files[1]) safetensors_close(files[1]);
    return 0;
}

/* Allocate working memory for Qwen3 model based on architecture fields. */
static void qwen3_alloc_work_buffers(qwen3_model_t *model) {
    int seq_len = QWEN3_MAX_SEQ_LEN;
    int hidden = model->hidden_size;
    int num_heads = model->num_heads;
    int num_kv_heads = model->num_kv_heads;
    int head_dim = model->head_dim;
    int intermediate = model->intermediate_size;

    model->hidden_state = malloc(seq_len * hidden * sizeof(float));
    model->residual = malloc(seq_len * hidden * sizeof(float));
    model->q_buf = malloc(seq_len * num_heads * head_dim * sizeof(float));
    model->k_buf = malloc(seq_len * num_kv_heads * head_dim * sizeof(float));
    model->v_buf = malloc(seq_len * num_kv_heads * head_dim * sizeof(float));
    model->attn_scores = malloc(num_heads * seq_len * seq_len * sizeof(float));
    model->attn_out = malloc(seq_len * num_heads * head_dim * sizeof(float));
    model->mlp_gate = malloc(seq_len * intermediate * sizeof(float));
    model->mlp_up = malloc(seq_len * intermediate * sizeof(float));
    model->mlp_out = malloc(seq_len * hidden * sizeof(float));
    model->norm_buf = malloc(seq_len * hidden * sizeof(float));

    model->attn_q_head = malloc(seq_len * head_dim * sizeof(float));
    model->attn_v_head = malloc(seq_len * head_dim * sizeof(float));
    model->attn_out_head = malloc(seq_len * head_dim * sizeof(float));
    model->attn_q_hsd = malloc(seq_len * num_heads * head_dim * sizeof(float));
    model->attn_k_hsd = malloc(seq_len * num_heads * head_dim * sizeof(float));
    model->attn_v_hsd = malloc(seq_len * num_heads * head_dim * sizeof(float));
    model->attn_out_hsd = malloc(seq_len * num_heads * head_dim * sizeof(float));

    for (int i = 0; i < 3; i++) {
        model->layer_outputs[i] = malloc(seq_len * hidden * sizeof(float));
    }
}

qwen3_model_t *qwen3_model_load(const char *model_dir) {
    qwen3_model_t *model = calloc(1, sizeof(qwen3_model_t));
    if (!model) return NULL;

    /* Parse config, fall back to defaults */
    if (parse_qwen3_config(model_dir, model) != 0) {
        qwen3_set_defaults(model);
    }

    model->layers = calloc(model->num_layers, sizeof(qwen3_layer_t));
    if (!model->layers) {
        free(model);
        return NULL;
    }

    /* Open safetensors shards dynamically */
    safetensors_file_t *files[QWEN3_MAX_SHARDS];
    int num_files = open_safetensors_shards(model_dir, files, QWEN3_MAX_SHARDS);
    if (num_files == 0) {
        fprintf(stderr, "qwen3_model_load: failed to open safetensors files\n");
        free(model->layers);
        free(model);
        return NULL;
    }

    /* Load embedding weights */
    model->embed_tokens = load_tensor(files, num_files, "model.embed_tokens.weight");
    if (!model->embed_tokens) {
        fprintf(stderr, "qwen3_model_load: failed to load embed_tokens\n");
        goto error;
    }

    /* Load layer weights */
    for (int i = 0; i < model->num_layers; i++) {
        if (load_layer_weights(&model->layers[i], files, num_files, i) != 0) {
            fprintf(stderr, "qwen3_model_load: failed to load layer %d\n", i);
            goto error;
        }
    }

    /* Load final norm */
    model->norm_weight = load_tensor(files, num_files, "model.norm.weight");
    if (!model->norm_weight) {
        fprintf(stderr, "qwen3_model_load: failed to load final norm\n");
        goto error;
    }

    for (int i = 0; i < num_files; i++) safetensors_close(files[i]);

    /* Compute RoPE frequencies */
    int max_seq = QWEN3_MAX_SEQ_LEN;
    int half_dim = model->head_dim / 2;
    model->rope_cos = malloc(max_seq * half_dim * sizeof(float));
    model->rope_sin = malloc(max_seq * half_dim * sizeof(float));
    compute_rope_freqs(model->rope_cos, model->rope_sin, max_seq,
                       model->head_dim, model->rope_theta);

    /* Allocate working memory */
    qwen3_alloc_work_buffers(model);

    return model;

error:
    for (int i = 0; i < num_files; i++) safetensors_close(files[i]);
    qwen3_model_free(model);
    return NULL;
}

/* Load model in mmap mode - keeps safetensors files open and loads layer weights
 * on-demand during forward pass. Reduces peak memory from ~16GB to ~2GB. */
qwen3_model_t *qwen3_model_load_mmap(const char *model_dir) {
    qwen3_model_t *model = calloc(1, sizeof(qwen3_model_t));
    if (!model) return NULL;

    model->use_mmap = 1;

    /* Parse config, fall back to defaults */
    if (parse_qwen3_config(model_dir, model) != 0) {
        qwen3_set_defaults(model);
    }

#ifdef USE_METAL
    /* Enable bf16 GPU acceleration when Metal is available.
     * Set FLUX_QWEN3_NO_BF16=1 to disable for debugging. */
    model->use_bf16 = (flux_metal_available() && !getenv("FLUX_QWEN3_NO_BF16")) ? 1 : 0;
    if (model->use_bf16) {
        if (flux_verbose)
            fprintf(stderr, "Qwen3: bf16 GPU acceleration enabled\n");
    }
#endif
    model->layers = calloc(model->num_layers, sizeof(qwen3_layer_t));
    if (!model->layers) {
        free(model);
        return NULL;
    }

    /* Open safetensors shards dynamically and keep them open */
    model->num_sf_files = open_safetensors_shards(model_dir, model->sf_files, QWEN3_MAX_SHARDS);
    if (model->num_sf_files == 0) {
        fprintf(stderr, "qwen3_model_load_mmap: failed to open safetensors files\n");
        goto error;
    }

    /* Load only embeddings - needed for all tokens */
    model->embed_tokens = load_tensor(model->sf_files, model->num_sf_files,
                                       "model.embed_tokens.weight");
    if (!model->embed_tokens) {
        fprintf(stderr, "qwen3_model_load_mmap: failed to load embed_tokens\n");
        goto error;
    }

    /* Load final norm (small) */
    model->norm_weight = load_tensor(model->sf_files, model->num_sf_files,
                                      "model.norm.weight");
    if (!model->norm_weight) {
        fprintf(stderr, "qwen3_model_load_mmap: failed to load final norm\n");
        goto error;
    }

    /* DON'T load layer weights - they'll be loaded on-demand in forward pass */
    if (flux_verbose)
        fprintf(stderr, "Mmap mode: layer weights will be loaded on-demand\n");

    /* Compute RoPE frequencies */
    int max_seq = QWEN3_MAX_SEQ_LEN;
    int half_dim = model->head_dim / 2;
    model->rope_cos = malloc(max_seq * half_dim * sizeof(float));
    model->rope_sin = malloc(max_seq * half_dim * sizeof(float));
    compute_rope_freqs(model->rope_cos, model->rope_sin, max_seq,
                       model->head_dim, model->rope_theta);

    /* Allocate working memory */
    qwen3_alloc_work_buffers(model);

    return model;

error:
    qwen3_model_free(model);
    return NULL;
}

void qwen3_model_free(qwen3_model_t *model) {
    if (!model) return;

    free(model->embed_tokens);
    free(model->norm_weight);
    free(model->rope_cos);
    free(model->rope_sin);

    if (model->layers) {
        for (int i = 0; i < model->num_layers; i++) {
            qwen3_layer_t *layer = &model->layers[i];
            free(layer->input_layernorm_weight);
            free(layer->post_attention_layernorm_weight);
            free(layer->attn.q_proj_weight);
            free(layer->attn.k_proj_weight);
            free(layer->attn.v_proj_weight);
            free(layer->attn.o_proj_weight);
            free(layer->attn.q_norm_weight);
            free(layer->attn.k_norm_weight);
            free(layer->mlp.gate_proj_weight);
            free(layer->mlp.up_proj_weight);
            free(layer->mlp.down_proj_weight);
        }
        free(model->layers);
    }

    free(model->hidden_state);
    free(model->residual);
    free(model->q_buf);
    free(model->k_buf);
    free(model->v_buf);
    free(model->attn_scores);
    free(model->attn_out);
    free(model->mlp_gate);
    free(model->mlp_up);
    free(model->mlp_out);
    free(model->norm_buf);

    /* Free attention work buffers */
    free(model->attn_q_head);
    free(model->attn_v_head);
    free(model->attn_out_head);
    free(model->attn_q_hsd);
    free(model->attn_k_hsd);
    free(model->attn_v_hsd);
    free(model->attn_out_hsd);

    for (int i = 0; i < 3; i++) {
        free(model->layer_outputs[i]);
    }

    /* Close mmap'd safetensors files if open */
    for (int i = 0; i < model->num_sf_files; i++) {
        if (model->sf_files[i]) safetensors_close(model->sf_files[i]);
    }

    free(model);
}

/* ========================================================================
 * Combined Encoder API
 * ======================================================================== */

qwen3_encoder_t *qwen3_encoder_load(const char *model_dir, int use_mmap) {
    qwen3_encoder_t *enc = calloc(1, sizeof(qwen3_encoder_t));
    if (!enc) return NULL;

    /* Load tokenizer */
    char tok_path[512];
    snprintf(tok_path, sizeof(tok_path), "%s/tokenizer/tokenizer.json", model_dir);
    enc->tokenizer = qwen3_tokenizer_load(tok_path);
    if (!enc->tokenizer) {
        fprintf(stderr, "qwen3_encoder_load: failed to load tokenizer\n");
        free(enc);
        return NULL;
    }

    /* Load model - use mmap mode if requested (saves ~14GB RAM) */
    char model_path[512];
    snprintf(model_path, sizeof(model_path), "%s/text_encoder", model_dir);
    if (use_mmap) {
        enc->model = qwen3_model_load_mmap(model_path);
    } else {
        enc->model = qwen3_model_load(model_path);
    }
    if (!enc->model) {
        fprintf(stderr, "qwen3_encoder_load: failed to load model\n");
        qwen3_tokenizer_free(enc->tokenizer);
        free(enc);
        return NULL;
    }

    return enc;
}

void qwen3_encoder_free(qwen3_encoder_t *enc) {
    if (!enc) return;
    qwen3_tokenizer_free(enc->tokenizer);
    qwen3_model_free(enc->model);
    free(enc);
}

float *qwen3_encode_text(qwen3_encoder_t *enc, const char *prompt) {
    if (!enc || !enc->tokenizer || !enc->model || !prompt) return NULL;

    /* Tokenize with chat template */
    int num_tokens;
    int *tokens = qwen3_tokenize_chat(enc->tokenizer, prompt, &num_tokens, QWEN3_MAX_SEQ_LEN);
    if (!tokens) return NULL;

    /* Pad to max length */
    int *attention_mask = malloc(QWEN3_MAX_SEQ_LEN * sizeof(int));
    int *padded_tokens = qwen3_pad_tokens(tokens, num_tokens, QWEN3_MAX_SEQ_LEN, attention_mask);
    free(tokens);

    if (!padded_tokens) {
        free(attention_mask);
        return NULL;
    }

    /* Forward pass */
    float *embeddings = qwen3_forward(enc->model, padded_tokens, attention_mask, QWEN3_MAX_SEQ_LEN);

    free(padded_tokens);
    free(attention_mask);

    return embeddings;
}
