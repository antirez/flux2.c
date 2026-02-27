/*
 * Mistral-Small-3.2-24B Text Encoder Implementation
 *
 * Implements Mistral encoder for FLUX.2-dev text encoding.
 * Architecture nearly identical to Qwen3 (GQA, SwiGLU, RoPE, RMSNorm).
 *
 * Key differences from Qwen3:
 * - No Q/K per-head RMS normalization
 * - Different extraction layers (10, 20, 30)
 * - Different RMS norm epsilon (1e-5 vs 1e-6)
 * - Always mmap mode (model is ~48GB)
 * - Pooled embedding output (last real token hidden state)
 *
 * Forked from iris_qwen3.c.
 */

#include "iris_mistral.h"
#include "iris_safetensors.h"
#include "iris_kernels.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef USE_BLAS
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#endif

#ifdef USE_METAL
#include "iris_metal.h"
#endif

/* GPU acceleration threshold (same as Qwen3) */
#define MISTRAL_MIN_GPU_ELEMENTS (10 * 1024 * 1024)

/* Maximum safetensors shards */
#define MISTRAL_MAX_SHARDS 32

/* ========================================================================
 * Data Structures
 * ======================================================================== */

typedef struct {
    float *q_proj_weight;      /* [num_heads * head_dim, hidden] */
    float *k_proj_weight;      /* [num_kv_heads * head_dim, hidden] */
    float *v_proj_weight;      /* [num_kv_heads * head_dim, hidden] */
    float *o_proj_weight;      /* [hidden, num_heads * head_dim] */
    /* NO q_norm / k_norm — Mistral doesn't use per-head QK normalization */
    uint16_t *q_proj_weight_bf16;
    uint16_t *k_proj_weight_bf16;
    uint16_t *v_proj_weight_bf16;
    uint16_t *o_proj_weight_bf16;
} mistral_attention_t;

typedef struct {
    float *gate_proj_weight;   /* [intermediate, hidden] */
    float *up_proj_weight;     /* [intermediate, hidden] */
    float *down_proj_weight;   /* [hidden, intermediate] */
    uint16_t *gate_proj_weight_bf16;
    uint16_t *up_proj_weight_bf16;
    uint16_t *down_proj_weight_bf16;
} mistral_mlp_t;

typedef struct {
    float *input_layernorm_weight;
    float *post_attention_layernorm_weight;
    mistral_attention_t attn;
    mistral_mlp_t mlp;
} mistral_layer_t;

struct mistral_model {
    /* Architecture (from config.json) */
    int hidden_size;
    int intermediate_size;
    int num_heads;
    int num_kv_heads;
    int head_dim;
    int vocab_size;
    float rope_theta;
    int text_dim;          /* 3 * hidden_size */
    int num_layers;

    /* Embedding layer */
    float *embed_tokens;   /* [vocab_size, hidden] */

    /* Transformer layers */
    mistral_layer_t *layers;

    /* Final layer norm */
    float *norm_weight;    /* [hidden] */

    /* RoPE precomputed */
    float *rope_cos;       /* [max_seq_len, head_dim/2] */
    float *rope_sin;       /* [max_seq_len, head_dim/2] */

    /* Working memory */
    float *hidden_state;
    float *residual;
    float *q_buf, *k_buf, *v_buf;
    float *attn_scores;
    float *attn_out;
    float *mlp_gate, *mlp_up, *mlp_out;
    float *norm_buf;
    float *attn_q_head, *attn_v_head, *attn_out_head;

    /* Output layer storage */
    float *layer_outputs[3];

    /* Pooled embedding (last real token hidden state) */
    float *pooled_output;
    int pooled_seq_idx;    /* Index of last real token */

    /* Mmap mode */
    int use_mmap;
    safetensors_file_t *sf_files[MISTRAL_MAX_SHARDS];
    int num_sf_files;

    /* BF16 GPU acceleration */
    int use_bf16;
};

/* Forward declarations for mmap streaming */
static int mistral_load_layer_weights(mistral_layer_t *layer, safetensors_file_t **files,
                                       int num_files, int layer_idx);
#ifdef USE_METAL
static int mistral_load_layer_small_f32(mistral_layer_t *layer, safetensors_file_t **files,
                                         int num_files, int layer_idx);
static int mistral_load_layer_bf16(mistral_layer_t *layer, safetensors_file_t **files,
                                    int num_files, int layer_idx);
#endif
static void mistral_free_layer_weights(mistral_layer_t *layer);

/* ========================================================================
 * Basic Operations
 * ======================================================================== */

static void mistral_linear(float *y, const float *x, const float *W,
                            int seq_len, int in_dim, int out_dim) {
#ifdef USE_METAL
    size_t matrix_elements = (size_t)seq_len * out_dim;
    if (iris_metal_available() && matrix_elements >= MISTRAL_MIN_GPU_ELEMENTS) {
        iris_metal_sgemm_cached(0, 1, seq_len, out_dim, in_dim,
                                1.0f, x, in_dim, W, in_dim,
                                0.0f, y, out_dim);
        return;
    }
#endif
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                seq_len, out_dim, in_dim,
                1.0f, x, in_dim, W, in_dim,
                0.0f, y, out_dim);
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

static void mistral_rms_norm(float *out, const float *x, const float *weight,
                              int seq_len, int hidden, float eps) {
    for (int s = 0; s < seq_len; s++) {
        const float *x_row = x + s * hidden;
        float *out_row = out + s * hidden;

        float sum_sq = 0.0f;
        for (int i = 0; i < hidden; i++) {
            sum_sq += x_row[i] * x_row[i];
        }
        float rms = sqrtf(sum_sq / hidden + eps);
        float rms_inv = 1.0f / rms;

        for (int i = 0; i < hidden; i++) {
            out_row[i] = x_row[i] * rms_inv * weight[i];
        }
    }
}

static void mistral_softmax(float *x, int len) {
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
 * RoPE
 * ======================================================================== */

static void mistral_compute_rope_freqs(float *cos_out, float *sin_out,
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

static void mistral_apply_rope(float *q, float *k, const float *cos_cache, const float *sin_cache,
                                int seq_len, int num_q_heads, int num_kv_heads, int head_dim) {
    int half_dim = head_dim / 2;

    for (int s = 0; s < seq_len; s++) {
        const float *cos_row = cos_cache + s * half_dim;
        const float *sin_row = sin_cache + s * half_dim;

        for (int h = 0; h < num_q_heads; h++) {
            float *q_head = q + s * num_q_heads * head_dim + h * head_dim;
            for (int i = 0; i < half_dim; i++) {
                float x0 = q_head[i];
                float x1 = q_head[i + half_dim];
                q_head[i] = x0 * cos_row[i] - x1 * sin_row[i];
                q_head[i + half_dim] = x0 * sin_row[i] + x1 * cos_row[i];
            }
        }

        for (int h = 0; h < num_kv_heads; h++) {
            float *k_head = k + s * num_kv_heads * head_dim + h * head_dim;
            for (int i = 0; i < half_dim; i++) {
                float x0 = k_head[i];
                float x1 = k_head[i + half_dim];
                k_head[i] = x0 * cos_row[i] - x1 * sin_row[i];
                k_head[i + half_dim] = x0 * sin_row[i] + x1 * cos_row[i];
            }
        }
    }
}

/* ========================================================================
 * Attention (CPU Path)
 * ======================================================================== */

static void mistral_attention_forward(mistral_model_t *model, mistral_layer_t *layer,
                                       int seq_len, const int *attention_mask) {
    int num_heads = model->num_heads;
    int num_kv_heads = model->num_kv_heads;
    int head_dim = model->head_dim;
    int hidden = model->hidden_size;
    int kv_dim = num_kv_heads * head_dim;
    int q_dim = num_heads * head_dim;
    float scale = 1.0f / sqrtf((float)head_dim);

    /* Q, K, V projections */
    mistral_linear(model->q_buf, model->norm_buf, layer->attn.q_proj_weight,
                   seq_len, hidden, q_dim);
    mistral_linear(model->k_buf, model->norm_buf, layer->attn.k_proj_weight,
                   seq_len, hidden, kv_dim);
    mistral_linear(model->v_buf, model->norm_buf, layer->attn.v_proj_weight,
                   seq_len, hidden, kv_dim);

    /* NO Q/K RMS norm — Mistral skips this step */

    /* Apply RoPE */
    mistral_apply_rope(model->q_buf, model->k_buf, model->rope_cos, model->rope_sin,
                       seq_len, num_heads, num_kv_heads, head_dim);

#ifdef USE_METAL
    if (iris_metal_available()) {
        if (iris_metal_causal_attention(model->attn_out,
                                         model->q_buf, model->k_buf, model->v_buf,
                                         attention_mask,
                                         seq_len, num_heads, num_kv_heads,
                                         head_dim, scale)) {
            goto output_proj;
        }
    }
#endif

    /* CPU fallback: GQA attention */
    {
        int heads_per_kv = num_heads / num_kv_heads;

        for (int h = 0; h < num_heads; h++) {
            int kv_h = h / heads_per_kv;
            float *scores = model->attn_scores + h * seq_len * seq_len;

            const float *q_strided = model->q_buf + h * head_dim;
            const float *k_strided = model->k_buf + kv_h * head_dim;

#ifdef USE_BLAS
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        seq_len, seq_len, head_dim,
                        scale, q_strided, q_dim, k_strided, kv_dim,
                        0.0f, scores, seq_len);
#else
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

            /* Causal mask + padding mask + softmax */
            for (int i = 0; i < seq_len; i++) {
                for (int j = 0; j < seq_len; j++) {
                    if (j > i) scores[i * seq_len + j] = -1e9f;
                    if (attention_mask && attention_mask[j] == 0)
                        scores[i * seq_len + j] = -1e9f;
                }
                mistral_softmax(scores + i * seq_len, seq_len);
            }

            const float *v_strided = model->v_buf + kv_h * head_dim;
            float *out_strided = model->attn_out + h * head_dim;

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

#ifdef USE_METAL
output_proj:
#endif
    /* Output projection */
    mistral_linear(model->hidden_state, model->attn_out, layer->attn.o_proj_weight,
                   seq_len, q_dim, hidden);
}

/* ========================================================================
 * MLP (CPU Path)
 * ======================================================================== */

static void mistral_mlp_forward(mistral_model_t *model, mistral_layer_t *layer, int seq_len) {
    int hidden = model->hidden_size;
    int intermediate = model->intermediate_size;

    /* Gate and Up projections */
    mistral_linear(model->mlp_gate, model->norm_buf, layer->mlp.gate_proj_weight,
                   seq_len, hidden, intermediate);
    mistral_linear(model->mlp_up, model->norm_buf, layer->mlp.up_proj_weight,
                   seq_len, hidden, intermediate);

    /* SwiGLU: silu(gate) * up */
    for (int i = 0; i < seq_len * intermediate; i++) {
        float gate = model->mlp_gate[i];
        float sigmoid_val = 1.0f / (1.0f + fast_expf(-gate));
        model->mlp_gate[i] = gate * sigmoid_val * model->mlp_up[i];
    }

    /* Down projection */
    mistral_linear(model->mlp_out, model->mlp_gate, layer->mlp.down_proj_weight,
                   seq_len, intermediate, hidden);
}

/* ========================================================================
 * Layer Forward (CPU)
 * ======================================================================== */

static void mistral_layer_forward(mistral_model_t *model, mistral_layer_t *layer,
                                   int seq_len, const int *attention_mask) {
    int hidden = model->hidden_size;

    memcpy(model->residual, model->hidden_state, seq_len * hidden * sizeof(float));
    mistral_rms_norm(model->norm_buf, model->hidden_state, layer->input_layernorm_weight,
                     seq_len, hidden, MISTRAL_RMS_NORM_EPS);
    mistral_attention_forward(model, layer, seq_len, attention_mask);
    for (int i = 0; i < seq_len * hidden; i++)
        model->hidden_state[i] += model->residual[i];

    memcpy(model->residual, model->hidden_state, seq_len * hidden * sizeof(float));
    mistral_rms_norm(model->norm_buf, model->hidden_state, layer->post_attention_layernorm_weight,
                     seq_len, hidden, MISTRAL_RMS_NORM_EPS);
    mistral_mlp_forward(model, layer, seq_len);
    for (int i = 0; i < seq_len * hidden; i++)
        model->hidden_state[i] = model->residual[i] + model->mlp_out[i];
}

#ifdef USE_METAL
/* ========================================================================
 * BF16 GPU-Accelerated Paths
 * ======================================================================== */

static iris_gpu_tensor_t mistral_f32_to_bf16_tensor(const float *data, int n) {
    iris_gpu_tensor_t f32_tensor = iris_gpu_tensor_create(data, n);
    if (!f32_tensor) return NULL;
    iris_gpu_tensor_t bf16_tensor = iris_gpu_tensor_f32_to_bf16(f32_tensor);
    iris_gpu_tensor_free(f32_tensor);
    return bf16_tensor;
}

/* GPU-only attention: no Q/K norm (Mistral-specific) */
static iris_gpu_tensor_t mistral_attention_gpu(mistral_model_t *model,
                                                mistral_layer_t *layer,
                                                iris_gpu_tensor_t norm_out,
                                                int seq_len,
                                                const int *attention_mask) {
    int num_heads = model->num_heads;
    int num_kv_heads = model->num_kv_heads;
    int head_dim = model->head_dim;
    int hidden = model->hidden_size;
    int q_dim = num_heads * head_dim;
    int kv_dim = num_kv_heads * head_dim;
    float scale = 1.0f / sqrtf((float)head_dim);

    /* Q, K, V projections */
    iris_gpu_tensor_t q = iris_gpu_linear_bf16_native(norm_out, layer->attn.q_proj_weight_bf16,
                                                       seq_len, hidden, q_dim);
    iris_gpu_tensor_t k = iris_gpu_linear_bf16_native(norm_out, layer->attn.k_proj_weight_bf16,
                                                       seq_len, hidden, kv_dim);
    iris_gpu_tensor_t v = iris_gpu_linear_bf16_native(norm_out, layer->attn.v_proj_weight_bf16,
                                                       seq_len, hidden, kv_dim);
    if (!q || !k || !v) goto fail_qkv;

    /* NO Q/K norm — Mistral skips this */

    /* RoPE */
    iris_gpu_rope_text_bf16(q, k, model->rope_cos, model->rope_sin,
                             seq_len, num_heads, num_kv_heads, head_dim);

    /* Causal attention */
    iris_gpu_tensor_t attn_out = iris_gpu_tensor_alloc_f16(seq_len * q_dim);
    if (!attn_out) goto fail_qkv;
    if (!iris_gpu_causal_attention_bf16(attn_out, q, k, v, attention_mask,
                                        seq_len, num_heads, num_kv_heads,
                                        head_dim, scale)) {
        iris_gpu_tensor_free(attn_out);
        goto fail_qkv;
    }
    iris_gpu_tensor_free(q);
    iris_gpu_tensor_free(k);
    iris_gpu_tensor_free(v);

    /* O projection */
    iris_gpu_tensor_t out = iris_gpu_linear_bf16_native(attn_out, layer->attn.o_proj_weight_bf16,
                                                         seq_len, q_dim, hidden);
    iris_gpu_tensor_free(attn_out);
    return out;

fail_qkv:
    if (q) iris_gpu_tensor_free(q);
    if (k) iris_gpu_tensor_free(k);
    if (v) iris_gpu_tensor_free(v);
    return NULL;
}

/* GPU-only MLP */
static iris_gpu_tensor_t mistral_mlp_gpu(mistral_model_t *model, mistral_layer_t *layer,
                                           iris_gpu_tensor_t norm_out, int seq_len) {
    int hidden = model->hidden_size;
    int intermediate = model->intermediate_size;

    iris_gpu_tensor_t gate = iris_gpu_linear_bf16_native(norm_out, layer->mlp.gate_proj_weight_bf16,
                                                          seq_len, hidden, intermediate);
    iris_gpu_tensor_t up = iris_gpu_linear_bf16_native(norm_out, layer->mlp.up_proj_weight_bf16,
                                                        seq_len, hidden, intermediate);
    if (!gate || !up) {
        if (gate) iris_gpu_tensor_free(gate);
        if (up) iris_gpu_tensor_free(up);
        return NULL;
    }

    iris_gpu_silu_mul_bf16(gate, up, seq_len * intermediate);
    iris_gpu_tensor_free(up);

    iris_gpu_tensor_t out = iris_gpu_linear_bf16_native(gate, layer->mlp.down_proj_weight_bf16,
                                                         seq_len, intermediate, hidden);
    iris_gpu_tensor_free(gate);
    return out;
}

/* Fully GPU-resident forward pass */
static int mistral_forward_gpu(mistral_model_t *model, int seq_len, const int *attention_mask) {
    int hidden = model->hidden_size;
    int last_layer = MISTRAL_OUTPUT_LAYER_3;

    iris_gpu_batch_begin();
    iris_gpu_tensor_t hidden_gpu = mistral_f32_to_bf16_tensor(model->hidden_state, seq_len * hidden);
    if (!hidden_gpu) {
        iris_gpu_batch_end();
        return 0;
    }

    iris_gpu_tensor_t saved[3] = {NULL, NULL, NULL};
    for (int i = 0; i < 3; i++) {
        saved[i] = iris_gpu_tensor_alloc_f16(seq_len * hidden);
        if (!saved[i]) {
            for (int j = 0; j <= i; j++) if (saved[j]) iris_gpu_tensor_free(saved[j]);
            iris_gpu_tensor_free(hidden_gpu);
            iris_gpu_batch_end();
            return 0;
        }
    }

    int ok = 1;

    for (int layer_idx = 0; layer_idx <= last_layer; layer_idx++) {
        mistral_layer_t *layer = &model->layers[layer_idx];

        /* Load weights on demand */
        if (model->use_mmap) {
            if (mistral_load_layer_small_f32(layer, model->sf_files, model->num_sf_files, layer_idx) != 0) {
                ok = 0; break;
            }
            mistral_load_layer_bf16(layer, model->sf_files, model->num_sf_files, layer_idx);
        }

        if (!layer->attn.q_proj_weight_bf16 || !layer->mlp.gate_proj_weight_bf16) {
            ok = 0; break;
        }

        /* Input RMS norm */
        iris_gpu_tensor_t norm_w = mistral_f32_to_bf16_tensor(layer->input_layernorm_weight, hidden);
        iris_gpu_tensor_t norm_out = iris_gpu_tensor_alloc_f16(seq_len * hidden);
        if (!norm_w || !norm_out) {
            if (norm_w) iris_gpu_tensor_free(norm_w);
            if (norm_out) iris_gpu_tensor_free(norm_out);
            ok = 0; break;
        }
        iris_gpu_rms_norm_bf16(norm_out, hidden_gpu, norm_w, seq_len, hidden, MISTRAL_RMS_NORM_EPS);
        iris_gpu_tensor_free(norm_w);

        /* Attention */
        iris_gpu_tensor_t attn_out = mistral_attention_gpu(model, layer, norm_out, seq_len, attention_mask);
        iris_gpu_tensor_free(norm_out);
        if (!attn_out) { ok = 0; break; }

        /* Residual */
        iris_gpu_add_bf16(hidden_gpu, hidden_gpu, attn_out, seq_len * hidden);
        iris_gpu_tensor_free(attn_out);

        /* Post-attention RMS norm */
        iris_gpu_tensor_t post_norm_w = mistral_f32_to_bf16_tensor(layer->post_attention_layernorm_weight, hidden);
        norm_out = iris_gpu_tensor_alloc_f16(seq_len * hidden);
        if (!post_norm_w || !norm_out) {
            if (post_norm_w) iris_gpu_tensor_free(post_norm_w);
            if (norm_out) iris_gpu_tensor_free(norm_out);
            ok = 0; break;
        }
        iris_gpu_rms_norm_bf16(norm_out, hidden_gpu, post_norm_w, seq_len, hidden, MISTRAL_RMS_NORM_EPS);
        iris_gpu_tensor_free(post_norm_w);

        /* MLP */
        iris_gpu_tensor_t mlp_out = mistral_mlp_gpu(model, layer, norm_out, seq_len);
        iris_gpu_tensor_free(norm_out);
        if (!mlp_out) { ok = 0; break; }

        /* Residual */
        iris_gpu_add_bf16(hidden_gpu, hidden_gpu, mlp_out, seq_len * hidden);
        iris_gpu_tensor_free(mlp_out);

        /* Free layer weights */
        if (model->use_mmap) {
            mistral_free_layer_weights(layer);
        }

        /* Save at extraction layers */
        if (layer_idx == MISTRAL_OUTPUT_LAYER_1)
            iris_gpu_copy_bf16(saved[0], hidden_gpu, seq_len * hidden);
        else if (layer_idx == MISTRAL_OUTPUT_LAYER_2)
            iris_gpu_copy_bf16(saved[1], hidden_gpu, seq_len * hidden);
        else if (layer_idx == MISTRAL_OUTPUT_LAYER_3)
            iris_gpu_copy_bf16(saved[2], hidden_gpu, seq_len * hidden);

        if (iris_text_progress_callback)
            iris_text_progress_callback(layer_idx, last_layer + 1);
    }

    if (!ok) {
        iris_gpu_batch_end();
        for (int i = 0; i < 3; i++) if (saved[i]) iris_gpu_tensor_free(saved[i]);
        iris_gpu_tensor_free(hidden_gpu);
        return 0;
    }

    /* Convert saved bf16 → f32 */
    iris_gpu_tensor_t saved_f32[3] = {NULL, NULL, NULL};
    for (int i = 0; i < 3; i++)
        saved_f32[i] = iris_gpu_tensor_bf16_to_f32(saved[i]);

    iris_gpu_batch_end();

    if (iris_text_progress_callback)
        iris_text_progress_callback(last_layer, last_layer + 1);

    /* Read results to CPU */
    int read_ok = 1;
    for (int i = 0; i < 3; i++) {
        if (saved_f32[i]) {
            iris_gpu_tensor_read(saved_f32[i], model->layer_outputs[i]);
            iris_gpu_tensor_free(saved_f32[i]);
        } else {
            read_ok = 0;
        }
        iris_gpu_tensor_free(saved[i]);
    }
    iris_gpu_tensor_free(hidden_gpu);

    return read_ok;
}

#endif /* USE_METAL */

/* ========================================================================
 * Forward Pass
 * ======================================================================== */

float *mistral_forward(mistral_model_t *model, const int *input_ids,
                       const int *attention_mask, int seq_len) {
    int hidden = model->hidden_size;
    int last_layer = MISTRAL_OUTPUT_LAYER_3;
    float *output;

    /* Find last real token position for pooled embedding */
    model->pooled_seq_idx = 0;
    if (attention_mask) {
        for (int i = seq_len - 1; i >= 0; i--) {
            if (attention_mask[i] == 1) {
                model->pooled_seq_idx = i;
                break;
            }
        }
    }

    /* Embedding lookup */
    for (int s = 0; s < seq_len; s++) {
        int token_id = input_ids[s];
        if (token_id >= 0 && token_id < model->vocab_size) {
            memcpy(model->hidden_state + s * hidden,
                   model->embed_tokens + token_id * hidden,
                   hidden * sizeof(float));
        } else {
            memset(model->hidden_state + s * hidden, 0, hidden * sizeof(float));
        }
    }

    /* Try GPU-resident path */
#ifdef USE_METAL
    if (model->use_bf16 && iris_metal_available() && seq_len <= 512) {
        if (mistral_forward_gpu(model, seq_len, attention_mask))
            goto concatenate;
    }
#endif

    /* CPU / mixed path */
    for (int layer_idx = 0; layer_idx <= last_layer; layer_idx++) {
        if (model->use_mmap) {
            if (mistral_load_layer_weights(&model->layers[layer_idx], model->sf_files,
                                            model->num_sf_files, layer_idx) != 0) {
                fprintf(stderr, "Failed to load Mistral layer %d weights\n", layer_idx);
                return NULL;
            }
        }

        mistral_layer_forward(model, &model->layers[layer_idx], seq_len, attention_mask);

        if (model->use_mmap) {
            mistral_free_layer_weights(&model->layers[layer_idx]);
        }

        if (layer_idx == MISTRAL_OUTPUT_LAYER_1)
            memcpy(model->layer_outputs[0], model->hidden_state, seq_len * hidden * sizeof(float));
        else if (layer_idx == MISTRAL_OUTPUT_LAYER_2)
            memcpy(model->layer_outputs[1], model->hidden_state, seq_len * hidden * sizeof(float));
        else if (layer_idx == MISTRAL_OUTPUT_LAYER_3)
            memcpy(model->layer_outputs[2], model->hidden_state, seq_len * hidden * sizeof(float));

        if (iris_text_progress_callback)
            iris_text_progress_callback(layer_idx, last_layer + 1);
    }

    /* Concatenate extracted layers → [seq_len, 3*hidden] */
#ifdef USE_METAL
concatenate: (void)0;
#endif
    {
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
    }

    /* Store pooled embedding (last real token from layer 30 output) */
    if (model->pooled_output) {
        memcpy(model->pooled_output,
               model->layer_outputs[2] + model->pooled_seq_idx * hidden,
               hidden * sizeof(float));
    }

    return output;
}

const float *mistral_get_pooled(mistral_model_t *model) {
    if (!model || !model->pooled_output) return NULL;
    return model->pooled_output;
}

/* ========================================================================
 * Weight Loading
 * ======================================================================== */

static float *mistral_load_tensor(safetensors_file_t **files, int num_files, const char *name) {
    for (int f = 0; f < num_files; f++) {
        const safetensor_t *t = safetensors_find(files[f], name);
        if (t) return safetensors_get_f32(files[f], t);
    }
    fprintf(stderr, "Error: required tensor not found: %s\n", name);
    return NULL;
}

#ifdef USE_METAL
static uint16_t *mistral_load_tensor_bf16(safetensors_file_t **files, int num_files, const char *name) {
    for (int f = 0; f < num_files; f++) {
        const safetensor_t *t = safetensors_find(files[f], name);
        if (t && safetensor_is_bf16(t)) {
            return safetensors_get_bf16_direct(files[f], t);
        }
    }
    return NULL;
}

static int mistral_load_layer_small_f32(mistral_layer_t *layer, safetensors_file_t **files,
                                         int num_files, int layer_idx) {
    char name[256];

    snprintf(name, sizeof(name), "model.layers.%d.input_layernorm.weight", layer_idx);
    layer->input_layernorm_weight = mistral_load_tensor(files, num_files, name);

    snprintf(name, sizeof(name), "model.layers.%d.post_attention_layernorm.weight", layer_idx);
    layer->post_attention_layernorm_weight = mistral_load_tensor(files, num_files, name);

    /* No Q/K norm for Mistral */

    return (layer->input_layernorm_weight && layer->post_attention_layernorm_weight) ? 0 : -1;
}

static int mistral_load_layer_bf16(mistral_layer_t *layer, safetensors_file_t **files,
                                    int num_files, int layer_idx) {
    char name[256];

    snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.weight", layer_idx);
    layer->attn.q_proj_weight_bf16 = mistral_load_tensor_bf16(files, num_files, name);

    snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_proj.weight", layer_idx);
    layer->attn.k_proj_weight_bf16 = mistral_load_tensor_bf16(files, num_files, name);

    snprintf(name, sizeof(name), "model.layers.%d.self_attn.v_proj.weight", layer_idx);
    layer->attn.v_proj_weight_bf16 = mistral_load_tensor_bf16(files, num_files, name);

    snprintf(name, sizeof(name), "model.layers.%d.self_attn.o_proj.weight", layer_idx);
    layer->attn.o_proj_weight_bf16 = mistral_load_tensor_bf16(files, num_files, name);

    snprintf(name, sizeof(name), "model.layers.%d.mlp.gate_proj.weight", layer_idx);
    layer->mlp.gate_proj_weight_bf16 = mistral_load_tensor_bf16(files, num_files, name);

    snprintf(name, sizeof(name), "model.layers.%d.mlp.up_proj.weight", layer_idx);
    layer->mlp.up_proj_weight_bf16 = mistral_load_tensor_bf16(files, num_files, name);

    snprintf(name, sizeof(name), "model.layers.%d.mlp.down_proj.weight", layer_idx);
    layer->mlp.down_proj_weight_bf16 = mistral_load_tensor_bf16(files, num_files, name);

    return (layer->attn.q_proj_weight_bf16 && layer->attn.k_proj_weight_bf16 &&
            layer->attn.v_proj_weight_bf16 && layer->attn.o_proj_weight_bf16 &&
            layer->mlp.gate_proj_weight_bf16 && layer->mlp.up_proj_weight_bf16 &&
            layer->mlp.down_proj_weight_bf16);
}
#endif

static int mistral_load_layer_weights(mistral_layer_t *layer, safetensors_file_t **files,
                                       int num_files, int layer_idx) {
    char name[256];

    snprintf(name, sizeof(name), "model.layers.%d.input_layernorm.weight", layer_idx);
    layer->input_layernorm_weight = mistral_load_tensor(files, num_files, name);

    snprintf(name, sizeof(name), "model.layers.%d.post_attention_layernorm.weight", layer_idx);
    layer->post_attention_layernorm_weight = mistral_load_tensor(files, num_files, name);

    snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.weight", layer_idx);
    layer->attn.q_proj_weight = mistral_load_tensor(files, num_files, name);

    snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_proj.weight", layer_idx);
    layer->attn.k_proj_weight = mistral_load_tensor(files, num_files, name);

    snprintf(name, sizeof(name), "model.layers.%d.self_attn.v_proj.weight", layer_idx);
    layer->attn.v_proj_weight = mistral_load_tensor(files, num_files, name);

    snprintf(name, sizeof(name), "model.layers.%d.self_attn.o_proj.weight", layer_idx);
    layer->attn.o_proj_weight = mistral_load_tensor(files, num_files, name);

    snprintf(name, sizeof(name), "model.layers.%d.mlp.gate_proj.weight", layer_idx);
    layer->mlp.gate_proj_weight = mistral_load_tensor(files, num_files, name);

    snprintf(name, sizeof(name), "model.layers.%d.mlp.up_proj.weight", layer_idx);
    layer->mlp.up_proj_weight = mistral_load_tensor(files, num_files, name);

    snprintf(name, sizeof(name), "model.layers.%d.mlp.down_proj.weight", layer_idx);
    layer->mlp.down_proj_weight = mistral_load_tensor(files, num_files, name);

    if (!layer->input_layernorm_weight || !layer->post_attention_layernorm_weight ||
        !layer->attn.q_proj_weight || !layer->attn.k_proj_weight ||
        !layer->attn.v_proj_weight || !layer->attn.o_proj_weight ||
        !layer->mlp.gate_proj_weight || !layer->mlp.up_proj_weight ||
        !layer->mlp.down_proj_weight) {
        return -1;
    }
    return 0;
}

static void mistral_free_layer_weights(mistral_layer_t *layer) {
    free(layer->input_layernorm_weight);
    free(layer->post_attention_layernorm_weight);
    free(layer->attn.q_proj_weight);
    free(layer->attn.k_proj_weight);
    free(layer->attn.v_proj_weight);
    free(layer->attn.o_proj_weight);
    free(layer->mlp.gate_proj_weight);
    free(layer->mlp.up_proj_weight);
    free(layer->mlp.down_proj_weight);

    memset(layer, 0, sizeof(mistral_layer_t));
}

/* ========================================================================
 * Config Parsing
 * ======================================================================== */

static int parse_mistral_config(const char *model_dir, mistral_model_t *model) {
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

    if ((p = strstr(buf, "\"hidden_size\"")))
        if ((p = strchr(p, ':'))) hidden = atoi(p + 1);
    if ((p = strstr(buf, "\"intermediate_size\"")))
        if ((p = strchr(p, ':'))) intermediate = atoi(p + 1);
    if ((p = strstr(buf, "\"num_attention_heads\"")))
        if ((p = strchr(p, ':'))) num_heads = atoi(p + 1);
    if ((p = strstr(buf, "\"num_key_value_heads\"")))
        if ((p = strchr(p, ':'))) num_kv_heads = atoi(p + 1);
    if ((p = strstr(buf, "\"head_dim\"")))
        if ((p = strchr(p, ':'))) head_dim = atoi(p + 1);
    if ((p = strstr(buf, "\"vocab_size\"")))
        if ((p = strchr(p, ':'))) vocab_size = atoi(p + 1);
    if ((p = strstr(buf, "\"num_hidden_layers\"")))
        if ((p = strchr(p, ':'))) num_layers = atoi(p + 1);
    if ((p = strstr(buf, "\"rope_theta\"")))
        if ((p = strchr(p, ':'))) rope_theta = atof(p + 1);

    if (hidden <= 0 || num_heads <= 0) return -1;

    model->hidden_size = hidden;
    model->intermediate_size = intermediate > 0 ? intermediate : 32768;
    model->num_heads = num_heads;
    model->num_kv_heads = num_kv_heads > 0 ? num_kv_heads : 8;
    model->head_dim = head_dim > 0 ? head_dim : (hidden / num_heads);
    model->vocab_size = vocab_size > 0 ? vocab_size : 131072;
    model->num_layers = num_layers > 0 ? num_layers : 40;
    model->rope_theta = rope_theta > 0 ? rope_theta : 1000000000.0f;
    model->text_dim = 3 * hidden;

    return 0;
}

/* ========================================================================
 * Safetensors Shard Discovery
 * ======================================================================== */

static int mistral_open_shards(const char *model_dir,
                                safetensors_file_t **files, int max_files) {
    char path[1024];

    /* Try index JSON first */
    snprintf(path, sizeof(path), "%s/model.safetensors.index.json", model_dir);
    FILE *f = fopen(path, "r");
    if (f) {
        fseek(f, 0, SEEK_END);
        long fsize = ftell(f);
        fseek(f, 0, SEEK_SET);
        char *buf = malloc(fsize + 1);
        if (!buf) { fclose(f); return 0; }
        fread(buf, 1, fsize, f);
        buf[fsize] = '\0';
        fclose(f);

        char shard_names[MISTRAL_MAX_SHARDS][128];
        int num_shards = 0;

        char *p = buf;
        while ((p = strstr(p, "model-")) != NULL) {
            char *end = strstr(p, ".safetensors");
            if (!end) { p++; continue; }
            end += strlen(".safetensors");

            int len = (int)(end - p);
            if (len >= 128) { p = end; continue; }

            char name[128];
            memcpy(name, p, len);
            name[len] = '\0';

            int found = 0;
            for (int i = 0; i < num_shards; i++) {
                if (strcmp(shard_names[i], name) == 0) { found = 1; break; }
            }
            if (!found && num_shards < max_files && num_shards < MISTRAL_MAX_SHARDS) {
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
                    fprintf(stderr, "mistral: failed to open shard %s\n", path);
                    for (int j = 0; j < i; j++) safetensors_close(files[j]);
                    return 0;
                }
            }
            return num_shards;
        }
    }

    /* Fallback: single model.safetensors */
    snprintf(path, sizeof(path), "%s/model.safetensors", model_dir);
    files[0] = safetensors_open(path);
    if (files[0]) return 1;

    return 0;
}

/* ========================================================================
 * Model Loading (always mmap)
 * ======================================================================== */

static void mistral_alloc_work_buffers(mistral_model_t *model) {
    int seq_len = MISTRAL_MAX_SEQ_LEN;
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

    for (int i = 0; i < 3; i++) {
        model->layer_outputs[i] = malloc(seq_len * hidden * sizeof(float));
    }

    model->pooled_output = malloc(hidden * sizeof(float));
}

mistral_model_t *mistral_model_load_mmap(const char *model_dir) {
    mistral_model_t *model = calloc(1, sizeof(mistral_model_t));
    if (!model) return NULL;

    model->use_mmap = 1;

    if (parse_mistral_config(model_dir, model) != 0) {
        fprintf(stderr, "mistral_model_load_mmap: failed to parse config.json\n");
        free(model);
        return NULL;
    }

    if (iris_verbose)
        fprintf(stderr, "Mistral config: hidden=%d, heads=%d/%d, head_dim=%d, layers=%d, ff=%d, rope_theta=%.0f\n",
                model->hidden_size, model->num_heads, model->num_kv_heads,
                model->head_dim, model->num_layers, model->intermediate_size, model->rope_theta);

#ifdef USE_METAL
    model->use_bf16 = (iris_metal_available() && !getenv("IRIS_MISTRAL_NO_BF16")) ? 1 : 0;
    if (model->use_bf16 && iris_verbose)
        fprintf(stderr, "Mistral: bf16 GPU acceleration enabled\n");
#endif

    model->layers = calloc(model->num_layers, sizeof(mistral_layer_t));
    if (!model->layers) {
        free(model);
        return NULL;
    }

    /* Open safetensors shards */
    model->num_sf_files = mistral_open_shards(model_dir, model->sf_files, MISTRAL_MAX_SHARDS);
    if (model->num_sf_files == 0) {
        fprintf(stderr, "mistral_model_load_mmap: failed to open safetensors files\n");
        goto error;
    }

    /* Load embeddings (large but needed for all tokens) */
    model->embed_tokens = mistral_load_tensor(model->sf_files, model->num_sf_files,
                                               "model.embed_tokens.weight");
    if (!model->embed_tokens) {
        fprintf(stderr, "mistral_model_load_mmap: failed to load embed_tokens\n");
        goto error;
    }

    /* Load final norm */
    model->norm_weight = mistral_load_tensor(model->sf_files, model->num_sf_files,
                                              "model.norm.weight");
    if (!model->norm_weight) {
        fprintf(stderr, "mistral_model_load_mmap: failed to load final norm\n");
        goto error;
    }

    if (iris_verbose)
        fprintf(stderr, "Mistral mmap mode: layer weights will be loaded on-demand\n");

    /* Compute RoPE frequencies */
    int max_seq = MISTRAL_MAX_SEQ_LEN;
    int half_dim = model->head_dim / 2;
    model->rope_cos = malloc(max_seq * half_dim * sizeof(float));
    model->rope_sin = malloc(max_seq * half_dim * sizeof(float));
    mistral_compute_rope_freqs(model->rope_cos, model->rope_sin, max_seq,
                                model->head_dim, model->rope_theta);

    /* Allocate working memory */
    mistral_alloc_work_buffers(model);

    return model;

error:
    mistral_model_free(model);
    return NULL;
}

void mistral_model_free(mistral_model_t *model) {
    if (!model) return;

    free(model->embed_tokens);
    free(model->norm_weight);
    free(model->rope_cos);
    free(model->rope_sin);

    if (model->layers) {
        for (int i = 0; i < model->num_layers; i++) {
            mistral_layer_t *layer = &model->layers[i];
            free(layer->input_layernorm_weight);
            free(layer->post_attention_layernorm_weight);
            free(layer->attn.q_proj_weight);
            free(layer->attn.k_proj_weight);
            free(layer->attn.v_proj_weight);
            free(layer->attn.o_proj_weight);
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
    free(model->attn_q_head);
    free(model->attn_v_head);
    free(model->attn_out_head);

    for (int i = 0; i < 3; i++) free(model->layer_outputs[i]);
    free(model->pooled_output);

    for (int i = 0; i < model->num_sf_files; i++) {
        if (model->sf_files[i]) safetensors_close(model->sf_files[i]);
    }

    free(model);
}

/* ========================================================================
 * Combined Encoder API
 * ======================================================================== */

mistral_encoder_t *mistral_encoder_load(const char *model_dir) {
    mistral_encoder_t *enc = calloc(1, sizeof(mistral_encoder_t));
    if (!enc) return NULL;

    /* Load tokenizer */
    char tok_path[512];
    snprintf(tok_path, sizeof(tok_path), "%s/tokenizer/tokenizer.json", model_dir);

    /* Try tokenizer/ subdir first, then text_encoder/tokenizer.json */
    enc->tokenizer = mistral_tokenizer_load(tok_path);
    if (!enc->tokenizer) {
        snprintf(tok_path, sizeof(tok_path), "%s/text_encoder/tokenizer.json", model_dir);
        enc->tokenizer = mistral_tokenizer_load(tok_path);
    }
    if (!enc->tokenizer) {
        fprintf(stderr, "mistral_encoder_load: failed to load tokenizer\n");
        free(enc);
        return NULL;
    }

    /* Load model (always mmap — Mistral is ~48GB) */
    char model_path[512];
    snprintf(model_path, sizeof(model_path), "%s/text_encoder", model_dir);
    enc->model = mistral_model_load_mmap(model_path);
    if (!enc->model) {
        fprintf(stderr, "mistral_encoder_load: failed to load model\n");
        mistral_tokenizer_free(enc->tokenizer);
        free(enc);
        return NULL;
    }

    return enc;
}

void mistral_encoder_free(mistral_encoder_t *enc) {
    if (!enc) return;
    mistral_tokenizer_free(enc->tokenizer);
    mistral_model_free(enc->model);
    free(enc);
}

float *mistral_encode_text(mistral_encoder_t *enc, const char *prompt,
                           int *out_num_tokens) {
    if (!enc || !enc->tokenizer || !enc->model || !prompt) return NULL;

    /* Tokenize with chat template */
    int num_tokens;
    int *tokens = mistral_tokenize_chat(enc->tokenizer, prompt, &num_tokens,
                                         MISTRAL_MAX_SEQ_LEN);
    if (!tokens) return NULL;

    /* Pad to max length */
    int *attention_mask = malloc(MISTRAL_MAX_SEQ_LEN * sizeof(int));
    int *padded_tokens = mistral_pad_tokens(tokens, num_tokens, MISTRAL_MAX_SEQ_LEN, attention_mask);
    free(tokens);

    if (!padded_tokens) {
        free(attention_mask);
        return NULL;
    }

    if (out_num_tokens) *out_num_tokens = num_tokens;

    /* Forward pass */
    float *embeddings = mistral_forward(enc->model, padded_tokens, attention_mask, MISTRAL_MAX_SEQ_LEN);

    free(padded_tokens);
    free(attention_mask);

    return embeddings;
}
