/*
 * FLUX Diffusion Transformer Implementation
 *
 * Flux2Transformer2DModel - Rectified Flow Transformer for image generation.
 *
 * Architecture (klein 4B):
 * - 5 double-stream blocks (MM-DiT: separate image/text processing, joint attention)
 * - 20 single-stream blocks (parallel DiT: fused QKV+FFN)
 * - 24 attention heads, 128 dim per head (3072 hidden)
 * - No bias parameters
 * - SwiGLU activation
 * - RoPE positional embeddings
 * - Shared AdaLN-Zero modulation
 */

#include "flux.h"
#include "flux_kernels.h"
#include "flux_safetensors.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

/* External timing counters from flux_sample.c */
extern double flux_timing_transformer_total;
extern double flux_timing_transformer_double;
extern double flux_timing_transformer_single;
extern double flux_timing_transformer_final;

/* Helper to get current time in ms (wall-clock) */
static double tf_get_time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

/* Use BLAS for matrix operations when enabled via Makefile */
#ifdef USE_BLAS
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#endif

/* Use Metal for GPU acceleration when available */
#ifdef USE_METAL
#include "flux_metal.h"
#endif

/* Helper macro for using bf16 linear layer when available
 * Uses bf16 if w_bf16 is not NULL (GPU), otherwise falls back to f32 */
#define LINEAR_BF16_OR_F32(out, x, w_f32, w_bf16, seq, in_dim, out_dim) \
    do { \
        if ((w_bf16) != NULL) { \
            flux_linear_nobias_bf16((out), (x), (w_bf16), (seq), (in_dim), (out_dim)); \
        } else { \
            flux_linear_nobias((out), (x), (w_f32), (seq), (in_dim), (out_dim)); \
        } \
    } while(0)

/* Gated add: out += gate * proj, where gate is [hidden] and proj is [seq, hidden]
 * Double loop avoids modulo which prevents vectorization.
 */
static inline void gated_add(float *out, const float *gate, const float *proj,
                             int seq, int hidden) {
    for (int s = 0; s < seq; s++) {
        for (int i = 0; i < hidden; i++) {
            out[s * hidden + i] += gate[i] * proj[s * hidden + i];
        }
    }
}

/* ========================================================================
 * Transformer Data Structures
 * ======================================================================== */

/* AdaLN-Zero modulation parameters (shared across blocks) */
typedef struct {
    float *mod_weight;      /* [hidden * 6] for double, [hidden * 3] for single */
} adaln_t;

/* Double-stream block (MM-DiT style) */
typedef struct {
    /* Image stream - separate Q, K, V weights (f32 and bf16) */
    float *img_q_weight;            /* [hidden, hidden] (f32) */
    float *img_k_weight;            /* [hidden, hidden] (f32) */
    float *img_v_weight;            /* [hidden, hidden] (f32) */
    uint16_t *img_q_weight_bf16;    /* [hidden, hidden] (bf16) */
    uint16_t *img_k_weight_bf16;    /* [hidden, hidden] (bf16) */
    uint16_t *img_v_weight_bf16;    /* [hidden, hidden] (bf16) */
    float *img_norm_q_weight;       /* [head_dim] - QK norm on Q (always f32) */
    float *img_norm_k_weight;       /* [head_dim] - QK norm on K (always f32) */
    float *img_proj_weight;         /* [hidden, hidden] (f32) */
    uint16_t *img_proj_weight_bf16; /* [hidden, hidden] (bf16) */
    float *img_mlp_gate_weight;     /* [mlp_hidden, hidden] (f32) */
    float *img_mlp_up_weight;       /* [mlp_hidden, hidden] (f32) */
    float *img_mlp_down_weight;     /* [hidden, mlp_hidden] (f32) */
    uint16_t *img_mlp_gate_weight_bf16; /* [mlp_hidden, hidden] (bf16) */
    uint16_t *img_mlp_up_weight_bf16;   /* [mlp_hidden, hidden] (bf16) */
    uint16_t *img_mlp_down_weight_bf16; /* [hidden, mlp_hidden] (bf16) */

    /* Text stream - separate Q, K, V weights (f32 and bf16) */
    float *txt_q_weight;            /* [hidden, hidden] (f32) */
    float *txt_k_weight;            /* [hidden, hidden] (f32) */
    float *txt_v_weight;            /* [hidden, hidden] (f32) */
    uint16_t *txt_q_weight_bf16;    /* [hidden, hidden] (bf16) */
    uint16_t *txt_k_weight_bf16;    /* [hidden, hidden] (bf16) */
    uint16_t *txt_v_weight_bf16;    /* [hidden, hidden] (bf16) */
    float *txt_norm_q_weight;       /* [head_dim] - QK norm on Q (always f32) */
    float *txt_norm_k_weight;       /* [head_dim] - QK norm on K (always f32) */
    float *txt_proj_weight;         /* [hidden, hidden] (f32) */
    uint16_t *txt_proj_weight_bf16; /* [hidden, hidden] (bf16) */
    float *txt_mlp_gate_weight;     /* [mlp_hidden, hidden] (f32) */
    float *txt_mlp_up_weight;       /* [mlp_hidden, hidden] (f32) */
    float *txt_mlp_down_weight;     /* [hidden, mlp_hidden] (f32) */
    uint16_t *txt_mlp_gate_weight_bf16; /* [mlp_hidden, hidden] (bf16) */
    uint16_t *txt_mlp_up_weight_bf16;   /* [mlp_hidden, hidden] (bf16) */
    uint16_t *txt_mlp_down_weight_bf16; /* [hidden, mlp_hidden] (bf16) */
} double_block_t;

/* Single-stream block (Parallel DiT style, fused) */
typedef struct {
    /* Fused QKV + FFN input projection - bf16 for GPU, f32 as fallback */
    float *qkv_mlp_weight;          /* [hidden*3 + mlp_hidden*2, hidden] (f32) */
    uint16_t *qkv_mlp_weight_bf16;  /* [hidden*3 + mlp_hidden*2, hidden] (bf16) */
    /* QK normalization - always f32 (small) */
    float *norm_q_weight;           /* [head_dim] */
    float *norm_k_weight;           /* [head_dim] */
    /* Fused attention out + FFN down projection - bf16 for GPU, f32 as fallback */
    float *proj_mlp_weight;         /* [hidden, hidden + mlp_hidden] (f32) */
    uint16_t *proj_mlp_weight_bf16; /* [hidden, hidden + mlp_hidden] (bf16) */
} single_block_t;

/* Timestep embedding MLP
 * FLUX.2-klein uses 256-dim sinusoidal embedding (128 frequencies)
 * linear_1: [hidden, 256] - projects sinusoidal to hidden
 * linear_2: [hidden, hidden] - another linear layer
 */
typedef struct {
    float *fc1_weight;              /* [hidden, 256] */
    float *fc2_weight;              /* [hidden, hidden] */
    int sincos_dim;                 /* 256 for FLUX.2-klein */
} time_embed_t;

/* Full transformer context */
typedef struct flux_transformer {
    /* Configuration */
    int hidden_size;        /* 3072 */
    int num_heads;          /* 24 */
    int head_dim;           /* 128 */
    int mlp_hidden;         /* hidden * 3 = 9216 */
    int num_double_layers;  /* 5 */
    int num_single_layers;  /* 20 */
    int text_dim;           /* 7680 */
    int latent_channels;    /* 128 */
    float rope_theta;       /* 2000 */
    int rope_dim;           /* 128 */
    int use_bf16;           /* Use bf16 weights (1) or f32 (0) */

    /* Input projections */
    float *img_in_weight;   /* [hidden, latent_channels] */
    float *txt_in_weight;   /* [hidden, text_dim] */
    uint16_t *img_in_weight_bf16;   /* [hidden, latent_channels] (bf16) */
    uint16_t *txt_in_weight_bf16;   /* [hidden, text_dim] (bf16) */

    /* Timestep embedding */
    time_embed_t time_embed;
    float *time_freq;       /* [hidden/2] - precomputed sinusoidal frequencies */

    /* Shared AdaLN modulation (f32 and bf16) */
    float *adaln_double_img_weight;  /* [hidden * 6, hidden] for double block img stream */
    float *adaln_double_txt_weight;  /* [hidden * 6, hidden] for double block txt stream */
    float *adaln_single_weight;      /* [hidden * 3, hidden] for single block */
    uint16_t *adaln_double_img_weight_bf16;  /* (bf16) */
    uint16_t *adaln_double_txt_weight_bf16;  /* (bf16) */
    uint16_t *adaln_single_weight_bf16;      /* (bf16) */

    /* Transformer blocks */
    double_block_t *double_blocks;
    single_block_t *single_blocks;

    /* Final layer */
    float *final_norm_weight;       /* [hidden] (always f32) */
    float *final_proj_weight;       /* [latent_channels, hidden] */
    uint16_t *final_proj_weight_bf16; /* [latent_channels, hidden] (bf16) */

    /* RoPE frequencies (precomputed) */
    float *rope_freqs;              /* [max_seq, head_dim/2, 2] - legacy 1D */
    float *rope_cos;                /* [max_seq, axis_dim] - 2D cos frequencies */
    float *rope_sin;                /* [max_seq, axis_dim] - 2D sin frequencies */
    int max_seq_len;
    int axis_dim;                   /* 32 for FLUX (128 head_dim / 4 axes) */

    /* Working memory */
    float *img_hidden;              /* [max_img_seq, hidden] */
    float *txt_hidden;              /* [max_txt_seq, hidden] */
    float *q, *k, *v;               /* [max_seq, hidden] */
    float *attn_out;                /* [max_seq, hidden] */
    float *mlp_buffer;              /* [max_seq, mlp_hidden] */
    float *work1, *work2;
    size_t work_size;

    /* Pre-allocated attention workspaces to avoid malloc in hot path */
    float *attn_q_t;                /* [max_seq, hidden] transposed Q */
    float *attn_k_t;                /* [max_seq, hidden] transposed K */
    float *attn_v_t;                /* [max_seq, hidden] transposed V */
    float *attn_out_t;              /* [max_seq, hidden] transposed output */
    float *attn_scores;             /* [max_seq, max_seq] attention scores */
    float *attn_cat_k;              /* [max_seq, hidden] concatenated K */
    float *attn_cat_v;              /* [max_seq, hidden] concatenated V */

    /* Single-block work buffers (pre-allocated to avoid malloc in hot path) */
    float *single_q;                /* [max_seq, hidden] */
    float *single_k;                /* [max_seq, hidden] */
    float *single_v;                /* [max_seq, hidden] */
    float *single_mlp_gate;         /* [max_seq, mlp_hidden] */
    float *single_mlp_up;           /* [max_seq, mlp_hidden] */
    float *single_attn_out;         /* [max_seq, hidden] */
    float *single_concat;           /* [max_seq, hidden + mlp_hidden] */

    /* FFN work buffers (shared by double and single blocks) */
    float *ffn_gate;                /* [max_seq, mlp_hidden] */
    float *ffn_up;                  /* [max_seq, mlp_hidden] */

    /* Double-block work buffers */
    float *t_emb_silu;              /* [hidden] */
    float *double_mod_img;          /* [hidden * 6] */
    float *double_mod_txt;          /* [hidden * 6] */
    float *double_img_attn_out;     /* [max_seq, hidden] */
    float *double_txt_attn_out;     /* [max_seq, hidden] */
} flux_transformer_t;

/* Forward declarations */
void flux_transformer_free(flux_transformer_t *tf);

/* ========================================================================
 * RoPE (Rotary Position Embeddings)
 * ======================================================================== */

/* Precompute RoPE frequencies for given positions (1D version) */
static void compute_rope_freqs(float *freqs, int max_seq, int dim, float theta) {
    int half_dim = dim / 2;

    for (int pos = 0; pos < max_seq; pos++) {
        for (int d = 0; d < half_dim; d++) {
            float freq = 1.0f / powf(theta, (float)(2 * d) / (float)dim);
            float angle = (float)pos * freq;
            freqs[pos * half_dim * 2 + d * 2] = cosf(angle);
            freqs[pos * half_dim * 2 + d * 2 + 1] = sinf(angle);
        }
    }
}

/* Compute 2D RoPE frequencies for image tokens (h, w positions)
 * FLUX uses axes_dims_rope: [32, 32, 32, 32] = 128 total
 * Position IDs format: (T, H, W, L) where:
 * - Axis 0 (dims 0-31): T position (always 0 for images)
 * - Axis 1 (dims 32-63): H position (y/height coordinate)
 * - Axis 2 (dims 64-95): W position (x/width coordinate)
 * - Axis 3 (dims 96-127): L position (always 0 for images)
 */
static void compute_rope_2d(float *cos_out, float *sin_out,
                            int patch_h, int patch_w, int axis_dim, float theta) {
    int half_axis = axis_dim / 2;  /* 16 dims per half-axis */
    int seq = patch_h * patch_w;
    (void)seq;  /* Unused but kept for documentation */

    /* Precompute base frequencies on stack (axis_dim is always 32, half_axis=16) */
    float base_freqs[16];
    for (int d = 0; d < half_axis; d++) {
        base_freqs[d] = 1.0f / powf(theta, (float)(2 * d) / (float)axis_dim);
    }

    for (int hy = 0; hy < patch_h; hy++) {
        for (int wx = 0; wx < patch_w; wx++) {
            int pos = hy * patch_w + wx;
            float *cos_p = cos_out + pos * axis_dim * 4;  /* 4 axes * 32 dims each = 128 */
            float *sin_p = sin_out + pos * axis_dim * 4;

            /* Axis 0 (dims 0-31): T position = 0, so cos=1, sin=0 */
            for (int d = 0; d < axis_dim; d++) {
                cos_p[d] = 1.0f;
                sin_p[d] = 0.0f;
            }

            /* Axis 1 (dims 32-63): H position (y/height)
             * Python RoPE stacks [cos, -sin, sin, cos] as 2x2 matrix per freq.
             * For apply_rope: out = [[cos, -sin], [sin, cos]] @ [x0, x1]
             * We store cos/sin per pair and apply_rope_2d handles the rotation.
             */
            for (int d = 0; d < half_axis; d++) {
                float angle_h = (float)hy * base_freqs[d];
                float cos_h = cosf(angle_h);
                float sin_h = sinf(angle_h);
                /* Each frequency contributes to a pair of dimensions */
                cos_p[axis_dim + d * 2] = cos_h;
                cos_p[axis_dim + d * 2 + 1] = cos_h;
                sin_p[axis_dim + d * 2] = sin_h;
                sin_p[axis_dim + d * 2 + 1] = sin_h;
            }

            /* Axis 2 (dims 64-95): W position (x/width) */
            for (int d = 0; d < half_axis; d++) {
                float angle_w = (float)wx * base_freqs[d];
                float cos_w = cosf(angle_w);
                float sin_w = sinf(angle_w);
                cos_p[axis_dim * 2 + d * 2] = cos_w;
                cos_p[axis_dim * 2 + d * 2 + 1] = cos_w;
                sin_p[axis_dim * 2 + d * 2] = sin_w;
                sin_p[axis_dim * 2 + d * 2 + 1] = sin_w;
            }

            /* Axis 3 (dims 96-127): L position = 0, so cos=1, sin=0 */
            for (int d = 0; d < axis_dim; d++) {
                cos_p[axis_dim * 3 + d] = 1.0f;
                sin_p[axis_dim * 3 + d] = 0.0f;
            }
        }
    }
}

/* Apply 2D RoPE to image Q/K: x shape [seq, heads * head_dim]
 * Matches diffusers apply_rotary_emb with use_real=True, use_real_unbind_dim=-1
 * For each pair (i, i+1): out[i] = x[i]*cos - x[i+1]*sin, out[i+1] = x[i+1]*cos + x[i]*sin
 * cos/sin have 128 dims per position (4 axes * 32 dims)
 */
static void apply_rope_2d(float *x, const float *cos_freq, const float *sin_freq,
                          int seq, int heads, int head_dim, int axis_dim) {
    (void)axis_dim;  /* head_dim = 128 = 4 * axis_dim (axis_dim = 32) */
    for (int s = 0; s < seq; s++) {
        const float *cos_s = cos_freq + s * head_dim;  /* [128] */
        const float *sin_s = sin_freq + s * head_dim;

        for (int h = 0; h < heads; h++) {
            float *vec = x + (s * heads + h) * head_dim;

            /* Apply rotation to all 128 dims in pairs (0,1), (2,3), ... (126,127) */
            for (int d = 0; d < head_dim; d += 2) {
                float cos_val = cos_s[d];  /* cos[d] == cos[d+1] due to repeat_interleave */
                float sin_val = sin_s[d];
                float x0 = vec[d];
                float x1 = vec[d + 1];
                /* Complex rotation: (x0 + i*x1) * (cos + i*sin) */
                vec[d] = x0 * cos_val - x1 * sin_val;
                vec[d + 1] = x1 * cos_val + x0 * sin_val;
            }
        }
    }
}

/* Compute text RoPE frequencies for axis 3 (L dimension)
 * Text tokens have position IDs (T=0, H=0, W=0, L=seq_idx) where L = 0..seq-1
 * So axes 0-2 are identity, and axis 3 has the sequence position
 */
static void compute_rope_text(float *cos_out, float *sin_out,
                              int txt_seq, int axis_dim, float theta) {
    int half_axis = axis_dim / 2;  /* 16 */
    int head_dim = axis_dim * 4;   /* 128 */

    /* Precompute base frequencies on stack (axis_dim is always 32, half_axis=16) */
    float base_freqs[16];
    for (int d = 0; d < half_axis; d++) {
        base_freqs[d] = 1.0f / powf(theta, (float)(2 * d) / (float)axis_dim);
    }

    for (int s = 0; s < txt_seq; s++) {
        float *cos_p = cos_out + s * head_dim;  /* 128 dims */
        float *sin_p = sin_out + s * head_dim;

        /* Axes 0, 1, 2 (dims 0-95): T=H=W=0, so identity */
        for (int d = 0; d < axis_dim * 3; d++) {
            cos_p[d] = 1.0f;
            sin_p[d] = 0.0f;
        }

        /* Axis 3 (dims 96-127): L position = s (sequence index) */
        for (int d = 0; d < half_axis; d++) {
            float angle = (float)s * base_freqs[d];
            float cos_l = cosf(angle);
            float sin_l = sinf(angle);
            cos_p[axis_dim * 3 + d * 2] = cos_l;
            cos_p[axis_dim * 3 + d * 2 + 1] = cos_l;
            sin_p[axis_dim * 3 + d * 2] = sin_l;
            sin_p[axis_dim * 3 + d * 2 + 1] = sin_l;
        }
    }
}

/* ========================================================================
 * Timestep Embedding
 * ======================================================================== */

/* Sinusoidal timestep embedding
 * Matches diffusers get_timestep_embedding with:
 *   flip_sin_to_cos=True, downscale_freq_shift=0
 * Output format: [cos(all freqs), sin(all freqs)]
 */
static void get_timestep_embedding(float *out, float t, int dim, float max_period) {
    int half = dim / 2;
    float log_max = logf(max_period);

    for (int i = 0; i < half; i++) {
        /* freq = exp(-log(max_period) * i / half_dim) */
        float freq = expf(-log_max * (float)i / (float)half);
        float angle = t * freq;
        out[i] = cosf(angle);           /* cos part first (flip_sin_to_cos=True) */
        out[i + half] = sinf(angle);    /* sin part second */
    }
}

/* Forward through time embedding MLP
 * work: pre-allocated buffer of size hidden for intermediate result
 */
static void time_embed_forward(float *out, const float *t_sincos,
                               const time_embed_t *te, int hidden, float *work) {
    /* MLP: fc1 (256->hidden) -> SiLU -> fc2 (hidden->hidden) */
    int sincos_dim = te->sincos_dim;

    /* fc1: [sincos_dim] -> [hidden] */
    flux_linear_nobias(work, t_sincos, te->fc1_weight, 1, sincos_dim, hidden);

    /* SiLU */
    flux_silu(work, hidden);

    /* fc2: [hidden] -> [hidden] */
    flux_linear_nobias(out, work, te->fc2_weight, 1, hidden, hidden);
}

/* ========================================================================
 * AdaLN-Zero Modulation
 * ======================================================================== */

/* Apply AdaLN: out = (1 + scale) * LayerNorm(x) + shift
 * This is the standard DiT/FLUX formulation where scale is centered at 0
 * FLUX2 uses LayerNorm (not RMSNorm) with elementwise_affine=False before modulation
 */
static void apply_adaln(float *out, const float *x,
                        const float *shift, const float *scale,
                        int seq, int hidden, float eps) {
    /* Layer Norm (subtract mean, divide by std) + AdaLN modulation
     * Note: Flux2 uses LayerNorm with elementwise_affine=False (no learned weights)
     * Vectorized using Accelerate framework on Apple platforms.
     */
#if defined(__APPLE__) && defined(USE_BLAS)
    /* Vectorized implementation using vDSP */
    for (int s = 0; s < seq; s++) {
        const float *x_row = x + s * hidden;
        float *out_row = out + s * hidden;

        /* Compute mean using vDSP_meanv */
        float mean;
        vDSP_meanv(x_row, 1, &mean, hidden);

        /* Compute variance: sum((x - mean)^2) / n */
        /* First subtract mean: temp = x - mean */
        vDSP_vsadd(x_row, 1, &(float){-mean}, out_row, 1, hidden);

        /* Then square: temp = temp^2 */
        vDSP_vsq(out_row, 1, out_row, 1, hidden);

        /* Sum the squares */
        float var_sum;
        vDSP_sve(out_row, 1, &var_sum, hidden);
        float var = var_sum / hidden;
        float std_inv = 1.0f / sqrtf(var + eps);

        /* Apply normalization: out = (x - mean) * std_inv */
        vDSP_vsadd(x_row, 1, &(float){-mean}, out_row, 1, hidden);
        vDSP_vsmul(out_row, 1, &std_inv, out_row, 1, hidden);

        /* Apply modulation: out = (1 + scale) * out + shift
         * Could use vDSP_vma but need temp buffer; scalar loop is fast enough */
        for (int i = 0; i < hidden; i++) {
            out_row[i] = (1.0f + scale[i]) * out_row[i] + shift[i];
        }
    }
#else
    /* Scalar fallback */
    for (int s = 0; s < seq; s++) {
        const float *x_row = x + s * hidden;
        float *out_row = out + s * hidden;

        /* Compute mean */
        float sum = 0.0f;
        for (int i = 0; i < hidden; i++) {
            sum += x_row[i];
        }
        float mean = sum / hidden;

        /* Compute variance */
        float var_sum = 0.0f;
        for (int i = 0; i < hidden; i++) {
            float diff = x_row[i] - mean;
            var_sum += diff * diff;
        }
        float var = var_sum / hidden;
        float std_inv = 1.0f / sqrtf(var + eps);

        /* Apply Layer Norm + AdaLN modulation */
        for (int i = 0; i < hidden; i++) {
            float norm = (x_row[i] - mean) * std_inv;
            out_row[i] = (1.0f + scale[i]) * norm + shift[i];
        }
    }
#endif
}

/* Apply QK normalization (RMSNorm per head)
 * Vectorized using Accelerate framework on Apple platforms. */
static void apply_qk_norm(float *q, float *k,
                          const float *q_weight, const float *k_weight,
                          int seq, int heads, int head_dim, float eps) {
#if defined(__APPLE__) && defined(USE_BLAS)
    /* Vectorized implementation using vDSP */
    for (int s = 0; s < seq; s++) {
        for (int h = 0; h < heads; h++) {
            /* Q normalization: x = x * rsqrt(mean(x^2) + eps) * weight */
            float *qh = q + s * heads * head_dim + h * head_dim;
            float sum_sq;
            vDSP_svesq(qh, 1, &sum_sq, head_dim);
            float rms_inv = 1.0f / sqrtf(sum_sq / head_dim + eps);

            /* Apply: qh = qh * rms_inv * q_weight */
            vDSP_vsmul(qh, 1, &rms_inv, qh, 1, head_dim);
            vDSP_vmul(qh, 1, q_weight, 1, qh, 1, head_dim);

            /* K normalization */
            float *kh = k + s * heads * head_dim + h * head_dim;
            vDSP_svesq(kh, 1, &sum_sq, head_dim);
            rms_inv = 1.0f / sqrtf(sum_sq / head_dim + eps);

            vDSP_vsmul(kh, 1, &rms_inv, kh, 1, head_dim);
            vDSP_vmul(kh, 1, k_weight, 1, kh, 1, head_dim);
        }
    }
#else
    /* Scalar fallback */
    for (int s = 0; s < seq; s++) {
        for (int h = 0; h < heads; h++) {
            /* Q normalization */
            float *qh = q + s * heads * head_dim + h * head_dim;
            float sum_sq = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                sum_sq += qh[d] * qh[d];
            }
            float rms_inv = 1.0f / sqrtf(sum_sq / head_dim + eps);
            for (int d = 0; d < head_dim; d++) {
                qh[d] = qh[d] * rms_inv * q_weight[d];
            }

            /* K normalization */
            float *kh = k + s * heads * head_dim + h * head_dim;
            sum_sq = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                sum_sq += kh[d] * kh[d];
            }
            rms_inv = 1.0f / sqrtf(sum_sq / head_dim + eps);
            for (int d = 0; d < head_dim; d++) {
                kh[d] = kh[d] * rms_inv * k_weight[d];
            }
        }
    }
#endif
}

/* ========================================================================
 * Attention Layer
 * ======================================================================== */

/* Multi-head self-attention */

#if defined(USE_METAL) || defined(USE_BLAS)
/* Transpose from [seq, heads, head_dim] to [heads, seq, head_dim]
 * Needed for batched attention that processes each head separately */
static void transpose_shd_to_hsd(float *out, const float *in,
                                  int seq, int heads, int head_dim) {
    for (int s = 0; s < seq; s++) {
        for (int h = 0; h < heads; h++) {
            const float *src = in + (s * heads + h) * head_dim;
            float *dst = out + (h * seq + s) * head_dim;
            memcpy(dst, src, head_dim * sizeof(float));
        }
    }
}

/* Transpose from [heads, seq, head_dim] to [seq, heads, head_dim] */
static void transpose_hsd_to_shd(float *out, const float *in,
                                  int seq, int heads, int head_dim) {
    for (int h = 0; h < heads; h++) {
        for (int s = 0; s < seq; s++) {
            const float *src = in + (h * seq + s) * head_dim;
            float *dst = out + (s * heads + h) * head_dim;
            memcpy(dst, src, head_dim * sizeof(float));
        }
    }
}
#endif /* USE_METAL || USE_BLAS */

/* Multi-head attention with BLAS optimization
 * Uses pre-allocated workspace buffers from transformer struct
 */
static void mha_forward(float *out, const float *q, const float *k, const float *v,
                        int seq, int heads, int head_dim, flux_transformer_t *tf) {
    float scale = 1.0f / sqrtf((float)head_dim);
    (void)heads; /* hidden = heads * head_dim, but we use tf->hidden_size */

#ifdef USE_METAL
    /* Try fused attention kernel first - operates directly on [seq, hidden] layout
     * This avoids CPU transpose overhead */
    if (flux_metal_attention_fused(out, q, k, v, seq, seq, tf->num_heads, head_dim, scale)) {
        return;  /* Success - no transpose needed */
    }

    /* Try GPU-accelerated batched attention when Metal is available */
    if (flux_metal_available()) {
        /* Use pre-allocated buffers from transformer */
        float *q_t = tf->attn_q_t;
        float *k_t = tf->attn_k_t;
        float *v_t = tf->attn_v_t;
        float *out_t = tf->attn_out_t;
        float *scores = tf->attn_scores;

        /* Transpose to [heads, seq, head_dim] for GPU batched attention */
        transpose_shd_to_hsd(q_t, q, seq, tf->num_heads, head_dim);
        transpose_shd_to_hsd(k_t, k, seq, tf->num_heads, head_dim);
        transpose_shd_to_hsd(v_t, v, seq, tf->num_heads, head_dim);

        flux_metal_attention(out_t, q_t, k_t, v_t, scores,
                             tf->num_heads, seq, seq, head_dim, scale);

        /* Transpose output back to [seq, heads, head_dim] */
        transpose_hsd_to_shd(out, out_t, seq, tf->num_heads, head_dim);
        return;
    }
#endif

    /* CPU fallback: Use BLAS-optimized attention (faster) or flash attention (memory-efficient) */
#ifdef USE_BLAS
    /* BLAS path: transpose + batched matrix multiply per head */
    {
        float *q_t = tf->attn_q_t;
        float *k_t = tf->attn_k_t;
        float *v_t = tf->attn_v_t;
        float *out_t = tf->attn_out_t;
        float *scores = tf->attn_scores;

        /* Transpose to [heads, seq, head_dim] for efficient BLAS operations */
        transpose_shd_to_hsd(q_t, q, seq, tf->num_heads, head_dim);
        transpose_shd_to_hsd(k_t, k, seq, tf->num_heads, head_dim);
        transpose_shd_to_hsd(v_t, v, seq, tf->num_heads, head_dim);

        /* Process each head with BLAS */
        for (int h = 0; h < tf->num_heads; h++) {
            float *qh = q_t + h * seq * head_dim;
            float *kh = k_t + h * seq * head_dim;
            float *vh = v_t + h * seq * head_dim;
            float *oh = out_t + h * seq * head_dim;
            float *sh = scores + h * seq * seq;

            /* scores = Q @ K^T using BLAS */
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        seq, seq, head_dim,
                        scale, qh, head_dim, kh, head_dim,
                        0.0f, sh, seq);

            /* Softmax */
            flux_softmax(sh, seq, seq);

            /* out = scores @ V using BLAS */
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        seq, head_dim, seq,
                        1.0f, sh, seq, vh, head_dim,
                        0.0f, oh, head_dim);
        }

        /* Transpose output back to [seq, heads, head_dim] */
        transpose_hsd_to_shd(out, out_t, seq, tf->num_heads, head_dim);
    }
#else
    /* Generic fallback: Use flash attention (memory-efficient, no transpose needed) */
    flux_flash_attention(out, q, k, v, seq, seq, tf->num_heads, head_dim, scale);
#endif
}

/* Joint attention (for double blocks) - image and text attend to each other
 * Uses pre-allocated workspace buffers from transformer struct
 */
static void joint_attention(float *img_out, float *txt_out,
                            const float *img_q, const float *img_k, const float *img_v,
                            const float *txt_q, const float *txt_k, const float *txt_v,
                            int img_seq, int txt_seq, int heads, int head_dim,
                            flux_transformer_t *tf) {
    int total_seq = img_seq + txt_seq;
    int hidden = heads * head_dim;
    float scale = 1.0f / sqrtf((float)head_dim);

    /* Use pre-allocated buffers for K/V concatenation */
    float *cat_k = tf->attn_cat_k;
    float *cat_v = tf->attn_cat_v;

    /* Concatenate K, V from both streams in [seq, heads, head_dim] format
     * IMPORTANT: Python (official Flux2) concatenates as [TEXT, IMAGE]
     */
    memcpy(cat_k, txt_k, txt_seq * hidden * sizeof(float));
    memcpy(cat_v, txt_v, txt_seq * hidden * sizeof(float));
    memcpy(cat_k + txt_seq * hidden, img_k, img_seq * hidden * sizeof(float));
    memcpy(cat_v + txt_seq * hidden, img_v, img_seq * hidden * sizeof(float));

#ifdef USE_METAL
    /* Try fused attention kernel first - operates directly on [seq, hidden] layout
     * This avoids CPU transpose overhead */
    if (flux_metal_attention_fused(img_out, img_q, cat_k, cat_v,
                                   img_seq, total_seq, heads, head_dim, scale) &&
        flux_metal_attention_fused(txt_out, txt_q, cat_k, cat_v,
                                   txt_seq, total_seq, heads, head_dim, scale)) {
        return;  /* Success - no transpose needed */
    }

    /* Try GPU-accelerated batched attention when Metal is available */
    if (flux_metal_available()) {
        /* Use pre-allocated buffers for transposed data */
        float *img_q_t = tf->attn_q_t;
        float *txt_q_t = tf->attn_q_t + img_seq * hidden;
        float *cat_k_t = tf->attn_k_t;
        float *cat_v_t = tf->attn_v_t;
        float *img_out_t = tf->attn_out_t;
        float *txt_out_t = tf->attn_out_t + img_seq * hidden;
        float *scores = tf->attn_scores;

        /* Transpose to [heads, seq, head_dim] for GPU batched attention */
        transpose_shd_to_hsd(img_q_t, img_q, img_seq, heads, head_dim);
        transpose_shd_to_hsd(txt_q_t, txt_q, txt_seq, heads, head_dim);
        transpose_shd_to_hsd(cat_k_t, cat_k, total_seq, heads, head_dim);
        transpose_shd_to_hsd(cat_v_t, cat_v, total_seq, heads, head_dim);

        /* Image attention: img_Q @ cat_K^T, softmax, @ cat_V */
        flux_metal_attention(img_out_t, img_q_t, cat_k_t, cat_v_t, scores,
                             heads, img_seq, total_seq, head_dim, scale);
        /* Text attention: txt_Q @ cat_K^T, softmax, @ cat_V */
        flux_metal_attention(txt_out_t, txt_q_t, cat_k_t, cat_v_t, scores,
                             heads, txt_seq, total_seq, head_dim, scale);

        /* Transpose outputs back */
        transpose_hsd_to_shd(img_out, img_out_t, img_seq, heads, head_dim);
        transpose_hsd_to_shd(txt_out, txt_out_t, txt_seq, heads, head_dim);
        return;
    }
#endif

    /* CPU fallback: Use BLAS-optimized attention (faster) or flash attention (memory-efficient) */
#ifdef USE_BLAS
    /* BLAS path: transpose + batched matrix multiply per head */
    {
        float *img_q_t = tf->attn_q_t;
        float *txt_q_t = tf->attn_q_t + img_seq * hidden;
        float *cat_k_t = tf->attn_k_t;
        float *cat_v_t = tf->attn_v_t;
        float *img_out_t = tf->attn_out_t;
        float *txt_out_t = tf->attn_out_t + img_seq * hidden;
        float *scores = tf->attn_scores;

        /* Transpose to [heads, seq, head_dim] for efficient BLAS operations */
        transpose_shd_to_hsd(img_q_t, img_q, img_seq, heads, head_dim);
        transpose_shd_to_hsd(txt_q_t, txt_q, txt_seq, heads, head_dim);
        transpose_shd_to_hsd(cat_k_t, cat_k, total_seq, heads, head_dim);
        transpose_shd_to_hsd(cat_v_t, cat_v, total_seq, heads, head_dim);

        /* Process each head with BLAS */
        for (int h = 0; h < heads; h++) {
            float *img_qh = img_q_t + h * img_seq * head_dim;
            float *txt_qh = txt_q_t + h * txt_seq * head_dim;
            float *kh = cat_k_t + h * total_seq * head_dim;
            float *vh = cat_v_t + h * total_seq * head_dim;
            float *img_oh = img_out_t + h * img_seq * head_dim;
            float *txt_oh = txt_out_t + h * txt_seq * head_dim;
            float *img_sh = scores;  /* Reuse scores buffer */
            float *txt_sh = scores + img_seq * total_seq;

            /* Image attention: img_Q @ cat_K^T */
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        img_seq, total_seq, head_dim,
                        scale, img_qh, head_dim, kh, head_dim,
                        0.0f, img_sh, total_seq);
            flux_softmax(img_sh, img_seq, total_seq);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        img_seq, head_dim, total_seq,
                        1.0f, img_sh, total_seq, vh, head_dim,
                        0.0f, img_oh, head_dim);

            /* Text attention: txt_Q @ cat_K^T */
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        txt_seq, total_seq, head_dim,
                        scale, txt_qh, head_dim, kh, head_dim,
                        0.0f, txt_sh, total_seq);
            flux_softmax(txt_sh, txt_seq, total_seq);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        txt_seq, head_dim, total_seq,
                        1.0f, txt_sh, total_seq, vh, head_dim,
                        0.0f, txt_oh, head_dim);
        }

        /* Transpose outputs back */
        transpose_hsd_to_shd(img_out, img_out_t, img_seq, heads, head_dim);
        transpose_hsd_to_shd(txt_out, txt_out_t, txt_seq, heads, head_dim);
    }
#else
    /* Generic fallback: Use flash attention (memory-efficient, no transpose needed) */
    flux_flash_attention(img_out, img_q, cat_k, cat_v,
                         img_seq, total_seq, heads, head_dim, scale);
    flux_flash_attention(txt_out, txt_q, cat_k, cat_v,
                         txt_seq, total_seq, heads, head_dim, scale);
#endif
}

/* ========================================================================
 * SwiGLU FFN
 * ======================================================================== */

/* SwiGLU FFN with optional bf16 weights - uses pre-allocated buffers from tf */
static void swiglu_ffn_bf16(float *out, const float *x,
                            const float *gate_weight, const float *up_weight,
                            const float *down_weight,
                            const uint16_t *gate_weight_bf16,
                            const uint16_t *up_weight_bf16,
                            const uint16_t *down_weight_bf16,
                            int seq, int hidden, int mlp_hidden,
                            flux_transformer_t *tf) {
    /* Use pre-allocated FFN work buffers */
    float *gate = tf->ffn_gate;
    float *up = tf->ffn_up;

    /* Gate and up projections - these are independent, batch them */
    flux_gpu_begin_batch();
    LINEAR_BF16_OR_F32(gate, x, gate_weight, gate_weight_bf16, seq, hidden, mlp_hidden);
    LINEAR_BF16_OR_F32(up, x, up_weight, up_weight_bf16, seq, hidden, mlp_hidden);
    flux_gpu_end_batch();

    /* SiLU(gate) * up */
    flux_silu(gate, seq * mlp_hidden);
    flux_mul_inplace(gate, up, seq * mlp_hidden);

    /* Down projection */
    LINEAR_BF16_OR_F32(out, gate, down_weight, down_weight_bf16, seq, mlp_hidden, hidden);

    /* No free - using pre-allocated buffers */
}

/* ========================================================================
 * Double-Stream Block (MM-DiT)
 * ======================================================================== */

static void double_block_forward(float *img_hidden, float *txt_hidden,
                                 const double_block_t *block,
                                 const float *t_emb,
                                 const float *img_adaln_weight,
                                 const float *txt_adaln_weight,
                                 const float *img_rope_cos, const float *img_rope_sin,
                                 const float *txt_rope_cos, const float *txt_rope_sin,
                                 int img_seq, int txt_seq,
                                 flux_transformer_t *tf) {
    int hidden = tf->hidden_size;
    int heads = tf->num_heads;
    int head_dim = tf->head_dim;
    int mlp_hidden = tf->mlp_hidden;
    float eps = 1e-6f;

    /* Compute AdaLN parameters (6 per stream: shift1, scale1, gate1, shift2, scale2, gate2)
     * adaln_weight is [hidden*6, hidden], t_emb is [hidden]
     * Output: 6 * hidden parameters per stream
     * FLUX applies SiLU to t_emb before the modulation projection
     */
    int mod_size = hidden * 6;

    /* Apply SiLU to t_emb for modulation - use pre-allocated buffer */
    float *t_emb_silu = tf->t_emb_silu;
    for (int i = 0; i < hidden; i++) {
        float x = t_emb[i];
        t_emb_silu[i] = x / (1.0f + expf(-x));  /* SiLU = x * sigmoid(x) */
    }

    /* Image stream modulation - use pre-allocated buffer */
    float *img_mod = tf->double_mod_img;
    flux_linear_nobias(img_mod, t_emb_silu, img_adaln_weight, 1, hidden, mod_size);

    float *img_shift1 = img_mod;
    float *img_scale1 = img_mod + hidden;
    float *img_gate1 = img_mod + hidden * 2;
    float *img_shift2 = img_mod + hidden * 3;
    float *img_scale2 = img_mod + hidden * 4;
    float *img_gate2 = img_mod + hidden * 5;

    /* Text stream modulation - use pre-allocated buffer */
    float *txt_mod = tf->double_mod_txt;
    flux_linear_nobias(txt_mod, t_emb_silu, txt_adaln_weight, 1, hidden, mod_size);

    float *txt_shift1 = txt_mod;
    float *txt_scale1 = txt_mod + hidden;
    float *txt_gate1 = txt_mod + hidden * 2;
    float *txt_shift2 = txt_mod + hidden * 3;
    float *txt_scale2 = txt_mod + hidden * 4;
    float *txt_gate2 = txt_mod + hidden * 5;

    /* Image stream: AdaLN -> QKV -> QK-norm -> RoPE */
    float *img_norm = tf->work1;
    apply_adaln(img_norm, img_hidden, img_shift1, img_scale1, img_seq, hidden, eps);

#ifdef DEBUG_DOUBLE_BLOCK
    static int block_idx = 0;
    if (block_idx == 0) {
        fprintf(stderr, "\n[DBL] img_mod[0,:10] (shift1): ");
        for (int i = 0; i < 10; i++) fprintf(stderr, "%.6f ", img_shift1[i]);
        fprintf(stderr, "\n[DBL] img_mod[0,3072:3082] (scale1): ");
        for (int i = 0; i < 10; i++) fprintf(stderr, "%.6f ", img_scale1[i]);
        fprintf(stderr, "\n[DBL] After AdaLN img_norm[0,0,:10]: ");
        for (int i = 0; i < 10; i++) fprintf(stderr, "%.6f ", img_norm[i]);
        fprintf(stderr, "\n");
    }
#endif

    /* Separate Q, K, V projections (fixes interleaved output bug)
     * Note: These 3 projections are independent - batch them for GPU efficiency */
    float *img_q = tf->work2;
    float *img_k = img_q + img_seq * hidden;
    float *img_v = img_k + img_seq * hidden;

    flux_gpu_begin_batch();
    LINEAR_BF16_OR_F32(img_q, img_norm, block->img_q_weight, block->img_q_weight_bf16,
                       img_seq, hidden, hidden);
    LINEAR_BF16_OR_F32(img_k, img_norm, block->img_k_weight, block->img_k_weight_bf16,
                       img_seq, hidden, hidden);
    LINEAR_BF16_OR_F32(img_v, img_norm, block->img_v_weight, block->img_v_weight_bf16,
                       img_seq, hidden, hidden);
    flux_gpu_end_batch();

    /* Apply QK normalization (per-head RMSNorm) */
    apply_qk_norm(img_q, img_k, block->img_norm_q_weight, block->img_norm_k_weight,
                  img_seq, heads, head_dim, eps);

    /* Apply 2D RoPE to image Q, K (using h, w positions) */
    int axis_dim = 32;
    apply_rope_2d(img_q, img_rope_cos, img_rope_sin, img_seq, heads, head_dim, axis_dim);
    apply_rope_2d(img_k, img_rope_cos, img_rope_sin, img_seq, heads, head_dim, axis_dim);

#ifdef DEBUG_DOUBLE_BLOCK
    if (block_idx == 0) {
        fprintf(stderr, "[DBL] After Q proj img_q[0,0,:5]: ");
        for (int i = 0; i < 5; i++) fprintf(stderr, "%.6f ", img_q[i]);
        fprintf(stderr, "\n[DBL] After RoPE img_q[0,0,:5]: ");
        for (int i = 0; i < 5; i++) fprintf(stderr, "%.6f ", img_q[i]);
        fprintf(stderr, "\n");
    }
#endif

    /* Text stream: AdaLN -> QKV -> QK-norm -> RoPE */
    float *txt_norm = img_norm + img_seq * hidden;
    apply_adaln(txt_norm, txt_hidden, txt_shift1, txt_scale1, txt_seq, hidden, eps);

    /* Separate Q, K, V projections for text
     * Note: These 3 projections are independent - batch them for GPU efficiency */
    float *txt_q = img_v + img_seq * hidden;  /* After img_v */
    float *txt_k = txt_q + txt_seq * hidden;
    float *txt_v = txt_k + txt_seq * hidden;

    flux_gpu_begin_batch();
    LINEAR_BF16_OR_F32(txt_q, txt_norm, block->txt_q_weight, block->txt_q_weight_bf16,
                       txt_seq, hidden, hidden);
    LINEAR_BF16_OR_F32(txt_k, txt_norm, block->txt_k_weight, block->txt_k_weight_bf16,
                       txt_seq, hidden, hidden);
    LINEAR_BF16_OR_F32(txt_v, txt_norm, block->txt_v_weight, block->txt_v_weight_bf16,
                       txt_seq, hidden, hidden);
    flux_gpu_end_batch();

    /* Apply QK normalization */
    apply_qk_norm(txt_q, txt_k, block->txt_norm_q_weight, block->txt_norm_k_weight,
                  txt_seq, heads, head_dim, eps);

    /* Apply text RoPE - text tokens have position IDs (0, 0, 0, L) where L is sequence index
     * This applies rotation in axis 3 (dims 96-127)
     */
    apply_rope_2d(txt_q, txt_rope_cos, txt_rope_sin, txt_seq, heads, head_dim, axis_dim);
    apply_rope_2d(txt_k, txt_rope_cos, txt_rope_sin, txt_seq, heads, head_dim, axis_dim);

    /* Joint attention - use pre-allocated buffers */
    float *img_attn_out = tf->double_img_attn_out;
    float *txt_attn_out = tf->double_txt_attn_out;

    joint_attention(img_attn_out, txt_attn_out,
                    img_q, img_k, img_v,
                    txt_q, txt_k, txt_v,
                    img_seq, txt_seq, heads, head_dim, tf);

#ifdef DEBUG_DOUBLE_BLOCK
    if (block_idx == 0) {
        fprintf(stderr, "[DBL] After attn img_attn_out[0,0,:5]: ");
        for (int i = 0; i < 5; i++) fprintf(stderr, "%.6f ", img_attn_out[i]);
        fprintf(stderr, "\n");
    }
#endif

    /* Project attention output
     * Note: img_proj and txt_proj are independent - batch them for GPU efficiency */
    float *img_proj = tf->work1;
    float *txt_proj = img_proj + img_seq * hidden;

    flux_gpu_begin_batch();
    LINEAR_BF16_OR_F32(img_proj, img_attn_out, block->img_proj_weight, block->img_proj_weight_bf16,
                       img_seq, hidden, hidden);
    LINEAR_BF16_OR_F32(txt_proj, txt_attn_out, block->txt_proj_weight, block->txt_proj_weight_bf16,
                       txt_seq, hidden, hidden);
    flux_gpu_end_batch();

#ifdef DEBUG_DOUBLE_BLOCK
    if (block_idx == 0) {
        fprintf(stderr, "[DBL] After proj img_proj[0,0,:5]: ");
        for (int i = 0; i < 5; i++) fprintf(stderr, "%.6f ", img_proj[i]);
        fprintf(stderr, "\n");
        fprintf(stderr, "[DBL] gate1[0:5]: ");
        for (int i = 0; i < 5; i++) fprintf(stderr, "%.6f ", img_gate1[i]);
        fprintf(stderr, "\n");
    }
#endif

    /* Apply gate and add residual - use vectorized helper */
    gated_add(img_hidden, img_gate1, img_proj, img_seq, hidden);
    gated_add(txt_hidden, txt_gate1, txt_proj, txt_seq, hidden);

#ifdef DEBUG_DOUBLE_BLOCK
    if (block_idx == 0) {
        fprintf(stderr, "[DBL] After attn residual img_hidden[0,0,:5]: ");
        for (int i = 0; i < 5; i++) fprintf(stderr, "%.6f ", img_hidden[i]);
        fprintf(stderr, "\n");
    }
#endif

    /* FFN for image */
    apply_adaln(img_norm, img_hidden, img_shift2, img_scale2, img_seq, hidden, eps);

#ifdef DEBUG_DOUBLE_BLOCK
    if (block_idx == 0) {
        fprintf(stderr, "[DBL] FFN input (after AdaLN) img_norm[0,0,:5]: ");
        for (int i = 0; i < 5; i++) fprintf(stderr, "%.6f ", img_norm[i]);
        fprintf(stderr, "\n");
    }
#endif

    swiglu_ffn_bf16(img_proj, img_norm,
                    block->img_mlp_gate_weight, block->img_mlp_up_weight,
                    block->img_mlp_down_weight,
                    block->img_mlp_gate_weight_bf16, block->img_mlp_up_weight_bf16,
                    block->img_mlp_down_weight_bf16,
                    img_seq, hidden, mlp_hidden, tf);

#ifdef DEBUG_DOUBLE_BLOCK
    if (block_idx == 0) {
        fprintf(stderr, "[DBL] FFN output img_proj[0,0,:5]: ");
        for (int i = 0; i < 5; i++) fprintf(stderr, "%.6f ", img_proj[i]);
        fprintf(stderr, "\n");
        fprintf(stderr, "[DBL] gate2[0:5]: ");
        for (int i = 0; i < 5; i++) fprintf(stderr, "%.6f ", img_gate2[i]);
        fprintf(stderr, "\n");
    }
#endif

    gated_add(img_hidden, img_gate2, img_proj, img_seq, hidden);

#ifdef DEBUG_DOUBLE_BLOCK
    fprintf(stderr, "[DBL%d] After FFN residual img_hidden[0,0,:5]: ", block_idx);
    for (int i = 0; i < 5; i++) fprintf(stderr, "%.6f ", img_hidden[i]);
    fprintf(stderr, "\n");
#endif

    /* FFN for text */
    apply_adaln(txt_norm, txt_hidden, txt_shift2, txt_scale2, txt_seq, hidden, eps);
    swiglu_ffn_bf16(txt_proj, txt_norm,
                    block->txt_mlp_gate_weight, block->txt_mlp_up_weight,
                    block->txt_mlp_down_weight,
                    block->txt_mlp_gate_weight_bf16, block->txt_mlp_up_weight_bf16,
                    block->txt_mlp_down_weight_bf16,
                    txt_seq, hidden, mlp_hidden, tf);
    gated_add(txt_hidden, txt_gate2, txt_proj, txt_seq, hidden);

    /* No free - using pre-allocated buffers */

#ifdef DEBUG_DOUBLE_BLOCK
    block_idx++;
#endif
}

/* ========================================================================
 * Single-Stream Block (Parallel DiT)
 * ======================================================================== */

#ifdef USE_METAL
/* GPU-optimized single block forward using persistent GPU tensors
 * Keeps activations on GPU throughout the block to minimize memory transfers.
 * Returns 1 if GPU path was used, 0 to fall back to CPU path.
 */
static int single_block_forward_gpu(float *hidden, const single_block_t *block,
                                    const float *t_emb, const float *adaln_weight,
                                    const float *img_rope_cos, const float *img_rope_sin,
                                    const float *txt_rope_cos, const float *txt_rope_sin,
                                    int seq, int img_offset, flux_transformer_t *tf) {
    /* Check if GPU tensors are available */
    if (!flux_metal_available() || !flux_metal_shaders_available()) return 0;
    if (block->qkv_mlp_weight_bf16 == NULL) return 0;  /* Need bf16 weights */

    int h_size = tf->hidden_size;
    int heads = tf->num_heads;
    int head_dim = tf->head_dim;
    int mlp_hidden = tf->mlp_hidden;
    float eps = 1e-6f;
    int axis_dim = 32;

    /* === Phase 1: AdaLN modulation (small, keep on CPU) === */
    int mod_size = h_size * 3;
    float *t_emb_silu = tf->t_emb_silu;
    for (int i = 0; i < h_size; i++) {
        float x = t_emb[i];
        t_emb_silu[i] = x / (1.0f + expf(-x));
    }
    float *mod_params = tf->work2 + tf->max_seq_len * h_size * 3;
    flux_linear_nobias(mod_params, t_emb_silu, adaln_weight, 1, h_size, mod_size);

    float *shift = mod_params;
    float *scale = mod_params + h_size;
    float *gate = mod_params + h_size * 2;

    /* === Phase 2: Create GPU tensors and enter batch mode === */
    flux_gpu_batch_begin();

    /* Create input tensor on GPU */
    flux_gpu_tensor_t hidden_gpu = flux_gpu_tensor_create(hidden, seq * h_size);
    if (!hidden_gpu) {
        flux_gpu_batch_end();
        return 0;
    }

    /* Allocate output tensors */
    flux_gpu_tensor_t norm_gpu = flux_gpu_tensor_alloc(seq * h_size);
    int fused_dim = h_size * 3 + mlp_hidden * 2;
    flux_gpu_tensor_t fused_gpu = flux_gpu_tensor_alloc(seq * fused_dim);
    flux_gpu_tensor_t q_gpu = flux_gpu_tensor_alloc(seq * h_size);
    flux_gpu_tensor_t k_gpu = flux_gpu_tensor_alloc(seq * h_size);
    flux_gpu_tensor_t v_gpu = flux_gpu_tensor_alloc(seq * h_size);
    flux_gpu_tensor_t gate_gpu = flux_gpu_tensor_alloc(seq * mlp_hidden);
    flux_gpu_tensor_t up_gpu = flux_gpu_tensor_alloc(seq * mlp_hidden);
    flux_gpu_tensor_t attn_out_gpu = flux_gpu_tensor_alloc(seq * h_size);
    flux_gpu_tensor_t concat_gpu = flux_gpu_tensor_alloc(seq * (h_size + mlp_hidden));
    flux_gpu_tensor_t proj_out_gpu = flux_gpu_tensor_alloc(seq * h_size);

    if (!norm_gpu || !fused_gpu || !q_gpu || !k_gpu || !v_gpu ||
        !gate_gpu || !up_gpu || !attn_out_gpu || !concat_gpu || !proj_out_gpu) {
        /* Cleanup and fall back */
        if (hidden_gpu) flux_gpu_tensor_free(hidden_gpu);
        if (norm_gpu) flux_gpu_tensor_free(norm_gpu);
        if (fused_gpu) flux_gpu_tensor_free(fused_gpu);
        if (q_gpu) flux_gpu_tensor_free(q_gpu);
        if (k_gpu) flux_gpu_tensor_free(k_gpu);
        if (v_gpu) flux_gpu_tensor_free(v_gpu);
        if (gate_gpu) flux_gpu_tensor_free(gate_gpu);
        if (up_gpu) flux_gpu_tensor_free(up_gpu);
        if (attn_out_gpu) flux_gpu_tensor_free(attn_out_gpu);
        if (concat_gpu) flux_gpu_tensor_free(concat_gpu);
        if (proj_out_gpu) flux_gpu_tensor_free(proj_out_gpu);
        flux_gpu_batch_end();
        return 0;
    }

    /* === Phase 3: AdaLN normalization on GPU === */
    flux_gpu_adaln_norm(norm_gpu, hidden_gpu, shift, scale, seq, h_size, eps);

    /* === Phase 4: Fused QKV + MLP projection on GPU === */
    flux_gpu_tensor_t fused_result = flux_gpu_linear_bf16(norm_gpu,
                                                          block->qkv_mlp_weight_bf16,
                                                          seq, h_size, fused_dim);
    if (!fused_result) {
        /* Cleanup and fall back */
        flux_gpu_tensor_free(hidden_gpu);
        flux_gpu_tensor_free(norm_gpu);
        flux_gpu_tensor_free(fused_gpu);
        flux_gpu_tensor_free(q_gpu);
        flux_gpu_tensor_free(k_gpu);
        flux_gpu_tensor_free(v_gpu);
        flux_gpu_tensor_free(gate_gpu);
        flux_gpu_tensor_free(up_gpu);
        flux_gpu_tensor_free(attn_out_gpu);
        flux_gpu_tensor_free(concat_gpu);
        flux_gpu_tensor_free(proj_out_gpu);
        flux_gpu_batch_end();
        return 0;
    }

    /* === Phase 5: Split fused output on GPU === */
    flux_gpu_split_qkv_mlp(fused_result, q_gpu, k_gpu, v_gpu, gate_gpu, up_gpu,
                           seq, h_size, mlp_hidden);

    /* === Phase 6: QK RMSNorm on GPU === */
    flux_gpu_qk_rms_norm(q_gpu, k_gpu, block->norm_q_weight, block->norm_k_weight,
                         seq, heads, head_dim, eps);

    /* === Phase 7: Apply unified RoPE on GPU (handles text+image in one call) === */
    flux_gpu_rope_unified(q_gpu, k_gpu,
                          txt_rope_cos, txt_rope_sin,
                          img_rope_cos, img_rope_sin,
                          seq, img_offset, heads, head_dim, axis_dim);

    /* === Phase 8: Self-attention on GPU === */
    float attn_scale = 1.0f / sqrtf((float)head_dim);
    if (!flux_gpu_attention_fused(attn_out_gpu, q_gpu, k_gpu, v_gpu,
                                  seq, seq, heads, head_dim, attn_scale)) {
        /* Fall back to CPU attention - need to sync and copy */
        flux_gpu_batch_end();
        float *q_cpu = tf->single_q;
        float *k_cpu = tf->single_k;
        float *v_cpu = tf->single_v;
        flux_gpu_tensor_read(q_gpu, q_cpu);
        flux_gpu_tensor_read(k_gpu, k_cpu);
        flux_gpu_tensor_read(v_gpu, v_cpu);
        float *attn_out_cpu = tf->single_attn_out;
        mha_forward(attn_out_cpu, q_cpu, k_cpu, v_cpu, seq, heads, head_dim, tf);
        memcpy(flux_gpu_tensor_data(attn_out_gpu), attn_out_cpu, seq * h_size * sizeof(float));
        flux_gpu_batch_begin();
    }

    /* === Phase 9: SwiGLU on GPU === */
    flux_gpu_silu_mul(gate_gpu, up_gpu, seq * mlp_hidden);

    /* === Phase 10: Concat attention + MLP outputs on GPU === */
    flux_gpu_concat_attn_mlp(attn_out_gpu, gate_gpu, concat_gpu, seq, h_size, mlp_hidden);

    /* === Phase 11: Final projection on GPU === */
    /* Free pre-allocated tensor since linear returns a new one */
    flux_gpu_tensor_free(proj_out_gpu);
    proj_out_gpu = flux_gpu_linear_bf16(concat_gpu, block->proj_mlp_weight_bf16,
                                        seq, h_size + mlp_hidden, h_size);
    if (!proj_out_gpu) {
        /* Fall back to CPU projection */
        flux_gpu_batch_end();
        float *concat_cpu = tf->single_concat;
        flux_gpu_tensor_read(concat_gpu, concat_cpu);
        float *proj_out_cpu = tf->work1;
        flux_linear_nobias_bf16(proj_out_cpu, concat_cpu, block->proj_mlp_weight_bf16,
                                seq, h_size + mlp_hidden, h_size);
        gated_add(hidden, gate, proj_out_cpu, seq, h_size);
        goto cleanup;
    }

    /* === Phase 12: Gated add residual on GPU === */
    flux_gpu_gated_add(hidden_gpu, gate, proj_out_gpu, seq, h_size);

    /* === Phase 13: Sync and copy result back === */
    flux_gpu_batch_end();
    flux_gpu_tensor_read(hidden_gpu, hidden);

cleanup:
    /* === Cleanup GPU tensors === */
    flux_gpu_tensor_free(hidden_gpu);
    flux_gpu_tensor_free(norm_gpu);
    flux_gpu_tensor_free(fused_result);
    flux_gpu_tensor_free(q_gpu);
    flux_gpu_tensor_free(k_gpu);
    flux_gpu_tensor_free(v_gpu);
    flux_gpu_tensor_free(gate_gpu);
    flux_gpu_tensor_free(up_gpu);
    flux_gpu_tensor_free(attn_out_gpu);
    flux_gpu_tensor_free(concat_gpu);
    flux_gpu_tensor_free(proj_out_gpu);

    return 1;  /* GPU path succeeded */
}
#endif /* USE_METAL */

static void single_block_forward(float *hidden, const single_block_t *block,
                                 const float *t_emb, const float *adaln_weight,
                                 const float *img_rope_cos, const float *img_rope_sin,
                                 const float *txt_rope_cos, const float *txt_rope_sin,
                                 int seq, int img_offset, flux_transformer_t *tf) {
    /* seq = total_seq (txt + img)
     * img_offset = txt_seq (where image starts in the [txt, img] concatenation)
     */
    int h_size = tf->hidden_size;
    int heads = tf->num_heads;
    int head_dim = tf->head_dim;
    int mlp_hidden = tf->mlp_hidden;
    int img_seq = seq - img_offset;  /* Number of image tokens */
    float eps = 1e-6f;

    /* Compute AdaLN parameters (3: shift, scale, gate)
     * adaln_weight is [hidden*3, hidden], t_emb is [hidden]
     * FLUX applies SiLU to t_emb before the modulation projection
     */
    int mod_size = h_size * 3;

    /* Apply SiLU to t_emb for modulation - use pre-allocated buffer */
    float *t_emb_silu = tf->t_emb_silu;
    for (int i = 0; i < h_size; i++) {
        float x = t_emb[i];
        t_emb_silu[i] = x / (1.0f + expf(-x));
    }

    /* Use end of work2 for mod_params (3*hidden = 9216 floats, work2 has max_seq*hidden*4) */
    float *mod_params = tf->work2 + tf->max_seq_len * h_size * 3;
    flux_linear_nobias(mod_params, t_emb_silu, adaln_weight, 1, h_size, mod_size);

    float *shift = mod_params;
    float *scale = mod_params + h_size;
    float *gate = mod_params + h_size * 2;

    /* Norm */
    float *norm = tf->work1;
    apply_adaln(norm, hidden, shift, scale, seq, h_size, eps);

    /* Fused QKV + FFN input projection
     * Output: [seq, fused_dim] where fused_dim = [Q, K, V, gate, up]
     * Layout per position: [3072 Q, 3072 K, 3072 V, 9216 gate, 9216 up] = 27648 total
     */
    int fused_dim = h_size * 3 + mlp_hidden * 2;
    float *fused_out = tf->work2;
    LINEAR_BF16_OR_F32(fused_out, norm, block->qkv_mlp_weight, block->qkv_mlp_weight_bf16,
                       seq, h_size, fused_dim);

    /* Split outputs: use pre-allocated buffers
     * Each position has [Q, K, V, gate, up] concatenated
     */
    float *q = tf->single_q;
    float *k = tf->single_k;
    float *v = tf->single_v;
    float *mlp_gate = tf->single_mlp_gate;
    float *mlp_up = tf->single_mlp_up;

    for (int s = 0; s < seq; s++) {
        float *row = fused_out + s * fused_dim;
        memcpy(q + s * h_size, row, h_size * sizeof(float));
        memcpy(k + s * h_size, row + h_size, h_size * sizeof(float));
        memcpy(v + s * h_size, row + h_size * 2, h_size * sizeof(float));
        memcpy(mlp_gate + s * mlp_hidden, row + h_size * 3, mlp_hidden * sizeof(float));
        memcpy(mlp_up + s * mlp_hidden, row + h_size * 3 + mlp_hidden, mlp_hidden * sizeof(float));
    }

    /* Apply QK normalization */
    apply_qk_norm(q, k, block->norm_q_weight, block->norm_k_weight,
                  seq, heads, head_dim, eps);

    /* Apply RoPE: layout is [txt, img]
     * - Text portion (0 to img_offset-1): RoPE in axis 3 (L dimension)
     * - Image portion (img_offset to seq-1): 2D RoPE based on H/W positions
     */
    int axis_dim = 32;
    int txt_seq = img_offset;

    /* Text portion: apply RoPE in axis 3 (L dimension = sequence position) */
    apply_rope_2d(q, txt_rope_cos, txt_rope_sin, txt_seq, heads, head_dim, axis_dim);
    apply_rope_2d(k, txt_rope_cos, txt_rope_sin, txt_seq, heads, head_dim, axis_dim);

    /* Image portion: apply 2D RoPE starting at img_offset */
    float *img_q = q + img_offset * h_size;
    float *img_k = k + img_offset * h_size;
    apply_rope_2d(img_q, img_rope_cos, img_rope_sin, img_seq, heads, head_dim, axis_dim);
    apply_rope_2d(img_k, img_rope_cos, img_rope_sin, img_seq, heads, head_dim, axis_dim);

    /* Self-attention - use pre-allocated buffer */
    float *attn_out = tf->single_attn_out;
    mha_forward(attn_out, q, k, v, seq, heads, head_dim, tf);

    /* SwiGLU: silu(gate) * up */
    flux_silu(mlp_gate, seq * mlp_hidden);
    flux_mul_inplace(mlp_gate, mlp_up, seq * mlp_hidden);

    /* Fused output projection: [attn_out, mlp_out] -> hidden
     * proj_mlp_weight: [hidden, hidden + mlp_hidden]
     * Use pre-allocated concat buffer
     */
    float *concat = tf->single_concat;
    for (int s = 0; s < seq; s++) {
        memcpy(concat + s * (h_size + mlp_hidden),
               attn_out + s * h_size, h_size * sizeof(float));
        memcpy(concat + s * (h_size + mlp_hidden) + h_size,
               mlp_gate + s * mlp_hidden, mlp_hidden * sizeof(float));
    }

    float *proj_out = tf->work1;
    LINEAR_BF16_OR_F32(proj_out, concat, block->proj_mlp_weight, block->proj_mlp_weight_bf16,
                       seq, h_size + mlp_hidden, h_size);

    /* Apply gate and add residual - use vectorized helper */
    gated_add(hidden, gate, proj_out, seq, h_size);

    /* No free - using pre-allocated buffers */
}

/* ========================================================================
 * Full Transformer Forward Pass
 * ======================================================================== */

float *flux_transformer_forward(flux_transformer_t *tf,
                                const float *img_latent, int img_h, int img_w,
                                const float *txt_emb, int txt_seq,
                                float timestep) {
    int hidden = tf->hidden_size;
    int img_seq = img_h * img_w;
    int head_dim = tf->head_dim;
    int axis_dim = 32;  /* FLUX uses axes_dims_rope: [32, 32, 32, 32] */

    /* Get timestep embedding
     * FLUX.2-klein uses 256-dim sinusoidal (128 frequencies), not hidden_size
     * Use t_emb_silu as work buffer (it's not used until double blocks)
     */
    int sincos_dim = tf->time_embed.sincos_dim;
    float *t_emb = (float *)malloc(hidden * sizeof(float));
    float t_sincos[256];  /* sincos_dim is always 256 */
    get_timestep_embedding(t_sincos, timestep * 1000.0f, sincos_dim, 10000.0f);
    time_embed_forward(t_emb, t_sincos, &tf->time_embed, hidden, tf->t_emb_silu);

    /* Compute 2D RoPE frequencies for image tokens based on actual dimensions
     * img_h, img_w are the patch grid dimensions (e.g., 4x4 for 64x64 image)
     */
    /* Allocate RoPE: 4 axes * 32 dims = 128 dims per position (matches head_dim) */
    float *img_rope_cos = (float *)malloc(img_seq * axis_dim * 4 * sizeof(float));
    float *img_rope_sin = (float *)malloc(img_seq * axis_dim * 4 * sizeof(float));
    compute_rope_2d(img_rope_cos, img_rope_sin, img_h, img_w, axis_dim, tf->rope_theta);

    /* Compute text RoPE frequencies - text tokens have position IDs (0, 0, 0, L)
     * where L is the sequence index. RoPE is applied in axis 3 (dims 96-127)
     */
    float *txt_rope_cos = (float *)malloc(txt_seq * head_dim * sizeof(float));
    float *txt_rope_sin = (float *)malloc(txt_seq * head_dim * sizeof(float));
    compute_rope_text(txt_rope_cos, txt_rope_sin, txt_seq, axis_dim, tf->rope_theta);

    /* Transpose input from NCHW [channels, h, w] to NLC [seq, channels] format
     * Input: img_latent[c * img_seq + pos] for channel c at position pos
     * Output: transposed[pos * channels + c]
     */
    int channels = tf->latent_channels;
    float *img_transposed = (float *)malloc(img_seq * channels * sizeof(float));
    for (int pos = 0; pos < img_seq; pos++) {
        for (int c = 0; c < channels; c++) {
            img_transposed[pos * channels + c] = img_latent[c * img_seq + pos];
        }
    }

    /* Project image latent to hidden */
    float *img_hidden = tf->img_hidden;
    LINEAR_BF16_OR_F32(img_hidden, img_transposed, tf->img_in_weight, tf->img_in_weight_bf16,
                       img_seq, tf->latent_channels, hidden);
    free(img_transposed);

    /* Project text embeddings to hidden */
    float *txt_hidden = tf->txt_hidden;
    LINEAR_BF16_OR_F32(txt_hidden, txt_emb, tf->txt_in_weight, tf->txt_in_weight_bf16,
                       txt_seq, tf->text_dim, hidden);

#ifdef DEBUG_TRANSFORMER
    /* Debug: print intermediate values for comparison with Python */
    fprintf(stderr, "\n[DEBUG] t_emb first 10: ");
    for (int i = 0; i < 10; i++) fprintf(stderr, "%.6f ", t_emb[i]);
    fprintf(stderr, "\n");

    fprintf(stderr, "[DEBUG] img_hidden[0,0,:5] (img_proj): ");
    for (int i = 0; i < 5; i++) fprintf(stderr, "%.6f ", img_hidden[i]);
    fprintf(stderr, "\n");

    fprintf(stderr, "[DEBUG] txt_hidden[0,0,:5] (txt_proj): ");
    for (int i = 0; i < 5; i++) fprintf(stderr, "%.6f ", txt_hidden[i]);
    fprintf(stderr, "\n");

    fprintf(stderr, "[DEBUG] img_rope_cos[0, :10]: ");
    for (int i = 0; i < 10; i++) fprintf(stderr, "%.6f ", img_rope_cos[i]);
    fprintf(stderr, "\n");

    fprintf(stderr, "[DEBUG] txt_rope_cos[10, 96:106]: ");
    for (int i = 96; i < 106; i++) fprintf(stderr, "%.6f ", txt_rope_cos[10 * head_dim + i]);
    fprintf(stderr, "\n");
#endif

    /* Double-stream blocks */
    double double_start = tf_get_time_ms();
    for (int i = 0; i < tf->num_double_layers; i++) {
        double_block_forward(img_hidden, txt_hidden,
                             &tf->double_blocks[i],
                             t_emb,
                             tf->adaln_double_img_weight,
                             tf->adaln_double_txt_weight,
                             img_rope_cos, img_rope_sin,
                             txt_rope_cos, txt_rope_sin,
                             img_seq, txt_seq, tf);
        if (flux_substep_callback)
            flux_substep_callback(FLUX_SUBSTEP_DOUBLE_BLOCK, i, tf->num_double_layers);
#ifdef DEBUG_TRANSFORMER
        if (i == 0) {
            fprintf(stderr, "\n[DEBUG] After double block 0:\n");
            fprintf(stderr, "[DEBUG] img_hidden[0,0,:10]: ");
            for (int d = 0; d < 10; d++) fprintf(stderr, "%.6f ", img_hidden[d]);
            fprintf(stderr, "\n");
            float sum = 0, sum_sq = 0;
            for (int d = 0; d < img_seq * hidden; d++) {
                sum += img_hidden[d];
                sum_sq += img_hidden[d] * img_hidden[d];
            }
            float mean = sum / (img_seq * hidden);
            float std = sqrtf(sum_sq / (img_seq * hidden) - mean * mean);
            fprintf(stderr, "[DEBUG] img_hidden mean=%.6f, std=%.6f\n", mean, std);
        }
#endif
    }

    double double_time = tf_get_time_ms() - double_start;

    /* Concatenate text and image for single-stream blocks
     * Python uses [txt, img] order for concatenation
     */
    int total_seq = img_seq + txt_seq;
    float *concat_hidden = (float *)malloc(total_seq * hidden * sizeof(float));
    memcpy(concat_hidden, txt_hidden, txt_seq * hidden * sizeof(float));
    memcpy(concat_hidden + txt_seq * hidden, img_hidden,
           img_seq * hidden * sizeof(float));

    /* Single-stream blocks */
    double single_start = tf_get_time_ms();
    for (int i = 0; i < tf->num_single_layers; i++) {
#ifdef USE_METAL
        /* Try GPU-optimized path first */
        if (!single_block_forward_gpu(concat_hidden, &tf->single_blocks[i],
                                      t_emb, tf->adaln_single_weight,
                                      img_rope_cos, img_rope_sin,
                                      txt_rope_cos, txt_rope_sin,
                                      total_seq, txt_seq, tf))
#endif
        {
            /* Fall back to CPU path */
            single_block_forward(concat_hidden, &tf->single_blocks[i],
                                 t_emb, tf->adaln_single_weight,
                                 img_rope_cos, img_rope_sin,
                                 txt_rope_cos, txt_rope_sin,
                                 total_seq, txt_seq, tf);  /* txt_seq is the offset to image */
        }
        if (flux_substep_callback)
            flux_substep_callback(FLUX_SUBSTEP_SINGLE_BLOCK, i, tf->num_single_layers);

#ifdef DEBUG_SINGLE_BLOCK
        if (i == 0 || i == 9 || i == 19) {
            /* Print image portion (starts at txt_seq offset) */
            float *img_part = concat_hidden + txt_seq * hidden;
            fprintf(stderr, "[SGL%d] img_hidden[0,0,:5]: ", i);
            for (int d = 0; d < 5; d++) fprintf(stderr, "%.6f ", img_part[d]);
            fprintf(stderr, "\n");
        }
#endif
    }
    double single_time = tf_get_time_ms() - single_start;

    /* Extract image hidden states (image is after text) */
    memcpy(img_hidden, concat_hidden + txt_seq * hidden, img_seq * hidden * sizeof(float));
    free(concat_hidden);

#ifdef DEBUG_FINAL_LAYER
    fprintf(stderr, "[FINAL] Before final layer img_hidden[0,0,:5]: ");
    for (int d = 0; d < 5; d++) fprintf(stderr, "%.6f ", img_hidden[d]);
    fprintf(stderr, "\n");
#endif

    /* Final layer: AdaLN modulation -> project to latent channels
     * norm_out.linear.weight is [6144, 3072] = [shift, scale] projection
     * Apply SiLU to t_emb before modulation projection (FLUX architecture)
     */
    double final_start = tf_get_time_ms();
    /* Reuse pre-allocated t_emb_silu buffer */
    for (int i = 0; i < hidden; i++) {
        float x = t_emb[i];
        tf->t_emb_silu[i] = x / (1.0f + expf(-x));
    }

    /* Reuse double_mod_img buffer for final_mod (needs hidden*2, has hidden*6) */
    float *final_mod = tf->double_mod_img;
    flux_linear_nobias(final_mod, tf->t_emb_silu, tf->final_norm_weight, 1, hidden, hidden * 2);

    /* Python: scale, shift = mod.chunk(2, dim=1) - scale is first half, shift is second half */
    float *final_scale = final_mod;
    float *final_shift = final_mod + hidden;

    float *final_norm = tf->work1;
    apply_adaln(final_norm, img_hidden, final_shift, final_scale, img_seq, hidden, 1e-6f);

    float *output_nlc = (float *)malloc(img_seq * tf->latent_channels * sizeof(float));
    LINEAR_BF16_OR_F32(output_nlc, final_norm, tf->final_proj_weight, tf->final_proj_weight_bf16,
                       img_seq, hidden, tf->latent_channels);

    /* Transpose output from NLC [seq, channels] to NCHW [channels, h, w] format
     * Input: output_nlc[pos * channels + c]
     * Output: output[c * img_seq + pos]
     */
    float *output = (float *)malloc(img_seq * tf->latent_channels * sizeof(float));
    for (int pos = 0; pos < img_seq; pos++) {
        for (int c = 0; c < channels; c++) {
            output[c * img_seq + pos] = output_nlc[pos * channels + c];
        }
    }
    free(output_nlc);

    free(t_emb);
    free(img_rope_cos);
    free(img_rope_sin);
    free(txt_rope_cos);
    free(txt_rope_sin);

    double final_time = tf_get_time_ms() - final_start;

    /* Update global timing counters */
    flux_timing_transformer_double += double_time;
    flux_timing_transformer_single += single_time;
    flux_timing_transformer_final += final_time;
    flux_timing_transformer_total += double_time + single_time + final_time;

    if (flux_substep_callback)
        flux_substep_callback(FLUX_SUBSTEP_FINAL_LAYER, 0, 1);

    return output;
}

/* ========================================================================
 * Transformer Loading
 * ======================================================================== */

static float *read_floats(FILE *f, int count) {
    float *data = (float *)malloc(count * sizeof(float));
    if (!data) return NULL;
    if (fread(data, sizeof(float), count, f) != (size_t)count) {
        free(data);
        return NULL;
    }
    return data;
}

flux_transformer_t *flux_transformer_load(FILE *f) {
    flux_transformer_t *tf = calloc(1, sizeof(flux_transformer_t));
    if (!tf) return NULL;

    /* Read config */
    uint32_t config[10];
    if (fread(config, sizeof(uint32_t), 10, f) != 10) goto error;

    tf->hidden_size = config[0];
    tf->num_heads = config[1];
    tf->head_dim = config[2];
    tf->mlp_hidden = config[3];
    tf->num_double_layers = config[4];
    tf->num_single_layers = config[5];
    tf->text_dim = config[6];
    tf->latent_channels = config[7];
    tf->max_seq_len = config[8];
    tf->rope_dim = config[9];

    float rope_theta;
    if (fread(&rope_theta, sizeof(float), 1, f) != 1) goto error;
    tf->rope_theta = rope_theta;

    /* Read input projections */
    tf->img_in_weight = read_floats(f, tf->hidden_size * tf->latent_channels);
    tf->txt_in_weight = read_floats(f, tf->hidden_size * tf->text_dim);

    /* Read time embedding (binary format - deprecated, use safetensors) */
    tf->time_embed.sincos_dim = 256;  /* Match safetensors model */
    tf->time_embed.fc1_weight = read_floats(f, tf->hidden_size * 256);
    tf->time_embed.fc2_weight = read_floats(f, tf->hidden_size * tf->hidden_size);

    /* Read double blocks (binary format - deprecated, use safetensors) */
    tf->double_blocks = calloc(tf->num_double_layers, sizeof(double_block_t));
    for (int i = 0; i < tf->num_double_layers; i++) {
        double_block_t *b = &tf->double_blocks[i];
        int h = tf->hidden_size;
        int mlp = tf->mlp_hidden;
        int head_dim = tf->head_dim;

        /* QK norm weights (per head) */
        b->img_norm_q_weight = read_floats(f, head_dim);
        b->img_norm_k_weight = read_floats(f, head_dim);
        b->img_q_weight = read_floats(f, h * h);
        b->img_k_weight = read_floats(f, h * h);
        b->img_v_weight = read_floats(f, h * h);
        b->img_proj_weight = read_floats(f, h * h);
        b->img_mlp_gate_weight = read_floats(f, mlp * h);
        b->img_mlp_up_weight = read_floats(f, mlp * h);
        b->img_mlp_down_weight = read_floats(f, h * mlp);

        b->txt_norm_q_weight = read_floats(f, head_dim);
        b->txt_norm_k_weight = read_floats(f, head_dim);
        b->txt_q_weight = read_floats(f, h * h);
        b->txt_k_weight = read_floats(f, h * h);
        b->txt_v_weight = read_floats(f, h * h);
        b->txt_proj_weight = read_floats(f, h * h);
        b->txt_mlp_gate_weight = read_floats(f, mlp * h);
        b->txt_mlp_up_weight = read_floats(f, mlp * h);
        b->txt_mlp_down_weight = read_floats(f, h * mlp);
    }

    /* Read single blocks (binary format - deprecated, use safetensors) */
    tf->single_blocks = calloc(tf->num_single_layers, sizeof(single_block_t));
    for (int i = 0; i < tf->num_single_layers; i++) {
        single_block_t *b = &tf->single_blocks[i];
        int h = tf->hidden_size;
        int mlp = tf->mlp_hidden;
        int head_dim = tf->head_dim;

        b->norm_q_weight = read_floats(f, head_dim);
        b->norm_k_weight = read_floats(f, head_dim);
        b->qkv_mlp_weight = read_floats(f, (h * 3 + mlp * 2) * h);
        b->proj_mlp_weight = read_floats(f, h * (h + mlp));
    }

    /* Read final layer */
    tf->final_norm_weight = read_floats(f, tf->hidden_size);
    tf->final_proj_weight = read_floats(f, tf->latent_channels * tf->hidden_size);

    /* Precompute RoPE frequencies */
    tf->rope_freqs = (float *)malloc(tf->max_seq_len * tf->head_dim * sizeof(float));
    compute_rope_freqs(tf->rope_freqs, tf->max_seq_len, tf->head_dim, tf->rope_theta);

    /* Allocate working memory */
    int max_seq = tf->max_seq_len;
    int hidden = tf->hidden_size;
    tf->img_hidden = (float *)malloc(max_seq * hidden * sizeof(float));
    tf->txt_hidden = (float *)malloc(max_seq * hidden * sizeof(float));
    tf->work_size = max_seq * hidden * 4 * sizeof(float);
    tf->work1 = (float *)malloc(tf->work_size);
    tf->work2 = (float *)malloc(tf->work_size);

    /* Pre-allocated attention workspaces to avoid malloc in hot path */
    tf->attn_q_t = (float *)malloc(max_seq * hidden * sizeof(float));
    tf->attn_k_t = (float *)malloc(max_seq * hidden * sizeof(float));
    tf->attn_v_t = (float *)malloc(max_seq * hidden * sizeof(float));
    tf->attn_out_t = (float *)malloc(max_seq * hidden * sizeof(float));
    /* For GPU batched attention, need space for all heads' scores simultaneously */
    tf->attn_scores = (float *)malloc((size_t)tf->num_heads * max_seq * max_seq * sizeof(float));
    tf->attn_cat_k = (float *)malloc(max_seq * hidden * sizeof(float));
    tf->attn_cat_v = (float *)malloc(max_seq * hidden * sizeof(float));

    return tf;

error:
    flux_transformer_free(tf);
    return NULL;
}

void flux_transformer_free(flux_transformer_t *tf) {
    if (!tf) return;

    free(tf->img_in_weight);
    free(tf->txt_in_weight);
    free(tf->time_embed.fc1_weight);
    free(tf->time_embed.fc2_weight);

    if (tf->double_blocks) {
        for (int i = 0; i < tf->num_double_layers; i++) {
            double_block_t *b = &tf->double_blocks[i];
            free(b->img_norm_q_weight);
            free(b->img_norm_k_weight);
            free(b->img_q_weight);
            free(b->img_k_weight);
            free(b->img_v_weight);
            free(b->img_proj_weight);
            free(b->img_mlp_gate_weight);
            free(b->img_mlp_up_weight);
            free(b->img_mlp_down_weight);
            free(b->txt_norm_q_weight);
            free(b->txt_norm_k_weight);
            free(b->txt_q_weight);
            free(b->txt_k_weight);
            free(b->txt_v_weight);
            free(b->txt_proj_weight);
            free(b->txt_mlp_gate_weight);
            free(b->txt_mlp_up_weight);
            free(b->txt_mlp_down_weight);
        }
        free(tf->double_blocks);
    }

    if (tf->single_blocks) {
        for (int i = 0; i < tf->num_single_layers; i++) {
            single_block_t *b = &tf->single_blocks[i];
            free(b->norm_q_weight);
            free(b->norm_k_weight);
            free(b->qkv_mlp_weight);
            free(b->proj_mlp_weight);
        }
        free(tf->single_blocks);
    }

    free(tf->final_norm_weight);
    free(tf->final_proj_weight);
    free(tf->rope_freqs);
    free(tf->img_hidden);
    free(tf->txt_hidden);
    free(tf->work1);
    free(tf->work2);
    free(tf->adaln_double_img_weight);
    free(tf->adaln_double_txt_weight);
    free(tf->adaln_single_weight);

    /* Free attention workspace buffers */
    free(tf->attn_q_t);
    free(tf->attn_k_t);
    free(tf->attn_v_t);
    free(tf->attn_out_t);
    free(tf->attn_scores);
    free(tf->attn_cat_k);
    free(tf->attn_cat_v);

    /* Free single-block work buffers */
    free(tf->single_q);
    free(tf->single_k);
    free(tf->single_v);
    free(tf->single_mlp_gate);
    free(tf->single_mlp_up);
    free(tf->single_attn_out);
    free(tf->single_concat);

    /* Free FFN work buffers */
    free(tf->ffn_gate);
    free(tf->ffn_up);

    /* Free double-block work buffers */
    free(tf->t_emb_silu);
    free(tf->double_mod_img);
    free(tf->double_mod_txt);
    free(tf->double_img_attn_out);
    free(tf->double_txt_attn_out);

    free(tf);
}

/* ========================================================================
 * Safetensors Loading
 * ======================================================================== */

static float *get_sf_tensor_tf(safetensors_file_t *sf, const char *name) {
    const safetensor_t *t = safetensors_find(sf, name);
    if (!t) {
        fprintf(stderr, "Error: required tensor %s not found\n", name);
        return NULL;
    }
    return safetensors_get_f32(sf, t);
}

/* Get tensor as bf16 (for GPU acceleration) */
static uint16_t *get_sf_tensor_bf16(safetensors_file_t *sf, const char *name) {
    const safetensor_t *t = safetensors_find(sf, name);
    if (!t) {
        return NULL;  /* Not an error - bf16 is optional */
    }
    if (!safetensor_is_bf16(t)) {
        return NULL;  /* Not bf16, will use f32 version */
    }
    return safetensors_get_bf16(sf, t);
}

flux_transformer_t *flux_transformer_load_safetensors(safetensors_file_t *sf) {
    flux_transformer_t *tf = calloc(1, sizeof(flux_transformer_t));
    if (!tf) return NULL;

    char name[256];

    /* Set config based on FLUX.2-klein-4B */
    tf->hidden_size = 3072;
    tf->num_heads = 24;
    tf->head_dim = 128;
    tf->mlp_hidden = 9216;
    tf->num_double_layers = 5;
    tf->num_single_layers = 20;
    tf->text_dim = 7680;
    tf->latent_channels = 128;
    /* Max sequence length must accommodate image + text tokens combined.
     * At 1024x1024: img_seq = (1024/8)^2 = 16384, txt_seq = 512, total = 16896
     * At 2048x2048: img_seq = (2048/8)^2 = 65536 (requires 17GB+ for attn scores!)
     * We set 18000 to support up to ~1024x1024 with margin.
     */
    tf->max_seq_len = 18000;
    tf->rope_dim = 128;
    tf->rope_theta = 2000.0f;

    /* Enable bf16 mode if Metal GPU is available */
#ifdef USE_METAL
    tf->use_bf16 = flux_metal_available();
    if (tf->use_bf16) {
        printf("Using bf16 weights for GPU acceleration\n");
    }
#else
    tf->use_bf16 = 0;
#endif

    int h = tf->hidden_size;
    int mlp = tf->mlp_hidden;

    /* Input projections */
    tf->img_in_weight = get_sf_tensor_tf(sf, "x_embedder.weight");
    tf->txt_in_weight = get_sf_tensor_tf(sf, "context_embedder.weight");
    if (tf->use_bf16) {
        tf->img_in_weight_bf16 = get_sf_tensor_bf16(sf, "x_embedder.weight");
        tf->txt_in_weight_bf16 = get_sf_tensor_bf16(sf, "context_embedder.weight");
    }

    /* Time embedding
     * FLUX.2-klein uses 256-dim sinusoidal embedding (128 frequencies)
     * linear_1: [3072, 256], linear_2: [3072, 3072]
     */
    tf->time_embed.sincos_dim = 256;
    tf->time_embed.fc1_weight = get_sf_tensor_tf(sf,
        "time_guidance_embed.timestep_embedder.linear_1.weight");
    tf->time_embed.fc2_weight = get_sf_tensor_tf(sf,
        "time_guidance_embed.timestep_embedder.linear_2.weight");

    /* Modulation weights */
    tf->adaln_double_img_weight = get_sf_tensor_tf(sf,
        "double_stream_modulation_img.linear.weight");
    tf->adaln_double_txt_weight = get_sf_tensor_tf(sf,
        "double_stream_modulation_txt.linear.weight");
    tf->adaln_single_weight = get_sf_tensor_tf(sf,
        "single_stream_modulation.linear.weight");
    if (tf->use_bf16) {
        tf->adaln_double_img_weight_bf16 = get_sf_tensor_bf16(sf,
            "double_stream_modulation_img.linear.weight");
        tf->adaln_double_txt_weight_bf16 = get_sf_tensor_bf16(sf,
            "double_stream_modulation_txt.linear.weight");
        tf->adaln_single_weight_bf16 = get_sf_tensor_bf16(sf,
            "single_stream_modulation.linear.weight");
    }

    /* Double blocks */
    tf->double_blocks = calloc(tf->num_double_layers, sizeof(double_block_t));
    for (int i = 0; i < tf->num_double_layers; i++) {
        double_block_t *b = &tf->double_blocks[i];

        /* Image attention - QK norm weights (always f32) */
        snprintf(name, sizeof(name), "transformer_blocks.%d.attn.norm_q.weight", i);
        b->img_norm_q_weight = get_sf_tensor_tf(sf, name);
        snprintf(name, sizeof(name), "transformer_blocks.%d.attn.norm_k.weight", i);
        b->img_norm_k_weight = get_sf_tensor_tf(sf, name);

        /* Image Q, K, V projections (separate) */
        snprintf(name, sizeof(name), "transformer_blocks.%d.attn.to_q.weight", i);
        b->img_q_weight = get_sf_tensor_tf(sf, name);
        if (tf->use_bf16) b->img_q_weight_bf16 = get_sf_tensor_bf16(sf, name);
        snprintf(name, sizeof(name), "transformer_blocks.%d.attn.to_k.weight", i);
        b->img_k_weight = get_sf_tensor_tf(sf, name);
        if (tf->use_bf16) b->img_k_weight_bf16 = get_sf_tensor_bf16(sf, name);
        snprintf(name, sizeof(name), "transformer_blocks.%d.attn.to_v.weight", i);
        b->img_v_weight = get_sf_tensor_tf(sf, name);
        if (tf->use_bf16) b->img_v_weight_bf16 = get_sf_tensor_bf16(sf, name);

        snprintf(name, sizeof(name), "transformer_blocks.%d.attn.to_out.0.weight", i);
        b->img_proj_weight = get_sf_tensor_tf(sf, name);
        if (tf->use_bf16) b->img_proj_weight_bf16 = get_sf_tensor_bf16(sf, name);

        /* Image FFN - linear_in contains gate and up fused (18432 = 2*9216) */
        snprintf(name, sizeof(name), "transformer_blocks.%d.ff.linear_in.weight", i);
        float *ff_in = get_sf_tensor_tf(sf, name);
        if (ff_in) {
            /* Split into gate and up */
            b->img_mlp_gate_weight = malloc(mlp * h * sizeof(float));
            b->img_mlp_up_weight = malloc(mlp * h * sizeof(float));
            memcpy(b->img_mlp_gate_weight, ff_in, mlp * h * sizeof(float));
            memcpy(b->img_mlp_up_weight, ff_in + mlp * h, mlp * h * sizeof(float));
            free(ff_in);
        }
        if (tf->use_bf16) {
            uint16_t *ff_in_bf16 = get_sf_tensor_bf16(sf, name);
            if (ff_in_bf16) {
                b->img_mlp_gate_weight_bf16 = malloc(mlp * h * sizeof(uint16_t));
                b->img_mlp_up_weight_bf16 = malloc(mlp * h * sizeof(uint16_t));
                memcpy(b->img_mlp_gate_weight_bf16, ff_in_bf16, mlp * h * sizeof(uint16_t));
                memcpy(b->img_mlp_up_weight_bf16, ff_in_bf16 + mlp * h, mlp * h * sizeof(uint16_t));
                free(ff_in_bf16);
            }
        }

        snprintf(name, sizeof(name), "transformer_blocks.%d.ff.linear_out.weight", i);
        b->img_mlp_down_weight = get_sf_tensor_tf(sf, name);
        if (tf->use_bf16) b->img_mlp_down_weight_bf16 = get_sf_tensor_bf16(sf, name);

        /* Text stream - QK norm weights (always f32) */
        snprintf(name, sizeof(name), "transformer_blocks.%d.attn.norm_added_q.weight", i);
        b->txt_norm_q_weight = get_sf_tensor_tf(sf, name);
        snprintf(name, sizeof(name), "transformer_blocks.%d.attn.norm_added_k.weight", i);
        b->txt_norm_k_weight = get_sf_tensor_tf(sf, name);

        /* Text Q, K, V projections (separate) */
        snprintf(name, sizeof(name), "transformer_blocks.%d.attn.add_q_proj.weight", i);
        b->txt_q_weight = get_sf_tensor_tf(sf, name);
        if (tf->use_bf16) b->txt_q_weight_bf16 = get_sf_tensor_bf16(sf, name);
        snprintf(name, sizeof(name), "transformer_blocks.%d.attn.add_k_proj.weight", i);
        b->txt_k_weight = get_sf_tensor_tf(sf, name);
        if (tf->use_bf16) b->txt_k_weight_bf16 = get_sf_tensor_bf16(sf, name);
        snprintf(name, sizeof(name), "transformer_blocks.%d.attn.add_v_proj.weight", i);
        b->txt_v_weight = get_sf_tensor_tf(sf, name);
        if (tf->use_bf16) b->txt_v_weight_bf16 = get_sf_tensor_bf16(sf, name);

        snprintf(name, sizeof(name), "transformer_blocks.%d.attn.to_add_out.weight", i);
        b->txt_proj_weight = get_sf_tensor_tf(sf, name);
        if (tf->use_bf16) b->txt_proj_weight_bf16 = get_sf_tensor_bf16(sf, name);

        snprintf(name, sizeof(name), "transformer_blocks.%d.ff_context.linear_in.weight", i);
        float *txt_ff_in = get_sf_tensor_tf(sf, name);
        if (txt_ff_in) {
            b->txt_mlp_gate_weight = malloc(mlp * h * sizeof(float));
            b->txt_mlp_up_weight = malloc(mlp * h * sizeof(float));
            memcpy(b->txt_mlp_gate_weight, txt_ff_in, mlp * h * sizeof(float));
            memcpy(b->txt_mlp_up_weight, txt_ff_in + mlp * h, mlp * h * sizeof(float));
            free(txt_ff_in);
        }
        if (tf->use_bf16) {
            uint16_t *txt_ff_in_bf16 = get_sf_tensor_bf16(sf, name);
            if (txt_ff_in_bf16) {
                b->txt_mlp_gate_weight_bf16 = malloc(mlp * h * sizeof(uint16_t));
                b->txt_mlp_up_weight_bf16 = malloc(mlp * h * sizeof(uint16_t));
                memcpy(b->txt_mlp_gate_weight_bf16, txt_ff_in_bf16, mlp * h * sizeof(uint16_t));
                memcpy(b->txt_mlp_up_weight_bf16, txt_ff_in_bf16 + mlp * h, mlp * h * sizeof(uint16_t));
                free(txt_ff_in_bf16);
            }
        }

        snprintf(name, sizeof(name), "transformer_blocks.%d.ff_context.linear_out.weight", i);
        b->txt_mlp_down_weight = get_sf_tensor_tf(sf, name);
        if (tf->use_bf16) b->txt_mlp_down_weight_bf16 = get_sf_tensor_bf16(sf, name);
    }

    /* Single blocks */
    tf->single_blocks = calloc(tf->num_single_layers, sizeof(single_block_t));
    for (int i = 0; i < tf->num_single_layers; i++) {
        single_block_t *b = &tf->single_blocks[i];

        /* QK norm weights (always f32, small) */
        snprintf(name, sizeof(name), "single_transformer_blocks.%d.attn.norm_q.weight", i);
        b->norm_q_weight = get_sf_tensor_tf(sf, name);
        snprintf(name, sizeof(name), "single_transformer_blocks.%d.attn.norm_k.weight", i);
        b->norm_k_weight = get_sf_tensor_tf(sf, name);

        /* Major linear weights - load bf16 version for GPU acceleration */
        snprintf(name, sizeof(name), "single_transformer_blocks.%d.attn.to_qkv_mlp_proj.weight", i);
        b->qkv_mlp_weight = get_sf_tensor_tf(sf, name);
        if (tf->use_bf16) {
            b->qkv_mlp_weight_bf16 = get_sf_tensor_bf16(sf, name);
        }

        snprintf(name, sizeof(name), "single_transformer_blocks.%d.attn.to_out.weight", i);
        b->proj_mlp_weight = get_sf_tensor_tf(sf, name);
        if (tf->use_bf16) {
            b->proj_mlp_weight_bf16 = get_sf_tensor_bf16(sf, name);
        }
    }

    /* Final layer */
    tf->final_norm_weight = get_sf_tensor_tf(sf, "norm_out.linear.weight");
    tf->final_proj_weight = get_sf_tensor_tf(sf, "proj_out.weight");
    if (tf->use_bf16) {
        tf->final_proj_weight_bf16 = get_sf_tensor_bf16(sf, "proj_out.weight");
    }

    /* Precompute RoPE frequencies */
    tf->rope_freqs = malloc(tf->max_seq_len * tf->head_dim * sizeof(float));
    if (tf->rope_freqs) {
        compute_rope_freqs(tf->rope_freqs, tf->max_seq_len, tf->head_dim, tf->rope_theta);
    }

    /* Allocate working memory */
    int max_seq = tf->max_seq_len;
    int hidden = tf->hidden_size;
    tf->img_hidden = malloc(max_seq * hidden * sizeof(float));
    tf->txt_hidden = malloc(max_seq * hidden * sizeof(float));
    tf->work_size = max_seq * hidden * 4 * sizeof(float);
    tf->work1 = malloc(tf->work_size);
    tf->work2 = malloc(tf->work_size);

    /* Pre-allocated attention workspaces to avoid malloc in hot path */
    tf->attn_q_t = malloc(max_seq * hidden * sizeof(float));
    tf->attn_k_t = malloc(max_seq * hidden * sizeof(float));
    tf->attn_v_t = malloc(max_seq * hidden * sizeof(float));
    tf->attn_out_t = malloc(max_seq * hidden * sizeof(float));
    /* For GPU batched attention, need space for all heads' scores simultaneously */
    tf->attn_scores = malloc((size_t)tf->num_heads * max_seq * max_seq * sizeof(float));
    tf->attn_cat_k = malloc(max_seq * hidden * sizeof(float));
    tf->attn_cat_v = malloc(max_seq * hidden * sizeof(float));

    /* Single-block work buffers (pre-allocated to avoid malloc in hot path) */
    tf->single_q = malloc(max_seq * hidden * sizeof(float));
    tf->single_k = malloc(max_seq * hidden * sizeof(float));
    tf->single_v = malloc(max_seq * hidden * sizeof(float));
    tf->single_mlp_gate = malloc((size_t)max_seq * mlp * sizeof(float));
    tf->single_mlp_up = malloc((size_t)max_seq * mlp * sizeof(float));
    tf->single_attn_out = malloc(max_seq * hidden * sizeof(float));
    tf->single_concat = malloc((size_t)max_seq * (hidden + mlp) * sizeof(float));

    /* FFN work buffers (shared by double and single blocks) */
    tf->ffn_gate = malloc((size_t)max_seq * mlp * sizeof(float));
    tf->ffn_up = malloc((size_t)max_seq * mlp * sizeof(float));

    /* Double-block work buffers */
    tf->t_emb_silu = malloc(hidden * sizeof(float));
    tf->double_mod_img = malloc(hidden * 6 * sizeof(float));
    tf->double_mod_txt = malloc(hidden * 6 * sizeof(float));
    tf->double_img_attn_out = malloc(max_seq * hidden * sizeof(float));
    tf->double_txt_attn_out = malloc(max_seq * hidden * sizeof(float));

    if (!tf->img_hidden || !tf->txt_hidden || !tf->work1 || !tf->work2 ||
        !tf->attn_q_t || !tf->attn_k_t || !tf->attn_v_t || !tf->attn_out_t ||
        !tf->attn_scores || !tf->attn_cat_k || !tf->attn_cat_v ||
        !tf->single_q || !tf->single_k || !tf->single_v ||
        !tf->single_mlp_gate || !tf->single_mlp_up || !tf->single_attn_out ||
        !tf->single_concat || !tf->ffn_gate || !tf->ffn_up ||
        !tf->t_emb_silu || !tf->double_mod_img || !tf->double_mod_txt ||
        !tf->double_img_attn_out || !tf->double_txt_attn_out) {
        flux_transformer_free(tf);
        return NULL;
    }

    return tf;
}
