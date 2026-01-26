/*
 * Qwen3 Text Generation with KV Cache
 *
 * Implements autoregressive generation for Qwen3-4B using the same weights
 * as the FLUX text encoder.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#elif defined(__AVX512F__) && defined(__AVX512BW__)
#include <immintrin.h>
#define USE_AVX512
#elif defined(__AVX2__)
#include <immintrin.h>
#define USE_AVX2
#endif

#include "qwen3_generate.h"
#include "flux_qwen3.h"
#include "flux_safetensors.h"

/* ========================================================================
 * Constants
 * ======================================================================== */

#define GEN_INITIAL_SEQ_LEN 256   /* Initial KV cache capacity */
#define GEN_MAX_SEQ_LEN 131072    /* Absolute max (128K tokens, ~800MB KV cache) */

/* Special token IDs */
#define QWEN3_EOS_ID 151643       /* <|endoftext|> */
#define QWEN3_IM_START_ID 151644  /* <|im_start|> */
#define QWEN3_IM_END_ID 151645    /* <|im_end|> */

/* ========================================================================
 * KV Cache Structure
 * ======================================================================== */

typedef struct {
    float *k;  /* [max_seq, num_kv_heads * head_dim] */
    float *v;  /* [max_seq, num_kv_heads * head_dim] */
} kv_cache_layer_t;

typedef struct {
    kv_cache_layer_t *layers;  /* [num_layers] */
    int num_layers;
    int capacity;      /* Current allocated capacity */
    int cur_seq_len;   /* Current position in cache */
} kv_cache_t;

/* ========================================================================
 * Q8_0 Quantization (int8 weights with per-row scale)
 * Reduces memory bandwidth by 4x compared to f32
 * ======================================================================== */

typedef struct {
    int8_t *data;    /* [out_dim, in_dim] quantized weights */
    float *scale;    /* [out_dim] scale factors (one per row) */
    int out_dim;
    int in_dim;
} q8_weight_t;

/* Convert bf16 to f32 */
static inline float bf16_to_f32(uint16_t bf16) {
    uint32_t f32_bits = (uint32_t)bf16 << 16;
    float result;
    memcpy(&result, &f32_bits, sizeof(float));
    return result;
}

/* Quantize f32 weights to Q8_0 format */
static q8_weight_t *q8_quantize(const float *weights, int out_dim, int in_dim) {
    q8_weight_t *q = malloc(sizeof(q8_weight_t));
    if (!q) return NULL;

    q->out_dim = out_dim;
    q->in_dim = in_dim;
    q->data = malloc((size_t)out_dim * in_dim * sizeof(int8_t));
    q->scale = malloc(out_dim * sizeof(float));

    if (!q->data || !q->scale) {
        free(q->data);
        free(q->scale);
        free(q);
        return NULL;
    }

    /* Quantize each row independently */
    for (int i = 0; i < out_dim; i++) {
        const float *row = weights + i * in_dim;
        int8_t *qrow = q->data + i * in_dim;

        /* Find max absolute value in row */
        float max_abs = 0.0f;
        for (int j = 0; j < in_dim; j++) {
            float abs_val = fabsf(row[j]);
            if (abs_val > max_abs) max_abs = abs_val;
        }

        /* Compute scale (map max to 127) */
        float scale = max_abs / 127.0f;
        q->scale[i] = scale;

        /* Quantize row */
        if (scale > 0) {
            float inv_scale = 1.0f / scale;
            for (int j = 0; j < in_dim; j++) {
                float val = row[j] * inv_scale;
                /* Clamp and round */
                if (val > 127.0f) val = 127.0f;
                if (val < -127.0f) val = -127.0f;
                qrow[j] = (int8_t)roundf(val);
            }
        } else {
            /* All zeros */
            memset(qrow, 0, in_dim);
        }
    }

    return q;
}

static void q8_free(q8_weight_t *q) {
    if (q) {
        free(q->data);
        free(q->scale);
        free(q);
    }
}

/* Quantize bf16 weights directly to Q8_0 format (no intermediate f32 allocation) */
static q8_weight_t *q8_quantize_bf16(const uint16_t *weights_bf16, int out_dim, int in_dim) {
    q8_weight_t *q = malloc(sizeof(q8_weight_t));
    if (!q) return NULL;

    q->out_dim = out_dim;
    q->in_dim = in_dim;
    q->data = malloc((size_t)out_dim * in_dim * sizeof(int8_t));
    q->scale = malloc(out_dim * sizeof(float));

    if (!q->data || !q->scale) {
        free(q->data);
        free(q->scale);
        free(q);
        return NULL;
    }

    /* Quantize each row independently */
    for (int i = 0; i < out_dim; i++) {
        const uint16_t *row_bf16 = weights_bf16 + i * in_dim;
        int8_t *qrow = q->data + i * in_dim;

        /* Find max absolute value in row (convert bf16->f32 on the fly) */
        float max_abs = 0.0f;
#if defined(__ARM_NEON)
        float32x4_t max_v = vdupq_n_f32(0.0f);
        int j = 0;
        for (; j + 7 < in_dim; j += 8) {
            uint16x8_t bf16_v = vld1q_u16(row_bf16 + j);
            uint32x4_t lo = vshll_n_u16(vget_low_u16(bf16_v), 16);
            uint32x4_t hi = vshll_n_u16(vget_high_u16(bf16_v), 16);
            float32x4_t f0 = vreinterpretq_f32_u32(lo);
            float32x4_t f1 = vreinterpretq_f32_u32(hi);
            max_v = vmaxq_f32(max_v, vabsq_f32(f0));
            max_v = vmaxq_f32(max_v, vabsq_f32(f1));
        }
        max_abs = vmaxvq_f32(max_v);
        for (; j < in_dim; j++) {
            float val = bf16_to_f32(row_bf16[j]);
            float abs_val = fabsf(val);
            if (abs_val > max_abs) max_abs = abs_val;
        }
#elif defined(USE_AVX512)
        __m512 max_v = _mm512_setzero_ps();
        __m512 sign_mask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFFFFFF));
        int j = 0;
        for (; j + 15 < in_dim; j += 16) {
            /* Load 16 bf16, convert to f32 by shifting left 16 bits */
            __m256i bf16_v = _mm256_loadu_si256((const __m256i*)(row_bf16 + j));
            __m512i f32_bits = _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_v), 16);
            __m512 f32_v = _mm512_castsi512_ps(f32_bits);
            __m512 abs_v = _mm512_and_ps(f32_v, sign_mask);
            max_v = _mm512_max_ps(max_v, abs_v);
        }
        max_abs = _mm512_reduce_max_ps(max_v);
        for (; j < in_dim; j++) {
            float val = bf16_to_f32(row_bf16[j]);
            float abs_val = fabsf(val);
            if (abs_val > max_abs) max_abs = abs_val;
        }
#elif defined(USE_AVX2)
        __m256 max_v = _mm256_setzero_ps();
        __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
        int j = 0;
        for (; j + 7 < in_dim; j += 8) {
            /* Load 8 bf16, convert to f32 */
            __m128i bf16_v = _mm_loadu_si128((const __m128i*)(row_bf16 + j));
            __m256i f32_bits = _mm256_slli_epi32(_mm256_cvtepu16_epi32(bf16_v), 16);
            __m256 f32_v = _mm256_castsi256_ps(f32_bits);
            __m256 abs_v = _mm256_and_ps(f32_v, sign_mask);
            max_v = _mm256_max_ps(max_v, abs_v);
        }
        /* Horizontal max */
        __m128 lo = _mm256_castps256_ps128(max_v);
        __m128 hi = _mm256_extractf128_ps(max_v, 1);
        lo = _mm_max_ps(lo, hi);
        lo = _mm_max_ps(lo, _mm_shuffle_ps(lo, lo, _MM_SHUFFLE(2, 3, 0, 1)));
        lo = _mm_max_ps(lo, _mm_shuffle_ps(lo, lo, _MM_SHUFFLE(1, 0, 3, 2)));
        max_abs = _mm_cvtss_f32(lo);
        for (; j < in_dim; j++) {
            float val = bf16_to_f32(row_bf16[j]);
            float abs_val = fabsf(val);
            if (abs_val > max_abs) max_abs = abs_val;
        }
#else
        for (int j = 0; j < in_dim; j++) {
            float val = bf16_to_f32(row_bf16[j]);
            float abs_val = fabsf(val);
            if (abs_val > max_abs) max_abs = abs_val;
        }
#endif

        /* Compute scale (map max to 127) */
        float scale = max_abs / 127.0f;
        q->scale[i] = scale;

        /* Quantize row */
        if (scale > 0) {
            float inv_scale = 1.0f / scale;
#if defined(__ARM_NEON)
            float32x4_t inv_scale_v = vdupq_n_f32(inv_scale);
            j = 0;
            for (; j + 7 < in_dim; j += 8) {
                uint16x8_t bf16_v = vld1q_u16(row_bf16 + j);
                uint32x4_t lo = vshll_n_u16(vget_low_u16(bf16_v), 16);
                uint32x4_t hi = vshll_n_u16(vget_high_u16(bf16_v), 16);
                float32x4_t f0 = vreinterpretq_f32_u32(lo);
                float32x4_t f1 = vreinterpretq_f32_u32(hi);
                /* Scale and convert to int32 */
                int32x4_t i0 = vcvtnq_s32_f32(vmulq_f32(f0, inv_scale_v));
                int32x4_t i1 = vcvtnq_s32_f32(vmulq_f32(f1, inv_scale_v));
                /* Narrow to int16 then int8 with saturation */
                int16x4_t s0 = vqmovn_s32(i0);
                int16x4_t s1 = vqmovn_s32(i1);
                int8x8_t b = vqmovn_s16(vcombine_s16(s0, s1));
                vst1_s8(qrow + j, b);
            }
            for (; j < in_dim; j++) {
                float val = bf16_to_f32(row_bf16[j]) * inv_scale;
                if (val > 127.0f) val = 127.0f;
                if (val < -127.0f) val = -127.0f;
                qrow[j] = (int8_t)roundf(val);
            }
#elif defined(USE_AVX512)
            __m512 inv_scale_v = _mm512_set1_ps(inv_scale);
            j = 0;
            for (; j + 15 < in_dim; j += 16) {
                /* Load 16 bf16, convert to f32 */
                __m256i bf16_v = _mm256_loadu_si256((const __m256i*)(row_bf16 + j));
                __m512i f32_bits = _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_v), 16);
                __m512 f32_v = _mm512_castsi512_ps(f32_bits);
                /* Scale and convert to int32 with rounding */
                __m512 scaled = _mm512_mul_ps(f32_v, inv_scale_v);
                __m512i i32 = _mm512_cvtps_epi32(scaled);
                /* Pack to int16 then int8 with saturation */
                __m256i i16 = _mm512_cvtsepi32_epi16(i32);
                __m128i i8 = _mm256_cvtsepi16_epi8(i16);
                _mm_storeu_si128((__m128i*)(qrow + j), i8);
            }
            for (; j < in_dim; j++) {
                float val = bf16_to_f32(row_bf16[j]) * inv_scale;
                if (val > 127.0f) val = 127.0f;
                if (val < -127.0f) val = -127.0f;
                qrow[j] = (int8_t)roundf(val);
            }
#elif defined(USE_AVX2)
            __m256 inv_scale_v = _mm256_set1_ps(inv_scale);
            j = 0;
            for (; j + 7 < in_dim; j += 8) {
                /* Load 8 bf16, convert to f32 */
                __m128i bf16_v = _mm_loadu_si128((const __m128i*)(row_bf16 + j));
                __m256i f32_bits = _mm256_slli_epi32(_mm256_cvtepu16_epi32(bf16_v), 16);
                __m256 f32_v = _mm256_castsi256_ps(f32_bits);
                /* Scale and convert to int32 with rounding */
                __m256 scaled = _mm256_mul_ps(f32_v, inv_scale_v);
                __m256i i32 = _mm256_cvtps_epi32(scaled);
                /* Pack to int16 then int8 with saturation */
                __m128i i16 = _mm_packs_epi32(_mm256_castsi256_si128(i32),
                                              _mm256_extracti128_si256(i32, 1));
                __m128i i8 = _mm_packs_epi16(i16, i16);
                _mm_storel_epi64((__m128i*)(qrow + j), i8);
            }
            for (; j < in_dim; j++) {
                float val = bf16_to_f32(row_bf16[j]) * inv_scale;
                if (val > 127.0f) val = 127.0f;
                if (val < -127.0f) val = -127.0f;
                qrow[j] = (int8_t)roundf(val);
            }
#else
            for (int j = 0; j < in_dim; j++) {
                float val = bf16_to_f32(row_bf16[j]) * inv_scale;
                if (val > 127.0f) val = 127.0f;
                if (val < -127.0f) val = -127.0f;
                qrow[j] = (int8_t)roundf(val);
            }
#endif
        } else {
            /* All zeros */
            memset(qrow, 0, in_dim);
        }
    }

    return q;
}

/* Quantize float vector to int8 with scale - for use with vdot */
static void quantize_vector_q8(const float *x, int8_t *x_q8, float *x_scale, int n) {
#if defined(__ARM_NEON)
    /* Find max absolute value using NEON */
    float32x4_t max_v = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 15 < n; i += 16) {
        float32x4_t v0 = vld1q_f32(x + i);
        float32x4_t v1 = vld1q_f32(x + i + 4);
        float32x4_t v2 = vld1q_f32(x + i + 8);
        float32x4_t v3 = vld1q_f32(x + i + 12);
        max_v = vmaxq_f32(max_v, vabsq_f32(v0));
        max_v = vmaxq_f32(max_v, vabsq_f32(v1));
        max_v = vmaxq_f32(max_v, vabsq_f32(v2));
        max_v = vmaxq_f32(max_v, vabsq_f32(v3));
    }
    float max_abs = vmaxvq_f32(max_v);
    for (; i < n; i++) {
        float abs_val = fabsf(x[i]);
        if (abs_val > max_abs) max_abs = abs_val;
    }

    /* Compute scale */
    *x_scale = max_abs / 127.0f;
    if (*x_scale == 0.0f) {
        memset(x_q8, 0, n);
        return;
    }

    /* Quantize using NEON: float -> int32 -> int16 -> int8 */
    float inv_scale = 127.0f / max_abs;
    float32x4_t scale_v = vdupq_n_f32(inv_scale);
    i = 0;
    for (; i + 15 < n; i += 16) {
        /* Load 16 floats */
        float32x4_t f0 = vld1q_f32(x + i);
        float32x4_t f1 = vld1q_f32(x + i + 4);
        float32x4_t f2 = vld1q_f32(x + i + 8);
        float32x4_t f3 = vld1q_f32(x + i + 12);

        /* Scale and convert to int32 (with rounding) */
        int32x4_t i0 = vcvtnq_s32_f32(vmulq_f32(f0, scale_v));
        int32x4_t i1 = vcvtnq_s32_f32(vmulq_f32(f1, scale_v));
        int32x4_t i2 = vcvtnq_s32_f32(vmulq_f32(f2, scale_v));
        int32x4_t i3 = vcvtnq_s32_f32(vmulq_f32(f3, scale_v));

        /* Narrow to int16 with saturation */
        int16x4_t s0 = vqmovn_s32(i0);
        int16x4_t s1 = vqmovn_s32(i1);
        int16x4_t s2 = vqmovn_s32(i2);
        int16x4_t s3 = vqmovn_s32(i3);
        int16x8_t s01 = vcombine_s16(s0, s1);
        int16x8_t s23 = vcombine_s16(s2, s3);

        /* Narrow to int8 with saturation */
        int8x8_t b0 = vqmovn_s16(s01);
        int8x8_t b1 = vqmovn_s16(s23);
        int8x16_t result = vcombine_s8(b0, b1);

        /* Store 16 int8s */
        vst1q_s8(x_q8 + i, result);
    }
    /* Handle remainder */
    for (; i < n; i++) {
        float val = x[i] * inv_scale;
        if (val > 127.0f) val = 127.0f;
        if (val < -127.0f) val = -127.0f;
        x_q8[i] = (int8_t)roundf(val);
    }
#elif defined(USE_AVX512)
    /* Find max absolute value using AVX-512 */
    __m512 max_v = _mm512_setzero_ps();
    __m512 sign_mask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFFFFFF));
    int i = 0;
    for (; i + 15 < n; i += 16) {
        __m512 v = _mm512_loadu_ps(x + i);
        __m512 abs_v = _mm512_and_ps(v, sign_mask);
        max_v = _mm512_max_ps(max_v, abs_v);
    }
    float max_abs = _mm512_reduce_max_ps(max_v);
    for (; i < n; i++) {
        float abs_val = fabsf(x[i]);
        if (abs_val > max_abs) max_abs = abs_val;
    }

    /* Compute scale */
    *x_scale = max_abs / 127.0f;
    if (*x_scale == 0.0f) {
        memset(x_q8, 0, n);
        return;
    }

    /* Quantize using AVX-512: float -> int32 -> int16 -> int8 */
    float inv_scale = 127.0f / max_abs;
    __m512 scale_v = _mm512_set1_ps(inv_scale);
    i = 0;
    for (; i + 15 < n; i += 16) {
        /* Load 16 floats */
        __m512 f = _mm512_loadu_ps(x + i);
        /* Scale and convert to int32 with rounding */
        __m512i i32 = _mm512_cvtps_epi32(_mm512_mul_ps(f, scale_v));
        /* Pack to int16 then int8 with saturation */
        __m256i i16 = _mm512_cvtsepi32_epi16(i32);
        __m128i i8 = _mm256_cvtsepi16_epi8(i16);
        _mm_storeu_si128((__m128i*)(x_q8 + i), i8);
    }
    /* Handle remainder */
    for (; i < n; i++) {
        float val = x[i] * inv_scale;
        if (val > 127.0f) val = 127.0f;
        if (val < -127.0f) val = -127.0f;
        x_q8[i] = (int8_t)roundf(val);
    }
#elif defined(USE_AVX2)
    /* Find max absolute value using AVX2 */
    __m256 max_v = _mm256_setzero_ps();
    __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    int i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        __m256 abs_v = _mm256_and_ps(v, sign_mask);
        max_v = _mm256_max_ps(max_v, abs_v);
    }
    /* Horizontal max */
    __m128 lo = _mm256_castps256_ps128(max_v);
    __m128 hi = _mm256_extractf128_ps(max_v, 1);
    lo = _mm_max_ps(lo, hi);
    lo = _mm_max_ps(lo, _mm_shuffle_ps(lo, lo, _MM_SHUFFLE(2, 3, 0, 1)));
    lo = _mm_max_ps(lo, _mm_shuffle_ps(lo, lo, _MM_SHUFFLE(1, 0, 3, 2)));
    float max_abs = _mm_cvtss_f32(lo);
    for (; i < n; i++) {
        float abs_val = fabsf(x[i]);
        if (abs_val > max_abs) max_abs = abs_val;
    }

    /* Compute scale */
    *x_scale = max_abs / 127.0f;
    if (*x_scale == 0.0f) {
        memset(x_q8, 0, n);
        return;
    }

    /* Quantize using AVX2: float -> int32 -> int16 -> int8 */
    float inv_scale = 127.0f / max_abs;
    __m256 scale_v = _mm256_set1_ps(inv_scale);
    i = 0;
    for (; i + 7 < n; i += 8) {
        /* Load 8 floats */
        __m256 f = _mm256_loadu_ps(x + i);
        /* Scale and convert to int32 with rounding */
        __m256i i32 = _mm256_cvtps_epi32(_mm256_mul_ps(f, scale_v));
        /* Pack to int16 then int8 with saturation */
        __m128i i16 = _mm_packs_epi32(_mm256_castsi256_si128(i32),
                                       _mm256_extracti128_si256(i32, 1));
        __m128i i8 = _mm_packs_epi16(i16, i16);
        _mm_storel_epi64((__m128i*)(x_q8 + i), i8);
    }
    /* Handle remainder */
    for (; i < n; i++) {
        float val = x[i] * inv_scale;
        if (val > 127.0f) val = 127.0f;
        if (val < -127.0f) val = -127.0f;
        x_q8[i] = (int8_t)roundf(val);
    }
#else
    /* Scalar fallback */
    float max_abs = 0.0f;
    for (int i = 0; i < n; i++) {
        float abs_val = fabsf(x[i]);
        if (abs_val > max_abs) max_abs = abs_val;
    }

    *x_scale = max_abs / 127.0f;
    if (*x_scale == 0.0f) {
        memset(x_q8, 0, n);
        return;
    }

    float inv_scale = 1.0f / *x_scale;
    for (int i = 0; i < n; i++) {
        float val = x[i] * inv_scale;
        if (val > 127.0f) val = 127.0f;
        if (val < -127.0f) val = -127.0f;
        x_q8[i] = (int8_t)roundf(val);
    }
#endif
}

/* Q8 linear with pre-quantized input (avoids redundant quantization and malloc) */
static void linear_q8_preq(float *y, const int8_t *x_q8, float x_scale,
                           const q8_weight_t *W) {
    int out_dim = W->out_dim;
    int in_dim = W->in_dim;

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    /* Use vsdotq_s32: 4 int8×int8 -> int32 dot product per instruction */
    for (int i = 0; i < out_dim; i++) {
        const int8_t *wrow = W->data + i * in_dim;
        float scale = W->scale[i] * x_scale;

        int32x4_t acc0 = vdupq_n_s32(0);
        int32x4_t acc1 = vdupq_n_s32(0);
        int32x4_t acc2 = vdupq_n_s32(0);
        int32x4_t acc3 = vdupq_n_s32(0);

        int j = 0;
        for (; j + 63 < in_dim; j += 64) {
            /* Load 64 int8 from x and W */
            int8x16_t x0 = vld1q_s8(x_q8 + j);
            int8x16_t x1 = vld1q_s8(x_q8 + j + 16);
            int8x16_t x2 = vld1q_s8(x_q8 + j + 32);
            int8x16_t x3 = vld1q_s8(x_q8 + j + 48);

            int8x16_t w0 = vld1q_s8(wrow + j);
            int8x16_t w1 = vld1q_s8(wrow + j + 16);
            int8x16_t w2 = vld1q_s8(wrow + j + 32);
            int8x16_t w3 = vld1q_s8(wrow + j + 48);

            /* vsdotq: 4 lanes of 4 int8×int8 = 16 products accumulated into 4 int32s */
            acc0 = vdotq_s32(acc0, x0, w0);
            acc1 = vdotq_s32(acc1, x1, w1);
            acc2 = vdotq_s32(acc2, x2, w2);
            acc3 = vdotq_s32(acc3, x3, w3);
        }

        /* Combine accumulators */
        acc0 = vaddq_s32(acc0, acc1);
        acc2 = vaddq_s32(acc2, acc3);
        acc0 = vaddq_s32(acc0, acc2);
        int32_t sum = vaddvq_s32(acc0);

        /* Handle remaining elements */
        for (; j < in_dim; j++) {
            sum += (int32_t)x_q8[j] * (int32_t)wrow[j];
        }

        y[i] = (float)sum * scale;
    }
#elif defined(__ARM_NEON)
    /* NEON without dot product - use int16 intermediate */
    for (int i = 0; i < out_dim; i++) {
        const int8_t *wrow = W->data + i * in_dim;
        float scale = W->scale[i] * x_scale;

        int32x4_t acc0 = vdupq_n_s32(0);
        int32x4_t acc1 = vdupq_n_s32(0);

        int j = 0;
        for (; j + 15 < in_dim; j += 16) {
            int8x16_t xv = vld1q_s8(x_q8 + j);
            int8x16_t wv = vld1q_s8(wrow + j);

            /* Multiply low 8 elements */
            int16x8_t prod_lo = vmull_s8(vget_low_s8(xv), vget_low_s8(wv));
            int16x8_t prod_hi = vmull_s8(vget_high_s8(xv), vget_high_s8(wv));

            /* Accumulate to int32 */
            acc0 = vaddw_s16(acc0, vget_low_s16(prod_lo));
            acc0 = vaddw_s16(acc0, vget_high_s16(prod_lo));
            acc1 = vaddw_s16(acc1, vget_low_s16(prod_hi));
            acc1 = vaddw_s16(acc1, vget_high_s16(prod_hi));
        }

        acc0 = vaddq_s32(acc0, acc1);
        int32_t sum = vaddvq_s32(acc0);

        for (; j < in_dim; j++) {
            sum += (int32_t)x_q8[j] * (int32_t)wrow[j];
        }

        y[i] = (float)sum * scale;
    }
#elif defined(USE_AVX512)
    /* AVX-512: Process 64 int8 elements at a time using int16 intermediate.
     * We sign-extend int8 to int16, multiply, then use madd to sum pairs into int32. */
    for (int i = 0; i < out_dim; i++) {
        const int8_t *wrow = W->data + i * in_dim;
        float scale = W->scale[i] * x_scale;

        __m512i acc0 = _mm512_setzero_si512();
        __m512i acc1 = _mm512_setzero_si512();

        int j = 0;
        for (; j + 63 < in_dim; j += 64) {
            /* Load 64 int8 from x and W (as two 256-bit loads each) */
            __m256i x8_lo = _mm256_loadu_si256((const __m256i*)(x_q8 + j));
            __m256i x8_hi = _mm256_loadu_si256((const __m256i*)(x_q8 + j + 32));
            __m256i w8_lo = _mm256_loadu_si256((const __m256i*)(wrow + j));
            __m256i w8_hi = _mm256_loadu_si256((const __m256i*)(wrow + j + 32));

            /* Sign-extend int8 to int16: 32 int8 -> 32 int16 in 512-bit register */
            __m512i x16_lo = _mm512_cvtepi8_epi16(x8_lo);
            __m512i x16_hi = _mm512_cvtepi8_epi16(x8_hi);
            __m512i w16_lo = _mm512_cvtepi8_epi16(w8_lo);
            __m512i w16_hi = _mm512_cvtepi8_epi16(w8_hi);

            /* Multiply and add pairs: (x[2i]*w[2i] + x[2i+1]*w[2i+1]) -> 16 int32 */
            __m512i prod_lo = _mm512_madd_epi16(x16_lo, w16_lo);
            __m512i prod_hi = _mm512_madd_epi16(x16_hi, w16_hi);

            acc0 = _mm512_add_epi32(acc0, prod_lo);
            acc1 = _mm512_add_epi32(acc1, prod_hi);
        }

        /* Handle 32-element chunk if remaining */
        if (j + 31 < in_dim) {
            __m256i x8 = _mm256_loadu_si256((const __m256i*)(x_q8 + j));
            __m256i w8 = _mm256_loadu_si256((const __m256i*)(wrow + j));
            __m512i x16 = _mm512_cvtepi8_epi16(x8);
            __m512i w16 = _mm512_cvtepi8_epi16(w8);
            __m512i prod = _mm512_madd_epi16(x16, w16);
            acc0 = _mm512_add_epi32(acc0, prod);
            j += 32;
        }

        /* Combine accumulators and reduce */
        acc0 = _mm512_add_epi32(acc0, acc1);
        int32_t sum = _mm512_reduce_add_epi32(acc0);

        /* Handle remaining elements */
        for (; j < in_dim; j++) {
            sum += (int32_t)x_q8[j] * (int32_t)wrow[j];
        }

        y[i] = (float)sum * scale;
    }
#elif defined(USE_AVX2)
    /* AVX2: Process 32 int8 elements at a time using int16 intermediate */
    for (int i = 0; i < out_dim; i++) {
        const int8_t *wrow = W->data + i * in_dim;
        float scale = W->scale[i] * x_scale;

        __m256i acc0 = _mm256_setzero_si256();
        __m256i acc1 = _mm256_setzero_si256();

        int j = 0;
        for (; j + 31 < in_dim; j += 32) {
            /* Load 32 int8 from x and W (as two 128-bit loads each) */
            __m128i x8_lo = _mm_loadu_si128((const __m128i*)(x_q8 + j));
            __m128i x8_hi = _mm_loadu_si128((const __m128i*)(x_q8 + j + 16));
            __m128i w8_lo = _mm_loadu_si128((const __m128i*)(wrow + j));
            __m128i w8_hi = _mm_loadu_si128((const __m128i*)(wrow + j + 16));

            /* Sign-extend int8 to int16: 16 int8 -> 16 int16 in 256-bit register */
            __m256i x16_lo = _mm256_cvtepi8_epi16(x8_lo);
            __m256i x16_hi = _mm256_cvtepi8_epi16(x8_hi);
            __m256i w16_lo = _mm256_cvtepi8_epi16(w8_lo);
            __m256i w16_hi = _mm256_cvtepi8_epi16(w8_hi);

            /* Multiply and add pairs: (x[2i]*w[2i] + x[2i+1]*w[2i+1]) -> 8 int32 */
            __m256i prod_lo = _mm256_madd_epi16(x16_lo, w16_lo);
            __m256i prod_hi = _mm256_madd_epi16(x16_hi, w16_hi);

            acc0 = _mm256_add_epi32(acc0, prod_lo);
            acc1 = _mm256_add_epi32(acc1, prod_hi);
        }

        /* Combine accumulators */
        acc0 = _mm256_add_epi32(acc0, acc1);
        /* Horizontal sum: 8 int32 -> 1 int32 */
        __m128i lo = _mm256_castsi256_si128(acc0);
        __m128i hi = _mm256_extracti128_si256(acc0, 1);
        lo = _mm_add_epi32(lo, hi);
        lo = _mm_add_epi32(lo, _mm_shuffle_epi32(lo, _MM_SHUFFLE(2, 3, 0, 1)));
        lo = _mm_add_epi32(lo, _mm_shuffle_epi32(lo, _MM_SHUFFLE(1, 0, 3, 2)));
        int32_t sum = _mm_cvtsi128_si32(lo);

        /* Handle remaining elements */
        for (; j < in_dim; j++) {
            sum += (int32_t)x_q8[j] * (int32_t)wrow[j];
        }

        y[i] = (float)sum * scale;
    }
#else
    /* Scalar fallback */
    for (int i = 0; i < out_dim; i++) {
        const int8_t *wrow = W->data + i * in_dim;
        float scale = W->scale[i] * x_scale;

        int32_t sum = 0;
        for (int j = 0; j < in_dim; j++) {
            sum += (int32_t)x_q8[j] * (int32_t)wrow[j];
        }
        y[i] = (float)sum * scale;
    }
#endif
}

/* ========================================================================
 * Generator Structure
 * ======================================================================== */

struct qwen3_generator {
    /* Model components */
    qwen3_tokenizer_t *tokenizer;
    float *embed_tokens;     /* [vocab_size, hidden] */
    q8_weight_t *embed_tokens_q8;  /* Q8 version for LM head projection */
    float *norm_weight;      /* [hidden] - final layer norm */

    /* Transformer layers (reuse from flux_qwen3.c structures) */
    void *layers;            /* qwen3_layer_t array */
    int num_layers;

    /* RoPE is computed on-the-fly (no precomputed tables) */

    /* KV cache */
    kv_cache_t cache;

    /* Working memory (single token) */
    float *hidden_state;     /* [1, hidden] for single token */
    float *residual;
    float *q_buf;
    float *k_buf;
    float *v_buf;
    float *attn_out;
    float *mlp_gate;
    float *mlp_up;
    float *mlp_out;
    float *norm_buf;
    float *logits;           /* [vocab_size] */

    /* Token decode buffer */
    char decode_buf[256];

    /* Cached quantized input for Q8 linear ops (avoids redundant quantization) */
    int8_t *x_q8_cache;      /* [hidden] - pre-allocated buffer for quantized x */
    float x_q8_scale;        /* Scale factor for cached x_q8 */

    /* Safetensors files (keep open for mmap) */
    safetensors_file_t *files[2];
    int num_files;
};

/* ========================================================================
 * Forward Declarations (internal functions from flux_qwen3.c we need)
 * ======================================================================== */

/* We'll reimplement the core ops here for single-token inference */

static void rms_norm(float *out, const float *x, const float *weight,
                     int hidden, float eps) {
    float sum_sq = 0.0f;
    for (int i = 0; i < hidden; i++) {
        sum_sq += x[i] * x[i];
    }
    float rms = sqrtf(sum_sq / hidden + eps);
    float inv_rms = 1.0f / rms;
    for (int i = 0; i < hidden; i++) {
        out[i] = x[i] * inv_rms * weight[i];
    }
}

static void head_rms_norm(float *x, const float *weight,
                          int num_heads, int head_dim, float eps) {
    for (int h = 0; h < num_heads; h++) {
        float *head = x + h * head_dim;
        float sum_sq = 0.0f;
        for (int i = 0; i < head_dim; i++) {
            sum_sq += head[i] * head[i];
        }
        float rms = sqrtf(sum_sq / head_dim + eps);
        float inv_rms = 1.0f / rms;
        for (int i = 0; i < head_dim; i++) {
            head[i] = head[i] * inv_rms * weight[i];
        }
    }
}

/* Dot product */
static inline float vec_dot(const float *a, const float *b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

/* y += alpha * x */
static inline void vec_axpy(float *y, float alpha, const float *x, int n) {
    for (int i = 0; i < n; i++) {
        y[i] += alpha * x[i];
    }
}

static void apply_rope_single(float *q, float *k, int pos,
                              int num_q_heads, int num_kv_heads, int head_dim) {
    int half_dim = head_dim / 2;
    float theta = QWEN3_ROPE_THETA;

    /* Apply RoPE to Q (compute sin/cos on-the-fly) */
    for (int h = 0; h < num_q_heads; h++) {
        float *q_head = q + h * head_dim;
        for (int i = 0; i < half_dim; i++) {
            float freq = 1.0f / powf(theta, (float)(2 * i) / head_dim);
            float angle = pos * freq;
            float cos_val = cosf(angle);
            float sin_val = sinf(angle);
            float x0 = q_head[i];
            float x1 = q_head[i + half_dim];
            q_head[i] = x0 * cos_val - x1 * sin_val;
            q_head[i + half_dim] = x0 * sin_val + x1 * cos_val;
        }
    }

    /* Apply RoPE to K */
    for (int h = 0; h < num_kv_heads; h++) {
        float *k_head = k + h * head_dim;
        for (int i = 0; i < half_dim; i++) {
            float freq = 1.0f / powf(theta, (float)(2 * i) / head_dim);
            float angle = pos * freq;
            float cos_val = cosf(angle);
            float sin_val = sinf(angle);
            float x0 = k_head[i];
            float x1 = k_head[i + half_dim];
            k_head[i] = x0 * cos_val - x1 * sin_val;
            k_head[i + half_dim] = x0 * sin_val + x1 * cos_val;
        }
    }
}

static float silu(float x) {
    return x / (1.0f + expf(-x));
}

/* ========================================================================
 * Layer Weights Structure (matches flux_qwen3.c)
 * ======================================================================== */

typedef struct {
    float *q_proj_weight;
    float *k_proj_weight;
    float *v_proj_weight;
    float *o_proj_weight;
    float *q_norm_weight;
    float *k_norm_weight;
    /* Q8 quantized weights (owned, allocated during load) */
    q8_weight_t *q_proj_q8;
    q8_weight_t *k_proj_q8;
    q8_weight_t *v_proj_q8;
    q8_weight_t *o_proj_q8;
} gen_attention_t;

typedef struct {
    float *gate_proj_weight;
    float *up_proj_weight;
    float *down_proj_weight;
    /* Q8 quantized weights */
    q8_weight_t *gate_proj_q8;
    q8_weight_t *up_proj_q8;
    q8_weight_t *down_proj_q8;
} gen_mlp_t;

typedef struct {
    float *input_layernorm_weight;
    float *post_attention_layernorm_weight;
    gen_attention_t attn;
    gen_mlp_t mlp;
} gen_layer_t;

/* ========================================================================
 * KV Cache Management
 * ======================================================================== */

static int kv_cache_init(kv_cache_t *cache, int num_layers, int initial_capacity) {
    cache->num_layers = num_layers;
    cache->capacity = initial_capacity;
    cache->cur_seq_len = 0;

    int kv_dim = QWEN3_NUM_KV_HEADS * QWEN3_HEAD_DIM;

    cache->layers = calloc(num_layers, sizeof(kv_cache_layer_t));
    if (!cache->layers) return -1;

    for (int i = 0; i < num_layers; i++) {
        cache->layers[i].k = calloc(initial_capacity * kv_dim, sizeof(float));
        cache->layers[i].v = calloc(initial_capacity * kv_dim, sizeof(float));
        if (!cache->layers[i].k || !cache->layers[i].v) return -1;
    }

    return 0;
}

/* Grow KV cache capacity. Returns 0 on success, -1 on failure. */
static int kv_cache_grow(kv_cache_t *cache, int new_capacity) {
    if (new_capacity <= cache->capacity) return 0;
    if (new_capacity > GEN_MAX_SEQ_LEN) {
        fprintf(stderr, "Error: sequence length %d exceeds maximum %d\n",
                new_capacity, GEN_MAX_SEQ_LEN);
        return -1;
    }

    int kv_dim = QWEN3_NUM_KV_HEADS * QWEN3_HEAD_DIM;

    for (int i = 0; i < cache->num_layers; i++) {
        float *new_k = realloc(cache->layers[i].k, new_capacity * kv_dim * sizeof(float));
        float *new_v = realloc(cache->layers[i].v, new_capacity * kv_dim * sizeof(float));
        if (!new_k || !new_v) {
            fprintf(stderr, "Error: failed to grow KV cache to %d tokens\n", new_capacity);
            return -1;
        }
        cache->layers[i].k = new_k;
        cache->layers[i].v = new_v;
    }

    cache->capacity = new_capacity;
    return 0;
}

static void kv_cache_free(kv_cache_t *cache) {
    if (!cache->layers) return;
    for (int i = 0; i < cache->num_layers; i++) {
        free(cache->layers[i].k);
        free(cache->layers[i].v);
    }
    free(cache->layers);
    cache->layers = NULL;
}

static void kv_cache_reset(kv_cache_t *cache) {
    cache->cur_seq_len = 0;
    /* No need to zero - we only read up to cur_seq_len */
}

/* ========================================================================
 * Single Token Forward Pass
 * ======================================================================== */

static void attention_with_cache(qwen3_generator_t *gen, gen_layer_t *layer,
                                  int layer_idx, int pos) {
    int num_heads = QWEN3_NUM_HEADS;
    int num_kv_heads = QWEN3_NUM_KV_HEADS;
    int head_dim = QWEN3_HEAD_DIM;
    int hidden = QWEN3_HIDDEN_SIZE;
    int kv_dim = num_kv_heads * head_dim;
    int q_dim = num_heads * head_dim;
    int heads_per_kv = num_heads / num_kv_heads;
    float scale = 1.0f / sqrtf((float)head_dim);

    /* Q, K, V projections for single token (always Q8) */
    quantize_vector_q8(gen->norm_buf, gen->x_q8_cache, &gen->x_q8_scale, hidden);
    linear_q8_preq(gen->q_buf, gen->x_q8_cache, gen->x_q8_scale, layer->attn.q_proj_q8);
    linear_q8_preq(gen->k_buf, gen->x_q8_cache, gen->x_q8_scale, layer->attn.k_proj_q8);
    linear_q8_preq(gen->v_buf, gen->x_q8_cache, gen->x_q8_scale, layer->attn.v_proj_q8);

    /* Q/K RMS normalization */
    head_rms_norm(gen->q_buf, layer->attn.q_norm_weight, num_heads, head_dim, QWEN3_RMS_NORM_EPS);
    head_rms_norm(gen->k_buf, layer->attn.k_norm_weight, num_kv_heads, head_dim, QWEN3_RMS_NORM_EPS);

    /* Apply RoPE */
    apply_rope_single(gen->q_buf, gen->k_buf, pos, num_heads, num_kv_heads, head_dim);

    /* Store K, V in cache */
    kv_cache_layer_t *kv = &gen->cache.layers[layer_idx];
    memcpy(kv->k + pos * kv_dim, gen->k_buf, kv_dim * sizeof(float));
    memcpy(kv->v + pos * kv_dim, gen->v_buf, kv_dim * sizeof(float));

    /* Attention: Q @ K^T, softmax, @ V */
    /* For single token query, we compute attention against all cached K, V */
    int seq_len = pos + 1;
    float *scores = gen->mlp_gate;  /* Reuse buffer for scores [seq_len] */

    for (int h = 0; h < num_heads; h++) {
        int kv_h = h / heads_per_kv;  /* Which KV head this Q head uses */
        float *q_head = gen->q_buf + h * head_dim;
        float *out_head = gen->attn_out + h * head_dim;

        /* Compute attention scores: scores[s] = q_head · k[s] * scale */
        float max_score = -1e30f;
        for (int s = 0; s < seq_len; s++) {
            float *k_s = kv->k + s * kv_dim + kv_h * head_dim;
            /* Dot product */
            float score = vec_dot(q_head, k_s, head_dim) * scale;
            scores[s] = score;
            if (score > max_score) max_score = score;
        }

        /* Softmax */
        float sum_exp = 0.0f;
        for (int s = 0; s < seq_len; s++) {
            scores[s] = expf(scores[s] - max_score);
            sum_exp += scores[s];
        }
        float inv_sum = 1.0f / sum_exp;
        for (int s = 0; s < seq_len; s++) {
            scores[s] *= inv_sum;
        }

        /* Weighted sum of V: out = sum(scores[s] * v[s]) */
        /* Use BLAS: out = V^T @ scores where V is [seq_len, head_dim] */
        memset(out_head, 0, head_dim * sizeof(float));
        for (int s = 0; s < seq_len; s++) {
            float *v_s = kv->v + s * kv_dim + kv_h * head_dim;
            /* out += scores[s] * v_s */
            vec_axpy(out_head, scores[s], v_s, head_dim);
        }
    }

    /* Output projection (always Q8) */
    quantize_vector_q8(gen->attn_out, gen->x_q8_cache, &gen->x_q8_scale, q_dim);
    linear_q8_preq(gen->mlp_out, gen->x_q8_cache, gen->x_q8_scale, layer->attn.o_proj_q8);
}

static void mlp_forward(qwen3_generator_t *gen, gen_layer_t *layer) {
    int hidden = QWEN3_HIDDEN_SIZE;
    int intermediate = QWEN3_INTERMEDIATE_SIZE;

    /* Gate and Up projections (always Q8) */
    quantize_vector_q8(gen->norm_buf, gen->x_q8_cache, &gen->x_q8_scale, hidden);
    linear_q8_preq(gen->mlp_gate, gen->x_q8_cache, gen->x_q8_scale, layer->mlp.gate_proj_q8);
    linear_q8_preq(gen->mlp_up, gen->x_q8_cache, gen->x_q8_scale, layer->mlp.up_proj_q8);

    /* SwiGLU: silu(gate) * up */
    for (int i = 0; i < intermediate; i++) {
        gen->mlp_gate[i] = silu(gen->mlp_gate[i]) * gen->mlp_up[i];
    }

    /* Down projection (always Q8) */
    quantize_vector_q8(gen->mlp_gate, gen->x_q8_cache, &gen->x_q8_scale, intermediate);
    linear_q8_preq(gen->mlp_out, gen->x_q8_cache, gen->x_q8_scale, layer->mlp.down_proj_q8);
}

static void forward_token(qwen3_generator_t *gen, int token_id, int pos) {
    int hidden = QWEN3_HIDDEN_SIZE;
    gen_layer_t *layers = (gen_layer_t *)gen->layers;

    /* Token embedding */
    memcpy(gen->hidden_state, gen->embed_tokens + token_id * hidden, hidden * sizeof(float));

    /* Process each layer */
    for (int l = 0; l < gen->num_layers; l++) {
        gen_layer_t *layer = &layers[l];

        /* Save residual */
        memcpy(gen->residual, gen->hidden_state, hidden * sizeof(float));

        /* Pre-attention norm */
        rms_norm(gen->norm_buf, gen->hidden_state, layer->input_layernorm_weight,
                 hidden, QWEN3_RMS_NORM_EPS);

        /* Attention with KV cache */
        attention_with_cache(gen, layer, l, pos);

        /* Residual connection */
        for (int i = 0; i < hidden; i++) {
            gen->hidden_state[i] = gen->residual[i] + gen->mlp_out[i];
        }

        /* Save residual */
        memcpy(gen->residual, gen->hidden_state, hidden * sizeof(float));

        /* Post-attention norm */
        rms_norm(gen->norm_buf, gen->hidden_state, layer->post_attention_layernorm_weight,
                 hidden, QWEN3_RMS_NORM_EPS);

        /* MLP */
        mlp_forward(gen, layer);

        /* Residual connection */
        for (int i = 0; i < hidden; i++) {
            gen->hidden_state[i] = gen->residual[i] + gen->mlp_out[i];
        }
    }

    /* Final norm */
    rms_norm(gen->norm_buf, gen->hidden_state, gen->norm_weight, hidden, QWEN3_RMS_NORM_EPS);

    /* LM head: logits = norm_buf @ embed_tokens^T (weight tying, Q8) */
    quantize_vector_q8(gen->norm_buf, gen->x_q8_cache, &gen->x_q8_scale, hidden);
    linear_q8_preq(gen->logits, gen->x_q8_cache, gen->x_q8_scale, gen->embed_tokens_q8);
}

/* Forward pass for multiple tokens (prefill) */
static void forward_tokens_batch(qwen3_generator_t *gen, const int *token_ids, int seq_len) {
    /* We only have Q8 weights (no f32 for BLAS), so use token-by-token processing.
     * This is slower for prefill but uses much less memory. */
    for (int i = 0; i < seq_len; i++) {
        forward_token(gen, token_ids[i], i);
    }
    gen->cache.cur_seq_len = seq_len;
}

/* ========================================================================
 * Sampling
 * ======================================================================== */

int qwen3_sample(const float *logits, int vocab_size,
                 const qwen3_gen_params_t *params,
                 const int *past_tokens, int num_past) {
    float *probs = malloc(vocab_size * sizeof(float));
    int *indices = malloc(vocab_size * sizeof(int));

    /* Apply temperature */
    float temp = params->temperature;
    if (temp < 0.01f) temp = 0.01f;

    /* Apply repetition penalty */
    float *logits_mod = malloc(vocab_size * sizeof(float));
    memcpy(logits_mod, logits, vocab_size * sizeof(float));

    if (params->repeat_penalty != 1.0f && past_tokens && num_past > 0) {
        for (int i = 0; i < num_past; i++) {
            int tok = past_tokens[i];
            if (tok >= 0 && tok < vocab_size) {
                if (logits_mod[tok] > 0) {
                    logits_mod[tok] /= params->repeat_penalty;
                } else {
                    logits_mod[tok] *= params->repeat_penalty;
                }
            }
        }
    }

    /* Find max for numerical stability */
    float max_logit = logits_mod[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits_mod[i] > max_logit) max_logit = logits_mod[i];
    }

    /* Softmax with temperature */
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        probs[i] = expf((logits_mod[i] - max_logit) / temp);
        indices[i] = i;
        sum += probs[i];
    }
    for (int i = 0; i < vocab_size; i++) {
        probs[i] /= sum;
    }

    /* Top-k filtering */
    int k = params->top_k;
    if (k > 0 && k < vocab_size) {
        /* Sort by probability (descending) */
        for (int i = 0; i < k; i++) {
            for (int j = i + 1; j < vocab_size; j++) {
                if (probs[j] > probs[i]) {
                    float tmp_p = probs[i]; probs[i] = probs[j]; probs[j] = tmp_p;
                    int tmp_i = indices[i]; indices[i] = indices[j]; indices[j] = tmp_i;
                }
            }
        }
        /* Zero out everything after top-k */
        for (int i = k; i < vocab_size; i++) {
            probs[i] = 0.0f;
        }
        /* Renormalize */
        sum = 0.0f;
        for (int i = 0; i < k; i++) sum += probs[i];
        for (int i = 0; i < k; i++) probs[i] /= sum;
    }

    /* Top-p (nucleus) filtering */
    float p = params->top_p;
    if (p < 1.0f && p > 0.0f) {
        /* Sort by probability if not already sorted */
        if (k <= 0 || k >= vocab_size) {
            for (int i = 0; i < vocab_size - 1; i++) {
                for (int j = i + 1; j < vocab_size; j++) {
                    if (probs[j] > probs[i]) {
                        float tmp_p = probs[i]; probs[i] = probs[j]; probs[j] = tmp_p;
                        int tmp_i = indices[i]; indices[i] = indices[j]; indices[j] = tmp_i;
                    }
                }
            }
        }
        /* Find cutoff */
        float cumsum = 0.0f;
        int cutoff = vocab_size;
        for (int i = 0; i < vocab_size; i++) {
            cumsum += probs[i];
            if (cumsum >= p) {
                cutoff = i + 1;
                break;
            }
        }
        /* Zero out everything after cutoff */
        for (int i = cutoff; i < vocab_size; i++) {
            probs[i] = 0.0f;
        }
        /* Renormalize */
        sum = 0.0f;
        for (int i = 0; i < cutoff; i++) sum += probs[i];
        if (sum > 0.0f) {
            for (int i = 0; i < cutoff; i++) probs[i] /= sum;
        }
    }

    /* Sample from distribution */
    float r = (float)rand() / (float)RAND_MAX;
    float cumsum = 0.0f;
    int sampled = indices[0];
    for (int i = 0; i < vocab_size; i++) {
        cumsum += probs[i];
        if (r < cumsum) {
            sampled = indices[i];
            break;
        }
    }

    free(probs);
    free(indices);
    free(logits_mod);

    return sampled;
}

/* ========================================================================
 * Weight Loading
 * ======================================================================== */

/* Load tensor as f32 (allocates memory) */
static float *load_tensor_f32(safetensors_file_t **files, int num_files, const char *name) {
    for (int i = 0; i < num_files; i++) {
        const safetensor_t *t = safetensors_find(files[i], name);
        if (t) {
            return safetensors_get_f32(files[i], t);
        }
    }
    return NULL;
}

/* Get direct pointer to bf16 tensor data (mmap'd, no allocation) */
static const uint16_t *load_tensor_bf16_direct(safetensors_file_t **files, int num_files, const char *name) {
    for (int i = 0; i < num_files; i++) {
        const safetensor_t *t = safetensors_find(files[i], name);
        if (t) {
            return safetensors_get_bf16_direct(files[i], t);
        }
    }
    return NULL;
}

/* Load layer weights: small norms as f32, large projections directly to Q8 via mmap */
static int load_layer_weights(gen_layer_t *layer, safetensors_file_t **files,
                              int num_files, int layer_idx) {
    char name[256];
    int hidden = QWEN3_HIDDEN_SIZE;
    int q_dim = QWEN3_NUM_HEADS * QWEN3_HEAD_DIM;
    int kv_dim = QWEN3_NUM_KV_HEADS * QWEN3_HEAD_DIM;
    int intermediate = QWEN3_INTERMEDIATE_SIZE;

    /* Layer norms - small, load as f32 */
    snprintf(name, sizeof(name), "model.layers.%d.input_layernorm.weight", layer_idx);
    layer->input_layernorm_weight = load_tensor_f32(files, num_files, name);

    snprintf(name, sizeof(name), "model.layers.%d.post_attention_layernorm.weight", layer_idx);
    layer->post_attention_layernorm_weight = load_tensor_f32(files, num_files, name);

    snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_norm.weight", layer_idx);
    layer->attn.q_norm_weight = load_tensor_f32(files, num_files, name);

    snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_norm.weight", layer_idx);
    layer->attn.k_norm_weight = load_tensor_f32(files, num_files, name);

    /* Large projection weights - mmap bf16 and quantize directly to Q8 */
    const uint16_t *bf16_ptr;

    snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.weight", layer_idx);
    bf16_ptr = load_tensor_bf16_direct(files, num_files, name);
    layer->attn.q_proj_q8 = bf16_ptr ? q8_quantize_bf16(bf16_ptr, q_dim, hidden) : NULL;

    snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_proj.weight", layer_idx);
    bf16_ptr = load_tensor_bf16_direct(files, num_files, name);
    layer->attn.k_proj_q8 = bf16_ptr ? q8_quantize_bf16(bf16_ptr, kv_dim, hidden) : NULL;

    snprintf(name, sizeof(name), "model.layers.%d.self_attn.v_proj.weight", layer_idx);
    bf16_ptr = load_tensor_bf16_direct(files, num_files, name);
    layer->attn.v_proj_q8 = bf16_ptr ? q8_quantize_bf16(bf16_ptr, kv_dim, hidden) : NULL;

    snprintf(name, sizeof(name), "model.layers.%d.self_attn.o_proj.weight", layer_idx);
    bf16_ptr = load_tensor_bf16_direct(files, num_files, name);
    layer->attn.o_proj_q8 = bf16_ptr ? q8_quantize_bf16(bf16_ptr, hidden, q_dim) : NULL;

    snprintf(name, sizeof(name), "model.layers.%d.mlp.gate_proj.weight", layer_idx);
    bf16_ptr = load_tensor_bf16_direct(files, num_files, name);
    layer->mlp.gate_proj_q8 = bf16_ptr ? q8_quantize_bf16(bf16_ptr, intermediate, hidden) : NULL;

    snprintf(name, sizeof(name), "model.layers.%d.mlp.up_proj.weight", layer_idx);
    bf16_ptr = load_tensor_bf16_direct(files, num_files, name);
    layer->mlp.up_proj_q8 = bf16_ptr ? q8_quantize_bf16(bf16_ptr, intermediate, hidden) : NULL;

    snprintf(name, sizeof(name), "model.layers.%d.mlp.down_proj.weight", layer_idx);
    bf16_ptr = load_tensor_bf16_direct(files, num_files, name);
    layer->mlp.down_proj_q8 = bf16_ptr ? q8_quantize_bf16(bf16_ptr, hidden, intermediate) : NULL;

    /* f32 projection weights not used - set to NULL */
    layer->attn.q_proj_weight = NULL;
    layer->attn.k_proj_weight = NULL;
    layer->attn.v_proj_weight = NULL;
    layer->attn.o_proj_weight = NULL;
    layer->mlp.gate_proj_weight = NULL;
    layer->mlp.up_proj_weight = NULL;
    layer->mlp.down_proj_weight = NULL;

    /* Check all weights loaded */
    return (layer->input_layernorm_weight && layer->post_attention_layernorm_weight &&
            layer->attn.q_norm_weight && layer->attn.k_norm_weight &&
            layer->attn.q_proj_q8 && layer->attn.k_proj_q8 &&
            layer->attn.v_proj_q8 && layer->attn.o_proj_q8 &&
            layer->mlp.gate_proj_q8 && layer->mlp.up_proj_q8 &&
            layer->mlp.down_proj_q8) ? 0 : -1;
}

/* ========================================================================
 * Public API Implementation
 * ======================================================================== */

qwen3_generator_t *qwen3_generator_create(const char *model_dir) {
    qwen3_generator_t *gen = calloc(1, sizeof(qwen3_generator_t));
    if (!gen) return NULL;

    /* Load tokenizer */
    char path[512];
    snprintf(path, sizeof(path), "%s/tokenizer/tokenizer.json", model_dir);
    gen->tokenizer = qwen3_tokenizer_load(path);
    if (!gen->tokenizer) {
        fprintf(stderr, "Failed to load tokenizer from %s\n", path);
        qwen3_generator_free(gen);
        return NULL;
    }

    /* Load safetensors files */
    snprintf(path, sizeof(path), "%s/text_encoder/model-00001-of-00002.safetensors", model_dir);
    gen->files[0] = safetensors_open(path);
    snprintf(path, sizeof(path), "%s/text_encoder/model-00002-of-00002.safetensors", model_dir);
    gen->files[1] = safetensors_open(path);
    gen->num_files = 2;

    if (!gen->files[0] || !gen->files[1]) {
        fprintf(stderr, "Failed to load model files\n");
        qwen3_generator_free(gen);
        return NULL;
    }

    /* Load embeddings and final norm */
    gen->embed_tokens = load_tensor_f32(gen->files, 2, "model.embed_tokens.weight");
    gen->norm_weight = load_tensor_f32(gen->files, 2, "model.norm.weight");

    if (!gen->embed_tokens || !gen->norm_weight) {
        fprintf(stderr, "Failed to load embedding weights\n");
        qwen3_generator_free(gen);
        return NULL;
    }

    /* Quantize embedding matrix for LM head (weight tying) */
    gen->embed_tokens_q8 = q8_quantize(gen->embed_tokens, QWEN3_VOCAB_SIZE, QWEN3_HIDDEN_SIZE);
    if (!gen->embed_tokens_q8) {
        fprintf(stderr, "Failed to quantize embedding matrix\n");
        qwen3_generator_free(gen);
        return NULL;
    }

    /* Load layers */
    gen->num_layers = QWEN3_NUM_LAYERS;
    gen->layers = calloc(gen->num_layers, sizeof(gen_layer_t));
    if (!gen->layers) {
        qwen3_generator_free(gen);
        return NULL;
    }

    fprintf(stderr, "Loading %d layers", gen->num_layers);
    for (int i = 0; i < gen->num_layers; i++) {
        if (load_layer_weights(&((gen_layer_t *)gen->layers)[i], gen->files, 2, i) != 0) {
            fprintf(stderr, "\nFailed to load layer %d\n", i);
            qwen3_generator_free(gen);
            return NULL;
        }
        if ((i + 1) % 6 == 0) {
            fprintf(stderr, ".");
            fflush(stderr);
        }
    }
    fprintf(stderr, " done\n");

    /* Initialize KV cache (starts small, grows dynamically) */
    if (kv_cache_init(&gen->cache, gen->num_layers, GEN_INITIAL_SEQ_LEN) != 0) {
        fprintf(stderr, "Failed to initialize KV cache\n");
        qwen3_generator_free(gen);
        return NULL;
    }

    /* Allocate working memory */
    int hidden = QWEN3_HIDDEN_SIZE;
    int intermediate = QWEN3_INTERMEDIATE_SIZE;
    int q_dim = QWEN3_NUM_HEADS * QWEN3_HEAD_DIM;
    int kv_dim = QWEN3_NUM_KV_HEADS * QWEN3_HEAD_DIM;

    gen->hidden_state = malloc(hidden * sizeof(float));
    gen->residual = malloc(hidden * sizeof(float));
    gen->q_buf = malloc(q_dim * sizeof(float));
    gen->k_buf = malloc(kv_dim * sizeof(float));
    gen->v_buf = malloc(kv_dim * sizeof(float));
    gen->attn_out = malloc(q_dim * sizeof(float));
    gen->mlp_gate = malloc(intermediate * sizeof(float));
    gen->mlp_up = malloc(intermediate * sizeof(float));
    gen->mlp_out = malloc(hidden * sizeof(float));
    gen->norm_buf = malloc(hidden * sizeof(float));
    gen->logits = malloc(QWEN3_VOCAB_SIZE * sizeof(float));
    /* For Q8 linear caching - sized for largest dim (intermediate=9728) */
    gen->x_q8_cache = malloc(intermediate * sizeof(int8_t));

    if (!gen->hidden_state || !gen->residual || !gen->q_buf || !gen->k_buf ||
        !gen->v_buf || !gen->attn_out || !gen->mlp_gate || !gen->mlp_up ||
        !gen->mlp_out || !gen->norm_buf || !gen->logits || !gen->x_q8_cache) {
        fprintf(stderr, "Failed to allocate working memory\n");
        qwen3_generator_free(gen);
        return NULL;
    }

    return gen;
}

void qwen3_generator_free(qwen3_generator_t *gen) {
    if (!gen) return;

    qwen3_tokenizer_free(gen->tokenizer);

    /* Free layer weights */
    if (gen->layers) {
        gen_layer_t *layers = (gen_layer_t *)gen->layers;
        for (int i = 0; i < gen->num_layers; i++) {
            free(layers[i].input_layernorm_weight);
            free(layers[i].post_attention_layernorm_weight);
            free(layers[i].attn.q_proj_weight);
            free(layers[i].attn.k_proj_weight);
            free(layers[i].attn.v_proj_weight);
            free(layers[i].attn.o_proj_weight);
            free(layers[i].attn.q_norm_weight);
            free(layers[i].attn.k_norm_weight);
            free(layers[i].mlp.gate_proj_weight);
            free(layers[i].mlp.up_proj_weight);
            free(layers[i].mlp.down_proj_weight);
            /* Free Q8 weights */
            q8_free(layers[i].attn.q_proj_q8);
            q8_free(layers[i].attn.k_proj_q8);
            q8_free(layers[i].attn.v_proj_q8);
            q8_free(layers[i].attn.o_proj_q8);
            q8_free(layers[i].mlp.gate_proj_q8);
            q8_free(layers[i].mlp.up_proj_q8);
            q8_free(layers[i].mlp.down_proj_q8);
        }
        free(gen->layers);
    }

    free(gen->embed_tokens);
    q8_free(gen->embed_tokens_q8);
    free(gen->norm_weight);

    kv_cache_free(&gen->cache);

    free(gen->hidden_state);
    free(gen->residual);
    free(gen->q_buf);
    free(gen->k_buf);
    free(gen->v_buf);
    free(gen->attn_out);
    free(gen->mlp_gate);
    free(gen->mlp_up);
    free(gen->mlp_out);
    free(gen->norm_buf);
    free(gen->logits);
    free(gen->x_q8_cache);

    for (int i = 0; i < gen->num_files; i++) {
        safetensors_close(gen->files[i]);
    }

    free(gen);
}

void qwen3_generator_reset(qwen3_generator_t *gen) {
    if (gen) {
        kv_cache_reset(&gen->cache);
    }
}

float *qwen3_forward_token(qwen3_generator_t *gen, int token_id) {
    if (!gen || token_id < 0 || token_id >= QWEN3_VOCAB_SIZE) return NULL;

    int pos = gen->cache.cur_seq_len;

    /* Grow KV cache if needed (double capacity) */
    if (pos >= gen->cache.capacity) {
        int new_capacity = gen->cache.capacity * 2;
        if (new_capacity > GEN_MAX_SEQ_LEN) new_capacity = GEN_MAX_SEQ_LEN;
        if (pos >= new_capacity) {
            fprintf(stderr, "Error: sequence length %d exceeds maximum %d\n",
                    pos, GEN_MAX_SEQ_LEN);
            return NULL;
        }
        if (kv_cache_grow(&gen->cache, new_capacity) != 0) return NULL;
    }

    forward_token(gen, token_id, pos);
    gen->cache.cur_seq_len++;

    return gen->logits;
}

const char *qwen3_decode_token(qwen3_generator_t *gen, int token_id) {
    if (!gen || !gen->tokenizer) return "";
    return qwen3_detokenize_single(gen->tokenizer, token_id);
}

int qwen3_eos_token_id(void) { return QWEN3_IM_END_ID; }
int qwen3_bos_token_id(void) { return QWEN3_IM_START_ID; }

/* Shared generation loop used by both qwen3_generate and qwen3_generate_continue */
static char *generate_loop(qwen3_generator_t *gen,
                           const qwen3_gen_params_t *params,
                           const int *prompt_tokens, int num_prompt_tokens,
                           qwen3_token_callback callback, void *user_data) {
    int capacity = 256;
    int *output_tokens = malloc(capacity * sizeof(int));
    int num_output = 0;

    /* Track tokens for repetition penalty (grows dynamically) */
    int all_capacity = num_prompt_tokens + 256;
    int *all_tokens = malloc(all_capacity * sizeof(int));
    if (prompt_tokens && num_prompt_tokens > 0) {
        memcpy(all_tokens, prompt_tokens, num_prompt_tokens * sizeof(int));
    }
    int all_count = num_prompt_tokens;

    /* max_tokens=0 means unlimited (until EOS or context limit) */
    int max_iter = params->max_tokens > 0 ? params->max_tokens : GEN_MAX_SEQ_LEN;

    for (int i = 0; i < max_iter; i++) {
        int next_token = qwen3_sample(gen->logits, QWEN3_VOCAB_SIZE, params,
                                       all_tokens, all_count);

        if (next_token == QWEN3_IM_END_ID || next_token == QWEN3_EOS_ID) {
            break;
        }

        /* Store token (grow buffer if needed) */
        if (num_output >= capacity) {
            capacity *= 2;
            int *tmp = realloc(output_tokens, capacity * sizeof(int));
            if (!tmp) {
                free(output_tokens);
                free(all_tokens);
                return NULL;
            }
            output_tokens = tmp;
        }
        output_tokens[num_output++] = next_token;

        /* Grow all_tokens buffer if needed */
        if (all_count >= all_capacity) {
            all_capacity *= 2;
            int *tmp = realloc(all_tokens, all_capacity * sizeof(int));
            if (!tmp) {
                free(output_tokens);
                free(all_tokens);
                return NULL;
            }
            all_tokens = tmp;
        }
        all_tokens[all_count++] = next_token;

        /* Callback with decoded token */
        if (callback) {
            const char *tok_str = qwen3_decode_token(gen, next_token);
            if (callback(tok_str, user_data) != 0) {
                break;
            }
        }

        /* Forward next token (may fail if context limit reached) */
        if (!qwen3_forward_token(gen, next_token)) {
            break;  /* Context limit reached */
        }
    }

    free(all_tokens);

    char *result = qwen3_detokenize(gen->tokenizer, output_tokens, num_output);
    free(output_tokens);

    return result;
}

char *qwen3_generate(qwen3_generator_t *gen,
                     const char *prompt,
                     const qwen3_gen_params_t *params,
                     qwen3_token_callback callback,
                     void *user_data) {
    if (!gen || !prompt) return NULL;

    qwen3_gen_params_t p;
    if (params) {
        p = *params;
    } else {
        p = (qwen3_gen_params_t)QWEN3_GEN_PARAMS_DEFAULT;
    }

    /* Check if this is a continuation or new conversation */
    int is_continuation = (gen->cache.cur_seq_len > 0);

    /* Tokenize prompt */
    int num_tokens;
    int *tokens;
    int max_new_tokens = GEN_MAX_SEQ_LEN - gen->cache.cur_seq_len;

    if (p.use_chat) {
        if (is_continuation) {
            /* Multi-turn: close previous assistant turn, add new user turn */
            tokens = qwen3_tokenize_chat_continue(gen->tokenizer, prompt, &num_tokens, max_new_tokens);
        } else {
            /* First turn */
            tokens = qwen3_tokenize_chat(gen->tokenizer, prompt, &num_tokens, max_new_tokens);
        }
    } else {
        /* Raw mode: always reset for simplicity */
        qwen3_generator_reset(gen);
        tokens = qwen3_tokenize(gen->tokenizer, prompt, &num_tokens, GEN_MAX_SEQ_LEN);
    }

    if (!tokens || num_tokens == 0) {
        fprintf(stderr, "Failed to tokenize prompt\n");
        return NULL;
    }

    /* Process prompt tokens (prefill) */
    if (!is_continuation && gen->cache.cur_seq_len == 0) {
        /* Use batched prefill for initial prompt - much faster */
        forward_tokens_batch(gen, tokens, num_tokens);
    } else {
        /* Token-by-token for continuations (appending to existing cache) */
        for (int i = 0; i < num_tokens; i++) {
            qwen3_forward_token(gen, tokens[i]);
        }
    }

    /* Generate using shared loop */
    char *result = generate_loop(gen, &p, tokens, num_tokens, callback, user_data);
    free(tokens);
    return result;
}

char *qwen3_generate_continue(qwen3_generator_t *gen,
                              const char *prompt,
                              const qwen3_gen_params_t *params,
                              qwen3_token_callback callback,
                              void *user_data) {
    if (!gen || !prompt) return NULL;

    qwen3_gen_params_t p;
    if (params) {
        p = *params;
    } else {
        p = (qwen3_gen_params_t)QWEN3_GEN_PARAMS_DEFAULT;
    }

    /* Tokenize continuation (raw, no template) */
    int num_tokens;
    int *tokens = qwen3_tokenize(gen->tokenizer, prompt, &num_tokens, GEN_MAX_SEQ_LEN);

    if (!tokens || num_tokens == 0) {
        return NULL;
    }

    /* Process new tokens (appending to existing cache) */
    for (int i = 0; i < num_tokens; i++) {
        qwen3_forward_token(gen, tokens[i]);
    }

    /* Generate using shared loop */
    char *result = generate_loop(gen, &p, tokens, num_tokens, callback, user_data);
    free(tokens);
    return result;
}
