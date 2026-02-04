/*
 * ROCm Custom Kernel Tests
 * 
 * Tests for silu, rms_norm, softmax, rope, etc.
 */

#include "flux_rocm.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>

static float* gpu_ptr(flux_rocm_tensor_t t) {
    return (float*)flux_rocm_tensor_gpu_ptr(t);
}

/* ========================================================================
 * CPU Reference Implementations
 * ======================================================================== */

static void cpu_silu(float *x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = x[i] / (1.0f + expf(-x[i]));
    }
}

static void cpu_silu_mul(float *gate, const float *up, int n) {
    for (int i = 0; i < n; i++) {
        float g = gate[i];
        gate[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}

static void cpu_rms_norm(float *out, const float *x, const float *weight,
                         int seq_len, int hidden, float eps) {
    for (int s = 0; s < seq_len; s++) {
        const float *row = x + s * hidden;
        float *out_row = out + s * hidden;
        
        float sum_sq = 0;
        for (int i = 0; i < hidden; i++) {
            sum_sq += row[i] * row[i];
        }
        float rms_inv = 1.0f / sqrtf(sum_sq / hidden + eps);
        
        for (int i = 0; i < hidden; i++) {
            out_row[i] = row[i] * rms_inv * weight[i];
        }
    }
}

static void cpu_softmax(float *x, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        float *row = x + r * cols;
        
        float max_val = row[0];
        for (int i = 1; i < cols; i++) {
            if (row[i] > max_val) max_val = row[i];
        }
        
        float sum = 0;
        for (int i = 0; i < cols; i++) {
            row[i] = expf(row[i] - max_val);
            sum += row[i];
        }
        
        for (int i = 0; i < cols; i++) {
            row[i] /= sum;
        }
    }
}

static void cpu_adaln_norm(float *out, const float *x,
                           const float *shift, const float *scale,
                           int seq_len, int hidden, float eps) {
    for (int s = 0; s < seq_len; s++) {
        const float *row = x + s * hidden;
        float *out_row = out + s * hidden;
        
        /* Mean */
        float mean = 0;
        for (int i = 0; i < hidden; i++) mean += row[i];
        mean /= hidden;
        
        /* Variance */
        float var = 0;
        for (int i = 0; i < hidden; i++) {
            float diff = row[i] - mean;
            var += diff * diff;
        }
        var /= hidden;
        float var_inv = 1.0f / sqrtf(var + eps);
        
        /* Apply */
        for (int i = 0; i < hidden; i++) {
            float norm = (row[i] - mean) * var_inv;
            out_row[i] = (1.0f + scale[i]) * norm + shift[i];
        }
    }
}

/* ========================================================================
 * Test Functions
 * ======================================================================== */

static float max_diff(const float *a, const float *b, int n) {
    float max_d = 0;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > max_d) max_d = d;
    }
    return max_d;
}

static int test_silu() {
    printf("Testing SiLU... ");
    
    const int n = 1024 * 1024;
    float *x_cpu = (float*)malloc(n * sizeof(float));
    float *x_gpu_out = (float*)malloc(n * sizeof(float));
    
    for (int i = 0; i < n; i++) {
        x_cpu[i] = (float)(i % 100) / 50.0f - 1.0f;  /* Range [-1, 1] */
    }
    
    /* GPU */
    flux_rocm_tensor_t t_x = flux_rocm_tensor_create(x_cpu, n);
    flux_rocm_silu(gpu_ptr(t_x), n);
    flux_rocm_sync();
    flux_rocm_tensor_read(t_x, x_gpu_out, n);
    
    /* CPU reference */
    cpu_silu(x_cpu, n);
    
    float diff = max_diff(x_cpu, x_gpu_out, n);
    
    flux_rocm_tensor_free(t_x);
    free(x_cpu);
    free(x_gpu_out);
    
    if (diff < 1e-5) {
        printf("PASS (max diff: %.2e)\n", diff);
        return 1;
    } else {
        printf("FAIL (max diff: %.2e)\n", diff);
        return 0;
    }
}

static int test_silu_mul() {
    printf("Testing SiLU multiply... ");
    
    const int n = 1024 * 1024;
    float *gate_cpu = (float*)malloc(n * sizeof(float));
    float *up = (float*)malloc(n * sizeof(float));
    float *gate_gpu_out = (float*)malloc(n * sizeof(float));
    
    for (int i = 0; i < n; i++) {
        gate_cpu[i] = (float)(i % 100) / 50.0f - 1.0f;
        up[i] = (float)((i * 7) % 100) / 50.0f - 1.0f;
    }
    
    /* GPU */
    flux_rocm_tensor_t t_gate = flux_rocm_tensor_create(gate_cpu, n);
    flux_rocm_tensor_t t_up = flux_rocm_tensor_create(up, n);
    flux_rocm_silu_mul(gpu_ptr(t_gate), gpu_ptr(t_up), n);
    flux_rocm_sync();
    flux_rocm_tensor_read(t_gate, gate_gpu_out, n);
    
    /* CPU reference */
    cpu_silu_mul(gate_cpu, up, n);
    
    float diff = max_diff(gate_cpu, gate_gpu_out, n);
    
    flux_rocm_tensor_free(t_gate);
    flux_rocm_tensor_free(t_up);
    free(gate_cpu);
    free(up);
    free(gate_gpu_out);
    
    if (diff < 1e-5) {
        printf("PASS (max diff: %.2e)\n", diff);
        return 1;
    } else {
        printf("FAIL (max diff: %.2e)\n", diff);
        return 0;
    }
}

static int test_rms_norm() {
    printf("Testing RMSNorm... ");
    
    const int seq = 512, hidden = 3072;
    const float eps = 1e-6f;
    
    float *x = (float*)malloc(seq * hidden * sizeof(float));
    float *weight = (float*)malloc(hidden * sizeof(float));
    float *out_cpu = (float*)malloc(seq * hidden * sizeof(float));
    float *out_gpu = (float*)malloc(seq * hidden * sizeof(float));
    
    for (int i = 0; i < seq * hidden; i++) {
        x[i] = (float)(i % 1000) / 500.0f - 1.0f;
    }
    for (int i = 0; i < hidden; i++) {
        weight[i] = 1.0f + (float)(i % 10) / 100.0f;
    }
    
    /* GPU */
    flux_rocm_tensor_t t_x = flux_rocm_tensor_create(x, seq * hidden);
    flux_rocm_tensor_t t_out = flux_rocm_tensor_alloc(seq * hidden);
    flux_rocm_tensor_t t_w = flux_rocm_tensor_create(weight, hidden);
    
    flux_rocm_rms_norm(gpu_ptr(t_out), gpu_ptr(t_x), gpu_ptr(t_w), seq, hidden, eps);
    flux_rocm_sync();
    flux_rocm_tensor_read(t_out, out_gpu, seq * hidden);
    
    /* CPU reference */
    cpu_rms_norm(out_cpu, x, weight, seq, hidden, eps);
    
    float diff = max_diff(out_cpu, out_gpu, seq * hidden);
    
    flux_rocm_tensor_free(t_x);
    flux_rocm_tensor_free(t_out);
    flux_rocm_tensor_free(t_w);
    free(x); free(weight); free(out_cpu); free(out_gpu);
    
    if (diff < 1e-4) {
        printf("PASS (max diff: %.2e)\n", diff);
        return 1;
    } else {
        printf("FAIL (max diff: %.2e)\n", diff);
        return 0;
    }
}

static int test_softmax() {
    printf("Testing Softmax... ");
    
    const int rows = 768, cols = 768;  /* Attention matrix size */
    
    float *x_cpu = (float*)malloc(rows * cols * sizeof(float));
    float *x_gpu_out = (float*)malloc(rows * cols * sizeof(float));
    
    for (int i = 0; i < rows * cols; i++) {
        x_cpu[i] = (float)(i % 1000) / 100.0f - 5.0f;  /* Range [-5, 5] */
    }
    
    /* GPU */
    flux_rocm_tensor_t t_x = flux_rocm_tensor_create(x_cpu, rows * cols);
    flux_rocm_softmax(gpu_ptr(t_x), rows, cols);
    flux_rocm_sync();
    flux_rocm_tensor_read(t_x, x_gpu_out, rows * cols);
    
    /* CPU reference */
    cpu_softmax(x_cpu, rows, cols);
    
    float diff = max_diff(x_cpu, x_gpu_out, rows * cols);
    
    flux_rocm_tensor_free(t_x);
    free(x_cpu);
    free(x_gpu_out);
    
    if (diff < 1e-5) {
        printf("PASS (max diff: %.2e)\n", diff);
        return 1;
    } else {
        printf("FAIL (max diff: %.2e)\n", diff);
        return 0;
    }
}

static int test_adaln_norm() {
    printf("Testing AdaLN... ");
    
    const int seq = 256, hidden = 3072;
    const float eps = 1e-6f;
    
    float *x = (float*)malloc(seq * hidden * sizeof(float));
    float *shift = (float*)malloc(hidden * sizeof(float));
    float *scale = (float*)malloc(hidden * sizeof(float));
    float *out_cpu = (float*)malloc(seq * hidden * sizeof(float));
    float *out_gpu = (float*)malloc(seq * hidden * sizeof(float));
    
    for (int i = 0; i < seq * hidden; i++) {
        x[i] = (float)(i % 1000) / 500.0f - 1.0f;
    }
    for (int i = 0; i < hidden; i++) {
        shift[i] = (float)(i % 100) / 1000.0f;
        scale[i] = (float)(i % 50) / 500.0f;
    }
    
    /* GPU */
    flux_rocm_tensor_t t_x = flux_rocm_tensor_create(x, seq * hidden);
    flux_rocm_tensor_t t_out = flux_rocm_tensor_alloc(seq * hidden);
    flux_rocm_tensor_t t_shift = flux_rocm_tensor_create(shift, hidden);
    flux_rocm_tensor_t t_scale = flux_rocm_tensor_create(scale, hidden);
    
    flux_rocm_adaln_norm(gpu_ptr(t_out), gpu_ptr(t_x), 
                         gpu_ptr(t_shift), gpu_ptr(t_scale),
                         seq, hidden, eps);
    flux_rocm_sync();
    flux_rocm_tensor_read(t_out, out_gpu, seq * hidden);
    
    /* CPU reference */
    cpu_adaln_norm(out_cpu, x, shift, scale, seq, hidden, eps);
    
    float diff = max_diff(out_cpu, out_gpu, seq * hidden);
    
    flux_rocm_tensor_free(t_x);
    flux_rocm_tensor_free(t_out);
    flux_rocm_tensor_free(t_shift);
    flux_rocm_tensor_free(t_scale);
    free(x); free(shift); free(scale); free(out_cpu); free(out_gpu);
    
    if (diff < 1e-4) {
        printf("PASS (max diff: %.2e)\n", diff);
        return 1;
    } else {
        printf("FAIL (max diff: %.2e)\n", diff);
        return 0;
    }
}

/* ========================================================================
 * RoPE Tests
 * ======================================================================== */

static void cpu_rope_2d(float *x, const float *cos_freq, const float *sin_freq,
                        int seq, int heads, int head_dim, int axis_dim) {
    int pairs_per_head = axis_dim / 2;
    
    for (int s = 0; s < seq; s++) {
        for (int h = 0; h < heads; h++) {
            for (int p = 0; p < pairs_per_head; p++) {
                int x_idx = s * (heads * head_dim) + h * head_dim + p * 2;
                int freq_idx = s * pairs_per_head + p;
                
                float x0 = x[x_idx];
                float x1 = x[x_idx + 1];
                float cos_val = cos_freq[freq_idx];
                float sin_val = sin_freq[freq_idx];
                
                x[x_idx]     = x0 * cos_val - x1 * sin_val;
                x[x_idx + 1] = x0 * sin_val + x1 * cos_val;
            }
        }
    }
}

/* ========================================================================
 * Attention Test
 * ======================================================================== */

static void cpu_attention(float *out, const float *Q, const float *K, const float *V,
                          int seq_q, int seq_k, int num_heads, int head_dim, float scale) {
    int hidden = num_heads * head_dim;
    
    /* For each head */
    for (int h = 0; h < num_heads; h++) {
        /* Compute scores = Q @ K^T * scale */
        float *scores = (float*)malloc(seq_q * seq_k * sizeof(float));
        
        for (int sq = 0; sq < seq_q; sq++) {
            for (int sk = 0; sk < seq_k; sk++) {
                float sum = 0;
                for (int d = 0; d < head_dim; d++) {
                    float q_val = Q[sq * hidden + h * head_dim + d];
                    float k_val = K[sk * hidden + h * head_dim + d];
                    sum += q_val * k_val;
                }
                scores[sq * seq_k + sk] = sum * scale;
            }
        }
        
        /* Softmax per row */
        for (int sq = 0; sq < seq_q; sq++) {
            float max_val = scores[sq * seq_k];
            for (int sk = 1; sk < seq_k; sk++) {
                if (scores[sq * seq_k + sk] > max_val) 
                    max_val = scores[sq * seq_k + sk];
            }
            
            float sum = 0;
            for (int sk = 0; sk < seq_k; sk++) {
                scores[sq * seq_k + sk] = expf(scores[sq * seq_k + sk] - max_val);
                sum += scores[sq * seq_k + sk];
            }
            for (int sk = 0; sk < seq_k; sk++) {
                scores[sq * seq_k + sk] /= sum;
            }
        }
        
        /* out = scores @ V */
        for (int sq = 0; sq < seq_q; sq++) {
            for (int d = 0; d < head_dim; d++) {
                float sum = 0;
                for (int sk = 0; sk < seq_k; sk++) {
                    float score = scores[sq * seq_k + sk];
                    float v_val = V[sk * hidden + h * head_dim + d];
                    sum += score * v_val;
                }
                out[sq * hidden + h * head_dim + d] = sum;
            }
        }
        
        free(scores);
    }
}

static int test_attention() {
    printf("Testing Attention... ");
    
    const int seq = 64, num_heads = 8, head_dim = 64;
    const int hidden = num_heads * head_dim;
    const float scale = 1.0f / sqrtf((float)head_dim);
    
    float *Q = (float*)malloc(seq * hidden * sizeof(float));
    float *K = (float*)malloc(seq * hidden * sizeof(float));
    float *V = (float*)malloc(seq * hidden * sizeof(float));
    float *out_cpu = (float*)malloc(seq * hidden * sizeof(float));
    float *out_gpu = (float*)malloc(seq * hidden * sizeof(float));
    
    /* Initialize with small values to avoid numerical issues */
    for (int i = 0; i < seq * hidden; i++) {
        Q[i] = (float)((i * 7) % 100) / 500.0f - 0.1f;
        K[i] = (float)((i * 11) % 100) / 500.0f - 0.1f;
        V[i] = (float)((i * 13) % 100) / 500.0f - 0.1f;
    }
    
    /* CPU reference */
    cpu_attention(out_cpu, Q, K, V, seq, seq, num_heads, head_dim, scale);
    
    /* GPU */
    flux_rocm_tensor_t t_Q = flux_rocm_tensor_create(Q, seq * hidden);
    flux_rocm_tensor_t t_K = flux_rocm_tensor_create(K, seq * hidden);
    flux_rocm_tensor_t t_V = flux_rocm_tensor_create(V, seq * hidden);
    flux_rocm_tensor_t t_out = flux_rocm_tensor_alloc(seq * hidden);
    
    int ok = flux_rocm_attention_fused(gpu_ptr(t_out), gpu_ptr(t_Q), gpu_ptr(t_K), gpu_ptr(t_V),
                                       seq, seq, num_heads, head_dim, scale);
    
    if (!ok) {
        printf("FAIL (attention returned error)\n");
        flux_rocm_tensor_free(t_Q);
        flux_rocm_tensor_free(t_K);
        flux_rocm_tensor_free(t_V);
        flux_rocm_tensor_free(t_out);
        free(Q); free(K); free(V); free(out_cpu); free(out_gpu);
        return 0;
    }
    
    flux_rocm_sync();
    flux_rocm_tensor_read(t_out, out_gpu, seq * hidden);
    
    float diff = max_diff(out_cpu, out_gpu, seq * hidden);
    
    flux_rocm_tensor_free(t_Q);
    flux_rocm_tensor_free(t_K);
    flux_rocm_tensor_free(t_V);
    flux_rocm_tensor_free(t_out);
    free(Q); free(K); free(V); free(out_cpu); free(out_gpu);
    
    if (diff < 1e-4) {
        printf("PASS (max diff: %.2e)\n", diff);
        return 1;
    } else {
        printf("FAIL (max diff: %.2e)\n", diff);
        return 0;
    }
}

static int test_rope_2d() {
    printf("Testing RoPE 2D... ");
    
    const int seq = 256, heads = 24, head_dim = 128, axis_dim = 32;
    const int pairs_per_head = axis_dim / 2;
    
    float *x_cpu = (float*)malloc(seq * heads * head_dim * sizeof(float));
    float *x_gpu_out = (float*)malloc(seq * heads * head_dim * sizeof(float));
    float *cos_freq = (float*)malloc(seq * pairs_per_head * sizeof(float));
    float *sin_freq = (float*)malloc(seq * pairs_per_head * sizeof(float));
    
    /* Initialize */
    for (int i = 0; i < seq * heads * head_dim; i++) {
        x_cpu[i] = (float)(i % 1000) / 500.0f - 1.0f;
    }
    for (int i = 0; i < seq * pairs_per_head; i++) {
        float angle = (float)i * 0.01f;
        cos_freq[i] = cosf(angle);
        sin_freq[i] = sinf(angle);
    }
    
    /* GPU */
    flux_rocm_tensor_t t_x = flux_rocm_tensor_create(x_cpu, seq * heads * head_dim);
    flux_rocm_tensor_t t_cos = flux_rocm_tensor_create(cos_freq, seq * pairs_per_head);
    flux_rocm_tensor_t t_sin = flux_rocm_tensor_create(sin_freq, seq * pairs_per_head);
    
    flux_rocm_rope_2d(gpu_ptr(t_x), gpu_ptr(t_cos), gpu_ptr(t_sin),
                      seq, heads, head_dim, axis_dim);
    flux_rocm_sync();
    flux_rocm_tensor_read(t_x, x_gpu_out, seq * heads * head_dim);
    
    /* CPU reference */
    cpu_rope_2d(x_cpu, cos_freq, sin_freq, seq, heads, head_dim, axis_dim);
    
    float diff = max_diff(x_cpu, x_gpu_out, seq * heads * head_dim);
    
    flux_rocm_tensor_free(t_x);
    flux_rocm_tensor_free(t_cos);
    flux_rocm_tensor_free(t_sin);
    free(x_cpu); free(x_gpu_out); free(cos_freq); free(sin_freq);
    
    if (diff < 1e-5) {
        printf("PASS (max diff: %.2e)\n", diff);
        return 1;
    } else {
        printf("FAIL (max diff: %.2e)\n", diff);
        return 0;
    }
}

/* ========================================================================
 * Performance Benchmarks
 * ======================================================================== */

static void bench_kernels() {
    printf("\nKernel Performance Benchmarks:\n");
    
    const int seq = 768, hidden = 3072;
    const int iters = 100;
    
    flux_rocm_tensor_t x = flux_rocm_tensor_alloc(seq * hidden);
    flux_rocm_tensor_t out = flux_rocm_tensor_alloc(seq * hidden);
    flux_rocm_tensor_t weight = flux_rocm_tensor_alloc(hidden);
    flux_rocm_tensor_t gate = flux_rocm_tensor_alloc(seq * hidden);
    flux_rocm_tensor_t up = flux_rocm_tensor_alloc(seq * hidden);
    
    /* Warmup */
    flux_rocm_silu(gpu_ptr(x), seq * hidden);
    flux_rocm_sync();
    
    /* SiLU */
    {
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; i++) {
            flux_rocm_silu(gpu_ptr(x), seq * hidden);
        }
        flux_rocm_sync();
        auto end = std::chrono::high_resolution_clock::now();
        double us = std::chrono::duration<double, std::micro>(end - start).count() / iters;
        double gb_s = (seq * hidden * sizeof(float) * 2.0) / (us * 1000.0);  /* read + write */
        printf("  SiLU (%dx%d): %.1f us (%.1f GB/s)\n", seq, hidden, us, gb_s);
    }
    
    /* SiLU mul */
    {
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; i++) {
            flux_rocm_silu_mul(gpu_ptr(gate), gpu_ptr(up), seq * hidden);
        }
        flux_rocm_sync();
        auto end = std::chrono::high_resolution_clock::now();
        double us = std::chrono::duration<double, std::micro>(end - start).count() / iters;
        double gb_s = (seq * hidden * sizeof(float) * 3.0) / (us * 1000.0);
        printf("  SiLU*mul (%dx%d): %.1f us (%.1f GB/s)\n", seq, hidden, us, gb_s);
    }
    
    /* RMSNorm */
    {
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; i++) {
            flux_rocm_rms_norm(gpu_ptr(out), gpu_ptr(x), gpu_ptr(weight), seq, hidden, 1e-6f);
        }
        flux_rocm_sync();
        auto end = std::chrono::high_resolution_clock::now();
        double us = std::chrono::duration<double, std::micro>(end - start).count() / iters;
        printf("  RMSNorm (%dx%d): %.1f us\n", seq, hidden, us);
    }
    
    /* Softmax */
    {
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; i++) {
            flux_rocm_softmax(gpu_ptr(x), seq, seq);
        }
        flux_rocm_sync();
        auto end = std::chrono::high_resolution_clock::now();
        double us = std::chrono::duration<double, std::micro>(end - start).count() / iters;
        printf("  Softmax (%dx%d): %.1f us\n", seq, seq, us);
    }
    
    /* AdaLN */
    {
        flux_rocm_tensor_t shift = flux_rocm_tensor_alloc(hidden);
        flux_rocm_tensor_t scale = flux_rocm_tensor_alloc(hidden);
        
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; i++) {
            flux_rocm_adaln_norm(gpu_ptr(out), gpu_ptr(x), 
                                 gpu_ptr(shift), gpu_ptr(scale),
                                 seq, hidden, 1e-6f);
        }
        flux_rocm_sync();
        auto end = std::chrono::high_resolution_clock::now();
        double us = std::chrono::duration<double, std::micro>(end - start).count() / iters;
        printf("  AdaLN (%dx%d): %.1f us\n", seq, hidden, us);
        
        flux_rocm_tensor_free(shift);
        flux_rocm_tensor_free(scale);
    }
    
    flux_rocm_tensor_free(x);
    flux_rocm_tensor_free(out);
    flux_rocm_tensor_free(weight);
    flux_rocm_tensor_free(gate);
    flux_rocm_tensor_free(up);
    
    /* Attention benchmark */
    {
        const int attn_seq = 768, attn_heads = 24, attn_head_dim = 128;
        const int attn_hidden = attn_heads * attn_head_dim;
        const float attn_scale = 1.0f / sqrtf((float)attn_head_dim);
        
        flux_rocm_tensor_t Q = flux_rocm_tensor_alloc(attn_seq * attn_hidden);
        flux_rocm_tensor_t K = flux_rocm_tensor_alloc(attn_seq * attn_hidden);
        flux_rocm_tensor_t V = flux_rocm_tensor_alloc(attn_seq * attn_hidden);
        flux_rocm_tensor_t attn_out = flux_rocm_tensor_alloc(attn_seq * attn_hidden);
        
        /* Warmup */
        flux_rocm_attention_fused(gpu_ptr(attn_out), gpu_ptr(Q), gpu_ptr(K), gpu_ptr(V),
                                  attn_seq, attn_seq, attn_heads, attn_head_dim, attn_scale);
        flux_rocm_sync();
        
        const int attn_iters = 20;
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < attn_iters; i++) {
            flux_rocm_attention_fused(gpu_ptr(attn_out), gpu_ptr(Q), gpu_ptr(K), gpu_ptr(V),
                                      attn_seq, attn_seq, attn_heads, attn_head_dim, attn_scale);
        }
        flux_rocm_sync();
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count() / attn_iters;
        
        printf("  Attention (seq=%d, heads=%d): %.2f ms\n", attn_seq, attn_heads, ms);
        
        flux_rocm_tensor_free(Q);
        flux_rocm_tensor_free(K);
        flux_rocm_tensor_free(V);
        flux_rocm_tensor_free(attn_out);
    }
}

int main() {
    printf("=== FLUX ROCm Kernel Test ===\n\n");
    
    if (!flux_rocm_init()) {
        printf("Failed to initialize ROCm\n");
        return 1;
    }
    
    int passed = 0, total = 0;
    
    total++; if (test_silu()) passed++;
    total++; if (test_silu_mul()) passed++;
    total++; if (test_rms_norm()) passed++;
    total++; if (test_softmax()) passed++;
    total++; if (test_adaln_norm()) passed++;
    total++; if (test_attention()) passed++;
    total++; if (test_rope_2d()) passed++;
    
    bench_kernels();
    
    printf("\n=== Results: %d/%d tests passed ===\n", passed, total);
    
    flux_rocm_cleanup();
    
    return (passed == total) ? 0 : 1;
}
