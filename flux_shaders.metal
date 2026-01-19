/*
 * flux_shaders.metal - Metal compute shaders for FLUX inference
 *
 * These kernels accelerate operations that run on CPU otherwise:
 * - RMSNorm (used in QK normalization)
 * - LayerNorm + AdaLN modulation
 * - SiLU activation
 * - Softmax (row-wise)
 */

#include <metal_stdlib>
using namespace metal;

/* ========================================================================
 * RMSNorm: out[i] = x[i] * rsqrt(mean(x^2) + eps) * weight[i]
 * ======================================================================== */

/* RMSNorm kernel - processes one row per threadgroup
 * x: [seq, hidden], weight: [hidden], out: [seq, hidden]
 */
kernel void rms_norm(
    device const float *x [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant int &hidden [[buffer(3)]],
    constant float &eps [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]
) {
    threadgroup float shared_sum[256];  // For reduction

    device const float *x_row = x + row * hidden;
    device float *out_row = out + row * hidden;

    // Compute partial sum of squares
    float local_sum = 0.0f;
    for (int i = tid; i < hidden; i += threads) {
        float val = x_row[i];
        local_sum += val * val;
    }
    shared_sum[tid] = local_sum;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction for sum
    for (uint stride = threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Compute RMS inverse
    float rms_inv = rsqrt(shared_sum[0] / float(hidden) + eps);

    // Apply normalization with weight
    for (int i = tid; i < hidden; i += threads) {
        out_row[i] = x_row[i] * rms_inv * weight[i];
    }
}

/* QK RMSNorm - processes Q and K for all heads in a sequence position
 * q: [seq, heads*head_dim], k: [seq, heads*head_dim]
 * q_weight, k_weight: [head_dim]
 */
kernel void qk_rms_norm(
    device float *q [[buffer(0)]],
    device float *k [[buffer(1)]],
    device const float *q_weight [[buffer(2)]],
    device const float *k_weight [[buffer(3)]],
    constant int &heads [[buffer(4)]],
    constant int &head_dim [[buffer(5)]],
    constant float &eps [[buffer(6)]],
    uint2 pos [[thread_position_in_grid]]  // (seq_idx, head_idx)
) {
    uint seq_idx = pos.x;
    uint head_idx = pos.y;

    uint hidden = heads * head_dim;
    uint offset = seq_idx * hidden + head_idx * head_dim;

    // RMSNorm for Q
    float sum_sq = 0.0f;
    for (int d = 0; d < head_dim; d++) {
        float val = q[offset + d];
        sum_sq += val * val;
    }
    float rms_inv = rsqrt(sum_sq / float(head_dim) + eps);
    for (int d = 0; d < head_dim; d++) {
        q[offset + d] = q[offset + d] * rms_inv * q_weight[d];
    }

    // RMSNorm for K
    sum_sq = 0.0f;
    for (int d = 0; d < head_dim; d++) {
        float val = k[offset + d];
        sum_sq += val * val;
    }
    rms_inv = rsqrt(sum_sq / float(head_dim) + eps);
    for (int d = 0; d < head_dim; d++) {
        k[offset + d] = k[offset + d] * rms_inv * k_weight[d];
    }
}

/* ========================================================================
 * LayerNorm + AdaLN modulation
 * out = (1 + scale) * norm(x) + shift
 * where norm(x) = (x - mean) / sqrt(var + eps)
 * ======================================================================== */

kernel void adaln_norm(
    device const float *x [[buffer(0)]],
    device const float *shift [[buffer(1)]],
    device const float *scale [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant int &hidden [[buffer(4)]],
    constant float &eps [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]
) {
    threadgroup float shared_sum[256];
    threadgroup float shared_sum_sq[256];

    device const float *x_row = x + row * hidden;
    device float *out_row = out + row * hidden;

    // Compute partial sums for mean and variance
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    for (int i = tid; i < hidden; i += threads) {
        float val = x_row[i];
        local_sum += val;
        local_sum_sq += val * val;
    }
    shared_sum[tid] = local_sum;
    shared_sum_sq[tid] = local_sum_sq;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction
    for (uint stride = threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
            shared_sum_sq[tid] += shared_sum_sq[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Compute mean and std_inv
    float mean = shared_sum[0] / float(hidden);
    float var = shared_sum_sq[0] / float(hidden) - mean * mean;
    float std_inv = rsqrt(var + eps);

    // Apply LayerNorm + AdaLN modulation
    for (int i = tid; i < hidden; i += threads) {
        float norm = (x_row[i] - mean) * std_inv;
        out_row[i] = (1.0f + scale[i]) * norm + shift[i];
    }
}

/* ========================================================================
 * SiLU activation: x * sigmoid(x) = x / (1 + exp(-x))
 * ======================================================================== */

kernel void silu(
    device float *x [[buffer(0)]],
    constant int &n [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < uint(n)) {
        float val = x[gid];
        x[gid] = val / (1.0f + exp(-val));
    }
}

/* SiLU with multiply: gate = silu(gate) * up (SwiGLU style) */
kernel void silu_mul(
    device float *gate [[buffer(0)]],
    device const float *up [[buffer(1)]],
    constant int &n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < uint(n)) {
        float g = gate[gid];
        float silu_g = g / (1.0f + exp(-g));
        gate[gid] = silu_g * up[gid];
    }
}

/* ========================================================================
 * Gated Add: out += gate * proj
 * gate: [hidden], proj: [seq, hidden], out: [seq, hidden]
 * ======================================================================== */

kernel void gated_add(
    device float *out [[buffer(0)]],
    device const float *gate [[buffer(1)]],
    device const float *proj [[buffer(2)]],
    constant int &seq [[buffer(3)]],
    constant int &hidden [[buffer(4)]],
    uint2 pos [[thread_position_in_grid]]  // (seq_idx, hidden_idx)
) {
    uint s = pos.x;
    uint h = pos.y;
    if (s < uint(seq) && h < uint(hidden)) {
        uint idx = s * hidden + h;
        out[idx] += gate[h] * proj[idx];
    }
}

/* ========================================================================
 * Split Fused QKV+MLP Output
 * fused: [seq, fused_dim] where fused_dim = hidden*3 + mlp_hidden*2
 * Splits into: q, k, v [seq, hidden], gate, up [seq, mlp_hidden]
 * ======================================================================== */

kernel void split_qkv_mlp(
    device const float *fused [[buffer(0)]],  // [seq, fused_dim]
    device float *q [[buffer(1)]],            // [seq, hidden]
    device float *k [[buffer(2)]],            // [seq, hidden]
    device float *v [[buffer(3)]],            // [seq, hidden]
    device float *gate [[buffer(4)]],         // [seq, mlp_hidden]
    device float *up [[buffer(5)]],           // [seq, mlp_hidden]
    constant int &seq [[buffer(6)]],
    constant int &hidden [[buffer(7)]],
    constant int &mlp_hidden [[buffer(8)]],
    uint2 pos [[thread_position_in_grid]]  // (seq_idx, element_idx within largest output)
) {
    uint s = pos.x;
    uint e = pos.y;

    if (s >= uint(seq)) return;

    int fused_dim = hidden * 3 + mlp_hidden * 2;
    device const float *row = fused + s * fused_dim;

    // Copy Q (dims 0 to hidden-1)
    if (e < uint(hidden)) {
        q[s * hidden + e] = row[e];
        k[s * hidden + e] = row[hidden + e];
        v[s * hidden + e] = row[hidden * 2 + e];
    }

    // Copy gate and up (dims 0 to mlp_hidden-1)
    if (e < uint(mlp_hidden)) {
        gate[s * mlp_hidden + e] = row[hidden * 3 + e];
        up[s * mlp_hidden + e] = row[hidden * 3 + mlp_hidden + e];
    }
}

/* ========================================================================
 * Concat Attention + MLP outputs for fused projection
 * attn: [seq, hidden], mlp: [seq, mlp_hidden]
 * out: [seq, hidden + mlp_hidden]
 * ======================================================================== */

kernel void concat_attn_mlp(
    device const float *attn [[buffer(0)]],   // [seq, hidden]
    device const float *mlp [[buffer(1)]],    // [seq, mlp_hidden]
    device float *out [[buffer(2)]],          // [seq, hidden + mlp_hidden]
    constant int &seq [[buffer(3)]],
    constant int &hidden [[buffer(4)]],
    constant int &mlp_hidden [[buffer(5)]],
    uint2 pos [[thread_position_in_grid]]  // (seq_idx, element_idx)
) {
    uint s = pos.x;
    uint e = pos.y;

    if (s >= uint(seq)) return;

    int out_dim = hidden + mlp_hidden;
    device float *out_row = out + s * out_dim;

    // Copy attention output
    if (e < uint(hidden)) {
        out_row[e] = attn[s * hidden + e];
    }

    // Copy MLP output
    if (e < uint(mlp_hidden)) {
        out_row[hidden + e] = mlp[s * mlp_hidden + e];
    }
}

/* ========================================================================
 * Softmax (row-wise): out[i] = exp(x[i] - max) / sum(exp(x - max))
 * ======================================================================== */

kernel void softmax(
    device float *x [[buffer(0)]],
    constant int &rows [[buffer(1)]],
    constant int &cols [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]
) {
    threadgroup float shared_max[256];
    threadgroup float shared_sum[256];

    device float *row_ptr = x + row * cols;

    // Find max (for numerical stability)
    float local_max = -INFINITY;
    for (int i = tid; i < cols; i += threads) {
        local_max = max(local_max, row_ptr[i]);
    }
    shared_max[tid] = local_max;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce to find global max
    for (uint stride = threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_max[tid] = max(shared_max[tid], shared_max[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float max_val = shared_max[0];

    // Compute exp and sum
    float local_sum = 0.0f;
    for (int i = tid; i < cols; i += threads) {
        float e = exp(row_ptr[i] - max_val);
        row_ptr[i] = e;  // Store exp temporarily
        local_sum += e;
    }
    shared_sum[tid] = local_sum;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce to find total sum
    for (uint stride = threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sum = shared_sum[0];

    // Normalize
    float inv_sum = 1.0f / sum;
    for (int i = tid; i < cols; i += threads) {
        row_ptr[i] *= inv_sum;
    }
}

/* ========================================================================
 * RoPE (Rotary Position Embeddings)
 * ======================================================================== */

/* Apply 2D RoPE to Q or K tensor
 * x: [seq, heads*head_dim]
 * cos, sin: [seq, head_dim]  (precomputed frequencies)
 */
kernel void apply_rope_2d(
    device float *x [[buffer(0)]],
    device const float *cos_freq [[buffer(1)]],
    device const float *sin_freq [[buffer(2)]],
    constant int &seq [[buffer(3)]],
    constant int &heads [[buffer(4)]],
    constant int &head_dim [[buffer(5)]],
    constant int &axis_dim [[buffer(6)]],  // 32 for FLUX
    uint2 pos [[thread_position_in_grid]]  // (seq_idx, head_idx)
) {
    uint seq_idx = pos.x;
    uint head_idx = pos.y;

    if (seq_idx >= uint(seq) || head_idx >= uint(heads)) return;

    uint hidden = heads * head_dim;
    device float *vec = x + seq_idx * hidden + head_idx * head_dim;
    device const float *cos_row = cos_freq + seq_idx * head_dim;
    device const float *sin_row = sin_freq + seq_idx * head_dim;

    // RoPE rotation for each axis (4 axes of 32 dims each = 128)
    int half_axis = axis_dim / 2;  // 16

    for (int axis = 0; axis < 4; axis++) {
        int axis_offset = axis * axis_dim;
        for (int d = 0; d < half_axis; d++) {
            int i0 = axis_offset + d;
            int i1 = axis_offset + half_axis + d;

            float c = cos_row[i0];
            float s = sin_row[i0];

            float x0 = vec[i0];
            float x1 = vec[i1];

            vec[i0] = x0 * c - x1 * s;
            vec[i1] = x0 * s + x1 * c;
        }
    }
}

/* ========================================================================
 * Unified RoPE for Text+Image (Single Block Forward)
 * Applies different frequency tables to text and image portions in one pass.
 * Text portion: positions [0, img_offset)
 * Image portion: positions [img_offset, seq)
 * ======================================================================== */
kernel void apply_rope_unified(
    device float *x [[buffer(0)]],
    device const float *txt_cos [[buffer(1)]],
    device const float *txt_sin [[buffer(2)]],
    device const float *img_cos [[buffer(3)]],
    device const float *img_sin [[buffer(4)]],
    constant int &seq [[buffer(5)]],
    constant int &img_offset [[buffer(6)]],
    constant int &heads [[buffer(7)]],
    constant int &head_dim [[buffer(8)]],
    constant int &axis_dim [[buffer(9)]],
    uint2 pos [[thread_position_in_grid]]
) {
    uint seq_idx = pos.x;
    uint head_idx = pos.y;

    if (seq_idx >= uint(seq) || head_idx >= uint(heads)) return;

    uint hidden = heads * head_dim;
    device float *vec = x + seq_idx * hidden + head_idx * head_dim;

    // Select appropriate frequency table based on position
    device const float *cos_row;
    device const float *sin_row;

    if (seq_idx < uint(img_offset)) {
        // Text portion: use text frequencies indexed by seq_idx
        cos_row = txt_cos + seq_idx * head_dim;
        sin_row = txt_sin + seq_idx * head_dim;
    } else {
        // Image portion: use image frequencies indexed by (seq_idx - img_offset)
        uint img_idx = seq_idx - uint(img_offset);
        cos_row = img_cos + img_idx * head_dim;
        sin_row = img_sin + img_idx * head_dim;
    }

    // RoPE rotation: apply to consecutive pairs (d, d+1) matching CPU implementation
    // cos[d] == cos[d+1] due to repeat_interleave in frequency generation
    for (int d = 0; d < head_dim; d += 2) {
        float c = cos_row[d];
        float s = sin_row[d];

        float x0 = vec[d];
        float x1 = vec[d + 1];

        // Complex rotation: (x0 + i*x1) * (cos + i*sin)
        vec[d] = x0 * c - x1 * s;
        vec[d + 1] = x1 * c + x0 * s;
    }
}

/* ========================================================================
 * Fused Non-Causal Attention for Transformer (FLUX)
 * Processes all heads in parallel without causal masking.
 *
 * This kernel computes one output row per threadgroup:
 * out[query_idx, head] = softmax(Q @ K^T * scale) @ V
 *
 * Works directly on [seq, heads*head_dim] layout without transpose.
 * Supports different Q and K/V sequence lengths (for joint attention).
 * ======================================================================== */

kernel void attention_fused(
    device const float *Q [[buffer(0)]],      // [seq_q, heads * head_dim]
    device const float *K [[buffer(1)]],      // [seq_k, heads * head_dim]
    device const float *V [[buffer(2)]],      // [seq_k, heads * head_dim]
    device float *out [[buffer(3)]],          // [seq_q, heads * head_dim]
    constant int &seq_q [[buffer(4)]],        // Query sequence length
    constant int &seq_k [[buffer(5)]],        // Key/Value sequence length
    constant int &num_heads [[buffer(6)]],
    constant int &head_dim [[buffer(7)]],
    constant float &scale [[buffer(8)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],   // (query_idx, head_idx, 0)
    uint3 tid_pos [[thread_position_in_threadgroup]],
    uint3 tg_size [[threads_per_threadgroup]]
) {
    // Shared memory for scores and reductions
    threadgroup float shared_scores[1024];  // Up to 1024 seq_k (768 for 256x256)
    threadgroup float shared_max[256];
    threadgroup float shared_sum[256];

    int query_idx = tg_pos.x;
    int head_idx = tg_pos.y;
    uint tid = tid_pos.x;
    uint threads = tg_size.x;

    if (query_idx >= seq_q || head_idx >= num_heads) return;

    int hidden = num_heads * head_dim;

    // Pointers to this position's Q and output (layout: [seq, heads*head_dim])
    device const float *q_row = Q + query_idx * hidden + head_idx * head_dim;
    device float *out_row = out + query_idx * hidden + head_idx * head_dim;

    // K and V have same layout, head offset is same
    device const float *K_head = K + head_idx * head_dim;
    device const float *V_head = V + head_idx * head_dim;

    // ========== Phase 1: Compute Q @ K^T ==========
    float local_max = -INFINITY;

    for (int key_idx = tid; key_idx < seq_k; key_idx += threads) {
        // Dot product: Q[query_idx, head] · K[key_idx, head]
        float dot = 0.0f;
        device const float *k_row = K_head + key_idx * hidden;
        for (int d = 0; d < head_dim; d++) {
            dot += q_row[d] * k_row[d];
        }
        float score = dot * scale;
        shared_scores[key_idx] = score;
        local_max = max(local_max, score);
    }

    shared_max[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ========== Phase 2: Find global max ==========
    for (uint stride = threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride && tid + stride < threads) {
            shared_max[tid] = max(shared_max[tid], shared_max[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float max_val = shared_max[0];

    // ========== Phase 3: Compute exp(score - max) and sum ==========
    float local_sum = 0.0f;
    for (int key_idx = tid; key_idx < seq_k; key_idx += threads) {
        float e = exp(shared_scores[key_idx] - max_val);
        shared_scores[key_idx] = e;
        local_sum += e;
    }

    shared_sum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ========== Phase 4: Find total sum ==========
    for (uint stride = threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride && tid + stride < threads) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_sum = 1.0f / shared_sum[0];

    // ========== Phase 5: Normalize scores ==========
    for (int key_idx = tid; key_idx < seq_k; key_idx += threads) {
        shared_scores[key_idx] *= inv_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ========== Phase 6: Compute output = scores @ V ==========
    for (int d = tid; d < head_dim; d += threads) {
        float acc = 0.0f;
        for (int key_idx = 0; key_idx < seq_k; key_idx++) {
            float v_val = V_head[key_idx * hidden + d];
            acc += shared_scores[key_idx] * v_val;
        }
        out_row[d] = acc;
    }
}

/* ========================================================================
 * Fused Causal Attention for Text Encoder (Qwen3)
 * Processes all heads in parallel with causal masking and GQA support.
 *
 * This kernel computes one output row per threadgroup:
 * out[query_idx, head] = softmax(Q @ K^T * scale + causal_mask) @ V
 *
 * GQA: Multiple Q heads share the same K/V heads
 * (e.g., 32 Q heads / 8 KV heads = 4 Q heads per KV)
 * ======================================================================== */

/* Fused causal attention - one threadgroup per (query_pos, head) pair
 * Q: [seq, num_q_heads * head_dim]
 * K: [seq, num_kv_heads * head_dim]
 * V: [seq, num_kv_heads * head_dim]
 * out: [seq, num_q_heads * head_dim]
 * attn_mask: [seq] - 1 for valid tokens, 0 for padding (optional, can be null)
 */
kernel void causal_attention_fused(
    device const float *Q [[buffer(0)]],
    device const float *K [[buffer(1)]],
    device const float *V [[buffer(2)]],
    device float *out [[buffer(3)]],
    device const int *attn_mask [[buffer(4)]],  // Attention mask (1=valid, 0=padding)
    constant int &seq [[buffer(5)]],
    constant int &num_q_heads [[buffer(6)]],
    constant int &num_kv_heads [[buffer(7)]],
    constant int &head_dim [[buffer(8)]],
    constant float &scale [[buffer(9)]],
    constant int &use_mask [[buffer(10)]],  // Whether to apply attn_mask
    uint3 tg_pos [[threadgroup_position_in_grid]],   // (query_idx, head_idx, 0)
    uint3 tid_pos [[thread_position_in_threadgroup]],
    uint3 tg_size [[threads_per_threadgroup]]
) {
    // Shared memory for scores and reductions
    threadgroup float shared_scores[512];  // For attention scores (up to 512 seq len)
    threadgroup float shared_max[256];
    threadgroup float shared_sum[256];

    int query_idx = tg_pos.x;
    int head_idx = tg_pos.y;
    uint tid = tid_pos.x;
    uint threads = tg_size.x;

    if (query_idx >= seq || head_idx >= num_q_heads) return;

    // GQA: map Q head to KV head
    int heads_per_kv = num_q_heads / num_kv_heads;
    int kv_head_idx = head_idx / heads_per_kv;

    int q_dim = num_q_heads * head_dim;
    int kv_dim = num_kv_heads * head_dim;

    // Pointers to this head's Q, K, V
    device const float *q_row = Q + query_idx * q_dim + head_idx * head_dim;
    device const float *K_head = K + kv_head_idx * head_dim;
    device const float *V_head = V + kv_head_idx * head_dim;
    device float *out_row = out + query_idx * q_dim + head_idx * head_dim;

    // ========== Phase 1: Compute Q @ K^T with causal mask ==========
    // Each thread computes scores for a subset of key positions
    float local_max = -INFINITY;

    for (int key_idx = tid; key_idx < seq; key_idx += threads) {
        // Causal mask: only attend to positions <= query_idx
        // Attention mask: only attend to valid tokens (mask[key_idx] != 0)
        bool masked = (key_idx > query_idx);
        if (use_mask && attn_mask[key_idx] == 0) {
            masked = true;
        }

        if (masked) {
            shared_scores[key_idx] = -INFINITY;
        } else {
            // Dot product: Q[query_idx, head] · K[key_idx, kv_head]
            float dot = 0.0f;
            device const float *k_row = K_head + key_idx * kv_dim;
            for (int d = 0; d < head_dim; d++) {
                dot += q_row[d] * k_row[d];
            }
            float score = dot * scale;
            shared_scores[key_idx] = score;
            local_max = max(local_max, score);
        }
    }

    // Store local max for reduction
    shared_max[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ========== Phase 2: Find global max (parallel reduction) ==========
    for (uint stride = threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride && tid + stride < threads) {
            shared_max[tid] = max(shared_max[tid], shared_max[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float max_val = shared_max[0];

    // ========== Phase 3: Compute exp(score - max) and sum ==========
    float local_sum = 0.0f;
    for (int key_idx = tid; key_idx < seq; key_idx += threads) {
        float score = shared_scores[key_idx];
        if (score > -1e30f) {  // Not masked
            float e = exp(score - max_val);
            shared_scores[key_idx] = e;
            local_sum += e;
        } else {
            shared_scores[key_idx] = 0.0f;
        }
    }

    shared_sum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ========== Phase 4: Find total sum (parallel reduction) ==========
    for (uint stride = threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride && tid + stride < threads) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sum = shared_sum[0];
    float inv_sum = 1.0f / sum;

    // ========== Phase 5: Normalize scores ==========
    for (int key_idx = tid; key_idx < seq; key_idx += threads) {
        shared_scores[key_idx] *= inv_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ========== Phase 6: Compute output = scores @ V ==========
    // Each thread computes a subset of output dimensions
    for (int d = tid; d < head_dim; d += threads) {
        float acc = 0.0f;
        for (int key_idx = 0; key_idx < seq; key_idx++) {
            float score = shared_scores[key_idx];
            if (score > 0.0f) {  // Skip zeros (masked positions)
                float v_val = V_head[key_idx * kv_dim + d];
                acc += score * v_val;
            }
        }
        out_row[d] = acc;
    }
}
