/*
 * FLUX CUDA Attention Helpers
 *
 * CUDA-only helpers for attention operations that keep intermediate tensors
 * on GPU and run softmax/masking directly on device.
 */

#ifndef FLUX_CUDA_H
#define FLUX_CUDA_H

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Single-head attention:
 * out[seq_q, head_dim] = softmax((q @ k^T) * scale + masks) @ v
 *
 * q: [seq_q, head_dim]
 * k: [seq_k, head_dim]
 * v: [seq_k, head_dim]
 *
 * causal: apply causal mask (j > i masked)
 * attention_mask: optional [seq_k] mask (0 = masked, non-zero = valid), can be NULL
 * prefer_bf16: try BF16 tensor-core QK matmul via cuBLASLt first
 *
 * Returns 1 on success, 0 on failure (caller should fall back).
 */
int flux_cuda_attention_single(float *out,
                               const float *q, const float *k, const float *v,
                               int seq_q, int seq_k, int head_dim,
                               float scale, int causal,
                               const int *attention_mask,
                               int prefer_bf16);

/*
 * Batched attention for equal-head layouts:
 * q/k/v/out are [heads, seq, head_dim] contiguous (head-major).
 *
 * causal and attention_mask semantics are the same as single-head.
 * Returns 1 on success, 0 on failure.
 */
int flux_cuda_attention_batched(float *out,
                                const float *q, const float *k, const float *v,
                                int heads, int seq_q, int seq_k, int head_dim,
                                float scale, int causal,
                                const int *attention_mask);

/*
 * Batched attention with sequence-major input/output layout:
 * q/k/v/out are [seq, heads * head_dim] contiguous (sequence-major).
 *
 * Internally transposes on GPU, runs the same CUDA batched attention path,
 * then transposes back. This avoids CPU transpose overhead in single-block
 * transformer paths that already operate in [seq, hidden] layout.
 *
 * Returns 1 on success, 0 on failure.
 */
int flux_cuda_attention_batched_shd(float *out,
                                    const float *q, const float *k, const float *v,
                                    int heads, int seq_q, int seq_k, int head_dim,
                                    float scale, int causal,
                                    const int *attention_mask);

/* Same as flux_cuda_attention_batched_shd(), but operates directly on device
 * pointers and leaves output on device (no host/device copies). */
int flux_cuda_attention_batched_shd_device(float *d_out,
                                           const float *d_q, const float *d_k, const float *d_v,
                                           int heads, int seq_q, int seq_k, int head_dim,
                                           float scale, int causal,
                                           const int *attention_mask);

/* Set CUDA stream used by CUDA attention/op helpers.
 * Pass NULL to use the default stream. */
int flux_cuda_ops_set_stream(void *stream_handle);

/* CUDA device kernels used by CUDA-resident transformer blocks. */
int flux_cuda_adaln_norm_device(float *d_out, const float *d_x,
                                const float *d_shift, const float *d_scale,
                                int seq, int hidden, float eps);
int flux_cuda_split_qkv_mlp_device(const float *d_fused,
                                   float *d_q, float *d_k, float *d_v,
                                   float *d_gate, float *d_up,
                                   int seq, int hidden, int mlp_hidden);
int flux_cuda_qk_rms_norm_device(float *d_q, float *d_k,
                                 const float *d_q_weight, const float *d_k_weight,
                                 int seq, int heads, int head_dim, float eps);
int flux_cuda_rope_unified_device(float *d_q, float *d_k,
                                  const float *d_txt_cos, const float *d_txt_sin,
                                  const float *d_img_cos, const float *d_img_sin,
                                  int seq, int img_offset, int heads, int head_dim);
int flux_cuda_silu_mul_device(float *d_gate, const float *d_up, int n);
int flux_cuda_concat_attn_mlp_device(const float *d_attn, const float *d_mlp,
                                     float *d_out, int seq, int hidden, int mlp_hidden);
int flux_cuda_gated_add_device(float *d_hidden, const float *d_gate,
                               const float *d_proj, int seq, int hidden);
int flux_cuda_concat_seq_device(float *d_out, const float *d_a, const float *d_b,
                                int seq_a, int seq_b, int hidden);

/* Generic CUDA tensor ops used by CUDA-resident VAE decode path. */
int flux_cuda_im2col_nchw_rows_device(float *d_col, const float *d_in,
                                      int in_ch, int H, int W,
                                      int kH, int kW, int stride, int padding,
                                      int outH, int outW,
                                      int row_offset, int tile_h);
int flux_cuda_add_bias_rows_device(float *d_rows, const float *d_bias,
                                   int rows, int cols);
int flux_cuda_rows_to_nchw_tile_device(float *d_out, const float *d_rows,
                                       int out_ch, int outH, int outW,
                                       int row_offset, int tile_h);
int flux_cuda_nchw_to_rows_device(float *d_rows, const float *d_x,
                                  int channels, int H, int W);
int flux_cuda_rows_to_nchw_device(float *d_x, const float *d_rows,
                                  int channels, int H, int W);
int flux_cuda_group_norm_nchw_device(float *d_out, const float *d_x,
                                     const float *d_gamma, const float *d_beta,
                                     int batch, int channels, int H, int W,
                                     int num_groups, float eps);
int flux_cuda_silu_device(float *d_x, int n);
int flux_cuda_add_inplace_device(float *d_a, const float *d_b, int n);
int flux_cuda_upsample_nearest2x_nchw_device(float *d_out, const float *d_in,
                                             int channels, int H, int W);

#ifdef __cplusplus
}
#endif

#endif /* FLUX_CUDA_H */
