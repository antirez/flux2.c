/*
 * FLUX.2 klein 4B - Pure C Inference Engine
 *
 * A dependency-free C implementation for image synthesis using the
 * FLUX.2 klein 4B rectified flow transformer model.
 *
 * Usage:
 *   flux_ctx *ctx = flux_load_dir("path/to/model");
 *   if (!ctx) { handle error }
 *
 *   flux_params params = FLUX_PARAMS_DEFAULT;
 *   flux_image *img = flux_generate(ctx, "a cat sitting on a rainbow",
 * &params); flux_image_save(img, "output.png"); flux_image_free(img);
 *   flux_free(ctx);
 */

#ifndef FLUX_H
#define FLUX_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================
 * Configuration Constants
 * ======================================================================== */

/* Model architecture (klein 4B) */
#define FLUX_HIDDEN_SIZE 3072
#define FLUX_NUM_HEADS 24
#define FLUX_HEAD_DIM 128
#define FLUX_NUM_DOUBLE_LAYERS 5
#define FLUX_NUM_SINGLE_LAYERS 20
#define FLUX_MLP_RATIO 3.0f
#define FLUX_TEXT_DIM 7680
#define FLUX_LATENT_CHANNELS 128
#define FLUX_ROPE_THETA 2000.0f

/* VAE architecture */
#define FLUX_VAE_Z_CHANNELS 32
#define FLUX_VAE_BASE_CH 128
#define FLUX_VAE_CH_MULT_0 1
#define FLUX_VAE_CH_MULT_1 2
#define FLUX_VAE_CH_MULT_2 4
#define FLUX_VAE_CH_MULT_3 4
#define FLUX_VAE_NUM_RES 2
#define FLUX_VAE_GROUPS 32

/* Tokenizer */
#define FLUX_MAX_SEQ_LEN 512
#define FLUX_VOCAB_HASH_SIZE 150001

/* ========================================================================
 * Opaque Types
 * ======================================================================== */

typedef struct flux_ctx flux_ctx;
typedef struct flux_image flux_image;
typedef struct flux_tokenizer flux_tokenizer;
typedef struct flux_vae flux_vae_t;

/* ========================================================================
 * Image Structure
 * ======================================================================== */

struct flux_image {
  int width;
  int height;
  int channels;  /* 3 for RGB, 4 for RGBA */
  uint8_t *data; /* Row-major, channel-interleaved */
};

/* ========================================================================
 * Generation Parameters
 * ======================================================================== */

typedef struct {
  int width;            /* Output width (default: 1024) */
  int height;           /* Output height (default: 1024) */
  int num_steps;        /* Inference steps (default: 4 for klein) */
  float guidance_scale; /* CFG scale (default: 1.0 for klein) */
  int64_t seed;         /* Random seed (-1 for random) */
  float strength;       /* For img2img: 0.0-1.0 (default: 0.75) */
} flux_params;

/* Default parameters */
#define FLUX_PARAMS_DEFAULT {256, 256, 4, 1.0f, -1, 0.75f}

/* ========================================================================
 * Core API
 * ======================================================================== */

/*
 * Load model from HuggingFace-style directory containing safetensors files.
 * Directory should contain: vae/, transformer/, tokenizer/ subdirectories.
 * Returns NULL on error.
 */
flux_ctx *flux_load_dir(const char *model_dir);

/*
 * Free model and all associated resources.
 */
void flux_free(flux_ctx *ctx);

/*
 * Release the text encoder to free ~8GB of memory.
 * Call this after encoding if you don't need to encode more prompts.
 * The encoder will be reloaded automatically if needed for a new prompt.
 */
void flux_release_text_encoder(flux_ctx *ctx);

/*
 * Text-to-image generation.
 * Returns newly allocated image, caller must free with flux_image_free().
 * Returns NULL on error.
 */
flux_image *flux_generate(flux_ctx *ctx, const char *prompt,
                          const flux_params *params);

/*
 * Image-to-image generation.
 * Takes an input image and modifies it according to the prompt.
 * strength controls how much the image changes (0.0 = no change, 1.0 = full
 * generation).
 */
flux_image *flux_img2img(flux_ctx *ctx, const char *prompt,
                         const flux_image *input, const flux_params *params);

/*
 * Multi-reference generation (up to 4 reference images for klein).
 */
flux_image *flux_multiref(flux_ctx *ctx, const char *prompt,
                          const flux_image **refs, int num_refs,
                          const flux_params *params);

/*
 * Text-to-image generation with pre-computed embeddings.
 * text_emb: float array of shape [text_seq, FLUX_TEXT_DIM]
 * text_seq: number of text tokens (typically 512)
 */
flux_image *flux_generate_with_embeddings(flux_ctx *ctx, const float *text_emb,
                                          int text_seq,
                                          const flux_params *params);

/*
 * Generate image with external embeddings and external noise.
 * For testing and debugging to match Python exactly.
 * noise: [latent_channels, height/16, width/16] in NCHW format
 * noise_size: total number of floats in noise array
 */
flux_image *flux_generate_with_embeddings_and_noise(
    flux_ctx *ctx, const float *text_emb, int text_seq, const float *noise,
    int noise_size, const flux_params *params);

/* ========================================================================
 * Image I/O
 * ======================================================================== */

/*
 * Load image from file (PNG or PPM).
 * Returns NULL on error.
 */
flux_image *flux_image_load(const char *path);

/*
 * Save image to file (format determined by extension).
 * Supports: .png, .ppm
 * Returns 0 on success, -1 on error.
 */
int flux_image_save(const flux_image *img, const char *path);

/*
 * Create a new image with given dimensions.
 */
flux_image *flux_image_create(int width, int height, int channels);

/*
 * Free image memory.
 */
void flux_image_free(flux_image *img);

/*
 * Resize image using bilinear interpolation.
 */
flux_image *flux_image_resize(const flux_image *img, int new_width,
                              int new_height);

/* ========================================================================
 * Utility Functions
 * ======================================================================== */

/*
 * Set random seed for reproducible generation.
 */
void flux_set_seed(int64_t seed);

/*
 * Get model info string.
 */
const char *flux_model_info(flux_ctx *ctx);

/*
 * Get last error message.
 */
const char *flux_get_error(void);

/* ========================================================================
 * Advanced / Low-level API
 * ======================================================================== */

/*
 * Encode image to latent space using VAE encoder.
 * Returns latent tensor [1, 128, H/16, W/16].
 * Caller must free() the returned pointer.
 */
float *flux_encode_image(flux_ctx *ctx, const flux_image *img, int *out_h,
                         int *out_w);

/*
 * Decode latent to image using VAE decoder.
 */
flux_image *flux_decode_latent(flux_ctx *ctx, const float *latent, int latent_h,
                               int latent_w);

/* Decode latent - GPU Optimized (ROCm) */
flux_image *flux_vae_decode_gpu(flux_vae_t *vae, const float *latent, int batch,
                                int latent_h, int latent_w);

/*
 * Encode text prompt to embeddings.
 * Returns embedding tensor [1, seq_len, 7680].
 * Caller must free() the returned pointer.
 */
float *flux_encode_text(flux_ctx *ctx, const char *prompt, int *out_seq_len);

/*
 * Run single denoising step.
 * z: current latent [1, 128, H, W]
 * t: timestep (0.0 to 1.0)
 * text_emb: text embeddings
 * Returns velocity prediction.
 */
float *flux_denoise_step(flux_ctx *ctx, const float *z, float t,
                         const float *text_emb, int text_len, int latent_h,
                         int latent_w);

#ifdef __cplusplus
}
#endif

#endif /* FLUX_H */
