/*
 * FLUX Main Implementation
 *
 * Main entry point for the FLUX.2 klein 4B inference engine.
 * Ties together all components: tokenizer, text encoder, VAE, transformer, sampling.
 */

#include "flux.h"
#include "flux_kernels.h"
#include "flux_safetensors.h"
#include "flux_qwen3.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>

/* ========================================================================
 * Forward Declarations for Internal Types
 * ======================================================================== */

typedef struct flux_tokenizer flux_tokenizer;
typedef struct flux_vae flux_vae_t;
typedef struct flux_transformer flux_transformer_t;

/* Internal function declarations */
extern flux_tokenizer *flux_tokenizer_load(const char *path);
extern void flux_tokenizer_free(flux_tokenizer *tok);
extern int *flux_tokenize(flux_tokenizer *tok, const char *text,
                          int *num_tokens, int max_len);

extern flux_vae_t *flux_vae_load(FILE *f);
extern flux_vae_t *flux_vae_load_safetensors(safetensors_file_t *sf);
extern void flux_vae_free(flux_vae_t *vae);
extern float *flux_vae_encode(flux_vae_t *vae, const float *img,
                              int batch, int H, int W, int *out_h, int *out_w);
extern flux_image *flux_vae_decode(flux_vae_t *vae, const float *latent,
                                   int batch, int latent_h, int latent_w);
extern float *flux_image_to_tensor(const flux_image *img);

extern flux_transformer_t *flux_transformer_load(FILE *f);
extern flux_transformer_t *flux_transformer_load_safetensors(safetensors_file_t *sf);
extern void flux_transformer_free(flux_transformer_t *tf);
extern float *flux_transformer_forward(flux_transformer_t *tf,
                                        const float *img_latent, int img_h, int img_w,
                                        const float *txt_emb, int txt_seq,
                                        float timestep);

extern float *flux_sample_euler(void *transformer, void *text_encoder,
                                float *z, int batch, int channels, int h, int w,
                                const float *text_emb, int text_seq,
                                const float *null_emb,
                                const float *schedule, int num_steps,
                                float guidance_scale,
                                void (*progress_callback)(int step, int total));
extern float *flux_linear_schedule(int num_steps);
extern float *flux_init_noise(int batch, int channels, int h, int w, int64_t seed);

/* ========================================================================
 * Text Encoder (Qwen3)
 * ======================================================================== */

/* Qwen3 text encoder is implemented in flux_qwen3.c */

/* ========================================================================
 * Main Context Structure
 * ======================================================================== */

struct flux_ctx {
    /* Components */
    flux_tokenizer *tokenizer;
    qwen3_encoder_t *qwen3_encoder;
    flux_vae_t *vae;
    flux_transformer_t *transformer;

    /* Configuration */
    int max_width;
    int max_height;
    int default_steps;
    float default_guidance;

    /* Model info */
    char model_name[64];
    char model_version[32];
    char model_dir[512];  /* For reloading text encoder if released */
};

/* Global error message */
static char g_error_msg[256] = {0};

const char *flux_get_error(void) {
    return g_error_msg;
}

static void set_error(const char *msg) {
    strncpy(g_error_msg, msg, sizeof(g_error_msg) - 1);
    g_error_msg[sizeof(g_error_msg) - 1] = '\0';
}

/* ========================================================================
 * Model Loading from HuggingFace-style directory with safetensors files
 * ======================================================================== */

static int file_exists(const char *path) {
    struct stat st;
    return stat(path, &st) == 0;
}

flux_ctx *flux_load_dir(const char *model_dir) {
    char path[512];

    flux_ctx *ctx = calloc(1, sizeof(flux_ctx));
    if (!ctx) {
        set_error("Out of memory");
        return NULL;
    }

    /* Set defaults - max 1024x1024 due to O(seq^2) attention memory */
    ctx->max_width = 1024;
    ctx->max_height = 1024;
    ctx->default_steps = 4;
    ctx->default_guidance = 1.0f;
    strncpy(ctx->model_name, "FLUX.2-klein-4B", sizeof(ctx->model_name) - 1);
    strncpy(ctx->model_version, "1.0", sizeof(ctx->model_version) - 1);
    strncpy(ctx->model_dir, model_dir, sizeof(ctx->model_dir) - 1);

    /* Load VAE only at startup (~300MB).
     * Transformer and text encoder are loaded on-demand during generation
     * to support systems with limited RAM (e.g., 16GB). */
    snprintf(path, sizeof(path), "%s/vae/diffusion_pytorch_model.safetensors", model_dir);
    if (file_exists(path)) {
        safetensors_file_t *sf = safetensors_open(path);
        if (sf) {
            ctx->vae = flux_vae_load_safetensors(sf);
            safetensors_close(sf);
        }
    }

    /* Verify VAE is loaded */
    if (!ctx->vae) {
        set_error("Failed to load VAE - cannot generate images");
        flux_free(ctx);
        return NULL;
    }

    /* Verify transformer file exists (will be loaded on-demand) */
    snprintf(path, sizeof(path), "%s/transformer/diffusion_pytorch_model.safetensors", model_dir);
    if (!file_exists(path)) {
        set_error("Transformer model file not found");
        flux_free(ctx);
        return NULL;
    }

    /* Text encoder will also be loaded on-demand */

    /* Initialize RNG */
    flux_rng_seed((uint64_t)time(NULL));

    return ctx;
}

void flux_free(flux_ctx *ctx) {
    if (!ctx) return;

    flux_tokenizer_free(ctx->tokenizer);
    qwen3_encoder_free(ctx->qwen3_encoder);
    flux_vae_free(ctx->vae);
    flux_transformer_free(ctx->transformer);

    free(ctx);
}

void flux_release_text_encoder(flux_ctx *ctx) {
    if (!ctx || !ctx->qwen3_encoder) return;

    qwen3_encoder_free(ctx->qwen3_encoder);
    ctx->qwen3_encoder = NULL;
}

/* Load transformer on-demand if not already loaded */
static int flux_load_transformer_if_needed(flux_ctx *ctx) {
    if (ctx->transformer) return 1;  /* Already loaded */

    char path[512];
    snprintf(path, sizeof(path), "%s/transformer/diffusion_pytorch_model.safetensors",
             ctx->model_dir);

    if (flux_phase_callback) flux_phase_callback("loading transformer", 0);
    safetensors_file_t *sf = safetensors_open(path);
    if (sf) {
        ctx->transformer = flux_transformer_load_safetensors(sf);
        safetensors_close(sf);
    }
    if (flux_phase_callback) flux_phase_callback("loading transformer", 1);

    if (!ctx->transformer) {
        set_error("Failed to load transformer");
        return 0;
    }
    return 1;
}

/* Get transformer for debugging */
void *flux_get_transformer(flux_ctx *ctx) {
    return ctx ? ctx->transformer : NULL;
}

/* ========================================================================
 * Text Encoding
 * ======================================================================== */

float *flux_encode_text(flux_ctx *ctx, const char *prompt, int *out_seq_len) {
    if (!ctx || !prompt) {
        *out_seq_len = 0;
        return NULL;
    }

    /* Reload encoder if it was released */
    if (!ctx->qwen3_encoder && ctx->model_dir[0]) {
        if (flux_phase_callback) flux_phase_callback("reloading encoder", 0);
        ctx->qwen3_encoder = qwen3_encoder_load(ctx->model_dir);
        if (flux_phase_callback) flux_phase_callback("reloading encoder", 1);
        if (!ctx->qwen3_encoder) {
            fprintf(stderr, "Warning: Failed to reload text encoder\n");
        }
    }

    if (!ctx->qwen3_encoder) {
        /* Return zero embeddings if encoder not available */
        *out_seq_len = QWEN3_MAX_SEQ_LEN;
        return (float *)calloc(QWEN3_MAX_SEQ_LEN * QWEN3_TEXT_DIM, sizeof(float));
    }

    /* Encode text using Qwen3 */
    if (flux_phase_callback) flux_phase_callback("encoding text", 0);
    float *embeddings = qwen3_encode_text(ctx->qwen3_encoder, prompt);
    if (flux_phase_callback) flux_phase_callback("encoding text", 1);

    *out_seq_len = QWEN3_MAX_SEQ_LEN;  /* Always 512 */
    return embeddings;
}

/* ========================================================================
 * Image Generation
 * ======================================================================== */

flux_image *flux_generate(flux_ctx *ctx, const char *prompt,
                          const flux_params *params) {
    if (!ctx || !prompt) {
        set_error("Invalid context or prompt");
        return NULL;
    }

    /* Use defaults if params is NULL */
    flux_params p;
    if (params) {
        p = *params;
    } else {
        p = (flux_params)FLUX_PARAMS_DEFAULT;
    }

    /* Validate dimensions */
    if (p.width <= 0) p.width = 1024;
    if (p.height <= 0) p.height = 1024;
    if (p.num_steps <= 0) p.num_steps = 4;  /* Klein default */
    if (p.guidance_scale <= 0) p.guidance_scale = 1.0f;

    /* Ensure dimensions are divisible by 16 */
    p.width = (p.width / 16) * 16;
    p.height = (p.height / 16) * 16;
    if (p.width < 64) p.width = 64;
    if (p.height < 64) p.height = 64;

    /* Encode text */
    int text_seq;
    float *text_emb = flux_encode_text(ctx, prompt, &text_seq);
    if (!text_emb) {
        set_error("Failed to encode prompt");
        return NULL;
    }

    /* Release text encoder to free ~8GB before loading transformer */
    flux_release_text_encoder(ctx);

    /* Load transformer now (after text encoder is freed to reduce peak memory) */
    if (!flux_load_transformer_if_needed(ctx)) {
        free(text_emb);
        return NULL;
    }

    /* Compute latent dimensions */
    int latent_h = p.height / 16;
    int latent_w = p.width / 16;

    /* Initialize noise */
    int64_t seed = (p.seed < 0) ? (int64_t)time(NULL) : p.seed;
    float *z = flux_init_noise(1, FLUX_LATENT_CHANNELS, latent_h, latent_w, seed);

    /* Get schedule */
    float *schedule = flux_linear_schedule(p.num_steps);

    /* Sample */
    float *latent = flux_sample_euler(
        ctx->transformer, ctx->qwen3_encoder,
        z, 1, FLUX_LATENT_CHANNELS, latent_h, latent_w,
        text_emb, text_seq,
        NULL,  /* No null embedding for klein */
        schedule, p.num_steps,
        p.guidance_scale,
        NULL
    );

    free(z);
    free(schedule);
    free(text_emb);

    if (!latent) {
        set_error("Sampling failed");
        return NULL;
    }

    /* Decode latent to image */
    flux_image *img = NULL;
    if (ctx->vae) {
        if (flux_phase_callback) flux_phase_callback("decoding image", 0);
        img = flux_vae_decode(ctx->vae, latent, 1, latent_h, latent_w);
        if (flux_phase_callback) flux_phase_callback("decoding image", 1);
    }

    free(latent);

    return img;
}

/* ========================================================================
 * Generation with Pre-computed Embeddings
 * ======================================================================== */

/* Forward declaration for official schedule */
extern float *flux_official_schedule(int num_steps, int image_seq_len);

flux_image *flux_generate_with_embeddings(flux_ctx *ctx,
                                           const float *text_emb, int text_seq,
                                           const flux_params *params) {
    if (!ctx || !text_emb) {
        set_error("Invalid context or embeddings");
        return NULL;
    }

    /* Load transformer if not already loaded */
    if (!flux_load_transformer_if_needed(ctx)) {
        return NULL;
    }

    flux_params p;
    if (params) {
        p = *params;
    } else {
        p = (flux_params)FLUX_PARAMS_DEFAULT;
    }

    /* Validate dimensions */
    if (p.width <= 0) p.width = 512;
    if (p.height <= 0) p.height = 512;
    if (p.num_steps <= 0) p.num_steps = 4;
    if (p.guidance_scale <= 0) p.guidance_scale = 1.0f;

    p.width = (p.width / 16) * 16;
    p.height = (p.height / 16) * 16;
    if (p.width < 64) p.width = 64;
    if (p.height < 64) p.height = 64;

    /* Compute latent dimensions */
    int latent_h = p.height / 16;
    int latent_w = p.width / 16;
    int image_seq_len = latent_h * latent_w;

    /* Initialize noise */
    int64_t seed = (p.seed < 0) ? (int64_t)time(NULL) : p.seed;
    float *z = flux_init_noise(1, FLUX_LATENT_CHANNELS, latent_h, latent_w, seed);

    /* Get official FLUX.2 schedule (matches Python) */
    float *schedule = flux_official_schedule(p.num_steps, image_seq_len);

    /* Sample */
    float *latent = flux_sample_euler(
        ctx->transformer, ctx->qwen3_encoder,
        z, 1, FLUX_LATENT_CHANNELS, latent_h, latent_w,
        text_emb, text_seq,
        NULL,
        schedule, p.num_steps,
        p.guidance_scale,
        NULL
    );

    free(z);
    free(schedule);

    if (!latent) {
        set_error("Sampling failed");
        return NULL;
    }

    /* Decode latent to image */
    flux_image *img = NULL;
    if (ctx->vae) {
        if (flux_phase_callback) flux_phase_callback("decoding image", 0);
        img = flux_vae_decode(ctx->vae, latent, 1, latent_h, latent_w);
        if (flux_phase_callback) flux_phase_callback("decoding image", 1);
    } else {
        set_error("No VAE loaded");
        free(latent);
        return NULL;
    }

    free(latent);
    return img;
}

/* Generate with external embeddings and external noise */
flux_image *flux_generate_with_embeddings_and_noise(flux_ctx *ctx,
                                                     const float *text_emb, int text_seq,
                                                     const float *noise, int noise_size,
                                                     const flux_params *params) {
    if (!ctx || !text_emb || !noise) {
        set_error("Invalid context, embeddings, or noise");
        return NULL;
    }

    /* Load transformer if not already loaded */
    if (!flux_load_transformer_if_needed(ctx)) {
        return NULL;
    }

    flux_params p;
    if (params) {
        p = *params;
    } else {
        p = (flux_params)FLUX_PARAMS_DEFAULT;
    }

    /* Validate dimensions */
    if (p.width <= 0) p.width = 512;
    if (p.height <= 0) p.height = 512;
    if (p.num_steps <= 0) p.num_steps = 4;
    if (p.guidance_scale <= 0) p.guidance_scale = 1.0f;

    p.width = (p.width / 16) * 16;
    p.height = (p.height / 16) * 16;
    if (p.width < 64) p.width = 64;
    if (p.height < 64) p.height = 64;

    /* Compute latent dimensions */
    int latent_h = p.height / 16;
    int latent_w = p.width / 16;
    int image_seq_len = latent_h * latent_w;
    int expected_noise_size = FLUX_LATENT_CHANNELS * latent_h * latent_w;

    if (noise_size != expected_noise_size) {
        char err[256];
        snprintf(err, sizeof(err), "Noise size mismatch: got %d, expected %d",
                 noise_size, expected_noise_size);
        set_error(err);
        return NULL;
    }

    /* Copy external noise */
    float *z = (float *)malloc(expected_noise_size * sizeof(float));
    memcpy(z, noise, expected_noise_size * sizeof(float));

    /* Get official FLUX.2 schedule (matches Python) */
    float *schedule = flux_official_schedule(p.num_steps, image_seq_len);

    /* Sample */
    float *latent = flux_sample_euler(
        ctx->transformer, ctx->qwen3_encoder,
        z, 1, FLUX_LATENT_CHANNELS, latent_h, latent_w,
        text_emb, text_seq,
        NULL,
        schedule, p.num_steps,
        p.guidance_scale,
        NULL
    );

    free(z);
    free(schedule);

    if (!latent) {
        set_error("Sampling failed");
        return NULL;
    }

    /* Decode latent to image */
    flux_image *img = NULL;
    if (ctx->vae) {
        if (flux_phase_callback) flux_phase_callback("decoding image", 0);
        img = flux_vae_decode(ctx->vae, latent, 1, latent_h, latent_w);
        if (flux_phase_callback) flux_phase_callback("decoding image", 1);
    } else {
        set_error("No VAE loaded");
        free(latent);
        return NULL;
    }

    free(latent);
    return img;
}

/* ========================================================================
 * Image-to-Image Generation
 * ======================================================================== */

flux_image *flux_img2img(flux_ctx *ctx, const char *prompt,
                         const flux_image *input, const flux_params *params) {
    if (!ctx || !prompt || !input) {
        set_error("Invalid parameters");
        return NULL;
    }

    flux_params p;
    if (params) {
        p = *params;
    } else {
        p = (flux_params)FLUX_PARAMS_DEFAULT;
    }

    /* Use input image dimensions if not specified */
    if (p.width <= 0) p.width = input->width;
    if (p.height <= 0) p.height = input->height;

    /* Ensure divisible by 16 */
    p.width = (p.width / 16) * 16;
    p.height = (p.height / 16) * 16;

    /* Resize input if needed */
    flux_image *resized = NULL;
    const flux_image *img_to_use = input;
    if (input->width != p.width || input->height != p.height) {
        resized = flux_image_resize(input, p.width, p.height);
        if (!resized) {
            set_error("Failed to resize input image");
            return NULL;
        }
        img_to_use = resized;
    }

    /* Encode text */
    int text_seq;
    float *text_emb = flux_encode_text(ctx, prompt, &text_seq);
    if (!text_emb) {
        if (resized) flux_image_free(resized);
        set_error("Failed to encode prompt");
        return NULL;
    }

    /* Release text encoder to free ~8GB before loading transformer */
    flux_release_text_encoder(ctx);

    /* Load transformer now (after text encoder is freed to reduce peak memory) */
    if (!flux_load_transformer_if_needed(ctx)) {
        free(text_emb);
        return NULL;
    }

    /* Encode image to latent */
    float *img_tensor = flux_image_to_tensor(img_to_use);
    if (resized) flux_image_free(resized);

    int latent_h, latent_w;
    float *img_latent = NULL;

    if (ctx->vae) {
        img_latent = flux_vae_encode(ctx->vae, img_tensor, 1,
                                     p.height, p.width, &latent_h, &latent_w);
    } else {
        /* Placeholder if no VAE */
        latent_h = p.height / 16;
        latent_w = p.width / 16;
        img_latent = (float *)calloc(FLUX_LATENT_CHANNELS * latent_h * latent_w, sizeof(float));
    }

    free(img_tensor);

    if (!img_latent) {
        free(text_emb);
        set_error("Failed to encode image");
        return NULL;
    }

    /* Add noise based on strength */
    float strength = p.strength;
    if (strength < 0) strength = 0;
    if (strength > 1) strength = 1;

    /*
     * For distilled models like FLUX klein (4-step), we should always use
     * the full number of steps. The strength controls how much noise is added
     * to the image, not how many steps are skipped.
     *
     * Approach: Use the full schedule, but start the latent at the noise
     * level corresponding to strength.
     */
    int num_steps = p.num_steps;
    float *schedule = flux_linear_schedule(num_steps);

    /* The noise level to add matches strength */
    float t_start = strength;

    /* Add noise to match t_start level */
    int latent_size = FLUX_LATENT_CHANNELS * latent_h * latent_w;
    int64_t seed = (p.seed < 0) ? (int64_t)time(NULL) : p.seed;
    flux_rng_seed((uint64_t)seed);

    for (int i = 0; i < latent_size; i++) {
        float noise = flux_random_normal();
        img_latent[i] = (1.0f - t_start) * img_latent[i] + t_start * noise;
    }

    /* Sample */
    float *latent = flux_sample_euler(
        ctx->transformer, ctx->qwen3_encoder,
        img_latent, 1, FLUX_LATENT_CHANNELS, latent_h, latent_w,
        text_emb, text_seq, NULL,
        schedule, num_steps,
        p.guidance_scale, NULL
    );

    free(img_latent);
    free(schedule);
    free(text_emb);

    if (!latent) {
        set_error("Sampling failed");
        return NULL;
    }

    /* Decode */
    flux_image *result = NULL;
    if (ctx->vae) {
        if (flux_phase_callback) flux_phase_callback("decoding image", 0);
        result = flux_vae_decode(ctx->vae, latent, 1, latent_h, latent_w);
        if (flux_phase_callback) flux_phase_callback("decoding image", 1);
    }

    free(latent);
    return result;
}

/* ========================================================================
 * Multi-Reference Generation
 * ======================================================================== */

flux_image *flux_multiref(flux_ctx *ctx, const char *prompt,
                          const flux_image **refs, int num_refs,
                          const flux_params *params) {
    /* For now, just use the first reference if available */
    if (refs && num_refs > 0) {
        return flux_img2img(ctx, prompt, refs[0], params);
    }
    return flux_generate(ctx, prompt, params);
}

/* ========================================================================
 * Utility Functions
 * ======================================================================== */

void flux_set_seed(int64_t seed) {
    flux_rng_seed((uint64_t)seed);
}

const char *flux_model_info(flux_ctx *ctx) {
    static char info[256];
    if (!ctx) {
        return "No model loaded";
    }
    snprintf(info, sizeof(info), "%s v%s (max %dx%d, %d steps)",
             ctx->model_name, ctx->model_version,
             ctx->max_width, ctx->max_height, ctx->default_steps);
    return info;
}

/* ========================================================================
 * Low-level API
 * ======================================================================== */

float *flux_encode_image(flux_ctx *ctx, const flux_image *img,
                         int *out_h, int *out_w) {
    if (!ctx || !img || !ctx->vae) {
        *out_h = *out_w = 0;
        return NULL;
    }

    float *tensor = flux_image_to_tensor(img);
    if (!tensor) return NULL;

    float *latent = flux_vae_encode(ctx->vae, tensor, 1,
                                    img->height, img->width, out_h, out_w);
    free(tensor);
    return latent;
}

flux_image *flux_decode_latent(flux_ctx *ctx, const float *latent,
                               int latent_h, int latent_w) {
    if (!ctx || !latent || !ctx->vae) return NULL;
    if (flux_phase_callback) flux_phase_callback("decoding image", 0);
    flux_image *img = flux_vae_decode(ctx->vae, latent, 1, latent_h, latent_w);
    if (flux_phase_callback) flux_phase_callback("decoding image", 1);
    return img;
}

float *flux_denoise_step(flux_ctx *ctx, const float *z, float t,
                         const float *text_emb, int text_len,
                         int latent_h, int latent_w) {
    if (!ctx || !z || !text_emb) return NULL;

    /* Load transformer if not already loaded */
    if (!flux_load_transformer_if_needed(ctx)) {
        return NULL;
    }

    return flux_transformer_forward(ctx->transformer,
                                    z, latent_h, latent_w,
                                    text_emb, text_len, t);
}
