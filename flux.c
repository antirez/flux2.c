/*
 * FLUX Main Implementation
 *
 * Main entry point for the FLUX.2 klein 4B inference engine.
 * Ties together all components: tokenizer, text encoder, VAE, transformer, sampling.
 */

#include "flux.h"
#include "flux_kernels.h"
#include "flux_safetensors.h"
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
typedef struct flux_text_encoder flux_text_encoder_t;

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
 * Text Encoder (Simplified)
 * ======================================================================== */

/*
 * For the klein 4B model, we use a simplified text encoder.
 * The full model would use Mistral 3.1, but that's ~24B parameters.
 *
 * This simplified version uses a smaller transformer for text embedding,
 * or can load pre-computed embeddings.
 */

struct flux_text_encoder {
    int hidden_size;        /* 2560 (such that 3 * 2560 = 7680) */
    int num_layers;         /* e.g., 12 */
    int num_heads;          /* e.g., 20 */
    int vocab_size;
    int max_seq_len;

    /* Embeddings */
    float *token_embeddings;
    float *position_embeddings;

    /* Transformer layers */
    void *layers;  /* Placeholder for actual layer weights */

    /* Output: stack 3 layers and concatenate */
    int output_layers[3];

    /* Working memory */
    float *hidden;
};

/* Simple text encoder forward pass */
static float *text_encoder_forward(flux_text_encoder_t *enc,
                                   const int *tokens, int seq_len) {
    if (!enc) {
        /* Return zero embeddings for testing
         * Without T5-XXL text encoder, we can't properly encode text.
         * Zero embeddings will test if denoising works.
         */
        float *emb = (float *)calloc(seq_len * FLUX_TEXT_DIM, sizeof(float));
        return emb;
    }

    /* TODO: Implement full text encoder forward pass */
    /* For now, just use token embeddings */
    float *emb = (float *)malloc(seq_len * FLUX_TEXT_DIM * sizeof(float));

    for (int s = 0; s < seq_len; s++) {
        int token = tokens[s];
        if (token < 0 || token >= enc->vocab_size) token = 0;

        /* Get embedding and replicate to text_dim */
        for (int d = 0; d < FLUX_TEXT_DIM; d++) {
            int src_d = d % enc->hidden_size;
            emb[s * FLUX_TEXT_DIM + d] = enc->token_embeddings[token * enc->hidden_size + src_d];
        }
    }

    return emb;
}

static void text_encoder_free(flux_text_encoder_t *enc) {
    if (!enc) return;
    free(enc->token_embeddings);
    free(enc->position_embeddings);
    free(enc->hidden);
    free(enc);
}

/* ========================================================================
 * Main Context Structure
 * ======================================================================== */

struct flux_ctx {
    /* Components */
    flux_tokenizer *tokenizer;
    flux_text_encoder_t *text_encoder;
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

    /* State */
    int verbose;
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

    /* Set defaults */
    ctx->max_width = 2048;
    ctx->max_height = 2048;
    ctx->default_steps = 4;
    ctx->default_guidance = 1.0f;
    strncpy(ctx->model_name, "FLUX.2-klein-4B", sizeof(ctx->model_name) - 1);
    strncpy(ctx->model_version, "1.0", sizeof(ctx->model_version) - 1);

    /* Load VAE */
    snprintf(path, sizeof(path), "%s/vae/diffusion_pytorch_model.safetensors", model_dir);
    if (file_exists(path)) {
        fprintf(stderr, "Loading VAE from %s\n", path);
        safetensors_file_t *sf = safetensors_open(path);
        if (sf) {
            ctx->vae = flux_vae_load_safetensors(sf);
            safetensors_close(sf);
            if (!ctx->vae) {
                fprintf(stderr, "Warning: Failed to load VAE weights\n");
            } else {
                fprintf(stderr, "  VAE loaded successfully\n");
            }
        } else {
            fprintf(stderr, "Warning: Cannot open VAE file\n");
        }
    }

    /* Load transformer */
    snprintf(path, sizeof(path), "%s/transformer/diffusion_pytorch_model.safetensors", model_dir);
    if (file_exists(path)) {
        fprintf(stderr, "Loading transformer from %s\n", path);
        safetensors_file_t *sf = safetensors_open(path);
        if (sf) {
            ctx->transformer = flux_transformer_load_safetensors(sf);
            safetensors_close(sf);
            if (!ctx->transformer) {
                fprintf(stderr, "Warning: Failed to load transformer weights\n");
            } else {
                fprintf(stderr, "  Transformer loaded successfully\n");
            }
        } else {
            fprintf(stderr, "Warning: Cannot open transformer file\n");
        }
    }

    /* Load tokenizer vocabulary */
    snprintf(path, sizeof(path), "%s/tokenizer/vocab.json", model_dir);
    if (file_exists(path)) {
        ctx->tokenizer = flux_tokenizer_load(path);
        if (ctx->tokenizer) {
            fprintf(stderr, "  Tokenizer loaded\n");
        }
    }

    /* Verify required components are loaded */
    if (!ctx->vae) {
        set_error("Failed to load VAE - cannot generate images");
        flux_free(ctx);
        return NULL;
    }

    if (!ctx->transformer) {
        set_error("Failed to load transformer - cannot generate images");
        flux_free(ctx);
        return NULL;
    }

    /* Warn about text encoder */
    if (!ctx->text_encoder) {
        fprintf(stderr, "\nWARNING: Text encoder not loaded!\n");
        fprintf(stderr, "  Images will be generated with empty text conditioning.\n");
        fprintf(stderr, "  For proper text-to-image generation, the Qwen3 text encoder\n");
        fprintf(stderr, "  needs to be implemented (8GB model in text_encoder/).\n\n");
    }

    /* Initialize RNG */
    flux_rng_seed((uint64_t)time(NULL));

    return ctx;
}

void flux_free(flux_ctx *ctx) {
    if (!ctx) return;

    flux_tokenizer_free(ctx->tokenizer);
    text_encoder_free(ctx->text_encoder);
    flux_vae_free(ctx->vae);
    flux_transformer_free(ctx->transformer);

    free(ctx);
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

    /* Tokenize */
    int num_tokens;
    int *tokens;

    if (ctx->tokenizer) {
        tokens = flux_tokenize(ctx->tokenizer, prompt, &num_tokens, FLUX_MAX_SEQ_LEN);
    } else {
        /* Simple character-level tokenization as fallback */
        int len = strlen(prompt);
        num_tokens = (len > FLUX_MAX_SEQ_LEN - 2) ? FLUX_MAX_SEQ_LEN - 2 : len;
        tokens = (int *)malloc((num_tokens + 2) * sizeof(int));
        tokens[0] = 1;  /* BOS */
        for (int i = 0; i < num_tokens; i++) {
            tokens[i + 1] = (unsigned char)prompt[i];
        }
        tokens[num_tokens + 1] = 2;  /* EOS */
        num_tokens += 2;
    }

    /* Encode to embeddings */
    float *embeddings = text_encoder_forward(ctx->text_encoder, tokens, num_tokens);

    free(tokens);

    *out_seq_len = num_tokens;
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

    if (ctx->verbose) {
        fprintf(stderr, "Generating %dx%d image with %d steps\n",
                p.width, p.height, p.num_steps);
        fprintf(stderr, "Prompt: %s\n", prompt);
    }

    /* Encode text */
    int text_seq;
    float *text_emb = flux_encode_text(ctx, prompt, &text_seq);
    if (!text_emb) {
        set_error("Failed to encode prompt");
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
        ctx->transformer, ctx->text_encoder,
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
        img = flux_vae_decode(ctx->vae, latent, 1, latent_h, latent_w);
    } else {
        /* Create placeholder image if no VAE */
        img = flux_image_create(p.width, p.height, 3);
        if (img) {
            /* Fill with gradient based on latent */
            for (int y = 0; y < p.height; y++) {
                for (int x = 0; x < p.width; x++) {
                    int lx = x / 16;
                    int ly = y / 16;
                    if (lx < latent_w && ly < latent_h) {
                        int idx = ly * latent_w + lx;
                        float v = latent[idx];
                        v = (v + 3.0f) / 6.0f;  /* Normalize roughly to [0,1] */
                        if (v < 0) v = 0;
                        if (v > 1) v = 1;
                        img->data[(y * p.width + x) * 3 + 0] = (uint8_t)(v * 255);
                        img->data[(y * p.width + x) * 3 + 1] = (uint8_t)(v * 200);
                        img->data[(y * p.width + x) * 3 + 2] = (uint8_t)(v * 150);
                    }
                }
            }
        }
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

    if (ctx->verbose) {
        fprintf(stderr, "Generating %dx%d with external embeddings (%d tokens)\n",
                p.width, p.height, text_seq);
    }

    /* Compute latent dimensions */
    int latent_h = p.height / 16;
    int latent_w = p.width / 16;
    int image_seq_len = latent_h * latent_w;

    /* Initialize noise */
    int64_t seed = (p.seed < 0) ? (int64_t)time(NULL) : p.seed;
    float *z = flux_init_noise(1, FLUX_LATENT_CHANNELS, latent_h, latent_w, seed);

    /* Get official FLUX.2 schedule (matches Python) */
    float *schedule = flux_official_schedule(p.num_steps, image_seq_len);

    if (ctx->verbose) {
        fprintf(stderr, "Schedule: [");
        for (int i = 0; i <= p.num_steps; i++) {
            fprintf(stderr, "%.4f%s", schedule[i], i < p.num_steps ? ", " : "");
        }
        fprintf(stderr, "]\n");
    }

    /* Sample */
    float *latent = flux_sample_euler(
        ctx->transformer, ctx->text_encoder,
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
        img = flux_vae_decode(ctx->vae, latent, 1, latent_h, latent_w);
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

    if (ctx->verbose) {
        fprintf(stderr, "Generating %dx%d with external embeddings (%d tokens) and noise\n",
                p.width, p.height, text_seq);
    }

    /* Copy external noise */
    float *z = (float *)malloc(expected_noise_size * sizeof(float));
    memcpy(z, noise, expected_noise_size * sizeof(float));

    /* Get official FLUX.2 schedule (matches Python) */
    float *schedule = flux_official_schedule(p.num_steps, image_seq_len);

    if (ctx->verbose) {
        fprintf(stderr, "Schedule: [");
        for (int i = 0; i <= p.num_steps; i++) {
            fprintf(stderr, "%.4f%s", schedule[i], i < p.num_steps ? ", " : "");
        }
        fprintf(stderr, "]\n");
    }

    /* Sample */
    float *latent = flux_sample_euler(
        ctx->transformer, ctx->text_encoder,
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
        img = flux_vae_decode(ctx->vae, latent, 1, latent_h, latent_w);
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

    int latent_size = FLUX_LATENT_CHANNELS * latent_h * latent_w;
    int64_t seed = (p.seed < 0) ? (int64_t)time(NULL) : p.seed;
    flux_rng_seed((uint64_t)seed);

    for (int i = 0; i < latent_size; i++) {
        float noise = flux_random_normal();
        img_latent[i] = (1.0f - strength) * img_latent[i] + strength * noise;
    }

    /* Adjust number of steps based on strength */
    int num_steps = (int)(p.num_steps * strength);
    if (num_steps < 1) num_steps = 1;

    /* Get schedule (starting from strength, not 1.0) */
    float *schedule = (float *)malloc((num_steps + 1) * sizeof(float));
    for (int i = 0; i <= num_steps; i++) {
        schedule[i] = strength * (1.0f - (float)i / num_steps);
    }

    /* Sample */
    float *latent = flux_sample_euler(
        ctx->transformer, ctx->text_encoder,
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
        result = flux_vae_decode(ctx->vae, latent, 1, latent_h, latent_w);
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
    return flux_vae_decode(ctx->vae, latent, 1, latent_h, latent_w);
}

float *flux_denoise_step(flux_ctx *ctx, const float *z, float t,
                         const float *text_emb, int text_len,
                         int latent_h, int latent_w) {
    if (!ctx || !z || !text_emb || !ctx->transformer) return NULL;

    return flux_transformer_forward(ctx->transformer,
                                    z, latent_h, latent_w,
                                    text_emb, text_len, t);
}
