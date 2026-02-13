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
#if defined(__unix__) || defined(__APPLE__)
#include <pthread.h>
#endif

#ifdef USE_METAL
#include "flux_metal.h"
#endif

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
extern flux_transformer_t *flux_transformer_load_safetensors_mmap(safetensors_file_t *sf);
extern void flux_transformer_free(flux_transformer_t *tf);
extern float *flux_transformer_forward(flux_transformer_t *tf,
                                        const float *img_latent, int img_h, int img_w,
                                        const float *txt_emb, int txt_seq,
                                        float timestep);

extern float *flux_sample_euler(void *transformer, void *text_encoder,
                                float *z, int batch, int channels, int h, int w,
                                const float *text_emb, int text_seq,
                                const float *schedule, int num_steps,
                                void (*progress_callback)(int step, int total));
extern float *flux_sample_euler_with_refs(void *transformer, void *text_encoder,
                                          float *z, int batch, int channels, int h, int w,
                                          const float *ref_latent, int ref_h, int ref_w,
                                          int t_offset,
                                          const float *text_emb, int text_seq,
                                          const float *schedule, int num_steps,
                                          void (*progress_callback)(int step, int total));

/* Multi-reference support */
typedef struct {
    const float *latent;
    int h, w;
    int t_offset;
} flux_ref_t;

extern float *flux_sample_euler_with_multi_refs(void *transformer, void *text_encoder,
                                                float *z, int batch, int channels, int h, int w,
                                                const flux_ref_t *refs, int num_refs,
                                                const float *text_emb, int text_seq,
                                                const float *schedule, int num_steps,
                                                void (*progress_callback)(int step, int total));

/* CFG sampling (for base model) */
extern float *flux_sample_euler_cfg(void *transformer, void *text_encoder,
                                     float *z, int batch, int channels, int h, int w,
                                     const float *text_emb_cond, int text_seq_cond,
                                     const float *text_emb_uncond, int text_seq_uncond,
                                     float guidance_scale,
                                     const float *schedule, int num_steps,
                                     void (*progress_callback)(int step, int total));
extern float *flux_sample_euler_cfg_with_refs(void *transformer, void *text_encoder,
                                               float *z, int batch, int channels, int h, int w,
                                               const float *ref_latent, int ref_h, int ref_w,
                                               int t_offset,
                                               const float *text_emb_cond, int text_seq_cond,
                                               const float *text_emb_uncond, int text_seq_uncond,
                                               float guidance_scale,
                                               const float *schedule, int num_steps,
                                               void (*progress_callback)(int step, int total));
extern float *flux_sample_euler_cfg_with_multi_refs(void *transformer, void *text_encoder,
                                                     float *z, int batch, int channels, int h, int w,
                                                     const flux_ref_t *refs, int num_refs,
                                                     const float *text_emb_cond, int text_seq_cond,
                                                     const float *text_emb_uncond, int text_seq_uncond,
                                                     float guidance_scale,
                                                     const float *schedule, int num_steps,
                                                     void (*progress_callback)(int step, int total));

extern float *flux_linear_schedule(int num_steps);
extern float *flux_power_schedule(int num_steps, float alpha);
extern float *flux_official_schedule(int num_steps, int image_seq_len);
extern float *flux_init_noise(int batch, int channels, int h, int w, int64_t seed);

/* Return schedule based on params: linear, power, or official shifted sigmoid. */
static float *flux_selected_schedule(const flux_params *p, int image_seq_len) {
    if (p->power_schedule)
        return flux_power_schedule(p->num_steps, p->power_alpha);
    if (p->linear_schedule)
        return flux_linear_schedule(p->num_steps);
    return flux_official_schedule(p->num_steps, image_seq_len);
}

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
    int is_distilled;  /* 1 = distilled (4-step), 0 = base (50-step CFG) */
    int text_dim;      /* Text embedding dimension (7680 for 4B, varies for 9B) */
    int is_non_commercial; /* 1 if model has non-commercial license (9B) */
    int num_heads;     /* Transformer attention heads (24 for 4B, 32 for 9B) */

    /* Model info */
    char model_name[64];
    char model_version[32];
    char model_dir[512];  /* For reloading text encoder if released */

    /* Memory mode */
    int use_mmap;  /* Use mmap for text encoder (lower memory, slower) */
    int keep_text_encoder_loaded;  /* Keep Qwen resident across generations */
};

/* Global error message */
static char g_error_msg[256] = {0};

const char *flux_get_error(void) {
    return g_error_msg;
}

void flux_set_step_image_callback(flux_ctx *ctx, flux_step_image_cb_t callback) {
    flux_step_image_callback = callback;
    flux_step_image_vae = callback ? ctx->vae : NULL;
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
    char path[1024];

    flux_ctx *ctx = calloc(1, sizeof(flux_ctx));
    if (!ctx) {
        set_error("Out of memory");
        return NULL;
    }

    /* Set defaults - max 1792x1792 (requires ~18GB VAE work buffers) */
    ctx->max_width = FLUX_VAE_MAX_DIM;
    ctx->max_height = FLUX_VAE_MAX_DIM;
    strncpy(ctx->model_version, "1.0", sizeof(ctx->model_version) - 1);
    strncpy(ctx->model_dir, model_dir, sizeof(ctx->model_dir) - 1);

    /* Autodetect model type from model_index.json.
     * Distilled model has "is_distilled": true, base model does not. */
    ctx->is_distilled = 1;  /* Default to distilled */
    snprintf(path, sizeof(path), "%s/model_index.json", model_dir);
    if (file_exists(path)) {
        FILE *f = fopen(path, "r");
        if (f) {
            char buf[4096];
            size_t n = fread(buf, 1, sizeof(buf) - 1, f);
            buf[n] = '\0';
            fclose(f);
            /* If "is_distilled" is present and true, it's distilled.
             * If absent, it's the base model. */
            if (!strstr(buf, "\"is_distilled\": true") &&
                !strstr(buf, "\"is_distilled\":true")) {
                ctx->is_distilled = 0;
            }
        }
    }

    /* Read transformer/config.json to determine model size (4B vs 9B)
     * and text embedding dimension. */
    int num_heads = 24;  /* default 4B */
    ctx->text_dim = 7680;  /* default 4B: 3 * 2560 */
    snprintf(path, sizeof(path), "%s/transformer/config.json", model_dir);
    if (file_exists(path)) {
        FILE *f = fopen(path, "r");
        if (f) {
            char buf[4096];
            size_t n = fread(buf, 1, sizeof(buf) - 1, f);
            buf[n] = '\0';
            fclose(f);
            char *p;
            if ((p = strstr(buf, "\"num_attention_heads\""))) {
                if ((p = strchr(p, ':'))) num_heads = atoi(p + 1);
            }
            int joint_dim = 0;
            if ((p = strstr(buf, "\"joint_attention_dim\""))) {
                if ((p = strchr(p, ':'))) joint_dim = atoi(p + 1);
            }
            if (joint_dim > 0) ctx->text_dim = joint_dim;
        }
    }

    /* Determine model variant name based on architecture size.
     * 4B: 24 heads (hidden=3072), 9B: more heads (hidden>3072). */
    int hidden_size = num_heads * 128;  /* head_dim is always 128 */
    const char *size_label = (hidden_size > 3072) ? "9B" : "4B";
    ctx->is_non_commercial = (hidden_size > 3072) ? 1 : 0;
    ctx->num_heads = num_heads;

    if (ctx->is_distilled) {
        ctx->default_steps = 4;
        ctx->default_guidance = 1.0f;
        snprintf(ctx->model_name, sizeof(ctx->model_name),
                 "FLUX.2-klein-%s", size_label);
    } else {
        ctx->default_steps = 50;
        ctx->default_guidance = 4.0f;
        snprintf(ctx->model_name, sizeof(ctx->model_name),
                 "FLUX.2-klein-base-%s", size_label);
    }

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

    /* Verify transformer dir exists (will be loaded on-demand) */
    snprintf(path, sizeof(path), "%s/transformer/config.json", model_dir);
    if (!file_exists(path)) {
        /* Fallback: check for single safetensors file */
        snprintf(path, sizeof(path), "%s/transformer/diffusion_pytorch_model.safetensors", model_dir);
        if (!file_exists(path)) {
            set_error("Transformer model not found (missing config.json and safetensors)");
            flux_free(ctx);
            return NULL;
        }
    }
    /* Text encoder and transformer are loaded on-demand to reduce peak memory. */

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

void flux_set_mmap(flux_ctx *ctx, int enable) {
    if (ctx) ctx->use_mmap = enable;
}

void flux_set_keep_text_encoder(flux_ctx *ctx, int enable) {
    if (ctx) ctx->keep_text_encoder_loaded = enable ? 1 : 0;
}

int flux_is_distilled(flux_ctx *ctx) {
    return ctx ? ctx->is_distilled : 1;
}

void flux_set_base_mode(flux_ctx *ctx) {
    if (!ctx) return;
    ctx->is_distilled = 0;
    ctx->default_steps = 50;
    ctx->default_guidance = 4.0f;
    const char *size_label = ctx->is_non_commercial ? "9B" : "4B";
    snprintf(ctx->model_name, sizeof(ctx->model_name),
             "FLUX.2-klein-base-%s", size_label);
}

void flux_release_text_encoder(flux_ctx *ctx) {
    if (!ctx || !ctx->qwen3_encoder) return;

    qwen3_encoder_free(ctx->qwen3_encoder);
    ctx->qwen3_encoder = NULL;

#ifdef USE_METAL
    /* Reset all GPU state to ensure clean slate for transformer.
     * This clears weight caches, activation pools, and pending commands. */
    flux_metal_reset();
#endif
}

/* Load transformer on-demand if not already loaded. */
static int flux_load_transformer_internal(flux_ctx *ctx, int emit_phase_callbacks) {
    if (ctx->transformer) return 1;  /* Already loaded */

    char path[1024];
    snprintf(path, sizeof(path), "%s/transformer/diffusion_pytorch_model.safetensors",
             ctx->model_dir);

    if (emit_phase_callbacks && flux_phase_callback) flux_phase_callback("Loading FLUX.2 transformer", 0);
    safetensors_file_t *sf = safetensors_open(path);
    if (sf) {
        if (ctx->use_mmap) {
            /* Mmap mode: load only small weights, keep sf open for on-demand loading.
             * The transformer takes ownership of sf and will close it on free. */
            ctx->transformer = flux_transformer_load_safetensors_mmap(sf);
        } else {
            ctx->transformer = flux_transformer_load_safetensors(sf);
            safetensors_close(sf);
        }
    }
    if (emit_phase_callbacks && flux_phase_callback) flux_phase_callback("Loading FLUX.2 transformer", 1);

    if (!ctx->transformer) {
        set_error("Failed to load transformer");
        return 0;
    }
    return 1;
}

static int flux_load_transformer_if_needed(flux_ctx *ctx) {
    return flux_load_transformer_internal(ctx, 1);
}

/* Load transformer with server-friendly fallback:
 * keep text encoder resident when requested, but retry after releasing it
 * if peak memory is too high during first transformer load. */
static int flux_ensure_transformer_ready(flux_ctx *ctx) {
    if (ctx->transformer) return 1;

    if (!ctx->keep_text_encoder_loaded && ctx->qwen3_encoder) {
        flux_release_text_encoder(ctx);
        return flux_load_transformer_if_needed(ctx);
    }

    if (flux_load_transformer_if_needed(ctx)) {
        return 1;
    }

    if (ctx->keep_text_encoder_loaded && ctx->qwen3_encoder) {
        fprintf(stderr,
                "Warning: Transformer load failed with persistent text encoder; "
                "retrying after releasing text encoder\n");
        flux_release_text_encoder(ctx);
        return flux_load_transformer_if_needed(ctx);
    }

    return 0;
}

#if defined(__unix__) || defined(__APPLE__)
typedef struct {
    flux_ctx *ctx;
    pthread_t thread;
    int started;
    int ok;
} flux_tf_preload_task_t;

static int flux_overlap_preload_enabled(void) {
    static int cached = -1;
    if (cached == -1) {
        cached = getenv("FLUX_OVERLAP_PRELOAD") ? 1 : 0;
    }
    return cached;
}

static void *flux_transformer_preload_thread(void *arg) {
    flux_tf_preload_task_t *task = (flux_tf_preload_task_t *)arg;
    task->ok = flux_load_transformer_internal(task->ctx, 0);
    return NULL;
}

static void flux_transformer_preload_begin(flux_ctx *ctx, flux_tf_preload_task_t *task) {
    if (!task) return;
    memset(task, 0, sizeof(*task));
    task->ctx = ctx;

    if (!ctx || ctx->transformer) return;
    if (!flux_overlap_preload_enabled()) return;

    if (pthread_create(&task->thread, NULL, flux_transformer_preload_thread, task) == 0) {
        task->started = 1;
    }
}

static void flux_transformer_preload_join(flux_tf_preload_task_t *task) {
    if (!task || !task->started) return;
    pthread_join(task->thread, NULL);
    task->started = 0;
}
#else
typedef struct { int started; int ok; } flux_tf_preload_task_t;
static void flux_transformer_preload_begin(flux_ctx *ctx, flux_tf_preload_task_t *task) {
    (void)ctx;
    if (task) memset(task, 0, sizeof(*task));
}
static void flux_transformer_preload_join(flux_tf_preload_task_t *task) { (void)task; }
#endif

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

    /* Load encoder if not already loaded */
    if (!ctx->qwen3_encoder && ctx->model_dir[0]) {
        if (flux_phase_callback) flux_phase_callback("Loading Qwen3 encoder", 0);
        ctx->qwen3_encoder = qwen3_encoder_load(ctx->model_dir, ctx->use_mmap);
        if (flux_phase_callback) flux_phase_callback("Loading Qwen3 encoder", 1);
        if (!ctx->qwen3_encoder) {
            fprintf(stderr, "Warning: Failed to load Qwen3 text encoder\n");
        }
    }

    if (!ctx->qwen3_encoder) {
        /* Return zero embeddings if encoder not available */
        *out_seq_len = QWEN3_MAX_SEQ_LEN;
        return (float *)calloc(QWEN3_MAX_SEQ_LEN * ctx->text_dim, sizeof(float));
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
    if (p.width <= 0) p.width = FLUX_DEFAULT_WIDTH;
    if (p.height <= 0) p.height = FLUX_DEFAULT_HEIGHT;
    if (p.num_steps <= 0) p.num_steps = ctx->default_steps;
    float guidance = (p.guidance > 0) ? p.guidance : ctx->default_guidance;

    /* Ensure dimensions are divisible by 16 */
    p.width = (p.width / 16) * 16;
    p.height = (p.height / 16) * 16;
    if (p.width < 64) p.width = 64;
    if (p.height < 64) p.height = 64;
    if (p.width > FLUX_VAE_MAX_DIM || p.height > FLUX_VAE_MAX_DIM) {
        set_error("Image dimensions exceed maximum (1792x1792)");
        return NULL;
    }

    /* Optional overlap: load transformer in parallel while text is encoding.
     * This is disabled by default and enabled via FLUX_OVERLAP_PRELOAD=1. */
    flux_tf_preload_task_t tf_preload;
    flux_transformer_preload_begin(ctx, &tf_preload);

    /* Encode text (and unconditioned text for CFG in base model) */
    int text_seq;
    float *text_emb = flux_encode_text(ctx, prompt, &text_seq);
    if (!text_emb) {
        flux_transformer_preload_join(&tf_preload);
        set_error("Failed to encode prompt");
        return NULL;
    }

    float *text_emb_uncond = NULL;
    int text_seq_uncond = 0;
    if (!ctx->is_distilled) {
        text_emb_uncond = flux_encode_text(ctx, "", &text_seq_uncond);
        if (!text_emb_uncond) {
            flux_transformer_preload_join(&tf_preload);
            free(text_emb);
            set_error("Failed to encode empty prompt for CFG");
            return NULL;
        }
    }

    /* Ensure any async preload attempt is complete before load checks/fallback. */
    flux_transformer_preload_join(&tf_preload);

    /* Ensure transformer is ready (with optional low-memory fallback). */
    if (!flux_ensure_transformer_ready(ctx)) {
        free(text_emb);
        free(text_emb_uncond);
        return NULL;
    }

    /* Compute latent dimensions */
    int latent_h = p.height / 16;
    int latent_w = p.width / 16;
    int image_seq_len = latent_h * latent_w;

    /* Initialize noise */
    int64_t seed = (p.seed < 0) ? (int64_t)time(NULL) : p.seed;
    float *z = flux_init_noise(1, FLUX_LATENT_CHANNELS, latent_h, latent_w, seed);

    /* Get schedule */
    float *schedule = flux_selected_schedule(&p, image_seq_len);

    /* Sample */
    float *latent;
    if (ctx->is_distilled) {
        latent = flux_sample_euler(
            ctx->transformer, ctx->qwen3_encoder,
            z, 1, FLUX_LATENT_CHANNELS, latent_h, latent_w,
            text_emb, text_seq,
            schedule, p.num_steps,
            NULL
        );
    } else {
        latent = flux_sample_euler_cfg(
            ctx->transformer, ctx->qwen3_encoder,
            z, 1, FLUX_LATENT_CHANNELS, latent_h, latent_w,
            text_emb, text_seq,
            text_emb_uncond, text_seq_uncond,
            guidance,
            schedule, p.num_steps,
            NULL
        );
    }

    free(z);
    free(schedule);
    free(text_emb);
    free(text_emb_uncond);

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

flux_image *flux_generate_with_embeddings(flux_ctx *ctx,
                                           const float *text_emb, int text_seq,
                                           const flux_params *params) {
    if (!ctx || !text_emb) {
        set_error("Invalid context or embeddings");
        return NULL;
    }

    /* This API only supports the distilled (non-CFG) sampler since the
     * caller provides a single embedding.  Warn if used with a base model
     * because results will be incorrect without CFG. */
    if (!ctx->is_distilled) {
        fprintf(stderr, "Warning: flux_generate_with_embeddings() does not "
                        "support CFG. Use flux_generate() for base models.\n");
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
    if (p.width <= 0) p.width = FLUX_DEFAULT_WIDTH;
    if (p.height <= 0) p.height = FLUX_DEFAULT_HEIGHT;
    if (p.num_steps <= 0) p.num_steps = ctx->default_steps;

    p.width = (p.width / 16) * 16;
    p.height = (p.height / 16) * 16;
    if (p.width < 64) p.width = 64;
    if (p.height < 64) p.height = 64;
    if (p.width > FLUX_VAE_MAX_DIM || p.height > FLUX_VAE_MAX_DIM) {
        set_error("Image dimensions exceed maximum (1792x1792)");
        return NULL;
    }

    /* Compute latent dimensions */
    int latent_h = p.height / 16;
    int latent_w = p.width / 16;
    int image_seq_len = latent_h * latent_w;

    /* Initialize noise */
    int64_t seed = (p.seed < 0) ? (int64_t)time(NULL) : p.seed;
    float *z = flux_init_noise(1, FLUX_LATENT_CHANNELS, latent_h, latent_w, seed);

    /* Get schedule */
    float *schedule = flux_selected_schedule(&p, image_seq_len);

    /* Sample - note: pre-computed embeddings only support distilled path.
     * CFG requires two embeddings which the caller doesn't provide. */
    float *latent = flux_sample_euler(
        ctx->transformer, ctx->qwen3_encoder,
        z, 1, FLUX_LATENT_CHANNELS, latent_h, latent_w,
        text_emb, text_seq,
        schedule, p.num_steps,
        NULL  /* progress_callback */
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
    if (p.width <= 0) p.width = FLUX_DEFAULT_WIDTH;
    if (p.height <= 0) p.height = FLUX_DEFAULT_HEIGHT;
    if (p.num_steps <= 0) p.num_steps = ctx->default_steps;

    p.width = (p.width / 16) * 16;
    p.height = (p.height / 16) * 16;
    if (p.width < 64) p.width = 64;
    if (p.height < 64) p.height = 64;
    if (p.width > FLUX_VAE_MAX_DIM || p.height > FLUX_VAE_MAX_DIM) {
        set_error("Image dimensions exceed maximum (1792x1792)");
        return NULL;
    }

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

    /* Get schedule */
    float *schedule = flux_selected_schedule(&p, image_seq_len);

    /* Sample */
    float *latent = flux_sample_euler(
        ctx->transformer, ctx->qwen3_encoder,
        z, 1, FLUX_LATENT_CHANNELS, latent_h, latent_w,
        text_emb, text_seq,
        schedule, p.num_steps,
        NULL  /* progress_callback */
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
 * Attention Memory Budget
 * ======================================================================== */

/* 4 GB — MPSTemporaryNDArray hard limit. */
#define ATTENTION_MAX_BYTES ((size_t)4ULL << 30)

/* Compute worst-case attention matrix size in bytes.
 * All image dimensions are in pixels (multiples of 16).
 * ref_dims is [h0, w0, h1, w1, ...] in pixels. */
static size_t attention_bytes(int num_heads,
                              int out_h, int out_w,
                              const int *ref_dims, int num_refs,
                              int txt_seq)
{
    size_t total_seq = (size_t)(out_h / 16) * (out_w / 16);
    for (int i = 0; i < num_refs; i++)
        total_seq += (size_t)(ref_dims[i*2] / 16) * (ref_dims[i*2+1] / 16);
    total_seq += txt_seq;
    return (size_t)num_heads * total_seq * total_seq * sizeof(float);
}

/* Shrink reference pixel dimensions so attention fits under 4 GB.
 * ref_dims: [h0, w0, h1, w1, ...] in pixels, modified in-place.
 * Returns 1 if any reference was shrunk, 0 if already fits. */
static int fit_refs_for_attention(int num_heads,
                                  int out_h, int out_w,
                                  int *ref_dims, int num_refs,
                                  int txt_seq)
{
    if (attention_bytes(num_heads, out_h, out_w,
                        ref_dims, num_refs, txt_seq) <= ATTENTION_MAX_BYTES)
        return 0;

    int shrunk = 0;
    for (;;) {
        /* Find reference with the most latent tokens. */
        int best = -1;
        size_t best_tok = 0;
        for (int i = 0; i < num_refs; i++) {
            size_t tok = (size_t)(ref_dims[i*2] / 16) *
                         (ref_dims[i*2+1] / 16);
            if (tok > best_tok) { best_tok = tok; best = i; }
        }
        if (best < 0 || best_tok <= 1) break;  /* can't shrink further */

        /* Scale both dimensions by 0.9, round down to multiple of 16. */
        int h = (int)(ref_dims[best*2] * 0.9f) / 16 * 16;
        int w = (int)(ref_dims[best*2+1] * 0.9f) / 16 * 16;
        if (h < 16) h = 16;
        if (w < 16) w = 16;

        /* No progress — already at minimum. */
        if (h == ref_dims[best*2] && w == ref_dims[best*2+1]) break;

        ref_dims[best*2]   = h;
        ref_dims[best*2+1] = w;
        shrunk = 1;

        if (attention_bytes(num_heads, out_h, out_w,
                            ref_dims, num_refs, txt_seq) <= ATTENTION_MAX_BYTES)
            break;
    }
    return shrunk;
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

    /* Clamp to VAE max dimensions, preserving aspect ratio */
    if (p.width > FLUX_VAE_MAX_DIM || p.height > FLUX_VAE_MAX_DIM) {
        float scale = (float)FLUX_VAE_MAX_DIM /
                      (p.width > p.height ? p.width : p.height);
        p.width = (int)(p.width * scale);
        p.height = (int)(p.height * scale);
    }

    /* Ensure divisible by 16 */
    p.width = (p.width / 16) * 16;
    p.height = (p.height / 16) * 16;

    /* Check attention memory budget — shrink reference if needed. */
    int ref_w = p.width, ref_h = p.height;
    {
        int ref_dims[2] = { p.height, p.width };
        if (fit_refs_for_attention(ctx->num_heads, p.height, p.width,
                                    ref_dims, 1, FLUX_MAX_SEQ_LEN)) {
            fprintf(stderr, "Note: reference image resized from %dx%d to %dx%d "
                    "(GPU attention memory limit)\n",
                    p.width, p.height, ref_dims[1], ref_dims[0]);
            ref_h = ref_dims[0];
            ref_w = ref_dims[1];
        }
    }

    /* Resize input if needed */
    flux_image *resized = NULL;
    const flux_image *img_to_use = input;
    if (input->width != ref_w || input->height != ref_h) {
        resized = flux_image_resize(input, ref_w, ref_h);
        if (!resized) {
            set_error("Failed to resize input image");
            return NULL;
        }
        img_to_use = resized;
    }

    /* Resolve steps and guidance */
    if (p.num_steps <= 0) p.num_steps = ctx->default_steps;
    float guidance = (p.guidance > 0) ? p.guidance : ctx->default_guidance;

    /* Optional overlap: load transformer in parallel while text is encoding. */
    flux_tf_preload_task_t tf_preload;
    flux_transformer_preload_begin(ctx, &tf_preload);

    /* Encode text */
    int text_seq;
    float *text_emb = flux_encode_text(ctx, prompt, &text_seq);
    if (!text_emb) {
        flux_transformer_preload_join(&tf_preload);
        if (resized) flux_image_free(resized);
        set_error("Failed to encode prompt");
        return NULL;
    }

    float *text_emb_uncond = NULL;
    int text_seq_uncond = 0;
    if (!ctx->is_distilled) {
        text_emb_uncond = flux_encode_text(ctx, "", &text_seq_uncond);
        if (!text_emb_uncond) {
            flux_transformer_preload_join(&tf_preload);
            free(text_emb);
            if (resized) flux_image_free(resized);
            set_error("Failed to encode empty prompt for CFG");
            return NULL;
        }
    }

    flux_transformer_preload_join(&tf_preload);

    /* Ensure transformer is ready (with optional low-memory fallback). */
    if (!flux_ensure_transformer_ready(ctx)) {
        free(text_emb);
        free(text_emb_uncond);
        if (resized) flux_image_free(resized);
        return NULL;
    }

    /* Encode image to latent */
    if (flux_phase_callback) flux_phase_callback("encoding reference image", 0);
    float *img_tensor = flux_image_to_tensor(img_to_use);
    if (resized) flux_image_free(resized);

    int latent_h, latent_w;
    float *img_latent = NULL;

    if (ctx->vae) {
        img_latent = flux_vae_encode(ctx->vae, img_tensor, 1,
                                     ref_h, ref_w, &latent_h, &latent_w);
    } else {
        /* Placeholder if no VAE */
        latent_h = ref_h / 16;
        latent_w = ref_w / 16;
        img_latent = (float *)calloc(FLUX_LATENT_CHANNELS * latent_h * latent_w, sizeof(float));
    }

    free(img_tensor);
    if (flux_phase_callback) flux_phase_callback("encoding reference image", 1);

    if (!img_latent) {
        free(text_emb);
        free(text_emb_uncond);
        set_error("Failed to encode image");
        return NULL;
    }

    /*
     * FLUX.2 img2img uses in-context conditioning:
     * - Reference image is encoded to latent with T offset in RoPE (T=10)
     * - Target image starts from pure noise (T=0)
     * - Both are concatenated as tokens in the transformer
     * - Model attends to reference via joint attention
     * - Only target tokens are output
     *
     * This is fundamentally different from traditional img2img that adds
     * noise directly to the encoded image.
     */
    int num_steps = p.num_steps;
    int out_lat_h = p.height / 16;
    int out_lat_w = p.width / 16;
    int image_seq_len = out_lat_h * out_lat_w;  /* For schedule calculation */

    /* Get schedule */
    float *schedule = flux_selected_schedule(&p, image_seq_len);

    /* Initialize target latent with pure noise */
    int64_t seed = (p.seed < 0) ? (int64_t)time(NULL) : p.seed;
    float *z = flux_init_noise(1, FLUX_LATENT_CHANNELS, out_lat_h, out_lat_w, seed);

    /* Reference image latent is img_latent, with T offset = 10 */
    int t_offset = 10;

    /* Sample using in-context conditioning */
    float *latent;
    if (ctx->is_distilled) {
        latent = flux_sample_euler_with_refs(
            ctx->transformer, ctx->qwen3_encoder,
            z, 1, FLUX_LATENT_CHANNELS, out_lat_h, out_lat_w,
            img_latent, latent_h, latent_w,
            t_offset,
            text_emb, text_seq,
            schedule, num_steps,
            NULL
        );
    } else {
        latent = flux_sample_euler_cfg_with_refs(
            ctx->transformer, ctx->qwen3_encoder,
            z, 1, FLUX_LATENT_CHANNELS, out_lat_h, out_lat_w,
            img_latent, latent_h, latent_w,
            t_offset,
            text_emb, text_seq,
            text_emb_uncond, text_seq_uncond,
            guidance,
            schedule, num_steps,
            NULL
        );
    }

    free(z);
    free(img_latent);
    free(schedule);
    free(text_emb);
    free(text_emb_uncond);

    if (!latent) {
        set_error("Sampling failed");
        return NULL;
    }

    /* Decode */
    flux_image *result = NULL;
    if (ctx->vae) {
        if (flux_phase_callback) flux_phase_callback("decoding image", 0);
        result = flux_vae_decode(ctx->vae, latent, 1, out_lat_h, out_lat_w);
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
    if (!ctx || !prompt) {
        set_error("Invalid parameters");
        return NULL;
    }

    /* No references - text-to-image */
    if (!refs || num_refs == 0) {
        return flux_generate(ctx, prompt, params);
    }

    /* Single reference - use optimized path */
    if (num_refs == 1) {
        return flux_img2img(ctx, prompt, refs[0], params);
    }

    flux_params p;
    if (params) {
        p = *params;
    } else {
        p = (flux_params)FLUX_PARAMS_DEFAULT;
    }

    /* Use first reference dimensions if not specified */
    if (p.width <= 0) p.width = refs[0]->width;
    if (p.height <= 0) p.height = refs[0]->height;

    /* Clamp to VAE max dimensions */
    if (p.width > FLUX_VAE_MAX_DIM || p.height > FLUX_VAE_MAX_DIM) {
        float scale = (float)FLUX_VAE_MAX_DIM /
                      (p.width > p.height ? p.width : p.height);
        p.width = (int)(p.width * scale);
        p.height = (int)(p.height * scale);
    }

    p.width = (p.width / 16) * 16;
    p.height = (p.height / 16) * 16;

    /* Resolve steps and guidance */
    if (p.num_steps <= 0) p.num_steps = ctx->default_steps;
    float guidance = (p.guidance > 0) ? p.guidance : ctx->default_guidance;

    /* Optional overlap: load transformer in parallel while text is encoding. */
    flux_tf_preload_task_t tf_preload;
    flux_transformer_preload_begin(ctx, &tf_preload);

    /* Encode text */
    int text_seq;
    float *text_emb = flux_encode_text(ctx, prompt, &text_seq);
    if (!text_emb) {
        flux_transformer_preload_join(&tf_preload);
        set_error("Failed to encode prompt");
        return NULL;
    }

    float *text_emb_uncond = NULL;
    int text_seq_uncond = 0;
    if (!ctx->is_distilled) {
        text_emb_uncond = flux_encode_text(ctx, "", &text_seq_uncond);
        if (!text_emb_uncond) {
            flux_transformer_preload_join(&tf_preload);
            free(text_emb);
            set_error("Failed to encode empty prompt for CFG");
            return NULL;
        }
    }

    flux_transformer_preload_join(&tf_preload);

    /* Ensure transformer is ready (with optional low-memory fallback). */
    if (!flux_ensure_transformer_ready(ctx)) {
        free(text_emb);
        free(text_emb_uncond);
        return NULL;
    }

    /* Build reference pixel dimensions, clamped and rounded to 16. */
    int *ref_pixel_dims = (int *)malloc(num_refs * 2 * sizeof(int));
    for (int i = 0; i < num_refs; i++) {
        int rh = (refs[i]->height / 16) * 16;
        int rw = (refs[i]->width / 16) * 16;
        if (rh > FLUX_VAE_MAX_DIM) rh = FLUX_VAE_MAX_DIM;
        if (rw > FLUX_VAE_MAX_DIM) rw = FLUX_VAE_MAX_DIM;
        if (rh < 16) rh = 16;
        if (rw < 16) rw = 16;
        ref_pixel_dims[i*2]   = rh;
        ref_pixel_dims[i*2+1] = rw;
    }

    /* Shrink references if attention would exceed 4 GB. */
    if (fit_refs_for_attention(ctx->num_heads, p.height, p.width,
                                ref_pixel_dims, num_refs, FLUX_MAX_SEQ_LEN)) {
        fprintf(stderr,
                "Note: reference images resized to fit GPU attention "
                "memory limit\n");
    }

    /* Encode all reference images */
    flux_ref_t *ref_latents = (flux_ref_t *)malloc(num_refs * sizeof(flux_ref_t));
    float **ref_data = (float **)malloc(num_refs * sizeof(float *));
    flux_image **resized_imgs = (flux_image **)calloc(num_refs, sizeof(flux_image *));

    for (int i = 0; i < num_refs; i++) {
        const flux_image *ref = refs[i];
        const flux_image *img_to_use = ref;

        int ref_h = ref_pixel_dims[i*2];
        int ref_w = ref_pixel_dims[i*2+1];

        /* Resize only if dimensions differ from original */
        if (ref->width != ref_w || ref->height != ref_h) {
            resized_imgs[i] = flux_image_resize(ref, ref_w, ref_h);
            if (!resized_imgs[i]) {
                for (int j = 0; j < i; j++) {
                    free(ref_data[j]);
                    if (resized_imgs[j]) flux_image_free(resized_imgs[j]);
                }
                free(ref_latents);
                free(ref_data);
                free(resized_imgs);
                free(ref_pixel_dims);
                free(text_emb);
                free(text_emb_uncond);
                set_error("Failed to resize reference image");
                return NULL;
            }
            img_to_use = resized_imgs[i];
        }

        /* Encode to latent at reference's own size */
        float *tensor = flux_image_to_tensor(img_to_use);
        int lat_h, lat_w;
        ref_data[i] = flux_vae_encode(ctx->vae, tensor, 1,
                                       img_to_use->height, img_to_use->width,
                                       &lat_h, &lat_w);
        free(tensor);

        if (!ref_data[i]) {
            for (int j = 0; j < i; j++) {
                free(ref_data[j]);
                if (resized_imgs[j]) flux_image_free(resized_imgs[j]);
            }
            if (resized_imgs[i]) flux_image_free(resized_imgs[i]);
            free(ref_latents);
            free(ref_data);
            free(resized_imgs);
            free(ref_pixel_dims);
            free(text_emb);
            free(text_emb_uncond);
            set_error("Failed to encode reference image");
            return NULL;
        }

        ref_latents[i].latent = ref_data[i];
        ref_latents[i].h = lat_h;
        ref_latents[i].w = lat_w;
        ref_latents[i].t_offset = 10 * (i + 1);  /* 10, 20, 30, ... */
    }

    /* Free resized images (latents are now encoded) */
    for (int i = 0; i < num_refs; i++) {
        if (resized_imgs[i]) flux_image_free(resized_imgs[i]);
    }
    free(resized_imgs);
    free(ref_pixel_dims);

    int latent_h = p.height / 16;
    int latent_w = p.width / 16;
    int image_seq_len = latent_h * latent_w;

    float *schedule = flux_selected_schedule(&p, image_seq_len);
    int64_t seed = (p.seed < 0) ? (int64_t)time(NULL) : p.seed;
    float *z = flux_init_noise(1, FLUX_LATENT_CHANNELS, latent_h, latent_w, seed);

    /* Sample with multi-reference conditioning */
    float *latent;
    if (ctx->is_distilled) {
        latent = flux_sample_euler_with_multi_refs(
            ctx->transformer, ctx->qwen3_encoder,
            z, 1, FLUX_LATENT_CHANNELS, latent_h, latent_w,
            ref_latents, num_refs,
            text_emb, text_seq,
            schedule, p.num_steps,
            NULL
        );
    } else {
        latent = flux_sample_euler_cfg_with_multi_refs(
            ctx->transformer, ctx->qwen3_encoder,
            z, 1, FLUX_LATENT_CHANNELS, latent_h, latent_w,
            ref_latents, num_refs,
            text_emb, text_seq,
            text_emb_uncond, text_seq_uncond,
            guidance,
            schedule, p.num_steps,
            NULL
        );
    }

    /* Cleanup */
    free(z);
    for (int i = 0; i < num_refs; i++) {
        free(ref_data[i]);
    }
    free(ref_data);
    free(ref_latents);
    free(schedule);
    free(text_emb);
    free(text_emb_uncond);

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
    snprintf(info, sizeof(info), "%s v%s (%s, %d steps, guidance %.1f)",
             ctx->model_name, ctx->model_version,
             ctx->is_distilled ? "distilled" : "base",
             ctx->default_steps, ctx->default_guidance);
    return info;
}

int flux_text_dim(flux_ctx *ctx) {
    return ctx ? ctx->text_dim : 7680;
}

int flux_is_non_commercial(flux_ctx *ctx) {
    return ctx ? ctx->is_non_commercial : 0;
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

/* Debug function: img2img with external inputs from Python */
flux_image *flux_img2img_debug_py(flux_ctx *ctx, const flux_params *params) {
    if (!ctx) {
        set_error("Invalid context");
        return NULL;
    }

    flux_params p;
    if (params) {
        p = *params;
    } else {
        p = (flux_params)FLUX_PARAMS_DEFAULT;
    }

    /* Load Python's noise */
    FILE *f_noise = fopen("/tmp/py_noise.bin", "rb");
    if (!f_noise) {
        set_error("Cannot open /tmp/py_noise.bin");
        return NULL;
    }
    fseek(f_noise, 0, SEEK_END);
    int noise_size = ftell(f_noise) / sizeof(float);
    fseek(f_noise, 0, SEEK_SET);
    float *noise = (float *)malloc(noise_size * sizeof(float));
    fread(noise, sizeof(float), noise_size, f_noise);
    fclose(f_noise);
    fprintf(stderr, "[DEBUG] Loaded noise: %d floats\n", noise_size);

    /* Load Python's ref_latent */
    FILE *f_ref = fopen("/tmp/py_ref_latent.bin", "rb");
    if (!f_ref) {
        free(noise);
        set_error("Cannot open /tmp/py_ref_latent.bin");
        return NULL;
    }
    fseek(f_ref, 0, SEEK_END);
    int ref_size = ftell(f_ref) / sizeof(float);
    fseek(f_ref, 0, SEEK_SET);
    float *ref_latent = (float *)malloc(ref_size * sizeof(float));
    fread(ref_latent, sizeof(float), ref_size, f_ref);
    fclose(f_ref);
    fprintf(stderr, "[DEBUG] Loaded ref_latent: %d floats\n", ref_size);

    /* Load Python's text_emb */
    FILE *f_txt = fopen("/tmp/py_text_emb.bin", "rb");
    if (!f_txt) {
        free(noise);
        free(ref_latent);
        set_error("Cannot open /tmp/py_text_emb.bin");
        return NULL;
    }
    fseek(f_txt, 0, SEEK_END);
    int txt_size = ftell(f_txt) / sizeof(float);
    fseek(f_txt, 0, SEEK_SET);
    float *text_emb = (float *)malloc(txt_size * sizeof(float));
    fread(text_emb, sizeof(float), txt_size, f_txt);
    fclose(f_txt);
    int text_seq = 512;
    fprintf(stderr, "[DEBUG] Loaded text_emb: %d floats (%d x %d)\n",
            txt_size, text_seq, txt_size / text_seq);

    /* Load transformer */
    if (!flux_load_transformer_if_needed(ctx)) {
        free(noise);
        free(ref_latent);
        free(text_emb);
        return NULL;
    }

    /* Dimensions */
    int latent_h = p.height / 16;
    int latent_w = p.width / 16;
    int image_seq_len = latent_h * latent_w;

    /* Get schedule */
    float *schedule = flux_selected_schedule(&p, image_seq_len);

    /* Sample with refs */
    float *latent = flux_sample_euler_with_refs(
        ctx->transformer, NULL,
        noise, 1, FLUX_LATENT_CHANNELS, latent_h, latent_w,
        ref_latent, latent_h, latent_w,
        10,  /* t_offset */
        text_emb, text_seq,
        schedule, p.num_steps,
        NULL  /* progress_callback */
    );

    free(noise);
    free(ref_latent);
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
