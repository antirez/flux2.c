/*
 * FLUX Sampling Implementation
 *
 * Rectified Flow sampling for image generation.
 * Uses Euler method for ODE integration.
 */

#include "flux.h"
#include "flux_kernels.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

#ifdef USE_METAL
#include "flux_metal.h"
#endif

/* Timing utilities for performance analysis - use wall-clock time */
static double get_time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

/* Cumulative timing for denoising breakdown */
double flux_timing_transformer_total = 0.0;
double flux_timing_transformer_double = 0.0;
double flux_timing_transformer_single = 0.0;
double flux_timing_transformer_final = 0.0;

void flux_reset_timing(void) {
    flux_timing_transformer_total = 0.0;
    flux_timing_transformer_double = 0.0;
    flux_timing_transformer_single = 0.0;
    flux_timing_transformer_final = 0.0;
}

/* ========================================================================
 * Timestep Schedules
 * ======================================================================== */

/*
 * Linear timestep schedule from 1.0 to 0.0
 * Returns array of num_steps+1 values: [1.0, ..., 0.0]
 */
float *flux_linear_schedule(int num_steps) {
    float *schedule = (float *)malloc((num_steps + 1) * sizeof(float));
    for (int i = 0; i <= num_steps; i++) {
        schedule[i] = 1.0f - (float)i / (float)num_steps;
    }
    return schedule;
}

/*
 * Shifted sigmoid schedule (better for flow matching)
 * shift controls where the inflection point is
 */
float *flux_sigmoid_schedule(int num_steps, float shift) {
    float *schedule = (float *)malloc((num_steps + 1) * sizeof(float));

    for (int i = 0; i <= num_steps; i++) {
        float t = (float)i / (float)num_steps;
        /* Shifted sigmoid: more steps at the end */
        float x = (t - 0.5f) * 10.0f + shift;
        schedule[i] = 1.0f - 1.0f / (1.0f + expf(-x));
    }

    /* Ensure endpoints */
    schedule[0] = 1.0f;
    schedule[num_steps] = 0.0f;

    return schedule;
}

/*
 * Resolution-dependent schedule (as used in FLUX.2)
 * Higher resolutions use more steps at the start
 */
float *flux_resolution_schedule(int num_steps, int height, int width) {
    float *schedule = (float *)malloc((num_steps + 1) * sizeof(float));

    /* Compute shift based on resolution */
    int pixels = height * width;
    float shift = 0.0f;
    if (pixels >= 1024 * 1024) {
        shift = 1.0f;  /* High res: more early steps */
    } else if (pixels >= 512 * 512) {
        shift = 0.5f;
    }

    for (int i = 0; i <= num_steps; i++) {
        float t = (float)i / (float)num_steps;
        /* Apply shift */
        t = powf(t, 1.0f + shift * 0.5f);
        schedule[i] = 1.0f - t;
    }

    return schedule;
}

/*
 * FLUX.2 official schedule with empirical mu calculation
 * Matches Python's get_schedule() function from official flux2 code
 */
static float compute_empirical_mu(int image_seq_len, int num_steps) {
    const float a1 = 8.73809524e-05f, b1 = 1.89833333f;
    const float a2 = 0.00016927f, b2 = 0.45666666f;

    if (image_seq_len > 4300) {
        return a2 * image_seq_len + b2;
    }

    float m_200 = a2 * image_seq_len + b2;
    float m_10 = a1 * image_seq_len + b1;

    float a = (m_200 - m_10) / 190.0f;
    float b = m_200 - 200.0f * a;
    return a * num_steps + b;
}

static float generalized_time_snr_shift(float t, float mu, float sigma) {
    /* t / (1 - t) with exp(mu) shift */
    if (t <= 0.0f) return 0.0f;
    if (t >= 1.0f) return 1.0f;
    return expf(mu) / (expf(mu) + powf(1.0f / t - 1.0f, sigma));
}

float *flux_official_schedule(int num_steps, int image_seq_len) {
    float *schedule = (float *)malloc((num_steps + 1) * sizeof(float));
    float mu = compute_empirical_mu(image_seq_len, num_steps);

    for (int i = 0; i <= num_steps; i++) {
        float t = 1.0f - (float)i / (float)num_steps;  /* Linear from 1 to 0 */
        schedule[i] = generalized_time_snr_shift(t, mu, 1.0f);
    }

    return schedule;
}

/* ========================================================================
 * Euler Sampler for Rectified Flow
 * ======================================================================== */

/*
 * Single Euler step:
 * z_next = z_t + (t_next - t_curr) * v(z_t, t_curr)
 *
 * Where v is the velocity predicted by the model.
 * In rectified flow: v = (z_data - z_noise) at timestep t
 */

typedef struct flux_transformer flux_transformer_t;
typedef struct flux_vae flux_vae_t;

/* Forward declarations */
extern float *flux_transformer_forward(flux_transformer_t *tf,
                                       const float *img_latent, int img_h, int img_w,
                                       const float *txt_emb, int txt_seq,
                                       float timestep);

/*
 * Sample using Euler method.
 *
 * z: initial noise [batch, channels, h, w]
 * text_emb: text embeddings [seq_len, hidden]
 * schedule: timestep schedule [num_steps + 1]
 * num_steps: number of denoising steps
 * guidance_scale: classifier-free guidance scale (1.0 = no guidance)
 */
float *flux_sample_euler(void *transformer, void *text_encoder,
                         float *z, int batch, int channels, int h, int w,
                         const float *text_emb, int text_seq,
                         const float *null_emb,  /* For CFG */
                         const float *schedule, int num_steps,
                         float guidance_scale,
                         void (*progress_callback)(int step, int total)) {
    (void)text_encoder;  /* Reserved for future use */
    flux_transformer_t *tf = (flux_transformer_t *)transformer;
    int latent_size = batch * channels * h * w;

    /* Working buffers */
    float *z_curr = (float *)malloc(latent_size * sizeof(float));
    float *v_cond = NULL;
    float *v_uncond = NULL;

    flux_copy(z_curr, z, latent_size);

#ifdef USE_METAL
    flux_metal_clear_error();
#endif

    int use_cfg = (guidance_scale > 1.0f && null_emb != NULL);

    /* Reset timing counters */
    flux_reset_timing();
    double total_denoising_start = get_time_ms();
    double step_times[64];  /* Max 64 steps */

    for (int step = 0; step < num_steps; step++) {
        float t_curr = schedule[step];
        float t_next = schedule[step + 1];
        float dt = t_next - t_curr;  /* Negative for denoising */

        double step_start = get_time_ms();

        /* Notify step start */
        if (flux_step_callback)
            flux_step_callback(step + 1, num_steps);

        /* Predict velocity with conditioning */
        v_cond = flux_transformer_forward(tf, z_curr, h, w,
                                          text_emb, text_seq, t_curr);
        if (!v_cond) {
            free(z_curr);
            return NULL;
        }

        if (use_cfg) {
            /* Predict unconditional velocity */
            v_uncond = flux_transformer_forward(tf, z_curr, h, w,
                                                null_emb, text_seq, t_curr);
            if (!v_uncond) {
                free(v_cond);
                free(z_curr);
                return NULL;
            }

            /* CFG: v = v_uncond + scale * (v_cond - v_uncond) */
            for (int i = 0; i < latent_size; i++) {
                v_cond[i] = v_uncond[i] + guidance_scale * (v_cond[i] - v_uncond[i]);
            }

            free(v_uncond);
        }

        /* Euler step: z_next = z_curr + dt * v */
        flux_axpy(z_curr, dt, v_cond, latent_size);

        free(v_cond);

        step_times[step] = get_time_ms() - step_start;

        if (progress_callback) {
            progress_callback(step + 1, num_steps);
        }
    }

    /* Print timing summary */
    double total_denoising = get_time_ms() - total_denoising_start;
    fprintf(stderr, "\nDenoising timing breakdown:\n");
    for (int step = 0; step < num_steps; step++) {
        fprintf(stderr, "  Step %d: %.1f ms\n", step + 1, step_times[step]);
    }
    fprintf(stderr, "  Total denoising: %.1f ms (%.2f s)\n", total_denoising, total_denoising / 1000.0);
    fprintf(stderr, "  Transformer breakdown:\n");
    fprintf(stderr, "    Double blocks: %.1f ms (%.1f%%)\n",
            flux_timing_transformer_double, 100.0 * flux_timing_transformer_double / flux_timing_transformer_total);
    fprintf(stderr, "    Single blocks: %.1f ms (%.1f%%)\n",
            flux_timing_transformer_single, 100.0 * flux_timing_transformer_single / flux_timing_transformer_total);
    fprintf(stderr, "    Final layer:   %.1f ms (%.1f%%)\n",
            flux_timing_transformer_final, 100.0 * flux_timing_transformer_final / flux_timing_transformer_total);
    fprintf(stderr, "    Total:         %.1f ms\n", flux_timing_transformer_total);

    return z_curr;
}

/*
 * Sample using Euler method with stochastic noise injection.
 * This can help with diversity and quality.
 */
float *flux_sample_euler_ancestral(void *transformer,
                                   float *z, int batch, int channels, int h, int w,
                                   const float *text_emb, int text_seq,
                                   const float *schedule, int num_steps,
                                   float guidance_scale, float eta,
                                   void (*progress_callback)(int step, int total)) {
    (void)guidance_scale;  /* Reserved for CFG support */
    flux_transformer_t *tf = (flux_transformer_t *)transformer;
    int latent_size = batch * channels * h * w;

    float *z_curr = (float *)malloc(latent_size * sizeof(float));
    float *noise = (float *)malloc(latent_size * sizeof(float));

    flux_copy(z_curr, z, latent_size);

#ifdef USE_METAL
    flux_metal_clear_error();
#endif

    for (int step = 0; step < num_steps; step++) {
        float t_curr = schedule[step];
        float t_next = schedule[step + 1];
        float dt = t_next - t_curr;

        /* Predict velocity */
        float *v = flux_transformer_forward(tf, z_curr, h, w,
                                            text_emb, text_seq, t_curr);
        if (!v) {
            free(noise);
            free(z_curr);
            return NULL;
        }

        /* Euler step */
        flux_axpy(z_curr, dt, v, latent_size);

        /* Add noise (ancestral sampling) */
        if (eta > 0 && step < num_steps - 1) {
            float sigma = eta * sqrtf(fabsf(dt));
            flux_randn(noise, latent_size);
            flux_axpy(z_curr, sigma, noise, latent_size);
        }

        free(v);

        if (progress_callback) {
            progress_callback(step + 1, num_steps);
        }
    }

    free(noise);
    return z_curr;
}

/* ========================================================================
 * Heun Sampler (2nd order)
 * ======================================================================== */

/*
 * Heun's method (improved Euler):
 * 1. Predict: z_pred = z_t + dt * v(z_t, t)
 * 2. Correct: z_next = z_t + dt/2 * (v(z_t, t) + v(z_pred, t+dt))
 */
float *flux_sample_heun(void *transformer,
                        float *z, int batch, int channels, int h, int w,
                        const float *text_emb, int text_seq,
                        const float *schedule, int num_steps,
                        float guidance_scale,
                        void (*progress_callback)(int step, int total)) {
    (void)guidance_scale;  /* Reserved for CFG support */
    flux_transformer_t *tf = (flux_transformer_t *)transformer;
    int latent_size = batch * channels * h * w;

    float *z_curr = (float *)malloc(latent_size * sizeof(float));
    float *z_pred = (float *)malloc(latent_size * sizeof(float));

    flux_copy(z_curr, z, latent_size);

#ifdef USE_METAL
    flux_metal_clear_error();
#endif

    for (int step = 0; step < num_steps; step++) {
        float t_curr = schedule[step];
        float t_next = schedule[step + 1];
        float dt = t_next - t_curr;

        /* First velocity estimate */
        float *v1 = flux_transformer_forward(tf, z_curr, h, w,
                                             text_emb, text_seq, t_curr);
        if (!v1) {
            free(z_pred);
            free(z_curr);
            return NULL;
        }

        /* Predict next state */
        flux_copy(z_pred, z_curr, latent_size);
        flux_axpy(z_pred, dt, v1, latent_size);

        /* Second velocity estimate (only if not last step) */
        if (step < num_steps - 1) {
            float *v2 = flux_transformer_forward(tf, z_pred, h, w,
                                                 text_emb, text_seq, t_next);
            if (!v2) {
                free(v1);
                free(z_pred);
                free(z_curr);
                return NULL;
            }

            /* Heun correction: z_next = z_curr + dt/2 * (v1 + v2) */
            for (int i = 0; i < latent_size; i++) {
                z_curr[i] += 0.5f * dt * (v1[i] + v2[i]);
            }

            free(v2);
        } else {
            /* Last step: just use Euler */
            flux_axpy(z_curr, dt, v1, latent_size);
        }

        free(v1);

        if (progress_callback) {
            progress_callback(step + 1, num_steps);
        }
    }

    free(z_pred);
    return z_curr;
}

/* ========================================================================
 * Latent Noise Initialization
 * ======================================================================== */

/*
 * Initialize latent noise for generation.
 * For rectified flow, we start from pure noise (t=1).
 */
float *flux_init_noise(int batch, int channels, int h, int w, int64_t seed) {
    int size = batch * channels * h * w;
    float *noise = (float *)malloc(size * sizeof(float));

    if (seed >= 0) {
        flux_rng_seed((uint64_t)seed);
    }

    flux_randn(noise, size);
    return noise;
}

/*
 * Initialize latent for img2img.
 * Blend between encoded image and noise based on strength.
 */
float *flux_init_img2img(const float *img_latent, float strength,
                         int batch, int channels, int h, int w, int64_t seed) {
    int size = batch * channels * h * w;
    float *latent = (float *)malloc(size * sizeof(float));

    if (seed >= 0) {
        flux_rng_seed((uint64_t)seed);
    }

    /* z = (1 - strength) * img_latent + strength * noise */
    float noise_scale = strength;
    float img_scale = 1.0f - strength;

    for (int i = 0; i < size; i++) {
        float noise = flux_random_normal();
        latent[i] = img_scale * img_latent[i] + noise_scale * noise;
    }

    return latent;
}

/* ========================================================================
 * Full Generation Pipeline
 * ======================================================================== */

/*
 * Complete text-to-image generation pipeline.
 */
typedef struct flux_ctx flux_ctx;

/* Forward declaration */
extern flux_ctx *flux_get_ctx(void);

float *flux_generate_latent(void *ctx_ptr,
                            const float *text_emb, int text_seq,
                            int height, int width,
                            int num_steps, float guidance_scale,
                            int64_t seed,
                            void (*progress_callback)(int step, int total)) {
    /* Compute latent dimensions */
    int latent_h = height / 16;
    int latent_w = width / 16;
    int channels = FLUX_LATENT_CHANNELS;

    /* Initialize noise */
    float *z = flux_init_noise(1, channels, latent_h, latent_w, seed);

    /* Get schedule (4 steps for klein distilled) */
    float *schedule = flux_linear_schedule(num_steps);

    /* Sample */
    /* Note: For klein, guidance_scale should be 1.0 (guidance-distilled) */
    float *latent = flux_sample_euler(ctx_ptr, NULL,
                                      z, 1, channels, latent_h, latent_w,
                                      text_emb, text_seq,
                                      NULL,  /* No null embedding for klein */
                                      schedule, num_steps,
                                      guidance_scale,
                                      progress_callback);

    free(z);
    free(schedule);

    return latent;
}

/* ========================================================================
 * Legacy Progress Callback (for backwards compatibility)
 * ======================================================================== */

/* Legacy callback for step-level progress (called from sampling loop) */
void (*flux_progress_callback)(int, int) = NULL;

void flux_set_progress_callback(void (*callback)(int, int)) {
    flux_progress_callback = callback;
}
