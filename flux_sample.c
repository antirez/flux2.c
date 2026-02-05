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
 * Power schedule: denser steps at the start (high noise), sparser at the end.
 * schedule[i] = 1 - (i/n)^alpha
 * alpha=1.0 is linear, alpha=2.0 is quadratic, etc.
 */
float *flux_power_schedule(int num_steps, float alpha) {
    float *schedule = (float *)malloc((num_steps + 1) * sizeof(float));
    for (int i = 0; i <= num_steps; i++) {
        float t = (float)i / (float)num_steps;
        schedule[i] = 1.0f - powf(t, alpha);
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

/* Forward declaration for in-context conditioning (img2img) */
extern float *flux_transformer_forward_with_refs(flux_transformer_t *tf,
                                                 const float *img_latent, int img_h, int img_w,
                                                 const float *ref_latent, int ref_h, int ref_w,
                                                 int t_offset,
                                                 const float *txt_emb, int txt_seq,
                                                 float timestep);

/* Forward declaration for multi-reference conditioning */
typedef struct {
    const float *latent;  /* Reference latent in NCHW format */
    int h, w;             /* Latent dimensions */
    int t_offset;         /* RoPE T coordinate (10, 20, 30, ...) */
} flux_ref_t;

extern float *flux_transformer_forward_with_multi_refs(flux_transformer_t *tf,
                                                       const float *img_latent, int img_h, int img_w,
                                                       const flux_ref_t *refs, int num_refs,
                                                       const float *txt_emb, int txt_seq,
                                                       float timestep);

/* VAE decode for step image callback */
extern flux_image *flux_vae_decode(flux_vae_t *vae, const float *latent,
                                   int batch, int latent_h, int latent_w);
extern void flux_image_free(flux_image *img);

/*
 * Sample using Euler method.
 *
 * z: initial noise [batch, channels, h, w]
 * text_emb: text embeddings [seq_len, hidden]
 * schedule: timestep schedule [num_steps + 1]
 * num_steps: number of denoising steps
 */
float *flux_sample_euler(void *transformer, void *text_encoder,
                         float *z, int batch, int channels, int h, int w,
                         const float *text_emb, int text_seq,
                         const float *schedule, int num_steps,
                         void (*progress_callback)(int step, int total)) {
    (void)text_encoder;  /* Reserved for future use */
    flux_transformer_t *tf = (flux_transformer_t *)transformer;
    int latent_size = batch * channels * h * w;

    /* Working buffers */
    float *z_curr = (float *)malloc(latent_size * sizeof(float));
    float *v_cond = NULL;

    flux_copy(z_curr, z, latent_size);

    /* Reset timing counters */
    flux_reset_timing();
    double total_denoising_start = get_time_ms();
    double step_times[FLUX_MAX_STEPS];

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

        /* Euler step: z_next = z_curr + dt * v */
        flux_axpy(z_curr, dt, v_cond, latent_size);

        free(v_cond);

        step_times[step] = get_time_ms() - step_start;

        if (progress_callback) {
            progress_callback(step + 1, num_steps);
        }

        /* Step image callback - decode and display intermediate result */
        if (flux_step_image_callback && flux_step_image_vae) {
            flux_image *img = flux_vae_decode((flux_vae_t *)flux_step_image_vae,
                                              z_curr, 1, h, w);
            if (img) {
                flux_step_image_callback(step + 1, num_steps, img);
                flux_image_free(img);
            }
        }
    }

    /* Print timing summary */
    if (flux_verbose) {
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
    }

    return z_curr;
}

/*
 * Sample using Euler method with in-context conditioning for img2img.
 *
 * This implements FLUX.2's approach where reference images are passed
 * as additional tokens with a distinct T coordinate in RoPE, allowing
 * the model to attend to them during generation.
 *
 * z: initial noise [batch, channels, h, w] - target image starts as pure noise
 * ref_latent: reference image in latent space [channels, ref_h, ref_w]
 * t_offset: RoPE T coordinate for reference (10 for first ref, 20 for second, etc.)
 */
float *flux_sample_euler_with_refs(void *transformer, void *text_encoder,
                                   float *z, int batch, int channels, int h, int w,
                                   const float *ref_latent, int ref_h, int ref_w,
                                   int t_offset,
                                   const float *text_emb, int text_seq,
                                   const float *schedule, int num_steps,
                                   void (*progress_callback)(int step, int total)) {
    (void)text_encoder;  /* Reserved for future use */
    flux_transformer_t *tf = (flux_transformer_t *)transformer;
    int latent_size = batch * channels * h * w;

    /* Working buffer */
    float *z_curr = (float *)malloc(latent_size * sizeof(float));
    flux_copy(z_curr, z, latent_size);

    /* Reset timing counters */
    flux_reset_timing();
    double total_denoising_start = get_time_ms();
    double step_times[FLUX_MAX_STEPS];

    for (int step = 0; step < num_steps; step++) {
        float t_curr = schedule[step];
        float t_next = schedule[step + 1];
        float dt = t_next - t_curr;

        double step_start = get_time_ms();

        /* Notify step start */
        if (flux_step_callback)
            flux_step_callback(step + 1, num_steps);

        /* Predict velocity with reference image conditioning */
        float *v = flux_transformer_forward_with_refs(tf,
                                                      z_curr, h, w,
                                                      ref_latent, ref_h, ref_w,
                                                      t_offset,
                                                      text_emb, text_seq,
                                                      t_curr);

        /* Euler step: z_next = z_curr + dt * v */
        flux_axpy(z_curr, dt, v, latent_size);

        free(v);

        step_times[step] = get_time_ms() - step_start;

        if (progress_callback) {
            progress_callback(step + 1, num_steps);
        }

        /* Step image callback - decode and display intermediate result */
        if (flux_step_image_callback && flux_step_image_vae) {
            flux_image *img = flux_vae_decode((flux_vae_t *)flux_step_image_vae,
                                              z_curr, 1, h, w);
            if (img) {
                flux_step_image_callback(step + 1, num_steps, img);
                flux_image_free(img);
            }
        }
    }

    /* Print timing summary */
    if (flux_verbose) {
        double total_denoising = get_time_ms() - total_denoising_start;
        fprintf(stderr, "\nDenoising timing breakdown (img2img with refs):\n");
        for (int step = 0; step < num_steps; step++) {
            fprintf(stderr, "  Step %d: %.1f ms\n", step + 1, step_times[step]);
        }
        fprintf(stderr, "  Total denoising: %.1f ms (%.2f s)\n", total_denoising, total_denoising / 1000.0);
    }

    return z_curr;
}

/*
 * Sample using Euler method with multiple reference images.
 * Each reference gets a different T offset in RoPE (10, 20, 30, ...).
 */
float *flux_sample_euler_with_multi_refs(void *transformer, void *text_encoder,
                                         float *z, int batch, int channels, int h, int w,
                                         const flux_ref_t *refs, int num_refs,
                                         const float *text_emb, int text_seq,
                                         const float *schedule, int num_steps,
                                         void (*progress_callback)(int step, int total)) {
    (void)text_encoder;
    flux_transformer_t *tf = (flux_transformer_t *)transformer;
    int latent_size = batch * channels * h * w;

    float *z_curr = (float *)malloc(latent_size * sizeof(float));
    flux_copy(z_curr, z, latent_size);

    flux_reset_timing();
    double total_denoising_start = get_time_ms();
    double step_times[FLUX_MAX_STEPS];

    for (int step = 0; step < num_steps; step++) {
        float t_curr = schedule[step];
        float t_next = schedule[step + 1];
        float dt = t_next - t_curr;

        double step_start = get_time_ms();

        if (flux_step_callback)
            flux_step_callback(step + 1, num_steps);

        /* Predict velocity with multiple reference images */
        float *v = flux_transformer_forward_with_multi_refs(tf,
                                                            z_curr, h, w,
                                                            refs, num_refs,
                                                            text_emb, text_seq,
                                                            t_curr);

        /* Euler step */
        flux_axpy(z_curr, dt, v, latent_size);
        free(v);

        step_times[step] = get_time_ms() - step_start;

        if (progress_callback)
            progress_callback(step + 1, num_steps);

        if (flux_step_image_callback && flux_step_image_vae) {
            flux_image *img = flux_vae_decode((flux_vae_t *)flux_step_image_vae,
                                              z_curr, 1, h, w);
            if (img) {
                flux_step_image_callback(step + 1, num_steps, img);
                flux_image_free(img);
            }
        }
    }

    if (flux_verbose) {
        double total_denoising = get_time_ms() - total_denoising_start;
        fprintf(stderr, "\nDenoising timing breakdown (multi-ref, %d refs):\n", num_refs);
        for (int step = 0; step < num_steps; step++) {
            fprintf(stderr, "  Step %d: %.1f ms\n", step + 1, step_times[step]);
        }
        fprintf(stderr, "  Total denoising: %.1f ms (%.2f s)\n", total_denoising, total_denoising / 1000.0);
    }

    return z_curr;
}

/* ========================================================================
 * CFG (Classifier-Free Guidance) Samplers for Base Model
 *
 * These run the transformer twice per step: once with empty text (uncond)
 * and once with the real prompt (cond), then combine:
 *   v = v_uncond + guidance_scale * (v_cond - v_uncond)
 * ======================================================================== */

/*
 * Euler sampler with CFG for text-to-image.
 */
float *flux_sample_euler_cfg(void *transformer, void *text_encoder,
                              float *z, int batch, int channels, int h, int w,
                              const float *text_emb_cond, int text_seq_cond,
                              const float *text_emb_uncond, int text_seq_uncond,
                              float guidance_scale,
                              const float *schedule, int num_steps,
                              void (*progress_callback)(int step, int total)) {
    (void)text_encoder;
    flux_transformer_t *tf = (flux_transformer_t *)transformer;
    int latent_size = batch * channels * h * w;

    float *z_curr = (float *)malloc(latent_size * sizeof(float));
    flux_copy(z_curr, z, latent_size);

    flux_reset_timing();
    double total_denoising_start = get_time_ms();
    double step_times[FLUX_MAX_STEPS];

    for (int step = 0; step < num_steps; step++) {
        float t_curr = schedule[step];
        float t_next = schedule[step + 1];
        float dt = t_next - t_curr;

        double step_start = get_time_ms();

        if (flux_step_callback)
            flux_step_callback(step + 1, num_steps);

        /* Unconditioned prediction */
        float *v_uncond = flux_transformer_forward(tf, z_curr, h, w,
                                                    text_emb_uncond, text_seq_uncond,
                                                    t_curr);

        /* Conditioned prediction */
        float *v_cond = flux_transformer_forward(tf, z_curr, h, w,
                                                  text_emb_cond, text_seq_cond,
                                                  t_curr);

        /* CFG combine: v = v_uncond + scale * (v_cond - v_uncond) */
        for (int i = 0; i < latent_size; i++) {
            float v = v_uncond[i] + guidance_scale * (v_cond[i] - v_uncond[i]);
            z_curr[i] += dt * v;
        }

        free(v_uncond);
        free(v_cond);

        step_times[step] = get_time_ms() - step_start;

        if (progress_callback)
            progress_callback(step + 1, num_steps);

        if (flux_step_image_callback && flux_step_image_vae) {
            flux_image *img = flux_vae_decode((flux_vae_t *)flux_step_image_vae,
                                              z_curr, 1, h, w);
            if (img) {
                flux_step_image_callback(step + 1, num_steps, img);
                flux_image_free(img);
            }
        }
    }

    if (flux_verbose) {
        double total_denoising = get_time_ms() - total_denoising_start;
        fprintf(stderr, "\nDenoising timing breakdown (CFG, guidance=%.1f):\n", guidance_scale);
        for (int step = 0; step < num_steps; step++) {
            fprintf(stderr, "  Step %d: %.1f ms\n", step + 1, step_times[step]);
        }
        fprintf(stderr, "  Total denoising: %.1f ms (%.2f s)\n", total_denoising, total_denoising / 1000.0);
    }

    return z_curr;
}

/*
 * Euler sampler with CFG and single reference image (img2img).
 */
float *flux_sample_euler_cfg_with_refs(void *transformer, void *text_encoder,
                                        float *z, int batch, int channels, int h, int w,
                                        const float *ref_latent, int ref_h, int ref_w,
                                        int t_offset,
                                        const float *text_emb_cond, int text_seq_cond,
                                        const float *text_emb_uncond, int text_seq_uncond,
                                        float guidance_scale,
                                        const float *schedule, int num_steps,
                                        void (*progress_callback)(int step, int total)) {
    (void)text_encoder;
    flux_transformer_t *tf = (flux_transformer_t *)transformer;
    int latent_size = batch * channels * h * w;

    float *z_curr = (float *)malloc(latent_size * sizeof(float));
    flux_copy(z_curr, z, latent_size);

    flux_reset_timing();
    double total_denoising_start = get_time_ms();
    double step_times[FLUX_MAX_STEPS];

    for (int step = 0; step < num_steps; step++) {
        float t_curr = schedule[step];
        float t_next = schedule[step + 1];
        float dt = t_next - t_curr;

        double step_start = get_time_ms();

        if (flux_step_callback)
            flux_step_callback(step + 1, num_steps);

        /* Unconditioned prediction (with ref) */
        float *v_uncond = flux_transformer_forward_with_refs(tf,
                              z_curr, h, w,
                              ref_latent, ref_h, ref_w, t_offset,
                              text_emb_uncond, text_seq_uncond, t_curr);

        /* Conditioned prediction (with ref) */
        float *v_cond = flux_transformer_forward_with_refs(tf,
                            z_curr, h, w,
                            ref_latent, ref_h, ref_w, t_offset,
                            text_emb_cond, text_seq_cond, t_curr);

        /* CFG combine */
        for (int i = 0; i < latent_size; i++) {
            float v = v_uncond[i] + guidance_scale * (v_cond[i] - v_uncond[i]);
            z_curr[i] += dt * v;
        }

        free(v_uncond);
        free(v_cond);

        step_times[step] = get_time_ms() - step_start;

        if (progress_callback)
            progress_callback(step + 1, num_steps);

        if (flux_step_image_callback && flux_step_image_vae) {
            flux_image *img = flux_vae_decode((flux_vae_t *)flux_step_image_vae,
                                              z_curr, 1, h, w);
            if (img) {
                flux_step_image_callback(step + 1, num_steps, img);
                flux_image_free(img);
            }
        }
    }

    if (flux_verbose) {
        double total_denoising = get_time_ms() - total_denoising_start;
        fprintf(stderr, "\nDenoising timing breakdown (CFG img2img, guidance=%.1f):\n", guidance_scale);
        for (int step = 0; step < num_steps; step++) {
            fprintf(stderr, "  Step %d: %.1f ms\n", step + 1, step_times[step]);
        }
        fprintf(stderr, "  Total denoising: %.1f ms (%.2f s)\n", total_denoising, total_denoising / 1000.0);
    }

    return z_curr;
}

/*
 * Euler sampler with CFG and multiple reference images.
 */
float *flux_sample_euler_cfg_with_multi_refs(void *transformer, void *text_encoder,
                                              float *z, int batch, int channels, int h, int w,
                                              const flux_ref_t *refs, int num_refs,
                                              const float *text_emb_cond, int text_seq_cond,
                                              const float *text_emb_uncond, int text_seq_uncond,
                                              float guidance_scale,
                                              const float *schedule, int num_steps,
                                              void (*progress_callback)(int step, int total)) {
    (void)text_encoder;
    flux_transformer_t *tf = (flux_transformer_t *)transformer;
    int latent_size = batch * channels * h * w;

    float *z_curr = (float *)malloc(latent_size * sizeof(float));
    flux_copy(z_curr, z, latent_size);

    flux_reset_timing();
    double total_denoising_start = get_time_ms();
    double step_times[FLUX_MAX_STEPS];

    for (int step = 0; step < num_steps; step++) {
        float t_curr = schedule[step];
        float t_next = schedule[step + 1];
        float dt = t_next - t_curr;

        double step_start = get_time_ms();

        if (flux_step_callback)
            flux_step_callback(step + 1, num_steps);

        /* Unconditioned prediction (with refs) */
        float *v_uncond = flux_transformer_forward_with_multi_refs(tf,
                              z_curr, h, w,
                              refs, num_refs,
                              text_emb_uncond, text_seq_uncond, t_curr);

        /* Conditioned prediction (with refs) */
        float *v_cond = flux_transformer_forward_with_multi_refs(tf,
                            z_curr, h, w,
                            refs, num_refs,
                            text_emb_cond, text_seq_cond, t_curr);

        /* CFG combine */
        for (int i = 0; i < latent_size; i++) {
            float v = v_uncond[i] + guidance_scale * (v_cond[i] - v_uncond[i]);
            z_curr[i] += dt * v;
        }

        free(v_uncond);
        free(v_cond);

        step_times[step] = get_time_ms() - step_start;

        if (progress_callback)
            progress_callback(step + 1, num_steps);

        if (flux_step_image_callback && flux_step_image_vae) {
            flux_image *img = flux_vae_decode((flux_vae_t *)flux_step_image_vae,
                                              z_curr, 1, h, w);
            if (img) {
                flux_step_image_callback(step + 1, num_steps, img);
                flux_image_free(img);
            }
        }
    }

    if (flux_verbose) {
        double total_denoising = get_time_ms() - total_denoising_start;
        fprintf(stderr, "\nDenoising timing breakdown (CFG multi-ref, %d refs, guidance=%.1f):\n",
                num_refs, guidance_scale);
        for (int step = 0; step < num_steps; step++) {
            fprintf(stderr, "  Step %d: %.1f ms\n", step + 1, step_times[step]);
        }
        fprintf(stderr, "  Total denoising: %.1f ms (%.2f s)\n", total_denoising, total_denoising / 1000.0);
    }

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
                                   float eta,
                                   void (*progress_callback)(int step, int total)) {
    flux_transformer_t *tf = (flux_transformer_t *)transformer;
    int latent_size = batch * channels * h * w;

    float *z_curr = (float *)malloc(latent_size * sizeof(float));
    float *noise = (float *)malloc(latent_size * sizeof(float));

    flux_copy(z_curr, z, latent_size);

    for (int step = 0; step < num_steps; step++) {
        float t_curr = schedule[step];
        float t_next = schedule[step + 1];
        float dt = t_next - t_curr;

        /* Predict velocity */
        float *v = flux_transformer_forward(tf, z_curr, h, w,
                                            text_emb, text_seq, t_curr);

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

        /* Step image callback - decode and display intermediate result */
        if (flux_step_image_callback && flux_step_image_vae) {
            flux_image *img = flux_vae_decode((flux_vae_t *)flux_step_image_vae,
                                              z_curr, 1, h, w);
            if (img) {
                flux_step_image_callback(step + 1, num_steps, img);
                flux_image_free(img);
            }
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
                        void (*progress_callback)(int step, int total)) {
    flux_transformer_t *tf = (flux_transformer_t *)transformer;
    int latent_size = batch * channels * h * w;

    float *z_curr = (float *)malloc(latent_size * sizeof(float));
    float *z_pred = (float *)malloc(latent_size * sizeof(float));

    flux_copy(z_curr, z, latent_size);

    for (int step = 0; step < num_steps; step++) {
        float t_curr = schedule[step];
        float t_next = schedule[step + 1];
        float dt = t_next - t_curr;

        /* First velocity estimate */
        float *v1 = flux_transformer_forward(tf, z_curr, h, w,
                                             text_emb, text_seq, t_curr);

        /* Predict next state */
        flux_copy(z_pred, z_curr, latent_size);
        flux_axpy(z_pred, dt, v1, latent_size);

        /* Second velocity estimate (only if not last step) */
        if (step < num_steps - 1) {
            float *v2 = flux_transformer_forward(tf, z_pred, h, w,
                                                 text_emb, text_seq, t_next);

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

        /* Step image callback - decode and display intermediate result */
        if (flux_step_image_callback && flux_step_image_vae) {
            flux_image *img = flux_vae_decode((flux_vae_t *)flux_step_image_vae,
                                              z_curr, 1, h, w);
            if (img) {
                flux_step_image_callback(step + 1, num_steps, img);
                flux_image_free(img);
            }
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
 *
 * Size-independent noise: We generate noise at max latent size (112x112)
 * and subsample to target size. This ensures the same seed produces
 * similar compositions at different resolutions.
 */
#define NOISE_MAX_LATENT_DIM 112  /* 1792/16 = 112 */

float *flux_init_noise(int batch, int channels, int h, int w, int64_t seed) {
    int target_size = batch * channels * h * w;
    float *noise = (float *)malloc(target_size * sizeof(float));

    if (seed >= 0) {
        flux_rng_seed((uint64_t)seed);
    }

    /* If target is max size or larger, just generate directly */
    if (h >= NOISE_MAX_LATENT_DIM && w >= NOISE_MAX_LATENT_DIM) {
        flux_randn(noise, target_size);
        return noise;
    }

    /* Generate noise at max latent size */
    int max_h = NOISE_MAX_LATENT_DIM;
    int max_w = NOISE_MAX_LATENT_DIM;
    int max_size = batch * channels * max_h * max_w;
    float *max_noise = (float *)malloc(max_size * sizeof(float));
    flux_randn(max_noise, max_size);

    /* Subsample to target size using nearest-neighbor */
    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channels; c++) {
            for (int ty = 0; ty < h; ty++) {
                for (int tx = 0; tx < w; tx++) {
                    /* Map target position to source position */
                    int sy = ty * max_h / h;
                    int sx = tx * max_w / w;

                    int src_idx = ((b * channels + c) * max_h + sy) * max_w + sx;
                    int dst_idx = ((b * channels + c) * h + ty) * w + tx;
                    noise[dst_idx] = max_noise[src_idx];
                }
            }
        }
    }

    free(max_noise);
    return noise;
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
                            int num_steps,
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

    /* Sample (FLUX.2-klein is guidance-distilled, no CFG needed) */
    float *latent = flux_sample_euler(ctx_ptr, NULL,
                                      z, 1, channels, latent_h, latent_w,
                                      text_emb, text_seq,
                                      schedule, num_steps,
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
