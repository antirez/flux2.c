/*
 * FLUX CLI Application
 *
 * Command-line interface for FLUX.2 klein 4B image generation.
 *
 * Usage:
 *   flux -d model/ -p "prompt" -o output.png [options]
 *
 * Options:
 *   -d, --dir PATH        Path to model directory (safetensors)
 *   -p, --prompt TEXT     Text prompt for generation
 *   -o, --output PATH     Output image path
 *   -W, --width N         Output width (default: 256)
 *   -H, --height N        Output height (default: 256)
 *   -s, --steps N         Number of sampling steps (default: 4)
 *   -g, --guidance N      Guidance scale (default: 1.0)
 *   -S, --seed N          Random seed (-1 for random)
 *   -i, --input PATH      Input image for img2img
 *   -t, --strength N      Img2img strength (0.0-1.0)
 *   -q, --quiet           No output, just generate
 *   -v, --verbose         Extra detailed output
 *   -h, --help            Show help
 */

#include "flux.h"
#include "flux_kernels.h"
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef USE_METAL
#include "flux_metal.h"
#endif

/* ========================================================================
 * Verbosity Levels
 * ======================================================================== */

typedef enum {
  OUTPUT_QUIET = 0,  /* No output */
  OUTPUT_NORMAL = 1, /* Progress and essential info */
  OUTPUT_VERBOSE = 2 /* Detailed debugging info */
} output_level_t;

static output_level_t output_level = OUTPUT_NORMAL;

/* ========================================================================
 * CLI Progress Callbacks
 * ======================================================================== */

static int cli_current_step = 0;
static int cli_legend_printed = 0;

/* Called at the start of each sampling step */
static void cli_step_callback(int step, int total) {
  if (output_level == OUTPUT_QUIET)
    return;

  /* Print legend before first step */
  if (!cli_legend_printed) {
    fprintf(stderr, "Denoising (d=double block, s=single blocks, F=final):\n");
    cli_legend_printed = 1;
  }

  /* Print newline to end previous step's progress (if any) */
  if (cli_current_step > 0) {
    fprintf(stderr, "\n");
  }
  cli_current_step = step;
  fprintf(stderr, "  Step %d/%d ", step, total);
  fflush(stderr);
}

/* Called for each substep within transformer forward */
static void cli_substep_callback(flux_substep_type_t type, int index,
                                 int total) {
  if (output_level == OUTPUT_QUIET)
    return;
  (void)total;

  switch (type) {
  case FLUX_SUBSTEP_DOUBLE_BLOCK:
    fputc('d', stderr);
    break;
  case FLUX_SUBSTEP_SINGLE_BLOCK:
    /* Print 's' every 5 single blocks to avoid too much output */
    if ((index + 1) % 5 == 0) {
      fputc('s', stderr);
    }
    break;
  case FLUX_SUBSTEP_FINAL_LAYER:
    fputc('F', stderr);
    break;
  }
  fflush(stderr);
}

/* Track phase timing */
static clock_t cli_phase_start;
static const char *cli_current_phase = NULL;

/* Called at phase boundaries (encoding text, decoding image, etc.) */
static void cli_phase_callback(const char *phase, int done) {
  if (output_level == OUTPUT_QUIET)
    return;

  if (!done) {
    /* If we were showing step progress, end that line first */
    if (cli_current_step > 0) {
      fprintf(stderr, "\n");
      cli_current_step = 0;
    }

    /* Phase starting */
    cli_current_phase = phase;
    cli_phase_start = clock();

    /* Capitalize first letter for display */
    char display[64];
    strncpy(display, phase, sizeof(display) - 1);
    display[sizeof(display) - 1] = '\0';
    if (display[0] >= 'a' && display[0] <= 'z') {
      display[0] -= 32;
    }

    fprintf(stderr, "%s...", display);
    fflush(stderr);
  } else {
    /* Phase finished */
    double elapsed = (double)(clock() - cli_phase_start) / CLOCKS_PER_SEC;
    fprintf(stderr, " done (%.1fs)\n", elapsed);
    cli_current_phase = NULL;
  }
}

/* Set up CLI progress callbacks */
static void cli_setup_progress(void) {
  cli_current_step = 0;
  cli_legend_printed = 0;
  cli_current_phase = NULL;
  flux_step_callback = cli_step_callback;
  flux_substep_callback = cli_substep_callback;
  flux_phase_callback = cli_phase_callback;
}

/* Clean up after generation (print final newline) */
static void cli_finish_progress(void) {
  if (cli_current_step > 0) {
    fprintf(stderr, "\n");
    cli_current_step = 0;
  }
  flux_step_callback = NULL;
  flux_substep_callback = NULL;
  flux_phase_callback = NULL;
}

/* ========================================================================
 * Timing Helper
 * ======================================================================== */

static clock_t timer_start;

static void timer_begin(void) { timer_start = clock(); }

static double timer_end(void) {
  return (double)(clock() - timer_start) / CLOCKS_PER_SEC;
}

/* ========================================================================
 * Output Helpers
 * ======================================================================== */

/* Print if not quiet */
#define LOG_NORMAL(...)                                                        \
  do {                                                                         \
    if (output_level >= OUTPUT_NORMAL)                                         \
      fprintf(stderr, __VA_ARGS__);                                            \
  } while (0)

/* Print only in verbose mode */
#define LOG_VERBOSE(...)                                                       \
  do {                                                                         \
    if (output_level >= OUTPUT_VERBOSE)                                        \
      fprintf(stderr, __VA_ARGS__);                                            \
  } while (0)

/* ========================================================================
 * Usage and Help
 * ======================================================================== */

/* Default values */
#define DEFAULT_WIDTH 256
#define DEFAULT_HEIGHT 256
#define DEFAULT_STEPS 4
#define DEFAULT_GUIDANCE 1.0f
#define DEFAULT_STRENGTH 0.75f

static void print_usage(const char *prog) {
  fprintf(stderr, "FLUX.2 klein 4B - Pure C Image Generation\n\n");
  fprintf(stderr, "Usage: %s [options]\n\n", prog);
  fprintf(stderr, "Required:\n");
  fprintf(stderr, "  -d, --dir PATH        Path to model directory\n");
  fprintf(stderr, "  -p, --prompt TEXT     Text prompt for generation\n");
  fprintf(stderr, "  -o, --output PATH     Output image path (.png, .ppm)\n\n");
  fprintf(stderr, "Generation options:\n");
  fprintf(stderr, "  -W, --width N         Output width (default: %d)\n",
          DEFAULT_WIDTH);
  fprintf(stderr, "  -H, --height N        Output height (default: %d)\n",
          DEFAULT_HEIGHT);
  fprintf(stderr, "  -s, --steps N         Sampling steps (default: %d)\n",
          DEFAULT_STEPS);
  fprintf(stderr, "  -g, --guidance N      Guidance scale (default: %.1f)\n",
          DEFAULT_GUIDANCE);
  fprintf(stderr, "  -S, --seed N          Random seed (-1 for random)\n\n");
  fprintf(stderr, "Image-to-image options:\n");
  fprintf(stderr, "  -i, --input PATH      Input image for img2img\n");
  fprintf(stderr,
          "  -t, --strength N      Strength 0.0-1.0 (default: %.2f)\n\n",
          DEFAULT_STRENGTH);
  fprintf(stderr, "Output options:\n");
  fprintf(stderr, "  -q, --quiet           Silent mode, no output\n");
  fprintf(stderr, "  -v, --verbose         Detailed output\n\n");
  fprintf(stderr, "Other options:\n");
  fprintf(stderr,
          "  -e, --embeddings PATH Load pre-computed text embeddings\n");
  fprintf(stderr, "  -h, --help            Show this help\n\n");
  fprintf(stderr, "Examples:\n");
  fprintf(stderr, "  %s -d model/ -p \"a cat on a rainbow\" -o cat.png\n",
          prog);
  fprintf(stderr,
          "  %s -d model/ -p \"oil painting\" -i photo.png -o art.png -t 0.7\n",
          prog);
}

/* ========================================================================
 * Main
 * ======================================================================== */

int main(int argc, char *argv[]) {
#ifdef USE_METAL
  flux_metal_init();
#endif

  /* Command line options */
  static struct option long_options[] = {
      {"dir", required_argument, 0, 'd'},
      {"prompt", required_argument, 0, 'p'},
      {"output", required_argument, 0, 'o'},
      {"width", required_argument, 0, 'W'},
      {"height", required_argument, 0, 'H'},
      {"steps", required_argument, 0, 's'},
      {"guidance", required_argument, 0, 'g'},
      {"seed", required_argument, 0, 'S'},
      {"input", required_argument, 0, 'i'},
      {"strength", required_argument, 0, 't'},
      {"embeddings", required_argument, 0, 'e'},
      {"noise", required_argument, 0, 'n'},
      {"quiet", no_argument, 0, 'q'},
      {"verbose", no_argument, 0, 'v'},
      {"help", no_argument, 0, 'h'},
      {"version", no_argument, 0, 'V'},
      {0, 0, 0, 0}};

  /* Parse arguments */
  char *model_dir = NULL;
  char *prompt = NULL;
  char *output_path = NULL;
  char *input_path = NULL;
  char *embeddings_path = NULL;
  char *noise_path = NULL;

  flux_params params = {.width = DEFAULT_WIDTH,
                        .height = DEFAULT_HEIGHT,
                        .num_steps = DEFAULT_STEPS,
                        .guidance_scale = DEFAULT_GUIDANCE,
                        .seed = -1,
                        .strength = DEFAULT_STRENGTH};

  int width_set = 0, height_set = 0;

  int opt;
  while ((opt = getopt_long(argc, argv, "d:p:o:W:H:s:g:S:i:t:e:n:qvhV",
                            long_options, NULL)) != -1) {
    switch (opt) {
    case 'd':
      model_dir = optarg;
      break;
    case 'p':
      prompt = optarg;
      break;
    case 'o':
      output_path = optarg;
      break;
    case 'W':
      params.width = atoi(optarg);
      width_set = 1;
      break;
    case 'H':
      params.height = atoi(optarg);
      height_set = 1;
      break;
    case 's':
      params.num_steps = atoi(optarg);
      break;
    case 'g':
      params.guidance_scale = atof(optarg);
      break;
    case 'S':
      params.seed = atoll(optarg);
      break;
    case 'i':
      input_path = optarg;
      break;
    case 't':
      params.strength = atof(optarg);
      break;
    case 'e':
      embeddings_path = optarg;
      break;
    case 'n':
      noise_path = optarg;
      break;
    case 'q':
      output_level = OUTPUT_QUIET;
      break;
    case 'v':
      output_level = OUTPUT_VERBOSE;
      break;
    case 'h':
      print_usage(argv[0]);
      return 0;
    case 'V':
      fprintf(stderr, "FLUX.2 klein 4B v1.0.0\n");
      return 0;
    default:
      print_usage(argv[0]);
      return 1;
    }
  }

  /* Validate required arguments */
  if (!model_dir) {
    fprintf(stderr, "Error: Model directory (-d) is required\n\n");
    print_usage(argv[0]);
    return 1;
  }
  if (!prompt && !embeddings_path) {
    fprintf(stderr,
            "Error: Prompt (-p) or embeddings file (-e) is required\n\n");
    print_usage(argv[0]);
    return 1;
  }
  if (!output_path) {
    fprintf(stderr, "Error: Output path (-o) is required\n\n");
    print_usage(argv[0]);
    return 1;
  }

  /* Validate parameters */
  if (params.width < 64 || params.width > 4096) {
    fprintf(stderr, "Error: Width must be between 64 and 4096\n");
    return 1;
  }
  if (params.height < 64 || params.height > 4096) {
    fprintf(stderr, "Error: Height must be between 64 and 4096\n");
    return 1;
  }
  if (params.num_steps < 1 || params.num_steps > 100) {
    fprintf(stderr, "Error: Steps must be between 1 and 100\n");
    return 1;
  }
  if (params.strength < 0.0f || params.strength > 1.0f) {
    fprintf(stderr, "Error: Strength must be between 0.0 and 1.0\n");
    return 1;
  }

  /* Set seed */
  int64_t actual_seed;
  if (params.seed >= 0) {
    actual_seed = params.seed;
  } else {
    actual_seed = (int64_t)time(NULL);
  }
  flux_set_seed(actual_seed);
  LOG_NORMAL("Seed: %lld\n", (long long)actual_seed);

  /* Verbose header */
  LOG_VERBOSE("FLUX.2 klein 4B Image Generator\n");
  LOG_VERBOSE("================================\n");
  LOG_VERBOSE("Model: %s\n", model_dir);
  if (prompt)
    LOG_VERBOSE("Prompt: %s\n", prompt);
  LOG_VERBOSE("Output: %s\n", output_path);
  LOG_VERBOSE("Size: %dx%d\n", params.width, params.height);
  LOG_VERBOSE("Steps: %d\n", params.num_steps);
  if (input_path) {
    LOG_VERBOSE("Input: %s\n", input_path);
    LOG_VERBOSE("Strength: %.2f\n", params.strength);
  }
  LOG_VERBOSE("\n");

  /* Load model */
  LOG_NORMAL("Loading model...");
  if (output_level >= OUTPUT_NORMAL)
    fflush(stderr);
  timer_begin();

  flux_ctx *ctx = flux_load_dir(model_dir);
  if (!ctx) {
    fprintf(stderr, "\nError: Failed to load model: %s\n", flux_get_error());
    return 1;
  }

  double load_time = timer_end();
  LOG_NORMAL(" done (%.1fs)\n", load_time);
  LOG_VERBOSE("  Model info: %s\n", flux_model_info(ctx));

  /* Set up progress callbacks (for normal and verbose modes) */
  if (output_level >= OUTPUT_NORMAL) {
    cli_setup_progress();
  }

  /* Generate image */
  flux_image *output = NULL;
  clock_t total_start = clock();

  if (input_path) {
    /* ============== Image-to-image mode ============== */
    LOG_NORMAL("Loading input image...");
    if (output_level >= OUTPUT_NORMAL)
      fflush(stderr);
    timer_begin();

    flux_image *input = flux_image_load(input_path);
    if (!input) {
      fprintf(stderr, "\nError: Failed to load input image: %s\n", input_path);
      flux_free(ctx);
      return 1;
    }

    LOG_NORMAL(" done (%.1fs)\n", timer_end());
    LOG_VERBOSE("  Input: %dx%d, %d channels\n", input->width, input->height,
                input->channels);

    /* Use input image dimensions if not explicitly set */
    if (!width_set)
      params.width = input->width;
    if (!height_set)
      params.height = input->height;

    /* Generate */
    output = flux_img2img(ctx, prompt, input, &params);
    flux_image_free(input);

  } else if (embeddings_path) {
    /* ============== External embeddings mode ============== */
    LOG_NORMAL("Loading embeddings...");
    if (output_level >= OUTPUT_NORMAL)
      fflush(stderr);
    timer_begin();

    FILE *emb_file = fopen(embeddings_path, "rb");
    if (!emb_file) {
      fprintf(stderr, "\nError: Failed to open embeddings file: %s\n",
              embeddings_path);
      flux_free(ctx);
      return 1;
    }

    fseek(emb_file, 0, SEEK_END);
    long file_size = ftell(emb_file);
    fseek(emb_file, 0, SEEK_SET);

    int text_dim = FLUX_TEXT_DIM;
    int text_seq = file_size / (text_dim * sizeof(float));

    float *text_emb = (float *)malloc(file_size);
    if (fread(text_emb, 1, file_size, emb_file) != (size_t)file_size) {
      fprintf(stderr, "\nError: Failed to read embeddings file\n");
      free(text_emb);
      fclose(emb_file);
      flux_free(ctx);
      return 1;
    }
    fclose(emb_file);

    LOG_NORMAL(" done (%.1fs)\n", timer_end());
    LOG_VERBOSE("  Embeddings: %d tokens x %d dims (%.2f MB)\n", text_seq,
                text_dim, file_size / (1024.0 * 1024.0));

    /* Load noise if provided */
    float *noise = NULL;
    int noise_size = 0;
    if (noise_path) {
      LOG_VERBOSE("Loading noise from %s...\n", noise_path);

      FILE *noise_file = fopen(noise_path, "rb");
      if (!noise_file) {
        fprintf(stderr, "Error: Failed to open noise file: %s\n", noise_path);
        free(text_emb);
        flux_free(ctx);
        return 1;
      }

      fseek(noise_file, 0, SEEK_END);
      long noise_file_size = ftell(noise_file);
      fseek(noise_file, 0, SEEK_SET);

      noise_size = noise_file_size / sizeof(float);
      noise = (float *)malloc(noise_file_size);
      if (fread(noise, 1, noise_file_size, noise_file) !=
          (size_t)noise_file_size) {
        fprintf(stderr, "Error: Failed to read noise file\n");
        free(noise);
        free(text_emb);
        fclose(noise_file);
        flux_free(ctx);
        return 1;
      }
      fclose(noise_file);
      LOG_VERBOSE("  Noise: %d floats\n", noise_size);
    }

    /* Generate */
    if (noise) {
      output = flux_generate_with_embeddings_and_noise(
          ctx, text_emb, text_seq, noise, noise_size, &params);
      free(noise);
    } else {
      output = flux_generate_with_embeddings(ctx, text_emb, text_seq, &params);
    }
    free(text_emb);

  } else {
    /* ============== Text-to-image mode ============== */
    /* Note: flux_generate handles text encoding internally.
     * We can't easily time it separately without modifying the library.
     * The progress callbacks will show denoising progress. */
    output = flux_generate(ctx, prompt, &params);
  }

  /* Finish progress display */
  cli_finish_progress();

  if (!output) {
    fprintf(stderr, "Error: Generation failed: %s\n", flux_get_error());
    flux_free(ctx);
    return 1;
  }

  double total_time = (double)(clock() - total_start) / CLOCKS_PER_SEC;
  LOG_VERBOSE("Generated in %.1fs total\n", total_time);
  LOG_VERBOSE("  Output: %dx%d, %d channels\n", output->width, output->height,
              output->channels);

  /* Save output */
  LOG_NORMAL("Saving...");
  if (output_level >= OUTPUT_NORMAL)
    fflush(stderr);
  timer_begin();

  if (flux_image_save(output, output_path) != 0) {
    fprintf(stderr, "\nError: Failed to save image: %s\n", output_path);
    flux_image_free(output);
    flux_free(ctx);
    return 1;
  }

  LOG_NORMAL(" %s (%.1fs)\n", output_path, timer_end());

  /* Cleanup */
  flux_image_free(output);
  flux_free(ctx);

#ifdef USE_METAL
  flux_metal_cleanup();
#endif

  return 0;
}
