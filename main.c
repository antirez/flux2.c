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
 *   -S, --seed N          Random seed (-1 for random)
 *   -i, --input PATH      Input image for img2img
 *   -q, --quiet           No output, just generate
 *   -v, --verbose         Extra detailed output
 *   -h, --help            Show help
 */

#include "flux.h"
#include "flux_kernels.h"
#include "flux_cli.h"
#include "terminals.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <time.h>
#include <sys/time.h>

#ifdef USE_METAL
#include "flux_metal.h"
#endif

#ifdef USE_ROCM
#include "rocm/flux_rocm.h"
#endif

/* ========================================================================
 * Verbosity Levels
 * ======================================================================== */

typedef enum {
    OUTPUT_QUIET = 0,    /* No output */
    OUTPUT_NORMAL = 1,   /* Progress and essential info */
    OUTPUT_VERBOSE = 2   /* Detailed debugging info */
} output_level_t;

static output_level_t output_level = OUTPUT_NORMAL;

/* ========================================================================
 * CLI Progress Callbacks
 * ======================================================================== */

static int cli_current_step = 0;
static int cli_legend_printed = 0;
static term_graphics_proto cli_graphics_proto = TERM_PROTO_NONE;

/* Called at the start of each sampling step */
static void cli_step_callback(int step, int total) {
    if (output_level == OUTPUT_QUIET) return;

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
static void cli_substep_callback(flux_substep_type_t type, int index, int total) {
    if (output_level == OUTPUT_QUIET) return;
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

/* Track phase timing (wall-clock) */
static struct timeval cli_phase_start_tv;
static const char *cli_current_phase = NULL;

/* Called at phase boundaries (encoding text, decoding image, etc.) */
static void cli_phase_callback(const char *phase, int done) {
    if (output_level == OUTPUT_QUIET) return;

    if (!done) {
        /* If we were showing step progress, end that line first */
        if (cli_current_step > 0) {
            fprintf(stderr, "\n");
            cli_current_step = 0;
        }

        /* Phase starting */
        cli_current_phase = phase;
        gettimeofday(&cli_phase_start_tv, NULL);

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
        struct timeval now;
        gettimeofday(&now, NULL);
        double elapsed = (now.tv_sec - cli_phase_start_tv.tv_sec) +
                         (now.tv_usec - cli_phase_start_tv.tv_usec) / 1000000.0;
        fprintf(stderr, " done (%.1fs)\n", elapsed);
        cli_current_phase = NULL;
    }
}

/* Set up CLI progress callbacks */
/* Step image callback - display intermediate images in terminal */
static void cli_step_image_callback(int step, int total, const flux_image *img) {
    (void)total;
    fprintf(stderr, "\n[Step %d]\n", step);
    terminal_display_image(img, cli_graphics_proto);
}

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
 * Timing Helper (wall-clock time)
 * ======================================================================== */

static struct timeval timer_start_tv;

static void timer_begin(void) {
    gettimeofday(&timer_start_tv, NULL);
}

static double timer_end(void) {
    struct timeval now;
    gettimeofday(&now, NULL);
    return (now.tv_sec - timer_start_tv.tv_sec) +
           (now.tv_usec - timer_start_tv.tv_usec) / 1000000.0;
}

/* ========================================================================
 * Output Helpers
 * ======================================================================== */

/* Print if not quiet */
#define LOG_NORMAL(...) do { if (output_level >= OUTPUT_NORMAL) fprintf(stderr, __VA_ARGS__); } while(0)

/* Print only in verbose mode */
#define LOG_VERBOSE(...) do { if (output_level >= OUTPUT_VERBOSE) fprintf(stderr, __VA_ARGS__); } while(0)

/* ========================================================================
 * Usage and Help
 * ======================================================================== */

/* Default values */
#define DEFAULT_WIDTH 256
#define DEFAULT_HEIGHT 256
#define DEFAULT_STEPS 4
#define MAX_INPUT_IMAGES 16

static void print_usage(const char *prog) {
    fprintf(stderr, "FLUX.2 klein 4B - Pure C Image Generation\n\n");
    fprintf(stderr, "Usage: %s [options]\n\n", prog);
    fprintf(stderr, "Required:\n");
    fprintf(stderr, "  -d, --dir PATH        Path to model directory\n");
    fprintf(stderr, "  -p, --prompt TEXT     Text prompt for generation\n");
    fprintf(stderr, "  -o, --output PATH     Output image path (.png, .ppm)\n\n");
    fprintf(stderr, "Generation options:\n");
    fprintf(stderr, "  -W, --width N         Output width (default: %d)\n", DEFAULT_WIDTH);
    fprintf(stderr, "  -H, --height N        Output height (default: %d)\n", DEFAULT_HEIGHT);
    fprintf(stderr, "  -s, --steps N         Sampling steps (default: %d)\n", DEFAULT_STEPS);
    fprintf(stderr, "  -S, --seed N          Random seed (-1 for random)\n\n");
    fprintf(stderr, "Reference images (img2img / multi-reference):\n");
    fprintf(stderr, "  -i, --input PATH      Reference image (can specify up to %d)\n", MAX_INPUT_IMAGES);
    fprintf(stderr, "                        Multiple -i flags combine images via in-context conditioning\n\n");
    fprintf(stderr, "Output options:\n");
    fprintf(stderr, "  -q, --quiet           Silent mode, no output\n");
    fprintf(stderr, "  -v, --verbose         Detailed output\n");
    fprintf(stderr, "      --show            Display image in terminal (auto-detects Kitty/Ghostty/iTerm2)\n");
    fprintf(stderr, "      --show-steps      Display each denoising step (slower)\n\n");
    fprintf(stderr, "Other options:\n");
    fprintf(stderr, "  -e, --embeddings PATH Load pre-computed text embeddings\n");
    fprintf(stderr, "  -m, --mmap            Use memory-mapped weights (default, fastest on MPS)\n");
    fprintf(stderr, "      --no-mmap         Disable mmap, load all weights upfront\n");
    fprintf(stderr, "  -h, --help            Show this help\n\n");
    fprintf(stderr, "Examples:\n");
    fprintf(stderr, "  %s -d model/ -p \"a cat on a rainbow\" -o cat.png\n", prog);
    fprintf(stderr, "  %s -d model/ -p \"oil painting\" -i photo.png -o art.png\n", prog);
    fprintf(stderr, "  %s -d model/ -p \"combine them\" -i car.png -i beach.png -o result.png\n", prog);
}

/* ========================================================================
 * Main
 * ======================================================================== */

int main(int argc, char *argv[]) {
#ifdef USE_METAL
    flux_metal_init();
#elif defined(USE_ROCM)
    if (flux_rocm_init()) {
        /* ROCm initialized successfully - message printed by init */
    } else {
        fprintf(stderr, "ROCm: Falling back to BLAS\n");
    }
#elif defined(USE_BLAS)
    fprintf(stderr, "BLAS: CPU acceleration enabled (Accelerate/OpenBLAS)\n");
#else
    fprintf(stderr, "Generic: Pure C backend (no acceleration)\n");
#endif

    /* Command line options */
    static struct option long_options[] = {
        {"dir",        required_argument, 0, 'd'},
        {"prompt",     required_argument, 0, 'p'},
        {"output",     required_argument, 0, 'o'},
        {"width",      required_argument, 0, 'W'},
        {"height",     required_argument, 0, 'H'},
        {"steps",      required_argument, 0, 's'},
        {"seed",       required_argument, 0, 'S'},
        {"input",      required_argument, 0, 'i'},
        {"embeddings", required_argument, 0, 'e'},
        {"noise",      required_argument, 0, 'n'},
        {"quiet",      no_argument,       0, 'q'},
        {"verbose",    no_argument,       0, 'v'},
        {"help",       no_argument,       0, 'h'},
        {"version",    no_argument,       0, 'V'},
        {"mmap",       no_argument,       0, 'm'},
        {"no-mmap",    no_argument,       0, 'M'},
        {"show",       no_argument,       0, 'k'},
        {"show-steps", no_argument,       0, 'K'},
        {"debug-py",   no_argument,       0, 'D'},
        {0, 0, 0, 0}
    };

    /* Parse arguments */
    char *model_dir = NULL;
    char *prompt = NULL;
    char *output_path = NULL;
    char *input_paths[MAX_INPUT_IMAGES] = {NULL};
    int num_inputs = 0;
    char *embeddings_path = NULL;
    char *noise_path = NULL;

    flux_params params = {
        .width = DEFAULT_WIDTH,
        .height = DEFAULT_HEIGHT,
        .num_steps = DEFAULT_STEPS,
        .seed = -1
    };

    int width_set = 0, height_set = 0;
    int use_mmap = 1;  /* mmap is default (fastest on MPS) */
    int show_image = 0;
    int show_steps = 0;
    int debug_py = 0;
    term_graphics_proto graphics_proto = detect_terminal_graphics();

    int opt;
    while ((opt = getopt_long(argc, argv, "d:p:o:W:H:s:g:S:i:t:e:n:qvhVmMD",
                              long_options, NULL)) != -1) {
        switch (opt) {
            case 'd': model_dir = optarg; break;
            case 'p': prompt = optarg; break;
            case 'o': output_path = optarg; break;
            case 'W': params.width = atoi(optarg); width_set = 1; break;
            case 'H': params.height = atoi(optarg); height_set = 1; break;
            case 's': params.num_steps = atoi(optarg); break;
            case 'S': params.seed = atoll(optarg); break;
            case 'i':
                if (num_inputs < MAX_INPUT_IMAGES) {
                    input_paths[num_inputs++] = optarg;
                } else {
                    fprintf(stderr, "Warning: Maximum %d input images supported\n", MAX_INPUT_IMAGES);
                }
                break;
            case 'e': embeddings_path = optarg; break;
            case 'n': noise_path = optarg; break;
            case 'q': output_level = OUTPUT_QUIET; break;
            case 'v': output_level = OUTPUT_VERBOSE; break;
            case 'h': print_usage(argv[0]); return 0;
            case 'V':
                fprintf(stderr, "FLUX.2 klein 4B v1.0.0\n");
                return 0;
            case 'm': use_mmap = 1; break;
            case 'M': use_mmap = 0; break;
            case 'k': show_image = 1; break;
            case 'K': show_steps = 1; break;
            case 'D': debug_py = 1; break;
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

    /* Interactive mode: -d provided but no -p, -e, -o, or --debug-py */
    int interactive_mode = (!prompt && !embeddings_path && !output_path && !debug_py);

    if (!interactive_mode) {
        if (!prompt && !embeddings_path && !debug_py) {
            fprintf(stderr, "Error: Prompt (-p) or embeddings file (-e) is required\n\n");
            print_usage(argv[0]);
            return 1;
        }
        if (!output_path) {
            fprintf(stderr, "Error: Output path (-o) is required\n\n");
            print_usage(argv[0]);
            return 1;
        }
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
    if (params.num_steps < 1 || params.num_steps > FLUX_MAX_STEPS) {
        fprintf(stderr, "Error: Steps must be between 1 and %d\n", FLUX_MAX_STEPS);
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
    if (prompt) LOG_VERBOSE("Prompt: %s\n", prompt);
    LOG_VERBOSE("Output: %s\n", output_path);
    LOG_VERBOSE("Size: %dx%d\n", params.width, params.height);
    LOG_VERBOSE("Steps: %d\n", params.num_steps);
    if (num_inputs > 0) {
        LOG_VERBOSE("Input images: %d\n", num_inputs);
        for (int i = 0; i < num_inputs; i++) {
            LOG_VERBOSE("  [%d] %s\n", i + 1, input_paths[i]);
        }
    }
    LOG_VERBOSE("\n");

    /* Load model (VAE only at startup, other components loaded on-demand) */
    LOG_NORMAL("Loading VAE...");
    if (output_level >= OUTPUT_NORMAL) fflush(stderr);
    timer_begin();

    flux_ctx *ctx = flux_load_dir(model_dir);
    if (!ctx) {
        fprintf(stderr, "\nError: Failed to load model: %s\n", flux_get_error());
        return 1;
    }

    /* Enable mmap mode if requested (reduces memory, slower inference) */
    if (use_mmap) {
        flux_set_mmap(ctx, 1);
        LOG_VERBOSE("  Using mmap mode for text encoder (lower memory)\n");
    }

    double load_time = timer_end();
    LOG_NORMAL(" done (%.1fs)\n", load_time);
    LOG_VERBOSE("  Model info: %s\n", flux_model_info(ctx));

    /* Interactive mode: start REPL */
    if (interactive_mode) {
        int rc = flux_cli_run(ctx, model_dir);
        flux_free(ctx);
        return rc;
    }

    /* Set up progress callbacks (for normal and verbose modes) */
    if (output_level >= OUTPUT_NORMAL) {
        cli_setup_progress();
    }

    /* Set up step image callback if requested */
    if (show_steps) {
        if (graphics_proto == TERM_PROTO_NONE) {
            fprintf(stderr, "Warning: --show-steps requires a supported terminal (Kitty, Ghostty, or iTerm2)\n");
        } else {
            cli_graphics_proto = graphics_proto;
            flux_set_step_image_callback(ctx, cli_step_image_callback);
        }
    }

    /* Generate image */
    flux_image *output = NULL;
    struct timeval total_start_tv;
    gettimeofday(&total_start_tv, NULL);

    if (debug_py) {
        /* ============== Debug mode: use Python inputs ============== */
        LOG_NORMAL("Debug mode: loading Python inputs from /tmp/py_*.bin\n");
        output = flux_img2img_debug_py(ctx, &params);
    } else if (num_inputs > 0) {
        /* ============== Image-to-image mode (single or multi-reference) ============== */
        LOG_NORMAL("Loading %d input image%s...", num_inputs, num_inputs > 1 ? "s" : "");
        if (output_level >= OUTPUT_NORMAL) fflush(stderr);
        timer_begin();

        flux_image *inputs[MAX_INPUT_IMAGES];
        for (int i = 0; i < num_inputs; i++) {
            inputs[i] = flux_image_load(input_paths[i]);
            if (!inputs[i]) {
                fprintf(stderr, "\nError: Failed to load input image: %s\n", input_paths[i]);
                for (int j = 0; j < i; j++) flux_image_free(inputs[j]);
                flux_free(ctx);
                return 1;
            }
        }

        LOG_NORMAL(" done (%.1fs)\n", timer_end());
        for (int i = 0; i < num_inputs; i++) {
            LOG_VERBOSE("  Input[%d]: %dx%d, %d channels\n",
                        i + 1, inputs[i]->width, inputs[i]->height, inputs[i]->channels);
        }

        /* Use first input image dimensions if not explicitly set */
        if (!width_set) params.width = inputs[0]->width;
        if (!height_set) params.height = inputs[0]->height;

        /* Generate with multi-reference */
        output = flux_multiref(ctx, prompt, (const flux_image **)inputs, num_inputs, &params);

        for (int i = 0; i < num_inputs; i++) {
            flux_image_free(inputs[i]);
        }

    } else if (embeddings_path) {
        /* ============== External embeddings mode ============== */
        LOG_NORMAL("Loading embeddings...");
        if (output_level >= OUTPUT_NORMAL) fflush(stderr);
        timer_begin();

        FILE *emb_file = fopen(embeddings_path, "rb");
        if (!emb_file) {
            fprintf(stderr, "\nError: Failed to open embeddings file: %s\n", embeddings_path);
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
        LOG_VERBOSE("  Embeddings: %d tokens x %d dims (%.2f MB)\n",
                    text_seq, text_dim, file_size / (1024.0 * 1024.0));

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
            if (fread(noise, 1, noise_file_size, noise_file) != (size_t)noise_file_size) {
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
            output = flux_generate_with_embeddings_and_noise(ctx, text_emb, text_seq,
                                                              noise, noise_size, &params);
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

    /* Clear step image callback if it was set */
    if (show_steps) {
        flux_set_step_image_callback(ctx, NULL);
    }

    if (!output) {
        fprintf(stderr, "Error: Generation failed: %s\n", flux_get_error());
        flux_free(ctx);
        return 1;
    }

    struct timeval total_end_tv;
    gettimeofday(&total_end_tv, NULL);
    double total_time = (total_end_tv.tv_sec - total_start_tv.tv_sec) +
                        (total_end_tv.tv_usec - total_start_tv.tv_usec) / 1000000.0;
    LOG_VERBOSE("Generated in %.1fs total\n", total_time);
    LOG_VERBOSE("  Output: %dx%d, %d channels\n",
                output->width, output->height, output->channels);

    /* Save output */
    LOG_NORMAL("Saving...");
    if (output_level >= OUTPUT_NORMAL) fflush(stderr);
    timer_begin();

    if (flux_image_save_with_seed(output, output_path, actual_seed) != 0) {
        fprintf(stderr, "\nError: Failed to save image: %s\n", output_path);
        flux_image_free(output);
        flux_free(ctx);
        return 1;
    }

    LOG_NORMAL(" %s %dx%d (%.1fs)\n", output_path, output->width, output->height, timer_end());

    /* Display image in terminal if requested */
    if (show_image) {
        terminal_display_png(output_path, graphics_proto);
    }

    /* Print total time (always, unless quiet) */
    struct timeval final_tv;
    gettimeofday(&final_tv, NULL);
    double total_time_final = (final_tv.tv_sec - total_start_tv.tv_sec) +
                              (final_tv.tv_usec - total_start_tv.tv_usec) / 1000000.0;
    LOG_NORMAL("Total generation time: %.1f seconds\n", load_time + total_time_final);

    /* Cleanup */
    flux_image_free(output);
    flux_free(ctx);

#ifdef USE_METAL
    flux_metal_cleanup();
#endif
#ifdef USE_ROCM
    flux_rocm_cleanup();
#endif

    return 0;
}
