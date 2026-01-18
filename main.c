/*
 * FLUX CLI Application
 *
 * Command-line interface for FLUX.2 klein 4B image generation.
 *
 * Usage:
 *   flux -m model.bin -p "prompt" -o output.png [options]
 *
 * Options:
 *   -m, --model PATH      Path to model file (.bin)
 *   -d, --dir PATH        Path to model directory (safetensors)
 *   -p, --prompt TEXT     Text prompt for generation
 *   -o, --output PATH     Output image path
 *   -W, --width N         Output width (default: 1024)
 *   -H, --height N        Output height (default: 1024)
 *   -s, --steps N         Number of sampling steps (default: 4)
 *   -g, --guidance N      Guidance scale (default: 1.0)
 *   -S, --seed N          Random seed (-1 for random)
 *   -i, --input PATH      Input image for img2img
 *   -t, --strength N      Img2img strength (0.0-1.0)
 *   -v, --verbose         Enable verbose output
 *   -h, --help            Show help
 */

#include "flux.h"
#include "flux_kernels.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <time.h>

/* ========================================================================
 * CLI Progress Callbacks
 * ======================================================================== */

static int cli_current_step = 0;

/* Called at the start of each sampling step */
static void cli_step_callback(int step, int total) {
    /* Print newline to end previous step's progress (if any) */
    if (cli_current_step > 0) {
        fprintf(stderr, "\n");
    }
    cli_current_step = step;
    fprintf(stderr, "Step %d/%d ", step, total);
    fflush(stderr);
}

/* Called for each substep within transformer forward */
static void cli_substep_callback(flux_substep_type_t type, int index, int total) {
    (void)total;  /* We could use this for more detailed progress */

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

/* Set up CLI progress callbacks */
static void cli_setup_progress(void) {
    cli_current_step = 0;
    flux_step_callback = cli_step_callback;
    flux_substep_callback = cli_substep_callback;
}

/* Clean up after generation (print final newline) */
static void cli_finish_progress(void) {
    if (cli_current_step > 0) {
        fprintf(stderr, "\n");
        cli_current_step = 0;
    }
    flux_step_callback = NULL;
    flux_substep_callback = NULL;
}

/* Default values */
#define DEFAULT_WIDTH 1024
#define DEFAULT_HEIGHT 1024
#define DEFAULT_STEPS 4
#define DEFAULT_GUIDANCE 1.0f
#define DEFAULT_STRENGTH 0.75f

static void print_usage(const char *prog) {
    fprintf(stderr, "FLUX.2 klein 4B - Pure C Image Generation\n\n");
    fprintf(stderr, "Usage: %s [options]\n\n", prog);
    fprintf(stderr, "Required:\n");
    fprintf(stderr, "  -d, --dir PATH        Path to model directory (safetensors)\n");
    fprintf(stderr, "  -p, --prompt TEXT     Text prompt for generation\n");
    fprintf(stderr, "  -o, --output PATH     Output image path (.png, .ppm)\n\n");
    fprintf(stderr, "Generation options:\n");
    fprintf(stderr, "  -W, --width N         Output width (default: %d)\n", DEFAULT_WIDTH);
    fprintf(stderr, "  -H, --height N        Output height (default: %d)\n", DEFAULT_HEIGHT);
    fprintf(stderr, "  -s, --steps N         Sampling steps (default: %d)\n", DEFAULT_STEPS);
    fprintf(stderr, "  -g, --guidance N      Guidance scale (default: %.1f)\n", DEFAULT_GUIDANCE);
    fprintf(stderr, "  -S, --seed N          Random seed (-1 for random)\n\n");
    fprintf(stderr, "Image-to-image options:\n");
    fprintf(stderr, "  -i, --input PATH      Input image for img2img\n");
    fprintf(stderr, "  -t, --strength N      Strength 0.0-1.0 (default: %.2f)\n\n", DEFAULT_STRENGTH);
    fprintf(stderr, "Other options:\n");
    fprintf(stderr, "  -e, --embeddings PATH Load text embeddings from binary file\n");
    fprintf(stderr, "  -v, --verbose         Enable verbose output\n");
    fprintf(stderr, "  -h, --help            Show this help\n\n");
    fprintf(stderr, "Examples:\n");
    fprintf(stderr, "  %s -d model/ -p \"a cat on a rainbow\" -o cat.png\n", prog);
    fprintf(stderr, "  %s -d model/ -p \"oil painting style\" -i photo.png -o art.png -t 0.7\n", prog);
}

static void print_version(void) {
    fprintf(stderr, "FLUX.2 klein 4B Inference Engine\n");
    fprintf(stderr, "Version: 1.0.0\n");
    fprintf(stderr, "Pure C implementation, no external dependencies\n");
}

int main(int argc, char *argv[]) {
    /* Command line options */
    static struct option long_options[] = {
        {"dir",      required_argument, 0, 'd'},
        {"prompt",   required_argument, 0, 'p'},
        {"output",   required_argument, 0, 'o'},
        {"width",    required_argument, 0, 'W'},
        {"height",   required_argument, 0, 'H'},
        {"steps",    required_argument, 0, 's'},
        {"guidance", required_argument, 0, 'g'},
        {"seed",     required_argument, 0, 'S'},
        {"input",    required_argument, 0, 'i'},
        {"strength", required_argument, 0, 't'},
        {"embeddings", required_argument, 0, 'e'},
        {"noise",    required_argument, 0, 'n'},
        {"verbose",  no_argument,       0, 'v'},
        {"help",     no_argument,       0, 'h'},
        {"version",  no_argument,       0, 'V'},
        {0, 0, 0, 0}
    };

    /* Parse arguments */
    char *model_dir = NULL;
    char *prompt = NULL;
    char *output_path = NULL;
    char *input_path = NULL;
    char *embeddings_path = NULL;
    char *noise_path = NULL;

    flux_params params = {
        .width = DEFAULT_WIDTH,
        .height = DEFAULT_HEIGHT,
        .num_steps = DEFAULT_STEPS,
        .guidance_scale = DEFAULT_GUIDANCE,
        .seed = -1,
        .strength = DEFAULT_STRENGTH
    };

    int verbose = 0;

    int opt;
    while ((opt = getopt_long(argc, argv, "d:p:o:W:H:s:g:S:i:t:e:n:vhV",
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
                break;
            case 'H':
                params.height = atoi(optarg);
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
            case 'v':
                verbose = 1;
                break;
            case 'h':
                print_usage(argv[0]);
                return 0;
            case 'V':
                print_version();
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
        fprintf(stderr, "Error: Prompt (-p) or embeddings file (-e) is required\n\n");
        print_usage(argv[0]);
        return 1;
    }
    if (!output_path) {
        fprintf(stderr, "Error: Output path is required (-o)\n\n");
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

    if (verbose) {
        fprintf(stderr, "FLUX.2 klein 4B Image Generator\n");
        fprintf(stderr, "================================\n");
        fprintf(stderr, "Model: %s\n", model_dir);
        fprintf(stderr, "Prompt: %s\n", prompt);
        fprintf(stderr, "Output: %s\n", output_path);
        fprintf(stderr, "Size: %dx%d\n", params.width, params.height);
        fprintf(stderr, "Steps: %d\n", params.num_steps);
        fprintf(stderr, "Guidance: %.2f\n", params.guidance_scale);
        if (params.seed >= 0) {
            fprintf(stderr, "Seed: %lld\n", (long long)params.seed);
        } else {
            fprintf(stderr, "Seed: random\n");
        }
        if (input_path) {
            fprintf(stderr, "Input: %s\n", input_path);
            fprintf(stderr, "Strength: %.2f\n", params.strength);
        }
        fprintf(stderr, "\n");
    }

    /* Load model */
    if (verbose) {
        fprintf(stderr, "Loading model...\n");
    }

    clock_t start = clock();

    flux_ctx *ctx = flux_load_dir(model_dir);
    if (!ctx) {
        fprintf(stderr, "Error: Failed to load model: %s\n", flux_get_error());
        return 1;
    }

    if (verbose) {
        double load_time = (double)(clock() - start) / CLOCKS_PER_SEC;
        fprintf(stderr, "Model loaded in %.2f seconds\n", load_time);
        fprintf(stderr, "Model info: %s\n\n", flux_model_info(ctx));

        /* Set up progress callbacks for verbose mode */
        cli_setup_progress();
    }

    /* Set seed */
    if (params.seed >= 0) {
        flux_set_seed(params.seed);
    } else {
        flux_set_seed(time(NULL));
    }

    /* Generate image */
    flux_image *output = NULL;

    start = clock();

    if (input_path) {
        /* Image-to-image mode */
        if (verbose) {
            fprintf(stderr, "Loading input image...\n");
        }

        flux_image *input = flux_image_load(input_path);
        if (!input) {
            fprintf(stderr, "Error: Failed to load input image: %s\n", input_path);
            flux_free(ctx);
            return 1;
        }

        if (verbose) {
            fprintf(stderr, "Input: %dx%d, %d channels\n",
                    input->width, input->height, input->channels);
            fprintf(stderr, "Generating...\n");
        }

        output = flux_img2img(ctx, prompt, input, &params);
        flux_image_free(input);
    } else if (embeddings_path) {
        /* Text-to-image mode with external embeddings */
        if (verbose) {
            fprintf(stderr, "Loading embeddings from %s...\n", embeddings_path);
        }

        /* Load embeddings file */
        FILE *emb_file = fopen(embeddings_path, "rb");
        if (!emb_file) {
            fprintf(stderr, "Error: Failed to open embeddings file: %s\n", embeddings_path);
            flux_free(ctx);
            return 1;
        }

        /* Get file size and compute dimensions */
        fseek(emb_file, 0, SEEK_END);
        long file_size = ftell(emb_file);
        fseek(emb_file, 0, SEEK_SET);

        /* Expected: [1, 512, 7680] = 512 * 7680 * 4 bytes = 15728640 bytes */
        int text_dim = FLUX_TEXT_DIM;  /* 7680 */
        int text_seq = file_size / (text_dim * sizeof(float));

        if (verbose) {
            fprintf(stderr, "Embeddings: %d tokens x %d dims (%.2f MB)\n",
                    text_seq, text_dim, file_size / (1024.0 * 1024.0));
        }

        float *text_emb = (float *)malloc(file_size);
        if (fread(text_emb, 1, file_size, emb_file) != (size_t)file_size) {
            fprintf(stderr, "Error: Failed to read embeddings file\n");
            free(text_emb);
            fclose(emb_file);
            flux_free(ctx);
            return 1;
        }
        fclose(emb_file);

        /* Load noise if provided */
        float *noise = NULL;
        int noise_size = 0;
        if (noise_path) {
            if (verbose) {
                fprintf(stderr, "Loading noise from %s...\n", noise_path);
            }

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

            if (verbose) {
                fprintf(stderr, "Noise: %d floats (%.2f KB)\n",
                        noise_size, noise_file_size / 1024.0);
            }
        }

        if (verbose) {
            fprintf(stderr, "Generating with external embeddings%s...\n",
                    noise ? " and noise" : "");
        }

        if (noise) {
            output = flux_generate_with_embeddings_and_noise(ctx, text_emb, text_seq,
                                                              noise, noise_size, &params);
            free(noise);
        } else {
            output = flux_generate_with_embeddings(ctx, text_emb, text_seq, &params);
        }
        free(text_emb);
    } else {
        /* Text-to-image mode with internal text encoder */
        if (verbose) {
            fprintf(stderr, "Generating...\n");
        }

        output = flux_generate(ctx, prompt, &params);
    }

    /* Finish progress display */
    if (verbose) {
        cli_finish_progress();
    }

    if (!output) {
        fprintf(stderr, "Error: Generation failed: %s\n", flux_get_error());
        flux_free(ctx);
        return 1;
    }

    double gen_time = (double)(clock() - start) / CLOCKS_PER_SEC;

    if (verbose) {
        fprintf(stderr, "Generated in %.2f seconds\n", gen_time);
        fprintf(stderr, "Output: %dx%d, %d channels\n",
                output->width, output->height, output->channels);
    }

    /* Save output */
    if (verbose) {
        fprintf(stderr, "Saving to %s...\n", output_path);
    }

    if (flux_image_save(output, output_path) != 0) {
        fprintf(stderr, "Error: Failed to save image: %s\n", output_path);
        flux_image_free(output);
        flux_free(ctx);
        return 1;
    }

    if (verbose) {
        fprintf(stderr, "Done!\n");
    } else {
        printf("%s\n", output_path);
    }

    /* Cleanup */
    flux_image_free(output);
    flux_free(ctx);

    return 0;
}
