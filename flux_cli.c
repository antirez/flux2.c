/*
 * FLUX Interactive CLI Mode
 *
 * A REPL-style interface for image generation. Type prompts to generate
 * images, use bang commands (!help, !save, etc.) for control.
 *
 * Usage: flux -d model/  (without -p starts interactive mode)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include <unistd.h>
#include <sys/stat.h>
#include <errno.h>

#include "flux.h"
#include "flux_kernels.h"  /* For flux_step_callback */
#include "flux_qwen3.h"  /* For QWEN3_MAX_SEQ_LEN, QWEN3_TEXT_DIM */
#include "embcache.h"
#include "linenoise.h"
#include "terminals.h"

/* ======================================================================
 * Constants
 * ====================================================================== */

#define CLI_HISTORY_FILE ".flux_history"
#define CLI_MAX_PATH 4096
#define CLI_DEFAULT_WIDTH 256
#define CLI_DEFAULT_HEIGHT 256
#define CLI_DEFAULT_STEPS 4
#define CLI_REFS_INITIAL 16   /* Initial capacity for $N reference cache */
#define CLI_MAX_PROMPT_REFS 16  /* Max references in a single prompt */

/* Terminal graphics protocol (detected at startup) */
static term_graphics_proto cli_term_proto = TERM_PROTO_NONE;

/* ======================================================================
 * Session State
 * ====================================================================== */

/* Reference tracking for $N syntax */
typedef struct {
    char *path;                  /* Path to image file (dynamically allocated) */
} cli_ref;

typedef struct {
    flux_ctx *ctx;
    char model_dir[CLI_MAX_PATH];
    char tmpdir[CLI_MAX_PATH];
    char last_image[CLI_MAX_PATH];
    int width;
    int height;
    int steps;
    int64_t seed;
    int image_count;
    int show_enabled;
    int open_enabled;
    /* Reference tracking (dynamic array, never forgets) */
    cli_ref *refs;               /* Dynamic array of references */
    int refs_capacity;           /* Current allocated capacity */
    int refs_count;              /* Number of references (next_id - 1) */
} cli_state;

static cli_state state;

/* ======================================================================
 * Reference Management ($N syntax)
 * ====================================================================== */

/* Register a new image and return its $N id (starts from 1 to match image-0001.png) */
static int ref_add(const char *path) {
    /* Grow array if needed */
    if (state.refs_count >= state.refs_capacity) {
        int new_cap = state.refs_capacity ? state.refs_capacity * 2 : CLI_REFS_INITIAL;
        cli_ref *new_refs = realloc(state.refs, new_cap * sizeof(cli_ref));
        if (!new_refs) {
            fprintf(stderr, "Error: Out of memory for references\n");
            return -1;
        }
        state.refs = new_refs;
        state.refs_capacity = new_cap;
    }

    int id = state.refs_count + 1;  /* IDs are 1-based */
    state.refs[state.refs_count].path = strdup(path);
    if (!state.refs[state.refs_count].path) {
        fprintf(stderr, "Error: Out of memory for path\n");
        return -1;
    }
    state.refs_count++;
    return id;
}

/* Lookup a reference by $N id. Returns path or NULL if not found. */
static const char *ref_lookup(int id) {
    if (id < 1 || id > state.refs_count) return NULL;
    return state.refs[id - 1].path;  /* Direct access, IDs are 1-based */
}

/* ======================================================================
 * Utility Functions
 * ====================================================================== */

/* Skip leading whitespace. */
static char *skip_spaces(char *s) {
    while (*s && isspace((unsigned char)*s)) s++;
    return s;
}

/* Trim trailing whitespace in place. */
static void trim_trailing(char *s) {
    int len = strlen(s);
    while (len > 0 && isspace((unsigned char)s[len-1])) {
        s[--len] = '\0';
    }
}

/* Case-insensitive prefix match. */
static int starts_with_ci(const char *s, const char *prefix) {
    while (*prefix) {
        if (tolower((unsigned char)*s) != tolower((unsigned char)*prefix))
            return 0;
        s++;
        prefix++;
    }
    return 1;
}

/* Parse "WxH" or "WXH" from string, return pointer past it or NULL. */
static char *parse_size(const char *s, int *w, int *h) {
    char *end;
    long lw = strtol(s, &end, 10);
    if (end == s || (*end != 'x' && *end != 'X')) return NULL;
    char *mid = end + 1;
    long lh = strtol(mid, &end, 10);
    if (end == mid) return NULL;
    if (lw < 64 || lw > 1792 || lh < 64 || lh > 1792) return NULL;
    *w = (int)lw;
    *h = (int)lh;
    return end;
}

/* Extract size from prompt (beginning or end). Returns prompt without size.
 * Caller must free the returned string. Returns NULL if no size found. */
static char *extract_size_from_prompt(const char *prompt, int *w, int *h) {
    char *result = NULL;
    const char *p = prompt;

    /* Try beginning */
    p = skip_spaces((char *)prompt);
    char *after = parse_size(p, w, h);
    if (after) {
        after = skip_spaces(after);
        if (*after) {
            result = strdup(after);
            return result;
        }
    }

    /* Try end - scan backwards for digits */
    int len = strlen(prompt);
    int i = len - 1;

    /* Skip trailing spaces */
    while (i >= 0 && isspace((unsigned char)prompt[i])) i--;
    if (i < 0) return NULL;

    /* Find start of potential size pattern (digits) */
    while (i >= 0 && isdigit((unsigned char)prompt[i])) i--;
    if (i < 0 || (prompt[i] != 'x' && prompt[i] != 'X')) return NULL;
    int x_pos = i;
    i--;
    while (i >= 0 && isdigit((unsigned char)prompt[i])) i--;

    /* Must have space or start before the size */
    if (i >= 0 && !isspace((unsigned char)prompt[i])) return NULL;

    int size_start = i + 1;
    if (size_start >= x_pos) return NULL;  /* No width digits */

    /* Parse the size we found */
    if (!parse_size(prompt + size_start, w, h)) return NULL;

    /* Build result without the size */
    result = malloc(len + 1);
    if (!result) return NULL;

    /* Copy up to size_start, trim trailing space */
    int copy_len = size_start;
    while (copy_len > 0 && isspace((unsigned char)prompt[copy_len - 1])) copy_len--;
    memcpy(result, prompt, copy_len);
    result[copy_len] = '\0';

    return result;
}

/* ======================================================================
 * Temp Directory Management
 * ====================================================================== */

static int create_tmpdir(void) {
    snprintf(state.tmpdir, sizeof(state.tmpdir), "/tmp/flux-XXXXXX");
    if (mkdtemp(state.tmpdir) == NULL) {
        fprintf(stderr, "Error: Cannot create temp directory: %s\n",
                strerror(errno));
        return -1;
    }
    return 0;
}

static void get_image_path(char *buf, size_t size) {
    state.image_count++;
    snprintf(buf, size, "%s/image-%04d.png", state.tmpdir, state.image_count);
}

/* ======================================================================
 * Image Display
 * ====================================================================== */

static void display_image(const char *path) {
    if (state.show_enabled) {
        flux_image *img = flux_image_load(path);
        if (img) {
            terminal_display_image(img, cli_term_proto);
            flux_image_free(img);
        }
    }
    if (state.open_enabled) {
#ifdef __APPLE__
        char cmd[CLI_MAX_PATH + 16];
        snprintf(cmd, sizeof(cmd), "open \"%s\" 2>/dev/null &", path);
        system(cmd);
#endif
    }
}

/* ======================================================================
 * Generation
 * ====================================================================== */

static int cli_current_step = 0;

static void cli_step_callback(int step, int total) {
    cli_current_step = step;
    int percent = (step * 100) / total;
    terminal_progress_set(percent);
}

static int generate_image(const char *prompt, const char *ref_image,
                          int explicit_width, int explicit_height) {
    flux_params params = FLUX_PARAMS_DEFAULT;
    params.num_steps = state.steps;

    /* Determine seed */
    int64_t actual_seed;
    if (state.seed >= 0) {
        actual_seed = state.seed;
    } else {
        actual_seed = (int64_t)time(NULL) ^ (int64_t)rand();
    }
    params.seed = actual_seed;
    printf("Seed: %lld\n", (long long)actual_seed);

    /* Setup progress callback */
    cli_current_step = 0;
    flux_step_callback = cli_step_callback;
    terminal_progress_indeterminate();

    /* Start timing */
    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    /* Generate */
    flux_image *img;
    if (ref_image) {
        flux_image *ref = flux_image_load(ref_image);
        if (!ref) {
            fprintf(stderr, "Error: Cannot load '%s'\n", ref_image);
            return -1;
        }
        /* Use explicit size if provided, otherwise use reference image dimensions */
        if (explicit_width > 0 && explicit_height > 0) {
            params.width = explicit_width;
            params.height = explicit_height;
        } else {
            params.width = ref->width;
            params.height = ref->height;
        }
        printf("Generating %dx%d (img2img)...\n", params.width, params.height);
        img = flux_img2img(state.ctx, prompt, ref, &params);
        flux_image_free(ref);
    } else {
        /* Text-to-image: use explicit size if provided, otherwise state defaults */
        if (explicit_width > 0 && explicit_height > 0) {
            params.width = explicit_width;
            params.height = explicit_height;
        } else {
            params.width = state.width;
            params.height = state.height;
        }
        printf("Generating %dx%d...\n", params.width, params.height);

        /* Try to use cached embedding */
        float *embeddings = emb_cache_lookup(prompt);
        if (embeddings) {
            printf("(using cached embedding)\n");
            img = flux_generate_with_embeddings(state.ctx, embeddings,
                                                 QWEN3_MAX_SEQ_LEN, &params);
            free(embeddings);
        } else {
            /* Encode and cache for next time */
            int seq_len;
            embeddings = flux_encode_text(state.ctx, prompt, &seq_len);
            if (embeddings) {
                /* Cache the embedding (4-bit quantized) */
                emb_cache_store(prompt, embeddings,
                                seq_len * QWEN3_TEXT_DIM);

                /* Release text encoder to free memory before generation */
                flux_release_text_encoder(state.ctx);

                img = flux_generate_with_embeddings(state.ctx, embeddings,
                                                     seq_len, &params);
                free(embeddings);
            } else {
                img = NULL;
            }
        }
    }

    /* End timing */
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    double elapsed = (end_time.tv_sec - start_time.tv_sec) +
                     (end_time.tv_nsec - start_time.tv_nsec) / 1e9;

    if (!img) {
        fprintf(stderr, "Error: Generation failed: %s\n", flux_get_error());
        flux_step_callback = NULL;
        terminal_progress_remove();
        return -1;
    }

    /* Save to temp */
    char path[CLI_MAX_PATH];
    get_image_path(path, sizeof(path));
    flux_image_save_with_seed(img, path, actual_seed);
    flux_image_free(img);

    /* Update last image and register as reference */
    strncpy(state.last_image, path, sizeof(state.last_image) - 1);
    int ref_id = ref_add(path);

    /* Cleanup progress */
    flux_step_callback = NULL;
    terminal_progress_remove();

    printf("Done -> %s (ref $%d) [%.2fs]\n", path, ref_id, elapsed);
    display_image(path);

    return 0;
}

static int generate_multiref(const char *prompt, const char **ref_paths, int num_refs,
                             int explicit_width, int explicit_height) {
    flux_params params = FLUX_PARAMS_DEFAULT;
    params.num_steps = state.steps;

    /* Determine seed */
    int64_t actual_seed;
    if (state.seed >= 0) {
        actual_seed = state.seed;
    } else {
        actual_seed = (int64_t)time(NULL) ^ (int64_t)rand();
    }
    params.seed = actual_seed;
    printf("Seed: %lld\n", (long long)actual_seed);

    /* Setup progress callback */
    cli_current_step = 0;
    flux_step_callback = cli_step_callback;
    terminal_progress_indeterminate();

    /* Start timing */
    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    /* Load reference images */
    flux_image **refs = (flux_image **)malloc(num_refs * sizeof(flux_image *));
    for (int i = 0; i < num_refs; i++) {
        refs[i] = flux_image_load(ref_paths[i]);
        if (!refs[i]) {
            fprintf(stderr, "Error: Cannot load '%s'\n", ref_paths[i]);
            for (int j = 0; j < i; j++) flux_image_free(refs[j]);
            free(refs);
            return -1;
        }
    }

    /* Use explicit size if provided, otherwise default to first reference image */
    if (explicit_width > 0 && explicit_height > 0) {
        params.width = explicit_width;
        params.height = explicit_height;
    } else {
        params.width = refs[0]->width;
        params.height = refs[0]->height;
    }

    printf("Generating %dx%d (multi-ref, %d images)...\n",
           params.width, params.height, num_refs);

    flux_image *img = flux_multiref(state.ctx, prompt,
                                     (const flux_image **)refs, num_refs, &params);

    /* Free reference images */
    for (int i = 0; i < num_refs; i++) {
        flux_image_free(refs[i]);
    }
    free(refs);

    /* End timing */
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    double elapsed = (end_time.tv_sec - start_time.tv_sec) +
                     (end_time.tv_nsec - start_time.tv_nsec) / 1e9;

    if (!img) {
        fprintf(stderr, "Error: Generation failed: %s\n", flux_get_error());
        flux_step_callback = NULL;
        terminal_progress_remove();
        return -1;
    }

    /* Save to temp */
    char path[CLI_MAX_PATH];
    get_image_path(path, sizeof(path));
    flux_image_save_with_seed(img, path, actual_seed);
    flux_image_free(img);

    /* Update last image and register as reference */
    strncpy(state.last_image, path, sizeof(state.last_image) - 1);
    int ref_id = ref_add(path);

    /* Cleanup progress */
    flux_step_callback = NULL;
    terminal_progress_remove();

    printf("Done -> %s (ref $%d) [%.2fs]\n", path, ref_id, elapsed);
    display_image(path);

    return 0;
}

/* ======================================================================
 * Bang Commands
 * ====================================================================== */

static void cmd_help(void) {
    printf("\n");
    printf("Commands:\n");
    printf("  !help                 Show this help\n");
    printf("  !save [filename]      Save last image to file\n");
    printf("  !load <filename>      Load image for img2img\n");
    printf("  !seed <n>             Set seed (-1 for random)\n");
    printf("  !size <W>x<H>         Set default size\n");
    printf("  !steps <n>            Set sampling steps\n");
    printf("  !explore <n> <prompt> Generate n thumbnail variations\n");
    printf("  !show                 Toggle terminal display\n");
    printf("  !open                 Toggle auto-open (macOS)\n");
    printf("  !quit                 Exit\n");
    printf("\n");
    printf("Prompt syntax:\n");
    printf("  <prompt>              Generate image from prompt\n");
    printf("  512x512 <prompt>      Set size inline\n");
    printf("  $ <prompt>            Img2img using last/loaded image\n");
    printf("  $N <prompt>           Img2img using reference $N\n");
    printf("  $1 $3 <prompt>        Multi-reference (combine images)\n");
    printf("\n");
    printf("Each generated/loaded image gets a $N reference ID.\n");
    printf("\n");
}

static void cmd_save(char *arg) {
    arg = skip_spaces(arg);

    if (state.last_image[0] == '\0') {
        fprintf(stderr, "Error: No image to save.\n");
        return;
    }

    char dest[CLI_MAX_PATH];
    if (*arg) {
        strncpy(dest, arg, sizeof(dest) - 1);
        dest[sizeof(dest) - 1] = '\0';
    } else {
        /* Generate default name */
        time_t t = time(NULL);
        snprintf(dest, sizeof(dest), "flux_%ld.png", (long)t);
    }

    /* Copy file */
    flux_image *img = flux_image_load(state.last_image);
    if (!img) {
        fprintf(stderr, "Error: Cannot load last image.\n");
        return;
    }
    if (flux_image_save(img, dest) == 0) {
        printf("Saved: %s\n", dest);
    } else {
        fprintf(stderr, "Error: Cannot save to '%s'\n", dest);
    }
    flux_image_free(img);
}

static void cmd_load(char *arg) {
    arg = skip_spaces(arg);

    if (!*arg) {
        fprintf(stderr, "Usage: !load <filename>\n");
        return;
    }

    flux_image *img = flux_image_load(arg);
    if (!img) {
        fprintf(stderr, "Error: Cannot load '%s'\n", arg);
        return;
    }

    /* Save to temp so we have a consistent path */
    char path[CLI_MAX_PATH];
    get_image_path(path, sizeof(path));
    flux_image_save(img, path);

    /* Update state and register as reference */
    strncpy(state.last_image, path, sizeof(state.last_image) - 1);
    int ref_id = ref_add(path);
    state.width = img->width;
    state.height = img->height;

    printf("Loaded: %s (%dx%d, ref $%d)\n", arg, img->width, img->height, ref_id);

    /* Display in Kitty-capable terminal */
    if (state.show_enabled) {
        terminal_display_image(img, cli_term_proto);
    }

    flux_image_free(img);
}

static void cmd_seed(char *arg) {
    arg = skip_spaces(arg);

    if (!*arg) {
        if (state.seed < 0) {
            printf("Seed: random\n");
        } else {
            printf("Seed: %lld\n", (long long)state.seed);
        }
        return;
    }

    char *end;
    long long val = strtoll(arg, &end, 10);
    if (end == arg) {
        fprintf(stderr, "Error: Invalid seed.\n");
        return;
    }
    state.seed = (int64_t)val;
    if (state.seed < 0) {
        printf("Seed: random\n");
    } else {
        printf("Seed: %lld\n", (long long)state.seed);
    }
}

static void cmd_size(char *arg) {
    arg = skip_spaces(arg);

    if (!*arg) {
        printf("Size: %dx%d\n", state.width, state.height);
        return;
    }

    int w, h;
    if (!parse_size(arg, &w, &h)) {
        fprintf(stderr, "Error: Invalid size. Use WxH (e.g., 512x512).\n");
        return;
    }
    state.width = (w / 16) * 16;
    state.height = (h / 16) * 16;
    printf("Size: %dx%d\n", state.width, state.height);
}

static void cmd_steps(char *arg) {
    arg = skip_spaces(arg);

    if (!*arg) {
        printf("Steps: %d\n", state.steps);
        return;
    }

    int val = atoi(arg);
    if (val < 1 || val > 256) {
        fprintf(stderr, "Error: Steps must be 1-256.\n");
        return;
    }
    state.steps = val;
    printf("Steps: %d\n", state.steps);
}

static void cmd_explore(char *arg) {
    arg = skip_spaces(arg);

    if (!state.show_enabled) {
        fprintf(stderr, "Error: !explore requires display mode. "
                        "Use !show to enable.\n");
        return;
    }

    /* Parse count */
    char *end;
    int count = (int)strtol(arg, &end, 10);
    if (end == arg || count < 1 || count > 100) {
        fprintf(stderr, "Usage: !explore <count> <prompt>\n");
        return;
    }

    char *prompt = skip_spaces(end);
    if (!*prompt) {
        fprintf(stderr, "Usage: !explore <count> <prompt>\n");
        return;
    }

    /* Check for inline size (beginning or end of prompt) */
    char *prompt_to_free = NULL;
    int w, h;
    char *extracted = extract_size_from_prompt(prompt, &w, &h);
    if (extracted) {
        state.width = (w / 16) * 16;
        state.height = (h / 16) * 16;
        printf("Default size: %dx%d\n", state.width, state.height);
        prompt = extracted;
        prompt_to_free = extracted;
    }

    if (!*prompt) {
        fprintf(stderr, "Error: Empty prompt.\n");
        free(prompt_to_free);
        return;
    }

    printf("Generating %d images at %dx%d...\n",
           count, state.width, state.height);

    /* Try cached embedding first */
    int seq_len = QWEN3_MAX_SEQ_LEN;
    float *embeddings = emb_cache_lookup(prompt);

    if (embeddings) {
        printf("(using cached embedding)\n");
    } else {
        /* Encode and cache */
        embeddings = flux_encode_text(state.ctx, prompt, &seq_len);
        if (!embeddings) {
            fprintf(stderr, "Error: Failed to encode prompt.\n");
            free(prompt_to_free);
            return;
        }
        /* Cache for future use */
        emb_cache_store(prompt, embeddings, seq_len * QWEN3_TEXT_DIM);
        flux_release_text_encoder(state.ctx);
    }

    flux_params params = FLUX_PARAMS_DEFAULT;
    params.width = state.width;
    params.height = state.height;
    params.num_steps = state.steps;

    for (int i = 0; i < count; i++) {
        int64_t seed = (int64_t)time(NULL) ^ (int64_t)rand() ^ (int64_t)i;
        params.seed = seed;

        printf("  [%d/%d] Seed: %lld ", i + 1, count, (long long)seed);
        fflush(stdout);

        flux_image *img = flux_generate_with_embeddings(state.ctx, embeddings,
                                                         seq_len, &params);
        if (img) {
            terminal_display_image(img, cli_term_proto);
            flux_image_free(img);
            printf("\n");
        } else {
            printf("(failed)\n");
        }
    }

    free(embeddings);
    free(prompt_to_free);
    printf("Done. Use !seed <n> then re-run prompt at full size.\n");
}

static void cmd_show(void) {
    state.show_enabled = !state.show_enabled;
    printf("Display: %s\n", state.show_enabled ? "ON" : "OFF");
}

static void cmd_open(void) {
    state.open_enabled = !state.open_enabled;
    printf("Auto-open: %s\n", state.open_enabled ? "ON" : "OFF");
}

/* ======================================================================
 * Command Dispatch
 * ====================================================================== */

/* Process a bang command. Returns 1 if should quit, 0 otherwise. */
static int process_command(char *line) {
    char *cmd = line + 1;  /* Skip '!' */
    cmd = skip_spaces(cmd);

    if (starts_with_ci(cmd, "help")) {
        cmd_help();
    } else if (starts_with_ci(cmd, "save")) {
        cmd_save(cmd + 4);
    } else if (starts_with_ci(cmd, "load")) {
        cmd_load(cmd + 4);
    } else if (starts_with_ci(cmd, "seed")) {
        cmd_seed(cmd + 4);
    } else if (starts_with_ci(cmd, "size")) {
        cmd_size(cmd + 4);
    } else if (starts_with_ci(cmd, "steps")) {
        cmd_steps(cmd + 5);
    } else if (starts_with_ci(cmd, "explore")) {
        cmd_explore(cmd + 7);
    } else if (starts_with_ci(cmd, "show")) {
        cmd_show();
    } else if (starts_with_ci(cmd, "open")) {
        cmd_open();
    } else if (starts_with_ci(cmd, "quit") || starts_with_ci(cmd, "exit")) {
        return 1;
    } else {
        fprintf(stderr, "Unknown command. Type !help for help.\n");
    }
    return 0;
}

/* ======================================================================
 * Prompt Processing
 * ====================================================================== */

static int process_prompt(char *line) {
    line = skip_spaces(line);
    if (!*line) return 0;

    char *prompt_to_free = NULL;

    /* Parse $N references at the beginning */
    const char *ref_paths[CLI_MAX_PROMPT_REFS];
    int num_refs = 0;

    while (*line == '$' && num_refs < CLI_MAX_PROMPT_REFS) {
        line++;  /* skip $ */
        /* $N requires digit immediately after $ (no spaces) */
        if (isdigit((unsigned char)*line)) {
            /* $N - parse number and lookup reference */
            char *end;
            long id = strtol(line, &end, 10);
            const char *path = ref_lookup((int)id);
            if (!path) {
                fprintf(stderr, "Error: Reference $%ld not found.\n", id);
                return 0;
            }
            ref_paths[num_refs++] = path;
            line = end;
        } else {
            /* Just $ (followed by space or other) - use last image */
            if (state.last_image[0] == '\0') {
                fprintf(stderr, "Error: No image for img2img. "
                                "Generate or !load an image first.\n");
                return 0;
            }
            ref_paths[num_refs++] = state.last_image;
        }
        line = skip_spaces(line);
    }

    /* Check for inline size (beginning or end of prompt) */
    int explicit_w = 0, explicit_h = 0;
    char *extracted = extract_size_from_prompt(line, &explicit_w, &explicit_h);
    if (extracted) {
        /* Align to 16 pixels */
        explicit_w = (explicit_w / 16) * 16;
        explicit_h = (explicit_h / 16) * 16;
        printf("Size: %dx%d\n", explicit_w, explicit_h);
        line = extracted;
        prompt_to_free = extracted;
    }

    if (!*line) {
        /* If only size was provided, set it as new default */
        if (explicit_w > 0 && explicit_h > 0) {
            state.width = explicit_w;
            state.height = explicit_h;
            printf("Default size: %dx%d\n", state.width, state.height);
        } else {
            fprintf(stderr, "Error: Empty prompt.\n");
        }
        free(prompt_to_free);
        return 0;
    }

    /* Generate based on number of references */
    if (num_refs == 0) {
        generate_image(line, NULL, explicit_w, explicit_h);
    } else if (num_refs == 1) {
        generate_image(line, ref_paths[0], explicit_w, explicit_h);
    } else {
        /* Multi-reference generation */
        generate_multiref(line, ref_paths, num_refs, explicit_w, explicit_h);
    }

    free(prompt_to_free);
    return 0;
}

/* ======================================================================
 * Main REPL
 * ====================================================================== */

static void print_banner(void) {
    printf("\nFLUX.2 Interactive Mode\n");
    printf("=======================\n");
    printf("Model: %s\n", state.model_dir);

    if (state.show_enabled) {
        const char *proto_name = (cli_term_proto == TERM_PROTO_KITTY) ? "Kitty" :
                                 (cli_term_proto == TERM_PROTO_ITERM2) ? "iTerm2" : "unknown";
        printf("Display: %s graphics enabled\n", proto_name);
    } else {
        printf("Display: Images saved to %s/\n", state.tmpdir);
    }

    printf("\nType a prompt to generate an image. Use !help for commands.\n");
    printf("Default size: %dx%d | Steps: %d | Seed: %s\n\n",
           state.width, state.height, state.steps,
           state.seed < 0 ? "random" : "fixed");
}

int flux_cli_run(flux_ctx *ctx, const char *model_dir) {
    /* Initialize state */
    memset(&state, 0, sizeof(state));
    state.ctx = ctx;
    strncpy(state.model_dir, model_dir, sizeof(state.model_dir) - 1);
    state.width = CLI_DEFAULT_WIDTH;
    state.height = CLI_DEFAULT_HEIGHT;
    state.steps = CLI_DEFAULT_STEPS;
    state.seed = -1;

    /* Initialize embedding cache */
    emb_cache_init();

    /* Detect terminal graphics support */
    cli_term_proto = detect_terminal_graphics();
    state.show_enabled = (cli_term_proto != TERM_PROTO_NONE);

    /* Create temp directory */
    if (create_tmpdir() < 0) {
        return 1;
    }

    /* Setup history */
    char history_path[CLI_MAX_PATH];
    const char *home = getenv("HOME");
    if (home) {
        snprintf(history_path, sizeof(history_path), "%s/%s",
                 home, CLI_HISTORY_FILE);
        linenoiseHistoryLoad(history_path);
    }
    linenoiseHistorySetMaxLen(500);

    print_banner();

    /* Main loop */
    char *line;
    while ((line = linenoise("flux> ")) != NULL) {
        trim_trailing(line);

        if (*line) {
            linenoiseHistoryAdd(line);

            char *trimmed = skip_spaces(line);
            if (*trimmed == '!') {
                if (process_command(trimmed)) {
                    free(line);
                    break;
                }
            } else {
                process_prompt(trimmed);
            }
        }
        free(line);
    }

    /* Save history */
    if (home) {
        linenoiseHistorySave(history_path);
    }

    /* Cleanup embedding cache */
    emb_cache_free();

    printf("Goodbye.\n");
    return 0;
}
