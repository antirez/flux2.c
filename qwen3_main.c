/*
 * Qwen3 Text Generation CLI
 *
 * Interactive chat and text completion with Qwen3-4B.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <signal.h>

#include "qwen3_generate.h"
#include "linenoise.h"

/* ========================================================================
 * Globals
 * ======================================================================== */

static qwen3_generator_t *g_gen = NULL;
static qwen3_gen_params_t g_params = QWEN3_GEN_PARAMS_DEFAULT;
static int g_verbose = 0;
static volatile sig_atomic_t g_interrupted = 0;

/* ========================================================================
 * Signal Handler
 * ======================================================================== */

static void sigint_handler(int sig) {
    (void)sig;
    g_interrupted = 1;
}

/* ========================================================================
 * Token Callback (streaming output)
 * ======================================================================== */

static int print_token(const char *token, void *user_data) {
    (void)user_data;
    if (g_interrupted) return 1;  /* Stop generation */
    printf("%s", token);
    fflush(stdout);
    return 0;  /* Continue */
}

/* ========================================================================
 * Command Processing
 * ======================================================================== */

static void print_help(void) {
    printf("\nQwen3-4B Text Generation\n");
    printf("========================\n\n");
    printf("Commands:\n");
    printf("  !help             Show this help\n");
    printf("  !temp <value>     Set temperature (current: %.2f)\n", g_params.temperature);
    printf("  !topk <value>     Set top-k (current: %d, 0=disabled)\n", g_params.top_k);
    printf("  !topp <value>     Set top-p (current: %.2f)\n", g_params.top_p);
    printf("  !repeat <value>   Set repeat penalty (current: %.2f)\n", g_params.repeat_penalty);
    printf("  !max <value>      Set max tokens (current: %d)\n", g_params.max_tokens);
    printf("  !chat             Toggle chat mode (current: %s)\n", g_params.use_chat ? "on" : "off");
    printf("  !reset            Reset conversation (clear KV cache)\n");
    printf("  !params           Show current parameters\n");
    printf("  !quit             Exit\n");
    printf("\nType any text to generate a response.\n\n");
}

static void print_params(void) {
    printf("\nCurrent parameters:\n");
    printf("  temperature:    %.2f\n", g_params.temperature);
    printf("  top_k:          %d\n", g_params.top_k);
    printf("  top_p:          %.2f\n", g_params.top_p);
    printf("  repeat_penalty: %.2f\n", g_params.repeat_penalty);
    printf("  max_tokens:     %d\n", g_params.max_tokens);
    printf("  chat_mode:      %s\n", g_params.use_chat ? "on" : "off");
    printf("\n");
}

static int process_command(const char *cmd) {
    if (strncmp(cmd, "!help", 5) == 0) {
        print_help();
    } else if (strncmp(cmd, "!temp ", 6) == 0) {
        float val = atof(cmd + 6);
        if (val > 0.0f && val < 10.0f) {
            g_params.temperature = val;
            printf("Temperature set to %.2f\n", val);
        } else {
            printf("Invalid temperature (use 0.01-10.0)\n");
        }
    } else if (strncmp(cmd, "!topk ", 6) == 0) {
        int val = atoi(cmd + 6);
        if (val >= 0) {
            g_params.top_k = val;
            printf("Top-k set to %d\n", val);
        }
    } else if (strncmp(cmd, "!topp ", 6) == 0) {
        float val = atof(cmd + 6);
        if (val > 0.0f && val <= 1.0f) {
            g_params.top_p = val;
            printf("Top-p set to %.2f\n", val);
        } else {
            printf("Invalid top-p (use 0.01-1.0)\n");
        }
    } else if (strncmp(cmd, "!repeat ", 8) == 0) {
        float val = atof(cmd + 8);
        if (val >= 1.0f && val < 10.0f) {
            g_params.repeat_penalty = val;
            printf("Repeat penalty set to %.2f\n", val);
        } else {
            printf("Invalid repeat penalty (use 1.0-10.0)\n");
        }
    } else if (strncmp(cmd, "!max ", 5) == 0) {
        int val = atoi(cmd + 5);
        if (val > 0 && val <= 4096) {
            g_params.max_tokens = val;
            printf("Max tokens set to %d\n", val);
        } else {
            printf("Invalid max tokens (use 1-4096)\n");
        }
    } else if (strncmp(cmd, "!chat", 5) == 0) {
        g_params.use_chat = !g_params.use_chat;
        printf("Chat mode %s\n", g_params.use_chat ? "enabled" : "disabled");
    } else if (strncmp(cmd, "!reset", 6) == 0) {
        qwen3_generator_reset(g_gen);
        printf("Conversation reset.\n");
    } else if (strncmp(cmd, "!params", 7) == 0) {
        print_params();
    } else if (strncmp(cmd, "!quit", 5) == 0 || strncmp(cmd, "!exit", 5) == 0) {
        return 1;  /* Signal exit */
    } else {
        printf("Unknown command. Type !help for help.\n");
    }
    return 0;
}

/* ========================================================================
 * Interactive Mode
 * ======================================================================== */

static void run_interactive(void) {
    printf("\nQwen3-4B Interactive Mode\n");
    printf("Type !help for commands, or enter a prompt to generate text.\n");
    printf("Press Ctrl+C to interrupt generation.\n\n");

    linenoiseHistorySetMaxLen(100);

    /* Install signal handler for Ctrl+C */
    struct sigaction sa, sa_old;
    sa.sa_handler = sigint_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGINT, &sa, &sa_old);

    char *line;
    while ((line = linenoise("qwen3> ")) != NULL) {
        /* Skip empty lines */
        if (*line == '\0') {
            free(line);
            continue;
        }

        linenoiseHistoryAdd(line);

        /* Check for commands */
        if (line[0] == '!') {
            if (process_command(line)) {
                free(line);
                break;
            }
            free(line);
            continue;
        }

        /* Generate response */
        printf("\n");
        g_interrupted = 0;  /* Reset interrupt flag */

        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        char *response = qwen3_generate(g_gen, line, &g_params, print_token, NULL);

        clock_gettime(CLOCK_MONOTONIC, &end);
        double elapsed = (end.tv_sec - start.tv_sec) +
                        (end.tv_nsec - start.tv_nsec) / 1e9;

        if (g_interrupted) {
            printf("\n[interrupted]\n\n");
        } else {
            printf("\n\n");
            if (g_verbose && response) {
                int num_tokens = strlen(response) / 4;  /* Rough estimate */
                printf("[%.1fs, ~%.1f tok/s]\n\n", elapsed, num_tokens / elapsed);
            }
        }

        free(response);
        free(line);
    }

    /* Restore original signal handler */
    sigaction(SIGINT, &sa_old, NULL);

    printf("Goodbye.\n");
}

/* ========================================================================
 * One-shot Mode
 * ======================================================================== */

static void run_oneshot(const char *prompt) {
    /* Install signal handler for Ctrl+C */
    struct sigaction sa, sa_old;
    sa.sa_handler = sigint_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGINT, &sa, &sa_old);

    g_interrupted = 0;

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    char *response = qwen3_generate(g_gen, prompt, &g_params, print_token, NULL);

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) +
                    (end.tv_nsec - start.tv_nsec) / 1e9;

    if (g_interrupted) {
        printf("\n[interrupted]\n");
    } else {
        printf("\n");
        if (g_verbose) {
            fprintf(stderr, "\n[Generated in %.1fs]\n", elapsed);
        }
    }

    /* Restore original signal handler */
    sigaction(SIGINT, &sa_old, NULL);

    free(response);
}

/* ========================================================================
 * Main
 * ======================================================================== */

static void usage(const char *prog) {
    fprintf(stderr, "Qwen3-4B Text Generation\n\n");
    fprintf(stderr, "Usage: %s -d <model_dir> [options]\n\n", prog);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -d <dir>      Model directory (required)\n");
    fprintf(stderr, "  -p <prompt>   Generate from prompt (non-interactive)\n");
    fprintf(stderr, "  -t <temp>     Temperature (default: 0.7)\n");
    fprintf(stderr, "  -k <topk>     Top-k sampling (default: 40, 0=disabled)\n");
    fprintf(stderr, "  -n <tokens>   Max tokens to generate (default: 0=unlimited)\n");
    fprintf(stderr, "  -r            Raw mode (no chat template)\n");
    fprintf(stderr, "  -v            Verbose output\n");
    fprintf(stderr, "  -h            Show this help\n");
    fprintf(stderr, "\nExamples:\n");
    fprintf(stderr, "  %s -d ~/model -p \"What is the capital of France?\"\n", prog);
    fprintf(stderr, "  %s -d ~/model  # Interactive chat mode\n", prog);
}

int main(int argc, char **argv) {
    const char *model_dir = NULL;
    const char *prompt = NULL;

    /* Parse arguments */
    int opt;
    while ((opt = getopt(argc, argv, "d:p:t:k:n:rvh")) != -1) {
        switch (opt) {
            case 'd':
                model_dir = optarg;
                break;
            case 'p':
                prompt = optarg;
                break;
            case 't':
                g_params.temperature = atof(optarg);
                break;
            case 'k':
                g_params.top_k = atoi(optarg);
                break;
            case 'n':
                g_params.max_tokens = atoi(optarg);
                break;
            case 'r':
                g_params.use_chat = 0;
                break;
            case 'v':
                g_verbose = 1;
                break;
            case 'h':
            default:
                usage(argv[0]);
                return opt == 'h' ? 0 : 1;
        }
    }

    if (!model_dir) {
        fprintf(stderr, "Error: Model directory required (-d)\n\n");
        usage(argv[0]);
        return 1;
    }

    /* Seed random number generator */
    srand(time(NULL));

    /* Load model */
    fprintf(stderr, "Loading Qwen3-4B from %s...\n", model_dir);
    g_gen = qwen3_generator_create(model_dir);
    if (!g_gen) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    fprintf(stderr, "Model loaded.\n\n");

    /* Run */
    if (prompt) {
        run_oneshot(prompt);
    } else {
        run_interactive();
    }

    /* Cleanup */
    qwen3_generator_free(g_gen);
    return 0;
}
