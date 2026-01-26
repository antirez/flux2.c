/*
 * Qwen3 Text Generation
 *
 * Implements autoregressive text generation with KV caching for Qwen3-4B.
 * Uses the same weights as the FLUX text encoder.
 */

#ifndef QWEN3_GENERATE_H
#define QWEN3_GENERATE_H

#include "flux_qwen3.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================
 * Generation Parameters
 * ======================================================================== */

typedef struct {
    float temperature;    /* Sampling temperature (1.0 = neutral) */
    int top_k;            /* Top-k sampling (0 = disabled) */
    float top_p;          /* Top-p/nucleus sampling (1.0 = disabled) */
    float repeat_penalty; /* Repetition penalty (1.0 = disabled) */
    int max_tokens;       /* Maximum tokens to generate */
    int use_chat;         /* Use chat template (1) or raw mode (0) */
} qwen3_gen_params_t;

#define QWEN3_GEN_PARAMS_DEFAULT { \
    .temperature = 0.7f, \
    .top_k = 40, \
    .top_p = 0.9f, \
    .repeat_penalty = 1.1f, \
    .max_tokens = 0, /* 0 = unlimited (until EOS or context full) */ \
    .use_chat = 1 \
}

/* ========================================================================
 * Generator Context (includes KV cache)
 * ======================================================================== */

typedef struct qwen3_generator qwen3_generator_t;

/*
 * Create a generator context from a model directory.
 * Loads the model weights and allocates KV cache.
 */
qwen3_generator_t *qwen3_generator_create(const char *model_dir);

/*
 * Free generator and all associated resources.
 */
void qwen3_generator_free(qwen3_generator_t *gen);

/*
 * Reset the KV cache (call between separate conversations).
 */
void qwen3_generator_reset(qwen3_generator_t *gen);

/* ========================================================================
 * Generation API
 * ======================================================================== */

/*
 * Generate text from a prompt.
 * Returns newly allocated string (caller must free).
 *
 * If params is NULL, uses QWEN3_GEN_PARAMS_DEFAULT.
 *
 * The callback (if not NULL) is called for each generated token with:
 *   - token: the token string
 *   - user_data: passed through from caller
 * Return 0 from callback to continue, non-zero to stop generation.
 */
typedef int (*qwen3_token_callback)(const char *token, void *user_data);

char *qwen3_generate(qwen3_generator_t *gen,
                     const char *prompt,
                     const qwen3_gen_params_t *params,
                     qwen3_token_callback callback,
                     void *user_data);

/*
 * Continue generation from current state (for multi-turn conversations).
 * The KV cache is preserved from previous generation.
 */
char *qwen3_generate_continue(qwen3_generator_t *gen,
                              const char *prompt,
                              const qwen3_gen_params_t *params,
                              qwen3_token_callback callback,
                              void *user_data);

/* ========================================================================
 * Low-level API (for custom generation loops)
 * ======================================================================== */

/*
 * Process a single token and return logits.
 * Updates KV cache internally.
 * Returns pointer to logits array [vocab_size] (valid until next call).
 */
float *qwen3_forward_token(qwen3_generator_t *gen, int token_id);

/*
 * Sample next token from logits.
 */
int qwen3_sample(const float *logits, int vocab_size,
                 const qwen3_gen_params_t *params,
                 const int *past_tokens, int num_past);

/*
 * Decode token ID to string.
 * Returns pointer to internal buffer (valid until next call).
 */
const char *qwen3_decode_token(qwen3_generator_t *gen, int token_id);

/*
 * Get special token IDs.
 */
int qwen3_eos_token_id(void);
int qwen3_bos_token_id(void);

#ifdef __cplusplus
}
#endif

#endif /* QWEN3_GENERATE_H */
