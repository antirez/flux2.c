/*
 * Mistral-Small-3.2-24B Text Encoder for Iris
 *
 * Implements the Mistral text encoder for FLUX.2-dev image generation.
 * Architecture is very similar to Qwen3 (GQA, SwiGLU, RoPE, RMSNorm)
 * with key differences: no Q/K norm, different extraction layers (10/20/30),
 * and a pooled embedding output for the transformer's text_embedder.
 */

#ifndef IRIS_MISTRAL_H
#define IRIS_MISTRAL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================
 * Architecture Constants
 * ======================================================================== */

/* Fixed constants */
#define MISTRAL_MAX_SEQ_LEN     512
#define MISTRAL_RMS_NORM_EPS    1e-5f

/* Output layers to extract (0-indexed).
 * Hidden states from layers 10, 20, 30 are concatenated → [seq, 3*hidden]. */
#define MISTRAL_OUTPUT_LAYER_1  10
#define MISTRAL_OUTPUT_LAYER_2  20
#define MISTRAL_OUTPUT_LAYER_3  30

/* ========================================================================
 * Opaque Types
 * ======================================================================== */

typedef struct mistral_model mistral_model_t;
typedef struct mistral_tokenizer mistral_tokenizer_t;

/* ========================================================================
 * Tokenizer API
 * ======================================================================== */

/*
 * Load tokenizer from HuggingFace tokenizer.json file.
 */
mistral_tokenizer_t *mistral_tokenizer_load(const char *tokenizer_json_path);

/*
 * Free tokenizer resources.
 */
void mistral_tokenizer_free(mistral_tokenizer_t *tok);

/*
 * Tokenize text with Mistral chat template.
 * Format: [INST] {prompt} [/INST]
 */
int *mistral_tokenize_chat(mistral_tokenizer_t *tok, const char *prompt,
                           int *num_tokens, int max_len);

/*
 * Pad tokens to max_len with PAD token.
 * Returns new array, caller must free original.
 */
int *mistral_pad_tokens(int *tokens, int num_tokens, int max_len, int *attention_mask);

/*
 * Get token string by ID.
 */
const char *mistral_get_token(mistral_tokenizer_t *tok, int id);

/* ========================================================================
 * Model API
 * ======================================================================== */

/*
 * Load Mistral model in mmap mode from HuggingFace model directory.
 * Only loads embeddings + final norm; layer weights loaded on-demand.
 */
mistral_model_t *mistral_model_load_mmap(const char *model_dir);

/*
 * Free model resources.
 */
void mistral_model_free(mistral_model_t *model);

/*
 * Run forward pass to generate text embeddings.
 *
 * input_ids: Token IDs [seq_len]
 * attention_mask: Attention mask [seq_len] (1 for real tokens, 0 for padding)
 * seq_len: Length of input sequences
 *
 * Returns: Embedding array [seq_len, 3*hidden_size] (caller must free)
 * Extracts hidden states from layers 10, 20, 30 and concatenates them.
 */
float *mistral_forward(mistral_model_t *model,
                       const int *input_ids,
                       const int *attention_mask,
                       int seq_len);

/*
 * Get pooled embedding after forward pass (last real token hidden state).
 * Returns pointer to internal buffer [hidden_size]. Do NOT free.
 * Valid until next forward pass or model_free.
 */
const float *mistral_get_pooled(mistral_model_t *model);

/* ========================================================================
 * Combined Text Encoder API
 * ======================================================================== */

typedef struct mistral_encoder {
    mistral_tokenizer_t *tokenizer;
    mistral_model_t *model;
} mistral_encoder_t;

/*
 * Load complete text encoder (tokenizer + model).
 * model_dir should contain both tokenizer/ and text_encoder/ subdirectories.
 * Always uses mmap mode (Mistral is ~48GB, must stream layers).
 */
mistral_encoder_t *mistral_encoder_load(const char *model_dir);

/*
 * Free encoder resources.
 */
void mistral_encoder_free(mistral_encoder_t *enc);

/*
 * Encode text prompt to embeddings.
 * Returns: Embedding array [512, 3*hidden_size] (caller must free)
 * Also stores pooled embedding internally (retrieve via mistral_get_pooled on enc->model).
 * out_num_tokens: if non-NULL, receives the number of real (non-padding) tokens.
 */
float *mistral_encode_text(mistral_encoder_t *enc, const char *prompt,
                           int *out_num_tokens);

#ifdef __cplusplus
}
#endif

#endif /* IRIS_MISTRAL_H */
