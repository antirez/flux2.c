/*
 * Qwen3 Text Encoder for FLUX.2-klein
 *
 * Implements Qwen3-4B (2560 hidden dim variant) text encoder that produces
 * embeddings for FLUX image generation.
 */

#ifndef FLUX_QWEN3_H
#define FLUX_QWEN3_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================
 * Architecture Constants (from text_encoder/config.json)
 * ======================================================================== */

#define QWEN3_HIDDEN_SIZE      2560
#define QWEN3_INTERMEDIATE_SIZE 9728
#define QWEN3_NUM_HEADS        32
#define QWEN3_NUM_KV_HEADS     8
#define QWEN3_HEAD_DIM         128
#define QWEN3_NUM_LAYERS       36
#define QWEN3_VOCAB_SIZE       151936
#define QWEN3_MAX_SEQ_LEN      512
#define QWEN3_RMS_NORM_EPS     1e-6f
#define QWEN3_ROPE_THETA       1000000.0f

/* Output layers to extract (0-indexed)
 * Python uses hidden_states[9,18,27] which are outputs AFTER layers 8,17,26
 * since hidden_states[0] is embedding and hidden_states[i] is output after layer i-1 */
#define QWEN3_OUTPUT_LAYER_1   8
#define QWEN3_OUTPUT_LAYER_2   17
#define QWEN3_OUTPUT_LAYER_3   26

/* Final text embedding dimension: 3 × hidden_size = 7680 */
#define QWEN3_TEXT_DIM         (3 * QWEN3_HIDDEN_SIZE)

/* ========================================================================
 * Opaque Types
 * ======================================================================== */

typedef struct qwen3_model qwen3_model_t;
typedef struct qwen3_tokenizer qwen3_tokenizer_t;

/* ========================================================================
 * Tokenizer API
 * ======================================================================== */

/*
 * Load tokenizer from HuggingFace tokenizer.json file.
 */
qwen3_tokenizer_t *qwen3_tokenizer_load(const char *tokenizer_json_path);

/*
 * Free tokenizer resources.
 */
void qwen3_tokenizer_free(qwen3_tokenizer_t *tok);

/*
 * Tokenize text (raw, no template).
 */
int *qwen3_tokenize(qwen3_tokenizer_t *tok, const char *text,
                    int *num_tokens, int max_len);

/*
 * Tokenize text with chat template (first turn).
 * Format: <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
 */
int *qwen3_tokenize_chat(qwen3_tokenizer_t *tok, const char *prompt,
                         int *num_tokens, int max_len);

/*
 * Tokenize text for multi-turn continuation.
 * Closes previous assistant turn and adds new user turn.
 * Format: <|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
 */
int *qwen3_tokenize_chat_continue(qwen3_tokenizer_t *tok, const char *prompt,
                                   int *num_tokens, int max_len);

/*
 * Detokenize array of token IDs to string.
 * Caller must free the returned string.
 */
char *qwen3_detokenize(qwen3_tokenizer_t *tok, const int *tokens, int num_tokens);

/*
 * Detokenize a single token ID.
 * Returns pointer to internal buffer (valid until next call).
 */
const char *qwen3_detokenize_single(qwen3_tokenizer_t *tok, int token_id);

/*
 * Pad tokens to max_len with PAD token.
 * Returns new array, caller must free original.
 */
int *qwen3_pad_tokens(int *tokens, int num_tokens, int max_len, int *attention_mask);

/*
 * Get token string by ID.
 */
const char *qwen3_get_token(qwen3_tokenizer_t *tok, int id);

/*
 * Get token ID by string.
 */
int qwen3_get_id(qwen3_tokenizer_t *tok, const char *token);

/* ========================================================================
 * Model API
 * ======================================================================== */

/*
 * Load Qwen3 model weights from HuggingFace model directory.
 * Directory should contain model-00001-of-00002.safetensors, etc.
 */
qwen3_model_t *qwen3_model_load(const char *model_dir);

/*
 * Free model resources.
 */
void qwen3_model_free(qwen3_model_t *model);

/*
 * Run forward pass to generate text embeddings.
 *
 * input_ids: Token IDs [seq_len]
 * attention_mask: Attention mask [seq_len] (1 for real tokens, 0 for padding)
 * seq_len: Length of input sequences
 *
 * Returns: Embedding array [seq_len, 7680] (caller must free)
 * Extracts hidden states from layers 9, 18, 27 and concatenates them.
 */
float *qwen3_forward(qwen3_model_t *model,
                     const int *input_ids,
                     const int *attention_mask,
                     int seq_len);

/* ========================================================================
 * Combined Text Encoder API
 * ======================================================================== */

typedef struct qwen3_encoder {
    qwen3_tokenizer_t *tokenizer;
    qwen3_model_t *model;
} qwen3_encoder_t;

/*
 * Load complete text encoder (tokenizer + model).
 * model_dir should contain both tokenizer/ and text_encoder/ subdirectories.
 * use_mmap: if true, use memory-mapped bf16 weights (saves ~8GB, slower inference)
 */
qwen3_encoder_t *qwen3_encoder_load(const char *model_dir, int use_mmap);

/*
 * Free encoder resources.
 */
void qwen3_encoder_free(qwen3_encoder_t *enc);

/*
 * Encode text prompt to embeddings.
 * Returns: Embedding array [512, 7680] (caller must free)
 */
float *qwen3_encode_text(qwen3_encoder_t *enc, const char *prompt);

#ifdef __cplusplus
}
#endif

#endif /* FLUX_QWEN3_H */
