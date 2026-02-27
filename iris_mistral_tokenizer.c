/*
 * Mistral-Small-3.2-24B Tokenizer Implementation
 *
 * BPE tokenizer for Mistral text encoder (Tekken/SentencePiece-based BPE).
 * Loads from HuggingFace tokenizer.json format. Core BPE algorithm is
 * identical to Qwen3, but uses different byte encoding, special tokens,
 * and chat template.
 *
 * Key differences from Qwen3 tokenizer:
 * - No GPT-2 byte-to-unicode encoding (Tekken works on raw UTF-8)
 * - Chat template: [INST] {prompt} [/INST] (not <|im_start|>)
 * - Special tokens discovered from tokenizer.json (not hardcoded IDs)
 * - Byte fallback tokens: <0xNN> for unknown bytes
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdint.h>
#include "iris_kernels.h"

/* ========================================================================
 * Configuration
 * ======================================================================== */

#define MISTRAL_MAX_TOKEN_LEN 256
#define MISTRAL_MAX_SEQ_LEN   512
#define MISTRAL_HASH_SIZE     300007  /* Prime > 2 * max vocab_size */

/* Default special token IDs (overridden from tokenizer.json if found) */
#define MISTRAL_DEFAULT_BOS_ID  1
#define MISTRAL_DEFAULT_EOS_ID  2
#define MISTRAL_DEFAULT_PAD_ID  0

/* ========================================================================
 * Data Structures
 * ======================================================================== */

typedef struct {
    char *token;
    int id;
} mistral_vocab_entry_t;

typedef struct {
    char *left;
    char *right;
    int rank;
} mistral_bpe_merge_t;

typedef struct mistral_tokenizer {
    /* Vocabulary: id -> token string */
    char **vocab;
    int vocab_size;

    /* Hash table: token string -> id */
    mistral_vocab_entry_t *vocab_hash;
    int hash_size;

    /* BPE merges */
    mistral_bpe_merge_t *merges;
    int num_merges;

    /* Merge rank lookup hash table */
    int *merge_ranks;

    /* Special token IDs (discovered from tokenizer.json) */
    int bos_id;
    int eos_id;
    int pad_id;
    int inst_start_id;  /* [INST] token, or -1 if not found */
    int inst_end_id;    /* [/INST] token, or -1 if not found */

    /* Whether to use GPT-2 byte encoding (0=raw UTF-8, 1=byte-to-unicode) */
    int use_byte_encoding;
} mistral_tokenizer_t;

/* ========================================================================
 * Byte-Level BPE Encoding Table (GPT-2 style, optional for some tokenizers)
 * ======================================================================== */

static int mistral_byte_to_unicode[256];
static int mistral_unicode_to_byte[512];
static int mistral_byte_encoder_initialized = 0;

static void mistral_init_byte_encoder(void) {
    if (mistral_byte_encoder_initialized) return;

    for (int i = 33; i <= 126; i++) {
        mistral_byte_to_unicode[i] = i;
        mistral_unicode_to_byte[i] = i;
    }
    for (int i = 161; i <= 172; i++) {
        mistral_byte_to_unicode[i] = i;
        mistral_unicode_to_byte[i] = i;
    }
    for (int i = 174; i <= 255; i++) {
        mistral_byte_to_unicode[i] = i;
        mistral_unicode_to_byte[i] = i;
    }

    int offset = 256;
    for (int i = 0; i < 256; i++) {
        if (mistral_byte_to_unicode[i] == 0 && i != 33) {
            mistral_byte_to_unicode[i] = offset;
            mistral_unicode_to_byte[offset] = i;
            offset++;
        }
    }
    mistral_byte_to_unicode[0] = 256;
    mistral_unicode_to_byte[256] = 0;

    mistral_byte_encoder_initialized = 1;
}

static int mistral_encode_byte_to_utf8(unsigned char b, char *out) {
    mistral_init_byte_encoder();
    int cp = mistral_byte_to_unicode[b];
    if (cp < 128) {
        out[0] = (char)cp;
        return 1;
    } else if (cp < 2048) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    }
    out[0] = '?';
    return 1;
}

/* ========================================================================
 * Hash Functions
 * ======================================================================== */

static unsigned int mistral_hash_string(const char *str) {
    unsigned int hash = 2166136261u;
    while (*str) {
        hash ^= (unsigned char)*str++;
        hash *= 16777619u;
    }
    return hash;
}

static void mistral_vocab_hash_insert(mistral_vocab_entry_t *table, int hash_size,
                                       const char *token, int id) {
    unsigned int h = mistral_hash_string(token) % hash_size;
    int probes = 0;
    while (table[h].token != NULL && probes < hash_size) {
        if (strcmp(table[h].token, token) == 0) return;
        h = (h + 1) % hash_size;
        probes++;
    }
    if (probes < hash_size) {
        table[h].token = strdup(token);
        table[h].id = id;
    }
}

static int mistral_vocab_hash_lookup(const mistral_vocab_entry_t *table, int hash_size,
                                      const char *token) {
    unsigned int h = mistral_hash_string(token) % hash_size;
    int probes = 0;
    while (table[h].token != NULL && probes < hash_size) {
        if (strcmp(table[h].token, token) == 0) return table[h].id;
        h = (h + 1) % hash_size;
        probes++;
    }
    return -1;
}

/* ========================================================================
 * JSON Parsing Helpers
 * ======================================================================== */

static const char *mistral_skip_ws(const char *p) {
    while (*p && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
    return p;
}

static char *mistral_parse_json_string(const char **pp) {
    const char *p = *pp;
    if (*p != '"') return NULL;
    p++;

    const char *start = p;
    int len = 0;
    while (*p && *p != '"') {
        if (*p == '\\' && p[1]) { p += 2; len++; }
        else { p++; len++; }
    }

    char *result = malloc(len + 1);
    if (!result) return NULL;

    p = start;
    int i = 0;
    while (*p && *p != '"') {
        if (*p == '\\' && p[1]) {
            p++;
            switch (*p) {
                case 'n': result[i++] = '\n'; break;
                case 'r': result[i++] = '\r'; break;
                case 't': result[i++] = '\t'; break;
                case '\\': result[i++] = '\\'; break;
                case '"': result[i++] = '"'; break;
                case 'u': {
                    if (p[1] && p[2] && p[3] && p[4]) {
                        char hex[5] = {p[1], p[2], p[3], p[4], 0};
                        int cp = (int)strtol(hex, NULL, 16);
                        p += 4;
                        if (cp < 0x80) {
                            result[i++] = (char)cp;
                        } else if (cp < 0x800) {
                            result[i++] = (char)(0xC0 | (cp >> 6));
                            result[i++] = (char)(0x80 | (cp & 0x3F));
                            len++;
                        } else {
                            result[i++] = (char)(0xE0 | (cp >> 12));
                            result[i++] = (char)(0x80 | ((cp >> 6) & 0x3F));
                            result[i++] = (char)(0x80 | (cp & 0x3F));
                            len += 2;
                        }
                    }
                    break;
                }
                default: result[i++] = *p; break;
            }
            p++;
        } else {
            result[i++] = *p++;
        }
    }
    result[i] = '\0';

    if (*p == '"') p++;
    *pp = p;
    return result;
}

static int mistral_parse_json_int(const char **pp) {
    const char *p = *pp;
    int neg = 0;
    if (*p == '-') { neg = 1; p++; }
    int val = 0;
    while (*p >= '0' && *p <= '9') {
        val = val * 10 + (*p - '0');
        p++;
    }
    *pp = p;
    return neg ? -val : val;
}

static const char *mistral_skip_json_value(const char *p) {
    p = mistral_skip_ws(p);
    if (*p == '"') {
        p++;
        while (*p && *p != '"') {
            if (*p == '\\' && p[1]) p += 2;
            else p++;
        }
        if (*p == '"') p++;
    } else if (*p == '{') {
        int depth = 1; p++;
        while (*p && depth > 0) {
            if (*p == '{') depth++;
            else if (*p == '}') depth--;
            else if (*p == '"') {
                p++;
                while (*p && *p != '"') {
                    if (*p == '\\' && p[1]) p += 2;
                    else p++;
                }
            }
            p++;
        }
    } else if (*p == '[') {
        int depth = 1; p++;
        while (*p && depth > 0) {
            if (*p == '[') depth++;
            else if (*p == ']') depth--;
            else if (*p == '"') {
                p++;
                while (*p && *p != '"') {
                    if (*p == '\\' && p[1]) p += 2;
                    else p++;
                }
            }
            p++;
        }
    } else {
        while (*p && *p != ',' && *p != '}' && *p != ']' &&
               *p != ' ' && *p != '\n' && *p != '\r' && *p != '\t') p++;
    }
    return p;
}

/* ========================================================================
 * Tokenizer Loading
 * ======================================================================== */

mistral_tokenizer_t *mistral_tokenizer_load(const char *tokenizer_json_path) {
    FILE *f = fopen(tokenizer_json_path, "rb");
    if (!f) {
        fprintf(stderr, "mistral_tokenizer_load: cannot open %s\n", tokenizer_json_path);
        return NULL;
    }

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char *json = malloc(size + 1);
    if (!json) { fclose(f); return NULL; }
    fread(json, 1, size, f);
    json[size] = '\0';
    fclose(f);

    mistral_tokenizer_t *tok = calloc(1, sizeof(mistral_tokenizer_t));
    if (!tok) { free(json); return NULL; }

    tok->hash_size = MISTRAL_HASH_SIZE;
    tok->vocab_hash = calloc(tok->hash_size, sizeof(mistral_vocab_entry_t));
    if (!tok->vocab_hash) { free(tok); free(json); return NULL; }

    /* Set default special token IDs */
    tok->bos_id = MISTRAL_DEFAULT_BOS_ID;
    tok->eos_id = MISTRAL_DEFAULT_EOS_ID;
    tok->pad_id = MISTRAL_DEFAULT_PAD_ID;
    tok->inst_start_id = -1;
    tok->inst_end_id = -1;

    /* Detect byte encoding mode from pre_tokenizer.
     * If "ByteLevel" is found → GPT-2 style byte encoding.
     * If "ByteFallback" is found → raw UTF-8 with byte fallback. */
    tok->use_byte_encoding = 0;
    if (strstr(json, "\"ByteLevel\"")) {
        tok->use_byte_encoding = 1;
    }

    /* Parse vocabulary from "model": { "vocab": { ... } } */
    const char *p = strstr(json, "\"model\"");
    if (!p) {
        fprintf(stderr, "mistral_tokenizer_load: no model section\n");
        goto error;
    }

    p = strstr(p, "\"vocab\"");
    if (!p) {
        fprintf(stderr, "mistral_tokenizer_load: no vocab section\n");
        goto error;
    }

    p = strchr(p, '{');
    if (!p) goto error;
    p++;

    /* Count vocab entries */
    int vocab_count = 0;
    const char *count_p = p;
    int depth = 1;
    while (*count_p && depth > 0) {
        if (*count_p == '{') depth++;
        else if (*count_p == '}') depth--;
        else if (*count_p == '"' && depth == 1) {
            vocab_count++;
            count_p++;
            while (*count_p && *count_p != '"') {
                if (*count_p == '\\' && count_p[1]) count_p += 2;
                else count_p++;
            }
        }
        count_p++;
    }

    tok->vocab_size = vocab_count;
    tok->vocab = calloc(vocab_count + 1000, sizeof(char *));
    if (!tok->vocab) goto error;

    /* Parse vocab entries */
    p = mistral_skip_ws(p);
    int max_id = 0;
    while (*p && *p != '}') {
        if (*p == '"') {
            char *token = mistral_parse_json_string(&p);
            p = mistral_skip_ws(p);
            if (*p == ':') p++;
            p = mistral_skip_ws(p);
            int id = mistral_parse_json_int(&p);

            if (token && id >= 0 && id < vocab_count + 1000) {
                tok->vocab[id] = token;
                mistral_vocab_hash_insert(tok->vocab_hash, tok->hash_size, token, id);
                if (id > max_id) max_id = id;
            } else {
                free(token);
            }

            p = mistral_skip_ws(p);
            if (*p == ',') p++;
            p = mistral_skip_ws(p);
        } else {
            p++;
        }
    }

    /* Parse merges from "model": { "merges": [ ... ] } */
    p = strstr(json, "\"merges\"");
    if (!p) {
        fprintf(stderr, "mistral_tokenizer_load: no merges section\n");
        goto error;
    }

    p = strchr(p, '[');
    if (!p) goto error;
    p++;

    /* Count merges */
    int merge_count = 0;
    count_p = p;
    depth = 1;
    while (*count_p && depth > 0) {
        if (*count_p == '[') {
            if (depth == 1) merge_count++;
            depth++;
        } else if (*count_p == ']') {
            depth--;
        } else if (*count_p == '"') {
            count_p++;
            while (*count_p && *count_p != '"') {
                if (*count_p == '\\' && count_p[1]) count_p += 2;
                else count_p++;
            }
        }
        if (*count_p) count_p++;
    }

    tok->num_merges = merge_count;
    tok->merges = calloc(merge_count, sizeof(mistral_bpe_merge_t));
    tok->merge_ranks = calloc(tok->hash_size, sizeof(int));
    if (!tok->merges || !tok->merge_ranks) goto error;

    for (int i = 0; i < tok->hash_size; i++) {
        tok->merge_ranks[i] = -1;
    }

    /* Parse merges */
    p = mistral_skip_ws(p);
    int merge_idx = 0;
    while (*p && *p != ']' && merge_idx < merge_count) {
        if (*p == '[') {
            p++;
            p = mistral_skip_ws(p);

            char *left = NULL, *right = NULL;
            if (*p == '"') left = mistral_parse_json_string(&p);
            p = mistral_skip_ws(p);
            if (*p == ',') p++;
            p = mistral_skip_ws(p);
            if (*p == '"') right = mistral_parse_json_string(&p);

            while (*p && *p != ']') p++;
            if (*p == ']') p++;

            if (left && right) {
                tok->merges[merge_idx].left = left;
                tok->merges[merge_idx].right = right;
                tok->merges[merge_idx].rank = merge_idx;

                int len1 = strlen(left);
                int len2 = strlen(right);
                char *key = malloc(len1 + len2 + 2);
                memcpy(key, left, len1);
                key[len1] = ' ';
                memcpy(key + len1 + 1, right, len2);
                key[len1 + len2 + 1] = '\0';

                unsigned int h = mistral_hash_string(key) % tok->hash_size;
                int probes = 0;
                while (tok->merge_ranks[h] != -1 && probes < tok->hash_size) {
                    h = (h + 1) % tok->hash_size;
                    probes++;
                }
                if (probes < tok->hash_size) {
                    tok->merge_ranks[h] = merge_idx;
                }
                free(key);
            } else {
                free(left);
                free(right);
            }

            merge_idx++;
            p = mistral_skip_ws(p);
            if (*p == ',') p++;
            p = mistral_skip_ws(p);
        } else {
            p++;
        }
    }

    /* Parse added_tokens for special tokens */
    p = strstr(json, "\"added_tokens\"");
    if (p) {
        p = strchr(p, '[');
        if (p) {
            p++;
            while (*p && *p != ']') {
                if (*p == '{') {
                    p++;
                    char *content = NULL;
                    int id = -1;

                    while (*p && *p != '}') {
                        p = mistral_skip_ws(p);
                        if (*p == '"') {
                            char *key = mistral_parse_json_string(&p);
                            p = mistral_skip_ws(p);
                            if (*p == ':') p++;
                            p = mistral_skip_ws(p);

                            if (key && strcmp(key, "content") == 0 && *p == '"') {
                                content = mistral_parse_json_string(&p);
                            } else if (key && strcmp(key, "id") == 0) {
                                id = mistral_parse_json_int(&p);
                            } else {
                                p = mistral_skip_json_value(p);
                            }
                            free(key);
                        }
                        p = mistral_skip_ws(p);
                        if (*p == ',') p++;
                    }

                    if (content && id >= 0) {
                        if (id < vocab_count + 1000) {
                            if (!tok->vocab[id]) {
                                tok->vocab[id] = content;
                                mistral_vocab_hash_insert(tok->vocab_hash, tok->hash_size, content, id);
                                if (id > max_id) max_id = id;
                                content = NULL;
                            }
                        }

                        /* Detect known special tokens */
                        if (content) {
                            if (strcmp(content, "<s>") == 0) tok->bos_id = id;
                            else if (strcmp(content, "</s>") == 0) tok->eos_id = id;
                            else if (strcmp(content, "[INST]") == 0) tok->inst_start_id = id;
                            else if (strcmp(content, "[/INST]") == 0) tok->inst_end_id = id;
                        } else {
                            /* content was moved to vocab, check from vocab */
                            const char *v = tok->vocab[id];
                            if (v) {
                                if (strcmp(v, "<s>") == 0) tok->bos_id = id;
                                else if (strcmp(v, "</s>") == 0) tok->eos_id = id;
                                else if (strcmp(v, "[INST]") == 0) tok->inst_start_id = id;
                                else if (strcmp(v, "[/INST]") == 0) tok->inst_end_id = id;
                            }
                        }
                    }
                    free(content);

                    if (*p == '}') p++;
                }
                p = mistral_skip_ws(p);
                if (*p == ',') p++;
            }
        }
    }

    tok->vocab_size = max_id + 1;

    free(json);

    if (iris_verbose)
        fprintf(stderr, " Mistral tokenizer loaded (%d vocab, bos=%d, eos=%d, byte_enc=%d)\n",
                tok->vocab_size, tok->bos_id, tok->eos_id, tok->use_byte_encoding);

    return tok;

error:
    free(json);
    if (tok) {
        free(tok->vocab_hash);
        free(tok->vocab);
        free(tok->merges);
        free(tok->merge_ranks);
        free(tok);
    }
    return NULL;
}

void mistral_tokenizer_free(mistral_tokenizer_t *tok) {
    if (!tok) return;

    if (tok->vocab) {
        for (int i = 0; i < tok->vocab_size; i++) free(tok->vocab[i]);
        free(tok->vocab);
    }

    if (tok->vocab_hash) {
        for (int i = 0; i < tok->hash_size; i++) free(tok->vocab_hash[i].token);
        free(tok->vocab_hash);
    }

    if (tok->merges) {
        for (int i = 0; i < tok->num_merges; i++) {
            free(tok->merges[i].left);
            free(tok->merges[i].right);
        }
        free(tok->merges);
    }

    free(tok->merge_ranks);
    free(tok);
}

/* ========================================================================
 * Merge Rank Lookup
 * ======================================================================== */

static int mistral_get_merge_rank(mistral_tokenizer_t *tok, const char *left, const char *right) {
    int len1 = strlen(left);
    int len2 = strlen(right);
    char *key = malloc(len1 + len2 + 2);
    if (!key) return -1;
    memcpy(key, left, len1);
    key[len1] = ' ';
    memcpy(key + len1 + 1, right, len2);
    key[len1 + len2 + 1] = '\0';

    unsigned int h = mistral_hash_string(key) % tok->hash_size;
    int probes = 0;
    while (tok->merge_ranks[h] != -1 && probes < tok->hash_size) {
        int rank = tok->merge_ranks[h];
        if (rank >= 0 && rank < tok->num_merges) {
            if (strcmp(tok->merges[rank].left, left) == 0 &&
                strcmp(tok->merges[rank].right, right) == 0) {
                free(key);
                return rank;
            }
        }
        h = (h + 1) % tok->hash_size;
        probes++;
    }

    free(key);
    return -1;
}

/* ========================================================================
 * BPE Tokenization
 * ======================================================================== */

typedef struct mistral_token_node {
    char *text;
    struct mistral_token_node *next;
} mistral_token_node_t;

static mistral_token_node_t *mistral_create_node(const char *text) {
    mistral_token_node_t *node = malloc(sizeof(mistral_token_node_t));
    if (node) {
        node->text = strdup(text);
        node->next = NULL;
    }
    return node;
}

static void mistral_free_token_list(mistral_token_node_t *head) {
    while (head) {
        mistral_token_node_t *next = head->next;
        free(head->text);
        free(head);
        head = next;
    }
}

static mistral_token_node_t *mistral_bpe_encode_word(mistral_tokenizer_t *tok, const char *word) {
    int len = strlen(word);
    if (len == 0) return NULL;

    /* Start with character-level tokens */
    mistral_token_node_t *head = NULL;
    mistral_token_node_t *tail = NULL;

    const char *p = word;
    while (*p) {
        int char_len = 1;
        unsigned char c = (unsigned char)*p;
        if ((c & 0xE0) == 0xC0) char_len = 2;
        else if ((c & 0xF0) == 0xE0) char_len = 3;
        else if ((c & 0xF8) == 0xF0) char_len = 4;

        char buf[8];
        memcpy(buf, p, char_len);
        buf[char_len] = '\0';

        mistral_token_node_t *node = mistral_create_node(buf);
        if (!head) head = node;
        else tail->next = node;
        tail = node;

        p += char_len;
    }

    /* Apply BPE merges */
    int changed = 1;
    while (changed) {
        changed = 0;

        int best_rank = tok->num_merges + 1;
        mistral_token_node_t *best_node = NULL;

        for (mistral_token_node_t *node = head; node && node->next; node = node->next) {
            int rank = mistral_get_merge_rank(tok, node->text, node->next->text);
            if (rank >= 0 && rank < best_rank) {
                best_rank = rank;
                best_node = node;
            }
        }

        if (best_node) {
            int len1 = strlen(best_node->text);
            int len2 = strlen(best_node->next->text);
            char *merged = malloc(len1 + len2 + 1);
            memcpy(merged, best_node->text, len1);
            memcpy(merged + len1, best_node->next->text, len2);
            merged[len1 + len2] = '\0';

            free(best_node->text);
            best_node->text = merged;

            mistral_token_node_t *to_free = best_node->next;
            best_node->next = to_free->next;
            free(to_free->text);
            free(to_free);

            changed = 1;
        }
    }

    return head;
}

/* Convert text to byte-level encoding (only used when use_byte_encoding=1) */
static char *mistral_text_to_bytes(const char *text) {
    mistral_init_byte_encoder();
    int len = strlen(text);
    char *result = malloc(len * 2 + 1);
    if (!result) return NULL;

    int j = 0;
    for (int i = 0; i < len; i++) {
        j += mistral_encode_byte_to_utf8((unsigned char)text[i], result + j);
    }
    result[j] = '\0';
    return result;
}

/* ========================================================================
 * Pre-tokenization
 * ======================================================================== */

static char **mistral_pretokenize(const char *text, int *num_chunks) {
    int capacity = 64;
    char **chunks = malloc(capacity * sizeof(char *));
    int count = 0;

    const char *p = text;
    while (*p) {
        const char *start = p;

        if (*p == '\'' && p[1]) {
            char lower = tolower(p[1]);
            if (lower == 's' || lower == 't' || lower == 'm' || lower == 'd') {
                p += 2;
            } else if ((lower == 'r' || lower == 'v' || lower == 'l') && p[2] &&
                       (tolower(p[2]) == 'e' || tolower(p[2]) == 'l')) {
                p += 3;
            } else {
                p++;
            }
        }
        else if ((*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z') ||
                 (unsigned char)*p >= 128) {
            while (*p && ((*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z') ||
                          (unsigned char)*p >= 128)) {
                if ((unsigned char)*p >= 128) {
                    if (((unsigned char)*p & 0xE0) == 0xC0) p += 2;
                    else if (((unsigned char)*p & 0xF0) == 0xE0) p += 3;
                    else if (((unsigned char)*p & 0xF8) == 0xF0) p += 4;
                    else p++;
                } else {
                    p++;
                }
            }
        }
        else if (*p >= '0' && *p <= '9') {
            while (*p >= '0' && *p <= '9') p++;
        }
        else if (*p == ' ' && p[1] && (isalpha(p[1]) || (unsigned char)p[1] >= 128)) {
            p++;
            while (*p && ((*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z') ||
                          (unsigned char)*p >= 128)) {
                if ((unsigned char)*p >= 128) {
                    if (((unsigned char)*p & 0xE0) == 0xC0) p += 2;
                    else if (((unsigned char)*p & 0xF0) == 0xE0) p += 3;
                    else if (((unsigned char)*p & 0xF8) == 0xF0) p += 4;
                    else p++;
                } else {
                    p++;
                }
            }
        }
        else if (*p == ' ' && p[1] >= '0' && p[1] <= '9') {
            p++;
            while (*p >= '0' && *p <= '9') p++;
        }
        else if (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t') {
            while (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t') p++;
        }
        else {
            p++;
        }

        if (p > start) {
            int len = p - start;
            char *chunk = malloc(len + 1);
            memcpy(chunk, start, len);
            chunk[len] = '\0';

            if (count >= capacity) {
                capacity *= 2;
                chunks = realloc(chunks, capacity * sizeof(char *));
            }
            chunks[count++] = chunk;
        }
    }

    *num_chunks = count;
    return chunks;
}

/* ========================================================================
 * Main Tokenization API
 * ======================================================================== */

static int *mistral_tokenize(mistral_tokenizer_t *tok, const char *text,
                              int *num_tokens, int max_len) {
    if (max_len <= 0) max_len = MISTRAL_MAX_SEQ_LEN;

    int num_chunks;
    char **chunks = mistral_pretokenize(text, &num_chunks);

    int capacity = 256;
    int *tokens = malloc(capacity * sizeof(int));
    int total = 0;

    for (int c = 0; c < num_chunks && total < max_len; c++) {
        char *input_text;
        if (tok->use_byte_encoding) {
            input_text = mistral_text_to_bytes(chunks[c]);
        } else {
            input_text = strdup(chunks[c]);
        }
        if (!input_text) { free(chunks[c]); continue; }

        mistral_token_node_t *bpe_tokens = mistral_bpe_encode_word(tok, input_text);

        for (mistral_token_node_t *node = bpe_tokens; node && total < max_len; node = node->next) {
            int id = mistral_vocab_hash_lookup(tok->vocab_hash, tok->hash_size, node->text);
            if (id >= 0) {
                if (total >= capacity) {
                    capacity *= 2;
                    tokens = realloc(tokens, capacity * sizeof(int));
                }
                tokens[total++] = id;
            }
            /* TODO: byte fallback for unknown tokens (<0xNN>) */
        }

        mistral_free_token_list(bpe_tokens);
        free(input_text);
        free(chunks[c]);
    }
    free(chunks);

    *num_tokens = total;
    return tokens;
}

/* Tokenize with Mistral chat template for FLUX.2-dev.
 * Format: <s>[INST] {prompt} [/INST] */
int *mistral_tokenize_chat(mistral_tokenizer_t *tok, const char *prompt,
                           int *num_tokens, int max_len) {
    if (max_len <= 0) max_len = MISTRAL_MAX_SEQ_LEN;

    int capacity = 256;
    int *tokens = malloc(capacity * sizeof(int));
    int total = 0;

    /* BOS token */
    tokens[total++] = tok->bos_id;

    /* [INST] token */
    if (tok->inst_start_id >= 0) {
        tokens[total++] = tok->inst_start_id;
    } else {
        /* Tokenize [INST] as text */
        int n;
        int *inst_tokens = mistral_tokenize(tok, "[INST]", &n, max_len - total);
        for (int i = 0; i < n && total < max_len; i++) {
            if (total >= capacity) { capacity *= 2; tokens = realloc(tokens, capacity * sizeof(int)); }
            tokens[total++] = inst_tokens[i];
        }
        free(inst_tokens);
    }

    /* Space + prompt */
    {
        /* Prepend space to prompt for proper tokenization */
        int prompt_len = strlen(prompt);
        char *spaced_prompt = malloc(prompt_len + 2);
        spaced_prompt[0] = ' ';
        memcpy(spaced_prompt + 1, prompt, prompt_len + 1);

        int n;
        int *prompt_tokens = mistral_tokenize(tok, spaced_prompt, &n, max_len - total);
        for (int i = 0; i < n && total < max_len; i++) {
            if (total >= capacity) { capacity *= 2; tokens = realloc(tokens, capacity * sizeof(int)); }
            tokens[total++] = prompt_tokens[i];
        }
        free(prompt_tokens);
        free(spaced_prompt);
    }

    /* Space + [/INST] */
    if (tok->inst_end_id >= 0) {
        /* Tokenize the space before [/INST] */
        int n;
        int *space_tokens = mistral_tokenize(tok, " ", &n, max_len - total);
        for (int i = 0; i < n && total < max_len; i++) {
            if (total >= capacity) { capacity *= 2; tokens = realloc(tokens, capacity * sizeof(int)); }
            tokens[total++] = space_tokens[i];
        }
        free(space_tokens);

        if (total < max_len) {
            if (total >= capacity) { capacity *= 2; tokens = realloc(tokens, capacity * sizeof(int)); }
            tokens[total++] = tok->inst_end_id;
        }
    } else {
        int n;
        int *inst_tokens = mistral_tokenize(tok, " [/INST]", &n, max_len - total);
        for (int i = 0; i < n && total < max_len; i++) {
            if (total >= capacity) { capacity *= 2; tokens = realloc(tokens, capacity * sizeof(int)); }
            tokens[total++] = inst_tokens[i];
        }
        free(inst_tokens);
    }

    *num_tokens = total;
    return tokens;
}

int *mistral_pad_tokens(int *tokens, int num_tokens, int max_len, int *attention_mask) {
    int *padded = malloc(max_len * sizeof(int));
    if (!padded) return NULL;

    for (int i = 0; i < max_len; i++) {
        if (i < num_tokens) {
            padded[i] = tokens[i];
            if (attention_mask) attention_mask[i] = 1;
        } else {
            padded[i] = MISTRAL_DEFAULT_PAD_ID;
            if (attention_mask) attention_mask[i] = 0;
        }
    }

    return padded;
}

const char *mistral_get_token(mistral_tokenizer_t *tok, int id) {
    if (!tok || id < 0 || id >= tok->vocab_size) return NULL;
    return tok->vocab[id];
}
