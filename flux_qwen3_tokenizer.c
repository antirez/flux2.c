/*
 * Qwen3 Tokenizer Implementation
 *
 * BPE (Byte Pair Encoding) tokenizer for Qwen3 text encoder.
 * Loads directly from HuggingFace tokenizer.json format.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdint.h>
#include "flux_kernels.h"

/* ========================================================================
 * Configuration
 * ======================================================================== */

#define QWEN3_VOCAB_SIZE 151936
#define QWEN3_MAX_TOKEN_LEN 256
#define QWEN3_MAX_SEQ_LEN 512
#define QWEN3_HASH_SIZE 300007  /* Prime > 2 * vocab_size */

/* Special token IDs */
#define QWEN3_PAD_ID 151643      /* <|endoftext|> */
#define QWEN3_IM_START_ID 151644 /* <|im_start|> */
#define QWEN3_IM_END_ID 151645   /* <|im_end|> */
#define QWEN3_THINK_START_ID 151667 /* <think> */
#define QWEN3_THINK_END_ID 151668   /* </think> */

/* ========================================================================
 * Data Structures
 * ======================================================================== */

typedef struct {
    char *token;
    int id;
} vocab_entry_t;

typedef struct {
    char *left;
    char *right;
    int rank;  /* Lower rank = higher priority (merge first) */
} bpe_merge_t;

typedef struct qwen3_tokenizer {
    /* Vocabulary: id -> token string */
    char **vocab;
    int vocab_size;

    /* Hash table: token string -> id */
    vocab_entry_t *vocab_hash;
    int hash_size;

    /* BPE merges */
    bpe_merge_t *merges;
    int num_merges;

    /* Merge rank lookup: "left right" -> rank */
    int *merge_ranks;  /* Hash table: hash("left right") -> rank, or -1 */
} qwen3_tokenizer_t;

/* ========================================================================
 * Byte-Level BPE Encoding Table
 * ======================================================================== */

/*
 * GPT-2/Qwen style byte-to-unicode mapping.
 * Bytes 33-126 and 161-172 and 174-255 map to themselves.
 * Other bytes (0-32, 127-160, 173) map to 256+i for uniqueness.
 */
static int byte_to_unicode[256];
static int unicode_to_byte[512];
static int byte_encoder_initialized = 0;

static void init_byte_encoder(void) {
    if (byte_encoder_initialized) return;

    /* Printable ASCII and extended Latin */
    for (int i = 33; i <= 126; i++) {
        byte_to_unicode[i] = i;
        unicode_to_byte[i] = i;
    }
    for (int i = 161; i <= 172; i++) {
        byte_to_unicode[i] = i;
        unicode_to_byte[i] = i;
    }
    for (int i = 174; i <= 255; i++) {
        byte_to_unicode[i] = i;
        unicode_to_byte[i] = i;
    }

    /* Map remaining bytes to 256+ range */
    int offset = 256;
    for (int i = 0; i < 256; i++) {
        if (byte_to_unicode[i] == 0 && i != 33) {  /* 33 maps to itself */
            byte_to_unicode[i] = offset;
            unicode_to_byte[offset] = i;
            offset++;
        }
    }
    /* Fix: byte 0 should also be mapped */
    byte_to_unicode[0] = 256;
    unicode_to_byte[256] = 0;

    byte_encoder_initialized = 1;
}

/* Encode a byte to its unicode character (UTF-8) */
static int encode_byte_to_utf8(unsigned char b, char *out) {
    init_byte_encoder();
    int cp = byte_to_unicode[b];
    if (cp < 128) {
        out[0] = (char)cp;
        return 1;
    } else if (cp < 2048) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    }
    /* Shouldn't reach here for byte-level BPE */
    out[0] = '?';
    return 1;
}


/* ========================================================================
 * Hash Functions
 * ======================================================================== */

static unsigned int hash_string(const char *str) {
    unsigned int hash = 2166136261u;
    while (*str) {
        hash ^= (unsigned char)*str++;
        hash *= 16777619u;
    }
    return hash;
}

static void vocab_hash_insert(vocab_entry_t *table, int hash_size,
                              const char *token, int id) {
    unsigned int h = hash_string(token) % hash_size;
    int probes = 0;
    while (table[h].token != NULL && probes < hash_size) {
        if (strcmp(table[h].token, token) == 0) {
            return;  /* Already exists */
        }
        h = (h + 1) % hash_size;
        probes++;
    }
    if (probes < hash_size) {
        table[h].token = strdup(token);
        table[h].id = id;
    }
}

static int vocab_hash_lookup(const vocab_entry_t *table, int hash_size,
                             const char *token) {
    unsigned int h = hash_string(token) % hash_size;
    int probes = 0;
    while (table[h].token != NULL && probes < hash_size) {
        if (strcmp(table[h].token, token) == 0) {
            return table[h].id;
        }
        h = (h + 1) % hash_size;
        probes++;
    }
    return -1;
}

/* ========================================================================
 * JSON Parsing Helpers (minimal, just for tokenizer.json)
 * ======================================================================== */

/* Skip whitespace */
static const char *skip_ws(const char *p) {
    while (*p && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
    return p;
}

/* Parse a JSON string, returns allocated string, advances *p past closing quote */
static char *parse_json_string(const char **pp) {
    const char *p = *pp;
    if (*p != '"') return NULL;
    p++;

    /* Find string length first (handling escapes) */
    const char *start = p;
    int len = 0;
    while (*p && *p != '"') {
        if (*p == '\\' && p[1]) {
            p += 2;
            len++;
        } else {
            p++;
            len++;
        }
    }

    /* Allocate and copy */
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
                    /* Parse \uXXXX */
                    if (p[1] && p[2] && p[3] && p[4]) {
                        char hex[5] = {p[1], p[2], p[3], p[4], 0};
                        int cp = (int)strtol(hex, NULL, 16);
                        p += 4;
                        /* Encode as UTF-8 */
                        if (cp < 0x80) {
                            result[i++] = (char)cp;
                        } else if (cp < 0x800) {
                            result[i++] = (char)(0xC0 | (cp >> 6));
                            result[i++] = (char)(0x80 | (cp & 0x3F));
                            len++;  /* Need more space */
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

/* Parse a JSON integer */
static int parse_json_int(const char **pp) {
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

/* Skip a JSON value (string, number, object, array, bool, null) */
static const char *skip_json_value(const char *p) {
    p = skip_ws(p);
    if (*p == '"') {
        p++;
        while (*p && *p != '"') {
            if (*p == '\\' && p[1]) p += 2;
            else p++;
        }
        if (*p == '"') p++;
    } else if (*p == '{') {
        int depth = 1;
        p++;
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
        int depth = 1;
        p++;
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
        /* number, bool, null */
        while (*p && *p != ',' && *p != '}' && *p != ']' &&
               *p != ' ' && *p != '\n' && *p != '\r' && *p != '\t') p++;
    }
    return p;
}

/* ========================================================================
 * Tokenizer Loading
 * ======================================================================== */

qwen3_tokenizer_t *qwen3_tokenizer_load(const char *tokenizer_json_path) {
    /* Read file */
    FILE *f = fopen(tokenizer_json_path, "rb");
    if (!f) {
        fprintf(stderr, "qwen3_tokenizer_load: cannot open %s\n", tokenizer_json_path);
        return NULL;
    }

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char *json = malloc(size + 1);
    if (!json) {
        fclose(f);
        return NULL;
    }
    fread(json, 1, size, f);
    json[size] = '\0';
    fclose(f);

    qwen3_tokenizer_t *tok = calloc(1, sizeof(qwen3_tokenizer_t));
    if (!tok) {
        free(json);
        return NULL;
    }

    tok->hash_size = QWEN3_HASH_SIZE;
    tok->vocab_hash = calloc(tok->hash_size, sizeof(vocab_entry_t));
    if (!tok->vocab_hash) {
        free(tok);
        free(json);
        return NULL;
    }

    /* Parse vocabulary from "model": { "vocab": { ... } } */
    const char *p = strstr(json, "\"model\"");
    if (!p) {
        fprintf(stderr, "qwen3_tokenizer_load: no model section\n");
        goto error;
    }

    p = strstr(p, "\"vocab\"");
    if (!p) {
        fprintf(stderr, "qwen3_tokenizer_load: no vocab section\n");
        goto error;
    }

    /* Skip to opening brace */
    p = strchr(p, '{');
    if (!p) goto error;
    p++;

    /* Count vocab entries first */
    int vocab_count = 0;
    const char *count_p = p;
    int depth = 1;
    while (*count_p && depth > 0) {
        if (*count_p == '{') depth++;
        else if (*count_p == '}') depth--;
        else if (*count_p == '"' && depth == 1) {
            vocab_count++;
            /* Skip the string */
            count_p++;
            while (*count_p && *count_p != '"') {
                if (*count_p == '\\' && count_p[1]) count_p += 2;
                else count_p++;
            }
        }
        count_p++;
    }
    /* vocab_count is the number of string keys (values are integers, not strings) */

    tok->vocab_size = vocab_count;
    tok->vocab = calloc(vocab_count + 1000, sizeof(char *));  /* Extra for added_tokens */
    if (!tok->vocab) goto error;

    /* Parse vocab entries */
    p = skip_ws(p);
    int max_id = 0;
    while (*p && *p != '}') {
        if (*p == '"') {
            char *token = parse_json_string(&p);
            p = skip_ws(p);
            if (*p == ':') p++;
            p = skip_ws(p);
            int id = parse_json_int(&p);

            if (token && id >= 0 && id < vocab_count + 1000) {
                tok->vocab[id] = token;
                vocab_hash_insert(tok->vocab_hash, tok->hash_size, token, id);
                if (id > max_id) max_id = id;
            } else {
                free(token);
            }

            p = skip_ws(p);
            if (*p == ',') p++;
            p = skip_ws(p);
        } else {
            p++;
        }
    }

    /* Parse merges from "model": { "merges": [ ... ] }
     * Merges are arrays like: [["Ġ", "Ġ"], ["ĠĠ", "ĠĠ"], ...]
     */
    p = strstr(json, "\"merges\"");
    if (!p) {
        fprintf(stderr, "qwen3_tokenizer_load: no merges section\n");
        goto error;
    }

    p = strchr(p, '[');
    if (!p) goto error;
    p++;

    /* Count merges by counting '[' characters at depth 1 */
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
    tok->merges = calloc(merge_count, sizeof(bpe_merge_t));
    tok->merge_ranks = calloc(tok->hash_size, sizeof(int));
    if (!tok->merges || !tok->merge_ranks) goto error;

    /* Initialize merge_ranks to -1 */
    for (int i = 0; i < tok->hash_size; i++) {
        tok->merge_ranks[i] = -1;
    }

    /* Parse merges - format is [["left", "right"], ...] */
    p = skip_ws(p);
    int merge_idx = 0;
    while (*p && *p != ']' && merge_idx < merge_count) {
        if (*p == '[') {
            p++;
            p = skip_ws(p);

            /* Parse left string */
            char *left = NULL;
            char *right = NULL;
            if (*p == '"') {
                left = parse_json_string(&p);
            }
            p = skip_ws(p);
            if (*p == ',') p++;
            p = skip_ws(p);

            /* Parse right string */
            if (*p == '"') {
                right = parse_json_string(&p);
            }

            /* Skip to closing ] */
            while (*p && *p != ']') p++;
            if (*p == ']') p++;

            if (left && right) {
                tok->merges[merge_idx].left = left;
                tok->merges[merge_idx].right = right;
                tok->merges[merge_idx].rank = merge_idx;

                /* Add to merge rank lookup: hash "left right" */
                int len1 = strlen(left);
                int len2 = strlen(right);
                char *key = malloc(len1 + len2 + 2);
                memcpy(key, left, len1);
                key[len1] = ' ';
                memcpy(key + len1 + 1, right, len2);
                key[len1 + len2 + 1] = '\0';

                unsigned int h = hash_string(key) % tok->hash_size;
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

            p = skip_ws(p);
            if (*p == ',') p++;
            p = skip_ws(p);
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
                    /* Parse added token object */
                    p++;
                    char *content = NULL;
                    int id = -1;

                    while (*p && *p != '}') {
                        p = skip_ws(p);
                        if (*p == '"') {
                            char *key = parse_json_string(&p);
                            p = skip_ws(p);
                            if (*p == ':') p++;
                            p = skip_ws(p);

                            if (key && strcmp(key, "content") == 0 && *p == '"') {
                                content = parse_json_string(&p);
                            } else if (key && strcmp(key, "id") == 0) {
                                id = parse_json_int(&p);
                            } else {
                                p = skip_json_value(p);
                            }
                            free(key);
                        }
                        p = skip_ws(p);
                        if (*p == ',') p++;
                    }

                    if (content && id >= 0) {
                        if (id < vocab_count + 1000) {
                            if (!tok->vocab[id]) {
                                tok->vocab[id] = content;
                                vocab_hash_insert(tok->vocab_hash, tok->hash_size, content, id);
                                if (id > max_id) max_id = id;
                                content = NULL;  /* Don't free */
                            }
                        }
                    }
                    free(content);

                    if (*p == '}') p++;
                }
                p = skip_ws(p);
                if (*p == ',') p++;
            }
        }
    }

    tok->vocab_size = max_id + 1;

    free(json);

    if (flux_verbose)
        fprintf(stderr, " Qwen3 tokenizer loaded (%d vocab)\n",
                tok->vocab_size);

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

void qwen3_tokenizer_free(qwen3_tokenizer_t *tok) {
    if (!tok) return;

    if (tok->vocab) {
        for (int i = 0; i < tok->vocab_size; i++) {
            free(tok->vocab[i]);
        }
        free(tok->vocab);
    }

    if (tok->vocab_hash) {
        for (int i = 0; i < tok->hash_size; i++) {
            free(tok->vocab_hash[i].token);
        }
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

static int get_merge_rank(qwen3_tokenizer_t *tok, const char *left, const char *right) {
    /* Build "left right" string */
    int len1 = strlen(left);
    int len2 = strlen(right);
    char *key = malloc(len1 + len2 + 2);
    if (!key) return -1;
    memcpy(key, left, len1);
    key[len1] = ' ';
    memcpy(key + len1 + 1, right, len2);
    key[len1 + len2 + 1] = '\0';

    /* Lookup in hash table */
    unsigned int h = hash_string(key) % tok->hash_size;
    int probes = 0;
    while (tok->merge_ranks[h] != -1 && probes < tok->hash_size) {
        int rank = tok->merge_ranks[h];
        if (rank >= 0 && rank < tok->num_merges) {
            /* Check if this is the right merge */
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

/* Token list node for BPE */
typedef struct token_node {
    char *text;
    struct token_node *next;
} token_node_t;

static token_node_t *create_node(const char *text) {
    token_node_t *node = malloc(sizeof(token_node_t));
    if (node) {
        node->text = strdup(text);
        node->next = NULL;
    }
    return node;
}

static void free_token_list(token_node_t *head) {
    while (head) {
        token_node_t *next = head->next;
        free(head->text);
        free(head);
        head = next;
    }
}

/* Apply BPE to a single word (already in byte-level encoding) */
static token_node_t *bpe_encode_word(qwen3_tokenizer_t *tok, const char *word) {
    int len = strlen(word);
    if (len == 0) return NULL;

    /* Start with character-level tokens */
    token_node_t *head = NULL;
    token_node_t *tail = NULL;

    const char *p = word;
    while (*p) {
        /* Get one UTF-8 character */
        int char_len = 1;
        unsigned char c = (unsigned char)*p;
        if ((c & 0xE0) == 0xC0) char_len = 2;
        else if ((c & 0xF0) == 0xE0) char_len = 3;
        else if ((c & 0xF8) == 0xF0) char_len = 4;

        char buf[8];
        memcpy(buf, p, char_len);
        buf[char_len] = '\0';

        token_node_t *node = create_node(buf);
        if (!head) head = node;
        else tail->next = node;
        tail = node;

        p += char_len;
    }

    /* Apply BPE merges */
    int changed = 1;
    while (changed) {
        changed = 0;

        /* Find best merge (lowest rank) */
        int best_rank = tok->num_merges + 1;
        token_node_t *best_node = NULL;

        for (token_node_t *node = head; node && node->next; node = node->next) {
            int rank = get_merge_rank(tok, node->text, node->next->text);
            if (rank >= 0 && rank < best_rank) {
                best_rank = rank;
                best_node = node;
            }
        }

        /* Apply best merge */
        if (best_node) {
            /* Merge best_node and best_node->next */
            int len1 = strlen(best_node->text);
            int len2 = strlen(best_node->next->text);
            char *merged = malloc(len1 + len2 + 1);
            memcpy(merged, best_node->text, len1);
            memcpy(merged + len1, best_node->next->text, len2);
            merged[len1 + len2] = '\0';

            free(best_node->text);
            best_node->text = merged;

            token_node_t *to_free = best_node->next;
            best_node->next = to_free->next;
            free(to_free->text);
            free(to_free);

            changed = 1;
        }
    }

    return head;
}

/* Convert text to byte-level encoding */
static char *text_to_bytes(const char *text) {
    init_byte_encoder();
    int len = strlen(text);

    /* Allocate worst case: each byte could become 2 UTF-8 bytes */
    char *result = malloc(len * 2 + 1);
    if (!result) return NULL;

    int j = 0;
    for (int i = 0; i < len; i++) {
        j += encode_byte_to_utf8((unsigned char)text[i], result + j);
    }
    result[j] = '\0';

    return result;
}

/* ========================================================================
 * Pre-tokenization (GPT-style regex split)
 * ======================================================================== */

/*
 * Simplified pre-tokenizer that handles the main cases:
 * - Contractions: 's, 't, 're, 've, 'm, 'll, 'd
 * - Words with optional leading space
 * - Numbers
 * - Punctuation/symbols
 * - Whitespace
 */
static char **pretokenize(const char *text, int *num_chunks) {
    int capacity = 64;
    char **chunks = malloc(capacity * sizeof(char *));
    int count = 0;

    const char *p = text;
    while (*p) {
        const char *start = p;

        /* Check for contractions */
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
        /* Letters (possibly with leading space) */
        else if ((*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z') ||
                 (unsigned char)*p >= 128) {
            while (*p && ((*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z') ||
                          (unsigned char)*p >= 128)) {
                if ((unsigned char)*p >= 128) {
                    /* Skip UTF-8 continuation bytes */
                    if (((unsigned char)*p & 0xE0) == 0xC0) p += 2;
                    else if (((unsigned char)*p & 0xF0) == 0xE0) p += 3;
                    else if (((unsigned char)*p & 0xF8) == 0xF0) p += 4;
                    else p++;
                } else {
                    p++;
                }
            }
        }
        /* Numbers */
        else if (*p >= '0' && *p <= '9') {
            while (*p >= '0' && *p <= '9') p++;
        }
        /* Space followed by word - keep space with word */
        else if (*p == ' ' && p[1] && (isalpha(p[1]) || (unsigned char)p[1] >= 128)) {
            p++;  /* Include the space */
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
        /* Space followed by number */
        else if (*p == ' ' && p[1] >= '0' && p[1] <= '9') {
            p++;
            while (*p >= '0' && *p <= '9') p++;
        }
        /* Whitespace */
        else if (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t') {
            while (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t') p++;
        }
        /* Single character/punctuation */
        else {
            p++;
        }

        /* Add chunk */
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

/*
 * Tokenize text to token IDs.
 * Returns array of token IDs, caller must free.
 */
int *qwen3_tokenize(qwen3_tokenizer_t *tok, const char *text,
                    int *num_tokens, int max_len) {
    if (max_len <= 0) max_len = QWEN3_MAX_SEQ_LEN;

    /* Pre-tokenize */
    int num_chunks;
    char **chunks = pretokenize(text, &num_chunks);

    /* Tokenize each chunk */
    int capacity = 256;
    int *tokens = malloc(capacity * sizeof(int));
    int total = 0;

    for (int c = 0; c < num_chunks && total < max_len; c++) {
        /* Convert to byte-level encoding */
        char *byte_text = text_to_bytes(chunks[c]);
        if (!byte_text) continue;

        /* Apply BPE */
        token_node_t *bpe_tokens = bpe_encode_word(tok, byte_text);

        /* Convert to token IDs */
        for (token_node_t *node = bpe_tokens; node && total < max_len; node = node->next) {
            int id = vocab_hash_lookup(tok->vocab_hash, tok->hash_size, node->text);
            if (id >= 0) {
                if (total >= capacity) {
                    capacity *= 2;
                    tokens = realloc(tokens, capacity * sizeof(int));
                }
                tokens[total++] = id;
            }
        }

        free_token_list(bpe_tokens);
        free(byte_text);
        free(chunks[c]);
    }
    free(chunks);

    *num_tokens = total;
    return tokens;
}

/*
 * Apply chat template and tokenize.
 * Format: <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
 */
int *qwen3_tokenize_chat(qwen3_tokenizer_t *tok, const char *prompt,
                         int *num_tokens, int max_len) {
    if (max_len <= 0) max_len = QWEN3_MAX_SEQ_LEN;

    int capacity = 256;
    int *tokens = malloc(capacity * sizeof(int));
    int total = 0;

    /* Add special tokens and text */
    /* <|im_start|> */
    tokens[total++] = QWEN3_IM_START_ID;

    /* "user\n" */
    int n;
    int *user_tokens = qwen3_tokenize(tok, "user\n", &n, max_len - total);
    for (int i = 0; i < n && total < max_len; i++) {
        if (total >= capacity) {
            capacity *= 2;
            tokens = realloc(tokens, capacity * sizeof(int));
        }
        tokens[total++] = user_tokens[i];
    }
    free(user_tokens);

    /* prompt */
    int *prompt_tokens = qwen3_tokenize(tok, prompt, &n, max_len - total);
    for (int i = 0; i < n && total < max_len; i++) {
        if (total >= capacity) {
            capacity *= 2;
            tokens = realloc(tokens, capacity * sizeof(int));
        }
        tokens[total++] = prompt_tokens[i];
    }
    free(prompt_tokens);

    /* <|im_end|>\n */
    if (total < max_len) {
        if (total >= capacity) {
            capacity *= 2;
            tokens = realloc(tokens, capacity * sizeof(int));
        }
        tokens[total++] = QWEN3_IM_END_ID;
    }
    int *newline_tokens = qwen3_tokenize(tok, "\n", &n, max_len - total);
    for (int i = 0; i < n && total < max_len; i++) {
        if (total >= capacity) {
            capacity *= 2;
            tokens = realloc(tokens, capacity * sizeof(int));
        }
        tokens[total++] = newline_tokens[i];
    }
    free(newline_tokens);

    /* <|im_start|> */
    if (total < max_len) {
        if (total >= capacity) {
            capacity *= 2;
            tokens = realloc(tokens, capacity * sizeof(int));
        }
        tokens[total++] = QWEN3_IM_START_ID;
    }

    /* "assistant\n" */
    int *asst_tokens = qwen3_tokenize(tok, "assistant\n", &n, max_len - total);
    for (int i = 0; i < n && total < max_len; i++) {
        if (total >= capacity) {
            capacity *= 2;
            tokens = realloc(tokens, capacity * sizeof(int));
        }
        tokens[total++] = asst_tokens[i];
    }
    free(asst_tokens);

    /* <think>\n\n</think>\n\n */
    if (total < max_len) {
        if (total >= capacity) {
            capacity *= 2;
            tokens = realloc(tokens, capacity * sizeof(int));
        }
        tokens[total++] = QWEN3_THINK_START_ID;
    }
    int *think_newlines = qwen3_tokenize(tok, "\n\n", &n, max_len - total);
    for (int i = 0; i < n && total < max_len; i++) {
        if (total >= capacity) {
            capacity *= 2;
            tokens = realloc(tokens, capacity * sizeof(int));
        }
        tokens[total++] = think_newlines[i];
    }
    free(think_newlines);

    if (total < max_len) {
        if (total >= capacity) {
            capacity *= 2;
            tokens = realloc(tokens, capacity * sizeof(int));
        }
        tokens[total++] = QWEN3_THINK_END_ID;
    }

    think_newlines = qwen3_tokenize(tok, "\n\n", &n, max_len - total);
    for (int i = 0; i < n && total < max_len; i++) {
        if (total >= capacity) {
            capacity *= 2;
            tokens = realloc(tokens, capacity * sizeof(int));
        }
        tokens[total++] = think_newlines[i];
    }
    free(think_newlines);

    *num_tokens = total;
    return tokens;
}

/*
 * Pad token sequence to max_len with PAD tokens.
 * Returns new array, caller must free original.
 */
int *qwen3_pad_tokens(int *tokens, int num_tokens, int max_len, int *attention_mask) {
    int *padded = malloc(max_len * sizeof(int));
    if (!padded) return NULL;

    for (int i = 0; i < max_len; i++) {
        if (i < num_tokens) {
            padded[i] = tokens[i];
            if (attention_mask) attention_mask[i] = 1;
        } else {
            padded[i] = QWEN3_PAD_ID;
            if (attention_mask) attention_mask[i] = 0;
        }
    }

    return padded;
}

/* ========================================================================
 * Debug / Utility Functions
 * ======================================================================== */

/* Get token string by ID */
const char *qwen3_get_token(qwen3_tokenizer_t *tok, int id) {
    if (!tok || id < 0 || id >= tok->vocab_size) return NULL;
    return tok->vocab[id];
}

/* Get token ID by string */
int qwen3_get_id(qwen3_tokenizer_t *tok, const char *token) {
    if (!tok || !token) return -1;
    return vocab_hash_lookup(tok->vocab_hash, tok->hash_size, token);
}
