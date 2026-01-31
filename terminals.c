/*
 * terminals.c - Terminal graphics protocol support
 *
 * Supports multiple terminal graphics protocols:
 *   - Kitty graphics protocol (Kitty, Ghostty)
 *   - iTerm2 inline image protocol
 *
 * The unified API (terminal_display_*) auto-detects and uses the appropriate
 * protocol based on environment variables.
 */

#include "terminals.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* ======================================================================
 * Terminal Detection
 * ====================================================================== */

/*
 * Detect terminal graphics capability from environment variables.
 * Checks for Kitty, Ghostty, and iTerm2.
 */
term_graphics_proto detect_terminal_graphics(void) {
    /* Kitty: KITTY_WINDOW_ID is set */
    if (getenv("KITTY_WINDOW_ID"))
        return TERM_PROTO_KITTY;

    /* Ghostty: GHOSTTY_RESOURCES_DIR is set (uses Kitty protocol) */
    if (getenv("GHOSTTY_RESOURCES_DIR"))
        return TERM_PROTO_KITTY;

    /* iTerm2: TERM_PROGRAM=iTerm.app or ITERM_SESSION_ID is set */
    const char *term_program = getenv("TERM_PROGRAM");
    if ((term_program && strcmp(term_program, "iTerm.app") == 0) ||
        getenv("ITERM_SESSION_ID"))
        return TERM_PROTO_ITERM2;

    return TERM_PROTO_NONE;
}

/* ======================================================================
 * Base64 Encoding (shared by all protocols)
 * ====================================================================== */

static const char b64_table[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/* Base64 encode data, returns malloc'd string (caller must free) */
static char *base64_encode(const unsigned char *data, size_t len, size_t *out_len) {
    size_t encoded_len = 4 * ((len + 2) / 3);
    char *encoded = malloc(encoded_len + 1);
    if (!encoded) return NULL;

    size_t i, j;
    for (i = 0, j = 0; i < len; ) {
        uint32_t a = i < len ? data[i++] : 0;
        uint32_t b = i < len ? data[i++] : 0;
        uint32_t c = i < len ? data[i++] : 0;
        uint32_t triple = (a << 16) | (b << 8) | c;

        encoded[j++] = b64_table[(triple >> 18) & 0x3F];
        encoded[j++] = b64_table[(triple >> 12) & 0x3F];
        encoded[j++] = b64_table[(triple >> 6) & 0x3F];
        encoded[j++] = b64_table[triple & 0x3F];
    }

    /* Add padding */
    int pad = len % 3;
    if (pad) {
        encoded[encoded_len - 1] = '=';
        if (pad == 1) encoded[encoded_len - 2] = '=';
    }

    encoded[encoded_len] = '\0';
    if (out_len) *out_len = encoded_len;
    return encoded;
}

/* ======================================================================
 * Kitty Graphics Protocol
 *
 * Format: \033_G<control>;<base64-payload>\033\\
 * Supports raw pixel data (f=24 RGB, f=32 RGBA) and PNG (f=100).
 * Data is sent in chunks to avoid terminal buffer issues.
 * ====================================================================== */

/*
 * Send data using Kitty graphics protocol in chunks.
 * format: 24=RGB, 32=RGBA, 100=PNG
 */
static int kitty_send_data(const unsigned char *data, size_t size,
                           int format, int width, int height) {
    size_t b64_len;
    char *b64_data = base64_encode(data, size, &b64_len);
    if (!b64_data) return -1;

    /* Send in chunks (4096 bytes of base64 per chunk is safe) */
    const size_t chunk_size = 4096;
    size_t offset = 0;
    int first = 1;

    while (offset < b64_len) {
        size_t remaining = b64_len - offset;
        size_t this_chunk = remaining < chunk_size ? remaining : chunk_size;
        int more = (offset + this_chunk) < b64_len;

        if (first) {
            /* First chunk: a=T (transmit+display), f=format, t=d (direct) */
            if (format == 100) {
                /* PNG: no dimensions needed */
                printf("\033_Ga=T,f=100,t=d,m=%d;", more ? 1 : 0);
            } else {
                /* Raw: include dimensions */
                printf("\033_Ga=T,f=%d,s=%d,v=%d,m=%d;",
                       format, width, height, more ? 1 : 0);
            }
            first = 0;
        } else {
            /* Continuation chunk */
            printf("\033_Gm=%d;", more ? 1 : 0);
        }

        fwrite(b64_data + offset, 1, this_chunk, stdout);
        printf("\033\\");
        offset += this_chunk;
    }

    fflush(stdout);
    free(b64_data);
    return 0;
}

int kitty_display_png(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "kitty: cannot open %s\n", path);
        return -1;
    }

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (size <= 0) {
        fclose(f);
        return -1;
    }

    unsigned char *png_data = malloc(size);
    if (!png_data) {
        fclose(f);
        return -1;
    }

    if (fread(png_data, 1, size, f) != (size_t)size) {
        free(png_data);
        fclose(f);
        return -1;
    }
    fclose(f);

    int result = kitty_send_data(png_data, size, 100, 0, 0);
    free(png_data);
    printf("\n");
    return result;
}

int kitty_display_image(const flux_image *img) {
    if (!img || !img->data) return -1;

    size_t data_size = (size_t)img->width * img->height * img->channels;
    int format = (img->channels == 4) ? 32 : 24;

    int result = kitty_send_data(img->data, data_size, format,
                                  img->width, img->height);
    printf("\n");
    return result;
}

/* ======================================================================
 * iTerm2 Inline Image Protocol
 *
 * Format: \033]1337;File=inline=1:<base64-payload>\a
 * Only supports encoded image formats (PNG, JPEG, etc.), not raw pixels.
 * See: https://iterm2.com/documentation-images.html
 * ====================================================================== */

static int iterm2_send_png(const unsigned char *png_data, size_t png_size) {
    size_t b64_len;
    char *b64_data = base64_encode(png_data, png_size, &b64_len);
    if (!b64_data) return -1;

    /* iTerm2 protocol: OSC 1337 ; File=inline=1 : <base64> BEL */
    printf("\033]1337;File=inline=1:");
    fwrite(b64_data, 1, b64_len, stdout);
    printf("\a\n");
    fflush(stdout);

    free(b64_data);
    return 0;
}

int iterm2_display_png(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "iterm2: cannot open %s\n", path);
        return -1;
    }

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (size <= 0) {
        fclose(f);
        return -1;
    }

    unsigned char *png_data = malloc(size);
    if (!png_data) {
        fclose(f);
        return -1;
    }

    if (fread(png_data, 1, size, f) != (size_t)size) {
        free(png_data);
        fclose(f);
        return -1;
    }
    fclose(f);

    int result = iterm2_send_png(png_data, size);
    free(png_data);
    return result;
}

/*
 * Display raw image data in iTerm2.
 * iTerm2 requires PNG format, so we encode to PNG via a temp file.
 * This is transparent to the caller - API matches kitty_display_image().
 */
int iterm2_display_image(const flux_image *img) {
    if (!img || !img->data) return -1;

    /* Create temp file for PNG */
    char tmppath[] = "/tmp/flux_iterm_XXXXXX.png";
    int fd = mkstemps(tmppath, 4);
    if (fd < 0) {
        fprintf(stderr, "iterm2: cannot create temp file\n");
        return -1;
    }
    close(fd);

    /* Save image as PNG */
    if (flux_image_save(img, tmppath) != 0) {
        unlink(tmppath);
        return -1;
    }

    /* Display the PNG */
    int result = iterm2_display_png(tmppath);

    /* Clean up */
    unlink(tmppath);
    return result;
}

/* ======================================================================
 * Unified Terminal API
 *
 * These functions automatically use the appropriate protocol.
 * ====================================================================== */

int terminal_display_png(const char *path, term_graphics_proto proto) {
    switch (proto) {
        case TERM_PROTO_KITTY:
            return kitty_display_png(path);
        case TERM_PROTO_ITERM2:
            return iterm2_display_png(path);
        default:
            return -1;
    }
}

int terminal_display_image(const flux_image *img, term_graphics_proto proto) {
    switch (proto) {
        case TERM_PROTO_KITTY:
            return kitty_display_image(img);
        case TERM_PROTO_ITERM2:
            return iterm2_display_image(img);
        default:
            return -1;
    }
}

/* ==================================================================
 * OSC 9.4 Progress Protocol
 * Format: ESC ] 9 ; 4 ; state ; value ST
 * ST (String Terminator) = ESC \ or BEL
 * ================================================================== */

void terminal_progress_remove(void) {
    printf("\033]9;4;0\033\\");
    fflush(stdout);
}

void terminal_progress_set(int percent) {
    if (percent < 0) percent = 0;
    if (percent > 100) percent = 100;
    printf("\033]9;4;1;%d\033\\", percent);
    fflush(stdout);
}

void terminal_progress_error(void) {
    printf("\033]9;4;2\033\\");
    fflush(stdout);
}

void terminal_progress_indeterminate(void) {
    printf("\033]9;4;3\033\\");
    fflush(stdout);
}

void terminal_progress_paused(void) {
    printf("\033]9;4;4\033\\");
    fflush(stdout);
}
