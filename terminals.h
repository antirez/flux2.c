/*
 * terminals.h - Terminal graphics protocol support
 *
 * Supports Kitty graphics protocol (Kitty, Ghostty) and iTerm2 inline images.
 */

#ifndef TERMINALS_H
#define TERMINALS_H

#include "flux.h"

/* ======================================================================
 * Terminal Protocol Types
 * ====================================================================== */

typedef enum {
    TERM_PROTO_NONE = 0, /* No terminal graphics support detected */
    TERM_PROTO_KITTY,    /* Kitty graphics protocol (also used by Ghostty) */
    TERM_PROTO_ITERM2    /* iTerm2 inline image protocol */
} term_graphics_proto;

/* ======================================================================
 * Terminal Detection
 * ====================================================================== */

/*
 * Detect terminal graphics capability from environment variables.
 * Returns the appropriate protocol, or TERM_PROTO_NONE if not detected.
 */
term_graphics_proto detect_terminal_graphics(void);

/* ======================================================================
 * Kitty Graphics Protocol
 * ====================================================================== */

/*
 * Display PNG file using Kitty graphics protocol.
 * Returns 0 on success, -1 on error.
 */
int kitty_display_png(const char *path);

/*
 * Display raw image data using Kitty graphics protocol.
 * Sends raw RGB/RGBA pixels directly (no encoding needed).
 * Returns 0 on success, -1 on error.
 */
int kitty_display_image(const flux_image *img);

/* ======================================================================
 * iTerm2 Inline Image Protocol
 * ====================================================================== */

/*
 * Display PNG file using iTerm2 inline image protocol.
 * Returns 0 on success, -1 on error.
 */
int iterm2_display_png(const char *path);

/*
 * Display raw image data using iTerm2 inline image protocol.
 * Internally encodes to PNG (iTerm2 requires an image format).
 * Returns 0 on success, -1 on error.
 */
int iterm2_display_image(const flux_image *img);

/* ======================================================================
 * Zoom Setting
 * ====================================================================== */

/*
 * Set the zoom factor for terminal image display (default: 2 for Retina).
 * Values: 1 = native pixels, 2 = 2x (Retina), 3 = 3x, etc.
 */
void terminal_set_zoom(int zoom);

/* ======================================================================
 * Unified Terminal API
 * ====================================================================== */

/*
 * Display PNG file using the specified protocol.
 * Returns 0 on success, -1 on error.
 */
int terminal_display_png(const char *path, term_graphics_proto proto);

/*
 * Display raw image data using the specified protocol.
 * For Kitty: sends raw pixels directly.
 * For iTerm2: internally encodes to PNG first.
 * Returns 0 on success, -1 on error.
 */
int terminal_display_image(const flux_image *img, term_graphics_proto proto);

/* ==================================================================
 * OSC 9.4 Progress Protocol (ConEmu/Windows Terminal)
 * Format: ESC ] 9 ; 4 ; st ; pr ST
 * ================================================================== */

/* Remove progress indicator */
void terminal_progress_remove(void);

/* Set progress value (0-100) */
void terminal_progress_set(int percent);

/* Set error state in progress */
void terminal_progress_error(void);

/* Set indeterminate progress state */
void terminal_progress_indeterminate(void);

/* Set paused state in progress */
void terminal_progress_paused(void);

#endif /* TERMINALS_H */
