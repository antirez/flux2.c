/*
 * JPEG Decoder Fuzzer
 *
 * Performs mutation-based fuzzing on JPEG files to find crashes and
 * memory corruption bugs in the decoder.
 *
 * Uses fork() to isolate each decode attempt, preventing memory corruption
 * from affecting the fuzzer state.
 *
 * Usage: ./fuzz_jpeg [iterations] [seed]
 */

#define JPEG_IMPLEMENTATION
#include "../jpeg.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <time.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/wait.h>

/* Statistics */
static uint64_t total_iterations = 0;
static uint64_t total_crashes = 0;
static uint64_t total_loads = 0;
static uint64_t total_nulls = 0;

/* Source images to fuzz - variety of baseline, progressive, grayscale, subsampling */
static const char *source_images[] = {
    /* Baseline RGB */
    "testorig.jpg",
    "testimgint.jpg",
    "testorig_444.jpg",
    "testorig_420.jpg",
    "sample.jpg",
    "hopper.jpg",
    "wizard.jpg",
    /* Grayscale baseline */
    "cd1.1.jpg",
    "cd2.1.jpg",
    /* Progressive RGB */
    "testorig_prog.jpg",
    "monkey_prog.jpg",
    /* Progressive grayscale */
    "gray_prog.jpg",
};
static const int num_sources = sizeof(source_images) / sizeof(source_images[0]);

/* Read file into memory */
static uint8_t *read_file(const char *path, size_t *out_size) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;

    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    fseek(f, 0, SEEK_SET);

    uint8_t *data = malloc(size);
    if (!data) {
        fclose(f);
        return NULL;
    }

    if (fread(data, 1, size, f) != size) {
        free(data);
        fclose(f);
        return NULL;
    }

    fclose(f);
    *out_size = size;
    return data;
}

/* Write file from memory */
static int write_file(const char *path, const uint8_t *data, size_t size) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;

    if (fwrite(data, 1, size, f) != size) {
        fclose(f);
        return -1;
    }

    fclose(f);
    return 0;
}

/* Mutation strategies */
typedef enum {
    MUT_FLIP_BIT,        /* Flip a single bit */
    MUT_FLIP_BYTE,       /* Flip all bits in a byte */
    MUT_RANDOM_BYTE,     /* Replace byte with random value */
    MUT_ZERO_BYTE,       /* Set byte to zero */
    MUT_FF_BYTE,         /* Set byte to 0xFF */
    MUT_DELETE_BYTES,    /* Delete 1-16 bytes */
    MUT_INSERT_BYTES,    /* Insert 1-16 random bytes */
    MUT_SWAP_BYTES,      /* Swap two bytes */
    MUT_REPEAT_REGION,   /* Repeat a region */
    MUT_ARITHMETIC,      /* Add/subtract small value */
    MUT_MULTI_BIT,       /* Flip multiple bits */
    MUT_TRUNCATE,        /* Truncate file */
    MUT_COUNT
} mutation_type;

/* Apply a random mutation to the data */
static size_t mutate(uint8_t *data, size_t size, size_t max_size, mutation_type mut) {
    if (size == 0) return size;

    size_t pos = rand() % size;
    int count, i;
    size_t pos2, len;
    uint8_t tmp;

    switch (mut) {
    case MUT_FLIP_BIT:
        data[pos] ^= (1 << (rand() % 8));
        break;

    case MUT_FLIP_BYTE:
        data[pos] ^= 0xFF;
        break;

    case MUT_RANDOM_BYTE:
        data[pos] = rand() & 0xFF;
        break;

    case MUT_ZERO_BYTE:
        data[pos] = 0;
        break;

    case MUT_FF_BYTE:
        data[pos] = 0xFF;
        break;

    case MUT_DELETE_BYTES:
        count = 1 + (rand() % 16);
        if (pos + count > size) count = size - pos;
        if (count > 0 && pos + count < size) {
            memmove(data + pos, data + pos + count, size - pos - count);
            size -= count;
        }
        break;

    case MUT_INSERT_BYTES:
        count = 1 + (rand() % 16);
        if (size + count <= max_size) {
            memmove(data + pos + count, data + pos, size - pos);
            for (i = 0; i < count; i++) {
                data[pos + i] = rand() & 0xFF;
            }
            size += count;
        }
        break;

    case MUT_SWAP_BYTES:
        pos2 = rand() % size;
        tmp = data[pos];
        data[pos] = data[pos2];
        data[pos2] = tmp;
        break;

    case MUT_REPEAT_REGION:
        len = 1 + (rand() % 32);
        if (pos + len > size) len = size - pos;
        pos2 = rand() % size;
        if (pos2 + len <= size) {
            memmove(data + pos2, data + pos, len);  /* Use memmove for potential overlap */
        }
        break;

    case MUT_ARITHMETIC:
        data[pos] = (data[pos] + (rand() % 71) - 35) & 0xFF;
        break;

    case MUT_MULTI_BIT:
        count = 2 + (rand() % 7);  /* 2-8 bits */
        for (i = 0; i < count; i++) {
            size_t bit_pos = rand() % (size * 8);
            data[bit_pos / 8] ^= (1 << (bit_pos % 8));
        }
        break;

    case MUT_TRUNCATE:
        /* Truncate to 10%-90% of original */
        size = (size * (10 + rand() % 80)) / 100;
        if (size < 2) size = 2;
        break;

    default:
        break;
    }

    return size;
}

/* Print progress */
static void print_progress(int force) {
    static time_t last_print = 0;
    time_t now = time(NULL);

    if (force || now > last_print) {
        printf("\r[%llu iterations] loads: %llu, nulls: %llu, crashes: %llu   ",
               (unsigned long long)total_iterations,
               (unsigned long long)total_loads,
               (unsigned long long)total_nulls,
               (unsigned long long)total_crashes);
        fflush(stdout);
        last_print = now;
    }
}

/* Save crashing input for later analysis */
static void save_crash(const uint8_t *data, size_t size, int sig, uint64_t iter) {
    char filename[256];
    snprintf(filename, sizeof(filename), "crash_%d_%llu.jpg",
             sig, (unsigned long long)iter);
    if (write_file(filename, data, size) == 0) {
        printf("\n[!] Saved crashing input to %s\n", filename);
    }
}

/* Test one mutated input in a forked child process */
static int test_decode(const uint8_t *data, size_t size) {
    pid_t pid = fork();

    if (pid < 0) {
        perror("fork");
        return -1;  /* Fork failed */
    }

    if (pid == 0) {
        /* Child process - try to decode */
        jpeg_image *img = jpeg_load_mem(data, size);
        if (img) {
            /* Access the data to catch lazy crashes */
            volatile uint8_t sum = 0;
            for (size_t i = 0; i < (size_t)(img->width * img->height * img->channels); i += 1024) {
                sum += img->data[i];
            }
            (void)sum;
            jpeg_free(img);
            _exit(0);  /* Success */
        } else {
            _exit(1);  /* Returned NULL */
        }
    }

    /* Parent process - wait for child */
    int status;
    waitpid(pid, &status, 0);

    if (WIFEXITED(status)) {
        return WEXITSTATUS(status);  /* 0 = loaded, 1 = NULL */
    } else if (WIFSIGNALED(status)) {
        return 128 + WTERMSIG(status);  /* Crashed with signal */
    }

    return -1;
}

/* Run fuzzing loop */
static void fuzz(uint64_t iterations, unsigned int seed) {
    printf("JPEG Decoder Fuzzer\n");
    printf("===================\n");
    printf("Iterations: %llu\n", (unsigned long long)iterations);
    printf("Seed: %u\n", seed);
    printf("Source images: %d\n\n", num_sources);

    srand(seed);

    /* Load all source images into memory */
    uint8_t **sources = malloc(num_sources * sizeof(uint8_t *));
    size_t *source_sizes = malloc(num_sources * sizeof(size_t));
    if (!sources || !source_sizes) {
        fprintf(stderr, "Failed to allocate source arrays\n");
        exit(1);
    }

    for (int i = 0; i < num_sources; i++) {
        sources[i] = read_file(source_images[i], &source_sizes[i]);
        if (!sources[i]) {
            fprintf(stderr, "Failed to load %s\n", source_images[i]);
            exit(1);
        }
        printf("Loaded %s (%zu bytes)\n", source_images[i], source_sizes[i]);
    }
    printf("\nFuzzing...\n\n");

    /* Allocate mutation buffer (2x largest source for insertions) */
    size_t max_source = 0;
    for (int i = 0; i < num_sources; i++) {
        if (source_sizes[i] > max_source) max_source = source_sizes[i];
    }
    size_t buf_size = max_source * 2;
    uint8_t *buf = malloc(buf_size);
    if (!buf) {
        fprintf(stderr, "Failed to allocate mutation buffer\n");
        exit(1);
    }

    for (uint64_t iter = 0; iter < iterations; iter++) {
        total_iterations = iter + 1;

        /* Pick a random source */
        int src_idx = rand() % num_sources;
        size_t size = source_sizes[src_idx];
        memcpy(buf, sources[src_idx], size);

        /* Apply 1-5 random mutations */
        int num_mutations = 1 + (rand() % 5);
        for (int m = 0; m < num_mutations; m++) {
            mutation_type mut = rand() % MUT_COUNT;
            size = mutate(buf, size, buf_size, mut);
        }

        /* Test the mutated JPEG in a child process */
        int result = test_decode(buf, size);

        if (result == 0) {
            total_loads++;
        } else if (result == 1) {
            total_nulls++;
        } else if (result > 128) {
            /* Crashed with signal */
            int sig = result - 128;
            total_crashes++;
            printf("\n[CRASH] Signal %d at iteration %llu\n", sig, (unsigned long long)iter);
            save_crash(buf, size, sig, iter);
        }

        print_progress(0);
    }

    print_progress(1);
    printf("\n\nFuzzing complete.\n");
    printf("Total: %llu iterations, %llu successful loads, %llu nulls, %llu crashes\n",
           (unsigned long long)total_iterations,
           (unsigned long long)total_loads,
           (unsigned long long)total_nulls,
           (unsigned long long)total_crashes);

    /* Cleanup */
    free(buf);
    for (int i = 0; i < num_sources; i++) {
        free(sources[i]);
    }
    free(sources);
    free(source_sizes);
}

int main(int argc, char **argv) {
    uint64_t iterations = 100000;  /* Default 100k iterations */
    unsigned int seed = (unsigned int)time(NULL);

    if (argc > 1) {
        iterations = strtoull(argv[1], NULL, 10);
    }
    if (argc > 2) {
        seed = (unsigned int)atoi(argv[2]);
    }

    fuzz(iterations, seed);
    return total_crashes > 0 ? 1 : 0;
}
