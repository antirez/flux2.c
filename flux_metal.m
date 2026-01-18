/*
 * FLUX Metal Acceleration - Implementation
 *
 * Uses Metal Performance Shaders (MPS) for GPU-accelerated matrix operations.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include "flux_metal.h"
#include <stdio.h>

/* Global Metal state */
static id<MTLDevice> g_device = nil;
static id<MTLCommandQueue> g_queue = nil;
static int g_initialized = 0;

int flux_metal_init(void) {
    if (g_initialized) return 1;

    @autoreleasepool {
        /* Get default Metal device */
        g_device = MTLCreateSystemDefaultDevice();
        if (!g_device) {
            fprintf(stderr, "Metal: No GPU device found\n");
            return 0;
        }

        /* Check if this is Apple Silicon (not Intel) */
        if (![g_device supportsFamily:MTLGPUFamilyApple7]) {
            /* MTLGPUFamilyApple7 is M1 and later */
            /* Fall back to Apple6 (A14) or check for MPS support */
            if (![g_device supportsFamily:MTLGPUFamilyApple6]) {
                fprintf(stderr, "Metal: GPU does not support required features\n");
                g_device = nil;
                return 0;
            }
        }

        /* Create command queue */
        g_queue = [g_device newCommandQueue];
        if (!g_queue) {
            fprintf(stderr, "Metal: Failed to create command queue\n");
            g_device = nil;
            return 0;
        }

        g_initialized = 1;
        fprintf(stderr, "Metal: Initialized with %s\n", [[g_device name] UTF8String]);
    }

    return 1;
}

int flux_metal_available(void) {
    return g_initialized;
}

void flux_metal_cleanup(void) {
    if (!g_initialized) return;

    @autoreleasepool {
        g_queue = nil;
        g_device = nil;
        g_initialized = 0;
    }
}

void flux_metal_sgemm(int transpose_a, int transpose_b,
                      int M, int N, int K,
                      float alpha,
                      const float *A, int lda,
                      const float *B, int ldb,
                      float beta,
                      float *C, int ldc) {
    if (!g_initialized) return;

    @autoreleasepool {
        /* Compute actual matrix dimensions accounting for transpose */
        int rowsA = transpose_a ? K : M;
        int colsA = transpose_a ? M : K;
        int rowsB = transpose_b ? N : K;
        int colsB = transpose_b ? K : N;

        /* Create Metal buffers */
        size_t sizeA = (size_t)rowsA * lda * sizeof(float);
        size_t sizeB = (size_t)rowsB * ldb * sizeof(float);
        size_t sizeC = (size_t)M * ldc * sizeof(float);

        id<MTLBuffer> bufferA = [g_device newBufferWithBytes:A
                                                      length:sizeA
                                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferB = [g_device newBufferWithBytes:B
                                                      length:sizeB
                                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferC = [g_device newBufferWithBytes:C
                                                      length:sizeC
                                                     options:MTLResourceStorageModeShared];

        /* Create matrix descriptors */
        MPSMatrixDescriptor *descA = [MPSMatrixDescriptor
            matrixDescriptorWithRows:rowsA
                             columns:colsA
                            rowBytes:lda * sizeof(float)
                            dataType:MPSDataTypeFloat32];

        MPSMatrixDescriptor *descB = [MPSMatrixDescriptor
            matrixDescriptorWithRows:rowsB
                             columns:colsB
                            rowBytes:ldb * sizeof(float)
                            dataType:MPSDataTypeFloat32];

        MPSMatrixDescriptor *descC = [MPSMatrixDescriptor
            matrixDescriptorWithRows:M
                             columns:N
                            rowBytes:ldc * sizeof(float)
                            dataType:MPSDataTypeFloat32];

        /* Create MPS matrices */
        MPSMatrix *matrixA = [[MPSMatrix alloc] initWithBuffer:bufferA descriptor:descA];
        MPSMatrix *matrixB = [[MPSMatrix alloc] initWithBuffer:bufferB descriptor:descB];
        MPSMatrix *matrixC = [[MPSMatrix alloc] initWithBuffer:bufferC descriptor:descC];

        /* Create matrix multiplication kernel */
        MPSMatrixMultiplication *matmul = [[MPSMatrixMultiplication alloc]
            initWithDevice:g_device
               transposeLeft:transpose_a ? YES : NO
              transposeRight:transpose_b ? YES : NO
                  resultRows:M
               resultColumns:N
             interiorColumns:K
                       alpha:alpha
                        beta:beta];

        /* Create command buffer and encode */
        id<MTLCommandBuffer> cmdBuffer = [g_queue commandBuffer];
        [matmul encodeToCommandBuffer:cmdBuffer
                           leftMatrix:matrixA
                          rightMatrix:matrixB
                         resultMatrix:matrixC];

        /* Execute and wait */
        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        /* Copy result back */
        memcpy(C, [bufferC contents], sizeC);
    }
}

void flux_metal_sgemm_batch(int transpose_a, int transpose_b,
                            int M, int N, int K,
                            float alpha,
                            const float *A, int lda, int stride_a,
                            const float *B, int ldb, int stride_b,
                            float beta,
                            float *C, int ldc, int stride_c,
                            int batch_count) {
    /* For batched operations, execute sequentially for now */
    /* TODO: Use MPSMatrixMultiplication with batched matrices for better performance */
    for (int i = 0; i < batch_count; i++) {
        flux_metal_sgemm(transpose_a, transpose_b,
                         M, N, K, alpha,
                         A + i * stride_a, lda,
                         B + i * stride_b, ldb,
                         beta,
                         C + i * stride_c, ldc);
    }
}

void flux_metal_sync(void) {
    /* All our operations are synchronous for now */
}

size_t flux_metal_memory_used(void) {
    if (!g_initialized || !g_device) return 0;
    return [g_device currentAllocatedSize];
}
