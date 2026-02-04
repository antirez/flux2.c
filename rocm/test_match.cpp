#include "flux_rocm.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cblas.h>

int main() {
    flux_rocm_init();
    
    // Match actual dimensions from flux
    const int seq = 512, in_dim = 3072, out_dim = 3072;
    
    float *x = (float*)malloc(seq * in_dim * sizeof(float));
    float *W = (float*)malloc(out_dim * in_dim * sizeof(float));
    float *y_blas = (float*)malloc(seq * out_dim * sizeof(float));
    float *y_rocm = (float*)malloc(seq * out_dim * sizeof(float));
    
    // Random init
    for (int i = 0; i < seq * in_dim; i++) x[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    for (int i = 0; i < out_dim * in_dim; i++) W[i] = (float)(rand() % 1000) / 10000.0f - 0.05f;
    
    // BLAS: y = x @ W^T
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                seq, out_dim, in_dim,
                1.0f, x, in_dim, W, in_dim,
                0.0f, y_blas, out_dim);
    
    // ROCm
    flux_rocm_tensor_t t_x = flux_rocm_tensor_create(x, seq * in_dim);
    flux_rocm_tensor_t t_y = flux_rocm_linear(t_x, W, seq, in_dim, out_dim);
    if (t_y) {
        flux_rocm_tensor_read(t_y, y_rocm, seq * out_dim);
        flux_rocm_tensor_free(t_y);
    }
    flux_rocm_tensor_free(t_x);
    
    // Compare
    float max_diff = 0, avg_diff = 0;
    int bad_count = 0;
    for (int i = 0; i < seq * out_dim; i++) {
        float diff = fabsf(y_blas[i] - y_rocm[i]);
        if (diff > max_diff) max_diff = diff;
        avg_diff += diff;
        if (diff > 0.01) bad_count++;
    }
    avg_diff /= (seq * out_dim);
    
    printf("BLAS first 5: %.4f %.4f %.4f %.4f %.4f\n", y_blas[0], y_blas[1], y_blas[2], y_blas[3], y_blas[4]);
    printf("ROCm first 5: %.4f %.4f %.4f %.4f %.4f\n", y_rocm[0], y_rocm[1], y_rocm[2], y_rocm[3], y_rocm[4]);
    printf("Max diff: %.6f, Avg diff: %.6f, Bad count: %d/%d\n", max_diff, avg_diff, bad_count, seq*out_dim);
    
    free(x); free(W); free(y_blas); free(y_rocm);
    flux_rocm_cleanup();
    return (max_diff > 0.01) ? 1 : 0;
}
