#include <stdio.h>

#include "athread.h"

void gemm_128_k64(float *A, float *B, float *C);

enum { M = 128, N = 128, K = 64 };

static float A[M * K];
static float B[K * N];
static float C[M * N];
static float expected[M * N];

int main(void) {
    int index;
    int k;
    int m;
    int n;

    for (index = 0; index < M * K; ++index) {
        A[index] = (float)((index % 17) - 8) * 0.125f;
    }
    for (index = 0; index < K * N; ++index) {
        B[index] = (float)((index % 13) - 6) * 0.0625f;
    }
    for (index = 0; index < M * N; ++index) {
        C[index] = -1.0f;
        expected[index] = 0.0f;
    }
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            for (int k = 0; k < K; ++k) {
                expected[m * N + n] += A[m * K + k] * B[k * N + n];
            }
        }
    }

    /* A standalone -mhybrid executable owns CRTS initialization. */
    athread_init();
    gemm_128_k64(A, B, C);

    for (m = 0; m < M; ++m) {
        for (n = 0; n < N; ++n) {
            float scale;
            float diff;
            index = m * N + n;
            scale = expected[index] < 0.0f ? -expected[index] : expected[index];
            diff = C[index] - expected[index];
            scale = scale < 1.0f ? 1.0f : scale;
            diff = diff < 0.0f ? -diff : diff;
            if (diff > 1.0e-4f * scale) {
                fprintf(
                    stderr,
                    "gemm mismatch at (%d,%d): got=%f expected=%f\n",
                    m,
                    n,
                    C[index],
                    expected[index]
                );
                return 1;
            }
        }
    }
    printf("gemm_128_k64 passed: M=128 N=128 K=64\n");
    return 0;
}
