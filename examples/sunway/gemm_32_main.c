#include <stdio.h>

#include "athread.h"

void gemm_32(float *A, float *B, float *C);

enum { M = 32, N = 32, K = 32 };

static float A[M * K];
static float B[K * N];
static float C[M * N];
static float expected[M * N];

int main(void) {
    int i;
    int j;
    int k;
    int index;

    for (index = 0; index < M * K; ++index) {
        A[index] = (float)((index % 11) - 5) * 0.125f;
    }
    for (index = 0; index < K * N; ++index) {
        B[index] = (float)((index % 7) - 3) * 0.0625f;
    }
    for (index = 0; index < M * N; ++index) {
        C[index] = -1.0f;
        expected[index] = 0.0f;
    }
    for (i = 0; i < M; ++i) {
        for (j = 0; j < N; ++j) {
            for (k = 0; k < K; ++k) {
                expected[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }

    /* A standalone -mhybrid executable owns CRTS initialization. */
    athread_init();
    gemm_32(A, B, C);

    for (i = 0; i < M; ++i) {
        for (j = 0; j < N; ++j) {
            float scale;
            float diff;
            index = i * N + j;
            scale = expected[index] < 0.0f ? -expected[index] : expected[index];
            diff = C[index] - expected[index];
            scale = scale < 1.0f ? 1.0f : scale;
            diff = diff < 0.0f ? -diff : diff;
            if (diff > 1.0e-4f * scale) {
                fprintf(
                    stderr,
                    "gemm mismatch at (%d,%d): got=%f expected=%f\n",
                    i,
                    j,
                    C[index],
                    expected[index]
                );
                return 1;
            }
        }
    }
    printf("gemm_32 passed: M=32 N=32 K=32\n");
    return 0;
}
