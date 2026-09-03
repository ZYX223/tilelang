#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <time.h>

#include "athread.h"

void gemm_128_k64(float *A, float *B, float *C);

enum { M = 128, N = 128, K = 64, MEASURED_RUNS = 7 };

static float A[M * K];
static float B[K * N];
static float C[M * N];
static float expected[M * N];

static double monotonic_ms(void) {
    struct timespec timestamp;
    if (clock_gettime(CLOCK_MONOTONIC, &timestamp) != 0) {
        return -1.0;
    }
    return (double)timestamp.tv_sec * 1000.0 + (double)timestamp.tv_nsec / 1000000.0;
}

static void sort_elapsed(double *values, int count) {
    int i;
    for (i = 1; i < count; ++i) {
        double value = values[i];
        int position = i;
        while (position > 0 && values[position - 1] > value) {
            values[position] = values[position - 1];
            --position;
        }
        values[position] = value;
    }
}

int main(void) {
    int index;
    int k;
    int m;
    int n;
    int run;
    double elapsed_ms[MEASURED_RUNS];

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
    /* Warm the launch/runtime path before collecting in-process kernel timing. */
    gemm_128_k64(A, B, C);
    for (run = 0; run < MEASURED_RUNS; ++run) {
        double started = monotonic_ms();
        double finished;
        if (started < 0.0) {
            fprintf(stderr, "clock_gettime(CLOCK_MONOTONIC) failed\n");
            return 1;
        }
        gemm_128_k64(A, B, C);
        finished = monotonic_ms();
        if (finished < 0.0) {
            fprintf(stderr, "clock_gettime(CLOCK_MONOTONIC) failed\n");
            return 1;
        }
        elapsed_ms[run] = finished - started;
    }

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
    sort_elapsed(elapsed_ms, MEASURED_RUNS);
    printf("gemm_128_k64 passed: M=128 N=128 K=64\n");
    printf(
        "gemm_128_k64 median_ms: %.6f over %d runs\n",
        elapsed_ms[MEASURED_RUNS / 2],
        MEASURED_RUNS
    );
    return 0;
}
