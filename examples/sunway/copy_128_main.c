#include <stdio.h>

#include "athread.h"

void copy_128(float *A, float *B);

static float input[128];
static float output[128];

int main(void) {
    int i;
    for (i = 0; i < 128; ++i) {
        input[i] = (float)i + 0.25f;
        output[i] = -1.0f;
    }

    /* A standalone -mhybrid executable owns CRTS initialization. */
    athread_init();
    copy_128(input, output);

    for (i = 0; i < 128; ++i) {
        if (output[i] != input[i]) {
            fprintf(stderr, "copy mismatch at %d: got=%f expected=%f\n", i, output[i], input[i]);
            return 1;
        }
    }
    printf("copy_128 passed: %d elements\n", 128);
    return 0;
}
