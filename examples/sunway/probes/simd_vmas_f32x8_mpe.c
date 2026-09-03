#include <stdio.h>

#include "athread.h"
#include "simd_vmas_f32x8_common.h"

extern SLAVE_FUN(simd_vmas_f32x8_cpe)(simd_vmas_f32x8_args_t *);

static float b[8] __attribute__((aligned(32)));
static float c[8] __attribute__((aligned(32)));
static float out[8] __attribute__((aligned(32)));

int main(void) {
    simd_vmas_f32x8_args_t args;
    int lane;

    args.a = 2.0f;
    args.b = b;
    args.c = c;
    args.out = out;
    for (lane = 0; lane < 8; ++lane) {
        b[lane] = (float)(lane + 1) * 0.5f;
        c[lane] = (float)lane * 0.25f;
        out[lane] = -1.0f;
    }

    athread_init();
    athread_spawn(simd_vmas_f32x8_cpe, &args);
    athread_join();

    for (lane = 0; lane < 8; ++lane) {
        float expected = args.a * b[lane] + c[lane];
        if (out[lane] != expected) {
            fprintf(
                stderr,
                "simd_vmas mismatch at lane %d: got=%f expected=%f\n",
                lane,
                out[lane],
                expected
            );
            return 1;
        }
    }
    printf("simd_vmas_f32x8 passed: 8 lanes\n");
    return 0;
}
