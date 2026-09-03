#include "slave.h"
#include "simd.h"
#include "simd_vmas_f32x8_common.h"

__thread_local simd_vmas_f32x8_args_t ldm_args;
volatile __thread_local int reply;
__thread_local float b_local[8] __attribute__((aligned(32)));
__thread_local float c_local[8] __attribute__((aligned(32)));
__thread_local float out_local[8] __attribute__((aligned(32)));

void simd_vmas_f32x8_cpe(simd_vmas_f32x8_args_t *global_args) {
    floatv8 a_vector;
    floatv8 b_vector;
    floatv8 c_vector;

    reply = 0;
    athread_get(
        PE_MODE,
        global_args,
        &ldm_args,
        sizeof(simd_vmas_f32x8_args_t),
        (void *)&reply,
        0,
        0,
        0
    );
    while (reply != 1) {
    }

    /* One CPE is enough to validate the compiler and native SIMD contract. */
    if (_MYID != 0) {
        return;
    }

    reply = 0;
    athread_get(PE_MODE, ldm_args.b, b_local, sizeof(b_local), (void *)&reply, 0, 0, 0);
    while (reply != 1) {
    }
    reply = 0;
    athread_get(PE_MODE, ldm_args.c, c_local, sizeof(c_local), (void *)&reply, 0, 0, 0);
    while (reply != 1) {
    }

    a_vector = simd_set_floatv8(
        ldm_args.a,
        ldm_args.a,
        ldm_args.a,
        ldm_args.a,
        ldm_args.a,
        ldm_args.a,
        ldm_args.a,
        ldm_args.a
    );
    b_vector = *(floatv8 *)b_local;
    c_vector = *(floatv8 *)c_local;
    *(floatv8 *)out_local = simd_vmas(a_vector, b_vector, c_vector);

    reply = 0;
    athread_put(PE_MODE, out_local, ldm_args.out, sizeof(out_local), (void *)&reply, 0, 0);
    while (reply != 1) {
    }
}
