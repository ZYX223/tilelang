from __future__ import annotations

import tilelang.language as T


def make_gemm_32(*, workers: int = 64, num_stages: int = 1):
    @T.prim_func
    def gemm_32(
        A: T.Tensor((32, 32), "float32"),
        B: T.Tensor((32, 32), "float32"),
        C: T.Tensor((32, 32), "float32"),
    ):
        with T.Kernel(2, 2, threads=workers) as (bx, by):
            A_shared = T.alloc_shared((16, 32), "float32")
            B_shared = T.alloc_shared((32, 16), "float32")
            C_local = T.alloc_fragment((16, 16), "float32")

            T.clear(C_local)
            for ko in T.Pipelined(1, num_stages=num_stages):
                T.copy(A[by * 16, ko * 32], A_shared)
                T.copy(B[ko * 32, bx * 16], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * 16, bx * 16])

    return gemm_32


def make_gemm_m32_n16_k32(*, workers: int = 64, num_stages: int = 1):
    @T.prim_func
    def gemm_m32_n16_k32(
        A: T.Tensor((32, 32), "float32"),
        B: T.Tensor((32, 16), "float32"),
        C: T.Tensor((32, 16), "float32"),
    ):
        with T.Kernel(1, 2, threads=workers) as (bx, by):
            A_shared = T.alloc_shared((16, 32), "float32")
            B_shared = T.alloc_shared((32, 16), "float32")
            C_local = T.alloc_fragment((16, 16), "float32")

            T.clear(C_local)
            for ko in T.Pipelined(1, num_stages=num_stages):
                T.copy(A[by * 16, ko * 32], A_shared)
                T.copy(B[ko * 32, bx * 16], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * 16, bx * 16])

    return gemm_m32_n16_k32
