"""Generate the square scalar-GEMM Sunway AOT project."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import tilelang
import tilelang.language as T


@T.prim_func
def gemm_32(
    A: T.Tensor((32, 32), "float32"),
    B: T.Tensor((32, 32), "float32"),
    C: T.Tensor((32, 32), "float32"),
):
    with T.Kernel(2, 2, threads=64) as (bx, by):
        A_shared = T.alloc_shared((16, 32), "float32")
        B_shared = T.alloc_shared((32, 16), "float32")
        C_local = T.alloc_fragment((16, 16), "float32")

        T.clear(C_local)
        for ko in T.Pipelined(1, num_stages=1):
            T.copy(A[by * 16, ko * 32], A_shared)
            T.copy(B[ko * 32, bx * 16], B_shared)
            T.gemm(A_shared, B_shared, C_local)
        T.copy(C_local, C[by * 16, bx * 16])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    artifact = tilelang.lower(
        gemm_32,
        target={"kind": "sunway", "output_dir": str(args.output_dir), "output_indices": [2]},
        runtime_only=True,
    )
    shutil.copy2(Path(__file__).with_name("gemm_32_main.c"), args.output_dir)
    print(f"generated {args.output_dir} ({len(artifact.kernel_source)} CPE C bytes)")


if __name__ == "__main__":
    main()
