"""Generate the first Sunway AOT project from a TileLang ``T.copy`` kernel."""

from __future__ import annotations

import argparse
from pathlib import Path

import tilelang
import tilelang.language as T


@T.prim_func
def copy_128(A: T.Tensor((128,), "float32"), B: T.Tensor((128,), "float32")):
    T.copy(A, B)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    artifact = tilelang.lower(
        copy_128,
        target={"kind": "sunway", "output_dir": str(args.output_dir)},
        runtime_only=True,
    )
    print(f"generated {args.output_dir} ({len(artifact.kernel_source)} CPE C bytes)")


if __name__ == "__main__":
    main()
