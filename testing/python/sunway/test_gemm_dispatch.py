from __future__ import annotations

from tilelang.backend.module import create_backend_context
from tilelang.tileop.gemm.registry import resolve_gemm_impl


def test_sunway_scalar_gemm_is_registered() -> None:
    target = create_backend_context({"kind": "sunway"}).target

    implementation = resolve_gemm_impl("sunway.scalar", target)

    from tilelang.sunway.op.gemm.gemm_scalar import GemmScalar

    assert implementation is GemmScalar
