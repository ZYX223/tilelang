"""Dispatch Sunway semantic verification and S2-to-S3 lowering by operator."""

from __future__ import annotations

from tvm import IRModule, tirx

from ..op.copy.verify import verify_copy_native_func, verify_copy_semantic_func
from ..op.gemm.lower import lower_gemm_semantic_to_native_tir
from ..op.gemm.verify import verify_gemm_native_func, verify_gemm_semantic_func
from ..target import SunwayTargetConfig
from ..tir_utils import lower_semantic_calls_to_native, map_prim_funcs


_GEMM_KERNEL_KINDS = frozenset({"gemm_scalar", "gemm_simd"})


def verify_semantic_tir(mod: IRModule, config: SunwayTargetConfig) -> IRModule:
    """Verify each S2 function through its operator-owned legality checker."""

    def verify(func: tirx.PrimFunc) -> tirx.PrimFunc:
        if str(func.attrs.get("sunway.phase", "")) != "S2":
            raise ValueError("Sunway S2 verifier received a function from another phase")
        kernel_kind = str(func.attrs.get("sunway.kernel_kind", ""))
        if kernel_kind in _GEMM_KERNEL_KINDS:
            return verify_gemm_semantic_func(func, config)
        if kernel_kind == "copy":
            return verify_copy_semantic_func(func, config)
        raise ValueError(f"Sunway S2 verifier does not support kernel kind {kernel_kind!r}")

    return map_prim_funcs(mod, verify)


def lower_semantic_to_native_tir(
    mod: IRModule,
    config: SunwayTargetConfig,
) -> IRModule:
    """Resolve S2 semantic leaves to S3 SW9A ABI-level operations."""

    verify_semantic_tir(mod, config)
    kernel_kinds = {
        str(func.attrs.get("sunway.kernel_kind", ""))
        for func in mod.functions.values()
        if isinstance(func, tirx.PrimFunc)
    }
    if len(kernel_kinds) == 1 and kernel_kinds <= _GEMM_KERNEL_KINDS:
        return lower_gemm_semantic_to_native_tir(mod, config)
    if kernel_kinds == {"copy"}:
        return lower_semantic_calls_to_native(mod)
    raise ValueError(f"Sunway S3 lowering does not support kernel kinds {sorted(kernel_kinds)!r}")


def verify_native_tir(mod: IRModule, config: SunwayTargetConfig) -> IRModule:
    """Verify each codegen-ready S3 function through its operator checker."""

    def verify(func: tirx.PrimFunc) -> tirx.PrimFunc:
        if str(func.attrs.get("sunway.phase", "")) != "S3":
            raise ValueError("Sunway S3 verifier received a function from another phase")
        kernel_kind = str(func.attrs.get("sunway.kernel_kind", ""))
        if kernel_kind in _GEMM_KERNEL_KINDS:
            return verify_gemm_native_func(func, config)
        if kernel_kind == "copy":
            return verify_copy_native_func(func, config)
        raise ValueError(f"Sunway S3 verifier does not support kernel kind {kernel_kind!r}")

    return map_prim_funcs(mod, verify)
