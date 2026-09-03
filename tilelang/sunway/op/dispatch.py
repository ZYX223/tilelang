"""Select the operator-owned S1-to-S2 lowering path."""

from __future__ import annotations

from tvm import IRModule, tirx

from ..target import SunwayTargetConfig
from .copy import lower_tile_copy_to_semantic_tir
from .gemm import lower_gemm_program_to_semantic_tir


def _contains_tile_op(mod: IRModule, name: str) -> bool:
    found = False

    def visit(node: object) -> None:
        nonlocal found
        if isinstance(node, tirx.Call) and str(getattr(node.op, "name", "")) == name:
            found = True

    for func in mod.functions.values():
        if isinstance(func, tirx.PrimFunc):
            tirx.stmt_functor.post_order_visit(func.body, visit)
    return found


def lower_program_to_semantic_tir(
    mod: IRModule,
    config: SunwayTargetConfig,
) -> IRModule:
    """Dispatch composite GEMM before the standalone copy fallback."""

    if _contains_tile_op(mod, "tl.tileop.gemm"):
        return lower_gemm_program_to_semantic_tir(mod, config)
    if _contains_tile_op(mod, "tl.tileop.copy"):
        return lower_tile_copy_to_semantic_tir(mod, config)
    raise ValueError("Sunway S2 lowering found no supported TileLang operation")
