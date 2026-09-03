"""Orchestrate progressive S1/S2/S3 lowering for Sunway GEMM."""

from __future__ import annotations

import tilelang
from tvm import IRModule, tirx

from ...target import SunwayTargetConfig
from ...tir_utils import lower_semantic_calls_to_native
from .gemm_vmad import lower_gemm_compute_to_simd
from .plan import SunwayGemmPlan
from .schedule import (
    attach_gemm_plan,
    bound_block_tile_extents,
    expand_sunway_dma_2d,
    lower_sunway_kernel_bindings,
    restore_mesh_round_dimensions,
    restore_serial_dimensions,
    restore_unit_block_dimensions,
)
from .utils import only_prim_func
from .verify import verify_gemm_native_tir, verify_gemm_semantic_tir


def lower_gemm_program_to_semantic_tir(
    mod: IRModule,
    config: SunwayTargetConfig,
) -> IRModule:
    """Lower one canonical tiled GEMM from S1 to inspectable semantic S2 TIR."""

    source_func = only_prim_func(mod)
    plan = SunwayGemmPlan.from_prim_func(source_func, config)
    block_tiles = bound_block_tile_extents(source_func)
    mod = tilelang.transform.LayoutInference()(mod)
    mod = tilelang.transform.LowerTileOp()(mod)
    mod = lower_sunway_kernel_bindings(mod, config, plan)
    mod = expand_sunway_dma_2d(mod, config, plan)
    mod = tilelang.transform.PlanAndUpdateBufferAllocationLocation()(mod)
    mod = tilelang.transform.LowerOpaqueBlock()(mod)
    mod = tirx.transform.Simplify()(mod)
    if plan.compute == "simd":
        mod = lower_gemm_compute_to_simd(mod, plan)
    if plan.ownership == "mesh_2d":
        mod = restore_mesh_round_dimensions(mod, plan)
    else:
        mod = restore_unit_block_dimensions(mod, *block_tiles)
    return attach_gemm_plan(mod, plan, block_tiles)


def lower_gemm_semantic_to_native_tir(
    mod: IRModule,
    config: SunwayTargetConfig,
) -> IRModule:
    """Resolve semantic leaves and prepare GEMM TIR for mechanical C emission."""

    verify_gemm_semantic_tir(mod, config)
    func = only_prim_func(mod)
    block_tiles_m = int(func.attrs["sunway.block_tiles_m"])
    block_tiles_n = int(func.attrs["sunway.block_tiles_n"])
    ownership = str(func.attrs.get("sunway.ownership", "single"))
    mod = lower_semantic_calls_to_native(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tirx.transform.Simplify()(mod)
    mod = tirx.transform.RemoveNoOp()(mod)
    if ownership == "mesh_2d":
        rounds_m = (block_tiles_m + config.cpe_rows - 1) // config.cpe_rows
        rounds_n = (block_tiles_n + config.cpe_cols - 1) // config.cpe_cols
        mod = restore_serial_dimensions(
            mod,
            "by_round",
            rounds_m,
            "bx_round",
            rounds_n,
        )
    else:
        mod = restore_unit_block_dimensions(mod, block_tiles_m, block_tiles_n)
    return verify_gemm_native_tir(mod, config)
