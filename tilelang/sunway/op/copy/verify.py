"""Legality checks for the scheduled Sunway copy path."""

from __future__ import annotations

from tvm import tirx

from ...target import SunwayTargetConfig
from ...tir_utils import (
    NATIVE_CALLS,
    SEMANTIC_CALLS,
    named_calls,
    positive_attr,
    verify_call_counts,
)


_COPY_PLAN_ATTRS = (
    "sunway.total_elements",
    "sunway.element_bytes",
    "sunway.dma_alignment",
    "sunway.dma_bytes",
    "sunway.cpe_count",
    "sunway.tile_elements",
    "sunway.tile_count",
    "sunway.active_cpes",
    "sunway.iterations_per_cpe",
    "sunway.final_tile_elements",
    "sunway.ldm_bytes",
)


def _verify_copy_plan_metadata(
    func: tirx.PrimFunc,
    config: SunwayTargetConfig,
    phase: str,
) -> None:
    values = {key: positive_attr(func, key, phase) for key in _COPY_PLAN_ATTRS}
    total = values["sunway.total_elements"]
    element_bytes = values["sunway.element_bytes"]
    alignment = values["sunway.dma_alignment"]
    tile_elements = values["sunway.tile_elements"]
    tile_count = values["sunway.tile_count"]
    cpe_count = values["sunway.cpe_count"]
    active_cpes = values["sunway.active_cpes"]
    iterations = values["sunway.iterations_per_cpe"]
    final_tile_elements = values["sunway.final_tile_elements"]
    tile_bytes = values["sunway.dma_bytes"]
    ldm_bytes = values["sunway.ldm_bytes"]

    expected_cpes = config.cpe_rows * config.cpe_cols
    if cpe_count != expected_cpes:
        raise ValueError(
            f"Sunway {phase} invariant failed: CPE plan uses {cpe_count}, "
            f"target mesh provides {expected_cpes}"
        )
    if alignment != config.dma_alignment:
        raise ValueError(
            f"Sunway {phase} invariant failed: DMA alignment metadata is {alignment}, "
            f"target requires {config.dma_alignment}"
        )
    if tile_bytes != tile_elements * element_bytes:
        raise ValueError(f"Sunway {phase} invariant failed: tile byte metadata is inconsistent")
    if tile_bytes % alignment or total * element_bytes % alignment:
        raise ValueError(f"Sunway {phase} invariant failed: DMA byte counts are not aligned")

    expected_tile_count = (total + tile_elements - 1) // tile_elements
    expected_final = total - (expected_tile_count - 1) * tile_elements
    expected_active = min(cpe_count, expected_tile_count)
    expected_iterations = (expected_tile_count + cpe_count - 1) // cpe_count
    if tile_count != expected_tile_count:
        raise ValueError(f"Sunway {phase} invariant failed: tile ownership does not cover input")
    if final_tile_elements != expected_final or final_tile_elements * element_bytes % alignment:
        raise ValueError(f"Sunway {phase} invariant failed: final DMA tile is invalid")
    if active_cpes != expected_active or iterations != expected_iterations:
        raise ValueError(
            f"Sunway {phase} invariant failed: CPE ownership metadata is inconsistent"
        )
    if ldm_bytes < tile_bytes:
        raise ValueError(f"Sunway {phase} invariant failed: LDM metadata omits the tile buffer")
    if ldm_bytes > config.ldm_bytes_per_cpe:
        raise ValueError(
            f"Sunway {phase} invariant failed: LDM plan uses {ldm_bytes} bytes, "
            f"but target limit is {config.ldm_bytes_per_cpe}"
        )

    loops: list[tirx.For] = []
    tirx.stmt_functor.post_order_visit(
        func.body,
        lambda node: loops.append(node) if isinstance(node, tirx.For) else None,
    )
    if len(loops) != 1 or not isinstance(loops[0].extent, tirx.IntImm):
        raise ValueError(f"Sunway {phase} invariant failed: expected one static ownership loop")
    if int(loops[0].extent) != iterations:
        raise ValueError(
            f"Sunway {phase} invariant failed: ownership loop extent is inconsistent"
        )


def verify_copy_semantic_func(
    func: tirx.PrimFunc,
    config: SunwayTargetConfig,
) -> tirx.PrimFunc:
    phase = "S2"
    names = set(named_calls(func.body))
    native = sorted(names & NATIVE_CALLS)
    if native:
        raise ValueError(
            f"Sunway S2 invariant failed: native ABI call {native[0]!r} appeared before S3"
        )
    if "tl.tileop.copy" in names:
        raise ValueError("Sunway S2 invariant failed: T.copy was not materialized")
    _verify_copy_plan_metadata(func, config, phase)
    verify_call_counts(
        func,
        {
            "tilelang_sunway_pe_id": 1,
            "tilelang_sunway_dma_get": 1,
            "tilelang_sunway_dma_put": 1,
            "tilelang_sunway_dma_wait": 2,
        },
        phase,
    )
    return func


def verify_copy_native_func(
    func: tirx.PrimFunc,
    config: SunwayTargetConfig,
) -> tirx.PrimFunc:
    phase = "S3"
    names = set(named_calls(func.body))
    semantic = sorted(names & SEMANTIC_CALLS)
    if semantic:
        raise ValueError(
            f"Sunway S3 invariant failed: semantic call {semantic[0]!r} remains after lowering"
        )
    if "tl.tileop.copy" in names:
        raise ValueError("Sunway S3 invariant failed: T.copy remains after lowering")
    _verify_copy_plan_metadata(func, config, phase)
    verify_call_counts(
        func,
        {
            "_MYID": 1,
            "athread_get": 1,
            "athread_put": 1,
            "tilelang_sunway_reply_wait": 2,
        },
        phase,
    )
    return func
