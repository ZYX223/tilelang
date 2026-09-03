"""Static ownership and LDM planning for Sunway copy operations."""

from __future__ import annotations

import math
from dataclasses import dataclass

from tvm import tirx

from ...target import SunwayTargetConfig


_SW64_POINTER_BYTES = 8
_REPLY_BYTES = 4


@dataclass(frozen=True, slots=True)
class SunwayCopyPlan:
    """A legal single-CG ownership and LDM plan for one contiguous copy."""

    total_elements: int
    element_bytes: int
    alignment_elements: int
    cpe_count: int
    tile_elements: int
    tile_count: int
    active_cpes: int
    iterations_per_cpe: int
    final_tile_elements: int
    tile_bytes: int
    ldm_bytes: int


def _static_shape(buffer: tirx.Buffer) -> tuple[int, ...]:
    shape: list[int] = []
    for extent in buffer.shape:
        if not isinstance(extent, tirx.IntImm):
            raise ValueError("Sunway copy analysis requires statically shaped buffers")
        shape.append(int(extent))
    if any(extent <= 0 for extent in shape):
        raise ValueError("Sunway copy analysis requires positive buffer extents")
    return tuple(shape)


def _dtype_bytes(buffer: tirx.Buffer) -> int:
    bits = int(buffer.dtype.bits) * int(buffer.dtype.lanes)
    if bits % 8:
        raise ValueError(f"Sunway DMA requires a byte-addressable dtype, got {buffer.dtype}")
    return bits // 8


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _align_down(value: int, alignment: int) -> int:
    return value // alignment * alignment


def analyze_copy(
    source: tirx.Buffer,
    destination: tirx.Buffer,
    config: SunwayTargetConfig,
    *,
    argument_count: int,
) -> SunwayCopyPlan:
    """Choose an aligned tile and a grid-stride ownership plan for ``T.copy``."""

    source_shape = _static_shape(source)
    destination_shape = _static_shape(destination)
    if destination_shape != source_shape or destination.dtype != source.dtype:
        raise ValueError("Sunway copy source and destination must have the same static shape and dtype")

    element_bytes = _dtype_bytes(source)
    if config.dma_alignment <= 0 or config.dma_alignment % element_bytes:
        raise ValueError(
            "Sunway DMA alignment must be positive and divisible by the copy element size"
        )
    if config.cpe_rows <= 0 or config.cpe_cols <= 0:
        raise ValueError("Sunway copy requires a non-empty CPE mesh")
    if argument_count <= 0:
        raise ValueError("Sunway copy requires at least one launch argument")

    total_elements = math.prod(source_shape)
    total_bytes = total_elements * element_bytes
    if total_bytes % config.dma_alignment:
        raise ValueError(
            f"Sunway copy total byte count {total_bytes} violates "
            f"{config.dma_alignment}-byte DMA alignment"
        )

    alignment_elements = config.dma_alignment // element_bytes
    cpe_count = config.cpe_rows * config.cpe_cols

    # The copied launch descriptor and DMA reply counter also live in CPE LDM.
    # Reserve them before choosing a tile so every accepted plan fits at runtime.
    fixed_ldm_bytes = argument_count * _SW64_POINTER_BYTES + _REPLY_BYTES
    available_tile_bytes = _align_down(
        config.ldm_bytes_per_cpe - fixed_ldm_bytes,
        config.dma_alignment,
    )
    if available_tile_bytes < config.dma_alignment:
        raise ValueError(
            f"Sunway fixed LDM overhead of {fixed_ldm_bytes} bytes leaves no room "
            f"for one {config.dma_alignment}-byte DMA tile"
        )

    elements_per_cpe = (total_elements + cpe_count - 1) // cpe_count
    ideal_tile_elements = _align_up(elements_per_cpe, alignment_elements)
    max_tile_elements = available_tile_bytes // element_bytes
    tile_elements = min(ideal_tile_elements, max_tile_elements)
    tile_count = (total_elements + tile_elements - 1) // tile_elements
    final_tile_elements = total_elements - (tile_count - 1) * tile_elements
    if final_tile_elements % alignment_elements:
        raise ValueError("Sunway copy final tile does not satisfy DMA alignment")

    active_cpes = min(cpe_count, tile_count)
    iterations_per_cpe = (tile_count + cpe_count - 1) // cpe_count
    tile_bytes = tile_elements * element_bytes
    ldm_bytes = fixed_ldm_bytes + tile_bytes
    return SunwayCopyPlan(
        total_elements=total_elements,
        element_bytes=element_bytes,
        alignment_elements=alignment_elements,
        cpe_count=cpe_count,
        tile_elements=tile_elements,
        tile_count=tile_count,
        active_cpes=active_cpes,
        iterations_per_cpe=iterations_per_cpe,
        final_tile_elements=final_tile_elements,
        tile_bytes=tile_bytes,
        ldm_bytes=ldm_bytes,
    )
