"""Sunway semantic lowering passes for the Paper-1 copy MVP."""

from __future__ import annotations

import math
from collections.abc import Callable

from tvm import IRModule, tirx

from .target import SunwayTargetConfig


_SEMANTIC_TO_NATIVE = {
    "tilelang_sunway_pe_id": "_MYID",
    "tilelang_sunway_dma_get": "athread_get",
    "tilelang_sunway_dma_put": "athread_put",
    "tilelang_sunway_dma_wait": "tilelang_sunway_reply_wait",
}


def _map_prim_funcs(mod: IRModule, rewrite: Callable[[tirx.PrimFunc], tirx.PrimFunc]) -> IRModule:
    functions = {global_var: rewrite(func) if isinstance(func, tirx.PrimFunc) else func for global_var, func in mod.functions.items()}
    return IRModule(functions, attrs=mod.attrs, global_infos=mod.global_infos)


def annotate_sunway_tir(mod: IRModule) -> IRModule:
    """Mark the untouched TileLang TIR as the S1 backend input."""

    return _map_prim_funcs(mod, lambda func: func.with_attr("sunway.phase", "S1"))


def _call_name(call: tirx.Call) -> str | None:
    op_name = getattr(call.op, "name", None)
    if op_name == "tl.tileop.copy":
        return op_name
    if op_name == "tirx.call_extern" and call.args and isinstance(call.args[0], tirx.StringImm):
        return call.args[0].value
    return None


def _find_copy_call(func: tirx.PrimFunc) -> tirx.Call:
    copies: list[tirx.Call] = []

    def visit(node: object) -> None:
        if isinstance(node, tirx.Call) and _call_name(node) == "tl.tileop.copy":
            copies.append(node)

    tirx.stmt_functor.post_order_visit(func.body, visit)
    if len(copies) != 1:
        raise ValueError(f"Sunway Paper-1 lowering currently requires exactly one T.copy per PrimFunc; found {len(copies)}")
    return copies[0]


def _region_buffer(region: object, role: str) -> tirx.Buffer:
    if not isinstance(region, tirx.Call) or getattr(region.op, "name", None) != "tl.region":
        raise ValueError(f"Sunway T.copy {role} must be a direct TileLang buffer region")
    base = region.args[0]
    if not isinstance(base, tirx.BufferLoad):
        raise ValueError(f"Sunway T.copy {role} region must start from a buffer load")
    return base.buffer


def _static_elements(buffer: tirx.Buffer) -> int:
    extents: list[int] = []
    for extent in buffer.shape:
        if not isinstance(extent, tirx.IntImm):
            raise ValueError("Sunway Paper-1 lowering only supports statically shaped T.copy buffers")
        extents.append(int(extent))
    return math.prod(extents)


def _dtype_bytes(buffer: tirx.Buffer) -> int:
    dtype = buffer.dtype
    bits = int(dtype.bits) * int(dtype.lanes)
    if bits % 8:
        raise ValueError(f"Sunway DMA requires byte-addressable data, got {dtype}")
    return bits // 8


def _copy_schedule(
    source: tirx.Buffer,
    destination: tirx.Buffer,
    config: SunwayTargetConfig,
) -> tuple[int, int, int]:
    total = _static_elements(source)
    if _static_elements(destination) != total or destination.dtype != source.dtype:
        raise ValueError("Sunway T.copy source and destination must have the same static shape and dtype")

    element_bytes = _dtype_bytes(source)
    if config.dma_alignment % element_bytes:
        raise ValueError("Sunway DMA alignment must be divisible by the T.copy element size")

    cpe_count = config.cpe_rows * config.cpe_cols
    alignment_elements = config.dma_alignment // element_bytes
    tile_elements = math.ceil(total / cpe_count)
    tile_elements = math.ceil(tile_elements / alignment_elements) * alignment_elements
    if total % tile_elements:
        raise ValueError("Sunway Paper-1 T.copy requires an alignment-sized static decomposition without a tail")

    active_cpes = total // tile_elements
    tile_bytes = tile_elements * element_bytes
    if active_cpes > cpe_count:
        raise ValueError("Sunway T.copy decomposition exceeds the configured CPE mesh")
    if tile_bytes > config.ldm_bytes_per_cpe:
        raise ValueError(f"Sunway T.copy tile requires {tile_bytes} LDM bytes, exceeding the {config.ldm_bytes_per_cpe}-byte CPE budget")
    if tile_bytes % config.dma_alignment:
        raise ValueError("Sunway T.copy tile does not satisfy DMA byte alignment")
    return tile_elements, tile_bytes, active_cpes


def _lower_copy_func(func: tirx.PrimFunc, config: SunwayTargetConfig) -> tirx.PrimFunc:
    copy = _find_copy_call(func)
    source = _region_buffer(copy.args[0], "source")
    destination = _region_buffer(copy.args[1], "destination")
    tile_elements, tile_bytes, active_cpes = _copy_schedule(source, destination, config)

    pe_id = tirx.Var("pe_id", "int32")
    tile = tirx.decl_buffer(
        (tile_elements,),
        source.dtype,
        name="copy_tile",
        scope="local.ldm",
        data_alignment=config.dma_alignment,
    )
    reply = tirx.decl_buffer((1,), "int32", name="reply", scope="local")
    source_offset = pe_id * tile_elements
    destination_offset = pe_id * tile_elements

    # S2 is the ownership and transfer decision point. DMA calls are semantic
    # leaves here, so later generic loop passes cannot split or reorder them.
    owned_copy = tirx.SeqStmt(
        [
            tirx.BufferStore(reply, 0, [0]),
            tirx.Evaluate(
                tirx.call_extern(
                    "int32",
                    "tilelang_sunway_dma_get",
                    source.access_ptr("r", offset=source_offset, extent=tile_elements),
                    tile.access_ptr("w", extent=tile_elements),
                    tile_bytes,
                    reply.access_ptr("rw", extent=1),
                )
            ),
            tirx.Evaluate(tirx.call_extern("int32", "tilelang_sunway_dma_wait", reply.access_ptr("rw", extent=1), 1)),
            tirx.BufferStore(reply, 0, [0]),
            tirx.Evaluate(
                tirx.call_extern(
                    "int32",
                    "tilelang_sunway_dma_put",
                    tile.access_ptr("r", extent=tile_elements),
                    destination.access_ptr("w", offset=destination_offset, extent=tile_elements),
                    tile_bytes,
                    reply.access_ptr("rw", extent=1),
                )
            ),
            tirx.Evaluate(tirx.call_extern("int32", "tilelang_sunway_dma_wait", reply.access_ptr("rw", extent=1), 1)),
        ]
    )
    body = tirx.SeqStmt(
        [
            tirx.Bind(pe_id, tirx.call_extern("int32", "tilelang_sunway_pe_id")),
            tirx.AllocBuffer(tile),
            tirx.AllocBuffer(reply),
            tirx.IfThenElse(pe_id < active_cpes, owned_copy, None),
        ]
    )
    lowered = func.with_body(body)
    for key, value in {
        "sunway.phase": "S2",
        "sunway.kernel_kind": "copy",
        "sunway.tile_elements": tile_elements,
        "sunway.dma_bytes": tile_bytes,
        "sunway.active_cpes": active_cpes,
    }.items():
        lowered = lowered.with_attr(key, value)
    return lowered


def lower_tile_copy_to_semantic_tir(mod: IRModule, config: SunwayTargetConfig) -> IRModule:
    """Lower the first supported TileLang operation into S2 Sunway TIR."""

    return _map_prim_funcs(mod, lambda func: _lower_copy_func(func, config))


def _lower_semantic_call(node: object) -> object | None:
    if not isinstance(node, tirx.Call):
        return None
    name = _call_name(node)
    native_name = _SEMANTIC_TO_NATIVE.get(name)
    if native_name is None:
        return None
    return tirx.call_extern(node.dtype, native_name, *node.args[1:])


def lower_semantic_to_native_tir(mod: IRModule) -> IRModule:
    """Resolve S2 semantic leaves to S3 SW9A ABI-level operations."""

    def rewrite(func: tirx.PrimFunc) -> tirx.PrimFunc:
        body = tirx.stmt_functor.ir_transform(func.body, None, _lower_semantic_call)
        return func.with_body(body).with_attr("sunway.phase", "S3")

    return _map_prim_funcs(mod, rewrite)
