"""Lower TileLang copy operations to scheduled Sunway semantic TIR."""

from __future__ import annotations

from tvm import IRModule, tirx

from ...target import SunwayTargetConfig
from ...tir_utils import call_name, map_prim_funcs
from .plan import analyze_copy


def _find_copy_call(func: tirx.PrimFunc) -> tirx.Call:
    copies: list[tirx.Call] = []

    def visit(node: object) -> None:
        if isinstance(node, tirx.Call) and call_name(node) == "tl.tileop.copy":
            copies.append(node)

    tirx.stmt_functor.post_order_visit(func.body, visit)
    if len(copies) != 1:
        raise ValueError(
            "Sunway Paper-1 lowering currently requires exactly one T.copy "
            f"per PrimFunc; found {len(copies)}"
        )
    return copies[0]


def _region_buffer(region: object, role: str) -> tirx.Buffer:
    if not isinstance(region, tirx.Call) or getattr(region.op, "name", None) != "tl.region":
        raise ValueError(f"Sunway T.copy {role} must be a direct TileLang buffer region")
    base = region.args[0]
    if not isinstance(base, tirx.BufferLoad):
        raise ValueError(f"Sunway T.copy {role} region must start from a buffer load")
    return base.buffer


def _lower_copy_func(func: tirx.PrimFunc, config: SunwayTargetConfig) -> tirx.PrimFunc:
    copy = _find_copy_call(func)
    source = _region_buffer(copy.args[0], "source")
    destination = _region_buffer(copy.args[1], "destination")
    argument_count = sum(param in func.buffer_map for param in func.params)
    plan = analyze_copy(
        source,
        destination,
        config,
        argument_count=argument_count,
    )

    pe_id = tirx.Var("pe_id", "int32")
    tile_iteration = tirx.Var("tile_iteration", "int32")
    tile = tirx.decl_buffer(
        (plan.tile_elements,),
        source.dtype,
        name="copy_tile",
        scope="local.ldm",
        data_alignment=config.dma_alignment,
    )
    reply = tirx.decl_buffer((1,), "int32", name="reply", scope="local")
    tile_index = pe_id + tile_iteration * plan.cpe_count
    tile_offset = tile_index * plan.tile_elements
    valid_elements = tirx.min(plan.tile_elements, plan.total_elements - tile_offset)
    valid_bytes = valid_elements * plan.element_bytes

    # S2 fixes each CPE's grid-stride tile ownership. Later phases only lower
    # this schedule; they never infer ownership from generated C text.
    owned_copy = tirx.SeqStmt(
        [
            tirx.BufferStore(reply, 0, [0]),
            tirx.Evaluate(
                tirx.call_extern(
                    "int32",
                    "tilelang_sunway_dma_get",
                    source.access_ptr("r", offset=tile_offset, extent=valid_elements),
                    tile.access_ptr("w", extent=valid_elements),
                    valid_bytes,
                    reply.access_ptr("rw", extent=1),
                )
            ),
            tirx.Evaluate(
                tirx.call_extern(
                    "int32",
                    "tilelang_sunway_dma_wait",
                    reply.access_ptr("rw", extent=1),
                    1,
                )
            ),
            tirx.BufferStore(reply, 0, [0]),
            tirx.Evaluate(
                tirx.call_extern(
                    "int32",
                    "tilelang_sunway_dma_put",
                    tile.access_ptr("r", extent=valid_elements),
                    destination.access_ptr("w", offset=tile_offset, extent=valid_elements),
                    valid_bytes,
                    reply.access_ptr("rw", extent=1),
                )
            ),
            tirx.Evaluate(
                tirx.call_extern(
                    "int32",
                    "tilelang_sunway_dma_wait",
                    reply.access_ptr("rw", extent=1),
                    1,
                )
            ),
        ]
    )
    body = tirx.SeqStmt(
        [
            tirx.Bind(pe_id, tirx.call_extern("int32", "tilelang_sunway_pe_id")),
            tirx.AllocBuffer(tile),
            tirx.AllocBuffer(reply),
            tirx.For(
                tile_iteration,
                0,
                plan.iterations_per_cpe,
                tirx.ForKind.SERIAL,
                tirx.IfThenElse(tile_index < plan.tile_count, owned_copy, None),
            ),
        ]
    )
    lowered = func.with_body(body)
    for key, value in {
        "sunway.phase": "S2",
        "sunway.kernel_kind": "copy",
        "sunway.total_elements": plan.total_elements,
        "sunway.element_bytes": plan.element_bytes,
        "sunway.dma_alignment": config.dma_alignment,
        "sunway.dma_bytes": plan.tile_bytes,
        "sunway.cpe_count": plan.cpe_count,
        "sunway.tile_elements": plan.tile_elements,
        "sunway.tile_count": plan.tile_count,
        "sunway.active_cpes": plan.active_cpes,
        "sunway.iterations_per_cpe": plan.iterations_per_cpe,
        "sunway.final_tile_elements": plan.final_tile_elements,
        "sunway.ldm_bytes": plan.ldm_bytes,
    }.items():
        lowered = lowered.with_attr(key, value)
    return lowered


def lower_tile_copy_to_semantic_tir(
    mod: IRModule,
    config: SunwayTargetConfig,
) -> IRModule:
    """Lower one supported TileLang copy operation into S2 Sunway TIR."""

    return map_prim_funcs(mod, lambda func: _lower_copy_func(func, config))
