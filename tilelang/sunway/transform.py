"""Sunway semantic lowering passes for the Paper-1 copy MVP."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable

from tvm import IRModule, tirx

from .analysis import analyze_copy
from .target import SunwayTargetConfig


_SEMANTIC_TO_NATIVE = {
    "tilelang_sunway_pe_id": "_MYID",
    "tilelang_sunway_dma_get": "athread_get",
    "tilelang_sunway_dma_put": "athread_put",
    "tilelang_sunway_dma_wait": "tilelang_sunway_reply_wait",
}
_SEMANTIC_CALLS = frozenset(_SEMANTIC_TO_NATIVE)
_NATIVE_CALLS = frozenset(_SEMANTIC_TO_NATIVE.values())
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


def _map_prim_funcs(mod: IRModule, rewrite: Callable[[tirx.PrimFunc], tirx.PrimFunc]) -> IRModule:
    functions = {
        global_var: rewrite(func) if isinstance(func, tirx.PrimFunc) else func
        for global_var, func in mod.functions.items()
    }
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

    # S2 owns the scheduling decision. Each CPE walks its grid-stride sequence
    # of logical tiles; S3 and codegen only lower this already-fixed structure.
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
            tirx.Evaluate(tirx.call_extern("int32", "tilelang_sunway_dma_wait", reply.access_ptr("rw", extent=1), 1)),
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
            tirx.Evaluate(tirx.call_extern("int32", "tilelang_sunway_dma_wait", reply.access_ptr("rw", extent=1), 1)),
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


def lower_tile_copy_to_semantic_tir(mod: IRModule, config: SunwayTargetConfig) -> IRModule:
    """Lower the first supported TileLang operation into S2 Sunway TIR."""

    return _map_prim_funcs(mod, lambda func: _lower_copy_func(func, config))


def _named_calls(func: tirx.PrimFunc) -> list[str]:
    names: list[str] = []

    def visit(node: object) -> None:
        if isinstance(node, tirx.Call):
            name = _call_name(node)
            if name is not None:
                names.append(name)

    tirx.stmt_functor.post_order_visit(func.body, visit)
    return names


def _positive_attr(func: tirx.PrimFunc, key: str, phase: str) -> int:
    value = func.attrs.get(key)
    if value is None:
        raise ValueError(f"Sunway {phase} invariant failed: required attribute {key} is missing")
    result = int(value)
    if result <= 0:
        raise ValueError(
            f"Sunway {phase} invariant failed: {key} must be positive, got {result}"
        )
    return result


def _verify_copy_plan_metadata(
    func: tirx.PrimFunc,
    config: SunwayTargetConfig,
    phase: str,
) -> None:
    values = {key: _positive_attr(func, key, phase) for key in _COPY_PLAN_ATTRS}
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
        raise ValueError(f"Sunway {phase} invariant failed: CPE ownership metadata is inconsistent")
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
        raise ValueError(f"Sunway {phase} invariant failed: ownership loop extent is inconsistent")


def _verify_call_counts(func: tirx.PrimFunc, expected: dict[str, int], phase: str) -> None:
    counts = Counter(_named_calls(func))
    for name, count in expected.items():
        if counts[name] != count:
            raise ValueError(
                f"Sunway {phase} invariant failed: expected {count} {name} call(s), "
                f"found {counts[name]}"
            )


def verify_semantic_tir(mod: IRModule, config: SunwayTargetConfig) -> IRModule:
    """Check that S2 contains a complete abstract copy plan and no SW9A ABI calls."""

    def verify(func: tirx.PrimFunc) -> tirx.PrimFunc:
        phase = "S2"
        if str(func.attrs.get("sunway.phase", "")) != phase:
            raise ValueError("Sunway S2 verifier received a function from another phase")
        if str(func.attrs.get("sunway.kernel_kind", "")) != "copy":
            raise ValueError("Sunway S2 verifier currently supports copy kernels only")

        names = set(_named_calls(func))
        native = sorted(names & _NATIVE_CALLS)
        if native:
            raise ValueError(
                f"Sunway S2 invariant failed: native ABI call {native[0]!r} appeared before S3"
            )
        if "tl.tileop.copy" in names:
            raise ValueError("Sunway S2 invariant failed: T.copy was not materialized")
        _verify_copy_plan_metadata(func, config, phase)
        _verify_call_counts(
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

    return _map_prim_funcs(mod, verify)


def _lower_semantic_call(node: object) -> object | None:
    if not isinstance(node, tirx.Call):
        return None
    name = _call_name(node)
    native_name = _SEMANTIC_TO_NATIVE.get(name)
    if native_name is None:
        return None
    return tirx.call_extern(node.dtype, native_name, *node.args[1:])


def lower_semantic_to_native_tir(
    mod: IRModule,
    config: SunwayTargetConfig,
) -> IRModule:
    """Resolve S2 semantic leaves to S3 SW9A ABI-level operations."""

    verify_semantic_tir(mod, config)

    def rewrite(func: tirx.PrimFunc) -> tirx.PrimFunc:
        body = tirx.stmt_functor.ir_transform(func.body, None, _lower_semantic_call)
        return func.with_body(body).with_attr("sunway.phase", "S3")

    return _map_prim_funcs(mod, rewrite)


def verify_native_tir(mod: IRModule, config: SunwayTargetConfig) -> IRModule:
    """Check that S3 is complete enough for mechanical MPE/CPE C emission."""

    def verify(func: tirx.PrimFunc) -> tirx.PrimFunc:
        phase = "S3"
        if str(func.attrs.get("sunway.phase", "")) != phase:
            raise ValueError("Sunway S3 verifier received a function from another phase")
        names = set(_named_calls(func))
        semantic = sorted(names & _SEMANTIC_CALLS)
        if semantic:
            raise ValueError(
                f"Sunway S3 invariant failed: semantic call {semantic[0]!r} remains after lowering"
            )
        if "tl.tileop.copy" in names:
            raise ValueError("Sunway S3 invariant failed: T.copy remains after lowering")
        _verify_copy_plan_metadata(func, config, phase)
        _verify_call_counts(
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

    return _map_prim_funcs(mod, verify)
