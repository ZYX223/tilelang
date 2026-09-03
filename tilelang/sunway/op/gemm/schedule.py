"""Materialize CPE ownership, row DMA, and GEMM plan metadata in S2."""

from __future__ import annotations

from tvm import IRModule, tirx

from ...target import SunwayTargetConfig
from .plan import SunwayGemmPlan
from .utils import call_name, map_prim_funcs, rewrite_for, static_int


_DMA_2D_CALLS = {
    "tilelang_sunway_dma_get_2d": "tilelang_sunway_dma_get",
    "tilelang_sunway_dma_put_2d": "tilelang_sunway_dma_put",
}


def lower_sunway_kernel_bindings(
    mod: IRModule,
    config: SunwayTargetConfig,
    plan: SunwayGemmPlan,
) -> IRModule:
    """Map logical TileLang workers and tiles to the selected CPE schedule."""

    expected_workers = config.cpe_rows * config.cpe_cols

    def rewrite_func(func: tirx.PrimFunc) -> tirx.PrimFunc:
        pe_id = tirx.Var("pe_id", "int32")
        pe_row = tirx.Var("pe_row", "int32")
        pe_col = tirx.Var("pe_col", "int32")
        thread_x_sites = 0

        def rewrite(node: object) -> object | None:
            nonlocal thread_x_sites
            if not isinstance(node, tirx.For) or node.thread_binding is None:
                return None
            tag = str(node.thread_binding.thread_tag)
            if tag in {"blockIdx.x", "blockIdx.y"}:
                if plan.ownership == "mesh_2d":
                    coordinate = pe_col if tag == "blockIdx.x" else pe_row
                    stride = config.cpe_cols if tag == "blockIdx.x" else config.cpe_rows
                    dimension = "bx" if tag == "blockIdx.x" else "by"
                    tile_extent = static_int(node.extent, f"{tag} extent")
                    rounds = (tile_extent + stride - 1) // stride
                    round_var = tirx.Var(f"{dimension}_round", "int32")
                    tile_coordinate = coordinate + round_var * stride
                    tile_body = tirx.stmt_functor.substitute(
                        node.body,
                        {node.loop_var: tile_coordinate},
                    )
                    guarded_body = tirx.IfThenElse(
                        tile_coordinate < tile_extent,
                        tile_body,
                        None,
                        span=getattr(node, "span", None),
                    )
                    return tirx.For(
                        round_var,
                        0,
                        rounds,
                        tirx.ForKind.SERIAL,
                        guarded_body,
                        span=getattr(node, "span", None),
                    )
                return rewrite_for(node, node.body, kind=tirx.ForKind.SERIAL)
            if tag == "blockIdx.z":
                if static_int(node.extent, tag) != 1:
                    raise ValueError("Sunway G0 requires blockIdx.z extent 1")
                return tirx.stmt_functor.substitute(node.body, {node.loop_var: node.min})
            if tag == "threadIdx.x":
                workers = static_int(node.extent, "logical worker count")
                if workers != expected_workers:
                    raise ValueError(
                        f"Sunway G0 requires {expected_workers} logical workers, got {workers}"
                    )
                thread_x_sites += 1
                worker_body = tirx.stmt_functor.substitute(node.body, {node.loop_var: pe_id})
                if plan.ownership == "mesh_2d":
                    return worker_body
                # Layout lowering may fold its guard to true. S2 therefore
                # records single-CPE ownership explicitly after TileOp lowering.
                return tirx.IfThenElse(
                    pe_id == 0,
                    worker_body,
                    None,
                    span=getattr(node, "span", None),
                )
            if tag in {"threadIdx.y", "threadIdx.z"}:
                if static_int(node.extent, tag) != 1:
                    raise ValueError(f"Sunway G0 requires {tag} extent 1")
                return tirx.stmt_functor.substitute(node.body, {node.loop_var: node.min})
            raise ValueError(f"Sunway G0 does not support thread binding {tag}")

        body = tirx.stmt_functor.ir_transform(func.body, None, rewrite)
        if thread_x_sites != 1:
            raise ValueError(f"Sunway G0 requires one threadIdx.x binding, found {thread_x_sites}")
        bindings: list[tirx.Stmt] = [
            tirx.Bind(pe_id, tirx.call_extern("int32", "tilelang_sunway_pe_id")),
        ]
        if plan.ownership == "mesh_2d":
            bindings.extend(
                [
                    tirx.Bind(pe_row, tirx.FloorDiv(pe_id, config.cpe_cols)),
                    tirx.Bind(pe_col, tirx.FloorMod(pe_id, config.cpe_cols)),
                ]
            )
        return func.with_body(tirx.SeqStmt([*bindings, body]))

    return map_prim_funcs(mod, rewrite_func)


def _offset_access_ptr(
    pointer: tirx.PrimExpr,
    element_offset: tirx.PrimExpr,
    row_elements: int,
) -> tirx.Call:
    if (
        not isinstance(pointer, tirx.Call)
        or str(getattr(pointer.op, "name", "")) != "tirx.tvm_access_ptr"
    ):
        raise ValueError("Sunway G0 DMA requires a canonical tvm_access_ptr")
    if len(pointer.args) != 5:
        raise ValueError("Sunway G0 DMA received an invalid tvm_access_ptr")
    arguments = list(pointer.args)
    arguments[2] = arguments[2] + element_offset
    arguments[3] = tirx.IntImm("int32", row_elements)
    return tirx.Call(
        str(pointer.dtype),
        pointer.op,
        arguments,
        span=getattr(pointer, "span", None),
    )


def expand_sunway_dma_2d(
    mod: IRModule,
    config: SunwayTargetConfig,
    plan: SunwayGemmPlan,
) -> IRModule:
    """Expand each abstract 2D descriptor to an aligned row DMA and wait loop."""

    if plan.input_dtype != "float32":
        raise ValueError(f"Sunway G0 DMA does not support input dtype {plan.input_dtype}")
    element_bytes = 4

    def rewrite_func(func: tirx.PrimFunc) -> tirx.PrimFunc:
        reply = tirx.decl_buffer((1,), "int32", name="reply", scope="local")
        site_index = 0

        def rewrite(node: object) -> object | None:
            nonlocal site_index
            if not isinstance(node, tirx.Evaluate) or not isinstance(node.value, tirx.Call):
                return None
            call = node.value
            name = call_name(call)
            semantic_name = _DMA_2D_CALLS.get(name)
            if semantic_name is None:
                return None
            if len(call.args) != 7:
                raise ValueError(
                    f"Sunway G0 DMA descriptor {name} has an invalid argument count"
                )

            rows = static_int(call.args[3], "DMA row count")
            row_bytes = static_int(call.args[4], "DMA row byte count")
            source_stride_bytes = static_int(call.args[5], "DMA source byte stride")
            destination_stride_bytes = static_int(call.args[6], "DMA destination byte stride")
            for value, description in (
                (row_bytes, "row byte count"),
                (source_stride_bytes, "source byte stride"),
                (destination_stride_bytes, "destination byte stride"),
            ):
                if value % element_bytes:
                    raise ValueError(
                        f"Sunway G0 DMA {description} must be divisible by {element_bytes}"
                    )
            if row_bytes % config.dma_alignment:
                raise ValueError(
                    f"Sunway G0 DMA row byte count {row_bytes} violates "
                    f"{config.dma_alignment}-byte alignment"
                )

            row = tirx.Var(f"sunway_dma_row_{site_index}", "int32")
            site_index += 1
            source = _offset_access_ptr(
                call.args[1],
                row * (source_stride_bytes // element_bytes),
                row_bytes // element_bytes,
            )
            destination = _offset_access_ptr(
                call.args[2],
                row * (destination_stride_bytes // element_bytes),
                row_bytes // element_bytes,
            )
            reply_ptr = reply.access_ptr("rw", extent=1)
            row_body = tirx.SeqStmt(
                [
                    tirx.BufferStore(reply, 0, [0]),
                    tirx.Evaluate(
                        tirx.call_extern(
                            "int32",
                            semantic_name,
                            source,
                            destination,
                            row_bytes,
                            reply_ptr,
                        )
                    ),
                    tirx.Evaluate(
                        tirx.call_extern(
                            "int32",
                            "tilelang_sunway_dma_wait",
                            reply_ptr,
                            1,
                        )
                    ),
                ]
            )
            return tirx.For(
                row,
                0,
                rows,
                tirx.ForKind.SERIAL,
                row_body,
                span=getattr(node, "span", None),
            )

        body = tirx.stmt_functor.ir_transform(func.body, None, rewrite)
        if site_index != 3:
            raise ValueError(f"Sunway G0 requires three 2D DMA sites, found {site_index}")
        return func.with_body(tirx.SeqStmt([tirx.AllocBuffer(reply), body]))

    return map_prim_funcs(mod, rewrite_func)


def block_tile_extents(func: tirx.PrimFunc) -> tuple[int, int]:
    extents: dict[str, int] = {}

    def visit(node: object) -> None:
        if isinstance(node, tirx.For) and node.loop_var.name in {"bx", "by"}:
            extents[node.loop_var.name] = static_int(
                node.extent,
                f"{node.loop_var.name} extent",
            )

    tirx.stmt_functor.post_order_visit(func.body, visit)
    if set(extents) != {"bx", "by"}:
        raise ValueError("Sunway G0 requires serial bx/by block tile loops")
    return extents["by"], extents["bx"]


def bound_block_tile_extents(func: tirx.PrimFunc) -> tuple[int, int]:
    extents: dict[str, int] = {}

    def visit(node: object) -> None:
        if not isinstance(node, tirx.For) or node.thread_binding is None:
            return
        tag = str(node.thread_binding.thread_tag)
        if tag in {"blockIdx.x", "blockIdx.y"}:
            extents[tag] = static_int(node.extent, f"{tag} extent")

    tirx.stmt_functor.post_order_visit(func.body, visit)
    if set(extents) != {"blockIdx.x", "blockIdx.y"}:
        raise ValueError("Sunway G0 requires blockIdx.x/y tile bindings")
    return extents["blockIdx.y"], extents["blockIdx.x"]


def restore_serial_dimensions(
    mod: IRModule,
    y_name: str,
    y_extent: int,
    x_name: str,
    x_extent: int,
) -> IRModule:
    """Keep explicit schedule dimensions after extent-one loop cleanup."""

    def rewrite(func: tirx.PrimFunc) -> tirx.PrimFunc:
        loop_names: set[str] = set()
        tirx.stmt_functor.post_order_visit(
            func.body,
            lambda node: loop_names.add(node.loop_var.name) if isinstance(node, tirx.For) else None,
        )
        body = func.body
        missing_x = x_name not in loop_names
        missing_y = y_name not in loop_names
        if missing_x and missing_y:
            if not isinstance(body, tirx.SeqStmt) or not body.seq:
                raise ValueError(
                    "Sunway G0 cannot restore unit block dimensions around the S2 payload"
                )
            statements = list(body.seq)
            payload = statements[-1]
            payload = tirx.For(
                tirx.Var(y_name, "int32"),
                0,
                y_extent,
                tirx.ForKind.SERIAL,
                payload,
            )
            payload = tirx.For(
                tirx.Var(x_name, "int32"),
                0,
                x_extent,
                tirx.ForKind.SERIAL,
                payload,
            )
            statements[-1] = payload
            body = tirx.SeqStmt(statements)
        elif missing_x:
            inserted = False

            def insert_x(node: object) -> object | None:
                nonlocal inserted
                if inserted or not isinstance(node, tirx.For) or node.loop_var.name != y_name:
                    return None
                inserted = True
                return tirx.For(
                    tirx.Var(x_name, "int32"),
                    0,
                    x_extent,
                    tirx.ForKind.SERIAL,
                    node,
                )

            body = tirx.stmt_functor.ir_transform(body, None, insert_x)
            if not inserted:
                raise ValueError("Sunway G0 could not restore the unit bx dimension")
        elif missing_y:
            inserted = False

            def insert_y(node: object) -> object | None:
                nonlocal inserted
                if inserted or not isinstance(node, tirx.For) or node.loop_var.name != x_name:
                    return None
                inserted = True
                inner = tirx.For(
                    tirx.Var(y_name, "int32"),
                    0,
                    y_extent,
                    tirx.ForKind.SERIAL,
                    node.body,
                )
                return rewrite_for(node, inner, kind=tirx.ForKind.SERIAL)

            body = tirx.stmt_functor.ir_transform(body, None, insert_y)
            if not inserted:
                raise ValueError("Sunway G0 could not restore the unit by dimension")
        return func.with_body(body)

    return map_prim_funcs(mod, rewrite)


def restore_unit_block_dimensions(
    mod: IRModule,
    block_tiles_m: int,
    block_tiles_n: int,
) -> IRModule:
    return restore_serial_dimensions(mod, "by", block_tiles_m, "bx", block_tiles_n)


def restore_mesh_round_dimensions(mod: IRModule, plan: SunwayGemmPlan) -> IRModule:
    rounds_m = (plan.block_tiles_m + plan.cpe_rows - 1) // plan.cpe_rows
    rounds_n = (plan.block_tiles_n + plan.cpe_cols - 1) // plan.cpe_cols
    return restore_serial_dimensions(mod, "by_round", rounds_m, "bx_round", rounds_n)


def attach_gemm_plan(
    mod: IRModule,
    plan: SunwayGemmPlan,
    expected_block_tiles: tuple[int, int],
) -> IRModule:
    """Attach the S2 contract consumed by verification and code generation."""

    def rewrite(func: tirx.PrimFunc) -> tirx.PrimFunc:
        if plan.ownership == "single":
            block_tiles_m, block_tiles_n = block_tile_extents(func)
            if (block_tiles_m, block_tiles_n) != expected_block_tiles:
                raise ValueError("Sunway G0 block tile loops changed during S2 lowering")
        else:
            block_tiles_m, block_tiles_n = expected_block_tiles
        attrs = {
            "sunway.phase": "S2",
            "sunway.kernel_kind": "gemm_simd" if plan.compute == "simd" else "gemm_scalar",
            "sunway.cpe_count": plan.workers,
            "sunway.active_cpes": plan.active_cpes,
            "sunway.cpe_rows": plan.cpe_rows,
            "sunway.cpe_cols": plan.cpe_cols,
            "sunway.ownership": plan.ownership,
            "sunway.compute": plan.compute,
            "sunway.block_tiles_m": block_tiles_m,
            "sunway.block_tiles_n": block_tiles_n,
            "sunway.k_panels": plan.k_panels,
            "sunway.global_m": plan.global_m,
            "sunway.global_n": plan.global_n,
            "sunway.global_k": plan.global_k,
            "sunway.tile_m": plan.tile_m,
            "sunway.tile_n": plan.tile_n,
            "sunway.tile_k": plan.tile_k,
            "sunway.vector_width": plan.vector_width,
            "sunway.pipeline_stages": plan.stages,
            "sunway.ldm_bytes": plan.ldm_bytes,
        }
        for key, value in attrs.items():
            func = func.with_attr(key, value)
        return func

    return map_prim_funcs(mod, rewrite)
