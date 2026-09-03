"""Progressive S1-to-S2 lowering for the scalar Sunway GEMM baseline."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable

import tilelang
from tvm import IRModule, tirx

from .op.gemm.plan import SunwayGemmPlan
from .target import SunwayTargetConfig


_DMA_2D_CALLS = {
    "tilelang_sunway_dma_get_2d": "tilelang_sunway_dma_get",
    "tilelang_sunway_dma_put_2d": "tilelang_sunway_dma_put",
}
_DMA_CALLS = frozenset(_DMA_2D_CALLS.values())
_NATIVE_CALLS = frozenset({"_MYID", "athread_get", "athread_put", "tilelang_sunway_reply_wait"})
_SEMANTIC_CALLS = frozenset(
    {
        "tilelang_sunway_pe_id",
        "tilelang_sunway_dma_get",
        "tilelang_sunway_dma_put",
        "tilelang_sunway_dma_wait",
    }
)


def _map_prim_funcs(mod: IRModule, rewrite: Callable[[tirx.PrimFunc], tirx.PrimFunc]) -> IRModule:
    functions = {
        global_var: rewrite(func) if isinstance(func, tirx.PrimFunc) else func
        for global_var, func in mod.functions.items()
    }
    return IRModule(functions, attrs=mod.attrs, global_infos=mod.global_infos)


def _only_prim_func(mod: IRModule) -> tirx.PrimFunc:
    functions = [func for func in mod.functions.values() if isinstance(func, tirx.PrimFunc)]
    if len(functions) != 1:
        raise ValueError(f"Sunway G0 requires one PrimFunc per module, found {len(functions)}")
    return functions[0]


def _call_name(call: tirx.Call) -> str:
    op_name = str(getattr(call.op, "name", ""))
    if op_name == "tirx.call_extern" and call.args and isinstance(call.args[0], tirx.StringImm):
        return call.args[0].value
    return op_name


def _named_calls(node: object) -> list[str]:
    names: list[str] = []
    tirx.stmt_functor.post_order_visit(
        node,
        lambda child: names.append(_call_name(child)) if isinstance(child, tirx.Call) else None,
    )
    return names


def _static_int(value: tirx.PrimExpr, description: str) -> int:
    if not isinstance(value, tirx.IntImm):
        raise ValueError(f"Sunway G0 requires a static {description}, got {value}")
    return int(value)


def _rewrite_for(
    loop: tirx.For,
    body: tirx.Stmt,
    *,
    kind: tirx.ForKind,
    thread_binding=None,
) -> tirx.For:
    return tirx.For(
        loop.loop_var,
        loop.min,
        loop.extent,
        kind,
        body,
        thread_binding=thread_binding,
        annotations=loop.annotations,
        step=loop.step,
        span=getattr(loop, "span", None),
    )


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
                    tile_extent = _static_int(node.extent, f"{tag} extent")
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
                return _rewrite_for(node, node.body, kind=tirx.ForKind.SERIAL)
            if tag == "blockIdx.z":
                if _static_int(node.extent, tag) != 1:
                    raise ValueError("Sunway G0 requires blockIdx.z extent 1")
                return tirx.stmt_functor.substitute(node.body, {node.loop_var: node.min})
            if tag == "threadIdx.x":
                workers = _static_int(node.extent, "logical worker count")
                if workers != expected_workers:
                    raise ValueError(f"Sunway G0 requires {expected_workers} logical workers, got {workers}")
                thread_x_sites += 1
                worker_body = tirx.stmt_functor.substitute(node.body, {node.loop_var: pe_id})
                if plan.ownership == "mesh_2d":
                    return worker_body
                # TileOp layout lowering may fold its own guard to true; S2 owns
                # the target-visible single-CPE ownership decision explicitly.
                return tirx.IfThenElse(pe_id == 0, worker_body, None, span=getattr(node, "span", None))
            if tag in {"threadIdx.y", "threadIdx.z"}:
                if _static_int(node.extent, tag) != 1:
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
        body = tirx.SeqStmt([*bindings, body])
        return func.with_body(body)

    return _map_prim_funcs(mod, rewrite_func)


def _offset_access_ptr(
    pointer: tirx.PrimExpr,
    element_offset: tirx.PrimExpr,
    row_elements: int,
) -> tirx.Call:
    if not isinstance(pointer, tirx.Call) or str(getattr(pointer.op, "name", "")) != "tirx.tvm_access_ptr":
        raise ValueError("Sunway G0 DMA requires a canonical tvm_access_ptr")
    if len(pointer.args) != 5:
        raise ValueError("Sunway G0 DMA received an invalid tvm_access_ptr")
    arguments = list(pointer.args)
    arguments[2] = arguments[2] + element_offset
    arguments[3] = tirx.IntImm("int32", row_elements)
    return tirx.Call(str(pointer.dtype), pointer.op, arguments, span=getattr(pointer, "span", None))


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
            name = _call_name(call)
            semantic_name = _DMA_2D_CALLS.get(name)
            if semantic_name is None:
                return None
            if len(call.args) != 7:
                raise ValueError(f"Sunway G0 DMA descriptor {name} has an invalid argument count")

            rows = _static_int(call.args[3], "DMA row count")
            row_bytes = _static_int(call.args[4], "DMA row byte count")
            source_stride_bytes = _static_int(call.args[5], "DMA source byte stride")
            destination_stride_bytes = _static_int(call.args[6], "DMA destination byte stride")
            for value, description in (
                (row_bytes, "row byte count"),
                (source_stride_bytes, "source byte stride"),
                (destination_stride_bytes, "destination byte stride"),
            ):
                if value % element_bytes:
                    raise ValueError(f"Sunway G0 DMA {description} must be divisible by {element_bytes}")
            if row_bytes % config.dma_alignment:
                raise ValueError(
                    f"Sunway G0 DMA row byte count {row_bytes} violates "
                    f"{config.dma_alignment}-byte alignment"
                )

            row = tirx.Var(f"sunway_dma_row_{site_index}", "int32")
            site_index += 1
            source = _offset_access_ptr(
                call.args[1], row * (source_stride_bytes // element_bytes), row_bytes // element_bytes
            )
            destination = _offset_access_ptr(
                call.args[2], row * (destination_stride_bytes // element_bytes), row_bytes // element_bytes
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
            return tirx.For(row, 0, rows, tirx.ForKind.SERIAL, row_body, span=getattr(node, "span", None))

        body = tirx.stmt_functor.ir_transform(func.body, None, rewrite)
        if site_index != 3:
            raise ValueError(f"Sunway G0 requires three 2D DMA sites, found {site_index}")
        return func.with_body(tirx.SeqStmt([tirx.AllocBuffer(reply), body]))

    return _map_prim_funcs(mod, rewrite_func)


def _block_tile_extents(func: tirx.PrimFunc) -> tuple[int, int]:
    extents: dict[str, int] = {}

    def visit(node: object) -> None:
        if isinstance(node, tirx.For) and node.loop_var.name in {"bx", "by"}:
            extents[node.loop_var.name] = _static_int(node.extent, f"{node.loop_var.name} extent")

    tirx.stmt_functor.post_order_visit(func.body, visit)
    if set(extents) != {"bx", "by"}:
        raise ValueError("Sunway G0 requires serial bx/by block tile loops")
    return extents["by"], extents["bx"]


def _bound_block_tile_extents(func: tirx.PrimFunc) -> tuple[int, int]:
    extents: dict[str, int] = {}

    def visit(node: object) -> None:
        if not isinstance(node, tirx.For) or node.thread_binding is None:
            return
        tag = str(node.thread_binding.thread_tag)
        if tag in {"blockIdx.x", "blockIdx.y"}:
            extents[tag] = _static_int(node.extent, f"{tag} extent")

    tirx.stmt_functor.post_order_visit(func.body, visit)
    if set(extents) != {"blockIdx.x", "blockIdx.y"}:
        raise ValueError("Sunway G0 requires blockIdx.x/y tile bindings")
    return extents["blockIdx.y"], extents["blockIdx.x"]


def _restore_serial_dimensions(
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
                raise ValueError("Sunway G0 cannot restore unit block dimensions around the S2 payload")
            statements = list(body.seq)
            payload = statements[-1]
            payload = tirx.For(tirx.Var(y_name, "int32"), 0, y_extent, tirx.ForKind.SERIAL, payload)
            payload = tirx.For(tirx.Var(x_name, "int32"), 0, x_extent, tirx.ForKind.SERIAL, payload)
            statements[-1] = payload
            body = tirx.SeqStmt(statements)
        elif missing_x:
            inserted = False

            def insert_x(node: object) -> object | None:
                nonlocal inserted
                if inserted or not isinstance(node, tirx.For) or node.loop_var.name != y_name:
                    return None
                inserted = True
                # LowerOpaqueBlock legally removes extent-one loops. Restore
                # bx immediately around by, below function-level resources.
                return tirx.For(tirx.Var(x_name, "int32"), 0, x_extent, tirx.ForKind.SERIAL, node)

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
                inner = tirx.For(tirx.Var(y_name, "int32"), 0, y_extent, tirx.ForKind.SERIAL, node.body)
                return _rewrite_for(node, inner, kind=tirx.ForKind.SERIAL)

            body = tirx.stmt_functor.ir_transform(body, None, insert_y)
            if not inserted:
                raise ValueError("Sunway G0 could not restore the unit by dimension")
        return func.with_body(body)

    return _map_prim_funcs(mod, rewrite)


def _restore_unit_block_dimensions(
    mod: IRModule,
    block_tiles_m: int,
    block_tiles_n: int,
) -> IRModule:
    return _restore_serial_dimensions(mod, "by", block_tiles_m, "bx", block_tiles_n)


def _restore_mesh_round_dimensions(mod: IRModule, plan: SunwayGemmPlan) -> IRModule:
    rounds_m = (plan.block_tiles_m + plan.cpe_rows - 1) // plan.cpe_rows
    rounds_n = (plan.block_tiles_n + plan.cpe_cols - 1) // plan.cpe_cols
    return _restore_serial_dimensions(mod, "by_round", rounds_m, "bx_round", rounds_n)


def attach_scalar_gemm_plan(
    mod: IRModule,
    plan: SunwayGemmPlan,
    expected_block_tiles: tuple[int, int],
) -> IRModule:
    """Attach the S2 contract consumed by verification, codegen, and manifests."""

    def rewrite(func: tirx.PrimFunc) -> tirx.PrimFunc:
        if plan.ownership == "single":
            block_tiles_m, block_tiles_n = _block_tile_extents(func)
            if (block_tiles_m, block_tiles_n) != expected_block_tiles:
                raise ValueError("Sunway G0 block tile loops changed during S2 lowering")
        else:
            block_tiles_m, block_tiles_n = expected_block_tiles
        attrs = {
            "sunway.phase": "S2",
            "sunway.kernel_kind": "gemm_scalar",
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

    return _map_prim_funcs(mod, rewrite)


def lower_gemm_program_to_semantic_tir(mod: IRModule, config: SunwayTargetConfig) -> IRModule:
    """Lower one canonical tiled GEMM from S1 to inspectable semantic S2 TIR."""

    source_func = _only_prim_func(mod)
    plan = SunwayGemmPlan.from_prim_func(source_func, config)
    if plan.compute == "simd":
        raise ValueError("Sunway G1 SIMD compute lowering is not installed")
    block_tiles = _bound_block_tile_extents(source_func)
    mod = tilelang.transform.LayoutInference()(mod)
    mod = tilelang.transform.LowerTileOp()(mod)
    mod = lower_sunway_kernel_bindings(mod, config, plan)
    mod = expand_sunway_dma_2d(mod, config, plan)
    mod = tilelang.transform.PlanAndUpdateBufferAllocationLocation()(mod)
    mod = tilelang.transform.LowerOpaqueBlock()(mod)
    mod = tirx.transform.Simplify()(mod)
    if plan.ownership == "mesh_2d":
        mod = _restore_mesh_round_dimensions(mod, plan)
    else:
        mod = _restore_unit_block_dimensions(mod, *block_tiles)
    return attach_scalar_gemm_plan(mod, plan, block_tiles)


def _is_worker_zero(condition: tirx.PrimExpr) -> bool:
    if not isinstance(condition, tirx.EQ):
        return False
    operands = (condition.a, condition.b)
    has_zero = any(isinstance(value, tirx.IntImm) and int(value) == 0 for value in operands)
    has_pe_id = any(isinstance(value, tirx.Var) and value.name == "pe_id" for value in operands)
    return has_zero and has_pe_id


def _uses_var(node: object, name: str) -> bool:
    found = False

    def visit(candidate: object) -> None:
        nonlocal found
        if isinstance(candidate, tirx.Var) and candidate.name == name:
            found = True

    tirx.stmt_functor.post_order_visit(node, visit)
    return found


def _contains_node_type(node: object, node_type: type) -> bool:
    found = False

    def visit(candidate: object) -> None:
        nonlocal found
        if isinstance(candidate, node_type):
            found = True

    tirx.stmt_functor.post_order_visit(node, visit)
    return found


def _mesh_owner_count(block_tiles_m: int, block_tiles_n: int, rows: int, cols: int) -> int:
    owners = {
        (by % rows) * cols + (bx % cols)
        for by in range(block_tiles_m)
        for bx in range(block_tiles_n)
    }
    return len(owners)


def _required_positive_attr(func: tirx.PrimFunc, key: str) -> int:
    value = func.attrs.get(key)
    if value is None:
        raise ValueError(f"Sunway S2 GEMM invariant failed: required attribute {key} is missing")
    result = int(value)
    if result <= 0:
        raise ValueError(f"Sunway S2 GEMM invariant failed: {key} must be positive, got {result}")
    return result


def _verify_gemm_semantic_func(func: tirx.PrimFunc, config: SunwayTargetConfig) -> tirx.PrimFunc:
    if str(func.attrs.get("sunway.phase", "")) != "S2":
        raise ValueError("Sunway S2 GEMM verifier received a function from another phase")
    if str(func.attrs.get("sunway.kernel_kind", "")) != "gemm_scalar":
        raise ValueError("Sunway S2 GEMM verifier received another kernel kind")

    metadata = {
        key: _required_positive_attr(func, key)
        for key in (
            "sunway.cpe_count",
            "sunway.active_cpes",
            "sunway.cpe_rows",
            "sunway.cpe_cols",
            "sunway.block_tiles_m",
            "sunway.block_tiles_n",
            "sunway.k_panels",
            "sunway.tile_m",
            "sunway.tile_n",
            "sunway.tile_k",
            "sunway.vector_width",
            "sunway.pipeline_stages",
            "sunway.ldm_bytes",
        )
    }
    ownership = str(func.attrs.get("sunway.ownership", ""))
    compute = str(func.attrs.get("sunway.compute", ""))
    if ownership not in {"single", "mesh_2d"}:
        raise ValueError(f"Sunway S2 GEMM invariant failed: unsupported ownership {ownership!r}")
    if compute not in {"scalar", "simd"}:
        raise ValueError(f"Sunway S2 GEMM invariant failed: unsupported compute mode {compute!r}")
    expected_workers = config.cpe_rows * config.cpe_cols
    if metadata["sunway.cpe_count"] != expected_workers:
        raise ValueError("Sunway S2 GEMM invariant failed: CPE count does not match the target mesh")
    if metadata["sunway.cpe_rows"] != config.cpe_rows or metadata["sunway.cpe_cols"] != config.cpe_cols:
        raise ValueError("Sunway S2 GEMM invariant failed: CPE mesh metadata is inconsistent")
    expected_active_cpes = 1
    if ownership == "mesh_2d":
        expected_active_cpes = _mesh_owner_count(
            metadata["sunway.block_tiles_m"],
            metadata["sunway.block_tiles_n"],
            config.cpe_rows,
            config.cpe_cols,
        )
    if metadata["sunway.active_cpes"] != expected_active_cpes:
        raise ValueError("Sunway S2 GEMM invariant failed: active CPE metadata is inconsistent")
    if metadata["sunway.pipeline_stages"] != 1:
        raise ValueError("Sunway S2 GEMM invariant failed: G0 requires one pipeline stage")
    if metadata["sunway.ldm_bytes"] > config.ldm_bytes_per_cpe:
        raise ValueError(
            f"Sunway S2 GEMM invariant failed: LDM plan uses {metadata['sunway.ldm_bytes']} bytes, "
            f"but target limit is {config.ldm_bytes_per_cpe}"
        )

    names = _named_calls(func.body)
    counts = Counter(names)
    native = sorted(set(names) & _NATIVE_CALLS)
    if native:
        raise ValueError(f"Sunway S2 GEMM invariant failed: native ABI call {native[0]!r} appeared before S3")
    tile_ops = sorted(name for name in set(names) if name.startswith("tl.tileop."))
    if tile_ops:
        raise ValueError(f"Sunway S2 GEMM invariant failed: residual TileOp {tile_ops[0]!r}")
    expected_calls = {
        "tilelang_sunway_pe_id": 1,
        "tilelang_sunway_dma_get": 2,
        "tilelang_sunway_dma_put": 1,
        "tilelang_sunway_dma_wait": 3,
    }
    for name, expected in expected_calls.items():
        if counts[name] != expected:
            raise ValueError(
                f"Sunway S2 GEMM invariant failed: expected {expected} {name} call(s), found {counts[name]}"
            )

    loops: list[tirx.For] = []
    owners: list[tirx.IfThenElse] = []
    binds: list[tirx.Bind] = []

    def visit(node: object) -> None:
        if isinstance(node, tirx.For):
            loops.append(node)
        elif isinstance(node, tirx.IfThenElse) and _is_worker_zero(node.condition):
            owners.append(node)
        elif isinstance(node, tirx.Bind):
            binds.append(node)

    tirx.stmt_functor.post_order_visit(func.body, visit)
    if any(loop.thread_binding is not None for loop in loops):
        raise ValueError("Sunway S2 GEMM invariant failed: thread binding remains after ownership lowering")
    bind_map = {bind.var.name: bind for bind in binds}
    if ownership == "single":
        block_loops = {loop.loop_var.name: loop for loop in loops if loop.loop_var.name in {"bx", "by"}}
        if set(block_loops) != {"bx", "by"} or any(
            loop.kind != tirx.ForKind.SERIAL for loop in block_loops.values()
        ):
            raise ValueError("Sunway S2 GEMM invariant failed: block tiles must be static serial bx/by loops")
        if int(block_loops["by"].extent) != metadata["sunway.block_tiles_m"] or int(
            block_loops["bx"].extent
        ) != metadata["sunway.block_tiles_n"]:
            raise ValueError("Sunway S2 GEMM invariant failed: block tile loop metadata is inconsistent")
        if set(bind_map) != {"pe_id"}:
            raise ValueError("Sunway S2 GEMM invariant failed: expected one semantic PE id binding")

        owned_names = [name for owner in owners for name in _named_calls(owner.then_case)]
        if Counter(owned_names)["tilelang_sunway_dma_get"] != 2 or Counter(owned_names)[
            "tilelang_sunway_dma_put"
        ] != 1:
            raise ValueError("Sunway S2 GEMM invariant failed: every DMA site must be owned by CPE zero")
    else:
        if owners:
            raise ValueError("Sunway S2 GEMM invariant failed: mesh ownership retained a CPE-zero guard")
        if set(bind_map) != {"pe_id", "pe_row", "pe_col"}:
            raise ValueError("Sunway S2 GEMM invariant failed: mesh coordinate bindings are incomplete")
        if _call_name(bind_map["pe_id"].value) != "tilelang_sunway_pe_id":
            raise ValueError("Sunway S2 GEMM invariant failed: PE id binding is not semantic")
        row_value = bind_map["pe_row"].value
        col_value = bind_map["pe_col"].value
        if not isinstance(row_value, tirx.FloorDiv) or not _uses_var(row_value, "pe_id"):
            raise ValueError("Sunway S2 GEMM invariant failed: PE row mapping is invalid")
        if not isinstance(col_value, tirx.FloorMod) or not _uses_var(col_value, "pe_id"):
            raise ValueError("Sunway S2 GEMM invariant failed: PE column mapping is invalid")
        round_loops = {
            loop.loop_var.name: loop
            for loop in loops
            if loop.loop_var.name in {"bx_round", "by_round"}
        }
        expected_rounds = {
            "by_round": (
                metadata["sunway.block_tiles_m"] + config.cpe_rows - 1
            )
            // config.cpe_rows,
            "bx_round": (
                metadata["sunway.block_tiles_n"] + config.cpe_cols - 1
            )
            // config.cpe_cols,
        }
        if set(round_loops) != set(expected_rounds) or any(
            loop.kind != tirx.ForKind.SERIAL
            or int(loop.extent) != expected_rounds[name]
            for name, loop in round_loops.items()
        ):
            raise ValueError("Sunway S2 GEMM invariant failed: mesh round loops are invalid")
        guard_conditions: list[tirx.PrimExpr] = []

        def collect_guard(node: object) -> None:
            if isinstance(node, tirx.IfThenElse):
                guard_conditions.append(node.condition)

        tirx.stmt_functor.post_order_visit(func.body, collect_guard)
        has_row_guard = any(
            _uses_var(condition, "pe_row")
            or (
                _uses_var(condition, "pe_id")
                and _contains_node_type(condition, tirx.FloorDiv)
            )
            for condition in guard_conditions
        )
        has_col_guard = any(
            _uses_var(condition, "pe_col")
            or (
                _uses_var(condition, "pe_id")
                and _contains_node_type(condition, tirx.FloorMod)
            )
            for condition in guard_conditions
        )
        needs_row_guard = metadata["sunway.block_tiles_m"] % config.cpe_rows != 0
        needs_col_guard = metadata["sunway.block_tiles_n"] % config.cpe_cols != 0
        if (needs_row_guard and not has_row_guard) or (needs_col_guard and not has_col_guard):
            raise ValueError("Sunway S2 GEMM invariant failed: mesh bounds guards are incomplete")

    dma_row_loops = [loop for loop in loops if loop.loop_var.name.startswith("sunway_dma_row_")]
    if len(dma_row_loops) != 3:
        raise ValueError("Sunway S2 GEMM invariant failed: expected three row-transfer loops")
    for loop in dma_row_loops:
        row_names = _named_calls(loop.body)
        issues = [name for name in row_names if name in _DMA_CALLS]
        waits = [name for name in row_names if name == "tilelang_sunway_dma_wait"]
        if len(issues) != 1 or len(waits) != 1 or row_names.index(issues[0]) > row_names.index(waits[0]):
            raise ValueError("Sunway S2 GEMM invariant failed: each DMA issue must be followed by one wait")
        issue_calls: list[tirx.Call] = []

        def collect_issue(node: object, calls: list[tirx.Call] = issue_calls) -> None:
            if isinstance(node, tirx.Call) and _call_name(node) in _DMA_CALLS:
                calls.append(node)

        tirx.stmt_functor.post_order_visit(
            loop.body,
            collect_issue,
        )
        row_bytes = _static_int(issue_calls[0].args[3], "DMA row byte count")
        if row_bytes % config.dma_alignment:
            raise ValueError("Sunway S2 GEMM invariant failed: DMA row bytes are not aligned")
    return func


def verify_gemm_semantic_tir(mod: IRModule, config: SunwayTargetConfig) -> IRModule:
    """Verify that semantic S2 completely owns the scalar GEMM schedule."""

    return _map_prim_funcs(mod, lambda func: _verify_gemm_semantic_func(func, config))


def _is_static_compact(buffer: tirx.Buffer) -> bool:
    if not buffer.shape or any(not isinstance(extent, tirx.IntImm) or int(extent) <= 0 for extent in buffer.shape):
        return False
    if not buffer.strides:
        return True
    if len(buffer.strides) != len(buffer.shape) or any(not isinstance(stride, tirx.IntImm) for stride in buffer.strides):
        return False
    expected = 1
    for extent, stride in zip(reversed(buffer.shape), reversed(buffer.strides), strict=True):
        if int(stride) != expected:
            return False
        expected *= int(extent)
    return True


def _verify_gemm_native_func(func: tirx.PrimFunc, config: SunwayTargetConfig) -> tirx.PrimFunc:
    if str(func.attrs.get("sunway.phase", "")) != "S3":
        raise ValueError("Sunway S3 GEMM verifier received a function from another phase")
    if str(func.attrs.get("sunway.kernel_kind", "")) != "gemm_scalar":
        raise ValueError("Sunway S3 GEMM verifier received another kernel kind")

    metadata = {
        key: _required_positive_attr(func, key)
        for key in (
            "sunway.cpe_count",
            "sunway.active_cpes",
            "sunway.cpe_rows",
            "sunway.cpe_cols",
            "sunway.block_tiles_m",
            "sunway.block_tiles_n",
            "sunway.k_panels",
            "sunway.tile_m",
            "sunway.tile_n",
            "sunway.tile_k",
            "sunway.vector_width",
            "sunway.pipeline_stages",
            "sunway.ldm_bytes",
        )
    }
    ownership = str(func.attrs.get("sunway.ownership", ""))
    expected_workers = config.cpe_rows * config.cpe_cols
    expected_active_cpes = 1
    if ownership == "mesh_2d":
        expected_active_cpes = _mesh_owner_count(
            metadata["sunway.block_tiles_m"],
            metadata["sunway.block_tiles_n"],
            config.cpe_rows,
            config.cpe_cols,
        )
    if (
        metadata["sunway.cpe_count"] != expected_workers
        or metadata["sunway.cpe_rows"] != config.cpe_rows
        or metadata["sunway.cpe_cols"] != config.cpe_cols
        or metadata["sunway.active_cpes"] != expected_active_cpes
    ):
        raise ValueError("Sunway S3 GEMM invariant failed: invalid CPE ownership metadata")
    if metadata["sunway.pipeline_stages"] != 1:
        raise ValueError("Sunway S3 GEMM invariant failed: G0 requires one pipeline stage")
    if metadata["sunway.ldm_bytes"] > config.ldm_bytes_per_cpe:
        raise ValueError(
            f"Sunway S3 GEMM invariant failed: LDM plan uses {metadata['sunway.ldm_bytes']} bytes, "
            f"but target limit is {config.ldm_bytes_per_cpe}"
        )

    names = _named_calls(func.body)
    counts = Counter(names)
    semantic = sorted(set(names) & _SEMANTIC_CALLS)
    if semantic:
        raise ValueError(f"Sunway S3 GEMM invariant failed: semantic call {semantic[0]!r} remains")
    tile_ops = sorted(name for name in set(names) if name.startswith("tl.tileop."))
    if tile_ops:
        raise ValueError(f"Sunway S3 GEMM invariant failed: residual TileOp {tile_ops[0]!r}")
    expected_calls = {
        "_MYID": 1,
        "athread_get": 2,
        "athread_put": 1,
        "tilelang_sunway_reply_wait": 3,
    }
    for name, expected in expected_calls.items():
        if counts[name] != expected:
            raise ValueError(
                f"Sunway S3 GEMM invariant failed: expected {expected} {name} call(s), found {counts[name]}"
            )

    loops: list[tirx.For] = []
    allocated: list[tirx.Buffer] = []

    def visit(node: object) -> None:
        if isinstance(node, tirx.For):
            loops.append(node)
        elif isinstance(node, tirx.AllocBuffer):
            allocated.append(node.buffer)

    tirx.stmt_functor.post_order_visit(func.body, visit)
    if any(loop.thread_binding is not None for loop in loops):
        raise ValueError("Sunway S3 GEMM invariant failed: thread binding remains")
    buffers = [*func.buffer_map.values(), *allocated]
    if not buffers or any(not _is_static_compact(buffer) for buffer in buffers):
        raise ValueError("Sunway S3 GEMM invariant failed: expected statically shaped compact buffers")

    float_multiply = False
    float_add = False
    has_load = False
    has_store = False

    def visit_arithmetic(node: object) -> None:
        nonlocal float_multiply, float_add, has_load, has_store
        if isinstance(node, tirx.Mul) and str(node.dtype) == "float32":
            float_multiply = True
        elif isinstance(node, tirx.Add) and str(node.dtype) == "float32":
            float_add = True
        elif isinstance(node, tirx.BufferLoad):
            has_load = True
        elif isinstance(node, tirx.BufferStore):
            has_store = True

    tirx.stmt_functor.post_order_visit(func.body, visit_arithmetic)
    if not (float_multiply and float_add and has_load and has_store):
        raise ValueError("Sunway S3 GEMM invariant failed: scalar FP32 multiply/add is missing")

    dimension_names = {"bx", "by"} if ownership == "single" else {"bx_round", "by_round"}
    block_loops = {loop.loop_var.name: loop for loop in loops if loop.loop_var.name in dimension_names}
    if set(block_loops) != dimension_names or any(
        loop.kind != tirx.ForKind.SERIAL for loop in block_loops.values()
    ):
        raise ValueError("Sunway S3 GEMM invariant failed: ownership loops must remain static and serial")
    dma_row_loops = [loop for loop in loops if loop.loop_var.name.startswith("sunway_dma_row_")]
    if len(dma_row_loops) != 3:
        raise ValueError("Sunway S3 GEMM invariant failed: expected three native row-transfer loops")
    for loop in dma_row_loops:
        row_names = _named_calls(loop.body)
        issues = [name for name in row_names if name in {"athread_get", "athread_put"}]
        waits = [name for name in row_names if name == "tilelang_sunway_reply_wait"]
        if len(issues) != 1 or len(waits) != 1 or row_names.index(issues[0]) > row_names.index(waits[0]):
            raise ValueError("Sunway S3 GEMM invariant failed: native DMA issue/wait ordering is invalid")
    return func


def verify_gemm_native_tir(mod: IRModule, config: SunwayTargetConfig) -> IRModule:
    """Verify codegen-ready G0 TIR after resolving target ABI leaves."""

    return _map_prim_funcs(mod, lambda func: _verify_gemm_native_func(func, config))


def lower_gemm_semantic_to_native_tir(mod: IRModule, config: SunwayTargetConfig) -> IRModule:
    """Resolve semantic leaves and prepare scalar GEMM TIR for mechanical C emission."""

    verify_gemm_semantic_tir(mod, config)
    from .transform import _lower_semantic_calls_to_native

    block_tiles_m = int(_only_prim_func(mod).attrs["sunway.block_tiles_m"])
    block_tiles_n = int(_only_prim_func(mod).attrs["sunway.block_tiles_n"])
    ownership = str(_only_prim_func(mod).attrs.get("sunway.ownership", "single"))
    mod = _lower_semantic_calls_to_native(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tirx.transform.Simplify()(mod)
    mod = tirx.transform.RemoveNoOp()(mod)
    if ownership == "mesh_2d":
        rounds_m = (block_tiles_m + config.cpe_rows - 1) // config.cpe_rows
        rounds_n = (block_tiles_n + config.cpe_cols - 1) // config.cpe_cols
        mod = _restore_serial_dimensions(mod, "by_round", rounds_m, "bx_round", rounds_n)
    else:
        mod = _restore_unit_block_dimensions(mod, block_tiles_m, block_tiles_n)
    return verify_gemm_native_tir(mod, config)
