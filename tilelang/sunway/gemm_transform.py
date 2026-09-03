"""Progressive S1-to-S2 lowering for the scalar Sunway GEMM baseline."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable

import tilelang
from tvm import IRModule, tirx

from .op.gemm.plan import SunwayScalarGemmPlan
from .target import SunwayTargetConfig


_DMA_2D_CALLS = {
    "tilelang_sunway_dma_get_2d": "tilelang_sunway_dma_get",
    "tilelang_sunway_dma_put_2d": "tilelang_sunway_dma_put",
}
_DMA_CALLS = frozenset(_DMA_2D_CALLS.values())
_NATIVE_CALLS = frozenset({"_MYID", "athread_get", "athread_put", "tilelang_sunway_reply_wait"})


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


def lower_sunway_kernel_bindings(mod: IRModule, config: SunwayTargetConfig) -> IRModule:
    """Serialize block tiles and map the logical worker binding to the CPE id."""

    expected_workers = config.cpe_rows * config.cpe_cols

    def rewrite_func(func: tirx.PrimFunc) -> tirx.PrimFunc:
        pe_id = tirx.Var("pe_id", "int32")
        thread_x_sites = 0

        def rewrite(node: object) -> object | None:
            nonlocal thread_x_sites
            if not isinstance(node, tirx.For) or node.thread_binding is None:
                return None
            tag = str(node.thread_binding.thread_tag)
            if tag in {"blockIdx.x", "blockIdx.y"}:
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
        body = tirx.SeqStmt(
            [
                tirx.Bind(pe_id, tirx.call_extern("int32", "tilelang_sunway_pe_id")),
                body,
            ]
        )
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
    plan: SunwayScalarGemmPlan,
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


def _restore_unit_block_dimensions(
    mod: IRModule,
    block_tiles_m: int,
    block_tiles_n: int,
) -> IRModule:
    """Keep the explicit two-dimensional tile grid after opaque-block cleanup."""

    def rewrite(func: tirx.PrimFunc) -> tirx.PrimFunc:
        loop_names: set[str] = set()
        tirx.stmt_functor.post_order_visit(
            func.body,
            lambda node: loop_names.add(node.loop_var.name) if isinstance(node, tirx.For) else None,
        )
        body = func.body
        missing_x = "bx" not in loop_names
        missing_y = "by" not in loop_names
        if missing_x and missing_y:
            if not isinstance(body, tirx.SeqStmt) or not body.seq:
                raise ValueError("Sunway G0 cannot restore unit block dimensions around the S2 payload")
            statements = list(body.seq)
            payload = statements[-1]
            payload = tirx.For(tirx.Var("by", "int32"), 0, block_tiles_m, tirx.ForKind.SERIAL, payload)
            payload = tirx.For(tirx.Var("bx", "int32"), 0, block_tiles_n, tirx.ForKind.SERIAL, payload)
            statements[-1] = payload
            body = tirx.SeqStmt(statements)
        elif missing_x:
            inserted = False

            def insert_x(node: object) -> object | None:
                nonlocal inserted
                if inserted or not isinstance(node, tirx.For) or node.loop_var.name != "by":
                    return None
                inserted = True
                # LowerOpaqueBlock legally removes extent-one loops. Restore
                # bx immediately around by, below function-level resources.
                return tirx.For(tirx.Var("bx", "int32"), 0, block_tiles_n, tirx.ForKind.SERIAL, node)

            body = tirx.stmt_functor.ir_transform(body, None, insert_x)
            if not inserted:
                raise ValueError("Sunway G0 could not restore the unit bx dimension")
        elif missing_y:
            inserted = False

            def insert_y(node: object) -> object | None:
                nonlocal inserted
                if inserted or not isinstance(node, tirx.For) or node.loop_var.name != "bx":
                    return None
                inserted = True
                inner = tirx.For(tirx.Var("by", "int32"), 0, block_tiles_m, tirx.ForKind.SERIAL, node.body)
                return _rewrite_for(node, inner, kind=tirx.ForKind.SERIAL)

            body = tirx.stmt_functor.ir_transform(body, None, insert_y)
            if not inserted:
                raise ValueError("Sunway G0 could not restore the unit by dimension")
        return func.with_body(body)

    return _map_prim_funcs(mod, rewrite)


def attach_scalar_gemm_plan(
    mod: IRModule,
    plan: SunwayScalarGemmPlan,
    expected_block_tiles: tuple[int, int],
) -> IRModule:
    """Attach the S2 contract consumed by verification, codegen, and manifests."""

    def rewrite(func: tirx.PrimFunc) -> tirx.PrimFunc:
        block_tiles_m, block_tiles_n = _block_tile_extents(func)
        if (block_tiles_m, block_tiles_n) != expected_block_tiles:
            raise ValueError("Sunway G0 block tile loops changed during S2 lowering")
        attrs = {
            "sunway.phase": "S2",
            "sunway.kernel_kind": "gemm_scalar",
            "sunway.cpe_count": plan.workers,
            "sunway.active_cpes": 1,
            "sunway.block_tiles_m": block_tiles_m,
            "sunway.block_tiles_n": block_tiles_n,
            "sunway.tile_m": plan.tile_m,
            "sunway.tile_n": plan.tile_n,
            "sunway.tile_k": plan.tile_k,
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
    plan = SunwayScalarGemmPlan.from_prim_func(source_func, config)
    block_tiles = _bound_block_tile_extents(source_func)
    mod = tilelang.transform.LayoutInference()(mod)
    mod = tilelang.transform.LowerTileOp()(mod)
    mod = lower_sunway_kernel_bindings(mod, config)
    mod = expand_sunway_dma_2d(mod, config, plan)
    mod = tilelang.transform.PlanAndUpdateBufferAllocationLocation()(mod)
    mod = tilelang.transform.LowerOpaqueBlock()(mod)
    mod = tirx.transform.Simplify()(mod)
    mod = _restore_unit_block_dimensions(mod, *block_tiles)
    return attach_scalar_gemm_plan(mod, plan, block_tiles)


def _is_worker_zero(condition: tirx.PrimExpr) -> bool:
    if not isinstance(condition, tirx.EQ):
        return False
    operands = (condition.a, condition.b)
    has_zero = any(isinstance(value, tirx.IntImm) and int(value) == 0 for value in operands)
    has_pe_id = any(isinstance(value, tirx.Var) and value.name == "pe_id" for value in operands)
    return has_zero and has_pe_id


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
            "sunway.block_tiles_m",
            "sunway.block_tiles_n",
            "sunway.tile_m",
            "sunway.tile_n",
            "sunway.tile_k",
            "sunway.pipeline_stages",
            "sunway.ldm_bytes",
        )
    }
    expected_workers = config.cpe_rows * config.cpe_cols
    if metadata["sunway.cpe_count"] != expected_workers:
        raise ValueError("Sunway S2 GEMM invariant failed: CPE count does not match the target mesh")
    if metadata["sunway.active_cpes"] != 1:
        raise ValueError("Sunway S2 GEMM invariant failed: G0 must activate exactly one CPE")
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
    block_loops = {loop.loop_var.name: loop for loop in loops if loop.loop_var.name in {"bx", "by"}}
    if set(block_loops) != {"bx", "by"} or any(
        loop.kind != tirx.ForKind.SERIAL for loop in block_loops.values()
    ):
        raise ValueError("Sunway S2 GEMM invariant failed: block tiles must be static serial bx/by loops")
    if int(block_loops["by"].extent) != metadata["sunway.block_tiles_m"] or int(
        block_loops["bx"].extent
    ) != metadata["sunway.block_tiles_n"]:
        raise ValueError("Sunway S2 GEMM invariant failed: block tile loop metadata is inconsistent")
    if len(binds) != 1 or binds[0].var.name != "pe_id":
        raise ValueError("Sunway S2 GEMM invariant failed: expected one semantic PE id binding")

    owned_names = [name for owner in owners for name in _named_calls(owner.then_case)]
    if Counter(owned_names)["tilelang_sunway_dma_get"] != 2 or Counter(owned_names)[
        "tilelang_sunway_dma_put"
    ] != 1:
        raise ValueError("Sunway S2 GEMM invariant failed: every DMA site must be owned by CPE zero")

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
