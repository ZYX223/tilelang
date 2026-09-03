"""S2 and S3 legality checks for Sunway GEMM schedules."""

from __future__ import annotations

from collections import Counter

from tvm import IRModule, arith, tirx

from ...target import SunwayTargetConfig
from .utils import call_name, map_prim_funcs, named_calls, static_int


_DMA_CALLS = frozenset({"tilelang_sunway_dma_get", "tilelang_sunway_dma_put"})
_NATIVE_CALLS = frozenset(
    {
        "_MYID",
        "athread_get",
        "athread_put",
        "tilelang_sunway_reply_wait",
        "tilelang_sunway_native_fma_f32x8",
    }
)
_SEMANTIC_CALLS = frozenset(
    {
        "tilelang_sunway_pe_id",
        "tilelang_sunway_dma_get",
        "tilelang_sunway_dma_put",
        "tilelang_sunway_dma_wait",
        "tilelang_sunway_fma_f32x8",
    }
)


def _is_worker_zero(condition: tirx.PrimExpr) -> bool:
    if not isinstance(condition, tirx.EQ):
        return False
    operands = (condition.a, condition.b)
    has_zero = any(
        isinstance(value, tirx.IntImm) and int(value) == 0 for value in operands
    )
    has_pe_id = any(
        isinstance(value, tirx.Var) and value.name == "pe_id" for value in operands
    )
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


def _mesh_owner_count(
    block_tiles_m: int,
    block_tiles_n: int,
    rows: int,
    cols: int,
) -> int:
    owners = {
        (by % rows) * cols + (bx % cols)
        for by in range(block_tiles_m)
        for bx in range(block_tiles_n)
    }
    return len(owners)


def _required_positive_attr(func: tirx.PrimFunc, key: str) -> int:
    value = func.attrs.get(key)
    if value is None:
        raise ValueError(
            f"Sunway S2 GEMM invariant failed: required attribute {key} is missing"
        )
    result = int(value)
    if result <= 0:
        raise ValueError(
            f"Sunway S2 GEMM invariant failed: {key} must be positive, got {result}"
        )
    return result


def _verify_f32x8_call(
    func: tirx.PrimFunc,
    *,
    call_name_: str,
    k_panels: int,
    phase: str,
) -> None:
    calls: list[tirx.Call] = []
    k_panel_loops: list[tirx.For] = []

    def visit(node: object) -> None:
        if isinstance(node, tirx.Call) and call_name(node) == call_name_:
            calls.append(node)
        elif isinstance(node, tirx.For) and node.loop_var.name == "ko":
            k_panel_loops.append(node)

    tirx.stmt_functor.post_order_visit(func.body, visit)
    if len(calls) != 1:
        raise ValueError(
            f"Sunway {phase} GEMM invariant failed: expected one {call_name_} call"
        )
    if len(k_panel_loops) != 1 or int(k_panel_loops[0].extent) != k_panels:
        raise ValueError(f"Sunway {phase} GEMM invariant failed: invalid K-panel loop")
    if named_calls(k_panel_loops[0].body).count(call_name_) != 1:
        raise ValueError(
            f"Sunway {phase} GEMM invariant failed: FP32x8 FMA is outside the K-panel loop"
        )

    call = calls[0]
    if len(call.args) != 4 or str(call.args[1].dtype) != "float32":
        raise ValueError(
            f"Sunway {phase} GEMM invariant failed: invalid FP32x8 FMA signature"
        )
    analyzer = arith.Analyzer()
    for pointer in call.args[2:]:
        if (
            not isinstance(pointer, tirx.Call)
            or str(getattr(pointer.op, "name", "")) != "tirx.tvm_access_ptr"
            or len(pointer.args) != 5
            or not isinstance(pointer.args[3], tirx.IntImm)
            or int(pointer.args[3]) != 8
        ):
            raise ValueError(
                f"Sunway {phase} GEMM invariant failed: invalid FP32x8 access pointer"
            )
        remainder = analyzer.simplify(tirx.FloorMod(pointer.args[2], 8))
        if not isinstance(remainder, tirx.IntImm) or int(remainder) != 0:
            raise ValueError(
                f"Sunway {phase} GEMM invariant failed: FP32x8 pointers must be 32-byte aligned"
            )


def verify_gemm_semantic_func(
    func: tirx.PrimFunc,
    config: SunwayTargetConfig,
) -> tirx.PrimFunc:
    if str(func.attrs.get("sunway.phase", "")) != "S2":
        raise ValueError("Sunway S2 GEMM verifier received a function from another phase")
    compute = str(func.attrs.get("sunway.compute", ""))
    expected_kernel_kind = "gemm_simd" if compute == "simd" else "gemm_scalar"
    if str(func.attrs.get("sunway.kernel_kind", "")) != expected_kernel_kind:
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
    if ownership not in {"single", "mesh_2d"}:
        raise ValueError(
            f"Sunway S2 GEMM invariant failed: unsupported ownership {ownership!r}"
        )
    if compute not in {"scalar", "simd"}:
        raise ValueError(f"Sunway S2 GEMM invariant failed: unsupported compute mode {compute!r}")
    if compute == "simd" and metadata["sunway.vector_width"] != 8:
        raise ValueError(
            "Sunway S2 GEMM invariant failed: native FP32 compute requires vector width 8"
        )
    expected_workers = config.cpe_rows * config.cpe_cols
    if metadata["sunway.cpe_count"] != expected_workers:
        raise ValueError(
            "Sunway S2 GEMM invariant failed: CPE count does not match the target mesh"
        )
    if (
        metadata["sunway.cpe_rows"] != config.cpe_rows
        or metadata["sunway.cpe_cols"] != config.cpe_cols
    ):
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

    names = named_calls(func.body)
    counts = Counter(names)
    native = sorted(set(names) & _NATIVE_CALLS)
    if native:
        raise ValueError(
            f"Sunway S2 GEMM invariant failed: native ABI call {native[0]!r} appeared before S3"
        )
    tile_ops = sorted(name for name in set(names) if name.startswith("tl.tileop."))
    if tile_ops:
        raise ValueError(f"Sunway S2 GEMM invariant failed: residual TileOp {tile_ops[0]!r}")
    expected_calls = {
        "tilelang_sunway_pe_id": 1,
        "tilelang_sunway_dma_get": 2,
        "tilelang_sunway_dma_put": 1,
        "tilelang_sunway_dma_wait": 3,
        "tilelang_sunway_fma_f32x8": 1 if compute == "simd" else 0,
    }
    for name, expected in expected_calls.items():
        if counts[name] != expected:
            raise ValueError(
                f"Sunway S2 GEMM invariant failed: expected {expected} {name} call(s), "
                f"found {counts[name]}"
            )

    float_arithmetic: list[object] = []
    tirx.stmt_functor.post_order_visit(
        func.body,
        lambda node: float_arithmetic.append(node)
        if isinstance(node, (tirx.Add, tirx.Mul)) and str(node.dtype) == "float32"
        else None,
    )
    has_float_add = any(isinstance(node, tirx.Add) for node in float_arithmetic)
    has_float_multiply = any(isinstance(node, tirx.Mul) for node in float_arithmetic)
    if compute == "scalar" and not (has_float_add and has_float_multiply):
        raise ValueError("Sunway S2 GEMM invariant failed: scalar FP32 multiply/add is missing")
    if compute == "simd" and float_arithmetic:
        raise ValueError(
            "Sunway S2 GEMM invariant failed: SIMD compute retained scalar FP32 arithmetic"
        )
    if compute == "simd":
        _verify_f32x8_call(
            func,
            call_name_="tilelang_sunway_fma_f32x8",
            k_panels=metadata["sunway.k_panels"],
            phase="S2",
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
        raise ValueError(
            "Sunway S2 GEMM invariant failed: thread binding remains after ownership lowering"
        )
    bind_map = {bind.var.name: bind for bind in binds}
    if ownership == "single":
        block_loops = {
            loop.loop_var.name: loop
            for loop in loops
            if loop.loop_var.name in {"bx", "by"}
        }
        if set(block_loops) != {"bx", "by"} or any(
            loop.kind != tirx.ForKind.SERIAL for loop in block_loops.values()
        ):
            raise ValueError(
                "Sunway S2 GEMM invariant failed: block tiles must be static serial bx/by loops"
            )
        if (
            int(block_loops["by"].extent) != metadata["sunway.block_tiles_m"]
            or int(block_loops["bx"].extent) != metadata["sunway.block_tiles_n"]
        ):
            raise ValueError(
                "Sunway S2 GEMM invariant failed: block tile loop metadata is inconsistent"
            )
        if set(bind_map) != {"pe_id"}:
            raise ValueError(
                "Sunway S2 GEMM invariant failed: expected one semantic PE id binding"
            )

        owned_names = [name for owner in owners for name in named_calls(owner.then_case)]
        if (
            Counter(owned_names)["tilelang_sunway_dma_get"] != 2
            or Counter(owned_names)["tilelang_sunway_dma_put"] != 1
        ):
            raise ValueError(
                "Sunway S2 GEMM invariant failed: every DMA site must be owned by CPE zero"
            )
    else:
        if owners:
            raise ValueError(
                "Sunway S2 GEMM invariant failed: mesh ownership retained a CPE-zero guard"
            )
        if set(bind_map) != {"pe_id", "pe_row", "pe_col"}:
            raise ValueError(
                "Sunway S2 GEMM invariant failed: mesh coordinate bindings are incomplete"
            )
        if call_name(bind_map["pe_id"].value) != "tilelang_sunway_pe_id":
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
        if (needs_row_guard and not has_row_guard) or (
            needs_col_guard and not has_col_guard
        ):
            raise ValueError(
                "Sunway S2 GEMM invariant failed: mesh bounds guards are incomplete"
            )

    dma_row_loops = [
        loop for loop in loops if loop.loop_var.name.startswith("sunway_dma_row_")
    ]
    if len(dma_row_loops) != 3:
        raise ValueError("Sunway S2 GEMM invariant failed: expected three row-transfer loops")
    for loop in dma_row_loops:
        row_names = named_calls(loop.body)
        issues = [name for name in row_names if name in _DMA_CALLS]
        waits = [name for name in row_names if name == "tilelang_sunway_dma_wait"]
        if (
            len(issues) != 1
            or len(waits) != 1
            or row_names.index(issues[0]) > row_names.index(waits[0])
        ):
            raise ValueError(
                "Sunway S2 GEMM invariant failed: each DMA issue must be followed by one wait"
            )
        issue_calls: list[tirx.Call] = []

        def collect_issue(node: object, calls: list[tirx.Call] = issue_calls) -> None:
            if isinstance(node, tirx.Call) and call_name(node) in _DMA_CALLS:
                calls.append(node)

        tirx.stmt_functor.post_order_visit(loop.body, collect_issue)
        row_bytes = static_int(issue_calls[0].args[3], "DMA row byte count")
        if row_bytes % config.dma_alignment:
            raise ValueError("Sunway S2 GEMM invariant failed: DMA row bytes are not aligned")
    return func


def verify_gemm_semantic_tir(mod: IRModule, config: SunwayTargetConfig) -> IRModule:
    """Verify that semantic S2 completely owns the GEMM schedule."""

    return map_prim_funcs(mod, lambda func: verify_gemm_semantic_func(func, config))


def _is_static_compact(buffer: tirx.Buffer) -> bool:
    if not buffer.shape or any(
        not isinstance(extent, tirx.IntImm) or int(extent) <= 0
        for extent in buffer.shape
    ):
        return False
    if not buffer.strides:
        return True
    if len(buffer.strides) != len(buffer.shape) or any(
        not isinstance(stride, tirx.IntImm) for stride in buffer.strides
    ):
        return False
    expected = 1
    for extent, stride in zip(reversed(buffer.shape), reversed(buffer.strides), strict=True):
        if int(stride) != expected:
            return False
        expected *= int(extent)
    return True


def verify_gemm_native_func(
    func: tirx.PrimFunc,
    config: SunwayTargetConfig,
) -> tirx.PrimFunc:
    if str(func.attrs.get("sunway.phase", "")) != "S3":
        raise ValueError("Sunway S3 GEMM verifier received a function from another phase")
    compute = str(func.attrs.get("sunway.compute", ""))
    expected_kernel_kind = "gemm_simd" if compute == "simd" else "gemm_scalar"
    if str(func.attrs.get("sunway.kernel_kind", "")) != expected_kernel_kind:
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
    if compute == "simd" and metadata["sunway.vector_width"] != 8:
        raise ValueError(
            "Sunway S3 GEMM invariant failed: native FP32 compute requires vector width 8"
        )
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

    names = named_calls(func.body)
    counts = Counter(names)
    semantic = sorted(set(names) & _SEMANTIC_CALLS)
    if semantic:
        raise ValueError(
            f"Sunway S3 GEMM invariant failed: semantic call {semantic[0]!r} remains"
        )
    tile_ops = sorted(name for name in set(names) if name.startswith("tl.tileop."))
    if tile_ops:
        raise ValueError(f"Sunway S3 GEMM invariant failed: residual TileOp {tile_ops[0]!r}")
    expected_calls = {
        "_MYID": 1,
        "athread_get": 2,
        "athread_put": 1,
        "tilelang_sunway_reply_wait": 3,
        "tilelang_sunway_native_fma_f32x8": 1 if compute == "simd" else 0,
    }
    for name, expected in expected_calls.items():
        if counts[name] != expected:
            raise ValueError(
                f"Sunway S3 GEMM invariant failed: expected {expected} {name} call(s), "
                f"found {counts[name]}"
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
        raise ValueError(
            "Sunway S3 GEMM invariant failed: expected statically shaped compact buffers"
        )

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
    if compute == "scalar" and not (
        float_multiply and float_add and has_load and has_store
    ):
        raise ValueError("Sunway S3 GEMM invariant failed: scalar FP32 multiply/add is missing")
    if compute == "simd" and (float_multiply or float_add):
        raise ValueError(
            "Sunway S3 GEMM invariant failed: SIMD compute retained scalar FP32 arithmetic"
        )
    if compute == "simd":
        _verify_f32x8_call(
            func,
            call_name_="tilelang_sunway_native_fma_f32x8",
            k_panels=metadata["sunway.k_panels"],
            phase="S3",
        )

    dimension_names = {"bx", "by"} if ownership == "single" else {"bx_round", "by_round"}
    block_loops = {
        loop.loop_var.name: loop
        for loop in loops
        if loop.loop_var.name in dimension_names
    }
    if set(block_loops) != dimension_names or any(
        loop.kind != tirx.ForKind.SERIAL for loop in block_loops.values()
    ):
        raise ValueError(
            "Sunway S3 GEMM invariant failed: ownership loops must remain static and serial"
        )
    dma_row_loops = [
        loop for loop in loops if loop.loop_var.name.startswith("sunway_dma_row_")
    ]
    if len(dma_row_loops) != 3:
        raise ValueError(
            "Sunway S3 GEMM invariant failed: expected three native row-transfer loops"
        )
    for loop in dma_row_loops:
        row_names = named_calls(loop.body)
        issues = [name for name in row_names if name in {"athread_get", "athread_put"}]
        waits = [name for name in row_names if name == "tilelang_sunway_reply_wait"]
        if (
            len(issues) != 1
            or len(waits) != 1
            or row_names.index(issues[0]) > row_names.index(waits[0])
        ):
            raise ValueError(
                "Sunway S3 GEMM invariant failed: native DMA issue/wait ordering is invalid"
            )
    return func


def verify_gemm_native_tir(mod: IRModule, config: SunwayTargetConfig) -> IRModule:
    """Verify codegen-ready GEMM TIR after resolving target ABI leaves."""

    return map_prim_funcs(mod, lambda func: verify_gemm_native_func(func, config))
