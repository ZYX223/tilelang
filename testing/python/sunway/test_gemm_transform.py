from __future__ import annotations

from collections import Counter

import pytest
from tilelang import tvm
from tilelang.backend.module import create_backend_context
from tvm import IRModule, tirx

from testing.python.sunway.gemm_cases import (
    make_gemm_32,
    make_gemm_128_k64,
    make_gemm_m32_n16_k32,
)
from tilelang.sunway.op.gemm.plan import SunwayGemmPlan
from tilelang.sunway.gemm_transform import (
    lower_gemm_program_to_semantic_tir,
    lower_gemm_semantic_to_native_tir,
    verify_gemm_semantic_tir,
    verify_gemm_native_tir,
)
from tilelang.sunway.target import SunwayTargetConfig
from tilelang.sunway.transform import annotate_sunway_tir


def _only_prim_func(mod: IRModule) -> tirx.PrimFunc:
    functions = [func for func in mod.functions.values() if isinstance(func, tirx.PrimFunc)]
    assert len(functions) == 1
    return functions[0]


def _call_name(call: tirx.Call) -> str:
    op_name = str(getattr(call.op, "name", ""))
    if op_name == "tirx.call_extern" and call.args and isinstance(call.args[0], tirx.StringImm):
        return call.args[0].value
    return op_name


def _call_names(node: object) -> list[str]:
    names: list[str] = []

    def visit(candidate: object) -> None:
        if isinstance(candidate, tirx.Call):
            names.append(_call_name(candidate))

    tirx.stmt_functor.post_order_visit(node, visit)
    return names


def _is_worker_zero(condition: tirx.PrimExpr) -> bool:
    if not isinstance(condition, tirx.EQ):
        return False
    operands = (condition.a, condition.b)
    has_zero = any(isinstance(value, tirx.IntImm) and int(value) == 0 for value in operands)
    has_pe_id = any(isinstance(value, tirx.Var) and value.name == "pe_id" for value in operands)
    return has_zero and has_pe_id


def _lower(factory, config: SunwayTargetConfig | None = None) -> IRModule:
    config = config or SunwayTargetConfig()
    target = create_backend_context({"kind": "sunway"}).target
    mod = tvm.IRModule.from_expr(factory())
    with target:
        s1 = annotate_sunway_tir(tirx.transform.BindTarget(target)(mod))
        s2 = lower_gemm_program_to_semantic_tir(s1, config)
    return verify_gemm_semantic_tir(s2, config)


def _rewrite_body(mod: IRModule, rewrite) -> IRModule:
    func = _only_prim_func(mod)
    body = tirx.stmt_functor.ir_transform(func.body, None, rewrite)
    return IRModule({"main": func.with_body(body)})


def _replace_first_extern(mod: IRModule, old_name: str, new_name: str) -> IRModule:
    replaced = False

    def rewrite(node: object) -> object | None:
        nonlocal replaced
        if replaced or not isinstance(node, tirx.Call) or _call_name(node) != old_name:
            return None
        replaced = True
        return tirx.call_extern(str(node.dtype), new_name, *node.args[1:])

    result = _rewrite_body(mod, rewrite)
    assert replaced
    return result


def _lower_s3(factory=make_gemm_32, config: SunwayTargetConfig | None = None) -> IRModule:
    config = config or SunwayTargetConfig()
    return lower_gemm_semantic_to_native_tir(_lower(factory, config), config)


def test_g1_plan_records_mesh_multik_and_simd_schedule() -> None:
    config = SunwayTargetConfig(gemm_ownership="mesh_2d", gemm_compute="simd")

    plan = SunwayGemmPlan.from_prim_func(make_gemm_128_k64(), config)

    assert plan.block_tiles_m == 8
    assert plan.block_tiles_n == 8
    assert plan.k_panels == 2
    assert plan.global_m == 128
    assert plan.global_n == 128
    assert plan.global_k == 64
    assert plan.cpe_rows == 8
    assert plan.cpe_cols == 8
    assert plan.active_cpes == 64
    assert plan.ownership == "mesh_2d"
    assert plan.compute == "simd"
    assert plan.vector_width == 8


def test_g1_plan_rejects_an_unsupported_simd_width() -> None:
    config = SunwayTargetConfig(
        gemm_ownership="mesh_2d",
        gemm_compute="simd",
        simd_width=4,
    )

    with pytest.raises(ValueError, match="requires SIMD width 8"):
        SunwayGemmPlan.from_prim_func(make_gemm_128_k64(), config)


@pytest.mark.parametrize(
    ("factory", "block_tiles_m", "block_tiles_n"),
    [
        (make_gemm_32, 2, 2),
        (make_gemm_m32_n16_k32, 2, 1),
    ],
)
def test_s2_materializes_scalar_gemm_schedule(factory, block_tiles_m: int, block_tiles_n: int) -> None:
    func = _only_prim_func(_lower(factory))

    assert str(func.attrs["sunway.phase"]) == "S2"
    assert str(func.attrs["sunway.kernel_kind"]) == "gemm_scalar"
    assert int(func.attrs["sunway.cpe_count"]) == 64
    assert int(func.attrs["sunway.active_cpes"]) == 1
    assert int(func.attrs["sunway.block_tiles_m"]) == block_tiles_m
    assert int(func.attrs["sunway.block_tiles_n"]) == block_tiles_n
    assert int(func.attrs["sunway.tile_m"]) == 16
    assert int(func.attrs["sunway.tile_n"]) == 16
    assert int(func.attrs["sunway.tile_k"]) == 32
    assert int(func.attrs["sunway.pipeline_stages"]) == 1
    assert int(func.attrs["sunway.ldm_bytes"]) == (16 * 32 + 32 * 16 + 16 * 16) * 4 + 3 * 8 + 4
    assert isinstance(func.body, tirx.SeqStmt)
    assert isinstance(func.body.seq[0], tirx.AllocBuffer)
    assert isinstance(func.body.seq[1], tirx.Bind)

    loops: list[tirx.For] = []
    ownership: list[tirx.IfThenElse] = []
    binds: list[tirx.Bind] = []

    def visit(node: object) -> None:
        if isinstance(node, tirx.For):
            loops.append(node)
        elif isinstance(node, tirx.IfThenElse) and _is_worker_zero(node.condition):
            ownership.append(node)
        elif isinstance(node, tirx.Bind):
            binds.append(node)

    tirx.stmt_functor.post_order_visit(func.body, visit)
    names = _call_names(func.body)
    counts = Counter(names)

    block_loops = {loop.loop_var.name: loop for loop in loops if loop.loop_var.name in {"bx", "by"}}
    assert set(block_loops) == {"bx", "by"}
    assert all(loop.kind == tirx.ForKind.SERIAL for loop in block_loops.values())
    assert all(loop.thread_binding is None for loop in loops)
    assert counts["tilelang_sunway_pe_id"] == 1
    assert counts["tilelang_sunway_dma_get"] == 2
    assert counts["tilelang_sunway_dma_put"] == 1
    assert counts["tilelang_sunway_dma_wait"] == 3
    assert not any(name.startswith("athread_") for name in names)
    assert not any(name.startswith("tl.tileop.") for name in names)
    assert len(binds) == 1 and binds[0].var.name == "pe_id"

    owned_names = [name for owner in ownership for name in _call_names(owner.then_case)]
    assert Counter(owned_names)["tilelang_sunway_dma_get"] == 2
    assert Counter(owned_names)["tilelang_sunway_dma_put"] == 1


def test_g0_rejects_non_native_worker_count() -> None:
    with pytest.raises(ValueError, match="G0 requires 64 logical workers"):
        _lower(lambda: make_gemm_32(workers=128))


def test_g0_rejects_multiple_pipeline_stages() -> None:
    with pytest.raises(ValueError, match="G0 requires num_stages=1"):
        _lower(lambda: make_gemm_32(num_stages=2))


def test_g0_rejects_an_oversized_ldm_plan() -> None:
    with pytest.raises(ValueError, match="LDM plan uses .* target limit"):
        _lower(make_gemm_32, SunwayTargetConfig(ldm_bytes_per_cpe=2048))


def test_s2_verifier_rejects_a_missing_dma_wait() -> None:
    s2 = _lower(make_gemm_32)
    invalid = _replace_first_extern(s2, "tilelang_sunway_dma_wait", "removed_dma_wait")

    with pytest.raises(ValueError, match="expected 3 tilelang_sunway_dma_wait"):
        verify_gemm_semantic_tir(invalid, SunwayTargetConfig())


def test_s2_verifier_rejects_a_native_dma_call() -> None:
    s2 = _lower(make_gemm_32)
    invalid = _replace_first_extern(s2, "tilelang_sunway_dma_get", "athread_get")

    with pytest.raises(ValueError, match="native ABI call.*athread_get"):
        verify_gemm_semantic_tir(invalid, SunwayTargetConfig())


def test_s2_verifier_rejects_a_residual_gemm_tileop() -> None:
    s2 = _lower(make_gemm_32)
    replaced = False

    def rewrite(node: object) -> object | None:
        nonlocal replaced
        if replaced or not isinstance(node, tirx.Call) or _call_name(node) != "tilelang_sunway_dma_wait":
            return None
        replaced = True
        return tirx.Call(str(node.dtype), tirx.op.Op.get("tl.tileop.gemm"), list(node.args[1:]))

    invalid = _rewrite_body(s2, rewrite)
    assert replaced
    with pytest.raises(ValueError, match="residual TileOp.*tl.tileop.gemm"):
        verify_gemm_semantic_tir(invalid, SunwayTargetConfig())


def test_s2_verifier_rejects_a_non_serial_block_loop() -> None:
    s2 = _lower(make_gemm_32)
    replaced = False

    def rewrite(node: object) -> object | None:
        nonlocal replaced
        if replaced or not isinstance(node, tirx.For) or node.loop_var.name != "bx":
            return None
        replaced = True
        return tirx.For(
            node.loop_var,
            node.min,
            node.extent,
            tirx.ForKind.PARALLEL,
            node.body,
            annotations=node.annotations,
            step=node.step,
            span=getattr(node, "span", None),
        )

    invalid = _rewrite_body(s2, rewrite)
    assert replaced
    with pytest.raises(ValueError, match="block tiles must be static serial"):
        verify_gemm_semantic_tir(invalid, SunwayTargetConfig())


@pytest.mark.parametrize("factory", [make_gemm_32, make_gemm_m32_n16_k32])
def test_s3_resolves_native_leaves_and_preserves_static_buffers(factory) -> None:
    func = _only_prim_func(_lower_s3(factory))
    names = _call_names(func.body)
    counts = Counter(names)
    allocated: list[tirx.Buffer] = []
    tirx.stmt_functor.post_order_visit(
        func.body,
        lambda node: allocated.append(node.buffer) if isinstance(node, tirx.AllocBuffer) else None,
    )

    assert str(func.attrs["sunway.phase"]) == "S3"
    assert counts["tilelang_sunway_dma_get"] == 0
    assert counts["tilelang_sunway_dma_put"] == 0
    assert counts["tilelang_sunway_dma_wait"] == 0
    assert counts["athread_get"] == 2
    assert counts["athread_put"] == 1
    assert counts["tilelang_sunway_reply_wait"] == 3
    assert allocated
    assert all(all(isinstance(dim, tirx.IntImm) for dim in buffer.shape) for buffer in allocated)


def test_s3_verifier_rejects_a_residual_semantic_call() -> None:
    s3 = _lower_s3()
    invalid = _replace_first_extern(s3, "athread_get", "tilelang_sunway_dma_get")

    with pytest.raises(ValueError, match="semantic call.*tilelang_sunway_dma_get"):
        verify_gemm_native_tir(invalid, SunwayTargetConfig())


def test_s3_verifier_rejects_a_residual_thread_binding() -> None:
    s3 = _lower_s3()
    replaced = False

    def rewrite(node: object) -> object | None:
        nonlocal replaced
        if replaced or not isinstance(node, tirx.For) or node.loop_var.name != "bx":
            return None
        replaced = True
        thread_binding = tirx.IterVar(
            tvm.ir.Range(node.min, node.extent),
            node.loop_var,
            tirx.IterVar.ThreadIndex,
            "blockIdx.x",
        )
        return tirx.For(
            node.loop_var,
            node.min,
            node.extent,
            tirx.ForKind.THREAD_BINDING,
            node.body,
            thread_binding=thread_binding,
            annotations=node.annotations,
            step=node.step,
        )

    invalid = _rewrite_body(s3, rewrite)
    assert replaced
    with pytest.raises(ValueError, match="thread binding remains"):
        verify_gemm_native_tir(invalid, SunwayTargetConfig())


def test_s3_verifier_rejects_a_symbolic_allocated_buffer() -> None:
    s3 = _lower_s3()
    func = _only_prim_func(s3)
    symbolic = tirx.decl_buffer((tirx.Var("symbolic_extent", "int32"),), "float32", name="symboliclk")
    invalid = IRModule({"main": func.with_body(tirx.SeqStmt([tirx.AllocBuffer(symbolic), func.body]))})

    with pytest.raises(ValueError, match="statically shaped compact buffers"):
        verify_gemm_native_tir(invalid, SunwayTargetConfig())


def test_s3_verifier_rejects_missing_scalar_multiply() -> None:
    s3 = _lower_s3()

    def rewrite(node: object) -> object | None:
        if isinstance(node, tirx.Mul) and str(node.dtype) == "float32":
            return node.a
        return None

    invalid = _rewrite_body(s3, rewrite)
    with pytest.raises(ValueError, match="scalar FP32 multiply/add"):
        verify_gemm_native_tir(invalid, SunwayTargetConfig())


def test_s3_verifier_rejects_ldm_metadata_above_budget() -> None:
    s3 = _lower_s3()
    func = _only_prim_func(s3).with_attr("sunway.ldm_bytes", 64 * 1024 + 1)

    with pytest.raises(ValueError, match="LDM plan uses 65537 bytes.*limit is 65536"):
        verify_gemm_native_tir(IRModule({"main": func}), SunwayTargetConfig())
