from __future__ import annotations

import pytest
import tilelang  # noqa: F401 - initializes TileLang's bundled TVM path
import tilelang.language as T
import tvm
from tvm import IRModule, tirx

from tilelang.sunway.analysis import analyze_copy
from tilelang.sunway.target import SunwayTargetConfig
from tilelang.sunway.transform import (
    annotate_sunway_tir,
    lower_semantic_to_native_tir,
    lower_tile_copy_to_semantic_tir,
    verify_native_tir,
    verify_semantic_tir,
)


def _buffer(elements: int, dtype: str = "float32", name: str = "buffer") -> tirx.Buffer:
    return tirx.decl_buffer((elements,), dtype, name=name)


def _only_prim_func(mod: IRModule) -> tirx.PrimFunc:
    functions = [func for func in mod.functions.values() if isinstance(func, tirx.PrimFunc)]
    assert len(functions) == 1
    return functions[0]


def _extern_calls(func: tirx.PrimFunc) -> list[tirx.Call]:
    calls: list[tirx.Call] = []

    def visit(node: object) -> None:
        if (
            isinstance(node, tirx.Call)
            and getattr(node.op, "name", None) == "tirx.call_extern"
        ):
            calls.append(node)

    tirx.stmt_functor.post_order_visit(func.body, visit)
    return calls


def _extern_name(call: tirx.Call) -> str:
    assert isinstance(call.args[0], tirx.StringImm)
    return call.args[0].value


def _replace_extern(mod: IRModule, old_name: str, new_name: str) -> IRModule:
    func = _only_prim_func(mod)

    def rewrite(node: object) -> object | None:
        if isinstance(node, tirx.Call) and _extern_name_or_none(node) == old_name:
            return tirx.call_extern(node.dtype, new_name, *node.args[1:])
        return None

    body = tirx.stmt_functor.ir_transform(func.body, None, rewrite)
    return IRModule({"main": func.with_body(body)})


def _extern_name_or_none(call: tirx.Call) -> str | None:
    if (
        getattr(call.op, "name", None) != "tirx.call_extern"
        or not call.args
        or not isinstance(call.args[0], tirx.StringImm)
    ):
        return None
    return call.args[0].value


@T.prim_func
def copy_4096(A: T.Tensor((4096,), "float32"), B: T.Tensor((4096,), "float32")):
    T.copy(A, B)


@T.prim_func
def copy_260(A: T.Tensor((260,), "float32"), B: T.Tensor((260,), "float32")):
    T.copy(A, B)


def test_copy_plan_preserves_the_canonical_copy_128_schedule() -> None:
    plan = analyze_copy(
        _buffer(128, name="A"),
        _buffer(128, name="B"),
        SunwayTargetConfig(),
        argument_count=2,
    )

    assert plan.tile_elements == 4
    assert plan.tile_count == 32
    assert plan.active_cpes == 32
    assert plan.iterations_per_cpe == 1
    assert plan.final_tile_elements == 4


def test_copy_plan_uses_grid_stride_tiles_when_ldm_caps_the_tile() -> None:
    plan = analyze_copy(
        _buffer(4096, name="A"),
        _buffer(4096, name="B"),
        SunwayTargetConfig(ldm_bytes_per_cpe=256),
        argument_count=2,
    )

    assert plan.tile_elements == 56
    assert plan.tile_count == 74
    assert plan.active_cpes == 64
    assert plan.iterations_per_cpe == 2
    assert plan.final_tile_elements == 8
    assert plan.ldm_bytes <= 256


def test_copy_plan_accepts_an_aligned_short_final_tile() -> None:
    plan = analyze_copy(
        _buffer(260, name="A"),
        _buffer(260, name="B"),
        SunwayTargetConfig(),
        argument_count=2,
    )

    assert plan.tile_elements == 8
    assert plan.tile_count == 33
    assert plan.final_tile_elements == 4


def test_copy_plan_rejects_a_total_size_that_is_not_dma_aligned() -> None:
    with pytest.raises(ValueError, match="total byte count.*16-byte DMA alignment"):
        analyze_copy(
            _buffer(130, name="A"),
            _buffer(130, name="B"),
            SunwayTargetConfig(),
            argument_count=2,
        )


def test_copy_plan_rejects_an_ldm_budget_smaller_than_fixed_overhead() -> None:
    with pytest.raises(ValueError, match="fixed LDM overhead"):
        analyze_copy(
            _buffer(128, name="A"),
            _buffer(128, name="B"),
            SunwayTargetConfig(ldm_bytes_per_cpe=16),
            argument_count=2,
        )


def test_s2_materializes_grid_stride_ownership_for_large_copy() -> None:
    config = SunwayTargetConfig(ldm_bytes_per_cpe=256)
    s1 = annotate_sunway_tir(IRModule({"copy_4096": copy_4096}))
    s2 = lower_tile_copy_to_semantic_tir(s1, config)
    func = _only_prim_func(s2)

    loops: list[tirx.For] = []
    tirx.stmt_functor.post_order_visit(
        func.body,
        lambda node: loops.append(node) if isinstance(node, tirx.For) else None,
    )
    calls = _extern_calls(func)
    names = {_extern_name(call) for call in calls}
    dma_get = next(call for call in calls if _extern_name(call) == "tilelang_sunway_dma_get")

    assert len(loops) == 1
    assert int(loops[0].extent) == 2
    assert int(func.attrs["sunway.tile_count"]) == 74
    assert int(func.attrs["sunway.iterations_per_cpe"]) == 2
    assert int(func.attrs["sunway.ldm_bytes"]) == 244
    assert "tilelang_sunway_dma_get" in names
    assert "tilelang_sunway_dma_put" in names
    assert "athread_get" not in names
    assert not isinstance(dma_get.args[3], tirx.IntImm)


def test_s2_represents_an_aligned_short_final_tile() -> None:
    s1 = annotate_sunway_tir(IRModule({"copy_260": copy_260}))
    s2 = lower_tile_copy_to_semantic_tir(s1, SunwayTargetConfig())
    func = _only_prim_func(s2)

    assert int(func.attrs["sunway.tile_elements"]) == 8
    assert int(func.attrs["sunway.tile_count"]) == 33
    assert int(func.attrs["sunway.final_tile_elements"]) == 4


def test_s2_verifier_accepts_a_legal_semantic_copy() -> None:
    config = SunwayTargetConfig(ldm_bytes_per_cpe=256)
    s1 = annotate_sunway_tir(IRModule({"copy_4096": copy_4096}))
    s2 = lower_tile_copy_to_semantic_tir(s1, config)

    tvm.ir.assert_structural_equal(verify_semantic_tir(s2, config), s2)


def test_s2_verifier_rejects_a_native_abi_call() -> None:
    config = SunwayTargetConfig()
    s1 = annotate_sunway_tir(IRModule({"copy_260": copy_260}))
    s2 = lower_tile_copy_to_semantic_tir(s1, config)
    invalid = _replace_extern(s2, "tilelang_sunway_dma_get", "athread_get")

    with pytest.raises(ValueError, match="S2.*native ABI call.*athread_get"):
        verify_semantic_tir(invalid, config)


def test_s2_verifier_rejects_broken_ownership_metadata() -> None:
    config = SunwayTargetConfig()
    s1 = annotate_sunway_tir(IRModule({"copy_260": copy_260}))
    s2 = lower_tile_copy_to_semantic_tir(s1, config)
    func = _only_prim_func(s2).with_attr("sunway.tile_count", 0)

    with pytest.raises(ValueError, match="S2.*sunway.tile_count.*positive"):
        verify_semantic_tir(IRModule({"main": func}), config)


def test_s2_verifier_rejects_ldm_over_budget() -> None:
    config = SunwayTargetConfig(ldm_bytes_per_cpe=256)
    s1 = annotate_sunway_tir(IRModule({"copy_4096": copy_4096}))
    s2 = lower_tile_copy_to_semantic_tir(s1, config)
    func = _only_prim_func(s2).with_attr("sunway.ldm_bytes", 257)

    with pytest.raises(ValueError, match="S2.*LDM plan uses 257 bytes.*limit is 256"):
        verify_semantic_tir(IRModule({"main": func}), config)


def test_s3_verifier_rejects_a_remaining_semantic_call() -> None:
    config = SunwayTargetConfig()
    s1 = annotate_sunway_tir(IRModule({"copy_260": copy_260}))
    s2 = lower_tile_copy_to_semantic_tir(s1, config)
    func = _only_prim_func(s2).with_attr("sunway.phase", "S3")

    with pytest.raises(ValueError, match="S3.*semantic call.*tilelang_sunway_dma_get"):
        verify_native_tir(IRModule({"main": func}), config)


def test_s3_lowering_resolves_all_semantic_calls() -> None:
    config = SunwayTargetConfig()
    s1 = annotate_sunway_tir(IRModule({"copy_260": copy_260}))
    s2 = lower_tile_copy_to_semantic_tir(s1, config)

    s3 = lower_semantic_to_native_tir(verify_semantic_tir(s2, config), config)
    verify_native_tir(s3, config)
    names = {_extern_name(call) for call in _extern_calls(_only_prim_func(s3))}

    assert "athread_get" in names
    assert "athread_put" in names
    assert "tilelang_sunway_dma_get" not in names
    assert "tilelang_sunway_dma_put" not in names
