from __future__ import annotations

import pytest
import tilelang
from tilelang.backend.module import create_backend_context
from tilelang import tvm
from tilelang.tileop.gemm.registry import resolve_gemm_impl
from tvm import tirx

from testing.python.sunway.gemm_cases import make_gemm_32, make_gemm_m32_n16_k32


def _call_name(call: tirx.Call) -> str:
    op_name = str(getattr(call.op, "name", ""))
    if op_name == "tirx.call_extern" and call.args and isinstance(call.args[0], tirx.StringImm):
        return call.args[0].value
    return op_name


def _lower_all_tileops(factory) -> tirx.PrimFunc:
    target = create_backend_context({"kind": "sunway"}).target
    mod = tvm.IRModule.from_expr(factory())
    with target:
        mod = tirx.transform.BindTarget(target)(mod)
        mod = tilelang.transform.LayoutInference()(mod)
        mod = tilelang.transform.LowerTileOp()(mod)
    return next(func for func in mod.functions.values() if isinstance(func, tirx.PrimFunc))


def test_sunway_scalar_gemm_is_registered() -> None:
    target = create_backend_context({"kind": "sunway"}).target

    implementation = resolve_gemm_impl("sunway.scalar", target)

    from tilelang.sunway.op.gemm.gemm_scalar import GemmScalar

    assert implementation is GemmScalar


@pytest.mark.parametrize("factory", [make_gemm_32, make_gemm_m32_n16_k32])
def test_canonical_gemm_lowers_all_tileops(factory) -> None:
    func = _lower_all_tileops(factory)
    names: list[str] = []
    nodes: list[object] = []

    def visit(node: object) -> None:
        nodes.append(node)
        if isinstance(node, tirx.Call):
            names.append(_call_name(node))

    tirx.stmt_functor.post_order_visit(func.body, visit)

    assert not any(name.startswith("tl.tileop.") for name in names)
    assert names.count("tilelang_sunway_dma_get_2d") == 2
    assert names.count("tilelang_sunway_dma_put_2d") == 1
    assert not any(name.startswith("athread_") for name in names)
    assert any(isinstance(node, tirx.Mul) for node in nodes)
    assert any(isinstance(node, tirx.BufferLoad) for node in nodes)
    assert any(isinstance(node, tirx.BufferStore) for node in nodes)
