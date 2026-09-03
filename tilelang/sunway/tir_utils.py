"""Shared Sunway TIR traversal helpers and semantic leaf definitions."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable

from tvm import IRModule, tirx


SEMANTIC_TO_NATIVE = {
    "tilelang_sunway_pe_id": "_MYID",
    "tilelang_sunway_dma_get": "athread_get",
    "tilelang_sunway_dma_put": "athread_put",
    "tilelang_sunway_dma_wait": "tilelang_sunway_reply_wait",
    "tilelang_sunway_fma_f32x8": "tilelang_sunway_native_fma_f32x8",
}
SEMANTIC_CALLS = frozenset(SEMANTIC_TO_NATIVE)
NATIVE_CALLS = frozenset(SEMANTIC_TO_NATIVE.values())


def map_prim_funcs(
    mod: IRModule,
    rewrite: Callable[[tirx.PrimFunc], tirx.PrimFunc],
) -> IRModule:
    """Apply a backend rewrite while preserving module-level metadata."""

    functions = {
        global_var: rewrite(func) if isinstance(func, tirx.PrimFunc) else func
        for global_var, func in mod.functions.items()
    }
    return IRModule(functions, attrs=mod.attrs, global_infos=mod.global_infos)


def call_name(call: tirx.Call) -> str | None:
    op_name = getattr(call.op, "name", None)
    if isinstance(op_name, str) and op_name.startswith("tl.tileop."):
        return op_name
    if op_name == "tirx.call_extern" and call.args and isinstance(call.args[0], tirx.StringImm):
        return call.args[0].value
    return None


def named_calls(node: object) -> list[str]:
    names: list[str] = []

    def visit(candidate: object) -> None:
        if isinstance(candidate, tirx.Call):
            name = call_name(candidate)
            if name is not None:
                names.append(name)

    tirx.stmt_functor.post_order_visit(node, visit)
    return names


def positive_attr(func: tirx.PrimFunc, key: str, phase: str) -> int:
    value = func.attrs.get(key)
    if value is None:
        raise ValueError(f"Sunway {phase} invariant failed: required attribute {key} is missing")
    result = int(value)
    if result <= 0:
        raise ValueError(
            f"Sunway {phase} invariant failed: {key} must be positive, got {result}"
        )
    return result


def verify_call_counts(
    func: tirx.PrimFunc,
    expected: dict[str, int],
    phase: str,
) -> None:
    counts = Counter(named_calls(func.body))
    for name, count in expected.items():
        if counts[name] != count:
            raise ValueError(
                f"Sunway {phase} invariant failed: expected {count} {name} call(s), "
                f"found {counts[name]}"
            )


def _lower_semantic_call(node: object) -> object | None:
    if not isinstance(node, tirx.Call):
        return None
    name = call_name(node)
    native_name = SEMANTIC_TO_NATIVE.get(name)
    if native_name is None:
        return None
    return tirx.call_extern(node.dtype, native_name, *node.args[1:])


def lower_semantic_calls_to_native(mod: IRModule) -> IRModule:
    """Resolve target-independent Sunway leaves to the verified SW9A ABI."""

    def rewrite(func: tirx.PrimFunc) -> tirx.PrimFunc:
        body = tirx.stmt_functor.ir_transform(func.body, None, _lower_semantic_call)
        return func.with_body(body).with_attr("sunway.phase", "S3")

    return map_prim_funcs(mod, rewrite)
