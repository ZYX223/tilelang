"""TIR utilities shared by Sunway GEMM scheduling and verification."""

from __future__ import annotations

from collections.abc import Callable

from tvm import IRModule, tirx


def map_prim_funcs(
    mod: IRModule,
    rewrite: Callable[[tirx.PrimFunc], tirx.PrimFunc],
) -> IRModule:
    functions = {
        global_var: rewrite(func) if isinstance(func, tirx.PrimFunc) else func
        for global_var, func in mod.functions.items()
    }
    return IRModule(functions, attrs=mod.attrs, global_infos=mod.global_infos)


def only_prim_func(mod: IRModule) -> tirx.PrimFunc:
    functions = [func for func in mod.functions.values() if isinstance(func, tirx.PrimFunc)]
    if len(functions) != 1:
        raise ValueError(f"Sunway G0 requires one PrimFunc per module, found {len(functions)}")
    return functions[0]


def call_name(call: tirx.Call) -> str:
    op_name = str(getattr(call.op, "name", ""))
    if op_name == "tirx.call_extern" and call.args and isinstance(call.args[0], tirx.StringImm):
        return call.args[0].value
    return op_name


def named_calls(node: object) -> list[str]:
    names: list[str] = []
    tirx.stmt_functor.post_order_visit(
        node,
        lambda child: names.append(call_name(child)) if isinstance(child, tirx.Call) else None,
    )
    return names


def static_int(value: tirx.PrimExpr, description: str) -> int:
    if not isinstance(value, tirx.IntImm):
        raise ValueError(f"Sunway G0 requires a static {description}, got {value}")
    return int(value)


def rewrite_for(
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
