"""Extract the codegen-facing kernel and ABI model from lowered S3 TIR."""

from __future__ import annotations

import re
from dataclasses import dataclass

from tvm import IRModule, tirx


DTYPE_TO_C = {
    "float32": "float",
    "float64": "double",
    "int8": "signed char",
    "uint8": "unsigned char",
    "int16": "short",
    "uint16": "unsigned short",
    "int32": "int",
    "uint32": "unsigned int",
    "int64": "long",
    "uint64": "unsigned long",
}
NATIVE_FMA_F32X8 = "tilelang_sunway_native_fma_f32x8"


def c_identifier(value: object) -> str:
    identifier = re.sub(r"[^A-Za-z0-9_]", "_", str(value))
    if not identifier or identifier[0].isdigit():
        identifier = f"_{identifier}"
    return identifier


def c_type(dtype: object) -> str:
    name = str(dtype)
    try:
        return DTYPE_TO_C[name]
    except KeyError as error:
        raise ValueError(f"Sunway Paper-1 C codegen does not support dtype {name}") from error


def static_extent(buffer: tirx.Buffer) -> int:
    extent = 1
    for dim in buffer.shape:
        if not isinstance(dim, tirx.IntImm):
            raise ValueError(f"Sunway C codegen requires static local buffer {buffer.name}")
        extent *= int(dim)
    return extent


def extern_name(call: tirx.Call) -> str | None:
    if getattr(call.op, "name", None) != "tirx.call_extern":
        return None
    if not call.args or not isinstance(call.args[0], tirx.StringImm):
        return None
    return call.args[0].value


def contains_extern(node: object, name: str) -> bool:
    found = False

    def visit(candidate: object) -> None:
        nonlocal found
        if isinstance(candidate, tirx.Call) and extern_name(candidate) == name:
            found = True

    tirx.stmt_functor.post_order_visit(node, visit)
    return found


@dataclass(frozen=True, slots=True)
class Kernel:
    """The small, target-independent ABI view consumed by both C emitters."""

    name: str
    func: tirx.PrimFunc
    parameters: tuple[tirx.Buffer, ...]
    locals: tuple[tirx.Buffer, ...]

    @property
    def args_type(self) -> str:
        return f"{self.name}_args_t"

    @property
    def cpe_entry(self) -> str:
        return f"{self.name}_cpe"

    @property
    def uses_f32x8(self) -> bool:
        return contains_extern(self.func.body, NATIVE_FMA_F32X8)


def extract_kernel(mod: IRModule) -> Kernel:
    """Validate S3 and collect parameters and local buffers without parsing text."""

    prim_funcs = [func for func in mod.functions.values() if isinstance(func, tirx.PrimFunc)]
    if len(prim_funcs) != 1:
        raise ValueError(
            "Sunway Paper-1 codegen currently requires one PrimFunc per AOT project; "
            f"found {len(prim_funcs)}"
        )
    func = prim_funcs[0]
    if str(func.attrs.get("sunway.phase", "")) != "S3":
        raise ValueError("Sunway C codegen requires S3 lowered TIR")

    name = c_identifier(func.attrs.get("global_symbol", "sunway_kernel"))
    parameters = tuple(
        func.buffer_map[param] for param in func.params if param in func.buffer_map
    )
    locals_: list[tirx.Buffer] = []

    def visit(node: object) -> None:
        if isinstance(node, tirx.AllocBuffer):
            locals_.append(node.buffer)

    tirx.stmt_functor.post_order_visit(func.body, visit)
    return Kernel(name=name, func=func, parameters=parameters, locals=tuple(locals_))
