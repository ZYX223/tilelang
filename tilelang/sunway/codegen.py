"""Structural MPE/CPE C emission from lowered Sunway TIR."""

from __future__ import annotations

import re
from dataclasses import dataclass

from tvm import IRModule, tirx
from tvm.target import Target

from .runtime.manifest import SunwayKernelArgument, SunwayKernelManifest
from .target import get_sunway_target_config


_DTYPE_TO_C = {
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

_BINARY_OPERATORS = {
    tirx.Add: "+",
    tirx.Sub: "-",
    tirx.Mul: "*",
    tirx.Div: "/",
    tirx.FloorDiv: "/",
    tirx.Mod: "%",
    tirx.FloorMod: "%",
    tirx.LT: "<",
    tirx.LE: "<=",
    tirx.GT: ">",
    tirx.GE: ">=",
    tirx.EQ: "==",
    tirx.NE: "!=",
    tirx.And: "&&",
    tirx.Or: "||",
}

_BINARY_INTRINSICS = {
    "tirx.bitwise_and": "&",
    "tirx.shift_right": ">>",
}


def _c_identifier(value: object) -> str:
    identifier = re.sub(r"[^A-Za-z0-9_]", "_", str(value))
    if not identifier or identifier[0].isdigit():
        identifier = f"_{identifier}"
    return identifier


def _c_type(dtype: object) -> str:
    name = str(dtype)
    try:
        return _DTYPE_TO_C[name]
    except KeyError as error:
        raise ValueError(f"Sunway Paper-1 C codegen does not support dtype {name}") from error


def _static_extent(buffer: tirx.Buffer) -> int:
    extent = 1
    for dim in buffer.shape:
        if not isinstance(dim, tirx.IntImm):
            raise ValueError(f"Sunway C codegen requires static local buffer {buffer.name}")
        extent *= int(dim)
    return extent


def _extern_name(call: tirx.Call) -> str | None:
    if getattr(call.op, "name", None) != "tirx.call_extern":
        return None
    if not call.args or not isinstance(call.args[0], tirx.StringImm):
        return None
    return call.args[0].value


@dataclass(frozen=True, slots=True)
class _Kernel:
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


def _extract_kernel(mod: IRModule) -> _Kernel:
    prim_funcs = [func for func in mod.functions.values() if isinstance(func, tirx.PrimFunc)]
    if len(prim_funcs) != 1:
        raise ValueError(f"Sunway Paper-1 codegen currently requires one PrimFunc per AOT project; found {len(prim_funcs)}")
    func = prim_funcs[0]
    if str(func.attrs.get("sunway.phase", "")) != "S3":
        raise ValueError("Sunway C codegen requires S3 lowered TIR")

    name = _c_identifier(func.attrs.get("global_symbol", "sunway_kernel"))
    parameters = tuple(func.buffer_map[param] for param in func.params if param in func.buffer_map)
    locals_: list[tirx.Buffer] = []

    def visit(node: object) -> None:
        if isinstance(node, tirx.AllocBuffer):
            locals_.append(node.buffer)

    tirx.stmt_functor.post_order_visit(func.body, visit)
    return _Kernel(name=name, func=func, parameters=parameters, locals=tuple(locals_))


class _CPEEmitter:
    """Emit statement and expression nodes without recovering code from text dumps."""

    def __init__(self, kernel: _Kernel):
        self.kernel = kernel
        self.lines: list[str] = []
        self.indent = 0
        self.buffer_bases: dict[tirx.Var, str] = {buffer.data: f"ldm_args.{_c_identifier(buffer.name)}" for buffer in kernel.parameters}
        for buffer in kernel.locals:
            self.buffer_bases[buffer.data] = _c_identifier(buffer.name)
        self.bound_values: dict[tirx.Var, tirx.PrimExpr] = {}

        def collect_bind(node: object) -> None:
            if isinstance(node, tirx.Bind):
                self.bound_values[node.var] = node.value

        tirx.stmt_functor.post_order_visit(kernel.func.body, collect_bind)

    def line(self, text: str = "") -> None:
        self.lines.append(f"{'    ' * self.indent}{text}" if text else "")

    def emit(self) -> str:
        header = f"{self.kernel.name}_common.h"
        self.line('#include "slave.h"')
        self.line(f'#include "{header}"')
        self.line()
        self.line(f"__thread_local {self.kernel.args_type} ldm_args;")
        for buffer in self.kernel.locals:
            c_name = _c_identifier(buffer.name)
            if c_name == "reply" and _c_type(buffer.dtype) == "int" and _static_extent(buffer) == 1:
                self.line("volatile __thread_local int reply;")
            else:
                self.line(f"__thread_local {_c_type(buffer.dtype)} {c_name}[{_static_extent(buffer)}];")
        self.line()
        self.line(f"void {self.kernel.cpe_entry}({self.kernel.args_type} *global_args) {{")
        self.indent += 1
        # Every CPE first copies the small launch descriptor into its own LDM.
        # This is the backend ABI prologue; kernel statements start after it.
        self.line("reply = 0;")
        self.line(f"athread_get(PE_MODE, global_args, &ldm_args, sizeof({self.kernel.args_type}), (void *)&reply, 0, 0, 0);")
        self.line("while (reply != 1) {")
        self.line("}")
        self.line()
        self.emit_stmt(self.kernel.func.body)
        self.indent -= 1
        self.line("}")
        return "\n".join(self.lines) + "\n"

    def emit_stmt(self, stmt: object) -> None:
        if isinstance(stmt, tirx.SeqStmt):
            for child in stmt.seq:
                self.emit_stmt(child)
            return
        if isinstance(stmt, tirx.AllocBuffer):
            return
        if isinstance(stmt, tirx.DeclBuffer):
            # LowerOpaqueBlock may retain a shaped alias over an allocated data
            # variable. The backing C array is emitted from AllocBuffer once.
            return
        if isinstance(stmt, tirx.AttrStmt):
            if str(stmt.attr_key) != "lexical_alloc_scope":
                raise TypeError(f"Sunway CPE codegen does not support attribute {stmt.attr_key!r}")
            self.emit_stmt(stmt.body)
            return
        if isinstance(stmt, tirx.Bind):
            return
        if isinstance(stmt, tirx.BufferStore):
            base = self._buffer_base(stmt.buffer)
            offset = self._emit_flat_index(stmt.buffer, list(stmt.indices))
            scalar_reply = (
                base == "reply"
                and len(stmt.indices) == 1
                and isinstance(stmt.indices[0], tirx.IntImm)
                and int(stmt.indices[0]) == 0
            )
            target = base if scalar_reply else f"{base}[{offset}]"
            self.line(f"{target} = {self.emit_expr(stmt.value)};")
            return
        if isinstance(stmt, tirx.For):
            if stmt.kind != tirx.ForKind.SERIAL:
                raise TypeError("Sunway CPE codegen currently supports serial TIR loops only")
            loop_var = _c_identifier(stmt.loop_var.name)
            start = self.emit_expr(stmt.min)
            extent = self.emit_expr(stmt.extent)
            stop = extent if start == "0" else f"({start} + {extent})"
            if stmt.step is None or (
                isinstance(stmt.step, tirx.IntImm) and int(stmt.step) == 1
            ):
                increment = f"++{loop_var}"
            else:
                increment = f"{loop_var} += {self.emit_expr(stmt.step)}"
            self.line(f"for (int {loop_var} = {start}; {loop_var} < {stop}; {increment}) {{")
            self.indent += 1
            self.emit_stmt(stmt.body)
            self.indent -= 1
            self.line("}")
            return
        if isinstance(stmt, tirx.IfThenElse):
            self.line(f"if ({self.emit_expr(stmt.condition)}) {{")
            self.indent += 1
            self.emit_stmt(stmt.then_case)
            self.indent -= 1
            if stmt.else_case is None:
                self.line("}")
            else:
                self.line("} else {")
                self.indent += 1
                self.emit_stmt(stmt.else_case)
                self.indent -= 1
                self.line("}")
            return
        if isinstance(stmt, tirx.Evaluate) and isinstance(stmt.value, tirx.Call):
            self.emit_call(stmt.value)
            return
        raise TypeError(f"Sunway CPE codegen does not support statement {type(stmt).__name__}")

    def emit_call(self, call: tirx.Call) -> None:
        name = _extern_name(call)
        args = list(call.args[1:])
        if name == "athread_get":
            source, destination, byte_count, reply = map(self.emit_expr, args)
            self.line(f"athread_get(PE_MODE, {source}, {destination}, {byte_count}, (void *){reply}, 0, 0, 0);")
            return
        if name == "athread_put":
            source, destination, byte_count, reply = map(self.emit_expr, args)
            self.line(f"athread_put(PE_MODE, {source}, {destination}, {byte_count}, (void *){reply}, 0, 0);")
            return
        if name == "tilelang_sunway_reply_wait":
            reply, count = map(self.emit_expr, args)
            value = "reply" if reply == "&reply" else f"*({reply})"
            self.line(f"while ({value} != {count}) {{")
            self.line("}")
            return
        raise ValueError(f"Sunway CPE codegen does not support external call {name!r}")

    def emit_expr(self, expr: object) -> str:
        if isinstance(expr, tirx.IntImm):
            return str(int(expr))
        if isinstance(expr, tirx.FloatImm):
            return repr(float(expr))
        if isinstance(expr, tirx.Var):
            for var, value in self.bound_values.items():
                if var.same_as(expr):
                    return self.emit_expr(value)
            return _c_identifier(expr.name)
        if isinstance(expr, tirx.BufferLoad):
            base = self._buffer_base(expr.buffer)
            offset = self._emit_flat_index(expr.buffer, list(expr.indices))
            return base if base == "reply" and offset == "0" else f"{base}[{offset}]"
        if isinstance(expr, tirx.Cast):
            return f"(({_c_type(expr.dtype)})({self.emit_expr(expr.value)}))"
        if isinstance(expr, tirx.Min):
            left = self.emit_expr(expr.a)
            right = self.emit_expr(expr.b)
            return f"(({left} < {right}) ? {left} : {right})"
        for node_type, operator in _BINARY_OPERATORS.items():
            if isinstance(expr, node_type):
                return f"({self.emit_expr(expr.a)} {operator} {self.emit_expr(expr.b)})"
        if isinstance(expr, tirx.Call):
            name = _extern_name(expr)
            if name == "_MYID":
                return "_MYID"
            op_name = getattr(expr.op, "name", None)
            if op_name in _BINARY_INTRINSICS and len(expr.args) == 2:
                operator = _BINARY_INTRINSICS[op_name]
                return f"({self.emit_expr(expr.args[0])} {operator} {self.emit_expr(expr.args[1])})"
            if op_name == "tirx.tvm_access_ptr":
                return self._emit_access_ptr(expr)
            if op_name == "tirx.address_of":
                return self._emit_address_of(expr)
        raise TypeError(f"Sunway CPE codegen does not support expression {type(expr).__name__}")

    def _emit_address_of(self, call: tirx.Call) -> str:
        load = call.args[0]
        if not isinstance(load, tirx.BufferLoad) or len(load.indices) != 1:
            raise TypeError("Sunway address_of currently requires a one-dimensional buffer load")
        base = self._buffer_base(load.buffer)
        offset = self.emit_expr(load.indices[0])
        if base == "reply" and offset == "0":
            return "&reply"
        return base if offset == "0" else f"({base} + {offset})"

    def _emit_flat_index(self, buffer: tirx.Buffer, indices: list[object]) -> str:
        if len(indices) != len(buffer.shape):
            raise TypeError("Sunway CPE codegen buffer rank mismatch")
        for dim in buffer.shape:
            if not isinstance(dim, tirx.IntImm):
                raise TypeError("Sunway CPE codegen requires static buffer shapes")
        if not indices:
            return "0"
        offset = f"({self.emit_expr(indices[0])})"
        for index, dim in zip(indices[1:], buffer.shape[1:], strict=True):
            offset = f"(({offset}) * {int(dim)} + ({self.emit_expr(index)}))"
        return offset

    def _emit_access_ptr(self, call: tirx.Call) -> str:
        data = call.args[1]
        if not isinstance(data, tirx.Var):
            raise TypeError("Sunway access_ptr base must be a buffer data variable")
        base = self._var_base(data)
        offset = self.emit_expr(call.args[2])
        if base == "reply":
            return "&reply"
        return base if offset == "0" else f"({base} + {offset})"

    def _var_base(self, var: tirx.Var) -> str:
        for data, base in self.buffer_bases.items():
            if data.same_as(var):
                return base
        raise ValueError(f"Sunway CPE codegen cannot resolve buffer variable {var.name}")

    def _buffer_base(self, buffer: tirx.Buffer) -> str:
        return self._var_base(buffer.data)


def _emit_common_header(kernel: _Kernel) -> str:
    guard = f"TILELANG_SUNWAY_{kernel.name.upper()}_COMMON_H_"
    fields = "\n".join(f"    {_c_type(buffer.dtype)} *{_c_identifier(buffer.name)};" for buffer in kernel.parameters)
    return f"#ifndef {guard}\n#define {guard}\n\ntypedef struct {kernel.name}_args {{\n{fields}\n}} {kernel.args_type};\n\n#endif\n"


def _emit_mpe(kernel: _Kernel) -> str:
    declarations = ", ".join(f"{_c_type(buffer.dtype)} *{_c_identifier(buffer.name)}" for buffer in kernel.parameters)
    assignments = "\n".join(f"    args.{_c_identifier(buffer.name)} = {_c_identifier(buffer.name)};" for buffer in kernel.parameters)
    return (
        '#include "athread.h"\n'
        f'#include "{kernel.name}_common.h"\n\n'
        f"extern SLAVE_FUN({kernel.cpe_entry})({kernel.args_type} *);\n\n"
        f"void {kernel.name}({declarations}) {{\n"
        f"    {kernel.args_type} args;\n"
        f"{assignments}\n\n"
        "    /* swrun or the hybrid Python launcher owns CRTS initialization. */\n"
        f"    athread_spawn({kernel.cpe_entry}, &args);\n"
        "    athread_join();\n"
        "}\n"
    )


def _emit_manifest(kernel: _Kernel, output_indices: tuple[int, ...]) -> SunwayKernelManifest:
    invalid = [index for index in output_indices if index < 0 or index >= len(kernel.parameters)]
    if invalid:
        raise ValueError(f"Sunway output parameter indices are out of range: {invalid}")
    outputs = frozenset(output_indices)
    arguments = tuple(
        SunwayKernelArgument(
            name=_c_identifier(buffer.name),
            dtype=str(buffer.dtype),
            shape=tuple(int(dim) for dim in buffer.shape),
            role="output" if index in outputs else "input",
        )
        for index, buffer in enumerate(kernel.parameters)
    )
    return SunwayKernelManifest(
        kernel_name=kernel.name,
        symbol=kernel.name,
        arguments=arguments,
        artifacts={
            "header": f"{kernel.name}_common.h",
            "mpe": f"mpe_{kernel.name}.c",
            "cpe": f"cpe_{kernel.name}.c",
        },
    )


def build_without_compile(mod: IRModule, target: Target):
    """Emit an AOT project and return its CPE source as a TVM source module."""

    kernel = _extract_kernel(mod)
    header = _emit_common_header(kernel)
    mpe = _emit_mpe(kernel)
    cpe = _CPEEmitter(kernel).emit()

    config = get_sunway_target_config(target)
    if config.output_dir is not None:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        for filename, content in {
            f"{kernel.name}_common.h": header,
            f"mpe_{kernel.name}.c": mpe,
            f"cpe_{kernel.name}.c": cpe,
        }.items():
            (config.output_dir / filename).write_text(content, encoding="utf-8")
        _emit_manifest(kernel, config.output_indices).write(config.output_dir / "manifest.json")

    import tvm.runtime._ffi_api as runtime_ffi

    return runtime_ffi.CSourceModuleCreate(cpe, "c", [kernel.cpe_entry], None)
