"""Mechanically emit CPE C statements and expressions from verified S3 TIR."""

from __future__ import annotations

from tvm import tirx

from .model import (
    NATIVE_FMA_F32X8,
    Kernel,
    c_identifier,
    c_type,
    extern_name,
    static_extent,
)


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


class CPEEmitter:
    """Emit statement and expression nodes without recovering code from TIR text."""

    def __init__(self, kernel: Kernel):
        self.kernel = kernel
        self.lines: list[str] = []
        self.indent = 0
        self.buffer_bases: dict[tirx.Var, str] = {
            buffer.data: f"ldm_args.{c_identifier(buffer.name)}"
            for buffer in kernel.parameters
        }
        for buffer in kernel.locals:
            self.buffer_bases[buffer.data] = c_identifier(buffer.name)
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
        if self.kernel.uses_f32x8:
            self.line('#include "simd.h"')
        self.line(f'#include "{header}"')
        self.line()
        if self.kernel.uses_f32x8:
            self._emit_f32x8_helper()
            self.line()
        self.line(f"__thread_local {self.kernel.args_type} ldm_args;")
        for buffer in self.kernel.locals:
            c_name = c_identifier(buffer.name)
            if c_name == "reply" and c_type(buffer.dtype) == "int" and static_extent(buffer) == 1:
                self.line("volatile __thread_local int reply;")
            else:
                alignment = " __attribute__((aligned(32)))" if self.kernel.uses_f32x8 else ""
                self.line(
                    f"__thread_local {c_type(buffer.dtype)} {c_name}[{static_extent(buffer)}]"
                    f"{alignment};"
                )
        self.line()
        self.line(f"void {self.kernel.cpe_entry}({self.kernel.args_type} *global_args) {{")
        self.indent += 1
        # Every CPE copies the small launch descriptor into its own LDM before
        # executing the already scheduled kernel body.
        self.line("reply = 0;")
        self.line(
            "athread_get(PE_MODE, global_args, &ldm_args, "
            f"sizeof({self.kernel.args_type}), (void *)&reply, 0, 0, 0);"
        )
        self.line("while (reply != 1) {")
        self.line("}")
        self.line()
        self.emit_stmt(self.kernel.func.body)
        self.indent -= 1
        self.line("}")
        return "\n".join(self.lines) + "\n"

    def _emit_f32x8_helper(self) -> None:
        """Emit the native leaf validated by the standalone SWGCC probe."""

        self.line(
            f"static inline void {NATIVE_FMA_F32X8}(float a, const float *b, float *c) {{"
        )
        self.indent += 1
        self.line("floatv8 a_vector = simd_set_floatv8(a, a, a, a, a, a, a, a);")
        self.line("floatv8 b_vector = *(const floatv8 *)b;")
        self.line("floatv8 c_vector = *(floatv8 *)c;")
        self.line("*(floatv8 *)c = simd_vmas(a_vector, b_vector, c_vector);")
        self.indent -= 1
        self.line("}")

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
                raise TypeError(
                    f"Sunway CPE codegen does not support attribute {stmt.attr_key!r}"
                )
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
            loop_var = c_identifier(stmt.loop_var.name)
            start = self.emit_expr(stmt.min)
            extent = self.emit_expr(stmt.extent)
            stop = extent if start == "0" else f"({start} + {extent})"
            if stmt.step is None or (
                isinstance(stmt.step, tirx.IntImm) and int(stmt.step) == 1
            ):
                increment = f"++{loop_var}"
            else:
                increment = f"{loop_var} += {self.emit_expr(stmt.step)}"
            self.line(
                f"for (int {loop_var} = {start}; {loop_var} < {stop}; {increment}) {{"
            )
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
        raise TypeError(
            f"Sunway CPE codegen does not support statement {type(stmt).__name__}"
        )

    def emit_call(self, call: tirx.Call) -> None:
        name = extern_name(call)
        args = list(call.args[1:])
        if name == "athread_get":
            source, destination, byte_count, reply = map(self.emit_expr, args)
            self.line(
                f"athread_get(PE_MODE, {source}, {destination}, {byte_count}, "
                f"(void *){reply}, 0, 0, 0);"
            )
            return
        if name == "athread_put":
            source, destination, byte_count, reply = map(self.emit_expr, args)
            self.line(
                f"athread_put(PE_MODE, {source}, {destination}, {byte_count}, "
                f"(void *){reply}, 0, 0);"
            )
            return
        if name == "tilelang_sunway_reply_wait":
            reply, count = map(self.emit_expr, args)
            value = "reply" if reply == "&reply" else f"*({reply})"
            self.line(f"while ({value} != {count}) {{")
            self.line("}")
            return
        if name == NATIVE_FMA_F32X8:
            scalar, b_pointer, c_pointer = map(self.emit_expr, args)
            self.line(f"{NATIVE_FMA_F32X8}({scalar}, {b_pointer}, {c_pointer});")
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
            return c_identifier(expr.name)
        if isinstance(expr, tirx.BufferLoad):
            base = self._buffer_base(expr.buffer)
            offset = self._emit_flat_index(expr.buffer, list(expr.indices))
            return base if base == "reply" and offset == "0" else f"{base}[{offset}]"
        if isinstance(expr, tirx.Cast):
            return f"(({c_type(expr.dtype)})({self.emit_expr(expr.value)}))"
        if isinstance(expr, tirx.Min):
            left = self.emit_expr(expr.a)
            right = self.emit_expr(expr.b)
            return f"(({left} < {right}) ? {left} : {right})"
        for node_type, operator in _BINARY_OPERATORS.items():
            if isinstance(expr, node_type):
                return f"({self.emit_expr(expr.a)} {operator} {self.emit_expr(expr.b)})"
        if isinstance(expr, tirx.Call):
            name = extern_name(expr)
            if name == "_MYID":
                return "_MYID"
            op_name = getattr(expr.op, "name", None)
            if op_name in _BINARY_INTRINSICS and len(expr.args) == 2:
                operator = _BINARY_INTRINSICS[op_name]
                return (
                    f"({self.emit_expr(expr.args[0])} {operator} "
                    f"{self.emit_expr(expr.args[1])})"
                )
            if op_name == "tirx.tvm_access_ptr":
                return self._emit_access_ptr(expr)
            if op_name == "tirx.address_of":
                return self._emit_address_of(expr)
        raise TypeError(
            f"Sunway CPE codegen does not support expression {type(expr).__name__}"
        )

    def _emit_address_of(self, call: tirx.Call) -> str:
        load = call.args[0]
        if not isinstance(load, tirx.BufferLoad) or len(load.indices) != 1:
            raise TypeError(
                "Sunway address_of currently requires a one-dimensional buffer load"
            )
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
