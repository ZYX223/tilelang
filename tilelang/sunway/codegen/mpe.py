"""Emit the MPE launch wrapper and the shared MPE/CPE ABI header."""

from __future__ import annotations

from .model import Kernel, c_identifier, c_type


def emit_common_header(kernel: Kernel) -> str:
    guard = f"TILELANG_SUNWAY_{kernel.name.upper()}_COMMON_H_"
    fields = "\n".join(
        f"    {c_type(buffer.dtype)} *{c_identifier(buffer.name)};"
        for buffer in kernel.parameters
    )
    return (
        f"#ifndef {guard}\n#define {guard}\n\n"
        f"typedef struct {kernel.name}_args {{\n{fields}\n}} {kernel.args_type};\n\n#endif\n"
    )


def emit_mpe(kernel: Kernel) -> str:
    declarations = ", ".join(
        f"{c_type(buffer.dtype)} *{c_identifier(buffer.name)}"
        for buffer in kernel.parameters
    )
    assignments = "\n".join(
        f"    args.{c_identifier(buffer.name)} = {c_identifier(buffer.name)};"
        for buffer in kernel.parameters
    )
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
