"""Package verified S3 TIR as a standalone Sunway MPE/CPE AOT project."""

from __future__ import annotations

from tvm import IRModule
from tvm.target import Target

from ..runtime.manifest import SunwayKernelArgument, SunwayKernelManifest
from ..target import get_sunway_target_config
from .cpe import CPEEmitter
from .model import Kernel, c_identifier, extract_kernel
from .mpe import emit_common_header, emit_mpe


def emit_manifest(
    kernel: Kernel,
    output_indices: tuple[int, ...],
) -> SunwayKernelManifest:
    invalid = [index for index in output_indices if index < 0 or index >= len(kernel.parameters)]
    if invalid:
        raise ValueError(f"Sunway output parameter indices are out of range: {invalid}")
    outputs = frozenset(output_indices)
    arguments = tuple(
        SunwayKernelArgument(
            name=c_identifier(buffer.name),
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

    kernel = extract_kernel(mod)
    header = emit_common_header(kernel)
    mpe = emit_mpe(kernel)
    cpe = CPEEmitter(kernel).emit()

    config = get_sunway_target_config(target)
    if config.output_dir is not None:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        for filename, content in {
            f"{kernel.name}_common.h": header,
            f"mpe_{kernel.name}.c": mpe,
            f"cpe_{kernel.name}.c": cpe,
        }.items():
            (config.output_dir / filename).write_text(content, encoding="utf-8")
        emit_manifest(kernel, config.output_indices).write(config.output_dir / "manifest.json")

    import tvm.runtime._ffi_api as runtime_ffi

    return runtime_ffi.CSourceModuleCreate(cpe, "c", [kernel.cpe_entry], None)
