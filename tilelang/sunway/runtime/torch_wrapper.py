"""PyTorch-facing allocation and legacy custom-operator registration helpers."""

from __future__ import annotations

import re
from pathlib import Path

from .adapter import SunwayKernelAdapter
from .manifest import SunwayKernelManifest


_CPP_DTYPES = {
    "float32": ("float", "at::kFloat"),
    "float64": ("double", "at::kDouble"),
    "int32": ("int", "at::kInt"),
    "int64": ("int64_t", "at::kLong"),
}


class SunwayTorchOperator:
    """Expose a manifest-driven kernel as a PyTorch-compatible callable."""

    def __init__(self, adapter: SunwayKernelAdapter, *, tensor_module=None, registered_kernel=None) -> None:
        if tensor_module is None:
            import torch

            tensor_module = torch
        self.adapter = adapter
        self.tensor_module = tensor_module
        self.registered_kernel = registered_kernel
        self.input_arguments = tuple(argument for argument in adapter.arguments if argument["role"] in {"input", "inout"})
        self.output_arguments = tuple(argument for argument in adapter.arguments if argument["role"] == "output")
        if not self.output_arguments:
            raise ValueError("Sunway PyTorch operator requires at least one output argument")

    @classmethod
    def from_registered_package(cls, package_dir, *, tensor_module=None):
        if tensor_module is None:
            import torch

            tensor_module = torch
        adapter = SunwayKernelAdapter(package_dir)
        torch_library = adapter.manifest.get("artifacts", {}).get("torch_library")
        if not torch_library:
            raise ValueError("Sunway kernel package does not contain a PyTorch registration library")
        torch_library_path = Path(adapter.package_dir) / torch_library
        if not torch_library_path.is_file():
            raise FileNotFoundError(f"Missing Sunway PyTorch registration library {torch_library_path}")
        tensor_module.ops.load_library(str(torch_library_path))
        namespace = tensor_module.ops.tilelang_sunway
        registered_kernel = getattr(namespace, adapter.manifest["kernel_name"])
        return cls(adapter, tensor_module=tensor_module, registered_kernel=registered_kernel)

    def __call__(self, *inputs):
        if len(inputs) != len(self.input_arguments):
            raise ValueError(f"Sunway PyTorch operator expects {len(self.input_arguments)} inputs, got {len(inputs)}")
        device = inputs[0].device
        values_by_name = {argument["name"]: tensor for argument, tensor in zip(self.input_arguments, inputs)}
        outputs = []
        for argument in self.output_arguments:
            dtype = getattr(self.tensor_module, argument["dtype"])
            tensor = self.tensor_module.empty(tuple(argument["shape"]), dtype=dtype, device=device)
            values_by_name[argument["name"]] = tensor
            outputs.append(tensor)

        ordered = [values_by_name[argument["name"]] for argument in self.adapter.arguments]
        if self.registered_kernel is not None:
            return self.registered_kernel(*ordered)
        self.adapter.invoke(*ordered)
        return outputs[0] if len(outputs) == 1 else tuple(outputs)


def _cpp_identifier(value: str) -> str:
    identifier = re.sub(r"[^A-Za-z0-9_]", "_", value)
    if not identifier or identifier[0].isdigit():
        identifier = f"_{identifier}"
    return identifier


def torch_extension_name(manifest: SunwayKernelManifest) -> str:
    return f"tilelang_sunway_{_cpp_identifier(manifest.kernel_name)}_ops.so"


def render_torch_registration_source(manifest: SunwayKernelManifest, *, namespace: str = "tilelang_sunway") -> str:
    """Generate a lightweight PyTorch 1.5 registration unit.

    Output tensors remain explicit arguments. The Python wrapper allocates
    them, which avoids pulling the full ATen factory/template surface into the
    older Sunway C++ compiler.
    """

    outputs = tuple(argument for argument in manifest.arguments if argument.role == "output")
    if len(outputs) != 1:
        raise ValueError("Legacy Sunway PyTorch registration currently requires exactly one output")
    for argument in manifest.arguments:
        if argument.dtype not in _CPP_DTYPES:
            raise ValueError(f"Unsupported PyTorch wrapper dtype {argument.dtype}")

    wrapper_name = f"tilelang_sunway_{_cpp_identifier(manifest.kernel_name)}"
    boxed_wrapper_name = f"{wrapper_name}_boxed"
    declaration_args = ", ".join(f"{_CPP_DTYPES[argument.dtype][0]}* {argument.name}" for argument in manifest.arguments)
    boxed_arguments = [
        f"    at::Tensor {argument.name} = torch::jit::peek(*stack, {index}, {len(manifest.arguments)}).toTensor();"
        for index, argument in enumerate(manifest.arguments)
    ]
    checks = []
    for argument in manifest.arguments:
        _, aten_dtype = _CPP_DTYPES[argument.dtype]
        checks.extend(
            [
                f'    TORCH_CHECK({argument.name}.unsafeGetTensorImpl()->device().is_cpu(), "{argument.name} must be an MPE CPU tensor");',
                f'    TORCH_CHECK({argument.name}.is_contiguous(), "{argument.name} must be contiguous");',
                f'    TORCH_CHECK({argument.name}.scalar_type() == {aten_dtype}, "{argument.name} dtype mismatch");',
                f'    TORCH_CHECK({argument.name}.dim() == {len(argument.shape)}, "{argument.name} rank mismatch");',
            ]
        )
        checks.extend(
            f'    TORCH_CHECK({argument.name}.size({index}) == {int(dim)}, "{argument.name} shape mismatch");'
            for index, dim in enumerate(argument.shape)
        )

    output = outputs[0]
    schema_arguments = ", ".join(f"Tensor {argument.name}" for argument in manifest.arguments)
    schema = f"{namespace}::{manifest.kernel_name}({schema_arguments}) -> Tensor"
    call_arguments = []
    for argument in manifest.arguments:
        c_type, _ = _CPP_DTYPES[argument.dtype]
        call_arguments.append(f"{argument.name}.data_ptr<{c_type}>()")

    return "\n".join(
        [
            "#include <ATen/core/TensorBody.h>",
            "#include <ATen/core/stack.h>",
            "#include <ATen/core/op_registration/op_registration.h>",
            "#include <cstdint>",
            "",
            f'extern "C" void {manifest.symbol}({declaration_args});',
            "",
            f"void {boxed_wrapper_name}(const c10::OperatorHandle&, c10::Stack* stack) {{",
            *boxed_arguments,
            *checks,
            f"    {manifest.symbol}({', '.join(call_arguments)});",
            f"    torch::jit::drop(*stack, {len(manifest.arguments)});",
            f"    torch::jit::push(*stack, {output.name});",
            "}",
            "",
            "static auto registry = c10::RegisterOperators()",
            "    .op(c10::RegisterOperators::options()",
            f'        .schema("{schema}")',
            f"        .catchAllKernel<&{boxed_wrapper_name}>());",
            "",
        ]
    )
