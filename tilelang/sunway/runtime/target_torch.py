"""Self-contained PyTorch wrapper copied into generated SW9A packages."""

# Keep this module importable by the Python 3.6 runtime installed on SW9A.

import os

from tilelang_sunway_adapter import SunwayKernelAdapter


class SunwayTorchOperator:
    """Allocate manifest outputs and dispatch the registered torch.ops kernel."""

    def __init__(self, package_dir, tensor_module=None):
        if tensor_module is None:
            import torch

            tensor_module = torch
        self.tensor_module = tensor_module
        self.adapter = SunwayKernelAdapter(package_dir)
        self.input_arguments = tuple(argument for argument in self.adapter.arguments if argument["role"] in ("input", "inout"))
        self.output_arguments = tuple(argument for argument in self.adapter.arguments if argument["role"] == "output")
        if len(self.output_arguments) != 1:
            raise ValueError("Registered Sunway PyTorch kernels require exactly one output")

        torch_library = self.adapter.manifest.get("artifacts", {}).get("torch_library")
        if not torch_library:
            raise ValueError("Sunway package does not contain a PyTorch registration library")
        torch_library_path = os.path.join(self.adapter.package_dir, torch_library)
        if not os.path.isfile(torch_library_path):
            raise OSError(f"Missing Sunway PyTorch registration library {torch_library_path}")
        self.tensor_module.ops.load_library(torch_library_path)
        namespace = self.tensor_module.ops.tilelang_sunway
        self.kernel = getattr(namespace, self.adapter.manifest["kernel_name"])

    def __call__(self, *inputs):
        if len(inputs) != len(self.input_arguments):
            raise ValueError(f"Sunway PyTorch operator expects {len(self.input_arguments)} inputs, got {len(inputs)}")
        device = inputs[0].device
        values_by_name = dict((argument["name"], tensor) for argument, tensor in zip(self.input_arguments, inputs))
        for argument in self.output_arguments:
            dtype = getattr(self.tensor_module, argument["dtype"])
            values_by_name[argument["name"]] = self.tensor_module.empty(tuple(argument["shape"]), dtype=dtype, device=device)
        ordered = [values_by_name[argument["name"]] for argument in self.adapter.arguments]
        return self.kernel(*ordered)


def load(package_dir, tensor_module=None):
    return SunwayTorchOperator(package_dir, tensor_module=tensor_module)
