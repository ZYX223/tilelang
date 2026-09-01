"""Pure-Python pointer ABI adapter that can be copied to a Python 3.6 target."""

# Keep this module free of TileLang, TVM, dataclasses, and modern annotation
# syntax. The generated operator package can copy it verbatim onto SW9A.

import ctypes
import json
import os


class SunwayKernelAdapter:
    """Load one generated library and validate tensor arguments before launch."""

    def __init__(self, package_dir, library_loader=None):
        self.package_dir = os.path.abspath(os.fspath(package_dir))
        manifest_path = os.path.join(self.package_dir, "manifest.json")
        with open(manifest_path) as manifest_file:
            self.manifest = json.load(manifest_file)

        if self.manifest.get("schema_version") != 1:
            raise ValueError("Unsupported Sunway kernel manifest schema")
        self.arguments = tuple(self.manifest["arguments"])
        library_name = self.manifest.get("artifacts", {}).get("library")
        if not library_name:
            raise ValueError("Sunway kernel package does not contain a shared library")

        loader = library_loader or ctypes.CDLL
        self.library_path = os.path.join(self.package_dir, library_name)
        # torch.ops registration library resolves the generated C ABI symbol
        # from this handle, so it must be visible in the process-wide scope.
        self.library = loader(self.library_path, mode=ctypes.RTLD_GLOBAL)
        self.function = getattr(self.library, self.manifest["symbol"])
        self.function.argtypes = [ctypes.c_void_p] * len(self.arguments)
        self.function.restype = None

    def invoke(self, *tensors):
        if len(tensors) != len(self.arguments):
            raise ValueError(
                "Sunway kernel {} expects {} tensor arguments, got {}".format(
                    self.manifest["kernel_name"], len(self.arguments), len(tensors)
                )
            )

        pointers = []
        for argument, tensor in zip(self.arguments, tensors):
            self._validate_tensor(argument, tensor)
            pointers.append(ctypes.c_void_p(int(tensor.data_ptr())))
        self.function(*pointers)

    @staticmethod
    def _validate_tensor(argument, tensor):
        expected_shape = tuple(int(dim) for dim in argument["shape"])
        actual_shape = tuple(int(dim) for dim in tensor.shape)
        if actual_shape != expected_shape:
            raise ValueError(
                "Sunway argument {} shape mismatch: expected {}, got {}".format(argument["name"], expected_shape, actual_shape)
            )

        actual_dtype = str(tensor.dtype)
        if actual_dtype.startswith("torch."):
            actual_dtype = actual_dtype[len("torch.") :]
        if actual_dtype != argument["dtype"]:
            raise ValueError(
                "Sunway argument {} dtype mismatch: expected {}, got {}".format(argument["name"], argument["dtype"], actual_dtype)
            )

        device = getattr(tensor, "device", "cpu")
        device_type = getattr(device, "type", str(device).split(":", 1)[0])
        if device_type != "cpu":
            raise ValueError("Sunway argument {} must use MPE CPU memory".format(argument["name"]))
        if not tensor.is_contiguous():
            raise ValueError("Sunway argument {} must be contiguous".format(argument["name"]))
