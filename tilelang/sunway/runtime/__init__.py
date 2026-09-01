"""Build- and run-time helpers for generated Sunway kernels."""

from .adapter import SunwayKernelAdapter
from .executor import SunwaySlurmRelayExecutor, SunwaySubmittedJob
from .library import SunwayArtifactPackage, SunwayLibraryGenerator, SunwayToolchain, SunwayTorchSDK
from .manifest import SunwayKernelArgument, SunwayKernelManifest
from .torch_wrapper import SunwayTorchOperator, render_torch_registration_source

__all__ = [
    "SunwayArtifactPackage",
    "SunwayKernelAdapter",
    "SunwayKernelArgument",
    "SunwayKernelManifest",
    "SunwayLibraryGenerator",
    "SunwaySlurmRelayExecutor",
    "SunwaySubmittedJob",
    "SunwayTorchOperator",
    "SunwayToolchain",
    "SunwayTorchSDK",
    "render_torch_registration_source",
]
