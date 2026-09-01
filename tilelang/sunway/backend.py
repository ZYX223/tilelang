"""Sunway backend manifest."""

from tilelang.backend.device_codegen import DeviceCodegen
from tilelang.backend.execution_backend import ExecutionBackendSpec
from tilelang.backend.module import BackendModule, register_backend
from tilelang.backend.pass_pipeline import PassPipeline

from . import codegen, pipeline
from .target import is_sunway_target


BACKEND = register_backend(
    BackendModule(
        name="sunway",
        target_kinds=("c",),
        supports_target=is_sunway_target,
        pipelines={"c": PassPipeline("c", pipeline.SunwayPassPipelineBody)},
        device_codegens={"c": DeviceCodegen("sunway", build_without_compile=codegen.build_without_compile)},
        execution_backends=(ExecutionBackendSpec("sunway_aot"),),
    )
)
