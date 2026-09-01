"""Sunway S1/S2/S3 pass pipeline registration."""

from __future__ import annotations

from tvm import IRModule, tirx
from tvm.target import Target

from .target import get_sunway_target_config
from .transform import (
    annotate_sunway_tir,
    lower_semantic_to_native_tir,
    lower_tile_copy_to_semantic_tir,
)


def _dump_checkpoint(mod: IRModule, filename: str, target: Target) -> None:
    config = get_sunway_target_config(target)
    if config.output_dir is None:
        return
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / filename).write_text(mod.script(), encoding="utf-8")


def SunwayPassPipelineBody(mod: IRModule, target: Target) -> IRModule:
    """Run the inspectable progressive lowering used by the Sunway AOT path."""

    config = get_sunway_target_config(target)

    s1 = annotate_sunway_tir(tirx.transform.BindTarget(target)(mod))
    _dump_checkpoint(s1, "s1_annotated_tir.txt", target)

    s2 = lower_tile_copy_to_semantic_tir(s1, config)
    _dump_checkpoint(s2, "s2_semantic_tir.txt", target)

    s3 = lower_semantic_to_native_tir(s2)
    _dump_checkpoint(s3, "s3_lowered_tir.txt", target)
    return s3
