from __future__ import annotations

import tilelang
import tilelang.language as T
from tilelang.backend.module import create_backend_context
from tilelang.sunway.target import get_sunway_target_config


def test_sunway_target_selects_dedicated_backend() -> None:
    context = create_backend_context({"kind": "sunway"}, execution_backend="sunway_aot")

    assert context.name == "sunway"
    assert context.target.kind.name == "c"
    assert "sunway" in context.target.keys
    assert context.execution_backend.name == "sunway_aot"


def test_sunway_target_configs_are_isolated(tmp_path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first = create_backend_context({"kind": "sunway", "output_dir": str(first_dir)})
    second = create_backend_context({"kind": "sunway", "output_dir": str(second_dir)})

    assert get_sunway_target_config(first.target).output_dir == first_dir
    assert get_sunway_target_config(second.target).output_dir == second_dir


def test_sunway_copy_lowers_to_aot_project(tmp_path) -> None:
    @T.prim_func
    def copy_128(A: T.Tensor((128,), "float32"), B: T.Tensor((128,), "float32")):
        T.copy(A, B)

    artifact = tilelang.lower(
        copy_128,
        target={"kind": "sunway", "output_dir": str(tmp_path)},
        runtime_only=True,
    )

    s1 = (tmp_path / "s1_annotated_tir.txt").read_text()
    s2 = (tmp_path / "s2_semantic_tir.txt").read_text()
    s3 = (tmp_path / "s3_lowered_tir.txt").read_text()
    header = (tmp_path / "copy_128_common.h").read_text()
    mpe = (tmp_path / "mpe_copy_128.c").read_text()
    cpe = (tmp_path / "cpe_copy_128.c").read_text()

    # TVM Script prints the registered ``tl.tileop.copy`` node as ``T.copy``.
    assert "T.copy(" in s1
    assert "tilelang_sunway_dma_get" in s2
    assert "tilelang_sunway_dma_put" in s2
    assert "athread_get" not in s2
    assert "athread_get" in s3
    assert "athread_put" in s3
    assert "copy_128_args_t" in header
    assert "athread_spawn" in mpe
    assert "athread_join" in mpe
    assert "_MYID" in cpe
    assert "athread_get" in cpe
    assert "athread_put" in cpe
    assert artifact.kernel_source == cpe
