from __future__ import annotations

import tilelang
import tilelang.language as T
from tilelang.backend.module import create_backend_context
from tilelang.sunway.runtime import SunwayKernelManifest
from tilelang.sunway.target import get_sunway_target_config
from testing.python.sunway.gemm_cases import make_gemm_32, make_gemm_m32_n16_k32


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
        target={"kind": "sunway", "output_dir": str(tmp_path), "output_indices": [1]},
        runtime_only=True,
    )

    s1 = (tmp_path / "s1_annotated_tir.txt").read_text()
    s2 = (tmp_path / "s2_semantic_tir.txt").read_text()
    s3 = (tmp_path / "s3_lowered_tir.txt").read_text()
    header = (tmp_path / "copy_128_common.h").read_text()
    mpe = (tmp_path / "mpe_copy_128.c").read_text()
    cpe = (tmp_path / "cpe_copy_128.c").read_text()
    manifest = SunwayKernelManifest.read(tmp_path / "manifest.json")

    # TVM Script prints the registered ``tl.tileop.copy`` node as ``T.copy``.
    assert "T.copy(" in s1
    assert "tilelang_sunway_dma_get" in s2
    assert "tilelang_sunway_dma_put" in s2
    assert "athread_get" not in s2
    assert "athread_get" in s3
    assert "athread_put" in s3
    assert "copy_128_args_t" in header
    assert "athread_init" not in mpe
    assert "athread_spawn" in mpe
    assert "athread_join" in mpe
    assert "_MYID" in cpe
    assert "athread_get" in cpe
    assert "athread_put" in cpe
    assert artifact.kernel_source == cpe
    assert [argument.role for argument in manifest.arguments] == ["input", "output"]


def test_sunway_large_copy_emits_a_grid_stride_cpe_loop(tmp_path) -> None:
    @T.prim_func
    def copy_4096(A: T.Tensor((4096,), "float32"), B: T.Tensor((4096,), "float32")):
        T.copy(A, B)

    tilelang.lower(
        copy_4096,
        target={
            "kind": "sunway",
            "output_dir": str(tmp_path),
            "output_indices": [1],
            "ldm_bytes_per_cpe": 256,
        },
        runtime_only=True,
    )

    cpe = (tmp_path / "cpe_copy_4096.c").read_text()
    assert "for (int tile_iteration = 0; tile_iteration < 2; ++tile_iteration)" in cpe
    assert "tile_iteration * 64" in cpe
    assert "< 74" in cpe
    assert "?" in cpe


def test_sunway_scalar_gemm_lowers_to_aot_project(tmp_path) -> None:
    artifact = tilelang.lower(
        make_gemm_32(),
        target={"kind": "sunway", "output_dir": str(tmp_path), "output_indices": [2]},
        runtime_only=True,
    )

    header = (tmp_path / "gemm_32_common.h").read_text()
    mpe = (tmp_path / "mpe_gemm_32.c").read_text()
    cpe = (tmp_path / "cpe_gemm_32.c").read_text()
    manifest = SunwayKernelManifest.read(tmp_path / "manifest.json")

    assert "void gemm_32_cpe(" in cpe
    assert "_MYID == 0" in cpe
    assert "for (int bx = 0; bx < 2; ++bx)" in cpe
    assert "for (int by = 0; by < 2; ++by)" in cpe
    assert "athread_get(PE_MODE" in cpe
    assert "athread_put(PE_MODE" in cpe
    assert "C_local[" in cpe
    assert "A_shared[" in cpe
    assert "B_shared[" in cpe
    assert " * " in cpe
    assert " + " in cpe
    assert header.count("float *") == 3
    assert mpe.count("athread_spawn") == 1
    assert mpe.count("athread_join") == 1
    assert "athread_init" not in mpe
    assert artifact.kernel_source == cpe
    assert [argument.role for argument in manifest.arguments] == ["input", "input", "output"]
    assert [argument.shape for argument in manifest.arguments] == [(32, 32), (32, 32), (32, 32)]


def test_sunway_non_square_scalar_gemm_uses_the_same_aot_emitter(tmp_path) -> None:
    tilelang.lower(
        make_gemm_m32_n16_k32(),
        target={"kind": "sunway", "output_dir": str(tmp_path), "output_indices": [2]},
        runtime_only=True,
    )

    cpe = (tmp_path / "cpe_gemm_m32_n16_k32.c").read_text()
    manifest = SunwayKernelManifest.read(tmp_path / "manifest.json")

    assert "for (int bx = 0; bx < 1; ++bx)" in cpe
    assert "for (int by = 0; by < 2; ++by)" in cpe
    assert [argument.shape for argument in manifest.arguments] == [(32, 32), (32, 16), (32, 16)]
    assert [argument.role for argument in manifest.arguments] == ["input", "input", "output"]
