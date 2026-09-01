from __future__ import annotations

import json
import stat
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from tilelang.sunway.runtime import (
    SunwayKernelAdapter,
    SunwayKernelArgument,
    SunwayKernelManifest,
    SunwayLibraryGenerator,
    SunwaySlurmRelayExecutor,
    SunwayTorchSDK,
    SunwayTorchOperator,
    SunwayToolchain,
    render_torch_registration_source,
)


def _write_fake_compiler(path: Path) -> Path:
    path.write_text(
        """#!/usr/bin/env python3
import pathlib
import sys

args = sys.argv[1:]
output = pathlib.Path(args[args.index("-o") + 1])
output.write_text("fake artifact\\n", encoding="utf-8")
with output.with_suffix(output.suffix + ".args").open("w", encoding="utf-8") as log:
    log.write("\\n".join(args))
""",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _write_fake_scheduler(path: Path, output: str) -> Path:
    path.write_text(
        f"""#!/bin/sh
printf '%s\\n' "$@" > "{path}.args"
printf '%s\\n' "{output}"
""",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _write_copy_project(project_dir: Path) -> None:
    project_dir.mkdir()
    (project_dir / "copy_128_common.h").write_text("typedef struct copy_128_args copy_128_args_t;\n", encoding="utf-8")
    (project_dir / "mpe_copy_128.c").write_text("void copy_128(float *A, float *B) {}\n", encoding="utf-8")
    (project_dir / "cpe_copy_128.c").write_text("void copy_128_cpe(void *args) {}\n", encoding="utf-8")


def _copy_manifest() -> SunwayKernelManifest:
    return SunwayKernelManifest(
        kernel_name="copy_128",
        symbol="copy_128",
        arguments=(
            SunwayKernelArgument("A", "float32", (128,), "input"),
            SunwayKernelArgument("B", "float32", (128,), "output"),
        ),
    )


class _CPUDevice:
    type = "cpu"


class _TestTensor:
    def __init__(self, values: list[float]):
        import ctypes

        self._storage = (ctypes.c_float * len(values))(*values)
        self.shape = (len(values),)
        self.dtype = "float32"
        self.device = _CPUDevice()

    def is_contiguous(self) -> bool:
        return True

    def data_ptr(self) -> int:
        import ctypes

        return ctypes.addressof(self._storage)

    def values(self) -> list[float]:
        return list(self._storage)


class _TestTorch:
    float32 = "float32"

    @staticmethod
    def empty(shape, *, dtype, device):
        assert dtype == "float32"
        assert device.type == "cpu"
        size = 1
        for dim in shape:
            size *= dim
        return _TestTensor([0.0] * size)


def _compile_host_copy_library(package_dir: Path) -> None:
    source = package_dir / "copy.c"
    source.write_text(
        "void copy_128(float *A, float *B) { for (int i = 0; i < 128; ++i) B[i] = A[i]; }\n",
        encoding="utf-8",
    )
    subprocess.run(
        ["cc", "-shared", "-fPIC", str(source), "-o", str(package_dir / "libcopy_128.so")],
        check=True,
    )
    manifest = _copy_manifest()
    manifest = replace(manifest, artifacts={"library": "libcopy_128.so"})
    manifest.write(package_dir / "manifest.json")


def test_sunway_manifest_round_trips_without_tilelang_runtime(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest = _copy_manifest()

    manifest.write(manifest_path)
    loaded = SunwayKernelManifest.read(manifest_path)

    assert loaded == manifest
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["arguments"][1]["role"] == "output"


def test_library_generator_compiles_mpe_and_cpe_into_shared_package(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    package_dir = tmp_path / "package"
    _write_copy_project(project_dir)
    compiler = _write_fake_compiler(tmp_path / "fake-swgcc")
    toolchain = SunwayToolchain(compiler=compiler)

    package = SunwayLibraryGenerator(
        project_dir=project_dir,
        package_dir=package_dir,
        manifest=_copy_manifest(),
        toolchain=toolchain,
    ).compile_shared()

    assert package.library_path == package_dir / "libcopy_128.so"
    assert package.library_path.is_file()
    assert package.manifest_path.is_file()
    assert (package_dir / "tilelang_sunway_adapter.py").is_file()
    assert (package_dir / "tilelang_sunway_torch.py").is_file()
    registration = (package_dir / "torch_extension.cpp").read_text(encoding="utf-8")
    assert 'schema("tilelang_sunway::copy_128(Tensor A, Tensor B) -> Tensor")' in registration
    assert (package_dir / "mpe_copy_128.o.args").read_text().splitlines()[:2] == ["-mhost", "-fPIC"]
    assert (package_dir / "cpe_copy_128.o.args").read_text().splitlines()[:3] == ["-mslave", "-msimd", "-mieee"]
    link_args = (package_dir / "libcopy_128.so.args").read_text().splitlines()
    assert "-mdynamic" in link_args
    assert "-shared" in link_args


def test_library_generator_can_emit_current_swrun_executable(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    package_dir = tmp_path / "package"
    _write_copy_project(project_dir)
    (project_dir / "copy_128_main.c").write_text("int main(void) { return 0; }\n", encoding="utf-8")
    compiler = _write_fake_compiler(tmp_path / "fake-swgcc")

    package = SunwayLibraryGenerator(
        project_dir=project_dir,
        package_dir=package_dir,
        manifest=_copy_manifest(),
        toolchain=SunwayToolchain(compiler=compiler),
    ).compile_executable()

    assert package.executable_path == package_dir / "copy_128"
    assert package.executable_path.is_file()
    link_args = (package_dir / "copy_128.args").read_text().splitlines()
    assert "-mhybrid" in link_args
    assert str(package_dir / "copy_128_main.o") in link_args


def test_library_generator_cross_compiles_boxed_torch_registration(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    package_dir = tmp_path / "package"
    _write_copy_project(project_dir)
    compiler = _write_fake_compiler(tmp_path / "fake-swgcc")
    toolchain = SunwayToolchain(compiler=compiler, cxx_compiler=compiler, sysroot=tmp_path)
    torch_sdk = SunwayTorchSDK(
        include_root=tmp_path / "torch-include",
        library_root=tmp_path / "torch-lib",
    )
    torch_sdk.include_root.mkdir()
    torch_sdk.library_root.mkdir()
    generator = SunwayLibraryGenerator(
        project_dir=project_dir,
        package_dir=package_dir,
        manifest=_copy_manifest(),
        toolchain=toolchain,
    )
    generator.compile_shared()

    package = generator.compile_torch_extension(torch_sdk)

    assert package.torch_library_path == package_dir / "tilelang_sunway_copy_128_ops.so"
    compile_args = (package_dir / "torch_extension.o.args").read_text().splitlines()
    assert compile_args[:4] == ["-mhost", "-fPIC", "-O2", "-std=c++14"]
    link_args = (package_dir / "tilelang_sunway_copy_128_ops.so.args").read_text().splitlines()
    assert "-lc10" in link_args
    assert "-ltorch_cpu" in link_args
    assert "-lcopy_128" not in link_args
    packaged_manifest = SunwayKernelManifest.read(package.manifest_path)
    assert packaged_manifest.artifacts["torch_library"] == "tilelang_sunway_copy_128_ops.so"


def test_kernel_adapter_invokes_pointer_abi_from_manifest(tmp_path: Path) -> None:
    _compile_host_copy_library(tmp_path)
    adapter = SunwayKernelAdapter(tmp_path)
    source = _TestTensor([float(index) for index in range(128)])
    destination = _TestTensor([-1.0] * 128)

    adapter.invoke(source, destination)

    assert destination.values() == source.values()


def test_kernel_adapter_exports_kernel_symbol_for_torch_registration(tmp_path: Path) -> None:
    import ctypes

    manifest = replace(_copy_manifest(), artifacts={"library": "libcopy_128.so"})
    manifest.write(tmp_path / "manifest.json")
    calls = []

    class FakeFunction:
        argtypes = None
        restype = None

    class FakeLibrary:
        copy_128 = FakeFunction()

    def loader(path, *, mode):
        calls.append((path, mode))
        return FakeLibrary()

    SunwayKernelAdapter(tmp_path, library_loader=loader)

    assert calls == [(str(tmp_path / "libcopy_128.so"), ctypes.RTLD_GLOBAL)]


def test_kernel_adapter_rejects_shape_mismatch_before_launch(tmp_path: Path) -> None:
    _compile_host_copy_library(tmp_path)
    adapter = SunwayKernelAdapter(tmp_path)

    with pytest.raises(ValueError, match="B.*shape"):
        adapter.invoke(_TestTensor([0.0] * 128), _TestTensor([0.0] * 64))


def test_torch_operator_allocates_manifest_outputs_and_calls_adapter(tmp_path: Path) -> None:
    _compile_host_copy_library(tmp_path)
    operator = SunwayTorchOperator(SunwayKernelAdapter(tmp_path), tensor_module=_TestTorch)
    source = _TestTensor([float(index) for index in range(128)])

    output = operator(source)

    assert output.values() == source.values()


def test_torch_operator_loads_registered_package_and_hides_output_argument(tmp_path: Path) -> None:
    _compile_host_copy_library(tmp_path)
    extension = tmp_path / "tilelang_sunway_copy_128_ops.so"
    extension.write_text("fake registration library\n", encoding="utf-8")
    manifest = SunwayKernelManifest.read(tmp_path / "manifest.json")
    manifest = replace(
        manifest,
        artifacts={**manifest.artifacts, "torch_library": extension.name},
    )
    manifest.write(tmp_path / "manifest.json")

    loaded_libraries = []

    class RegisteredNamespace:
        @staticmethod
        def copy_128(source, destination):
            for index, value in enumerate(source.values()):
                destination._storage[index] = value
            return destination

    class RegisteredOps:
        tilelang_sunway = RegisteredNamespace()

        @staticmethod
        def load_library(path):
            loaded_libraries.append(path)

    class RegisteredTorch(_TestTorch):
        ops = RegisteredOps()

    operator = SunwayTorchOperator.from_registered_package(tmp_path, tensor_module=RegisteredTorch)
    source = _TestTensor([float(index) for index in range(128)])

    output = operator(source)

    assert output.values() == source.values()
    assert loaded_libraries == [str(extension)]


def test_torch_registration_source_targets_legacy_swtorch_api() -> None:
    source = render_torch_registration_source(_copy_manifest(), namespace="tilelang_sunway")

    assert "#include <ATen/core/TensorBody.h>" in source
    assert "#include <torch/script.h>" not in source
    assert 'extern "C" void copy_128(float* A, float* B);' in source
    assert "const c10::OperatorHandle&, c10::Stack* stack" in source
    assert 'schema("tilelang_sunway::copy_128(Tensor A, Tensor B) -> Tensor")' in source
    assert "catchAllKernel<&tilelang_sunway_copy_128_boxed>()" in source
    assert "A.unsafeGetTensorImpl()->device().is_cpu()" in source
    assert "A.device().is_cpu()" not in source
    assert "copy_128(A.data_ptr<float>(), B.data_ptr<float>());" in source
    assert "torch::jit::push(*stack, B);" in source


def test_slurm_relay_submits_dell_job_that_launches_swrun_on_9a(tmp_path: Path) -> None:
    package_dir = tmp_path / "copy-package"
    package_dir.mkdir()
    executable = package_dir / "copy_128"
    executable.write_text("fake executable\n", encoding="utf-8")
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    _copy_manifest().write(package_dir / "manifest.json")
    sbatch = _write_fake_scheduler(tmp_path / "sbatch", "8421")

    job = SunwaySlurmRelayExecutor(
        remote_host="root@10.10.10.22",
        remote_root="/tmp/tilelang-jobs",
        partition="q_dell",
        account="sunway-project",
        sbatch=sbatch,
    ).submit(package_dir, executable="copy_128")

    assert job.scheduler_job_id == "8421"
    assert job.submit_script.is_file()
    script = job.submit_script.read_text(encoding="utf-8")
    assert "scp" in script
    assert "copy-package/." not in script
    assert '"$REMOTE_DIR/package"' in script
    assert "root@10.10.10.22" in script
    assert "swrun -E 64 -i ./copy_128" in script
    submit_args = Path(f"{sbatch}.args").read_text(encoding="utf-8").splitlines()
    assert submit_args[:6] == [
        "--parsable",
        "--partition",
        "q_dell",
        "--account",
        "sunway-project",
        "--job-name",
    ]


def test_slurm_relay_reads_active_job_state(tmp_path: Path) -> None:
    squeue = _write_fake_scheduler(tmp_path / "squeue", "RUNNING")
    executor = SunwaySlurmRelayExecutor(
        remote_host="root@10.10.10.22",
        remote_root="/tmp/tilelang-jobs",
        squeue=squeue,
    )

    assert executor.status("8421") == "RUNNING"
