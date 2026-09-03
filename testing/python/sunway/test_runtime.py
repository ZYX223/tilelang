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
    SunwayPythonSDK,
    SunwaySSHExecutor,
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


def test_standalone_copy_example_owns_crts_initialization() -> None:
    example = Path(__file__).parents[3] / "examples" / "sunway" / "copy_128_main.c"
    source = example.read_text(encoding="utf-8")

    assert '#include "athread.h"' in source
    assert source.count("athread_init();") == 1


@pytest.mark.parametrize(
    ("filename", "signature", "success_line"),
    [
        (
            "gemm_32_main.c",
            "void gemm_32(float *A, float *B, float *C);",
            "gemm_32 passed: M=32 N=32 K=32",
        ),
        (
            "gemm_m32_n16_k32_main.c",
            "void gemm_m32_n16_k32(float *A, float *B, float *C);",
            "gemm_m32_n16_k32 passed: M=32 N=16 K=32",
        ),
    ],
)
def test_standalone_gemm_examples_own_one_crts_initialization(
    filename: str,
    signature: str,
    success_line: str,
) -> None:
    example = Path(__file__).parents[3] / "examples" / "sunway" / filename
    source = example.read_text(encoding="utf-8")

    assert source.count("athread_init();") == 1
    assert signature in source
    assert success_line in source


def test_generic_aot_scripts_do_not_name_a_kernel() -> None:
    example_dir = Path(__file__).parents[3] / "examples" / "sunway"
    package_source = (example_dir / "package_aot.py").read_text(encoding="utf-8")
    run_source = (example_dir / "run_aot.py").read_text(encoding="utf-8")

    for name in ("copy_128", "gemm_32", "gemm_m32_n16_k32", "gemm_128_k64"):
        assert name not in package_source
        assert name not in run_source


def test_sunway_codegen_selects_simd_from_the_native_leaf_not_a_kernel_name() -> None:
    source = (Path(__file__).parents[3] / "tilelang" / "sunway" / "codegen.py").read_text(
        encoding="utf-8"
    )

    assert "tilelang_sunway_native_fma_f32x8" in source
    assert "gemm_128_k64" not in source


def test_distributed_multik_gemm_example_keeps_official_frontend_and_reference() -> None:
    example_dir = Path(__file__).parents[3] / "examples" / "sunway"
    generator = (example_dir / "gemm_128_k64.py").read_text(encoding="utf-8")
    main = (example_dir / "gemm_128_k64_main.c").read_text(encoding="utf-8")

    assert "with T.Kernel(8, 8, threads=64)" in generator
    assert "for ko in T.Pipelined(2, num_stages=1)" in generator
    assert generator.count("T.copy(") == 3
    assert generator.count("T.gemm(") == 1
    assert '"gemm_ownership": "mesh_2d"' in generator
    assert 'choices=("scalar", "simd")' in generator
    assert 'target["gemm_compute"] = args.compute' in generator

    assert main.count("athread_init();") == 1
    assert "void gemm_128_k64(float *A, float *B, float *C);" in main
    assert "for (int m = 0; m < M; ++m)" in main
    assert "for (int n = 0; n < N; ++n)" in main
    assert "for (int k = 0; k < K; ++k)" in main
    assert "gemm_128_k64 passed: M=128 N=128 K=64" in main


def test_distributed_gemm_main_measures_seven_post_warmup_kernel_calls() -> None:
    main = (
        Path(__file__).parents[3] / "examples" / "sunway" / "gemm_128_k64_main.c"
    ).read_text(encoding="utf-8")

    assert "#include <time.h>" in main
    assert "MEASURED_RUNS = 7" in main
    assert "clock_gettime(CLOCK_MONOTONIC" in main
    assert main.count("gemm_128_k64(A, B, C);") == 2
    assert "sort_elapsed(elapsed_ms, MEASURED_RUNS);" in main
    success = main.index("gemm_128_k64 passed: M=128 N=128 K=64")
    timing = main.index("gemm_128_k64 median_ms: %.6f over %d runs")
    assert success < timing


def test_sunway_guide_records_g1_reproduction_and_scope() -> None:
    guide = (Path(__file__).parents[3] / "docs" / "get_started" / "sunway.md").read_text(
        encoding="utf-8"
    )
    normalized = " ".join(guide.split())

    assert "## Distributed Multi-K SIMD GEMM G1" in guide
    assert '--compute scalar' in guide
    assert '--compute simd' in guide
    assert '"gemm_ownership": "mesh_2d"' in guide
    assert "gemm_128_k64 median_ms: 0.170961 over 7 runs" in guide
    assert "gemm_128_k64 median_ms: 0.109881 over 7 runs" in guide
    assert "synchronous DMA" in normalized
    assert "does not implement double buffering" in normalized
    assert "does not package a PyTorch operator" in normalized


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


def test_library_generator_startup_links_python_and_cpe_bundle(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    package_dir = tmp_path / "package"
    _write_copy_project(project_dir)
    compiler = _write_fake_compiler(tmp_path / "fake-swgcc")
    toolchain = SunwayToolchain(compiler=compiler, sysroot=tmp_path)
    python_include = tmp_path / "python-include"
    python_include.mkdir()
    python_library = tmp_path / "libpython3.6m.so.1.0"
    python_library.write_text("fake Python library\n", encoding="utf-8")
    generator = SunwayLibraryGenerator(
        project_dir=project_dir,
        package_dir=package_dir,
        manifest=_copy_manifest(),
        toolchain=toolchain,
    )
    generator.compile_shared()

    package = generator.compile_python_launcher(
        SunwayPythonSDK(include_root=python_include, library_path=python_library)
    )

    assert package.python_launcher_path == package_dir / "tilelang_swpython"
    assert package.python_launcher_path.is_file()
    assert (package_dir / "swpython_launcher.c").is_file()
    link_args = (package_dir / "tilelang_swpython.args").read_text().splitlines()
    assert "-mdynamic" in link_args
    assert "-mhybrid" not in link_args
    assert "-Wl,--no-as-needed" in link_args
    assert str(package_dir / "libcopy_128.so") in link_args
    assert "-Wl,-rpath,$ORIGIN:/usr/sw/lib:/usr/sw/swpython/lib" in link_args
    packaged_manifest = SunwayKernelManifest.read(package.manifest_path)
    assert packaged_manifest.artifacts["python_launcher"] == "tilelang_swpython"


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


@pytest.mark.parametrize("deployment_id", [".", ".."])
def test_ssh_executor_rejects_dot_path_deployment_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, deployment_id: str
) -> None:
    package_dir = tmp_path / "copy-package"
    package_dir.mkdir()
    calls = []

    def fake_run(command, *, check, capture_output, text):
        calls.append((command, check))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("tilelang.sunway.runtime.executor.subprocess.run", fake_run)
    executor = SunwaySSHExecutor(
        remote_host="root@10.10.10.22",
        remote_root="/tmp/tilelang-runs",
    )

    with pytest.raises(ValueError, match="Unsafe Sunway deployment id"):
        executor.deploy(package_dir, deployment_id=deployment_id)

    assert calls == []


def test_ssh_executor_replaces_existing_deployment_before_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_dir = tmp_path / "copy-package"
    package_dir.mkdir()
    calls = []

    def fake_run(command, *, check, capture_output, text):
        calls.append((command, check))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("tilelang.sunway.runtime.executor.subprocess.run", fake_run)
    executor = SunwaySSHExecutor(
        remote_host="root@10.10.10.22",
        remote_root="/tmp/tilelang-runs",
    )

    executor.deploy(package_dir, deployment_id="reused-run")
    executor.deploy(package_dir, deployment_id="reused-run")

    prepare = [
        "ssh",
        "root@10.10.10.22",
        "rm -rf /tmp/tilelang-runs/reused-run && mkdir -p /tmp/tilelang-runs/reused-run",
    ]
    copy = [
        "scp",
        "-r",
        str(package_dir),
        "root@10.10.10.22:/tmp/tilelang-runs/reused-run/package",
    ]
    assert calls == [(prepare, True), (copy, True), (prepare, True), (copy, True)]


def test_ssh_executor_deploys_package_and_launches_swrun_on_9a(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_dir = tmp_path / "copy-package"
    package_dir.mkdir()
    executable = package_dir / "copy_128"
    executable.write_text("fake executable\n", encoding="utf-8")
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    _copy_manifest().write(package_dir / "manifest.json")
    calls = []

    def fake_run(command, *, check, capture_output, text):
        calls.append((command, check))
        if command[0] == "ssh" and "swrun" in command[-1]:
            return subprocess.CompletedProcess(command, 0, "copy_128 passed: 128 elements\n", "")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("tilelang.sunway.runtime.executor.subprocess.run", fake_run)
    executor = SunwaySSHExecutor(
        remote_host="root@10.10.10.22",
        remote_root="/tmp/tilelang-runs",
    )

    result = executor.deploy_and_run(
        package_dir,
        executable="copy_128",
        deployment_id="run-8421",
    )

    assert result.succeeded
    assert result.stdout == "copy_128 passed: 128 elements\n"
    assert result.deployment.remote_directory == "/tmp/tilelang-runs/run-8421/package"
    assert calls == [
        (
            [
                "ssh",
                "root@10.10.10.22",
                "rm -rf /tmp/tilelang-runs/run-8421 && "
                "mkdir -p /tmp/tilelang-runs/run-8421",
            ],
            True,
        ),
        (
            [
                "scp",
                "-r",
                str(package_dir),
                "root@10.10.10.22:/tmp/tilelang-runs/run-8421/package",
            ],
            True,
        ),
        (
            [
                "ssh",
                "root@10.10.10.22",
                "cd /tmp/tilelang-runs/run-8421/package && chmod +x ./copy_128 && "
                "swrun -E 64 -i ./copy_128",
            ],
            False,
        ),
    ]


def test_ssh_executor_returns_remote_kernel_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_dir = tmp_path / "copy-package"
    package_dir.mkdir()
    executable = package_dir / "copy_128"
    executable.write_text("fake executable\n", encoding="utf-8")

    def fake_run(command, *, check, capture_output, text):
        if command[0] == "ssh" and "swrun" in command[-1]:
            return subprocess.CompletedProcess(command, 7, "", "kernel failed\n")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("tilelang.sunway.runtime.executor.subprocess.run", fake_run)
    executor = SunwaySSHExecutor(
        remote_host="root@10.10.10.22",
        remote_root="/tmp/tilelang-runs",
    )

    result = executor.deploy_and_run(
        package_dir,
        executable="copy_128",
        deployment_id="failed-run",
    )

    assert not result.succeeded
    assert result.returncode == 7
    assert result.stderr == "kernel failed\n"


def test_ssh_executor_runs_startup_linked_python_without_swrun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_dir = tmp_path / "copy-package"
    package_dir.mkdir()
    (package_dir / "tilelang_swpython").write_text("fake launcher\n", encoding="utf-8")
    (package_dir / "run_copy.py").write_text("print('copy passed')\n", encoding="utf-8")
    calls = []

    def fake_run(command, *, check, capture_output, text):
        calls.append((command, check))
        if command[0] == "ssh" and "./tilelang_swpython ./run_copy.py" in command[-1]:
            return subprocess.CompletedProcess(command, 0, "copy passed\n", "")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("tilelang.sunway.runtime.executor.subprocess.run", fake_run)
    executor = SunwaySSHExecutor(
        remote_host="root@10.10.10.22",
        remote_root="/tmp/tilelang-runs",
    )

    result = executor.deploy_and_run_python(
        package_dir,
        launcher="tilelang_swpython",
        script="run_copy.py",
        deployment_id="python-run",
    )

    assert result.succeeded
    assert result.stdout == "copy passed\n"
    remote_command = calls[-1][0][-1]
    assert ". /usr/sw/swpython/setenv" in remote_command
    assert "PYTHONHOME=/usr/sw/swpython" in remote_command
    assert "STASK_SEG_DATA=64" in remote_command
    assert "./tilelang_swpython ./run_copy.py" in remote_command
    assert "swrun" not in remote_command
