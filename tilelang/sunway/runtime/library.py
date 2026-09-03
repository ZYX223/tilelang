"""SWGCC compilation of generated MPE/CPE source bundles."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path

from .manifest import SunwayKernelManifest
from .torch_wrapper import render_torch_registration_source, torch_extension_name


@dataclass(frozen=True, slots=True)
class SunwayToolchain:
    """Compiler path and stable flags for the SW39000/1307 toolchain."""

    compiler: Path
    cxx_compiler: Path | None = None
    mpe_flags: tuple[str, ...] = ("-mhost", "-fPIC")
    cpe_flags: tuple[str, ...] = ("-mslave", "-msimd", "-mieee", "-fPIC")
    shared_link_flags: tuple[str, ...] = ("-mdynamic",)
    executable_link_flags: tuple[str, ...] = ("-mhybrid",)
    sysroot: Path | None = None

    @classmethod
    def from_sdk_roots(cls, *, toolchain_root: Path, overlay_root: Path) -> SunwayToolchain:
        """Use the relocated 1307 compiler plus the SW9A PIC-runtime specs."""

        toolchain_root = Path(toolchain_root)
        overlay_root = Path(overlay_root)
        compiler = overlay_root / "bin" / "swgcc1307"
        if not compiler.is_file():
            raise FileNotFoundError(f"Missing SW9A compiler wrapper {compiler}")
        cxx_compiler = toolchain_root / "usr" / "bin" / "sw_64sw6a-sunway-linux-gnu-g++"
        if not cxx_compiler.is_file():
            raise FileNotFoundError(f"Missing SW9A C++ compiler {cxx_compiler}")
        return cls(compiler=compiler, cxx_compiler=cxx_compiler, sysroot=toolchain_root)


@dataclass(frozen=True, slots=True)
class SunwayTorchSDK:
    """Headers and link libraries copied from one SW PyTorch installation."""

    include_root: Path
    library_root: Path
    runtime_library_root: str = "/usr/sw/swpython/lib/python3.6/site-packages/torch/lib"
    cxx_abi: int = 1


@dataclass(frozen=True, slots=True)
class SunwayPythonSDK:
    """Headers and shared library for the Python installed on SW9A."""

    include_root: Path
    library_path: Path
    runtime_home: str = "/usr/sw/swpython"
    runtime_library_root: str = "/usr/sw/swpython/lib"


@dataclass(frozen=True, slots=True)
class SunwayArtifactPackage:
    """Paths produced by one LibraryGenerator invocation."""

    package_dir: Path
    manifest_path: Path
    library_path: Path | None = None
    executable_path: Path | None = None
    torch_library_path: Path | None = None
    python_launcher_path: Path | None = None


class SunwayLibraryGenerator:
    """Compile one generated Sunway project without importing TVM on the target."""

    def __init__(
        self,
        *,
        project_dir: Path,
        package_dir: Path,
        manifest: SunwayKernelManifest,
        toolchain: SunwayToolchain,
    ) -> None:
        self.project_dir = Path(project_dir)
        self.package_dir = Path(package_dir)
        self.manifest = manifest
        self.toolchain = toolchain

    def compile_shared(self) -> SunwayArtifactPackage:
        sources = self._stage_sources()
        mpe_object = self._compile_mpe(sources["mpe"])
        cpe_object = self._compile_cpe(sources["cpe"])
        library = self.package_dir / f"lib{self.manifest.kernel_name}.so"
        self._run(
            *self.toolchain.shared_link_flags,
            "-shared",
            *self._sysroot_link_flags(),
            str(mpe_object),
            str(cpe_object),
            f"-Wl,-soname,{library.name}",
            "-o",
            str(library),
        )
        runtime_files = self._stage_runtime_files()
        manifest_path = self._write_manifest({"library": library.name, **sources, **runtime_files})
        return SunwayArtifactPackage(self.package_dir, manifest_path, library_path=library)

    def compile_executable(self) -> SunwayArtifactPackage:
        sources = self._stage_sources(include_main=True)
        mpe_object = self._compile_mpe(sources["mpe"])
        cpe_object = self._compile_cpe(sources["cpe"])
        main_object = self._compile_host_source(sources["main"])
        executable = self.package_dir / self.manifest.kernel_name
        self._run(
            *self.toolchain.executable_link_flags,
            str(main_object),
            str(mpe_object),
            str(cpe_object),
            "-o",
            str(executable),
        )
        manifest_path = self._write_manifest({"executable": executable.name, **sources})
        return SunwayArtifactPackage(self.package_dir, manifest_path, executable_path=executable)

    def compile_torch_extension(self, torch_sdk: SunwayTorchSDK) -> SunwayArtifactPackage:
        """Cross-compile the boxed torch.ops registration library on the build host."""

        if self.toolchain.cxx_compiler is None:
            raise ValueError("Sunway C++ compiler is required for the PyTorch registration library")
        include_root = Path(torch_sdk.include_root)
        library_root = Path(torch_sdk.library_root)
        if not include_root.is_dir() or not library_root.is_dir():
            raise FileNotFoundError("Sunway PyTorch SDK include and library directories must exist")

        source = self.package_dir / "torch_extension.cpp"
        kernel_library = self.package_dir / f"lib{self.manifest.kernel_name}.so"
        if not source.is_file() or not kernel_library.is_file():
            raise FileNotFoundError("compile_shared() must run before compile_torch_extension()")

        torch_object = self.package_dir / "torch_extension.o"
        include_dirs = (
            include_root,
            include_root / "torch" / "csrc" / "api" / "include",
            include_root / "TH",
            include_root / "THC",
        )
        self._run_with(
            self.toolchain.cxx_compiler,
            "-mhost",
            "-fPIC",
            "-O2",
            "-std=c++14",
            f"-D_GLIBCXX_USE_CXX11_ABI={torch_sdk.cxx_abi}",
            *(f"-I{path}" for path in include_dirs),
            "-c",
            str(source),
            "-o",
            str(torch_object),
        )

        torch_library = self.package_dir / torch_extension_name(self.manifest)
        self._run_with(
            self.toolchain.cxx_compiler,
            "-mhost",
            "-shared",
            *self._sysroot_link_flags(),
            str(torch_object),
            f"-L{library_root}",
            "-lc10",
            "-ltorch",
            "-ltorch_cpu",
            f"-Wl,-rpath,{torch_sdk.runtime_library_root}",
            "-o",
            str(torch_library),
        )

        packaged_manifest = SunwayKernelManifest.read(self.package_dir / "manifest.json")
        artifacts = dict(packaged_manifest.artifacts)
        artifacts["torch_library"] = torch_library.name
        packaged_manifest = replace(packaged_manifest, artifacts=artifacts)
        manifest_path = self.package_dir / "manifest.json"
        packaged_manifest.write(manifest_path)
        return SunwayArtifactPackage(
            self.package_dir,
            manifest_path,
            library_path=kernel_library,
            torch_library_path=torch_library,
        )

    def compile_python_launcher(self, python_sdk: SunwayPythonSDK) -> SunwayArtifactPackage:
        """Link Python and the generated CPE library into one SW9A AOT image.

        SWGCC dynamic mode discovers CPE text from dependencies present when the
        process starts. Loading a new mixed MPE/CPE library after startup leaves
        its CPE text unmapped, so the launcher must retain the kernel library as
        a DT_NEEDED dependency even though Python resolves its C ABI at runtime.
        """

        include_root = Path(python_sdk.include_root)
        python_library = Path(python_sdk.library_path)
        kernel_library = self.package_dir / f"lib{self.manifest.kernel_name}.so"
        if not include_root.is_dir() or not python_library.is_file():
            raise FileNotFoundError("Sunway Python SDK headers and shared library must exist")
        if not kernel_library.is_file():
            raise FileNotFoundError("compile_shared() must run before compile_python_launcher()")

        source = self.package_dir / "swpython_launcher.c"
        shutil.copy2(Path(__file__).with_name("swpython_launcher.c"), source)
        launcher_object = self.package_dir / "swpython_launcher.o"
        self._run(
            "-mhost",
            "-O2",
            f"-I{include_root}",
            "-c",
            str(source),
            "-o",
            str(launcher_object),
        )

        launcher = self.package_dir / "tilelang_swpython"
        self._run(
            *self.toolchain.shared_link_flags,
            *self._sysroot_link_flags(),
            "-Wl,--dynamic-linker=/usr/sw/lib/ld-linux.so.2",
            "-Wl,--export-dynamic",
            "-Wl,--undefined=__crts_cg_shared",
            "-Wl,--undefined=__crts_cg_proc",
            str(launcher_object),
            str(python_library),
            "-Wl,--no-as-needed",
            str(kernel_library),
            "-Wl,--as-needed",
            "-lpthread",
            "-ldl",
            "-lutil",
            "-lm",
            f"-Wl,-rpath,$ORIGIN:/usr/sw/lib:{python_sdk.runtime_library_root}",
            "-o",
            str(launcher),
        )

        packaged_manifest = SunwayKernelManifest.read(self.package_dir / "manifest.json")
        artifacts = dict(packaged_manifest.artifacts)
        artifacts.update(
            {
                "python_launcher": launcher.name,
                "python_launcher_source": source.name,
            }
        )
        packaged_manifest = replace(packaged_manifest, artifacts=artifacts)
        manifest_path = self.package_dir / "manifest.json"
        packaged_manifest.write(manifest_path)
        torch_library_name = artifacts.get("torch_library")
        torch_library = self.package_dir / torch_library_name if torch_library_name else None
        return SunwayArtifactPackage(
            self.package_dir,
            manifest_path,
            library_path=kernel_library,
            torch_library_path=torch_library,
            python_launcher_path=launcher,
        )

    def compile_torch_bundle(
        self,
        *,
        torch_sdk: SunwayTorchSDK,
        python_sdk: SunwayPythonSDK,
    ) -> SunwayArtifactPackage:
        """Build the kernel, torch.ops registration, and startup-linked Python."""

        self.compile_shared()
        self.compile_torch_extension(torch_sdk)
        return self.compile_python_launcher(python_sdk)

    def _stage_sources(self, *, include_main: bool = False) -> dict[str, Path]:
        self.package_dir.mkdir(parents=True, exist_ok=True)
        names = {
            "header": f"{self.manifest.kernel_name}_common.h",
            "mpe": f"mpe_{self.manifest.kernel_name}.c",
            "cpe": f"cpe_{self.manifest.kernel_name}.c",
        }
        if include_main:
            names["main"] = f"{self.manifest.kernel_name}_main.c"

        staged: dict[str, Path] = {}
        for key, name in names.items():
            source = self.project_dir / name
            if not source.is_file():
                raise FileNotFoundError(f"Missing generated Sunway source {source}")
            destination = self.package_dir / name
            shutil.copy2(source, destination)
            staged[key] = destination
        return staged

    def _compile_mpe(self, source: Path) -> Path:
        output = self.package_dir / f"{source.stem}.o"
        self._run(*self.toolchain.mpe_flags, f"-I{self.package_dir}", "-c", str(source), "-o", str(output))
        return output

    def _compile_cpe(self, source: Path) -> Path:
        output = self.package_dir / f"{source.stem}.o"
        self._run(*self.toolchain.cpe_flags, f"-I{self.package_dir}", "-c", str(source), "-o", str(output))
        return output

    def _compile_host_source(self, source: Path) -> Path:
        output = self.package_dir / f"{source.stem}.o"
        self._run(*self.toolchain.mpe_flags, f"-I{self.package_dir}", "-c", str(source), "-o", str(output))
        return output

    def _write_manifest(self, artifacts: dict[str, object]) -> Path:
        relative_artifacts = {key: Path(value).name for key, value in artifacts.items()}
        manifest = replace(self.manifest, artifacts=relative_artifacts)
        path = self.package_dir / "manifest.json"
        manifest.write(path)
        return path

    def _stage_runtime_files(self) -> dict[str, Path]:
        adapter = self.package_dir / "tilelang_sunway_adapter.py"
        shutil.copy2(Path(__file__).with_name("adapter.py"), adapter)
        torch_wrapper = self.package_dir / "tilelang_sunway_torch.py"
        shutil.copy2(Path(__file__).with_name("target_torch.py"), torch_wrapper)
        torch_extension = self.package_dir / "torch_extension.cpp"
        torch_extension.write_text(render_torch_registration_source(self.manifest), encoding="utf-8")
        return {
            "adapter": adapter,
            "torch_wrapper": torch_wrapper,
            "torch_extension": torch_extension,
        }

    def _run(self, *arguments: str) -> None:
        self._run_with(self.toolchain.compiler, *arguments)

    @staticmethod
    def _run_with(compiler: Path, *arguments: str) -> None:
        command = [str(compiler), *arguments]
        subprocess.run(command, check=True, text=True)

    def _sysroot_link_flags(self) -> tuple[str, ...]:
        if self.toolchain.sysroot is None:
            return ()
        return (f"-Wl,--sysroot={self.toolchain.sysroot}",)
