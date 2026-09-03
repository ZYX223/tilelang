"""Package generated copy_128 sources for standalone or SWPyTorch execution."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from tilelang.sunway.runtime import (
    SunwayKernelManifest,
    SunwayLibraryGenerator,
    SunwayPythonSDK,
    SunwayToolchain,
    SunwayTorchSDK,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--toolchain-root", type=Path, required=True)
    parser.add_argument("--overlay-root", type=Path, required=True)
    parser.add_argument("--artifact", choices=("executable", "torch"), required=True)
    parser.add_argument("--swtorch-sdk-root", type=Path)
    args = parser.parse_args()

    manifest = SunwayKernelManifest.read(args.generated_dir / "manifest.json")
    toolchain = SunwayToolchain.from_sdk_roots(
        toolchain_root=args.toolchain_root,
        overlay_root=args.overlay_root,
    )
    generator = SunwayLibraryGenerator(
        project_dir=args.generated_dir,
        package_dir=args.package_dir,
        manifest=manifest,
        toolchain=toolchain,
    )

    if args.artifact == "executable":
        package = generator.compile_executable()
        print(package.executable_path)
        return

    if args.swtorch_sdk_root is None:
        parser.error("--swtorch-sdk-root is required for --artifact torch")
    package = generator.compile_torch_bundle(
        torch_sdk=SunwayTorchSDK(
            include_root=args.swtorch_sdk_root / "torch" / "include",
            library_root=args.swtorch_sdk_root / "torch" / "lib",
        ),
        python_sdk=SunwayPythonSDK(
            include_root=args.swtorch_sdk_root / "python" / "include" / "python3.6m",
            library_path=args.swtorch_sdk_root / "python" / "lib" / "libpython3.6m.so.1.0",
        ),
    )
    shutil.copy2(Path(__file__).with_name("run_copy_128_torch.py"), args.package_dir)
    print(package.python_launcher_path)


if __name__ == "__main__":
    main()
