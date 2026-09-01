"""Package generated copy_128 sources for standalone or SWPyTorch execution."""

from __future__ import annotations

import argparse
from pathlib import Path

from tilelang.sunway.runtime import (
    SunwayKernelManifest,
    SunwayLibraryGenerator,
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
    generator.compile_shared()
    package = generator.compile_torch_extension(
        SunwayTorchSDK(
            include_root=args.swtorch_sdk_root / "torch" / "include",
            library_root=args.swtorch_sdk_root / "torch" / "lib",
        )
    )
    print(package.torch_library_path)


if __name__ == "__main__":
    main()
