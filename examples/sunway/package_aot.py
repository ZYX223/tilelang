"""Cross-compile any generated Sunway AOT project into a hybrid executable."""

from __future__ import annotations

import argparse
from pathlib import Path

from tilelang.sunway.runtime import SunwayKernelManifest, SunwayLibraryGenerator, SunwayToolchain


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--toolchain-root", type=Path, required=True)
    parser.add_argument("--overlay-root", type=Path, required=True)
    args = parser.parse_args()

    manifest = SunwayKernelManifest.read(args.generated_dir / "manifest.json")
    toolchain = SunwayToolchain.from_sdk_roots(
        toolchain_root=args.toolchain_root,
        overlay_root=args.overlay_root,
    )
    package = SunwayLibraryGenerator(
        project_dir=args.generated_dir,
        package_dir=args.package_dir,
        manifest=manifest,
        toolchain=toolchain,
    ).compile_executable()
    print(package.executable_path)


if __name__ == "__main__":
    main()
