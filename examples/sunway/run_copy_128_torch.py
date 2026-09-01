"""Run a packaged TileLang copy kernel against SWPyTorch tensors on SW9A."""

import argparse
import sys

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", required=True)
    args = parser.parse_args()

    sys.path.insert(0, args.package_dir)
    from tilelang_sunway_torch import load

    source = torch.arange(128, dtype=torch.float32)
    output = load(args.package_dir)(source)
    torch.testing.assert_allclose(output, source)
    print("SWPyTorch copy_128 passed: 128 elements")


if __name__ == "__main__":
    main()
