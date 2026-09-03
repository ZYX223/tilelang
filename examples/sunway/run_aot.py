"""Deploy and run any packaged Sunway hybrid executable on SW9A."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tilelang.sunway.runtime import SunwaySSHExecutor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--remote-host", required=True)
    parser.add_argument("--remote-root", default="/tmp/tilelang-runs")
    parser.add_argument("--executable", required=True)
    parser.add_argument("--deployment-id")
    args = parser.parse_args()

    result = SunwaySSHExecutor(
        remote_host=args.remote_host,
        remote_root=args.remote_root,
    ).deploy_and_run(
        args.package_dir,
        executable=args.executable,
        deployment_id=args.deployment_id,
    )
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
