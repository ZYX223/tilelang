"""Dell-side job submission for staging and launching SW9A executables."""

from __future__ import annotations

import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path


_REMOTE_HOST_PATTERN = re.compile(r"^[A-Za-z0-9_.@-]+$")
_SLURM_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True, slots=True)
class SunwaySubmittedJob:
    """Identity and local evidence paths for one submitted relay job."""

    scheduler_job_id: str
    submit_script: Path
    stdout_pattern: Path
    stderr_pattern: Path


class SunwaySlurmRelayExecutor:
    """Queue an SSH/SCP relay on Dell Slurm, then launch the binary with swrun.

    Dell Slurm accounts for the relay process only. It does not reserve the SW9A
    node until that node is formally added to the scheduler configuration.
    """

    def __init__(
        self,
        *,
        remote_host: str,
        remote_root: str,
        partition: str = "q_dell",
        account: str | None = None,
        sbatch: str | Path = "sbatch",
        squeue: str | Path = "squeue",
        ssh: str | Path = "ssh",
        scp: str | Path = "scp",
        launcher: tuple[str, ...] = ("swrun", "-E", "64", "-i"),
    ) -> None:
        if not _REMOTE_HOST_PATTERN.fullmatch(remote_host):
            raise ValueError(f"Unsafe Sunway remote host {remote_host!r}")
        if not remote_root.startswith("/") or any(character.isspace() for character in remote_root):
            raise ValueError("Sunway remote job root must be an absolute path without whitespace")
        if account is not None and not _SLURM_NAME_PATTERN.fullmatch(account):
            raise ValueError(f"Unsafe Slurm account {account!r}")
        self.remote_host = remote_host
        self.remote_root = remote_root.rstrip("/")
        self.partition = partition
        self.account = account
        self.sbatch = str(sbatch)
        self.squeue = str(squeue)
        self.ssh = str(ssh)
        self.scp = str(scp)
        self.launcher = launcher

    def submit(self, package_dir: Path, *, executable: str, arguments: tuple[str, ...] = ()) -> SunwaySubmittedJob:
        package_dir = Path(package_dir).resolve()
        executable_path = package_dir / executable
        if not executable_path.is_file():
            raise FileNotFoundError(f"Missing Sunway executable {executable_path}")

        script_path = package_dir / f"submit_{executable}.sh"
        stdout_pattern = package_dir / "slurm-%j.out"
        stderr_pattern = package_dir / "slurm-%j.err"
        script_path.write_text(self._render_script(package_dir, executable, arguments), encoding="utf-8")
        script_path.chmod(0o755)

        command = [
            self.sbatch,
            "--parsable",
            "--partition",
            self.partition,
        ]
        if self.account is not None:
            command.extend(["--account", self.account])
        command.extend(
            [
                "--job-name",
                f"tl-sw-{executable}",
                "--output",
                str(stdout_pattern),
                "--error",
                str(stderr_pattern),
                str(script_path),
            ]
        )
        try:
            completed = subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as error:
            detail = (error.stderr or error.stdout or str(error)).strip()
            raise RuntimeError(f"Dell Slurm rejected the Sunway relay job: {detail}") from error
        job_id = completed.stdout.strip().split(";", 1)[0]
        if not job_id:
            raise RuntimeError("Dell sbatch did not return a job id")
        return SunwaySubmittedJob(job_id, script_path, stdout_pattern, stderr_pattern)

    def status(self, scheduler_job_id: str) -> str:
        completed = subprocess.run(
            [self.squeue, "--noheader", "--jobs", str(scheduler_job_id), "--format", "%T"],
            check=True,
            capture_output=True,
            text=True,
        )
        state = completed.stdout.strip().splitlines()
        return state[0].strip() if state else "COMPLETED_OR_UNKNOWN"

    def _render_script(self, package_dir: Path, executable: str, arguments: tuple[str, ...]) -> str:
        remote_dir = f"{self.remote_root}/${{SLURM_JOB_ID}}"
        launch = " ".join(shlex.quote(value) for value in (*self.launcher, f"./{executable}", *arguments))
        local_source = shlex.quote(str(package_dir))
        ssh = shlex.quote(self.ssh)
        scp = shlex.quote(self.scp)
        host = shlex.quote(self.remote_host)
        return "\n".join(
            [
                "#!/bin/bash",
                "set -uo pipefail",
                f'REMOTE_DIR="{remote_dir}"',
                "rc=0",
                f"{ssh} {host} \"mkdir -p '$REMOTE_DIR'\" &&",
                f'{scp} -r {local_source} {host}:"$REMOTE_DIR/package" &&',
                f"{ssh} {host} \"cd '$REMOTE_DIR/package' && chmod +x ./{shlex.quote(executable)} && {launch}\" || rc=$?",
                f'printf "%s\\n" "$rc" > {shlex.quote(str(package_dir))}/slurm-${{SLURM_JOB_ID}}.rc',
                'exit "$rc"',
                "",
            ]
        )
