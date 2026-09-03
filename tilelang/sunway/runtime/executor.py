"""Direct deployment and execution of generated kernels on an SW9A host."""

from __future__ import annotations

import re
import shlex
import subprocess
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


_REMOTE_HOST_PATTERN = re.compile(r"^[A-Za-z0-9_.@-]+$")
_REMOTE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
_REMOTE_ROOT_PATTERN = re.compile(r"^/[A-Za-z0-9_./-]+$")
_ENV_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True, slots=True)
class SunwayDeployment:
    """Location of one package copied to the SW9A host."""

    remote_host: str
    remote_directory: str


@dataclass(frozen=True, slots=True)
class SunwayExecutionResult:
    """Captured result of one synchronous remote kernel launch."""

    deployment: SunwayDeployment
    returncode: int
    stdout: str
    stderr: str

    @property
    def succeeded(self) -> bool:
        return self.returncode == 0


class SunwaySSHExecutor:
    """Copy a package to SW9A and run a standalone or Python AOT image.

    This executor intentionally has no scheduler semantics: the caller owns the
    SW9A allocation and the method returns only after the remote command exits.
    """

    def __init__(
        self,
        *,
        remote_host: str,
        remote_root: str = "/tmp/tilelang-runs",
        ssh: str | Path = "ssh",
        scp: str | Path = "scp",
        launcher: tuple[str, ...] = ("swrun", "-E", "64", "-i"),
        swpython_home: str = "/usr/sw/swpython",
        swpython_setenv: str = "/usr/sw/swpython/setenv",
    ) -> None:
        if not _REMOTE_HOST_PATTERN.fullmatch(remote_host):
            raise ValueError(f"Unsafe Sunway remote host {remote_host!r}")
        if not _REMOTE_ROOT_PATTERN.fullmatch(remote_root):
            raise ValueError(f"Unsafe Sunway remote root {remote_root!r}")
        if ".." in PurePosixPath(remote_root).parts:
            raise ValueError("Sunway remote root must not contain '..'")
        if not launcher:
            raise ValueError("Sunway launcher must not be empty")
        if not _REMOTE_ROOT_PATTERN.fullmatch(swpython_home):
            raise ValueError(f"Unsafe SWPython home {swpython_home!r}")
        if not _REMOTE_ROOT_PATTERN.fullmatch(swpython_setenv):
            raise ValueError(f"Unsafe SWPython environment script {swpython_setenv!r}")
        self.remote_host = remote_host
        self.remote_root = remote_root.rstrip("/")
        self.ssh = str(ssh)
        self.scp = str(scp)
        self.launcher = launcher
        self.swpython_home = swpython_home
        self.swpython_setenv = swpython_setenv

    def deploy(self, package_dir: Path, *, deployment_id: str | None = None) -> SunwayDeployment:
        """Copy a complete artifact package into an isolated remote directory."""

        package_dir = Path(package_dir).resolve()
        if not package_dir.is_dir():
            raise FileNotFoundError(f"Missing Sunway package directory {package_dir}")
        deployment_id = deployment_id or uuid.uuid4().hex
        if not _REMOTE_NAME_PATTERN.fullmatch(deployment_id):
            raise ValueError(f"Unsafe Sunway deployment id {deployment_id!r}")

        remote_parent = f"{self.remote_root}/{deployment_id}"
        remote_directory = f"{remote_parent}/package"
        self._run_checked(
            [self.ssh, self.remote_host, f"mkdir -p {shlex.quote(remote_parent)}"],
            action="create the remote deployment directory",
        )
        self._run_checked(
            [self.scp, "-r", str(package_dir), f"{self.remote_host}:{remote_directory}"],
            action="copy the Sunway package",
        )
        return SunwayDeployment(self.remote_host, remote_directory)

    def run(
        self,
        deployment: SunwayDeployment,
        *,
        executable: str,
        arguments: tuple[str, ...] = (),
    ) -> SunwayExecutionResult:
        """Run an executable already present in a deployed package."""

        if deployment.remote_host != self.remote_host:
            raise ValueError("Deployment belongs to a different Sunway host")
        if not _REMOTE_NAME_PATTERN.fullmatch(executable):
            raise ValueError(f"Unsafe Sunway executable name {executable!r}")

        executable_path = f"./{executable}"
        launch = " ".join(
            shlex.quote(value) for value in (*self.launcher, executable_path, *arguments)
        )
        remote_command = " && ".join(
            [
                f"cd {shlex.quote(deployment.remote_directory)}",
                f"chmod +x {shlex.quote(executable_path)}",
                launch,
            ]
        )
        completed = subprocess.run(
            [self.ssh, self.remote_host, remote_command],
            check=False,
            capture_output=True,
            text=True,
        )
        return SunwayExecutionResult(
            deployment=deployment,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

    def deploy_and_run(
        self,
        package_dir: Path,
        *,
        executable: str,
        arguments: tuple[str, ...] = (),
        deployment_id: str | None = None,
    ) -> SunwayExecutionResult:
        """Deploy a package and synchronously launch its executable."""

        executable_path = Path(package_dir).resolve() / executable
        if not executable_path.is_file():
            raise FileNotFoundError(f"Missing Sunway executable {executable_path}")
        deployment = self.deploy(package_dir, deployment_id=deployment_id)
        return self.run(deployment, executable=executable, arguments=arguments)

    def run_python(
        self,
        deployment: SunwayDeployment,
        *,
        launcher: str,
        script: str,
        arguments: tuple[str, ...] = (),
        environment: Mapping[str, str] | None = None,
    ) -> SunwayExecutionResult:
        """Run Python inside the startup-linked dynamic CPE image.

        This path intentionally does not invoke ``swrun``. The ``-mdynamic``
        Python launcher owns the stask allocation and initializes CRTS before
        importing SWPyTorch or the generated torch.ops registration library.
        """

        if deployment.remote_host != self.remote_host:
            raise ValueError("Deployment belongs to a different Sunway host")
        if not _REMOTE_NAME_PATTERN.fullmatch(launcher):
            raise ValueError(f"Unsafe Sunway Python launcher name {launcher!r}")
        if not _REMOTE_NAME_PATTERN.fullmatch(script):
            raise ValueError(f"Unsafe Sunway Python script name {script!r}")

        runtime_environment = {
            "PYTHONHOME": self.swpython_home,
            "LD_BIND_NOW": "1",
            "SPAWN_MODE": "auto",
            "STASK_SEG_DATA": "64",
            "STASK_SEG_TEXT": "64",
            "STASK_SEG_PRIV": "2",
        }
        if environment is not None:
            for name, value in environment.items():
                if not _ENV_NAME_PATTERN.fullmatch(name):
                    raise ValueError(f"Unsafe Sunway environment name {name!r}")
                runtime_environment[name] = str(value)

        executable_path = f"./{launcher}"
        script_path = f"./{script}"
        exports = " ".join(
            f"{name}={shlex.quote(value)}" for name, value in runtime_environment.items()
        )
        launch = " ".join(
            shlex.quote(value) for value in (executable_path, script_path, *arguments)
        )
        remote_command = " && ".join(
            [
                f"cd {shlex.quote(deployment.remote_directory)}",
                f". {shlex.quote(self.swpython_setenv)}",
                f"export {exports}",
                f"chmod +x {shlex.quote(executable_path)}",
                launch,
            ]
        )
        completed = subprocess.run(
            [self.ssh, self.remote_host, remote_command],
            check=False,
            capture_output=True,
            text=True,
        )
        return SunwayExecutionResult(
            deployment=deployment,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

    def deploy_and_run_python(
        self,
        package_dir: Path,
        *,
        launcher: str,
        script: str,
        arguments: tuple[str, ...] = (),
        environment: Mapping[str, str] | None = None,
        deployment_id: str | None = None,
    ) -> SunwayExecutionResult:
        """Deploy an AOT Python bundle and execute one packaged script."""

        package_dir = Path(package_dir).resolve()
        for name in (launcher, script):
            path = package_dir / name
            if not path.is_file():
                raise FileNotFoundError(f"Missing Sunway Python bundle file {path}")
        deployment = self.deploy(package_dir, deployment_id=deployment_id)
        return self.run_python(
            deployment,
            launcher=launcher,
            script=script,
            arguments=arguments,
            environment=environment,
        )

    @staticmethod
    def _run_checked(command: list[str], *, action: str) -> None:
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as error:
            detail = (error.stderr or error.stdout or str(error)).strip()
            raise RuntimeError(f"Failed to {action}: {detail}") from error
