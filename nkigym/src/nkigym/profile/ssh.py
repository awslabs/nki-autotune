"""Dedicated SSH transport for one installed Trn2 profile worker."""

from __future__ import annotations

import os
import secrets
import shlex
import shutil
import subprocess
import time
from pathlib import Path

_SSH_OPTIONS = (
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=15",
    "-o",
    "StrictHostKeyChecking=no",
    "-o",
    "ControlMaster=auto",
    "-o",
    "ControlPersist=30",
    "-o",
    "ControlPath=~/.ssh/nkigym-%C",
)
_REMOTE_PYTHON = '"$HOME"/venvs/kernel-env/bin/python'
_REMOTE_RUN_ROOT = ".cache/nkigym-profile/runs"


class SSHTransportError(RuntimeError):
    """Infrastructure failure with the complete transport log."""

    def __init__(self, message: str, log: str) -> None:
        """Store the concise failure and all completed command output."""
        super().__init__(message)
        self.log = log


class _CommandRunner:
    """Run transport subprocesses against one shared deadline."""

    def __init__(self, timeout_s: int) -> None:
        """Start one bounded transport operation."""
        self.deadline = time.monotonic() + timeout_s
        self.lines: list[str] = []

    @property
    def log(self) -> str:
        """Return all stage output collected so far."""
        return "".join(self.lines)

    def run(self, stage: str, command: list[str], input_text: str | None) -> None:
        """Run one stage and fail with its output when it is unsuccessful."""
        remaining_s = self.deadline - time.monotonic()
        if remaining_s <= 0:
            raise SSHTransportError(f"{stage} exceeded the SSH profile timeout", self.log)
        self.lines.append(f"==> {stage}\n")
        try:
            completed = subprocess.run(
                command, input=input_text, text=True, capture_output=True, timeout=remaining_s, check=False
            )
        except subprocess.TimeoutExpired as error:
            self._record_timeout_output(error)
            raise SSHTransportError(f"{stage} exceeded the SSH profile timeout", self.log) from error
        self.lines.extend((completed.stdout, completed.stderr))
        if completed.returncode != 0:
            raise SSHTransportError(f"{stage} failed with exit {completed.returncode}", self.log)

    def cleanup(self, host: str, remote_run: str) -> None:
        """Best-effort removal of one remote scratch directory."""
        command = ["ssh", *_SSH_OPTIONS, host, f'rm -rf "$HOME"/{remote_run}']
        try:
            completed = subprocess.run(command, text=True, capture_output=True, timeout=15, check=False)
            if completed.returncode != 0:
                self.lines.append("==> remote cleanup failed\n")
                self.lines.extend((completed.stdout, completed.stderr))
        except subprocess.TimeoutExpired:
            self.lines.append("==> remote cleanup timed out\n")

    def _record_timeout_output(self, error: subprocess.TimeoutExpired) -> None:
        """Preserve partial output attached to a timeout."""
        stdout = error.stdout.decode() if isinstance(error.stdout, bytes) else error.stdout
        stderr = error.stderr.decode() if isinstance(error.stderr, bytes) else error.stderr
        if stdout:
            self.lines.append(stdout)
        if stderr:
            self.lines.append(stderr)


def profile_over_ssh(host: str, kernel_path: Path, request_path: Path, output_dir: Path, timeout_s: int) -> str:
    """Profile one kernel remotely and return the complete transport log."""
    _validate_host(host)
    _require_command("ssh")
    _require_command("rsync")
    if timeout_s <= 0:
        raise ValueError("SSH profile timeout must be positive")
    if not kernel_path.is_file():
        raise FileNotFoundError(f"kernel source not found: {kernel_path}")
    if not request_path.is_file():
        raise FileNotFoundError(f"profile request not found: {request_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    request_text = request_path.read_text(encoding="utf-8")
    run_id = f"{time.time_ns()}-{os.getpid()}-{secrets.token_hex(4)}"
    remote_run = f"{_REMOTE_RUN_ROOT}/{run_id}"
    remote_output = f"{remote_run}/output"
    rsync_shell = shlex.join(("ssh", *_SSH_OPTIONS))
    runner = _CommandRunner(timeout_s)
    failure: SSHTransportError | None = None
    try:
        runner.run(
            "Checking installed profile worker",
            [
                "ssh",
                *_SSH_OPTIONS,
                host,
                (
                    f"test -x {_REMOTE_PYTHON} && "
                    f"{_REMOTE_PYTHON} -c 'import nkigym.profile.worker' && "
                    f'mkdir -p "$HOME"/{remote_run}'
                ),
            ],
            None,
        )
        runner.run(
            "Uploading kernel.py",
            ["rsync", "-az", "-e", rsync_shell, str(kernel_path), f"{host}:{remote_run}/kernel.py"],
            None,
        )
        runner.run(
            "Profiling kernel",
            [
                "ssh",
                *_SSH_OPTIONS,
                host,
                (
                    f"{_REMOTE_PYTHON} -m nkigym.profile.worker "
                    f'--kernel "$HOME"/{remote_run}/kernel.py '
                    f'--output "$HOME"/{remote_output}'
                ),
            ],
            request_text,
        )
        runner.run(
            "Downloading profile artifacts",
            ["rsync", "-az", "-e", rsync_shell, f"{host}:{remote_output}/", f"{output_dir}/"],
            None,
        )
    except SSHTransportError as error:
        failure = error
    finally:
        runner.cleanup(host, remote_run)
    if failure is not None:
        raise SSHTransportError(str(failure), runner.log) from failure
    return runner.log


def profile_batch_over_ssh(host: str, input_dir: Path, request_path: Path, output_dir: Path, timeout_s: int) -> str:
    """Profile a directory of labeled kernels in one remote batch."""
    _validate_host(host)
    _require_command("ssh")
    _require_command("rsync")
    if timeout_s <= 0:
        raise ValueError("SSH profile timeout must be positive")
    if not input_dir.is_dir():
        raise FileNotFoundError(f"batch profile input directory not found: {input_dir}")
    if not request_path.is_file():
        raise FileNotFoundError(f"batch profile request not found: {request_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    request_text = request_path.read_text(encoding="utf-8")
    run_id = f"{time.time_ns()}-{os.getpid()}-{secrets.token_hex(4)}"
    remote_run = f"{_REMOTE_RUN_ROOT}/{run_id}"
    remote_input = f"{remote_run}/input"
    remote_output = f"{remote_run}/output"
    rsync_shell = shlex.join(("ssh", *_SSH_OPTIONS))
    runner = _CommandRunner(timeout_s)
    failure: SSHTransportError | None = None
    try:
        runner.run(
            "Checking installed batch profile worker",
            [
                "ssh",
                *_SSH_OPTIONS,
                host,
                (
                    f"test -x {_REMOTE_PYTHON} && "
                    f"{_REMOTE_PYTHON} -c 'import nkigym.profile.batch_worker' && "
                    f'mkdir -p "$HOME"/{remote_input}'
                ),
            ],
            None,
        )
        runner.run(
            "Uploading batch kernels",
            ["rsync", "-az", "-e", rsync_shell, f"{input_dir}/", f"{host}:{remote_input}/"],
            None,
        )
        runner.run(
            "Profiling kernel batch",
            [
                "ssh",
                *_SSH_OPTIONS,
                host,
                (
                    f"{_REMOTE_PYTHON} -m nkigym.profile.batch_worker "
                    f'--input "$HOME"/{remote_input} '
                    f'--output "$HOME"/{remote_output}'
                ),
            ],
            request_text,
        )
        runner.run(
            "Downloading batch profile artifacts",
            ["rsync", "-az", "-e", rsync_shell, f"{host}:{remote_output}/", f"{output_dir}/"],
            None,
        )
    except SSHTransportError as error:
        failure = error
    finally:
        runner.cleanup(host, remote_run)
    if failure is not None:
        raise SSHTransportError(str(failure), runner.log) from failure
    return runner.log


def _validate_host(host: str) -> None:
    """Reject empty or option-shaped SSH destinations."""
    if not host or host.startswith("-") or any(character.isspace() for character in host):
        raise ValueError(f"invalid SSH host {host!r}")


def _require_command(command: str) -> None:
    """Fail before transport when one local executable is unavailable."""
    if shutil.which(command) is None:
        raise FileNotFoundError(f"{command} is not on PATH")


__all__ = ["SSHTransportError", "profile_batch_over_ssh", "profile_over_ssh"]
