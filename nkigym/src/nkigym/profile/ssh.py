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
_TRANSPORT_LOG_FILE = "transport.log"


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

    @property
    def remaining_s(self) -> float:
        """Return seconds remaining before the shared deadline."""
        return self.deadline - time.monotonic()

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
            self.record_timeout_output(error)
            raise SSHTransportError(f"{stage} exceeded the SSH profile timeout", self.log) from error
        self.lines.extend((completed.stdout, completed.stderr))
        if completed.returncode != 0:
            raise SSHTransportError(f"{stage} failed with exit {completed.returncode}", self.log)

    def cleanup(self, host: str, remote_run: str, terminate_process_group: bool) -> None:
        """Best-effort termination and removal of one remote scratch directory."""
        remote_directory = f'"$HOME"/{remote_run}'
        terminate = ""
        if terminate_process_group:
            terminate = (
                'if test -s "$run/worker.pgid"; then pgid=$(cat "$run/worker.pgid"); '
                'case "$pgid" in ""|*[!0-9]*) exit 2 ;; esac; test "$pgid" -gt 1 || exit 2; '
                'kill -TERM -- "-$pgid" 2>/dev/null || true; attempts=0; '
                'while kill -0 -- "-$pgid" 2>/dev/null && test "$attempts" -lt 20; do '
                "sleep 0.1; attempts=$((attempts + 1)); done; "
                'kill -KILL -- "-$pgid" 2>/dev/null || true; fi; '
            )
        command = ["ssh", *_SSH_OPTIONS, host, f"run={remote_directory}; {terminate}" 'rm -rf "$run"']
        try:
            completed = subprocess.run(command, text=True, capture_output=True, timeout=15, check=False)
            if completed.returncode != 0:
                self.lines.append("==> remote cleanup failed\n")
                self.lines.extend((completed.stdout, completed.stderr))
        except subprocess.TimeoutExpired:
            self.lines.append("==> remote cleanup timed out\n")

    def record_timeout_output(self, error: subprocess.TimeoutExpired) -> None:
        """Preserve partial output attached to a timeout."""
        for output in (error.stdout, error.stderr):
            text = output.decode() if isinstance(output, bytes) else output
            if text:
                self.lines.append(text)


def profile_over_ssh(host: str, input_path: Path, request_path: Path, output_dir: Path, timeout_s: int) -> str:
    """Execute one kernel request remotely and return the complete transport log."""
    if not input_path.is_file() and not input_path.is_dir():
        raise FileNotFoundError(f"kernel request input not found: {input_path}")
    return _profile_over_ssh(host, input_path, request_path, output_dir, timeout_s)


def _profile_over_ssh(host: str, input_path: Path, request_path: Path, output_dir: Path, timeout_s: int) -> str:
    """Run one kernel execution/profile transport."""
    _validate_host(host)
    _require_command("ssh")
    _require_command("rsync")
    if timeout_s <= 0:
        raise ValueError("SSH profile timeout must be positive")
    if not request_path.is_file():
        raise FileNotFoundError(f"profile request not found: {request_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    request_text = request_path.read_text(encoding="utf-8")
    remote_run = f"{_REMOTE_RUN_ROOT}/{time.time_ns()}-{os.getpid()}-{secrets.token_hex(4)}"
    directory_input = input_path.is_dir()
    remote_input = f"{remote_run}/input" if directory_input else f"{remote_run}/kernel.py"
    remote_output = f"{remote_run}/output"
    rsync_shell = shlex.join(("ssh", *_SSH_OPTIONS))
    runner = _CommandRunner(timeout_s)
    upload_source = f"{input_path}/" if directory_input else str(input_path)
    upload_target = f"{host}:{remote_input}/" if directory_input else f"{host}:{remote_input}"
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
                    f'mkdir -p "$HOME"/{remote_input if directory_input else remote_run}'
                ),
            ],
            None,
        )
        runner.run("Uploading kernel request", ["rsync", "-az", "-e", rsync_shell, upload_source, upload_target], None)
        runner.run(
            "Executing and profiling kernel",
            [
                "ssh",
                *_SSH_OPTIONS,
                host,
                (
                    f"{_REMOTE_PYTHON} -m nkigym.profile.worker "
                    f'{"--input" if directory_input else "--kernel"} "$HOME"/{remote_input} '
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
        raise SSHTransportError(str(error), runner.log) from error
    finally:
        runner.cleanup(host, remote_run, False)
        (output_dir / _TRANSPORT_LOG_FILE).write_text(runner.log, encoding="utf-8")
    return runner.log


def _validate_host(host: str) -> None:
    """Reject empty or option-shaped SSH destinations."""
    if not host or host.startswith("-") or any(character.isspace() for character in host):
        raise ValueError(f"invalid SSH host {host!r}")


def _require_command(command: str) -> None:
    """Fail before transport when one local executable is unavailable."""
    if shutil.which(command) is None:
        raise FileNotFoundError(f"{command} is not on PATH")
