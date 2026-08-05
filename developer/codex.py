"""Noninteractive Codex CLI adapter with minimal metadata parsing."""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import time
from pathlib import Path

from developer.types import CodexInvocationResult

_TERMINATION_GRACE_SECONDS = 5
_LOGGER = logging.getLogger(__name__)


class CodexRunner:
    """Run initial and resumed Codex turns in one candidate worktree."""

    def __init__(self, executable: str, timeout_seconds: int) -> None:
        """Store process controls for all turns in a workflow."""
        self.executable = executable
        self.timeout_seconds = timeout_seconds

    def run(self, worktree: Path, prompt: str, attempt_directory: Path, thread_id: str | None) -> CodexInvocationResult:
        """Run one Codex turn and persist its raw output streams."""
        event_log = attempt_directory / "codex-events.jsonl"
        stderr_log = attempt_directory / "codex-stderr.log"
        final_message = attempt_directory / "final-message.md"
        final_message.touch()
        command = self._command(worktree, final_message, thread_id)
        mode = "new thread" if thread_id is None else f"resume {thread_id}"
        _LOGGER.info("editing | started | %s | artifacts=%s", mode, attempt_directory)
        started = time.monotonic()
        timed_out = False
        exit_code = 127
        with event_log.open("w", encoding="utf-8") as events, stderr_log.open("w", encoding="utf-8") as errors:
            try:
                process = subprocess.Popen(
                    command,
                    cwd=worktree,
                    stdin=subprocess.PIPE,
                    stdout=events,
                    stderr=errors,
                    text=True,
                    start_new_session=True,
                )
            except OSError as error:
                errors.write(f"failed to start Codex: {error}\n")
            else:
                try:
                    process.communicate(prompt, timeout=self.timeout_seconds)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    self._terminate(process)
                except KeyboardInterrupt:
                    self._terminate(process)
                    raise
                if process.returncode is None:
                    raise RuntimeError("Codex process completed without a return code")
                exit_code = process.returncode
        if timed_out:
            with stderr_log.open("a", encoding="utf-8") as errors:
                errors.write(f"Codex exceeded timeout of {self.timeout_seconds} seconds\n")
        duration = time.monotonic() - started
        observed_thread_id = extract_thread_id(event_log)
        result = CodexInvocationResult(
            command=tuple(command),
            exit_code=exit_code,
            timed_out=timed_out,
            duration_seconds=duration,
            thread_id=observed_thread_id,
            event_log=event_log,
            stderr_log=stderr_log,
            final_message=final_message,
        )
        status = "passed" if result.passed else "failed"
        observed = result.thread_id if result.thread_id is not None else "unavailable"
        _LOGGER.info(
            "editing | %s | thread=%s | duration=%.1fs | events=%s",
            status,
            observed,
            result.duration_seconds,
            result.event_log,
        )
        return result

    def _command(self, worktree: Path, final_message: Path, thread_id: str | None) -> list[str]:
        """Build an initial or resumed noninteractive Codex command."""
        if thread_id is None:
            command = [
                self.executable,
                "exec",
                "--json",
                "--sandbox",
                "workspace-write",
                "--cd",
                str(worktree),
                "--output-last-message",
                str(final_message),
                "-",
            ]
        else:
            command = [
                self.executable,
                "exec",
                "resume",
                "--json",
                "--output-last-message",
                str(final_message),
                thread_id,
                "-",
            ]
        return command

    def _terminate(self, process: subprocess.Popen[str]) -> None:
        """Terminate a timed-out Codex process group and wait for shutdown."""
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            process.wait()
        else:
            try:
                process.communicate(timeout=_TERMINATION_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    process.wait()
                else:
                    process.communicate()


def extract_thread_id(event_log: Path) -> str | None:
    """Extract only the thread identifier from Codex JSONL operational events."""
    thread_id: str | None = None
    with event_log.open(encoding="utf-8") as events:
        for line in events:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict) and event.get("type") == "thread.started":
                candidate = event.get("thread_id")
                if isinstance(candidate, str) and candidate:
                    if thread_id is not None and candidate != thread_id:
                        raise RuntimeError(f"Codex event log contains conflicting thread IDs: {thread_id}, {candidate}")
                    thread_id = candidate
    return thread_id
