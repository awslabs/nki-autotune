"""Command-line support for durable nkigym workflows."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from self_evolve.run import create_run
from self_evolve.types import RunStatus
from self_evolve.workflow import accept_run, check_run, status_run, tune_run, validate_run
from self_evolve.workloads import load_workload, workload_names

_RUN_COMMANDS: dict[str, Callable[[Path], RunStatus]] = {
    "accept": accept_run,
    "check": check_run,
    "status": status_run,
    "tune": tune_run,
    "validate": validate_run,
}


def _parser() -> argparse.ArgumentParser:
    """Build the support-command parser."""
    parser = argparse.ArgumentParser(prog="develop.py", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    start_parser = subparsers.add_parser("start", help="create a run from one kernel-library workload")
    start_parser.add_argument("workload", choices=workload_names())
    start_parser.add_argument("--shape", required=True, help="shape key registered for the workload")
    start_parser.add_argument("--host", required=True, help="SSH destination for the Trn2 profile worker")
    start_parser.add_argument("--rounds", required=True, type=int, help="number of refinement cycles")
    start_parser.add_argument("--artifact-root", type=Path)
    start_parser.add_argument("--base-revision", default="HEAD")

    for command in _RUN_COMMANDS:
        command_parser = subparsers.add_parser(command, help=f"{command} one durable run")
        command_parser.add_argument("run_directory", type=Path)
    return parser


def _dispatch(arguments: argparse.Namespace) -> RunStatus:
    """Execute one parsed command."""
    command = str(arguments.command)
    if command == "start":
        workload = load_workload(str(arguments.workload), str(arguments.shape))
        status = create_run(
            workload,
            str(arguments.host),
            int(arguments.rounds),
            artifact_root=arguments.artifact_root,
            base_revision=str(arguments.base_revision),
        )
    else:
        handler = _RUN_COMMANDS[command]
        status = handler(Path(arguments.run_directory))
    return status


def main(argv: Sequence[str] | None = None) -> int:
    """Run one support command and print machine-readable status."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    exit_code = 0
    try:
        status = _dispatch(_parser().parse_args(argv))
        print(json.dumps(status.as_dict(), indent=2, sort_keys=True))
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        exit_code = 2
    return exit_code


__all__ = ["main"]
