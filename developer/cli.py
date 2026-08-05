"""Read durable records produced by the callable developer workflow."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    """Build the status-only command-line parser."""
    parser = argparse.ArgumentParser(prog="python -m developer")
    subparsers = parser.add_subparsers(dest="command", required=True)
    status_parser = subparsers.add_parser("status", help="show the durable record for one run")
    status_parser.add_argument("run_directory", type=Path)
    return parser


def _status(run_directory: Path) -> int:
    """Print one run record as formatted JSON."""
    record_path = run_directory.expanduser().resolve() / "run.json"
    decoded = json.loads(record_path.read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError(f"run record is not a JSON object: {record_path}")
    print(json.dumps(decoded, indent=2, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch the status command and return its process exit code."""
    arguments = _parser().parse_args(argv)
    exit_code = 0
    try:
        if arguments.command != "status":
            raise ValueError(f"unsupported command: {arguments.command}")
        exit_code = _status(Path(arguments.run_directory))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        exit_code = 2
    return exit_code


__all__ = ["main"]
