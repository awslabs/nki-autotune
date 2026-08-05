"""Fixed command-line worker installed once on an SSH Trn2 host."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from nkigym.profile._runner import run_profile
from nkigym.profile.protocol import parse_request, result_payload


def _parse_args() -> argparse.Namespace:
    """Parse fixed kernel and artifact paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _main() -> None:
    """Read one request from stdin, profile its kernel, and write artifacts."""
    args = _parse_args()
    kernel_path = Path(args.kernel).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    if not kernel_path.is_file():
        raise FileNotFoundError(f"kernel source not found: {kernel_path}")
    request = parse_request(json.load(sys.stdin))
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True)
    result = run_profile(
        kernel_path=kernel_path, func_name=request.func_name, config=request.config, output_dir=output_dir
    )
    (output_dir / "result.json").write_text(json.dumps(result_payload(result), indent=2) + "\n", encoding="utf-8")
    if result.profiler_summary is not None:
        (output_dir / "profile_summary.json").write_text(
            json.dumps(result.profiler_summary, indent=2) + "\n", encoding="utf-8"
        )
    status = "success" if result.error is None else "kernel failure"
    print(f"nkigym profile worker: {request.func_name}: {status}", flush=True)


if __name__ == "__main__":
    _main()
