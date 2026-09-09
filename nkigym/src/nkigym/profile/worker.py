"""Fixed command-line worker installed once on an SSH Trn2 host."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

from nkigym.profile._benchmark import available_logical_cores
from nkigym.profile._runner import _resolve_dtype, run_profile
from nkigym.profile.protocol import parse_request
from nkigym.profile.types import ProfileConfig


def _parse_args() -> argparse.Namespace:
    """Parse fixed input and artifact paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--kernel")
    sources.add_argument("--input")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _load_exact_inputs(input_dir: Path, config: ProfileConfig) -> dict[str, np.ndarray]:
    """Load exact inputs uploaded in deterministic specification order."""
    inputs: dict[str, np.ndarray] = {}
    for index, (name, (shape, dtype_name)) in enumerate(config.input_specs.items()):
        path = input_dir / "inputs" / f"input_{index:03d}.bin"
        inputs[name] = np.fromfile(path, dtype=_resolve_dtype(dtype_name)).reshape(shape)
    return inputs


def _main() -> None:
    """Read one request, execute through the unified backend, and write artifacts."""
    args = _parse_args()
    input_path = Path(args.input or args.kernel).expanduser().resolve()
    input_dir = input_path if args.input else input_path.parent
    kernel_path = input_dir / "kernel.py" if args.input else input_path
    output_dir = Path(args.output).expanduser().resolve()
    if not kernel_path.is_file():
        raise FileNotFoundError(f"kernel source not found: {kernel_path}")
    request = parse_request(json.load(sys.stdin))
    inputs = _load_exact_inputs(input_dir, request.config) if args.input else None
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True)
    result = run_profile(
        kernel_path=kernel_path,
        func_name=request.func_name,
        config=request.config,
        output_dir=output_dir,
        visible_core=available_logical_cores(request.config.lnc)[0],
        compiler_jobs=None,
        inputs=inputs,
        capture_outputs=inputs is not None,
    )
    (output_dir / "result.json").write_text(json.dumps(vars(result), indent=2) + "\n", encoding="utf-8")
    if result.profiler_summary is not None:
        (output_dir / "profile_summary.json").write_text(
            json.dumps(result.profiler_summary, indent=2) + "\n", encoding="utf-8"
        )
    print(f"nkigym worker: {'ok' if result.error is None else 'failed'}", flush=True)


if __name__ == "__main__":
    _main()
