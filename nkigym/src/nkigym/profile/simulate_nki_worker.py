"""Standalone remote worker for batched fp32 NKI simulation."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import pickle
import re
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import cast

import nki
import numpy as np

_FP_DTYPES_NON_FP32 = (
    "bfloat16",
    "float16",
    "float8_e4m3",
    "float8_e4m3fn",
    "float8_e4m3fn_x4",
    "float8_e5m2",
    "float8_e5m2_x4",
    "float4_e2m1fn_x4",
    "tfloat32",
)
ArrayResult = np.ndarray | tuple[np.ndarray, ...]
_SerializedCase = tuple[int, str, str, str, dict[str, np.ndarray], ArrayResult]
_FailurePayload = dict[str, int | str]
_WORKER_CASES: list[_SerializedCase] = []
_WORKER_ATOL = 0.0
_WORKER_RTOL = 0.0


def _fp32_source(source: str) -> str:
    """Rewrite reduced-precision NKI language dtypes to fp32."""
    rewritten = source
    for dtype in _FP_DTYPES_NON_FP32:
        rewritten = re.sub(rf"\bnl\.{re.escape(dtype)}\b", "nl.float32", rewritten)
    return rewritten


def _simulate_source_fp32(source: str, func_name: str, inputs: dict[str, np.ndarray], source_name: str) -> ArrayResult:
    """Execute one standalone rendered kernel through the fp32 simulator."""
    namespace: dict = {}
    code = compile(_fp32_source(source), source_name, "exec")
    exec(code, namespace)  # noqa: S102
    kernel = namespace.get(func_name)
    if kernel is None:
        raise AttributeError(f"rendered kernel has no function {func_name!r}")
    simulated = nki.simulate(kernel)
    cast_inputs = {
        name: value.astype(np.float32) if value.dtype.kind == "f" else value for name, value in inputs.items()
    }
    result = simulated(**cast_inputs)
    if isinstance(result, tuple):
        output: ArrayResult = tuple(np.asarray(value) for value in result)
    else:
        output = np.asarray(result)
    return output


def _simulate_case(position: int) -> _FailurePayload | None:
    """Simulate one globally initialized case and return its failure."""
    case_index, label, source, func_name, inputs, expected = _WORKER_CASES[position]
    failure = None
    try:
        actual = _simulate_source_fp32(source, func_name, inputs, f"<batch-case-{case_index}>")
        actual_outputs = actual if isinstance(actual, tuple) else (actual,)
        expected_outputs = expected if isinstance(expected, tuple) else (expected,)
        if len(actual_outputs) != len(expected_outputs):
            raise AssertionError(f"{label}: returned {len(actual_outputs)} outputs, expected {len(expected_outputs)}")
        for actual_output, expected_output in zip(actual_outputs, expected_outputs, strict=True):
            np.testing.assert_allclose(
                actual_output, expected_output, atol=_WORKER_ATOL, rtol=_WORKER_RTOL, err_msg=label
            )
    except Exception as error:
        failure = {
            "case_index": case_index,
            "label": label,
            "exception_type": type(error).__name__,
            "traceback": traceback.format_exc(),
        }
    return failure


def _read_worker_request(request_path: Path) -> tuple[list[_SerializedCase], float, float, int]:
    """Load one trusted controller-generated worker request."""
    with request_path.open("rb") as request_file:
        payload = cast(tuple[list[_SerializedCase], float, float, int], pickle.load(request_file))
    if not isinstance(payload, tuple) or len(payload) != 4:
        raise ValueError("malformed batch simulation request")
    cases, atol, rtol, worker_count = payload
    if not isinstance(worker_count, int) or isinstance(worker_count, bool) or worker_count <= 0:
        raise ValueError("batch simulation worker count must be positive")
    return cases, atol, rtol, worker_count


def _worker_result(
    cases: list[_SerializedCase], atol: float, rtol: float, worker_count: int
) -> dict[str, int | _FailurePayload | None]:
    """Simulate one assigned partition and return compact result metadata."""
    global _WORKER_ATOL, _WORKER_CASES, _WORKER_RTOL
    _WORKER_CASES = cases
    _WORKER_ATOL = atol
    _WORKER_RTOL = rtol
    failures: list[_FailurePayload | None] = []
    if cases:
        active_workers = min(len(cases), worker_count, multiprocessing.cpu_count())
        if active_workers == 1:
            failures = [_simulate_case(position) for position in range(len(cases))]
        else:
            context = multiprocessing.get_context("fork")
            with ProcessPoolExecutor(max_workers=active_workers, mp_context=context) as executor:
                failures = list(executor.map(_simulate_case, range(len(cases)), chunksize=1))
    failure_position = next((position for position, failure in enumerate(failures) if failure is not None), None)
    failure = None if failure_position is None else failures[failure_position]
    completed = len(cases) if failure_position is None else failure_position
    return {"assigned": len(cases), "completed": completed, "failure": failure}


def _run_worker(request_path: Path, result_path: Path) -> None:
    """Run one remote request and atomically write its result metadata."""
    cases, atol, rtol, worker_count = _read_worker_request(request_path)
    result = _worker_result(cases, atol, rtol, worker_count)
    temporary_path = result_path.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(result_path)


def _parse_args() -> argparse.Namespace:
    """Parse the private remote-worker command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("request", type=Path)
    parser.add_argument("result", type=Path)
    return parser.parse_args()


def _main() -> int:
    """Run the private remote-worker entry point."""
    args = _parse_args()
    if not args.worker:
        raise ValueError("simulate_nki_worker.py is only executable in worker mode")
    _run_worker(args.request, args.result)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
