"""Parallel profile worker installed once on an SSH Trn2 host."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Protocol

from nkigym.profile._runner import run_profile
from nkigym.profile.protocol import batch_result_payload, parse_batch_request, result_payload
from nkigym.profile.types import BatchProfileJob, BatchProfileRequest, ProfileResult


class _CoreQueue(Protocol):
    """Minimal multiprocessing queue interface used by profile workers."""

    def get(self, block: bool = True, timeout: float | None = None) -> int:
        """Reserve one logical NeuronCore."""
        ...

    def put(self, item: int, block: bool = True, timeout: float | None = None) -> None:
        """Release one logical NeuronCore."""
        ...


def _parse_args() -> argparse.Namespace:
    """Parse fixed batch input and artifact paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _neuron_environment(lnc: int) -> dict[str, str]:
    """Return the environment used to discover logical NeuronCores."""
    environment = dict(os.environ)
    path_entries = (
        str(Path(sys.executable).parent),
        "/opt/aws/neuron/bin",
        *environment.get("PATH", "").split(os.pathsep),
    )
    environment["PATH"] = os.pathsep.join(dict.fromkeys(path_entries))
    environment["NEURON_LOGICAL_NC_CONFIG"] = str(lnc)
    return environment


def _logical_core_count(lnc: int) -> int:
    """Discover the number of logical NeuronCores available on this host."""
    completed = subprocess.run(
        ["neuron-ls", "--json-output"], text=True, capture_output=True, check=False, env=_neuron_environment(lnc)
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"neuron-ls failed with exit {completed.returncode}: {detail}")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"neuron-ls returned invalid JSON: {error}") from error
    if not isinstance(payload, list):
        raise RuntimeError("neuron-ls output must be a JSON array")
    counts: list[int] = []
    for device in payload:
        if isinstance(device, dict):
            count = device.get("nc_count")
            if isinstance(count, int) and not isinstance(count, bool):
                counts.append(count)
    core_count = sum(counts)
    if core_count <= 0:
        raise RuntimeError("neuron-ls reported no logical NeuronCores")
    return core_count


def _write_result(output_dir: Path, result: ProfileResult) -> None:
    """Write one profile result and its optional summary."""
    (output_dir / "result.json").write_text(json.dumps(result_payload(result), indent=2) + "\n", encoding="utf-8")
    if result.profiler_summary is not None:
        (output_dir / "profile_summary.json").write_text(
            json.dumps(result.profiler_summary, indent=2) + "\n", encoding="utf-8"
        )


def _profile_job(
    input_dir: Path, output_dir: Path, job: BatchProfileJob, core_queue: _CoreQueue, compiler_jobs: int
) -> str:
    """Compile and profile one job on a reserved logical NeuronCore."""
    visible_core = core_queue.get()
    try:
        job_output_dir = output_dir / job.label
        job_output_dir.mkdir(parents=True)
        result = run_profile(
            kernel_path=input_dir / job.label / "kernel.py",
            func_name=job.request.func_name,
            config=job.request.config,
            output_dir=job_output_dir,
            visible_core=visible_core,
            compiler_jobs=compiler_jobs,
        )
        _write_result(job_output_dir, result)
    finally:
        core_queue.put(visible_core)
    return job.label


def _active_workers(request: BatchProfileRequest) -> tuple[int, int]:
    """Resolve outer process count and per-compiler CPU parallelism."""
    cpu_count = os.cpu_count() or 1
    lnc = request.jobs[0].request.config.lnc
    workers = min(request.max_workers, len(request.jobs), _logical_core_count(lnc), cpu_count)
    compiler_jobs = max(1, math.ceil(cpu_count / workers))
    return workers, compiler_jobs


def _run_batch(input_dir: Path, output_dir: Path, request: BatchProfileRequest) -> tuple[float, int]:
    """Run all jobs with process isolation and distinct logical NeuronCores."""
    workers, compiler_jobs = _active_workers(request)
    context = multiprocessing.get_context("spawn")
    started = time.monotonic()
    with context.Manager() as manager:
        core_queue = manager.Queue()
        for core_id in range(workers):
            core_queue.put(core_id)
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
            futures = [
                executor.submit(_profile_job, input_dir, output_dir, job, core_queue, compiler_jobs)
                for job in request.jobs
            ]
            completed_labels = tuple(future.result() for future in futures)
    expected_labels = tuple(job.label for job in request.jobs)
    if completed_labels != expected_labels:
        raise RuntimeError(f"batch worker completed unexpected labels: {completed_labels}")
    return time.monotonic() - started, workers


def _main() -> None:
    """Read one batch request from stdin and write all profile artifacts."""
    args = _parse_args()
    input_dir = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"batch input directory not found: {input_dir}")
    request = parse_batch_request(json.load(sys.stdin))
    for job in request.jobs:
        kernel_path = input_dir / job.label / "kernel.py"
        if not kernel_path.is_file():
            raise FileNotFoundError(f"kernel source not found: {kernel_path}")
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True)
    elapsed_s, workers = _run_batch(input_dir, output_dir, request)
    labels = tuple(job.label for job in request.jobs)
    (output_dir / "batch_result.json").write_text(
        json.dumps(batch_result_payload(elapsed_s, workers, labels), indent=2) + "\n", encoding="utf-8"
    )
    print(f"nkigym batch profile worker: {len(labels)} kernels with {workers} workers", flush=True)


if __name__ == "__main__":
    _main()
