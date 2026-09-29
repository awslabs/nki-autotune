"""Publish and summarize saved NAKB comparison measurements.

Use ``python -m kernel_library.nakb_comparison --publish validation.json``
after installing an already validated ladder. This module runs no experiments.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from statistics import fmean
from tempfile import NamedTemporaryFile
from typing import Literal, TypedDict, cast
from uuid import uuid4

import nkigym
from benchmark import NAKB_WORKLOADS
from kernel_library import BEST_NKIGYM_LADDERS

REPOSITORY = Path(__file__).resolve().parents[1]
RESULTS_DIRECTORY = REPOSITORY / "artifacts/nakb_latency_comparison"
WORKLOADS = {
    f"{family}_{index}": workload
    for family, workloads in NAKB_WORKLOADS.items()
    for index, workload in enumerate(workloads)
}
BASELINES = {name: workload["nakb_latency_ms"] for name, workload in WORKLOADS.items()}


class MeasurementProvenance(TypedDict, total=False):
    """Optional version fields; older rows have unverified freshness."""

    source_sha256: str
    kernel_sha256: str
    measured_utc: str
    trn2_host: str


class Measurement(MeasurementProvenance):
    """A configuration whose intermediate and final correctness checks passed."""

    workload: str
    nakb_latency_ms: float
    nkigym_latency_ms: float
    relative_latency: float
    seed: int
    source: str


class Statistics(TypedDict):
    """Equal-weight statistics over the recorded configurations."""

    configuration_count: int
    expected_configuration_count: int
    mean_relative_latency: float | None
    mean_latency_reduction_percent: float | None
    faster_count: int
    slower_count: int
    equal_count: int


class Measurements(Statistics):
    """Schema version 1 for both live runs and explicitly imported diagnostics."""

    schema_version: Literal[1]
    run_id: str
    updated_utc: str
    status: Literal["running", "passed", "failed", "imported", "updated"]
    diagnostic_only: bool
    complete_single_run: bool
    description: str
    seed: int | None
    cpu_hosts: list[str]
    trn2_hosts: list[str]
    error: str | None
    workloads: dict[str, Measurement]


def _source_digest(directory: Path) -> str:
    """Hash Python sources without traversing cached experiments or bytecode."""
    digest = sha256()
    for root, directories, filenames in os.walk(directory):
        directories[:] = sorted(name for name in directories if not name.startswith(".") and name != "__pycache__")
        for name in sorted(filenames):
            if name.endswith(".py"):
                path = Path(root) / name
                digest.update(path.relative_to(directory).as_posix().encode() + b"\0")
                digest.update(sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def source_state() -> tuple[str, dict[str, str]]:
    """Identify the repository revision and each backend-plus-ladder version.

    A ladder edit changes only that configuration's hash. Backend or reference
    edits change every hash. This check reads source files and executes no kernels.
    """
    backend = REPOSITORY / "nkigym/src/nkigym"
    if nkigym.__file__ is None or Path(nkigym.__file__).resolve().parent != backend.resolve():
        raise RuntimeError("Measurements must use the NKIGym backend from this repository")
    backend_digest = sha256((_source_digest(backend) + _source_digest(REPOSITORY / "benchmark")).encode()).hexdigest()
    hashes = {
        name: sha256(
            backend_digest.encode()
            + json.dumps(
                [
                    (type(transform).__module__, type(transform).__qualname__, asdict(option))
                    for transform, option in BEST_NKIGYM_LADDERS.get(name, ())
                ],
                sort_keys=True,
                allow_nan=False,
            ).encode()
        ).hexdigest()
        for name in BASELINES
    }
    revision = sha256(
        backend_digest.encode()
        + (REPOSITORY / "kernel_library/_best_nkigym.py").read_bytes()
        + json.dumps(hashes, sort_keys=True).encode()
    ).hexdigest()
    return revision, hashes


def stale_workloads(measurements: Measurements) -> list[str]:
    """List missing, changed, and unversioned configurations against current source."""
    _, hashes = source_state()
    rows = measurements["workloads"]
    return [name for name, digest in hashes.items() if name not in rows or rows[name].get("source_sha256") != digest]


def statistics(rows: dict[str, Measurement]) -> Statistics:
    """Compute the arithmetic mean of ratios, including every regression."""
    ratios = [row["nkigym_latency_ms"] / row["nakb_latency_ms"] for _, row in sorted(rows.items())]
    mean_ratio = fmean(ratios) if ratios else None
    return {
        "configuration_count": len(rows),
        "expected_configuration_count": len(BASELINES),
        "mean_relative_latency": mean_ratio,
        "mean_latency_reduction_percent": 100 * (1 - mean_ratio) if mean_ratio is not None else None,
        "faster_count": sum(ratio < 1 for ratio in ratios),
        "slower_count": sum(ratio > 1 for ratio in ratios),
        "equal_count": sum(ratio == 1 for ratio in ratios),
    }


def validate_row(name: str, row: Measurement) -> None:
    """Check a row's identity, frozen baseline, latency ratio, and provenance."""
    if row["workload"] != name or not isinstance(row["source"], str) or not row["source"].strip():
        raise ValueError(f"Missing or inconsistent measurement provenance: {name}")
    values = (row["nakb_latency_ms"], row["nkigym_latency_ms"], row["relative_latency"])
    if not all(type(value) in (float, int) and math.isfinite(value) and value > 0 for value in values):
        raise ValueError(f"Latencies and ratios must be finite and positive: {name}")
    if row["nakb_latency_ms"] != BASELINES[name]:
        raise ValueError(f"NAKB latency differs from the frozen benchmark: {name}")
    if not math.isclose(row["relative_latency"], values[1] / values[0], rel_tol=1e-12):
        raise ValueError(f"Recorded relative latency disagrees with the measured latencies: {name}")
    if type(row["seed"]) is not int or not 0 <= row["seed"] < 1 << 63:
        raise ValueError(f"Expected a 63-bit validation seed: {name}")
    for key in ("source_sha256", "kernel_sha256"):
        if key in row:
            digest = row.get(key)
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError(f"Expected a SHA-256 digest for {name}: {key}")
    if "measured_utc" in row and datetime.fromisoformat(row["measured_utc"]).utcoffset() is None:
        raise ValueError(f"measured_utc must include its timezone: {name}")


def validate_measurements(measurements: Measurements, require_complete: bool) -> None:
    """Reject incompatible data, incomplete plots, and inconsistent summaries."""
    if measurements["schema_version"] != 1 or type(measurements["schema_version"]) is not int:
        raise ValueError("Unsupported measurement schema_version; expected 1")
    rows = measurements["workloads"]
    missing = sorted(set(BASELINES) - set(rows))
    unexpected = sorted(set(rows) - set(BASELINES))
    if unexpected or (require_complete and missing):
        raise ValueError(f"Registry coverage mismatch; missing={missing}, unexpected={unexpected}")
    flags = (measurements["diagnostic_only"], measurements["complete_single_run"])
    if any(type(flag) is not bool for flag in flags) or (not flags[0] and (not flags[1] or missing)):
        raise ValueError("Incomplete or mixed-run measurements must be diagnostic_only")
    if flags[1] and (missing or any(row["seed"] != measurements["seed"] for row in rows.values())):
        raise ValueError("complete_single_run requires full registry coverage and one validation seed")
    for name, row in rows.items():
        validate_row(name, row)
    expected = statistics(rows)
    for key, value in expected.items():
        actual = measurements[key]
        matches = (
            type(actual) in (int, float) and math.isclose(actual, value, rel_tol=1e-12, abs_tol=1e-12)
            if isinstance(value, float)
            else actual == value and type(actual) is type(value)
        )
        if not matches:
            raise ValueError(f"Recorded {key} disagrees with the individual measurements")
    if measurements["status"] not in {"running", "passed", "failed", "imported", "updated"}:
        raise ValueError("Unknown measurement status")
    if require_complete and measurements["status"] == "running":
        raise ValueError("The benchmark is still running; select a finished run")
    if measurements["status"] == "passed" and (not flags[1] or flags[0]):
        raise ValueError("A passed benchmark requires one complete, non-diagnostic run")
    if not measurements["run_id"] or Path(measurements["run_id"]).name != measurements["run_id"]:
        raise ValueError("run_id must be a non-empty filename component")
    if datetime.fromisoformat(measurements["updated_utc"]).utcoffset() is None:
        raise ValueError("updated_utc must include its timezone")


def _atomic_write(path: Path, content: str) -> None:
    """Replace one JSON document only after its complete contents are written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=".measurements-", delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(content)
            stream.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def save_measurements(directory: Path, measurements: Measurements) -> None:
    """Checkpoint one run and the fixed latest-results file without merging runs."""
    measurements.update(**statistics(measurements["workloads"]))
    measurements["updated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    validate_measurements(measurements, require_complete=False)
    content = json.dumps(measurements, indent=2, sort_keys=True, allow_nan=False) + "\n"
    _atomic_write(directory / "runs" / f"{measurements['run_id']}.json", content)
    _atomic_write(directory / "measurements.json", content)


@contextmanager
def _writer_lock(directory: Path) -> Iterator[None]:
    """Prevent a publisher or benchmark from replacing another active writer's rows."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".writer.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("A measurement publisher or ladder benchmark is already running") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def load_measurements(directory: Path) -> Measurements:
    """Read and validate the authoritative measurement file, including partial runs."""
    measurements = cast(Measurements, json.loads((directory / "measurements.json").read_text()))
    validate_measurements(measurements, require_complete=False)
    return measurements


@contextmanager
def record_run(
    directory: Path, seed: int, cpu_hosts: tuple[str, ...], trn2_hosts: tuple[str, ...]
) -> Iterator[Measurements]:
    """Record one explicit benchmark run, preserving progress on failure."""
    if type(seed) is not int or not 0 <= seed < 1 << 63:
        raise ValueError("Expected a 63-bit validation seed")
    with _writer_lock(directory):
        revision, _ = source_state()
        measurements: Measurements = {
            **statistics({}),
            "schema_version": 1,
            "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8],
            "updated_utc": "",
            "status": "running",
            "diagnostic_only": True,
            "complete_single_run": False,
            "description": "Ladder benchmark; each recorded configuration passed intermediate and Trn2 correctness checks.",
            "seed": seed,
            "cpu_hosts": list(cpu_hosts),
            "trn2_hosts": list(trn2_hosts),
            "error": None,
            "workloads": {},
        }
        save_measurements(directory, measurements)
        try:
            yield measurements
            if source_state()[0] != revision:
                raise RuntimeError("Repository source changed during measurement; retry with source held fixed")
            measurements["status"] = "passed"
        except BaseException as error:
            measurements["status"] = "failed"
            measurements["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            measurements["complete_single_run"] = set(measurements["workloads"]) == set(BASELINES)
            measurements["diagnostic_only"] = not measurements["complete_single_run"]
            save_measurements(directory, measurements)


def read_updates(path: Path) -> dict[str, Measurement]:
    """Read an existing successful validation result or saved workload snapshot.

    Source hashes are preserved only when already recorded; publishing does not
    certify an old measurement against today's backend or installed ladder.
    """
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a validation result or measurement snapshot: {path}")
    if "workloads" in raw:
        rows = cast(dict[str, Measurement], raw["workloads"])
    else:
        if raw.get("correct") is not True:
            raise ValueError(f"Only successful, already validated results can be published: {path}")
        row = cast(
            Measurement,
            {
                "workload": raw["workload"],
                "nkigym_latency_ms": raw["latency_ms"],
                "nakb_latency_ms": raw["nakb_latency_ms"],
                "relative_latency": raw["relative_latency"],
                "seed": raw["seed"],
                "source": str(path.resolve()),
                **{
                    key: raw[key]
                    for key in ("source_sha256", "kernel_sha256", "measured_utc", "trn2_host")
                    if key in raw
                },
            },
        )
        rows = {row["workload"]: row}
    if not isinstance(rows, dict) or not rows:
        raise ValueError(f"No workload measurements found: {path}")
    for name, row in rows.items():
        if name not in BASELINES:
            raise ValueError(f"Unknown workload in {path}: {name}")
        validate_row(name, row)
    return rows


def publish_measurements(directory: Path, updates: dict[str, Measurement]) -> Measurements:
    """Merge already validated installed-kernel results without replaying or profiling."""
    if not updates or set(updates) - set(BASELINES):
        raise ValueError("Publish at least one measurement, using only registered workloads")
    for name, row in updates.items():
        validate_row(name, row)
    with _writer_lock(directory):
        rows = load_measurements(directory)["workloads"] if (directory / "measurements.json").is_file() else {}
        rows.update(updates)
        measurements: Measurements = {
            **statistics(rows),
            "schema_version": 1,
            "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8],
            "updated_utc": "",
            "status": "updated",
            "diagnostic_only": True,
            "complete_single_run": False,
            "description": "Mixed-run diagnostic; saved validation results published after installing ladder updates.",
            "seed": None,
            "cpu_hosts": [],
            "trn2_hosts": sorted({row["trn2_host"] for row in rows.values() if "trn2_host" in row}),
            "error": None,
            "workloads": rows,
        }
        save_measurements(directory, measurements)
    return measurements


def main() -> None:
    """Publish existing result files or print statistics from the shared saved record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--publish", type=Path, nargs="+", metavar="RESULT.json", help="Merge already validated results"
    )
    args = parser.parse_args()
    if args.publish:
        updates: dict[str, Measurement] = {}
        for path in cast(list[Path], args.publish):
            rows = read_updates(path)
            if set(updates) & set(rows):
                parser.error(f"Multiple results for the same workload: {path}")
            updates.update(rows)
        measurements = publish_measurements(RESULTS_DIRECTORY, updates)
    else:
        if not (RESULTS_DIRECTORY / "measurements.json").is_file():
            parser.exit(1, "No saved measurements. Publish existing validation results with --publish RESULT.json.\n")
        measurements = load_measurements(RESULTS_DIRECTORY)
    stale = stale_workloads(measurements)
    print(
        f"run_id={measurements['run_id']} saved_configurations={measurements['configuration_count']}/{len(BASELINES)}"
    )
    ratio = measurements["mean_relative_latency"]
    if ratio is not None:
        print(
            f"Saved measurements: mean_relative_latency={ratio:.9f} mean_latency_reduction_percent={100 * (1 - ratio):.6f}"
        )
    if stale:
        print(f"Source freshness unverified for {len(stale)}/{len(BASELINES)} configurations; no experiments were run.")
    print(
        f"status={measurements['status']} diagnostic_only={measurements['diagnostic_only']} "
        f"complete_single_run={measurements['complete_single_run']}"
    )


if __name__ == "__main__":
    main()
