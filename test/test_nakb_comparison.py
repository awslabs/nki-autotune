"""Check publication and plotting of saved results without repeating experiments."""

import json
import sys
from importlib import import_module
from pathlib import Path
from types import ModuleType

import pytest

from kernel_library import nakb_comparison as comparison


def _row(name: str, digest: str, seed: int, ratio: float) -> comparison.Measurement:
    """Create a synthetic measurement against the unchanged frozen baseline."""
    baseline = comparison.BASELINES[name]
    return {
        "workload": name,
        "nakb_latency_ms": baseline,
        "nkigym_latency_ms": baseline * ratio,
        "relative_latency": ratio,
        "seed": seed,
        "source": f"fixture/{name}/result.json",
        "source_sha256": digest,
        "kernel_sha256": "a" * 64,
        "measured_utc": "2026-09-28T00:00:00+00:00",
        "trn2_host": "trn2-worker.example.org",
    }


def _validation(name: str) -> dict[str, str | float | int | bool]:
    """Represent an existing successful confirmation result."""
    return {
        "workload": name,
        "correct": True,
        "latency_ms": comparison.BASELINES[name] * 1.3,
        "nakb_latency_ms": comparison.BASELINES[name],
        "relative_latency": 1.3,
        "seed": 42,
    }


@pytest.fixture
def plotting(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Load the plotting script bundled with the repository skill."""
    directory = Path(__file__).resolve().parents[1] / ".agents/skills/compare-nakb/scripts"
    monkeypatch.syspath_prepend(str(directory))
    return import_module("plot_nakb_comparison")


@pytest.fixture
def measured_repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict[str, str]]:
    """Write a complete synthetic run with controllable source identities."""
    hashes = {name: "1" * 64 for name in comparison.BASELINES}
    monkeypatch.setattr(comparison, "source_state", lambda: ("initial", hashes.copy()))
    with comparison.record_run(tmp_path, 1, ("cpu-worker.example.org",), ("trn2-worker.example.org",)) as measurements:
        measurements["workloads"].update({name: _row(name, digest, 1, 0.8) for name, digest in hashes.items()})
    return tmp_path, hashes


def test_publish_saved_validation_preserves_other_rows_and_history(
    measured_repository: tuple[Path, dict[str, str]],
) -> None:
    """Publish an existing regression without changing other results or inventing freshness."""
    directory, hashes = measured_repository
    before = comparison.load_measurements(directory)
    archive = directory / "runs" / f"{before['run_id']}.json"
    original_bytes = archive.read_bytes()
    changed = next(iter(hashes))
    source = directory / "validation.json"
    source.write_text(json.dumps(_validation(changed)))
    after = comparison.publish_measurements(directory, comparison.read_updates(source))
    assert after["workloads"][changed]["relative_latency"] == 1.3
    assert after["workloads"][changed]["source"] == str(source.resolve())
    assert "source_sha256" not in after["workloads"][changed]
    assert after["mean_relative_latency"] == pytest.approx(0.8 + 0.5 / len(hashes))
    assert after["diagnostic_only"] and not after["complete_single_run"]
    assert after["status"] == "updated" and after["seed"] is None
    assert archive.read_bytes() == original_bytes
    assert all(after["workloads"][name] == row for name, row in before["workloads"].items() if name != changed)


@pytest.mark.parametrize("invalid", ["correctness", "baseline", "ratio", "latency"])
def test_invalid_saved_result_does_not_replace_measurements(
    measured_repository: tuple[Path, dict[str, str]], invalid: str
) -> None:
    """Reject invalid or unsuccessful results before touching the shared snapshot."""
    directory, hashes = measured_repository
    original = (directory / "measurements.json").read_bytes()
    payload = _validation(next(iter(hashes)))
    changes: dict[str, tuple[str, float | bool]] = {
        "correctness": ("correct", False),
        "baseline": ("nakb_latency_ms", 99.0),
        "ratio": ("relative_latency", 99.0),
        "latency": ("latency_ms", -1.0),
    }
    key, value = changes[invalid]
    payload[key] = value
    source = directory / "invalid.json"
    source.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        comparison.publish_measurements(directory, comparison.read_updates(source))
    assert (directory / "measurements.json").read_bytes() == original


def test_legacy_snapshot_can_be_published_and_plotted_without_remeasurement(
    measured_repository: tuple[Path, dict[str, str]], plotting: ModuleType
) -> None:
    """Reuse saved numeric results while preserving their unverified source provenance."""
    directory, hashes = measured_repository
    rows = comparison.load_measurements(directory)["workloads"]
    for row in rows.values():
        row.pop("source_sha256")
    source = directory / "legacy.json"
    source.write_text(json.dumps({"workloads": rows}))
    output = directory / "published"
    result = comparison.publish_measurements(output, comparison.read_updates(source))
    assert result["configuration_count"] == len(hashes)
    assert result["diagnostic_only"] and not result["complete_single_run"]
    assert all("source_sha256" not in row for row in result["workloads"].values())
    plotted, snapshot, _ = plotting.load_measurements(output / "measurements.json")
    assert len(plotted) == len(hashes)
    assert plotting.measurement_status(snapshot) == "mixed-run diagnostic"


def test_source_identity_distinguishes_ladder_and_backend_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Source diagnostics distinguish one ladder edit from a backend change."""
    monkeypatch.setattr(comparison, "_source_digest", lambda directory: "stable")
    revision, before = comparison.source_state()
    name = next(name for name, ladder in comparison.BEST_NKIGYM_LADDERS.items() if ladder)
    monkeypatch.setitem(comparison.BEST_NKIGYM_LADDERS, name, comparison.BEST_NKIGYM_LADDERS[name][1:])
    changed_revision, changed = comparison.source_state()
    assert revision != changed_revision
    assert {key for key in before if before[key] != changed[key]} == {name}
    monkeypatch.setattr(comparison, "_source_digest", lambda directory: "changed")
    _, backend_changed = comparison.source_state()
    assert all(backend_changed[key] != changed[key] for key in changed)


def test_source_digest_ignores_experiment_caches(tmp_path: Path) -> None:
    """Cached kernels do not affect the identity of the installed backend."""
    (tmp_path / "module.py").write_text("value = 1\n")
    before = comparison._source_digest(tmp_path)
    cache = tmp_path / ".cache" / "experiment"
    cache.mkdir(parents=True)
    (cache / "kernel.py").write_text("value = 2\n")
    assert comparison._source_digest(tmp_path) == before
    (tmp_path / "module.py").write_text("value = 3\n")
    assert comparison._source_digest(tmp_path) != before


def test_publisher_cannot_overwrite_active_benchmark(measured_repository: tuple[Path, dict[str, str]]) -> None:
    """Keep an ongoing explicit benchmark's checkpoint intact."""
    directory, hashes = measured_repository
    rows = {name: _row(name, digest, 2, 0.8) for name, digest in hashes.items()}
    with comparison.record_run(directory, 2, ("cpu-worker.example.org",), ("trn2-worker.example.org",)) as active:
        active["workloads"].update(rows)
        with pytest.raises(RuntimeError, match="already running"):
            comparison.publish_measurements(directory, rows)
        assert comparison.load_measurements(directory)["run_id"] == active["run_id"]


@pytest.mark.parametrize("unversioned", [False, True])
def test_plot_reads_saved_results_without_checking_backend(
    measured_repository: tuple[Path, dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
    unversioned: bool,
    plotting: ModuleType,
) -> None:
    """Plot available results by default without replaying or inspecting current kernels."""
    directory, hashes = measured_repository
    path = directory / "measurements.json"
    _, current, _ = plotting.load_measurements(path)
    if unversioned:
        for row in current["workloads"].values():
            row.pop("source_sha256")
        comparison.save_measurements(directory, current)
    else:
        hashes[next(iter(hashes))] = "2" * 64
    monkeypatch.setattr(comparison, "source_state", lambda: pytest.fail("Plotting inspected current kernels"))
    rows, saved, _ = plotting.load_measurements(path)
    assert len(rows) == len(hashes)
    assert plotting.measurement_status(saved) == "one complete run"


def test_incomplete_snapshot_is_not_presented_as_full_registry(tmp_path: Path, plotting: ModuleType) -> None:
    """Keep the full-registry plotting requirement while publishing partial progress."""
    name = next(iter(comparison.BASELINES))
    comparison.publish_measurements(tmp_path, {name: _row(name, "1" * 64, 1, 0.8)})
    with pytest.raises(ValueError, match="Registry coverage mismatch"):
        plotting.load_measurements(tmp_path / "measurements.json")


def test_publish_command_reports_saved_mean(
    measured_repository: tuple[Path, dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The publication command consumes a saved result and labels the resulting statistic."""
    directory, hashes = measured_repository
    source = directory / "validation.json"
    source.write_text(json.dumps(_validation(next(iter(hashes)))))
    monkeypatch.setattr(comparison, "RESULTS_DIRECTORY", directory)
    monkeypatch.setattr(sys, "argv", ["nakb_comparison", "--publish", str(source)])
    comparison.main()
    captured = capsys.readouterr()
    assert "Saved measurements: mean_relative_latency=" in captured.out
    assert "Source freshness unverified" in captured.out
    assert "status=updated diagnostic_only=True" in captured.out
