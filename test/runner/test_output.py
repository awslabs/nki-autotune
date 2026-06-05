"""Unit tests for the local cache-writer + ProfileOutput summary."""

import json

from autotune.runner.output import ProfileOutput, write_results_json
from autotune.runner.types import ProfileResult


def _summary_result(name: str, total_time: float, mfu: float) -> ProfileResult:
    """A ProfileResult shaped like a successful HW run."""
    return ProfileResult(
        kernel_name=name,
        hardware_output="[128, 512] float32",
        profiler_summary={
            "total_time": total_time,
            "mfu_estimated_percent": mfu,
        },
    )


def test_write_results_json_no_hosts_key(tmp_path):
    """results.json metadata must not carry a 'hosts' key in local mode."""
    results = [_summary_result("k_0.py", 0.001, 0.9)]
    write_results_json(str(tmp_path), num_kernels=1, results=results, wallclock_s=1.5)

    data = json.loads((tmp_path / "results.json").read_text())
    assert "hosts" not in data["metadata"]
    assert data["metadata"]["num_kernels"] == 1
    assert data["metadata"]["wallclock_s"] == 1.5
    assert data["metrics"]["success"] == 1
    assert data["kernels"][0]["kernel_name"] == "k_0.py"


def test_write_results_json_sorts_kernels_numerically(tmp_path):
    """kernel_2 must precede kernel_10 in the index."""
    results = [
        _summary_result("kernel_10.py", 0.002, 0.5),
        _summary_result("kernel_2.py", 0.001, 0.5),
    ]
    write_results_json(str(tmp_path), num_kernels=2, results=results, wallclock_s=1.0)
    data = json.loads((tmp_path / "results.json").read_text())
    names = [k["kernel_name"] for k in data["kernels"]]
    assert names == ["kernel_2.py", "kernel_10.py"]


def test_profile_output_str_has_no_hosts_line():
    """The human summary must not reference hosts (field removed)."""
    out = ProfileOutput(
        results=[_summary_result("k_0.py", 0.001, 0.9)],
        compiler_logs={},
        elapsed_s=2.0,
        cache_dir="",
    )
    text = str(out)
    assert "Hosts:" not in text
    assert "Succeeded:" in text
