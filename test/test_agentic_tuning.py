"""Hardware acceptance for deterministic heuristic schedule search."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from config import MFU_PROFILE_HOSTS

from kernel_library import load_workload
from nkigym.search import run_heuristic_search

MAX_HISTORICAL_MFU_REGRESSION_POINTS = 1.0
SCHEDULER_OFF_ARGS = ("enable-linear-scan-allocation=false", "enable-instruction-scheduling=false")


@pytest.mark.parametrize(
    ("workload_name", "workload_shape", "profile_host"),
    (
        pytest.param("attention", "q16384_kv16384_d128", MFU_PROFILE_HOSTS[0], id="attention"),
        pytest.param("matmul-lhs", "m2048_k2048_n2048", MFU_PROFILE_HOSTS[1], id="matmul-lhs"),
        pytest.param("matmul-lhs-t", "m2048_k2048_n2048", MFU_PROFILE_HOSTS[0], id="matmul-lhs-t"),
        pytest.param("rmsnorm-matmul", "m2048_k2048_n2048", MFU_PROFILE_HOSTS[1], id="rmsnorm-matmul"),
    ),
)
def test_heuristic_schedule_search_reaches_recorded_mfu(
    workload_name: str, workload_shape: str, profile_host: str, tmp_path: Path
) -> None:
    """Heuristic search reaches the recorded workload MFU on hardware."""
    workload = load_workload(workload_name, workload_shape)
    historical_best = workload.historical_best_mfu
    assert historical_best is not None, f"{workload_name} has no recorded historical MFU"
    target_mfu_percent = historical_best - MAX_HISTORICAL_MFU_REGRESSION_POINTS
    result = run_heuristic_search(
        workload.f_nkigym,
        workload.input_specs,
        profile_host,
        trace_dir=tmp_path / workload_name,
        neuronx_cc_args=SCHEDULER_OFF_ARGS,
    )
    evaluation = result.best_candidate.evaluation
    candidate_score = evaluation.score
    measured_score = evaluation.metrics.get("mfu_percent")
    assert candidate_score is not None, f"{workload_name} heuristic search produced no measured MFU score"
    assert isinstance(measured_score, float), f"{workload_name} produced malformed measurement evidence"
    assert math.isfinite(candidate_score), f"{workload_name} produced invalid MFU {candidate_score!r}"
    assert measured_score == candidate_score, f"{workload_name} selected an unmeasured heuristic score"
    assert (
        candidate_score >= target_mfu_percent
    ), f"{workload_name} reached {candidate_score:.2f}% MFU, expected at least {target_mfu_percent:.2f}%"
