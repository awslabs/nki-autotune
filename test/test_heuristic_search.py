"""Hardware acceptance tests for fresh heuristic schedule search."""

from __future__ import annotations

from pathlib import Path

import pytest
from config import MFU_PROFILE_HOSTS

from kernel_library import WORKLOADS
from nkigym.search import run_heuristic_search
from nkigym.synthesis import synthesize_numpy_to_nkigym

SCHEDULER_OFF_ARGS = ("enable-linear-scan-allocation=false", "enable-instruction-scheduling=false")
MFU_RELATIVE_TOLERANCE = 0.005
HEURISTIC_TARGETS = tuple(
    pytest.param(name, MFU_PROFILE_HOSTS[index % len(MFU_PROFILE_HOSTS)], id=name)
    for index, name in enumerate(WORKLOADS)
)


@pytest.mark.parametrize(("workload_name", "profile_host"), HEURISTIC_TARGETS)
def test_fresh_heuristic_search_reaches_within_tolerance_of_best_historical_mfu(
    workload_name: str, profile_host: str, tmp_path: Path
) -> None:
    """Fresh heuristic search reaches within 0.5% of the historical MFU."""
    workload = WORKLOADS[workload_name]
    kernel = synthesize_numpy_to_nkigym(workload["numpy_ref"], workload["input_specs"])
    historical_mfu = workload["best_historical_mfu"]
    minimum_mfu = historical_mfu * (1.0 - MFU_RELATIVE_TOLERANCE)
    result = run_heuristic_search(
        kernel.function,
        kernel.input_specs,
        profile_host,
        target_score=minimum_mfu,
        trace_dir=tmp_path / workload_name,
        neuronx_cc_args=SCHEDULER_OFF_ARGS,
    )
    best_score = result.best_node.evaluation.score
    assert best_score is not None, f"{workload_name} produced no measured MFU"
    assert best_score >= minimum_mfu, (
        f"{workload_name} reached {best_score:.2f}% MFU, "
        f"more than 0.5% below best_historical_mfu={historical_mfu:.2f}%"
    )
