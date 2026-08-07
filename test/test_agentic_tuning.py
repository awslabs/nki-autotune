"""Hardware acceptance for profiler-guided agentic tuning."""

from __future__ import annotations

import pytest
from config import MFU_PROFILE_HOSTS

from kernel_library import load_workload
from nkigym.search import run_profiled_refinement

SCHEDULER_OFF_ARGS = ("enable-linear-scan-allocation=false", "enable-instruction-scheduling=false")


@pytest.mark.parametrize(
    ("workload_name", "workload_shape", "target_mfu_percent", "profile_host"),
    (
        pytest.param("attention", "q16384_kv16384_d128", 45.43, MFU_PROFILE_HOSTS[0], id="attention"),
        pytest.param("matmul-lhs", "m2048_k2048_n2048", 86.46, MFU_PROFILE_HOSTS[1], id="matmul-lhs"),
        pytest.param("matmul-lhs-t", "m2048_k2048_n2048", 89.92, MFU_PROFILE_HOSTS[0], id="matmul-lhs-t"),
        pytest.param("rmsnorm-matmul", "m2048_k2048_n2048", 85.99, MFU_PROFILE_HOSTS[1], id="rmsnorm-matmul"),
    ),
)
def test_agentic_tuning_reaches_target_mfu(
    workload_name: str, workload_shape: str, target_mfu_percent: float, profile_host: str
) -> None:
    """Agentic tuning reaches the required workload MFU."""
    workload = load_workload(workload_name, workload_shape)
    result = run_profiled_refinement(
        workload.f_nkigym,
        workload.input_specs,
        profile_host,
        target_score=target_mfu_percent,
        neuronx_cc_args=SCHEDULER_OFF_ARGS,
    )
    candidate_score = None if result.best_node_id is None else result.best_node.evaluation.score
    assert candidate_score is not None, f"{workload_name} agentic tuning produced no valid measured MFU score"
    assert (
        candidate_score >= target_mfu_percent
    ), f"{workload_name} reached {candidate_score:.2f}% MFU, expected at least {target_mfu_percent:.2f}%"
