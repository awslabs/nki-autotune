"""Hardware acceptance tests for policy-driven iterative search."""

from __future__ import annotations

from pathlib import Path

import pytest
from config import PROFILE_HOSTS

from kernel_library import WORKLOADS
from nkigym.search import Policy, run_search
from nkigym.synthesis import synthesize_numpy_to_nkigym

SCHEDULER_OFF_ARGS = ("enable-linear-scan-allocation=false", "enable-instruction-scheduling=false")
LATENCY_RELATIVE_TOLERANCE = 0.03
MAX_TRANSFORMS_PER_EVALUATION = 10
SEARCH_TARGETS = tuple(
    pytest.param(name, PROFILE_HOSTS[index % len(PROFILE_HOSTS)], id=name) for index, name in enumerate(WORKLOADS)
)


SEARCH_POLICIES = (pytest.param(Policy(), id="epsilon-greedy"),)


@pytest.mark.parametrize(("workload_name", "profile_host"), SEARCH_TARGETS)
@pytest.mark.parametrize("policy", SEARCH_POLICIES)
def test_search_policy_reaches_within_tolerance_of_best_historical_latency(
    workload_name: str, profile_host: str, policy: Policy, tmp_path: Path
) -> None:
    """Each search policy reaches within 3% of the best historical latency."""
    workload = WORKLOADS[workload_name]
    kernel = synthesize_numpy_to_nkigym(workload["numpy_ref"], workload["input_specs"])
    historical_latency_ms = workload["best_historical_latency_ms"]
    maximum_latency_ms = historical_latency_ms * (1.0 + LATENCY_RELATIVE_TOLERANCE)
    result = run_search(
        kernel.function,
        kernel.input_specs,
        profile_host,
        policy=policy,
        max_transforms_per_evaluation=MAX_TRANSFORMS_PER_EVALUATION,
        target_latency_ms=maximum_latency_ms,
        trace_dir=tmp_path / workload_name,
        neuronx_cc_args=SCHEDULER_OFF_ARGS,
    )
    best_latency_ms = result.best_latency_ms
    assert best_latency_ms <= maximum_latency_ms, (
        f"{workload_name} reached {best_latency_ms:.6f} ms, more than 3% slower than "
        f"best_historical_latency_ms={historical_latency_ms:.6f} ms"
    )
