"""Hardware acceptance tests for policy-driven iterative search."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from kernel_library import NAKB_WORKLOADS, Workload
from nkigym.search import Policy, run_search
from nkigym.synthesis import synthesize_torch_to_nkigym

SCHEDULER_OFF_ARGS = ("enable-linear-scan-allocation=false", "enable-instruction-scheduling=false")
LATENCY_RELATIVE_TOLERANCE = 0.02
MAX_TRANSFORMS_PER_EVALUATION = 10
SEARCH_SAMPLE_SIZE = 5
SEARCH_SAMPLE_SEED = random.SystemRandom().randrange(1 << 63)


def _sample_search_workloads(seed: int) -> dict[str, Workload]:
    """Sample one configuration from five distinct NAKB workload types."""
    if len(NAKB_WORKLOADS) < SEARCH_SAMPLE_SIZE:
        raise RuntimeError(f"kernel_library has fewer than {SEARCH_SAMPLE_SIZE} workload types")
    rng = random.Random(seed)
    workload_types = rng.sample(tuple(NAKB_WORKLOADS), SEARCH_SAMPLE_SIZE)
    sampled: dict[str, Workload] = {}
    for workload_type in workload_types:
        workloads = NAKB_WORKLOADS[workload_type]
        workload_index = rng.randrange(len(workloads))
        sampled[f"{workload_type}_{workload_index}"] = workloads[workload_index]
    return sampled


SEARCH_WORKLOADS = _sample_search_workloads(SEARCH_SAMPLE_SEED)
SEARCH_TARGETS = tuple(pytest.param(name, id=name) for name in SEARCH_WORKLOADS)


@pytest.fixture(scope="module", autouse=True)
def report_search_sample() -> None:
    """Report the random sample so a run can be reproduced."""
    print(f"search_sample_seed={SEARCH_SAMPLE_SEED}", flush=True)
    print(f"search_sample_workloads={','.join(SEARCH_WORKLOADS)}", flush=True)


@pytest.mark.parametrize("workload_name", SEARCH_TARGETS)
def test_transform_selection_policy_reaches_nakb_performance(
    workload_name: str, trn2_hosts: tuple[str, ...], tmp_path: Path
) -> None:
    """The transform-selection policy reaches within 2% of historical NAKB performance."""
    workload = SEARCH_WORKLOADS[workload_name]
    workload_index = tuple(SEARCH_WORKLOADS).index(workload_name)
    profile_host = trn2_hosts[workload_index % len(trn2_hosts)]
    kernel = synthesize_torch_to_nkigym(workload["torch_ref"], workload["input_specs"])
    historical_latency_ms = workload["best_historical_latency_ms"]
    maximum_latency_ms = historical_latency_ms * (1.0 + LATENCY_RELATIVE_TOLERANCE)
    result = run_search(
        kernel.function,
        kernel.input_specs,
        profile_host,
        policy=Policy(),
        max_transforms_per_evaluation=MAX_TRANSFORMS_PER_EVALUATION,
        target_latency_ms=maximum_latency_ms,
        trace_dir=tmp_path / workload_name,
        neuronx_cc_args=SCHEDULER_OFF_ARGS,
    )
    best_latency_ms = result.best_latency_ms
    assert best_latency_ms <= maximum_latency_ms, (
        f"{workload_name} sample seed {SEARCH_SAMPLE_SEED} reached {best_latency_ms:.6f} ms, more than 2% slower than "
        f"best_historical_latency_ms={historical_latency_ms:.6f} ms"
    )
