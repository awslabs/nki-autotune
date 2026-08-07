"""Hardware MFU regression coverage for the best-known generated kernels."""

from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from config import MFU_PROFILE_HOSTS

from kernel_library import load_workload
from nkigym.codegen import render
from nkigym.environment import KernelMDP
from nkigym.profile import profile
from nkigym.search.types import InputSpecs

MIN_LHS_T_MFU_PERCENT = 90.0
MIN_RMSNORM_MATMUL_MFU_PERCENT = 85.0
MIN_ATTENTION_MFU_PERCENT = 44.0
MAX_RMSNORM_MATMUL_MFU_GAP_PERCENT = 3.0
ENDPOINT_PROFILE_TIMEOUT_SECONDS = 1800
SCHEDULER_OFF_ARGS = ("enable-linear-scan-allocation=false", "enable-instruction-scheduling=false")
ENDPOINT_SHAPES = {
    "attention": "q16384_kv16384_d128",
    "matmul-lhs": "m2048_k2048_n2048",
    "matmul-lhs-t": "m2048_k2048_n2048",
    "rmsnorm-matmul": "m2048_k2048_n2048",
}


@dataclass(frozen=True)
class _Endpoint:
    """One rendered final state from a retained workload ladder."""

    name: str
    kernel: str
    func_name: str
    input_specs: InputSpecs


def _build_endpoint(name: str, shape: str) -> _Endpoint:
    """Replay one retained ladder and render its final state."""
    workload = load_workload(name, shape)
    environment = KernelMDP(
        workload.f_nkigym,
        workload.input_specs,
        transforms=[transform for transform, _option in workload.best_action_ladder],
    )
    state = environment.reset()
    for action in workload.best_action_ladder:
        state = environment.step(state, action)
    endpoint = _Endpoint(
        name=name, kernel=render(state), func_name=f"nki_{state.func_name}", input_specs=workload.input_specs
    )
    return endpoint


def _build_endpoints() -> dict[str, _Endpoint]:
    """Build every fixed MFU endpoint from its kernel-library ladder."""
    endpoints = {name: _build_endpoint(name, shape) for name, shape in ENDPOINT_SHAPES.items()}
    return endpoints


def _profile_endpoint(host: str, endpoint: _Endpoint, cache_dir: Path) -> float:
    """Profile one generated endpoint on a Trn2 NeuronCore."""
    mfu_percent, _latency_ms = profile(
        host=host,
        kernel=endpoint.kernel,
        func_name=endpoint.func_name,
        input_specs=endpoint.input_specs,
        cache_dir=cache_dir,
        neuronx_cc_args=SCHEDULER_OFF_ARGS,
        lnc=1,
        timeout_s=ENDPOINT_PROFILE_TIMEOUT_SECONDS,
    )
    return mfu_percent


def _validated_mfu(name: str, value: float) -> float:
    """Reject malformed hardware measurements."""
    if not math.isfinite(value) or value < 0.0 or value > 100.0:
        raise AssertionError(f"generated {name} endpoint returned invalid MFU {value!r}")
    return value


def _profile_endpoint_group(host: str, requests: tuple[tuple[str, _Endpoint, Path], ...]) -> dict[str, float]:
    """Profile one sequential endpoint group on a single Trn2 host."""
    measurements = {
        name: _validated_mfu(name, _profile_endpoint(host, endpoint, cache_dir))
        for name, endpoint, cache_dir in requests
    }
    return measurements


def test_best_known_generated_kernels_do_not_regress_mfu(tmp_path: Path) -> None:
    """Best-known generated endpoints retain their established hardware MFU."""
    cache_dir = tmp_path
    endpoints = _build_endpoints()
    lhs_t_host, lhs_host = MFU_PROFILE_HOSTS
    lhs_t_requests = (
        ("lhsT", endpoints["matmul-lhs-t"], cache_dir / "matmul_lhs_t_rhs"),
        ("attention", endpoints["attention"], cache_dir / "attention"),
    )
    lhs_requests = (
        ("lhs", endpoints["matmul-lhs"], cache_dir / "matmul_lhs_rhs"),
        ("RMSNorm+matmul", endpoints["rmsnorm-matmul"], cache_dir / "rmsnorm_matmul"),
    )
    with ThreadPoolExecutor(max_workers=len(MFU_PROFILE_HOSTS)) as executor:
        lhs_t_future = executor.submit(_profile_endpoint_group, lhs_t_host, lhs_t_requests)
        lhs_future = executor.submit(_profile_endpoint_group, lhs_host, lhs_requests)
        measurements_by_name = {**lhs_t_future.result(), **lhs_future.result()}
    lhs_t_mfu = measurements_by_name["lhsT"]
    lhs_mfu = measurements_by_name["lhs"]
    rmsnorm_mfu = measurements_by_name["RMSNorm+matmul"]
    attention_mfu = measurements_by_name["attention"]
    rmsnorm_gap = lhs_mfu - rmsnorm_mfu
    measurements = {
        "lhs_t_mfu_percent": lhs_t_mfu,
        "lhs_mfu_percent": lhs_mfu,
        "rmsnorm_matmul_mfu_percent": rmsnorm_mfu,
        "rmsnorm_matmul_gap_percent": rmsnorm_gap,
        "attention_mfu_percent": attention_mfu,
    }
    print(json.dumps(measurements, indent=2, sort_keys=True), flush=True)

    assert (
        lhs_t_mfu >= MIN_LHS_T_MFU_PERCENT
    ), f"lhsT measured {lhs_t_mfu:.2f}% MFU, expected at least {MIN_LHS_T_MFU_PERCENT:.2f}%"
    assert rmsnorm_mfu >= MIN_RMSNORM_MATMUL_MFU_PERCENT, (
        f"RMSNorm+matmul measured {rmsnorm_mfu:.2f}% MFU, " f"expected at least {MIN_RMSNORM_MATMUL_MFU_PERCENT:.2f}%"
    )
    assert rmsnorm_gap <= MAX_RMSNORM_MATMUL_MFU_GAP_PERCENT, (
        f"RMSNorm+matmul trailed lhs matmul by {rmsnorm_gap:.2f} MFU points, "
        f"expected at most {MAX_RMSNORM_MATMUL_MFU_GAP_PERCENT:.2f}"
    )
    assert (
        attention_mfu >= MIN_ATTENTION_MFU_PERCENT
    ), f"attention measured {attention_mfu:.2f}% MFU, expected at least {MIN_ATTENTION_MFU_PERCENT:.2f}%"
