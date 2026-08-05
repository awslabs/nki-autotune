"""Hardware MFU regression coverage for the best-known generated kernels."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

from kernel_library.attention import ladder as attention_ladder
from kernel_library.matmul.lhs_rhs import ladder as lhs_ladder
from kernel_library.matmul.lhsT_rhs import ladder as lhs_t_ladder
from kernel_library.rmsnorm_matmul import ladder as rmsnorm_matmul_ladder
from nkigym.codegen import render
from nkigym.ir import KernelIR
from nkigym.profile import profile

MIN_LHS_T_MFU_PERCENT = 90.0
MIN_RMSNORM_MATMUL_MFU_PERCENT = 85.0
MIN_ATTENTION_MFU_PERCENT = 44.0
MAX_RMSNORM_MATMUL_MFU_GAP_PERCENT = 3.0
ENDPOINT_PROFILE_TIMEOUT_SECONDS = 1800
SCHEDULER_OFF_ARGS = ("enable-linear-scan-allocation=false", "enable-instruction-scheduling=false")


def _profile_endpoint(
    host: str, endpoint: KernelIR, input_specs: dict[str, tuple[tuple[int, ...], str]], cache_dir: Path
) -> float:
    """Profile one generated endpoint on a Trn2 NeuronCore."""
    mfu_percent, _latency_ms = profile(
        host=host,
        kernel=render(endpoint),
        func_name=f"nki_{endpoint.func_name}",
        input_specs=input_specs,
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


def test_best_known_generated_kernels_do_not_regress_mfu(tmp_path: Path) -> None:
    """Best-known generated endpoints retain their established hardware MFU."""
    host = os.environ.get("NKI_PROFILE_HOST", "gym-1")
    configured_directory = os.environ.get("DEVELOPER_GATE_ARTIFACT_DIRECTORY")
    cache_dir = Path(configured_directory) if configured_directory is not None else tmp_path
    lhs_t_endpoint = lhs_t_ladder._build_ladder()[-1]
    lhs_endpoint = lhs_ladder._build_ladder()[-1]
    rmsnorm_endpoint = rmsnorm_matmul_ladder._build_ladder()[-1]
    attention_specs = attention_ladder._input_specs(attention_ladder.SEQUENCE_LENGTH)
    attention_endpoint = attention_ladder._build_ladder(attention_specs)[-1]

    lhs_t_mfu = _validated_mfu(
        "lhsT", _profile_endpoint(host, lhs_t_endpoint, lhs_t_ladder.INPUT_SPECS, cache_dir / "matmul_lhs_t_rhs")
    )
    lhs_mfu = _validated_mfu(
        "lhs", _profile_endpoint(host, lhs_endpoint, lhs_ladder.INPUT_SPECS, cache_dir / "matmul_lhs_rhs")
    )
    rmsnorm_mfu = _validated_mfu(
        "RMSNorm+matmul",
        _profile_endpoint(host, rmsnorm_endpoint, rmsnorm_matmul_ladder.INPUT_SPECS, cache_dir / "rmsnorm_matmul"),
    )
    attention_mfu = _validated_mfu(
        "attention", _profile_endpoint(host, attention_endpoint, attention_specs, cache_dir / "attention")
    )
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
