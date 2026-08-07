"""Hardware MFU regression coverage for the best-known generated kernels."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

from nkigym.profile import profile
from nkigym.search.types import InputSpecs

MIN_LHS_T_MFU_PERCENT = 90.0
MIN_RMSNORM_MATMUL_MFU_PERCENT = 85.0
MIN_ATTENTION_MFU_PERCENT = 44.0
MAX_RMSNORM_MATMUL_MFU_GAP_PERCENT = 3.0
ENDPOINT_PROFILE_TIMEOUT_SECONDS = 1800
SCHEDULER_OFF_ARGS = ("enable-linear-scan-allocation=false", "enable-instruction-scheduling=false")
GATE_ARTIFACT_DIRECTORY_ENV = "NKIGYM_GATE_ARTIFACT_DIRECTORY"
MFU_ENDPOINT_MANIFEST_ENV = "NKIGYM_MFU_ENDPOINT_MANIFEST"
EXPECTED_ENDPOINTS = {"attention", "matmul-lhs", "matmul-lhs-t", "rmsnorm-matmul"}


@dataclass(frozen=True)
class _Endpoint:
    """One rendered kernel supplied by the workflow gate."""

    name: str
    kernel: str
    func_name: str
    input_specs: InputSpecs


def _decode_input_specs(value: object) -> InputSpecs:
    """Decode and validate endpoint input specifications."""
    if not isinstance(value, dict) or not value:
        raise AssertionError("endpoint input_specs must be a non-empty object")
    input_specs: InputSpecs = {}
    for name, raw_spec in value.items():
        if not isinstance(name, str) or not name.isidentifier() or not isinstance(raw_spec, dict):
            raise AssertionError(f"invalid endpoint input specification: {name!r}")
        raw_shape = raw_spec.get("shape")
        dtype = raw_spec.get("dtype")
        if (
            not isinstance(raw_shape, list)
            or not raw_shape
            or any(
                not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 1
                for dimension in raw_shape
            )
        ):
            raise AssertionError(f"endpoint input {name!r} must have a positive integer shape")
        if not isinstance(dtype, str) or not dtype:
            raise AssertionError(f"endpoint input {name!r} must have a dtype")
        input_specs[name] = (tuple(raw_shape), dtype)
    return input_specs


def _decode_endpoint(value: object) -> _Endpoint:
    """Decode one endpoint manifest entry."""
    if not isinstance(value, dict):
        raise AssertionError("endpoint entry must be an object")
    name = value.get("name")
    kernel = value.get("kernel")
    func_name = value.get("func_name")
    if not isinstance(name, str) or not name:
        raise AssertionError("endpoint name must be a non-empty string")
    if not isinstance(kernel, str) or not kernel.strip():
        raise AssertionError(f"endpoint {name!r} kernel must be non-empty")
    if not isinstance(func_name, str) or not func_name.isidentifier():
        raise AssertionError(f"endpoint {name!r} func_name must be a Python identifier")
    return _Endpoint(
        name=name, kernel=kernel, func_name=func_name, input_specs=_decode_input_specs(value.get("input_specs"))
    )


def _load_endpoints() -> dict[str, _Endpoint]:
    """Load the controller-generated endpoint manifest."""
    configured_path = os.environ.get(MFU_ENDPOINT_MANIFEST_ENV)
    if configured_path is None:
        raise AssertionError(f"{MFU_ENDPOINT_MANIFEST_ENV} must identify the endpoint manifest")
    manifest_path = Path(configured_path).expanduser().resolve()
    decoded = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(decoded, dict) or decoded.get("schema_version") != 1:
        raise AssertionError(f"unsupported endpoint manifest: {manifest_path}")
    raw_endpoints = decoded.get("endpoints")
    if not isinstance(raw_endpoints, list):
        raise AssertionError("endpoint manifest entries must be a list")
    endpoints = {_endpoint.name: _endpoint for _endpoint in map(_decode_endpoint, raw_endpoints)}
    if set(endpoints) != EXPECTED_ENDPOINTS or len(endpoints) != len(raw_endpoints):
        raise AssertionError(
            f"endpoint manifest names must be exactly {sorted(EXPECTED_ENDPOINTS)}, got {sorted(endpoints)}"
        )
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


def test_best_known_generated_kernels_do_not_regress_mfu(tmp_path: Path) -> None:
    """Best-known generated endpoints retain their established hardware MFU."""
    host = os.environ.get("NKI_PROFILE_HOST", "gym-1")
    configured_directory = os.environ.get(GATE_ARTIFACT_DIRECTORY_ENV)
    cache_dir = Path(configured_directory) if configured_directory is not None else tmp_path
    endpoints = _load_endpoints()

    lhs_t_mfu = _validated_mfu(
        "lhsT", _profile_endpoint(host, endpoints["matmul-lhs-t"], cache_dir / "matmul_lhs_t_rhs")
    )
    lhs_mfu = _validated_mfu("lhs", _profile_endpoint(host, endpoints["matmul-lhs"], cache_dir / "matmul_lhs_rhs"))
    rmsnorm_mfu = _validated_mfu(
        "RMSNorm+matmul", _profile_endpoint(host, endpoints["rmsnorm-matmul"], cache_dir / "rmsnorm_matmul")
    )
    attention_mfu = _validated_mfu(
        "attention", _profile_endpoint(host, endpoints["attention"], cache_dir / "attention")
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
