"""M2048/K2048/N2048 RMSNorm followed by matrix multiplication workload."""

from __future__ import annotations

import numpy as np

from nkigym.profile import InputSpecs

_EPSILON = 1e-6


def _input_generator(input_specs: InputSpecs, seed: int) -> dict[str, np.ndarray]:
    """Generate small uniform FP32 RMSNorm inputs."""
    rng = np.random.default_rng(seed)
    return {
        name: rng.uniform(-0.1, 0.1, size=shape).astype(np.float32) for name, (shape, _dtype) in input_specs.items()
    }


def _numpy_ref(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Compute row-wise RMSNorm followed by matmul in FP32."""
    lhs_fp32 = lhs.astype(np.float32)
    normalized = lhs_fp32 / np.sqrt(np.mean(np.square(lhs_fp32), axis=1, keepdims=True) + _EPSILON)
    return normalized @ rhs.astype(np.float32)


WORKLOAD = {
    "numpy_ref": _numpy_ref,
    "input_specs": {"lhs": ((2048, 2048), "bfloat16"), "rhs": ((2048, 2048), "bfloat16")},
    "input_generator": _input_generator,
    "atol": 1e-3,
    "rtol": 2e-2,
    "best_historical_latency_ms": 0.25112,
}
