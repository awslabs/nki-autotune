"""M2048/K2048/N2048 row-major matrix multiplication workload."""

from __future__ import annotations

import numpy as np

from nkigym.profile import InputSpecs


def _numpy_ref(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Compute ``lhs @ rhs`` in FP32."""
    return lhs.astype(np.float32) @ rhs.astype(np.float32)


def _input_generator(input_specs: InputSpecs, seed: int) -> dict[str, np.ndarray]:
    """Generate uniform FP32 matrix inputs."""
    rng = np.random.default_rng(seed)
    return {name: rng.random(shape).astype(np.float32) for name, (shape, _dtype) in input_specs.items()}


WORKLOAD = {
    "numpy_ref": _numpy_ref,
    "input_specs": {"lhs": ((2048, 2048), "bfloat16"), "rhs": ((2048, 2048), "bfloat16")},
    "input_generator": _input_generator,
    "atol": 1e-3,
    "rtol": 1e-3,
    "best_historical_latency_ms": 0.24978,
}
