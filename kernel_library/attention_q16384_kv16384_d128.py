"""Q16384/KV16384/D128 scaled dot-product attention workload."""

from __future__ import annotations

import numpy as np

from nkigym.profile import InputSpecs

_HEAD_DIM = 128


def _input_generator(input_specs: InputSpecs, seed: int) -> dict[str, np.ndarray]:
    """Generate uniform FP32 attention inputs."""
    rng = np.random.default_rng(seed)
    return {name: rng.random(shape).astype(np.float32) for name, (shape, _dtype) in input_specs.items()}


def _numpy_ref(query: np.ndarray, key: np.ndarray, value: np.ndarray) -> np.ndarray:
    """Compute scaled dot-product attention in FP32."""
    scores = query.astype(np.float32).T @ key.astype(np.float32) / np.sqrt(_HEAD_DIM)
    scores -= np.max(scores, axis=1, keepdims=True)
    probabilities = np.exp(scores)
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    return probabilities @ value.astype(np.float32)


WORKLOAD = {
    "numpy_ref": _numpy_ref,
    "input_specs": {
        "query": ((_HEAD_DIM, 16384), "bfloat16"),
        "key": ((_HEAD_DIM, 16384), "bfloat16"),
        "value": ((16384, _HEAD_DIM), "bfloat16"),
    },
    "input_generator": _input_generator,
    "atol": 1e-5,
    "rtol": 2e-2,
    "best_historical_latency_ms": 3.7639,
}
