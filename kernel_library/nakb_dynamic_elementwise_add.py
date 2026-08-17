"""Measured NAKB ``dynamic_elementwise_add`` workload target."""

from __future__ import annotations

import numpy as np
import torch

from kernel_library import TorchReference
from nkigym.profile import InputSpecs

_TOKENS = 512

_HIDDEN = 256


def dynamic_elementwise_add_torch_ref(
    input_a: torch.Tensor, input_b: torch.Tensor, num_m_tiles: torch.Tensor | None = None
) -> torch.Tensor:
    """PyTorch reference implementation of dynamic elementwise addition."""
    return input_a + input_b


def _input_generator(input_specs: InputSpecs, seed: int) -> dict[str, np.ndarray]:
    """Generate replayable Gaussian source tensors."""
    rng = np.random.default_rng(seed)
    return {name: rng.standard_normal(shape).astype(np.float32) for name, (shape, _dtype) in input_specs.items()}


_torch_ref = TorchReference(
    dynamic_elementwise_add_torch_ref, ("input_a", "input_b"), bound_kwargs={"num_m_tiles": None}
)


WORKLOADS = (
    {
        "torch_ref": _torch_ref,
        "input_specs": {"input_a": ((_TOKENS, _HIDDEN), "float32"), "input_b": ((_TOKENS, _HIDDEN), "float32")},
        "input_generator": _input_generator,
        "atol": 1e-3,
        "rtol": 1e-3,
        "nakb_latency_ms": 0.019116637,
        "best_historical_latency_ms": 0.019116637,
    },
)
