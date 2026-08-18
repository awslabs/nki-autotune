"""Measured NAKB ``cumsum`` workload targets."""

from __future__ import annotations

import ml_dtypes
import numpy as np
import torch

from kernel_library import TorchReference
from nkigym.profile import InputSpecs


def cumsum_torch_ref(x: torch.Tensor, axis: int = -1) -> torch.Tensor:
    """PyTorch reference implementation of cumulative sum."""
    return {"output_0": torch.cumsum(x, dim=axis)}


def _input_generator(input_specs: InputSpecs, seed: int) -> dict[str, np.ndarray]:
    """Generate deterministic arrays matching one strict workload contract."""
    rng = np.random.default_rng(seed)
    special_dtypes = {"bfloat16": np.dtype(ml_dtypes.bfloat16), "float8_e4m3": np.dtype(ml_dtypes.float8_e4m3)}
    inputs: dict[str, np.ndarray] = {}
    for name, (shape, dtype_name) in input_specs.items():
        dtype = special_dtypes.get(dtype_name)
        if dtype is None:
            dtype = np.dtype(dtype_name)
        if np.issubdtype(dtype, np.bool_):
            values = np.ones(shape, dtype=np.bool_)
        elif np.issubdtype(dtype, np.integer):
            values = np.zeros(shape, dtype=dtype)
        elif name == "cos" or "weight" in name or "gamma" in name or "scale" in name:
            values = np.ones(shape, dtype=np.float32)
        elif name == "sin" or "bias" in name:
            values = np.zeros(shape, dtype=np.float32)
        elif name == "expert_affinities":
            values = rng.random(shape, dtype=np.float32)
            denominator = np.sum(values, axis=-1, keepdims=True)
            values = np.divide(values, denominator, out=np.zeros_like(values), where=denominator != 0)
        else:
            values = rng.standard_normal(shape, dtype=np.float32) * 0.1
        inputs[name] = values.astype(dtype)
    return inputs


_torch_ref_0 = TorchReference(cumsum_torch_ref, ("x",), bound_kwargs={"axis": -1})


WORKLOADS = (
    {
        "torch_ref": _torch_ref_0,
        "input_specs": {"x": ((1, 256), "float32")},
        "input_generator": _input_generator,
        "atol": 1e-05,
        "rtol": 1e-05,
        "nakb_latency_ms": 0.013543313,
        "best_historical_latency_ms": 0.013543313,
    },
    {
        "torch_ref": _torch_ref_0,
        "input_specs": {"x": ((128, 2048), "float32")},
        "input_generator": _input_generator,
        "atol": 1e-05,
        "rtol": 1e-05,
        "nakb_latency_ms": 0.024991628,
        "best_historical_latency_ms": 0.024991628,
    },
    {
        "torch_ref": _torch_ref_0,
        "input_specs": {"x": ((2048, 4096), "float32")},
        "input_generator": _input_generator,
        "atol": 1e-05,
        "rtol": 1e-05,
        "nakb_latency_ms": 0.224320483,
        "best_historical_latency_ms": 0.132285627,
    },
    {
        "torch_ref": _torch_ref_0,
        "input_specs": {"x": ((256, 10, 8192), "bfloat16")},
        "input_generator": _input_generator,
        "atol": 1e-05,
        "rtol": 1e-05,
        "nakb_latency_ms": 0.422968506,
        "best_historical_latency_ms": 0.422968506,
    },
    {
        "torch_ref": _torch_ref_0,
        "input_specs": {"x": ((64, 4, 1024), "bfloat16")},
        "input_generator": _input_generator,
        "atol": 1e-05,
        "rtol": 1e-05,
        "nakb_latency_ms": 0.020699967,
        "best_historical_latency_ms": 0.020699967,
    },
)
