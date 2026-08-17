"""Measured NAKB ``find_nonzero_indices_with_count`` workload targets."""

from __future__ import annotations

import ml_dtypes
import numpy as np
import torch

from kernel_library import TorchReference
from nkigym.profile import InputSpecs

PADDING_VALUE = -1


def find_nonzero_indices_with_count_torch_ref(input_tensor: torch.Tensor) -> dict[str, torch.Tensor]:
    """PyTorch reference for find_nonzero_indices_with_count kernel."""
    tokens = input_tensor.shape[-1]
    output = torch.full((1, tokens + 1), PADDING_VALUE, dtype=torch.int32)
    nonzero_indices = torch.nonzero(input_tensor[0], as_tuple=False).squeeze(-1)
    count = nonzero_indices.shape[0]
    if count > 0:
        output[0, :count] = nonzero_indices.to(torch.int32)
    output[0, -1] = count
    return {"output": output}


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


_torch_ref_0 = TorchReference(find_nonzero_indices_with_count_torch_ref, ("input_tensor",))


WORKLOADS = (
    {
        "torch_ref": _torch_ref_0,
        "input_specs": {"input_tensor": ((1, 128), "float32")},
        "input_generator": _input_generator,
        "atol": 0.0,
        "rtol": 0.0,
        "nakb_latency_ms": 0.012624147,
        "best_historical_latency_ms": 0.012624147,
    },
    {
        "torch_ref": _torch_ref_0,
        "input_specs": {"input_tensor": ((1, 256), "float32")},
        "input_generator": _input_generator,
        "atol": 0.0,
        "rtol": 0.0,
        "nakb_latency_ms": 0.01528331,
        "best_historical_latency_ms": 0.01528331,
    },
    {
        "torch_ref": _torch_ref_0,
        "input_specs": {"input_tensor": ((1, 64), "float32")},
        "input_generator": _input_generator,
        "atol": 0.0,
        "rtol": 0.0,
        "nakb_latency_ms": 0.014275811,
        "best_historical_latency_ms": 0.014275811,
    },
)
