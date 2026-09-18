"""Measured NAKB ``cascaded_max`` workload targets."""

from __future__ import annotations

import ml_dtypes
import numpy as np
import torch

from benchmark import AccuracySpec, InputSpecs, TorchReference


def cascaded_max_torch_ref(input_tensor: torch.Tensor) -> dict[str, torch.Tensor]:
    """Cascaded max torch reference implementation."""
    max_values, max_indices = torch.max(input_tensor, dim=-1, keepdim=True)
    return {"max_values": max_values, "max_indices": max_indices.to(torch.int32)}


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


_torch_ref_0 = TorchReference(cascaded_max_torch_ref, ("input_tensor",))


WORKLOADS = (
    {
        "accuracy": AccuracySpec(atol=1e-05, rtol=1e-05, mode="global_max", output_indices=None, output_tolerances=()),
        "torch_ref": _torch_ref_0,
        "input_specs": {"input_tensor": ((1, 1, 16000), "float32")},
        "input_generator": _input_generator,
        "nakb_latency_ms": 0.017452472,
    },
    {
        "accuracy": AccuracySpec(atol=1e-05, rtol=1e-05, mode="global_max", output_indices=None, output_tolerances=()),
        "torch_ref": _torch_ref_0,
        "input_specs": {"input_tensor": ((1, 1, 256), "float32")},
        "input_generator": _input_generator,
        "nakb_latency_ms": 0.014358311,
    },
    {
        "accuracy": AccuracySpec(atol=1e-05, rtol=1e-05, mode="global_max", output_indices=None, output_tolerances=()),
        "torch_ref": _torch_ref_0,
        "input_specs": {"input_tensor": ((1, 1, 3168), "float32")},
        "input_generator": _input_generator,
        "nakb_latency_ms": 0.015344143,
    },
    {
        "accuracy": AccuracySpec(atol=1e-05, rtol=1e-05, mode="global_max", output_indices=None, output_tolerances=()),
        "torch_ref": _torch_ref_0,
        "input_specs": {"input_tensor": ((1, 7, 3999), "float32")},
        "input_generator": _input_generator,
        "nakb_latency_ms": 0.023373297,
    },
    {
        "accuracy": AccuracySpec(atol=1e-05, rtol=1e-05, mode="global_max", output_indices=None, output_tolerances=()),
        "torch_ref": _torch_ref_0,
        "input_specs": {"input_tensor": ((128, 1, 3168), "float32")},
        "input_generator": _input_generator,
        "nakb_latency_ms": 0.02359163,
    },
    {
        "accuracy": AccuracySpec(atol=1e-05, rtol=1e-05, mode="global_max", output_indices=None, output_tolerances=()),
        "torch_ref": _torch_ref_0,
        "input_specs": {"input_tensor": ((4, 5, 8192), "float32")},
        "input_generator": _input_generator,
        "nakb_latency_ms": 0.0210808,
    },
    {
        "accuracy": AccuracySpec(atol=1e-05, rtol=1e-05, mode="global_max", output_indices=None, output_tolerances=()),
        "torch_ref": _torch_ref_0,
        "input_specs": {"input_tensor": ((8, 5, 4058), "float32")},
        "input_generator": _input_generator,
        "nakb_latency_ms": 0.021144967,
    },
    {
        "accuracy": AccuracySpec(atol=1e-05, rtol=1e-05, mode="global_max", output_indices=None, output_tolerances=()),
        "torch_ref": _torch_ref_0,
        "input_specs": {"input_tensor": ((8, 5, 8192), "float32")},
        "input_generator": _input_generator,
        "nakb_latency_ms": 0.027949956,
    },
)
