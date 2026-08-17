"""Measured NAKB ``argsort_unstable`` workload targets."""

from __future__ import annotations

import ml_dtypes
import numpy as np
import torch

from kernel_library import TorchReference
from nkigym.profile import InputSpecs

_ELEMS_PER_PASS = 8


def argsort_unstable_torch_ref(
    data: torch.Tensor, descending: bool = False, output_in_sbuf: bool = False
) -> torch.Tensor:
    """Argsort unstable, matching ordering produced by argsort_unstable kernel."""
    data_f32 = data.flatten().float().clone()
    elements = data_f32.shape[0]
    num_passes = elements // _ELEMS_PER_PASS
    indices = torch.zeros(elements, dtype=torch.int32)
    for pass_idx in range(num_passes):
        top_vals, _ = torch.topk(data_f32, _ELEMS_PER_PASS, largest=True, sorted=True)
        pass_indices = torch.zeros(_ELEMS_PER_PASS, dtype=torch.int32)
        for val_idx in range(_ELEMS_PER_PASS - 1, -1, -1):
            val = top_vals[val_idx]
            matches = torch.where(data_f32 == val)[0]
            pos = matches[0].item()
            pass_indices[val_idx] = pos
            data_f32[pos] = float("-inf")
        if descending:
            start = _ELEMS_PER_PASS * pass_idx
            indices[start : start + _ELEMS_PER_PASS] = pass_indices
        else:
            start = _ELEMS_PER_PASS * (num_passes - pass_idx) - 1
            for elem_idx in range(_ELEMS_PER_PASS):
                indices[start - elem_idx] = pass_indices[elem_idx]
    return indices.unsqueeze(0)


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


_torch_ref_0 = TorchReference(
    argsort_unstable_torch_ref, ("data",), bound_kwargs={"descending": True, "output_in_sbuf": False}
)
_torch_ref_1 = TorchReference(
    argsort_unstable_torch_ref, ("data",), bound_kwargs={"descending": False, "output_in_sbuf": False}
)


WORKLOADS = (
    {
        "torch_ref": _torch_ref_0,
        "input_specs": {"data": ((1, 128), "float32")},
        "input_generator": _input_generator,
        "atol": 0.0,
        "rtol": 0.0,
        "nakb_latency_ms": 0.026384126,
        "best_historical_latency_ms": 0.026384126,
    },
    {
        "torch_ref": _torch_ref_0,
        "input_specs": {"data": ((1, 32), "float32")},
        "input_generator": _input_generator,
        "atol": 0.0,
        "rtol": 0.0,
        "nakb_latency_ms": 0.015657476,
        "best_historical_latency_ms": 0.015657476,
    },
    {
        "torch_ref": _torch_ref_1,
        "input_specs": {"data": ((1, 32), "float32")},
        "input_generator": _input_generator,
        "atol": 0.0,
        "rtol": 0.0,
        "nakb_latency_ms": 0.014759144,
        "best_historical_latency_ms": 0.014759144,
    },
    {
        "torch_ref": _torch_ref_1,
        "input_specs": {"data": ((1, 512), "float32")},
        "input_generator": _input_generator,
        "atol": 0.0,
        "rtol": 0.0,
        "nakb_latency_ms": 0.117943149,
        "best_historical_latency_ms": 0.117943149,
    },
)
