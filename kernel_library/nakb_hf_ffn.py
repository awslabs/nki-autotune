"""Measured NAKB ``hf_ffn`` workload targets."""

from __future__ import annotations

import ml_dtypes
import numpy as np
import torch

from kernel_library import TorchReference
from nkigym.profile import InputSpecs


def hf_ffn_nonorm_sm_torch_ref(
    x_hf: torch.Tensor, w_in_hf: torch.Tensor, w_out_hf: torch.Tensor, b_in_col: torch.Tensor
) -> torch.Tensor:
    """Head-first FFN forward without normalization."""
    f32 = torch.float32
    x_hf = x_hf.to(f32)
    w_in_hf = w_in_hf.to(f32)
    w_out_hf = w_out_hf.to(f32)
    b_in = b_in_col.to(f32).reshape(1, -1)
    x_sm = x_hf.transpose(0, 1)
    h = x_sm @ w_in_hf + b_in
    h = torch.nn.functional.gelu(h, approximate="tanh")
    out = h @ w_out_hf
    return out


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


_torch_ref_0 = TorchReference(hf_ffn_nonorm_sm_torch_ref, ("x_hf", "w_in_hf", "w_out_hf", "b_in_col"))


WORKLOADS = (
    {
        "torch_ref": _torch_ref_0,
        "input_specs": {
            "x_hf": ((4096, 768), "bfloat16"),
            "w_in_hf": ((4096, 16384), "bfloat16"),
            "w_out_hf": ((16384, 4096), "bfloat16"),
            "b_in_col": ((16384, 1), "bfloat16"),
        },
        "input_generator": _input_generator,
        "atol": 1e-05,
        "rtol": 1e-05,
        "nakb_latency_ms": 1.393264489,
        "best_historical_latency_ms": 1.393264489,
    },
)
