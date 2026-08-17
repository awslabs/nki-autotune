"""Measured NAKB ``shared_expert_mlp_bwd`` workload targets."""

from __future__ import annotations

import ml_dtypes
import numpy as np
import torch

from kernel_library import TorchReference
from nkigym.profile import InputSpecs


def shared_expert_mlp_bwd_torch_ref(
    hidden_states: torch.Tensor,
    up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    relu_up: torch.Tensor,
    output_grad: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Reference backward for the squared-ReLU dense shared-expert MLP."""
    in_dtype = hidden_states.dtype
    x = hidden_states.to(torch.float32)
    w_up = up_proj_weight.to(torch.float32)
    w_down = down_proj_weight.to(torch.float32)
    r = relu_up.to(torch.float32)
    d_out = output_grad.to(torch.float32)
    a = r * r
    d_a = d_out @ w_down.T
    down_proj_weight_grad = a.T @ d_out
    d_u = d_a * (2.0 * r)
    hidden_states_grad = d_u @ w_up.T
    up_proj_weight_grad = x.T @ d_u
    return {
        "hidden_states_grad": hidden_states_grad.to(in_dtype),
        "up_proj_weight_grad": up_proj_weight_grad.to(in_dtype),
        "down_proj_weight_grad": down_proj_weight_grad.to(in_dtype),
    }


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
    shared_expert_mlp_bwd_torch_ref, ("hidden_states", "up_proj_weight", "down_proj_weight", "relu_up", "output_grad")
)


WORKLOADS = (
    {
        "torch_ref": _torch_ref_0,
        "input_specs": {
            "hidden_states": ((1024, 8192), "bfloat16"),
            "up_proj_weight": ((8192, 512), "bfloat16"),
            "down_proj_weight": ((512, 8192), "bfloat16"),
            "relu_up": ((1024, 512), "bfloat16"),
            "output_grad": ((1024, 8192), "bfloat16"),
        },
        "input_generator": _input_generator,
        "atol": 1e-05,
        "rtol": 1e-05,
        "nakb_latency_ms": 0.322401163,
        "best_historical_latency_ms": 0.322401163,
    },
)
