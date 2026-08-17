"""Measured NAKB ``rope`` workload targets."""

from __future__ import annotations

import ml_dtypes
import numpy as np
import torch

from kernel_library import TorchReference
from nkigym.profile import InputSpecs


def rope_torch_ref(
    x_in: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    lnc_shard: bool = False,
    contiguous_layout: bool = True,
    relayout_in_sbuf: bool = False,
) -> dict[str, torch.Tensor]:
    """Torch reference for RoPE kernel."""
    d_head, batch_size, n_heads, _ = x_in.shape
    if d_head not in (64, 128):
        raise ValueError(f"[NCC_INKI016] Kernel validation exception: d_head must be 64 or 128, got {d_head}")
    x_out = torch.empty_like(x_in)
    for batch_index in range(batch_size):
        for head_index in range(n_heads):
            x_out[:, batch_index, head_index, :] = _rope_single_head(
                x_in[:, batch_index, head_index, :], cos[:, batch_index, :], sin[:, batch_index, :], contiguous_layout
            )
    return {"x_out": x_out}


def _rope_single_head(
    x_in: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, contiguous_layout: bool
) -> torch.Tensor:
    """Apply RoPE to a single head."""
    d_head = x_in.shape[0]
    x = x_in.T
    if contiguous_layout:
        new_x = torch.empty_like(x)
        new_x[:, ::2] = x[:, : d_head // 2]
        new_x[:, 1::2] = x[:, d_head // 2 :]
        x = new_x
    freqs_cos = cos.T
    freqs_sin = sin.T
    xri = x.reshape(x.shape[:-1] + (-1, 2))
    x_r, x_i = xri[..., 0], xri[..., 1]
    x_out_r = x_r * freqs_cos - x_i * freqs_sin
    x_out_i = x_r * freqs_sin + x_i * freqs_cos
    x_out = torch.stack([x_out_r, x_out_i], dim=-1).reshape(x.shape)
    if contiguous_layout:
        x_out = torch.cat((x_out[:, 0::2], x_out[:, 1::2]), dim=1)
    return x_out.T


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
    rope_torch_ref,
    ("x_in", "cos", "sin"),
    bound_kwargs={"lnc_shard": True, "contiguous_layout": False, "relayout_in_sbuf": False},
)
_torch_ref_1 = TorchReference(
    rope_torch_ref,
    ("x_in", "cos", "sin"),
    bound_kwargs={"lnc_shard": True, "contiguous_layout": True, "relayout_in_sbuf": False},
)


WORKLOADS = (
    {
        "torch_ref": _torch_ref_0,
        "input_specs": {
            "x_in": ((128, 32, 4, 64), "bfloat16"),
            "cos": ((64, 32, 64), "bfloat16"),
            "sin": ((64, 32, 64), "bfloat16"),
        },
        "input_generator": _input_generator,
        "atol": 1e-05,
        "rtol": 1e-05,
        "nakb_latency_ms": 0.068329893,
        "best_historical_latency_ms": 0.068329893,
    },
    {
        "torch_ref": _torch_ref_1,
        "input_specs": {
            "x_in": ((128, 1, 8, 16), "bfloat16"),
            "cos": ((64, 1, 16), "bfloat16"),
            "sin": ((64, 1, 16), "bfloat16"),
        },
        "input_generator": _input_generator,
        "atol": 1e-05,
        "rtol": 1e-05,
        "nakb_latency_ms": 0.020464968,
        "best_historical_latency_ms": 0.020464968,
    },
    {
        "torch_ref": _torch_ref_1,
        "input_specs": {
            "x_in": ((128, 64, 8, 128), "bfloat16"),
            "cos": ((64, 64, 128), "bfloat16"),
            "sin": ((64, 64, 128), "bfloat16"),
        },
        "input_generator": _input_generator,
        "atol": 1e-05,
        "rtol": 1e-05,
        "nakb_latency_ms": 0.223437984,
        "best_historical_latency_ms": 0.223437984,
    },
    {
        "torch_ref": _torch_ref_1,
        "input_specs": {
            "x_in": ((64, 64, 8, 128), "bfloat16"),
            "cos": ((32, 64, 128), "bfloat16"),
            "sin": ((32, 64, 128), "bfloat16"),
        },
        "input_generator": _input_generator,
        "atol": 1e-05,
        "rtol": 1e-05,
        "nakb_latency_ms": 0.176862224,
        "best_historical_latency_ms": 0.176862224,
    },
    {
        "torch_ref": _torch_ref_1,
        "input_specs": {
            "x_in": ((64, 8, 1, 128), "bfloat16"),
            "cos": ((32, 8, 128), "bfloat16"),
            "sin": ((32, 8, 128), "bfloat16"),
        },
        "input_generator": _input_generator,
        "atol": 1e-05,
        "rtol": 1e-05,
        "nakb_latency_ms": 0.018605804,
        "best_historical_latency_ms": 0.018605804,
    },
)
