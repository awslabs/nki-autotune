"""Measured NAKB ``conv1d`` workload targets."""

from __future__ import annotations

from enum import IntEnum

import ml_dtypes
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from kernel_library import TorchReference
from nkigym.profile import InputSpecs


class ActFnType(IntEnum):
    """NAKB activation choices for convolution references."""

    SiLU = 0
    GELU = 1
    GELU_Tanh_Approx = 2
    Swish = 3
    ReLU = 4


def conv1d_torch_ref(
    x_in: torch.Tensor,
    filters: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: int = 1,
    padding: tuple[int, int] = (0, 0),
    dilation: int = 1,
    activation_fn: ActFnType | None = None,
    lnc_shard: bool = False,
) -> dict[str, torch.Tensor]:
    """PyTorch reference implementation of the NAKB 1D convolution."""
    kernel_size, input_channels, output_channels = filters.shape
    pad_left, pad_right = padding
    if pad_left != pad_right:
        x_in = F.pad(x_in, (pad_left, pad_right), mode="constant", value=0)
        conv_padding = 0
    else:
        conv_padding = pad_left
    conv = nn.Conv1d(
        in_channels=input_channels,
        out_channels=output_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=conv_padding,
        dilation=dilation,
        bias=bias is not None,
    ).to(x_in.dtype)
    with torch.no_grad():
        conv.weight.copy_(filters.permute(2, 1, 0))
        if bias is not None:
            conv.bias.copy_(bias)
    output = conv(x_in)
    if activation_fn is not None:
        if activation_fn == ActFnType.SiLU:
            output = F.silu(output)
        elif activation_fn == ActFnType.GELU:
            output = F.gelu(output)
        elif activation_fn == ActFnType.GELU_Tanh_Approx:
            output = F.gelu(output, approximate="tanh")
        elif activation_fn == ActFnType.Swish:
            output = F.silu(output)
        elif activation_fn == ActFnType.ReLU:
            output = F.relu(output)
        else:
            raise ValueError(f"Unsupported activation function: {activation_fn}")
    return {"out": output.detach()}


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
    conv1d_torch_ref,
    ("x_in", "filters", "bias"),
    bound_kwargs={"stride": 1, "padding": (2, 2), "dilation": 1, "activation_fn": 1, "lnc_shard": False},
)
_torch_ref_1 = TorchReference(
    conv1d_torch_ref,
    ("x_in", "filters", "bias"),
    bound_kwargs={"stride": 2, "padding": (2, 2), "dilation": 2, "activation_fn": 1, "lnc_shard": False},
)
_torch_ref_2 = TorchReference(
    conv1d_torch_ref,
    ("x_in", "filters", "bias"),
    bound_kwargs={"stride": 1, "padding": (1, 1), "dilation": 1, "activation_fn": 1, "lnc_shard": False},
)
_torch_ref_3 = TorchReference(
    conv1d_torch_ref,
    ("x_in", "filters"),
    bound_kwargs={
        "bias": None,
        "stride": 1,
        "padding": (0, 0),
        "dilation": 1,
        "activation_fn": None,
        "lnc_shard": False,
    },
)
_torch_ref_4 = TorchReference(
    conv1d_torch_ref,
    ("x_in", "filters"),
    bound_kwargs={
        "bias": None,
        "stride": 1,
        "padding": (1, 1),
        "dilation": 1,
        "activation_fn": None,
        "lnc_shard": False,
    },
)


WORKLOADS = (
    {
        "torch_ref": _torch_ref_0,
        "input_specs": {
            "x_in": ((1, 64, 100), "float32"),
            "filters": ((5, 64, 128), "float32"),
            "bias": ((128,), "float32"),
        },
        "input_generator": _input_generator,
        "atol": 1e-05,
        "rtol": 1e-05,
        "nakb_latency_ms": 0.020219135,
        "best_historical_latency_ms": 0.020219135,
    },
    {
        "torch_ref": _torch_ref_1,
        "input_specs": {
            "x_in": ((2, 64, 128), "float32"),
            "filters": ((5, 64, 128), "float32"),
            "bias": ((128,), "float32"),
        },
        "input_generator": _input_generator,
        "atol": 1e-05,
        "rtol": 1e-05,
        "nakb_latency_ms": 0.0212383,
        "best_historical_latency_ms": 0.0212383,
    },
    {
        "torch_ref": _torch_ref_2,
        "input_specs": {
            "x_in": ((2, 80, 1500), "float32"),
            "filters": ((3, 80, 384), "float32"),
            "bias": ((384,), "float32"),
        },
        "input_generator": _input_generator,
        "atol": 1e-05,
        "rtol": 1e-05,
        "nakb_latency_ms": 0.116704817,
        "best_historical_latency_ms": 0.116704817,
    },
    {
        "torch_ref": _torch_ref_3,
        "input_specs": {"x_in": ((1, 16, 20), "float32"), "filters": ((3, 16, 32), "float32")},
        "input_generator": _input_generator,
        "atol": 1e-05,
        "rtol": 1e-05,
        "nakb_latency_ms": 0.017380807,
        "best_historical_latency_ms": 0.017380807,
    },
    {
        "torch_ref": _torch_ref_4,
        "input_specs": {"x_in": ((1, 256, 200), "float32"), "filters": ((3, 256, 256), "float32")},
        "input_generator": _input_generator,
        "atol": 1e-05,
        "rtol": 1e-05,
        "nakb_latency_ms": 0.02552996,
        "best_historical_latency_ms": 0.02552996,
    },
)
