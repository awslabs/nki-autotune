"""Measured NAKB ``output_projection_tkg`` workload targets."""

from __future__ import annotations

from enum import IntEnum

import ml_dtypes
import numpy as np
import torch

from kernel_library import TorchReference
from nkigym.profile import InputSpecs

_FP8_E4M3_MAX = 240.0


class QuantizationType(IntEnum):
    """NAKB quantization choices."""

    NONE = 0
    STATIC = 1
    ROW = 2
    MX = 3
    STATIC_MX = 4
    ROW_MX = 5


def output_projection_tkg_torch_ref(
    attention: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    quantization_type: QuantizationType = QuantizationType.NONE,
    weight_scale: torch.Tensor | None = None,
    input_scale: torch.Tensor | None = None,
    TRANSPOSE_OUT: bool = False,
    OUT_IN_SB: bool = False,
    sbm: object | None = None,
) -> dict[str, torch.Tensor]:
    """PyTorch reference implementation of output projection for TKG."""
    if quantization_type not in (QuantizationType.NONE, QuantizationType.STATIC, QuantizationType.ROW):
        raise ValueError(f"Unsupported registered quantization type: {quantization_type}")
    head_dimension, batch_size, heads, sequence_length = attention.shape
    hidden = weight.shape[1]
    attention = attention.float()
    weight = weight.float()
    attn = attention.permute(1, 3, 2, 0).reshape(batch_size * sequence_length, heads * head_dimension)
    if quantization_type == QuantizationType.STATIC:
        weight_multiplier = weight_scale[0, 0].float()
        input_multiplier = input_scale[0, 0].float()
        attn = torch.clamp(attn / input_multiplier, -_FP8_E4M3_MAX, _FP8_E4M3_MAX)
    elif quantization_type == QuantizationType.ROW:
        weight_multiplier = weight_scale[0, :].float()
    out = attn @ weight
    if quantization_type == QuantizationType.STATIC:
        out = out * (weight_multiplier * input_multiplier)
    elif quantization_type == QuantizationType.ROW:
        out = out * weight_multiplier
    if bias is not None:
        out = out + bias.float()
    if TRANSPOSE_OUT:
        partition = 128
        lnc = 2 if hidden % (2 * partition) == 0 else 1
        out = out.reshape(batch_size * sequence_length, lnc, partition, hidden // (lnc * partition)).permute(2, 1, 3, 0)
    return {"out": out}


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
    output_projection_tkg_torch_ref,
    ("attention", "weight", "bias"),
    bound_kwargs={
        "quantization_type": 0,
        "weight_scale": None,
        "input_scale": None,
        "TRANSPOSE_OUT": True,
        "OUT_IN_SB": False,
        "sbm": None,
    },
)
_torch_ref_1 = TorchReference(
    output_projection_tkg_torch_ref,
    ("attention", "weight", "bias", "weight_scale"),
    bound_kwargs={"quantization_type": 2, "input_scale": None, "TRANSPOSE_OUT": False, "OUT_IN_SB": False, "sbm": None},
)
_torch_ref_2 = TorchReference(
    output_projection_tkg_torch_ref,
    ("attention", "weight", "bias", "weight_scale", "input_scale"),
    bound_kwargs={"quantization_type": 1, "TRANSPOSE_OUT": True, "OUT_IN_SB": False, "sbm": None},
)
_torch_ref_3 = TorchReference(
    output_projection_tkg_torch_ref,
    ("attention", "weight", "bias"),
    bound_kwargs={
        "quantization_type": 0,
        "weight_scale": None,
        "input_scale": None,
        "TRANSPOSE_OUT": False,
        "OUT_IN_SB": False,
        "sbm": None,
    },
)
_torch_ref_4 = TorchReference(
    output_projection_tkg_torch_ref,
    ("attention", "weight"),
    bound_kwargs={
        "bias": None,
        "quantization_type": 0,
        "weight_scale": None,
        "input_scale": None,
        "TRANSPOSE_OUT": False,
        "OUT_IN_SB": False,
        "sbm": None,
    },
)


WORKLOADS = (
    {
        "torch_ref": _torch_ref_0,
        "input_specs": {
            "attention": ((128, 128, 8, 3), "bfloat16"),
            "weight": ((1024, 3072), "bfloat16"),
            "bias": ((1, 3072), "bfloat16"),
        },
        "input_generator": _input_generator,
        "atol": 0.05,
        "rtol": 0.05,
        "nakb_latency_ms": 0.054059083,
        "best_historical_latency_ms": 0.054059083,
    },
    {
        "torch_ref": _torch_ref_0,
        "input_specs": {
            "attention": ((128, 4, 8, 4), "bfloat16"),
            "weight": ((1024, 16384), "bfloat16"),
            "bias": ((1, 16384), "bfloat16"),
        },
        "input_generator": _input_generator,
        "atol": 0.05,
        "rtol": 0.05,
        "nakb_latency_ms": 0.117928149,
        "best_historical_latency_ms": 0.117928149,
    },
    {
        "torch_ref": _torch_ref_0,
        "input_specs": {
            "attention": ((32, 512, 8, 4), "bfloat16"),
            "weight": ((256, 3072), "bfloat16"),
            "bias": ((1, 3072), "bfloat16"),
        },
        "input_generator": _input_generator,
        "atol": 0.05,
        "rtol": 0.05,
        "nakb_latency_ms": 0.075299882,
        "best_historical_latency_ms": 0.075299882,
    },
    {
        "torch_ref": _torch_ref_1,
        "input_specs": {
            "attention": ((128, 16, 8, 1), "bfloat16"),
            "weight": ((1024, 3072), "float8_e4m3"),
            "bias": ((1, 3072), "bfloat16"),
            "weight_scale": ((128, 3072), "float32"),
        },
        "input_generator": _input_generator,
        "atol": 0.05,
        "rtol": 0.05,
        "nakb_latency_ms": 0.033562448,
        "best_historical_latency_ms": 0.033562448,
    },
    {
        "torch_ref": _torch_ref_2,
        "input_specs": {
            "attention": ((128, 16, 8, 1), "bfloat16"),
            "weight": ((1024, 8192), "float8_e4m3"),
            "bias": ((1, 8192), "bfloat16"),
            "weight_scale": ((128, 1), "float32"),
            "input_scale": ((128, 1), "float32"),
        },
        "input_generator": _input_generator,
        "atol": 0.05,
        "rtol": 0.05,
        "nakb_latency_ms": 0.053856582,
        "best_historical_latency_ms": 0.053856582,
    },
    {
        "torch_ref": _torch_ref_3,
        "input_specs": {
            "attention": ((128, 4, 2, 4), "bfloat16"),
            "weight": ((256, 16384), "bfloat16"),
            "bias": ((1, 16384), "bfloat16"),
        },
        "input_generator": _input_generator,
        "atol": 0.05,
        "rtol": 0.05,
        "nakb_latency_ms": 0.039242439,
        "best_historical_latency_ms": 0.039242439,
    },
    {
        "torch_ref": _torch_ref_3,
        "input_specs": {
            "attention": ((128, 4, 2, 4), "bfloat16"),
            "weight": ((256, 8192), "bfloat16"),
            "bias": ((1, 8192), "bfloat16"),
        },
        "input_generator": _input_generator,
        "atol": 0.05,
        "rtol": 0.05,
        "nakb_latency_ms": 0.03246995,
        "best_historical_latency_ms": 0.03246995,
    },
    {
        "torch_ref": _torch_ref_4,
        "input_specs": {"attention": ((64, 4, 8, 4), "bfloat16"), "weight": ((512, 3072), "bfloat16")},
        "input_generator": _input_generator,
        "atol": 0.05,
        "rtol": 0.05,
        "nakb_latency_ms": 0.03017162,
        "best_historical_latency_ms": 0.03017162,
    },
)
