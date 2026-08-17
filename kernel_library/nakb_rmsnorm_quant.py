"""Measured NAKB ``rmsnorm_quant`` workload targets."""

from __future__ import annotations

from enum import IntEnum
from types import SimpleNamespace

import ml_dtypes
import numpy as np
import torch

from kernel_library import TorchReference
from nkigym.profile import InputSpecs


class NormType(IntEnum):
    """NAKB normalization choices."""

    NO_NORM = 0
    RMS_NORM = 1
    LAYER_NORM = 2
    RMS_NORM_SKIP_GAMMA = 3


class QuantizationType(IntEnum):
    """NAKB quantization choices."""

    NONE = 0
    STATIC = 1
    ROW = 2
    MX = 3
    STATIC_MX = 4
    ROW_MX = 5


def rmsnorm_quant_torch_ref(
    hidden: torch.Tensor,
    ln_w: torch.Tensor,
    kargs: SimpleNamespace,
    input_dequant_scale: torch.Tensor | None = None,
    pre_norm_gamma: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
) -> dict[str, np.ndarray | None]:
    """Torch reference for the NAKB RMSNorm-Quant kernel."""
    fp8_range = 240.0
    inp = hidden.numpy().astype(np.float32)
    gamma = ln_w.numpy().astype(np.float32)
    if input_dequant_scale is not None:
        in_dq_scale = input_dequant_scale.numpy().astype(np.float32)
    else:
        in_dq_scale = None
    quant_only = kargs.norm_type == NormType.NO_NORM
    quant_type = kargs.quantization_type
    eps = kargs.eps
    lower_bound = kargs.lower_bound

    if pre_norm_gamma is not None:
        pre_norm_weight = pre_norm_gamma.numpy().astype(np.float32)
        rms = np.sqrt(np.mean(np.square(inp), axis=-1, keepdims=True) + eps)
        inp = inp * np.reciprocal(rms)
        inp *= pre_norm_weight

    residual_out = None
    if residual is not None:
        residual_values = residual.numpy().astype(np.float32)
        inp = inp + residual_values
        residual_out = inp.copy()

    if quant_only:
        norm = inp
    else:
        rms = np.sqrt(np.mean(np.square(inp), axis=-1, keepdims=True) + eps)
        norm = inp * np.reciprocal(rms)
        norm *= gamma
    if quant_type == QuantizationType.ROW:
        norm_abs_max = np.abs(norm).max(axis=-1, keepdims=True)
        if lower_bound > 0:
            norm_abs_max = np.clip(norm_abs_max, a_min=None, a_max=lower_bound)
            norm = np.clip(norm, a_min=-lower_bound, a_max=lower_bound)
        dequant_scale = norm_abs_max / fp8_range
        quant_scale = np.reciprocal(dequant_scale)
        norm_quant = norm * quant_scale
        dequant_scale = dequant_scale.astype(np.float32)
    elif quant_type == QuantizationType.STATIC:
        if in_dq_scale is None:
            raise ValueError("input_dequant_scale is required for static quantization")
        quant_scale = np.reciprocal(in_dq_scale[0, 0])
        norm = norm * quant_scale
        norm_quant = np.clip(norm, a_min=-fp8_range, a_max=fp8_range)
        dequant_scale = None
    else:
        raise ValueError(f"Unsupported quantization type: {quant_type}")
    norm_quant = norm_quant.astype(ml_dtypes.float8_e4m3)
    return {"norm_quant": norm_quant, "dequant_scale": dequant_scale, "residual_out": residual_out}


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
    rmsnorm_quant_torch_ref,
    ("hidden", "ln_w"),
    bound_kwargs={
        "kargs": SimpleNamespace(
            quantization_type=QuantizationType.ROW, lower_bound=0.0, norm_type=NormType.NO_NORM, eps=1e-6
        ),
        "input_dequant_scale": None,
        "pre_norm_gamma": None,
        "residual": None,
    },
)
_torch_ref_1 = TorchReference(
    rmsnorm_quant_torch_ref,
    ("hidden", "ln_w", "input_dequant_scale"),
    bound_kwargs={
        "kargs": SimpleNamespace(
            quantization_type=QuantizationType.STATIC, lower_bound=0.0, norm_type=NormType.NO_NORM, eps=1e-6
        ),
        "pre_norm_gamma": None,
        "residual": None,
    },
)
_torch_ref_2 = TorchReference(
    rmsnorm_quant_torch_ref,
    ("hidden", "ln_w"),
    bound_kwargs={
        "kargs": SimpleNamespace(
            quantization_type=QuantizationType.ROW, lower_bound=0.0, norm_type=NormType.RMS_NORM, eps=1e-6
        ),
        "input_dequant_scale": None,
        "pre_norm_gamma": None,
        "residual": None,
    },
)
_torch_ref_3 = TorchReference(
    rmsnorm_quant_torch_ref,
    ("hidden", "ln_w", "input_dequant_scale"),
    bound_kwargs={
        "kargs": SimpleNamespace(
            quantization_type=QuantizationType.STATIC, lower_bound=0.0, norm_type=NormType.RMS_NORM, eps=1e-6
        ),
        "pre_norm_gamma": None,
        "residual": None,
    },
)
_torch_ref_4 = TorchReference(
    rmsnorm_quant_torch_ref,
    ("hidden", "ln_w"),
    bound_kwargs={
        "kargs": SimpleNamespace(
            quantization_type=QuantizationType.ROW, lower_bound=0.5, norm_type=NormType.RMS_NORM, eps=1e-6
        ),
        "input_dequant_scale": None,
        "pre_norm_gamma": None,
        "residual": None,
    },
)
_torch_ref_5 = TorchReference(
    rmsnorm_quant_torch_ref,
    ("hidden", "ln_w"),
    bound_kwargs={
        "kargs": SimpleNamespace(
            quantization_type=QuantizationType.ROW, lower_bound=0.5, norm_type=NormType.NO_NORM, eps=1e-6
        ),
        "input_dequant_scale": None,
        "pre_norm_gamma": None,
        "residual": None,
    },
)


WORKLOADS = (
    {
        "torch_ref": _torch_ref_0,
        "input_specs": {"hidden": ((1, 160, 16384), "bfloat16"), "ln_w": ((16384,), "bfloat16")},
        "input_generator": _input_generator,
        "atol": 0.05,
        "rtol": 0.05,
        "nakb_latency_ms": 0.062539902,
        "best_historical_latency_ms": 0.062539902,
    },
    {
        "torch_ref": _torch_ref_0,
        "input_specs": {"hidden": ((1, 2, 16384), "bfloat16"), "ln_w": ((16384,), "bfloat16")},
        "input_generator": _input_generator,
        "atol": 0.05,
        "rtol": 0.05,
        "nakb_latency_ms": 0.043716598,
        "best_historical_latency_ms": 0.043716598,
    },
    {
        "torch_ref": _torch_ref_1,
        "input_specs": {
            "hidden": ((1, 160, 16384), "bfloat16"),
            "ln_w": ((16384,), "bfloat16"),
            "input_dequant_scale": ((128, 1), "float32"),
        },
        "input_generator": _input_generator,
        "atol": 0.05,
        "rtol": 0.05,
        "nakb_latency_ms": 0.058325742,
        "best_historical_latency_ms": 0.058325742,
    },
    {
        "torch_ref": _torch_ref_1,
        "input_specs": {
            "hidden": ((1, 2, 16384), "bfloat16"),
            "ln_w": ((16384,), "bfloat16"),
            "input_dequant_scale": ((128, 1), "float32"),
        },
        "input_generator": _input_generator,
        "atol": 0.05,
        "rtol": 0.05,
        "nakb_latency_ms": 0.041439935,
        "best_historical_latency_ms": 0.041439935,
    },
    {
        "torch_ref": _torch_ref_2,
        "input_specs": {"hidden": ((1, 2, 16384), "bfloat16"), "ln_w": ((16384,), "bfloat16")},
        "input_generator": _input_generator,
        "atol": 0.05,
        "rtol": 0.05,
        "nakb_latency_ms": 0.080931541,
        "best_historical_latency_ms": 0.080931541,
    },
    {
        "torch_ref": _torch_ref_2,
        "input_specs": {"hidden": ((1, 2048, 16384), "bfloat16"), "ln_w": ((16384,), "bfloat16")},
        "input_generator": _input_generator,
        "atol": 0.05,
        "rtol": 0.05,
        "nakb_latency_ms": 0.57984576,
        "best_historical_latency_ms": 0.57984576,
    },
    {
        "torch_ref": _torch_ref_3,
        "input_specs": {
            "hidden": ((1, 2, 16384), "bfloat16"),
            "ln_w": ((16384,), "bfloat16"),
            "input_dequant_scale": ((128, 1), "float32"),
        },
        "input_generator": _input_generator,
        "atol": 0.05,
        "rtol": 0.05,
        "nakb_latency_ms": 0.075650716,
        "best_historical_latency_ms": 0.075650716,
    },
    {
        "torch_ref": _torch_ref_3,
        "input_specs": {
            "hidden": ((1, 2048, 16384), "bfloat16"),
            "ln_w": ((16384,), "bfloat16"),
            "input_dequant_scale": ((128, 1), "float32"),
        },
        "input_generator": _input_generator,
        "atol": 0.05,
        "rtol": 0.05,
        "nakb_latency_ms": 0.513666698,
        "best_historical_latency_ms": 0.513666698,
    },
    {
        "torch_ref": _torch_ref_4,
        "input_specs": {"hidden": ((1, 2, 16384), "bfloat16"), "ln_w": ((16384,), "bfloat16")},
        "input_generator": _input_generator,
        "atol": 0.05,
        "rtol": 0.05,
        "nakb_latency_ms": 0.08107154,
        "best_historical_latency_ms": 0.08107154,
    },
    {
        "torch_ref": _torch_ref_5,
        "input_specs": {"hidden": ((1, 2, 16384), "bfloat16"), "ln_w": ((16384,), "bfloat16")},
        "input_generator": _input_generator,
        "atol": 0.05,
        "rtol": 0.05,
        "nakb_latency_ms": 0.046419928,
        "best_historical_latency_ms": 0.046419928,
    },
)
