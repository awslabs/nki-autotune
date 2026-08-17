"""Measured NAKB ``depthwise_conv1d`` workload targets."""

from __future__ import annotations

import ml_dtypes
import numpy as np
import torch
import torch.nn.functional as F

from kernel_library import TorchReference
from nkigym.profile import InputSpecs


def depthwise_conv1d_implicit_gemm_torch_ref(
    img_ref: torch.Tensor,
    filter_ref: torch.Tensor,
    padding: tuple = ((0, 0), (0, 0)),
    stride: tuple = (1, 1),
    rhs_dilation: tuple = (1, 1),
    lhs_dilation: tuple = (1, 1),
    feature_group_count: int = 1,
    batch_group_count: int = 1,
    in_perm: tuple | None = None,
    kern_perm: tuple | None = None,
    out_perm: tuple | None = None,
) -> dict[str, torch.Tensor]:
    """PyTorch reference implementation of depthwise Conv1D."""
    channels = img_ref.shape[1]
    padding_pytorch = (0, padding[1][0])
    output = F.conv2d(img_ref, filter_ref, bias=None, stride=stride, padding=padding_pytorch, groups=channels)
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


_torch_ref_0 = TorchReference(
    depthwise_conv1d_implicit_gemm_torch_ref,
    ("img_ref", "filter_ref"),
    bound_kwargs={
        "padding": ((0, 0), (1, 1)),
        "stride": (1, 2),
        "rhs_dilation": (1, 1),
        "lhs_dilation": (1, 1),
        "feature_group_count": 128,
        "batch_group_count": 1,
        "in_perm": None,
        "kern_perm": None,
        "out_perm": None,
    },
)
_torch_ref_1 = TorchReference(
    depthwise_conv1d_implicit_gemm_torch_ref,
    ("img_ref", "filter_ref"),
    bound_kwargs={
        "padding": ((0, 0), (1, 1)),
        "stride": (1, 1),
        "rhs_dilation": (1, 1),
        "lhs_dilation": (1, 1),
        "feature_group_count": 16,
        "batch_group_count": 1,
        "in_perm": None,
        "kern_perm": None,
        "out_perm": None,
    },
)
_torch_ref_2 = TorchReference(
    depthwise_conv1d_implicit_gemm_torch_ref,
    ("img_ref", "filter_ref"),
    bound_kwargs={
        "padding": ((0, 0), (2, 2)),
        "stride": (1, 1),
        "rhs_dilation": (1, 1),
        "lhs_dilation": (1, 1),
        "feature_group_count": 64,
        "batch_group_count": 1,
        "in_perm": None,
        "kern_perm": None,
        "out_perm": None,
    },
)


WORKLOADS = (
    {
        "torch_ref": _torch_ref_0,
        "input_specs": {"img_ref": ((4, 128, 1, 256), "float32"), "filter_ref": ((128, 1, 1, 3), "float32")},
        "input_generator": _input_generator,
        "atol": 1e-05,
        "rtol": 1e-05,
        "nakb_latency_ms": 0.640987332,
        "best_historical_latency_ms": 0.640987332,
    },
    {
        "torch_ref": _torch_ref_1,
        "input_specs": {"img_ref": ((1, 16, 1, 32), "float32"), "filter_ref": ((16, 1, 1, 3), "float32")},
        "input_generator": _input_generator,
        "atol": 1e-05,
        "rtol": 1e-05,
        "nakb_latency_ms": 0.017853306,
        "best_historical_latency_ms": 0.017853306,
    },
    {
        "torch_ref": _torch_ref_2,
        "input_specs": {"img_ref": ((2, 64, 1, 128), "float32"), "filter_ref": ((64, 1, 1, 5), "float32")},
        "input_generator": _input_generator,
        "atol": 1e-05,
        "rtol": 1e-05,
        "nakb_latency_ms": 0.095890684,
        "best_historical_latency_ms": 0.095890684,
    },
)
