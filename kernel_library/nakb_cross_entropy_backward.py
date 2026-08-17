"""Measured NAKB ``cross_entropy_backward`` workload targets."""

from __future__ import annotations

import ml_dtypes
import numpy as np
import torch
import torch.nn.functional as F

from kernel_library import TorchReference
from nkigym.profile import InputSpecs


def cross_entropy_backward_torch_ref(
    logits_hbm: torch.Tensor,
    targets_hbm: torch.Tensor,
    lse_state_hbm: torch.Tensor | None = None,
    reduction: str = "mean",
    positions_per_batch: int = 32,
    chunk_size: int = 32768,
    dtype: torch.dtype | None = None,
    inplace: bool = True,
) -> dict[str, torch.Tensor | None]:
    """PyTorch reference implementation of cross entropy backward pass."""
    if isinstance(reduction, int):
        reduction = "mean" if reduction == 0 else "sum"
    if reduction not in ("mean", "sum"):
        raise ValueError(f"Unknown reduction: {reduction}. Use 'mean' or 'sum'.")
    targets = targets_hbm.long() if targets_hbm.dtype != torch.long else targets_hbm
    logits_copy = logits_hbm.detach().clone().requires_grad_(True)
    loss = F.cross_entropy(logits_copy, targets, reduction=reduction)
    loss.backward()
    return {"grad_logits_hbm": logits_copy.grad}


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
    cross_entropy_backward_torch_ref,
    ("logits_hbm", "targets_hbm"),
    bound_kwargs={
        "lse_state_hbm": None,
        "reduction": "mean",
        "positions_per_batch": 32,
        "chunk_size": 32768,
        "dtype": None,
        "inplace": True,
    },
)
_torch_ref_1 = TorchReference(
    cross_entropy_backward_torch_ref,
    ("logits_hbm", "targets_hbm"),
    bound_kwargs={
        "lse_state_hbm": None,
        "reduction": "sum",
        "positions_per_batch": 32,
        "chunk_size": 32768,
        "dtype": None,
        "inplace": True,
    },
)


WORKLOADS = (
    {
        "torch_ref": _torch_ref_0,
        "input_specs": {"logits_hbm": ((1, 8032), "bfloat16"), "targets_hbm": ((1,), "int32")},
        "input_generator": _input_generator,
        "atol": 1e-05,
        "rtol": 1e-05,
        "nakb_latency_ms": 0.051545753,
        "best_historical_latency_ms": 0.051545753,
    },
    {
        "torch_ref": _torch_ref_0,
        "input_specs": {"logits_hbm": ((16384, 8032), "bfloat16"), "targets_hbm": ((16384,), "int32")},
        "input_generator": _input_generator,
        "atol": 1e-05,
        "rtol": 1e-05,
        "nakb_latency_ms": 3.409941339,
        "best_historical_latency_ms": 3.409941339,
    },
    {
        "torch_ref": _torch_ref_0,
        "input_specs": {"logits_hbm": ((4096, 32320), "float32"), "targets_hbm": ((4096,), "int32")},
        "input_generator": _input_generator,
        "atol": 1e-05,
        "rtol": 1e-05,
        "nakb_latency_ms": 4.655005227,
        "best_historical_latency_ms": 4.655005227,
    },
    {
        "torch_ref": _torch_ref_0,
        "input_specs": {"logits_hbm": ((8192, 32320), "bfloat16"), "targets_hbm": ((8192,), "int32")},
        "input_generator": _input_generator,
        "atol": 1e-05,
        "rtol": 1e-05,
        "nakb_latency_ms": 8.034641613,
        "best_historical_latency_ms": 8.034641613,
    },
    {
        "torch_ref": _torch_ref_1,
        "input_specs": {"logits_hbm": ((16384, 8032), "float32"), "targets_hbm": ((16384,), "int32")},
        "input_generator": _input_generator,
        "atol": 1e-05,
        "rtol": 1e-05,
        "nakb_latency_ms": 4.811285816,
        "best_historical_latency_ms": 4.811285816,
    },
)
