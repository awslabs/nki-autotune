"""Measured NAKB ``cross_entropy_forward`` workload targets."""

from __future__ import annotations

import ml_dtypes
import numpy as np
import torch
import torch.nn.functional as F

from kernel_library import TorchReference
from nkigym.profile import InputSpecs


def cross_entropy_forward_torch_ref(
    logits_hbm: torch.Tensor,
    targets_hbm: torch.Tensor,
    positions_per_batch: int = 32,
    chunk_size: int = 32768,
    dtype: torch.dtype | None = None,
) -> dict[str, torch.Tensor]:
    """PyTorch reference implementation of cross entropy forward pass."""
    targets = targets_hbm.long() if targets_hbm.dtype != torch.long else targets_hbm
    loss = F.cross_entropy(logits_hbm, targets, reduction="none")
    lse = torch.logsumexp(logits_hbm, dim=1)
    return {"loss_hbm": loss, "lse_state_hbm": lse}


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
    cross_entropy_forward_torch_ref,
    ("logits_hbm", "targets_hbm"),
    bound_kwargs={"positions_per_batch": 32, "chunk_size": 32768, "dtype": None},
)


WORKLOADS = (
    {
        "torch_ref": _torch_ref_0,
        "input_specs": {"logits_hbm": ((256, 32000), "float32"), "targets_hbm": ((256,), "int32")},
        "input_generator": _input_generator,
        "atol": 1e-05,
        "rtol": 1e-05,
        "nakb_latency_ms": 0.335703642,
        "best_historical_latency_ms": 0.335703642,
    },
    {
        "torch_ref": _torch_ref_0,
        "input_specs": {"logits_hbm": ((32, 128256), "float32"), "targets_hbm": ((32,), "int32")},
        "input_generator": _input_generator,
        "atol": 1e-05,
        "rtol": 1e-05,
        "nakb_latency_ms": 0.513581698,
        "best_historical_latency_ms": 0.513581698,
    },
    {
        "torch_ref": _torch_ref_0,
        "input_specs": {"logits_hbm": ((32, 4096), "float32"), "targets_hbm": ((32,), "int32")},
        "input_generator": _input_generator,
        "atol": 1e-05,
        "rtol": 1e-05,
        "nakb_latency_ms": 0.062041569,
        "best_historical_latency_ms": 0.062041569,
    },
    {
        "torch_ref": _torch_ref_0,
        "input_specs": {"logits_hbm": ((64, 32000), "float32"), "targets_hbm": ((64,), "int32")},
        "input_generator": _input_generator,
        "atol": 1e-05,
        "rtol": 1e-05,
        "nakb_latency_ms": 0.198568857,
        "best_historical_latency_ms": 0.198568857,
    },
)
