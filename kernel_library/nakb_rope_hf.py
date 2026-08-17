"""Measured NAKB ``rope_hf`` workload targets."""

from __future__ import annotations

import ml_dtypes
import numpy as np
import torch

from kernel_library import TorchReference
from nkigym.profile import InputSpecs


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half of the hidden dimensions of the input."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _rotate_half_backward(x: torch.Tensor) -> torch.Tensor:
    """Apply the backward half rotation."""
    half = x.shape[-1] // 2
    return torch.cat((x[..., half:], -x[..., :half]), dim=-1)


def rope_hf_torch_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    q_out: torch.Tensor | None,
    k_out: torch.Tensor | None,
    cos: torch.Tensor | None = None,
    sin: torch.Tensor | None = None,
    rope_cache: torch.Tensor | None = None,
    backward: bool = False,
) -> dict[str, torch.Tensor]:
    """Torch reference for the Hugging Face RoPE layout."""
    if rope_cache is not None:
        half = rope_cache.shape[-1] // 2
        cos_val = rope_cache[..., :half]
        sin_val = rope_cache[..., half:]
        if cos_val.ndim == 2:
            cos_val = cos_val.unsqueeze(0)
            sin_val = sin_val.unsqueeze(0)
    else:
        cos_val = cos
        sin_val = sin
    cos_val = cos_val.unsqueeze(1).to(q.dtype)
    sin_val = sin_val.unsqueeze(1).to(q.dtype)
    if backward:
        q_embed = q * cos_val + _rotate_half_backward(q * sin_val)
        k_embed = k * cos_val + _rotate_half_backward(k * sin_val)
    else:
        q_embed = q * cos_val + _rotate_half(q) * sin_val
        k_embed = k * cos_val + _rotate_half(k) * sin_val
    return {"q_out": q_embed, "k_out": k_embed}


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
    rope_hf_torch_ref,
    ("q", "k", "cos", "sin"),
    bound_kwargs={"q_out": None, "k_out": None, "rope_cache": None, "backward": False},
)
_torch_ref_1 = TorchReference(
    rope_hf_torch_ref,
    ("q", "k", "cos", "sin"),
    bound_kwargs={"q_out": None, "k_out": None, "rope_cache": None, "backward": True},
)


WORKLOADS = (
    {
        "torch_ref": _torch_ref_0,
        "input_specs": {
            "q": ((1, 32, 256, 128), "bfloat16"),
            "k": ((1, 8, 256, 128), "bfloat16"),
            "cos": ((1, 256, 128), "bfloat16"),
            "sin": ((1, 256, 128), "bfloat16"),
        },
        "input_generator": _input_generator,
        "atol": 1e-05,
        "rtol": 1e-05,
        "nakb_latency_ms": 0.065473231,
        "best_historical_latency_ms": 0.065473231,
    },
    {
        "torch_ref": _torch_ref_0,
        "input_specs": {
            "q": ((1, 32, 256, 128), "float32"),
            "k": ((1, 8, 256, 128), "float32"),
            "cos": ((1, 256, 128), "float32"),
            "sin": ((1, 256, 128), "float32"),
        },
        "input_generator": _input_generator,
        "atol": 1e-05,
        "rtol": 1e-05,
        "nakb_latency_ms": 0.065124898,
        "best_historical_latency_ms": 0.065124898,
    },
    {
        "torch_ref": _torch_ref_0,
        "input_specs": {
            "q": ((1, 32, 512, 64), "float32"),
            "k": ((1, 8, 512, 64), "float32"),
            "cos": ((1, 512, 64), "float32"),
            "sin": ((1, 512, 64), "float32"),
        },
        "input_generator": _input_generator,
        "atol": 1e-05,
        "rtol": 1e-05,
        "nakb_latency_ms": 0.077855711,
        "best_historical_latency_ms": 0.077855711,
    },
    {
        "torch_ref": _torch_ref_0,
        "input_specs": {
            "q": ((2, 16, 512, 128), "bfloat16"),
            "k": ((2, 4, 512, 128), "bfloat16"),
            "cos": ((2, 512, 128), "bfloat16"),
            "sin": ((2, 512, 128), "bfloat16"),
        },
        "input_generator": _input_generator,
        "atol": 1e-05,
        "rtol": 1e-05,
        "nakb_latency_ms": 0.079684875,
        "best_historical_latency_ms": 0.079684875,
    },
    {
        "torch_ref": _torch_ref_1,
        "input_specs": {
            "q": ((1, 32, 256, 128), "float32"),
            "k": ((1, 8, 256, 128), "float32"),
            "cos": ((1, 256, 128), "float32"),
            "sin": ((1, 256, 128), "float32"),
        },
        "input_generator": _input_generator,
        "atol": 1e-05,
        "rtol": 1e-05,
        "nakb_latency_ms": 0.064676566,
        "best_historical_latency_ms": 0.064676566,
    },
)
