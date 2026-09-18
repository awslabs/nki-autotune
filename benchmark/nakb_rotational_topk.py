"""Measured NAKB ``rotational_topk`` workload targets."""

from __future__ import annotations

import ml_dtypes
import numpy as np
import torch

from benchmark import AccuracySpec, InputSpecs, TorchReference


def topk_torch_ref(inp: torch.Tensor, topk_k: int, topk_sorted: int) -> dict[str, torch.Tensor]:
    """TopK torch reference implementation."""
    values, indices = torch.topk(inp, k=topk_k, dim=-1, largest=True, sorted=bool(topk_sorted))
    return {"topk_values": values, "topk_indices": indices.to(torch.int32)}


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
    topk_torch_ref, ("inp",), bound_kwargs={"topk_k": 256, "topk_sorted": 1}, selection_ties="any"
)
_torch_ref_1 = TorchReference(topk_torch_ref, ("inp",), bound_kwargs={"topk_k": 3, "topk_sorted": 1})
_torch_ref_2 = TorchReference(
    topk_torch_ref, ("inp",), bound_kwargs={"topk_k": 1, "topk_sorted": 1}, selection_ties="any"
)
_torch_ref_3 = TorchReference(
    topk_torch_ref, ("inp",), bound_kwargs={"topk_k": 8, "topk_sorted": 1}, selection_ties="any"
)
_torch_ref_4 = TorchReference(
    topk_torch_ref, ("inp",), bound_kwargs={"topk_k": 2048, "topk_sorted": 1}, selection_ties="any"
)
_torch_ref_5 = TorchReference(topk_torch_ref, ("inp",), bound_kwargs={"topk_k": 10, "topk_sorted": 1})
_torch_ref_6 = TorchReference(topk_torch_ref, ("inp",), bound_kwargs={"topk_k": 20, "topk_sorted": 0})
_torch_ref_7 = TorchReference(topk_torch_ref, ("inp",), bound_kwargs={"topk_k": 5, "topk_sorted": 1})


WORKLOADS = (
    {
        "accuracy": AccuracySpec(atol=1e-08, rtol=1e-05, mode="topk", output_indices=None, output_tolerances=()),
        "torch_ref": _torch_ref_0,
        "input_specs": {"inp": ((1, 16000), "float32")},
        "input_generator": _input_generator,
        "nakb_latency_ms": 0.114017322,
    },
    {
        "accuracy": AccuracySpec(atol=1e-08, rtol=1e-05, mode="topk", output_indices=None, output_tolerances=()),
        "torch_ref": _torch_ref_0,
        "input_specs": {"inp": ((1, 25600), "float32")},
        "input_generator": _input_generator,
        "nakb_latency_ms": 0.13653062,
    },
    {
        "accuracy": AccuracySpec(atol=1e-08, rtol=1e-05, mode="topk", output_indices=None, output_tolerances=()),
        "torch_ref": _torch_ref_0,
        "input_specs": {"inp": ((1, 3168), "bfloat16")},
        "input_generator": _input_generator,
        "nakb_latency_ms": 0.093454021,
    },
    {
        "accuracy": AccuracySpec(atol=1e-08, rtol=1e-05, mode="topk", output_indices=None, output_tolerances=()),
        "torch_ref": _torch_ref_0,
        "input_specs": {"inp": ((1, 3168), "float32")},
        "input_generator": _input_generator,
        "nakb_latency_ms": 0.088945694,
    },
    {
        "accuracy": AccuracySpec(atol=1e-08, rtol=1e-05, mode="topk", output_indices=None, output_tolerances=()),
        "torch_ref": _torch_ref_0,
        "input_specs": {"inp": ((20, 8192), "float32")},
        "input_generator": _input_generator,
        "nakb_latency_ms": 0.186643041,
    },
    {
        "accuracy": AccuracySpec(atol=1e-08, rtol=1e-05, mode="topk", output_indices=None, output_tolerances=()),
        "torch_ref": _torch_ref_0,
        "input_specs": {"inp": ((40, 4058), "float32")},
        "input_generator": _input_generator,
        "nakb_latency_ms": 0.142070612,
    },
    {
        "accuracy": AccuracySpec(atol=1e-08, rtol=1e-05, mode="topk", output_indices=None, output_tolerances=()),
        "torch_ref": _torch_ref_0,
        "input_specs": {"inp": ((40, 8192), "float32")},
        "input_generator": _input_generator,
        "nakb_latency_ms": 0.197013026,
    },
    {
        "accuracy": AccuracySpec(atol=0.001, rtol=0.001, mode="global_max", output_indices=None, output_tolerances=()),
        "torch_ref": _torch_ref_1,
        "input_specs": {"inp": ((1, 2048), "float32")},
        "input_generator": _input_generator,
        "nakb_latency_ms": 0.017518306,
    },
    {
        "accuracy": AccuracySpec(atol=1e-08, rtol=1e-05, mode="topk", output_indices=None, output_tolerances=()),
        "torch_ref": _torch_ref_2,
        "input_specs": {"inp": ((1, 3168), "float32")},
        "input_generator": _input_generator,
        "nakb_latency_ms": 0.022140798,
    },
    {
        "accuracy": AccuracySpec(atol=1e-08, rtol=1e-05, mode="topk", output_indices=None, output_tolerances=()),
        "torch_ref": _torch_ref_3,
        "input_specs": {"inp": ((1, 3168), "float32")},
        "input_generator": _input_generator,
        "nakb_latency_ms": 0.020824134,
    },
    {
        "accuracy": AccuracySpec(atol=1e-08, rtol=1e-05, mode="topk", output_indices=None, output_tolerances=()),
        "torch_ref": _torch_ref_4,
        "input_specs": {"inp": ((1, 8192), "float32")},
        "input_generator": _input_generator,
        "nakb_latency_ms": 2.032394324,
    },
    {
        "accuracy": AccuracySpec(atol=0.001, rtol=0.001, mode="global_max", output_indices=None, output_tolerances=()),
        "torch_ref": _torch_ref_5,
        "input_specs": {"inp": ((32, 4096), "float32")},
        "input_generator": _input_generator,
        "nakb_latency_ms": 0.035071611,
    },
    {
        "accuracy": AccuracySpec(atol=0.001, rtol=0.001, mode="global_max", output_indices=None, output_tolerances=()),
        "torch_ref": _torch_ref_6,
        "input_specs": {"inp": ((64, 8192), "float32")},
        "input_generator": _input_generator,
        "nakb_latency_ms": 0.062557402,
    },
    {
        "accuracy": AccuracySpec(atol=0.001, rtol=0.001, mode="global_max", output_indices=None, output_tolerances=()),
        "torch_ref": _torch_ref_7,
        "input_specs": {"inp": ((8, 1024), "float32")},
        "input_generator": _input_generator,
        "nakb_latency_ms": 0.016570807,
    },
)
