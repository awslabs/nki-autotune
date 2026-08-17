"""Measured NAKB ``find_nonzero_indices`` workload targets."""

from __future__ import annotations

import ml_dtypes
import numpy as np
import torch

from kernel_library import TorchReference
from nkigym.profile import InputSpecs


def find_nonzero_indices_torch_ref(
    input_tensor: torch.Tensor,
    col_start_id: torch.Tensor | None = None,
    n_cols: int | None = None,
    chunk_size: int | None = None,
    index_dtype: torch.dtype = torch.int32,
) -> dict[str, torch.Tensor]:
    """PyTorch reference for find_nonzero_indices kernel."""
    tokens, full_columns = input_tensor.shape
    if col_start_id is not None:
        start_column = col_start_id.item()
        output_columns = n_cols
    else:
        start_column = 0
        output_columns = full_columns
    indices = torch.full((output_columns, tokens), -1, dtype=index_dtype)
    nonzero_counts = torch.zeros(output_columns, dtype=torch.int32)
    for column_index in range(output_columns):
        column = input_tensor[:, start_column + column_index]
        nonzero = torch.nonzero(column, as_tuple=False).squeeze(-1)
        count = nonzero.shape[0]
        nonzero_counts[column_index] = count
        if count > 0:
            indices[column_index, :count] = nonzero.to(index_dtype)
    return {"indices": indices, "nonzero_counts": nonzero_counts}


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
    find_nonzero_indices_torch_ref,
    ("input_tensor",),
    bound_kwargs={"col_start_id": None, "n_cols": None, "chunk_size": None},
)


WORKLOADS = (
    {
        "torch_ref": _torch_ref_0,
        "input_specs": {"input_tensor": ((128, 16), "float32")},
        "input_generator": _input_generator,
        "atol": 0.0,
        "rtol": 0.0,
        "nakb_latency_ms": 0.021746633,
        "best_historical_latency_ms": 0.021746633,
    },
    {
        "torch_ref": _torch_ref_0,
        "input_specs": {"input_tensor": ((256, 32), "float32")},
        "input_generator": _input_generator,
        "atol": 0.0,
        "rtol": 0.0,
        "nakb_latency_ms": 0.02948912,
        "best_historical_latency_ms": 0.02948912,
    },
)
