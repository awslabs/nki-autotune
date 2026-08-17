"""Measured NAKB ``build_all_to_all_v_metadata`` workload targets."""

from __future__ import annotations

import ml_dtypes
import numpy as np
import torch

from kernel_library import TorchReference
from nkigym.profile import InputSpecs


def kernel_assert(condition: bool, error_text: str) -> None:
    """Raise NAKB's kernel validation assertion."""
    assert condition, (  # noqa: S101
        f"[INTERNAL_ERROR] [NCC_INKI016] Kernel validation exception: {error_text} - "
        "Please check the validation message and adjust kernel inputs accordingly"
    )


def build_all_to_all_v_metadata_torch_ref(
    expert_index: torch.Tensor,
    replica_group_size: int,
    E: int,
    recv_counts_known: bool = False,
    has_rdispls: bool = False,
) -> torch.Tensor:
    """Compute send counts and displacements for all_to_all_v."""
    kernel_assert(
        not recv_counts_known, f"Torch ref does not yet support recv_counts_known=True, got {recv_counts_known=}"
    )
    kernel_assert(not has_rdispls, f"Torch ref does not yet support has_rdispls=True, got {has_rdispls=}")
    kernel_assert(
        E % replica_group_size == 0, f"Expected E divisible by replica_group_size, got {E=}, {replica_group_size=}"
    )
    per_expert_counts = torch.bincount(expert_index.flatten().int(), minlength=E)
    send_counts = per_expert_counts.reshape(replica_group_size, -1).sum(dim=1).to(torch.int32)
    send_displs = torch.zeros(replica_group_size, dtype=torch.int32)
    send_displs[1:] = torch.cumsum(send_counts, 0)[:-1]
    recv_counts = torch.zeros(replica_group_size, dtype=torch.int32)
    rows = [send_counts, send_displs, recv_counts]
    return torch.stack(rows, dim=0)


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
    build_all_to_all_v_metadata_torch_ref,
    ("expert_index",),
    bound_kwargs={"replica_group_size": 128, "E": 256, "recv_counts_known": False, "has_rdispls": False},
)
_torch_ref_1 = TorchReference(
    build_all_to_all_v_metadata_torch_ref,
    ("expert_index",),
    bound_kwargs={"replica_group_size": 128, "E": 128, "recv_counts_known": False, "has_rdispls": False},
)
_torch_ref_2 = TorchReference(
    build_all_to_all_v_metadata_torch_ref,
    ("expert_index",),
    bound_kwargs={"replica_group_size": 16, "E": 128, "recv_counts_known": False, "has_rdispls": False},
)
_torch_ref_3 = TorchReference(
    build_all_to_all_v_metadata_torch_ref,
    ("expert_index",),
    bound_kwargs={"replica_group_size": 1024, "E": 1024, "recv_counts_known": False, "has_rdispls": False},
)


WORKLOADS = (
    {
        "torch_ref": _torch_ref_0,
        "input_specs": {"expert_index": ((1, 8), "int32")},
        "input_generator": _input_generator,
        "atol": 0.0,
        "rtol": 0.0,
        "nakb_latency_ms": 0.017807472,
        "best_historical_latency_ms": 0.017807472,
    },
    {
        "torch_ref": _torch_ref_1,
        "input_specs": {"expert_index": ((32, 4), "int32")},
        "input_generator": _input_generator,
        "atol": 0.0,
        "rtol": 0.0,
        "nakb_latency_ms": 0.017992472,
        "best_historical_latency_ms": 0.017992472,
    },
    {
        "torch_ref": _torch_ref_1,
        "input_specs": {"expert_index": ((4, 4), "int32")},
        "input_generator": _input_generator,
        "atol": 0.0,
        "rtol": 0.0,
        "nakb_latency_ms": 0.01682164,
        "best_historical_latency_ms": 0.01682164,
    },
    {
        "torch_ref": _torch_ref_2,
        "input_specs": {"expert_index": ((32, 4), "int32")},
        "input_generator": _input_generator,
        "atol": 0.0,
        "rtol": 0.0,
        "nakb_latency_ms": 0.019341637,
        "best_historical_latency_ms": 0.019341637,
    },
    {
        "torch_ref": _torch_ref_3,
        "input_specs": {"expert_index": ((512, 8), "int32")},
        "input_generator": _input_generator,
        "atol": 0.0,
        "rtol": 0.0,
        "nakb_latency_ms": 0.062098237,
        "best_historical_latency_ms": 0.062098237,
    },
)
