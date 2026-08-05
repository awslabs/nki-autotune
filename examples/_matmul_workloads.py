"""Shared canonical matmul workloads for the search examples."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from nkigym.ops import nkigym_kernel
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.ops.transpose import NKITranspose
from nkigym.transforms import (
    BufferCompaction,
    BufferLayout,
    CancelTransposePair,
    CodeMotion,
    Fuse,
    InsertTransposePair,
    Reorder,
    RFactor,
    SoftwarePipeline,
    Split,
    TransposeThroughLoad,
    TransposeThroughMatmul,
    TransposeThroughTensorCopy,
)

InputSpecs = dict[str, tuple[tuple[int, ...], str]]
K, M, N = 2048, 2048, 2048
SKEWED_K, SKEWED_M, SKEWED_N = 3584, 4096, 128


@dataclass(frozen=True)
class Workload:
    """Canonical graph and NumPy reference for one matmul workload."""

    name: str
    input_specs: InputSpecs
    f_numpy: Callable[..., np.ndarray]
    f_nkigym: Callable[..., np.ndarray]


def f_lhs_t_rhs_numpy(lhs_T: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Return the NumPy reference for ``lhs_T.T @ rhs``."""
    return lhs_T.T @ rhs


@nkigym_kernel
def f_lhs_t_rhs_nkigym(lhs_T, rhs):
    """Return the canonical SSA graph for ``lhs_T.T @ rhs``."""
    sbuf_lhs_T = NKILoad()(src=lhs_T)
    sbuf_rhs = NKILoad()(src=rhs)
    psum_prod = NKIMatmul()(stationary=sbuf_lhs_T, moving=sbuf_rhs)
    sbuf_prod = NKITensorCopy()(src=psum_prod)
    hbm_out = NKIStore()(src=sbuf_prod)
    return hbm_out


def f_lhs_rhs_numpy(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Return the NumPy reference for ``lhs @ rhs``."""
    return lhs @ rhs


@nkigym_kernel
def f_lhs_rhs_nkigym(lhs, rhs):
    """Return the canonical SSA graph for ``lhs @ rhs``."""
    sbuf_lhs = NKILoad()(src=lhs)
    psum_lhs_T = NKITranspose()(data=sbuf_lhs)
    sbuf_lhs_T = NKITensorCopy()(src=psum_lhs_T)
    sbuf_rhs = NKILoad()(src=rhs)
    psum_prod = NKIMatmul()(stationary=sbuf_lhs_T, moving=sbuf_rhs)
    sbuf_prod = NKITensorCopy()(src=psum_prod)
    hbm_out = NKIStore()(src=sbuf_prod)
    return hbm_out


LHS_T_RHS = Workload(
    name="matmul_lhsT_rhs",
    input_specs={"lhs_T": ((K, M), "bfloat16"), "rhs": ((K, N), "bfloat16")},
    f_numpy=f_lhs_t_rhs_numpy,
    f_nkigym=f_lhs_t_rhs_nkigym,
)
LHS_RHS = Workload(
    name="matmul_lhs_rhs",
    input_specs={"lhs": ((M, K), "bfloat16"), "rhs": ((K, N), "bfloat16")},
    f_numpy=f_lhs_rhs_numpy,
    f_nkigym=f_lhs_rhs_nkigym,
)
SKEWED_LHS_T_RHS = Workload(
    name="skewed_matmul_lhsT_rhs",
    input_specs={"lhs_T": ((SKEWED_K, SKEWED_M), "bfloat16"), "rhs": ((SKEWED_K, SKEWED_N), "bfloat16")},
    f_numpy=f_lhs_t_rhs_numpy,
    f_nkigym=f_lhs_t_rhs_nkigym,
)
TRANSPOSE_DEMO = Workload(
    name="transpose_demo",
    input_specs={"lhs": ((4096, 1024), "bfloat16"), "rhs": ((1024, 128), "bfloat16")},
    f_numpy=f_lhs_rhs_numpy,
    f_nkigym=f_lhs_rhs_nkigym,
)
TRANSPOSE_TRANSFORMS = [
    InsertTransposePair(),
    CancelTransposePair(),
    TransposeThroughLoad(),
    TransposeThroughMatmul(),
    TransposeThroughTensorCopy(),
]
NON_TRANSPOSE_TRANSFORMS = [
    Split(),
    Fuse(),
    Reorder(),
    CodeMotion(),
    RFactor(),
    SoftwarePipeline(),
    BufferLayout(),
    BufferCompaction(),
]
TRANSFORMS = [*TRANSPOSE_TRANSFORMS, *NON_TRANSPOSE_TRANSFORMS]
