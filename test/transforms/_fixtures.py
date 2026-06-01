"""Shared canonical-IR fixture for transform tests.

Builds the canonical :class:`KernelIR` for the same matmul described
by ``kernel_transforms.py``: ``lhs_T(K=2048, M=2048).T @ rhs(K, N=2048)``.
"""

from __future__ import annotations

from nkigym.ir import KernelIR, build_initial_ir
from nkigym.ops import nkigym_kernel
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy

K, M, N = 2048, 2048, 2048
INPUT_SPECS: dict[str, tuple[tuple[int, ...], str]] = {"lhs_T": ((K, M), "bfloat16"), "rhs": ((K, N), "bfloat16")}


@nkigym_kernel
def f_matmul(lhs_T, rhs):
    """``lhs_T.T @ rhs`` — load, matmul, drain, store (SSA)."""
    sbuf_lhs_T = NKILoad()(src=lhs_T)
    sbuf_rhs = NKILoad()(src=rhs)
    psum_prod = NKIMatmul()(stationary=sbuf_lhs_T, moving=sbuf_rhs)
    sbuf_prod = NKITensorCopy()(src=psum_prod)
    hbm_out = NKIStore()(src=sbuf_prod)
    return hbm_out


def build_canonical_ir() -> KernelIR:
    """Build the canonical :class:`KernelIR` for the matmul fixture."""
    return build_initial_ir(f_matmul, INPUT_SPECS)
