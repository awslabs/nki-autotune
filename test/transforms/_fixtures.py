"""Shared canonical-IR fixture for transform tests.

Builds the canonical :class:`KernelIR` for the same matmul described
by ``kernel_transforms.py``: ``lhs_T(K=2048, M=2048).T @ rhs(K, N=2048)``.
"""

from __future__ import annotations

from nkigym.ir import KernelIR, build_initial_ir
from nkigym.ops import nkigym_kernel
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.memset import NKIMemset
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy

K, M, N = 2048, 2048, 2048
INPUT_SPECS: dict[str, tuple[int, ...]] = {"lhs_T": (K, M), "rhs": (K, N)}


@nkigym_kernel
def f_matmul(lhs_T, rhs):
    """``lhs_T.T @ rhs`` — load, memset, matmul, drain, store."""
    sbuf_lhs_T = NKIAlloc(location="sbuf", shape=(K, M), dtype="bfloat16")()
    sbuf_rhs = NKIAlloc(location="sbuf", shape=(K, N), dtype="bfloat16")()
    psum_prod = NKIAlloc(location="psum", shape=(M, N), dtype="float32")()
    sbuf_prod = NKIAlloc(location="sbuf", shape=(M, N), dtype="bfloat16")()
    hbm_out = NKIAlloc(location="shared_hbm", shape=(M, N), dtype="bfloat16")()
    NKILoad()(src=lhs_T, dst=sbuf_lhs_T)
    NKILoad()(src=rhs, dst=sbuf_rhs)
    NKIMemset(value=0.0)(dst=psum_prod)
    NKIMatmul()(stationary=sbuf_lhs_T, moving=sbuf_rhs, dst=psum_prod)
    NKITensorCopy()(src=psum_prod, dst=sbuf_prod)
    NKIStore()(src=sbuf_prod, dst=hbm_out)
    return hbm_out


def build_canonical_ir() -> KernelIR:
    """Build the canonical :class:`KernelIR` for the matmul fixture."""
    return build_initial_ir(f_matmul, INPUT_SPECS)
