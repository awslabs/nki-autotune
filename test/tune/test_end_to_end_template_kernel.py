"""End-to-end: K-outside template kernel (via RFactor).

NOTE: This test file references DecomposeReduction which is being replaced by
RFactor in Task 16. This test will be rewritten after Task 16 lands to use the
new RFactor atom instead.

TEMPORARILY SKIPPED until Task 16 completes.
"""

import numpy as np
import pytest

from nkigym.ops import nkigym_kernel
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.memset import NKIMemset
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy


@nkigym_kernel
def _matmul_k(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """First-class buffers matmul fixture."""
    lhs_sbuf = NKIAlloc(location="sbuf", shape=(2048, 2048), dtype="bfloat16")()
    rhs_sbuf = NKIAlloc(location="sbuf", shape=(2048, 2048), dtype="bfloat16")()
    psum_acc = NKIAlloc(location="psum", shape=(2048, 2048), dtype="float32")()
    sbuf_prod = NKIAlloc(location="sbuf", shape=(2048, 2048), dtype="bfloat16")()
    hbm_out = NKIAlloc(location="hbm", shape=(2048, 2048), dtype="bfloat16")()
    NKILoad()(src=lhs, dst=lhs_sbuf)
    NKILoad()(src=rhs, dst=rhs_sbuf)
    NKIMemset(value=0.0)(dst=psum_acc)
    NKIMatmul()(stationary=lhs_sbuf, moving=rhs_sbuf, dst=psum_acc)
    NKITensorCopy()(src=psum_acc, dst=sbuf_prod)
    NKIStore()(src=sbuf_prod, dst=hbm_out)
    return hbm_out


_INPUT_SPECS: dict[str, dict] = {
    "lhs": {"shape": (2048, 2048), "dtype": "bfloat16"},
    "rhs": {"shape": (2048, 2048), "dtype": "bfloat16"},
}


@pytest.mark.skip(reason="DecomposeReduction removed; rewrite for RFactor in Task 16")
def test_decompose_reduction_fissions_matmul_into_three_trees():
    """OBSOLETE: will be rewritten for RFactor."""


@pytest.mark.skip(reason="DecomposeReduction removed; rewrite for RFactor in Task 16")
def test_end_to_end_decompose_plus_reorder_renders():
    """OBSOLETE: will be rewritten for RFactor + Reorder composition."""
