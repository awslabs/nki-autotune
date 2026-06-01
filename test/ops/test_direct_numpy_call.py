"""Direct numpy invocation of @nkigym_kernel functions (SSA form).

A @nkigym_kernel-decorated function behaves as a numpy simulator when
called on bare numpy arrays: each NKIOp's ``_run`` allocates and returns
its output, and the per-op ``_check_roles`` enforces the load/compute/store
lineage at runtime. The decorator asserts the return carries role
``"stored"`` (the NKIStore output) or ``"shared_hbm"``.
"""

import numpy as np
import pytest

from nkigym.ops import nkigym_kernel
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy

K, M, N = 128, 128, 128


@nkigym_kernel
def _lhsT_matmul(lhs_T, rhs):
    lhs_T_sbuf = NKILoad()(src=lhs_T)
    rhs_sbuf = NKILoad()(src=rhs)
    psum_acc = NKIMatmul()(stationary=lhs_T_sbuf, moving=rhs_sbuf)
    sbuf_prod = NKITensorCopy()(src=psum_acc)
    hbm_out = NKIStore()(src=sbuf_prod)
    return hbm_out


def test_direct_call_computes_matmul():
    rng = np.random.default_rng(0)
    lhs_T = rng.standard_normal((K, M)).astype(np.float32)
    rhs = rng.standard_normal((K, N)).astype(np.float32)
    actual = _lhsT_matmul(lhs_T=lhs_T, rhs=rhs)
    expected = lhs_T.T @ rhs
    assert np.allclose(np.asarray(actual), expected, atol=1e-4, rtol=1e-4)


def test_direct_call_returns_stored_role():
    rng = np.random.default_rng(0)
    lhs_T = rng.standard_normal((K, M)).astype(np.float32)
    rhs = rng.standard_normal((K, N)).astype(np.float32)
    out = _lhsT_matmul(lhs_T=lhs_T, rhs=rhs)
    assert getattr(out, "role", None) == "stored"


def test_load_rejects_non_param_src():
    """NKILoad's _check_roles fires when src is not an HBM param."""
    rng = np.random.default_rng(0)
    sbuf = NKILoad()(src=rng.standard_normal((16, 16)).astype(np.float32))
    with pytest.raises(TypeError, match="NKILoad.*expects HBM param"):
        NKILoad()(src=sbuf)


def test_store_rejects_non_sbuf_src():
    """NKIStore's _check_roles fires when src is PSUM instead of SBUF."""
    rng = np.random.default_rng(0)
    lhs = NKILoad()(src=rng.standard_normal((16, 16)).astype(np.float32))
    rhs = NKILoad()(src=rng.standard_normal((16, 16)).astype(np.float32))
    psum = NKIMatmul()(stationary=lhs, moving=rhs)
    with pytest.raises(TypeError, match="NKIStore.*expects sbuf"):
        NKIStore()(src=psum)
