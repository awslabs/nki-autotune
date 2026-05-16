"""Direct numpy invocation of @nkigym_kernel functions.

A @nkigym_kernel-decorated function must behave as a numpy simulator
when called on bare numpy arrays: each NKIOp's ``_run`` body writes
through its ``dst`` operand (or ``reduce_res`` for activation_reduce),
and the per-op ``_check_roles`` enforces the load/compute/store lineage
at runtime. The decorator asserts the return value carries role
``"shared_hbm"`` (direct return of the alloc'd HBM buffer) or ``"stored"``
(direct return of NKIStore).
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

K, M, N = 128, 128, 128
_INPUT_SPECS = {"lhs_T": ((K, M), "float32"), "rhs": ((K, N), "float32")}


@nkigym_kernel
def _lhsT_matmul(lhs_T, rhs):
    lhs_T_sbuf = NKIAlloc(location="sbuf", shape=(K, M), dtype="float32")()
    rhs_sbuf = NKIAlloc(location="sbuf", shape=(K, N), dtype="float32")()
    psum_acc = NKIAlloc(location="psum", shape=(M, N), dtype="float32")()
    sbuf_prod = NKIAlloc(location="sbuf", shape=(M, N), dtype="float32")()
    hbm_out = NKIAlloc(location="shared_hbm", shape=(M, N), dtype="float32")()

    NKILoad()(src=lhs_T, dst=lhs_T_sbuf)
    NKILoad()(src=rhs, dst=rhs_sbuf)
    NKIMemset(value=0.0)(dst=psum_acc)
    NKIMatmul()(stationary=lhs_T_sbuf, moving=rhs_sbuf, dst=psum_acc)
    NKITensorCopy()(src=psum_acc, dst=sbuf_prod)
    NKIStore()(src=sbuf_prod, dst=hbm_out)
    return hbm_out


def test_direct_call_computes_matmul():
    rng = np.random.default_rng(0)
    lhs_T = rng.standard_normal((K, M)).astype(np.float32)
    rhs = rng.standard_normal((K, N)).astype(np.float32)
    actual = _lhsT_matmul(lhs_T=lhs_T, rhs=rhs)
    expected = lhs_T.T @ rhs
    assert np.allclose(np.asarray(actual), expected, atol=1e-4, rtol=1e-4)


def test_direct_call_returns_hbm_role():
    rng = np.random.default_rng(0)
    lhs_T = rng.standard_normal((K, M)).astype(np.float32)
    rhs = rng.standard_normal((K, N)).astype(np.float32)
    out = _lhsT_matmul(lhs_T=lhs_T, rhs=rhs)
    assert getattr(out, "role", None) == "shared_hbm"


def test_load_rejects_non_param_src():
    """NKILoad's per-op _check_roles fires when src is SBUF instead of HBM param."""
    sbuf = NKIAlloc(location="sbuf", shape=(16, 16), dtype="float32")()
    dst = NKIAlloc(location="sbuf", shape=(16, 16), dtype="float32")()
    with pytest.raises(TypeError, match="NKILoad.*expects HBM param"):
        NKILoad()(src=sbuf, dst=dst)


def test_store_rejects_non_sbuf_src():
    """NKIStore's per-op _check_roles fires when src is PSUM instead of SBUF."""
    psum = NKIAlloc(location="psum", shape=(16, 16), dtype="float32")()
    dst = NKIAlloc(location="shared_hbm", shape=(16, 16), dtype="float32")()
    with pytest.raises(TypeError, match="NKIStore.*expects sbuf"):
        NKIStore()(src=psum, dst=dst)


def test_nkigym_kernel_rejects_non_stored_non_hbm_return():
    """The decorator rejects kernels that return something other than hbm/stored-roled arrays."""

    @nkigym_kernel
    def bad_return(x):
        sbuf = NKIAlloc(location="sbuf", shape=(16, 16), dtype="float32")()
        NKILoad()(src=x, dst=sbuf)
        return sbuf

    inp = np.ones((16, 16), dtype=np.float32)
    with pytest.raises(TypeError, match="returned role='sbuf'"):
        bad_return(inp)
