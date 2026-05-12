"""Local fp32 CPU-sim verify helper."""

import numpy as np
import pytest

from nkigym.codegen.canonical import build_canonical_module
from nkigym.codegen.render import render
from nkigym.ops import nkigym_kernel
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.memset import NKIMemset
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.tune.verify import _verify, _verify_fns

K, M, N = 512, 512, 512


@nkigym_kernel
def _lhsT_matmul(lhs_T, rhs):
    lhs_T_sbuf = NKIAlloc(location="sbuf", shape=(K, M), dtype="bfloat16")()
    rhs_sbuf = NKIAlloc(location="sbuf", shape=(K, N), dtype="bfloat16")()
    psum_acc = NKIAlloc(location="psum", shape=(M, N), dtype="float32")()
    sbuf_prod = NKIAlloc(location="sbuf", shape=(M, N), dtype="bfloat16")()
    hbm_out = NKIAlloc(location="hbm", shape=(M, N), dtype="bfloat16")()

    NKILoad()(src=lhs_T, dst=lhs_T_sbuf)
    NKILoad()(src=rhs, dst=rhs_sbuf)
    NKIMemset(value=0.0)(dst=psum_acc)
    NKIMatmul()(stationary=lhs_T_sbuf, moving=rhs_sbuf, dst=psum_acc)
    NKITensorCopy()(src=psum_acc, dst=sbuf_prod)
    NKIStore()(src=sbuf_prod, dst=hbm_out)
    return hbm_out


_INPUT_SPECS = {"lhs_T": ((K, M), "bfloat16"), "rhs": ((K, N), "bfloat16")}
_CANONICAL_SPECS = {name: {"shape": shape, "dtype": dt} for name, (shape, dt) in _INPUT_SPECS.items()}


def test_verify_passes_for_canonical_render():
    """Rendered canonical source matches the decorated f_nkigym exactly."""
    module = build_canonical_module(_lhsT_matmul, _CANONICAL_SPECS)
    source = render(module)
    _verify(source, _lhsT_matmul, _INPUT_SPECS)


def test_verify_raises_when_golden_diverges_from_kernel():
    """Rendered canonical kernel paired with a golden that computes something else must raise."""
    module = build_canonical_module(_lhsT_matmul, _CANONICAL_SPECS)
    source = render(module)

    def wrong_golden(lhs_T, rhs):
        """Pretends to be the kernel's golden but returns zeros."""
        return np.zeros((M, N), dtype=np.float32)

    wrong_golden.__name__ = _lhsT_matmul.__name__

    with pytest.raises(AssertionError, match="kernel vs f_nkigym: max_abs"):
        _verify(source, wrong_golden, _INPUT_SPECS)


def test_verify_fns_passes_when_fns_agree():
    """Decorated f_nkigym (runs numpy _run methods) matches an equivalent plain-numpy reference."""

    def f_numpy(lhs_T, rhs):
        return lhs_T.T @ rhs

    _verify_fns(_lhsT_matmul, f_numpy, _INPUT_SPECS)


def test_verify_fns_raises_when_fns_disagree():
    """f_nkigym computes lhs_T.T @ rhs; a reference that computes lhs_T @ rhs must raise."""

    def f_numpy(lhs_T, rhs):
        return lhs_T @ rhs

    with pytest.raises(AssertionError, match="f_nkigym vs f_numpy: max_abs"):
        _verify_fns(_lhsT_matmul, f_numpy, _INPUT_SPECS)
