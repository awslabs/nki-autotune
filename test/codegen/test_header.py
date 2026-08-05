"""Tests for code-generation headers and returns."""

import pytest

from nkigym.codegen import emit_header
from nkigym.ir import build_initial_ir
from nkigym.ops import nkigym_kernel
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy

K, M, N = 2048, 2048, 2048
INPUT_SPECS: dict[str, tuple[tuple[int, ...], str]] = {"lhs_T": ((K, M), "bfloat16"), "rhs": ((K, N), "bfloat16")}


@nkigym_kernel
def _matmul_fixture(lhs_T, rhs):
    """Return a canonical matmul for header and return tests."""
    sbuf_lhs_T = NKILoad()(src=lhs_T)
    sbuf_rhs = NKILoad()(src=rhs)
    psum_acc = NKIMatmul()(stationary=sbuf_lhs_T, moving=sbuf_rhs)
    sbuf_prod = NKITensorCopy()(src=psum_acc)
    hbm_out = NKIStore()(src=sbuf_prod)
    return hbm_out


@pytest.fixture(scope="module")
def generated_header() -> str:
    """Return the matmul fixture's generated header."""
    ir = build_initial_ir(_matmul_fixture, INPUT_SPECS)
    return emit_header(ir)


def test_header_contract(generated_header: str) -> None:
    """The header emits imports, signature, and shapes without body or return work."""
    header = generated_header
    assert "import nki" in header
    assert "import nki.isa as nisa" in header
    assert "import nki.language as nl" in header
    assert "@nki.jit" in header
    assert "def nki__matmul_fixture(lhs_T, rhs):" in header
    assert header.index("@nki.jit") < header.index("def nki__matmul_fixture")
    assert f"assert lhs_T.shape == ({K}, {M})" in header
    assert f"assert rhs.shape == ({K}, {N})" in header
    assert "shared_hbm" not in header
    assert "hbm_out = nl.ndarray" not in header
    assert "return hbm_out" not in header
    assert "sbuf_lhs_T = nl.ndarray" not in header
    assert "psum_acc = nl.ndarray" not in header
    assert "sbuf_prod = nl.ndarray" not in header
