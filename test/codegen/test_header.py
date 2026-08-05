"""Tests for code-generation headers and returns."""

import pytest

from nkigym.codegen import emit_header, emit_return
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
def generated_parts() -> tuple[str, str]:
    """Return the matmul fixture's generated header and return statement."""
    ir = build_initial_ir(_matmul_fixture, INPUT_SPECS)
    return emit_header(ir), emit_return(ir)


def test_header_contract(generated_parts: tuple[str, str]) -> None:
    """The header emits imports, signature, and shapes without body or return work."""
    header, _return = generated_parts
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


def test_return_contract(generated_parts: tuple[str, str]) -> None:
    """The return emitter emits only one indented return statement."""
    _header, return_source = generated_parts
    assert return_source == "    return hbm_out\n"
    assert "nl.ndarray" not in return_source
    assert "shared_hbm" not in return_source


def test_single_parameter_header_and_return() -> None:
    """A one-input kernel follows the same header and return split."""

    @nkigym_kernel
    def identity(x):
        sbuf_x = NKILoad()(src=x)
        hbm_y = NKIStore()(src=sbuf_x)
        return hbm_y

    ir = build_initial_ir(identity, {"x": ((128, 512), "bfloat16")})
    header = emit_header(ir)
    ret = emit_return(ir)
    assert "def nki_identity(x):" in header
    assert "assert x.shape == (128, 512)" in header
    assert "shared_hbm" not in header
    assert "return" not in header
    assert ret == "    return hbm_y\n"
