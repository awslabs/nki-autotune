"""Tests for :func:`nkigym.codegen.emit_header` and :func:`nkigym.codegen.emit_return`.

The renderer composes ``header + body + ret`` — header carries imports
+ ``@nki.jit`` + signature + per-param shape asserts; ret carries the
HBM allocation for the return tensor and the trailing ``return``.
"""

import pytest

from nkigym.codegen import emit_header, emit_return
from nkigym.ir import build_initial_ir
from nkigym.ops import nkigym_kernel
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.memset import NKIMemset
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy

K, M, N = 2048, 2048, 2048
_INPUT_SPECS: dict[str, tuple[int, ...]] = {"lhs_T": (K, M), "rhs": (K, N)}


@nkigym_kernel
def _matmul_fixture(lhs_T, rhs):
    """``lhs_T.T @ rhs`` fixture for header / return tests."""
    sbuf_lhs_T = NKIAlloc(location="sbuf", shape=(K, M), dtype="bfloat16")()
    sbuf_rhs = NKIAlloc(location="sbuf", shape=(K, N), dtype="bfloat16")()
    psum_acc = NKIAlloc(location="psum", shape=(M, N), dtype="float32")()
    sbuf_prod = NKIAlloc(location="sbuf", shape=(M, N), dtype="bfloat16")()
    hbm_out = NKIAlloc(location="shared_hbm", shape=(M, N), dtype="bfloat16")()

    NKILoad()(src=lhs_T, dst=sbuf_lhs_T)
    NKILoad()(src=rhs, dst=sbuf_rhs)
    NKIMemset(value=0.0)(dst=psum_acc)
    NKIMatmul()(stationary=sbuf_lhs_T, moving=sbuf_rhs, dst=psum_acc)
    NKITensorCopy()(src=psum_acc, dst=sbuf_prod)
    NKIStore()(src=sbuf_prod, dst=hbm_out)
    return hbm_out


@pytest.fixture(scope="module")
def header_source() -> str:
    """Build the matmul fixture's KernelIR and emit just the header."""
    ir = build_initial_ir(_matmul_fixture, _INPUT_SPECS)
    return emit_header(ir)


@pytest.fixture(scope="module")
def return_source() -> str:
    """Build the matmul fixture's KernelIR and emit just the return block."""
    ir = build_initial_ir(_matmul_fixture, _INPUT_SPECS)
    return emit_return(ir)


def test_emits_nki_imports(header_source: str) -> None:
    """The header brings ``nki``, ``nki.isa as nisa``, and ``nki.language as nl`` into scope."""
    assert "import nki" in header_source
    assert "import nki.isa as nisa" in header_source
    assert "import nki.language as nl" in header_source


def test_emits_jit_decorator_and_signature(header_source: str) -> None:
    """``@nki.jit`` precedes ``def nki_<func_name>(<param_names>):`` in signature order."""
    assert "@nki.jit" in header_source
    assert "def nki__matmul_fixture(lhs_T, rhs):" in header_source
    assert header_source.index("@nki.jit") < header_source.index("def nki__matmul_fixture")


def test_emits_shape_assertions_per_param(header_source: str) -> None:
    """Each param gets an ``assert <name>.shape == (...)`` against its declared ``input_specs`` shape."""
    assert f"assert lhs_T.shape == ({K}, {M})" in header_source
    assert f"assert rhs.shape == ({K}, {N})" in header_source


def test_header_omits_return_tensor_hbm_init(header_source: str) -> None:
    """Return-tensor init is :func:`emit_return`'s job — it must not appear in the header."""
    assert "shared_hbm" not in header_source
    assert "hbm_out = nl.ndarray" not in header_source


def test_header_omits_return_statement(header_source: str) -> None:
    """``return ...`` belongs in :func:`emit_return`."""
    assert "return hbm_out" not in header_source


def test_header_omits_intermediate_allocs(header_source: str) -> None:
    """SBUF/PSUM allocs are body work — the header never initialises them."""
    assert "sbuf_lhs_T = nl.ndarray" not in header_source
    assert "psum_acc = nl.ndarray" not in header_source
    assert "sbuf_prod = nl.ndarray" not in header_source


def test_emit_return_is_just_return_statement(return_source: str) -> None:
    """:func:`emit_return` emits exactly one indented ``return <return_name>`` line."""
    assert return_source == "    return hbm_out\n"


def test_emit_return_does_not_allocate_return_tensor(return_source: str) -> None:
    """The HBM allocation for the return tensor is :func:`emit_body`'s job, not :func:`emit_return`."""
    assert "nl.ndarray" not in return_source
    assert "shared_hbm" not in return_source


def test_single_param_kernel_header_and_return() -> None:
    """A 1-input kernel splits cleanly into header (signature + assert) and return (single ``return`` line)."""

    @nkigym_kernel
    def identity(x):
        sbuf_x = NKIAlloc(location="sbuf", shape=(128, 512), dtype="bfloat16")()
        hbm_y = NKIAlloc(location="shared_hbm", shape=(128, 512), dtype="bfloat16")()
        NKILoad()(src=x, dst=sbuf_x)
        NKIStore()(src=sbuf_x, dst=hbm_y)
        return hbm_y

    ir = build_initial_ir(identity, {"x": (128, 512)})
    header = emit_header(ir)
    ret = emit_return(ir)
    assert "def nki_identity(x):" in header
    assert "assert x.shape == (128, 512)" in header
    assert "shared_hbm" not in header
    assert "return" not in header
    assert ret == "    return hbm_y\n"
