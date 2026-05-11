"""Unit tests for the Fuse atom on the iter-var IR."""

import ast

import pytest

from nkigym.codegen.canonical import build_canonical_module
from nkigym.codegen.ir import ForNode, SBlock
from nkigym.codegen.render import render
from nkigym.ops import nkigym_kernel
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.memset import NKIMemset
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.tune import AtomLegalityError
from nkigym.tune.fuse import Fuse, enumerate_fuse_atoms


@nkigym_kernel
def _matmul_large(lhs_T, rhs):
    """2048 matmul fixture."""
    lhs_T_sbuf = NKIAlloc(location="sbuf", shape=(2048, 2048), dtype="bfloat16")()
    rhs_sbuf = NKIAlloc(location="sbuf", shape=(2048, 2048), dtype="bfloat16")()
    psum_acc = NKIAlloc(location="psum", shape=(2048, 2048), dtype="float32")()
    sbuf_prod = NKIAlloc(location="sbuf", shape=(2048, 2048), dtype="bfloat16")()
    hbm_out = NKIAlloc(location="hbm", shape=(2048, 2048), dtype="bfloat16")()
    NKILoad()(src=lhs_T, dst=lhs_T_sbuf)
    NKILoad()(src=rhs, dst=rhs_sbuf)
    NKIMemset(value=0.0)(dst=psum_acc)
    NKIMatmul()(stationary=lhs_T_sbuf, moving=rhs_sbuf, dst=psum_acc)
    NKITensorCopy()(src=psum_acc, dst=sbuf_prod)
    NKIStore()(src=sbuf_prod, dst=hbm_out)
    return hbm_out


_SPECS = {"lhs_T": {"shape": (2048, 2048), "dtype": "bfloat16"}, "rhs": {"shape": (2048, 2048), "dtype": "bfloat16"}}


def _find_subtree_with_op(module, op_name: str) -> ForNode:
    """Return the root ForNode whose subtree contains an SBlock with ``op_name``."""

    def has_op(node) -> bool:
        if isinstance(node, SBlock):
            return any(c.op_cls.__name__ == op_name for c in node.body)
        return any(has_op(c) for c in node.children)

    for root in module.body:
        if isinstance(root, ForNode) and has_op(root):
            return root
    raise AssertionError(f"No subtree with op {op_name}")


def _tensor_copy_pair(module) -> tuple[int, int]:
    """Return the (outer_var_id, inner_var_id) of the tensor_copy d1-d3 nest."""
    outer = _find_subtree_with_op(module, "NKITensorCopy")
    inner = outer.children[0]
    assert isinstance(outer, ForNode) and isinstance(inner, ForNode)
    return outer.iter_var.var_id, inner.iter_var.var_id


def test_fuse_adjacent_par_par_creates_synthetic_dim() -> None:
    """Fusing the tensor_copy d1(16) > d3(1) pair yields a single ForNode on
    the synthetic ``d1_x_d3`` dim with extent 16. Under per-op tiling,
    tensor_copy's d3 loop has extent=1, unlike matmul's d3 extent=4."""
    module = build_canonical_module(_matmul_large, _SPECS)
    outer_id, inner_id = _tensor_copy_pair(module)
    atom = Fuse(outer_iter_var_id=outer_id, inner_iter_var_id=inner_id)
    assert atom.is_legal(module)
    new_mod = atom.apply(module)

    new_subtree = _find_subtree_with_op(new_mod, "NKITensorCopy")
    assert isinstance(new_subtree, ForNode)
    assert new_subtree.iter_var.dim_id == "d1_x_d3"
    assert new_subtree.iter_var.extent == 16


def test_fuse_registers_synthetic_dim_info() -> None:
    """The new dim appears in ``module.dims`` with total_size equal to the
    fused extent. Under per-op tiling, tensor_copy's d1×d3 = 16×1 = 16."""
    module = build_canonical_module(_matmul_large, _SPECS)
    outer_id, inner_id = _tensor_copy_pair(module)
    atom = Fuse(outer_iter_var_id=outer_id, inner_iter_var_id=inner_id)
    new_mod = atom.apply(module)
    assert "d1_x_d3" in new_mod.dims
    """Synthetic dim lives in module.dims with total_size = fused extent."""
    assert new_mod.dims["d1_x_d3"].total_size == 16


def test_fuse_renderer_emits_div_and_mod() -> None:
    """The rendered source opens ``for i_d1_x_d3_0 in range(16):`` and the
    body's slot expressions spell the div/mod decomposition. Under per-op
    tiling, tensor_copy's d3 has extent=1, so the modulus is 1."""
    module = build_canonical_module(_matmul_large, _SPECS)
    outer_id, inner_id = _tensor_copy_pair(module)
    atom = Fuse(outer_iter_var_id=outer_id, inner_iter_var_id=inner_id)
    new_mod = atom.apply(module)
    source = render(new_mod)
    assert "for i_d1_x_d3_0 in range(16):" in source
    assert "// 1" in source or "% 1" in source


def test_fuse_non_adjacent_rejects() -> None:
    """Pairing iter vars from sibling subtrees is illegal — the outer
    ForNode's sole child ForNode must bind the requested inner id."""
    module = build_canonical_module(_matmul_large, _SPECS)
    load_subtree = _find_subtree_with_op(module, "NKILoad")
    matmul_subtree = _find_subtree_with_op(module, "NKIMatmul")
    load_outer_id = load_subtree.iter_var.var_id
    """Pick a d3 iter var from the matmul subtree."""
    matmul_inner = matmul_subtree.children[0]
    assert isinstance(matmul_inner, ForNode)
    matmul_d3 = matmul_inner.children[0]
    assert isinstance(matmul_d3, ForNode)
    atom = Fuse(outer_iter_var_id=load_outer_id, inner_iter_var_id=matmul_d3.iter_var.var_id)
    assert not atom.is_legal(module)


def test_fuse_apply_raises_on_illegal() -> None:
    """``apply`` re-validates and raises ``AtomLegalityError`` on an illegal atom."""
    module = build_canonical_module(_matmul_large, _SPECS)
    load_subtree = _find_subtree_with_op(module, "NKILoad")
    matmul_subtree = _find_subtree_with_op(module, "NKIMatmul")
    matmul_inner = matmul_subtree.children[0]
    assert isinstance(matmul_inner, ForNode)
    matmul_d3 = matmul_inner.children[0]
    assert isinstance(matmul_d3, ForNode)
    atom = Fuse(outer_iter_var_id=load_subtree.iter_var.var_id, inner_iter_var_id=matmul_d3.iter_var.var_id)
    with pytest.raises(AtomLegalityError):
        atom.apply(module)


def test_enumerate_fuse_atoms_yields_legal_atoms_only() -> None:
    """Every atom emitted by the enumerator passes ``is_legal``."""
    module = build_canonical_module(_matmul_large, _SPECS)
    atoms = enumerate_fuse_atoms(module)
    assert atoms, "expected at least one fuse candidate in canonical matmul"
    for atom in atoms:
        assert atom.is_legal(module), f"enumerator emitted illegal atom {atom!r}"


def test_fuse_rendered_source_parses() -> None:
    """After Fuse + render, the produced Python source parses cleanly."""
    module = build_canonical_module(_matmul_large, _SPECS)
    outer_id, inner_id = _tensor_copy_pair(module)
    new_mod = Fuse(outer_iter_var_id=outer_id, inner_iter_var_id=inner_id).apply(module)
    source = render(new_mod)
    ast.parse(source)
