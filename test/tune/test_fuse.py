"""Tests for the Fuse atom.

Same-axis Fuse requires outer.extent == 1 and eagerly rewrites access
patterns; no side-table. Cross-axis Fuse with dependent access patterns
is rejected at ``is_legal``.
"""

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
    """Return the (outer_var_id, inner_var_id) of the tensor_copy d3
    outer/inner pair.

    Post-Task 4 the tensor_copy spine is ``d1_outer(16) > d1_inner(128)
    > d3_outer(1) > d3_inner(2048) > SBlock``. We pick the d3 pair
    because its abstract axis is ``F`` with ``MAX_TILE_SIZE[F] = None`` —
    Task 7's MIN/MAX check on innermost leaves the fuse legal. (The d1
    pair would fuse to extent 2048 exceeding ``MAX_TILE_SIZE[P]=128``.)
    """
    d1 = module.axis_id_by_name("d1")
    d3 = module.axis_id_by_name("d3")
    outer = _find_subtree_with_op(module, "NKITensorCopy")
    assert isinstance(outer, ForNode) and outer.iter_var.axis_id == d1
    assert len(outer.children) == 1
    d1_inner = outer.children[0]
    assert isinstance(d1_inner, ForNode) and d1_inner.iter_var.axis_id == d1
    assert len(d1_inner.children) == 1
    d3_outer = d1_inner.children[0]
    assert isinstance(d3_outer, ForNode) and d3_outer.iter_var.axis_id == d3
    assert len(d3_outer.children) == 1
    d3_inner = d3_outer.children[0]
    assert isinstance(d3_inner, ForNode) and d3_inner.iter_var.axis_id == d3
    return d3_outer.iter_var.var_id, d3_inner.iter_var.var_id


def test_fuse_adjacent_par_par_creates_synthetic_dim() -> None:
    """Fusing the tensor_copy d3 outer(1) > d3 inner(2048) pair yields a
    single ForNode. Same-axis fuse preserves the axis_id (d3); extent
    becomes ``1 * 2048 == 2048``. Post-Task 4 canonical splits each axis
    into outer+inner ForNodes; Task 7's MIN/MAX check leaves the d3 pair
    legal because tensor_copy's ``F`` axis has ``MAX_TILE_SIZE=None``."""
    module = build_canonical_module(_matmul_large, _SPECS)
    d1 = module.axis_id_by_name("d1")
    d3 = module.axis_id_by_name("d3")
    outer_id, inner_id = _tensor_copy_pair(module)
    atom = Fuse(outer_iter_var_id=outer_id, inner_iter_var_id=inner_id)
    assert atom.is_legal(module)
    new_mod = atom.apply(module)

    """The fused node lives below the d1 outer+inner ForNodes; walk down."""
    subtree_root = _find_subtree_with_op(new_mod, "NKITensorCopy")
    assert isinstance(subtree_root, ForNode) and subtree_root.iter_var.axis_id == d1
    d1_inner = subtree_root.children[0]
    assert isinstance(d1_inner, ForNode) and d1_inner.iter_var.axis_id == d1
    fused_node = d1_inner.children[0]
    assert isinstance(fused_node, ForNode)
    """Same-axis fuse preserves axis_id (d3); extent becomes the product."""
    assert fused_node.iter_var.axis_id == d3
    assert fused_node.iter_var.extent == 2048


def test_fuse_registers_synthetic_dim_info() -> None:
    """Same-axis fuse preserves axis_id; no new axis is registered. The
    existing axis's ``total_size`` is unchanged — identity remains the
    original ``d3`` axis."""
    module = build_canonical_module(_matmul_large, _SPECS)
    d3 = module.axis_id_by_name("d3")
    outer_id, inner_id = _tensor_copy_pair(module)
    before_axes = set(module.axes)
    atom = Fuse(outer_iter_var_id=outer_id, inner_iter_var_id=inner_id)
    new_mod = atom.apply(module)
    """Same-axis fuse does not allocate a new Axis."""
    assert set(new_mod.axes) == before_axes
    """The original axis still carries its declared total_size."""
    assert new_mod.axes[d3].total_size == 2048


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


_SPECS_LARGE = {
    "lhs_T": {"shape": (2048, 2048), "dtype": "bfloat16"},
    "rhs": {"shape": (2048, 2048), "dtype": "bfloat16"},
}


def test_fuse_rejects_when_fused_extent_exceeds_matmul_n_max() -> None:
    """Fusing matmul N outer+inner (trip=4 * tile=512 -> 2048) violates MAX_TILE_SIZE[N]=512.

    Canonical matmul N lives on dim ``d3`` (the N axis after unification)
    as ``ForNode(extent=4) > ForNode(extent=512)``. The inner IS the
    matmul SBlock's innermost for dim ``d3``. Fusing gives extent=2048 >
    MAX=512 -> must be rejected.
    """
    module = build_canonical_module(_matmul_large, input_specs=_SPECS_LARGE)
    d3 = module.axis_id_by_name("d3")

    def find_matmul_n_pair(module):
        """Return (outer_id, inner_id) for matmul's N axis (d3): outer extent=4, inner extent=512."""
        outer_id = None
        inner_id = None

        def scan(node):
            nonlocal outer_id, inner_id
            if isinstance(node, ForNode):
                iv = node.iter_var
                if iv.axis_id == d3:
                    if iv.extent == 4 and outer_id is None:
                        outer_id = iv.var_id
                    elif iv.extent == 512 and inner_id is None and outer_id is not None:
                        inner_id = iv.var_id
                for c in node.children:
                    scan(c)

        for r in module.body:
            scan(r)
        return outer_id, inner_id

    outer_id, inner_id = find_matmul_n_pair(module)
    assert outer_id is not None and inner_id is not None, "could not find d3 outer+inner"
    atom = Fuse(outer_iter_var_id=outer_id, inner_iter_var_id=inner_id)
    """Fused extent = 4 * 512 = 2048 > MAX_TILE_SIZE[N]=512 -> illegal."""
    assert atom.is_legal(module) is False
