"""Unit tests for the Reorder atom (iter-var-keyed, n-ary)."""

import pytest

from nkigym.codegen.canonical import build_canonical_module
from nkigym.codegen.ir import ForNode, SBlock
from nkigym.codegen.render import render
from nkigym.ops import nkigym_kernel
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.base import AxisRole
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.memset import NKIMemset
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.tune import AtomLegalityError
from nkigym.tune.reorder import Reorder, enumerate_reorder_atoms


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


def _collect_fornodes(module):
    """Walk module.body and return all ForNodes with their paths + iter vars."""
    results = []

    def walk(node, path):
        if isinstance(node, ForNode):
            results.append((path, node))
            for i, c in enumerate(node.children):
                walk(c, path + (i,))

    for i, root in enumerate(module.body):
        walk(root, (i,))
    return results


def _is_under_matmul(module, iv_id):
    """True if the ForNode binding iv_id is in the matmul's subtree."""
    fns = _collect_fornodes(module)
    result = False
    for path, node in fns:
        if node.iter_var.var_id == iv_id:
            descendant = module.body[path[0]]

            def has_mm(n):
                if isinstance(n, SBlock):
                    return any(c.op_cls.__name__ == "NKIMatmul" for c in n.body)
                return any(has_mm(c) for c in n.children)

            result = has_mm(descendant)
            break
    return result


def _find_matmul_subtree_iter_vars(module):
    """Return (d0 ACC, d1 PAR, d3 PAR) iter vars on matmul's subtree."""
    fns = _collect_fornodes(module)
    d0_acc = next(
        iv.var_id for _, node in fns if (iv := node.iter_var).dim_id == "d0" and iv.role == AxisRole.ACCUMULATION
    )
    d1 = next(
        iv.var_id
        for _, node in fns
        if (iv := node.iter_var).dim_id == "d1" and iv.role == AxisRole.PARALLEL and _is_under_matmul(module, iv.var_id)
    )
    d3 = next(
        iv.var_id
        for _, node in fns
        if (iv := node.iter_var).dim_id == "d3" and iv.role == AxisRole.PARALLEL and _is_under_matmul(module, iv.var_id)
    )
    return d0_acc, d1, d3


def test_reorder_par_par_adjacent_is_legal() -> None:
    """Swapping d1 PAR and d3 PAR adjacent loops in matmul subtree is legal."""
    module = build_canonical_module(_matmul_large, _SPECS)
    d0_acc, d1, d3 = _find_matmul_subtree_iter_vars(module)
    """Canonical order: d0_acc > d1_par > d3_par > matmul. Reorder to d0, d3, d1."""
    atom = Reorder(iter_var_ids=(d0_acc, d3, d1))
    assert atom.is_legal(module)


def test_reorder_apply_reshapes_tree() -> None:
    """After Reorder, the tree's ForNode chain matches the requested order."""
    module = build_canonical_module(_matmul_large, _SPECS)
    d0_acc, d1, d3 = _find_matmul_subtree_iter_vars(module)
    atom = Reorder(iter_var_ids=(d0_acc, d3, d1))
    new_mod = atom.apply(module)

    """Find the matmul subtree root."""
    fns = _collect_fornodes(new_mod)
    root_for_d0 = next(n for _, n in fns if n.iter_var.var_id == d0_acc)
    """Top to bottom: d0 → d3 → d1 → SBlock."""
    child = root_for_d0.children[0]
    assert isinstance(child, ForNode)
    assert child.iter_var.var_id == d3
    grand = child.children[0]
    assert isinstance(grand, ForNode)
    assert grand.iter_var.var_id == d1


def test_reorder_rejects_par_acc_with_write_on_par_dim() -> None:
    """Reordering d1 PAR outside d0 ACC where d1 indexes a write position is illegal.

    Here: matmul canonical has d0 ACC > d1 PAR > d3 PAR > matmul. d1 indexes
    psum_acc's M axis (matmul.reads_writes). Reordering d1, d0 would require
    d1-purity on the ACC subtree — but matmul's RMW write depends on d1 →
    impure. Illegal.
    """
    module = build_canonical_module(_matmul_large, _SPECS)
    d0_acc, d1, _d3 = _find_matmul_subtree_iter_vars(module)
    atom = Reorder(iter_var_ids=(d1, d0_acc))
    assert not atom.is_legal(module)


def test_reorder_apply_raises_on_illegal() -> None:
    """Calling apply on an illegal reorder raises AtomLegalityError."""
    module = build_canonical_module(_matmul_large, _SPECS)
    d0_acc, d1, _d3 = _find_matmul_subtree_iter_vars(module)
    atom = Reorder(iter_var_ids=(d1, d0_acc))
    with pytest.raises(AtomLegalityError):
        atom.apply(module)


def test_reorder_round_trip_restores_canonical() -> None:
    """Reorder(a, b) + Reorder(b, a) restores the original tree's loop order."""
    module = build_canonical_module(_matmul_large, _SPECS)
    d0_acc, d1, d3 = _find_matmul_subtree_iter_vars(module)

    first = Reorder(iter_var_ids=(d0_acc, d3, d1)).apply(module)
    second = Reorder(iter_var_ids=(d0_acc, d1, d3)).apply(first)

    """Original matmul subtree order: d0_acc → d1 → d3. After 2 reorders,
    same order restored."""
    fns = _collect_fornodes(second)
    root_for_d0 = next(n for _, n in fns if n.iter_var.var_id == d0_acc)
    child = root_for_d0.children[0]
    assert isinstance(child, ForNode)
    assert child.iter_var.var_id == d1
    grand = child.children[0]
    assert isinstance(grand, ForNode)
    assert grand.iter_var.var_id == d3


def test_reorder_requires_contiguous_chain() -> None:
    """Iter vars from disjoint subtrees cannot be reordered together."""
    module = build_canonical_module(_matmul_large, _SPECS)
    d0_acc, _d1, _d3 = _find_matmul_subtree_iter_vars(module)
    """Find a d0 from the LOAD subtree (non-matmul)."""
    fns = _collect_fornodes(module)
    d0_load = next(
        iv.var_id
        for _, node in fns
        if (iv := node.iter_var).dim_id == "d0" and iv.role == AxisRole.PARALLEL and iv.var_id != d0_acc
    )
    """Attempt to reorder a load's d0 with matmul's d0 — disjoint subtrees, illegal."""
    atom = Reorder(iter_var_ids=(d0_load, d0_acc))
    assert not atom.is_legal(module)


def test_enumerate_reorder_atoms_yields_legal_atoms_only() -> None:
    """Enumerator only yields legal reorderings."""
    module = build_canonical_module(_matmul_large, _SPECS)
    atoms = enumerate_reorder_atoms(module)
    for atom in atoms:
        assert atom.is_legal(module)
    assert len(atoms) > 0


def test_reorder_preserves_other_subtrees() -> None:
    """Reorder on matmul subtree does not affect sibling load/store subtrees."""
    import ast

    module = build_canonical_module(_matmul_large, _SPECS)
    d0_acc, d1, d3 = _find_matmul_subtree_iter_vars(module)
    orig_source = render(module)
    new_mod = Reorder(iter_var_ids=(d0_acc, d3, d1)).apply(module)
    new_source = render(new_mod)

    """Both renders should be parseable Python."""
    ast.parse(new_source)
    """Loads + stores + memset lines unchanged."""
    assert "nisa.memset" in new_source
    assert new_source.count("nisa.dma_copy") == orig_source.count("nisa.dma_copy")
