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
    """Return (d0 ACC outer, d1 PAR outer, d3 PAR outer) iter vars on matmul's subtree.

    Post-Task 4 each axis yields outer+inner ForNodes. This helper picks
    the outer (trip) iter-var of each axis — the first ForNode per dim
    in the matmul spine.
    """
    d0 = module.axis_id_by_name("d0")
    d1 = module.axis_id_by_name("d1")
    d3 = module.axis_id_by_name("d3")
    fns = _collect_fornodes(module)
    d0_acc = next(
        iv.var_id for _, node in fns if (iv := node.iter_var).axis_id == d0 and iv.role == AxisRole.ACCUMULATION
    )
    d1_iv = next(
        iv.var_id
        for _, node in fns
        if (iv := node.iter_var).axis_id == d1 and iv.role == AxisRole.PARALLEL and _is_under_matmul(module, iv.var_id)
    )
    d3_iv = next(
        iv.var_id
        for _, node in fns
        if (iv := node.iter_var).axis_id == d3 and iv.role == AxisRole.PARALLEL and _is_under_matmul(module, iv.var_id)
    )
    return d0_acc, d1_iv, d3_iv


def _find_matmul_d1_inner_d3_outer(module):
    """Return the contiguous (d1_inner, d3_outer) pair in matmul's spine.

    Post-Task 4 the matmul spine is ``d0_outer > d0_inner > d1_outer >
    d1_inner > d3_outer > d3_inner > SBlock``. The first adjacent pair
    of ForNodes whose role-commute is PAR/PAR (both parallel) is the
    ``(d1_inner, d3_outer)`` pair — both bind PARALLEL iter-vars.
    """
    d1 = module.axis_id_by_name("d1")
    d3 = module.axis_id_by_name("d3")
    fns = _collect_fornodes(module)
    d1_ids = [
        n.iter_var.var_id for _, n in fns if n.iter_var.axis_id == d1 and _is_under_matmul(module, n.iter_var.var_id)
    ]
    d3_ids = [
        n.iter_var.var_id for _, n in fns if n.iter_var.axis_id == d3 and _is_under_matmul(module, n.iter_var.var_id)
    ]
    assert len(d1_ids) == 2 and len(d3_ids) == 2, "expected outer+inner per axis under matmul"
    d1_inner = d1_ids[1]
    d3_outer = d3_ids[0]
    return d1_inner, d3_outer


def test_reorder_par_par_adjacent_is_legal() -> None:
    """Swapping the adjacent (d1_inner, d3_outer) PAR/PAR pair in matmul's
    spine is legal — both iter-vars are PARALLEL, role-commute is PAR×PAR.
    """
    module = build_canonical_module(_matmul_large, _SPECS)
    d1_inner, d3_outer = _find_matmul_d1_inner_d3_outer(module)
    atom = Reorder(iter_var_ids=(d3_outer, d1_inner))
    assert atom.is_legal(module)


def test_reorder_apply_reshapes_tree() -> None:
    """After Reorder, the ForNode chain matches the requested order."""
    module = build_canonical_module(_matmul_large, _SPECS)
    d1_inner, d3_outer = _find_matmul_d1_inner_d3_outer(module)
    atom = Reorder(iter_var_ids=(d3_outer, d1_inner))
    new_mod = atom.apply(module)

    """Find the d3_outer ForNode in the new tree. Its only child must now
    be d1_inner (swap succeeded)."""
    fns = _collect_fornodes(new_mod)
    new_d3_outer = next(n for _, n in fns if n.iter_var.var_id == d3_outer)
    child = new_d3_outer.children[0]
    assert isinstance(child, ForNode)
    assert child.iter_var.var_id == d1_inner


def test_reorder_rejects_par_acc_with_write_on_par_dim() -> None:
    """Reordering d1 PAR outside d0 ACC where d1 indexes a write position is illegal.

    Matmul canonical spine has d0_outer ACC at the top with d1/d3 PAR
    deeper. d1 outer indexes psum_acc's M axis (matmul.reads_writes).
    Reordering (d1_outer, d0_inner) — hoisting d1 above d0_inner's
    ACC-subtree — would require d1-purity on the ACC subtree, but
    matmul's RMW write depends on d1 → impure. Illegal.
    """
    module = build_canonical_module(_matmul_large, _SPECS)
    d0 = module.axis_id_by_name("d0")
    d1 = module.axis_id_by_name("d1")
    """Find d0_inner (second d0 iter-var under matmul) and d1_outer."""
    fns = _collect_fornodes(module)
    d0_ids = [
        n.iter_var.var_id for _, n in fns if n.iter_var.axis_id == d0 and _is_under_matmul(module, n.iter_var.var_id)
    ]
    d1_ids = [
        n.iter_var.var_id for _, n in fns if n.iter_var.axis_id == d1 and _is_under_matmul(module, n.iter_var.var_id)
    ]
    assert len(d0_ids) >= 2 and len(d1_ids) >= 1
    d0_inner = d0_ids[1]
    d1_outer = d1_ids[0]
    atom = Reorder(iter_var_ids=(d1_outer, d0_inner))
    assert not atom.is_legal(module)


def test_reorder_apply_raises_on_illegal() -> None:
    """Calling apply on an illegal reorder raises AtomLegalityError."""
    module = build_canonical_module(_matmul_large, _SPECS)
    d0 = module.axis_id_by_name("d0")
    d1 = module.axis_id_by_name("d1")
    fns = _collect_fornodes(module)
    d0_ids = [
        n.iter_var.var_id for _, n in fns if n.iter_var.axis_id == d0 and _is_under_matmul(module, n.iter_var.var_id)
    ]
    d1_ids = [
        n.iter_var.var_id for _, n in fns if n.iter_var.axis_id == d1 and _is_under_matmul(module, n.iter_var.var_id)
    ]
    assert len(d0_ids) >= 2 and len(d1_ids) >= 1
    atom = Reorder(iter_var_ids=(d1_ids[0], d0_ids[1]))
    with pytest.raises(AtomLegalityError):
        atom.apply(module)


def test_reorder_round_trip_restores_canonical() -> None:
    """Reorder(a, b) + Reorder(b, a) restores the original loop order."""
    module = build_canonical_module(_matmul_large, _SPECS)
    d1_inner, d3_outer = _find_matmul_d1_inner_d3_outer(module)

    first = Reorder(iter_var_ids=(d3_outer, d1_inner)).apply(module)
    second = Reorder(iter_var_ids=(d1_inner, d3_outer)).apply(first)

    """Original order: d1_inner > d3_outer. After round-trip, same chain."""
    fns = _collect_fornodes(second)
    restored_d1_inner = next(n for _, n in fns if n.iter_var.var_id == d1_inner)
    child = restored_d1_inner.children[0]
    assert isinstance(child, ForNode)
    assert child.iter_var.var_id == d3_outer


def test_reorder_requires_contiguous_chain() -> None:
    """Iter vars from disjoint subtrees cannot be reordered together."""
    module = build_canonical_module(_matmul_large, _SPECS)
    d0 = module.axis_id_by_name("d0")
    d0_acc, _d1, _d3 = _find_matmul_subtree_iter_vars(module)
    """Find a d0 from the LOAD subtree (non-matmul)."""
    fns = _collect_fornodes(module)
    d0_load = next(
        iv.var_id
        for _, node in fns
        if (iv := node.iter_var).axis_id == d0 and iv.role == AxisRole.PARALLEL and iv.var_id != d0_acc
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
    d1_inner, d3_outer = _find_matmul_d1_inner_d3_outer(module)
    orig_source = render(module)
    new_mod = Reorder(iter_var_ids=(d3_outer, d1_inner)).apply(module)
    new_source = render(new_mod)

    """Both renders should be parseable Python."""
    ast.parse(new_source)
    """Loads + stores + memset lines unchanged."""
    assert "nisa.memset" in new_source
    assert new_source.count("nisa.dma_copy") == orig_source.count("nisa.dma_copy")
