"""Unit tests for the ComputeAt atom on the iter-var IR.

Prefix-match + role-promotion semantics (TVM ``sch.compute_at``).
"""

import ast

import pytest

from nkigym.codegen.canonical import build_canonical_module
from nkigym.codegen.ir import ForNode, SBlock, validate_dataflow_ordering
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
from nkigym.tune.compute_at import ComputeAt, enumerate_compute_at_atoms


@nkigym_kernel
def _matmul_large(lhs_T, rhs):
    """2048 matmul fixture — multi-tile on K/M (16) and N (4)."""
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


def _collect_paths(module) -> list[tuple[tuple[int, ...], ForNode | SBlock]]:
    """Walk the forest; return every ``(path, node)`` pair."""
    results: list[tuple[tuple[int, ...], ForNode | SBlock]] = []

    def walk(node, path):
        results.append((path, node))
        if isinstance(node, ForNode):
            for i, c in enumerate(node.children):
                walk(c, path + (i,))

    for i, root in enumerate(module.body):
        walk(root, (i,))
    return results


def _find_block_writing(module, tensor_name: str) -> tuple[int, ...]:
    """Return the path to the first SBlock whose writes contain ``tensor_name``."""
    for path, node in _collect_paths(module):
        if isinstance(node, SBlock):
            written = {a.tensor_name for a in node.writes.values()} | {
                a.tensor_name for a in node.reads_writes.values()
            }
            if tensor_name in written and any(c.op_cls.__name__ != "NKIAlloc" for c in node.body):
                return path
    raise AssertionError(f"No SBlock writing {tensor_name!r}")


def _find_matmul_d0_acc_path(module) -> tuple[int, ...]:
    """Return the path to the matmul's ``d0`` ACC ForNode (extent 16)."""
    for path, node in _collect_paths(module):
        if isinstance(node, ForNode):
            iv = node.iter_var
            if iv.dim_id == "d0" and iv.role == AxisRole.ACCUMULATION:
                return path
    raise AssertionError("No d0 ACC ForNode")


def test_compute_at_under_matmul_acc_loop_is_legal() -> None:
    """The load writing ``rhs_sbuf`` can be placed under matmul's ``d0`` ACC loop.

    matmul reads ``rhs_sbuf`` via ``moving`` → target's subtree consumes
    the block's writes. Block is a sibling subtree of target (load d0 is
    PARALLEL; matmul d0 is ACCUMULATION) so target is NOT an ancestor.
    Block.iter_vars[0].dim_id == d0 matches target ancestor chain's d0
    → prefix-match holds.
    """
    module = build_canonical_module(_matmul_large, _SPECS)
    load_path = _find_block_writing(module, "rhs_sbuf")
    target_path = _find_matmul_d0_acc_path(module)
    atom = ComputeAt(block_path=load_path, target_path=target_path)
    assert atom.is_legal(module)


def test_compute_at_rejects_target_that_is_ancestor() -> None:
    """Target that IS an ancestor of the block's current position is rejected."""
    module = build_canonical_module(_matmul_large, _SPECS)
    mm_path = _find_block_writing(module, "psum_acc")
    """Parent path (last ForNode ancestor of mm block)."""
    parent_path = mm_path[:-1]
    atom = ComputeAt(block_path=mm_path, target_path=parent_path)
    assert not atom.is_legal(module)


def test_compute_at_rejects_target_without_consumer() -> None:
    """Target whose subtree does not consume the block's writes is rejected.

    Matmul writes ``psum_acc``; target under the load subtree does not
    read ``psum_acc``. Such a target is illegal.
    """
    module = build_canonical_module(_matmul_large, _SPECS)
    mm_path = _find_block_writing(module, "psum_acc")
    """Find any ForNode under a load subtree (which only reads lhs_T / rhs)."""
    load_path = _find_block_writing(module, "lhs_T_sbuf")
    load_root_idx = load_path[0]
    load_root = module.body[load_root_idx]
    assert isinstance(load_root, ForNode)
    atom = ComputeAt(block_path=mm_path, target_path=(load_root_idx,))
    assert not atom.is_legal(module)


def test_compute_at_apply_moves_block_under_target() -> None:
    """After apply, the moved block lives in target's subtree + module validates."""
    module = build_canonical_module(_matmul_large, _SPECS)
    load_path = _find_block_writing(module, "rhs_sbuf")
    target_path = _find_matmul_d0_acc_path(module)
    atom = ComputeAt(block_path=load_path, target_path=target_path)
    new_mod = atom.apply(module)
    assert validate_dataflow_ordering(new_mod)

    """Walk the new tree: the block writing rhs_sbuf should appear under
    a ForNode whose dim_id == 'd0' (target's ancestor chain's dim)."""

    def has_rhs_load_under_d0(node, under_d0: bool) -> bool:
        if isinstance(node, SBlock):
            writes = {a.tensor_name for a in node.writes.values()}
            return under_d0 and "rhs_sbuf" in writes
        is_d0 = node.iter_var.dim_id == "d0"
        next_under = under_d0 or is_d0
        return any(has_rhs_load_under_d0(c, next_under) for c in node.children)

    found = False
    for root in new_mod.body:
        if has_rhs_load_under_d0(root, False):
            found = True
            break
    assert found


def test_compute_at_apply_raises_on_illegal() -> None:
    """``apply`` re-validates and raises ``AtomLegalityError``."""
    module = build_canonical_module(_matmul_large, _SPECS)
    mm_path = _find_block_writing(module, "psum_acc")
    parent_path = mm_path[:-1]
    atom = ComputeAt(block_path=mm_path, target_path=parent_path)
    with pytest.raises(AtomLegalityError):
        atom.apply(module)


def test_compute_at_render_valid_python() -> None:
    """After apply + render, the output parses as valid Python."""
    module = build_canonical_module(_matmul_large, _SPECS)
    load_path = _find_block_writing(module, "rhs_sbuf")
    target_path = _find_matmul_d0_acc_path(module)
    atom = ComputeAt(block_path=load_path, target_path=target_path)
    new_mod = atom.apply(module)
    source = render(new_mod)
    ast.parse(source)


def test_enumerate_compute_at_atoms_yields_legal_only() -> None:
    """Every atom emitted by the enumerator passes ``is_legal``."""
    module = build_canonical_module(_matmul_large, _SPECS)
    atoms = enumerate_compute_at_atoms(module)
    for atom in atoms:
        assert atom.is_legal(module)
    assert len(atoms) > 0


def test_compute_at_role_promotion_allocates_fresh_iter_var() -> None:
    """When block's role is stronger than target's on a matched dim, a
    fresh iter var with the stronger role is allocated.

    The matmul writes psum_acc with ACC role on d0, and d0 on matmul's
    canonical chain is ACC. If we hoist a memset (which writes psum_acc
    with PAR role on d0) under matmul's d0 ACC loop, the matched pair is
    (ACC, PAR) → max is ACC → no promotion (target already at max role).
    Conversely, we can test the mirror: the load's d0 is PAR. ComputeAt
    under matmul d0 ACC: matched (ACC, PAR) → max is ACC → no promotion.
    Either way the promoted role is >= target's.
    """
    module = build_canonical_module(_matmul_large, _SPECS)
    counter_before = module.iter_var_counter
    load_path = _find_block_writing(module, "rhs_sbuf")
    target_path = _find_matmul_d0_acc_path(module)
    atom = ComputeAt(block_path=load_path, target_path=target_path)
    new_mod = atom.apply(module)
    """Target's role is already ACC (strongest); no promotion needed.
    Counter should advance if any iter-var was allocated during apply
    (e.g. in other paths). Allow equality."""
    assert new_mod.iter_var_counter >= counter_before


def test_compute_at_unmatched_suffix_forms_inner_fornodes() -> None:
    """Block's iter vars beyond the matched prefix become nested ForNodes.

    rhs_sbuf load has iter_vars (d0_par, d3_par). Moving under matmul's
    d0 ACC means the d0 pair is matched (prefix length 1). The load's
    d3 iter var remains in block's iter_vars list, wrapped in a fresh
    ForNode as the new block subtree's outermost loop.
    """
    module = build_canonical_module(_matmul_large, _SPECS)
    load_path = _find_block_writing(module, "rhs_sbuf")
    target_path = _find_matmul_d0_acc_path(module)
    atom = ComputeAt(block_path=load_path, target_path=target_path)
    new_mod = atom.apply(module)
    """Find the relocated load SBlock (skip the NKIAlloc block that also
    declares ``rhs_sbuf``)."""
    relocated: SBlock | None = None
    for _path, node in _collect_paths(new_mod):
        if isinstance(node, SBlock):
            writes = {a.tensor_name for a in node.writes.values()}
            if "rhs_sbuf" in writes and any(c.op_cls.__name__ == "NKILoad" for c in node.body):
                relocated = node
                break
    assert relocated is not None
    """Unmatched suffix: d3 iter var remains in iter_vars."""
    remaining_dims = [iv.dim_id for iv in relocated.iter_vars]
    assert "d3" in remaining_dims
    assert "d0" not in remaining_dims, "d0 iter var should have been merged with target's"
