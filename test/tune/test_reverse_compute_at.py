"""Unit tests for the ReverseComputeAt atom on the iter-var IR.

Dual of :class:`ComputeAt`: prefix-match + role-promotion semantics
identical; only the dataflow direction flips (target's subtree contains
a producer of one of block's reads, not a consumer of block's writes).
"""

import ast

import pytest

from nkigym.codegen.canonical import build_canonical_module
from nkigym.codegen.ir import ForNode, SBlock, validate_dataflow_ordering
from nkigym.codegen.render import render
from nkigym.ops import nkigym_kernel
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.memset import NKIMemset
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.tune import AtomLegalityError
from nkigym.tune.reverse_compute_at import ReverseComputeAt, enumerate_reverse_compute_at_atoms


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


def _find_block_with_op(module, op_name: str) -> tuple[int, ...]:
    """Return the path to the first SBlock whose body contains ``op_name``."""
    for path, node in _collect_paths(module):
        if isinstance(node, SBlock) and any(c.op_cls.__name__ == op_name for c in node.body):
            return path
    raise AssertionError(f"No SBlock containing {op_name!r}")


def _find_block_writing(module, tensor_name: str) -> tuple[int, ...]:
    """Return the path to the first non-alloc SBlock whose writes contain ``tensor_name``."""
    for path, node in _collect_paths(module):
        if isinstance(node, SBlock):
            written = {a.tensor_name for a in node.writes.values()} | {
                a.tensor_name for a in node.reads_writes.values()
            }
            if tensor_name in written and any(c.op_cls.__name__ != "NKIAlloc" for c in node.body):
                return path
    raise AssertionError(f"No SBlock writing {tensor_name!r}")


def _has_op_in_subtree(node, op_name: str) -> bool:
    """True iff any SBlock under ``node`` contains ``op_name``."""
    result: bool
    if isinstance(node, SBlock):
        result = any(c.op_cls.__name__ == op_name for c in node.body)
    else:
        result = any(_has_op_in_subtree(c, op_name) for c in node.children)
    return result


def _find_tensorcopy_root_fornode_path(module) -> tuple[int, ...]:
    """Path to the root ForNode whose subtree contains NKITensorCopy (root[9])."""
    for path, node in _collect_paths(module):
        if len(path) == 1 and isinstance(node, ForNode) and _has_op_in_subtree(node, "NKITensorCopy"):
            return path
    raise AssertionError("No root ForNode containing NKITensorCopy")


def test_reverse_compute_at_store_under_tensor_copy_loop_is_legal() -> None:
    """The store (reads ``sbuf_prod``) can be placed under tensor_copy's outer loop.

    tensor_copy writes ``sbuf_prod`` → target's subtree produces the block's
    read. Both SBlocks have iter_vars [d1-PAR, d3-PAR], and tensor_copy's
    outer ForNode is d1-PAR; prefix-match holds at length 1.
    """
    module = build_canonical_module(_matmul_large, _SPECS)
    store_path = _find_block_with_op(module, "NKIStore")
    tc_root_path = _find_tensorcopy_root_fornode_path(module)
    atom = ReverseComputeAt(block_path=store_path, target_path=tc_root_path)
    assert atom.is_legal(module)


def test_reverse_compute_at_rejects_target_that_is_ancestor() -> None:
    """Target that IS an ancestor of the block's current position is rejected."""
    module = build_canonical_module(_matmul_large, _SPECS)
    store_path = _find_block_with_op(module, "NKIStore")
    """Parent path (last ForNode ancestor of store block)."""
    parent_path = store_path[:-1]
    atom = ReverseComputeAt(block_path=store_path, target_path=parent_path)
    assert not atom.is_legal(module)


def test_reverse_compute_at_rejects_target_without_producer() -> None:
    """Target whose subtree does not produce the block's reads is rejected.

    Store reads ``sbuf_prod``; the load root subtree writes ``lhs_T_sbuf``
    / ``rhs_sbuf`` but never ``sbuf_prod`` → illegal.
    """
    module = build_canonical_module(_matmul_large, _SPECS)
    store_path = _find_block_with_op(module, "NKIStore")
    """Find a root ForNode containing only a load subtree (no tensor_copy/matmul)."""
    load_root_path: tuple[int, ...] | None = None
    for path, node in _collect_paths(module):
        if len(path) == 1 and isinstance(node, ForNode) and _has_op_in_subtree(node, "NKILoad"):
            if not _has_op_in_subtree(node, "NKITensorCopy") and not _has_op_in_subtree(node, "NKIMatmul"):
                load_root_path = path
                break
    assert load_root_path is not None
    atom = ReverseComputeAt(block_path=store_path, target_path=load_root_path)
    assert not atom.is_legal(module)


def test_reverse_compute_at_apply_moves_block_under_target() -> None:
    """After apply, the moved block lives in target's subtree + module validates."""
    module = build_canonical_module(_matmul_large, _SPECS)
    store_path = _find_block_with_op(module, "NKIStore")
    tc_root_path = _find_tensorcopy_root_fornode_path(module)
    atom = ReverseComputeAt(block_path=store_path, target_path=tc_root_path)
    new_mod = atom.apply(module)
    assert validate_dataflow_ordering(new_mod)

    """Walk the new tree: the store should appear under a ForNode whose
    axis_id is d1 (target's ancestor chain's axis)."""
    d1_new = new_mod.axis_id_by_name("d1")

    def has_store_under_d1(node, under_d1: bool) -> bool:
        result: bool
        if isinstance(node, SBlock):
            result = under_d1 and any(c.op_cls.__name__ == "NKIStore" for c in node.body)
        else:
            is_d1 = node.iter_var.axis_id == d1_new
            next_under = under_d1 or is_d1
            result = any(has_store_under_d1(c, next_under) for c in node.children)
        return result

    found = any(has_store_under_d1(root, False) for root in new_mod.body)
    assert found


def test_reverse_compute_at_apply_raises_on_illegal() -> None:
    """``apply`` re-validates and raises ``AtomLegalityError``."""
    module = build_canonical_module(_matmul_large, _SPECS)
    store_path = _find_block_with_op(module, "NKIStore")
    parent_path = store_path[:-1]
    atom = ReverseComputeAt(block_path=store_path, target_path=parent_path)
    with pytest.raises(AtomLegalityError):
        atom.apply(module)


def test_reverse_compute_at_render_valid_python() -> None:
    """After apply + render, the output parses as valid Python."""
    module = build_canonical_module(_matmul_large, _SPECS)
    store_path = _find_block_with_op(module, "NKIStore")
    tc_root_path = _find_tensorcopy_root_fornode_path(module)
    atom = ReverseComputeAt(block_path=store_path, target_path=tc_root_path)
    new_mod = atom.apply(module)
    source = render(new_mod)
    ast.parse(source)


def test_enumerate_reverse_compute_at_atoms_yields_legal_only() -> None:
    """Every atom emitted by the enumerator passes ``is_legal``."""
    module = build_canonical_module(_matmul_large, _SPECS)
    atoms = enumerate_reverse_compute_at_atoms(module)
    for atom in atoms:
        assert atom.is_legal(module)
    assert len(atoms) > 0


def test_reverse_compute_at_unmatched_suffix_forms_inner_fornodes() -> None:
    """Block's iter vars beyond the matched prefix become nested ForNodes.

    Store has iter_vars ``[d1_outer, d1_inner, d3_outer, d3_inner]``
    (Task 4: each dim yields outer+inner). Target (tensor_copy's root
    d1 ForNode) has a single d1 ForNode ancestor — prefix match is
    length 1, matching only the block's d1 outer. The remaining suffix
    ``[d1_inner, d3_outer, d3_inner]`` stays in the relocated block's
    ``iter_vars`` and becomes nested ForNodes.
    """
    module = build_canonical_module(_matmul_large, _SPECS)
    store_path = _find_block_with_op(module, "NKIStore")
    tc_root_path = _find_tensorcopy_root_fornode_path(module)
    atom = ReverseComputeAt(block_path=store_path, target_path=tc_root_path)
    new_mod = atom.apply(module)
    """Find the relocated store SBlock."""
    relocated: SBlock | None = None
    for _path, node in _collect_paths(new_mod):
        if isinstance(node, SBlock) and any(c.op_cls.__name__ == "NKIStore" for c in node.body):
            relocated = node
            break
    assert relocated is not None
    """Unmatched suffix: d3 outer+inner remain, plus d1 inner (target's
    ancestor chain only covers d1 outer)."""
    remaining_names = [new_mod.axes[iv.axis_id].name for iv in relocated.iter_vars]
    assert "d3" in remaining_names
    assert (
        remaining_names.count("d1") == 1
    ), f"expected d1 outer merged, inner remaining (1 d1 left); got {remaining_names}"


def test_reverse_compute_at_role_promotion_on_matmul_under_load() -> None:
    """Matmul block under load's d0 PAR ForNode promotes target to ACC.

    Matmul reads ``lhs_T_sbuf``; lhs_T load writes it. Matmul's iter_vars
    begin with d0-ACC; load's outer ForNode is d0-PAR. Matched pair
    (target PAR, block ACC): block role is stronger → target promotes to
    ACC. A fresh iter var is allocated.
    """
    module = build_canonical_module(_matmul_large, _SPECS)
    d0 = module.axis_id_by_name("d0")
    counter_before = module.iter_var_counter
    mm_path = _find_block_with_op(module, "NKIMatmul")
    """Find lhs_T load's outer d0-PAR ForNode (root[5], depth-1)."""
    load_root_path: tuple[int, ...] | None = None
    for path, node in _collect_paths(module):
        if len(path) == 1 and isinstance(node, ForNode) and node.iter_var.axis_id == d0:
            if _has_op_in_subtree(node, "NKILoad"):
                """Pick the one writing lhs_T_sbuf."""
                for _sp, sn in _collect_paths(module):
                    if (
                        isinstance(sn, SBlock)
                        and any(c.op_cls.__name__ == "NKILoad" for c in sn.body)
                        and "lhs_T_sbuf" in {a.tensor_name for a in sn.writes.values()}
                    ):
                        if _sp[0] == path[0]:
                            load_root_path = path
                            break
                if load_root_path is not None:
                    break
    assert load_root_path is not None
    atom = ReverseComputeAt(block_path=mm_path, target_path=load_root_path)
    assert atom.is_legal(module)
    new_mod = atom.apply(module)
    """Counter advanced — a fresh iter var was allocated for the promoted target.

    Dataflow validation is NOT asserted here: placing matmul under the
    lhs_T load subtree makes matmul precede the rhs load and the
    psum memset in DFS order, so the result is not a valid schedule.
    This test only verifies the role-promotion mechanics; a legal
    compose with full producer chain is tested elsewhere."""
    assert new_mod.iter_var_counter > counter_before
