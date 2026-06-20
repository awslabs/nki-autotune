"""Shared structural move for ComputeAt / ReverseComputeAt.

A move relocates one block under a target loop via region-regen: solve each
moved-block dim into target-covered + residual (``_domain_solve``),
regenerate residual loops + rebind, splice under the target at ``index``,
then ``normalize_block`` reconciles names / trip-1 / region offsets on both
the moved block (its regenerated residual nest) and the fork block.
Direction (``is_reverse``) does not change the structural steps; the
caller's legality check differs.
"""

from __future__ import annotations

from nkigym.ir import KernelIR
from nkigym.ir.tree import BlockNode, ForNode, ISANode, KernelTree, role_of
from nkigym.ops.base import AxisRole
from nkigym.transforms._domain_solve import (
    DimDomain,
    _enclosing_block,
    dim_loops_of_block,
    enclosing_dim_loops,
    regen_and_rebind,
    solve_iter_domains,
)
from nkigym.transforms._normalize import normalize_block
from nkigym.transforms._tree_ops import _replace_in_parent_children
from nkigym.transforms.base import TransformLegalityError


def _move(ir: KernelIR, block_nid: int, target_loop_nid: int, index: int, is_reverse: bool) -> None:
    """Relocate ``block_nid`` under ``target_loop_nid`` in place (region-regen).

    Caller has checked legality and deep-copied. ``index`` follows TVM
    convention: ``-1`` append, ``-2`` prepend, ``>=0`` explicit slot among
    the target loop's children. ``is_reverse`` is structurally inert.

    ``normalize_block`` runs twice: once on ``block_nid`` to rebuild the
    moved block's iter_values + region ``lo`` offsets from its regenerated
    residual loops (``regen_and_rebind`` leaves these as a skeleton), and
    once on the fork block to reconcile names / trip-1 there. The two scopes
    are disjoint (``_block_local_descendants`` does not cross BlockNode
    boundaries), so the order between them is immaterial.
    """
    tree = ir.tree
    moved = dim_loops_of_block(tree, block_nid)
    target = enclosing_dim_loops(tree, target_loop_nid)
    solved = solve_iter_domains(moved, target)
    regen_and_rebind(tree, block_nid, solved)
    _splice_under_target(tree, block_nid, target_loop_nid, index)
    normalize_block(tree, block_nid)
    fork = _enclosing_block(tree, target_loop_nid)
    normalize_block(tree, fork)
    _assert_single_parent(tree)


def _assert_single_parent(tree: KernelTree) -> None:
    """Raise loudly if any node has more than one parent after a move.

    The tree is a forest: every node except the root has exactly one parent.
    A splice that detaches a node from its old parent without removing the old
    edge leaves the node double-parented (a real corruption, not a legal-move
    distinction); failing here keeps that bug loud instead of surfacing as a
    downstream ``Dependency``/render crash on a malformed tree.
    """
    multi = [n for n in tree.graph.nodes if len(list(tree.graph.predecessors(n))) > 1]
    if multi:
        detail = {n: list(tree.graph.predecessors(n)) for n in multi}
        raise ValueError(f"_move left nodes with multiple parents: {detail}")


def _check_move_realizable(ir: KernelIR, block_nid: int, target_loop_nid: int) -> dict[str, DimDomain]:
    """Raise TransformLegalityError if the target's coverage cannot cleanly
    re-domain the moved block — a pure, read-only check (no tree mutation).

    This is the realizability prefix of ``_move``: ``solve_iter_domains`` over
    the moved block's ``dim_loops_of_block`` and the target's
    ``enclosing_dim_loops`` raises ``DomainSolveError`` when a target dim's
    coverage does not divide the moved dim's extent (partial coverage that no
    residual loop can express). ``analyze`` relies on this rejection to filter
    such candidates; surfacing it as ``TransformLegalityError`` keeps that
    contract without the deep-copy ``_move`` simulation. Other structural
    invariants (single body leaf, single parent after splice) are guaranteed by
    construction for a structurally-valid candidate and are asserted inside the
    real ``_move`` when ``apply`` runs.
    """
    moved = dim_loops_of_block(ir.tree, block_nid)
    target = enclosing_dim_loops(ir.tree, target_loop_nid)
    try:
        solved = solve_iter_domains(moved, target)
    except (ValueError, KeyError) as e:
        raise TransformLegalityError(
            f"move(block={block_nid} under loop={target_loop_nid}) is not structurally realizable: {e}"
        ) from e
    _check_no_reduction_axis_covered(ir, block_nid, target_loop_nid, solved)
    _check_no_reduction_replicated(ir, block_nid, target_loop_nid, moved)
    return solved


def _check_no_reduction_replicated(ir: KernelIR, block_nid: int, target_loop_nid: int, moved: dict) -> None:
    """Reject sinking a reduction block under a target loop iterating a dim the
    block does NOT tile (would replicate the accumulation, not re-init it).

    A block with an ACCUMULATION axis accumulates into a carried buffer
    (matmul → ``psum_prod``) whose init (memset) sits outside the block. If the
    target loop iterates a dim absent from the moved block's ``dim_loops`` — i.e.
    a dim the block writes at FULL extent (no per-tile index) — the block is
    blindly replicated across that loop's iterations, each re-running the whole
    K accumulation into the SAME PSUM region without an intervening re-init →
    the result is summed ``trip`` times (sim: partial/garbled output, not NaN).
    A PARALLEL producer (load/store/tensor_copy) replicated over such a loop is a
    benign recompute; only an ACCUMULATION block corrupts. No legal ladder move
    of a reduction block iterates a dim it lacks (verified), so this never
    over-rejects.
    """
    block = ir.tree.data(block_nid)
    assert isinstance(block, BlockNode)
    if not any(iv.role == AxisRole.ACCUMULATION for iv in block.iter_vars):
        return
    target = enclosing_dim_loops(ir.tree, target_loop_nid)
    replicated = sorted(set(target) - set(moved))
    if replicated:
        raise TransformLegalityError(
            f"move(block={block_nid} under loop={target_loop_nid}) replicates a reduction over "
            f"dim(s) {replicated} the block does not tile (writes them at full extent); the "
            f"accumulation would re-run per iteration into an un-reinitialised accumulator"
        )


def _own_carry_loop_nids(ir: KernelIR, block_nid: int) -> set[int]:
    """Loop NIDs that the moved block's accumulator-init DOMINATES.

    The block's RMW operand (its carried accumulator) is initialised by a
    sibling memset whose write CARRYs into the reduction loop — recorded as a
    ``CARRY`` edge ``init_writer -> loop_nid`` in ``ir.dependency``. Covering a
    reduction axis by one of these loops is the SAFE enclosing-reduction case
    (init dominates); covering by any other loop is foreign (init does not
    dominate -> NaN). Returns the loop NIDs into which the block's own
    accumulator's writers carry.
    """
    block = ir.tree.data(block_nid)
    assert isinstance(block, BlockNode)
    leaf = next(d for d in ir.tree.descendants(block_nid) if isinstance(ir.tree.data(d), ISANode))
    isa = ir.tree.data(leaf)
    acc_tensors = {
        isa.operand_bindings[slot].tensor for slot in isa.op_cls.RMW_OPERANDS if slot in isa.operand_bindings
    }
    out: set[int] = set()
    for tensor in acc_tensors:
        for writer in ir.dependency.touches_by_tensor.get(tensor, ()):
            for _w, loop_nid, attrs in ir.dependency.graph.out_edges(writer, data=True):
                if attrs.get("kind") == "CARRY":
                    out.add(loop_nid)
    return out


def _check_no_reduction_axis_covered(ir: KernelIR, block_nid: int, target_loop_nid: int, solved: dict) -> None:
    """Reject a move that covers the moved block's ACCUMULATION (reduction) axis
    with a loop the block's own init does NOT dominate (foreign covering loop).

    A reduction axis (matmul K, two-stage fold ko) must iterate as a contiguous
    nest bracketed by its init (memset) before and drain (tensor_copy) after.
    ``solve_iter_domains`` marks a dim *covered* when the target's enclosing
    loops drive it (``target_loops`` non-empty, residual collapsed). Covering a
    reduction axis by an enclosing loop is SAFE when that loop is one the block's
    own init dominates (a CARRY edge from the block's accumulator-init writer
    into that loop NID exists); covering by any other loop is foreign (init does
    not dominate -> NaN). The block's own-carry loop NIDs are resolved via
    ``_own_carry_loop_nids``; covering NIDs come from the target's enclosing
    ForNodes on the covered dim; if any covering NID is not in own-carry, reject.
    NID-comparison, NOT loop_var-comparison, is load-bearing: after RFactor a
    foreign K-loop and the fold's own ko-loop are BOTH named ``i_d0_0`` (dim
    ``d0``). A var comparison silently admits the foreign case -> NaN.
    """
    block = ir.tree.data(block_nid)
    assert isinstance(block, BlockNode)
    own_carry = _own_carry_loop_nids(ir, block_nid)
    target_nid_by_var = {ir.tree.data(nid).loop_var: nid
                         for nid in (target_loop_nid, *ir.tree.ancestors(target_loop_nid))
                         if isinstance(ir.tree.data(nid), ForNode)}
    result: None = None
    for dim, domain in solved.items():
        if not domain.target_loops:
            continue
        try:
            role = role_of(block, dim)
        except KeyError:
            continue
        if role != AxisRole.ACCUMULATION:
            continue
        covering_nids = {target_nid_by_var[lv] for lv, _e in domain.target_loops if lv in target_nid_by_var}
        if covering_nids and covering_nids <= own_carry:
            continue
        raise TransformLegalityError(
            f"move(block={block_nid} under loop={target_loop_nid}) would cover reduction axis "
            f"{dim!r} (ACCUMULATION) with enclosing loops {domain.target_loops} the block's own "
            f"init does not dominate; a foreign covering loop breaks init-domination"
        )
    return result


def _check_move_preserves_dependencies(
    ir: KernelIR, block_nid: int, target_loop_nid: int, index: int, is_reverse: bool
) -> None:
    """Raise TransformLegalityError if the proposed move would make any
    dependency edge incident to the moved block point backward.

    Pure topological query — no deep copy, no ``_move``. Asks
    ``Dependency.first_backward_edge_for_insertion`` on the **original**
    program's dependency graph: edge *directions* are frozen at construction,
    and the moved leaf's post-splice preorder position is computed analytically
    from ``(target_loop_nid, index)``. One span-based, edge-kind-agnostic rule
    covers reduction-init domination and consumer-before-producer ordering
    alike. ``is_reverse`` does not change the check — both faces forbid the same
    backward edges; only their structural splice differs.

    Directions MUST come from ``ir.dependency`` (the pre-move graph). Rebuilding
    ``Dependency`` on a moved tree would be wrong: ``_build`` re-derives every
    flow edge from execution order, so a PARALLEL producer sunk past its
    consumer flips from RAW ``producer->consumer`` to WAR ``consumer->producer``
    and the violation disappears (matmul reads uninitialised data -> NaN).
    Freezing directions keeps the RAW orientation, so the post-splice backward
    span is detected.

    Realizability (target coverage divides the moved extent) is a separate
    concern checked by ``_check_move_realizable`` before this; here we assume a
    realizable candidate and only test ordering.
    """
    solved = _check_move_realizable(ir, block_nid, target_loop_nid)
    target_nid_by_var = {
        ir.tree.data(nid).loop_var: nid
        for nid in (target_loop_nid, *ir.tree.ancestors(target_loop_nid))
        if isinstance(ir.tree.data(nid), ForNode)
    }
    skip_cover_loops = frozenset(
        target_nid_by_var[lv] for dom in solved.values() for lv, _ext in dom.target_loops if lv in target_nid_by_var
    )
    moved_leaf = ir.dependency._resolve(block_nid)
    offending = ir.dependency.first_backward_edge_for_insertion(
        moved_leaf, target_loop_nid, index, skip_cover_loops=skip_cover_loops
    )
    result: None = None
    if offending is not None:
        a, b = offending
        raise TransformLegalityError(
            f"move(block={block_nid} under loop={target_loop_nid}) reorders dependency "
            f"edge {a}->{b} backward (a carried buffer's init/drain cannot enter its "
            f"reduction loop, nor a consumer precede its producer)"
        )
    return result


def _splice_under_target(tree: KernelTree, block_nid: int, target_loop_nid: int, index: int) -> None:
    """Detach ``block_nid`` from its parent and insert under the target loop at ``index``."""
    old_parent = tree.parent(block_nid)
    assert old_parent is not None, f"moved block {block_nid} has no parent"
    _replace_in_parent_children(tree, old_parent, [block_nid], [])
    children = tree.children(target_loop_nid)
    if index == -1:
        pos = len(children)
    elif index == -2:
        pos = 0
    elif index >= 0:
        pos = index
    else:
        raise ValueError(f"_splice_under_target: unsupported index {index} (use -1 append, -2 prepend, or >=0)")
    new_order = children[:pos] + [block_nid] + children[pos:]
    for child in children:
        tree.graph.remove_edge(target_loop_nid, child)
    for child in new_order:
        tree.graph.add_edge(target_loop_nid, child)


__all__ = ["_move", "_check_move_preserves_dependencies"]
