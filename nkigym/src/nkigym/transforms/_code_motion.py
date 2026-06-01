"""Shared structural move mechanics for ComputeAt / ReverseComputeAt.

A move relocates one block under a target loop: same-axis loops the
target's enclosing nest fully covers collapse (their iter_var binds to the
target's loop var instead); uncovered loops stay as the moved block's
private inner nest; the block is spliced under the target at ``index``.
Direction (sink producer vs lift consumer) only affects the caller's
legality check and which neighbor bounds the insertion gap — the
structural move is identical, so both share this function.
"""

from __future__ import annotations

from nkigym.ir import KernelIR
from nkigym.ir.expr import Expr, Var, substitute
from nkigym.ir.tree import BlockNode, BufferRegion, ForNode, ISANode, KernelTree
from nkigym.transforms._tree_ops import _block_local_descendants, _replace_in_parent_children


def _compute_at_impl(ir: KernelIR, block_nid: int, target_loop_nid: int, index: int, is_reverse: bool) -> None:
    """Move ``block_nid`` under ``target_loop_nid`` in place (full-coverage only).

    Mutates ``ir.tree``. Caller has already checked legality and deep-copied.
    ``index`` is the insertion position among the target loop's body children
    (TVM convention: -1 last legal, -2 earliest legal, >=0 explicit).
    ``is_reverse`` is accepted for symmetry but does not change the steps.

    Args:
        ir: kernel IR whose tree is mutated in place.
        block_nid: the BlockNode to relocate.
        target_loop_nid: the ForNode under which ``block_nid`` is spliced.
        index: insertion position among the target loop's body children.
        is_reverse: caller-direction flag; structurally inert here.
    """
    tree = ir.tree
    block = tree.data(block_nid)
    assert isinstance(block, BlockNode), f"_compute_at_impl: block_nid={block_nid} is not a BlockNode"
    target_enclosing = _enclosing_loop_vars_by_axis(tree, target_loop_nid)
    moved_loops = _block_loops_by_axis(tree, block_nid)
    covered_var_to_target_var: dict[str, str] = {}
    for axis, (_moved_loop_nid, moved_var) in moved_loops.items():
        if axis in target_enclosing:
            covered_var_to_target_var[moved_var] = target_enclosing[axis]
    _collapse_covered_loops(tree, block_nid, covered_var_to_target_var)
    _rebind_block(tree, block_nid, covered_var_to_target_var)
    _splice_under_target(tree, block_nid, target_loop_nid, index)


def _enclosing_loop_vars_by_axis(tree: KernelTree, target_loop_nid: int) -> dict[str, str]:
    """Map concrete axis -> loop_var for every ForNode at/above ``target_loop_nid`` within its block.

    Args:
        tree: the kernel tree.
        target_loop_nid: the destination ForNode.

    Returns:
        A dict from concrete axis name to the binding loop_var.
    """
    block_nid = _enclosing_block(tree, target_loop_nid)
    block = tree.data(block_nid)
    assert isinstance(block, BlockNode), f"enclosing block of {target_loop_nid} is not a BlockNode"
    var_to_axis = _loop_var_to_axis(block)
    out: dict[str, str] = {}
    for nid in [target_loop_nid, *tree.ancestors(target_loop_nid)]:
        data = tree.data(nid)
        if isinstance(data, ForNode) and data.loop_var in var_to_axis:
            out[var_to_axis[data.loop_var]] = data.loop_var
    return out


def _block_loops_by_axis(tree: KernelTree, block_nid: int) -> dict[str, tuple[int, str]]:
    """Map concrete axis -> (loop_nid, loop_var) for each ForNode ``block_nid`` owns.

    Args:
        tree: the kernel tree.
        block_nid: the moved block.

    Returns:
        A dict from concrete axis name to ``(ForNode nid, loop_var)``.
    """
    block = tree.data(block_nid)
    assert isinstance(block, BlockNode), f"block_nid={block_nid} is not a BlockNode"
    var_to_axis = _loop_var_to_axis(block)
    out: dict[str, tuple[int, str]] = {}
    for nid in _block_local_descendants(tree, block_nid):
        data = tree.data(nid)
        if isinstance(data, ForNode) and data.loop_var in var_to_axis:
            out[var_to_axis[data.loop_var]] = (nid, data.loop_var)
    return out


def _loop_var_to_axis(block: BlockNode) -> dict[str, str]:
    """Invert ``iter_values``: bare-Var loop_var name -> iter_var axis.

    Only bare ``Var`` bindings invert cleanly; affine combinations (post-Split)
    are skipped, so a fully-Split axis is treated as uncovered.

    Args:
        block: the block whose iter_var bindings are inverted.

    Returns:
        A dict from loop_var name to concrete axis name.
    """
    out: dict[str, str] = {}
    for iv, value in zip(block.iter_vars, block.iter_values):
        if isinstance(value, Var):
            out[value.name] = iv.axis
    return out


def _collapse_covered_loops(tree: KernelTree, block_nid: int, covered_vars: dict[str, str]) -> None:
    """Remove the moved block's ForNodes whose loop_var is covered.

    Each removed loop's children are re-linked to its parent (preserving
    sibling order), then the loop node is dropped from the graph.

    Args:
        tree: the kernel tree.
        block_nid: the moved block.
        covered_vars: loop_var names whose loops collapse.
    """
    to_remove = [nid for nid in _block_local_descendants(tree, block_nid) if _is_covered_for(tree, nid, covered_vars)]
    for nid in to_remove:
        parent = tree.parent(nid)
        assert parent is not None, f"covered loop {nid} has no parent"
        children = tree.children(nid)
        _replace_in_parent_children(tree, parent, [nid], children)
        tree.graph.remove_node(nid)


def _is_covered_for(tree: KernelTree, nid: int, covered_vars: dict[str, str]) -> bool:
    """Return True iff ``nid`` is a ForNode whose loop_var is in ``covered_vars``.

    Args:
        tree: the kernel tree.
        nid: candidate node id.
        covered_vars: covered loop_var names.

    Returns:
        Whether ``nid`` is a covered ForNode.
    """
    data = tree.data(nid)
    return isinstance(data, ForNode) and data.loop_var in covered_vars


def _rebind_block(tree: KernelTree, block_nid: int, covered_vars: dict[str, str]) -> None:
    """Substitute each covered loop_var with the target's loop_var.

    Rewrites the block's ``iter_values`` / ``reads`` / ``writes`` and every
    block-local ISA leaf's ``operand_bindings``.

    Args:
        tree: the kernel tree.
        block_nid: the moved block.
        covered_vars: map from collapsed loop_var to target loop_var.
    """
    subs: dict[str, Expr] = {old: Var(name=new) for old, new in covered_vars.items()}
    block = tree.data(block_nid)
    assert isinstance(block, BlockNode), f"block_nid={block_nid} is not a BlockNode"
    new_block = BlockNode(
        iter_vars=block.iter_vars,
        iter_values=tuple(substitute(v, subs) for v in block.iter_values),
        reads=tuple(_sub_region(r, subs) for r in block.reads),
        writes=tuple(_sub_region(w, subs) for w in block.writes),
        alloc_buffers=block.alloc_buffers,
        annotations=dict(block.annotations),
        axis_map=block.axis_map,
    )
    tree.graph.nodes[block_nid]["data"] = new_block
    for nid in _block_local_descendants(tree, block_nid):
        data = tree.data(nid)
        if isinstance(data, ISANode):
            new_bindings = {slot: _sub_region(r, subs) for slot, r in data.operand_bindings.items()}
            tree.graph.nodes[nid]["data"] = ISANode(
                op_cls=data.op_cls, operand_bindings=new_bindings, kwargs=dict(data.kwargs)
            )


def _sub_region(region: BufferRegion, subs: dict[str, Expr]) -> BufferRegion:
    """Apply ``Var`` substitutions to both lo and width of every range.

    Args:
        region: the region to rewrite.
        subs: substitution map applied to each range endpoint.

    Returns:
        A new ``BufferRegion`` with substituted ranges.
    """
    return BufferRegion(
        tensor=region.tensor,
        ranges=tuple((substitute(lo, subs), substitute(width, subs)) for lo, width in region.ranges),
    )


def _splice_under_target(tree: KernelTree, block_nid: int, target_loop_nid: int, index: int) -> None:
    """Detach the moved block and insert it among the target loop's body children.

    ``index`` follows the TVM convention: ``-1`` appends last, ``-2`` prepends
    first, ``>=0`` is an explicit position.

    Args:
        tree: the kernel tree.
        block_nid: the moved block.
        target_loop_nid: the destination ForNode.
        index: insertion position among the target loop's children.
    """
    old_parent = tree.parent(block_nid)
    assert old_parent is not None, f"moved block {block_nid} has no parent to detach from"
    _replace_in_parent_children(tree, old_parent, [block_nid], [])
    target_children = tree.children(target_loop_nid)
    if index == -1:
        pos = len(target_children)
    elif index == -2:
        pos = 0
    else:
        pos = index
    new_order = target_children[:pos] + [block_nid] + target_children[pos:]
    for child in target_children:
        tree.graph.remove_edge(target_loop_nid, child)
    for child in new_order:
        tree.graph.add_edge(target_loop_nid, child)


def _enclosing_block(tree: KernelTree, nid: int) -> int:
    """Walk ancestors of ``nid`` until a BlockNode; return its nid.

    Args:
        tree: the kernel tree.
        nid: the starting node id.

    Returns:
        The nid of the nearest BlockNode ancestor.
    """
    result: int | None = None
    for anc in reversed(tree.ancestors(nid)):
        if isinstance(tree.data(anc), BlockNode):
            result = anc
            break
    if result is None:
        raise ValueError(f"no enclosing BlockNode for {nid}")
    return result


__all__ = ["_compute_at_impl"]
