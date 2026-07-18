"""Shared structural move for CodeMotion (the merged compute-at / reverse-compute-at).

A move relocates one block under a target loop by a **verbatim residual splice**,
NOT region regeneration. Precondition (``_check_same_loop_prefix``): the target's
enclosing loop nest (outermost down to ``target_loop_nid``), restricted to dimensions
the moved block binds, must be an exact ordered ``(dimension, extent)`` prefix of the
moved block's bound loop sequence. The move substitutes the matched target loop
variables for loops supplied by a different target scope, removes the covered local
loops, and re-parents the untouched residual loops plus the leaf under the target.

This intentionally avoids TVM-style per-dim domain solving: a partial split of a
shared dim, or a different loop order, is REJECTED loudly (``Split``/``Reorder``
first), so every legal move is a clean structural merge.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace

from nkigym.ir import KernelIR
from nkigym.ir.arith.expr import Expr, Var, substitute, to_affine
from nkigym.ir.dependency import Dependency, _access_invariant_across, _leaf_operand_regions, _tensor_carried_across
from nkigym.ir.tree import BlockNode, BufferRegion, ForNode, ISANode, KernelTree
from nkigym.ops.base import AxisRole
from nkigym.transforms._domain_solve import _dim_from_loopvar
from nkigym.transforms._normalize import normalize_block
from nkigym.transforms._tree_ops import _block_local_descendants, _replace_in_parent_children
from nkigym.transforms.base import Transform, TransformLegalityError, TransformOption


@dataclass(frozen=True)
class _PrefixPlan:
    """Exact-prefix loop correspondence for one CodeMotion move."""

    target_loop_nids: tuple[int, ...]
    local_loop_nids: tuple[int, ...]
    matched_loop_nids: tuple[tuple[int, int], ...]
    matched_local_nids: tuple[int, ...]
    duplicated_target_nids: tuple[int, ...]


def _move(ir: KernelIR, block_nid: int, target_loop_nid: int, index: int) -> None:
    """Relocate ``block_nid`` under ``target_loop_nid`` by a verbatim residual splice.

    Caller has checked legality (``_check_same_loop_prefix``, so the target's
    dependent nest is an exact dimension/extent prefix) and deep-copied.
    ``index`` follows TVM convention: ``-1`` append, ``-2`` prepend, ``>=0`` slot.

    Matched local loops may use different identifiers from their corresponding
    target loops. Their uses are explicitly rebound before the loops are removed.
    Residual loops receive temporary collision-free names before the splice, then
    ``normalize_block`` restores dense names and bindings in the moved block's own
    scope. The target block's unchanged loop nest is not normalized.
    """
    tree = ir.tree
    plan = _prefix_plan(tree, block_nid, target_loop_nid)
    _prepare_block_for_splice(tree, block_nid, plan)
    _strip_local_prefix_loops(tree, block_nid, len(plan.matched_local_nids))
    _splice_under_target(tree, block_nid, target_loop_nid, index)
    normalize_block(tree, block_nid)
    _assert_single_parent(tree)


def _prepare_block_for_splice(tree: KernelTree, block_nid: int, plan: _PrefixPlan) -> None:
    """Rebind matched loops and temporarily rename every residual local loop."""
    substitutions: dict[str, Expr] = {
        tree.loop(local_nid).loop_var: Var(name=tree.loop(target_nid).loop_var)
        for local_nid, target_nid in plan.matched_loop_nids
    }
    used_names = {node.loop_var for nid in tree.preorder() if isinstance((node := tree.data(nid)), ForNode)}
    matched_local = set(plan.matched_local_nids)
    for local_nid in plan.local_loop_nids:
        if local_nid in matched_local:
            continue
        loop = tree.loop(local_nid)
        dim = _dim_from_loopvar(loop.loop_var)
        temporary = f"i_{dim}__move_{local_nid}"
        while temporary in used_names:
            temporary = f"{temporary}_"
        used_names.add(temporary)
        substitutions[loop.loop_var] = Var(name=temporary)
        tree.graph.nodes[local_nid]["data"] = ForNode(loop_var=temporary, extent=loop.extent)
    _substitute_block_loop_vars(tree, block_nid, substitutions)


def _substitute_block_loop_vars(tree: KernelTree, block_nid: int, substitutions: dict[str, Expr]) -> None:
    """Substitute loop identifiers throughout one leaf block's binding scope."""
    block = tree.block(block_nid)
    new_block = replace(
        block,
        iter_values=tuple(substitute(value, substitutions) for value in block.iter_values),
        reads=tuple(_substitute_region(region, substitutions) for region in block.reads),
        writes=tuple(_substitute_region(region, substitutions) for region in block.writes),
    )
    tree.graph.nodes[block_nid]["data"] = new_block
    for nid in _block_local_descendants(tree, block_nid):
        node = tree.data(nid)
        if not isinstance(node, ISANode):
            continue
        bindings = {slot: _substitute_region(region, substitutions) for slot, region in node.operand_bindings.items()}
        tree.graph.nodes[nid]["data"] = replace(node, operand_bindings=bindings)


def _substitute_region(region: BufferRegion, substitutions: dict[str, Expr]) -> BufferRegion:
    """Apply loop-variable substitutions to one buffer region."""
    ranges = tuple(
        (substitute(lower, substitutions), substitute(width, substitutions)) for lower, width in region.ranges
    )
    return replace(region, ranges=ranges)


def _strip_local_prefix_loops(tree: KernelTree, block_nid: int, count: int) -> None:
    """Remove the outermost ``count`` block-local ForNodes of ``block_nid``.

    The block's body is a single chain ``block -> For -> ... -> For -> leaf``; the
    outermost ``count`` of those ForNodes are the prefix the target already provides
    after :func:`_prepare_block_for_splice` has rebound their uses to the target loop
    variables. Each is spliced out by reconnecting its sole child to its parent,
    leaving the residual loops + leaf attached to the block.
    ``count == 0`` is a no-op (the whole moved nest is residual).
    """
    for _ in range(count):
        child = tree.children(block_nid)
        assert len(child) == 1, f"block {block_nid} body is not a single loop chain: children {child}"
        loop = child[0]
        assert isinstance(tree.data(loop), ForNode), f"expected ForNode to strip; got {type(tree.data(loop)).__name__}"
        grandchildren = tree.children(loop)
        tree.graph.remove_node(loop)
        for gc in grandchildren:
            tree.graph.add_edge(block_nid, gc)


def _target_loop_seq(tree: KernelTree, target_loop_nid: int) -> list[tuple[str, int]]:
    """Ordered ``(loop_var, extent)`` of every ForNode from outermost down to the target.

    The target's enclosing nest in TRUE tree order (outermost→innermost, all dims
    interleaved), built from the ancestor chain — not the per-dim-grouped map, which
    loses interleave order. This is the prefix the moved block must match.
    """
    chain = [*tree.ancestors(target_loop_nid), target_loop_nid]
    return [(node.loop_var, node.extent) for n in chain if isinstance((node := tree.data(n)), ForNode)]


def _moved_loop_seq(tree: KernelTree, block_nid: int) -> list[tuple[str, int]]:
    """Ordered ``(loop_var, extent)`` of the moved block's full loop nest.

    Outermost→innermost: the ForNodes enclosing the block (above it) followed by the
    block-local loop chain (inside it, down to the single ISA leaf). True tree order.
    """
    leaf = next(d for d in tree.preorder(block_nid) if isinstance(tree.data(d), ISANode))
    chain = [*tree.ancestors(leaf), leaf]
    return [(node.loop_var, node.extent) for n in chain if isinstance((node := tree.data(n)), ForNode)]


def _target_loop_nids(tree: KernelTree, target_loop_nid: int) -> list[int]:
    """ForNode nids from the outermost target ancestor through the target."""
    chain = [*tree.ancestors(target_loop_nid), target_loop_nid]
    return [nid for nid in chain if isinstance(tree.data(nid), ForNode)]


def _local_loop_nids(tree: KernelTree, block_nid: int) -> list[int]:
    """Block-local ForNode nids in execution order."""
    leaf = next(nid for nid in tree.preorder(block_nid) if isinstance(tree.data(nid), ISANode))
    chain = tree.ancestors(leaf)
    start = chain.index(block_nid) + 1
    return [nid for nid in chain[start:] if isinstance(tree.data(nid), ForNode)]


def _bound_loop_dims(block: BlockNode) -> dict[str, str]:
    """Map each loop variable in the block's iter bindings to its concrete dimension."""
    result: dict[str, str] = {}
    for iter_var, value in zip(block.iter_vars, block.iter_values):
        for name in to_affine(value):
            if name is not None:
                result[name] = iter_var.axis
    return result


def _prefix_plan(tree: KernelTree, block_nid: int, target_loop_nid: int) -> _PrefixPlan:
    """Match target loops to the moved block's bound prefix by dimension and extent."""
    target_nids = _target_loop_nids(tree, target_loop_nid)
    local_nids = _local_loop_nids(tree, block_nid)
    leaf = next(nid for nid in tree.preorder(block_nid) if isinstance(tree.data(nid), ISANode))
    moved_nids = [nid for nid in tree.ancestors(leaf) if isinstance(tree.data(nid), ForNode)]
    enclosing_nids = set(moved_nids) - set(local_nids)
    bound_dims = _bound_loop_dims(tree.block(block_nid))
    dependent_dims = set(bound_dims.values())
    bound_nids = [nid for nid in moved_nids if tree.loop(nid).loop_var in bound_dims]
    matched: list[tuple[int, int]] = []
    duplicated: list[int] = []
    bound_index = 0
    for target_nid in target_nids:
        target_loop = tree.loop(target_nid)
        if target_nid in enclosing_nids and target_loop.loop_var not in bound_dims:
            continue
        target_dim = _dim_from_loopvar(target_loop.loop_var)
        if target_dim not in dependent_dims:
            duplicated.append(target_nid)
            continue
        if bound_index >= len(bound_nids):
            raise TransformLegalityError(
                f"move(block={block_nid} under loop={target_loop_nid}) has no bound "
                f"{target_dim} loop matching target {target_loop.loop_var!r}; the "
                f"distinct target execution scope cannot replace the moved block's scope"
            )
        moved_nid = bound_nids[bound_index]
        moved_loop = tree.loop(moved_nid)
        moved_dim = bound_dims.get(moved_loop.loop_var)
        if (moved_dim, moved_loop.extent) != (target_dim, target_loop.extent):
            raise TransformLegalityError(
                f"move(block={block_nid} under loop={target_loop_nid}) requires an exact "
                f"(dimension, extent) prefix; target={(target_dim, target_loop.extent)} "
                f"does not match moved={(moved_dim, moved_loop.extent)} "
                f"and cannot replace its execution scope "
                f"(Split / Reorder the mismatched loop first)"
            )
        matched.append((moved_nid, target_nid))
        bound_index += 1
    matched_moved = {moved_nid for moved_nid, _target_nid in matched}
    lost_enclosing = [
        tree.loop(nid).loop_var for nid in bound_nids if nid in enclosing_nids and nid not in matched_moved
    ]
    if lost_enclosing:
        raise TransformLegalityError(
            f"move(block={block_nid} under loop={target_loop_nid}) would detach the block "
            f"from enclosing loop(s) {lost_enclosing} that bind its iter_values"
        )
    local_set = set(local_nids)
    matched_local_nids = tuple(moved_nid for moved_nid, _target_nid in matched if moved_nid in local_set)
    if matched_local_nids != tuple(local_nids[: len(matched_local_nids)]):
        raise TransformLegalityError(
            f"move(block={block_nid} under loop={target_loop_nid}) matched local loops "
            f"{matched_local_nids} that are not an outer prefix of {local_nids}"
        )
    return _PrefixPlan(
        target_loop_nids=tuple(target_nids),
        local_loop_nids=tuple(local_nids),
        matched_loop_nids=tuple(matched),
        matched_local_nids=matched_local_nids,
        duplicated_target_nids=tuple(duplicated),
    )


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


def _check_same_loop_prefix(ir: KernelIR, block_nid: int, target_loop_nid: int) -> list[tuple[str, int]]:
    """Raise TransformLegalityError unless the target's enclosing loops, restricted
    to the moved block's dependent dimensions, are an exact ``(dimension, extent)``
    prefix of the moved block's bound loop sequence.

    Pure, read-only. Returns ``target_seq`` (the full target nest) for the dependency
    check. Target loops already enclosing the moved block retain their node identity.
    A new target loop on a dimension absent from the block's bindings is a duplication
    loop: correct for a pure producer, but rejected for an accumulation block.

    A dependent-dim mismatch (different extent / dim / order) rejects loudly
    (``Split`` / ``Reorder`` first). Matching by dimension instead of identifier is
    required because independently normalized blocks can assign different dense
    names to equivalent loops.
    """
    target_seq = _target_loop_seq(ir.tree, target_loop_nid)
    block = ir.tree.data(block_nid)
    assert isinstance(block, BlockNode)
    plan = _prefix_plan(ir.tree, block_nid, target_loop_nid)
    _check_no_reduction_replicated(ir, block_nid, target_loop_nid, plan.duplicated_target_nids)
    return target_seq


def _check_no_reduction_replicated(
    ir: KernelIR, block_nid: int, target_loop_nid: int, duplicated_target_nids: tuple[int, ...]
) -> None:
    """Reject sinking a reduction block under a target loop var the block does NOT
    bind (would replicate the accumulation, not re-init it).

    A block with an ACCUMULATION axis accumulates into a carried buffer
    (matmul → ``psum_prod``) whose init (memset) sits outside the block. The
    ``duplicated_target_nids`` are target loops that are neither existing shared
    ancestors nor matched dependent-prefix loops. Splicing under one repeats the
    whole accumulation into the same region. Parallel producers may be recomputed;
    accumulation blocks may not.
    """
    block = ir.tree.data(block_nid)
    assert isinstance(block, BlockNode)
    if not any(iv.role == AxisRole.ACCUMULATION for iv in block.iter_vars):
        return
    if duplicated_target_nids:
        replicated = [ir.tree.loop(nid).loop_var for nid in duplicated_target_nids]
        raise TransformLegalityError(
            f"move(block={block_nid} under loop={target_loop_nid}) replicates a reduction over "
            f"loop(s) {replicated} the block does not bind; the accumulation would re-run per "
            f"iteration into an un-reinitialised accumulator"
        )


def _crossed_execution_loops(ir: KernelIR, block_nid: int, target_loop_nid: int) -> list[int]:
    """Return loops added to or removed from the moved leaf's execution scope."""
    tree = ir.tree
    leaf = ir.dependency._resolve(block_nid)
    old_loops = [nid for nid in tree.ancestors(leaf) if isinstance(tree.data(nid), ForNode)]
    plan = _prefix_plan(tree, block_nid, target_loop_nid)
    local_prefix_drop = len(plan.matched_local_nids)
    new_loops = [*plan.target_loop_nids, *plan.local_loop_nids[local_prefix_drop:]]

    unmatched_new = list(new_loops)
    crossed: list[int] = []
    for old_nid in old_loops:
        match = next((index for index, new_nid in enumerate(unmatched_new) if new_nid == old_nid), None)
        if match is None:
            crossed.append(old_nid)
        else:
            unmatched_new.pop(match)
    crossed.extend(unmatched_new)
    return crossed


def _plain_written_tensors(node: ISANode) -> set[str]:
    """Return tensors written by output operands that are not read-modify-write."""
    input_slots = getattr(node.op_cls, "INPUT_OPERANDS", frozenset())
    return {
        region.tensor
        for slot, region in node.operand_bindings.items()
        if slot not in input_slots and slot not in node.op_cls.RMW_OPERANDS
    }


def _check_no_rmw_reset_scope_change(ir: KernelIR, block_nid: int, target_loop_nid: int) -> None:
    """Reject changing a plain reset's frequency relative to an invariant RMW.

    A plain write followed by an RMW of the same region is its reset. Moving that
    writer or the RMW across a loop where both accesses are invariant changes a
    per-iteration reset into a loop-carried accumulator, or the reverse, without
    reversing any dependency edge.
    """
    tree = ir.tree
    moved_leaf = ir.dependency._resolve(block_nid)
    moved_node = tree.data(moved_leaf)
    assert isinstance(moved_node, ISANode)
    plain_writes = _plain_written_tensors(moved_node)
    crossed_loops = _crossed_execution_loops(ir, block_nid, target_loop_nid)
    if plain_writes:
        for _producer, consumer, attrs in ir.dependency.graph.out_edges(moved_leaf, data=True):
            tensor = attrs.get("tensor")
            if tensor not in plain_writes or not _leaf_operand_regions(tree, consumer, tensor, rmw_only=True):
                continue
            for loop_nid in crossed_loops:
                loop = tree.data(loop_nid)
                assert isinstance(loop, ForNode)
                if consumer not in tree.descendants(loop_nid):
                    continue
                if not _access_invariant_across(tree, moved_leaf, loop.loop_var, tensor):
                    continue
                if not _access_invariant_across(tree, consumer, loop.loop_var, tensor):
                    continue
                raise TransformLegalityError(
                    f"move(block={block_nid} under loop={target_loop_nid}) changes reset "
                    f"frequency for tensor {tensor!r} across read-modify-write loop "
                    f"{loop_nid} ({loop.loop_var!r})"
                )

    rmw_tensors = {
        region.tensor for slot, region in moved_node.operand_bindings.items() if slot in moved_node.op_cls.RMW_OPERANDS
    }
    for producer, _consumer, attrs in ir.dependency.graph.in_edges(moved_leaf, data=True):
        tensor = attrs.get("tensor")
        producer_node = tree.data(producer)
        assert isinstance(producer_node, ISANode)
        if tensor not in rmw_tensors or tensor not in _plain_written_tensors(producer_node):
            continue
        for loop_nid in crossed_loops:
            loop = tree.data(loop_nid)
            assert isinstance(loop, ForNode)
            if producer not in tree.descendants(loop_nid):
                continue
            if not _access_invariant_across(tree, producer, loop.loop_var, tensor):
                continue
            if not _access_invariant_across(tree, moved_leaf, loop.loop_var, tensor):
                continue
            raise TransformLegalityError(
                f"move(block={block_nid} under loop={target_loop_nid}) changes reset "
                f"frequency for tensor {tensor!r} across read-modify-write loop "
                f"{loop_nid} ({loop.loop_var!r})"
            )


def _check_no_consumer_hoisted_out_of_producer_loop(ir: KernelIR, block_nid: int, target_loop_nid: int) -> None:
    """Reject hoisting a consumer away from a repeated invariant producer."""
    tree = ir.tree
    moved_leaf = ir.dependency._resolve(block_nid)
    old_loops = set(tree.ancestors(moved_leaf))
    crossed_loops = _crossed_execution_loops(ir, block_nid, target_loop_nid)
    for producer, _consumer, attrs in ir.dependency.graph.in_edges(moved_leaf, data=True):
        tensor = attrs.get("tensor")
        if tensor is None:
            continue
        for loop_nid in crossed_loops:
            if loop_nid not in old_loops or producer not in tree.descendants(loop_nid):
                continue
            loop = tree.data(loop_nid)
            assert isinstance(loop, ForNode)
            if _tensor_carried_across(tree, loop_nid, tensor):
                continue
            if not _access_invariant_across(tree, producer, loop.loop_var, tensor):
                continue
            if not _access_invariant_across(tree, moved_leaf, loop.loop_var, tensor):
                continue
            raise TransformLegalityError(
                f"move(block={block_nid} under loop={target_loop_nid}) changes consumer "
                f"execution scope relative to producer {producer} for invariant tensor "
                f"{tensor!r} across loop {loop_nid} ({loop.loop_var!r})"
            )


def _check_move_preserves_dependencies(ir: KernelIR, block_nid: int, target_loop_nid: int, index: int) -> None:
    """Raise TransformLegalityError if the proposed move would make any
    dependency edge incident to the moved block point backward.

    Pure topological query — no deep copy, no ``_move``. Asks
    ``Dependency.first_backward_edge_for_insertion`` on the **original**
    program's dependency graph: edge *directions* are frozen at construction,
    and the moved leaf's post-splice preorder position is computed analytically
    from ``(target_loop_nid, index)``. Span-promotion delivers reduction-init
    domination and coverage — one span-based, edge-kind-agnostic rule covers
    both reduction-init domination and consumer-before-producer ordering.

    Directions MUST come from ``ir.dependency`` (the pre-move graph). Rebuilding
    ``Dependency`` on a moved tree would be wrong: ``_build`` re-derives every
    flow edge from execution order, so a PARALLEL producer sunk past its
    consumer flips from RAW ``producer->consumer`` to WAR ``consumer->producer``
    and the violation disappears (matmul reads uninitialised data -> NaN).
    Freezing directions keeps the RAW orientation, so the post-splice backward
    span is detected.
    """
    _check_same_loop_prefix(ir, block_nid, target_loop_nid)
    moved_leaf = ir.dependency._resolve(block_nid)
    offending = ir.dependency.first_backward_edge_for_insertion(moved_leaf, target_loop_nid, index)
    result: None = None
    if offending is not None:
        a, b = offending
        raise TransformLegalityError(
            f"move(block={block_nid} under loop={target_loop_nid}) reorders dependency "
            f"edge {a}->{b} backward (a carried buffer's init/drain cannot enter its "
            f"reduction loop, nor a consumer precede its producer)"
        )
    _check_no_rmw_reset_scope_change(ir, block_nid, target_loop_nid)
    _check_no_consumer_hoisted_out_of_producer_loop(ir, block_nid, target_loop_nid)
    return result


def _check_versioned_pipeline_boundary(ir: KernelIR, block_nid: int, target_loop_nid: int) -> None:
    """Reject moving a multi-version buffer access into or out of a pipeline loop."""
    touched: set[str] = set()
    for nid in [block_nid, *ir.tree.descendants(block_nid)]:
        node = ir.tree.data(nid)
        if isinstance(node, ISANode):
            touched.update(region.tensor for region in node.operand_bindings.values())
    versioned = {name for name in touched if ir.buffer(name).versions > 1}
    if not versioned:
        return
    old_ancestors = set(ir.tree.ancestors(block_nid))
    new_ancestors = set(ir.tree.ancestors(target_loop_nid)) | {target_loop_nid}
    for owner_nid in ir.tree.blocks():
        owner = ir.tree.data(owner_nid)
        assert isinstance(owner, BlockNode)
        annotation = owner.annotations.get("software_pipeline")
        if annotation is None:
            continue
        pipeline_loop = annotation["loop_nid"]
        if (pipeline_loop in old_ancestors) != (pipeline_loop in new_ancestors):
            raise TransformLegalityError(
                f"move(block={block_nid} under loop={target_loop_nid}) crosses software pipeline "
                f"loop {pipeline_loop} while touching versioned buffer(s) {sorted(versioned)}"
            )


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


@dataclass(frozen=True)
class CodeMotionOption(TransformOption):
    """Relocate ``block_nid`` under ``target_loop_nid`` at child slot ``index``.

    One option type for both directions of motion: sinking a producer under a
    consumer's loop and lifting a consumer under a producer's loop are the same
    structural splice, distinguished only by the dependency graph — not a flag.
    """

    block_nid: int
    target_loop_nid: int
    index: int


class CodeMotion(Transform[CodeMotionOption]):
    """Relocate one block under a target loop (the merged former ComputeAt/ReverseComputeAt).

    Legality is dependency-ordering (span-promotion) + the structural same-prefix
    merge + the reduction-replication guard. There is NO output-block guard: the
    block writing the return tensor is relocatable when ordering permits (e.g. the
    k11->k12 store-sink under the matmul's N loop).
    """

    def apply(self, ir: KernelIR, option: CodeMotionOption) -> KernelIR:
        """Re-check legality, deep-copy, move, rebuild deps, return.

        Structural-only: the block relocation + Dependency rebuild. Buffer
        placement/shape/frame is now an explicit BufferCompaction step, not an
        anonymous tail (see the 2026-07-14 BufferCompaction design).
        """
        self._check_legality(ir, option)
        new_ir = copy.deepcopy(ir)
        _move(new_ir, block_nid=option.block_nid, target_loop_nid=option.target_loop_nid, index=option.index)
        new_ir.dependency = Dependency(new_ir.tree)
        return new_ir

    def analyze(self, ir: KernelIR) -> list[CodeMotionOption]:
        """Enumerate (block, target loop, index) triples passing legality."""
        options: list[CodeMotionOption] = []
        leaf_blocks = [
            nid
            for nid in ir.tree.blocks()
            if nid != ir.tree.root
            and sum(1 for d in ir.tree.descendants(nid) if isinstance(ir.tree.data(d), ISANode)) == 1
        ]
        for block_nid in leaf_blocks:
            for target_nid in ir.tree.preorder():
                if not isinstance(ir.tree.data(target_nid), ForNode):
                    continue
                for index in self._legal_indices(ir, block_nid, target_nid):
                    opt = CodeMotionOption(block_nid=block_nid, target_loop_nid=target_nid, index=index)
                    try:
                        self._check_legality(ir, opt)
                    except TransformLegalityError:
                        continue
                    options.append(opt)
        return options

    def _legal_indices(self, ir: KernelIR, block_nid: int, target_nid: int) -> list[int]:
        """Slots in the insertion gap (lp, fc] among the target loop's children.

        Bounded below by the last child holding a producer of the moved block and
        above by the first child holding a consumer — symmetric in both, which is
        why one enumeration serves producer-sink and consumer-lift alike.
        """
        children = ir.tree.children(target_nid)
        producers = ir.dependency.producers(block_nid)
        consumers = ir.dependency.consumers(block_nid)
        lp = -1
        fc = len(children)
        for i, child in enumerate(children):
            sub = ir.tree.descendants(child) | {child}
            if sub & producers:
                lp = i
            if sub & consumers and i < fc:
                fc = i
        return list(range(lp + 1, fc + 1))

    def _check_legality(self, ir: KernelIR, option: CodeMotionOption) -> None:
        """Structural checks (target/block in graph, target a ForNode, target not a
        descendant of the block) then span-promotion ordering. No output guard."""
        if option.target_loop_nid not in ir.tree.graph:
            raise TransformLegalityError(f"target_loop_nid={option.target_loop_nid} not in tree")
        if not isinstance(ir.tree.data(option.target_loop_nid), ForNode):
            raise TransformLegalityError(
                f"CodeMotion requires target_loop_nid to be a ForNode; got "
                f"{type(ir.tree.data(option.target_loop_nid)).__name__}"
            )
        if option.block_nid not in ir.tree.graph:
            raise TransformLegalityError(f"block_nid={option.block_nid} not in tree")
        if option.target_loop_nid in ir.tree.descendants(option.block_nid):
            raise TransformLegalityError(
                f"target_loop_nid={option.target_loop_nid} is a descendant of moved block "
                f"{option.block_nid} (cannot move under its own loop)"
            )
        _check_versioned_pipeline_boundary(ir, option.block_nid, option.target_loop_nid)
        _check_move_preserves_dependencies(ir, option.block_nid, option.target_loop_nid, option.index)


__all__ = ["_move", "_check_move_preserves_dependencies", "CodeMotion", "CodeMotionOption"]
