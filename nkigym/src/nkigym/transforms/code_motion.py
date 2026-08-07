"""Move one block under a loop by merging an exact loop-prefix match."""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from fractions import Fraction
from math import prod
from weakref import WeakKeyDictionary

from nkigym.ir import KernelIR
from nkigym.ir.arith.expr import Expr, Var, substitute, to_affine
from nkigym.ir.dependency import (
    Dependency,
    _access_invariant_across,
    _leaf_operand_regions,
    _rmw_operand_slots,
    _tensor_carried_across,
)
from nkigym.ir.interval import regions_disjoint
from nkigym.ir.tree import BlockNode, BufferRegion, ForNode, ISANode, KernelTree
from nkigym.ops.base import AxisRole
from nkigym.transforms.base import Transform, TransformLegalityError, TransformOption
from nkigym.transforms.helper.access_pattern import subtree_has_access_patterns
from nkigym.transforms.helper.normalize import _dim_from_loopvar, normalize_block
from nkigym.transforms.helper.tree_ops import (
    _block_local_descendants,
    _replace_in_parent_children,
    invalidate_stale_software_pipelines,
)


@dataclass(frozen=True)
class _PrefixPlan:
    """Exact-prefix loop correspondence for one CodeMotion move."""

    target_loop_nids: tuple[int, ...]
    local_loop_nids: tuple[int, ...]
    matched_loop_nids: tuple[tuple[int, int], ...]
    matched_local_nids: tuple[int, ...]
    duplicated_target_nids: tuple[int, ...]


@dataclass(frozen=True)
class _AnalysisContext:
    """Code-motion facts shared by every option analyzed on one IR."""

    versioned_by_block: dict[int, frozenset[str]]
    pipeline_stages: dict[int, dict[int, int]]
    topology: tuple[dict[int, int], dict[int, tuple[int, ...]], dict[int, frozenset[int]]]


_ANCESTORS: WeakKeyDictionary[KernelTree, dict[int, tuple[int, ...]]] = WeakKeyDictionary()
_DESCENDANTS: WeakKeyDictionary[KernelTree, dict[int, frozenset[int]]] = WeakKeyDictionary()
_PREORDERS: WeakKeyDictionary[KernelTree, dict[int | None, tuple[int, ...]]] = WeakKeyDictionary()
_OWNING_BLOCKS: WeakKeyDictionary[KernelTree, dict[int, int]] = WeakKeyDictionary()
_OWNED_LEAVES: WeakKeyDictionary[KernelTree, dict[int, int]] = WeakKeyDictionary()
_DEPENDENCY_LEAVES: WeakKeyDictionary[KernelTree, dict[int, int]] = WeakKeyDictionary()
_LOOP_STRIDES: WeakKeyDictionary[KernelTree, dict[tuple[int, int], Fraction]] = WeakKeyDictionary()
_CROSSED_LOOPS: WeakKeyDictionary[KernelTree, dict[tuple[int, int, _PrefixPlan], tuple[int, ...]]] = WeakKeyDictionary()
_TARGET_LOOPS: WeakKeyDictionary[KernelTree, dict[int, tuple[int, ...]]] = WeakKeyDictionary()
_LOCAL_LOOPS: WeakKeyDictionary[KernelTree, dict[int, tuple[int, ...]]] = WeakKeyDictionary()
_BOUND_DIMS: WeakKeyDictionary[KernelTree, dict[int, dict[str, str]]] = WeakKeyDictionary()


def regions_overlap(
    ir: KernelIR, first_leaf: int, first_region: BufferRegion, second_leaf: int, second_region: BufferRegion
) -> bool:
    """Return whether two regions of one materialized tensor may overlap."""
    extents = {**ir.dependency.info(first_leaf).extents, **ir.dependency.info(second_leaf).extents}
    buffer = ir.buffer(first_region.tensor)
    return not regions_disjoint(first_region, second_region, buffer, buffer, extents)


def loop_carries_plain_state(ir: KernelIR, loop_nid: int, tensor: str, excluded_leaf: int) -> bool:
    """Return whether a read-before-write carries a tensor between iterations."""
    tree = ir.tree
    loop = tree.loop(loop_nid)
    accesses: list[tuple[int, tuple[BufferRegion, ...], tuple[BufferRegion, ...]]] = []
    for leaf in tree.preorder(loop_nid):
        if leaf == excluded_leaf or not isinstance(tree.data(leaf), ISANode):
            continue
        info = ir.dependency.info(leaf)
        reads = tuple(
            region
            for region in info.read_regions
            if region.tensor == tensor and _access_invariant_across(tree, leaf, loop.loop_var, tensor)
        )
        writes = tuple(
            region
            for region in info.write_regions
            if region.tensor == tensor and _access_invariant_across(tree, leaf, loop.loop_var, tensor)
        )
        if reads or writes:
            accesses.append((leaf, reads, writes))
    return any(
        not any(
            regions_overlap(ir, prior_leaf, prior_write, read_leaf, read_region)
            for prior_leaf, _prior_reads, prior_writes in accesses[:read_index]
            for prior_write in prior_writes
        )
        and any(
            regions_overlap(ir, later_leaf, later_write, read_leaf, read_region)
            for later_leaf, _later_reads, later_writes in accesses[read_index:]
            for later_write in later_writes
        )
        for read_index, (read_leaf, read_regions, _read_writes) in enumerate(accesses)
        for read_region in read_regions
    )


def _ancestors(tree: KernelTree, nid: int) -> tuple[int, ...]:
    """Return one cached root-first ancestor chain."""
    cache = _ANCESTORS.setdefault(tree, {})
    ancestors = cache.get(nid)
    if ancestors is None:
        ancestors = tuple(tree.ancestors(nid))
        cache[nid] = ancestors
    return ancestors


def _descendants(tree: KernelTree, nid: int) -> frozenset[int]:
    """Return one cached descendant set."""
    cache = _DESCENDANTS.setdefault(tree, {})
    descendants = cache.get(nid)
    if descendants is None:
        descendants = frozenset(tree.descendants(nid))
        cache[nid] = descendants
    return descendants


def _preorder(tree: KernelTree, nid: int | None = None) -> tuple[int, ...]:
    """Return one cached preorder traversal."""
    cache = _PREORDERS.setdefault(tree, {})
    preorder = cache.get(nid)
    if preorder is None:
        preorder = tuple(tree.preorder(nid))
        cache[nid] = preorder
    return preorder


def _clear_analysis_cache(tree: KernelTree) -> None:
    """Discard cached structural facts after mutating ``tree``."""
    for cache in (
        _ANCESTORS,
        _DESCENDANTS,
        _PREORDERS,
        _OWNING_BLOCKS,
        _OWNED_LEAVES,
        _DEPENDENCY_LEAVES,
        _LOOP_STRIDES,
        _CROSSED_LOOPS,
        _TARGET_LOOPS,
        _LOCAL_LOOPS,
        _BOUND_DIMS,
    ):
        cache.pop(tree, None)


def _move(ir: KernelIR, block_nid: int, target_loop_nid: int, index: int) -> None:
    """Relocate a block by rebinding and removing its matched loop prefix."""
    tree = ir.tree
    same_parent = tree.parent(block_nid) == target_loop_nid
    if same_parent:
        _splice_under_target(tree, block_nid, target_loop_nid, index)
    else:
        plan = _prefix_plan(tree, block_nid, target_loop_nid)
        _prepare_block_for_splice(tree, block_nid, plan)
        _strip_local_prefix_loops(tree, block_nid, len(plan.matched_local_nids))
        _splice_under_target(tree, block_nid, target_loop_nid, index)
        normalize_block(tree, block_nid)
    _assert_single_parent(tree)
    _clear_analysis_cache(tree)


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
    """Remove the outermost matched loops from one block-local loop chain."""
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
    """Return target-loop ancestors in true outer-to-inner tree order."""
    chain = [*_ancestors(tree, target_loop_nid), target_loop_nid]
    return [(node.loop_var, node.extent) for n in chain if isinstance((node := tree.data(n)), ForNode)]


def _target_loop_nids(tree: KernelTree, target_loop_nid: int) -> list[int]:
    """ForNode nids from the outermost target ancestor through the target."""
    cache = _TARGET_LOOPS.setdefault(tree, {})
    target_nids = cache.get(target_loop_nid)
    if target_nids is None:
        chain = [*_ancestors(tree, target_loop_nid), target_loop_nid]
        target_nids = tuple(nid for nid in chain if isinstance(tree.data(nid), ForNode))
        cache[target_loop_nid] = target_nids
    return list(target_nids)


def _local_loop_nids(tree: KernelTree, block_nid: int) -> list[int]:
    """Block-local ForNode nids in execution order."""
    cache = _LOCAL_LOOPS.setdefault(tree, {})
    local_nids = cache.get(block_nid)
    if local_nids is None:
        leaf = next(nid for nid in _preorder(tree, block_nid) if isinstance(tree.data(nid), ISANode))
        chain = _ancestors(tree, leaf)
        start = chain.index(block_nid) + 1
        local_nids = tuple(nid for nid in chain[start:] if isinstance(tree.data(nid), ForNode))
        cache[block_nid] = local_nids
    return list(local_nids)


def _bound_loop_dims(block: BlockNode) -> dict[str, str]:
    """Map each loop variable in the block's iter bindings to its concrete dimension."""
    result: dict[str, str] = {}
    for iter_var, value in zip(block.iter_vars, block.iter_values):
        for name in to_affine(value):
            if name is not None:
                result[name] = iter_var.axis
    return result


def _cached_bound_loop_dims(tree: KernelTree, block_nid: int) -> dict[str, str]:
    """Return cached loop-variable dimensions for one block."""
    cache = _BOUND_DIMS.setdefault(tree, {})
    dimensions = cache.get(block_nid)
    if dimensions is None:
        dimensions = _bound_loop_dims(tree.block(block_nid))
        cache[block_nid] = dimensions
    return dimensions


def _owning_block(tree: KernelTree, nid: int) -> int:
    """Return the nearest BlockNode ancestor that owns ``nid``."""
    cache = _OWNING_BLOCKS.setdefault(tree, {})
    owner = cache.get(nid)
    if owner is None:
        owner = next(
            (ancestor for ancestor in reversed(_ancestors(tree, nid)) if isinstance(tree.data(ancestor), BlockNode)),
            None,
        )
        if owner is None:
            raise TransformLegalityError(f"node {nid} has no enclosing BlockNode")
        cache[nid] = owner
    return owner


def _owned_leaf(tree: KernelTree, block_nid: int) -> int:
    """Return the ISA leaf directly owned by ``block_nid``."""
    cache = _OWNED_LEAVES.setdefault(tree, {})
    leaf = cache.get(block_nid)
    if leaf is None:
        owned = [
            nid
            for nid in _preorder(tree, block_nid)
            if isinstance(tree.data(nid), ISANode) and _owning_block(tree, nid) == block_nid
        ]
        if len(owned) != 1:
            raise TransformLegalityError(f"block {block_nid} must own exactly one ISA leaf; found {owned}")
        leaf = owned[0]
        cache[block_nid] = leaf
    return leaf


def _dependency_leaf(ir: KernelIR, block_nid: int) -> int:
    """Return the sole dependency endpoint in a leaf block or container."""
    cache = _DEPENDENCY_LEAVES.setdefault(ir.tree, {})
    leaf = cache.get(block_nid)
    if leaf is None:
        leaves = [
            nid
            for nid in _preorder(ir.tree, block_nid)
            if isinstance(ir.tree.data(nid), ISANode) and nid in ir.dependency.graph
        ]
        if len(leaves) != 1:
            raise TransformLegalityError(
                f"block {block_nid} must contain exactly one dependency endpoint; found {leaves}"
            )
        leaf = leaves[0]
        cache[block_nid] = leaf
    return leaf


def _bound_execution_extents(tree: KernelTree, block_nid: int, bound_names: set[str]) -> dict[str, int]:
    """Return consistent loop extents for a leaf block or structural container."""
    try:
        leaves = (_owned_leaf(tree, block_nid),)
    except TransformLegalityError:
        leaves = tuple(nid for nid in _preorder(tree, block_nid) if isinstance(tree.data(nid), ISANode))
        if not leaves:
            raise TransformLegalityError(f"block {block_nid} has no ISA descendants")
    extent_maps: list[dict[str, int]] = []
    for leaf in leaves:
        extents: dict[str, int] = {}
        for nid in _ancestors(tree, leaf):
            node = tree.data(nid)
            if not isinstance(node, ForNode) or node.loop_var not in bound_names:
                continue
            if node.loop_var in extents:
                raise TransformLegalityError(
                    f"block {block_nid} has duplicate bound loop name {node.loop_var!r} in its execution scope"
                )
            extents[node.loop_var] = node.extent
        missing = bound_names - extents.keys()
        if missing:
            raise TransformLegalityError(f"block {block_nid} has no execution loops for bindings {sorted(missing)}")
        extent_maps.append(extents)
    first = extent_maps[0]
    if any(extents != first for extents in extent_maps[1:]):
        raise TransformLegalityError(f"container block {block_nid} has inconsistent bound loop extents")
    return first


def _loop_element_stride(tree: KernelTree, block_nid: int, loop_nid: int) -> Fraction:
    """Return one bound loop's logical element stride in ``block_nid``."""
    cache = _LOOP_STRIDES.setdefault(tree, {})
    key = (block_nid, loop_nid)
    cached = cache.get(key)
    if cached is not None:
        return cached
    block = tree.block(block_nid)
    loop_var = tree.loop(loop_nid).loop_var
    matches = [
        (iter_var, to_affine(value))
        for iter_var, value in zip(block.iter_vars, block.iter_values)
        if loop_var in to_affine(value)
    ]
    if len(matches) != 1:
        raise TransformLegalityError(
            f"loop {loop_nid} ({loop_var!r}) must bind exactly one iter_var in block {block_nid}"
        )
    iter_var, affine = matches[0]
    bound_names = {name for name in affine if name is not None}
    extents = _bound_execution_extents(tree, block_nid, bound_names)
    domain_extent = iter_var.dom[1] - iter_var.dom[0]
    stride = Fraction(affine[loop_var] * domain_extent, prod(extents.values()))
    cache[key] = stride
    return stride


def _prefix_plan(tree: KernelTree, block_nid: int, target_loop_nid: int) -> _PrefixPlan:
    """Match target loops to the moved block's bound prefix by dimension and extent."""
    target_nids = _target_loop_nids(tree, target_loop_nid)
    local_nids = _local_loop_nids(tree, block_nid)
    leaf = next(nid for nid in _preorder(tree, block_nid) if isinstance(tree.data(nid), ISANode))
    moved_nids = [nid for nid in _ancestors(tree, leaf) if isinstance(tree.data(nid), ForNode)]
    enclosing_nids = set(moved_nids) - set(local_nids)
    bound_dims = _cached_bound_loop_dims(tree, block_nid)
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
        if moved_nid != target_nid:
            target_block_nid = _owning_block(tree, target_nid)
            target_bound_dims = _cached_bound_loop_dims(tree, target_block_nid)
            if target_loop.loop_var in target_bound_dims:
                moved_stride = _loop_element_stride(tree, block_nid, moved_nid)
                target_stride = _loop_element_stride(tree, target_block_nid, target_nid)
                if moved_stride != target_stride:
                    raise TransformLegalityError(
                        f"move(block={block_nid} under loop={target_loop_nid}) cannot replace "
                        f"{moved_loop.loop_var!r} with {target_loop.loop_var!r}: logical element "
                        f"strides differ ({moved_stride} != {target_stride})"
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
    """Raise if a move leaves any node with multiple parents."""
    multi = [n for n in tree.graph.nodes if len(list(tree.graph.predecessors(n))) > 1]
    if multi:
        detail = {n: list(tree.graph.predecessors(n)) for n in multi}
        raise ValueError(f"_move left nodes with multiple parents: {detail}")


def _check_same_loop_prefix(
    ir: KernelIR, block_nid: int, target_loop_nid: int, plan: _PrefixPlan | None = None
) -> list[tuple[str, int]]:
    """Require the target loops to match the moved block's dependent prefix."""
    target_seq = _target_loop_seq(ir.tree, target_loop_nid)
    block = ir.tree.data(block_nid)
    assert isinstance(block, BlockNode)
    resolved_plan = plan if plan is not None else _prefix_plan(ir.tree, block_nid, target_loop_nid)
    _check_no_partial_input_replicated(ir, block_nid, target_loop_nid, resolved_plan.duplicated_target_nids)
    _check_no_mutating_input_replicated(ir, block_nid, target_loop_nid, resolved_plan.duplicated_target_nids)
    _check_no_feedback_output_replicated(ir, block_nid, target_loop_nid, resolved_plan.duplicated_target_nids)
    _check_no_reduction_replicated(ir, block_nid, target_loop_nid, resolved_plan.duplicated_target_nids)
    return target_seq


def _check_no_partial_input_replicated(
    ir: KernelIR, block_nid: int, target_loop_nid: int, duplicated_target_nids: tuple[int, ...]
) -> None:
    """Reject a replicated consumer that enters its producer's tiled loop."""
    moved_leaf = _dependency_leaf(ir, block_nid)
    moved_reads = ir.dependency.info(moved_leaf).read_regions
    for loop_nid in duplicated_target_nids:
        loop = ir.tree.loop(loop_nid)
        descendants = _descendants(ir.tree, loop_nid)
        for producer, _consumer, attrs in ir.dependency.graph.in_edges(moved_leaf, data=True):
            tensor = attrs.get("tensor")
            if producer not in descendants or not isinstance(tensor, str):
                continue
            producer_writes = tuple(
                region for region in ir.dependency.info(producer).write_regions if region.tensor == tensor
            )
            consumer_reads = tuple(region for region in moved_reads if region.tensor == tensor)
            producer_varies = any(
                loop.loop_var in to_affine(lower) for region in producer_writes for lower, _width in region.ranges
            )
            consumer_is_invariant = bool(consumer_reads) and all(
                loop.loop_var not in to_affine(lower) for region in consumer_reads for lower, _width in region.ranges
            )
            if producer_varies and consumer_is_invariant:
                raise TransformLegalityError(
                    f"move(block={block_nid} under loop={target_loop_nid}) replicates a full read "
                    f"of tensor {tensor!r} inside loop {loop_nid}, whose producer writes a "
                    f"different slice per iteration"
                )


def _check_no_mutating_input_replicated(
    ir: KernelIR, block_nid: int, target_loop_nid: int, duplicated_target_nids: tuple[int, ...]
) -> None:
    """Reject recomputation that closes a feedback path through a block input."""
    moved_leaf = _dependency_leaf(ir, block_nid)
    moved_reads = ir.dependency.info(moved_leaf).read_regions
    for loop_nid in duplicated_target_nids:
        for writer in _preorder(ir.tree, loop_nid):
            if not isinstance(ir.tree.data(writer), ISANode):
                continue
            if not ir.dependency.must_precede(moved_leaf, writer):
                continue
            for write_region in ir.dependency.info(writer).write_regions:
                for read_region in moved_reads:
                    if read_region.tensor != write_region.tensor:
                        continue
                    if regions_overlap(ir, moved_leaf, read_region, writer, write_region):
                        raise TransformLegalityError(
                            f"move(block={block_nid} under loop={target_loop_nid}) replicates "
                            f"a feedback read of tensor {read_region.tensor!r} across loop "
                            f"{loop_nid}, whose downstream path writes the same region"
                        )


def _check_no_feedback_output_replicated(
    ir: KernelIR, block_nid: int, target_loop_nid: int, duplicated_target_nids: tuple[int, ...]
) -> None:
    """Reject a replicated write that feeds an earlier loop operation."""
    moved_leaf = _dependency_leaf(ir, block_nid)
    moved_writes = ir.dependency.info(moved_leaf).write_regions
    for loop_nid in duplicated_target_nids:
        for reader in _preorder(ir.tree, loop_nid):
            if not isinstance(ir.tree.data(reader), ISANode):
                continue
            if not ir.dependency.must_precede(reader, moved_leaf):
                continue
            for read_region in ir.dependency.info(reader).read_regions:
                for write_region in moved_writes:
                    if read_region.tensor != write_region.tensor:
                        continue
                    if regions_overlap(ir, reader, read_region, moved_leaf, write_region):
                        raise TransformLegalityError(
                            f"move(block={block_nid} under loop={target_loop_nid}) replicates "
                            f"a feedback write to tensor {write_region.tensor!r} across loop "
                            f"{loop_nid}, whose earlier path reads the same region"
                        )


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


def _crossed_execution_loops(ir: KernelIR, block_nid: int, target_loop_nid: int, plan: _PrefixPlan) -> list[int]:
    """Return loops added to or removed from the moved leaf's execution scope."""
    tree = ir.tree
    cache = _CROSSED_LOOPS.setdefault(tree, {})
    key = (block_nid, target_loop_nid, plan)
    cached = cache.get(key)
    if cached is not None:
        return list(cached)
    leaf = _dependency_leaf(ir, block_nid)
    old_loops = [nid for nid in _ancestors(tree, leaf) if isinstance(tree.data(nid), ForNode)]
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
    cache[key] = tuple(crossed)
    return crossed


def _plain_written_tensors(node: ISANode) -> set[str]:
    """Return tensors written by output operands that are not read-modify-write."""
    input_slots = getattr(node.op_cls, "INPUT_OPERANDS", frozenset())
    rmw_slots = _rmw_operand_slots(node)
    return {
        region.tensor
        for slot, region in node.operand_bindings.items()
        if slot not in input_slots and slot not in rmw_slots
    }


def _rmw_value_spans_loop(ir: KernelIR, rmw_leaf: int, loop_nid: int, tensor: str) -> bool:
    """Return whether a RAW consumer observes values from multiple iterations."""
    loop = ir.tree.loop(loop_nid)
    loop_descendants = _descendants(ir.tree, loop_nid)
    spans = any(
        attrs.get("kind") == "RAW"
        and attrs.get("tensor") == tensor
        and (consumer not in loop_descendants or _access_invariant_across(ir.tree, consumer, loop.loop_var, tensor))
        for _producer, consumer, attrs in ir.dependency.graph.out_edges(rmw_leaf, data=True)
    )
    return spans


def _check_no_rmw_reset_scope_change(ir: KernelIR, block_nid: int, target_loop_nid: int, plan: _PrefixPlan) -> None:
    """Reject changing a plain reset's frequency relative to an RMW.

    A plain write followed by an RMW of the same region is its reset. Moving that
    writer or the RMW across a loop changes behavior when both accesses are
    invariant, or when a tiled RMW remains live after the loop. Neither case
    necessarily reverses a dependency edge.
    """
    tree = ir.tree
    moved_leaf = _dependency_leaf(ir, block_nid)
    moved_node = tree.data(moved_leaf)
    assert isinstance(moved_node, ISANode)
    plain_writes = _plain_written_tensors(moved_node)
    crossed_loops = _crossed_execution_loops(ir, block_nid, target_loop_nid, plan)
    if plain_writes:
        for tensor in plain_writes:
            for loop_nid in crossed_loops:
                loop = tree.loop(loop_nid)
                if not _access_invariant_across(tree, moved_leaf, loop.loop_var, tensor):
                    continue
                if loop_carries_plain_state(ir, loop_nid, tensor, moved_leaf):
                    raise TransformLegalityError(
                        f"move(block={block_nid} under loop={target_loop_nid}) changes reset "
                        f"frequency for carried tensor {tensor!r} across loop {loop_nid} "
                        f"({loop.loop_var!r})"
                    )
        for _producer, consumer, attrs in ir.dependency.graph.out_edges(moved_leaf, data=True):
            tensor = attrs.get("tensor")
            if tensor not in plain_writes or not _leaf_operand_regions(tree, consumer, tensor, rmw_only=True):
                continue
            for loop_nid in crossed_loops:
                loop = tree.data(loop_nid)
                assert isinstance(loop, ForNode)
                if consumer not in _descendants(tree, loop_nid):
                    continue
                if not _access_invariant_across(tree, moved_leaf, loop.loop_var, tensor):
                    continue
                if not _access_invariant_across(tree, consumer, loop.loop_var, tensor) and not _rmw_value_spans_loop(
                    ir, consumer, loop_nid, tensor
                ):
                    continue
                raise TransformLegalityError(
                    f"move(block={block_nid} under loop={target_loop_nid}) changes reset "
                    f"frequency for tensor {tensor!r} across read-modify-write loop "
                    f"{loop_nid} ({loop.loop_var!r})"
                )

    rmw_slots = _rmw_operand_slots(moved_node)
    rmw_tensors = {region.tensor for slot, region in moved_node.operand_bindings.items() if slot in rmw_slots}
    for producer, _consumer, attrs in ir.dependency.graph.in_edges(moved_leaf, data=True):
        tensor = attrs.get("tensor")
        producer_node = tree.data(producer)
        assert isinstance(producer_node, ISANode)
        if tensor not in rmw_tensors or tensor not in _plain_written_tensors(producer_node):
            continue
        for loop_nid in crossed_loops:
            loop = tree.data(loop_nid)
            assert isinstance(loop, ForNode)
            if producer not in _descendants(tree, loop_nid):
                continue
            if not _access_invariant_across(tree, producer, loop.loop_var, tensor):
                continue
            if not _access_invariant_across(tree, moved_leaf, loop.loop_var, tensor) and not _rmw_value_spans_loop(
                ir, moved_leaf, loop_nid, tensor
            ):
                continue
            raise TransformLegalityError(
                f"move(block={block_nid} under loop={target_loop_nid}) changes reset "
                f"frequency for tensor {tensor!r} across read-modify-write loop "
                f"{loop_nid} ({loop.loop_var!r})"
            )


def _loop_reinitializes_tensor(tree: KernelTree, loop_nid: int, tensor: str, excluded_leaves: frozenset[int]) -> bool:
    """Return whether a repeated invariant plain write resets ``tensor``."""
    loop = tree.loop(loop_nid)
    result = any(
        nid not in excluded_leaves
        and isinstance((node := tree.data(nid)), ISANode)
        and tensor in _plain_written_tensors(node)
        and _access_invariant_across(tree, nid, loop.loop_var, tensor)
        for nid in _descendants(tree, loop_nid)
    )
    return result


def _check_no_consumer_hoisted_out_of_producer_loop(
    ir: KernelIR, block_nid: int, target_loop_nid: int, plan: _PrefixPlan
) -> None:
    """Reject hoisting a consumer away from a repeated invariant producer."""
    tree = ir.tree
    moved_leaf = _dependency_leaf(ir, block_nid)
    old_loops = set(_ancestors(tree, moved_leaf))
    crossed_loops = _crossed_execution_loops(ir, block_nid, target_loop_nid, plan)
    for producer, _consumer, attrs in ir.dependency.graph.in_edges(moved_leaf, data=True):
        tensor = attrs.get("tensor")
        if tensor is None:
            continue
        for loop_nid in crossed_loops:
            if loop_nid not in old_loops or producer not in _descendants(tree, loop_nid):
                continue
            loop = tree.data(loop_nid)
            assert isinstance(loop, ForNode)
            if _loop_reinitializes_tensor(tree, loop_nid, tensor, frozenset((producer, moved_leaf))):
                raise TransformLegalityError(
                    f"move(block={block_nid} under loop={target_loop_nid}) delays consumer "
                    f"past repeated write to tensor {tensor!r} in loop {loop_nid} "
                    f"({loop.loop_var!r})"
                )
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


def _leaf_execution_invariant_across(tree: KernelTree, leaf_nid: int, loop_var: str) -> bool:
    """Return whether every operand region of one ISA leaf is loop-invariant."""
    node = tree.data(leaf_nid)
    assert isinstance(node, ISANode)
    return all(
        loop_var not in to_affine(lo) for region in node.operand_bindings.values() for lo, _width in region.ranges
    )


def _check_no_producer_moved_out_of_consumer_loop(
    ir: KernelIR, block_nid: int, target_loop_nid: int, plan: _PrefixPlan
) -> None:
    """Reject moving a producer away from a consumer sharing its invariant slice."""
    tree = ir.tree
    moved_leaf = _dependency_leaf(ir, block_nid)
    old_loops = set(_ancestors(tree, moved_leaf))
    crossed_loops = _crossed_execution_loops(ir, block_nid, target_loop_nid, plan)
    for _producer, consumer, attrs in ir.dependency.graph.out_edges(moved_leaf, data=True):
        tensor = attrs.get("tensor")
        if tensor is None:
            continue
        for loop_nid in crossed_loops:
            if loop_nid not in old_loops or consumer not in _descendants(tree, loop_nid):
                continue
            loop = tree.data(loop_nid)
            assert isinstance(loop, ForNode)
            if _tensor_carried_across(tree, loop_nid, tensor):
                continue
            if _leaf_execution_invariant_across(tree, moved_leaf, loop.loop_var):
                continue
            if not _access_invariant_across(tree, moved_leaf, loop.loop_var, tensor):
                continue
            if not _access_invariant_across(tree, consumer, loop.loop_var, tensor):
                continue
            raise TransformLegalityError(
                f"move(block={block_nid} under loop={target_loop_nid}) changes producer "
                f"execution scope relative to consumer {consumer} for invariant tensor "
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
    _check_move_changes_position(ir.tree, block_nid, target_loop_nid, index)
    plan = _prefix_plan(ir.tree, block_nid, target_loop_nid)
    _check_same_loop_prefix(ir, block_nid, target_loop_nid, plan)
    moved_leaf = _dependency_leaf(ir, block_nid)
    offending = ir.dependency.first_backward_edge_for_insertion(moved_leaf, target_loop_nid, index)
    result: None = None
    if offending is not None:
        a, b = offending
        raise TransformLegalityError(
            f"move(block={block_nid} under loop={target_loop_nid}) reorders dependency "
            f"edge {a}->{b} backward (a carried buffer's init/drain cannot enter its "
            f"reduction loop, nor a consumer precede its producer)"
        )
    _check_move_scope_changes(ir, block_nid, target_loop_nid, plan)
    return result


def _check_move_scope_changes(ir: KernelIR, block_nid: int, target_loop_nid: int, plan: _PrefixPlan) -> None:
    """Reject loop-scope changes not represented by dependency direction."""
    _check_no_rmw_reset_scope_change(ir, block_nid, target_loop_nid, plan)
    _check_no_consumer_hoisted_out_of_producer_loop(ir, block_nid, target_loop_nid, plan)
    _check_no_producer_moved_out_of_consumer_loop(ir, block_nid, target_loop_nid, plan)


def _analysis_context(ir: KernelIR, block_nids: list[int]) -> _AnalysisContext:
    """Collect versioned tensors and pipeline loops once for one analysis."""
    topology = ir.dependency._topology()
    _ANCESTORS[ir.tree] = topology[1]
    _DESCENDANTS[ir.tree] = topology[2]
    _PREORDERS[ir.tree] = {None: tuple(sorted(topology[0], key=topology[0].__getitem__))}
    buffers = ir.all_buffers()
    versioned_names = {name for name, buffer in buffers.items() if buffer.versions > 1}
    versioned_by_block: dict[int, frozenset[str]] = {}
    for block_nid in block_nids:
        touched: set[str] = set()
        for nid in [block_nid, *_descendants(ir.tree, block_nid)]:
            node = ir.tree.data(nid)
            if isinstance(node, ISANode):
                touched.update(region.tensor for region in node.operand_bindings.values())
        versioned_by_block[block_nid] = frozenset(touched & versioned_names)
    pipeline_stages = {
        annotation["loop_nid"]: dict(zip(annotation["children"], annotation["stages"]))
        for nid in ir.tree.blocks()
        if (annotation := ir.tree.block(nid).annotations.get("software_pipeline")) is not None
    }
    return _AnalysisContext(versioned_by_block=versioned_by_block, pipeline_stages=pipeline_stages, topology=topology)


def _direct_pipeline_child(tree: KernelTree, pipeline_loop: int, nid: int) -> int | None:
    """Return the direct pipeline-loop child containing ``nid``."""
    chain = (*_ancestors(tree, nid), nid)
    child = None
    if pipeline_loop in chain:
        position = chain.index(pipeline_loop) + 1
        if position < len(chain):
            child = chain[position]
    return child


def _check_pipeline_boundary(
    ir: KernelIR, block_nid: int, target_loop_nid: int, context: _AnalysisContext | None
) -> frozenset[int]:
    """Reject moves that change software-pipeline timing or versioning."""
    if context is None:
        context = _analysis_context(ir, [block_nid])
    versioned = context.versioned_by_block[block_nid]
    old_ancestors = set(_ancestors(ir.tree, block_nid))
    new_ancestors = set(_ancestors(ir.tree, target_loop_nid)) | {target_loop_nid}
    crossed_loops = frozenset(
        pipeline_loop
        for pipeline_loop in context.pipeline_stages
        if (pipeline_loop in old_ancestors) != (pipeline_loop in new_ancestors)
    )
    for pipeline_loop, stage_by_child in context.pipeline_stages.items():
        old_child = _direct_pipeline_child(ir.tree, pipeline_loop, block_nid)
        target_child = _direct_pipeline_child(ir.tree, pipeline_loop, target_loop_nid)
        if (
            old_child in stage_by_child
            and target_child in stage_by_child
            and stage_by_child[old_child] != stage_by_child[target_child]
            and old_child != block_nid
        ):
            raise TransformLegalityError(
                f"move(block={block_nid} under loop={target_loop_nid}) crosses software pipeline "
                f"loop {pipeline_loop} from stage {stage_by_child[old_child]} "
                f"to stage {stage_by_child[target_child]}"
            )
        if versioned and pipeline_loop in crossed_loops:
            raise TransformLegalityError(
                f"move(block={block_nid} under loop={target_loop_nid}) crosses software pipeline "
                f"loop {pipeline_loop} while touching versioned buffer(s) {sorted(versioned)}"
            )
    return crossed_loops


def _splice_under_target(tree: KernelTree, block_nid: int, target_loop_nid: int, index: int) -> None:
    """Detach ``block_nid`` from its parent and insert under the target loop at ``index``."""
    old_parent = tree.parent(block_nid)
    assert old_parent is not None, f"moved block {block_nid} has no parent"
    _replace_in_parent_children(tree, old_parent, [block_nid], [])
    _prune_empty_loop_chain(tree, old_parent, target_loop_nid)
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


def _prune_empty_loop_chain(tree: KernelTree, nid: int, stop_nid: int) -> None:
    """Remove empty old-scope loops without removing the insertion target."""
    current = nid
    while current != stop_nid and isinstance(tree.data(current), ForNode) and not tree.children(current):
        parent = tree.parent(current)
        if parent is None:
            raise AssertionError(f"empty loop {current} has no parent")
        tree.graph.remove_node(current)
        current = parent


def _check_move_changes_position(tree: KernelTree, block_nid: int, target_loop_nid: int, index: int) -> None:
    """Reject a splice that leaves a block in its existing child slot."""
    if tree.parent(block_nid) != target_loop_nid:
        return
    original = tree.children(target_loop_nid)
    remaining = [child for child in original if child != block_nid]
    if index == -1:
        pos = len(remaining)
    elif index == -2:
        pos = 0
    elif index >= 0:
        pos = index
    else:
        raise TransformLegalityError(
            f"CodeMotion index={index} is unsupported; use -1 append, -2 prepend, or a nonnegative child slot"
        )
    reordered = remaining[:pos] + [block_nid] + remaining[pos:]
    if reordered == original:
        raise TransformLegalityError(
            f"move(block={block_nid} under loop={target_loop_nid} at index={index}) "
            f"does not change the block's child slot"
        )


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
        invalidated_pipeline_loops = self._check_legality(ir, option)
        new_ir = copy.deepcopy(ir)
        _move(new_ir, block_nid=option.block_nid, target_loop_nid=option.target_loop_nid, index=option.index)
        invalidate_stale_software_pipelines(new_ir, invalidated_pipeline_loops)
        new_ir.dependency = Dependency(new_ir.tree)
        return new_ir

    def analyze(self, ir: KernelIR) -> list[CodeMotionOption]:
        """Enumerate (block, target loop, index) triples passing legality."""
        options: list[CodeMotionOption] = []
        leaf_blocks = [
            nid
            for nid in ir.tree.blocks()
            if nid != ir.tree.root
            and sum(1 for d in _descendants(ir.tree, nid) if isinstance(ir.tree.data(d), ISANode)) == 1
        ]
        context = _analysis_context(ir, leaf_blocks)
        for block_nid in leaf_blocks:
            for target_nid in _preorder(ir.tree):
                if not isinstance(ir.tree.data(target_nid), ForNode):
                    continue
                indices = self._legal_indices(ir, block_nid, target_nid)
                if not indices:
                    continue
                try:
                    self._check_static_legality(ir, block_nid, target_nid, context)
                    plan = _prefix_plan(ir.tree, block_nid, target_nid)
                    _check_same_loop_prefix(ir, block_nid, target_nid, plan)
                    _check_move_scope_changes(ir, block_nid, target_nid, plan)
                except TransformLegalityError:
                    continue
                for index in indices:
                    moved_leaf = _dependency_leaf(ir, block_nid)
                    if (
                        ir.dependency.first_backward_edge_for_insertion(
                            moved_leaf, target_nid, index, topology=context.topology
                        )
                        is None
                    ):
                        options.append(CodeMotionOption(block_nid=block_nid, target_loop_nid=target_nid, index=index))
        return options

    def _legal_indices(self, ir: KernelIR, block_nid: int, target_nid: int) -> list[int]:
        """Slots in the insertion gap (lp, fc] among the target loop's children.

        Bounded below by the last child holding a producer of the moved block and
        above by the first child holding a consumer — symmetric in both, which is
        why one enumeration serves producer-sink and consumer-lift alike.
        """
        children = ir.tree.children(target_nid)
        moved_leaf = _dependency_leaf(ir, block_nid)
        producers = ir.dependency.producers(moved_leaf)
        consumers = ir.dependency.consumers(moved_leaf)
        lp = -1
        fc = len(children)
        for i, child in enumerate(children):
            sub = _descendants(ir.tree, child) | {child}
            if sub & producers:
                lp = i
            if sub & consumers and i < fc:
                fc = i
        indices = list(range(lp + 1, fc + 1))
        legal: list[int] = []
        for index in indices:
            try:
                _check_move_changes_position(ir.tree, block_nid, target_nid, index)
            except TransformLegalityError:
                continue
            legal.append(index)
        return legal

    def _check_static_legality(
        self, ir: KernelIR, block_nid: int, target_loop_nid: int, context: _AnalysisContext | None
    ) -> frozenset[int]:
        """Check option legality that does not depend on an insertion slot."""
        if target_loop_nid not in ir.tree.graph:
            raise TransformLegalityError(f"target_loop_nid={target_loop_nid} not in tree")
        if not isinstance(ir.tree.data(target_loop_nid), ForNode):
            raise TransformLegalityError(
                f"CodeMotion requires target_loop_nid to be a ForNode; got "
                f"{type(ir.tree.data(target_loop_nid)).__name__}"
            )
        if block_nid not in ir.tree.graph:
            raise TransformLegalityError(f"block_nid={block_nid} not in tree")
        if subtree_has_access_patterns(ir.tree, block_nid):
            raise TransformLegalityError("CodeMotion cannot move a block with an explicit access pattern")
        if target_loop_nid in _descendants(ir.tree, block_nid):
            raise TransformLegalityError(
                f"target_loop_nid={target_loop_nid} is a descendant of moved block "
                f"{block_nid} (cannot move under its own loop)"
            )
        return _check_pipeline_boundary(ir, block_nid, target_loop_nid, context)

    def _check_legality(self, ir: KernelIR, option: CodeMotionOption) -> frozenset[int]:
        """Structural checks (target/block in graph, target a ForNode, target not a
        descendant of the block) then span-promotion ordering. No output guard."""
        invalidated_pipeline_loops = self._check_static_legality(ir, option.block_nid, option.target_loop_nid, None)
        _check_move_preserves_dependencies(ir, option.block_nid, option.target_loop_nid, option.index)
        return invalidated_pipeline_loops


__all__ = ["_move", "_check_move_preserves_dependencies", "CodeMotion", "CodeMotionOption"]
