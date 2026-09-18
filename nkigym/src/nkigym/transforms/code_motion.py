"""Move one block across one scope boundary or one adjacent sibling."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, replace
from fractions import Fraction
from weakref import WeakKeyDictionary

from nkigym.ir import KernelIR
from nkigym.ir.arith.expr import Expr, Var, affine_coefficient, expr_variables, to_affine
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
from nkigym.transforms.base import Transform, TransformLegalityError, TransformOption, copy_for_rewrite
from nkigym.transforms.copy_propagation import independent_loop_accesses
from nkigym.transforms.helper.access_pattern import subtree_has_access_patterns
from nkigym.transforms.helper.normalize import _substitute_block_regions
from nkigym.transforms.helper.tree_ops import _block_local_descendants, _replace_in_parent_children
from nkigym.transforms.split import _tensorized_loop_element_stride


@dataclass(frozen=True)
class _PrefixPlan:
    """Exact-prefix loop correspondence for one CodeMotion move."""

    target_loop_nids: tuple[int, ...]
    local_loop_nids: tuple[int, ...]
    matched_loop_nids: tuple[tuple[int, int], ...]
    matched_local_nids: tuple[int, ...]
    duplicated_target_nids: tuple[int, ...]
    restored_loop_nids: tuple[int, ...]


@dataclass(frozen=True)
class _AnalysisContext:
    """Code-motion facts shared by every option analyzed on one IR."""

    leaf_blocks: tuple[int, ...]
    pipeline_stages: dict[int, dict[int, int]]
    topology: tuple[dict[int, int], dict[int, tuple[int, ...]], dict[int, frozenset[int]]]


@dataclass(frozen=True)
class _PrefixBlockFacts:
    """Moved-block loop facts shared across candidate target loops."""

    local_nids: tuple[int, ...]
    moved_nids: tuple[int, ...]
    enclosing_nids: frozenset[int]
    bound_dims: dict[str, str]
    dependent_dims: frozenset[str]
    bound_nids: tuple[int, ...]


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
_PREFIX_BLOCK_FACTS: WeakKeyDictionary[KernelTree, dict[int, _PrefixBlockFacts]] = WeakKeyDictionary()


def regions_overlap(
    ir: KernelIR, first_leaf: int, first_region: BufferRegion, second_leaf: int, second_region: BufferRegion
) -> bool:
    """Return whether two regions of one materialized tensor may overlap."""
    extents = {**ir.dependency.info(first_leaf).extents, **ir.dependency.info(second_leaf).extents}
    buffer = ir.param_buffers.get(first_region.tensor) or ir.dependency.info(first_leaf).buffers[first_region.tensor]
    return not regions_disjoint(first_region, second_region, buffer, buffer, extents)


def loop_carries_plain_state(ir: KernelIR, loop_nid: int, tensor: str, excluded_leaf: int) -> bool:
    """Return whether a read-before-write carries a tensor between iterations."""
    tree = ir.tree
    loop = tree.loop(loop_nid)
    accesses: list[tuple[int, tuple[BufferRegion, ...], tuple[BufferRegion, ...]]] = []
    for leaf in tree.preorder(loop_nid):
        if leaf == excluded_leaf or not isinstance(tree.data(leaf), ISANode):
            continue
        if not _access_invariant_across(tree, leaf, loop.loop_var, tensor):
            continue
        info = ir.dependency.info(leaf)
        reads = tuple(region for region in info.read_regions if region.tensor == tensor)
        writes = tuple(region for region in info.write_regions if region.tensor == tensor)
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
    if nid not in cache:
        cache[nid] = tuple(tree.ancestors(nid))
    return cache[nid]


def _descendants(tree: KernelTree, nid: int) -> frozenset[int]:
    """Return one cached descendant set."""
    cache = _DESCENDANTS.setdefault(tree, {})
    if nid not in cache:
        cache[nid] = frozenset(tree.descendants(nid))
    return cache[nid]


def _preorder(tree: KernelTree, nid: int | None = None) -> tuple[int, ...]:
    """Return one cached preorder traversal."""
    cache = _PREORDERS.setdefault(tree, {})
    if nid not in cache:
        cache[nid] = tuple(tree.preorder(nid))
    return cache[nid]


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
        _PREFIX_BLOCK_FACTS,
    ):
        cache.pop(tree, None)


def _move(ir: KernelIR, block_nid: int, target_loop_nid: int, index: int) -> None:
    """Relocate a block while preserving its required loop iterations."""
    tree = ir.tree
    same_parent = tree.parent(block_nid) == target_loop_nid
    if same_parent:
        _splice_under_target(tree, block_nid, target_loop_nid, index)
    else:
        plan = _prefix_plan(tree, block_nid, target_loop_nid)
        _prepare_block_for_splice(tree, block_nid, plan)
        _strip_local_prefix_loops(tree, plan.matched_local_nids)
        _splice_under_target(tree, block_nid, target_loop_nid, index)
    root = tree.block(tree.root)
    shards = root.annotations.get("program_shards")
    if isinstance(shards, dict) and any(nid not in tree.graph for nid in shards):
        annotations = dict(root.annotations)
        annotations["program_shards"] = {nid: programs for nid, programs in shards.items() if nid in tree.graph}
        tree.graph.nodes[tree.root]["data"] = replace(root, annotations=annotations)
    _assert_single_parent(tree)
    _clear_analysis_cache(tree)


def _prepare_block_for_splice(tree: KernelTree, block_nid: int, plan: _PrefixPlan) -> None:
    """Rebind matched loops, rename local loops, and restore exited iterations."""
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
        temporary = f"i_move_{local_nid}"
        while temporary in used_names:
            temporary = f"{temporary}_"
        used_names.add(temporary)
        substitutions[loop.loop_var] = Var(name=temporary)
        tree.graph.nodes[local_nid]["data"] = ForNode(loop_var=temporary, extent=loop.extent)
    for nid in tree.preorder(block_nid):
        if isinstance(tree.data(nid), BlockNode):
            _substitute_block_regions(tree, nid, substitutions)
    for nid in reversed(plan.restored_loop_nids):
        children = tree.children(block_nid)
        clone = tree.add_node(tree.loop(nid))
        _replace_in_parent_children(tree, block_nid, children, [clone])
        for child in children:
            tree.graph.add_edge(clone, child)


def _strip_local_prefix_loops(tree: KernelTree, loops: tuple[int, ...]) -> None:
    """Remove matched loops at their actual parents, preserving wrapper blocks."""
    for loop in loops:
        parent = tree.parent(loop)
        assert parent is not None, f"matched loop {loop} has no parent"
        assert isinstance(tree.data(loop), ForNode), f"expected ForNode to strip; got {type(tree.data(loop)).__name__}"
        grandchildren = tree.children(loop)
        _replace_in_parent_children(tree, parent, [loop], grandchildren)
        tree.graph.remove_node(loop)


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
    bindings = zip(block.iter_vars, block.iter_values)
    return {name: iter_var.axis for iter_var, value in bindings for name in expr_variables(value)}


def _cached_bound_loop_dims(tree: KernelTree, block_nid: int) -> dict[str, str]:
    """Return cached loop-variable dimensions for one block."""
    cache = _BOUND_DIMS.setdefault(tree, {})
    dimensions = cache.get(block_nid)
    if dimensions is None:
        dimensions = _bound_loop_dims(tree.block(block_nid))
        cache[block_nid] = dimensions
    return dimensions


def _prefix_block_facts(tree: KernelTree, block_nid: int) -> _PrefixBlockFacts:
    """Return cached loop-prefix facts for one moved block."""
    cache = _PREFIX_BLOCK_FACTS.setdefault(tree, {})
    facts = cache.get(block_nid)
    if facts is None:
        local_nids = tuple(_local_loop_nids(tree, block_nid))
        leaf = next(nid for nid in _preorder(tree, block_nid) if isinstance(tree.data(nid), ISANode))
        moved_nids = tuple(nid for nid in _ancestors(tree, leaf) if isinstance(tree.data(nid), ForNode))
        bound_dims = _cached_bound_loop_dims(tree, _owning_block(tree, leaf))
        facts = _PrefixBlockFacts(
            local_nids=local_nids,
            moved_nids=moved_nids,
            enclosing_nids=frozenset(moved_nids) - frozenset(local_nids),
            bound_dims=bound_dims,
            dependent_dims=frozenset(bound_dims.values()),
            bound_nids=tuple(nid for nid in moved_nids if tree.loop(nid).loop_var in bound_dims),
        )
        cache[block_nid] = facts
    return facts


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


def _loop_element_stride(tree: KernelTree, block_nid: int, loop_nid: int) -> Fraction:
    """Return one bound loop's logical element stride in ``block_nid``."""
    cache = _LOOP_STRIDES.setdefault(tree, {})
    key = (block_nid, loop_nid)
    if key not in cache:
        loop_var = tree.loop(loop_nid).loop_var
        block = tree.block(block_nid)
        direct = [
            (iter_var, value)
            for iter_var, value in zip(block.iter_vars, block.iter_values)
            if loop_var in expr_variables(value)
        ]
        abstract = (
            next((name for name, axis in block.axis_map.items() if axis == direct[0][0].axis), None)
            if len(direct) == 1 and direct[0][1] == Var(name=loop_var)
            else None
        )
        leaf = tree.isa(_owned_leaf(tree, block_nid))
        folded = abstract is not None and any(
            len(group) > 1 and abstract in group
            for slot in leaf.operand_bindings
            for group in leaf.op_cls.operand_axis_groups(slot)
        )
        cache[key] = Fraction(1) if folded else _tensorized_loop_element_stride(tree, block_nid, loop_nid)
    return cache[key]


def _try_prefix_plan(
    tree: KernelTree, block_nid: int, target_loop_nid: int, facts: _PrefixBlockFacts | None = None
) -> _PrefixPlan | None:
    """Resolve an exact loop-prefix match, or return ``None`` when incompatible."""
    crossed = set(_ancestors(tree, block_nid)) ^ set((*_ancestors(tree, target_loop_nid), target_loop_nid))
    if any(isinstance(node := tree.data(nid), BlockNode) and "predicate" in node.annotations for nid in crossed):
        return None
    target_nids = _target_loop_nids(tree, target_loop_nid)
    facts = _prefix_block_facts(tree, block_nid) if facts is None else facts
    matched: list[tuple[int, int]] = []
    duplicated: list[int] = []
    bound_index = 0
    for target_nid in target_nids:
        target_loop = tree.loop(target_nid)
        if target_nid in facts.enclosing_nids and target_loop.loop_var not in facts.bound_dims:
            continue
        target_block_nid = _owning_block(tree, target_nid)
        target_dim = _cached_bound_loop_dims(tree, target_block_nid).get(target_loop.loop_var)
        if target_dim is None or target_dim not in facts.dependent_dims:
            duplicated.append(target_nid)
            continue
        if bound_index >= len(facts.bound_nids):
            return None
        moved_nid = facts.bound_nids[bound_index]
        moved_loop = tree.loop(moved_nid)
        moved_dim = facts.bound_dims.get(moved_loop.loop_var)
        if (moved_dim, moved_loop.extent) != (target_dim, target_loop.extent):
            return None
        if moved_nid != target_nid:
            moved_owner = _owning_block(tree, moved_nid)
            moved_stride = _loop_element_stride(tree, moved_owner, moved_nid)
            target_stride = _loop_element_stride(tree, target_block_nid, target_nid)
            if moved_stride != target_stride:
                return None
        matched.append((moved_nid, target_nid))
        bound_index += 1
    matched_moved = {moved_nid for moved_nid, _target_nid in matched}
    lost_enclosing = tuple(nid for nid in facts.bound_nids if nid in facts.enclosing_nids and nid not in matched_moved)
    local_set = set(facts.local_nids)
    matched_local_nids = tuple(moved_nid for moved_nid, _target_nid in matched if moved_nid in local_set)
    if matched_local_nids != facts.local_nids[: len(matched_local_nids)]:
        return None
    removed_enclosing = facts.enclosing_nids - set(target_nids)
    scopes = sum(bool(tree.block(nid).alloc_buffers) for nid in crossed if isinstance(tree.data(nid), BlockNode))
    boundaries = len(matched_local_nids) + len(duplicated) + len(removed_enclosing)
    if boundaries + scopes > 1:
        return None
    return _PrefixPlan(
        target_loop_nids=tuple(target_nids),
        local_loop_nids=facts.local_nids,
        matched_loop_nids=tuple(matched),
        matched_local_nids=matched_local_nids,
        duplicated_target_nids=tuple(duplicated),
        restored_loop_nids=lost_enclosing,
    )


def _prefix_plan(tree: KernelTree, block_nid: int, target_loop_nid: int) -> _PrefixPlan:
    """Require one exact loop-prefix match."""
    plan = _try_prefix_plan(tree, block_nid, target_loop_nid)
    if plan is None:
        raise TransformLegalityError(
            f"move(block={block_nid} under loop={target_loop_nid}) requires an exact compatible loop prefix"
        )
    return plan


def _assert_single_parent(tree: KernelTree) -> None:
    """Raise if a move leaves any node with multiple parents."""
    multi = {nid: list(parents) for nid, parents in tree.graph.pred.items() if len(parents) > 1}
    if multi:
        raise ValueError(f"_move left nodes with multiple parents: {multi}")


def _check_same_loop_prefix(
    ir: KernelIR, block_nid: int, target_loop_nid: int, plan: _PrefixPlan | None = None
) -> None:
    """Require the target loops to match the moved block's dependent prefix."""
    resolved_plan = plan if plan is not None else _prefix_plan(ir.tree, block_nid, target_loop_nid)
    _check_matched_tensor_partitions(ir, block_nid, resolved_plan)
    _check_no_partial_input_replicated(ir, block_nid, target_loop_nid, resolved_plan.duplicated_target_nids)
    _check_no_mutating_input_replicated(ir, block_nid, target_loop_nid, resolved_plan)
    _check_no_feedback_output_replicated(ir, block_nid, target_loop_nid, resolved_plan.duplicated_target_nids)
    _check_no_reduction_replicated(ir, block_nid, target_loop_nid, resolved_plan.duplicated_target_nids)


def _check_matched_tensor_partitions(ir: KernelIR, block_nid: int, plan: _PrefixPlan) -> None:
    """Require matched loops to select the same shared-tensor partitions."""
    moved_leaf = _dependency_leaf(ir, block_nid)
    for moved_nid, target_nid in plan.matched_loop_nids:
        if moved_nid == target_nid:
            continue
        edges = (
            *ir.dependency.graph.in_edges(moved_leaf, data=True),
            *ir.dependency.graph.out_edges(moved_leaf, data=True),
        )
        moved_var, target_var = ir.tree.loop(moved_nid).loop_var, ir.tree.loop(target_nid).loop_var
        for producer, consumer, attrs in edges:
            other_leaf = producer if consumer == moved_leaf else consumer
            if other_leaf not in _descendants(ir.tree, target_nid):
                continue
            tensor = attrs["tensor"]
            signatures = [
                frozenset(
                    tuple(affine_coefficient(lower, loop_var) for lower, _width in region.ranges)
                    for region in _leaf_operand_regions(ir.tree, leaf, tensor, rmw_only=False)
                )
                for leaf, loop_var in ((moved_leaf, moved_var), (other_leaf, target_var))
            ]
            if any(None in signature for group in signatures for signature in group) or signatures[0] != signatures[1]:
                raise TransformLegalityError(f"matched loops select different partitions of tensor {tensor!r}")


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
                loop.loop_var in expr_variables(lower) for region in producer_writes for lower, _width in region.ranges
            )
            consumer_is_invariant = bool(consumer_reads) and all(
                loop.loop_var not in expr_variables(lower)
                for region in consumer_reads
                for lower, _width in region.ranges
            )
            if producer_varies and consumer_is_invariant:
                raise TransformLegalityError(
                    f"move(block={block_nid} under loop={target_loop_nid}) replicates a full read "
                    f"of tensor {tensor!r} inside loop {loop_nid}, whose producer writes a "
                    f"different slice per iteration"
                )


def _check_no_mutating_input_replicated(ir: KernelIR, block_nid: int, target_loop_nid: int, plan: _PrefixPlan) -> None:
    """Reject repeated invariant accesses across a downstream overlapping write."""
    moved_leaf = _dependency_leaf(ir, block_nid)
    info = ir.dependency.info(moved_leaf)
    moved_reads = info.read_regions
    if plan.duplicated_target_nids and any(
        read.tensor == write.tensor and regions_overlap(ir, moved_leaf, read, moved_leaf, write)
        for read in moved_reads
        for write in info.write_regions
    ):
        raise TransformLegalityError("CodeMotion cannot replicate an operation that overwrites its own input")
    loops = [(nid, None) for nid in plan.duplicated_target_nids] + [(b, a) for a, b in plan.matched_loop_nids if a != b]
    for loop_nid, source_loop in loops:
        for access_region in (*moved_reads, *info.write_regions):
            if source_loop is not None and not _access_invariant_across(
                ir.tree, moved_leaf, ir.tree.loop(source_loop).loop_var, access_region.tensor
            ):
                continue
            for writer in ir.dependency.touches_by_tensor.get(access_region.tensor, ()):
                if loop_nid not in _ancestors(ir.tree, writer) or not ir.dependency.must_precede(moved_leaf, writer):
                    continue
                for write_region in ir.dependency.info(writer).write_regions:
                    if access_region.tensor != write_region.tensor:
                        continue
                    if regions_overlap(ir, moved_leaf, access_region, writer, write_region):
                        raise TransformLegalityError(
                            f"move(block={block_nid} under loop={target_loop_nid}) repeats "
                            f"an overlapping access to tensor {access_region.tensor!r} across loop "
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
    """Reject replicating an operation whose execution carries axis state.

    A block with an ACCUMULATION axis accumulates into a carried buffer
    (matmul → ``psum_prod``) whose init (memset) sits outside the block. The
    ``duplicated_target_nids`` are target loops that are neither existing shared
    ancestors nor matched dependent-prefix loops. Splicing under one repeats the
    whole accumulation into the same region. Parallel producers may be recomputed;
    accumulation blocks may not.
    """
    block = ir.tree.data(block_nid)
    assert isinstance(block, BlockNode)
    stateful = {iv.role for iv in block.iter_vars if iv.role != AxisRole.PARALLEL}
    if not stateful:
        return
    if duplicated_target_nids:
        replicated = [ir.tree.loop(nid).loop_var for nid in duplicated_target_nids]
        raise TransformLegalityError(
            f"move(block={block_nid} under loop={target_loop_nid}) replicates stateful "
            f"{sorted(role.value for role in stateful)} execution over loop(s) {replicated}"
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
    loop_var = ir.tree.loop(loop_nid).loop_var
    result = False
    for _producer, consumer, attrs in ir.dependency.graph.out_edges(rmw_leaf, data=True):
        if attrs.get("kind") != "RAW" or attrs.get("tensor") != tensor:
            continue
        if loop_nid not in _ancestors(ir.tree, consumer) or _access_invariant_across(
            ir.tree, consumer, loop_var, tensor
        ):
            result = True
            break
    return result


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
    moved_vars = {target: tree.loop(source).loop_var for source, target in plan.matched_loop_nids}
    if plain_writes:
        for tensor in plain_writes:
            for loop_nid in crossed_loops:
                loop = tree.loop(loop_nid)
                if not _access_invariant_across(tree, moved_leaf, moved_vars.get(loop_nid, loop.loop_var), tensor):
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
                if not _access_invariant_across(tree, moved_leaf, moved_vars.get(loop_nid, loop.loop_var), tensor):
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
            if not _access_invariant_across(
                tree, moved_leaf, moved_vars.get(loop_nid, loop.loop_var), tensor
            ) and not _rmw_value_spans_loop(ir, moved_leaf, loop_nid, tensor):
                continue
            raise TransformLegalityError(
                f"move(block={block_nid} under loop={target_loop_nid}) changes reset "
                f"frequency for tensor {tensor!r} across read-modify-write loop "
                f"{loop_nid} ({loop.loop_var!r})"
            )


def _loop_reinitializes_tensor(tree: KernelTree, loop_nid: int, tensor: str, excluded_leaves: frozenset[int]) -> bool:
    """Return whether a repeated invariant plain write resets ``tensor``."""
    loop = tree.loop(loop_nid)
    return any(
        nid not in excluded_leaves
        and isinstance((node := tree.data(nid)), ISANode)
        and tensor in _plain_written_tensors(node)
        and _access_invariant_across(tree, nid, loop.loop_var, tensor)
        for nid in _descendants(tree, loop_nid)
    )


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
            if _tensor_carried_across(ir.dependency, loop_nid, tensor):
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
        loop_var not in expr_variables(lo) for region in node.operand_bindings.values() for lo, _width in region.ranges
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
            if _tensor_carried_across(ir.dependency, loop_nid, tensor):
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


def _check_no_partial_producer_moved_into_consumer_loop(
    ir: KernelIR, block_nid: int, target_loop_nid: int, plan: _PrefixPlan
) -> None:
    """Reject a tiled producer moved before a full-domain consumer."""
    tree = ir.tree
    moved_leaf = _dependency_leaf(ir, block_nid)
    for moved_loop_nid, matched_target_nid in plan.matched_loop_nids:
        if moved_loop_nid == matched_target_nid:
            continue
        moved_var = tree.loop(moved_loop_nid).loop_var
        target_var = tree.loop(matched_target_nid).loop_var
        target_descendants = _descendants(tree, matched_target_nid)
        for _producer, consumer, attrs in ir.dependency.graph.out_edges(moved_leaf, data=True):
            tensor = attrs.get("tensor")
            if not isinstance(tensor, str) or consumer not in target_descendants:
                continue
            producer_regions = tuple(
                region for region in ir.dependency.info(moved_leaf).write_regions if region.tensor == tensor
            )
            consumer_regions = tuple(
                region for region in ir.dependency.info(consumer).read_regions if region.tensor == tensor
            )
            producer_varies = any(
                moved_var in expr_variables(lower) for region in producer_regions for lower, _width in region.ranges
            )
            consumer_invariant = bool(consumer_regions) and all(
                target_var not in expr_variables(lower)
                for region in consumer_regions
                for lower, _width in region.ranges
            )
            if producer_varies and consumer_invariant:
                raise TransformLegalityError(
                    f"move(block={block_nid} under loop={target_loop_nid}) exposes a partial "
                    f"{tensor!r} tile to full-domain consumer {consumer}"
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
    _check_move_changes_position(ir, block_nid, target_loop_nid, index)
    plan = None if ir.tree.parent(block_nid) == target_loop_nid else _prefix_plan(ir.tree, block_nid, target_loop_nid)
    if plan is not None:
        _check_access_pattern_motion(ir, block_nid, target_loop_nid, plan)
        _check_same_loop_prefix(ir, block_nid, target_loop_nid, plan)
    moved_leaf = _dependency_leaf(ir, block_nid)
    offending = ir.dependency.first_backward_edge_for_insertion(
        moved_leaf, target_loop_nid, index, moved_root_nid=block_nid
    )
    if offending is not None:
        a, b = offending
        raise TransformLegalityError(
            f"move(block={block_nid} under loop={target_loop_nid}) reorders dependency "
            f"edge {a}->{b} backward (a carried buffer's init/drain cannot enter its "
            f"reduction loop, nor a consumer precede its producer)"
        )
    if plan is not None:
        _check_move_scope_changes(ir, block_nid, target_loop_nid, plan)


def _check_access_pattern_motion(ir: KernelIR, block_nid: int, target_loop_nid: int, plan: _PrefixPlan) -> None:
    """Preserve explicit views without duplicating or restoring iterations."""
    if not subtree_has_access_patterns(ir.tree, block_nid) or ir.tree.parent(block_nid) == target_loop_nid:
        return
    if plan.duplicated_target_nids or plan.restored_loop_nids:
        raise TransformLegalityError("CodeMotion cannot duplicate or restore iterations of an explicit access pattern")


def _check_move_scope_changes(ir: KernelIR, block_nid: int, target_loop_nid: int, plan: _PrefixPlan) -> None:
    """Reject loop-scope changes not represented by dependency direction."""
    if ir.tree.parent(block_nid) == target_loop_nid:
        return
    source_path = set(_ancestors(ir.tree, block_nid))
    destination_path = {*_ancestors(ir.tree, target_loop_nid), target_loop_nid}
    exited_loops = {nid for nid in source_path - destination_path if isinstance(ir.tree.data(nid), ForNode)}
    entered_loops = {nid for nid in destination_path - source_path if isinstance(ir.tree.data(nid), ForNode)}
    if exited_loops and entered_loops:
        raise TransformLegalityError("CodeMotion must leave one loop scope before entering a different loop scope")
    leaf = _dependency_leaf(ir, block_nid)
    if any(not independent_loop_accesses(ir, nid, leaf) for nid in plan.restored_loop_nids):
        raise TransformLegalityError("CodeMotion cannot separate cross-iteration memory dependencies")
    touched = ir.dependency.info(leaf)
    for owner in source_path - destination_path:
        node = ir.tree.data(owner)
        if isinstance(node, BlockNode) and any(b.name in touched.reads | touched.writes for b in node.alloc_buffers):
            raise TransformLegalityError("CodeMotion requires BufferPlacement before crossing an allocation scope")
    _check_program_shard_scope(ir, block_nid, plan)
    if _crossed_execution_loops(ir, block_nid, target_loop_nid, plan):
        _check_no_rmw_reset_scope_change(ir, block_nid, target_loop_nid, plan)
        _check_no_consumer_hoisted_out_of_producer_loop(ir, block_nid, target_loop_nid, plan)
        _check_no_producer_moved_out_of_consumer_loop(ir, block_nid, target_loop_nid, plan)
    _check_no_partial_producer_moved_into_consumer_loop(ir, block_nid, target_loop_nid, plan)


def _check_program_shard_scope(ir: KernelIR, block_nid: int, plan: _PrefixPlan) -> None:
    """Require code motion to preserve the block's program execution counts."""
    value = ir.tree.block(ir.tree.root).annotations.get("program_shards", {})
    if not isinstance(value, dict):
        raise TransformLegalityError(f"invalid program_shards annotation: {value!r}")
    facts = _prefix_block_facts(ir.tree, block_nid)
    local_tail = plan.local_loop_nids[len(plan.matched_local_nids) :]
    old_programs = sorted(value[nid] for nid in facts.moved_nids if nid in value)
    new_programs = sorted(value[nid] for nid in (*plan.target_loop_nids, *local_tail) if nid in value)
    if old_programs != new_programs:
        raise TransformLegalityError("CodeMotion cannot change a block's program execution scope")


def _analysis_context(ir: KernelIR) -> _AnalysisContext:
    """Collect versioned tensors and pipeline loops once for one analysis."""
    topology = ir.dependency._topology()
    preorder = tuple(sorted(topology[0], key=topology[0].__getitem__))
    _ANCESTORS[ir.tree] = topology[1]
    _DESCENDANTS[ir.tree] = topology[2]
    _PREORDERS[ir.tree] = {None: preorder}
    tree = ir.tree
    leaf_count: dict[int, int] = {}
    sole_leaf: dict[int, int] = {}
    pipeline_stages: dict[int, dict[int, int]] = {}
    for nid in reversed(preorder):
        data = tree.data(nid)
        if isinstance(data, ISANode):
            leaf_count[nid] = 1
            sole_leaf[nid] = nid
            continue
        if isinstance(data, BlockNode):
            annotation = data.annotations.get("software_pipeline")
            if annotation is not None:
                pipeline_stages[annotation["loop_nid"]] = dict(zip(annotation["children"], annotation["stages"]))
        children = tree.children(nid)
        count = sum(leaf_count[child] for child in children)
        leaf_count[nid] = count
        if count == 1:
            sole_leaf[nid] = next(sole_leaf[child] for child in children if leaf_count[child])
    leaf_blocks = tuple(
        nid for nid in preorder if nid != tree.root and isinstance(tree.data(nid), BlockNode) and leaf_count[nid] == 1
    )
    _DEPENDENCY_LEAVES.setdefault(tree, {}).update((nid, sole_leaf[nid]) for nid in leaf_blocks)
    return _AnalysisContext(leaf_blocks=leaf_blocks, pipeline_stages=pipeline_stages, topology=topology)


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
) -> None:
    """Reject moves that change an active software-pipeline schedule."""
    if context is None:
        context = _analysis_context(ir)
    if not context.pipeline_stages:
        return
    old_ancestors = set(_ancestors(ir.tree, block_nid))
    new_ancestors = set(_ancestors(ir.tree, target_loop_nid)) | {target_loop_nid}
    crossed_loops = frozenset(
        pipeline_loop
        for pipeline_loop in context.pipeline_stages
        if (pipeline_loop in old_ancestors) != (pipeline_loop in new_ancestors)
    )
    if crossed_loops:
        raise TransformLegalityError(
            f"move(block={block_nid} under loop={target_loop_nid}) crosses software pipeline "
            f"loop(s) {sorted(crossed_loops)}"
        )
    for pipeline_loop, stage_by_child in context.pipeline_stages.items():
        old_child = _direct_pipeline_child(ir.tree, pipeline_loop, block_nid)
        target_child = _direct_pipeline_child(ir.tree, pipeline_loop, target_loop_nid)
        if old_child != target_child and (old_child in stage_by_child or target_child in stage_by_child):
            raise TransformLegalityError(
                f"move(block={block_nid} under loop={target_loop_nid}) changes the staged child "
                f"of software pipeline loop {pipeline_loop}"
            )


def _splice_under_target(tree: KernelTree, block_nid: int, target_loop_nid: int, index: int) -> None:
    """Detach ``block_nid`` from its parent and insert under the target loop at ``index``."""
    old_parent = tree.parent(block_nid)
    assert old_parent is not None, f"moved block {block_nid} has no parent"
    _replace_in_parent_children(tree, old_parent, [block_nid], [])
    _prune_empty_loop_chain(tree, old_parent, target_loop_nid)
    children = tree.children(target_loop_nid)
    if index < -2:
        raise ValueError(f"_splice_under_target: unsupported index {index} (use -1 append, -2 prepend, or >=0)")
    pos = len(children) if index == -1 else 0 if index == -2 else index
    new_order = children[:pos] + [block_nid] + children[pos:]
    tree.graph.remove_edges_from((target_loop_nid, child) for child in children)
    tree.graph.add_edges_from((target_loop_nid, child) for child in new_order)


def _prune_empty_loop_chain(tree: KernelTree, nid: int, stop_nid: int) -> None:
    """Remove empty old-scope loops without removing the insertion target."""
    current = nid
    while current != stop_nid and isinstance(tree.data(current), ForNode) and not tree.children(current):
        parent = tree.parent(current)
        if parent is None:
            raise AssertionError(f"empty loop {current} has no parent")
        tree.graph.remove_node(current)
        current = parent


def _check_move_changes_position(ir: KernelIR, block_nid: int, target_loop_nid: int, index: int) -> None:
    """Require an adjacent sibling swap or a scope change preserving sibling order."""
    tree = ir.tree
    if tree.parent(block_nid) != target_loop_nid:
        _check_scope_order(ir, block_nid, target_loop_nid, index)
        return
    original = tree.children(target_loop_nid)
    pos = len(original) - 1 if index == -1 else 0 if index == -2 else index
    if not 0 <= pos < len(original) or abs(pos - original.index(block_nid)) != 1:
        raise TransformLegalityError("same-scope CodeMotion must cross one adjacent sibling boundary")


def _check_scope_order(ir: KernelIR, block_nid: int, target_nid: int, index: int) -> None:
    """Reject a scope change that also crosses another instruction in source order."""
    tree = ir.tree
    children = tree.children(target_nid)
    if not 0 <= index <= len(children):
        raise TransformLegalityError("cross-scope CodeMotion requires a valid child slot")
    order = _preorder(tree)
    positions = ir.dependency._topology()[0]
    anchor = children[index] if index < len(children) else _preorder(tree, target_nid)[-1]
    boundary = positions[anchor] + (index == len(children))
    lower, upper = sorted((positions[block_nid], boundary))
    moved = _descendants(tree, block_nid)
    if any(isinstance(tree.data(nid), ISANode) and nid not in moved for nid in order[lower:upper]):
        raise TransformLegalityError("cross-scope CodeMotion must preserve sibling instruction order")


@dataclass(frozen=True)
class CodeMotionOption(TransformOption):
    """Relocate ``block_nid`` under ``target_loop_nid`` at child slot ``index``.

    One option type for both directions of motion: sinking a producer under a
    consumer's loop and lifting a consumer under a producer's loop are the same
    structural splice, distinguished only by the dependency graph — not a flag.
    The target may also be an existing operation block; entering it preserves
    sibling instruction order and leaves allocation placement to a separate action.
    For an adjacent sibling swap, ``target_loop_nid`` names the current parent block.
    Hoisting to an ancestor block recreates any still-required iteration loop.
    Moving between distinct enclosing loop nests requires a hoist followed by a sink.
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

        Buffer declarations and shapes follow separate placement and compaction actions.
        """
        self._check_legality(ir, option)
        new_ir = copy_for_rewrite(ir)
        _move(new_ir, block_nid=option.block_nid, target_loop_nid=option.target_loop_nid, index=option.index)
        new_ir.dependency = Dependency(new_ir.tree)
        return new_ir

    def analyze(self, ir: KernelIR) -> list[CodeMotionOption]:
        """Enumerate (block, target loop, index) triples passing legality."""
        options: list[CodeMotionOption] = []
        context = _analysis_context(ir)
        loops = {nid for nid in context.topology[0] if isinstance(ir.tree.data(nid), ForNode)}
        blocks = {nid for nid in context.topology[0] if isinstance(ir.tree.data(nid), BlockNode)}
        loop_ancestors = {nid: loops.intersection(parents) for nid, parents in context.topology[1].items()}
        for block_nid in context.leaf_blocks:
            prefix_facts = _prefix_block_facts(ir.tree, block_nid)
            moved_leaf = _dependency_leaf(ir, block_nid)
            consumers = ir.dependency.direct_consumers(moved_leaf)
            related_leaves = {
                moved_leaf,
                *ir.dependency.direct_producers(moved_leaf),
                *consumers,
                *(producer for consumer in consumers for producer in ir.dependency.direct_producers(consumer)),
            }
            nearby: list[int] = []
            if (parent := ir.tree.parent(block_nid)) is not None:
                siblings, sibling_indices = ir.dependency.child_order(parent)
                position = sibling_indices[block_nid]
                nearby = list(siblings[max(0, position - 1) : position] + siblings[position + 1 : position + 2])
                related_leaves.update(leaf for sibling in nearby for leaf in ir.tree.leaves(sibling))
            target_loops = (
                set().union(*(loop_ancestors[leaf] for leaf in related_leaves)) - context.topology[2][block_nid]
            )
            target_loops.update(blocks.intersection(context.topology[1][block_nid]))
            if parent is not None:
                target_loops.update(blocks.intersection(nearby))
            for target_nid in sorted(target_loops, key=context.topology[0].__getitem__):
                try:
                    self._check_static_legality(ir, block_nid, target_nid, context)
                    plan = (
                        None if parent == target_nid else _try_prefix_plan(ir.tree, block_nid, target_nid, prefix_facts)
                    )
                    if parent != target_nid:
                        if plan is None:
                            continue
                        _check_access_pattern_motion(ir, block_nid, target_nid, plan)
                except TransformLegalityError:
                    continue
                indices = self._legal_indices(ir, block_nid, target_nid)
                if not indices:
                    continue
                legal_indices: list[int] = []
                for index in indices:
                    offending = ir.dependency.first_backward_edge_for_insertion(
                        moved_leaf, target_nid, index, topology=context.topology, moved_root_nid=block_nid
                    )
                    if offending is None:
                        legal_indices.append(index)
                    elif offending[0] == moved_leaf:
                        break
                if not legal_indices:
                    continue
                if plan is not None:
                    try:
                        _check_same_loop_prefix(ir, block_nid, target_nid, plan)
                        _check_move_scope_changes(ir, block_nid, target_nid, plan)
                    except TransformLegalityError:
                        continue
                selected_indices = legal_indices if ir.tree.parent(block_nid) == target_nid else legal_indices[:1]
                options.extend(
                    CodeMotionOption(block_nid=block_nid, target_loop_nid=target_nid, index=index)
                    for index in selected_indices
                )
        return options

    def _legal_indices(self, ir: KernelIR, block_nid: int, target_nid: int) -> list[int]:
        """Slots in the insertion gap (lp, fc] among the target loop's children.

        Bounded below by the last child holding a producer of the moved block and
        above by the first child holding a consumer — symmetric in both, which is
        why one enumeration serves producer-sink and consumer-lift alike.
        """
        children, child_indices = ir.dependency.child_order(target_nid)
        moved_leaf = _dependency_leaf(ir, block_nid)
        producers = set(ir.dependency.direct_producers(moved_leaf))
        consumers = set(ir.dependency.direct_consumers(moved_leaf))
        ancestors = ir.dependency._topology()[1]
        direct = {
            leaf: child_indices[(*ancestors[leaf], leaf)[ancestors[leaf].index(target_nid) + 1]]
            for leaf in producers | consumers
            if target_nid in ancestors[leaf]
        }
        lp = max((direct[leaf] for leaf in producers if leaf in direct), default=-1)
        fc = min((direct[leaf] for leaf in consumers if leaf in direct), default=len(children))
        if ir.tree.parent(block_nid) == target_nid:
            position = child_indices[block_nid]
            return [index for index in (position - 1, position + 1) if 0 <= index < len(children) and lp < index <= fc]
        order, _ancestors, descendants = ir.dependency._topology()
        leaves = ir.dependency.blocks
        position = bisect_left(leaves, order[moved_leaf], key=order.__getitem__)
        lower = order[leaves[position - 1]] + 1 if position else 0
        upper = order[leaves[position + 1]] if position + 1 < len(leaves) else len(order)
        boundaries = [order[child] for child in children] + [order[target_nid] + len(descendants[target_nid]) + 1]
        return [index for index in range(lp + 1, fc + 1) if lower <= boundaries[index] <= upper]

    def _check_static_legality(
        self, ir: KernelIR, block_nid: int, target_loop_nid: int, context: _AnalysisContext | None
    ) -> None:
        """Check option legality that does not depend on an insertion slot."""
        if context is None:
            if target_loop_nid not in ir.tree.graph:
                raise TransformLegalityError(f"target_loop_nid={target_loop_nid} not in tree")
            if not isinstance(ir.tree.data(target_loop_nid), (ForNode, BlockNode)):
                raise TransformLegalityError("CodeMotion requires an existing loop or block scope")
            if block_nid not in ir.tree.graph:
                raise TransformLegalityError(f"block_nid={block_nid} not in tree")
            ancestor_target = target_loop_nid in _ancestors(ir.tree, block_nid)
            if isinstance(ir.tree.data(target_loop_nid), BlockNode) and not ancestor_target:
                if ir.tree.parent(target_loop_nid) != ir.tree.parent(block_nid):
                    raise TransformLegalityError("a new block scope must be a sibling of the moved block")
            if target_loop_nid == block_nid or target_loop_nid in _descendants(ir.tree, block_nid):
                raise TransformLegalityError(
                    f"target_loop_nid={target_loop_nid} is a descendant of moved block "
                    f"{block_nid} (cannot move under its own loop)"
                )
        if context is None or context.pipeline_stages:
            _check_pipeline_boundary(ir, block_nid, target_loop_nid, context)

    def _check_legality(self, ir: KernelIR, option: CodeMotionOption) -> None:
        """Structural checks (target/block in graph, target a ForNode, target not a
        descendant of the block) then span-promotion ordering. No output guard."""
        self._check_static_legality(ir, option.block_nid, option.target_loop_nid, None)
        if ir.tree.parent(option.block_nid) != option.target_loop_nid:
            legal_indices = self._legal_indices(ir, option.block_nid, option.target_loop_nid)
            if not legal_indices:
                raise TransformLegalityError("cross-scope CodeMotion has no order-preserving insertion slot")
            expected = legal_indices[0]
            if option.index != expected:
                raise TransformLegalityError(
                    f"cross-scope CodeMotion requires stable insertion index {expected}; got {option.index}"
                )
        _check_move_preserves_dependencies(ir, option.block_nid, option.target_loop_nid, option.index)


__all__ = ["_move", "_check_move_preserves_dependencies", "CodeMotion", "CodeMotionOption"]
