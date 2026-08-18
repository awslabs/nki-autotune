"""Contract-driven online-fusion transform."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import cast

from nkigym.codegen.recurrence import (
    _access_regions,
    _derive,
    _emit_copy,
    _emit_initializer,
    _Lowering,
    _Plan,
    _plan_buffers,
    _recurrence_buffers,
    _stage_region,
    _stage_regions,
)
from nkigym.ir import KernelIR
from nkigym.ir.arith.expr import Const, Expr, Mul, Var, to_affine
from nkigym.ir.recurrence import _build_match, _compatible_block, _evaluate, _Match, _Stage
from nkigym.ir.tree import PARTITION_DIM, BlockNode, Buffer, BufferRegion, ForNode, ISANode, IterVar, KernelTree
from nkigym.ops.base import AxisRole, BilinearReductionContract, ReductionContract
from nkigym.ops.store import NKIStore
from nkigym.search.state_facts import operation_facts
from nkigym.transforms.base import Transform, TransformLegalityError, TransformOption, copy_for_rewrite
from nkigym.transforms.helper.canonical_rewrite import block_chain, finalize_rewrite, owning_block
from nkigym.transforms.helper.operation_builder import NameSupply, OperationBuilder, OperationScope
from nkigym.transforms.helper.value_graph import ValueGraph, build_value_graph, contract_input_operands

_INCREMENTAL_ANNOTATION = "online_fusion_incremental"


@dataclass(frozen=True)
class _Prefix:
    """Live recurrence prefix retained for one completion action."""

    roots: tuple[int, ...]
    added_buffers: tuple[str, ...]
    carrier: int
    loop: int
    roll_forward: tuple[int, ...]
    derivation_leaves: tuple[int, ...]
    plans: tuple[_Plan, ...]
    scopes: tuple[OperationScope | None, ...]
    regions: tuple[BufferRegion, ...]


@dataclass(frozen=True)
class _Incremental:
    """Guarded metadata for extending a retained prefix one stage at a time."""

    complete: _Match
    graph: ValueGraph
    chunk_size: int
    prefix: _Prefix
    remaining: tuple[_Match, ...]
    nodes: tuple[tuple[int, BlockNode | ForNode | ISANode, tuple[int, ...]], ...]
    buffers: tuple[Buffer, ...]


def _detect_matches(ir: KernelIR, complete: bool) -> list[_Match]:
    """Detect maximal chains or their next independently useful prefix."""
    matches: list[_Match] = []
    axes = tuple(
        (
            axis
            for axis in _candidate_axes(ir)
            if all((block == ir.tree.root or _compatible_block(ir, block, axis) for block in ir.tree.blocks()))
        )
    )
    if not axes:
        return matches
    graph = build_value_graph(ir)
    for axis in axes:
        evaluation = _evaluate(ir, graph, axis)
        if len(evaluation.stages) < 2:
            continue
        maximal = _build_match(ir, graph, axis, evaluation)
        if maximal is None:
            continue
        selected = maximal
        if not complete and len(maximal.stages) > 2:
            prefixes = tuple(
                (
                    _build_match(ir, graph, axis, evaluation, stage_count=count)
                    for count in range(2, len(maximal.stages))
                )
            )
            if any((prefix is None for prefix in prefixes)):
                continue
            proven = tuple((cast(_Match, prefix) for prefix in prefixes))
            sizes = tuple(
                (
                    size
                    for size in proven[0].chunk_sizes
                    if all((size in match.chunk_sizes for match in (*proven[1:], maximal)))
                )
            )
            selected = replace(proven[0], chunk_sizes=sizes)
        if selected.chunk_sizes:
            matches.append(selected)
    return matches


def _candidate_axes(ir: KernelIR) -> tuple[str, ...]:
    """Return concrete axes used by associative reductions."""
    axes: set[str] = set()
    for nid in ir.tree.preorder():
        node = ir.tree.data(nid)
        if isinstance(node, ISANode):
            contract = node.op_cls.algebraic_contract(node.kwargs)
            if isinstance(contract, (ReductionContract, BilinearReductionContract)):
                axes.add(ir.tree.block(owning_block(ir.tree, nid)).axis_map[contract.reduction_axis])
    return tuple(sorted(axes))


def _stage_scope(ir: KernelIR, graph: ValueGraph, stage: _Stage, progress_axis: str) -> OperationScope:
    """Return mapped loop geometry with the progress axis removed."""
    block = ir.tree.block(stage.reducer_block)
    chain = block_chain(ir.tree, stage.reducer_block)
    if chain is None:
        raise ValueError(f"online reducer block {stage.reducer_block} is not a canonical chain")
    retained = [
        (iter_var, value)
        for iter_var, value in zip(block.iter_vars, block.iter_values)
        if iter_var.axis != progress_axis
    ]
    values = tuple((value for _iter_var, value in retained))
    loop_vars = {name for value in values for name in to_affine(value) if name is not None}
    loops = tuple((item for item in chain[1:-1] if isinstance(item, ForNode) and item.loop_var in loop_vars))
    axes = graph.tensor_axes[stage.state_tensor]
    scoped = replace(
        block,
        iter_vars=tuple((iter_var for iter_var, _value in retained)),
        iter_values=values,
        reads=(),
        writes=(),
        alloc_buffers=(),
        axis_map={abstract: concrete for abstract, concrete in zip(("P", "F"), axes)},
    )
    return OperationScope(scoped, loops)


def _mapped(match: _Match, ir: KernelIR) -> bool:
    """Return whether any recurrence state spans partition tiles."""
    names = [stage.state_tensor for stage in match.stages[:-1]] + [match.external_outputs[0]]
    return any((ir.buffer(name).shape[0] > PARTITION_DIM for name in names))


def _region_shape(region: BufferRegion) -> tuple[int, ...]:
    """Return the constant shape represented by a region."""
    shape: list[int] = []
    for _lower, width in region.ranges:
        if not isinstance(width, Const):
            raise ValueError("online recurrence widths must be constant")
        shape.append(width.value)
    return tuple(shape)


def _hbm_region(region: BufferRegion, tensor: str) -> BufferRegion:
    """Map an on-chip partition tile to element-addressed HBM."""
    ranges = list(region.ranges)
    lower, width = ranges[0]
    ranges[0] = (Mul(left=lower, right=Const(value=PARTITION_DIM)), width)
    return BufferRegion(tensor=tensor, ranges=tuple(ranges))


def _root_insertion(
    ir: KernelIR,
    match: _Match,
    extra_blocks: frozenset[int] = frozenset(),
    extra_leaves: frozenset[int] = frozenset(),
    retain_old: bool = False,
) -> int | None:
    """Return a root slot preserving every dependency boundary edge."""
    tree = ir.tree
    absorbed_leaves = {*match.derivation_leaves, *extra_leaves}
    absorbed_blocks = {*match.absorbed_blocks, *extra_blocks}
    roots = tree.children(tree.root)
    remaining = roots if retain_old else [root for root in roots if root not in absorbed_blocks]
    positions = {block: index for index, block in enumerate(remaining)}
    lower = 0
    upper = len(remaining)
    for producer, consumer in ir.dependency.graph.edges:
        producer_absorbed = producer in absorbed_leaves
        consumer_absorbed = consumer in absorbed_leaves
        if producer_absorbed == consumer_absorbed:
            continue
        outside = consumer if producer_absorbed else producer
        block = owning_block(tree, outside)
        if block not in positions:
            return None
        if producer_absorbed:
            upper = min(upper, positions[block])
        else:
            lower = max(lower, positions[block] + 1)
    if lower > upper:
        return None
    if retain_old:
        return lower
    first = min((roots.index(block) for block in absorbed_blocks))
    preferred = sum((roots.index(block) < first for block in remaining))
    return min(max(preferred, lower), upper)


def _set_root_children(tree: KernelTree, removed: tuple[int, ...], added: tuple[int, ...], index: int) -> None:
    """Replace arbitrary root children at one insertion slot."""
    roots = tree.children(tree.root)
    removed_set = set(removed)
    if not removed_set.issubset(roots):
        raise ValueError(f"online-fusion blocks are not root children: {removed_set - set(roots)}")
    remaining = [root for root in roots if root not in removed_set and root not in added]
    order = remaining[:index] + list(added) + remaining[index:]
    for root in roots:
        tree.graph.remove_edge(tree.root, root)
    for root in order:
        tree.graph.add_edge(tree.root, root)


def _seed_buffers(ir: KernelIR, buffers: Mapping[str, Buffer], prunable: frozenset[str]) -> None:
    """Update declarations and attach missing live buffers at the root."""
    touched = {
        region.tensor
        for nid in ir.tree.preorder()
        if isinstance(ir.tree.data(nid), ISANode)
        for region in ir.tree.isa(nid).operand_bindings.values()
    }
    missing = touched - set(buffers) - set(ir.param_buffers)
    if missing:
        raise AssertionError(f"online lowering has no buffers for {sorted(missing)}")
    declared: set[str] = set()
    for block_nid in ir.tree.blocks():
        block = ir.tree.block(block_nid)
        allocations = tuple(
            (
                buffers.get(buffer.name, buffer)
                for buffer in block.alloc_buffers
                if buffer.name not in prunable or buffer.name in touched
            )
        )
        declared.update((buffer.name for buffer in allocations))
        if allocations != block.alloc_buffers:
            ir.tree.graph.nodes[block_nid]["data"] = replace(block, alloc_buffers=allocations)
    root = ir.tree.block(ir.tree.root)
    additions = tuple(
        (
            buffer
            for name, buffer in buffers.items()
            if name in touched and name not in ir.param_buffers and name not in declared
        )
    )
    if additions:
        ir.tree.graph.nodes[ir.tree.root]["data"] = replace(root, alloc_buffers=(*root.alloc_buffers, *additions))


def _match_tensors(match: _Match, graph: ValueGraph) -> frozenset[str]:
    """Return every tensor crossing the matched derivation."""
    tensors = {stage.state_tensor for stage in match.stages}
    for nid in match.derivation_leaves:
        tensors.add(graph.outputs[nid])
        tensors.update(graph.inputs[nid].values())
    return frozenset(tensors)


def _carrier(tree: KernelTree, match: _Match, chunk_size: int, parent: int | None) -> tuple[int, int]:
    """Append one sequential carrier block and loop."""
    loop_var = f"i_{match.progress_axis}_online"
    block = BlockNode(
        iter_vars=(IterVar(axis=match.progress_axis, dom=(0, match.progress_extent), role=AxisRole.SEQUENTIAL),),
        iter_values=(Var(name=loop_var),),
        reads=(),
        writes=(),
        alloc_buffers=(),
    )
    carrier = tree.add_node(block, parent=parent)
    loop = tree.add_node(ForNode(loop_var=loop_var, extent=match.progress_extent // chunk_size), parent=carrier)
    return (carrier, loop)


def _group_scopes(
    ir: KernelIR, match: _Match, graph: ValueGraph, scopes: tuple[OperationScope, ...]
) -> tuple[OperationScope, ...] | None:
    """Return scopes when every mapped block shares an explicit outer split."""
    output_axes = graph.tensor_axes[match.external_outputs[0]]
    if not output_axes:
        return None
    axis = output_axes[0]
    values = [
        value
        for iter_var, value in zip(scopes[-1].block.iter_vars, scopes[-1].block.iter_values)
        if iter_var.axis == axis
    ]
    variables = set(to_affine(values[0])) if len(values) == 1 else set()
    variables.discard(None)
    loops = [loop for loop in scopes[-1].loops if loop.loop_var in variables]
    if len(loops) < 2:
        return None
    outer = loops[0]
    for nid in match.derivation_leaves:
        block_nid = owning_block(ir.tree, nid)
        block = ir.tree.block(block_nid)
        mapped_values = [value for iter_var, value in zip(block.iter_vars, block.iter_values) if iter_var.axis == axis]
        if not mapped_values:
            continue
        chain = block_chain(ir.tree, block_nid)
        variables = set(to_affine(mapped_values[0])) if len(mapped_values) == 1 else set()
        matching = (
            []
            if chain is None
            else [
                item
                for item in chain[1:-1]
                if isinstance(item, ForNode) and item.loop_var == outer.loop_var and item.extent == outer.extent
            ]
        )
        count = (
            0
            if chain is None
            else sum((isinstance(item, ForNode) and item.loop_var in variables for item in chain[1:-1]))
        )
        if len(mapped_values) != 1 or len(matching) != 1 or count < 2:
            return None
    return scopes


def _can_lower(ir: KernelIR, match: _Match, chunk_size: int, prefix: bool = False) -> bool:
    """Return whether ordinary-IR lowering supports one option."""
    states = {stage.state_tensor for stage in match.stages}
    valid = (
        chunk_size in match.chunk_sizes
        and bool(match.external_outputs)
        and match.external_outputs[0] == match.stages[-1].state_tensor
        and set(match.external_outputs).issubset(states)
        and match.deferred_factor is None
    )
    valid = valid and bool(match.absorbed_blocks)
    valid = valid and all((ir.tree.parent(block) == ir.tree.root for block in match.absorbed_blocks))
    graph = build_value_graph(ir)
    valid = valid and all(
        not isinstance((contract := graph.contracts[stage.reducer_leaf]), ReductionContract)
        or contract.mapped_output_operand is None
        or contract.mapped_op_cls is not None
        for stage in match.stages
    )
    if prefix:
        valid = valid and match.incremental_prefix and chunk_size < match.progress_extent
        state_buffers = [ir.buffer(stage.state_tensor) for stage in match.stages]
        valid = valid and all(
            (
                buffer.location in {"sbuf", "psum"}
                and len(buffer.shape) == 1
                and buffer.shape[0] >= PARTITION_DIM
                and buffer.shape[0] % PARTITION_DIM == 0
                for buffer in state_buffers
            )
        )
        return valid and _root_insertion(ir, match, retain_old=True) is not None
    names = [stage.state_tensor for stage in match.stages[:-1]] + [match.external_outputs[0]]
    state_buffers = [ir.buffer(name) for name in names]
    valid = valid and all(
        (
            buffer.location in {"sbuf", "psum"}
            and len(buffer.shape) in {1, 2}
            and buffer.shape[0] >= PARTITION_DIM
            and buffer.shape[0] % PARTITION_DIM == 0
            for buffer in state_buffers
        )
    )
    valid = valid and all((len(ir.buffer(stage.state_tensor).shape) == 1 for stage in match.stages[:-1]))
    if valid and _mapped(match, ir):
        scopes = tuple((_stage_scope(ir, graph, stage, match.progress_axis) for stage in match.stages))
        valid = _group_scopes(ir, match, graph, scopes) is not None
    return valid and _root_insertion(ir, match) is not None


def _preserved_chunk_size(ir: KernelIR, match: _Match) -> int | None:
    """Return the chain's existing common tile width on the progress axis."""
    sizes: set[int] = set()
    for nid in match.derivation_leaves:
        leaf = ir.tree.isa(nid)
        block = ir.tree.block(owning_block(ir.tree, nid))
        for slot, region in leaf.operand_bindings.items():
            for index, abstract in enumerate(leaf.op_cls.OPERAND_AXES[slot]):
                if block.axis_map.get(abstract) == match.progress_axis:
                    width = region.ranges[index][1]
                    if not isinstance(width, Const):
                        return None
                    sizes.add(width.value)
    size = min(sizes, default=0)
    return size if size in match.chunk_sizes and all(value % size == 0 for value in sizes) else None


def _new_context(
    ir: KernelIR,
    match: _Match,
    graph: ValueGraph,
    chunk_size: int,
    parent: int,
    buffers: dict[str, Buffer],
    names: NameSupply,
    regions: dict[str, BufferRegion],
    scopes: tuple[OperationScope | None, ...],
    progress: Expr,
    carry: str | None = None,
) -> _Lowering:
    """Construct one recurrence lowering context."""
    builder = OperationBuilder(ir.tree, parent, buffers, names, regions)
    return _Lowering(ir, match, graph, chunk_size, progress, builder, scopes, carry)


def _lower_complete(ir: KernelIR, match: _Match, chunk_size: int) -> None:
    """Lower one complete proven recurrence."""
    if not _can_lower(ir, match, chunk_size):
        raise ValueError(f"online-fusion match {match.match_id} cannot lower with chunk_size={chunk_size}")
    graph = build_value_graph(ir)
    if _mapped(match, ir):
        scopes = tuple((_stage_scope(ir, graph, stage, match.progress_axis) for stage in match.stages))
        grouped = _group_scopes(ir, match, graph, scopes)
        assert grouped is not None
        _lower_grouped(ir, match, graph, grouped, chunk_size)
    else:
        _lower_tile(ir, match, graph, chunk_size)


def _lower_tile(ir: KernelIR, match: _Match, graph: ValueGraph, chunk_size: int) -> None:
    """Lower a recurrence whose states fit one partition tile."""
    original = ir.all_buffers()
    names = NameSupply(set(original))
    plans, added = _plan_buffers(ir, match, graph, names, False, False)
    buffers = _recurrence_buffers(ir, match, original)
    buffers.update(added)
    init = OperationBuilder(ir.tree, None, buffers, names)
    roots = [_emit_initializer(init, plan.state, stage.combinator.identity) for stage, plan in zip(match.stages, plans)]
    carrier, loop = _carrier(ir.tree, match, chunk_size, None)
    context = _new_context(
        ir,
        match,
        graph,
        chunk_size,
        loop,
        buffers,
        names,
        {},
        tuple((None for _stage in match.stages)),
        Var(name=f"i_{match.progress_axis}_online"),
    )
    _derive(context, plans)
    roots.append(carrier)
    insertion = _root_insertion(ir, match)
    assert insertion is not None
    old = match.absorbed_blocks
    _set_root_children(ir.tree, old, tuple(roots), insertion)
    for block in old:
        ir.tree.graph.remove_nodes_from({block, *ir.tree.descendants(block)})
    _seed_buffers(ir, buffers, _match_tensors(match, graph))
    finalize_rewrite(ir)


def _lower_grouped(
    ir: KernelIR, match: _Match, graph: ValueGraph, scopes: tuple[OperationScope, ...], chunk_size: int
) -> None:
    """Keep one explicit mapped group on chip across the progress loop."""
    original = ir.all_buffers()
    names = NameSupply(set(original))
    plans, added = _plan_buffers(ir, match, graph, names, True, False)
    buffers = _recurrence_buffers(ir, match, original)
    buffers.update(added)
    stage_regions = tuple((_stage_region(ir, graph, stage) for stage in match.stages))
    regions = _stage_regions(plans, stage_regions)
    group = ir.tree.add_node(BlockNode(iter_vars=(), iter_values=(), reads=(), writes=(), alloc_buffers=()))
    body = ir.tree.add_node(
        BlockNode(iter_vars=(), iter_values=(), reads=(), writes=(), alloc_buffers=()), parent=group
    )
    init = OperationBuilder(ir.tree, body, buffers, names, regions)
    if chunk_size < match.progress_extent:
        for index, (stage, plan) in enumerate(zip(match.stages, plans)):
            init.scope = scopes[index]
            _emit_initializer(init, plan.state, stage.combinator.identity)
        _carrier_nid, parent = _carrier(ir.tree, match, chunk_size, body)
        progress: Expr = Var(name=f"i_{match.progress_axis}_online")
    else:
        parent = body
        progress = Const(value=0)
    context = _new_context(ir, match, graph, chunk_size, parent, buffers, names, regions, scopes, progress)
    _derive(context, plans)
    insertion = _root_insertion(ir, match)
    assert insertion is not None
    old = match.absorbed_blocks
    _set_root_children(ir.tree, old, (group,), insertion)
    for block in old:
        ir.tree.graph.remove_nodes_from({block, *ir.tree.descendants(block)})
    _seed_buffers(ir, buffers, _match_tensors(match, graph))
    finalize_rewrite(ir)


def _rewrite_reducer_as_map(ir: KernelIR, stage: _Stage, contract: ReductionContract) -> None:
    """Retain a dual-output reducer's mapped pointwise result."""
    leaf = ir.tree.isa(stage.reducer_leaf)
    mapped = contract.mapped_output_operand
    op_cls = contract.mapped_op_cls
    if mapped is None or op_cls is None:
        raise ValueError("mapped reducer has no mapped-output lowering")
    bindings = {
        slot: region
        for slot, region in leaf.operand_bindings.items()
        if slot in contract.mapped_input_operands or slot == mapped
    }
    bindings[contract.mapped_op_output_operand] = bindings.pop(mapped)
    kwargs = {name: value for name, value in leaf.kwargs.items() if name not in contract.mapped_excluded_kwargs}
    reads, writes = _access_regions(op_cls, bindings, kwargs)
    block = ir.tree.block(stage.reducer_block)
    axis = block.axis_map[contract.reduction_axis]
    iter_vars = tuple((replace(var, role=AxisRole.PARALLEL) if var.axis == axis else var for var in block.iter_vars))
    ir.tree.graph.nodes[stage.reducer_block]["data"] = replace(block, iter_vars=iter_vars, reads=reads, writes=writes)
    ir.tree.graph.nodes[stage.reducer_leaf]["data"] = ISANode(op_cls=op_cls, operand_bindings=bindings, kwargs=kwargs)


def _lower_prefix(ir: KernelIR, match: _Match, complete: _Match, chunk_size: int) -> _Prefix:
    """Emit a live two-stage recurrence while retaining the suffix."""
    if not _can_lower(ir, match, chunk_size, prefix=True):
        raise ValueError(f"online-fusion prefix {match.match_id} cannot lower")
    graph = build_value_graph(ir)
    original = ir.all_buffers()
    names = NameSupply(set(original))
    plans, added = _plan_buffers(ir, match, graph, names, False, True)
    buffers = dict(original)
    buffers.update(added)
    mapped = _mapped(match, ir)
    prefix_scopes = (
        tuple((_stage_scope(ir, graph, stage, match.progress_axis) for stage in match.stages))
        if mapped
        else tuple((None for _stage in match.stages))
    )
    complete_scopes = (
        tuple((_stage_scope(ir, graph, stage, complete.progress_axis) for stage in complete.stages))
        if mapped
        else tuple((None for _stage in complete.stages))
    )
    complete_regions = tuple((_stage_region(ir, graph, stage) for stage in complete.stages))
    regions = (
        _stage_regions(plans, tuple((_stage_region(ir, graph, stage) for stage in match.stages))) if mapped else {}
    )
    init = OperationBuilder(ir.tree, None, buffers, names, regions)
    roots: list[int] = []
    for index, (stage, plan) in enumerate(zip(match.stages, plans)):
        init.scope = prefix_scopes[index]
        roots.append(_emit_initializer(init, plan.state, stage.combinator.identity))
    carrier, loop = _carrier(ir.tree, match, chunk_size, None)
    context = _new_context(
        ir,
        match,
        graph,
        chunk_size,
        loop,
        buffers,
        names,
        regions,
        prefix_scopes,
        Var(name=f"i_{match.progress_axis}_online"),
    )
    _derive(context, plans, roll_forward=False)
    rolls: list[int] = []
    for index, plan in enumerate(plans):
        context.builder.scope = prefix_scopes[index]
        rolls.append(_emit_copy(context.builder, plan.current, plan.state))
    roots.append(carrier)
    insertion = _root_insertion(ir, match, retain_old=True)
    assert insertion is not None
    _set_root_children(ir.tree, (), tuple(roots), insertion)
    removed: list[int] = []
    for stage in match.stages:
        contract = graph.contracts[stage.reducer_leaf]
        if isinstance(contract, ReductionContract) and contract.mapped_output_operand is not None:
            _rewrite_reducer_as_map(ir, stage, contract)
        else:
            removed.append(stage.reducer_block)
    _set_root_children(ir.tree, tuple(removed), (), 0)
    for block in removed:
        ir.tree.graph.remove_nodes_from({block, *ir.tree.descendants(block)})
    _seed_buffers(ir, buffers, frozenset())
    finalize_rewrite(ir)
    added_names = tuple((name for name in buffers if name not in original and name in ir.all_buffers()))
    return _Prefix(
        tuple(roots),
        added_names,
        carrier,
        loop,
        tuple(rolls),
        match.derivation_leaves,
        plans,
        complete_scopes,
        complete_regions,
    )


def _extend_prefix(ir: KernelIR, match: _Match, graph: ValueGraph, prefix: _Prefix, chunk_size: int) -> _Prefix:
    """Append one non-final recurrence stage while retaining the suffix."""
    index = len(prefix.plans)
    if not match.incremental_prefix or len(match.stages) != index + 1:
        raise ValueError("incremental extension requires exactly one stage")
    original = ir.all_buffers()
    names = NameSupply(set(original))
    stage = match.stages[index]
    state = stage.state_tensor
    state_buffer = replace(ir.buffer(state), location="sbuf", storage_dtype="float32")
    contribution_leaf = match.stages[index].reducer_leaf
    contribution_source = ir.buffer(graph.outputs[contribution_leaf])
    raw_contribution: str | None = None
    if contribution_source.location == "psum":
        raw_contribution = names.fresh(f"{state}_online_partial")
        contribution = names.fresh(f"{state}_online_chunk")
        contribution_buffer = replace(contribution_source, name=contribution, location="sbuf", storage_dtype="float32")
    else:
        contribution = names.fresh(f"{state}_online_chunk")
        contribution_buffer = replace(contribution_source, name=contribution, storage_dtype="float32")
    current = names.fresh(f"{state}_online_current")
    plan = _Plan(state, contribution, current, raw_contribution)
    plans = (*prefix.plans, plan)
    buffers = _recurrence_buffers(ir, match, original)
    buffers[state] = state_buffer
    buffers[current] = replace(state_buffer, name=current)
    buffers[contribution] = contribution_buffer
    if raw_contribution is not None:
        buffers[raw_contribution] = replace(contribution_source, name=raw_contribution, storage_dtype="float32")
    mapped = _mapped(match, ir)
    regions = _stage_regions(plans, prefix.regions) if mapped else {}
    init = OperationBuilder(ir.tree, None, buffers, names, regions, prefix.scopes[index])
    init_root = _emit_initializer(init, state, stage.combinator.identity)
    builder = OperationBuilder(ir.tree, None, buffers, names, regions)
    context = _Lowering(
        ir, match, graph, chunk_size, Var(name=ir.tree.loop(prefix.loop).loop_var), builder, prefix.scopes
    )
    remap = {prior.state_tensor: prior_plan.current for prior, prior_plan in zip(match.stages[:index], prefix.plans)}
    selected = frozenset(match.derivation_leaves) - frozenset(prefix.derivation_leaves)
    previous = set(ir.tree.graph)
    _derive(context, plans, remap, selected, roll_forward=False)
    suffix = [
        nid
        for nid in ir.tree.graph
        if nid not in previous and ir.tree.parent(nid) is None and isinstance(ir.tree.data(nid), BlockNode)
    ]
    context.builder.scope = prefix.scopes[index]
    roll = _emit_copy(context.builder, plan.current, plan.state)
    _insert_detached_roots(ir.tree, prefix.carrier, [init_root], before=True)
    _insert_children_before(ir.tree, prefix.loop, prefix.roll_forward, [*suffix, roll])
    contract = graph.contracts[stage.reducer_leaf]
    if isinstance(contract, ReductionContract) and contract.mapped_output_operand is not None:
        _rewrite_reducer_as_map(ir, stage, contract)
    else:
        _set_root_children(ir.tree, (stage.reducer_block,), (), 0)
        ir.tree.graph.remove_nodes_from({stage.reducer_block, *ir.tree.descendants(stage.reducer_block)})
    _seed_buffers(ir, buffers, frozenset())
    finalize_rewrite(ir)
    added = (*prefix.added_buffers, *(name for name in buffers if name not in original and name in ir.all_buffers()))
    return _Prefix(
        prefix.roots,
        added,
        prefix.carrier,
        prefix.loop,
        (roll, *prefix.roll_forward),
        match.derivation_leaves,
        plans,
        prefix.scopes,
        prefix.regions,
    )


def _complete_prefix(ir: KernelIR, match: _Match, graph: ValueGraph, prefix: _Prefix, chunk_size: int) -> None:
    """Append the final stage to one retained recurrence prefix."""
    if len(match.stages) != len(prefix.plans) + 1:
        raise ValueError("incremental completion requires exactly one stage")
    all_buffers = ir.all_buffers()
    names = NameSupply(set(all_buffers))
    final_stage = match.stages[-1]
    source = ir.buffer(match.external_outputs[0])
    state = match.external_outputs[0]
    state_buffer = replace(source, name=state, location="sbuf", storage_dtype="float32")
    contribution_leaf = match.stages[-1].reducer_leaf
    contribution_source = ir.buffer(graph.outputs[contribution_leaf])
    raw_contribution: str | None = None
    if contribution_source.location == "psum":
        raw_contribution = names.fresh(f"{final_stage.state_tensor}_online_partial")
        contribution = names.fresh(f"{final_stage.state_tensor}_online_chunk")
        contribution_buffer = replace(contribution_source, name=contribution, location="sbuf", storage_dtype="float32")
    else:
        contribution = names.fresh(f"{final_stage.state_tensor}_online_chunk")
        contribution_buffer = replace(contribution_source, name=contribution, storage_dtype="float32")
    final_plan = _Plan(state, contribution, state, raw_contribution)
    plans = (*prefix.plans, final_plan)
    buffers = _recurrence_buffers(ir, match, all_buffers)
    buffers[state] = state_buffer
    buffers[contribution] = contribution_buffer
    if raw_contribution is not None:
        buffers[raw_contribution] = replace(contribution_source, name=raw_contribution, storage_dtype="float32")
    mapped = _mapped(match, ir)
    buffers[match.external_outputs[0]] = replace(source, location="sbuf", storage_dtype="float32")
    regions = _stage_regions(plans, prefix.regions) if mapped else {}
    carry: str | None = None
    init_roots: list[int] = []
    if mapped:
        output = ir.return_name
        carry = names.fresh(f"{output}_online_carry")
        buffers[carry] = replace(buffers[output], name=carry, dtype="float32", storage_dtype="float32")
        final_region = prefix.regions[-1]
        regions[carry] = _hbm_region(final_region, carry)
        zero = names.fresh(f"{match.external_outputs[0]}_online_zero")
        shape = _region_shape(final_region)
        buffers[zero] = Buffer(name=zero, shape=shape, dtype="float32", location="sbuf", storage_dtype="float32")
        regions[zero] = BufferRegion(
            tensor=zero, ranges=tuple(((Const(value=0), Const(value=extent)) for extent in shape))
        )
        init = OperationBuilder(ir.tree, None, buffers, names, regions, prefix.scopes[-1])
        init_roots.append(_emit_initializer(init, zero, 0.0))
        init_roots.append(init.append(NKIStore, {"src": regions[zero], "dst": regions[carry]}, {}))
    else:
        init = OperationBuilder(ir.tree, None, buffers, names)
        init_roots.append(_emit_initializer(init, state, final_stage.combinator.identity))
    builder = OperationBuilder(ir.tree, None, buffers, names, regions)
    context = _Lowering(
        ir, match, graph, chunk_size, Var(name=ir.tree.loop(prefix.loop).loop_var), builder, prefix.scopes, carry
    )
    remap = {stage.state_tensor: plan.current for stage, plan in zip(match.stages[:-1], prefix.plans)}
    selected = frozenset(match.derivation_leaves) - frozenset(prefix.derivation_leaves)
    previous = set(ir.tree.graph)
    _derive(context, plans, remap, selected, roll_forward=False)
    suffix = [
        nid
        for nid in ir.tree.graph
        if nid not in previous and ir.tree.parent(nid) is None and isinstance(ir.tree.data(nid), BlockNode)
    ]
    _insert_detached_roots(ir.tree, prefix.carrier, init_roots, before=True)
    _insert_children_before(ir.tree, prefix.loop, prefix.roll_forward, suffix)
    old = [block for block in match.absorbed_blocks if block in ir.tree.graph]
    roots = ir.tree.children(ir.tree.root)
    for block in old:
        if block in roots:
            ir.tree.graph.remove_edge(ir.tree.root, block)
        ir.tree.graph.remove_nodes_from({block, *ir.tree.descendants(block)})
    _seed_buffers(ir, buffers, _match_tensors(match, graph))
    finalize_rewrite(ir)


def _insert_detached_roots(tree: KernelTree, anchor: int, blocks: list[int], before: bool) -> None:
    """Attach detached roots immediately before or after one anchor."""
    if not blocks:
        return
    roots = tree.children(tree.root)
    index = roots.index(anchor) + (0 if before else 1)
    order = roots[:index] + blocks + roots[index:]
    for root in roots:
        tree.graph.remove_edge(tree.root, root)
    for root in order:
        tree.graph.add_edge(tree.root, root)


def _insert_children_before(tree: KernelTree, parent: int, anchors: tuple[int, ...], blocks: list[int]) -> None:
    """Attach detached child blocks before a contiguous anchor sequence."""
    if not blocks:
        return
    children = tree.children(parent)
    index = children.index(anchors[0])
    if tuple(children[index : index + len(anchors)]) != anchors:
        raise ValueError("online roll-forward blocks are no longer contiguous")
    order = children[:index] + blocks + children[index:]
    for child in children:
        tree.graph.remove_edge(parent, child)
    for child in order:
        tree.graph.add_edge(parent, child)


def _matching_complete(ir: KernelIR, prefix: _Match) -> _Match:
    """Return the unique maximal chain extending a prefix."""
    candidates = [
        match
        for match in _detect_matches(ir, complete=True)
        if match.progress_axis == prefix.progress_axis
        and match.stages[: len(prefix.stages)] == prefix.stages
        and match.deferred_factor is None
        and set(match.chunk_sizes) & set(prefix.chunk_sizes)
    ]
    if len(candidates) != 1:
        raise TransformLegalityError(
            f"online-fusion prefix {prefix.match_id} has {len(candidates)} complete extensions"
        )
    return candidates[0]


def _continuations(ir: KernelIR, prefix: _Match, complete: _Match) -> tuple[_Match, ...]:
    """Build every one-stage extension from the same contract evaluation."""
    graph = build_value_graph(ir)
    evaluation = _evaluate(ir, graph, prefix.progress_axis)
    result: list[_Match] = []
    for count in range(len(prefix.stages) + 1, len(complete.stages)):
        match = _build_match(ir, graph, prefix.progress_axis, evaluation, stage_count=count)
        if match is None or match.deferred_factor is not None:
            raise TransformLegalityError(f"online-fusion chain has no valid {count}-stage prefix")
        result.append(match)
    return (*result, complete)


def _capture_incremental(
    ir: KernelIR, complete: _Match, graph: ValueGraph, chunk_size: int, prefix: _Prefix, remaining: tuple[_Match, ...]
) -> _Incremental:
    """Capture a conservative structural guard for completion."""
    nodes = tuple(
        (
            (nid, copy.deepcopy(ir.tree.data(nid)), tuple(ir.tree.children(nid)))
            for nid in sorted(ir.tree.graph)
            if nid != ir.tree.root
        )
    )
    buffers = tuple((copy.deepcopy(buffer) for buffer in ir.all_buffers().values()))
    return _Incremental(complete, graph, chunk_size, prefix, remaining, nodes, buffers)


def _incremental_state(ir: KernelIR) -> _Incremental | None:
    """Return validated incremental metadata from the root."""
    value = ir.tree.block(ir.tree.root).annotations.get(_INCREMENTAL_ANNOTATION)
    if value is not None and (not isinstance(value, _Incremental)):
        raise ValueError(f"malformed {_INCREMENTAL_ANNOTATION} annotation")
    return value


def _incremental_intact(ir: KernelIR, state: _Incremental) -> bool:
    """Return whether completion can consume the retained prefix."""
    expected = {nid: (payload, children) for nid, payload, children in state.nodes}
    actual = set(ir.tree.graph) - {ir.tree.root}
    intact = actual == set(expected)
    if intact:
        intact = all(
            (
                ir.tree.data(nid) == payload and tuple(ir.tree.children(nid)) == children
                for nid, (payload, children) in expected.items()
            )
        )
    if intact:
        buffers = ir.all_buffers()
        intact = all((buffers.get(buffer.name) == buffer for buffer in state.buffers))
    return intact


@dataclass(frozen=True)
class OnlineFusionOption(TransformOption):
    """One proven recurrence fusion."""

    match_id: tuple[str, tuple[int, ...]]


class OnlineFusion(Transform[OnlineFusionOption]):
    """Rewrite one algebraically separable reduction chain into online form."""

    def analyze(self, ir: KernelIR) -> list[OnlineFusionOption]:
        """Enumerate contract-proven and lowering-supported options."""
        state = _incremental_state(ir)
        if len(ir.return_names) != 1:
            options = []
        elif state is not None:
            options = (
                [OnlineFusionOption(state.remaining[0].match_id)]
                if state.remaining and _incremental_intact(ir, state)
                else []
            )
        elif (facts := operation_facts(ir)).has_unknown_contract or not facts.has_reduction:
            options = []
        else:
            options = [
                OnlineFusionOption(match.match_id)
                for match in _detect_matches(ir, complete=False)
                if (chunk_size := _preserved_chunk_size(ir, match)) is not None
                and _can_lower(ir, match, chunk_size, prefix=match.incremental_prefix)
            ]
        return options

    def apply(self, ir: KernelIR, option: OnlineFusionOption) -> KernelIR:
        """Re-check one option, deep-copy, and lower its recurrence."""
        state = _incremental_state(ir)
        if state is not None:
            if (
                not state.remaining
                or option.match_id != state.remaining[0].match_id
                or (not _incremental_intact(ir, state))
            ):
                raise TransformLegalityError(f"illegal OnlineFusion completion option: {option}")
            result = copy_for_rewrite(ir)
            copied = _incremental_state(result)
            assert copied is not None
            next_match = copied.remaining[0]
            final = len(copied.remaining) == 1
            prefix = copied.prefix
            if final:
                _complete_prefix(result, next_match, copied.graph, copied.prefix, copied.chunk_size)
            else:
                prefix = _extend_prefix(result, next_match, copied.graph, copied.prefix, copied.chunk_size)
            root = result.tree.block(result.tree.root)
            annotations = dict(root.annotations)
            if final:
                del annotations[_INCREMENTAL_ANNOTATION]
            else:
                annotations[_INCREMENTAL_ANNOTATION] = _capture_incremental(
                    result, copied.complete, copied.graph, copied.chunk_size, prefix, copied.remaining[1:]
                )
            result.tree.graph.nodes[result.tree.root]["data"] = replace(root, annotations=annotations)
            return result
        matches = {match.match_id: match for match in _detect_matches(ir, complete=False)}
        match = matches.get(option.match_id)
        chunk_size = None if match is None else _preserved_chunk_size(ir, match)
        if (
            match is None
            or chunk_size is None
            or not _can_lower(ir, match, chunk_size, prefix=match.incremental_prefix)
        ):
            raise TransformLegalityError(f"illegal OnlineFusion option: {option}")
        result = copy_for_rewrite(ir)
        copied_matches = {candidate.match_id: candidate for candidate in _detect_matches(result, complete=False)}
        copied_match = copied_matches[option.match_id]
        copied_chunk_size = _preserved_chunk_size(result, copied_match)
        assert copied_chunk_size == chunk_size
        if copied_match.incremental_prefix:
            complete = _matching_complete(result, copied_match)
            graph = build_value_graph(result)
            remaining = _continuations(result, copied_match, complete)
            prefix = _lower_prefix(result, copied_match, complete, chunk_size)
            state = _capture_incremental(result, complete, graph, chunk_size, prefix, remaining)
            root = result.tree.block(result.tree.root)
            annotations = dict(root.annotations)
            annotations[_INCREMENTAL_ANNOTATION] = state
            result.tree.graph.nodes[result.tree.root]["data"] = replace(root, annotations=annotations)
        else:
            _lower_complete(result, copied_match, chunk_size)
        return result
