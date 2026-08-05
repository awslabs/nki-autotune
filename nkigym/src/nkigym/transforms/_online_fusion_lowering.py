"""Ordinary-IR lowering for a proven online-fusion chain."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

from nkigym.ir import KernelIR
from nkigym.ir.arith.expr import Add, Const, Expr, Mul, Var, substitute, to_affine
from nkigym.ir.tree import PARTITION_DIM, BlockNode, Buffer, BufferRegion, ForNode, ISANode, IterVar, KernelTree
from nkigym.ops.activation import NKIActivation
from nkigym.ops.activation_reduce import NKIActivationReduce
from nkigym.ops.base import AxisRole, CopyContract, ReductionContract
from nkigym.ops.load import NKILoad
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_scalar import NKITensorScalar
from nkigym.ops.tensor_scalar_reduce import NKITensorScalarReduce
from nkigym.transforms._canonical_rewrite import block_chain, finalize_rewrite, owning_block
from nkigym.transforms._online_fusion_analysis import build_value_graph
from nkigym.transforms._online_fusion_recurrence import (
    NameSupply,
    RecurrenceIR,
    RecurrenceScope,
    access_regions,
    append_additive_update,
    append_copy,
    append_corrected_update,
    append_initializer,
    append_manual_block,
    append_scaled_output,
    append_tensor_tensor,
    compile_correction,
    compile_factor,
)
from nkigym.transforms._online_fusion_types import (
    OnlineFusionMatch,
    OnlineFusionStage,
    ValueGraph,
    contract_output_operand,
    factor_states,
)


@dataclass(frozen=True)
class _StageBuffers:
    """Materialized buffers for one recurrence stage."""

    state: str
    contribution: str
    current: str


@dataclass(frozen=True)
class _ProgressTiling:
    """Per-operation tiling inside one online recurrence chunk."""

    loop_var: str | None
    tile_size: int
    trip_count: int


@dataclass(frozen=True)
class _TerminalStore:
    """Terminal copy from the matched on-chip value to the returned HBM tensor."""

    leaf: int
    block: int
    source_region: BufferRegion
    output_region: BufferRegion


@dataclass(frozen=True)
class _MappedGroupSchedule:
    """Mapped recurrence schedule selected by an explicit partition split."""

    stage_scopes: tuple[RecurrenceScope, ...]


@dataclass(frozen=True)
class OnlineFusionPrefixLowering:
    """Structure retained for an incremental online-fusion completion."""

    roots: tuple[int, ...]
    added_buffers: tuple[str, ...]
    carrier_nid: int
    loop_nid: int
    roll_forward_blocks: tuple[int, ...]
    derivation_leaves: tuple[int, ...]
    stage_buffers: tuple[tuple[str, str, str], ...]
    stage_scopes: tuple[RecurrenceScope | None, ...]
    stage_regions: tuple[BufferRegion, ...]


@dataclass
class _LoweringContext:
    """Mutable state shared by graph-lowering helpers."""

    ir: KernelIR
    match: OnlineFusionMatch
    chunk_size: int
    progress_index: Expr
    tensor_axes: Mapping[str, tuple[str, ...]]
    recurrence: RecurrenceIR
    stage_scopes: tuple[RecurrenceScope | None, ...]
    carry_tensor: str | None = None
    output_tensor: str | None = None
    outer_loop_var: str | None = None


def can_lower_online_fusion(ir: KernelIR, match: OnlineFusionMatch, chunk_size: int) -> bool:
    """Return whether the ordinary-IR lowering supports this proven match."""
    valid = chunk_size in match.chunk_sizes and len(match.external_outputs) == 1
    if valid:
        blocks = list(match.absorbed_blocks)
        valid = bool(blocks) and all(ir.tree.parent(block) == ir.tree.root for block in blocks)
    if valid:
        state_names = [stage.state_tensor for stage in match.stages[:-1]] + [match.external_outputs[0]]
        state_buffers = [ir.buffer(name) for name in state_names]
        valid = all(
            buffer.location in {"sbuf", "psum"}
            and len(buffer.shape) in {1, 2}
            and buffer.shape[0] >= 128
            and buffer.shape[0] % 128 == 0
            for buffer in state_buffers
        )
        valid = valid and all(len(ir.buffer(stage.state_tensor).shape) == 1 for stage in match.stages[:-1])
    if valid and chunk_size == match.progress_extent:
        valid = False
        if _uses_mapped_state(ir, match):
            graph = build_value_graph(ir)
            stage_scopes = tuple(_stage_scope(ir, graph, stage, match.progress_axis) for stage in match.stages)
            valid = _mapped_group_schedule(ir, match, graph, stage_scopes) is not None
    if valid and _uses_mapped_state(ir, match):
        graph = build_value_graph(ir)
        terminal = _terminal_store(ir, match, graph)
        valid = terminal is not None
        if terminal is not None:
            valid = ir.tree.parent(terminal.block) == ir.tree.root
            valid = (
                valid
                and _root_insertion_index(
                    ir, match, extra_blocks=frozenset((terminal.block,)), extra_leaves=frozenset((terminal.leaf,))
                )
                is not None
            )
    elif valid:
        valid = _root_insertion_index(ir, match) is not None
    return valid


def can_lower_online_fusion_prefix(ir: KernelIR, match: OnlineFusionMatch, chunk_size: int) -> bool:
    """Return whether a retained recurrence prefix can be emitted."""
    valid = (
        match.incremental_prefix
        and chunk_size in match.chunk_sizes
        and chunk_size < match.progress_extent
        and len(match.stages) >= 2
        and len(match.external_outputs) == 1
    )
    if valid:
        blocks = list(match.absorbed_blocks)
        valid = bool(blocks) and all(ir.tree.parent(block) == ir.tree.root for block in blocks)
    if valid:
        state_buffers = [ir.buffer(stage.state_tensor) for stage in match.stages]
        valid = all(
            buffer.location in {"sbuf", "psum"}
            and len(buffer.shape) == 1
            and buffer.shape[0] >= 128
            and buffer.shape[0] % 128 == 0
            for buffer in state_buffers
        )
    if valid:
        valid = _prefix_insertion_index(ir, match) is not None
    return valid


def lower_online_fusion(ir: KernelIR, match: OnlineFusionMatch, chunk_size: int) -> None:
    """Replace one proven chain with explicit online recurrence blocks."""
    if not can_lower_online_fusion(ir, match, chunk_size):
        raise ValueError(f"online-fusion match {match.match_id} cannot lower with chunk_size={chunk_size}")
    if _uses_mapped_state(ir, match):
        _lower_mapped_online_fusion(ir, match, chunk_size)
    else:
        _lower_tile_online_fusion(ir, match, chunk_size)


def lower_online_fusion_prefix(
    ir: KernelIR, match: OnlineFusionMatch, complete: OnlineFusionMatch, chunk_size: int
) -> OnlineFusionPrefixLowering:
    """Replace prefix reducers with a live recurrence while retaining the materialized suffix."""
    if not can_lower_online_fusion_prefix(ir, match, chunk_size):
        raise ValueError(f"online-fusion prefix {match.match_id} cannot lower with chunk_size={chunk_size}")
    graph = build_value_graph(ir)
    original_buffers = ir.all_buffers()
    names = NameSupply(set(original_buffers))
    stage_buffers, new_buffers = _plan_stage_buffers(
        ir, match, graph, names, fresh_states=False, preserve_deferred_dtype=False, separate_final_current=True
    )
    buffers = dict(original_buffers)
    buffers.update(new_buffers)
    mapped = _uses_mapped_state(ir, match)
    if mapped:
        prefix_scopes = tuple(_stage_scope(ir, graph, stage, match.progress_axis) for stage in match.stages)
        stage_scopes = tuple(_stage_scope(ir, graph, stage, complete.progress_axis) for stage in complete.stages)
        complete_regions = tuple(_stage_output_region(ir, graph, stage) for stage in complete.stages)
        output_regions = tuple(_stage_output_region(ir, graph, stage) for stage in match.stages)
        regions = _stage_regions(stage_buffers, output_regions)
    else:
        prefix_scopes = tuple(None for _stage in match.stages)
        stage_scopes = tuple(None for _stage in complete.stages)
        complete_regions = tuple(_stage_output_region(ir, graph, stage) for stage in complete.stages)
        regions = {}

    tree = ir.tree
    init_context = RecurrenceIR(tree=tree, parent=None, buffers=buffers, names=names, regions=regions)
    init_blocks: list[int] = []
    for index, (stage, plan) in enumerate(zip(match.stages, stage_buffers)):
        init_context.scope = prefix_scopes[index]
        init_blocks.append(append_initializer(init_context, plan.state, stage.combinator.identity))

    loop_var = f"i_{match.progress_axis}_online"
    carrier = BlockNode(
        iter_vars=(IterVar(axis=match.progress_axis, dom=(0, match.progress_extent), role=AxisRole.SEQUENTIAL),),
        iter_values=(Var(name=loop_var),),
        reads=(),
        writes=(),
        alloc_buffers=(),
    )
    carrier_nid = tree.add_node(carrier)
    loop_nid = tree.add_node(ForNode(loop_var=loop_var, extent=match.progress_extent // chunk_size), parent=carrier_nid)
    recurrence = RecurrenceIR(tree=tree, parent=loop_nid, buffers=buffers, names=names, regions=regions)
    context = _LoweringContext(
        ir=ir,
        match=match,
        chunk_size=chunk_size,
        progress_index=Var(name=loop_var),
        tensor_axes=graph.tensor_axes,
        recurrence=recurrence,
        stage_scopes=prefix_scopes,
    )
    _append_derivation(context, graph, stage_buffers, append_roll_forward=False)
    roll_forward_blocks: list[int] = []
    for stage_index, plan in enumerate(stage_buffers):
        recurrence.scope = prefix_scopes[stage_index]
        roll_forward_blocks.append(append_copy(recurrence, plan.current, plan.state))

    roots = (*init_blocks, carrier_nid)
    insertion_index = _prefix_insertion_index(ir, match)
    if insertion_index is None:
        raise ValueError(f"online-fusion prefix {match.match_id} has no dependency-valid insertion")
    _insert_root_blocks(tree, list(roots), insertion_index)
    removed_reducers: list[int] = []
    for stage in match.stages:
        contract = graph.contracts[stage.reducer_leaf]
        if isinstance(contract, ReductionContract) and contract.mapped_output_operand is not None:
            _rewrite_reducer_as_map(ir, stage, contract)
        else:
            removed_reducers.append(stage.reducer_block)
    _remove_root_blocks(tree, removed_reducers)
    for block_nid in removed_reducers:
        tree.graph.remove_nodes_from({block_nid, *tree.descendants(block_nid)})
    _seed_buffers(ir, buffers, frozenset())
    finalize_rewrite(ir)
    added_buffers = tuple(name for name in buffers if name not in original_buffers and name in ir.all_buffers())
    return OnlineFusionPrefixLowering(
        roots=roots,
        added_buffers=added_buffers,
        carrier_nid=carrier_nid,
        loop_nid=loop_nid,
        roll_forward_blocks=tuple(roll_forward_blocks),
        derivation_leaves=match.derivation_leaves,
        stage_buffers=tuple((plan.state, plan.contribution, plan.current) for plan in stage_buffers),
        stage_scopes=stage_scopes,
        stage_regions=complete_regions,
    )


def complete_online_fusion_prefix(
    ir: KernelIR, match: OnlineFusionMatch, graph: ValueGraph, prefix: OnlineFusionPrefixLowering, chunk_size: int
) -> None:
    """Append one final recurrence stage to a retained live prefix."""
    prefix_plans = tuple(_StageBuffers(*names) for names in prefix.stage_buffers)
    if len(match.stages) != len(prefix_plans) + 1:
        raise ValueError("incremental online fusion requires exactly one final-stage extension")
    if prefix.loop_nid not in ir.tree.graph or prefix.carrier_nid not in ir.tree.graph:
        raise ValueError("incremental online-fusion carrier is missing")

    all_buffers = ir.all_buffers()
    names = NameSupply(set(all_buffers))
    final_stage = match.stages[-1]
    state_source = ir.buffer(match.external_outputs[0])
    deferred_state = match.deferred_factor is not None and len(match.stages) - 1 == match.deferred_factor.stage
    if deferred_state:
        state_name = names.fresh(f"{final_stage.state_tensor}_online_state")
        state_buffer = replace(state_source, name=state_name, storage_dtype="float32")
    else:
        state_name = match.external_outputs[0]
        state_buffer = replace(state_source, storage_dtype="float32")
    contribution_leaf = _stage_contribution_leaf(match, graph, len(match.stages) - 1)
    contribution_source = graph.output_by_leaf[contribution_leaf]
    contribution_name = names.fresh(f"{final_stage.state_tensor}_online_chunk")
    contribution_buffer = replace(ir.buffer(contribution_source), name=contribution_name, storage_dtype="float32")
    final_plan = _StageBuffers(state=state_name, contribution=contribution_name, current=state_name)
    plans = (*prefix_plans, final_plan)

    buffers = _localized_buffers(ir, match, graph.tensor_axes, all_buffers, chunk_size)
    buffers[state_name] = state_buffer
    buffers[contribution_name] = contribution_buffer
    mapped = _uses_mapped_state(ir, match)
    regions = _stage_regions(plans, prefix.stage_regions) if mapped else {}
    carry_tensor: str | None = None
    output_tensor: str | None = None
    terminal: _TerminalStore | None = None
    init_roots: list[int] = []
    tree = ir.tree

    if mapped:
        terminal = _terminal_store(ir, match, graph)
        if terminal is None:
            raise ValueError(f"mapped online-fusion match {match.match_id} has no terminal HBM store")
        output_tensor = ir.return_name
        output_buffer = buffers[output_tensor]
        carry_tensor = names.fresh(f"{output_tensor}_online_carry")
        buffers[carry_tensor] = replace(output_buffer, name=carry_tensor, dtype="float32", storage_dtype="float32")
        final_region = prefix.stage_regions[-1]
        regions[carry_tensor] = _hbm_tile_region(final_region, carry_tensor)
        regions[output_tensor] = _hbm_tile_region(final_region, output_tensor)
        zero_tensor = names.fresh(f"{match.external_outputs[0]}_online_zero")
        zero_shape = _region_shape(final_region)
        buffers[zero_tensor] = Buffer(
            name=zero_tensor, shape=zero_shape, dtype="float32", location="sbuf", storage_dtype="float32"
        )
        regions[zero_tensor] = BufferRegion(
            tensor=zero_tensor, ranges=tuple((Const(value=0), Const(value=extent)) for extent in zero_shape)
        )
        init_context = RecurrenceIR(
            tree=tree, parent=None, buffers=buffers, names=names, regions=regions, scope=prefix.stage_scopes[-1]
        )
        init_roots.append(append_initializer(init_context, zero_tensor, 0.0))
        init_roots.append(
            append_manual_block(
                tree,
                None,
                NKIStore,
                {"src": regions[zero_tensor], "dst": regions[carry_tensor]},
                {},
                prefix.stage_scopes[-1],
            )
        )
    else:
        init_context = RecurrenceIR(tree=tree, parent=None, buffers=buffers, names=names)
        init_roots.append(append_initializer(init_context, final_plan.state, final_stage.combinator.identity))

    recurrence = RecurrenceIR(tree=tree, parent=None, buffers=buffers, names=names, regions=regions)
    context = _LoweringContext(
        ir=ir,
        match=match,
        chunk_size=chunk_size,
        progress_index=Var(name=tree.loop(prefix.loop_nid).loop_var),
        tensor_axes=graph.tensor_axes,
        recurrence=recurrence,
        stage_scopes=prefix.stage_scopes,
        carry_tensor=carry_tensor,
        output_tensor=output_tensor,
    )
    initial_remap = {stage.state_tensor: plan.current for stage, plan in zip(match.stages[:-1], prefix_plans)}
    completion_leaves = frozenset(match.derivation_leaves) - frozenset(prefix.derivation_leaves)
    nodes_before_suffix = set(tree.graph)
    _append_derivation(
        context, graph, plans, initial_remap=initial_remap, append_roll_forward=False, selected_leaves=completion_leaves
    )
    suffix_roots = _new_detached_roots(tree, nodes_before_suffix)
    epilogue_roots = _append_deferred_epilogue(context, plans)

    _insert_root_blocks_before(tree, prefix.carrier_nid, init_roots)
    _insert_child_blocks_before(tree, prefix.loop_nid, prefix.roll_forward_blocks, suffix_roots)
    _insert_root_blocks_after(tree, prefix.carrier_nid, epilogue_roots)

    old_blocks = [block for block in match.absorbed_blocks if block in tree.graph]
    if terminal is not None:
        old_blocks.append(terminal.block)
    _remove_root_blocks(tree, old_blocks)
    for block_nid in old_blocks:
        tree.graph.remove_nodes_from({block_nid, *tree.descendants(block_nid)})
    _seed_buffers(ir, buffers, _match_tensors(match, graph))
    finalize_rewrite(ir)


def _rewrite_reducer_as_map(ir: KernelIR, stage: OnlineFusionStage, contract: ReductionContract) -> None:
    """Retain the mapped output of a dual-output reducer as a pointwise block."""
    leaf = ir.tree.isa(stage.reducer_leaf)
    mapped_operand = contract.mapped_output_operand
    if mapped_operand is None:
        raise ValueError("mapped reducer rewrite requires a mapped output")
    if leaf.op_cls is NKIActivationReduce:
        op_cls = NKIActivation
        input_operands = ("data", "bias")
    elif leaf.op_cls is NKITensorScalarReduce:
        op_cls = NKITensorScalar
        input_operands = ("data", "operand0")
    else:
        raise ValueError(f"unsupported dual-output reducer {leaf.op_cls.__name__}")
    bindings = {
        slot: region
        for slot, region in leaf.operand_bindings.items()
        if slot in input_operands or slot == mapped_operand
    }
    bindings["dst"] = bindings.pop(mapped_operand)
    kwargs = {name: value for name, value in leaf.kwargs.items() if name != "reduce_op"}
    reads, writes = access_regions(op_cls, bindings, kwargs)
    block = ir.tree.block(stage.reducer_block)
    iter_vars = tuple(
        (
            replace(iter_var, role=AxisRole.PARALLEL)
            if iter_var.axis == ir.tree.block(stage.reducer_block).axis_map[contract.reduction_axis]
            else iter_var
        )
        for iter_var in block.iter_vars
    )
    ir.tree.graph.nodes[stage.reducer_block]["data"] = replace(block, iter_vars=iter_vars, reads=reads, writes=writes)
    ir.tree.graph.nodes[stage.reducer_leaf]["data"] = ISANode(op_cls=op_cls, operand_bindings=bindings, kwargs=kwargs)


def _lower_tile_online_fusion(ir: KernelIR, match: OnlineFusionMatch, chunk_size: int) -> None:
    """Lower a chain whose complete state fits in one partition tile."""
    graph = build_value_graph(ir)
    all_buffers = ir.all_buffers()
    names = NameSupply(set(all_buffers))
    stage_buffers, new_buffers = _plan_stage_buffers(
        ir, match, graph, names, fresh_states=False, preserve_deferred_dtype=False, separate_final_current=False
    )
    buffers = _localized_buffers(ir, match, graph.tensor_axes, all_buffers, chunk_size)
    buffers.update(new_buffers)
    insertion_index = _root_insertion_index(ir, match)
    if insertion_index is None:
        raise AssertionError(f"online-fusion match {match.match_id} has no dependency-valid root insertion")
    loop_var = f"i_{match.progress_axis}_online"
    tree = ir.tree

    init_context = RecurrenceIR(tree=tree, parent=None, buffers=buffers, names=names)
    init_blocks = []
    for stage, plan in zip(match.stages, stage_buffers):
        init_blocks.append(append_initializer(init_context, plan.state, stage.combinator.identity))
    carrier = BlockNode(
        iter_vars=(IterVar(axis=match.progress_axis, dom=(0, match.progress_extent), role=AxisRole.SEQUENTIAL),),
        iter_values=(Var(name=loop_var),),
        reads=(),
        writes=(),
        alloc_buffers=(),
    )
    carrier_nid = tree.add_node(carrier)
    loop_nid = tree.add_node(ForNode(loop_var=loop_var, extent=match.progress_extent // chunk_size), parent=carrier_nid)
    recurrence = RecurrenceIR(tree=tree, parent=loop_nid, buffers=buffers, names=names)
    context = _LoweringContext(
        ir=ir,
        match=match,
        chunk_size=chunk_size,
        progress_index=Var(name=loop_var),
        tensor_axes=graph.tensor_axes,
        recurrence=recurrence,
        stage_scopes=tuple(None for _stage in match.stages),
    )
    _append_derivation(context, graph, stage_buffers)
    epilogue_blocks = _append_deferred_epilogue(context, stage_buffers)

    old_blocks = list(match.absorbed_blocks)
    _replace_root_blocks(tree, old_blocks, [*init_blocks, carrier_nid, *epilogue_blocks], insertion_index)
    for block_nid in old_blocks:
        tree.graph.remove_nodes_from({block_nid, *tree.descendants(block_nid)})
    _seed_buffers(ir, buffers, _match_tensors(match, graph))
    finalize_rewrite(ir)


def _lower_mapped_online_fusion(ir: KernelIR, match: OnlineFusionMatch, chunk_size: int) -> None:
    """Lower a recurrence mapped over multiple partition tiles."""
    graph = build_value_graph(ir)
    stage_scopes = tuple(_stage_scope(ir, graph, stage, match.progress_axis) for stage in match.stages)
    group_schedule = _mapped_group_schedule(ir, match, graph, stage_scopes)
    if group_schedule is None:
        _lower_hbm_carried_online_fusion(ir, match, graph, stage_scopes, chunk_size)
    else:
        _lower_grouped_online_fusion(ir, match, graph, group_schedule, chunk_size)


def _lower_hbm_carried_online_fusion(
    ir: KernelIR,
    match: OnlineFusionMatch,
    graph: ValueGraph,
    stage_scopes: tuple[RecurrenceScope, ...],
    chunk_size: int,
) -> None:
    """Lower mapped state through an FP32 HBM carry."""
    terminal = _terminal_store(ir, match, graph)
    if terminal is None:
        raise AssertionError(f"mapped online-fusion match {match.match_id} has no terminal HBM store")
    all_buffers = ir.all_buffers()
    names = NameSupply(set(all_buffers))
    stage_buffers, new_buffers = _plan_stage_buffers(
        ir, match, graph, names, fresh_states=False, preserve_deferred_dtype=False, separate_final_current=False
    )
    buffers = _localized_buffers(ir, match, graph.tensor_axes, all_buffers, chunk_size)
    buffers.update(new_buffers)
    output_tensor = ir.return_name
    output_buffer = buffers[output_tensor]
    carry_tensor = names.fresh(f"{output_tensor}_online_carry")
    buffers[carry_tensor] = replace(output_buffer, name=carry_tensor, dtype="float32", storage_dtype="float32")
    stage_regions = tuple(_stage_output_region(ir, graph, stage) for stage in match.stages)
    regions = _stage_regions(stage_buffers, stage_regions)
    regions[carry_tensor] = _hbm_tile_region(stage_regions[-1], carry_tensor)
    regions[output_tensor] = _hbm_tile_region(stage_regions[-1], output_tensor)

    tree = ir.tree
    init_context = RecurrenceIR(tree=tree, parent=None, buffers=buffers, names=names, regions=regions)
    init_blocks: list[int] = []
    for index, (stage, plan) in enumerate(zip(match.stages[:-1], stage_buffers[:-1])):
        init_context.scope = stage_scopes[index]
        init_blocks.append(append_initializer(init_context, plan.state, stage.combinator.identity))

    final_region = stage_regions[-1]
    zero_tensor = names.fresh(f"{match.external_outputs[0]}_online_zero")
    zero_shape = _region_shape(final_region)
    buffers[zero_tensor] = Buffer(
        name=zero_tensor, shape=zero_shape, dtype="float32", location="sbuf", storage_dtype="float32"
    )
    regions[zero_tensor] = BufferRegion(
        tensor=zero_tensor, ranges=tuple((Const(value=0), Const(value=extent)) for extent in zero_shape)
    )
    init_context.scope = stage_scopes[-1]
    init_blocks.append(append_initializer(init_context, zero_tensor, 0.0))
    init_blocks.append(
        append_manual_block(
            tree, None, NKIStore, {"src": regions[zero_tensor], "dst": regions[carry_tensor]}, {}, stage_scopes[-1]
        )
    )

    insertion_index = _root_insertion_index(
        ir, match, extra_blocks=frozenset((terminal.block,)), extra_leaves=frozenset((terminal.leaf,))
    )
    if insertion_index is None:
        raise AssertionError(f"mapped online-fusion match {match.match_id} has no dependency-valid insertion")
    loop_var = f"i_{match.progress_axis}_online"
    carrier = BlockNode(
        iter_vars=(IterVar(axis=match.progress_axis, dom=(0, match.progress_extent), role=AxisRole.SEQUENTIAL),),
        iter_values=(Var(name=loop_var),),
        reads=(),
        writes=(),
        alloc_buffers=(),
    )
    carrier_nid = tree.add_node(carrier)
    loop_nid = tree.add_node(ForNode(loop_var=loop_var, extent=match.progress_extent // chunk_size), parent=carrier_nid)
    recurrence = RecurrenceIR(tree=tree, parent=loop_nid, buffers=buffers, names=names, regions=regions)
    context = _LoweringContext(
        ir=ir,
        match=match,
        chunk_size=chunk_size,
        progress_index=Var(name=loop_var),
        tensor_axes=graph.tensor_axes,
        recurrence=recurrence,
        stage_scopes=stage_scopes,
        carry_tensor=carry_tensor,
        output_tensor=output_tensor,
    )
    _append_derivation(context, graph, stage_buffers)
    epilogue_blocks = _append_deferred_epilogue(context, stage_buffers)

    old_blocks = [*match.absorbed_blocks, terminal.block]
    _replace_root_blocks(tree, old_blocks, [*init_blocks, carrier_nid, *epilogue_blocks], insertion_index)
    for block_nid in old_blocks:
        tree.graph.remove_nodes_from({block_nid, *tree.descendants(block_nid)})
    _seed_buffers(ir, buffers, _match_tensors(match, graph))
    finalize_rewrite(ir)


def _lower_grouped_online_fusion(
    ir: KernelIR, match: OnlineFusionMatch, graph: ValueGraph, schedule: _MappedGroupSchedule, chunk_size: int
) -> None:
    """Keep one explicitly split partition group on chip across the progress loop."""
    terminal = _terminal_store(ir, match, graph)
    if terminal is None:
        raise AssertionError(f"mapped online-fusion match {match.match_id} has no terminal HBM store")
    all_buffers = ir.all_buffers()
    names = NameSupply(set(all_buffers))
    stage_buffers, new_buffers = _plan_stage_buffers(
        ir, match, graph, names, fresh_states=False, preserve_deferred_dtype=True, separate_final_current=False
    )
    buffers = _localized_buffers(ir, match, graph.tensor_axes, all_buffers, chunk_size)
    buffers.update(new_buffers)
    output_tensor = ir.return_name
    stage_regions = tuple(_stage_output_region(ir, graph, stage) for stage in match.stages)
    regions = _stage_regions(stage_buffers, stage_regions)
    regions[output_tensor] = _hbm_tile_region(stage_regions[-1], output_tensor)

    tree = ir.tree
    group_root = tree.add_node(BlockNode(iter_vars=(), iter_values=(), reads=(), writes=(), alloc_buffers=()))
    group_body = tree.add_node(
        BlockNode(iter_vars=(), iter_values=(), reads=(), writes=(), alloc_buffers=()), parent=group_root
    )

    init_context = RecurrenceIR(tree=tree, parent=group_body, buffers=buffers, names=names, regions=regions)
    if chunk_size < match.progress_extent:
        for index, (stage, plan) in enumerate(zip(match.stages, stage_buffers)):
            init_context.scope = schedule.stage_scopes[index]
            append_initializer(init_context, plan.state, stage.combinator.identity)

    loop_var = f"i_{match.progress_axis}_online"
    carrier = BlockNode(
        iter_vars=(IterVar(axis=match.progress_axis, dom=(0, match.progress_extent), role=AxisRole.SEQUENTIAL),),
        iter_values=(Var(name=loop_var),),
        reads=(),
        writes=(),
        alloc_buffers=(),
    )
    recurrence_parent = group_body
    progress_index: Expr = Const(value=0)
    if chunk_size < match.progress_extent:
        carrier_nid = tree.add_node(carrier, parent=group_body)
        recurrence_parent = tree.add_node(
            ForNode(loop_var=loop_var, extent=match.progress_extent // chunk_size), parent=carrier_nid
        )
        progress_index = Var(name=loop_var)
    recurrence = RecurrenceIR(tree=tree, parent=recurrence_parent, buffers=buffers, names=names, regions=regions)
    context = _LoweringContext(
        ir=ir,
        match=match,
        chunk_size=chunk_size,
        progress_index=progress_index,
        tensor_axes=graph.tensor_axes,
        recurrence=recurrence,
        stage_scopes=schedule.stage_scopes,
        output_tensor=output_tensor,
    )
    _append_derivation(context, graph, stage_buffers)
    final_scope = schedule.stage_scopes[-1]
    store_source = stage_buffers[-1].current
    if match.deferred_factor is not None:
        store_source = _append_grouped_deferred_epilogue(context, stage_buffers, group_body)
    append_manual_block(
        tree, group_body, NKIStore, {"src": regions[store_source], "dst": regions[output_tensor]}, {}, final_scope
    )

    insertion_index = _root_insertion_index(
        ir, match, extra_blocks=frozenset((terminal.block,)), extra_leaves=frozenset((terminal.leaf,))
    )
    if insertion_index is None:
        raise AssertionError(f"mapped online-fusion match {match.match_id} has no dependency-valid insertion")
    old_blocks = [*match.absorbed_blocks, terminal.block]
    _replace_root_blocks(tree, old_blocks, [group_root], insertion_index)
    for block_nid in old_blocks:
        tree.graph.remove_nodes_from({block_nid, *tree.descendants(block_nid)})
    _seed_buffers(ir, buffers, _match_tensors(match, graph))
    finalize_rewrite(ir)


def _mapped_group_schedule(
    ir: KernelIR, match: OnlineFusionMatch, graph: ValueGraph, stage_scopes: tuple[RecurrenceScope, ...]
) -> _MappedGroupSchedule | None:
    """Return an on-chip grouping schedule when every mapped block has the same outer split."""
    schedule: _MappedGroupSchedule | None = None
    output_axes = graph.tensor_axes[match.external_outputs[0]]
    if output_axes:
        mapped_axis = output_axes[0]
        final_scope = stage_scopes[-1]
        axis_loops = _scope_axis_loops(final_scope, mapped_axis)
        if len(axis_loops) >= 2:
            outer_loop = axis_loops[0]
            if _derivation_uses_group_loop(ir, match, mapped_axis, outer_loop):
                schedule = _MappedGroupSchedule(stage_scopes=stage_scopes)
    return schedule


def _append_grouped_deferred_epilogue(context: _LoweringContext, plans: tuple[_StageBuffers, ...], parent: int) -> str:
    """Apply one deferred factor after a grouped on-chip recurrence."""
    deferred = context.match.deferred_factor
    if deferred is None:
        raise ValueError("grouped deferred epilogue requires a deferred factor")
    normalized = context.match.external_outputs[0]
    final_region = context.recurrence.region(plans[deferred.stage].current)
    context.recurrence.regions[normalized] = replace(final_region, tensor=normalized)
    states = {index: plan.state for index, plan in enumerate(plans[: deferred.stage])}
    state_indices = sorted(factor_states(deferred.factor))
    if not state_indices:
        raise ValueError("grouped deferred factor does not reference an online state")
    factor_scope = context.stage_scopes[state_indices[-1]]
    output_scope = context.stage_scopes[deferred.stage]
    if factor_scope is None or output_scope is None:
        raise ValueError("grouped deferred factor has no mapped recurrence scope")
    epilogue = RecurrenceIR(
        tree=context.recurrence.tree,
        parent=parent,
        buffers=context.recurrence.buffers,
        names=context.recurrence.names,
        regions=context.recurrence.regions,
        scope=factor_scope,
    )
    factor = compile_factor(epilogue, deferred.factor, states, "deferred_factor")
    epilogue.scope = output_scope
    append_scaled_output(epilogue, plans[deferred.stage].current, factor, normalized)
    return normalized


def _scope_axis_loops(scope: RecurrenceScope, axis: str) -> tuple[ForNode, ...]:
    """Return one scope's loops that form the requested concrete axis."""
    values = [value for iter_var, value in zip(scope.block.iter_vars, scope.block.iter_values) if iter_var.axis == axis]
    loop_vars = set(to_affine(values[0])) if len(values) == 1 else set()
    loop_vars.discard(None)
    return tuple(loop for loop in scope.loops if loop.loop_var in loop_vars)


def _derivation_uses_group_loop(ir: KernelIR, match: OnlineFusionMatch, mapped_axis: str, outer_loop: ForNode) -> bool:
    """Return whether every derivation block mapped on this axis has the explicit group loop."""
    valid = True
    for leaf_nid in match.derivation_leaves:
        block_nid = owning_block(ir.tree, leaf_nid)
        block = ir.tree.block(block_nid)
        values = [value for iter_var, value in zip(block.iter_vars, block.iter_values) if iter_var.axis == mapped_axis]
        if not values:
            continue
        chain = block_chain(ir.tree, block_nid)
        loop_vars = set(to_affine(values[0])) if len(values) == 1 else set()
        matching = (
            []
            if chain is None
            else [
                payload
                for payload in chain[1:-1]
                if isinstance(payload, ForNode)
                and payload.loop_var == outer_loop.loop_var
                and payload.extent == outer_loop.extent
            ]
        )
        axis_loop_count = (
            0
            if chain is None
            else sum(isinstance(payload, ForNode) and payload.loop_var in loop_vars for payload in chain[1:-1])
        )
        if len(values) != 1 or len(matching) != 1 or axis_loop_count < 2:
            valid = False
            break
    return valid


def _uses_mapped_state(ir: KernelIR, match: OnlineFusionMatch) -> bool:
    """Return whether any recurrence state spans multiple partition tiles."""
    state_names = [stage.state_tensor for stage in match.stages[:-1]] + [match.external_outputs[0]]
    return any(ir.buffer(name).shape[0] > 128 for name in state_names)


def _terminal_store(ir: KernelIR, match: OnlineFusionMatch, graph: ValueGraph) -> _TerminalStore | None:
    """Return the unique copy from the matched output to the kernel return tensor."""
    output_leaves = [
        leaf for leaf in match.derivation_leaves if graph.output_by_leaf[leaf] == match.external_outputs[0]
    ]
    result: _TerminalStore | None = None
    if len(output_leaves) == 1:
        successors = [leaf for leaf in graph.successors[output_leaves[0]] if leaf not in match.derivation_leaves]
        if len(successors) == 1:
            leaf_nid = successors[0]
            leaf = ir.tree.isa(leaf_nid)
            contract = graph.contracts[leaf_nid]
            if isinstance(contract, CopyContract) and leaf.op_cls is NKIStore:
                source_region = leaf.operand_bindings[contract.input_operand]
                output_region = leaf.operand_bindings[contract.output_operand]
                output_buffer = ir.buffer(output_region.tensor)
                if output_region.tensor == ir.return_name and output_buffer.location == "shared_hbm":
                    result = _TerminalStore(
                        leaf=leaf_nid,
                        block=owning_block(ir.tree, leaf_nid),
                        source_region=source_region,
                        output_region=output_region,
                    )
    return result


def _root_insertion_index(
    ir: KernelIR,
    match: OnlineFusionMatch,
    extra_blocks: frozenset[int] = frozenset(),
    extra_leaves: frozenset[int] = frozenset(),
) -> int | None:
    """Return a root slot preserving every dependency crossing the rewrite."""
    tree = ir.tree
    absorbed_leaves = {*match.derivation_leaves, *extra_leaves}
    absorbed_blocks = {*match.absorbed_blocks, *extra_blocks}
    root_children = tree.children(tree.root)
    remaining = [child for child in root_children if child not in absorbed_blocks]
    positions = {block: index for index, block in enumerate(remaining)}
    must_precede: set[int] = set()
    must_follow: set[int] = set()
    valid = True
    for producer, consumer in ir.dependency.graph.edges:
        producer_absorbed = producer in absorbed_leaves
        consumer_absorbed = consumer in absorbed_leaves
        if producer_absorbed == consumer_absorbed:
            continue
        outside_leaf = consumer if producer_absorbed else producer
        outside_block = owning_block(tree, outside_leaf)
        if outside_block not in positions:
            valid = False
            break
        if producer_absorbed:
            must_follow.add(outside_block)
        else:
            must_precede.add(outside_block)
    result: int | None = None
    if valid:
        lower = max((positions[block] + 1 for block in must_precede), default=0)
        upper = min((positions[block] for block in must_follow), default=len(remaining))
        if lower <= upper:
            first_absorbed = min(root_children.index(block) for block in absorbed_blocks)
            preferred = sum(1 for block in remaining if root_children.index(block) < first_absorbed)
            result = min(max(preferred, lower), upper)
    return result


def _prefix_insertion_index(ir: KernelIR, match: OnlineFusionMatch) -> int | None:
    """Return a root slot where the live prefix precedes every retained consumer."""
    tree = ir.tree
    root_children = tree.children(tree.root)
    positions = {block: index for index, block in enumerate(root_children)}
    absorbed_leaves = set(match.derivation_leaves)
    lower = 0
    upper = len(root_children)
    valid = True
    for producer, consumer in ir.dependency.graph.edges:
        producer_absorbed = producer in absorbed_leaves
        consumer_absorbed = consumer in absorbed_leaves
        if producer_absorbed == consumer_absorbed:
            continue
        outside_leaf = consumer if producer_absorbed else producer
        outside_block = owning_block(tree, outside_leaf)
        if outside_block not in positions:
            valid = False
            break
        if producer_absorbed:
            upper = min(upper, positions[outside_block])
        else:
            lower = max(lower, positions[outside_block] + 1)
    result: int | None = None
    if valid and lower <= upper:
        result = lower
    return result


def _replace_root_blocks(tree: KernelTree, old_blocks: list[int], new_blocks: list[int], insertion_index: int) -> None:
    """Remove arbitrary root blocks and insert their replacement at one slot."""
    root = tree.root
    old = set(old_blocks)
    new = set(new_blocks)
    siblings = tree.children(root)
    if not old.issubset(siblings):
        raise ValueError(f"online-fusion blocks are not all root children: {old - set(siblings)}")
    remaining = [child for child in siblings if child not in old and child not in new]
    if insertion_index < 0 or insertion_index > len(remaining):
        raise ValueError(f"invalid online-fusion root insertion index {insertion_index}")
    order = remaining[:insertion_index] + new_blocks + remaining[insertion_index:]
    for child in siblings:
        tree.graph.remove_edge(root, child)
    for child in order:
        tree.graph.add_edge(root, child)


def _insert_root_blocks(tree: KernelTree, new_blocks: list[int], insertion_index: int) -> None:
    """Insert detached blocks into the root without removing existing work."""
    root = tree.root
    siblings = tree.children(root)
    if insertion_index < 0 or insertion_index > len(siblings):
        raise ValueError(f"invalid online-fusion root insertion index {insertion_index}")
    new = set(new_blocks)
    if any(block in siblings for block in new):
        raise ValueError(f"online-fusion root insertion contains attached blocks {new & set(siblings)}")
    order = siblings[:insertion_index] + new_blocks + siblings[insertion_index:]
    for child in siblings:
        tree.graph.remove_edge(root, child)
    for child in order:
        tree.graph.add_edge(root, child)


def _remove_root_blocks(tree: KernelTree, removed_blocks: list[int]) -> None:
    """Detach selected root blocks without changing remaining sibling order."""
    removed = set(removed_blocks)
    siblings = tree.children(tree.root)
    if not removed.issubset(siblings):
        raise ValueError(f"online-fusion blocks are not all root children: {removed - set(siblings)}")
    for block_nid in removed_blocks:
        tree.graph.remove_edge(tree.root, block_nid)


def _new_detached_roots(tree: KernelTree, previous_nodes: set[int]) -> list[int]:
    """Return newly added block roots that have not yet been attached."""
    return [
        nid
        for nid in tree.graph.nodes
        if nid not in previous_nodes and tree.parent(nid) is None and isinstance(tree.data(nid), BlockNode)
    ]


def _insert_root_blocks_before(tree: KernelTree, anchor: int, blocks: list[int]) -> None:
    """Attach detached root blocks immediately before one existing root child."""
    if not blocks:
        return
    siblings = tree.children(tree.root)
    index = siblings.index(anchor)
    _insert_root_blocks(tree, blocks, index)


def _insert_root_blocks_after(tree: KernelTree, anchor: int, blocks: list[int]) -> None:
    """Attach detached root blocks immediately after one existing root child."""
    if not blocks:
        return
    siblings = tree.children(tree.root)
    index = siblings.index(anchor) + 1
    _insert_root_blocks(tree, blocks, index)


def _insert_child_blocks_before(tree: KernelTree, parent: int, anchors: tuple[int, ...], blocks: list[int]) -> None:
    """Attach detached blocks before one contiguous child suffix."""
    if not blocks:
        return
    if not anchors:
        raise ValueError("online-fusion completion has no state roll-forward anchor")
    siblings = tree.children(parent)
    index = siblings.index(anchors[0])
    if tuple(siblings[index : index + len(anchors)]) != anchors:
        raise ValueError("online-fusion state roll-forward blocks are no longer contiguous")
    order = siblings[:index] + blocks + siblings[index:]
    for child in siblings:
        tree.graph.remove_edge(parent, child)
    for child in order:
        tree.graph.add_edge(parent, child)


def _append_derivation(
    context: _LoweringContext,
    graph: ValueGraph,
    stage_buffers: tuple[_StageBuffers, ...],
    start_index: int = 0,
    initial_remap: Mapping[str, str] | None = None,
    append_roll_forward: bool = True,
    selected_leaves: frozenset[int] | None = None,
) -> None:
    """Clone the per-chunk derivation and insert each recurrence update."""
    tensor_remap = dict(initial_remap or {})
    contribution_leaves = [
        _stage_contribution_leaf(context.match, graph, index) for index in range(len(context.match.stages))
    ]
    stage_by_leaf = {leaf: index for index, leaf in enumerate(contribution_leaves)}
    stage_output_plan = {
        graph.output_by_leaf[leaf]: plan.contribution for leaf, plan in zip(contribution_leaves, stage_buffers)
    }
    derivation_leaves = context.match.derivation_leaves[start_index:]
    if selected_leaves is not None:
        derivation_leaves = tuple(leaf for leaf in derivation_leaves if leaf in selected_leaves)
    for leaf_nid in derivation_leaves:
        deferred = context.match.deferred_factor
        if deferred is not None and leaf_nid == deferred.producer_leaf:
            continue
        if deferred is not None and leaf_nid == deferred.bypass_leaf:
            output = graph.output_by_leaf[leaf_nid]
            source = graph.input_tensors_by_leaf[leaf_nid][deferred.passthrough_operand]
            tensor_remap[output] = tensor_remap.get(source, source)
            continue
        output = graph.output_by_leaf[leaf_nid]
        _append_cloned_block(context, graph, leaf_nid, tensor_remap, stage_output_plan.get(output))
        stage_index = stage_by_leaf.get(leaf_nid)
        if stage_index is not None:
            _append_stage_update(context, stage_index, stage_buffers)
            stage = context.match.stages[stage_index]
            tensor_remap[stage.state_tensor] = stage_buffers[stage_index].current
    if append_roll_forward:
        for stage_index, plan in enumerate(stage_buffers[:-1]):
            context.recurrence.scope = context.stage_scopes[stage_index]
            append_copy(context.recurrence, plan.current, plan.state)


def _plan_stage_buffers(
    ir: KernelIR,
    match: OnlineFusionMatch,
    graph: ValueGraph,
    names: NameSupply,
    fresh_states: bool,
    preserve_deferred_dtype: bool,
    separate_final_current: bool,
) -> tuple[tuple[_StageBuffers, ...], dict[str, Buffer]]:
    """Choose state, contribution, and current-state buffers per stage."""
    plans: list[_StageBuffers] = []
    buffers: dict[str, Buffer] = {}
    last = len(match.stages) - 1
    for index, stage in enumerate(match.stages):
        deferred_state = match.deferred_factor is not None and index == match.deferred_factor.stage
        if fresh_states:
            state_name = names.fresh(f"{stage.state_tensor}_online_state")
            state_buffer = replace(ir.buffer(stage.state_tensor), name=state_name, storage_dtype="float32")
        elif deferred_state:
            state_name = names.fresh(f"{stage.state_tensor}_online_state")
            source = ir.buffer(match.external_outputs[0])
            storage_dtype = source.storage_dtype if preserve_deferred_dtype else "float32"
            state_buffer = replace(source, name=state_name, storage_dtype=storage_dtype)
        else:
            state_name = match.external_outputs[0] if index == last else stage.state_tensor
            state_buffer = replace(ir.buffer(state_name), storage_dtype="float32")
        buffers[state_name] = state_buffer
        contribution_leaf = _stage_contribution_leaf(match, graph, index)
        contribution_source = graph.output_by_leaf[contribution_leaf]
        contribution_name = contribution_source
        if index != last or contribution_name == state_name or contribution_leaf != stage.reducer_leaf:
            contribution_name = names.fresh(f"{stage.state_tensor}_online_chunk")
            source = ir.buffer(contribution_source)
            storage_dtype = source.storage_dtype if deferred_state and preserve_deferred_dtype else "float32"
            buffers[contribution_name] = replace(source, name=contribution_name, storage_dtype=storage_dtype)
        if index == last and not separate_final_current:
            current_name = state_name
        else:
            current_name = names.fresh(f"{stage.state_tensor}_online_current")
            buffers[current_name] = replace(state_buffer, name=current_name)
        plans.append(_StageBuffers(state=state_name, contribution=contribution_name, current=current_name))
    return tuple(plans), buffers


def _stage_contribution_leaf(match: OnlineFusionMatch, graph: ValueGraph, stage_index: int) -> int:
    """Return the leaf whose output is combined into one recurrence stage."""
    stage = match.stages[stage_index]
    result = stage.reducer_leaf
    if stage_index == len(match.stages) - 1:
        reducer_index = match.derivation_leaves.index(stage.reducer_leaf)
        suffix = match.derivation_leaves[reducer_index + 1 :]
        if len(suffix) == 1:
            candidate = suffix[0]
            contract = graph.contracts[candidate]
            inputs = graph.input_tensors_by_leaf[candidate]
            if (
                isinstance(contract, CopyContract)
                and inputs.get(contract.input_operand) == stage.state_tensor
                and graph.output_by_leaf[candidate] == match.external_outputs[0]
            ):
                result = candidate
    return result


def _append_deferred_epilogue(context: _LoweringContext, plans: tuple[_StageBuffers, ...]) -> list[int]:
    """Apply a deferred final-stage factor once after the sequential loop."""
    deferred = context.match.deferred_factor
    epilogue_roots: list[int] = []
    if deferred is not None:
        states = {index: plan.state for index, plan in enumerate(plans[: deferred.stage])}
        if context.carry_tensor is None:
            carrier_nid = context.recurrence.tree.add_node(
                BlockNode(iter_vars=(), iter_values=(), reads=(), writes=(), alloc_buffers=())
            )
            epilogue = RecurrenceIR(
                tree=context.recurrence.tree,
                parent=carrier_nid,
                buffers=context.recurrence.buffers,
                names=context.recurrence.names,
                regions=context.recurrence.regions,
            )
            factor = compile_factor(epilogue, deferred.factor, states, "deferred_factor")
            append_scaled_output(epilogue, plans[deferred.stage].current, factor, context.match.external_outputs[0])
            epilogue_roots.append(carrier_nid)
        else:
            if context.output_tensor is None:
                raise AssertionError("mapped online recurrence has no output tensor")
            scope = context.stage_scopes[deferred.stage]
            if scope is None:
                raise AssertionError("mapped deferred epilogue has no output scope")
            carrier_nid, parent, substitutions = _append_scope_carrier(context.recurrence.tree, scope)
            final_region = context.recurrence.region(plans[deferred.stage].current)
            tile_shape = _region_shape(final_region)
            local_region_ranges = tuple((Const(value=0), Const(value=extent)) for extent in tile_shape)
            numerator = context.recurrence.names.fresh("online_deferred_numerator")
            normalized = context.match.external_outputs[0]
            template = context.recurrence.buffers[plans[deferred.stage].current]
            epilogue_regions = {
                tensor: _substitute_region(region, substitutions)
                for tensor, region in context.recurrence.regions.items()
            }
            for tensor in (numerator, normalized):
                context.recurrence.buffers[tensor] = replace(
                    template,
                    name=tensor,
                    shape=tile_shape,
                    location="sbuf",
                    storage_dtype="float32",
                    versions=1,
                    list_len=1,
                )
                epilogue_regions[tensor] = BufferRegion(tensor=tensor, ranges=local_region_ranges)
            epilogue = RecurrenceIR(
                tree=context.recurrence.tree,
                parent=parent,
                buffers=context.recurrence.buffers,
                names=context.recurrence.names,
                regions=epilogue_regions,
                scope=RecurrenceScope(block=context.recurrence.tree.block(carrier_nid), loops=()),
                localize_temps=True,
            )
            append_manual_block(
                epilogue.tree,
                epilogue.parent,
                NKILoad,
                {"src": epilogue.region(context.carry_tensor), "dst": epilogue.region(numerator)},
                {},
            )
            factor = compile_factor(epilogue, deferred.factor, states, "deferred_factor")
            append_scaled_output(epilogue, numerator, factor, normalized)
            append_manual_block(
                epilogue.tree,
                epilogue.parent,
                NKIStore,
                {"src": epilogue.region(normalized), "dst": epilogue.region(context.output_tensor)},
                {},
            )
            epilogue_roots.append(carrier_nid)
    return epilogue_roots


def _append_scope_carrier(tree: KernelTree, scope: RecurrenceScope) -> tuple[int, int, dict[str, Expr]]:
    """Append one mapped carrier with loop variables unique to the epilogue."""
    substitutions: dict[str, Expr] = {}
    loops: list[ForNode] = []
    for loop in scope.loops:
        loop_var = _fresh_loop_var(tree, f"{loop.loop_var}_online_epilogue")
        substitutions[loop.loop_var] = Var(name=loop_var)
        loops.append(replace(loop, loop_var=loop_var))
    carrier = replace(
        scope.block,
        iter_values=tuple(substitute(value, substitutions) for value in scope.block.iter_values),
        reads=(),
        writes=(),
        alloc_buffers=(),
    )
    carrier_nid = tree.add_node(carrier)
    parent = carrier_nid
    for loop in loops:
        parent = tree.add_node(loop, parent=parent)
    return carrier_nid, parent, substitutions


def _fresh_loop_var(tree: KernelTree, stem: str) -> str:
    """Return a loop variable not used anywhere in the current tree."""
    used = {payload.loop_var for nid in tree.graph.nodes if isinstance((payload := tree.data(nid)), ForNode)}
    candidate = stem
    suffix = 1
    while candidate in used:
        candidate = f"{stem}_{suffix}"
        suffix += 1
    return candidate


def _substitute_region(region: BufferRegion, substitutions: dict[str, Expr]) -> BufferRegion:
    """Substitute loop variables in one mapped buffer region."""
    return replace(
        region,
        ranges=tuple(
            (substitute(lower, substitutions), substitute(width, substitutions)) for lower, width in region.ranges
        ),
    )


def _stage_output_region(ir: KernelIR, graph: ValueGraph, stage: OnlineFusionStage) -> BufferRegion:
    """Return the reducer's state-output region."""
    leaf = ir.tree.isa(stage.reducer_leaf)
    output_operand = contract_output_operand(graph.contracts[stage.reducer_leaf])
    return leaf.operand_bindings[output_operand]


def _stage_scope(ir: KernelIR, graph: ValueGraph, stage: OnlineFusionStage, progress_axis: str) -> RecurrenceScope:
    """Return the reducer's mapped geometry with its reduction axis removed."""
    tree = ir.tree
    block = tree.block(stage.reducer_block)
    chain = block_chain(tree, stage.reducer_block)
    if chain is None:
        raise ValueError(f"online reducer block {stage.reducer_block} is not a canonical chain")
    retained_pairs = [
        (iter_var, value)
        for iter_var, value in zip(block.iter_vars, block.iter_values)
        if iter_var.axis != progress_axis
    ]
    retained_values = tuple(value for _iter_var, value in retained_pairs)
    loop_vars = {name for value in retained_values for name in to_affine(value) if name is not None}
    loops = tuple(payload for payload in chain[1:-1] if isinstance(payload, ForNode) and payload.loop_var in loop_vars)
    state_axes = graph.tensor_axes[stage.state_tensor]
    abstract_axes = ("P", "F")
    axis_map = {abstract: concrete for abstract, concrete in zip(abstract_axes, state_axes)}
    scoped_block = replace(
        block,
        iter_vars=tuple(iter_var for iter_var, _value in retained_pairs),
        iter_values=retained_values,
        reads=(),
        writes=(),
        alloc_buffers=(),
        axis_map=axis_map,
    )
    return RecurrenceScope(block=scoped_block, loops=loops)


def _stage_regions(
    plans: tuple[_StageBuffers, ...], output_regions: tuple[BufferRegion, ...]
) -> dict[str, BufferRegion]:
    """Map every recurrence buffer to its stage's mapped output region."""
    regions: dict[str, BufferRegion] = {}
    for plan, output_region in zip(plans, output_regions):
        for tensor in (plan.state, plan.contribution, plan.current):
            regions[tensor] = replace(output_region, tensor=tensor)
    return regions


def _region_shape(region: BufferRegion) -> tuple[int, ...]:
    """Return the constant tile shape represented by ``region``."""
    shape: list[int] = []
    for _lower, width in region.ranges:
        if not isinstance(width, Const):
            raise ValueError(f"online recurrence region width must be constant, got {width!r}")
        shape.append(width.value)
    return tuple(shape)


def _hbm_tile_region(on_chip_region: BufferRegion, tensor: str) -> BufferRegion:
    """Map an on-chip output tile region into element-addressed HBM."""
    ranges = list(on_chip_region.ranges)
    partition_lower, partition_width = ranges[0]
    ranges[0] = (Mul(left=partition_lower, right=Const(value=PARTITION_DIM)), partition_width)
    return BufferRegion(tensor=tensor, ranges=tuple(ranges))


def _localized_buffers(
    ir: KernelIR,
    match: OnlineFusionMatch,
    tensor_axes: Mapping[str, tuple[str, ...]],
    buffers: dict[str, Buffer],
    chunk_size: int,
) -> dict[str, Buffer]:
    """Shrink progress-carrying internal buffers to one chunk."""
    result = dict(buffers)
    internal = {
        region.tensor
        for leaf_nid in match.derivation_leaves
        if leaf_nid in ir.tree.graph
        for region in ir.tree.isa(leaf_nid).operand_bindings.values()
        if region.tensor not in match.external_inputs and region.tensor not in match.external_outputs
    }
    for name in internal:
        axes = tensor_axes.get(name, ())
        if name in result and match.progress_axis in axes:
            index = axes.index(match.progress_axis)
            shape = list(result[name].shape)
            shape[index] = chunk_size
            result[name] = replace(result[name], shape=tuple(shape))
        if name in result and len(result[name].shape) == 1:
            result[name] = replace(result[name], storage_dtype="float32")
    return result


def _append_cloned_block(
    context: _LoweringContext,
    graph: ValueGraph,
    leaf_nid: int,
    tensor_remap: dict[str, str],
    output_override: str | None,
) -> int:
    """Clone one canonical block into the shared sequential loop."""
    tree = context.ir.tree
    old_block_nid = owning_block(tree, leaf_nid)
    chain = block_chain(tree, old_block_nid)
    if chain is None:
        raise ValueError(f"block {old_block_nid} is not a canonical single-leaf chain")
    old_block = tree.block(old_block_nid)
    old_leaf = tree.isa(leaf_nid)
    output_operand = contract_output_operand(graph.contracts[leaf_nid])
    progress_vars: set[str] = set()
    iter_values = list(old_block.iter_values)
    for index, iter_var in enumerate(old_block.iter_vars):
        if iter_var.axis == context.match.progress_axis:
            progress_vars.update(var for var in to_affine(iter_values[index]) if var is not None)
    tiling = _progress_tiling(context, old_leaf, progress_vars)
    bindings: dict[str, BufferRegion] = {}
    for slot, region in old_leaf.operand_bindings.items():
        tensor = tensor_remap.get(region.tensor, region.tensor)
        if slot == output_operand and output_override is not None:
            tensor = output_override
        bindings[slot] = _localized_region(context, region, tensor, tiling)
    reads, writes = access_regions(old_leaf.op_cls, bindings, old_leaf.kwargs)
    for index, iter_var in enumerate(old_block.iter_vars):
        if iter_var.axis == context.match.progress_axis:
            iter_values[index] = _global_progress_value(context, tiling)
    block = replace(old_block, iter_values=tuple(iter_values), reads=reads, writes=writes, alloc_buffers=())
    block_nid = tree.add_node(block, parent=context.recurrence.parent)
    parent = block_nid
    for payload in chain[1:-1]:
        if isinstance(payload, ForNode) and payload.loop_var == context.outer_loop_var:
            continue
        if isinstance(payload, ForNode) and payload.loop_var in progress_vars:
            if tiling.trip_count > 1:
                parent = tree.add_node(replace(payload, extent=tiling.trip_count), parent=parent)
        else:
            parent = tree.add_node(payload, parent=parent)
    tree.add_node(
        ISANode(op_cls=old_leaf.op_cls, operand_bindings=bindings, kwargs=dict(old_leaf.kwargs)), parent=parent
    )
    return block_nid


def _progress_tiling(context: _LoweringContext, leaf: ISANode, progress_vars: set[str]) -> _ProgressTiling:
    """Derive the original ISA tile and the loop count within one chunk."""
    tile_sizes: set[int] = set()
    for region in leaf.operand_bindings.values():
        axes = context.tensor_axes[region.tensor]
        if context.match.progress_axis not in axes:
            continue
        index = axes.index(context.match.progress_axis)
        width = region.ranges[index][1]
        if not isinstance(width, Const):
            raise ValueError(f"online progress width must be constant, got {width!r}")
        tile_sizes.add(min(width.value, context.chunk_size))
    if not tile_sizes:
        return _ProgressTiling(loop_var=None, tile_size=context.chunk_size, trip_count=1)
    if len(tile_sizes) != 1:
        raise ValueError(f"online operation has inconsistent progress tiles {sorted(tile_sizes)}")
    tile_size = next(iter(tile_sizes))
    if context.chunk_size % tile_size != 0:
        raise ValueError(f"chunk size {context.chunk_size} is not divisible by tile size {tile_size}")
    trip_count = context.chunk_size // tile_size
    loop_var: str | None = None
    if trip_count > 1:
        if len(progress_vars) != 1:
            raise ValueError(f"multi-tile online operation requires one progress loop, got {sorted(progress_vars)}")
        loop_var = next(iter(progress_vars))
    return _ProgressTiling(loop_var=loop_var, tile_size=tile_size, trip_count=trip_count)


def _global_progress_value(context: _LoweringContext, tiling: _ProgressTiling) -> Expr:
    """Return the global tile coordinate represented by the cloned block."""
    outer = context.progress_index
    if tiling.trip_count > 1:
        if tiling.loop_var is None:
            raise AssertionError("multi-tile progress has no local loop variable")
        outer = Mul(left=outer, right=Const(value=tiling.trip_count))
        outer = Add(left=outer, right=Var(name=tiling.loop_var))
    return outer


def _localized_region(
    context: _LoweringContext, region: BufferRegion, tensor: str, tiling: _ProgressTiling
) -> BufferRegion:
    """Retarget one region and localize its progress dimension."""
    axes = context.tensor_axes[region.tensor]
    ranges = list(region.ranges)
    if context.match.progress_axis in axes:
        index = axes.index(context.match.progress_axis)
        buffer = context.ir.buffer(region.tensor)
        local_element, local_tile = _local_progress_offsets(tiling)
        if region.tensor in context.match.external_inputs:
            if buffer.location == "shared_hbm" or index > 0:
                outer = Mul(left=context.progress_index, right=Const(value=context.chunk_size))
                lo = _add_offset(outer, local_element)
            else:
                outer = Mul(left=context.progress_index, right=Const(value=tiling.trip_count))
                lo = _add_offset(outer, local_tile)
        else:
            lo = local_tile if buffer.location != "shared_hbm" and index == 0 else local_element
        ranges[index] = (lo, Const(value=tiling.tile_size))
    return BufferRegion(tensor=tensor, ranges=tuple(ranges))


def _local_progress_offsets(tiling: _ProgressTiling) -> tuple[Expr, Expr]:
    """Return element and on-chip tile offsets within the current chunk."""
    element: Expr = Const(value=0)
    tile: Expr = Const(value=0)
    if tiling.loop_var is not None:
        tile = Var(name=tiling.loop_var)
        element = Mul(left=tile, right=Const(value=tiling.tile_size))
    return element, tile


def _add_offset(left: Expr, right: Expr) -> Expr:
    """Add offsets without retaining a redundant zero term."""
    result = left
    if not isinstance(right, Const) or right.value != 0:
        result = Add(left=left, right=right)
    return result


def _append_stage_update(context: _LoweringContext, stage_index: int, plans: tuple[_StageBuffers, ...]) -> None:
    """Append one combiner or corrected additive recurrence."""
    stage = context.match.stages[stage_index]
    plan = plans[stage_index]
    context.recurrence.scope = context.stage_scopes[stage_index]
    is_final = stage_index == len(plans) - 1
    single_chunk = context.chunk_size == context.match.progress_extent
    if single_chunk:
        append_copy(context.recurrence, plan.contribution, plan.current)
    elif is_final and context.carry_tensor is not None:
        append_manual_block(
            context.recurrence.tree,
            context.recurrence.parent,
            NKILoad,
            {"src": context.recurrence.region(context.carry_tensor), "dst": context.recurrence.region(plan.state)},
            {},
            context.recurrence.scope,
        )
    if not single_chunk and stage_index == 0:
        append_tensor_tensor(context.recurrence, plan.state, plan.contribution, plan.current, stage.combinator.combiner)
    elif not single_chunk:
        factor = stage.factor
        deferred = context.match.deferred_factor
        if deferred is not None and stage_index == deferred.stage:
            factor = deferred.recurrence_factor
        if deferred is not None and stage_index == deferred.stage and factor is None:
            append_additive_update(context.recurrence, plan.state, plan.contribution, plan.current)
        elif factor is None:
            raise ValueError(f"online stage {stage_index} has no correction factor")
        else:
            update_scope = context.recurrence.scope
            state_indices = sorted(factor_states(factor))
            if state_indices:
                context.recurrence.scope = context.stage_scopes[state_indices[-1]]
            old_states = {index: prior.state for index, prior in enumerate(plans[:stage_index])}
            new_states = {index: prior.current for index, prior in enumerate(plans[:stage_index])}
            correction = compile_correction(
                context.recurrence, factor, old_states, new_states, f"stage{stage_index}_correction"
            )
            context.recurrence.scope = update_scope
            append_corrected_update(context.recurrence, plan.state, correction, plan.contribution, plan.current)
    if is_final and context.carry_tensor is not None:
        if context.output_tensor is None:
            raise AssertionError("mapped online recurrence has no output tensor")
        destinations = (context.carry_tensor,)
        if context.match.deferred_factor is None:
            destinations = (*destinations, context.output_tensor)
        for destination in destinations:
            append_manual_block(
                context.recurrence.tree,
                context.recurrence.parent,
                NKIStore,
                {"src": context.recurrence.region(plan.current), "dst": context.recurrence.region(destination)},
                {},
                context.recurrence.scope,
            )


def _match_tensors(match: OnlineFusionMatch, graph: ValueGraph) -> frozenset[str]:
    """Return tensors owned by one online-fusion rewrite boundary."""
    tensors: set[str] = set()
    for leaf_nid in match.derivation_leaves:
        tensors.add(graph.output_by_leaf[leaf_nid])
        tensors.update(graph.input_tensors_by_leaf[leaf_nid].values())
    tensors.update(stage.state_tensor for stage in match.stages)
    return frozenset(tensors)


def _seed_buffers(ir: KernelIR, buffers: dict[str, Buffer], prunable: frozenset[str]) -> None:
    """Update matched declarations in place and attach only missing live buffers at root."""
    tree = ir.tree
    touched = {
        region.tensor
        for nid in tree.preorder()
        if isinstance(tree.data(nid), ISANode)
        for region in tree.isa(nid).operand_bindings.values()
    }
    missing = touched - set(buffers) - set(ir.param_buffers)
    if missing:
        raise AssertionError(f"online lowering has no buffer declarations for {sorted(missing)}")

    declared: set[str] = set()
    for block_nid in tree.blocks():
        block = tree.block(block_nid)
        allocations: list[Buffer] = []
        for buffer in block.alloc_buffers:
            if buffer.name in prunable and buffer.name not in touched:
                continue
            replacement = buffers.get(buffer.name, buffer)
            allocations.append(replacement)
            declared.add(replacement.name)
        updated = tuple(allocations)
        if updated != block.alloc_buffers:
            tree.graph.nodes[block_nid]["data"] = replace(block, alloc_buffers=updated)

    root = tree.block(tree.root)
    additions = tuple(
        buffer
        for name, buffer in buffers.items()
        if name in touched and name not in ir.param_buffers and name not in declared
    )
    if additions:
        tree.graph.nodes[tree.root]["data"] = replace(root, alloc_buffers=(*root.alloc_buffers, *additions))


__all__ = [
    "OnlineFusionPrefixLowering",
    "can_lower_online_fusion",
    "can_lower_online_fusion_prefix",
    "complete_online_fusion_prefix",
    "lower_online_fusion",
    "lower_online_fusion_prefix",
]
