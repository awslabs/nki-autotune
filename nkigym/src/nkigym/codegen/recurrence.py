"""Lower recurrence expressions and updates into nkigym operations."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from nkigym.ir.arith.expr import Add, Const, Expr, Mul, to_affine
from nkigym.ir.ir import KernelIR
from nkigym.ir.recurrence import _Match, _Stage
from nkigym.ir.tree import Buffer, BufferRegion, ForNode, ISANode
from nkigym.ops.activation import NKIActivation
from nkigym.ops.base import NKIOp
from nkigym.ops.base import _recurrence_factor_states as _factor_states
from nkigym.ops.base import _RecurrenceFactor as _Factor
from nkigym.ops.load import NKILoad
from nkigym.ops.memset import NKIMemset
from nkigym.ops.scalar_tensor_tensor import NKIScalarTensorTensor
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.ops.tensor_scalar import NKITensorScalar
from nkigym.ops.tensor_tensor import NKITensorTensor
from nkigym.transforms.helper.canonical_rewrite import block_chain, owning_block
from nkigym.transforms.helper.operation_builder import NameSupply, OperationBuilder, OperationScope
from nkigym.transforms.helper.value_graph import ValueGraph

_Compiled = str | float


@dataclass(frozen=True)
class _Plan:
    """Buffers for one recurrence stage."""

    state: str
    contribution: str
    current: str
    raw_contribution: str | None = None


@dataclass
class _Lowering:
    """State shared while cloning one per-chunk derivation."""

    ir: KernelIR
    match: _Match
    graph: ValueGraph
    chunk_size: int
    progress_index: Expr
    builder: OperationBuilder
    scopes: tuple[OperationScope | None, ...]
    carry_tensor: str | None = None


def _plan_buffers(
    ir: KernelIR, match: _Match, graph: ValueGraph, names: NameSupply, separate_final_current: bool
) -> tuple[tuple[_Plan, ...], dict[str, Buffer]]:
    """Choose state, contribution, and current buffers."""
    if match.deferred_factor is not None:
        raise ValueError("online lowering does not support deferred recurrence factors")
    plans: list[_Plan] = []
    buffers: dict[str, Buffer] = {}
    last = len(match.stages) - 1
    for index, stage in enumerate(match.stages):
        state = match.external_outputs[0] if index == last else stage.state_tensor
        buffers[state] = replace(ir.buffer(state), location="sbuf", storage_dtype="float32")
        contribution = graph.outputs[stage.reducer_leaf]
        source = ir.buffer(contribution)
        raw_contribution: str | None = None
        if source.location == "psum":
            raw_contribution = names.fresh(f"{stage.state_tensor}_online_partial")
            buffers[raw_contribution] = replace(source, name=raw_contribution, storage_dtype="float32")
            source = replace(source, location="sbuf")
        if raw_contribution is not None or index != last or contribution == state:
            contribution = names.fresh(f"{stage.state_tensor}_online_chunk")
            buffers[contribution] = replace(source, name=contribution, storage_dtype="float32")
        current = state
        if index != last or separate_final_current:
            current = names.fresh(f"{stage.state_tensor}_online_current")
            buffers[current] = replace(buffers[state], name=current)
        plans.append(_Plan(state, contribution, current, raw_contribution))
    return (tuple(plans), buffers)


def _stage_region(ir: KernelIR, graph: ValueGraph, stage: _Stage) -> BufferRegion:
    """Return one reducer state-output region."""
    return ir.tree.isa(stage.reducer_leaf).operand_bindings[graph.contracts[stage.reducer_leaf].output_operand]


def _stage_regions(plans: tuple[_Plan, ...], regions: tuple[BufferRegion, ...]) -> dict[str, BufferRegion]:
    """Map every stage buffer to its reducer output region."""
    return {
        tensor: replace(region, tensor=tensor)
        for plan, region in zip(plans, regions)
        for tensor in (plan.state, plan.contribution, plan.current, plan.raw_contribution)
        if tensor is not None
    }


def _clone_block(context: _Lowering, nid: int, remap: Mapping[str, str], output_override: str | None) -> None:
    """Clone one canonical operation into the recurrence loop."""
    tree = context.ir.tree
    old_block_nid = owning_block(tree, nid)
    old_block, leaf = tree.block(old_block_nid), tree.isa(nid)
    loops = tuple(tree.loop(ancestor) for ancestor in tree.ancestors(nid) if isinstance(tree.data(ancestor), ForNode))
    contract = context.graph.contracts[nid]
    progress_vars: set[str] = set()
    values = list(old_block.iter_values)
    for index, iter_var in enumerate(old_block.iter_vars):
        if iter_var.axis == context.match.progress_axis:
            progress_vars.update((name for name in to_affine(values[index]) if name is not None))
            values[index] = context.progress_index
    _check_progress_width(context, leaf)
    bindings: dict[str, BufferRegion] = {}
    for slot, region in leaf.operand_bindings.items():
        tensor = remap.get(region.tensor, region.tensor)
        if slot == contract.output_operand and output_override is not None:
            tensor = output_override
        bindings[slot] = _localized_region(context, region, tensor)
    kwargs = dict(leaf.kwargs)
    for abstract, (key, slot) in getattr(leaf.op_cls, "SPLIT_OFFSET_KWARGS", {}).items():
        if old_block.axis_map.get(abstract) == context.match.progress_axis:
            local = bindings[slot].ranges[leaf.op_cls.operand_dimension(slot, abstract)][0]
            kwargs[key] = Add(left=Mul(left=context.progress_index, right=Const(value=context.chunk_size)), right=local)
    reads, writes = _access_regions(leaf.op_cls, bindings, kwargs)
    block = replace(old_block, iter_values=tuple(values), reads=reads, writes=writes, alloc_buffers=())
    parent = tree.add_node(block, parent=context.builder.parent)
    for item in loops:
        if item.loop_var not in progress_vars:
            parent = tree.add_node(item, parent=parent)
    tree.add_node(ISANode(op_cls=leaf.op_cls, operand_bindings=bindings, kwargs=kwargs), parent=parent)


def _check_progress_width(context: _Lowering, leaf: ISANode) -> None:
    """Require explicit tiling to match the retained recurrence chunk."""
    sizes: set[int] = set()
    for region in leaf.operand_bindings.values():
        axes = context.graph.tensor_axes[region.tensor]
        if context.match.progress_axis in axes:
            width = region.ranges[axes.index(context.match.progress_axis)][1]
            if not isinstance(width, Const):
                raise ValueError("online progress width must be constant")
            sizes.add(width.value)
    if sizes and sizes != {context.chunk_size}:
        raise ValueError(f"online progress tiles {sorted(sizes)} must match chunk size {context.chunk_size}")


def _localized_region(context: _Lowering, region: BufferRegion, tensor: str) -> BufferRegion:
    """Retarget one region while preserving its per-chunk storage coordinates."""
    axes = context.graph.tensor_axes[region.tensor]
    ranges = list(region.ranges)
    if context.match.progress_axis in axes:
        index = axes.index(context.match.progress_axis)
        buffer = context.ir.buffer(region.tensor)
        stride = context.chunk_size if buffer.location == "shared_hbm" or index > 0 else 1
        lower = Mul(left=context.progress_index, right=Const(value=stride))
        ranges[index] = (lower, Const(value=context.chunk_size))
    return BufferRegion(tensor=tensor, ranges=tuple(ranges))


def _access_regions(
    op_cls: type[NKIOp], bindings: Mapping[str, BufferRegion], kwargs: Mapping[str, Any]
) -> tuple[tuple[BufferRegion, ...], tuple[BufferRegion, ...]]:
    """Derive reads and writes from operation metadata."""
    rmw = op_cls.rmw_operands(dict(kwargs))
    reads = tuple(region for slot, region in bindings.items() if slot in op_cls.INPUT_OPERANDS or slot in rmw)
    writes = tuple(region for slot, region in bindings.items() if slot not in op_cls.INPUT_OPERANDS)
    return (reads, writes)


def _compile_factor(builder: OperationBuilder, factor: _Factor, states: Mapping[int, str], stem: str) -> str:
    """Materialize one state factor and return its tensor."""
    tensor = _compile_value(builder, factor, states, stem)
    if not isinstance(tensor, str):
        raise ValueError(f"factor {factor!r} did not materialize a tensor")
    return tensor


def _compile_value(builder: OperationBuilder, factor: _Factor, states: Mapping[int, str], stem: str) -> _Compiled:
    """Recursively compile a tensor/literal factor."""
    if factor.stage is not None:
        return states[factor.stage]
    if factor.literal is not None:
        return factor.literal
    if len(factor.operands) == 1:
        operand_factor, scale, bias = _flatten_affine(factor)
        operand = _compile_factor(builder, operand_factor, states, f"{stem}_arg")
        output = builder.temp(f"{stem}_{factor.operator}", operand)
        kwargs: dict[str, Any] = {"op": factor.operator}
        if scale != 1.0:
            kwargs["scale"] = scale
        if bias != 0.0:
            kwargs["bias"] = bias
        builder.append(NKIActivation, {"data": operand, "dst": output}, kwargs)
        return output
    if len(factor.operands) == 2:
        left = _compile_value(builder, factor.operands[0], states, f"{stem}_left")
        right = _compile_value(builder, factor.operands[1], states, f"{stem}_right")
        return _compile_binary(builder, factor.operator, left, right, stem)
    raise TypeError(f"unsupported factor {factor!r}")


def _flatten_affine(factor: _Factor) -> tuple[_Factor, float, float]:
    """Fold nested copy factors into one unary operation."""
    operand = factor.operands[0]
    scale = factor.scale
    bias = factor.bias
    while operand.operator == "copy" and len(operand.operands) == 1:
        bias = operand.bias * scale + bias
        scale *= operand.scale
        operand = operand.operands[0]
    return (operand, scale, bias)


def _compile_binary(
    builder: OperationBuilder, operator: str, left: _Compiled, right: _Compiled, stem: str
) -> _Compiled:
    """Compile one binary tensor/literal factor."""
    if isinstance(left, float) and isinstance(right, float):
        functions = {
            "add": lambda a, b: a + b,
            "subtract": lambda a, b: a - b,
            "multiply": lambda a, b: a * b,
            "maximum": max,
        }
        return float(functions[operator](left, right))
    tensor = left if isinstance(left, str) else right
    if not isinstance(tensor, str):
        raise ValueError("binary factor has no tensor operand")
    output = builder.temp(stem, tensor)
    if isinstance(left, str) and isinstance(right, str):
        _emit_tensor_tensor(builder, left, right, output, operator)
    else:
        literal = right if isinstance(left, str) else left
        if not isinstance(literal, float):
            raise ValueError("scalar factor requires a floating literal")
        kwargs: dict[str, Any] = {"op0": operator, "operand0": literal}
        if isinstance(left, float):
            kwargs["reverse0"] = True
        builder.append(NKITensorScalar, {"data": tensor, "dst": output}, kwargs)
    return output


def _regular_correction(factor: _Factor) -> bool:
    """Require a correction that does not divide by a possibly zero state."""
    if factor.operator == "multiply" and len(factor.operands) == 2:
        return all(_regular_correction(operand) for operand in factor.operands)
    return factor.operator == "exp" and len(factor.operands) == 1


def _compile_correction(
    builder: OperationBuilder, factor: _Factor, old_states: Mapping[int, str], new_states: Mapping[int, str], stem: str
) -> str:
    """Materialize the stable ratio ``factor(new) / factor(old)``."""
    if not _regular_correction(factor):
        raise ValueError("online correction requires a nonsingular exponential ratio")
    if factor.operator == "multiply" and len(factor.operands) == 2:
        left = _compile_correction(builder, factor.operands[0], old_states, new_states, f"{stem}_left")
        right = _compile_correction(builder, factor.operands[1], old_states, new_states, f"{stem}_right")
        output = builder.temp(stem, left)
        _emit_tensor_tensor(builder, left, right, output, "multiply")
        return output
    operand, scale, _bias = _flatten_affine(factor)
    old = _compile_factor(builder, operand, old_states, f"{stem}_old_arg")
    new = _compile_factor(builder, operand, new_states, f"{stem}_new_arg")
    difference = builder.temp(f"{stem}_difference", new)
    _emit_tensor_tensor(builder, new, old, difference, "subtract")
    output = builder.temp(stem, difference)
    kwargs: dict[str, Any] = {"op": "exp"}
    if scale != 1.0:
        kwargs["scale"] = scale
    builder.append(NKIActivation, {"data": difference, "dst": output}, kwargs)
    return output


def _emit_tensor_tensor(builder: OperationBuilder, left: str, right: str, output: str, operator: str) -> None:
    """Emit one tensor-tensor operation."""
    builder.append(NKITensorTensor, {"data1": left, "data2": right, "dst": output}, {"op": operator})


def _derive(
    context: _Lowering,
    plans: tuple[_Plan, ...],
    initial_remap: Mapping[str, str] | None = None,
    selected: frozenset[int] | None = None,
    roll_forward: bool = True,
) -> None:
    """Clone per-chunk work and append recurrence updates."""
    remap = dict(initial_remap or {})
    contributions = [stage.reducer_leaf for stage in context.match.stages]
    stage_by_leaf = {leaf: index for index, leaf in enumerate(contributions)}
    overrides = {
        context.graph.outputs[leaf]: plan.raw_contribution or plan.contribution
        for leaf, plan in zip(contributions, plans)
    }
    leaves = context.match.derivation_leaves
    if selected is not None:
        leaves = tuple((nid for nid in leaves if nid in selected))
    for nid in leaves:
        output = context.graph.outputs[nid]
        _clone_block(context, nid, remap, overrides.get(output))
        stage_index = stage_by_leaf.get(nid)
        if stage_index is not None:
            plan = plans[stage_index]
            if plan.raw_contribution is not None:
                context.builder.scope = context.scopes[stage_index]
                _emit_copy(context.builder, plan.raw_contribution, plan.contribution)
            _update_stage(context, stage_index, plans)
            remap[context.match.stages[stage_index].state_tensor] = plans[stage_index].current
    if roll_forward:
        for index, plan in enumerate(plans[:-1]):
            context.builder.scope = context.scopes[index]
            _emit_copy(context.builder, plan.current, plan.state)


def _emit_copy(builder: OperationBuilder, source: str, destination: str) -> int:
    """Emit one explicit tensor copy."""
    return builder.append(NKITensorCopy, {"src": source, "dst": destination}, {})


def _emit_initializer(builder: OperationBuilder, tensor: str, value: float) -> int:
    """Emit one full-region initializer."""
    return builder.append(NKIMemset, {"dst": tensor}, {"value": value})


def _update_stage(context: _Lowering, index: int, plans: tuple[_Plan, ...]) -> None:
    """Append one recurrence combiner and any HBM carry traffic."""
    stage = context.match.stages[index]
    plan = plans[index]
    context.builder.scope = context.scopes[index]
    final = index == len(plans) - 1
    single_chunk = context.chunk_size == context.match.progress_extent
    if single_chunk:
        _emit_copy(context.builder, plan.contribution, plan.current)
    elif final and context.carry_tensor is not None:
        context.builder.append(NKILoad, {"src": context.carry_tensor, "dst": plan.state}, {})
    if not single_chunk and index == 0:
        _emit_tensor_tensor(context.builder, plan.state, plan.contribution, plan.current, stage.combinator.combiner)
    elif not single_chunk:
        factor = stage.factor
        if factor is None:
            raise ValueError(f"online stage {index} has no correction factor")
        else:
            update_scope = context.builder.scope
            state_indices = sorted(_factor_states(factor))
            if state_indices:
                context.builder.scope = context.scopes[state_indices[-1]]
            old = {prior: prior_plan.state for prior, prior_plan in enumerate(plans[:index])}
            new = {prior: prior_plan.current for prior, prior_plan in enumerate(plans[:index])}
            correction = _compile_correction(context.builder, factor, old, new, f"stage{index}_correction")
            context.builder.scope = update_scope
            context.builder.append(
                NKIScalarTensorTensor,
                {"data": plan.state, "operand0": correction, "operand1": plan.contribution, "dst": plan.current},
                {"op0": "multiply", "op1": "add"},
            )
    if final and context.carry_tensor is not None:
        context.builder.append(NKIStore, {"src": plan.current, "dst": context.carry_tensor}, {})
