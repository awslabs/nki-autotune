"""Lower recurrence expressions and updates into nkigym operations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from nkigym.ir.arith.expr import Add, Const, Expr, Mul, Var, to_affine
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

_Compiled = tuple[str | None, float | None]


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
    outer_loop_var: str | None = None


def _plan_buffers(
    ir: KernelIR,
    match: _Match,
    graph: ValueGraph,
    names: NameSupply,
    preserve_deferred_dtype: bool,
    separate_final_current: bool,
) -> tuple[tuple[_Plan, ...], dict[str, Buffer]]:
    """Choose state, contribution, and current buffers."""
    plans: list[_Plan] = []
    buffers: dict[str, Buffer] = {}
    last = len(match.stages) - 1
    for index, stage in enumerate(match.stages):
        deferred = match.deferred_factor is not None and index == match.deferred_factor.stage
        if deferred:
            state = names.fresh(f"{stage.state_tensor}_online_state")
            source = ir.buffer(match.external_outputs[0])
            dtype = source.storage_dtype if preserve_deferred_dtype else "float32"
            buffers[state] = replace(source, name=state, location="sbuf", storage_dtype=dtype)
        else:
            state = match.external_outputs[0] if index == last else stage.state_tensor
            buffers[state] = replace(ir.buffer(state), location="sbuf", storage_dtype="float32")
        contribution = graph.outputs[stage.reducer_leaf]
        source = ir.buffer(contribution)
        raw_contribution: str | None = None
        if source.location == "psum":
            raw_contribution = names.fresh(f"{stage.state_tensor}_online_partial")
            buffers[raw_contribution] = replace(source, name=raw_contribution, storage_dtype="float32")
            contribution = names.fresh(f"{stage.state_tensor}_online_chunk")
            dtype = source.storage_dtype if deferred and preserve_deferred_dtype else "float32"
            buffers[contribution] = replace(source, name=contribution, location="sbuf", storage_dtype=dtype)
        elif index != last or contribution == state:
            contribution = names.fresh(f"{stage.state_tensor}_online_chunk")
            dtype = source.storage_dtype if deferred and preserve_deferred_dtype else "float32"
            buffers[contribution] = replace(source, name=contribution, storage_dtype=dtype)
        current = state
        if index != last or separate_final_current:
            current = names.fresh(f"{stage.state_tensor}_online_current")
            buffers[current] = replace(buffers[state], name=current)
        plans.append(_Plan(state, contribution, current, raw_contribution))
    output = match.external_outputs[0]
    buffers[output] = replace(ir.buffer(output), location="sbuf", storage_dtype="float32")
    return (tuple(plans), buffers)


def _recurrence_buffers(ir: KernelIR, match: _Match, buffers: Mapping[str, Buffer]) -> dict[str, Buffer]:
    """Preserve declarations while applying recurrence-required storage dtypes."""
    result = dict(buffers)
    internal = {
        region.tensor
        for nid in match.derivation_leaves
        if nid in ir.tree.graph
        for region in ir.tree.isa(nid).operand_bindings.values()
        if region.tensor not in match.external_inputs and region.tensor not in match.external_outputs
    }
    for name in internal:
        if name in result and len(result[name].shape) == 1:
            result[name] = replace(result[name], storage_dtype="float32")
    return result


def _stage_region(ir: KernelIR, graph: ValueGraph, stage: _Stage) -> BufferRegion:
    """Return one reducer state-output region."""
    contract = graph.contracts[stage.reducer_leaf]
    return ir.tree.isa(stage.reducer_leaf).operand_bindings[contract.output_operand]


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
    chain = block_chain(tree, old_block_nid)
    if chain is None:
        raise ValueError(f"block {old_block_nid} is not a canonical chain")
    old_block, leaf = tree.block(old_block_nid), tree.isa(nid)
    contract = context.graph.contracts[nid]
    progress_vars: set[str] = set()
    values = list(old_block.iter_values)
    for index, iter_var in enumerate(old_block.iter_vars):
        if iter_var.axis == context.match.progress_axis:
            progress_vars.update((name for name in to_affine(values[index]) if name is not None))
    loop_var, tile_size, trip_count = _progress_tiling(context, leaf, progress_vars)
    bindings: dict[str, BufferRegion] = {}
    for slot, region in leaf.operand_bindings.items():
        tensor = remap.get(region.tensor, region.tensor)
        if slot == contract.output_operand and output_override is not None:
            tensor = output_override
        bindings[slot] = _localized_region(context, region, tensor, loop_var, tile_size, trip_count)
    kwargs = dict(leaf.kwargs)
    for abstract, (key, slot) in getattr(leaf.op_cls, "SPLIT_OFFSET_KWARGS", {}).items():
        if old_block.axis_map.get(abstract) == context.match.progress_axis:
            local = bindings[slot].ranges[leaf.op_cls.OPERAND_AXES[slot].index(abstract)][0]
            kwargs[key] = Add(left=Mul(left=context.progress_index, right=Const(value=context.chunk_size)), right=local)
    reads, writes = _access_regions(leaf.op_cls, bindings, kwargs)
    for index, iter_var in enumerate(old_block.iter_vars):
        if iter_var.axis == context.match.progress_axis:
            progress: Expr = context.progress_index
            if trip_count > 1:
                assert loop_var is not None
                progress = Add(left=Mul(left=progress, right=Const(value=trip_count)), right=Var(name=loop_var))
            values[index] = progress
    block = replace(old_block, iter_values=tuple(values), reads=reads, writes=writes, alloc_buffers=())
    parent = tree.add_node(block, parent=context.builder.parent)
    for item in chain[1:-1]:
        if isinstance(item, ForNode) and item.loop_var == context.outer_loop_var:
            continue
        if isinstance(item, ForNode) and item.loop_var in progress_vars:
            if trip_count > 1:
                parent = tree.add_node(replace(item, extent=trip_count), parent=parent)
        else:
            parent = tree.add_node(item, parent=parent)
    tree.add_node(ISANode(op_cls=leaf.op_cls, operand_bindings=bindings, kwargs=kwargs), parent=parent)


def _progress_tiling(context: _Lowering, leaf: ISANode, progress_vars: set[str]) -> tuple[str | None, int, int]:
    """Derive the operation tile and local trip count."""
    sizes: set[int] = set()
    for region in leaf.operand_bindings.values():
        axes = context.graph.tensor_axes[region.tensor]
        if context.match.progress_axis in axes:
            width = region.ranges[axes.index(context.match.progress_axis)][1]
            if not isinstance(width, Const):
                raise ValueError("online progress width must be constant")
            sizes.add(min(width.value, context.chunk_size))
    if not sizes:
        return (None, context.chunk_size, 1)
    if len(sizes) != 1:
        raise ValueError(f"inconsistent online progress tiles {sorted(sizes)}")
    tile = next(iter(sizes))
    if context.chunk_size % tile:
        raise ValueError(f"chunk size {context.chunk_size} is not divisible by tile {tile}")
    trips = context.chunk_size // tile
    if trips > 1 and len(progress_vars) != 1:
        raise ValueError("multi-tile online operation requires one progress loop")
    return (next(iter(progress_vars)) if trips > 1 else None, tile, trips)


def _localized_region(
    context: _Lowering, region: BufferRegion, tensor: str, loop_var: str | None, tile_size: int, trip_count: int
) -> BufferRegion:
    """Retarget one region and localize its progress dimension."""
    axes = context.graph.tensor_axes[region.tensor]
    ranges = list(region.ranges)
    if context.match.progress_axis in axes:
        index = axes.index(context.match.progress_axis)
        buffer = context.ir.buffer(region.tensor)
        local_tile: Expr = Var(name=loop_var) if loop_var is not None else Const(value=0)
        local_element: Expr = (
            Mul(left=local_tile, right=Const(value=tile_size)) if loop_var is not None else Const(value=0)
        )
        if region.tensor in context.match.external_inputs:
            if buffer.location == "shared_hbm" or index > 0:
                lower = Add(
                    left=Mul(left=context.progress_index, right=Const(value=context.chunk_size)), right=local_element
                )
            else:
                lower = Add(left=Mul(left=context.progress_index, right=Const(value=trip_count)), right=local_tile)
        else:
            lower = local_tile if buffer.location != "shared_hbm" and index == 0 else local_element
        if isinstance(lower, Add) and isinstance(lower.right, Const) and lower.right.value == 0:
            lower = lower.left
        ranges[index] = (lower, Const(value=tile_size))
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
    tensor, _literal = _compile_value(builder, factor, states, stem)
    if tensor is None:
        raise ValueError(f"factor {factor!r} did not materialize a tensor")
    return tensor


def _compile_value(builder: OperationBuilder, factor: _Factor, states: Mapping[int, str], stem: str) -> _Compiled:
    """Recursively compile a tensor/literal factor."""
    if factor.stage is not None:
        return (states[factor.stage], None)
    if factor.literal is not None:
        return (None, factor.literal)
    if len(factor.operands) == 1:
        operand_factor, scale, bias = _flatten_affine(factor)
        operand = _compile_factor(builder, operand_factor, states, f"{stem}_arg")
        output = builder.temp(f"{stem}_{factor.operator}", operand)
        kwargs: dict[str, Any] = {"op": factor.operator}
        if scale != 1.0:
            kwargs["scale"] = scale
        if bias != 0.0:
            kwargs["bias"] = bias
        builder.append(NKIActivation, {"data": builder.region(operand), "dst": builder.region(output)}, kwargs)
        return (output, None)
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
    left_tensor, left_literal = left
    right_tensor, right_literal = right
    if left_tensor is not None and right_tensor is not None:
        output = builder.temp(stem, left_tensor)
        _emit_tensor_tensor(builder, left_tensor, right_tensor, output, operator)
        return (output, None)
    if left_tensor is not None and right_literal is not None:
        output = builder.temp(stem, left_tensor)
        _emit_tensor_scalar(builder, left_tensor, right_literal, output, operator, False)
        return (output, None)
    if right_tensor is not None and left_literal is not None:
        output = builder.temp(stem, right_tensor)
        _emit_tensor_scalar(builder, right_tensor, left_literal, output, operator, True)
        return (output, None)
    if left_literal is not None and right_literal is not None:
        functions = {
            "add": lambda a, b: a + b,
            "subtract": lambda a, b: a - b,
            "multiply": lambda a, b: a * b,
            "maximum": max,
        }
        return (None, float(functions[operator](left_literal, right_literal)))
    raise ValueError("binary factor has neither tensor nor literal operands")


def _compile_correction(
    builder: OperationBuilder, factor: _Factor, old_states: Mapping[int, str], new_states: Mapping[int, str], stem: str
) -> str:
    """Materialize the stable ratio ``factor(new) / factor(old)``."""
    if factor.literal is not None:
        raise ValueError("constant correction does not materialize a tensor")
    if factor.operator == "multiply" and len(factor.operands) == 2:
        left = _compile_correction(builder, factor.operands[0], old_states, new_states, f"{stem}_left")
        right = _compile_correction(builder, factor.operands[1], old_states, new_states, f"{stem}_right")
        output = builder.temp(stem, left)
        _emit_tensor_tensor(builder, left, right, output, "multiply")
        return output
    if len(factor.operands) == 1 and factor.operator in {"rsqrt", "exp", "reciprocal"}:
        operand, scale, bias = _flatten_affine(factor)
        old = _compile_factor(builder, operand, old_states, f"{stem}_old_arg")
        new = _compile_factor(builder, operand, new_states, f"{stem}_new_arg")
        if factor.operator == "exp":
            difference = builder.temp(f"{stem}_difference", new)
            _emit_tensor_tensor(builder, new, old, difference, "subtract")
            output = builder.temp(stem, difference)
            kwargs: dict[str, Any] = {"op": "exp"}
            if scale != 1.0:
                kwargs["scale"] = scale
            builder.append(NKIActivation, {"data": builder.region(difference), "dst": builder.region(output)}, kwargs)
            return output
        old_affine = _emit_affine(builder, old, scale, bias, f"{stem}_old")
        new_affine = _emit_affine(builder, new, scale, bias, f"{stem}_new")
        if factor.operator == "rsqrt":
            old_affine = _emit_unary(builder, old, "sqrt", scale, bias, f"{stem}_old_sqrt")
            new_affine = _emit_unary(builder, new, "rsqrt", scale, bias, f"{stem}_new_rsqrt")
            output = builder.temp(stem, new_affine)
            _emit_tensor_tensor(builder, new_affine, old_affine, output, "multiply")
            return output
        return _emit_ratio(builder, old_affine, new_affine, stem)
    old = _compile_factor(builder, factor, old_states, f"{stem}_old")
    new = _compile_factor(builder, factor, new_states, f"{stem}_new")
    return _emit_ratio(builder, new, old, stem)


def _emit_ratio(builder: OperationBuilder, numerator: str, denominator: str, stem: str) -> str:
    """Materialize one tensor ratio."""
    inverse = _emit_unary(builder, denominator, "reciprocal", 1.0, 0.0, f"{stem}_inverse")
    output = builder.temp(stem, numerator)
    _emit_tensor_tensor(builder, numerator, inverse, output, "multiply")
    return output


def _emit_unary(builder: OperationBuilder, data: str, operator: str, scale: float, bias: float, stem: str) -> str:
    """Emit one affine activation."""
    output = builder.temp(stem, data)
    kwargs: dict[str, Any] = {"op": operator}
    if scale != 1.0:
        kwargs["scale"] = scale
    if bias != 0.0:
        kwargs["bias"] = bias
    builder.append(NKIActivation, {"data": builder.region(data), "dst": builder.region(output)}, kwargs)
    return output


def _emit_affine(builder: OperationBuilder, data: str, scale: float, bias: float, stem: str) -> str:
    """Emit a non-identity affine copy."""
    return data if scale == 1.0 and bias == 0.0 else _emit_unary(builder, data, "copy", scale, bias, stem)


def _emit_tensor_tensor(builder: OperationBuilder, left: str, right: str, output: str, operator: str) -> None:
    """Emit one tensor-tensor operation."""
    builder.append(
        NKITensorTensor,
        {"data1": builder.region(left), "data2": builder.region(right), "dst": builder.region(output)},
        {"op": operator},
    )


def _emit_tensor_scalar(
    builder: OperationBuilder, data: str, operand: float, output: str, operator: str, reverse: bool
) -> None:
    """Emit one literal tensor-scalar operation."""
    kwargs: dict[str, Any] = {"op0": operator, "operand0": operand}
    if reverse:
        kwargs["reverse0"] = True
    builder.append(NKITensorScalar, {"data": builder.region(data), "dst": builder.region(output)}, kwargs)


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
        if (deferred := context.match.deferred_factor) is not None and nid == deferred.producer_leaf:
            continue
        if deferred is not None and nid == deferred.bypass_leaf:
            source = context.graph.inputs[nid][deferred.passthrough_operand]
            remap[context.graph.outputs[nid]] = remap.get(source, source)
            continue
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
    return builder.append(NKITensorCopy, {"src": builder.region(source), "dst": builder.region(destination)}, {})


def _emit_initializer(builder: OperationBuilder, tensor: str, value: float) -> int:
    """Emit one full-region initializer."""
    return builder.append(NKIMemset, {"dst": builder.region(tensor)}, {"value": value})


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
        context.builder.append(
            NKILoad,
            {"src": context.builder.region(context.carry_tensor), "dst": context.builder.region(plan.state)},
            {},
        )
    if not single_chunk and index == 0:
        _emit_tensor_tensor(context.builder, plan.state, plan.contribution, plan.current, stage.combinator.combiner)
    elif not single_chunk:
        factor = stage.factor
        deferred = context.match.deferred_factor
        if deferred is not None and index == deferred.stage:
            factor = deferred.recurrence_factor
        if deferred is not None and index == deferred.stage and (factor is None):
            context.builder.append(
                NKIScalarTensorTensor,
                {
                    "data": context.builder.region(plan.state),
                    "operand1": context.builder.region(plan.contribution),
                    "dst": context.builder.region(plan.current),
                },
                {"op0": "multiply", "operand0": 1.0, "op1": "add"},
            )
        elif factor is None:
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
                {
                    "data": context.builder.region(plan.state),
                    "operand0": context.builder.region(correction),
                    "operand1": context.builder.region(plan.contribution),
                    "dst": context.builder.region(plan.current),
                },
                {"op0": "multiply", "op1": "add"},
            )
    if final and context.carry_tensor is not None:
        context.builder.append(
            NKIStore,
            {"src": context.builder.region(plan.current), "dst": context.builder.region(context.carry_tensor)},
            {},
        )
