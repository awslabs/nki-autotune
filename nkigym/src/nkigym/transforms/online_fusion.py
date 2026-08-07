"""Contract-driven online-fusion transform."""

from __future__ import annotations

# fmt: off
import copy
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from numbers import Real
from typing import Any, Literal, cast

from nkigym.ir import KernelIR
from nkigym.ir.arith.expr import Add, Const, Expr, Mul, Var, substitute, to_affine
from nkigym.ir.tree import PARTITION_DIM, BlockNode, Buffer, BufferRegion, ForNode, ISANode, IterVar, KernelTree
from nkigym.ops.activation import NKIActivation
from nkigym.ops.activation_reduce import NKIActivationReduce
from nkigym.ops.base import (
    AxisRole,
    BilinearReductionContract,
    CopyContract,
    InitializerContract,
    NKIOp,
    OperatorContract,
    PermutationContract,
    PointwiseContract,
    PointwiseSequenceContract,
    ReduceCombinator,
    ReductionContract,
)
from nkigym.ops.load import NKILoad
from nkigym.ops.memset import NKIMemset
from nkigym.ops.scalar_tensor_tensor import NKIScalarTensorTensor
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.ops.tensor_scalar import NKITensorScalar
from nkigym.ops.tensor_scalar_reduce import NKITensorScalarReduce
from nkigym.ops.tensor_tensor import NKITensorTensor
from nkigym.transforms.base import Transform, TransformLegalityError, TransformOption
from nkigym.transforms.helper.canonical_rewrite import (
    block_chain,
    canonical_spec,
    finalize_rewrite,
    is_canonical_block,
    owning_block,
)
from nkigym.transforms.helper.operation_builder import NameSupply, OperationBuilder, OperationScope
from nkigym.transforms.helper.value_graph import ValueGraph, build_value_graph, contract_input_operands

_INCREMENTAL_ANNOTATION = 'online_fusion_incremental'
_ValueKind = Literal['constant', 'residual', 'state', 'additive', 'multiplicative', 'unknown']
@dataclass(frozen=True)
class _Factor:
    """One scalar or mapped expression over preceding recurrence states."""; operator: str; operands: tuple[_Factor, ...] = (); stage: int | None = None
    literal: float | None = None; scale: float = 1.0; bias: float = 0.0
@dataclass(frozen=True)
class _Stage:
    """One ordered reduction state."""; reducer_leaf: int; reducer_block: int; state_tensor: str; combinator: ReduceCombinator; factor: _Factor | None
    factor_axes: tuple[str, ...]
@dataclass(frozen=True)
class _Deferred:
    """One factor applied after the sequential recurrence."""; stage: int; factor: _Factor; recurrence_factor: _Factor | None; producer_leaf: int
    bypass_leaf: int; passthrough_operand: str
@dataclass(frozen=True)
class _Match:
    """A proven recurrence chain and exact rewrite boundary."""; progress_axis: str; progress_extent: int; stages: tuple[_Stage, ...]
    derivation_leaves: tuple[int, ...]; absorbed_blocks: tuple[int, ...]; external_inputs: tuple[str, ...]; external_outputs: tuple[str, ...]
    chunk_sizes: tuple[int, ...]; deferred_factor: _Deferred | None; incremental_prefix: bool = False
    @property
    def match_id(self) -> tuple[str, tuple[int, ...]]:
        """Return the stable public option identity."""; return (self.progress_axis, tuple((stage.reducer_leaf for stage in self.stages)))
@dataclass(frozen=True)
class _Value:
    """Abstract state/chunk separation result."""; kind: _ValueKind; factor: _Factor | None = None; factor_axes: tuple[str, ...] = ()
    depends_on_progress: bool = False
@dataclass(frozen=True)
class _Evaluation:
    """Algebraic interpretation for one candidate progress axis."""; stages: tuple[_Stage, ...]; by_tensor: Mapping[str, _Value]; by_leaf: Mapping[int, _Value]
def _factor_states(factor: _Factor | None) -> frozenset[int]:
    """Return all recurrence stages referenced by a factor."""; states: set[int] = set()
    if factor is not None:
        if factor.stage is not None: states.add(factor.stage)
        for operand in factor.operands: states.update(_factor_states(operand))
    return frozenset(states)
def _state_factor(stage: int) -> _Factor:
    """Construct one recurrence-state reference."""; return _Factor(operator='state', stage=stage)
def _constant_factor(value: float) -> _Factor:
    """Construct one scalar literal."""; return _Factor(operator='constant', literal=value)
def _unary_factor(operator: str, operand: _Factor, scale: float=1.0, bias: float=0.0) -> _Factor:
    """Construct one affine unary factor."""; return _Factor(operator=operator, operands=(operand,), scale=scale, bias=bias)
def _binary_factor(operator: str, left: _Factor, right: _Factor) -> _Factor:
    """Construct one binary factor."""; return _Factor(operator=operator, operands=(left, right))
def _evaluate(ir: KernelIR, graph: ValueGraph, progress_axis: str) -> _Evaluation:
    """Propagate state/chunk separation through operation contracts."""; by_tensor: dict[str, _Value] = {}; by_leaf: dict[int, _Value] = {}
    stages: list[_Stage] = []
    for nid in graph.leaves:
        leaf = ir.tree.isa(nid); contract = graph.contracts[nid]
        values = {slot: _operand_value(leaf, slot, contract, graph, by_tensor, progress_axis) for slot in contract_input_operands(contract) if _operand_available(leaf, contract, slot)}
        value, stage, mapped = _transfer(ir, graph, nid, contract, values, progress_axis, tuple(stages))
        if isinstance(contract, ReductionContract) and contract.mapped_output_operand is not None:
            region = leaf.operand_bindings.get(contract.mapped_output_operand)
            if region is None or mapped is None: raise ValueError(f'{leaf.op_cls.__name__} has no bound mapped output')
            by_tensor[region.tensor] = mapped
        by_tensor[graph.outputs[nid]] = value; by_leaf[nid] = value
        if stage is not None: stages.append(stage)
    return _Evaluation(tuple(stages), by_tensor, by_leaf)
def _operand_available(leaf: ISANode, contract: OperatorContract, slot: str) -> bool:
    """Return whether an operand is a tensor binding or independent literal."""
    bias = contract.bias_operand if isinstance(contract, (PointwiseContract, ReductionContract)) else None
    return slot in leaf.operand_bindings or (slot != bias and slot in leaf.kwargs)
def _operand_value(leaf: ISANode, slot: str, contract: OperatorContract, graph: ValueGraph, values: Mapping[str, _Value], progress_axis: str) -> _Value:
    """Resolve one tensor or literal operand."""; del contract; region = leaf.operand_bindings.get(slot)
    if region is not None:
        result = values.get(region.tensor)
        if result is None: result = _Value('residual', depends_on_progress=progress_axis in graph.tensor_axes[region.tensor])
    else: literal = leaf.kwargs.get(slot); result = _Value('constant', _constant_factor(float(literal))) if isinstance(literal, Real) else _Value('unknown')
    return result
def _transfer(ir: KernelIR, graph: ValueGraph, nid: int, contract: OperatorContract, inputs: Mapping[str, _Value], progress_axis: str, stages: tuple[_Stage, ...]) -> tuple[_Value, _Stage | None, _Value | None]:
    """Apply one contract's separation transfer rule."""; stage: _Stage | None = None; mapped: _Value | None = None
    if isinstance(contract, PointwiseContract): value = _pointwise(contract, inputs)
    elif isinstance(contract, (CopyContract, PermutationContract)): value = inputs[contract.input_operand]
    elif isinstance(contract, ReductionContract):
        mapped = _copy_affine(inputs[contract.input_operand], contract.scale, contract.bias)
        if contract.bias_operand is not None and contract.bias_operand in inputs: mapped = _add(mapped, inputs[contract.bias_operand], 'add')
        if contract.map_operator != 'copy': mapped = _unary(contract.map_operator, mapped)
        value, stage = _reduce(ir, graph, nid, contract.combinator, contract.reduction_axis, mapped, progress_axis, stages)
    elif isinstance(contract, BilinearReductionContract):
        contribution = _multiply(inputs[contract.left_operand], inputs[contract.right_operand])
        value, stage = _reduce(ir, graph, nid, contract.combinator, contract.reduction_axis, contribution, progress_axis, stages)
    elif isinstance(contract, InitializerContract): value = _Value('constant', _constant_factor(contract.value))
    elif isinstance(contract, PointwiseSequenceContract): value = _Value('unknown')
    else: raise TypeError(f'unsupported contract {type(contract).__name__}')
    return (value, stage, mapped)
def _pointwise(contract: PointwiseContract, inputs: Mapping[str, _Value]) -> _Value:
    """Interpret one unary or binary pointwise operation."""; operands = tuple((inputs[name] for name in contract.input_operands))
    if len(operands) == 1:
        if contract.bias_operand is not None and contract.bias_operand in inputs:
            value = _copy_affine(operands[0], contract.scale, contract.bias); value = _add(value, inputs[contract.bias_operand], 'add')
            result = value if contract.operator == 'copy' else _unary(contract.operator, value)
        else: result = _unary(contract.operator, operands[0], contract.scale, contract.bias)
    elif len(operands) == 2:
        left, right = reversed(operands) if contract.reverse else operands
        if contract.operator == 'multiply': result = _multiply(left, right)
        elif contract.operator in {'add', 'subtract'}: result = _add(left, right, contract.operator)
        elif contract.operator == 'maximum' and left.kind == right.kind == 'state':
            assert left.factor is not None and right.factor is not None
            result = _Value('state', _binary_factor('maximum', left.factor, right.factor), tuple(dict.fromkeys((*left.factor_axes, *right.factor_axes))))
        else: result = _Value('unknown')
    else: result = _Value('unknown')
    return result
def _unary(operator: str, value: _Value, scale: float=1.0, bias: float=0.0) -> _Value:
    """Apply one unary map while retaining only proven separability."""
    if operator == 'copy': result = _copy_affine(value, scale, bias)
    elif value.kind == 'state' and value.factor is not None: result = _Value('state', _unary_factor(operator, value.factor, scale, bias), value.factor_axes)
    elif value.kind == 'residual': result = value
    elif value.kind == 'constant': result = _Value('residual')
    elif operator == 'exp' and value.kind == 'additive' and (value.factor is not None):
        result = _Value('multiplicative', _unary_factor('exp', value.factor, scale, bias), value.factor_axes, value.depends_on_progress)
    elif operator in {'reciprocal', 'square'} and value.kind == 'multiplicative' and (value.factor is not None):
        factor = _unary_factor('reciprocal', value.factor) if operator == 'reciprocal' else _binary_factor('multiply', value.factor, value.factor)
        result = _Value('multiplicative', factor, value.factor_axes, value.depends_on_progress)
    else: result = _Value('unknown')
    return result
def _copy_affine(value: _Value, scale: float, bias: float) -> _Value:
    """Apply an affine identity map."""
    if value.kind == 'state' and value.factor is not None: result = _Value('state', _unary_factor('copy', value.factor, scale, bias), value.factor_axes)
    elif value.kind == 'additive' and value.factor is not None:
        result = _Value('additive', _unary_factor('copy', value.factor, scale, bias), value.factor_axes, value.depends_on_progress)
    elif value.kind in {'residual', 'constant'}: result = _Value('residual', depends_on_progress=value.depends_on_progress)
    elif scale == 1.0 and bias == 0.0: result = value
    else: result = _Value('unknown')
    return result
def _multiply(left: _Value, right: _Value) -> _Value:
    """Multiply values while retaining separable state factors."""
    if 'unknown' in {left.kind, right.kind} or 'additive' in {left.kind, right.kind}: return _Value('unknown')
    states = _factor_states(left.factor) | _factor_states(right.factor); factor: _Factor | None = None
    if states and left.factor is not None and (right.factor is not None):
        if left.factor == _constant_factor(1.0): factor = right.factor
        elif right.factor == _constant_factor(1.0): factor = left.factor
        else: factor = _binary_factor('multiply', left.factor, right.factor)
    elif states: factor = left.factor if _factor_states(left.factor) else right.factor
    axes = tuple(dict.fromkeys((*left.factor_axes, *right.factor_axes))); progress = left.depends_on_progress or right.depends_on_progress
    residual = left.kind in {'residual', 'multiplicative'} or right.kind in {'residual', 'multiplicative'}
    if not states and left.kind == right.kind == 'constant':
        assert left.factor is not None and right.factor is not None; assert left.factor.literal is not None and right.factor.literal is not None
        result = _Value('constant', _constant_factor(left.factor.literal * right.factor.literal))
    elif factor is None: result = _Value('residual', depends_on_progress=progress)
    else: result = _Value('multiplicative' if residual else 'state', factor, axes, progress if residual else False)
    return result
def _add(left: _Value, right: _Value, operator: str) -> _Value:
    """Add or subtract values while retaining state/residual separation."""
    if 'unknown' in {left.kind, right.kind} or 'multiplicative' in {left.kind, right.kind}: return _Value('unknown')
    left_factor = left.factor; right_factor = right.factor; states = _factor_states(left_factor) | _factor_states(right_factor)
    if operator == 'subtract' and right_factor is not None: right_factor = _unary_factor('copy', right_factor, -1.0)
    if states and left_factor is not None and (right_factor is not None): factor = _binary_factor('add', left_factor, right_factor)
    elif states: factor = left_factor if left_factor is not None else right_factor
    else: factor = None
    axes = tuple(dict.fromkeys((*left.factor_axes, *right.factor_axes))); progress = left.depends_on_progress or right.depends_on_progress
    residual = left.kind in {'residual', 'additive', 'constant'} or right.kind in {'residual', 'additive', 'constant'}
    if factor is None: result = _Value('residual', depends_on_progress=progress)
    else: result = _Value('additive' if residual else 'state', factor, axes, progress if residual else False)
    return result
def _reduce(ir: KernelIR, graph: ValueGraph, nid: int, combinator: ReduceCombinator, abstract_axis: str, contribution: _Value, progress_axis: str, stages: tuple[_Stage, ...]) -> tuple[_Value, _Stage | None]:
    """Recognize one recurrence stage or propagate an unrelated reduction."""; block_nid = owning_block(ir.tree, nid); block = ir.tree.block(block_nid)
    concrete_axis = block.axis_map[abstract_axis]; output = graph.outputs[nid]; output_axes = graph.tensor_axes[output]; stage: _Stage | None = None
    if concrete_axis != progress_axis:
        incompatible = contribution.factor is not None and (concrete_axis in contribution.factor_axes or not set(contribution.factor_axes).issubset(output_axes))
        value = _Value('unknown') if incompatible else contribution
    else:
        prior_states = _factor_states(contribution.factor)
        first = not stages and contribution.kind in {'residual', 'constant'} and contribution.depends_on_progress
        later = bool(stages) and combinator.combiner == 'add' and (contribution.kind == 'multiplicative') and (contribution.factor is not None) and bool(prior_states) and prior_states.issubset(range(len(stages))) and (progress_axis not in contribution.factor_axes)
        if first or later:
            stage = _Stage(reducer_leaf=nid, reducer_block=block_nid, state_tensor=output, combinator=combinator, factor=contribution.factor if later else None, factor_axes=contribution.factor_axes if later else ())
            value = _Value('state', _state_factor(len(stages)), output_axes)
        else: value = _Value('unknown')
    return (value, stage)
def _detect_matches(ir: KernelIR, complete: bool) -> list[_Match]:
    """Detect maximal chains or their next independently useful prefix."""; matches: list[_Match] = []
    axes = tuple((axis for axis in _candidate_axes(ir) if all((block == ir.tree.root or _compatible_block(ir, block, axis) for block in ir.tree.blocks()))))
    if not axes: return matches
    graph = build_value_graph(ir)
    for axis in axes:
        evaluation = _evaluate(ir, graph, axis)
        if len(evaluation.stages) < 2: continue
        maximal = _build_match(ir, graph, axis, evaluation)
        if maximal is None: continue
        selected = maximal
        if not complete and len(maximal.stages) > 2:
            prefixes = tuple((_build_match(ir, graph, axis, evaluation, stage_count=count) for count in range(2, len(maximal.stages))))
            if any((prefix is None for prefix in prefixes)): continue
            proven = tuple((cast(_Match, prefix) for prefix in prefixes)); sizes = tuple((size for size in proven[0].chunk_sizes if all((size in match.chunk_sizes for match in (*proven[1:], maximal)))))
            selected = replace(proven[0], chunk_sizes=sizes)
        if selected.chunk_sizes: matches.append(selected)
    return matches
def _candidate_axes(ir: KernelIR) -> tuple[str, ...]:
    """Return concrete axes used by associative reductions."""; axes: set[str] = set()
    for nid in ir.tree.preorder():
        node = ir.tree.data(nid)
        if not isinstance(node, ISANode): continue
        contract = node.op_cls.algebraic_contract(node.kwargs)
        if isinstance(contract, (ReductionContract, BilinearReductionContract)):
            axes.add(ir.tree.block(owning_block(ir.tree, nid)).axis_map[contract.reduction_axis])
    return tuple(sorted(axes))
def _compatible_block(ir: KernelIR, block_nid: int, progress_axis: str) -> bool:
    """Accept canonical blocks and exact non-progress factorizations."""
    if is_canonical_block(ir, block_nid): return True
    chain = block_chain(ir.tree, block_nid)
    if chain is None or not isinstance(chain[-1], ISANode): return False
    block = ir.tree.block(block_nid); leaf = chain[-1]; names = {slot: region.tensor for slot, region in leaf.operand_bindings.items()}
    spec = canonical_spec(ir, leaf.op_cls, names, block.axis_map, leaf.kwargs)
    if spec is None: return False
    substitutions = _iter_substitutions(spec.block, block)
    loops = tuple((item for item in chain[1:-1] if isinstance(item, ForNode)))
    if substitutions is None or not _factored_loops_match(spec.loops, loops, progress_axis): return False
    expected_block = replace(spec.block, iter_values=block.iter_values, reads=tuple((_substitute_region(region, substitutions) for region in spec.block.reads)), writes=tuple((_substitute_region(region, substitutions) for region in spec.block.writes)))
    expected_leaf = replace(spec.leaf, operand_bindings={slot: _substitute_region(region, substitutions) for slot, region in spec.leaf.operand_bindings.items()})
    return replace(block, alloc_buffers=()) == expected_block and leaf == expected_leaf
def _iter_substitutions(canonical: BlockNode, actual: BlockNode) -> dict[str, Expr] | None:
    """Map canonical variables to factored linearized values."""; result: dict[str, Expr] = {}
    valid = canonical.iter_vars == actual.iter_vars and canonical.axis_map == actual.axis_map
    for expected, replacement in zip(canonical.iter_values, actual.iter_values):
        if not valid: break
        if isinstance(expected, Var): result[expected.name] = replacement
        elif expected != replacement: valid = False
    return result if valid else None
def _factored_loops_match(canonical: tuple[ForNode, ...], actual: tuple[ForNode, ...], progress_axis: str) -> bool:
    """Check dense loop groups and forbid progress-axis factorization."""; cursor = 0
    for expected in canonical:
        axis = _loop_axis(expected.loop_var); group: list[ForNode] = []
        while cursor < len(actual) and _loop_axis(actual[cursor].loop_var) == axis: group.append(actual[cursor]); cursor += 1
        product = 1
        for loop in group: product *= loop.extent
        names = [loop.loop_var for loop in group]
        if not group or product != expected.extent or names != [f'i_{axis}_{index}' for index in range(len(group))] or (axis == progress_axis and group != [expected]):
            return False
    return cursor == len(actual)
def _loop_axis(loop_var: str) -> str:
    """Return the concrete axis encoded in a normalized loop variable."""; body = loop_var[2:] if loop_var.startswith('i_') else loop_var
    return body.rsplit('_', 1)[0]
def _substitute_region(region: BufferRegion, substitutions: Mapping[str, Expr]) -> BufferRegion:
    """Substitute variables in one buffer region."""
    return replace(region, ranges=tuple(((substitute(lower, dict(substitutions)), substitute(width, dict(substitutions))) for lower, width in region.ranges)))
def _build_match(ir: KernelIR, graph: ValueGraph, progress_axis: str, evaluation: _Evaluation, stage_count: int | None=None) -> _Match | None:
    """Build the absorbed subgraph and external boundary."""; stages = evaluation.stages if stage_count is None else evaluation.stages[:stage_count]
    stage_leaves = {stage.reducer_leaf for stage in stages}; relevant = _ancestors(graph, stage_leaves)
    absorbed = {nid for nid in relevant if nid in stage_leaves or evaluation.by_leaf[nid].depends_on_progress or bool(_factor_states(evaluation.by_leaf[nid].factor))}
    incremental = stage_count is not None
    for nid in tuple(absorbed): absorbed.update(graph.initializers.get(graph.outputs[nid], ()))
    external_inputs: list[str] = []; external_outputs: list[str] = []
    for nid in graph.leaves:
        if nid not in absorbed: continue
        for slot, tensor in graph.inputs[nid].items():
            if graph.predecessors[nid][slot] not in absorbed and tensor not in external_inputs: external_inputs.append(tensor)
        if any((consumer not in absorbed for consumer in graph.successors[nid])):
            output = graph.outputs[nid]
            if output not in external_outputs: external_outputs.append(output)
    if incremental: external_outputs = [stages[-1].state_tensor]; valid_boundary = len(stages) == stage_count
    else: valid_boundary = external_outputs == [stages[-1].state_tensor]
    sizes = _chunk_sizes(ir, absorbed, progress_axis)
    if not valid_boundary or not sizes: return None
    order = {nid: index for index, nid in enumerate(graph.leaves)}; derivation = tuple(sorted(absorbed, key=order.__getitem__))
    return _Match(progress_axis=progress_axis, progress_extent=ir.axis_extent(progress_axis), stages=stages, derivation_leaves=derivation, absorbed_blocks=tuple((owning_block(ir.tree, nid) for nid in derivation)), external_inputs=tuple(external_inputs), external_outputs=tuple(external_outputs), chunk_sizes=sizes, deferred_factor=None if incremental else _detect_deferred(graph, evaluation), incremental_prefix=incremental)
def _detect_deferred(graph: ValueGraph, evaluation: _Evaluation) -> _Deferred | None:
    """Find a unique broadcast factor movable after the recurrence."""; stage_index = len(evaluation.stages) - 1; final = evaluation.stages[stage_index]
    factors = _flatten_product(final.factor)
    reciprocal = [factor for factor in factors if factor.operator == 'reciprocal' and factor.scale == 1.0 and (factor.bias == 0.0) and (len(factor.operands) == 1) and (factor.operands[0].stage is not None) and (factor.operands[0].stage < stage_index)]
    candidates: list[_Deferred] = []
    if len(factors) > 1 and len(reciprocal) == 1:
        deferred = reciprocal[0]; state = deferred.operands[0].stage; assert state is not None
        remaining = tuple((factor for factor in factors if factor is not deferred)); recurrence = _product_factor(remaining)
        if _positive_sum_stage(graph, evaluation, state):
            producers = [nid for nid in graph.leaves if evaluation.by_leaf[nid].factor == deferred and isinstance(graph.contracts[nid], PointwiseContract) and (cast(PointwiseContract, graph.contracts[nid]).operator == 'reciprocal')]
            candidates = _deferred_candidates(graph, evaluation, stage_index, deferred, recurrence, producers)
    if not candidates and final.factor is not None:
        states = _factor_states(final.factor)
        if states and all((index < stage_index for index in states)):
            producers = [nid for nid in graph.leaves if evaluation.by_leaf[nid].factor == final.factor and (not evaluation.by_leaf[nid].depends_on_progress) and isinstance(graph.contracts[nid], PointwiseContract) and (len(cast(PointwiseContract, graph.contracts[nid]).input_operands) == 1)]
            candidates = _deferred_candidates(graph, evaluation, stage_index, final.factor, None, producers)
    return candidates[0] if len(candidates) == 1 else None
def _deferred_candidates(graph: ValueGraph, evaluation: _Evaluation, stage_index: int, deferred: _Factor, recurrence: _Factor | None, producers: list[int]) -> list[_Deferred]:
    """Return shape-preserving multiply bypasses for candidate producers."""; result: list[_Deferred] = []; final = evaluation.stages[stage_index]
    ancestors = _ancestors(graph, {final.reducer_leaf})
    for producer in producers:
        successors = graph.successors[producer]
        if len(successors) != 1: continue
        combine = successors[0]; contract = graph.contracts[combine]
        if combine not in ancestors or not isinstance(contract, PointwiseContract) or contract.operator != 'multiply' or (len(contract.input_operands) != 2) or (evaluation.by_leaf[combine].factor != final.factor) or (not _transparent_path(graph, combine, final.reducer_leaf)):
            continue
        inputs = graph.inputs[combine]
        slots = [slot for slot in contract.input_operands if graph.predecessors[combine].get(slot) == producer and evaluation.by_tensor[inputs[slot]].factor == deferred]
        if len(slots) != 1: continue
        factor_slot = slots[0]; passthrough = next((slot for slot in contract.input_operands if slot != factor_slot)); source = inputs[passthrough]
        output = graph.outputs[combine]
        if factor_slot in contract.broadcast_operands and passthrough not in contract.broadcast_operands and (graph.tensor_axes[source] == graph.tensor_axes[output]) and (evaluation.by_tensor[source].factor == recurrence):
            result.append(_Deferred(stage_index, deferred, recurrence, producer, combine, passthrough))
    return result
def _positive_sum_stage(graph: ValueGraph, evaluation: _Evaluation, index: int) -> bool:
    """Prove one state is an additive reduction of exponentials."""
    if index < 0 or index >= len(evaluation.stages): return False
    stage = evaluation.stages[index]; reducer = graph.contracts[stage.reducer_leaf]
    if not isinstance(reducer, ReductionContract) or stage.combinator.combiner != 'add': return False
    producer = graph.predecessors[stage.reducer_leaf].get(reducer.input_operand)
    if producer is None: return False
    contract = graph.contracts[producer]; return isinstance(contract, PointwiseContract) and contract.operator == 'exp'
def _flatten_product(factor: _Factor | None) -> tuple[_Factor, ...]:
    """Return ordered leaves of a multiplication factor."""
    if factor is None: return ()
    if factor.operator == 'multiply' and len(factor.operands) == 2: return (*_flatten_product(factor.operands[0]), *_flatten_product(factor.operands[1]))
    return (factor,)
def _product_factor(factors: tuple[_Factor, ...]) -> _Factor | None:
    """Rebuild an ordered product."""; result: _Factor | None = None
    for factor in factors: result = factor if result is None else _binary_factor('multiply', result, factor)
    return result
def _transparent_path(graph: ValueGraph, start: int, final: int) -> bool:
    """Check for one copy/permutation path to a reducer."""; current = start
    while current != final and len(graph.successors[current]) == 1:
        current = graph.successors[current][0]
        if current != final and (not isinstance(graph.contracts[current], (CopyContract, PermutationContract))): return False
    return current == final
def _ancestors(graph: ValueGraph, starts: set[int]) -> set[int]:
    """Return semantic producer ancestors including the starting leaves."""; result = set(starts); stack = list(starts)
    while stack:
        for producer in graph.predecessors[stack.pop()].values():
            if producer is not None and producer not in result: result.add(producer); stack.append(producer)
    return result
def _chunk_sizes(ir: KernelIR, absorbed: set[int], progress_axis: str) -> tuple[int, ...]:
    """Enumerate divisors tileable by every operation in the chain."""; extent = ir.axis_extent(progress_axis); result: list[int] = []
    for size in range(1, extent + 1):
        valid = extent % size == 0
        for nid in absorbed:
            leaf = ir.tree.isa(nid); block = ir.tree.block(owning_block(ir.tree, nid))
            for abstract, concrete in block.axis_map.items():
                if concrete == progress_axis:
                    minimum = leaf.op_cls.MIN_TILE_SIZE.get(abstract, 1); maximum = leaf.op_cls.MAX_TILE_SIZE.get(abstract)
                    tile = size if maximum is None else min(size, maximum); valid = valid and size >= minimum and (size % tile == 0)
        if valid: result.append(size)
    return tuple(result)
_Compiled = tuple[str | None, float | None]
def _compile_factor(builder: OperationBuilder, factor: _Factor, states: Mapping[int, str], stem: str) -> str:
    """Materialize one state factor and return its tensor."""; tensor, _literal = _compile_value(builder, factor, states, stem)
    if tensor is None: raise ValueError(f'factor {factor!r} did not materialize a tensor')
    return tensor
def _compile_value(builder: OperationBuilder, factor: _Factor, states: Mapping[int, str], stem: str) -> _Compiled:
    """Recursively compile a tensor/literal factor."""
    if factor.stage is not None: result = (states[factor.stage], None)
    elif factor.literal is not None: result = (None, factor.literal)
    elif len(factor.operands) == 1:
        operand_factor, scale, bias = _flatten_affine(factor); operand = _compile_factor(builder, operand_factor, states, f'{stem}_arg')
        output = builder.temp(f'{stem}_{factor.operator}', operand); kwargs: dict[str, Any] = {'op': factor.operator}
        if scale != 1.0: kwargs['scale'] = scale
        if bias != 0.0: kwargs['bias'] = bias
        builder.append(NKIActivation, {'data': builder.region(operand), 'dst': builder.region(output)}, kwargs); result = (output, None)
    elif len(factor.operands) == 2:
        left = _compile_value(builder, factor.operands[0], states, f'{stem}_left'); right = _compile_value(builder, factor.operands[1], states, f'{stem}_right')
        result = _compile_binary(builder, factor.operator, left, right, stem)
    else: raise TypeError(f'unsupported factor {factor!r}')
    return result
def _flatten_affine(factor: _Factor) -> tuple[_Factor, float, float]:
    """Fold nested copy factors into one unary operation."""; operand = factor.operands[0]; scale = factor.scale; bias = factor.bias
    while operand.operator == 'copy' and len(operand.operands) == 1: bias = operand.bias * scale + bias; scale *= operand.scale; operand = operand.operands[0]
    return (operand, scale, bias)
def _compile_binary(builder: OperationBuilder, operator: str, left: _Compiled, right: _Compiled, stem: str) -> _Compiled:
    """Compile one binary tensor/literal factor."""; left_tensor, left_literal = left; right_tensor, right_literal = right
    if left_tensor is not None and right_tensor is not None:
        output = builder.temp(stem, left_tensor); _emit_tensor_tensor(builder, left_tensor, right_tensor, output, operator); result = (output, None)
    elif left_tensor is not None and right_literal is not None:
        output = builder.temp(stem, left_tensor); _emit_tensor_scalar(builder, left_tensor, right_literal, output, operator, False); result = (output, None)
    elif right_tensor is not None and left_literal is not None:
        output = builder.temp(stem, right_tensor); _emit_tensor_scalar(builder, right_tensor, left_literal, output, operator, True); result = (output, None)
    elif left_literal is not None and right_literal is not None:
        functions = {'add': lambda a, b: a + b, 'subtract': lambda a, b: a - b, 'multiply': lambda a, b: a * b, 'maximum': max}
        result = (None, float(functions[operator](left_literal, right_literal)))
    else: raise ValueError('binary factor has neither tensor nor literal operands')
    return result
def _compile_correction(builder: OperationBuilder, factor: _Factor, old_states: Mapping[int, str], new_states: Mapping[int, str], stem: str) -> str:
    """Materialize the stable ratio ``factor(new) / factor(old)``."""
    if factor.literal is not None: raise ValueError('constant correction does not materialize a tensor')
    if factor.operator == 'multiply' and len(factor.operands) == 2:
        left = _compile_correction(builder, factor.operands[0], old_states, new_states, f'{stem}_left')
        right = _compile_correction(builder, factor.operands[1], old_states, new_states, f'{stem}_right'); output = builder.temp(stem, left)
        _emit_tensor_tensor(builder, left, right, output, 'multiply'); return output
    if len(factor.operands) == 1 and factor.operator in {'rsqrt', 'exp', 'reciprocal'}:
        operand, scale, bias = _flatten_affine(factor); old = _compile_factor(builder, operand, old_states, f'{stem}_old_arg')
        new = _compile_factor(builder, operand, new_states, f'{stem}_new_arg')
        if factor.operator == 'exp':
            difference = builder.temp(f'{stem}_difference', new); _emit_tensor_tensor(builder, new, old, difference, 'subtract')
            output = builder.temp(stem, difference); kwargs: dict[str, Any] = {'op': 'exp'}
            if scale != 1.0: kwargs['scale'] = scale
            builder.append(NKIActivation, {'data': builder.region(difference), 'dst': builder.region(output)}, kwargs); return output
        old_affine = _emit_affine(builder, old, scale, bias, f'{stem}_old'); new_affine = _emit_affine(builder, new, scale, bias, f'{stem}_new')
        if factor.operator == 'rsqrt':
            old_affine = _emit_unary(builder, old, 'sqrt', scale, bias, f'{stem}_old_sqrt')
            new_affine = _emit_unary(builder, new, 'rsqrt', scale, bias, f'{stem}_new_rsqrt'); output = builder.temp(stem, new_affine)
            _emit_tensor_tensor(builder, new_affine, old_affine, output, 'multiply'); return output
        return _emit_ratio(builder, old_affine, new_affine, stem)
    old = _compile_factor(builder, factor, old_states, f'{stem}_old'); new = _compile_factor(builder, factor, new_states, f'{stem}_new')
    return _emit_ratio(builder, new, old, stem)
def _emit_ratio(builder: OperationBuilder, numerator: str, denominator: str, stem: str) -> str:
    """Materialize one tensor ratio."""; inverse = _emit_unary(builder, denominator, 'reciprocal', 1.0, 0.0, f'{stem}_inverse')
    output = builder.temp(stem, numerator); _emit_tensor_tensor(builder, numerator, inverse, output, 'multiply'); return output
def _emit_unary(builder: OperationBuilder, data: str, operator: str, scale: float, bias: float, stem: str) -> str:
    """Emit one affine activation."""; output = builder.temp(stem, data); kwargs: dict[str, Any] = {'op': operator}
    if scale != 1.0: kwargs['scale'] = scale
    if bias != 0.0: kwargs['bias'] = bias
    builder.append(NKIActivation, {'data': builder.region(data), 'dst': builder.region(output)}, kwargs); return output
def _emit_affine(builder: OperationBuilder, data: str, scale: float, bias: float, stem: str) -> str:
    """Emit a non-identity affine copy."""; return data if scale == 1.0 and bias == 0.0 else _emit_unary(builder, data, 'copy', scale, bias, stem)
def _emit_tensor_tensor(builder: OperationBuilder, left: str, right: str, output: str, operator: str) -> None:
    """Emit one tensor-tensor operation."""
    builder.append(NKITensorTensor, {'data1': builder.region(left), 'data2': builder.region(right), 'dst': builder.region(output)}, {'op': operator})
def _emit_tensor_scalar(builder: OperationBuilder, data: str, operand: float, output: str, operator: str, reverse: bool) -> None:
    """Emit one literal tensor-scalar operation."""; kwargs: dict[str, Any] = {'op0': operator, 'operand0': operand}
    if reverse: kwargs['reverse0'] = True
    builder.append(NKITensorScalar, {'data': builder.region(data), 'dst': builder.region(output)}, kwargs)
@dataclass(frozen=True)
class _Plan:
    """Buffers for one recurrence stage."""; state: str; contribution: str; current: str; raw_contribution: str | None = None
@dataclass
class _Lowering:
    """State shared while cloning one per-chunk derivation."""; ir: KernelIR; match: _Match; graph: ValueGraph; chunk_size: int; progress_index: Expr
    builder: OperationBuilder; scopes: tuple[OperationScope | None, ...]; carry_tensor: str | None = None; outer_loop_var: str | None = None
@dataclass(frozen=True)
class _Prefix:
    """Live recurrence prefix retained for one completion action."""; roots: tuple[int, ...]; added_buffers: tuple[str, ...]; carrier: int; loop: int
    roll_forward: tuple[int, ...]; derivation_leaves: tuple[int, ...]; plans: tuple[_Plan, ...]; scopes: tuple[OperationScope | None, ...]
    regions: tuple[BufferRegion, ...]
@dataclass(frozen=True)
class _Incremental:
    """Guarded metadata for extending a retained prefix one stage at a time."""; complete: _Match; graph: ValueGraph; chunk_size: int; prefix: _Prefix
    remaining: tuple[_Match, ...]
    nodes: tuple[tuple[int, BlockNode | ForNode | ISANode, tuple[int, ...]], ...]; buffers: tuple[Buffer, ...]
def _plan_buffers(ir: KernelIR, match: _Match, graph: ValueGraph, names: NameSupply, preserve_deferred_dtype: bool, separate_final_current: bool) -> tuple[tuple[_Plan, ...], dict[str, Buffer]]:
    """Choose state, contribution, and current buffers."""; plans: list[_Plan] = []; buffers: dict[str, Buffer] = {}; last = len(match.stages) - 1
    for index, stage in enumerate(match.stages):
        deferred = match.deferred_factor is not None and index == match.deferred_factor.stage
        if deferred:
            state = names.fresh(f'{stage.state_tensor}_online_state'); source = ir.buffer(match.external_outputs[0])
            dtype = source.storage_dtype if preserve_deferred_dtype else 'float32'; buffers[state] = replace(source, name=state, location='sbuf', storage_dtype=dtype)
        else:
            state = match.external_outputs[0] if index == last else stage.state_tensor
            buffers[state] = replace(ir.buffer(state), location='sbuf', storage_dtype='float32')
        contribution_leaf = _contribution_leaf(match, graph, index); contribution = graph.outputs[contribution_leaf]
        source = ir.buffer(contribution); raw_contribution: str | None = None
        if source.location == 'psum':
            raw_contribution = names.fresh(f'{stage.state_tensor}_online_partial')
            buffers[raw_contribution] = replace(source, name=raw_contribution, storage_dtype='float32')
            contribution = names.fresh(f'{stage.state_tensor}_online_chunk')
            dtype = source.storage_dtype if deferred and preserve_deferred_dtype else 'float32'
            buffers[contribution] = replace(source, name=contribution, location='sbuf', storage_dtype=dtype)
        elif index != last or contribution == state or contribution_leaf != stage.reducer_leaf:
            contribution = names.fresh(f'{stage.state_tensor}_online_chunk'); source = ir.buffer(graph.outputs[contribution_leaf])
            dtype = source.storage_dtype if deferred and preserve_deferred_dtype else 'float32'
            buffers[contribution] = replace(source, name=contribution, storage_dtype=dtype)
        current = state
        if index != last or separate_final_current:
            current = names.fresh(f'{stage.state_tensor}_online_current'); buffers[current] = replace(buffers[state], name=current)
        plans.append(_Plan(state, contribution, current, raw_contribution))
    output = match.external_outputs[0]
    buffers[output] = replace(ir.buffer(output), location='sbuf', storage_dtype='float32')
    return (tuple(plans), buffers)
def _contribution_leaf(match: _Match, graph: ValueGraph, index: int) -> int:
    """Return the reducer leaf whose output contributes to one stage."""; del graph; return match.stages[index].reducer_leaf
def _localized_buffers(ir: KernelIR, match: _Match, graph: ValueGraph, buffers: Mapping[str, Buffer], chunk_size: int) -> dict[str, Buffer]:
    """Shrink progress-carrying internal buffers to one chunk."""; result = dict(buffers)
    internal = {region.tensor for nid in match.derivation_leaves if nid in ir.tree.graph for region in ir.tree.isa(nid).operand_bindings.values() if region.tensor not in match.external_inputs and region.tensor not in match.external_outputs}
    for name in internal:
        axes = graph.tensor_axes.get(name, ())
        if name in result and match.progress_axis in axes:
            shape = list(result[name].shape); shape[axes.index(match.progress_axis)] = chunk_size; result[name] = replace(result[name], shape=tuple(shape))
        if name in result and len(result[name].shape) == 1: result[name] = replace(result[name], storage_dtype='float32')
    return result
def _stage_region(ir: KernelIR, graph: ValueGraph, stage: _Stage) -> BufferRegion:
    """Return one reducer state-output region."""; contract = graph.contracts[stage.reducer_leaf]
    return ir.tree.isa(stage.reducer_leaf).operand_bindings[contract.output_operand]
def _stage_scope(ir: KernelIR, graph: ValueGraph, stage: _Stage, progress_axis: str) -> OperationScope:
    """Return mapped loop geometry with the progress axis removed."""; block = ir.tree.block(stage.reducer_block)
    chain = block_chain(ir.tree, stage.reducer_block)
    if chain is None: raise ValueError(f'online reducer block {stage.reducer_block} is not a canonical chain')
    retained = [(iter_var, value) for iter_var, value in zip(block.iter_vars, block.iter_values) if iter_var.axis != progress_axis]
    values = tuple((value for _iter_var, value in retained)); loop_vars = {name for value in values for name in to_affine(value) if name is not None}
    loops = tuple((item for item in chain[1:-1] if isinstance(item, ForNode) and item.loop_var in loop_vars)); axes = graph.tensor_axes[stage.state_tensor]
    scoped = replace(block, iter_vars=tuple((iter_var for iter_var, _value in retained)), iter_values=values, reads=(), writes=(), alloc_buffers=(), axis_map={abstract: concrete for abstract, concrete in zip(('P', 'F'), axes)})
    return OperationScope(scoped, loops)
def _mapped(match: _Match, ir: KernelIR) -> bool:
    """Return whether any recurrence state spans partition tiles."""; names = [stage.state_tensor for stage in match.stages[:-1]] + [match.external_outputs[0]]
    return any((ir.buffer(name).shape[0] > PARTITION_DIM for name in names))
def _stage_regions(plans: tuple[_Plan, ...], regions: tuple[BufferRegion, ...]) -> dict[str, BufferRegion]:
    """Map every stage buffer to its reducer output region."""; result: dict[str, BufferRegion] = {}
    for plan, region in zip(plans, regions):
        for tensor in (plan.state, plan.contribution, plan.current): result[tensor] = replace(region, tensor=tensor)
        if plan.raw_contribution is not None: result[plan.raw_contribution] = replace(region, tensor=plan.raw_contribution)
    return result
def _derive(context: _Lowering, plans: tuple[_Plan, ...], initial_remap: Mapping[str, str] | None=None, selected: frozenset[int] | None=None, roll_forward: bool=True) -> None:
    """Clone per-chunk work and append recurrence updates."""; remap = dict(initial_remap or {})
    contributions = [_contribution_leaf(context.match, context.graph, index) for index in range(len(plans))]
    stage_by_leaf = {leaf: index for index, leaf in enumerate(contributions)}
    overrides = {context.graph.outputs[leaf]: plan.raw_contribution or plan.contribution for leaf, plan in zip(contributions, plans)}; leaves = context.match.derivation_leaves
    if selected is not None: leaves = tuple((nid for nid in leaves if nid in selected))
    for nid in leaves:
        deferred = context.match.deferred_factor
        if deferred is not None and nid == deferred.producer_leaf: continue
        if deferred is not None and nid == deferred.bypass_leaf:
            source = context.graph.inputs[nid][deferred.passthrough_operand]; remap[context.graph.outputs[nid]] = remap.get(source, source); continue
        output = context.graph.outputs[nid]; _clone_block(context, nid, remap, overrides.get(output)); stage_index = stage_by_leaf.get(nid)
        if stage_index is not None:
            plan = plans[stage_index]
            if plan.raw_contribution is not None:
                context.builder.scope = context.scopes[stage_index]; _emit_copy(context.builder, plan.raw_contribution, plan.contribution)
            _update_stage(context, stage_index, plans); remap[context.match.stages[stage_index].state_tensor] = plans[stage_index].current
    if roll_forward:
        for index, plan in enumerate(plans[:-1]): context.builder.scope = context.scopes[index]; _emit_copy(context.builder, plan.current, plan.state)
def _clone_block(context: _Lowering, nid: int, remap: Mapping[str, str], output_override: str | None) -> None:
    """Clone one canonical operation into the recurrence loop."""; tree = context.ir.tree; old_block_nid = owning_block(tree, nid)
    chain = block_chain(tree, old_block_nid)
    if chain is None: raise ValueError(f'block {old_block_nid} is not a canonical chain')
    old_block = tree.block(old_block_nid); leaf = tree.isa(nid); contract = context.graph.contracts[nid]; progress_vars: set[str] = set()
    values = list(old_block.iter_values)
    for index, iter_var in enumerate(old_block.iter_vars):
        if iter_var.axis == context.match.progress_axis: progress_vars.update((name for name in to_affine(values[index]) if name is not None))
    loop_var, tile_size, trip_count = _progress_tiling(context, leaf, progress_vars); bindings: dict[str, BufferRegion] = {}
    for slot, region in leaf.operand_bindings.items():
        tensor = remap.get(region.tensor, region.tensor)
        if slot == contract.output_operand and output_override is not None: tensor = output_override
        bindings[slot] = _localized_region(context, region, tensor, loop_var, tile_size, trip_count)
    reads, writes = _access_regions(leaf.op_cls, bindings, leaf.kwargs)
    for index, iter_var in enumerate(old_block.iter_vars):
        if iter_var.axis == context.match.progress_axis:
            progress: Expr = context.progress_index
            if trip_count > 1: assert loop_var is not None; progress = Add(left=Mul(left=progress, right=Const(value=trip_count)), right=Var(name=loop_var))
            values[index] = progress
    block = replace(old_block, iter_values=tuple(values), reads=reads, writes=writes, alloc_buffers=())
    parent = tree.add_node(block, parent=context.builder.parent)
    for item in chain[1:-1]:
        if isinstance(item, ForNode) and item.loop_var == context.outer_loop_var: continue
        if isinstance(item, ForNode) and item.loop_var in progress_vars:
            if trip_count > 1: parent = tree.add_node(replace(item, extent=trip_count), parent=parent)
        else: parent = tree.add_node(item, parent=parent)
    tree.add_node(ISANode(op_cls=leaf.op_cls, operand_bindings=bindings, kwargs=dict(leaf.kwargs)), parent=parent)
def _progress_tiling(context: _Lowering, leaf: ISANode, progress_vars: set[str]) -> tuple[str | None, int, int]:
    """Derive the operation tile and local trip count."""; sizes: set[int] = set()
    for region in leaf.operand_bindings.values():
        axes = context.graph.tensor_axes[region.tensor]
        if context.match.progress_axis in axes:
            width = region.ranges[axes.index(context.match.progress_axis)][1]
            if not isinstance(width, Const): raise ValueError('online progress width must be constant')
            sizes.add(min(width.value, context.chunk_size))
    if not sizes: return (None, context.chunk_size, 1)
    if len(sizes) != 1: raise ValueError(f'inconsistent online progress tiles {sorted(sizes)}')
    tile = next(iter(sizes))
    if context.chunk_size % tile: raise ValueError(f'chunk size {context.chunk_size} is not divisible by tile {tile}')
    trips = context.chunk_size // tile
    if trips > 1 and len(progress_vars) != 1: raise ValueError('multi-tile online operation requires one progress loop')
    return (next(iter(progress_vars)) if trips > 1 else None, tile, trips)
def _localized_region(context: _Lowering, region: BufferRegion, tensor: str, loop_var: str | None, tile_size: int, trip_count: int) -> BufferRegion:
    """Retarget one region and localize its progress dimension."""; axes = context.graph.tensor_axes[region.tensor]; ranges = list(region.ranges)
    if context.match.progress_axis in axes:
        index = axes.index(context.match.progress_axis); buffer = context.ir.buffer(region.tensor)
        local_tile: Expr = Var(name=loop_var) if loop_var is not None else Const(value=0)
        local_element: Expr = Mul(left=local_tile, right=Const(value=tile_size)) if loop_var is not None else Const(value=0)
        if region.tensor in context.match.external_inputs:
            if buffer.location == 'shared_hbm' or index > 0:
                lower = Add(left=Mul(left=context.progress_index, right=Const(value=context.chunk_size)), right=local_element)
            else: lower = Add(left=Mul(left=context.progress_index, right=Const(value=trip_count)), right=local_tile)
        else: lower = local_tile if buffer.location != 'shared_hbm' and index == 0 else local_element
        if isinstance(lower, Add) and isinstance(lower.right, Const) and (lower.right.value == 0): lower = lower.left
        ranges[index] = (lower, Const(value=tile_size))
    return BufferRegion(tensor=tensor, ranges=tuple(ranges))
def _access_regions(op_cls: type[NKIOp], bindings: Mapping[str, BufferRegion], kwargs: Mapping[str, Any]) -> tuple[tuple[BufferRegion, ...], tuple[BufferRegion, ...]]:
    """Derive reads and writes from operation metadata."""; reads: list[BufferRegion] = []; writes: list[BufferRegion] = []
    rmw = op_cls.rmw_operands(dict(kwargs))
    for slot, region in bindings.items():
        if slot in op_cls.INPUT_OPERANDS: reads.append(region)
        elif slot in rmw: reads.append(region); writes.append(region)
        else: writes.append(region)
    return (tuple(reads), tuple(writes))
def _emit_copy(builder: OperationBuilder, source: str, destination: str) -> int:
    """Emit one explicit tensor copy."""; return builder.append(NKITensorCopy, {'src': builder.region(source), 'dst': builder.region(destination)}, {})
def _emit_initializer(builder: OperationBuilder, tensor: str, value: float) -> int:
    """Emit one full-region initializer."""; return builder.append(NKIMemset, {'dst': builder.region(tensor)}, {'value': value})
def _update_stage(context: _Lowering, index: int, plans: tuple[_Plan, ...]) -> None:
    """Append one recurrence combiner and any HBM carry traffic."""; stage = context.match.stages[index]; plan = plans[index]
    context.builder.scope = context.scopes[index]; final = index == len(plans) - 1; single_chunk = context.chunk_size == context.match.progress_extent
    if single_chunk: _emit_copy(context.builder, plan.contribution, plan.current)
    elif final and context.carry_tensor is not None:
        context.builder.append(NKILoad, {'src': context.builder.region(context.carry_tensor), 'dst': context.builder.region(plan.state)}, {})
    if not single_chunk and index == 0: _emit_tensor_tensor(context.builder, plan.state, plan.contribution, plan.current, stage.combinator.combiner)
    elif not single_chunk:
        factor = stage.factor; deferred = context.match.deferred_factor
        if deferred is not None and index == deferred.stage: factor = deferred.recurrence_factor
        if deferred is not None and index == deferred.stage and (factor is None):
            context.builder.append(NKIScalarTensorTensor, {'data': context.builder.region(plan.state), 'operand1': context.builder.region(plan.contribution), 'dst': context.builder.region(plan.current)}, {'op0': 'multiply', 'operand0': 1.0, 'op1': 'add'})
        elif factor is None: raise ValueError(f'online stage {index} has no correction factor')
        else:
            update_scope = context.builder.scope; state_indices = sorted(_factor_states(factor))
            if state_indices: context.builder.scope = context.scopes[state_indices[-1]]
            old = {prior: prior_plan.state for prior, prior_plan in enumerate(plans[:index])}
            new = {prior: prior_plan.current for prior, prior_plan in enumerate(plans[:index])}
            correction = _compile_correction(context.builder, factor, old, new, f'stage{index}_correction'); context.builder.scope = update_scope
            context.builder.append(NKIScalarTensorTensor, {'data': context.builder.region(plan.state), 'operand0': context.builder.region(correction), 'operand1': context.builder.region(plan.contribution), 'dst': context.builder.region(plan.current)}, {'op0': 'multiply', 'op1': 'add'})
    if final and context.carry_tensor is not None:
        context.builder.append(NKIStore, {'src': context.builder.region(plan.current), 'dst': context.builder.region(context.carry_tensor)}, {})
def _root_insertion(ir: KernelIR, match: _Match, extra_blocks: frozenset[int]=frozenset(), extra_leaves: frozenset[int]=frozenset(), retain_old: bool=False) -> int | None:
    """Return a root slot preserving every dependency boundary edge."""; tree = ir.tree; absorbed_leaves = {*match.derivation_leaves, *extra_leaves}
    absorbed_blocks = {*match.absorbed_blocks, *extra_blocks}; roots = tree.children(tree.root)
    remaining = roots if retain_old else [root for root in roots if root not in absorbed_blocks]
    positions = {block: index for index, block in enumerate(remaining)}; lower = 0; upper = len(remaining)
    for producer, consumer in ir.dependency.graph.edges:
        producer_absorbed = producer in absorbed_leaves; consumer_absorbed = consumer in absorbed_leaves
        if producer_absorbed == consumer_absorbed: continue
        outside = consumer if producer_absorbed else producer; block = owning_block(tree, outside)
        if block not in positions: return None
        if producer_absorbed: upper = min(upper, positions[block])
        else: lower = max(lower, positions[block] + 1)
    if lower > upper: return None
    if retain_old: return lower
    first = min((roots.index(block) for block in absorbed_blocks)); preferred = sum((roots.index(block) < first for block in remaining))
    return min(max(preferred, lower), upper)
def _set_root_children(tree: KernelTree, removed: tuple[int, ...], added: tuple[int, ...], index: int) -> None:
    """Replace arbitrary root children at one insertion slot."""; roots = tree.children(tree.root); removed_set = set(removed)
    if not removed_set.issubset(roots): raise ValueError(f'online-fusion blocks are not root children: {removed_set - set(roots)}')
    remaining = [root for root in roots if root not in removed_set and root not in added]; order = remaining[:index] + list(added) + remaining[index:]
    for root in roots: tree.graph.remove_edge(tree.root, root)
    for root in order: tree.graph.add_edge(tree.root, root)
def _seed_buffers(ir: KernelIR, buffers: Mapping[str, Buffer], prunable: frozenset[str]) -> None:
    """Update declarations and attach missing live buffers at the root."""
    touched = {region.tensor for nid in ir.tree.preorder() if isinstance(ir.tree.data(nid), ISANode) for region in ir.tree.isa(nid).operand_bindings.values()}
    missing = touched - set(buffers) - set(ir.param_buffers)
    if missing: raise AssertionError(f'online lowering has no buffers for {sorted(missing)}')
    declared: set[str] = set()
    for block_nid in ir.tree.blocks():
        block = ir.tree.block(block_nid)
        allocations = tuple((buffers.get(buffer.name, buffer) for buffer in block.alloc_buffers if buffer.name not in prunable or buffer.name in touched))
        declared.update((buffer.name for buffer in allocations))
        if allocations != block.alloc_buffers: ir.tree.graph.nodes[block_nid]['data'] = replace(block, alloc_buffers=allocations)
    root = ir.tree.block(ir.tree.root)
    additions = tuple((buffer for name, buffer in buffers.items() if name in touched and name not in ir.param_buffers and (name not in declared)))
    if additions: ir.tree.graph.nodes[ir.tree.root]['data'] = replace(root, alloc_buffers=(*root.alloc_buffers, *additions))
def _match_tensors(match: _Match, graph: ValueGraph) -> frozenset[str]:
    """Return every tensor crossing the matched derivation."""; tensors = {stage.state_tensor for stage in match.stages}
    for nid in match.derivation_leaves: tensors.add(graph.outputs[nid]); tensors.update(graph.inputs[nid].values())
    return frozenset(tensors)
def _carrier(tree: KernelTree, match: _Match, chunk_size: int, parent: int | None) -> tuple[int, int]:
    """Append one sequential carrier block and loop."""; loop_var = f'i_{match.progress_axis}_online'
    block = BlockNode(iter_vars=(IterVar(axis=match.progress_axis, dom=(0, match.progress_extent), role=AxisRole.SEQUENTIAL),), iter_values=(Var(name=loop_var),), reads=(), writes=(), alloc_buffers=())
    carrier = tree.add_node(block, parent=parent); loop = tree.add_node(ForNode(loop_var=loop_var, extent=match.progress_extent // chunk_size), parent=carrier)
    return (carrier, loop)
def _region_shape(region: BufferRegion) -> tuple[int, ...]:
    """Return the constant shape represented by a region."""; shape: list[int] = []
    for _lower, width in region.ranges:
        if not isinstance(width, Const): raise ValueError('online recurrence widths must be constant')
        shape.append(width.value)
    return tuple(shape)
def _hbm_region(region: BufferRegion, tensor: str) -> BufferRegion:
    """Map an on-chip partition tile to element-addressed HBM."""; ranges = list(region.ranges); lower, width = ranges[0]
    ranges[0] = (Mul(left=lower, right=Const(value=PARTITION_DIM)), width); return BufferRegion(tensor=tensor, ranges=tuple(ranges))
def _group_scopes(ir: KernelIR, match: _Match, graph: ValueGraph, scopes: tuple[OperationScope, ...]) -> tuple[OperationScope, ...] | None:
    """Return scopes when every mapped block shares an explicit outer split."""; output_axes = graph.tensor_axes[match.external_outputs[0]]
    if not output_axes: return None
    axis = output_axes[0]; values = [value for iter_var, value in zip(scopes[-1].block.iter_vars, scopes[-1].block.iter_values) if iter_var.axis == axis]
    variables = set(to_affine(values[0])) if len(values) == 1 else set(); variables.discard(None)
    loops = [loop for loop in scopes[-1].loops if loop.loop_var in variables]
    if len(loops) < 2: return None
    outer = loops[0]
    for nid in match.derivation_leaves:
        block_nid = owning_block(ir.tree, nid); block = ir.tree.block(block_nid)
        mapped_values = [value for iter_var, value in zip(block.iter_vars, block.iter_values) if iter_var.axis == axis]
        if not mapped_values: continue
        chain = block_chain(ir.tree, block_nid); variables = set(to_affine(mapped_values[0])) if len(mapped_values) == 1 else set()
        matching = [] if chain is None else [item for item in chain[1:-1] if isinstance(item, ForNode) and item.loop_var == outer.loop_var and (item.extent == outer.extent)]
        count = 0 if chain is None else sum((isinstance(item, ForNode) and item.loop_var in variables for item in chain[1:-1]))
        if len(mapped_values) != 1 or len(matching) != 1 or count < 2: return None
    return scopes
def _can_lower(ir: KernelIR, match: _Match, chunk_size: int, prefix: bool=False) -> bool:
    """Return whether ordinary-IR lowering supports one option."""; valid = chunk_size in match.chunk_sizes and len(match.external_outputs) == 1
    valid = valid and bool(match.absorbed_blocks); valid = valid and all((ir.tree.parent(block) == ir.tree.root for block in match.absorbed_blocks))
    if prefix:
        valid = valid and match.incremental_prefix and (chunk_size < match.progress_extent); states = [ir.buffer(stage.state_tensor) for stage in match.stages]
        valid = valid and all((buffer.location in {'sbuf', 'psum'} and len(buffer.shape) == 1 and (buffer.shape[0] >= PARTITION_DIM) and (buffer.shape[0] % PARTITION_DIM == 0) for buffer in states))
        return valid and _root_insertion(ir, match, retain_old=True) is not None
    names = [stage.state_tensor for stage in match.stages[:-1]] + [match.external_outputs[0]]; states = [ir.buffer(name) for name in names]
    valid = valid and all((buffer.location in {'sbuf', 'psum'} and len(buffer.shape) in {1, 2} and (buffer.shape[0] >= PARTITION_DIM) and (buffer.shape[0] % PARTITION_DIM == 0) for buffer in states))
    valid = valid and all((len(ir.buffer(stage.state_tensor).shape) == 1 for stage in match.stages[:-1])); graph = build_value_graph(ir)
    if valid and chunk_size == match.progress_extent:
        if not _mapped(match, ir): return False
        scopes = tuple((_stage_scope(ir, graph, stage, match.progress_axis) for stage in match.stages))
        valid = _group_scopes(ir, match, graph, scopes) is not None
    return valid and _root_insertion(ir, match) is not None
def _new_context(ir: KernelIR, match: _Match, graph: ValueGraph, chunk_size: int, parent: int, buffers: dict[str, Buffer], names: NameSupply, regions: dict[str, BufferRegion], scopes: tuple[OperationScope | None, ...], progress: Expr, carry: str | None=None) -> _Lowering:
    """Construct one recurrence lowering context."""; builder = OperationBuilder(ir.tree, parent, buffers, names, regions)
    return _Lowering(ir, match, graph, chunk_size, progress, builder, scopes, carry)
def _lower_complete(ir: KernelIR, match: _Match, chunk_size: int) -> None:
    """Lower one complete proven recurrence."""
    if not _can_lower(ir, match, chunk_size): raise ValueError(f'online-fusion match {match.match_id} cannot lower with chunk_size={chunk_size}')
    graph = build_value_graph(ir)
    if _mapped(match, ir):
        scopes = tuple((_stage_scope(ir, graph, stage, match.progress_axis) for stage in match.stages)); grouped = _group_scopes(ir, match, graph, scopes)
        if grouped is not None: _lower_grouped(ir, match, graph, grouped, chunk_size)
        else: _lower_hbm(ir, match, graph, scopes, chunk_size)
    else: _lower_tile(ir, match, graph, chunk_size)
def _lower_tile(ir: KernelIR, match: _Match, graph: ValueGraph, chunk_size: int) -> None:
    """Lower a recurrence whose states fit one partition tile."""; original = ir.all_buffers(); names = NameSupply(set(original))
    plans, added = _plan_buffers(ir, match, graph, names, False, False); buffers = _localized_buffers(ir, match, graph, original, chunk_size)
    buffers.update(added); init = OperationBuilder(ir.tree, None, buffers, names)
    roots = [_emit_initializer(init, plan.state, stage.combinator.identity) for stage, plan in zip(match.stages, plans)]
    carrier, loop = _carrier(ir.tree, match, chunk_size, None)
    context = _new_context(ir, match, graph, chunk_size, loop, buffers, names, {}, tuple((None for _stage in match.stages)), Var(name=f'i_{match.progress_axis}_online'))
    _derive(context, plans); roots.extend((carrier, *_append_epilogue(context, plans))); insertion = _root_insertion(ir, match); assert insertion is not None
    old = match.absorbed_blocks; _set_root_children(ir.tree, old, tuple(roots), insertion)
    for block in old: ir.tree.graph.remove_nodes_from({block, *ir.tree.descendants(block)})
    _seed_buffers(ir, buffers, _match_tensors(match, graph)); finalize_rewrite(ir)
def _lower_hbm(ir: KernelIR, match: _Match, graph: ValueGraph, scopes: tuple[OperationScope, ...], chunk_size: int) -> None:
    """Lower mapped states through an fp32 HBM carry."""; original = ir.all_buffers()
    names = NameSupply(set(original)); plans, added = _plan_buffers(ir, match, graph, names, False, False)
    buffers = _localized_buffers(ir, match, graph, original, chunk_size); buffers.update(added)
    stage_regions = tuple((_stage_region(ir, graph, stage) for stage in match.stages)); regions = _stage_regions(plans, stage_regions); output = ir.return_name
    carry = names.fresh(f'{output}_online_carry'); buffers[carry] = replace(buffers[output], name=carry, dtype='float32', storage_dtype='float32')
    regions[carry] = _hbm_region(stage_regions[-1], carry)
    init = OperationBuilder(ir.tree, None, buffers, names, regions); roots: list[int] = []
    for index, (stage, plan) in enumerate(zip(match.stages[:-1], plans[:-1])):
        init.scope = scopes[index]; roots.append(_emit_initializer(init, plan.state, stage.combinator.identity))
    final_region = stage_regions[-1]; zero = names.fresh(f'{match.external_outputs[0]}_online_zero'); shape = _region_shape(final_region)
    buffers[zero] = Buffer(name=zero, shape=shape, dtype='float32', location='sbuf', storage_dtype='float32')
    regions[zero] = BufferRegion(tensor=zero, ranges=tuple(((Const(value=0), Const(value=extent)) for extent in shape))); init.scope = scopes[-1]
    roots.append(_emit_initializer(init, zero, 0.0)); roots.append(init.append(NKIStore, {'src': regions[zero], 'dst': regions[carry]}, {}))
    carrier, loop = _carrier(ir.tree, match, chunk_size, None)
    context = _new_context(ir, match, graph, chunk_size, loop, buffers, names, regions, scopes, Var(name=f'i_{match.progress_axis}_online'), carry)
    _derive(context, plans); roots.extend((carrier, *_append_epilogue(context, plans)))
    insertion = _root_insertion(ir, match); assert insertion is not None
    old = match.absorbed_blocks; _set_root_children(ir.tree, old, tuple(roots), insertion)
    for block in old: ir.tree.graph.remove_nodes_from({block, *ir.tree.descendants(block)})
    _seed_buffers(ir, buffers, _match_tensors(match, graph)); finalize_rewrite(ir)
def _lower_grouped(ir: KernelIR, match: _Match, graph: ValueGraph, scopes: tuple[OperationScope, ...], chunk_size: int) -> None:
    """Keep one explicit mapped group on chip across the progress loop."""
    original = ir.all_buffers(); names = NameSupply(set(original)); plans, added = _plan_buffers(ir, match, graph, names, True, False)
    buffers = _localized_buffers(ir, match, graph, original, chunk_size); buffers.update(added)
    stage_regions = tuple((_stage_region(ir, graph, stage) for stage in match.stages)); regions = _stage_regions(plans, stage_regions)
    group = ir.tree.add_node(BlockNode(iter_vars=(), iter_values=(), reads=(), writes=(), alloc_buffers=()))
    body = ir.tree.add_node(BlockNode(iter_vars=(), iter_values=(), reads=(), writes=(), alloc_buffers=()), parent=group)
    init = OperationBuilder(ir.tree, body, buffers, names, regions)
    if chunk_size < match.progress_extent:
        for index, (stage, plan) in enumerate(zip(match.stages, plans)):
            init.scope = scopes[index]; _emit_initializer(init, plan.state, stage.combinator.identity)
        _carrier_nid, parent = _carrier(ir.tree, match, chunk_size, body); progress: Expr = Var(name=f'i_{match.progress_axis}_online')
    else: parent = body; progress = Const(value=0)
    context = _new_context(ir, match, graph, chunk_size, parent, buffers, names, regions, scopes, progress); _derive(context, plans)
    if match.deferred_factor is not None:
        deferred = match.deferred_factor; context.builder.parent = body; states = {index: plan.state for index, plan in enumerate(plans[:deferred.stage])}
        state_indices = sorted(_factor_states(deferred.factor)); context.builder.scope = scopes[state_indices[-1]]
        factor = _compile_factor(context.builder, deferred.factor, states, 'deferred_factor'); source = match.external_outputs[0]
        context.builder.regions[source] = replace(context.builder.region(plans[deferred.stage].current), tensor=source)
        context.builder.scope = scopes[deferred.stage]; _emit_scaled(context.builder, plans[deferred.stage].current, factor, source)
    insertion = _root_insertion(ir, match); assert insertion is not None
    old = match.absorbed_blocks; _set_root_children(ir.tree, old, (group,), insertion)
    for block in old: ir.tree.graph.remove_nodes_from({block, *ir.tree.descendants(block)})
    _seed_buffers(ir, buffers, _match_tensors(match, graph)); finalize_rewrite(ir)
def _emit_scaled(builder: OperationBuilder, data: str, factor: str, output: str) -> None:
    """Emit one broadcast multiplication."""
    builder.append(NKITensorScalar, {'data': builder.region(data), 'operand0': builder.region(factor), 'dst': builder.region(output)}, {'op0': 'multiply'})
def _append_epilogue(context: _Lowering, plans: tuple[_Plan, ...]) -> list[int]:
    """Apply one deferred final factor after the sequential loop."""; deferred = context.match.deferred_factor
    if deferred is None: return []
    states = {index: plan.state for index, plan in enumerate(plans[:deferred.stage])}
    if context.carry_tensor is None:
        root = context.builder.tree.add_node(BlockNode(iter_vars=(), iter_values=(), reads=(), writes=(), alloc_buffers=()))
        builder = OperationBuilder(context.builder.tree, root, context.builder.buffers, context.builder.names, context.builder.regions)
        factor = _compile_factor(builder, deferred.factor, states, 'deferred_factor')
        _emit_scaled(builder, plans[deferred.stage].current, factor, context.match.external_outputs[0]); return [root]
    scope = context.scopes[deferred.stage]
    if scope is None: raise AssertionError('mapped deferred factor has no scope')
    root, parent, substitutions = _scope_carrier(context.builder.tree, scope); final_region = context.builder.region(plans[deferred.stage].current)
    shape = _region_shape(final_region); local_ranges = tuple(((Const(value=0), Const(value=extent)) for extent in shape))
    numerator = context.builder.names.fresh('online_deferred_numerator'); normalized = context.match.external_outputs[0]
    template = context.builder.buffers[plans[deferred.stage].current]
    regions = {tensor: _substitute_region(region, substitutions) for tensor, region in context.builder.regions.items()}
    context.builder.buffers[numerator] = replace(template, name=numerator, shape=shape, location='sbuf', storage_dtype='float32', versions=1, list_len=1)
    regions[numerator] = BufferRegion(tensor=numerator, ranges=local_ranges)
    regions[normalized] = replace(_substitute_region(final_region, substitutions), tensor=normalized)
    output_scope = OperationScope(context.builder.tree.block(root), ())
    builder = OperationBuilder(context.builder.tree, parent, context.builder.buffers, context.builder.names, regions, output_scope, True)
    builder.append(NKILoad, {'src': builder.region(context.carry_tensor), 'dst': builder.region(numerator)}, {}, scope=None)
    state_indices = sorted(_factor_states(deferred.factor)); factor_scope = context.scopes[state_indices[-1]]
    if factor_scope is None: raise AssertionError('mapped deferred factor has no state scope')
    factor_block = replace(factor_scope.block, iter_values=tuple((substitute(value, substitutions) for value in factor_scope.block.iter_values)), reads=(), writes=(), alloc_buffers=())
    builder.scope = OperationScope(factor_block, ()); factor = _compile_factor(builder, deferred.factor, states, 'deferred_factor')
    builder.scope = output_scope; _emit_scaled(builder, numerator, factor, normalized); return [root]
def _scope_carrier(tree: KernelTree, scope: OperationScope) -> tuple[int, int, dict[str, Expr]]:
    """Clone one mapped scope with fresh loop variables."""; used = {item.loop_var for nid in tree.graph if isinstance((item := tree.data(nid)), ForNode)}
    substitutions: dict[str, Expr] = {}; loops: list[ForNode] = []
    for loop in scope.loops:
        stem = f'{loop.loop_var}_online_epilogue'; name = stem; suffix = 1
        while name in used: name = f'{stem}_{suffix}'; suffix += 1
        used.add(name); substitutions[loop.loop_var] = Var(name=name); loops.append(replace(loop, loop_var=name))
    block = replace(scope.block, iter_values=tuple((substitute(value, substitutions) for value in scope.block.iter_values)), reads=(), writes=(), alloc_buffers=())
    root = tree.add_node(block); parent = root
    for loop in loops: parent = tree.add_node(loop, parent=parent)
    return (root, parent, substitutions)
def _rewrite_reducer_as_map(ir: KernelIR, stage: _Stage, contract: ReductionContract) -> None:
    """Retain a dual-output reducer's mapped pointwise result."""; leaf = ir.tree.isa(stage.reducer_leaf); mapped = contract.mapped_output_operand
    if mapped is None: raise ValueError('mapped reducer has no mapped output')
    if leaf.op_cls is NKIActivationReduce: op_cls = NKIActivation; inputs = ('data', 'bias')
    elif leaf.op_cls is NKITensorScalarReduce: op_cls = NKITensorScalar; inputs = ('data', 'operand0')
    else: raise ValueError(f'unsupported dual-output reducer {leaf.op_cls.__name__}')
    bindings = {slot: region for slot, region in leaf.operand_bindings.items() if slot in inputs or slot == mapped}; bindings['dst'] = bindings.pop(mapped)
    kwargs = {name: value for name, value in leaf.kwargs.items() if name != 'reduce_op'}; reads, writes = _access_regions(op_cls, bindings, kwargs)
    block = ir.tree.block(stage.reducer_block); axis = block.axis_map[contract.reduction_axis]
    iter_vars = tuple((replace(var, role=AxisRole.PARALLEL) if var.axis == axis else var for var in block.iter_vars))
    ir.tree.graph.nodes[stage.reducer_block]['data'] = replace(block, iter_vars=iter_vars, reads=reads, writes=writes)
    ir.tree.graph.nodes[stage.reducer_leaf]['data'] = ISANode(op_cls=op_cls, operand_bindings=bindings, kwargs=kwargs)
def _lower_prefix(ir: KernelIR, match: _Match, complete: _Match, chunk_size: int) -> _Prefix:
    """Emit a live two-stage recurrence while retaining the suffix."""
    if not _can_lower(ir, match, chunk_size, prefix=True): raise ValueError(f'online-fusion prefix {match.match_id} cannot lower')
    graph = build_value_graph(ir); original = ir.all_buffers(); names = NameSupply(set(original))
    plans, added = _plan_buffers(ir, match, graph, names, False, True); buffers = dict(original); buffers.update(added); mapped = _mapped(match, ir)
    prefix_scopes = tuple((_stage_scope(ir, graph, stage, match.progress_axis) for stage in match.stages)) if mapped else tuple((None for _stage in match.stages))
    complete_scopes = tuple((_stage_scope(ir, graph, stage, complete.progress_axis) for stage in complete.stages)) if mapped else tuple((None for _stage in complete.stages))
    complete_regions = tuple((_stage_region(ir, graph, stage) for stage in complete.stages))
    regions = _stage_regions(plans, tuple((_stage_region(ir, graph, stage) for stage in match.stages))) if mapped else {}
    init = OperationBuilder(ir.tree, None, buffers, names, regions); roots: list[int] = []
    for index, (stage, plan) in enumerate(zip(match.stages, plans)):
        init.scope = prefix_scopes[index]; roots.append(_emit_initializer(init, plan.state, stage.combinator.identity))
    carrier, loop = _carrier(ir.tree, match, chunk_size, None)
    context = _new_context(ir, match, graph, chunk_size, loop, buffers, names, regions, prefix_scopes, Var(name=f'i_{match.progress_axis}_online'))
    _derive(context, plans, roll_forward=False); rolls: list[int] = []
    for index, plan in enumerate(plans): context.builder.scope = prefix_scopes[index]; rolls.append(_emit_copy(context.builder, plan.current, plan.state))
    roots.append(carrier); insertion = _root_insertion(ir, match, retain_old=True); assert insertion is not None
    _set_root_children(ir.tree, (), tuple(roots), insertion); removed: list[int] = []
    for stage in match.stages:
        contract = graph.contracts[stage.reducer_leaf]
        if isinstance(contract, ReductionContract) and contract.mapped_output_operand is not None: _rewrite_reducer_as_map(ir, stage, contract)
        else: removed.append(stage.reducer_block)
    _set_root_children(ir.tree, tuple(removed), (), 0)
    for block in removed: ir.tree.graph.remove_nodes_from({block, *ir.tree.descendants(block)})
    _seed_buffers(ir, buffers, frozenset()); finalize_rewrite(ir)
    added_names = tuple((name for name in buffers if name not in original and name in ir.all_buffers()))
    return _Prefix(tuple(roots), added_names, carrier, loop, tuple(rolls), match.derivation_leaves, plans, complete_scopes, complete_regions)
def _extend_prefix(ir: KernelIR, match: _Match, graph: ValueGraph, prefix: _Prefix, chunk_size: int) -> _Prefix:
    """Append one non-final recurrence stage while retaining the suffix."""
    index = len(prefix.plans)
    if not match.incremental_prefix or len(match.stages) != index + 1: raise ValueError('incremental extension requires exactly one stage')
    original = ir.all_buffers(); names = NameSupply(set(original)); stage = match.stages[index]; state = stage.state_tensor
    state_buffer = replace(ir.buffer(state), location='sbuf', storage_dtype='float32'); contribution_leaf = _contribution_leaf(match, graph, index)
    contribution_source = ir.buffer(graph.outputs[contribution_leaf]); raw_contribution: str | None = None
    if contribution_source.location == 'psum':
        raw_contribution = names.fresh(f'{state}_online_partial'); contribution = names.fresh(f'{state}_online_chunk')
        contribution_buffer = replace(contribution_source, name=contribution, location='sbuf', storage_dtype='float32')
    else: contribution = names.fresh(f'{state}_online_chunk'); contribution_buffer = replace(contribution_source, name=contribution, storage_dtype='float32')
    current = names.fresh(f'{state}_online_current'); plan = _Plan(state, contribution, current, raw_contribution); plans = (*prefix.plans, plan)
    buffers = _localized_buffers(ir, match, graph, original, chunk_size); buffers[state] = state_buffer; buffers[current] = replace(state_buffer, name=current); buffers[contribution] = contribution_buffer
    if raw_contribution is not None: buffers[raw_contribution] = replace(contribution_source, name=raw_contribution, storage_dtype='float32')
    mapped = _mapped(match, ir); regions = _stage_regions(plans, prefix.regions) if mapped else {}
    init = OperationBuilder(ir.tree, None, buffers, names, regions, prefix.scopes[index]); init_root = _emit_initializer(init, state, stage.combinator.identity)
    builder = OperationBuilder(ir.tree, None, buffers, names, regions); context = _Lowering(ir, match, graph, chunk_size, Var(name=ir.tree.loop(prefix.loop).loop_var), builder, prefix.scopes)
    remap = {prior.state_tensor: prior_plan.current for prior, prior_plan in zip(match.stages[:index], prefix.plans)}
    selected = frozenset(match.derivation_leaves) - frozenset(prefix.derivation_leaves); previous = set(ir.tree.graph); _derive(context, plans, remap, selected, roll_forward=False)
    suffix = [nid for nid in ir.tree.graph if nid not in previous and ir.tree.parent(nid) is None and isinstance(ir.tree.data(nid), BlockNode)]
    context.builder.scope = prefix.scopes[index]; roll = _emit_copy(context.builder, plan.current, plan.state)
    _insert_detached_roots(ir.tree, prefix.carrier, [init_root], before=True); _insert_children_before(ir.tree, prefix.loop, prefix.roll_forward, [*suffix, roll])
    contract = graph.contracts[stage.reducer_leaf]
    if isinstance(contract, ReductionContract) and contract.mapped_output_operand is not None: _rewrite_reducer_as_map(ir, stage, contract)
    else: _set_root_children(ir.tree, (stage.reducer_block,), (), 0); ir.tree.graph.remove_nodes_from({stage.reducer_block, *ir.tree.descendants(stage.reducer_block)})
    _seed_buffers(ir, buffers, frozenset()); finalize_rewrite(ir)
    added = (*prefix.added_buffers, *(name for name in buffers if name not in original and name in ir.all_buffers()))
    return _Prefix(prefix.roots, added, prefix.carrier, prefix.loop, (roll, *prefix.roll_forward), match.derivation_leaves, plans, prefix.scopes, prefix.regions)
def _complete_prefix(ir: KernelIR, match: _Match, graph: ValueGraph, prefix: _Prefix, chunk_size: int) -> None:
    """Append the final stage to one retained recurrence prefix."""
    if len(match.stages) != len(prefix.plans) + 1: raise ValueError('incremental completion requires exactly one stage')
    all_buffers = ir.all_buffers(); names = NameSupply(set(all_buffers)); final_stage = match.stages[-1]; source = ir.buffer(match.external_outputs[0])
    deferred = match.deferred_factor is not None and match.deferred_factor.stage == len(match.stages) - 1
    state = names.fresh(f'{final_stage.state_tensor}_online_state') if deferred else match.external_outputs[0]
    state_buffer = replace(source, name=state, location='sbuf', storage_dtype='float32'); contribution_leaf = _contribution_leaf(match, graph, len(match.stages) - 1)
    contribution_source = ir.buffer(graph.outputs[contribution_leaf]); raw_contribution: str | None = None
    if contribution_source.location == 'psum':
        raw_contribution = names.fresh(f'{final_stage.state_tensor}_online_partial')
        contribution = names.fresh(f'{final_stage.state_tensor}_online_chunk')
        contribution_buffer = replace(contribution_source, name=contribution, location='sbuf', storage_dtype='float32')
    else:
        contribution = names.fresh(f'{final_stage.state_tensor}_online_chunk')
        contribution_buffer = replace(contribution_source, name=contribution, storage_dtype='float32')
    final_plan = _Plan(state, contribution, state, raw_contribution); plans = (*prefix.plans, final_plan); buffers = _localized_buffers(ir, match, graph, all_buffers, chunk_size)
    buffers[state] = state_buffer; buffers[contribution] = contribution_buffer
    if raw_contribution is not None: buffers[raw_contribution] = replace(contribution_source, name=raw_contribution, storage_dtype='float32')
    mapped = _mapped(match, ir)
    buffers[match.external_outputs[0]] = replace(source, location='sbuf', storage_dtype='float32')
    regions = _stage_regions(plans, prefix.regions) if mapped else {}; carry: str | None = None
    init_roots: list[int] = []
    if mapped:
        output = ir.return_name; carry = names.fresh(f'{output}_online_carry')
        buffers[carry] = replace(buffers[output], name=carry, dtype='float32', storage_dtype='float32'); final_region = prefix.regions[-1]
        regions[carry] = _hbm_region(final_region, carry)
        zero = names.fresh(f'{match.external_outputs[0]}_online_zero'); shape = _region_shape(final_region)
        buffers[zero] = Buffer(name=zero, shape=shape, dtype='float32', location='sbuf', storage_dtype='float32')
        regions[zero] = BufferRegion(tensor=zero, ranges=tuple(((Const(value=0), Const(value=extent)) for extent in shape)))
        init = OperationBuilder(ir.tree, None, buffers, names, regions, prefix.scopes[-1]); init_roots.append(_emit_initializer(init, zero, 0.0))
        init_roots.append(init.append(NKIStore, {'src': regions[zero], 'dst': regions[carry]}, {}))
    else: init = OperationBuilder(ir.tree, None, buffers, names); init_roots.append(_emit_initializer(init, state, final_stage.combinator.identity))
    builder = OperationBuilder(ir.tree, None, buffers, names, regions)
    context = _Lowering(ir, match, graph, chunk_size, Var(name=ir.tree.loop(prefix.loop).loop_var), builder, prefix.scopes, carry)
    remap = {stage.state_tensor: plan.current for stage, plan in zip(match.stages[:-1], prefix.plans)}
    selected = frozenset(match.derivation_leaves) - frozenset(prefix.derivation_leaves); previous = set(ir.tree.graph)
    _derive(context, plans, remap, selected, roll_forward=False)
    suffix = [nid for nid in ir.tree.graph if nid not in previous and ir.tree.parent(nid) is None and isinstance(ir.tree.data(nid), BlockNode)]
    epilogue = _append_epilogue(context, plans); _insert_detached_roots(ir.tree, prefix.carrier, init_roots, before=True)
    _insert_children_before(ir.tree, prefix.loop, prefix.roll_forward, suffix); _insert_detached_roots(ir.tree, prefix.carrier, epilogue, before=False)
    old = [block for block in match.absorbed_blocks if block in ir.tree.graph]
    roots = ir.tree.children(ir.tree.root)
    for block in old:
        if block in roots: ir.tree.graph.remove_edge(ir.tree.root, block)
        ir.tree.graph.remove_nodes_from({block, *ir.tree.descendants(block)})
    _seed_buffers(ir, buffers, _match_tensors(match, graph)); finalize_rewrite(ir)
def _insert_detached_roots(tree: KernelTree, anchor: int, blocks: list[int], before: bool) -> None:
    """Attach detached roots immediately before or after one anchor."""
    if not blocks: return
    roots = tree.children(tree.root); index = roots.index(anchor) + (0 if before else 1); order = roots[:index] + blocks + roots[index:]
    for root in roots: tree.graph.remove_edge(tree.root, root)
    for root in order: tree.graph.add_edge(tree.root, root)
def _insert_children_before(tree: KernelTree, parent: int, anchors: tuple[int, ...], blocks: list[int]) -> None:
    """Attach detached child blocks before a contiguous anchor sequence."""
    if not blocks: return
    children = tree.children(parent); index = children.index(anchors[0])
    if tuple(children[index:index + len(anchors)]) != anchors: raise ValueError('online roll-forward blocks are no longer contiguous')
    order = children[:index] + blocks + children[index:]
    for child in children: tree.graph.remove_edge(parent, child)
    for child in order: tree.graph.add_edge(parent, child)
def _matching_complete(ir: KernelIR, prefix: _Match) -> _Match:
    """Return the unique maximal chain extending a prefix."""
    candidates = [match for match in _detect_matches(ir, complete=True) if match.progress_axis == prefix.progress_axis and match.stages[:len(prefix.stages)] == prefix.stages and set(match.chunk_sizes) & set(prefix.chunk_sizes)]
    if len(candidates) != 1: raise TransformLegalityError(f'online-fusion prefix {prefix.match_id} has {len(candidates)} complete extensions')
    return candidates[0]
def _continuations(ir: KernelIR, prefix: _Match, complete: _Match) -> tuple[_Match, ...]:
    """Build every one-stage extension from the same contract evaluation."""; graph = build_value_graph(ir); evaluation = _evaluate(ir, graph, prefix.progress_axis); result: list[_Match] = []
    for count in range(len(prefix.stages) + 1, len(complete.stages)):
        match = _build_match(ir, graph, prefix.progress_axis, evaluation, stage_count=count)
        if match is None: raise TransformLegalityError(f'online-fusion chain has no valid {count}-stage prefix')
        result.append(match)
    return (*result, complete)
def _capture_incremental(ir: KernelIR, complete: _Match, graph: ValueGraph, chunk_size: int, prefix: _Prefix, remaining: tuple[_Match, ...]) -> _Incremental:
    """Capture a conservative structural guard for completion."""
    nodes = tuple(((nid, copy.deepcopy(ir.tree.data(nid)), tuple(ir.tree.children(nid))) for nid in sorted(ir.tree.graph) if nid != ir.tree.root))
    buffers = tuple((copy.deepcopy(buffer) for buffer in ir.all_buffers().values())); return _Incremental(complete, graph, chunk_size, prefix, remaining, nodes, buffers)
def _incremental_state(ir: KernelIR) -> _Incremental | None:
    """Return validated incremental metadata from the root."""; value = ir.tree.block(ir.tree.root).annotations.get(_INCREMENTAL_ANNOTATION)
    if value is not None and (not isinstance(value, _Incremental)): raise ValueError(f'malformed {_INCREMENTAL_ANNOTATION} annotation')
    return value
def _incremental_intact(ir: KernelIR, state: _Incremental) -> bool:
    """Return whether completion can consume the retained prefix."""; expected = {nid: (payload, children) for nid, payload, children in state.nodes}
    actual = set(ir.tree.graph) - {ir.tree.root}; intact = actual == set(expected)
    if intact: intact = all((ir.tree.data(nid) == payload and tuple(ir.tree.children(nid)) == children for nid, (payload, children) in expected.items()))
    if intact: buffers = ir.all_buffers(); intact = all((buffers.get(buffer.name) == buffer for buffer in state.buffers))
    return intact
@dataclass(frozen=True)
class OnlineFusionOption(TransformOption):
    """One proven recurrence and sequential chunk size."""; match_id: tuple[str, tuple[int, ...]]; chunk_size: int
class OnlineFusion(Transform[OnlineFusionOption]):
    """Rewrite one algebraically separable reduction chain into online form."""
    def analyze(self, ir: KernelIR) -> list[OnlineFusionOption]:
        """Enumerate contract-proven and lowering-supported options."""; state = _incremental_state(ir)
        if state is not None: options = [OnlineFusionOption(state.remaining[0].match_id, state.chunk_size)] if state.remaining and _incremental_intact(ir, state) else []
        else:
            options = [OnlineFusionOption(match.match_id, chunk_size) for match in _detect_matches(ir, complete=False) for chunk_size in match.chunk_sizes if _can_lower(ir, match, chunk_size, prefix=match.incremental_prefix)]
        return options
    def apply(self, ir: KernelIR, option: OnlineFusionOption) -> KernelIR:
        """Re-check one option, deep-copy, and lower its recurrence."""; state = _incremental_state(ir)
        if state is not None:
            if not state.remaining or option.match_id != state.remaining[0].match_id or option.chunk_size != state.chunk_size or (not _incremental_intact(ir, state)):
                raise TransformLegalityError(f'illegal OnlineFusion completion option: {option}')
            result = copy.deepcopy(ir); copied = _incremental_state(result); assert copied is not None
            next_match = copied.remaining[0]; final = len(copied.remaining) == 1; prefix = copied.prefix
            if final: _complete_prefix(result, next_match, copied.graph, copied.prefix, option.chunk_size)
            else: prefix = _extend_prefix(result, next_match, copied.graph, copied.prefix, option.chunk_size)
            root = result.tree.block(result.tree.root); annotations = dict(root.annotations)
            if final: del annotations[_INCREMENTAL_ANNOTATION]
            else: annotations[_INCREMENTAL_ANNOTATION] = _capture_incremental(result, copied.complete, copied.graph, copied.chunk_size, prefix, copied.remaining[1:])
            result.tree.graph.nodes[result.tree.root]['data'] = replace(root, annotations=annotations); return result
        matches = {match.match_id: match for match in _detect_matches(ir, complete=False)}; match = matches.get(option.match_id)
        if match is None or not _can_lower(ir, match, option.chunk_size, prefix=match.incremental_prefix):
            raise TransformLegalityError(f'illegal OnlineFusion option: {option}')
        result = copy.deepcopy(ir); copied_matches = {candidate.match_id: candidate for candidate in _detect_matches(result, complete=False)}
        copied_match = copied_matches[option.match_id]
        if copied_match.incremental_prefix:
            complete = _matching_complete(result, copied_match); graph = build_value_graph(result)
            remaining = _continuations(result, copied_match, complete)
            prefix = _lower_prefix(result, copied_match, complete, option.chunk_size)
            state = _capture_incremental(result, complete, graph, option.chunk_size, prefix, remaining); root = result.tree.block(result.tree.root)
            annotations = dict(root.annotations); annotations[_INCREMENTAL_ANNOTATION] = state
            result.tree.graph.nodes[result.tree.root]['data'] = replace(root, annotations=annotations)
        else: _lower_complete(result, copied_match, option.chunk_size)
        return result
__all__ = ['OnlineFusion', 'OnlineFusionOption']
# fmt: on
