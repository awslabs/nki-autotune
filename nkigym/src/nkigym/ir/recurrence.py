"""Contract-driven recurrence analysis and structural IR lowering helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from numbers import Real
from typing import cast

from nkigym.ir.arith.expr import Expr, Var, substitute
from nkigym.ir.ir import KernelIR
from nkigym.ir.tree import BlockNode, BufferRegion, ForNode, ISANode
from nkigym.ops.base import (
    BilinearReductionContract,
    CopyContract,
    InitializerContract,
    OperatorContract,
    PermutationContract,
    PointwiseContract,
    PointwiseSequenceContract,
    ReduceCombinator,
    ReductionContract,
)
from nkigym.ops.base import _add_recurrence_values as _add
from nkigym.ops.base import _apply_recurrence_unary as _unary
from nkigym.ops.base import _copy_recurrence_affine as _copy_affine
from nkigym.ops.base import _evaluate_pointwise_contract as _pointwise
from nkigym.ops.base import _flatten_recurrence_product as _flatten_product
from nkigym.ops.base import _multiply_recurrence_values as _multiply
from nkigym.ops.base import _recurrence_constant_factor as _constant_factor
from nkigym.ops.base import _recurrence_factor_states as _factor_states
from nkigym.ops.base import _recurrence_product_factor as _product_factor
from nkigym.ops.base import _recurrence_state_factor as _state_factor
from nkigym.ops.base import _RecurrenceFactor as _Factor
from nkigym.ops.base import _RecurrenceValue as _Value
from nkigym.transforms.helper.canonical_rewrite import block_chain, canonical_spec, is_canonical_block, owning_block
from nkigym.transforms.helper.value_graph import ValueGraph, contract_input_operands


@dataclass(frozen=True)
class _Stage:
    """One ordered reduction state."""

    reducer_leaf: int
    reducer_block: int
    state_tensor: str
    combinator: ReduceCombinator
    factor: _Factor | None
    factor_axes: tuple[str, ...]


@dataclass(frozen=True)
class _Deferred:
    """One factor applied after the sequential recurrence."""

    stage: int
    factor: _Factor
    recurrence_factor: _Factor | None
    producer_leaf: int
    bypass_leaf: int
    passthrough_operand: str


@dataclass(frozen=True)
class _Match:
    """A proven recurrence chain and exact rewrite boundary."""

    progress_axis: str
    progress_extent: int
    stages: tuple[_Stage, ...]
    derivation_leaves: tuple[int, ...]
    absorbed_blocks: tuple[int, ...]
    external_inputs: tuple[str, ...]
    external_outputs: tuple[str, ...]
    chunk_sizes: tuple[int, ...]
    deferred_factor: _Deferred | None
    incremental_prefix: bool = False

    @property
    def match_id(self) -> tuple[str, tuple[int, ...]]:
        """Return the stable public option identity."""
        return (self.progress_axis, tuple((stage.reducer_leaf for stage in self.stages)))


@dataclass(frozen=True)
class _Evaluation:
    """Algebraic interpretation for one candidate progress axis."""

    stages: tuple[_Stage, ...]
    by_tensor: Mapping[str, _Value]
    by_leaf: Mapping[int, _Value]


def _evaluate(ir: KernelIR, graph: ValueGraph, progress_axis: str) -> _Evaluation:
    """Propagate state/chunk separation through operation contracts."""
    by_tensor: dict[str, _Value] = {}
    by_leaf: dict[int, _Value] = {}
    stages: list[_Stage] = []
    for nid in graph.leaves:
        leaf = ir.tree.isa(nid)
        contract = graph.contracts[nid]
        values = {
            slot: _operand_value(leaf, slot, contract, graph, by_tensor, progress_axis)
            for slot in contract_input_operands(contract)
            if _operand_available(leaf, contract, slot)
        }
        value, stage, mapped = _transfer(ir, graph, nid, contract, values, progress_axis, tuple(stages))
        if isinstance(contract, ReductionContract) and contract.mapped_output_operand is not None:
            region = leaf.operand_bindings.get(contract.mapped_output_operand)
            if region is None or mapped is None:
                raise ValueError(f"{leaf.op_cls.__name__} has no bound mapped output")
            by_tensor[region.tensor] = mapped
        by_tensor[graph.outputs[nid]] = value
        by_leaf[nid] = value
        if stage is not None:
            stages.append(stage)
    return _Evaluation(tuple(stages), by_tensor, by_leaf)


def _operand_available(leaf: ISANode, contract: OperatorContract, slot: str) -> bool:
    """Return whether an operand is a tensor binding or independent literal."""
    bias = contract.bias_operand if isinstance(contract, (PointwiseContract, ReductionContract)) else None
    return slot in leaf.operand_bindings or (slot != bias and slot in leaf.kwargs)


def _operand_value(
    leaf: ISANode,
    slot: str,
    contract: OperatorContract,
    graph: ValueGraph,
    values: Mapping[str, _Value],
    progress_axis: str,
) -> _Value:
    """Resolve one tensor or literal operand."""
    del contract
    region = leaf.operand_bindings.get(slot)
    if region is not None:
        result = values.get(region.tensor)
        if result is None:
            result = _Value("residual", depends_on_progress=progress_axis in graph.tensor_axes[region.tensor])
    else:
        literal = leaf.kwargs.get(slot)
        result = (
            _Value("constant", _constant_factor(float(literal))) if isinstance(literal, Real) else _Value("unknown")
        )
    return result


def _transfer(
    ir: KernelIR,
    graph: ValueGraph,
    nid: int,
    contract: OperatorContract,
    inputs: Mapping[str, _Value],
    progress_axis: str,
    stages: tuple[_Stage, ...],
) -> tuple[_Value, _Stage | None, _Value | None]:
    """Apply one contract's separation transfer rule."""
    stage: _Stage | None = None
    mapped: _Value | None = None
    if isinstance(contract, PointwiseContract):
        value = _pointwise(contract, inputs)
    elif isinstance(contract, (CopyContract, PermutationContract)):
        value = inputs[contract.input_operand]
    elif isinstance(contract, ReductionContract):
        mapped = _copy_affine(inputs[contract.input_operand], contract.scale, contract.bias)
        if contract.bias_operand is not None and contract.bias_operand in inputs:
            mapped = _add(mapped, inputs[contract.bias_operand], "add")
        if contract.map_operator != "copy":
            mapped = _unary(contract.map_operator, mapped)
        value, stage = _reduce(
            ir, graph, nid, contract.combinator, contract.reduction_axis, mapped, progress_axis, stages
        )
    elif isinstance(contract, BilinearReductionContract):
        contribution = _multiply(inputs[contract.left_operand], inputs[contract.right_operand])
        value, stage = _reduce(
            ir, graph, nid, contract.combinator, contract.reduction_axis, contribution, progress_axis, stages
        )
    elif isinstance(contract, InitializerContract):
        value = _Value("constant", _constant_factor(contract.value))
    elif isinstance(contract, PointwiseSequenceContract):
        value = _Value("unknown")
    else:
        raise TypeError(f"unsupported contract {type(contract).__name__}")
    return (value, stage, mapped)


def _reduce(
    ir: KernelIR,
    graph: ValueGraph,
    nid: int,
    combinator: ReduceCombinator,
    abstract_axis: str,
    contribution: _Value,
    progress_axis: str,
    stages: tuple[_Stage, ...],
) -> tuple[_Value, _Stage | None]:
    """Recognize one recurrence stage or propagate an unrelated reduction."""
    block_nid = owning_block(ir.tree, nid)
    block = ir.tree.block(block_nid)
    concrete_axis = block.axis_map[abstract_axis]
    output = graph.outputs[nid]
    output_axes = graph.tensor_axes[output]
    stage: _Stage | None = None
    if concrete_axis != progress_axis:
        incompatible = contribution.factor is not None and (
            concrete_axis in contribution.factor_axes or not set(contribution.factor_axes).issubset(output_axes)
        )
        value = _Value("unknown") if incompatible else contribution
    else:
        prior_states = _factor_states(contribution.factor)
        first = not stages and contribution.kind in {"residual", "constant"} and contribution.depends_on_progress
        later = (
            bool(stages)
            and combinator.combiner == "add"
            and (contribution.kind == "multiplicative")
            and (contribution.factor is not None)
            and bool(prior_states)
            and prior_states.issubset(range(len(stages)))
            and (progress_axis not in contribution.factor_axes)
        )
        if first or later:
            stage = _Stage(
                reducer_leaf=nid,
                reducer_block=block_nid,
                state_tensor=output,
                combinator=combinator,
                factor=contribution.factor if later else None,
                factor_axes=contribution.factor_axes if later else (),
            )
            value = _Value("state", _state_factor(len(stages)), output_axes)
        else:
            value = _Value("unknown")
    return (value, stage)


def _compatible_block(ir: KernelIR, block_nid: int, progress_axis: str) -> bool:
    """Accept canonical blocks and exact non-progress factorizations."""
    if is_canonical_block(ir, block_nid):
        return True
    chain = block_chain(ir.tree, block_nid)
    if chain is None or not isinstance(chain[-1], ISANode):
        return False
    block = ir.tree.block(block_nid)
    leaf = chain[-1]
    names = {slot: region.tensor for slot, region in leaf.operand_bindings.items()}
    spec = canonical_spec(ir, leaf.op_cls, names, block.axis_map, leaf.kwargs)
    if spec is None:
        return False
    substitutions = _iter_substitutions(spec.block, block)
    loops = tuple((item for item in chain[1:-1] if isinstance(item, ForNode)))
    if substitutions is None or not _factored_loops_match(spec.loops, loops, progress_axis):
        return False
    expected_block = replace(
        spec.block,
        iter_values=block.iter_values,
        reads=tuple((_substitute_region(region, substitutions) for region in spec.block.reads)),
        writes=tuple((_substitute_region(region, substitutions) for region in spec.block.writes)),
    )
    expected_leaf = replace(
        spec.leaf,
        operand_bindings={
            slot: _substitute_region(region, substitutions) for slot, region in spec.leaf.operand_bindings.items()
        },
    )
    return replace(block, alloc_buffers=()) == expected_block and leaf == expected_leaf


def _iter_substitutions(canonical: BlockNode, actual: BlockNode) -> dict[str, Expr] | None:
    """Map canonical variables to factored linearized values."""
    result: dict[str, Expr] = {}
    valid = canonical.iter_vars == actual.iter_vars and canonical.axis_map == actual.axis_map
    for expected, replacement in zip(canonical.iter_values, actual.iter_values):
        if not valid:
            break
        if isinstance(expected, Var):
            result[expected.name] = replacement
        elif expected != replacement:
            valid = False
    return result if valid else None


def _factored_loops_match(canonical: tuple[ForNode, ...], actual: tuple[ForNode, ...], progress_axis: str) -> bool:
    """Check dense loop groups and forbid progress-axis factorization."""
    cursor = 0
    for expected in canonical:
        axis = _loop_axis(expected.loop_var)
        group: list[ForNode] = []
        while cursor < len(actual) and _loop_axis(actual[cursor].loop_var) == axis:
            group.append(actual[cursor])
            cursor += 1
        product = 1
        for loop in group:
            product *= loop.extent
        names = [loop.loop_var for loop in group]
        if (
            not group
            or product != expected.extent
            or names != [f"i_{axis}_{index}" for index in range(len(group))]
            or (axis == progress_axis and group != [expected])
        ):
            return False
    return cursor == len(actual)


def _loop_axis(loop_var: str) -> str:
    """Return the concrete axis encoded in a normalized loop variable."""
    return (loop_var[2:] if loop_var.startswith("i_") else loop_var).rsplit("_", 1)[0]


def _substitute_region(region: BufferRegion, substitutions: Mapping[str, Expr]) -> BufferRegion:
    """Substitute variables in one buffer region."""
    return replace(
        region,
        ranges=tuple(
            (
                (substitute(lower, dict(substitutions)), substitute(width, dict(substitutions)))
                for lower, width in region.ranges
            )
        ),
    )


def _build_match(
    ir: KernelIR, graph: ValueGraph, progress_axis: str, evaluation: _Evaluation, stage_count: int | None = None
) -> _Match | None:
    """Build the absorbed subgraph and external boundary."""
    stages = evaluation.stages if stage_count is None else evaluation.stages[:stage_count]
    stage_leaves = {stage.reducer_leaf for stage in stages}
    relevant = _ancestors(graph, stage_leaves)
    absorbed = {
        nid
        for nid in relevant
        if nid in stage_leaves
        or evaluation.by_leaf[nid].depends_on_progress
        or bool(_factor_states(evaluation.by_leaf[nid].factor))
    }
    incremental = stage_count is not None
    for nid in tuple(absorbed):
        absorbed.update(graph.initializers.get(graph.outputs[nid], ()))
    external_inputs: list[str] = []
    external_outputs: list[str] = []
    for nid in graph.leaves:
        if nid not in absorbed:
            continue
        for slot, tensor in graph.inputs[nid].items():
            if graph.predecessors[nid][slot] not in absorbed and tensor not in external_inputs:
                external_inputs.append(tensor)
        if any((consumer not in absorbed for consumer in graph.successors[nid])):
            output = graph.outputs[nid]
            if output not in external_outputs:
                external_outputs.append(output)
    if incremental:
        external_outputs = [stages[-1].state_tensor]
        valid_boundary = len(stages) == stage_count
    else:
        valid_boundary = external_outputs == [stages[-1].state_tensor]
    sizes = _chunk_sizes(ir, absorbed, progress_axis)
    if not valid_boundary or not sizes:
        return None
    order = {nid: index for index, nid in enumerate(graph.leaves)}
    derivation = tuple(sorted(absorbed, key=order.__getitem__))
    return _Match(
        progress_axis=progress_axis,
        progress_extent=ir.axis_extent(progress_axis),
        stages=stages,
        derivation_leaves=derivation,
        absorbed_blocks=tuple((owning_block(ir.tree, nid) for nid in derivation)),
        external_inputs=tuple(external_inputs),
        external_outputs=tuple(external_outputs),
        chunk_sizes=sizes,
        deferred_factor=None if incremental else _detect_deferred(graph, evaluation),
        incremental_prefix=incremental,
    )


def _detect_deferred(graph: ValueGraph, evaluation: _Evaluation) -> _Deferred | None:
    """Find a unique broadcast factor movable after the recurrence."""
    stage_index = len(evaluation.stages) - 1
    final = evaluation.stages[stage_index]
    factors = _flatten_product(final.factor)
    reciprocal = [
        factor
        for factor in factors
        if factor.operator == "reciprocal"
        and factor.scale == 1.0
        and (factor.bias == 0.0)
        and (len(factor.operands) == 1)
        and (factor.operands[0].stage is not None)
        and (factor.operands[0].stage < stage_index)
    ]
    candidates: list[_Deferred] = []
    if len(factors) > 1 and len(reciprocal) == 1:
        deferred = reciprocal[0]
        state = deferred.operands[0].stage
        assert state is not None
        remaining = tuple((factor for factor in factors if factor is not deferred))
        recurrence = _product_factor(remaining)
        if _positive_sum_stage(graph, evaluation, state):
            producers = [
                nid
                for nid in graph.leaves
                if evaluation.by_leaf[nid].factor == deferred
                and isinstance(graph.contracts[nid], PointwiseContract)
                and (cast(PointwiseContract, graph.contracts[nid]).operator == "reciprocal")
            ]
            candidates = _deferred_candidates(graph, evaluation, stage_index, deferred, recurrence, producers)
    if not candidates and final.factor is not None:
        states = _factor_states(final.factor)
        if states and all((index < stage_index for index in states)):
            producers = [
                nid
                for nid in graph.leaves
                if evaluation.by_leaf[nid].factor == final.factor
                and (not evaluation.by_leaf[nid].depends_on_progress)
                and isinstance(graph.contracts[nid], PointwiseContract)
                and (len(cast(PointwiseContract, graph.contracts[nid]).input_operands) == 1)
            ]
            candidates = _deferred_candidates(graph, evaluation, stage_index, final.factor, None, producers)
    return candidates[0] if len(candidates) == 1 else None


def _deferred_candidates(
    graph: ValueGraph,
    evaluation: _Evaluation,
    stage_index: int,
    deferred: _Factor,
    recurrence: _Factor | None,
    producers: list[int],
) -> list[_Deferred]:
    """Return shape-preserving multiply bypasses for candidate producers."""
    result: list[_Deferred] = []
    final = evaluation.stages[stage_index]
    ancestors = _ancestors(graph, {final.reducer_leaf})
    for producer in producers:
        successors = graph.successors[producer]
        if len(successors) != 1:
            continue
        combine = successors[0]
        contract = graph.contracts[combine]
        if (
            combine not in ancestors
            or not isinstance(contract, PointwiseContract)
            or contract.operator != "multiply"
            or (len(contract.input_operands) != 2)
            or (evaluation.by_leaf[combine].factor != final.factor)
            or (not _transparent_path(graph, combine, final.reducer_leaf))
        ):
            continue
        inputs = graph.inputs[combine]
        slots = [
            slot
            for slot in contract.input_operands
            if graph.predecessors[combine].get(slot) == producer
            and evaluation.by_tensor[inputs[slot]].factor == deferred
        ]
        if len(slots) != 1:
            continue
        factor_slot = slots[0]
        passthrough = next((slot for slot in contract.input_operands if slot != factor_slot))
        source = inputs[passthrough]
        output = graph.outputs[combine]
        if (
            factor_slot in contract.broadcast_operands
            and passthrough not in contract.broadcast_operands
            and (graph.tensor_axes[source] == graph.tensor_axes[output])
            and (evaluation.by_tensor[source].factor == recurrence)
        ):
            result.append(_Deferred(stage_index, deferred, recurrence, producer, combine, passthrough))
    return result


def _positive_sum_stage(graph: ValueGraph, evaluation: _Evaluation, index: int) -> bool:
    """Prove one state is an additive reduction of exponentials."""
    if index < 0 or index >= len(evaluation.stages):
        return False
    stage = evaluation.stages[index]
    reducer = graph.contracts[stage.reducer_leaf]
    if not isinstance(reducer, ReductionContract) or stage.combinator.combiner != "add":
        return False
    producer = graph.predecessors[stage.reducer_leaf].get(reducer.input_operand)
    if producer is None:
        return False
    contract = graph.contracts[producer]
    return isinstance(contract, PointwiseContract) and contract.operator == "exp"


def _transparent_path(graph: ValueGraph, start: int, final: int) -> bool:
    """Check for one copy/permutation path to a reducer."""
    current = start
    while current != final and len(graph.successors[current]) == 1:
        current = graph.successors[current][0]
        if current != final and (not isinstance(graph.contracts[current], (CopyContract, PermutationContract))):
            return False
    return current == final


def _ancestors(graph: ValueGraph, starts: set[int]) -> set[int]:
    """Return semantic producer ancestors including the starting leaves."""
    result = set(starts)
    stack = list(starts)
    while stack:
        for producer in graph.predecessors[stack.pop()].values():
            if producer is not None and producer not in result:
                result.add(producer)
                stack.append(producer)
    return result


def _chunk_sizes(ir: KernelIR, absorbed: set[int], progress_axis: str) -> tuple[int, ...]:
    """Enumerate divisors tileable by every operation in the chain."""
    extent = ir.axis_extent(progress_axis)
    result: list[int] = []
    for size in range(1, extent + 1):
        valid = extent % size == 0
        for nid in absorbed:
            leaf = ir.tree.isa(nid)
            block = ir.tree.block(owning_block(ir.tree, nid))
            for abstract, concrete in block.axis_map.items():
                if concrete == progress_axis:
                    minimum = leaf.op_cls.MIN_TILE_SIZE.get(abstract, 1)
                    maximum = leaf.op_cls.MAX_TILE_SIZE.get(abstract)
                    tile = size if maximum is None else min(size, maximum)
                    valid = valid and size >= minimum and (size % tile == 0)
        if valid:
            result.append(size)
    return tuple(result)
