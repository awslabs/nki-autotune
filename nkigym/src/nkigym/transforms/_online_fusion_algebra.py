"""Algebraic transfer rules for online-fusion detection."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from numbers import Real

from nkigym.ir import KernelIR
from nkigym.ir.tree import ISANode
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
from nkigym.transforms._canonical_rewrite import owning_block
from nkigym.transforms._online_fusion_types import (
    BinaryFactor,
    ConstantFactor,
    FactorExpression,
    OnlineFusionStage,
    StateFactor,
    UnaryFactor,
    ValueGraph,
    contract_input_operands,
    factor_states,
)


class ValueKind(str, Enum):
    """How a value depends on online state and the current chunk."""

    CONSTANT = "constant"
    RESIDUAL = "residual"
    STATE = "state"
    ADDITIVE = "additive"
    MULTIPLICATIVE = "multiplicative"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class SeparatedValue:
    """State/chunk separation result for one tensor."""

    kind: ValueKind
    factor: FactorExpression | None
    factor_axes: tuple[str, ...]
    depends_on_progress: bool


@dataclass(frozen=True)
class AlgebraEvaluation:
    """Interpretation result for one candidate progress axis."""

    stages: tuple[OnlineFusionStage, ...]
    values_by_tensor: Mapping[str, SeparatedValue]
    values_by_leaf: Mapping[int, SeparatedValue]


def evaluate_algebra(ir: KernelIR, graph: ValueGraph, progress_axis: str) -> AlgebraEvaluation:
    """Propagate state/chunk separation for one progress axis."""
    values_by_tensor: dict[str, SeparatedValue] = {}
    values_by_leaf: dict[int, SeparatedValue] = {}
    stages: list[OnlineFusionStage] = []
    for leaf_nid in graph.leaves:
        leaf = ir.tree.isa(leaf_nid)
        contract = graph.contracts[leaf_nid]
        inputs = {
            slot: _operand_value(leaf, slot, graph, values_by_tensor, progress_axis)
            for slot in contract_input_operands(contract)
            if _operand_is_available(leaf, contract, slot)
        }
        value, stage, mapped_output = _transfer(ir, graph, leaf_nid, contract, inputs, progress_axis, tuple(stages))
        if isinstance(contract, ReductionContract) and contract.mapped_output_operand is not None:
            if mapped_output is None:
                raise AssertionError(f"{type(contract).__name__} has no mapped output value")
            mapped_region = leaf.operand_bindings.get(contract.mapped_output_operand)
            if mapped_region is None:
                raise ValueError(
                    f"{leaf.op_cls.__name__} contract mapped output " f"{contract.mapped_output_operand!r} is unbound"
                )
            values_by_tensor[mapped_region.tensor] = mapped_output
        output_tensor = graph.output_by_leaf[leaf_nid]
        values_by_tensor[output_tensor] = value
        values_by_leaf[leaf_nid] = value
        if stage is not None:
            stages.append(stage)
    return AlgebraEvaluation(stages=tuple(stages), values_by_tensor=values_by_tensor, values_by_leaf=values_by_leaf)


def _operand_is_available(leaf: ISANode, contract: OperatorContract, slot: str) -> bool:
    """Return whether ``slot`` is a tensor binding or independent literal."""
    available = slot in leaf.operand_bindings
    bias_operand = contract.bias_operand if isinstance(contract, (PointwiseContract, ReductionContract)) else None
    if slot != bias_operand:
        available = available or slot in leaf.kwargs
    return available


def _operand_value(
    leaf: ISANode, slot: str, graph: ValueGraph, values_by_tensor: Mapping[str, SeparatedValue], progress_axis: str
) -> SeparatedValue:
    """Resolve one tensor binding or literal operand to an abstract value."""
    result = _unknown()
    region = leaf.operand_bindings.get(slot)
    if region is not None:
        prior = values_by_tensor.get(region.tensor)
        if prior is not None:
            result = prior
        else:
            axes = graph.tensor_axes[region.tensor]
            result = _residual(progress_axis in axes)
    else:
        literal = leaf.kwargs.get(slot)
        if isinstance(literal, Real):
            result = _constant(float(literal))
    return result


def _transfer(
    ir: KernelIR,
    graph: ValueGraph,
    leaf_nid: int,
    contract: OperatorContract,
    inputs: Mapping[str, SeparatedValue],
    progress_axis: str,
    stages: tuple[OnlineFusionStage, ...],
) -> tuple[SeparatedValue, OnlineFusionStage | None, SeparatedValue | None]:
    """Apply one contract's separation transfer rule."""
    stage: OnlineFusionStage | None = None
    mapped_output: SeparatedValue | None = None
    if isinstance(contract, PointwiseContract):
        value = _pointwise(contract, inputs)
    elif isinstance(contract, (CopyContract, PermutationContract)):
        value = inputs[contract.input_operand]
    elif isinstance(contract, ReductionContract):
        mapped_input = _copy_affine(inputs[contract.input_operand], contract.scale, contract.bias)
        if contract.bias_operand is not None and contract.bias_operand in inputs:
            mapped_input = _additive(mapped_input, inputs[contract.bias_operand], "add")
        mapped = (
            mapped_input
            if contract.map_operator == "copy"
            else _apply_unary(contract.map_operator, mapped_input, 1.0, 0.0)
        )
        mapped_output = mapped
        value, stage = _reduce(
            ir, graph, leaf_nid, contract.combinator, contract.reduction_axis, mapped, progress_axis, stages
        )
    elif isinstance(contract, BilinearReductionContract):
        product = _multiply(inputs[contract.left_operand], inputs[contract.right_operand])
        value, stage = _reduce(
            ir, graph, leaf_nid, contract.combinator, contract.reduction_axis, product, progress_axis, stages
        )
    elif isinstance(contract, InitializerContract):
        value = _constant(contract.value)
    elif isinstance(contract, PointwiseSequenceContract):
        value = _unknown()
    else:
        raise TypeError(f"unsupported contract {type(contract).__name__}")
    return value, stage, mapped_output


def _pointwise(contract: PointwiseContract, inputs: Mapping[str, SeparatedValue]) -> SeparatedValue:
    """Apply a unary or binary pointwise contract."""
    operands = tuple(inputs[name] for name in contract.input_operands)
    if len(operands) == 1:
        if contract.bias_operand is not None and contract.bias_operand in inputs:
            mapped = _copy_affine(operands[0], contract.scale, contract.bias)
            mapped = _additive(mapped, inputs[contract.bias_operand], "add")
            result = mapped if contract.operator == "copy" else _apply_unary(contract.operator, mapped, 1.0, 0.0)
        else:
            result = _apply_unary(contract.operator, operands[0], contract.scale, contract.bias)
    elif len(operands) == 2:
        left, right = operands
        if contract.reverse:
            left, right = right, left
        if contract.operator == "multiply":
            result = _multiply(left, right)
        elif contract.operator in {"add", "subtract"}:
            result = _additive(left, right, contract.operator)
        elif contract.operator == "maximum":
            result = _state_binary(left, right, "maximum")
        else:
            result = _unknown()
    else:
        result = _unknown()
    return result


def _apply_unary(operator: str, value: SeparatedValue, scale: float, bias: float) -> SeparatedValue:
    """Apply a unary map while preserving only proven separability."""
    depends = value.depends_on_progress
    if operator == "copy":
        result = _copy_affine(value, scale, bias)
    elif value.kind == ValueKind.STATE and value.factor is not None:
        result = _state(UnaryFactor(operator=operator, operand=value.factor, scale=scale, bias=bias), value.factor_axes)
    elif value.kind == ValueKind.RESIDUAL:
        result = _residual(depends)
    elif value.kind == ValueKind.CONSTANT:
        result = _constant(0.0) if operator == "copy" and scale == 0.0 else _residual(False)
    elif operator == "exp" and value.kind == ValueKind.ADDITIVE and value.factor is not None:
        factor = UnaryFactor(operator="exp", operand=value.factor, scale=scale, bias=bias)
        result = _multiplicative(factor, value.factor_axes, depends)
    elif operator == "reciprocal" and value.kind == ValueKind.MULTIPLICATIVE and value.factor is not None:
        factor = UnaryFactor(operator="reciprocal", operand=value.factor)
        result = _multiplicative(factor, value.factor_axes, depends)
    elif operator == "square" and value.kind == ValueKind.MULTIPLICATIVE and value.factor is not None:
        factor = BinaryFactor(operator="multiply", left=value.factor, right=value.factor)
        result = _multiplicative(factor, value.factor_axes, depends)
    else:
        result = _unknown()
    return result


def _copy_affine(value: SeparatedValue, scale: float, bias: float) -> SeparatedValue:
    """Apply an affine identity map."""
    if value.kind == ValueKind.STATE and value.factor is not None:
        result = _state(UnaryFactor(operator="copy", operand=value.factor, scale=scale, bias=bias), value.factor_axes)
    elif value.kind == ValueKind.ADDITIVE and value.factor is not None:
        factor = UnaryFactor(operator="copy", operand=value.factor, scale=scale, bias=bias)
        result = _additively_separated(factor, value.factor_axes, value.depends_on_progress)
    elif value.kind in {ValueKind.RESIDUAL, ValueKind.CONSTANT}:
        result = _residual(value.depends_on_progress)
    elif scale == 1.0 and bias == 0.0:
        result = value
    else:
        result = _unknown()
    return result


def _multiply(left: SeparatedValue, right: SeparatedValue) -> SeparatedValue:
    """Multiply two values and combine separable state factors."""
    if ValueKind.UNKNOWN in {left.kind, right.kind} or ValueKind.ADDITIVE in {left.kind, right.kind}:
        result = _unknown()
    else:
        left_factor = left.factor
        right_factor = right.factor
        has_state = bool(factor_states(left_factor) | factor_states(right_factor))
        factor: FactorExpression | None = None
        if has_state and left_factor is not None and right_factor is not None:
            factor = _factor_multiply(left_factor, right_factor)
        elif has_state and left_factor is not None and factor_states(left_factor):
            factor = left_factor
        elif has_state and right_factor is not None and factor_states(right_factor):
            factor = right_factor
        axes = _merge_axes(left.factor_axes, right.factor_axes)
        depends = left.depends_on_progress or right.depends_on_progress
        has_residual = left.kind in {ValueKind.RESIDUAL, ValueKind.MULTIPLICATIVE} or right.kind in {
            ValueKind.RESIDUAL,
            ValueKind.MULTIPLICATIVE,
        }
        if not has_state and left.kind == ValueKind.CONSTANT and right.kind == ValueKind.CONSTANT:
            assert isinstance(left_factor, ConstantFactor) and isinstance(right_factor, ConstantFactor)
            result = _constant(left_factor.value * right_factor.value)
        elif factor is None:
            result = _residual(depends)
        elif has_residual:
            result = _multiplicative(factor, axes, depends)
        else:
            result = _state(factor, axes)
    return result


def _additive(left: SeparatedValue, right: SeparatedValue, operator: str) -> SeparatedValue:
    """Add or subtract values when state and residual terms remain separate."""
    if ValueKind.UNKNOWN in {left.kind, right.kind} or ValueKind.MULTIPLICATIVE in {left.kind, right.kind}:
        result = _unknown()
    else:
        left_factor = left.factor
        right_factor = right.factor
        has_state = bool(factor_states(left_factor) | factor_states(right_factor))
        if operator == "subtract" and right_factor is not None:
            right_factor = UnaryFactor(operator="copy", operand=right_factor, scale=-1.0)
        if has_state and left_factor is not None and right_factor is not None:
            factor: FactorExpression | None = BinaryFactor(operator="add", left=left_factor, right=right_factor)
        elif has_state:
            factor = left_factor if left_factor is not None else right_factor
        else:
            factor = None
        axes = _merge_axes(left.factor_axes, right.factor_axes)
        depends = left.depends_on_progress or right.depends_on_progress
        has_residual = left.kind in {ValueKind.RESIDUAL, ValueKind.ADDITIVE, ValueKind.CONSTANT} or right.kind in {
            ValueKind.RESIDUAL,
            ValueKind.ADDITIVE,
            ValueKind.CONSTANT,
        }
        if factor is None:
            result = _residual(depends)
        elif has_residual:
            result = _additively_separated(factor, axes, depends)
        else:
            result = _state(factor, axes)
    return result


def _state_binary(left: SeparatedValue, right: SeparatedValue, operator: str) -> SeparatedValue:
    """Apply a binary operation only when both operands are state-only."""
    if left.kind == ValueKind.STATE and right.kind == ValueKind.STATE:
        assert left.factor is not None and right.factor is not None
        result = _state(
            BinaryFactor(operator=operator, left=left.factor, right=right.factor),
            _merge_axes(left.factor_axes, right.factor_axes),
        )
    else:
        result = _unknown()
    return result


def _reduce(
    ir: KernelIR,
    graph: ValueGraph,
    leaf_nid: int,
    combinator: ReduceCombinator,
    abstract_axis: str,
    contribution: SeparatedValue,
    progress_axis: str,
    stages: tuple[OnlineFusionStage, ...],
) -> tuple[SeparatedValue, OnlineFusionStage | None]:
    """Reduce a contribution and create a stage when the recurrence is proven."""
    block_nid = owning_block(ir.tree, leaf_nid)
    block = ir.tree.block(block_nid)
    concrete_axis = block.axis_map[abstract_axis]
    output_tensor = graph.output_by_leaf[leaf_nid]
    output_axes = graph.tensor_axes[output_tensor]
    stage: OnlineFusionStage | None = None
    if concrete_axis != progress_axis:
        if contribution.factor is not None and (
            concrete_axis in contribution.factor_axes or not set(contribution.factor_axes).issubset(output_axes)
        ):
            value = _unknown()
        else:
            value = contribution
    else:
        prior_states = factor_states(contribution.factor)
        first_stage = (
            not stages
            and contribution.kind in {ValueKind.RESIDUAL, ValueKind.CONSTANT}
            and contribution.depends_on_progress
        )
        later_stage = (
            bool(stages)
            and combinator.combiner == "add"
            and contribution.kind == ValueKind.MULTIPLICATIVE
            and contribution.factor is not None
            and bool(prior_states)
            and prior_states.issubset(set(range(len(stages))))
            and progress_axis not in contribution.factor_axes
        )
        if first_stage or later_stage:
            factor = contribution.factor if later_stage else None
            factor_axes = contribution.factor_axes if later_stage else ()
            stage = OnlineFusionStage(
                reducer_leaf=leaf_nid,
                reducer_block=block_nid,
                state_tensor=output_tensor,
                combinator=combinator,
                factor=factor,
                factor_axes=factor_axes,
            )
            value = _state(StateFactor(stage=len(stages)), output_axes)
        else:
            value = _unknown()
    return value, stage


def _factor_multiply(left: FactorExpression, right: FactorExpression) -> FactorExpression:
    """Multiply factors with constant-one simplification."""
    result: FactorExpression
    if isinstance(left, ConstantFactor) and left.value == 1.0:
        result = right
    elif isinstance(right, ConstantFactor) and right.value == 1.0:
        result = left
    else:
        result = BinaryFactor(operator="multiply", left=left, right=right)
    return result


def _merge_axes(left: tuple[str, ...], right: tuple[str, ...]) -> tuple[str, ...]:
    """Return a stable union of concrete factor axes."""
    return tuple(dict.fromkeys((*left, *right)))


def _constant(value: float) -> SeparatedValue:
    """Construct a literal value."""
    return SeparatedValue(ValueKind.CONSTANT, ConstantFactor(value), (), False)


def _residual(depends_on_progress: bool) -> SeparatedValue:
    """Construct a state-independent value."""
    return SeparatedValue(ValueKind.RESIDUAL, None, (), depends_on_progress)


def _state(factor: FactorExpression, axes: tuple[str, ...]) -> SeparatedValue:
    """Construct a state-only value."""
    return SeparatedValue(ValueKind.STATE, factor, axes, False)


def _additively_separated(factor: FactorExpression, axes: tuple[str, ...], depends_on_progress: bool) -> SeparatedValue:
    """Construct ``state_term + residual``."""
    return SeparatedValue(ValueKind.ADDITIVE, factor, axes, depends_on_progress)


def _multiplicative(factor: FactorExpression, axes: tuple[str, ...], depends_on_progress: bool) -> SeparatedValue:
    """Construct ``state_factor * residual``."""
    return SeparatedValue(ValueKind.MULTIPLICATIVE, factor, axes, depends_on_progress)


def _unknown() -> SeparatedValue:
    """Construct an unsupported algebraic value."""
    return SeparatedValue(ValueKind.UNKNOWN, None, (), False)


__all__ = ["AlgebraEvaluation", "SeparatedValue", "ValueKind", "evaluate_algebra"]
