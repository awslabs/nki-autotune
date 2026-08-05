"""Shared types for contract-driven online fusion."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

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


@dataclass(frozen=True)
class StateFactor:
    """Reference to one preceding online state."""

    stage: int


@dataclass(frozen=True)
class ConstantFactor:
    """Scalar constant in a state factor."""

    value: float


@dataclass(frozen=True)
class UnaryFactor:
    """Unary state-factor expression ``operator(operand * scale + bias)``."""

    operator: str
    operand: FactorExpression
    scale: float = 1.0
    bias: float = 0.0


@dataclass(frozen=True)
class BinaryFactor:
    """Binary state-factor expression."""

    operator: str
    left: FactorExpression
    right: FactorExpression


FactorExpression = StateFactor | ConstantFactor | UnaryFactor | BinaryFactor


@dataclass(frozen=True)
class OnlineFusionStage:
    """One ordered reduction state in an online-fusion chain."""

    reducer_leaf: int
    reducer_block: int
    state_tensor: str
    combinator: ReduceCombinator
    factor: FactorExpression | None
    factor_axes: tuple[str, ...]


@dataclass(frozen=True)
class DeferredFactor:
    """One final-stage factor that can be applied after the recurrence."""

    stage: int
    factor: FactorExpression
    recurrence_factor: FactorExpression | None
    producer_leaf: int
    bypass_leaf: int
    passthrough_operand: str


@dataclass(frozen=True)
class OnlineFusionMatch:
    """A contract-proven online-fusion chain and its rewrite boundary."""

    progress_axis: str
    progress_extent: int
    stages: tuple[OnlineFusionStage, ...]
    derivation_leaves: tuple[int, ...]
    absorbed_blocks: tuple[int, ...]
    external_inputs: tuple[str, ...]
    external_outputs: tuple[str, ...]
    chunk_sizes: tuple[int, ...]
    deferred_factor: DeferredFactor | None
    incremental_prefix: bool = False

    @property
    def match_id(self) -> tuple[str, tuple[int, ...]]:
        """Return a stable identity suitable for a transform option."""
        return (self.progress_axis, tuple(stage.reducer_leaf for stage in self.stages))


@dataclass(frozen=True)
class ValueGraph:
    """Canonical SSA use-def graph derived from ISA operand bindings."""

    leaves: tuple[int, ...]
    contracts: Mapping[int, OperatorContract]
    output_by_leaf: Mapping[int, str]
    input_tensors_by_leaf: Mapping[int, Mapping[str, str]]
    predecessors: Mapping[int, Mapping[str, int | None]]
    successors: Mapping[int, tuple[int, ...]]
    tensor_axes: Mapping[str, tuple[str, ...]]
    initializers_by_tensor: Mapping[str, tuple[int, ...]]


def contract_input_operands(contract: OperatorContract) -> tuple[str, ...]:
    """Return semantic input slots for one contract."""
    result: tuple[str, ...]
    if isinstance(contract, PointwiseContract):
        result = contract.input_operands
        if contract.bias_operand is not None:
            result = (*result, contract.bias_operand)
    elif isinstance(contract, PointwiseSequenceContract):
        result = contract.input_operands
    elif isinstance(contract, ReductionContract):
        result = (contract.input_operand,)
        if contract.bias_operand is not None:
            result = (*result, contract.bias_operand)
    elif isinstance(contract, BilinearReductionContract):
        result = (contract.left_operand, contract.right_operand)
    elif isinstance(contract, (PermutationContract, CopyContract)):
        result = (contract.input_operand,)
    elif isinstance(contract, InitializerContract):
        result = ()
    else:
        raise TypeError(f"unsupported contract {type(contract).__name__}")
    return result


def contract_output_operand(contract: OperatorContract) -> str:
    """Return the single semantic output slot for one contract."""
    return contract.output_operand


def factor_states(expression: FactorExpression | None) -> frozenset[int]:
    """Return all state references in a factor expression."""
    states: set[int] = set()
    if isinstance(expression, StateFactor):
        states.add(expression.stage)
    elif isinstance(expression, UnaryFactor):
        states.update(factor_states(expression.operand))
    elif isinstance(expression, BinaryFactor):
        states.update(factor_states(expression.left))
        states.update(factor_states(expression.right))
    return frozenset(states)


__all__ = [
    "BinaryFactor",
    "ConstantFactor",
    "DeferredFactor",
    "FactorExpression",
    "OnlineFusionMatch",
    "OnlineFusionStage",
    "StateFactor",
    "UnaryFactor",
    "ValueGraph",
    "contract_input_operands",
    "contract_output_operand",
    "factor_states",
]
