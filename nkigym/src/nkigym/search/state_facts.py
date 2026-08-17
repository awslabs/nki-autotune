"""Compact read-only facts shared by transform legality analyzers."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol, cast, runtime_checkable

from nkigym.ops.base import (
    BilinearReductionContract,
    CopyContract,
    InitializerContract,
    NKIOp,
    PermutationContract,
    PointwiseContract,
    ReductionContract,
)


@runtime_checkable
class _OperationTree(Protocol):
    """Tree traversal surface required by operation-fact analysis."""

    def preorder(self, nid: int | None = None) -> Iterator[int]:
        """Yield stable node ids in pre-order."""
        ...

    def data(self, nid: int) -> object:
        """Return one tree node payload."""
        ...


@dataclass(frozen=True)
class OperationFacts:
    """Operation and algebraic-contract features of one schedule tree."""

    op_classes: frozenset[type[NKIOp]]
    pointwise_operators: frozenset[str]
    has_copy: bool
    has_initializer: bool
    has_reduction: bool
    has_batched_permutation: bool
    has_rfactor: bool
    has_unknown_contract: bool

    def has_ops(self, *op_classes: type[NKIOp]) -> bool:
        """Return whether every required operation class occurs."""
        return self.op_classes.issuperset(op_classes)


def compute_operation_facts(tree: _OperationTree) -> OperationFacts:
    """Derive compact operation facts from one immutable schedule tree."""
    op_classes: set[type[NKIOp]] = set()
    pointwise_operators: set[str] = set()
    has_copy = has_initializer = has_reduction = has_batched_permutation = has_rfactor = False
    has_unknown_contract = False
    for nid in tree.preorder():
        node = tree.data(nid)
        op_cls = getattr(node, "op_cls", None)
        kwargs = getattr(node, "kwargs", None)
        if not isinstance(op_cls, type) or not issubclass(op_cls, NKIOp) or not isinstance(kwargs, dict):
            continue
        typed_op_cls = cast(type[NKIOp], op_cls)
        op_classes.add(typed_op_cls)
        has_rfactor = has_rfactor or typed_op_cls.RFACTOR_RECIPE is not None
        contract = typed_op_cls.algebraic_contract(kwargs)
        has_unknown_contract = has_unknown_contract or contract is None
        if isinstance(contract, PointwiseContract):
            pointwise_operators.add(contract.operator)
        has_copy = has_copy or isinstance(contract, CopyContract)
        has_initializer = has_initializer or isinstance(contract, InitializerContract)
        has_reduction = has_reduction or isinstance(contract, (ReductionContract, BilinearReductionContract))
        has_batched_permutation = has_batched_permutation or (
            isinstance(contract, PermutationContract) and contract.batching is not None
        )
    return OperationFacts(
        op_classes=frozenset(op_classes),
        pointwise_operators=frozenset(pointwise_operators),
        has_copy=has_copy,
        has_initializer=has_initializer,
        has_reduction=has_reduction,
        has_batched_permutation=has_batched_permutation,
        has_rfactor=has_rfactor,
        has_unknown_contract=has_unknown_contract,
    )


def operation_facts(ir: object) -> OperationFacts:
    """Return serialized operation facts or derive them for an in-process IR."""
    facts = getattr(ir, "_operation_facts", None)
    if isinstance(facts, OperationFacts):
        return facts
    tree = getattr(ir, "tree")
    if not isinstance(tree, _OperationTree):
        raise TypeError(f"operation facts require a schedule tree, got {type(tree).__name__}")
    return compute_operation_facts(tree)


__all__ = ["OperationFacts", "compute_operation_facts", "operation_facts"]
