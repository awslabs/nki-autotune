"""Base classes for the rewrite-transform interface.

Each concrete transform under :mod:`nkigym.transforms` subclasses
:class:`Transform` and exposes:

* ``analyze(ir) -> list[TransformOption]`` — enumerate every legal
  option for this transform on ``ir``.
* ``apply(ir, option) -> KernelIR`` — re-check legality, deep-copy
  ``ir``, mutate the copy, return it. Raises
  :class:`TransformLegalityError` on illegal options. Loud failures
  only — no try/except recovery.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from numbers import Real
from typing import Any, Generic, TypeVar
from weakref import WeakKeyDictionary

import networkx as nx

from nkigym.ir import KernelIR, KernelTree
from nkigym.ir.tree import BlockNode, BufferRegion, ISANode
from nkigym.ops.activation import NKIActivation
from nkigym.ops.base import PointwiseContract
from nkigym.ops.reciprocal import NKIReciprocal
from nkigym.ops.tensor_scalar import NKITensorScalar

_PIPELINE_OVERLAPS: WeakKeyDictionary[KernelTree, frozenset[int]] = WeakKeyDictionary()


@dataclass(frozen=True)
class TransformOption:
    """Marker base for per-transform option payloads.

    Subclasses are frozen dataclasses (so options are hashable, useful
    for deduplication in samplers).
    """


class TransformLegalityError(ValueError):
    """Raised by :meth:`Transform.apply` when ``option`` is illegal for ``ir``."""


def _clone_tree(tree: KernelTree) -> KernelTree:
    """Clone mutable tree state while sharing frozen value payloads."""
    result = object.__new__(KernelTree)
    graph = nx.DiGraph()
    source = vars(tree.graph)
    target = vars(graph)
    target["graph"] = tree.graph.graph.copy()
    target["_node"] = {nid: attrs.copy() for nid, attrs in source["_node"].items()}
    target["_succ"] = {nid: edges.copy() for nid, edges in source["_succ"].items()}
    target["_pred"] = {nid: edges.copy() for nid, edges in source["_pred"].items()}
    target["_adj"] = target["_succ"]
    result.graph = graph
    result._next_id = tree.next_node_id
    result.root = tree.root
    return result


def copy_for_rewrite(ir: KernelIR) -> KernelIR:
    """Copy mutable IR state without cloning dependency data that will be rebuilt."""
    tree = _clone_tree(ir.tree)
    if (overlap := _PIPELINE_OVERLAPS.get(ir.tree)) is not None:
        _PIPELINE_OVERLAPS[tree] = overlap
    return KernelIR(
        func_name=ir.func_name,
        param_names=list(ir.param_names),
        return_names=ir.return_names,
        tree=tree,
        dependency=ir.dependency,
        param_buffers=dict(ir.param_buffers),
    )


def software_pipeline_overlap_nodes(ir: KernelIR) -> frozenset[int]:
    """Return every node comparable with an active pipeline loop."""
    cached = _PIPELINE_OVERLAPS.get(ir.tree)
    if cached is not None:
        return cached
    result: set[int] = set()
    for block_nid in ir.tree.blocks():
        annotation = ir.tree.block(block_nid).annotations.get("software_pipeline")
        if annotation is None:
            continue
        loop_nid = annotation["loop_nid"]
        result.update((loop_nid, *ir.tree.ancestors(loop_nid), *ir.tree.descendants(loop_nid)))
    overlap = frozenset(result)
    _PIPELINE_OVERLAPS[ir.tree] = overlap
    return overlap


def invalidate_software_pipeline_overlap(tree: KernelTree) -> None:
    """Discard cached overlap facts after pipeline annotations change."""
    _PIPELINE_OVERLAPS.pop(tree, None)


def intersects_software_pipeline(
    ir: KernelIR, nids: tuple[int, ...], overlap_nodes: frozenset[int] | None = None
) -> bool:
    """Return whether any selected node overlaps an active pipeline scope."""
    active = software_pipeline_overlap_nodes(ir) if overlap_nodes is None else overlap_nodes
    return not active.isdisjoint(nids)


def active_block_axes_align(producer_leaf: ISANode, producer_block: BlockNode, consumer_block: BlockNode) -> bool:
    """Return whether active producer axes exactly match one consumer block."""
    active_abstract = {
        abstract
        for slot, region in producer_leaf.operand_bindings.items()
        for abstract in producer_leaf.op_cls.OPERAND_AXES[slot][: len(region.ranges)]
    }
    axis_map = {
        abstract: concrete for abstract, concrete in producer_block.axis_map.items() if abstract in active_abstract
    }
    active_concrete = set(axis_map.values())
    retained = tuple(
        (iter_var, iter_value)
        for iter_var, iter_value in zip(producer_block.iter_vars, producer_block.iter_values)
        if iter_var.axis in active_concrete
    )
    return (
        tuple(item[0] for item in retained) == consumer_block.iter_vars
        and tuple(item[1] for item in retained) == consumer_block.iter_values
        and axis_map == consumer_block.axis_map
    )


@dataclass(frozen=True)
class ActivationComposition:
    """One pointwise producer absorbed into an adjacent activation."""

    producer_block_nid: int
    consumer_block_nid: int
    consumer_leaf_nid: int
    data: BufferRegion
    intermediate: BufferRegion
    kwargs: dict[str, Any]


def resolve_activation_composition(
    ir: KernelIR,
    producer_block_nid: int,
    consumer_block_nid: int,
    unique_consumer: Callable[[KernelIR, str, int, int], bool],
) -> ActivationComposition | None:
    """Resolve one affine or sqrt producer into one activation consumer."""
    result: ActivationComposition | None = None
    producer_leaf_nid = _sole_isa_leaf(ir, producer_block_nid)
    consumer_leaf_nid = _owned_isa_leaf(ir, consumer_block_nid)
    if producer_leaf_nid is not None and consumer_leaf_nid is not None:
        producer_leaf = ir.tree.isa(producer_leaf_nid)
        consumer_leaf = ir.tree.isa(consumer_leaf_nid)
        producer_contract = producer_leaf.op_cls.algebraic_contract(producer_leaf.kwargs)
        consumer_contract = consumer_leaf.op_cls.algebraic_contract(consumer_leaf.kwargs)
        if isinstance(producer_contract, PointwiseContract) and isinstance(consumer_contract, PointwiseContract):
            intermediate = producer_leaf.operand_bindings.get(producer_contract.output_operand)
            consumer_input = (
                consumer_leaf.operand_bindings.get(consumer_contract.input_operands[0])
                if len(consumer_contract.input_operands) == 1
                else None
            )
            composition = _activation_composition(producer_leaf, producer_contract, consumer_leaf, consumer_contract)
            producer_block = ir.tree.block(producer_block_nid)
            consumer_block = ir.tree.block(consumer_block_nid)
            if intermediate is not None and consumer_input == intermediate and composition is not None:
                data, kwargs = composition
                legal = (
                    active_block_axes_align(producer_leaf, producer_block, consumer_block)
                    and not producer_leaf.access_patterns
                    and not consumer_leaf.access_patterns
                    and intermediate.tensor not in ir.param_buffers
                    and intermediate.tensor not in ir.return_names
                    and ir.buffer(intermediate.tensor).location != "shared_hbm"
                    and ir.buffer(data.tensor).location in NKIActivation.INPUT_LOCATIONS["data"]
                    and all(buffer.name == intermediate.tensor for buffer in producer_block.alloc_buffers)
                    and unique_consumer(ir, intermediate.tensor, producer_leaf_nid, consumer_leaf_nid)
                )
                if legal:
                    result = ActivationComposition(
                        producer_block_nid=producer_block_nid,
                        consumer_block_nid=consumer_block_nid,
                        consumer_leaf_nid=consumer_leaf_nid,
                        data=data,
                        intermediate=intermediate,
                        kwargs=kwargs,
                    )
    return result


def _activation_composition(
    producer_leaf: ISANode, producer: PointwiseContract, consumer_leaf: ISANode, consumer: PointwiseContract
) -> tuple[BufferRegion, dict[str, Any]] | None:
    """Return the input and kwargs for one native activation composition."""
    result: tuple[BufferRegion, dict[str, Any]] | None = None
    data = producer_leaf.operand_bindings.get("data")
    no_tensor_bias = "bias" not in producer_leaf.operand_bindings and "bias" not in consumer_leaf.operand_bindings
    reciprocal = consumer.operator == "reciprocal" and consumer.scale == 1.0 and consumer.bias == 0.0
    if (
        data is not None
        and no_tensor_bias
        and producer_leaf.op_cls is NKIActivation
        and producer.operator == "sqrt"
        and consumer_leaf.op_cls in {NKIActivation, NKIReciprocal}
        and reciprocal
    ):
        result = (data, _activation_kwargs("rsqrt", producer.scale, producer.bias))
    elif data is not None and no_tensor_bias and consumer_leaf.op_cls is NKIActivation:
        affine = _literal_affine(producer_leaf, producer)
        if affine is not None:
            affine_data, scale, bias = affine
            result = (
                affine_data,
                _activation_kwargs(consumer.operator, scale * consumer.scale, bias * consumer.scale + consumer.bias),
            )
    return result


def _literal_affine(node: ISANode, contract: PointwiseContract) -> tuple[BufferRegion, float, float] | None:
    """Return one literal affine producer as ``data, scale, bias``."""
    result: tuple[BufferRegion, float, float] | None = None
    data = node.operand_bindings.get("data")
    if data is not None and node.op_cls is NKIActivation and contract.operator == "copy":
        result = (data, contract.scale, contract.bias)
    elif data is not None and node.op_cls is NKITensorScalar and "operand0" not in node.operand_bindings:
        literal = node.kwargs.get("operand0")
        if isinstance(literal, Real) and not isinstance(literal, bool):
            value = float(literal)
            if contract.operator == "add":
                result = (data, 1.0, value)
            elif contract.operator == "subtract":
                result = (data, -1.0, value) if contract.reverse else (data, 1.0, -value)
            elif contract.operator == "multiply":
                result = (data, value, 0.0)
    return result


def _activation_kwargs(operator: str, scale: float, bias: float) -> dict[str, Any]:
    """Return normalized scalar activation keyword arguments."""
    kwargs: dict[str, Any] = {"op": operator}
    if scale != 1.0:
        kwargs["scale"] = scale
    if bias != 0.0:
        kwargs["bias"] = bias
    return kwargs


def _sole_isa_leaf(ir: KernelIR, block_nid: int) -> int | None:
    """Return the sole ISA leaf in one block subtree."""
    leaves = [nid for nid in ir.tree.preorder(block_nid) if isinstance(ir.tree.data(nid), ISANode)]
    return leaves[0] if len(leaves) == 1 else None


def _owned_isa_leaf(ir: KernelIR, block_nid: int) -> int | None:
    """Return the sole ISA leaf owned by one block."""
    leaves = [
        nid
        for nid in ir.tree.preorder(block_nid)
        if isinstance(ir.tree.data(nid), ISANode)
        and next(
            (
                ancestor
                for ancestor in reversed(ir.tree.ancestors(nid))
                if isinstance(ir.tree.data(ancestor), BlockNode)
            ),
            None,
        )
        == block_nid
    ]
    return leaves[0] if len(leaves) == 1 else None


_OptionT = TypeVar("_OptionT", bound=TransformOption)


class Transform(Generic[_OptionT]):
    """Base class for stateless rewrite transforms.

    Subclasses override :meth:`analyze` and :meth:`apply`. Instances
    carry no state — the same instance can be reused across many
    ``ir``'s.
    """

    def analyze(self, ir: KernelIR) -> list[_OptionT]:
        """Return every legal option for this transform on ``ir``."""
        raise NotImplementedError

    def apply(self, ir: KernelIR, option: _OptionT) -> KernelIR:
        """Re-check legality, deep-copy ``ir``, mutate the copy, return it."""
        raise NotImplementedError


__all__ = [
    "ActivationComposition",
    "Transform",
    "TransformLegalityError",
    "TransformOption",
    "active_block_axes_align",
    "copy_for_rewrite",
    "invalidate_software_pipeline_overlap",
    "intersects_software_pipeline",
    "resolve_activation_composition",
    "software_pipeline_overlap_nodes",
]
