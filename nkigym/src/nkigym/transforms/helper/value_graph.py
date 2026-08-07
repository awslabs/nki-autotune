"""Shared contract-level SSA graph construction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

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
    ReductionContract,
)
from nkigym.transforms.helper.canonical_rewrite import owning_block


@dataclass(frozen=True)
class ValueGraph:
    """Semantic SSA use-def graph derived from operation contracts."""

    leaves: tuple[int, ...]
    contracts: Mapping[int, OperatorContract]
    outputs: Mapping[int, str]
    inputs: Mapping[int, Mapping[str, str]]
    predecessors: Mapping[int, Mapping[str, int | None]]
    successors: Mapping[int, tuple[int, ...]]
    tensor_axes: Mapping[str, tuple[str, ...]]
    initializers: Mapping[str, tuple[int, ...]]


def contract_input_operands(contract: OperatorContract) -> tuple[str, ...]:
    """Return semantic input slots for one operation contract."""
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


def build_value_graph(ir: KernelIR) -> ValueGraph:
    """Build the contract-level SSA graph in tree order."""
    leaves = tuple(nid for nid in ir.tree.preorder() if isinstance(ir.tree.data(nid), ISANode))
    contracts: dict[int, OperatorContract] = {}
    outputs: dict[int, str] = {}
    inputs: dict[int, dict[str, str]] = {}
    predecessors: dict[int, dict[str, int | None]] = {}
    successors: dict[int, list[int]] = {leaf: [] for leaf in leaves}
    tensor_axes: dict[str, tuple[str, ...]] = {}
    initializers: dict[str, list[int]] = {}
    producers: dict[str, int] = {}
    for nid in leaves:
        leaf = ir.tree.isa(nid)
        contract = leaf.op_cls.algebraic_contract(leaf.kwargs)
        if contract is None:
            raise ValueError(f"{leaf.op_cls.__name__} has no algebraic contract")
        contracts[nid] = contract
        axis_map = ir.tree.block(owning_block(ir.tree, nid)).axis_map
        for slot, region in leaf.operand_bindings.items():
            axes = tuple(axis_map[axis] for axis in leaf.op_cls.OPERAND_AXES[slot] if axis in axis_map)
            prior = tensor_axes.get(region.tensor)
            if prior is not None and prior != axes:
                raise ValueError(f"tensor {region.tensor!r} has inconsistent axes {prior} and {axes}")
            tensor_axes[region.tensor] = axes
        bound = {
            slot: leaf.operand_bindings[slot].tensor
            for slot in contract_input_operands(contract)
            if slot in leaf.operand_bindings
        }
        inputs[nid] = bound
        predecessors[nid] = {}
        for slot, tensor in bound.items():
            producer = producers.get(tensor)
            predecessors[nid][slot] = producer
            if producer is not None:
                successors[producer].append(nid)
        if contract.output_operand not in leaf.operand_bindings:
            raise ValueError(f"{leaf.op_cls.__name__} output {contract.output_operand!r} is unbound")
        output = leaf.operand_bindings[contract.output_operand].tensor
        outputs[nid] = output
        output_slots = (contract.output_operand,)
        if isinstance(contract, ReductionContract) and contract.mapped_output_operand is not None:
            output_slots = (*output_slots, contract.mapped_output_operand)
        for slot in output_slots:
            if slot not in leaf.operand_bindings:
                raise ValueError(f"{leaf.op_cls.__name__} output {slot!r} is unbound")
            producers[leaf.operand_bindings[slot].tensor] = nid
        if isinstance(contract, InitializerContract):
            initializers.setdefault(output, []).append(nid)
    return ValueGraph(
        leaves=leaves,
        contracts=contracts,
        outputs=outputs,
        inputs=inputs,
        predecessors=predecessors,
        successors={nid: tuple(dict.fromkeys(items)) for nid, items in successors.items()},
        tensor_axes=tensor_axes,
        initializers={tensor: tuple(items) for tensor, items in initializers.items()},
    )


__all__ = ["ValueGraph", "build_value_graph", "contract_input_operands"]
