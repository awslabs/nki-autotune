"""Shared analysis for one-dimensional SPMD loop sharding."""

from __future__ import annotations

from collections.abc import Mapping

from nkigym.ir import KernelIR
from nkigym.ir.arith.expr import Expr, Var, substitute, to_affine
from nkigym.ir.tree import BlockNode, ForNode, ISANode

PROGRAM_SHARDS_ANNOTATION = "program_shards"


def configured_program_shards(ir: KernelIR) -> dict[int, int]:
    """Return the validated materialized-loop shard mapping."""
    value = ir.tree.block(ir.tree.root).annotations.get(PROGRAM_SHARDS_ANNOTATION, {})
    if not isinstance(value, dict) or any(
        not isinstance(loop_nid, int)
        or loop_nid not in ir.tree.graph
        or not isinstance(ir.tree.data(loop_nid), ForNode)
        or not isinstance(programs, int)
        or programs < 2
        for loop_nid, programs in value.items()
    ):
        raise ValueError(f"invalid {PROGRAM_SHARDS_ANNOTATION} annotation: {value!r}")
    return dict(value)


program_sharded_loops = configured_program_shards


def block_has_axis(ir: KernelIR, block_nid: int, axis: str) -> bool:
    """Return whether one block declares ``axis``."""
    return any(iter_var.axis == axis for iter_var in ir.tree.block(block_nid).iter_vars)


def axis_loop_for_block(ir: KernelIR, block_nid: int, axis: str) -> int | None:
    """Return the outermost local loop that materializes one block axis."""
    block = ir.tree.block(block_nid)
    leaf_nid = _direct_leaf(ir, block_nid)
    return None if leaf_nid is None else _axis_loop(ir, block_nid, leaf_nid, block, axis)


def owning_block(ir: KernelIR, leaf_nid: int) -> int:
    """Return the nearest block that directly owns one ISA leaf."""
    block_nid = next(
        (nid for nid in reversed(ir.tree.ancestors(leaf_nid)) if isinstance(ir.tree.data(nid), BlockNode)), None
    )
    if block_nid is None:
        raise ValueError(f"ISA leaf {leaf_nid} has no owning block")
    return block_nid


def operation_axis_iterations(
    ir: KernelIR, leaf_nid: int, abstract_axis: str, substitutions: Mapping[str, Expr]
) -> tuple[Expr, ...]:
    """Return active loop iterations implementing one operation axis."""
    block = ir.tree.block(owning_block(ir, leaf_nid))
    concrete_axis = block.axis_map.get(abstract_axis)
    values = [value for iter_var, value in zip(block.iter_vars, block.iter_values) if iter_var.axis == concrete_axis]
    if len(values) != 1:
        raise ValueError(f"operation axis {abstract_axis!r} has {len(values)} bindings")
    variables = to_affine(values[0])
    return tuple(
        substitutions.get(node.loop_var, Var(name=node.loop_var))
        for nid in ir.tree.ancestors(leaf_nid)
        if isinstance((node := ir.tree.data(nid)), ForNode) and node.loop_var in variables
    )


def operation_axis_value(ir: KernelIR, leaf_nid: int, concrete_axis: str, substitutions: Mapping[str, Expr]) -> Expr:
    """Return one operation's current iteration value for a concrete axis."""
    block = ir.tree.block(owning_block(ir, leaf_nid))
    values = [value for iter_var, value in zip(block.iter_vars, block.iter_values) if iter_var.axis == concrete_axis]
    if len(values) != 1:
        raise ValueError(f"operation axis {concrete_axis!r} has {len(values)} bindings")
    return substitute(values[0], dict(substitutions))


def _direct_leaf(ir: KernelIR, block_nid: int) -> int | None:
    """Return the sole ISA leaf directly owned by one block."""
    leaves = [
        nid
        for nid in ir.tree.preorder(block_nid)
        if isinstance(ir.tree.data(nid), ISANode) and owning_block(ir, nid) == block_nid
    ]
    return leaves[0] if len(leaves) == 1 else None


def _axis_loop(ir: KernelIR, block_nid: int, leaf_nid: int, block: BlockNode, axis: str) -> int | None:
    """Return the outermost enclosing loop contributing to one concrete axis."""
    values = [value for iter_var, value in zip(block.iter_vars, block.iter_values) if iter_var.axis == axis]
    result: int | None = None
    if len(values) == 1:
        variables = set(to_affine(values[0]))
        ancestors = ir.tree.ancestors(leaf_nid)
        if block_nid in ancestors:
            result = next(
                (
                    nid
                    for nid in ancestors
                    if isinstance(ir.tree.data(nid), ForNode) and ir.tree.loop(nid).loop_var in variables
                ),
                None,
            )
    return result
