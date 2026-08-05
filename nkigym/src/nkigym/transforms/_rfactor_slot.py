"""Slot-style RFactor for free-axis reductions."""

from __future__ import annotations

from dataclasses import dataclass, replace

from nkigym.ir import Const, KernelIR, Var, to_affine
from nkigym.ir.arith.expr import Expr
from nkigym.ir.tree import BlockNode, Buffer, BufferRegion, ForNode, ISANode, IterVar
from nkigym.ops.base import AxisRole, ReductionContract
from nkigym.ops.tensor_reduce import NKITensorReduce
from nkigym.transforms._canonical_rewrite import (
    append_root_buffers,
    finalize_rewrite,
    fresh_name,
    owning_block,
    single_leaf,
)
from nkigym.transforms._tree_ops import _replace_in_parent_children

_SUPPORTED_COMBINERS = frozenset({"add", "maximum", "multiply"})


@dataclass(frozen=True)
class _SlotMatch:
    """Resolved split reduction loop and its output."""

    loop_nid: int
    block_nid: int
    leaf_nid: int
    contract: ReductionContract
    reduction_axis: str
    output_region: BufferRegion
    output_abstract_axis: str


class SlotRFactor:
    """Rewrite one split reduction into partial slots and a final fold."""

    def rfactorable(self, ir: KernelIR, loop_nid: int) -> bool:
        """Return whether ``loop_nid`` satisfies the slot recipe."""
        return self._resolve(ir, loop_nid) is not None

    def emit(self, ir: KernelIR, loop_nid: int) -> None:
        """Apply the slot recipe in place and rebuild placement/dependencies."""
        match = self._resolve(ir, loop_nid)
        if match is None:
            raise AssertionError(f"slot RFactor match disappeared for loop {loop_nid}")

        tree = ir.tree
        loop = tree.loop(loop_nid)
        block = tree.block(match.block_nid)
        leaf = tree.isa(match.leaf_nid)
        output_buffer = ir.buffer(match.output_region.tensor)
        slot_buffer = Buffer(
            name=fresh_name(ir, f"{match.output_region.tensor}_rfactor"),
            shape=(output_buffer.shape[0], loop.extent),
            dtype="float32",
            location="sbuf",
        )
        partial_write = BufferRegion(
            tensor=slot_buffer.name, ranges=(match.output_region.ranges[0], (Var(name=loop.loop_var), Const(value=1)))
        )
        partial_read = BufferRegion(
            tensor=slot_buffer.name, ranges=(match.output_region.ranges[0], (Const(value=0), Const(value=loop.extent)))
        )

        append_root_buffers(ir, (slot_buffer,))
        self._retarget_reduction(ir, match, partial_write)
        final_block, final_loops, final_leaf = self._make_final_fold(ir, match, partial_read, loop.extent)
        self._insert_after_source(ir, match.block_nid, final_block, final_loops, final_leaf)
        finalize_rewrite(ir)

    def _resolve(self, ir: KernelIR, loop_nid: int) -> _SlotMatch | None:
        """Resolve one local split loop to a slot-style reduction."""
        result: _SlotMatch | None = None
        if loop_nid in ir.tree.graph and isinstance(ir.tree.data(loop_nid), ForNode):
            candidates = [
                nid
                for nid in ir.tree.descendants(loop_nid)
                if isinstance((node := ir.tree.data(nid)), ISANode) and node.op_cls.RFACTOR_RECIPE == "slot"
            ]
            if len(candidates) == 1:
                leaf_nid = candidates[0]
                block_nid = owning_block(ir.tree, leaf_nid)
                leaf = ir.tree.isa(leaf_nid)
                block = ir.tree.block(block_nid)
                contract = leaf.op_cls.algebraic_contract(leaf.kwargs)
                if isinstance(contract, ReductionContract):
                    result = self._validate_match(ir, loop_nid, block_nid, leaf_nid, block, leaf, contract)
        return result

    def _validate_match(
        self,
        ir: KernelIR,
        loop_nid: int,
        block_nid: int,
        leaf_nid: int,
        block: BlockNode,
        leaf: ISANode,
        contract: ReductionContract,
    ) -> _SlotMatch | None:
        """Return a match when the split loop and reduction metadata agree."""
        result: _SlotMatch | None = None
        reduction_axis = block.axis_map.get(contract.reduction_axis)
        reduction_value = self._iter_value(block, reduction_axis)
        local_axis_loops = self._local_axis_loops(ir, block_nid, leaf_nid, reduction_value)
        output_region = leaf.operand_bindings.get(contract.output_operand)
        output_axes = leaf.op_cls.OPERAND_AXES.get(contract.output_operand, ())
        input_region = leaf.operand_bindings.get(contract.input_operand)
        output_ranges = () if output_region is None else output_region.ranges
        valid_output = (
            output_region is not None
            and len(output_ranges) == 1
            and len(output_axes) == 1
            and output_region.tensor in ir.all_buffers()
            and len(ir.buffer(output_region.tensor).shape) == 1
        )
        valid = (
            reduction_axis is not None
            and reduction_value is not None
            and self._role(block, reduction_axis) == AxisRole.ACCUMULATION
            and local_axis_loops == [loop_nid]
            and ir.tree.children(loop_nid) == [leaf_nid]
            and single_leaf(ir.tree, block_nid) == leaf_nid
            and ir.tree.loop(loop_nid).extent > 1
            and contract.combinator.combiner in _SUPPORTED_COMBINERS
            and valid_output
            and input_region is not None
            and all(ir.tree.loop(loop_nid).loop_var not in to_affine(lower) for lower, _width in output_ranges)
            and any(ir.tree.loop(loop_nid).loop_var in to_affine(lower) for lower, _width in input_region.ranges)
        )
        if valid:
            assert output_region is not None
            assert reduction_axis is not None
            result = _SlotMatch(
                loop_nid=loop_nid,
                block_nid=block_nid,
                leaf_nid=leaf_nid,
                contract=contract,
                reduction_axis=reduction_axis,
                output_region=output_region,
                output_abstract_axis=output_axes[0],
            )
        return result

    def _iter_value(self, block: BlockNode, axis: str | None) -> Expr | None:
        """Return the iter value for ``axis``."""
        result: Expr | None = None
        if axis is not None:
            for iter_var, value in zip(block.iter_vars, block.iter_values):
                if iter_var.axis == axis:
                    result = value
                    break
        return result

    def _local_axis_loops(self, ir: KernelIR, block_nid: int, leaf_nid: int, value: Expr | None) -> list[int]:
        """Return local loops whose variables bind ``value``."""
        loops: list[int] = []
        if value is not None:
            binding_vars = {name for name in to_affine(value) if name is not None}
            ancestors = ir.tree.ancestors(leaf_nid)
            block_index = ancestors.index(block_nid)
            loops = [
                nid
                for nid in ancestors[block_index + 1 :]
                if isinstance((node := ir.tree.data(nid)), ForNode) and node.loop_var in binding_vars
            ]
        return loops

    def _role(self, block: BlockNode, axis: str) -> AxisRole:
        """Return one block axis role."""
        roles = [iter_var.role for iter_var in block.iter_vars if iter_var.axis == axis]
        if len(roles) != 1:
            raise ValueError(f"block has {len(roles)} roles for axis {axis!r}")
        return roles[0]

    def _retarget_reduction(self, ir: KernelIR, match: _SlotMatch, partial_write: BufferRegion) -> None:
        """Make each loop iteration write one independent partial slot."""
        block = ir.tree.block(match.block_nid)
        leaf = ir.tree.isa(match.leaf_nid)
        iter_vars = tuple(
            replace(iter_var, role=AxisRole.PARALLEL) if iter_var.axis == match.reduction_axis else iter_var
            for iter_var in block.iter_vars
        )
        writes = tuple(partial_write if region == match.output_region else region for region in block.writes)
        bindings = dict(leaf.operand_bindings)
        bindings[match.contract.output_operand] = partial_write
        ir.tree.graph.nodes[match.block_nid]["data"] = replace(block, iter_vars=iter_vars, writes=writes)
        ir.tree.graph.nodes[match.leaf_nid]["data"] = replace(leaf, operand_bindings=bindings)

    def _make_final_fold(
        self, ir: KernelIR, match: _SlotMatch, partial_read: BufferRegion, num_slots: int
    ) -> tuple[BlockNode, tuple[ForNode, ...], ISANode]:
        """Build the short reduction that closes the factored axis."""
        source_block = ir.tree.block(match.block_nid)
        output_axis = source_block.axis_map[match.output_abstract_axis]
        output_iter_var, output_iter_value = next(
            (iter_var, value)
            for iter_var, value in zip(source_block.iter_vars, source_block.iter_values)
            if iter_var.axis == output_axis
        )
        factor_axis = self._fresh_axis(ir)
        block = BlockNode(
            iter_vars=(
                replace(output_iter_var, role=AxisRole.PARALLEL),
                IterVar(axis=factor_axis, dom=(0, num_slots), role=AxisRole.ACCUMULATION),
            ),
            iter_values=(output_iter_value, Const(value=0)),
            reads=(partial_read,),
            writes=(match.output_region,),
            axis_map={"P": output_axis, "F": factor_axis},
        )
        leaf = ISANode(
            op_cls=NKITensorReduce,
            operand_bindings={"data": partial_read, "dst": match.output_region},
            kwargs={"op": match.contract.combinator.combiner, "axis": 1},
        )
        binding_vars = {name for name in to_affine(output_iter_value) if name is not None}
        ancestors = ir.tree.ancestors(match.leaf_nid)
        block_index = ancestors.index(match.block_nid)
        loops = tuple(
            node
            for nid in ancestors[block_index + 1 :]
            if isinstance((node := ir.tree.data(nid)), ForNode) and node.loop_var in binding_vars
        )
        return block, loops, leaf

    def _fresh_axis(self, ir: KernelIR) -> str:
        """Return a fresh dense concrete axis name."""
        axes = {iter_var.axis for block_nid in ir.tree.blocks() for iter_var in ir.tree.block(block_nid).iter_vars}
        index = 0
        while f"d{index}" in axes:
            index += 1
        return f"d{index}"

    def _insert_after_source(
        self, ir: KernelIR, source_block_nid: int, block: BlockNode, loops: tuple[ForNode, ...], leaf: ISANode
    ) -> None:
        """Insert a canonical final-fold block after the fused source block."""
        parent = ir.tree.parent(source_block_nid)
        if parent is None:
            raise AssertionError(f"slot source block {source_block_nid} has no parent")
        block_nid = ir.tree.add_node(block)
        cursor = block_nid
        for loop in loops:
            cursor = ir.tree.add_node(loop, parent=cursor)
        ir.tree.add_node(leaf, parent=cursor)
        _replace_in_parent_children(ir.tree, parent, [source_block_nid], [source_block_nid, block_nid])


__all__ = ["SlotRFactor"]
