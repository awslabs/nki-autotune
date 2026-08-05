"""Slot-style RFactor for free-axis reductions."""

from __future__ import annotations

from dataclasses import dataclass, replace

from nkigym.ir import Const, KernelIR, Var, to_affine
from nkigym.ir.arith.expr import Expr
from nkigym.ir.dependency import Dependency
from nkigym.ir.tree import BlockNode, Buffer, BufferRegion, ForNode, ISANode, IterVar
from nkigym.ops.base import AxisRole, ReductionContract
from nkigym.ops.tensor_reduce import NKITensorReduce
from nkigym.transforms._access_pattern import subtree_has_access_patterns
from nkigym.transforms._canonical_rewrite import append_root_buffers, fresh_name, owning_block, single_leaf
from nkigym.transforms._normalize import normalize_block
from nkigym.transforms._tile_region import retile_region
from nkigym.transforms._tree_ops import _replace_in_parent_children, invalidate_stale_software_pipelines
from nkigym.transforms.split import (
    _build_for_chain,
    _covers_exactly,
    _current_tensorize_width,
    _factorizations,
    _min_tile_floor,
)

_SUPPORTED_COMBINERS = frozenset({"add", "maximum", "multiply"})


@dataclass(frozen=True)
class _SlotMatch:
    """Resolved unsplit tensorized reduction and its output."""

    block_nid: int
    leaf_nid: int
    contract: ReductionContract
    reduction_axis: str
    output_region: BufferRegion
    output_abstract_axis: str


class SlotRFactor:
    """Factor one tensorized reduction into partial slots and a final fold.

    Slot-recipe reductions are not read-modify-write operations. Introducing
    only an outer loop and a narrower tile would overwrite the output with the
    last partial, so the factor loop, independent slots, and final fold are one
    inseparable RFactor rewrite rather than a Split followed by RFactor.
    """

    def analyze(self, ir: KernelIR) -> list[tuple[int, str, tuple[int, int]]]:
        """Return legal ``(leaf, axis, factors)`` slot-factorization choices."""
        options: list[tuple[int, str, tuple[int, int]]] = []
        for leaf_nid in ir.tree.preorder():
            node = ir.tree.data(leaf_nid)
            if not isinstance(node, ISANode) or node.op_cls.RFACTOR_RECIPE != "slot":
                continue
            block = ir.tree.block(owning_block(ir.tree, leaf_nid))
            contract = node.op_cls.algebraic_contract(node.kwargs)
            if not isinstance(contract, ReductionContract):
                continue
            target_axis = block.axis_map.get(contract.reduction_axis)
            if target_axis is None:
                continue
            current = _current_tensorize_width(node, block, target_axis)
            if current is None:
                continue
            for factors in _factorizations(current):
                if self.rfactorable(ir, leaf_nid, target_axis, factors):
                    options.append((leaf_nid, target_axis, factors))
        return options

    def rfactorable(self, ir: KernelIR, leaf_nid: int, target_axis: str, factors: tuple[int, int]) -> bool:
        """Return whether one unsplit tensorized reduction satisfies the slot recipe."""
        return self._resolve_source(ir, leaf_nid, target_axis, factors) is not None

    def emit(self, ir: KernelIR, leaf_nid: int, target_axis: str, factors: tuple[int, int]) -> None:
        """Apply the complete slot RFactor recipe in place."""
        match = self._resolve_source(ir, leaf_nid, target_axis, factors)
        if match is None:
            raise AssertionError(
                f"slot RFactor match disappeared for leaf {leaf_nid}, axis {target_axis!r}, factors {factors}"
            )

        output_buffer = ir.buffer(match.output_region.tensor)
        slot_buffer = Buffer(
            name=fresh_name(ir, f"{match.output_region.tensor}_rfactor"),
            shape=(output_buffer.shape[0], factors[0]),
            dtype="float32",
            location="sbuf",
        )
        append_root_buffers(ir, (slot_buffer,))
        loop_nid = self._factor_source_to_slots(ir, match, target_axis, factors, slot_buffer.name)
        loop = ir.tree.loop(loop_nid)
        partial_read = BufferRegion(
            tensor=slot_buffer.name, ranges=(match.output_region.ranges[0], (Const(value=0), Const(value=loop.extent)))
        )

        final_block, final_loops, final_leaf = self._make_final_fold(ir, match, partial_read, loop.extent)
        self._insert_after_source(ir, match.block_nid, final_block, final_loops, final_leaf)
        invalidate_stale_software_pipelines(ir)
        ir.dependency = Dependency(ir.tree)

    def _resolve_source(
        self, ir: KernelIR, leaf_nid: int, target_axis: str, factors: tuple[int, int]
    ) -> _SlotMatch | None:
        """Resolve a reduction whose loop, slots, and fold can be factored together."""
        result: _SlotMatch | None = None
        if leaf_nid in ir.tree.graph and isinstance(ir.tree.data(leaf_nid), ISANode):
            leaf = ir.tree.isa(leaf_nid)
            if leaf.op_cls.RFACTOR_RECIPE == "slot" and not subtree_has_access_patterns(ir.tree, leaf_nid):
                block_nid = owning_block(ir.tree, leaf_nid)
                block = ir.tree.block(block_nid)
                contract = leaf.op_cls.algebraic_contract(leaf.kwargs)
                current = _current_tensorize_width(leaf, block, target_axis)
                floor = _min_tile_floor(leaf, block, target_axis)
                reduction_axis = (
                    block.axis_map.get(contract.reduction_axis) if isinstance(contract, ReductionContract) else None
                )
                reduction_value = self._iter_value(block, reduction_axis)
                output_region = (
                    leaf.operand_bindings.get(contract.output_operand)
                    if isinstance(contract, ReductionContract)
                    else None
                )
                output_axes = (
                    leaf.op_cls.OPERAND_AXES.get(contract.output_operand, ())
                    if isinstance(contract, ReductionContract)
                    else ()
                )
                input_region = (
                    leaf.operand_bindings.get(contract.input_operand)
                    if isinstance(contract, ReductionContract)
                    else None
                )
                roles = [iter_var.role for iter_var in block.iter_vars if iter_var.axis == target_axis]
                output_buffer = (
                    ir.buffer(output_region.tensor)
                    if output_region is not None and output_region.tensor in ir.all_buffers()
                    else None
                )
                valid = (
                    isinstance(contract, ReductionContract)
                    and reduction_axis == target_axis
                    and roles == [AxisRole.ACCUMULATION]
                    and reduction_value is not None
                    and self._local_axis_loops(ir, block_nid, leaf_nid, reduction_value) == []
                    and single_leaf(ir.tree, block_nid) == leaf_nid
                    and contract.combinator.combiner in _SUPPORTED_COMBINERS
                    and contract.output_operand not in leaf.op_cls.rmw_operands(leaf.kwargs)
                    and output_region is not None
                    and len(output_region.ranges) == 1
                    and len(output_axes) == 1
                    and output_buffer is not None
                    and len(output_buffer.shape) == 1
                    and input_region is not None
                    and len(factors) == 2
                    and all(factor >= 2 for factor in factors)
                    and current is not None
                    and _covers_exactly(factors, current)
                    and (floor is None or factors[-1] >= floor)
                )
                if valid:
                    assert isinstance(contract, ReductionContract)
                    assert reduction_axis is not None
                    assert output_region is not None
                    result = _SlotMatch(
                        block_nid=block_nid,
                        leaf_nid=leaf_nid,
                        contract=contract,
                        reduction_axis=reduction_axis,
                        output_region=output_region,
                        output_abstract_axis=next(iter(output_axes)),
                    )
        return result

    def _factor_source_to_slots(
        self, ir: KernelIR, match: _SlotMatch, target_axis: str, factors: tuple[int, int], slot_name: str
    ) -> int:
        """Create the factor loop and make every iteration write one partial slot."""
        leaf = ir.tree.isa(match.leaf_nid)
        parent_nid = ir.tree.parent(match.leaf_nid)
        if parent_nid is None:
            raise AssertionError(f"slot reduction leaf {match.leaf_nid} has no parent")
        block = ir.tree.block(match.block_nid)

        top_nid, bottom_nid = _build_for_chain(ir.tree, f"i_{target_axis}", factors[:-1])
        factor_loop = ir.tree.loop(top_nid)
        partial_write = BufferRegion(
            tensor=slot_name, ranges=(match.output_region.ranges[0], (Var(name=factor_loop.loop_var), Const(value=1)))
        )

        inverse_axis_map = {concrete: abstract for abstract, concrete in block.axis_map.items()}
        abstract_axis = inverse_axis_map.get(target_axis)
        new_width = factors[-1]

        def set_width(lo: Expr, _width: int) -> tuple[Expr, int]:
            """Keep the offset while setting the factored reduction width."""
            return lo, new_width

        bindings = {
            slot: retile_region(region, leaf.op_cls.OPERAND_AXES[slot], abstract_axis, set_width)
            for slot, region in leaf.operand_bindings.items()
        }
        bindings[match.contract.output_operand] = partial_write

        tensor_to_axes = {
            leaf.operand_bindings[slot].tensor: leaf.op_cls.OPERAND_AXES[slot] for slot in leaf.operand_bindings
        }
        iter_vars = tuple(
            replace(iter_var, role=AxisRole.PARALLEL) if iter_var.axis == match.reduction_axis else iter_var
            for iter_var in block.iter_vars
        )
        writes = tuple(
            (
                partial_write
                if region == match.output_region
                else retile_region(region, tensor_to_axes.get(region.tensor, ()), abstract_axis, set_width)
            )
            for region in block.writes
        )
        ir.tree.graph.nodes[match.block_nid]["data"] = replace(
            block,
            iter_vars=iter_vars,
            reads=tuple(
                retile_region(region, tensor_to_axes.get(region.tensor, ()), abstract_axis, set_width)
                for region in block.reads
            ),
            writes=writes,
        )
        ir.tree.graph.nodes[match.leaf_nid]["data"] = replace(leaf, operand_bindings=bindings)
        ir.tree.graph.add_edge(bottom_nid, match.leaf_nid)
        _replace_in_parent_children(ir.tree, parent_nid, [match.leaf_nid], [top_nid])
        normalize_block(ir.tree, match.block_nid)
        return top_nid

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
