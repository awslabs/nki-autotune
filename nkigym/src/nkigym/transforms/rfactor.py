"""RFactor transforms for read-modify-write and slot-style reductions."""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace

from nkigym.ir import KernelIR
from nkigym.ir.arith.expr import Const, Expr, NonAffineError, Var, substitute, to_affine
from nkigym.ir.dependency import Dependency
from nkigym.ir.tree import PARTITION_DIM, BlockNode, Buffer, BufferRegion, ForNode, ISANode, IterVar, KernelTree
from nkigym.ops.base import AxisRole, NKIOp, ReductionContract
from nkigym.ops.memset import NKIMemset
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.ops.tensor_reduce import NKITensorReduce
from nkigym.ops.tensor_tensor import NKITensorTensor
from nkigym.transforms.base import Transform, TransformLegalityError, TransformOption
from nkigym.transforms.helper.access_pattern import subtree_has_access_patterns
from nkigym.transforms.helper.canonical_rewrite import append_root_buffers, fresh_name, owning_block, single_leaf
from nkigym.transforms.helper.normalize import normalize_block
from nkigym.transforms.helper.tile_region import retile_region
from nkigym.transforms.helper.tree_ops import _replace_in_parent_children, invalidate_stale_software_pipelines
from nkigym.transforms.split import (
    _build_for_chain,
    _covers_exactly,
    _current_tensorize_width,
    _factorizations,
    _min_tile_floor,
)

_RMW_STAGING_BUFFER = "sbuf_rfactor"
_RMW_COMBINERS = frozenset({"add", "multiply"})
_SUPPORTED_COMBINERS = frozenset({"add", "maximum", "multiply"})


def _role_of(block: BlockNode, axis: str) -> AxisRole:
    """Return the role ``block`` assigns to ``axis``, defaulting to parallel."""
    role = next((iter_var.role for iter_var in block.iter_vars if iter_var.axis == axis), AxisRole.PARALLEL)
    return role


@dataclass(frozen=True)
class _SlotMatch:
    """Resolved unsplit tensorized reduction and its output."""

    block_nid: int
    leaf_nid: int
    contract: ReductionContract
    reduction_axis: str
    output_region: BufferRegion
    output_abstract_axis: str


class _SlotRFactor:
    """Factor one tensorized reduction into partial slots and a final fold."""

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
                    and not self._has_nested_output_touch(ir, block_nid, leaf_nid, output_region)
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

    def _has_nested_output_touch(
        self, ir: KernelIR, block_nid: int, leaf_nid: int, output_region: BufferRegion | None
    ) -> bool:
        """Return whether an output dependency executes inside the source block."""
        nested = ir.tree.descendants(block_nid)
        output_tensor = output_region.tensor if output_region is not None else None
        result = False
        if output_tensor is not None:
            for consumer in ir.dependency.direct_consumers(leaf_nid):
                info = ir.dependency.info(consumer)
                if consumer in nested and output_tensor in info.reads | info.writes:
                    result = True
                    break
        return result

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


@dataclass(frozen=True)
class RFactorOption(TransformOption):
    """Describe an RMW loop or slot-style reduction factorization."""

    target_loop_nid: int
    factor_axis: int = 0
    factors: tuple[int, int] | None = None
    target_axis: str | None = None


class RFactor(Transform[RFactorOption]):
    """One-stage → two-stage accumulation: per-``ko`` PSUM partial + SBUF fold."""

    def analyze(self, ir: KernelIR) -> list[RFactorOption]:
        """Enumerate every fully legal ACCUMULATION loop of an rfactorable op."""
        options: list[RFactorOption] = []
        for nid in ir.tree.preorder():
            if not isinstance(ir.tree.data(nid), ForNode):
                continue
            if self._rfactorable(ir, nid):
                options.append(RFactorOption(target_loop_nid=nid, factor_axis=0))
        for leaf_nid, target_axis, factors in _SlotRFactor().analyze(ir):
            options.append(
                RFactorOption(target_loop_nid=leaf_nid, factor_axis=0, factors=factors, target_axis=target_axis)
            )
        return options

    def apply(self, ir: KernelIR, option: RFactorOption) -> KernelIR:
        """Re-check legality, deep-copy, emit the two-stage accumulation, return."""
        self._check_legality(ir, option)
        new_ir = copy.deepcopy(ir)
        if option.factors is not None and option.target_axis is not None:
            _SlotRFactor().emit(new_ir, option.target_loop_nid, option.target_axis, option.factors)
        else:
            self._emit_rmw(new_ir, option)
        return new_ir

    def _rfactorable(self, ir: KernelIR, loop_nid: int) -> bool:
        """Return whether ``loop_nid`` supports the RMW recipe."""
        leaf = self._owning_matmul_leaf(ir, loop_nid)
        result = False
        if leaf is not None:
            op_cls = ir.tree.isa(leaf).op_cls
            block_nid = self._enclosing_block_nid(ir.tree, leaf)
            block = ir.tree.block(block_nid)
            axis = self._loop_axis(ir, loop_nid, block)
            axis_loops: list[int] = []
            if axis is not None:
                binding_vars = self._axis_binding_loopvars(block, axis)
                axis_loops = [
                    nid
                    for nid in ir.tree.ancestors(leaf)
                    if isinstance((node := ir.tree.data(nid)), ForNode)
                    and node.loop_var in binding_vars
                    and block_nid in ir.tree.ancestors(nid)
                ]
            if (
                self._supports_rmw_op(op_cls)
                and axis is not None
                and _role_of(block, axis) == AxisRole.ACCUMULATION
                and len(axis_loops) == 2
                and axis_loops[0] == loop_nid
                and self._init_block_is_retargetable(ir, loop_nid, leaf)
                and self._drain_block_is_removable(ir, loop_nid, leaf)
                and self._gadget_region_fits_output(ir, loop_nid, leaf)
            ):
                result = True
        return result

    def _supports_rmw_op(self, op_cls: type[NKIOp]) -> bool:
        """Whether ``op_cls`` satisfies the currently implemented 2D PSUM recipe."""
        reduction_axes = [axis for axis, role in op_cls.AXIS_ROLES.items() if role == AxisRole.ACCUMULATION]
        output_axes = op_cls.OPERAND_AXES.get("dst", ())
        reducer = op_cls.REDUCE_COMBINATOR
        return (
            op_cls.RFACTOR_RECIPE == "rmw"
            and reducer is not None
            and reducer.combiner in _RMW_COMBINERS
            and op_cls.RMW_OPERANDS == frozenset({"dst"})
            and op_cls.OUTPUT_LOCATION == "psum"
            and len(reduction_axes) == 1
            and len(output_axes) == 2
            and reduction_axes[0] not in output_axes
        )

    def _init_block_is_retargetable(self, ir: KernelIR, loop_nid: int, matmul_leaf: int) -> bool:
        """Whether one canonical identity memset initializes PSUM outside ``ko``."""
        matmul = ir.tree.data(matmul_leaf)
        assert isinstance(matmul, ISANode)
        psum_name = matmul.operand_bindings["dst"].tensor
        reducer = matmul.op_cls.REDUCE_COMBINATOR
        assert reducer is not None
        inits = [
            nid
            for nid in ir.tree.preorder()
            if isinstance((node := ir.tree.data(nid)), ISANode)
            and node.op_cls.NAME == "memset"
            and node.operand_bindings["dst"].tensor == psum_name
        ]
        result = False
        if len(inits) == 1:
            init_nid = inits[0]
            init_block = self._enclosing_block_nid(ir.tree, init_nid)
            preorder = list(ir.tree.preorder())
            init = ir.tree.isa(init_nid)
            result = (
                single_leaf(ir.tree, init_block) == init_nid
                and loop_nid not in ir.tree.ancestors(init_nid)
                and preorder.index(init_nid) < preorder.index(matmul_leaf)
                and init.kwargs.get("value") == float(reducer.identity)
            )
        return result

    def _drain_block_is_removable(self, ir: KernelIR, loop_nid: int, matmul_leaf: int) -> bool:
        """Whether the sole drain is an outside-``ko`` identity copy in its own block."""
        matmul = ir.tree.data(matmul_leaf)
        assert isinstance(matmul, ISANode)
        psum_name = matmul.operand_bindings["dst"].tensor
        drains: list[int] = []
        for nid in ir.tree.preorder():
            node = ir.tree.data(nid)
            if (
                isinstance(node, ISANode)
                and node.op_cls.NAME == "tensor_copy"
                and node.operand_bindings["src"].tensor == psum_name
            ):
                drains.append(nid)
        result = False
        if len(drains) == 1:
            drain_nid = drains[0]
            drain_block = self._enclosing_block_nid(ir.tree, drain_nid)
            leaves = [nid for nid in ir.tree.preorder(drain_block) if isinstance(ir.tree.data(nid), ISANode)]
            drain = ir.tree.isa(drain_nid)
            out_name = drain.operand_bindings["dst"].tensor
            preorder = list(ir.tree.preorder())
            result = (
                leaves == drains
                and loop_nid not in ir.tree.ancestors(drain_nid)
                and preorder.index(matmul_leaf) < preorder.index(drain_nid)
                and drain.operand_bindings["src"].ranges == drain.operand_bindings["dst"].ranges
                and ir.buffer(out_name).location == "sbuf"
            )
        return result

    def _gadget_region_fits_output(self, ir: KernelIR, loop_nid: int, matmul_leaf: int) -> bool:
        """Return whether the generated region fits PSUM and SBUF."""
        ki_nid = self._ki_loop_nid(ir, loop_nid)
        matmul = ir.tree.data(matmul_leaf)
        assert isinstance(matmul, ISANode)
        psum_name = matmul.operand_bindings["dst"].tensor
        out_name = self._drain_out_tensor(ir.tree, psum_name)
        dst_region = matmul.operand_bindings["dst"]
        free_footprint = self._free_footprint(ir, ki_nid, matmul_leaf)
        loop_extents = {
            node.loop_var: node.extent
            for nid in ir.tree.ancestors(matmul_leaf)
            if isinstance((node := ir.tree.data(nid)), ForNode)
        }
        result = False
        if free_footprint is not None:
            free_lo, free_extent = free_footprint
            region = self._partition_region(out_name, dst_region.ranges[0][0], free_lo, free_extent)
            result = all(
                self._region_axis_fits(lo, width, axis, buf, loop_extents)
                for buf in (ir.buffer(psum_name), ir.buffer(out_name))
                for axis, (lo, width) in enumerate(region.ranges)
            )
        return result

    def _region_axis_fits(self, lo: Expr, width: Expr, axis: int, buf: Buffer, loop_extents: dict[str, int]) -> bool:
        """Return whether one affine gadget axis stays within ``buf.shape``."""
        result = False
        coeffs: dict[str | None, int] | None = None
        width_value = width.value if isinstance(width, Const) else None
        if width_value is not None:
            try:
                coeffs = to_affine(lo)
            except (NonAffineError, TypeError):
                coeffs = None
        if coeffs is not None and width_value is not None:
            lower = coeffs.get(None, 0)
            upper = lower
            bounded = axis < len(buf.shape)
            for var, coeff in coeffs.items():
                if var is None:
                    continue
                extent = loop_extents.get(var)
                if extent is None:
                    bounded = False
                    break
                span = coeff * (extent - 1)
                if span < 0:
                    lower += span
                else:
                    upper += span
            if axis == 0 and buf.location in ("sbuf", "psum") and width_value == PARTITION_DIM:
                lower *= PARTITION_DIM
                upper *= PARTITION_DIM
            result = bounded and lower >= 0 and upper + width_value <= buf.shape[axis]
        return result

    def _check_legality(self, ir: KernelIR, option: RFactorOption) -> None:
        """Raise TransformLegalityError if the option is not a valid RFactor."""
        has_factors = option.factors is not None
        has_target_axis = option.target_axis is not None
        if has_factors != has_target_axis:
            raise TransformLegalityError("RFactor slot options must provide both factors and target_axis")
        if has_factors:
            self._check_slot_legality(ir, option)
        else:
            self._check_rmw_legality(ir, option)

    def _check_slot_legality(self, ir: KernelIR, option: RFactorOption) -> None:
        """Raise when ``option`` is not a complete slot-style RFactor."""
        if option.factor_axis != 0:
            raise TransformLegalityError(f"RFactor factor_axis must be 0 for the slot recipe; got {option.factor_axis}")
        factors = option.factors
        target_axis = option.target_axis
        if factors is None or target_axis is None:
            raise AssertionError("slot RFactor legality requires factors and target_axis")
        if not _SlotRFactor().rfactorable(ir, option.target_loop_nid, target_axis, factors):
            raise TransformLegalityError(
                f"RFactor target {option.target_loop_nid} is not a legal slot reduction "
                f"for axis {target_axis!r} and factors {factors}"
            )

    def _check_rmw_legality(self, ir: KernelIR, option: RFactorOption) -> None:
        """Raise TransformLegalityError if the option is not a valid rmw RFactor."""
        nid = option.target_loop_nid
        if nid not in ir.tree.graph or not isinstance(ir.tree.data(nid), ForNode):
            raise TransformLegalityError(f"RFactor target {nid} is not a ForNode in the tree")
        if option.factor_axis != 0:
            raise TransformLegalityError(
                f"RFactor factor_axis must be 0 for the fused rmw recipe; got {option.factor_axis}"
            )
        if not self._rfactorable(ir, nid):
            raise TransformLegalityError(
                f"RFactor target loop {nid} is not a legal reduction: an rmw recipe must be "
                f"the outermost of exactly two loops binding an ACCUMULATION axis, "
                f"have canonical outside-loop init and identity-mapped drain blocks, "
                f"use a supported combiner, and fit a contiguous gadget footprint "
                f"within PSUM/output capacity"
            )

    def _emit_rmw(self, ir: KernelIR, option: RFactorOption) -> None:
        """Emit the fused per-``ko`` PSUM partial and SBUF fold."""
        tree = ir.tree
        ko_loop = tree.data(option.target_loop_nid)
        assert isinstance(ko_loop, ForNode)
        ko_var = ko_loop.loop_var

        matmul_leaf = self._owning_matmul_leaf(ir, option.target_loop_nid)
        assert matmul_leaf is not None
        matmul_block_nid = self._enclosing_block_nid(tree, matmul_leaf)
        matmul_node = tree.data(matmul_leaf)
        assert isinstance(matmul_node, ISANode)
        op_cls = matmul_node.op_cls
        reducer = op_cls.REDUCE_COMBINATOR
        assert reducer is not None

        psum_name = matmul_node.operand_bindings["dst"].tensor
        out_name = self._drain_out_tensor(tree, psum_name)
        staging_name = self._staging_buffer_name(ir)
        identity = float(reducer.identity)
        combiner = reducer.combiner

        ki_nid = self._ki_loop_nid(ir, option.target_loop_nid)
        footprint = self._footprint(ir, ki_nid, matmul_leaf)
        part_lo = matmul_node.operand_bindings["dst"].ranges[0][0]
        free_footprint = self._free_footprint(ir, ki_nid, matmul_leaf)
        assert free_footprint is not None
        free_lo, free_extent = free_footprint

        self._add_rf_buffer(ir, psum_name, out_name, staging_name)
        self._flip_matmul_k_role(tree, matmul_block_nid)
        self._retarget_init(tree, psum_name, out_name)
        self._remove_flat_block(tree, self._reader_leaf(tree, psum_name, "tensor_copy"))
        self._nest_memset(
            ir, matmul_leaf, ki_nid, psum_name, ko_var, footprint, part_lo, free_lo, free_extent, identity
        )
        copy_block_nid = self._nest_copy(
            ir, matmul_leaf, ki_nid, psum_name, staging_name, ko_var, footprint, part_lo, free_lo, free_extent
        )
        self._nest_combine(
            ir,
            matmul_leaf,
            ki_nid,
            copy_block_nid,
            out_name,
            staging_name,
            ko_var,
            footprint,
            part_lo,
            free_lo,
            free_extent,
            combiner,
        )

        invalidate_stale_software_pipelines(ir)
        ir.dependency = Dependency(tree)

    def _enclosing_block_nid(self, tree: KernelTree, nid: int) -> int:
        """Nearest enclosing BlockNode nid of ``nid`` (deepest ancestor block)."""
        for anc in reversed(tree.ancestors(nid)):
            if isinstance(tree.data(anc), BlockNode):
                return anc
        raise TransformLegalityError(f"no enclosing BlockNode for {nid}")

    def _ki_loop_nid(self, ir: KernelIR, ko_loop_nid: int) -> int:
        """Return the innermost accumulation loop at or below ``ko``."""
        tree = ir.tree
        matmul_leaf = self._owning_matmul_leaf(ir, ko_loop_nid)
        assert matmul_leaf is not None
        block_nid = self._enclosing_block_nid(tree, matmul_leaf)
        block = tree.block(block_nid)
        op_cls = self._op_cls_of_block(tree, block_nid)
        reduction_abstract = next(a for a, role in op_cls.AXIS_ROLES.items() if role == AxisRole.ACCUMULATION)
        k_axis = block.axis_map[reduction_abstract]
        k_binding_vars = self._axis_binding_loopvars(block, k_axis)
        k_loops = [
            a
            for a in tree.ancestors(matmul_leaf)
            if isinstance((node := tree.data(a)), ForNode)
            and node.loop_var in k_binding_vars
            and block_nid in tree.ancestors(a)
        ]
        return k_loops[-1]

    def _axis_binding_loopvars(self, block: BlockNode, axis: str) -> set[str]:
        """Loop vars appearing in the iter_value of ``axis`` (the loops that bind it)."""
        value = next(v for iv, v in zip(block.iter_vars, block.iter_values) if iv.axis == axis)
        return {n for n in to_affine(value) if n is not None}

    def _footprint(self, ir: KernelIR, ki_loop_nid: int, matmul_leaf: int) -> list[tuple[str, int]]:
        """Return partition loops strictly between ``ki`` and the matmul."""
        tree = ir.tree
        block = self._enclosing_block(ir, matmul_leaf)
        m_abstract = tree.isa(matmul_leaf).op_cls.OPERAND_AXES["dst"][0]
        m_axis = block.axis_map[m_abstract]
        m_binding_vars = self._axis_binding_loopvars(block, m_axis)
        between = [
            a
            for a in tree.ancestors(matmul_leaf)
            if isinstance(tree.data(a), ForNode) and ki_loop_nid in tree.ancestors(a)
        ]
        return [
            (tree.loop(a).loop_var, tree.loop(a).extent) for a in between if tree.loop(a).loop_var in m_binding_vars
        ]

    def _free_footprint(self, ir: KernelIR, ki_loop_nid: int, matmul_leaf: int) -> tuple[Expr, int] | None:
        """Return one contiguous free-axis interval swept below ``ki``, if representable."""
        tree = ir.tree
        block = self._enclosing_block(ir, matmul_leaf)
        matmul = tree.isa(matmul_leaf)
        dst_region = matmul.operand_bindings["dst"]
        free_abstract = matmul.op_cls.OPERAND_AXES["dst"][1]
        free_axis = block.axis_map[free_abstract]
        free_binding_vars = self._axis_binding_loopvars(block, free_axis)
        tile_width = dst_region.ranges[1][1]
        absorbed = {
            node.loop_var: node.extent
            for a in tree.ancestors(matmul_leaf)
            if isinstance((node := tree.data(a)), ForNode)
            and ki_loop_nid in tree.ancestors(a)
            and node.loop_var in free_binding_vars
        }
        result: tuple[Expr, int] | None = None
        coeffs: dict[str | None, int] | None = None
        try:
            coeffs = to_affine(dst_region.ranges[1][0])
        except (NonAffineError, TypeError):
            coeffs = None
        if coeffs is not None and isinstance(tile_width, Const):
            strides = sorted((coeffs.get(var, 0), var, extent) for var, extent in absorbed.items())
            width = tile_width.value
            contiguous = True
            for stride, _var, extent in strides:
                if stride != width:
                    contiguous = False
                    break
                width *= extent
            if contiguous:
                free_lo = substitute(dst_region.ranges[1][0], {var: Const(value=0) for var in absorbed})
                result = free_lo, width
        return result

    def _op_cls_of_block(self, tree: KernelTree, block_nid: int) -> type[NKIOp]:
        """Return the sole rfactorable op class under ``block_nid``."""
        leaves = [
            nid
            for nid in tree.descendants(block_nid)
            if isinstance(tree.data(nid), ISANode) and owning_block(tree, nid) == block_nid
        ]
        rfactorable = [n for n in leaves if tree.isa(n).op_cls.RFACTOR_RECIPE is not None]
        if len(rfactorable) != 1:
            raise TransformLegalityError(
                f"block {block_nid} must own exactly one rfactorable leaf; got {len(rfactorable)}"
            )
        return tree.isa(rfactorable[0]).op_cls

    def _drain_out_tensor(self, tree: KernelTree, psum_name: str) -> str:
        """Tensor the drain ``tensor_copy`` writes (reads ``psum_name``, writes SBUF out)."""
        for nid in tree.preorder():
            data = tree.data(nid)
            if isinstance(data, ISANode) and data.op_cls.NAME == "tensor_copy":
                if data.operand_bindings["src"].tensor == psum_name:
                    return data.operand_bindings["dst"].tensor
        raise TransformLegalityError(f"no drain tensor_copy reading {psum_name!r}")

    def _staging_buffer_name(self, ir: KernelIR) -> str:
        """Return the first available deterministic staging-buffer name."""
        names = set(ir.all_buffers())
        candidate = _RMW_STAGING_BUFFER
        suffix = 0
        while candidate in names:
            suffix += 1
            candidate = f"{_RMW_STAGING_BUFFER}_{suffix}"
        return candidate

    def _add_rf_buffer(self, ir: KernelIR, psum_name: str, out_name: str, staging_name: str) -> None:
        """Add the full-frame SBUF staging buffer beside the PSUM."""
        tree = ir.tree
        out_buf = ir.buffer(out_name)
        rf_buf = Buffer(
            name=staging_name,
            shape=out_buf.shape,
            dtype=out_buf.dtype,
            location="sbuf",
            storage_dtype=out_buf.storage_dtype,
        )
        for nid in tree.blocks():
            block = tree.data(nid)
            assert isinstance(block, BlockNode)
            if any(buf.name == psum_name for buf in block.alloc_buffers):
                tree.graph.nodes[nid]["data"] = replace(block, alloc_buffers=(*block.alloc_buffers, rf_buf))
                return
        raise TransformLegalityError(f"no block allocates {psum_name!r}")

    def _flip_matmul_k_role(self, tree: KernelTree, block_nid: int) -> None:
        """Flip the matmul K role from accumulation to parallel."""
        block = tree.data(block_nid)
        assert isinstance(block, BlockNode)
        op_cls = self._op_cls_of_block(tree, block_nid)
        reduction_abstract = next(a for a, role in op_cls.AXIS_ROLES.items() if role == AxisRole.ACCUMULATION)
        k_axis = block.axis_map[reduction_abstract]
        new_iter_vars = tuple(
            replace(iv, role=AxisRole.PARALLEL) if iv.axis == k_axis else iv for iv in block.iter_vars
        )
        tree.graph.nodes[block_nid]["data"] = replace(block, iter_vars=new_iter_vars)

    def _retarget_init(self, tree: KernelTree, psum_name: str, out_name: str) -> None:
        """Retarget the flat PSUM memset to the SBUF accumulator."""
        leaf_nid = self._writer_leaf(tree, psum_name, "memset")
        block_nid = self._enclosing_block_nid(tree, leaf_nid)
        block = tree.data(block_nid)
        assert isinstance(block, BlockNode)
        new_writes = tuple(self._retarget_region(w, psum_name, out_name) for w in block.writes)
        new_reads = tuple(self._retarget_region(r, psum_name, out_name) for r in block.reads)
        tree.graph.nodes[block_nid]["data"] = replace(block, reads=new_reads, writes=new_writes)
        leaf = tree.data(leaf_nid)
        assert isinstance(leaf, ISANode)
        new_bindings = {
            slot: self._retarget_region(region, psum_name, out_name) for slot, region in leaf.operand_bindings.items()
        }
        tree.graph.nodes[leaf_nid]["data"] = ISANode(
            op_cls=leaf.op_cls, operand_bindings=new_bindings, kwargs=dict(leaf.kwargs)
        )

    def _retarget_region(self, region: BufferRegion, old_tensor: str, new_tensor: str) -> BufferRegion:
        """Rename ``region``'s tensor ``old_tensor`` → ``new_tensor`` (ranges unchanged)."""
        if region.tensor != old_tensor:
            return region
        return BufferRegion(tensor=new_tensor, ranges=region.ranges)

    def _nest_memset(
        self,
        ir: KernelIR,
        matmul_leaf: int,
        ki_nid: int,
        psum_name: str,
        ko_var: str,
        footprint: list[tuple[str, int]],
        part_lo: Expr,
        free_lo: Expr,
        free_extent: int,
        identity: float,
    ) -> None:
        """Splice a PSUM memset before ``ki``."""
        region = self._partition_region(psum_name, part_lo, free_lo, free_extent)
        block = self._gadget_block(
            ir, matmul_leaf, ko_var, footprint, free_extent, AxisRole.PARALLEL, reads=(), writes=(region,)
        )
        leaf = ISANode(op_cls=NKIMemset, operand_bindings={"dst": region}, kwargs={"value": identity})
        self._splice_beside_ki(ir.tree, ki_nid, block, footprint, leaf, insert_after=None)

    def _nest_copy(
        self,
        ir: KernelIR,
        matmul_leaf: int,
        ki_nid: int,
        psum_name: str,
        staging_name: str,
        ko_var: str,
        footprint: list[tuple[str, int]],
        part_lo: Expr,
        free_lo: Expr,
        free_extent: int,
    ) -> int:
        """Splice the PSUM-to-SBUF copy after ``ki``."""
        src = self._partition_region(psum_name, part_lo, free_lo, free_extent)
        dst = self._partition_region(staging_name, part_lo, free_lo, free_extent)
        block = self._gadget_block(
            ir, matmul_leaf, ko_var, footprint, free_extent, AxisRole.PARALLEL, reads=(src,), writes=(dst,)
        )
        leaf = ISANode(op_cls=NKITensorCopy, operand_bindings={"src": src, "dst": dst}, kwargs={})
        return self._splice_beside_ki(ir.tree, ki_nid, block, footprint, leaf, insert_after=ki_nid)

    def _nest_combine(
        self,
        ir: KernelIR,
        matmul_leaf: int,
        ki_nid: int,
        copy_block_nid: int,
        out_name: str,
        staging_name: str,
        ko_var: str,
        footprint: list[tuple[str, int]],
        part_lo: Expr,
        free_lo: Expr,
        free_extent: int,
        combiner: str,
    ) -> None:
        """Splice the cross-``ko`` SBUF fold after the copy."""
        out_region = self._partition_region(out_name, part_lo, free_lo, free_extent)
        rf_region = self._partition_region(staging_name, part_lo, free_lo, free_extent)
        block = self._gadget_block(
            ir,
            matmul_leaf,
            ko_var,
            footprint,
            free_extent,
            AxisRole.ACCUMULATION,
            reads=(out_region, rf_region),
            writes=(out_region,),
        )
        leaf = ISANode(
            op_cls=NKITensorTensor,
            operand_bindings={"data1": out_region, "data2": rf_region, "dst": out_region},
            kwargs={"op": combiner},
        )
        self._splice_beside_ki(ir.tree, ki_nid, block, footprint, leaf, insert_after=copy_block_nid)

    def _remove_flat_block(self, tree: KernelTree, leaf_nid: int) -> None:
        """Delete the canonical flat block owning ``leaf_nid``."""
        block_nid = self._enclosing_block_nid(tree, leaf_nid)
        parent = tree.parent(block_nid)
        assert parent is not None
        remaining = [c for c in tree.children(parent) if c != block_nid]
        _replace_in_parent_children(tree, parent, [block_nid], [])
        assert tree.children(parent) == remaining
        for nid in tree.descendants(block_nid) | {block_nid}:
            tree.graph.remove_node(nid)

    def _partition_region(self, tensor: str, part_lo: Expr, free_lo: Expr, free_extent: int) -> BufferRegion:
        """Build the canonical partition/free-axis region."""
        return BufferRegion(
            tensor=tensor, ranges=((part_lo, Const(value=PARTITION_DIM)), (free_lo, Const(value=free_extent)))
        )

    def _gadget_block(
        self,
        ir: KernelIR,
        matmul_leaf: int,
        ko_var: str,
        footprint: list[tuple[str, int]],
        free_extent: int,
        d0_role: AxisRole,
        reads: tuple[BufferRegion, ...],
        writes: tuple[BufferRegion, ...],
    ) -> BlockNode:
        """Build a per-``ki`` gadget block."""
        tree = ir.tree
        block = self._enclosing_block(ir, matmul_leaf)
        op_cls = tree.isa(matmul_leaf).op_cls
        reduction_abstract = next(axis for axis, role in op_cls.AXIS_ROLES.items() if role == AxisRole.ACCUMULATION)
        output_axes = op_cls.OPERAND_AXES["dst"]
        k_axis = block.axis_map[reduction_abstract]
        m_axis = block.axis_map[output_axes[0]]
        free_axis = block.axis_map[output_axes[1]]
        k_dom = next(iv.dom for iv in block.iter_vars if iv.axis == k_axis)
        m_value = next(v for iv, v in zip(block.iter_vars, block.iter_values) if iv.axis == m_axis)
        m_dom = next(iv.dom for iv in block.iter_vars if iv.axis == m_axis)
        free_dom = next(iv.dom for iv in block.iter_vars if iv.axis == free_axis)
        return BlockNode(
            iter_vars=(
                IterVar(axis=k_axis, dom=k_dom, role=d0_role),
                IterVar(axis=m_axis, dom=m_dom, role=AxisRole.PARALLEL),
                IterVar(axis=free_axis, dom=free_dom, role=AxisRole.PARALLEL),
            ),
            iter_values=(Var(name=ko_var), m_value, Const(value=0)),
            reads=reads,
            writes=writes,
            alloc_buffers=(),
            axis_map={"K": k_axis, "P": m_axis, "F": free_axis},
        )

    def _splice_beside_ki(
        self,
        tree: KernelTree,
        ki_loop_nid: int,
        block: BlockNode,
        footprint: list[tuple[str, int]],
        leaf: ISANode,
        insert_after: int | None,
    ) -> int:
        """Splice a gadget block beside ``ki_loop_nid``."""
        parent = tree.parent(ki_loop_nid)
        assert parent is not None, f"ki loop {ki_loop_nid} has no parent"
        block_nid = tree.add_node(block, parent=parent)
        cursor = block_nid
        for loop_var, extent in footprint:
            cursor = tree.add_node(ForNode(loop_var=loop_var, extent=extent), parent=cursor)
        tree.add_node(leaf, parent=cursor)
        siblings = [c for c in tree.children(parent) if c != block_nid]
        anchor = ki_loop_nid if insert_after is None else insert_after
        anchor_index = siblings.index(anchor)
        insertion_index = anchor_index if insert_after is None else anchor_index + 1
        new_order = [*siblings[:insertion_index], block_nid, *siblings[insertion_index:]]
        for child in tree.children(parent):
            tree.graph.remove_edge(parent, child)
        for child in new_order:
            tree.graph.add_edge(parent, child)
        return block_nid

    def _writer_leaf(self, tree: KernelTree, tensor: str, op_name: str) -> int:
        """The single ISA leaf with NAME ``op_name`` that writes ``tensor`` (dst slot)."""
        for nid in tree.preorder():
            data = tree.data(nid)
            if isinstance(data, ISANode) and data.op_cls.NAME == op_name:
                if data.operand_bindings.get("dst") is not None and data.operand_bindings["dst"].tensor == tensor:
                    return nid
        raise TransformLegalityError(f"no {op_name} writing {tensor!r}")

    def _reader_leaf(self, tree: KernelTree, tensor: str, op_name: str) -> int:
        """The single ISA leaf with NAME ``op_name`` that reads ``tensor`` (src slot)."""
        for nid in tree.preorder():
            data = tree.data(nid)
            if isinstance(data, ISANode) and data.op_cls.NAME == op_name:
                if data.operand_bindings.get("src") is not None and data.operand_bindings["src"].tensor == tensor:
                    return nid
        raise TransformLegalityError(f"no {op_name} reading {tensor!r}")

    def _owning_matmul_leaf(self, ir: KernelIR, loop_nid: int) -> int | None:
        """The single ISA leaf under ``loop_nid`` whose op is rfactorable, or None."""
        leaves = [
            d
            for d in ir.tree.descendants(loop_nid)
            if isinstance((node := ir.tree.data(d)), ISANode) and node.op_cls.RFACTOR_RECIPE is not None
        ]
        return leaves[0] if len(leaves) == 1 else None

    def _enclosing_block(self, ir: KernelIR, nid: int) -> BlockNode:
        """Nearest enclosing BlockNode payload of ``nid``."""
        for anc in reversed(ir.tree.ancestors(nid)):
            data = ir.tree.data(anc)
            if isinstance(data, BlockNode):
                return data
        raise TransformLegalityError(f"no enclosing BlockNode for {nid}")

    def _loop_axis(self, ir: KernelIR, loop_nid: int, block: BlockNode) -> str | None:
        """The concrete axis the loop's loop_var binds, via the block's iter_values."""
        loop_var = ir.tree.loop(loop_nid).loop_var
        for iv, value in zip(block.iter_vars, block.iter_values):
            if loop_var in to_affine(value):
                return iv.axis
        return None


__all__ = ["RFactor", "RFactorOption"]
