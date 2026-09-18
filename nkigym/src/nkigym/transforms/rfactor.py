"""RFactor transforms for read-modify-write and slot-style reductions."""

from __future__ import annotations

from dataclasses import dataclass, replace

from nkigym.ir import KernelIR
from nkigym.ir.arith.expr import Add, Const, Expr, Mul, NonAffineError, Var, expr_variables, to_affine
from nkigym.ir.program_sharding import configured_program_shards
from nkigym.ir.tree import PARTITION_DIM, BlockNode, Buffer, BufferRegion, ForNode, ISANode, IterVar, KernelTree
from nkigym.ops.base import AxisRole, NKIOp, ReductionContract
from nkigym.ops.memset import NKIMemset
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.ops.tensor_reduce import NKITensorReduce
from nkigym.ops.tensor_tensor import NKITensorTensor
from nkigym.transforms.base import (
    Transform,
    TransformLegalityError,
    TransformOption,
    copy_for_rewrite,
    intersects_software_pipeline,
    software_pipeline_overlap_nodes,
)
from nkigym.transforms.helper.access_pattern import subtree_has_access_patterns
from nkigym.transforms.helper.canonical_rewrite import (
    append_root_buffers,
    finalize_rewrite,
    fresh_name,
    owning_block,
    replace_input_binding,
    single_leaf,
)
from nkigym.transforms.helper.operation_builder import NameSupply, OperationBuilder, OperationScope
from nkigym.transforms.helper.tile_region import retile_region
from nkigym.transforms.helper.tree_ops import _replace_in_parent_children
from nkigym.transforms.split import _covers_exactly, _current_tensorize_width, _factorizations, _min_tile_floor

_RMW_COMBINERS = frozenset({"add", "multiply"})
_SUPPORTED_COMBINERS = frozenset({"add", "maximum"})


def _role_of(block: BlockNode, axis: str) -> AxisRole:
    """Return the role ``block`` assigns to ``axis``, defaulting to parallel."""
    role = next((iter_var.role for iter_var in block.iter_vars if iter_var.axis == axis), AxisRole.PARALLEL)
    return role


def _fresh_axis(ir: KernelIR) -> str:
    """Return a fresh dense concrete axis name."""
    axes = {iter_var.axis for block_nid in ir.tree.blocks() for iter_var in ir.tree.block(block_nid).iter_vars}
    index = 0
    while f"d{index}" in axes:
        index += 1
    return f"d{index}"


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
    """Tile one native reduction into a serial accumulation loop."""

    def analyze(self, ir: KernelIR) -> list[tuple[int, str, tuple[int, int]]]:
        """Return legal ``(leaf, axis, factors)`` slot-factorization choices."""
        options: list[tuple[int, str, tuple[int, int]]] = []
        buffers = ir.all_buffers()
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
                if self.rfactorable(ir, leaf_nid, target_axis, factors, buffers):
                    options.append((leaf_nid, target_axis, factors))
        return options

    def rfactorable(
        self,
        ir: KernelIR,
        leaf_nid: int,
        target_axis: str,
        factors: tuple[int, int],
        buffers: dict[str, Buffer] | None = None,
    ) -> bool:
        """Return whether one unsplit tensorized reduction satisfies the slot recipe."""
        resolved_buffers = ir.all_buffers() if buffers is None else buffers
        return self._resolve_source(ir, leaf_nid, target_axis, factors, resolved_buffers) is not None

    def emit(self, ir: KernelIR, leaf_nid: int, target_axis: str, factors: tuple[int, int]) -> None:
        """Preserve the reduction while introducing one explicit tile loop."""
        match = self._resolve_source(ir, leaf_nid, target_axis, factors, ir.all_buffers())
        if match is None:
            raise AssertionError(
                f"slot RFactor match disappeared for leaf {leaf_nid}, axis {target_axis!r}, factors {factors}"
            )

        output = ir.buffer(match.output_region.tensor)
        partial, state = (
            replace(
                output,
                name=fresh_name(ir, f"{output.name}_{suffix}"),
                dtype="float32",
                storage_dtype="float32",
                list_len=1,
                versions=1,
            )
            for suffix in ("partial", "accumulator")
        )
        append_root_buffers(ir, (partial, state))
        partial_region = replace(match.output_region, tensor=partial.name)
        state_region = replace(match.output_region, tensor=state.name)
        block, leaf = ir.tree.block(match.block_nid), ir.tree.isa(leaf_nid)
        parent = ir.tree.parent(leaf_nid)
        if parent is None:
            raise AssertionError("native reduction has no enclosing scope")
        factor_axis = _fresh_axis(ir)
        index = Var(name=f"i_{factor_axis}_0")
        loop = ir.tree.add_node(ForNode(loop_var=index.name, extent=factors[0]))
        abstract_axis = next(abstract for abstract, concrete in block.axis_map.items() if concrete == target_axis)

        def tiled(lower: Expr, _width: int) -> tuple[Expr, int]:
            """Address one reduction tile without changing surrounding coordinates."""
            return Add(left=lower, right=Mul(left=index, right=Const(value=factors[1]))), factors[1]

        bindings = {
            slot: retile_region(region, leaf.op_cls.operand_axis_groups(slot), abstract_axis, tiled)
            for slot, region in leaf.operand_bindings.items()
        }
        bindings[match.contract.output_operand] = partial_region
        kwargs = dict(leaf.kwargs)
        if (offset := getattr(leaf.op_cls, "SPLIT_OFFSET_KWARGS", {}).get(abstract_axis)) is not None:
            key, slot = offset
            kwargs[key] = bindings[slot].ranges[leaf.op_cls.operand_dimension(slot, abstract_axis)][0]
        values = tuple(
            (
                Add(left=Mul(left=value, right=Const(value=factors[0])), right=index)
                if variable.axis == target_axis
                else value
            )
            for variable, value in zip(block.iter_vars, block.iter_values)
        )
        row_axis = block.axis_map[match.output_abstract_axis]
        row, row_value = next(
            (variable, value)
            for variable, value in zip(block.iter_vars, block.iter_values)
            if variable.axis == row_axis
        )
        row_scope = BlockNode(iter_vars=(row,), iter_values=(row_value,), reads=(), writes=(), axis_map={"P": row_axis})
        builder = OperationBuilder(ir.tree, None, ir.all_buffers(), NameSupply(set(ir.all_buffers())))
        initializer = builder.append(
            NKIMemset,
            {"dst": state_region},
            {"value": match.contract.combinator.identity},
            OperationScope(row_scope, ()),
        )
        drain = builder.append(
            NKITensorCopy, {"src": state_region, "dst": match.output_region}, {}, OperationScope(row_scope, ())
        )
        builder.parent = loop
        builder.append(leaf.op_cls, bindings, kwargs, OperationScope(replace(block, iter_values=values), ()))
        update_scope = replace(
            row_scope,
            iter_vars=(row, IterVar(axis=factor_axis, dom=(0, factors[0]), role=AxisRole.ACCUMULATION)),
            iter_values=(row_value, index),
            axis_map={"P": row_axis, "F": factor_axis},
        )
        builder.append(
            NKITensorTensor,
            {"data1": state_region, "data2": partial_region, "dst": state_region},
            {"op": match.contract.combinator.combiner},
            OperationScope(update_scope, ()),
        )
        _replace_in_parent_children(ir.tree, parent, [leaf_nid], [initializer, loop, drain])
        ir.tree.graph.remove_node(leaf_nid)
        finalize_rewrite(ir)

    def _resolve_source(
        self, ir: KernelIR, leaf_nid: int, target_axis: str, factors: tuple[int, int], buffers: dict[str, Buffer]
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
                output_buffer = buffers.get(output_region.tensor) if output_region is not None else None
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
                    and output_buffer.location == "sbuf"
                    and output_buffer.physical_dtype() in {"bfloat16", "float16", "float32", "tfloat32"}
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


@dataclass(frozen=True)
class RFactorOption(TransformOption):
    """Tile a native reduction serially, or privatize an existing reduction loop."""

    target_loop_nid: int
    factor_axis: int = 0
    factors: tuple[int, int] | None = None
    target_axis: str | None = None


@dataclass(frozen=True)
class _SerialReduction:
    """One associative state update and its independent output loops."""

    loop_nid: int
    update_block: int
    update_leaf: int
    state: BufferRegion
    partial: BufferRegion
    output_loops: tuple[int, ...]


def _serial_reduction(ir: KernelIR, loop_nid: int, buffers: dict[str, Buffer] | None = None) -> _SerialReduction | None:
    """Recognize one state-independent associative update over a factor loop."""
    if loop_nid not in ir.tree.graph or not isinstance(ir.tree.data(loop_nid), ForNode):
        return None
    descendants = set(ir.tree.descendants(loop_nid))
    if configured_program_shards(ir).keys() & (descendants | {loop_nid}):
        return None
    loop = ir.tree.loop(loop_nid)
    candidates = []
    for nid in ir.tree.preorder(loop_nid):
        node = ir.tree.data(nid)
        if not isinstance(node, ISANode) or node.op_cls is not NKITensorTensor:
            continue
        state = node.operand_bindings.get("dst")
        partial_operand = "data2" if node.operand_bindings.get("data1") == state else "data1"
        partial = node.operand_bindings.get(partial_operand)
        block_nid = owning_block(ir.tree, nid)
        block = ir.tree.block(block_nid)
        if (
            state is not None
            and partial is not None
            and state.tensor != partial.tensor
            and (
                node.operand_bindings == {"data1": state, "data2": partial, "dst": state}
                or node.operand_bindings == {"data1": partial, "data2": state, "dst": state}
            )
            and node.kwargs.get("op") in _RMW_COMBINERS | {"maximum"}
            and not node.access_patterns
            and "program_ownership" not in node.kwargs
            and tuple(width for _, width in state.ranges) == tuple(width for _, width in partial.ranges)
            and any(
                variable.role == AxisRole.ACCUMULATION and loop.loop_var in expr_variables(value)
                for variable, value in zip(block.iter_vars, block.iter_values)
            )
            and not any(loop.loop_var in expr_variables(value) for bounds in state.ranges for value in bounds)
            and not any(loop.loop_var in expr_variables(value) for bounds in partial.ranges for value in bounds)
        ):
            candidates.append((nid, block_nid, state, partial))
    if len(candidates) != 1:
        return None
    leaf, block, state, partial = candidates[0]
    buffers = ir.all_buffers() if buffers is None else buffers
    if (
        buffers[state.tensor].location != "sbuf"
        or len(buffers[state.tensor].shape) not in {1, 2}
        or buffers[partial.tensor].location not in {"sbuf", "psum"}
        or any(buffers[r.tensor].physical_dtype() != "float32" for r in (state, partial))
    ):
        return None
    if set(ir.dependency.touches_by_tensor[state.tensor]) & descendants != {leaf}:
        return None
    if not any(
        nid in descendants and partial.tensor in ir.dependency.info(nid).writes
        for nid in ir.dependency.direct_producers(leaf)
    ):
        return None
    ancestors = ir.tree.ancestors(leaf)
    output_loops = tuple(
        nid for nid in ancestors[ancestors.index(loop_nid) + 1 :] if isinstance(ir.tree.data(nid), ForNode)
    )
    if not _independent_output_iterations(ir, output_loops, state, buffers[state.tensor]):
        return None
    return _SerialReduction(loop_nid, block, leaf, state, partial, output_loops)


def _independent_output_iterations(ir: KernelIR, loops: tuple[int, ...], region: BufferRegion, buffer: Buffer) -> bool:
    """Require inner iterations to address disjoint accumulator tiles."""
    extents = {ir.tree.loop(nid).loop_var: ir.tree.loop(nid).extent for nid in loops}
    remaining = set(extents)
    for axis, (lower, width) in enumerate(region.ranges):
        names = remaining & expr_variables(lower)
        if not names:
            continue
        if not isinstance(width, Const):
            return False
        try:
            coefficients = to_affine(lower)
        except NonAffineError:
            return False
        unit = buffer.partition_extent() if axis == 0 else 1
        strides = sorted((abs(coefficients.get(name, 0)) * unit, name) for name in names)
        span = width.value
        for stride, name in strides:
            if stride < span:
                return False
            span += stride * (extents[name] - 1)
        remaining.difference_update(names)
    return not remaining


def _owned_operation_scope(ir: KernelIR, leaf_nid: int) -> OperationScope:
    """Capture the owning block and local loops of one existing instruction."""
    block_nid = owning_block(ir.tree, leaf_nid)
    ancestors = ir.tree.ancestors(leaf_nid)
    loops = tuple(
        ir.tree.loop(nid)
        for nid in ancestors[ancestors.index(block_nid) + 1 :]
        if isinstance(ir.tree.data(nid), ForNode)
    )
    return OperationScope(ir.tree.block(block_nid), loops)


def _privatize_serial_reduction(ir: KernelIR, match: _SerialReduction) -> None:
    """Give each factor a private partial slot while retaining the accumulation schedule."""
    loop = ir.tree.loop(match.loop_nid)
    state = ir.buffer(match.state.tensor)
    slots = replace(
        state,
        name=fresh_name(ir, f"{state.name}_rfactor"),
        shape=(state.shape[0] * loop.extent, *state.shape[1:]),
        partition_size=state.partition_extent(),
        list_len=1,
        versions=1,
    )
    append_root_buffers(ir, (slots,))
    lower, width = match.state.ranges[0]
    slot = BufferRegion(
        tensor=slots.name,
        ranges=(
            (
                Add(left=Mul(left=Var(name=loop.loop_var), right=Const(value=state.logical_tile_count())), right=lower),
                width,
            ),
            *match.state.ranges[1:],
        ),
    )
    update, block = ir.tree.isa(match.update_leaf), ir.tree.block(match.update_block)
    copy_scope = replace(
        block,
        iter_vars=tuple(
            replace(variable, role=AxisRole.PARALLEL) if loop.loop_var in expr_variables(value) else variable
            for variable, value in zip(block.iter_vars, block.iter_values)
        ),
    )
    builder = OperationBuilder(ir.tree, None, ir.all_buffers(), NameSupply(set(ir.all_buffers())))
    copy = builder.append(
        NKITensorCopy,
        {"src": match.partial, "dst": slot},
        {key: value for key, value in update.kwargs.items() if key != "op"},
        OperationScope(copy_scope, ()),
    )
    fold = builder.append(
        NKITensorTensor,
        {operand: slot if region == match.partial else region for operand, region in update.operand_bindings.items()},
        dict(update.kwargs),
        OperationScope(block, ()),
    )
    parent = ir.tree.parent(match.update_leaf)
    if parent is None:
        raise AssertionError("serial reduction update has no parent")
    _replace_in_parent_children(ir.tree, parent, [match.update_leaf], [copy, fold])
    ir.tree.graph.remove_node(match.update_leaf)
    ir.tree.graph.nodes[match.update_block]["data"] = replace(block, writes=(*block.writes, slot))
    finalize_rewrite(ir)


@dataclass(frozen=True)
class _RMWAnalysis:
    """Tree-order and buffer facts shared by RMW recipe candidates."""

    buffers: dict[str, Buffer]
    order: dict[int, int]


class RFactor(Transform[RFactorOption]):
    """Expose reduction tiles or privatize one already-materialized reduction factor."""

    def analyze(self, ir: KernelIR) -> list[RFactorOption]:
        """Enumerate every fully legal ACCUMULATION loop of an rfactorable op."""
        options: list[RFactorOption] = []
        overlap_nodes = software_pipeline_overlap_nodes(ir)
        analysis = _RMWAnalysis(buffers=ir.all_buffers(), order={nid: i for i, nid in enumerate(ir.tree.preorder())})
        for nid in ir.tree.preorder():
            if not isinstance(ir.tree.data(nid), ForNode):
                continue
            if _serial_reduction(ir, nid, analysis.buffers) is not None and not intersects_software_pipeline(
                ir, (nid,), overlap_nodes
            ):
                options.append(RFactorOption(target_loop_nid=nid))
            elif self._rfactorable(ir, nid, analysis) and not intersects_software_pipeline(
                ir, self._rmw_rewrite_nodes(ir, nid), overlap_nodes
            ):
                options.append(RFactorOption(target_loop_nid=nid, factor_axis=0))
        for leaf_nid, target_axis, factors in _SlotRFactor().analyze(ir):
            if not intersects_software_pipeline(ir, (leaf_nid,), overlap_nodes):
                options.append(
                    RFactorOption(target_loop_nid=leaf_nid, factor_axis=0, factors=factors, target_axis=target_axis)
                )
        return options

    def apply(self, ir: KernelIR, option: RFactorOption) -> KernelIR:
        """Re-check legality, deep-copy, emit the two-stage accumulation, return."""
        self._check_legality(ir, option)
        new_ir = copy_for_rewrite(ir)
        if option.factors is not None and option.target_axis is not None:
            _SlotRFactor().emit(new_ir, option.target_loop_nid, option.target_axis, option.factors)
        elif (serial := _serial_reduction(new_ir, option.target_loop_nid)) is not None:
            _privatize_serial_reduction(new_ir, serial)
        else:
            self._emit_rmw(new_ir, option)
        return new_ir

    def _rfactorable(self, ir: KernelIR, loop_nid: int, analysis: _RMWAnalysis | None = None) -> bool:
        """Return whether ``loop_nid`` supports the RMW recipe."""
        if analysis is None:
            analysis = _RMWAnalysis(
                buffers=ir.all_buffers(), order={nid: i for i, nid in enumerate(ir.tree.preorder())}
            )
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
                    if isinstance((node := ir.tree.data(nid)), ForNode) and node.loop_var in binding_vars
                ]
            if (
                self._supports_rmw_op(op_cls)
                and axis is not None
                and _role_of(block, axis) == AxisRole.ACCUMULATION
                and len(axis_loops) == 2
                and axis_loops[0] == loop_nid
                and self._init_block_is_retargetable(ir, loop_nid, leaf, analysis)
                and self._drain_block_is_removable(ir, loop_nid, leaf, analysis)
                and self._ki_loop_nid(ir, loop_nid) is not None
                and self._gadget_region_fits_output(ir, loop_nid, leaf, analysis)
            ):
                result = True
        if result:
            rewritten = self._rmw_rewrite_nodes(ir, loop_nid)
            changed = {node for root in rewritten for node in (root, *ir.tree.descendants(root))}
            result = not bool(configured_program_shards(ir).keys() & changed)
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

    def _init_block_is_retargetable(
        self, ir: KernelIR, loop_nid: int, matmul_leaf: int, analysis: _RMWAnalysis
    ) -> bool:
        """Whether one canonical identity memset initializes PSUM outside ``ko``."""
        matmul = ir.tree.data(matmul_leaf)
        assert isinstance(matmul, ISANode)
        psum_name = matmul.operand_bindings["dst"].tensor
        reducer = matmul.op_cls.REDUCE_COMBINATOR
        assert reducer is not None
        inits = [
            nid
            for nid in ir.dependency.touches_by_tensor.get(psum_name, ())
            if isinstance((node := ir.tree.data(nid)), ISANode)
            and node.op_cls.NAME == "memset"
            and node.operand_bindings["dst"].tensor == psum_name
        ]
        result = False
        if len(inits) == 1:
            init_nid = inits[0]
            init_block = self._enclosing_block_nid(ir.tree, init_nid)
            init = ir.tree.isa(init_nid)
            result = (
                single_leaf(ir.tree, init_block) == init_nid
                and loop_nid not in ir.tree.ancestors(init_nid)
                and analysis.order[init_nid] < analysis.order[matmul_leaf]
                and init.kwargs.get("value") == float(reducer.identity)
            )
        return result

    def _drain_block_is_removable(self, ir: KernelIR, loop_nid: int, matmul_leaf: int, analysis: _RMWAnalysis) -> bool:
        """Whether the sole consumer is an outside-``ko`` identity drain."""
        matmul = ir.tree.data(matmul_leaf)
        assert isinstance(matmul, ISANode)
        psum_name = matmul.operand_bindings["dst"].tensor
        drains = [
            nid
            for nid in ir.dependency.touches_by_tensor.get(psum_name, ())
            if (
                isinstance((node := ir.tree.data(nid)), ISANode)
                and node.op_cls.NAME == "tensor_copy"
                and node.operand_bindings["src"].tensor == psum_name
            )
        ]
        result = False
        if len(drains) == 1:
            drain_nid = drains[0]
            drain_block = self._enclosing_block_nid(ir.tree, drain_nid)
            leaves = [nid for nid in ir.tree.preorder(drain_block) if isinstance(ir.tree.data(nid), ISANode)]
            drain = ir.tree.isa(drain_nid)
            out_name = drain.operand_bindings["dst"].tensor
            result = (
                leaves == drains
                and ir.dependency.direct_consumers(matmul_leaf) == [drain_nid]
                and loop_nid not in ir.tree.ancestors(drain_nid)
                and analysis.order[matmul_leaf] < analysis.order[drain_nid]
                and drain.operand_bindings["src"].ranges == drain.operand_bindings["dst"].ranges
                and analysis.buffers[out_name].location == "sbuf"
            )
        return result

    def _gadget_region_fits_output(self, ir: KernelIR, loop_nid: int, matmul_leaf: int, analysis: _RMWAnalysis) -> bool:
        """Return whether the generated region fits PSUM and SBUF."""
        ki_nid = self._ki_loop_nid(ir, loop_nid)
        if ki_nid is None:
            return False
        matmul = ir.tree.data(matmul_leaf)
        assert isinstance(matmul, ISANode)
        psum_name = matmul.operand_bindings["dst"].tensor
        out_name = self._drain_out_tensor(ir, psum_name)
        dst_region = matmul.operand_bindings["dst"]
        free_footprint = self._free_footprint(ir, matmul_leaf)
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
                for buf in (analysis.buffers[psum_name], analysis.buffers[out_name])
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
            if intersects_software_pipeline(ir, (option.target_loop_nid,)):
                raise TransformLegalityError("RFactor cannot rewrite an active software-pipeline scope")
            self._check_slot_legality(ir, option)
        elif _serial_reduction(ir, option.target_loop_nid) is not None:
            if option.factor_axis != 0 or intersects_software_pipeline(ir, (option.target_loop_nid,)):
                raise TransformLegalityError("illegal serial-reduction privatization")
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
        if intersects_software_pipeline(ir, self._rmw_rewrite_nodes(ir, nid)):
            raise TransformLegalityError("RFactor cannot rewrite an active software-pipeline scope")

    def _rmw_rewrite_nodes(self, ir: KernelIR, loop_nid: int) -> tuple[int, ...]:
        """Return the existing tree sites changed by the RMW recipe."""
        matmul_leaf = self._owning_matmul_leaf(ir, loop_nid)
        if matmul_leaf is None:
            raise TransformLegalityError(f"RFactor target loop {loop_nid} has no RMW operation")
        psum_name = ir.tree.isa(matmul_leaf).operand_bindings["dst"].tensor
        init_block = self._enclosing_block_nid(ir.tree, self._writer_leaf(ir.tree, psum_name, "memset"))
        drain_block = self._enclosing_block_nid(ir.tree, self._reader_leaf(ir.tree, psum_name, "tensor_copy"))
        return loop_nid, init_block, drain_block

    def _emit_rmw(self, ir: KernelIR, option: RFactorOption) -> None:
        """Regroup a PSUM reduction and fold each partial directly into SBUF."""
        tree = ir.tree
        ko = tree.loop(option.target_loop_nid)
        matmul_leaf = self._owning_matmul_leaf(ir, option.target_loop_nid)
        assert matmul_leaf is not None
        matmul = tree.isa(matmul_leaf)
        reducer = matmul.op_cls.REDUCE_COMBINATOR
        assert reducer is not None
        psum = matmul.operand_bindings["dst"]
        drain_leaf = self._reader_leaf(tree, psum.tensor, "tensor_copy")
        output = ir.buffer(tree.isa(drain_leaf).operand_bindings["dst"].tensor)
        state = replace(
            output,
            name=fresh_name(ir, f"{output.name}_accumulator"),
            dtype="float32",
            storage_dtype="float32",
            list_len=1,
            versions=1,
        )
        append_root_buffers(ir, (state,))
        builder = OperationBuilder(tree, None, ir.all_buffers(), NameSupply(set(ir.all_buffers())))
        init_leaf = self._writer_leaf(tree, psum.tensor, "memset")
        initializer = tree.isa(init_leaf)
        init_block = owning_block(tree, init_leaf)
        state_init = builder.append(
            NKIMemset,
            {"dst": replace(initializer.operand_bindings["dst"], tensor=state.name)},
            dict(initializer.kwargs),
            _owned_operation_scope(ir, init_leaf),
        )
        init_parent = tree.parent(init_block)
        if init_parent is None:
            raise AssertionError("RMW initializer has no parent")
        _replace_in_parent_children(tree, init_parent, [init_block], [state_init, init_block])
        ki = self._ki_loop_nid(ir, option.target_loop_nid)
        if ki is None:
            raise TransformLegalityError("RFactor target has no inner accumulation loop")
        footprint = tuple(
            ForNode(loop_var=name, extent=extent) for name, extent in self._footprint(ir, ki, matmul_leaf)
        )
        scope = self._gadget_block(ir, matmul_leaf, ko.loop_var, reads=(), writes=())
        reset = builder.append(NKIMemset, {"dst": psum}, {"value": reducer.identity}, OperationScope(scope, footprint))
        state_region = replace(psum, tensor=state.name)
        update_scope = replace(
            scope,
            iter_vars=tuple(
                replace(variable, role=AxisRole.ACCUMULATION) if variable.axis == scope.axis_map["K"] else variable
                for variable in scope.iter_vars
            ),
        )
        update = builder.append(
            NKITensorTensor,
            {"data1": psum, "data2": state_region, "dst": state_region},
            {"op": reducer.combiner},
            OperationScope(update_scope, footprint),
        )
        parent = tree.parent(ki)
        if parent is None:
            raise AssertionError("inner reduction loop has no parent")
        _replace_in_parent_children(tree, parent, [ki], [reset, ki, update])
        replace_input_binding(ir, drain_leaf, "src", state.name)
        finalize_rewrite(ir)

    def _enclosing_block_nid(self, tree: KernelTree, nid: int) -> int:
        """Nearest enclosing BlockNode nid of ``nid`` (deepest ancestor block)."""
        for anc in reversed(tree.ancestors(nid)):
            if isinstance(tree.data(anc), BlockNode):
                return anc
        raise TransformLegalityError(f"no enclosing BlockNode for {nid}")

    def _ki_loop_nid(self, ir: KernelIR, ko_loop_nid: int) -> int | None:
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
        return k_loops[-1] if k_loops else None

    def _axis_binding_loopvars(self, block: BlockNode, axis: str) -> set[str]:
        """Loop vars appearing in the iter_value of ``axis`` (the loops that bind it)."""
        value = next(v for iv, v in zip(block.iter_vars, block.iter_values) if iv.axis == axis)
        return {n for n in to_affine(value) if n is not None}

    def _footprint(self, ir: KernelIR, ki_loop_nid: int, matmul_leaf: int) -> list[tuple[str, int]]:
        """Retain every output-axis loop strictly between ``ki`` and the matmul."""
        tree = ir.tree
        block = self._enclosing_block(ir, matmul_leaf)
        output_vars = {
            name
            for abstract in tree.isa(matmul_leaf).op_cls.OPERAND_AXES["dst"]
            for name in self._axis_binding_loopvars(block, block.axis_map[abstract])
        }
        between = [
            a
            for a in tree.ancestors(matmul_leaf)
            if isinstance(tree.data(a), ForNode) and ki_loop_nid in tree.ancestors(a)
        ]
        return [(tree.loop(a).loop_var, tree.loop(a).extent) for a in between if tree.loop(a).loop_var in output_vars]

    def _free_footprint(self, ir: KernelIR, matmul_leaf: int) -> tuple[Expr, int] | None:
        """Preserve the matmul instruction's free-axis offset and width."""
        lower, width = ir.tree.isa(matmul_leaf).operand_bindings["dst"].ranges[1]
        return (lower, width.value) if isinstance(width, Const) and width.value > 0 else None

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

    def _drain_out_tensor(self, ir: KernelIR, psum_name: str) -> str:
        """Tensor the drain ``tensor_copy`` writes (reads ``psum_name``, writes SBUF out)."""
        for nid in ir.dependency.touches_by_tensor.get(psum_name, ()):
            data = ir.tree.data(nid)
            if isinstance(data, ISANode) and data.op_cls.NAME == "tensor_copy":
                if data.operand_bindings["src"].tensor == psum_name:
                    return data.operand_bindings["dst"].tensor
        raise TransformLegalityError(f"no drain tensor_copy reading {psum_name!r}")

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
        free_value = next(
            value for variable, value in zip(block.iter_vars, block.iter_values) if variable.axis == free_axis
        )
        return BlockNode(
            iter_vars=(
                IterVar(axis=k_axis, dom=k_dom, role=AxisRole.PARALLEL),
                IterVar(axis=m_axis, dom=m_dom, role=AxisRole.PARALLEL),
                IterVar(axis=free_axis, dom=free_dom, role=AxisRole.PARALLEL),
            ),
            iter_values=(Var(name=ko_var), m_value, free_value),
            reads=reads,
            writes=writes,
            alloc_buffers=(),
            axis_map={"K": k_axis, "P": m_axis, "F": free_axis},
        )

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
