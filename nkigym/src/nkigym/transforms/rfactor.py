"""RFactor transforms for read-modify-write and slot-style reductions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import gcd, prod

from nkigym.ir import KernelIR
from nkigym.ir.arith.analyzer import Analyzer
from nkigym.ir.arith.expr import Add, Const, Expr, Mod, Mul, NonAffineError, Var, expr_variables, to_affine
from nkigym.ir.program_sharding import configured_program_shards
from nkigym.ir.tree import BlockNode, Buffer, BufferRegion, ForNode, ISANode, IterVar
from nkigym.ops.base import (
    AxisRole,
    BilinearReductionContract,
    InitializerContract,
    ReduceCombinator,
    ReductionContract,
)
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
    replace_buffer,
    replace_input_binding,
    single_leaf,
)
from nkigym.transforms.helper.operation_builder import NameSupply, OperationBuilder, OperationScope
from nkigym.transforms.helper.tile_region import retile_region
from nkigym.transforms.helper.tree_ops import _replace_in_parent_children
from nkigym.transforms.program_shard import _drain_operand
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
        leaf = ir.tree.isa(leaf_nid)
        direct = output.physical_dtype() == "float32" and all(
            region.tensor != output.name
            for slot, region in leaf.operand_bindings.items()
            if slot != match.contract.output_operand
        )
        if direct:
            state = output
        append_root_buffers(ir, (partial,) if direct else (partial, state))
        partial_region = replace(match.output_region, tensor=partial.name)
        state_region = replace(match.output_region, tensor=state.name)
        block = ir.tree.block(match.block_nid)
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
        drains = (
            []
            if direct
            else [
                builder.append(
                    NKITensorCopy, {"src": state_region, "dst": match.output_region}, {}, OperationScope(row_scope, ())
                )
            ]
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
        _replace_in_parent_children(ir.tree, parent, [leaf_nid], [initializer, loop, *drains])
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
        or buffers[partial.tensor].location != "sbuf"
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
    """Give each materialized SBUF contribution a private slot without changing the fold schedule."""
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
class _MatrixFactor:
    """One closed matrix reduction with a complete materialized factor domain."""

    product: int
    block: int
    initializer: int
    drain: int
    drain_operand: str
    axis: str
    loops: tuple[int, ...]
    buffer: Buffer
    owner: int
    combinator: ReduceCombinator


def _dense_reduction_index(ir: KernelIR, loops: tuple[int, ...]) -> Expr:
    """Return the mixed-radix iteration index of the given ordered loops."""
    index: Expr = Const(value=0)
    for nid in loops:
        loop = ir.tree.loop(nid)
        index = Add(left=Mul(left=index, right=Const(value=loop.extent)), right=Var(name=loop.loop_var))
    return Analyzer().simplify(index)


def _matrix_reduction_domain(ir: KernelIR, leaf: int) -> tuple[str, tuple[int, ...]] | None:
    """Require one complete, densely enumerated materialized contraction."""
    node = ir.tree.isa(leaf)
    contract = node.op_cls.algebraic_contract(node.kwargs)
    if (
        not isinstance(contract, BilinearReductionContract)
        or contract.output_operand != "dst"
        or node.op_cls.RFACTOR_RECIPE != "rmw"
        or node.op_cls.OUTPUT_LOCATION != "psum"
        or node.op_cls.rmw_operands(node.kwargs) != frozenset({"dst"})
        or sum(role is AxisRole.ACCUMULATION for role in node.op_cls.AXIS_ROLES.values()) != 1
        or contract.combinator.combiner not in _RMW_COMBINERS
    ):
        return None
    block = ir.tree.block(owning_block(ir.tree, leaf))
    axis = block.axis_map.get(contract.reduction_axis)
    if axis is None or sum(value == axis for value in block.axis_map.values()) != 1:
        return None
    variable, value = next((iv, value) for iv, value in zip(block.iter_vars, block.iter_values) if iv.axis == axis)
    ancestors = tuple(nid for nid in ir.tree.ancestors(leaf) if isinstance(ir.tree.data(nid), ForNode))
    names = [ir.tree.loop(nid).loop_var for nid in ancestors]
    loops = tuple(nid for nid in ancestors if ir.tree.loop(nid).loop_var in expr_variables(value))
    tile = _current_tensorize_width(node, block, axis)
    if (
        variable.role is not AxisRole.ACCUMULATION
        or variable.dom[0] != 0
        or not loops
        or len(names) != len(set(names))
        or tile is None
        or tile < 1
        or variable.dom[1] != tile * prod(ir.tree.loop(nid).extent for nid in loops)
        or not Analyzer().can_prove_equal(value, _dense_reduction_index(ir, loops))
    ):
        return None
    return axis, loops


def _complete_psum_view(ir: KernelIR, leaf: int, region: BufferRegion, buffer: Buffer) -> bool:
    """Prove one full-partition, unpadded logical view fits its current buffer."""
    if len(region.ranges) != 2:
        return False
    free_width = region.ranges[1][1]
    if (
        region.ranges[0][1] != Const(value=buffer.partition_extent())
        or not isinstance(free_width, Const)
        or free_width.value < 1
    ):
        return False
    analyzer = Analyzer()
    names = []
    for nid in ir.tree.ancestors(leaf):
        node = ir.tree.data(nid)
        if isinstance(node, ForNode):
            names.append(node.loop_var)
            analyzer.bind(node.loop_var, 0, node.extent)
    if len(names) != len(set(names)):
        return False
    for axis, (lower, _width) in enumerate(region.ranges):
        lo, hi = analyzer.const_int_bound(lower)
        limit = buffer.logical_tile_count() if axis == 0 else buffer.shape[1]
        span = 1 if axis == 0 else free_width.value
        if lo is None or hi is None or lo < 0 or hi + span > limit:
            return False
    return True


def _matrix_factor_match(
    ir: KernelIR, leaf: int, buffers: dict[str, Buffer], order: dict[int, int], overlap: frozenset[int]
) -> _MatrixFactor | None:
    """Require the exact reset/product/drain dataflow before privatization."""
    domain = _matrix_reduction_domain(ir, leaf)
    if domain is None:
        return None
    axis, loops = domain
    node = ir.tree.isa(leaf)
    contract = node.op_cls.algebraic_contract(node.kwargs)
    assert isinstance(contract, BilinearReductionContract)
    output = node.operand_bindings["dst"]
    buffer = buffers[output.tensor]
    touches = tuple(ir.dependency.touches_by_tensor[buffer.name])
    if (
        buffer.location != "psum"
        or buffer.physical_dtype() != "float32"
        or len(buffer.shape) != 2
        or buffer.list_len != 1
        or buffer.versions != 1
        or buffer.name in ir.param_buffers
        or buffer.name in ir.return_names
        or len(touches) != 3
        or expr_variables(output.ranges[0][0]) & {ir.tree.loop(nid).loop_var for nid in loops}
        or expr_variables(output.ranges[1][0]) & {ir.tree.loop(nid).loop_var for nid in loops}
    ):
        return None
    initializers, drains = [], []
    for nid in touches:
        operation = ir.tree.isa(nid)
        relation = operation.op_cls.algebraic_contract(operation.kwargs)
        if isinstance(relation, InitializerContract) and relation.output_operand == "dst":
            if relation.value == contract.combinator.identity:
                initializers.append(nid)
        operand = _drain_operand(operation)
        if operand is not None and operation.operand_bindings[operand].tensor == buffer.name:
            drains.append((nid, operand))
        for slot, region in operation.operand_bindings.items():
            if region.tensor == buffer.name and (
                slot in operation.access_patterns or not _complete_psum_view(ir, nid, region, buffer)
            ):
                return None
    if len(initializers) != 1 or len(drains) != 1:
        return None
    initializer, (drain, operand) = initializers[0], drains[0]
    drain_owner = owning_block(ir.tree, drain)
    drain_ancestors = ir.tree.ancestors(drain)
    drain_loops = tuple(
        nid
        for nid in drain_ancestors[drain_ancestors.index(drain_owner) + 1 :]
        if isinstance(ir.tree.data(nid), ForNode)
    )
    owners = [
        nid for nid in ir.tree.blocks() if any(item.name == buffer.name for item in ir.tree.block(nid).alloc_buffers)
    ]
    if (
        len({initializer, leaf, drain}) != 3
        or not order[initializer] < order[loops[0]] < order[leaf] < order[drain]
        or order[drain_owner] <= order[leaf]
        or loops[0] in ir.tree.ancestors(drain)
        or len(owners) != 1
        or any(owners[0] not in ir.tree.ancestors(nid) for nid in touches)
        or buffers[ir.tree.isa(drain).operand_bindings["dst"].tensor].location != "sbuf"
        or ir.tree.isa(drain).access_patterns
        or not _independent_output_iterations(ir, drain_loops, ir.tree.isa(drain).operand_bindings[operand], buffer)
        or any(
            ir.tree.block(nid).annotations
            for touch in touches
            for nid in ir.tree.ancestors(touch)
            if isinstance(ir.tree.data(nid), BlockNode) and nid != ir.tree.root
        )
        or configured_program_shards(ir).keys() & {loops[0], *ir.tree.descendants(loops[0])}
        or intersects_software_pipeline(ir, (loops[0], initializer, drain), overlap)
    ):
        return None
    return _MatrixFactor(
        leaf,
        owning_block(ir.tree, leaf),
        initializer,
        drain,
        operand,
        axis,
        loops,
        buffer,
        owners[0],
        contract.combinator,
    )


def _private_factor_bank_fit(ir: KernelIR, match: _MatrixFactor, loop: int) -> bool:
    """Prove every private matrix write remains inside one512-element PSUM bank."""
    region = ir.tree.isa(match.product).operand_bindings["dst"]
    width = region.ranges[1][1]
    if not isinstance(width, Const):
        return False
    slot = Add(
        left=Mul(left=Var(name=ir.tree.loop(loop).loop_var), right=Const(value=match.buffer.logical_tile_count())),
        right=region.ranges[0][0],
    )
    offset = Add(
        left=Mul(left=slot, right=Const(value=match.buffer.per_tile_physical_shape()[2])), right=region.ranges[1][0]
    )
    analyzer = Analyzer()
    for ancestor in ir.tree.ancestors(match.product):
        node = ir.tree.data(ancestor)
        if isinstance(node, ForNode):
            analyzer.bind(node.loop_var, 0, node.extent)
    remainder = analyzer.simplify(Mod(left=offset, right=Const(value=512)))
    _lower, upper = analyzer.const_int_bound(remainder)
    if upper is not None and upper + width.value <= 512:
        return True
    try:
        terms = to_affine(offset)
    except NonAffineError:
        return False
    unit = gcd(512, *(abs(coefficient) for name, coefficient in terms.items() if name is not None))
    return 512 - unit + terms.get(None, 0) % unit + width.value <= 512


def _matrix_factor_options(ir: KernelIR, overlap: frozenset[int]) -> dict[int, _MatrixFactor]:
    """Index unambiguous matrix factors in one traversal of reduction leaves."""
    scope_leaves: dict[int, list[int]] = {}
    order = {nid: index for index, nid in enumerate(ir.tree.preorder())}
    buffers = ir.all_buffers()
    for nid in order:
        node = ir.tree.data(nid)
        if isinstance(node, ISANode) and node.op_cls.RFACTOR_RECIPE is not None:
            for ancestor in ir.tree.ancestors(nid):
                if isinstance(ir.tree.data(ancestor), ForNode):
                    scope_leaves.setdefault(ancestor, []).append(nid)
    matches: dict[int, _MatrixFactor | None] = {}
    result = {}
    for loop, leaves in scope_leaves.items():
        if len(leaves) != 1:
            continue
        leaf = leaves[0]
        if leaf not in matches:
            matches[leaf] = _matrix_factor_match(ir, leaf, buffers, order, overlap)
        match = matches[leaf]
        if match is not None and loop in match.loops and _private_factor_bank_fit(ir, match, loop):
            result[loop] = match
    return result


def _factor_region(region: BufferRegion, index: Expr, tiles: int) -> BufferRegion:
    """Address the same output view in one factor's private partition tiles."""
    lower, width = region.ranges[0]
    analyzer = Analyzer()
    leading = analyzer.simplify(Add(left=Mul(left=index, right=Const(value=tiles)), right=lower))
    return replace(
        region, ranges=((leading, width), *((analyzer.simplify(lower), span) for lower, span in region.ranges[1:]))
    )


def _factor_scope(block: BlockNode, axis: str, index: Expr, extent: int, role: AxisRole) -> BlockNode:
    """Add one factor coordinate without changing the surrounding output scope."""
    key = "RF"
    while key in block.axis_map:
        key += "_"
    return replace(
        block,
        iter_vars=(*block.iter_vars, IterVar(axis=axis, dom=(0, extent), role=role)),
        iter_values=(*block.iter_values, index),
        axis_map={**block.axis_map, key: axis},
        alloc_buffers=(),
    )


def _factor_product(ir: KernelIR, match: _MatrixFactor, loop: int) -> str:
    """Separate the selected parallel coordinate from the residual contraction."""
    block = ir.tree.block(match.block)
    residual_axis = _fresh_axis(ir)
    extent = ir.tree.loop(loop).extent
    residual = _dense_reduction_index(ir, tuple(nid for nid in match.loops if nid != loop))
    variables, values = [], []
    for variable, value in zip(block.iter_vars, block.iter_values, strict=True):
        variables.append(
            replace(variable, axis=residual_axis, dom=(0, variable.dom[1] // extent))
            if variable.axis == match.axis
            else variable
        )
        values.append(residual if variable.axis == match.axis else value)
    block = replace(
        block,
        iter_vars=tuple(variables),
        iter_values=tuple(values),
        axis_map={key: residual_axis if value == match.axis else value for key, value in block.axis_map.items()},
    )
    ir.tree.graph.nodes[match.block]["data"] = block
    factor_axis = _fresh_axis(ir)
    index = Var(name=ir.tree.loop(loop).loop_var)
    updated = _factor_scope(block, factor_axis, index, extent, AxisRole.PARALLEL)
    updated = replace(updated, alloc_buffers=block.alloc_buffers)

    def rewrite(region: BufferRegion) -> BufferRegion:
        """Redirect only the selected accumulator's complete views."""
        return (
            _factor_region(region, index, match.buffer.logical_tile_count())
            if region.tensor == match.buffer.name
            else region
        )

    node = ir.tree.isa(match.product)
    ir.tree.graph.nodes[match.product]["data"] = replace(
        node, operand_bindings={slot: rewrite(region) for slot, region in node.operand_bindings.items()}
    )
    ir.tree.graph.nodes[match.block]["data"] = replace(
        updated, reads=tuple(map(rewrite, updated.reads)), writes=tuple(map(rewrite, updated.writes))
    )
    return factor_axis


def _factor_initializer(ir: KernelIR, match: _MatrixFactor, factor_axis: str, extent: int) -> None:
    """Retarget the existing reset to every private factor at the same site."""
    index = Var(name=f"i_{factor_axis}_0")
    node = ir.tree.isa(match.initializer)
    owner = owning_block(ir.tree, match.initializer)
    block = ir.tree.block(owner)
    region = _factor_region(node.operand_bindings["dst"], index, match.buffer.logical_tile_count())
    ir.tree.graph.nodes[match.initializer]["data"] = replace(
        node, operand_bindings={**node.operand_bindings, "dst": region}
    )
    updated = _factor_scope(block, factor_axis, index, extent, AxisRole.PARALLEL)
    ir.tree.graph.nodes[owner]["data"] = replace(updated, writes=(region,), alloc_buffers=block.alloc_buffers)
    parent = ir.tree.parent(match.initializer)
    if parent is None:
        raise AssertionError("matrix initializer has no execution parent")
    factor_loop = ir.tree.add_node(ForNode(loop_var=index.name, extent=extent))
    _replace_in_parent_children(ir.tree, parent, [match.initializer], [factor_loop])
    ir.tree.graph.add_edge(factor_loop, match.initializer)


def _fold_matrix_factors(ir: KernelIR, match: _MatrixFactor, factor_axis: str, extent: int, state: Buffer) -> None:
    """Fold private PSUM values directly before the original observable drain."""
    drain = ir.tree.isa(match.drain)
    source = drain.operand_bindings[match.drain_operand]
    state_region = replace(source, tensor=state.name)
    owner = owning_block(ir.tree, match.drain)
    original_scope = _owned_operation_scope(ir, match.drain)
    scope = replace(original_scope.block, reads=(), writes=(), alloc_buffers=())
    builder = OperationBuilder(ir.tree, None, ir.all_buffers(), NameSupply(set(ir.all_buffers())))
    reset = builder.append(
        NKIMemset,
        {"dst": state_region},
        {"value": match.combinator.identity},
        OperationScope(scope, original_scope.loops),
    )
    index = Var(name=f"i_{factor_axis}_0")
    factor_loop = ir.tree.add_node(ForNode(loop_var=index.name, extent=extent))
    builder.parent = factor_loop
    fold_scope = _factor_scope(scope, factor_axis, index, extent, AxisRole.ACCUMULATION)
    builder.append(
        NKITensorTensor,
        {
            "data1": _factor_region(source, index, match.buffer.logical_tile_count()),
            "data2": state_region,
            "dst": state_region,
        },
        {"op": match.combinator.combiner},
        OperationScope(fold_scope, original_scope.loops),
    )
    parent = ir.tree.parent(owner)
    if parent is None:
        raise AssertionError("matrix drain has no execution parent")
    replace_input_binding(ir, match.drain, match.drain_operand, state.name)
    _replace_in_parent_children(ir.tree, parent, [owner], [reset, factor_loop, owner])


def _emit_matrix_factor(ir: KernelIR, match: _MatrixFactor, loop: int) -> None:
    """Privatize one existing contraction factor without changing input order."""
    extent = ir.tree.loop(loop).extent
    replace_buffer(
        ir,
        replace(
            match.buffer,
            shape=(match.buffer.shape[0] * extent, match.buffer.shape[1]),
            partition_size=match.buffer.partition_extent(),
        ),
    )
    state = replace(
        match.buffer,
        name=fresh_name(ir, f"{match.buffer.name}_accumulator"),
        dtype="float32",
        storage_dtype="float32",
        location="sbuf",
    )
    owner = ir.tree.block(match.owner)
    ir.tree.graph.nodes[match.owner]["data"] = replace(owner, alloc_buffers=(*owner.alloc_buffers, state))
    factor_axis = _factor_product(ir, match, loop)
    _factor_initializer(ir, match, factor_axis, extent)
    _fold_matrix_factors(ir, match, factor_axis, extent, state)
    finalize_rewrite(ir)


class RFactor(Transform[RFactorOption]):
    """Expose native reduction tiles or privatize one materialized factor."""

    def analyze(self, ir: KernelIR) -> list[RFactorOption]:
        """Offer one independent factor decision for each proven reduction."""
        options: list[RFactorOption] = []
        overlap = software_pipeline_overlap_nodes(ir)
        buffers = ir.all_buffers()
        matrices = _matrix_factor_options(ir, overlap)
        for nid in ir.tree.preorder():
            if not isinstance(ir.tree.data(nid), ForNode):
                continue
            serial = _serial_reduction(ir, nid, buffers)
            if serial is not None and not intersects_software_pipeline(ir, (nid,), overlap):
                options.append(RFactorOption(target_loop_nid=nid))
            elif nid in matrices:
                options.append(RFactorOption(target_loop_nid=nid))
        for leaf, axis, factors in _SlotRFactor().analyze(ir):
            if not intersects_software_pipeline(ir, (leaf,), overlap):
                options.append(RFactorOption(target_loop_nid=leaf, factors=factors, target_axis=axis))
        return options

    def apply(self, ir: KernelIR, option: RFactorOption) -> KernelIR:
        """Recheck, copy, and factor exactly the selected reduction dimension."""
        if option not in self.analyze(ir):
            raise TransformLegalityError(f"illegal RFactor option: {option}")
        result = copy_for_rewrite(ir)
        if option.factors is not None and option.target_axis is not None:
            _SlotRFactor().emit(result, option.target_loop_nid, option.target_axis, option.factors)
        elif (serial := _serial_reduction(result, option.target_loop_nid)) is not None:
            _privatize_serial_reduction(result, serial)
        else:
            match = _matrix_factor_options(result, software_pipeline_overlap_nodes(result)).get(option.target_loop_nid)
            if match is None:
                raise AssertionError(f"matrix RFactor option disappeared after copying: {option}")
            _emit_matrix_factor(result, match, option.target_loop_nid)
        return result


__all__ = ["RFactor", "RFactorOption"]
