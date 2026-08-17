"""Deterministic schedule rules over legal atomic transforms."""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from nkigym.environment import Action
from nkigym.ir import KernelIR
from nkigym.ir.tree import BlockNode, ForNode, ISANode
from nkigym.transforms import (
    BatchPermutation,
    BatchPermutationOption,
    BufferCompaction,
    BufferCompactionOption,
    BufferLayout,
    BufferLayoutOption,
    BufferPlacement,
    BufferPlacementOption,
    CodeMotion,
    CodeMotionOption,
    CommonSubexpressionElimination,
    CommonSubexpressionEliminationOption,
    CopyPropagation,
    CopyPropagationOption,
    DecomposeBroadcastSubtract,
    DecomposeBroadcastSubtractOption,
    EliminateIdentityInitializer,
    EliminateIdentityInitializerOption,
    Fuse,
    FuseOption,
    FusePointwise,
    FusePointwiseOption,
    OnlineFusion,
    OnlineFusionOption,
    Reorder,
    ReorderOption,
    RFactor,
    RFactorOption,
    SoftwarePipeline,
    SoftwarePipelineOption,
    Split,
    SplitOption,
    Transform,
    TransformOption,
    TransposeThroughLoad,
    TransposeThroughLoadOption,
    TransposeThroughTensorCopy,
)

_LOOP_DIMENSION = re.compile(r"^i_(d[0-9]+)_")


@dataclass(frozen=True)
class ScheduleStep:
    """One selected atomic transform and its deterministic rationale."""

    action: Action
    rationale: str


@dataclass(frozen=True)
class SchedulePlan:
    """One completely lowered heuristic schedule."""

    state: KernelIR
    steps: tuple[ScheduleStep, ...]
    family: str
    strategy: str


class _PlanBuilder:
    """Apply uniquely selected legal options while retaining an action trace."""

    def __init__(self, state: KernelIR) -> None:
        """Initialize a plan from one canonical state."""
        self.state = state
        self.steps: list[ScheduleStep] = []

    def apply(
        self, transform: Transform[Any], predicate: Callable[[TransformOption], bool], rationale: str
    ) -> TransformOption:
        """Apply the unique legal option accepted by ``predicate``."""
        matches = [option for option in transform.analyze(self.state) if predicate(option)]
        if len(matches) != 1:
            raise RuntimeError(
                f"{type(transform).__name__}: expected one option for {rationale}, found {len(matches)}: {matches}"
            )
        option = matches[0]
        self.state = transform.apply(self.state, option)
        self.steps.append(ScheduleStep(action=(transform, option), rationale=rationale))
        return option

    def apply_first(self, transform: Transform[Any], rationale: str) -> TransformOption:
        """Apply the sole legal option exposed by ``transform``."""
        option = self.apply(transform, lambda _option: True, rationale)
        return option


def _maybe_direct_leaf(ir: KernelIR, block_nid: int) -> int | None:
    """Return a block's direct ISA leaf when it owns exactly one."""
    leaves: list[int] = []
    for nid in ir.tree.preorder(block_nid):
        if not isinstance(ir.tree.data(nid), ISANode):
            continue
        owner = next(
            parent for parent in reversed(ir.tree.ancestors(nid)) if isinstance(ir.tree.data(parent), BlockNode)
        )
        if owner == block_nid:
            leaves.append(nid)
    leaf = leaves[0] if len(leaves) == 1 else None
    return leaf


def _direct_leaf(ir: KernelIR, block_nid: int) -> int:
    """Return the single ISA leaf directly owned by ``block_nid``."""
    leaf = _maybe_direct_leaf(ir, block_nid)
    if leaf is None:
        raise RuntimeError(f"block {block_nid} does not own exactly one direct ISA leaf")
    return leaf


def _operation(ir: KernelIR, block_nid: int) -> ISANode:
    """Return the ISA operation directly owned by ``block_nid``."""
    operation = ir.tree.isa(_direct_leaf(ir, block_nid))
    return operation


def _operation_name(ir: KernelIR, block_nid: int) -> str:
    """Return one block's ISA operation name."""
    name = _operation(ir, block_nid).op_cls.NAME
    return name


def _operands(ir: KernelIR, block_nid: int) -> dict[str, str]:
    """Return one block's operand slots mapped to tensor names."""
    bindings = {slot: region.tensor for slot, region in _operation(ir, block_nid).operand_bindings.items()}
    return bindings


def _matching_blocks(ir: KernelIR, operation_name: str, required_operands: dict[str, str]) -> list[int]:
    """Return blocks matching an operation name and operand subset."""
    blocks: list[int] = []
    for block_nid in ir.tree.blocks():
        leaf = _maybe_direct_leaf(ir, block_nid)
        if leaf is None:
            continue
        operation = ir.tree.isa(leaf)
        operands = {slot: region.tensor for slot, region in operation.operand_bindings.items()}
        if operation.op_cls.NAME == operation_name and all(
            operands.get(slot) == tensor for slot, tensor in required_operands.items()
        ):
            blocks.append(block_nid)
    return blocks


def _unique_block(ir: KernelIR, operation_name: str, required_operands: dict[str, str]) -> int:
    """Return the unique block matching one semantic operation."""
    blocks = _matching_blocks(ir, operation_name, required_operands)
    if len(blocks) != 1:
        raise RuntimeError(
            f"expected one {operation_name} block with operands {required_operands}, found {len(blocks)}: {blocks}"
        )
    return blocks[0]


def _loop_dimension(ir: KernelIR, loop_nid: int) -> str:
    """Return the concrete dimension bound by one generated loop."""
    loop_var = ir.tree.loop(loop_nid).loop_var
    match = _LOOP_DIMENSION.match(loop_var)
    if match is None:
        raise RuntimeError(f"cannot identify dimension from loop variable {loop_var!r}")
    return match.group(1)


def _own_loops(ir: KernelIR, block_nid: int) -> list[int]:
    """Return loops owned by one block in outer-to-inner order."""
    leaf = _direct_leaf(ir, block_nid)
    descendants = ir.tree.descendants(block_nid)
    loops = [
        ancestor
        for ancestor in ir.tree.ancestors(leaf)
        if ancestor in descendants and isinstance(ir.tree.data(ancestor), ForNode)
    ]
    return loops


def _enclosing_loops(ir: KernelIR, block_nid: int) -> list[int]:
    """Return every loop enclosing one block's direct ISA leaf."""
    leaf = _direct_leaf(ir, block_nid)
    loops = [ancestor for ancestor in ir.tree.ancestors(leaf) if isinstance(ir.tree.data(ancestor), ForNode)]
    return loops


def _enclosing_loop(ir: KernelIR, block_nid: int, dimension: str, extent: int) -> int:
    """Return the unique enclosing loop with a dimension and trip count."""
    loops = [
        nid
        for nid in _enclosing_loops(ir, block_nid)
        if _loop_dimension(ir, nid) == dimension and ir.tree.loop(nid).extent == extent
    ]
    if len(loops) != 1:
        raise RuntimeError(
            f"block {block_nid}: expected one enclosing {dimension} loop with extent {extent}, found {loops}"
        )
    return loops[0]


def _loop(ir: KernelIR, block_nid: int, dimension: str, extent: int) -> int:
    """Return the unique owned loop with a concrete dimension and trip count."""
    loops = [
        nid
        for nid in _own_loops(ir, block_nid)
        if _loop_dimension(ir, nid) == dimension and ir.tree.loop(nid).extent == extent
    ]
    if len(loops) != 1:
        raise RuntimeError(
            f"block {block_nid}: expected one {dimension} loop with extent {extent}, found {len(loops)}: {loops}"
        )
    return loops[0]


def _outermost_loop(ir: KernelIR, block_nid: int, dimension: str) -> int:
    """Return the outermost loop owned by a block for one dimension."""
    loops = [nid for nid in _own_loops(ir, block_nid) if _loop_dimension(ir, nid) == dimension]
    if not loops:
        raise RuntimeError(f"block {block_nid} has no loop for dimension {dimension}")
    return loops[0]


def _innermost_loop(ir: KernelIR, block_nid: int, dimension: str) -> int:
    """Return the innermost loop owned by a block for one dimension."""
    loops = [nid for nid in _own_loops(ir, block_nid) if _loop_dimension(ir, nid) == dimension]
    if not loops:
        raise RuntimeError(f"block {block_nid} has no loop for dimension {dimension}")
    return loops[-1]


def _largest_factor_at_most(value: int, limit: int) -> int:
    """Return the largest divisor of ``value`` no greater than ``limit``."""
    factors = [factor for factor in range(1, min(value, limit) + 1) if value % factor == 0]
    result = max(factors)
    return result


def _balanced_factors(value: int) -> tuple[int, int]:
    """Return a near-square outer-to-inner factorization."""
    inner = _largest_factor_at_most(value, math.isqrt(value))
    factors = (value // inner, inner)
    return factors


def _split_loop(builder: _PlanBuilder, loop_nid: int, factors: tuple[int, int], rationale: str) -> None:
    """Split one explicit loop into the requested factors."""
    builder.apply(
        Split(),
        lambda option: isinstance(option, SplitOption)
        and option.target_nid == loop_nid
        and option.target_axis is None
        and option.factors == factors,
        rationale,
    )


def _split_tensorized(
    builder: _PlanBuilder, block_nid: int, dimension: str, factors: tuple[int, int], rationale: str
) -> None:
    """Split one block's tensorized dimension."""
    leaf_nid = _direct_leaf(builder.state, block_nid)
    builder.apply(
        Split(),
        lambda option: isinstance(option, SplitOption)
        and option.target_nid == leaf_nid
        and option.target_axis == dimension
        and option.factors == factors,
        rationale,
    )


def _reorder(builder: _PlanBuilder, outer_nid: int, inner_nid: int, rationale: str) -> None:
    """Swap one adjacent loop pair."""
    builder.apply(
        Reorder(),
        lambda option: isinstance(option, ReorderOption)
        and option.outer_nid == outer_nid
        and option.inner_nid == inner_nid,
        rationale,
    )


def _move(builder: _PlanBuilder, block_nid: int, target_loop_nid: int, index: int, rationale: str) -> None:
    """Move one block to a selected loop child slot."""
    children = [child for child in builder.state.tree.children(target_loop_nid) if child != block_nid]
    effective_index = len(children) if index == -1 else 0 if index == -2 else index
    builder.apply(
        CodeMotion(),
        lambda option: isinstance(option, CodeMotionOption)
        and option.block_nid == block_nid
        and option.target_loop_nid == target_loop_nid
        and option.index == effective_index,
        rationale,
    )


def _place(builder: _PlanBuilder, tensor: str) -> None:
    """Move one buffer declaration to its lifetime-safe scope."""
    builder.apply(
        BufferPlacement(),
        lambda option: isinstance(option, BufferPlacementOption) and option.tensor == tensor,
        f"place {tensor} at its lifetime-safe scope",
    )


def _compact(builder: _PlanBuilder, tensor: str) -> None:
    """Compact one buffer after its accesses become local."""
    builder.apply(
        BufferCompaction(),
        lambda option: isinstance(option, BufferCompactionOption) and option.tensor == tensor,
        f"compact localized buffer {tensor}",
    )


def _layout(builder: _PlanBuilder, tensor: str, list_len: int) -> None:
    """Set one buffer's allocation-list granularity."""
    builder.apply(
        BufferLayout(),
        lambda option: isinstance(option, BufferLayoutOption)
        and option.tensor == tensor
        and option.list_len == list_len,
        f"set {tensor} allocation list length to {list_len}",
    )


def _matmul_blocks(ir: KernelIR) -> tuple[int, dict[str, str]]:
    """Return the single matmul block and its operand bindings."""
    blocks = _matching_blocks(ir, "nc_matmul", {})
    if len(blocks) != 1:
        raise RuntimeError(f"matmul schedule expects one nc_matmul block, found {len(blocks)}")
    block_nid = blocks[0]
    operands = _operands(ir, block_nid)
    return block_nid, operands


def _bubble_dimension_outward(builder: _PlanBuilder, block_nid: int, dimension: str) -> None:
    """Move one block dimension outward through every preceding owned loop."""
    done = False
    while not done:
        loops = _own_loops(builder.state, block_nid)
        target_index = next(
            index for index, nid in enumerate(loops) if _loop_dimension(builder.state, nid) == dimension
        )
        if target_index == 0:
            done = True
        else:
            _reorder(
                builder,
                loops[target_index - 1],
                loops[target_index],
                f"move {dimension} outward in the matmul loop nest",
            )


def _sink_inner_dimension(builder: _PlanBuilder, block_nid: int, dimension: str) -> None:
    """Move an innermost factor through following loops of other dimensions."""
    done = False
    while not done:
        loops = _own_loops(builder.state, block_nid)
        target = _innermost_loop(builder.state, block_nid, dimension)
        target_index = loops.index(target)
        if target_index == len(loops) - 1:
            done = True
        else:
            _reorder(builder, target, loops[target_index + 1], f"move the inner {dimension} factor inward")


def _localize_free_axis_block(
    builder: _PlanBuilder,
    block_nid: int,
    free_dimension: str,
    parallel_dimension: str,
    target_loop_nid: int,
    index: int,
) -> None:
    """Tile and move a drain-style block under the matmul free-axis loop."""
    free_extent = builder.state.axis_extent(free_dimension)
    free_tile = min(512, free_extent)
    factors = (free_extent // free_tile, free_tile)
    _split_tensorized(builder, block_nid, free_dimension, factors, f"tile {free_dimension} to the ISA free-axis width")
    free_loop = _outermost_loop(builder.state, block_nid, free_dimension)
    parallel_loop = _outermost_loop(builder.state, block_nid, parallel_dimension)
    _reorder(builder, parallel_loop, free_loop, f"make {free_dimension} the block's outer tile loop")
    _move(
        builder, block_nid, target_loop_nid, index, f"co-locate {_operation_name(builder.state, block_nid)} with matmul"
    )


def _localize_moving_load(
    builder: _PlanBuilder, block_nid: int, reduction_dimension: str, free_dimension: str, target_loop_nid: int
) -> None:
    """Tile and move the matmul moving-operand load under reduction."""
    reduction_loop = _outermost_loop(builder.state, block_nid, reduction_dimension)
    reduction_trip = builder.state.tree.loop(reduction_loop).extent
    reduction_inner = _largest_factor_at_most(reduction_trip, 8)
    _split_loop(
        builder,
        reduction_loop,
        (reduction_trip // reduction_inner, reduction_inner),
        "split the moving load reduction tiles for reuse",
    )
    free_extent = builder.state.axis_extent(free_dimension)
    free_tile = min(512, free_extent)
    _split_tensorized(
        builder, block_nid, free_dimension, (free_extent // free_tile, free_tile), "tile the moving load free axis"
    )
    _bubble_dimension_outward(builder, block_nid, free_dimension)
    _move(builder, block_nid, target_loop_nid, 0, "load the moving operand inside the reduction tile")


def _localize_stationary_load(
    builder: _PlanBuilder, block_nid: int, reduction_dimension: str, parallel_dimension: str, target_loop_nid: int
) -> None:
    """Tile and move a plain stationary-operand load under the row tile."""
    parallel_extent = builder.state.axis_extent(parallel_dimension)
    parallel_tile = min(512, parallel_extent)
    _split_tensorized(
        builder,
        block_nid,
        parallel_dimension,
        (parallel_extent // parallel_tile, parallel_tile),
        "tile the stationary load row axis",
    )
    reduction_loop = _outermost_loop(builder.state, block_nid, reduction_dimension)
    reduction_trip = builder.state.tree.loop(reduction_loop).extent
    reduction_inner = _largest_factor_at_most(reduction_trip, 8)
    _split_loop(
        builder,
        reduction_loop,
        (reduction_trip // reduction_inner, reduction_inner),
        "split the stationary load reduction tiles",
    )
    inner_reduction = _innermost_loop(builder.state, block_nid, reduction_dimension)
    parallel_loop = _outermost_loop(builder.state, block_nid, parallel_dimension)
    _reorder(builder, inner_reduction, parallel_loop, "reuse stationary tiles across the row tile")
    _move(builder, block_nid, target_loop_nid, 0, "load the stationary operand inside the row tile")


def _plan_matmul(ir: KernelIR) -> SchedulePlan:
    """Build a high-utilization schedule for a single matmul graph."""
    builder = _PlanBuilder(ir)
    if TransposeThroughTensorCopy().analyze(builder.state):
        builder.apply_first(TransposeThroughTensorCopy(), "commute the input transpose through its copy")
        transpose_through_load = TransposeThroughLoad()
        transpose_options = transpose_through_load.analyze(builder.state)
        first_tile = max(option.first_tile for option in transpose_options)
        builder.apply(
            transpose_through_load,
            lambda option: isinstance(option, TransposeThroughLoadOption) and option.first_tile == first_tile,
            "fold the input transpose into the load",
        )

    matmul_block, operands = _matmul_blocks(builder.state)
    matmul_node = builder.state.tree.block(matmul_block)
    reduction_dimension = matmul_node.axis_map["K"]
    parallel_dimension = matmul_node.axis_map["M"]
    free_dimension = matmul_node.axis_map["N"]
    _bubble_dimension_outward(builder, matmul_block, free_dimension)

    reduction_loop = _outermost_loop(builder.state, matmul_block, reduction_dimension)
    reduction_trip = builder.state.tree.loop(reduction_loop).extent
    reduction_inner = _largest_factor_at_most(reduction_trip, 8)
    _split_loop(
        builder,
        reduction_loop,
        (reduction_trip // reduction_inner, reduction_inner),
        "split the matmul reduction into outer and reusable inner tiles",
    )
    parallel_loop = _outermost_loop(builder.state, matmul_block, parallel_dimension)
    parallel_trip = builder.state.tree.loop(parallel_loop).extent
    _split_loop(builder, parallel_loop, _balanced_factors(parallel_trip), "balance the matmul row-tile loops")
    _sink_inner_dimension(builder, matmul_block, reduction_dimension)

    accumulator = operands["dst"]
    _layout(builder, accumulator, builder.state.buffer(accumulator).logical_tile_count())
    copy_block = _unique_block(builder.state, "tensor_copy", {"src": accumulator})
    copied = _operands(builder.state, copy_block)["dst"]
    store_block = _unique_block(builder.state, "dma_copy", {"src": copied})
    memset_block = _unique_block(builder.state, "memset", {"dst": accumulator})
    free_loop = _outermost_loop(builder.state, matmul_block, free_dimension)
    _localize_free_axis_block(builder, copy_block, free_dimension, parallel_dimension, free_loop, 1)
    _localize_free_axis_block(builder, store_block, free_dimension, parallel_dimension, free_loop, 2)
    _place(builder, copied)
    _compact(builder, copied)
    _layout(builder, copied, builder.state.buffer(copied).logical_tile_count())
    _localize_free_axis_block(builder, memset_block, free_dimension, parallel_dimension, free_loop, 0)
    _place(builder, accumulator)
    _compact(builder, accumulator)

    moving = operands["moving"]
    moving_block = _unique_block(builder.state, "dma_copy", {"dst": moving})
    reduction_outer = _outermost_loop(builder.state, matmul_block, reduction_dimension)
    _localize_moving_load(builder, moving_block, reduction_dimension, free_dimension, reduction_outer)
    _place(builder, moving)
    _compact(builder, moving)
    _layout(builder, moving, builder.state.buffer(moving).logical_tile_count())

    stationary = operands["stationary"]
    stationary_dma_blocks = _matching_blocks(builder.state, "dma_copy", {"dst": stationary})
    if stationary_dma_blocks:
        row_outer = _outermost_loop(builder.state, matmul_block, parallel_dimension)
        _localize_stationary_load(builder, stationary_dma_blocks[0], reduction_dimension, parallel_dimension, row_outer)
        _place(builder, stationary)
        _compact(builder, stationary)
        _layout(builder, stationary, builder.state.buffer(stationary).logical_tile_count())

    reduction_outer = _outermost_loop(builder.state, matmul_block, reduction_dimension)
    builder.apply(
        RFactor(),
        lambda option: isinstance(option, RFactorOption)
        and option.target_loop_nid == reduction_outer
        and option.factor_axis == 0,
        "factor the outer matmul reduction",
    )
    _layout(builder, accumulator, 1)
    _compact(builder, accumulator)
    rfactor_buffers = [name for name in builder.state.all_buffers() if name.startswith("sbuf_rfactor")]
    if len(rfactor_buffers) != 1:
        raise RuntimeError(f"expected one RFactor buffer, found {rfactor_buffers}")
    _compact(builder, rfactor_buffers[0])
    if not stationary_dma_blocks:
        _layout(builder, stationary, builder.state.buffer(stationary).logical_tile_count())

    plan = SchedulePlan(state=builder.state, steps=tuple(builder.steps), family="matmul", strategy="tiled")
    return plan


def _pipeline_matmul_plan(
    plan: SchedulePlan,
    strategy: str,
    dimension: str,
    innermost: bool,
    stages: tuple[int, ...],
    layout: tuple[str, int] | None,
) -> SchedulePlan:
    """Add one semantic pipeline and optional allocation choice to a matmul plan."""
    builder = _PlanBuilder(plan.state)
    matmul_block, _operands_by_name = _matmul_blocks(builder.state)
    loop_nid = (
        _innermost_loop(builder.state, matmul_block, dimension)
        if innermost
        else _outermost_loop(builder.state, matmul_block, dimension)
    )
    builder.apply(
        SoftwarePipeline(),
        lambda option: isinstance(option, SoftwarePipelineOption)
        and option.loop_nid == loop_nid
        and option.stages == stages,
        f"pipeline the matmul {dimension} loop with stages {stages}",
    )
    if layout is not None:
        tensor, list_len = layout
        if builder.state.buffer(tensor).list_len != list_len:
            _layout(builder, tensor, list_len)
    candidate = SchedulePlan(
        state=builder.state, steps=(*plan.steps, *builder.steps), family=plan.family, strategy=strategy
    )
    return candidate


def _matmul_profile_plans(plan: SchedulePlan) -> tuple[SchedulePlan, ...]:
    """Build a bounded profile beam for matmul schedules with an HBM transpose."""
    matmul_block, operands = _matmul_blocks(plan.state)
    matmul_node = plan.state.tree.block(matmul_block)
    reduction_dimension = matmul_node.axis_map["K"]
    parallel_dimension = matmul_node.axis_map["M"]
    free_dimension = matmul_node.axis_map["N"]
    stationary = operands["stationary"]
    transpose_blocks = _matching_blocks(plan.state, "dma_transpose", {"dst": stationary})
    if not transpose_blocks:
        return (plan,)
    if len(transpose_blocks) != 1:
        raise RuntimeError(f"expected one stationary DMA transpose, found {transpose_blocks}")

    builder = _PlanBuilder(plan.state)
    if builder.state.buffer(stationary).list_len != 1:
        _layout(builder, stationary, 1)
    transpose_block = transpose_blocks[0]
    reduction_loop = _outermost_loop(builder.state, transpose_block, reduction_dimension)
    reduction_trip = builder.state.tree.loop(reduction_loop).extent
    if reduction_trip < 4 or reduction_trip % 2:
        return (plan,)
    _split_loop(
        builder,
        reduction_loop,
        (reduction_trip // 2, 2),
        "split the stationary transpose reduction loop into two-operation batches",
    )
    batch_loop = _innermost_loop(builder.state, transpose_block, reduction_dimension)
    builder.apply(
        BatchPermutation(),
        lambda option: isinstance(option, BatchPermutationOption) and option.loop_nid == batch_loop,
        "batch adjacent stationary DMA transposes",
    )
    batched = SchedulePlan(
        state=builder.state, steps=(*plan.steps, *builder.steps), family=plan.family, strategy="batched-transpose"
    )
    update_blocks = _matching_blocks(batched.state, "tensor_tensor", {})
    if len(update_blocks) != 1:
        raise RuntimeError(f"expected one matmul accumulation update, found {update_blocks}")
    output = _operands(batched.state, update_blocks[0])["dst"]
    output_list_len = _largest_factor_at_most(batched.state.buffer(output).logical_tile_count(), 8)
    moving = operands["moving"]
    candidates = (
        batched,
        _pipeline_matmul_plan(batched, "batched-transpose-inner-overlap", parallel_dimension, True, (0, 1, 1, 2), None),
        _pipeline_matmul_plan(
            batched,
            "batched-transpose-inner-output-list",
            parallel_dimension,
            True,
            (0, 0, 1, 2),
            (output, output_list_len),
        ),
        _pipeline_matmul_plan(
            batched, "batched-transpose-outer-output-list", free_dimension, False, (0, 1, 1), (output, output_list_len)
        ),
        _pipeline_matmul_plan(
            batched, "batched-transpose-reduction-prefetch", reduction_dimension, False, (0, 1), (moving, 1)
        ),
    )
    return candidates


def _layout_plan(plan: SchedulePlan, strategy: str, tensor: str, list_len: int) -> SchedulePlan:
    """Create one candidate by changing a single legal allocation granularity."""
    builder = _PlanBuilder(plan.state)
    _layout(builder, tensor, list_len)
    candidate = SchedulePlan(
        state=builder.state, steps=(*plan.steps, *builder.steps), family=plan.family, strategy=strategy
    )
    return candidate


def _attention_profile_plans(plan: SchedulePlan) -> tuple[SchedulePlan, ...]:
    """Build allocation finalists for the long-lived attention row state."""
    score_matmul, output_matmul = _attention_matmuls(plan.state)
    row_dimension = plan.state.tree.block(score_matmul).axis_map["M"]
    row_extent = plan.state.axis_extent(row_dimension)
    row_states = [
        name
        for name, buffer in plan.state.all_buffers().items()
        if buffer.location == "sbuf"
        and buffer.shape == (row_extent,)
        and buffer.versions == 1
        and buffer.logical_tile_count() > 1
    ]
    if len(row_states) != 2:
        raise RuntimeError(f"expected two full-row attention state buffers, found {row_states}")
    candidates = [plan]
    for tensor in row_states:
        logical_tiles = plan.state.buffer(tensor).logical_tile_count()
        list_len = _largest_factor_at_most(logical_tiles, 8)
        candidates.append(_layout_plan(plan, f"online-pipelined-{tensor}-list", tensor, list_len))
    value = _operands(plan.state, output_matmul)["moving"]
    value_tiles = plan.state.buffer(value).logical_tile_count()
    value_list_len = _largest_factor_at_most(value_tiles, 2)
    if value_list_len > 1:
        candidates.append(_layout_plan(plan, "online-pipelined-value-list", value, value_list_len))
    return tuple(candidates)


def _copy_propagate_to_operation(builder: _PlanBuilder, operation_name: str, rationale: str) -> None:
    """Propagate the copy immediately preceding a selected consumer operation."""
    builder.apply(
        CopyPropagation(),
        lambda option: isinstance(option, CopyPropagationOption)
        and _operation_name(builder.state, option.consumer_block_nid) == operation_name,
        rationale,
    )


def _split_row_chain(builder: _PlanBuilder, row_dimension: str, last_block_nid: int) -> None:
    """Split every row loop through a selected terminal block."""
    reached_last = False
    blocks = list(builder.state.tree.blocks())
    for block_nid in blocks:
        if block_nid == builder.state.tree.root or reached_last:
            continue
        matching_loops = [
            nid for nid in _own_loops(builder.state, block_nid) if _loop_dimension(builder.state, nid) == row_dimension
        ]
        if matching_loops:
            row_loop = matching_loops[0]
            trip = builder.state.tree.loop(row_loop).extent
            inner = _largest_factor_at_most(trip, 8)
            _split_loop(
                builder,
                row_loop,
                (trip // inner, inner),
                f"split the shared {row_dimension} row loop for online scheduling",
            )
        reached_last = block_nid == last_block_nid
    if not reached_last:
        raise RuntimeError(f"row-chain terminal block {last_block_nid} was not visited")


def _apply_largest_online_fusion(builder: _PlanBuilder) -> OnlineFusionOption:
    """Apply the largest available online-fusion chunk."""
    options = OnlineFusion().analyze(builder.state)
    if not options:
        raise RuntimeError("online fusion exposed no legal options")
    chunk_size = max(option.chunk_size for option in options)
    option = builder.apply(
        OnlineFusion(),
        lambda candidate: isinstance(candidate, OnlineFusionOption) and candidate.chunk_size == chunk_size,
        f"fuse the reduction and consumer online with chunk size {chunk_size}",
    )
    if not isinstance(option, OnlineFusionOption):
        raise TypeError(f"expected OnlineFusionOption, got {type(option).__name__}")
    return option


def _rms_body_blocks(ir: KernelIR, stationary: str, partial: str, final_accumulator: str) -> tuple[int, ...]:
    """Return the online RMSNorm row body in dependency order."""
    transpose_block = _unique_block(ir, "dma_transpose", {"dst": stationary})
    source = _operands(ir, transpose_block)["src"]
    reduction_block = _unique_block(ir, "activation_reduce", {"data": source})
    reduction_operands = _operands(ir, reduction_block)
    reduction_chunk = reduction_operands["reduce_res"]
    reduction_copy = _unique_block(ir, "tensor_copy", {"src": reduction_chunk})
    reduction_current = _operands(ir, reduction_copy)["dst"]
    factor_block = _unique_block(ir, "activation", {"data": reduction_current})
    factor = _operands(ir, factor_block)["dst"]
    partial_copy = _unique_block(ir, "tensor_copy", {"src": partial})
    output_chunk = _operands(ir, partial_copy)["dst"]
    state_copy = _unique_block(ir, "tensor_copy", {"src": output_chunk})
    output_state = _operands(ir, state_copy)["dst"]
    update_block = _unique_block(ir, "tensor_scalar", {"data": output_state, "operand0": factor})
    output_copy = _unique_block(ir, "tensor_copy", {"src": final_accumulator})
    output = _operands(ir, output_copy)["dst"]
    store_block = _unique_block(ir, "dma_copy", {"src": output})
    initializer = _unique_block(ir, "memset", {"dst": partial})
    matmul_block = _unique_block(ir, "nc_matmul", {"stationary": stationary, "dst": partial})
    blocks = (
        initializer,
        transpose_block,
        reduction_block,
        reduction_copy,
        factor_block,
        matmul_block,
        partial_copy,
        state_copy,
        update_block,
        output_copy,
        store_block,
    )
    return blocks


def _move_rms_body(builder: _PlanBuilder, target_loop_nid: int, body_blocks: tuple[int, ...]) -> None:
    """Gather the online RMSNorm body under one shared row loop."""
    for index, block_nid in enumerate(body_blocks, 1):
        _move(
            builder,
            block_nid,
            target_loop_nid,
            index,
            f"place {_operation_name(builder.state, block_nid)} at row-body slot {index}",
        )


def _rms_buffer_order(
    ir: KernelIR, moving: str, stationary: str, partial: str
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Derive placement and compaction order from the online RMSNorm body."""
    if len(ir.return_names) != 1:
        raise RuntimeError("RMS heuristic requires a single-output kernel")
    transpose_block = _unique_block(ir, "dma_transpose", {"dst": stationary})
    source = _operands(ir, transpose_block)["src"]
    reduction_block = _unique_block(ir, "activation_reduce", {"data": source})
    reduction_operands = _operands(ir, reduction_block)
    scratch = reduction_operands["dst"]
    chunk = reduction_operands["reduce_res"]
    factor_block = _matching_blocks(ir, "activation", {})
    deferred_blocks = [
        block_nid
        for block_nid in factor_block
        if _operands(ir, block_nid).get("data") == chunk
        or _operands(ir, block_nid).get("data", "").startswith("sbuf_square_sum_online")
    ]
    if len(deferred_blocks) != 1:
        raise RuntimeError(f"expected one deferred RMS factor block, found {deferred_blocks}")
    deferred = _operands(ir, deferred_blocks[0])["dst"]
    store_blocks = [
        block_nid
        for block_nid in _matching_blocks(ir, "dma_copy", {})
        if _operands(ir, block_nid).get("dst") == ir.return_names[0]
    ]
    if len(store_blocks) != 1:
        raise RuntimeError(f"expected one final RMS store, found {store_blocks}")
    output = _operands(ir, store_blocks[0])["src"]
    placement = (moving, scratch, deferred, chunk, output, partial, stationary, source)
    compaction = (source, stationary, partial, output, chunk, deferred, scratch)
    return placement, compaction


def _fuse_row_loops(builder: _PlanBuilder, row_dimension: str) -> int:
    """Fuse the two split row loops enclosing the unified online body."""
    option = builder.apply(
        Fuse(),
        lambda candidate: isinstance(candidate, FuseOption)
        and candidate.target_axis is None
        and len(candidate.target_nids) == 2
        and all(_loop_dimension(builder.state, nid) == row_dimension for nid in candidate.target_nids),
        "fuse the row-loop factors after the body is unified",
    )
    if not isinstance(option, FuseOption):
        raise TypeError(f"expected FuseOption, got {type(option).__name__}")
    fused_loops = [
        nid
        for nid in builder.state.tree.preorder()
        if isinstance(builder.state.tree.data(nid), ForNode)
        and _loop_dimension(builder.state, nid) == row_dimension
        and len(builder.state.tree.children(nid)) > 1
    ]
    if len(fused_loops) != 1:
        raise RuntimeError(f"expected one unified row loop, found {fused_loops}")
    return fused_loops[0]


def _pipeline_rms_body(builder: _PlanBuilder, loop_nid: int) -> None:
    """Pipeline the unified RMSNorm body by load, compute, update, and store."""
    child_count = len(builder.state.tree.children(loop_nid))
    if child_count < 4:
        raise RuntimeError(f"RMS pipeline requires at least four children, found {child_count}")
    stages = (0,) + (1,) * (child_count - 3) + (2, 3)
    builder.apply(
        SoftwarePipeline(),
        lambda option: isinstance(option, SoftwarePipelineOption)
        and option.loop_nid == loop_nid
        and option.stages == stages,
        "pipeline the unified row body across load, compute, update, and store",
    )


def _plan_rmsnorm_matmul(ir: KernelIR) -> SchedulePlan:
    """Build the online, row-local RMSNorm plus matmul schedule."""
    builder = _PlanBuilder(ir)
    matmul_block, matmul_operands = _matmul_blocks(builder.state)
    matmul_node = builder.state.tree.block(matmul_block)
    row_dimension = matmul_node.axis_map["M"]
    reduction_dimension = matmul_node.axis_map["K"]
    free_dimension = matmul_node.axis_map["N"]
    accumulator = matmul_operands["dst"]
    output_copy = _unique_block(builder.state, "tensor_copy", {"src": accumulator})
    _split_row_chain(builder, row_dimension, output_copy)
    _apply_largest_online_fusion(builder)

    matmul_block, matmul_operands = _matmul_blocks(builder.state)
    moving = matmul_operands["moving"]
    stationary = matmul_operands["stationary"]
    partial = matmul_operands["dst"]
    moving_block = _unique_block(builder.state, "dma_copy", {"dst": moving})
    free_extent = builder.state.axis_extent(free_dimension)
    free_tile = min(512, free_extent)
    _split_tensorized(
        builder,
        moving_block,
        free_dimension,
        (free_extent // free_tile, free_tile),
        "tile the RMS matmul moving operand",
    )
    initializer = _unique_block(builder.state, "memset", {"dst": partial})
    _split_tensorized(
        builder,
        initializer,
        free_dimension,
        (free_extent // free_tile, free_tile),
        "tile the online partial initializer",
    )
    final_copy = _unique_block(builder.state, "tensor_copy", {"src": accumulator})
    _split_tensorized(
        builder, final_copy, free_dimension, (free_extent // free_tile, free_tile), "tile the final output copy"
    )
    output = _operands(builder.state, final_copy)["dst"]
    store_block = _unique_block(builder.state, "dma_copy", {"src": output})
    store_row_loop = _outermost_loop(builder.state, store_block, row_dimension)
    store_trip = builder.state.tree.loop(store_row_loop).extent
    store_inner = _largest_factor_at_most(store_trip, 8)
    _split_loop(builder, store_row_loop, (store_trip // store_inner, store_inner), "split the output store row loop")
    _split_tensorized(
        builder, store_block, free_dimension, (free_extent // free_tile, free_tile), "tile the output store free axis"
    )
    moving_free_loop = _outermost_loop(builder.state, moving_block, free_dimension)
    moving_reduction_loop = _outermost_loop(builder.state, moving_block, reduction_dimension)
    _reorder(builder, moving_reduction_loop, moving_free_loop, "make the moving-load free tile outermost")
    _sink_inner_dimension(builder, matmul_block, reduction_dimension)
    _copy_propagate_to_operation(builder, "activation", "remove the online factor adapter copy")

    transpose_block = _unique_block(builder.state, "dma_transpose", {"dst": stationary})
    source = _operands(builder.state, transpose_block)["src"]
    source_load = _unique_block(builder.state, "dma_copy", {"dst": source})
    target_loop = _innermost_loop(builder.state, source_load, row_dimension)
    body_blocks = _rms_body_blocks(builder.state, stationary, partial, accumulator)
    _move_rms_body(builder, target_loop, body_blocks)
    _copy_propagate_to_operation(builder, "activation", "propagate the row reduction chunk into its factor")
    builder.apply(
        CopyPropagation(),
        lambda option: isinstance(option, CopyPropagationOption)
        and _operation_name(builder.state, option.consumer_block_nid) == "tensor_copy"
        and _operands(builder.state, option.copy_block_nid).get("src") == partial,
        "remove the redundant online-output chunk copy",
    )
    _copy_propagate_to_operation(builder, "tensor_scalar", "propagate the online output state into its update")
    builder.apply_first(FusePointwise(), "fuse the output update into its copy")

    placement, compaction = _rms_buffer_order(builder.state, moving, stationary, partial)
    for tensor in placement:
        _place(builder, tensor)
    for tensor in compaction:
        _compact(builder, tensor)
    unified_loop = _fuse_row_loops(builder, row_dimension)
    _pipeline_rms_body(builder, unified_loop)
    builder.apply_first(BatchPermutation(), "batch the localized RMS transpose")
    _layout(builder, moving, free_extent // free_tile)
    plan = SchedulePlan(
        state=builder.state, steps=tuple(builder.steps), family="rmsnorm-matmul", strategy="online-pipelined"
    )
    return plan


def _attention_matmuls(ir: KernelIR) -> tuple[int, int]:
    """Return score and output matmul blocks from their free-axis extents."""
    blocks = _matching_blocks(ir, "nc_matmul", {})
    if len(blocks) != 2:
        raise RuntimeError(f"attention schedule expects two matmuls, found {blocks}")
    ordered = sorted(blocks, key=lambda block_nid: ir.axis_extent(ir.tree.block(block_nid).axis_map["N"]))
    output_block, score_block = ordered
    return score_block, output_block


def _fuse_pointwise_reduction(builder: _PlanBuilder, reduction: str) -> None:
    """Fuse one pointwise producer into a named reduction."""
    builder.apply(
        FusePointwise(),
        lambda option: isinstance(option, FusePointwiseOption)
        and _operation_name(builder.state, option.consumer_block_nid) == "tensor_reduce"
        and _operation(builder.state, option.consumer_block_nid).kwargs.get("op") == reduction,
        f"fuse pointwise work into the {reduction} row reduction",
    )


def _fuse_attention_correction(builder: _PlanBuilder, stage: str) -> None:
    """Fuse one online correction subtraction into its exponential."""
    builder.apply(
        FusePointwise(),
        lambda option: isinstance(option, FusePointwiseOption)
        and _operands(builder.state, option.consumer_block_nid).get("dst", "").startswith(stage),
        f"fuse the {stage} correction",
    )


def _cleanup_attention_online_graph(builder: _PlanBuilder) -> None:
    """Canonicalize the graph exposed by complete attention online fusion."""
    _fuse_pointwise_reduction(builder, "maximum")
    _fuse_pointwise_reduction(builder, "add")
    builder.apply_first(DecomposeBroadcastSubtract(), "decompose the broadcast subtraction for native fusion")
    builder.apply(
        FusePointwise(),
        lambda option: isinstance(option, FusePointwiseOption)
        and _operation_name(builder.state, option.consumer_block_nid) == "activation_reduce",
        "fuse centering into exponentiation and row summation",
    )
    _fuse_attention_correction(builder, "online_stage1")
    _fuse_attention_correction(builder, "online_stage2")
    builder.apply_first(CommonSubexpressionElimination(), "share the identical online correction")


def _rfactor_tensorized_reduction(
    builder: _PlanBuilder, block_nid: int, dimension: str, factors: tuple[int, int], rationale: str
) -> None:
    """Factor one tensorized reduction into partials and a fold."""
    leaf_nid = _direct_leaf(builder.state, block_nid)
    builder.apply(
        RFactor(),
        lambda option: isinstance(option, RFactorOption)
        and option.target_loop_nid == leaf_nid
        and option.target_axis == dimension
        and option.factors == factors
        and option.factor_axis == 0,
        rationale,
    )


def _attention_body(ir: KernelIR, score_matmul: int, output_matmul: int) -> tuple[dict[str, int], dict[str, str]]:
    """Resolve online-attention body blocks and tensors by dataflow."""
    score_operands = _operands(ir, score_matmul)
    score_accumulator = score_operands["dst"]
    score_copy = _unique_block(ir, "tensor_copy", {"src": score_accumulator})
    scores = _operands(ir, score_copy)["dst"]
    max_reduce = _unique_block(ir, "tensor_scalar_reduce", {"data": scores})
    max_reduce_operands = _operands(ir, max_reduce)
    scaled = max_reduce_operands["dst"]
    max_rfactor = max_reduce_operands["reduce_res"]
    max_fold = _unique_block(ir, "tensor_reduce", {"data": max_rfactor})
    max_chunk = _operands(ir, max_fold)["dst"]
    max_combine_candidates = [
        block_nid
        for block_nid in _matching_blocks(ir, "tensor_tensor", {})
        if _operands(ir, block_nid).get("data2") == max_chunk
    ]
    if len(max_combine_candidates) != 1:
        raise RuntimeError(f"expected one row-max combine, found {max_combine_candidates}")
    max_combine = max_combine_candidates[0]
    max_combine_operands = _operands(ir, max_combine)
    max_state = max_combine_operands["data1"]
    max_current = max_combine_operands["dst"]
    negative_candidates = [
        block_nid
        for block_nid in _matching_blocks(ir, "activation", {"data": max_current})
        if _operation(ir, block_nid).kwargs.get("op") == "copy"
    ]
    if len(negative_candidates) != 1:
        raise RuntimeError(f"expected one negated row maximum, found {negative_candidates}")
    negative = negative_candidates[0]
    negative_tensor = _operands(ir, negative)["dst"]
    sum_reduce = _unique_block(ir, "activation_reduce", {"data": scaled, "bias": negative_tensor})
    sum_reduce_operands = _operands(ir, sum_reduce)
    exponential = sum_reduce_operands["dst"]
    sum_rfactor = sum_reduce_operands["reduce_res"]
    sum_fold = _unique_block(ir, "tensor_reduce", {"data": sum_rfactor})
    sum_chunk = _operands(ir, sum_fold)["dst"]
    stage1_candidates = [
        block_nid
        for block_nid in _matching_blocks(ir, "activation", {"data": max_current})
        if _operands(ir, block_nid).get("dst", "").startswith("online_stage1")
    ]
    if len(stage1_candidates) != 1:
        raise RuntimeError(f"expected one stage-1 correction, found {stage1_candidates}")
    stage1 = stage1_candidates[0]
    correction = _operands(ir, stage1)["dst"]
    sum_update = _unique_block(ir, "scalar_tensor_tensor", {"operand0": correction, "operand1": sum_chunk})
    sum_update_operands = _operands(ir, sum_update)
    sum_state = sum_update_operands["data"]
    sum_current = sum_update_operands["dst"]
    probability = _unique_block(ir, "dma_transpose", {"src": exponential})
    probability_tensor = _operands(ir, probability)["dst"]

    output_operands = _operands(ir, output_matmul)
    partial = output_operands["dst"]
    value = output_operands["moving"]
    output_copy = _unique_block(ir, "tensor_copy", {"src": partial})
    output_chunk = _operands(ir, output_copy)["dst"]
    carry_load_candidates = [
        block_nid
        for block_nid in _matching_blocks(ir, "dma_copy", {})
        if _operands(ir, block_nid).get("dst", "").startswith("psum_output_online_state")
    ]
    if len(carry_load_candidates) != 1:
        raise RuntimeError(f"expected one attention carry load, found {carry_load_candidates}")
    carry_load = carry_load_candidates[0]
    carry_load_operands = _operands(ir, carry_load)
    carry = carry_load_operands["src"]
    output_state = carry_load_operands["dst"]
    output_update = _unique_block(
        ir, "scalar_tensor_tensor", {"data": output_state, "operand0": correction, "operand1": output_chunk}
    )
    carry_store = _unique_block(ir, "dma_copy", {"src": output_state, "dst": carry})
    max_copy = _unique_block(ir, "tensor_copy", {"src": max_current, "dst": max_state})
    sum_copy = _unique_block(ir, "tensor_copy", {"src": sum_current, "dst": sum_state})
    score_initializer = _unique_block(ir, "memset", {"dst": score_accumulator})
    output_initializer = _unique_block(ir, "memset", {"dst": partial})
    blocks = {
        "score_initializer": score_initializer,
        "score_matmul": score_matmul,
        "score_copy": score_copy,
        "max_reduce": max_reduce,
        "max_fold": max_fold,
        "max_combine": max_combine,
        "negative": negative,
        "sum_reduce": sum_reduce,
        "sum_fold": sum_fold,
        "stage1": stage1,
        "sum_update": sum_update,
        "probability": probability,
        "output_initializer": output_initializer,
        "output_matmul": output_matmul,
        "output_copy": output_copy,
        "carry_load": carry_load,
        "output_update": output_update,
        "carry_store": carry_store,
        "max_copy": max_copy,
        "sum_copy": sum_copy,
    }
    tensors = {
        "score_accumulator": score_accumulator,
        "scores": scores,
        "scaled": scaled,
        "max_rfactor": max_rfactor,
        "max_chunk": max_chunk,
        "max_current": max_current,
        "negative": negative_tensor,
        "exponential": exponential,
        "sum_rfactor": sum_rfactor,
        "sum_chunk": sum_chunk,
        "correction": correction,
        "sum_current": sum_current,
        "probability": probability_tensor,
        "partial": partial,
        "value": value,
        "output_chunk": output_chunk,
        "output_state": output_state,
        "key": score_operands["moving"],
    }
    return blocks, tensors


def _gather_attention_row_body(builder: _PlanBuilder, target_loop: int, blocks: dict[str, int]) -> None:
    """Gather the complete score, recurrence, and output body under one row loop."""
    prepend = (
        "probability",
        "sum_update",
        "stage1",
        "sum_fold",
        "sum_reduce",
        "negative",
        "max_combine",
        "max_fold",
        "max_reduce",
        "score_copy",
        "score_matmul",
        "score_initializer",
    )
    append = ("output_matmul", "output_copy", "carry_load", "output_update", "carry_store", "max_copy", "sum_copy")
    for name in prepend:
        block_nid = blocks[name]
        _move(builder, block_nid, target_loop, 0, f"prepend attention row-body operation {name}")
    for name in append:
        block_nid = blocks[name]
        _move(builder, block_nid, target_loop, -1, f"append attention row-body operation {name}")


def _localize_attention_score_path(
    builder: _PlanBuilder, blocks: dict[str, int], tensors: dict[str, str], chunk_dimension: str, chunk_size: int
) -> int:
    """Tile the score initializer and copy inside the score matmul chunk."""
    factors = (chunk_size // 512, 512)
    _split_tensorized(
        builder, blocks["score_initializer"], chunk_dimension, factors, "tile the score initializer to the online chunk"
    )
    _split_tensorized(
        builder, blocks["score_copy"], chunk_dimension, factors, "tile the score drain to the online chunk"
    )
    score_chunk_loop = _loop(builder.state, blocks["score_matmul"], chunk_dimension, chunk_size // 512)
    _move(builder, blocks["score_initializer"], score_chunk_loop, 0, "initialize each score chunk locally")
    _move(builder, blocks["score_copy"], score_chunk_loop, -1, "drain each score chunk locally")
    return score_chunk_loop


def _factor_attention_output(
    builder: _PlanBuilder, blocks: dict[str, int], chunk_dimension: str, chunk_size: int, score_chunk_loop: int
) -> str:
    """Factor the output reduction and fold score-copy propagation."""
    reduction_loop = _outermost_loop(builder.state, blocks["output_matmul"], chunk_dimension)
    reduction_trip = builder.state.tree.loop(reduction_loop).extent
    inner = chunk_size // 512
    _split_loop(
        builder,
        reduction_loop,
        (reduction_trip // inner, inner),
        "split the output reduction for independent partial accumulation",
    )
    outer = _outermost_loop(builder.state, blocks["output_matmul"], chunk_dimension)
    builder.apply(
        RFactor(),
        lambda option: isinstance(option, RFactorOption)
        and option.target_loop_nid == outer
        and option.factor_axis == 0,
        "factor the outer output reduction",
    )
    _move(builder, blocks["max_reduce"], score_chunk_loop, 3, "consume score tiles directly in the max reduction")
    _copy_propagate_to_operation(builder, "tensor_scalar_reduce", "remove the materialized score-copy buffer")
    rfactor_buffers = [name for name in builder.state.all_buffers() if name.startswith("sbuf_rfactor")]
    if len(rfactor_buffers) != 1:
        raise RuntimeError(f"expected one output RFactor buffer, found {rfactor_buffers}")
    return rfactor_buffers[0]


def _localize_attention_epilogue(builder: _PlanBuilder, row_dimension: str) -> tuple[str, str, str]:
    """Move the deferred attention epilogue and remove its final copy."""
    reciprocal_candidates = [
        block_nid
        for block_nid in _matching_blocks(builder.state, "activation", {})
        if _operation(builder.state, block_nid).kwargs.get("op") == "reciprocal"
    ]
    if len(reciprocal_candidates) != 1:
        raise RuntimeError(f"expected one reciprocal epilogue block, found {reciprocal_candidates}")
    reciprocal_block = reciprocal_candidates[0]
    reciprocal = _operands(builder.state, reciprocal_block)["dst"]
    epilogue_load_candidates = [
        block_nid
        for block_nid in _matching_blocks(builder.state, "dma_copy", {})
        if _operands(builder.state, block_nid).get("dst", "").startswith("online_deferred_numerator")
    ]
    if len(epilogue_load_candidates) != 1:
        raise RuntimeError(f"expected one deferred numerator load, found {epilogue_load_candidates}")
    numerator_block = epilogue_load_candidates[0]
    numerator = _operands(builder.state, numerator_block)["dst"]
    final_update = _unique_block(builder.state, "tensor_scalar", {"data": numerator, "operand0": reciprocal})
    final_accumulator = _operands(builder.state, final_update)["dst"]
    target_loop = _enclosing_loop(
        builder.state, reciprocal_block, row_dimension, builder.state.axis_extent(row_dimension) // 128
    )
    final_copy = _unique_block(builder.state, "tensor_copy", {"src": final_accumulator})
    output = _operands(builder.state, final_copy)["dst"]
    store = _unique_block(builder.state, "dma_copy", {"src": output})
    _move(builder, final_copy, target_loop, -1, "append the final attention copy to the deferred epilogue")
    _move(builder, store, target_loop, -1, "append the final attention store to the deferred epilogue")
    builder.apply(
        CopyPropagation(),
        lambda option: isinstance(option, CopyPropagationOption)
        and option.copy_block_nid == final_copy
        and option.consumer_block_nid == store,
        "store the deferred attention result directly",
    )
    return reciprocal, numerator, final_accumulator


def _attention_buffer_orders(
    tensors: dict[str, str], reciprocal: str, numerator: str, final_accumulator: str, output_rfactor: str
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return lifetime-oriented placement and compaction orders."""
    placement = (
        reciprocal,
        numerator,
        final_accumulator,
        output_rfactor,
        tensors["partial"],
        tensors["score_accumulator"],
        tensors["correction"],
        tensors["output_chunk"],
        tensors["output_state"],
        tensors["sum_current"],
        tensors["sum_chunk"],
        tensors["max_current"],
        tensors["max_chunk"],
        tensors["probability"],
        tensors["exponential"],
        tensors["scaled"],
        tensors["negative"],
        tensors["max_rfactor"],
        tensors["sum_rfactor"],
        tensors["value"],
        tensors["key"],
    )
    compaction = (
        tensors["sum_rfactor"],
        tensors["max_rfactor"],
        tensors["negative"],
        tensors["scaled"],
        tensors["exponential"],
        tensors["probability"],
        tensors["max_chunk"],
        tensors["max_current"],
        tensors["sum_chunk"],
        tensors["sum_current"],
        tensors["output_state"],
        tensors["output_chunk"],
        tensors["correction"],
        tensors["score_accumulator"],
        tensors["partial"],
        final_accumulator,
        output_rfactor,
    )
    return placement, compaction


def _finalize_attention_pipeline(
    builder: _PlanBuilder, blocks: dict[str, int], tensors: dict[str, str], target_loop: int, chunk_dimension: str
) -> None:
    """Remove the score initializer, order stages, and batch the transpose."""
    builder.apply(
        EliminateIdentityInitializer(),
        lambda option: isinstance(option, EliminateIdentityInitializerOption)
        and option.tensor == tensors["score_accumulator"],
        "eliminate the score identity initializer",
    )
    _move(builder, blocks["probability"], target_loop, 5, "place probability transpose after exponentiation")
    _move(builder, blocks["sum_copy"], target_loop, 9, "place row-sum state copy before output accumulation")
    child_count = len(builder.state.tree.children(target_loop))
    stages = (0,) * 4 + (1,) * 6 + (2,) * (child_count - 10)
    builder.apply(
        SoftwarePipeline(),
        lambda option: isinstance(option, SoftwarePipelineOption)
        and option.loop_nid == target_loop
        and option.stages == stages,
        "pipeline the complete online-attention row body",
    )
    transpose_loop = _outermost_loop(builder.state, blocks["probability"], chunk_dimension)
    transpose_trip = builder.state.tree.loop(transpose_loop).extent
    _split_loop(
        builder,
        transpose_loop,
        (transpose_trip // 2, 2),
        "split probability-transpose batches for hardware permutation",
    )
    batch_loop = _loop(builder.state, blocks["probability"], chunk_dimension, 2)
    builder.apply(
        BatchPermutation(),
        lambda option: isinstance(option, BatchPermutationOption) and option.loop_nid == batch_loop,
        "permute the two localized transpose batches",
    )


def _plan_attention(ir: KernelIR) -> SchedulePlan:
    """Build a complete row-local online-attention schedule."""
    builder = _PlanBuilder(ir)
    fusion = _apply_largest_online_fusion(builder)
    builder.apply_first(OnlineFusion(), "complete the dependent online-attention recurrence")
    _cleanup_attention_online_graph(builder)
    score_matmul, output_matmul = _attention_matmuls(builder.state)
    score_node = builder.state.tree.block(score_matmul)
    chunk_dimension = score_node.axis_map["N"]
    row_dimension = score_node.axis_map["M"]
    factors = (fusion.chunk_size // 512, 512)
    max_reduce = _matching_blocks(builder.state, "tensor_scalar_reduce", {})
    sum_reduce = _matching_blocks(builder.state, "activation_reduce", {})
    if len(max_reduce) != 1 or len(sum_reduce) != 1:
        raise RuntimeError(f"expected one max and sum reducer, found max={max_reduce}, sum={sum_reduce}")
    _rfactor_tensorized_reduction(
        builder, max_reduce[0], chunk_dimension, factors, "factor the online row-maximum reduction"
    )
    _rfactor_tensorized_reduction(
        builder, sum_reduce[0], chunk_dimension, factors, "factor the online row-sum reduction"
    )
    score_matmul, output_matmul = _attention_matmuls(builder.state)
    output_node = builder.state.tree.block(output_matmul)
    output_reduction_dimension = output_node.axis_map["K"]
    output_row_dimension = output_node.axis_map["M"]
    output_reduction_loop = _outermost_loop(builder.state, output_matmul, output_reduction_dimension)
    output_row_loop = _outermost_loop(builder.state, output_matmul, output_row_dimension)
    _reorder(builder, output_reduction_loop, output_row_loop, "make the attention output row loop outermost")
    blocks, tensors = _attention_body(builder.state, score_matmul, output_matmul)
    target_loop = _outermost_loop(builder.state, blocks["output_initializer"], row_dimension)
    _gather_attention_row_body(builder, target_loop, blocks)
    score_chunk_loop = _localize_attention_score_path(builder, blocks, tensors, chunk_dimension, fusion.chunk_size)
    output_rfactor = _factor_attention_output(builder, blocks, chunk_dimension, fusion.chunk_size, score_chunk_loop)
    reciprocal, numerator, final_accumulator = _localize_attention_epilogue(builder, row_dimension)
    placement, compaction = _attention_buffer_orders(tensors, reciprocal, numerator, final_accumulator, output_rfactor)
    for tensor in placement:
        _place(builder, tensor)
    for tensor in compaction:
        _compact(builder, tensor)
    _finalize_attention_pipeline(builder, blocks, tensors, target_loop, chunk_dimension)
    plan = SchedulePlan(
        state=builder.state, steps=tuple(builder.steps), family="attention", strategy="online-pipelined"
    )
    return plan


def operation_names(ir: KernelIR) -> tuple[str, ...]:
    """Return direct ISA operation names in execution order."""
    names = tuple(ir.tree.isa(nid).op_cls.NAME for nid in ir.tree.preorder() if isinstance(ir.tree.data(nid), ISANode))
    return names


def build_heuristic_plan(ir: KernelIR) -> SchedulePlan:
    """Detect the kernel family and lower its deterministic schedule rules."""
    names = operation_names(ir)
    matmul_count = names.count("nc_matmul")
    if matmul_count == 2 and "tensor_reduce" in names:
        plan = _plan_attention(ir)
    elif matmul_count == 1 and "activation_reduce" in names:
        plan = _plan_rmsnorm_matmul(ir)
    elif matmul_count == 1:
        plan = _plan_matmul(ir)
    else:
        raise ValueError(f"no heuristic schedule is implemented for operation sequence {names}")
    return plan


def build_heuristic_plans(ir: KernelIR) -> tuple[SchedulePlan, ...]:
    """Build the bounded candidate set selected by semantic schedule heuristics."""
    plan = build_heuristic_plan(ir)
    if plan.family == "matmul":
        plans = _matmul_profile_plans(plan)
    elif plan.family == "attention":
        plans = _attention_profile_plans(plan)
    else:
        plans = (plan,)
    return plans


__all__ = ["SchedulePlan", "ScheduleStep", "build_heuristic_plan", "build_heuristic_plans", "operation_names"]
