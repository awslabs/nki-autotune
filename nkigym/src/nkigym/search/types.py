"""Public types for iterative schedule refinement."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from re import findall
from typing import Any, cast

from nkigym.codegen import render
from nkigym.ir import KernelIR
from nkigym.ir.arith.expr import Const, to_affine
from nkigym.ir.tree import PARTITION_DIM, BlockNode, ForNode, ISANode
from nkigym.ops.base import AxisRole
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.profile.types import ProfileMetrics
from nkigym.search.program_sharding import axis_loop_for_block, configured_program_shards, owning_block
from nkigym.transforms import Transform, TransformOption

Action = tuple[Transform[Any], TransformOption]
_TRANSFORM_PRIORITY = {
    "CommonSubexpressionElimination": 100,
    "CommuteBroadcastFactor": 92,
    "SetFirstWriteOverwrite": 96,
    "EliminateIdentityInitializer": 95,
    "ProgramShard": 94,
    "ProgramStorePartition": 95,
    "FusePointwise": 90,
    "CopyPropagation": 85,
    "EliminateDeadProducer": 84,
    "DecomposeBroadcastSubtract": 80,
    "OnlineFusion": 91,
    "CodeMotion": 70,
    "BufferPlacement": 65,
    "BufferRegionNormalization": 64,
    "BufferCompaction": 60,
    "RFactor": 55,
    "SoftwarePipeline": 56,
    "BufferLayout": 45,
    "Fuse": 40,
    "Split": 35,
    "Reorder": 30,
    "BatchPermutation": 60,
    "TransposeThroughMatmul": 92,
    "TransposeThroughTensorCopy": 20,
    "TransposePair": 15,
}


@dataclass(frozen=True)
class PolicyContext:
    """State exposed to a policy before one refinement step."""

    state: KernelIR
    transforms: tuple[Transform[Any], ...]
    legal_actions: tuple[Action, ...]
    evaluations: tuple[ProfileMetrics, ...]
    compile_failures: tuple[str, ...]
    evaluation_attempts: int
    max_transforms: int


class Policy:
    """Deterministically refine legal actions using generic schedule heuristics."""

    def __init__(self) -> None:
        """Initialize the set of options already selected by this policy."""
        self._selected: set[tuple[str, str]] = set()

    def select_actions(self, context: PolicyContext) -> tuple[Action, ...]:
        """Return an ordered transform sequence, or an empty tuple to finish."""
        budget = context.max_transforms
        regressed = (
            not context.compile_failures
            and len(context.evaluations) > 1
            and context.evaluations[-1].latency_ms > context.evaluations[-2].latency_ms
        )
        if regressed and not _continues_program_shard(context.state, context.legal_actions):
            budget = 1
        legal_actions, state = context.legal_actions, context.state
        selected: list[Action] = []
        batch_blocked: set[tuple[str, str]] = set()
        compile_pending = not context.evaluations or bool(context.compile_failures)
        failure_location = _failure_location(context.compile_failures)
        failure_tensors = _failure_tensors(state, context.compile_failures, failure_location)
        bank_failure = bool(context.compile_failures and "multiple psum banks" in context.compile_failures[-1].lower())
        failure_tile_size = _failure_tile_size(context.compile_failures)
        access_tile_size = _failure_access_tile_size(state, failure_tensors)
        for _index in range(budget):
            blocked = self._selected | batch_blocked
            candidates = [action for action in legal_actions if _action_key(state, action) not in blocked]
            if not candidates:
                break
            tile_preferences = _tile_preferences(state)
            detected_access_tile = _failure_access_tile_size(state, failure_tensors)
            if detected_access_tile is not None:
                access_tile_size = detected_access_tile
            independent_shard_extent = max(
                (
                    state.tree.loop(getattr(option, "loop_nid")).extent
                    for transform, option in candidates
                    if type(transform).__name__ == "ProgramShard" and getattr(option, "reduction_tensor", None) is None
                ),
                default=0,
            )
            split_preparations = (
                {}
                if _continues_program_shard(state, tuple(candidates))
                else _split_transform_preparations(state, candidates, context.transforms)
            )
            ranked = sorted(
                candidates,
                key=lambda candidate: _action_score(
                    state,
                    candidate,
                    compile_pending,
                    failure_location,
                    failure_tensors,
                    tile_preferences,
                    bank_failure,
                    failure_tile_size,
                    access_tile_size,
                    split_preparations,
                    independent_shard_extent,
                ),
                reverse=True,
            )
            action = ranked[0]
            score = _action_score(
                state,
                action,
                compile_pending,
                failure_location,
                failure_tensors,
                tile_preferences,
                bank_failure,
                failure_tile_size,
                access_tile_size,
                split_preparations,
                independent_shard_extent,
            )
            if score[0] <= 0 or (selected and score[0] <= 35):
                break
            transform, option = action
            if selected and compile_pending and type(transform).__name__ == "BufferLayout":
                break
            self._selected.add(_action_key(state, action))
            state = transform.apply(state, option)
            selected.append(action)
            remaining_failures = frozenset(failure_tensors & state.all_buffers().keys())
            diagnostic_split = type(transform).__name__ == "Split" and score[0] == 89
            if compile_pending and failure_tensors and (not remaining_failures or diagnostic_split):
                break
            failure_tensors = remaining_failures
            if type(transform).__name__ == "Reorder":
                batch_blocked.add(_action_key(state, action))
            compacted_tensor = getattr(option, "tensor", None)
            if (
                compile_pending
                and type(transform).__name__ == "BufferCompaction"
                and compacted_tensor in failure_tensors
            ):
                recovery_tile_size = failure_tile_size if failure_tile_size is not None else access_tile_size
                if (
                    recovery_tile_size is None
                    or state.buffer(compacted_tensor).per_tile_physical_shape()[-1] <= recovery_tile_size
                ):
                    break
            transform_name = type(transform).__name__
            if transform_name == "ProgramShard" and getattr(option, "reduction_tensor", None) is None:
                axis = getattr(option, "axis", None)
                programs = getattr(option, "programs", None)
                legal_actions = tuple(
                    (transform, candidate)
                    for candidate in transform.analyze(state)
                    if getattr(candidate, "axis", None) == axis and getattr(candidate, "programs", None) == programs
                )
                if legal_actions:
                    continue
                break
            if transform_name in {"Fuse", "ProgramShard", "SoftwarePipeline", "TransposeThroughMatmul"}:
                break
            if transform_name == "Split" and score[0] == 79:
                break
            legal_actions = tuple(
                (candidate_transform, candidate_option)
                for candidate_transform in context.transforms
                for candidate_option in candidate_transform.analyze(state)
            )
        return tuple(selected)


def _action_key(state: KernelIR, action: Action) -> tuple[str, str]:
    """Return a stable policy-local identity for one legal action."""
    transform, option = action
    payload = vars(option)
    fields = (
        "tensor",
        "list_len",
        "block_nid",
        "copy_block_nid",
        "producer_block_nid",
        "redundant_block_nid",
        "initializer_leaf_nid",
        "target_nid",
        "target_loop_nid",
        "target_axis",
        "target_nids",
        "factors",
        "stages",
        "axis",
        "programs",
        "loop_nid",
        "outer_nid",
        "consumer_nid",
        "consumer_operand",
        "index",
        "operand",
        "source",
        "transpose_nid",
        "first_transpose_nid",
    )
    identity = tuple((name, payload[name]) for name in fields if name in payload)
    tensor = payload.get("tensor")
    if type(transform).__name__ in {"BufferPlacement", "BufferRegionNormalization", "BufferCompaction"} and isinstance(
        tensor, str
    ):
        access_state = tuple(
            (
                leaf_nid,
                tuple(nid for nid in state.tree.ancestors(leaf_nid) if isinstance(state.tree.data(nid), ForNode)),
                tuple(
                    (slot, region)
                    for slot, region in state.tree.isa(leaf_nid).operand_bindings.items()
                    if region.tensor == tensor
                ),
            )
            for leaf_nid in state.dependency.touches_by_tensor.get(tensor, ())
        )
        identity += (("buffer", state.buffer(tensor)), ("access_state", access_state))
    if type(transform).__name__ == "Reorder":
        loop_nids = (payload.get("outer_nid"), payload.get("inner_nid"))
        if all(isinstance(nid, int) and nid in state.tree.graph for nid in loop_nids):
            loops = tuple(state.tree.loop(cast(int, nid)) for nid in loop_nids)
            identity += (("loop_payloads", tuple((loop.loop_var, loop.extent) for loop in loops)),)
    return (type(transform).__name__, repr(identity or option))


def _split_transform_preparations(
    state: KernelIR, candidates: list[Action], transforms: tuple[Transform[Any], ...]
) -> dict[tuple[str, str], int]:
    """Return first legal splits on shortest paths to split-sensitive transforms."""
    split_transform = next((transform for transform in transforms if type(transform).__name__ == "Split"), None)
    if split_transform is None:
        return {}
    first_options = [option for transform, option in candidates if transform is split_transform]
    if not first_options:
        return {}
    preparations: dict[tuple[str, str], int] = {}
    for target_transform in transforms:
        depth_limit = target_transform.SPLIT_PREPARATION_DEPTH
        if (
            depth_limit <= 0
            or not target_transform.split_preparation_applicable(state)
            or target_transform.analyze(state)
        ):
            continue
        for depth in range(1, depth_limit + 1):
            path = _split_path_to_transform(
                state, split_transform, target_transform, depth, _representative_splits(first_options)
            )
            if path:
                key = _action_key(state, (split_transform, path[0]))
                priority = _TRANSFORM_PRIORITY.get(type(target_transform).__name__, 0) + 1
                preparations[key] = max(preparations.get(key, priority), priority)
                break
    return preparations


def _split_path_to_transform(
    state: KernelIR,
    split_transform: Transform[Any],
    target_transform: Transform[Any],
    depth: int,
    options: tuple[TransformOption, ...] | None = None,
) -> tuple[TransformOption, ...]:
    """Return one legal split path that makes ``target_transform`` legal."""
    if target_transform.analyze(state):
        return ()
    if depth == 0:
        return ()
    choices = _representative_splits(split_transform.analyze(state)) if options is None else options
    for option in choices:
        next_state = split_transform.apply(state, option)
        if target_transform.split_preparation_ready(next_state):
            return (option,)
        suffix = _split_path_to_transform(next_state, split_transform, target_transform, depth - 1)
        if suffix:
            return (option, *suffix)
    return ()


def _representative_splits(options: list[TransformOption]) -> tuple[TransformOption, ...]:
    """Keep the widest accelerator-sized tensor tile for each split target."""
    selected: dict[tuple[object, object], TransformOption] = {}
    for option in options:
        payload = vars(option)
        factors = payload.get("factors")
        target_axis = payload.get("target_axis")
        if target_axis is None or not isinstance(factors, tuple) or factors[-1] > NKIMatmul.MAX_TILE_SIZE["N"]:
            continue
        key = (payload.get("target_nid"), target_axis)
        current = selected.get(key)
        if current is None or factors[-1] > cast(tuple[int, ...], vars(current)["factors"])[-1]:
            selected[key] = option
    return tuple(selected.values())


def _action_score(
    state: KernelIR,
    action: Action,
    compile_pending: bool,
    failure_location: str | None,
    failure_tensors: frozenset[str],
    tile_preferences: dict[str, int],
    bank_failure: bool,
    failure_tile_size: int | None,
    access_tile_size: int | None,
    split_preparations: dict[tuple[str, str], int],
    independent_shard_extent: int,
) -> tuple[int, int, str]:
    """Rank one action by generic transform semantics and IR structure."""
    transform, option = action
    transform_name, payload = type(transform).__name__, vars(option)
    priority, structural = _TRANSFORM_PRIORITY.get(transform_name, 0), 0
    target_loop_nid, tensor = payload.get("target_loop_nid"), payload.get("tensor")
    list_len, factors = payload.get("list_len"), payload.get("factors")
    stages = payload.get("stages")
    preparation_priority = split_preparations.get(_action_key(state, action))
    if preparation_priority is not None and isinstance(factors, tuple):
        priority, structural = preparation_priority, factors[-1]
    elif transform_name == "TransposePair" and "consumer_nid" in payload:
        priority = -1
    elif transform_name == "ProgramShard":
        axis = payload.get("axis")
        reduction_tensor = payload.get("reduction_tensor")
        if payload.get("programs") == 1 or compile_pending:
            priority = -1
        elif axis in _program_sharded_axes(state):
            priority = _TRANSFORM_PRIORITY["ProgramShard"]
        elif reduction_tensor is not None:
            priority = (
                _TRANSFORM_PRIORITY["ProgramShard"]
                if state.tree.loop(payload["loop_nid"]).extent > independent_shard_extent
                else 57
            )
        else:
            extent = state.tree.loop(payload["loop_nid"]).extent
            axis_blocks = sum(any(v.axis == axis for v in state.tree.block(n).iter_vars) for n in state.tree.blocks())
            kernel_wide = axis_blocks * 2 >= sum(1 for _ in state.tree.blocks())
            eligible = kernel_wide and (
                not (reduction_axes := _program_reduction_axes(state)) or axis in reduction_axes
            )
            short_wide = eligible and payload.get("programs", 1) < extent < 32
            priority = (
                71
                if short_wide and len(state.tree.children(payload["loop_nid"])) > 1
                else 58 + 15 * (extent > 32) if eligible else 35
            )
            if extent == payload.get("programs") and axis_blocks >= 32:
                priority = _TRANSFORM_PRIORITY["ProgramShard"]
        structural = state.tree.loop(payload["loop_nid"]).extent
    elif transform_name == "CommuteBroadcastFactor":
        priority = _TRANSFORM_PRIORITY["OnlineFusion"] + 1
    elif transform_name == "OnlineFusion":
        priority = _TRANSFORM_PRIORITY["RFactor"] - 1
    elif transform_name == "BatchPermutation" and isinstance(payload.get("loop_nid"), int):
        loop_nid = payload["loop_nid"]
        leaf_nid = state.tree.children(loop_nid)[0]
        load_source = any(
            state.tree.isa(producer).op_cls is NKILoad for producer in state.dependency.direct_producers(leaf_nid)
        )
        priority = -1 if compile_pending else (50 if load_source else 75)
        structural = state.tree.loop(loop_nid).extent
    elif transform_name == "SoftwarePipeline" and compile_pending:
        priority = -1
    elif transform_name == "Fuse":
        priority, structural = _fuse_score(state, payload, compile_pending)
    elif transform_name == "Split":
        split_score = _split_score(
            state,
            payload,
            tile_preferences,
            compile_pending,
            failure_location,
            failure_tensors,
            bank_failure,
            failure_tile_size,
            access_tile_size,
        )
        priority, structural = split_score
    elif transform_name == "RFactor":
        split_score = _rfactor_score(state, payload, tile_preferences, compile_pending)
        priority, structural = split_score
    elif transform_name == "Reorder":
        priority, structural = _reorder_score(state, payload, failure_location, failure_tensors, compile_pending)
    if transform_name == "CodeMotion" and isinstance(target_loop_nid, int):
        block_nid = payload.get("block_nid")
        priority, structural = _motion_score(
            state, block_nid, target_loop_nid, failure_location, failure_tensors, compile_pending
        )
    elif transform_name in {"BufferPlacement", "BufferRegionNormalization", "BufferCompaction"} and isinstance(
        tensor, str
    ):
        buffer = state.buffer(tensor)
        if compile_pending and failure_location is None:
            priority = -1
        elif failure_tensors and tensor not in failure_tensors:
            priority = -1
        elif failure_location not in (None, "resource") and buffer.location != failure_location:
            priority = -1
        elif failure_tensors:
            recovery_tile_size = failure_tile_size if failure_tile_size is not None else access_tile_size
            if recovery_tile_size is not None and buffer.per_tile_physical_shape()[-1] > recovery_tile_size:
                priority = 72 if transform_name == "BufferPlacement" else 71
            else:
                priority = 74 if transform_name == "BufferPlacement" else 73
        structural = prod(buffer.shape)
    elif transform_name == "BufferLayout" and isinstance(list_len, int) and isinstance(tensor, str):
        buffer = state.buffer(tensor)
        if failure_tensors and tensor not in failure_tensors:
            priority = -1
        elif failure_location not in (None, "resource") and buffer.location != failure_location:
            priority = -1
        elif buffer.location == "psum":
            priority = 58
        elif compile_pending:
            priority = -1
        else:
            priority = 58
        structural = list_len
    elif transform_name == "SoftwarePipeline" and isinstance(stages, tuple):
        counts = tuple(stages.count(stage) for stage in range(max(stages) + 1))
        priority = -1 if max(stages) == 0 else priority
        structural = len(stages) * 1000 - 100 * (max(counts) - min(counts)) + counts[0]
    return (priority, structural, repr(option))


def _program_reduction_axes(state: KernelIR) -> frozenset[str]:
    """Return axes whose materialized accumulation loop is already sharded."""
    shards = configured_program_shards(state)
    return frozenset(
        iter_var.axis
        for block_nid in state.tree.blocks()
        for iter_var in state.tree.block(block_nid).iter_vars
        if iter_var.role is AxisRole.ACCUMULATION and axis_loop_for_block(state, block_nid, iter_var.axis) in shards
    )


def _program_sharded_axes(state: KernelIR) -> frozenset[str]:
    """Return concrete axes already assigned to logical NeuronCores."""
    shards = configured_program_shards(state)
    return frozenset(
        iter_var.axis
        for block_nid in state.tree.blocks()
        for iter_var in state.tree.block(block_nid).iter_vars
        if axis_loop_for_block(state, block_nid, iter_var.axis) in shards
    )


def _continues_program_shard(state: KernelIR, actions: tuple[Action, ...]) -> bool:
    """Return whether legal actions continue one already active shard axis."""
    active = _program_sharded_axes(state)
    return any(
        type(transform).__name__ == "ProgramShard"
        and getattr(option, "reduction_tensor", None) is None
        and getattr(option, "axis", None) in active
        for transform, option in actions
    )


def _fuse_score(state: KernelIR, payload: dict[str, Any], compile_pending: bool) -> tuple[int, int]:
    """Promote one tensorized fuse when it amortizes underfilled lanes."""
    result = (-1, 0)
    target_nids = payload.get("target_nids")
    target_axis = payload.get("target_axis")
    if compile_pending or not isinstance(target_nids, tuple) or not isinstance(target_axis, str):
        return result
    leaf_nid = target_nids[-1]
    if leaf_nid not in state.tree.graph or not isinstance(state.tree.data(leaf_nid), ISANode):
        return result
    block_nid = next(
        (nid for nid in reversed(state.tree.ancestors(leaf_nid)) if isinstance(state.tree.data(nid), BlockNode)), None
    )
    if block_nid is None:
        return result
    block = state.tree.block(block_nid)
    abstract_axis = next((abstract for abstract, concrete in block.axis_map.items() if concrete == target_axis), None)
    operation = state.tree.isa(leaf_nid)
    cross_sections: list[int] = []
    capacities: list[int] = []
    for slot, region in operation.operand_bindings.items():
        axes = operation.op_cls.OPERAND_AXES[slot]
        if abstract_axis not in axes or any(not isinstance(width, Const) for _lower, width in region.ranges):
            continue
        target_index = axes.index(abstract_axis)
        widths = cast(tuple[Const, ...], tuple(width for _lower, width in region.ranges))
        cross_sections.append(prod(width.value for index, width in enumerate(widths) if index != target_index))
        capacities.extend(
            maximum
            for axis in axes
            if axis != abstract_axis and isinstance((maximum := operation.op_cls.MAX_TILE_SIZE.get(axis)), int)
        )
    if cross_sections and capacities and max(cross_sections) < max(capacities):
        loop_extent = prod(state.tree.loop(nid).extent for nid in target_nids[:-1])
        result = (69, loop_extent * max(capacities) // max(cross_sections))
    return result


def _tile_preferences(state: KernelIR) -> dict[str, int]:
    """Return the strongest explicit or bounded hint for each dimension."""
    preferred: dict[str, int] = {}
    maximums: dict[str, int] = {}
    for block_nid in state.tree.blocks():
        leaves = [nid for nid in state.tree.descendants(block_nid) if isinstance(state.tree.data(nid), ISANode)]
        if len(leaves) != 1:
            continue
        operation = state.tree.isa(leaves[0])
        block = state.tree.block(block_nid)
        for abstract_axis, concrete_axis in block.axis_map.items():
            preference = (
                None if operation.op_cls is NKILoad else operation.op_cls.PREFERRED_TILE_SIZE.get(abstract_axis)
            )
            maximum = operation.op_cls.MAX_TILE_SIZE.get(abstract_axis)
            if isinstance(preference, int) and preference > 1:
                preferred[concrete_axis] = max(preferred.get(concrete_axis, preference), preference)
            elif isinstance(maximum, int) and maximum > 1:
                maximums[concrete_axis] = max(maximums.get(concrete_axis, maximum), maximum)
    return {**maximums, **preferred}


def _split_score(
    state: KernelIR,
    payload: dict[str, Any],
    tile_preferences: dict[str, int],
    compile_pending: bool,
    failure_location: str | None,
    failure_tensors: frozenset[str],
    bank_failure: bool,
    failure_tile_size: int | None,
    access_tile_size: int | None,
) -> tuple[int, int]:
    """Score a split that moves an innermost tile toward a hardware hint."""
    factors = payload.get("factors")
    target_axis = payload.get("target_axis")
    target_nid = payload.get("target_nid")
    result = (-1, 0)
    if isinstance(factors, tuple) and len(factors) == 2 and all(isinstance(value, int) for value in factors):
        preferred = _target_tile_preference(state, target_nid, target_axis, tile_preferences)
        target_tensors = (
            {
                region.tensor
                for nid in state.tree.preorder(target_nid)
                if isinstance(state.tree.data(nid), ISANode)
                for region in state.tree.isa(nid).operand_bindings.values()
            }
            if isinstance(target_nid, int) and target_nid in state.tree.graph
            else set()
        )
        addresses_failure = bool(target_tensors & failure_tensors)
        limit = failure_tile_size if failure_tile_size is not None else access_tile_size
        diagnostic_tile = addresses_failure and factors[-1] <= (limit or 0) < prod(factors)
        bank_local = (
            bank_failure
            and access_tile_size is None
            and isinstance(target_nid, int)
            and target_nid in state.tree.graph
            and isinstance(state.tree.data(target_nid), ISANode)
            and (operation := state.tree.isa(target_nid)).op_cls is NKIMatmul
            and operation.operand_bindings["dst"].tensor in failure_tensors
            and factors[-1] == 128
        )
        producer_aligned = _split_aligns_failure_producer(state, target_nid, factors, failure_tensors)
        neutral_load = compile_pending and failure_location is None and not failure_tensors
        if neutral_load and _small_rmw_load_split(state, target_nid):
            result = (79, factors[-1])
        elif producer_aligned:
            result = (90, factors[-1])
        elif bank_local:
            result = (89, _factor_score(factors, 128))
        elif limit is not None and diagnostic_tile:
            result = (89, _factor_score(factors, limit))
        elif (
            limit is None
            and preferred is not None
            and prod(factors) > preferred
            and (not failure_tensors or addresses_failure)
        ):
            result = (_TRANSFORM_PRIORITY["DecomposeBroadcastSubtract"] - 2, _factor_score(factors, preferred))
        elif target_axis is None and compile_pending and failure_location is None and not failure_tensors:
            result = (34, -abs(factors[0] - factors[1]))
        elif (
            limit is None
            and isinstance(target_axis, str)
            and compile_pending
            and (not failure_tensors or addresses_failure)
            and (failure_location is None or preferred is None or prod(factors) > preferred)
        ):
            priority = 33 if failure_location is None else 34
            result = (priority, -abs(factors[0] - factors[1]))
        elif preferred is None and not compile_pending:
            result = (35, -abs(factors[0] - factors[1]))
    return result


def _small_rmw_load_split(state: KernelIR, target_nid: object) -> bool:
    """Return whether one small load feeds later in-place updates."""
    if not isinstance(target_nid, int) or target_nid not in state.tree.graph:
        return False
    node = state.tree.data(target_nid)
    if not isinstance(node, ISANode) or node.op_cls is not NKILoad:
        return False
    tensor = node.operand_bindings["dst"].tensor
    return _buffer_nbytes(state, tensor) < 1 << 23 and any(
        tensor in state.dependency.info(nid).reads & state.dependency.info(nid).writes
        for nid in state.dependency.touches_by_tensor[tensor]
    )


def _split_aligns_failure_producer(
    state: KernelIR, target_nid: object, factors: tuple[int, int], failure_tensors: frozenset[str]
) -> bool:
    """Return whether a loop factor matches another diagnosed-buffer access."""
    if (
        not failure_tensors
        or not isinstance(target_nid, int)
        or target_nid not in state.tree.graph
        or not isinstance(state.tree.data(target_nid), ForNode)
    ):
        return False
    axes = _loop_axes(state, target_nid, target_nid)
    if len(axes) != 1:
        return False
    axis = next(iter(axes))
    target_leaves = frozenset(
        nid for nid in state.tree.preorder(target_nid) if isinstance(state.tree.data(nid), ISANode)
    )
    tensors = {
        region.tensor
        for leaf_nid in target_leaves
        for region in state.tree.isa(leaf_nid).operand_bindings.values()
        if region.tensor in failure_tensors
    }
    for consumer in target_leaves:
        for producer in state.dependency.direct_producers(consumer):
            shared = state.dependency.info(producer).writes & state.dependency.info(consumer).reads & tensors
            for tensor in shared:
                producer_loop = axis_loop_for_block(state, owning_block(state, producer), axis)
                if producer_loop is not None and state.tree.loop(producer_loop).extent == factors[0]:
                    return True
    for tensor in tensors:
        if state.buffer(tensor).location != "psum":
            continue
        for leaf_nid in state.dependency.touches_by_tensor.get(tensor, ()):
            access_loop = axis_loop_for_block(state, owning_block(state, leaf_nid), axis)
            if (
                leaf_nid not in target_leaves
                and access_loop is not None
                and state.tree.loop(access_loop).extent == factors[0]
            ):
                return True
    return False


def _rfactor_score(
    state: KernelIR, payload: dict[str, Any], tile_preferences: dict[str, int], compile_pending: bool
) -> tuple[int, int]:
    """Score reduction factorization without repeatedly shrinking one axis."""
    factors = payload.get("factors")
    target_axis = payload.get("target_axis")
    target_nid = payload.get("target_loop_nid")
    result = (-1, 0)
    if isinstance(factors, tuple) and all(isinstance(value, int) for value in factors):
        preferred = _target_tile_preference(state, target_nid, target_axis, tile_preferences)
        if preferred is not None and prod(factors) > preferred:
            result = (_TRANSFORM_PRIORITY["DecomposeBroadcastSubtract"] - 1, _factor_score(factors, preferred))
    elif not compile_pending:
        result = (_TRANSFORM_PRIORITY["RFactor"], 0)
    return result


def _target_tile_preference(
    state: KernelIR, target_nid: object, target_axis: object, tile_preferences: dict[str, int]
) -> int | None:
    """Resolve operation-local and producer-local guidance before shared hints."""
    result = tile_preferences.get(target_axis) if isinstance(target_axis, str) else None
    if (
        isinstance(target_nid, int)
        and target_nid in state.tree.graph
        and isinstance(state.tree.data(target_nid), ISANode)
        and isinstance(target_axis, str)
    ):
        block_nid = next(
            (nid for nid in reversed(state.tree.ancestors(target_nid)) if isinstance(state.tree.data(nid), BlockNode)),
            None,
        )
        if block_nid is not None:
            block = state.tree.block(block_nid)
            abstract_axis = next(
                (abstract for abstract, concrete in block.axis_map.items() if concrete == target_axis), None
            )
            operation = state.tree.isa(target_nid)
            if abstract_axis is not None:
                preference = operation.op_cls.PREFERRED_TILE_SIZE.get(abstract_axis)
                maximum = operation.op_cls.MAX_TILE_SIZE.get(abstract_axis)
                if (
                    operation.op_cls is NKILoad
                    and _buffer_nbytes(state, operation.operand_bindings["dst"].tensor) < 1 << 23
                ):
                    result = None
                elif isinstance(preference, int):
                    result = preference
                elif (
                    producer_preference := _producer_tile_preference(state, target_nid, abstract_axis, maximum)
                ) is not None:
                    result = producer_preference
                elif isinstance(maximum, int):
                    result = maximum
    return result


def _producer_tile_preference(state: KernelIR, target_nid: int, abstract_axis: str, maximum: int | None) -> int | None:
    """Return a compatible direct producer's region width on one consumer axis."""
    target = state.tree.isa(target_nid)
    widths: list[int] = []
    for operand in target.op_cls.INPUT_OPERANDS:
        axes = target.op_cls.OPERAND_AXES.get(operand, ())
        region = target.operand_bindings.get(operand)
        if region is None or abstract_axis not in axes:
            continue
        index = axes.index(abstract_axis)
        for producer_nid in state.dependency.direct_producers(target_nid):
            producer = state.tree.isa(producer_nid)
            for producer_region in producer.operand_bindings.values():
                if producer_region.tensor != region.tensor or len(producer_region.ranges) <= index:
                    continue
                width = producer_region.ranges[index][1]
                if isinstance(width, Const) and width.value > 1 and (maximum is None or width.value <= maximum):
                    widths.append(width.value)
    return max(widths) if widths else None


def _factor_score(factors: tuple[int, ...], preferred: int) -> int:
    """Prefer the largest inner factor that does not exceed the hint."""
    inner = factors[-1]
    return -abs(preferred - inner) - (preferred if inner > preferred else 0)


def _failure_location(failures: tuple[str, ...]) -> str | None:
    """Return the on-chip memory named by the latest compiler diagnostic."""
    message = failures[-1].lower() if failures else ""
    if "psum" in message:
        return "psum"
    if "state buffer" in message or "sbuf" in message or "@sb" in message:
        return "sbuf"
    return "resource" if failures else None


def _failure_tile_size(failures: tuple[str, ...]) -> int | None:
    """Return the compiler-reported free-axis tile limit, when present."""
    matches = findall(r"tile size must be <=\s*\d+x(\d+)", failures[-1]) if failures else []
    return int(matches[-1]) if matches else None


def _failure_access_tile_size(state: KernelIR, tensors: frozenset[str]) -> int | None:
    """Return the smallest access width when a failing PSUM is physically wider."""
    if not tensors or any(state.buffer(tensor).location != "psum" for tensor in tensors):
        return None
    widths = {
        width.value
        for tensor in tensors
        for leaf_nid in state.dependency.touches_by_tensor.get(tensor, ())
        for region in state.tree.isa(leaf_nid).operand_bindings.values()
        if region.tensor == tensor and isinstance((width := region.ranges[-1][1]), Const)
    }
    smallest = min(widths) if widths else None
    return (
        smallest
        if smallest is not None
        and any(state.buffer(tensor).per_tile_physical_shape()[-1] > smallest for tensor in tensors)
        else None
    )


def _failure_tensors(state: KernelIR, failures: tuple[str, ...], failure_location: str | None) -> frozenset[str]:
    """Return buffers whose emitted shape matches the latest diagnostic."""
    message = failures[-1] if failures else ""
    lines = findall(r'File "[^"]+",line (\d+)', message)
    source = render(state).splitlines()
    operand = (
        "moving"
        if "moving input tile size" in message
        else "stationary" if "stationary input tile size" in message else "dst"
    )
    source_line = source[int(lines[-1]) - 1] if lines else ""
    if failure_location == "sbuf" and "State buffer allocation failed" in message:
        input_names = findall(r"\b(?:bias|data|data1|data2|moving|src|stationary)=([A-Za-z_]\w*)", source_line)
        matched_inputs = frozenset(
            name for name in input_names if name in state.all_buffers() and state.buffer(name).location == "sbuf"
        )
        if matched_inputs:
            reported_sizes = tuple(map(int, findall(r"total of:\s*(\d+) Bytes", message)))
            threshold = max(max(reported_sizes, default=0) // 4, 1 << 20)
            large_inputs = frozenset(tensor for tensor in matched_inputs if _buffer_nbytes(state, tensor) >= threshold)
            loaded_inputs = frozenset(
                tensor
                for tensor in large_inputs
                for leaf_nid in state.dependency.touches_by_tensor.get(tensor, ())
                if state.tree.isa(leaf_nid).op_cls is NKILoad and tensor in state.dependency.info(leaf_nid).writes
            )
            return loaded_inputs or large_inputs or matched_inputs
    operands = findall(rf"\b{operand}=([A-Za-z_]\w*)", source_line)
    if operands and operands[-1] in state.all_buffers():
        tensor = operands[-1]
        tile_size = _failure_tile_size(failures)
        if (
            operand == "moving"
            and tile_size is not None
            and state.buffer(tensor).per_tile_physical_shape()[-1] <= tile_size
        ):
            destinations = findall(r"\bdst=([A-Za-z_]\w*)", source_line)
            if destinations and destinations[-1] in state.all_buffers():
                tensor = destinations[-1]
        if failure_location == "resource" or state.buffer(tensor).location == failure_location:
            return frozenset({tensor})
    shapes = findall(r"\[(\d+),\s*(\d+),\s*(\d+)\]", message)
    shape = tuple(map(int, shapes[-1])) if shapes else ()
    matches = frozenset(
        name
        for name, buffer in state.all_buffers().items()
        if buffer.location == failure_location and buffer.per_tile_physical_shape() == shape
    )
    if matches or failure_location != "psum" or "multiple psum banks" not in message.lower():
        return matches
    return frozenset(
        operation.operand_bindings["dst"].tensor
        for nid in state.tree.preorder()
        if isinstance(state.tree.data(nid), ISANode)
        and (operation := state.tree.isa(nid)).op_cls is NKIMatmul
        and isinstance((width := operation.operand_bindings["dst"].ranges[-1][1]), Const)
        and state.buffer(operation.operand_bindings["dst"].tensor).per_tile_physical_shape()[-1] > width.value
    )


def _buffer_nbytes(state: KernelIR, tensor: str) -> int:
    """Return one buffer's physical allocation size in bytes."""
    widths = {"float8_e4m3": 1, "float16": 2, "bfloat16": 2, "float32": 4, "int32": 4, "uint32": 4}
    return prod((buffer := state.buffer(tensor)).physical_shape()) * widths[buffer.physical_dtype()]


def _reorder_score(
    state: KernelIR,
    payload: dict[str, Any],
    failure_location: str | None,
    failure_tensors: frozenset[str],
    compile_pending: bool,
) -> tuple[int, int]:
    """Promote a parallel tile loop outside accumulation or a smaller outer loop."""
    outer_nid = payload.get("outer_nid")
    inner_nid = payload.get("inner_nid")
    result = (-1, 0)
    if (
        (failure_location is not None or not compile_pending)
        and isinstance(outer_nid, int)
        and isinstance(inner_nid, int)
    ):
        outer, inner = state.tree.loop(outer_nid), state.tree.loop(inner_nid)
        tensors = {
            region.tensor
            for nid in state.tree.preorder(inner_nid)
            if isinstance(state.tree.data(nid), ISANode)
            for region in state.tree.isa(nid).operand_bindings.values()
        }
        locations = {state.buffer(tensor).location for tensor in tensors}
        addresses_failure = (
            bool(tensors & failure_tensors)
            if failure_tensors
            else failure_location is None or failure_location in locations
        )
        outer_roles = _loop_roles(state, outer_nid, inner_nid)
        inner_roles = _loop_roles(state, inner_nid, inner_nid)
        cross_axis = _loop_axes(state, outer_nid, inner_nid).isdisjoint(_loop_axes(state, inner_nid, inner_nid))
        psum_failure = failure_location == "psum" or any(
            state.buffer(tensor).location == "psum" for tensor in failure_tensors
        )
        writes_psum = any(
            state.tree.isa(nid).op_cls is NKIMatmul
            and state.buffer(state.tree.isa(nid).operand_bindings["dst"].tensor).location == "psum"
            for nid in state.tree.preorder(inner_nid)
            if isinstance(state.tree.data(nid), ISANode)
        )
        keeps_input_streamed = (
            AxisRole.ACCUMULATION in outer_roles
            and AxisRole.PARALLEL in inner_roles
            and _reorder_streams_failure_input(state, inner_nid, outer_nid, failure_tensors)
        )
        accumulator_locality = (
            (psum_failure or writes_psum)
            and not keeps_input_streamed
            and AxisRole.ACCUMULATION in outer_roles
            and AxisRole.PARALLEL in inner_roles
        )
        aligns_accesses = _reorder_aligns_failure_accesses(state, outer_nid, inner_nid, failure_tensors)
        streams_input = (
            failure_location == "sbuf"
            and AxisRole.PARALLEL in outer_roles
            and AxisRole.ACCUMULATION in inner_roles
            and _reorder_streams_failure_input(state, outer_nid, inner_nid, failure_tensors)
        )
        orders_parallel_work = cross_axis and inner.extent > outer.extent and AxisRole.ACCUMULATION not in inner_roles
        if addresses_failure and (accumulator_locality or aligns_accesses or streams_input or orders_parallel_work):
            structural = inner.extent - outer.extent
            if accumulator_locality:
                structural += 1 << 30
            elif aligns_accesses:
                structural += 1 << 29
            elif streams_input:
                structural += 1 << 28
            result = (71, structural)
    return result


def _reorder_streams_failure_input(
    state: KernelIR, outer_nid: int, inner_nid: int, failure_tensors: frozenset[str]
) -> bool:
    """Return whether interchange reuses a diagnosed input across the outer loop."""
    outer_var, inner_var = state.tree.loop(outer_nid).loop_var, state.tree.loop(inner_nid).loop_var
    for nid in state.tree.preorder(inner_nid):
        if not isinstance(state.tree.data(nid), ISANode):
            continue
        for slot in (operation := state.tree.isa(nid)).op_cls.INPUT_OPERANDS:
            region = operation.operand_bindings.get(slot)
            if region is None or (
                region.tensor not in failure_tensors
                and (slot, state.buffer(region.tensor).location, _buffer_nbytes(state, region.tensor) >= 1 << 23)
                != ("moving", "sbuf", True)
            ):
                continue
            lower_vars = [to_affine(lower) for lower, _width in region.ranges]
            if all(outer_var not in affine for affine in lower_vars) and any(
                inner_var in affine for affine in lower_vars
            ):
                return True
    return False


def _reorder_aligns_failure_accesses(
    state: KernelIR, outer_nid: int, inner_nid: int, failure_tensors: frozenset[str]
) -> bool:
    """Return whether swapping two loops matches another failing-buffer access."""
    outer_axes = _loop_axes(state, outer_nid, inner_nid)
    inner_axes = _loop_axes(state, inner_nid, inner_nid)
    for tensor in failure_tensors:
        for leaf_nid in state.dependency.touches_by_tensor.get(tensor, ()):
            block_nid = owning_block(state, leaf_nid)
            ancestors = state.tree.ancestors(leaf_nid)
            for inner_axis in inner_axes:
                target_outer = axis_loop_for_block(state, block_nid, inner_axis)
                for outer_axis in outer_axes:
                    target_inner = axis_loop_for_block(state, block_nid, outer_axis)
                    if (
                        target_outer is not None
                        and target_inner is not None
                        and ancestors.index(target_outer) < ancestors.index(target_inner)
                    ):
                        return True
    return False


def _loop_roles(state: KernelIR, loop_nid: int, subtree_nid: int) -> set[AxisRole]:
    """Return operation-local roles bound to one loop in a subtree."""
    loop_var = state.tree.loop(loop_nid).loop_var
    block_nids = {
        block_nid
        for leaf_nid in state.tree.preorder(subtree_nid)
        if isinstance(state.tree.data(leaf_nid), ISANode)
        for block_nid in reversed(state.tree.ancestors(leaf_nid))
        if isinstance(state.tree.data(block_nid), BlockNode)
    }
    roles = {
        iter_var.role
        for block_nid in block_nids
        for iter_var, value in zip(
            state.tree.block(block_nid).iter_vars, state.tree.block(block_nid).iter_values, strict=True
        )
        if loop_var in to_affine(value)
    }
    return roles


def _loop_axes(state: KernelIR, loop_nid: int, subtree_nid: int) -> set[str]:
    """Return concrete operation axes bound to one loop in a subtree."""
    loop_var = state.tree.loop(loop_nid).loop_var
    return {
        iter_var.axis
        for leaf_nid in state.tree.preorder(subtree_nid)
        if isinstance(state.tree.data(leaf_nid), ISANode)
        for iter_var, value in zip(
            state.tree.block(owning_block(state, leaf_nid)).iter_vars,
            state.tree.block(owning_block(state, leaf_nid)).iter_values,
            strict=True,
        )
        if loop_var in to_affine(value)
    }


def _motion_score(
    state: KernelIR,
    block_nid: object,
    target_loop_nid: int,
    failure_location: str | None,
    failure_tensors: frozenset[str],
    compile_pending: bool,
) -> tuple[int, int]:
    """Rank locality motion when it addresses the active resource diagnostic."""
    result = (-1, 0)
    if isinstance(block_nid, int) and block_nid in state.tree.graph:
        if compile_pending and state.tree.parent(block_nid) == target_loop_nid:
            return result
        block = state.tree.block(block_nid)
        block_axes = {iter_var.axis for iter_var in block.iter_vars}
        existing_loops = {nid for nid in state.tree.ancestors(block_nid) if isinstance(state.tree.data(nid), ForNode)}
        target_path = (*state.tree.ancestors(target_loop_nid), target_loop_nid)
        if any(
            not _loop_axes(state, nid, target_loop_nid) & block_axes
            for nid in target_path
            if isinstance(state.tree.data(nid), ForNode) and nid not in existing_loops
        ):
            return result
        target_loop_var = state.tree.loop(target_loop_nid).loop_var
        target_leaf_nids = [
            nid for nid in state.tree.preorder(target_loop_nid) if isinstance(state.tree.data(nid), ISANode)
        ]
        target_axes = {
            iter_var.axis
            for leaf_nid in target_leaf_nids
            for iter_var, value in zip(
                state.tree.block(owning_block(state, leaf_nid)).iter_vars,
                state.tree.block(owning_block(state, leaf_nid)).iter_values,
                strict=True,
            )
            if target_loop_var in to_affine(value)
        }
        if not target_axes & block_axes:
            return result
        tensors = {region.tensor for region in (*block.reads, *block.writes)}
        locations = {state.buffer(tensor).location for tensor in tensors}
        target_accesses = 0
        moved_leaf_nids = {
            nid
            for nid in state.tree.preorder(block_nid)
            if isinstance(state.tree.data(nid), ISANode) and owning_block(state, nid) == block_nid
        }
        target_leaf_set = set(target_leaf_nids) - moved_leaf_nids
        producer_targets = target_leaf_set & {
            producer for leaf_nid in moved_leaf_nids for producer in state.dependency.direct_producers(leaf_nid)
        }
        consumer_targets = target_leaf_set & {
            consumer for leaf_nid in moved_leaf_nids for consumer in state.dependency.direct_consumers(leaf_nid)
        }
        co_producer_targets = target_leaf_set & {
            producer
            for leaf_nid in moved_leaf_nids
            for consumer in state.dependency.direct_consumers(leaf_nid)
            for producer in state.dependency.direct_producers(consumer)
        }
        if failure_tensors:
            moved = frozenset(state.tree.preorder(block_nid))
            target_leaves = [state.tree.isa(nid) for nid in target_leaf_nids if nid not in moved]
            target_tensors = {region.tensor for leaf in target_leaves for region in leaf.operand_bindings.values()}
            addresses_failure = bool(
                tensors & failure_tensors
                and target_tensors & failure_tensors
                or (tensors | target_tensors) & failure_tensors
                and (producer_targets or consumer_targets or co_producer_targets)
            )
            target_accesses = sum(
                any(region.tensor in failure_tensors for region in leaf.operand_bindings.values())
                for leaf in target_leaves
            )
        else:
            addresses_failure = failure_location in locations
        if not compile_pending or addresses_failure:
            volume = sum(prod(state.buffer(tensor).shape) for tensor in tensors)
            topology = state.dependency._topology()[0]
            upstream_order = len(topology) - min(topology[nid] for nid in moved_leaf_nids)
            free_axis_target = any(
                target_loop_var in to_affine(region.ranges[-1][0])
                for nid in target_leaf_set
                for region in state.tree.isa(nid).operand_bindings.values()
                if region.tensor in failure_tensors
            )
            structural = (
                (len(consumer_targets & co_producer_targets) << 61)
                + (len(producer_targets) << 60)
                + (len(consumer_targets) << 55)
                + (len(co_producer_targets) << 54 if not producer_targets and not consumer_targets else 0)
                + (upstream_order << 50)
                + (target_accesses << 40)
                + volume * state.tree.loop(target_loop_nid).extent
            )
            direct = producer_targets or consumer_targets or not failure_tensors
            priority = (
                72
                if direct and compile_pending and failure_location == "sbuf"
                else 70 if direct else 69 if free_axis_target else 67
            )
            result = (priority, structural)
    return result


@dataclass(frozen=True)
class SearchResult:
    """Summary of one completed refinement run."""

    best_latency_ms: float
    transforms_applied: int
    evaluations_run: int
    finish_reason: str
