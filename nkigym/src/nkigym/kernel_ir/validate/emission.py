"""Per-op emission Placement in a group's render slot sequence.

A group's ``_render_group`` linearizes into a sequence of slots::

    before[0], for block_0: before[1], ..., for block_{N-1}: before[N],
    for ltile_0: ..., for ltile_{N-1}: before[2N], after[2N-1], ...,
    after[N], after[N-1], ..., after[0]

Every ``(phase, depth)`` is a slot. Source order:
``before[k1] < before[k2] < after[k2-1] < after[k1-1]`` for
``k1 <= k2``; linearize via ``source_position(p) = p.depth if
p.phase == "before" else 4*N - p.depth``.
"""

from dataclasses import dataclass
from typing import Literal

from nkigym.kernel_ir.context.context import KernelContext
from nkigym.kernel_ir.graph.graph import KernelGraph
from nkigym.ops.base import NKIOp


def compute_staged_set(context: KernelContext, graph: KernelGraph) -> set[str]:
    """PSUM-produced tensors that require an SBUF staging buffer."""
    result: set[str] = set()
    producer_loc = _producer_isa_loc_map(context, graph)
    for group in graph.groups:
        for op in group.ops:
            input_locs = type(op).INPUT_LOCS
            for role, tname in context.op_inputs.get(op, {}).items():
                if tname not in context.logical_tensors:
                    continue
                if producer_loc.get(tname) != "psum":
                    continue
                if input_locs.get(role) == "sbuf":
                    result.add(tname)
    if producer_loc.get(context.return_name) == "psum":
        result.add(context.return_name)
    return result


def _producer_isa_loc_map(context: KernelContext, graph: KernelGraph) -> dict[str, str]:
    """Return ``{tensor_name: ISA_LOC of its producer}``."""
    result: dict[str, str] = {}
    for group in graph.groups:
        for op in group.ops:
            loc = type(op).ISA_LOC
            for name in context.op_outputs.get(op, []):
                result[name] = loc
    return result


@dataclass(frozen=True)
class Placement:
    """A render slot = (phase, depth)."""

    phase: Literal["before", "after"]
    depth: int

    def source_position(self, n: int) -> int:
        """Total order: ``before[k] -> k``; ``after[k] -> 4n - k``."""
        return self.depth if self.phase == "before" else 4 * n - self.depth

    def loop_open(self, loop_depth: int) -> bool:
        """True iff the loop at ``loop_depth`` is open at this slot."""
        return loop_depth < self.depth


def block_trip(context: KernelContext, dim_id: str) -> int:
    """Block-loop trip count for ``dim_id``."""
    di = context.dimensions[dim_id]
    return di.dim_size // (context.ltiles_per_block.get(dim_id, 1) * di.logical_tile_size)


def ltile_trip(context: KernelContext, dim_id: str) -> int:
    """Ltile-loop trip count for ``dim_id``."""
    return context.ltiles_per_block.get(dim_id, 1)


def material_blocking_dims(context: KernelContext, op: NKIOp, dim_order: list[str]) -> set[str]:
    """Blocking dims whose loops have trip > 1."""
    blocking = context.op_blocking_dims.get(op, set()) & set(dim_order)
    return {d for d in blocking if block_trip(context, d) > 1 or ltile_trip(context, d) > 1}


def op_depth_floor(context: KernelContext, graph: KernelGraph, op: NKIOp, group_idx: int) -> int:
    """Smallest ``slot.depth`` at which this op can legally emit."""
    dim_order = graph.groups[group_idx].dim_order
    n = len(dim_order)
    positions = _op_tensor_dim_positions(context, op, dim_order)
    floor = n + max(positions) + 1 if positions else 0
    return floor


def _op_tensor_dim_positions(context: KernelContext, op: NKIOp, dim_order: list[str]) -> list[int]:
    """Return positions in ``dim_order`` of every dim any of the op's tensors carries."""
    touched: set[str] = set()
    for name in list(context.op_inputs.get(op, {}).values()) + list(context.op_outputs.get(op, [])):
        tinfo = context.logical_tensors.get(name)
        if tinfo is not None:
            touched.update(tinfo.dim_ids)
    return [dim_order.index(d) for d in touched if d in dim_order]


def _op_input_tensor_names(context: KernelContext, op: NKIOp) -> list[str]:
    """Every tensor-valued input — positional inputs plus tensor-valued kwargs."""
    result = list(context.op_inputs.get(op, {}).values())
    tensors_set = set(context.logical_tensors)
    for _name, expr in context.op_kwargs.get(op, {}).items():
        if expr in tensors_set and expr not in result:
            result.append(expr)
    return result


def _producer_op_of(context: KernelContext, graph: KernelGraph, tensor_name: str) -> NKIOp | None:
    """Return the op producing ``tensor_name`` or None."""
    result: NKIOp | None = None
    for group in graph.groups:
        for op in group.ops:
            if tensor_name in context.op_outputs.get(op, []):
                result = op
                break
        if result is not None:
            break
    return result


def producer_stage_placement(
    context: KernelContext,
    graph: KernelGraph,
    producer_op: NKIOp,
    group_idx: int,
    op_to_group: dict[int, int],
    staged: set[str],
    memo: dict[int, Placement],
) -> Placement:
    """Slot at which the producer's output becomes valid for a consumer to read."""
    op_cls = type(producer_op)
    dim_order = graph.groups[group_idx].dim_order
    blocking_barrier: Placement | None = None
    if op_cls.ISA_LOC == "psum":
        material = material_blocking_dims(context, producer_op, dim_order)
        produced = context.op_outputs.get(producer_op, [])
        if material and any(t in staged for t in produced):
            blocking_barrier = Placement("after", min(dim_order.index(d) for d in material))
    if blocking_barrier is not None:
        result = blocking_barrier
    else:
        result = op_emission_placement(context, graph, producer_op, group_idx, op_to_group, staged, memo)
    memo[id(producer_op)] = result
    return result


def op_emission_placement(
    context: KernelContext,
    graph: KernelGraph,
    op: NKIOp,
    group_idx: int,
    op_to_group: dict[int, int],
    staged: set[str],
    memo: dict[int, Placement] | None = None,
) -> Placement:
    """Narrowest legal Placement for ``op`` in ``group_idx``."""
    memo = memo if memo is not None else {}
    if id(op) not in memo:
        memo[id(op)] = _compute_placement(context, graph, op, group_idx, op_to_group, staged, memo)
    return memo[id(op)]


def _compute_placement(
    context: KernelContext,
    graph: KernelGraph,
    op: NKIOp,
    group_idx: int,
    op_to_group: dict[int, int],
    staged: set[str],
    memo: dict[int, Placement],
) -> Placement:
    """Body of ``op_emission_placement``."""
    dim_order = graph.groups[group_idx].dim_order
    n = len(dim_order)
    floor = op_depth_floor(context, graph, op, group_idx)
    barriers: list[Placement] = []
    for tname in _op_input_tensor_names(context, op):
        producer = _producer_op_of(context, graph, tname)
        if producer is None or producer is op or op_to_group.get(id(producer)) != group_idx:
            continue
        barriers.append(producer_stage_placement(context, graph, producer, group_idx, op_to_group, staged, memo))
    candidates = [Placement("before", d) for d in range(floor, 2 * n + 1)] + [
        Placement("after", d) for d in range(2 * n - 1, -1, -1)
    ]
    chosen: Placement | None = None
    for cand in candidates:
        cand_sp = cand.source_position(n)
        if all(cand_sp >= b.source_position(n) for b in barriers):
            chosen = cand
            break
    if chosen is None:
        raise ValueError(f"No legal emission Placement for op {op!r} in group {group_idx}: floor={floor}")
    memo[id(op)] = chosen
    return chosen
