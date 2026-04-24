"""Per-op emission Placement in a group's render slot sequence.

A group's ``_render_group`` linearizes into a pair-interleaved
sequence of slots::

    before[0], for block_0: before[1], for ltile_0: before[2],
    for block_1: before[3], for ltile_1: before[4], ...,
    for block_{N-1}: before[2N-1], for ltile_{N-1}: before[2N],
    after[2N-1], after[2N-2], ..., after[0]

Every ``(phase, depth)`` is a slot. The slot coordinate for dim
at ``pos`` is ``block_depth(pos) = 2*pos`` (before the block loop
opens) and ``ltile_depth(pos) = 2*pos + 1`` (before the ltile
loop opens). Body emits at depth ``2*N``. Source order:
``before[k1] < before[k2] < after[k2-1] < after[k1-1]`` for
``k1 <= k2``; linearize via ``source_position(p) = p.depth if
p.phase == "before" else 4*N - p.depth``.
"""

from dataclasses import dataclass
from typing import Literal

from nkigym.kernel_ir.ir import KernelIR
from nkigym.ops.base import NKIOp
from nkigym.ops.dma import NKIDMATranspose, NKILoad, NKIStore


def block_depth(pos: int) -> int:
    """Slot depth at which the block loop for dim ``pos`` opens."""
    return 2 * pos


def ltile_depth(pos: int) -> int:
    """Slot depth at which the ltile loop for dim ``pos`` opens."""
    return 2 * pos + 1


def body_depth(n: int) -> int:
    """Innermost body slot depth for a group of ``n`` ordered dims."""
    return 2 * n


def compute_staged_set(ir: KernelIR) -> set[str]:
    """PSUM-produced tensors that require an SBUF staging buffer."""
    result: set[str] = set()
    producer_loc = _producer_isa_loc_map(ir)
    for group in ir.groups:
        for op in group.ops:
            input_locs = type(op).INPUT_LOCS
            for role, tname in ir.op_inputs.get(op, {}).items():
                if not ir.has_tensor(tname):
                    continue
                if producer_loc.get(tname) != "psum":
                    continue
                if input_locs.get(role) == "sbuf":
                    result.add(tname)
    if producer_loc.get(ir.return_name) == "psum":
        result.add(ir.return_name)
    return result


def _producer_isa_loc_map(ir: KernelIR) -> dict[str, str]:
    """Return ``{tensor_name: ISA_LOC of its producer}``."""
    result: dict[str, str] = {}
    for group in ir.groups:
        for op in group.ops:
            loc = type(op).ISA_LOC
            for name in ir.op_outputs.get(op, []):
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


def block_trip(ir: KernelIR, dim_id: str) -> int:
    """Block-loop trip count for ``dim_id``."""
    di = ir.dimensions[dim_id]
    return di.dim_size // (ir.ltiles_per_block.get(dim_id, 1) * di.logical_tile_size)


def ltile_trip(ir: KernelIR, dim_id: str) -> int:
    """Ltile-loop trip count for ``dim_id``."""
    return ir.ltiles_per_block.get(dim_id, 1)


def material_blocking_dims(ir: KernelIR, op: NKIOp, dim_order: list[str]) -> set[str]:
    """Blocking dims whose loops have trip > 1."""
    blocking = ir.op_blocking_dims.get(op, set()) & set(dim_order)
    return {d for d in blocking if block_trip(ir, d) > 1 or ltile_trip(ir, d) > 1}


def op_depth_floor(ir: KernelIR, op: NKIOp, group_idx: int) -> int:
    """Smallest ``slot.depth`` at which this op can legally emit.

    For DMA ops (Load/Store/DMATranspose) the floor is derived
    from the SBUF tensor's per-dim tier placement: ``full`` dims
    add no constraint (loop need not be open), ``per_block`` dims
    force emission inside the block loop (depth > ``2*pos``),
    ``per_tile`` dims force emission inside the ltile loop (depth
    > ``2*pos+1``). For all other ops the floor is the body slot.
    """
    dim_order = ir.groups[group_idx].dim_order
    positions = _op_tensor_dim_positions(ir, op, dim_order)
    if isinstance(op, (NKILoad, NKIStore, NKIDMATranspose)):
        floor = _dma_depth_floor(ir, op, group_idx)
    else:
        floor = body_depth(len(dim_order)) if positions else 0
    return floor


def _dma_depth_floor(ir: KernelIR, op: NKIOp, group_idx: int) -> int:
    """DMA-specific floor: alloc-depth of the op's SBUF tensor.

    The DMA must emit at or below the buffer's alloc depth — any
    earlier would reference an undeclared buffer. Alloc depth is
    driven by the buffer's effective ``BufferPlacement``.
    """
    from nkigym.kernel_ir.placement_semantics import alloc_depth, buffer_dim_positions, effective_placement

    group = ir.groups[group_idx]
    sbuf_tensor = _dma_sbuf_tensor(ir, op)
    if not sbuf_tensor or not ir.has_tensor(sbuf_tensor):
        return 0
    tinfo = ir.tensor_info(sbuf_tensor)
    placement = effective_placement(ir, group_idx, sbuf_tensor)
    positions = buffer_dim_positions(tinfo.dim_ids, group.dim_order)
    return alloc_depth(placement, positions)


def _dma_sbuf_tensor(ir: KernelIR, op: NKIOp) -> str:
    """Return the SBUF-side tensor name for a DMA op (Load/DMATranspose: output; Store: input)."""
    if isinstance(op, NKIStore):
        name = ir.op_inputs.get(op, {}).get("data", "")
    else:
        outputs = ir.op_outputs.get(op, [])
        name = outputs[0] if outputs else ""
    return name


def _op_tensor_dim_positions(ir: KernelIR, op: NKIOp, dim_order: list[str]) -> list[int]:
    """Return positions in ``dim_order`` of every dim any of the op's tensors carries."""
    touched: set[str] = set()
    for name in list(ir.op_inputs.get(op, {}).values()) + list(ir.op_outputs.get(op, [])):
        if ir.has_tensor(name):
            touched.update(ir.tensor_info(name).dim_ids)
    return [dim_order.index(d) for d in touched if d in dim_order]


def _op_input_tensor_names(ir: KernelIR, op: NKIOp) -> list[str]:
    """Every tensor-valued input — positional inputs plus tensor-valued kwargs."""
    result = list(ir.op_inputs.get(op, {}).values())
    tensors_set = set(ir.logical_tensors) | set(ir.physical_buffers)
    for _name, expr in ir.op_kwargs.get(op, {}).items():
        if expr in tensors_set and expr not in result:
            result.append(expr)
    return result


def _producer_op_of(ir: KernelIR, tensor_name: str) -> NKIOp | None:
    """Return the op producing ``tensor_name`` or None."""
    result: NKIOp | None = None
    for group in ir.groups:
        for op in group.ops:
            if tensor_name in ir.op_outputs.get(op, []):
                result = op
                break
        if result is not None:
            break
    return result


def producer_stage_placement(
    ir: KernelIR,
    producer_op: NKIOp,
    group_idx: int,
    op_to_group: dict[int, int],
    staged: set[str],
    memo: dict[int, Placement],
) -> Placement:
    """Slot at which the producer's output becomes valid for a consumer to read."""
    op_cls = type(producer_op)
    dim_order = ir.groups[group_idx].dim_order
    blocking_barrier: Placement | None = None
    if op_cls.ISA_LOC == "psum":
        material = material_blocking_dims(ir, producer_op, dim_order)
        produced = ir.op_outputs.get(producer_op, [])
        if material and any(t in staged for t in produced):
            blocking_barrier = Placement("after", block_depth(min(dim_order.index(d) for d in material)))
    if blocking_barrier is not None:
        result = blocking_barrier
    else:
        result = op_emission_placement(ir, producer_op, group_idx, op_to_group, staged, memo)
    memo[id(producer_op)] = result
    return result


def op_emission_placement(
    ir: KernelIR,
    op: NKIOp,
    group_idx: int,
    op_to_group: dict[int, int],
    staged: set[str],
    memo: dict[int, Placement] | None = None,
) -> Placement:
    """Narrowest legal Placement for ``op`` in ``group_idx``."""
    memo = memo if memo is not None else {}
    if id(op) not in memo:
        memo[id(op)] = _compute_placement(ir, op, group_idx, op_to_group, staged, memo)
    return memo[id(op)]


def _compute_placement(
    ir: KernelIR, op: NKIOp, group_idx: int, op_to_group: dict[int, int], staged: set[str], memo: dict[int, Placement]
) -> Placement:
    """Body of ``op_emission_placement``."""
    dim_order = ir.groups[group_idx].dim_order
    n = len(dim_order)
    floor = op_depth_floor(ir, op, group_idx)
    barriers: list[Placement] = []
    for tname in _op_input_tensor_names(ir, op):
        producer = _producer_op_of(ir, tname)
        if producer is None or producer is op or op_to_group.get(id(producer)) != group_idx:
            continue
        barriers.append(producer_stage_placement(ir, producer, group_idx, op_to_group, staged, memo))
    candidates = [Placement("before", d) for d in range(floor, body_depth(n) + 1)] + [
        Placement("after", d) for d in range(body_depth(n) - 1, -1, -1)
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
