"""Per-op emission Placement in a group's render slot sequence.

A group's `_render_group` linearizes into a sequence of slots::

    before[0], for block_0: before[1], ..., for block_{N-1}: before[N],
    for ltile_0: ..., for ltile_{N-1}: before[2N], after[2N-1], ...,
    after[N], after[N-1], ..., after[0]

Every `(phase, depth)` is a slot. Loops at depths ``0..depth-1`` are open at
both ``before[depth]`` and ``after[depth]``. Source order: ``before[k1] <
before[k2] < after[k2-1] < after[k1-1]`` for ``k1 <= k2``; linearize via
``source_position(p) = p.depth if p.phase == "before" else 4*N - p.depth``.

Per-op Placement replaces the old hardcoded ``2 * N`` emission depth. An op's
narrowest legal slot is bounded below by its material-loop scope and above by
every intra-group producer's stage slot.
"""

from dataclasses import dataclass
from typing import Any, Literal

KernelIR = Any


def compute_staged_set(ir: KernelIR) -> set[str]:
    """PSUM-produced tensors that require an SBUF staging buffer.

    Mirrors ``codegen.buffers.find_psum_tensors_needing_sbuf`` but
    lives here so ``validate._check_emission_feasibility`` can call
    it without triggering the codegen-package import cycle.
    """
    da = ir.dim_analysis
    graph = ir.op_graph
    result: set[str] = set()
    for consumer_idx, (inputs, _outputs) in enumerate(graph.op_tensors):
        input_locs = graph.op_classes[consumer_idx].INPUT_LOCS
        for role, tensor_name in inputs.items():
            if tensor_name not in da.tensors:
                continue
            if graph.producer_isa_loc(tensor_name) != "psum":
                continue
            if input_locs.get(role) == "sbuf":
                result.add(tensor_name)
    if graph.producer_isa_loc(da.return_name) == "psum":
        result.add(da.return_name)
    return result


@dataclass(frozen=True)
class Placement:
    """A render slot = (phase, depth). `before[d]` and `after[d]` share the same open-loop set (loops at depths 0..d-1)."""

    phase: Literal["before", "after"]
    depth: int

    def source_position(self, n: int) -> int:
        """Total order: before[k] -> k; after[k] -> 4n - k. Lower = earlier in source."""
        return self.depth if self.phase == "before" else 4 * n - self.depth

    def loop_open(self, loop_depth: int) -> bool:
        """True iff the loop at ``loop_depth`` is open at this slot."""
        return loop_depth < self.depth


def block_trip(ir: KernelIR, dim_id: str) -> int:
    """Block-loop trip count for ``dim_id``."""
    di = ir.dim_analysis.dims[dim_id]
    return di.dim_size // (ir.ltiles_per_block.get(dim_id, 1) * di.logical_tile_size)


def ltile_trip(ir: KernelIR, dim_id: str) -> int:
    """Ltile-loop trip count for ``dim_id``."""
    return ir.ltiles_per_block.get(dim_id, 1)


def material_blocking_dims(ir: KernelIR, op_idx: int, dim_order: list[str]) -> set[str]:
    """Blocking dims of the op that create a real barrier (some loop has trip > 1).

    Trip-1 blocking loops are no-ops — stage-before-loop and stage-after-loop
    are the same position. Only dims with at least one material loop (block
    or ltile trip > 1) constitute a barrier that requires consumer hoisting.
    """
    blocking = ir.dim_analysis.op_blocking_dims(op_idx) & set(dim_order)
    return {d for d in blocking if block_trip(ir, d) > 1 or ltile_trip(ir, d) > 1}


def op_depth_floor(ir: KernelIR, op_idx: int, group_idx: int) -> int:
    """Smallest ``slot.depth`` at which this op can legally emit.

    Derived from the op's tensor dims: an op emits at the narrowest
    scope in which every dim it touches has its block- AND ltile-
    loop open. For each dim the op touches, the ltile loop is the
    later-opening one (depth ``N + pos + 1``); the floor is the
    max over all such ``N + pos + 1`` values. Ops whose tensors
    touch every dim in the group thus land at ``2 * N`` (innermost
    body) — preserving existing behavior for pre-rewrite ops.
    Rewrite-inserted ops whose tensors skip some dims (e.g. a
    running buffer that carries only the partition axis) get a
    shallower floor, emitting outside the skipped dims' loops.
    """
    dim_order = ir.fusion_groups[group_idx].dim_order
    n = len(dim_order)
    relevant_positions = _op_tensor_dim_positions(ir, op_idx, dim_order)
    floor = n + max(relevant_positions) + 1 if relevant_positions else 0
    return floor


def _op_tensor_dim_positions(ir: KernelIR, op_idx: int, dim_order: list[str]) -> list[int]:
    """Return the positions in ``dim_order`` of every dim any of the op's tensors carries."""
    da = ir.dim_analysis
    graph = ir.op_graph
    touched_dims: set[str] = set()
    for tensor_name in graph.op_tensor_names(op_idx):
        tinfo = da.tensors.get(tensor_name)
        if tinfo is not None:
            touched_dims.update(tinfo.dim_ids)
    return [dim_order.index(d) for d in touched_dims if d in dim_order]


def _op_input_tensor_names(ir: KernelIR, op_idx: int) -> list[str]:
    """Every tensor-valued input to ``op_idx`` — positional inputs plus tensor-valued scalar kwargs.

    A consumer that reads multiple buffers (e.g., ``scalar_tensor_tensor``
    with ``data``, ``operand0``, ``operand1`` all tensor-valued) depends on
    every producer; the barrier is the union.
    """
    graph = ir.op_graph
    tensors_set = set(ir.dim_analysis.tensors)
    result = list(graph.op_tensors[op_idx][0].values())
    for _name, expr in graph.op_all_kwargs[op_idx].items():
        if expr in tensors_set and expr not in result:
            result.append(expr)
    return result


def producer_stage_placement(
    ir: KernelIR,
    producer_op_idx: int,
    group_idx: int,
    op_to_group: dict[int, int],
    staged: set[str],
    memo: dict[int, Placement],
) -> Placement:
    """Slot at which the producer's output becomes valid for a consumer to read.

    PSUM producer with material blocking and staged output: ``after`` the
    outermost material blocking dim's block-loop closes. PSUM producer with
    only non-material blocking (or unstaged), and non-PSUM producers: the
    producer's own emission slot (stage inlines there, or direct read).
    """
    op_cls = ir.op_graph.op_classes[producer_op_idx]
    dim_order = ir.fusion_groups[group_idx].dim_order
    blocking_barrier: Placement | None = None
    if op_cls.ISA_LOC == "psum":
        material = material_blocking_dims(ir, producer_op_idx, dim_order)
        produced = ir.op_graph.op_tensors[producer_op_idx][1]
        if material and any(t in staged for t in produced):
            blocking_barrier = Placement("after", min(dim_order.index(d) for d in material))
    result = (
        blocking_barrier
        if blocking_barrier is not None
        else op_emission_placement(ir, producer_op_idx, group_idx, op_to_group, staged, memo)
    )
    memo[producer_op_idx] = result
    return result


def op_emission_placement(
    ir: KernelIR,
    op_idx: int,
    group_idx: int,
    op_to_group: dict[int, int],
    staged: set[str],
    memo: dict[int, Placement] | None = None,
) -> Placement:
    """Narrowest legal Placement for ``op_idx`` in ``group_idx``.

    Raises ``ValueError`` when no slot satisfies both the scope lower bound
    and all producer barriers — the feasibility check in ``validate`` catches
    this and rejects the draw.
    """
    memo = memo if memo is not None else {}
    if op_idx not in memo:
        memo[op_idx] = _compute_placement(ir, op_idx, group_idx, op_to_group, staged, memo)
    return memo[op_idx]


def _compute_placement(
    ir: KernelIR, op_idx: int, group_idx: int, op_to_group: dict[int, int], staged: set[str], memo: dict[int, Placement]
) -> Placement:
    """Body of ``op_emission_placement`` — split out to keep the outer function at one return."""
    dim_order = ir.fusion_groups[group_idx].dim_order
    n = len(dim_order)
    floor = op_depth_floor(ir, op_idx, group_idx)
    graph = ir.op_graph
    barriers: list[Placement] = []
    for t in _op_input_tensor_names(ir, op_idx):
        producer = graph.producer_op(t)
        if producer is None or producer == op_idx or op_to_group.get(producer) != group_idx:
            continue
        barriers.append(producer_stage_placement(ir, producer, group_idx, op_to_group, staged, memo))
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
        raise ValueError(
            f"No legal emission Placement for op {op_idx} in group {group_idx}: " f"floor={floor}, barriers={barriers}"
        )
    memo[op_idx] = chosen
    return chosen
