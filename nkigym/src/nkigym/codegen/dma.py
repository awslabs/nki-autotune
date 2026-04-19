"""DMA codegen: HBM↔SBUF and PSUM↔SBUF transfer rendering.

Load, stage, and store emit at the tensor's feasibility-interval
lower bound (adjusted for the producer's blocking semantics on
stage/store). Slice expressions are built from in-scope loop
vars and tiers; gadgets are thin ISA wrappers that don't loop.
"""

from nkigym.codegen.buffers import num_tiles
from nkigym.codegen.group_loops import DepthPlan
from nkigym.kernel_ir import KernelIR, get_tpb
from nkigym.kernel_ir.dim_analysis import TensorInfo, op_blocking_dims
from nkigym.kernel_ir.validate import tier_depth_range


def build_op_to_group(ir: KernelIR) -> dict[int, int]:
    """Build the op-index → fusion-group-index map."""
    result: dict[int, int] = {}
    for gi, group in enumerate(ir.fusion_groups):
        for op_idx in group.op_indices:
            result[op_idx] = gi
    return result


def producer_finished_depth(ir: KernelIR, producer: int, dim_order: list[str]) -> tuple[int, set[str]]:
    """Return (depth at which the producer op has finished all writes, blocking dims).

    ``depth`` is ``i_min`` — the position of the outermost
    blocking dim in ``dim_order`` — when the op has blocking
    dims that overlap ``dim_order``; otherwise ``2 * N`` (the
    innermost body).
    """
    da = ir.dim_analysis
    op_cls = ir.op_graph.op_classes[producer]
    blocking = op_blocking_dims(op_cls, da.per_op_axis_maps[producer]) & set(dim_order)
    depth = min(dim_order.index(d) for d in blocking) if blocking else 2 * len(dim_order)
    return depth, blocking


def render_hbm_loads(ir: KernelIR, op_to_group: dict[int, int]) -> DepthPlan:
    """Plan HBM→SBUF loads for every kernel input.

    Each input tensor is loaded once at the tier-feasibility
    depth in the group that contains its earliest consumer. The
    slice written covers every SBUF slot the consumer will read
    with that tier.
    """
    da = ir.dim_analysis
    graph = ir.op_graph
    plan: DepthPlan = {}

    for tensor_name in da.param_names:
        tinfo = da.tensors[tensor_name]
        touching = graph.ops_touching(tensor_name)
        if not touching:
            raise ValueError(f"Kernel input {tensor_name!r} has no consumers")
        group_idx = op_to_group[touching[0]]
        dim_order = ir.fusion_groups[group_idx].dim_order
        depth = _tier_depth(ir, group_idx, tensor_name, tinfo, dim_order)
        dst = f"sbuf_{tensor_name}{_sbuf_slice(ir, group_idx, tensor_name, tinfo, dim_order, depth)}"
        src = f"{tensor_name}{_hbm_slice(ir, tinfo, dim_order, depth)}"
        plan.setdefault(group_idx, {}).setdefault(depth, []).append(f"load_block({dst}, {src})")

    return plan


def render_psum_staging(ir: KernelIR, op_to_group: dict[int, int], staged: set[str]) -> tuple[DepthPlan, DepthPlan]:
    """Plan PSUM→SBUF staging for each PSUM tensor in *staged*, emitted at ``max(producer_finished_depth, tier_depth)``.

    Ops with OUTPUT (non-blocking) ptile dims own their own staging
    in ``render_nki_ops._render_op_block`` — each output ptile
    iteration produces a distinct output tile and needs a
    per-iteration stage to its own SBUF slot. This function skips
    those producers. Ops whose ptile dims are ALL blocking fall
    through and get staged at group scope (outside every block /
    ltile / ptile loop), which is the only math-valid place to
    read a PSUM accumulator.
    """
    da = ir.dim_analysis
    graph = ir.op_graph
    before: DepthPlan = {}
    after: DepthPlan = {}

    for tensor_name in sorted(staged):
        producer = graph.producer_op(tensor_name)
        if producer is None:
            continue
        if _has_output_ptile_dims(ir, producer):
            continue
        group_idx = op_to_group[producer]
        dim_order = ir.fusion_groups[group_idx].dim_order
        tinfo = da.tensors[tensor_name]

        producer_depth, blocking = producer_finished_depth(ir, producer, dim_order)
        if blocking:
            depth = producer_depth
            uses_after = True
        else:
            depth = max(producer_depth, _tier_depth(ir, group_idx, tensor_name, tinfo, dim_order))
            uses_after = False

        dst = f"sbuf_{tensor_name}{_sbuf_slice(ir, group_idx, tensor_name, tinfo, dim_order, depth)}"
        src = f"psum_{tensor_name}"
        line = f"stage_block({dst}, {src})"
        (after if uses_after else before).setdefault(group_idx, {}).setdefault(depth, []).append(line)

    return before, after


def ptile_loop_dims(ir: KernelIR, op_idx: int) -> list[tuple[str, int]]:
    """Dims that need an ``i_ptile_{d}`` loop for this op, outer-to-inner."""
    da = ir.dim_analysis
    op_tiles = da.op_tile_sizes[op_idx]
    result: list[tuple[str, int]] = []
    seen: set[str] = set()
    for tensor_name in ir.op_graph.op_tensor_names(op_idx):
        tinfo = da.tensors.get(tensor_name)
        if tinfo is None:
            continue
        for dim_id in tinfo.dim_ids:
            if dim_id in seen:
                continue
            seen.add(dim_id)
            di = da.dims[dim_id]
            op_slots = op_tiles.get(dim_id, di.physical_tile_size) // di.physical_tile_size
            total_slots = di.num_ptiles
            if total_slots > op_slots:
                result.append((dim_id, total_slots // op_slots))
    return result


def _has_output_ptile_dims(ir: KernelIR, op_idx: int) -> bool:
    """True iff this op has at least one non-blocking ptile dim (output-tile axis)."""
    op_cls = ir.op_graph.op_classes[op_idx]
    blocking = op_blocking_dims(op_cls, ir.dim_analysis.per_op_axis_maps[op_idx])
    return any(dim_id not in blocking for dim_id, _ in ptile_loop_dims(ir, op_idx))


def render_hbm_store(ir: KernelIR, op_to_group: dict[int, int]) -> tuple[DepthPlan, DepthPlan]:
    """Plan the SBUF→HBM store for the return tensor — into the after-plan when the producer writes PSUM (sequencing after the stage), otherwise before-plan."""
    da = ir.dim_analysis
    graph = ir.op_graph
    ret = da.return_name
    producer = graph.producer_op(ret)
    before: DepthPlan = {}
    after: DepthPlan = {}
    if producer is not None:
        group_idx = op_to_group[producer]
        tinfo = da.tensors[ret]
        dim_order = ir.fusion_groups[group_idx].dim_order

        op_cls = graph.op_classes[producer]
        if op_cls.ISA_LOC == "psum":
            producer_depth, blocking = producer_finished_depth(ir, producer, dim_order)
            if blocking:
                depth = producer_depth
                uses_after = True
            else:
                depth = max(producer_depth, _tier_depth(ir, group_idx, ret, tinfo, dim_order))
                uses_after = False
        else:
            depth = max(2 * len(dim_order), _tier_depth(ir, group_idx, ret, tinfo, dim_order))
            uses_after = False

        dst = f"{ret}{_hbm_slice(ir, tinfo, dim_order, depth)}"
        src = f"sbuf_{ret}{_sbuf_slice(ir, group_idx, ret, tinfo, dim_order, depth)}"
        line = f"store_block({dst}, {src})"
        (after if uses_after else before).setdefault(group_idx, {}).setdefault(depth, []).append(line)
    return before, after


def _tier_depth(ir: KernelIR, group_idx: int, tensor_name: str, tinfo: TensorInfo, dim_order: list[str]) -> int:
    """Lower bound of the tensor's feasibility interval in ``group_idx`` — ``max`` of each dim's ``tier_depth_range`` low."""
    n = len(dim_order)
    pos = {d: i for i, d in enumerate(dim_order)}
    lo = 0
    placements = ir.fusion_groups[group_idx].tensor_placements
    for d in tinfo.dim_ids:
        if d not in pos:
            continue
        key = ("sbuf", tensor_name, d)
        if key not in placements:
            continue
        tier = placements[key]
        lo = max(lo, tier_depth_range(tier, pos[d], n)[0])
    return lo


def _sbuf_slice(
    ir: KernelIR, group_idx: int, tensor_name: str, tinfo: TensorInfo, dim_order: list[str], depth: int
) -> str:
    """Slice into an SBUF ``(phys_p, num_tiles_p, num_tiles_f, phys_f)`` (or 2D) buffer covering the portion active at ``depth``."""
    return _sbuf_slice_ptile(ir, group_idx, tensor_name, tinfo, dim_order, depth, ptile_dims=frozenset())


def sbuf_ptile_slice(
    ir: KernelIR, group_idx: int, tensor_name: str, tinfo: TensorInfo, dim_order: list[str], ptile_dims: frozenset[str]
) -> str:
    """Per-ptile SBUF slice at the innermost body depth.

    Like ``_sbuf_slice`` but narrows each dim listed in
    ``ptile_dims`` to a single slot indexed by ``i_ptile_{d}``,
    added on top of any in-scope block/ltile loop offset.
    """
    return _sbuf_slice_ptile(
        ir, group_idx, tensor_name, tinfo, dim_order, depth=2 * len(dim_order), ptile_dims=ptile_dims
    )


def _sbuf_slice_ptile(
    ir: KernelIR,
    group_idx: int,
    tensor_name: str,
    tinfo: TensorInfo,
    dim_order: list[str],
    depth: int,
    ptile_dims: frozenset[str],
) -> str:
    """Internal: build the full SBUF slice string with optional per-ptile narrowing."""
    da = ir.dim_analysis
    dim_ids = tinfo.dim_ids
    if len(dim_ids) == 2:
        d_p, d_f = dim_ids
        tp = da.dims[d_p].physical_tile_size
        tf = da.dims[d_f].physical_tile_size
        p_idx = _sbuf_axis_index(ir, group_idx, tensor_name, d_p, dim_order, depth, ptile_dims)
        f_idx = _sbuf_axis_index(ir, group_idx, tensor_name, d_f, dim_order, depth, ptile_dims)
        expr = f"[0:{tp}, {p_idx}, {f_idx}, 0:{tf}]"
    else:
        d_p = dim_ids[0]
        tp = da.dims[d_p].physical_tile_size
        p_idx = _sbuf_axis_index(ir, group_idx, tensor_name, d_p, dim_order, depth, ptile_dims)
        expr = f"[0:{tp}, {p_idx}]"
    return expr


def _sbuf_axis_index(
    ir: KernelIR,
    group_idx: int,
    tensor_name: str,
    dim_id: str,
    dim_order: list[str],
    depth: int,
    ptile_dims: frozenset[str] = frozenset(),
) -> str:
    """Per-dim slot range for a block-scoped SBUF slice at ``depth``.

    Always yields a ``start:end`` range so the slice keeps every
    SBUF axis — block-level gadgets iterate over the resulting
    slot count. ``start`` is a loop-var expression for dims whose
    tiered loop is already open at ``depth``; ``end`` extends by
    however many slots are in-flight at that depth. Dims listed
    in ``ptile_dims`` add a ``+ i_ptile_{d}`` to the start and
    narrow the range to a single slot.
    """
    slots = num_tiles(ir, tensor_name, dim_id)
    num_ptiles = ir.dim_analysis.dims[dim_id].num_ptiles
    expr = f"0:{slots}"
    placements = ir.fusion_groups[group_idx].tensor_placements
    key = ("sbuf", tensor_name, dim_id)
    if slots > 1 and dim_id in dim_order and key in placements:
        tier = placements[key]
        tpb = get_tpb(ir, dim_id)
        n = len(dim_order)
        i = dim_order.index(dim_id)
        expr = _slot_range(tier, dim_id, i, n, depth, tpb, num_ptiles, slots, dim_id in ptile_dims)
    elif dim_id in ptile_dims:
        expr = f"i_ptile_{dim_id}:i_ptile_{dim_id} + 1"
    return expr


def _slot_range(
    tier: str, dim_id: str, pos: int, n: int, depth: int, tpb: int, num_ptiles: int, slots: int, ptile: bool
) -> str:
    """Slot-range expression for one dim given its tier and the loops in scope at ``depth``.

    When ``ptile`` is ``True``, the range is narrowed to a
    single slot at ``start + i_ptile_{dim_id}``.
    """
    block_active = depth > pos
    tile_active = depth > n + pos
    block_stride = tpb * num_ptiles
    if tier == "full" and tile_active:
        start = f"i_block_{dim_id} * {block_stride} + i_ltile_{dim_id} * {num_ptiles}"
        expr = _narrow_or_range(start, num_ptiles, ptile, dim_id)
    elif tier == "full" and block_active:
        start = f"i_block_{dim_id} * {block_stride}"
        expr = _narrow_or_range(start, block_stride, ptile, dim_id)
    elif tier == "per_block" and tile_active:
        start = f"i_ltile_{dim_id} * {num_ptiles}"
        expr = _narrow_or_range(start, num_ptiles, ptile, dim_id)
    elif tier == "per_tile":
        expr = f"i_ptile_{dim_id}:i_ptile_{dim_id} + 1" if ptile else f"0:{num_ptiles}"
    else:
        expr = f"0:{slots}"
    return expr


def _narrow_or_range(start: str, width: int, ptile: bool, dim_id: str) -> str:
    """Return either a single-slot range ``start + i_ptile_{d} : start + i_ptile_{d} + 1`` or the full ``start:start + width`` range."""
    return f"{start} + i_ptile_{dim_id}:{start} + i_ptile_{dim_id} + 1" if ptile else f"{start}:{start} + {width}"


def _hbm_slice(ir: KernelIR, tinfo: TensorInfo, dim_order: list[str], depth: int) -> str:
    """HBM slice covering the in-flight portion of the tensor.

    Each axis is ``start:end`` built from loop vars in scope at
    ``depth`` and the tier on that dim.
    """
    dim_ids = tinfo.dim_ids
    par = _hbm_axis_range(ir, dim_ids[0], dim_order, depth)
    if len(dim_ids) == 2:
        free = _hbm_axis_range(ir, dim_ids[1], dim_order, depth)
        expr = f"[{par}, {free}]"
    else:
        expr = f"[{par}]"
    return expr


def _hbm_axis_range(ir: KernelIR, dim_id: str, dim_order: list[str], depth: int) -> str:
    """HBM ``start:end`` range for one dim, derived from loop vars in scope at ``depth``."""
    di = ir.dim_analysis.dims[dim_id]
    logical = di.logical_tile_size
    block_stride = get_tpb(ir, dim_id) * logical

    if dim_id not in dim_order or depth <= dim_order.index(dim_id):
        rng = f"0:{di.dim_size}"
    elif depth <= len(dim_order) + dim_order.index(dim_id):
        start = f"i_block_{dim_id} * {block_stride}"
        rng = f"{start}:{start} + {block_stride}"
    else:
        start = f"i_block_{dim_id} * {block_stride} + i_ltile_{dim_id} * {logical}"
        rng = f"{start}:{start} + {logical}"
    return rng
