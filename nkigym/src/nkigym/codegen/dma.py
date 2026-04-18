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
        for op_idx in group:
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
        dim_order = ir.group_dim_orders[group_idx]
        depth = _tier_depth(ir, tensor_name, tinfo, dim_order)
        dst = f"sbuf_{tensor_name}{_sbuf_slice(ir, tensor_name, tinfo, dim_order, depth)}"
        src = f"{tensor_name}{_hbm_slice(ir, tinfo, dim_order, depth)}"
        plan.setdefault(group_idx, {}).setdefault(depth, []).append(f"load_block({dst}, {src})")

    return plan


def render_psum_staging(ir: KernelIR, op_to_group: dict[int, int], staged: set[str]) -> tuple[DepthPlan, DepthPlan]:
    """Plan PSUM→SBUF staging for each PSUM tensor in *staged*, emitted at ``max(producer_finished_depth, tier_depth)``."""
    da = ir.dim_analysis
    graph = ir.op_graph
    before: DepthPlan = {}
    after: DepthPlan = {}

    for tensor_name in sorted(staged):
        producer = graph.producer_op(tensor_name)
        if producer is None:
            continue
        group_idx = op_to_group[producer]
        dim_order = ir.group_dim_orders[group_idx]
        tinfo = da.tensors[tensor_name]

        producer_depth, blocking = producer_finished_depth(ir, producer, dim_order)
        tier_depth = _tier_depth(ir, tensor_name, tinfo, dim_order)
        depth = max(producer_depth, tier_depth)
        uses_after = bool(blocking) and depth == producer_depth

        dst = f"sbuf_{tensor_name}{_sbuf_slice(ir, tensor_name, tinfo, dim_order, depth)}"
        src = f"psum_{tensor_name}"
        line = f"stage_block({dst}, {src})"
        (after if uses_after else before).setdefault(group_idx, {}).setdefault(depth, []).append(line)

    return before, after


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
        dim_order = ir.group_dim_orders[group_idx]

        op_cls = graph.op_classes[producer]
        if op_cls.ISA_LOC == "psum":
            producer_depth, _ = producer_finished_depth(ir, producer, dim_order)
            uses_after = True
        else:
            producer_depth = 2 * len(dim_order)
            uses_after = False

        tier_depth = _tier_depth(ir, ret, tinfo, dim_order)
        depth = max(producer_depth, tier_depth)
        dst = f"{ret}{_hbm_slice(ir, tinfo, dim_order, depth)}"
        src = f"sbuf_{ret}{_sbuf_slice(ir, ret, tinfo, dim_order, depth)}"
        line = f"store_block({dst}, {src})"
        (after if uses_after else before).setdefault(group_idx, {}).setdefault(depth, []).append(line)
    return before, after


def _tier_depth(ir: KernelIR, tensor_name: str, tinfo: TensorInfo, dim_order: list[str]) -> int:
    """Lower bound of the tensor's feasibility interval — ``max`` of each dim's ``tier_depth_range`` low."""
    n = len(dim_order)
    pos = {d: i for i, d in enumerate(dim_order)}
    lo = 0
    for d in tinfo.dim_ids:
        if d not in pos:
            continue
        tier = ir.tensor_placements[(tensor_name, d)]
        lo = max(lo, tier_depth_range(tier, pos[d], n)[0])
    return lo


def _sbuf_slice(ir: KernelIR, tensor_name: str, tinfo: TensorInfo, dim_order: list[str], depth: int) -> str:
    """Slice into an SBUF ``(phys_p, num_tiles_p, num_tiles_f, phys_f)`` (or 2D) buffer covering the portion active at ``depth``."""
    da = ir.dim_analysis
    dim_ids = tinfo.dim_ids
    if len(dim_ids) == 2:
        d_p, d_f = dim_ids
        tp = da.dims[d_p].physical_tile_size
        tf = da.dims[d_f].physical_tile_size
        p_idx = _sbuf_axis_index(ir, tensor_name, d_p, dim_order, depth)
        f_idx = _sbuf_axis_index(ir, tensor_name, d_f, dim_order, depth)
        expr = f"[0:{tp}, {p_idx}, {f_idx}, 0:{tf}]"
    else:
        d_p = dim_ids[0]
        tp = da.dims[d_p].physical_tile_size
        p_idx = _sbuf_axis_index(ir, tensor_name, d_p, dim_order, depth)
        expr = f"[0:{tp}, {p_idx}]"
    return expr


def _sbuf_axis_index(ir: KernelIR, tensor_name: str, dim_id: str, dim_order: list[str], depth: int) -> str:
    """Per-dim slot range for a block-scoped SBUF slice at ``depth``.

    Always yields a ``start:end`` range so the slice keeps every
    SBUF axis — block-level gadgets iterate over the resulting
    slot count. ``start`` is a loop-var expression for dims whose
    tiered loop is already open at ``depth``; ``end`` extends by
    however many slots are in-flight at that depth.
    """
    slots = num_tiles(ir, tensor_name, dim_id)
    num_ptiles = ir.dim_analysis.dims[dim_id].num_ptiles
    expr = f"0:{slots}"
    if slots > 1 and dim_id in dim_order:
        tier = ir.tensor_placements[(tensor_name, dim_id)]
        tpb = get_tpb(ir, dim_id)
        n = len(dim_order)
        i = dim_order.index(dim_id)
        expr = _slot_range(tier, dim_id, i, n, depth, tpb, num_ptiles, slots)
    return expr


def _slot_range(tier: str, dim_id: str, pos: int, n: int, depth: int, tpb: int, num_ptiles: int, slots: int) -> str:
    """Slot-range expression for one dim given its tier and the loops in scope at ``depth``."""
    block_active = depth > pos
    tile_active = depth > n + pos
    block_stride = tpb * num_ptiles
    if tier == "full" and tile_active:
        start = f"i_block_{dim_id} * {block_stride} + i_ltile_{dim_id} * {num_ptiles}"
        expr = f"{start}:{start} + {num_ptiles}"
    elif tier == "full" and block_active:
        start = f"i_block_{dim_id} * {block_stride}"
        expr = f"{start}:{start} + {block_stride}"
    elif tier == "per_block" and tile_active:
        start = f"i_ltile_{dim_id} * {num_ptiles}"
        expr = f"{start}:{start} + {num_ptiles}"
    elif tier == "per_tile":
        expr = f"0:{num_ptiles}"
    else:
        expr = f"0:{slots}"
    return expr


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
