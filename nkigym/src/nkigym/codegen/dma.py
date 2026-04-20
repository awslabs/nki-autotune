"""DMA codegen: HBM↔SBUF and PSUM↔SBUF transfer rendering.

Load, stage, and store emit at the tensor's feasibility-interval
lower bound (adjusted for the producer's blocking semantics on
stage/store). SBUF access uses the list-of-2D-tiles model
(``sbuf_buffer.SbufBuffer``); each gadget call names a contiguous
``[p_start:p_start+p_count][f_start:f_start+f_count]`` sub-block
and the gadget Python-iterates per leaf.
"""

from nkigym.codegen.buffers import sbuf_buffer
from nkigym.codegen.group_loops import DepthPlan
from nkigym.codegen.sbuf_buffer import AxisAccess
from nkigym.kernel_ir import KernelIR, get_tpb
from nkigym.kernel_ir.dim_analysis import TensorInfo, op_blocking_dims
from nkigym.kernel_ir.emission import material_blocking_dims, op_emission_placement
from nkigym.kernel_ir.validate import tier_depth_range
from nkigym.ops.transpose import NKITranspose

_TIER_RANK = {"per_tile": 0, "per_block": 1, "full": 2}


def build_op_to_group(ir: KernelIR) -> dict[int, int]:
    """Build the op-index → fusion-group-index map."""
    result: dict[int, int] = {}
    for gi, group in enumerate(ir.fusion_groups):
        for op_idx in group.op_indices:
            result[op_idx] = gi
    return result


_FUSED_TRANSPOSES_CACHE: dict[int, dict[int, str]] = {}


def find_fused_transposes(ir: KernelIR) -> dict[int, str]:
    """Return ``{op_idx: hbm_input_name}`` for every ``NKITranspose`` that fuses into an HBM ``dma_transpose`` load.

    Fusion fires iff the transpose's ``data`` input is an HBM
    kernel input with this transpose as its sole consumer AND the
    output is not the kernel return tensor. The transpose op is
    elided from ``render_nki_ops``; ``render_hbm_loads`` emits a
    single ``load_block(..., transpose=True)`` that writes the
    transposed layout directly into the consumer's SBUF buffer.
    Cached on ``id(ir)`` — ``render_ir`` is on the hot sampling
    path (thousands of calls per ``sample_variants`` run).
    """
    result = _FUSED_TRANSPOSES_CACHE.get(id(ir))
    if result is None:
        graph = ir.op_graph
        return_name = ir.dim_analysis.return_name
        result = {}
        for op_idx, op_cls in enumerate(graph.op_classes):
            if op_cls is not NKITranspose:
                continue
            inputs, outputs = graph.op_tensors[op_idx]
            data = inputs.get("data")
            if data is None or graph.producer_op(data) is not None:
                continue
            if graph.ops_touching(data) != [op_idx]:
                continue
            if outputs[0] == return_name:
                continue
            result[op_idx] = data
        _FUSED_TRANSPOSES_CACHE[id(ir)] = result
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


def render_hbm_loads(ir: KernelIR, op_to_group: dict[int, int], elided: dict[int, str]) -> DepthPlan:
    """Plan HBM→SBUF loads for every kernel input.

    Each input tensor is loaded once at the tier-feasibility
    depth in the group that contains its earliest consumer. The
    sub-block written covers every SBUF slot the consumer will
    read at that depth. When an elided transpose consumes the
    input, a single ``load_block(..., transpose=True)`` fills the
    transpose output's SBUF buffer directly instead.
    """
    da = ir.dim_analysis
    graph = ir.op_graph
    plan: DepthPlan = {}
    fused_inputs = set(elided.values())

    for op_idx in sorted(elided):
        group_idx = op_to_group[op_idx]
        depth, line = _fused_load_line(ir, op_idx, group_idx)
        plan.setdefault(group_idx, {}).setdefault(depth, []).append(line)

    for tensor_name in da.param_names:
        if tensor_name in fused_inputs:
            continue
        tinfo = da.tensors[tensor_name]
        touching = graph.ops_touching(tensor_name)
        if not touching:
            raise ValueError(f"Kernel input {tensor_name!r} has no consumers")
        group_idx = op_to_group[touching[0]]
        dim_order = ir.fusion_groups[group_idx].dim_order
        depth = _tier_depth(ir, group_idx, tensor_name, tinfo, dim_order)
        plan.setdefault(group_idx, {}).setdefault(depth, []).append(
            _gadget_call("load_block", ir, group_idx, tensor_name, tinfo, dim_order, depth, sbuf_is_dst=True)
        )

    return plan


def _fused_load_line(ir: KernelIR, op_idx: int, group_idx: int) -> tuple[int, str]:
    """Return ``(depth, line)`` for a fused ``dma_transpose`` HBM load.

    ``sbuf`` arg and sub-block bounds key on the transpose output
    (the tensor the downstream consumer reads); the HBM slice
    keys on the input's own dim_ids, which yields a pre-transpose
    shape ``(f_count * F, p_count * P)`` — exactly what
    ``load_block(..., transpose=True)`` expects.
    """
    graph = ir.op_graph
    da = ir.dim_analysis
    inputs, outputs = graph.op_tensors[op_idx]
    input_name = inputs["data"]
    output_name = outputs[0]
    dim_order = ir.fusion_groups[group_idx].dim_order
    output_tinfo = da.tensors[output_name]
    depth = _tier_depth(ir, group_idx, output_name, output_tinfo, dim_order)
    buf = sbuf_buffer(ir, output_name)
    p_access, f_access = _axis_access(ir, group_idx, output_name, output_tinfo, dim_order, depth)
    p_start, p_count, f_start, f_count = buf.range(p_access, f_access)
    input_tinfo = da.tensors[input_name]
    mem_expr = f"{input_name}{_hbm_slice(ir, input_tinfo, dim_order, depth)}"
    sbuf_arg = f"sbuf_{output_name}"
    line = f"load_block({sbuf_arg}, {mem_expr}, {p_start}, {p_count}, {f_start}, {f_count}, transpose=True)"
    return depth, line


def render_psum_staging(ir: KernelIR, op_to_group: dict[int, int], staged: set[str]) -> tuple[DepthPlan, DepthPlan]:
    """Plan PSUM→SBUF staging for blocking PSUM producers at the outermost-blocking-loop close.

    Ops with output (non-blocking) ptile dims own their own staging
    in ``render_nki_ops._render_op_block`` (per-iteration stage
    inside the output ptile loop). Non-blocking producers without
    output ptile dims get their stage emitted inline right after
    the ISA call by ``render_nki_ops`` (see ``inline_stage_line``)
    so intra-group consumers see the staged SBUF. Remaining cases
    — blocking producers — are staged here in the after-plan,
    outside the outermost blocking loop so the PSUM read happens
    only once per output tile.
    """
    da = ir.dim_analysis
    graph = ir.op_graph
    before: DepthPlan = {}
    after: DepthPlan = {}

    for tensor_name in sorted(staged):
        producer = graph.producer_op(tensor_name)
        if producer is None:
            continue
        if has_output_ptile_dims(ir, producer):
            continue
        group_idx = op_to_group[producer]
        dim_order = ir.fusion_groups[group_idx].dim_order
        material = material_blocking_dims(ir, producer, dim_order)
        if not material:
            continue
        producer_depth = min(dim_order.index(d) for d in material)
        tinfo = da.tensors[tensor_name]
        line = _gadget_call(
            "stage_block", ir, group_idx, tensor_name, tinfo, dim_order, producer_depth, sbuf_is_dst=True, psum_src=True
        )
        after.setdefault(group_idx, {}).setdefault(producer_depth, []).append(line)

    return before, after


def inline_stage_line(
    ir: KernelIR, group_idx: int, producer_op_idx: int, tensor_name: str, dim_order: list[str], depth: int
) -> str:
    """PSUM→SBUF stage line for a non-material-blocking producer, emitted at the producer's depth."""
    _ = producer_op_idx
    tinfo = ir.dim_analysis.tensors[tensor_name]
    return _gadget_call(
        "stage_block", ir, group_idx, tensor_name, tinfo, dim_order, depth, sbuf_is_dst=True, psum_src=True
    )


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


def has_output_ptile_dims(ir: KernelIR, op_idx: int) -> bool:
    """True iff this op has at least one non-blocking ptile dim (output-tile axis)."""
    op_cls = ir.op_graph.op_classes[op_idx]
    blocking = op_blocking_dims(op_cls, ir.dim_analysis.per_op_axis_maps[op_idx])
    return any(dim_id not in blocking for dim_id, _ in ptile_loop_dims(ir, op_idx))


def render_hbm_store(ir: KernelIR, op_to_group: dict[int, int], staged: set[str]) -> tuple[DepthPlan, DepthPlan]:
    """Plan the SBUF→HBM store for the return tensor at the producer's emission slot."""
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
        placement = op_emission_placement(ir, producer, group_idx, op_to_group, staged)
        op_cls = graph.op_classes[producer]
        is_psum = op_cls.ISA_LOC == "psum"
        material = material_blocking_dims(ir, producer, dim_order) if is_psum else set()
        if is_psum and material:
            depth = min(dim_order.index(d) for d in material)
            uses_after = True
        else:
            depth = max(placement.depth, _tier_depth(ir, group_idx, ret, tinfo, dim_order))
            uses_after = placement.phase == "after"
        line = _gadget_call("store_block", ir, group_idx, ret, tinfo, dim_order, depth, sbuf_is_dst=False)
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


def _gadget_call(
    gadget: str,
    ir: KernelIR,
    group_idx: int,
    tensor_name: str,
    tinfo: TensorInfo,
    dim_order: list[str],
    depth: int,
    sbuf_is_dst: bool,
    psum_src: bool = False,
) -> str:
    """Emit ``gadget(sbuf=..., mem=..., p_start=..., p_count=..., f_start=..., f_count=...)``.

    For load/stage, ``sbuf`` is the dst list-of-lists and ``mem``
    is the 2D HBM/PSUM region; for store, ``sbuf`` is the src list
    and ``mem`` is the 2D HBM dst. The sub-block bounds come from
    ``SbufBuffer.range`` with the in-scope axes bound and the rest
    spanning their full factor.
    """
    buf = sbuf_buffer(ir, tensor_name)
    p_access, f_access = _axis_access(ir, group_idx, tensor_name, tinfo, dim_order, depth)
    p_start, p_count, f_start, f_count = buf.range(p_access, f_access)
    sbuf_arg = f"sbuf_{tensor_name}"
    mem_expr = _mem_expr(ir, tensor_name, tinfo, dim_order, depth, psum_src)
    bounds = f"{p_start}, {p_count}, {f_start}, {f_count}"
    first, second = (sbuf_arg, mem_expr) if sbuf_is_dst else (mem_expr, sbuf_arg)
    return f"{gadget}({first}, {second}, {bounds})"


def _axis_access(
    ir: KernelIR, group_idx: int, tensor_name: str, tinfo: TensorInfo, dim_order: list[str], depth: int
) -> tuple[AxisAccess, AxisAccess]:
    """Return ``(p_access, f_access)`` for a gadget emission at ``depth``.

    Each axis's ``AxisAccess.block`` / ``ltile`` is bound to the
    loop var when that loop is in scope, or left ``None`` when it
    isn't. ``ptile`` is unused by gadgets (they span every
    physical tile inside a leaf).
    """
    dim_ids = tinfo.dim_ids
    p_axis = _scope_access(group_idx, tensor_name, dim_ids[0], dim_order, depth, ir)
    f_axis = (
        _scope_access(group_idx, tensor_name, dim_ids[1], dim_order, depth, ir)
        if len(dim_ids) == 2
        else AxisAccess(block="0", ltile="0")
    )
    return p_axis, f_axis


def _scope_access(
    group_idx: int, tensor_name: str, dim_id: str, dim_order: list[str], depth: int, ir: KernelIR
) -> AxisAccess:
    """Bind ``block`` / ``ltile`` for one dim based on tier and in-scope loops.

    A dim outside ``dim_order`` has no loops → both bound to
    ``"0"`` (the single list slot at ``list_slots == 1``). A dim
    inside ``dim_order`` binds its loop var iff the tier keeps
    that list factor (``full`` keeps both; ``per_block`` keeps
    only ltile; ``per_tile`` keeps neither) AND the loop is in
    scope at ``depth``. A list factor that the tier does not
    keep collapses to ``"0"`` (the factor is 1 at allocation).
    """
    placements = ir.fusion_groups[group_idx].tensor_placements
    key = ("sbuf", tensor_name, dim_id)
    tier = placements.get(key, "per_tile")
    block: str | None = "0"
    ltile: str | None = "0"
    if dim_id in dim_order:
        pos = dim_order.index(dim_id)
        n = len(dim_order)
        block = _bind(tier, "full", depth > pos, f"i_block_{dim_id}")
        ltile = _bind(tier, "per_block", depth > n + pos, f"i_ltile_{dim_id}")
    return AxisAccess(block=block, ltile=ltile)


def _bind(tier: str, required_tier: str, loop_open: bool, var: str) -> str | None:
    """Bind a list-factor loop var when the tier keeps it AND the loop is in scope.

    ``required_tier`` is the minimum tier that materializes the
    factor (``"full"`` for the block factor, ``"per_block"`` for
    the ltile factor). When the tier doesn't keep the factor, the
    list length is 1 and we return ``"0"``. When the tier keeps
    the factor, we return the loop var if it's in scope, else
    ``None`` so the caller spans the full range.
    """
    kept = _TIER_RANK[tier] >= _TIER_RANK[required_tier]
    result: str | None
    if not kept:
        result = "0"
    elif loop_open:
        result = var
    else:
        result = None
    return result


def _mem_expr(
    ir: KernelIR, tensor_name: str, tinfo: TensorInfo, dim_order: list[str], depth: int, psum_src: bool
) -> str:
    """Return the 2D HBM or PSUM argument expression passed to the gadget."""
    if psum_src:
        expr = f"psum_{tensor_name}"
    else:
        expr = f"{tensor_name}{_hbm_slice(ir, tinfo, dim_order, depth)}"
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
