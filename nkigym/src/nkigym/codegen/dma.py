"""DMA codegen: HBM↔SBUF and PSUM↔SBUF transfer rendering.

``NKILoad`` / ``NKIStore`` / ``NKIDMATranspose`` ops in
``OpGraph`` emit their own DMA lines at the op's own emission
slot — ``render_nki_ops`` dispatches to ``dma_load_line`` /
``dma_store_line`` / ``dma_transpose_line`` here. Legacy
HBM-synthesis from tensor metadata is gone; DMA is fully
ir-driven.

``render_psum_staging`` / ``inline_stage_line`` remain because
PSUM→SBUF staging is still an implicit renderer concern (not
a ir node) — the producer writes to PSUM, the renderer
emits ``stage_block`` after its accumulation loops close.
"""

from nkigym.codegen.buffers import sbuf_buffer
from nkigym.codegen.group_loops import DepthPlan
from nkigym.codegen.matmul_block_detect import gadget_absorbed_dims, is_matmul_block_candidate
from nkigym.codegen.sbuf_buffer import AxisAccess, buffer_ident
from nkigym.kernel_ir import KernelIR
from nkigym.kernel_ir.types import TensorInfo
from nkigym.kernel_ir.validate import tier_depth_range
from nkigym.kernel_ir.validate.emission import (
    Placement,
    block_depth,
    body_depth,
    ltile_depth,
    material_blocking_dims,
    op_emission_placement,
)
from nkigym.ops.base import NKIOp
from nkigym.ops.dma import NKIDMATranspose, NKIStore

_TIER_RANK = {"per_tile": 0, "per_block": 1, "full": 2}


def render_dma_op(
    ir: KernelIR,
    op: NKIOp,
    gi: int,
    op_to_group: dict[int, int],
    staged: set[str],
    memo: dict[int, Placement],
    before_plan: DepthPlan,
    after_plan: DepthPlan,
) -> None:
    """Render ``NKILoad`` / ``NKIStore`` / ``NKIDMATranspose`` at its own emission slot."""
    placement = op_emission_placement(ir, op, gi, op_to_group, staged, memo)
    renderer = (
        dma_store_line
        if isinstance(op, NKIStore)
        else (dma_transpose_line if isinstance(op, NKIDMATranspose) else dma_load_line)
    )
    target = before_plan if placement.phase == "before" else after_plan
    target.setdefault(gi, {}).setdefault(placement.depth, []).append(renderer(ir, op, gi, placement.depth))


def build_op_to_group(ir: KernelIR) -> dict[int, int]:
    """Build the ``id(op) -> group_idx`` map."""
    result: dict[int, int] = {}
    for gi, group in enumerate(ir.groups):
        for op in group.ops:
            result[id(op)] = gi
    return result


def producer_finished_depth(ir: KernelIR, producer: NKIOp, dim_order: list[str]) -> tuple[int, set[str]]:
    """Return depth at which the producer op has finished all writes + its blocking dims."""
    blocking = ir.op_blocking_dims.get(producer, set()) & set(dim_order)
    depth = block_depth(min(dim_order.index(d) for d in blocking)) if blocking else body_depth(len(dim_order))
    return depth, blocking


def dma_load_line(ir: KernelIR, op: NKIOp, group_idx: int, depth: int) -> str:
    """Emit ``load_block(sbuf_<out>, <in>[...], ...)`` for an ``NKILoad`` node."""
    ir = ir
    inputs = ir.op_inputs.get(op, {})
    outputs = ir.op_outputs.get(op, [])
    hbm_name = inputs["data"]
    sbuf_name = outputs[0]
    tinfo = ir.logical_tensors[sbuf_name]
    dim_order = ir.groups[group_idx].dim_order
    return _gadget_call(
        "load_block", ir, group_idx, sbuf_name, tinfo, dim_order, depth, sbuf_is_dst=True, hbm_name=hbm_name
    )


def dma_store_line(ir: KernelIR, op: NKIOp, group_idx: int, depth: int) -> str:
    """Emit ``store_block(<out>[...], sbuf_<in>, ...)`` for an ``NKIStore`` node."""
    ir = ir
    inputs = ir.op_inputs.get(op, {})
    sbuf_name = inputs["data"]
    hbm_name = ir.return_name
    tinfo = ir.logical_tensors[sbuf_name]
    dim_order = ir.groups[group_idx].dim_order
    return _gadget_call(
        "store_block", ir, group_idx, sbuf_name, tinfo, dim_order, depth, sbuf_is_dst=False, hbm_name=hbm_name
    )


def dma_transpose_line(ir: KernelIR, op: NKIOp, group_idx: int, depth: int) -> str:
    """Emit ``load_block(..., transpose=True)`` for an ``NKIDMATranspose`` composite."""
    ir = ir
    inputs = ir.op_inputs.get(op, {})
    outputs = ir.op_outputs.get(op, [])
    hbm_name = inputs["data"]
    sbuf_name = outputs[0]
    tinfo = ir.logical_tensors[sbuf_name]
    dim_order = ir.groups[group_idx].dim_order
    buf = sbuf_buffer(ir, sbuf_name)
    p_access, f_access = _axis_access(ir, group_idx, sbuf_name, tinfo, dim_order, depth)
    p_start, p_count, f_start, f_count = buf.range(p_access, f_access)
    input_tinfo = ir.logical_tensors[hbm_name]
    mem_expr = f"{hbm_name}{_hbm_slice(ir, input_tinfo, dim_order, depth, gadget_absorbed_dims(ir, group_idx))}"
    sbuf_arg = f"sbuf_{buffer_ident(sbuf_name)}"
    return f"load_block({sbuf_arg}, {mem_expr}, {p_start}, {p_count}, {f_start}, {f_count}, transpose=True)"


def tier_depth(ir: KernelIR, group_idx: int, tensor_name: str, tinfo: TensorInfo, dim_order: list[str]) -> int:
    """Lower bound of the tensor's feasibility interval in ``group_idx``."""
    n = len(dim_order)
    pos = {d: i for i, d in enumerate(dim_order)}
    lo = 0
    placements = ir.groups[group_idx].tensor_placements
    for d in tinfo.dim_ids:
        if d not in pos:
            continue
        key = ("sbuf", tensor_name, d)
        if key not in placements:
            continue
        tier = placements[key]
        lo = max(lo, tier_depth_range(tier, pos[d], n)[0])
    return lo


def render_psum_staging(ir: KernelIR, op_to_group: dict[int, int], staged: set[str]) -> tuple[DepthPlan, DepthPlan]:
    """Plan PSUM→SBUF staging for blocking PSUM producers.

    Skips producers whose op is dispatched to the matmul_block
    gadget — that path writes directly into the running-sum SBUF
    buffer and has no PSUM→SBUF staging to emit.
    """
    before: DepthPlan = {}
    after: DepthPlan = {}
    ir = ir
    for tensor_name in sorted(staged):
        producer = _producer_of(ir, tensor_name)
        if producer is None or has_output_ptile_dims(ir, producer):
            continue
        gi = op_to_group[id(producer)]
        if is_matmul_block_candidate(ir, producer, gi):
            continue
        dim_order = ir.groups[gi].dim_order
        material = material_blocking_dims(ir, producer, dim_order)
        if not material:
            continue
        producer_depth = block_depth(min(dim_order.index(d) for d in material))
        tinfo = ir.logical_tensors[tensor_name]
        line = _gadget_call(
            "stage_block", ir, gi, tensor_name, tinfo, dim_order, producer_depth, sbuf_is_dst=True, psum_src=True
        )
        after.setdefault(gi, {}).setdefault(producer_depth, []).append(line)
    return before, after


def _producer_of(ir: KernelIR, tensor_name: str) -> NKIOp | None:
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


def inline_stage_line(
    ir: KernelIR, group_idx: int, producer: NKIOp, tensor_name: str, dim_order: list[str], depth: int
) -> str:
    """PSUM→SBUF stage line emitted at the producer's depth."""
    _ = producer
    tinfo = ir.logical_tensors[tensor_name]
    return _gadget_call(
        "stage_block", ir, group_idx, tensor_name, tinfo, dim_order, depth, sbuf_is_dst=True, psum_src=True
    )


def ptile_loop_dims(ir: KernelIR, op: NKIOp) -> list[tuple[str, int]]:
    """Dims needing an ``i_ptile_{d}`` loop for ``op``."""
    ir = ir
    op_tiles = ir.op_tile_sizes.get(op, {})
    result: list[tuple[str, int]] = []
    seen: set[str] = set()
    tensor_names = [*ir.op_inputs.get(op, {}).values(), *ir.op_outputs.get(op, [])]
    for tname in tensor_names:
        tinfo = ir.logical_tensors.get(tname)
        if tinfo is None:
            continue
        for dim_id in tinfo.dim_ids:
            if dim_id in seen:
                continue
            seen.add(dim_id)
            di = ir.dimensions[dim_id]
            op_slots = op_tiles.get(dim_id, di.physical_tile_size) // di.physical_tile_size
            total_slots = di.num_ptiles
            if total_slots > op_slots:
                result.append((dim_id, total_slots // op_slots))
    return result


def has_output_ptile_dims(ir: KernelIR, op: NKIOp) -> bool:
    """True iff this op has at least one non-blocking ptile dim."""
    blocking = ir.op_blocking_dims.get(op, set())
    return any(dim_id not in blocking for dim_id, _ in ptile_loop_dims(ir, op))


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
    hbm_name: str | None = None,
) -> str:
    """Emit a gadget call.

    ``tensor_name`` names the SBUF side (buffer + placement lookup).
    ``hbm_name`` overrides the HBM-side tensor name used in the
    memory slice — needed for Load/Store where the SBUF alias
    differs from the kernel-input/return HBM tensor name.
    """
    buf = sbuf_buffer(ir, tensor_name)
    p_access, f_access = _axis_access(ir, group_idx, tensor_name, tinfo, dim_order, depth)
    p_start, p_count, f_start, f_count = buf.range(p_access, f_access)
    sbuf_arg = f"sbuf_{buffer_ident(tensor_name)}"
    effective_hbm = hbm_name if hbm_name is not None else tensor_name
    mem_expr = _mem_expr(ir, effective_hbm, tinfo, dim_order, depth, psum_src, gadget_absorbed_dims(ir, group_idx))
    bounds = f"{p_start}, {p_count}, {f_start}, {f_count}"
    first, second = (sbuf_arg, mem_expr) if sbuf_is_dst else (mem_expr, sbuf_arg)
    return f"{gadget}({first}, {second}, {bounds})"


def _axis_access(
    ir: KernelIR, group_idx: int, tensor_name: str, tinfo: TensorInfo, dim_order: list[str], depth: int
) -> tuple[AxisAccess, AxisAccess]:
    """Return ``(p_access, f_access)`` for a gadget emission at ``depth``."""
    absorbed = _absorbed_dims(ir, group_idx)
    dim_ids = tinfo.dim_ids
    p_axis = _scope_access(group_idx, tensor_name, dim_ids[0], dim_order, depth, ir, absorbed)
    if len(dim_ids) == 2:
        f_axis = _scope_access(group_idx, tensor_name, dim_ids[1], dim_order, depth, ir, absorbed)
    else:
        f_axis = AxisAccess(block="0", ltile="0")
    return p_axis, f_axis


def _absorbed_dims(ir: KernelIR, group_idx: int) -> set[str]:
    """Return the ltile-absorbed dim set for ``group_idx``."""
    return gadget_absorbed_dims(ir, group_idx)


def _scope_access(
    group_idx: int, tensor_name: str, dim_id: str, dim_order: list[str], depth: int, ir: KernelIR, absorbed: set[str]
) -> AxisAccess:
    """Bind ``block`` / ``ltile`` for one dim based on tier and in-scope loops."""
    placements = ir.groups[group_idx].tensor_placements
    key = ("sbuf", tensor_name, dim_id)
    tier = placements.get(key, "per_tile")
    block: str | None = "0"
    ltile: str | None = "0"
    if dim_id in dim_order:
        pos = dim_order.index(dim_id)
        block = _bind(tier, "full", depth > block_depth(pos), f"i_block_{dim_id}")
        ltile_is_open = depth > ltile_depth(pos) and dim_id not in absorbed
        ltile = _bind(tier, "per_block", ltile_is_open, f"i_ltile_{dim_id}")
    return AxisAccess(block=block, ltile=ltile)


def _bind(tier: str, required_tier: str, loop_open: bool, var: str) -> str | None:
    """Bind a list-factor loop var when the tier keeps it AND the loop is in scope."""
    kept = _TIER_RANK[tier] >= _TIER_RANK[required_tier]
    if not kept:
        result: str | None = "0"
    elif loop_open:
        result = var
    else:
        result = None
    return result


def _mem_expr(
    ir: KernelIR,
    tensor_name: str,
    tinfo: TensorInfo,
    dim_order: list[str],
    depth: int,
    psum_src: bool,
    absorbed: set[str],
) -> str:
    """Return the 2D HBM or PSUM argument expression passed to the gadget."""
    if psum_src:
        expr = f"psum_{buffer_ident(tensor_name)}"
    else:
        expr = f"{tensor_name}{_hbm_slice(ir, tinfo, dim_order, depth, absorbed)}"
    return expr


def _hbm_slice(ir: KernelIR, tinfo: TensorInfo, dim_order: list[str], depth: int, absorbed: set[str]) -> str:
    """HBM slice covering the in-flight portion of the tensor."""
    dim_ids = tinfo.dim_ids
    par = _hbm_axis_range(ir, dim_ids[0], dim_order, depth, absorbed)
    if len(dim_ids) == 2:
        free = _hbm_axis_range(ir, dim_ids[1], dim_order, depth, absorbed)
        expr = f"[{par}, {free}]"
    else:
        expr = f"[{par}]"
    return expr


def _hbm_axis_range(ir: KernelIR, dim_id: str, dim_order: list[str], depth: int, absorbed: set[str]) -> str:
    """HBM ``start:end`` range for one dim.

    ``absorbed`` carries dims whose ltile loops are absorbed by a
    gadget — for those dims the slice always spans one whole
    block (block_stride wide) even when the emission depth is
    inside the (would-be-open) ltile slot.
    """
    ir = ir
    di = ir.dimensions[dim_id]
    logical = di.logical_tile_size
    block_stride = ir.ltiles_per_block.get(dim_id, 1) * logical
    ltile_slot_open = dim_id in dim_order and depth > ltile_depth(dim_order.index(dim_id)) and dim_id not in absorbed
    if dim_id not in dim_order or depth <= block_depth(dim_order.index(dim_id)):
        rng = f"0:{di.dim_size}"
    elif not ltile_slot_open:
        start = f"i_block_{dim_id} * {block_stride}"
        rng = f"{start}:{start} + {block_stride}"
    else:
        start = f"i_block_{dim_id} * {block_stride} + i_ltile_{dim_id} * {logical}"
        rng = f"{start}:{start} + {logical}"
    return rng
