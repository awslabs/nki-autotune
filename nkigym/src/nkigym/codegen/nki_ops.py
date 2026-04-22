"""NKI op rendering: ISA calls + PSUM memsets."""

from nkigym.codegen.buffers import producer_op, producer_op_tiles, psum_tile_count, psum_tile_slice, sbuf_buffer
from nkigym.codegen.compute_skip import record_op_delta, snapshot_before_lengths, wrap_skip_groups
from nkigym.codegen.dma import (
    dma_load_line,
    dma_store_line,
    dma_transpose_line,
    has_output_ptile_dims,
    inline_stage_line,
    ptile_loop_dims,
)
from nkigym.codegen.group_loops import DepthPlan
from nkigym.codegen.online_fusion import render_online_fusion_op
from nkigym.codegen.reduction import apply_reduction_plan, reduced_outputs_with_multichunk, scratch_shape
from nkigym.codegen.sbuf_buffer import AxisAccess
from nkigym.kernel_ir import KernelIR
from nkigym.kernel_ir.context.context import TensorInfo
from nkigym.kernel_ir.validate.emission import Placement, material_blocking_dims, op_emission_placement
from nkigym.ops.base import NKIOp
from nkigym.ops.dma import NKIDMATranspose, NKILoad, NKIStore
from nkigym.ops.online_fusion_chain import NKIOnlineFusionChain


def render_nki_ops(ir: KernelIR, op_to_group: dict[int, int], staged: set[str]) -> tuple[DepthPlan, DepthPlan]:
    """Plan ISA calls, per-op stages, and PSUM memsets.

    Tracks which lines each op contributes to ``before_plan`` via
    a per-op delta record so ``wrap_skip_groups`` can split the
    innermost-body lines of ``skip_spec``-annotated groups into
    the three-state classifier branches.
    """
    before_plan: DepthPlan = {}
    after_plan: DepthPlan = {}
    memo: dict[int, Placement] = {}
    per_op_lines: dict[int, dict[int, list[str]]] = {}
    for gi, group in enumerate(ir.graph.groups):
        for op in group.ops:
            baseline = snapshot_before_lengths(before_plan, gi)
            _render_one_op(ir, op, gi, op_to_group, staged, memo, before_plan, after_plan)
            record_op_delta(before_plan, gi, baseline, op, per_op_lines)
    wrap_skip_groups(ir, before_plan, after_plan, per_op_lines)
    return before_plan, after_plan


def _render_one_op(
    ir: KernelIR,
    op: NKIOp,
    gi: int,
    op_to_group: dict[int, int],
    staged: set[str],
    memo: dict[int, Placement],
    before_plan: DepthPlan,
    after_plan: DepthPlan,
) -> None:
    """Render one op into ``before_plan`` / ``after_plan``."""
    op_cls = type(op)
    if issubclass(op_cls, (NKILoad, NKIStore, NKIDMATranspose)):
        _render_dma_op(ir, op, gi, op_to_group, staged, memo, before_plan, after_plan)
    elif issubclass(op_cls, NKIOnlineFusionChain):
        render_online_fusion_op(ir, op, gi, before_plan)
    else:
        context = ir.context
        dim_order = ir.graph.groups[gi].dim_order
        placement = op_emission_placement(context, ir.graph, op, gi, op_to_group, staged, memo)
        reduced = reduced_outputs_with_multichunk(ir, op) if op_cls.ISA_LOC != "psum" else []
        scratch_override = {r.role: f"{r.tensor_name}_chunk" for r in reduced}
        block_lines = list(_render_op_block(ir, op, gi, staged, placement, scratch_override))
        if reduced:
            block_lines = apply_reduction_plan(ir, op, gi, reduced, placement, block_lines, before_plan)
        elif op_cls.ISA_LOC == "psum":
            block_lines = _apply_psum_plan(ir, op, gi, dim_order, staged, placement, block_lines, before_plan)
        target = before_plan if placement.phase == "before" else after_plan
        target.setdefault(gi, {}).setdefault(placement.depth, []).extend(block_lines)


def _render_dma_op(
    ir: KernelIR,
    op: NKIOp,
    gi: int,
    op_to_group: dict[int, int],
    staged: set[str],
    memo: dict[int, Placement],
    before_plan: DepthPlan,
    after_plan: DepthPlan,
) -> None:
    """Render a ``NKILoad`` / ``NKIStore`` / ``NKIDMATranspose`` op at its own emission slot."""
    context = ir.context
    placement = op_emission_placement(context, ir.graph, op, gi, op_to_group, staged, memo)
    if isinstance(op, NKIStore):
        line = dma_store_line(ir, op, gi, placement.depth)
    elif isinstance(op, NKIDMATranspose):
        line = dma_transpose_line(ir, op, gi, placement.depth)
    else:
        line = dma_load_line(ir, op, gi, placement.depth)
    target = before_plan if placement.phase == "before" else after_plan
    target.setdefault(gi, {}).setdefault(placement.depth, []).append(line)


def _apply_psum_plan(
    ir: KernelIR,
    op: NKIOp,
    gi: int,
    dim_order: list[str],
    staged: set[str],
    placement: Placement,
    block_lines: list[str],
    before_plan: DepthPlan,
) -> list[str]:
    """Handle PSUM memsets + inline stage lines; mutates ``before_plan`` for material-blocking memsets."""
    context = ir.context
    op_cls = type(op)
    material = material_blocking_dims(context, op, dim_order)
    memset_lines: list[str] = []
    if op_cls.PSUM_DTYPE == "float32":
        for oname in context.op_outputs.get(op, []):
            memset_lines.extend(_memset_lines(ir, oname))
    if material:
        i_min = min(dim_order.index(d) for d in material)
        before_plan.setdefault(gi, {}).setdefault(i_min, []).extend(memset_lines)
    else:
        block_lines = memset_lines + block_lines
        block_lines.extend(_inline_stage_lines(ir, op, gi, dim_order, staged, placement))
    return block_lines


def _inline_stage_lines(
    ir: KernelIR, op: NKIOp, group_idx: int, dim_order: list[str], staged: set[str], placement: Placement
) -> list[str]:
    """Stage lines for a non-material-blocking PSUM producer at the producer's Placement."""
    result: list[str] = []
    if not has_output_ptile_dims(ir, op):
        for oname in ir.context.op_outputs.get(op, []):
            if oname in staged:
                result.append(inline_stage_line(ir, group_idx, op, oname, dim_order, placement.depth))
    return result


def _memset_lines(ir: KernelIR, tensor_name: str) -> list[str]:
    """Emit ``nisa.memset`` over every PSUM tile of *tensor_name*."""
    tinfo = ir.context.logical_tensors[tensor_name]
    count = psum_tile_count(ir, tensor_name, tinfo)
    slice_expr = psum_tile_slice(ir, tensor_name, tinfo)
    list_prefix = "" if count == 1 else f"[i_pt_{tensor_name}]"
    body = f"nisa.memset(psum_{tensor_name}{list_prefix}{slice_expr}, 0.0)"
    return [body] if count == 1 else [f"for i_pt_{tensor_name} in range({count}):", f"    {body}"]


def _render_op_block(
    ir: KernelIR,
    op: NKIOp,
    group_idx: int,
    staged: set[str],
    placement: Placement,
    scratch_override: dict[str, str] | None = None,
) -> list[str]:
    """Render one op's ISA call wrapped in ``i_ptile_{d}`` loops, with per-ptile stage.

    ``scratch_override`` maps output-role → scratch tensor name.
    For roles in the map, the emitted destination writes to a
    direct-alloc SBUF scratch (``sbuf_<scratch_name>[0:P, 0:F]``)
    instead of the role's normal SBUF buffer — used by the
    multi-chunk reduction codegen to capture per-chunk output
    before the combine step.
    """
    lines: list[str] = []
    output_ptile, blocking_ptile = _partition_ptile_dims(ir, op)
    ptile_dims = output_ptile + blocking_ptile
    call_line = _isa_call_line(ir, group_idx, op, staged, ptile_dims, placement, scratch_override)
    for depth, (dim_id, count) in enumerate(output_ptile):
        lines.append("    " * depth + f"for i_ptile_{dim_id} in range({count}):")
    output_indent = "    " * len(output_ptile)
    for depth, (dim_id, count) in enumerate(blocking_ptile):
        lines.append(output_indent + "    " * depth + f"for i_ptile_{dim_id} in range({count}):")
    body_indent = output_indent + "    " * len(blocking_ptile)
    lines.append(body_indent + call_line)
    if output_ptile:
        stage_lines = _ptile_stage_lines(ir, op, group_idx, staged, output_ptile, placement)
        lines.extend(output_indent + line for line in stage_lines)
    return lines


def _partition_ptile_dims(ir: KernelIR, op: NKIOp) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """Split this op's ptile dims into (output, blocking) by its blocking-dim set."""
    blocking = ir.context.op_blocking_dims.get(op, set())
    output: list[tuple[str, int]] = []
    accum: list[tuple[str, int]] = []
    for dim_id, count in ptile_loop_dims(ir, op):
        (accum if dim_id in blocking else output).append((dim_id, count))
    return output, accum


def _ptile_stage_lines(
    ir: KernelIR, op: NKIOp, group_idx: int, staged: set[str], ptile_dims: list[tuple[str, int]], placement: Placement
) -> list[str]:
    """Per-ptile PSUM→SBUF stage lines for PSUM outputs that need staging."""
    _ = placement
    context = ir.context
    ptile_set = {dim_id for dim_id, _ in ptile_dims}
    op_cls = type(op)
    outputs = context.op_outputs.get(op, []) if op_cls.ISA_LOC == "psum" else []
    lines: list[str] = []
    for oname in outputs:
        if oname not in staged:
            continue
        tinfo = context.logical_tensors[oname]
        lines.append(_per_ptile_stage_call(ir, group_idx, op, oname, tinfo, ptile_set, placement))
    return lines


def _per_ptile_stage_call(
    ir: KernelIR,
    group_idx: int,
    op: NKIOp,
    tensor_name: str,
    tinfo: TensorInfo,
    ptile_dims: set[str],
    placement: Placement,
) -> str:
    """Emit ``nisa.tensor_copy`` from PSUM into the one SBUF tile produced this ptile iteration."""
    buf = sbuf_buffer(ir, tensor_name)
    p_access, f_access = _op_axis_access(ir, group_idx, op, tensor_name, tinfo, ptile_dims, placement, staging=False)
    sbuf_expr = buf.get_tile(p_access, f_access)
    psum_expr = _psum_access(ir, tensor_name, tinfo, ptile_dims)
    return f"nisa.tensor_copy({sbuf_expr}, {psum_expr})"


def _isa_call_line(
    ir: KernelIR,
    group_idx: int,
    op: NKIOp,
    staged: set[str],
    ptile_dims: list[tuple[str, int]],
    placement: Placement,
    scratch_override: dict[str, str] | None = None,
) -> str:
    """Format the nisa.* ISA call string for one op."""
    context = ir.context
    op_cls = type(op)
    ptile_lookup = {dim_id for dim_id, _ in ptile_dims}
    dst_map = _dst_exprs(ir, group_idx, op, ptile_lookup, placement, scratch_override)
    dst = next(iter(dst_map.values()))
    operands = _operand_exprs(ir, group_idx, op, staged, ptile_lookup, placement)
    scalar_kwargs = dict(context.op_kwargs.get(op, {}))
    _rewrite_tensor_valued_scalars(ir, group_idx, op, scalar_kwargs, staged, ptile_lookup, placement)
    for ax_name, expr in dst_map.items():
        scalar_kwargs[f"__dst_{ax_name}"] = expr
    _inject_tile_geometry(ir, op, scalar_kwargs, ptile_lookup, placement)
    return op_cls.format_isa_call(dst, operands, scalar_kwargs)


def _inject_tile_geometry(
    ir: KernelIR, op: NKIOp, scalar_kwargs: dict[str, str], ptile_dims: set[str], placement: Placement
) -> None:
    """Inject per-operand-axis tile-start and tile-size into ``scalar_kwargs``."""
    context = ir.context
    op_cls = type(op)
    inputs = context.op_inputs.get(op, {})
    op_tiles = context.op_tile_sizes.get(op, {})
    dim_order = _group_dim_order(ir, op)
    for role, axes in op_cls.OPERAND_AXES.items():
        tensor_name = inputs.get(role)
        if tensor_name is None or tensor_name not in context.logical_tensors:
            continue
        tinfo = context.logical_tensors[tensor_name]
        for ax_name, dim_id in zip(axes, tinfo.dim_ids):
            scalar_kwargs[f"__tile_start_{ax_name}"] = _tile_start_expr(ir, dim_id, ptile_dims, placement, dim_order)
            scalar_kwargs[f"__tile_size_{ax_name}"] = str(
                op_tiles.get(dim_id, context.dimensions[dim_id].physical_tile_size)
            )


def _group_dim_order(ir: KernelIR, op: NKIOp) -> list[str]:
    """Return the dim_order of the fusion group containing ``op``."""
    for group in ir.graph.groups:
        if op in group.ops:
            return group.dim_order
    raise ValueError(f"op {op!r} not in any fusion group")


def _tile_start_expr(
    ir: KernelIR, dim_id: str, ptile_dims: set[str], placement: Placement, dim_order: list[str]
) -> str:
    """Global element-offset start expression for one dim."""
    context = ir.context
    di = context.dimensions[dim_id]
    tpb = context.ltiles_per_block.get(dim_id, 1)
    logical = di.logical_tile_size
    phys = di.physical_tile_size
    block_stride = logical * tpb
    terms: list[str] = []
    n = len(dim_order)
    if dim_id in dim_order:
        pos = dim_order.index(dim_id)
        if placement.loop_open(pos):
            terms.append(f"i_block_{dim_id} * {block_stride}" if block_stride > 1 else f"i_block_{dim_id}")
        if placement.loop_open(n + pos):
            terms.append(f"i_ltile_{dim_id} * {logical}" if logical > 1 else f"i_ltile_{dim_id}")
    if dim_id in ptile_dims:
        terms.append(f"i_ptile_{dim_id} * {phys}" if phys > 1 else f"i_ptile_{dim_id}")
    return " + ".join(terms) if terms else "0"


def _rewrite_tensor_valued_scalars(
    ir: KernelIR,
    group_idx: int,
    op: NKIOp,
    scalar_kwargs: dict[str, str],
    staged: set[str],
    ptile_dims: set[str],
    placement: Placement,
) -> None:
    """Rewrite scalar kwargs whose value is a tensor name to a buffer access."""
    context = ir.context
    op_cls = type(op)
    operand_roles = set(op_cls.OPERAND_AXES)
    for name, expr in list(scalar_kwargs.items()):
        if name in operand_roles or name.startswith("__"):
            continue
        if expr not in context.logical_tensors:
            continue
        tinfo = context.logical_tensors[expr]
        producer = producer_op(ir, expr)
        is_psum = producer is not None and type(producer).ISA_LOC == "psum" and expr not in staged
        scalar_kwargs[name] = _access_expr(ir, group_idx, op, expr, tinfo, ptile_dims, is_psum, placement)


def _dst_exprs(
    ir: KernelIR,
    group_idx: int,
    op: NKIOp,
    ptile_dims: set[str],
    placement: Placement,
    scratch_override: dict[str, str] | None = None,
) -> dict[str, str]:
    """Destination expressions keyed by the op's ``OUTPUT_AXES`` names."""
    context = ir.context
    op_cls = type(op)
    outputs = context.op_outputs.get(op, [])
    overrides = scratch_override or {}
    result: dict[str, str] = {}
    is_psum = op_cls.ISA_LOC == "psum"
    for ax_name, oname in zip(op_cls.OUTPUT_AXES, outputs):
        if ax_name in overrides:
            tinfo = context.logical_tensors[oname]
            p, f = scratch_shape(context, tinfo)
            result[ax_name] = f"sbuf_{overrides[ax_name]}[0:{p}, 0:{f}]"
        else:
            tinfo = context.logical_tensors[oname]
            result[ax_name] = _access_expr(ir, group_idx, op, oname, tinfo, ptile_dims, is_psum, placement)
    return result


def _operand_exprs(
    ir: KernelIR, group_idx: int, op: NKIOp, staged: set[str], ptile_dims: set[str], placement: Placement
) -> dict[str, str]:
    """Operand expressions keyed by the op's input role names."""
    context = ir.context
    op_cls = type(op)
    inputs = context.op_inputs.get(op, {})
    result: dict[str, str] = {}
    for role, tensor_name in inputs.items():
        if role not in op_cls.OPERAND_AXES:
            continue
        tinfo = context.logical_tensors.get(tensor_name)
        if tinfo is None:
            continue
        producer = producer_op(ir, tensor_name)
        is_psum = False
        if producer is not None and type(producer).ISA_LOC == "psum":
            input_loc = op_cls.INPUT_LOCS.get(role, "sbuf")
            if input_loc != "sbuf" or tensor_name not in staged:
                is_psum = True
        result[role] = _access_expr(ir, group_idx, op, tensor_name, tinfo, ptile_dims, is_psum, placement)
    return result


def _access_expr(
    ir: KernelIR,
    group_idx: int,
    op: NKIOp,
    tensor_name: str,
    tinfo: TensorInfo,
    ptile_dims: set[str],
    is_psum: bool,
    placement: Placement,
) -> str:
    """Build the full buffer access string — PSUM flat slice or SBUF list-of-2D tile."""
    if is_psum:
        expr = _psum_access(ir, tensor_name, tinfo, ptile_dims)
    else:
        expr = _sbuf_access(ir, group_idx, op, tensor_name, tinfo, ptile_dims, placement)
    return expr


def _sbuf_access(
    ir: KernelIR,
    group_idx: int,
    op: NKIOp,
    tensor_name: str,
    tinfo: TensorInfo,
    ptile_dims: set[str],
    placement: Placement,
) -> str:
    """One-tile SBUF access via ``SbufBuffer.get_tile``."""
    buf = sbuf_buffer(ir, tensor_name)
    p_access, f_access = _op_axis_access(ir, group_idx, op, tensor_name, tinfo, ptile_dims, placement, staging=False)
    return buf.get_tile(p_access, f_access)


def _op_axis_access(
    ir: KernelIR,
    group_idx: int,
    op: NKIOp,
    tensor_name: str,
    tinfo: TensorInfo,
    ptile_dims: set[str],
    placement: Placement,
    staging: bool,
) -> tuple[AxisAccess, AxisAccess]:
    """Return ``(p_access, f_access)`` for an op-body access at ``placement``."""
    dim_ids = tinfo.dim_ids
    p_ptile = _op_ptile_var(ir, op, dim_ids[0], ptile_dims)
    p_axis = _inner_body_access(ir, group_idx, tensor_name, dim_ids[0], placement, ptile=p_ptile)
    if len(dim_ids) == 2:
        d_f = dim_ids[1]
        ptile_f = None if staging else _op_ptile_var(ir, op, d_f, ptile_dims)
        f_axis = _inner_body_access(ir, group_idx, tensor_name, d_f, placement, ptile=ptile_f)
    else:
        f_axis = AxisAccess(block="0", ltile="0")
    return p_axis, f_axis


def _op_ptile_var(ir: KernelIR, op: NKIOp, dim_id: str, ptile_dims: set[str]) -> str | None:
    """Return ``i_ptile_{d}`` if this op iterates ``dim_id`` at ptile granularity, else ``None``."""
    context = ir.context
    di = context.dimensions[dim_id]
    op_tile = context.op_tile_sizes.get(op, {}).get(dim_id, di.physical_tile_size)
    op_ptiles = op_tile // di.physical_tile_size
    total_ptiles = di.num_ptiles
    return f"i_ptile_{dim_id}" if dim_id in ptile_dims and total_ptiles > op_ptiles else None


def _inner_body_access(
    ir: KernelIR, group_idx: int, tensor_name: str, dim_id: str, placement: Placement, ptile: str | None = None
) -> AxisAccess:
    """``AxisAccess`` for a tensor at ``placement``."""
    placements = ir.graph.groups[group_idx].tensor_placements
    tier = placements.get(("sbuf", tensor_name, dim_id), "per_tile")
    dim_order = ir.graph.groups[group_idx].dim_order
    block = "0"
    ltile = "0"
    if dim_id in dim_order:
        pos = dim_order.index(dim_id)
        n = len(dim_order)
        if tier == "full" and placement.loop_open(pos):
            block = f"i_block_{dim_id}"
        if tier in ("per_block", "full") and placement.loop_open(n + pos):
            ltile = f"i_ltile_{dim_id}"
    return AxisAccess(block=block, ltile=ltile, ptile=ptile)


def _psum_access(ir: KernelIR, tensor_name: str, tinfo: TensorInfo, ptile_dims: set[str]) -> str:
    """Index expression into a PSUM allocation."""
    slice_expr = psum_tile_slice(ir, tensor_name, tinfo)
    list_idx = _psum_list_index(ir, tensor_name, tinfo, ptile_dims)
    return f"psum_{tensor_name}{slice_expr}" if list_idx is None else f"psum_{tensor_name}[{list_idx}]{slice_expr}"


def _psum_list_index(ir: KernelIR, tensor_name: str, tinfo: TensorInfo, ptile_dims: set[str]) -> str | None:
    """Row-major flat list index into a PSUM list of 2D tiles."""
    context = ir.context
    op_tiles = producer_op_tiles(ir, tensor_name)
    total = psum_tile_count(ir, tensor_name, tinfo)
    expr: str | None = None
    if total > 1:
        terms: list[str] = []
        stride = total
        for d in tinfo.dim_ids:
            di = context.dimensions[d]
            op_tile = op_tiles.get(d, di.physical_tile_size)
            slots = max(1, di.logical_tile_size // op_tile)
            stride //= slots
            idx = f"i_ptile_{d}" if d in ptile_dims else "0"
            if slots > 1:
                terms.append(f"{idx} * {stride}" if stride > 1 else idx)
        expr = " + ".join(terms) if terms else "0"
    return expr
