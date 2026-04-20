"""NKI op rendering: ISA calls + PSUM memsets.

Each op's ISA block emits at a per-op ``Placement`` computed by
``emission.op_emission_placement`` — the narrowest slot where
every material loop over ``dims(op) & dim_order`` is open AND
every intra-group producer's stage has fired. PSUM memsets for
material-blocking producers fire at the outermost blocking
loop's depth so the accumulator is zeroed once per output tile.
Ops whose tile is smaller than a dim's logical tile are wrapped
in per-dim ``i_ptile_{d}`` loops.

SBUF access goes through ``SbufBuffer.get_tile(AxisAccess)`` —
always a ``sbuf_X[p_slot][f_slot][0:P, f_start:f_end]`` form with
genuine 2D memref semantics. PSUM access stays as a direct flat
``psum_X[0:P, 0:F]`` slice (optionally wrapped in a list index
when the producer writes multiple tiles concurrently). List-
factor loop vars (``i_block_{d}`` / ``i_ltile_{d}``) collapse to
``"0"`` when the op's Placement sits outside the corresponding
loop.
"""

from nkigym.codegen.buffers import producer_op_tiles, psum_tile_count, psum_tile_slice, sbuf_buffer
from nkigym.codegen.dma import has_output_ptile_dims, inline_stage_line, ptile_loop_dims
from nkigym.codegen.group_loops import DepthPlan
from nkigym.codegen.sbuf_buffer import AxisAccess
from nkigym.kernel_ir import KernelIR
from nkigym.kernel_ir.dim_analysis import TensorInfo, op_blocking_dims
from nkigym.kernel_ir.emission import Placement, material_blocking_dims, op_emission_placement


def render_nki_ops(ir: KernelIR, op_to_group: dict[int, int], staged: set[str]) -> tuple[DepthPlan, DepthPlan]:
    """Plan ISA calls, per-op stages, and PSUM memsets; return (before, after) plans.

    Each op's ISA block is placed at its own ``Placement``. Non-
    material-blocking PSUM producers inline their stage after the
    ISA block at the same slot; material-blocking producers get
    their memset at the outermost blocking loop's depth (before)
    and are staged via ``render_psum_staging`` at the same depth
    in the after-plan.
    """
    graph = ir.op_graph
    before_plan: DepthPlan = {}
    after_plan: DepthPlan = {}
    memo: dict[int, Placement] = {}
    for op_idx, op_cls in enumerate(graph.op_classes):
        group_idx = op_to_group[op_idx]
        dim_order = ir.fusion_groups[group_idx].dim_order
        placement = op_emission_placement(ir, op_idx, group_idx, op_to_group, staged, memo)
        block_lines = list(_render_op_block(ir, op_idx, op_to_group, staged, placement))
        if op_cls.ISA_LOC == "psum":
            material = material_blocking_dims(ir, op_idx, dim_order)
            memset_lines: list[str] = []
            if op_cls.PSUM_DTYPE == "float32":
                """Memset only the fp32-PSUM-accumulating ops (matmul). Transpose
                and elementwise PSUM writers use bf16 PSUM where real hardware
                rejects memset (NCC_IBIR441); and those ops overwrite their PSUM
                tile each call rather than accumulating, so zeroing is unneeded.
                """
                for oname in graph.op_tensors[op_idx][1]:
                    memset_lines.extend(_memset_lines(ir, oname))
            if material:
                i_min = min(dim_order.index(d) for d in material)
                before_plan.setdefault(group_idx, {}).setdefault(i_min, []).extend(memset_lines)
            else:
                block_lines = memset_lines + block_lines
                block_lines.extend(_inline_stage_lines(ir, op_idx, group_idx, dim_order, staged, placement))
        target = before_plan if placement.phase == "before" else after_plan
        target.setdefault(group_idx, {}).setdefault(placement.depth, []).extend(block_lines)
    return before_plan, after_plan


def _inline_stage_lines(
    ir: KernelIR, op_idx: int, group_idx: int, dim_order: list[str], staged: set[str], placement: Placement
) -> list[str]:
    """Stage lines for a non-material-blocking PSUM producer; emitted at the producer's Placement."""
    return (
        []
        if has_output_ptile_dims(ir, op_idx)
        else [
            inline_stage_line(ir, group_idx, op_idx, oname, dim_order, placement.depth)
            for oname in ir.op_graph.op_tensors[op_idx][1]
            if oname in staged
        ]
    )


def _memset_lines(ir: KernelIR, tensor_name: str) -> list[str]:
    """Emit ``nisa.memset`` over every PSUM tile of *tensor_name* — one call, optionally inside a Python for loop."""
    tinfo = ir.dim_analysis.tensors[tensor_name]
    count = psum_tile_count(ir, tensor_name, tinfo)
    slice_expr = psum_tile_slice(ir, tensor_name, tinfo)
    list_prefix = "" if count == 1 else f"[i_pt_{tensor_name}]"
    body = f"nisa.memset(psum_{tensor_name}{list_prefix}{slice_expr}, 0.0)"
    return [body] if count == 1 else [f"for i_pt_{tensor_name} in range({count}):", f"    {body}"]


def _render_op_block(
    ir: KernelIR, op_idx: int, op_to_group: dict[int, int], staged: set[str], placement: Placement
) -> list[str]:
    """Render one op's ISA call wrapped in ``i_ptile_{d}`` loops, with a fused per-ptile stage.

    Ptile dims split into output (non-blocking) and blocking by
    the op's ``BLOCKING_AXES``. Output ptile loops outer, blocking
    ptile loops inner. Per-op PSUM→SBUF stage fires INSIDE output
    ptile loops but OUTSIDE blocking ptile loops — the generic
    math-validity rule: read PSUM only after every blocking loop
    that accumulates into it has closed.
    """
    lines: list[str] = []
    group_idx = op_to_group[op_idx]
    output_ptile, blocking_ptile = _partition_ptile_dims(ir, op_idx)
    ptile_dims = output_ptile + blocking_ptile
    call_line = _isa_call_line(ir, group_idx, op_idx, staged, ptile_dims, placement)
    for depth, (dim_id, count) in enumerate(output_ptile):
        lines.append("    " * depth + f"for i_ptile_{dim_id} in range({count}):")
    output_indent = "    " * len(output_ptile)
    for depth, (dim_id, count) in enumerate(blocking_ptile):
        lines.append(output_indent + "    " * depth + f"for i_ptile_{dim_id} in range({count}):")
    body_indent = output_indent + "    " * len(blocking_ptile)
    lines.append(body_indent + call_line)
    if output_ptile:
        stage_lines = _ptile_stage_lines(ir, op_idx, op_to_group, staged, output_ptile, placement)
        lines.extend(output_indent + line for line in stage_lines)
    return lines


def _partition_ptile_dims(ir: KernelIR, op_idx: int) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """Split this op's ptile dims into (output, blocking) by its ``BLOCKING_AXES``."""
    op_cls = ir.op_graph.op_classes[op_idx]
    blocking = op_blocking_dims(op_cls, ir.dim_analysis.per_op_axis_maps[op_idx])
    output: list[tuple[str, int]] = []
    accum: list[tuple[str, int]] = []
    for dim_id, count in ptile_loop_dims(ir, op_idx):
        (accum if dim_id in blocking else output).append((dim_id, count))
    return output, accum


def _ptile_stage_lines(
    ir: KernelIR,
    op_idx: int,
    op_to_group: dict[int, int],
    staged: set[str],
    ptile_dims: list[tuple[str, int]],
    placement: Placement,
) -> list[str]:
    """Per-ptile PSUM→SBUF stage lines for PSUM outputs that need staging."""
    graph = ir.op_graph
    op_cls = graph.op_classes[op_idx]
    ptile_set = {dim_id for dim_id, _ in ptile_dims}
    outputs = graph.op_tensors[op_idx][1] if op_cls.ISA_LOC == "psum" else ()
    da = ir.dim_analysis
    group_idx = op_to_group[op_idx]
    lines: list[str] = []
    for oname in outputs:
        if oname not in staged:
            continue
        tinfo = da.tensors[oname]
        lines.append(_per_ptile_stage_call(ir, group_idx, op_idx, oname, tinfo, ptile_set, placement))
    return lines


def _per_ptile_stage_call(
    ir: KernelIR,
    group_idx: int,
    op_idx: int,
    tensor_name: str,
    tinfo: TensorInfo,
    ptile_dims: set[str],
    placement: Placement,
) -> str:
    """Emit ``nisa.tensor_copy`` from PSUM into the one SBUF tile produced this ptile iteration."""
    buf = sbuf_buffer(ir, tensor_name)
    p_access, f_access = _op_axis_access(
        ir, group_idx, op_idx, tensor_name, tinfo, ptile_dims, placement, staging=False
    )
    sbuf_expr = buf.get_tile(p_access, f_access)
    psum_expr = _psum_access(ir, tensor_name, tinfo, ptile_dims)
    return f"nisa.tensor_copy({sbuf_expr}, {psum_expr})"


def _isa_call_line(
    ir: KernelIR, group_idx: int, op_idx: int, staged: set[str], ptile_dims: list[tuple[str, int]], placement: Placement
) -> str:
    """Format the nisa.* ISA call string for one op."""
    op_cls = ir.op_graph.op_classes[op_idx]
    ptile_lookup = {dim_id for dim_id, _ in ptile_dims}
    dst_map = _dst_exprs(ir, group_idx, op_idx, ptile_lookup, placement)
    dst = next(iter(dst_map.values()))
    operands = _operand_exprs(ir, group_idx, op_idx, staged, ptile_lookup, placement)
    scalar_kwargs = dict(ir.op_graph.op_all_kwargs[op_idx])
    _rewrite_tensor_valued_scalars(ir, group_idx, op_idx, scalar_kwargs, staged, ptile_lookup, placement)
    for ax_name, expr in dst_map.items():
        scalar_kwargs[f"__dst_{ax_name}"] = expr
    _inject_tile_geometry(ir, op_idx, scalar_kwargs, ptile_lookup, placement)
    return op_cls.format_isa_call(dst, operands, scalar_kwargs)


def _inject_tile_geometry(
    ir: KernelIR, op_idx: int, scalar_kwargs: dict[str, str], ptile_dims: set[str], placement: Placement
) -> None:
    """Inject per-operand-axis tile-start and tile-size into ``scalar_kwargs``.

    Emits ``i_block_{d}`` / ``i_ltile_{d}`` in the start expression
    only when the corresponding loop is open at ``placement``; a
    closed loop binds its iteration variable is nothing (the
    outer scope has exactly one value for that iteration, which
    is the first slot).
    """
    da = ir.dim_analysis
    op_cls = ir.op_graph.op_classes[op_idx]
    inputs, _outputs = ir.op_graph.op_tensors[op_idx]
    op_tiles = da.op_tile_sizes[op_idx]
    dim_order = (
        ir.fusion_groups[ir.op_graph and _find_group(ir, op_idx)].dim_order if False else _group_dim_order(ir, op_idx)
    )
    for role, axes in op_cls.OPERAND_AXES.items():
        tensor_name = inputs.get(role)
        if tensor_name is None or tensor_name not in da.tensors:
            continue
        tinfo = da.tensors[tensor_name]
        for ax_name, dim_id in zip(axes, tinfo.dim_ids):
            scalar_kwargs[f"__tile_start_{ax_name}"] = _tile_start_expr(ir, dim_id, ptile_dims, placement, dim_order)
            scalar_kwargs[f"__tile_size_{ax_name}"] = str(op_tiles.get(dim_id, da.dims[dim_id].physical_tile_size))


def _group_dim_order(ir: KernelIR, op_idx: int) -> list[str]:
    """Return the dim_order of the fusion group containing ``op_idx``."""
    for group in ir.fusion_groups:
        if op_idx in group.op_indices:
            return group.dim_order
    raise ValueError(f"op_idx {op_idx} not in any fusion group")


def _tile_start_expr(
    ir: KernelIR, dim_id: str, ptile_dims: set[str], placement: Placement, dim_order: list[str]
) -> str:
    """Global element-offset start expression for one dim of the current iteration.

    ``i_block_{d}`` / ``i_ltile_{d}`` are included only when the
    corresponding loop is open at ``placement`` — closed loops
    contribute nothing (the outer binding is the single iteration
    the loop would have had).
    """
    di = ir.dim_analysis.dims[dim_id]
    tpb = ir.ltiles_per_block.get(dim_id, 1)
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
    op_idx: int,
    scalar_kwargs: dict[str, str],
    staged: set[str],
    ptile_dims: set[str],
    placement: Placement,
) -> None:
    """Rewrite scalar kwargs whose value is a tensor name to a buffer access."""
    da = ir.dim_analysis
    graph = ir.op_graph
    op_cls = graph.op_classes[op_idx]
    operand_roles = set(op_cls.OPERAND_AXES)
    for name, expr in list(scalar_kwargs.items()):
        if name in operand_roles or name.startswith("__"):
            continue
        if expr not in da.tensors:
            continue
        tinfo = da.tensors[expr]
        is_psum = graph.producer_isa_loc(expr) == "psum" and expr not in staged
        scalar_kwargs[name] = _access_expr(ir, group_idx, op_idx, expr, tinfo, ptile_dims, is_psum, placement)


def _dst_exprs(ir: KernelIR, group_idx: int, op_idx: int, ptile_dims: set[str], placement: Placement) -> dict[str, str]:
    """Destination expressions keyed by the op's ``OUTPUT_AXES`` names."""
    graph = ir.op_graph
    op_cls = graph.op_classes[op_idx]
    outputs = graph.op_tensors[op_idx][1]
    result: dict[str, str] = {}
    is_psum = op_cls.ISA_LOC == "psum"
    for ax_name, oname in zip(op_cls.OUTPUT_AXES, outputs):
        tinfo = ir.dim_analysis.tensors[oname]
        result[ax_name] = _access_expr(ir, group_idx, op_idx, oname, tinfo, ptile_dims, is_psum, placement)
    return result


def _operand_exprs(
    ir: KernelIR, group_idx: int, op_idx: int, staged: set[str], ptile_dims: set[str], placement: Placement
) -> dict[str, str]:
    """Operand expressions keyed by the op's input role names."""
    graph = ir.op_graph
    da = ir.dim_analysis
    op_cls = graph.op_classes[op_idx]
    inputs = graph.op_tensors[op_idx][0]
    result: dict[str, str] = {}
    for role, tensor_name in inputs.items():
        if role not in op_cls.OPERAND_AXES:
            continue
        tinfo = da.tensors.get(tensor_name)
        if tinfo is None:
            continue
        is_psum = False
        if graph.producer_isa_loc(tensor_name) == "psum":
            input_loc = op_cls.INPUT_LOCS.get(role, "sbuf")
            if input_loc != "sbuf" or tensor_name not in staged:
                is_psum = True
        result[role] = _access_expr(ir, group_idx, op_idx, tensor_name, tinfo, ptile_dims, is_psum, placement)
    return result


def _access_expr(
    ir: KernelIR,
    group_idx: int,
    op_idx: int,
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
        expr = _sbuf_access(ir, group_idx, op_idx, tensor_name, tinfo, ptile_dims, placement)
    return expr


def _sbuf_access(
    ir: KernelIR,
    group_idx: int,
    op_idx: int,
    tensor_name: str,
    tinfo: TensorInfo,
    ptile_dims: set[str],
    placement: Placement,
) -> str:
    """One-tile SBUF access via ``SbufBuffer.get_tile``.

    List-factor loop vars are bound only when the corresponding
    loop is open at ``placement``; otherwise they collapse to
    ``"0"`` (tier-materialized slot 0).
    """
    buf = sbuf_buffer(ir, tensor_name)
    p_access, f_access = _op_axis_access(
        ir, group_idx, op_idx, tensor_name, tinfo, ptile_dims, placement, staging=False
    )
    return buf.get_tile(p_access, f_access)


def _op_axis_access(
    ir: KernelIR,
    group_idx: int,
    op_idx: int,
    tensor_name: str,
    tinfo: TensorInfo,
    ptile_dims: set[str],
    placement: Placement,
    staging: bool,
) -> tuple[AxisAccess, AxisAccess]:
    """Return ``(p_access, f_access)`` for an op-body access at ``placement``."""
    dim_ids = tinfo.dim_ids
    p_ptile = _op_ptile_var(ir, op_idx, dim_ids[0], ptile_dims)
    p_axis = _inner_body_access(ir, group_idx, tensor_name, dim_ids[0], placement, ptile=p_ptile)
    if len(dim_ids) == 2:
        d_f = dim_ids[1]
        ptile_f = None if staging else _op_ptile_var(ir, op_idx, d_f, ptile_dims)
        f_axis = _inner_body_access(ir, group_idx, tensor_name, d_f, placement, ptile=ptile_f)
    else:
        f_axis = AxisAccess(block="0", ltile="0")
    return p_axis, f_axis


def _op_ptile_var(ir: KernelIR, op_idx: int, dim_id: str, ptile_dims: set[str]) -> str | None:
    """Return ``i_ptile_{d}`` if this op iterates ``dim_id`` at ptile granularity, else ``None``."""
    da = ir.dim_analysis
    di = da.dims[dim_id]
    op_tile = da.op_tile_sizes[op_idx].get(dim_id, di.physical_tile_size)
    op_ptiles = op_tile // di.physical_tile_size
    total_ptiles = di.num_ptiles
    return f"i_ptile_{dim_id}" if dim_id in ptile_dims and total_ptiles > op_ptiles else None


def _inner_body_access(
    ir: KernelIR, group_idx: int, tensor_name: str, dim_id: str, placement: Placement, ptile: str | None = None
) -> AxisAccess:
    """``AxisAccess`` for a tensor at ``placement`` — bind each kept list factor to its loop var iff the loop is open."""
    placements = ir.fusion_groups[group_idx].tensor_placements
    tier = placements.get(("sbuf", tensor_name, dim_id), "per_tile")
    dim_order = ir.fusion_groups[group_idx].dim_order
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
    da = ir.dim_analysis
    op_tiles = producer_op_tiles(ir, tensor_name)
    total = psum_tile_count(ir, tensor_name, tinfo)
    expr: str | None = None
    if total > 1:
        terms: list[str] = []
        stride = total
        for d in tinfo.dim_ids:
            di = da.dims[d]
            op_tile = op_tiles.get(d, di.physical_tile_size)
            slots = max(1, di.logical_tile_size // op_tile)
            stride //= slots
            idx = f"i_ptile_{d}" if d in ptile_dims else "0"
            if slots > 1:
                terms.append(f"{idx} * {stride}" if stride > 1 else idx)
        expr = " + ".join(terms) if terms else "0"
    return expr
