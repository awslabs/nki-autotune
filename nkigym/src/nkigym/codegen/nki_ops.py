"""NKI op rendering: ISA calls + PSUM memsets.

ISA calls emit at the innermost body (depth ``2 * N``); PSUM
memsets for blocking producers emit at depth ``i_min`` so the
blocking accumulator is zeroed before its block loop opens. Ops
whose tile is smaller than a dim's logical tile are wrapped in
per-dim ``i_ptile_{d}`` loops.
"""

from nkigym.codegen.buffers import num_tiles, producer_op_tiles, psum_tile_count, psum_tile_slice
from nkigym.codegen.dma import producer_finished_depth, ptile_loop_dims, sbuf_ptile_slice
from nkigym.codegen.group_loops import DepthPlan
from nkigym.kernel_ir import KernelIR
from nkigym.kernel_ir.dim_analysis import TensorInfo


def render_nki_ops(ir: KernelIR, op_to_group: dict[int, int], staged: set[str]) -> DepthPlan:
    """Plan ISA calls and memsets for every op, keyed by group."""
    graph = ir.op_graph
    plan: DepthPlan = {}

    for op_idx, op_cls in enumerate(graph.op_classes):
        group_idx = op_to_group[op_idx]
        dim_order = ir.fusion_groups[group_idx].dim_order
        n = len(dim_order)

        if op_cls.ISA_LOC == "psum":
            i_min, blocking = producer_finished_depth(ir, op_idx, dim_order)
            if blocking:
                for oname in graph.op_tensors[op_idx][1]:
                    plan.setdefault(group_idx, {}).setdefault(i_min, []).extend(_memset_lines(ir, oname))

        plan.setdefault(group_idx, {}).setdefault(2 * n, []).extend(_render_op_block(ir, op_idx, op_to_group, staged))

    return plan


def _memset_lines(ir: KernelIR, tensor_name: str) -> list[str]:
    """Emit ``nisa.memset`` over every PSUM tile of *tensor_name* — one call, optionally inside a Python for loop."""
    tinfo = ir.dim_analysis.tensors[tensor_name]
    count = psum_tile_count(ir, tensor_name, tinfo)
    slice_expr = psum_tile_slice(ir, tensor_name, tinfo)
    list_prefix = "" if count == 1 else f"[i_pt_{tensor_name}]"
    body = f"nisa.memset(psum_{tensor_name}{list_prefix}{slice_expr}, 0.0)"
    return [body] if count == 1 else [f"for i_pt_{tensor_name} in range({count}):", f"    {body}"]


def _render_op_block(ir: KernelIR, op_idx: int, op_to_group: dict[int, int], staged: set[str]) -> list[str]:
    """Render one op's ISA call wrapped in ``i_ptile_{d}`` loops, with a fused per-ptile stage.

    When the op writes PSUM and has ptile dims, a per-iteration
    ``nisa.tensor_copy(sbuf_dst_slot, psum_tile)`` is appended
    inside the innermost ptile loop so each iteration's PSUM
    output lands in the matching SBUF slot before the next
    iteration overwrites PSUM. This makes list-of-N PSUM
    buffering unnecessary for the baseline — single-buffered
    PSUM + per-iteration stage gives correct results when
    ``psum_tile_count == 1``.
    """
    lines: list[str] = []
    ptile_dims = ptile_loop_dims(ir, op_idx)
    group_idx = op_to_group[op_idx]
    call_line = _isa_call_line(ir, group_idx, op_idx, staged, ptile_dims)
    for depth, (dim_id, count) in enumerate(ptile_dims):
        lines.append("    " * depth + f"for i_ptile_{dim_id} in range({count}):")
    body_indent = "    " * len(ptile_dims)
    lines.append(body_indent + call_line)
    if ptile_dims:
        stage_lines = _ptile_stage_lines(ir, op_idx, op_to_group, staged, ptile_dims)
        lines.extend(body_indent + line for line in stage_lines)
    return lines


def _ptile_stage_lines(
    ir: KernelIR, op_idx: int, op_to_group: dict[int, int], staged: set[str], ptile_dims: list[tuple[str, int]]
) -> list[str]:
    """Per-ptile PSUM→SBUF stage lines for PSUM outputs that need staging.

    Emits one ``nisa.tensor_copy`` per output tensor, indexed by
    the current ptile iteration using ``sbuf_ptile_slice`` so
    the SBUF destination slice accounts for in-scope block/ltile
    loops as well as the narrowing ``i_ptile_{d}`` index. Runs
    inside the innermost ptile loop so only one PSUM tile is
    live at a time.
    """
    graph = ir.op_graph
    op_cls = graph.op_classes[op_idx]
    ptile_set: frozenset[str] = frozenset(dim_id for dim_id, _ in ptile_dims)
    outputs = graph.op_tensors[op_idx][1] if op_cls.ISA_LOC == "psum" else ()
    da = ir.dim_analysis
    group_idx = op_to_group[op_idx]
    dim_order = ir.fusion_groups[group_idx].dim_order
    lines: list[str] = []
    for oname in outputs:
        if oname not in staged:
            continue
        tinfo = da.tensors[oname]
        sbuf_idx = sbuf_ptile_slice(ir, group_idx, oname, tinfo, dim_order, ptile_set)
        psum_idx = psum_tile_slice(ir, oname, tinfo)
        lines.append(f"nisa.tensor_copy(sbuf_{oname}{sbuf_idx}, psum_{oname}{psum_idx})")
    return lines


def _isa_call_line(
    ir: KernelIR, group_idx: int, op_idx: int, staged: set[str], ptile_dims: list[tuple[str, int]]
) -> str:
    """Format the nisa.* ISA call string for one op."""
    op_cls = ir.op_graph.op_classes[op_idx]
    ptile_lookup = {dim_id for dim_id, _ in ptile_dims}
    dst_map = _dst_exprs(ir, group_idx, op_idx, ptile_lookup)
    dst = next(iter(dst_map.values()))
    operands = _operand_exprs(ir, group_idx, op_idx, staged, ptile_lookup)

    scalar_kwargs = dict(ir.op_graph.op_all_kwargs[op_idx])
    _rewrite_tensor_valued_scalars(ir, group_idx, op_idx, scalar_kwargs, staged, ptile_lookup)
    for ax_name, expr in dst_map.items():
        scalar_kwargs[f"__dst_{ax_name}"] = expr
    return op_cls.format_isa_call(dst, operands, scalar_kwargs)


def _rewrite_tensor_valued_scalars(
    ir: KernelIR, group_idx: int, op_idx: int, scalar_kwargs: dict[str, str], staged: set[str], ptile_dims: set[str]
) -> None:
    """Rewrite scalar kwargs whose value is a tensor name to a buffer slice."""
    da = ir.dim_analysis
    graph = ir.op_graph
    op_cls = graph.op_classes[op_idx]
    operand_roles = set(op_cls.OPERAND_AXES)
    for name, expr in list(scalar_kwargs.items()):
        if name in operand_roles or name.startswith("__dst_"):
            continue
        if expr not in da.tensors:
            continue
        tinfo = da.tensors[expr]
        is_psum = graph.producer_isa_loc(expr) == "psum" and expr not in staged
        buf_name = f"psum_{expr}" if is_psum else f"sbuf_{expr}"
        scalar_kwargs[name] = f"{buf_name}{_tile_index_expr(ir, group_idx, op_idx, expr, tinfo, ptile_dims, is_psum)}"


def _dst_exprs(ir: KernelIR, group_idx: int, op_idx: int, ptile_dims: set[str]) -> dict[str, str]:
    """Destination expressions keyed by the op's ``OUTPUT_AXES`` names."""
    graph = ir.op_graph
    op_cls = graph.op_classes[op_idx]
    outputs = graph.op_tensors[op_idx][1]
    result: dict[str, str] = {}
    is_psum = op_cls.ISA_LOC == "psum"
    for ax_name, oname in zip(op_cls.OUTPUT_AXES, outputs):
        tinfo = ir.dim_analysis.tensors[oname]
        idx = _dst_index_expr(ir, group_idx, op_idx, oname, tinfo, ptile_dims, is_psum)
        result[ax_name] = f"{op_cls.ISA_LOC}_{oname}{idx}"
    return result


def _operand_exprs(ir: KernelIR, group_idx: int, op_idx: int, staged: set[str], ptile_dims: set[str]) -> dict[str, str]:
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
        buf_name = f"sbuf_{tensor_name}"
        if graph.producer_isa_loc(tensor_name) == "psum":
            input_loc = op_cls.INPUT_LOCS.get(role, "sbuf")
            if input_loc != "sbuf" or tensor_name not in staged:
                buf_name = f"psum_{tensor_name}"
                is_psum = True

        result[role] = f"{buf_name}{_tile_index_expr(ir, group_idx, op_idx, tensor_name, tinfo, ptile_dims, is_psum)}"

    return result


def _psum_list_index(ir: KernelIR, tensor_name: str, tinfo: TensorInfo, ptile_dims: set[str]) -> str | None:
    """Row-major flat list index into a PSUM list of 2D tiles.

    Returns ``None`` when the PSUM allocation is a single
    ``nl.ndarray`` (no list wrap). A dim with multiple op-tile
    slots contributes ``i_ptile_{d}`` if a ptile loop covers it,
    else ``0``.
    """
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


def _dst_index_expr(
    ir: KernelIR, group_idx: int, op_idx: int, tensor_name: str, tinfo: TensorInfo, ptile_dims: set[str], is_psum: bool
) -> str:
    """Destination index — PSUM list-slice or SBUF 4D slice."""
    expr = (
        _psum_index_expr(ir, tensor_name, tinfo, ptile_dims)
        if is_psum
        else _sbuf_dst_index_expr(ir, group_idx, op_idx, tinfo, ptile_dims)
    )
    return expr


def _tile_index_expr(
    ir: KernelIR, group_idx: int, op_idx: int, tensor_name: str, tinfo: TensorInfo, ptile_dims: set[str], is_psum: bool
) -> str:
    """Operand index — PSUM list-slice or SBUF 4D slice (optionally reshaped)."""
    expr = (
        _psum_index_expr(ir, tensor_name, tinfo, ptile_dims)
        if is_psum
        else _sbuf_tile_index_expr(ir, group_idx, op_idx, tensor_name, tinfo, ptile_dims)
    )
    return expr


def _psum_index_expr(ir: KernelIR, tensor_name: str, tinfo: TensorInfo, ptile_dims: set[str]) -> str:
    """Index expression into a PSUM allocation.

    Single-tile PSUM: ``[0:P, 0:F]``.
    Multi-tile PSUM: ``[list_idx][0:P, 0:F]``.
    """
    slice_expr = psum_tile_slice(ir, tensor_name, tinfo)
    list_idx = _psum_list_index(ir, tensor_name, tinfo, ptile_dims)
    return slice_expr if list_idx is None else f"[{list_idx}]{slice_expr}"


def _sbuf_dst_index_expr(ir: KernelIR, group_idx: int, op_idx: int, tinfo: TensorInfo, ptile_dims: set[str]) -> str:
    """Destination index into a 4D (or 2D) SBUF buffer — no reshape."""
    da = ir.dim_analysis
    op_tiles = da.op_tile_sizes[op_idx]
    dim_ids = tinfo.dim_ids
    if len(dim_ids) == 2:
        d_p, d_f = dim_ids
        di_tp = da.dims[d_p].physical_tile_size
        di_tf = da.dims[d_f].physical_tile_size
        ptile_p = _slot_index(d_p, di_tp, op_tiles.get(d_p, di_tp), ptile_dims)
        ptile_f = _slot_index(d_f, di_tf, op_tiles.get(d_f, di_tf), ptile_dims)
        idx = f"[0:{di_tp}, {ptile_p}, {ptile_f}, 0:{di_tf}]"
    else:
        d_p = dim_ids[0]
        di_tp = da.dims[d_p].physical_tile_size
        ptile_p = _slot_index(d_p, di_tp, op_tiles.get(d_p, di_tp), ptile_dims)
        idx = f"[0:{di_tp}, {ptile_p}]"
    return idx


def _sbuf_tile_index_expr(
    ir: KernelIR, group_idx: int, op_idx: int, tensor_name: str, tinfo: TensorInfo, ptile_dims: set[str]
) -> str:
    """Operand access into an SBUF buffer.

    Reshape goes before the slice because the NKI simulator
    rejects ``.reshape(...)`` on a pre-sliced view. Partition-
    axis slots stay as a separate dim (flattening them into the
    free axis would mix partition rows into free positions).
    """
    da = ir.dim_analysis
    op_tiles = da.op_tile_sizes[op_idx]
    dim_ids = tinfo.dim_ids
    expr: str
    if len(dim_ids) == 1:
        d_p = dim_ids[0]
        di_tp = da.dims[d_p].physical_tile_size
        op_tp = op_tiles.get(d_p, di_tp)
        n_slots_p = num_tiles(ir, tensor_name, d_p)
        if n_slots_p == 1:
            expr = f"[0:{di_tp}, 0]"
        else:
            flat = _flat_axis_range(ir, group_idx, tensor_name, d_p, di_tp, op_tp, ptile_dims)
            expr = f".reshape(({di_tp} * {n_slots_p},))[{flat}]"
    else:
        d_p, d_f = dim_ids
        di_tp = da.dims[d_p].physical_tile_size
        di_tf = da.dims[d_f].physical_tile_size
        op_tp = op_tiles.get(d_p, di_tp)
        op_tf = op_tiles.get(d_f, di_tf)
        n_slots_p = num_tiles(ir, tensor_name, d_p)
        n_slots_f = num_tiles(ir, tensor_name, d_f)

        par_range = f"0:{di_tp}"
        if n_slots_p == 1 and n_slots_f == 1:
            expr = f"[{par_range}, 0, 0, 0:{di_tf}]"
        elif n_slots_p == 1:
            total_f = n_slots_f * di_tf
            flat = _flat_axis_range(ir, group_idx, tensor_name, d_f, di_tf, op_tf, ptile_dims)
            expr = f".reshape(({di_tp}, {total_f}))[{par_range}, {flat}]"
        elif n_slots_f == 1:
            mid = _axis_slot_index(ir, group_idx, tensor_name, d_p, op_tp // di_tp, ptile_dims)
            expr = f".reshape(({di_tp}, {n_slots_p}, {di_tf}))[{par_range}, {mid}, 0:{di_tf}]"
        else:
            total_f = n_slots_f * di_tf
            mid = _axis_slot_index(ir, group_idx, tensor_name, d_p, op_tp // di_tp, ptile_dims)
            flat = _flat_axis_range(ir, group_idx, tensor_name, d_f, di_tf, op_tf, ptile_dims)
            expr = f".reshape(({di_tp}, {n_slots_p}, {total_f}))[{par_range}, {mid}, {flat}]"
    return expr


def _flat_axis_range(
    ir: KernelIR, group_idx: int, tensor_name: str, dim_id: str, phys: int, op_size: int, ptile_dims: set[str]
) -> str:
    """Flat ``start:end`` range over a single flattened axis (``num_tiles * phys``)."""
    start = _slot_expr(ir, group_idx, tensor_name, dim_id, phys, ptile_dims)
    end = f"{start} + {op_size}" if start != "0" else str(op_size)
    return f"{start}:{end}"


def _axis_slot_index(
    ir: KernelIR, group_idx: int, tensor_name: str, dim_id: str, op_slots: int, ptile_dims: set[str]
) -> str:
    """Integer (or slice) index for a num_tiles axis kept as a separate dimension."""
    slot_start = _slot_expr(ir, group_idx, tensor_name, dim_id, 1, ptile_dims)
    if op_slots <= 1:
        expr = slot_start
    elif slot_start == "0":
        expr = f"0:{op_slots}"
    else:
        expr = f"{slot_start}:{slot_start} + {op_slots}"
    return expr


def _slot_expr(ir: KernelIR, group_idx: int, tensor_name: str, dim_id: str, unit: int, ptile_dims: set[str]) -> str:
    """Flat slot offset on ``dim_id``, combining tier iteration with ptile iteration.

    ``unit`` is the per-slot stride — ``phys_tile`` for slot-
    addressed layouts, ``1`` when the caller passes a raw slot
    count.
    """
    num_ptiles = ir.dim_analysis.dims[dim_id].num_ptiles
    tier = ir.fusion_groups[group_idx].tensor_placements.get((tensor_name, dim_id), "per_tile")
    tpb = ir.ltiles_per_block.get(dim_id, 1)

    terms: list[str] = []
    if tier == "full":
        stride = unit * tpb * num_ptiles
        terms.append(f"i_block_{dim_id} * {stride}" if stride > 1 else f"i_block_{dim_id}")
    if tier in ("per_block", "full"):
        stride = unit * num_ptiles
        terms.append(f"i_ltile_{dim_id} * {stride}" if stride > 1 else f"i_ltile_{dim_id}")
    if dim_id in ptile_dims:
        terms.append(f"i_ptile_{dim_id} * {unit}" if unit > 1 else f"i_ptile_{dim_id}")
    return " + ".join(terms) if terms else "0"


def _slot_index(dim_id: str, physical_tile: int, op_tile: int, ptile_dims: set[str]) -> str:
    """Physical-tile slice index for one dim of the op's tile — ``i_ptile_{d}``-scaled, ``0:n`` static span, or ``0``."""
    num_ptiles = op_tile // physical_tile
    if dim_id in ptile_dims:
        expr = (
            f"i_ptile_{dim_id}"
            if num_ptiles <= 1
            else f"i_ptile_{dim_id} * {num_ptiles}:(i_ptile_{dim_id} + 1) * {num_ptiles}"
        )
    elif num_ptiles > 1:
        expr = f"0:{num_ptiles}"
    else:
        expr = "0"
    return expr
