"""build_ir and render_ir: math function -> KernelIR -> NKI source."""

import inspect
from collections.abc import Callable

import numpy as np

from nkigym.codegen.ir import DimInfo, KernelIR, OpDimInfo, OpGraph, OpInfo, TensorInfo
from nkigym.codegen.parse import find_ops
from nkigym.ops.base import NKIOp, RenderContext, Tensor
from nkigym.ops.common import ind, linear_expr


def _dim_size(ctx: RenderContext, dim_id: str) -> int | None:
    """Return the size of dim_id from the first tensor that carries it."""
    return next(
        (s for t in ctx.tensors.values() if t.dim_ids for d, s in zip(t.dim_ids, t.shape, strict=True) if d == dim_id),
        None,
    )


def _unify_dim(
    op: NKIOp, ctx: RenderContext, per_op_maps: list[dict[str, str]], abstract: str, old_id: str, new_id: str
) -> None:
    """Validate sizes match, then rename old_id -> new_id everywhere."""
    old_size = _dim_size(ctx, old_id)
    new_size = _dim_size(ctx, new_id)
    if old_size is not None and new_size is not None and old_size != new_size:
        raise ValueError(
            f"Op {op.NAME!r}: axis {abstract!r} unifies"
            f" {old_id!r} (size {old_size}) with {new_id!r} (size {new_size})"
        )
    for tensor in ctx.tensors.values():
        if tensor.dim_ids and old_id in tensor.dim_ids:
            tensor.dim_ids = tuple(new_id if d == old_id else d for d in tensor.dim_ids)
    for axis_map in per_op_maps:
        for ax, concrete in axis_map.items():
            if concrete == old_id:
                axis_map[ax] = new_id


def _local_axis_map(
    op: NKIOp,
    operand_map: dict[str, str],
    ctx: RenderContext,
    dim_counter: list[int],
    per_op_maps: list[dict[str, str]],
) -> dict[str, str]:
    """Build per-op abstract->concrete axis mapping, unifying shared dims."""
    local: dict[str, str] = {}
    for slot, axes in op.OPERAND_AXES.items():
        tensor = ctx.tensors[operand_map[slot]]
        if tensor.dim_ids:
            for abstract, concrete in zip(axes, tensor.dim_ids, strict=True):
                if abstract in local and local[abstract] != concrete:
                    _unify_dim(op, ctx, per_op_maps, abstract, old_id=concrete, new_id=local[abstract])
                else:
                    local[abstract] = concrete
        else:
            for abstract in axes:
                if abstract not in local:
                    local[abstract] = f"d{dim_counter[0]}"
                    dim_counter[0] += 1
            tensor.dim_ids = tuple(local[a] for a in axes)
    return local


def _resolve_output(op: NKIOp, operand_map: dict[str, str], ctx: RenderContext) -> tuple[tuple[int, ...], str]:
    """Trace output shape and dtype from an op's operand axes."""
    axis_sizes: dict[str, int] = {}
    for slot, axes in op.OPERAND_AXES.items():
        for axis_label, size in zip(axes, ctx.tensors[operand_map[slot]].shape, strict=True):
            axis_sizes[axis_label] = size
    first_slot = next(iter(op.OPERAND_AXES))
    dtype = ctx.tensors[operand_map[first_slot]].dtype
    output_axes = next(iter(op.OUTPUT_AXES.values()))
    return tuple(axis_sizes[a] for a in output_axes), dtype


def _process_ops(
    ops: list[tuple[NKIOp, dict[str, str], str]], ctx: RenderContext
) -> tuple[list[dict[str, str]], dict[str, int], dict[str, int], dict[str, int]]:
    """Forward pass: assign dim IDs, create output tensors, unify tile sizes."""
    dim_counter = [0]
    per_op_maps: list[dict[str, str]] = []
    for op, operand_map, output_name in ops:
        local = _local_axis_map(op, operand_map, ctx, dim_counter, per_op_maps)
        per_op_maps.append(local)
        output_shape, output_dtype = _resolve_output(op, operand_map, ctx)
        output_axes = next(iter(op.OUTPUT_AXES.values()))
        dim_ids = tuple(local[a] for a in output_axes)
        ctx.tensors[output_name] = Tensor(
            name=output_name, shape=output_shape, dtype=output_dtype, location="", dim_ids=dim_ids
        )
    dim_tiles: dict[str, int] = {}
    dim_min_tiles: dict[str, int] = {}
    for (op, _, _), local in zip(ops, per_op_maps, strict=True):
        for abstract_axis, limit in op.TILE_LIMITS.items():
            dim_id = local[abstract_axis]
            dim_tiles[dim_id] = max(dim_tiles.get(dim_id, limit), limit)
            dim_min_tiles[dim_id] = min(dim_min_tiles.get(dim_id, limit), limit)
    dim_sizes: dict[str, int] = {}
    for tensor in ctx.tensors.values():
        for d, s in zip(tensor.dim_ids, tensor.shape, strict=True):
            dim_sizes[d] = s
    for dim_id in dim_tiles:
        dim_tiles[dim_id] = min(dim_tiles[dim_id], dim_sizes[dim_id])
    for dim_id in dim_min_tiles:
        dim_min_tiles[dim_id] = min(dim_min_tiles[dim_id], dim_sizes[dim_id])
    return per_op_maps, dim_tiles, dim_min_tiles, dim_sizes


def build_ir(func: Callable[..., np.ndarray], input_specs: dict[str, tuple]) -> KernelIR:
    """Construct initial KernelIR with section 5.3 defaults."""
    param_names = list(inspect.signature(func).parameters.keys())
    for name in param_names:
        if name not in input_specs:
            raise ValueError(f"Missing input_spec for parameter: {name!r}")
    ctx = RenderContext()
    for name in param_names:
        ctx.tensors[name] = Tensor(name=name, shape=input_specs[name][0], dtype=input_specs[name][1], location="hbm")
    ops, return_name = find_ops(func)
    per_op_maps, dim_tiles, dim_min_tiles, dim_sizes = _process_ops(ops, ctx)
    dims = {d: DimInfo(dim_sizes[d], dim_tiles[d], dim_min_tiles[d]) for d in dim_tiles}
    tensor_producers: dict[str, int] = {}
    tensors: dict[str, TensorInfo] = {}
    for name in param_names:
        t = ctx.tensors[name]
        tensors[name] = TensorInfo(t.dim_ids, t.shape, t.dtype, "hbm")
    for i, (op, _, output_name) in enumerate(ops):
        tensor_producers[output_name] = i
        t = ctx.tensors[output_name]
        tensors[output_name] = TensorInfo(t.dim_ids, t.shape, t.dtype, op.ISA_LOC)
    op_nodes: list[OpInfo] = []
    for i, (op, operand_map, output_name) in enumerate(ops):
        preds = [tensor_producers[n] for n in operand_map.values() if n in tensor_producers]
        per_dim: dict[str, OpDimInfo] = {}
        for abstract_axis, limit in op.TILE_LIMITS.items():
            dim_id = per_op_maps[i][abstract_axis]
            ot = min(limit, dims[dim_id].unified_tile_size)
            per_dim[dim_id] = OpDimInfo(ot, dims[dim_id].unified_tile_size // ot, ot // dims[dim_id].min_tile_size)
        op_nodes.append(
            OpInfo(
                op.NAME,
                type(op),
                dict(operand_map),
                output_name,
                dict(per_op_maps[i]),
                per_dim,
                preds,
                op.BLOCKING_AXES,
            )
        )
    op_graph = OpGraph(op_nodes, tensor_producers)
    fusion_groups = [[i] for i in range(len(ops))]
    tiles_per_block: dict[tuple[int, str], int] = {}
    for i, node in enumerate(op_nodes):
        for dim_id in node.per_dim:
            tiles_per_block[(i, dim_id)] = 1
    buffer_degrees: dict[tuple[int, str, str], int] = {}
    for gi, group in enumerate(fusion_groups):
        for oi in group:
            for d in tensors[op_nodes[oi].output].dim_ids:
                buffer_degrees[(gi, op_nodes[oi].output, d)] = 1
    loop_order: list[list[str]] = []
    for group in fusion_groups:
        all_dims: set[str] = set()
        for oi in group:
            all_dims.update(op_nodes[oi].per_dim.keys())
        loop_order.append(sorted(all_dims, key=lambda d: int(d[1:])))
    load_positions: dict[str, int] = {}
    for gi, group in enumerate(fusion_groups):
        lo = loop_order[gi]
        pos_map = {d: i for i, d in enumerate(lo)}
        for oi in group:
            for tname in op_nodes[oi].operands.values():
                if tensors[tname].isa_loc != "hbm":
                    continue
                if tname in load_positions:
                    continue
                dep_positions = [pos_map[d] for d in tensors[tname].dim_ids if d in pos_map]
                load_positions[tname] = max(dep_positions) + 1 if dep_positions else 0
    return KernelIR(
        func_name=func.__name__,
        param_names=param_names,
        return_name=return_name,
        dims=dims,
        tensors=tensors,
        op_graph=op_graph,
        input_specs=input_specs,
        fusion_groups=fusion_groups,
        tiles_per_block=tiles_per_block,
        buffer_degrees=buffer_degrees,
        loop_order=loop_order,
        load_positions=load_positions,
    )


def _block_ofs(ir: KernelIR, tname: str, nblk: dict[str, int]) -> tuple[str, str]:
    """HBM block offsets for load_tensor_block; ``"0"`` when trip count is 1."""
    d0, d1 = ir.tensors[tname].dim_ids
    po = f"i_block_{d0} * {ir.dims[d0].unified_tile_size}" if nblk[d0] > 1 else "0"
    fo = f"i_block_{d1} * {ir.dims[d1].unified_tile_size}" if nblk[d1] > 1 else "0"
    return (po, fo)


def _tile_idx(dim_id: str, tpb: int, nig: int, nblk: int) -> list[tuple[str, int, int]]:
    """Tile index terms at op-tile granularity (section 5.2 formula)."""
    return [(f"i_block_{dim_id}", tpb * nig, nblk), (f"i_tile_{dim_id}", nig, tpb), (f"i_ig_{dim_id}", 1, nig)]


def _local_staging_idx(dim_id: str, odi: OpDimInfo, tpb: int) -> str:
    """Local min-tile index within a 1-unified-tile staging buffer."""
    return linear_expr(
        [(f"i_tile_{dim_id}", odi.num_ig * odi.tiles_per_ig, tpb), (f"i_ig_{dim_id}", odi.tiles_per_ig, odi.num_ig)]
    )


def _global_min_tile_idx(dim_id: str, odi: OpDimInfo, tpb: int, nblk: int) -> str:
    """Global min-tile index into a full-range cross-group buffer."""
    return linear_expr(
        [
            (f"i_block_{dim_id}", tpb * odi.num_ig * odi.tiles_per_ig, nblk),
            (f"i_tile_{dim_id}", odi.num_ig * odi.tiles_per_ig, tpb),
            (f"i_ig_{dim_id}", odi.tiles_per_ig, odi.num_ig),
        ]
    )


def _sbuf_slice(tname: str, d0: str, d1: str, ir: KernelIR, op: OpInfo, tpb: dict, nblk: dict) -> str:
    """Build 4D SBUF slice for an operand (staging or cross-group)."""
    m0, m1 = ir.dims[d0].min_tile_size, ir.dims[d1].min_tile_size
    if ir.tensors[tname].isa_loc == "hbm":
        i0 = _local_staging_idx(d0, op.per_dim[d0], tpb[d0])
        i1 = _local_staging_idx(d1, op.per_dim[d1], tpb[d1])
    else:
        i0 = _global_min_tile_idx(d0, op.per_dim[d0], tpb[d0], nblk[d0])
        i1 = _global_min_tile_idx(d1, op.per_dim[d1], tpb[d1], nblk[d1])
    return f"sbuf_{tname}[0:{m0}, {i0}, {i1}, 0:{m1}]"


def _emit_group_buffers(
    ir: KernelIR,
    op: OpInfo,
    pos: dict[str, int],
    k_pos: int | None,
    rendered_k_depth: int | None,
    tpb: dict,
    nblk: dict,
) -> tuple[list[str], str, int, int]:
    """Emit buffer allocations and memset. Returns (lines, psum_name, op_p, op_f)."""
    lines: list[str] = []
    for tname in op.operands.values():
        ti = ir.tensors[tname]
        if ti.isa_loc != "hbm":
            continue
        d0, d1 = ti.dim_ids
        p0 = ir.dims[d0].unified_tile_size // ir.dims[d0].min_tile_size
        p1 = ir.dims[d1].unified_tile_size // ir.dims[d1].min_tile_size
        sh = (ir.dims[d0].min_tile_size, p0, p1, ir.dims[d1].min_tile_size)
        lines.append(f"sbuf_{tname} = nl.ndarray({sh}, dtype=nl.{ti.dtype}, buffer=nl.sbuf)")
    out_ti = ir.tensors[op.output]
    od0, od1 = out_ti.dim_ids
    op_p = op.per_dim[od0].op_tile_size
    op_f = op.per_dim[od1].op_tile_size
    psum_dtype = op.op_cls.PSUM_DTYPE or out_ti.dtype
    psum_name = f"psum_{op.output}"
    if op.op_cls.ISA_LOC == "psum" and op.blocking_axes:
        """PSUM accumulator — blocking op (section 5.2 item 2)."""
        pm_b = tpb[od0] * op.per_dim[od0].num_ig
        pm = nblk[od0] * pm_b if k_pos is not None and pos[od0] > k_pos else pm_b
        pn_b = tpb[od1] * op.per_dim[od1].num_ig
        pn = nblk[od1] * pn_b if k_pos is not None and pos[od1] > k_pos else pn_b
        lines.append(f"{psum_name} = nl.ndarray(({op_p}, {pm}, {pn}, {op_f}), dtype=nl.{psum_dtype}, buffer=nl.psum)")
        if rendered_k_depth is not None:
            psl = f"0:{op_p}, 0:{pm}, 0:{pn}, 0:{op_f}"
            lines.append(f"{ind(rendered_k_depth)}nisa.memset({psum_name}[{psl}], value=0.0)")
    elif op.op_cls.ISA_LOC == "psum":
        """PSUM temp — non-blocking op (section 5.2 item 2). Always paired with SBUF output."""
        psum_name = f"psum_{op.output}_tmp"
        lines.append(f"{psum_name} = nl.ndarray(({op_p}, 1, 1, {op_f}), dtype=nl.{psum_dtype}, buffer=nl.psum)")
        m0, m1 = ir.dims[od0].min_tile_size, ir.dims[od1].min_tile_size
        t0, t1 = out_ti.shape[0] // m0, out_ti.shape[1] // m1
        lines.append(
            f"sbuf_{op.output} = nl.ndarray(({m0}, {t0}, {t1}, {m1}), dtype=nl.{out_ti.dtype}, buffer=nl.sbuf)"
        )
    return lines, psum_name, op_p, op_f


def _emit_group_isa(
    ir: KernelIR, op: OpInfo, psum_name: str, op_p: int, op_f: int, tpb: dict, nblk: dict, nig: dict, depth: int
) -> list[str]:
    """Emit ISA call and unconditional tensor_copy for non-blocking PSUM ops."""
    lines: list[str] = []
    out_d0, out_d1 = ir.tensors[op.output].dim_ids
    operand_exprs: dict[str, str] = {}
    for role, tname in op.operands.items():
        td0, td1 = ir.tensors[tname].dim_ids
        operand_exprs[role] = _sbuf_slice(tname, td0, td1, ir, op, tpb, nblk)
    if op.op_cls.ISA_LOC == "psum" and op.blocking_axes:
        """Blocking PSUM: accumulator pattern — ISA call only, save handled separately."""
        idx_M = linear_expr(_tile_idx(out_d0, tpb[out_d0], nig[out_d0], nblk[out_d0]))
        idx_N = linear_expr(_tile_idx(out_d1, tpb[out_d1], nig[out_d1], nblk[out_d1]))
        psl = f"0:{op_p}, {idx_M}, {idx_N}, 0:{op_f}"
        lines.append(f"{ind(depth)}{op.op_cls.format_isa_call(psum_name + '[' + psl + ']', operand_exprs)}")
    elif op.op_cls.ISA_LOC == "psum":
        """Non-blocking PSUM: degree-1 temp + unconditional tensor_copy (section 5.2 item 4)."""
        psl = f"0:{op_p}, 0, 0, 0:{op_f}"
        lines.append(f"{ind(depth)}{op.op_cls.format_isa_call(psum_name + '[' + psl + ']', operand_exprs)}")
        g0 = _global_min_tile_idx(out_d0, op.per_dim[out_d0], tpb[out_d0], nblk[out_d0])
        g1 = _global_min_tile_idx(out_d1, op.per_dim[out_d1], tpb[out_d1], nblk[out_d1])
        m0, m1 = ir.dims[out_d0].min_tile_size, ir.dims[out_d1].min_tile_size
        lines.append(f"{ind(depth)}nisa.tensor_copy(sbuf_{op.output}[0:{m0}, {g0}, {g1}, 0:{m1}], {psum_name}[{psl}])")
    return lines


def _emit_group_save(
    ir: KernelIR,
    op: OpInfo,
    psum_name: str,
    pos: dict[str, int],
    k_pos: int | None,
    rendered_k_depth: int | None,
    tpb: dict,
    nblk: dict,
    is_final: bool,
) -> list[str]:
    """Emit save_tensor_block after blocking loop or after all loops."""
    lines: list[str] = []
    if is_final and op.op_cls.ISA_LOC == "psum" and op.blocking_axes and k_pos is not None:
        """Blocking PSUM final: save from PSUM accum (section 5.2 item 6)."""
        assert rendered_k_depth is not None
        od0, od1 = ir.tensors[op.output].dim_ids
        m_pos, n_pos = pos[od0], pos[od1]
        ps = tpb[od0] * ir.dims[od0].unified_tile_size
        fs = tpb[od1] * ir.dims[od1].unified_tile_size
        po = f"i_block_{od0} * {ps}" if m_pos < k_pos and nblk[od0] > 1 else "0"
        fo = f"i_block_{od1} * {fs}" if n_pos < k_pos and nblk[od1] > 1 else "0"
        lines.append(f"{ind(rendered_k_depth)}save_tensor_block({op.output}, {psum_name}, par_ofs={po}, free_ofs={fo})")
    elif is_final and op.op_cls.ISA_LOC == "psum":
        """Non-blocking PSUM final: data in SBUF from tensor_copy, save after all loops."""
        lines.append(f"{ind(0)}save_tensor_block({op.output}, sbuf_{op.output}, par_ofs=0, free_ofs=0)")
    return lines


def _rendered_depths(lo: list[str], nblk: dict, tpb: dict, nig: dict, simplify: bool) -> tuple[list[int], int]:
    """Rendered depth at each block-loop slot, plus final compute depth."""
    blk_rd = [0]
    d = 0
    for dim_id in lo:
        if not simplify or nblk[dim_id] > 1:
            d += 1
        blk_rd.append(d)
    for dim_id in lo:
        if not simplify or tpb[dim_id] > 1:
            d += 1
    for dim_id in lo:
        if not simplify or nig[dim_id] > 1:
            d += 1
    return blk_rd, d


def _emit_loops(
    lo: list[str],
    nblk: dict,
    tpb: dict,
    nig: dict,
    blk_rd: list[int],
    loads_at: dict[int, list[str]],
    ir: KernelIR,
    simplify: bool,
) -> list[str]:
    """Emit block/tile/ig loops, interleaving loads at their positions."""
    lines: list[str] = []
    for tname in loads_at.get(0, []):
        po, fo = _block_ofs(ir, tname, nblk)
        lines.append(f"{ind(0)}load_tensor_block(sbuf_{tname}, {tname}, par_ofs={po}, free_ofs={fo})")
    for i, dim_id in enumerate(lo):
        if not simplify or nblk[dim_id] > 1:
            lines.append(f"{ind(blk_rd[i])}for i_block_{dim_id} in range({nblk[dim_id]}):")
        for tname in loads_at.get(i + 1, []):
            po, fo = _block_ofs(ir, tname, nblk)
            lines.append(f"{ind(blk_rd[i + 1])}load_tensor_block(sbuf_{tname}, {tname}, par_ofs={po}, free_ofs={fo})")
    d = blk_rd[-1]
    for dim_id in lo:
        if not simplify or tpb[dim_id] > 1:
            lines.append(f"{ind(d)}for i_tile_{dim_id} in range({tpb[dim_id]}):")
            d += 1
    for dim_id in lo:
        if not simplify or nig[dim_id] > 1:
            lines.append(f"{ind(d)}for i_ig_{dim_id} in range({nig[dim_id]}):")
            d += 1
    return lines


def _render_group(ir: KernelIR, group_idx: int, simplify: bool) -> list[str]:
    """Render one fusion group. When *simplify*, trivial loops are omitted."""
    group = ir.fusion_groups[group_idx]
    lo = ir.loop_order[group_idx]
    pos = {d: i for i, d in enumerate(lo)}
    n = len(lo)
    if len(group) != 1:
        raise NotImplementedError("Multi-op fused groups not yet in generic renderer")
    op_idx = group[0]
    op = ir.op_graph.nodes[op_idx]
    is_final = op.output == ir.return_name
    blocking_positions = [pos[op.dim_map[a]] for a in op.blocking_axes if a in op.dim_map]
    k_pos: int | None = min(blocking_positions) if blocking_positions else None
    tpb: dict[str, int] = {}
    nblk: dict[str, int] = {}
    nig: dict[str, int] = {}
    for dim_id in lo:
        di = ir.dims[dim_id]
        t = ir.tiles_per_block.get((op_idx, dim_id), 1)
        tpb[dim_id] = t
        nblk[dim_id] = di.dim_size // (t * di.unified_tile_size)
        nig[dim_id] = op.per_dim[dim_id].num_ig
    blk_rd, compute_depth = _rendered_depths(lo, nblk, tpb, nig, simplify)
    rendered_k_depth = blk_rd[k_pos] if k_pos is not None else None
    lines, psum_name, op_p, op_f = _emit_group_buffers(ir, op, pos, k_pos, rendered_k_depth, tpb, nblk)
    loads_at: dict[int, list[str]] = {}
    for tname in op.operands.values():
        if ir.tensors[tname].isa_loc == "hbm":
            loads_at.setdefault(ir.load_positions.get(tname, n), []).append(tname)
    lines.extend(_emit_loops(lo, nblk, tpb, nig, blk_rd, loads_at, ir, simplify))
    lines.extend(_emit_group_isa(ir, op, psum_name, op_p, op_f, tpb, nblk, nig, compute_depth))
    lines.extend(_emit_group_save(ir, op, psum_name, pos, k_pos, rendered_k_depth, tpb, nblk, is_final))
    return lines


def render_ir(ir: KernelIR, simplify: bool) -> str:
    """Generic renderer: KernelIR -> NKI source (section 5.2)."""
    ret_ti = ir.tensors[ir.return_name]
    shape_str = ", ".join(str(s) for s in ret_ti.shape)
    hdr = f"def {ir.func_name}_kernel(" + ", ".join(ir.param_names) + "):"
    lines = [
        "import nki",
        "import nki.language as nl",
        "import nki.isa as nisa",
        "from nkigym.gadgets import load_tensor_block, save_tensor_block",
        "",
        "",
        "@nki.jit",
        hdr,
    ]
    for name in ir.param_names:
        ss = ", ".join(str(s) for s in ir.input_specs[name][0])
        lines.append(f"{ind(1)}assert {name}.shape == ({ss})")
    lines.append(f"{ind(1)}{ir.return_name} = nl.ndarray(({shape_str}), dtype=nl.{ret_ti.dtype}, buffer=nl.shared_hbm)")
    for group_idx in range(len(ir.fusion_groups)):
        for line in _render_group(ir, group_idx, simplify):
            lines.append(f"{ind(1)}{line}")
    lines.append(f"{ind(1)}return {ir.return_name}")
    return "\n".join(lines)
