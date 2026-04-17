"""DMA codegen: HBM↔SBUF and PSUM↔SBUF transfer rendering.

Positions are derived from KernelIR:

- HBM loads: each kernel input tensor is loaded once in the
  fusion group that contains its earliest consumer op. Depth
  within the group's nest is driven by ``tensor_placements``
  tiers per dim (per_tile / per_block / full).
- PSUM staging: for each PSUM tensor needing SBUF staging, emit
  ``stage_tensor_block`` in the producing op's group. Non-blocking
  ops stage at the innermost body; blocking ops stage after the
  outermost blocking dim's block loop closes.
- HBM store: the return tensor is stored inside the producing
  group's innermost body.
"""

from nkigym.codegen.buffers import find_psum_tensors_needing_sbuf
from nkigym.codegen.group_loops import DepthPlan
from nkigym.kernel_ir import KernelIR, get_tpb
from nkigym.kernel_ir.dim_analysis import TensorInfo, op_blocking_dims


def build_op_to_group(ir: KernelIR) -> dict[int, int]:
    """Build the op-index → fusion-group-index map."""
    result: dict[int, int] = {}
    for gi, group in enumerate(ir.fusion_groups):
        for op_idx in group:
            result[op_idx] = gi
    return result


def render_hbm_loads(ir: KernelIR, op_to_group: dict[int, int]) -> DepthPlan:
    """Plan HBM→SBUF loads for every kernel input.

    Returns a per-group injection map ``{group_idx: {depth: [lines]}}``
    where ``depth`` is an offset within the group's nest (``0`` =
    before the first block loop, ``N`` = between block/tile phases,
    ``2N`` = innermost body).
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
        depth = _load_depth(ir, tensor_name, tinfo, dim_order)
        par_ofs, free_ofs = _tensor_offsets(ir, tinfo, dim_order)
        line = f"load_tensor_block(sbuf_{tensor_name}, {tensor_name}, {par_ofs}, {free_ofs})"
        plan.setdefault(group_idx, {}).setdefault(depth, []).append(line)

    return plan


def render_psum_staging(ir: KernelIR, op_to_group: dict[int, int]) -> tuple[DepthPlan, DepthPlan]:
    """Plan PSUM→SBUF staging for each PSUM tensor needing SBUF.

    Non-blocking producers stage at depth ``2 * N`` (innermost
    body). Blocking producers stage at depth ``i_min`` — the
    smallest position of a blocking dim in the group's
    ``dim_order``. Under phase-grouped emission, depth ``i_min``
    sits just outside the outermost blocking dim's block loop,
    so every blocking dim's loops (block and tile) have iterated.
    """
    da = ir.dim_analysis
    graph = ir.op_graph
    before: DepthPlan = {}
    after: DepthPlan = {}

    for tensor_name in sorted(find_psum_tensors_needing_sbuf(ir)):
        producer = graph.producer_op(tensor_name)
        if producer is None:
            continue
        group_idx = op_to_group[producer]
        dim_order = ir.group_dim_orders[group_idx]
        n = len(dim_order)
        op_cls = graph.op_classes[producer]
        axis_map = da.per_op_axis_maps[producer]
        blocking = op_blocking_dims(op_cls, axis_map) & set(dim_order)

        line = f"stage_tensor_block(sbuf_{tensor_name}, psum_{tensor_name})"
        if blocking:
            i_min = min(dim_order.index(d) for d in blocking)
            after.setdefault(group_idx, {}).setdefault(i_min, []).append(line)
        else:
            before.setdefault(group_idx, {}).setdefault(2 * n, []).append(line)

    return before, after


def render_hbm_store(ir: KernelIR, op_to_group: dict[int, int]) -> tuple[int, int, str] | None:
    """Plan the SBUF→HBM store for the return tensor.

    The store is emitted inside the producing group's innermost
    body (depth ``2 * N``) so the group's loop variables are in
    scope for the HBM offset expressions. Pass-through kernels
    (return tensor is a kernel input) need no store.
    """
    da = ir.dim_analysis
    ret = da.return_name
    producer = ir.op_graph.producer_op(ret)
    result: tuple[int, int, str] | None = None
    if producer is not None:
        group_idx = op_to_group[producer]
        tinfo = da.tensors[ret]
        dim_order = ir.group_dim_orders[group_idx]
        par_ofs, free_ofs = _tensor_offsets(ir, tinfo, dim_order)
        line = f"store_tensor_block({ret}, sbuf_{ret}, {par_ofs}, {free_ofs})"
        result = (group_idx, 2 * len(dim_order), line)
    return result


def _load_depth(ir: KernelIR, tensor_name: str, tinfo: TensorInfo, dim_order: list[str]) -> int:
    """Compute the depth at which to emit the load for a tensor.

    For each dim ``d`` of ``t`` at position ``i`` in ``dim_order``
    (group has ``N`` dims): ``per_tile`` requires depth ≥ ``N + i + 1``,
    ``per_block`` requires depth ≥ ``i + 1``, ``full`` imposes no
    constraint. Returned depth is the max across relevant dims.
    Dims not in ``dim_order`` impose no constraint.
    """
    n = len(dim_order)
    pos = {d: i for i, d in enumerate(dim_order)}

    required = 0
    for d in tinfo.dim_ids:
        if d not in pos:
            continue
        tier = ir.tensor_placements[(tensor_name, d)]
        i = pos[d]
        if tier == "per_tile":
            required = max(required, n + i + 1)
        elif tier == "per_block":
            required = max(required, i + 1)
        elif tier == "full":
            pass
        else:
            raise ValueError(f"Unknown placement tier {tier!r} for ({tensor_name}, {d})")
    return required


def _tensor_offsets(ir: KernelIR, tinfo: TensorInfo, dim_order: list[str]) -> tuple[str, str]:
    """Build HBM offset expressions for a tensor's partition and free axes.

    Offsets are built only from loop variables the group owns
    (``dim_order``). Dims not in the group's nest contribute
    ``0`` — the tensor is invariant over them at this scope.
    """
    par_ofs = _dim_offset(tinfo.dim_ids[0], ir, dim_order)
    if len(tinfo.dim_ids) == 2:
        free_ofs = _dim_offset(tinfo.dim_ids[1], ir, dim_order)
    else:
        free_ofs = "0"
    return par_ofs, free_ofs


def _dim_offset(dim_id: str, ir: KernelIR, dim_order: list[str]) -> str:
    """Build the HBM offset expression for one dim, if the group loops over it."""
    expr = "0"
    if dim_id in dim_order:
        di = ir.dim_analysis.dims[dim_id]
        tpb = get_tpb(ir, dim_id)
        logical = di.logical_tile_size
        block_stride = tpb * logical
        expr = f"i_block_{dim_id} * {block_stride} + i_ltile_{dim_id} * {logical}"
    return expr
