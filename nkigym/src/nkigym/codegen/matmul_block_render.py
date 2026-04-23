"""Two-level matmul rendering via the ``matmul_block`` gadget.

Detection + emission of the PSUM-inner / SBUF-outer accumulation
pattern that matches ``nki_matmul_fully_optimized_``'s structure.
Candidate ``NKIMatmul`` ops are those whose reduction dim ``K`` is
blocked (``num_blocks[K] > 1``) AND whose ``K``-carrying SBUF
inputs (``stationary``, ``moving``) have ``per_block`` (or
``per_tile``) placement on ``K`` — i.e. the K inputs are reloaded
across outer-K iterations, so a running SBUF accumulator must
persist across them.

When the pattern matches, render emits one ``matmul_block(...)``
call at the innermost M/N scope *outside* the inner-K ltile loop;
the gadget does the inner-K ``nc_matmul`` accumulation in PSUM
then tensor-tensor-adds into the caller's SBUF output tile. A
pre-memset of the SBUF output runs once at group entry.

When the pattern does NOT match, ``_render_one_op`` falls back to
the classic PSUM-only path (unchanged).
"""

from nkigym.codegen.buffers import sbuf_buffer
from nkigym.codegen.group_loops import DepthPlan
from nkigym.codegen.matmul_block_detect import is_matmul_block_candidate
from nkigym.codegen.sbuf_buffer import AxisAccess, buffer_ident
from nkigym.kernel_ir import KernelIR
from nkigym.kernel_ir.validate.emission import block_depth, body_depth, ltile_depth
from nkigym.ops.base import NKIOp

__all__ = ["is_matmul_block_candidate", "render_matmul_block_op"]

_TIER_RANK = {"per_tile": 0, "per_block": 1, "full": 2}


def render_matmul_block_op(ir: KernelIR, op: NKIOp, group_idx: int, before_plan: DepthPlan) -> None:
    """Emit ``matmul_block`` + pre-memset into ``before_plan``.

    Emits at ``body_depth(n)`` — after every dim's block loop
    opens but without any ltile loops (those are absorbed by the
    gadget via ``_gadget_absorbed_dims`` in ``group_loops.py``).

    Pre-memset of ``sbuf_<output>`` goes at depth 0 (group top).
    """
    context = ir.context
    group = ir.graph.groups[group_idx]
    dim_order = group.dim_order
    n = len(dim_order)
    axis_map = context.op_axis_map.get(op, {})
    k_dim = axis_map["K"]
    m_dim = axis_map["M"]
    n_dim = axis_map["N"]

    out_name = context.op_outputs.get(op, [])[0]
    out_tinfo = context.logical_tensors[out_name]
    inputs = context.op_inputs.get(op, {})
    stat_name = inputs["stationary"]
    mov_name = inputs["moving"]

    emit_depth = body_depth(n)

    p_start, p_count = _block_slab_range(ir, group_idx, out_name, m_dim)
    f_start, f_count = _block_slab_range(ir, group_idx, out_name, n_dim)
    k_start, k_count = _k_slab_range(ir, group_idx, stat_name, k_dim)
    tile_m = context.op_tile_sizes.get(op, {}).get(m_dim, context.dimensions[m_dim].physical_tile_size)
    tile_n = context.op_tile_sizes.get(op, {}).get(n_dim, context.dimensions[n_dim].physical_tile_size)

    sbuf_out_arg = f"sbuf_{buffer_ident(out_name)}"
    stat_arg = f"sbuf_{buffer_ident(stat_name)}"
    mov_arg = f"sbuf_{buffer_ident(mov_name)}"

    call = (
        f"matmul_block({sbuf_out_arg}, {p_start}, {p_count}, {f_start}, {f_count}, "
        f"{stat_arg}, {mov_arg}, {k_start}, {k_count}, {tile_m}, {tile_n})"
    )
    before_plan.setdefault(group_idx, {}).setdefault(emit_depth, []).append(call)

    memset_lines = _render_output_memsets(ir, out_name, out_tinfo)
    memset_depth = _output_memset_depth(ir, group_idx, out_name, dim_order)
    before_plan.setdefault(group_idx, {}).setdefault(memset_depth, []).extend(memset_lines)


def _output_memset_depth(ir: KernelIR, group_idx: int, tensor_name: str, dim_order: list[str]) -> int:
    """Depth at which the SBUF output's memset should run.

    Matches the alloc depth of ``tensor_name``: the memset must
    fire inside the same block-loop scope where the buffer is
    declared fresh, never earlier (buffer undefined) or later
    (memset overrun by accumulation writes).
    """
    placements = ir.graph.groups[group_idx].tensor_placements
    tinfo = ir.context.logical_tensors[tensor_name]
    depth = 0
    for dim_id in tinfo.dim_ids:
        if dim_id not in dim_order:
            continue
        tier = placements.get(("sbuf", tensor_name, dim_id), "per_tile")
        if tier == "full":
            continue
        pos = dim_order.index(dim_id)
        depth = max(depth, 2 * pos + 1)
    return depth


def _tensor_dim_ids(tinfo: object) -> tuple[str, ...]:
    """Extract dim_ids from a TensorInfo."""
    return getattr(tinfo, "dim_ids")


def _block_slab_range(ir: KernelIR, group_idx: int, tensor_name: str, dim_id: str) -> tuple[str, int]:
    """Return ``(start_slot_expr, count)`` for one block's slab of ``tensor_name`` along ``dim_id``.

    A "block slab" is the portion a single ``i_block_{dim}``
    iteration brings into SBUF — ``ltiles_per_block`` leaves
    at the list level. ``start`` names the first slot; ``count``
    is the number of contiguous slots.
    """
    placements = ir.graph.groups[group_idx].tensor_placements
    tier = placements.get(("sbuf", tensor_name, dim_id), "per_tile")
    tpb = ir.context.ltiles_per_block.get(dim_id, 1)
    if tier == "full":
        start = f"i_block_{dim_id} * {tpb}" if tpb > 1 else f"i_block_{dim_id}"
    elif tier == "per_block":
        start = "0"
    else:
        raise ValueError(f"matmul_block requires block-slab placement on {tensor_name!r}[{dim_id}]; got {tier!r}")
    return start, tpb


def _k_slab_range(ir: KernelIR, group_idx: int, tensor_name: str, k_dim: str) -> tuple[str, int]:
    """Return ``(k_start_slot, k_count)`` along the K axis for one inner-K slab."""
    placements = ir.graph.groups[group_idx].tensor_placements
    tier = placements.get(("sbuf", tensor_name, k_dim), "per_tile")
    if tier != "per_block":
        raise ValueError(f"matmul_block requires K-input tier=per_block on {tensor_name!r}; got {tier!r}")
    tpb = ir.context.ltiles_per_block.get(k_dim, 1)
    return "0", tpb


def _render_output_memsets(ir: KernelIR, tensor_name: str, tinfo: object) -> list[str]:
    """Emit memset loops zeroing every slot of the SBUF output buffer at kernel start."""
    _ = tinfo
    buf = sbuf_buffer(ir, tensor_name)
    leaf_expr = f"sbuf_{buffer_ident(tensor_name)}[i_pmr][i_fmr]"
    return [
        f"for i_pmr in range({buf.p.list_slots}):",
        f"    for i_fmr in range({buf.f.list_slots}):",
        f"        nisa.memset({leaf_expr}[0:{buf.p.logical}, 0:{buf.f.logical}], 0.0)",
    ]


_ = (_tensor_dim_ids, AxisAccess, block_depth, ltile_depth)
