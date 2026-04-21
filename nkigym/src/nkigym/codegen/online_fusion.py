"""Render path for ``NKIOnlineFusionChain`` composite ops.

The composite is one op graph node that subsumes an X + Accumulation
chain. Its render path emits the full fused loop body: running-
buffer declarations and memsets outside the accumulation dim's
loops, and the per-iteration body (snapshot, X update, σ chain,
rescale-accumulate, matmul chunk) inside. Both populate the same
``before_plan`` the standard emitter uses, so the composite
integrates with the rest of the renderer without special hooks.

Entry point: ``render_online_fusion_op(ir, op_idx, group_idx,
before_plan)``. Called from ``render_nki_ops`` when the op class is
a subclass of ``NKIOnlineFusionChain``.
"""

from nkigym.codegen.buffers import sbuf_buffer
from nkigym.codegen.group_loops import DepthPlan
from nkigym.codegen.sbuf_buffer import AxisAccess
from nkigym.kernel_ir import KernelIR
from nkigym.ops.online_fusion_chain import NKIOnlineFusionChain


def render_online_fusion_op(ir: KernelIR, op_idx: int, group_idx: int, before_plan: DepthPlan) -> None:
    """Emit the composite's declarations + per-iteration body into ``before_plan``.

    Declarations (SBUF running buffers, PSUM chunk) go at the top
    of the group (depth 0). The per-iteration body goes at the
    innermost body (depth ``2 * N``). Running buffers live in the
    emitted source only — they are not registered in
    ``DimAnalysis.tensors`` so the rest of the IR's buffer
    machinery never sees them.

    Dispatches on ``op_cls.SCALE_ROLE`` for the specific σ chain
    shape. Today supports ``rsqrt_then_mul`` only.
    """
    op_cls = ir.op_graph.op_classes[op_idx]
    assert issubclass(op_cls, NKIOnlineFusionChain), "caller must gate on NKIOnlineFusionChain"
    dim_order = ir.fusion_groups[group_idx].dim_order
    n = len(dim_order)
    ctx = _RenderContext(ir=ir, op_idx=op_idx, group_idx=group_idx, dim_order=dim_order, n=n)
    role = op_cls.SCALE_ROLE
    if role == "rsqrt_then_mul":
        _render_rsqrt_then_mul(ctx, before_plan)
    else:
        raise NotImplementedError(f"online-fusion render for scale_role={role!r} not yet implemented")


class _RenderContext:
    """Bundle of everything the per-role render helpers need.

    Avoids threading seven parameters through every helper. Populated
    once at the top of ``render_online_fusion_op``.
    """

    def __init__(self, ir: KernelIR, op_idx: int, group_idx: int, dim_order: list[str], n: int) -> None:
        """Cache references the per-role emitters read."""
        self.ir = ir
        self.op_idx = op_idx
        self.group_idx = group_idx
        self.dim_order = dim_order
        self.n = n
        raw_cls = ir.op_graph.op_classes[op_idx]
        assert issubclass(raw_cls, NKIOnlineFusionChain), "RenderContext built for non-composite op"
        self.op_cls: type[NKIOnlineFusionChain] = raw_cls
        self.inputs_map, self.outputs_list = ir.op_graph.op_tensors[op_idx]


def _render_rsqrt_then_mul(ctx: _RenderContext, before_plan: DepthPlan) -> None:
    """RMSNorm + matmul shape.

    Expected inputs (by convention in the composite's OPERAND_AXES
    keys): ``a_in`` (pre-norm data, blocked along the
    accumulation dim), ``b_in`` (matmul RHS). Expected output:
    ``out`` (running matmul accumulator).
    """
    ir = ctx.ir
    op_cls = ctx.op_cls
    acc_dim = op_cls.ACCUMULATION_DIM
    a_name = ctx.inputs_map["a_in"]
    b_name = ctx.inputs_map["b_in"]
    out_name = ctx.outputs_list[0]
    (affine_kwargs,) = op_cls.INNER_OP_KWARGS
    a_dims = ir.dim_analysis.tensors[a_name].dim_ids
    out_dims = ir.dim_analysis.tensors[out_name].dim_ids
    partition_dim = a_dims[0]
    partition_size = ir.dim_analysis.dims[partition_dim].physical_tile_size
    out_free_size = ir.dim_analysis.dims[out_dims[1]].logical_tile_size
    acc_tile = ir.dim_analysis.dims[acc_dim].logical_tile_size
    out_dtype = ir.dim_analysis.tensors[out_name].dtype
    outer_depth = _outermost_depth(ctx, acc_dim)
    inner_depth = 2 * ctx.n
    decl_lines, memset_lines = _rsqrt_outer_declarations(ctx.op_idx, partition_size, out_free_size, acc_tile, out_dtype)
    before_plan.setdefault(ctx.group_idx, {}).setdefault(0, []).extend(decl_lines)
    before_plan.setdefault(ctx.group_idx, {}).setdefault(outer_depth, []).extend(memset_lines)
    inner_lines = _rsqrt_inner_body(ctx, a_name, b_name, out_name, affine_kwargs)
    before_plan.setdefault(ctx.group_idx, {}).setdefault(inner_depth, []).extend(inner_lines)


def _outermost_depth(ctx: _RenderContext, acc_dim: str) -> int:
    """Depth at which to emit declarations + memsets — just outside the accumulation dim's loops.

    The accumulation dim's block loop opens at position
    ``dim_order.index(acc_dim) + 1``; the ltile loop opens at
    ``N + dim_order.index(acc_dim) + 1``. We emit at the latter
    minus one — i.e. inside every outer loop but strictly outside
    the accumulation dim's own ltile. That places the memset at
    the boundary between phases, fired once per outer-dim
    iteration.
    """
    acc_pos = ctx.dim_order.index(acc_dim)
    return ctx.n + acc_pos


def _rsqrt_outer_declarations(
    op_idx: int, partition_size: int, out_free_size: int, acc_tile: int, out_dtype: str
) -> tuple[list[str], list[str]]:
    """Running-buffer declarations (once at group top) + per-accumulation memsets.

    Declarations emit at depth 0 so the compiler's allocator sees
    every PSUM / SBUF buffer alongside the rest of the group's
    buffers, avoiding the "new buffer every loop iteration"
    pattern that causes PSUM OOM. Memsets emit at
    ``_outermost_depth`` so running state resets once per output-
    tile iteration.
    """
    uid = op_idx
    ptiles = acc_tile // partition_size
    declarations = [
        f"sbuf_running_s_{uid} = nl.ndarray(({partition_size}, 1), dtype=nl.float32, buffer=nl.sbuf)",
        f"sbuf_prev_s_{uid} = nl.ndarray(({partition_size}, 1), dtype=nl.float32, buffer=nl.sbuf)",
        f"sbuf_inv_rms_{uid} = nl.ndarray(({partition_size}, 1), dtype=nl.float32, buffer=nl.sbuf)",
        f"sbuf_prev_inv_{uid} = nl.ndarray(({partition_size}, 1), dtype=nl.float32, buffer=nl.sbuf)",
        f"sbuf_recip_prev_{uid} = nl.ndarray(({partition_size}, 1), dtype=nl.float32, buffer=nl.sbuf)",
        f"sbuf_sigma_{uid} = nl.ndarray(({partition_size}, 1), dtype=nl.float32, buffer=nl.sbuf)",
        f"sbuf_delta_{uid} = nl.ndarray(({partition_size}, 1), dtype=nl.float32, buffer=nl.sbuf)",
        f"sbuf_scaled_{uid} = nl.ndarray(({partition_size}, 1), dtype=nl.float32, buffer=nl.sbuf)",
        f"sbuf_prev_scaled_{uid} = nl.ndarray(({partition_size}, 1), dtype=nl.float32, buffer=nl.sbuf)",
        f"sbuf_sq_{uid} = nl.ndarray(({partition_size}, {acc_tile}), dtype=nl.float32, buffer=nl.sbuf)",
        f"sbuf_a_normed_{uid} = nl.ndarray(({partition_size}, {acc_tile}), dtype=nl.bfloat16, buffer=nl.sbuf)",
        f"sbuf_a_t_{uid} = [nl.ndarray(({partition_size}, {partition_size}),"
        f" dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range({ptiles})]",
        f"psum_a_t_{uid} = nl.ndarray(({partition_size}, {partition_size}), dtype=nl.bfloat16, buffer=nl.psum)",
        f"psum_chunk_{uid} = nl.ndarray(({partition_size}, {out_free_size}), dtype=nl.float32, buffer=nl.psum)",
        f"sbuf_running_out_{uid} = nl.ndarray(({partition_size}, {out_free_size}),"
        f" dtype=nl.{out_dtype}, buffer=nl.sbuf)",
    ]
    memsets = [
        f"nisa.memset(sbuf_running_s_{uid}[0:{partition_size}, 0:1], 0.0)",
        f"nisa.memset(sbuf_running_out_{uid}[0:{partition_size}, 0:{out_free_size}], 0.0)",
    ]
    return declarations, memsets


def _rsqrt_inner_body(
    ctx: _RenderContext, a_name: str, b_name: str, out_name: str, affine_kwargs: dict[str, str]
) -> list[str]:
    """Per-iteration body: snapshot, update, σ chain, a_normed, transpose, matmul, rescale-accumulate.

    Access expressions for external buffers (``a``, ``b``,
    ``result`` staging) go through the standard ``SbufBuffer``
    machinery so multi-buffering and tier placements stay
    consistent with the rest of the renderer.
    """
    ir = ctx.ir
    op_cls = ctx.op_cls
    acc_dim = op_cls.ACCUMULATION_DIM
    uid = ctx.op_idx
    a_dims = ir.dim_analysis.tensors[a_name].dim_ids
    out_dims = ir.dim_analysis.tensors[out_name].dim_ids
    partition_size = ir.dim_analysis.dims[a_dims[0]].physical_tile_size
    out_free_size = ir.dim_analysis.dims[out_dims[1]].logical_tile_size
    acc_tile = ir.dim_analysis.dims[acc_dim].logical_tile_size
    ptiles = acc_tile // partition_size
    matmul_ptile_var = f"i_ptile_{acc_dim}_{uid}_m"
    a_access = _sbuf_body_access(ctx, a_name)
    b_access = _sbuf_body_access(ctx, b_name, ptile_binding={acc_dim: matmul_ptile_var})
    out_access = _sbuf_body_access(ctx, out_name)
    op0_nl = _nl(affine_kwargs.get("op0", "'multiply'"))
    operand0 = affine_kwargs.get("operand0", "1.0")
    op1 = affine_kwargs.get("op1")
    operand1 = affine_kwargs.get("operand1")
    affine_tail = f", op1={_nl(op1)}, operand1={operand1}" if op1 and operand1 else ""
    return [
        f"nisa.activation_reduce(sbuf_sq_{uid}[0:{partition_size}, 0:{acc_tile}], nl.square,"
        f" {a_access}, nl.add, sbuf_delta_{uid}[0:{partition_size}, 0:1])",
        f"nisa.tensor_copy(sbuf_prev_s_{uid}[0:{partition_size}, 0:1],"
        f" sbuf_running_s_{uid}[0:{partition_size}, 0:1])",
        f"nisa.tensor_scalar(sbuf_running_s_{uid}[0:{partition_size}, 0:1],"
        f" sbuf_running_s_{uid}[0:{partition_size}, 0:1], op0=nl.add,"
        f" operand0=sbuf_delta_{uid}[0:{partition_size}, 0:1])",
        f"nisa.tensor_scalar(sbuf_scaled_{uid}[0:{partition_size}, 0:1],"
        f" sbuf_running_s_{uid}[0:{partition_size}, 0:1], op0={op0_nl},"
        f" operand0={operand0}{affine_tail})",
        f"nisa.activation(sbuf_inv_rms_{uid}[0:{partition_size}, 0:1], nl.rsqrt,"
        f" sbuf_scaled_{uid}[0:{partition_size}, 0:1])",
        f"nisa.tensor_scalar(sbuf_prev_scaled_{uid}[0:{partition_size}, 0:1],"
        f" sbuf_prev_s_{uid}[0:{partition_size}, 0:1], op0={op0_nl},"
        f" operand0={operand0}{affine_tail})",
        f"nisa.activation(sbuf_prev_inv_{uid}[0:{partition_size}, 0:1], nl.rsqrt,"
        f" sbuf_prev_scaled_{uid}[0:{partition_size}, 0:1])",
        f"nisa.reciprocal(sbuf_recip_prev_{uid}[0:{partition_size}, 0:1],"
        f" sbuf_prev_inv_{uid}[0:{partition_size}, 0:1])",
        f"nisa.tensor_scalar(sbuf_sigma_{uid}[0:{partition_size}, 0:1],"
        f" sbuf_inv_rms_{uid}[0:{partition_size}, 0:1], op0=nl.multiply,"
        f" operand0=sbuf_recip_prev_{uid}[0:{partition_size}, 0:1])",
        f"nisa.tensor_scalar(sbuf_a_normed_{uid}[0:{partition_size}, 0:{acc_tile}],"
        f" {a_access}, op0=nl.multiply,"
        f" operand0=sbuf_inv_rms_{uid}[0:{partition_size}, 0:1])",
        f"for i_ptile_{acc_dim}_{uid} in range({ptiles}):",
        f"    nisa.nc_transpose(psum_a_t_{uid}[0:{partition_size}, 0:{partition_size}],"
        f" sbuf_a_normed_{uid}[0:{partition_size},"
        f" i_ptile_{acc_dim}_{uid} * {partition_size}:i_ptile_{acc_dim}_{uid} * {partition_size} + {partition_size}])",
        f"    nisa.tensor_copy(sbuf_a_t_{uid}[i_ptile_{acc_dim}_{uid}][0:{partition_size}, 0:{partition_size}],"
        f" psum_a_t_{uid}[0:{partition_size}, 0:{partition_size}])",
        f"nisa.memset(psum_chunk_{uid}[0:{partition_size}, 0:{out_free_size}], 0.0)",
        f"for i_ptile_{acc_dim}_{uid}_m in range({ptiles}):",
        f"    nisa.nc_matmul(dst=psum_chunk_{uid}[0:{partition_size}, 0:{out_free_size}],"
        f" stationary=sbuf_a_t_{uid}[i_ptile_{acc_dim}_{uid}_m][0:{partition_size}, 0:{partition_size}],"
        f" moving={b_access})",
        f"nisa.scalar_tensor_tensor(sbuf_running_out_{uid}[0:{partition_size}, 0:{out_free_size}],"
        f" sbuf_running_out_{uid}[0:{partition_size}, 0:{out_free_size}],"
        f" nl.multiply, sbuf_sigma_{uid}[0:{partition_size}, 0:1],"
        f" nl.add, psum_chunk_{uid}[0:{partition_size}, 0:{out_free_size}])",
        f"nisa.tensor_copy({out_access}," f" sbuf_running_out_{uid}[0:{partition_size}, 0:{out_free_size}])",
    ]


def _sbuf_body_access(ctx: _RenderContext, tensor_name: str, ptile_binding: dict[str, str] | None = None) -> str:
    """SBUF access string for an external tensor at the composite's innermost slot.

    Uses ``SbufBuffer.get_tile`` with every loop-var bound. The
    per-dim tier (``per_tile`` / ``per_block`` / ``full``) controls
    which loop vars participate in the list-index; the standard
    buffer model handles the rest. ``ptile_binding`` maps a
    concrete dim id to a loop-var expression that should be bound
    as that dim's ptile slot — needed when the access sits inside
    a ptile loop (e.g. the matmul's K-ptile loop).
    """
    buf = sbuf_buffer(ctx.ir, tensor_name)
    placements = ctx.ir.fusion_groups[ctx.group_idx].tensor_placements
    tinfo = ctx.ir.dim_analysis.tensors[tensor_name]
    dim_ids = tinfo.dim_ids
    bindings = ptile_binding or {}
    p_access = _axis_access(ctx, tensor_name, dim_ids[0], placements, bindings.get(dim_ids[0]))
    if len(dim_ids) == 2:
        f_access = _axis_access(ctx, tensor_name, dim_ids[1], placements, bindings.get(dim_ids[1]))
    else:
        f_access = AxisAccess(block="0", ltile="0")
    return buf.get_tile(p_access, f_access)


def _axis_access(
    ctx: _RenderContext,
    tensor_name: str,
    dim_id: str,
    placements: dict[tuple[str, str, str], str],
    ptile_var: str | None = None,
) -> AxisAccess:
    """Axis-access for one dim of an external tensor, respecting tier placement.

    ``ptile_var`` binds the ptile slot for this dim when the access
    sits inside a ptile loop — partition-axis buffers with
    ``ptiles_per_ltile > 1`` store those slots at the Python-list
    level and require the binding to be non-None.
    """
    tier = placements.get(("sbuf", tensor_name, dim_id), "per_tile")
    block = "0"
    ltile = "0"
    if dim_id in ctx.dim_order:
        if tier == "full":
            block = f"i_block_{dim_id}"
        if tier in ("per_block", "full"):
            ltile = f"i_ltile_{dim_id}"
    return AxisAccess(block=block, ltile=ltile, ptile=ptile_var)


def _nl(value: str) -> str:
    """Convert a quoted string op name to an ``nl.*`` reference."""
    return f"nl.{value[1:-1]}" if value.startswith("'") and value.endswith("'") else value
