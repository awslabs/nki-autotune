"""Render path for ``NKIOnlineFusionChain`` composite ops."""

from dataclasses import dataclass
from typing import cast

from nkigym.codegen.buffers import sbuf_buffer
from nkigym.codegen.group_loops import DepthPlan
from nkigym.codegen.sbuf_buffer import AxisAccess
from nkigym.kernel_ir import KernelIR
from nkigym.kernel_ir.rewrites.online_fusion_spec import AccumulatorSpec, ScaleSpec
from nkigym.kernel_ir.validate.emission import block_depth
from nkigym.ops.base import NKIOp
from nkigym.ops.online_fusion_chain import NKIOnlineFusionChain


def render_online_fusion_op(ir: KernelIR, op: NKIOp, group_idx: int, before_plan: DepthPlan) -> None:
    """Emit the composite's full fused loop body into ``before_plan``."""
    ctx = _RenderContext.build(ir, op, group_idx)
    ctx.assert_supported()
    declarations: list[str] = []
    memsets: list[str] = []
    _emit_scale_declarations(ctx, declarations, memsets)
    _emit_accumulator_declarations(ctx, declarations, memsets)
    before_plan.setdefault(group_idx, {}).setdefault(0, []).extend(declarations)
    outer_depth = _outermost_depth(ctx, ctx.acc_dim)
    before_plan.setdefault(group_idx, {}).setdefault(outer_depth, []).extend(memsets)
    inner_depth = 2 * ctx.n
    inner_lines: list[str] = []
    _emit_scale_body(ctx, inner_lines)
    _emit_accumulator_bodies(ctx, inner_lines)
    before_plan.setdefault(group_idx, {}).setdefault(inner_depth, []).extend(inner_lines)


@dataclass
class _RenderContext:
    """Bundle of everything the parametric emitters need for one composite.

    Scratch buffers are named ``{sbuf/psum}_<role>_<tensor_name>``
    where ``tensor_name`` is the logical tensor the scratch is
    associated with: scale-side scratches use the X input's name
    (``x_name``); per-accumulator scratches use the accumulator's
    external output tensor name. This keeps generated code
    readable and free of synthetic ``id(op)`` tags.
    """

    ir: KernelIR
    op: NKIOp
    group_idx: int
    dim_order: list[str]
    n: int
    op_cls: type[NKIOnlineFusionChain]
    inputs_map: dict[str, str]
    outputs_list: list[str]
    scale: ScaleSpec
    accumulators: tuple[AccumulatorSpec, ...]
    acc_dim: str
    partition_size: int
    x_name: str

    @classmethod
    def build(cls, ir: KernelIR, op: NKIOp, group_idx: int) -> "_RenderContext":
        """Populate a ``_RenderContext`` from a live IR + composite op instance."""
        op_cls = type(op)
        assert issubclass(op_cls, NKIOnlineFusionChain), "_RenderContext built for non-composite op"
        dim_order = ir.graph.groups[group_idx].dim_order
        inputs_map = dict(ir.context.op_inputs.get(op, {}))
        outputs_list = list(ir.context.op_outputs.get(op, []))
        scale = op_cls.SCALE_SPEC
        assert scale is not None, "composite missing SCALE_SPEC"
        x_name = next(iter(inputs_map.values()))
        partition_dim = ir.context.logical_tensors[x_name].dim_ids[0]
        return cls(
            ir=ir,
            op=op,
            group_idx=group_idx,
            dim_order=dim_order,
            n=len(dim_order),
            op_cls=cast(type[NKIOnlineFusionChain], op_cls),
            inputs_map=inputs_map,
            outputs_list=outputs_list,
            scale=cast(ScaleSpec, scale),
            accumulators=cast(tuple[AccumulatorSpec, ...], op_cls.ACCUMULATOR_SPECS),
            acc_dim=op_cls.ACCUMULATION_DIM,
            partition_size=ir.context.dimensions[partition_dim].physical_tile_size,
            x_name=x_name,
        )

    def assert_supported(self) -> None:
        """Fail loudly on combinations the render path doesn't implement yet."""
        if self.scale.sigma_kind not in {"ratio_via_reciprocal", "exp_diff_via_activation"}:
            raise NotImplementedError(f"sigma_kind={self.scale.sigma_kind!r} not yet supported")
        for spec in self.accumulators:
            if spec.kind not in {"matmul", "activation_reduce"}:
                raise NotImplementedError(f"accumulator kind={spec.kind!r} not yet supported")


def _emit_scale_declarations(ctx: _RenderContext, declarations: list[str], memsets: list[str]) -> None:
    """Running-X / prev-X / σ + scratch-buffer decls + init memset."""
    x = ctx.x_name
    p = ctx.partition_size
    running_init = ctx.scale.init_value
    declarations.append(f"sbuf_running_{x} = nl.ndarray(({p}, 1), dtype=nl.float32, buffer=nl.sbuf)")
    declarations.append(f"sbuf_prev_{x} = nl.ndarray(({p}, 1), dtype=nl.float32, buffer=nl.sbuf)")
    declarations.append(f"sbuf_sigma_{x} = nl.ndarray(({p}, 1), dtype=nl.float32, buffer=nl.sbuf)")
    declarations.append(f"sbuf_delta_{x} = nl.ndarray(({p}, 1), dtype=nl.float32, buffer=nl.sbuf)")
    if ctx.scale.delta_op == "activation_reduce":
        acc_tile = ctx.ir.context.dimensions[ctx.acc_dim].logical_tile_size
        declarations.append(f"sbuf_sq_{x} = nl.ndarray(({p}, {acc_tile}), dtype=nl.float32, buffer=nl.sbuf)")
    if ctx.scale.sigma_kind == "ratio_via_reciprocal":
        declarations.append(f"sbuf_inv_running_{x} = nl.ndarray(({p}, 1), dtype=nl.float32, buffer=nl.sbuf)")
        declarations.append(f"sbuf_inv_prev_{x} = nl.ndarray(({p}, 1), dtype=nl.float32, buffer=nl.sbuf)")
        declarations.append(f"sbuf_recip_prev_{x} = nl.ndarray(({p}, 1), dtype=nl.float32, buffer=nl.sbuf)")
        for idx in range(len(ctx.scale.inverse_chain) - 1):
            declarations.append(
                f"sbuf_invtmp_running_{x}_{idx} = nl.ndarray(({p}, 1), dtype=nl.float32, buffer=nl.sbuf)"
            )
            declarations.append(f"sbuf_invtmp_prev_{x}_{idx} = nl.ndarray(({p}, 1), dtype=nl.float32, buffer=nl.sbuf)")
    memsets.append(f"nisa.memset(sbuf_running_{x}[0:{p}, 0:1], {running_init})")


def _emit_accumulator_declarations(ctx: _RenderContext, declarations: list[str], memsets: list[str]) -> None:
    """Emit per-accumulator running-out + scratch buffers."""
    p = ctx.partition_size
    for acc_idx, spec in enumerate(ctx.accumulators):
        out_name = _accumulator_output_tensor(ctx, spec)
        out_dims = ctx.ir.context.logical_tensors[out_name].dim_ids
        out_dtype = ctx.ir.context.logical_tensors[out_name].dtype
        free_size = ctx.ir.context.dimensions[out_dims[1]].logical_tile_size if len(out_dims) == 2 else 1
        free_slice = f"0:{free_size}" if free_size > 1 else "0:1"
        declarations.append(
            f"sbuf_running_out_{out_name} = nl.ndarray(({p}, {free_size})," f" dtype=nl.{out_dtype}, buffer=nl.sbuf)"
        )
        memsets.append(f"nisa.memset(sbuf_running_out_{out_name}[0:{p}, {free_slice}], 0.0)")
        if spec.kind == "matmul":
            _emit_matmul_accumulator_decls(ctx, spec, acc_idx, out_name, free_size, declarations)
        elif spec.kind == "activation_reduce":
            _emit_activation_reduce_accumulator_decls(ctx, spec, acc_idx, out_name, declarations)


def _emit_matmul_accumulator_decls(
    ctx: _RenderContext, spec: AccumulatorSpec, acc_idx: int, out_name: str, free_size: int, declarations: list[str]
) -> None:
    """Matmul-specific scratch buffers.

    ``sbuf_a_normed_{out_name}`` is only declared when this
    accumulator's stationary source is the a_normed path (no
    prior activation_reduce supplies it as ``sbuf_exp_*``). Keeps
    declarations in sync with ``_matmul_stationary_source``.
    """
    p = ctx.partition_size
    acc_tile = ctx.ir.context.dimensions[spec.ptile_free_dim].logical_tile_size
    ptiles = acc_tile // p
    declarations.append(f"psum_chunk_{out_name} = nl.ndarray(({p}, {free_size}), dtype=nl.float32, buffer=nl.psum)")
    if not _has_prior_activation_reduce(ctx, acc_idx):
        declarations.append(
            f"sbuf_a_normed_{out_name} = nl.ndarray(({p}, {acc_tile}), dtype=nl.bfloat16, buffer=nl.sbuf)"
        )
    declarations.append(
        f"sbuf_a_t_{out_name} = [nl.ndarray(({p}, {p}), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range({ptiles})]"
    )
    declarations.append(f"psum_a_t_{out_name} = nl.ndarray(({p}, {p}), dtype=nl.bfloat16, buffer=nl.psum)")


def _has_prior_activation_reduce(ctx: _RenderContext, acc_idx: int) -> bool:
    """True iff any earlier accumulator in the chain is an ``activation_reduce``."""
    return any(pr.kind == "activation_reduce" for pr in ctx.accumulators[:acc_idx])


def _emit_activation_reduce_accumulator_decls(
    ctx: _RenderContext, spec: AccumulatorSpec, acc_idx: int, out_name: str, declarations: list[str]
) -> None:
    """Activation_reduce-specific scratch buffers."""
    p = ctx.partition_size
    _ = spec
    _ = acc_idx
    acc_tile = ctx.ir.context.dimensions[ctx.acc_dim].logical_tile_size
    declarations.append(f"sbuf_chunk_sum_{out_name} = nl.ndarray(({p}, 1), dtype=nl.float32, buffer=nl.sbuf)")
    declarations.append(f"sbuf_exp_{out_name} = nl.ndarray(({p}, {acc_tile}), dtype=nl.bfloat16, buffer=nl.sbuf)")


def _emit_scale_body(ctx: _RenderContext, lines: list[str]) -> None:
    """Snapshot prev, compute delta, update running, compute σ."""
    x = ctx.x_name
    p = ctx.partition_size
    data_access = _scale_data_access(ctx)
    delta_expr = f"sbuf_delta_{x}[0:{p}, 0:1]"
    _emit_delta_compute(ctx, data_access, delta_expr, lines)
    lines.append(f"nisa.tensor_copy(sbuf_prev_{x}[0:{p}, 0:1], sbuf_running_{x}[0:{p}, 0:1])")
    _emit_running_update(ctx, delta_expr, lines)
    _emit_sigma_compute(ctx, lines)


def _scale_data_access(ctx: _RenderContext) -> str:
    """SBUF access string for the input tensor X's delta-op reads from."""
    return _sbuf_body_access(ctx, ctx.x_name)


def _emit_delta_compute(ctx: _RenderContext, data_access: str, delta_expr: str, lines: list[str]) -> None:
    """Emit the ISA call producing this iteration's delta from the input chunk."""
    x = ctx.x_name
    p = ctx.partition_size
    op = ctx.scale.delta_op
    kwargs = ctx.scale.delta_kwargs
    if op == "activation_reduce":
        acc_tile = ctx.ir.context.dimensions[ctx.acc_dim].logical_tile_size
        act = _nl(kwargs.get("op", "'square'"))
        red = _nl(kwargs.get("reduce_op", "'add'"))
        lines.append(
            f"nisa.activation_reduce(sbuf_sq_{x}[0:{p}, 0:{acc_tile}], {act}, {data_access}, {red}, {delta_expr})"
        )
    elif op == "tensor_reduce":
        red = _nl(kwargs.get("op", "'maximum'"))
        negate_tail = ", negate=True" if kwargs.get("negate") == "True" else ""
        lines.append(f"nisa.tensor_reduce({delta_expr}, {red}, {data_access}, 1{negate_tail})")
    else:
        raise NotImplementedError(f"delta_op={op!r} not yet supported")


def _emit_running_update(ctx: _RenderContext, delta_expr: str, lines: list[str]) -> None:
    """Combine delta into running via the scale's combinator."""
    x = ctx.x_name
    p = ctx.partition_size
    running = f"sbuf_running_{x}[0:{p}, 0:1]"
    combinator = ctx.scale.combinator
    tensor_tensor_ops = {"maximum": "nl.maximum", "minimum": "nl.minimum"}
    if combinator == "add":
        lines.append(f"nisa.tensor_scalar({running}, {running}, op0=nl.add, operand0={delta_expr})")
    elif combinator in tensor_tensor_ops:
        lines.append(f"nisa.tensor_tensor({running}, {running}, {delta_expr}, {tensor_tensor_ops[combinator]})")
    else:
        raise NotImplementedError(f"combinator={combinator!r} not yet supported")


def _emit_sigma_compute(ctx: _RenderContext, lines: list[str]) -> None:
    """Compute σ from (running, prev)."""
    x = ctx.x_name
    p = ctx.partition_size
    running = f"sbuf_running_{x}[0:{p}, 0:1]"
    prev = f"sbuf_prev_{x}[0:{p}, 0:1]"
    sigma = f"sbuf_sigma_{x}[0:{p}, 0:1]"
    if ctx.scale.sigma_kind == "ratio_via_reciprocal":
        inv_running = f"sbuf_inv_running_{x}[0:{p}, 0:1]"
        inv_prev = f"sbuf_inv_prev_{x}[0:{p}, 0:1]"
        recip_prev = f"sbuf_recip_prev_{x}[0:{p}, 0:1]"
        _emit_inverse_chain(ctx, source=running, dest=inv_running, scratch_label="running", lines=lines)
        _emit_inverse_chain(ctx, source=prev, dest=inv_prev, scratch_label="prev", lines=lines)
        lines.append(f"nisa.reciprocal({recip_prev}, {inv_prev})")
        lines.append(f"nisa.tensor_scalar({sigma}, {inv_running}, op0=nl.multiply, operand0={recip_prev})")
    elif ctx.scale.sigma_kind == "exp_diff_via_activation":
        lines.append(f"nisa.activation({sigma}, nl.exp, {prev}, bias={running}, scale=-1.0)")


def _emit_inverse_chain(ctx: _RenderContext, source: str, dest: str, scratch_label: str, lines: list[str]) -> None:
    """Chain the ScaleSpec's inverse steps from ``source`` to ``dest``."""
    x = ctx.x_name
    p = ctx.partition_size
    current = source
    steps = ctx.scale.inverse_chain
    for idx, step in enumerate(steps):
        is_last = idx == len(steps) - 1
        target = dest if is_last else f"sbuf_invtmp_{scratch_label}_{x}_{idx}[0:{p}, 0:1]"
        if step.op_name == "tensor_scalar":
            lines.append(_tensor_scalar_call(target, current, step.kwargs))
        elif step.op_name == "activation":
            act = _nl(step.kwargs.get("op", "'copy'"))
            lines.append(f"nisa.activation({target}, {act}, {current})")
        else:
            raise NotImplementedError(f"inverse-chain op_name={step.op_name!r} not yet supported")
        current = target


def _tensor_scalar_call(dst: str, data: str, kwargs: dict[str, str]) -> str:
    """Format a ``nisa.tensor_scalar`` call from a kwargs dict."""
    op0 = _nl(kwargs.get("op0", "'copy'"))
    operand0 = kwargs.get("operand0", "0.0")
    op1 = kwargs.get("op1")
    operand1 = kwargs.get("operand1")
    tail = f", op1={_nl(op1)}, operand1={operand1}" if op1 and operand1 else ""
    return f"nisa.tensor_scalar({dst}, {data}, op0={op0}, operand0={operand0}{tail})"


def _emit_accumulator_bodies(ctx: _RenderContext, lines: list[str]) -> None:
    """Per-accumulator per-chunk compute + σ-correct-and-accumulate."""
    for acc_idx, spec in enumerate(ctx.accumulators):
        if spec.kind == "matmul":
            _emit_matmul_accumulator_body(ctx, spec, acc_idx, lines)
        elif spec.kind == "activation_reduce":
            _emit_activation_reduce_accumulator_body(ctx, spec, lines)


def _emit_matmul_accumulator_body(ctx: _RenderContext, spec: AccumulatorSpec, acc_idx: int, lines: list[str]) -> None:
    """Normalize a_chunk → a_t → psum_chunk → running_out rescale-accumulate."""
    x = ctx.x_name
    p = ctx.partition_size
    acc_tile = ctx.ir.context.dimensions[spec.ptile_free_dim].logical_tile_size
    ptiles = acc_tile // p
    a_name = ctx.x_name
    b_name = _matmul_moving_input(ctx, spec)
    out_name = _accumulator_output_tensor(ctx, spec)
    out_dims = ctx.ir.context.logical_tensors[out_name].dim_ids
    free_size = ctx.ir.context.dimensions[out_dims[1]].logical_tile_size if len(out_dims) == 2 else 1
    matmul_ptile_var = f"i_ptile_{ctx.acc_dim}_{out_name}"
    b_access = _sbuf_body_access(ctx, b_name, ptile_binding={ctx.acc_dim: matmul_ptile_var})
    out_access = _sbuf_body_access(ctx, out_name)
    sigma = f"sbuf_sigma_{x}[0:{p}, 0:1]"
    stationary_src = _matmul_stationary_source(ctx, acc_idx, a_name, out_name, acc_tile)
    lines.extend(stationary_src.prelude_lines)
    tpose_var = f"i_ptile_{ctx.acc_dim}_{out_name}_t"
    lines.append(f"for {tpose_var} in range({ptiles}):")
    lines.append(
        f"    nisa.nc_transpose(psum_a_t_{out_name}[0:{p}, 0:{p}],"
        f" {stationary_src.buffer_name}[0:{p}, {tpose_var} * {p}:{tpose_var} * {p} + {p}])"
    )
    lines.append(
        f"    nisa.tensor_copy(sbuf_a_t_{out_name}[{tpose_var}][0:{p}, 0:{p}]," f" psum_a_t_{out_name}[0:{p}, 0:{p}])"
    )
    lines.append(f"nisa.memset(psum_chunk_{out_name}[0:{p}, 0:{free_size}], 0.0)")
    lines.append(f"for {matmul_ptile_var} in range({ptiles}):")
    lines.append(
        f"    nisa.nc_matmul(dst=psum_chunk_{out_name}[0:{p}, 0:{free_size}],"
        f" stationary=sbuf_a_t_{out_name}[{matmul_ptile_var}][0:{p}, 0:{p}],"
        f" moving={b_access})"
    )
    running_out = f"sbuf_running_out_{out_name}[0:{p}, 0:{free_size}]"
    lines.append(
        f"nisa.scalar_tensor_tensor({running_out}, {running_out}, nl.multiply, {sigma},"
        f" nl.add, psum_chunk_{out_name}[0:{p}, 0:{free_size}])"
    )
    lines.append(f"nisa.tensor_copy({out_access}, {running_out})")


def _emit_activation_reduce_accumulator_body(ctx: _RenderContext, spec: AccumulatorSpec, lines: list[str]) -> None:
    """Per-chunk activation_reduce producing chunk_sum, then σ-correct-and-accumulate."""
    x = ctx.x_name
    p = ctx.partition_size
    data_access = _sbuf_body_access(ctx, ctx.x_name)
    out_name = _accumulator_output_tensor(ctx, spec)
    out_access = _sbuf_body_access(ctx, out_name)
    source_kwargs = dict(spec.source_kwargs)
    act = _nl(source_kwargs.get("op", "'exp'"))
    red = _nl(source_kwargs.get("reduce_op", "'add'"))
    running = f"sbuf_running_{x}[0:{p}, 0:1]"
    chunk_sum = f"sbuf_chunk_sum_{out_name}[0:{p}, 0:1]"
    sigma = f"sbuf_sigma_{x}[0:{p}, 0:1]"
    running_out = f"sbuf_running_out_{out_name}[0:{p}, 0:1]"
    acc_tile = ctx.ir.context.dimensions[ctx.acc_dim].logical_tile_size
    lines.append(
        f"nisa.activation_reduce(sbuf_exp_{out_name}[0:{p}, 0:{acc_tile}], {act},"
        f" {data_access}, {red}, {chunk_sum}, bias={running})"
    )
    lines.append(f"nisa.scalar_tensor_tensor({running_out}, {running_out}, nl.multiply, {sigma}, nl.add, {chunk_sum})")
    lines.append(f"nisa.tensor_copy({out_access}, {running_out})")


def _matmul_moving_input(ctx: _RenderContext, spec: AccumulatorSpec) -> str:
    """Return the external input feeding the matmul's ``moving`` operand."""
    _ = spec
    assert len(ctx.inputs_map) >= 2, "matmul accumulator needs two external inputs"
    x_input = next(iter(ctx.inputs_map))
    moving_candidates = [role for role in ctx.inputs_map if role != x_input]
    return ctx.inputs_map[moving_candidates[0]]


def _accumulator_output_tensor(ctx: _RenderContext, spec: AccumulatorSpec) -> str:
    """Return the external-output tensor name this accumulator writes to."""
    output_axes = ctx.op_cls.OUTPUT_AXES
    role_idx = list(output_axes).index(spec.output_role)
    return ctx.outputs_list[role_idx]


@dataclass
class _StationarySource:
    """Matmul stationary source: the pre-declared SBUF buffer + any prelude lines."""

    buffer_name: str
    prelude_lines: list[str]


def _matmul_stationary_source(
    ctx: _RenderContext, matmul_acc_idx: int, a_name: str, out_name: str, acc_tile: int
) -> _StationarySource:
    """Return the SBUF buffer the matmul's transpose should read from.

    Must mirror ``_emit_matmul_accumulator_decls`` exactly: when a
    prior activation_reduce exists, read from its ``sbuf_exp_*``;
    otherwise, use the ``sbuf_a_normed_*`` declared upstream.
    """
    x = ctx.x_name
    p = ctx.partition_size
    prior_exp_idx = next(
        (i for i, pr in enumerate(ctx.accumulators[:matmul_acc_idx]) if pr.kind == "activation_reduce"), None
    )
    if prior_exp_idx is not None:
        prior_out = _accumulator_output_tensor(ctx, ctx.accumulators[prior_exp_idx])
        result = _StationarySource(buffer_name=f"sbuf_exp_{prior_out}", prelude_lines=[])
    else:
        a_access = _sbuf_body_access(ctx, a_name)
        result = _StationarySource(
            buffer_name=f"sbuf_a_normed_{out_name}",
            prelude_lines=[
                f"nisa.tensor_scalar(sbuf_a_normed_{out_name}[0:{p}, 0:{acc_tile}],"
                f" {a_access}, op0=nl.multiply, operand0=sbuf_inv_running_{x}[0:{p}, 0:1])"
            ],
        )
    return result


def _outermost_depth(ctx: _RenderContext, acc_dim: str) -> int:
    """Depth to emit running-state memsets — just before acc_dim's BLOCK loop opens."""
    return block_depth(ctx.dim_order.index(acc_dim))


def _sbuf_body_access(ctx: _RenderContext, tensor_name: str, ptile_binding: dict[str, str] | None = None) -> str:
    """SBUF access string for an external tensor at the composite's innermost slot."""
    buf = sbuf_buffer(ctx.ir, tensor_name)
    placements = ctx.ir.graph.groups[ctx.group_idx].tensor_placements
    tinfo = ctx.ir.context.logical_tensors[tensor_name]
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
    """Axis-access for one dim of an external tensor."""
    tier = placements.get(("sbuf", tensor_name, dim_id), "per_tile")
    block = "0"
    ltile = "0"
    if dim_id in ctx.dim_order:
        if tier == "full":
            block = f"i_block_{dim_id}"
        if tier in ("per_block", "full"):
            ltile = f"i_ltile_{dim_id}"
    return AxisAccess(block=block, ltile=ltile, ptile=ptile_var)


def _nl(value: str | None) -> str:
    """Convert a quoted string op name to an ``nl.*`` reference."""
    raw = "'copy'" if value is None else value
    return f"nl.{raw[1:-1]}" if raw.startswith("'") and raw.endswith("'") else raw
