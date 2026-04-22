"""Online-fusion IR rewrite on ``(KernelContext, KernelGraph)``.

Replaces the X op and its accumulator chain (spread across
singleton groups) with a single group holding one
``NKIOnlineFusionChain`` instance. Absorbed-intermediate ops are
removed; the composite is registered in ``KernelContext`` with
its resolved inputs, outputs, axis map, tile sizes, and blocking
dim.
"""

from collections.abc import Iterator
from dataclasses import replace

from nkigym.kernel_ir.context.context import DimRole, KernelContext, TensorInfo
from nkigym.kernel_ir.graph.fusion_group import FusionGroup
from nkigym.kernel_ir.graph.graph import KernelGraph, rebuild_edges
from nkigym.kernel_ir.rewrites.online_fusion_detect import OnlineFusionCandidate
from nkigym.kernel_ir.rewrites.online_fusion_spec import AccumulatorSpec, InverseStep, ScaleSpec
from nkigym.ops.base import NKIOp
from nkigym.ops.online_fusion_chain import make_online_fusion_class


def _all_ops(graph: KernelGraph) -> list[NKIOp]:
    """Flat walk over all ops across singleton groups."""
    return [op for group in graph.groups for op in group.ops]


def rewrite_one_candidate(
    context: KernelContext, graph: KernelGraph, candidate: OnlineFusionCandidate
) -> tuple[KernelContext, KernelGraph]:
    """Apply one candidate's rewrite; return the mutated ``(context, graph)``."""
    absorbed = _absorbed_ops(context, graph, candidate)
    inputs, outputs = _external_boundary(context, graph, absorbed, candidate)
    composite_cls = _build_composite_class(context, candidate, inputs, outputs)
    composite_op = composite_cls()
    return _reassemble(context, graph, absorbed, composite_op, inputs, outputs, candidate)


def _absorbed_ops(context: KernelContext, graph: KernelGraph, candidate: OnlineFusionCandidate) -> set[NKIOp]:
    """Every op on a data-flow path from X to any accumulator."""
    ops = _all_ops(graph)
    forward = _bfs(context, ops, [candidate.x_op], forward=True)
    backward = _bfs(context, ops, list(candidate.accumulator_ops), forward=False)
    return forward & backward


def _build_adjacency(context: KernelContext, ops: list[NKIOp], forward: bool) -> dict[NKIOp, list[NKIOp]]:
    """Build producer→consumer (forward) or consumer→producer (backward) adjacency."""
    adjacency: dict[NKIOp, list[NKIOp]] = {op: [] for op in ops}
    for producer, consumer, _name in _iter_data_edges(context, ops):
        src, dst = (producer, consumer) if forward else (consumer, producer)
        adjacency[src].append(dst)
    return adjacency


def _iter_data_edges(context: KernelContext, ops: list[NKIOp]) -> Iterator[tuple[NKIOp, NKIOp, str]]:
    """Yield ``(producer, consumer, tensor_name)`` triples via tensor-name matching."""
    for producer in ops:
        for name in context.op_outputs.get(producer, []):
            for consumer in ops:
                if name in context.op_inputs.get(consumer, {}).values():
                    yield producer, consumer, name


def _bfs(context: KernelContext, ops: list[NKIOp], starts: list[NKIOp], forward: bool) -> set[NKIOp]:
    """BFS following data-flow edges (forward/backward) using op-instance identity."""
    adjacency = _build_adjacency(context, ops, forward)
    seen = set(starts)
    stack = list(starts)
    while stack:
        node = stack.pop()
        for neighbor in adjacency.get(node, []):
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return seen


def _external_boundary(
    context: KernelContext, graph: KernelGraph, absorbed: set[NKIOp], candidate: OnlineFusionCandidate
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Return ``(external_inputs, external_outputs)`` for the composite."""
    ops = _all_ops(graph)
    external_in_names = _collect_external_inputs(context, ops, absorbed)
    a_in, b_in = _assign_ab_roles(context, external_in_names, candidate)
    inputs = _inputs_from_roles(a_in, b_in)
    external_out_names = _collect_external_outputs(context, graph, ops, absorbed)
    outputs = [(f"out_{i}", name) for i, name in enumerate(external_out_names)]
    return inputs, outputs


def _inputs_from_roles(a_in: str | None, b_in: str | None) -> list[tuple[str, str]]:
    """Filter optional role bindings into an ordered ``[(role, name)]`` list."""
    result: list[tuple[str, str]] = []
    if a_in is not None:
        result.append(("a_in", a_in))
    if b_in is not None:
        result.append(("b_in", b_in))
    return result


def _collect_external_inputs(context: KernelContext, ops: list[NKIOp], absorbed: set[NKIOp]) -> list[str]:
    """Tensors read by absorbed ops but produced outside the absorbed set."""
    produced_internal: set[str] = set()
    for op in absorbed:
        produced_internal.update(context.op_outputs.get(op, []))
    names: list[str] = []
    seen: set[str] = set()
    for op in ops:
        if op not in absorbed:
            continue
        for tname in context.op_inputs.get(op, {}).values():
            if tname in produced_internal or tname in seen or tname not in context.logical_tensors:
                continue
            names.append(tname)
            seen.add(tname)
    return names


def _collect_external_outputs(
    context: KernelContext, graph: KernelGraph, ops: list[NKIOp], absorbed: set[NKIOp]
) -> list[str]:
    """Tensors produced by absorbed ops and consumed outside (or the kernel return)."""
    names: list[str] = []
    seen: set[str] = set()
    for op in ops:
        if op not in absorbed:
            continue
        for tname in context.op_outputs.get(op, []):
            if tname in seen:
                continue
            if _is_externally_used(context, graph, tname, absorbed):
                names.append(tname)
                seen.add(tname)
    return names


def _assign_ab_roles(
    context: KernelContext, external_names: list[str], candidate: OnlineFusionCandidate
) -> tuple[str | None, str | None]:
    """Assign ``a_in`` (X + stationary side) and ``b_in`` (matmul moving) roles."""
    mm_op: NKIOp | None = None
    for op in candidate.accumulator_ops:
        if type(op).NAME == "nc_matmul":
            mm_op = op
            break
    moving_name: str | None = None
    if mm_op is not None:
        moving_name = context.op_inputs.get(mm_op, {}).get("moving")
    b_in = moving_name if moving_name in external_names else None
    a_candidates = [n for n in external_names if n != b_in]
    a_in = a_candidates[0] if a_candidates else None
    return a_in, b_in


def _is_externally_used(context: KernelContext, graph: KernelGraph, tname: str, absorbed: set[NKIOp]) -> bool:
    """True iff ``tname`` is read by a non-absorbed op or is the kernel return."""
    external_consumer = False
    for op in _all_ops(graph):
        if op in absorbed:
            continue
        if tname in context.op_inputs.get(op, {}).values():
            external_consumer = True
            break
    return tname == context.return_name or external_consumer


def _build_composite_class(
    context: KernelContext,
    candidate: OnlineFusionCandidate,
    inputs: list[tuple[str, str]],
    outputs: list[tuple[str, str]],
) -> type[NKIOp]:
    """Build the ``NKIOnlineFusionChain`` subclass for this candidate."""
    input_axes, input_locs, tile_limits = _build_axis_maps(context, inputs)
    output_axes, output_tile_limits = _build_output_axis_maps(context, outputs)
    tile_limits.update(output_tile_limits)
    scale_spec = _build_scale_spec(candidate, context)
    accumulator_specs = _build_accumulator_specs(candidate, context, outputs)
    return make_online_fusion_class(
        label=candidate.scale_role,
        accumulation_dim=candidate.blocking_dim,
        input_tensor_names=tuple(name for _, name in inputs),
        input_axes=input_axes,
        input_locs=input_locs,
        output_tensor_names=tuple(name for _, name in outputs),
        output_axes=output_axes,
        tile_limits=tile_limits,
        blocking_axes=frozenset(),
        scale_spec=scale_spec,
        accumulator_specs=accumulator_specs,
    )


def _build_axis_maps(
    context: KernelContext, role_pairs: list[tuple[str, str]]
) -> tuple[dict[str, tuple[str, ...]], dict[str, str], dict[str, int]]:
    """Per-input-role axis labels, locs, tile limits."""
    axes: dict[str, tuple[str, ...]] = {}
    locs: dict[str, str] = {}
    tile_limits: dict[str, int] = {}
    for role, tname in role_pairs:
        dims = context.logical_tensors[tname].dim_ids
        labels = tuple(f"{role}_{i}" for i in range(len(dims)))
        axes[role] = labels
        locs[role] = "sbuf"
        for label, dim_id in zip(labels, dims):
            tile_limits[label] = context.dimensions[dim_id].physical_tile_size
    return axes, locs, tile_limits


def _build_output_axis_maps(
    context: KernelContext, role_pairs: list[tuple[str, str]]
) -> tuple[dict[str, tuple[str, ...]], dict[str, int]]:
    """Per-output-role axis labels and tile limits."""
    axes: dict[str, tuple[str, ...]] = {}
    tile_limits: dict[str, int] = {}
    for role, tname in role_pairs:
        dims = context.logical_tensors[tname].dim_ids
        labels = tuple(f"{role}_{i}" for i in range(len(dims)))
        axes[role] = labels
        for label, dim_id in zip(labels, dims):
            tile_limits[label] = context.dimensions[dim_id].physical_tile_size
    return axes, tile_limits


def _build_scale_spec(candidate: OnlineFusionCandidate, context: KernelContext) -> ScaleSpec:
    """Construct the ``ScaleSpec`` for this candidate's scale role."""
    builders = {
        "rsqrt_then_mul": lambda: _scale_spec_rsqrt_then_mul(candidate, context),
        "exp_bias": _scale_spec_exp_bias,
    }
    builder = builders.get(candidate.scale_role)
    if builder is None:
        raise NotImplementedError(f"online-fusion scale_role={candidate.scale_role!r} not yet supported")
    return builder()


def _scale_spec_rsqrt_then_mul(candidate: OnlineFusionCandidate, context: KernelContext) -> ScaleSpec:
    """Extract the sum-family ``ScaleSpec`` from the user's inverse chain."""
    affine_op = _find_affine_op(context, candidate.x_op)
    affine_kwargs = dict(context.op_kwargs.get(affine_op, {})) if affine_op is not None else {}
    inverse_chain = (
        InverseStep(op_name="tensor_scalar", kwargs=affine_kwargs),
        InverseStep(op_name="activation", kwargs={"op": "'rsqrt'"}),
    )
    return ScaleSpec(
        combinator="add",
        init_value="0.0",
        delta_op="activation_reduce",
        delta_kwargs={"op": "'square'", "reduce_op": "'add'"},
        inverse_chain=inverse_chain,
        sigma_kind="ratio_via_reciprocal",
    )


def _scale_spec_exp_bias() -> ScaleSpec:
    """Max-family spec — running stores NEGATED max (reference convention)."""
    return ScaleSpec(
        combinator="minimum",
        init_value="float('inf')",
        delta_op="tensor_reduce",
        delta_kwargs={"op": "'maximum'", "negate": "True"},
        inverse_chain=(),
        sigma_kind="exp_diff_via_activation",
    )


def _build_accumulator_specs(
    candidate: OnlineFusionCandidate, context: KernelContext, outputs: list[tuple[str, str]]
) -> tuple[AccumulatorSpec, ...]:
    """One ``AccumulatorSpec`` per accumulator op."""
    specs: list[AccumulatorSpec] = []
    for acc_op in candidate.accumulator_ops:
        role = _accumulator_output_role(context, acc_op, outputs)
        kind = {"nc_matmul": "matmul", "activation_reduce": "activation_reduce"}.get(type(acc_op).NAME)
        if kind is None:
            raise NotImplementedError(f"accumulator op {type(acc_op).NAME!r} not yet supported")
        specs.append(
            AccumulatorSpec(
                kind=kind,
                output_role=role,
                source_op=acc_op,
                ptile_free_dim=candidate.blocking_dim if kind == "matmul" else "",
                source_kwargs=tuple(context.op_kwargs.get(acc_op, {}).items()),
            )
        )
    return tuple(specs)


def _accumulator_output_role(context: KernelContext, acc_op: NKIOp, outputs: list[tuple[str, str]]) -> str:
    """Return the composite output role for an accumulator op's externally-consumed output."""
    acc_outputs = set(context.op_outputs.get(acc_op, []))
    for role, tname in outputs:
        if tname in acc_outputs:
            return role
    raise ValueError(f"accumulator {acc_op!r} has no external output in {outputs}")


def _find_affine_op(context: KernelContext, x_op: NKIOp) -> NKIOp | None:
    """Locate the user's ``tensor_scalar`` affine op that reads X's reduction output."""
    x_outputs = set(context.op_outputs.get(x_op, []))
    result: NKIOp | None = None
    for op in context.op_inputs:
        if type(op).NAME != "tensor_scalar":
            continue
        if any(name in x_outputs for name in context.op_inputs.get(op, {}).values()):
            result = op
            break
    return result


def _composite_axis_map(
    context: KernelContext, composite_cls: type[NKIOp], inputs: list[tuple[str, str]], outputs: list[tuple[str, str]]
) -> tuple[dict[str, str], dict[str, int]]:
    """Build the composite's axis_map and tile_sizes dicts."""
    axis_map: dict[str, str] = {}
    tile_sizes: dict[str, int] = {}
    for role, tname in inputs:
        for label, dim_id in zip(composite_cls.OPERAND_AXES[role], context.logical_tensors[tname].dim_ids):
            axis_map[label] = dim_id
            tile_sizes[dim_id] = context.dimensions[dim_id].physical_tile_size
    for role, tname in outputs:
        for label, dim_id in zip(composite_cls.OUTPUT_AXES[role], context.logical_tensors[tname].dim_ids):
            axis_map[label] = dim_id
            tile_sizes[dim_id] = context.dimensions[dim_id].physical_tile_size
    return axis_map, tile_sizes


def _reassemble(
    context: KernelContext,
    graph: KernelGraph,
    absorbed: set[NKIOp],
    composite_op: NKIOp,
    inputs: list[tuple[str, str]],
    outputs: list[tuple[str, str]],
    candidate: OnlineFusionCandidate,
) -> tuple[KernelContext, KernelGraph]:
    """Build the post-rewrite ``(context, graph)`` pair."""
    composite_cls = type(composite_op)
    composite_inputs = dict(inputs)
    composite_outputs = [name for _, name in outputs]
    composite_axis_map, composite_tile_sizes = _composite_axis_map(context, composite_cls, inputs, outputs)

    new_op_inputs = {op: v for op, v in context.op_inputs.items() if op not in absorbed}
    new_op_outputs = {op: v for op, v in context.op_outputs.items() if op not in absorbed}
    new_op_kwargs = {op: v for op, v in context.op_kwargs.items() if op not in absorbed}
    new_op_axis_map = {op: v for op, v in context.op_axis_map.items() if op not in absorbed}
    new_op_tile_sizes = {op: v for op, v in context.op_tile_sizes.items() if op not in absorbed}
    new_op_blocking_dims = {op: v for op, v in context.op_blocking_dims.items() if op not in absorbed}

    new_op_inputs[composite_op] = composite_inputs
    new_op_outputs[composite_op] = composite_outputs
    new_op_kwargs[composite_op] = {}
    new_op_axis_map[composite_op] = composite_axis_map
    new_op_tile_sizes[composite_op] = composite_tile_sizes
    new_op_blocking_dims[composite_op] = {candidate.blocking_dim}

    new_dims = dict(context.dimensions)
    new_dims[candidate.blocking_dim] = replace(new_dims[candidate.blocking_dim], role=DimRole.ACCUMULATION)
    new_logical_tensors = _retain_external_tensors(context, graph, absorbed, outputs)

    new_context = replace(
        context,
        dimensions=new_dims,
        logical_tensors=new_logical_tensors,
        op_inputs=new_op_inputs,
        op_outputs=new_op_outputs,
        op_kwargs=new_op_kwargs,
        op_axis_map=new_op_axis_map,
        op_tile_sizes=new_op_tile_sizes,
        op_blocking_dims=new_op_blocking_dims,
    )

    survivors: list[FusionGroup] = []
    inserted = False
    for group in graph.groups:
        surviving_ops = [op for op in group.ops if op not in absorbed]
        if not surviving_ops and not inserted and any(op in absorbed for op in group.ops):
            survivors.append(FusionGroup(ops=[composite_op]))
            inserted = True
        elif surviving_ops:
            survivors.append(
                FusionGroup(
                    ops=surviving_ops,
                    dim_order=list(group.dim_order),
                    buffer_degrees=dict(group.buffer_degrees),
                    tensor_placements=dict(group.tensor_placements),
                    skip_spec=group.skip_spec,
                )
            )
    if not inserted:
        survivors.append(FusionGroup(ops=[composite_op]))
    new_graph = KernelGraph(groups=survivors)
    rebuild_edges(new_graph, new_context)
    return new_context, new_graph


def _retain_external_tensors(
    context: KernelContext, graph: KernelGraph, absorbed: set[NKIOp], outputs: list[tuple[str, str]]
) -> dict[str, TensorInfo]:
    """Drop tensors that only absorbed-and-internal ops referenced."""
    external_names = {name for _, name in outputs}
    referenced: set[str] = set(external_names)
    for op in _all_ops(graph):
        if op in absorbed:
            continue
        for name in context.op_inputs.get(op, {}).values():
            referenced.add(name)
        for name in context.op_outputs.get(op, []):
            referenced.add(name)
    produced_in_original: set[str] = set()
    for op in _all_ops(graph):
        produced_in_original.update(context.op_outputs.get(op, []))
    referenced.update(name for name in context.logical_tensors if name not in produced_in_original)
    return {name: tinfo for name, tinfo in context.logical_tensors.items() if name in referenced}
