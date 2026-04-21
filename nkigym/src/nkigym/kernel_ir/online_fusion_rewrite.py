"""Online-fusion IR rewrite.

Replaces the X op and its accumulator chain with a single composite
``NKIOnlineFusionChain`` subclass node. The composite subsumes the
entire producer→accumulator chain; rendered via
``codegen.online_fusion`` as one fused loop body. See
``online_fusion_plan.md`` for the algorithm and contract.
"""

from dataclasses import replace

from nkigym.kernel_ir.dim_analysis import DimAnalysis, DimRole, TensorInfo
from nkigym.kernel_ir.online_fusion_detect import OnlineFusionCandidate
from nkigym.kernel_ir.online_fusion_spec import AccumulatorSpec, InverseStep, ScaleSpec
from nkigym.kernel_ir.op_graph import OpGraph
from nkigym.ops.base import NKIOp
from nkigym.ops.online_fusion_chain import make_online_fusion_class


def rewrite_one_candidate(
    da: DimAnalysis, graph: OpGraph, candidate: OnlineFusionCandidate
) -> tuple[DimAnalysis, OpGraph]:
    """Apply one candidate's rewrite; return the mutated (da, graph).

    Public entry point for the ``OnlineFusionPattern.apply`` driver
    step. Unchanged behavior vs the prior ``_rewrite_candidate``
    helper — renamed so the pattern can import it without touching
    a private API.
    """
    absorbed = _absorbed_ops(graph, candidate.x_op_idx, candidate.accumulator_op_indices)
    inputs, outputs = _external_boundary(da, graph, absorbed, candidate)
    return_name = _rewrite_return_name(da, outputs, absorbed)
    composite_cls = _build_composite_class(da, graph, candidate, inputs, outputs)
    new_graph = _reassemble_graph(graph, absorbed, composite_cls, inputs, outputs)
    new_da = _reassemble_dim_analysis(da, graph, absorbed, composite_cls, inputs, outputs, return_name, candidate)
    return new_da, new_graph


def _absorbed_ops(graph: OpGraph, x_op_idx: int, accumulator_op_indices: tuple[int, ...]) -> set[int]:
    """Every op on a data-flow path from X to any accumulator.

    Walks forward from ``x_op_idx`` and backward from each
    accumulator; intersection is the absorbed set. Includes both
    endpoints.
    """
    forward = _reachable(graph, [x_op_idx], direction="forward")
    backward = _reachable(graph, list(accumulator_op_indices), direction="backward")
    return forward & backward


def _reachable(graph: OpGraph, starts: list[int], direction: str) -> set[int]:
    """BFS from ``starts`` following ``edges`` in ``direction`` (forward/backward)."""
    n = len(graph.op_classes)
    adjacency: list[list[int]] = [[] for _ in range(n)]
    for producer, consumer, _tensor, _role in graph.edges:
        if direction == "forward":
            adjacency[producer].append(consumer)
        else:
            adjacency[consumer].append(producer)
    seen = set(starts)
    stack = list(starts)
    while stack:
        node = stack.pop()
        for neighbor in adjacency[node]:
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return seen


def _external_boundary(
    da: DimAnalysis, graph: OpGraph, absorbed: set[int], candidate: OnlineFusionCandidate
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Return ``(external_inputs, external_outputs)`` for the composite.

    Inputs use semantic role names (``a_in`` = X + stationary data,
    ``b_in`` = matmul moving RHS). Outputs use positional
    ``out_i`` roles so multi-accumulator composites disambiguate.
    """
    produced_internal: set[str] = set()
    for op_idx in absorbed:
        produced_internal.update(graph.op_tensors[op_idx][1])
    external_in_names: list[str] = []
    seen_in: set[str] = set()
    for op_idx in sorted(absorbed):
        for tname in graph.op_tensors[op_idx][0].values():
            if tname in produced_internal or tname in seen_in or tname not in da.tensors:
                continue
            external_in_names.append(tname)
            seen_in.add(tname)
    a_in, b_in = _assign_ab_roles(graph, absorbed, external_in_names, candidate)
    inputs: list[tuple[str, str]] = []
    if a_in is not None:
        inputs.append(("a_in", a_in))
    if b_in is not None:
        inputs.append(("b_in", b_in))
    external_out_names: list[str] = []
    seen_out: set[str] = set()
    for op_idx in sorted(absorbed):
        for tname in graph.op_tensors[op_idx][1]:
            if tname in seen_out:
                continue
            if _is_externally_used(graph, tname, absorbed, da.return_name):
                external_out_names.append(tname)
                seen_out.add(tname)
    outputs = [(f"out_{i}", name) for i, name in enumerate(external_out_names)]
    return inputs, outputs


def _assign_ab_roles(
    graph: OpGraph, absorbed: set[int], external_names: list[str], candidate: OnlineFusionCandidate
) -> tuple[str | None, str | None]:
    """Assign ``a_in`` (X + stationary side) and ``b_in`` (matmul moving) roles.

    Uses the accumulator matmul's ``moving`` input to identify
    ``b_in``; ``a_in`` is the remaining external input (there
    should be exactly one for ``rsqrt_then_mul``).
    """
    mm_idx = next((i for i in candidate.accumulator_op_indices if graph.op_classes[i].NAME == "nc_matmul"), None)
    moving_name: str | None = None
    if mm_idx is not None:
        moving_name = graph.op_tensors[mm_idx][0].get("moving")
    b_in = moving_name if moving_name in external_names else None
    a_candidates = [n for n in external_names if n != b_in]
    a_in = a_candidates[0] if a_candidates else None
    return a_in, b_in


def _is_externally_used(graph: OpGraph, tname: str, absorbed: set[int], return_name: str) -> bool:
    """True iff ``tname`` is read by a non-absorbed op or is the kernel return."""
    external_consumer = any(
        tname in graph.op_tensors[consumer][0].values()
        for consumer in range(len(graph.op_classes))
        if consumer not in absorbed
    )
    return tname == return_name or external_consumer


def _rewrite_return_name(da: DimAnalysis, outputs: list[tuple[str, str]], absorbed: set[int]) -> str:
    """Kernel return name stays the same — composite adopts the original output tensor name."""
    _ = absorbed
    _ = outputs
    return da.return_name


def _build_composite_class(
    da: DimAnalysis,
    graph: OpGraph,
    candidate: OnlineFusionCandidate,
    inputs: list[tuple[str, str]],
    outputs: list[tuple[str, str]],
) -> type[NKIOp]:
    """Build the ``NKIOnlineFusionChain`` subclass for this candidate.

    Parametric: derives a ``ScaleSpec`` from the candidate's
    ``scale_role`` and the X op's kwargs, and one
    ``AccumulatorSpec`` per accumulator op. The render path reads
    these specs to emit the fused loop body without any role-
    string branching.
    """
    input_axes, input_locs, tile_limits = _build_axis_maps(da, inputs)
    output_axes, output_tile_limits = _build_output_axis_maps(da, outputs)
    tile_limits.update(output_tile_limits)
    scale_spec = _build_scale_spec(candidate, graph)
    accumulator_specs = _build_accumulator_specs(candidate, graph, outputs)
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
    da: DimAnalysis, role_pairs: list[tuple[str, str]]
) -> tuple[dict[str, tuple[str, ...]], dict[str, str], dict[str, int]]:
    """Per-input-role axis labels (unique-per-role prefix), locs, tile limits."""
    axes: dict[str, tuple[str, ...]] = {}
    locs: dict[str, str] = {}
    tile_limits: dict[str, int] = {}
    for role, tname in role_pairs:
        dims = da.tensors[tname].dim_ids
        labels = tuple(f"{role}_{i}" for i in range(len(dims)))
        axes[role] = labels
        locs[role] = "sbuf"
        for label, dim_id in zip(labels, dims):
            tile_limits[label] = da.dims[dim_id].physical_tile_size
    return axes, locs, tile_limits


def _build_output_axis_maps(
    da: DimAnalysis, role_pairs: list[tuple[str, str]]
) -> tuple[dict[str, tuple[str, ...]], dict[str, int]]:
    """Build per-output-role axis labels and tile limits (no locs on outputs)."""
    axes: dict[str, tuple[str, ...]] = {}
    tile_limits: dict[str, int] = {}
    for role, tname in role_pairs:
        dims = da.tensors[tname].dim_ids
        labels = tuple(f"{role}_{i}" for i in range(len(dims)))
        axes[role] = labels
        for label, dim_id in zip(labels, dims):
            tile_limits[label] = da.dims[dim_id].physical_tile_size
    return axes, tile_limits


def _build_scale_spec(candidate: OnlineFusionCandidate, graph: OpGraph) -> ScaleSpec:
    """Construct the ``ScaleSpec`` for this candidate's scale role.

    * ``rsqrt_then_mul``: sum-like running X, inverse chain of
      ``tensor_scalar(×1/K +eps)`` then ``activation(rsqrt)`` read
      off the user's graph, σ = inv(running) · reciprocal(inv(prev)).
    * ``exp_bias``: max-like running X, no inverse chain (σ is
      produced by a single ``activation(exp, data=prev,
      bias=running, scale=-1.0)``).
    """
    builders = {
        "rsqrt_then_mul": lambda: _scale_spec_rsqrt_then_mul(candidate, graph),
        "exp_bias": _scale_spec_exp_bias,
    }
    builder = builders.get(candidate.scale_role)
    if builder is None:
        raise NotImplementedError(f"online-fusion scale_role={candidate.scale_role!r} not yet supported")
    return builder()


def _scale_spec_rsqrt_then_mul(candidate: OnlineFusionCandidate, graph: OpGraph) -> ScaleSpec:
    """Extract the sum-family ``ScaleSpec`` from the user's inverse chain."""
    affine_idx = _find_affine_op(graph, candidate.x_op_idx, candidate.accumulator_op_indices)
    affine_kwargs = dict(graph.op_all_kwargs[affine_idx]) if affine_idx is not None else {}
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
    candidate: OnlineFusionCandidate, graph: OpGraph, outputs: list[tuple[str, str]]
) -> tuple[AccumulatorSpec, ...]:
    """One ``AccumulatorSpec`` per accumulator op.

    The output role assigned to each accumulator mirrors the
    output role in the composite's ``OUTPUT_AXES`` — the detector
    guarantees an external output exists for each accumulator that
    surviving downstream ops read.
    """
    specs: list[AccumulatorSpec] = []
    for acc_idx in candidate.accumulator_op_indices:
        op_cls = graph.op_classes[acc_idx]
        role = _accumulator_output_role(graph, acc_idx, outputs)
        kind = {"nc_matmul": "matmul", "activation_reduce": "activation_reduce"}.get(op_cls.NAME)
        if kind is None:
            raise NotImplementedError(f"accumulator op {op_cls.NAME!r} not yet supported")
        specs.append(
            AccumulatorSpec(
                kind=kind,
                output_role=role,
                source_op_idx=acc_idx,
                ptile_free_dim=candidate.blocking_dim if kind == "matmul" else "",
                source_kwargs=tuple(graph.op_all_kwargs[acc_idx].items()),
            )
        )
    return tuple(specs)


def _accumulator_output_role(graph: OpGraph, acc_idx: int, outputs: list[tuple[str, str]]) -> str:
    """Return the composite output role for an accumulator op's externally-consumed output.

    Accumulators produce one or more output tensors; at least one
    of them is in the composite's external output list (otherwise
    the accumulator wouldn't be user-visible). Pick the first
    match.
    """
    acc_outputs = set(graph.op_tensors[acc_idx][1])
    for role, tname in outputs:
        if tname in acc_outputs:
            return role
    raise ValueError(f"accumulator {acc_idx} has no external output in {outputs}")


def _find_affine_op(graph: OpGraph, x_op_idx: int, accumulator_op_indices: tuple[int, ...]) -> int | None:
    """Locate the user's ``tensor_scalar`` affine op that reads X's reduction output.

    This is the first ``tensor_scalar`` consumer of the X op's
    reduced-axis output. Its kwargs carry the ``op0`` / ``operand0``
    / ``op1`` / ``operand1`` for the ``x/K + eps`` pattern that
    the composite re-emits on both ``running_s`` and ``prev_s``.
    """
    _ = accumulator_op_indices
    x_outputs = set(graph.op_tensors[x_op_idx][1])
    matches = [
        op_idx
        for op_idx, op_cls in enumerate(graph.op_classes)
        if op_cls.NAME == "tensor_scalar" and any(name in x_outputs for name in graph.op_tensors[op_idx][0].values())
    ]
    return matches[0] if matches else None


def _reassemble_graph(
    graph: OpGraph,
    absorbed: set[int],
    composite_cls: type[NKIOp],
    inputs: list[tuple[str, str]],
    outputs: list[tuple[str, str]],
) -> OpGraph:
    """Keep non-absorbed ops; insert composite topologically via ``_composite_insert_position``.

    Consumer edges require producer-before-consumer in op-index
    order (``_rebuild_edges`` walks index-sorted), so the composite
    must slot between its surviving producers and consumers.
    """
    survivors = [op_idx for op_idx in range(len(graph.op_classes)) if op_idx not in absorbed]
    insert_at = _composite_insert_position(graph, absorbed, inputs, outputs)
    new_op_classes: list[type[NKIOp]] = []
    new_op_tensors: list[tuple[dict[str, str], list[str]]] = []
    new_op_all_kwargs: list[dict[str, str]] = []
    composite_inputs = {role: name for role, name in inputs}
    composite_outputs = [name for _, name in outputs]
    inserted = False
    for orig_idx in survivors:
        if not inserted and orig_idx >= insert_at:
            new_op_classes.append(composite_cls)
            new_op_tensors.append((composite_inputs, composite_outputs))
            new_op_all_kwargs.append({})
            inserted = True
        new_op_classes.append(graph.op_classes[orig_idx])
        new_op_tensors.append((dict(graph.op_tensors[orig_idx][0]), list(graph.op_tensors[orig_idx][1])))
        new_op_all_kwargs.append(dict(graph.op_all_kwargs[orig_idx]))
    if not inserted:
        new_op_classes.append(composite_cls)
        new_op_tensors.append((composite_inputs, composite_outputs))
        new_op_all_kwargs.append({})
    edges = _rebuild_edges(new_op_tensors)
    return OpGraph(op_classes=new_op_classes, edges=edges, op_tensors=new_op_tensors, op_all_kwargs=new_op_all_kwargs)


def _composite_insert_position(
    graph: OpGraph, absorbed: set[int], inputs: list[tuple[str, str]], outputs: list[tuple[str, str]]
) -> int:
    """Earliest survivor-index the composite can legally occupy.

    Must sit after every external-input producer and before every
    external-output consumer. Defaults to ``min(absorbed)`` when
    no consumer survives.
    """
    input_names = {name for _, name in inputs}
    output_names = {name for _, name in outputs}
    producers_max = -1
    for op_idx in range(len(graph.op_classes)):
        if op_idx in absorbed:
            continue
        if any(out in input_names for out in graph.op_tensors[op_idx][1]):
            producers_max = max(producers_max, op_idx)
    consumers_min = len(graph.op_classes)
    for op_idx in range(len(graph.op_classes)):
        if op_idx in absorbed:
            continue
        if any(inp in output_names for inp in graph.op_tensors[op_idx][0].values()):
            consumers_min = min(consumers_min, op_idx)
    absorbed_min = min(absorbed) if absorbed else producers_max + 1
    return max(producers_max + 1, min(absorbed_min, consumers_min))


def _rebuild_edges(op_tensors: list[tuple[dict[str, str], list[str]]]) -> list[tuple[int, int, str, str]]:
    """Recompute edges from the (possibly mutated) op-tensor layout."""
    edges: list[tuple[int, int, str, str]] = []
    producer_of: dict[str, int] = {}
    for op_idx, (inputs, outputs) in enumerate(op_tensors):
        for role, tname in inputs.items():
            producer = producer_of.get(tname)
            if producer is not None:
                edges.append((producer, op_idx, tname, role))
        for oname in outputs:
            producer_of[oname] = op_idx
    return edges


def _reassemble_dim_analysis(
    da: DimAnalysis,
    graph: OpGraph,
    absorbed: set[int],
    composite_cls: type[NKIOp],
    inputs: list[tuple[str, str]],
    outputs: list[tuple[str, str]],
    return_name: str,
    candidate: OnlineFusionCandidate,
) -> DimAnalysis:
    """Per-op arrays match the new graph's op order (composite inserted at same slot as in ``_reassemble_graph``)."""
    insert_at = _composite_insert_position(graph, absorbed, inputs, outputs)
    survivors = [op_idx for op_idx in range(len(graph.op_classes)) if op_idx not in absorbed]
    composite_axis_map, composite_tile_sizes = _composite_axis_map(da, composite_cls, inputs, outputs)
    new_per_op_axis_maps: list[dict[str, str]] = []
    new_op_tile_sizes: list[dict[str, int]] = []
    new_per_op_blocking_dims: list[set[str]] = []
    inserted = False
    for orig_idx in survivors:
        if not inserted and orig_idx >= insert_at:
            new_per_op_axis_maps.append(composite_axis_map)
            new_op_tile_sizes.append(composite_tile_sizes)
            new_per_op_blocking_dims.append({candidate.blocking_dim})
            inserted = True
        new_per_op_axis_maps.append(dict(da.per_op_axis_maps[orig_idx]))
        new_op_tile_sizes.append(dict(da.op_tile_sizes[orig_idx]))
        new_per_op_blocking_dims.append(set(da.per_op_blocking_dims[orig_idx]))
    if not inserted:
        new_per_op_axis_maps.append(composite_axis_map)
        new_op_tile_sizes.append(composite_tile_sizes)
        new_per_op_blocking_dims.append({candidate.blocking_dim})
    new_dims = dict(da.dims)
    new_dims[candidate.blocking_dim] = replace(new_dims[candidate.blocking_dim], role=DimRole.ACCUMULATION)
    new_tensors = _retain_external_tensors(da.tensors, graph, absorbed, outputs)
    return replace(
        da,
        dims=new_dims,
        tensors=new_tensors,
        per_op_axis_maps=new_per_op_axis_maps,
        op_tile_sizes=new_op_tile_sizes,
        per_op_blocking_dims=new_per_op_blocking_dims,
        return_name=return_name,
    )


def _composite_axis_map(
    da: DimAnalysis, composite_cls: type[NKIOp], inputs: list[tuple[str, str]], outputs: list[tuple[str, str]]
) -> tuple[dict[str, str], dict[str, int]]:
    """Build the composite's axis_map and tile_sizes dicts.

    ``axis_map`` maps every abstract label in
    ``OPERAND_AXES`` / ``OUTPUT_AXES`` to a concrete dim id.
    ``tile_sizes`` maps concrete dim ids to their physical tile.
    """
    axis_map: dict[str, str] = {}
    tile_sizes: dict[str, int] = {}
    for role, tname in inputs:
        for label, dim_id in zip(composite_cls.OPERAND_AXES[role], da.tensors[tname].dim_ids):
            axis_map[label] = dim_id
            tile_sizes[dim_id] = da.dims[dim_id].physical_tile_size
    for role, tname in outputs:
        for label, dim_id in zip(composite_cls.OUTPUT_AXES[role], da.tensors[tname].dim_ids):
            axis_map[label] = dim_id
            tile_sizes[dim_id] = da.dims[dim_id].physical_tile_size
    return axis_map, tile_sizes


def _retain_external_tensors(
    tensors: dict[str, TensorInfo], graph: OpGraph, absorbed: set[int], outputs: list[tuple[str, str]]
) -> dict[str, TensorInfo]:
    """Drop tensors that only the absorbed-and-internal ops referenced.

    Kernel inputs (tensors with no producer in the ORIGINAL graph)
    are retained. Outputs claimed by the composite are retained.
    Tensors still read or written by surviving non-composite ops
    are retained. Purely absorbed-internal intermediates are dropped
    so ``tensor_buffers`` doesn't allocate for them.
    """
    external_names = {name for _, name in outputs}
    survivors = [op_idx for op_idx in range(len(graph.op_classes)) if op_idx not in absorbed]
    referenced: set[str] = set(external_names)
    for op_idx in survivors:
        for name in graph.op_tensors[op_idx][0].values():
            referenced.add(name)
        for name in graph.op_tensors[op_idx][1]:
            referenced.add(name)
    produced_in_original: set[str] = set()
    for op_idx in range(len(graph.op_classes)):
        produced_in_original.update(graph.op_tensors[op_idx][1])
    referenced.update(name for name in tensors if name not in produced_in_original)
    return {name: tinfo for name, tinfo in tensors.items() if name in referenced}
