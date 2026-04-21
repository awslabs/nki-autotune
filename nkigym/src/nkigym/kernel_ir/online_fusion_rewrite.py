"""Online-fusion IR rewrite.

Replaces the X op and its accumulator chain with a single composite
op node in the op graph. The composite (``NKIOnlineFusionChain``
subclass, built per candidate) subsumes the entire producer→
accumulator chain and renders as one fused loop body through
``codegen.online_fusion``.

Algorithm per candidate:

1. Walk the op graph to find every op on a path from ``x_op_idx``
   to an accumulator. These "absorbed" ops, plus X itself, are
   removed from the rewritten graph.
2. Determine external inputs (tensors read by absorbed ops but
   produced outside the absorbed set) and external outputs
   (tensors consumed outside the absorbed set, plus the kernel
   return tensor if it was produced inside).
3. Build an ``NKIOnlineFusionChain`` subclass with concrete
   ``OPERAND_AXES`` / ``OUTPUT_AXES`` / etc. wired to the external
   tensors' concrete dims.
4. Append the composite op to the graph; delete the absorbed ops;
   renumber indices.
5. Promote the accumulation dim to ``ACCUMULATION`` in ``da.dims``.
6. Retarget the kernel return to the composite's output (if it
   was among the absorbed ops' outputs).

Forced-merge clusters are empty — the composite is already one
node, so the partition sampler keeps it in a singleton group
naturally.
"""

from dataclasses import replace

from nkigym.kernel_ir.dim_analysis import DimAnalysis, DimRole, TensorInfo
from nkigym.kernel_ir.online_fusion_detect import OnlineFusionCandidate
from nkigym.kernel_ir.op_graph import OpGraph
from nkigym.ops.base import NKIOp
from nkigym.ops.online_fusion_chain import make_online_fusion_class


def apply_online_fusion(
    da: DimAnalysis, graph: OpGraph, candidates: list[OnlineFusionCandidate]
) -> tuple[DimAnalysis, OpGraph, list[frozenset[int]]]:
    """Rewrite ``da`` / ``graph`` for every supported candidate.

    Returns the rewritten dim analysis, op graph, and forced-merge
    clusters (empty for composite-node rewrites).
    """
    supported = [c for c in candidates if c.scale_role == "rsqrt_then_mul"]
    current_da = da
    current_graph = graph
    for candidate in supported:
        current_da, current_graph = _rewrite_candidate(current_da, current_graph, candidate)
    return current_da, current_graph, []


def _rewrite_candidate(
    da: DimAnalysis, graph: OpGraph, candidate: OnlineFusionCandidate
) -> tuple[DimAnalysis, OpGraph]:
    """Apply one candidate's rewrite; return the mutated (da, graph)."""
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

    Role names are semantic for the scale-role family so the
    render path can address each input by its semantic role rather
    than positional index:

    * ``a_in``: the external input feeding BOTH X's reduction and
      the accumulator's stationary-side chain (the "data" being
      normalized). For ``rsqrt_then_mul``, this is the pre-norm
      tensor.
    * ``b_in``: the external input feeding ONLY the accumulator's
      moving-side (the matmul RHS).
    * ``out``: the composite's output tensor — adopts the
      accumulator's original output name so downstream consumers
      don't need rewiring.
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
    outputs = [("out", name) for name in external_out_names]
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

    Abstract-axis labels are generated per role with a unique
    prefix so the axis-map inversion is unambiguous. Only the
    affine-op kwargs from the user's inverse chain are carried
    through — the composite's render path needs them to reproduce
    ``running_s/K + eps`` on both the current and previous
    running-sum values.
    """
    input_axes: dict[str, tuple[str, ...]] = {}
    input_locs: dict[str, str] = {}
    tile_limits: dict[str, int] = {}
    for role, tname in inputs:
        dims = da.tensors[tname].dim_ids
        labels = tuple(f"{role}_{i}" for i in range(len(dims)))
        input_axes[role] = labels
        input_locs[role] = "sbuf"
        for label, dim_id in zip(labels, dims):
            tile_limits[label] = da.dims[dim_id].physical_tile_size
    output_axes: dict[str, tuple[str, ...]] = {}
    for role, tname in outputs:
        dims = da.tensors[tname].dim_ids
        labels = tuple(f"{role}_{i}" for i in range(len(dims)))
        output_axes[role] = labels
        for label, dim_id in zip(labels, dims):
            tile_limits[label] = da.dims[dim_id].physical_tile_size
    affine_op_idx = _find_affine_op(graph, candidate.x_op_idx, candidate.accumulator_op_indices)
    affine_kwargs = dict(graph.op_all_kwargs[affine_op_idx]) if affine_op_idx is not None else {}
    return make_online_fusion_class(
        scale_role=candidate.scale_role,
        accumulation_dim=candidate.blocking_dim,
        input_tensor_names=tuple(name for _, name in inputs),
        input_axes=input_axes,
        input_locs=input_locs,
        output_tensor_names=tuple(name for _, name in outputs),
        output_axes=output_axes,
        tile_limits=tile_limits,
        blocking_axes=frozenset(),
        inner_op_kwargs=(affine_kwargs,),
    )


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
    """Build the new op graph: keep non-absorbed ops, append composite, renumber."""
    survivors = [op_idx for op_idx in range(len(graph.op_classes)) if op_idx not in absorbed]
    new_op_classes = [graph.op_classes[op_idx] for op_idx in survivors]
    new_op_tensors = [(dict(graph.op_tensors[op_idx][0]), list(graph.op_tensors[op_idx][1])) for op_idx in survivors]
    new_op_all_kwargs = [dict(graph.op_all_kwargs[op_idx]) for op_idx in survivors]
    new_op_classes.append(composite_cls)
    composite_inputs = {role: name for role, name in inputs}
    composite_outputs = [name for _, name in outputs]
    new_op_tensors.append((composite_inputs, composite_outputs))
    new_op_all_kwargs.append({})
    edges = _rebuild_edges(new_op_tensors)
    return OpGraph(op_classes=new_op_classes, edges=edges, op_tensors=new_op_tensors, op_all_kwargs=new_op_all_kwargs)


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
    """Build a DimAnalysis consistent with the rewritten graph.

    Drops per-op entries for absorbed ops; appends one entry for
    the composite. Promotes the candidate's blocking dim to
    ``ACCUMULATION``. Keeps every tensor in ``da.tensors`` —
    internal intermediates absorbed by the composite are still
    referenced by the original (now-dropped) ops in upstream
    callers; leaving them behind is harmless because the renderer
    only allocates for tensors produced by surviving ops.
    """
    survivors = [op_idx for op_idx in range(len(graph.op_classes)) if op_idx not in absorbed]
    new_per_op_axis_maps = [dict(da.per_op_axis_maps[op_idx]) for op_idx in survivors]
    new_op_tile_sizes = [dict(da.op_tile_sizes[op_idx]) for op_idx in survivors]
    new_per_op_blocking_dims = [set(da.per_op_blocking_dims[op_idx]) for op_idx in survivors]
    composite_axis_map, composite_tile_sizes = _composite_axis_map(da, composite_cls, inputs, outputs)
    new_per_op_axis_maps.append(composite_axis_map)
    new_op_tile_sizes.append(composite_tile_sizes)
    new_per_op_blocking_dims.append(set())
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
