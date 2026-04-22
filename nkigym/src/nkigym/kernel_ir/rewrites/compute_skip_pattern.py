"""``ComputeSkipPattern`` — absorb an ``NKIAffineSelect``'s producers AND consumers.

Lifts the causal-mask predicate from a ``NKIAffineSelect`` op to
tile granularity. Detects one ``NKIAffineSelect`` whose free-axis
pattern is a single ``[step, count]`` pair (the causal-mask
shape), then walks:

* **Upstream** — ops whose outputs flow exclusively into the
  mask's data input. These run on ``mask_and_compute`` /
  ``compute_only`` tiles; skipped on ``skip_all`` tiles because
  their result feeds only the masked chain.
* **Downstream** — ops whose inputs originate from the mask's
  output, in a chain shape that propagates ``-inf → 0`` through
  softmax-style ops (``tensor_scalar`` scale, ``tensor_reduce``
  max, ``activation_reduce`` exp+add, ``activation`` reciprocal,
  ``nc_matmul``, ``tensor_scalar`` final multiply) or absorbed
  composites (``NKIOnlineFusionChain``). These run on
  ``mask_and_compute`` / ``compute_only`` tiles; skipped on
  ``skip_all`` tiles.

Apply: absorb every identified op into one ``FusionGroup`` and
tag it with a ``ComputeSkipSpec``. The codegen picks up the spec
and wraps those ops' inner-body emission with a three-state
``if`` classifier keyed on the tile's ``(partition, free)``
starts.
"""

import ast
from dataclasses import dataclass, replace

from nkigym.kernel_ir.context.context import KernelContext
from nkigym.kernel_ir.graph.compute_skip_spec import ComputeSkipSpec
from nkigym.kernel_ir.graph.fusion_group import FusionGroup
from nkigym.kernel_ir.graph.graph import KernelGraph, rebuild_edges
from nkigym.ops.base import NKIOp


@dataclass(frozen=True)
class _Match:
    """One match instance — the affine-select op + absorbed partners."""

    affine_select_op: NKIOp
    upstream_ops: tuple[NKIOp, ...]
    downstream_ops: tuple[NKIOp, ...]


class ComputeSkipPattern:
    """Pattern-rewrite: absorb an ``NKIAffineSelect`` into a skip-annotated group."""

    name = "compute_skip"

    def match(self, context: KernelContext, graph: KernelGraph) -> list[_Match]:
        """Return every ``NKIAffineSelect`` whose predicate supports tile skipping."""
        matches: list[_Match] = []
        for group in graph.groups:
            for op in group.ops:
                if type(op).NAME != "affine_select":
                    continue
                if _already_absorbed(graph, op):
                    continue
                if not _predicate_classifiable(context, op):
                    continue
                up, down = _trace_chain(context, graph, op)
                matches.append(_Match(affine_select_op=op, upstream_ops=tuple(up), downstream_ops=tuple(down)))
        return matches

    def apply(self, context: KernelContext, graph: KernelGraph, instance: _Match) -> tuple[KernelContext, KernelGraph]:
        """Absorb the chain into one skip-annotated ``FusionGroup``."""
        absorbed = {instance.affine_select_op, *instance.upstream_ops, *instance.downstream_ops}
        spec = _build_spec(context, graph, instance, absorbed)
        new_groups = _rebuild_groups(graph, absorbed, instance, spec)
        new_graph = KernelGraph(groups=new_groups)
        rebuild_edges(new_graph, context)
        return context, new_graph


def _already_absorbed(graph: KernelGraph, op: NKIOp) -> bool:
    """True iff ``op``'s group already has a ``skip_spec`` (don't re-match)."""
    absorbed = False
    for group in graph.groups:
        if op in group.ops and group.skip_spec is not None:
            absorbed = True
            break
    return absorbed


def _predicate_classifiable(context: KernelContext, op: NKIOp) -> bool:
    """True iff the affine_select's free-axis pattern is a single ``[step, count]`` pair."""
    kwargs = context.op_kwargs.get(op, {})
    pattern_raw = kwargs.get("pattern")
    classifiable = False
    if pattern_raw is not None:
        parsed = ast.literal_eval(pattern_raw)
        classifiable = isinstance(parsed, list) and len(parsed) == 1 and len(parsed[0]) == 2
    return classifiable


def _trace_chain(context: KernelContext, graph: KernelGraph, affine_op: NKIOp) -> tuple[list[NKIOp], list[NKIOp]]:
    """Return ``(upstream_ops, downstream_ops)`` absorbed around ``affine_op``."""
    all_ops = [op for group in graph.groups for op in group.ops]
    producers_of = _producers_of_map(context, all_ops)
    consumers_of = _consumers_of_map(context, all_ops)
    affine_inputs = list(context.op_inputs.get(affine_op, {}).values())
    affine_outputs = list(context.op_outputs.get(affine_op, []))
    upstream = _trace_upstream(context, all_ops, affine_op, affine_inputs, producers_of, consumers_of)
    downstream = _trace_downstream(context, all_ops, affine_op, affine_outputs, consumers_of)
    return upstream, downstream


def _producers_of_map(context: KernelContext, ops: list[NKIOp]) -> dict[str, NKIOp]:
    """Map tensor_name → producer op (single producer per tensor invariant)."""
    result: dict[str, NKIOp] = {}
    for op in ops:
        for name in context.op_outputs.get(op, []):
            result[name] = op
    return result


def _consumers_of_map(context: KernelContext, ops: list[NKIOp]) -> dict[str, list[NKIOp]]:
    """Map tensor_name → list of consumer ops (kwargs-as-tensor included)."""
    result: dict[str, list[NKIOp]] = {}
    tensors_set = set(context.logical_tensors)
    for op in ops:
        names: set[str] = set()
        for tname in context.op_inputs.get(op, {}).values():
            names.add(tname)
        for _k, expr in context.op_kwargs.get(op, {}).items():
            if expr in tensors_set:
                names.add(expr)
        for name in names:
            result.setdefault(name, []).append(op)
    return result


def _trace_upstream(
    context: KernelContext,
    ops: list[NKIOp],
    affine_op: NKIOp,
    affine_inputs: list[str],
    producers_of: dict[str, NKIOp],
    consumers_of: dict[str, list[NKIOp]],
) -> list[NKIOp]:
    """BFS back from ``affine_op.inputs`` absorbing producers whose outputs feed only the mask chain."""
    absorbed: list[NKIOp] = []
    stack = list(affine_inputs)
    visited: set[int] = {id(affine_op)}
    while stack:
        tname = stack.pop()
        producer = producers_of.get(tname)
        if producer is None or id(producer) in visited:
            continue
        if type(producer).NAME in {"dma_load", "dma_store", "dma_transpose"}:
            continue
        consumer_ids = {id(c) for c in consumers_of.get(tname, [])}
        outside = consumer_ids - {id(o) for o in absorbed} - visited
        if outside:
            continue
        visited.add(id(producer))
        absorbed.append(producer)
        stack.extend(context.op_inputs.get(producer, {}).values())
    _ = ops
    return absorbed


def _trace_downstream(
    context: KernelContext,
    ops: list[NKIOp],
    affine_op: NKIOp,
    affine_outputs: list[str],
    consumers_of: dict[str, list[NKIOp]],
) -> list[NKIOp]:
    """BFS forward from ``affine_op.outputs`` absorbing mask-propagating consumers.

    Absorbs only ops that still carry the mask's free-axis dim in
    every tensor they touch — i.e. ops that iterate over the
    reduction axis (``d2`` for attention). Ops whose output drops
    the free dim (reducer-style) are the boundary and are NOT
    absorbed.

    Also stops at multi-chunk reducer → consumer boundaries:
    absorbing a multi-chunk reducer into the same group as its
    consumer interleaves the reducer's partial running state with
    the consumer's read (e.g. ``tensor_reduce(max)`` feeding
    ``activation_reduce(exp, bias=-max)`` inside the same d2
    loop — the exp chunk sees a partial max, breaking softmax
    stabilization). Online-fusion composites resolve this via σ
    so they're allowed to absorb both sides, but the raw
    reducer/consumer chain is not.
    """
    axis_map = context.op_axis_map.get(affine_op, {})
    free_dim = axis_map.get("F")
    absorbed: list[NKIOp] = []
    stack = list(affine_outputs)
    visited: set[int] = {id(affine_op)}
    propagating = {"tensor_scalar", "tensor_reduce", "activation", "activation_reduce", "nc_matmul", "nc_transpose"}
    while stack:
        tname = stack.pop()
        for consumer in consumers_of.get(tname, []):
            if id(consumer) in visited:
                continue
            cls_name = type(consumer).NAME
            is_propagating = cls_name in propagating
            is_composite = cls_name in {"online_fusion_chain", "compute_skip_chain"}
            if not is_propagating and not is_composite:
                continue
            if not _op_iterates_over_dim(context, consumer, free_dim):
                continue
            if _would_cross_reducer_boundary(context, absorbed, consumer):
                continue
            visited.add(id(consumer))
            absorbed.append(consumer)
            stack.extend(context.op_outputs.get(consumer, []))
    _ = ops
    return absorbed


def _would_cross_reducer_boundary(context: KernelContext, already_absorbed: list[NKIOp], consumer: NKIOp) -> bool:
    """True iff absorbing ``consumer`` would place it in a group with a multi-chunk reducer producer.

    A multi-chunk reducer is a non-composite op whose output
    drops its reduction dim while the reduction dim's block-loop
    trip is > 1. The online-fusion composite implements its own
    σ-corrected accumulation so it does NOT count as a
    multi-chunk reducer here.
    """
    consumer_inputs = set(context.op_inputs.get(consumer, {}).values())
    crosses = False
    for prior in already_absorbed:
        prior_cls = type(prior).NAME
        if prior_cls in {"online_fusion_chain", "compute_skip_chain"}:
            continue
        if any(tname in consumer_inputs for tname in _multichunk_reduced_outputs(context, prior)):
            crosses = True
            break
    return crosses


def _multichunk_reduced_outputs(context: KernelContext, op: NKIOp) -> list[str]:
    """Tensor outputs of ``op`` produced via a multi-chunk reduction."""
    op_cls = type(op)
    kwargs = context.op_kwargs.get(op, {})
    axis_map = context.op_axis_map.get(op, {})
    blocking = context.op_blocking_dims.get(op, set())
    outputs = context.op_outputs.get(op, [])
    result: list[str] = []
    for role_idx, role in enumerate(op_cls.OUTPUT_AXES):
        if role_idx >= len(outputs):
            continue
        if op_cls.resolve_reduce_combinator(role, kwargs) is None:
            continue
        out_axes = op_cls.OUTPUT_AXES[role]
        out_dims = {axis_map.get(ax) for ax in out_axes if axis_map.get(ax) is not None}
        reduction_dims = blocking - out_dims
        if _any_multichunk(context, reduction_dims):
            result.append(outputs[role_idx])
    return result


def _any_multichunk(context: KernelContext, dim_ids: set[str]) -> bool:
    """True iff any dim's block-loop trip is > 1."""
    found = False
    for d in dim_ids:
        di = context.dimensions.get(d)
        if di is None:
            continue
        num_blocks = di.dim_size // (context.ltiles_per_block.get(d, 1) * di.logical_tile_size)
        if num_blocks > 1:
            found = True
            break
    return found


def _op_iterates_over_dim(context: KernelContext, op: NKIOp, dim_id: str | None) -> bool:
    """True iff ``op``'s blocking dims include ``dim_id`` (it iterates over that axis)."""
    result = False
    if dim_id is not None:
        blocking = context.op_blocking_dims.get(op, set())
        if dim_id in blocking:
            result = True
        else:
            for tname in list(context.op_inputs.get(op, {}).values()) + list(context.op_outputs.get(op, [])):
                tinfo = context.logical_tensors.get(tname)
                if tinfo is not None and dim_id in tinfo.dim_ids:
                    result = True
                    break
    return result


def _build_spec(context: KernelContext, graph: KernelGraph, instance: _Match, absorbed: set[NKIOp]) -> ComputeSkipSpec:
    """Build the ``ComputeSkipSpec`` from the matched affine-select op."""
    affine = instance.affine_select_op
    kwargs = context.op_kwargs.get(affine, {})
    axis_map = context.op_axis_map.get(affine, {})
    pattern = ast.literal_eval(kwargs["pattern"])
    step, _count = pattern[0]
    channel_mul = int(kwargs.get("channel_multiplier", "0").strip("'\""))
    offset = int(kwargs.get("offset", "0").strip("'\""))
    cmp_raw = kwargs.get("cmp_op", "'greater_equal'")
    cmp_op = cmp_raw[1:-1] if cmp_raw.startswith("'") and cmp_raw.endswith("'") else cmp_raw
    partition_dim = axis_map["P"]
    free_dim = axis_map["F"]
    tile_sizes = context.op_tile_sizes.get(affine, {})
    p_size = tile_sizes.get(partition_dim, context.dimensions[partition_dim].logical_tile_size)
    f_size = tile_sizes.get(free_dim, context.dimensions[free_dim].logical_tile_size)
    on_false_value = kwargs.get("on_false_value", "float('-inf')")
    boundary = _find_boundary_tensors(context, graph, absorbed)
    return ComputeSkipSpec(
        affine_select_op=affine,
        upstream_ops=instance.upstream_ops,
        downstream_ops=instance.downstream_ops,
        partition_dim_id=partition_dim,
        free_dim_id=free_dim,
        channel_multiplier=channel_mul,
        free_step=int(step),
        offset=offset,
        cmp_op=cmp_op,
        partition_tile_size=int(p_size),
        free_tile_size=int(f_size),
        boundary_tensors=boundary,
        on_false_value=on_false_value,
    )


def _find_boundary_tensors(context: KernelContext, graph: KernelGraph, absorbed: set[NKIOp]) -> tuple[str, ...]:
    """Return tensors produced inside ``absorbed`` but consumed by non-absorbed ops.

    On ``skip_all`` tiles these tensors must be memset to the
    affine-select's ``on_false_value`` so external consumers
    (which read the full tensor across every tile) see the
    mask-out sentinel on skipped tiles.
    """
    produced_inside: set[str] = set()
    for op in absorbed:
        produced_inside.update(context.op_outputs.get(op, []))
    external_consumers: set[str] = set()
    for group in graph.groups:
        for op in group.ops:
            if op in absorbed:
                continue
            for tname in context.op_inputs.get(op, {}).values():
                if tname in produced_inside:
                    external_consumers.add(tname)
    return tuple(sorted(external_consumers))


def _rebuild_groups(
    graph: KernelGraph, absorbed: set[NKIOp], instance: _Match, spec: ComputeSkipSpec
) -> list[FusionGroup]:
    """Produce a new group list with a single skip-annotated group replacing the absorbed ones."""
    upstream_topo = list(reversed(instance.upstream_ops))
    new_group_ops = [*upstream_topo, instance.affine_select_op, *instance.downstream_ops]
    inserted = False
    new_groups: list[FusionGroup] = []
    for group in graph.groups:
        absorbed_here = [op for op in group.ops if op in absorbed]
        surviving = [op for op in group.ops if op not in absorbed]
        if absorbed_here and not inserted:
            new_groups.append(FusionGroup(ops=new_group_ops, skip_spec=spec))
            inserted = True
        if surviving:
            new_groups.append(
                replace(
                    group,
                    ops=surviving,
                    dim_order=list(group.dim_order),
                    buffer_degrees=dict(group.buffer_degrees),
                    tensor_placements=dict(group.tensor_placements),
                )
            )
    return new_groups
