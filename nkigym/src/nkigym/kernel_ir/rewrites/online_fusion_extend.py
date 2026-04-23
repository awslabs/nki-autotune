"""Online-fusion extension rewrite.

Takes an existing ``NKIOnlineFusionChain`` composite plus new
accumulator(s) that trivial fusion has pulled into the same
group, and produces a new composite with appended
``ACCUMULATOR_SPECS``. The composite's ``SCALE_SPEC`` is inherited
unchanged — all accumulators in one composite share σ by
construction.
"""

from dataclasses import replace as _replace

from nkigym.kernel_ir.context.context import KernelContext
from nkigym.kernel_ir.graph.graph import KernelGraph
from nkigym.kernel_ir.rewrites.online_fusion_core import (
    build_accumulator_specs,
    build_axis_maps,
    build_output_axis_maps,
    is_externally_used,
    reassemble,
)
from nkigym.kernel_ir.rewrites.online_fusion_detect import OnlineFusionCandidate
from nkigym.kernel_ir.rewrites.online_fusion_spec import AccumulatorSpec
from nkigym.ops.base import NKIOp
from nkigym.ops.online_fusion_chain import NKIOnlineFusionChain, make_online_fusion_class


def rewrite_extend(
    context: KernelContext, graph: KernelGraph, candidate: OnlineFusionCandidate
) -> tuple[KernelContext, KernelGraph]:
    """Extend an existing composite by appending new same-group accumulators."""
    old_composite = candidate.x_op
    assert isinstance(old_composite, NKIOnlineFusionChain)
    old_cls = type(old_composite)
    absorbed = _extend_absorbed_ops(context, graph, candidate, old_composite)
    new_inputs = _extend_inputs(context, old_cls, candidate, absorbed)
    kept_old_names = _kept_old_output_names(context, graph, old_cls, absorbed)
    new_acc_names = _new_accumulator_output_names(context, candidate)
    combined_names = kept_old_names + new_acc_names
    new_outputs = [(f"out_{i}", name) for i, name in enumerate(combined_names)]
    inherited_specs = _inherit_old_specs(old_cls, kept_old_names, new_outputs)
    new_composite_cls = _build_extended_composite_class(
        context, old_cls, candidate, new_inputs, new_outputs, inherited_specs
    )
    new_composite_op = new_composite_cls()
    return reassemble(context, graph, absorbed, new_composite_op, new_inputs, new_outputs, candidate)


def _kept_old_output_names(
    context: KernelContext, graph: KernelGraph, old_cls: type[NKIOnlineFusionChain], absorbed: set[NKIOp]
) -> list[str]:
    """Old output tensor names still consumed outside the extended composite."""
    kept: list[str] = []
    for _role, name in zip(old_cls.OUTPUT_AXES.keys(), old_cls.OUTPUT_TENSOR_NAMES):
        if is_externally_used(context, graph, name, absorbed):
            kept.append(name)
    return kept


def _new_accumulator_output_names(context: KernelContext, candidate: OnlineFusionCandidate) -> list[str]:
    """Tensor names produced by the new accumulators in spec order."""
    names: list[str] = []
    for acc_op in candidate.accumulator_ops:
        outputs = context.op_outputs.get(acc_op, [])
        if outputs:
            names.append(outputs[0])
    return names


def _inherit_old_specs(
    old_cls: type[NKIOnlineFusionChain], kept_old_names: list[str], new_outputs: list[tuple[str, str]]
) -> tuple[AccumulatorSpec, ...]:
    """Retain all old ACCUMULATOR_SPECS; rewrite ``output_role`` when the old tensor is still external.

    Specs whose old output tensor is now internal keep their kind/source metadata so the per-iteration
    body still emits; their ``output_role`` stays as the old role string but no external copy is emitted.
    """
    name_to_role = {name: role for role, name in new_outputs}
    remapped: list[AccumulatorSpec] = []
    for spec in old_cls.ACCUMULATOR_SPECS:
        assert isinstance(spec, AccumulatorSpec), "old composite carries non-AccumulatorSpec entries"
        old_name = _old_spec_output_name(spec, old_cls)
        new_role = name_to_role.get(old_name, spec.output_role)
        remapped.append(_replace(spec, output_role=new_role))
    _ = kept_old_names
    return tuple(remapped)


def _old_spec_output_name(spec: AccumulatorSpec, old_cls: type[NKIOnlineFusionChain]) -> str:
    """Resolve the tensor name an old spec writes to via its old ``output_role`` index."""
    roles = list(old_cls.OUTPUT_AXES.keys())
    idx = roles.index(spec.output_role) if spec.output_role in roles else -1
    return old_cls.OUTPUT_TENSOR_NAMES[idx] if 0 <= idx < len(old_cls.OUTPUT_TENSOR_NAMES) else ""


def _extend_absorbed_ops(
    context: KernelContext, graph: KernelGraph, candidate: OnlineFusionCandidate, old_composite: NKIOp
) -> set[NKIOp]:
    """Absorbed = old composite + new accumulators + intermediate ops linking composite outputs to accumulators.

    Only ops whose data lineage traces FROM a composite output (e.g. ``nc_transpose(exp_S)``) are
    absorbed. The moving-side producers (e.g. ``dma_load(V)``) stay external — they supply a
    separate input to the composite, not an internal chain.
    """
    group_of = {id(op): gi for gi, group in enumerate(graph.groups) for op in group.ops}
    comp_gi = group_of[id(old_composite)]
    absorbed: set[NKIOp] = {old_composite, *candidate.accumulator_ops}
    x_outputs = set(context.op_outputs.get(old_composite, []))
    for acc_op in candidate.accumulator_ops:
        absorbed.update(_walk_composite_output_chain(context, graph, acc_op, x_outputs, comp_gi, group_of))
    return absorbed


def _walk_composite_output_chain(
    context: KernelContext,
    graph: KernelGraph,
    acc_op: NKIOp,
    x_outputs: set[str],
    comp_gi: int,
    group_of: dict[int, int],
) -> set[NKIOp]:
    """Absorb only ops on the data-flow path starting from one of the composite's outputs."""
    absorbed: set[NKIOp] = set()
    stack: list[str] = list(context.op_inputs.get(acc_op, {}).values())
    visited: set[str] = set()
    while stack:
        name = stack.pop()
        if name in visited or name in x_outputs:
            continue
        visited.add(name)
        producer = _find_producer_in_graph(context, graph, name)
        if producer is None or group_of.get(id(producer)) != comp_gi or producer in absorbed:
            continue
        if not _reaches_composite_output(context, graph, producer, x_outputs, comp_gi, group_of):
            continue
        absorbed.add(producer)
        stack.extend(context.op_inputs.get(producer, {}).values())
    return absorbed


def _reaches_composite_output(
    context: KernelContext, graph: KernelGraph, op: NKIOp, x_outputs: set[str], comp_gi: int, group_of: dict[int, int]
) -> bool:
    """True iff ``op``'s inputs trace back (same-group only) to any composite output."""
    stack: list[str] = list(context.op_inputs.get(op, {}).values())
    visited: set[str] = set()
    reached = False
    while stack and not reached:
        name = stack.pop()
        if name in visited:
            continue
        visited.add(name)
        if name in x_outputs:
            reached = True
            continue
        producer = _find_producer_in_graph(context, graph, name)
        if producer is None or group_of.get(id(producer)) != comp_gi:
            continue
        stack.extend(context.op_inputs.get(producer, {}).values())
    return reached


def _find_producer_in_graph(context: KernelContext, graph: KernelGraph, tensor_name: str) -> NKIOp | None:
    """First op in ``graph`` that outputs ``tensor_name`` (or None)."""
    result: NKIOp | None = None
    for group in graph.groups:
        for op in group.ops:
            if tensor_name in context.op_outputs.get(op, []):
                result = op
                break
        if result is not None:
            break
    return result


def _extend_inputs(
    context: KernelContext, old_cls: type[NKIOnlineFusionChain], candidate: OnlineFusionCandidate, absorbed: set[NKIOp]
) -> list[tuple[str, str]]:
    """Combine old external inputs with any new external inputs (including kernel inputs via absorbed DMA loads)."""
    old_inputs = list(zip(old_cls.OPERAND_AXES.keys(), old_cls.INPUT_TENSOR_NAMES))
    old_input_names = {name for _, name in old_inputs}
    produced_internal: set[str] = set()
    for op in absorbed:
        produced_internal.update(context.op_outputs.get(op, []))
    new_external: list[str] = []
    seen: set[str] = set()
    for op in absorbed:
        if op is candidate.x_op:
            continue
        for tname in context.op_inputs.get(op, {}).values():
            if (
                tname not in produced_internal
                and tname not in old_input_names
                and tname in context.logical_tensors
                and tname not in seen
            ):
                new_external.append(tname)
                seen.add(tname)
    combined = list(old_inputs)
    for name in new_external:
        combined.append((f"b_in_{len(combined)}", name))
    return combined


def _extend_outputs(
    context: KernelContext,
    graph: KernelGraph,
    old_cls: type[NKIOnlineFusionChain],
    candidate: OnlineFusionCandidate,
    absorbed: set[NKIOp],
) -> list[tuple[str, str]]:
    """Carry forward old outputs still externally used + add new accumulator outputs."""
    kept_old = _kept_old_outputs(context, graph, old_cls, absorbed)
    new_accumulator_outputs = _extend_new_outputs(context, candidate, kept_old)
    return kept_old + new_accumulator_outputs


def _kept_old_outputs(
    context: KernelContext, graph: KernelGraph, old_cls: type[NKIOnlineFusionChain], absorbed: set[NKIOp]
) -> list[tuple[str, str]]:
    """Old outputs that are still consumed outside the new absorbed set or are the kernel return."""
    kept: list[tuple[str, str]] = []
    for role, name in zip(old_cls.OUTPUT_AXES.keys(), old_cls.OUTPUT_TENSOR_NAMES):
        if is_externally_used(context, graph, name, absorbed):
            kept.append((role, name))
    return kept


def _extend_new_outputs(
    context: KernelContext, candidate: OnlineFusionCandidate, kept_old: list[tuple[str, str]]
) -> list[tuple[str, str]]:
    """Output-role bindings for each new accumulator, numbered after ``kept_old``."""
    result: list[tuple[str, str]] = []
    next_idx = len(kept_old)
    for acc_op in candidate.accumulator_ops:
        outputs = context.op_outputs.get(acc_op, [])
        if outputs:
            result.append((f"out_{next_idx}", outputs[0]))
            next_idx += 1
    return result


def _build_extended_composite_class(
    context: KernelContext,
    old_cls: type[NKIOnlineFusionChain],
    candidate: OnlineFusionCandidate,
    inputs: list[tuple[str, str]],
    outputs: list[tuple[str, str]],
    inherited_specs: tuple[AccumulatorSpec, ...],
) -> type[NKIOnlineFusionChain]:
    """Build the new composite class by combining old ScaleSpec with appended ACCUMULATOR_SPECS."""
    input_axes, input_locs, tile_limits = build_axis_maps(context, inputs)
    output_axes, output_tile_limits = build_output_axis_maps(context, outputs)
    tile_limits.update(output_tile_limits)
    new_specs = build_accumulator_specs(candidate, context, outputs)
    combined_specs = inherited_specs + new_specs
    return make_online_fusion_class(
        label=f"{candidate.scale_role}_ext",
        accumulation_dim=old_cls.ACCUMULATION_DIM,
        input_tensor_names=tuple(name for _, name in inputs),
        input_axes=input_axes,
        input_locs=input_locs,
        output_tensor_names=tuple(name for _, name in outputs),
        output_axes=output_axes,
        tile_limits=tile_limits,
        blocking_axes=frozenset(),
        scale_spec=old_cls.SCALE_SPEC,
        accumulator_specs=combined_specs,
    )
