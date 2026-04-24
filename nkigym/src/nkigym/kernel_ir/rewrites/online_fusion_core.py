"""Shared helpers for online-fusion rewrites (create + extend).

Split out of ``online_fusion_rewrite`` so the ``online_fusion_extend``
module can reuse axis-map building, reassembly, and the externally-used
predicate without creating an import cycle.
"""

from dataclasses import replace

from nkigym.kernel_ir.fusion_group import FusionGroup
from nkigym.kernel_ir.ir import KernelIR, rebuild_edges
from nkigym.kernel_ir.rewrites.online_fusion_detect import OnlineFusionCandidate
from nkigym.kernel_ir.rewrites.online_fusion_spec import AccumulatorSpec
from nkigym.kernel_ir.types import DimRole, TensorInfo
from nkigym.ops.base import NKIOp


def all_ops(ir: KernelIR) -> list[NKIOp]:
    """Flat walk over all ops across groups."""
    return [op for group in ir.groups for op in group.ops]


def is_externally_used(ir: KernelIR, tname: str, absorbed: set[NKIOp]) -> bool:
    """True iff ``tname`` is read by a non-absorbed op or is the kernel return."""
    external_consumer = False
    for op in all_ops(ir):
        if op in absorbed:
            continue
        if tname in ir.op_inputs.get(op, {}).values():
            external_consumer = True
            break
    return tname == ir.return_name or external_consumer


def build_axis_maps(
    ir: KernelIR, role_pairs: list[tuple[str, str]]
) -> tuple[dict[str, tuple[str, ...]], dict[str, str], dict[str, int]]:
    """Per-input-role axis labels, locs, tile limits."""
    axes: dict[str, tuple[str, ...]] = {}
    locs: dict[str, str] = {}
    tile_limits: dict[str, int] = {}
    for role, tname in role_pairs:
        dims = ir.tensor_info(tname).dim_ids
        labels = tuple(f"{role}_{i}" for i in range(len(dims)))
        axes[role] = labels
        locs[role] = "sbuf"
        for label, dim_id in zip(labels, dims):
            tile_limits[label] = ir.dimensions[dim_id].physical_tile_size
    return axes, locs, tile_limits


def build_output_axis_maps(
    ir: KernelIR, role_pairs: list[tuple[str, str]]
) -> tuple[dict[str, tuple[str, ...]], dict[str, int]]:
    """Per-output-role axis labels and tile limits."""
    axes: dict[str, tuple[str, ...]] = {}
    tile_limits: dict[str, int] = {}
    for role, tname in role_pairs:
        dims = ir.tensor_info(tname).dim_ids
        labels = tuple(f"{role}_{i}" for i in range(len(dims)))
        axes[role] = labels
        for label, dim_id in zip(labels, dims):
            tile_limits[label] = ir.dimensions[dim_id].physical_tile_size
    return axes, tile_limits


def build_accumulator_specs(
    candidate: OnlineFusionCandidate, ir: KernelIR, outputs: list[tuple[str, str]]
) -> tuple[AccumulatorSpec, ...]:
    """One ``AccumulatorSpec`` per accumulator op in the candidate."""
    specs: list[AccumulatorSpec] = []
    for acc_op in candidate.accumulator_ops:
        role = _accumulator_output_role(ir, acc_op, outputs)
        kind = {"nc_matmul": "matmul", "activation_reduce": "activation_reduce"}.get(type(acc_op).NAME)
        if kind is None:
            raise NotImplementedError(f"accumulator op {type(acc_op).NAME!r} not yet supported")
        specs.append(
            AccumulatorSpec(
                kind=kind,
                output_role=role,
                source_op=acc_op,
                ptile_free_dim=candidate.blocking_dim if kind == "matmul" else "",
                source_kwargs=tuple(ir.op_kwargs.get(acc_op, {}).items()),
            )
        )
    return tuple(specs)


def _accumulator_output_role(ir: KernelIR, acc_op: NKIOp, outputs: list[tuple[str, str]]) -> str:
    """Return the composite output role for an accumulator op's externally-consumed output."""
    acc_outputs = set(ir.op_outputs.get(acc_op, []))
    for role, tname in outputs:
        if tname in acc_outputs:
            return role
    raise ValueError(f"accumulator {acc_op!r} has no external output in {outputs}")


def composite_axis_map(
    ir: KernelIR, composite_cls: type[NKIOp], inputs: list[tuple[str, str]], outputs: list[tuple[str, str]]
) -> tuple[dict[str, str], dict[str, int]]:
    """Build the composite's axis_map and tile_sizes dicts."""
    axis_map: dict[str, str] = {}
    tile_sizes: dict[str, int] = {}
    for role, tname in inputs:
        for label, dim_id in zip(composite_cls.OPERAND_AXES[role], ir.tensor_info(tname).dim_ids):
            axis_map[label] = dim_id
            tile_sizes[dim_id] = ir.dimensions[dim_id].physical_tile_size
    for role, tname in outputs:
        for label, dim_id in zip(composite_cls.OUTPUT_AXES[role], ir.tensor_info(tname).dim_ids):
            axis_map[label] = dim_id
            tile_sizes[dim_id] = ir.dimensions[dim_id].physical_tile_size
    return axis_map, tile_sizes


def reassemble(
    ir: KernelIR,
    absorbed: set[NKIOp],
    composite_op: NKIOp,
    inputs: list[tuple[str, str]],
    outputs: list[tuple[str, str]],
    candidate: OnlineFusionCandidate,
) -> tuple[KernelIR, KernelIR]:
    """Build the post-rewrite ``(ir, ir)`` pair."""
    composite_cls = type(composite_op)
    composite_inputs = dict(inputs)
    composite_outputs = [name for _, name in outputs]
    composite_axis_m, composite_tile_sizes = composite_axis_map(ir, composite_cls, inputs, outputs)
    new_context = _build_new_context(
        ir,
        absorbed,
        composite_op,
        composite_inputs,
        composite_outputs,
        composite_axis_m,
        composite_tile_sizes,
        outputs,
        candidate,
    )
    new_graph = _build_new_graph(ir, absorbed, composite_op, new_context)
    return new_context, new_graph


def _build_new_context(
    ir: KernelIR,
    absorbed: set[NKIOp],
    composite_op: NKIOp,
    composite_inputs: dict[str, str],
    composite_outputs: list[str],
    composite_axis_m: dict[str, str],
    composite_tile_sizes: dict[str, int],
    outputs: list[tuple[str, str]],
    candidate: OnlineFusionCandidate,
) -> KernelIR:
    """Build post-rewrite ir — dicts filtered, composite entry added, dim promoted to ACCUMULATION."""
    new_op_inputs = {op: v for op, v in ir.op_inputs.items() if op not in absorbed}
    new_op_outputs = {op: v for op, v in ir.op_outputs.items() if op not in absorbed}
    new_op_kwargs = {op: v for op, v in ir.op_kwargs.items() if op not in absorbed}
    new_op_axis_map = {op: v for op, v in ir.op_axis_map.items() if op not in absorbed}
    new_op_tile_sizes = {op: v for op, v in ir.op_tile_sizes.items() if op not in absorbed}
    new_op_blocking_dims = {op: v for op, v in ir.op_blocking_dims.items() if op not in absorbed}
    new_op_skip_spec = {op: v for op, v in ir.op_skip_spec.items() if op not in absorbed}
    new_op_inputs[composite_op] = composite_inputs
    new_op_outputs[composite_op] = composite_outputs
    new_op_kwargs[composite_op] = {}
    new_op_axis_map[composite_op] = composite_axis_m
    new_op_tile_sizes[composite_op] = composite_tile_sizes
    new_op_blocking_dims[composite_op] = {candidate.blocking_dim}
    inherited_predicate = next((ir.op_skip_spec[op] for op in absorbed if op in ir.op_skip_spec), None)
    if inherited_predicate is not None:
        new_op_skip_spec[composite_op] = inherited_predicate
    new_dims = dict(ir.dimensions)
    new_dims[candidate.blocking_dim] = replace(new_dims[candidate.blocking_dim], role=DimRole.ACCUMULATION)
    new_logical_tensors = _retain_external_tensors(ir, absorbed, outputs)
    return replace(
        ir,
        dimensions=new_dims,
        logical_tensors=new_logical_tensors,
        op_inputs=new_op_inputs,
        op_outputs=new_op_outputs,
        op_kwargs=new_op_kwargs,
        op_axis_map=new_op_axis_map,
        op_tile_sizes=new_op_tile_sizes,
        op_blocking_dims=new_op_blocking_dims,
        op_skip_spec=new_op_skip_spec,
    )


def _build_new_graph(ir: KernelIR, absorbed: set[NKIOp], composite_op: NKIOp, new_ir: KernelIR) -> KernelIR:
    """Build post-rewrite ir: drop absorbed ops, insert composite, split mixed groups into pre/post.

    A group whose surviving ops are all upstream of the composite stays unchanged (runs before).
    A group whose surviving ops are all downstream stays unchanged (runs after). A group containing
    both (TF merged across the composite's boundary) is split in two so the post-rewrite DAG stays
    acyclic.
    """
    survivors: list[FusionGroup] = []
    composite_inputs = set(new_context.op_inputs.get(composite_op, {}).values())
    composite_outputs = set(new_context.op_outputs.get(composite_op, []))
    inserted = False
    for group in ir.groups:
        surviving_ops = [op for op in group.ops if op not in absorbed]
        if not surviving_ops and not inserted and any(op in absorbed for op in group.ops):
            survivors.append(FusionGroup(ops=[composite_op]))
            inserted = True
            continue
        if not surviving_ops:
            continue
        pre, post = _split_pre_post(surviving_ops, composite_inputs, composite_outputs, new_context, absorbed)
        if pre and post and not inserted:
            survivors.append(FusionGroup(ops=pre))
            survivors.append(FusionGroup(ops=[composite_op]))
            survivors.append(FusionGroup(ops=post))
            inserted = True
        elif pre and post:
            survivors.append(FusionGroup(ops=pre))
            survivors.append(FusionGroup(ops=post))
        else:
            survivors.append(
                FusionGroup(
                    ops=surviving_ops,
                    dim_order=list(group.dim_order),
                    buffer_degrees=dict(group.buffer_degrees),
                    buffer_placements=dict(group.buffer_placements),
                )
            )
    if not inserted:
        survivors.append(FusionGroup(ops=[composite_op]))
    new_graph = KernelIR(groups=survivors)
    rebuild_edges(new_graph, new_context)
    return new_graph


def _split_pre_post(
    surviving_ops: list[NKIOp],
    composite_inputs: set[str],
    composite_outputs: set[str],
    new_ir: KernelIR,
    absorbed: set[NKIOp],
) -> tuple[list[NKIOp], list[NKIOp]]:
    """Partition surviving ops into pre/post halves relative to the composite.

    An op is "post" if any of its input tensors is a composite output OR if any of its input
    tensors is produced by an already-classified post op. Everything else is "pre".
    """
    post_set: set[NKIOp] = set()
    post_produced: set[str] = set()
    changed = True
    while changed:
        changed = False
        for op in surviving_ops:
            if op in post_set:
                continue
            inputs = set(new_context.op_inputs.get(op, {}).values())
            if inputs & (composite_outputs | post_produced):
                post_set.add(op)
                post_produced.update(new_context.op_outputs.get(op, []))
                changed = True
    pre = [op for op in surviving_ops if op not in post_set]
    post = [op for op in surviving_ops if op in post_set]
    _ = composite_inputs
    _ = absorbed
    return pre, post


def _retain_external_tensors(
    ir: KernelIR, absorbed: set[NKIOp], outputs: list[tuple[str, str]]
) -> dict[str, TensorInfo]:
    """Drop tensors that only absorbed-and-internal ops referenced."""
    _ = TensorInfo
    external_names = {name for _, name in outputs}
    referenced: set[str] = set(external_names)
    for op in all_ops(ir):
        if op in absorbed:
            continue
        for name in ir.op_inputs.get(op, {}).values():
            referenced.add(name)
        for name in ir.op_outputs.get(op, []):
            referenced.add(name)
    produced_in_original: set[str] = set()
    for op in all_ops(ir):
        produced_in_original.update(ir.op_outputs.get(op, []))
    referenced.update(name for name in ir.logical_tensors if name not in produced_in_original)
    return {name: tinfo for name, tinfo in ir.logical_tensors.items() if name in referenced}
