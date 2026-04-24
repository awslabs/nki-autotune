"""``propagate_compute_skip`` — mandatory pre-pass that lifts ``NKIAffineSelect`` to per-op annotations.

Runs exactly once after ``insert_dma_nodes`` and before any
``REWRITES`` sampling. For every ``NKIAffineSelect`` in the IR:

1. Extract the causal predicate coefficients + the ``on_false_value``.
2. BFS upstream — every producer whose output flows exclusively
   into the affine_select's data chain gets the predicate.
3. BFS downstream — carry the mask sentinel through the chain.
4. Annotate ops in ``op_skip_spec`` with the predicate.
5. Delete the ``NKIAffineSelect`` from the ir; rewire
   consumers of the masked tensor to read the upstream source.
"""

import ast
from dataclasses import replace

from nkigym.kernel_ir.compute_skip_spec import SkipPredicate
from nkigym.kernel_ir.fusion_group import FusionGroup
from nkigym.kernel_ir.ir import KernelIR, rebuild_edges
from nkigym.ops.base import NKIOp

_REDUCER_IDENTITY: dict[str, float] = {"add": 0.0, "maximum": float("-inf"), "minimum": float("inf"), "multiply": 1.0}


def propagate_compute_skip(ir: KernelIR) -> KernelIR:
    """Lift every ``NKIAffineSelect`` to per-op skip predicates; remove the standalone op."""
    current = ir
    while True:
        affine_op = _find_first_affine_select(current)
        if affine_op is None:
            break
        current = _propagate_one(current, affine_op)
    return current


def _find_first_affine_select(ir: KernelIR) -> NKIOp | None:
    """Return the first ``NKIAffineSelect`` still in the ir, or ``None``."""
    result: NKIOp | None = None
    for group in ir.groups:
        for op in group.ops:
            if type(op).NAME == "affine_select":
                result = op
                break
        if result is not None:
            break
    return result


def _propagate_one(ir: KernelIR, affine_op: NKIOp) -> KernelIR:
    """Propagate one affine-select: annotate upstream + downstream; remove the op; rewire."""
    predicate_base = _build_predicate_base(ir, affine_op)
    upstream = _trace_upstream(ir, affine_op)
    downstream = _trace_downstream_with_numerics(ir, affine_op, predicate_base)
    ir = _annotate_ops(ir, predicate_base, upstream, downstream, affine_op)
    ir = _remove_affine_select(ir, affine_op)
    return ir


def _build_predicate_base(ir: KernelIR, affine_op: NKIOp) -> SkipPredicate:
    """Extract the predicate coefficients from an ``NKIAffineSelect`` op's kwargs."""
    kwargs = ir.op_kwargs.get(affine_op, {})
    axis_map = ir.op_axis_map.get(affine_op, {})
    pattern = ast.literal_eval(kwargs["pattern"])
    step, _count = pattern[0]
    channel_mul = int(kwargs.get("channel_multiplier", "0").strip("'\""))
    offset = int(kwargs.get("offset", "0").strip("'\""))
    cmp_raw = kwargs.get("cmp_op", "'greater_equal'")
    cmp_op = cmp_raw[1:-1] if cmp_raw.startswith("'") and cmp_raw.endswith("'") else cmp_raw
    partition_dim = axis_map["P"]
    free_dim = axis_map["F"]
    tile_sizes = ir.op_tile_sizes.get(affine_op, {})
    p_size = tile_sizes.get(partition_dim, ir.dimensions[partition_dim].logical_tile_size)
    f_size = tile_sizes.get(free_dim, ir.dimensions[free_dim].logical_tile_size)
    on_false_value = kwargs.get("on_false_value", "float('-inf')")
    return SkipPredicate(
        partition_dim_id=partition_dim,
        free_dim_id=free_dim,
        channel_multiplier=channel_mul,
        free_step=int(step),
        offset=offset,
        cmp_op=cmp_op,
        partition_tile_size=int(p_size),
        free_tile_size=int(f_size),
        on_false_value=on_false_value,
    )


def _trace_upstream(ir: KernelIR, affine_op: NKIOp) -> list[NKIOp]:
    """BFS back from ``affine_op.inputs`` absorbing producers whose outputs feed only the mask chain."""
    all_ops = [op for group in ir.groups for op in group.ops]
    producers_of = _producers_of_map(ir, all_ops)
    consumers_of = _consumers_of_map(ir, all_ops)
    affine_inputs = list(ir.op_inputs.get(affine_op, {}).values())
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
        stack.extend(ir.op_inputs.get(producer, {}).values())
    return absorbed


def _producers_of_map(ir: KernelIR, ops: list[NKIOp]) -> dict[str, NKIOp]:
    """Map tensor_name → producer op."""
    result: dict[str, NKIOp] = {}
    for op in ops:
        for name in ir.op_outputs.get(op, []):
            result[name] = op
    return result


def _consumers_of_map(ir: KernelIR, ops: list[NKIOp]) -> dict[str, list[NKIOp]]:
    """Map tensor_name → consumer op list (kwargs-as-tensor included)."""
    result: dict[str, list[NKIOp]] = {}
    tensors_set = set(ir.logical_tensors) | set(ir.physical_buffers)
    for op in ops:
        names: set[str] = set()
        for tname in ir.op_inputs.get(op, {}).values():
            names.add(tname)
        for _k, expr in ir.op_kwargs.get(op, {}).items():
            if expr in tensors_set:
                names.add(expr)
        for name in names:
            result.setdefault(name, []).append(op)
    return result


def _trace_downstream_with_numerics(ir: KernelIR, affine_op: NKIOp, base_pred: SkipPredicate) -> list[NKIOp]:
    """BFS forward annotating every consumer reachable from ``affine_op``."""
    all_ops = [op for group in ir.groups for op in group.ops]
    consumers_of = _consumers_of_map(ir, all_ops)
    starting_value = _parse_on_false(base_pred.on_false_value)
    if starting_value is None:
        raise ValueError(f"Compute-skip: cannot parse on_false_value={base_pred.on_false_value!r} as a scalar literal.")
    free_dim = base_pred.free_dim_id
    stack: list[tuple[str, float]] = [(name, starting_value) for name in ir.op_outputs.get(affine_op, [])]
    visited: set[int] = {id(affine_op)}
    annotated: list[NKIOp] = []
    while stack:
        tname, current_value = stack.pop()
        for consumer in consumers_of.get(tname, []):
            if id(consumer) in visited:
                continue
            next_value = _propagate_through_or_raise(ir, consumer, current_value, free_dim)
            visited.add(id(consumer))
            annotated.append(consumer)
            for out in ir.op_outputs.get(consumer, []):
                if _tensor_has_dim(ir, out, free_dim):
                    stack.append((out, next_value))
    return annotated


def _tensor_has_dim(ir: KernelIR, tensor_name: str, dim_id: str) -> bool:
    """True iff the tensor carries ``dim_id`` in its logical ``dim_ids``."""
    if not ir.has_tensor(tensor_name):
        return False
    return dim_id in ir.tensor_info(tensor_name).dim_ids


def _propagate_through_or_raise(ir: KernelIR, op: NKIOp, input_value: float, free_dim: str) -> float:
    """Compute the op's output sentinel, raising if the op is not numerically compatible with skip."""
    op_cls = type(op)
    kwargs = ir.op_kwargs.get(op, {})
    axis_map = ir.op_axis_map.get(op, {})
    blocking = ir.op_blocking_dims.get(op, set())
    elementwise = op_cls.propagate_mask_value(kwargs, input_value)
    reducer_role = _find_reducer_role_on_free_dim(op_cls, axis_map, blocking, free_dim)
    effective = elementwise if elementwise is not None else input_value
    if reducer_role is not None:
        combinator = _element_level_combinator(op_cls, reducer_role, kwargs)
        identity = _REDUCER_IDENTITY.get(combinator) if combinator is not None else None
        if identity is None or not _isclose(effective, identity):
            raise ValueError(
                f"Compute-skip: op {op_cls.NAME!r} reduces along free dim {free_dim!r} with combinator "
                f"{combinator!r} (identity={identity!r}), but the post-elementwise mask sentinel is "
                f"{effective!r} — skipping is not numerically valid."
            )
        result = effective
    else:
        if elementwise is None:
            raise ValueError(
                f"Compute-skip: op {op_cls.NAME!r} isn't analyzable for mask propagation on input {input_value!r}."
            )
        result = elementwise
    return result


def _element_level_combinator(op_cls: type[NKIOp], role: str, kwargs: dict[str, str]) -> str | None:
    """Return the element-level reducer name, bypassing chunk-combine adjustments like ``negate``."""
    spec = op_cls.REDUCE_COMBINATOR.get(role)
    result: str | None = None
    if spec is not None:
        if spec.startswith("__"):
            result = spec[2:]
        else:
            raw = kwargs.get(spec)
            if raw is not None:
                result = raw[1:-1] if raw.startswith("'") and raw.endswith("'") else raw
    return result


def _find_reducer_role_on_free_dim(
    op_cls: type[NKIOp], axis_map: dict[str, str], blocking: set[str], free_dim: str
) -> str | None:
    """Return the output role whose reduction dim is the mask's free dim, or ``None``."""
    result: str | None = None
    class_blocking_concrete = {axis_map.get(ax) for ax in op_cls.BLOCKING_AXES}
    if free_dim in class_blocking_concrete and free_dim in blocking:
        for role, axes in op_cls.OUTPUT_AXES.items():
            out_dims = {axis_map.get(ax) for ax in axes}
            if free_dim not in out_dims:
                result = role
                break
    return result


def _isclose(a: float, b: float) -> bool:
    """Compare two floats treating ``±inf`` as exact self-matches."""
    return (a == b) or (a == float("inf") and b == float("inf")) or (a == float("-inf") and b == float("-inf"))


def _parse_on_false(raw: str) -> float | None:
    """Parse the ``on_false_value`` literal to a float."""
    result: float | None = None
    if raw == "float('-inf')":
        result = float("-inf")
    elif raw == "float('inf')":
        result = float("inf")
    else:
        try:
            result = float(raw)
        except ValueError:
            result = None
    return result


def _annotate_ops(
    ir: KernelIR, base_pred: SkipPredicate, upstream: list[NKIOp], downstream: list[NKIOp], affine_op: NKIOp
) -> KernelIR:
    """Write per-op ``SkipPredicate`` entries into ``ir.op_skip_spec``."""
    new_skip = dict(ir.op_skip_spec)
    source_op = _find_affine_source_op(ir, affine_op)
    for op in upstream:
        inject = op is source_op
        new_skip[op] = replace(base_pred, inject_mask=inject)
    for op in downstream:
        new_skip[op] = replace(base_pred, inject_mask=False)
    return replace(ir, op_skip_spec=new_skip)


def _find_affine_source_op(ir: KernelIR, affine_op: NKIOp) -> NKIOp | None:
    """Return the op producing the tensor that ``affine_op`` reads via ``on_true_tile``."""
    source_tensor = ir.op_inputs.get(affine_op, {}).get("on_true_tile")
    result: NKIOp | None = None
    if source_tensor is not None:
        for op, outs in ir.op_outputs.items():
            if source_tensor in outs:
                result = op
                break
    return result


def _remove_affine_select(ir: KernelIR, affine_op: NKIOp) -> KernelIR:
    """Drop the affine-select op from every IR dict and from its group; rewire consumers."""
    source_tensor = ir.op_inputs.get(affine_op, {}).get("on_true_tile")
    masked_tensors = list(ir.op_outputs.get(affine_op, []))
    new_op_inputs = {op: _rewire(inputs, masked_tensors, source_tensor) for op, inputs in ir.op_inputs.items()}
    new_op_kwargs = {op: _rewire(kw, masked_tensors, source_tensor) for op, kw in ir.op_kwargs.items()}
    new_op_inputs.pop(affine_op, None)
    new_op_kwargs.pop(affine_op, None)
    new_op_outputs = {op: v for op, v in ir.op_outputs.items() if op is not affine_op}
    new_op_axis_map = {op: v for op, v in ir.op_axis_map.items() if op is not affine_op}
    new_op_tile_sizes = {op: v for op, v in ir.op_tile_sizes.items() if op is not affine_op}
    new_op_blocking_dims = {op: v for op, v in ir.op_blocking_dims.items() if op is not affine_op}
    new_op_skip_spec = {op: v for op, v in ir.op_skip_spec.items() if op is not affine_op}
    new_tensors = {name: info for name, info in ir.logical_tensors.items() if name not in masked_tensors}
    new_groups: list[FusionGroup] = []
    for group in ir.groups:
        surviving = [op for op in group.ops if op is not affine_op]
        if surviving:
            new_groups.append(
                replace(
                    group,
                    ops=surviving,
                    dim_order=list(group.dim_order),
                    buffer_degrees=dict(group.buffer_degrees),
                    buffer_placements=dict(group.buffer_placements),
                )
            )
    new_ir = replace(
        ir,
        logical_tensors=new_tensors,
        op_inputs=new_op_inputs,
        op_outputs=new_op_outputs,
        op_kwargs=new_op_kwargs,
        op_axis_map=new_op_axis_map,
        op_tile_sizes=new_op_tile_sizes,
        op_blocking_dims=new_op_blocking_dims,
        op_skip_spec=new_op_skip_spec,
        groups=new_groups,
        edges=[],
    )
    rebuild_edges(new_ir)
    return new_ir


def _rewire(mapping: dict[str, str], masked_tensors: list[str], source_tensor: str | None) -> dict[str, str]:
    """Replace every value in ``mapping`` equal to a masked tensor with the upstream source."""
    result = dict(mapping)
    if source_tensor is not None:
        result = {k: source_tensor if v in masked_tensors else v for k, v in mapping.items()}
    return result
