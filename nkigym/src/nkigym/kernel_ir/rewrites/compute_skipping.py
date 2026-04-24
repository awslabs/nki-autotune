"""``ComputeSkipping``: causal-mask tile skipping.

Lifts the per-element causal predicate from ``NKIAffineSelect`` into
a tile-granularity three-state classifier that absorbs the producers
AND consumers of the masked tensor. See ``compute_skipping.md`` for
the math.

Semantics:

* **Predicate.** The affine_select op carries
  ``kwargs["offset"]`` / ``kwargs["channel_multiplier"]`` /
  ``kwargs["free_step"]`` / ``kwargs["on_false_value"]`` — the
  closed-form causal predicate
  ``offset + p*channel_multiplier + f*free_step >= 0``.
* **Upstream trace.** Walk producers of the masked tensor; an op
  is absorbed upstream iff every consumer of its output is itself
  absorbed (output flows exclusively into the masked region).
* **Downstream trace.** Walk consumers; an op is absorbed downstream
  iff (a) it still iterates over the mask's free axis, and (b) it
  either produces the kernel output or every one of its consumers
  is absorbed. Ops that reduce away the free axis (e.g. final
  ``1/S`` normalization) are **not** absorbed.
* **Three states**, emitted per-op via ``Op.attrs["skip_spec"]``:
    - ``skip_all`` — predicate is always False on the tile → no
      absorbed op runs.
    - ``compute_only`` — predicate is always True on the tile →
      every absorbed op runs; the affine_select op itself is
      replaced with ``tensor_copy``.
    - ``mask_and_compute`` — predicate straddles the boundary →
      every absorbed op runs, including affine_select.

Output:

* Deletes the ``NKIAffineSelect`` op from ``ir.ops``; rewires
  consumers of its output to read its input ``data`` directly (the
  mask is now inlined into the per-op skip_spec).
* Attaches ``Op.attrs["skip_spec"] = ComputeSkipSpec(...)`` on
  every absorbed op; the first absorbed op downstream gains
  ``attrs["mask_and_compute"] = True`` so codegen inlines the
  mask comparison at that site.
* Records the set of boundary tensors (absorbed-op outputs read
  by non-absorbed ops) on ``ComputeSkipSpec.boundary_tensors`` so
  codegen can emit a ``memset(tile, on_false_value)`` on
  ``skip_all`` tiles for cross-boundary tensors with BOTH the
  partition and free dim.
"""

from dataclasses import dataclass, field, replace

from nkigym.kernel_ir.ir import KernelIR, Op


@dataclass(frozen=True)
class ComputeSkipSpec:
    """Per-op annotation attached by ``ComputeSkipping``.

    Attributes:
        offset: Constant term of the affine predicate.
        channel_multiplier: Coefficient on the partition index.
        free_step: Coefficient on the free index.
        on_false_value: Sentinel written on masked elements
            (typically ``-float("inf")`` for softmax).
        partition_dim: Dim id whose index enters ``p*channel_multiplier``.
        free_dim: Dim id whose index enters ``f*free_step``.
        boundary_tensors: Tensor names whose per-tile ``skip_all``
            branch must emit a ``memset(on_false_value)`` so that
            non-absorbed readers see the sentinel.
        is_mask_site: True only on the op that owns the inlined
            ``affine_select`` comparison — codegen emits the
            mask+compute branch here.
    """

    offset: float
    channel_multiplier: float
    free_step: float
    on_false_value: float
    partition_dim: str
    free_dim: str
    boundary_tensors: frozenset[str] = field(default_factory=frozenset)
    is_mask_site: bool = False


@dataclass(frozen=True)
class ComputeSkipMatch:
    """One ``NKIAffineSelect`` absorption to apply."""

    affine_select_index: int
    upstream_indices: tuple[int, ...]
    downstream_indices: tuple[int, ...]


class ComputeSkipping:
    """Lift causal affine_select into per-tile skip groups."""

    name = "compute_skipping"

    def match(self, ir: KernelIR) -> list[ComputeSkipMatch]:
        """Enumerate every ``NKIAffineSelect`` op with a legal absorption set."""
        matches: list[ComputeSkipMatch] = []
        for i, op in enumerate(ir.ops):
            if op.kind != "NKIAffineSelect":
                continue
            if op.attrs.get("skip_spec") is not None:
                """Already absorbed — idempotent."""
                continue
            upstream = _trace_upstream(ir, i)
            downstream = _trace_downstream(ir, i)
            if not upstream and not downstream:
                continue
            matches.append(
                ComputeSkipMatch(
                    affine_select_index=i, upstream_indices=tuple(upstream), downstream_indices=tuple(downstream)
                )
            )
        return matches

    def apply(self, ir: KernelIR, instance: ComputeSkipMatch) -> KernelIR:
        """Delete the affine_select, inline its mask, annotate absorbed ops."""
        as_op = ir.ops[instance.affine_select_index]
        if as_op.kind != "NKIAffineSelect":
            raise ValueError(
                f"ComputeSkipping.apply: expected NKIAffineSelect at "
                f"{instance.affine_select_index}, got {as_op.kind!r}"
            )
        spec_fields = _extract_affine_predicate(as_op)
        absorbed_indices = set(instance.upstream_indices) | set(instance.downstream_indices)
        boundary = _compute_boundary_tensors(ir, absorbed_indices)
        mask_site_index = _pick_mask_site(ir, instance.downstream_indices)

        new_ops = _build_rewritten_ops(
            ir=ir,
            as_index=instance.affine_select_index,
            absorbed_indices=absorbed_indices,
            mask_site_index=mask_site_index,
            spec_fields=spec_fields,
            boundary=boundary,
        )
        new_edges = _rebuild_edges_after_drop(
            ir.edges, dropped={instance.affine_select_index}, redirect=_affine_redirect(as_op)
        )
        return replace(ir, ops=new_ops, edges=new_edges)


def _trace_upstream(ir: KernelIR, affine_select_index: int) -> list[int]:
    """Return op indices whose output flows exclusively into the affine_select's input.

    An op ``u`` is absorbed iff every consumer of every output of
    ``u`` is either ``affine_select`` itself or another already-
    absorbed op. Implemented as a reverse BFS from the seed input.
    """
    as_op = ir.ops[affine_select_index]
    if "data" not in as_op.inputs:
        return []
    seed = as_op.inputs["data"]
    absorbed: set[int] = set()
    frontier: list[str] = [seed]
    while frontier:
        tensor = frontier.pop()
        producer = ir.producer_of(tensor)
        if producer is None or producer == affine_select_index:
            continue
        if producer in absorbed:
            continue
        if not _all_consumers_absorbed(ir, producer, absorbed | {affine_select_index}):
            continue
        absorbed.add(producer)
        for role, tname in ir.ops[producer].inputs.items():
            _ = role
            frontier.append(tname)
    return sorted(absorbed)


def _trace_downstream(ir: KernelIR, affine_select_index: int) -> list[int]:
    """Return consumer op indices that propagate the mask sentinel.

    The trace keeps ops that still iterate over the mask's free
    dim and whose outputs either flow into the kernel return or
    into other absorbed ops. Ops that reduce the free axis away
    (e.g. final ``1/S`` normalization) terminate the walk.
    """
    as_op = ir.ops[affine_select_index]
    if not as_op.outputs:
        return []
    free_dim = as_op.kwargs.get("free_dim") or as_op.attrs.get("free_dim")
    absorbed: set[int] = set()
    frontier: list[str] = list(as_op.outputs)
    while frontier:
        tensor = frontier.pop()
        for ci, role in _consumers_of(ir, tensor):
            _ = role
            if ci in absorbed:
                continue
            consumer = ir.ops[ci]
            if free_dim is not None and not _op_uses_dim(ir, consumer, free_dim):
                continue
            absorbed.add(ci)
            for out in consumer.outputs:
                frontier.append(out)
    return sorted(absorbed)


def _all_consumers_absorbed(ir: KernelIR, op_index: int, absorbed: set[int]) -> bool:
    """True iff every consumer of every output of ``op_index`` is in ``absorbed``."""
    for out in ir.ops[op_index].outputs:
        for ci, _role in _consumers_of(ir, out):
            if ci not in absorbed:
                return False
    return True


def _consumers_of(ir: KernelIR, tensor_name: str) -> list[tuple[int, str]]:
    """``(op_index, role)`` pairs for every op that reads ``tensor_name``."""
    result: list[tuple[int, str]] = []
    for i, op in enumerate(ir.ops):
        for role, name in op.inputs.items():
            if name == tensor_name:
                result.append((i, role))
    return result


def _op_uses_dim(ir: KernelIR, op: Op, dim_id: str) -> bool:
    """True iff any of ``op``'s input or output tensors has ``dim_id``."""
    for tname in list(op.inputs.values()) + list(op.outputs):
        if not ir.has_tensor(tname):
            continue
        info = ir.tensor_info(tname)
        if dim_id in info.dim_ids:
            return True
    return dim_id in op.blocking_dims


def _extract_affine_predicate(as_op: Op) -> dict[str, float | str]:
    """Pull the affine coefficients out of the NKIAffineSelect op."""
    return {
        "offset": float(as_op.kwargs["offset"]),
        "channel_multiplier": float(as_op.kwargs["channel_multiplier"]),
        "free_step": float(as_op.kwargs["free_step"]),
        "on_false_value": float(as_op.kwargs["on_false_value"]),
        "partition_dim": str(as_op.kwargs.get("partition_dim", as_op.attrs.get("partition_dim", ""))),
        "free_dim": str(as_op.kwargs.get("free_dim", as_op.attrs.get("free_dim", ""))),
    }


def _compute_boundary_tensors(ir: KernelIR, absorbed: set[int]) -> frozenset[str]:
    """Tensors produced inside the skip group but read outside it.

    These need a ``memset(on_false_value)`` in the ``skip_all``
    branch so non-absorbed readers see the sentinel.
    """
    result: set[str] = set()
    for ai in absorbed:
        for out in ir.ops[ai].outputs:
            for ci, _role in _consumers_of(ir, out):
                if ci not in absorbed:
                    result.add(out)
                    break
    return frozenset(result)


def _pick_mask_site(ir: KernelIR, downstream: tuple[int, ...]) -> int | None:
    """The first downstream op (in IR order) becomes the mask+compute site."""
    if not downstream:
        return None
    return min(downstream)


def _affine_redirect(as_op: Op) -> dict[str, str]:
    """Map ``affine_select output → affine_select input``.

    Consumers of the affine_select's output read its input directly
    after the rewrite — the mask is inlined into the downstream
    op's ``skip_spec``.
    """
    if not as_op.outputs or "data" not in as_op.inputs:
        return {}
    return {as_op.outputs[0]: as_op.inputs["data"]}


def _build_rewritten_ops(
    ir: KernelIR,
    as_index: int,
    absorbed_indices: set[int],
    mask_site_index: int | None,
    spec_fields: dict[str, float | str],
    boundary: frozenset[str],
) -> list[Op]:
    """Drop the affine_select and tag each absorbed op with ``skip_spec``."""
    partition_dim = str(spec_fields["partition_dim"])
    free_dim = str(spec_fields["free_dim"])
    new_ops: list[Op] = []
    for i, op in enumerate(ir.ops):
        if i == as_index:
            continue
        if i in absorbed_indices:
            spec = ComputeSkipSpec(
                offset=float(spec_fields["offset"]),
                channel_multiplier=float(spec_fields["channel_multiplier"]),
                free_step=float(spec_fields["free_step"]),
                on_false_value=float(spec_fields["on_false_value"]),
                partition_dim=partition_dim,
                free_dim=free_dim,
                boundary_tensors=boundary,
                is_mask_site=(i == mask_site_index),
            )
            new_attrs = dict(op.attrs)
            new_attrs["skip_spec"] = spec
            if i == mask_site_index:
                new_attrs["mask_and_compute"] = True
            new_inputs = _redirect_inputs(op.inputs, as_op=ir.ops[as_index])
            new_ops.append(replace(op, attrs=new_attrs, inputs=new_inputs))
        else:
            new_inputs = _redirect_inputs(op.inputs, as_op=ir.ops[as_index])
            if new_inputs == op.inputs:
                new_ops.append(op)
            else:
                new_ops.append(replace(op, inputs=new_inputs))
    return new_ops


def _redirect_inputs(inputs: dict[str, str], as_op: Op) -> dict[str, str]:
    """Rewrite any input reading the affine_select's output to read its input instead."""
    if not as_op.outputs or "data" not in as_op.inputs:
        return inputs
    masked = as_op.outputs[0]
    source = as_op.inputs["data"]
    if masked not in inputs.values():
        return inputs
    return {role: (source if name == masked else name) for role, name in inputs.items()}


def _rebuild_edges_after_drop(
    edges: list[tuple[int, int, str, str]], dropped: set[int], redirect: dict[str, str]
) -> list[tuple[int, int, str, str]]:
    """Renumber indices past the dropped ops; skip edges touching them.

    Consumers of a redirected tensor now read the redirect target;
    those edges are dropped here since producers stay in place.
    """
    index_map: dict[int, int] = {}
    next_new = 0
    for old in range(max(edges, key=lambda e: max(e[0], e[1]))[1] + 1 if edges else 0):
        if old in dropped:
            continue
        index_map[old] = next_new
        next_new += 1
    new_edges: list[tuple[int, int, str, str]] = []
    for src, dst, tensor, role in edges:
        if src in dropped or dst in dropped:
            continue
        if tensor in redirect:
            continue
        new_edges.append((index_map[src], index_map[dst], tensor, role))
    return new_edges
