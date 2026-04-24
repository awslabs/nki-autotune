"""Online-fusion pattern detector on ``(KernelIR, KernelIR)``.

Atomic matching: a candidate fires **only** when X and every
accumulator sharing that X live in adjacent groups after loop
fusion has absorbed the intermediates between them. Walks through
separable chains are scoped to the accumulator's own group — if
loop fusion hasn't pulled an intermediate op into ``acc_gi``,
the walk exits the group and the match quietly doesn't fire.

The separable-chain walk is NOT a general-purpose "discover what
loop fusion would do" engine; it's a scale-role classifier
constrained to ops loop fusion already co-grouped.
"""

from collections import defaultdict
from dataclasses import dataclass

from nkigym.kernel_ir.ir import KernelIR
from nkigym.ops.base import NKIOp
from nkigym.ops.online_fusion_chain import NKIOnlineFusionChain


@dataclass(frozen=True)
class OnlineFusionCandidate:
    """One X + Accumulation pattern match.

    ``mode="create"`` builds a new composite from ``x_op`` +
    accumulators living in an adjacent group.

    ``mode="extend"`` augments an existing ``NKIOnlineFusionChain``
    composite (``x_op``) with new accumulator(s) living in the
    same group as the composite (after an earlier TF merge).
    """

    x_op: NKIOp
    accumulator_ops: tuple[NKIOp, ...]
    blocking_dim: str
    scale_role: str
    mode: str = "create"


def _all_ops(ir: KernelIR) -> list[NKIOp]:
    """Flat walk over all ops across groups."""
    return [op for group in ir.groups for op in group.ops]


def _group_of_map(ir: KernelIR) -> dict[int, int]:
    """``id(op) -> group_idx`` lookup."""
    return {id(op): gi for gi, group in enumerate(ir.groups) for op in group.ops}


def _producer_op(ir: KernelIR, tensor_name: str) -> NKIOp | None:
    """Return the op that produces ``tensor_name`` (or None)."""
    result: NKIOp | None = None
    for op in _all_ops(ir):
        if tensor_name in ir.op_outputs.get(op, []):
            result = op
            break
    return result


def detect_online_fusion(ir: KernelIR) -> list[OnlineFusionCandidate]:
    """Return every atomic X + Accumulation candidate (both create and extend modes)."""
    candidates: list[OnlineFusionCandidate] = []
    group_of = _group_of_map(ir)
    for x_op in _all_ops(ir):
        if isinstance(x_op, NKIOnlineFusionChain):
            extended = _detect_extend(ir, x_op, group_of)
            if extended is not None:
                candidates.append(extended)
            continue
        blocking_dims = ir.op_blocking_dims.get(x_op, set())
        for dim_id in sorted(blocking_dims):
            fused = _match_on_dim(ir, x_op, dim_id, group_of)
            if fused is not None:
                candidates.append(fused)
    return candidates


def _detect_extend(
    ir: KernelIR, composite_op: NKIOnlineFusionChain, group_of: dict[int, int]
) -> OnlineFusionCandidate | None:
    """Find same-group accumulators whose data lineage traces back to ``composite_op``'s outputs."""
    op_cls = type(composite_op)
    acc_dim = op_cls.ACCUMULATION_DIM
    comp_gi = group_of[id(composite_op)]
    comp_outputs = set(ir.op_outputs.get(composite_op, []))
    existing_sources = _existing_accumulator_sources(op_cls)
    new_accs = _collect_new_accumulators(ir, composite_op, acc_dim, comp_gi, comp_outputs, existing_sources, group_of)
    return _build_extend_candidate(composite_op, acc_dim, new_accs) if (acc_dim and new_accs) else None


def _collect_new_accumulators(
    ir: KernelIR,
    composite_op: NKIOp,
    acc_dim: str,
    comp_gi: int,
    comp_outputs: set[str],
    existing_sources: set[NKIOp],
    group_of: dict[int, int],
) -> list[tuple[NKIOp, str]]:
    """Walk every op in the composite's group; return accumulator candidates not already absorbed."""
    result: list[tuple[NKIOp, str]] = []
    for consumer in sorted(_all_ops(ir), key=lambda op: id(op)):
        if not _is_extend_consumer(consumer, composite_op, comp_gi, acc_dim, existing_sources, group_of, ir):
            continue
        role = _classify_accumulator(ir, composite_op, comp_outputs, consumer, acc_dim, group_of, comp_gi)
        if role is not None:
            result.append((consumer, role))
    return result


def _is_extend_consumer(
    consumer: NKIOp,
    composite_op: NKIOp,
    comp_gi: int,
    acc_dim: str,
    existing_sources: set[NKIOp],
    group_of: dict[int, int],
    ir: KernelIR,
) -> bool:
    """True iff consumer is a same-group, acc-dim-blocking, not-already-absorbed extension candidate."""
    return (
        consumer is not composite_op
        and consumer not in existing_sources
        and group_of[id(consumer)] == comp_gi
        and acc_dim in ir.op_blocking_dims.get(consumer, set())
    )


def _build_extend_candidate(
    composite_op: NKIOp, acc_dim: str, new_accs: list[tuple[NKIOp, str]]
) -> OnlineFusionCandidate:
    """Fold per-accumulator roles into one OnlineFusionCandidate in extend mode."""
    acc_ops = tuple(op for op, _ in new_accs)
    role = new_accs[0][1]
    for _op, r in new_accs[1:]:
        role = _merge_roles(role, r)
    return OnlineFusionCandidate(
        x_op=composite_op, accumulator_ops=acc_ops, blocking_dim=acc_dim, scale_role=role, mode="extend"
    )


def _existing_accumulator_sources(op_cls: type[NKIOp]) -> set[NKIOp]:
    """Set of original ``source_op`` instances already absorbed as accumulators."""
    sources: set[NKIOp] = set()
    specs = getattr(op_cls, "ACCUMULATOR_SPECS", ())
    for spec in specs:
        source = getattr(spec, "source_op", None)
        if source is not None:
            sources.add(source)
    return sources


def _match_on_dim(ir: KernelIR, x_op: NKIOp, dim_id: str, group_of: dict[int, int]) -> OnlineFusionCandidate | None:
    """Emit one candidate iff all accumulators for ``(x_op, dim_id)`` share one adjacent group."""
    x_gi = group_of[id(x_op)]
    x_outputs = set(ir.op_outputs.get(x_op, []))
    by_group: dict[int, list[tuple[NKIOp, str]]] = defaultdict(list)
    for consumer in sorted(_all_ops(ir), key=lambda op: id(op)):
        consumer_gi = group_of[id(consumer)]
        if consumer_gi == x_gi:
            continue
        if dim_id not in ir.op_blocking_dims.get(consumer, set()):
            continue
        match_role = _classify_accumulator(ir, x_op, x_outputs, consumer, dim_id, group_of, consumer_gi)
        if match_role is None:
            continue
        by_group[consumer_gi].append((consumer, match_role))
    result: OnlineFusionCandidate | None = None
    if len(by_group) == 1:
        (pairs,) = by_group.values()
        acc_ops = tuple(op for op, _ in pairs)
        role = pairs[0][1]
        for _op, r in pairs[1:]:
            role = _merge_roles(role, r)
        result = OnlineFusionCandidate(x_op=x_op, accumulator_ops=acc_ops, blocking_dim=dim_id, scale_role=role)
    return result


def _classify_accumulator(
    ir: KernelIR, x_op: NKIOp, x_outputs: set[str], consumer: NKIOp, dim_id: str, group_of: dict[int, int], acc_gi: int
) -> str | None:
    """Return the online-fusion role label for this ``(X, consumer)`` pair, or None."""
    cls_name = type(consumer).NAME
    role: str | None = None
    if cls_name == "activation_reduce":
        role = _match_exp_bias(ir, x_outputs, consumer, dim_id)
    elif cls_name == "nc_matmul":
        role = _match_matmul_through_scale(ir, x_op, consumer, dim_id, group_of, acc_gi)
    return role


def _match_exp_bias(ir: KernelIR, x_outputs: set[str], consumer: NKIOp, dim_id: str) -> str | None:
    """True iff the consumer is ``activation_reduce(op='exp', bias=X, reduce_op='add')`` (atomic, no walk)."""
    kwargs = ir.op_kwargs.get(consumer, {})
    inputs = ir.op_inputs.get(consumer, {})
    bias_tensor = inputs.get("bias")
    role: str | None = None
    if (
        bias_tensor in x_outputs
        and _literal_op(kwargs.get("op")) == "exp"
        and _literal_op(kwargs.get("reduce_op")) == "add"
        and dim_id in ir.op_blocking_dims.get(consumer, set())
    ):
        role = "exp_bias"
    return role


def _match_matmul_through_scale(
    ir: KernelIR, x_op: NKIOp, matmul: NKIOp, dim_id: str, group_of: dict[int, int], acc_gi: int
) -> str | None:
    """True iff the matmul's stationary lineage (inside ``acc_gi``) carries X as a per-partition scale."""
    inputs = ir.op_inputs.get(matmul, {})
    stationary = inputs.get("stationary")
    role: str | None = None
    if stationary is not None and dim_id in ir.op_blocking_dims.get(matmul, set()):
        role = _walk_back_for_separable(ir, stationary, x_op, group_of, acc_gi)
    return role


def _walk_back_for_separable(
    ir: KernelIR, tensor_name: str, x_op: NKIOp, group_of: dict[int, int], acc_gi: int
) -> str | None:
    """BFS backward through separable-preserving ops **inside acc_gi**; label by the first X-consuming op."""
    x_outputs = set(ir.op_outputs.get(x_op, []))
    visited: set[str] = set()
    stack: list[str] = [tensor_name]
    role: str | None = None
    reached_x = False
    while stack and role is None and not reached_x:
        cur = stack.pop()
        if cur in visited:
            continue
        visited.add(cur)
        if cur in x_outputs:
            reached_x = True
            continue
        producer = _producer_op(ir, cur)
        if producer is None or group_of.get(id(producer)) != acc_gi:
            continue
        role = _separable_role_via(ir, producer, x_outputs, group_of, acc_gi)
        stack.extend(_separable_parents(ir, producer))
    if role is None and reached_x:
        role = "passthrough_mul"
    return role


def _separable_role_via(
    ir: KernelIR, op: NKIOp, x_outputs: set[str], group_of: dict[int, int], acc_gi: int
) -> str | None:
    """Return a role label iff this op consumes an X output in a separable slot."""
    cls_name = type(op).NAME
    kwargs = ir.op_kwargs.get(op, {})
    inputs = ir.op_inputs.get(op, {})
    role: str | None = None
    if cls_name == "tensor_scalar" and _literal_op(kwargs.get("op0")) == "multiply":
        operand = kwargs.get("operand0")
        if operand is not None and ir.has_tensor(operand):
            role = _scale_to_x_label(ir, operand, x_outputs, group_of, acc_gi)
    elif cls_name in {"activation", "activation_reduce"} and _literal_op(kwargs.get("op")) == "exp":
        bias = inputs.get("bias")
        if bias in x_outputs:
            role = "exp_bias"
    return role


_SCALE_WALK_OPS: frozenset[str] = frozenset({"activation", "tensor_scalar"})


def _scale_to_x_label(
    ir: KernelIR, scale_tensor: str, x_outputs: set[str], group_of: dict[int, int], acc_gi: int
) -> str | None:
    """Classify how ``scale_tensor`` reaches an X output — walk stays inside acc_gi."""
    act_ops: list[str] = []
    cur: str | None = scale_tensor
    ops_by_output: dict[str, NKIOp] = {name: op for op, names in ir.op_outputs.items() for name in names}
    reached_x = False
    while cur is not None and not reached_x:
        if cur in x_outputs:
            reached_x = True
            continue
        producer = ops_by_output.get(cur)
        in_group = producer is not None and group_of.get(id(producer)) == acc_gi
        cls_name = type(producer).NAME if producer is not None else ""
        cur = _next_scale_cur(ir, producer, cls_name, act_ops) if in_group else None
    label: str | None = None
    if reached_x:
        label = f"{act_ops[0]}_then_mul" if act_ops else "passthrough_mul"
    return label


def _next_scale_cur(ir: KernelIR, producer: NKIOp | None, cls_name: str, act_ops: list[str]) -> str | None:
    """Advance the walk through one separable-preserving producer; append activation label if any."""
    data: str | None = None
    if producer is not None and cls_name in _SCALE_WALK_OPS:
        if cls_name == "activation":
            act_ops.append(_literal_op(ir.op_kwargs.get(producer, {}).get("op")) or "")
        data = ir.op_inputs.get(producer, {}).get("data")
    return data


def _separable_parents(ir: KernelIR, op: NKIOp) -> list[str]:
    """Return parent tensors to keep walking through for separable-preserving ops."""
    cls_name = type(op).NAME
    inputs = ir.op_inputs.get(op, {})
    result: list[str] = []
    if cls_name == "nc_transpose" and "data" in inputs:
        result.append(inputs["data"])
    elif cls_name in {"tensor_scalar", "activation", "activation_reduce"} and "data" in inputs:
        result.append(inputs["data"])
    elif cls_name == "affine_select" and "on_true_tile" in inputs:
        result.append(inputs["on_true_tile"])
    return result


def _literal_op(raw: str | None) -> str | None:
    """Strip quotes from a traced op-name kwarg."""
    return None if raw is None else raw.strip("'\"")


def _merge_roles(a: str, b: str) -> str:
    """Join multiple roles on one X candidate into a stable compound label."""
    return a if a == b else f"{a}+{b}"
