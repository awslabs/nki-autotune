"""Online-fusion pattern detector on ``(KernelContext, KernelGraph)``.

Atomic matching: a candidate fires **only** when X and every
accumulator sharing that X live in adjacent groups after trivial
fusion has absorbed the intermediates between them. Walks through
separable chains are scoped to the accumulator's own group — if
trivial fusion hasn't pulled an intermediate op into ``acc_gi``,
the walk exits the group and the match quietly doesn't fire.

The separable-chain walk is NOT a general-purpose "discover what
trivial fusion would do" engine; it's a scale-role classifier
constrained to ops trivial fusion already co-grouped.
"""

from collections import defaultdict
from dataclasses import dataclass

from nkigym.kernel_ir.context.context import KernelContext
from nkigym.kernel_ir.graph.graph import KernelGraph
from nkigym.ops.base import NKIOp


@dataclass(frozen=True)
class OnlineFusionCandidate:
    """One X + Accumulation pattern match."""

    x_op: NKIOp
    accumulator_ops: tuple[NKIOp, ...]
    blocking_dim: str
    scale_role: str


def _all_ops(graph: KernelGraph) -> list[NKIOp]:
    """Flat walk over all ops across groups."""
    return [op for group in graph.groups for op in group.ops]


def _group_of_map(graph: KernelGraph) -> dict[int, int]:
    """``id(op) -> group_idx`` lookup."""
    return {id(op): gi for gi, group in enumerate(graph.groups) for op in group.ops}


def _producer_op(context: KernelContext, graph: KernelGraph, tensor_name: str) -> NKIOp | None:
    """Return the op that produces ``tensor_name`` (or None)."""
    result: NKIOp | None = None
    for op in _all_ops(graph):
        if tensor_name in context.op_outputs.get(op, []):
            result = op
            break
    return result


def detect_online_fusion(context: KernelContext, graph: KernelGraph) -> list[OnlineFusionCandidate]:
    """Return every atomic X + Accumulation candidate."""
    candidates: list[OnlineFusionCandidate] = []
    group_of = _group_of_map(graph)
    for x_op in _all_ops(graph):
        blocking_dims = context.op_blocking_dims.get(x_op, set())
        for dim_id in sorted(blocking_dims):
            fused = _match_on_dim(context, graph, x_op, dim_id, group_of)
            if fused is not None:
                candidates.append(fused)
    return candidates


def _match_on_dim(
    context: KernelContext, graph: KernelGraph, x_op: NKIOp, dim_id: str, group_of: dict[int, int]
) -> OnlineFusionCandidate | None:
    """Emit one candidate iff all accumulators for ``(x_op, dim_id)`` share one adjacent group."""
    x_gi = group_of[id(x_op)]
    x_outputs = set(context.op_outputs.get(x_op, []))
    by_group: dict[int, list[tuple[NKIOp, str]]] = defaultdict(list)
    for consumer in sorted(_all_ops(graph), key=lambda op: id(op)):
        consumer_gi = group_of[id(consumer)]
        if consumer_gi == x_gi:
            continue
        if dim_id not in context.op_blocking_dims.get(consumer, set()):
            continue
        match_role = _classify_accumulator(context, graph, x_op, x_outputs, consumer, dim_id, group_of, consumer_gi)
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
    context: KernelContext,
    graph: KernelGraph,
    x_op: NKIOp,
    x_outputs: set[str],
    consumer: NKIOp,
    dim_id: str,
    group_of: dict[int, int],
    acc_gi: int,
) -> str | None:
    """Return the online-fusion role label for this ``(X, consumer)`` pair, or None."""
    cls_name = type(consumer).NAME
    role: str | None = None
    if cls_name == "activation_reduce":
        role = _match_exp_bias(context, x_outputs, consumer, dim_id)
    elif cls_name == "nc_matmul":
        role = _match_matmul_through_scale(context, graph, x_op, consumer, dim_id, group_of, acc_gi)
    return role


def _match_exp_bias(context: KernelContext, x_outputs: set[str], consumer: NKIOp, dim_id: str) -> str | None:
    """True iff the consumer is ``activation_reduce(op='exp', bias=X, reduce_op='add')`` (atomic, no walk)."""
    kwargs = context.op_kwargs.get(consumer, {})
    inputs = context.op_inputs.get(consumer, {})
    bias_tensor = inputs.get("bias")
    role: str | None = None
    if (
        bias_tensor in x_outputs
        and _literal_op(kwargs.get("op")) == "exp"
        and _literal_op(kwargs.get("reduce_op")) == "add"
        and dim_id in context.op_blocking_dims.get(consumer, set())
    ):
        role = "exp_bias"
    return role


def _match_matmul_through_scale(
    context: KernelContext,
    graph: KernelGraph,
    x_op: NKIOp,
    matmul: NKIOp,
    dim_id: str,
    group_of: dict[int, int],
    acc_gi: int,
) -> str | None:
    """True iff the matmul's stationary lineage (inside ``acc_gi``) carries X as a per-partition scale."""
    inputs = context.op_inputs.get(matmul, {})
    stationary = inputs.get("stationary")
    role: str | None = None
    if stationary is not None and dim_id in context.op_blocking_dims.get(matmul, set()):
        role = _walk_back_for_separable(context, graph, stationary, x_op, group_of, acc_gi)
    return role


def _walk_back_for_separable(
    context: KernelContext, graph: KernelGraph, tensor_name: str, x_op: NKIOp, group_of: dict[int, int], acc_gi: int
) -> str | None:
    """BFS backward through separable-preserving ops **inside acc_gi**; label by the first X-consuming op."""
    x_outputs = set(context.op_outputs.get(x_op, []))
    visited: set[str] = set()
    stack: list[str] = [tensor_name]
    role: str | None = None
    while stack and role is None:
        cur = stack.pop()
        if cur in visited:
            continue
        visited.add(cur)
        producer = _producer_op(context, graph, cur)
        if producer is None:
            continue
        if group_of.get(id(producer)) != acc_gi:
            continue
        role = _separable_role_via(context, producer, x_outputs, group_of, acc_gi)
        stack.extend(_separable_parents(context, producer))
    return role


def _separable_role_via(
    context: KernelContext, op: NKIOp, x_outputs: set[str], group_of: dict[int, int], acc_gi: int
) -> str | None:
    """Return a role label iff this op consumes an X output in a separable slot."""
    cls_name = type(op).NAME
    kwargs = context.op_kwargs.get(op, {})
    inputs = context.op_inputs.get(op, {})
    role: str | None = None
    if cls_name == "tensor_scalar" and _literal_op(kwargs.get("op0")) == "multiply":
        operand = kwargs.get("operand0")
        if operand in context.logical_tensors:
            role = _scale_to_x_label(context, operand, x_outputs, group_of, acc_gi)
    elif cls_name in {"activation", "activation_reduce"} and _literal_op(kwargs.get("op")) == "exp":
        bias = inputs.get("bias")
        if bias in x_outputs:
            role = "exp_bias"
    return role


_SCALE_WALK_OPS: frozenset[str] = frozenset({"activation", "tensor_scalar"})


def _scale_to_x_label(
    context: KernelContext, scale_tensor: str, x_outputs: set[str], group_of: dict[int, int], acc_gi: int
) -> str | None:
    """Classify how ``scale_tensor`` reaches an X output — walk stays inside acc_gi."""
    act_ops: list[str] = []
    cur: str | None = scale_tensor
    ops_by_output: dict[str, NKIOp] = {name: op for op, names in context.op_outputs.items() for name in names}
    reached_x = False
    while cur is not None and not reached_x:
        if cur in x_outputs:
            reached_x = True
            continue
        producer = ops_by_output.get(cur)
        in_group = producer is not None and group_of.get(id(producer)) == acc_gi
        cls_name = type(producer).NAME if producer is not None else ""
        cur = _next_scale_cur(context, producer, cls_name, act_ops) if in_group else None
    label: str | None = None
    if reached_x:
        label = f"{act_ops[0]}_then_mul" if act_ops else "passthrough_mul"
    return label


def _next_scale_cur(context: KernelContext, producer: NKIOp | None, cls_name: str, act_ops: list[str]) -> str | None:
    """Advance the walk through one separable-preserving producer; append activation label if any."""
    data: str | None = None
    if producer is not None and cls_name in _SCALE_WALK_OPS:
        if cls_name == "activation":
            act_ops.append(_literal_op(context.op_kwargs.get(producer, {}).get("op")) or "")
        data = context.op_inputs.get(producer, {}).get("data")
    return data


def _separable_parents(context: KernelContext, op: NKIOp) -> list[str]:
    """Return parent tensors to keep walking through for separable-preserving ops."""
    cls_name = type(op).NAME
    inputs = context.op_inputs.get(op, {})
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
