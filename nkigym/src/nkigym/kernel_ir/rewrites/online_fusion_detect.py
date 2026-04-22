"""Online-fusion pattern detector on ``(KernelContext, KernelGraph)``.

Operates on op instances (as keyed in ``KernelContext``). The
input graph at detection time is assumed to be singleton
groups — one op per group — which holds at build time, before
the partition sampler runs.
"""

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
    """Flat walk over all ops across singleton groups."""
    return [op for group in graph.groups for op in group.ops]


def _producer_op(context: KernelContext, graph: KernelGraph, tensor_name: str) -> NKIOp | None:
    """Return the op that produces ``tensor_name`` (or None)."""
    result: NKIOp | None = None
    for op in _all_ops(graph):
        if tensor_name in context.op_outputs.get(op, []):
            result = op
            break
    return result


def _forward_reach(context: KernelContext, graph: KernelGraph) -> dict[NKIOp, set[NKIOp]]:
    """Forward transitive reach: ``reach[op]`` = ops downstream of ``op`` (including ``op``)."""
    ops = _all_ops(graph)
    adjacency: dict[NKIOp, list[NKIOp]] = {op: [] for op in ops}
    for op in ops:
        for name in context.op_outputs.get(op, []):
            for consumer in ops:
                if name in context.op_inputs.get(consumer, {}).values():
                    adjacency[op].append(consumer)
    reach: dict[NKIOp, set[NKIOp]] = {}
    for start in ops:
        visited: set[NKIOp] = {start}
        stack = [start]
        while stack:
            node = stack.pop()
            for succ in adjacency[node]:
                if succ not in visited:
                    visited.add(succ)
                    stack.append(succ)
        reach[start] = visited
    return reach


def detect_online_fusion(context: KernelContext, graph: KernelGraph) -> list[OnlineFusionCandidate]:
    """Return every X + Accumulation candidate."""
    candidates: list[OnlineFusionCandidate] = []
    reach = _forward_reach(context, graph)
    for x_op in _all_ops(graph):
        blocking_dims = context.op_blocking_dims.get(x_op, set())
        for dim_id in sorted(blocking_dims):
            fused = _match_on_dim(context, graph, x_op, dim_id, reach)
            if fused is not None:
                candidates.append(fused)
    return candidates


def _match_on_dim(
    context: KernelContext, graph: KernelGraph, x_op: NKIOp, dim_id: str, reach: dict[NKIOp, set[NKIOp]]
) -> OnlineFusionCandidate | None:
    """Find every accumulator that downstream-blocks on ``dim_id`` from ``x_op``."""
    x_outputs = set(context.op_outputs.get(x_op, []))
    acc_ops: list[NKIOp] = []
    role: str | None = None
    downstream = sorted(reach[x_op] - {x_op}, key=lambda op: id(op))
    for consumer in downstream:
        if dim_id not in context.op_blocking_dims.get(consumer, set()):
            continue
        match_role = _classify_accumulator(context, graph, x_op, x_outputs, consumer, dim_id)
        if match_role is None:
            continue
        acc_ops.append(consumer)
        role = match_role if role is None else _merge_roles(role, match_role)
    result: OnlineFusionCandidate | None = None
    if acc_ops and role is not None:
        result = OnlineFusionCandidate(x_op=x_op, accumulator_ops=tuple(acc_ops), blocking_dim=dim_id, scale_role=role)
    return result


def _classify_accumulator(
    context: KernelContext, graph: KernelGraph, x_op: NKIOp, x_outputs: set[str], consumer: NKIOp, dim_id: str
) -> str | None:
    """Return the online-fusion role label for this (X, consumer) pair, or None."""
    cls_name = type(consumer).NAME
    role: str | None = None
    if cls_name == "activation_reduce":
        role = _match_exp_bias(context, x_outputs, consumer, dim_id)
    elif cls_name == "nc_matmul":
        role = _match_matmul_through_scale(context, graph, x_op, consumer, dim_id)
    return role


def _match_exp_bias(context: KernelContext, x_outputs: set[str], consumer: NKIOp, dim_id: str) -> str | None:
    """True iff the consumer is ``activation_reduce(op='exp', bias=X, reduce_op='add')``."""
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
    context: KernelContext, graph: KernelGraph, x_op: NKIOp, matmul: NKIOp, dim_id: str
) -> str | None:
    """True iff the matmul's stationary lineage carries X as a per-partition scale."""
    inputs = context.op_inputs.get(matmul, {})
    stationary = inputs.get("stationary")
    role: str | None = None
    if stationary is not None and dim_id in context.op_blocking_dims.get(matmul, set()):
        role = _walk_back_for_separable(context, graph, stationary, x_op)
    return role


def _walk_back_for_separable(context: KernelContext, graph: KernelGraph, tensor_name: str, x_op: NKIOp) -> str | None:
    """BFS backward through separable-preserving ops; report the first X-consuming role."""
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
        role = _separable_role_via(context, producer, x_outputs)
        stack.extend(_separable_parents(context, producer))
    return role


def _separable_role_via(context: KernelContext, op: NKIOp, x_outputs: set[str]) -> str | None:
    """Return a role label iff this op consumes an X output in a separable slot."""
    cls_name = type(op).NAME
    kwargs = context.op_kwargs.get(op, {})
    inputs = context.op_inputs.get(op, {})
    role: str | None = None
    if cls_name == "tensor_scalar" and _literal_op(kwargs.get("op0")) == "multiply":
        operand = kwargs.get("operand0")
        if operand in context.logical_tensors:
            role = _scale_to_x_label(context, operand, x_outputs)
    elif cls_name in {"activation", "activation_reduce"} and _literal_op(kwargs.get("op")) == "exp":
        bias = inputs.get("bias")
        if bias in x_outputs:
            role = "exp_bias"
    return role


def _scale_to_x_label(context: KernelContext, scale_tensor: str, x_outputs: set[str]) -> str | None:
    """Classify how ``scale_tensor`` reaches an X output; label by the first inverse activation."""
    role: str | None = None
    act_ops: list[str] = []
    cur: str | None = scale_tensor
    ops = list(context.op_outputs)
    while cur is not None and cur not in x_outputs:
        producer: NKIOp | None = None
        for op in ops:
            if cur in context.op_outputs.get(op, []):
                producer = op
                break
        if producer is None:
            cur = None
            break
        cls_name = type(producer).NAME
        kwargs = context.op_kwargs.get(producer, {})
        if cls_name == "activation":
            act_ops.append(_literal_op(kwargs.get("op")) or "")
            cur = context.op_inputs.get(producer, {}).get("data")
        elif cls_name == "tensor_scalar":
            cur = context.op_inputs.get(producer, {}).get("data")
        else:
            cur = None
    if cur is not None:
        role = f"{act_ops[0]}_then_mul" if act_ops else "passthrough_mul"
    return role


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
