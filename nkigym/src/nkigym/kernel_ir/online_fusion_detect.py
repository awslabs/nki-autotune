"""Online-fusion pattern detector.

Identifies X + Accumulation patterns in the ``(DimAnalysis, OpGraph)``
pair produced by the analysis stages of ``build_ir``. The rewrite
at ``online_fusion_rewrite.py`` then mutates both to materialize the
fused-loop machinery. Detector runs on the raw analysis objects (not
a full ``KernelIR``) so the rewrite can fire before the rejection
sampler assembles the IR.

Recognized patterns:

* **Reducer → matmul** (e.g. rmsnorm + matmul). X = an op whose
  ``BLOCKING_AXES`` resolve to a concrete blocking dim ``d``; a
  downstream matmul blocks on the same ``d``; the reduction output
  propagates to the matmul's stationary operand through a chain
  that preserves multiplicative separability (``tensor_scalar``
  affine, ``activation`` unary inverse, ``tensor_scalar(multiply,
  operand0=<reduced>)``, transpose, matmul).
* **Reducer → activation_reduce** (attention's first fusion). X =
  reducer on ``d``; consumer ``activation_reduce`` with ``op='exp'``
  uses the reduced tensor as ``bias`` on the same ``d``. Trivially
  separable as ``exp(bias + data) = e^bias · e^data``.

Unrecognized patterns are silently skipped.
"""

from dataclasses import dataclass

from nkigym.kernel_ir.dim_analysis import DimAnalysis
from nkigym.kernel_ir.op_graph import OpGraph


@dataclass(frozen=True)
class OnlineFusionCandidate:
    """One X + Accumulation pattern match.

    Attributes:
        x_op_idx: Op whose accumulation barrier online fusion breaks.
            Its output carries the running reduction state ``O_0``
            in Algorithm 4.
        accumulator_op_indices: Ops currently blocking on
            ``blocking_dim`` because their per-tile contribution
            depends on the full ``O_0_K``. After the rewrite they
            consume the per-chunk partial ``O_0_j`` and emit a
            scale-coefficient correction each iteration.
        blocking_dim: Concrete dim id (e.g. ``"d1"``) that this
            rewrite promotes from ``SERIAL`` to ``ACCUMULATION``.
        scale_role: Shape of the ``g_B(O_0)`` chain — one of
            ``"rsqrt_then_mul"``, ``"reciprocal_then_mul"``,
            ``"passthrough_mul"``, or ``"exp_bias"``. Mixed-role
            candidates are joined with ``"+"`` (rare).
    """

    x_op_idx: int
    accumulator_op_indices: tuple[int, ...]
    blocking_dim: str
    scale_role: str


def detect_online_fusion(da: DimAnalysis, graph: OpGraph) -> list[OnlineFusionCandidate]:
    """Return every X + Accumulation candidate in ``(da, graph)``.

    Each candidate describes one producer-reducer ``x_op_idx`` plus
    the set of downstream ops that block on the same dim because
    they depend on the full reduction. Producers that block on
    multiple dims yield one candidate per dim.
    """
    candidates: list[OnlineFusionCandidate] = []
    reach = _forward_reach(graph)
    for x_op_idx in range(len(graph.op_classes)):
        blocking_dims = da.op_blocking_dims(x_op_idx)
        for dim_id in sorted(blocking_dims):
            fused = _match_on_dim(da, graph, x_op_idx, dim_id, reach)
            if fused is not None:
                candidates.append(fused)
    return candidates


def _match_on_dim(
    da: DimAnalysis, graph: OpGraph, x_op_idx: int, dim_id: str, reach: list[set[int]]
) -> OnlineFusionCandidate | None:
    """Find every accumulator that downstream-blocks on ``dim_id`` from ``x_op_idx``."""
    x_outputs = set(graph.op_tensors[x_op_idx][1])
    acc_ops: list[int] = []
    role: str | None = None
    for consumer_idx in sorted(reach[x_op_idx] - {x_op_idx}):
        if dim_id not in da.op_blocking_dims(consumer_idx):
            continue
        match_role = _classify_accumulator(da, graph, x_op_idx, x_outputs, consumer_idx, dim_id)
        if match_role is None:
            continue
        acc_ops.append(consumer_idx)
        role = match_role if role is None else _merge_roles(role, match_role)
    result = (
        None
        if not acc_ops or role is None
        else OnlineFusionCandidate(
            x_op_idx=x_op_idx, accumulator_op_indices=tuple(acc_ops), blocking_dim=dim_id, scale_role=role
        )
    )
    return result


def _classify_accumulator(
    da: DimAnalysis, graph: OpGraph, x_op_idx: int, x_outputs: set[str], consumer_idx: int, dim_id: str
) -> str | None:
    """Return the online-fusion role label for this (X, consumer) pair, or None."""
    cls_name = graph.op_classes[consumer_idx].NAME
    role = None
    if cls_name == "activation_reduce":
        role = _match_exp_bias(da, graph, x_outputs, consumer_idx, dim_id)
    elif cls_name == "nc_matmul":
        role = _match_matmul_through_scale(da, graph, x_op_idx, consumer_idx, dim_id)
    return role


def _match_exp_bias(da: DimAnalysis, graph: OpGraph, x_outputs: set[str], consumer_idx: int, dim_id: str) -> str | None:
    """True iff the consumer is ``activation_reduce(op='exp', bias=X, reduce_op='add')``."""
    kwargs = graph.op_all_kwargs[consumer_idx]
    inputs = graph.op_tensors[consumer_idx][0]
    bias_tensor = inputs.get("bias")
    role = None
    if (
        bias_tensor in x_outputs
        and _literal_op(kwargs.get("op")) == "exp"
        and _literal_op(kwargs.get("reduce_op")) == "add"
        and dim_id in da.op_blocking_dims(consumer_idx)
    ):
        role = "exp_bias"
    return role


def _match_matmul_through_scale(
    da: DimAnalysis, graph: OpGraph, x_op_idx: int, matmul_idx: int, dim_id: str
) -> str | None:
    """True iff the matmul's stationary lineage carries X as a per-partition scale."""
    inputs = graph.op_tensors[matmul_idx][0]
    stationary = inputs.get("stationary")
    role = None
    if stationary is not None and dim_id in da.op_blocking_dims(matmul_idx):
        role = _walk_back_for_separable(da, graph, stationary, x_op_idx)
    return role


def _walk_back_for_separable(da: DimAnalysis, graph: OpGraph, tensor_name: str, x_op_idx: int) -> str | None:
    """BFS backward through separable-preserving ops; report the first X-consuming role."""
    x_outputs = set(graph.op_tensors[x_op_idx][1])
    visited: set[str] = set()
    stack: list[str] = [tensor_name]
    role: str | None = None
    while stack and role is None:
        cur = stack.pop()
        if cur in visited:
            continue
        visited.add(cur)
        producer = graph.producer_op(cur)
        if producer is None:
            continue
        role = _separable_role_via(da, graph, producer, x_outputs)
        stack.extend(_separable_parents(graph, producer))
    return role


def _separable_role_via(da: DimAnalysis, graph: OpGraph, op_idx: int, x_outputs: set[str]) -> str | None:
    """Return a role label iff this op consumes an X output in a separable slot."""
    cls_name = graph.op_classes[op_idx].NAME
    kwargs = graph.op_all_kwargs[op_idx]
    inputs = graph.op_tensors[op_idx][0]
    role: str | None = None
    if cls_name == "tensor_scalar" and _literal_op(kwargs.get("op0")) == "multiply":
        operand = kwargs.get("operand0")
        if operand in da.tensors:
            role = _scale_to_x_label(graph, operand, x_outputs)
    elif cls_name in {"activation", "activation_reduce"} and _literal_op(kwargs.get("op")) == "exp":
        bias = inputs.get("bias")
        if bias in x_outputs:
            role = "exp_bias"
    return role


def _scale_to_x_label(graph: OpGraph, scale_tensor: str, x_outputs: set[str]) -> str | None:
    """Classify how ``scale_tensor`` reaches an X output; label by the first inverse activation."""
    role = None
    act_ops: list[str] = []
    cur: str | None = scale_tensor
    while cur is not None and cur not in x_outputs:
        producer = graph.producer_op(cur)
        if producer is None:
            cur = None
            break
        cls_name = graph.op_classes[producer].NAME
        kwargs = graph.op_all_kwargs[producer]
        if cls_name == "activation":
            act_ops.append(_literal_op(kwargs.get("op")) or "")
            cur = graph.op_tensors[producer][0]["data"]
        elif cls_name == "tensor_scalar":
            cur = graph.op_tensors[producer][0]["data"]
        else:
            cur = None
    if cur is not None:
        role = f"{act_ops[0]}_then_mul" if act_ops else "passthrough_mul"
    return role


def _separable_parents(graph: OpGraph, op_idx: int) -> list[str]:
    """Return parent tensors to keep walking through for separable-preserving ops."""
    cls_name = graph.op_classes[op_idx].NAME
    inputs = graph.op_tensors[op_idx][0]
    result: list[str] = []
    if cls_name == "nc_transpose" and "data" in inputs:
        result.append(inputs["data"])
    elif cls_name in {"tensor_scalar", "activation", "activation_reduce"} and "data" in inputs:
        result.append(inputs["data"])
    elif cls_name == "affine_select" and "on_true_tile" in inputs:
        result.append(inputs["on_true_tile"])
    return result


def _forward_reach(graph: OpGraph) -> list[set[int]]:
    """Forward transitive reach: ``reach[u]`` = ops downstream of ``u`` (including ``u``)."""
    n = len(graph.op_classes)
    adjacency: list[list[int]] = [[] for _ in range(n)]
    for producer, consumer, _tensor, _role in graph.edges:
        adjacency[producer].append(consumer)
    reach: list[set[int]] = [set() for _ in range(n)]
    for start in range(n):
        stack = [start]
        visited = reach[start]
        visited.add(start)
        while stack:
            node = stack.pop()
            for successor in adjacency[node]:
                if successor not in visited:
                    visited.add(successor)
                    stack.append(successor)
    return reach


def _literal_op(raw: str | None) -> str | None:
    """Strip quotes from a traced op-name kwarg, returning ``"exp"`` from ``"'exp'"``."""
    return None if raw is None else raw.strip("'\"")


def _merge_roles(a: str, b: str) -> str:
    """Join multiple roles on one X candidate into a stable compound label."""
    return a if a == b else f"{a}+{b}"
