"""Op-level dependency graph over an ``OpGraph``.

Nodes are ``ParsedOp.idx`` values. Edges are inferred from tensor
identities: reads = ``operand_names`` restricted to ``OPERAND_AXES``
slots; writes = ``output_names``. Built once by
:func:`nkigym.codegen.graph.parse_and_resolve` and attached to the
resulting :class:`~nkigym.codegen.graph.OpGraph`.

Consumers of the dep graph compose the persisted op-level maps with
subtree helpers (``subtree_reads`` / ``subtree_writes`` /
``commutes``) that walk a :class:`~nkigym.codegen.loop_forest.LoopNode`
and aggregate per-op edges.
"""

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class DepGraph:
    """Op-level dependency DAG over an :class:`OpGraph`.

    Attributes:
        producer: Maps tensor name to the ``ParsedOp.idx`` that writes
            it. Parameter tensors (never written by any op) map to
            ``None``. Every tensor in ``op_graph.tensors`` appears as
            a key.
        consumers: Maps tensor name to a tuple of ``ParsedOp.idx``
            values that read the tensor, in source order.
        reads: Maps ``ParsedOp.idx`` to the frozenset of tensor names
            the op reads. Only operand slots whose key is in the op
            class's ``OPERAND_AXES`` are counted.
        writes: Maps ``ParsedOp.idx`` to the frozenset of tensor names
            the op writes (``ParsedOp.output_names``).
    """

    producer: dict[str, int | None]
    consumers: dict[str, tuple[int, ...]]
    reads: dict[int, frozenset[str]]
    writes: dict[int, frozenset[str]]


def build_dep_graph(ops: list["ParsedOp"], tensors: Mapping[str, object]) -> DepGraph:
    """Build the :class:`DepGraph` for a parsed op list.

    Walks ``ops`` once; for each op derives its read set from
    ``op_cls.OPERAND_AXES`` slots (restricted to slots actually bound
    in ``operand_names``) and its write set from ``output_names``.

    Args:
        ops: Parsed ops in source order.
        tensors: All tensor names present in the op graph. Used only
            to pre-populate ``producer`` with ``None`` for tensors
            never written (parameters).

    Returns:
        A fully resolved :class:`DepGraph`.

    Raises:
        ValueError: Two ops declare the same tensor name in their
            ``output_names`` — SSA violation.
    """
    reads: dict[int, frozenset[str]] = {}
    writes: dict[int, frozenset[str]] = {}
    producer: dict[str, int | None] = {name: None for name in tensors}
    consumers_mut: dict[str, list[int]] = {name: [] for name in tensors}
    for op in ops:
        op_reads: set[str] = set()
        for slot in op.op_cls.OPERAND_AXES:
            if slot in op.operand_names:
                op_reads.add(op.operand_names[slot])
        op_writes = set(op.output_names)
        reads[op.idx] = frozenset(op_reads)
        writes[op.idx] = frozenset(op_writes)
        for t in op_writes:
            if producer.get(t) is not None:
                raise ValueError(f"duplicate write on tensor {t!r}: ops {producer[t]} and {op.idx}")
            producer[t] = op.idx
        for t in op_reads:
            consumers_mut.setdefault(t, []).append(op.idx)
    consumers = {t: tuple(idxs) for t, idxs in consumers_mut.items()}
    return DepGraph(producer=producer, consumers=consumers, reads=reads, writes=writes)


def subtree_ops(node: "LoopNode | BodyLeaf") -> frozenset[int]:
    """Return the set of ``op_idx`` values appearing in ``BodyLeaf`` descendants of ``node``.

    A ``BodyLeaf`` contributes its own ``op_idx``; a ``LoopNode``
    contributes the union over its children.
    """
    collected: set[int] = set()
    _collect_op_idx(node, collected)
    return frozenset(collected)


def _collect_op_idx(node: "LoopNode | BodyLeaf", into: set[int]) -> None:
    """Recursive helper for :func:`subtree_ops`."""
    from nkigym.codegen.loop_forest import BodyLeaf

    if isinstance(node, BodyLeaf):
        into.add(node.op_idx)
        return
    for child in node.children:
        _collect_op_idx(child, into)


def subtree_reads(node: "LoopNode | BodyLeaf", dep: DepGraph) -> frozenset[str]:
    """Return the union of ``dep.reads`` over every ``BodyLeaf`` under ``node``."""
    result: set[str] = set()
    for op_idx in subtree_ops(node):
        result |= dep.reads.get(op_idx, frozenset())
    return frozenset(result)


def subtree_writes(node: "LoopNode | BodyLeaf", dep: DepGraph) -> frozenset[str]:
    """Return the union of ``dep.writes`` over every ``BodyLeaf`` under ``node``."""
    result: set[str] = set()
    for op_idx in subtree_ops(node):
        result |= dep.writes.get(op_idx, frozenset())
    return frozenset(result)


def commutes(a: "LoopNode | BodyLeaf", b: "LoopNode | BodyLeaf", dep: DepGraph) -> bool:
    """Return ``True`` iff subtrees ``a`` and ``b`` share no RAW, WAR, or WAW edge.

    Two subtrees commute when the respective read/write sets are
    pair-wise disjoint across all three conflict flavours:

    * RAW: ``writes(a) ∩ reads(b)``
    * WAR: ``reads(a) ∩ writes(b)``
    * WAW: ``writes(a) ∩ writes(b)``
    """
    wa = subtree_writes(a, dep)
    ra = subtree_reads(a, dep)
    wb = subtree_writes(b, dep)
    rb = subtree_reads(b, dep)
    return not (wa & rb) and not (ra & wb) and not (wa & wb)
