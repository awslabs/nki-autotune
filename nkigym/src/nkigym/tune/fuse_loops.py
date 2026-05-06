"""``FuseLoops`` rewrite — fuse two adjacent sibling loops anywhere in the forest.

Generalises the outermost-only atom from the original design §5 to work
at any depth in the ``LoopForest`` tree. The legality rule is unchanged
— two adjacent ``LoopNode`` siblings with the same ``dim_id``, same
``trip_count``, and both ``PARALLEL``-role — but it now applies to
sibling pairs inside any ``LoopNode``'s children list, not just to
forest-root pairs.

The atom's identity is ``(path, boundary, dim_id)``:

* ``path`` — tuple of child indices from the forest root to the parent
  whose children contain the fusion pair. Empty tuple ``()`` means the
  forest itself (recovers the original outer-root behaviour).
* ``boundary`` — ``(i, i+1)`` adjacent-pair index inside that parent's
  children list.
* ``dim_id`` — the concrete dim the two siblings iterate; guards the
  atom against stale bindings when the caller stores a rewrite list.

One fused pair per atom — atoms compose by re-enumerating against the
current forest after each apply.
"""

from dataclasses import dataclass

from nkigym.codegen.dep_graph import commutes
from nkigym.codegen.graph import OpGraph
from nkigym.codegen.loop_forest import BodyLeaf, LoopForest, LoopNode
from nkigym.ops.base import AxisRole


@dataclass(frozen=True)
class FuseLoops:
    """Fuse two adjacent sibling ``LoopNode``s.

    Attributes:
        path: Child indices from the forest root down to the parent
            whose children list contains the fusion pair. Empty tuple
            targets the forest itself.
        boundary: ``(i, i+1)`` position pair inside the targeted parent's
            children list (or the forest, when ``path`` is empty).
        dim_id: Concrete dim both targeted siblings must iterate.
    """

    path: tuple[int, ...]
    boundary: tuple[int, int]
    dim_id: str

    def is_legal(self, op_graph: OpGraph, forest: LoopForest) -> bool:
        """Return ``True`` when the atom can be applied at ``(path, boundary)``.

        Three layers:

        1. Resolve ``path``; bail if stale.
        2. Three-field match on the endpoints: same ``dim_id``, same
           ``trip_count``, both ``PARALLEL``.
        3. Topological adjacency: every sibling strictly between ``i``
           and ``j`` must commute with BOTH endpoints — so it can be
           pushed to the left of the producer without breaking any
           RAW / WAR / WAW edge. Skipped when ``j == i + 1``.

        Layer 3 consults ``op_graph.dep``; when ``j == i + 1`` the
        topological check is vacuous and ``op_graph`` may be ``None``
        (supported for hand-built test forests).
        """
        siblings = _resolve_siblings(forest, self.path)
        if siblings is None:
            return False
        if not _pair_is_fusable(siblings, self.boundary, self.dim_id):
            return False
        i, j = self.boundary
        if j == i + 1:
            return True
        return _intervening_siblings_commute(op_graph, siblings, i, j)

    def apply(self, op_graph: OpGraph, forest: LoopForest) -> tuple[OpGraph, LoopForest]:
        """Fuse the pair at ``(path, boundary)``; consumer absorbs producer.

        Siblings strictly between ``i`` and ``j`` slide to the left of
        ``i``'s original position, keeping their relative order. The
        fused ``LoopNode`` lands at ``j``'s original slot. Its children
        are ``producer.children ++ consumer.children`` — producer body
        first, preserving the RAW edge. The fused loop inherits the
        consumer's ``name``. ``op_graph`` passes through unchanged.
        """
        new_forest = _rewrite_forest(forest, self.path, self.boundary, self.dim_id)
        return op_graph, new_forest


def enumerate_fusion_atoms(op_graph: OpGraph | None, forest: LoopForest) -> list[FuseLoops]:
    """Return every legal :class:`FuseLoops` atom in ``forest``.

    Walks the forest recursively: at every children list (the forest
    itself plus every ``LoopNode.children``), enumerates every pair
    ``(i, j)`` with ``0 ≤ i < j < len(siblings)`` and emits one atom
    per pair that passes both the three-field match and the
    topological-adjacency check.

    Args:
        op_graph: Used for the topological-adjacency check when a pair
            has intervening siblings. May be ``None`` on hand-built
            test forests that only exercise literal-adjacent pairs;
            in that case a pair with ``j > i + 1`` is conservatively
            rejected.
        forest: The forest to enumerate over.

    Returns:
        List of atoms in depth-first order.
    """
    atoms: list[FuseLoops] = []
    _collect(op_graph, forest, path=(), atoms=atoms)
    return atoms


def _collect(
    op_graph: OpGraph | None, siblings: list[LoopNode | BodyLeaf], path: tuple[int, ...], atoms: list[FuseLoops]
) -> None:
    """Recursive helper for :func:`enumerate_fusion_atoms`."""
    n = len(siblings)
    for i in range(n):
        for j in range(i + 1, n):
            if not _pair_is_fusable(siblings, (i, j), dim_id=None):
                continue
            if j > i + 1:
                if op_graph is None:
                    continue
                if not _intervening_siblings_commute(op_graph, siblings, i, j):
                    continue
            a = siblings[i]
            assert isinstance(a, LoopNode)
            atoms.append(FuseLoops(path=path, boundary=(i, j), dim_id=a.dim_id))
    for idx, child in enumerate(siblings):
        if isinstance(child, LoopNode):
            _collect(op_graph, child.children, path=path + (idx,), atoms=atoms)


def _resolve_siblings(forest: LoopForest, path: tuple[int, ...]) -> list[LoopNode | BodyLeaf] | None:
    """Walk ``path`` from the forest root. Returns the children list at that depth.

    Returns ``None`` when the path is invalid (index out of range or
    traversing through a ``BodyLeaf``).
    """
    siblings: list[LoopNode | BodyLeaf] = forest
    for idx in path:
        if idx < 0 or idx >= len(siblings):
            return None
        node = siblings[idx]
        if not isinstance(node, LoopNode):
            return None
        siblings = node.children
    return siblings


def _pair_is_fusable(siblings: list[LoopNode | BodyLeaf], boundary: tuple[int, int], dim_id: str | None) -> bool:
    """Check the three-field fusion rule on a specific pair.

    When ``dim_id`` is ``None`` the check accepts any shared dim
    (enumerator path). When ``dim_id`` is specified (``is_legal``
    path) the pair must also match it. The pair is identified by
    ``boundary = (i, j)`` with ``i < j``; non-adjacent pairs are
    accepted here — the topological-adjacency check lives in
    :func:`_intervening_siblings_commute`.
    """
    i, j = boundary
    if not (0 <= i < j < len(siblings)):
        return False
    a = siblings[i]
    b = siblings[j]
    if not isinstance(a, LoopNode) or not isinstance(b, LoopNode):
        return False
    if dim_id is not None and (a.dim_id != dim_id or b.dim_id != dim_id):
        return False
    return (
        a.dim_id == b.dim_id
        and a.role == AxisRole.PARALLEL
        and b.role == AxisRole.PARALLEL
        and a.trip_count == b.trip_count
    )


def _intervening_siblings_commute(op_graph: OpGraph, siblings: list[LoopNode | BodyLeaf], i: int, j: int) -> bool:
    """Return ``True`` when every sibling strictly between ``i`` and ``j``
    commutes with both endpoints.

    Consults ``op_graph.dep`` to form RAW/WAR/WAW judgments at
    subtree granularity. Callers must have already verified the
    three-field match on the endpoints; this helper is only concerned
    with the movement legality of the intervening siblings.
    """
    dep = op_graph.dep
    producer = siblings[i]
    consumer = siblings[j]
    for k in range(i + 1, j):
        survivor = siblings[k]
        if not commutes(survivor, producer, dep):
            return False
        if not commutes(survivor, consumer, dep):
            return False
    return True


def _rewrite_forest(forest: LoopForest, path: tuple[int, ...], boundary: tuple[int, int], dim_id: str) -> LoopForest:
    """Return a new forest with the pair at ``(path, boundary)`` merged.

    Structural sharing: subtrees outside the rewrite site are passed
    through by reference, not deep-copied.
    """
    if not path:
        return _apply_fuse_in_siblings(forest, boundary, dim_id)
    idx, rest = path[0], path[1:]
    parent = forest[idx]
    assert isinstance(parent, LoopNode)
    new_children = _rewrite_forest(parent.children, rest, boundary, dim_id)
    new_parent = LoopNode(
        dim_id=parent.dim_id,
        trip_count=parent.trip_count,
        role=parent.role,
        children=new_children,
        reduce_op=parent.reduce_op,
        name=parent.name,
    )
    return [*forest[:idx], new_parent, *forest[idx + 1 :]]


def _apply_fuse_in_siblings(
    siblings: list[LoopNode | BodyLeaf], boundary: tuple[int, int], dim_id: str
) -> list[LoopNode | BodyLeaf]:
    """Merge one pair inside ``siblings`` — producer absorbed by consumer.

    Layout after apply:
        siblings[:i] ++ survivors ++ [fused] ++ siblings[j+1:]

    where ``survivors = siblings[i+1 : j]`` keeps the original
    relative order. The fused ``LoopNode`` lands at ``j``'s original
    slot; its children are
    ``producer.children ++ consumer.children``.

    For ``j == i + 1`` ``survivors`` is empty and the output matches
    the literal-adjacent case byte-for-byte.
    """
    i, j = boundary
    producer = siblings[i]
    consumer = siblings[j]
    assert isinstance(producer, LoopNode) and isinstance(consumer, LoopNode)
    survivors = list(siblings[i + 1 : j])
    merged = LoopNode(
        dim_id=dim_id,
        trip_count=consumer.trip_count,
        role=AxisRole.PARALLEL,
        children=[*producer.children, *consumer.children],
        name=consumer.name,
    )
    return [*siblings[:i], *survivors, merged, *siblings[j + 1 :]]
