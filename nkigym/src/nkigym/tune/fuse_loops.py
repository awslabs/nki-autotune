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
        """Return ``True`` when the atom can be applied at ``(path, boundary)``."""
        _ = op_graph
        siblings = _resolve_siblings(forest, self.path)
        if siblings is None:
            return False
        return _pair_is_fusable(siblings, self.boundary, self.dim_id)

    def apply(self, op_graph: OpGraph, forest: LoopForest) -> tuple[OpGraph, LoopForest]:
        """Merge the two adjacent siblings at ``(path, boundary)`` into one ``LoopNode``.

        The merged node's children = ``A.children ++ B.children`` in
        program order. ``op_graph`` is returned unchanged (structural
        rewrite).
        """
        new_forest = _rewrite_forest(forest, self.path, self.boundary, self.dim_id)
        return op_graph, new_forest


def enumerate_fusion_atoms(forest: LoopForest) -> list[FuseLoops]:
    """Return every legal :class:`FuseLoops` atom in ``forest``.

    Walks the forest recursively: at every children list (the forest
    itself plus every ``LoopNode.children``), enumerates adjacent pairs
    and emits one atom per fusable pair.
    """
    atoms: list[FuseLoops] = []
    _collect(forest, path=(), atoms=atoms)
    return atoms


def _collect(siblings: list[LoopNode | BodyLeaf], path: tuple[int, ...], atoms: list[FuseLoops]) -> None:
    """Recursive helper for :func:`enumerate_fusion_atoms`."""
    for i in range(len(siblings) - 1):
        if _pair_is_fusable(siblings, (i, i + 1), dim_id=None):
            a = siblings[i]
            assert isinstance(a, LoopNode)
            atoms.append(FuseLoops(path=path, boundary=(i, i + 1), dim_id=a.dim_id))
    for idx, child in enumerate(siblings):
        if isinstance(child, LoopNode):
            _collect(child.children, path=path + (idx,), atoms=atoms)


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
    """Check the three-field fusion rule on a specific adjacent pair.

    When ``dim_id`` is ``None`` the check is used by the enumerator and
    accepts any shared dim. When ``dim_id`` is specified (the
    ``is_legal`` path), the pair must also match it.
    """
    i, j = boundary
    if j != i + 1 or i < 0 or j >= len(siblings):
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


def _rewrite_forest(forest: LoopForest, path: tuple[int, ...], boundary: tuple[int, int], dim_id: str) -> LoopForest:
    """Return a new forest with the pair at ``(path, boundary)`` merged.

    Structural sharing: subtrees outside the rewrite site are passed
    through by reference, not deep-copied.
    """
    if not path:
        return _merge_pair(forest, boundary, dim_id)
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


def _merge_pair(
    siblings: list[LoopNode | BodyLeaf], boundary: tuple[int, int], dim_id: str
) -> list[LoopNode | BodyLeaf]:
    """Merge one adjacent pair inside ``siblings``.

    The merged loop inherits ``a``'s ``name`` — either side's name is
    valid (fusion requires identical dim_id + trip_count, so a caller
    that named them the same at build time would see the same name
    either way), and taking the left one is arbitrary but deterministic.
    """
    i, j = boundary
    a = siblings[i]
    b = siblings[j]
    assert isinstance(a, LoopNode) and isinstance(b, LoopNode)
    merged = LoopNode(
        dim_id=dim_id, trip_count=a.trip_count, role=AxisRole.PARALLEL, children=[*a.children, *b.children], name=a.name
    )
    return [*siblings[:i], merged, *siblings[j + 1 :]]
