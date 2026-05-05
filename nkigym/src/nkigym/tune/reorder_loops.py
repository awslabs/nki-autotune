"""``ReorderLoops`` rewrite — local perfect-nest loop interchange.

Swaps a ``LoopNode`` with its unique ``LoopNode`` child. Classical
rectangular polyhedral interchange: legality depends only on the
swap pair (not the surrounding forest), so the atom composes cleanly
with ``FuseLoops`` and the future hoist primitive.

The atom's identity is ``(path, outer_dim, inner_dim)``:

* ``path`` — tuple of child indices from the forest root down to the
  outer ``LoopNode`` of the swap pair.
* ``outer_dim`` / ``inner_dim`` — dim ids the two loops iterate;
  guard the atom against stale bindings when the caller stores a
  rewrite list across intervening rewrites.

Self-inverse: applying the same atom twice restores the original
forest (structurally). The ``tune`` stage uses forest-state hashing
(:func:`nkigym.codegen.loop_forest.hash_forest`) to break cycles in
the random-draw loop.
"""

from collections.abc import Callable
from dataclasses import dataclass

from nkigym.codegen.graph import OpGraph
from nkigym.codegen.loop_forest import BodyLeaf, LoopForest, LoopNode, _resolve_node
from nkigym.ops.base import AxisRole


def _roles_commute(a: LoopNode, b: LoopNode) -> bool:
    """Return True iff swapping loops ``a`` and ``b`` preserves semantics.

    * Any SEQUENTIAL involvement → False (non-associative state).
    * PAR×PAR → True.
    * Mixed PAR / ACC → True (ordering changes accumulator footprint,
      not correctness).
    * ACC×ACC → True iff both have the same non-None reduce_op.
    """
    result: bool
    if a.role == AxisRole.SEQUENTIAL or b.role == AxisRole.SEQUENTIAL:
        result = False
    elif a.role == AxisRole.ACCUMULATION and b.role == AxisRole.ACCUMULATION:
        result = a.reduce_op is not None and a.reduce_op == b.reduce_op
    else:
        result = True
    return result


@dataclass(frozen=True)
class ReorderLoops:
    """Swap a LoopNode with its unique LoopNode child.

    Attributes:
        path: Child indices from the forest root down to (and including)
            the outer LoopNode of the swap pair. A length-1 path
            ``(idx,)`` targets ``forest[idx]``; a length-2 path
            ``(idx, j)`` targets ``forest[idx].children[j]``. Empty
            path is invalid (:meth:`is_legal` returns False).
        outer_dim: Dim id the outer loop iterates; guards against stale
            atoms after unrelated rewrites.
        inner_dim: Dim id the inner loop iterates; guards against stale
            atoms after unrelated rewrites.
    """

    path: tuple[int, ...]
    outer_dim: str
    inner_dim: str

    def is_legal(self, op_graph: OpGraph, forest: LoopForest) -> bool:
        """Return True when the swap pair exists, is locally perfect, and role-commutes."""
        _ = op_graph
        result: bool
        outer = _resolve_node(forest, self.path)
        if not isinstance(outer, LoopNode):
            result = False
        elif outer.dim_id != self.outer_dim:
            result = False
        elif len(outer.children) != 1:
            result = False
        else:
            inner = outer.children[0]
            if not isinstance(inner, LoopNode):
                result = False
            elif inner.dim_id != self.inner_dim:
                result = False
            elif not _roles_commute(outer, inner):
                result = False
            else:
                result = True
        return result

    def apply(self, op_graph: OpGraph, forest: LoopForest) -> tuple[OpGraph, LoopForest]:
        """Swap outer and inner; grandchildren subtree is passed by reference."""
        new_forest = _rewrite_at_path(forest, self.path, _swap_pair)
        return op_graph, new_forest


def _swap_pair(outer: LoopNode) -> LoopNode:
    """Swap ``outer`` with its unique LoopNode child.

    outer -> inner -> grandchildren  becomes
    inner -> outer -> grandchildren.

    Each node carries its ``name`` verbatim through the swap — the loop
    variable the outer node printed before still names that same loop
    after the swap, even though the loop now sits at a deeper tree
    position. Loop identity is preserved.
    """
    assert len(outer.children) == 1
    inner = outer.children[0]
    assert isinstance(inner, LoopNode)
    new_outer = LoopNode(
        dim_id=outer.dim_id,
        trip_count=outer.trip_count,
        role=outer.role,
        children=list(inner.children),
        reduce_op=outer.reduce_op,
        name=outer.name,
    )
    return LoopNode(
        dim_id=inner.dim_id,
        trip_count=inner.trip_count,
        role=inner.role,
        children=[new_outer],
        reduce_op=inner.reduce_op,
        name=inner.name,
    )


def _rewrite_at_path(
    forest: LoopForest, path: tuple[int, ...], transform: Callable[[LoopNode], LoopNode]
) -> LoopForest:
    """Return a new forest with ``transform`` applied to the node at ``path``.

    Ancestors along ``path`` are reconstructed; everything outside the
    edit site is passed through by reference.
    """
    if len(path) == 1:
        idx = path[0]
        target = forest[idx]
        assert isinstance(target, LoopNode)
        replacement = transform(target)
        return [*forest[:idx], replacement, *forest[idx + 1 :]]
    idx, rest = path[0], path[1:]
    parent = forest[idx]
    assert isinstance(parent, LoopNode)
    new_children = _rewrite_at_path(parent.children, rest, transform)
    new_parent = LoopNode(
        dim_id=parent.dim_id,
        trip_count=parent.trip_count,
        role=parent.role,
        children=new_children,
        reduce_op=parent.reduce_op,
        name=parent.name,
    )
    return [*forest[:idx], new_parent, *forest[idx + 1 :]]


def enumerate_reorder_atoms(forest: LoopForest) -> list[ReorderLoops]:
    """Return every legal :class:`ReorderLoops` atom in ``forest``.

    Walks the forest recursively: at every LoopNode whose single child
    is another LoopNode and whose role commutes with that child,
    emits one atom.
    """
    atoms: list[ReorderLoops] = []
    _collect_reorder(forest, path=(), atoms=atoms)
    return atoms


def _collect_reorder(siblings: list[LoopNode | BodyLeaf], path: tuple[int, ...], atoms: list[ReorderLoops]) -> None:
    """Recursive helper for :func:`enumerate_reorder_atoms`."""
    for idx, node in enumerate(siblings):
        if isinstance(node, LoopNode):
            if len(node.children) == 1 and isinstance(node.children[0], LoopNode):
                inner = node.children[0]
                if _roles_commute(node, inner):
                    atoms.append(ReorderLoops(path=path + (idx,), outer_dim=node.dim_id, inner_dim=inner.dim_id))
            _collect_reorder(node.children, path=path + (idx,), atoms=atoms)
