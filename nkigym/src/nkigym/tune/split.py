"""``Split`` rewrite — split a loop into outer x inner; emit tail pair if needed.

When ``factor`` does not divide ``trip_count``, two sibling nests are emitted
at the split site: the "full" pair (``floor(N/factor)`` outer iters of
``factor`` inner trip each) and the "tail" pair (1 outer iter of
``N % factor`` inner trip). Matches TVM's ``LoopPartition`` semantics —
no predication, separate loops.
"""

from copy import deepcopy
from dataclasses import dataclass, replace

from nkigym.codegen.ir import BodyLeaf, KernelModule, LoopNode, TreeIR, resolve_node


@dataclass(frozen=True)
class Split:
    """Split the target LoopNode into outer x inner by ``factor``.

    Attributes:
        loop_path: Forest path to the target LoopNode.
        factor: Inner trip count (must be >= 1).
    """

    loop_path: tuple[int, ...]
    factor: int

    def is_legal(self, module: KernelModule) -> bool:
        """Target must be a LoopNode; ``factor`` must be positive."""
        result: bool
        if self.factor < 1:
            result = False
        else:
            target = resolve_node(module.body, self.loop_path)
            result = isinstance(target, LoopNode)
        return result

    def apply(self, module: KernelModule) -> KernelModule:
        """Replace target with one or two sibling (outer, inner) nests.

        The new outer/inner LoopNodes carry ``name=None`` at construction,
        and Split's ``deepcopy`` of the target's children preserves their
        existing canonical names. Without a canonical-rename pass, the
        renderer's ``len(existing)``-based name fallback would collide
        with those preserved child names (see followup doc, Bug #3).
        Canonical-rename is therefore applied across the whole body
        before returning, matching the post-apply contract of
        ``ComputeAt``, ``ReverseComputeAt``, ``HoistInvariant``, and
        ``DecomposeReduction``.
        """
        from nkigym.tune.compute_at import _rename_canonical

        target = resolve_node(module.body, self.loop_path)
        assert isinstance(target, LoopNode)
        n = target.trip_count
        f = self.factor
        full_iters = n // f
        tail_iters = n % f
        replacement: list[LoopNode | BodyLeaf] = []
        if full_iters > 0:
            replacement.append(_make_split_pair(target, outer_trip=full_iters, inner_trip=f))
        if tail_iters > 0:
            replacement.append(_make_split_pair(target, outer_trip=1, inner_trip=tail_iters))
        new_body = _replace_with_siblings(module.body, self.loop_path, replacement)
        new_body = _rename_canonical(new_body)
        return replace(module, body=new_body)


def _make_split_pair(target: LoopNode, outer_trip: int, inner_trip: int) -> LoopNode:
    """Build outer LoopNode wrapping one inner LoopNode; children copied from target."""
    inner = LoopNode(
        dim_id=target.dim_id,
        trip_count=inner_trip,
        role=target.role,
        children=deepcopy(target.children),
        reduce_op=target.reduce_op,
        pipeline_depth=1,
    )
    outer = LoopNode(
        dim_id=target.dim_id,
        trip_count=outer_trip,
        role=target.role,
        children=[inner],
        reduce_op=target.reduce_op,
        pipeline_depth=target.pipeline_depth,
    )
    return outer


def _replace_with_siblings(body: TreeIR, path: tuple[int, ...], replacement: list[LoopNode | BodyLeaf]) -> TreeIR:
    """Replace the node at ``path`` with ``replacement`` (one or more siblings)."""
    if not path:
        raise ValueError("_replace_with_siblings: path must be non-empty")
    if len(path) == 1:
        idx = path[0]
        return [*body[:idx], *replacement, *body[idx + 1 :]]
    idx, rest = path[0], path[1:]
    parent = body[idx]
    assert isinstance(parent, LoopNode)
    new_children = _replace_with_siblings(parent.children, rest, replacement)
    new_parent = LoopNode(
        dim_id=parent.dim_id,
        trip_count=parent.trip_count,
        role=parent.role,
        children=new_children,
        reduce_op=parent.reduce_op,
        name=parent.name,
        pipeline_depth=parent.pipeline_depth,
    )
    return [*body[:idx], new_parent, *body[idx + 1 :]]


def enumerate_split_atoms(module: KernelModule) -> list[Split]:
    """Emit one atom per (LoopNode, divisor factor > 1 and < trip_count).

    Only emits divisors to keep the search space tractable; non-divisor
    splits are still legal at apply-time but are not proposed by default.
    """
    atoms: list[Split] = []

    def walk(node: LoopNode | BodyLeaf, path: tuple[int, ...]) -> None:
        if isinstance(node, LoopNode):
            n = node.trip_count
            for f in range(2, n):
                if n % f == 0:
                    atoms.append(Split(loop_path=path, factor=f))
            for i, c in enumerate(node.children):
                walk(c, path + (i,))

    for i, root in enumerate(module.body):
        walk(root, (i,))
    return atoms
