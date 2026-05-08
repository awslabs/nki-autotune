"""``Split`` rewrite — split a loop into outer x inner by a divisor factor.

``factor`` must divide ``trip_count``; non-divisor factors are rejected at
``is_legal`` and raise :class:`AtomLegalityError` at ``apply``. The older
tail-sibling emission (two sibling nests for non-divisor factors) is gone —
it violated the 1N canonical form invariant; see spec
``docs/superpowers/specs/2026-05-08-canonical-1N-and-computeat-partial-coverage-design.md``.
"""

from copy import deepcopy
from dataclasses import dataclass, replace

from nkigym.codegen.ir import BodyLeaf, KernelModule, LoopNode, TreeIR, resolve_node
from nkigym.tune import AtomLegalityError


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
        """Target must be a LoopNode; ``factor`` must be a positive divisor of ``trip_count``."""
        result: bool
        if self.factor < 1:
            result = False
        else:
            target = resolve_node(module.body, self.loop_path)
            if not isinstance(target, LoopNode):
                result = False
            else:
                result = target.trip_count % self.factor == 0
        return result

    def apply(self, module: KernelModule) -> KernelModule:
        """Replace target with a single outer × inner pair.

        Rejects non-divisor factors via :class:`AtomLegalityError` — the
        old tail-sibling emission path is gone. Tail-siblings violated
        the 1N invariant (sibling subtrees with the same dim and
        mismatched trips make downstream atoms brittle; see spec
        `docs/superpowers/specs/2026-05-08-canonical-1N-and-computeat-partial-coverage-design.md`).
        Canonical-rename runs across the whole body after replacement,
        matching the post-apply contract of ``ComputeAt`` et al.
        """
        from nkigym.tune.compute_at import _rename_canonical

        target = resolve_node(module.body, self.loop_path)
        assert isinstance(target, LoopNode)
        n = target.trip_count
        f = self.factor
        if n % f != 0:
            raise AtomLegalityError(f"Split.apply: factor {f} does not divide trip_count {n} at {self.loop_path}")
        outer_trip = n // f
        replacement = [_make_split_pair(target, outer_trip=outer_trip, inner_trip=f)]
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

    Only divisor factors are legal under the current contract; non-divisor
    factors are rejected at both ``is_legal`` and ``apply`` time (see
    :class:`Split`).
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
