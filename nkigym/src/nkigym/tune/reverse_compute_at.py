"""``ReverseComputeAt`` rewrite - mirror of :class:`ComputeAt`.

Moves a consumer leaf under a loop whose subtree contains one of its
producers. Legality is the dataflow dual: the target loop must contain at
least one leaf that writes a tensor the moving leaf reads.

Reuses helpers from :mod:`nkigym.tune.compute_at` via function-level
imports to keep the atoms independently importable while sharing
implementation.
"""

from dataclasses import dataclass, replace

from nkigym.codegen.ir import BodyLeaf, KernelModule, LoopNode, leaves_under, resolve_node


@dataclass(frozen=True)
class ReverseComputeAt:
    """Move a consumer leaf under a loop in a producer's scope.

    Attributes:
        leaf_path: Path to the consumer leaf to move.
        target_loop_path: Path to the LoopNode under which the leaf will be
            placed; target's subtree must contain at least one producer of
            the leaf being moved.
    """

    leaf_path: tuple[int, ...]
    target_loop_path: tuple[int, ...]

    def is_legal(self, module: KernelModule) -> bool:
        """Check dataflow and structural preconditions."""
        result: bool
        leaf = resolve_node(module.body, self.leaf_path)
        target = resolve_node(module.body, self.target_loop_path)
        if not isinstance(leaf, BodyLeaf):
            result = False
        elif not isinstance(target, LoopNode):
            result = False
        else:
            from nkigym.tune.compute_at import _is_ancestor

            if _is_ancestor(self.target_loop_path, self.leaf_path):
                result = False
            else:
                producer_found = any(
                    bool(set(leaf.reads.values()) & set(descendant.writes)) for descendant in leaves_under(target)
                )
                result = producer_found
        return result

    def apply(self, module: KernelModule) -> KernelModule:
        """Remove leaf; regenerate uncovered dims under target; insert; canonical-rename."""
        from nkigym.tune.compute_at import (
            _ancestor_dims,
            _append_under,
            _find_node_path,
            _remove_at_path,
            _rename_canonical,
            _wrap_leaf_with_dims,
        )

        leaf = resolve_node(module.body, self.leaf_path)
        assert isinstance(leaf, BodyLeaf)
        target_node = resolve_node(module.body, self.target_loop_path)
        assert isinstance(target_node, LoopNode)
        body_without = _remove_at_path(module.body, self.leaf_path)
        """Pruning below the target never shifts the target's id(), because
        ``_remove_at_path`` only rebuilds ancestors of the removed node. Sibling
        subtrees pass by reference. Find the target's new path in body_without
        by walking for the same id() — this is O(tree) but safe across pruning."""
        new_target_path = _find_node_path(body_without, id(target_node))
        if new_target_path is None:
            raise ValueError(
                f"ReverseComputeAt.apply: target LoopNode was consumed by removal — "
                f"leaf_path={self.leaf_path}, target_loop_path={self.target_loop_path}"
            )
        ancestor_dims = _ancestor_dims(body_without, new_target_path)
        leaf_dims = list(leaf.dim_role.keys())
        needed = [d for d in leaf_dims if d not in ancestor_dims]
        regenerated = _wrap_leaf_with_dims(leaf, needed, module)
        new_body = _append_under(body_without, new_target_path, regenerated)
        new_body = _rename_canonical(new_body)
        return replace(module, body=new_body)


def enumerate_reverse_compute_at_atoms(module: KernelModule) -> list[ReverseComputeAt]:
    """Emit every legal ``(consumer_leaf, target_loop)`` pair."""
    leaves: list[tuple[tuple[int, ...], BodyLeaf]] = []
    loops: list[tuple[tuple[int, ...], LoopNode]] = []

    def collect_leaves(node: LoopNode | BodyLeaf, path: tuple[int, ...]) -> None:
        """Walk ``node`` collecting paths to every BodyLeaf."""
        if isinstance(node, BodyLeaf):
            leaves.append((path, node))
        else:
            for i, c in enumerate(node.children):
                collect_leaves(c, path + (i,))

    def collect_loops(node: LoopNode | BodyLeaf, path: tuple[int, ...]) -> None:
        """Walk ``node`` collecting paths to every LoopNode."""
        if isinstance(node, LoopNode):
            loops.append((path, node))
            for i, c in enumerate(node.children):
                collect_loops(c, path + (i,))

    for i, root in enumerate(module.body):
        collect_leaves(root, (i,))
        collect_loops(root, (i,))

    atoms: list[ReverseComputeAt] = []
    for leaf_path, _leaf in leaves:
        for loop_path, _loop in loops:
            atom = ReverseComputeAt(leaf_path=leaf_path, target_loop_path=loop_path)
            if atom.is_legal(module):
                atoms.append(atom)
    return atoms
