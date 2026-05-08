"""``HoistInvariant`` rewrite — LICM within a leaf's own tree.

Moves a leaf outward when the crossed loops' ``dim_id``s don't appear in
the leaf's metadata (``axis_map`` / ``dim_role``). Complement of
:class:`ComputeAt` — used when no consumer sits under the target loop,
so the move is pure loop-invariant code motion.

Legality:

* Leaf resolves to a ``BodyLeaf``.
* Target resolves to a ``LoopNode``.
* Target is a strict ancestor of the leaf's current position.
* No crossed loop (between target and leaf's current parent) has a
  ``dim_id`` the leaf references.
"""

from dataclasses import dataclass, replace

from nkigym.codegen.ir import BodyLeaf, KernelModule, LoopNode, TreeIR, resolve_node


@dataclass(frozen=True)
class HoistInvariant:
    """Move ``leaf_path`` outward to sit under ``target_loop_path``.

    Attributes:
        leaf_path: Path to the leaf to hoist.
        target_loop_path: Path to the strict-ancestor LoopNode under which
            the leaf will be placed.
    """

    leaf_path: tuple[int, ...]
    target_loop_path: tuple[int, ...]

    def is_legal(self, module: KernelModule) -> bool:
        """Check LICM preconditions."""
        result: bool
        leaf = resolve_node(module.body, self.leaf_path)
        if not isinstance(leaf, BodyLeaf):
            result = False
        else:
            target = resolve_node(module.body, self.target_loop_path)
            if not isinstance(target, LoopNode):
                result = False
            elif not _is_strict_ancestor(self.target_loop_path, self.leaf_path):
                result = False
            else:
                crossed = _dims_between(module.body, self.target_loop_path, self.leaf_path)
                leaf_dims = set(leaf.dim_role.keys())
                result = not (leaf_dims & crossed)
        return result

    def apply(self, module: KernelModule) -> KernelModule:
        """Remove leaf from current position; append under target; re-canonicalize names."""
        from nkigym.tune.compute_at import _append_under, _remove_at_path, _rename_canonical

        leaf = resolve_node(module.body, self.leaf_path)
        assert isinstance(leaf, BodyLeaf)
        body_without = _remove_at_path(module.body, self.leaf_path)
        new_body = _append_under(body_without, self.target_loop_path, leaf)
        new_body = _rename_canonical(new_body)
        return replace(module, body=new_body)


def _is_strict_ancestor(ancestor: tuple[int, ...], descendant: tuple[int, ...]) -> bool:
    """True iff ``ancestor`` is a strict prefix of ``descendant``."""
    return len(ancestor) < len(descendant) and descendant[: len(ancestor)] == ancestor


def _dims_between(body: TreeIR, ancestor_path: tuple[int, ...], descendant_path: tuple[int, ...]) -> set[str]:
    """Return ``dim_id``s of every LoopNode strictly between ``ancestor_path`` and the leaf's parent.

    Both endpoints excluded; interior LoopNodes along the path contribute
    their ``dim_id``s. The descendant itself (a leaf) is not considered.
    """
    dims: set[str] = set()
    siblings: list[LoopNode | BodyLeaf] = list(body)
    depth = 0
    for idx in descendant_path[:-1]:
        if idx >= len(siblings):
            break
        node = siblings[idx]
        if isinstance(node, LoopNode):
            if depth >= len(ancestor_path):
                dims.add(node.dim_id)
            siblings = node.children
        else:
            break
        depth += 1
    return dims


def enumerate_hoist_invariant_atoms(module: KernelModule) -> list[HoistInvariant]:
    """Emit one atom per (leaf, strict-ancestor loop) where all crossed dims are invariant."""
    atoms: list[HoistInvariant] = []
    leaves: list[tuple[tuple[int, ...], BodyLeaf]] = []

    def collect(node: LoopNode | BodyLeaf, path: tuple[int, ...]) -> None:
        """Walk ``node`` collecting paths to every BodyLeaf."""
        if isinstance(node, BodyLeaf):
            leaves.append((path, node))
        else:
            for i, c in enumerate(node.children):
                collect(c, path + (i,))

    for i, root in enumerate(module.body):
        collect(root, (i,))

    for leaf_path, _leaf in leaves:
        """Walk every strict ancestor path of this leaf; emit an atom per eligible target."""
        for k in range(1, len(leaf_path)):
            target_path = leaf_path[:k]
            atom = HoistInvariant(leaf_path=leaf_path, target_loop_path=target_path)
            if atom.is_legal(module):
                atoms.append(atom)
    return atoms
