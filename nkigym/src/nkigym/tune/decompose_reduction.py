"""``DecomposeReduction`` rewrite — fission a reducer op's subtree.

A canonical reducer op (matmul, activation_reduce) currently lives as a
single subtree rooted at a spatial-loop nest; within that subtree, the
init phase, the reduction-axis chain containing the update phase, and the
drain phase sit as siblings. This atom replaces that single subtree with
three separate sibling trees:

* init tree — the spatial loops wrapping only the init phase leaf.
* update tree — the spatial loops + the reduction-axis chain wrapping
  only the update phase leaf.
* drain tree — the spatial loops wrapping only the drain phase leaf.

After fission, the update tree's reduction axis can be freely reordered
via :class:`Reorder` (the subtree becomes leaf-pure). Accumulator-buffer
widening is left to the renderer's LCA pass.

Matches TVM's ``DecomposeReduction`` semantics.
"""

from dataclasses import dataclass, replace

from nkigym.codegen.ir import BodyLeaf, KernelModule, LoopNode, leaves_under, resolve_node
from nkigym.ops.base import AxisRole


@dataclass(frozen=True)
class DecomposeReduction:
    """Fission a reducer op into init/update/drain sibling trees.

    Attributes:
        leaf_path: Path to the reducer's update-phase leaf (``compute`` for
            matmul, ``reduce_step`` for activation_reduce).
        target_loop_path: Path to the outer LoopNode whose subtree will be
            replaced with the three fissioned trees. Must be a strict
            ancestor of ``leaf_path`` and must not sit inside the
            reduction axis (legality rejects ACCUMULATION-role targets).
    """

    leaf_path: tuple[int, ...]
    target_loop_path: tuple[int, ...]

    def is_legal(self, module: KernelModule) -> bool:
        """Check structural preconditions.

        Rejects non-reducer leaves, non-loop targets, ACCUMULATION-role
        targets (would sit inside the reduction axis), and any target
        that is not a strict ancestor of ``leaf_path``.
        """
        result: bool
        leaf = resolve_node(module.body, self.leaf_path)
        target = resolve_node(module.body, self.target_loop_path)
        if not isinstance(leaf, BodyLeaf):
            result = False
        elif leaf.phase not in ("compute", "reduce_step"):
            result = False
        elif not isinstance(target, LoopNode):
            result = False
        elif target.role == AxisRole.ACCUMULATION:
            result = False
        elif not _is_strict_ancestor(self.target_loop_path, self.leaf_path):
            result = False
        else:
            result = True
        return result

    def apply(self, module: KernelModule) -> KernelModule:
        """Split the target subtree into init / update / drain siblings.

        Each sibling clones the target's loop structure; init and drain drop
        any ACCUMULATION-role loops (reduction axes don't iterate around a
        bulk init/drain), while the update tree keeps them.
        """
        from nkigym.tune.compute_at import _rename_canonical
        from nkigym.tune.split import _replace_with_siblings

        target = resolve_node(module.body, self.target_loop_path)
        assert isinstance(target, LoopNode)
        init_leaf, update_leaf, drain_leaf = _find_phase_leaves(target)
        if update_leaf is None:
            return module
        trees: list[LoopNode | BodyLeaf] = []
        init_tree = _rebuild_with_leaf(target, init_leaf, exclude_reducing=True)
        if init_tree is not None:
            trees.append(init_tree)
        update_tree = _rebuild_with_leaf(target, update_leaf, exclude_reducing=False)
        if update_tree is not None:
            trees.append(update_tree)
        drain_tree = _rebuild_with_leaf(target, drain_leaf, exclude_reducing=True)
        if drain_tree is not None:
            trees.append(drain_tree)
        new_body = _replace_with_siblings(module.body, self.target_loop_path, trees)
        new_body = _rename_canonical(new_body)
        return replace(module, body=new_body)


def _is_strict_ancestor(ancestor: tuple[int, ...], descendant: tuple[int, ...]) -> bool:
    """Return True iff ``ancestor`` is a strict prefix of ``descendant``."""
    return len(ancestor) < len(descendant) and descendant[: len(ancestor)] == ancestor


def _find_phase_leaves(target: LoopNode) -> tuple[BodyLeaf | None, BodyLeaf | None, BodyLeaf | None]:
    """Return ``(init, update, drain)`` leaves under ``target``.

    Phase name mapping:

    - matmul: ``psum_init``, ``compute``, ``drain``.
    - activation_reduce: ``reduce_step`` (update) and ``reduce_close``
      (drain only; no init phase exists — ``None`` returned).
    """
    init_leaf: BodyLeaf | None = None
    update_leaf: BodyLeaf | None = None
    drain_leaf: BodyLeaf | None = None
    for leaf in leaves_under(target):
        if leaf.phase == "psum_init":
            init_leaf = leaf
        elif leaf.phase in ("compute", "reduce_step"):
            update_leaf = leaf
        elif leaf.phase in ("drain", "reduce_close"):
            drain_leaf = leaf
    return init_leaf, update_leaf, drain_leaf


def _rebuild_with_leaf(target: LoopNode, leaf: BodyLeaf | None, exclude_reducing: bool) -> LoopNode | BodyLeaf | None:
    """Clone ``target``'s nest, retaining only ``leaf`` at the deepest point.

    If ``exclude_reducing`` is True, any ACCUMULATION-role LoopNode along
    the walk is dropped (so init / drain blocks don't iterate the reduction
    axis).
    """
    if leaf is None:
        return None

    def walk(node: LoopNode | BodyLeaf) -> LoopNode | BodyLeaf | None:
        """Recursive clone filter, keeping only the target leaf."""
        if isinstance(node, BodyLeaf):
            return leaf if node is leaf else None
        if exclude_reducing and node.role == AxisRole.ACCUMULATION:
            """Drop this LoopNode; recurse into children and surface the
            first non-None inner result."""
            inner_result: LoopNode | BodyLeaf | None = None
            for c in node.children:
                inner = walk(c)
                if inner is not None:
                    inner_result = inner
                    break
            return inner_result
        new_children: list[LoopNode | BodyLeaf] = []
        for c in node.children:
            r = walk(c)
            if r is not None:
                new_children.append(r)
        if not new_children:
            return None
        return LoopNode(
            dim_id=node.dim_id,
            trip_count=node.trip_count,
            role=node.role,
            children=new_children,
            reduce_op=node.reduce_op,
            name=node.name,
            pipeline_depth=node.pipeline_depth,
        )

    return walk(target)


def enumerate_decompose_reduction_atoms(module: KernelModule) -> list[DecomposeReduction]:
    """Emit one atom per (reducer leaf, eligible ancestor LoopNode)."""
    atoms: list[DecomposeReduction] = []

    def walk(node: LoopNode | BodyLeaf, path: tuple[int, ...]) -> None:
        """Walk ``node``, emitting atoms at each reducer leaf we encounter."""
        if isinstance(node, BodyLeaf):
            if node.phase in ("compute", "reduce_step"):
                for k in range(1, len(path)):
                    target_path = path[:k]
                    atom = DecomposeReduction(leaf_path=path, target_loop_path=target_path)
                    if atom.is_legal(module):
                        atoms.append(atom)
        else:
            for i, c in enumerate(node.children):
                walk(c, path + (i,))

    for i, root in enumerate(module.body):
        walk(root, (i,))
    return atoms
