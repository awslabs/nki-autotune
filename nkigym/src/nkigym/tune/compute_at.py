"""``ComputeAt`` rewrite — move a leaf under a target loop; regenerate inner loops.

Subsumes the old ``FuseLoops`` atom: moving a producer leaf under a
consumer's innermost loop is the producer-fusion case. Also supports
moving leaves to any target loop whose subtree contains a consumer.

Legality:

* Target loop must resolve to a :class:`LoopNode`.
* Target must NOT be an ancestor of the leaf's current position.
* Target's subtree must contain at least one consumer of the leaf being
  moved (dataflow constraint — checked by tree-walking
  :func:`leaves_under` and intersecting writes with reads).

After apply, all :class:`LoopNode` names are re-assigned canonically as
``i_<dim>_<ordinal>``.
"""

from dataclasses import dataclass, replace

from nkigym.codegen.ir import BodyLeaf, KernelModule, LoopNode, TreeIR, leaves_under, resolve_node


@dataclass(frozen=True)
class ComputeAt:
    """Move ``leaf_path`` under ``target_loop_path`` in the forest.

    Attributes:
        leaf_path: Path to the leaf to move.
        target_loop_path: Path to the LoopNode under which the leaf will
            be placed.
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
        elif _is_ancestor(self.target_loop_path, self.leaf_path):
            result = False
        else:
            result = any(_reads_leaf_writes(descendant, leaf) for descendant in leaves_under(target))
        return result

    def apply(self, module: KernelModule) -> KernelModule:
        """Remove ``leaf_path``; regenerate uncovered dims; insert under target.

        Canonical loop-var names are re-assigned across the whole forest
        after insertion.
        """
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
                f"ComputeAt.apply: target LoopNode was consumed by removal — "
                f"leaf_path={self.leaf_path}, target_loop_path={self.target_loop_path}"
            )
        ancestor_dims = _ancestor_dims(body_without, new_target_path)
        leaf_dims = list(leaf.dim_role.keys())
        needed = [d for d in leaf_dims if d not in ancestor_dims]
        regenerated = _wrap_leaf_with_dims(leaf, needed, module)
        new_body = _append_under(body_without, new_target_path, regenerated)
        new_body = _rename_canonical(new_body)
        return replace(module, body=new_body)


def enumerate_compute_at_atoms(module: KernelModule) -> list[ComputeAt]:
    """Emit every legal ``(leaf, target_loop)`` pair across the forest."""
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

    atoms: list[ComputeAt] = []
    for leaf_path, _leaf in leaves:
        for loop_path, _loop in loops:
            atom = ComputeAt(leaf_path=leaf_path, target_loop_path=loop_path)
            if atom.is_legal(module):
                atoms.append(atom)
    return atoms


def _reads_leaf_writes(maybe_consumer: BodyLeaf, producer: BodyLeaf) -> bool:
    """Return True iff any of ``producer``'s writes appears in ``maybe_consumer``'s reads."""
    return bool(set(maybe_consumer.reads.values()) & set(producer.writes))


def _is_ancestor(maybe_ancestor: tuple[int, ...], descendant: tuple[int, ...]) -> bool:
    """Return True iff ``maybe_ancestor`` is a strict prefix of ``descendant``."""
    return len(maybe_ancestor) < len(descendant) and descendant[: len(maybe_ancestor)] == maybe_ancestor


def _remove_at_path(body: TreeIR, path: tuple[int, ...]) -> TreeIR:
    """Return a new body with the node at ``path`` removed.

    Ancestor LoopNodes whose children become empty are pruned recursively
    so the parent tree does not retain empty loops after removal.
    """
    if not path:
        raise ValueError("_remove_at_path: path must be non-empty")
    if len(path) == 1:
        return [*body[: path[0]], *body[path[0] + 1 :]]
    idx, rest = path[0], path[1:]
    parent = body[idx]
    assert isinstance(parent, LoopNode)
    new_children = _remove_at_path(parent.children, rest)
    if not new_children:
        return [*body[:idx], *body[idx + 1 :]]
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


def _ancestor_dims(body: TreeIR, path: tuple[int, ...]) -> set[str]:
    """Return ``dim_id``s of every LoopNode along ``path`` from body root."""
    dims: set[str] = set()
    siblings: list[LoopNode | BodyLeaf] = list(body)
    for idx in path:
        if idx >= len(siblings):
            break
        node = siblings[idx]
        if isinstance(node, LoopNode):
            dims.add(node.dim_id)
            siblings = node.children
        else:
            break
    return dims


def _ancestor_trip_products(body: TreeIR, path: tuple[int, ...]) -> dict[str, int]:
    """Product of trip_counts per dim_id along ``path`` from body root.

    Walks the same path as :func:`_ancestor_dims` but multiplies each
    ancestor LoopNode's trip_count into a per-dim accumulator. Dims
    not on the path are absent from the result (callers treat absence
    as coverage of 1).
    """
    products: dict[str, int] = {}
    siblings: list[LoopNode | BodyLeaf] = list(body)
    for idx in path:
        if idx >= len(siblings):
            break
        node = siblings[idx]
        if isinstance(node, LoopNode):
            products[node.dim_id] = products.get(node.dim_id, 1) * node.trip_count
            siblings = node.children
        else:
            break
    return products


def _wrap_leaf_with_dims(leaf: BodyLeaf, dims: list[str], module: KernelModule) -> LoopNode | BodyLeaf:
    """Wrap ``leaf`` in the canonical 2N-per-dim chain over ``dims``.

    Returns ``leaf`` directly when ``dims`` is empty.
    """
    if not dims:
        return leaf
    node: LoopNode | BodyLeaf = leaf
    for d in reversed(dims):
        role = leaf.dim_role[d]
        num_t = module.dims[d].num_tiles
        tile_node = LoopNode(dim_id=d, trip_count=1, role=role, children=[node])
        block_node = LoopNode(dim_id=d, trip_count=num_t, role=role, children=[tile_node])
        node = block_node
    return node


def _append_under(body: TreeIR, target_path: tuple[int, ...], new_node: LoopNode | BodyLeaf) -> TreeIR:
    """Append ``new_node`` to the children of the LoopNode at ``target_path``.

    If ``target_path`` is empty, ``new_node`` is appended at the forest
    root instead.
    """
    if not target_path:
        return [*body, new_node]
    idx, rest = target_path[0], target_path[1:]
    parent = body[idx]
    assert isinstance(parent, LoopNode)
    if not rest:
        new_children = [*parent.children, new_node]
    else:
        new_children = _append_under(parent.children, rest, new_node)
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


def _find_node_path(body: TreeIR, target_id: int) -> tuple[int, ...] | None:
    """Return the path of the node whose ``id()`` matches ``target_id``, or None."""

    def walk(node: LoopNode | BodyLeaf, path: tuple[int, ...]) -> tuple[int, ...] | None:
        """Recurse into ``node`` searching for the target id."""
        result: tuple[int, ...] | None = None
        if id(node) == target_id:
            result = path
        elif isinstance(node, LoopNode):
            for i, c in enumerate(node.children):
                r = walk(c, path + (i,))
                if r is not None:
                    result = r
                    break
        return result

    found: tuple[int, ...] | None = None
    for i, root in enumerate(body):
        r = walk(root, (i,))
        if r is not None:
            found = r
            break
    return found


def _rename_canonical(body: TreeIR) -> TreeIR:
    """Re-assign ``i_<dim>_<ordinal>`` names across the tree.

    Ordinals are per-root; siblings share ancestor counts via the
    ``counts`` dict, which is restored after visiting each child so that
    sibling-to-sibling ordinals do not leak down to later subtrees.
    """

    def walk(node: LoopNode | BodyLeaf, counts: dict[str, int]) -> LoopNode | BodyLeaf:
        """Re-name ``node`` (if a LoopNode) and recurse into its children."""
        if isinstance(node, BodyLeaf):
            return node
        k = counts.get(node.dim_id, 0)
        new_name = f"i_{node.dim_id}_{k}"
        counts[node.dim_id] = k + 1
        new_children = [walk(c, counts) for c in node.children]
        counts[node.dim_id] = k
        return LoopNode(
            dim_id=node.dim_id,
            trip_count=node.trip_count,
            role=node.role,
            children=new_children,
            reduce_op=node.reduce_op,
            name=new_name,
            pipeline_depth=node.pipeline_depth,
        )

    return [walk(root, {}) for root in body]
