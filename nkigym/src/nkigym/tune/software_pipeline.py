"""``SoftwarePipeline`` rewrite — set a LoopNode's ``pipeline_depth``.

Structural change lives entirely in rendering — the forest tree shape
is unchanged; only the target ``LoopNode``'s ``pipeline_depth`` field
is updated. At render time the node emits a prologue + skewed body +
epilogue sequence.
"""

from dataclasses import dataclass

from nkigym.codegen.graph import OpGraph
from nkigym.codegen.loop_forest import BodyLeaf, LoopForest, LoopNode, _resolve_node
from nkigym.codegen.render import assign_stages, required_tiles


@dataclass(frozen=True)
class SoftwarePipeline:
    """Set a LoopNode's ``pipeline_depth`` to ``depth``.

    Attributes:
        loop_path: Child indices from the forest root down to (and
            including) the target LoopNode.
        depth: New pipeline depth. ``1`` is un-pipelined; ``>=2``
            requires enough per-tensor ``total_slots`` in the subtree.
    """

    loop_path: tuple[int, ...]
    depth: int

    def is_legal(self, op_graph: OpGraph, forest: LoopForest) -> bool:
        """Return True when the atom's parameters identify a valid rewrite."""
        target = _resolve_node(forest, self.loop_path) if self.loop_path else None
        if not isinstance(target, LoopNode):
            return False
        if self.depth < 1:
            return False
        if self.depth == target.pipeline_depth:
            return False
        if target.trip_count < self.depth:
            return False
        if self.depth == 1:
            return True
        stages = assign_stages(target, op_graph.dep)
        if not stages:
            return False
        max_stage = max(stages.values())
        if self.depth != max_stage + 1:
            return False
        """Check per-tensor skew vs total_slots."""
        for tensor in op_graph.tensors.values():
            if target.dim_id not in tensor.dim_ids:
                continue
            skew = _tensor_skew_in_subtree(tensor.name, target, op_graph, stages)
            if skew == 0:
                continue
            r = required_tiles(tensor, target.dim_id, op_graph, forest)
            total = r * tensor.buffer_degree[target.dim_id]
            if total < skew + 1:
                return False
        return True

    def apply(self, op_graph: OpGraph, forest: LoopForest) -> tuple[OpGraph, LoopForest]:
        """Return a new forest with the target LoopNode's pipeline_depth set."""
        new_forest = _rewrite_forest(forest, self.loop_path, self.depth)
        return op_graph, new_forest


def _rewrite_forest(forest: LoopForest, path: tuple[int, ...], depth: int) -> LoopForest:
    """Return a new forest where the node at ``path`` has pipeline_depth = depth.

    Ancestors along the path are reconstructed; everything outside the
    edit site is passed through by reference.
    """
    if len(path) == 1:
        idx = path[0]
        target = forest[idx]
        assert isinstance(target, LoopNode)
        replacement = LoopNode(
            dim_id=target.dim_id,
            trip_count=target.trip_count,
            role=target.role,
            children=target.children,
            reduce_op=target.reduce_op,
            name=target.name,
            pipeline_depth=depth,
        )
        return [*forest[:idx], replacement, *forest[idx + 1 :]]
    idx, rest = path[0], path[1:]
    parent = forest[idx]
    assert isinstance(parent, LoopNode)
    new_children = _rewrite_forest(parent.children, rest, depth)
    new_parent = LoopNode(
        dim_id=parent.dim_id,
        trip_count=parent.trip_count,
        role=parent.role,
        children=new_children,
        reduce_op=parent.reduce_op,
        name=parent.name,
        pipeline_depth=parent.pipeline_depth,
    )
    return [*forest[:idx], new_parent, *forest[idx + 1 :]]


def _tensor_skew_in_subtree(
    tensor_name: str, loop_node: LoopNode, op_graph: OpGraph, stages: dict[tuple[int, str], int]
) -> int:
    """Return ``max(consumer_stages) - producer_stage`` for ``tensor_name`` in the subtree.

    Returns 0 when the tensor's producer is absent or has no consumers
    in the subtree.
    """
    producer = op_graph.dep.producer.get(tensor_name)
    consumers = op_graph.dep.consumers.get(tensor_name, ())
    leaves = _collect_leaves(loop_node)
    producer_stage: int | None = None
    consumer_stages: list[int] = []
    for leaf in leaves:
        s = stages.get((leaf.op_idx, leaf.phase))
        if s is None:
            continue
        if leaf.op_idx == producer:
            producer_stage = s
        if leaf.op_idx in consumers:
            consumer_stages.append(s)
    if producer_stage is None or not consumer_stages:
        return 0
    return max(consumer_stages) - producer_stage


def _collect_leaves(node: LoopNode | BodyLeaf) -> list[BodyLeaf]:
    """Gather every BodyLeaf under ``node`` in tree (DFS) order."""
    out: list[BodyLeaf] = []

    def walk(n: LoopNode | BodyLeaf) -> None:
        if isinstance(n, BodyLeaf):
            out.append(n)
        else:
            for c in n.children:
                walk(c)

    walk(node)
    return out


def enumerate_software_pipeline_atoms(op_graph: OpGraph, forest: LoopForest) -> list[SoftwarePipeline]:
    """Return every legal :class:`SoftwarePipeline` atom for the current state.

    Walks every LoopNode in the forest recursively. For each node with a
    non-empty stages table (has >=1 body leaf), yields atoms for the
    legal depth candidates — ``chain_len`` (full pipeline) and ``1``
    (reset, only when current depth > 1) — filtered by ``is_legal``.
    """
    atoms: list[SoftwarePipeline] = []

    def visit(node: LoopNode | BodyLeaf, path: tuple[int, ...]) -> None:
        if isinstance(node, BodyLeaf):
            return
        stages = assign_stages(node, op_graph.dep)
        if stages:
            chain_len = max(stages.values()) + 1
            """Only chain_len itself is legal (full coverage) and chain_len=1 is
            a no-op. So atom-worthy depths are either exactly chain_len (pipeline)
            or 1 (reset — legal only when current pipeline_depth > 1)."""
            candidate_depths: set[int] = set()
            if chain_len >= 2:
                candidate_depths.add(chain_len)
            if node.pipeline_depth > 1:
                candidate_depths.add(1)
            for depth in candidate_depths:
                if depth == node.pipeline_depth:
                    continue
                atom = SoftwarePipeline(loop_path=path, depth=depth)
                if atom.is_legal(op_graph, forest):
                    atoms.append(atom)
        for idx, child in enumerate(node.children):
            visit(child, path + (idx,))

    for i, root in enumerate(forest):
        visit(root, (i,))
    return atoms
