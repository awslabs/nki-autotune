"""``SoftwarePipeline`` rewrite — set ``pipeline_depth`` on a target LoopNode.

Legality: depth must equal the producer-consumer chain length of the loop's
subtree. The renderer consumes ``LoopNode.pipeline_depth > 1`` to emit
prologue/body/epilogue.
"""

from dataclasses import dataclass, replace

from nkigym.codegen.ir import BodyLeaf, KernelModule, LoopNode, leaves_under, replace_at_path, resolve_node


@dataclass(frozen=True)
class SoftwarePipeline:
    """Set ``module.body[..loop_path..].pipeline_depth = depth``.

    Attributes:
        loop_path: Forest path to the target LoopNode.
        depth: Pipeline depth; must equal the subtree's chain length.
    """

    loop_path: tuple[int, ...]
    depth: int

    def is_legal(self, module: KernelModule) -> bool:
        """Return True iff target is a LoopNode and ``depth`` matches the chain length."""
        target = resolve_node(module.body, self.loop_path)
        result: bool
        if not isinstance(target, LoopNode):
            result = False
        elif self.depth < 1:
            result = False
        else:
            chain = _chain_length_of_subtree(target)
            result = self.depth == chain
        return result

    def apply(self, module: KernelModule) -> KernelModule:
        """Return a new module with the target LoopNode's ``pipeline_depth`` set."""
        target = resolve_node(module.body, self.loop_path)
        assert isinstance(target, LoopNode)
        new_target = LoopNode(
            dim_id=target.dim_id,
            trip_count=target.trip_count,
            role=target.role,
            children=list(target.children),
            reduce_op=target.reduce_op,
            name=target.name,
            pipeline_depth=self.depth,
        )
        new_body = replace_at_path(module.body, self.loop_path, new_target)
        return replace(module, body=new_body)


def _chain_length_of_subtree(loop: LoopNode) -> int:
    """Return the number of distinct BodyLeafs in ``loop``'s subtree.

    This approximates the producer-consumer chain length; for the shipped
    canonical forests (which consist of linear chains of load -> compute ->
    store phases), one leaf per stage matches the intended pipeline depth.
    """
    return sum(1 for _ in leaves_under(loop))


def enumerate_software_pipeline_atoms(module: KernelModule) -> list[SoftwarePipeline]:
    """Emit one SoftwarePipeline atom per LoopNode with chain length > 1.

    Already-pipelined loops (pipeline_depth equal to chain length) are
    skipped to avoid self-moves.
    """
    atoms: list[SoftwarePipeline] = []

    def walk(node: BodyLeaf | LoopNode, path: tuple[int, ...]) -> None:
        if isinstance(node, LoopNode):
            chain = _chain_length_of_subtree(node)
            if chain > 1 and node.pipeline_depth != chain:
                atoms.append(SoftwarePipeline(loop_path=path, depth=chain))
            for i, child in enumerate(node.children):
                walk(child, path + (i,))

    for i, root in enumerate(module.body):
        walk(root, (i,))
    return atoms
