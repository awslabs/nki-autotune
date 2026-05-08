"""``Fuse`` rewrite - collapse a perfectly nested loop pair into one.

The fused loop has ``trip = outer.trip_count * inner.trip_count`` and a
synthetic ``dim_id = "<outer_dim>_x_<inner_dim>"``. The renderer is
responsible for recovering the two original loop variables via div/mod
at emit time via the ``FusedDim`` metadata stashed in the fused loop's
``dim_id`` naming scheme.

Legality:

* ``inner_path == outer_path + (0,)``, outer has exactly one child loop.
* SEQUENTIAL never fuses.
* PAR x PAR: fused role is PARALLEL.
* ACC x ACC: legal iff same non-None ``reduce_op``; fused role retains that op.
* Mixed PAR x ACC: illegal (iter-space collapse requires identical semantics).
"""

from dataclasses import dataclass, replace

from nkigym.codegen.ir import BodyLeaf, KernelModule, LoopNode, replace_at_path, resolve_node
from nkigym.ops.base import AxisRole


@dataclass(frozen=True)
class Fuse:
    """Collapse a perfect-nest outer+inner LoopNode pair.

    Attributes:
        outer_path: Forest path to the outer LoopNode.
        inner_path: Must equal ``outer_path + (0,)`` (guards against stale atoms).
    """

    outer_path: tuple[int, ...]
    inner_path: tuple[int, ...]

    def is_legal(self, module: KernelModule) -> bool:
        """Perfect nest + role compatibility."""
        result: bool
        if self.inner_path != self.outer_path + (0,):
            result = False
        else:
            outer = resolve_node(module.body, self.outer_path)
            if not isinstance(outer, LoopNode) or len(outer.children) != 1:
                result = False
            else:
                inner = outer.children[0]
                if not isinstance(inner, LoopNode):
                    result = False
                elif outer.role == AxisRole.SEQUENTIAL or inner.role == AxisRole.SEQUENTIAL:
                    result = False
                elif outer.role != inner.role:
                    result = False
                elif outer.role == AxisRole.ACCUMULATION:
                    result = outer.reduce_op is not None and outer.reduce_op == inner.reduce_op
                else:
                    result = True
        return result

    def apply(self, module: KernelModule) -> KernelModule:
        """Replace the outer+inner pair with a single fused LoopNode."""
        outer = resolve_node(module.body, self.outer_path)
        assert isinstance(outer, LoopNode)
        inner = outer.children[0]
        assert isinstance(inner, LoopNode)
        fused = LoopNode(
            dim_id=f"{outer.dim_id}_x_{inner.dim_id}",
            trip_count=outer.trip_count * inner.trip_count,
            role=outer.role,
            children=list(inner.children),
            reduce_op=outer.reduce_op,
            name=None,
            pipeline_depth=outer.pipeline_depth,
        )
        new_body = replace_at_path(module.body, self.outer_path, fused)
        return replace(module, body=new_body)


def enumerate_fuse_atoms(module: KernelModule) -> list[Fuse]:
    """Emit one Fuse atom per perfect-nest pair whose roles commute."""
    atoms: list[Fuse] = []

    def walk(node: LoopNode | BodyLeaf, path: tuple[int, ...]) -> None:
        if isinstance(node, LoopNode):
            if len(node.children) == 1 and isinstance(node.children[0], LoopNode):
                atom = Fuse(outer_path=path, inner_path=path + (0,))
                if atom.is_legal(module):
                    atoms.append(atom)
            for i, c in enumerate(node.children):
                walk(c, path + (i,))

    for i, root in enumerate(module.body):
        walk(root, (i,))
    return atoms
