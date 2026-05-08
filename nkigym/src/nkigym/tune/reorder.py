"""``Reorder`` rewrite — adjacent loop interchange with subtree-purity legality.

Legality rules:

* ``inner_path == outer_path + (0,)`` and outer has exactly one child loop
  (perfect-nest shape).
* Role pair rules:
    - PAR x PAR: legal.
    - ACC x ACC same ``reduce_op``: legal.
    - PAR x ACC: legal iff ACC's subtree contains no leaf whose write region
      is indexed by PAR's ``dim_id`` (subtree-purity check via
      ``BodyLeaf.axis_map`` / ``dim_role``). After ``DecomposeReduction``
      strips init/drain leaves, update trees pass.
    - SEQ involvement: illegal.

This is the replacement for the old ``ReorderLoops`` atom; the old module
stays until Task 20 cleanup.
"""

from dataclasses import dataclass, replace

from nkigym.codegen.ir import BodyLeaf, KernelModule, LoopNode, leaves_under, replace_at_path, resolve_node
from nkigym.ops.base import AxisRole


@dataclass(frozen=True)
class Reorder:
    """Swap an outer LoopNode with its unique LoopNode child.

    Attributes:
        outer_path: Path to the outer LoopNode in ``module.body``.
        inner_path: Must equal ``outer_path + (0,)``; guards against stale
            atoms across rewrites.
    """

    outer_path: tuple[int, ...]
    inner_path: tuple[int, ...]

    def is_legal(self, module: KernelModule) -> bool:
        """Return True when the pair is a perfect nest and roles commute."""
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
                else:
                    result = _roles_commute(outer, inner)
        return result

    def apply(self, module: KernelModule) -> KernelModule:
        """Swap outer and inner; grandchildren subtree passes by reference."""
        outer = resolve_node(module.body, self.outer_path)
        assert isinstance(outer, LoopNode)
        inner = outer.children[0]
        assert isinstance(inner, LoopNode)
        new_outer = LoopNode(
            dim_id=outer.dim_id,
            trip_count=outer.trip_count,
            role=outer.role,
            children=list(inner.children),
            reduce_op=outer.reduce_op,
            name=outer.name,
            pipeline_depth=outer.pipeline_depth,
        )
        new_inner = LoopNode(
            dim_id=inner.dim_id,
            trip_count=inner.trip_count,
            role=inner.role,
            children=[new_outer],
            reduce_op=inner.reduce_op,
            name=inner.name,
            pipeline_depth=inner.pipeline_depth,
        )
        new_body = replace_at_path(module.body, self.outer_path, new_inner)
        return replace(module, body=new_body)


def _roles_commute(a: LoopNode, b: LoopNode) -> bool:
    """Return True iff swapping ``a`` and ``b`` preserves semantics.

    SEQUENTIAL never commutes. PAR x PAR always commutes. ACC x ACC commutes
    iff both share a non-None ``reduce_op``. PAR x ACC commutes iff the ACC
    subtree is leaf-pure w.r.t. the PAR dim.
    """
    result: bool
    if a.role == AxisRole.SEQUENTIAL or b.role == AxisRole.SEQUENTIAL:
        result = False
    elif a.role == AxisRole.PARALLEL and b.role == AxisRole.PARALLEL:
        result = True
    elif a.role == AxisRole.ACCUMULATION and b.role == AxisRole.ACCUMULATION:
        result = a.reduce_op is not None and a.reduce_op == b.reduce_op
    else:
        par_dim = a.dim_id if a.role == AxisRole.PARALLEL else b.dim_id
        acc = a if a.role == AxisRole.ACCUMULATION else b
        result = _subtree_pure_wrt_dim(acc, par_dim)
    return result


def _subtree_pure_wrt_dim(node: LoopNode | BodyLeaf, par_dim: str) -> bool:
    """Return True iff no leaf under ``node`` writes a tensor indexed by ``par_dim`` as PARALLEL.

    Uses each leaf's ``axis_map`` and op-local ``dim_role`` to determine
    whether a write position's index depends on the outer PAR loop.
    """
    for leaf in leaves_under(node):
        if not leaf.writes:
            continue
        for concrete_dim in leaf.axis_map.values():
            if concrete_dim == par_dim and leaf.dim_role.get(concrete_dim) == AxisRole.PARALLEL:
                return False
    return True


def enumerate_reorder_atoms(module: KernelModule) -> list[Reorder]:
    """Every legal adjacent-swap atom in the forest."""
    atoms: list[Reorder] = []

    def walk(node: BodyLeaf | LoopNode, path: tuple[int, ...]) -> None:
        if not isinstance(node, LoopNode):
            return
        if len(node.children) == 1 and isinstance(node.children[0], LoopNode):
            atom = Reorder(outer_path=path, inner_path=path + (0,))
            if atom.is_legal(module):
                atoms.append(atom)
        for i, child in enumerate(node.children):
            walk(child, path + (i,))

    for i, root in enumerate(module.body):
        walk(root, (i,))
    return atoms
