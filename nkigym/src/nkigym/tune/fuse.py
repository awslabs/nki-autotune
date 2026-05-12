"""``Fuse`` atom — TVM-style ``sch.fuse(outer, inner)``.

Collapses a perfectly-nested ForNode pair ``ForNode(v_outer) >
ForNode(v_inner) > ...`` into a single ``ForNode(v_fused)`` whose
iter-var binds a synthetic dim ``<outer_dim>_x_<inner_dim>`` with
extent ``outer.extent * inner.extent``.

Legality:

- The outer ForNode has exactly one child, which is a ForNode binding
  the inner iter var. No other ForNodes / SBlocks interleave.
- SEQUENTIAL never fuses — sequential state would be reordered.
- All other role pairs are legal; the fused role takes the lattice
  max ``PAR ⊂ SEQ ⊂ ACC``. Only PAR/PAR and ACC/ACC actually compose,
  because SEQUENTIAL is rejected above.

Apply:

1. Retire ``v_outer`` and ``v_inner``.
2. Allocate ``v_fused`` with dim_id ``<outer_dim>_x_<inner_dim>`` and
   extent ``outer.extent * inner.extent``.
3. Register a synthetic ``DimInfo`` for the new dim so downstream
   passes (e.g. :func:`place_buffers`) can look it up.
4. Record the outer/inner decomposition in
   ``module.fused_iter_var_map`` so the renderer emits outer references
   as ``(fused_name // inner_extent)`` and inner as ``(fused_name %
   inner_extent)``.
5. Replace the two-ForNode chain with a single ``ForNode(v_fused)``
   that keeps the inner's children.
6. Rewrite every SBlock.iter_vars: consecutive ``(v_outer, v_inner)``
   collapses to ``(v_fused,)`` in the same list position.

``BufferAccess.pattern`` is untouched — the retired iter-var ids still
appear in ``iter_var_coeffs``, and the renderer's ``_resolve_iv_name``
helper decomposes them via ``module.fused_iter_var_map``.

See ``docs/superpowers/specs/2026-05-10-iter-var-refactor-design.md`` 4.3.
"""

from dataclasses import dataclass, replace

from nkigym.codegen.ir import DimInfo, ForNode, IterVar, KernelModule, SBlock, TreeIR, replace_at_path
from nkigym.ops.base import AxisRole
from nkigym.tune import AtomLegalityError
from nkigym.tune.split import _abstract_axis_for, _op_cls_for_block

_ROLE_RANK: dict[AxisRole, int] = {AxisRole.PARALLEL: 0, AxisRole.SEQUENTIAL: 1, AxisRole.ACCUMULATION: 2}
"""Role lattice ``PAR ⊂ SEQ ⊂ ACC``. Fusion takes the max rank."""


@dataclass(frozen=True)
class Fuse:
    """Collapse a parent-child ForNode pair into a single ForNode.

    Attributes:
        outer_iter_var_id: Target outer ForNode's iter-var id.
        inner_iter_var_id: Target inner ForNode's iter-var id. The
            inner ForNode must be the outer's sole child.
    """

    outer_iter_var_id: int
    inner_iter_var_id: int

    def is_legal(self, module: KernelModule) -> bool:
        """Structural + role preconditions + MIN/MAX tile check when fuse touches innermost.

        If the inner loop is the innermost tile loop for any descendant
        leaf (i.e. its iter-var is the last one for its dim in the leaf's
        ``iter_vars``), the fused extent must satisfy ``MIN <=
        fused_extent <= MAX`` for that leaf's op on the corresponding
        abstract axis. When the fuse crosses different dim ids the
        resulting synthetic dim has no op-axis mapping; in that case the
        MIN/MAX check is skipped.
        """
        pair = _find_pair(module, self.outer_iter_var_id, self.inner_iter_var_id)
        result = False
        if pair is not None:
            outer, inner, _path = pair
            role_a = outer.iter_var.role
            role_b = inner.iter_var.role
            if role_a != AxisRole.SEQUENTIAL and role_b != AxisRole.SEQUENTIAL:
                result = self._check_min_max(outer, inner)
        return result

    def _check_min_max(self, outer: ForNode, inner: ForNode) -> bool:
        """Validate fused extent against MIN/MAX bounds when fuse touches an innermost.

        Cases:
            - ``d_o == d_i`` and ``inner`` is the last iter-var for this
              dim in a descendant leaf's ``iter_vars``: post-fuse the new
              innermost for this leaf becomes the fused iter-var, so its
              extent must be in ``[MIN, MAX]`` for the leaf's op on that
              abstract axis.
            - ``d_o != d_i`` (cross-dim fuse): the resulting iter-var
              carries a synthetic dim id not present in any op's
              ``axis_map``. The leaf no longer has individual iter-vars
              for the retired dims; MIN/MAX checks against the synthetic
              dim are not meaningful, so skip.
        """
        d_o = outer.iter_var.dim_id
        d_i = inner.iter_var.dim_id
        legal = True
        if d_o == d_i:
            fused_dim = d_o
            fused_extent = outer.iter_var.extent * inner.iter_var.extent
            inner_id = inner.iter_var.var_id
            legal = _walk_leaves_for_min_max(inner.children, fused_dim, inner_id, fused_extent)
        return legal

    def apply(self, module: KernelModule) -> KernelModule:
        """Execute the fuse.

        Raises:
            AtomLegalityError: ``is_legal`` returns False against ``module``.
        """
        if not self.is_legal(module):
            raise AtomLegalityError(f"Fuse.apply: illegal {self!r}")
        pair = _find_pair(module, self.outer_iter_var_id, self.inner_iter_var_id)
        assert pair is not None
        outer_node, inner_node, outer_path = pair
        v_outer = outer_node.iter_var
        v_inner = inner_node.iter_var

        fused_dim = f"{v_outer.dim_id}_x_{v_inner.dim_id}"
        fused_extent = v_outer.extent * v_inner.extent
        fused_role = max(v_outer.role, v_inner.role, key=_ROLE_RANK.__getitem__)
        v_fused = module.allocate_iter_var(dim_id=fused_dim, extent=fused_extent, role=fused_role)

        """Register synthetic DimInfo. Under per-op tiling, trip count
        lives on v_fused.extent; the synthetic DimInfo only needs its
        dim_id and total_size. total_size = fused_extent because the
        fused loop iterates fused_extent times (outer.extent * inner.extent).
        Buffer sizing consults writer access extents, not this field."""
        module.dims[fused_dim] = DimInfo(dim_id=fused_dim, total_size=fused_extent)

        """Record the decomposition so the renderer can emit
        ``(fused // inner_extent)`` and ``(fused % inner_extent)``."""
        module.fused_iter_var_map[v_outer.var_id] = (v_fused.var_id, v_inner.extent, True)
        module.fused_iter_var_map[v_inner.var_id] = (v_fused.var_id, v_inner.extent, False)

        new_body = _collapse_iter_var_lists(module.body, v_outer.var_id, v_inner.var_id, v_fused)
        new_fornode = ForNode(
            iter_var=v_fused, children=list(inner_node.children), name=None, annotations=dict(outer_node.annotations)
        )
        new_body = replace_at_path(new_body, outer_path, new_fornode)
        return replace(module, body=new_body)


def _find_pair(module: KernelModule, outer_id: int, inner_id: int) -> tuple[ForNode, ForNode, tuple[int, ...]] | None:
    """Return ``(outer_node, inner_node, outer_path)`` for the legal pair.

    Requires the outer ForNode to have exactly one child ForNode whose
    iter_var id matches ``inner_id``. Returns ``None`` if no such pair
    exists.
    """

    def walk(node: ForNode | SBlock, path: tuple[int, ...]) -> tuple[ForNode, ForNode, tuple[int, ...]] | None:
        result: tuple[ForNode, ForNode, tuple[int, ...]] | None = None
        if isinstance(node, ForNode):
            if (
                node.iter_var.var_id == outer_id
                and len(node.children) == 1
                and isinstance(node.children[0], ForNode)
                and node.children[0].iter_var.var_id == inner_id
            ):
                result = (node, node.children[0], path)
            else:
                for i, child in enumerate(node.children):
                    found = walk(child, path + (i,))
                    if found is not None:
                        result = found
                        break
        return result

    discovered: tuple[ForNode, ForNode, tuple[int, ...]] | None = None
    for i, root in enumerate(module.body):
        found = walk(root, (i,))
        if found is not None:
            discovered = found
            break
    return discovered


def _walk_leaves_for_min_max(
    children: list[ForNode | SBlock], fused_dim: str, inner_id: int, fused_extent: int
) -> bool:
    """Walk a ForNode's descendants; for each SBlock whose innermost
    iter-var for ``fused_dim`` is the fuse's inner iter-var, check the
    fused extent against ``MIN``/``MAX`` on the op's abstract axis.
    """
    legal = True

    def visit(node: ForNode | SBlock) -> None:
        nonlocal legal
        if not legal:
            return
        if isinstance(node, SBlock):
            ivs_for_dim = [iv for iv in node.iter_vars if iv.dim_id == fused_dim]
            if not ivs_for_dim or ivs_for_dim[-1].var_id != inner_id:
                return
            abstract_axis = _abstract_axis_for(node, fused_dim)
            op_cls = _op_cls_for_block(node)
            if abstract_axis is None or op_cls is None:
                return
            min_tile = op_cls.MIN_TILE_SIZE.get(abstract_axis)
            max_tile = op_cls.MAX_TILE_SIZE.get(abstract_axis)
            if min_tile is not None and fused_extent < min_tile:
                legal = False
            if max_tile is not None and fused_extent > max_tile:
                legal = False
        else:
            for c in node.children:
                visit(c)

    for c in children:
        visit(c)
    return legal


def _collapse_iter_var_lists(body: TreeIR, outer_id: int, inner_id: int, v_fused: IterVar) -> TreeIR:
    """Rewrite every SBlock.iter_vars: replace consecutive ``(outer_id,
    inner_id)`` with ``(v_fused,)`` in the same list position.

    The canonical builder places the two ids adjacent in reducer order
    (outer first), and Split preserves that adjacency. If the pair is
    not adjacent in a block's list, that block is left unchanged —
    ``is_legal`` already guarantees the enclosing ForNodes are adjacent
    in the tree so such a case implies the block belongs to a sibling
    subtree that never saw both ids.
    """

    def rewrite_block(block: SBlock) -> SBlock:
        new_ivs: list[IterVar] = []
        i = 0
        changed = False
        while i < len(block.iter_vars):
            iv = block.iter_vars[i]
            if iv.var_id == outer_id and i + 1 < len(block.iter_vars) and block.iter_vars[i + 1].var_id == inner_id:
                new_ivs.append(v_fused)
                i += 2
                changed = True
            else:
                new_ivs.append(iv)
                i += 1
        result = block
        if changed:
            result = SBlock(
                iter_vars=new_ivs,
                reads=block.reads,
                writes=block.writes,
                reads_writes=block.reads_writes,
                body=block.body,
                annotations=dict(block.annotations),
            )
        return result

    def rewrite_node(node: ForNode | SBlock) -> ForNode | SBlock:
        if isinstance(node, SBlock):
            return rewrite_block(node)
        return ForNode(
            iter_var=node.iter_var,
            children=[rewrite_node(c) for c in node.children],
            name=node.name,
            annotations=dict(node.annotations),
        )

    return [rewrite_node(n) for n in body]


def enumerate_fuse_atoms(module: KernelModule) -> list[Fuse]:
    """Emit one :class:`Fuse` atom per legal perfect-nest pair in the forest."""
    atoms: list[Fuse] = []

    def walk(node: ForNode | SBlock) -> None:
        if isinstance(node, ForNode):
            if len(node.children) == 1 and isinstance(node.children[0], ForNode):
                outer_id = node.iter_var.var_id
                inner_id = node.children[0].iter_var.var_id
                atom = Fuse(outer_iter_var_id=outer_id, inner_iter_var_id=inner_id)
                if atom.is_legal(module):
                    atoms.append(atom)
            for child in node.children:
                walk(child)

    for root in module.body:
        walk(root)
    return atoms
