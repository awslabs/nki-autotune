"""``Split`` atom - partition a ForNode's iter var into outer inner.

TVM-style ``sch.split(loop, factor)``:

- Retire the target ForNode's ``iter_var`` ``v``.
- Allocate ``v_outer`` (extent = v.extent / factor) + ``v_inner``
  (extent = factor); inherit ``dim_id`` + ``role``.
- Replace the target ForNode with nested ForNode(v_outer)
  ForNode(v_inner) <original children>.
- Rewrite every BufferAccess.pattern that references ``v.var_id``:
  ``v v_outer * factor + v_inner`` via AccessRange.iter_var_coeffs.
- Update every SBlock.iter_vars list: replace ``v`` with (v_outer,
  v_inner) in the same position.
- Reject non-divisor factors via AtomLegalityError.

See ``docs/superpowers/specs/2026-05-10-iter-var-refactor-design.md`` 4.1.
"""

from dataclasses import dataclass, replace

from nkigym.codegen.ir import (
    AccessRange,
    BufferAccess,
    ForNode,
    IterVar,
    KernelModule,
    SBlock,
    TreeIR,
    replace_at_path,
    resolve_node,
)
from nkigym.tune import AtomLegalityError


@dataclass(frozen=True)
class Split:
    """Partition a ForNode's iter var into outer inner.

    Attributes:
        loop_path: Path to the target ForNode in ``module.body``.
        factor: Inner extent. Must satisfy ``1 < factor < target.extent``
            and divide ``target.extent``.
    """

    loop_path: tuple[int, ...]
    factor: int

    def is_legal(self, module: KernelModule) -> bool:
        """Structural + divisibility preconditions."""
        target = resolve_node(module.body, self.loop_path)
        result = False
        if isinstance(target, ForNode):
            iv = target.iter_var
            if 1 < self.factor < iv.extent and iv.extent % self.factor == 0:
                result = True
        return result

    def apply(self, module: KernelModule) -> KernelModule:
        """Execute the split; return a new module with rewired body + new iter vars.

        Raises:
            AtomLegalityError: ``is_legal`` returns False.
        """
        if not self.is_legal(module):
            raise AtomLegalityError(f"Split.apply: illegal {self!r}")
        target = resolve_node(module.body, self.loop_path)
        assert isinstance(target, ForNode)
        old_iv = target.iter_var
        outer_extent = old_iv.extent // self.factor
        inner_extent = self.factor
        v_outer = module.allocate_iter_var(old_iv.dim_id, outer_extent, old_iv.role)
        v_inner = module.allocate_iter_var(old_iv.dim_id, inner_extent, old_iv.role)

        """Rewrite all BufferAccess.pattern entries referencing old_iv.var_id."""
        new_body = _rewrite_patterns(module.body, old_iv.var_id, v_outer.var_id, v_inner.var_id, inner_extent)

        """Update every SBlock.iter_vars list: replace old_iv with (v_outer, v_inner)."""
        new_body = _update_iter_var_lists(new_body, old_iv.var_id, v_outer, v_inner)

        """Replace the target ForNode with nested ForNode(v_outer) ForNode(v_inner)
        original children. Preserve the target's annotations on the outer."""
        target_now = resolve_node(new_body, self.loop_path)
        assert isinstance(target_now, ForNode)
        new_inner = ForNode(iter_var=v_inner, children=list(target_now.children), name=None, annotations={})
        new_outer = ForNode(iter_var=v_outer, children=[new_inner], name=None, annotations=dict(target_now.annotations))
        new_body = replace_at_path(new_body, self.loop_path, new_outer)
        return replace(module, body=new_body)


def _rewrite_patterns(body: TreeIR, old_id: int, outer_id: int, inner_id: int, inner_extent: int) -> TreeIR:
    """Rewrite every BufferAccess.pattern entry referencing ``old_id`` to
    ``outer_id * inner_extent + inner_id``."""

    def rewrite_access(acc: BufferAccess) -> BufferAccess:
        if old_id not in acc.iter_var_ids:
            return acc
        new_ids_list = [inner_id if i == old_id else i for i in acc.iter_var_ids]
        if outer_id not in new_ids_list:
            new_ids_list.append(outer_id)
        new_pattern: list[AccessRange] = []
        for ar in acc.pattern:
            coeffs = _coeffs_rewrite(ar, old_id, outer_id, inner_id, inner_extent)
            new_pattern.append(AccessRange.make(coeffs, ar.const_offset, ar.extent))
        return BufferAccess(tensor_name=acc.tensor_name, iter_var_ids=tuple(new_ids_list), pattern=tuple(new_pattern))

    def rewrite_block(block: SBlock) -> SBlock:
        return SBlock(
            iter_vars=block.iter_vars,
            reads={k: rewrite_access(v) for k, v in block.reads.items()},
            writes={k: rewrite_access(v) for k, v in block.writes.items()},
            reads_writes={k: rewrite_access(v) for k, v in block.reads_writes.items()},
            body=block.body,
            annotations=dict(block.annotations),
        )

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


def _coeffs_rewrite(ar: AccessRange, old_id: int, outer_id: int, inner_id: int, inner_extent: int) -> dict[int, int]:
    """Return the new coeff dict after rewriting ``old_id`` to
    ``outer_id * inner_extent + inner_id``."""
    coeffs = ar.coeffs
    result = coeffs
    if old_id in coeffs:
        old_c = coeffs.pop(old_id)
        coeffs[outer_id] = coeffs.get(outer_id, 0) + old_c * inner_extent
        coeffs[inner_id] = coeffs.get(inner_id, 0) + old_c
        result = coeffs
    return result


def _update_iter_var_lists(body: TreeIR, old_id: int, v_outer: IterVar, v_inner: IterVar) -> TreeIR:
    """Update every SBlock.iter_vars: replace entry with var_id==old_id
    by (v_outer, v_inner) in the same list position."""

    def update_block(block: SBlock) -> SBlock:
        new_ivs: list[IterVar] = []
        replaced = False
        for iv in block.iter_vars:
            if iv.var_id == old_id:
                new_ivs.append(v_outer)
                new_ivs.append(v_inner)
                replaced = True
            else:
                new_ivs.append(iv)
        result = block
        if replaced:
            result = SBlock(
                iter_vars=new_ivs,
                reads=block.reads,
                writes=block.writes,
                reads_writes=block.reads_writes,
                body=block.body,
                annotations=dict(block.annotations),
            )
        return result

    def update_node(node: ForNode | SBlock) -> ForNode | SBlock:
        if isinstance(node, SBlock):
            return update_block(node)
        return ForNode(
            iter_var=node.iter_var,
            children=[update_node(c) for c in node.children],
            name=node.name,
            annotations=dict(node.annotations),
        )

    return [update_node(n) for n in body]


def enumerate_split_atoms(module: KernelModule) -> list[Split]:
    """Emit Split(path, factor) for every ForNode divisor factor (1, extent)."""
    atoms: list[Split] = []

    def walk(node: ForNode | SBlock, path: tuple[int, ...]) -> None:
        if isinstance(node, ForNode):
            extent = node.iter_var.extent
            for factor in range(2, extent):
                if extent % factor == 0:
                    atom = Split(loop_path=path, factor=factor)
                    if atom.is_legal(module):
                        atoms.append(atom)
            for i, child in enumerate(node.children):
                walk(child, path + (i,))

    for i, root in enumerate(module.body):
        walk(root, (i,))
    return atoms
